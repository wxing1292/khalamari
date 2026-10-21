use std::collections::HashSet;

use half::bf16;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::gdn::GDNOutputGate;
use inference_executor_core::attn::gdn::reference::GDNRecurrentReferenceInput;
use inference_executor_core::attn::gdn::reference::gdn_output_norm_gate_reference;
use inference_executor_core::attn::gdn::reference::gdn_recurrent_reference;
use inference_executor_core::attn::gdn::reference::gdn_short_conv_reference;
use inference_executor_core::mlp::dense::reference::QuantizedAffineReferenceShape;
use inference_executor_core::mlp::dense::reference::quantized_affine_reference;
use inference_executor_core::replay::ReplayBucketPolicy;

use super::GDN;
use super::GDN_NUM_ACTIVE_TOKENS;
use super::GDNInput;
use super::GDNLayerStateBindings;
use super::GDNMetalConfig;
use super::GDNReplayTopology;
use super::GDNWeights;
use super::add_gdn_replay_arguments;
use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;
use crate::attn::gdn::state_table::GDNPreparedRequestState;
use crate::def::layer::ReplayLayer;
use crate::replay::Replay;
use crate::replay::ReplayComponent;

const NUM_TOTAL_REQUESTS: u32 = 8;
const NUM_TOTAL_TOKENS: u32 = 32;
const NUM_SOURCE_STATE_SLOTS: usize = 8;
const NUM_STATE_SLOTS: usize = NUM_SOURCE_STATE_SLOTS + NUM_TOTAL_TOKENS as usize + 1;
const HIDDEN_DIM: usize = 32;
const QK_HEAD_DIM: usize = 32;
const V_HEAD_DIM: usize = 32;
const CONV_KERNEL_SIZE: usize = 3;
const GROUP_SIZE: usize = 32;
const BITS: usize = 8;
const NORM_EPS: f32 = 1.0e-6;
const STATE_CANARY: f32 = -0.75;

#[test]
fn test_replay_matches_cpu_reference_across_independent_active_domains() {
    assert_replay_matches_cpu_reference(GDNOutputGate::Silu);
}

#[test]
fn test_sigmoid_gate_replay_matches_cpu_reference_across_independent_active_domains() {
    assert_replay_matches_cpu_reference(GDNOutputGate::Sigmoid);
}

fn assert_replay_matches_cpu_reference(output_gate: GDNOutputGate) {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let runtime = &stream;
    let core = GDNCore {
        output_gate,
        ..fixture_core()
    };
    let config = fixture_config();
    let component = GDN::new(&device, core.clone(), config);
    let scratch = component.new_scratch(NUM_TOTAL_TOKENS as usize);
    let metadata = GDNMetadataBuffers::new(&device, NUM_TOTAL_REQUESTS as usize, NUM_TOTAL_TOKENS as usize);
    let policy = ReplayBucketPolicy::new(NUM_TOTAL_REQUESTS);
    let hidden_values = generated_bf16_values(NUM_TOTAL_TOKENS as usize * HIDDEN_DIM, 0x792A_5F13, 0.5);
    let hidden = bf16_buffer(&device, &hidden_values);
    let next_hidden = Buffer::new_zeroed_elements(&device, NUM_TOTAL_TOKENS as usize * HIDDEN_DIM, Dtype::Bfloat16);
    let weights = FixtureWeights::new(&device, &core);
    let conv_stride = core.qkv_dim() * core.conv_state_len();
    let recurrent_stride = core.num_v_heads * core.v_head_dim * core.qk_head_dim;
    let source_conv = generated_values(NUM_SOURCE_STATE_SLOTS * conv_stride, 0x792A_5F14, 0.03125);
    let source_recurrent = generated_values(NUM_SOURCE_STATE_SLOTS * recurrent_stride, 0x792A_5F15, 0.0078125);
    let initial_conv_arena = state_arena(&source_conv, conv_stride);
    let initial_recurrent_arena = state_arena(&source_recurrent, recurrent_stride);
    let conv_state_arena = bf16_buffer(&device, &initial_conv_arena);
    let recurrent_state_arena = bf16_buffer(&device, &initial_recurrent_arena);
    let replay_alpha =
        Buffer::new_zeroed_elements(&device, NUM_TOTAL_TOKENS as usize * core.num_v_heads, Dtype::Float32);
    let replay_k = Buffer::new_zeroed_elements(
        &device,
        NUM_TOTAL_TOKENS as usize * core.num_qk_heads * core.qk_head_dim,
        Dtype::Float32,
    );
    let replay_u = Buffer::new_zeroed_elements(
        &device,
        NUM_TOTAL_TOKENS as usize * core.num_v_heads * core.v_head_dim,
        Dtype::Float32,
    );
    let replay_qkv = Buffer::new_zeroed_elements(&device, NUM_TOTAL_TOKENS as usize * core.qkv_dim(), Dtype::Bfloat16);
    let mut replay = Replay::new("test GDN", TestGDN(component));
    let mut recorded_keys = HashSet::new();
    let cases: &[(&[u32], u32)] = &[
        (&[1], 0),
        (&[2, 2, 2, 2], 4),
        (&[3, 2, 2], 1),
        (&[2, 1], 0),
        (&[1, 1, 1, 1], 0),
        (&[2, 2, 2, 2], 0),
        (&[2, 2, 2, 2], 3),
        (&[2, 2, 2, 2], 1),
        (&[7, 8, 9], 3),
        (&[9, 7, 1, 4], 2),
        (&[9, 7, 1, 4], 0),
        (&[9, 7, 1, 4], 4),
        (&[9, 7, 1, 4], 1),
    ];

    for materialize_candidate_states in [false, true] {
        for &(counts, num_active_recurrent_requests) in cases {
            let cu_tokens = cumulative_tokens(counts);
            let num_active_tokens = *cu_tokens.last().unwrap();
            let num_active_requests = counts.len() as u32;
            let source_slots = (0..num_active_requests).collect::<Vec<_>>();
            let mut recurrent_write_slots = vec![u32::MAX; num_active_tokens as usize];
            let mut conv_write_slots = recurrent_write_slots.clone();
            for window in cu_tokens.windows(2).take(num_active_recurrent_requests as usize) {
                for token_index in window[0]..window[1] {
                    let is_final = token_index + 1 == window[1];
                    if is_final || (materialize_candidate_states && token_index % 3 != 1) {
                        recurrent_write_slots[token_index as usize] = NUM_SOURCE_STATE_SLOTS as u32 + token_index;
                        conv_write_slots[token_index as usize] =
                            NUM_SOURCE_STATE_SLOTS as u32 + NUM_TOTAL_TOKENS - 1 - token_index;
                    }
                }
            }
            let prepared = GDNPreparedRequestState {
                src_recurrent_state_slots: source_slots.clone(),
                src_conv_state_slots: source_slots,
                flat_recurrent_state_write_slots: recurrent_write_slots,
                flat_conv_state_write_slots: conv_write_slots,
            };
            let num_total_tokens = num_active_tokens.max(8);
            let shape = replay.component().0.prepare(
                &metadata,
                &cu_tokens,
                num_active_recurrent_requests,
                &prepared,
                &policy,
                num_total_tokens,
            );
            let input = GDNInput {
                hidden_state: &hidden,
                next_hidden_state: &next_hidden,
                scratch: scratch.bindings(),
                batch_metadata: &metadata,
                state: GDNLayerStateBindings {
                    conv_state: &conv_state_arena,
                    conv_state_offset_bytes: 0,
                    next_conv_state: &conv_state_arena,
                    next_conv_state_offset_bytes: 0,
                    recurrent_state_arena: &recurrent_state_arena,
                    recurrent_state_arena_offset_bytes: 0,
                    replay: inference_backend_metal::components::gdn::compute::ReplayBuffers {
                        alpha: &replay_alpha,
                        k: &replay_k,
                        u: &replay_u,
                        qkv: &replay_qkv,
                        token_offset: 0,
                        num_total_tokens: NUM_TOTAL_TOKENS,
                    },
                },
                weights: weights.bindings(),
                num_active_tokens: ReplayU32::Parameter(GDN_NUM_ACTIVE_TOKENS),
            };
            let (key, cache_hit) = replay.record(runtime, &input);
            let seen = !recorded_keys.insert(key);
            assert_eq!(cache_hit, seen);

            let previous_conv = read_bf16_values(&conv_state_arena, initial_conv_arena.len());
            let previous_recurrent = read_bf16_values(&recurrent_state_arena, initial_recurrent_arena.len());
            let previous_output = read_bf16_values(&next_hidden, hidden_values.len());
            let mut arguments = ReplayArguments::new();
            add_gdn_replay_arguments(shape, num_active_recurrent_requests, &mut arguments);
            runtime
                .submit_replay_with_arguments(replay.replay(&key), &arguments)
                .wait();

            let reference = gdn_reference(
                &core,
                config,
                &cu_tokens,
                &prepared,
                &hidden_values,
                &source_conv,
                &source_recurrent,
                &previous_conv,
                &previous_recurrent,
                &weights,
            );
            let actual_output = read_bf16_values(&next_hidden, hidden_values.len());
            assert_close(&actual_output[..reference.output.len()], &reference.output, 0.0625);
            assert_eq!(
                &actual_output[reference.output.len()..],
                &previous_output[reference.output.len()..]
            );
            let actual_conv = read_bf16_values(&conv_state_arena, initial_conv_arena.len());
            let actual_recurrent = read_bf16_values(&recurrent_state_arena, initial_recurrent_arena.len());
            assert_close(&actual_conv, &reference.conv_state_arena, 0.0025);
            assert_close(&actual_recurrent, &reference.recurrent_state_arena, 0.0025);
            assert_unchanged_slots(
                &actual_conv,
                &previous_conv,
                &prepared.flat_conv_state_write_slots,
                conv_stride,
            );
            assert_unchanged_slots(
                &actual_recurrent,
                &previous_recurrent,
                &prepared.flat_recurrent_state_write_slots,
                recurrent_stride,
            );
        }
    }
}

struct TestGDN(GDN);

impl ReplayComponent for TestGDN {
    type Key = (u32, u32, GDNReplayTopology);
    type Input<'a> = GDNInput<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        let shape = input.batch_metadata.replay_shape();
        (
            shape.num_total_reqs,
            shape.num_total_tokens,
            self.0.replay_topology(input.batch_metadata),
        )
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        <GDN as ReplayLayer>::record(&self.0, recorder, *input);
    }
}

struct FixtureWeights {
    qkvabz_weight: Buffer,
    qkvabz_scales: Buffer,
    qkvabz_biases: Buffer,
    conv_weight: Buffer,
    norm_weight: Buffer,
    a_log: Buffer,
    dt_bias: Buffer,
    output_weight: Buffer,
    output_scales: Buffer,
    output_biases: Buffer,
    qkvabz_weight_values: Vec<u8>,
    qkvabz_scale_values: Vec<f32>,
    qkvabz_bias_values: Vec<f32>,
    conv_weight_values: Vec<f32>,
    norm_weight_values: Vec<f32>,
    a_log_values: Vec<f32>,
    dt_bias_values: Vec<f32>,
    output_weight_values: Vec<u8>,
    output_scale_values: Vec<f32>,
    output_bias_values: Vec<f32>,
}

impl FixtureWeights {
    fn new(device: &Device, core: &GDNCore) -> Self {
        let qkvabz_shape = affine_reference_shape(1, core.qkvabz_dim(), core.hidden_dim);
        let output_shape = affine_reference_shape(1, core.hidden_dim, core.v_dim());
        let (qkvabz_weight_values, qkvabz_scale_values, qkvabz_bias_values) = affine_values(qkvabz_shape, 0xA812_40B1);
        let (output_weight_values, output_scale_values, output_bias_values) = affine_values(output_shape, 0xA812_40B2);
        let conv_weight_values = generated_bf16_values(core.qkv_dim() * core.conv_kernel_size, 0xA812_40B3, 0.0625);
        let norm_weight_values = (0..core.v_head_dim)
            .map(|index| bf16::from_f32(0.75 + (index % 7) as f32 * 0.03125).to_f32())
            .collect::<Vec<_>>();
        let a_log_values = vec![bf16::from_f32(-0.5).to_f32(); core.num_v_heads];
        let dt_bias_values = vec![bf16::from_f32(0.125).to_f32(); core.num_v_heads];
        Self {
            qkvabz_weight: Buffer::from_slice(device, &qkvabz_weight_values),
            qkvabz_scales: Buffer::from_slice(device, &qkvabz_scale_values),
            qkvabz_biases: Buffer::from_slice(device, &qkvabz_bias_values),
            conv_weight: bf16_buffer(device, &conv_weight_values),
            norm_weight: bf16_buffer(device, &norm_weight_values),
            a_log: bf16_buffer(device, &a_log_values),
            dt_bias: bf16_buffer(device, &dt_bias_values),
            output_weight: Buffer::from_slice(device, &output_weight_values),
            output_scales: Buffer::from_slice(device, &output_scale_values),
            output_biases: Buffer::from_slice(device, &output_bias_values),
            qkvabz_weight_values,
            qkvabz_scale_values,
            qkvabz_bias_values,
            conv_weight_values,
            norm_weight_values,
            a_log_values,
            dt_bias_values,
            output_weight_values,
            output_scale_values,
            output_bias_values,
        }
    }

    fn bindings(&self) -> GDNWeights<'_> {
        GDNWeights {
            qkvabz_weight: &self.qkvabz_weight,
            qkvabz_scales: &self.qkvabz_scales,
            qkvabz_biases: &self.qkvabz_biases,
            conv_weight: &self.conv_weight,
            norm_weight: &self.norm_weight,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            output_weight: &self.output_weight,
            output_scales: &self.output_scales,
            output_biases: &self.output_biases,
        }
    }
}

struct GDNReference {
    output: Vec<f32>,
    conv_state_arena: Vec<f32>,
    recurrent_state_arena: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn gdn_reference(
    core: &GDNCore,
    config: GDNMetalConfig,
    cu_tokens: &[u32],
    prepared: &GDNPreparedRequestState,
    hidden: &[f32],
    source_conv: &[f32],
    source_recurrent: &[f32],
    initial_conv_arena: &[f32],
    initial_recurrent_arena: &[f32],
    weights: &FixtureWeights,
) -> GDNReference {
    let num_requests = cu_tokens.len() - 1;
    let num_tokens = *cu_tokens.last().unwrap() as usize;
    let qkvabz = bf16_round_trip(&quantized_affine_reference(
        affine_reference_shape(num_tokens, core.qkvabz_dim(), core.hidden_dim),
        &hidden[..num_tokens * core.hidden_dim],
        &weights.qkvabz_weight_values,
        &weights.qkvabz_scale_values,
        &weights.qkvabz_bias_values,
    ));
    let mut qkv = Vec::with_capacity(num_tokens * core.qkv_dim());
    let mut a = Vec::with_capacity(num_tokens * core.num_v_heads);
    let mut b = Vec::with_capacity(num_tokens * core.num_v_heads);
    let mut z = Vec::with_capacity(num_tokens * core.v_dim());
    for row in qkvabz.chunks_exact(core.qkvabz_dim()) {
        let qkv_end = core.qkv_dim();
        let a_end = qkv_end + core.num_v_heads;
        let b_end = a_end + core.num_v_heads;
        qkv.extend_from_slice(&row[..qkv_end]);
        a.extend_from_slice(&row[qkv_end..a_end]);
        b.extend_from_slice(&row[a_end..b_end]);
        z.extend_from_slice(&row[b_end..]);
    }
    let conv_stride = core.qkv_dim() * core.conv_state_len();
    let recurrent_stride = core.num_v_heads * core.v_head_dim * core.qk_head_dim;
    let quantized_source_conv = bf16_round_trip(source_conv);
    let quantized_source_recurrent = bf16_round_trip(source_recurrent);
    let quantized_conv_weight = bf16_round_trip(&weights.conv_weight_values);
    let quantized_a_log = bf16_round_trip(&weights.a_log_values);
    let quantized_dt_bias = bf16_round_trip(&weights.dt_bias_values);
    let conv = gdn_short_conv_reference(
        core,
        cu_tokens,
        &quantized_source_conv[..num_requests * conv_stride],
        &qkv,
        &quantized_conv_weight,
    );
    let quantized_conv_qkv = bf16_round_trip(&conv.conv_qkv);
    let recurrent = gdn_recurrent_reference(
        core,
        GDNRecurrentReferenceInput {
            cu_tokens,
            source_recurrent_state: &quantized_source_recurrent[..num_requests * recurrent_stride],
            conv_qkv: &quantized_conv_qkv,
            a: &a,
            b: &b,
            a_log: &quantized_a_log,
            dt_bias: &quantized_dt_bias,
        },
    );
    let quantized_recurrent_output = bf16_round_trip(&recurrent.recurrent_output);
    let norm_gated = gdn_output_norm_gate_reference(
        core,
        &quantized_recurrent_output,
        &z,
        &bf16_round_trip(&weights.norm_weight_values),
        config.norm_eps,
    );
    let output = bf16_round_trip(&quantized_affine_reference(
        affine_reference_shape(num_tokens, core.hidden_dim, core.v_dim()),
        &bf16_round_trip(&norm_gated),
        &weights.output_weight_values,
        &weights.output_scale_values,
        &weights.output_bias_values,
    ));
    let mut conv_state_arena = bf16_round_trip(initial_conv_arena);
    let mut recurrent_state_arena = bf16_round_trip(initial_recurrent_arena);
    for (request_index, window) in cu_tokens.windows(2).enumerate() {
        let request_start = window[0] as usize;
        for token_index in request_start..window[1] as usize {
            let conv_slot = prepared.flat_conv_state_write_slots[token_index];
            let recurrent_slot = prepared.flat_recurrent_state_write_slots[token_index];
            if conv_slot == u32::MAX && recurrent_slot == u32::MAX {
                continue;
            }
            let prefix_tokens = [0, (token_index + 1 - request_start) as u32];
            let prefix_conv = gdn_short_conv_reference(
                core,
                &prefix_tokens,
                &quantized_source_conv[request_index * conv_stride..(request_index + 1) * conv_stride],
                &qkv[request_start * core.qkv_dim()..(token_index + 1) * core.qkv_dim()],
                &quantized_conv_weight,
            );
            let prefix_recurrent = gdn_recurrent_reference(
                core,
                GDNRecurrentReferenceInput {
                    cu_tokens: &prefix_tokens,
                    source_recurrent_state: &quantized_source_recurrent
                        [request_index * recurrent_stride..(request_index + 1) * recurrent_stride],
                    conv_qkv: &bf16_round_trip(&prefix_conv.conv_qkv),
                    a: &a[request_start * core.num_v_heads..(token_index + 1) * core.num_v_heads],
                    b: &b[request_start * core.num_v_heads..(token_index + 1) * core.num_v_heads],
                    a_log: &quantized_a_log,
                    dt_bias: &quantized_dt_bias,
                },
            );
            if conv_slot != u32::MAX {
                let start = conv_slot as usize * conv_stride;
                conv_state_arena[start..start + conv_stride]
                    .copy_from_slice(&bf16_round_trip(&prefix_conv.next_conv_state));
            }
            if recurrent_slot != u32::MAX {
                let start = recurrent_slot as usize * recurrent_stride;
                recurrent_state_arena[start..start + recurrent_stride]
                    .copy_from_slice(&bf16_round_trip(&prefix_recurrent.next_recurrent_state));
            }
        }
    }
    GDNReference {
        output,
        conv_state_arena,
        recurrent_state_arena,
    }
}

fn fixture_core() -> GDNCore {
    GDNCore {
        model_layer_index: 0,
        hidden_dim: HIDDEN_DIM,
        num_qk_heads: 1,
        qk_head_dim: QK_HEAD_DIM,
        num_v_heads: 1,
        v_head_dim: V_HEAD_DIM,
        conv_kernel_size: CONV_KERNEL_SIZE,
        q_scale: (QK_HEAD_DIM as f32).sqrt().recip(),
        output_gate: GDNOutputGate::Silu,
    }
}

fn fixture_config() -> GDNMetalConfig {
    GDNMetalConfig {
        group_size: GROUP_SIZE as u32,
        bits: BITS as u32,
        norm_eps: NORM_EPS,
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        qkvabz_scale_bias_dtype: Dtype::Float32,
        output_scale_bias_dtype: Dtype::Float32,
    }
}

fn cumulative_tokens(counts: &[u32]) -> Vec<u32> {
    let mut cumulative = Vec::with_capacity(counts.len() + 1);
    cumulative.push(0);
    for &count in counts {
        cumulative.push(cumulative.last().copied().unwrap() + count);
    }
    cumulative
}

fn state_arena(source: &[f32], stride: usize) -> Vec<f32> {
    let mut arena = vec![STATE_CANARY; NUM_STATE_SLOTS * stride];
    arena[..source.len()].copy_from_slice(source);
    arena
}

fn affine_reference_shape(num_rows: usize, output_dim: usize, input_dim: usize) -> QuantizedAffineReferenceShape {
    QuantizedAffineReferenceShape {
        num_rows,
        output_dim,
        input_dim,
        group_size: GROUP_SIZE,
        bits: BITS,
    }
}

fn affine_values(shape: QuantizedAffineReferenceShape, seed: u32) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let weight = (0..shape.weight_bytes())
        .map(|index| ((index * 31 + seed as usize) % 251) as u8)
        .collect::<Vec<_>>();
    let scales = (0..shape.affine_param_len())
        .map(|index| 0.000_75 + (index % 5) as f32 * 0.000_05)
        .collect::<Vec<_>>();
    let biases = (0..shape.affine_param_len())
        .map(|index| -0.105 + (index % 7) as f32 * 0.001)
        .collect::<Vec<_>>();
    (weight, scales, biases)
}

fn generated_values(count: usize, mut state: u32, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn generated_bf16_values(count: usize, state: u32, scale: f32) -> Vec<f32> {
    generated_values(count, state, scale)
        .into_iter()
        .map(|value| bf16::from_f32(value).to_f32())
        .collect()
}

fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    Buffer::from_slice(
        device,
        &values
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>(),
    )
}

fn read_bf16_values(buffer: &Buffer, len: usize) -> Vec<f32> {
    buffer
        .read_typed::<u16>(0, len)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect()
}

fn f32_to_bf16(value: f32) -> bf16 {
    bf16::from_f32(value)
}

fn bf16_to_f32(value: bf16) -> f32 {
    value.to_f32()
}

fn bf16_round_trip(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&value| bf16_to_f32(f32_to_bf16(value))).collect()
}

fn assert_unchanged_slots(actual: &[f32], previous: &[f32], write_slots: &[u32], stride: usize) {
    for slot in 0..NUM_STATE_SLOTS {
        if !write_slots.contains(&(slot as u32)) {
            assert_eq!(
                &actual[slot * stride..(slot + 1) * stride],
                &previous[slot * stride..(slot + 1) * stride],
                "inactive state slot {slot} changed",
            );
        }
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    let mean_abs_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f32>()
        / actual.len().max(1) as f32;
    let max_abs_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let mean_abs_tolerance = tolerance * 0.1;
    assert!(
        max_abs_error <= tolerance && mean_abs_error <= mean_abs_tolerance,
        "GDN quality mismatch: max_abs_error={max_abs_error} max_abs_tolerance={tolerance} \
         mean_abs_error={mean_abs_error} mean_abs_tolerance={mean_abs_tolerance}"
    );
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual - expected).abs();
        assert!(
            difference <= tolerance,
            "GDN mismatch at {index}: actual={actual} expected={expected} difference={difference} \
             tolerance={tolerance}"
        );
    }
}
