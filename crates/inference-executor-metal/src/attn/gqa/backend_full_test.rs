use std::collections::HashSet;
use std::mem::size_of;

use half::bf16;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::gqa::fp8::bf16_to_f32;
use inference_backend_metal::components::gqa::fp8::bf16_to_fp8_e4m3;
use inference_backend_metal::components::gqa::fp8::f32_to_bf16;
use inference_backend_metal::components::gqa::fp8::fp8_e4m3_to_bf16;
use inference_backend_metal::components::rms_norm_rope::RopeScaling;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_executor_core::attn::GQACore;
use inference_executor_core::attn::gqa::reference::GQAReferenceInput;
use inference_executor_core::attn::gqa::reference::projected_gqa_reference;
use inference_executor_core::mlp::dense::reference::QuantizedAffineReferenceShape;
use inference_executor_core::mlp::dense::reference::quantized_affine_reference;

use super::GQA;
use super::GQA_NUM_ACTIVE_TOKENS;
use super::GQAInput;
use super::GQAKVCacheBindings;
use super::GQAMetalConfig;
use super::GQAReplayTopology;
use super::GQAWeights;
use super::add_gqa_replay_arguments;
use crate::attn::gqa::request_page_table::GQAPageTableLayout;
use crate::def::layer::ReplayLayer;
use crate::replay::Replay;
use crate::replay::ReplayComponent;

const NUM_TOTAL_TOKENS: u32 = 8;
const HIDDEN_DIM: usize = 32;
const HEAD_DIM: usize = 32;
const NUM_Q_HEADS: usize = 1;
const NUM_KV_HEADS: usize = 1;
const GROUP_SIZE: usize = 32;
const BITS: usize = 8;
const NUM_TOKENS_PER_PAGE: usize = 8;
const NORM_EPS: f32 = 1.0e-6;
const ROPE_THETA: f32 = 10_000.0;

#[test]
fn test_replay_matches_cpu_reference_across_active_counts() {
    run_replay_reference(HEAD_DIM, None);
}

#[test]
fn test_direct_output_replay_and_output_mode_cache_identity() {
    run_replay_reference(128, Some(&[true, false, true]));
}

fn run_replay_reference(head_dim: usize, output_modes: Option<&[bool]>) {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let runtime = &stream;
    let core = fixture_core(head_dim);
    let config = fixture_config(head_dim);
    let page_table_layout = GQAPageTableLayout {
        num_req_slots: 1,
        num_blocks: 1,
        num_gqa_layers: 1,
        num_page_ids_per_block: 1,
    };
    let mut component = GQA::new(
        &device,
        core.clone(),
        config,
        NUM_TOTAL_TOKENS as usize,
        page_table_layout,
    );
    if output_modes.is_some() {
        let sdpa_config = super::gqa_sdpa_config(config, core.num_q_heads, core.num_kv_heads, core.head_dim);
        let variant =
            inference_backend_metal::components::gqa::sdpa::ExecutionVariant::sliced_q_per_head(sdpa_config, 8, 16, 1);
        component.sdpa_selector = crate::attn::gqa::sdpa::Selector::for_partitions(
            inference_backend_metal::components::gqa::sdpa::Registry::from_variants(sdpa_config, vec![variant]),
            NUM_TOTAL_TOKENS,
            1,
            |_| 1,
        );
    }
    let scratch = component.new_scratch();
    let policy = component.replay_bucket_policy();
    let metadata = component.new_metadata();
    let hidden_values = generated_bf16_values(NUM_TOTAL_TOKENS as usize * HIDDEN_DIM, 0x8A2E_91D4);
    let hidden = bf16_buffer(&device, &hidden_values);
    let next_hidden = Buffer::new_zeroed_elements(&device, NUM_TOTAL_TOKENS as usize * HIDDEN_DIM, Dtype::Bfloat16);
    let kv_pages = Buffer::new_zeroed(&device, config.page_bytes as usize);
    let page_ids = Buffer::from_slice(&device, &[0_u32]);
    let weights = FixtureWeights::new(&device, &core);
    let mut replay = Replay::new("test GQA", TestGQA(component));
    let mut recorded_keys = HashSet::new();

    for num_active_tokens in [1_u32, 8, 3, 7, 2, 6, 4, 5] {
        for &direct_output in output_modes.unwrap_or(&[false]) {
            let component = &replay.component().0;
            let shape = if output_modes.is_some() {
                let sdpa_config = super::gqa_sdpa_config(config, core.num_q_heads, core.num_kv_heads, core.head_dim);
                let variant = inference_backend_metal::components::gqa::sdpa::ExecutionVariant::sliced_q_per_head(
                    sdpa_config,
                    8,
                    16,
                    1,
                );
                let requests = crate::attn::gqa::sdpa::RequestShape::from_batch(&[0], &[0, num_active_tokens]);
                let selection = component.sdpa_selector.select_partitioned(
                    variant,
                    NUM_TOTAL_TOKENS,
                    direct_output,
                    &requests,
                    &policy,
                    NUM_TOTAL_TOKENS,
                );
                metadata.update(&[0], &[0], &[0, num_active_tokens], &selection)
            } else {
                component.prepare(
                    &metadata,
                    &[0],
                    &[0],
                    &[0, num_active_tokens],
                    &policy,
                    NUM_TOTAL_TOKENS,
                )
            };
            let topology = replay.component().0.replay_topology(&metadata);
            let input = GQAInput {
                page_table_layout,
                gqa_layer_index: ReplayU32::Fixed(0),
                batch_metadata: &metadata,
                hidden_state: &hidden,
                next_hidden_state: &next_hidden,
                kv_cache: GQAKVCacheBindings {
                    kv_pages: &kv_pages,
                    page_ids: &page_ids,
                },
                weights: weights.bindings(),
                scratch: scratch.bindings(),
                num_active_tokens: ReplayU32::Parameter(GQA_NUM_ACTIVE_TOKENS),
            };
            let (key, cache_hit) = replay.record(runtime, &input);
            let seen = !recorded_keys.insert(key);
            assert_eq!(cache_hit, seen);

            kv_pages.zero_bytes(0, config.page_bytes as usize);
            next_hidden.zero_bytes(0, next_hidden.len_bytes());
            let mut arguments = ReplayArguments::new();
            add_gqa_replay_arguments(shape, topology, &mut arguments);
            runtime
                .submit_replay_with_arguments(replay.replay(&key), &arguments)
                .wait();

            let reference = gqa_reference(&core, config, num_active_tokens as usize, &hidden_values, &weights);
            let actual_hidden = read_bf16_values(&next_hidden, reference.output.len());
            assert_close(&actual_hidden, &reference.output, 0.03125);
            let actual_pages = kv_pages
                .read_typed::<u8>(0, config.page_bytes as usize)
                .into_iter()
                .map(|bits| bf16_to_f32(fp8_e4m3_to_bf16(bits)))
                .collect::<Vec<_>>();
            assert_close(&actual_pages, &reference.kv_pages, 0.0);
        }
    }
    if output_modes.is_some() {
        assert_eq!(
            recorded_keys.len(),
            2,
            "direct and Map/Reduce executions need distinct replay keys with equal padded extents"
        );
    }
}

struct TestGQA(GQA);

impl ReplayComponent for TestGQA {
    type Key = (u32, u32, u32, GQAReplayTopology);
    type Input<'a> = GQAInput<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        let shape = input.batch_metadata.replay_shape();
        (
            shape.num_total_tokens,
            shape.num_total_q_token_tiles,
            shape.num_total_sdpa_map_task_templates,
            self.0.replay_topology(input.batch_metadata),
        )
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        <GQA as ReplayLayer>::record(&self.0, recorder, *input);
    }
}

struct FixtureWeights {
    qgkv_weight: Buffer,
    qgkv_scales: Buffer,
    qgkv_biases: Buffer,
    q_norm_weight: Buffer,
    k_norm_weight: Buffer,
    output_weight: Buffer,
    output_scales: Buffer,
    output_biases: Buffer,
    qgkv_weight_values: Vec<u8>,
    qgkv_scale_values: Vec<f32>,
    qgkv_bias_values: Vec<f32>,
    q_norm_values: Vec<f32>,
    k_norm_values: Vec<f32>,
    output_weight_values: Vec<u8>,
    output_scale_values: Vec<f32>,
    output_bias_values: Vec<f32>,
}

impl FixtureWeights {
    fn new(device: &Device, core: &GQACore) -> Self {
        let qgkv_shape = affine_reference_shape(1, core.qgkv_dim(), core.hidden_dim);
        let output_shape = affine_reference_shape(1, core.hidden_dim, core.q_dim());
        let (qgkv_weight_values, qgkv_scale_values, qgkv_bias_values) = affine_values(qgkv_shape, 0xC81A_1137);
        let (output_weight_values, output_scale_values, output_bias_values) = affine_values(output_shape, 0xC81A_1138);
        let q_norm_values = (0..core.head_dim)
            .map(|index| bf16::from_f32(0.75 + (index % 7) as f32 * 0.03125).to_f32())
            .collect::<Vec<_>>();
        let k_norm_values = (0..core.head_dim)
            .map(|index| bf16::from_f32(0.875 + (index % 5) as f32 * 0.025).to_f32())
            .collect::<Vec<_>>();
        Self {
            qgkv_weight: Buffer::from_slice(device, &qgkv_weight_values),
            qgkv_scales: bf16_buffer(device, &qgkv_scale_values),
            qgkv_biases: bf16_buffer(device, &qgkv_bias_values),
            q_norm_weight: bf16_buffer(device, &q_norm_values),
            k_norm_weight: bf16_buffer(device, &k_norm_values),
            output_weight: Buffer::from_slice(device, &output_weight_values),
            output_scales: bf16_buffer(device, &output_scale_values),
            output_biases: bf16_buffer(device, &output_bias_values),
            qgkv_weight_values,
            qgkv_scale_values,
            qgkv_bias_values,
            q_norm_values,
            k_norm_values,
            output_weight_values,
            output_scale_values,
            output_bias_values,
        }
    }

    fn bindings(&self) -> GQAWeights<'_> {
        GQAWeights {
            qgkv_weight: &self.qgkv_weight,
            qgkv_scales: &self.qgkv_scales,
            qgkv_biases: &self.qgkv_biases,
            q_norm_weight: &self.q_norm_weight,
            k_norm_weight: &self.k_norm_weight,
            output_weight: &self.output_weight,
            output_scales: &self.output_scales,
            output_biases: &self.output_biases,
        }
    }
}

struct GQAReference {
    output: Vec<f32>,
    kv_pages: Vec<f32>,
}

fn gqa_reference(
    core: &GQACore,
    config: GQAMetalConfig,
    num_tokens: usize,
    hidden: &[f32],
    weights: &FixtureWeights,
) -> GQAReference {
    let hidden = &hidden[..num_tokens * core.hidden_dim];
    let qgkv = bf16_round_trip(&quantized_affine_reference(
        affine_reference_shape(num_tokens, core.qgkv_dim(), core.hidden_dim),
        hidden,
        &weights.qgkv_weight_values,
        &weights.qgkv_scale_values,
        &weights.qgkv_bias_values,
    ));
    let mut q = Vec::with_capacity(num_tokens * core.q_dim());
    let mut g = Vec::with_capacity(num_tokens * core.g_dim());
    let mut k = Vec::with_capacity(num_tokens * core.k_dim());
    let mut v = Vec::with_capacity(num_tokens * core.v_dim());
    for q_token_head_values in qgkv.chunks_exact(core.qgkv_dim()) {
        let q_end = core.q_dim();
        let g_end = q_end + core.g_dim();
        let k_end = g_end + core.k_dim();
        q.extend_from_slice(&q_token_head_values[..q_end]);
        g.extend_from_slice(&q_token_head_values[q_end..g_end]);
        k.extend_from_slice(&q_token_head_values[g_end..k_end]);
        v.extend_from_slice(&q_token_head_values[k_end..]);
    }
    let q = rms_norm_rope_reference(&q, num_tokens, core.num_q_heads, &weights.q_norm_values, config);
    let k = rms_norm_rope_reference(&k, num_tokens, core.num_kv_heads, &weights.k_norm_values, config);
    let quantized_k = fp8_e4m3_round_trip(&k);
    let quantized_v = fp8_e4m3_round_trip(&v);
    let attention = bf16_round_trip(&projected_gqa_reference(
        core,
        GQAReferenceInput {
            cu_tokens: &[0, num_tokens as u32],
            token_indices: &[0],
            q: &q,
            context_k_by_req: &[&quantized_k],
            context_v_by_req: &[&quantized_v],
        },
    ));
    let gated = attention
        .iter()
        .zip(&g)
        .map(|(&attention, &gate)| bf16_round_trip_value(attention * sigmoid_bf16(gate)))
        .collect::<Vec<_>>();
    let output = bf16_round_trip(&quantized_affine_reference(
        affine_reference_shape(num_tokens, core.hidden_dim, core.q_dim()),
        &gated,
        &weights.output_weight_values,
        &weights.output_scale_values,
        &weights.output_bias_values,
    ));
    GQAReference {
        output,
        kv_pages: kv_page_reference(core, &k, &v, config),
    }
}

fn rms_norm_rope_reference(
    input: &[f32],
    num_tokens: usize,
    num_heads: usize,
    norm_weight: &[f32],
    config: GQAMetalConfig,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    let rope_half = config.rope_dim as usize / 2;
    let head_dim = norm_weight.len();
    for token_index in 0..num_tokens {
        for head_index in 0..num_heads {
            let q_token_head_begin = (token_index * num_heads + head_index) * head_dim;
            let q_token_head_values = &input[q_token_head_begin..q_token_head_begin + head_dim];
            let inv_rms = (q_token_head_values.iter().map(|value| value * value).sum::<f32>() / head_dim as f32
                + config.norm_eps)
                .sqrt()
                .recip();
            let mut normalized = q_token_head_values
                .iter()
                .zip(norm_weight)
                .map(|(&value, &weight)| {
                    let scaled = bf16_round_trip_value(value * inv_rms);
                    bf16_round_trip_value(weight * scaled)
                })
                .collect::<Vec<_>>();
            for dim in 0..rope_half {
                let theta = token_index as f32 / config.rope_theta.powf(2.0 * dim as f32 / config.rope_dim as f32);
                let (sin, cos) = theta.sin_cos();
                let x1 = normalized[dim];
                let x2 = normalized[dim + rope_half];
                normalized[dim] = bf16_round_trip_value(x1 * cos - x2 * sin);
                normalized[dim + rope_half] = bf16_round_trip_value(x1 * sin + x2 * cos);
            }
            output[q_token_head_begin..q_token_head_begin + head_dim].copy_from_slice(&normalized);
        }
    }
    output
}

fn kv_page_reference(core: &GQACore, k: &[f32], v: &[f32], config: GQAMetalConfig) -> Vec<f32> {
    let mut pages = vec![0.0; config.page_bytes as usize];
    let head_dim = core.head_dim;
    let num_tokens = k.len() / core.k_dim();
    for token_index in 0..num_tokens {
        for head_index in 0..core.num_kv_heads {
            for dim in 0..head_dim {
                let source = (token_index * core.num_kv_heads + head_index) * head_dim + dim;
                let k_target = (head_index * NUM_TOKENS_PER_PAGE + token_index) * head_dim + dim;
                let v_target = ((core.num_kv_heads + head_index) * NUM_TOKENS_PER_PAGE + token_index) * head_dim + dim;
                pages[k_target] = bf16_to_f32(fp8_e4m3_to_bf16(bf16_to_fp8_e4m3(f32_to_bf16(k[source]))));
                pages[v_target] = bf16_to_f32(fp8_e4m3_to_bf16(bf16_to_fp8_e4m3(f32_to_bf16(v[source]))));
            }
        }
    }
    pages
}

fn sigmoid_bf16(value: f32) -> f32 {
    let absolute = bf16_round_trip_value(value.abs());
    let exponential = bf16_round_trip_value((-absolute).exp());
    let denominator = bf16_round_trip_value(1.0 + exponential);
    let positive = bf16_round_trip_value(1.0 / denominator);
    if value < 0.0 {
        bf16_round_trip_value(1.0 - positive)
    } else {
        positive
    }
}

fn fixture_core(head_dim: usize) -> GQACore {
    GQACore::new(
        0,
        HIDDEN_DIM,
        head_dim,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        (head_dim as f32).sqrt().recip(),
    )
}

fn fixture_config(head_dim: usize) -> GQAMetalConfig {
    GQAMetalConfig {
        group_size: GROUP_SIZE as u32,
        bits: BITS as u32,
        page_bytes: (2 * NUM_KV_HEADS * NUM_TOKENS_PER_PAGE * head_dim * size_of::<u8>()) as u32,
        rope_dim: head_dim as u32,
        norm_eps: NORM_EPS,
        rope_theta: ROPE_THETA,
        rope_scaling: RopeScaling::Default,
        io_dtype: Dtype::Bfloat16,
    }
}

fn affine_reference_shape(num_tokens: usize, output_dim: usize, input_dim: usize) -> QuantizedAffineReferenceShape {
    QuantizedAffineReferenceShape {
        num_rows: num_tokens,
        output_dim,
        input_dim,
        group_size: GROUP_SIZE,
        bits: BITS,
    }
}

fn affine_values(shape: QuantizedAffineReferenceShape, seed: u32) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let weight = (0..shape.weight_bytes())
        .map(|index| ((index * 29 + seed as usize) % 251) as u8)
        .collect::<Vec<_>>();
    let scales = (0..shape.affine_param_len())
        .map(|index| bf16::from_f32(0.000_75 + (index % 5) as f32 * 0.000_05).to_f32())
        .collect::<Vec<_>>();
    let biases = (0..shape.affine_param_len())
        .map(|index| bf16::from_f32(-0.105 + (index % 7) as f32 * 0.001).to_f32())
        .collect::<Vec<_>>();
    (weight, scales, biases)
}

fn generated_bf16_values(count: usize, mut state: u32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bf16_round_trip_value(((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0)
        })
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

fn bf16_round_trip(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&value| bf16_round_trip_value(value)).collect()
}

fn bf16_round_trip_value(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(value))
}

fn fp8_e4m3_round_trip(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&value| bf16_to_f32(fp8_e4m3_to_bf16(bf16_to_fp8_e4m3(f32_to_bf16(value)))))
        .collect()
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
    let mean_abs_tolerance = tolerance * 0.25;
    assert!(
        max_abs_error <= tolerance && mean_abs_error <= mean_abs_tolerance,
        "GQA quality mismatch: max_abs_error={max_abs_error} max_abs_tolerance={tolerance} \
         mean_abs_error={mean_abs_error} mean_abs_tolerance={mean_abs_tolerance}"
    );
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "GQA mismatch at {index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}
