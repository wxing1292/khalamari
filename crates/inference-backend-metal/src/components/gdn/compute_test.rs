use std::mem::size_of;

use half::bf16;
use inference_executor_core::attn::gdn::GDNCore;
use inference_executor_core::attn::gdn::GDNOutputGate;
use inference_executor_core::attn::gdn::reference::GDNRecurrentReferenceInput;
use inference_executor_core::attn::gdn::reference::gdn_output_norm_gate_reference;
use inference_executor_core::attn::gdn::reference::gdn_recurrent_reference;
use inference_executor_core::attn::gdn::reference::gdn_short_conv_reference;

use super::Buffers;
use super::Compute;
use super::ComputeConstants;
use super::Config;
use super::MixedInvocation;
use super::OutputGate;
use super::Shape;
use super::recurrent;
use super::recurrent_snapshot;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayU32;
use crate::metal::Stream;

const NUM_ACTIVE_REQUESTS: ReplayParameterKey = ReplayParameterKey::new("test.gdn_compute.num_active_requests");
const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.gdn_compute.num_active_tokens");
const NUM_ACTIVE_RECURRENT_REQUESTS: ReplayParameterKey =
    ReplayParameterKey::new("test.gdn_compute.num_active_recurrent_requests");

#[test]
fn test_execution_segments() {
    for (num_tokens, num_spec_tokens, recurrent) in [(1, 0, false), (8, 0, false), (9, 0, true), (9, 1, false)] {
        assert_eq!(super::uses_recurrent(num_tokens, num_spec_tokens), recurrent);
    }
}

#[test]
#[should_panic(expected = "GDN candidate recurrent-state constants require Dv divisible by the SIMDgroup V-row count")]
fn test_candidate_register_v_rejects_v_dim_tail() {
    let mut config = fixture_config();
    config.v_head_dim = 5;
    config.validate();
}

#[test]
#[should_panic(expected = "GDN recurrent V-row tile must divide Dv and fit device shared memory")]
fn test_new_insufficient_shared_memory() {
    let config = Config {
        qk_head_dim: 512,
        ..fixture_config()
    };
    ComputeConstants::from_config(config, 32 * 1024);
}

#[test]
#[should_panic(expected = "GDN candidate register-V requires Dqk divisible by the SIMDgroup width")]
fn test_candidate_register_v_rejects_qk_dim_tail_at_init() {
    let mut config = fixture_config();
    config.qk_head_dim = 33;
    config.validate();
}

#[test]
#[should_panic(expected = "GDN candidate recurrent thread-block V-row count must divide Dv")]
fn test_candidate_register_v_rejects_incomplete_grouped_launch_at_init() {
    let mut config = fixture_config();
    config.v_head_dim = 6;
    config.validate();
}

#[test]
#[should_panic(expected = "GDN convolution exceeds the shader u32 count domain")]
fn test_shape_rejects_shader_count_overflow() {
    let config = Config {
        projection_layout: super::ProjectionLayout::Separate,
        num_qk_heads: 1,
        qk_head_dim: 32,
        num_v_heads: 1,
        v_head_dim: 4,
        conv_kernel_size: 2,
        q_scale: 1.0,
        norm_eps: 1.0e-6,
        output_gate: OutputGate::Silu,
    };
    let shape = Shape {
        num_total_reqs: 1,
        num_total_tokens: 1 << 30,
    };
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let kernels = Compute::new(&device, config);
    let buffer = Buffer::new_zeroed(&device, size_of::<u16>());
    let mut builder = stream.create_replay_program();

    builder.record(kernels.invoke(
        shape,
        Buffers {
            qkv: &buffer,
            a: &buffer,
            b: &buffer,
            z: &buffer,
            conv_weight: &buffer,
            norm_weight: &buffer,
            a_log: &buffer,
            dt_bias: &buffer,
            cu_tokens: &buffer,
            src_recurrent_state_slots: &buffer,
            src_conv_state_slots: &buffer,
            flat_recurrent_state_write_slots: &buffer,
            flat_conv_state_write_slots: &buffer,
            conv_state: &buffer,
            conv_state_offset_bytes: 0,
            next_conv_state: &buffer,
            next_conv_state_offset_bytes: 0,
            recurrent_state_arena: &buffer,
            recurrent_state_arena_offset_bytes: 0,
            conv_qkv: &buffer,
            recurrent_output: &buffer,
            norm_gated_output: &buffer,
        },
        ReplayU32::Fixed(shape.num_total_reqs),
        ReplayU32::Fixed(shape.num_total_tokens),
    ));
}

#[test]
fn test_final_recurrent_state_fixed() {
    let shape = fixture_shape(1, 1);
    let cu_tokens = vec![0_u32, 1];
    let src_state_slots = vec![0_u32];
    let dst_slot_ids = vec![0_u32];
    let qkv = fixture_values(fixture_config().num_qkv_values(shape), 0.03125, 3);
    let conv_state = fixture_values(fixture_config().num_conv_state_values(shape), 0.015625, 7);
    let recurrent_state = fixture_values(fixture_config().recurrent_state_stride(), 0.0078125, 11);
    let conv_weight = fixture_values(
        fixture_config().qkv_dim() as usize * fixture_config().conv_kernel_size as usize,
        0.00390625,
        13,
    );
    let a = fixture_values(
        shape.num_total_tokens as usize * fixture_config().num_v_heads as usize,
        0.0625,
        17,
    );
    let b = fixture_values(
        shape.num_total_tokens as usize * fixture_config().num_v_heads as usize,
        0.0625,
        19,
    );
    let z = fixture_values(fixture_config().num_recurrent_output_values(shape), 0.03125, 23);
    let norm_weight = vec![1.0_f32; fixture_config().v_head_dim as usize];
    let a_log = vec![-0.25_f32; fixture_config().num_v_heads as usize];
    let dt_bias = vec![0.125_f32; fixture_config().num_v_heads as usize];

    let actual = run_gdn_core(
        shape,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &z,
        &norm_weight,
        &a_log,
        &dt_bias,
        &cu_tokens,
        &src_state_slots,
        &dst_slot_ids,
    );
    assert_gdn_reference_matches(
        shape,
        &cu_tokens,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &a_log,
        &dt_bias,
        &actual,
        5.0e-3,
    );
}

#[test]
fn test_final_recurrent_state_random() {
    let random_seed = 0x729B_40D6;
    let shape = fixture_shape(1, 3);
    let cu_tokens = vec![0_u32, 3];
    let src_state_slots = vec![0_u32];
    let dst_slot_ids = vec![0_u32];
    let qkv = generated_values(fixture_config().num_qkv_values(shape), random_seed);
    let conv_state = generated_values(
        fixture_config().num_conv_state_values(shape),
        random_seed.wrapping_add(1),
    );
    let recurrent_state = generated_values(fixture_config().recurrent_state_stride(), random_seed.wrapping_add(2));
    let conv_weight = generated_values(
        fixture_config().qkv_dim() as usize * fixture_config().conv_kernel_size as usize,
        random_seed.wrapping_add(3),
    )
    .into_iter()
    .map(|value| bf16::from_f32(value * 0.125).to_f32())
    .collect::<Vec<_>>();
    let a = generated_values(
        shape.num_total_tokens as usize * fixture_config().num_v_heads as usize,
        random_seed.wrapping_add(4),
    );
    let b = generated_values(
        shape.num_total_tokens as usize * fixture_config().num_v_heads as usize,
        random_seed.wrapping_add(5),
    );
    let z = generated_values(
        fixture_config().num_recurrent_output_values(shape),
        random_seed.wrapping_add(6),
    );
    let norm_weight = vec![1.0_f32; fixture_config().v_head_dim as usize];
    let a_log = vec![-0.125_f32; fixture_config().num_v_heads as usize];
    let dt_bias = vec![0.0625_f32; fixture_config().num_v_heads as usize];

    let actual = run_gdn_core(
        shape,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &z,
        &norm_weight,
        &a_log,
        &dt_bias,
        &cu_tokens,
        &src_state_slots,
        &dst_slot_ids,
    );
    assert_gdn_reference_matches(
        shape,
        &cu_tokens,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &a_log,
        &dt_bias,
        &actual,
        5.0e-3,
    );
}

#[test]
fn test_ragged_multi_random() {
    let random_seed = 0xE13C_58A4;
    let shape = fixture_shape(2, 4);
    let cu_tokens = vec![0_u32, 1, 4];
    let src_state_slots = vec![0_u32, 1];
    let dst_slot_ids = vec![0_u32, 1];
    let qkv = generated_values(fixture_config().num_qkv_values(shape), random_seed);
    let conv_state = generated_values(
        fixture_config().num_conv_state_values(shape),
        random_seed.wrapping_add(1),
    );
    let recurrent_state = generated_values(
        shape.num_total_reqs as usize * fixture_config().recurrent_state_stride(),
        random_seed.wrapping_add(2),
    );
    let conv_weight = generated_values(
        fixture_config().qkv_dim() as usize * fixture_config().conv_kernel_size as usize,
        random_seed.wrapping_add(3),
    )
    .into_iter()
    .map(|value| bf16::from_f32(value * 0.125).to_f32())
    .collect::<Vec<_>>();
    let a = generated_values(
        shape.num_total_tokens as usize * fixture_config().num_v_heads as usize,
        random_seed.wrapping_add(4),
    );
    let b = generated_values(
        shape.num_total_tokens as usize * fixture_config().num_v_heads as usize,
        random_seed.wrapping_add(5),
    );
    let z = generated_values(
        fixture_config().num_recurrent_output_values(shape),
        random_seed.wrapping_add(6),
    );
    let norm_weight = vec![1.0_f32; fixture_config().v_head_dim as usize];
    let a_log = vec![-0.125_f32; fixture_config().num_v_heads as usize];
    let dt_bias = vec![0.0625_f32; fixture_config().num_v_heads as usize];

    let actual = run_gdn_core(
        shape,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &z,
        &norm_weight,
        &a_log,
        &dt_bias,
        &cu_tokens,
        &src_state_slots,
        &dst_slot_ids,
    );
    assert_gdn_reference_matches(
        shape,
        &cu_tokens,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &a_log,
        &dt_bias,
        &actual,
        5.0e-3,
    );
}

#[test]
fn test_candidate_state_preserves_distinct_source_slots() {
    const NUM_REQS: usize = 2;
    const ROWS_PER_REQ: usize = 4;
    const NUM_STATE_SLOTS: usize = 8;
    const CANARY: f32 = -777.0;

    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config {
        projection_layout: super::ProjectionLayout::Separate,
        num_qk_heads: 2,
        qk_head_dim: 128,
        num_v_heads: 4,
        v_head_dim: 128,
        conv_kernel_size: 3,
        q_scale: 128.0_f32.sqrt().recip(),
        norm_eps: 1.0e-6,
        output_gate: OutputGate::Silu,
    };
    let shape = fixture_shape(NUM_REQS as u32, (NUM_REQS * ROWS_PER_REQ) as u32);
    let kernels = Compute::new(&device, config);
    let qkv_values = fixture_values(config.num_qkv_values(shape), 0.03125, 3);
    let a_values = fixture_values(shape.num_total_tokens as usize * config.num_v_heads as usize, 0.0625, 5);
    let b_values = fixture_values(shape.num_total_tokens as usize * config.num_v_heads as usize, 0.0625, 7);
    let z_values = fixture_values(config.num_recurrent_output_values(shape), 0.03125, 11);
    let conv_weight_values = fixture_values(config.num_conv_weight_values(), 0.00390625, 13);
    let norm_weight_values = vec![1.0; config.v_head_dim as usize];
    let a_log_values = vec![-0.25; config.num_v_heads as usize];
    let dt_bias_values = vec![0.125; config.num_v_heads as usize];
    let cu_tokens_values = [0_u32, 4, 8];
    let src_recurrent_state_slot_values = [0_u32, 1];
    let src_conv_state_slot_values = [1_u32, 0];
    let recurrent_candidate_slot_values = [2_u32, 3, 4, u32::MAX, 5, 6, 7, u32::MAX];
    let conv_candidate_slot_values = [7_u32, 6, 5, u32::MAX, 4, 3, 2, u32::MAX];
    let qkv = bf16_buffer(&device, &qkv_values);
    let a = bf16_buffer(&device, &a_values);
    let b = bf16_buffer(&device, &b_values);
    let z = bf16_buffer(&device, &z_values);
    let conv_weight = bf16_buffer(&device, &conv_weight_values);
    let norm_weight = bf16_buffer(&device, &norm_weight_values);
    let a_log = bf16_buffer(&device, &a_log_values);
    let dt_bias = bf16_buffer(&device, &dt_bias_values);
    let cu_tokens = Buffer::from_slice(&device, &cu_tokens_values);
    let src_recurrent_state_slots = Buffer::from_slice(&device, &src_recurrent_state_slot_values);
    let src_conv_state_slots = Buffer::from_slice(&device, &src_conv_state_slot_values);
    let recurrent_candidate_slots = Buffer::from_slice(&device, &recurrent_candidate_slot_values);
    let conv_candidate_slots = Buffer::from_slice(&device, &conv_candidate_slot_values);
    let conv_stride = config.qkv_dim() as usize * config.conv_state_len() as usize;
    let recurrent_stride = config.recurrent_state_stride();
    let reference_core = GDNCore {
        model_layer_index: 0,
        hidden_dim: config.num_v_heads as usize * config.v_head_dim as usize,
        num_qk_heads: config.num_qk_heads as usize,
        qk_head_dim: config.qk_head_dim as usize,
        num_v_heads: config.num_v_heads as usize,
        v_head_dim: config.v_head_dim as usize,
        conv_kernel_size: config.conv_kernel_size as usize,
        q_scale: config.q_scale,
        output_gate: match config.output_gate {
            OutputGate::Silu => GDNOutputGate::Silu,
            OutputGate::Sigmoid => GDNOutputGate::Sigmoid,
        },
    };
    let source_conv = fixture_values(NUM_REQS * conv_stride, 0.015625, 17);
    let source_recurrent = fixture_values(NUM_REQS * recurrent_stride, 0.0078125, 19);

    let conv_arena = Buffer::new_zeroed_elements(&device, NUM_STATE_SLOTS * conv_stride, Dtype::Bfloat16);
    write_bf16(&conv_arena, 0, &vec![CANARY; NUM_STATE_SLOTS * conv_stride]);
    write_bf16(&conv_arena, 0, &source_conv);
    let recurrent_arena = Buffer::new_zeroed_elements(&device, NUM_STATE_SLOTS * recurrent_stride, Dtype::Bfloat16);
    write_bf16(&recurrent_arena, 0, &vec![CANARY; NUM_STATE_SLOTS * recurrent_stride]);
    write_bf16(&recurrent_arena, 0, &source_recurrent);
    let conv_qkv = Buffer::new_zeroed_elements(&device, config.num_qkv_values(shape), Dtype::Bfloat16);
    let recurrent_output =
        Buffer::new_zeroed_elements(&device, config.num_recurrent_output_values(shape), Dtype::Bfloat16);
    let norm_gated_output =
        Buffer::new_zeroed_elements(&device, config.num_recurrent_output_values(shape), Dtype::Bfloat16);
    let buffers = Buffers {
        qkv: &qkv,
        a: &a,
        b: &b,
        z: &z,
        conv_weight: &conv_weight,
        norm_weight: &norm_weight,
        a_log: &a_log,
        dt_bias: &dt_bias,
        cu_tokens: &cu_tokens,
        src_recurrent_state_slots: &src_recurrent_state_slots,
        src_conv_state_slots: &src_conv_state_slots,
        flat_recurrent_state_write_slots: &recurrent_candidate_slots,
        flat_conv_state_write_slots: &conv_candidate_slots,
        conv_state: &conv_arena,
        conv_state_offset_bytes: 0,
        next_conv_state: &conv_arena,
        next_conv_state_offset_bytes: 0,
        recurrent_state_arena: &recurrent_arena,
        recurrent_state_arena_offset_bytes: 0,
        conv_qkv: &conv_qkv,
        recurrent_output: &recurrent_output,
        norm_gated_output: &norm_gated_output,
    };
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_with_candidate_state_update(
        shape,
        buffers,
        ReplayU32::Fixed(shape.num_total_reqs),
        ReplayU32::Fixed(shape.num_total_tokens),
    ));
    stream.submit_replay(&builder.build()).wait();

    let actual_conv_qkv = read_bf16(&conv_qkv, 0, config.num_qkv_values(shape));
    let actual_recurrent_output = read_bf16(&recurrent_output, 0, config.num_recurrent_output_values(shape));
    let actual_norm_gated_output = read_bf16(&norm_gated_output, 0, config.num_recurrent_output_values(shape));
    let actual_conv_arena = read_bf16(&conv_arena, 0, NUM_STATE_SLOTS * conv_stride);
    let actual_recurrent_arena = read_bf16(&recurrent_arena, 0, NUM_STATE_SLOTS * recurrent_stride);
    let mut expected_conv_qkv = vec![0.0; config.num_qkv_values(shape)];
    let mut expected_recurrent_output = vec![0.0; config.num_recurrent_output_values(shape)];
    let mut expected_norm_gated_output = vec![0.0; config.num_recurrent_output_values(shape)];
    let mut expected_conv_arena = vec![CANARY; NUM_STATE_SLOTS * conv_stride];
    expected_conv_arena[..NUM_REQS * conv_stride].copy_from_slice(&source_conv);
    let mut expected_recurrent_arena = vec![CANARY; NUM_STATE_SLOTS * recurrent_stride];
    expected_recurrent_arena[..NUM_REQS * recurrent_stride].copy_from_slice(&source_recurrent);

    for req_index in 0..NUM_REQS {
        let flat_start = req_index * ROWS_PER_REQ;
        let source_conv_slot = src_conv_state_slot_values[req_index] as usize;
        let source_recurrent_slot = src_recurrent_state_slot_values[req_index] as usize;
        for num_prefix_rows in 1..=ROWS_PER_REQ {
            let qkv_start = flat_start * config.qkv_dim() as usize;
            let qkv_end = (flat_start + num_prefix_rows) * config.qkv_dim() as usize;
            let gate_start = flat_start * config.num_v_heads as usize;
            let gate_end = (flat_start + num_prefix_rows) * config.num_v_heads as usize;
            let output_start = flat_start * config.num_v_heads as usize * config.v_head_dim as usize;
            let output_end = (flat_start + num_prefix_rows) * config.num_v_heads as usize * config.v_head_dim as usize;
            let prefix_cu_tokens = [0_u32, num_prefix_rows as u32];
            let conv_reference = gdn_short_conv_reference(
                &reference_core,
                &prefix_cu_tokens,
                &source_conv[source_conv_slot * conv_stride..(source_conv_slot + 1) * conv_stride],
                &qkv_values[qkv_start..qkv_end],
                &conv_weight_values,
            );
            let recurrent_reference = gdn_recurrent_reference(
                &reference_core,
                GDNRecurrentReferenceInput {
                    cu_tokens: &prefix_cu_tokens,
                    source_recurrent_state: &source_recurrent
                        [source_recurrent_slot * recurrent_stride..(source_recurrent_slot + 1) * recurrent_stride],
                    conv_qkv: &conv_reference.conv_qkv,
                    a: &a_values[gate_start..gate_end],
                    b: &b_values[gate_start..gate_end],
                    a_log: &a_log_values,
                    dt_bias: &dt_bias_values,
                },
            );
            if num_prefix_rows == ROWS_PER_REQ {
                expected_conv_qkv[qkv_start..qkv_end].copy_from_slice(&conv_reference.conv_qkv);
                expected_recurrent_output[output_start..output_end]
                    .copy_from_slice(&recurrent_reference.recurrent_output);
                expected_norm_gated_output[output_start..output_end].copy_from_slice(&gdn_output_norm_gate_reference(
                    &reference_core,
                    &recurrent_reference.recurrent_output,
                    &z_values[output_start..output_end],
                    &norm_weight_values,
                    config.norm_eps,
                ));
            }
            let flat_row = flat_start + num_prefix_rows - 1;
            let conv_candidate_slot = conv_candidate_slot_values[flat_row];
            if conv_candidate_slot != u32::MAX {
                let candidate_slot = conv_candidate_slot as usize;
                expected_conv_arena[candidate_slot * conv_stride..(candidate_slot + 1) * conv_stride]
                    .copy_from_slice(&conv_reference.next_conv_state);
            }
            let recurrent_candidate_slot = recurrent_candidate_slot_values[flat_row];
            if recurrent_candidate_slot != u32::MAX {
                let candidate_slot = recurrent_candidate_slot as usize;
                expected_recurrent_arena[candidate_slot * recurrent_stride..(candidate_slot + 1) * recurrent_stride]
                    .copy_from_slice(&recurrent_reference.next_recurrent_state);
            }
        }
    }

    assert_close(&actual_conv_qkv, &expected_conv_qkv, 5.0e-3);
    assert_close(&actual_recurrent_output, &expected_recurrent_output, 5.0e-3);
    assert_close(&actual_norm_gated_output, &expected_norm_gated_output, 5.0e-3);
    assert_close(&actual_conv_arena, &bf16_round_trip(&expected_conv_arena), 5.0e-3);
    assert_close(
        &actual_recurrent_arena,
        &bf16_round_trip(&expected_recurrent_arena),
        5.0e-3,
    );
    assert_eq!(&actual_conv_arena[..NUM_REQS * conv_stride], source_conv);
    assert_eq!(&actual_recurrent_arena[..NUM_REQS * recurrent_stride], source_recurrent);
}

#[test]
fn test_candidate_replay_matches_reference_across_active_counts() {
    for num_total_tokens in [8, 7] {
        check_recurrent_replay_across_active_counts(num_total_tokens, true);
    }
}

#[test]
fn test_final_replay_matches_reference_across_active_counts() {
    for num_total_tokens in [8, 7] {
        check_recurrent_replay_across_active_counts(num_total_tokens, false);
    }
}

fn check_recurrent_replay_across_active_counts(num_total_tokens: u32, candidate_states: bool) {
    const NUM_REQUESTS: usize = 8;
    const NUM_STATE_SLOTS: usize = NUM_REQUESTS * 3;
    const CANARY: f32 = -777.0;

    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = fixture_config();
    let shape = fixture_shape(NUM_REQUESTS as u32, num_total_tokens);
    let kernels = Compute::new(&device, config);
    let qkv_values = fixture_values(config.num_qkv_values(shape), 0.03125, 3);
    let a_values = fixture_values(shape.num_total_tokens as usize * config.num_v_heads as usize, 0.0625, 5);
    let b_values = fixture_values(shape.num_total_tokens as usize * config.num_v_heads as usize, 0.0625, 7);
    let z_values = fixture_values(config.num_recurrent_output_values(shape), 0.03125, 11);
    let conv_weight_values = fixture_values(
        config.qkv_dim() as usize * config.conv_kernel_size as usize,
        0.00390625,
        13,
    );
    let norm_weight_values = vec![1.0_f32; config.v_head_dim as usize];
    let a_log_values = vec![-0.25_f32; config.num_v_heads as usize];
    let dt_bias_values = vec![0.125_f32; config.num_v_heads as usize];
    let conv_state_stride = config.qkv_dim() as usize * config.conv_state_len() as usize;
    let recurrent_state_stride = config.recurrent_state_stride();
    let source_conv_states = fixture_values(NUM_REQUESTS * conv_state_stride, 0.015625, 17);
    let source_recurrent_states = fixture_values(NUM_REQUESTS * recurrent_state_stride, 0.0078125, 19);

    let qkv = Buffer::new_zeroed_elements(&device, config.num_qkv_values(shape), Dtype::Bfloat16);
    let a = Buffer::new_zeroed_elements(
        &device,
        shape.num_total_tokens as usize * config.num_v_heads as usize,
        Dtype::Bfloat16,
    );
    let b = Buffer::new_zeroed_elements(
        &device,
        shape.num_total_tokens as usize * config.num_v_heads as usize,
        Dtype::Bfloat16,
    );
    let z = Buffer::new_zeroed_elements(&device, config.num_recurrent_output_values(shape), Dtype::Bfloat16);
    let conv_weight = bf16_buffer(&device, &conv_weight_values);
    let norm_weight = bf16_buffer(&device, &norm_weight_values);
    let a_log = bf16_buffer(&device, &a_log_values);
    let dt_bias = bf16_buffer(&device, &dt_bias_values);
    let cu_tokens = Buffer::from_slice(&device, &(0..=NUM_REQUESTS as u32).collect::<Vec<_>>());
    let src_slot_values = (0..NUM_REQUESTS as u32).map(|req| req * 3).collect::<Vec<_>>();
    let candidate_slot_values = (0..NUM_REQUESTS as u32).map(|req| req * 3 + 2).collect::<Vec<_>>();
    let src_state_slots = Buffer::from_slice(&device, &src_slot_values);
    let flat_materialized_state_slots = Buffer::from_slice(&device, &candidate_slot_values);
    let conv_state_arena = Buffer::new_zeroed_elements(&device, NUM_STATE_SLOTS * conv_state_stride, Dtype::Bfloat16);
    let recurrent_state_arena =
        Buffer::new_zeroed_elements(&device, NUM_STATE_SLOTS * recurrent_state_stride, Dtype::Bfloat16);
    let conv_qkv = Buffer::new_zeroed_elements(&device, config.num_qkv_values(shape), Dtype::Bfloat16);
    let recurrent_output =
        Buffer::new_zeroed_elements(&device, config.num_recurrent_output_values(shape), Dtype::Bfloat16);
    let norm_gated_output =
        Buffer::new_zeroed_elements(&device, config.num_recurrent_output_values(shape), Dtype::Bfloat16);
    let replay = {
        let mut builder = stream.create_replay_program();
        let buffers = Buffers {
            qkv: &qkv,
            a: &a,
            b: &b,
            z: &z,
            conv_weight: &conv_weight,
            norm_weight: &norm_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            cu_tokens: &cu_tokens,
            src_recurrent_state_slots: &src_state_slots,
            src_conv_state_slots: &src_state_slots,
            flat_recurrent_state_write_slots: &flat_materialized_state_slots,
            flat_conv_state_write_slots: &flat_materialized_state_slots,
            conv_state: &conv_state_arena,
            conv_state_offset_bytes: 0,
            next_conv_state: &conv_state_arena,
            next_conv_state_offset_bytes: 0,
            recurrent_state_arena: &recurrent_state_arena,
            recurrent_state_arena_offset_bytes: 0,
            conv_qkv: &conv_qkv,
            recurrent_output: &recurrent_output,
            norm_gated_output: &norm_gated_output,
        };
        if candidate_states {
            builder.record(kernels.invoke_with_candidate_state_update(
                shape,
                buffers,
                ReplayU32::Parameter(NUM_ACTIVE_REQUESTS),
                ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            ));
        } else {
            builder.record(kernels.invoke(
                shape,
                buffers,
                ReplayU32::Parameter(NUM_ACTIVE_REQUESTS),
                ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            ));
        }
        builder.build()
    };
    write_bf16(&qkv, 0, &qkv_values);
    write_bf16(&a, 0, &a_values);
    write_bf16(&b, 0, &b_values);
    write_bf16(&z, 0, &z_values);

    for num_active in [1_usize, 8, 3, 7, 2, 6, 4, 5]
        .into_iter()
        .filter(|&count| count <= num_total_tokens as usize)
    {
        let mut conv_arena_values = vec![CANARY; NUM_STATE_SLOTS * conv_state_stride];
        for req in 0..NUM_REQUESTS {
            let src_begin = req * conv_state_stride;
            let arena_begin = req * 3 * conv_state_stride;
            conv_arena_values[arena_begin..arena_begin + conv_state_stride]
                .copy_from_slice(&source_conv_states[src_begin..src_begin + conv_state_stride]);
        }
        write_bf16(&conv_state_arena, 0, &conv_arena_values);
        let mut recurrent_arena_values = vec![CANARY; NUM_STATE_SLOTS * recurrent_state_stride];
        for req in 0..NUM_REQUESTS {
            let src_begin = req * recurrent_state_stride;
            let arena_begin = req * 3 * recurrent_state_stride;
            recurrent_arena_values[arena_begin..arena_begin + recurrent_state_stride]
                .copy_from_slice(&source_recurrent_states[src_begin..src_begin + recurrent_state_stride]);
        }
        write_bf16(&recurrent_state_arena, 0, &recurrent_arena_values);

        stream
            .submit_replay_with_arguments(
                &replay,
                &ReplayArguments::new()
                    .with_u32(NUM_ACTIVE_REQUESTS, num_active as u32)
                    .with_u32(NUM_ACTIVE_TOKENS, num_active as u32),
            )
            .wait();

        let active_shape = fixture_shape(num_active as u32, num_active as u32);
        let active_cu_tokens = (0..=num_active as u32).collect::<Vec<_>>();
        let active_qkv_values = &qkv_values[..config.num_qkv_values(active_shape)];
        let active_a_values = &a_values[..num_active * config.num_v_heads as usize];
        let active_b_values = &b_values[..num_active * config.num_v_heads as usize];
        let active_z_values = &z_values[..config.num_recurrent_output_values(active_shape)];
        let conv_reference = gdn_short_conv_reference(
            &fixture_core(active_shape),
            &active_cu_tokens,
            &source_conv_states[..num_active * conv_state_stride],
            active_qkv_values,
            &conv_weight_values,
        );
        let recurrent_reference = gdn_recurrent_reference(
            &fixture_core(active_shape),
            GDNRecurrentReferenceInput {
                cu_tokens: &active_cu_tokens,
                source_recurrent_state: &source_recurrent_states[..num_active * recurrent_state_stride],
                conv_qkv: &conv_reference.conv_qkv,
                a: active_a_values,
                b: active_b_values,
                a_log: &a_log_values,
                dt_bias: &dt_bias_values,
            },
        );
        let norm_gate_reference = gdn_output_norm_gate_reference(
            &fixture_core(active_shape),
            &recurrent_reference.recurrent_output,
            active_z_values,
            &norm_weight_values,
            config.norm_eps,
        );
        assert_close(
            &read_bf16(&conv_qkv, 0, config.num_qkv_values(active_shape)),
            &conv_reference.conv_qkv,
            5.0e-3,
        );
        assert_close(
            &read_bf16(&recurrent_output, 0, config.num_recurrent_output_values(active_shape)),
            &recurrent_reference.recurrent_output,
            5.0e-3,
        );
        assert_close(
            &read_bf16(&norm_gated_output, 0, config.num_recurrent_output_values(active_shape)),
            &norm_gate_reference,
            5.0e-3,
        );
        let conv_state_after = read_bf16(&conv_state_arena, 0, NUM_STATE_SLOTS * conv_state_stride);
        let recurrent_state_after = read_bf16(&recurrent_state_arena, 0, NUM_STATE_SLOTS * recurrent_state_stride);
        let mut expected_conv_arena = conv_arena_values;
        let mut expected_recurrent_arena = recurrent_arena_values;
        for (req_index, &candidate_slot_value) in candidate_slot_values.iter().take(num_active).enumerate() {
            let candidate_slot = candidate_slot_value as usize;
            let conv_expected =
                &conv_reference.next_conv_state[req_index * conv_state_stride..(req_index + 1) * conv_state_stride];
            let recurrent_expected = &recurrent_reference.next_recurrent_state
                [req_index * recurrent_state_stride..(req_index + 1) * recurrent_state_stride];
            expected_conv_arena[candidate_slot * conv_state_stride..(candidate_slot + 1) * conv_state_stride]
                .copy_from_slice(conv_expected);
            expected_recurrent_arena
                [candidate_slot * recurrent_state_stride..(candidate_slot + 1) * recurrent_state_stride]
                .copy_from_slice(recurrent_expected);
        }
        assert_close(&conv_state_after, &bf16_round_trip(&expected_conv_arena), 5.0e-3);
        assert_close(
            &recurrent_state_after,
            &bf16_round_trip(&expected_recurrent_arena),
            5.0e-3,
        );
    }
}

#[test]
fn test_mixed_replay_matches_reference_across_prefill_decode_counts() {
    assert_mixed_replay_matches_reference(
        4,
        32,
        &[
            (0, &[1]),
            (4, &[9, 8, 7, 8]),
            (1, &[17, 2, 3]),
            (2, &[5, 11, 4, 2]),
            (2, &[8, 9]),
            (0, &[1, 4, 2, 3]),
            (1, &[1]),
        ],
    );
    // Request and token capacities are independent. An affine boundary can
    // select fewer total tokens than total requests, with valid active work.
    assert_mixed_replay_matches_reference(
        8,
        7,
        &[
            (0, &[1]),
            (7, &[1, 1, 1, 1, 1, 1, 1]),
            (1, &[2, 1]),
            (0, &[1, 1, 1, 1, 1, 1, 1]),
            (2, &[2, 3]),
        ],
    );
}

#[test]
fn test_sigmoid_gate_replay_matches_reference_across_prefill_decode_counts() {
    assert_mixed_replay(
        Config {
            output_gate: OutputGate::Sigmoid,
            ..fixture_config()
        },
        4,
        48,
        &[(0, &[1]), (1, &[17, 2, 3]), (2, &[5, 11, 4, 2]), (0, &[1, 4, 2, 3])],
    );
}

#[test]
fn test_recurrent_replay_across_head_dimensions() {
    // Cover padded V rows and multiple V blocks, including wider Q/K heads.
    for (qk_head_dim, v_head_dim) in [(32, 4), (128, 128), (192, 32), (256, 8), (256, 128)] {
        assert_mixed_replay(
            Config {
                qk_head_dim,
                v_head_dim,
                ..fixture_config()
            },
            4,
            48,
            &[
                (0, &[1]),
                (1, &[1]),
                (1, &[2]),
                (1, &[8]),
                (1, &[16]),
                (1, &[17]),
                (3, &[19, 17, 8, 4]),
                (1, &[33, 2, 3]),
                (2, &[5, 11, 4, 2]),
                (0, &[1, 4, 2, 3]),
                (1, &[1]),
            ],
        );
    }
}

// Record the same kernels with an explicit dependency before decode. This
// isolates scheduling from the CPU oracle's different floating-point order.
struct SerialMixedInvocation<'a>(MixedInvocation<'a>);

impl Operator for SerialMixedInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let invocation = self.0;
        let compute = invocation.compute;
        compute.record_short_conv(
            recorder,
            invocation.shape,
            &invocation.buffers,
            invocation.num_active_reqs,
            invocation.num_active_tokens,
            !invocation.write_candidate_states,
        );
        recurrent::record_recurrent_state(
            compute,
            recorder,
            invocation.shape,
            &invocation.buffers,
            invocation.num_active_recurrent_requests,
            invocation.write_candidate_states,
        );
        recorder.record_with_barrier_before(SerialDecodeInvocation(&invocation));
        compute.record_output_norm_gate(
            recorder,
            invocation.shape,
            &invocation.buffers,
            invocation.num_active_tokens,
        );
    }
}

struct SerialDecodeInvocation<'a>(&'a MixedInvocation<'a>);

impl Operator for SerialDecodeInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let invocation = self.0;
        if invocation.write_candidate_states {
            recurrent_snapshot::record_candidate_recurrent_state(
                invocation.compute,
                recorder,
                invocation.shape,
                &invocation.buffers,
                invocation.num_active_reqs,
                invocation.num_active_recurrent_requests,
            );
        } else {
            recurrent_snapshot::record_final_recurrent_state(
                invocation.compute,
                recorder,
                invocation.shape,
                &invocation.buffers,
                invocation.num_active_reqs,
                invocation.num_active_recurrent_requests,
            );
        }
    }
}

fn assert_mixed_replay_matches_reference(num_total_reqs: usize, num_total_tokens: usize, rounds: &[(u32, &[u32])]) {
    assert_mixed_replay(fixture_config(), num_total_reqs, num_total_tokens, rounds);
}

fn assert_mixed_replay(config: Config, num_total_reqs: usize, num_total_tokens: usize, rounds: &[(u32, &[u32])]) {
    let num_state_slots = num_total_reqs + num_total_tokens;
    const CANARY: f32 = -777.0;

    let device = Device::system_default();
    let stream = Stream::new(&device);
    let shape = fixture_shape(num_total_reqs as u32, num_total_tokens as u32);
    let kernels = Compute::new(&device, config);
    let qkv_values = bf16_round_trip(&fixture_values(config.num_qkv_values(shape), 0.03125, 3));
    let a_values = bf16_round_trip(&fixture_values(num_total_tokens, 0.0625, 5));
    let b_values = bf16_round_trip(&fixture_values(num_total_tokens, 0.0625, 7));
    let z_values = bf16_round_trip(&fixture_values(config.num_recurrent_output_values(shape), 0.03125, 11));
    let conv_weight_values = bf16_round_trip(&fixture_values(
        config.qkv_dim() as usize * config.conv_kernel_size as usize,
        0.00390625,
        13,
    ));
    let norm_weight_values = vec![1.0; config.v_head_dim as usize];
    let a_log_values = vec![-0.25; config.num_v_heads as usize];
    let dt_bias_values = vec![0.125; config.num_v_heads as usize];
    let conv_stride = config.qkv_dim() as usize * config.conv_state_len() as usize;
    let recurrent_stride = config.recurrent_state_stride();
    let source_conv_states = bf16_round_trip(&fixture_values(num_total_reqs * conv_stride, 0.015625, 17));
    let source_recurrent_states = bf16_round_trip(&fixture_values(num_total_reqs * recurrent_stride, 0.0078125, 19));
    let qkv = bf16_buffer(&device, &qkv_values);
    let a = bf16_buffer(&device, &a_values);
    let b = bf16_buffer(&device, &b_values);
    let z = bf16_buffer(&device, &z_values);
    let conv_weight = bf16_buffer(&device, &conv_weight_values);
    let norm_weight = bf16_buffer(&device, &norm_weight_values);
    let a_log = bf16_buffer(&device, &a_log_values);
    let dt_bias = bf16_buffer(&device, &dt_bias_values);
    let cu_tokens = Buffer::from_slice(&device, &vec![u32::MAX; num_total_reqs + 1]);
    let src_slots = Buffer::from_slice(&device, &(0..num_total_reqs as u32).collect::<Vec<_>>());
    let recurrent_write_slots = Buffer::from_slice(&device, &vec![u32::MAX; num_total_tokens]);
    let conv_write_slots = Buffer::from_slice(&device, &vec![u32::MAX; num_total_tokens]);
    let conv_state = Buffer::new_zeroed_elements(&device, num_state_slots * conv_stride, Dtype::Bfloat16);
    let recurrent_state = Buffer::new_zeroed_elements(&device, num_state_slots * recurrent_stride, Dtype::Bfloat16);
    let conv_qkv = Buffer::new_zeroed_elements(&device, config.num_qkv_values(shape), Dtype::Bfloat16);
    let recurrent_output =
        Buffer::new_zeroed_elements(&device, config.num_recurrent_output_values(shape), Dtype::Bfloat16);
    let norm_gated_output =
        Buffer::new_zeroed_elements(&device, config.num_recurrent_output_values(shape), Dtype::Bfloat16);

    let buffers = Buffers {
        qkv: &qkv,
        a: &a,
        b: &b,
        z: &z,
        conv_weight: &conv_weight,
        norm_weight: &norm_weight,
        a_log: &a_log,
        dt_bias: &dt_bias,
        cu_tokens: &cu_tokens,
        src_recurrent_state_slots: &src_slots,
        src_conv_state_slots: &src_slots,
        flat_recurrent_state_write_slots: &recurrent_write_slots,
        flat_conv_state_write_slots: &conv_write_slots,
        conv_state: &conv_state,
        conv_state_offset_bytes: 0,
        next_conv_state: &conv_state,
        next_conv_state_offset_bytes: 0,
        recurrent_state_arena: &recurrent_state,
        recurrent_state_arena_offset_bytes: 0,
        conv_qkv: &conv_qkv,
        recurrent_output: &recurrent_output,
        norm_gated_output: &norm_gated_output,
    };
    for write_candidate_states in [false, true] {
        let [replay, serial_replay] = [false, true].map(|serialized| {
            let mut builder = stream.create_replay_program();
            let invocation = kernels.invoke_mixed(
                shape,
                buffers,
                ReplayU32::Parameter(NUM_ACTIVE_REQUESTS),
                ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
                ReplayU32::Parameter(NUM_ACTIVE_RECURRENT_REQUESTS),
                write_candidate_states,
            );
            if serialized {
                builder.record(SerialMixedInvocation(invocation));
            } else {
                builder.record(invocation);
            }
            builder.build()
        });
        for &(num_prefill, request_lengths) in rounds {
            let mut cu_values = vec![0];
            for &length in request_lengths {
                cu_values.push(cu_values.last().unwrap() + length);
            }
            let num_tokens = *cu_values.last().unwrap() as usize;
            cu_tokens.write_typed(0, &vec![u32::MAX; num_total_reqs + 1]);
            cu_tokens.write_typed(0, &cu_values);
            let mut recurrent_slots = vec![u32::MAX; num_total_tokens];
            let mut conv_slots = vec![u32::MAX; num_total_tokens];
            // Final-only execution must ignore valid nonfinal destination
            // slots. Candidate execution must materialize those same rows.
            for (req, bounds) in cu_values.windows(2).enumerate() {
                for row in bounds[0] as usize..bounds[1] as usize {
                    let final_row = row + 1 == bounds[1] as usize;
                    if (final_row && req % 2 == 0) || (!final_row && row % 5 == 1) {
                        recurrent_slots[row] = (num_total_reqs + row) as u32;
                        conv_slots[row] = (num_state_slots - row - 1) as u32;
                    }
                }
            }
            recurrent_write_slots.write_typed(0, &recurrent_slots);
            conv_write_slots.write_typed(0, &conv_slots);
            let mut expected_conv_state = vec![CANARY; num_state_slots * conv_stride];
            expected_conv_state[..source_conv_states.len()].copy_from_slice(&source_conv_states);
            let mut expected_recurrent_state = vec![CANARY; num_state_slots * recurrent_stride];
            expected_recurrent_state[..source_recurrent_states.len()].copy_from_slice(&source_recurrent_states);
            let initial_conv_state = expected_conv_state.clone();
            let initial_recurrent_state = expected_recurrent_state.clone();
            write_bf16(&conv_state, 0, &initial_conv_state);
            write_bf16(&recurrent_state, 0, &initial_recurrent_state);
            let mut expected_conv_qkv = vec![CANARY; config.num_qkv_values(shape)];
            let mut expected_output = vec![CANARY; config.num_recurrent_output_values(shape)];
            let mut expected_norm_output = expected_output.clone();
            write_bf16(&conv_qkv, 0, &expected_conv_qkv);
            write_bf16(&recurrent_output, 0, &expected_output);
            write_bf16(&norm_gated_output, 0, &expected_norm_output);
            let arguments = ReplayArguments::new()
                .with_u32(NUM_ACTIVE_REQUESTS, request_lengths.len() as u32)
                .with_u32(NUM_ACTIVE_TOKENS, num_tokens as u32)
                .with_u32(NUM_ACTIVE_RECURRENT_REQUESTS, num_prefill);
            stream.submit_replay_with_arguments(&replay, &arguments).wait();

            for (req, bounds) in cu_values.windows(2).enumerate() {
                let begin = bounds[0] as usize;
                let end = bounds[1] as usize;
                for prefix_end in begin + 1..=end {
                    let core = GDNCore {
                        hidden_dim: config.v_head_dim as usize,
                        qk_head_dim: config.qk_head_dim as usize,
                        v_head_dim: config.v_head_dim as usize,
                        output_gate: match config.output_gate {
                            OutputGate::Silu => GDNOutputGate::Silu,
                            OutputGate::Sigmoid => GDNOutputGate::Sigmoid,
                        },
                        ..fixture_core(fixture_shape(1, (prefix_end - begin) as u32))
                    };
                    let prefix_cu = [0, (prefix_end - begin) as u32];
                    let conv_reference = gdn_short_conv_reference(
                        &core,
                        &prefix_cu,
                        &source_conv_states[req * conv_stride..(req + 1) * conv_stride],
                        &qkv_values[begin * config.qkv_dim() as usize..prefix_end * config.qkv_dim() as usize],
                        &conv_weight_values,
                    );
                    let conv_values = bf16_round_trip(&conv_reference.conv_qkv);
                    let recurrent_reference = gdn_recurrent_reference(
                        &core,
                        GDNRecurrentReferenceInput {
                            cu_tokens: &prefix_cu,
                            source_recurrent_state: &source_recurrent_states
                                [req * recurrent_stride..(req + 1) * recurrent_stride],
                            conv_qkv: &conv_values,
                            a: &a_values[begin..prefix_end],
                            b: &b_values[begin..prefix_end],
                            a_log: &a_log_values,
                            dt_bias: &dt_bias_values,
                        },
                    );
                    let row = prefix_end - 1;
                    if recurrent_slots[row] != u32::MAX && (write_candidate_states || prefix_end == end) {
                        let start = recurrent_slots[row] as usize * recurrent_stride;
                        expected_recurrent_state[start..start + recurrent_stride]
                            .copy_from_slice(&bf16_round_trip(&recurrent_reference.next_recurrent_state));
                    }
                    if conv_slots[row] != u32::MAX && (write_candidate_states || prefix_end == end) {
                        let start = conv_slots[row] as usize * conv_stride;
                        expected_conv_state[start..start + conv_stride]
                            .copy_from_slice(&bf16_round_trip(&conv_reference.next_conv_state));
                    }
                    if prefix_end == end {
                        let output_values = bf16_round_trip(&recurrent_reference.recurrent_output);
                        let norm_values = bf16_round_trip(&gdn_output_norm_gate_reference(
                            &core,
                            &output_values,
                            &z_values[begin * config.v_head_dim as usize..end * config.v_head_dim as usize],
                            &norm_weight_values,
                            config.norm_eps,
                        ));
                        expected_conv_qkv[begin * config.qkv_dim() as usize..end * config.qkv_dim() as usize]
                            .copy_from_slice(&conv_values);
                        expected_output[begin * config.v_head_dim as usize..end * config.v_head_dim as usize]
                            .copy_from_slice(&output_values);
                        expected_norm_output[begin * config.v_head_dim as usize..end * config.v_head_dim as usize]
                            .copy_from_slice(&norm_values);
                    }
                }
            }
            assert_close(
                &read_bf16(&conv_qkv, 0, expected_conv_qkv.len()),
                &bf16_round_trip(&expected_conv_qkv),
                5.0e-3,
            );
            assert_close(
                &read_bf16(&recurrent_output, 0, expected_output.len()),
                &bf16_round_trip(&expected_output),
                5.0e-3,
            );
            assert_close(
                &read_bf16(&norm_gated_output, 0, expected_norm_output.len()),
                &bf16_round_trip(&expected_norm_output),
                5.0e-3,
            );
            assert_close(
                &read_bf16(&conv_state, 0, expected_conv_state.len()),
                &bf16_round_trip(&expected_conv_state),
                5.0e-3,
            );
            assert_close(
                &read_bf16(&recurrent_state, 0, expected_recurrent_state.len()),
                &bf16_round_trip(&expected_recurrent_state),
                5.0e-3,
            );

            let checked_buffers = [
                &conv_qkv,
                &recurrent_output,
                &norm_gated_output,
                &conv_state,
                &recurrent_state,
            ];
            let concurrent_values =
                checked_buffers.map(|buffer| buffer.read_typed::<u16>(0, buffer.len_bytes() / size_of::<u16>()));
            write_bf16(&conv_state, 0, &initial_conv_state);
            write_bf16(&recurrent_state, 0, &initial_recurrent_state);
            for buffer in [&conv_qkv, &recurrent_output, &norm_gated_output] {
                write_bf16(buffer, 0, &vec![CANARY; buffer.len_bytes() / size_of::<u16>()]);
            }
            stream.submit_replay_with_arguments(&serial_replay, &arguments).wait();
            for (buffer, concurrent) in checked_buffers.into_iter().zip(concurrent_values) {
                assert_eq!(buffer.read_typed::<u16>(0, concurrent.len()), concurrent);
            }
        }
    }
}

#[test]
fn test_candidate_state_prefixes() {
    let shape = fixture_shape(1, 3);
    let cu_tokens = vec![0_u32, 3];
    let src_state_slots = vec![0_u32];
    let candidate_dst_slot_ids = vec![1_u32, 2, 3];
    let qkv = fixture_values(fixture_config().num_qkv_values(shape), 0.03125, 29);
    let conv_state = fixture_values(fixture_config().num_conv_state_values(shape), 0.015625, 31);
    let recurrent_state = fixture_values(fixture_config().recurrent_state_stride(), 0.0078125, 37);
    let conv_weight = fixture_values(
        fixture_config().qkv_dim() as usize * fixture_config().conv_kernel_size as usize,
        0.00390625,
        41,
    );
    let a = fixture_values(
        shape.num_total_tokens as usize * fixture_config().num_v_heads as usize,
        0.0625,
        43,
    );
    let b = fixture_values(
        shape.num_total_tokens as usize * fixture_config().num_v_heads as usize,
        0.0625,
        47,
    );
    let z = fixture_values(fixture_config().num_recurrent_output_values(shape), 0.03125, 53);
    let norm_weight = vec![1.0_f32; fixture_config().v_head_dim as usize];
    let a_log = vec![-0.25_f32; fixture_config().num_v_heads as usize];
    let dt_bias = vec![0.125_f32; fixture_config().num_v_heads as usize];

    let actual = run_gdn_core_with_candidate_state_update(
        shape,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &z,
        &norm_weight,
        &a_log,
        &dt_bias,
        &cu_tokens,
        &src_state_slots,
        &candidate_dst_slot_ids,
        5,
    );

    assert_gdn_reference_matches(
        shape,
        &cu_tokens,
        &qkv,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &a,
        &b,
        &a_log,
        &dt_bias,
        &actual.full,
        1.0e-3,
    );
    let core = fixture_core(shape);
    for verified_tokens in 1..=shape.num_total_tokens as usize {
        let prefix_cu_tokens = [0_u32, verified_tokens as u32];
        let conv_reference = gdn_short_conv_reference(
            &core,
            &prefix_cu_tokens,
            &conv_state,
            &qkv[..verified_tokens * fixture_config().qkv_dim() as usize],
            &conv_weight,
        );
        let recurrent_reference = gdn_recurrent_reference(
            &core,
            GDNRecurrentReferenceInput {
                cu_tokens: &prefix_cu_tokens,
                source_recurrent_state: &recurrent_state,
                conv_qkv: &conv_reference.conv_qkv,
                a: &a[..verified_tokens * fixture_config().num_v_heads as usize],
                b: &b[..verified_tokens * fixture_config().num_v_heads as usize],
                a_log: &a_log,
                dt_bias: &dt_bias,
            },
        );
        let slot = candidate_dst_slot_ids[verified_tokens - 1] as usize;
        assert_close(
            conv_state_slot(&actual.next_conv_state_arena, shape, slot),
            &conv_reference.next_conv_state,
            1.0e-3,
        );
        assert_close(
            recurrent_state_slot(&actual.recurrent_state_arena, shape, slot),
            &recurrent_reference.next_recurrent_state,
            1.0e-3,
        );
    }
}

#[test]
fn test_candidate_states_above_u32_byte_offset() {
    let shape = fixture_shape(1, 3);
    let num_state_slots = 5usize;
    let cu_tokens_values = [0_u32, 3];
    let src_state_slot_values = [0_u32];
    let candidate_dst_slot_id_values = [1_u32, 2, 3];
    let qkv_values = fixture_values(fixture_config().num_qkv_values(shape), 0.03125, 29);
    let conv_state_values = fixture_values(fixture_config().num_conv_state_values(shape), 0.015625, 31);
    let recurrent_state_values = fixture_values(fixture_config().recurrent_state_stride(), 0.0078125, 37);
    let conv_weight_values = fixture_values(
        fixture_config().qkv_dim() as usize * fixture_config().conv_kernel_size as usize,
        0.00390625,
        41,
    );
    let a_values = fixture_values(
        shape.num_total_tokens as usize * fixture_config().num_v_heads as usize,
        0.0625,
        43,
    );
    let b_values = fixture_values(
        shape.num_total_tokens as usize * fixture_config().num_v_heads as usize,
        0.0625,
        47,
    );
    let z_values = fixture_values(fixture_config().num_recurrent_output_values(shape), 0.03125, 53);
    let norm_weight_values = vec![1.0_f32; fixture_config().v_head_dim as usize];
    let a_log_values = vec![-0.25_f32; fixture_config().num_v_heads as usize];
    let dt_bias_values = vec![0.125_f32; fixture_config().num_v_heads as usize];

    let device = Device::system_default();
    let stream = Stream::new(&device);
    let kernels = Compute::new(&device, fixture_config());
    let qkv = bf16_buffer(&device, &qkv_values);
    let a = bf16_buffer(&device, &a_values);
    let b = bf16_buffer(&device, &b_values);
    let z = bf16_buffer(&device, &z_values);
    let conv_weight = bf16_buffer(&device, &conv_weight_values);
    let norm_weight = bf16_buffer(&device, &norm_weight_values);
    let a_log = bf16_buffer(&device, &a_log_values);
    let dt_bias = bf16_buffer(&device, &dt_bias_values);
    let cu_tokens = Buffer::from_slice(&device, &cu_tokens_values);
    let src_state_slots = Buffer::from_slice(&device, &src_state_slot_values);
    let candidate_dst_slot_ids = Buffer::from_slice(&device, &candidate_dst_slot_id_values);

    let high_base = u32::MAX as u64 + 1;
    let conv_state_offset_bytes = high_base;
    let next_conv_state_offset_bytes = high_base + 4096;
    let recurrent_state_offset_bytes = high_base + 8192;
    let recurrent_arena_bytes = (num_state_slots as u64)
        .checked_mul(
            fixture_config()
                .recurrent_state_stride()
                .checked_mul(size_of::<u16>())
                .expect("test recurrent-state byte stride must fit usize") as u64,
        )
        .expect("test recurrent-state arena byte length must fit u64");
    let state_arena = Buffer::new_uninit(
        &device,
        recurrent_state_offset_bytes
            .checked_add(recurrent_arena_bytes)
            .expect("test state arena byte length must fit u64"),
    );
    write_bf16(
        &state_arena,
        usize::try_from(conv_state_offset_bytes / size_of::<u16>() as u64)
            .expect("test convolution state offset must fit usize"),
        &conv_state_values,
    );
    write_bf16(
        &state_arena,
        usize::try_from(recurrent_state_offset_bytes / size_of::<u16>() as u64)
            .expect("test recurrent state offset must fit usize"),
        &recurrent_state_values,
    );

    let conv_qkv = Buffer::new_zeroed_elements(&device, fixture_config().num_qkv_values(shape), Dtype::Bfloat16);
    let recurrent_output = Buffer::new_zeroed_elements(
        &device,
        fixture_config().num_recurrent_output_values(shape),
        Dtype::Bfloat16,
    );
    let norm_gated_output = Buffer::new_zeroed_elements(
        &device,
        fixture_config().num_recurrent_output_values(shape),
        Dtype::Bfloat16,
    );
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_with_candidate_state_update(
        shape,
        Buffers {
            qkv: &qkv,
            a: &a,
            b: &b,
            z: &z,
            conv_weight: &conv_weight,
            norm_weight: &norm_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            cu_tokens: &cu_tokens,
            src_recurrent_state_slots: &src_state_slots,
            src_conv_state_slots: &src_state_slots,
            flat_recurrent_state_write_slots: &candidate_dst_slot_ids,
            flat_conv_state_write_slots: &candidate_dst_slot_ids,
            conv_state: &state_arena,
            conv_state_offset_bytes,
            next_conv_state: &state_arena,
            next_conv_state_offset_bytes,
            recurrent_state_arena: &state_arena,
            recurrent_state_arena_offset_bytes: recurrent_state_offset_bytes,
            conv_qkv: &conv_qkv,
            recurrent_output: &recurrent_output,
            norm_gated_output: &norm_gated_output,
        },
        ReplayU32::Fixed(shape.num_total_reqs),
        ReplayU32::Fixed(shape.num_total_tokens),
    ));
    stream.submit_replay(&builder.build()).wait();

    let next_conv_state_arena = read_bf16(
        &state_arena,
        usize::try_from(next_conv_state_offset_bytes / size_of::<u16>() as u64)
            .expect("test next convolution state offset must fit usize"),
        num_state_slots * fixture_config().qkv_dim() as usize * fixture_config().conv_state_len() as usize,
    );
    let recurrent_state_arena = read_bf16(
        &state_arena,
        usize::try_from(recurrent_state_offset_bytes / size_of::<u16>() as u64)
            .expect("test recurrent state offset must fit usize"),
        num_state_slots * fixture_config().recurrent_state_stride(),
    );
    let core = fixture_core(shape);
    for verified_tokens in 1..=shape.num_total_tokens as usize {
        let prefix_cu_tokens = [0_u32, verified_tokens as u32];
        let conv_reference = gdn_short_conv_reference(
            &core,
            &prefix_cu_tokens,
            &conv_state_values,
            &qkv_values[..verified_tokens * fixture_config().qkv_dim() as usize],
            &conv_weight_values,
        );
        let recurrent_reference = gdn_recurrent_reference(
            &core,
            GDNRecurrentReferenceInput {
                cu_tokens: &prefix_cu_tokens,
                source_recurrent_state: &recurrent_state_values,
                conv_qkv: &conv_reference.conv_qkv,
                a: &a_values[..verified_tokens * fixture_config().num_v_heads as usize],
                b: &b_values[..verified_tokens * fixture_config().num_v_heads as usize],
                a_log: &a_log_values,
                dt_bias: &dt_bias_values,
            },
        );
        let slot = candidate_dst_slot_id_values[verified_tokens - 1] as usize;
        assert_close(
            conv_state_slot(&next_conv_state_arena, shape, slot),
            &conv_reference.next_conv_state,
            1.0e-3,
        );
        assert_close(
            recurrent_state_slot(&recurrent_state_arena, shape, slot),
            &recurrent_reference.next_recurrent_state,
            1.0e-3,
        );
    }
}

struct GDNCoreOutputs {
    conv_qkv: Vec<f32>,
    next_conv_state: Vec<f32>,
    recurrent_output: Vec<f32>,
    recurrent_state: Vec<f32>,
}

struct GDNCoreWithCandidateStateOutputs {
    full: GDNCoreOutputs,
    next_conv_state_arena: Vec<f32>,
    recurrent_state_arena: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn run_gdn_core(
    shape: Shape,
    qkv_values: &[f32],
    conv_state_values: &[f32],
    recurrent_state_values: &[f32],
    conv_weight_values: &[f32],
    a_values: &[f32],
    b_values: &[f32],
    z_values: &[f32],
    norm_weight_values: &[f32],
    a_log_values: &[f32],
    dt_bias_values: &[f32],
    cu_tokens_values: &[u32],
    src_state_slot_values: &[u32],
    dst_slot_id_values: &[u32],
) -> GDNCoreOutputs {
    const STATE_PREFIX_VALUES: usize = 7;
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let kernels = Compute::new(&device, fixture_config());
    let qkv = bf16_buffer(&device, qkv_values);
    let a = bf16_buffer(&device, a_values);
    let b = bf16_buffer(&device, b_values);
    let z = bf16_buffer(&device, z_values);
    let conv_weight = bf16_buffer(&device, conv_weight_values);
    let norm_weight = bf16_buffer(&device, norm_weight_values);
    let a_log = bf16_buffer(&device, a_log_values);
    let dt_bias = bf16_buffer(&device, dt_bias_values);
    let cu_tokens = Buffer::from_slice(&device, cu_tokens_values);
    let src_state_slots = Buffer::from_slice(&device, src_state_slot_values);
    let mut flat_candidate_state_slot_values = vec![u32::MAX; shape.num_total_tokens as usize];
    for (req_index, &dst_state_slot) in dst_slot_id_values.iter().enumerate() {
        let flat_token_end = cu_tokens_values[req_index + 1] as usize;
        flat_candidate_state_slot_values[flat_token_end - 1] = dst_state_slot;
    }
    let flat_materialized_state_slots = Buffer::from_slice(&device, &flat_candidate_state_slot_values);
    let mut conv_state_values_with_prefix = vec![-1.0; STATE_PREFIX_VALUES];
    conv_state_values_with_prefix.extend_from_slice(conv_state_values);
    let conv_state = bf16_buffer(&device, &conv_state_values_with_prefix);
    let next_conv_state = Buffer::new_zeroed(
        &device,
        (STATE_PREFIX_VALUES + fixture_config().num_conv_state_values(shape)) * size_of::<u16>(),
    );
    let mut recurrent_state_values_with_prefix = vec![-1.0; STATE_PREFIX_VALUES];
    recurrent_state_values_with_prefix.extend_from_slice(recurrent_state_values);
    let recurrent_state_arena = bf16_buffer(&device, &recurrent_state_values_with_prefix);
    let state_offset_bytes = (STATE_PREFIX_VALUES * size_of::<u16>()) as u64;
    let conv_qkv = Buffer::new_zeroed(&device, fixture_config().num_qkv_values(shape) * size_of::<u16>());
    let recurrent_output = Buffer::new_zeroed(
        &device,
        fixture_config().num_recurrent_output_values(shape) * size_of::<u16>(),
    );
    let norm_gated_output = Buffer::new_zeroed(
        &device,
        fixture_config().num_recurrent_output_values(shape) * size_of::<u16>(),
    );
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke(
        shape,
        Buffers {
            qkv: &qkv,
            a: &a,
            b: &b,
            z: &z,
            conv_weight: &conv_weight,
            norm_weight: &norm_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            cu_tokens: &cu_tokens,
            src_recurrent_state_slots: &src_state_slots,
            src_conv_state_slots: &src_state_slots,
            flat_recurrent_state_write_slots: &flat_materialized_state_slots,
            flat_conv_state_write_slots: &flat_materialized_state_slots,
            conv_state: &conv_state,
            conv_state_offset_bytes: state_offset_bytes,
            next_conv_state: &next_conv_state,
            next_conv_state_offset_bytes: state_offset_bytes,
            recurrent_state_arena: &recurrent_state_arena,
            recurrent_state_arena_offset_bytes: state_offset_bytes,
            conv_qkv: &conv_qkv,
            recurrent_output: &recurrent_output,
            norm_gated_output: &norm_gated_output,
        },
        ReplayU32::Fixed(shape.num_total_reqs),
        ReplayU32::Fixed(shape.num_total_tokens),
    ));
    let replay = builder.build();
    stream.submit_replay(&replay).wait();

    GDNCoreOutputs {
        conv_qkv: read_bf16(&conv_qkv, 0, fixture_config().num_qkv_values(shape)),
        next_conv_state: read_bf16(
            &next_conv_state,
            STATE_PREFIX_VALUES,
            fixture_config().num_conv_state_values(shape),
        ),
        recurrent_output: read_bf16(
            &recurrent_output,
            0,
            fixture_config().num_recurrent_output_values(shape),
        ),
        recurrent_state: read_bf16(
            &recurrent_state_arena,
            STATE_PREFIX_VALUES,
            shape.num_total_reqs as usize * fixture_config().recurrent_state_stride(),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_gdn_core_with_candidate_state_update(
    shape: Shape,
    qkv_values: &[f32],
    conv_state_values: &[f32],
    recurrent_state_values: &[f32],
    conv_weight_values: &[f32],
    a_values: &[f32],
    b_values: &[f32],
    z_values: &[f32],
    norm_weight_values: &[f32],
    a_log_values: &[f32],
    dt_bias_values: &[f32],
    cu_tokens_values: &[u32],
    src_state_slot_values: &[u32],
    candidate_dst_slot_id_values: &[u32],
    num_state_slots: usize,
) -> GDNCoreWithCandidateStateOutputs {
    const STATE_PREFIX_VALUES: usize = 7;
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let kernels = Compute::new(&device, fixture_config());
    let qkv = bf16_buffer(&device, qkv_values);
    let a = bf16_buffer(&device, a_values);
    let b = bf16_buffer(&device, b_values);
    let z = bf16_buffer(&device, z_values);
    let conv_weight = bf16_buffer(&device, conv_weight_values);
    let norm_weight = bf16_buffer(&device, norm_weight_values);
    let a_log = bf16_buffer(&device, a_log_values);
    let dt_bias = bf16_buffer(&device, dt_bias_values);
    let cu_tokens = Buffer::from_slice(&device, cu_tokens_values);
    let src_state_slots = Buffer::from_slice(&device, src_state_slot_values);
    let candidate_dst_slot_ids = Buffer::from_slice(&device, candidate_dst_slot_id_values);
    let mut conv_state_values_with_prefix = vec![-1.0; STATE_PREFIX_VALUES];
    conv_state_values_with_prefix.extend_from_slice(conv_state_values);
    let conv_state = bf16_buffer(&device, &conv_state_values_with_prefix);
    let next_conv_state = Buffer::new_zeroed(
        &device,
        (STATE_PREFIX_VALUES
            + num_state_slots * fixture_config().qkv_dim() as usize * fixture_config().conv_state_len() as usize)
            * size_of::<u16>(),
    );
    let mut recurrent_state_arena_values =
        vec![0.0_f32; STATE_PREFIX_VALUES + num_state_slots * fixture_config().recurrent_state_stride()];
    recurrent_state_arena_values[STATE_PREFIX_VALUES..STATE_PREFIX_VALUES + fixture_config().recurrent_state_stride()]
        .copy_from_slice(recurrent_state_values);
    let recurrent_state_arena = bf16_buffer(&device, &recurrent_state_arena_values);
    let state_offset_bytes = (STATE_PREFIX_VALUES * size_of::<u16>()) as u64;
    let conv_qkv = Buffer::new_zeroed(&device, fixture_config().num_qkv_values(shape) * size_of::<u16>());
    let recurrent_output = Buffer::new_zeroed(
        &device,
        fixture_config().num_recurrent_output_values(shape) * size_of::<u16>(),
    );
    let norm_gated_output = Buffer::new_zeroed(
        &device,
        fixture_config().num_recurrent_output_values(shape) * size_of::<u16>(),
    );

    let mut builder = stream.create_replay_program();
    let core = Buffers {
        qkv: &qkv,
        a: &a,
        b: &b,
        z: &z,
        conv_weight: &conv_weight,
        norm_weight: &norm_weight,
        a_log: &a_log,
        dt_bias: &dt_bias,
        cu_tokens: &cu_tokens,
        src_recurrent_state_slots: &src_state_slots,
        src_conv_state_slots: &src_state_slots,
        flat_recurrent_state_write_slots: &candidate_dst_slot_ids,
        flat_conv_state_write_slots: &candidate_dst_slot_ids,
        conv_state: &conv_state,
        conv_state_offset_bytes: state_offset_bytes,
        next_conv_state: &next_conv_state,
        next_conv_state_offset_bytes: state_offset_bytes,
        recurrent_state_arena: &recurrent_state_arena,
        recurrent_state_arena_offset_bytes: state_offset_bytes,
        conv_qkv: &conv_qkv,
        recurrent_output: &recurrent_output,
        norm_gated_output: &norm_gated_output,
    };
    builder.record(kernels.invoke_with_candidate_state_update(
        shape,
        core,
        ReplayU32::Fixed(shape.num_total_reqs),
        ReplayU32::Fixed(shape.num_total_tokens),
    ));
    let replay = builder.build();
    stream.submit_replay(&replay).wait();

    GDNCoreWithCandidateStateOutputs {
        full: GDNCoreOutputs {
            conv_qkv: read_bf16(&conv_qkv, 0, fixture_config().num_qkv_values(shape)),
            next_conv_state: read_bf16(
                &next_conv_state,
                STATE_PREFIX_VALUES
                    + candidate_dst_slot_id_values[candidate_dst_slot_id_values.len() - 1] as usize
                        * fixture_config().qkv_dim() as usize
                        * fixture_config().conv_state_len() as usize,
                fixture_config().num_conv_state_values(shape),
            ),
            recurrent_output: read_bf16(
                &recurrent_output,
                0,
                fixture_config().num_recurrent_output_values(shape),
            ),
            recurrent_state: read_bf16(
                &recurrent_state_arena,
                STATE_PREFIX_VALUES
                    + candidate_dst_slot_id_values[candidate_dst_slot_id_values.len() - 1] as usize
                        * fixture_config().recurrent_state_stride(),
                fixture_config().recurrent_state_stride(),
            ),
        },
        next_conv_state_arena: read_bf16(
            &next_conv_state,
            STATE_PREFIX_VALUES,
            num_state_slots * fixture_config().qkv_dim() as usize * fixture_config().conv_state_len() as usize,
        ),
        recurrent_state_arena: read_bf16(
            &recurrent_state_arena,
            STATE_PREFIX_VALUES,
            num_state_slots * fixture_config().recurrent_state_stride(),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn assert_gdn_reference_matches(
    shape: Shape,
    cu_tokens: &[u32],
    qkv: &[f32],
    conv_state: &[f32],
    recurrent_state: &[f32],
    conv_weight: &[f32],
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    actual: &GDNCoreOutputs,
    tolerance: f32,
) {
    let core = fixture_core(shape);
    let quantized_qkv = bf16_round_trip(qkv);
    let quantized_conv_state = bf16_round_trip(conv_state);
    let quantized_recurrent_state = bf16_round_trip(recurrent_state);
    let quantized_conv_weight = bf16_round_trip(conv_weight);
    let quantized_a = bf16_round_trip(a);
    let quantized_b = bf16_round_trip(b);
    let quantized_a_log = bf16_round_trip(a_log);
    let quantized_dt_bias = bf16_round_trip(dt_bias);
    let conv_reference = gdn_short_conv_reference(
        &core,
        cu_tokens,
        &quantized_conv_state,
        &quantized_qkv,
        &quantized_conv_weight,
    );
    let quantized_conv_qkv = bf16_round_trip(&conv_reference.conv_qkv);
    let recurrent_reference = gdn_recurrent_reference(
        &core,
        GDNRecurrentReferenceInput {
            cu_tokens,
            source_recurrent_state: &quantized_recurrent_state,
            conv_qkv: &quantized_conv_qkv,
            a: &quantized_a,
            b: &quantized_b,
            a_log: &quantized_a_log,
            dt_bias: &quantized_dt_bias,
        },
    );

    assert_close(&actual.conv_qkv, &quantized_conv_qkv, tolerance);
    assert_close(
        &actual.next_conv_state,
        &bf16_round_trip(&conv_reference.next_conv_state),
        tolerance,
    );
    assert_close(
        &actual.recurrent_output,
        &bf16_round_trip(&recurrent_reference.recurrent_output),
        tolerance,
    );
    assert_close(
        &actual.recurrent_state,
        &bf16_round_trip(&recurrent_reference.next_recurrent_state),
        tolerance,
    );
}

fn fixture_shape(num_reqs: u32, num_tokens: u32) -> Shape {
    Shape {
        num_total_reqs: num_reqs,
        num_total_tokens: num_tokens,
    }
}

fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    Buffer::from_slice(
        device,
        &values
            .iter()
            .map(|&value| bf16::from_f32(value).to_bits())
            .collect::<Vec<_>>(),
    )
}

fn read_bf16(buffer: &Buffer, element_offset: usize, count: usize) -> Vec<f32> {
    buffer
        .read_typed::<u16>(element_offset, count)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect()
}

fn write_bf16(buffer: &Buffer, element_offset: usize, values: &[f32]) {
    buffer.write_typed(
        element_offset,
        &values
            .iter()
            .map(|&value| bf16::from_f32(value).to_bits())
            .collect::<Vec<_>>(),
    );
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

fn fixture_config() -> Config {
    Config {
        projection_layout: super::ProjectionLayout::Separate,
        num_qk_heads: 1,
        qk_head_dim: 32,
        num_v_heads: 1,
        v_head_dim: 8,
        conv_kernel_size: 3,
        q_scale: 1.0,
        norm_eps: 1.0e-6,
        output_gate: OutputGate::Silu,
    }
}

fn fixture_core(_shape: Shape) -> GDNCore {
    GDNCore {
        model_layer_index: 0,
        hidden_dim: fixture_config().num_v_heads as usize * fixture_config().v_head_dim as usize,
        num_qk_heads: fixture_config().num_qk_heads as usize,
        qk_head_dim: fixture_config().qk_head_dim as usize,
        num_v_heads: fixture_config().num_v_heads as usize,
        v_head_dim: fixture_config().v_head_dim as usize,
        conv_kernel_size: fixture_config().conv_kernel_size as usize,
        q_scale: 1.0,
        output_gate: GDNOutputGate::Silu,
    }
}

fn conv_state_slot(arena: &[f32], _shape: Shape, state_slot: usize) -> &[f32] {
    let conv_state_stride = fixture_config().qkv_dim() as usize * fixture_config().conv_state_len() as usize;
    &arena[state_slot * conv_state_stride..(state_slot + 1) * conv_state_stride]
}

fn recurrent_state_slot(arena: &[f32], _shape: Shape, state_slot: usize) -> &[f32] {
    let recurrent_state_stride = fixture_config().recurrent_state_stride();
    &arena[state_slot * recurrent_state_stride..(state_slot + 1) * recurrent_state_stride]
}

fn fixture_values(count: usize, scale: f32, pattern_offset: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index * 17 + pattern_offset) % 29) as f32 * scale - 14.0 * scale)
        .collect()
}

fn generated_values(count: usize, random_seed: u32) -> Vec<f32> {
    let mut state = random_seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    let mut max_error = 0.0_f32;
    let mut max_error_index = 0;
    let mut error_sum = 0.0_f32;
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
        let error = (actual_value - expected_value).abs();
        error_sum += error;
        if error > max_error {
            max_error = error;
            max_error_index = index;
        }
    }
    let mean_error = error_sum / actual.len().max(1) as f32;
    let mean_tolerance = tolerance * 0.1;
    assert!(
        max_error <= tolerance && mean_error <= mean_tolerance,
        "GDN reference mismatch: index={max_error_index} expected={} actual={} max_abs_error={max_error} \
         mean_abs_error={mean_error} max_abs_tolerance={tolerance} mean_abs_tolerance={mean_tolerance}",
        expected[max_error_index],
        actual[max_error_index],
    );
}

#[path = "compute_replay_test.rs"]
mod replay;
