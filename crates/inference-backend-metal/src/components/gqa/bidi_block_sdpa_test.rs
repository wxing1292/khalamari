use std::mem::size_of;

use half::bf16;

use super::*;
use crate::components::gqa::fp8::bf16_to_f32;
use crate::components::gqa::fp8::bf16_to_fp8_e4m3;
use crate::components::gqa::fp8::f32_to_bf16;
use crate::components::gqa::fp8::fp8_e4m3_to_bf16;
use crate::components::gqa::kv_page_write::PageTableLayout;
use crate::components::gqa::sdpa;
use crate::components::gqa::sdpa::sliced_q;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayU32;
use crate::metal::Stream;

const NUM_ACTIVE_Q_TOKEN_RANGES: ReplayParameterKey =
    ReplayParameterKey::new("test.gqa_bidi_block_sdpa.num_active_q_token_ranges");

#[test]
#[should_panic(expected = "complete request blocks")]
fn test_block_shape_rejects_partial_request_block() {
    Shape {
        num_total_tokens: 10,
        num_total_q_token_ranges: 2,
        num_total_partial_output_slots: 16,
    }
    .validate(Config {
        block_size: 9,
        max_q_tokens: 8,
        num_q_heads: 40,
        num_kv_heads: 8,
        head_dim: 128,
        scale: 1.0,
        dtype: Dtype::Bfloat16,
    });
}

#[test]
fn test_f32_kernel_matches_request_block_bidirectional_reference() {
    assert_kernel_matches_request_block_bidirectional_reference(Dtype::Float32, 1.0e-5);
}

#[test]
fn test_bf16_kernel_matches_request_block_bidirectional_reference() {
    assert_kernel_matches_request_block_bidirectional_reference(Dtype::Bfloat16, 1.0e-2);
}

#[test]
fn test_replay_active_q_token_ranges_match_cpu_reference() {
    const NUM_TOTAL_TOKENS: usize = 8;
    const HEAD_DIM: usize = 32;
    const ACTIVE_COUNTS: [u32; 8] = [1, 8, 3, 7, 2, 6, 4, 5];

    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config {
        block_size: NUM_TOTAL_TOKENS as u32,
        max_q_tokens: 1,
        num_q_heads: 1,
        num_kv_heads: 1,
        head_dim: HEAD_DIM as u32,
        scale: (HEAD_DIM as f32).sqrt().recip(),
        dtype: Dtype::Float32,
    };
    let shape = Shape {
        num_total_tokens: NUM_TOTAL_TOKENS as u32,
        num_total_q_token_ranges: NUM_TOTAL_TOKENS as u32,
        num_total_partial_output_slots: NUM_TOTAL_TOKENS as u32,
    };
    let q_values = (0..NUM_TOTAL_TOKENS * HEAD_DIM)
        .map(|index| index as f32 * 0.003 - 0.4)
        .collect::<Vec<_>>();
    let k_values = (0..NUM_TOTAL_TOKENS * HEAD_DIM)
        .map(|index| index as f32 * -0.002 + 0.3)
        .collect::<Vec<_>>();
    let v_values = (0..NUM_TOTAL_TOKENS * HEAD_DIM)
        .map(|index| index as f32 * 0.004 - 0.2)
        .collect::<Vec<_>>();
    let q = Buffer::from_slice(&device, &q_values);
    let local_k = Buffer::from_slice(&device, &k_values);
    let local_v = Buffer::from_slice(&device, &v_values);
    let q_token_ranges = Buffer::from_slice(
        &device,
        &(0..NUM_TOTAL_TOKENS as u32)
            .flat_map(|index| [index, index + 1])
            .collect::<Vec<_>>(),
    );
    let cu_sdpa_partial_outputs = Buffer::from_slice(&device, &(0..=NUM_TOTAL_TOKENS as u32).collect::<Vec<_>>());
    let partial_exp_sums = Buffer::new_zeroed_elements(&device, NUM_TOTAL_TOKENS, Dtype::Float32);
    let partial_max_logits = Buffer::new_zeroed_elements(&device, NUM_TOTAL_TOKENS, Dtype::Float32);
    let partial_output = Buffer::new_zeroed_elements(&device, NUM_TOTAL_TOKENS * HEAD_DIM, Dtype::Float32);
    let kernel = Compute::new(&device, config);
    let mut builder = stream.create_replay_program();
    builder.record(kernel.invoke(
        shape,
        ReplayU32::Parameter(NUM_ACTIVE_Q_TOKEN_RANGES),
        Buffers {
            q: &q,
            local_k: &local_k,
            local_v: &local_v,
            q_token_ranges: &q_token_ranges,
            cu_sdpa_partial_outputs: &cu_sdpa_partial_outputs,
            partial_exp_sums: &partial_exp_sums,
            partial_max_logits: &partial_max_logits,
            partial_output: &partial_output,
        },
    ));
    let replay = builder.build();

    for num_active_ranges in ACTIVE_COUNTS {
        let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_Q_TOKEN_RANGES, num_active_ranges);
        stream.submit_replay_with_arguments(&replay, &arguments).wait();
        let actual_exp_sums = partial_exp_sums.read_typed::<f32>(0, num_active_ranges as usize);
        let actual_max_logits = partial_max_logits.read_typed::<f32>(0, num_active_ranges as usize);
        let actual_output = partial_output.read_typed::<f32>(0, num_active_ranges as usize * HEAD_DIM);
        for q_token_index in 0..num_active_ranges as usize {
            let q_row = &q_values[q_token_index * HEAD_DIM..(q_token_index + 1) * HEAD_DIM];
            let scores = (0..NUM_TOTAL_TOKENS)
                .map(|kv_token_index| {
                    let key = &k_values[kv_token_index * HEAD_DIM..(kv_token_index + 1) * HEAD_DIM];
                    dot(q_row, key) * config.scale
                })
                .collect::<Vec<_>>();
            let max_logit = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores
                .iter()
                .map(|score| (*score - max_logit).exp())
                .collect::<Vec<_>>();
            let exp_sum = weights.iter().sum::<f32>();
            assert!((actual_max_logits[q_token_index] - max_logit).abs() < 1.0e-4);
            assert!((actual_exp_sums[q_token_index] - exp_sum).abs() < 1.0e-4);
            for dim in 0..HEAD_DIM {
                let expected = weights
                    .iter()
                    .enumerate()
                    .map(|(kv_token_index, weight)| weight * v_values[kv_token_index * HEAD_DIM + dim])
                    .sum::<f32>()
                    / exp_sum;
                let actual = actual_output[q_token_index * HEAD_DIM + dim];
                assert!(
                    (actual - expected).abs() < 1.0e-4,
                    "actual={actual} expected={expected}"
                );
            }
        }
    }
}

#[test]
fn test_full_history_composite() {
    assert_composite_matches_reference(&[0; 7]);
}

#[test]
fn test_windowed_history_composite() {
    // Use a scaled-down left window to exercise a different lower bound for
    // each Q token while the local block remains inside the window.
    assert_composite_matches_reference(&[4, 5, 6, 7, 8, 9, 10, 11]);
}

fn assert_composite_matches_reference(visible_kv_token_begins: &[u32]) {
    let num_q_tokens = visible_kv_token_begins.len();
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let num_history_tokens = 16usize;
    let num_q_heads = 5usize;
    let num_kv_heads = 1usize;
    let head_dim = 128usize;
    let tokens_per_page = 8usize;
    let page_bytes = 2 * num_kv_heads * tokens_per_page * head_dim * size_of::<u8>();
    let scale = (head_dim as f32).sqrt().recip();
    let sliced_q_config = sliced_q::Config {
        num_q_heads: num_q_heads as u32,
        num_kv_heads: num_kv_heads as u32,
        head_dim: head_dim as u32,
        scale,
        page_bytes: page_bytes as u32,
        dtype: Dtype::Bfloat16,
        page_table_layout: PageTableLayout {
            num_req_slots: 1,
            num_gqa_layers: 1,
            num_blocks: 2,
            num_page_ids_per_block: 1,
        },
    };
    let sdpa_config = sdpa::Config {
        io_dtype: Dtype::Bfloat16,
        num_q_heads: num_q_heads as u32,
        num_kv_heads: num_kv_heads as u32,
        head_dim: head_dim as u32,
        tokens_per_page: tokens_per_page as u32,
    };
    let execution = sdpa::ExecutionVariant::sliced_q_per_head(sdpa_config, 8, 16, 5);
    let sliced_q_shape = sliced_q::Shape {
        num_total_tokens: num_q_tokens as u32,
        num_total_q_token_tiles: 1,
        num_total_sdpa_map_task_templates: 3,
    };
    let block_config = Config {
        block_size: num_q_tokens as u32,
        max_q_tokens: 8,
        num_q_heads: num_q_heads as u32,
        num_kv_heads: num_kv_heads as u32,
        head_dim: head_dim as u32,
        scale,
        dtype: Dtype::Bfloat16,
    };
    let block_shape = Shape {
        num_total_tokens: num_q_tokens as u32,
        num_total_q_token_ranges: 1,
        num_total_partial_output_slots: 3,
    };

    // Keep the history and block maxima close but distinct. This makes the
    // test sensitive to the log base in the shared partial-state ABI.
    let q_values = vec![1.0; num_q_tokens * num_q_heads * head_dim];
    let history_k =
        vec![bf16_to_f32(fp8_e4m3_to_bf16(bf16_to_fp8_e4m3(f32_to_bf16(0.176_776_69)))); num_history_tokens * head_dim];
    let history_v =
        vec![bf16_to_f32(fp8_e4m3_to_bf16(bf16_to_fp8_e4m3(f32_to_bf16(1.0)))); num_history_tokens * head_dim];
    let local_k_values = vec![0.220_970_87; num_q_tokens * head_dim];
    let local_v_values = vec![-1.0; num_q_tokens * head_dim];
    let q = buffer(&device, &q_values, Dtype::Bfloat16);
    let local_k = buffer(&device, &local_k_values, Dtype::Bfloat16);
    let local_v = buffer(&device, &local_v_values, Dtype::Bfloat16);
    let mut page_values = Vec::new();
    for page_index in 0..2 {
        let begin = page_index * tokens_per_page * head_dim;
        let end = begin + tokens_per_page * head_dim;
        page_values.extend(
            history_k[begin..end]
                .iter()
                .copied()
                .map(|value| bf16_to_fp8_e4m3(f32_to_bf16(value))),
        );
        page_values.extend(
            history_v[begin..end]
                .iter()
                .copied()
                .map(|value| bf16_to_fp8_e4m3(f32_to_bf16(value))),
        );
    }
    let kv_pages = Buffer::from_slice(&device, &page_values);
    let req_slots = Buffer::from_slice(&device, &vec![0_u32; num_q_tokens]);
    let page_ids = Buffer::from_slice(&device, &[0_u32, 1]);
    let visible_kv_token_ranges = Buffer::from_slice(
        &device,
        &visible_kv_token_begins
            .iter()
            .flat_map(|&begin| [begin, 12])
            .collect::<Vec<_>>(),
    );
    let q_token_ranges = Buffer::from_slice(&device, &[0_u32, num_q_tokens as u32]);
    let map_task_templates = Buffer::from_slice(&device, &[0_u32, 0, 4, 0, 4, 12, u32::MAX, u32::MAX, u32::MAX]);
    let cu_partial_outputs = Buffer::from_slice(&device, &[0_u32, 3]);
    let partial_state_groups = 3 * 8 * num_q_heads;
    let partial_exp_sums = Buffer::new_zeroed_elements(&device, partial_state_groups, Dtype::Float32);
    let partial_max_logits = Buffer::new_zeroed_elements(&device, partial_state_groups, Dtype::Float32);
    let partial_output = Buffer::new_zeroed_elements(&device, partial_state_groups * head_dim, Dtype::Bfloat16);
    let output = Buffer::new_zeroed_elements(&device, num_q_tokens * num_q_heads * head_dim, Dtype::Bfloat16);

    let sliced_q = sliced_q::Compute::new(&device, sliced_q_config, execution, sliced_q_shape);
    let block = Compute::new(&device, block_config);
    let mut builder = stream.create_replay_program();
    builder.record(sliced_q.invoke_map(
        sliced_q::MapBuffers {
            q: &q,
            kv_pages: &kv_pages,
            req_slots: &req_slots,
            page_ids: &page_ids,
            visible_kv_token_ranges: &visible_kv_token_ranges,
            q_token_ranges: &q_token_ranges,
            sdpa_map_task_templates: &map_task_templates,
            partial_output: &partial_output,
            partial_exp_sums: &partial_exp_sums,
            partial_max_logits: &partial_max_logits,
        },
        ReplayU32::Fixed(0),
        ReplayU32::Fixed(num_q_tokens as u32),
        ReplayU32::Fixed(1),
        ReplayU32::Fixed(3),
    ));
    builder.record(block.invoke(
        block_shape,
        ReplayU32::Fixed(block_shape.num_total_q_token_ranges),
        Buffers {
            q: &q,
            local_k: &local_k,
            local_v: &local_v,
            q_token_ranges: &q_token_ranges,
            cu_sdpa_partial_outputs: &cu_partial_outputs,
            partial_exp_sums: &partial_exp_sums,
            partial_max_logits: &partial_max_logits,
            partial_output: &partial_output,
        },
    ));
    builder.record_with_barrier_before(sliced_q.invoke_reduce(
        sliced_q::ReduceBuffers {
            partial_output: &partial_output,
            partial_exp_sums: &partial_exp_sums,
            partial_max_logits: &partial_max_logits,
            q_token_ranges: &q_token_ranges,
            cu_sdpa_partial_outputs: &cu_partial_outputs,
            output: &output,
        },
        ReplayU32::Fixed(1),
    ));
    stream.submit_replay(&builder.build()).wait();

    let actual = read_values(&output, num_q_tokens * num_q_heads * head_dim, Dtype::Bfloat16);
    for (q_token_index, &visible_kv_token_begin) in visible_kv_token_begins.iter().enumerate() {
        let history_begin = visible_kv_token_begin as usize;
        for q_head_index in 0..num_q_heads {
            let q_begin = (q_token_index * num_q_heads + q_head_index) * head_dim;
            let query = &q_values[q_begin..q_begin + head_dim];
            let mut scores = Vec::new();
            let mut values = Vec::new();
            for history_token_index in history_begin..12 {
                let begin = history_token_index * head_dim;
                scores.push(dot(query, &history_k[begin..begin + head_dim]) * scale);
                values.push(&history_v[begin..begin + head_dim]);
            }
            for local_token_index in 0..num_q_tokens {
                let begin = local_token_index * head_dim;
                scores.push(dot(query, &local_k_values[begin..begin + head_dim]) * scale);
                values.push(&local_v_values[begin..begin + head_dim]);
            }
            let max_logit = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores
                .iter()
                .map(|score| (*score - max_logit).exp())
                .collect::<Vec<_>>();
            let weight_sum = weights.iter().sum::<f32>();
            for dim in 0..head_dim {
                let expected = weights
                    .iter()
                    .zip(&values)
                    .map(|(&weight, value)| weight * value[dim])
                    .sum::<f32>()
                    / weight_sum;
                let actual_value = actual[q_begin + dim];
                assert!(
                    (actual_value - expected).abs() < 3.0e-2,
                    "q_token={q_token_index} q_head={q_head_index} dim={dim} actual={actual_value} expected={expected}"
                );
            }
        }
    }
}

fn assert_kernel_matches_request_block_bidirectional_reference(dtype: Dtype, output_tolerance: f32) {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config {
        block_size: 3,
        max_q_tokens: 8,
        num_q_heads: 5,
        num_kv_heads: 1,
        head_dim: 32,
        scale: 32.0_f32.sqrt().recip(),
        dtype,
    };
    let shape = Shape {
        num_total_tokens: 6,
        num_total_q_token_ranges: 2,
        num_total_partial_output_slots: 8,
    };
    let q_values = round_for_dtype(
        (0..config.q_elements(shape))
            .map(|index| index as f32 * 0.03125 - 0.4)
            .collect::<Vec<_>>(),
        dtype,
    );
    let k_values = round_for_dtype(
        (0..config.kv_elements(shape))
            .map(|index| index as f32 * -0.025 + 0.3)
            .collect::<Vec<_>>(),
        dtype,
    );
    let v_values = round_for_dtype(
        (0..config.kv_elements(shape))
            .map(|index| index as f32 * 0.05 - 0.2)
            .collect::<Vec<_>>(),
        dtype,
    );
    let q = buffer(&device, &q_values, dtype);
    let local_k = buffer(&device, &k_values, dtype);
    let local_v = buffer(&device, &v_values, dtype);
    let q_token_ranges = Buffer::from_slice(&device, &[0_u32, 3, 3, 6]);
    let cu_sdpa_partial_outputs = Buffer::from_slice(&device, &[0_u32, 1, 2]);
    let partial_exp_sums =
        Buffer::new_zeroed_elements(&device, config.partial_output_stat_elements(shape), Dtype::Float32);
    let partial_max_logits =
        Buffer::new_zeroed_elements(&device, config.partial_output_stat_elements(shape), Dtype::Float32);
    let partial_output = Buffer::new_zeroed_elements(&device, config.partial_output_values(shape), dtype);
    let kernel = Compute::new(&device, config);
    let mut builder = stream.create_replay_program();
    builder.record(kernel.invoke(
        shape,
        ReplayU32::Fixed(shape.num_total_q_token_ranges),
        Buffers {
            q: &q,
            local_k: &local_k,
            local_v: &local_v,
            q_token_ranges: &q_token_ranges,
            cu_sdpa_partial_outputs: &cu_sdpa_partial_outputs,
            partial_exp_sums: &partial_exp_sums,
            partial_max_logits: &partial_max_logits,
            partial_output: &partial_output,
        },
    ));
    stream.submit_replay(&builder.build()).wait();

    let actual_partial_exp_sums = partial_exp_sums.read_typed::<f32>(0, config.partial_output_stat_elements(shape));
    let actual_partial_max_logits = partial_max_logits.read_typed::<f32>(0, config.partial_output_stat_elements(shape));
    let actual_output = read_values(&partial_output, config.partial_output_values(shape), dtype);
    for q_token_index in 0..shape.num_total_tokens as usize {
        let local_kv_token_begin = q_token_index / config.block_size as usize * config.block_size as usize;
        for q_head in 0..config.num_q_heads as usize {
            let q_start = (q_token_index * config.num_q_heads as usize + q_head) * config.head_dim as usize;
            let q_row = &q_values[q_start..q_start + config.head_dim as usize];
            let mut scores = Vec::new();
            for local_kv_offset in 0..config.block_size as usize {
                let kv_token_index = local_kv_token_begin + local_kv_offset;
                let kv_start = kv_token_index * config.head_dim as usize;
                let key = &k_values[kv_start..kv_start + config.head_dim as usize];
                scores.push(q_row.iter().zip(key).map(|(&q, &k)| q * k).sum::<f32>() * config.scale);
            }
            let max_logit = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores
                .iter()
                .map(|&score| (score - max_logit).exp())
                .collect::<Vec<_>>();
            let exp_sum = weights.iter().sum::<f32>();
            let partial_output_index = ((q_token_index / config.block_size as usize * config.num_q_heads as usize
                + q_head)
                * config.max_q_tokens as usize)
                + q_token_index % config.block_size as usize;
            assert!(
                (actual_partial_max_logits[partial_output_index] - max_logit).abs() < 1.0e-4,
                "actual={} expected={max_logit}",
                actual_partial_max_logits[partial_output_index]
            );
            assert!(
                (actual_partial_exp_sums[partial_output_index] - exp_sum).abs() < 1.0e-4,
                "actual={} expected={exp_sum}",
                actual_partial_exp_sums[partial_output_index]
            );
            for dim in 0..config.head_dim as usize {
                let expected = weights
                    .iter()
                    .enumerate()
                    .map(|(local_kv_offset, &weight)| {
                        let kv_token_index = local_kv_token_begin + local_kv_offset;
                        let kv_start = kv_token_index * config.head_dim as usize;
                        weight * v_values[kv_start + dim]
                    })
                    .sum::<f32>()
                    / exp_sum;
                let actual = actual_output[partial_output_index * config.head_dim as usize + dim];
                assert!(
                    (actual - expected).abs() < output_tolerance,
                    "actual={actual} expected={expected}"
                );
            }
        }
    }
}

fn round_for_dtype(values: Vec<f32>, dtype: Dtype) -> Vec<f32> {
    match dtype {
        Dtype::Float32 => values,
        Dtype::Bfloat16 => values.into_iter().map(|value| bf16::from_f32(value).to_f32()).collect(),
        dtype => panic!("unsupported test dtype {dtype:?}"),
    }
}

fn bf16_pattern(count: usize, step: f32, bias: f32) -> Vec<f32> {
    (0..count)
        .map(|index| bf16::from_f32((index % 97) as f32 * step + bias).to_f32())
        .collect()
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(&lhs, &rhs)| lhs * rhs).sum()
}

fn buffer(device: &Device, values: &[f32], dtype: Dtype) -> Buffer {
    match dtype {
        Dtype::Float32 => Buffer::from_slice(device, values),
        Dtype::Bfloat16 => {
            let bits = values
                .iter()
                .map(|value| bf16::from_f32(*value).to_bits())
                .collect::<Vec<_>>();
            Buffer::from_slice(device, &bits)
        },
        dtype => panic!("unsupported test dtype {dtype:?}"),
    }
}

fn read_values(buffer: &Buffer, count: usize, dtype: Dtype) -> Vec<f32> {
    match dtype {
        Dtype::Float32 => buffer.read_typed::<f32>(0, count),
        Dtype::Bfloat16 => {
            buffer
                .read_typed::<u16>(0, count)
                .into_iter()
                .map(|bits| bf16::from_bits(bits).to_f32())
                .collect()
        },
        dtype => panic!("unsupported test dtype {dtype:?}"),
    }
}
