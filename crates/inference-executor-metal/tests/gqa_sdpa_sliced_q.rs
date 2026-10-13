use std::mem::size_of;

use half::bf16;
use inference_backend_metal::components::gqa::fp8;
use inference_backend_metal::components::gqa::kv_page_write as backend_kv_page_write;
use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::components::gqa::sdpa::sliced_q as backend_sliced_q;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_executor_core::attn::GQACore;
use inference_executor_core::attn::gqa::reference::GQAReferenceInput;
use inference_executor_core::attn::gqa::reference::projected_gqa_reference;

const MAX_Q_TOKENS: u32 = 8;
const KV_TOKENS_PER_ITERATION: u32 = 16;
const NUM_ACTIVE_Q_TOKEN_TILES: ReplayParameterKey = ReplayParameterKey::new("test.gqa.tiled.active_q_tiles");
const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.gqa.tiled.active_tokens");
const NUM_ACTIVE_KV_SPLITS: ReplayParameterKey = ReplayParameterKey::new("test.gqa.tiled.active_kv_splits");

#[test]
fn test_fixed() {
    run_case(&[16], 8, 256, 16, 17);
}

#[test]
fn test_qwen38_27b_profile() {
    run_case(&[16], 6, 256, 16, 17);
}

#[test]
fn test_ragged() {
    run_case(&[1, 2, 4, 6, 8, 11, 16, 16], 6, 256, 16, 17);
}

#[test]
fn test_multiple_q_token_ranges() {
    run_case(&[32, 8], 8, 256, 16, 17);
}

#[test]
fn test_qwen3_profile() {
    run_case(&[1, 4, 8, 13], 5, 128, 16, 17);
}

#[test]
fn test_bucketed_replay_ignores_poisoned_q_token_range_tail() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let num_q_heads = 4usize;
    let num_kv_heads = 1usize;
    let head_dim = 128usize;
    let num_tokens_per_page = 8usize;
    let num_total_tokens = 16usize;
    let num_blocks = 2usize;
    let page_bytes = 2 * num_kv_heads * num_tokens_per_page * head_dim * size_of::<u8>();
    let config = backend_sliced_q::Config {
        num_q_heads: num_q_heads as u32,
        num_kv_heads: num_kv_heads as u32,
        head_dim: head_dim as u32,
        scale: (head_dim as f32).sqrt().recip(),
        page_bytes: page_bytes as u32,
        dtype: Dtype::Bfloat16,
        page_table_layout: backend_kv_page_write::PageTableLayout {
            num_req_slots: 1,
            num_blocks: num_blocks as u32,
            num_gqa_layers: 1,
            num_page_ids_per_block: 1,
        },
    };
    let execution = tiled_execution(config, num_q_heads as u32);
    let shape = backend_sliced_q::Shape {
        num_total_tokens: num_total_tokens as u32,
        num_total_q_token_tiles: 2,
        num_total_sdpa_map_task_templates: 2,
    };
    let q_values = pattern(num_total_tokens * num_q_heads * head_dim, 29, 0.015625);
    let k_values = pattern(num_total_tokens * head_dim, 31, 0.015625);
    let v_values = pattern(num_total_tokens * head_dim, 37, 0.03125);
    let q_bf16 = q_values
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let k_f32 = k_values
        .iter()
        .map(|&value| fp8::bf16_to_f32(fp8::fp8_e4m3_to_bf16(fp8::bf16_to_fp8_e4m3(fp8::f32_to_bf16(value)))))
        .collect::<Vec<_>>();
    let v_f32 = v_values
        .iter()
        .map(|&value| fp8::bf16_to_f32(fp8::fp8_e4m3_to_bf16(fp8::bf16_to_fp8_e4m3(fp8::f32_to_bf16(value)))))
        .collect::<Vec<_>>();
    let q_f32 = q_bf16
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect::<Vec<_>>();
    let mut kv_page_values = Vec::new();
    for block_index in 0..num_blocks {
        let start = block_index * num_tokens_per_page * head_dim;
        let end = start + num_tokens_per_page * head_dim;
        kv_page_values.extend(
            k_values[start..end]
                .iter()
                .map(|&value| fp8::bf16_to_fp8_e4m3(fp8::f32_to_bf16(value))),
        );
        kv_page_values.extend(
            v_values[start..end]
                .iter()
                .map(|&value| fp8::bf16_to_fp8_e4m3(fp8::f32_to_bf16(value))),
        );
    }

    let q = Buffer::from_slice(&device, &q_bf16);
    let kv_pages = Buffer::from_slice(&device, &kv_page_values);
    let req_slots = Buffer::from_slice(&device, &[0_u32; 16]);
    let page_ids = Buffer::from_slice(&device, &[0_u32, 1]);
    let visible_kv_token_ranges = Buffer::from_slice(
        &device,
        &(0_u32..16)
            .flat_map(|q_token_index| [0, q_token_index + 1])
            .collect::<Vec<_>>(),
    );
    let q_token_ranges = Buffer::from_slice(&device, &[0_u32, 5, u32::MAX, u32::MAX]);
    let task_templates = Buffer::from_slice(&device, &[0_u32, 0, 5, u32::MAX, u32::MAX, u32::MAX]);
    let cu_partial_outputs = Buffer::from_slice(&device, &[0_u32, 1, u32::MAX]);
    let num_reserved_partial_state_groups = 2 * MAX_Q_TOKENS as usize;
    let partial_output = Buffer::new_zeroed(
        &device,
        num_reserved_partial_state_groups * num_q_heads * head_dim * Dtype::Bfloat16.item_size(),
    );
    let partial_exp_sums = Buffer::new_zeroed(
        &device,
        num_reserved_partial_state_groups * num_q_heads * size_of::<f32>(),
    );
    let partial_max_logits = Buffer::new_zeroed(
        &device,
        num_reserved_partial_state_groups * num_q_heads * size_of::<f32>(),
    );
    let sentinel = bf16::from_f32(-123.0).to_bits();
    let output = Buffer::from_slice(&device, &vec![sentinel; num_total_tokens * num_q_heads * head_dim]);
    let kernels = backend_sliced_q::Compute::new(&device, config, execution, shape);
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_map(
        backend_sliced_q::MapBuffers {
            q: &q,
            kv_pages: &kv_pages,
            req_slots: &req_slots,
            page_ids: &page_ids,
            visible_kv_token_ranges: &visible_kv_token_ranges,
            q_token_ranges: &q_token_ranges,
            sdpa_map_task_templates: &task_templates,
            partial_output: &partial_output,
            partial_exp_sums: &partial_exp_sums,
            partial_max_logits: &partial_max_logits,
        },
        ReplayU32::Fixed(0),
        ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
        ReplayU32::Parameter(NUM_ACTIVE_Q_TOKEN_TILES),
        ReplayU32::Parameter(NUM_ACTIVE_KV_SPLITS),
    ));
    builder.record_with_barrier_before(kernels.invoke_reduce(
        backend_sliced_q::ReduceBuffers {
            partial_output: &partial_output,
            partial_exp_sums: &partial_exp_sums,
            partial_max_logits: &partial_max_logits,
            q_token_ranges: &q_token_ranges,
            cu_sdpa_partial_outputs: &cu_partial_outputs,
            output: &output,
        },
        ReplayU32::Parameter(NUM_ACTIVE_Q_TOKEN_TILES),
    ));
    let replay = builder.build();
    assert_eq!(replay.stats().parameter_count, 3);

    for num_active_tokens in [5_usize, 16] {
        let num_active_q_token_ranges = num_active_tokens.div_ceil(MAX_Q_TOKENS as usize);
        q_token_ranges.write_typed(
            0,
            if num_active_q_token_ranges == 1 {
                &[0_u32, 5, u32::MAX, u32::MAX]
            } else {
                &[0_u32, 8, 8, 16]
            },
        );
        task_templates.write_typed(
            0,
            if num_active_q_token_ranges == 1 {
                &[0_u32, 0, 5, u32::MAX, u32::MAX, u32::MAX]
            } else {
                &[0_u32, 0, 8, 1, 0, 16]
            },
        );
        cu_partial_outputs.write_typed(
            0,
            if num_active_q_token_ranges == 1 {
                &[0_u32, 1, u32::MAX]
            } else {
                &[0_u32, 1, 2]
            },
        );
        output.write_typed(0, &vec![sentinel; num_total_tokens * num_q_heads * head_dim]);
        stream
            .submit_replay_with_arguments(
                &replay,
                &ReplayArguments::new()
                    .with_u32(NUM_ACTIVE_TOKENS, num_active_tokens as u32)
                    .with_u32(NUM_ACTIVE_Q_TOKEN_TILES, num_active_q_token_ranges as u32)
                    .with_u32(NUM_ACTIVE_KV_SPLITS, num_active_q_token_ranges as u32),
            )
            .wait();

        let num_active_values = num_active_tokens * num_q_heads * head_dim;
        let actual = output
            .read_typed::<u16>(0, num_active_values)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        let expected = projected_gqa_reference(
            &GQACore::new(
                0,
                num_q_heads * head_dim,
                head_dim,
                num_q_heads,
                num_kv_heads,
                config.scale,
            ),
            GQAReferenceInput {
                cu_tokens: &[0, num_active_tokens as u32],
                token_indices: &[0],
                q: &q_f32[..num_active_values],
                context_k_by_req: &[&k_f32[..num_active_tokens * head_dim]],
                context_v_by_req: &[&v_f32[..num_active_tokens * head_dim]],
            },
        );
        let max_abs_diff = actual
            .iter()
            .zip(&expected)
            .map(|(actual_value, expected_value)| (actual_value - expected_value).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs_diff <= 0.02, "max_abs_diff={max_abs_diff}");
        assert_eq!(
            output.read_typed::<u16>(
                num_active_values,
                (num_total_tokens - num_active_tokens) * num_q_heads * head_dim,
            ),
            vec![sentinel; (num_total_tokens - num_active_tokens) * num_q_heads * head_dim]
        );
    }
}

fn run_case(
    num_tokens_per_req: &[usize],
    num_q_heads: usize,
    head_dim: usize,
    num_tokens_per_page: usize,
    start_token_index: usize,
) {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let num_kv_heads = 1;
    let num_tokens = num_tokens_per_req.iter().sum::<usize>();
    let num_blocks =
        (start_token_index + num_tokens_per_req.iter().copied().max().unwrap()).div_ceil(num_tokens_per_page);
    let page_bytes = 2 * num_kv_heads * num_tokens_per_page * head_dim * size_of::<u8>();
    let q_values = pattern(num_tokens * num_q_heads * head_dim, 29, 0.015625);
    let mut context_k_values_by_req = Vec::new();
    let mut context_v_values_by_req = Vec::new();
    let mut kv_page_values = Vec::new();
    for req_index in 0..num_tokens_per_req.len() {
        let num_kv_slots = num_blocks * num_tokens_per_page * head_dim;
        let k = pattern(num_kv_slots, 31 + req_index, 0.015625);
        let v = pattern(num_kv_slots, 37 + req_index, 0.03125);
        for block_index in 0..num_blocks {
            let start = block_index * num_tokens_per_page * head_dim;
            let end = start + num_tokens_per_page * head_dim;
            kv_page_values.extend(
                k[start..end]
                    .iter()
                    .map(|&value| fp8::bf16_to_fp8_e4m3(fp8::f32_to_bf16(value))),
            );
            kv_page_values.extend(
                v[start..end]
                    .iter()
                    .map(|&value| fp8::bf16_to_fp8_e4m3(fp8::f32_to_bf16(value))),
            );
        }
        context_k_values_by_req.push(k);
        context_v_values_by_req.push(v);
    }
    let context_k_by_req = context_k_values_by_req
        .iter()
        .map(|values| values.as_slice())
        .collect::<Vec<_>>();
    let context_v_by_req = context_v_values_by_req
        .iter()
        .map(|values| values.as_slice())
        .collect::<Vec<_>>();

    let mut req_slot_values = Vec::new();
    let mut flat_token_index_values = Vec::new();
    let mut q_token_range_values = Vec::new();
    let mut sdpa_map_task_template_values = Vec::new();
    let mut cu_sdpa_partial_output_values = vec![0u32];
    let mut cu_tokens = vec![0u32];
    let mut flat_token_start = 0u32;
    for (req_slot, &num_req_tokens) in num_tokens_per_req.iter().enumerate() {
        req_slot_values.extend(std::iter::repeat_n(req_slot as u32, num_req_tokens));
        flat_token_index_values.extend(start_token_index as u32..start_token_index as u32 + num_req_tokens as u32);
        let flat_token_end = flat_token_start + num_req_tokens as u32;
        let mut q_token_begin = flat_token_start;
        while q_token_begin < flat_token_end {
            let q_token_end = (q_token_begin + MAX_Q_TOKENS).min(flat_token_end);
            q_token_range_values.extend_from_slice(&[q_token_begin, q_token_end]);
            let q_token_range_index = q_token_range_values.len() / 2 - 1;
            let context_len = start_token_index as u32 + q_token_end - flat_token_start;
            let mut kv_token_begin = 0;
            while kv_token_begin < context_len {
                let kv_token_end = (kv_token_begin + KV_TOKENS_PER_ITERATION).min(context_len);
                sdpa_map_task_template_values.extend_from_slice(&[
                    q_token_range_index as u32,
                    kv_token_begin,
                    kv_token_end,
                ]);
                kv_token_begin = kv_token_end;
            }
            cu_sdpa_partial_output_values.push((sdpa_map_task_template_values.len() / 3) as u32);
            q_token_begin = q_token_end;
        }
        cu_tokens.push(flat_token_end);
        flat_token_start = flat_token_end;
    }
    let num_q_token_ranges = q_token_range_values.len() / 2;
    let num_sdpa_map_task_templates = sdpa_map_task_template_values.len() / 3;
    let num_total_sdpa_map_task_templates = num_sdpa_map_task_templates.next_power_of_two();
    sdpa_map_task_template_values.resize(num_total_sdpa_map_task_templates * 3, u32::MAX);
    let config = backend_sliced_q::Config {
        num_q_heads: num_q_heads.try_into().unwrap(),
        num_kv_heads: num_kv_heads.try_into().unwrap(),
        head_dim: head_dim.try_into().unwrap(),
        scale: (head_dim as f32).sqrt().recip(),
        page_bytes: page_bytes.try_into().unwrap(),
        dtype: Dtype::Bfloat16,
        page_table_layout: backend_kv_page_write::PageTableLayout {
            num_req_slots: num_tokens_per_req.len().try_into().unwrap(),
            num_blocks: num_blocks.try_into().unwrap(),
            num_gqa_layers: 1,
            num_page_ids_per_block: 1,
        },
    };
    let execution = tiled_execution(config, num_q_heads.try_into().unwrap());
    let shape = backend_sliced_q::Shape {
        num_total_tokens: num_tokens.try_into().unwrap(),
        num_total_q_token_tiles: num_q_token_ranges.try_into().unwrap(),
        num_total_sdpa_map_task_templates: num_total_sdpa_map_task_templates.try_into().unwrap(),
    };

    let q_bf16 = q_values
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let q_f32 = q_bf16
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect::<Vec<_>>();
    let context_k_f32_by_req = context_k_by_req
        .iter()
        .map(|k| {
            k.iter()
                .map(|&value| fp8::bf16_to_f32(fp8::fp8_e4m3_to_bf16(fp8::bf16_to_fp8_e4m3(fp8::f32_to_bf16(value)))))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let context_v_f32_by_req = context_v_by_req
        .iter()
        .map(|v| {
            v.iter()
                .map(|&value| fp8::bf16_to_f32(fp8::fp8_e4m3_to_bf16(fp8::bf16_to_fp8_e4m3(fp8::f32_to_bf16(value)))))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let context_k_refs = context_k_f32_by_req
        .iter()
        .zip(num_tokens_per_req)
        .map(|(k, &num_req_tokens)| &k[..(start_token_index + num_req_tokens) * head_dim])
        .collect::<Vec<_>>();
    let context_v_refs = context_v_f32_by_req
        .iter()
        .zip(num_tokens_per_req)
        .map(|(v, &num_req_tokens)| &v[..(start_token_index + num_req_tokens) * head_dim])
        .collect::<Vec<_>>();

    let expected = projected_gqa_reference(
        &GQACore::new(
            0,
            num_q_heads * head_dim,
            head_dim,
            num_q_heads,
            num_kv_heads,
            config.scale,
        ),
        GQAReferenceInput {
            cu_tokens: &cu_tokens,
            token_indices: &vec![start_token_index as u32; num_tokens_per_req.len()],
            q: &q_f32,
            context_k_by_req: &context_k_refs,
            context_v_by_req: &context_v_refs,
        },
    );

    let q = Buffer::from_slice(&device, &q_bf16);
    let kv_pages = Buffer::from_slice(&device, &kv_page_values);
    let req_slots = Buffer::from_slice(&device, &req_slot_values);
    let page_ids = Buffer::from_slice(
        &device,
        &(0..(num_tokens_per_req.len() * num_blocks) as u32).collect::<Vec<_>>(),
    );
    let visible_kv_token_ranges = Buffer::from_slice(
        &device,
        &flat_token_index_values
            .iter()
            .flat_map(|&q_token_index| [0, q_token_index + 1])
            .collect::<Vec<_>>(),
    );
    let q_token_ranges = Buffer::from_slice(&device, &q_token_range_values);
    let sdpa_map_task_templates = Buffer::from_slice(&device, &sdpa_map_task_template_values);
    let cu_sdpa_partial_outputs = Buffer::from_slice(&device, &cu_sdpa_partial_output_values);
    let num_reserved_partial_state_groups = num_total_sdpa_map_task_templates * MAX_Q_TOKENS as usize;
    let partial_output = Buffer::new_zeroed(
        &device,
        num_reserved_partial_state_groups * num_q_heads * head_dim * Dtype::Bfloat16.item_size(),
    );
    let partial_exp_sums = Buffer::new_zeroed(
        &device,
        num_reserved_partial_state_groups * num_q_heads * size_of::<f32>(),
    );
    let partial_max_logits = Buffer::new_zeroed(
        &device,
        num_reserved_partial_state_groups * num_q_heads * size_of::<f32>(),
    );
    let output = Buffer::new_zeroed(
        &device,
        num_tokens * num_q_heads * head_dim * Dtype::Bfloat16.item_size(),
    );
    let kernels = backend_sliced_q::Compute::new(&device, config, execution, shape);
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_map(
        backend_sliced_q::MapBuffers {
            q: &q,
            kv_pages: &kv_pages,
            req_slots: &req_slots,
            page_ids: &page_ids,
            visible_kv_token_ranges: &visible_kv_token_ranges,
            q_token_ranges: &q_token_ranges,
            sdpa_map_task_templates: &sdpa_map_task_templates,
            partial_output: &partial_output,
            partial_exp_sums: &partial_exp_sums,
            partial_max_logits: &partial_max_logits,
        },
        ReplayU32::Fixed(0),
        ReplayU32::Fixed(shape.num_total_tokens),
        ReplayU32::Fixed(shape.num_total_q_token_tiles),
        ReplayU32::Fixed(shape.num_total_sdpa_map_task_templates),
    ));
    builder.record_with_barrier_before(kernels.invoke_reduce(
        backend_sliced_q::ReduceBuffers {
            partial_output: &partial_output,
            partial_exp_sums: &partial_exp_sums,
            partial_max_logits: &partial_max_logits,
            q_token_ranges: &q_token_ranges,
            cu_sdpa_partial_outputs: &cu_sdpa_partial_outputs,
            output: &output,
        },
        ReplayU32::Fixed(shape.num_total_q_token_tiles),
    ));
    let replay = builder.build();
    stream.submit_replay(&replay).wait();
    let actual = output
        .read_typed::<u16>(0, expected.len())
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();

    let max_abs_diff = actual
        .iter()
        .zip(&expected)
        .map(|(actual_value, expected_value)| (actual_value - expected_value).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs_diff <= 0.02, "max_abs_diff={max_abs_diff}");
}

fn tiled_execution(config: backend_sliced_q::Config, max_q_heads: u32) -> backend_sdpa::ExecutionVariant {
    backend_sdpa::ExecutionVariant::sliced_q_per_head(
        backend_sdpa::Config {
            io_dtype: config.dtype,
            num_q_heads: config.num_q_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            tokens_per_page: config.num_tokens_per_page(),
        },
        MAX_Q_TOKENS,
        KV_TOKENS_PER_ITERATION,
        max_q_heads,
    )
}

fn pattern(len: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|index| (index % modulus) as f32 * scale - (modulus as f32 * scale * 0.5))
        .collect()
}
