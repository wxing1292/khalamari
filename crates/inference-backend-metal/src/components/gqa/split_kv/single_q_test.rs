use half::bf16;
use inference_executor_core::attn::gqa::GQACore;
use inference_executor_core::attn::gqa::reference::GQAReferenceInput;
use inference_executor_core::attn::gqa::reference::projected_gqa_reference;

use super::*;
use crate::components::gqa::fp8::MAX_FINITE;
use crate::components::gqa::fp8::bf16_to_f32;
use crate::components::gqa::fp8::bf16_to_fp8_e4m3;
use crate::components::gqa::fp8::f32_to_bf16;
use crate::components::gqa::fp8::fp8_e4m3_to_bf16;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::Stream;
use crate::test_support::ReplayTestCache;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.gqa.split_kv_single_q.active_tokens");
const NUM_ACTIVE_KV_SPLITS: ReplayParameterKey = ReplayParameterKey::new("test.gqa.paged.active_kv_splits");

#[test]
fn test_pipelines_reused_across_replay_capacities() {
    let device = Device::system_default();
    let config = fixture_config();
    let execution = fixture_execution(config);
    let shape = Shape {
        num_total_tokens: 8,
        num_total_sdpa_map_task_templates: 8,
    };
    let kernels = Compute::new(&device, config, execution, shape);
    for (num_total_tokens, num_total_sdpa_map_task_templates) in [(4, 4), (8, 16), (16, 16)] {
        let other = Compute::new(
            &device,
            config,
            execution,
            Shape {
                num_total_tokens,
                num_total_sdpa_map_task_templates,
            },
        );
        assert!(std::ptr::eq(kernels.map.as_raw(), other.map.as_raw()));
        assert!(std::ptr::eq(kernels.reduce.as_raw(), other.reduce.as_raw()));
    }
}

#[test]
#[should_panic(expected = "GQA SDPA query/output exceeds the shader u32 count domain")]
fn test_sdpa_shape_rejects_shader_count_overflow() {
    let config = Config {
        num_q_heads: 2,
        num_kv_heads: 1,
        head_dim: 2,
        scale: 1.0,
        page_bytes: 8,
        page_table_layout: PageTableLayout {
            num_req_slots: 1,
            num_gqa_layers: 1,
            num_blocks: 1,
            num_page_ids_per_block: 1,
        },
        dtype: Dtype::Bfloat16,
    };
    Shape {
        num_total_tokens: 1 << 30,
        num_total_sdpa_map_task_templates: 1,
    }
    .validate(config);
}

#[test]
fn test_fp8_kv_output_quality_against_bf16_oracle() {
    let mut config = fixture_config();
    config.page_bytes *= 16;
    let num_tokens = config.num_tokens_per_page() as usize;
    let q_values = bf16_round_trip(&generated_values(
        num_tokens * config.num_q_heads as usize * config.head_dim as usize,
        0x7E10_2041,
    ));
    let k_values = bf16_round_trip(&generated_values(
        num_tokens * config.num_kv_heads as usize * config.head_dim as usize,
        0x7E10_2042,
    ));
    let v_values = bf16_round_trip(&generated_values(
        num_tokens * config.num_kv_heads as usize * config.head_dim as usize,
        0x7E10_2043,
    ));
    let saturation_count = k_values
        .iter()
        .chain(&v_values)
        .filter(|value| value.abs() > MAX_FINITE)
        .count();
    assert_eq!(saturation_count, 0, "quality fixture must not saturate FP8 K/V values");
    let core = fixture_core(config);
    let fp8_k = fp8_e4m3_round_trip(&k_values);
    let fp8_v = fp8_e4m3_round_trip(&v_values);
    let bf16_output = projected_gqa_reference(
        &core,
        GQAReferenceInput {
            cu_tokens: &[0, num_tokens as u32],
            token_indices: &[0],
            q: &q_values,
            context_k_by_req: &[&k_values],
            context_v_by_req: &[&v_values],
        },
    );
    let fp8_output = projected_gqa_reference(
        &core,
        GQAReferenceInput {
            cu_tokens: &[0, num_tokens as u32],
            token_indices: &[0],
            q: &q_values,
            context_k_by_req: &[&fp8_k],
            context_v_by_req: &[&fp8_v],
        },
    );
    assert_quality(&fp8_output, &bf16_output, 0.025, 0.005);
}

#[test]
fn test_replay_matches_reference_across_active_counts() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let mut config = fixture_config();
    config.page_bytes *= 2;
    let shape = Shape {
        num_total_tokens: 8,
        num_total_sdpa_map_task_templates: 8,
    };
    let q = generated_values(config.num_output_values(shape), 0x8B1D_5A73);
    let kv_stride = config.num_kv_heads as usize * config.head_dim as usize;
    let k = generated_values(8 * kv_stride, 0x8B1D_5A74);
    let v = generated_values(8 * kv_stride, 0x8B1D_5A75);
    let kv_pages = kv_page_values(config, &[(&k, &v)]);
    let q_buffer = bf16_buffer(&device, &q);
    let kv_pages_buffer = Buffer::from_slice(&device, &kv_pages);
    let req_slots = Buffer::from_slice(&device, &[0_u32; 8]);
    let page_ids = Buffer::from_slice(&device, &[0_u32]);
    let task_templates = Buffer::from_slice(
        &device,
        &(0..8_u32)
            .flat_map(|token_index| [token_index, 0, token_index + 1])
            .collect::<Vec<_>>(),
    );
    let cu_partial_outputs = Buffer::from_slice(&device, &(0..=8_u32).collect::<Vec<_>>());
    let output = Buffer::new_zeroed_elements(&device, config.num_output_values(shape), Dtype::Bfloat16);
    let scratch = Scratch::new(&device, config, shape);
    let kernels = Compute::new(&device, config, fixture_execution(config), shape);
    let cache_key = (shape.num_total_tokens, shape.num_total_sdpa_map_task_templates);
    let mut cache = ReplayTestCache::new();
    let (_, cache_hit) = cache.record(cache_key, || {
        let mut builder = stream.create_replay_program();
        builder.record(kernels.invoke_map(
            scratch.map_buffers(&q_buffer, &kv_pages_buffer, &req_slots, &page_ids, &task_templates),
            ReplayU32::Fixed(0),
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            ReplayU32::Parameter(NUM_ACTIVE_KV_SPLITS),
        ));
        builder.record_with_barrier_before(kernels.invoke_reduce(
            scratch.reduce_buffers(&cu_partial_outputs, &output),
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
        ));
        builder.build()
    });
    assert!(!cache_hit);
    for num_active_tokens in [1_usize, 8, 3, 7, 2, 6, 4, 5] {
        let (replay, cache_hit) = cache.record(cache_key, || unreachable!());
        assert!(cache_hit);
        stream
            .submit_replay_with_arguments(
                replay,
                &ReplayArguments::new()
                    .with_u32(NUM_ACTIVE_TOKENS, num_active_tokens as u32)
                    .with_u32(NUM_ACTIVE_KV_SPLITS, num_active_tokens as u32),
            )
            .wait();
        let num_active_values = num_active_tokens * config.num_q_heads as usize * config.head_dim as usize;
        let actual = read_bf16(&output, num_active_values);
        let quantized_k = fp8_e4m3_round_trip(&k[..num_active_tokens * kv_stride]);
        let quantized_v = fp8_e4m3_round_trip(&v[..num_active_tokens * kv_stride]);
        let expected = projected_gqa_reference(
            &fixture_core(config),
            GQAReferenceInput {
                cu_tokens: &[0, num_active_tokens as u32],
                token_indices: &[0],
                q: &q[..num_active_values],
                context_k_by_req: &[&quantized_k],
                context_v_by_req: &[&quantized_v],
            },
        );
        assert_close(&actual, &expected, 2.0e-2);
    }
}

#[test]
fn test_ragged_requests_match_reference() {
    let random_seed = 0xD205_6AB9;
    let mut config = fixture_config();
    let mut shape = fixture_shape();
    shape.num_total_tokens = 3;
    shape.num_total_sdpa_map_task_templates = 4;
    config.page_table_layout.num_req_slots = 2;
    let q = generated_values(config.num_output_values(shape), random_seed);
    let kv_stride = config.num_kv_heads as usize * config.head_dim as usize;
    let req0_k = generated_values(2 * kv_stride, random_seed.wrapping_add(1));
    let req0_v = generated_values(2 * kv_stride, random_seed.wrapping_add(2));
    let req1_k = generated_values(2 * kv_stride, random_seed.wrapping_add(3));
    let req1_v = generated_values(2 * kv_stride, random_seed.wrapping_add(4));
    let kv_pages = kv_page_values(config, &[(&req0_k, &req0_v), (&req1_k, &req1_v)]);
    let actual = run_gqa_split_kv_single_q(
        config,
        shape,
        TestInput {
            q: &q,
            kv_pages: &kv_pages,
            req_slots: &[0, 1, 1],
            page_ids: &[0, 1],
            flat_token_indices: &[1, 0, 1],
        },
    );
    let quantized_req0_k = fp8_e4m3_round_trip(&req0_k);
    let quantized_req0_v = fp8_e4m3_round_trip(&req0_v);
    let quantized_req1_k = fp8_e4m3_round_trip(&req1_k);
    let quantized_req1_v = fp8_e4m3_round_trip(&req1_v);
    let expected = projected_gqa_reference(
        &fixture_core(config),
        GQAReferenceInput {
            cu_tokens: &[0, 1, 3],
            token_indices: &[1, 0],
            q: &q,
            context_k_by_req: &[&quantized_req0_k, &quantized_req1_k],
            context_v_by_req: &[&quantized_req0_v, &quantized_req1_v],
        },
    );
    assert_close(&actual, &expected, 2.0e-2);
}

#[test]
fn test_multiple_page_ids_per_block() {
    for head_dim in [2, 65, 128, 256] {
        for dtype in [Dtype::Bfloat16, Dtype::Float32] {
            let mut config = fixture_config();
            config.head_dim = head_dim;
            config.dtype = dtype;
            let tokens_per_page = if head_dim == 2 { 4 } else { 32 };
            config.page_bytes = 2 * config.num_kv_heads * tokens_per_page * head_dim;
            let tokens = tokens_per_page as usize * 2;
            let mut shape = fixture_shape();
            shape.num_total_tokens = 1;
            shape.num_total_sdpa_map_task_templates = 1;
            config.page_table_layout.num_page_ids_per_block = 2;
            let kv_stride = config.num_kv_heads as usize * config.head_dim as usize;
            let q = fixture_values(config.num_output_values(shape), 0.125, 3);
            let k = fixture_values(tokens * kv_stride, 0.0625, 5);
            let v = fixture_values(tokens * kv_stride, 0.25, 7);
            let kv_pages = kv_page_values(
                config,
                &[
                    (&k[..tokens / 2 * kv_stride], &v[..tokens / 2 * kv_stride]),
                    (&k[tokens / 2 * kv_stride..], &v[tokens / 2 * kv_stride..]),
                ],
            );
            let actual = run_gqa_split_kv_single_q(
                config,
                shape,
                TestInput {
                    q: &q,
                    kv_pages: &kv_pages,
                    req_slots: &[0],
                    page_ids: &[0, 1],
                    flat_token_indices: &[tokens as u32 - 1],
                },
            );
            let quantized_k = fp8_e4m3_round_trip(&k);
            let quantized_v = fp8_e4m3_round_trip(&v);
            let expected = projected_gqa_reference(
                &fixture_core(config),
                GQAReferenceInput {
                    cu_tokens: &[0, 1],
                    token_indices: &[tokens as u32 - 1],
                    q: &q,
                    context_k_by_req: &[&quantized_k],
                    context_v_by_req: &[&quantized_v],
                },
            );
            assert_close(&actual, &expected, 2.0e-2);
        }
    }
}

fn fixture_config() -> Config {
    let num_kv_heads = 2;
    let num_tokens_per_page = 4;
    let head_dim = 2;
    Config {
        num_q_heads: 4,
        num_kv_heads,
        head_dim,
        scale: 0.5,
        page_bytes: 2 * num_kv_heads * num_tokens_per_page * head_dim * size_of::<u8>() as u32,
        page_table_layout: PageTableLayout {
            num_req_slots: 1,
            num_blocks: 1,
            num_gqa_layers: 1,
            num_page_ids_per_block: 1,
        },
        dtype: Dtype::Bfloat16,
    }
}

fn fixture_execution(config: Config) -> sdpa::ExecutionVariant {
    sdpa::ExecutionVariant::single_q(
        config.sdpa_config(),
        4,
        config.head_dim.clamp(32, 128).next_power_of_two(),
        2,
    )
}

fn fixture_shape() -> Shape {
    Shape {
        num_total_tokens: 2,
        num_total_sdpa_map_task_templates: 2,
    }
}

fn fixture_core(config: Config) -> GQACore {
    let q_dim = config.num_q_heads as usize * config.head_dim as usize;
    GQACore::new(
        0,
        q_dim,
        config.head_dim as usize,
        config.num_q_heads as usize,
        config.num_kv_heads as usize,
        config.scale,
    )
}

struct TestInput<'a> {
    q: &'a [f32],
    kv_pages: &'a [u8],
    req_slots: &'a [u32],
    page_ids: &'a [u32],
    flat_token_indices: &'a [u32],
}

fn run_gqa_split_kv_single_q(config: Config, shape: Shape, input: TestInput<'_>) -> Vec<f32> {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let execution = fixture_execution(config);
    let kernels = Compute::new(&device, config, execution, shape);
    let q = match config.dtype {
        Dtype::Bfloat16 => bf16_buffer(&device, input.q),
        Dtype::Float32 => Buffer::from_slice(&device, input.q),
        _ => unreachable!(),
    };
    let kv_pages = Buffer::from_slice(&device, input.kv_pages);
    let req_slots = Buffer::from_slice(&device, input.req_slots);
    let page_ids = Buffer::from_slice(&device, input.page_ids);
    let (sdpa_map_task_template_values, cu_sdpa_partial_output_values) =
        sdpa_map_task_template_buffers(execution, shape, input.flat_token_indices);
    let sdpa_map_task_templates = Buffer::from_slice(&device, &sdpa_map_task_template_values);
    let cu_sdpa_partial_outputs = Buffer::from_slice(&device, &cu_sdpa_partial_output_values);
    let output = Buffer::new_zeroed(&device, config.q_bytes(shape));
    let scratch = Scratch::new(&device, config, shape);

    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_map(
        scratch.map_buffers(&q, &kv_pages, &req_slots, &page_ids, &sdpa_map_task_templates),
        ReplayU32::Fixed(0),
        ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
        ReplayU32::Parameter(NUM_ACTIVE_KV_SPLITS),
    ));
    builder.record_with_barrier_before(kernels.invoke_reduce(
        scratch.reduce_buffers(&cu_sdpa_partial_outputs, &output),
        ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
    ));
    let replay = builder.build();
    stream
        .submit_replay_with_arguments(
            &replay,
            &ReplayArguments::new()
                .with_u32(NUM_ACTIVE_TOKENS, shape.num_total_tokens)
                .with_u32(NUM_ACTIVE_KV_SPLITS, shape.num_total_sdpa_map_task_templates),
        )
        .wait();
    match config.dtype {
        Dtype::Bfloat16 => read_bf16(&output, config.num_output_values(shape)),
        Dtype::Float32 => output.read_typed::<f32>(0, config.num_output_values(shape)),
        _ => unreachable!(),
    }
}

fn sdpa_map_task_template_buffers(
    execution: sdpa::ExecutionVariant,
    shape: Shape,
    flat_token_indices: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let kv_tokens_per_iteration = execution.map.thread_block.kv_tokens_per_iteration;
    let num_kv_iterations = flat_token_indices
        .iter()
        .map(|&token_index| (token_index + 1).div_ceil(kv_tokens_per_iteration) as usize)
        .collect::<Vec<_>>();
    let mut num_sdpa_map_task_templates_by_q_token_range = vec![1_usize; flat_token_indices.len()];
    let mut num_sdpa_map_task_templates = num_sdpa_map_task_templates_by_q_token_range.len();
    while num_sdpa_map_task_templates < shape.num_total_sdpa_map_task_templates as usize {
        let Some(q_token_range_index) = (0..num_kv_iterations.len())
            .filter(|&q_token_range_index| {
                num_sdpa_map_task_templates_by_q_token_range[q_token_range_index]
                    < num_kv_iterations[q_token_range_index]
            })
            .max_by_key(|&q_token_range_index| {
                num_kv_iterations[q_token_range_index]
                    .div_ceil(num_sdpa_map_task_templates_by_q_token_range[q_token_range_index])
            })
        else {
            break;
        };
        num_sdpa_map_task_templates_by_q_token_range[q_token_range_index] += 1;
        num_sdpa_map_task_templates += 1;
    }
    let mut sdpa_map_task_templates = Vec::new();
    let mut cu_sdpa_partial_outputs = vec![0];
    for (q_token_range_index, &token_index) in flat_token_indices.iter().enumerate() {
        let context_len = token_index + 1;
        for sdpa_map_task_template_index in 0..num_sdpa_map_task_templates_by_q_token_range[q_token_range_index] {
            let kv_iteration_begin = num_kv_iterations[q_token_range_index] * sdpa_map_task_template_index
                / num_sdpa_map_task_templates_by_q_token_range[q_token_range_index];
            let kv_iteration_end = num_kv_iterations[q_token_range_index] * (sdpa_map_task_template_index + 1)
                / num_sdpa_map_task_templates_by_q_token_range[q_token_range_index];
            let kv_token_begin = kv_iteration_begin as u32 * kv_tokens_per_iteration;
            sdpa_map_task_templates.extend_from_slice(&[
                q_token_range_index as u32,
                kv_token_begin,
                context_len.min(kv_iteration_end as u32 * kv_tokens_per_iteration),
            ]);
        }
        cu_sdpa_partial_outputs.push((sdpa_map_task_templates.len() / 3) as u32);
    }
    assert!(sdpa_map_task_templates.len() / 3 <= shape.num_total_sdpa_map_task_templates as usize);
    sdpa_map_task_templates.resize(shape.num_total_sdpa_map_task_templates as usize * 3, u32::MAX);
    (sdpa_map_task_templates, cu_sdpa_partial_outputs)
}

fn kv_page_values(config: Config, pages: &[(&[f32], &[f32])]) -> Vec<u8> {
    let kv_stride = config.num_kv_heads as usize * config.head_dim as usize;
    let page_values = config.page_bytes as usize;
    let mut v = vec![0_u8; pages.len() * page_values];
    for (page_index, (k, page_v)) in pages.iter().enumerate() {
        assert_eq!(k.len(), page_v.len());
        assert_eq!(k.len() % kv_stride, 0);
        let num_tokens = k.len() / kv_stride;
        let num_tokens_per_page = config.num_tokens_per_page() as usize;
        assert!(num_tokens <= num_tokens_per_page);
        let page_base = page_index * page_values;
        for token in 0..num_tokens {
            for kv_head in 0..config.num_kv_heads as usize {
                for dim in 0..config.head_dim as usize {
                    let source = (token * config.num_kv_heads as usize + kv_head) * config.head_dim as usize + dim;
                    let k_target = page_base + (kv_head * num_tokens_per_page + token) * config.head_dim as usize + dim;
                    let v_target = page_base
                        + ((config.num_kv_heads as usize + kv_head) * num_tokens_per_page + token)
                            * config.head_dim as usize
                        + dim;
                    v[k_target] = bf16_to_fp8_e4m3(f32_to_bf16(k[source]));
                    v[v_target] = bf16_to_fp8_e4m3(f32_to_bf16(page_v[source]));
                }
            }
        }
    }
    v
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

fn read_bf16(buffer: &Buffer, count: usize) -> Vec<f32> {
    buffer
        .read_typed::<u16>(0, count)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect()
}

fn fp8_e4m3_round_trip(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&value| bf16_to_f32(fp8_e4m3_to_bf16(bf16_to_fp8_e4m3(f32_to_bf16(value)))))
        .collect()
}

fn bf16_round_trip(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&value| bf16_to_f32(f32_to_bf16(value))).collect()
}

fn fixture_values(count: usize, scale: f32, pattern_offset: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index * 11 + pattern_offset) % 23) as f32 * scale - 11.0 * scale)
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
    let mean_abs_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f32>()
        / actual.len().max(1) as f32;
    let mean_abs_tolerance = tolerance * 0.25;
    assert!(
        mean_abs_error <= mean_abs_tolerance,
        "GQA SingleQ mean mismatch: mean_abs_error={mean_abs_error} mean_abs_tolerance={mean_abs_tolerance}"
    );
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual_value - expected_value).abs();
        assert!(
            diff <= tolerance,
            "GQA reference mismatch at {index}: expected={expected_value} actual={actual_value} diff={diff} \
             tolerance={tolerance}"
        );
    }
}

fn assert_quality(actual: &[f32], expected: &[f32], max_abs_budget: f32, mean_abs_budget: f32) {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs_error = 0.0_f32;
    let mut sum_abs_error = 0.0_f32;
    for (actual, expected) in actual.iter().zip(expected) {
        let error = (actual - expected).abs();
        max_abs_error = max_abs_error.max(error);
        sum_abs_error += error;
    }
    let mean_abs_error = sum_abs_error / actual.len().max(1) as f32;
    assert!(
        max_abs_error <= max_abs_budget && mean_abs_error <= mean_abs_budget,
        "GQA FP8 quality mismatch: max_abs_error={max_abs_error} max_abs_budget={max_abs_budget} \
         mean_abs_error={mean_abs_error} mean_abs_budget={mean_abs_budget}"
    );
}
