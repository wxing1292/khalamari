use half::bf16;
use inference_executor_core::attn::gqa::GQACore;
use inference_executor_core::attn::gqa::reference::GQAReferenceInput;
use inference_executor_core::attn::gqa::reference::projected_gqa_reference;

use super::*;
use crate::components::gqa::fp8::bf16_to_f32;
use crate::components::gqa::fp8::bf16_to_fp8_e4m3;
use crate::components::gqa::fp8::f32_to_bf16;
use crate::components::gqa::fp8::fp8_e4m3_to_bf16;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::Stream;
use crate::test_support::ReplayTestCache;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.gqa.sdpa_sliced_q.num_active_tokens");
const NUM_ACTIVE_Q_TOKEN_TILES: ReplayParameterKey =
    ReplayParameterKey::new("test.gqa.sliced_q.num_active_q_token_tiles");
const NUM_ACTIVE_KV_SPLITS: ReplayParameterKey = ReplayParameterKey::new("test.gqa.sliced_q.num_active_kv_splits");

#[test]
fn test_pipelines_reused_across_replay_capacities() {
    let device = Device::system_default();
    let (config, shape) = sliced_q_workload(128, 8);
    let execution = sliced_q_execution(config);
    let kernels = Compute::new(&device, config, execution, shape);
    for (num_total_tokens, num_total_q_token_tiles, num_total_sdpa_map_task_templates) in
        [(4, 1, 2), (8, 2, 4), (16, 2, 8)]
    {
        let other = Compute::new(
            &device,
            config,
            execution,
            Shape {
                num_total_tokens,
                num_total_q_token_tiles,
                num_total_sdpa_map_task_templates,
            },
        );
        assert!(std::ptr::eq(kernels.map.as_raw(), other.map.as_raw()));
        assert!(std::ptr::eq(
            kernels.reduce.as_ref().unwrap().as_raw(),
            other.reduce.as_ref().unwrap().as_raw()
        ));
    }
}

#[test]
fn test_replay_bucketing() {
    assert_replay_bucketing();
}

#[test]
fn test_replay_paged_ranges() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    for (head_dim, tokens_per_page) in [(128, 8), (128, 16), (256, 8), (256, 16), (256, 32)] {
        for kv_tokens_per_iteration in [8, 16, 32] {
            let config = Config {
                num_q_heads: 6,
                num_kv_heads: 2,
                head_dim,
                scale: (head_dim as f32).sqrt().recip(),
                page_bytes: 4 * tokens_per_page * head_dim,
                dtype: Dtype::Bfloat16,
                page_table_layout: PageTableLayout {
                    num_req_slots: 2,
                    num_gqa_layers: 2,
                    num_blocks: 3,
                    num_page_ids_per_block: 2,
                },
            };
            let shape = Shape {
                num_total_tokens: 20,
                num_total_q_token_tiles: 4,
                num_total_sdpa_map_task_templates: 8,
            };
            let execution =
                sdpa::ExecutionVariant::sliced_q_per_head(config.sdpa_config(), 16, kv_tokens_per_iteration, 2);
            let kernels = Compute::new(&device, config, execution, shape);
            let q_values = generated_bf16_values(config.q_bytes(shape) as usize / 2, 17);
            let kv_width = (config.num_kv_heads * head_dim) as usize;
            let keys = [36, 7].map(|tokens| fp8_e4m3_round_trip(&generated_bf16_values(tokens * kv_width, 31)));
            let values = [36, 7].map(|tokens| fp8_e4m3_round_trip(&generated_bf16_values(tokens * kv_width, 47)));
            let mut page_table = vec![u32::MAX; 24];
            let mut pages = vec![0_u8; 12 * config.page_bytes as usize];
            let mut next_page = 11;
            for (req, req_slot) in [1_usize, 0].into_iter().enumerate() {
                let page_values = tokens_per_page as usize * kv_width;
                for (page_index, (k, v)) in keys[req]
                    .chunks(page_values)
                    .zip(values[req].chunks(page_values))
                    .enumerate()
                {
                    let mut page_k = vec![0.0; page_values];
                    let mut page_v = vec![0.0; page_values];
                    page_k[..k.len()].copy_from_slice(k);
                    page_v[..v.len()].copy_from_slice(v);
                    let page = kv_page_values(config, &page_k, &page_v);
                    let begin = next_page * config.page_bytes as usize;
                    pages[begin..begin + page.len()].copy_from_slice(&page);
                    page_table[req_slot * 12 + 6 + page_index] = next_page as u32;
                    next_page -= 1;
                }
            }
            let q = bf16_buffer(&device, &q_values);
            let kv_pages = Buffer::from_slice(&device, &pages);
            let page_ids = Buffer::from_slice(&device, &page_table);
            let mut slots = vec![1_u32; 20];
            slots[17] = 0;
            let req_slots = Buffer::from_slice(&device, &slots);
            let visible_ranges = Buffer::new_zeroed_elements(&device, 40, Dtype::Uint32);
            let ranges = Buffer::from_slice(&device, &[0_u32, 16, 16, 17, 17, 18, u32::MAX, u32::MAX]);
            let templates = Buffer::from_slice(
                &device,
                &[
                    0_u32,
                    0,
                    16,
                    0,
                    16,
                    35,
                    1,
                    0,
                    16,
                    1,
                    16,
                    36,
                    2,
                    0,
                    7,
                    2,
                    7,
                    7,
                    u32::MAX,
                    u32::MAX,
                    u32::MAX,
                    u32::MAX,
                    u32::MAX,
                    u32::MAX,
                ],
            );
            let cu_partials = Buffer::from_slice(&device, &[0_u32, 2, 4, 6, u32::MAX]);
            let partial_output = Buffer::new_zeroed(&device, config.partial_output_bytes(execution, shape));
            let partial_sums = Buffer::new_zeroed(&device, config.partial_output_stats_bytes(execution, shape));
            let partial_max = Buffer::new_zeroed(&device, config.partial_output_stats_bytes(execution, shape));
            let output = bf16_buffer(&device, &vec![42.0; q_values.len()]);
            let mut builder = stream.create_replay_program();
            builder.record(kernels.invoke_map(
                MapBuffers {
                    q: &q,
                    kv_pages: &kv_pages,
                    req_slots: &req_slots,
                    page_ids: &page_ids,
                    visible_kv_token_ranges: &visible_ranges,
                    q_token_ranges: &ranges,
                    sdpa_map_task_templates: &templates,
                    partial_output: &partial_output,
                    partial_exp_sums: &partial_sums,
                    partial_max_logits: &partial_max,
                },
                ReplayU32::Fixed(1),
                ReplayU32::Fixed(18),
                ReplayU32::Fixed(3),
                ReplayU32::Fixed(6),
            ));
            builder.record_with_barrier_before(kernels.invoke_reduce(
                ReduceBuffers {
                    partial_output: &partial_output,
                    partial_exp_sums: &partial_sums,
                    partial_max_logits: &partial_max,
                    q_token_ranges: &ranges,
                    cu_sdpa_partial_outputs: &cu_partials,
                    output: &output,
                },
                ReplayU32::Fixed(3),
            ));
            let replay = builder.build();
            let q_width = (config.num_q_heads * head_dim) as usize;
            for windowed in [false, true] {
                let mut visibility = Vec::new();
                let mut expected = Vec::new();
                for token in 0..18 {
                    let req = usize::from(token == 17);
                    let end = if req == 0 { 20 + token } else { 7 };
                    let begin = if windowed {
                        if req == 0 { end - 3 } else { end }
                    } else {
                        0
                    };
                    visibility.extend([begin as u32, end as u32]);
                    if begin == end {
                        expected.extend(vec![0.0; q_width]);
                    } else {
                        expected.extend(projected_gqa_reference(
                            &fixture_core(config),
                            GQAReferenceInput {
                                cu_tokens: &[0, 1],
                                token_indices: &[(end - begin - 1) as u32],
                                q: &q_values[token * q_width..(token + 1) * q_width],
                                context_k_by_req: &[&keys[req][begin * kv_width..end * kv_width]],
                                context_v_by_req: &[&values[req][begin * kv_width..end * kv_width]],
                            },
                        ));
                    }
                }
                visible_ranges.write_typed(0, &visibility);
                stream.submit_replay(&replay).wait();
                assert_close(&read_bf16_values(&output, 18 * q_width), &expected, 2.0e-2);
                assert!(
                    output
                        .read_typed::<u16>(18 * q_width, 2 * q_width)
                        .iter()
                        .all(|&value| value == bf16::from_f32(42.0).to_bits())
                );
            }
        }
    }
}

#[test]
fn test_q_token_head_assignments_and_direct_output() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    for (kind, ratio, tile, heads, stage, head_dim) in [
        (sdpa::MapKernelKind::QHeadsPerToken, 6, 4, 6, 8, 128),
        (sdpa::MapKernelKind::QTokensAndHeads, 6, 4, 6, 16, 128),
        (sdpa::MapKernelKind::QTokensAndHeads, 6, 4, 3, 32, 256),
        (sdpa::MapKernelKind::QHeadsPerToken, 8, 2, 4, 64, 256),
        (sdpa::MapKernelKind::QTokensAndHeads, 12, 4, 12, 128, 256),
        (sdpa::MapKernelKind::QTokensPerHead, 12, 1, 6, 64, 128),
        (sdpa::MapKernelKind::QTokensPerHead, 6, 32, 1, 8, 128),
        (sdpa::MapKernelKind::QHeadsPerToken, 8, 32, 8, 128, 256),
        (sdpa::MapKernelKind::QTokensAndHeads, 12, 16, 12, 64, 256),
        (sdpa::MapKernelKind::QHeadsPerToken, 5, 4, 3, 64, 256),
    ] {
        let context_tokens = if stage >= 64 { [280_usize, 257] } else { [24, 9] };
        let blocks = context_tokens[0].div_ceil(8);
        let num_pages = 2 * blocks;
        let config = Config {
            num_q_heads: ratio * 2,
            num_kv_heads: 2,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            page_bytes: 4 * 8 * head_dim,
            dtype: Dtype::Bfloat16,
            page_table_layout: PageTableLayout {
                num_req_slots: 2,
                num_gqa_layers: 1,
                num_blocks: blocks as u32,
                num_page_ids_per_block: 1,
            },
        };
        let execution = sdpa::ExecutionVariant::sliced_q(config.sdpa_config(), kind, tile, stage, heads);
        let q_width = (config.num_q_heads * head_dim) as usize;
        let kv_width = (config.num_kv_heads * head_dim) as usize;
        let q_values = generated_bf16_values(7 * q_width, 17);
        let keys = context_tokens.map(|n| fp8_e4m3_round_trip(&generated_bf16_values(n * kv_width, 31)));
        let values = context_tokens.map(|n| fp8_e4m3_round_trip(&generated_bf16_values(n * kv_width, 47)));
        let mut pages = vec![0_u8; num_pages * config.page_bytes as usize];
        for req in 0..2 {
            for page in 0..keys[req].len().div_ceil(8 * kv_width) {
                let begin = page * 8 * kv_width;
                let end = (begin + 8 * kv_width).min(keys[req].len());
                let mut k = vec![0.0; 8 * kv_width];
                let mut v = k.clone();
                k[..end - begin].copy_from_slice(&keys[req][begin..end]);
                v[..end - begin].copy_from_slice(&values[req][begin..end]);
                let bytes = kv_page_values(config, &k, &v);
                let physical = num_pages - 1 - (req * blocks + page);
                pages[physical * bytes.len()..(physical + 1) * bytes.len()].copy_from_slice(&bytes);
            }
        }
        let mut visibility = Vec::new();
        let mut expected = Vec::new();
        for token in 0..7 {
            let req = usize::from(token >= 5);
            let end = if req == 0 {
                context_tokens[0] - 4 + token
            } else {
                context_tokens[1] - 1 + token - 5
            };
            let begin = if token == 6 {
                end
            } else if token == 0 {
                0
            } else {
                end - 7
            };
            visibility.extend([begin as u32, end as u32]);
            if begin == end {
                expected.extend(vec![0.0; q_width]);
            } else {
                expected.extend(projected_gqa_reference(
                    &fixture_core(config),
                    GQAReferenceInput {
                        cu_tokens: &[0, 1],
                        token_indices: &[(end - begin - 1) as u32],
                        q: &q_values[token * q_width..(token + 1) * q_width],
                        context_k_by_req: &[&keys[req][begin * kv_width..end * kv_width]],
                        context_v_by_req: &[&values[req][begin * kv_width..end * kv_width]],
                    },
                ));
            }
        }
        let q = bf16_buffer(&device, &q_values);
        let kv_pages = Buffer::from_slice(&device, &pages);
        let page_ids = Buffer::from_slice(&device, &(0..num_pages as u32).rev().collect::<Vec<_>>());
        let req_slots = Buffer::from_slice(&device, &[0_u32, 0, 0, 0, 0, 1, 1]);
        let visible = Buffer::from_slice(&device, &visibility);
        let mut q_ranges = Vec::new();
        for (begin, end) in [(0_u32, 5_u32), (5, 7)] {
            for first in (begin..end).step_by(tile as usize) {
                q_ranges.extend([first, (first + tile).min(end)]);
            }
        }
        let ranges = Buffer::from_slice(&device, &q_ranges);
        for direct in [false, true] {
            let mut templates = Vec::new();
            let mut cu = vec![0_u32];
            for (index, range) in q_ranges.as_chunks::<2>().0.iter().enumerate() {
                let end = if range[0] < 5 {
                    context_tokens[0] as u32 - 5 + range[1]
                } else {
                    context_tokens[1] as u32 - 6 + range[1]
                };
                if direct {
                    templates.extend([index as u32, 0, end]);
                } else {
                    templates.extend([index as u32, 0, 8, index as u32, 8, end]);
                }
                cu.push((templates.len() / 3) as u32);
            }
            let shape = Shape {
                num_total_tokens: 7,
                num_total_q_token_tiles: (q_ranges.len() / 2) as u32,
                num_total_sdpa_map_task_templates: (templates.len() / 3) as u32,
            };
            let templates = Buffer::from_slice(&device, &templates);
            let cu = Buffer::from_slice(&device, &cu);
            let output = bf16_buffer(&device, &vec![42.0; q_values.len()]);
            let kernels = if direct {
                Compute::new_direct(&device, config, execution, shape)
            } else {
                Compute::new(&device, config, execution, shape)
            };
            let mut builder = stream.create_replay_program();
            if direct {
                builder.record(kernels.invoke_direct(
                    DirectBuffers {
                        q: &q,
                        kv_pages: &kv_pages,
                        req_slots: &req_slots,
                        page_ids: &page_ids,
                        visible_kv_token_ranges: &visible,
                        q_token_ranges: &ranges,
                        sdpa_map_task_templates: &templates,
                        output: &output,
                    },
                    ReplayU32::Fixed(0),
                    ReplayU32::Fixed(7),
                    ReplayU32::Fixed(shape.num_total_q_token_tiles),
                ));
            } else {
                let partial = Buffer::new_zeroed(&device, config.partial_output_bytes(execution, shape));
                let sums = Buffer::new_zeroed(&device, config.partial_output_stats_bytes(execution, shape));
                let maxs = Buffer::new_zeroed(&device, config.partial_output_stats_bytes(execution, shape));
                builder.record(kernels.invoke_map(
                    MapBuffers {
                        q: &q,
                        kv_pages: &kv_pages,
                        req_slots: &req_slots,
                        page_ids: &page_ids,
                        visible_kv_token_ranges: &visible,
                        q_token_ranges: &ranges,
                        sdpa_map_task_templates: &templates,
                        partial_output: &partial,
                        partial_exp_sums: &sums,
                        partial_max_logits: &maxs,
                    },
                    ReplayU32::Fixed(0),
                    ReplayU32::Fixed(7),
                    ReplayU32::Fixed(shape.num_total_q_token_tiles),
                    ReplayU32::Fixed(shape.num_total_sdpa_map_task_templates),
                ));
                builder.record_with_barrier_before(kernels.invoke_reduce(
                    ReduceBuffers {
                        partial_output: &partial,
                        partial_exp_sums: &sums,
                        partial_max_logits: &maxs,
                        q_token_ranges: &ranges,
                        cu_sdpa_partial_outputs: &cu,
                        output: &output,
                    },
                    ReplayU32::Fixed(shape.num_total_q_token_tiles),
                ));
            }
            stream.submit_replay(&builder.build()).wait();
            assert_close(&read_bf16_values(&output, q_values.len()), &expected, 2.0e-2);
        }
    }
}

#[test]
#[should_panic(expected = "GQA SplitKV SlicedQ supports only")]
fn test_shape_rejects_unsupported_profile() {
    let (config, shape) = sliced_q_workload(192, 8);
    shape.validate(config);
}

#[test]
#[should_panic(expected = "GQA SplitKV SlicedQ Q-token-range metadata exceeds the shader u32 element-index domain")]
fn test_shape_rejects_shader_index_overflow() {
    let shape = Shape {
        num_total_tokens: u32::MAX,
        num_total_q_token_tiles: u32::MAX,
        num_total_sdpa_map_task_templates: u32::MAX,
    };
    let (config, _) = sliced_q_workload(256, 16);
    shape.validate(config);
}

#[test]
#[should_panic(
    expected = "GQA SplitKV SlicedQ visible K/V-token-range metadata exceeds the shader u32 element-index domain"
)]
fn test_shape_rejects_visible_range_index_overflow() {
    let shape = Shape {
        num_total_tokens: u32::MAX,
        num_total_q_token_tiles: 1,
        num_total_sdpa_map_task_templates: 1,
    };
    let (config, _) = sliced_q_workload(256, 16);
    shape.validate(config);
}

// Force both implementations independently of the production hardware policy.
fn assert_replay_bucketing() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let (config, base_shape) = sliced_q_workload(128, 8);
    let shape = Shape {
        num_total_tokens: 4,
        num_total_sdpa_map_task_templates: 2,
        ..base_shape
    };
    let execution = sliced_q_execution(config);
    let kernels = Compute::new(&device, config, execution, shape);
    let q_values = generated_bf16_values(config.q_bytes(shape) as usize / size_of::<u16>(), 0x4751_4154);
    let kv_values_per_kind =
        config.num_tokens_per_page() as usize * config.num_kv_heads as usize * config.head_dim as usize;
    let k_values = generated_bf16_values(kv_values_per_kind, 0x4751_4155);
    let v_values = generated_bf16_values(kv_values_per_kind, 0x4751_4156);
    let q = bf16_buffer(&device, &q_values);
    let kv_pages = Buffer::from_slice(&device, &kv_page_values(config, &k_values, &v_values));
    let req_slots = Buffer::new_zeroed_elements(&device, shape.num_total_tokens as usize, Dtype::Uint32);
    let page_ids = Buffer::from_slice(&device, &[0_u32]);
    let visible_kv_token_ranges =
        Buffer::new_zeroed_elements(&device, shape.num_total_tokens as usize * 2, Dtype::Uint32);
    let q_token_ranges = Buffer::new_zeroed_elements(&device, 2, Dtype::Uint32);
    let sdpa_map_task_templates = Buffer::new_zeroed_elements(&device, 6, Dtype::Uint32);
    let cu_sdpa_partial_outputs = Buffer::from_slice(&device, &[0_u32, 2]);
    let partial_output = Buffer::new_zeroed(&device, config.partial_output_bytes(execution, shape));
    let partial_exp_sums = Buffer::new_zeroed(&device, config.partial_output_stats_bytes(execution, shape));
    let partial_max_logits = Buffer::new_zeroed(&device, config.partial_output_stats_bytes(execution, shape));
    let output = Buffer::new_zeroed(&device, config.q_bytes(shape));

    let cache_key = (
        shape.num_total_tokens,
        shape.num_total_q_token_tiles,
        shape.num_total_sdpa_map_task_templates,
    );
    let mut cache = ReplayTestCache::new();
    let (_, cache_hit) = cache.record(cache_key, || {
        let mut builder = stream.create_replay_program();
        builder.record(kernels.invoke_map(
            MapBuffers {
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
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            ReplayU32::Parameter(NUM_ACTIVE_Q_TOKEN_TILES),
            ReplayU32::Parameter(NUM_ACTIVE_KV_SPLITS),
        ));
        builder.record_with_barrier_before(kernels.invoke_reduce(
            ReduceBuffers {
                partial_output: &partial_output,
                partial_exp_sums: &partial_exp_sums,
                partial_max_logits: &partial_max_logits,
                q_token_ranges: &q_token_ranges,
                cu_sdpa_partial_outputs: &cu_sdpa_partial_outputs,
                output: &output,
            },
            ReplayU32::Parameter(NUM_ACTIVE_Q_TOKEN_TILES),
        ));
        builder.build()
    });
    assert!(!cache_hit);
    let q_values_per_token = config.num_q_heads as usize * config.head_dim as usize;
    let kv_values_per_token = config.num_kv_heads as usize * config.head_dim as usize;
    visible_kv_token_ranges.write_typed(
        0,
        &(0..shape.num_total_tokens)
            .flat_map(|q_token_index| [0, q_token_index + 1])
            .collect::<Vec<_>>(),
    );

    for num_active_tokens in [1_usize, 4, 3, 2] {
        q_token_ranges.write_typed(0, &[0_u32, num_active_tokens as u32]);
        // The first fixed-quota task has an empty history range. Map must
        // materialize its canonical empty partial, and Reduce must preserve
        // the result of the second task.
        sdpa_map_task_templates.write_typed(0, &[0_u32, 0, 0, 0, 0, num_active_tokens as u32]);
        let (replay, cache_hit) = cache.record(cache_key, || unreachable!());
        assert!(cache_hit);

        stream
            .submit_replay_with_arguments(
                replay,
                &ReplayArguments::new()
                    .with_u32(NUM_ACTIVE_TOKENS, num_active_tokens as u32)
                    .with_u32(NUM_ACTIVE_Q_TOKEN_TILES, 1)
                    .with_u32(NUM_ACTIVE_KV_SPLITS, 2),
            )
            .wait();

        let quantized_k = fp8_e4m3_round_trip(&k_values[..num_active_tokens * kv_values_per_token]);
        let quantized_v = fp8_e4m3_round_trip(&v_values[..num_active_tokens * kv_values_per_token]);
        let expected = projected_gqa_reference(
            &fixture_core(config),
            GQAReferenceInput {
                cu_tokens: &[0, num_active_tokens as u32],
                token_indices: &[0],
                q: &q_values[..num_active_tokens * q_values_per_token],
                context_k_by_req: &[&quantized_k],
                context_v_by_req: &[&quantized_v],
            },
        );
        let actual = read_bf16_values(&output, num_active_tokens * q_values_per_token);
        assert_close(&actual, &expected, 2.0e-2);
    }
}

fn sliced_q_workload(head_dim: u32, num_tokens_per_page: u32) -> (Config, Shape) {
    (
        Config {
            num_q_heads: 5,
            num_kv_heads: 1,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            page_bytes: 2 * num_tokens_per_page * head_dim * size_of::<u8>() as u32,
            dtype: Dtype::Bfloat16,
            page_table_layout: PageTableLayout {
                num_req_slots: 1,
                num_gqa_layers: 1,
                num_blocks: 1,
                num_page_ids_per_block: 1,
            },
        },
        Shape {
            num_total_tokens: 8,
            num_total_q_token_tiles: 1,
            num_total_sdpa_map_task_templates: 1,
        },
    )
}

fn sliced_q_execution(config: Config) -> sdpa::ExecutionVariant {
    sdpa::ExecutionVariant::sliced_q_per_head(config.sdpa_config(), 8, 16, 5)
}

fn fixture_core(config: Config) -> GQACore {
    GQACore::new(
        0,
        config.num_q_heads as usize * config.head_dim as usize,
        config.head_dim as usize,
        config.num_q_heads as usize,
        config.num_kv_heads as usize,
        config.scale,
    )
}

fn kv_page_values(config: Config, k: &[f32], v: &[f32]) -> Vec<u8> {
    assert_eq!(k.len(), v.len());
    let values_per_token = config.num_kv_heads as usize * config.head_dim as usize;
    let num_tokens_per_page = config.num_tokens_per_page() as usize;
    assert_eq!(k.len(), num_tokens_per_page * values_per_token);
    let mut page = vec![0_u8; 2 * k.len()];
    for token in 0..num_tokens_per_page {
        for kv_head in 0..config.num_kv_heads as usize {
            for dim in 0..config.head_dim as usize {
                let source = (token * config.num_kv_heads as usize + kv_head) * config.head_dim as usize + dim;
                let k_target = (kv_head * num_tokens_per_page + token) * config.head_dim as usize + dim;
                let v_target = ((config.num_kv_heads as usize + kv_head) * num_tokens_per_page + token)
                    * config.head_dim as usize
                    + dim;
                page[k_target] = bf16_to_fp8_e4m3(f32_to_bf16(k[source]));
                page[v_target] = bf16_to_fp8_e4m3(f32_to_bf16(v[source]));
            }
        }
    }
    page
}

fn fp8_e4m3_round_trip(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&value| bf16_to_f32(fp8_e4m3_to_bf16(bf16_to_fp8_e4m3(f32_to_bf16(value)))))
        .collect()
}

fn generated_bf16_values(count: usize, random_seed: u32) -> Vec<f32> {
    let mut state = random_seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bf16::from_f32(((state >> 8) as f32 / 16_777_216.0) - 0.5).to_f32()
        })
        .collect()
}

fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    Buffer::from_slice(device, &bf16_bits(values))
}

fn bf16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect()
}

fn read_bf16_values(buffer: &Buffer, count: usize) -> Vec<f32> {
    buffer
        .read_typed::<u16>(0, count)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
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
        "GQA SlicedQ mean mismatch: mean_abs_error={mean_abs_error} mean_abs_tolerance={mean_abs_tolerance}"
    );
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}
