use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::metal::Dtype;

use super::RequestShape;
use super::Selection;
use super::Selector;
use crate::attn::gqa::batch_metadata::GQAReplayBucketPolicy;

fn selector() -> Selector {
    selector_for(backend_sdpa::Config {
        io_dtype: Dtype::Bfloat16,
        num_q_heads: 24,
        num_kv_heads: 4,
        head_dim: 256,
        tokens_per_page: 16,
    })
}

fn selector_for(config: backend_sdpa::Config) -> Selector {
    let registry = backend_sdpa::Registry::new(config);
    let partial_state_group_capacity = 128 * registry.max_q_tokens_per_map_task() as usize;
    Selector::new(registry, 128, 128, partial_state_group_capacity)
}

fn selection(token_indices: &[u32], request_tokens: &[u32]) -> Selection {
    let mut cu_tokens = vec![0];
    for &num_tokens in request_tokens {
        cu_tokens.push(cu_tokens.last().copied().unwrap() + num_tokens);
    }
    let request_shapes = RequestShape::from_batch(token_indices, &cu_tokens);
    let num_total_tokens = cu_tokens.last().copied().unwrap();
    selector().select(&request_shapes, &GQAReplayBucketPolicy::new(128, 128), num_total_tokens)
}

fn is_sliced_kv(selection: &Selection) -> bool {
    selection.variant().map.kind == backend_sdpa::MapKernelKind::SlicedKv
}

#[test]
fn test_lookup_preserves_f32_execution_capability() {
    for head_dim in [128, 256] {
        let config = backend_sdpa::Config {
            io_dtype: Dtype::Float32,
            num_q_heads: 24,
            num_kv_heads: 4,
            head_dim,
            tokens_per_page: 16,
        };
        let selector = Selector::for_lookup(
            config,
            128,
            inference_executor_core::attn::GQAPageTableLayout {
                num_req_slots: 4,
                num_gqa_layers: 1,
                num_blocks: 1024,
                num_page_ids_per_block: 1,
            },
        );
        let policy = GQAReplayBucketPolicy::new(128, selector.limits().max_map_task_templates);
        for num_history_tokens in [0, 4096] {
            for num_q_tokens in [1, 8] {
                let selection = selector.select(
                    &[RequestShape {
                        num_history_tokens,
                        num_q_tokens,
                    }],
                    &policy,
                    num_q_tokens,
                );
                assert!(is_sliced_kv(&selection));
                assert!(selection.variant().supports(config));
            }
        }
    }
}

#[test]
fn test_lookup_preserves_single_query_kv_parallelism_across_short_context_boundary() {
    for (num_q_heads, num_kv_heads, head_dim) in [(24, 4, 256), (16, 2, 256), (40, 8, 128)] {
        let selector = Selector::for_lookup(
            backend_sdpa::Config {
                io_dtype: Dtype::Bfloat16,
                num_q_heads,
                num_kv_heads,
                head_dim,
                tokens_per_page: 16,
            },
            128,
            inference_executor_core::attn::GQAPageTableLayout {
                num_req_slots: 8,
                num_gqa_layers: 1,
                num_blocks: 128,
                num_page_ids_per_block: 1,
            },
        );
        let policy = GQAReplayBucketPolicy::new(128, selector.limits().max_map_task_templates);
        for history in [0, 31, 32, 128, 384, 511, 512, 1024] {
            for requests in [1, 4, 8] {
                let shapes = vec![
                    RequestShape {
                        num_history_tokens: history,
                        num_q_tokens: 1
                    };
                    requests
                ];
                let result = selector.select(&shapes, &policy, requests as u32);
                if history < 32 {
                    assert!(result.direct_output());
                } else {
                    assert!(is_sliced_kv(&result));
                    assert!(!result.direct_output());
                    assert_eq!(result.variant().map.thread_block.required_threads, 128);
                }
            }
        }
    }
}

#[test]
fn test_lookup_covers_ragged_queries_and_continuous_context_within_replay_capacity() {
    for (dim, ratio) in [
        (128, 1),
        (128, 6),
        (128, 8),
        (128, 12),
        (256, 6),
        (256, 8),
        (256, 12),
        (256, 32),
    ] {
        let config = backend_sdpa::Config {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: 4 * ratio,
            num_kv_heads: 4,
            head_dim: dim,
            tokens_per_page: 16,
        };
        let selector = Selector::for_lookup(
            config,
            2048,
            inference_executor_core::attn::GQAPageTableLayout {
                num_req_slots: 8,
                num_gqa_layers: 1,
                num_blocks: 16400,
                num_page_ids_per_block: 1,
            },
        );
        let limits = selector.limits();
        let policy = GQAReplayBucketPolicy::new(2048, limits.max_map_task_templates);
        for history in [
            0_u32, 63, 64, 255, 256, 511, 512, 513, 1023, 2048, 4095, 4096, 4097, 8191, 16383, 32768, 65535, 98317,
            131073, 262144,
        ] {
            for queries in [1_u32, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128, 256] {
                for requests in [1_u32, 2, 4, 8] {
                    let shapes: Vec<_> = (0..requests)
                        .map(|index| {
                            RequestShape {
                                num_history_tokens: history - history.min(index),
                                num_q_tokens: queries - index.min(queries - 1),
                            }
                        })
                        .collect();
                    let result = selector.select(&shapes, &policy, 2048);
                    assert!(result.replay_shape().num_total_sdpa_map_task_templates <= limits.max_map_task_templates);
                    assert!(
                        result.metrics().num_replay_reserved_partial_state_groups
                            <= limits.partial_state_group_capacity as u64
                    );
                    let mut end_by_range = vec![0; result.q_token_ranges().len()];
                    for map in result.map_task_templates() {
                        let end = &mut end_by_range[map.q_token_range_index as usize];
                        assert_eq!(map.request_local_kv_token_indices.start, *end);
                        *end = map.request_local_kv_token_indices.end;
                    }
                    for (range, end) in result.q_token_ranges().iter().zip(end_by_range) {
                        assert_eq!(range.num_visible_kv_tokens, end);
                    }
                }
            }
        }
    }
}

#[test]
fn test_partition_envelope_keeps_variant_maps_and_token_tiles_coupled() {
    let config = backend_sdpa::Config {
        io_dtype: Dtype::Bfloat16,
        num_q_heads: 24,
        num_kv_heads: 4,
        head_dim: 256,
        tokens_per_page: 16,
    };
    let single = backend_sdpa::ExecutionVariant::sliced_kv(config, 256, 128, 6);
    let sliced_q = backend_sdpa::ExecutionVariant::sliced_q_per_head(config, 8, 16, 6);
    let registry = backend_sdpa::Registry::from_variants(config, vec![single, sliced_q]);
    let selector = Selector::for_partitions(registry.clone(), 128, 8, |_| 32768_u32.div_ceil(512));
    assert_eq!(selector.limits().max_map_task_templates, 8192);
    // 23 sliced_q ranges * 64 splits pad to 1536 Maps, then reserve eight Q tokens each.
    assert_eq!(selector.limits().partial_state_group_capacity, 12288);
    assert!(selector.limits().partial_state_group_capacity < 8192 * 8);
    let selector = Selector::for_partitions(registry, 128, 8, |variant| {
        if variant.map.kind == backend_sdpa::MapKernelKind::SlicedKv {
            64
        } else {
            8
        }
    });
    assert_eq!(selector.limits().max_map_task_templates, 8192);
    assert_eq!(selector.limits().partial_state_group_capacity, 8192);
}

#[test]
fn test_partition_envelope_covers_positive_request_compositions() {
    fn visit(total: u32, parts: u32, prefix: &mut Vec<u32>, f: &mut impl FnMut(&[u32])) {
        if parts == 1 {
            prefix.push(total);
            f(prefix);
            prefix.pop();
        } else {
            for first in 1..=total - parts + 1 {
                prefix.push(first);
                visit(total - first, parts - 1, prefix, f);
                prefix.pop();
            }
        }
    }
    let config = backend_sdpa::Config {
        io_dtype: Dtype::Bfloat16,
        num_q_heads: 1,
        num_kv_heads: 1,
        head_dim: 128,
        tokens_per_page: 16,
    };
    let variants =
        [1, 2, 4, 8, 16, 32].map(|tile| backend_sdpa::ExecutionVariant::sliced_q_per_head(config, tile, 16, 1));
    let registry = backend_sdpa::Registry::from_variants(config, variants.to_vec());
    for max_tokens in 1..=16 {
        for max_requests in 1..=4 {
            let selector =
                Selector::for_partitions(registry.clone(), max_tokens, max_requests, |_| 130_u32.div_ceil(64));
            let limits = selector.limits();
            let policy = GQAReplayBucketPolicy::new(max_tokens, limits.max_map_task_templates);
            let mut max_observed_ranges = [0_u32; 6];
            for total in 1..=max_tokens {
                for requests in 1..=max_requests.min(total) {
                    visit(total, requests, &mut Vec::new(), &mut |tokens| {
                        let request_shapes = tokens
                            .iter()
                            .map(|&num_q_tokens| {
                                RequestShape {
                                    num_history_tokens: 130 - num_q_tokens,
                                    num_q_tokens,
                                }
                            })
                            .collect::<Vec<_>>();
                        for (i, &variant) in variants.iter().enumerate() {
                            let selection =
                                selector.select_partitioned(variant, 64, false, &request_shapes, &policy, total);
                            let shape = selection.replay_shape();
                            max_observed_ranges[i] = max_observed_ranges[i].max(shape.num_q_token_tiles);
                            assert!(shape.num_total_sdpa_map_task_templates <= limits.max_map_task_templates);
                            assert!(
                                selection.metrics().num_replay_reserved_partial_state_groups as usize
                                    <= limits.partial_state_group_capacity
                            );
                        }
                    });
                }
            }
            let requests = max_requests.min(max_tokens);
            for (i, variant) in variants.iter().enumerate() {
                assert_eq!(
                    max_observed_ranges[i],
                    requests + (max_tokens - requests) / variant.map.thread_block.max_q_tokens
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "GQA partition Map task-template count must fit u32")]
fn test_partition_envelope_rejects_map_count_overflow_before_allocation() {
    let config = backend_sdpa::Config {
        io_dtype: Dtype::Bfloat16,
        num_q_heads: 1,
        num_kv_heads: 1,
        head_dim: 128,
        tokens_per_page: 16,
    };
    let variant = backend_sdpa::ExecutionVariant::sliced_kv(config, 128, 128, 1);
    Selector::for_partitions(
        backend_sdpa::Registry::from_variants(config, vec![variant]),
        u32::MAX,
        1,
        |_| 2,
    );
}

#[test]
fn test_fixed_slices_cover_ragged_causal_ranges() {
    let config = backend_sdpa::Config {
        io_dtype: Dtype::Bfloat16,
        num_q_heads: 12,
        num_kv_heads: 2,
        head_dim: 128,
        tokens_per_page: 16,
    };
    let variant =
        backend_sdpa::ExecutionVariant::sliced_q(config, backend_sdpa::MapKernelKind::QTokensAndHeads, 4, 16, 6);
    let selector = Selector::new(
        backend_sdpa::Registry::from_variants(config, vec![variant]),
        16,
        64,
        256,
    );
    let requests = RequestShape::from_batch(&[120, 2], &[0, 5, 8]);
    let selection = selector.select_partitioned(variant, 64, false, &requests, &GQAReplayBucketPolicy::new(16, 64), 8);
    let ranges = selection.q_token_ranges();
    assert_eq!(
        ranges
            .iter()
            .map(|r| (r.request_index, r.flat_q_token_indices.clone(), r.num_visible_kv_tokens))
            .collect::<Vec<_>>(),
        vec![(0, 0..4, 124), (0, 4..5, 125), (1, 5..8, 5)]
    );
    assert_eq!(
        selection
            .map_task_templates()
            .iter()
            .map(|m| (m.q_token_range_index, m.request_local_kv_token_indices.clone()))
            .collect::<Vec<_>>(),
        vec![(0, 0..64), (0, 64..124), (1, 0..64), (1, 64..125), (2, 0..5)]
    );
    assert_eq!(selection.cu_partial_outputs_by_q_token_range(), &[0, 2, 4, 5]);
    assert_eq!(selection.metrics().max_kv_iterations_per_split, 4);
    assert!(!selection.direct_output());

    let selection = selector.select_partitioned(variant, 65, false, &requests, &GQAReplayBucketPolicy::new(16, 64), 8);
    assert_eq!(selection.q_token_ranges(), ranges);
    assert_eq!(
        selection
            .map_task_templates()
            .iter()
            .map(|m| (m.q_token_range_index, m.request_local_kv_token_indices.clone()))
            .collect::<Vec<_>>(),
        vec![(0, 0..65), (0, 65..124), (1, 0..65), (1, 65..125), (2, 0..5)]
    );
    assert_eq!(selection.cu_partial_outputs_by_q_token_range(), &[0, 2, 4, 5]);
    assert_eq!(selection.metrics().max_kv_iterations_per_split, 5);
    // Each split pads its own final staging tile, even when the complete range pads less.
    assert_eq!(selection.metrics().num_scheduled_qk_token_pairs, 1216);
    assert_eq!(selection.metrics().num_active_qk_token_pairs, 768);
}

#[test]
fn test_direct_output_preserves_padded_range_identity_without_partials() {
    let config = backend_sdpa::Config {
        io_dtype: Dtype::Bfloat16,
        num_q_heads: 1,
        num_kv_heads: 1,
        head_dim: 128,
        tokens_per_page: 16,
    };
    let variant = backend_sdpa::ExecutionVariant::sliced_q_per_head(config, 1, 16, 1);
    let selector = Selector::new(backend_sdpa::Registry::from_variants(config, vec![variant]), 9, 64, 1);
    let requests = RequestShape::from_batch(&[64, 2], &[0, 7, 9]);
    let selection = selector.select_partitioned(variant, 71, true, &requests, &GQAReplayBucketPolicy::new(9, 64), 9);
    let shape = selection.replay_shape();
    assert!(selection.direct_output());
    assert_eq!(shape.num_q_token_tiles, 9);
    assert_eq!(shape.num_sdpa_map_task_templates, 9);
    // The token-domain ceiling is 9. The independent Map ceiling would pad 9 to 12.
    assert_eq!(shape.num_total_q_token_tiles, shape.num_total_sdpa_map_task_templates);
    assert_eq!(selection.metrics().num_active_partial_state_groups, 0);
    assert_eq!(selection.metrics().num_replay_reserved_partial_state_groups, 0);
    for (range, map) in selection.q_token_ranges().iter().zip(selection.map_task_templates()) {
        assert_eq!(map.request_local_kv_token_indices, 0..range.num_visible_kv_tokens);
    }
}

#[test]
fn test_selector_preserves_measured_d256_page16_crossovers() {
    for (tokens, context) in [(1, 65536), (2, 65536), (4, 128), (8, 128), (25, 32)] {
        assert!(is_sliced_kv(&selection(&[context], &[tokens])));
    }
    for (tokens, context) in [(4, 512), (8, 512), (16, 128), (25, 128), (64, 0)] {
        assert!(!is_sliced_kv(&selection(&[context], &[tokens])));
    }
}

#[test]
fn test_selector_preserves_other_profile_selection() {
    let request_shapes = RequestShape::from_batch(&[0; 4], &[0, 2, 4, 6, 8]);
    let d256_page32 = selector_for(backend_sdpa::Config {
        io_dtype: Dtype::Bfloat16,
        num_q_heads: 24,
        num_kv_heads: 4,
        head_dim: 256,
        tokens_per_page: 32,
    })
    .select(&request_shapes, &GQAReplayBucketPolicy::new(128, 128), 8);
    assert_eq!(d256_page32.variant().map.thread_block.max_q_tokens, 8);
    assert_eq!(d256_page32.variant().map.thread_block.max_q_heads, 3);

    let d128_page16 = selector_for(backend_sdpa::Config {
        io_dtype: Dtype::Bfloat16,
        num_q_heads: 32,
        num_kv_heads: 4,
        head_dim: 128,
        tokens_per_page: 16,
    })
    .select(&request_shapes, &GQAReplayBucketPolicy::new(128, 128), 8);
    assert_eq!(d128_page16.variant().map.thread_block.max_q_tokens, 8);
    assert_eq!(d128_page16.variant().map.thread_block.max_q_heads, 8);

    let unsupported = selector_for(backend_sdpa::Config {
        io_dtype: Dtype::Bfloat16,
        num_q_heads: 32,
        num_kv_heads: 4,
        head_dim: 256,
        tokens_per_page: 4,
    })
    .select(&request_shapes, &GQAReplayBucketPolicy::new(128, 128), 8);
    assert_eq!(unsupported.variant().map.thread_block.max_q_tokens, 1);
}

#[test]
fn test_selector_prices_request_local_tail_ranges() {
    assert!(is_sliced_kv(&selection(&[65536; 8], &[1; 8])));
    assert!(is_sliced_kv(&selection(&[1024, 65536], &[64, 1])));
    assert!(is_sliced_kv(&selection(&[65536, 1024], &[1, 8])));
    assert!(!is_sliced_kv(&selection(&[65536, 65536], &[8, 1])));
}

#[test]
fn test_selection_materializes_current_sliced_q_capacity_and_metrics() {
    let tail_selection = selection(&[1024, 65536], &[64, 1]);
    assert!(is_sliced_kv(&tail_selection));

    let mixed_selection = selection(&[8192, 65536], &[64, 64]);
    let num_kv_splits = mixed_selection
        .cu_partial_outputs_by_q_token_range()
        .windows(2)
        .map(|offsets| offsets[1] - offsets[0])
        .collect::<Vec<_>>();
    assert_eq!(num_kv_splits, [vec![2; 8], vec![14; 8]].concat());

    let selection = selection(&[65536], &[25]);
    assert!(!is_sliced_kv(&selection));
    assert_eq!(selection.q_token_ranges().len(), 4);
    assert_eq!(selection.map_task_templates().len(), 128);
    assert_eq!(selection.replay_shape().num_total_sdpa_map_task_templates, 128);
    assert_eq!(selection.metrics().num_active_partial_state_groups, 800);
    assert_eq!(selection.metrics().num_active_partial_states, 800 * 24);
    assert_eq!(selection.metrics().num_reserved_partial_state_groups, 1024);
    assert_eq!(selection.metrics().num_replay_reserved_partial_state_groups, 1024);
}

#[test]
fn test_selection_map_tasks_cover_each_visible_kv_range_once() {
    for (history_tokens, q_tokens) in [
        (vec![1024], vec![25]),
        (vec![8192, 65536], vec![64, 64]),
        (vec![0], vec![1]),
        (vec![0], vec![128]),
        (vec![65536; 128], vec![1; 128]),
    ] {
        let selection = selection(&history_tokens, &q_tokens);
        let offsets = selection.cu_partial_outputs_by_q_token_range();
        let map = selection.variant().map.thread_block;
        let num_kv_iterations = selection
            .q_token_ranges()
            .iter()
            .map(|range| range.num_visible_kv_tokens.div_ceil(map.kv_tokens_per_iteration))
            .sum::<u32>();
        assert_eq!(
            selection.map_task_templates().len(),
            num_kv_iterations.min(128) as usize
        );
        for (range_index, range) in selection.q_token_ranges().iter().enumerate() {
            let templates =
                &selection.map_task_templates()[offsets[range_index] as usize..offsets[range_index + 1] as usize];
            assert_eq!(templates.first().unwrap().request_local_kv_token_indices.start, 0);
            assert_eq!(
                templates.last().unwrap().request_local_kv_token_indices.end,
                range.num_visible_kv_tokens
            );
            for pair in templates.windows(2) {
                assert_eq!(
                    pair[0].request_local_kv_token_indices.end,
                    pair[1].request_local_kv_token_indices.start
                );
            }
        }
    }
}

#[test]
fn test_dense_prefill_kv32_preserves_request_ranges_and_replay_capacity() {
    let registry = backend_sdpa::Registry::new(backend_sdpa::Config {
        io_dtype: Dtype::Bfloat16,
        num_q_heads: 24,
        num_kv_heads: 4,
        head_dim: 256,
        tokens_per_page: 16,
    });
    let partial_state_group_capacity = 2048 * registry.max_q_tokens_per_map_task() as usize;
    let selector = Selector::new(registry, 2048, 2048, partial_state_group_capacity);
    let policy = GQAReplayBucketPolicy::new(2048, 2048);
    for (histories, cu_tokens) in [
        (vec![0], vec![0, 1024]),
        (vec![4096], vec![0, 1024]),
        (vec![0, 16384], vec![0, 512, 1024]),
    ] {
        let requests = RequestShape::from_batch(&histories, &cu_tokens);
        let selection = selector.select(&requests, &policy, 2048);
        assert_eq!(selection.variant().map.thread_block.kv_tokens_per_iteration, 32);
        assert_eq!(selection.replay_shape().num_tokens, 1024);
        assert_eq!(selection.replay_shape().num_total_tokens, 2048);
        assert_eq!(selection.map_task_templates().len(), 256);
        for (index, range) in selection.q_token_ranges().iter().enumerate() {
            let offsets = selection.cu_partial_outputs_by_q_token_range();
            let templates = &selection.map_task_templates()[offsets[index] as usize..offsets[index + 1] as usize];
            assert_eq!(templates.first().unwrap().request_local_kv_token_indices.start, 0);
            assert_eq!(
                templates.last().unwrap().request_local_kv_token_indices.end,
                range.num_visible_kv_tokens
            );
            for pair in templates.windows(2) {
                assert_eq!(
                    pair[0].request_local_kv_token_indices.end,
                    pair[1].request_local_kv_token_indices.start
                );
            }
        }
    }
    let short = selector.select(&RequestShape::from_batch(&[4096], &[0, 256]), &policy, 2048);
    assert_eq!(short.variant().map.thread_block.kv_tokens_per_iteration, 16);
    let decode = selector.select(
        &RequestShape::from_batch(&vec![4096; 1024], &(0..=1024).collect::<Vec<_>>()),
        &policy,
        2048,
    );
    assert_eq!(decode.variant().map.thread_block.max_q_tokens, 1);
}

#[test]
fn test_map_capacity_preserves_token_domain_and_replay_padding() {
    let config = backend_sdpa::Config {
        io_dtype: Dtype::Bfloat16,
        num_q_heads: 1,
        num_kv_heads: 1,
        head_dim: 128,
        tokens_per_page: 8,
    };
    let variant = backend_sdpa::ExecutionVariant::sliced_kv(config, 64, 128, 1);
    let selector = Selector::new(backend_sdpa::Registry::from_variants(config, vec![variant]), 4, 32, 32);
    let policy = GQAReplayBucketPolicy::new(4, 32);
    let selection = selector.select(&RequestShape::from_batch(&[1088], &[0, 1]), &policy, 2);
    let shape = selection.replay_shape();
    assert_eq!(selector.limits().max_tokens, 4);
    assert_eq!(shape.num_tokens, 1);
    assert_eq!(shape.num_total_tokens, 2);
    assert_eq!(shape.num_sdpa_map_task_templates, 18);
    assert_eq!(shape.num_total_sdpa_map_task_templates, 20);
    assert_eq!(selection.metrics().num_replay_reserved_partial_state_groups, 20);
    assert_eq!(
        selection
            .map_task_templates()
            .last()
            .unwrap()
            .request_local_kv_token_indices,
        1088..1089
    );
    assert!(
        std::panic::catch_unwind(|| { selector.select(&RequestShape::from_batch(&[0], &[0, 5]), &policy, 4) }).is_err()
    );
}

#[test]
#[should_panic(expected = "GQA SDPA replay exceeds its partial-state-group capacity")]
fn test_partial_state_capacity_includes_replay_padding() {
    let config = backend_sdpa::Config {
        io_dtype: Dtype::Bfloat16,
        num_q_heads: 1,
        num_kv_heads: 1,
        head_dim: 128,
        tokens_per_page: 8,
    };
    let variant = backend_sdpa::ExecutionVariant::sliced_kv(config, 64, 128, 1);
    let selector = Selector::new(backend_sdpa::Registry::from_variants(config, vec![variant]), 4, 32, 18);
    selector.select(
        &RequestShape::from_batch(&[1088], &[0, 1]),
        &GQAReplayBucketPolicy::new(4, 32),
        2,
    );
}

#[test]
#[should_panic(expected = "GQA active token count must not exceed the total token count")]
fn test_total_token_count_rejects_active_overflow() {
    let request_shapes = RequestShape::from_batch(&[0], &[0, 5]);
    selector().select(&request_shapes, &GQAReplayBucketPolicy::new(128, 128), 4);
}

#[test]
#[should_panic(expected = "GQA total token count must not exceed the metadata capacity")]
fn test_total_token_count_rejects_metadata_overflow() {
    let request_shapes = RequestShape::from_batch(&[0], &[0, 5]);
    selector().select(&request_shapes, &GQAReplayBucketPolicy::new(128, 128), 129);
}
