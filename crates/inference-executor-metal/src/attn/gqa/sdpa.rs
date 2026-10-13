use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Range;

use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::replay::ReplayBucketPolicy;

use crate::attn::gqa::batch_metadata::GQAReplayBucketPolicy;

const D256_PAGE16_MIN_SELECTION_SCORE: u64 = 2048;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestShape {
    pub num_history_tokens: u32,
    pub num_q_tokens: u32,
}

impl RequestShape {
    pub fn from_batch(token_indices: &[u32], cu_tokens: &[u32]) -> Vec<Self> {
        assert_eq!(cu_tokens.len(), token_indices.len() + 1);
        assert_eq!(cu_tokens.first().copied(), Some(0));
        token_indices
            .iter()
            .zip(cu_tokens.windows(2))
            .map(|(&num_history_tokens, cu)| {
                assert!(cu[0] <= cu[1], "GQA batch cu_tokens must be nondecreasing");
                let num_q_tokens = cu[1] - cu[0];
                assert!(num_q_tokens > 0, "GQA requests must contain Q tokens");
                num_history_tokens
                    .checked_add(num_q_tokens)
                    .expect("GQA request context length must fit u32");
                Self {
                    num_history_tokens,
                    num_q_tokens,
                }
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectorLimits {
    pub max_tokens: u32,
    pub max_map_task_templates: u32,
    pub partial_state_group_capacity: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QTokenRange {
    pub request_index: u32,
    pub flat_q_token_indices: Range<u32>,
    /// KV-token count visible to the last Q token in this causal range.
    pub num_visible_kv_tokens: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MapTaskTemplate {
    pub q_token_range_index: u32,
    pub request_local_kv_token_indices: Range<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectionMetrics {
    pub num_scheduled_qk_token_pairs: u64,
    pub num_active_qk_token_pairs: u64,
    pub num_map_threadblocks_per_kv_head: u64,
    pub num_map_simdgroup_waves_per_kv_head: u64,
    pub num_active_partial_state_groups: u64,
    pub num_active_partial_states: u64,
    pub num_reserved_partial_state_groups: u64,
    pub num_replay_reserved_partial_state_groups: u64,
    pub max_kv_iterations_per_split: u32,
    pub num_logical_qk_token_pairs: u64,
}

impl SelectionMetrics {
    fn selection_overhead_score(self) -> u64 {
        self.num_map_threadblocks_per_kv_head
            .checked_add(self.num_map_simdgroup_waves_per_kv_head)
            .and_then(|score| score.checked_add(self.num_active_partial_state_groups))
            .and_then(|score| score.checked_add(self.num_replay_reserved_partial_state_groups))
            .expect("GQA SDPA selection overhead score must fit u64")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Selection {
    variant: backend_sdpa::ExecutionVariant,
    q_token_ranges: Vec<QTokenRange>,
    map_task_templates: Vec<MapTaskTemplate>,
    cu_partial_outputs_by_q_token_range: Vec<u32>,
    replay_shape: GQAReplayShape,
    metrics: SelectionMetrics,
}

impl Selection {
    pub fn variant(&self) -> backend_sdpa::ExecutionVariant {
        self.variant
    }

    pub fn direct_output(&self) -> bool {
        self.variant.map.kind != backend_sdpa::MapKernelKind::SlicedKv && !self.replay_shape.reduce_sdpa_partial_outputs
    }

    pub fn q_token_ranges(&self) -> &[QTokenRange] {
        &self.q_token_ranges
    }

    pub fn map_task_templates(&self) -> &[MapTaskTemplate] {
        &self.map_task_templates
    }

    pub fn cu_partial_outputs_by_q_token_range(&self) -> &[u32] {
        &self.cu_partial_outputs_by_q_token_range
    }

    pub fn replay_shape(&self) -> GQAReplayShape {
        self.replay_shape
    }

    pub fn metrics(&self) -> SelectionMetrics {
        self.metrics
    }
}

#[derive(Clone, Debug)]
pub struct Selector {
    registry: backend_sdpa::Registry,
    limits: SelectorLimits,
    lookup: bool,
}

impl Selector {
    pub fn new(
        registry: backend_sdpa::Registry,
        max_tokens: usize,
        max_map_task_templates: usize,
        partial_state_group_capacity: usize,
    ) -> Self {
        assert!(max_tokens > 0, "GQA SDPA selector requires tokens");
        assert!(
            max_map_task_templates > 0,
            "GQA SDPA selector requires Map task templates"
        );
        assert!(
            partial_state_group_capacity > 0,
            "GQA SDPA selector requires partial-state groups"
        );
        Self {
            registry,
            lookup: false,
            limits: SelectorLimits {
                max_tokens: max_tokens.try_into().expect("GQA SDPA token capacity must fit u32"),
                max_map_task_templates: max_map_task_templates
                    .try_into()
                    .expect("GQA SDPA Map task-template capacity must fit u32"),
                partial_state_group_capacity,
            },
        }
    }

    /// Initializes the shared resource envelope for explicit KV partitions.
    /// The policy supplies each variant's maximum splits per Q range over the configured context domain.
    /// This envelope applies to `select_partitioned`; automatic allocation has a separate split budget.
    pub fn for_partitions(
        registry: backend_sdpa::Registry,
        max_tokens: u32,
        max_requests: u32,
        max_kv_splits: impl Fn(backend_sdpa::ExecutionVariant) -> u32,
    ) -> Self {
        assert!(max_tokens > 0, "GQA partition envelope requires tokens");
        assert!(max_requests > 0, "GQA partition envelope requires requests");
        let max_requests = max_requests.min(max_tokens);
        let map_bounds = registry
            .variants()
            .iter()
            .map(|&variant| {
                let max_q_token_ranges =
                    max_requests + (max_tokens - max_requests) / variant.map.thread_block.max_q_tokens;
                let max_kv_splits = max_kv_splits(variant);
                assert!(max_kv_splits > 0, "GQA partition envelope requires KV splits");
                let max_map_task_templates = max_q_token_ranges
                    .checked_mul(max_kv_splits)
                    .expect("GQA partition Map task-template count must fit u32");
                (variant, max_map_task_templates)
            })
            .collect::<Vec<_>>();
        let max_map_task_templates = map_bounds.iter().map(|(_, maps)| *maps).max().unwrap();
        let map_buckets = ReplayBucketPolicy::new(max_map_task_templates);
        let partial_state_group_capacity = map_bounds
            .iter()
            .map(|(variant, maps)| {
                (map_buckets.capacity(*maps) as usize)
                    .checked_mul(variant.map.thread_block.max_q_tokens as usize)
                    .expect("GQA partition partial-state-group capacity must fit usize")
            })
            .max()
            .unwrap();
        Self::new(
            registry,
            max_tokens as usize,
            max_map_task_templates as usize,
            partial_state_group_capacity,
        )
    }

    /// Initializes the lookup routes and their reachable replay resource envelope.
    pub fn for_lookup(config: backend_sdpa::Config, max_tokens: usize, layout: GQAPageTableLayout) -> Self {
        config.validate();
        layout.validate();
        let max_tokens = u32::try_from(max_tokens).expect("GQA token capacity must fit u32");
        assert!(max_tokens > 0, "GQA lookup requires tokens");
        if config.io_dtype != Dtype::Bfloat16
            || !matches!(
                (config.head_dim, config.tokens_per_page),
                (128, 8 | 16) | (256, 8 | 16 | 32)
            )
            || config.q_heads_per_kv_head() > 32
        {
            let registry = backend_sdpa::Registry::new(config);
            let partials = (max_tokens as usize)
                .checked_mul(registry.max_q_tokens_per_map_task() as usize)
                .expect("GQA partial-state-group capacity must fit usize");
            return Self::new(registry, max_tokens as usize, max_tokens as usize, partials);
        }
        let max_visible_kv_tokens = layout
            .num_blocks
            .checked_mul(layout.num_page_ids_per_block)
            .and_then(|pages| pages.checked_mul(config.tokens_per_page))
            .expect("GQA page-table token capacity must fit u32");
        let mut variants = Vec::new();
        for query_capacity in [1_u32, 2, 4, 8, 16, 32] {
            if query_capacity > max_tokens.min(32).next_power_of_two() {
                break;
            }
            for end in [32_u32, 512, 4096, 8192] {
                for enough_decode_grid in [false, true] {
                    let variant = lookup_variant(config, query_capacity, end, enough_decode_grid);
                    if !variants.contains(&variant) {
                        variants.push(variant);
                    }
                }
            }
        }
        // Single-query slices are at least 256 tokens. Matrix threshold transitions
        // need at most ceil(512 / groups) splits before the final 512-token range.
        let max_splits = variants
            .iter()
            .map(|variant| {
                max_visible_kv_tokens
                    .div_ceil(256)
                    .max(512_u32.div_ceil(variant.map.thread_block.required_threads / 32))
            })
            .max()
            .unwrap();
        let max_maps = max_tokens.max(
            layout
                .num_req_slots
                .checked_mul(max_splits)
                .expect("GQA Map task-template capacity must fit u32"),
        );
        let registry = backend_sdpa::Registry::from_variants(config, variants);
        let max_partials = (max_maps as usize)
            .checked_mul(registry.max_q_tokens_per_map_task() as usize)
            .expect("GQA partial-state-group capacity must fit usize");
        let mut selector = Self::new(registry, max_tokens as usize, max_maps as usize, max_partials);
        selector.lookup = true;
        selector
    }

    pub fn limits(&self) -> SelectorLimits {
        self.limits
    }

    pub fn select(
        &self,
        request_shapes: &[RequestShape],
        policy: &GQAReplayBucketPolicy,
        num_total_tokens: u32,
    ) -> Selection {
        let num_tokens = self.validate_request_shapes(request_shapes, policy, num_total_tokens);

        let config = self.registry.config();
        let variants = self.registry.variants();
        if self.lookup {
            let (variant, mut slice, direct) = lookup_route(config, request_shapes);
            if !direct {
                // Sum ceil(visible / slice) <= ceil(sum_visible / slice) + ranges - 1.
                // Coarsen the KV slice to fit the initialized replay capacity.
                let tile = u64::from(variant.map.thread_block.max_q_tokens);
                let (num_ranges, visible) = request_shapes.iter().fold((0_u64, 0_u64), |(count, visible), shape| {
                    let queries = u64::from(shape.num_q_tokens);
                    let ranges = queries.div_ceil(tile);
                    let work =
                        ranges * u64::from(shape.num_history_tokens) + tile * ranges * (ranges - 1) / 2 + queries;
                    (count + ranges, visible + work)
                });
                let split_budget = u64::from(self.limits.max_map_task_templates) - num_ranges + 1;
                let required_slice = visible.div_ceil(split_budget);
                slice = u64::from(slice)
                    .max(required_slice)
                    .try_into()
                    .expect("GQA KV slice must fit u32");
            }
            debug_assert!(variants.contains(&variant));
            return self.materialize_candidate(variant, request_shapes, policy, num_total_tokens, Some(slice), direct);
        }
        let materialize =
            |variant| self.materialize_candidate(variant, request_shapes, policy, num_total_tokens, None, false);
        let sliced_kv = variants
            .iter()
            .copied()
            .find(|variant| variant.map.kind == backend_sdpa::MapKernelKind::SlicedKv);
        let mut sliced_q_variants = variants
            .iter()
            .copied()
            .filter(|variant| variant.map.kind != backend_sdpa::MapKernelKind::SlicedKv);
        let Some(first_sliced_q) = sliced_q_variants.next() else {
            return materialize(sliced_kv.expect("GQA SDPA registry requires an execution"));
        };
        let Some(sliced_kv) = sliced_kv else {
            return materialize(first_sliced_q);
        };

        // Q-range density rejects variants before KV split allocation and ABI materialization.
        let num_q_token_ranges: u32 = request_shapes
            .iter()
            .map(|shape| {
                shape
                    .num_q_tokens
                    .div_ceil(first_sliced_q.map.thread_block.max_q_tokens)
            })
            .sum();
        if u64::from(num_tokens) < 2 * u64::from(num_q_token_ranges) {
            return materialize(sliced_kv);
        }

        let full_q_heads = sliced_q_variants
            .map(|variant| variant.map.thread_block.max_q_heads)
            .fold(first_sliced_q.map.thread_block.max_q_heads, u32::max);
        let desired_q_heads = if !matches!((config.head_dim, config.tokens_per_page), (128, 8 | 16))
            && u64::from(num_tokens) < 4 * u64::from(num_q_token_ranges)
        {
            config.q_heads_per_kv_head().div_ceil(2).min(full_q_heads)
        } else {
            full_q_heads
        };
        // Large dense prefill amortizes a real KV32 tile. Retain the measured
        // KV16 selection for smaller or sparse request batches.
        let desired_kv_tokens = if config.head_dim == 256
            && num_tokens >= 1024
            && u64::from(num_tokens) >= 4 * u64::from(num_q_token_ranges)
        {
            32
        } else {
            16
        };
        let sliced_q = variants
            .iter()
            .copied()
            .filter(|variant| {
                variant.map.kind != backend_sdpa::MapKernelKind::SlicedKv
                    && variant.map.thread_block.max_q_heads == desired_q_heads
            })
            .max_by_key(|variant| variant.map.thread_block.kv_tokens_per_iteration == desired_kv_tokens)
            .unwrap_or(first_sliced_q);
        let sliced_q = materialize(sliced_q);
        if (config.head_dim, config.tokens_per_page) == (256, 16) {
            let sliced_kv = materialize(sliced_kv);
            if !prefer_d256_page16_sliced_q(&sliced_kv, &sliced_q) {
                return sliced_kv;
            }
        }
        sliced_q
    }

    /// Materializes one explicit kernel route and fixed KV slice.
    /// Direct output requires one complete KV slice per Q-token range.
    pub fn select_partitioned(
        &self,
        variant: backend_sdpa::ExecutionVariant,
        kv_slice: u32,
        direct_output: bool,
        request_shapes: &[RequestShape],
        policy: &GQAReplayBucketPolicy,
        num_total_tokens: u32,
    ) -> Selection {
        self.validate_request_shapes(request_shapes, policy, num_total_tokens);
        assert!(
            self.registry.variants().contains(&variant),
            "GQA route must belong to its initialized registry"
        );
        assert!(kv_slice > 0, "GQA KV slice must contain tokens");
        let max_visible_kv_tokens = request_shapes
            .iter()
            .map(|shape| {
                shape
                    .num_history_tokens
                    .checked_add(shape.num_q_tokens)
                    .expect("GQA request context length must fit u32")
            })
            .max()
            .unwrap();
        if direct_output {
            assert_ne!(
                variant.map.kind,
                backend_sdpa::MapKernelKind::SlicedKv,
                "GQA SlicedKv requires Map/Reduce output"
            );
            assert!(
                kv_slice >= max_visible_kv_tokens,
                "GQA direct output requires an unsplit KV range"
            );
        }
        self.materialize_candidate(
            variant,
            request_shapes,
            policy,
            num_total_tokens,
            Some(kv_slice),
            direct_output,
        )
    }

    fn validate_request_shapes(
        &self,
        request_shapes: &[RequestShape],
        policy: &GQAReplayBucketPolicy,
        num_total_tokens: u32,
    ) -> u32 {
        self.validate_policy(policy);
        assert!(!request_shapes.is_empty(), "GQA SDPA selector requires requests");
        let num_tokens = request_shapes
            .iter()
            .try_fold(0u32, |total, shape| total.checked_add(shape.num_q_tokens))
            .expect("GQA SDPA token count must fit u32");
        assert!(num_tokens > 0);
        assert!(num_tokens <= self.limits.max_tokens);
        assert!(
            num_tokens <= num_total_tokens,
            "GQA active token count must not exceed the total token count"
        );
        assert!(
            num_total_tokens <= policy.max_tokens(),
            "GQA total token count must not exceed the metadata capacity"
        );

        num_tokens
    }

    fn validate_policy(&self, policy: &GQAReplayBucketPolicy) {
        assert_eq!(
            policy.max_tokens(),
            self.limits.max_tokens,
            "GQA replay bucket policy must match the SDPA token capacity"
        );
        assert_eq!(
            policy.max_map_task_templates(),
            self.limits.max_map_task_templates,
            "GQA replay bucket policy must match the SDPA Map task-template capacity"
        );
    }
}

#[derive(Clone)]
struct QTokenRangeWork {
    q_token_range: QTokenRange,
    num_kv_iterations: u32,
    num_kv_splits: u32,
}

impl Selector {
    fn materialize_candidate(
        &self,
        variant: backend_sdpa::ExecutionVariant,
        request_shapes: &[RequestShape],
        policy: &GQAReplayBucketPolicy,
        num_total_tokens: u32,
        kv_slice: Option<u32>,
        direct_output: bool,
    ) -> Selection {
        let config = self.registry.config();
        let limits = self.limits;
        let map = variant.map.thread_block;
        let mut ranges = build_q_token_ranges(request_shapes, map.max_q_tokens, map.kv_tokens_per_iteration);
        if let Some(kv_slice) = kv_slice {
            let mut num_map_task_templates = 0_u64;
            for range in &mut ranges {
                range.num_kv_splits = range.q_token_range.num_visible_kv_tokens.div_ceil(kv_slice);
                num_map_task_templates += u64::from(range.num_kv_splits);
            }
            assert!(
                num_map_task_templates <= u64::from(limits.max_map_task_templates),
                "GQA fixed slices exceed the Map task-template capacity"
            );
        } else {
            // KV32 needs fewer partial outputs at the measured dense-prefill sizes.
            // The heap still assigns more splits to ranges with more visible KV work.
            let split_budget =
                if variant.map.kind != backend_sdpa::MapKernelKind::SlicedKv && map.kv_tokens_per_iteration == 32 {
                    (ranges.len() as u64 * 2).min(u64::from(limits.max_map_task_templates)) as u32
                } else {
                    limits.max_map_task_templates
                };
            allocate_kv_splits(&mut ranges, split_budget);
        }

        let mut q_token_ranges = Vec::with_capacity(ranges.len());
        let mut map_task_templates = Vec::new();
        let mut cu_partial_outputs_by_q_token_range = Vec::with_capacity(ranges.len() + 1);
        cu_partial_outputs_by_q_token_range.push(0);
        let mut max_kv_iterations_per_split = 0;
        for (q_token_range_index, range) in ranges.iter().enumerate() {
            q_token_ranges.push(range.q_token_range.clone());
            for kv_split_index in 0..range.num_kv_splits {
                let (kv_token_begin, kv_token_end) = if let Some(kv_slice) = kv_slice {
                    let begin = u64::from(kv_split_index) * u64::from(kv_slice);
                    let end = (begin + u64::from(kv_slice)).min(u64::from(range.q_token_range.num_visible_kv_tokens));
                    (begin as u32, end as u32)
                } else {
                    let kv_iteration_begin =
                        u64::from(range.num_kv_iterations) * u64::from(kv_split_index) / u64::from(range.num_kv_splits);
                    let kv_iteration_end = u64::from(range.num_kv_iterations) * u64::from(kv_split_index + 1)
                        / u64::from(range.num_kv_splits);
                    let begin = kv_iteration_begin as u32 * map.kv_tokens_per_iteration;
                    let end = range
                        .q_token_range
                        .num_visible_kv_tokens
                        .min(kv_iteration_end as u32 * map.kv_tokens_per_iteration);
                    (begin, end)
                };
                max_kv_iterations_per_split = max_kv_iterations_per_split
                    .max((kv_token_end - kv_token_begin).div_ceil(map.kv_tokens_per_iteration));
                map_task_templates.push(MapTaskTemplate {
                    q_token_range_index: q_token_range_index
                        .try_into()
                        .expect("GQA SDPA Q-token-range index must fit u32"),
                    request_local_kv_token_indices: kv_token_begin..kv_token_end,
                });
            }
            cu_partial_outputs_by_q_token_range.push(
                map_task_templates
                    .len()
                    .try_into()
                    .expect("GQA SDPA partial-output count must fit u32"),
            );
        }

        let num_q_token_ranges = q_token_ranges
            .len()
            .try_into()
            .expect("GQA SDPA Q-token-range count must fit u32");
        let num_map_task_templates = map_task_templates
            .len()
            .try_into()
            .expect("GQA SDPA Map task-template count must fit u32");
        assert!(num_map_task_templates <= limits.max_map_task_templates);
        let num_tokens = request_shapes.iter().map(|shape| shape.num_q_tokens).sum();
        let (num_total_tokens, num_total_q_token_ranges, mut num_total_map_task_templates) =
            policy.capacities(num_total_tokens, num_q_token_ranges, num_map_task_templates);
        if direct_output {
            // One direct Map owns each Q range. Both padded domains must have the same extent.
            num_total_map_task_templates = num_total_q_token_ranges;
            assert!(num_total_map_task_templates <= limits.max_map_task_templates);
        }
        let replay_shape = GQAReplayShape::new(
            num_tokens,
            num_total_tokens,
            num_q_token_ranges,
            num_total_q_token_ranges,
            num_map_task_templates,
            num_total_map_task_templates,
            !direct_output
                && (variant.map.kind != backend_sdpa::MapKernelKind::SlicedKv || num_map_task_templates > num_tokens),
        );

        let num_active_partial_state_groups = if direct_output {
            0
        } else {
            ranges
                .iter()
                .map(|range| {
                    u64::from(
                        range.q_token_range.flat_q_token_indices.end - range.q_token_range.flat_q_token_indices.start,
                    ) * u64::from(range.num_kv_splits)
                })
                .sum::<u64>()
        };
        let num_reserved_partial_state_groups = if direct_output {
            0
        } else {
            u64::from(num_map_task_templates) * u64::from(map.max_q_tokens)
        };
        let num_replay_reserved_partial_state_groups = if direct_output {
            0
        } else {
            u64::from(num_total_map_task_templates) * u64::from(map.max_q_tokens)
        };
        assert!(
            num_replay_reserved_partial_state_groups as usize <= limits.partial_state_group_capacity,
            "GQA SDPA replay exceeds its partial-state-group capacity"
        );
        let (num_scheduled_qk_token_pairs, num_active_qk_token_pairs) =
            map_task_templates
                .iter()
                .fold((0_u64, 0_u64), |(scheduled, active), template| {
                    let range = &q_token_ranges[template.q_token_range_index as usize];
                    let kv_tokens =
                        template.request_local_kv_token_indices.end - template.request_local_kv_token_indices.start;
                    let staged_kv_tokens = u64::from(kv_tokens.div_ceil(map.kv_tokens_per_iteration))
                        * u64::from(map.kv_tokens_per_iteration);
                    (
                        scheduled + staged_kv_tokens * u64::from(map.max_q_tokens),
                        active
                            + staged_kv_tokens
                                * u64::from(range.flat_q_token_indices.end - range.flat_q_token_indices.start),
                    )
                });
        let num_q_head_ranges_per_kv_head = config.q_heads_per_kv_head().div_ceil(map.max_q_heads);
        let num_map_threadblocks_per_kv_head =
            u64::from(num_map_task_templates) * u64::from(num_q_head_ranges_per_kv_head);
        let num_map_simdgroup_waves_per_kv_head =
            num_map_threadblocks_per_kv_head * u64::from(map.required_threads / 32);
        let num_logical_qk_token_pairs = request_shapes
            .iter()
            .map(|shape| {
                let num_q_tokens = u64::from(shape.num_q_tokens);
                num_q_tokens
                    .checked_mul(u64::from(shape.num_history_tokens) + 1)
                    .and_then(|pairs| {
                        num_q_tokens
                            .checked_mul(num_q_tokens.saturating_sub(1))
                            .and_then(|causal_pairs| pairs.checked_add(causal_pairs / 2))
                    })
                    .expect("GQA logical query-key pair count must fit u64")
            })
            .sum();
        let metrics = SelectionMetrics {
            num_scheduled_qk_token_pairs,
            num_active_qk_token_pairs,
            num_map_threadblocks_per_kv_head,
            num_map_simdgroup_waves_per_kv_head,
            num_active_partial_state_groups,
            num_active_partial_states: num_active_partial_state_groups
                .checked_mul(u64::from(config.num_q_heads))
                .expect("GQA active partial-state count must fit u64"),
            num_reserved_partial_state_groups,
            num_replay_reserved_partial_state_groups,
            max_kv_iterations_per_split,
            num_logical_qk_token_pairs,
        };

        Selection {
            variant,
            q_token_ranges,
            map_task_templates,
            cu_partial_outputs_by_q_token_range,
            replay_shape,
            metrics,
        }
    }
}

// Profiles: D128/G<=8, D128/G>8, D256/G<=8, D256/G>8.
// Columns: rounded per-request query capacities 1, 2, 4, 8, 16, 32.
// 0: D128 short query; 1: shared Q tokens/heads; 2: Q tokens per head; 3: single query.
const LOOKUP_STRATEGIES: [[u8; 6]; 4] = [
    [3, 0, 1, 1, 1, 1],
    [3, 1, 1, 1, 1, 1],
    [3, 1, 1, 2, 2, 2],
    [3, 1, 1, 1, 1, 1],
];

fn lookup_variant(
    config: backend_sdpa::Config,
    queries: u32,
    end: u32,
    enough_decode_grid: bool,
) -> backend_sdpa::ExecutionVariant {
    use backend_sdpa::MapKernelKind::QHeadsPerToken;
    use backend_sdpa::MapKernelKind::QTokensAndHeads;
    use backend_sdpa::MapKernelKind::QTokensPerHead;
    let ratio = config.q_heads_per_kv_head();
    let strategy =
        LOOKUP_STRATEGIES[usize::from(config.head_dim == 256) * 2 + usize::from(ratio > 8)][queries.ilog2() as usize];
    let (kind, tile, stage) = if end <= 32 || (queries > 1 && end <= 512) {
        (QTokensAndHeads, queries.min(1 << (32 / ratio).ilog2()), 16)
    } else {
        match strategy {
            0 if end <= 4096 => (QHeadsPerToken, queries, 16),
            1 => {
                let stage: u32 = if config.head_dim == 128 { 64 } else { 16 };
                let tile = queries.min(1 << (16384 / config.head_dim / ratio).ilog2());
                let joint = (tile * ratio).div_ceil(8);
                let head = tile * ratio.div_ceil(8);
                let kind = if head > joint && stage.div_ceil(joint) > 16 {
                    QHeadsPerToken
                } else {
                    QTokensAndHeads
                };
                (kind, tile, stage)
            },
            2 => (QTokensPerHead, 8, 16),
            3 if config.head_dim == 256 && ratio > 8 && enough_decode_grid => (QTokensAndHeads, 1, 16),
            0 | 3 => return backend_sdpa::ExecutionVariant::sliced_kv(config, config.head_dim, 128, ratio.min(8)),
            _ => unreachable!("invalid GQA lookup strategy"),
        }
    };
    backend_sdpa::ExecutionVariant::sliced_q(config, kind, tile, stage, ratio)
}

fn lookup_decode_slice(end: u32, work: u128, max_slice: u32) -> u32 {
    let slice = 1_u128 << (work / 1280).max(1).ilog2();
    slice.clamp(256, u128::from(max_slice)).min(u128::from(end)) as u32
}

fn lookup_route(
    config: backend_sdpa::Config,
    requests: &[RequestShape],
) -> (backend_sdpa::ExecutionVariant, u32, bool) {
    let queries = requests
        .iter()
        .map(|shape| shape.num_q_tokens)
        .max()
        .unwrap()
        .min(32)
        .next_power_of_two();
    let end = requests
        .iter()
        .map(|shape| shape.num_history_tokens + shape.num_q_tokens)
        .max()
        .unwrap();
    let visible: u128 = requests
        .iter()
        .map(|shape| u128::from(shape.num_history_tokens + shape.num_q_tokens))
        .sum();
    let ratio = config.q_heads_per_kv_head();
    let matrix_slice = lookup_decode_slice(end, visible * u128::from(config.num_kv_heads * ratio.div_ceil(8)), 512);
    let matrix_grid: u128 = requests
        .iter()
        .map(|shape| u128::from((shape.num_history_tokens + shape.num_q_tokens).div_ceil(matrix_slice)))
        .sum::<u128>()
        * u128::from(config.num_kv_heads * ratio.div_ceil(8));
    let variant = lookup_variant(config, queries, end, matrix_grid >= 160);
    let map = variant.map.thread_block;
    let strategy =
        LOOKUP_STRATEGIES[usize::from(config.head_dim == 256) * 2 + usize::from(ratio > 8)][queries.ilog2() as usize];
    let slice = if end <= 32 || (queries > 1 && end <= 512) {
        end
    } else if strategy == 3 {
        let max_slice = if variant.map.kind == backend_sdpa::MapKernelKind::SlicedKv {
            1024
        } else {
            512
        };
        lookup_decode_slice(
            end,
            visible
                * u128::from(config.num_kv_heads)
                * u128::from(ratio.div_ceil(map.max_q_heads))
                * u128::from(map.required_threads / 32),
            max_slice,
        )
    } else {
        let feature = u128::from(end).pow(2) * requests.len() as u128 * u128::from(map.required_threads / 32).pow(2);
        [(1_u128 << 26, 64), (1 << 30, 128), (1 << 34, 256), (u128::MAX, 512)]
            .into_iter()
            .find(|&(limit, _)| feature <= limit)
            .unwrap()
            .1
            .min(end)
    };
    let direct = variant.map.kind != backend_sdpa::MapKernelKind::SlicedKv && slice == end;
    (variant, slice, direct)
}

fn build_q_token_ranges(
    request_shapes: &[RequestShape],
    max_q_tokens: u32,
    kv_tokens_per_iteration: u32,
) -> Vec<QTokenRangeWork> {
    let mut ranges = Vec::new();
    let mut flat_request_begin = 0u32;
    for (request_index, shape) in request_shapes.iter().enumerate() {
        let flat_request_end = flat_request_begin
            .checked_add(shape.num_q_tokens)
            .expect("GQA flat Q-token end must fit u32");
        let mut flat_q_token_begin = flat_request_begin;
        while flat_q_token_begin < flat_request_end {
            let flat_q_token_end = flat_q_token_begin.saturating_add(max_q_tokens).min(flat_request_end);
            let num_visible_kv_tokens = shape
                .num_history_tokens
                .checked_add(flat_q_token_end - flat_request_begin)
                .expect("GQA visible KV-token count must fit u32");
            let num_kv_iterations = num_visible_kv_tokens.div_ceil(kv_tokens_per_iteration);
            // Validate the padded input range once, before materializing its KV splits.
            assert!(
                num_kv_iterations <= u32::MAX / kv_tokens_per_iteration,
                "GQA padded KV-token range must fit u32"
            );
            ranges.push(QTokenRangeWork {
                q_token_range: QTokenRange {
                    request_index: request_index.try_into().expect("GQA request index must fit u32"),
                    flat_q_token_indices: flat_q_token_begin..flat_q_token_end,
                    num_visible_kv_tokens,
                },
                num_kv_iterations,
                num_kv_splits: 1,
            });
            flat_q_token_begin = flat_q_token_end;
        }
        flat_request_begin = flat_request_end;
    }
    assert!(!ranges.is_empty());
    ranges
}

fn allocate_kv_splits(ranges: &mut [QTokenRangeWork], max_map_task_templates: u32) {
    let num_map_task_templates = ranges.len();
    debug_assert!(num_map_task_templates <= max_map_task_templates as usize);
    if num_map_task_templates == max_map_task_templates as usize {
        return;
    }
    if let [range] = ranges {
        range.num_kv_splits = range.num_kv_iterations.min(max_map_task_templates);
        return;
    }

    // Largest per-split KV work first; earlier Q ranges win ties.
    let mut candidates = ranges
        .iter()
        .enumerate()
        .filter(|(_, range)| range.num_kv_splits < range.num_kv_iterations)
        .map(|(index, range)| (range.num_kv_iterations.div_ceil(range.num_kv_splits), Reverse(index)))
        .collect::<BinaryHeap<_>>();

    for _ in num_map_task_templates..max_map_task_templates as usize {
        let Some((_, Reverse(index))) = candidates.pop() else {
            break;
        };
        let range = &mut ranges[index];
        range.num_kv_splits += 1;
        if range.num_kv_splits < range.num_kv_iterations {
            candidates.push((range.num_kv_iterations.div_ceil(range.num_kv_splits), Reverse(index)));
        }
    }
}

fn prefer_d256_page16_sliced_q(sliced_kv: &Selection, sliced_q: &Selection) -> bool {
    let selection_score = sliced_kv
        .metrics
        .num_logical_qk_token_pairs
        .checked_add(
            sliced_kv
                .metrics
                .selection_overhead_score()
                .min(sliced_q.metrics.selection_overhead_score()),
        )
        .expect("GQA SDPA selection score must fit u64");
    let sliced_q_has_sufficient_map_utilization = sliced_q
        .metrics
        .num_active_qk_token_pairs
        .checked_mul(2)
        .is_some_and(|active_pairs| active_pairs >= sliced_q.metrics.num_scheduled_qk_token_pairs);
    sliced_q_has_sufficient_map_utilization && selection_score >= D256_PAGE16_MIN_SELECTION_SCORE
}

#[cfg(test)]
#[path = "sdpa_test.rs"]
mod tests;
