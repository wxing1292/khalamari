use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Range;

use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_executor_core::attn::GQAReplayShape;

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
}

impl Selector {
    pub fn new(registry: backend_sdpa::Registry, max_tokens: usize) -> Self {
        assert!(max_tokens > 0, "GQA SDPA selector requires tokens");
        let max_map_task_templates = max_tokens
            .try_into()
            .expect("GQA SDPA Map task-template capacity must fit u32");
        // A full Map-template batch fits without a second partial-state budget.
        let partial_state_group_capacity = max_tokens
            .checked_mul(registry.max_q_tokens_per_map_task() as usize)
            .expect("GQA SDPA partial-state-group capacity must fit usize");
        Self {
            registry,
            limits: SelectorLimits {
                max_map_task_templates,
                partial_state_group_capacity,
            },
        }
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
        self.validate_policy(policy);
        assert!(!request_shapes.is_empty(), "GQA SDPA selector requires requests");
        let num_tokens = request_shapes
            .iter()
            .try_fold(0u32, |total, shape| total.checked_add(shape.num_q_tokens))
            .expect("GQA SDPA token count must fit u32");
        assert!(num_tokens > 0);
        assert!(num_tokens <= self.limits.max_map_task_templates);
        assert!(
            num_tokens <= num_total_tokens,
            "GQA active token count must not exceed the total token count"
        );
        assert!(
            num_total_tokens <= policy.max_tokens(),
            "GQA total token count must not exceed the metadata capacity"
        );

        let config = self.registry.config();
        let variants = self.registry.variants();
        let materialize =
            |variant| materialize_candidate(variant, config, request_shapes, self.limits, policy, num_total_tokens);
        let single_q = variants
            .iter()
            .copied()
            .find(|variant| variant.map.thread_block.max_q_tokens == 1);
        let mut tiled_variants = variants
            .iter()
            .copied()
            .filter(|variant| variant.map.thread_block.max_q_tokens > 1);
        let Some(first_tiled) = tiled_variants.next() else {
            return materialize(single_q.expect("GQA SDPA registry requires an execution"));
        };
        let Some(single_q) = single_q else {
            return materialize(first_tiled);
        };

        // Q-range density rejects variants before KV split allocation and ABI materialization.
        let num_q_token_ranges: u32 = request_shapes
            .iter()
            .map(|shape| shape.num_q_tokens.div_ceil(first_tiled.map.thread_block.max_q_tokens))
            .sum();
        if u64::from(num_tokens) < 2 * u64::from(num_q_token_ranges) {
            return materialize(single_q);
        }

        let full_q_heads = tiled_variants
            .map(|variant| variant.map.thread_block.max_q_heads)
            .fold(first_tiled.map.thread_block.max_q_heads, u32::max);
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
        let tiled_q = variants
            .iter()
            .copied()
            .filter(|variant| {
                variant.map.thread_block.max_q_tokens > 1 && variant.map.thread_block.max_q_heads == desired_q_heads
            })
            .max_by_key(|variant| variant.map.thread_block.kv_tokens_per_iteration == desired_kv_tokens)
            .unwrap_or(first_tiled);
        let tiled_q = materialize(tiled_q);
        if (config.head_dim, config.tokens_per_page) == (256, 16) {
            let single_q = materialize(single_q);
            if !prefer_d256_page16_tiled_q(&single_q, &tiled_q) {
                return single_q;
            }
        }
        tiled_q
    }

    fn validate_policy(&self, policy: &GQAReplayBucketPolicy) {
        assert_eq!(
            policy.max_tokens(),
            self.limits.max_map_task_templates,
            "GQA replay bucket policy must match the SDPA selection capacity"
        );
    }
}

#[derive(Clone)]
struct QTokenRangeWork {
    q_token_range: QTokenRange,
    num_kv_iterations: u32,
    num_kv_splits: u32,
}

fn materialize_candidate(
    variant: backend_sdpa::ExecutionVariant,
    config: backend_sdpa::Config,
    request_shapes: &[RequestShape],
    limits: SelectorLimits,
    policy: &GQAReplayBucketPolicy,
    num_total_tokens: u32,
) -> Selection {
    let map = variant.map.thread_block;
    let mut ranges = build_q_token_ranges(request_shapes, map.max_q_tokens, map.kv_tokens_per_iteration);
    // KV32 needs fewer partial outputs at the measured dense-prefill sizes.
    // The heap still assigns more splits to ranges with more visible KV work.
    let split_budget = if map.max_q_tokens > 1 && map.kv_tokens_per_iteration == 32 {
        (ranges.len() as u64 * 2).min(u64::from(limits.max_map_task_templates)) as u32
    } else {
        limits.max_map_task_templates
    };
    allocate_kv_splits(&mut ranges, split_budget);

    let mut q_token_ranges = Vec::with_capacity(ranges.len());
    let mut map_task_templates = Vec::new();
    let mut cu_partial_outputs_by_q_token_range = Vec::with_capacity(ranges.len() + 1);
    cu_partial_outputs_by_q_token_range.push(0);
    let mut max_kv_iterations_per_split = 0;
    for (q_token_range_index, range) in ranges.iter().enumerate() {
        q_token_ranges.push(range.q_token_range.clone());
        for kv_split_index in 0..range.num_kv_splits {
            let kv_iteration_begin =
                range.num_kv_iterations as u64 * u64::from(kv_split_index) / u64::from(range.num_kv_splits);
            let kv_iteration_end =
                range.num_kv_iterations as u64 * u64::from(kv_split_index + 1) / u64::from(range.num_kv_splits);
            let num_kv_iterations_per_split = (kv_iteration_end - kv_iteration_begin) as u32;
            max_kv_iterations_per_split = max_kv_iterations_per_split.max(num_kv_iterations_per_split);
            let kv_token_begin = kv_iteration_begin as u32 * map.kv_tokens_per_iteration;
            let kv_token_end = range
                .q_token_range
                .num_visible_kv_tokens
                .min(kv_iteration_end as u32 * map.kv_tokens_per_iteration);
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
    let (num_total_tokens, num_total_q_token_ranges, num_total_map_task_templates) =
        policy.capacities(num_total_tokens, num_q_token_ranges, num_map_task_templates);
    let replay_shape = GQAReplayShape::new(
        num_tokens,
        num_total_tokens,
        num_q_token_ranges,
        num_total_q_token_ranges,
        num_map_task_templates,
        num_total_map_task_templates,
        map.max_q_tokens > 1 || num_map_task_templates > num_tokens,
    );

    let num_active_partial_state_groups = ranges
        .iter()
        .map(|range| {
            u64::from(range.q_token_range.flat_q_token_indices.end - range.q_token_range.flat_q_token_indices.start)
                * u64::from(range.num_kv_splits)
        })
        .sum::<u64>();
    let num_reserved_partial_state_groups = u64::from(num_map_task_templates) * u64::from(map.max_q_tokens);
    let num_replay_reserved_partial_state_groups =
        u64::from(num_total_map_task_templates) * u64::from(map.max_q_tokens);
    assert!(
        num_reserved_partial_state_groups as usize <= limits.partial_state_group_capacity,
        "GQA SDPA selection exceeds its partial-state-group capacity"
    );
    let num_scheduled_qk_token_pairs = ranges
        .iter()
        .map(|range| {
            u64::from(range.num_kv_iterations) * u64::from(map.kv_tokens_per_iteration) * u64::from(map.max_q_tokens)
        })
        .sum();
    let num_active_qk_token_pairs = ranges
        .iter()
        .map(|range| {
            u64::from(range.num_kv_iterations)
                * u64::from(map.kv_tokens_per_iteration)
                * u64::from(
                    range.q_token_range.flat_q_token_indices.end - range.q_token_range.flat_q_token_indices.start,
                )
        })
        .sum();
    let num_q_head_ranges_per_kv_head = config.q_heads_per_kv_head().div_ceil(map.max_q_heads);
    let num_map_threadblocks_per_kv_head = u64::from(num_map_task_templates) * u64::from(num_q_head_ranges_per_kv_head);
    let num_map_simdgroup_waves_per_kv_head = num_map_threadblocks_per_kv_head * u64::from(map.required_threads / 32);
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

fn prefer_d256_page16_tiled_q(single_q: &Selection, tiled_q: &Selection) -> bool {
    let selection_score = single_q
        .metrics
        .num_logical_qk_token_pairs
        .checked_add(
            single_q
                .metrics
                .selection_overhead_score()
                .min(tiled_q.metrics.selection_overhead_score()),
        )
        .expect("GQA SDPA selection score must fit u64");
    let tiled_q_has_sufficient_map_utilization = tiled_q
        .metrics
        .num_active_qk_token_pairs
        .checked_mul(2)
        .is_some_and(|active_pairs| active_pairs >= tiled_q.metrics.num_scheduled_qk_token_pairs);
    tiled_q_has_sufficient_map_utilization && selection_score >= D256_PAGE16_MIN_SELECTION_SCORE
}

#[cfg(test)]
mod tests {
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
        Selector::new(backend_sdpa::Registry::new(config), 128)
    }

    fn selection(token_indices: &[u32], request_tokens: &[u32]) -> Selection {
        let mut cu_tokens = vec![0];
        for &num_tokens in request_tokens {
            cu_tokens.push(cu_tokens.last().copied().unwrap() + num_tokens);
        }
        let request_shapes = RequestShape::from_batch(token_indices, &cu_tokens);
        let num_total_tokens = cu_tokens.last().copied().unwrap();
        selector().select(&request_shapes, &GQAReplayBucketPolicy::new(128, &[]), num_total_tokens)
    }

    fn is_single_q(selection: &Selection) -> bool {
        selection.variant().map.thread_block.max_q_tokens == 1
    }

    #[test]
    fn test_selector_preserves_measured_d256_page16_crossovers() {
        for (tokens, context) in [(1, 65536), (2, 65536), (4, 128), (8, 128), (25, 32)] {
            assert!(is_single_q(&selection(&[context], &[tokens])));
        }
        for (tokens, context) in [(4, 512), (8, 512), (16, 128), (25, 128), (64, 0)] {
            assert!(!is_single_q(&selection(&[context], &[tokens])));
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
        .select(&request_shapes, &GQAReplayBucketPolicy::new(128, &[]), 8);
        assert_eq!(d256_page32.variant().map.thread_block.max_q_tokens, 8);
        assert_eq!(d256_page32.variant().map.thread_block.max_q_heads, 3);

        let d128_page16 = selector_for(backend_sdpa::Config {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: 32,
            num_kv_heads: 4,
            head_dim: 128,
            tokens_per_page: 16,
        })
        .select(&request_shapes, &GQAReplayBucketPolicy::new(128, &[]), 8);
        assert_eq!(d128_page16.variant().map.thread_block.max_q_tokens, 8);
        assert_eq!(d128_page16.variant().map.thread_block.max_q_heads, 8);

        let unsupported = selector_for(backend_sdpa::Config {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: 32,
            num_kv_heads: 4,
            head_dim: 256,
            tokens_per_page: 4,
        })
        .select(&request_shapes, &GQAReplayBucketPolicy::new(128, &[]), 8);
        assert_eq!(unsupported.variant().map.thread_block.max_q_tokens, 1);
    }

    #[test]
    fn test_selector_prices_request_local_tail_ranges() {
        assert!(is_single_q(&selection(&[65536; 8], &[1; 8])));
        assert!(is_single_q(&selection(&[1024, 65536], &[64, 1])));
        assert!(is_single_q(&selection(&[65536, 1024], &[1, 8])));
        assert!(!is_single_q(&selection(&[65536, 65536], &[8, 1])));
    }

    #[test]
    fn test_selection_materializes_current_tiled_q_capacity_and_metrics() {
        let tail_selection = selection(&[1024, 65536], &[64, 1]);
        assert!(is_single_q(&tail_selection));

        let mixed_selection = selection(&[8192, 65536], &[64, 64]);
        let num_kv_splits = mixed_selection
            .cu_partial_outputs_by_q_token_range()
            .windows(2)
            .map(|offsets| offsets[1] - offsets[0])
            .collect::<Vec<_>>();
        assert_eq!(num_kv_splits, [vec![2; 8], vec![14; 8]].concat());

        let selection = selection(&[65536], &[25]);
        assert!(!is_single_q(&selection));
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
        let selector = Selector::new(registry, 2048);
        let policy = GQAReplayBucketPolicy::new(2048, &[]);
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
    #[should_panic(expected = "GQA active token count must not exceed the total token count")]
    fn test_total_token_count_rejects_active_overflow() {
        let request_shapes = RequestShape::from_batch(&[0], &[0, 5]);
        selector().select(&request_shapes, &GQAReplayBucketPolicy::new(128, &[]), 4);
    }

    #[test]
    #[should_panic(expected = "GQA total token count must not exceed the metadata capacity")]
    fn test_total_token_count_rejects_metadata_overflow() {
        let request_shapes = RequestShape::from_batch(&[0], &[0, 5]);
        selector().select(&request_shapes, &GQAReplayBucketPolicy::new(128, &[]), 129);
    }
}
