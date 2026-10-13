//! History-attention execution selection for a fixed query-block width.

use std::cmp::Reverse;

use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::components::gqa::sdpa::ExecutionVariant;
use inference_executor_core::attn::BiDiBlockCapacity;

use crate::attn::bidi_block_gqa::capacity::BiDiBlockGQACapacity;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SelectionMetrics {
    history_kv_load_multiplicity_per_request: usize,
    kv_tokens_per_iteration: u32,
    padded_q_tokens_per_request: usize,
    partial_state_group_capacity: usize,
    max_q_heads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Selection {
    execution: ExecutionVariant,
    capacity: BiDiBlockGQACapacity,
    metrics: SelectionMetrics,
}

impl Selection {
    pub fn execution(self) -> ExecutionVariant {
        self.execution
    }

    pub fn capacity(self) -> BiDiBlockGQACapacity {
        self.capacity
    }
}

/// Selects one legal SplitKV execution for the fixed BiDiBlockGQA proposal width.
///
/// A multi-token proposal can reuse each history K/V tile across Q rows. The
/// selector compares the derived K/V load multiplicity, KV-iteration width,
/// padded Q rows, partial scratch extent, and Q-head coverage for every legal
/// execution.
pub struct Selector {
    registry: backend_sdpa::Registry,
    bidi_block_capacity: BiDiBlockCapacity,
}

impl Selector {
    pub fn new(registry: backend_sdpa::Registry, bidi_block_capacity: BiDiBlockCapacity) -> Self {
        bidi_block_capacity.validate();
        Self {
            registry,
            bidi_block_capacity,
        }
    }

    pub fn select(&self) -> Selection {
        let candidates = self
            .registry
            .variants()
            .iter()
            .copied()
            .map(|execution| self.materialize_candidate(execution));

        candidates
            .min_by_key(|selection| {
                let metrics = selection.metrics;
                (
                    metrics.history_kv_load_multiplicity_per_request,
                    Reverse(metrics.kv_tokens_per_iteration),
                    metrics.padded_q_tokens_per_request,
                    metrics.partial_state_group_capacity,
                    Reverse(metrics.max_q_heads),
                )
            })
            .expect("GQA SDPA registry must contain an execution variant")
    }

    fn materialize_candidate(&self, execution: ExecutionVariant) -> Selection {
        let map = execution.map.thread_block;
        let capacity = BiDiBlockGQACapacity::new(self.bidi_block_capacity, map.max_q_tokens);
        let q_token_ranges_per_request = self.bidi_block_capacity.block_size.div_ceil(capacity.max_q_tokens);
        let q_head_ranges_per_kv_head =
            (self.registry.config().q_heads_per_kv_head() as usize).div_ceil(map.max_q_heads as usize);
        Selection {
            execution,
            capacity,
            metrics: SelectionMetrics {
                history_kv_load_multiplicity_per_request: q_token_ranges_per_request
                    .checked_mul(q_head_ranges_per_kv_head)
                    .expect("BiDiBlockGQA history K/V load multiplicity must fit usize"),
                kv_tokens_per_iteration: map.kv_tokens_per_iteration,
                padded_q_tokens_per_request: q_token_ranges_per_request
                    .checked_mul(capacity.max_q_tokens)
                    .expect("BiDiBlockGQA padded Q-token count must fit usize"),
                partial_state_group_capacity: capacity.max_sdpa_partial_state_groups,
                max_q_heads: map.max_q_heads,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::metal::Dtype;

    use super::*;

    #[test]
    fn test_selector_uses_q_tile_only_when_it_reuses_history() {
        let sliced_kv = selector(config(128, 16), 1).select();
        let sliced_q = selector(config(128, 16), 7).select();

        assert_eq!(sliced_kv.execution().map.thread_block.max_q_tokens, 1);
        assert_eq!(sliced_q.execution().map.thread_block.max_q_tokens, 8);
        assert_eq!(sliced_q.capacity().max_q_tokens, 8);
    }

    #[test]
    fn test_selector_keeps_sliced_kv_for_unsupported_tiled_profile() {
        let selection = selector(config(128, 4), 7).select();

        assert_eq!(selection.execution().map.thread_block.max_q_tokens, 1);
        assert_eq!(selection.capacity().max_q_tokens, 1);
    }

    #[test]
    fn test_selector_minimizes_history_kv_loads() {
        let selection = selector(config(256, 32), 7).select();

        assert_eq!(selection.execution().map.thread_block.max_q_heads, 5);
    }

    fn selector(config: backend_sdpa::Config, block_size: usize) -> Selector {
        Selector::new(
            backend_sdpa::Registry::new(config),
            BiDiBlockCapacity::new(2, block_size),
        )
    }

    fn config(head_dim: u32, tokens_per_page: u32) -> backend_sdpa::Config {
        backend_sdpa::Config {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: 40,
            num_kv_heads: 8,
            head_dim,
            tokens_per_page,
        }
    }
}
