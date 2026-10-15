use std::collections::VecDeque;
use std::sync::Arc;

use ahash::AHashMap;
use ordered_float::NotNan;

use crate::compute::DecoderSyncBlocks;
use crate::compute::DeviceResourcePlacement;
use crate::compute::QueryTokens;
use crate::runtime::Token;
#[cfg(debug_assertions)]
use crate::runtime::decoder::trie_cache::DecoderBlock;
use crate::runtime::decoder::trie_cache::ImmutableBlock;
use crate::runtime::decoder::trie_cache::MultiLaneBlockCache;
use crate::runtime::decoder::trie_cache::MutableBlock;
use crate::runtime::decoder::trie_cache::SemiImmutableBlock;
use crate::runtime::resource::ConcreteResource;
use crate::runtime::resource::Resource;
use crate::runtime::resource::ResourceID;
use crate::runtime::resource::ResourcePlacement;

mod api;
#[cfg(all(test, debug_assertions))]
mod sanity_check_test;
mod token;

pub struct TrieDecoderBlocks<const N: usize, const P: usize, const L: usize, BC>
where
    BC: MultiLaneBlockCache<P, L>,
{
    block_cache: Arc<BC>,
    resources: AHashMap<ResourceID, Resource>,
    resource_placements: Vec<ResourcePlacement>,

    immutable_blocks: Vec<[ImmutableBlock<P, BC::Entry>; L]>,
    semi_immutable_blocks: VecDeque<[SemiImmutableBlock<N>; L]>,
    mutable_blocks: VecDeque<[MutableBlock<N>; L]>,

    num_history_tokens: usize,
    num_prompt_tokens: usize,
    queued_tokens: VecDeque<Token>,
    spec_tokens: Vec<Token>,
    spec_probs: Vec<NotNan<f32>>,
    spec_confidences: Vec<NotNan<f32>>,
    num_in_sync_blocks: usize,

    epoch: usize, // TODO there is no test
}

impl<const N: usize, const P: usize, const L: usize, BC> TrieDecoderBlocks<N, P, L, BC>
where
    BC: MultiLaneBlockCache<P, L>,
{
    const _ASSERT_N_GT_0: () = assert!(N > 0);
    const _ASSERT_N_FITS_RESOURCE_SEGMENT: () = assert!(N <= u16::MAX as usize);

    pub fn new(
        block_cache: Arc<BC>,
        resources: Vec<Resource>,
        resource_placements: Vec<ResourcePlacement>,
        history_tokens: impl IntoIterator<Item = Token>,
        prompt_tokens: impl IntoIterator<Item = Token>,
        sampled_tokens: impl IntoIterator<Item = Token>,
    ) -> Self {
        let resources = resources
            .into_iter()
            .map(|resource| (resource.id(), resource))
            .collect::<AHashMap<_, _>>();

        let mut queued_tokens: VecDeque<Token> = history_tokens.into_iter().collect();
        let num_history_tokens = queued_tokens.len();

        queued_tokens.extend(prompt_tokens);
        let num_prompt_tokens = queued_tokens.len() - num_history_tokens;

        queued_tokens.extend(sampled_tokens);
        assert!(!queued_tokens.is_empty());
        assert!(L - 1 <= queued_tokens.len());
        Self {
            block_cache,
            resources,
            resource_placements,

            immutable_blocks: vec![],
            semi_immutable_blocks: VecDeque::new(),
            mutable_blocks: VecDeque::new(),

            num_history_tokens,
            num_prompt_tokens,
            queued_tokens,
            spec_tokens: vec![],
            spec_probs: vec![],
            spec_confidences: vec![],
            num_in_sync_blocks: 0,

            epoch: 0,
        }
    }

    pub fn resource_symbolic_to_concrete(&mut self, resource: ConcreteResource) {
        let resource_id = resource.id();
        let owned_resource = self
            .resources
            .get_mut(&resource_id)
            .expect("resource materialization must reference a resource owned by its decoder blocks");
        assert!(
            owned_resource.is_symbolic(),
            "resource materialization must replace a symbolic resource"
        );
        assert_eq!(
            owned_resource.uri(),
            resource.uri(),
            "resource materialization must preserve the resource URI"
        );
        *owned_resource = Resource::Concrete(resource);
    }

    pub fn device_resource_placements(&self, query_tokens: &QueryTokens) -> Vec<DeviceResourcePlacement> {
        let query_token_start = query_tokens.token_index();
        let query_token_end = query_token_start + query_tokens.token_consumption();
        self.resource_placements
            .iter()
            .filter_map(|placement| {
                let intersects_query = placement.placements().iter().any(|&(token_index, _, len)| {
                    let token_end = token_index + len;
                    query_token_start.max(token_index) < query_token_end.min(token_end)
                });
                if !intersects_query {
                    return None;
                }
                let concrete = self
                    .resources
                    .get(&placement.resource_id())
                    .expect("device resource placement must reference an owned resource")
                    .concrete()
                    .expect("device resource placement requires a concrete resource");
                Some(DeviceResourcePlacement::new(
                    concrete.source().offset_bytes(),
                    concrete.source().len_bytes(),
                    placement.placements().to_vec(),
                ))
            })
            .collect()
    }

    fn unload_cached_resources(&mut self) {
        let cached_token_end = self.num_cached_tokens();
        let resources = &mut self.resources;
        for placement in &self.resource_placements {
            if placement.token_index_range().end > cached_token_end {
                continue;
            }

            let resource_id = placement.resource_id();
            if resources[&resource_id].is_symbolic() {
                continue;
            }

            let resource = resources.remove(&resource_id).unwrap();
            resources.insert(resource_id, resource.into_symbolic());
        }
    }

    #[cfg(debug_assertions)]
    pub fn sanity_check(&self) {
        let mut cached_tokens = vec![];
        let mut scheduled_tokens = vec![];
        let mut ready_tokens = vec![];
        let mut queued_tokens = vec![];
        let mut spec_tokens = vec![];
        let mut cache_lane_total_tokens = std::array::from_fn(|_| vec![]);

        for block_vec in self.immutable_blocks.iter() {
            sanity_check_materialized_block_vec(
                block_vec,
                &mut cached_tokens,
                &mut scheduled_tokens,
                &mut ready_tokens,
                &mut cache_lane_total_tokens,
            );
        }

        for block_vec in self.semi_immutable_blocks.iter() {
            sanity_check_materialized_block_vec(
                block_vec,
                &mut cached_tokens,
                &mut scheduled_tokens,
                &mut ready_tokens,
                &mut cache_lane_total_tokens,
            );
        }

        for block_vec in self.mutable_blocks.iter() {
            sanity_check_materialized_block_vec(
                block_vec,
                &mut cached_tokens,
                &mut scheduled_tokens,
                &mut ready_tokens,
                &mut cache_lane_total_tokens,
            );
        }

        queued_tokens.extend(self.queued_tokens.iter().copied());
        spec_tokens.extend(self.spec_tokens.iter().copied());
        debug_assert_eq!(
            self.spec_tokens.len(),
            self.spec_probs.len(),
            "speculative token and probability counts must match"
        );
        debug_assert_eq!(
            self.spec_tokens.len(),
            self.spec_confidences.len(),
            "speculative token and confidence counts must match"
        );

        debug_assert_eq!(self.num_cached_tokens(), cached_tokens.len());
        debug_assert_eq!(self.num_scheduled_tokens(), scheduled_tokens.len());
        debug_assert_eq!(self.num_ready_tokens(), ready_tokens.len());
        debug_assert_eq!(self.num_queued_tokens(), queued_tokens.len());
        debug_assert_eq!(self.num_spec_tokens(), spec_tokens.len());

        debug_assert_eq!(self.cached_tokens().collect::<Vec<_>>(), cached_tokens);
        debug_assert_eq!(self.scheduled_tokens().collect::<Vec<_>>(), scheduled_tokens);
        debug_assert_eq!(self.ready_tokens().collect::<Vec<_>>(), ready_tokens);
        debug_assert_eq!(self.queued_tokens().collect::<Vec<_>>(), queued_tokens);
        debug_assert_eq!(self.spec_tokens().collect::<Vec<_>>(), spec_tokens);

        debug_assert_eq!(
            self.num_history_tokens() + self.num_prompt_tokens() + self.num_sampled_tokens(),
            self.num_total_tokens(),
        );
        debug_assert_eq!(
            self.num_total_tokens(),
            cached_tokens.len() + scheduled_tokens.len() + ready_tokens.len() + queued_tokens.len()
        );
        debug_assert_eq!(self.num_spec_tokens(), spec_tokens.len());

        let total_tokens = self.total_tokens().collect::<Vec<_>>();
        debug_assert_eq!(
            self.history_tokens()
                .chain(self.prompt_tokens())
                .chain(self.sampled_tokens())
                .collect::<Vec<_>>(),
            total_tokens,
        );
        debug_assert_eq!(
            total_tokens,
            cached_tokens
                .iter()
                .chain(scheduled_tokens.iter())
                .chain(ready_tokens.iter())
                .chain(queued_tokens.iter())
                .copied()
                .collect::<Vec<_>>(),
        );

        sanity_check_cache_lane_token_windows(
            &cache_lane_total_tokens,
            &total_tokens,
            &spec_tokens,
            (self.immutable_blocks.len() + self.semi_immutable_blocks.len()) * N,
            cached_tokens.len(),
        );
    }

    pub fn prepare_blocks(&mut self) -> DecoderSyncBlocks {
        let num_in_sync_blocks = self.num_in_sync_blocks;

        let total_num_blocks =
            self.immutable_blocks.len() + self.semi_immutable_blocks.len() + self.mutable_blocks.len();
        debug_assert!(
            num_in_sync_blocks <= total_num_blocks,
            "num_in_sync_blocks={} exceeds total_num_blocks={}",
            num_in_sync_blocks,
            total_num_blocks
        );
        let num_blocks = total_num_blocks - num_in_sync_blocks;
        let mut kv_page_ids: Vec<Vec<Vec<u32>>> = (0..L).map(|_| Vec::with_capacity(num_blocks)).collect();
        let mut state_page_ids: Vec<Vec<Vec<u32>>> = (0..L).map(|_| Vec::with_capacity(num_blocks)).collect();
        if num_blocks != 0 {
            let immutable_start = num_in_sync_blocks.min(self.immutable_blocks.len());
            let semi_immutable_start = num_in_sync_blocks
                .saturating_sub(self.immutable_blocks.len())
                .min(self.semi_immutable_blocks.len());
            let mutable_start = num_in_sync_blocks
                .saturating_sub(self.immutable_blocks.len() + self.semi_immutable_blocks.len())
                .min(self.mutable_blocks.len());

            for block_vec in &self.immutable_blocks[immutable_start..] {
                for (lane, block) in block_vec.iter().enumerate() {
                    let block_ref = block.trie_node_ref();
                    kv_page_ids[lane].push(block_ref.kv_placement().page_ids().to_vec());
                    state_page_ids[lane].push(block_ref.state_placement().page_ids().to_vec());
                }
            }
            for block_vec in self.semi_immutable_blocks.iter().skip(semi_immutable_start) {
                for (lane, block) in block_vec.iter().enumerate() {
                    kv_page_ids[lane].push(block.kv_placement().page_ids().to_vec());
                    state_page_ids[lane].push(block.state_placement().page_ids().to_vec());
                }
            }
            for block_vec in self.mutable_blocks.iter().skip(mutable_start) {
                for (lane, block) in block_vec.iter().enumerate() {
                    kv_page_ids[lane].push(block.kv_placement().page_ids().to_vec());
                    state_page_ids[lane].push(block.state_placement().page_ids().to_vec());
                }
            }
        }
        self.num_in_sync_blocks = num_in_sync_blocks + num_blocks;
        DecoderSyncBlocks::new(num_in_sync_blocks, kv_page_ids, state_page_ids)
    }

    pub fn cancel_blocks(&mut self, sync_blocks: DecoderSyncBlocks) {
        self.num_in_sync_blocks = sync_blocks.block_index();
    }

    pub fn commit_blocks(&mut self, _sync_blocks: DecoderSyncBlocks) {
        // noop
    }
}

#[cfg(debug_assertions)]
fn sanity_check_materialized_block_vec<B, const L: usize>(
    block_vec: &[B; L],
    cached_tokens: &mut Vec<Token>,
    scheduled_tokens: &mut Vec<Token>,
    ready_tokens: &mut Vec<Token>,
    cache_lane_total_tokens: &mut [Vec<Token>; L],
) where
    B: DecoderBlock,
{
    let main_block = &block_vec[0];
    match (
        cached_tokens.is_empty(),
        scheduled_tokens.is_empty(),
        ready_tokens.is_empty(),
    ) {
        (_, true, true) => { /* noop */ },
        (_, false, true) => {
            debug_assert!(main_block.cached_tokens().is_empty());
        },
        (_, _, false) => {
            debug_assert!(main_block.cached_tokens().is_empty());
            debug_assert!(main_block.scheduled_tokens().is_empty());
        },
    }

    cached_tokens.extend_from_slice(main_block.cached_tokens());
    scheduled_tokens.extend_from_slice(main_block.scheduled_tokens());
    ready_tokens.extend_from_slice(main_block.ready_tokens());

    for (lane, block) in block_vec.iter().enumerate() {
        debug_assert_eq!(main_block.cached_tokens().len(), block.cached_tokens().len());
        debug_assert_eq!(main_block.scheduled_tokens().len(), block.scheduled_tokens().len());
        debug_assert_eq!(main_block.ready_tokens().len(), block.ready_tokens().len());
        debug_assert_eq!(main_block.total_tokens().len(), block.total_tokens().len());

        cache_lane_total_tokens[lane].extend_from_slice(block.total_tokens());
    }
}

#[cfg(debug_assertions)]
fn sanity_check_cache_lane_token_windows<const L: usize>(
    cache_lane_total_tokens: &[Vec<Token>; L],
    total_tokens: &[Token],
    spec_tokens: &[Token],
    num_fixed_tokens: usize,
    num_cached_tokens: usize,
) {
    debug_assert!(total_tokens.iter().all(|token| *token != Token::default()));
    let source = total_tokens.iter().chain(spec_tokens).copied().collect::<Vec<_>>();
    if num_cached_tokens > 0 && L > 1 {
        debug_assert_eq!(
            cache_lane_total_tokens[1][num_cached_tokens - 1],
            total_tokens[num_cached_tokens],
            "cached anchor changed"
        );
    }
    for (lane, tokens) in cache_lane_total_tokens.iter().enumerate() {
        // Immutable and semi-immutable token identities must not change.
        debug_assert_eq!(&tokens[..num_fixed_tokens], &source[lane..lane + num_fixed_tokens]);
        for (index, &token) in tokens.iter().enumerate().skip(num_fixed_tokens) {
            if index < num_cached_tokens {
                debug_assert_ne!(
                    token,
                    Token::default(),
                    "cached lane={lane} index={index} is a placeholder"
                );
            } else if let Some(&expected) = source.get(index + lane)
                && token != Token::default()
            {
                debug_assert_eq!(token, expected, "cache lane={lane} index={index}");
            }
        }
    }
    if num_cached_tokens > 0 {
        debug_assert_eq!(
            &cache_lane_total_tokens[0][..num_cached_tokens],
            &total_tokens[..num_cached_tokens]
        );
        // Adjacent cached lanes must overlap by one shifted token. New canonical
        // input may differ from the old speculative tail until commit.
        for (lane, tokens) in cache_lane_total_tokens.windows(2).enumerate() {
            debug_assert_eq!(
                &tokens[0][1..num_cached_tokens],
                &tokens[1][..num_cached_tokens - 1],
                "cached lane={} shifted window",
                lane + 1
            );
        }
    }
}

impl<const N: usize, const P: usize, const L: usize, BC> Drop for TrieDecoderBlocks<N, P, L, BC>
where
    BC: MultiLaneBlockCache<P, L>,
{
    fn drop(&mut self) {
        for block_vec in self.mutable_blocks.drain(..).rev() {
            self.block_cache.free_mutable_block(block_vec);
        }
        for block_vec in self.semi_immutable_blocks.drain(..).rev() {
            self.block_cache.free_semi_immutable_block(block_vec);
        }
    }
}
