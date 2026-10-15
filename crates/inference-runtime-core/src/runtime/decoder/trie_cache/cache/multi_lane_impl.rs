use std::mem::MaybeUninit;
use std::sync::Arc;

use event_listener::EventListener;
use futures_util::future::join_all;

use crate::runtime::decoder::KVBlockAllocator;
use crate::runtime::decoder::allocator::StateBlockAllocator;
use crate::runtime::decoder::trie_cache::AllocateMultiLaneMutableBlockResult;
use crate::runtime::decoder::trie_cache::AllocateSingleLaneMutableBlockResult;
use crate::runtime::decoder::trie_cache::BlockMetadata;
use crate::runtime::decoder::trie_cache::CommitMultiLaneMutableBlockResult;
use crate::runtime::decoder::trie_cache::CommitMultiLaneSemiImmutableBlockResult;
use crate::runtime::decoder::trie_cache::CommitSingleLaneMutableBlockResult;
use crate::runtime::decoder::trie_cache::CommitSingleLaneSemiImmutableBlockResult;
use crate::runtime::decoder::trie_cache::EvictionEntry;
use crate::runtime::decoder::trie_cache::ImmutableBlock;
use crate::runtime::decoder::trie_cache::MultiLaneBlockCache;
use crate::runtime::decoder::trie_cache::MutableBlock;
use crate::runtime::decoder::trie_cache::ReserveMultiLaneSemiImmutableBlockResult;
use crate::runtime::decoder::trie_cache::ReserveSingleLaneSemiImmutableBlockResult;
use crate::runtime::decoder::trie_cache::S3FIFOEntry;
use crate::runtime::decoder::trie_cache::SemiImmutableBlock;
use crate::runtime::decoder::trie_cache::SingleLaneBlockCache;
use crate::runtime::decoder::trie_cache::SingleLaneTrieBlockCache;
use crate::runtime::decoder::trie_cache::TrieNodeKey;

pub struct MultiLaneTrieBlockCache<const P: usize, const L: usize, KVA, SA, E = S3FIFOEntry<TrieNodeKey>>
where
    KVA: KVBlockAllocator,
    SA: StateBlockAllocator,
    E: EvictionEntry,
{
    block_cache_vec: [Arc<SingleLaneTrieBlockCache<P, KVA, SA, E>>; L],
}

impl<const P: usize, const L: usize, KVA, SA, E> MultiLaneTrieBlockCache<P, L, KVA, SA, E>
where
    KVA: KVBlockAllocator,
    SA: StateBlockAllocator,
    E: EvictionEntry,
{
    pub fn new(block_cache_vec: [Arc<SingleLaneTrieBlockCache<P, KVA, SA, E>>; L]) -> Self {
        Self { block_cache_vec }
    }
}

impl<const P: usize, const L: usize, KVA, SA, E> MultiLaneBlockCache<P, L> for MultiLaneTrieBlockCache<P, L, KVA, SA, E>
where
    KVA: KVBlockAllocator,
    SA: StateBlockAllocator,
    E: EvictionEntry,
{
    type Entry = E;

    fn try_evict(&self) -> bool {
        self.block_cache_vec
            .iter()
            .map(|cache| cache.try_evict())
            .fold(false, |acc, evicted| acc | evicted)
    }

    fn alloc_mutable_block<const N: usize>(&self) -> AllocateMultiLaneMutableBlockResult<N, L> {
        let mut block_vec: [MaybeUninit<MutableBlock<N>>; L] = new_uninit_array();

        for (lane, cache) in self.block_cache_vec.iter().enumerate() {
            let result = loop {
                match cache.alloc_mutable_block::<N>() {
                    AllocateSingleLaneMutableBlockResult::ResourceLimitExceeded if self.try_evict() => {},
                    result => break result,
                }
            };
            match result {
                AllocateSingleLaneMutableBlockResult::Mutable { block } => {
                    block_vec[lane].write(block);
                },
                AllocateSingleLaneMutableBlockResult::ResourceLimitExceeded => {
                    for (index, block) in block_vec.iter().enumerate().take(lane) {
                        let block = unsafe { block.assume_init_read() };
                        self.block_cache_vec[index].free_mutable_block(block);
                    }
                    return AllocateMultiLaneMutableBlockResult::ResourceLimitExceeded;
                },
            }
        }

        AllocateMultiLaneMutableBlockResult::Mutable {
            block_vec: unsafe { array_assume_init(block_vec) },
        }
    }

    fn free_mutable_block<const N: usize>(&self, block_vec: [MutableBlock<N>; L]) {
        for (block, cache) in block_vec.into_iter().zip(self.block_cache_vec.iter()) {
            cache.free_mutable_block(block);
        }
    }

    fn commit_mutable_block<const N: usize>(
        &self,
        parent_trie_node_key_vec: [Option<TrieNodeKey>; L],
        block_vec: [MutableBlock<N>; L],
    ) -> CommitMultiLaneMutableBlockResult<P, L, E> {
        let mut output_block_vec: [MaybeUninit<ImmutableBlock<P, E>>; L] = new_uninit_array();
        let mut collision = false;

        for (lane, (parent_trie_node_key, (block, cache))) in parent_trie_node_key_vec
            .into_iter()
            .zip(block_vec.into_iter().zip(self.block_cache_vec.iter()))
            .enumerate()
        {
            match cache.commit_mutable_block(parent_trie_node_key, block) {
                CommitSingleLaneMutableBlockResult::Immutable { block } => {
                    output_block_vec[lane].write(block);
                },
                CommitSingleLaneMutableBlockResult::ImmutableCollision { block } => {
                    collision = true;
                    output_block_vec[lane].write(block);
                },
            }
        }

        let output_block_vec = unsafe { array_assume_init(output_block_vec) };
        if collision {
            CommitMultiLaneMutableBlockResult::ImmutableCollision {
                block_vec: output_block_vec,
            }
        } else {
            CommitMultiLaneMutableBlockResult::Immutable {
                block_vec: output_block_vec,
            }
        }
    }

    fn reserve_semi_immutable_block<const N: usize>(
        &self,
        input_block_metadata_vec: [BlockMetadata<N>; L],
    ) -> ReserveMultiLaneSemiImmutableBlockResult<N, P, L, E> {
        let mut output_block_metadata_vec: Vec<(usize, BlockMetadata<N>)> = Vec::with_capacity(L);
        let mut waits: Vec<(usize, EventListener)> = Vec::with_capacity(L);
        let mut immutable_blocks: Vec<(usize, BlockMetadata<N>, ImmutableBlock<P, E>)> = Vec::with_capacity(L);
        let mut semi_immutable_blocks: Vec<(usize, BlockMetadata<N>, SemiImmutableBlock<N>)> = Vec::with_capacity(L);

        let mut iter = input_block_metadata_vec
            .into_iter()
            .zip(self.block_cache_vec.iter())
            .enumerate();
        for (lane, (block_metadata, cache)) in iter.by_ref() {
            let result = loop {
                match cache.reserve_semi_immutable_block(block_metadata.clone()) {
                    ReserveSingleLaneSemiImmutableBlockResult::ResourceLimitExceeded if self.try_evict() => {},
                    result => break result,
                }
            };
            match result {
                ReserveSingleLaneSemiImmutableBlockResult::SemiImmutable { block } => {
                    semi_immutable_blocks.push((lane, block_metadata, block));
                },
                ReserveSingleLaneSemiImmutableBlockResult::Immutable { block } => {
                    immutable_blocks.push((lane, block_metadata, block));
                },
                ReserveSingleLaneSemiImmutableBlockResult::Wait { wait } => {
                    output_block_metadata_vec.push((lane, block_metadata));
                    waits.push((lane, wait));
                },
                ReserveSingleLaneSemiImmutableBlockResult::ResourceLimitExceeded => {
                    output_block_metadata_vec.push((lane, block_metadata));
                    for (lane, block_metadata, block) in immutable_blocks.drain(..) {
                        drop(block);
                        output_block_metadata_vec.push((lane, block_metadata));
                    }
                    for (lane, block_metadata, block) in semi_immutable_blocks.drain(..) {
                        self.block_cache_vec[lane].free_semi_immutable_block(block);
                        output_block_metadata_vec.push((lane, block_metadata));
                    }
                    for (lane, (block_metadata, _)) in iter.by_ref() {
                        output_block_metadata_vec.push((lane, block_metadata));
                    }
                    output_block_metadata_vec.sort_by_key(|(lane, _)| *lane);
                    return ReserveMultiLaneSemiImmutableBlockResult::ResourceLimitExceeded {
                        block_metadata_vec: output_block_metadata_vec
                            .into_iter()
                            .map(|(_, block_metadata)| block_metadata)
                            .collect::<Vec<_>>()
                            .try_into()
                            .unwrap(),
                    };
                },
            }
        }

        if waits.len() == L {
            debug_assert!(waits.windows(2).all(|w| w[0].0 < w[1].0));
            debug_assert!(output_block_metadata_vec.windows(2).all(|w| w[0].0 < w[1].0));
            let block_metadata_vec: [BlockMetadata<N>; L] = output_block_metadata_vec
                .into_iter()
                .map(|(_, block_metadata)| block_metadata)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let wait_vec = waits.into_iter().map(|(_, wait)| wait).collect::<Vec<_>>();
            ReserveMultiLaneSemiImmutableBlockResult::Wait {
                block_metadata_vec,
                wait: Box::pin(async {
                    join_all(wait_vec).await;
                }),
            }
        } else if immutable_blocks.len() == L {
            debug_assert!(immutable_blocks.windows(2).all(|w| w[0].0 < w[1].0));
            let block_vec: [ImmutableBlock<P, E>; L] = immutable_blocks
                .into_iter()
                .map(|(_, _, block)| block)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            ReserveMultiLaneSemiImmutableBlockResult::Immutable { block_vec }
        } else if semi_immutable_blocks.len() == L {
            debug_assert!(semi_immutable_blocks.windows(2).all(|w| w[0].0 < w[1].0));
            let block_vec: [SemiImmutableBlock<N>; L] = semi_immutable_blocks
                .into_iter()
                .map(|(_, _, block)| block)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable { block_vec }
        } else {
            for (lane, block_metadata, block) in immutable_blocks.drain(..) {
                drop(block);
                output_block_metadata_vec.push((lane, block_metadata));
            }
            for (lane, block_metadata, block) in semi_immutable_blocks.drain(..) {
                self.block_cache_vec[lane].free_semi_immutable_block(block);
                output_block_metadata_vec.push((lane, block_metadata));
            }
            output_block_metadata_vec.sort_by_key(|(lane, _)| *lane);

            let block_metadata_vec: [BlockMetadata<N>; L] = output_block_metadata_vec
                .into_iter()
                .map(|(_, block_metadata)| block_metadata)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            match self.alloc_mutable_block::<N>() {
                AllocateMultiLaneMutableBlockResult::Mutable { mut block_vec } => {
                    for (block, block_metadata) in block_vec.iter_mut().zip(block_metadata_vec.iter()) {
                        block.insert_annotations(block_metadata.annotations().iter().cloned());
                    }
                    ReserveMultiLaneSemiImmutableBlockResult::Mutable { block_vec }
                },
                AllocateMultiLaneMutableBlockResult::ResourceLimitExceeded => {
                    ReserveMultiLaneSemiImmutableBlockResult::ResourceLimitExceeded { block_metadata_vec }
                },
            }
        }
    }

    fn free_semi_immutable_block<const N: usize>(&self, block_vec: [SemiImmutableBlock<N>; L]) {
        for (block, cache) in block_vec.into_iter().zip(self.block_cache_vec.iter()) {
            cache.free_semi_immutable_block(block);
        }
    }

    fn commit_semi_immutable_block<const N: usize>(
        &self,
        block_vec: [SemiImmutableBlock<N>; L],
    ) -> CommitMultiLaneSemiImmutableBlockResult<P, L, E> {
        let mut output_block_vec: [MaybeUninit<ImmutableBlock<P, E>>; L] = new_uninit_array();
        let mut collision = false;

        for (lane, (block, cache)) in block_vec.into_iter().zip(self.block_cache_vec.iter()).enumerate() {
            match cache.commit_semi_immutable_block(block) {
                CommitSingleLaneSemiImmutableBlockResult::Immutable { block } => {
                    output_block_vec[lane].write(block);
                },
                CommitSingleLaneSemiImmutableBlockResult::ImmutableCollision { block } => {
                    collision = true;
                    output_block_vec[lane].write(block);
                },
            }
        }

        let output_block_vec = unsafe { array_assume_init(output_block_vec) };
        if collision {
            CommitMultiLaneSemiImmutableBlockResult::ImmutableCollision {
                block_vec: output_block_vec,
            }
        } else {
            CommitMultiLaneSemiImmutableBlockResult::Immutable {
                block_vec: output_block_vec,
            }
        }
    }

    fn num_mutable_block(&self) -> usize {
        self.block_cache_vec.iter().map(|cache| cache.num_mutable_block()).sum()
    }

    fn num_semi_immutable_block(&self) -> usize {
        self.block_cache_vec
            .iter()
            .map(|cache| cache.num_semi_immutable_block())
            .sum()
    }

    fn num_pinned_immutable_block(&self) -> usize {
        self.block_cache_vec
            .iter()
            .map(|cache| cache.num_pinned_immutable_block())
            .sum()
    }

    fn num_unpinned_immutable_block(&self) -> usize {
        self.block_cache_vec
            .iter()
            .map(|cache| cache.num_unpinned_immutable_block())
            .sum()
    }
}

#[inline]
pub fn new_uninit_array<T, const L: usize>() -> [MaybeUninit<T>; L] {
    std::array::from_fn(|_| MaybeUninit::uninit())
}

#[inline]
unsafe fn array_assume_init<T, const L: usize>(maybe_uninit_array: [MaybeUninit<T>; L]) -> [T; L] {
    unsafe { std::ptr::read(&maybe_uninit_array as *const _ as *const [T; L]) }
}

#[path = "./multi_lane_test.rs"]
#[cfg(test)]
mod multi_lane_test;
