use futures_lite::future::Boxed;

use crate::runtime::decoder::trie_cache::BlockMetadata;
use crate::runtime::decoder::trie_cache::EvictionEntry;
use crate::runtime::decoder::trie_cache::ImmutableBlock;
use crate::runtime::decoder::trie_cache::MutableBlock;
use crate::runtime::decoder::trie_cache::S3FIFOEntry;
use crate::runtime::decoder::trie_cache::SemiImmutableBlock;
use crate::runtime::decoder::trie_cache::TrieNodeKey;

pub enum AllocateMultiLaneMutableBlockResult<const N: usize, const L: usize> {
    Mutable { block_vec: [MutableBlock<N>; L] },
    ResourceLimitExceeded,
}

pub enum CommitMultiLaneMutableBlockResult<const P: usize, const L: usize, E: EvictionEntry> {
    Immutable { block_vec: [ImmutableBlock<P, E>; L] },
    ImmutableCollision { block_vec: [ImmutableBlock<P, E>; L] },
}

pub enum ReserveMultiLaneSemiImmutableBlockResult<const N: usize, const P: usize, const L: usize, E: EvictionEntry> {
    Mutable {
        block_vec: [MutableBlock<N>; L],
    },
    SemiImmutable {
        block_vec: [SemiImmutableBlock<N>; L],
    },
    Immutable {
        block_vec: [ImmutableBlock<P, E>; L],
    },
    Wait {
        block_metadata_vec: [BlockMetadata<N>; L],
        wait: Boxed<()>,
    },
    ResourceLimitExceeded {
        block_metadata_vec: [BlockMetadata<N>; L],
    },
}

pub enum CommitMultiLaneSemiImmutableBlockResult<const P: usize, const L: usize, E: EvictionEntry> {
    Immutable { block_vec: [ImmutableBlock<P, E>; L] },
    ImmutableCollision { block_vec: [ImmutableBlock<P, E>; L] },
}

#[mockall::automock(type Entry = S3FIFOEntry<TrieNodeKey>;)]
pub trait MultiLaneBlockCache<const P: usize, const L: usize>: Send + Sync + 'static {
    type Entry: EvictionEntry;

    fn try_evict(&self) -> bool;

    fn alloc_mutable_block<const N: usize>(&self) -> AllocateMultiLaneMutableBlockResult<N, L>;
    fn free_mutable_block<const N: usize>(&self, block_vec: [MutableBlock<N>; L]);
    fn commit_mutable_block<const N: usize>(
        &self,
        parent_trie_node_key_vec: [Option<TrieNodeKey>; L],
        block_vec: [MutableBlock<N>; L],
    ) -> CommitMultiLaneMutableBlockResult<P, L, Self::Entry>;

    fn reserve_semi_immutable_block<const N: usize>(
        &self,
        input_block_metadata_vec: [BlockMetadata<N>; L],
    ) -> ReserveMultiLaneSemiImmutableBlockResult<N, P, L, Self::Entry>;
    fn free_semi_immutable_block<const N: usize>(&self, block_vec: [SemiImmutableBlock<N>; L]);
    fn commit_semi_immutable_block<const N: usize>(
        &self,
        block_vec: [SemiImmutableBlock<N>; L],
    ) -> CommitMultiLaneSemiImmutableBlockResult<P, L, Self::Entry>;

    fn num_mutable_block(&self) -> usize;
    fn num_semi_immutable_block(&self) -> usize;
    fn num_pinned_immutable_block(&self) -> usize;
    fn num_unpinned_immutable_block(&self) -> usize;
}
