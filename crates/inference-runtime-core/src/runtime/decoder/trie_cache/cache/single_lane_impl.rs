use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use ahash::HashSet;
use ahash::HashSetExt;
use ahash::RandomState as AHashRandomState;
use dashmap::DashMap;
use dashmap::Entry as DashEntry;
use smallvec::SmallVec;

use crate::channel::Shutdown;
use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;
use crate::runtime::decoder::KVBlockAllocationResult;
use crate::runtime::decoder::KVBlockAllocator;
use crate::runtime::decoder::allocator::StateBlockAllocationResult;
use crate::runtime::decoder::allocator::StateBlockAllocator;
use crate::runtime::decoder::trie_cache::AllocateSingleLaneMutableBlockResult;
use crate::runtime::decoder::trie_cache::BlockMetadata;
use crate::runtime::decoder::trie_cache::CommitSingleLaneMutableBlockResult;
use crate::runtime::decoder::trie_cache::CommitSingleLaneSemiImmutableBlockResult;
use crate::runtime::decoder::trie_cache::EvictionEntry;
use crate::runtime::decoder::trie_cache::EvictionPolicy;
use crate::runtime::decoder::trie_cache::ImmutableBlock;
use crate::runtime::decoder::trie_cache::InsertNodeResult;
use crate::runtime::decoder::trie_cache::MutableBlock;
use crate::runtime::decoder::trie_cache::ReserveSingleLaneSemiImmutableBlockResult;
use crate::runtime::decoder::trie_cache::S3FIFOClient;
use crate::runtime::decoder::trie_cache::S3FIFOEntry;
use crate::runtime::decoder::trie_cache::SemiImmutableBlock;
use crate::runtime::decoder::trie_cache::SingleLaneBlockCache;
use crate::runtime::decoder::trie_cache::Trie;
use crate::runtime::decoder::trie_cache::TrieEdge;
use crate::runtime::decoder::trie_cache::TrieNodeKey;
use crate::runtime::decoder::trie_cache::TrieNodeState;
use crate::runtime::decoder::trie_cache::TryEvictByKeyResult;
use crate::runtime::decoder::trie_cache::cache::Reservation;
use crate::runtime::decoder::trie_cache::cache::ReservationKey;

pub struct SingleLaneTrieBlockCache<const P: usize, KVA, SA, E = S3FIFOEntry<TrieNodeKey>>
where
    KVA: KVBlockAllocator,
    SA: StateBlockAllocator,
    E: EvictionEntry,
{
    kv_allocator: KVA,
    state_allocator: SA,
    trie: Arc<Trie<P, E>>,
    eviction_policy: Arc<dyn EvictionPolicy<TrieNodeKey, Entry = E>>,
    reservations: DashMap<ReservationKey, Reservation, AHashRandomState>,

    num_mutable_blocks: AtomicUsize,
    num_semi_immutable_blocks: AtomicUsize,
}

impl<const P: usize, KVA, SA> SingleLaneTrieBlockCache<P, KVA, SA>
where
    KVA: KVBlockAllocator,
    SA: StateBlockAllocator,
{
    pub fn new(kv_allocator: KVA, state_allocator: SA, capacity: usize, shutdown: Shutdown) -> Self {
        Self::with_eviction_policy(
            kv_allocator,
            state_allocator,
            Arc::new(S3FIFOClient::new(capacity, shutdown)),
        )
    }
}

impl<const P: usize, KVA, SA, E> SingleLaneTrieBlockCache<P, KVA, SA, E>
where
    KVA: KVBlockAllocator,
    SA: StateBlockAllocator,
    E: EvictionEntry,
{
    pub fn with_eviction_policy(
        kv_allocator: KVA,
        state_allocator: SA,
        eviction_policy: Arc<dyn EvictionPolicy<TrieNodeKey, Entry = E>>,
    ) -> Self {
        let trie = Arc::new(Trie::new(eviction_policy.clone()));

        Self {
            kv_allocator,
            state_allocator,
            trie,
            eviction_policy,
            reservations: DashMap::with_hasher(AHashRandomState::new()),
            num_mutable_blocks: AtomicUsize::new(0),
            num_semi_immutable_blocks: AtomicUsize::new(0),
        }
    }

    fn evict_one_immutable_block(&self) -> bool {
        let budget = self.trie.node_count().max(1);
        for _ in 0..budget {
            let Some(trie_node_key) = self.eviction_policy.evict_candidate() else {
                return false;
            };
            match self.trie.try_evict_by_key(trie_node_key) {
                TryEvictByKeyResult::Success => {
                    self.eviction_policy.accept_candidate(trie_node_key);
                    return true;
                },
                TryEvictByKeyResult::Missing => {
                    panic!("evict_one_immutable_block: candidate trie node must exist");
                },
                TryEvictByKeyResult::Rejected => {
                    self.eviction_policy.reject_candidate(trie_node_key);
                },
            }
        }
        false
    }

    fn try_get_immutable_block(
        &self,
        parent_trie_node_key: Option<TrieNodeKey>,
        trie_edge: &TrieEdge,
    ) -> Option<ImmutableBlock<P, E>> {
        let trie_node_key = self.trie.peek_by_parent(parent_trie_node_key, trie_edge)?.key();
        // TODO optimize perf here, should not release & acquire lock again
        if self.trie.try_external_pin_by_key(trie_node_key).is_none() {
            self.trie
                .remove_by_parent(parent_trie_node_key, trie_edge, trie_node_key);
            return None;
        }
        self.trie
            .peek_ref_by_key(trie_node_key)
            .eviction_entry()
            .expect("try_get_immutable_block: unpinned trie node must have an eviction entry")
            .touch();
        Some(ImmutableBlock::new(
            self.trie.clone(),
            trie_node_key,
            trie_edge.tokens().clone(),
        ))
    }

    fn child_pin_by_key(&self, parent_trie_node_key: Option<TrieNodeKey>) {
        let Some(parent_trie_node_key) = parent_trie_node_key else {
            return;
        };
        let child_pin_count = self.trie.child_pin_by_key(parent_trie_node_key);
        debug_assert!(
            child_pin_count.is_some(),
            "child_pin_parent: parent must be child-pinnable"
        );
    }

    fn child_unpin_by_key(&self, parent_trie_node_key: Option<TrieNodeKey>) {
        let Some(parent_trie_node_key) = parent_trie_node_key else {
            return;
        };
        let _ = self.trie.child_unpin_by_key(parent_trie_node_key);
    }
}

impl<const P: usize, KVA, SA, E> SingleLaneBlockCache<P> for SingleLaneTrieBlockCache<P, KVA, SA, E>
where
    KVA: KVBlockAllocator,
    SA: StateBlockAllocator,
    E: EvictionEntry,
{
    type Entry = E;

    fn try_evict(&self) -> bool {
        self.evict_one_immutable_block()
    }

    fn alloc_mutable_block<const N: usize>(&self) -> AllocateSingleLaneMutableBlockResult<N> {
        loop {
            let kv_placement = match self.kv_allocator.allocate() {
                KVBlockAllocationResult::Ok { block_placement } => block_placement,
                KVBlockAllocationResult::ResourceLimitExceeded => {
                    if self.evict_one_immutable_block() {
                        continue;
                    }
                    return AllocateSingleLaneMutableBlockResult::ResourceLimitExceeded;
                },
            };

            let state_placement = match self.state_allocator.allocate() {
                StateBlockAllocationResult::Ok { block_placement } => block_placement,
                StateBlockAllocationResult::ResourceLimitExceeded => {
                    if self.evict_one_immutable_block() {
                        continue;
                    }
                    return AllocateSingleLaneMutableBlockResult::ResourceLimitExceeded;
                },
            };

            self.num_mutable_blocks.fetch_add(1, Ordering::SeqCst);
            return AllocateSingleLaneMutableBlockResult::Mutable {
                block: MutableBlock::new(HashSet::new(), vec![], 0, 0, kv_placement, state_placement),
            };
        }
    }

    fn free_mutable_block<const N: usize>(&self, block: MutableBlock<N>) {
        drop(block);
        self.num_mutable_blocks.fetch_sub(1, Ordering::SeqCst);
    }

    fn commit_mutable_block<const N: usize>(
        &self,
        parent_trie_node_key: Option<TrieNodeKey>,
        block: MutableBlock<N>,
    ) -> CommitSingleLaneMutableBlockResult<P, E> {
        let (annotations, tokens, scheduled_token_index, ready_token_index, kv_placement, state_placement) =
            block.into_inner();
        self.num_mutable_blocks.fetch_sub(1, Ordering::SeqCst);
        let annotations: SmallVec<[BlockAnnotation; 1]> = annotations.into();
        let tokens: Arc<[Token]> = tokens.into();
        debug_assert_eq!(N, scheduled_token_index);
        debug_assert_eq!(N, ready_token_index);
        debug_assert_eq!(N, tokens.len());

        let trie_edge = TrieEdge::new(annotations.clone(), tokens.clone());
        let trie_node_key = self.trie.alloc_trie_node(
            parent_trie_node_key,
            annotations,
            tokens,
            kv_placement,
            state_placement,
            1,
        );
        let reservation_key = ReservationKey::new(parent_trie_node_key, trie_edge.clone());

        let result = match self
            .trie
            .insert_by_parent(parent_trie_node_key, &trie_edge, trie_node_key)
        {
            InsertNodeResult::Success { trie_node_key } => {
                CommitSingleLaneMutableBlockResult::Immutable {
                    block: ImmutableBlock::new(self.trie.clone(), trie_node_key, trie_edge.tokens().clone()),
                }
            },
            InsertNodeResult::Collision {
                trie_node_key: collision_trie_node_key,
            } => {
                debug_assert_eq!(
                    TrieNodeState::Valid,
                    self.trie.peek_by_key(collision_trie_node_key).state(),
                    "commit_mutable_block: insert collision node must be valid",
                );
                self.trie
                    .peek_ref_by_key(collision_trie_node_key)
                    .eviction_entry()
                    .expect("commit_mutable_block: collision trie node must have an eviction entry")
                    .touch();
                self.trie.free_trie_node(trie_node_key, 1);
                CommitSingleLaneMutableBlockResult::ImmutableCollision {
                    block: ImmutableBlock::new(self.trie.clone(), collision_trie_node_key, trie_edge.tokens().clone()),
                }
            },
        };
        if let Some((_, reservation)) = self.reservations.remove(&reservation_key) {
            reservation.notify();
        }
        result
    }

    fn reserve_semi_immutable_block<const N: usize>(
        &self,
        block_metadata: BlockMetadata<N>,
    ) -> ReserveSingleLaneSemiImmutableBlockResult<N, P, E> {
        let (parent_trie_node_key, annotations, trie_node_tokens) = block_metadata.into_inner();
        let trie_edge = TrieEdge::new(annotations, trie_node_tokens);
        if let Some(block) = self.try_get_immutable_block(parent_trie_node_key, &trie_edge) {
            return ReserveSingleLaneSemiImmutableBlockResult::Immutable { block };
        }
        let reservation_key = ReservationKey::new(parent_trie_node_key, trie_edge.clone());
        if let Some(reservation) = self.reservations.get(&reservation_key) {
            return ReserveSingleLaneSemiImmutableBlockResult::Wait {
                wait: reservation.listen(),
            };
        }

        let (kv_placement, state_placement) = loop {
            let kv_placement = match self.kv_allocator.allocate() {
                KVBlockAllocationResult::Ok { block_placement } => block_placement,
                KVBlockAllocationResult::ResourceLimitExceeded => {
                    if self.evict_one_immutable_block() {
                        continue;
                    }
                    return ReserveSingleLaneSemiImmutableBlockResult::ResourceLimitExceeded;
                },
            };
            let state_placement = match self.state_allocator.allocate() {
                StateBlockAllocationResult::Ok { block_placement } => block_placement,
                StateBlockAllocationResult::ResourceLimitExceeded => {
                    if self.evict_one_immutable_block() {
                        continue;
                    }
                    return ReserveSingleLaneSemiImmutableBlockResult::ResourceLimitExceeded;
                },
            };
            break (kv_placement, state_placement);
        };

        match self.reservations.entry(reservation_key) {
            DashEntry::Occupied(entry) => {
                ReserveSingleLaneSemiImmutableBlockResult::Wait {
                    wait: entry.get().listen(),
                }
            },
            DashEntry::Vacant(entry) => {
                if let Some(block) = self.try_get_immutable_block(parent_trie_node_key, &trie_edge) {
                    ReserveSingleLaneSemiImmutableBlockResult::Immutable { block }
                } else {
                    self.child_pin_by_key(parent_trie_node_key);
                    entry.insert(Reservation::new());
                    self.num_semi_immutable_blocks.fetch_add(1, Ordering::SeqCst);
                    let (annotations, trie_node_tokens) = trie_edge.into_inner();
                    ReserveSingleLaneSemiImmutableBlockResult::SemiImmutable {
                        block: SemiImmutableBlock::new(
                            parent_trie_node_key,
                            annotations,
                            trie_node_tokens,
                            0,
                            0,
                            kv_placement,
                            state_placement,
                        ),
                    }
                }
            },
        }
    }

    fn free_semi_immutable_block<const N: usize>(&self, block: SemiImmutableBlock<N>) {
        let (parent_trie_node_key, annotations, tokens, ..) = block.into_inner();
        self.num_semi_immutable_blocks.fetch_sub(1, Ordering::SeqCst);
        let reservation_key = ReservationKey::new(parent_trie_node_key, TrieEdge::new(annotations, tokens));
        if let Some((_, reservation)) = self.reservations.remove(&reservation_key) {
            reservation.notify();
        }
        self.child_unpin_by_key(parent_trie_node_key);
    }

    fn commit_semi_immutable_block<const N: usize>(
        &self,
        block: SemiImmutableBlock<N>,
    ) -> CommitSingleLaneSemiImmutableBlockResult<P, E> {
        let (
            parent_trie_node_key,
            annotations,
            tokens,
            scheduled_token_index,
            ready_token_index,
            kv_placement,
            state_placement,
        ) = block.into_inner();
        self.num_semi_immutable_blocks.fetch_sub(1, Ordering::SeqCst);
        debug_assert_eq!(scheduled_token_index, ready_token_index);
        debug_assert_eq!(ready_token_index, tokens.len());

        let trie_edge = TrieEdge::new(annotations.clone(), tokens.clone());
        let trie_node_key = self.trie.alloc_trie_node(
            parent_trie_node_key,
            annotations,
            tokens,
            kv_placement,
            state_placement,
            1,
        );
        let reservation_key = ReservationKey::new(parent_trie_node_key, trie_edge.clone());

        let result = match self
            .trie
            .insert_by_parent(parent_trie_node_key, &trie_edge, trie_node_key)
        {
            InsertNodeResult::Success { trie_node_key } => {
                self.child_unpin_by_key(parent_trie_node_key);
                CommitSingleLaneSemiImmutableBlockResult::Immutable {
                    block: ImmutableBlock::new(self.trie.clone(), trie_node_key, trie_edge.tokens().clone()),
                }
            },
            InsertNodeResult::Collision {
                trie_node_key: collision_trie_node_key,
            } => {
                self.child_unpin_by_key(parent_trie_node_key);
                debug_assert_eq!(
                    TrieNodeState::Valid,
                    self.trie.peek_by_key(collision_trie_node_key).state(),
                    "commit_semi_immutable_block: insert collision node must be valid",
                );
                self.trie
                    .peek_ref_by_key(collision_trie_node_key)
                    .eviction_entry()
                    .expect("commit_semi_immutable_block: collision trie node must have an eviction entry")
                    .touch();
                self.trie.free_trie_node(trie_node_key, 1);
                CommitSingleLaneSemiImmutableBlockResult::ImmutableCollision {
                    block: ImmutableBlock::new(self.trie.clone(), collision_trie_node_key, trie_edge.tokens().clone()),
                }
            },
        };

        if let Some((_, reservation)) = self.reservations.remove(&reservation_key) {
            reservation.notify();
        }
        result
    }

    fn num_mutable_block(&self) -> usize {
        self.num_mutable_blocks.load(Ordering::SeqCst)
    }

    fn num_semi_immutable_block(&self) -> usize {
        self.num_semi_immutable_blocks.load(Ordering::SeqCst)
    }

    fn num_pinned_immutable_block(&self) -> usize {
        self.trie.num_pinned_trie_node()
    }

    fn num_unpinned_immutable_block(&self) -> usize {
        let num_total = self.trie.num_total_trie_node();
        let num_pinned_immutable = self.num_pinned_immutable_block();
        debug_assert!(num_pinned_immutable <= num_total);
        num_total - num_pinned_immutable
    }
}

#[path = "./single_lane_test.rs"]
#[cfg(test)]
mod single_lane_test;
