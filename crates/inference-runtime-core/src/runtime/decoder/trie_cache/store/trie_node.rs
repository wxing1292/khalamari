use std::sync::Arc;

use ahash::AHashMap;
use inference_runtime_macro::sanity_check;
use slotmap::new_key_type;
use smallvec::SmallVec;

use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;
use crate::runtime::decoder::KVBlockPlacement;
use crate::runtime::decoder::StateBlockPlacement;
use crate::runtime::decoder::trie_cache::EvictionEntry;
use crate::runtime::decoder::trie_cache::TrieEdge;
use crate::runtime::decoder::trie_cache::store::DataKey;
use crate::runtime::decoder::trie_cache::store::DataValue;
use crate::runtime::decoder::trie_cache::store::PartitionedDataStore;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TrieNodeKey {
    partition_key: u32,
    local_key: LocalKVTrieNodeKey,
}

new_key_type! { pub struct LocalKVTrieNodeKey; }

impl DataKey for TrieNodeKey {
    type LocalKey = LocalKVTrieNodeKey;

    fn partition_key(&self) -> u32 {
        self.partition_key
    }

    fn local_key(&self) -> Self::LocalKey {
        self.local_key
    }

    fn from_parts(partition_key: u32, local_key: Self::LocalKey) -> Self {
        Self {
            partition_key,
            local_key,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrieNodeState {
    Valid,
    Tombstone,
}

#[derive(Debug)]
pub struct TrieNode<E: EvictionEntry> {
    partition_key: u32,
    state: TrieNodeState,

    external_pin_count: u32,
    child_pin_count: u32,
    parent_node_key: Option<TrieNodeKey>,
    child_node_keys: AHashMap<TrieEdge, TrieNodeKey>,
    annotations: SmallVec<[BlockAnnotation; 1]>,
    tokens: Arc<[Token]>,
    kv_placement: KVBlockPlacement,
    state_placement: StateBlockPlacement,
    eviction_entry: Option<Arc<E>>,
    is_eviction_entry_registered: bool,
}

impl<E: EvictionEntry> TrieNode<E> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        partition_key: u32,
        external_pin_count: u32,
        parent_node_key: Option<TrieNodeKey>,
        child_node_keys: AHashMap<TrieEdge, TrieNodeKey>,
        annotations: SmallVec<[BlockAnnotation; 1]>,
        tokens: Arc<[Token]>,
        kv_placement: KVBlockPlacement,
        state_placement: StateBlockPlacement,
    ) -> Self {
        Self {
            partition_key,
            state: TrieNodeState::Valid,
            external_pin_count,
            child_pin_count: 0,

            parent_node_key,
            child_node_keys,
            annotations,
            tokens,
            kv_placement,
            state_placement,
            eviction_entry: None,
            is_eviction_entry_registered: false,
        }
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn try_mark_tombstone(&mut self) -> bool {
        match self.state {
            TrieNodeState::Valid => {
                if self.external_pin_count == 0 && self.child_pin_count == 0 && self.child_node_keys.is_empty() {
                    self.state = TrieNodeState::Tombstone;
                    true
                } else {
                    false
                }
            },
            TrieNodeState::Tombstone => false,
        }
    }

    pub fn state(&self) -> TrieNodeState {
        self.state
    }

    pub fn is_tombstone(&self) -> bool {
        self.state == TrieNodeState::Tombstone
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn try_external_pin(&mut self) -> Option<u32> {
        match &self.state {
            TrieNodeState::Valid => {
                self.external_pin_count += 1;
                Some(self.external_pin_count)
            },
            TrieNodeState::Tombstone => None,
        }
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn external_unpin(&mut self) -> u32 {
        match &self.state {
            TrieNodeState::Valid => {
                debug_assert_ne!(
                    0, self.external_pin_count,
                    "external_unpin: external_pin_count must be > 0 before decrement"
                );
                self.external_pin_count -= 1;
                self.external_pin_count
            },
            TrieNodeState::Tombstone => unreachable!(),
        }
    }

    pub fn external_pin_count(&self) -> u32 {
        self.external_pin_count
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn try_child_pin(&mut self) -> Option<u32> {
        match self.state {
            TrieNodeState::Valid => {
                self.child_pin_count += 1;
                Some(self.child_pin_count)
            },
            TrieNodeState::Tombstone => None,
        }
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn child_unpin(&mut self) -> u32 {
        match self.state {
            TrieNodeState::Valid => {
                debug_assert_ne!(
                    0, self.child_pin_count,
                    "child_unpin: child_pin_count must be > 0 before decrement"
                );
                self.child_pin_count -= 1;
                self.child_pin_count
            },
            TrieNodeState::Tombstone => unreachable!(),
        }
    }

    pub fn child_pin_count(&self) -> u32 {
        self.child_pin_count
    }

    pub fn is_pinned(&self) -> bool {
        self.external_pin_count > 0 || self.child_pin_count > 0
    }

    pub fn parent_node_key(&self) -> Option<TrieNodeKey> {
        self.parent_node_key
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn insert_child_by_key(&mut self, child_trie_edge: TrieEdge, child_trie_node_key: TrieNodeKey) {
        let conflict = self.child_node_keys.insert(child_trie_edge, child_trie_node_key);
        debug_assert!(conflict.is_none(), "insert_child_by_key: child node already exists");
        self.child_pin_count += 1;
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    pub fn remove_child_by_key(&mut self, child_trie_edge: &TrieEdge) -> TrieNodeKey {
        let child_trie_node_key = self
            .child_node_keys
            .remove(child_trie_edge)
            .expect("remove_child_by_key: child node does not exist");
        debug_assert_ne!(
            0, self.child_pin_count,
            "remove_child_by_key: child_pin_count must be > 0 before decrement"
        );
        self.child_pin_count -= 1;
        child_trie_node_key
    }

    pub fn get_child_by_key(&self, child_trie_edge: &TrieEdge) -> Option<TrieNodeKey> {
        self.child_node_keys.get(child_trie_edge).cloned()
    }

    pub fn is_child_empty(&self) -> bool {
        self.child_node_keys.is_empty()
    }

    pub fn annotations(&self) -> &SmallVec<[BlockAnnotation; 1]> {
        &self.annotations
    }

    pub fn tokens(&self) -> &Arc<[Token]> {
        &self.tokens
    }

    pub fn kv_placement(&self) -> &KVBlockPlacement {
        &self.kv_placement
    }

    pub fn state_placement(&self) -> &StateBlockPlacement {
        &self.state_placement
    }

    pub fn eviction_entry(&self) -> Option<&Arc<E>> {
        self.eviction_entry.as_ref()
    }

    pub fn set_eviction_entry(&mut self, eviction_entry: Arc<E>) {
        assert!(
            self.eviction_entry.is_none(),
            "set_eviction_entry: trie node must not already have an eviction entry"
        );
        self.eviction_entry = Some(eviction_entry);
    }

    pub fn register_eviction_entry(&mut self) -> Option<Arc<E>> {
        if self.is_eviction_entry_registered {
            return None;
        }
        self.is_eviction_entry_registered = true;
        Some(
            self.eviction_entry()
                .expect("register_eviction_entry: trie node must have an eviction entry")
                .clone(),
        )
    }

    pub fn sanity_check(&self) {
        debug_assert!(
            self.child_pin_count as usize >= self.child_node_keys.len(),
            "sanity_check: child_pin_count must cover child node keys"
        );
        match &self.state {
            TrieNodeState::Valid => {},
            TrieNodeState::Tombstone => {
                debug_assert_eq!(
                    0, self.external_pin_count,
                    "sanity_check: tombstone trie node must have external_pin_count == 0"
                );
                debug_assert_eq!(
                    0, self.child_pin_count,
                    "sanity_check: tombstone trie node must have child_pin_count == 0"
                );
                debug_assert_eq!(
                    0,
                    self.child_node_keys.len(),
                    "sanity_check: tombstone trie node must have no child trie nodes"
                );
            },
        }
    }
}

impl<E: EvictionEntry> DataValue for TrieNode<E> {
    fn partition_key(&self) -> u32 {
        self.partition_key
    }
}

pub type TrieNodeStore<const P: usize, E> = PartitionedDataStore<P, TrieNodeKey, TrieNode<E>>;
