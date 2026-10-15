use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use ahash::RandomState as AHashRandomState;
use dashmap::DashMap;
use rand::rngs::SmallRng;

use crate::runtime::decoder::trie_cache::EvictionEntry;
use crate::runtime::decoder::trie_cache::EvictionPolicy;
use crate::runtime::decoder::trie_cache::TrieNodeKey;
use crate::runtime::decoder::trie_cache::TrieNodeStore;

mod edge;
pub use edge::TrieEdge;

mod by_key;
mod by_parent;
mod pin;
mod alloc_free;

#[derive(Debug)]
pub enum InsertNodeResult {
    Success { trie_node_key: TrieNodeKey },
    Collision { trie_node_key: TrieNodeKey },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TryEvictByKeyResult {
    Success,
    Missing,
    Rejected,
}

// TODO use a non random partition picker
thread_local! {
    static RNG: RefCell<SmallRng> = RefCell::new(rand::make_rng());
}

#[derive(Debug)]
pub struct Trie<const P: usize, E: EvictionEntry> {
    trie_roots: DashMap<TrieEdge, TrieNodeKey, AHashRandomState>,
    trie_nodes: TrieNodeStore<P, E>,
    eviction_policy: Arc<dyn EvictionPolicy<TrieNodeKey, Entry = E>>,
    num_pinned_trie_nodes: AtomicUsize,
}

impl<const P: usize, E: EvictionEntry> Trie<P, E> {
    pub fn new(eviction_policy: Arc<dyn EvictionPolicy<TrieNodeKey, Entry = E>>) -> Self {
        Self {
            trie_roots: DashMap::with_hasher(AHashRandomState::new()),
            trie_nodes: TrieNodeStore::new(),
            eviction_policy,
            num_pinned_trie_nodes: AtomicUsize::new(0),
        }
    }

    pub fn eviction_policy(&self) -> Arc<dyn EvictionPolicy<TrieNodeKey, Entry = E>> {
        self.eviction_policy.clone()
    }

    fn register_eviction_entry_by_key(&self, trie_node_key: TrieNodeKey) {
        let eviction_entry = self
            .trie_nodes
            .get_mut(trie_node_key)
            .expect("register_eviction_entry_by_key: trie node must exist")
            .register_eviction_entry();
        if let Some(eviction_entry) = eviction_entry {
            eviction_entry.register();
        }
    }

    pub fn node_count(&self) -> usize {
        self.trie_nodes.len()
    }

    pub fn num_pinned_trie_node(&self) -> usize {
        self.num_pinned_trie_nodes.load(Ordering::SeqCst)
    }

    pub fn num_total_trie_node(&self) -> usize {
        self.trie_nodes.len()
    }
}

#[path = "./alloc_free_test.rs"]
#[cfg(test)]
mod alloc_free_test;

#[path = "./by_key_test.rs"]
#[cfg(test)]
mod by_key_test;

#[path = "./by_parent_test.rs"]
#[cfg(test)]
mod by_parent_test;

#[path = "./pin_test.rs"]
#[cfg(test)]
mod pin_test;

#[path = "./test_utils.rs"]
#[cfg(test)]
mod test_utils;
