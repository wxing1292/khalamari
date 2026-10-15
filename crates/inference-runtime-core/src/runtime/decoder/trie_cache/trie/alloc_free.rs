use std::sync::Arc;
use std::sync::atomic::Ordering;

use ahash::AHashMap;
use rand::RngExt;
use smallvec::SmallVec;

use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;
use crate::runtime::decoder::KVBlockPlacement;
use crate::runtime::decoder::StateBlockPlacement;
use crate::runtime::decoder::trie_cache::EvictionEntry;
use crate::runtime::decoder::trie_cache::TrieNode;
use crate::runtime::decoder::trie_cache::TrieNodeKey;
use crate::runtime::decoder::trie_cache::trie::RNG;
use crate::runtime::decoder::trie_cache::trie::Trie;

impl<const P: usize, E: EvictionEntry> Trie<P, E> {
    pub fn alloc_trie_node(
        self: &Arc<Self>,
        parent_node_key: Option<TrieNodeKey>,
        annotations: SmallVec<[BlockAnnotation; 1]>,
        tokens: Arc<[Token]>,
        kv_placement: KVBlockPlacement,
        state_placement: StateBlockPlacement,
        init_external_pin_count: u32,
    ) -> TrieNodeKey {
        debug_assert!(1 <= init_external_pin_count);
        let partition_key = self.pick_trie_node_partition();
        let child_node_keys = AHashMap::new();
        let trie_node_key = self.trie_nodes.insert(TrieNode::new(
            partition_key,
            init_external_pin_count,
            parent_node_key,
            child_node_keys,
            annotations,
            tokens,
            kv_placement,
            state_placement,
        ));
        let eviction_entry = self.eviction_policy.new_entry(trie_node_key);
        self.trie_nodes
            .get_mut(trie_node_key)
            .expect("alloc_trie_node: inserted trie node must exist")
            .set_eviction_entry(eviction_entry);
        self.num_pinned_trie_nodes.fetch_add(1, Ordering::SeqCst);
        trie_node_key
    }

    pub fn free_trie_node(self: &Arc<Self>, trie_node_key: TrieNodeKey, final_external_pin_count: u32) {
        let trie_node = self
            .trie_nodes
            .remove(trie_node_key)
            .expect("free_trie_node: key must exist");
        debug_assert_eq!(
            final_external_pin_count,
            trie_node.external_pin_count(),
            "free_trie_node: trie node must have external_pin_count == 0 before free"
        );
        debug_assert_eq!(
            0,
            trie_node.child_pin_count(),
            "free_trie_node: trie node must have child_pin_count == 0 before free"
        );
        debug_assert!(
            trie_node.is_child_empty(),
            "free_trie_node: trie node must not have child nodes before free"
        );
        if final_external_pin_count > 0 {
            self.num_pinned_trie_nodes.fetch_sub(1, Ordering::SeqCst);
        }
        drop(trie_node);
    }

    #[inline]
    fn pick_trie_node_partition(&self) -> u32 {
        RNG.with(|rng| rng.borrow_mut().random::<u32>() % (P as u32))
    }
}
