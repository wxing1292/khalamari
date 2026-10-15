use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::runtime::decoder::trie_cache::EvictionEntry;
use crate::runtime::decoder::trie_cache::TrieNodeKey;
use crate::runtime::decoder::trie_cache::trie::Trie;

impl<const P: usize, E: EvictionEntry> Trie<P, E> {
    pub fn try_external_pin_by_key(self: &Arc<Self>, trie_node_key: TrieNodeKey) -> Option<u32> {
        let mut trie_node = self.trie_nodes.get_mut(trie_node_key)?;
        let exteral_pin_count = trie_node.try_external_pin();
        let pin_eviction =
            if exteral_pin_count == Some(1) && trie_node.child_pin_count() == 0 && trie_node.is_child_empty() {
                Some(
                    trie_node
                        .eviction_entry()
                        .expect("try_external_pin_by_key: unpinned trie node must have an eviction entry")
                        .clone(),
                )
            } else {
                None
            };
        if let Some(eviction) = &pin_eviction {
            eviction.pin();
        }
        drop(trie_node);
        if pin_eviction.is_some() {
            self.num_pinned_trie_nodes.fetch_add(1, Ordering::SeqCst);
        }
        exteral_pin_count
    }

    pub fn external_pin_by_key(self: &Arc<Self>, trie_node_key: TrieNodeKey) -> Option<u32> {
        let mut trie_node = self
            .trie_nodes
            .get_mut(trie_node_key)
            .expect("external_pin_by_key: trie node must exist");
        let exteral_pin_count = trie_node.try_external_pin();
        let pin_eviction =
            if exteral_pin_count == Some(1) && trie_node.child_pin_count() == 0 && trie_node.is_child_empty() {
                Some(
                    trie_node
                        .eviction_entry()
                        .expect("external_pin_by_key: unpinned trie node must have an eviction entry")
                        .clone(),
                )
            } else {
                None
            };
        if let Some(eviction) = &pin_eviction {
            eviction.pin();
        }
        drop(trie_node);
        if pin_eviction.is_some() {
            self.num_pinned_trie_nodes.fetch_add(1, Ordering::SeqCst);
        }
        exteral_pin_count
    }

    pub fn external_unpin_by_key(self: &Arc<Self>, trie_node_key: TrieNodeKey) -> u32 {
        let mut trie_node = self
            .trie_nodes
            .get_mut(trie_node_key)
            .expect("external_unpin_by_key: trie node must exist");
        let external_pin_count = trie_node.external_unpin();
        let (unpin_eviction, register_eviction) =
            if external_pin_count == 0 && trie_node.child_pin_count() == 0 && trie_node.is_child_empty() {
                (
                    trie_node
                        .eviction_entry()
                        .expect("external_unpin_by_key: trie node must have an eviction entry")
                        .clone(),
                    trie_node.register_eviction_entry(),
                )
            } else {
                return external_pin_count;
            };
        if let Some(eviction) = register_eviction {
            eviction.register();
        }
        unpin_eviction.unpin();
        drop(trie_node);
        self.num_pinned_trie_nodes.fetch_sub(1, Ordering::SeqCst);
        external_pin_count
    }

    pub fn try_child_pin_by_key(self: &Arc<Self>, trie_node_key: TrieNodeKey) -> Option<u32> {
        let mut trie_node = self.trie_nodes.get_mut(trie_node_key)?;
        let child_pin_count = trie_node.try_child_pin();
        let pin_eviction =
            if trie_node.external_pin_count() == 0 && child_pin_count == Some(1) && trie_node.is_child_empty() {
                Some(
                    trie_node
                        .eviction_entry()
                        .expect("try_child_pin_by_key: unpinned trie node must have an eviction entry")
                        .clone(),
                )
            } else {
                None
            };
        if let Some(eviction) = &pin_eviction {
            eviction.pin();
        }
        drop(trie_node);
        if pin_eviction.is_some() {
            self.num_pinned_trie_nodes.fetch_add(1, Ordering::SeqCst);
        }
        child_pin_count
    }

    pub fn child_pin_by_key(self: &Arc<Self>, trie_node_key: TrieNodeKey) -> Option<u32> {
        let mut trie_node = self
            .trie_nodes
            .get_mut(trie_node_key)
            .expect("child_pin_by_key: trie node must exist");
        let child_pin_count = trie_node.try_child_pin();
        let pin_eviction =
            if trie_node.external_pin_count() == 0 && child_pin_count == Some(1) && trie_node.is_child_empty() {
                Some(
                    trie_node
                        .eviction_entry()
                        .expect("child_pin_by_key: unpinned trie node must have an eviction entry")
                        .clone(),
                )
            } else {
                None
            };
        if let Some(eviction) = &pin_eviction {
            eviction.pin();
        }
        drop(trie_node);
        if pin_eviction.is_some() {
            self.num_pinned_trie_nodes.fetch_add(1, Ordering::SeqCst);
        }
        child_pin_count
    }

    pub fn child_unpin_by_key(self: &Arc<Self>, trie_node_key: TrieNodeKey) -> u32 {
        let mut trie_node = self
            .trie_nodes
            .get_mut(trie_node_key)
            .expect("child_unpin_by_key: trie node must exist");
        let child_pin_count = trie_node.child_unpin();
        let (unpin_eviction, register_eviction) =
            if trie_node.external_pin_count() == 0 && child_pin_count == 0 && trie_node.is_child_empty() {
                (
                    trie_node
                        .eviction_entry()
                        .expect("child_unpin_by_key: trie node must have an eviction entry")
                        .clone(),
                    trie_node.register_eviction_entry(),
                )
            } else {
                return child_pin_count;
            };
        if let Some(eviction) = register_eviction {
            eviction.register();
        }
        unpin_eviction.unpin();
        drop(trie_node);
        self.num_pinned_trie_nodes.fetch_sub(1, Ordering::SeqCst);
        child_pin_count
    }
}
