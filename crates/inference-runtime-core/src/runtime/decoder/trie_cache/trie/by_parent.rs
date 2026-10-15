use std::sync::Arc;
use std::sync::atomic::Ordering;

use dashmap::Entry as DashEntry;

use crate::runtime::decoder::trie_cache::DataStoreValueMut;
use crate::runtime::decoder::trie_cache::EvictionEntry;
use crate::runtime::decoder::trie_cache::InsertNodeResult;
use crate::runtime::decoder::trie_cache::TrieEdge;
use crate::runtime::decoder::trie_cache::TrieNode;
use crate::runtime::decoder::trie_cache::TrieNodeKey;
use crate::runtime::decoder::trie_cache::trie::Trie;

impl<const P: usize, E: EvictionEntry> Trie<P, E> {
    pub fn peek_by_parent(
        self: &Arc<Self>,
        parent_trie_node_key: Option<TrieNodeKey>,
        trie_edge: &TrieEdge,
    ) -> Option<DataStoreValueMut<'_, TrieNodeKey, TrieNode<E>>> {
        let trie_node_key = if let Some(parent_trie_node_key) = parent_trie_node_key {
            self.trie_nodes
                .get_mut(parent_trie_node_key)
                .expect("peek_by_parent: parent trie node must exist")
                .get_child_by_key(trie_edge)?
        } else {
            *self.trie_roots.get(trie_edge)?
        };

        let trie_node = self.trie_nodes.get_mut(trie_node_key);
        let Some(trie_node) = trie_node else {
            self.remove_by_parent(parent_trie_node_key, trie_edge, trie_node_key);
            return None;
        };
        if trie_node.is_tombstone() {
            drop(trie_node);
            self.remove_by_parent(parent_trie_node_key, trie_edge, trie_node_key);
            return None;
        }
        Some(trie_node)
    }

    pub fn insert_by_parent(
        self: &Arc<Self>,
        parent_trie_node_key: Option<TrieNodeKey>,
        trie_edge: &TrieEdge,
        trie_node_key: TrieNodeKey,
    ) -> InsertNodeResult {
        loop {
            let collision_trie_node_key = match parent_trie_node_key {
                Some(parent_trie_node_key) => {
                    let mut parent_trie_node = self
                        .trie_nodes
                        .get_mut(parent_trie_node_key)
                        .expect("insert_by_parent: parent trie node must exist");
                    if let Some(collision_trie_node_key) = parent_trie_node.get_child_by_key(trie_edge) {
                        Some(collision_trie_node_key)
                    } else {
                        let pin_eviction = if !parent_trie_node.is_pinned() {
                            Some(
                                parent_trie_node
                                    .eviction_entry()
                                    .expect("insert_by_parent: unpinned parent must have an eviction entry")
                                    .clone(),
                            )
                        } else {
                            None
                        };
                        parent_trie_node.insert_child_by_key(trie_edge.clone(), trie_node_key);
                        if let Some(eviction) = pin_eviction {
                            self.num_pinned_trie_nodes.fetch_add(1, Ordering::SeqCst);
                            eviction.pin();
                        }
                        None
                    }
                },
                None => {
                    match self.trie_roots.entry(trie_edge.clone()) {
                        DashEntry::Occupied(entry) => Some(*entry.get()),
                        DashEntry::Vacant(entry) => {
                            entry.insert(trie_node_key);
                            None
                        },
                    }
                },
            };

            if let Some(collision_trie_node_key) = collision_trie_node_key {
                if self.try_external_pin_by_key(collision_trie_node_key).is_none() {
                    if self.replace_by_parent(parent_trie_node_key, trie_edge, collision_trie_node_key, trie_node_key) {
                        self.register_eviction_entry_by_key(trie_node_key);
                        return InsertNodeResult::Success { trie_node_key };
                    } else {
                        continue;
                    }
                } else {
                    return InsertNodeResult::Collision {
                        trie_node_key: collision_trie_node_key,
                    };
                }
            } else {
                self.register_eviction_entry_by_key(trie_node_key);
                return InsertNodeResult::Success { trie_node_key };
            }
        }
    }

    pub fn replace_by_parent(
        self: &Arc<Self>,
        parent_trie_node_key: Option<TrieNodeKey>,
        trie_edge: &TrieEdge,
        from_trie_node_key: TrieNodeKey,
        to_trie_node_key: TrieNodeKey,
    ) -> bool {
        debug_assert_ne!(
            from_trie_node_key, to_trie_node_key,
            "replace_by_parent: source and destination trie node keys must differ"
        );

        match parent_trie_node_key {
            Some(parent_trie_node_key) => {
                let mut parent_trie_node = self
                    .trie_nodes
                    .get_mut(parent_trie_node_key)
                    .expect("replace_by_parent: parent trie node must exist");
                if parent_trie_node.get_child_by_key(trie_edge) != Some(from_trie_node_key) {
                    false
                } else {
                    let removed_trie_node_key = parent_trie_node.remove_child_by_key(trie_edge);
                    debug_assert_eq!(removed_trie_node_key, from_trie_node_key);
                    parent_trie_node.insert_child_by_key(trie_edge.clone(), to_trie_node_key);
                    true
                }
            },
            None => {
                match self.trie_roots.entry(trie_edge.clone()) {
                    DashEntry::Occupied(mut entry) => {
                        if entry.get() != &from_trie_node_key {
                            false
                        } else {
                            entry.insert(to_trie_node_key);
                            true
                        }
                    },
                    DashEntry::Vacant(_) => false,
                }
            },
        }
    }

    pub fn remove_by_parent(
        self: &Arc<Self>,
        parent_trie_node_key: Option<TrieNodeKey>,
        trie_edge: &TrieEdge,
        trie_node_key: TrieNodeKey,
    ) {
        match parent_trie_node_key {
            Some(parent_trie_node_key) => {
                let mut parent_trie_node = self
                    .trie_nodes
                    .get_mut(parent_trie_node_key)
                    .expect("remove_by_parent: parent trie node must exist");
                if parent_trie_node.get_child_by_key(trie_edge) == Some(trie_node_key) {
                    let removed_trie_node_key = parent_trie_node.remove_child_by_key(trie_edge);
                    debug_assert_eq!(removed_trie_node_key, trie_node_key);
                    let unpin_eviction = if !parent_trie_node.is_pinned() {
                        parent_trie_node
                            .eviction_entry()
                            .expect("remove_by_parent: formerly unpinned parent must have an eviction entry")
                            .clone()
                    } else {
                        return;
                    };
                    unpin_eviction.unpin();
                    drop(parent_trie_node);
                    self.num_pinned_trie_nodes.fetch_sub(1, Ordering::SeqCst);
                }
            },
            None => {
                self.trie_roots.remove_if(trie_edge, |_, value| value == &trie_node_key);
            },
        }
    }
}
