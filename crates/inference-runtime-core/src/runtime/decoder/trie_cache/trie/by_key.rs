use std::sync::Arc;

use crate::runtime::decoder::trie_cache::DataStoreValueMut;
use crate::runtime::decoder::trie_cache::DataStoreValueRef;
use crate::runtime::decoder::trie_cache::EvictionEntry;
use crate::runtime::decoder::trie_cache::TrieEdge;
use crate::runtime::decoder::trie_cache::TrieNode;
use crate::runtime::decoder::trie_cache::TrieNodeKey;
use crate::runtime::decoder::trie_cache::TryEvictByKeyResult;
use crate::runtime::decoder::trie_cache::trie::Trie;

impl<const P: usize, E: EvictionEntry> Trie<P, E> {
    pub fn peek_ref_by_key(
        self: &Arc<Self>,
        trie_node_key: TrieNodeKey,
    ) -> DataStoreValueRef<'_, TrieNodeKey, TrieNode<E>> {
        self.trie_nodes
            .get_ref(trie_node_key)
            .expect("peek_ref_by_key: trie node must exist")
    }

    pub fn peek_by_key(
        self: &Arc<Self>,
        trie_node_key: TrieNodeKey,
    ) -> DataStoreValueMut<'_, TrieNodeKey, TrieNode<E>> {
        self.trie_nodes
            .get_mut(trie_node_key)
            .expect("peek_by_key: trie node must exist")
    }

    pub fn try_evict_by_key(self: &Arc<Self>, trie_node_key: TrieNodeKey) -> TryEvictByKeyResult {
        let Some(mut trie_node) = self.trie_nodes.get_mut(trie_node_key) else {
            return TryEvictByKeyResult::Missing;
        };
        if !trie_node.try_mark_tombstone() {
            return TryEvictByKeyResult::Rejected;
        }

        let parent_trie_node_key = trie_node.parent_node_key();
        let trie_node_annotations = trie_node.annotations().clone();
        let trie_node_tokens = trie_node.tokens().clone();
        drop(trie_node);

        let trie_node_edge = TrieEdge::new(trie_node_annotations, trie_node_tokens);
        self.remove_by_parent(parent_trie_node_key, &trie_node_edge, trie_node_key);
        self.free_trie_node(trie_node_key, 0);
        TryEvictByKeyResult::Success
    }
}
