use super::test_utils::*;
use crate::runtime::decoder::trie_cache::TryEvictByKeyResult;

#[test]
fn test_try_external_pin_unpin_by_key() {
    let trie = new_trie();
    let trie_node_key = trie.alloc_trie_node(
        None,
        new_annotations(1),
        new_tokens(&[1]),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.free_trie_node(trie_node_key, 1);
    assert_eq!(0, trie.num_pinned_trie_node());

    assert_eq!(None, trie.try_external_pin_by_key(trie_node_key));
}

#[test]
fn test_external_pin_unpin_by_key() {
    let trie = new_trie();
    let trie_node_key = trie.alloc_trie_node(
        None,
        new_annotations(1),
        new_tokens(&[1]),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    assert_eq!(1, trie.num_pinned_trie_node());
    assert!(trie.eviction_policy().evict_candidate().is_none());

    assert_eq!(Some(2), trie.external_pin_by_key(trie_node_key));
    assert_eq!(1, trie.external_unpin_by_key(trie_node_key));

    assert_eq!(0, trie.external_unpin_by_key(trie_node_key));
    assert_eq!(0, trie.num_pinned_trie_node());
    assert_eq!(trie_node_key, trie.eviction_policy().evict_candidate().unwrap());

    assert_eq!(TryEvictByKeyResult::Success, trie.try_evict_by_key(trie_node_key));
    assert!(trie.trie_nodes.get_mut(trie_node_key).is_none());
}

#[test]
fn test_try_child_pin_unpin_by_key() {
    let trie = new_trie();
    let trie_node_key = trie.alloc_trie_node(
        None,
        new_annotations(1),
        new_tokens(&[1]),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.free_trie_node(trie_node_key, 1);
    assert_eq!(0, trie.num_pinned_trie_node());

    assert_eq!(None, trie.try_child_pin_by_key(trie_node_key));
}

#[test]
fn test_child_pin_unpin_by_key() {
    let trie = new_trie();
    let trie_node_key = trie.alloc_trie_node(
        None,
        new_annotations(1),
        new_tokens(&[1]),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.child_pin_by_key(trie_node_key);
    trie.external_unpin_by_key(trie_node_key);
    assert_eq!(1, trie.num_pinned_trie_node());
    assert!(trie.eviction_policy().evict_candidate().is_none());

    assert_eq!(Some(2), trie.child_pin_by_key(trie_node_key));
    assert_eq!(1, trie.child_unpin_by_key(trie_node_key));

    assert_eq!(0, trie.child_unpin_by_key(trie_node_key));
    assert_eq!(0, trie.num_pinned_trie_node());
    assert_eq!(trie_node_key, trie.eviction_policy().evict_candidate().unwrap());

    assert_eq!(TryEvictByKeyResult::Success, trie.try_evict_by_key(trie_node_key));
    assert!(trie.trie_nodes.get_mut(trie_node_key).is_none());
}
