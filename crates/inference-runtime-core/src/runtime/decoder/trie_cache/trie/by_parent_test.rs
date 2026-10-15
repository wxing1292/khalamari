use super::test_utils::*;
use crate::runtime::decoder::trie_cache::InsertNodeResult;

#[test]
fn test_insert_peek_replace_peek_remove_peek_by_parent_wo_parent() {
    let trie = new_trie();
    let edge = new_edge(1, &[1, 2]);
    let from_trie_node_key = trie.alloc_trie_node(
        None,
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.insert_by_parent(None, &edge, from_trie_node_key);

    let trie_node = trie.peek_by_parent(None, &edge).unwrap();
    assert_eq!(from_trie_node_key, trie_node.key());
    drop(trie_node);

    let to_trie_node_key = trie.alloc_trie_node(
        None,
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    assert!(trie.replace_by_parent(None, &edge, from_trie_node_key, to_trie_node_key));

    let trie_node = trie.peek_by_parent(None, &edge).unwrap();
    assert_eq!(to_trie_node_key, trie_node.key());
    drop(trie_node);

    trie.remove_by_parent(None, &edge, to_trie_node_key);
    assert!(trie.peek_by_parent(None, &edge).is_none());
    assert!(trie.trie_roots.is_empty());
}

#[test]
fn test_insert_peek_by_parent_wo_parent_tombstone_node() {
    let trie = new_trie();
    let edge = new_edge(1, &[1, 2]);
    let trie_node_key = trie.alloc_trie_node(
        None,
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.insert_by_parent(None, &edge, trie_node_key);
    let mut trie_node = trie.peek_by_parent(None, &edge).unwrap();
    assert_eq!(0, trie_node.external_unpin());
    assert!(trie_node.try_mark_tombstone());
    drop(trie_node);

    assert!(trie.peek_by_parent(None, &edge).is_none());
    assert!(trie.trie_roots.is_empty());
}

#[test]
fn test_insert_by_parent_wo_parent_replaces_tombstone_collision() {
    let trie = new_trie();
    let edge = new_edge(3, &[3, 4]);
    let from_trie_node_key = trie.alloc_trie_node(
        None,
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.insert_by_parent(None, &edge, from_trie_node_key);
    assert_eq!(0, trie.external_unpin_by_key(from_trie_node_key));
    assert!(trie.peek_by_key(from_trie_node_key).try_mark_tombstone());

    let to_trie_node_key = trie.alloc_trie_node(
        None,
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    let result = trie.insert_by_parent(None, &edge, to_trie_node_key);
    match result {
        InsertNodeResult::Success { trie_node_key } => assert_eq!(to_trie_node_key, trie_node_key),
        InsertNodeResult::Collision { .. } => unreachable!(),
    }
    assert_eq!(1, trie.num_pinned_trie_node());

    let trie_node = trie.peek_by_parent(None, &edge).unwrap();
    assert_eq!(to_trie_node_key, trie_node.key());
    drop(trie_node);

    trie.external_unpin_by_key(to_trie_node_key);
    trie.remove_by_parent(None, &edge, to_trie_node_key);
    assert!(trie.trie_roots.is_empty());
}

#[test]
fn test_insert_peek_by_parent_wo_parent_missing_edge() {
    let trie = new_trie();
    let edge = new_edge(1, &[1, 2]);
    let trie_node_key = trie.alloc_trie_node(
        None,
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.insert_by_parent(None, &edge, trie_node_key);
    trie.external_unpin_by_key(trie_node_key);
    trie.remove_by_parent(None, &edge, trie_node_key);

    assert!(trie.peek_by_parent(None, &edge).is_none());
    assert!(trie.trie_roots.is_empty());
}

#[test]
fn test_insert_by_parent_wo_parent_collision_valid_collision() {
    let trie = new_trie();
    let edge = new_edge(4, &[4, 5]);
    let existing_trie_node_key = trie.alloc_trie_node(
        None,
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.insert_by_parent(None, &edge, existing_trie_node_key);

    let new_trie_node_key = trie.alloc_trie_node(
        None,
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    let result = trie.insert_by_parent(None, &edge, new_trie_node_key);
    match result {
        InsertNodeResult::Success { .. } => unreachable!(),
        InsertNodeResult::Collision { trie_node_key } => assert_eq!(existing_trie_node_key, trie_node_key),
    }
    assert_eq!(2, trie.num_pinned_trie_node());

    let trie_node = trie.peek_by_parent(None, &edge).unwrap();
    assert_eq!(existing_trie_node_key, trie_node.key());
    assert_eq!(2, trie_node.external_pin_count());
    drop(trie_node);

    trie.free_trie_node(new_trie_node_key, 1);
}

#[test]
fn test_insert_peek_replace_peek_remove_peek_by_parent_w_parent() {
    let trie = new_trie();
    let parent_trie_node_key = trie.alloc_trie_node(
        None,
        new_annotations(1),
        new_tokens(&[1]),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    assert_eq!(0, trie.external_unpin_by_key(parent_trie_node_key));
    assert_eq!(0, trie.num_pinned_trie_node());
    let evict_candidate = trie.eviction_policy().evict_candidate().unwrap();
    assert_eq!(parent_trie_node_key, evict_candidate);
    trie.eviction_policy().reject_candidate(evict_candidate);

    let edge = new_edge(2, &[2, 3]);
    let from_trie_node_key = trie.alloc_trie_node(
        Some(parent_trie_node_key),
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.insert_by_parent(Some(parent_trie_node_key), &edge, from_trie_node_key);
    assert_eq!(2, trie.num_pinned_trie_node());
    assert!(trie.eviction_policy().evict_candidate().is_none());

    let trie_node = trie.peek_by_parent(Some(parent_trie_node_key), &edge).unwrap();
    assert_eq!(from_trie_node_key, trie_node.key());
    drop(trie_node);

    let to_trie_node_key = trie.alloc_trie_node(
        Some(parent_trie_node_key),
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    assert!(trie.replace_by_parent(Some(parent_trie_node_key), &edge, from_trie_node_key, to_trie_node_key,));

    let trie_node = trie.peek_by_parent(Some(parent_trie_node_key), &edge).unwrap();
    assert_eq!(to_trie_node_key, trie_node.key());
    drop(trie_node);

    trie.remove_by_parent(Some(parent_trie_node_key), &edge, to_trie_node_key);
    assert_eq!(2, trie.num_pinned_trie_node());
    let evict_candidate = trie.eviction_policy().evict_candidate().unwrap();
    assert_eq!(parent_trie_node_key, evict_candidate);
    trie.eviction_policy().reject_candidate(evict_candidate);

    assert!(trie.peek_by_parent(Some(parent_trie_node_key), &edge).is_none());
    let parent_trie_node = trie.peek_by_key(parent_trie_node_key);
    assert!(parent_trie_node.get_child_by_key(&edge).is_none());
    assert_eq!(0, parent_trie_node.child_pin_count());
}

#[test]
fn test_insert_peek_by_parent_w_parent_tombstone_node() {
    let trie = new_trie();
    let parent_trie_node_key = trie.alloc_trie_node(
        None,
        new_annotations(1),
        new_tokens(&[1]),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    let edge = new_edge(2, &[2, 3]);
    let trie_node_key = trie.alloc_trie_node(
        Some(parent_trie_node_key),
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.insert_by_parent(Some(parent_trie_node_key), &edge, trie_node_key);
    let mut trie_node = trie.peek_by_parent(Some(parent_trie_node_key), &edge).unwrap();
    assert_eq!(0, trie_node.external_unpin());
    assert!(trie_node.try_mark_tombstone());
    drop(trie_node);

    assert!(trie.peek_by_parent(Some(parent_trie_node_key), &edge).is_none());
    let parent_trie_node = trie.peek_by_key(parent_trie_node_key);
    assert!(parent_trie_node.get_child_by_key(&edge).is_none());
    assert_eq!(0, parent_trie_node.child_pin_count());
}

#[test]
fn test_insert_by_parent_w_parent_replaces_tombstone_collision() {
    let trie = new_trie();
    let parent_trie_node_key = trie.alloc_trie_node(
        None,
        new_annotations(4),
        new_tokens(&[4]),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    let edge = new_edge(5, &[5, 6]);
    let from_trie_node_key = trie.alloc_trie_node(
        Some(parent_trie_node_key),
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.insert_by_parent(Some(parent_trie_node_key), &edge, from_trie_node_key);
    assert_eq!(0, trie.external_unpin_by_key(from_trie_node_key));
    assert!(trie.peek_by_key(from_trie_node_key).try_mark_tombstone());

    let to_trie_node_key = trie.alloc_trie_node(
        Some(parent_trie_node_key),
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    let result = trie.insert_by_parent(Some(parent_trie_node_key), &edge, to_trie_node_key);
    match result {
        InsertNodeResult::Success { trie_node_key } => assert_eq!(to_trie_node_key, trie_node_key),
        InsertNodeResult::Collision { .. } => unreachable!(),
    }
    assert_eq!(2, trie.num_pinned_trie_node());

    let trie_node = trie.peek_by_parent(Some(parent_trie_node_key), &edge).unwrap();
    assert_eq!(to_trie_node_key, trie_node.key());
    drop(trie_node);

    let parent_trie_node = trie.peek_by_key(parent_trie_node_key);
    assert_eq!(Some(to_trie_node_key), parent_trie_node.get_child_by_key(&edge));
    drop(parent_trie_node);

    trie.external_unpin_by_key(to_trie_node_key);
    trie.remove_by_parent(Some(parent_trie_node_key), &edge, to_trie_node_key);
    let parent_trie_node = trie.peek_by_key(parent_trie_node_key);
    assert!(parent_trie_node.get_child_by_key(&edge).is_none());
    assert_eq!(0, parent_trie_node.child_pin_count());
}

#[test]
fn test_insert_by_parent_w_parent_collision_valid_collision() {
    let trie = new_trie();
    let parent_trie_node_key = trie.alloc_trie_node(
        None,
        new_annotations(6),
        new_tokens(&[6]),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    let edge = new_edge(7, &[7, 8]);
    let existing_trie_node_key = trie.alloc_trie_node(
        Some(parent_trie_node_key),
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.insert_by_parent(Some(parent_trie_node_key), &edge, existing_trie_node_key);

    let new_trie_node_key = trie.alloc_trie_node(
        Some(parent_trie_node_key),
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    let result = trie.insert_by_parent(Some(parent_trie_node_key), &edge, new_trie_node_key);
    match result {
        InsertNodeResult::Success { .. } => unreachable!(),
        InsertNodeResult::Collision { trie_node_key } => assert_eq!(existing_trie_node_key, trie_node_key),
    }
    assert_eq!(3, trie.num_pinned_trie_node());

    let trie_node = trie.peek_by_parent(Some(parent_trie_node_key), &edge).unwrap();
    assert_eq!(existing_trie_node_key, trie_node.key());
    assert_eq!(2, trie_node.external_pin_count());
    drop(trie_node);

    let parent_trie_node = trie.peek_by_key(parent_trie_node_key);
    assert_eq!(Some(existing_trie_node_key), parent_trie_node.get_child_by_key(&edge));
    drop(parent_trie_node);

    trie.free_trie_node(new_trie_node_key, 1);
}

#[test]
fn test_insert_peek_by_parent_w_parent_missing_edge() {
    let trie = new_trie();
    let parent_trie_node_key = trie.alloc_trie_node(
        None,
        new_annotations(1),
        new_tokens(&[1]),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    let edge = new_edge(2, &[2, 3]);
    let trie_node_key = trie.alloc_trie_node(
        Some(parent_trie_node_key),
        edge.annotations().clone(),
        edge.tokens().clone(),
        new_kv_placement(),
        new_state_placement(),
        1,
    );
    trie.insert_by_parent(Some(parent_trie_node_key), &edge, trie_node_key);
    trie.external_unpin_by_key(trie_node_key);
    trie.remove_by_parent(Some(parent_trie_node_key), &edge, trie_node_key);

    assert!(trie.peek_by_parent(Some(parent_trie_node_key), &edge).is_none());
    let parent_trie_node = trie.peek_by_key(parent_trie_node_key);
    assert!(parent_trie_node.get_child_by_key(&edge).is_none());
    assert_eq!(0, parent_trie_node.child_pin_count());
}
