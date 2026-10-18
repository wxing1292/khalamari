use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures_util::pin_mut;
use futures_util::task::noop_waker_ref;
use ordered_float::NotNan;
use smallvec::SmallVec;

use super::api_test_support::concrete_resource;
use crate::channel::Shutdown;
use crate::compute::DecoderSyncBlocks;
use crate::compute::QueryTokens;
use crate::compute::SampledTokens;
use crate::config::SamplingConfig;
use crate::memory::BlockAllocator;
use crate::memory::U32IDAllocator;
use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;
use crate::runtime::decoder::TPStateBlockAllocator;
use crate::runtime::decoder::allocator::TPKVBlockAllocator;
use crate::runtime::decoder::trie_cache::AllocateMultiLaneMutableBlockResult;
use crate::runtime::decoder::trie_cache::BlockMetadata;
use crate::runtime::decoder::trie_cache::CommitMultiLaneMutableBlockResult;
use crate::runtime::decoder::trie_cache::CommitMultiLaneSemiImmutableBlockResult;
use crate::runtime::decoder::trie_cache::DecoderBlock;
use crate::runtime::decoder::trie_cache::DecoderBlocks;
use crate::runtime::decoder::trie_cache::ImmutableBlock;
use crate::runtime::decoder::trie_cache::InitBlockOnceResult;
use crate::runtime::decoder::trie_cache::MultiLaneBlockCache;
use crate::runtime::decoder::trie_cache::MultiLaneTrieBlockCache;
use crate::runtime::decoder::trie_cache::MutableBlock;
use crate::runtime::decoder::trie_cache::ReserveMultiLaneSemiImmutableBlockResult;
use crate::runtime::decoder::trie_cache::SemiImmutableBlock;
use crate::runtime::decoder::trie_cache::SingleLaneTrieBlockCache;
use crate::runtime::decoder::trie_cache::TrieDecoderBlocks;
use crate::runtime::decoder::trie_cache::UninitBlockOnceResult;
use crate::runtime::request::AtomicRequestStatus;
use crate::runtime::request::InternalRequest;
use crate::runtime::request::RequestSlotAllocationResult;
use crate::runtime::request::RequestSlotAllocator;
use crate::runtime::resource::Resource;
use crate::runtime::resource::ResourceID;
use crate::runtime::resource::ResourcePlacement;
use crate::runtime::resource::ResourceTypeID;
use crate::runtime::resource::ResourceURI;
use crate::runtime::resource::SymbolicResource;
use crate::runtime::resource::processor::ResourceProcessors;
use crate::runtime::scheduler::PrepareResult;
use crate::runtime::scheduler::UserRequest;

const NUM_KV_PAGES_PER_BLOCK: usize = 1;
const NUM_STATE_PAGES_PER_BLOCK: usize = 1;
const BLOCK_CACHE_CAPACITY: usize = 1024;

const NUM_TOKEN_PER_BLOCK: usize = 4;
const NUM_TRIE_PARTITION: usize = 4;
const NUM_CACHE_LANE: usize = 4;

type TestSingleLaneTrieKVBlockCache =
    SingleLaneTrieBlockCache<NUM_TRIE_PARTITION, TPKVBlockAllocator, TPStateBlockAllocator>;
type TestMultiLaneTrieBlockCache =
    MultiLaneTrieBlockCache<NUM_TRIE_PARTITION, NUM_CACHE_LANE, TPKVBlockAllocator, TPStateBlockAllocator>;
type TestTrieKVBlocks =
    TrieDecoderBlocks<NUM_TOKEN_PER_BLOCK, NUM_TRIE_PARTITION, NUM_CACHE_LANE, TestMultiLaneTrieBlockCache>;

type TestImmutableBlock = ImmutableBlock<
    NUM_TRIE_PARTITION,
    <TestMultiLaneTrieBlockCache as MultiLaneBlockCache<NUM_TRIE_PARTITION, NUM_CACHE_LANE>>::Entry,
>;

#[test]
fn test_init_block_once_half_block_success_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);
    assert_block_counts(&blocks, 0, 0, 1);
    assert_total_tokens(
        &blocks.mutable_blocks[0],
        [
            token_vec([0, 1]),
            token_vec([1, 2]),
            token_vec([2, 3]),
            token_vec([3, 4]),
        ],
    );
    assert_mutable_annotations(&blocks.mutable_blocks[0], root_annotations([0, 1, 2]));
    assert_state(&blocks, 0, &[], &[], &token_vec([0, 1]), &token_vec([2, 3, 4]), &[]);
}

#[test]
fn test_init_block_once_half_block_resource_limit_exceeded_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4]);
    let block_cache = initialize_block_cache([2; NUM_CACHE_LANE]);
    let random_block_vec = allocate_mutable_block(&block_cache);
    let mut blocks = initialize_blocks(block_cache.clone(), total_tokens.clone());

    let InitBlockOnceResult::ResourceLimitExceeded = blocks.init_block_once() else {
        unreachable!()
    };

    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([0, 1, 2, 3, 4]), &[]);
    block_cache.free_mutable_block(random_block_vec);
}

#[test]
fn test_init_block_once_full_block_cache_hit_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    insert_immutable_block(&block_cache, 0, None, total_tokens.clone());
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_eq!(0, ready_token_slots);
    assert_block_counts(&blocks, 1, 0, 0);
    assert_total_tokens(
        &blocks.immutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );
    assert_immutable_annotations(&blocks.immutable_blocks[0], root_annotations([0, 1, 2]));
    assert_state(
        &blocks,
        0,
        &token_vec([0, 1, 2, 3]),
        &[],
        &[],
        &token_vec([4, 5, 6]),
        &[],
    );
}

#[test]
fn test_init_block_once_full_block_cache_miss_no_semi_immutable_success_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);
    assert_block_counts(&blocks, 0, 1, 0);
    assert_total_tokens(
        &blocks.semi_immutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );
    assert_semi_immutable_annotations(&blocks.semi_immutable_blocks[0], root_annotations([0, 1, 2]));
    assert_state(
        &blocks,
        0,
        &[],
        &[],
        &token_vec([0, 1, 2, 3]),
        &token_vec([4, 5, 6]),
        &[],
    );
}

#[test]
fn test_init_block_once_full_block_cache_miss_resource_not_found_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    let placement = ResourcePlacement::new(resource_id, vec![(1, 0, 2)], total_tokens.len());
    let mut blocks = TestTrieKVBlocks::new(
        block_cache,
        vec![symbolic_resource(resource_id)],
        vec![placement],
        std::iter::empty::<Token>(),
        std::iter::empty::<Token>(),
        total_tokens.clone(),
    );

    let InitBlockOnceResult::ResourceNotFound {
        ready_token_slots,
        resource_ids,
    } = blocks.init_block_once()
    else {
        unreachable!()
    };

    assert_eq!(0, ready_token_slots);
    assert_eq!(vec![resource_id], resource_ids);
    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &total_tokens, &[]);
}

#[test]
fn test_init_block_once_full_block_cache_miss_resource_found_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    let placement = ResourcePlacement::new(resource_id, vec![(1, 0, 2)], total_tokens.len());
    let (resource, _resource_allocator) = concrete_resource(resource_id, 2);
    let mut blocks = TestTrieKVBlocks::new(
        block_cache,
        vec![resource],
        vec![placement],
        std::iter::empty::<Token>(),
        std::iter::empty::<Token>(),
        total_tokens,
    );
    let expected_annotations = blocks.block_annotation_vec(0).map(SmallVec::into_vec);

    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);
    assert_block_counts(&blocks, 0, 1, 0);
    assert_total_tokens(
        &blocks.semi_immutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );
    assert_semi_immutable_annotations(&blocks.semi_immutable_blocks[0], expected_annotations);
    assert_state(
        &blocks,
        0,
        &[],
        &[],
        &token_vec([0, 1, 2, 3]),
        &token_vec([4, 5, 6]),
        &[],
    );
}

#[test]
fn test_init_block_once_full_block_cache_hit_bypasses_resource_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    let placement = ResourcePlacement::new(resource_id, vec![(1, 0, 2)], total_tokens.len());
    let mut blocks = TestTrieKVBlocks::new(
        block_cache.clone(),
        vec![symbolic_resource(resource_id)],
        vec![placement],
        std::iter::empty::<Token>(),
        std::iter::empty::<Token>(),
        total_tokens.clone(),
    );
    insert_immutable_block_with_annotations(&block_cache, total_tokens, blocks.block_annotation_vec(0));
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_eq!(0, ready_token_slots);
    assert_block_counts(&blocks, 1, 0, 0);
}

#[test]
fn test_init_block_once_full_block_cache_miss_no_semi_immutable_resource_limit_exceeded_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache([2; NUM_CACHE_LANE]);
    let random_block_vec = allocate_mutable_block(&block_cache);
    let mut blocks = initialize_blocks(block_cache.clone(), total_tokens.clone());
    let InitBlockOnceResult::ResourceLimitExceeded = blocks.init_block_once() else {
        unreachable!()
    };

    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([0, 1, 2, 3, 4, 5, 6]), &[]);
    block_cache.free_mutable_block(random_block_vec);
}

#[test]
fn test_init_block_once_full_block_cache_reserved_await_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut random_block_vec = reserve_semi_immutable_block(&block_cache, 0, None, total_tokens.clone());
    let mut blocks = initialize_blocks(block_cache.clone(), total_tokens.clone());
    let InitBlockOnceResult::Await { wait } = blocks.init_block_once() else {
        unreachable!()
    };
    pin_mut!(wait);
    let mut cx = Context::from_waker(noop_waker_ref());

    assert!(matches!(wait.as_mut().poll(&mut cx), Poll::Pending));
    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([0, 1, 2, 3, 4, 5, 6]), &[]);

    for block in &mut random_block_vec {
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(scheduled_tokens.len());
    }
    let CommitMultiLaneSemiImmutableBlockResult::Immutable {
        block_vec: random_block_vec,
    } = block_cache.commit_semi_immutable_block(random_block_vec)
    else {
        unreachable!()
    };

    assert!(matches!(wait.as_mut().poll(&mut cx), Poll::Ready(())));
    assert_total_tokens(
        &random_block_vec,
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );

    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_eq!(0, ready_token_slots);
    assert_block_counts(&blocks, 1, 0, 0);
    assert_total_tokens(
        &blocks.immutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );
    assert_immutable_annotations(&blocks.immutable_blocks[0], root_annotations([0, 1, 2]));
    assert_state(
        &blocks,
        0,
        &token_vec([0, 1, 2, 3]),
        &[],
        &[],
        &token_vec([4, 5, 6]),
        &[],
    );
}

#[test]
fn test_init_block_once_full_block_cache_bypass_with_semi_immutable_success_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_eq!(NUM_TOKEN_PER_BLOCK * 2, ready_token_slots);
    assert_block_counts(&blocks, 0, 1, 1);
    assert_total_tokens(
        &blocks.semi_immutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );
    assert_semi_immutable_annotations(&blocks.semi_immutable_blocks[0], root_annotations([0, 1, 2]));
    assert_total_tokens(
        &blocks.mutable_blocks[0],
        [
            token_vec([4, 5, 6, 7]),
            token_vec([5, 6, 7, 8]),
            token_vec([6, 7, 8, 9]),
            token_vec([7, 8, 9, 10]),
        ],
    );
    assert_mutable_annotations(&blocks.mutable_blocks[0], empty_annotations());
    assert_state(
        &blocks,
        0,
        &[],
        &[],
        &token_vec([0, 1, 2, 3, 4, 5, 6, 7]),
        &token_vec([8, 9, 10]),
        &[],
    );
}

#[test]
fn test_init_block_once_full_block_cache_bypass_with_semi_immutable_resource_limit_exceeded_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let block_cache = initialize_block_cache([2; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };
    let InitBlockOnceResult::ResourceLimitExceeded = blocks.init_block_once() else {
        unreachable!()
    };

    assert_block_counts(&blocks, 0, 1, 0);
    assert_total_tokens(
        &blocks.semi_immutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );
    assert_semi_immutable_annotations(&blocks.semi_immutable_blocks[0], root_annotations([0, 1, 2]));
    assert_state(
        &blocks,
        0,
        &[],
        &[],
        &token_vec([0, 1, 2, 3]),
        &token_vec([4, 5, 6, 7, 8, 9, 10]),
        &[],
    );
}

#[test]
fn test_init_block_once_full_and_half_block_cache_bypass_with_mutable_success_w_mtp() {
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache, token_vec([0, 1, 2, 3, 4]));

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };
    blocks.queued_tokens.extend(token_vec([5, 6, 7, 8, 9, 10, 11, 12]));
    blocks.try_mark_ready();

    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_eq!(NUM_TOKEN_PER_BLOCK * 2, ready_token_slots);
    assert_block_counts(&blocks, 0, 0, 2);
    assert_total_tokens(
        &blocks.mutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );
    assert_mutable_annotations(&blocks.mutable_blocks[0], root_annotations([0, 1, 2]));
    assert_total_tokens(
        &blocks.mutable_blocks[1],
        [
            token_vec([4, 5, 6, 7]),
            token_vec([5, 6, 7, 8]),
            token_vec([6, 7, 8, 9]),
            token_vec([7, 8, 9, 10]),
        ],
    );
    assert_mutable_annotations(&blocks.mutable_blocks[1], empty_annotations());
    assert_state(
        &blocks,
        0,
        &[],
        &[],
        &token_vec([0, 1, 2, 3, 4, 5, 6, 7]),
        &token_vec([8, 9, 10, 11, 12]),
        &[],
    );
}

#[test]
fn test_init_block_once_full_and_half_block_cache_bypass_with_mutable_resource_limit_exceeded_w_mtp() {
    let block_cache = initialize_block_cache([2; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache, token_vec([0, 1, 2, 3, 4]));

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };
    blocks.queued_tokens.extend(token_vec([5, 6, 7, 8, 9, 10, 11, 12]));
    blocks.try_mark_ready();

    let InitBlockOnceResult::ResourceLimitExceeded = blocks.init_block_once() else {
        unreachable!()
    };

    assert_block_counts(&blocks, 0, 0, 1);
    assert_total_tokens(
        &blocks.mutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );
    assert_mutable_annotations(&blocks.mutable_blocks[0], root_annotations([0, 1, 2]));
    assert_state(
        &blocks,
        0,
        &[],
        &[],
        &token_vec([0, 1, 2, 3]),
        &token_vec([4, 5, 6, 7, 8, 9, 10, 11, 12]),
        &[],
    );
}

#[test]
fn test_uninit_block_once_noop_block_w_mtp() {
    let total_tokens = token_vec([0, 1, 2]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([0, 1, 2]), &[]);

    let UninitBlockOnceResult::Success { cached_token_slots } = blocks.uninit_block_once();

    assert_eq!(0, cached_token_slots);
    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([0, 1, 2]), &[]);
}

#[test]
fn test_uninit_block_once_mutable_block_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_block_counts(&blocks, 0, 0, 1);
    assert_total_tokens(
        &blocks.mutable_blocks[0],
        [
            token_vec([0, 1]),
            token_vec([1, 2]),
            token_vec([2, 3]),
            token_vec([3, 4]),
        ],
    );
    assert_state(&blocks, 0, &[], &[], &token_vec([0, 1]), &token_vec([2, 3, 4]), &[]);

    let UninitBlockOnceResult::Success { cached_token_slots } = blocks.uninit_block_once();

    assert_eq!(0, cached_token_slots);
    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([0, 1, 2, 3, 4]), &[]);
}

#[test]
fn test_uninit_block_once_semi_immutable_block_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_block_counts(&blocks, 0, 1, 0);
    assert_total_tokens(
        &blocks.semi_immutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );
    assert_state(
        &blocks,
        0,
        &[],
        &[],
        &token_vec([0, 1, 2, 3]),
        &token_vec([4, 5, 6]),
        &[],
    );

    let UninitBlockOnceResult::Success { cached_token_slots } = blocks.uninit_block_once();

    assert_eq!(0, cached_token_slots);
    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([0, 1, 2, 3, 4, 5, 6]), &[]);
}

#[test]
fn test_uninit_block_once_immutable_block_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    insert_immutable_block(&block_cache, 0, None, total_tokens.clone());
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(0, ready_token_slots);

    assert_block_counts(&blocks, 1, 0, 0);
    assert_total_tokens(
        &blocks.immutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );
    assert_state(
        &blocks,
        0,
        &token_vec([0, 1, 2, 3]),
        &[],
        &[],
        &token_vec([4, 5, 6]),
        &[],
    );

    let UninitBlockOnceResult::Success { cached_token_slots } = blocks.uninit_block_once();

    assert_eq!(0, cached_token_slots);
    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([0, 1, 2, 3, 4, 5, 6]), &[]);
}

#[test]
#[cfg(debug_assertions)]
fn test_write_tokens_fixed_identity_w_mtp() {
    for cache_hit in [false, true] {
        let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6]);
        let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
        if cache_hit {
            insert_immutable_block(&block_cache, 0, None, total_tokens.clone());
        }
        let mut blocks = initialize_blocks(block_cache, total_tokens.clone());
        let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
            unreachable!()
        };
        let num_cached_tokens = blocks.num_cached_tokens();
        let mut tokens = token_vec([1, 2, 3, 4, 5]);
        blocks.write_tokens(1, 3, &tokens);

        // Only MTP2's final supplied token differs. Fixed lanes must not accept it.
        tokens[4] = Token::new(99);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            blocks.write_tokens(1, 3, &tokens);
        }));
        assert!(result.is_err());
        assert_eq!(num_cached_tokens, blocks.num_cached_tokens());
        assert_eq!(total_tokens, blocks.total_tokens().collect::<Vec<_>>());
        blocks.sanity_check();
    }
}

#[test]
#[cfg(debug_assertions)]
fn test_write_tokens_invalid_range_w_mtp() {
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache, token_vec([0, 1, 2, 3, 4]));
    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };
    for (token_index_start, token_index_end, tokens) in [
        (2, 2, token_vec([2, 3, 4])),
        (2, 1, token_vec([2, 3, 4])),
        (0, 5, token_vec([0, 1, 2, 3, 4, 5, 6, 7])),
        (0, 2, token_vec([0, 1])),
    ] {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            blocks.write_tokens(token_index_start, token_index_end, &tokens);
        }));
        assert!(result.is_err());
        blocks.sanity_check();
    }
}

#[test]
fn test_prepare_cancel_commit_prefill_zero_token_index_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);

    let query_tokens = blocks.prepare(1).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    match &query_tokens {
        QueryTokens::Prefill {
            epoch,
            token_index,
            tokens,
            window,
        } => {
            assert_eq!(0, *epoch);
            assert_eq!(0, *token_index);
            assert_eq!(total_tokens, *tokens);
            assert_eq!(1, *window);
        },
        _ => unreachable!(),
    }
    let expected_query_tokens = query_tokens.clone();
    let expected_sync_blocks = sync_blocks.clone();
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_state(&blocks, 1, &[], &token_vec([0]), &[], &token_vec([1, 2, 3]), &[]);

    blocks.cancel_blocks(sync_blocks);
    blocks.cancel(query_tokens);
    assert_state(&blocks, 0, &[], &[], &token_vec([0]), &token_vec([1, 2, 3]), &[]);

    let query_tokens = blocks.prepare(1).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    assert_eq!(expected_query_tokens, query_tokens);
    assert_eq!(expected_sync_blocks, sync_blocks);
    assert_state(&blocks, 1, &[], &token_vec([0]), &[], &token_vec([1, 2, 3]), &[]);

    let epoch = query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(query_tokens, SampledTokens::Prefill { epoch });
    assert_state(&blocks, 1, &token_vec([0]), &[], &[], &token_vec([1, 2, 3]), &[]);
}

#[test]
fn test_prepare_resource_not_found_w_mtp() {
    const REQUEST_ID: usize = 1;

    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    let placement = ResourcePlacement::new(resource_id, vec![(1, 0, 2)], total_tokens.len());
    let blocks = TestTrieKVBlocks::new(
        block_cache,
        vec![symbolic_resource(resource_id)],
        vec![placement],
        std::iter::empty::<Token>(),
        std::iter::empty::<Token>(),
        total_tokens,
    );
    let context_window = blocks.num_total_tokens() + 1;
    let (request_slot_allocator, _reset_rx) = RequestSlotAllocator::new(1, Shutdown::new());
    let RequestSlotAllocationResult::Ok { request_slot } = request_slot_allocator.allocate() else {
        unreachable!()
    };
    let (token_prob_tx, _token_prob_rx) = async_channel::bounded(1);
    let mut request = InternalRequest::new(
        REQUEST_ID,
        request_slot,
        AtomicRequestStatus::new(),
        blocks,
        None,
        Arc::new(ResourceProcessors::new()),
        token_prob_tx,
        SamplingConfig::default(),
        context_window,
    );

    let PrepareResult::BlockingAsyncTask { req } = request.prepare(NUM_TOKEN_PER_BLOCK) else {
        unreachable!()
    };

    assert_eq!(REQUEST_ID, req.request_id());
    assert_eq!(1, request.num_in_flight_blocking_async_tasks());
}

#[test]
fn test_prepare_commit_resource_unload_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    let placement = ResourcePlacement::new(resource_id, vec![(1, 0, 2), (5, 2, 1)], total_tokens.len());
    let (resource, resource_allocator) = concrete_resource(resource_id, 3);
    let mut blocks = TestTrieKVBlocks::new(
        block_cache,
        vec![resource],
        vec![placement],
        std::iter::empty::<Token>(),
        std::iter::empty::<Token>(),
        total_tokens,
    );

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };
    let query_tokens = blocks.prepare(NUM_TOKEN_PER_BLOCK).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    let epoch = query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(query_tokens, SampledTokens::Prefill { epoch });
    assert!(blocks.resources[&resource_id].is_concrete());

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };
    let query_tokens = blocks.prepare(2).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    let epoch = query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(query_tokens, SampledTokens::Prefill { epoch });
    assert_eq!(6, blocks.num_cached_tokens());
    assert!(blocks.resources[&resource_id].is_symbolic());
    assert_eq!(0, resource_allocator.alloc_segment(12).unwrap().offset_bytes());
}

#[test]
fn test_prepare_cancel_commit_prefill_nonzero_token_index_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6, 7]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    insert_immutable_block(&block_cache, 0, None, token_vec([0, 1, 2, 3, 4, 5, 6]));
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(0, ready_token_slots);
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);

    let query_tokens = blocks.prepare(1).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    match &query_tokens {
        QueryTokens::Prefill {
            epoch,
            token_index,
            tokens,
            window,
        } => {
            assert_eq!(0, *epoch);
            assert_eq!(NUM_TOKEN_PER_BLOCK, *token_index);
            assert_eq!(token_vec([4, 5, 6, 7]), *tokens);
            assert_eq!(1, *window);
        },
        _ => unreachable!(),
    }
    let expected_query_tokens = query_tokens.clone();
    let expected_sync_blocks = sync_blocks.clone();
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_state(
        &blocks,
        2,
        &token_vec([0, 1, 2, 3]),
        &token_vec([4]),
        &[],
        &token_vec([5, 6, 7]),
        &[],
    );

    blocks.cancel_blocks(sync_blocks);
    blocks.cancel(query_tokens);
    assert_state(
        &blocks,
        0,
        &token_vec([0, 1, 2, 3]),
        &[],
        &token_vec([4]),
        &token_vec([5, 6, 7]),
        &[],
    );

    let query_tokens = blocks.prepare(1).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    assert_eq!(expected_query_tokens, query_tokens);
    assert_eq!(expected_sync_blocks, sync_blocks);
    assert_state(
        &blocks,
        2,
        &token_vec([0, 1, 2, 3]),
        &token_vec([4]),
        &[],
        &token_vec([5, 6, 7]),
        &[],
    );

    let epoch = query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(query_tokens, SampledTokens::Prefill { epoch });
    assert_state(
        &blocks,
        2,
        &token_vec([0, 1, 2, 3, 4]),
        &[],
        &[],
        &token_vec([5, 6, 7]),
        &[],
    );
}

#[test]
fn test_prepare_cancel_commit_overlapping_prefill_decode_w_mtp() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);

    let prefill_query_tokens = blocks.prepare(4).unwrap();
    let prefill_sync_blocks = blocks.prepare_blocks();
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert!(3 <= ready_token_slots);
    let decode_query_tokens = blocks.prepare(3).unwrap();
    let decode_sync_blocks = blocks.prepare_blocks();
    assert_eq!(
        QueryTokens::Prefill {
            epoch: 0,
            token_index: 0,
            tokens: total_tokens.clone(),
            window: 4,
        },
        prefill_query_tokens
    );
    assert_eq!(
        QueryTokens::Decode {
            epoch: 0,
            token_index: 4,
            tokens: token_vec([4, 5, 6]),
            spec_tokens: token_vec([]),
        },
        decode_query_tokens
    );
    let expected_prefill_query_tokens = prefill_query_tokens.clone();
    let expected_decode_query_tokens = decode_query_tokens.clone();
    assert_state(&blocks, 2, &[], &token_vec([0, 1, 2, 3, 4, 5, 6]), &[], &[], &[]);

    blocks.cancel_blocks(decode_sync_blocks);
    blocks.cancel(decode_query_tokens);
    blocks.cancel_blocks(prefill_sync_blocks);
    blocks.cancel(prefill_query_tokens);
    assert_state(&blocks, 0, &[], &[], &token_vec([0, 1, 2, 3, 4, 5, 6]), &[], &[]);

    let prefill_query_tokens = blocks.prepare(4).unwrap();
    let prefill_sync_blocks = blocks.prepare_blocks();
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert!(3 <= ready_token_slots);
    let decode_query_tokens = blocks.prepare(3).unwrap();
    let decode_sync_blocks = blocks.prepare_blocks();
    assert_eq!(expected_prefill_query_tokens, prefill_query_tokens);
    assert_eq!(expected_decode_query_tokens, decode_query_tokens);

    let epoch = prefill_query_tokens.epoch();
    blocks.commit_blocks(prefill_sync_blocks);
    blocks.commit(prefill_query_tokens, SampledTokens::Prefill { epoch });
    let epoch = decode_query_tokens.epoch();
    blocks.commit_blocks(decode_sync_blocks);
    blocks.commit(
        decode_query_tokens,
        SampledTokens::Decode {
            epoch,
            validated_tokens: token_vec([]),
            validated_probs: vec![],
            sampled_token: Token::new(7),
            sampled_prob: NotNan::new(0.5).unwrap(),
            spec_tokens: token_vec([8, 9, 10]),
            spec_probs: vec![NotNan::new(0.5).unwrap(); 3],
            spec_confidences: vec![NotNan::new(0.5).unwrap(); 3],
        },
    );
    assert_state(
        &blocks,
        3,
        &token_vec([0, 1, 2, 3, 4, 5, 6]),
        &[],
        &[],
        &token_vec([7]),
        &token_vec([8, 9, 10]),
    );
}

#[test]
fn test_prepare_cancel_commit_decode_zero_token_index_w_mtp_w_spec_token() {
    let total_tokens = token_vec([0, 1, 2, 3]);
    let spec_tokens = token_vec([10, 11, 12]);
    let output_sampled_token = Token::new(20);
    let output_spec_tokens = token_vec([30, 31]);
    let output_spec_probs = vec![NotNan::new(0.6).unwrap(); output_spec_tokens.len()];
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());
    let mut ready_token_slots = 0;
    while ready_token_slots < 6 {
        let InitBlockOnceResult::Success {
            ready_token_slots: new_ready_token_slots,
        } = blocks.init_block_once()
        else {
            unreachable!()
        };
        ready_token_slots = new_ready_token_slots;
    }
    assert!(6 <= ready_token_slots);
    blocks.spec_tokens = spec_tokens.clone();
    blocks.spec_probs = vec![NotNan::new(1.0).unwrap(); spec_tokens.len()];
    blocks.spec_confidences = vec![NotNan::new(1.0).unwrap(); spec_tokens.len()];

    let query_tokens = blocks.prepare(6).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    match &query_tokens {
        QueryTokens::Decode {
            epoch,
            token_index,
            tokens,
            spec_tokens,
        } => {
            assert_eq!(0, *epoch);
            assert_eq!(0, *token_index);
            assert_eq!(total_tokens, *tokens);
            assert_eq!(token_vec([10, 11]), *spec_tokens);
        },
        _ => unreachable!(),
    }
    let expected_query_tokens = query_tokens.clone();
    let expected_sync_blocks = sync_blocks.clone();
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_state(
        &blocks,
        2,
        &[],
        &token_vec([0, 1, 2, 3]),
        &[],
        &[],
        &token_vec([10, 11]),
    );

    blocks.cancel_blocks(sync_blocks);
    blocks.cancel(query_tokens);
    assert_state(
        &blocks,
        0,
        &[],
        &[],
        &token_vec([0, 1, 2, 3]),
        &[],
        &token_vec([10, 11]),
    );

    let query_tokens = blocks.prepare(6).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    assert_eq!(expected_query_tokens, query_tokens);
    assert_eq!(expected_sync_blocks, sync_blocks);
    assert_state(
        &blocks,
        2,
        &[],
        &token_vec([0, 1, 2, 3]),
        &[],
        &[],
        &token_vec([10, 11]),
    );

    let epoch = query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(
        query_tokens,
        SampledTokens::Decode {
            epoch,
            validated_tokens: token_vec([10]),
            validated_probs: vec![NotNan::new(0.5).unwrap()],
            sampled_token: output_sampled_token,
            sampled_prob: NotNan::new(0.5).unwrap(),
            spec_confidences: vec![NotNan::new(1.0).unwrap(); output_spec_tokens.len()],
            spec_tokens: output_spec_tokens,
            spec_probs: output_spec_probs,
        },
    );
    assert_state(
        &blocks,
        2,
        &token_vec([0, 1, 2, 3, 10]),
        &[],
        &[],
        &token_vec([20]),
        &token_vec([30, 31]),
    );
}

#[test]
fn test_prepare_cancel_commit_decode_nonzero_token_index_w_mtp_w_spec_token() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6, 7]);
    let spec_tokens = token_vec([10, 11, 12]);
    let output_sampled_token = Token::new(20);
    let output_spec_tokens = token_vec([30, 31]);
    let output_spec_probs = vec![NotNan::new(0.6).unwrap(); output_spec_tokens.len()];
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    insert_immutable_block(&block_cache, 0, None, token_vec([0, 1, 2, 3, 4, 5, 6]));
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(0, ready_token_slots);
    let mut ready_token_slots = ready_token_slots;
    while ready_token_slots < 6 {
        let InitBlockOnceResult::Success {
            ready_token_slots: new_ready_token_slots,
        } = blocks.init_block_once()
        else {
            unreachable!()
        };
        ready_token_slots = new_ready_token_slots;
    }
    assert!(6 <= ready_token_slots);
    blocks.spec_tokens = spec_tokens.clone();
    blocks.spec_probs = vec![NotNan::new(1.0).unwrap(); spec_tokens.len()];
    blocks.spec_confidences = vec![NotNan::new(1.0).unwrap(); spec_tokens.len()];

    let query_tokens = blocks.prepare(6).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    match &query_tokens {
        QueryTokens::Decode {
            epoch,
            token_index,
            tokens,
            spec_tokens,
        } => {
            assert_eq!(0, *epoch);
            assert_eq!(NUM_TOKEN_PER_BLOCK, *token_index);
            assert_eq!(token_vec([4, 5, 6, 7]), *tokens);
            assert_eq!(token_vec([10, 11]), *spec_tokens);
        },
        _ => unreachable!(),
    }
    let expected_query_tokens = query_tokens.clone();
    let expected_sync_blocks = sync_blocks.clone();
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_state(
        &blocks,
        3,
        &token_vec([0, 1, 2, 3]),
        &token_vec([4, 5, 6, 7]),
        &[],
        &[],
        &token_vec([10, 11]),
    );

    blocks.cancel_blocks(sync_blocks);
    blocks.cancel(query_tokens);
    assert_state(
        &blocks,
        0,
        &token_vec([0, 1, 2, 3]),
        &[],
        &token_vec([4, 5, 6, 7]),
        &[],
        &token_vec([10, 11]),
    );

    let query_tokens = blocks.prepare(6).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    assert_eq!(expected_query_tokens, query_tokens);
    assert_eq!(expected_sync_blocks, sync_blocks);
    assert_state(
        &blocks,
        3,
        &token_vec([0, 1, 2, 3]),
        &token_vec([4, 5, 6, 7]),
        &[],
        &[],
        &token_vec([10, 11]),
    );

    let epoch = query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(
        query_tokens,
        SampledTokens::Decode {
            epoch,
            validated_tokens: token_vec([10]),
            validated_probs: vec![NotNan::new(0.5).unwrap()],
            sampled_token: output_sampled_token,
            sampled_prob: NotNan::new(0.5).unwrap(),
            spec_confidences: vec![NotNan::new(1.0).unwrap(); output_spec_tokens.len()],
            spec_tokens: output_spec_tokens.clone(),
            spec_probs: output_spec_probs,
        },
    );
    assert_state(
        &blocks,
        3,
        &token_vec([0, 1, 2, 3, 4, 5, 6, 7, 10]),
        &[],
        &[],
        &token_vec([20]),
        &output_spec_tokens,
    );
}

#[test]
fn test_prepare_commit_full_block_semi_immutable_collision_w_mtp() {
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache.clone(), token_vec([0, 1, 2, 3, 4, 5, 6]));
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);
    assert_block_counts(&blocks, 0, 1, 0);
    assert_total_tokens(
        &blocks.semi_immutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );
    assert_semi_immutable_annotations(&blocks.semi_immutable_blocks[0], root_annotations([0, 1, 2]));
    assert_state(
        &blocks,
        0,
        &[],
        &[],
        &token_vec([0, 1, 2, 3]),
        &token_vec([4, 5, 6]),
        &[],
    );

    let query_tokens = blocks.prepare(NUM_TOKEN_PER_BLOCK).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    match &query_tokens {
        QueryTokens::Prefill {
            epoch,
            token_index,
            tokens,
            window,
        } => {
            assert_eq!(0, *epoch);
            assert_eq!(0, *token_index);
            assert_eq!(token_vec([0, 1, 2, 3, 4, 5, 6]), *tokens);
            assert_eq!(NUM_TOKEN_PER_BLOCK, *window);
        },
        _ => unreachable!(),
    }
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_state(
        &blocks,
        1,
        &[],
        &token_vec([0, 1, 2, 3]),
        &[],
        &token_vec([4, 5, 6]),
        &[],
    );

    let random_block_vec = insert_immutable_block(&block_cache, 0, None, token_vec([0, 1, 2, 3, 4, 5, 6]));
    let epoch = query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(query_tokens, SampledTokens::Prefill { epoch });
    assert_eq!(0, blocks.num_in_sync_blocks);

    let sync_blocks = blocks.prepare_blocks();
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_block_counts(&blocks, 1, 0, 0);
    assert_total_tokens(
        &blocks.immutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );
    assert_immutable_annotations(&blocks.immutable_blocks[0], root_annotations([0, 1, 2]));
    assert_eq!(
        random_block_vec
            .iter()
            .map(|block| block.trie_node_key())
            .collect::<Vec<_>>(),
        blocks.immutable_blocks[0]
            .iter()
            .map(|block| block.trie_node_key())
            .collect::<Vec<_>>()
    );
    assert_state(
        &blocks,
        1,
        &token_vec([0, 1, 2, 3]),
        &[],
        &[],
        &token_vec([4, 5, 6]),
        &[],
    );
}

#[test]
fn test_prepare_commit_full_block_mutable_collision_w_mtp() {
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache.clone(), token_vec([0, 1, 2, 3, 4]));
    let InitBlockOnceResult::Success {
        ready_token_slots: ready_slots,
    } = blocks.init_block_once()
    else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_slots);
    blocks.queued_tokens.extend(token_vec([5, 6]));
    blocks.try_mark_ready();

    assert_block_counts(&blocks, 0, 0, 1);
    assert_total_tokens(
        &blocks.mutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );
    assert_mutable_annotations(&blocks.mutable_blocks[0], root_annotations([0, 1, 2]));
    assert_state(
        &blocks,
        0,
        &[],
        &[],
        &token_vec([0, 1, 2, 3]),
        &token_vec([4, 5, 6]),
        &[],
    );

    let query_tokens = blocks.prepare(NUM_TOKEN_PER_BLOCK).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    match &query_tokens {
        QueryTokens::Prefill {
            epoch,
            token_index,
            tokens,
            window,
        } => {
            assert_eq!(0, *epoch);
            assert_eq!(0, *token_index);
            assert_eq!(token_vec([0, 1, 2, 3, 4, 5, 6]), *tokens);
            assert_eq!(NUM_TOKEN_PER_BLOCK, *window);
        },
        _ => unreachable!(),
    }
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_state(
        &blocks,
        1,
        &[],
        &token_vec([0, 1, 2, 3]),
        &[],
        &token_vec([4, 5, 6]),
        &[],
    );

    let random_block_vec = insert_immutable_block(&block_cache, 0, None, token_vec([0, 1, 2, 3, 4, 5, 6]));
    let epoch = query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(query_tokens, SampledTokens::Prefill { epoch });
    assert_eq!(0, blocks.num_in_sync_blocks);

    let sync_blocks = blocks.prepare_blocks();
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_block_counts(&blocks, 1, 0, 0);
    assert_total_tokens(
        &blocks.immutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );
    assert_immutable_annotations(&blocks.immutable_blocks[0], root_annotations([0, 1, 2]));
    assert_eq!(
        random_block_vec
            .iter()
            .map(|block| block.trie_node_key())
            .collect::<Vec<_>>(),
        blocks.immutable_blocks[0]
            .iter()
            .map(|block| block.trie_node_key())
            .collect::<Vec<_>>()
    );
    assert_state(
        &blocks,
        1,
        &token_vec([0, 1, 2, 3]),
        &[],
        &[],
        &token_vec([4, 5, 6]),
        &[],
    );
}

#[test]
fn test_prepare_commit_mutable_collision_additional_validated_token_w_mtp() {
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache.clone(), token_vec([0, 1, 2, 3, 4, 5]));
    blocks.spec_tokens = token_vec([10]);
    blocks.spec_probs = vec![NotNan::new(1.0).unwrap()];
    blocks.spec_confidences = vec![NotNan::new(1.0).unwrap()];
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert!(7 <= ready_token_slots);

    let query_tokens = blocks.prepare(7).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    match &query_tokens {
        QueryTokens::Decode {
            epoch,
            token_index,
            tokens,
            spec_tokens,
        } => {
            assert_eq!(0, *epoch);
            assert_eq!(0, *token_index);
            assert_eq!(token_vec([0, 1, 2, 3, 4, 5]), *tokens);
            assert_eq!(token_vec([10]), *spec_tokens);
        },
        _ => unreachable!(),
    }
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_block_counts(&blocks, 0, 0, 2);
    assert_total_tokens(
        &blocks.mutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, u32::MAX]),
        ],
    );
    assert_mutable_annotations(&blocks.mutable_blocks[0], root_annotations([0, 1, 2]));
    assert_state(
        &blocks,
        2,
        &[],
        &token_vec([0, 1, 2, 3, 4, 5]),
        &[],
        &[],
        &token_vec([10]),
    );

    let random_block_vec = insert_immutable_block(&block_cache, 0, None, token_vec([0, 1, 2, 3, 4, 5, 10]));
    let epoch = query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(
        query_tokens,
        SampledTokens::Decode {
            epoch,
            validated_tokens: token_vec([10]),
            validated_probs: vec![NotNan::new(0.5).unwrap()],
            sampled_token: Token::new(20),
            sampled_prob: NotNan::new(0.5).unwrap(),
            spec_tokens: token_vec([30, 31]),
            spec_probs: vec![NotNan::new(0.6).unwrap(), NotNan::new(0.7).unwrap()],
            spec_confidences: vec![NotNan::new(1.0).unwrap(); 2],
        },
    );
    assert_eq!(0, blocks.num_in_sync_blocks);

    let sync_blocks = blocks.prepare_blocks();
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_block_counts(&blocks, 1, 0, 1);
    assert_total_tokens(
        &blocks.immutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 10]),
        ],
    );
    assert_immutable_annotations(&blocks.immutable_blocks[0], root_annotations([0, 1, 2]));
    assert_total_tokens(
        &blocks.mutable_blocks[0],
        [
            token_vec([4, 5, 10]),
            token_vec([5, 10, 20]),
            token_vec([10, 20, 30]),
            token_vec([20, 30, 31]),
        ],
    );
    assert_mutable_annotations(&blocks.mutable_blocks[0], empty_annotations());
    assert_eq!(
        random_block_vec
            .iter()
            .map(|block| block.trie_node_key())
            .collect::<Vec<_>>(),
        blocks.immutable_blocks[0]
            .iter()
            .map(|block| block.trie_node_key())
            .collect::<Vec<_>>()
    );
    assert_state(
        &blocks,
        2,
        &token_vec([0, 1, 2, 3, 4, 5, 10]),
        &[],
        &[],
        &token_vec([20]),
        &token_vec([30, 31]),
    );
}

#[test]
fn test_enqueue_tokens_w_mtp() {
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache, token_vec([0, 1, 2, 3, 4]));

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_block_counts(&blocks, 0, 0, 1);
    assert_total_tokens(
        &blocks.mutable_blocks[0],
        [
            token_vec([0, 1]),
            token_vec([1, 2]),
            token_vec([2, 3]),
            token_vec([3, 4]),
        ],
    );
    assert_state(&blocks, 0, &[], &[], &token_vec([0, 1]), &token_vec([2, 3, 4]), &[]);

    blocks.queued_tokens.extend(token_vec([5, 6, 7, 8]));
    blocks.try_mark_ready();
    assert_block_counts(&blocks, 0, 0, 1);
    assert_total_tokens(
        &blocks.mutable_blocks[0],
        [
            token_vec([0, 1, 2, 3]),
            token_vec([1, 2, 3, 4]),
            token_vec([2, 3, 4, 5]),
            token_vec([3, 4, 5, 6]),
        ],
    );
    assert_state(
        &blocks,
        0,
        &[],
        &[],
        &token_vec([0, 1, 2, 3]),
        &token_vec([4, 5, 6, 7, 8]),
        &[],
    );
}

#[test]
fn test_parent_node_w_mtp() {
    let root_total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6]);
    let child_total_tokens = token_vec([4, 5, 6, 7, 8, 9, 10]);
    let block_cache = initialize_block_cache([1024; NUM_CACHE_LANE]);
    let mut blocks = initialize_blocks(block_cache.clone(), root_total_tokens.clone());
    let expected_root_annotations: [SmallVec<[BlockAnnotation; 1]>; NUM_CACHE_LANE] =
        root_annotations([0, 1, 2]).map(Into::into);
    assert_eq!(std::array::from_fn(|_| None), blocks.parent_trie_node_key_vec(0));
    assert_eq!(expected_root_annotations, blocks.block_annotation_vec(0));

    let root_immutable_block_vec = insert_immutable_block(&block_cache, 0, None, root_total_tokens);
    blocks.queued_tokens = child_total_tokens.into();
    blocks.immutable_blocks.push(root_immutable_block_vec);
    let immutable_block_vec = &blocks.immutable_blocks[0];
    assert_eq!(
        std::array::from_fn(|lane| Some(immutable_block_vec[lane].trie_node_key())),
        blocks.parent_trie_node_key_vec(1)
    );
    assert_eq!(
        std::array::from_fn(|_| SmallVec::<[BlockAnnotation; 1]>::new()),
        blocks.block_annotation_vec(1)
    );
}

fn initialize_blocks(block_cache: Arc<TestMultiLaneTrieBlockCache>, total_tokens: Vec<Token>) -> TestTrieKVBlocks {
    assert!(NUM_CACHE_LANE - 1 <= total_tokens.len());
    TestTrieKVBlocks::new(
        block_cache,
        Vec::new(),
        Vec::new(),
        std::iter::empty::<Token>(),
        std::iter::empty::<Token>(),
        total_tokens,
    )
}

fn assert_sync_blocks(sync_blocks: &DecoderSyncBlocks, blocks: &TestTrieKVBlocks, block_index: usize) {
    let immutable_start = block_index.min(blocks.immutable_blocks.len());
    let semi_immutable_start = block_index
        .saturating_sub(blocks.immutable_blocks.len())
        .min(blocks.semi_immutable_blocks.len());
    let mutable_start = block_index
        .saturating_sub(blocks.immutable_blocks.len() + blocks.semi_immutable_blocks.len())
        .min(blocks.mutable_blocks.len());
    let mut expected_kv_page_ids: [Vec<Vec<u32>>; NUM_CACHE_LANE] = std::array::from_fn(|_| Vec::new());
    let mut expected_state_page_ids: [Vec<Vec<u32>>; NUM_CACHE_LANE] = std::array::from_fn(|_| Vec::new());
    for block_vec in &blocks.immutable_blocks[immutable_start..] {
        for (lane, block) in block_vec.iter().enumerate() {
            let block_ref = block.trie_node_ref();
            expected_kv_page_ids[lane].push(block_ref.kv_placement().page_ids().to_vec());
            expected_state_page_ids[lane].push(block_ref.state_placement().page_ids().to_vec());
        }
    }
    for block_vec in blocks.semi_immutable_blocks.iter().skip(semi_immutable_start) {
        for (lane, block) in block_vec.iter().enumerate() {
            expected_kv_page_ids[lane].push(block.kv_placement().page_ids().to_vec());
            expected_state_page_ids[lane].push(block.state_placement().page_ids().to_vec());
        }
    }
    for block_vec in blocks.mutable_blocks.iter().skip(mutable_start) {
        for (lane, block) in block_vec.iter().enumerate() {
            expected_kv_page_ids[lane].push(block.kv_placement().page_ids().to_vec());
            expected_state_page_ids[lane].push(block.state_placement().page_ids().to_vec());
        }
    }
    assert_eq!(block_index, sync_blocks.block_index());
    assert_eq!(&expected_kv_page_ids, sync_blocks.kv_page_ids());
    assert_eq!(&expected_state_page_ids, sync_blocks.state_page_ids());
}

fn initialize_block_cache(max_pages_per_lane: [usize; NUM_CACHE_LANE]) -> Arc<TestMultiLaneTrieBlockCache> {
    let page_id_allocators: [Arc<U32IDAllocator>; NUM_CACHE_LANE] =
        std::array::from_fn(|lane| Arc::new(U32IDAllocator::new(max_pages_per_lane[lane] as u64)));
    let block_cache_vec: [Arc<SingleLaneTrieBlockCache<NUM_TRIE_PARTITION, TPKVBlockAllocator, TPStateBlockAllocator>>;
        NUM_CACHE_LANE] = std::array::from_fn(|lane| {
        let kv_block_allocator = TPKVBlockAllocator::new(NUM_KV_PAGES_PER_BLOCK, page_id_allocators[lane].clone());
        let state_block_allocator =
            TPStateBlockAllocator::new(NUM_STATE_PAGES_PER_BLOCK, page_id_allocators[lane].clone());
        Arc::new(SingleLaneTrieBlockCache::new(
            kv_block_allocator,
            state_block_allocator,
            BLOCK_CACHE_CAPACITY,
            Shutdown::new(),
        ))
    });
    Arc::new(TestMultiLaneTrieBlockCache::new(block_cache_vec))
}

fn insert_immutable_block(
    block_cache: &TestMultiLaneTrieBlockCache,
    block_index: usize,
    parent_immutable_block_vec: Option<&[TestImmutableBlock; NUM_CACHE_LANE]>,
    total_tokens: Vec<Token>,
) -> [TestImmutableBlock; NUM_CACHE_LANE] {
    let block_metadata_vec = block_metadata_vec(block_index, parent_immutable_block_vec, total_tokens);
    let parent_trie_node_key_vec: [Option<_>; NUM_CACHE_LANE] =
        std::array::from_fn(|lane| block_metadata_vec[lane].parent_trie_node_key());
    let AllocateMultiLaneMutableBlockResult::Mutable { mut block_vec } =
        block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>()
    else {
        unreachable!()
    };
    for (block, block_metadata) in block_vec.iter_mut().zip(block_metadata_vec.iter()) {
        block.insert_annotations(block_metadata.annotations().iter().cloned());
        block.write_tokens(0, block_metadata.tokens());
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(scheduled_tokens.len());
    }

    let CommitMultiLaneMutableBlockResult::Immutable { block_vec } =
        block_cache.commit_mutable_block(parent_trie_node_key_vec, block_vec)
    else {
        unreachable!()
    };
    block_vec
}

fn insert_immutable_block_with_annotations(
    block_cache: &TestMultiLaneTrieBlockCache,
    total_tokens: Vec<Token>,
    annotations: [SmallVec<[BlockAnnotation; 1]>; NUM_CACHE_LANE],
) -> [TestImmutableBlock; NUM_CACHE_LANE] {
    assert_eq!(NUM_TOKEN_PER_BLOCK + NUM_CACHE_LANE - 1, total_tokens.len());
    let block_metadata_vec: [BlockMetadata<NUM_TOKEN_PER_BLOCK>; NUM_CACHE_LANE] = annotations
        .into_iter()
        .zip(total_tokens.windows(NUM_TOKEN_PER_BLOCK))
        .map(|(annotations, tokens)| BlockMetadata::new(None, annotations, Arc::<[Token]>::from(tokens.to_vec())))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let AllocateMultiLaneMutableBlockResult::Mutable { mut block_vec } =
        block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>()
    else {
        unreachable!()
    };
    for (block, block_metadata) in block_vec.iter_mut().zip(block_metadata_vec.iter()) {
        block.insert_annotations(block_metadata.annotations().iter().cloned());
        block.write_tokens(0, block_metadata.tokens());
        let scheduled_tokens = block.schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
        block.cache_tokens(scheduled_tokens.len());
    }
    let CommitMultiLaneMutableBlockResult::Immutable { block_vec } =
        block_cache.commit_mutable_block(std::array::from_fn(|_| None), block_vec)
    else {
        unreachable!()
    };
    block_vec
}

fn reserve_semi_immutable_block(
    block_cache: &TestMultiLaneTrieBlockCache,
    block_index: usize,
    parent_immutable_block_vec: Option<&[TestImmutableBlock; NUM_CACHE_LANE]>,
    total_tokens: Vec<Token>,
) -> [SemiImmutableBlock<NUM_TOKEN_PER_BLOCK>; NUM_CACHE_LANE] {
    let ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable { block_vec } = block_cache
        .reserve_semi_immutable_block(block_metadata_vec(
            block_index,
            parent_immutable_block_vec,
            total_tokens,
        ))
    else {
        unreachable!()
    };
    block_vec
}

fn allocate_mutable_block(
    block_cache: &TestMultiLaneTrieBlockCache,
) -> [MutableBlock<NUM_TOKEN_PER_BLOCK>; NUM_CACHE_LANE] {
    let AllocateMultiLaneMutableBlockResult::Mutable { block_vec } =
        block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>()
    else {
        unreachable!()
    };
    block_vec
}

fn block_metadata_vec(
    block_index: usize,
    parent_immutable_block_vec: Option<&[TestImmutableBlock; NUM_CACHE_LANE]>,
    total_tokens: Vec<Token>,
) -> [BlockMetadata<NUM_TOKEN_PER_BLOCK>; NUM_CACHE_LANE] {
    assert_eq!(NUM_TOKEN_PER_BLOCK + NUM_CACHE_LANE - 1, total_tokens.len());
    let parent_trie_node_key_vec: [Option<_>; NUM_CACHE_LANE] = if block_index == 0 {
        std::array::from_fn(|_| None)
    } else {
        let parent_immutable_block_vec = parent_immutable_block_vec
            .expect("block_metadata_vec: non-root kv block must have parent immutable kv block vec");
        std::array::from_fn(|lane| Some(parent_immutable_block_vec[lane].trie_node_key()))
    };
    total_tokens
        .windows(NUM_TOKEN_PER_BLOCK)
        .enumerate()
        .map(|(lane, tokens)| {
            let annotations = if lane == 0 {
                vec![].into()
            } else {
                if block_index == 0 {
                    let prefix_tokens: Arc<[Token]> = total_tokens[..lane].to_vec().into();
                    vec![BlockAnnotation::prefix_tokens(prefix_tokens)].into()
                } else {
                    vec![].into()
                }
            };
            BlockMetadata::new(
                parent_trie_node_key_vec[lane],
                annotations,
                Arc::<[Token]>::from(tokens.to_vec()),
            )
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn empty_annotations() -> [Vec<BlockAnnotation>; NUM_CACHE_LANE] {
    std::array::from_fn(|_| vec![])
}

fn root_annotations(values: impl AsRef<[u32]>) -> [Vec<BlockAnnotation>; NUM_CACHE_LANE] {
    let prefix_tokens = token_vec(values);
    assert_eq!(NUM_CACHE_LANE - 1, prefix_tokens.len());
    std::array::from_fn(|lane| {
        if lane == 0 {
            vec![]
        } else {
            vec![BlockAnnotation::prefix_tokens(prefix_tokens[..lane].to_vec().into())]
        }
    })
}

fn assert_mutable_annotations(
    block_vec: &[MutableBlock<NUM_TOKEN_PER_BLOCK>; NUM_CACHE_LANE],
    expected_annotations: [Vec<BlockAnnotation>; NUM_CACHE_LANE],
) {
    assert_eq!(
        expected_annotations,
        std::array::from_fn(|lane| block_vec[lane].annotations())
    );
}

fn assert_semi_immutable_annotations(
    block_vec: &[SemiImmutableBlock<NUM_TOKEN_PER_BLOCK>; NUM_CACHE_LANE],
    expected_annotations: [Vec<BlockAnnotation>; NUM_CACHE_LANE],
) {
    assert_eq!(
        expected_annotations,
        std::array::from_fn(|lane| block_vec[lane].annotations().to_vec())
    );
}

fn assert_immutable_annotations<const NUM_BLOCK: usize>(
    block_vec: &[TestImmutableBlock; NUM_BLOCK],
    expected_annotations: [Vec<BlockAnnotation>; NUM_BLOCK],
) {
    assert_eq!(
        expected_annotations,
        std::array::from_fn(|index| block_vec[index].annotations())
    );
}

fn assert_total_tokens<KVBlockType: DecoderBlock, const NUM_BLOCK: usize>(
    block_vec: &[KVBlockType; NUM_BLOCK],
    expected_total_tokens: [Vec<Token>; NUM_BLOCK],
) {
    assert_eq!(
        expected_total_tokens,
        std::array::from_fn(|index| block_vec[index].total_tokens().to_vec())
    );
}

fn token_vec(values: impl AsRef<[u32]>) -> Vec<Token> {
    values.as_ref().iter().copied().map(Token::new).collect()
}

fn symbolic_resource(resource_id: ResourceID) -> Resource {
    Resource::Symbolic(SymbolicResource::new(
        resource_id,
        ResourceURI::new("test://resource".to_string()),
    ))
}

fn assert_state(
    blocks: &TestTrieKVBlocks,
    block_index: usize,
    cached_tokens: &[Token],
    scheduled_tokens: &[Token],
    ready_tokens: &[Token],
    queued_tokens: &[Token],
    spec_tokens: &[Token],
) {
    assert_eq!(block_index, blocks.num_in_sync_blocks);
    assert_eq!(cached_tokens, blocks.cached_tokens().collect::<Vec<_>>());
    assert_eq!(scheduled_tokens, blocks.scheduled_tokens().collect::<Vec<_>>());
    assert_eq!(ready_tokens, blocks.ready_tokens().collect::<Vec<_>>());
    assert_eq!(queued_tokens, blocks.queued_tokens().collect::<Vec<_>>());
    assert_eq!(spec_tokens, blocks.spec_tokens().collect::<Vec<_>>());
}

fn assert_block_counts(blocks: &TestTrieKVBlocks, num_immutable: usize, num_semi_immutable: usize, num_mutable: usize) {
    assert_eq!(num_immutable, blocks.immutable_blocks.len());
    assert_eq!(num_semi_immutable, blocks.semi_immutable_blocks.len());
    assert_eq!(num_mutable, blocks.mutable_blocks.len());
}
