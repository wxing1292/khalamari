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
const NUM_CACHE_LANE: usize = 1;

type TestSingleLaneTrieKVBlockCache =
    SingleLaneTrieBlockCache<NUM_TRIE_PARTITION, TPKVBlockAllocator, TPStateBlockAllocator>;
type TestMultiLaneTrieKVBlockCache =
    MultiLaneTrieBlockCache<NUM_TRIE_PARTITION, NUM_CACHE_LANE, TPKVBlockAllocator, TPStateBlockAllocator>;
type TestTrieBlocks =
    TrieDecoderBlocks<NUM_TOKEN_PER_BLOCK, NUM_TRIE_PARTITION, NUM_CACHE_LANE, TestMultiLaneTrieKVBlockCache>;

type TestImmutableBlock = ImmutableBlock<
    NUM_TRIE_PARTITION,
    <TestMultiLaneTrieKVBlockCache as MultiLaneBlockCache<NUM_TRIE_PARTITION, NUM_CACHE_LANE>>::Entry,
>;

#[test]
fn test_init_block_once_half_block_success_single_lane() {
    let total_tokens = token_vec([1, 2]);
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);
    assert_block_counts(&blocks, 0, 0, 1);
    assert_total_tokens(&blocks.mutable_blocks[0], [token_vec([1, 2])]);
    assert_mutable_annotations(&blocks.mutable_blocks[0], empty_annotations());
    assert_state(&blocks, 0, &[], &[], &token_vec([1, 2]), &[], &[]);
}

#[test]
fn test_init_block_once_half_block_resource_limit_exceeded_single_lane() {
    let total_tokens = token_vec([1, 2]);
    let block_cache = initialize_block_cache(2);
    let random_block = allocate_mutable_block(&block_cache);
    let mut blocks = initialize_blocks(block_cache.clone(), total_tokens.clone());

    let InitBlockOnceResult::ResourceLimitExceeded = blocks.init_block_once() else {
        unreachable!()
    };

    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([1, 2]), &[]);
    block_cache.free_mutable_block(random_block);
}

#[test]
fn test_init_block_once_full_block_cache_bypass_single_lane() {
    let total_tokens = token_vec([1, 2, 3, 4]);
    let block_cache = initialize_block_cache(1024);
    insert_immutable_block(&block_cache, 0, None, total_tokens.clone());
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);
    assert_block_counts(&blocks, 0, 0, 1);
    assert_total_tokens(&blocks.mutable_blocks[0], [token_vec([1, 2, 3, 4])]);
    assert_mutable_annotations(&blocks.mutable_blocks[0], empty_annotations());
    assert_state(&blocks, 0, &[], &[], &token_vec([1, 2, 3, 4]), &[], &[]);
}

#[test]
fn test_init_block_once_full_and_half_block_cache_hit_single_lane() {
    let total_tokens = token_vec([1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache(1024);
    insert_immutable_block(&block_cache, 0, None, token_vec([1, 2, 3, 4]));
    let mut blocks = initialize_blocks(block_cache, total_tokens);

    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_eq!(0, ready_token_slots);
    assert_block_counts(&blocks, 1, 0, 0);
    assert_total_tokens(&blocks.immutable_blocks[0], [token_vec([1, 2, 3, 4])]);
    assert_immutable_annotations(&blocks.immutable_blocks[0], empty_annotations());
    assert_state(&blocks, 0, &token_vec([1, 2, 3, 4]), &[], &[], &token_vec([5, 6]), &[]);
}

#[test]
fn test_init_block_once_full_and_half_block_cache_miss_no_semi_immutable_success_single_lane() {
    let total_tokens = token_vec([1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);
    assert_block_counts(&blocks, 0, 1, 0);
    assert_total_tokens(&blocks.semi_immutable_blocks[0], [token_vec([1, 2, 3, 4])]);
    assert_semi_immutable_annotations(&blocks.semi_immutable_blocks[0], empty_annotations());
    assert_state(&blocks, 0, &[], &[], &token_vec([1, 2, 3, 4]), &token_vec([5, 6]), &[]);
}

#[test]
fn test_init_block_once_full_and_half_block_cache_miss_resource_not_found_single_lane() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5]);
    let block_cache = initialize_block_cache(1024);
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    let placement = ResourcePlacement::new(resource_id, vec![(1, 0, 2)], total_tokens.len());
    let mut blocks = TestTrieBlocks::new(
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
fn test_init_block_once_full_and_half_block_cache_miss_resource_found_single_lane() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5]);
    let block_cache = initialize_block_cache(1024);
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    let placement = ResourcePlacement::new(resource_id, vec![(1, 0, 2)], total_tokens.len());
    let (resource, _resource_allocator) = concrete_resource(resource_id, 2);
    let mut blocks = TestTrieBlocks::new(
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
    assert_total_tokens(&blocks.semi_immutable_blocks[0], [token_vec([0, 1, 2, 3])]);
    assert_semi_immutable_annotations(&blocks.semi_immutable_blocks[0], expected_annotations);
    assert_state(&blocks, 0, &[], &[], &token_vec([0, 1, 2, 3]), &token_vec([4, 5]), &[]);
}

#[test]
fn test_init_block_once_full_and_half_block_cache_hit_bypasses_resource_single_lane() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5]);
    let block_cache = initialize_block_cache(1024);
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    let placement = ResourcePlacement::new(resource_id, vec![(1, 0, 2)], total_tokens.len());
    let mut blocks = TestTrieBlocks::new(
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
fn test_init_block_once_full_and_half_block_cache_miss_no_semi_immutable_resource_limit_exceeded_single_lane() {
    let total_tokens = token_vec([1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache(2);
    let random_block = allocate_mutable_block(&block_cache);
    let mut blocks = initialize_blocks(block_cache.clone(), total_tokens.clone());

    let InitBlockOnceResult::ResourceLimitExceeded = blocks.init_block_once() else {
        unreachable!()
    };

    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([1, 2, 3, 4, 5, 6]), &[]);
    block_cache.free_mutable_block(random_block);
}

#[test]
fn test_init_block_once_full_and_half_block_cache_reserved_await_single_lane() {
    let total_tokens = token_vec([1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache(1024);
    let mut random_block = reserve_semi_immutable_block(&block_cache, 0, None, token_vec([1, 2, 3, 4]));
    let mut blocks = initialize_blocks(block_cache.clone(), total_tokens.clone());

    let InitBlockOnceResult::Await { wait } = blocks.init_block_once() else {
        unreachable!()
    };
    pin_mut!(wait);
    let mut cx = Context::from_waker(noop_waker_ref());

    assert!(matches!(wait.as_mut().poll(&mut cx), Poll::Pending));
    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([1, 2, 3, 4, 5, 6]), &[]);

    let scheduled_tokens = random_block[0].schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
    random_block[0].cache_tokens(scheduled_tokens.len());
    let CommitMultiLaneSemiImmutableBlockResult::Immutable {
        block_vec: random_block,
    } = block_cache.commit_semi_immutable_block(random_block)
    else {
        unreachable!()
    };

    assert!(matches!(wait.as_mut().poll(&mut cx), Poll::Ready(())));
    assert_total_tokens(&random_block, [token_vec([1, 2, 3, 4])]);

    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_eq!(0, ready_token_slots);
    assert_block_counts(&blocks, 1, 0, 0);
    assert_total_tokens(&blocks.immutable_blocks[0], [token_vec([1, 2, 3, 4])]);
    assert_immutable_annotations(&blocks.immutable_blocks[0], empty_annotations());
    assert_state(&blocks, 0, &token_vec([1, 2, 3, 4]), &[], &[], &token_vec([5, 6]), &[]);
}

#[test]
fn test_init_block_once_full_and_half_block_cache_bypass_with_semi_immutable_success_single_lane() {
    let total_tokens = token_vec([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_eq!(NUM_TOKEN_PER_BLOCK * 2, ready_token_slots);
    assert_block_counts(&blocks, 0, 1, 1);
    assert_total_tokens(&blocks.semi_immutable_blocks[0], [token_vec([1, 2, 3, 4])]);
    assert_semi_immutable_annotations(&blocks.semi_immutable_blocks[0], empty_annotations());
    assert_total_tokens(&blocks.mutable_blocks[0], [token_vec([5, 6, 7, 8])]);
    assert_mutable_annotations(&blocks.mutable_blocks[0], empty_annotations());
    assert_state(
        &blocks,
        0,
        &[],
        &[],
        &token_vec([1, 2, 3, 4, 5, 6, 7, 8]),
        &token_vec([9, 10]),
        &[],
    );
}

#[test]
fn test_init_block_once_full_and_half_block_cache_bypass_with_semi_immutable_resource_limit_exceeded_single_lane() {
    let total_tokens = token_vec([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let block_cache = initialize_block_cache(2);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };
    let InitBlockOnceResult::ResourceLimitExceeded = blocks.init_block_once() else {
        unreachable!()
    };

    assert_block_counts(&blocks, 0, 1, 0);
    assert_total_tokens(&blocks.semi_immutable_blocks[0], [token_vec([1, 2, 3, 4])]);
    assert_semi_immutable_annotations(&blocks.semi_immutable_blocks[0], empty_annotations());
    assert_state(
        &blocks,
        0,
        &[],
        &[],
        &token_vec([1, 2, 3, 4]),
        &token_vec([5, 6, 7, 8, 9, 10]),
        &[],
    );
}

#[test]
fn test_init_block_once_full_and_half_block_cache_bypass_with_mutable_success_single_lane() {
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache, token_vec([1, 2]));

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };
    blocks.queued_tokens.extend(token_vec([3, 4, 5, 6, 7, 8, 9, 10]));
    blocks.try_mark_ready();

    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_eq!(NUM_TOKEN_PER_BLOCK * 2, ready_token_slots);
    assert_block_counts(&blocks, 0, 0, 2);
    assert_total_tokens(&blocks.mutable_blocks[0], [token_vec([1, 2, 3, 4])]);
    assert_mutable_annotations(&blocks.mutable_blocks[0], empty_annotations());
    assert_total_tokens(&blocks.mutable_blocks[1], [token_vec([5, 6, 7, 8])]);
    assert_mutable_annotations(&blocks.mutable_blocks[1], empty_annotations());
    assert_state(
        &blocks,
        0,
        &[],
        &[],
        &token_vec([1, 2, 3, 4, 5, 6, 7, 8]),
        &token_vec([9, 10]),
        &[],
    );
}

#[test]
fn test_init_block_once_full_and_half_block_cache_bypass_with_mutable_resource_limit_exceeded_single_lane() {
    let block_cache = initialize_block_cache(2);
    let mut blocks = initialize_blocks(block_cache, token_vec([1, 2]));

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };
    blocks.queued_tokens.extend(token_vec([3, 4, 5, 6, 7, 8, 9, 10]));
    blocks.try_mark_ready();

    let InitBlockOnceResult::ResourceLimitExceeded = blocks.init_block_once() else {
        unreachable!()
    };

    assert_block_counts(&blocks, 0, 0, 1);
    assert_total_tokens(&blocks.mutable_blocks[0], [token_vec([1, 2, 3, 4])]);
    assert_mutable_annotations(&blocks.mutable_blocks[0], empty_annotations());
    assert_state(
        &blocks,
        0,
        &[],
        &[],
        &token_vec([1, 2, 3, 4]),
        &token_vec([5, 6, 7, 8, 9, 10]),
        &[],
    );
}

#[test]
fn test_uninit_block_once_noop_block_single_lane() {
    let total_tokens = token_vec([1, 2]);
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([1, 2]), &[]);

    let UninitBlockOnceResult::Success { cached_token_slots } = blocks.uninit_block_once();

    assert_eq!(0, cached_token_slots);
    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([1, 2]), &[]);
}

#[test]
fn test_uninit_block_once_mutable_block_single_lane() {
    let total_tokens = token_vec([1, 2]);
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_block_counts(&blocks, 0, 0, 1);
    assert_total_tokens(&blocks.mutable_blocks[0], [token_vec([1, 2])]);
    assert_state(&blocks, 0, &[], &[], &token_vec([1, 2]), &[], &[]);

    let UninitBlockOnceResult::Success { cached_token_slots } = blocks.uninit_block_once();

    assert_eq!(0, cached_token_slots);
    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([1, 2]), &[]);
}

#[test]
fn test_uninit_block_once_semi_immutable_block_single_lane() {
    let total_tokens = token_vec([1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };

    assert_block_counts(&blocks, 0, 1, 0);
    assert_total_tokens(&blocks.semi_immutable_blocks[0], [token_vec([1, 2, 3, 4])]);
    assert_state(&blocks, 0, &[], &[], &token_vec([1, 2, 3, 4]), &token_vec([5, 6]), &[]);

    let UninitBlockOnceResult::Success { cached_token_slots } = blocks.uninit_block_once();

    assert_eq!(0, cached_token_slots);
    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([1, 2, 3, 4, 5, 6]), &[]);
}

#[test]
fn test_uninit_block_once_immutable_block_single_lane() {
    let total_tokens = token_vec([1, 2, 3, 4, 5, 6]);
    let block_cache = initialize_block_cache(1024);
    insert_immutable_block(&block_cache, 0, None, token_vec([1, 2, 3, 4]));
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());

    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(0, ready_token_slots);

    assert_block_counts(&blocks, 1, 0, 0);
    assert_total_tokens(&blocks.immutable_blocks[0], [token_vec([1, 2, 3, 4])]);
    assert_state(&blocks, 0, &token_vec([1, 2, 3, 4]), &[], &[], &token_vec([5, 6]), &[]);

    let UninitBlockOnceResult::Success { cached_token_slots } = blocks.uninit_block_once();

    assert_eq!(0, cached_token_slots);
    assert_block_counts(&blocks, 0, 0, 0);
    assert_state(&blocks, 0, &[], &[], &[], &token_vec([1, 2, 3, 4, 5, 6]), &[]);
}

#[test]
fn test_prepare_cancel_commit_prefill_zero_token_index_single_lane() {
    let total_tokens = token_vec([1, 2, 3]);
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);

    let query_tokens = blocks.prepare(2).unwrap();
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
            assert_eq!(token_vec([1, 2]), *tokens);
            assert_eq!(2, *window);
        },
        _ => unreachable!(),
    }
    let expected_query_tokens = query_tokens.clone();
    let expected_sync_blocks = sync_blocks.clone();
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_state(&blocks, 1, &[], &token_vec([1, 2]), &token_vec([3]), &[], &[]);

    blocks.cancel_blocks(sync_blocks);
    blocks.cancel(query_tokens);
    assert_state(&blocks, 0, &[], &[], &token_vec([1, 2, 3]), &[], &[]);

    let query_tokens = blocks.prepare(2).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    assert_eq!(expected_query_tokens, query_tokens);
    assert_eq!(expected_sync_blocks, sync_blocks);
    assert_state(&blocks, 1, &[], &token_vec([1, 2]), &token_vec([3]), &[], &[]);

    let epoch = query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(query_tokens, SampledTokens::Prefill { epoch });
    assert_state(&blocks, 1, &token_vec([1, 2]), &[], &token_vec([3]), &[], &[]);
}

#[test]
fn test_prepare_resource_not_found_single_lane() {
    const REQUEST_ID: usize = 1;

    let total_tokens = token_vec([0, 1, 2, 3, 4, 5]);
    let block_cache = initialize_block_cache(1024);
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    let placement = ResourcePlacement::new(resource_id, vec![(1, 0, 2)], total_tokens.len());
    let blocks = TestTrieBlocks::new(
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
fn test_prepare_commit_resource_unload_single_lane() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    let block_cache = initialize_block_cache(1024);
    let resource_id = ResourceID::new(ResourceTypeID::new(7));
    let placement = ResourcePlacement::new(resource_id, vec![(1, 0, 2), (5, 2, 1)], total_tokens.len());
    let (resource, resource_allocator) = concrete_resource(resource_id, 3);
    let mut blocks = TestTrieBlocks::new(
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
fn test_prepare_cancel_commit_prefill_nonzero_token_index_single_lane() {
    let total_tokens = token_vec([1, 2, 3, 4, 5, 6, 7]);
    let block_cache = initialize_block_cache(1024);
    insert_immutable_block(&block_cache, 0, None, token_vec([1, 2, 3, 4]));
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(0, ready_token_slots);
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);

    let query_tokens = blocks.prepare(2).unwrap();
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
            assert_eq!(token_vec([5, 6]), *tokens);
            assert_eq!(2, *window);
        },
        _ => unreachable!(),
    }
    let expected_query_tokens = query_tokens.clone();
    let expected_sync_blocks = sync_blocks.clone();
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_state(
        &blocks,
        2,
        &token_vec([1, 2, 3, 4]),
        &token_vec([5, 6]),
        &token_vec([7]),
        &[],
        &[],
    );

    blocks.cancel_blocks(sync_blocks);
    blocks.cancel(query_tokens);
    assert_state(
        &blocks,
        0,
        &token_vec([1, 2, 3, 4]),
        &[],
        &token_vec([5, 6, 7]),
        &[],
        &[],
    );

    let query_tokens = blocks.prepare(2).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    assert_eq!(expected_query_tokens, query_tokens);
    assert_eq!(expected_sync_blocks, sync_blocks);
    assert_state(
        &blocks,
        2,
        &token_vec([1, 2, 3, 4]),
        &token_vec([5, 6]),
        &token_vec([7]),
        &[],
        &[],
    );

    let epoch = query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(query_tokens, SampledTokens::Prefill { epoch });
    assert_state(
        &blocks,
        2,
        &token_vec([1, 2, 3, 4, 5, 6]),
        &[],
        &token_vec([7]),
        &[],
        &[],
    );
}

#[test]
fn test_prepare_cancel_commit_overlapping_prefill_decode_single_lane() {
    let total_tokens = token_vec([1, 2, 3]);
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);

    let prefill_query_tokens = blocks.prepare(2).unwrap();
    let prefill_sync_blocks = blocks.prepare_blocks();
    let decode_query_tokens = blocks.prepare(1).unwrap();
    let decode_sync_blocks = blocks.prepare_blocks();
    assert_eq!(
        QueryTokens::Prefill {
            epoch: 0,
            token_index: 0,
            tokens: token_vec([1, 2]),
            window: 2,
        },
        prefill_query_tokens
    );
    assert_eq!(
        QueryTokens::Decode {
            epoch: 0,
            token_index: 2,
            tokens: token_vec([3]),
            spec_tokens: token_vec([]),
        },
        decode_query_tokens
    );
    let expected_prefill_query_tokens = prefill_query_tokens.clone();
    let expected_decode_query_tokens = decode_query_tokens.clone();
    assert_state(&blocks, 1, &[], &token_vec([1, 2, 3]), &[], &[], &[]);

    blocks.cancel_blocks(decode_sync_blocks);
    blocks.cancel(decode_query_tokens);
    blocks.cancel_blocks(prefill_sync_blocks);
    blocks.cancel(prefill_query_tokens);
    assert_state(&blocks, 0, &[], &[], &total_tokens, &[], &[]);

    let prefill_query_tokens = blocks.prepare(2).unwrap();
    let prefill_sync_blocks = blocks.prepare_blocks();
    let decode_query_tokens = blocks.prepare(1).unwrap();
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
            sampled_token: Token::new(4),
            sampled_prob: NotNan::new(0.5).unwrap(),
            spec_tokens: token_vec([]),
            spec_probs: vec![],
            spec_confidences: vec![],
        },
    );
    assert_state(&blocks, 1, &total_tokens, &[], &token_vec([4]), &[], &[]);
}

#[test]
fn test_prepare_cancel_commit_decode_zero_token_index_single_lane_wo_spec_token() {
    let total_tokens = token_vec([1, 2, 3]);
    let output_sampled_token = Token::new(4);
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);

    let query_tokens = blocks.prepare(3).unwrap();
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
            assert_eq!(token_vec([1, 2, 3]), *tokens);
            assert!(spec_tokens.is_empty());
        },
        _ => unreachable!(),
    }
    let expected_query_tokens = query_tokens.clone();
    let expected_sync_blocks = sync_blocks.clone();
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_state(&blocks, 1, &[], &token_vec([1, 2, 3]), &[], &[], &[]);

    blocks.cancel_blocks(sync_blocks);
    blocks.cancel(query_tokens);
    assert_state(&blocks, 0, &[], &[], &token_vec([1, 2, 3]), &[], &[]);

    let query_tokens = blocks.prepare(3).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    assert_eq!(expected_query_tokens, query_tokens);
    assert_eq!(expected_sync_blocks, sync_blocks);
    assert_state(&blocks, 1, &[], &token_vec([1, 2, 3]), &[], &[], &[]);

    let epoch = query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(
        query_tokens,
        SampledTokens::Decode {
            epoch,
            validated_tokens: vec![],
            validated_probs: vec![],
            sampled_token: output_sampled_token,
            sampled_prob: NotNan::new(0.5).unwrap(),
            spec_tokens: vec![],
            spec_probs: vec![],
            spec_confidences: vec![],
        },
    );
    assert_state(&blocks, 1, &token_vec([1, 2, 3]), &[], &token_vec([4]), &[], &[]);
}

#[test]
fn test_prepare_cancel_commit_decode_nonzero_token_index_single_lane_wo_spec_token() {
    let total_tokens = token_vec([1, 2, 3, 4, 5, 6, 7]);
    let output_sampled_token = Token::new(8);
    let block_cache = initialize_block_cache(1024);
    insert_immutable_block(&block_cache, 0, None, token_vec([1, 2, 3, 4]));
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(0, ready_token_slots);
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);

    let query_tokens = blocks.prepare(3).unwrap();
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
            assert_eq!(token_vec([5, 6, 7]), *tokens);
            assert!(spec_tokens.is_empty());
        },
        _ => unreachable!(),
    }
    let expected_query_tokens = query_tokens.clone();
    let expected_sync_blocks = sync_blocks.clone();
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_state(
        &blocks,
        2,
        &token_vec([1, 2, 3, 4]),
        &token_vec([5, 6, 7]),
        &[],
        &[],
        &[],
    );

    blocks.cancel_blocks(sync_blocks);
    blocks.cancel(query_tokens);
    assert_state(
        &blocks,
        0,
        &token_vec([1, 2, 3, 4]),
        &[],
        &token_vec([5, 6, 7]),
        &[],
        &[],
    );

    let query_tokens = blocks.prepare(3).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    assert_eq!(expected_query_tokens, query_tokens);
    assert_eq!(expected_sync_blocks, sync_blocks);
    assert_state(
        &blocks,
        2,
        &token_vec([1, 2, 3, 4]),
        &token_vec([5, 6, 7]),
        &[],
        &[],
        &[],
    );

    let epoch = query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(
        query_tokens,
        SampledTokens::Decode {
            epoch,
            validated_tokens: vec![],
            validated_probs: vec![],
            sampled_token: output_sampled_token,
            sampled_prob: NotNan::new(0.5).unwrap(),
            spec_tokens: vec![],
            spec_probs: vec![],
            spec_confidences: vec![],
        },
    );
    assert_state(
        &blocks,
        2,
        &token_vec([1, 2, 3, 4, 5, 6, 7]),
        &[],
        &token_vec([8]),
        &[],
        &[],
    );
}

#[test]
fn test_prepare_cancel_commit_decode_zero_token_index_single_lane_w_spec_token() {
    let total_tokens = token_vec([0]);
    let spec_tokens = token_vec([10, 11, 12]);
    let output_sampled_token = Token::new(20);
    let output_spec_tokens = token_vec([30, 31]);
    let output_spec_probs = vec![NotNan::new(0.6).unwrap(); output_spec_tokens.len()];
    let output_spec_confidences = vec![NotNan::new(0.75).unwrap(); output_spec_tokens.len()];
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);
    let num_spec_tokens = spec_tokens.len();
    blocks.spec_tokens = spec_tokens;
    blocks.spec_probs = vec![NotNan::new(1.0).unwrap(); num_spec_tokens];
    blocks.spec_confidences = vec![NotNan::new(1.0).unwrap(); num_spec_tokens];

    let query_tokens = blocks.prepare(3).unwrap();
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
    assert_state(&blocks, 1, &[], &token_vec([0]), &[], &[], &token_vec([10, 11]));

    blocks.cancel_blocks(sync_blocks);
    blocks.cancel(query_tokens);
    assert_state(&blocks, 0, &[], &[], &token_vec([0]), &[], &token_vec([10, 11]));

    let query_tokens = blocks.prepare(3).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    assert_eq!(expected_query_tokens, query_tokens);
    assert_eq!(expected_sync_blocks, sync_blocks);
    assert_state(&blocks, 1, &[], &token_vec([0]), &[], &[], &token_vec([10, 11]));

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
            spec_confidences: output_spec_confidences.clone(),
            spec_tokens: output_spec_tokens,
            spec_probs: output_spec_probs,
        },
    );
    assert_state(
        &blocks,
        1,
        &token_vec([0, 10]),
        &[],
        &token_vec([20]),
        &[],
        &token_vec([30, 31]),
    );
    assert_eq!(blocks.spec_confidences(), output_spec_confidences);
}

#[test]
fn test_prepare_cancel_commit_decode_nonzero_token_index_single_lane_w_spec_token() {
    let total_tokens = token_vec([0, 1, 2, 3, 4]);
    let spec_tokens = token_vec([10, 11, 12]);
    let output_sampled_token = Token::new(20);
    let output_spec_tokens = token_vec([30, 31]);
    let output_spec_probs = vec![NotNan::new(0.6).unwrap(); output_spec_tokens.len()];
    let block_cache = initialize_block_cache(1024);
    insert_immutable_block(&block_cache, 0, None, token_vec([0, 1, 2, 3]));
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(0, ready_token_slots);
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);
    let num_spec_tokens = spec_tokens.len();
    blocks.spec_tokens = spec_tokens;
    blocks.spec_probs = vec![NotNan::new(1.0).unwrap(); num_spec_tokens];
    blocks.spec_confidences = vec![NotNan::new(1.0).unwrap(); num_spec_tokens];

    let query_tokens = blocks.prepare(3).unwrap();
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
            assert_eq!(token_vec([4]), *tokens);
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
        &token_vec([0, 1, 2, 3]),
        &token_vec([4]),
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
        &token_vec([4]),
        &[],
        &token_vec([10, 11]),
    );

    let query_tokens = blocks.prepare(3).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    assert_eq!(expected_query_tokens, query_tokens);
    assert_eq!(expected_sync_blocks, sync_blocks);
    assert_state(
        &blocks,
        2,
        &token_vec([0, 1, 2, 3]),
        &token_vec([4]),
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
            spec_confidences: vec![NotNan::new(0.75).unwrap(); output_spec_tokens.len()],
            spec_tokens: output_spec_tokens.clone(),
            spec_probs: output_spec_probs,
        },
    );
    assert_state(
        &blocks,
        2,
        &token_vec([0, 1, 2, 3, 4, 10]),
        &[],
        &token_vec([20]),
        &[],
        &output_spec_tokens,
    );
}

#[test]
fn test_prepare_cancel_commit_full_block_single_lane() {
    let total_tokens = token_vec([1, 2, 3, 4]);
    let output_sampled_token = Token::new(5);
    let block_cache = initialize_block_cache(1024);
    insert_immutable_block(&block_cache, 0, None, total_tokens.clone());
    let mut blocks = initialize_blocks(block_cache, total_tokens.clone());
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);

    let query_tokens = blocks.prepare(NUM_TOKEN_PER_BLOCK).unwrap();
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
            assert_eq!(token_vec([1, 2, 3, 4]), *tokens);
            assert!(spec_tokens.is_empty());
        },
        _ => unreachable!(),
    }
    let expected_query_tokens = query_tokens.clone();
    let expected_sync_blocks = sync_blocks.clone();
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_state(&blocks, 1, &[], &token_vec([1, 2, 3, 4]), &[], &[], &[]);

    blocks.cancel_blocks(sync_blocks);
    blocks.cancel(query_tokens);
    assert_state(&blocks, 0, &[], &[], &token_vec([1, 2, 3, 4]), &[], &[]);

    let query_tokens = blocks.prepare(NUM_TOKEN_PER_BLOCK).unwrap();
    let sync_blocks = blocks.prepare_blocks();
    assert_eq!(expected_query_tokens, query_tokens);
    assert_eq!(expected_sync_blocks, sync_blocks);
    assert_state(&blocks, 1, &[], &token_vec([1, 2, 3, 4]), &[], &[], &[]);

    let epoch = query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(
        query_tokens,
        SampledTokens::Decode {
            epoch,
            validated_tokens: vec![],
            validated_probs: vec![],
            sampled_token: output_sampled_token,
            sampled_prob: NotNan::new(0.5).unwrap(),
            spec_tokens: vec![],
            spec_probs: vec![],
            spec_confidences: vec![],
        },
    );
    assert_state(&blocks, 0, &token_vec([1, 2, 3, 4]), &[], &[], &token_vec([5]), &[]);
}

#[test]
fn test_prepare_commit_full_and_half_block_semi_immutable_collision_single_lane() {
    let total_tokens = token_vec([0, 1, 2, 3, 4, 5]);
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache.clone(), total_tokens.clone());
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);
    assert_block_counts(&blocks, 0, 1, 0);
    assert_total_tokens(&blocks.semi_immutable_blocks[0], [token_vec([0, 1, 2, 3])]);
    assert_semi_immutable_annotations(&blocks.semi_immutable_blocks[0], empty_annotations());
    assert_state(&blocks, 0, &[], &[], &token_vec([0, 1, 2, 3]), &token_vec([4, 5]), &[]);

    let query_tokens = blocks.prepare(NUM_TOKEN_PER_BLOCK).unwrap();
    let expected_query_tokens = query_tokens.clone();
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
            assert_eq!(token_vec([0, 1, 2, 3]), *tokens);
            assert_eq!(NUM_TOKEN_PER_BLOCK, *window);
        },
        _ => unreachable!(),
    }
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_state(&blocks, 1, &[], &token_vec([0, 1, 2, 3]), &[], &token_vec([4, 5]), &[]);

    let random_block = insert_immutable_block(&block_cache, 0, None, token_vec([0, 1, 2, 3]));
    let epoch = expected_query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(expected_query_tokens, SampledTokens::Prefill { epoch });
    assert_eq!(0, blocks.num_in_sync_blocks);

    let sync_blocks = blocks.prepare_blocks();
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_block_counts(&blocks, 1, 0, 0);
    assert_total_tokens(&blocks.immutable_blocks[0], [token_vec([0, 1, 2, 3])]);
    assert_immutable_annotations(&blocks.immutable_blocks[0], empty_annotations());
    assert_eq!(
        random_block
            .iter()
            .map(|block| block.trie_node_key())
            .collect::<Vec<_>>(),
        blocks.immutable_blocks[0]
            .iter()
            .map(|block| block.trie_node_key())
            .collect::<Vec<_>>()
    );
    assert_state(&blocks, 1, &token_vec([0, 1, 2, 3]), &[], &[], &token_vec([4, 5]), &[]);
}

#[test]
fn test_prepare_commit_full_and_half_block_mutable_collision_single_lane() {
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache.clone(), token_vec([0, 1]));
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);
    blocks.queued_tokens.extend(token_vec([2, 3, 4, 5]));
    blocks.try_mark_ready();

    assert_block_counts(&blocks, 0, 0, 1);
    assert_total_tokens(&blocks.mutable_blocks[0], [token_vec([0, 1, 2, 3])]);
    assert_mutable_annotations(&blocks.mutable_blocks[0], empty_annotations());
    assert_state(&blocks, 0, &[], &[], &token_vec([0, 1, 2, 3]), &token_vec([4, 5]), &[]);

    let query_tokens = blocks.prepare(NUM_TOKEN_PER_BLOCK).unwrap();
    let expected_query_tokens = query_tokens.clone();
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
            assert_eq!(token_vec([0, 1, 2, 3]), *tokens);
            assert_eq!(NUM_TOKEN_PER_BLOCK, *window);
        },
        _ => unreachable!(),
    }
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_state(&blocks, 1, &[], &token_vec([0, 1, 2, 3]), &[], &token_vec([4, 5]), &[]);

    let random_block = insert_immutable_block(&block_cache, 0, None, token_vec([0, 1, 2, 3]));
    let epoch = expected_query_tokens.epoch();
    blocks.commit_blocks(sync_blocks);
    blocks.commit(expected_query_tokens, SampledTokens::Prefill { epoch });
    assert_eq!(0, blocks.num_in_sync_blocks);

    let sync_blocks = blocks.prepare_blocks();
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_block_counts(&blocks, 1, 0, 0);
    assert_total_tokens(&blocks.immutable_blocks[0], [token_vec([0, 1, 2, 3])]);
    assert_immutable_annotations(&blocks.immutable_blocks[0], empty_annotations());
    assert_eq!(
        random_block
            .iter()
            .map(|block| block.trie_node_key())
            .collect::<Vec<_>>(),
        blocks.immutable_blocks[0]
            .iter()
            .map(|block| block.trie_node_key())
            .collect::<Vec<_>>()
    );
    assert_state(&blocks, 1, &token_vec([0, 1, 2, 3]), &[], &[], &token_vec([4, 5]), &[]);
}

#[test]
fn test_prepare_commit_mutable_collision_additional_validated_token_single_lane() {
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache.clone(), token_vec([0, 1, 2]));
    blocks.spec_tokens = token_vec([10]);
    blocks.spec_probs = vec![NotNan::new(1.0).unwrap()];
    blocks.spec_confidences = vec![NotNan::new(1.0).unwrap()];
    let InitBlockOnceResult::Success { ready_token_slots } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_eq!(NUM_TOKEN_PER_BLOCK, ready_token_slots);

    let query_tokens = blocks.prepare(NUM_TOKEN_PER_BLOCK).unwrap();
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
            assert_eq!(token_vec([0, 1, 2]), *tokens);
            assert_eq!(token_vec([10]), *spec_tokens);
        },
        _ => unreachable!(),
    }
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_block_counts(&blocks, 0, 0, 1);
    assert_total_tokens(&blocks.mutable_blocks[0], [token_vec([0, 1, 2])]);
    assert_mutable_annotations(&blocks.mutable_blocks[0], empty_annotations());
    assert_state(&blocks, 1, &[], &token_vec([0, 1, 2]), &[], &[], &token_vec([10]));

    let random_block = insert_immutable_block(&block_cache, 0, None, token_vec([0, 1, 2, 10]));
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
            spec_confidences: vec![NotNan::new(0.75).unwrap(); 2],
        },
    );
    assert_eq!(0, blocks.num_in_sync_blocks);

    let sync_blocks = blocks.prepare_blocks();
    assert_sync_blocks(&sync_blocks, &blocks, 0);
    assert_block_counts(&blocks, 1, 0, 0);
    assert_total_tokens(&blocks.immutable_blocks[0], [token_vec([0, 1, 2, 10])]);
    assert_immutable_annotations(&blocks.immutable_blocks[0], empty_annotations());
    assert_eq!(
        random_block
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
        &token_vec([0, 1, 2, 10]),
        &[],
        &[],
        &token_vec([20]),
        &token_vec([30, 31]),
    );
}

#[test]
fn test_enqueue_tokens_single_lane() {
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache, token_vec([1, 2]));

    let InitBlockOnceResult::Success { .. } = blocks.init_block_once() else {
        unreachable!()
    };
    assert_block_counts(&blocks, 0, 0, 1);
    assert_total_tokens(&blocks.mutable_blocks[0], [token_vec([1, 2])]);
    assert_state(&blocks, 0, &[], &[], &token_vec([1, 2]), &[], &[]);

    blocks.queued_tokens.extend(token_vec([3, 4, 5, 6]));
    blocks.try_mark_ready();
    assert_block_counts(&blocks, 0, 0, 1);
    assert_total_tokens(&blocks.mutable_blocks[0], [token_vec([1, 2, 3, 4])]);
    assert_state(&blocks, 0, &[], &[], &token_vec([1, 2, 3, 4]), &token_vec([5, 6]), &[]);
}

#[test]
fn test_parent_node_single_lane() {
    let root_total_tokens = token_vec([1, 2, 3, 4]);
    let child_total_tokens = token_vec([5, 6, 7, 8]);
    let block_cache = initialize_block_cache(1024);
    let mut blocks = initialize_blocks(block_cache.clone(), root_total_tokens.clone());
    let expected_empty_annotations: [smallvec::SmallVec<[BlockAnnotation; 1]>; NUM_CACHE_LANE] =
        std::array::from_fn(|_| smallvec::SmallVec::new());
    assert_eq!([None], blocks.parent_trie_node_key_vec(0));
    assert_eq!(expected_empty_annotations, blocks.block_annotation_vec(0));

    let root_immutable_block_vec = insert_immutable_block(&block_cache, 0, None, root_total_tokens);
    blocks.queued_tokens = child_total_tokens.into();
    blocks.immutable_blocks.push(root_immutable_block_vec);
    let immutable_block_vec = &blocks.immutable_blocks[0];
    assert_eq!(
        [Some(immutable_block_vec[0].trie_node_key())],
        blocks.parent_trie_node_key_vec(1)
    );
    assert_eq!(expected_empty_annotations, blocks.block_annotation_vec(1));
}

fn initialize_blocks(block_cache: Arc<TestMultiLaneTrieKVBlockCache>, total_tokens: Vec<Token>) -> TestTrieBlocks {
    assert!(!total_tokens.is_empty());
    TestTrieBlocks::new(
        block_cache,
        Vec::new(),
        Vec::new(),
        std::iter::empty::<Token>(),
        std::iter::empty::<Token>(),
        total_tokens,
    )
}

fn assert_sync_blocks(sync_blocks: &DecoderSyncBlocks, blocks: &TestTrieBlocks, block_index: usize) {
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

fn initialize_block_cache(max_kv_pages: usize) -> Arc<TestMultiLaneTrieKVBlockCache> {
    let page_id_allocator = Arc::new(U32IDAllocator::new(max_kv_pages as u64));
    let kv_block_allocator = TPKVBlockAllocator::new(NUM_KV_PAGES_PER_BLOCK, page_id_allocator.clone());
    let state_block_allocator = TPStateBlockAllocator::new(NUM_STATE_PAGES_PER_BLOCK, page_id_allocator.clone());
    let single_lane_block_cache = Arc::new(TestSingleLaneTrieKVBlockCache::new(
        kv_block_allocator,
        state_block_allocator,
        BLOCK_CACHE_CAPACITY,
        Shutdown::new(),
    ));
    Arc::new(TestMultiLaneTrieKVBlockCache::new([single_lane_block_cache.clone()]))
}

fn insert_immutable_block(
    block_cache: &TestMultiLaneTrieKVBlockCache,
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
    }
    let scheduled_tokens = block_vec[0].schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
    block_vec[0].cache_tokens(scheduled_tokens.len());

    let CommitMultiLaneMutableBlockResult::Immutable { block_vec } =
        block_cache.commit_mutable_block(parent_trie_node_key_vec, block_vec)
    else {
        unreachable!()
    };
    block_vec
}

fn insert_immutable_block_with_annotations(
    block_cache: &TestMultiLaneTrieKVBlockCache,
    total_tokens: Vec<Token>,
    annotations: [SmallVec<[BlockAnnotation; 1]>; NUM_CACHE_LANE],
) -> [TestImmutableBlock; NUM_CACHE_LANE] {
    assert!(NUM_TOKEN_PER_BLOCK <= total_tokens.len());
    let block_metadata_vec: [BlockMetadata<NUM_TOKEN_PER_BLOCK>; NUM_CACHE_LANE] = [BlockMetadata::new(
        None,
        annotations[0].clone(),
        Arc::<[Token]>::from(total_tokens[..NUM_TOKEN_PER_BLOCK].to_vec()),
    )];
    let AllocateMultiLaneMutableBlockResult::Mutable { mut block_vec } =
        block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>()
    else {
        unreachable!()
    };
    for (block, block_metadata) in block_vec.iter_mut().zip(block_metadata_vec.iter()) {
        block.insert_annotations(block_metadata.annotations().iter().cloned());
        block.write_tokens(0, block_metadata.tokens());
    }
    let scheduled_tokens = block_vec[0].schedule_tokens(NUM_TOKEN_PER_BLOCK).to_vec();
    block_vec[0].cache_tokens(scheduled_tokens.len());
    let CommitMultiLaneMutableBlockResult::Immutable { block_vec } =
        block_cache.commit_mutable_block([None], block_vec)
    else {
        unreachable!()
    };
    block_vec
}

fn reserve_semi_immutable_block(
    block_cache: &TestMultiLaneTrieKVBlockCache,
    block_index: usize,
    parent_immutable_block_vec: Option<&[TestImmutableBlock; NUM_CACHE_LANE]>,
    total_tokens: Vec<Token>,
) -> [crate::runtime::decoder::trie_cache::SemiImmutableBlock<NUM_TOKEN_PER_BLOCK>; NUM_CACHE_LANE] {
    let block_metadata_vec = block_metadata_vec(block_index, parent_immutable_block_vec, total_tokens);
    let ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable { block_vec } =
        block_cache.reserve_semi_immutable_block(block_metadata_vec)
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
    assert_eq!(NUM_TOKEN_PER_BLOCK, total_tokens.len());
    let parent_trie_node_key = if block_index == 0 {
        None
    } else {
        Some(
            parent_immutable_block_vec
                .expect("block_metadata_vec: non-root kv block must have parent immutable kv block vec")[0]
                .trie_node_key(),
        )
    };

    [BlockMetadata::new(
        parent_trie_node_key,
        vec![].into(),
        Arc::<[Token]>::from(total_tokens),
    )]
}

fn allocate_mutable_block(
    block_cache: &TestMultiLaneTrieKVBlockCache,
) -> [MutableBlock<NUM_TOKEN_PER_BLOCK>; NUM_CACHE_LANE] {
    let AllocateMultiLaneMutableBlockResult::Mutable { block_vec } =
        block_cache.alloc_mutable_block::<NUM_TOKEN_PER_BLOCK>()
    else {
        unreachable!()
    };
    block_vec
}

fn empty_annotations() -> [Vec<BlockAnnotation>; NUM_CACHE_LANE] {
    std::array::from_fn(|_| vec![])
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
    block_vec: &[crate::runtime::decoder::trie_cache::SemiImmutableBlock<NUM_TOKEN_PER_BLOCK>; NUM_CACHE_LANE],
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
    blocks: &TestTrieBlocks,
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

fn assert_block_counts(blocks: &TestTrieBlocks, num_immutable: usize, num_semi_immutable: usize, num_mutable: usize) {
    assert_eq!(num_immutable, blocks.immutable_blocks.len());
    assert_eq!(num_semi_immutable, blocks.semi_immutable_blocks.len());
    assert_eq!(num_mutable, blocks.mutable_blocks.len());
}
