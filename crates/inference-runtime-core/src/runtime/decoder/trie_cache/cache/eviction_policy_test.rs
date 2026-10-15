use std::sync::Arc;
use std::sync::Mutex;

use smallvec::SmallVec;

use crate::memory::U32IDAllocator;
use crate::runtime::Token;
use crate::runtime::decoder::TPKVBlockAllocator;
use crate::runtime::decoder::TPStateBlockAllocator;
use crate::runtime::decoder::trie_cache::AllocateMultiLaneMutableBlockResult;
use crate::runtime::decoder::trie_cache::BlockMetadata;
use crate::runtime::decoder::trie_cache::CommitMultiLaneMutableBlockResult;
use crate::runtime::decoder::trie_cache::DecoderBlock;
use crate::runtime::decoder::trie_cache::EvictionEntry;
use crate::runtime::decoder::trie_cache::EvictionPolicy;
use crate::runtime::decoder::trie_cache::MultiLaneBlockCache;
use crate::runtime::decoder::trie_cache::MultiLaneTrieBlockCache;
use crate::runtime::decoder::trie_cache::ReserveMultiLaneSemiImmutableBlockResult;
use crate::runtime::decoder::trie_cache::SingleLaneTrieBlockCache;
use crate::runtime::decoder::trie_cache::TrieNodeKey;

#[test]
fn test_eviction_policy_lifecycle() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let key = Arc::new(Mutex::new(None));
    let mut policy = MockPolicy::new();
    let entry_events = events.clone();
    let entry_key = key.clone();
    policy.expect_new_entry().once().returning(move |key| {
        *entry_key.lock().unwrap() = Some(key);
        Arc::new(RecordingEntry(entry_events.clone()))
    });
    let candidate_key = key.clone();
    policy
        .expect_evict_candidate()
        .times(2)
        .returning(move || *candidate_key.lock().unwrap());
    let rejected_key = key.clone();
    policy
        .expect_reject_candidate()
        .withf(move |key| Some(*key) == *rejected_key.lock().unwrap())
        .once()
        .return_const(());
    policy
        .expect_accept_candidate()
        .withf(move |candidate| Some(*candidate) == *key.lock().unwrap())
        .once()
        .return_const(());

    let allocator = Arc::new(U32IDAllocator::new(8));
    let cache = SingleLaneTrieBlockCache::<4, _, _, _>::with_eviction_policy(
        TPKVBlockAllocator::new(1, allocator.clone()),
        TPStateBlockAllocator::new(1, allocator.clone()),
        Arc::new(policy),
    );
    let cache = MultiLaneTrieBlockCache::new([Arc::new(cache)]);
    let tokens: Arc<[Token]> = [1, 2, 3, 4].map(Token::new).into();
    let AllocateMultiLaneMutableBlockResult::Mutable { block_vec: [mut block] } = cache.alloc_mutable_block::<4>()
    else {
        panic!("block allocation must succeed");
    };
    block.write_tokens(0, &tokens);
    block.schedule_tokens(4);
    block.cache_tokens(4);
    let CommitMultiLaneMutableBlockResult::Immutable { block_vec } = cache.commit_mutable_block([None], [block]) else {
        panic!("the first publication must succeed");
    };
    drop(block_vec);

    let ReserveMultiLaneSemiImmutableBlockResult::Immutable { block_vec } =
        cache.reserve_semi_immutable_block([BlockMetadata::<4>::new(None, SmallVec::new(), tokens)])
    else {
        panic!("the published block must be reused");
    };
    assert!(!cache.try_evict());
    assert_eq!(2, allocator.used());
    drop(block_vec);
    assert!(cache.try_evict());
    assert_eq!(0, allocator.used());
    assert_eq!(
        ["register", "unpin", "pin", "touch", "unpin"],
        events.lock().unwrap().as_slice()
    );
}

#[derive(Debug)]
pub struct RecordingEntry(Arc<Mutex<Vec<&'static str>>>);

impl EvictionEntry for RecordingEntry {
    fn register(self: Arc<Self>) {
        self.0.lock().unwrap().push("register");
    }

    fn pin(&self) {
        self.0.lock().unwrap().push("pin");
    }

    fn unpin(&self) {
        self.0.lock().unwrap().push("unpin");
    }

    fn touch(&self) {
        self.0.lock().unwrap().push("touch");
    }
}

mockall::mock! {
    #[derive(Debug)]
    Policy {}

    impl EvictionPolicy<TrieNodeKey> for Policy {
        type Entry = RecordingEntry;

        fn new_entry(&self, key: TrieNodeKey) -> Arc<RecordingEntry>;
        fn evict_candidate(&self) -> Option<TrieNodeKey>;
        fn reject_candidate(&self, key: TrieNodeKey);
        fn accept_candidate(&self, key: TrieNodeKey);
    }
}
