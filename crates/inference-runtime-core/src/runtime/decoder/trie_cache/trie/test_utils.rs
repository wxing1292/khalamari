use std::sync::Arc;

use smallvec::SmallVec;
use smallvec::smallvec;
use uuid::Uuid;

use crate::channel::Shutdown;
use crate::memory::DeviceBlock;
use crate::memory::U32IDAllocator;
use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;
use crate::runtime::decoder::KVBlockPlacement;
use crate::runtime::decoder::ResourceSegment;
use crate::runtime::decoder::StateBlockPlacement;
use crate::runtime::decoder::trie_cache::S3FIFOClient;
use crate::runtime::decoder::trie_cache::S3FIFOEntry;
use crate::runtime::decoder::trie_cache::Trie;
use crate::runtime::decoder::trie_cache::TrieEdge;
use crate::runtime::decoder::trie_cache::TrieNodeKey;
use crate::runtime::resource::ResourceID;

pub const TEST_PARTITIONS: usize = 4;

pub fn new_trie() -> Arc<Trie<TEST_PARTITIONS, S3FIFOEntry<TrieNodeKey>>> {
    Arc::new(Trie::new(Arc::new(S3FIFOClient::new(32, Shutdown::new()))))
}

pub fn new_annotations(seed: u8) -> SmallVec<[BlockAnnotation; 1]> {
    let mut bytes = [seed; 16];
    bytes[6] = (bytes[6] & 0x0f) | 0x80;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let resource_id = ResourceID::from_uuid(Uuid::from_bytes(bytes)).unwrap();
    smallvec![BlockAnnotation::resource(ResourceSegment::new(
        resource_id,
        seed as u16,
        seed as u32,
        1,
    ))]
}

pub fn new_tokens(values: &[u32]) -> Arc<[Token]> {
    values.iter().copied().map(Token::new).collect::<Vec<_>>().into()
}

pub fn new_edge(seed: u8, values: &[u32]) -> TrieEdge {
    TrieEdge::new(new_annotations(seed), new_tokens(values))
}

pub fn new_kv_placement() -> KVBlockPlacement {
    let allocator = Arc::new(U32IDAllocator::new(8));
    let page_ids = allocator.alloc_many(1).unwrap();
    KVBlockPlacement::Device {
        block: DeviceBlock::tp(allocator, page_ids),
    }
}

pub fn new_state_placement() -> StateBlockPlacement {
    let allocator = Arc::new(U32IDAllocator::new(8));
    let page_ids = allocator.alloc_many(1).unwrap();
    StateBlockPlacement::Device {
        block: DeviceBlock::tp(allocator, page_ids),
    }
}
