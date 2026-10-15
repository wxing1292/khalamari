use std::cell::Cell;
use std::hash::Hash;
use std::sync::Arc;

use intrusive_collections::LinkedListLink;
use intrusive_collections::intrusive_adapter;

use crate::runtime::decoder::trie_cache::s3_fifo::entry::S3FIFOEntry;
use crate::runtime::decoder::trie_cache::s3_fifo::queue::Queue;

pub struct QueueEntry<K>
where
    K: Copy + Eq + Hash + Send + Sync + 'static,
{
    pub eviction_entry: Arc<S3FIFOEntry<K>>,
    pub queue_link: LinkedListLink,
    pub queue: Cell<Queue>,
    pub is_candidate: Cell<bool>,
}

intrusive_adapter!(pub QueueEntryLink<K> = Arc<QueueEntry<K>>: QueueEntry<K> { queue_link => LinkedListLink } where K: Copy + Eq + Hash + Send + Sync + 'static);
