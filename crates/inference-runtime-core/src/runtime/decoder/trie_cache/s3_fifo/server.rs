use std::hash::Hash;
use std::sync::Arc;

use ahash::AHashMap;
use ahash::RandomState as AHashRandomState;
use dashmap::DashMap;
use intrusive_collections::LinkedList;

use crate::channel::Shutdown;
use crate::runtime::decoder::trie_cache::s3_fifo::entry::S3FIFOEntry;
use crate::runtime::decoder::trie_cache::s3_fifo::queue::Queue;
use crate::runtime::decoder::trie_cache::s3_fifo::queue_entry::QueueEntry;
use crate::runtime::decoder::trie_cache::s3_fifo::queue_entry::QueueEntryLink;

pub struct S3FIFOServer<K>
where
    K: Copy + Eq + Hash + Send + Sync + 'static,
{
    entries: AHashMap<K, Arc<QueueEntry<K>>>,
    evictions: Arc<DashMap<K, Arc<S3FIFOEntry<K>>, AHashRandomState>>,
    small: LinkedList<QueueEntryLink<K>>,
    main: LinkedList<QueueEntryLink<K>>,
    num_small_entries: usize,
    num_main_entries: usize,
    shutdown: Shutdown,
}

impl<K> S3FIFOServer<K>
where
    K: Copy + Eq + Hash + Send + Sync + 'static,
{
    pub fn new(evictions: Arc<DashMap<K, Arc<S3FIFOEntry<K>>, AHashRandomState>>, shutdown: Shutdown) -> Self {
        Self {
            entries: AHashMap::with_hasher(AHashRandomState::new()),
            evictions,
            small: LinkedList::new(QueueEntryLink::new()),
            main: LinkedList::new(QueueEntryLink::new()),
            num_small_entries: 0,
            num_main_entries: 0,
            shutdown,
        }
    }

    pub fn shutdown(&self) -> &Shutdown {
        &self.shutdown
    }

    pub fn entry(&self, key: K) -> Option<Arc<QueueEntry<K>>> {
        self.entries.get(&key).cloned()
    }

    fn ensure_entry(&mut self, key: K) -> Option<Arc<QueueEntry<K>>> {
        if let Some(entry) = self.entry(key) {
            return Some(entry);
        }
        let eviction = self.evictions.get(&key)?.clone();
        let entry = Arc::new(QueueEntry {
            eviction_entry: eviction,
            queue_link: intrusive_collections::LinkedListLink::new(),
            queue: std::cell::Cell::new(Queue::S),
            is_candidate: std::cell::Cell::new(false),
        });
        let replaced = self.entries.insert(key, entry.clone());
        debug_assert!(
            replaced.is_none(),
            "ensure_entry: entry must not replace an existing key"
        );
        Some(entry)
    }

    pub fn list_mut(&mut self, queue: Queue) -> &mut LinkedList<QueueEntryLink<K>> {
        match queue {
            Queue::S => &mut self.small,
            Queue::M => &mut self.main,
        }
    }

    pub fn num_entries(&self, queue: Queue) -> usize {
        match queue {
            Queue::S => self.num_small_entries,
            Queue::M => self.num_main_entries,
        }
    }

    fn increment_num_entries(&mut self, queue: Queue) {
        match queue {
            Queue::S => self.num_small_entries += 1,
            Queue::M => self.num_main_entries += 1,
        }
    }

    fn decrement_num_entries(&mut self, queue: Queue) {
        let num_entries = match queue {
            Queue::S => &mut self.num_small_entries,
            Queue::M => &mut self.num_main_entries,
        };
        debug_assert!(0 < *num_entries, "decrement_num_entries: queue must not be empty");
        *num_entries -= 1;
    }

    pub fn attach(&mut self, entry: Arc<QueueEntry<K>>) {
        assert!(
            !entry.queue_link.is_linked(),
            "attach: entry must be detached before queue insertion"
        );
        debug_assert!(
            !entry.is_candidate.get(),
            "attach: candidate entry must resolve before queue insertion"
        );
        let queue = entry.queue.get();
        self.list_mut(queue).push_back(entry);
        self.increment_num_entries(queue);
    }

    pub fn detach(&mut self, entry: &Arc<QueueEntry<K>>) {
        assert!(entry.queue_link.is_linked(), "detach: entry must be attached");
        let queue = entry.queue.get();
        let removed = unsafe {
            self.list_mut(queue)
                .cursor_mut_from_ptr(Arc::as_ptr(entry))
                .remove()
                .expect("detach: entry must be linked in its queue")
        };
        assert!(
            Arc::ptr_eq(entry, &removed),
            "detach: removed entry must match requested entry"
        );
        drop(removed);
        self.decrement_num_entries(queue);
    }

    pub fn remove(&mut self, key: K) {
        if let Some(entry) = self.entry(key) {
            if entry.queue_link.is_linked() {
                self.detach(&entry);
            }
            let removed = self.entries.remove(&key);
            debug_assert!(removed.is_some(), "remove: entry must exist after unlink");
        }
        let _ = self.evictions.remove(&key);
    }

    pub fn re_evaluate(&mut self, key: K) {
        let Some(entry) = self.ensure_entry(key) else {
            return;
        };
        if entry.is_candidate.get() {
            return;
        }
        match (entry.eviction_entry.is_pinned(), entry.queue_link.is_linked()) {
            (true, true) => self.detach(&entry),
            (false, false) => self.attach(entry),
            _ => {},
        }
    }

    pub fn next_candidate(&mut self, queue: Queue) -> Option<K> {
        while let Some(entry) = self.list_mut(queue).pop_front() {
            self.decrement_num_entries(queue);
            debug_assert_eq!(
                queue,
                entry.queue.get(),
                "next_candidate: entry must be in requested queue"
            );
            debug_assert!(
                !entry.queue_link.is_linked(),
                "next_candidate: popped entry must be detached"
            );

            if queue == Queue::S && 1 < entry.eviction_entry.count() {
                entry.eviction_entry.reset_count();
                entry.queue.set(Queue::M);
                if !entry.eviction_entry.is_pinned() {
                    self.attach(entry);
                }
            } else if queue == Queue::M && entry.eviction_entry.dec_count() {
                if !entry.eviction_entry.is_pinned() {
                    self.attach(entry);
                }
            } else if !entry.eviction_entry.is_pinned() {
                entry.is_candidate.set(true);
                return Some(entry.eviction_entry.key());
            }
        }
        None
    }

    pub fn evict_candidate(&mut self, s_cap: usize, m_cap: usize) -> Option<K> {
        let prefer_s = self.num_entries(Queue::S) > s_cap || self.num_entries(Queue::M) <= m_cap;
        if prefer_s {
            self.next_candidate(Queue::S).or_else(|| self.next_candidate(Queue::M))
        } else {
            self.next_candidate(Queue::M).or_else(|| self.next_candidate(Queue::S))
        }
    }

    pub fn reject_candidate(&mut self, key: K) {
        let entry = self.entry(key).expect("reject_candidate: entry must exist");
        let is_candidate = entry.is_candidate.replace(false);
        debug_assert!(
            is_candidate,
            "reject_candidate: entry must be the outstanding candidate"
        );
        self.re_evaluate(key);
    }

    pub fn accept_candidate(&mut self, key: K) {
        let entry = self.entry(key).expect("accept_candidate: entry must exist");
        debug_assert!(
            entry.is_candidate.get(),
            "accept_candidate: entry must be the outstanding candidate"
        );
        debug_assert!(
            !entry.eviction_entry.is_pinned(),
            "accept_candidate: candidate must not be pinned"
        );
        self.remove(key);
    }
}
