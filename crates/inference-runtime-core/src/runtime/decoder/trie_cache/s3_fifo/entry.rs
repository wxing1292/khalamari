use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

use ahash::RandomState as AHashRandomState;
use crossbeam_channel::Sender;
use dashmap::DashMap;
use dashmap::Entry as DashEntry;

use crate::channel::Shutdown;
use crate::runtime::decoder::trie_cache::EvictionEntry;
use crate::runtime::decoder::trie_cache::s3_fifo::control::Control;

const MAX_COUNT: u8 = 3;

#[derive(Debug)]
pub struct S3FIFOEntry<K>
where
    K: Copy + Eq + Hash + Send + Sync + 'static,
{
    key: K,
    pinned: AtomicBool,
    count: AtomicU8,
    evictions: Weak<DashMap<K, Arc<S3FIFOEntry<K>>, AHashRandomState>>,
    signal_sender: Sender<Control<K>>,
    shutdown: Shutdown,
}

impl<K> S3FIFOEntry<K>
where
    K: Copy + Eq + Hash + Send + Sync + 'static,
{
    pub fn new(
        key: K,
        evictions: Weak<DashMap<K, Arc<S3FIFOEntry<K>>, AHashRandomState>>,
        signal_sender: Sender<Control<K>>,
        shutdown: Shutdown,
    ) -> Self {
        Self {
            key,
            pinned: AtomicBool::new(true),
            count: AtomicU8::new(1),
            evictions,
            signal_sender,
            shutdown,
        }
    }

    pub fn key(&self) -> K {
        self.key
    }

    fn inc_count(&self) {
        loop {
            let count = self.count.load(Ordering::Relaxed);
            if count == MAX_COUNT {
                return;
            }
            if self
                .count
                .compare_exchange_weak(count, count + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }
        }
    }

    pub fn dec_count(&self) -> bool {
        loop {
            let count = self.count.load(Ordering::Relaxed);
            if count <= 1 {
                return false;
            }
            if self
                .count
                .compare_exchange_weak(count, count - 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
    }

    pub fn is_pinned(&self) -> bool {
        self.pinned.load(Ordering::Acquire)
    }

    pub fn reset_count(&self) {
        self.count.store(1, Ordering::Relaxed);
    }

    pub fn count(&self) -> u8 {
        self.count.load(Ordering::Relaxed)
    }

    fn signal(&self) {
        if let Err(error) = self.signal_sender.send(Control::Reevaluate { key: self.key }) {
            if self.shutdown.is_shutdown() {
                return;
            }
            panic!("signal: S3FIFO worker must be running: {error}");
        }
    }
}

impl<K> EvictionEntry for S3FIFOEntry<K>
where
    K: Copy + Debug + Eq + Hash + Send + Sync + 'static,
{
    fn register(self: Arc<Self>) {
        debug_assert!(self.is_pinned(), "register: entry must start pinned");
        debug_assert_eq!(1, self.count(), "register: entry must start with count == 1");
        let Some(evictions) = self.evictions.upgrade() else {
            if self.shutdown.is_shutdown() {
                return;
            }
            panic!("register: eviction policy must own its registry");
        };
        match evictions.entry(self.key()) {
            DashEntry::Vacant(entry) => {
                entry.insert(self);
            },
            DashEntry::Occupied(_) => panic!("register: key must not already have an eviction entry"),
        }
    }

    fn pin(&self) {
        let pinned = self.pinned.swap(true, Ordering::AcqRel);
        debug_assert!(!pinned, "pin: entry must not already be pinned");
        self.signal();
    }

    fn unpin(&self) {
        let pinned = self.pinned.swap(false, Ordering::AcqRel);
        debug_assert!(pinned, "unpin: entry must be pinned");
        self.signal();
    }

    fn touch(&self) {
        self.inc_count();
    }
}
