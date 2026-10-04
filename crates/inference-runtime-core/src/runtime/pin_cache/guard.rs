#![allow(clippy::type_complexity)]

use std::hash::BuildHasher;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::Mutex;

use crate::runtime::pin_cache::Key;
use crate::runtime::pin_cache::Value;
use crate::runtime::pin_cache::Weight;
use crate::runtime::pin_cache::entry::Entry;
use crate::runtime::pin_cache::list::LinkedNode;
use crate::runtime::pin_cache::shard::Shard;

#[derive(Debug)]
pub struct PinGuard<K, V, W, H>
where
    K: Key,
    V: Value,
    W: Weight<K, V>,
    H: BuildHasher + Default + 'static,
{
    shard: Arc<Mutex<Shard<K, V, W, H>>>,
    node_nn: NonNull<LinkedNode<Entry<K, V>>>,
}

impl<K, V, W, H> PinGuard<K, V, W, H>
where
    K: Key,
    V: Value,
    W: Weight<K, V>,
    H: BuildHasher + Default + 'static,
{
    pub fn new(shard: Arc<Mutex<Shard<K, V, W, H>>>, node_nn: NonNull<LinkedNode<Entry<K, V>>>) -> Self {
        unsafe { debug_assert!(0 < node_nn.as_ref().pin_count()) };
        Self { shard, node_nn }
    }

    pub fn key(&self) -> &K {
        unsafe { self.node_nn.as_ref().key() }
    }

    pub fn value(&self) -> &V {
        unsafe { self.node_nn.as_ref().value() }
    }

    #[cfg(test)]
    pub fn node_ptr(&self) -> NonNull<LinkedNode<Entry<K, V>>> {
        self.node_nn
    }
}

impl<K, V, W, H> Clone for PinGuard<K, V, W, H>
where
    K: Key,
    V: Value,
    W: Weight<K, V>,
    H: BuildHasher + Default + 'static,
{
    #[inline]
    fn clone(&self) -> Self {
        let node_ref = unsafe { self.node_nn.as_ref() };
        let _ = node_ref.pin();
        Self {
            shard: self.shard.clone(),
            node_nn: self.node_nn,
        }
    }
}

impl<K, V, W, H> Drop for PinGuard<K, V, W, H>
where
    K: Key,
    V: Value,
    W: Weight<K, V>,
    H: BuildHasher + Default + 'static,
{
    #[inline]
    fn drop(&mut self) {
        let mut guard = self.shard.lock().unwrap();
        let node_ref = unsafe { self.node_nn.as_ref() };
        let pin_count = node_ref.unpin();
        if pin_count == 0 {
            guard.detach_unpin(self.node_nn);
        }
    }
}

unsafe impl<K, V, W, H> Send for PinGuard<K, V, W, H>
where
    K: Key,
    V: Value,
    W: Weight<K, V>,
    H: BuildHasher + Default + 'static,
{
}

unsafe impl<K, V, W, H> Sync for PinGuard<K, V, W, H>
where
    K: Key,
    V: Value,
    W: Weight<K, V>,
    H: BuildHasher + Default + 'static,
{
}
