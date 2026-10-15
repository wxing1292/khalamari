use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use slotmap::DenseSlotMap;

pub trait DataKey: Copy + Clone + Send + Sync + 'static {
    type LocalKey: slotmap::Key;

    fn partition_key(&self) -> u32;
    fn local_key(&self) -> Self::LocalKey;

    fn from_parts(partition_key: u32, local_key: Self::LocalKey) -> Self;
}

pub trait DataValue {
    fn partition_key(&self) -> u32;
}

#[derive(Clone, Debug)]
pub struct PartitionedDataStore<const P: usize, K, V>
where
    K: DataKey,
    V: DataValue,
{
    inner: Arc<PartitionedSlotMapInner<P, K, V>>,
}

impl<const P: usize, K, V> PartitionedDataStore<P, K, V>
where
    K: DataKey,
    V: DataValue,
{
    pub fn new() -> Self {
        Self {
            inner: Arc::new(PartitionedSlotMapInner::new()),
        }
    }

    delegate::delegate! {
        to self.inner {
            pub fn contains_key(&self, key: K) -> bool;
            pub fn insert(&self, value: V) -> K;
            pub fn remove(&self, key: K) -> Option<V>;
            pub fn get_ref(&self, key: K) -> Option<DataStoreValueRef<'_, K, V>>;
            pub fn get_mut(&self, key: K) -> Option<DataStoreValueMut<'_, K, V>>;
            pub fn is_empty(&self) -> bool;
            pub fn len(&self) -> usize;
        }
    }
}

#[derive(Debug)]
struct PartitionedSlotMapInner<const P: usize, K, V>
where
    K: DataKey,
    V: DataValue,
{
    size: AtomicUsize,
    data_array: [Mutex<DenseSlotMap<K::LocalKey, V>>; P],
}

impl<const P: usize, K, V> PartitionedSlotMapInner<P, K, V>
where
    K: DataKey,
    V: DataValue,
{
    pub fn new() -> Self {
        Self {
            size: AtomicUsize::new(0),
            data_array: std::array::from_fn(|_| Mutex::new(DenseSlotMap::with_key())),
        }
    }

    pub fn contains_key(&self, key: K) -> bool {
        let partition_key = key.partition_key();
        debug_assert!(
            (partition_key as usize) < P,
            "PartitionedSlotMapInner::contains_key: partition_key={partition_key:?} out of range for P={P}"
        );
        let guard = self.data_array[partition_key as usize].lock().unwrap();
        guard.contains_key(key.local_key())
    }

    pub fn insert(&self, value: V) -> K {
        self.size.fetch_add(1, Ordering::Relaxed);
        let partition_key = value.partition_key();
        debug_assert!(
            (partition_key as usize) < P,
            "PartitionedSlotMapInner::insert: partition_key={partition_key:?} out of range for P={P}"
        );
        let mut guard = self.data_array[partition_key as usize].lock().unwrap();
        let slot_map_key = guard.insert(value);
        K::from_parts(partition_key, slot_map_key)
    }

    pub fn remove(&self, key: K) -> Option<V> {
        let partition_key = key.partition_key();
        debug_assert!(
            (partition_key as usize) < P,
            "PartitionedSlotMapInner::remove: partition_key={partition_key:?} out of range for P={P}"
        );
        let mut guard = self.data_array[partition_key as usize].lock().unwrap();
        let value = guard.remove(key.local_key())?;
        self.size.fetch_sub(1, Ordering::Relaxed);
        Some(value)
    }

    pub fn get_ref(&self, key: K) -> Option<DataStoreValueRef<'_, K, V>> {
        let partition_key = key.partition_key();
        debug_assert!(
            (partition_key as usize) < P,
            "PartitionedSlotMapInner::get_ref: partition_key={partition_key:?} out of range for P={P}"
        );
        let guard = self.data_array[partition_key as usize].lock().unwrap();
        let value = guard.get(key.local_key())? as *const V;
        Some(DataStoreValueRef::new(guard, key, value))
    }

    pub fn get_mut(&self, key: K) -> Option<DataStoreValueMut<'_, K, V>> {
        let partition_key = key.partition_key();
        debug_assert!(
            (partition_key as usize) < P,
            "PartitionedSlotMapInner::get_mut: partition_key={partition_key:?} out of range for P={P}"
        );
        let mut guard = self.data_array[partition_key as usize].lock().unwrap();
        let value = guard.get_mut(key.local_key())? as *mut V;
        Some(DataStoreValueMut::new(guard, key, value))
    }

    pub fn is_empty(&self) -> bool {
        self.size.load(Ordering::Relaxed) == 0
    }

    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }
}

pub struct DataStoreValueRef<'a, K, V>
where
    K: DataKey,
    V: DataValue,
{
    _guard: MutexGuard<'a, DenseSlotMap<K::LocalKey, V>>,
    key: K,
    value: *const V,
}

impl<'a, K, V> DataStoreValueRef<'a, K, V>
where
    K: DataKey,
    V: DataValue,
{
    pub fn new(guard: MutexGuard<'a, DenseSlotMap<K::LocalKey, V>>, key: K, value: *const V) -> Self {
        Self {
            _guard: guard,

            key,
            value,
        }
    }

    pub fn key(&self) -> K {
        self.key
    }

    pub fn value_ref(&self) -> &V {
        self.deref()
    }
}

impl<'a, K, V> std::ops::Deref for DataStoreValueRef<'a, K, V>
where
    K: DataKey,
    V: DataValue,
{
    type Target = V;

    fn deref(&self) -> &V {
        unsafe { &*self.value }
    }
}

pub struct DataStoreValueMut<'a, K, V>
where
    K: DataKey,
    V: DataValue,
{
    _guard: MutexGuard<'a, DenseSlotMap<K::LocalKey, V>>,
    key: K,
    value: *mut V,
}

impl<'a, K, V> DataStoreValueMut<'a, K, V>
where
    K: DataKey,
    V: DataValue,
{
    pub fn new(guard: MutexGuard<'a, DenseSlotMap<K::LocalKey, V>>, key: K, value: *mut V) -> Self {
        Self {
            _guard: guard,

            key,
            value,
        }
    }

    pub fn key(&self) -> K {
        self.key
    }

    pub fn value_ref(&self) -> &V {
        self.deref()
    }

    pub fn value_mut(&mut self) -> &mut V {
        self.deref_mut()
    }
}

impl<'a, K, V> std::ops::Deref for DataStoreValueMut<'a, K, V>
where
    K: DataKey,
    V: DataValue,
{
    type Target = V;

    fn deref(&self) -> &V {
        unsafe { &*self.value }
    }
}

impl<'a, K, V> std::ops::DerefMut for DataStoreValueMut<'a, K, V>
where
    K: DataKey,
    V: DataValue,
{
    fn deref_mut(&mut self) -> &mut V {
        unsafe { &mut *self.value }
    }
}
