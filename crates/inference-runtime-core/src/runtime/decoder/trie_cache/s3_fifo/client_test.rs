use std::sync::Arc;

use super::client::S3FIFOClient;
use crate::channel::Shutdown;
use crate::runtime::decoder::trie_cache::EvictionEntry;
use crate::runtime::decoder::trie_cache::EvictionPolicy;

#[test]
fn test_shutdown() {
    let shutdown = Shutdown::new();
    let client = S3FIFOClient::<u8>::new(1, shutdown.clone());
    let eviction = client.new_entry(1);
    let weak_eviction = Arc::downgrade(&eviction);
    let late_eviction = client.new_entry(2);
    eviction.clone().register();
    eviction.unpin();
    drop(eviction);
    shutdown.shutdown();
    drop(client);
    late_eviction.register();
    assert!(weak_eviction.upgrade().is_none());
}

#[test]
fn test_register_is_fire_and_forget() {
    let client = S3FIFOClient::<u8>::new(1, Shutdown::new());
    let eviction = client.new_entry(1);
    eviction.clone().register();
    eviction.unpin();

    assert_eq!(Some(1), client.evict_candidate());
    client.accept_candidate(1);
}
