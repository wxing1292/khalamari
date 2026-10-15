use std::sync::Arc;

use ahash::RandomState as AHashRandomState;
use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use dashmap::DashMap;

use super::control::Control;
use super::entry::S3FIFOEntry;
use super::queue::Queue;
use super::server::S3FIFOServer;
use crate::channel::Shutdown;
use crate::runtime::decoder::trie_cache::EvictionEntry;

type Key = u8;

type Evictions = Arc<DashMap<Key, Arc<S3FIFOEntry<Key>>, AHashRandomState>>;

struct ServerHarness {
    server: S3FIFOServer<Key>,
    evictions: Evictions,
    sender: Sender<Control<Key>>,
    receiver: Receiver<Control<Key>>,
}

fn new_server() -> ServerHarness {
    let (sender, receiver) = crossbeam_channel::unbounded();
    let evictions = Arc::new(DashMap::with_hasher(AHashRandomState::new()));
    ServerHarness {
        server: S3FIFOServer::new(evictions.clone(), Shutdown::new()),
        evictions,
        sender,
        receiver,
    }
}

fn insert(
    server: &mut S3FIFOServer<Key>,
    evictions: &Evictions,
    sender: &Sender<Control<Key>>,
    key: Key,
) -> Arc<S3FIFOEntry<Key>> {
    let eviction = Arc::new(S3FIFOEntry::new(
        key,
        Arc::downgrade(evictions),
        sender.clone(),
        Shutdown::new(),
    ));
    assert!(evictions.insert(key, eviction.clone()).is_none());
    server.re_evaluate(key);
    eviction
}

fn re_evaluate(server: &mut S3FIFOServer<Key>, receiver: &Receiver<Control<Key>>, key: Key) {
    let Control::Reevaluate { key: signaled_key } = receiver
        .recv()
        .expect("re_evaluate: eviction operation must signal the server")
    else {
        panic!("re_evaluate: expected a Reevaluate signal");
    };
    assert_eq!(
        key, signaled_key,
        "re_evaluate: signal must identify the eviction entry"
    );
    server.re_evaluate(key);
}

fn unpin(server: &mut S3FIFOServer<Key>, receiver: &Receiver<Control<Key>>, eviction: &S3FIFOEntry<Key>) {
    eviction.unpin();
    re_evaluate(server, receiver, eviction.key());
}

fn promote_to_m(server: &mut S3FIFOServer<Key>, receiver: &Receiver<Control<Key>>, eviction: &S3FIFOEntry<Key>) {
    unpin(server, receiver, eviction);
    eviction.touch();
    assert_eq!(None, server.next_candidate(Queue::S));
    assert_eq!(Queue::M, server.entry(eviction.key()).unwrap().queue.get());
    assert_eq!(1, eviction.count());
}

#[test]
fn test_insert_remove() {
    let ServerHarness {
        mut server,
        evictions,
        sender,
        ..
    } = new_server();
    let eviction = insert(&mut server, &evictions, &sender, 1);
    assert!(server.entry(1).is_some());
    assert!(eviction.is_pinned());
    assert_eq!(1, eviction.count());

    server.remove(1);
    assert!(server.entry(1).is_none());
    assert!(!evictions.contains_key(&1));
    server.remove(1);
}

#[test]
fn test_insert_inc_dec_count_remove() {
    let ServerHarness {
        mut server,
        evictions,
        sender,
        ..
    } = new_server();
    let eviction = insert(&mut server, &evictions, &sender, 1);
    eviction.touch();
    eviction.touch();
    assert_eq!(3, eviction.count());
    assert!(eviction.dec_count());
    assert!(eviction.dec_count());
    assert_eq!(1, eviction.count());

    server.remove(1);
    assert!(server.entry(1).is_none());
}

#[test]
fn test_insert_pin_unpin_remove() {
    let ServerHarness {
        mut server,
        evictions,
        sender,
        receiver,
    } = new_server();
    let eviction = insert(&mut server, &evictions, &sender, 1);
    unpin(&mut server, &receiver, &eviction);
    assert!(server.entry(1).unwrap().queue_link.is_linked());

    eviction.pin();
    re_evaluate(&mut server, &receiver, 1);
    assert!(!server.entry(1).unwrap().queue_link.is_linked());

    server.remove(1);
    assert!(server.entry(1).is_none());
}

#[test]
fn test_evict_candidate_s_pin_count_1() {
    let ServerHarness {
        mut server,
        evictions,
        sender,
        receiver,
    } = new_server();
    let eviction = insert(&mut server, &evictions, &sender, 1);
    unpin(&mut server, &receiver, &eviction);
    eviction.pin();

    assert_eq!(None, server.next_candidate(Queue::S));
    let entry = server.entry(1).unwrap();
    assert!(!entry.queue_link.is_linked());
    assert_eq!(Queue::S, entry.queue.get());
    assert_eq!(1, eviction.count());
}

#[test]
fn test_evict_candidate_s_pin_count_2() {
    let ServerHarness {
        mut server,
        evictions,
        sender,
        receiver,
    } = new_server();
    let eviction = insert(&mut server, &evictions, &sender, 1);
    unpin(&mut server, &receiver, &eviction);
    eviction.touch();
    eviction.pin();

    assert_eq!(None, server.next_candidate(Queue::S));
    let entry = server.entry(1).unwrap();
    assert!(!entry.queue_link.is_linked());
    assert_eq!(Queue::M, entry.queue.get());
    assert_eq!(1, eviction.count());
}

#[test]
fn test_evict_candidate_s_unpin_count_1() {
    let ServerHarness {
        mut server,
        evictions,
        sender,
        receiver,
    } = new_server();
    let eviction = insert(&mut server, &evictions, &sender, 1);
    unpin(&mut server, &receiver, &eviction);

    assert_eq!(Some(1), server.next_candidate(Queue::S));
    assert!(server.entry(1).unwrap().is_candidate.get());
}

#[test]
fn test_evict_candidate_s_unpin_count_2() {
    let ServerHarness {
        mut server,
        evictions,
        sender,
        receiver,
    } = new_server();
    let eviction = insert(&mut server, &evictions, &sender, 1);
    unpin(&mut server, &receiver, &eviction);
    eviction.touch();

    assert_eq!(None, server.next_candidate(Queue::S));
    assert_eq!(Some(1), server.next_candidate(Queue::M));
}

#[test]
fn test_evict_candidate_m_pin_count_1() {
    let ServerHarness {
        mut server,
        evictions,
        sender,
        receiver,
    } = new_server();
    let eviction = insert(&mut server, &evictions, &sender, 1);
    promote_to_m(&mut server, &receiver, &eviction);
    eviction.pin();

    assert_eq!(None, server.next_candidate(Queue::M));
    assert!(!server.entry(1).unwrap().queue_link.is_linked());
    assert_eq!(1, eviction.count());
}

#[test]
fn test_evict_candidate_m_pin_count_2() {
    let ServerHarness {
        mut server,
        evictions,
        sender,
        receiver,
    } = new_server();
    let eviction = insert(&mut server, &evictions, &sender, 1);
    promote_to_m(&mut server, &receiver, &eviction);
    eviction.touch();
    eviction.pin();

    assert_eq!(None, server.next_candidate(Queue::M));
    assert!(!server.entry(1).unwrap().queue_link.is_linked());
    assert_eq!(1, eviction.count());
}

#[test]
fn test_evict_candidate_m_unpin_count_1() {
    let ServerHarness {
        mut server,
        evictions,
        sender,
        receiver,
    } = new_server();
    let eviction = insert(&mut server, &evictions, &sender, 1);
    promote_to_m(&mut server, &receiver, &eviction);

    assert_eq!(Some(1), server.next_candidate(Queue::M));
    assert!(server.entry(1).unwrap().is_candidate.get());
}

#[test]
fn test_evict_candidate_m_unpin_count_2() {
    let ServerHarness {
        mut server,
        evictions,
        sender,
        receiver,
    } = new_server();
    let eviction = insert(&mut server, &evictions, &sender, 1);
    promote_to_m(&mut server, &receiver, &eviction);
    eviction.touch();

    assert_eq!(Some(1), server.next_candidate(Queue::M));
}

#[test]
fn test_evict_candidate_s() {
    let ServerHarness {
        mut server,
        evictions,
        sender,
        receiver,
    } = new_server();
    let eviction = insert(&mut server, &evictions, &sender, 1);
    unpin(&mut server, &receiver, &eviction);

    assert_eq!(Some(1), server.evict_candidate(0, 0));
}

#[test]
fn test_evict_candidate_m() {
    let ServerHarness {
        mut server,
        evictions,
        sender,
        receiver,
    } = new_server();
    let eviction = insert(&mut server, &evictions, &sender, 1);
    promote_to_m(&mut server, &receiver, &eviction);

    assert_eq!(Some(1), server.evict_candidate(0, 0));
}

#[test]
fn test_evict_reject() {
    let ServerHarness {
        mut server,
        evictions,
        sender,
        receiver,
    } = new_server();
    let eviction = insert(&mut server, &evictions, &sender, 1);
    unpin(&mut server, &receiver, &eviction);
    assert_eq!(Some(1), server.next_candidate(Queue::S));

    server.reject_candidate(1);
    assert!(server.entry(1).unwrap().queue_link.is_linked());

    assert_eq!(Some(1), server.next_candidate(Queue::S));
    eviction.pin();
    server.reject_candidate(1);
    assert!(!server.entry(1).unwrap().queue_link.is_linked());
}

#[test]
fn test_evict_accept() {
    let ServerHarness {
        mut server,
        evictions,
        sender,
        receiver,
    } = new_server();
    let eviction = insert(&mut server, &evictions, &sender, 1);
    unpin(&mut server, &receiver, &eviction);
    assert_eq!(Some(1), server.next_candidate(Queue::S));

    server.accept_candidate(1);
    assert!(server.entry(1).is_none());
    assert!(!evictions.contains_key(&1));
}
