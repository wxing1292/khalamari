use std::cmp::max;
use std::cmp::min;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use std::thread::JoinHandle;

use ahash::RandomState as AHashRandomState;
use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use dashmap::DashMap;

use crate::channel::Shutdown;
use crate::runtime::decoder::trie_cache::EvictionPolicy;
use crate::runtime::decoder::trie_cache::s3_fifo::control::Control;
use crate::runtime::decoder::trie_cache::s3_fifo::entry::S3FIFOEntry;
use crate::runtime::decoder::trie_cache::s3_fifo::server::S3FIFOServer;

#[derive(Debug)]
pub struct S3FIFOClient<K>
where
    K: Copy + Eq + Hash + Send + Sync + 'static,
{
    control_sender: Sender<Control<K>>,
    evictions: Arc<DashMap<K, Arc<S3FIFOEntry<K>>, AHashRandomState>>,
    worker: Option<JoinHandle<()>>,
    shutdown: Shutdown,
}

impl<K> S3FIFOClient<K>
where
    K: Copy + Eq + Hash + Send + Sync + 'static,
{
    pub fn new(capacity: usize, shutdown: Shutdown) -> Self {
        assert!(1 <= capacity, "new: capacity must be positive");
        let s_cap = min(max(1, capacity / 10), capacity);
        let m_cap = capacity - s_cap;
        let (control_sender, control_receiver) = crossbeam_channel::unbounded();
        let evictions = Arc::new(DashMap::with_hasher(AHashRandomState::new()));
        let worker_shutdown = shutdown.clone();
        let worker_evictions = evictions.clone();
        let worker = std::thread::spawn(move || {
            s3_fifo_event_loop(control_receiver, worker_evictions, worker_shutdown, s_cap, m_cap)
        });
        Self {
            control_sender,
            evictions,
            worker: Some(worker),
            shutdown,
        }
    }

    fn send(&self, control: Control<K>) {
        if let Err(error) = self.control_sender.send(control) {
            if self.shutdown.is_shutdown() {
                return;
            }
            panic!("send: S3FIFO worker must be running: {error}");
        }
    }
}

impl<K> EvictionPolicy<K> for S3FIFOClient<K>
where
    K: Copy + Debug + Eq + Hash + Send + Sync + 'static,
{
    type Entry = S3FIFOEntry<K>;

    fn new_entry(&self, key: K) -> Arc<Self::Entry> {
        Arc::new(S3FIFOEntry::new(
            key,
            Arc::downgrade(&self.evictions),
            self.control_sender.clone(),
            self.shutdown.clone(),
        ))
    }

    fn evict_candidate(&self) -> Option<K> {
        let (reply_sender, reply_receiver) = crossbeam_channel::bounded(0);
        self.send(Control::EvictCandidate { reply: reply_sender });
        match reply_receiver.recv() {
            Ok(candidate) => candidate,
            Err(_) if self.shutdown.is_shutdown() => None,
            Err(error) => panic!("evict_candidate: S3FIFO worker must reply: {error}"),
        }
    }

    fn reject_candidate(&self, key: K) {
        let (reply_sender, reply_receiver) = crossbeam_channel::bounded(0);
        self.send(Control::RejectCandidate {
            key,
            reply: reply_sender,
        });
        if let Err(error) = reply_receiver.recv() {
            if self.shutdown.is_shutdown() {
                return;
            }
            panic!("reject_candidate: S3FIFO worker must reply: {error}");
        }
    }

    fn accept_candidate(&self, key: K) {
        let (reply_sender, reply_receiver) = crossbeam_channel::bounded(0);
        self.send(Control::AcceptCandidate {
            key,
            reply: reply_sender,
        });
        if let Err(error) = reply_receiver.recv() {
            if self.shutdown.is_shutdown() {
                return;
            }
            panic!("accept_candidate: S3FIFO worker must reply: {error}");
        }
    }
}

impl<K> Drop for S3FIFOClient<K>
where
    K: Copy + Eq + Hash + Send + Sync + 'static,
{
    fn drop(&mut self) {
        let _ = self.control_sender.send(Control::Shutdown);
        if let Some(worker) = self.worker.take() {
            worker.join().expect("drop: S3FIFO worker must not panic");
        }
    }
}

fn s3_fifo_event_loop<K>(
    control_receiver: Receiver<Control<K>>,
    evictions: Arc<DashMap<K, Arc<S3FIFOEntry<K>>, AHashRandomState>>,
    shutdown: Shutdown,
    s_cap: usize,
    m_cap: usize,
) where
    K: Copy + Eq + Hash + Send + Sync + 'static,
{
    let _span = tracing::info_span!("s3-fifo").entered();
    tracing::info!("started");

    let mut server = S3FIFOServer::new(evictions, shutdown);
    let shutdown_receiver = server.shutdown().sync_rx().clone();
    'event_loop: loop {
        let control = crossbeam_channel::select! {
            recv(shutdown_receiver) -> _ => {
                break 'event_loop;
            },
            recv(control_receiver) -> control => match control {
                Ok(control) => control,
                Err(_) => {
                    tracing::debug!("control channel closed, stopping");
                    break 'event_loop;
                },
            },
        };
        match control {
            Control::Reevaluate { key } => server.re_evaluate(key),
            Control::EvictCandidate { reply } => {
                let _ = reply.send(server.evict_candidate(s_cap, m_cap));
            },
            Control::RejectCandidate { key, reply } => {
                server.reject_candidate(key);
                let _ = reply.send(());
            },
            Control::AcceptCandidate { key, reply } => {
                server.accept_candidate(key);
                let _ = reply.send(());
            },
            Control::Shutdown => {
                tracing::debug!("client dropped, stopping");
                break 'event_loop;
            },
        }
    }

    tracing::info!("stopped");
}
