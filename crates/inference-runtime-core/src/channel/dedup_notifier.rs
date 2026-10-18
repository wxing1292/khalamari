use std::collections::HashSet;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::Mutex;

use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use crossbeam_channel::TrySendError;
use crossbeam_channel::bounded;

use crate::channel::Shutdown;

#[derive(Debug)]
pub struct DedupNotifier<T> {
    wake_tx: Sender<()>,
    pending: Mutex<HashSet<T>>,
    shutdown: Shutdown,
}

impl<T> DedupNotifier<T>
where
    T: Eq + Hash,
{
    pub fn new(shutdown: Shutdown) -> (Arc<Self>, Receiver<()>) {
        let (wake_tx, wake_rx) = bounded(1);
        (
            Arc::new(Self {
                wake_tx,
                pending: Mutex::new(HashSet::new()),
                shutdown,
            }),
            wake_rx,
        )
    }

    pub fn send_one(&self, value: T) {
        let mut pending = self.pending.lock().expect("dedup notifier mutex poisoned");
        if pending.insert(value) {
            self.notify();
        }
    }

    pub fn send_many(&self, values: impl IntoIterator<Item = T>) {
        let mut pending = self.pending.lock().expect("dedup notifier mutex poisoned");
        let mut changed = false;
        for value in values {
            changed |= pending.insert(value);
        }
        if changed {
            self.notify();
        }
    }

    fn notify(&self) {
        match self.wake_tx.try_send(()) {
            Ok(()) | Err(TrySendError::Full(())) => {},
            Err(TrySendError::Disconnected(())) => {
                if self.shutdown.is_shutdown() {
                    return;
                }
                panic!("dedup notifier receiver disconnected");
            },
        }
    }

    pub fn recv_many(&self) -> HashSet<T> {
        std::mem::take(&mut *self.pending.lock().expect("dedup notifier mutex poisoned"))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::DedupNotifier;
    use crate::channel::Shutdown;

    #[test]
    fn test_send_one_recv_many() {
        let (notifier, wake_rx) = DedupNotifier::new(Shutdown::new());

        notifier.send_one(3_u32);

        wake_rx.recv().unwrap();
        assert_eq!(notifier.recv_many(), HashSet::from([3]));
    }

    #[test]
    fn test_send_two_recv_many() {
        let (notifier, wake_rx) = DedupNotifier::new(Shutdown::new());

        notifier.send_one(3_u32);
        notifier.send_one(7);

        wake_rx.recv().unwrap();
        assert_eq!(notifier.recv_many(), HashSet::from([3, 7]));
    }

    #[test]
    fn test_send_many_recv_many() {
        let (notifier, wake_rx) = DedupNotifier::new(Shutdown::new());

        notifier.send_many([3_u32, 7, 3]);

        wake_rx.recv().unwrap();
        assert_eq!(notifier.recv_many(), HashSet::from([3, 7]));
    }
}
