use std::time::Duration;

use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::Sender;
use crossbeam_channel::bounded;
use futures_lite::future::block_on;
use tokio::sync::oneshot::error::TryRecvError;

use super::*;

#[test]
fn test_spawn_success() {
    let shutdown = Shutdown::new();
    let pool = SyncTaskPool::new(shutdown.clone(), 2).unwrap();
    let (mut first, first_release) = new_running_task(&pool, vec![11, 12]);
    let (second, second_release) = new_running_task(&pool, vec![21]);
    let error = pool.spawn(|| Err::<(), _>(Error::invalid_argument("task input")));

    assert!(matches!(first.try_recv(), Err(TryRecvError::Empty)));

    // Both workers started before receiving their results. Submitted work outlives the pool handle.
    drop(pool);
    first_release.send(()).unwrap();
    second_release.send(()).unwrap();
    assert_eq!(first.blocking_recv().unwrap().unwrap(), vec![11, 12]);
    assert_eq!(block_on(second).unwrap().unwrap(), vec![21]);
    assert!(matches!(error.blocking_recv().unwrap(), Err(Error::InvalidArgument(message)) if message == "task input"));
    assert!(!shutdown.is_shutdown());
}

#[test]
fn test_spawn_cancelled() {
    let pool = SyncTaskPool::new(Shutdown::new(), 1).unwrap();
    let (running, release) = new_running_task(&pool, 7);
    let (executed_tx, executed_rx) = bounded(1);
    let queued = pool.spawn(move || {
        executed_tx.send(()).unwrap();
        Ok(())
    });

    drop(queued);
    release.send(()).unwrap();
    assert!(matches!(
        executed_rx.recv_timeout(Duration::from_secs(1)),
        Err(RecvTimeoutError::Disconnected)
    ));
    assert_eq!(running.blocking_recv().unwrap().unwrap(), 7);
}

#[test]
fn test_spawn_shutdown() {
    let shutdown = Shutdown::new();
    let pool = SyncTaskPool::new(shutdown.clone(), 1).unwrap();
    let (running, release) = new_running_task(&pool, 7);
    let queued = pool.spawn(|| -> Result<()> { unreachable!("queued task must not run after shutdown") });

    shutdown.shutdown();
    let mut rejected = pool.spawn(|| -> Result<()> { unreachable!("new task must not run after shutdown") });
    assert!(matches!(rejected.try_recv().unwrap(), Err(Error::Unavailable(_))));
    release.send(()).unwrap();
    assert_eq!(running.blocking_recv().unwrap().unwrap(), 7);
    assert!(matches!(queued.blocking_recv().unwrap(), Err(Error::Unavailable(_))));
}

#[test]
fn test_spawn_panic() {
    let shutdown = Shutdown::new();
    let pool = SyncTaskPool::new(shutdown.clone(), 1).unwrap();
    let receiver = pool.spawn(|| -> Result<()> { panic!("sync task test panic") });

    assert!(matches!(
        receiver.blocking_recv().unwrap(),
        Err(Error::Internal(message)) if message == "sync task panicked: sync task test panic"
    ));
    assert!(!shutdown.is_shutdown());
    assert_eq!(pool.spawn(|| Ok(7)).blocking_recv().unwrap().unwrap(), 7);
}

fn new_running_task<T: Send + 'static>(pool: &SyncTaskPool, output: T) -> (Receiver<Result<T>>, Sender<()>) {
    let (started_tx, started_rx) = bounded(1);
    let (release_tx, release_rx) = bounded(1);
    let receiver = pool.spawn(move || {
        started_tx.send(()).unwrap();
        release_rx.recv().unwrap();
        Ok(output)
    });
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    (receiver, release_tx)
}
