use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::sync::Arc;

use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use tokio::sync::oneshot::Receiver;
use tokio::sync::oneshot::channel;
use tracing::Span;

use crate::Error;
use crate::Result;
use crate::channel::Shutdown;

/// Runs synchronous CPU tasks on a shared, fixed-size Rayon pool.
#[derive(Clone)]
pub struct SyncTaskPool {
    pool: Arc<ThreadPool>,
    shutdown: Shutdown,
    span: Span,
}

impl SyncTaskPool {
    pub fn new(shutdown: Shutdown, num_runners: usize) -> Result<Self> {
        assert!(0 < num_runners);

        let span = tracing::info_span!("sync-task-pool", num_runners);
        let shutdown_on_panic = shutdown.clone();
        let panic_span = span.clone();
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_runners)
            .thread_name(|runner_id| format!("sync-task-runner-{runner_id}"))
            // Rayon otherwise aborts the process for a panic outside the task guard.
            .panic_handler(move |_| {
                panic_span.in_scope(|| {
                    tracing::error!("sync task runner panicked");
                    shutdown_on_panic.shutdown();
                });
            })
            .build()
            .map_err(|err| crate::log_err_unavailable!("failed to create sync task pool: {err}"))?;

        Ok(Self {
            pool: Arc::new(pool),
            shutdown,
            span,
        })
    }

    /// Submits `task` immediately and returns its oneshot receiver.
    ///
    /// Use `try_recv`, `blocking_recv`, or `.await` to receive the task result.
    /// Workers skip tasks whose receiver is already dropped.
    /// Shutdown skips queued tasks with `Error::Unavailable`. Running tasks finish normally.
    /// A caught task panic returns `Error::Internal` without triggering shutdown.
    pub fn spawn<F, T>(&self, task: F) -> Receiver<Result<T>>
    where
        F: FnOnce() -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let (output_tx, output_rx) = channel();
        if self.shutdown.is_shutdown() {
            let _ = output_tx.send(Err(Error::unavailable("sync task pool is shut down")));
            return output_rx;
        }

        let shutdown = self.shutdown.clone();
        let span = self.span.clone();
        self.pool.spawn(move || {
            span.in_scope(|| {
                if output_tx.is_closed() {
                    return;
                }

                let unwind_result = catch_unwind(AssertUnwindSafe(|| {
                    if shutdown.is_shutdown() {
                        return Err(Error::unavailable("sync task pool is shut down"));
                    }
                    task()
                }));
                let result = match unwind_result {
                    Ok(result) => result,
                    Err(payload) => {
                        let message = if let Some(message) = payload.downcast_ref::<&str>() {
                            *message
                        } else if let Some(message) = payload.downcast_ref::<String>() {
                            message.as_str()
                        } else {
                            "non-string panic payload"
                        };
                        Err(crate::log_err_internal!("sync task panicked: {message}"))
                    },
                };
                // A dropped receiver discards the result.
                let _ = output_tx.send(result);
            });
        });

        output_rx
    }
}

#[cfg(test)]
#[path = "sync_task_pool_test.rs"]
mod tests;
