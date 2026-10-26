use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use async_channel::Receiver as AsyncReceiver;
use async_channel::Sender as AsyncSender;
use async_channel::unbounded as async_unbounded;
use crossbeam_channel::Sender as SyncSender;
use crossbeam_channel::TrySendError as SyncTrySendError;
use inference_runtime_core::Error;
use inference_runtime_core::Result;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::channel::ShutdownGuard;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::runtime::AtomicRequestStatus;
use inference_runtime_core::runtime::CompletionReason;
use inference_runtime_core::runtime::ExternalRequest;
use inference_runtime_core::runtime::InternalRequest;
use inference_runtime_core::runtime::RawRequestID;
use inference_runtime_core::runtime::RequestEvent;
use inference_runtime_core::runtime::RequestSlotAllocationResult;
use inference_runtime_core::runtime::RequestSlotAllocator;
use inference_runtime_core::runtime::RequestStatus;
use inference_runtime_core::runtime::Resource;
use inference_runtime_core::runtime::ResourcePlacement;
use inference_runtime_core::runtime::Token;
use inference_runtime_core::runtime::decoder::trie_cache::TrieDecoderBlocks;
use inference_runtime_core::runtime::resource::processor::ResourceProcessors;
use tokio::sync::oneshot;

use super::RuntimeBlockCache;
use super::RuntimeRequest;

struct PendingRequest<const N: usize, const L: usize, const P: usize> {
    request: RuntimeRequest<N, P, L>,
    idle_since: Instant,
}

enum SessionCommand {
    Resume {
        request_id: RawRequestID,
        prompt_tokens: Vec<Token>,
        sampling_config: SamplingConfig,
        result_tx: oneshot::Sender<Result<()>>,
    },
    EvictOne {
        result_tx: oneshot::Sender<bool>,
    },
    EvictExpired {
        max_idle: Duration,
        result_tx: oneshot::Sender<usize>,
    },
}

pub struct Sessions<const N: usize, const L: usize, const P: usize> {
    context_window: usize,
    block_cache: Arc<RuntimeBlockCache<P, L>>,
    resource_processors: Arc<ResourceProcessors>,
    request_slot_allocator: RequestSlotAllocator,
    new_req_tx: SyncSender<RuntimeRequest<N, P, L>>,
    command_tx: AsyncSender<SessionCommand>,
    cancel_tx: AsyncSender<RawRequestID>,
}

struct SessionActor<const N: usize, const L: usize, const P: usize> {
    pending: HashMap<RawRequestID, PendingRequest<N, L, P>>,
    context_window: usize,
    new_req_tx: SyncSender<RuntimeRequest<N, P, L>>,
    completed_turn_rx: AsyncReceiver<(RuntimeRequest<N, P, L>, CompletionReason)>,
    command_rx: AsyncReceiver<SessionCommand>,
    cancel_rx: AsyncReceiver<RawRequestID>,
    shutdown: Shutdown,
}

impl<const N: usize, const L: usize, const P: usize> Sessions<N, L, P> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context_window: usize,
        block_cache: Arc<RuntimeBlockCache<P, L>>,
        resource_processors: Arc<ResourceProcessors>,
        request_slot_allocator: RequestSlotAllocator,
        new_req_tx: SyncSender<RuntimeRequest<N, P, L>>,
        completed_turn_rx: AsyncReceiver<(RuntimeRequest<N, P, L>, CompletionReason)>,
        shutdown: Shutdown,
        async_task_handle: &tokio::runtime::Handle,
    ) -> Self {
        let (command_tx, command_rx) = async_unbounded();
        let (cancel_tx, cancel_rx) = async_unbounded();
        let actor = SessionActor {
            pending: HashMap::new(),
            context_window,
            new_req_tx: new_req_tx.clone(),
            completed_turn_rx,
            command_rx,
            cancel_rx,
            shutdown: shutdown.clone(),
        };
        let event_loop = async_task_handle.spawn(async move {
            let _shutdown_guard = ShutdownGuard::new(actor.shutdown.clone());
            actor.run().await;
        });
        drop(event_loop);
        Self {
            context_window,
            block_cache,
            resource_processors,
            request_slot_allocator,
            new_req_tx,
            command_tx,
            cancel_tx,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        &self,
        request_id: RawRequestID,
        history_tokens: Vec<Token>,
        prompt_tokens: Vec<Token>,
        sampled_tokens: Vec<Token>,
        resource_entries: Vec<(Resource, ResourcePlacement)>,
        sampling_config: SamplingConfig,
    ) -> Result<ExternalRequest> {
        let (request, external_request) = self.initialize(
            request_id,
            history_tokens,
            prompt_tokens,
            sampled_tokens,
            resource_entries,
            sampling_config,
        )?;
        submit_request(&self.new_req_tx, request)?;
        Ok(external_request)
    }

    pub async fn resume(
        &self,
        external_request: &ExternalRequest,
        prompt_tokens: Vec<Token>,
        sampling_config: SamplingConfig,
    ) -> Result<()> {
        let request_id = external_request.req_id();
        match external_request.status() {
            RequestStatus::Running | RequestStatus::Swapped => {},
            status if status.is_terminal() => return Err(status_error(status)),
            status => panic!("decode session cannot resume from request status {status:?}"),
        }
        let (result_tx, result_rx) = oneshot::channel();
        self.command_tx
            .send(SessionCommand::Resume {
                request_id,
                prompt_tokens,
                sampling_config,
                result_tx,
            })
            .await
            .map_err(|_| Error::unavailable("runtime is stopped"))?;
        result_rx.await.map_err(|_| Error::unavailable("runtime is stopped"))?
    }

    pub async fn evict_one_req(&self) -> Result<bool> {
        let (result_tx, result_rx) = oneshot::channel();
        self.command_tx
            .send(SessionCommand::EvictOne { result_tx })
            .await
            .map_err(|_| Error::unavailable("runtime is stopped"))?;
        result_rx.await.map_err(|_| Error::unavailable("runtime is stopped"))
    }

    pub async fn evict_expired(&self, max_idle: Duration) -> Result<usize> {
        let (result_tx, result_rx) = oneshot::channel();
        self.command_tx
            .send(SessionCommand::EvictExpired { max_idle, result_tx })
            .await
            .map_err(|_| Error::unavailable("runtime is stopped"))?;
        result_rx.await.map_err(|_| Error::unavailable("runtime is stopped"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn initialize(
        &self,
        request_id: RawRequestID,
        history_tokens: Vec<Token>,
        prompt_tokens: Vec<Token>,
        sampled_tokens: Vec<Token>,
        resource_entries: Vec<(Resource, ResourcePlacement)>,
        sampling_config: SamplingConfig,
    ) -> Result<(InternalRequest<N, P, L, RuntimeBlockCache<P, L>>, ExternalRequest)> {
        let (resources, resource_placements): (Vec<_>, Vec<_>) = resource_entries.into_iter().unzip();
        let num_initial_tokens = history_tokens.len() + prompt_tokens.len() + sampled_tokens.len();
        let min_initial_tokens = usize::max(1, L - 1);
        if num_initial_tokens < min_initial_tokens {
            return Err(Error::invalid_argument(format!(
                "decode request minimum initial token count is {min_initial_tokens} for {L} cache lanes, got {}",
                num_initial_tokens
            )));
        }
        if num_initial_tokens >= self.context_window {
            return Err(Error::invalid_argument(format!(
                "decode request initial token count={num_initial_tokens} must be less than context window={}",
                self.context_window
            )));
        }
        if sampled_tokens.len() >= sampling_config.max_sampled_tokens {
            return Err(Error::invalid_argument(format!(
                "decode request initial sampled token count={} must be less than max_sampled_tokens={}",
                sampled_tokens.len(),
                sampling_config.max_sampled_tokens
            )));
        }
        inference_runtime_core::runtime::validate_resources(&resources, &resource_placements, num_initial_tokens)?;
        let request_slot = match self.request_slot_allocator.allocate() {
            RequestSlotAllocationResult::Ok { request_slot } => request_slot,
            RequestSlotAllocationResult::ResourceLimitExceeded => {
                return Err(Error::unavailable("request slot capacity is exhausted"));
            },
        };
        let req_status = AtomicRequestStatus::new();
        let decoder_blocks = TrieDecoderBlocks::new(
            self.block_cache.clone(),
            resources,
            resource_placements,
            history_tokens,
            prompt_tokens,
            sampled_tokens,
        );
        let (event_tx, event_rx) = async_unbounded::<RequestEvent>();
        let request = RuntimeRequest::new(
            request_id,
            request_slot,
            req_status.clone(),
            decoder_blocks,
            self.resource_processors.clone(),
            event_tx,
            sampling_config,
            self.context_window,
        );
        assert!(
            request.store_running(),
            "a new internal request must enter the running state"
        );
        let external_request = ExternalRequest::new(request_id, req_status, event_rx, self.cancel_tx.clone());
        Ok((request, external_request))
    }
}

impl<const N: usize, const L: usize, const P: usize> SessionActor<N, L, P> {
    async fn run(mut self) {
        let shutdown_rx = self.shutdown.async_rx().clone();
        loop {
            tokio::select! {
                completed = self.completed_turn_rx.recv() => {
                    let Ok((request, reason)) = completed else {
                        break;
                    };
                    self.complete(request, reason);
                },
                command = self.command_rx.recv() => {
                    let Ok(command) = command else {
                        break;
                    };
                    self.handle(command);
                },
                cancelled = self.cancel_rx.recv() => {
                    let Ok(request_id) = cancelled else {
                        break;
                    };
                    self.pending.remove(&request_id);
                },
                _ = shutdown_rx.recv() => break,
            }
        }
        self.shutdown.shutdown();
    }

    fn complete(&mut self, request: RuntimeRequest<N, P, L>, reason: CompletionReason) {
        if request.status().is_terminal() {
            return;
        }
        let request_id = request.req_id();
        let Entry::Vacant(entry) = self.pending.entry(request_id) else {
            panic!("a decode session cannot own two pending requests")
        };
        entry
            .insert(PendingRequest {
                request,
                idle_since: Instant::now(),
            })
            .request
            .send_turn_completed(reason);
    }

    fn handle(&mut self, command: SessionCommand) {
        match command {
            SessionCommand::Resume {
                request_id,
                prompt_tokens,
                sampling_config,
                result_tx,
            } => {
                let result = self.resume(request_id, prompt_tokens, sampling_config);
                let _ = result_tx.send(result);
            },
            SessionCommand::EvictOne { result_tx } => {
                let _ = result_tx.send(evict_one(&mut self.pending));
            },
            SessionCommand::EvictExpired { max_idle, result_tx } => {
                let _ = result_tx.send(evict_expired(&mut self.pending, max_idle));
            },
        }
    }

    fn resume(
        &mut self,
        request_id: RawRequestID,
        prompt_tokens: Vec<Token>,
        sampling_config: SamplingConfig,
    ) -> Result<()> {
        let Some(pending_request) = self.pending.get_mut(&request_id) else {
            return Err(Error::evicted("decode session was evicted"));
        };
        let request = &mut pending_request.request;
        match request.status() {
            RequestStatus::Running => {},
            status if status.is_terminal() => return Err(status_error(status)),
            status => panic!("decode session cannot resume from request status {status:?}"),
        }

        // TODO(session-residency): Restore a partially evicted or SSD-offloaded request here before the next turn
        // starts.
        let num_session_tokens = request.num_total_tokens() + prompt_tokens.len();
        if num_session_tokens >= self.context_window {
            return Err(Error::invalid_argument(format!(
                "decode session token count={num_session_tokens} must be less than context window={}",
                self.context_window
            )));
        }
        request.start_turn(prompt_tokens, sampling_config);
        let request = self
            .pending
            .remove(&request_id)
            .expect("resumed session must remain pending until ownership transfer")
            .request;
        submit_request(&self.new_req_tx, request)
    }
}

fn submit_request<const N: usize, const L: usize, const P: usize>(
    new_req_tx: &SyncSender<RuntimeRequest<N, P, L>>,
    request: RuntimeRequest<N, P, L>,
) -> Result<()> {
    let request_id = request.req_id();
    match new_req_tx.try_send(request) {
        Ok(()) => {
            tracing::debug!(
                target: "inference-runtime-service::runtime",
                phase = "request.queued",
                request_id,
                "decode request queued"
            );
            Ok(())
        },
        Err(SyncTrySendError::Full(_)) => {
            unreachable!("request queue capacity must cover all request slots")
        },
        Err(SyncTrySendError::Disconnected(_)) => {
            tracing::debug!(
                target: "inference-runtime-service::runtime",
                phase = "request.runtime_stopped",
                request_id,
                "runtime is stopped"
            );
            Err(Error::unavailable("runtime is stopped"))
        },
    }
}

fn evict_one<const N: usize, const L: usize, const P: usize>(
    pending: &mut HashMap<RawRequestID, PendingRequest<N, L, P>>,
) -> bool {
    // TODO(session-eviction-index): Replace the linear scan when pending-session scale requires an index.
    let Some(request_id) = pending
        .iter()
        .min_by_key(|(request_id, pending_request)| (pending_request.idle_since, **request_id))
        .map(|(request_id, _)| *request_id)
    else {
        return false;
    };
    let evicted = pending
        .remove(&request_id)
        .expect("selected pending request must remain present");
    evicted.request.store_evicted();
    tracing::debug!(
        target: "inference-runtime-service::runtime",
        phase = "request.evicted",
        request_id,
        "idle decode session evicted"
    );
    true
}

fn evict_expired<const N: usize, const L: usize, const P: usize>(
    pending: &mut HashMap<RawRequestID, PendingRequest<N, L, P>>,
    max_idle: Duration,
) -> usize {
    let now = Instant::now();
    let mut num_evicted = 0;
    pending.retain(|request_id, pending_request| {
        if now.duration_since(pending_request.idle_since) < max_idle {
            return true;
        }
        pending_request.request.store_evicted();
        num_evicted += 1;
        tracing::debug!(
            target: "inference-runtime-service::runtime",
            phase = "request.evicted",
            request_id,
            "expired decode session evicted"
        );
        false
    });
    num_evicted
}

fn status_error(status: RequestStatus) -> Error {
    match status {
        RequestStatus::Cancelled => Error::cancelled("decode session was cancelled"),
        RequestStatus::TimedOut => Error::deadline_exceeded("decode session timed out"),
        RequestStatus::Aborted => Error::aborted("decode session was aborted"),
        RequestStatus::Evicted => Error::evicted("decode session was evicted"),
        RequestStatus::Completed(_) => Error::invalid_argument("decode session has terminal completion"),
        RequestStatus::Initialized | RequestStatus::Running | RequestStatus::Swapped => {
            panic!("active request status cannot map to a terminal session error")
        },
    }
}

#[cfg(test)]
#[path = "./session_test.rs"]
mod session_test;
