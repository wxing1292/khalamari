use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

use async_channel::bounded as async_bounded;
use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use crossbeam_channel::bounded as sync_bounded;
use inference_executor_core::model::ReplayableDecoderModel;
use inference_runtime_core::Result;
use inference_runtime_core::channel::DedupNotifier;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::channel::ShutdownGuard;
use inference_runtime_core::compute::DeviceRequest;
use inference_runtime_core::compute::DeviceResponse;
use inference_runtime_core::compute::ReplayableModelExecutorRequest;
use inference_runtime_core::compute::ReplayableModelExecutorResponse;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::config::SchedulerConfig;
use inference_runtime_core::log_err_internal;
use inference_runtime_core::log_err_unavailable;
use inference_runtime_core::memory::U32IDAllocator;
use inference_runtime_core::runtime::ExternalRequest;
use inference_runtime_core::runtime::InternalRequest;
use inference_runtime_core::runtime::RawRequestID;
use inference_runtime_core::runtime::RawRequestSlot;
use inference_runtime_core::runtime::RequestSlotAllocator;
use inference_runtime_core::runtime::Resource;
use inference_runtime_core::runtime::ResourcePlacement;
use inference_runtime_core::runtime::Token;
use inference_runtime_core::runtime::decoder::TPKVBlockAllocator;
use inference_runtime_core::runtime::decoder::TPStateBlockAllocator;
use inference_runtime_core::runtime::decoder::trie_cache::MultiLaneTrieBlockCache;
use inference_runtime_core::runtime::decoder::trie_cache::SingleLaneTrieBlockCache;
use inference_runtime_core::runtime::resource::processor::ResourceProcessors;
use inference_runtime_core::runtime::scheduler::EventLoop;
use inference_runtime_core::runtime::scheduler::FIFOBatcher;
use inference_runtime_core::runtime::scheduler::InstrumentedScheduler;
use inference_runtime_core::runtime::scheduler::ScheduleQueue;
use inference_runtime_core::runtime::scheduler::SimpleScheduler;
use inference_runtime_core::runtime::tasks::AsyncTaskPool;
use inference_runtime_core::runtime::tasks::AsyncTaskResp;

use crate::api::Inference;
use crate::consts::NUM_TRIE_PARTITION;
use crate::executor::ReplayableDecoderModelEventLoop;
use crate::executor::ReplayableModelExecutors;
use crate::rpc;
use crate::rpc::RPCService;

mod session;
use session::Sessions;

type RuntimeBlockCache<const P: usize, const L: usize> =
    MultiLaneTrieBlockCache<P, L, TPKVBlockAllocator, TPStateBlockAllocator>;
type RuntimeRequest<const N: usize, const P: usize, const L: usize> = InternalRequest<N, P, L, RuntimeBlockCache<P, L>>;

pub struct InferenceRuntime<const N: usize, const L: usize, const P: usize> {
    #[cfg(test)]
    model_runtime_config: RuntimeConfig,

    shutdown: Shutdown,

    sessions: Sessions<N, L, P>,
    model_executor_req_rx: Receiver<ReplayableModelExecutorRequest>,
    model_executor_resp_tx: Sender<ReplayableModelExecutorResponse>,
    request_slot_reset_notifier: Arc<DedupNotifier<RawRequestSlot>>,
    request_slot_reset_rx: Receiver<()>,
}

impl<const N: usize, const L: usize, const P: usize> InferenceRuntime<N, L, P> {
    pub fn new(
        model_runtime_config: RuntimeConfig,
        scheduler_config: SchedulerConfig,
        num_spec_tokens: usize,
        shutdown: Shutdown,
        async_task_handle: &tokio::runtime::Handle,
        resource_processors: Arc<ResourceProcessors>,
    ) -> Self {
        assert!(scheduler_config.max_requests > 0, "runtime requires request capacity");
        assert!(scheduler_config.max_tokens > 0, "runtime requires token capacity");
        assert!(
            scheduler_config.max_tokens_per_request > 0,
            "runtime requires per-request token capacity"
        );
        assert!(
            scheduler_config.max_tokens_per_request <= scheduler_config.max_tokens,
            "runtime per-request token capacity={} exceeds batch token capacity={}",
            scheduler_config.max_tokens_per_request,
            scheduler_config.max_tokens
        );
        assert!(scheduler_config.max_compute_slots > 0, "runtime requires compute slots");
        assert!(
            model_runtime_config.max_running_requests > 0,
            "runtime requires running-request capacity"
        );
        let min_initial_tokens = usize::max(1, L - 1);
        assert!(
            model_runtime_config.context_window > min_initial_tokens,
            "runtime context window={} must exceed the minimum initial token count={min_initial_tokens}",
            model_runtime_config.context_window
        );
        assert!(
            u32::try_from(model_runtime_config.context_window).is_ok(),
            "runtime context window must fit u32"
        );
        assert!(
            scheduler_config.max_requests <= model_runtime_config.max_running_requests,
            "scheduler batch request capacity={} exceeds runtime running-request capacity={}",
            scheduler_config.max_requests,
            model_runtime_config.max_running_requests
        );
        assert_eq!(
            N,
            model_runtime_config.num_tokens_per_cache_block(),
            "runtime service compile-time NUM_TOKENS_PER_CACHE_BLOCK={} must match runtime logical cache block={}",
            N,
            model_runtime_config.num_tokens_per_cache_block()
        );
        assert_eq!(
            L,
            model_runtime_config.num_cache_lanes(),
            "runtime service compile-time NUM_CACHE_LANE={} must match model cache lane count={}",
            L,
            model_runtime_config.num_cache_lanes()
        );

        let (req_slot_allocator, request_slot_reset_rx) =
            RequestSlotAllocator::new(model_runtime_config.max_running_requests as u64, shutdown.clone());
        let request_slot_reset_notifier = req_slot_allocator.reset_notifier();
        let page_id_allocator = Arc::new(U32IDAllocator::new(model_runtime_config.num_pages as u64));
        let block_cache = {
            let block_cache_vec = std::array::from_fn(|cache_lane| {
                let kv_block_allocator = TPKVBlockAllocator::new(
                    model_runtime_config.num_pages_per_kv_block(cache_lane),
                    page_id_allocator.clone(),
                );
                let state_block_allocator = TPStateBlockAllocator::new(
                    model_runtime_config.num_pages_per_state_block(cache_lane),
                    page_id_allocator.clone(),
                );
                let capacity = model_runtime_config.block_cache_capacity(cache_lane);
                Arc::new(SingleLaneTrieBlockCache::new(
                    kv_block_allocator,
                    state_block_allocator,
                    capacity,
                    shutdown.clone(),
                ))
            });
            Arc::new(MultiLaneTrieBlockCache::new(block_cache_vec))
        };

        let (new_req_tx, new_req_rx) = sync_bounded(model_runtime_config.max_running_requests);
        let (completed_turn_tx, completed_turn_rx) = async_bounded(model_runtime_config.max_running_requests);
        let model_executor_channel_capacity = scheduler_config
            .max_compute_slots
            .checked_add(1)
            .expect("model executor channel capacity must fit usize");
        let (model_executor_req_tx, model_executor_req_rx) = sync_bounded(model_executor_channel_capacity);
        let (model_executor_resp_tx, model_executor_resp_rx) = sync_bounded(model_executor_channel_capacity);
        let (async_task_req_tx, async_task_req_rx) = async_bounded(model_runtime_config.max_running_requests);
        let (async_task_resp_tx, async_task_resp_rx) =
            sync_bounded::<Box<dyn AsyncTaskResp>>(model_runtime_config.max_running_requests);
        let sessions = Sessions::new(
            model_runtime_config.context_window,
            block_cache.clone(),
            resource_processors.clone(),
            req_slot_allocator.clone(),
            new_req_tx,
            completed_turn_rx,
            shutdown.clone(),
            async_task_handle,
        );

        {
            let schedule_queue: ScheduleQueue<RuntimeRequest<N, P, L>, DeviceRequest, DeviceResponse> =
                ScheduleQueue::new(async_task_req_tx);
            let batcher = FIFOBatcher::new();
            let scheduler = InstrumentedScheduler::new(
                SimpleScheduler::new(
                    schedule_queue,
                    batcher,
                    scheduler_config.max_requests,
                    scheduler_config.max_tokens,
                    scheduler_config.max_tokens_per_request,
                    scheduler_config.max_compute_slots,
                ),
                num_spec_tokens,
            );
            let event_loop = EventLoop::new(
                new_req_rx,
                completed_turn_tx,
                async_task_resp_rx,
                model_executor_req_tx,
                model_executor_resp_rx,
                scheduler,
                req_slot_allocator,
                page_id_allocator,
                model_runtime_config.executor_hibernation_mode,
                model_runtime_config.executor_hibernation_timeout,
                shutdown.clone(),
            );
            let scheduler_shutdown = shutdown.clone();

            let scheduler_thread = std::thread::Builder::new()
                .name("inference-runtime-event-loop".to_string())
                .spawn(move || {
                    let _shutdown_guard = ShutdownGuard::new(scheduler_shutdown);
                    event_loop.event_loop()
                })
                .expect("inference runtime scheduler thread should start");
            drop(scheduler_thread);

            let async_task_pool = AsyncTaskPool::new(
                async_task_req_rx,
                async_task_resp_tx,
                shutdown.clone(),
                model_runtime_config.max_running_requests,
            );
            let async_task_shutdown = shutdown.clone();
            let async_task_join_handle = async_task_handle.spawn(async move {
                let _shutdown_guard = ShutdownGuard::new(async_task_shutdown);
                async_task_pool.event_loop().await
            });
            drop(async_task_join_handle);
        };

        Self {
            #[cfg(test)]
            model_runtime_config,

            shutdown,

            sessions,
            model_executor_req_rx,
            model_executor_resp_tx,
            request_slot_reset_notifier,
            request_slot_reset_rx,
        }
    }

    delegate::delegate! {
        to self.sessions {
            #[call(create)]
            #[allow(clippy::too_many_arguments)]
            pub fn create_session(
                &self,
                request_id: RawRequestID,
                history_tokens: Vec<Token>,
                prompt_tokens: Vec<Token>,
                sampled_tokens: Vec<Token>,
                resource_entries: Vec<(Resource, ResourcePlacement)>,
                sampling_config: SamplingConfig,
            ) -> Result<ExternalRequest>;

            #[call(resume)]
            pub async fn resume_session(
                &self,
                external_request: &ExternalRequest,
                prompt_tokens: Vec<Token>,
                sampling_config: SamplingConfig,
            ) -> Result<()>;

            pub async fn evict_one_req(&self) -> Result<bool>;
            pub async fn evict_expired(&self, max_idle: Duration) -> Result<usize>;
        }
    }

    pub fn model_executor_request_rx(&self) -> Receiver<ReplayableModelExecutorRequest> {
        self.model_executor_req_rx.clone()
    }

    pub fn model_executor_response_tx(&self) -> Sender<ReplayableModelExecutorResponse> {
        self.model_executor_resp_tx.clone()
    }

    pub fn request_slot_reset_notifier(&self) -> Arc<DedupNotifier<RawRequestSlot>> {
        self.request_slot_reset_notifier.clone()
    }

    pub fn request_slot_reset_rx(&self) -> Receiver<()> {
        self.request_slot_reset_rx.clone()
    }

    pub fn shutdown(&self) {
        self.shutdown.shutdown();
    }
}

impl<const N: usize, const L: usize, const P: usize> Drop for InferenceRuntime<N, L, P> {
    fn drop(&mut self) {
        self.shutdown.shutdown();
    }
}

pub fn serve_replay_model<const N: usize, const L: usize, M>(
    grpc_listen_addr: SocketAddr,
    http_listen_addr: SocketAddr,
    service: RPCService,
    resource_processors: Arc<ResourceProcessors>,
    model_runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
    executors: ReplayableModelExecutors<M>,
) -> Result<()>
where
    M: ReplayableDecoderModel,
{
    let shutdown = Shutdown::new();
    let server_tokio_runtime = tokio::runtime::Runtime::new()
        .map_err(|error| log_err_unavailable!("unable to initialize RPC async runtime: {error}"))?;
    let model_name = executors.decoder().model_name().to_string();
    let vocab_size = executors.decoder().vocab_size();
    let default_stop_sequences = executors.decoder().default_stop_sequences();
    let num_spec_tokens = executors.decoder().num_spec_tokens();
    let runtime = Arc::new(InferenceRuntime::<N, L, NUM_TRIE_PARTITION>::new(
        model_runtime_config,
        scheduler_config,
        num_spec_tokens,
        shutdown.clone(),
        server_tokio_runtime.handle(),
        resource_processors,
    ));
    let inference = Arc::new(Inference::new(runtime.clone(), vocab_size, default_stop_sequences));
    let server_shutdown = shutdown.clone();
    let server_thread = std::thread::Builder::new()
        .name("inference-rpc-servers".to_string())
        .spawn(move || {
            let _shutdown_guard = ShutdownGuard::new(server_shutdown.clone());
            server_tokio_runtime.block_on(rpc::run(
                grpc_listen_addr,
                http_listen_addr,
                model_name,
                service,
                inference,
                server_shutdown,
            ))
        })
        .map_err(|error| log_err_unavailable!("unable to start RPC server thread: {error}"))?;

    let executor = ReplayableDecoderModelEventLoop::new(
        runtime.model_executor_request_rx(),
        runtime.model_executor_response_tx(),
        runtime.request_slot_reset_notifier(),
        runtime.request_slot_reset_rx(),
        shutdown,
        executors,
        default_model_state_snapshot_path(),
    );
    executor.event_loop();
    runtime.shutdown();

    server_thread
        .join()
        .map_err(|_| log_err_internal!("RPC server thread panicked"))?
}

fn default_model_state_snapshot_path() -> PathBuf {
    static NEXT_SNAPSHOT_ID: AtomicU64 = AtomicU64::new(0);

    let snapshot_id = NEXT_SNAPSHOT_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("khalamari-model-state-{}-{snapshot_id}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use inference_runtime_core::Error;
    use inference_runtime_core::channel::Shutdown;
    use inference_runtime_core::compute::BatchDeviceResponse;
    use inference_runtime_core::compute::DeviceResponse;
    use inference_runtime_core::compute::ExecutorHibernationPlan;
    use inference_runtime_core::compute::QueryTokens;
    use inference_runtime_core::compute::ReplayableModelExecutorRequest;
    use inference_runtime_core::compute::ReplayableModelExecutorResponse;
    use inference_runtime_core::compute::SampledTokens;
    use inference_runtime_core::config::CacheLaneRuntimeConfig;
    use inference_runtime_core::config::DEFAULT_EXECUTOR_HIBERNATION_TIMEOUT;
    use inference_runtime_core::config::ExecutorHibernationMode;
    use inference_runtime_core::config::RuntimeConfig;
    use inference_runtime_core::config::SamplingConfig;
    use inference_runtime_core::config::SchedulerConfig;
    use inference_runtime_core::runtime::CompletionReason;
    use inference_runtime_core::runtime::RequestEvent;
    use inference_runtime_core::runtime::RequestStatus;
    use inference_runtime_core::runtime::Token;
    use inference_runtime_core::runtime::resource::processor::ResourceProcessors;
    use inference_runtime_core::runtime::scheduler::CommitResult;
    use inference_runtime_core::runtime::scheduler::ComputePhase;
    use inference_runtime_core::runtime::scheduler::PrepareResult;
    use inference_runtime_core::runtime::scheduler::UserRequest;
    use ordered_float::NotNan;

    use super::InferenceRuntime;
    use crate::api::Inference;
    use crate::api::decode::DecodeEvent;
    use crate::api::decode::DecodeRequest;

    #[test]
    fn test_runtime_accepts_a_logical_cache_block_larger_than_one_physical_kv_page() {
        let shutdown = Shutdown::new();
        let async_task_runtime = tokio::runtime::Runtime::new().expect("test Tokio runtime should initialize");
        let runtime_config = RuntimeConfig {
            max_running_requests: 1,
            executor_hibernation_timeout: DEFAULT_EXECUTOR_HIBERNATION_TIMEOUT,
            executor_hibernation_mode: ExecutorHibernationMode::Selected,
            context_window: 4096,
            num_tokens_per_cache_block: 1024,
            num_pages: 64,
            cache_lanes: vec![CacheLaneRuntimeConfig {
                num_pages_per_kv_block: 64,
                num_pages_per_state_block: 0,
                block_cache_capacity: 1,
            }],
        };
        assert_eq!(runtime_config.num_tokens_per_cache_block(), 1024);

        let runtime = InferenceRuntime::<1024, 1, 4>::new(
            runtime_config,
            SchedulerConfig {
                max_requests: 1,
                max_tokens: 1024,
                max_tokens_per_request: 1024,
                max_compute_slots: 1,
            },
            0,
            shutdown.clone(),
            async_task_runtime.handle(),
            Arc::new(ResourceProcessors::new()),
        );
        assert_eq!(runtime.model_runtime_config.num_tokens_per_cache_block(), 1024);
        shutdown.shutdown();
    }

    #[test]
    fn test_runtime_validates_initial_tokens_for_cache_lanes() {
        let (single_lane, single_lane_shutdown, _single_lane_async_runtime) = test_runtime::<1>();
        assert!(matches!(
            single_lane.sessions.initialize(
                1,
                vec![],
                vec![],
                vec![],
                vec![],
                SamplingConfig::default(),
            ),
            Err(Error::InvalidArgument(message)) if message.contains("minimum initial token count is 1")
        ));
        single_lane_shutdown.shutdown();

        let (mtp, mtp_shutdown, _mtp_async_runtime) = test_runtime::<4>();
        assert!(matches!(
            mtp.sessions.initialize(
                1,
                vec![],
                vec![Token::new(1), Token::new(2)],
                vec![],
                vec![],
                SamplingConfig::default(),
            ),
            Err(Error::InvalidArgument(message)) if message.contains("minimum initial token count is 3")
        ));
        assert!(
            mtp.sessions
                .initialize(
                    2,
                    vec![],
                    vec![Token::new(1), Token::new(2), Token::new(3)],
                    vec![],
                    vec![],
                    SamplingConfig::default(),
                )
                .is_ok()
        );
        mtp_shutdown.shutdown();
    }

    #[test]
    fn test_runtime_rejects_initial_tokens_without_output_capacity() {
        let (runtime, shutdown, _async_runtime) = test_runtime_with_context::<1>(4);
        let sampling = SamplingConfig {
            max_sampled_tokens: 2,
            ..SamplingConfig::default()
        };
        assert!(matches!(
            runtime.sessions.initialize(
                1,
                vec![Token::new(1)],
                vec![Token::new(2), Token::new(3)],
                vec![Token::new(4)],
                vec![],
                sampling,
            ),
            Err(Error::InvalidArgument(message))
                if message.contains("initial token count=4") && message.contains("context window=4")
        ));
        shutdown.shutdown();
    }

    #[test]
    fn test_runtime_rejects_exhausted_initial_sampled_limit() {
        let (runtime, shutdown, _async_runtime) = test_runtime_with_context::<1>(4);
        let sampling = SamplingConfig {
            max_sampled_tokens: 2,
            ..SamplingConfig::default()
        };
        assert!(matches!(
            runtime.sessions.initialize(
                1,
                vec![],
                vec![Token::new(1)],
                vec![Token::new(2), Token::new(3)],
                vec![],
                sampling,
            ),
            Err(Error::InvalidArgument(message))
                if message.contains("initial sampled token count=2")
                    && message.contains("max_sampled_tokens=2")
        ));
        shutdown.shutdown();
    }

    #[test]
    fn test_commit_reserves_sampled_token_at_context_limit() {
        let (runtime, shutdown, _async_runtime) = test_runtime_with_context::<1>(4);
        let sampling = SamplingConfig {
            max_sampled_tokens: 8,
            ..SamplingConfig::default()
        };
        let (mut request, external_request) = runtime
            .sessions
            .initialize(1, vec![], vec![Token::new(1), Token::new(2)], vec![], vec![], sampling)
            .unwrap();

        let first_query = prepare_decode(&mut request, 2);
        assert!(matches!(
            request.commit(decode_response(1, first_query, &[], 3, &[4, 5, 6])),
            CommitResult::Continue
        ));
        let RequestEvent::TokenProbs(first_visible) = external_request.event_rx().try_recv().unwrap() else {
            panic!("test request should produce token probabilities")
        };

        let second_query = prepare_decode(&mut request, 1);
        let QueryTokens::Decode { spec_tokens, .. } = &second_query else {
            panic!("test request should prepare Decode")
        };
        assert!(spec_tokens.is_empty());
        assert!(matches!(
            request.commit(decode_response(1, second_query, &[], 4, &[5, 6, 7])),
            CommitResult::Terminal
        ));
        let RequestEvent::TokenProbs(second_visible) = external_request.event_rx().try_recv().unwrap() else {
            panic!("test request should produce token probabilities")
        };
        assert_eq!(
            [first_visible.tokens, second_visible.tokens].concat(),
            vec![Token::new(3), Token::new(4)]
        );
        assert_eq!(
            external_request.status(),
            RequestStatus::Completed(CompletionReason::ContextLimit)
        );
        shutdown.shutdown();
    }

    #[test]
    fn test_commit_context_limit_precedes_tied_turn_limits() {
        assert_eq!(
            single_decode_completion(1, vec![vec![Token::new(4)]]),
            CompletionReason::ContextLimit
        );
        assert_eq!(single_decode_completion(1, vec![]), CompletionReason::ContextLimit);
    }

    #[test]
    fn test_decode_completes_at_context_window_through_runtime_event_loop() {
        let (runtime, shutdown, async_runtime) = test_runtime_with_context::<1>(4);
        let runtime = Arc::new(runtime);
        let inference = Inference::new(runtime.clone(), 1024, vec![]);
        let sampling = SamplingConfig {
            max_sampled_tokens: 8,
            ..SamplingConfig::default()
        };
        let mut response = inference
            .create_session(
                DecodeRequest::new(vec![Token::new(1), Token::new(2), Token::new(3)], vec![], sampling).unwrap(),
            )
            .unwrap();

        let ReplayableModelExecutorRequest::Batch(batch_request) = runtime
            .model_executor_request_rx()
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
        else {
            panic!("runtime should submit a device batch")
        };
        let mut dev_reqs = batch_request.dev_reqs;
        assert_eq!(dev_reqs.len(), 1);
        let dev_req = dev_reqs.pop().unwrap();
        let dev_resp = decode_response(dev_req.req_id, dev_req.decoder_query_tokens, &[], 4, &[]);
        runtime
            .model_executor_response_tx()
            .send(ReplayableModelExecutorResponse::Batch(BatchDeviceResponse::new(
                batch_request.seq,
                [dev_resp],
            )))
            .unwrap();

        async_runtime.block_on(async {
            let DecodeEvent::TokenProbs(token_probs) = response.recv_event().await.unwrap() else {
                panic!("decode response should contain the sampled token")
            };
            assert_eq!(token_probs.tokens, vec![Token::new(4)]);
            assert!(matches!(
                response.recv_event().await.unwrap(),
                DecodeEvent::Completed {
                    reason: CompletionReason::ContextLimit,
                    num_output_tokens: 1,
                }
            ));
        });
        shutdown.shutdown();
    }

    #[test]
    fn test_model_executor_hibernates_when_idle_and_starts_before_batch() {
        let (runtime, shutdown, _async_runtime) =
            test_runtime_with_executor_hibernation_timeout::<1>(Duration::from_millis(20));
        let request_rx = runtime.model_executor_request_rx();
        let response_tx = runtime.model_executor_response_tx();

        let ReplayableModelExecutorRequest::Stop(stopped_plan) =
            request_rx.recv_timeout(Duration::from_secs(1)).unwrap()
        else {
            panic!("idle runtime must stop the model executor")
        };
        assert_eq!(stopped_plan, ExecutorHibernationPlan::selected(vec![], vec![]));

        let _external_request = runtime
            .sessions
            .create(
                1,
                vec![],
                vec![Token::new(1)],
                vec![],
                vec![],
                SamplingConfig {
                    max_sampled_tokens: 1,
                    ..SamplingConfig::default()
                },
            )
            .unwrap();
        let ReplayableModelExecutorRequest::Start(started_plan) =
            request_rx.recv_timeout(Duration::from_secs(1)).unwrap()
        else {
            panic!("queued request must start the model executor")
        };
        assert_eq!(started_plan, stopped_plan);
        // A delayed Stopped acknowledgement can submit the batch behind Start.
        response_tx.send(ReplayableModelExecutorResponse::Stopped).unwrap();
        let ReplayableModelExecutorRequest::Batch(batch_request) =
            request_rx.recv_timeout(Duration::from_secs(1)).unwrap()
        else {
            panic!("runtime must queue the batch after Start")
        };
        assert!(matches!(
            request_rx.recv_timeout(Duration::from_millis(100)),
            Err(crossbeam_channel::RecvTimeoutError::Timeout)
        ));
        assert!(!shutdown.is_shutdown());
        response_tx.send(ReplayableModelExecutorResponse::Started).unwrap();
        assert!(matches!(
            request_rx.recv_timeout(Duration::from_millis(100)),
            Err(crossbeam_channel::RecvTimeoutError::Timeout)
        ));
        let dev_resp = batch_request
            .dev_reqs
            .into_iter()
            .map(|dev_req| decode_response(dev_req.req_id, dev_req.decoder_query_tokens, &[], 2, &[]));
        response_tx
            .send(ReplayableModelExecutorResponse::Batch(BatchDeviceResponse::new(
                batch_request.seq,
                dev_resp,
            )))
            .unwrap();
        let ReplayableModelExecutorRequest::Stop(active_plan) =
            request_rx.recv_timeout(Duration::from_secs(1)).unwrap()
        else {
            panic!("idle runtime must stop the model executor after the batch")
        };
        assert_eq!(
            active_plan,
            ExecutorHibernationPlan::selected(std::iter::once(0..1).collect(), std::iter::once(0..64).collect())
        );

        shutdown.shutdown();
    }

    #[test]
    fn test_model_executor_uses_fixed_all_hibernation_mode() {
        let (runtime, shutdown, _async_runtime) =
            test_runtime_with_executor_hibernation_mode::<1>(Duration::from_millis(20), ExecutorHibernationMode::All);

        let ReplayableModelExecutorRequest::Stop(plan) = runtime
            .model_executor_request_rx()
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
        else {
            panic!("idle runtime must stop the model executor")
        };
        assert_eq!(plan, ExecutorHibernationPlan::All);
        shutdown.shutdown();
    }

    fn test_runtime<const L: usize>() -> (InferenceRuntime<1024, L, 4>, Shutdown, tokio::runtime::Runtime) {
        test_runtime_with_executor_hibernation_timeout(DEFAULT_EXECUTOR_HIBERNATION_TIMEOUT)
    }

    fn test_runtime_with_context<const L: usize>(
        context_window: usize,
    ) -> (InferenceRuntime<1024, L, 4>, Shutdown, tokio::runtime::Runtime) {
        test_runtime_with_config(
            context_window,
            DEFAULT_EXECUTOR_HIBERNATION_TIMEOUT,
            ExecutorHibernationMode::Selected,
        )
    }

    fn test_runtime_with_executor_hibernation_timeout<const L: usize>(
        executor_hibernation_timeout: Duration,
    ) -> (InferenceRuntime<1024, L, 4>, Shutdown, tokio::runtime::Runtime) {
        test_runtime_with_executor_hibernation_mode(executor_hibernation_timeout, ExecutorHibernationMode::Selected)
    }

    fn test_runtime_with_executor_hibernation_mode<const L: usize>(
        executor_hibernation_timeout: Duration,
        executor_hibernation_mode: ExecutorHibernationMode,
    ) -> (InferenceRuntime<1024, L, 4>, Shutdown, tokio::runtime::Runtime) {
        test_runtime_with_config(4096, executor_hibernation_timeout, executor_hibernation_mode)
    }

    fn test_runtime_with_config<const L: usize>(
        context_window: usize,
        executor_hibernation_timeout: Duration,
        executor_hibernation_mode: ExecutorHibernationMode,
    ) -> (InferenceRuntime<1024, L, 4>, Shutdown, tokio::runtime::Runtime) {
        let shutdown = Shutdown::new();
        let async_runtime = tokio::runtime::Runtime::new().expect("test Tokio runtime should initialize");
        let runtime = InferenceRuntime::<1024, L, 4>::new(
            RuntimeConfig {
                max_running_requests: 1,
                executor_hibernation_timeout,
                executor_hibernation_mode,
                context_window,
                num_tokens_per_cache_block: 1024,
                num_pages: 64 * L,
                cache_lanes: (0..L)
                    .map(|_| {
                        CacheLaneRuntimeConfig {
                            num_pages_per_kv_block: 64,
                            num_pages_per_state_block: 0,
                            block_cache_capacity: 1,
                        }
                    })
                    .collect(),
            },
            SchedulerConfig {
                max_requests: 1,
                max_tokens: 1024,
                max_tokens_per_request: 1024,
                max_compute_slots: 1,
            },
            0,
            shutdown.clone(),
            async_runtime.handle(),
            Arc::new(ResourceProcessors::new()),
        );
        (runtime, shutdown, async_runtime)
    }

    fn prepare_decode(request: &mut super::RuntimeRequest<1024, 4, 1>, token_budget: usize) -> QueryTokens {
        match request.prepare(token_budget) {
            PrepareResult::Continue {
                dev_req,
                compute_phase: ComputePhase::Decode { .. },
            } => dev_req.decoder_query_tokens,
            _ => panic!("test request should prepare Decode"),
        }
    }

    fn single_decode_completion(max_sampled_tokens: usize, stop_sequences: Vec<Vec<Token>>) -> CompletionReason {
        let (runtime, shutdown, _async_runtime) = test_runtime_with_context::<1>(4);
        let sampling = SamplingConfig {
            max_sampled_tokens,
            stop_sequences,
            ..SamplingConfig::default()
        };
        let (mut request, external_request) = runtime
            .sessions
            .initialize(
                1,
                vec![],
                vec![Token::new(1), Token::new(2), Token::new(3)],
                vec![],
                vec![],
                sampling,
            )
            .unwrap();
        let query_tokens = prepare_decode(&mut request, 3);
        assert!(matches!(
            request.commit(decode_response(1, query_tokens, &[], 4, &[])),
            CommitResult::Terminal
        ));
        let RequestStatus::Completed(completion) = external_request.status() else {
            panic!("test request should complete")
        };
        shutdown.shutdown();
        completion
    }

    fn decode_response(
        req_id: usize,
        query_tokens: QueryTokens,
        validated_tokens: &[u32],
        sampled_token: u32,
        spec_tokens: &[u32],
    ) -> DeviceResponse {
        let probability = NotNan::new(1.0).unwrap();
        DeviceResponse {
            req_id,
            sampled_tokens: SampledTokens::Decode {
                epoch: query_tokens.epoch(),
                validated_tokens: validated_tokens.iter().copied().map(Token::new).collect(),
                validated_probs: vec![probability; validated_tokens.len()],
                sampled_token: Token::new(sampled_token),
                sampled_prob: probability,
                spec_tokens: spec_tokens.iter().copied().map(Token::new).collect(),
                spec_probs: vec![probability; spec_tokens.len()],
                spec_confidences: vec![probability; spec_tokens.len()],
            },
            query_tokens,
        }
    }
}
