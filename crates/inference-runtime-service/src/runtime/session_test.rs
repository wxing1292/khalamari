use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::RecvTimeoutError;
use inference_runtime_core::Error;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::DeviceResponse;
use inference_runtime_core::compute::ReplayableModelExecutorRequest;
use inference_runtime_core::compute::ReplayableModelExecutorResponse;
use inference_runtime_core::compute::SampledTokens;
use inference_runtime_core::config::CacheLaneRuntimeConfig;
use inference_runtime_core::config::DEFAULT_EXECUTOR_HIBERNATION_TIMEOUT;
use inference_runtime_core::config::ExecutorHibernationMode;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::config::SchedulerConfig;
use inference_runtime_core::runtime::Token;
use inference_runtime_core::runtime::resource::processor::ResourceProcessors;
use ordered_float::NotNan;

use crate::api::Inference;
use crate::api::decode::DecodeEvent;
use crate::api::decode::DecodeRequest;
use crate::api::decode::DecodeSession;
use crate::runtime::InferenceRuntime;

type TestRuntime = InferenceRuntime<1024, 1, 4>;
type TestSession = DecodeSession<1024, 1, 4>;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_evict_one_req() {
    let (runtime, shutdown) = new_runtime(2);
    let inference = Inference::new(runtime.clone(), 1024, Vec::new());
    let mut first = complete_turn(&runtime, &inference, 1, 101).await;
    let mut second = complete_turn(&runtime, &inference, 2, 202).await;
    let second_request_id = second.request_id();
    for token_id in [1024, u32::MAX] {
        assert!(matches!(
            inference.resume_session(&mut first, request(token_id)).await,
            Err(Error::InvalidArgument(message)) if message.contains("must be below model vocab_size=1024")
        ));
    }

    assert!(runtime.evict_one_req().await.unwrap());
    assert!(matches!(
        inference.resume_session(&mut first, request(3)).await,
        Err(Error::Evicted(_))
    ));

    inference.resume_session(&mut second, request(4)).await.unwrap();
    let ReplayableModelExecutorRequest::Batch(batch) = recv_executor_request(&runtime).await else {
        panic!("resident request must resume through an executor batch")
    };
    assert_eq!(batch.dev_reqs.len(), 1);
    assert_eq!(batch.dev_reqs[0].req_id, second_request_id);
    assert!(!runtime.evict_one_req().await.unwrap());

    drop(first);
    drop(second);
    drop(inference);
    shutdown_runtime(runtime, shutdown).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_evict_expired() {
    let (runtime, shutdown) = new_runtime(1);
    let inference = Inference::new(runtime.clone(), 1024, Vec::new());
    let mut session = complete_turn(&runtime, &inference, 1, 101).await;

    assert_eq!(runtime.evict_expired(Duration::MAX).await.unwrap(), 0);
    assert_eq!(runtime.evict_expired(Duration::ZERO).await.unwrap(), 1);
    assert_eq!(runtime.evict_expired(Duration::ZERO).await.unwrap(), 0);
    assert!(matches!(session.recv_event().await, Err(Error::Evicted(_))));

    let replacement = inference.create_session(request(2)).unwrap();
    let ReplayableModelExecutorRequest::Batch(batch) = recv_executor_request(&runtime).await else {
        panic!("eviction must release the request slot")
    };
    assert_eq!(batch.dev_reqs.len(), 1);
    assert_eq!(batch.dev_reqs[0].req_id, replacement.request_id());

    drop(session);
    drop(replacement);
    drop(inference);
    shutdown_runtime(runtime, shutdown).await;
}

async fn complete_turn(
    runtime: &Arc<TestRuntime>,
    inference: &Inference<1024, 1, 4>,
    input_token: u32,
    output_token: u32,
) -> TestSession {
    let mut session = inference.create_session(request(input_token)).unwrap();
    let ReplayableModelExecutorRequest::Batch(mut batch) = recv_executor_request(runtime).await else {
        panic!("decode turn must submit an executor batch")
    };
    assert_eq!(batch.dev_reqs.len(), 1);
    let request = batch.dev_reqs.pop().unwrap();
    let probability = NotNan::new(1.0).unwrap();
    let response = DeviceResponse {
        req_id: request.req_id,
        sampled_tokens: SampledTokens::Decode {
            epoch: request.decoder_query_tokens.epoch(),
            validated_tokens: Vec::new(),
            validated_probs: Vec::new(),
            sampled_token: Token::new(output_token),
            sampled_prob: probability,
            spec_tokens: Vec::new(),
            spec_probs: Vec::new(),
            spec_confidences: Vec::new(),
        },
        query_tokens: request.decoder_query_tokens,
    };
    runtime
        .model_executor_response_tx()
        .send(ReplayableModelExecutorResponse::Batch(BatchDeviceResponse::new(
            batch.seq,
            [response],
        )))
        .unwrap();
    let DecodeEvent::TokenProbs(token_probs) = session.recv_event().await.unwrap() else {
        panic!("decode turn must return sampled tokens")
    };
    assert_eq!(token_probs.tokens, vec![Token::new(output_token)]);
    assert!(matches!(
        session.recv_event().await.unwrap(),
        DecodeEvent::Completed { .. }
    ));
    session
}

async fn recv_executor_request(runtime: &Arc<TestRuntime>) -> ReplayableModelExecutorRequest {
    let request_rx = runtime.model_executor_request_rx();
    tokio::task::spawn_blocking(move || request_rx.recv_timeout(Duration::from_secs(1)).unwrap())
        .await
        .unwrap()
}

async fn shutdown_runtime(runtime: Arc<TestRuntime>, shutdown: Shutdown) {
    let request_slot_reset_rx = runtime.request_slot_reset_rx();
    shutdown.shutdown();
    drop(runtime);
    tokio::task::spawn_blocking(move || {
        loop {
            match request_slot_reset_rx.recv_timeout(Duration::from_secs(1)) {
                Ok(()) => {},
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => panic!("runtime event loop did not stop"),
            }
        }
    })
    .await
    .unwrap();
}

fn request(token: u32) -> DecodeRequest {
    DecodeRequest::new(
        vec![Token::new(token)],
        Vec::new(),
        SamplingConfig {
            max_sampled_tokens: 1,
            ..SamplingConfig::default()
        },
    )
    .unwrap()
}

fn new_runtime(max_running_requests: usize) -> (Arc<TestRuntime>, Shutdown) {
    let shutdown = Shutdown::new();
    let async_task_handle = tokio::runtime::Handle::current();
    let runtime = Arc::new(TestRuntime::new(
        RuntimeConfig {
            max_running_requests,
            executor_hibernation_timeout: DEFAULT_EXECUTOR_HIBERNATION_TIMEOUT,
            executor_hibernation_mode: ExecutorHibernationMode::Selected,
            context_window: 4096,
            num_tokens_per_cache_block: 1024,
            num_pages: 64 * max_running_requests,
            cache_lanes: vec![CacheLaneRuntimeConfig {
                num_pages_per_kv_block: 64,
                num_pages_per_state_block: 0,
                block_cache_capacity: max_running_requests,
            }],
        },
        SchedulerConfig {
            max_requests: max_running_requests,
            max_tokens: 1024 * max_running_requests,
            max_tokens_per_request: 1024,
            max_compute_slots: 1,
        },
        0,
        shutdown.clone(),
        &async_task_handle,
        Arc::new(ResourceProcessors::new()),
    ));
    (runtime, shutdown)
}
