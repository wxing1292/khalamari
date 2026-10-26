use inference_runtime_core::config::DEFAULT_SAMPLING_TEMPERATURE;
use inference_runtime_core::config::DEFAULT_SAMPLING_TOP_K;
use inference_runtime_core::config::DEFAULT_SAMPLING_TOP_P;
use inference_runtime_core::runtime::AtomicRequestStatus;
use ordered_float::NotNan;

use super::*;

#[test]
fn test_request_validation() {
    let mut sampling = fixture_sampling(1);
    assert!(matches!(
        DecodeRequest::new(vec![], vec![], sampling.clone()),
        Err(Error::InvalidArgument(_))
    ));
    sampling.max_sampled_tokens = 0;
    assert!(matches!(
        DecodeRequest::new(vec![fixture_token(1)], vec![], sampling.clone()),
        Err(Error::InvalidArgument(_))
    ));
    sampling.max_sampled_tokens = 1;
    sampling.temperature = f32::NAN;
    assert!(matches!(
        DecodeRequest::new(vec![fixture_token(1)], vec![], sampling.clone()),
        Err(Error::InvalidArgument(_))
    ));
    sampling.temperature = 1.0;
    sampling.top_k = 0;
    assert!(matches!(
        DecodeRequest::new(vec![fixture_token(1)], vec![], sampling.clone()),
        Err(Error::InvalidArgument(_))
    ));
    sampling.top_k = 1;
    sampling.top_p = 2.0;
    assert!(matches!(
        DecodeRequest::new(vec![fixture_token(1)], vec![], sampling.clone()),
        Err(Error::InvalidArgument(_))
    ));
    sampling.top_p = 1.0;
    sampling.stop_sequences.push(vec![]);
    assert!(matches!(
        DecodeRequest::new(vec![fixture_token(1)], vec![], sampling),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn test_stop_sequences_merge_and_deduplicate() {
    let mut input = vec![vec![fixture_token(3)], vec![fixture_token(1)]];
    merge_stop_sequences(&mut input, &[vec![fixture_token(2)], vec![fixture_token(1)]]);
    assert_eq!(
        input,
        vec![vec![fixture_token(1)], vec![fixture_token(2)], vec![fixture_token(3)]]
    );
}

#[tokio::test]
async fn test_response_preserves_token_chunks() {
    let (response, sender, status) = fixture_request(RequestStatus::Running);
    let mut response = response;
    sender
        .send(RequestEvent::TokenProbs(TokenProbs {
            tokens: vec![fixture_token(1), fixture_token(2)],
            probs: vec![NotNan::new(0.1).unwrap(), NotNan::new(0.2).unwrap()],
        }))
        .await
        .unwrap();
    sender
        .send(RequestEvent::TurnCompleted(CompletionReason::StopSequence))
        .await
        .unwrap();
    match response.recv_event().await.unwrap() {
        DecodeEvent::TokenProbs(chunk) => assert_eq!(chunk.tokens.len(), 2),
        _ => panic!("expected token chunk"),
    }
    assert!(matches!(
        response.recv_event().await.unwrap(),
        DecodeEvent::Completed { .. }
    ));
    assert_eq!(response.num_history_tokens(), 4);
    assert_eq!(status.load(), RequestStatus::Running);
}

#[tokio::test]
async fn test_response_completes_on_stop_sequence() {
    assert_completion_reason(CompletionReason::StopSequence).await;
}

#[tokio::test]
async fn test_response_completes_on_length_limit() {
    assert_completion_reason(CompletionReason::LengthLimit).await;
}

#[tokio::test]
async fn test_response_completes_on_context_limit() {
    assert_completion_reason(CompletionReason::ContextLimit).await;
}

#[tokio::test]
async fn test_response_maps_terminal_errors() {
    for status in [
        RequestStatus::Cancelled,
        RequestStatus::TimedOut,
        RequestStatus::Aborted,
        RequestStatus::Evicted,
    ] {
        let (response, sender, _) = fixture_request(status);
        let mut response = response;
        drop(sender);
        let error = response.recv_event().await.unwrap_err();
        assert!(matches!(
            (status, error),
            (RequestStatus::Cancelled, Error::Cancelled(_))
                | (RequestStatus::TimedOut, Error::DeadlineExceeded(_))
                | (RequestStatus::Aborted, Error::Aborted(_))
                | (RequestStatus::Evicted, Error::Evicted(_))
        ));
    }
}

#[test]
fn test_response_drop_cancels_request() {
    let (response, _sender, status) = fixture_request(RequestStatus::Initialized);
    drop(response);
    assert_eq!(status.load(), RequestStatus::Cancelled);
}

async fn assert_completion_reason(expected: CompletionReason) {
    if expected == CompletionReason::ContextLimit {
        let (response, sender, _) = fixture_request(RequestStatus::Completed(expected));
        let mut response = response;
        drop(sender);
        match response.recv_event().await.unwrap() {
            DecodeEvent::Completed { reason, .. } => assert_eq!(reason, expected),
            DecodeEvent::TokenProbs(_) => panic!("expected completion"),
        }
        return;
    }

    let (response, sender, _) = fixture_request(RequestStatus::Running);
    let mut response = response;
    sender.send(RequestEvent::TurnCompleted(expected)).await.unwrap();
    match response.recv_event().await.unwrap() {
        DecodeEvent::Completed { reason, .. } => assert_eq!(reason, expected),
        DecodeEvent::TokenProbs(_) => panic!("expected completion"),
    }
}

fn fixture_token(value: u32) -> Token {
    Token::new(value)
}

fn fixture_sampling(max_sampled_tokens: usize) -> SamplingConfig {
    SamplingConfig {
        max_sampled_tokens,
        temperature: DEFAULT_SAMPLING_TEMPERATURE,
        top_k: DEFAULT_SAMPLING_TOP_K,
        top_p: DEFAULT_SAMPLING_TOP_P,
        seed: None,
        stop_sequences: Vec::new(),
    }
}

fn fixture_request(
    status: RequestStatus,
) -> (
    DecodeSession<1, 1, 1>,
    async_channel::Sender<RequestEvent>,
    AtomicRequestStatus,
) {
    let (sender, receiver) = async_channel::unbounded();
    let (cancel_tx, _cancel_rx) = async_channel::unbounded();
    let request_status = AtomicRequestStatus::new();
    match status {
        RequestStatus::Cancelled => {
            request_status.store_cancelled();
        },
        RequestStatus::TimedOut => {
            request_status.store_timed_out();
        },
        RequestStatus::Aborted => {
            request_status.store_aborted();
        },
        RequestStatus::Evicted => {
            request_status.store_evicted();
        },
        RequestStatus::Completed(completion) => {
            request_status.store_completed(completion);
        },
        RequestStatus::Initialized => {},
        RequestStatus::Running | RequestStatus::Swapped => {
            request_status.store_running();
        },
    }
    let request = ExternalRequest::new(42, request_status.clone(), receiver, cancel_tx);
    (DecodeSession::new(request, 2), sender, request_status)
}
