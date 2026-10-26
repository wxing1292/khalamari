//! gRPC generation integration tests.

use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use hf_chat_template::ChatTemplate;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::DeviceResponse;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::compute::ReplayableModelExecutorRequest;
use inference_runtime_core::compute::ReplayableModelExecutorResponse;
use inference_runtime_core::compute::SampledTokens;
use inference_runtime_core::config::CacheLaneRuntimeConfig;
use inference_runtime_core::config::DEFAULT_EXECUTOR_HIBERNATION_TIMEOUT;
use inference_runtime_core::config::ExecutorHibernationMode;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::config::SchedulerConfig;
use inference_runtime_core::runtime::Token;
use inference_runtime_core::runtime::resource::processor::ResourceProcessors;
use inference_runtime_core::tokenizer::huggingface::HFTokenizer;
use inference_runtime_proto::inference_runtime_service::CompletionReason as ProtoCompletionReason;
use inference_runtime_proto::inference_runtime_service::GenerateMessagesRequest as ProtoGenerateMessagesRequest;
use inference_runtime_proto::inference_runtime_service::GenerateTokensRequest as ProtoGenerateTokensRequest;
use inference_runtime_proto::inference_runtime_service::GenerationConfig as ProtoGenerationConfig;
use inference_runtime_proto::inference_runtime_service::InputMessage as ProtoInputMessage;
use inference_runtime_proto::inference_runtime_service::TextMessage as ProtoTextMessage;
use inference_runtime_proto::inference_runtime_service::generate_messages_response::Event as MessageEvent;
use inference_runtime_proto::inference_runtime_service::generate_tokens_response::Event as TokenEvent;
use inference_runtime_proto::inference_runtime_service::inference_runtime_client::InferenceRuntimeClient;
use inference_runtime_proto::inference_runtime_service::inference_runtime_server::InferenceRuntimeServer;
use inference_runtime_proto::inference_runtime_service::input_message::Message as ProtoMessage;
use ordered_float::NotNan;
use tokenizers::AddedToken;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::Code;
use tonic::Request;
use tonic::transport::Server;

use super::GRPCServer;
use crate::api::Inference;
use crate::codec::qwen::QwenCodec;
use crate::rpc::RPCService;
use crate::runtime::InferenceRuntime;

type TestRuntime = InferenceRuntime<1024, 1, 4>;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_generate_tokens_stream_lifecycle() {
    let runtime = new_runtime(4);
    let inference = Arc::new(Inference::new(runtime.clone(), 1024, Vec::new()));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let listen_addr = listener.local_addr().unwrap();
    let (server_shutdown_tx, server_shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(
        Server::builder()
            .add_service(InferenceRuntimeServer::new(GRPCServer::new(
                inference,
                RPCService::Messages(Arc::new(fixture_message_codec())),
            )))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
                let _ = server_shutdown_rx.await;
            }),
    );
    let mut client = InferenceRuntimeClient::connect(format!("http://{listen_addr}"))
        .await
        .unwrap();
    let status = client
        .transcribe_audio(inference_runtime_proto::inference_runtime_service::AudioTranscribeReq::default())
        .await
        .unwrap_err();
    assert_eq!(status.code(), Code::Unimplemented);
    for token_id in [1024, u32::MAX] {
        let status = client.generate_tokens(fixture_request(token_id)).await.unwrap_err();
        assert_eq!(status.code(), Code::InvalidArgument);
        assert!(status.message().contains("must be below model vocab_size=1024"));
    }
    let empty_stream = tokio_stream::empty::<ProtoGenerateTokensRequest>();
    let status = client
        .generate_tokens_stream(Request::new(empty_stream))
        .await
        .unwrap_err();
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(status.message().contains("at least one request"));

    let (input_closed_tx, input_closed_rx) = std::sync::mpsc::channel();
    let driver_runtime = runtime.clone();
    let driver = tokio::task::spawn_blocking(move || drive_turns(driver_runtime, input_closed_rx, vec![Token::new(2)]));

    let (input_tx, mut input_rx) = mpsc::channel(3);
    input_tx.send(fixture_request(1)).await.unwrap();
    input_tx.send(fixture_request(2)).await.unwrap();
    input_tx
        .send(ProtoGenerateTokensRequest {
            tokens: Vec::new(),
            ..fixture_request(3)
        })
        .await
        .unwrap();
    drop(input_tx);
    let input = async_stream::stream! {
        while let Some(request) = input_rx.recv().await {
            yield request;
        }
        input_closed_tx.send(()).unwrap();
    };
    let mut output = client
        .generate_tokens_stream(Request::new(input))
        .await
        .unwrap()
        .into_inner();

    let first_chunk = output.message().await.unwrap().unwrap();
    let first_chunk_id = first_chunk.request_id;
    let Some(TokenEvent::Chunk(first_chunk)) = first_chunk.event else {
        panic!("first turn must start with a chunk")
    };
    assert_eq!(first_chunk.tokens, vec![101]);
    assert_eq!(first_chunk.probs, vec![1.0]);

    let first_completion = output.message().await.unwrap().unwrap();
    let first_completion_id = first_completion.request_id;
    let Some(TokenEvent::Completion(first_completion)) = first_completion.event else {
        panic!("first turn must end with one completion")
    };
    assert_eq!(first_completion.reason, ProtoCompletionReason::LengthLimit as i32);
    assert_eq!(first_completion.num_output_tokens, 1);

    let second_chunk = output.message().await.unwrap().unwrap();
    let second_chunk_id = second_chunk.request_id;
    let Some(TokenEvent::Chunk(second_chunk)) = second_chunk.event else {
        panic!("second turn must start with a chunk")
    };
    assert_eq!(second_chunk.tokens, vec![202]);
    assert_eq!(second_chunk.probs, vec![1.0]);

    let second_completion = output.message().await.unwrap().unwrap();
    let second_completion_id = second_completion.request_id;
    let Some(TokenEvent::Completion(second_completion)) = second_completion.event else {
        panic!("second turn must end with one completion")
    };
    assert_eq!(second_completion.reason, ProtoCompletionReason::ContextLimit as i32);
    assert_eq!(second_completion.num_output_tokens, 1);

    assert_ne!(first_chunk_id, 0);
    assert_eq!(first_chunk_id, first_completion_id);
    assert_eq!(first_chunk_id, second_chunk_id);
    assert_eq!(second_chunk_id, second_completion_id);

    assert!(output.message().await.unwrap().is_none());

    driver.await.unwrap();
    drop(output);
    drop(client);
    server_shutdown_tx.send(()).unwrap();
    server.await.unwrap().unwrap();
    runtime.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_generate_messages_stream_reuses_resident_request() {
    let runtime = new_runtime(5);
    let inference = Arc::new(Inference::new(runtime.clone(), 1024, Vec::new()));
    let codec = Arc::new(fixture_message_codec());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let listen_addr = listener.local_addr().unwrap();
    let (server_shutdown_tx, server_shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(
        Server::builder()
            .add_service(InferenceRuntimeServer::new(GRPCServer::new(
                inference,
                RPCService::Messages(codec),
            )))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
                let _ = server_shutdown_rx.await;
            }),
    );
    let mut client = InferenceRuntimeClient::connect(format!("http://{listen_addr}"))
        .await
        .unwrap();
    let empty_stream = tokio_stream::empty::<ProtoGenerateMessagesRequest>();
    let status = client
        .generate_messages_stream(Request::new(empty_stream))
        .await
        .unwrap_err();
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(status.message().contains("at least one request"));

    let (input_closed_tx, input_closed_rx) = std::sync::mpsc::channel();
    let driver_runtime = runtime.clone();
    let driver = tokio::task::spawn_blocking(move || {
        drive_turns(driver_runtime, input_closed_rx, vec![Token::new(4), Token::new(2)])
    });

    let (input_tx, mut input_rx) = mpsc::channel(3);
    input_tx.send(fixture_message_request(false)).await.unwrap();
    input_tx.send(fixture_message_request(true)).await.unwrap();
    input_tx
        .send(ProtoGenerateMessagesRequest {
            messages: Vec::new(),
            ..fixture_message_request(true)
        })
        .await
        .unwrap();
    drop(input_tx);
    let input = async_stream::stream! {
        while let Some(request) = input_rx.recv().await {
            yield request;
        }
        input_closed_tx.send(()).unwrap();
    };
    let mut output = client
        .generate_messages_stream(Request::new(input))
        .await
        .unwrap()
        .into_inner();

    let first_text = output.message().await.unwrap().unwrap();
    let request_id = first_text.request_id;
    assert!(matches!(first_text.event, Some(MessageEvent::Text(_))));
    let first_completion = output.message().await.unwrap().unwrap();
    let Some(MessageEvent::Completion(first_completion)) = first_completion.event else {
        panic!("first message turn must complete")
    };
    assert_eq!(first_completion.num_input_tokens, 1);

    let second_completion = loop {
        let response = output.message().await.unwrap().unwrap();
        assert_eq!(response.request_id, request_id);
        if let Some(MessageEvent::Completion(completion)) = response.event {
            break completion;
        }
    };
    assert_eq!(second_completion.num_input_tokens, 4);
    assert_eq!(second_completion.reason, ProtoCompletionReason::ContextLimit as i32);
    assert!(output.message().await.unwrap().is_none());

    driver.await.unwrap();
    drop(output);
    drop(client);
    server_shutdown_tx.send(()).unwrap();
    server.await.unwrap().unwrap();
    runtime.shutdown();
}

fn new_runtime(context_window: usize) -> Arc<TestRuntime> {
    let shutdown = Shutdown::new();
    let async_task_handle = tokio::runtime::Handle::current();
    Arc::new(TestRuntime::new(
        RuntimeConfig {
            max_running_requests: 1,
            executor_hibernation_timeout: DEFAULT_EXECUTOR_HIBERNATION_TIMEOUT,
            executor_hibernation_mode: ExecutorHibernationMode::Selected,
            context_window,
            num_tokens_per_cache_block: 1024,
            num_pages: 64,
            cache_lanes: vec![CacheLaneRuntimeConfig {
                num_pages_per_kv_block: 64,
                num_pages_per_state_block: 0,
                block_cache_capacity: 1,
            }],
        },
        SchedulerConfig {
            max_requests: 1,
            max_tokens: 1024,
            max_tokens_per_request: 1024,
            max_compute_slots: 1,
        },
        0,
        shutdown,
        &async_task_handle,
        Arc::new(ResourceProcessors::new()),
    ))
}

fn drive_turns(runtime: Arc<TestRuntime>, input_closed_rx: Receiver<()>, second_prompt: Vec<Token>) {
    let request_rx = runtime.model_executor_request_rx();
    let response_tx = runtime.model_executor_response_tx();
    let mut session_identity = None;
    for (turn_index, sampled_token) in [101, 202].into_iter().enumerate() {
        let ReplayableModelExecutorRequest::Batch(mut batch) = request_rx.recv_timeout(Duration::from_secs(5)).unwrap()
        else {
            panic!("decode turn must submit one executor batch")
        };
        assert_eq!(batch.dev_reqs.len(), 1);
        let request = batch.dev_reqs.pop().unwrap();
        match session_identity {
            Some(identity) => assert_eq!((request.req_id, request.req_slot), identity),
            None => session_identity = Some((request.req_id, request.req_slot)),
        }
        let query_tokens = request.decoder_query_tokens;
        let QueryTokens::Decode {
            token_index,
            tokens,
            spec_tokens,
            ..
        } = &query_tokens
        else {
            panic!("decode turn must submit Decode query tokens")
        };
        assert!(spec_tokens.is_empty());
        if turn_index == 0 {
            assert_eq!((*token_index, tokens.as_slice()), (0, &[Token::new(1)][..]));
        } else {
            let expected = std::iter::once(Token::new(101))
                .chain(second_prompt.iter().copied())
                .collect::<Vec<_>>();
            assert_eq!((*token_index, tokens.as_slice()), (1, expected.as_slice()));
        }
        if turn_index == 0 {
            input_closed_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        let probability = NotNan::new(1.0).unwrap();
        let response = DeviceResponse {
            req_id: request.req_id,
            sampled_tokens: SampledTokens::Decode {
                epoch: query_tokens.epoch(),
                validated_tokens: Vec::new(),
                validated_probs: Vec::new(),
                sampled_token: Token::new(sampled_token),
                sampled_prob: probability,
                spec_tokens: Vec::new(),
                spec_probs: Vec::new(),
                spec_confidences: Vec::new(),
            },
            query_tokens,
        };
        response_tx
            .send(ReplayableModelExecutorResponse::Batch(BatchDeviceResponse::new(
                batch.seq,
                [response],
            )))
            .unwrap();
    }
}

fn fixture_request(token: u32) -> ProtoGenerateTokensRequest {
    ProtoGenerateTokensRequest {
        tokens: vec![token],
        generation: Some(ProtoGenerationConfig {
            max_sampled_tokens: 1,
            stop_sequences: Vec::new(),
            temperature: None,
            top_k: None,
            top_p: None,
            seed: Some(token),
        }),
    }
}

fn fixture_message_request(continuation: bool) -> ProtoGenerateMessagesRequest {
    let mut messages = vec![ProtoInputMessage {
        message: Some(ProtoMessage::User(ProtoTextMessage {
            text: "one".to_string(),
        })),
    }];
    if continuation {
        messages = vec![ProtoInputMessage {
            message: Some(ProtoMessage::User(ProtoTextMessage {
                text: "two".to_string(),
            })),
        }];
    }
    ProtoGenerateMessagesRequest {
        messages,
        tools: None,
        generation: Some(ProtoGenerationConfig {
            max_sampled_tokens: 1,
            stop_sequences: Vec::new(),
            temperature: None,
            top_k: None,
            top_p: None,
            seed: Some(if continuation { 2 } else { 1 }),
        }),
        enable_thinking: false,
        reasoning_effort: None,
    }
}

fn fixture_message_codec() -> QwenCodec {
    let template = ChatTemplate::from_str(concat!(
        "{% for message in messages %}",
        "{% if message.role == 'assistant' %}{{ message.content }}<|im_end|>",
        "{% else %}{{ message.content }}{% endif %}",
        "{% endfor %}",
    ))
    .unwrap();
    let model = WordLevel::builder()
        .vocab(
            [
                ("[UNK]".to_string(), 0),
                ("one".to_string(), 1),
                ("two".to_string(), 2),
                ("</think>".to_string(), 3),
                ("<|im_end|>".to_string(), 4),
                ("answer".to_string(), 101),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
        .add_special_tokens([AddedToken::from("</think>", true), AddedToken::from("<|im_end|>", true)])
        .unwrap();
    QwenCodec::new(template, Arc::new(HFTokenizer::new(tokenizer))).unwrap()
}
