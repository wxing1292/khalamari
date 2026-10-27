//! gRPC generation RPC handlers.

use std::collections::HashSet;
use std::pin::Pin;

use futures_util::StreamExt;
use hf_chat_template::Content as HFContent;
use hf_chat_template::Message as HFMessage;
use inference_runtime_core::Result as RuntimeResult;
use inference_runtime_core::config::DEFAULT_SAMPLING_TEMPERATURE;
use inference_runtime_core::config::DEFAULT_SAMPLING_TOP_K;
use inference_runtime_core::config::DEFAULT_SAMPLING_TOP_P;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::runtime::CompletionReason;
use inference_runtime_core::runtime::Token;
use inference_runtime_proto::inference_runtime_service::AudioStreamReq;
use inference_runtime_proto::inference_runtime_service::AudioStreamResp;
use inference_runtime_proto::inference_runtime_service::AudioTranscribeReq as ProtoAudioTranscribeReq;
use inference_runtime_proto::inference_runtime_service::AudioTranscribeResp as ProtoAudioTranscribeResp;
use inference_runtime_proto::inference_runtime_service::CompletionReason as ProtoCompletionReason;
use inference_runtime_proto::inference_runtime_service::GenerateMessagesRequest as ProtoGenerateMessagesRequest;
use inference_runtime_proto::inference_runtime_service::GenerateMessagesResponse as ProtoGenerateMessagesResponse;
use inference_runtime_proto::inference_runtime_service::GenerateTokensRequest as ProtoGenerateTokensRequest;
use inference_runtime_proto::inference_runtime_service::GenerateTokensResponse as ProtoGenerateTokensResponse;
use inference_runtime_proto::inference_runtime_service::GenerationCompletion as ProtoGenerationCompletion;
use inference_runtime_proto::inference_runtime_service::GenerationConfig as ProtoGenerationConfig;
use inference_runtime_proto::inference_runtime_service::ReasoningEffort as ProtoReasoningEffort;
use inference_runtime_proto::inference_runtime_service::TextChunk;
use inference_runtime_proto::inference_runtime_service::TokenChunk;
use inference_runtime_proto::inference_runtime_service::ToolCall as ProtoToolCall;
use inference_runtime_proto::inference_runtime_service::generate_messages_response::Event as ProtoMessageEvent;
use inference_runtime_proto::inference_runtime_service::generate_tokens_response::Event as TokenEvent;
use inference_runtime_proto::inference_runtime_service::inference_runtime_server::InferenceRuntime;
use inference_runtime_proto::inference_runtime_service::input_message::Message as ProtoMessage;
use inference_runtime_proto::inference_runtime_service::message_content::Content as ProtoContent;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;
use tokio_stream::Stream;
use tonic::Request;
use tonic::Response;
use tonic::Status;
use tonic::Streaming;

use crate::api::audio::AudioTranscribeReq;
use crate::api::decode::DecodeEvent;
use crate::api::decode::DecodeRequest;
use crate::api::messages::MessageEvent;
use crate::api::messages::MessageRequest;
use crate::api::messages::MessageSession;
use crate::api::messages::ToolDelta;
use crate::rpc::grpc::GRPCServer;
use crate::rpc::grpc::map_error;
use crate::tool::ToolArguments;
use crate::tool::ToolCallID;
use crate::tool::ToolDefinition;
use crate::tool::ToolID;
use crate::tool::ToolInputSchema;

type TokenResponseStream = Pin<Box<dyn Stream<Item = Result<ProtoGenerateTokensResponse, Status>> + Send>>;
type MessageResponseStream = Pin<Box<dyn Stream<Item = Result<ProtoGenerateMessagesResponse, Status>> + Send>>;

#[async_trait::async_trait]
impl<const N: usize, const L: usize, const P: usize> InferenceRuntime for GRPCServer<N, L, P> {
    type GenerateMessagesStream = MessageResponseStream;
    type GenerateMessagesStreamStream = MessageResponseStream;
    type GenerateTokensStream = TokenResponseStream;
    type GenerateTokensStreamStream = TokenResponseStream;
    type StreamAudioStream = Pin<Box<dyn Stream<Item = Result<AudioStreamResp, Status>> + Send>>;

    async fn stream_audio(
        &self,
        request: Request<Streaming<AudioStreamReq>>,
    ) -> Result<Response<Self::StreamAudioStream>, Status> {
        let transcriber = self
            .audio_transcriber
            .as_ref()
            .ok_or_else(|| Status::unimplemented("audio transcription is not available for this model"))?;
        let input = request.into_inner().map(|request| {
            request.map_err(|error| inference_runtime_core::Error::unavailable(format!("audio stream input: {error}")))
        });
        let output = crate::rpc::audio_stream::responses(transcriber.clone(), input, self.shutdown.clone())
            .map(|response| response.map_err(map_error));
        Ok(Response::new(Box::pin(output)))
    }

    async fn transcribe_audio(
        &self,
        request: Request<ProtoAudioTranscribeReq>,
    ) -> Result<Response<ProtoAudioTranscribeResp>, Status> {
        let transcriber = self
            .audio_transcriber
            .as_ref()
            .ok_or_else(|| Status::unimplemented("audio transcription is not available for this model"))?;
        let request = request.into_inner();
        let request = AudioTranscribeReq::new(request.wav, &request.model, request.language, request.context)
            .map_err(map_error)?;
        let result = transcriber.transcribe(request).await.map_err(map_error)?;
        Ok(Response::new(ProtoAudioTranscribeResp {
            text: result.text,
            language: result.language,
        }))
    }

    async fn generate_tokens(
        &self,
        request: Request<ProtoGenerateTokensRequest>,
    ) -> Result<Response<Self::GenerateTokensStream>, Status> {
        let request = request.into_inner();
        let num_input_tokens = request.tokens.len();
        let request = map_token_request(request)?;
        let session = self.inference.create_session(request).map_err(map_error)?;
        Ok(Response::new(token_response_stream(session, num_input_tokens)))
    }

    async fn generate_tokens_stream(
        &self,
        request: Request<Streaming<ProtoGenerateTokensRequest>>,
    ) -> Result<Response<Self::GenerateTokensStreamStream>, Status> {
        let inference = self.inference.clone();
        let mut requests = request.into_inner();
        let first_request = requests
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("token stream requires at least one request"))?;
        let mut num_input_tokens = first_request.tokens.len();
        let mut session = inference
            .create_session(map_token_request(first_request)?)
            .map_err(map_error)?;
        let responses = async_stream::try_stream! {
            loop {
                let reason = loop {
                    let event = session.recv_event().await;
                    let completion_reason = match &event {
                        Ok(DecodeEvent::Completed { reason, .. }) => Some(*reason),
                        _ => None,
                    };
                    yield map_token_response(session.request_id() as u64, num_input_tokens, event)?;
                    if let Some(reason) = completion_reason {
                        break reason;
                    }
                };
                if reason == CompletionReason::ContextLimit {
                    break;
                }

                // An idle stream resumes with client input or ends when the resident session terminates.
                let next_request = tokio::select! {
                    next_request = requests.message() => next_request,
                    session_error = session.wait_for_session_end() => Err(map_error(session_error)),
                }?;
                match next_request {
                    Some(request) => {
                        let num_history_tokens = session.num_history_tokens();
                        num_input_tokens = num_history_tokens + request.tokens.len();
                        inference
                            .resume_session(&mut session, map_token_request(request)?)
                            .await
                            .map_err(map_error)?;
                    },
                    None => break,
                }
            }
        };
        Ok(Response::new(Box::pin(responses)))
    }

    async fn generate_messages(
        &self,
        request: Request<ProtoGenerateMessagesRequest>,
    ) -> Result<Response<Self::GenerateMessagesStream>, Status> {
        let generator = self
            .message_generator
            .as_ref()
            .ok_or_else(|| Status::unimplemented("message generation is not available for this model"))?;
        let session = generator
            .create_session(map_message_request(request.into_inner())?)
            .map_err(map_error)?;
        Ok(Response::new(message_response_stream(session)))
    }

    async fn generate_messages_stream(
        &self,
        request: Request<Streaming<ProtoGenerateMessagesRequest>>,
    ) -> Result<Response<Self::GenerateMessagesStreamStream>, Status> {
        let generator = self
            .message_generator
            .clone()
            .ok_or_else(|| Status::unimplemented("message generation is not available for this model"))?;
        let mut requests = request.into_inner();
        let first_request = requests
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("message stream requires at least one request"))?;
        let mut session = generator
            .create_session(map_message_request(first_request)?)
            .map_err(map_error)?;
        let responses = async_stream::try_stream! {
            loop {
                let request_id = session.request_id() as u64;
                let num_input_tokens = session.num_input_tokens();
                let mut turn = session.response_stream();
                let reason = loop {
                    let event = turn.next().await.expect("message response stream must complete each turn");
                    let completion_reason = match &event {
                        Ok(MessageEvent::Completed { reason, .. }) => Some(*reason),
                        _ => None,
                    };
                    yield map_message_response(request_id, num_input_tokens, event)?;
                    if let Some(reason) = completion_reason {
                        break reason;
                    }
                };
                if reason == CompletionReason::ContextLimit {
                    break;
                }

                // An idle stream resumes with client input or ends when the resident session terminates.
                let next_request = tokio::select! {
                    next_request = requests.message() => next_request,
                    session_error = session.wait_for_session_end() => Err(map_error(session_error)),
                }?;
                match next_request {
                    Some(request) => {
                        generator
                            .resume_session(&mut session, map_message_request(request)?)
                            .await
                            .map_err(map_error)?;
                    },
                    None => break,
                }
            }
        };
        Ok(Response::new(Box::pin(responses)))
    }
}

fn token_response_stream<const N: usize, const L: usize, const P: usize>(
    mut session: crate::api::decode::DecodeSession<N, L, P>,
    num_input_tokens: usize,
) -> TokenResponseStream {
    let request_id = session.request_id() as u64;
    Box::pin(async_stream::try_stream! {
        loop {
            let event = session.recv_event().await;
            let turn_completed = matches!(event, Ok(DecodeEvent::Completed { .. }));
            yield map_token_response(request_id, num_input_tokens, event)?;
            if turn_completed {
                break;
            }
        }
    })
}

fn message_response_stream<const N: usize, const L: usize, const P: usize>(
    session: MessageSession<N, L, P>,
) -> MessageResponseStream {
    let request_id = session.request_id() as u64;
    let num_input_tokens = session.num_input_tokens();
    let response = session.response_stream();
    Box::pin(response.map(move |event| map_message_response(request_id, num_input_tokens, event)))
}

fn map_token_request(request: ProtoGenerateTokensRequest) -> Result<DecodeRequest, Status> {
    let sampling = map_generation(request.generation)?;
    DecodeRequest::new(request.tokens.into_iter().map(Token::new).collect(), vec![], sampling).map_err(map_error)
}

fn map_message_request(request: ProtoGenerateMessagesRequest) -> Result<MessageRequest, Status> {
    let sampling = map_generation(request.generation)?;
    let messages = map_messages(request.messages)?;
    let tools = map_tool_delta(request.tools)?;
    let reasoning_effort = request
        .reasoning_effort
        .map(|effort| {
            let effort = ProtoReasoningEffort::try_from(effort)
                .map_err(|_| Status::invalid_argument("reasoning_effort is invalid"))?;
            match effort {
                ProtoReasoningEffort::Low => Ok("low".to_string()),
                ProtoReasoningEffort::Medium => Ok("medium".to_string()),
                ProtoReasoningEffort::High | ProtoReasoningEffort::Xhigh => Ok("xhigh".to_string()),
                ProtoReasoningEffort::Unspecified => {
                    Err(Status::invalid_argument("reasoning_effort must not be unspecified"))
                },
            }
        })
        .transpose()?;
    MessageRequest::new(messages, tools, sampling, request.enable_thinking, reasoning_effort).map_err(map_error)
}

fn map_generation(generation: Option<ProtoGenerationConfig>) -> Result<SamplingConfig, Status> {
    let generation = generation.ok_or_else(|| Status::invalid_argument("generation config is required"))?;
    Ok(SamplingConfig {
        max_sampled_tokens: generation.max_sampled_tokens as usize,
        temperature: generation.temperature.unwrap_or(DEFAULT_SAMPLING_TEMPERATURE),
        top_k: generation.top_k.unwrap_or(DEFAULT_SAMPLING_TOP_K as u32) as usize,
        top_p: generation.top_p.unwrap_or(DEFAULT_SAMPLING_TOP_P),
        seed: generation.seed,
        stop_sequences: generation
            .stop_sequences
            .into_iter()
            .map(|sequence| sequence.tokens.into_iter().map(Token::new).collect())
            .collect(),
    })
}

fn map_tool_delta(
    delta: Option<inference_runtime_proto::inference_runtime_service::ToolDelta>,
) -> Result<ToolDelta, Status> {
    let Some(delta) = delta else {
        return ToolDelta::new(Vec::new(), Vec::new()).map_err(map_error);
    };
    let insert = delta
        .insert
        .into_iter()
        .map(|definition| {
            let tool_id = ToolID::new(definition.id).map_err(map_error)?;
            let input_schema = serde_json::from_str::<Value>(&definition.input_schema_json)
                .map_err(|_| Status::invalid_argument("tool input_schema_json must be valid JSON"))?;
            let input_schema = ToolInputSchema::new(input_schema).map_err(map_error)?;
            Ok(ToolDefinition::new(tool_id, definition.description, input_schema))
        })
        .collect::<Result<Vec<_>, Status>>()?;
    let remove = delta
        .remove
        .into_iter()
        .map(|tool_id| ToolID::new(tool_id).map_err(map_error))
        .collect::<Result<Vec<_>, _>>()?;
    ToolDelta::new(insert, remove).map_err(map_error)
}

fn map_messages(
    messages: Vec<inference_runtime_proto::inference_runtime_service::InputMessage>,
) -> Result<Vec<HFMessage>, Status> {
    if messages.is_empty() {
        return Err(Status::invalid_argument("messages must not be empty"));
    }
    let mut seen_call_ids = HashSet::new();
    let num_messages = messages.len();
    let mut converted = Vec::with_capacity(num_messages);
    for (index, message) in messages.into_iter().enumerate() {
        let message = message
            .message
            .ok_or_else(|| Status::invalid_argument("message envelope must not be empty"))?;
        let message = match message {
            ProtoMessage::System(message) => {
                if index != 0 {
                    return Err(Status::invalid_argument("system message must be first"));
                }
                if message.text.is_empty() {
                    return Err(Status::invalid_argument("system message must not be empty"));
                }
                HFMessage::system(message.text)
            },
            ProtoMessage::User(message) => {
                if message.text.is_empty() {
                    return Err(Status::invalid_argument("user message must not be empty"));
                }
                HFMessage::user(message.text)
            },
            ProtoMessage::Assistant(message) => {
                let mut text = String::new();
                let mut reasoning = String::new();
                for content in message.content {
                    match content.content {
                        Some(ProtoContent::Text(chunk)) => text.push_str(&chunk),
                        Some(ProtoContent::Reasoning(chunk)) => reasoning.push_str(&chunk),
                        None => return Err(Status::invalid_argument("message content envelope must not be empty")),
                    }
                }
                if text.is_empty() && reasoning.is_empty() && message.tool_calls.is_empty() {
                    return Err(Status::invalid_argument(
                        "assistant message must include content or tool calls",
                    ));
                }
                let mut tool_calls = Vec::with_capacity(message.tool_calls.len());
                for call in message.tool_calls {
                    let call_id = ToolCallID::new(call.id).map_err(map_error)?;
                    if !seen_call_ids.insert(call_id.clone()) {
                        return Err(Status::invalid_argument("tool call IDs must be unique"));
                    }
                    let tool_id = ToolID::new(call.tool_id).map_err(map_error)?;
                    let arguments = serde_json::from_str::<Value>(&call.arguments_json)
                        .map_err(|_| Status::invalid_argument("tool call arguments_json must be valid JSON"))?;
                    let arguments = ToolArguments::new(arguments).map_err(map_error)?.into_value();
                    tool_calls.push(json!({
                        "id": call_id.as_str(),
                        "type": "function",
                        "function": { "name": tool_id.as_str(), "arguments": arguments },
                    }));
                }
                let mut extra = Map::new();
                if !reasoning.is_empty() {
                    extra.insert("reasoning_content".to_string(), Value::String(reasoning));
                }
                HFMessage {
                    role: "assistant".to_string(),
                    content: (!text.is_empty()).then_some(HFContent::Text(text)),
                    tool_calls,
                    extra,
                }
            },
            ProtoMessage::ToolResult(message) => {
                let call_id = ToolCallID::new(message.tool_call_id).map_err(map_error)?;
                let tool_id = ToolID::new(message.tool_id).map_err(map_error)?;
                let mut extra = Map::new();
                extra.insert("tool_call_id".to_string(), Value::String(call_id.into_string()));
                extra.insert("name".to_string(), Value::String(tool_id.into_string()));
                extra.insert("is_error".to_string(), Value::Bool(message.is_error));
                HFMessage {
                    role: "tool".to_string(),
                    content: Some(HFContent::Text(message.text.concat())),
                    tool_calls: Vec::new(),
                    extra,
                }
            },
        };
        converted.push(message);
    }
    if matches!(converted.last(), Some(HFMessage { role, .. }) if role == "system" || role == "assistant") {
        return Err(Status::invalid_argument(
            "conversation must end with a user or tool-result message",
        ));
    }
    Ok(converted)
}

fn map_token_response(
    request_id: u64,
    num_input_tokens: usize,
    event: RuntimeResult<DecodeEvent>,
) -> Result<ProtoGenerateTokensResponse, Status> {
    let event = match event.map_err(map_error)? {
        DecodeEvent::TokenProbs(token_probs) => {
            TokenEvent::Chunk(TokenChunk {
                tokens: token_probs.tokens.into_iter().map(Token::value).collect(),
                probs: token_probs.probs.into_iter().map(|prob| prob.into_inner()).collect(),
            })
        },
        DecodeEvent::Completed {
            reason,
            num_output_tokens,
        } => TokenEvent::Completion(map_completion(reason, num_input_tokens, num_output_tokens)),
    };
    Ok(ProtoGenerateTokensResponse {
        request_id,
        event: Some(event),
    })
}

fn map_message_response(
    request_id: u64,
    num_input_tokens: usize,
    event: RuntimeResult<MessageEvent>,
) -> Result<ProtoGenerateMessagesResponse, Status> {
    let event = match event.map_err(map_error)? {
        MessageEvent::Reasoning(text) => ProtoMessageEvent::Reasoning(TextChunk { text }),
        MessageEvent::Text(text) => ProtoMessageEvent::Text(TextChunk { text }),
        MessageEvent::ToolCall(call) => {
            ProtoMessageEvent::ToolCall(ProtoToolCall {
                id: call.id().as_str().to_string(),
                tool_id: call.tool_id().as_str().to_string(),
                arguments_json: call.arguments().as_value().to_string(),
            })
        },
        MessageEvent::Completed {
            reason,
            num_output_tokens,
        } => ProtoMessageEvent::Completion(map_completion(reason, num_input_tokens, num_output_tokens)),
    };
    Ok(ProtoGenerateMessagesResponse {
        request_id,
        event: Some(event),
    })
}

fn map_completion(
    reason: CompletionReason,
    num_input_tokens: usize,
    num_output_tokens: usize,
) -> ProtoGenerationCompletion {
    let reason = match reason {
        CompletionReason::StopSequence => ProtoCompletionReason::StopSequence,
        CompletionReason::LengthLimit => ProtoCompletionReason::LengthLimit,
        CompletionReason::ContextLimit => ProtoCompletionReason::ContextLimit,
    };
    ProtoGenerationCompletion {
        reason: reason as i32,
        num_input_tokens: num_input_tokens as u64,
        num_output_tokens: num_output_tokens as u64,
    }
}

#[cfg(test)]
#[path = "./generation_test.rs"]
mod generation_test;
