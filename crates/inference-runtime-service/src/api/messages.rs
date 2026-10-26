use std::collections::HashMap;
use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;

use futures_util::Stream;
use hf_chat_template::Message;
use inference_runtime_core::Error;
use inference_runtime_core::Result;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::runtime::CompletionReason;
use inference_runtime_core::runtime::RawRequestID;
use serde_json::Value;
use uuid::Uuid;

use crate::api::Inference;
use crate::api::decode::DecodeEvent;
use crate::api::decode::DecodeRequest;
use crate::api::decode::DecodeSession;
use crate::codec::qwen::QwenCodec;
use crate::codec::qwen::ResponseEvent;
use crate::tool::ToolArguments;
use crate::tool::ToolCallID;
use crate::tool::ToolDefinition;
use crate::tool::ToolID;

#[derive(Clone)]
pub struct MessageGenerator<const N: usize, const L: usize, const P: usize> {
    inference: Arc<Inference<N, L, P>>,
    codec: Arc<QwenCodec>,
}

impl<const N: usize, const L: usize, const P: usize> MessageGenerator<N, L, P> {
    pub fn new(inference: Arc<Inference<N, L, P>>, codec: Arc<QwenCodec>) -> Self {
        Self { inference, codec }
    }

    pub fn create_session(&self, request: MessageRequest) -> Result<MessageSession<N, L, P>> {
        validate_history(&request.messages)?;
        let tools = request.tools.apply(&[]);
        let tokens = self.codec.encode(
            request.messages,
            &tools,
            request.enable_thinking,
            request.reasoning_effort.as_deref(),
            true,
        )?;
        let num_input_tokens = tokens.len();
        let decode_request = DecodeRequest::new(tokens, vec![], request.sampling)?;
        let decode_session = self.inference.create_session(decode_request)?;
        let request_id = decode_session.request_id();
        Ok(MessageSession {
            request_id,
            decode_session: Arc::new(tokio::sync::Mutex::new(decode_session)),
            codec: self.codec.clone(),
            completed_turn: Arc::new(Mutex::new(None)),
            tools,
            enable_thinking: request.enable_thinking,
            reasoning_effort: request.reasoning_effort,
            num_input_tokens,
        })
    }

    pub async fn resume_session(&self, session: &mut MessageSession<N, L, P>, request: MessageRequest) -> Result<()> {
        if !request.tools.is_empty() {
            return Err(Error::invalid_argument(
                "a resumed Qwen message session cannot change tools; start a new stream",
            ));
        }
        if request.enable_thinking != session.enable_thinking || request.reasoning_effort != session.reasoning_effort {
            return Err(Error::invalid_argument(
                "a resumed Qwen message session cannot change the generation mode; start a new stream",
            ));
        }
        let previous_turn_closed = {
            let completed_turn = session.completed_turn.lock().unwrap();
            let completed_turn = completed_turn
                .as_ref()
                .expect("a message session can resume only after its response stream completes");
            validate_continuation(&request.messages, &completed_turn.pending_tool_calls)?;
            completed_turn.closed
        };
        let prompt_tokens = self.codec.encode_continuation(
            request.messages,
            request.enable_thinking,
            request.reasoning_effort.as_deref(),
            previous_turn_closed,
        )?;
        let mut decode_session = session.decode_session.lock().await;
        let num_input_tokens = decode_session.num_history_tokens() + prompt_tokens.len();
        let decode_request = DecodeRequest::new(prompt_tokens, vec![], request.sampling)?;
        self.inference
            .resume_session(&mut decode_session, decode_request)
            .await?;
        drop(decode_session);

        *session.completed_turn.lock().unwrap() = None;
        session.num_input_tokens = num_input_tokens;
        Ok(())
    }
}

pub struct MessageRequest {
    messages: Vec<Message>,
    tools: ToolDelta,
    sampling: SamplingConfig,
    enable_thinking: bool,
    reasoning_effort: Option<String>,
}

impl MessageRequest {
    pub fn new(
        messages: Vec<Message>,
        tools: ToolDelta,
        sampling: SamplingConfig,
        enable_thinking: bool,
        reasoning_effort: Option<String>,
    ) -> Result<Self> {
        if messages.is_empty() {
            return Err(Error::invalid_argument(
                "message request must include at least one message",
            ));
        }
        if reasoning_effort
            .as_deref()
            .is_some_and(|effort| !matches!(effort, "low" | "medium" | "xhigh"))
        {
            return Err(Error::invalid_argument(
                "reasoning_effort must be low, medium, or xhigh",
            ));
        }
        if !enable_thinking && reasoning_effort.is_some() {
            return Err(Error::invalid_argument(
                "reasoning_effort requires enable_thinking true",
            ));
        }
        Ok(Self {
            messages,
            tools,
            sampling,
            enable_thinking,
            reasoning_effort,
        })
    }
}

pub struct ToolDelta {
    insert: Vec<ToolDefinition>,
    remove: Vec<ToolID>,
}

impl ToolDelta {
    pub fn new(insert: Vec<ToolDefinition>, remove: Vec<ToolID>) -> Result<Self> {
        let mut insert_ids = HashSet::with_capacity(insert.len());
        for definition in &insert {
            if !insert_ids.insert(definition.tool_id().clone()) {
                return Err(Error::invalid_argument(format!(
                    "tool delta contains duplicate insert ID {:?}",
                    definition.tool_id().as_str()
                )));
            }
        }
        let mut remove_ids = HashSet::with_capacity(remove.len());
        for tool_id in &remove {
            if !remove_ids.insert(tool_id.clone()) {
                return Err(Error::invalid_argument(format!(
                    "tool delta contains duplicate remove ID {:?}",
                    tool_id.as_str()
                )));
            }
            if insert_ids.contains(tool_id) {
                return Err(Error::invalid_argument(format!(
                    "tool delta cannot insert and remove ID {:?} in one request",
                    tool_id.as_str()
                )));
            }
        }
        Ok(Self { insert, remove })
    }

    fn apply(&self, current: &[ToolDefinition]) -> Vec<ToolDefinition> {
        let mut tools = current
            .iter()
            .filter(|definition| !self.remove.contains(definition.tool_id()))
            .cloned()
            .collect::<Vec<_>>();
        for definition in &self.insert {
            match tools
                .iter_mut()
                .find(|current| current.tool_id() == definition.tool_id())
            {
                Some(current) => *current = definition.clone(),
                None => tools.push(definition.clone()),
            }
        }
        tools
    }

    fn is_empty(&self) -> bool {
        self.insert.is_empty() && self.remove.is_empty()
    }
}

pub struct MessageSession<const N: usize, const L: usize, const P: usize> {
    request_id: RawRequestID,
    decode_session: Arc<tokio::sync::Mutex<DecodeSession<N, L, P>>>,
    codec: Arc<QwenCodec>,
    completed_turn: Arc<Mutex<Option<CompletedTurn>>>,
    tools: Vec<ToolDefinition>,
    enable_thinking: bool,
    reasoning_effort: Option<String>,
    num_input_tokens: usize,
}

impl<const N: usize, const L: usize, const P: usize> MessageSession<N, L, P> {
    pub fn request_id(&self) -> RawRequestID {
        self.request_id
    }

    pub fn num_input_tokens(&self) -> usize {
        self.num_input_tokens
    }

    pub fn response_stream(&self) -> Pin<Box<dyn Stream<Item = Result<MessageEvent>> + Send + 'static>> {
        let decode_session = self.decode_session.clone();
        let response = async_stream::stream! {
            loop {
                let event = decode_session.lock().await.recv_event().await;
                let turn_completed = matches!(event, Ok(DecodeEvent::Completed { .. }));
                yield event;
                if turn_completed {
                    break;
                }
            }
        };
        let tool_ids = self.tools.iter().map(|tool| tool.tool_id().clone()).collect();
        let mut response = Box::pin(self.codec.decode(response, tool_ids, self.enable_thinking));
        let completed_turn = self.completed_turn.clone();
        Box::pin(async_stream::try_stream! {
            let mut turn_tool_calls = HashMap::new();
            while let Some(event) = futures_util::StreamExt::next(&mut response).await {
                match event? {
                    ResponseEvent::Thinking(text) => {
                        yield MessageEvent::Reasoning(text);
                    },
                    ResponseEvent::Text(text) => {
                        yield MessageEvent::Text(text);
                    },
                    ResponseEvent::ToolCall(call) => {
                        let call = MessageToolCall {
                            id: ToolCallID::new(Uuid::new_v4().to_string())
                                .expect("UUID tool call ID must be valid"),
                            tool_id: call.tool_id().clone(),
                            arguments: call.arguments().clone(),
                        };
                        assert!(
                            turn_tool_calls.insert(call.id.clone(), call.tool_id.clone()).is_none(),
                            "generated tool call IDs must be unique",
                        );
                        yield MessageEvent::ToolCall(call);
                    },
                    ResponseEvent::Completed {
                        reason,
                        num_output_tokens,
                        turn_closed,
                    } => {
                        assert!(
                            completed_turn
                                .lock()
                                .unwrap()
                                .replace(CompletedTurn {
                                    closed: turn_closed,
                                    pending_tool_calls: std::mem::take(&mut turn_tool_calls),
                                })
                                .is_none(),
                            "a message turn cannot complete twice",
                        );
                        yield MessageEvent::Completed { reason, num_output_tokens };
                    },
                }
            }
        })
    }

    pub async fn wait_for_session_end(&self) -> Error {
        self.decode_session.lock().await.wait_for_session_end().await
    }
}

struct CompletedTurn {
    closed: bool,
    pending_tool_calls: HashMap<ToolCallID, ToolID>,
}

#[derive(Debug)]
pub enum MessageEvent {
    Reasoning(String),
    Text(String),
    ToolCall(MessageToolCall),
    Completed {
        reason: CompletionReason,
        num_output_tokens: usize,
    },
}

#[derive(Clone, Debug)]
pub struct MessageToolCall {
    id: ToolCallID,
    tool_id: ToolID,
    arguments: ToolArguments,
}

impl MessageToolCall {
    pub fn id(&self) -> &ToolCallID {
        &self.id
    }

    pub fn tool_id(&self) -> &ToolID {
        &self.tool_id
    }

    pub fn arguments(&self) -> &ToolArguments {
        &self.arguments
    }
}

fn validate_history(messages: &[Message]) -> Result<()> {
    let mut seen_call_ids = HashSet::new();
    let mut pending_calls = HashMap::new();
    for (index, message) in messages.iter().enumerate() {
        if message.role != "tool" && !pending_calls.is_empty() {
            return Err(Error::invalid_argument(
                "every assistant tool call must have a result before the next turn",
            ));
        }
        match message.role.as_str() {
            "system" => {
                if index != 0 {
                    return Err(Error::invalid_argument("system message must be first"));
                }
            },
            "user" => {},
            "assistant" => {
                for call in &message.tool_calls {
                    let call_id = call
                        .get("id")
                        .and_then(Value::as_str)
                        .ok_or_else(|| Error::invalid_argument("assistant tool call must include an ID"))?;
                    let call_id = ToolCallID::new(call_id)?;
                    if !seen_call_ids.insert(call_id.clone()) {
                        return Err(Error::invalid_argument("tool call IDs must be unique"));
                    }
                    let tool_id = call
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| Error::invalid_argument("assistant tool call must include a tool ID"))?;
                    pending_calls.insert(call_id, ToolID::new(tool_id)?);
                }
            },
            "tool" => {
                let call_id = message
                    .extra
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::invalid_argument("tool result must include a tool call ID"))?;
                let call_id = ToolCallID::new(call_id)?;
                let expected_tool_id = pending_calls
                    .remove(&call_id)
                    .ok_or_else(|| Error::invalid_argument("tool result must match a preceding tool call"))?;
                if let Some(tool_id) = message.extra.get("name").and_then(Value::as_str)
                    && ToolID::new(tool_id)? != expected_tool_id
                {
                    return Err(Error::invalid_argument("tool result tool ID must match its tool call"));
                }
            },
            role => return Err(Error::invalid_argument(format!("unsupported message role {role:?}"))),
        }
    }
    if !pending_calls.is_empty() {
        return Err(Error::invalid_argument(
            "every assistant tool call must have a result before generation",
        ));
    }
    if matches!(messages.last(), Some(Message { role, .. }) if role == "system" || role == "assistant") {
        return Err(Error::invalid_argument(
            "conversation must end with a user or tool-result message",
        ));
    }
    Ok(())
}

fn validate_continuation(messages: &[Message], pending_calls: &HashMap<ToolCallID, ToolID>) -> Result<()> {
    let mut pending_calls = pending_calls.clone();
    for message in messages {
        if message.role != "tool" && !pending_calls.is_empty() {
            return Err(Error::invalid_argument(
                "every assistant tool call must have a result before the next user message",
            ));
        }
        match message.role.as_str() {
            "user" => {},
            "tool" => {
                let call_id = message
                    .extra
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::invalid_argument("tool result must include a tool call ID"))?;
                let call_id = ToolCallID::new(call_id)?;
                let expected_tool_id = pending_calls
                    .remove(&call_id)
                    .ok_or_else(|| Error::invalid_argument("tool result must match a pending tool call"))?;
                if let Some(tool_id) = message.extra.get("name").and_then(Value::as_str)
                    && ToolID::new(tool_id)? != expected_tool_id
                {
                    return Err(Error::invalid_argument("tool result tool ID must match its tool call"));
                }
            },
            _ => {
                return Err(Error::invalid_argument(
                    "message continuation accepts only new user and tool-result messages",
                ));
            },
        }
    }
    if !pending_calls.is_empty() {
        return Err(Error::invalid_argument(
            "every pending tool call must have a result before generation",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use hf_chat_template::Message;
    use serde_json::Map;
    use serde_json::Value;
    use serde_json::json;

    use super::ToolDelta;
    use super::validate_continuation;
    use crate::tool::ToolCallID;
    use crate::tool::ToolDefinition;
    use crate::tool::ToolID;
    use crate::tool::ToolInputSchema;

    #[test]
    fn test_tool_delta_overwrites_and_removes() {
        let old = definition("read", "old");
        let delta = ToolDelta::new(vec![definition("read", "new"), definition("write", "new")], vec![]).unwrap();
        let tools = delta.apply(&[old]);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].description(), Some("new"));

        let delta = ToolDelta::new(Vec::new(), vec![ToolID::new("read").unwrap()]).unwrap();
        assert_eq!(delta.apply(&tools), vec![definition("write", "new")]);
    }

    #[test]
    fn test_tool_delta_rejects_ambiguous_ids() {
        let result = ToolDelta::new(vec![definition("read", "new")], vec![ToolID::new("read").unwrap()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_continuation_consumes_pending_tool_calls() {
        let call_id = ToolCallID::new("call-1").unwrap();
        let tool_id = ToolID::new("read").unwrap();
        let pending_calls = HashMap::from([(call_id.clone(), tool_id.clone())]);
        let mut extra = Map::new();
        extra.insert("tool_call_id".to_string(), Value::String(call_id.into_string()));
        extra.insert("name".to_string(), Value::String(tool_id.into_string()));
        let tool_result = Message {
            role: "tool".to_string(),
            content: None,
            tool_calls: Vec::new(),
            extra,
        };

        assert!(validate_continuation(&[tool_result], &pending_calls).is_ok());
        assert!(validate_continuation(&[Message::user("next")], &pending_calls).is_err());
    }

    fn definition(id: &str, description: &str) -> ToolDefinition {
        ToolDefinition::new(
            ToolID::new(id).unwrap(),
            Some(description.to_string()),
            ToolInputSchema::new(json!({"type": "object"})).unwrap(),
        )
    }
}
