use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use async_stream::try_stream;
use futures_util::Stream;
use futures_util::StreamExt;
use hf_chat_template::ChatTemplate;
use hf_chat_template::Message;
use hf_chat_template::RenderInput;
use inference_runtime_core::Error;
use inference_runtime_core::Result;
use inference_runtime_core::chat_template;
use inference_runtime_core::runtime::CompletionReason;
use inference_runtime_core::runtime::Token;
use inference_runtime_core::tokenizer::Tokenizer;
use inference_runtime_core::tokenizer::huggingface::HFTokenizer;
use inference_runtime_core::tokenizer::huggingface::IncrementalDecoder;
use serde_json::Map;
use serde_json::Value;

use crate::api::decode::DecodeEvent;
use crate::tool::ToolArguments;
use crate::tool::ToolDefinition;
use crate::tool::ToolID;

pub struct QwenCodec {
    chat_template: ChatTemplate,
    tokenizer: Arc<HFTokenizer>,
    thinking_end_token: Token,
    turn_end_token: Token,
}

#[derive(Debug, PartialEq)]
pub enum ResponseEvent {
    Thinking(String),
    Text(String),
    ToolCall(ToolCall),
    Completed {
        reason: CompletionReason,
        num_output_tokens: usize,
        turn_closed: bool,
    },
}

#[derive(Debug, PartialEq)]
pub struct ToolCall {
    tool_id: ToolID,
    arguments: ToolArguments,
}

impl ToolCall {
    pub fn tool_id(&self) -> &ToolID {
        &self.tool_id
    }

    pub fn arguments(&self) -> &ToolArguments {
        &self.arguments
    }
}

impl QwenCodec {
    pub fn new(chat_template: ChatTemplate, tokenizer: Arc<HFTokenizer>) -> Result<Self> {
        let thinking_end_token = tokenizer
            .token(THINKING_END)
            .ok_or_else(|| inference_runtime_core::log_err_internal!("Qwen tokenizer is missing {THINKING_END:?}"))?;
        if tokenizer.decode(&[thinking_end_token])? != THINKING_END {
            return Err(inference_runtime_core::log_err_internal!(
                "Qwen tokenizer does not decode its thinking boundary as {THINKING_END:?}"
            ));
        }
        let turn_end_token = tokenizer
            .token(TURN_END)
            .ok_or_else(|| inference_runtime_core::log_err_internal!("Qwen tokenizer is missing {TURN_END:?}"))?;
        if tokenizer.decode(&[turn_end_token])? != TURN_END {
            return Err(inference_runtime_core::log_err_internal!(
                "Qwen tokenizer does not decode its turn boundary as {TURN_END:?}"
            ));
        }
        Ok(Self {
            chat_template,
            tokenizer,
            thinking_end_token,
            turn_end_token,
        })
    }

    pub fn load(model_dir: &Path) -> Result<Self> {
        let chat_template = chat_template::load(model_dir)?;
        let tokenizer = Arc::new(HFTokenizer::from_file(model_dir.join("tokenizer.json"))?);
        Self::new(chat_template, tokenizer)
    }

    pub fn encode(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        enable_thinking: bool,
        reasoning_effort: Option<&str>,
        preserve_thinking: bool,
    ) -> Result<Vec<Token>> {
        let prompt = self.render(messages, tools, enable_thinking, reasoning_effort, preserve_thinking)?;
        self.tokenizer.encode(&prompt)
    }

    pub fn encode_continuation(
        &self,
        messages: Vec<Message>,
        enable_thinking: bool,
        reasoning_effort: Option<&str>,
        previous_turn_closed: bool,
    ) -> Result<Vec<Token>> {
        let mut anchored_messages = vec![
            Message::user(CONTINUATION_ANCHOR_USER),
            Message::assistant(CONTINUATION_ANCHOR_ASSISTANT),
        ];
        anchored_messages.extend(messages);
        let prompt = self.render(anchored_messages, &[], enable_thinking, reasoning_effort, true)?;
        let suffix = prompt
            .split_once(CONTINUATION_ANCHOR_ASSISTANT)
            .map(|(_, suffix)| suffix)
            .ok_or_else(|| {
                inference_runtime_core::log_err_internal!(
                    "Qwen chat template did not preserve the continuation boundary"
                )
            })?;
        let suffix = if previous_turn_closed {
            suffix.strip_prefix(TURN_END).ok_or_else(|| {
                inference_runtime_core::log_err_internal!(
                    "Qwen chat template did not close the continuation anchor with {TURN_END:?}"
                )
            })?
        } else {
            suffix
        };
        self.tokenizer.encode(suffix)
    }

    fn closes_turn(&self, token: Token) -> bool {
        token == self.turn_end_token
    }

    fn render(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        enable_thinking: bool,
        reasoning_effort: Option<&str>,
        preserve_thinking: bool,
    ) -> Result<String> {
        let mut extra = Map::new();
        extra.insert("enable_thinking".to_string(), Value::Bool(enable_thinking));
        if let Some(reasoning_effort) = reasoning_effort {
            extra.insert(
                "reasoning_effort".to_string(),
                Value::String(reasoning_effort.to_string()),
            );
        }
        extra.insert("preserve_thinking".to_string(), Value::Bool(preserve_thinking));
        extra.insert("add_vision_id".to_string(), Value::Bool(false));
        let input = RenderInput {
            messages,
            tools: tools.iter().map(tool_value).collect::<Result<Vec<_>>>()?,
            add_generation_prompt: true,
            extra,
            ..Default::default()
        };
        let prompt = self.chat_template.render(&input).map_err(|error| {
            inference_runtime_core::log_info_invalid_argument!("unable to render Qwen request: {error}")
        })?;
        Ok(prompt)
    }

    pub fn decode<S>(
        self: &Arc<Self>,
        response: S,
        tool_ids: Vec<ToolID>,
        enable_thinking: bool,
    ) -> impl Stream<Item = Result<ResponseEvent>> + Send + 'static + use<S>
    where
        S: Stream<Item = Result<DecodeEvent>> + Send + 'static,
    {
        let codec = self.clone();
        try_stream! {
            let mut response = Box::pin(response);
            let mut decoder = IncrementalDecoder::without_special_tokens(&codec.tokenizer);
            let mut parser = Parser::new(tool_ids, enable_thinking);
            let mut last_output_token = None;
            loop {
                match response.next().await {
                    Some(Ok(DecodeEvent::TokenProbs(token_probs))) => {
                        last_output_token = token_probs.tokens.last().copied();
                        for event in decode_tokens(
                            &mut decoder,
                            &mut parser,
                            codec.thinking_end_token,
                            &token_probs.tokens,
                        )? {
                            yield event;
                        }
                    },
                    Some(Ok(DecodeEvent::Completed {
                        reason,
                        num_output_tokens,
                    })) => {
                        for event in parser.finish() {
                            yield event;
                        }
                        yield ResponseEvent::Completed {
                            reason,
                            num_output_tokens,
                            turn_closed: last_output_token.is_some_and(|token| codec.closes_turn(token)),
                        };
                        break;
                    },
                    Some(Err(error)) => Err::<(), Error>(error)?,
                    None => Err::<(), Error>(Error::internal(
                        "Qwen response ended without a completion event",
                    ))?,
                }
            }
        }
    }
}

fn tool_value(tool: &ToolDefinition) -> Result<Value> {
    let tool_id = tool.tool_id().as_str();
    if tool_id
        .chars()
        .any(|character| matches!(character, '<' | '>' | '\r' | '\n'))
    {
        return Err(Error::invalid_argument(format!(
            "Qwen tool ID {tool_id:?} contains a reserved delimiter"
        )));
    }
    let mut function = Map::new();
    function.insert("name".to_string(), Value::String(tool_id.to_string()));
    if let Some(description) = tool.description() {
        function.insert("description".to_string(), Value::String(description.to_string()));
    }
    function.insert("parameters".to_string(), tool.input_schema().as_value().clone());
    Ok(Value::Object(
        [
            ("type".to_string(), Value::String("function".to_string())),
            ("function".to_string(), Value::Object(function)),
        ]
        .into_iter()
        .collect(),
    ))
}

fn decode_tokens(
    decoder: &mut IncrementalDecoder<'_>,
    parser: &mut Parser,
    thinking_end_token: Token,
    tokens: &[Token],
) -> Result<Vec<ResponseEvent>> {
    let mut events = Vec::new();
    for token in tokens {
        let text = decoder.decode(std::slice::from_ref(token))?;
        if parser.is_thinking() && *token == thinking_end_token {
            if let Some(text) = text {
                let text = text.strip_suffix(THINKING_END).ok_or_else(|| {
                    inference_runtime_core::log_err_internal!(
                        "Qwen incremental decoder did not preserve its thinking boundary"
                    )
                })?;
                if !text.is_empty() {
                    events.extend(parser.feed(text));
                }
            }
            parser.end_thinking();
        } else if let Some(text) = text {
            events.extend(parser.feed(&text));
        }
    }
    Ok(events)
}

struct Parser {
    thinking: bool,
    tool_ids: HashSet<ToolID>,
    text_pending: String,
    in_tool_block: bool,
    tool_block: String,
}

const THINKING_END: &str = "</think>";
const TURN_END: &str = "<|im_end|>";
const CONTINUATION_ANCHOR_USER: &str = "khalamari continuation anchor";
const CONTINUATION_ANCHOR_ASSISTANT: &str = "<khalamari-continuation-boundary>";
const TOOL_START: &str = "<tool_call>";
const TOOL_END: &str = "</tool_call>";

impl Parser {
    fn new(tool_ids: impl IntoIterator<Item = ToolID>, enable_thinking: bool) -> Self {
        Self {
            thinking: enable_thinking,
            tool_ids: tool_ids.into_iter().collect(),
            text_pending: String::new(),
            in_tool_block: false,
            tool_block: String::new(),
        }
    }

    fn feed(&mut self, text: &str) -> Vec<ResponseEvent> {
        if self.thinking {
            vec![ResponseEvent::Thinking(text.to_string())]
        } else {
            self.feed_text(text)
        }
    }

    fn is_thinking(&self) -> bool {
        self.thinking
    }

    fn end_thinking(&mut self) {
        debug_assert!(self.thinking, "Qwen thinking must be active");
        self.thinking = false;
    }

    fn finish(&mut self) -> Vec<ResponseEvent> {
        if self.thinking {
            self.thinking = false;
        }
        let mut events = Vec::new();
        if self.in_tool_block {
            self.text_pending.push_str(TOOL_START);
            self.text_pending.push_str(&self.tool_block);
            self.in_tool_block = false;
            self.tool_block.clear();
        }
        if !self.text_pending.is_empty() {
            events.push(ResponseEvent::Text(std::mem::take(&mut self.text_pending)));
        }
        events
    }

    fn feed_text(&mut self, text: &str) -> Vec<ResponseEvent> {
        if self.in_tool_block {
            self.tool_block.push_str(text);
        } else {
            self.text_pending.push_str(text);
        }
        self.drain_text()
    }

    fn drain_text(&mut self) -> Vec<ResponseEvent> {
        let mut events = Vec::new();
        loop {
            if self.in_tool_block {
                let Some(index) = self.tool_block.find(TOOL_END) else {
                    break;
                };
                let body = self.tool_block[..index].to_string();
                let rest = self.tool_block[index + TOOL_END.len()..].to_string();
                self.tool_block.clear();
                self.in_tool_block = false;
                match self.parse_block(&body) {
                    Some(call) => events.push(ResponseEvent::ToolCall(call)),
                    None => events.push(ResponseEvent::Text(format!("{TOOL_START}{body}{TOOL_END}"))),
                }
                self.text_pending.push_str(&rest);
                continue;
            }
            if let Some(index) = self.text_pending.find(TOOL_START) {
                if index > 0 {
                    events.push(ResponseEvent::Text(self.text_pending[..index].to_string()));
                }
                self.text_pending.drain(..index + TOOL_START.len());
                self.in_tool_block = true;
                self.tool_block = std::mem::take(&mut self.text_pending);
                continue;
            }
            let keep = partial_marker_suffix(&self.text_pending, TOOL_START);
            if self.text_pending.len() > keep {
                let text = self.text_pending[..self.text_pending.len() - keep].to_string();
                self.text_pending.drain(..text.len());
                events.push(ResponseEvent::Text(text));
            }
            break;
        }
        events
    }

    fn parse_block(&self, body: &str) -> Option<ToolCall> {
        let body = body.strip_prefix('\n').unwrap_or(body);
        let body = body.strip_prefix("<function=")?;
        let end = body.find('>')?;
        let tool_id = ToolID::new(&body[..end]).ok()?;
        if !self.tool_ids.contains(&tool_id) {
            return None;
        }
        let mut rest = &body[end + 1..];
        rest = rest
            .strip_suffix("</function>\n")
            .or_else(|| rest.strip_suffix("</function>"))?;
        rest = rest.strip_prefix('\n').unwrap_or(rest);
        let mut arguments = Map::new();
        while !rest.is_empty() {
            rest = rest.strip_prefix("<parameter=")?;
            let end = rest.find('>')?;
            let parameter = &rest[..end];
            if parameter.is_empty() || arguments.contains_key(parameter) {
                return None;
            }
            rest = &rest[end + 1..];
            let value_end = rest.find("</parameter>")?;
            let value = rest[..value_end].trim_matches('\n');
            let value = serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()));
            arguments.insert(parameter.to_string(), value);
            rest = &rest[value_end + "</parameter>".len()..];
            rest = rest.strip_prefix('\n').unwrap_or(rest);
        }
        Some(ToolCall {
            tool_id,
            arguments: ToolArguments::new(Value::Object(arguments))
                .expect("parsed Qwen tool arguments must remain an object"),
        })
    }
}

fn partial_marker_suffix(text: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .find(|length| text.ends_with(&marker[..*length]))
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "qwen_test.rs"]
mod tests;
