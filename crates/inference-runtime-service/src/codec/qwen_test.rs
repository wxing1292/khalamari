use std::sync::Arc;

use futures_util::StreamExt;
use futures_util::stream;
use hf_chat_template::ChatTemplate;
use hf_chat_template::Message;
use inference_runtime_core::Error;
use inference_runtime_core::runtime::CompletionReason;
use inference_runtime_core::runtime::Token;
use inference_runtime_core::runtime::TokenProbs;
use inference_runtime_core::tokenizer::Tokenizer;
use inference_runtime_core::tokenizer::huggingface::HFTokenizer;
use ordered_float::NotNan;
use serde_json::json;
use tokenizers::AddedToken;
use tokenizers::models::wordlevel::WordLevel;

use super::Parser;
use super::QwenCodec;
use super::ResponseEvent;
use crate::api::decode::DecodeEvent;
use crate::tool::ToolDefinition;
use crate::tool::ToolID;
use crate::tool::ToolInputSchema;

#[test]
fn test_encode_request() {
    let source = concat!(
        "{% for message in messages %}{{ message.role }}:{{ message.content }}\n{% endfor %}",
        "assistant:{% if enable_thinking %}<think>{% else %}<think></think>{% endif %}"
    );
    let expected = "user:hello\nassistant:<think>";
    let template = ChatTemplate::from_str(source).unwrap();
    let codec = QwenCodec::new(template, Arc::new(fixture_tokenizer(&[expected]))).unwrap();

    let tokens = codec
        .encode(vec![Message::user("hello")], &[], false, None, true)
        .unwrap();

    assert_eq!(tokens, vec![Token::new(1), Token::new(2)]);
}

#[test]
fn test_encode_tools() {
    let template =
        ChatTemplate::from_str("{% for tool in tools %}{{ tool.type }}:{{ tool.function.name }}{% endfor %}").unwrap();
    let codec = QwenCodec::new(template, Arc::new(fixture_tokenizer(&["function:read_file"]))).unwrap();
    let tools = [ToolDefinition::new(
        ToolID::new("read_file").unwrap(),
        Some("Read a file".to_string()),
        ToolInputSchema::new(json!({"type": "object"})).unwrap(),
    )];

    let tokens = codec
        .encode(vec![Message::user("hello")], &tools, true, None, true)
        .unwrap();

    assert_eq!(tokens, vec![Token::new(1)]);
}

#[test]
fn test_encode_rejects_tool_id_delimiters() {
    let template = ChatTemplate::from_str("unused").unwrap();
    let codec = QwenCodec::new(template, Arc::new(fixture_tokenizer(&["unused"]))).unwrap();

    for tool_id in ["read>file", "read<file", "read\nfile", "read\rfile"] {
        let tools = [ToolDefinition::new(
            ToolID::new(tool_id).unwrap(),
            None,
            ToolInputSchema::new(json!({"type": "object"})).unwrap(),
        )];
        assert!(matches!(
            codec.encode(vec![Message::user("hello")], &tools, true, None, true),
            Err(Error::InvalidArgument(message)) if message.contains("reserved delimiter")
        ));
    }
}

#[test]
fn test_encode_rejects_invalid_messages() {
    let template = ChatTemplate::from_str("{{ raise_exception('messages required') if not messages }}").unwrap();
    let codec = QwenCodec::new(template, Arc::new(fixture_tokenizer(&["unused"]))).unwrap();

    assert!(matches!(
        codec.encode(Vec::new(), &[], true, None, true),
        Err(Error::InvalidArgument(message)) if message.contains("messages required")
    ));
}

#[test]
fn test_encode_continuation_preserves_sampled_history() {
    let template = ChatTemplate::from_str(concat!(
        "{% for message in messages %}",
        "{% if message.role == 'assistant' %}{{ message.content }}<|im_end|>",
        "{% else %}{{ message.content }}{% endif %}",
        "{% endfor %}",
        "{% if add_generation_prompt %}<|im_start|>assistant{% endif %}",
    ))
    .unwrap();
    let codec = QwenCodec::new(template, Arc::new(fixture_tokenizer(&["two", "assistant"]))).unwrap();

    assert_eq!(
        codec
            .encode_continuation(vec![Message::user("two")], false, None, false)
            .unwrap(),
        vec![Token::new(4), Token::new(1), Token::new(5), Token::new(2)],
    );
    assert_eq!(
        codec
            .encode_continuation(vec![Message::user("two")], false, None, true)
            .unwrap(),
        vec![Token::new(1), Token::new(5), Token::new(2)],
    );
}

#[test]
#[ignore = "requires KHALAMARI_QWEN_MODEL_DIR"]
fn test_checkpoint_continuation_contract() {
    let model_dir = std::env::var("KHALAMARI_QWEN_MODEL_DIR").expect("KHALAMARI_QWEN_MODEL_DIR must be set");
    let codec = QwenCodec::load(std::path::Path::new(&model_dir)).unwrap();
    let first = Message::user("one");
    let answer = Message::assistant("answer");
    let second = Message::user("two");
    let initial_tokens = codec.encode(vec![first.clone()], &[], false, None, true).unwrap();
    let answer_tokens = codec.tokenizer.encode("answer").unwrap();
    let full_tokens = codec
        .encode(vec![first, answer, second.clone()], &[], false, None, true)
        .unwrap();

    let mut closed_tokens = initial_tokens.clone();
    closed_tokens.extend(answer_tokens.iter().copied());
    closed_tokens.push(codec.turn_end_token);
    closed_tokens.extend(
        codec
            .encode_continuation(vec![second.clone()], false, None, true)
            .unwrap(),
    );
    assert_eq!(closed_tokens, full_tokens);

    let mut open_tokens = initial_tokens;
    open_tokens.extend(answer_tokens);
    open_tokens.extend(codec.encode_continuation(vec![second], false, None, false).unwrap());
    assert_eq!(open_tokens, full_tokens);
}

#[test]
fn test_tool_parser_streams_text() {
    let mut parser = fixture_parser();
    assert_eq!(parser.feed("hello "), vec![ResponseEvent::Text("hello ".to_string())]);
    assert_eq!(parser.finish(), Vec::<ResponseEvent>::new());
}

#[test]
fn test_tool_parser_handles_marker_splits() {
    for split in 1.."<tool_call>".len() {
        let mut parser = fixture_parser();
        assert!(parser.feed(&"<tool_call>"[..split]).is_empty());
        assert!(parser.feed(&"<tool_call>"[split..]).is_empty());
        let events = parser.feed("<function=read_file><parameter=path>README.md</parameter></function></tool_call>");
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ResponseEvent::ToolCall(call) if call.tool_id().as_str() == "read_file"));
    }
}

#[test]
fn test_parser_streams_thinking() {
    let mut parser = Parser::new(Vec::new(), true);
    assert_eq!(
        parser.feed("reasoning"),
        vec![ResponseEvent::Thinking("reasoning".to_string())]
    );
    parser.end_thinking();
    assert_eq!(
        parser.feed("\n\nanswer"),
        vec![ResponseEvent::Text("\n\nanswer".to_string())]
    );
    assert!(parser.finish().is_empty());
}

#[test]
fn test_tool_parser_handles_multiple_calls() {
    let mut parser = fixture_parser();
    let events = parser.feed(
        "<tool_call><function=read_file><parameter=path>a</parameter></function></\
         tool_call><tool_call><function=list_dir></function></tool_call>",
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, ResponseEvent::ToolCall(_)))
            .count(),
        2
    );

    let mut parser = fixture_parser();
    let events = parser.feed(
        "<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>",
    );
    assert!(matches!(
        &events[0],
        ResponseEvent::ToolCall(call)
            if call.tool_id().as_str() == "read_file"
                && call.arguments().as_value() == &serde_json::json!({"path": "README.md"})
    ));
}

#[test]
fn test_tool_parser_releases_malformed_call_as_text() {
    let mut parser = fixture_parser();
    assert_eq!(
        parser.feed("<tool_call><function=unknown></function></tool_call>"),
        vec![ResponseEvent::Text(
            "<tool_call><function=unknown></function></tool_call>".to_string()
        )]
    );

    let mut parser = fixture_parser();
    assert!(parser.feed("<tool_call><function=read_file>").is_empty());
    assert_eq!(
        parser.finish(),
        vec![ResponseEvent::Text("<tool_call><function=read_file>".to_string())]
    );
}

#[tokio::test]
async fn test_decode_stream() {
    let tokenizer = Arc::new(fixture_qwen_tokenizer());
    let codec = Arc::new(QwenCodec::new(ChatTemplate::from_str("unused").unwrap(), tokenizer).unwrap());
    let response = stream::iter([
        Ok(DecodeEvent::TokenProbs(TokenProbs {
            tokens: vec![Token::new(1)],
            probs: vec![NotNan::new(1.0).unwrap()],
        })),
        Ok(DecodeEvent::TokenProbs(TokenProbs {
            tokens: vec![Token::new(3), Token::new(2), Token::new(4)],
            probs: vec![NotNan::new(1.0).unwrap(); 3],
        })),
        Ok(DecodeEvent::Completed {
            reason: CompletionReason::StopSequence,
            num_output_tokens: 4,
        }),
    ]);

    let events = codec
        .decode(response, Vec::new(), true)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(
        events,
        vec![
            ResponseEvent::Thinking("reasoning".to_string()),
            ResponseEvent::Thinking(" ".to_string()),
            ResponseEvent::Text(" answer".to_string()),
            ResponseEvent::Completed {
                reason: CompletionReason::StopSequence,
                num_output_tokens: 4,
                turn_closed: true,
            },
        ]
    );
}

#[tokio::test]
async fn test_decode_forwards_error() {
    let tokenizer = Arc::new(fixture_tokenizer(&["hello"]));
    let codec = Arc::new(QwenCodec::new(ChatTemplate::from_str("unused").unwrap(), tokenizer).unwrap());
    let events = codec
        .decode(
            stream::iter([Err(Error::aborted("request aborted"))]),
            Vec::new(),
            false,
        )
        .collect::<Vec<_>>()
        .await;

    assert!(matches!(
        events.as_slice(),
        [Err(Error::Aborted(message))] if message == "request aborted"
    ));
}

#[tokio::test]
async fn test_decode_rejects_eof_without_completion() {
    let tokenizer = Arc::new(fixture_tokenizer(&["hello"]));
    let codec = Arc::new(QwenCodec::new(ChatTemplate::from_str("unused").unwrap(), tokenizer).unwrap());
    let events = codec
        .decode(
            stream::iter(Vec::<Result<DecodeEvent, Error>>::new()),
            Vec::new(),
            false,
        )
        .collect::<Vec<_>>()
        .await;

    assert!(matches!(
        events.as_slice(),
        [Err(Error::Internal(message))] if message == "Qwen response ended without a completion event"
    ));
}

fn fixture_parser() -> Parser {
    Parser::new(
        [ToolID::new("read_file").unwrap(), ToolID::new("list_dir").unwrap()],
        false,
    )
}

fn fixture_tokenizer(text: &[&str]) -> HFTokenizer {
    let vocab = std::iter::once(("[UNK]".to_string(), 0))
        .chain(
            text.iter()
                .copied()
                .chain(["</think>", "<|im_end|>", "<|im_start|>"])
                .enumerate()
                .map(|(index, text)| (text.to_string(), u32::try_from(index + 1).unwrap())),
        )
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer
        .add_special_tokens([
            AddedToken::from("</think>", true),
            AddedToken::from("<|im_end|>", true),
            AddedToken::from("<|im_start|>", true),
        ])
        .unwrap();
    HFTokenizer::new(tokenizer)
}

fn fixture_qwen_tokenizer() -> HFTokenizer {
    let vocab = [
        ("[UNK]".to_string(), 0),
        ("reasoning".to_string(), 1),
        ("answer".to_string(), 2),
        ("</think>".to_string(), 3),
        ("<|im_end|>".to_string(), 4),
    ]
    .into_iter()
    .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer
        .add_special_tokens([AddedToken::from("<|im_end|>", true)])
        .unwrap();
    HFTokenizer::new(tokenizer)
}
