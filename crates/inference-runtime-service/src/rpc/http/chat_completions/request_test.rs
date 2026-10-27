use axum::http::HeaderMap;
use axum::http::HeaderValue;
use serde_json::json;
use uuid::Uuid;

use crate::rpc::http::chat_completions::request::Content;
use crate::rpc::http::chat_completions::request::Message;
use crate::rpc::http::chat_completions::request::Request;
use crate::rpc::http::chat_completions::request::Tool;
use crate::rpc::http::chat_completions::request::ToolCall;
use crate::rpc::http::chat_completions::request::ToolChoice;
use crate::rpc::http::chat_completions::request::preprocess;
use crate::rpc::http::chat_completions::resolve_request_id;

#[test]
fn test_schema() {
    let request = serde_json::from_value::<Request>(json!({
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": [{"type": "text", "text": "read"}]},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"README.md\"}"
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "contents",
                "name": "read_file"
            }
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object"}
            }
        }],
        "tool_choice": "auto",
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_completion_tokens": 32,
        "temperature": 0.2,
        "top_k": 8,
        "top_p": 0.9,
        "seed": 42,
        "enable_thinking": true
    }))
    .unwrap();

    assert_eq!(request.model.as_deref(), Some("qwen"));
    assert_eq!(request.messages.len(), 4);
    assert!(matches!(
        &request.messages[1],
        Message::User {
            content: Content::Parts(parts)
        } if parts.len() == 1
    ));
    assert!(matches!(
        &request.messages[2],
        Message::Assistant { tool_calls, .. }
        if matches!(tool_calls.as_slice(), [ToolCall::Function { .. }])
    ));
    assert!(matches!(request.tools.as_slice(), [Tool::Function { .. }]));
    assert!(matches!(request.tool_choice, Some(ToolChoice::Auto)));
    assert!(request.stream);
    assert!(request.stream_options.unwrap().include_usage);
    assert_eq!(request.max_completion_tokens, Some(32));
    assert_eq!(request.temperature, Some(0.2));
    assert_eq!(request.top_k, Some(8));
    assert_eq!(request.top_p, Some(0.9));
    assert_eq!(request.seed, Some(42));
    assert_eq!(request.enable_thinking, Some(true));
}

#[test]
fn test_role_fields_are_structural() {
    for message in [
        json!({"role": "system", "content": "hello", "tool_call_id": "call_1"}),
        json!({"role": "developer", "content": "hello", "tool_call_id": "call_1"}),
        json!({"role": "user", "content": "hello", "tool_calls": []}),
        json!({"role": "assistant", "content": "hello", "name": "assistant"}),
        json!({"role": "tool", "content": "result"}),
    ] {
        assert!(serde_json::from_value::<Message>(message).is_err());
    }
}

#[test]
fn test_preprocess() {
    let request = serde_json::from_value::<Request>(json!({
        "model": "qwen",
        "messages": [
            {"role": "developer", "content": "be concise"},
            {"role": "user", "content": "read"}
        ],
        "store": false,
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object"},
                "strict": false
            }
        }],
        "stream": true,
        "stream_options": {"include_usage": true},
        "reasoning_effort": "medium",
        "max_completion_tokens": 4
    }))
    .unwrap();

    let Ok(_) = preprocess(request) else {
        panic!("Pi Chat Completions request must preprocess");
    };
}

#[test]
fn test_tool_history() {
    let request = serde_json::from_value::<Request>(json!({
        "model": "qwen",
        "messages": [
            {"role": "user", "content": "use the retired tool"},
            {
                "role": "assistant",
                "content": null,
                "reasoning_content": "I used the retired tool.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "retired_tool",
                        "arguments": "{}"
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "retired_tool",
                "content": "contents"
            }
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "active_tool",
                "parameters": {"type": "object"}
            }
        }],
        "max_completion_tokens": 4
    }))
    .unwrap();

    let Ok(_) = preprocess(request) else {
        panic!("valid Chat Completions request must preprocess");
    };

    let request = serde_json::from_value::<Request>(json!({
        "model": "qwen",
        "messages": [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "missing", "arguments": "{}"}
                }]
            }
        ]
    }))
    .unwrap();
    assert!(preprocess(request).is_err());
}

#[test]
fn test_tool_replay_with_caller_request_id() {
    let caller_request_id = Uuid::parse_str("67e55044-10b1-426f-9247-bb680e5fe0c8").unwrap();
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-request-id",
        HeaderValue::try_from(caller_request_id.to_string()).unwrap(),
    );
    let Ok(request_id) = resolve_request_id(&headers) else {
        panic!("caller request UUID must be valid");
    };
    let tool_call_id = Uuid::new_v4().to_string();
    assert!(preprocess(tool_replay_request(&tool_call_id)).is_ok());

    let Ok(replay_request_id) = resolve_request_id(&headers) else {
        panic!("replayed caller request UUID must be valid");
    };
    assert!(preprocess(tool_replay_request(&tool_call_id)).is_ok());

    assert_eq!(replay_request_id, request_id);
    assert_eq!(request_id, caller_request_id);
    assert_ne!(Uuid::parse_str(&tool_call_id).unwrap(), request_id);
}

#[test]
fn test_tool_replay_with_generated_request_ids() {
    let headers = HeaderMap::new();
    let Ok(request_id) = resolve_request_id(&headers) else {
        panic!("server must generate a request UUID");
    };
    let tool_call_id = Uuid::new_v4().to_string();
    assert!(preprocess(tool_replay_request(&tool_call_id)).is_ok());

    let Ok(replay_request_id) = resolve_request_id(&headers) else {
        panic!("server must generate a replay request UUID");
    };
    assert!(preprocess(tool_replay_request(&tool_call_id)).is_ok());

    assert_ne!(replay_request_id, request_id);
    let tool_call_id = Uuid::parse_str(&tool_call_id).unwrap();
    assert_ne!(tool_call_id, request_id);
    assert_ne!(tool_call_id, replay_request_id);
}

#[test]
fn test_tool_definitions() {
    for request in [
        json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {"type": "function", "function": {"name": "same", "parameters": {}}},
                {"type": "function", "function": {"name": "same", "parameters": {}}}
            ]
        }),
        json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object"},
                    "strict": true
                }
            }]
        }),
    ] {
        let request = serde_json::from_value::<Request>(request).unwrap();
        assert!(preprocess(request).is_err());
    }
}

#[test]
fn test_unsupported_options() {
    for request in [
        json!({
            "model": "",
            "messages": [{"role": "user", "content": "hello"}]
        }),
        json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": "required"
        }),
        json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "store": true
        }),
        json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "medium",
            "enable_thinking": false
        }),
    ] {
        let request = serde_json::from_value::<Request>(request).unwrap();
        assert!(preprocess(request).is_err());
    }
}

fn tool_replay_request(tool_call_id: &str) -> Request {
    serde_json::from_value(json!({
        "model": "qwen",
        "messages": [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"README.md\"}"
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "contents",
                "name": "read_file"
            }
        ],
        "max_completion_tokens": 4
    }))
    .unwrap()
}
