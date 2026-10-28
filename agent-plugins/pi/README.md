# khalamari Pi provider

This Pi extension connects Pi to the khalamari `GenerateMessagesStream` gRPC API. One HTTP/2 gRPC stream owns one
resident runtime session. Pi owns conversation history and compaction.

Use this guide to install the extension, configure Pi, and verify a request.
See the [service guide](../../docs/service.md) for checkpoint downloads and service configuration.

## Requirements

| Dependency or content | Contract |
| --- | --- |
| Pi | `0.84.4` or later |
| Runtime | Node.js/npm and a built khalamari text service |
| Input | Text and text tool results, no images |

## Install

Install a detached global copy for normal use. This copy does not depend on the repository path.

Run these commands from the khalamari repository root:

```sh
khalamari_extension="${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}/extensions/khalamari"

mkdir -p "$khalamari_extension/src"
cp agent-plugins/pi/package.json "$khalamari_extension/package.json"
cp agent-plugins/pi/package-lock.json "$khalamari_extension/package-lock.json"
cp crates/inference-runtime-proto/proto/inference_runtime.proto \
  "$khalamari_extension/inference_runtime.proto"
sed 's#../../../crates/inference-runtime-proto/proto/inference_runtime.proto#../inference_runtime.proto#' \
  agent-plugins/pi/src/index.ts > "$khalamari_extension/src/index.ts"
(
  cd "$khalamari_extension"
  npm ci --omit=dev --omit=peer
)
```

| Installation | Location/lifecycle |
| --- | --- |
| Default | `~/.pi/agent/extensions/`, no repository path in `~/.pi/agent/settings.json` |
| `PI_CODING_AGENT_DIR` set | Use that directory instead of `~/.pi/agent` |
| Extension/protobuf update | Repeat the copy, then restart Pi or run `/reload` |

### Extension development

Use a local-path package only for extension development. Run these commands from the repository root:

```sh
cd agent-plugins/pi
npm ci
npm run check
pi install "$PWD"
```

Pi stores a local package as a path reference. It does not copy the package. Run `pi remove "$PWD"` before you use
the detached global copy.

## Start khalamari

Start one text-generation service with gRPC enabled.
Run this example from the repository root to start Qwen3.8 27B with MTP:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.8-27B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.8-27B-MTP-4bit" \
  --spec-type mtp \
  --num-spec-tokens 1
```

The resident provider uses the gRPC address. The OpenAI-compatible provider uses the HTTP address.

## Configure Pi

The configuration below defines resident gRPC access and stateless HTTP access.
Add the provider entries to `~/.pi/agent/models.json`.
If you set `PI_CODING_AGENT_DIR`, use `models.json` in that directory.
Keep any existing provider entries.

```json
{
  "providers": {
    "local": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "unused",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": true,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_completion_tokens",
        "thinkingFormat": "qwen",
        "supportsStrictMode": false,
        "supportsLongCacheRetention": false
      },
      "models": [
        {
          "id": "qwen3_5",
          "name": "Qwen 3.8 27B Local HTTP",
          "reasoning": true,
          "contextWindow": 81920,
          "maxTokens": 8192,
          "thinkingLevelMap": {
            "off": null,
            "minimal": null,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
            "max": null
          },
          "samplingParams": {
            "temperature": 1,
            "top_k": 20,
            "top_p": 0.8
          }
        }
      ]
    },
    "khalamari": {
      "baseUrl": "http://127.0.0.1:50061",
      "api": "khalamari-messages",
      "apiKey": "unused",
      "models": [
        {
          "id": "qwen3_5",
          "name": "Qwen 3.8 27B Local Session",
          "reasoning": true,
          "contextWindow": 81920,
          "maxTokens": 8192,
          "thinkingLevelMap": {
            "off": null,
            "minimal": null,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
            "max": null
          },
          "samplingParams": {
            "temperature": 1,
            "top_k": 20,
            "top_p": 0.8
          }
        }
      ]
    }
  }
}
```

The resident provider uses these fields:

| Field | Owner and effect |
| --- | --- |
| `baseUrl` | The extension connects to this khalamari gRPC endpoint. |
| `api` | `khalamari-messages` selects the extension transport. |
| `id` | Pi uses this value for model selection. The running khalamari service owns the loaded checkpoint. |
| `contextWindow` | Pi uses this value for context accounting and compaction. It must not exceed the service limit. |
| `maxTokens` | The extension sends this value as the default per-turn sampled-token limit. |
| `temperature`, `top_k`, `top_p` | The extension sends these sampling values for each turn. |
| `thinkingLevelMap` | Pi maps its thinking level before the extension maps the level to khalamari reasoning effort. |

Pi turn options override the model sampling defaults. The plugin does not read model settings from environment
variables.

## Use

Start an interactive Pi session with the resident provider:

```sh
pi --model khalamari/qwen3_5 --thinking high
```

Use the HTTP provider when you want a stateless request path:

```sh
pi --model local/qwen3_5 --thinking high
```

### Resident session behavior

Pi owns the complete conversation. A resident stream retains the runtime request across turns:

```text
Pi context                  Resident gRPC stream          Runtime request
first turn: full context  -> create stream              -> materialize full prompt
later turn: new messages  -> reuse stream               -> append prompt suffix
```

The provider records the submitted message cursor. Later messages contain only new user or tool-result content.
These conditions control stream reuse:

| Condition | Effect |
| --- | --- |
| Same endpoint, model ID, and Pi session ID | Reuse the stream while its configuration stays the same. |
| No Pi session ID, or `cacheRetention` is `none` | Use a new stream for each request. |
| System prompt, tool definitions, or thinking configuration changes | Open a new stream and send the complete context. |
| `AbortSignal` | Close the full resident stream. |
| Compaction, history rewrite, or branch | Do not reuse the resident session. Start a new Pi session with the complete context. |

## Verify

Confirm that Pi can see the configured model:

```sh
pi --list-models khalamari
```

Run one request through the resident provider:

```sh
pi --model khalamari/qwen3_5 --thinking high --print "Reply with exactly: hello"
```
