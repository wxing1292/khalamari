# Qwen Service and Decode RPC

The service runs one transport-neutral `Inference` API over the runtime. The gRPC adapter submits model-ready token IDs
through its `decode` operation.

The HTTP adapter provides collected and streaming OpenAI-compatible Chat Completions.
The Qwen3-ASR service instead provides collected OpenAI-compatible audio transcription.
Both adapters use the same `decode` operation.

[`core.md`](core.md) defines runtime scheduling and page ownership. The executor documents define model execution.

Use these sections for common tasks:

| Task | Section |
| --- | --- |
| Download or convert a checkpoint and start a server | [Binaries and checkpoints](#binaries-and-checkpoints) |
| Submit an audio transcription | [HTTP audio transcription](#http-audio-transcription) |
| Generate text through gRPC | [gRPC generation](#grpc-generation) |
| Use collected or streaming chat responses | [HTTP Chat Completions](#http-chat-completions) |
| Inspect logging, timing, and lifecycle events | [Operational notes](#operational-notes) |
| Check correctness or measure end-to-end performance | [Correctness and long decode](#correctness-and-long-decode) and [End-to-end performance helper](#end-to-end-performance-helper) |
| Change service boundaries or tool-state handling | [Source ownership](#source-ownership) and [Tool state](#tool-state) |

## Source ownership

`crates/inference-runtime-service/src/api/` owns these functions:

- Token-level validation
- Process-local runtime request IDs
- Stop-sequence merging
- Submission and output streaming
- Cancellation when a response is dropped

`rpc/grpc/` owns protobuf conversion and tonic status mapping. `rpc/http/` owns HTTP listener setup, JSON parsing, and
HTTP error mapping.

The text Qwen server loads one immutable `QwenCodec` from the model directory.
It shares this codec with HTTP requests.
The Qwen3-ASR server loads one immutable `Qwen3ASRService` and its checkpoint tokenizer.

RPC converts wire objects to the codec domain. It also maps codec events back to wire objects. RPC does not implement
these functions:

- Prompt syntax
- Tokenization algorithms
- The Qwen tool dialect

`crates/inference-runtime-service/src/tool/` owns the transport-neutral and model-neutral tool domain. It does not own
these functions:

- Conversation storage
- Tool handlers
- Execution scheduling
- Prompt rendering
- Model-specific tool syntax

`inference-runtime-core/src/chat_template/` loads the checkpoint-authoritative Hugging Face chat template. It compiles
the template with `hf-chat-template`.

A standalone `chat_template.jinja` takes priority over an inline tokenizer-config template. `codec/qwen.rs` owns the
model-specific composition.

`QwenCodec` contains the compiled template and shared Hugging Face tokenizer. Its `encode` method renders messages and
tools before tokenization.

The `decode` method transforms a token stream into these outputs:

- Reasoning
- Answer text
- Tool calls
- A terminal event

When thinking is enabled, the codec consumes the Qwen `</think>` boundary. It never exposes model control tokens as
content.

Each response stream keeps its incremental detokenization and Qwen response-grammar state private.

The runtime request-event channel is intentionally `async_channel::unbounded`. It preserves committed token
probabilities and turn-completion events without blocking synchronous runtime commit. Dropping a `DecodeSession`
drops its `ExternalRequest`. This action cancels only that request. A slow transport consumer does not terminate the
request.

The cancellation channel is also unbounded. It connects `ExternalRequest` to the existing `Sessions` async
event loop. The same event loop receives completed turns. Cancellation does not create another async task and does not
cross the scheduler boundary.

The session-command channel is unbounded. A resume, single-request eviction, or idle GC call sends one command
and waits on its oneshot response. The `Sessions` task owns the pending-session map. Service API tasks do not lock or
access that map directly.

## Tool state

`ToolState` is the only lifecycle owner for state derived from ordered tool events in one conversation:

```text
ToolState
├── tools: currently callable ToolDefinition values
└── executions: HashMap<ToolCallID, ToolCallRequest>
```

`ToolState` does not retain completed calls. It also does not retain registration, unregistration, response, or
cancellation events. The persistent conversation history owns these events.

`ToolState::fold` applies history in order. This operation reconstructs the two derived collections.

`ToolRegistration` and `ToolUnregistration` are non-empty batches without duplicates. `register_tool` and
`unregister_tool` are atomic.

Each operation updates all definitions or leaves the state unchanged. Definition order follows conversation-history
order. Thus, a later prompt adapter can render tools deterministically.

Removing a tool prevents new calls. A later event can register the same `ToolID` again.

A `ToolCallRequest` has the following transport-neutral shape:

```text
ToolCallRequest
├── tool_id: ToolID
├── tool_call_id: ToolCallID
└── arguments: ToolArguments
```

`ToolState` is a correlation ledger. It is not a tool executor.

`request_execution` requires a currently registered `tool_id`. It also requires a `tool_call_id` that is not in flight.
Multiple requests can remain in flight.

`respond_execution` and `cancel_execution` address an execution only by `ToolCallID`. They do not examine the currently
callable definitions.

Thus, unregistration does not cancel an accepted call. The response for that call remains valid. Completed and canceled
entries leave the in-flight ledger immediately.

Execution iteration has no ordering contract.

A `ToolCallResponse` carries:

```text
ToolCallResponse
├── tool_call_id: ToolCallID
├── raw_content: Vec<ToolRawContent>
├── structured_content: Option<ToolStructuredContent>
└── is_error: bool
```

`ToolRawContent` is model-facing content and currently supports text. `ToolStructuredContent` is an arbitrary JSON value
for programmatic consumers.

The two content forms can coexist. A tool-level failure sets `is_error`. It also includes a safe, non-empty error
message for the model.

Service and protocol failures remain ordinary `Err(Error)` values. They are not false tool responses.

The event projection is:

```text
Registration     tools + definitions    executions unchanged
Unregistration   tools - definitions    executions unchanged
CallRequest      require active tool    executions + request
CallResponse     tools ignored          executions - request
CallCancellation tools ignored          executions - request
```

The persistent conversation history is the authority. Its position is the natural version.

`ToolState` is only a derived in-memory projection. It is not a history store or recoverable cache.

Calls from one model response are independently in flight. The agent or tool environment can run them concurrently.
The inference server does not own these controls:

- Approval and permissions
- Side-effect serialization
- Concurrency limits
- Handler cancellation
- Argument-schema validation

## Binaries and checkpoints

The service provides Qwen3, Qwen3-ASR, and Qwen3.5 binaries.
The Qwen3.5 binary names also apply to compatible Qwen3.6 and Qwen3.8 MLX checkpoints.
The text-generation binaries can use optional speculative models.

DSpark and DFlash2 support is experimental.
Their checkpoint contracts, CLI, cache sizing, and proposal policies may change.

The following table maps model families to binaries and checkpoints:

| Model                  | Binary           | Main checkpoint                      | Optional Spec checkpoint                  |
| ---------------------- | ---------------- | ------------------------------------ | ----------------------------------------- |
| Qwen3 dense 14B        | `qwen3`          | `mlx-community/Qwen3-14B-4bit`       | optional official Qwen3 DSpark checkpoint |
| Qwen3-ASR 1.7B         | `qwen3_asr`      | `mlx-community/Qwen3-ASR-1.7B-8bit`  | none                                      |
| Qwen3.8 dense 27B      | `qwen3_5_dense`  | `mlx-community/Qwen3.8-27B-4bit`     | matching MTP, Qwen3x DSpark, or Qwen3x DFlash2 |
| Qwen3.6 sparse 35B-A3B | `qwen3_5_sparse` | `mlx-community/Qwen3.6-35B-A3B-4bit` | matching MTP, Qwen3x DSpark, or Qwen3x DFlash2 |

Download with the Hugging Face CLI:

```sh
hf auth login
hf download mlx-community/Qwen3-14B-4bit --local-dir models/Qwen3-14B-4bit
hf download mlx-community/Qwen3-ASR-1.7B-8bit --local-dir models/Qwen3-ASR-1.7B-8bit
hf download mlx-community/Qwen3.8-27B-4bit --local-dir models/Qwen3.8-27B-4bit
hf download mlx-community/Qwen3.8-27B-MTP-4bit --local-dir models/Qwen3.8-27B-MTP-4bit
hf download RadixArk/Qwen3.8-27B-DSpark --local-dir models/Qwen3.8-27B-DSpark
hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir models/Qwen3.8-27B-DFlash2
```

Use the corresponding 35B-A3B names for the sparse model. MTP checkpoints contain Spec weights. They must match the
Main model family.

The DSpark and DFlash2 repositories contain BF16 source checkpoints. Convert these checkpoints before service startup.
Both converter subcommands use the same affine format:

- Matrix payloads use packed `U32` storage.
- Affine scales and biases use BF16 storage.
- Quantization calculations use F32.
- Packed codes use the final stored BF16 scale and bias values.
- Unquantized tensors preserve their source BF16 dtype.

### Qwen3x DSpark conversion

Convert an official BF16 Qwen3x DSpark checkpoint to the affine executor format:

```sh
cargo run --bin qwen3x_spec_quantize -- dspark \
  --input-dir /path/to/Qwen3-DSpark \
  --output-dir /path/to/Qwen3-DSpark-affine \
  --group-size 64 --bits 4 --markov-w2-bits 8
```

The output directory must not exist before you run the converter.
The converter writes `model.safetensors` and `model.safetensors.index.json`.
It preserves the confidence projection and bias as BF16.
The input must be the official BF16 DSpark checkpoint.
The flat DSpark query projection width can differ from `hidden_size`.
The converter and loader derive the query width from `num_attention_heads * head_dim`.
An affine checkpoint that was generated before confidence support does not contain these tensors.
Regenerate that checkpoint into a new output directory.

### Qwen3x DFlash2 conversion

Convert the official BF16 Qwen3.8 DFlash2 checkpoint to the affine executor format:

```sh
cargo run --bin qwen3x_spec_quantize -- dflash2 \
  --input-dir /path/to/Qwen3.8-27B-DFlash2 \
  --output-dir /path/to/Qwen3.8-27B-DFlash2-affine
```

The output directory must not exist before you run the converter.
The default policy uses group size 64 and 4-bit affine matrices.
It uses 6-bit affine matrices for layer 2 and layer 4 `v_proj` and `down_proj` weights.
This policy matches the tensor-level Q4_K_M choices in `z-lab/Qwen3.8-27B-DFlash2-GGUF`.
It preserves norms and dynamic-convolution base kernels as BF16.
The output checkpoint contains no unquantized BF16 weight matrix.

Qwen3 Main-only startup:

```sh
cargo run --release --bin qwen3 -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3-14B-4bit"
```

Qwen4x Main-only startup with a compatible affine checkpoint:

```sh
cargo run --release -p inference-runtime-service --bin qwen4x -- \
  --hf-model-dir /path/to/Qwen4x-affine
```

`qwen4x` uses the shared hybrid-server CLI and 2048-token logical cache blocks.
It rejects speculative model flags and resource inputs.
The executor owns the main KV, index K, and auxiliary-state page layout.
The complete resident n-gram table must fit memory.
See [Qwen4x](executor_qwen4x.md) for composition and validation limits.

Qwen3-ASR startup:

```sh
cargo run --release --bin qwen3_asr -- \
  --grpc-listen-addr 127.0.0.1:50051 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3-ASR-1.7B-8bit"
```

Qwen3-ASR uses 16-token logical cache blocks.
It defaults to 8,192 shared cache pages, two running requests, 128 flattened batch tokens, and 64 tokens for each
request transaction.
It uses one compute lane and does not use a speculative checkpoint.

Qwen3 startup with DSpark:

```sh
cargo run --release --bin qwen3 -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3-14B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3-DSpark-affine" \
  --spec-type dspark
```

The Qwen3 executor gets stop tokens from the checkpoint configuration when `generation_config.json` is absent.

Qwen3.5/Qwen3.6 dense startup with DSpark:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-27B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.6-27B-DSpark-affine" \
  --spec-type dspark
```

Qwen3.8 dense startup with DFlash2:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.8-27B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.8-27B-DFlash2-affine" \
  --spec-type dflash2
```

The Qwen3 service accepts only `--spec-type dspark`.
The Qwen3.5 services accept `mtp`, `dspark`, and `dflash2`.

Qwen3.5/Qwen3.6/Qwen3.8 startup with MTP enabled:

Dense:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.8-27B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.8-27B-MTP-4bit" \
  --spec-type mtp \
  --num-spec-tokens 4
```

Sparse:

```sh
cargo run --release --bin qwen3_5_sparse -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-35B-A3B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.6-35B-A3B-MTP-4bit" \
  --spec-type mtp
```

The normal MTP, DSpark, and DFlash2 commands use the same service and scheduler arguments.
Select the Spec checkpoint with these paired arguments:

```text
--hf-spec-model-dir DIR
--spec-type {mtp,dspark,dflash2}
```

An MTP checkpoint enables one speculative MTP step by default.
`--num-spec-tokens K` sets the generated proposal count for MTP, DSpark, or DFlash2.
It takes a positive `usize` value and remains fixed for the server lifetime.
The executor reuses the checkpoint's one physical MTP layer for K dependent logical steps.
Omit `--num-spec-tokens` to use one MTP step.
`--max-tokens-per-request` must not exceed `--max-tokens`.
For MTP with K speculative tokens, `--max-tokens-per-request` must be at least K.
An MTP decode request must contain at least K initial input tokens.
Without an override, DSpark generates checkpoint `block_size` proposals and DFlash2 generates `block_size - 1` proposals.
With `--num-spec-tokens K`, DSpark executes one anchor and `K - 1` MASK rows. All K rows predict proposals.
DFlash2 executes one anchor and K MASK rows. Only the MASK rows predict proposals.
The override changes draft computation, scratch, replay, and sampling. It does not change weights or checkpoint files.
K must be smaller than `max_position_embeddings`. DFlash2 also requires `K + 1 < sliding_window`.
The proposal count is independent of `--max-tokens-per-request`.
That option limits the Main verification batch.
The scheduler may verify only a proposal prefix.

`inference-runtime-launcher` owns the Qwen CLI schemas, worker package generation, Cargo builds, and process replacement.
It has no dependency on runtime core, model executors, or Metal.
The service consumes the same CLI schemas and validation.
`qwen3_5_dense` and `qwen3_5_sparse` validate the normal CLI without instantiating a model runtime.
For MTP with K speculative tokens, the launcher calculates `L = K + 1`.
For Vanilla, DSpark, and DFlash2, it uses `L = 1`.

Each launcher generates a small package under `target/specialized/<binary>_lanes_L`.
Its entry point calls the matching service function with const `L`.
Each L has a distinct package and executable name.
The service library and its dependencies do not depend on L.
The launcher replaces itself with the worker after Cargo finishes.
The worker enters the service directly.

All workers share the launcher's Cargo target directory, target mode, and profile.
An explicit `--target` remains explicit. A native build remains native.
A custom target directory or profile also applies to the worker.
The launcher `build.rs` records this build location. It does not generate model constants.
Worker manifests copy the workspace profiles and dependency overrides (`[patch]` and `[replace]`).
The launcher resolves relative override paths from the source workspace.
Worker lockfiles use the workspace dependency versions. A missing worker lockfile is restored from the workspace lockfile.
A cold worker build acquires the service dependencies once. Later workers reuse them.
The repository source, Cargo, and the selected Rust toolchain must remain available.

Cargo checks source and dependency freshness on each launch.
Generation preserves unchanged file timestamps. A warm launch reuses the existing executable.
Changing L selects another small worker package without rebuilding unchanged dependencies.
The package generation lock and Cargo build lock serialize concurrent builds.

The gRPC address defaults to `127.0.0.1:50051`. The HTTP address defaults to `127.0.0.1:8000`.

The executor hibernation timeout defaults to 300 seconds.
After this period without executable model work, the service writes model state to SSD and unloads model resources.
The listeners and runtime requests remain active.
The next executable batch loads weights and state before execution.
`--executor-hibernation-timeout-secs` accepts a positive integer.
`--executor-hibernation-mode` accepts `all` or `selected`. It defaults to `selected`.
Use `all` to write every state entry.

One lifecycle owner stops both listeners in these conditions:

- The runtime stops.
- A listener fails.
- The process receives SIGINT or SIGTERM.

`--num-spec-tokens` requires a supported `--spec-type` and `--hf-spec-model-dir`.
The service rejects zero or an incomplete Spec checkpoint argument pair.
For a Main-only run, omit both Spec checkpoint arguments and `--num-spec-tokens`.

Qwen uses 32 KiB physical cache pages. Qwen3 and Qwen3.5 default to 320K pages (327,680 pages, or 10 GiB).
The Qwen3-14B geometry stores 16 tokens in one physical page.

Its 16-token logical cache block uses 40 pages across 40 layers. Thus, the default holds 8,192 complete blocks. These
blocks contain 131,072 resident tokens in aggregate.

When DSpark or DFlash2 is enabled, the executor adds persistent Spec history pages to the same logical block.
The page count depends on the selected Spec layer and KV geometry.

Qwen3.5 keeps 2,048-token logical blocks to amortize its GDN snapshots. It defaults to 320K shared pages.
MTP step K adds K logical KV cache lanes to the Main lane.
All lanes allocate from the same shared physical-page arena.

At startup, each service derives the page count for one block from the initialized executor. The service rejects
`--num-cache-pages` when one complete block cannot fit.
The service classifies a model-executor initialization failure as an internal startup error.

The service also derives the runtime `context_window` from the Main model's `max_position_embeddings`. Vanilla and MTP
use the Main value. DSpark and DFlash2 subtract the runtime proposal count because the block-Spec model
applies RoPE to the Main sampled anchor and the complete proposal block. Startup configuration logs include the
effective `context_window`.

The rejection reports this dynamic minimum.

Recommendation: For performance comparisons, pass `--num-cache-pages` explicitly. This setting controls memory
pressure.

`Qwen3Config` and `Qwen35Config` resolve the running-request and per-batch capacities.
CLI checkpoint arguments remain optional parser inputs.
Configuration validation converts them to one `Vanilla`, `MTP`, `DSpark`, or `DFlash2` model mode.
The validated configuration does not store independent speculative-model options.
`--max-requests` defines both the running request-slot capacity and the per-batch request capacity.
The model service passes this value to the executor, `RuntimeConfig`, and `SchedulerConfig`.
Every submitted request owns one request slot. The request channel capacity equals the running request-slot capacity.

Admission assigns a slot before a request enters the scheduler.
GQA page tables, GDN request state, sampling state, and request-indexed workspaces use the same slot domain.
GDN retains the current state and destinations for the accepted final state and cache-block boundaries.
Decode writes transient per-forward deltas. It does not reserve a full state for every speculative prefix.
Buffers, scratch allocations, replay resources, and resident model resources remain reusable.

Qwen3.5 derives the same request-local GDN slot count for Vanilla, MTP, DSpark, and DFlash2:

```text
block_boundary_states = ceil(max_tokens_per_request / num_tokens_per_block)
materialized_states = 1 + block_boundary_states
state_slots = 1 + materialized_states
```

The first slot stores the current state. Materialized destinations store the accepted final state and crossed boundaries.
The total arena scales with `--max-requests * state_slots * full_model_state_bytes`.
The separate replay log scales with GDN layer count, `--max-tokens`, and the per-token delta/QKV widths.
Its allocation includes unused chunkwise-prefix rows. See [`executor_gdn.md`](executor_gdn.md) for layouts and commit ordering.

One default batch has these scheduler limits:

- 2 requests
- 128 flattened tokens
- 64 tokens for each request

## HTTP audio transcription

The `qwen3_asr` HTTP listener provides `POST /v1/audio/transcriptions`.
The request uses `multipart/form-data` and accepts these fields:

| Field | Requirement | Meaning |
| --- | --- | --- |
| `file` | Required, exactly once | Integer or F32 WAV audio |
| `model` | Required, exactly once | Non-empty caller model identifier |
| `language` | Optional, at most once | Checkpoint language name for forced-language decoding |
| `prompt` | Optional, at most once | System context before the audio turn |
| `response_format` | Optional, at most once | `json` or `text`; default is `json` |

The endpoint rejects unknown or repeated fields.
The request body limit is 32 MiB.
Audio preparation runs in a blocking CPU task before runtime admission.
The decoder emits at most 512 output tokens.

Start a transcription:

```sh
curl -sS http://127.0.0.1:8000/v1/audio/transcriptions \
  -F 'file=@/absolute/path/to/audio.wav;type=audio/wav' \
  -F 'model=qwen3-asr-1.7b'
```

The default JSON response has this shape:

```json
{"text":"transcribed text","language":"English"}
```

Pass `-F 'language=English'` to force one supported language.
Language matching is case-insensitive and the response uses the checkpoint spelling.
Pass `-F 'prompt=meeting about cache ownership'` to add context.
Pass `-F 'response_format=text'` to return only transcription text.

The first version accepts one WAV resource for each request.
It converts the input to mono F32 at 16 kHz and calculates log-Mel features on CPU.
It completes this CPU work before runtime admission.
The Audio encoder executor runs only when a required cache block misses and the resource remains symbolic.
It owns a worker thread, a Metal stream, the Audio Tower, and the Audio Tower weight lifecycle.
Service hibernation stops and starts this executor with the decoder lifecycle command.
See [`qwen3_asr.md`](qwen3_asr.md) for the resource and executor contract.

## gRPC generation

The service exports four symmetric RPCs:

| Input domain | One turn | Multiple turns |
| --- | --- | --- |
| Model-ready token IDs | `GenerateTokens` | `GenerateTokensStream` |
| Structured messages | `GenerateMessages` | `GenerateMessagesStream` |

`GenerateTokensRequest` contains token IDs and generation fields. The internal service API can also contain explicit
token positions for model-specific inputs such as M-RoPE. The gRPC request derives normal sequential token positions
and does not expose this field.

`GenerateTokensStream` requires one initial request. The first request contains the initial model input. Each later
request contains only the new prompt tokens for one turn. Each request supplies the generation configuration for that
turn. The token API does not apply a chat template or add a turn separator. The caller must include every token in the
new prompt suffix.

`GenerateMessagesRequest` has the same shape for every turn. The first request contains the initial messages and tool
definitions. Each later request contains only the new user or tool-result messages for one turn. The server retains
the exact sampled tokens. `QwenCodec` applies the checkpoint chat template only to the new turn boundary and messages.
It submits the resulting prompt suffix to the resident runtime request. It does not decode and re-encode sampled
assistant text.

The Qwen adapter records whether the sampled output ended with `<|im_end|>`. If it did, the next prompt suffix starts
with the template newline. Otherwise, the suffix starts with `<|im_end|>` and the template newline. This rule closes a
turn that ended at a caller stop sequence or sampled-token limit before it appends the next message.

Each turn can set a new `GenerationConfig`. `enable_thinking` and `reasoning_effort` must remain constant for one
`GenerateMessagesStream`. These fields affect the checkpoint prompt prefix. A change would rewrite resident history.

The streaming APIs are append-only. A caller that compacts, rewrites, or branches history must start a new stream.
The server does not infer a replacement context and does not fall back to stateless generation inside an existing
stream.

`ToolDelta.insert` adds a definition or overwrites the definition with the same ID. `ToolDelta.remove` deletes a
definition. Removing an absent ID has no effect. Duplicate IDs and an ID in both fields are invalid. The current Qwen
adapter accepts tool definitions only on the first turn because the checkpoint template puts them in the prompt
prefix. A tool change requires a new stream.

The server completes one turn before it reads the next request. Events from different turns cannot interleave. One
bidirectional stream owns one runtime session:

```text
client                     message adapter        Sessions               scheduler / executor
  |                              |                       |                          |
  | first turn ----------------->| render tokens         |                          |
  |                              |---------------------->| create InternalRequest   |
  |                              |                       | allocate request slot --->|
  |<--------------------------------------------------------------- output events |
  |                              |                       |<---- completed request    |
  |<---------------------------- completion ------------| store, then publish       |
  |                              |                       |                          |
  | new turn messages ---------->| render exact suffix   |                          |
  |                              | close prior turn      |                          |
  |                              | append suffix ------->| take pending request     |
  |                              |                       | enqueue same slot ------->|
  |<--------------------------------------------------------------- output events |
  |                              |                       |<---- completed request    |
  |<---------------------------- completion ------------| store, then publish       |
  |                              |                       |                          |
  | half-close / cancel -------->| drop adapter session  | cancel request ---------->|
```

The manager stores the completed internal request before it publishes the turn-completion event. Thus, the next input
can take the pending request without a completion-to-resume race. A new turn does not allocate a new request slot. It
retains the request's trie blocks, KV pages, GDN state pages, token history, and cursor. The executor still processes
the last sampled token and the newly appended prompt because those tokens do not yet have committed model state. It
does not process previously committed history again.

An idle stream between turns retains one request slot and its resident model state. `evict_one_req` removes the oldest
idle request. `evict_expired` removes all requests that exceed a supplied idle duration. Both APIs set an active
request status to `Evicted` before they release its request slot and remove its cache pins. They preserve an existing
terminal status. The trie cache can retain reusable blocks. The service returns `Evicted` when the caller tries to
resume an evicted session. While a bidirectional gRPC stream waits for its next input, it also waits for session
termination. Eviction ends that stream with gRPC `ABORTED`. The caller must start a new stream with the complete model
input.

The runtime does not invoke these APIs automatically. Request-slot pressure and page pressure require separate caller
policies. A later implementation can restore partially evicted or SSD-offloaded state at the resume boundary.

`StopSequence` and `LengthLimit` complete one turn. The bidirectional stream then waits for the next request.
`ContextLimit` terminates the runtime request and closes the bidirectional response stream after its completion event.

A request validation error, runtime error, or input stream error ends the RPC. A client input half-close prevents more
input. It does not cancel the active turn. The server finishes that turn, observes the half-close, and releases the
pending session. RPC cancellation drops the session and cancels its runtime request.

The server assigns one nonzero ID to the session. Each response envelope for all turns contains this ID. A one-turn
RPC uses the same lifecycle and releases the session after completion.

`GenerationCompletion.num_input_tokens` is the complete resident token count before sampling starts for that turn.
`GenerationCompletion.num_output_tokens` is the sampled token count for that turn.

Each response has one of these forms:

- `GenerateTokensResponse`: a token/probability chunk or one completion
- `GenerateMessagesResponse`: a reasoning chunk, text chunk, atomic tool call, or one completion

Completion reasons are `STOP_SEQUENCE`, `LENGTH_LIMIT`, and `CONTEXT_LIMIT`. EOF without a completion event
means that the stream failed.

### Codex protocol correspondence

The generation RPC is a model-serving substrate. It is not an agent protocol. A future Codex-compatible adapter can
own durable thread state, approvals, tools, commands, and item history above these RPCs.

| Current psi-dec concept | Codex app-server concept | Current difference |
| --- | --- | --- |
| gRPC bidirectional stream | initialized JSON-RPC connection plus a loaded thread | psi-dec has no handshake or client capabilities |
| resident runtime session | `Thread` loaded by `thread/start` or `thread/resume` | psi-dec state is transient and has no resumable thread ID |
| one `GenerateMessagesRequest` | `turn/start` input | psi-dec receives only the new append-only turn input |
| reasoning chunk | `item/reasoning/summaryTextDelta` | psi-dec streams model reasoning text, not an agent summary item |
| text chunk | `item/agentMessage/delta` | psi-dec has no `item/started` or `item/completed` envelope |
| atomic tool call | tool-call item lifecycle | psi-dec does not execute a tool or manage approval |
| generation completion | `turn/completed` | psi-dec reports a model stop reason, not an agent turn status |
| no current operation | `turn/steer` and `turn/interrupt` | future agent adapter owns active-turn control |
| no current operation | `thread/fork` | future agent adapter owns branch identity and state duplication |
| caller rebuilds context | `thread/compact/start` | Pi currently owns summary generation and retry |

Do not add durable thread operations to `GenerateMessagesStream`. A Codex-compatible service can translate a
`turn/start` into one generation turn and translate generation events into Codex items. This boundary keeps runtime
state reuse independent from agent persistence and tool execution.

The external diagnostic client remains a gRPC client:

```sh
cargo run --release --bin decode -- \
  --server-url http://127.0.0.1:50061 \
  --hf-model-dir "$PWD/models/Qwen3.6-35B-A3B-4bit" \
  --top-k 64 \
  --top-p 0.8 \
  --seed 59 \
  --chat-template auto \
  --prompt-str "Explain paged KV cache in one paragraph." \
  --max-sampled-tokens 8192 \
  --show-stats \
  --print-prompt
```

The client records the server ID from the first envelope. It requires the explicit completion event and streams
incremental text.

The client compares the incremental text with one final full decode. It reports inter-chunk metrics, not inter-token
metrics.

The client model directory supplies tokenizer and chat-template files. The server controls the Main model.

`--chat-template auto` runs the checkpoint template. It does not use a hard-coded Qwen prompt formatter.

Add `--disable-thinking` for the checkpoint-defined non-thinking generation prefix. `--chat-template raw` is an
explicit diagnostic bypass.

The [Pi provider guide](../agent-plugins/pi/README.md) gives the extension installation, configuration, and usage
procedure for `GenerateMessagesStream`.

## HTTP Chat Completions

The HTTP listener runs with gRPC. It provides the OpenAI-compatible `POST /v1/chat/completions` route.

### Request and response identity

Each Chat Completions request may supply an `x-request-id` header with a UUID.
The UUID is a request-correlation value.
The caller may reuse it when the caller replays a request.
The HTTP adapter generates a UUIDv4 when the header is absent.
In this case, a later replay gets a different request UUID.
The adapter returns the selected UUID in each successful response `x-request-id` header.

The request UUID remains in the HTTP protocol layer.
The runtime core and model executor continue to use the process-local numeric `RawRequestID`.
The response body uses an independent `chatcmpl-{response UUID}` ID.
Each new tool call uses an independent UUID as its tool-call ID.
The streaming `tool_calls[].index` value identifies the array position only.
It is not part of the tool-call identity.

Axum invokes the Chat Completions handler for each HTTP request.
The handler invocation owns its tool-call validation state.
The shared `HTTPServer` does not own a cross-request tool-call registry.
The handler reconstructs a tool-call ID set and pending-call map from the submitted conversation history.

The caller stores each tool-call UUID in the assistant message.
The caller uses the same UUID in the applicable tool-result message.
On resume, the caller sends the complete history again.
The new handler invocation scans this history and validates the correlation.
It rejects duplicate assistant tool-call IDs in one submitted history.
It does not compare tool-call IDs from independent HTTP requests.
It does not regenerate historical tool-call IDs.

### Matched gRPC and HTTP tests

The following command pairs use the same prompt, output limit, sampling values, seed, and thinking mode.
The HTTP `model` field is optional.
When the request omits it, the response uses the loaded executor model name.
When the request provides it, the value labels the response.
It does not select a different loaded model.

Use this gRPC command for a Qwen3 server:

```sh
cargo run --release --bin decode -- \
  --server-url http://127.0.0.1:50061 \
  --hf-model-dir "$PWD/models/Qwen3-14B-4bit" \
  --chat-template auto \
  --disable-thinking \
  --prompt-str "Reply with exactly: hello" \
  --max-sampled-tokens 16 \
  --temperature 0 \
  --top-k 1 \
  --top-p 1 \
  --seed 1
```

Use this equivalent HTTP command:

```sh
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
    "max_completion_tokens": 16,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "seed": 1,
    "enable_thinking": false
  }'
```

Use this gRPC command for a Qwen3.5/Qwen3.6/Qwen3.8 dense server:

```sh
cargo run --release --bin decode -- \
  --server-url http://127.0.0.1:50061 \
  --hf-model-dir "$PWD/models/Qwen3.8-27B-4bit" \
  --chat-template auto \
  --disable-thinking \
  --prompt-str "Reply with exactly: hello" \
  --max-sampled-tokens 16 \
  --temperature 0 \
  --top-k 1 \
  --top-p 1 \
  --seed 1
```

Use this equivalent HTTP command:

```sh
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
    "max_completion_tokens": 16,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "seed": 1,
    "enable_thinking": false
  }'
```

For the sparse Qwen3.5/Qwen3.6 server, use `"$PWD/models/Qwen3.6-35B-A3B-4bit"`.

### Response modes

Omit `stream` or set it to `false` to return one collected response:

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
    "max_completion_tokens": 16,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "seed": 1,
    "enable_thinking": false
  }'
```

Set `stream:true` to return server-sent events (SSE). `stream_options.include_usage:true` adds a separate usage chunk
before `[DONE]`:

```sh
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
    "stream": true,
    "stream_options": {"include_usage": true},
    "max_completion_tokens": 16,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "seed": 1,
    "enable_thinking": false
  }'
```

Both paths process messages and active function definitions through the shared `QwenCodec`. Each HTTP request creates
one `DecodeSession`.

The path maps the `MessageEvent` stream to content, tool-call, completion, and usage fields.

The service validates historical tool calls against their results. It does not validate them against currently active
definitions.

Thus, removing a tool prevents a new call without invalidating completed conversation history.

`tool_choice` supports `auto` and `none`. The service rejects `required`. It does not silently weaken the `required`
contract.

The endpoint is a documented OpenAI-compatible subset. It does not claim full API coverage.

The Pi `openai-completions` provider can use this endpoint directly. The wire adapter accepts these Pi fields:

- Non-persistent `store:false`
- Non-constrained `strict:false` tool definitions
- A leading `developer` message

Use this provider shape in `~/.pi/agent/models.json`. Set the model ID and limits to the loaded checkpoint:

```json
{
  "providers": {
    "psi-dec": {
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
      "models": [{
        "id": "qwen3.6-27b",
        "name": "psi-dec Qwen3.6 27B",
        "reasoning": true,
        "thinkingLevelMap": {"xhigh": "xhigh", "max": "xhigh"},
        "input": ["text"],
        "contextWindow": 262144,
        "maxTokens": 32768
      }]
    }
  }
}
```

Pi must send the complete message history on each turn. The service does not retain a conversation or model
continuation between HTTP requests.

A supplied `reasoning_effort` enables Qwen thinking. Streaming reasoning returns in `delta.reasoning_content`.
Later history can supply the reasoning as assistant `reasoning_content`.

The adapter rejects `reasoning_effort` with `enable_thinking:false` because the fields are contradictory.

## Operational notes

Run one GPU service at a time. The first request includes these operations:

- Model initialization
- Metal pipeline compilation
- Replay construction
- Cache warmup

Measure the first request separately from steady-state throughput.

Normal sampling submits these replay programs in one ordered Metal command buffer:

- MainTextEmbed
- MainResourceEmbed when the batch contains resource placements
- Main
- GatherUnembed
- Sampling

Speculative Main verification replaces Sampling with RejectionSampling.

An MTP proposal uses K ordered passes of MTPEmbed, MTP, GatherUnembed, and DraftSampling.
The public MTP lifecycle remains one `submit_spec -> wait -> read_spec` transaction.
The current implementation submits each dependent pass separately and reads its sampled token on the CPU before it
prepares the next pass.
All passes reuse the same recorded replay programs, weights, scratch, and bound workspaces.

A DSpark-enabled batch records DSpark Prefill after Main in the same ordered GPU submission.
DSpark Prefill creates persistent DSpark history K/V from the Main capture.
A decode-ready batch then records GPU Spec Decode prepare and DSpark Spec Decode before that submission.
DSpark Spec Decode contains DSparkEmbed, DSpark, DSparkGatherUnembed, and DSparkSampling.

A DFlash2-enabled batch uses the same integrated Main submission through its separate owner.
DFlash2 Prefill creates persistent DFlash2 history K/V from the Main capture.
A decode-ready batch records GPU Spec Decode prepare and DFlash2 Spec Decode after Prefill and before submission.
DFlash2 Spec Decode contains DFlash2Embed, the DFlash2 body, and DFlash2Output.
The body combines per-row sliding-history attention with bidirectional local-block attention and applies dynamic
grouped convolution.
DFlash2Output builds and samples the candidate lattice and writes sparse draft distributions.

`--logging debug` emits one DEBUG `phase="executor.batch.perf"` event after each non-empty executor batch. The event
uses the same schema for Vanilla, MTP, DSpark, and DFlash2. It also uses the same schema for prefill and decode batches:

```text
component="executor"
phase="executor.batch.perf"
model="qwen3.5"
model_mode="vanilla|mtp|dspark|dflash2"
batch_seq=42
num_reqs=4
num_input_tokens=12
num_spec_tokens=9
num_verified_tokens=6
acceptance_rate=0.6667
num_spec_token_by_index=[4, 3, 2]
num_verified_token_by_index=[3, 2, 1]
acceptance_rate_by_index=[0.7500, 0.6667, 0.5000]
executor_cpu_ms=20.7500
main_cpu_ms=14.2500
spec_cpu_ms=6.5000
main_gpu_ms=10.7500
rejection_gpu_ms=0.3500
spec_prepare_gpu_ms=0.0200
spec_prefill_gpu_ms=1.1000
spec_decode_gpu_ms=1.8000
spec_gpu_ms=2.9200
spec_passes=2
```

`num_verified_tokens` is the number of returned `validated_tokens`. It does not include the final `sampled_token`.
`acceptance_rate` is `num_verified_tokens / num_spec_tokens`.

The per-index rate is conditional. Index `i` is eligible only when the request contains speculative token `i` and
all earlier speculative tokens passed verification. `num_spec_token_by_index[i]` is this eligible count.
`num_verified_token_by_index[i]` is the verified count at the same index. The helper scripts sum these counts across
batches before they calculate `acceptance_rate_by_index`.

`main_cpu_ms` covers
`MainTextEmbed -> MainResourceEmbed? -> Main -> GatherUnembed -> Sampling/RejectionSampling -> submit/wait -> read`.
For Qwen3 and Qwen3.5 DSpark, and for Qwen3.5 DFlash2, it also covers GPU Spec Decode prepare, Spec Prefill, Spec
Decode, proposal sampling, and proposal read in the same ordered submission.
`spec_cpu_ms` covers a separate Spec lifecycle. Qwen3.5 MTP uses this boundary. DSpark and DFlash2 do not, so their
`spec_cpu_ms` is zero.
DSpark or DFlash2 prefill-only batches can have nonzero `main_cpu_ms` and zero `spec_passes`.
`executor_cpu_ms` is `main_cpu_ms + spec_cpu_ms`. Use `executor_cpu_ms` to compare the complete CPU-observed
Main-to-Spec lifecycle across the old split DSpark or DFlash2 lifecycle and the integrated lifecycle. Do not compare
old `main_cpu_ms` directly with integrated `main_cpu_ms` because the integrated value includes fixed-block Spec work.
`spec_passes` counts Spec Decode forwards. The `*_cpu_ms` fields are CPU-observed wall latencies. They include GPU
wait time and do not measure CPU active execution. The `*_gpu_ms` fields are GPU-observed Metal timestamp durations.

`MetalRuntime` enables relaxed Metal 4 GPU timestamps for Main and MTP submissions by default.
No environment variable is required. The runtime reuses its timestamp counter heap.

The GPU timing fields have these meanings:

- `main_gpu_ms` covers Main work before RejectionSampling. It includes ordinary Sampling when no rejection stage
  exists.
- `rejection_gpu_ms` covers RejectionSampling only.
- `spec_prepare_gpu_ms` covers Spec Decode prepare.
- `spec_prefill_gpu_ms` covers DSpark or DFlash2 Spec Prefill.
- `spec_decode_gpu_ms` covers Spec Decode and proposal sampling.
- `spec_gpu_ms` is `spec_prepare_gpu_ms + spec_prefill_gpu_ms + spec_decode_gpu_ms` for DSpark and DFlash2.
  For MTP, it sums all logical MTP submission durations. It excludes host recording, readback, and gaps between submissions.

The Spec prepare, Prefill, and Decode fields apply to the integrated DSpark and DFlash2 submission.
They are `unavailable` for MTP. The MTP total is `unavailable` if any logical submission lacks valid GPU timestamps.
An absent stage in an enabled integrated submission reports `0.0000`. All GPU fields report `unavailable` when the
Metal timestamps are unavailable or Metal does not return a valid ordered timestamp sequence. This fallback does not change
submission completion or output readback.

`--logging info` does not emit the executor batch performance event. `--logging debug` also emits request and response
diagnostics. The end-to-end performance helpers enable only the `inference-runtime-service::perf` DEBUG target when
the selected server logging level is INFO.

### Numerical failure diagnostics

For a Qwen3.5 numerical failure, run the following command from the repository root.
It records the local revision, OS build, hardware, command, batch diagnostics, and panic output in one file.
The model paths below match the 27B Main/MTP configuration. Keep the same input request that reproduces the failure.

```sh
set -o pipefail
{
  git rev-parse HEAD
  git status --short
  sw_vers
  sysctl -n machdep.cpu.brand_string hw.memsize
  set -x
  PSI_QWEN35_CHECK_FINITE=1 PSI_QWEN35_STATE_TRACE=1 RUST_BACKTRACE=full \
    cargo run --release --bin qwen3_5_dense -- \
      --grpc-listen-addr 127.0.0.1:50061 \
      --http-listen-addr 127.0.0.1:8000 \
      --hf-model-dir ~/Workspace/models/Qwen3.8-27B-4bit \
      --hf-spec-model-dir ~/Workspace/models/Qwen3.8-27B-MTP-4bit \
      --spec-type mtp --num-spec-tokens 1 --logging debug
} 2>&1 | tee /tmp/psi-dec-numerical-debug.log
```

Send the complete log and the reproducing request. A screenshot omits the preceding request and stage context.
The numerical panic identifies the first observed failing checkpoint, its buffer coordinates, value, and bit pattern.
Sampling failures include the available request slot, sample position, RNG domain, and sampling parameters.
Metal submission failures include the Foundation description of `userInfo`, including underlying error descriptions.
Use the preceding batch request log to associate a local row or request slot with the request ID.
The [Qwen guide](executor_qwen.md#numerical-checkpoints) defines checkpoint coverage and limitations.
The checks add GPU work. Do not compare this run's throughput with a normal run.

### Lifecycle diagnostics

Internal model `Start` and `Stop` commands emit INFO lifecycle events on the
`inference-runtime-service::lifecycle` target.
The events use `component="model"` with `start.begin`, `start.complete`, `stop.begin`, and `stop.complete` phases.
Completion events include elapsed milliseconds.
Failure events include the model name and error, and then trigger global shutdown.
Runtime core issues these commands after the configured executor hibernation timeout.

Executor profiling spans use the lifecycle hook names:

```text
embed_main -> forward_main -> unembed_main -> sample_main
submit_main -> read_main

embed_spec -> forward_spec -> unembed_spec -> sample_spec    MTP

Main -> RejectionSampling -> Spec Decode prepare -> Spec Prefill    DSpark or DFlash2
  -> Spec Decode -> proposal sampling
submit_main -> read_main
```

A pass is one proposal forward that runs.
For Qwen3.5, each logical MTP step, DSpark Decode block, or DFlash2 Decode block is one pass.
For Qwen3, each DSpark block forward is one pass.
Qwen3 runs DSpark Prefill for a prefill-only batch.
It does not run DSpark Decode because no sampled anchor exists.

The runtime emits non-empty periodical scheduler stats every 30 seconds. It resets these stats after each output.
Runtime shutdown always emits separate lifetime scheduler stats. It does not reset the lifetime stats.

Each scheduler stats output contains three tables. The scheduler API table contains enqueue and async-task counts.
It also contains prepare, cancel, commit, and execution counts and latency percentiles.
The Speculative acceptance table uses proposal
indexes as columns:

```text
spec stat | overall | index@0 | index@1 | ...
proposed  | ...
accepted  | ...
rate      | ...
```

The `overall` column sums the proposed and accepted counts across all indexes. Each remaining column reports one
proposal index. `proposed` counts each proposal position that the Spec forward produced. `accepted` counts the
responses whose `validated_tokens` include that position. `rate` is `accepted / proposed`. A column with no proposals
has rate `N/A`.

This rate differs from the conditional `acceptance_rate_by_index` in the executor batch performance event. The
scheduler denominator includes every produced proposal at the index. The executor event includes an index only when
all earlier proposal tokens passed verification.

The `GPU Execution Latency` table follows `Spec Acceptance`.
It reports GPU stage labels, measured batch counts, latency percentiles, and average latency.
It uses the same periodical and lifetime windows. It does not include CPU timings.
Runtime core transports and aggregates executor-owned labels and durations.

- `main` covers embed, all Main layers, gather/unembed, and sampling or rejection sampling.
  It sums the existing `main_gpu_ms` and `rejection_gpu_ms` boundaries when rejection sampling runs.
- `mtp` sums the GPU durations of all logical MTP submissions in the batch.
  Each submission includes hidden-state transfer, embed, all module layers, gather/unembed, and sampling.
- `dspark` or `dflash2` uses the existing `spec_gpu_ms` total for Spec prepare, Prefill, Decode, and proposal sampling.

The runtime collects these durations by default.
Unavailable GPU durations do not enter the histogram. A row with no measurements has count `0` and latency `-`.
A measured zero GPU duration remains a sample.
The scheduler emits this table at INFO even when executor batch DEBUG events are disabled.

Long-running service and runtime components use these spans:

```text
runtime
replayable-executor
async-task-pool
s3-fifo
grpc-server
http-server
```

Each component emits one INFO `started` event and one INFO `stopped` event in its span. Normal shutdown receipt does not
emit a separate INFO event. The `runtime` span belongs to the runtime-core event loop in
`runtime/scheduler/event_loop.rs`. It is not a second service-level wrapper.

Enqueue and swap-in operations report request counts. Prepare, cancel, and commit operations report counts and latency
percentiles.

Set `PSI_QWEN35_STATE_TRACE=1` to write executor lifecycle lines to standard error. These lines include:

- Replay cache hit or miss keys
- GDN restore and commit/publish decisions
- `gdn_commit_wait` for pending state reconstruction and page publication
- Synchronous `prepare_sync` timing

The timing fields are `gqa_us`, `gdn_states_us`, dependent `gdn_metadata_us`, and total `wall_us`.

Set `PSI_GDN_STATE_TRACE=1` only for the detailed GDN request-state transition trace. Both settings are diagnostic.
Leave them unset for throughput measurements.

`--profile component` and `--profile operation` enable the same coarse CPU tree. It contains prepare, model
input/forward/output, and commit.

These modes do not attribute GPU time to components or kernels. Use Metal capture or counters for GPU time.

## Correctness and long decode

For a release correctness check, first use deterministic sampling:

```text
--temperature 0 --top-k 1 --top-p 1 --seed 1
```

Use a prompt with an objective oracle. Then, run a long 8K-token generation.

Validate the dense and sparse models. If MTP paths changed, validate MTP off and on.

Record these facts:

- Prompt tokens and sampled tokens
- Termination reason and output sanity
- Commit and dirty state
- Model directory

Investigate cold-start stalls separately from later decode stalls.

## End-to-end performance helper

The Qwen3 helper runs Main-only and DSpark decode measurements:

```sh
scripts/qwen3_e2e_decode_perf.sh \
  --model <qwen3-model-dir> \
  --dspark <qwen3-affine-dspark-model-dir> \
  --runs 7
```

The helper uses the model directory for tokenization by default.
Use `--tokenizer` to select a different directory.
Use `--tokens` to select the comma-separated output-token counts.
The default Qwen3 matrix runs `14b_off` and `14b_dspark`.
The `14b_dspark` case uses the checkpoint `block_size` as its proposal count.
If the DSpark checkpoint is absent and no download repository is configured, the helper prints a warning and skips
that case. A missing Main checkpoint remains an error.

The Qwen3.5/3.6/3.8 helper runs controlled 27B/35B Main-only, MTP, DSpark, and DFlash2 comparisons:

```sh
scripts/qwen35_e2e_decode_perf.sh \
  --tokenizer <tokenizer-model-dir> \
  --model-27b <27b-model-dir> \
  --mtp-27b <27b-mtp-model-dir> \
  --dspark-27b <27b-affine-dspark-model-dir> \
  --dflash2-27b <27b-affine-dflash2-model-dir> \
  --model-35b <35b-model-dir> \
  --mtp-35b <35b-mtp-model-dir> \
  --dspark-35b <35b-affine-dspark-model-dir> \
  --dflash2-35b <35b-affine-dflash2-model-dir> \
  --runs 7
```

Use `27b_mtp1` through `27b_mtp4` and `35b_mtp1` through `35b_mtp4` to select an MTP proposal count.
The `*_mtp` aliases run MTP proposal counts 1, 2, 3, and 4.
Use `*_dspark_k1`, `*_dspark_k2`, `*_dspark_k3`, `*_dspark_k4`, or `*_dspark_k8` for DSpark.
Use the matching `*_dflash2_kN` names for DFlash2.
The base `*_dspark` and `*_dflash2` cases use checkpoint defaults.
The current DSpark checkpoint has `block_size=7`: seven drafts.
The current DFlash2 checkpoint has `dflash_config.block_size=8`: one anchor and seven drafts.
An explicit K=8 requests eight drafts. It is not the DFlash2 checkpoint's default eight-row query block.
Pass `--block-spec-tokens K` to override K for every block-spec case.
The default case matrix uses MTP counts 1 through 4 and checkpoint-default block-spec counts.
Pass explicit `_k1`, `_k2`, `_k3`, `_k4`, or `_k8` case names to compare other block-spec counts.

1. Main-only `27b_off` and `35b_off`
2. MTP counts 1, 2, 3, and 4 for each model
3. Checkpoint-default DSpark and DFlash2 for each model

The 27B cases use output limits 256 and 1024. The 35B cases use output limits 256 and 1024.

The default 27B Main and MTP checkpoints use Qwen3.8. The default 35B Main and MTP checkpoints use Qwen3.6.
Each case uses its Main checkpoint for tokenization by default. Use `--tokenizer` to override all cases.
The default `representative2` workload contains one fixed GSM8K prompt and the Chinese travel prompt (`chat`).
The helper repeats and summarizes each prompt independently. The default output contains a configuration table,
measurement progress, and a compact result table. The result table contains case, prompt, output limit, sampled tokens,
decode throughput, tokens per chunk, and verified/proposed speculative tokens. Use `--show-runs` to also print
machine-readable `CONFIG`, `RUN`, and `SUMMARY` rows. Each `RUN` and `SUMMARY` row contains the stable prompt ID. The
configuration output contains the prompt-set name, prompt IDs, prompt count, and prompt-set SHA-256.
For each case, prompt, and output limit, the helper hashes the generated output bytes before the performance footer.
It stops with `DETERMINISM_FAILED` if repeated runs with the fixed sampling configuration and seed have different hashes.
Use `--prompt <text>` to replace the set with one custom prompt.
If an MTP, DSpark, or DFlash2 checkpoint is absent and no download repository is configured, the helper prints a
warning and skips that case. A missing Main checkpoint remains an error.
Before it starts a DFlash2 case, the helper validates the affine config and safetensors headers.
It rejects a checkpoint that contains a BF16 matrix.
The helper stops the server after each explicit case.
Its process guard matches Qwen and Decode release-worker paths, the runtime service, and Cargo run/bench commands.
The guard does not match the perf helper filename in a `caffeinate` launcher command.
It applies the configured cooldown between runnable cases. The default cooldown is 8 seconds.
Each Spec summary label includes its actual proposal count, including checkpoint defaults.
Examples are `27b_mtp4`, `27b_dspark_k7`, and `27b_dflash2_k8`.
A global override replaces explicit case counts before deduplication.

Both helpers record these facts:

- Commit and dirty state
- Model directories
- Machine and operating system
- Cache and request capacity
- Scheduler capacities
- Sampling configuration and seed
- Prompt identity
- Trajectory fields

Both helpers also record cooldown and speculative-acceptance fields.

The [latest Main-only serving summary](executor_benchmarks.md#main-only-serving-results-2026-10-14) records the user-supplied Off results received on 2026-10-14.
The helper's checked-in comparison baseline remains the separate user-supplied table received on 2026-09-19.
The user identified main tip at receipt: `d6dc5da7`, with source tree `38f972e038150ee351c16dba6ae33bde7242670f`.
This reference precedes history cleanup and the SlicedQ pipeline-capacity fix. It does not measure that fix.
Dirty state is unknown. The helper retains the existing machine and configuration profile.
The table provides throughput, output counts, tokens per chunk, and accepted/proposed counts.
It does not provide raw CONFIG/SUMMARY rows, run count, variance, output hashes, TTFT, or inter-chunk latency.
The script does not invent these fields. Its configuration checks use the producing script's defaults.
The reference contains 48 rows at output limits of 256 and 1024 tokens.
It covers Main-only and MTP1 through MTP4 for both models, plus 27B DSpark K7 and DFlash2 K7.

A matching configuration and matching reported work counts produce `reference_status=summary-only`.
The `observed_decode_delta_pct` field is a descriptive ratio, not a controlled performance verdict.
A mismatch or missing row suppresses the delta. The reference is not a pass/fail threshold.
The reference records unknown provenance explicitly. The summary-only comparison does not establish a controlled performance verdict.
Use `--show-runs` for per-row deltas and mismatches. Use `--no-reference` to disable the comparison.
Run `python3 scripts/qwen35_e2e_decode_perf_test.py` for CPU-only script contract tests.

Use the [cross-engine serving matrix](executor_benchmarks.md#cross-engine-serving-matrix) for exact raw-token, retained-context, and request-count comparisons.
The matrix records fixed proposal counts, actual cache reuse, output counts, and engine provenance.

The Qwen3 helper does not contain a checked-in reference run. Record its comparison results outside the script with the
complete provenance required by [`executor_benchmarks.md`](executor_benchmarks.md).

The current default uses four running requests.

Summaries report these metrics:

- Decode throughput
- TTFT and prompt throughput
- RPC inter-chunk p50 and p95
- Tokens for each chunk
- Exact verified/proposed speculative-token rate for the matching server-log interval
- Conditional speculative-token acceptance rate at each index

For Main-only decoding, a chunk contains one token. Thus, inter-chunk time is inter-token latency.

With MTP, inter-chunk time measures burst cadence. Interpret it with tokens for each chunk and the acceptance rate.

Use `--case-cooldown-secs 0` only for an intentional sustained-load experiment.

Follow [`executor_benchmarks.md`](executor_benchmarks.md) before you make a performance claim.
