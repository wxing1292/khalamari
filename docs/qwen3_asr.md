# Qwen3-ASR

This document describes the current Qwen3-ASR implementation.
It covers the supported checkpoint, audio preparation, resource materialization, text-and-audio composition, and the
transcription service.

[`core.md`](core.md) defines runtime scheduling and cache ownership.
[`executor_qwen.md`](executor_qwen.md) defines the shared Qwen3 text-decoder execution path.
[`service.md`](service.md) defines startup and HTTP/gRPC operation.
Unresolved work is in [`future_work.md`](future_work.md).

## Supported scope

| Boundary | Supported contract |
| --- | --- |
| Checkpoint | `mlx-community/Qwen3-ASR-1.7B-8bit`, Apple Silicon |
| Decoder/encoder | Shared Qwen3 decoder, model-owned Audio executor and processor |
| Input | One WAV/resource/request, integer or F32 -> mono F32 at 16 kHz |
| Duration | At most 30 seconds after resampling |
| Server output | Greedy, collected HTTP/gRPC response |
| Resource lifetime | Private/request, common async-task/resource lifecycle, no cross-request materialization sharing |

Runtime scheduling has no audio-specific states.

## Source layout

| Root under `crates/` | Owner files |
| --- | --- |
| `inference-executor-core/src/model/qwen/v3_asr/` | `config.rs`, `input.rs`, `weight_layout.rs` |
| `inference-backend-metal/src/operators/` | `matmul_bf16.rs`, `bias_activation_bf16.rs`, `conv2d_unfold.rs` |
| `inference-backend-metal/src/components/` | `tower_block_attention.rs`, `audio_encoder_layout.rs`, `layer_norm.rs`, `resource_embed.rs` |
| `inference-executor-metal/src/model/` | `resource_arena.rs`, `resource_embed.rs`, `main_resource_embed.rs` |
| `inference-executor-metal/src/model/qwen/v3_asr/` | `audio.rs`, `resource.rs` |
| `inference-executor-metal/src/model/qwen/v3/executor/` | `mod.rs`, text/resource composition |
| `inference-runtime-core/src/runtime/` | `resource/`, `decoder/resource.rs`, `tasks/resource_materialization.rs` |
| `inference-runtime-service/src/` | `api/audio.rs`, `asr/`, `rpc/`, `bin/qwen_server/asr.rs`, `bin/qwen3_asr.rs` |

## Owner boundary

| Owner | Responsibility |
| --- | --- |
| Service | Shared `api/audio.rs` request admission and WAV conversion, resampling/normalization/log-Mel, prompt/placeholder tokens, prepared-source registration, language/result parsing |
| Runtime core | Resource identity/placement/segments, request/allocation lifetime, cache hit/miss, async tasks |
| ASR model integration | Shared resource arena, processor, standalone encoder worker/stream/weights, hidden replacement map |
| Shared Qwen3 executor | Text embedding, decoder, unembedding, sampling |
| Metal backend | Buffers, kernels, replay submission |

The initializer shares one resource arena between encoder and decoder.

Backend constructors validate static shader dimensions, including the MLX LayerNorm loop tile.
Invocation constructors validate dynamic counts, signed coordinates, padded layouts, and attention window ends.
They also validate bound buffer lengths and byte ranges, including byte offsets, before command recording.
Conv2D unfold input and output must use separate buffers. Invocation construction rejects aliases.
Invalid shapes or buffer ranges panic. Command recording and recorded submissions do not repeat these checks.
Single-row BF16 matmul uses MLX GEMV. Its invocation requires at least four output columns and weight element indices
that fit `i32`. Multi-row GEMM does not use these GEMV limits.

Runtime core does not decode media and does not parse model-specific metadata.
The executor does not schedule requests.

## Request flow

The service prepares CPU media before it submits the request to runtime admission:

```text
WAV bytes
  -> decode samples
  -> mix channels to mono
  -> resample to 16 kHz
  -> normalize
  -> calculate 128-bin log-Mel features
  -> calculate num_resource_tokens
  -> expand <|audio_pad|> placeholders
  -> create ResourcePlacement
  -> submit DecodeRequest
```

`ResourcePlacement` is stable for the request lifetime.
Each placement tuple is `(token_index, resource_index, num_resource_tokens)`.
`token_index` is absolute in the initial request token sequence.
`resource_index` is logical in the Audio Tower output sequence.

Runtime intersects each placement with cache-block token ranges.
The result is a `ResourceSegment` annotation with a block-local token index, a logical resource index, and a length.
The annotation does not contain an arena address.

The cache path is:

```text
initialize one cache block
  -> block cache hit
       -> keep the resource symbolic
       -> do not run audio materialization

  -> block cache miss with a resource segment
       -> resource is concrete
            -> continue Prefill

       -> resource is symbolic
            -> return a blocking ResourceMaterializationReq
            -> dispatch by ResourceTypeID
            -> run the Audio encoder executor
            -> return a ConcreteResource
            -> reschedule the request
            -> initialize the block again
```

The async task does not hold the complete internal request.
[`core.md`](core.md) defines task blocking, terminal release, and cache-block reservation behavior.

## Audio preparation and Audio Tower

The preprocessor derives its contract from `preprocessor_config.json`.
The supported checkpoint uses these values:

```text
sample_rate:       16000 Hz
feature_size:      128 Mel bins
n_fft:             400
hop_length:        160
n_samples:         480000
max_frames:        3000
```

The service uses an offline band-limited resampler when the WAV sample rate is not 16 kHz.
It calculates log-Mel features on CPU.
It sends the prepared feature tensor to the model-owned Audio encoder executor only after a required cache block
misses.

The Audio Tower executes the checkpoint encoder, chunk layout, and final projection.
The Audio encoder executor owns the worker thread, Metal stream, replay recording, weight residency, and submission.
The runtime-facing processor only resolves the prepared source and forwards it to the executor.
It writes decoder-width BF16 embeddings directly to a `MetalResourceArena` allocation.
The output-row contract is:

```text
complete_chunks = num_frames / 100
tail_frames = num_frames % 100
num_resource_tokens = complete_chunks * 13 + ceil(tail_frames / 8)
```

`ConcreteResource` owns the arena `OffsetAllocation`.
The allocation uses byte offsets and byte lengths.
Its RAII lifetime returns the range to the allocator after all request-owned device work is complete.

## Text and audio composition

The service constructs this prompt shape:

```text
<|im_start|>system
{context}<|im_end|>
<|im_start|>user
<|audio_start|>{audio placeholders}<|audio_end|><|im_end|>
<|im_start|>assistant
{optional forced-language prefix}
```

The service inserts audio placeholder token IDs directly.
It does not ask the tokenizer to parse a repeated placeholder string.
The number of placeholder tokens equals `num_resource_tokens`.

The executor first runs the normal vocabulary embedding for all query tokens.
If active resource rows exist, `ResourceEmbed` then replaces these hidden rows:

```text
destination_hidden_row <- source_resource_embedding
```

The model-level operation is replacement.
The Metal kernel can implement the source read as a gather.

`MappingTable` retains the configuration that defines its source and destination byte ranges.
The invocation constructor must reject a table whose configuration differs from the compute configuration.
Table construction must bound both dispatch values and mapping ABI indices by the shader `u32` domain.

The Audio Tower output width must equal the Qwen3 text hidden width.
The executor addresses a source row as:

```text
arena_offset_bytes + resource_index * hidden_dim_bytes
```

The executor receives the concrete byte range in `DeviceRequest`.
It does not look up an arena allocation through `ResourceID`.

Text-only Qwen3 uses the existing text embedding path.
It does not record or submit `ResourceEmbed`.

## Position model

Qwen3-ASR uses the request token index as its logical position.
The audio placeholder rows occupy normal consecutive token positions.
The Qwen3 text decoder applies the same scalar position to each interleaved M-RoPE axis.
The ASR path does not calculate image or video T/H/W positions.
It does not calculate a vision `rope_delta`.

## Restart and release contract

`ResourceID` remains stable for one admitted request.
`ResourceURI` identifies the registered prepared source.
The URI is opaque to runtime core.

A symbolic resource can become concrete after materialization.
A concrete resource becomes symbolic after all its placement ranges enter committed cache and the related device work
retires.
The transition drops the request-owned arena allocation, including when the last placement ends in a partial mutable
cache block.
The current service keeps the prepared source registration alive until transcription decode completes.

The internal request must retain a concrete resource while an in-flight device request can read its allocation.
Dropping the concrete resource releases its arena range.
Dropping the prepared-source registration removes the service-owned source.

Resource allocations are process-local and are not restart-durable.
A process restart cancels active requests and rebuilds resources for later requests.

## Verification

The checked-in tests cover these contracts:

- Strict checkpoint and preprocessor normalization.
- Prepared-audio dimensions and Audio Tower output-row geometry.
- Log-Mel values against a reference fixture.
- Tokenizer special-token IDs and an exact prompt-token fixture.
- Resource identity, placement, cache-block annotation, async-task, and RAII allocation lifecycle.
- Audio Tower weight bindings and numerical execution against the supported checkpoint.
- `ResourceEmbed` source and destination mappings.
- Text-only and resource-aware Qwen3 executor composition.
- Requested language, detected language, absent metadata, and empty transcription parsing.
- Qwen3-ASR CLI and runtime-capacity configuration.

The supported checkpoint and one real WAV sample also pass the release service path through
`POST /v1/audio/transcriptions`.
