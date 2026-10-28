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
| Server output | Greedy, collected HTTP/gRPC response; continuous client output uses language-aware speech snapshots |
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

The Rust client lives in `crates/inference-asr-client/`.
The executable owns capture, streaming and external checkpoint preparation. The ASR workflow requires no Python.

| Client source | Owner |
| --- | --- |
| `audio.rs` | FFmpeg child and pipes, canonical PCM, shared WAV encoding and recording lifecycle |
| `assets.rs` | Pinned Whisper download/copy, byte and SHA-256 validation, atomic publication |
| `config.rs` | One Serde/Clap schema, defaults, explicit CLI precedence and external validation |
| `session.rs` | `AudioSession`: capture, processing, output and recording lifetime |
| `segment.rs` | `AudioSegmenter`: sample timeline, bounded audio, immutable `AudioSegmentReq` snapshots, endpoints, revisions and final ordering |
| `models.rs` | CPU Silero state, native CT2 language detection/alignment and log-Mel |
| `language.rs` | Bounded acoustic observations and candidate language ranges |
| `transcribe.rs` | Shared CPU model/cache, Qwen routing, last-result reuse, Mandarin retry and alignment |
| `stream.rs` | `AudioStream`: VAD, segmenter, bounded tasks, cancellation and committed overlap history |
| `overlap.rs` | Timestamp ownership and deterministic reconciliation |
| `service.rs`, `service/stream.rs` | Collected WAV calls, duplex PCM transport, bounded commands, request cancellation and connection statistics |
| `output.rs` | Exclusive JSON/text/editor selection, output errors and the shared edit loop |
| `editor.rs`, `document.rs` | Raw event log, minimal text projection, explicit corrections and final-text snapshot |
| `terminal.rs` | Independent input descriptor, UTF-8 command input and shared display formatting |
| `main.rs` | Startup commands, replay and process cancellation |

The native `asr_e2e` example generates fixtures and replays the production executable through the service.

### Client composition and trust

The client resolves configuration once before opening models, files, or capture resources.
Each configuration field declares its JSON key and CLI option in the same owner.
Only command-line values with `ValueSource::CommandLine` override the file.
An explicit value equal to a built-in default still overrides the file.
The loader validates the file schema before it applies overrides.
It calculates the omitted sentence threshold once. Runtime code receives a concrete number.
The example configuration contains only workload overrides.

```text
example window: x[0:32000], mono i16, 16000 Hz

FFmpeg emits x[0:512], x[512:1024], ...
  +-> optional Recording: same PCM -> fixed WAV format -> buffered file
  |
  +-> collected mode: retain x[0:32000] -> one HTTP/gRPC request
  |
  +-> AudioStream owns VAD + AudioSegmenter + bounded tasks
        VAD: each PCM frame -> speech flag
        AudioSegmenter: source sample clock -> immutable AudioSegmentReq x[start:end]
          AudioTranscriber: shared Whisper gate/cache + immutable service client
            LID ranges -> Qwen requests -> text/spans + per-job metrics
              ordered final events -> publish overlap history once
                Output selects JSON, plain text, or editor
                  editor: raw event -> append log -> current text projection

HTTP/gRPC request -> service WAV validation -> log-Mel[128, frames]
  prepared source -> runtime cache miss -> Audio Tower -> resource embeddings
    decoder + embeddings -> transcription text -> client response
```

`AudioSession` owns capture, processing, terminal input, saved WAV and output for one run.
Its event loop waits for capture, completion, terminal input or a deadline.
`AudioStream` owns the service connection, streaming admission and completion. `AudioSegmenter` owns source coordinates and segment state.
It schedules available jobs before it waits for capture, completion, terminal input, or a deadline.
`AudioSegmentReq` owns its immutable audio allocation. Its read end derives from the allocation length.
Its final status derives from its endpoint reason.
Service-call metrics belong to the segment request. Streaming requests share one duplex connection for the recording.
Submitted audio counts remain integer samples until JSON serialization.

| Boundary | Contract |
| --- | --- |
| CLI/JSON configuration | Validate schema, values, paths and relationships once |
| FFmpeg pipe | Validate complete i16 samples, retain incomplete frames across cancellation |
| Submitted job | Keep immutable audio and stable source coordinates after pruning |
| Service/model output | Validate bounded responses and native output contracts at their owners |
| Live document update | Apply trusted session output without repeating replay validation |
| Replay or manual correction | Validate external coordinates, commit order or current revision before trusted commit |
| Output failure | Return the I/O error. Only invalid editor commands remain recoverable in the view |

Normal EOF drains inference. The editor can then use the same edit loop as event replay.
An empty terminal command requests stop, drains pending inference, saves output, and exits.
Cancellation drops the stream tasks and capture child. The editor attempts to save accepted final text and restores its terminal.

The raw event and document line have different lifetimes.
The log retains acoustic spans and metrics. The document retains only text, revision, and source coordinates.
The plain view retains one rendered draft. Both terminal views share formatting and JSON output routines.

Whisper and Qwen retain separate log-Mel implementations because their preprocessing contracts differ.
Whisper appends 160 zero samples and uses its own frame/padding convention and window precision.
Qwen derives its feature size and frame limits from the checkpoint and keeps preparation scratch local to each RPC.
Whisper compacts a truncated feature matrix in place after full normalization.
The service normalizes decoded input before resampling. It repeats normalization only when resampling can introduce overshoot.

### Task contracts

The client uses concrete request and response contracts at each processing boundary.

| Contract | Scope |
| --- | --- |
| `AudioSegmentReq` / `AudioSegmentResp` | One segment revision: immutable PCM snapshot to text, spans and metrics |
| `AudioLangRecognitionReq` / `AudioLangRecognitionResp` | One source range to merged language observations and work count |
| `AudioTranscribeReq` / `AudioTranscribeResp` | One collected WAV request to Qwen text and language |
| `AudioStreamReq` / `AudioStreamResp` | PCM append/release, range recognition, cancellation and graceful connection finish |

A segment can produce several requests. Its revision is allocated only at submission.
The session schedules asynchronous work with `JoinSet`. It cancels individual obsolete partials by their abort handles.
The client reuses `SyncTaskPool` for CPU language analysis and alignment.
The optional `Acoustics` owner creates and releases its model, language state, CPU pool and shutdown guard together.
Forced-language jobs without overlap do not create this owner.
`AudioTranscriber` retains a private shared connection handle for recognition. It does not expose transport lifecycle operations.
Its result cache keys use recording sample ranges and language. Recording coordinates identify immutable PCM within one connection.
The cache does not copy or compare PCM. Model and prompt context remain fixed for the connection.
Each language response counts its own uncached observations. The analyzer retains only its bounded observation cache.
The runtime core `AsyncTaskPool` retains its runtime request IDs and scheduler completion channels.
The client does not adapt audio segment identity to those runtime IDs.
Task futures own execution. The client does not add a second task struct that duplicates a request.

```text
AudioSession
  PCM i16[512] -> AudioStream -> AudioSegmenter -> AudioSegmentReq[segment, revision]
    append new PCM once -> AudioStreamClient -> service PCM cache
    spawn job -> AudioTranscriber
      shared CPU gate -> SyncTaskPool -> AudioLangRecognitionResp[ranges]
      private connection handle -> recognize[start,end,hint] -> HTTP WebSocket or gRPC -> text/language
      optional CPU alignment -> AudioSegmentResp[text, spans, metrics]
    accept revision / order finals -> AudioEvent -> Output -> Editor -> Document
```

Collected WAV RPCs retain their request and response schemas.
Streaming events retain their partial/final revision contract and add a `transport` statistics snapshot.
The editor owns the document through the session output path. Replay can create the same editor without audio capture.

### Incremental audio transport

Capture, FFmpeg conversion, VAD, Whisper language analysis, segment boundaries and transcript revisions remain in the client.
One recording owns one connection to the service.
The service owns a bounded PCM cache for that connection. It does not own the client's speech-session policy.
The HTTP endpoint is `GET /v1/audio/transcriptions/stream`, with a WebSocket upgrade and protobuf binary messages.
The gRPC method is `InferenceRuntime/StreamAudio`, with the same protobuf messages and shared RPC loop.
The collected WAV endpoints remain available.

```text
same source x[0:32000], mono i16 at 16000 Hz
  client: VAD + language ranges + partial/final revisions
    append x[0:16000] ------------------> connection cache x[0:16000]
    recognize [0,16000) ----------------> PCM -> log-Mel -> Audio Tower + decoder
    append x[16000:32000] --------------> connection cache x[0:32000]
    recognize [0,32000) ----------------> reuse cached PCM; run independent ASR
    release before 16000 --------------> keep x[16000:32000]

snapshot for an admitted request: shared byte views -> CPU preparation -> model resource
  cache release cannot invalidate those views
```

The client uploads audio only when it submits a segment job.
It appends the unseen suffix of that job's read range. Long silence can remain an explicit gap in the sample timeline.
Appends must be nonempty, aligned S16LE, ordered and disjoint. Each append must fit 30 seconds.
Recognition ranges must be nonempty, fully uploaded, retained and at most 30 seconds.
The service rejects a range that crosses an omitted gap. It does not synthesize silence.
Short nonempty language ranges retain the former 10-ms zero-padding behavior before log-Mel preparation.

The cache retains at most 32 MiB of PCM. A connection admits at most 64 unfinished recognition requests.
The client release watermark includes local retained audio and every unfinished job's read start.
This includes jobs that are still waiting for LID or may need overlap recovery.
The service prunes released prefixes. Admitted model jobs retain their own shared byte views until preparation completes.
Decoded appends own compact PCM allocations. Range snapshots share those allocations without retaining unrelated network-frame bytes.
The client command queue is bounded by its configured job capacity. Transport overload returns an explicit error.

Connection request IDs correlate service calls. They are separate from segment IDs and transcript revisions.
Cancelling a partial sends a cancellation command for its active service call.
Cancellation immediately returns the service admission slot. The completion loop discards that task's late result.
Dropping a connection or stopping the service drops its cache and pending decode futures.
An already-running blocking CPU preparation or submitted device operation can finish before its resources retire.
Graceful finish seals input, releases cached PCM, drains accepted jobs and returns final connection statistics.
The client completes this exchange before entering the editor's post-recording edit loop.

Each event's `transport` object is a cumulative snapshot, including accepted work from cancelled jobs.
It reports `uploaded_samples`, `requested_samples`, `peak_buffered_samples` and `buffered_samples`.
In JSON mode, stderr ends with `Audio transport: { ... }`, containing the acknowledged final statistics.
The acceptance runner checks these final statistics and records uploaded PCM bytes separately from requested audio.
PCM reuse removes repeated WAV encoding, upload and decode. Each recognition still computes log-Mel and model inference independently.
This transport does not provide model-native encoder or decoder state carry.

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

## Standalone client

The Rust client records the macOS microphone or converts an audio file through one FFmpeg implementation.
FFmpeg converts external codecs, sample rates and channel layouts to mono S16LE at 16 kHz.
The Rust adapter converts complete byte pairs to `i16` samples.
No Python interpreter participates in online audio processing or transcription state.

```text
100 ms at 48000 Hz, stereo = 4800 frames x 2 channels
  FFmpeg: decode + mix + resample
    stdout: 1600 mono S16LE samples = 3200 bytes
      FfmpegSource: retain incomplete reads; emit at most 512 i16 samples
        AudioSegmenter: own continuous PCM and sample positions
          AudioSegmentReq: immutable Arc<[i16]> + owned/read ranges
            bounded tasks: CPU LID -> concurrent Qwen service calls -> optional CPU alignment
          Event: revisable partial or immutable final
```

`FfmpegSource` owns the child, pipes, incomplete frame and bounded stderr diagnostics.
Enter requests shutdown, drains remaining audio and flushes transcription.
The client validates the FFmpeg exit before it flushes final inference. Capture failure cancels submitted jobs.
The client registers Ctrl-C, SIGTERM and SIGHUP handlers before native initialization. These signals cancel client work.
Dropping the source kills FFmpeg. Dropping an active service future cancels its HTTP request or gRPC stream.
Immediate server GPU cancellation is a separate contract.
Collected mode accepts 10 ms through 30 seconds and retains a failed request's recording for retry.
`--save` must name a new file. It also retains a complete continuous recording.
The writer buffers PCM writes. Finalization flushes the buffer and updates the WAV header before success.

### Native model setup

Build the Rust client and prepare assets once:

```sh
cargo build --release --bin asr
target/release/asr --setup-models
target/release/asr --stream
```

CMake and the Xcode command-line tools are required to build the macOS client.
Cargo downloads the checksum-pinned ONNX Runtime 1.31 static archive supplied by `ort-sys` during the first build.
The macOS archive also contains CoreML support. The VAD explicitly selects CPU execution.
The build script links Xcode's compiler runtime for the archive's Objective-C availability checks.
On Apple Silicon, the ASR deployment target defaults to macOS 13.4 to match the pinned archive.
An explicit `MACOSX_DEPLOYMENT_TARGET` must be 13.4 or later.
No ONNX Runtime shared library or runtime library path is required.
The client embeds the fixed Silero v6 model from [`assets/`](../crates/inference-asr-client/assets/README.md).
It loads the embedded bytes directly. It does not extract a model or runtime library to disk.

```text
build: pinned ORT static archive + Rust + Silero model bytes -> asr executable
run:   PCM[512] + retained context/state -> embedded Silero on CPU -> speech probability
       external Whisper directory      -> language detection / fixed-text alignment
```

`--setup-models` prepares only the pinned multilingual CT2 Whisper-tiny checkpoint.
It checks each file's byte count and SHA-256 before it replaces the destination.
It validates cached files before reuse and replaces corrupt entries from the pinned source.
A cancelled or failed preparation removes its temporary file.
`--whisper-source` copies the same pinned files from a local snapshot instead of downloading them.
The client does not download models during recording.
If the Whisper checkpoint cannot load, startup reports its path and the setup command before capture starts.
`--model-cache` selects the external Whisper asset directory.
`--lid-model` accepts a local CT2 checkpoint directory or a name under that directory.
The former `--vad-model` and `--onnx-runtime` flags and JSON settings are removed.
Silero explicitly selects CPU execution and retains recurrent state for one recording.
Initialization runs one silent frame and validates named probability and recurrent-state outputs before capture.
It discards warmup state. The recording starts with zero recurrent state.
ONNX output order does not define the output roles.

```text
startup: allocate input F32[1,576] + h/c F32[1,1,128] once
  embedded ONNX model -> speech_probs F32[1], hn/cn F32[1,1,128]
    validate names, types and shapes once -> discard warmup outputs

capture: previous input[512:576] -> input[0:64]
         PCM i16[0:n] / 32768   -> input[64:64+n], zero input[64+n:576]
  same input tensor + retained h/c tensors -> probability + hn/cn
    read by validated output names -> copy hn/cn into the retained h/c storage
```

Whisper log-Mel retains the FFT plan and scratch allocation.
Each Mel filter stores only its nonzero `(frequency_index, weight)` entries.
The projection reads those power bins in their original order. It does not multiply the zero coefficients.
The service's Qwen log-Mel preparation also stores nonzero filter entries and reuses FFT scratch across a WAV's frames.
That scratch belongs to the preparation call. The shared preprocessor remains immutable for concurrent requests.
Service FFT bin `k` uses frequency `sample_rate * k / n_fft` for both odd and even FFT lengths.

Whisper uses CPU int8 computation for LID and optional alignment.
Qwen3-ASR remains the only transcript generator.
An explicit `--language` bypasses LID. Positive overlap still needs Whisper alignment.
Recommendation: Leave `--language` unset for English/Chinese mixtures.
See [service commands](service.md#microphone-and-file-client).

### Continuous client

| Owner | Contract |
| --- | --- |
| `AudioSegmenter` | Continuous timeline, active range, pending finals and revision state |
| Submitted `AudioSegmentReq` | Immutable audio and stable coordinates after old buffers are pruned |
| `AudioTranscriber` | Shared acoustic model, bounded LID cache, concurrent service calls and last completed result |
| `AudioStream` | VAD/segmenter lifetime, bounded task admission, completion and committed overlap history |
| Consumer | Replace the complete partial; commit each final once |

Initialization converts validated duration settings to sample counts once.
The segmenter and language analyzer retain only their own sample-domain settings.
They do not clone the capture, transport, or model-path configuration.

```text
session receives x[0:16384], mono at 16000 Hz: 1.024 seconds
  submit immutable partial x[0:16384]
  capture keeps accepting x[16384:...]
    worker returns text -> partial(segment=0, revision=1)

pause / stable sentence + short pause / duration limit / stop
  finalize owned x[start:end]
    worker result -> final(segment=0, revision=2)
    next segment starts at or after end

retained audio:
  active range + pending final ranges + configured overlap/padding
  submitted jobs own Arc allocations independently of session pruning
```

Sample positions start at recording start and include silence.
They do not reset at segment boundaries or after pruning.
Right context and endpoint padding clamp to captured audio, including near the `u64` timeline limit.
Events expose `start_sample`, `end_sample`, `sample_rate`, `start_ms` and `end_ms`.
Millisecond values derive from the integer sample coordinates.
A partial can revise every word in its complete draft.
A final commits once, in segment order.
A closed segment rejects late partials. A draft cannot supply sentence evidence to a newer segment.
The terminal replaces one visible draft and shows its latest suffix when width is insufficient.
JSON Lines and final output retain complete text.

The optional editor keeps a separate document after recognition commits.
[`document.rs`](../crates/inference-asr-client/src/document.rs) owns finalized text, the latest draft, and explicit correction revisions.
[`editor.rs`](../crates/inference-asr-client/src/editor.rs) owns the terminal view, event log, and saved document.
It preserves original recognition events, including spans and metrics.
A correction replaces one finalized segment's text. It does not change its audio range.
The correction must name the current revision and allocate its checked successor.
Closed segments still reject partials and duplicate finals.
Manual corrections do not alter ASR context, worker history, or alignment metadata.
Live updates and replay share one document commit path.
Replay validates file input first. Prepared manual corrections do not repeat that validation after logging.
See [editor commands and replay](service.md#revisable-transcript-experiment).

```text
AudioSegmenter -> raw final[segment=7, revision=3] -> event log, unchanged
                          |
                          +-> Document[7]: text + source sample range
                                confirmed replacement
                                  -> correction[base=3, revision=4] -> event log
                                  -> Document[7].text -> terminal redraw / transcript.txt

replay: saved events -> validate coordinates, commit order, correction base -> same Document
```

The partial interval defaults to one second of new audio.
The ordinary VAD pause defaults to 0.5 seconds.
Two matching sentence-ending drafts can close a segment after 0.25 seconds.
The latest matching snapshot must cover the most recent speech.
`--silence-seconds` and `--sentence-silence-seconds` configure these heuristics.
These thresholds are policy choices, not calibrated model requirements.

The maximum owned range defaults to 24 seconds.
A duration cut searches the preceding three seconds for at least 80 ms of sustained quiet.
It retains the complete suffix. A forced cut can split a word without a suitable pause.
`--window-seconds` and `--lookback-seconds` configure this policy.

LID uses 0.5-second cells with up to 0.5 seconds of context on each side.
The observation cache is bounded to 256 windows by default.
Candidate language changes need nearby sustained quiet before they become ASR cuts.
Compatible ranges merge. Uncertain ranges use automatic Qwen decoding.
Confident Mandarin observations can trigger a Chinese-hint retry when automatic ASR returns no Han characters.
This guard does not detect every partial translation or recognition error.
`--context` supplies caller vocabulary to every internal request.

The event loop submits available work before it waits for audio, task completion, Enter or an input deadline.
It does not poll inference results on a timer.
`--max-in-flight` defaults to four jobs. Its supported range is 1 through 64.
Running jobs and completed finals waiting for ordered commit share this capacity.
Each job receives its revision at submission. Emitted revisions can have gaps when obsolete drafts are discarded.
Only an active segment can accept partial results. An accepted newer partial supersedes older revisions.
Closing a segment cancels its remaining partial jobs. Obsolete partial errors do not terminate the session.
Finals commit exactly once, in source segment order, even when service calls complete out of order.

```text
audio + completion + Enter + deadline -> AudioSession event loop -> AudioStream owns AudioSegmenter
  submit final 0 r3 ------+----------------------------> completes -> commit 0
  submit final 1 r2 ------|----> completes -> retain slot ----------> commit 1
                         |
  shared CPU model: acquire owned guard -> bounded blocking call -> release
  HTTP/gRPC: no CPU-model guard held while awaiting the response
  overlap: recognize/align ahead -> wait for preceding commit -> reconcile
```

The client shares one Whisper model and one bounded observation cache across jobs.
It acquires the CPU-model guard asynchronously before it submits work to the runtime core `SyncTaskPool`.
The fixed pool has one runner. Whisper retains its existing internal thread setting.
A canceled queued task is skipped. A running native call finishes without publishing its canceled result.
Cancellation cannot accumulate blocking calls behind that guard.
Tasks share one HTTP WebSocket or gRPC audio connection. Each segment request retains its own metrics.
Only ordered final commits publish overlap words to later jobs.

Final work has priority. Unsubmitted partials coalesce into the latest snapshot.
The live final queue capacity is `max_pending_finals + ceil(overlap_seconds / sentence_silence_seconds)`.
Live overload returns an explicit error before audio is silently lost.
At EOF, the already-buffered active range can become one additional final. Input is then closed.
Graceful flush waits for pending inference results.
Jobs reuse the last completed Qwen result only for the same recording sample range and language.
This does not reuse encoder computation or decoder KV.
Events report wait/processing durations, request count, submitted audio, LID work, cache hits and alignment calls.
These metrics describe emitted jobs. They exclude discarded or cancelled jobs.

### Optional ASR overlap

`--overlap-seconds` defaults to zero.
A positive value supplies bounded left and right context around disjoint owned ranges.
The owned window plus both contexts must fit 30 seconds.

```text
owned: A x[0:128000]             B x[128000:256000]
reads: A x[0:144000]             B x[112000:272000]
                  shared source audio
  Qwen text -> fixed Whisper tokens -> native attention DTW
    same text + matching interval -> shared prefix
    same text + distinct interval -> real repetition
    uncertain boundary            -> decode owned audio
```

Pending finals retain source ranges until right context arrives.
Stop flushes available context.
Whisper aligns fixed Qwen text without generating a transcript.
The attention grid is 20 ms. It does not establish exact word boundaries.
English lexical units are words. Chinese lexical units are characters.
Zero-duration times remain unresolved.
Stitching requires normalized text agreement and reciprocal midpoint containment.
Missing coverage, collapsed repetitions and uncertain boundaries select owned-audio recovery.
A boundary word confined to one 20-ms grid cell cannot establish ownership. This rule includes a zero-duration word.
Interval matching uses overflow-free midpoints and relative boundary distances on the source sample clock.
Unexpected backend failures propagate.
Only committed words within the overlap horizon remain in history.
JSON spans retain raw text, read bounds, selected transcription bounds, words and alignment status.
LID ranges remain in `language_ranges`. Overlap mode does not turn them into ASR cuts.
This mode uses no text LLM.
Recommendation: Keep the zero default until natural code-switch audio establishes a quality benefit.

### Verification

Run the core client contracts:

```sh
cargo test -p inference-asr-client
```

These tests cover audio ownership, revisions, endpointing, suffix retention, overlap ambiguity and VAD recurrence.
The ignored native migration-oracle test accepts reference JSON through `KHALAMARI_ASR_ORACLE`.
It compares log-Mel values, recurrent VAD outputs, language labels and fixed-text alignment coverage.
Native LID scores and attention times can differ between builds. They are not numerical parity guarantees.

After starting the release Qwen3-ASR service, run actual model acceptance:

```sh
cargo run --release -p inference-asr-client --example asr_e2e -- \
  --binary target/release/asr --binary-commit "$(git rev-parse HEAD)" \
  --model-dir /absolute/path/to/Qwen3-ASR-1.7B-8bit --official
```

The runner checks synthetic English, Chinese, mixtures and duration cuts.
`--official` adds concatenated public human speech in both language orders.
These cases do not establish spontaneous human code-switch accuracy.
A JSON `--manifest` can supply recordings, required fragments, repetition counts and endpoint expectations.
Relative recording and configuration paths resolve against the manifest directory.
`--prepare-only` writes the recordings and manifest without inference.
The runner copies one release binary into its output directory and uses that copy for every case.
Reports retain its hash, audio hashes, settings, evaluator source hash/Git state, collected baselines and raw streaming events.
`--binary-commit` records the caller's declared binary source commit. The evaluator does not infer this commit from its own checkout.
Protocol checks remain separate from transcription-quality checks.
Reports also retain the capture-ready time and each event's receipt time on one monotonic clock.
An optional manifest `speech` field supplies ordered, disjoint `[start_sample, end_sample]` reference intervals at 16 kHz.
The evaluator merges reference intervals across pauses shorter than 0.5 seconds.
It reports `final_latencies` from each reference turn's end to the matching final event.
These estimates include file replay pacing, FFmpeg buffering and startup after the capture-ready marker, and transport scheduling.
They exclude native model loading before that marker. They are not audio-device timestamp measurements.

Generate paired overlap cases:

```sh
cargo run --release -p inference-asr-client --example asr_e2e -- \
  --binary target/release/asr --paired-overlap --output target/asr-overlap-eval \
  --model-dir /absolute/path/to/Qwen3-ASR-1.7B-8bit
```

Homophones such as `cache` and `cash` can fail strict references.
Use explicit domain context when it is part of the intended workload.
Do not weaken a reference silently or count a protocol pass as a quality pass.

### VAD replacement gate

Silero remains the production VAD. Earshot 1.2.2 did not pass the paired replacement screen on 2026-10-20.
The screen froze the clean `d3b5a065` client and used the same Qwen checkpoint and service binary for both clients.
The candidate stayed outside production `src` and the workspace dependency graph.
The host was an Apple M3 Max with 48 GiB memory and macOS 27.0.1.

```text
same x[0:N], mono i16 at 16000 Hz
  Silero:  x[k:k+512] -> ORT + recurrent state -> score, threshold 0.5
  Earshot: x[k:k+256] -> embedded Rust model  -> score, threshold 0.5 or 0.55
    original samples + speech flags -> same AudioSegmenter, 200-ms padding
      manual speech ranges -> retained-speech and turn-tail coverage
      final scheduling sample -> endpoint latency
      Qwen HTTP result -> final receipt time on the replay clock
```

The fixed batch contained 30 manually annotated [TEN recordings](https://github.com/TEN-framework/ten-vad/tree/5675e379469984f452719f87a6a768398e9dbd85/testset),
six speech derivatives at -20 dB gain, six speech-free controls, and 11 existing English/Chinese recordings.
Odd-numbered TEN recordings and their derivatives supplied calibration. Even-numbered recordings supplied validation.
Each split also contained silence, Gaussian noise and a stationary tone. Derivatives are not independent recordings.

The validation split contained 21 inputs and 118.682 seconds of annotated speech.
Retained loss counts annotated speech samples outside the union of final owned ranges.
Turn-tail loss counts uncovered samples in each reference turn's last 100 ms.
Endpoint latency uses the source sample clock. It excludes LID, ASR, HTTP and output waiting.

| Validation metric | Silero: threshold 0.5, pause 0.5 s | Earshot: threshold 0.5, pause 0.5 s | Earshot: threshold 0.55, pause 0.45 s |
| --- | ---: | ---: | ---: |
| Raw speech false-negative rate | 12.12% | 9.12% | 12.56% |
| Retained speech loss, total | 1.965 s | 3.127 s | 4.043 s |
| Turn-tail loss, total | 0.275 s | 0.227 s | 0.243 s |
| Turns with tail loss | 3 | 3 | 3 |
| Endpoint latency, P50 | 0.515 s | 0.581 s | 0.515 s |
| Endpoint latency, P95 | 1.914 s | 5.525 s | 5.221 s |

Calibration tested 18 score thresholds and six pause durations. No combination passed all coverage and endpoint gates.
The `0.55` / `0.45 s` candidate missed only the calibration P95 gate and was retained for diagnostic ASR replay.
Lower raw false-negative rates did not establish better audio retention or final latency.
Some concentrated misses exceeded the padding horizon. Some false speech flags merged pauses into later turns.
Both clients passed the 13 existing English/Chinese streaming protocol and text-fragment cases.
Mixed-unit reference edit counts were unchanged, including one existing edit in the long mixed case.
This acceptance did not override the failed coverage and endpoint gates.

Six labelled diagnostic clips supplied 15 reference turns for two final-text latency replays per client, in A/B/B/A order.
These clips included delayed cases from the earlier screen. They are not an independent latency sample.
Earshot added an estimated 0.853 seconds at paired P50 and 8.138 seconds at paired P95.
The measurements used event receipt times and the capture-ready marker. Replay pacing and buffering affect their absolute values.
The separate sample-clock endpoint measurements do not have that wall-clock ambiguity.

Local evidence is in `target/asr-earshot/report.json`, with raw scores, source snapshots, recording hashes and replay commands.
Recommendation: Preserve the current VAD until a lighter candidate passes the same coverage, latency and transcription gates.
