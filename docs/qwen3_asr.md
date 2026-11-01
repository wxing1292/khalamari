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
| Duration per ASR request | At most 30 seconds after resampling |
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
| `audio.rs` | Independent capture task, FFmpeg child and pipes, canonical PCM and quiet-boundary search |
| `recording.rs` | Single append writer, positional readers, archive encoding and atomic publication |
| `analysis.rs` | Shared append/replace/seal contracts, label timelines, bounded CPU processors and reader tasks |
| `vad.rs` | Speech-detector contract and stateful Silero implementation |
| `diarization.rs`, `diarization/process.rs` | Diarizer contract, external process lifetime and validated activity updates |
| `assets.rs` | Pinned external checkpoints, byte and SHA-256 validation, atomic publication |
| `config.rs` | One Serde/Clap schema, defaults, explicit CLI precedence and external validation |
| `session.rs` | `AudioSession`: capture, processing, output and recording lifetime |
| `segment.rs` | `AudioSegmenter`: analysis cursor, immutable `AudioSegmentTranscribeReq` snapshots, endpoints and revisions |
| `models.rs` | CPU Silero state, native CT2 language detection/alignment, acoustic word construction and log-Mel |
| `language.rs` | `Acoustics`: shared Whisper weights, LID lifecycle/cache, fixed-text alignment and language-range planning |
| `transcribe.rs` | Qwen routing, last-result reuse, Mandarin retry and overlap reconciliation |
| `transcript.rs` | Transcript, span, word and turn records; speaker attribution and turn projection |
| `stream.rs` | `AudioStream`: independent analysis readers, bounded tasks, final ordering, speaker revisions and overlap history |
| `overlap.rs` | Timestamp ownership and deterministic reconciliation |
| `service.rs`, `service/stream.rs` | Transcription response, duplex PCM transport, bounded commands, request cancellation and connection statistics |
| `output.rs` | Exclusive JSON/text/editor selection, output errors and the shared edit loop |
| `editor.rs`, `document.rs` | Raw event log, speaker-attributed text projection, explicit corrections and final-text snapshot |
| `terminal.rs` | Independent input descriptor, UTF-8 command input and shared display formatting |
| `main.rs` | Startup commands, replay and process cancellation |

The native `asr_e2e` example replays the production executable through the service. The `asr_speaker_score` example scores file-local speaker intervals against RTTM.

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

FFmpeg -> one writer appends x[0:512], x[512:1024], ... to the recording
  archive reader -> FLAC
  live readers -> independent VAD, LID and diarization timelines -> segment ranges
    immutable job + language evidence -> Qwen service -> aligned text and speaker intervals
      ordered events -> JSONL, terminal or revisable editor
  sealed recording -> same offline pipeline, optional live boundary hints -> independent final events

HTTP/gRPC request -> service WAV validation -> log-Mel[128, frames]
  prepared source -> runtime cache miss -> Audio Tower -> resource embeddings
    decoder + embeddings -> transcription text -> client response
```

`AudioSession` owns capture, terminal input, archive publication, and output.
Its event loop handles input, capture completion and analysis events.
The same control loop stays active during model loading, reader readiness and connection drain.
Enter stops capture while analysis initializes; initialization does not restart after a terminal event.
`AudioStream` owns one pass, connection, bounded jobs and ordered final commits.
`AudioSegmenter` owns segmentation progress and revisions. The recording writer owns the source clock.
`AudioSegmentTranscribeReq` owns its immutable audio allocation. Its read end derives from the allocation length.
Its final status derives from its endpoint reason.
Service-call metrics belong to the segment request. Streaming requests share one duplex connection for the recording.
Submitted audio counts remain integer samples until JSON serialization.

| Boundary | Contract |
| --- | --- |
| CLI/JSON configuration | Validate schema, values, paths and relationships once |
| FFmpeg pipe | Validate complete i16 samples, retain incomplete frames across cancellation |
| Submitted job | Keep immutable audio and stable source coordinates after segmentation advances |
| Service/model output | Validate bounded responses and native output contracts at their owners |
| Live document update | Apply trusted session output without repeating replay validation |
| Replay or manual correction | Validate external coordinates, commit order or current revision before trusted commit |
| Output failure | Return the I/O error. Only invalid editor commands remain recoverable in the view |

Normal EOF drains inference. The editor can then use the same edit loop as event replay.
An empty terminal command requests stop, drains pending inference, saves output, and exits.
Cancellation drops ASR tasks, stops capture, and finishes archive publication. The editor attempts to save accepted final text and restores its terminal.

The raw event and document line have different lifetimes.
The log retains acoustic spans and metrics. The document retains only text, revision, and source coordinates.
Both terminal views use `Document` for the current text and speaker turns.
The plain view tracks whether a draft row is visible. It renders the current draft from `Document` after each accepted update.
It starts a fresh document for each analysis pass. It does not cache a second formatted copy of the draft.
Both views suppress speaker updates that do not change the document. The raw JSONL log retains these updates.
Status messages clear the visible draft row before writing and then restore it from the document.
The complete draft render fits one physical row, including metadata and wide Unicode characters.
If metadata leaves no room for useful text, the draft shows the transcript suffix without the metadata prefix.
Resize uses the current terminal width. Final lines and redirected output retain complete text.
The archive task publishes audio independently. The session reports its final path through the output owner.

```text
immutable recognition event -> Document owns final lines + current draft
                                  |
status / older final / resize -----+--> render current draft at current width
                                       20 columns: at most 19 display cells
                                       80 columns: at most 79 display cells
                                  |
                                  +--> final lines: complete text
raw event ----------------------------> JSONL: complete acoustic evidence
archive task -------------------------> file publication; session reports completion
```

Whisper and Qwen retain separate log-Mel implementations because their preprocessing contracts differ.
Whisper appends 160 zero samples and uses its own frame/padding convention and window precision.
Qwen derives its feature size and frame limits from the checkpoint and keeps preparation scratch local to each RPC.
Whisper compacts a truncated feature matrix in place after full normalization.
The service normalizes decoded input before resampling. It repeats normalization only when resampling can introduce overshoot.

### Task contracts

| Contract | Scope |
| --- | --- |
| `AudioSpeechDetector` | start / append / detect / stop on continuous recording input |
| `AudioLanguageRecognizer` | start / append / recognize / stop; unknown language remains valid |
| `AudioDiarizer` | start / append / diarize / stop; backend owns buffering and speaker state |
| `AudioAnalysisUpdate<T>` | Replace labels in an explicit range, or seal the next prefix |
| `AudioSegmentTranscribeReq / AudioSegmentTranscribeResp` | One immutable segment revision to text, alignment and metrics |
| `AudioTranscribeReq / AudioTranscribeResp` | Collected WAV request to Qwen text and language |
| `AudioStreamReq / AudioStreamResp` | Incremental PCM transport, range recognition and release |

Every analyzer accepts each sample exactly once. Its append range must start at its previous accepted end.
The analyzer can read overlap from the log. Input notifications do not copy PCM.
Stop closes input. The next analysis call must flush and seal all accepted input.
A call can return no updates. An empty Replace removes existing labels in its range.
Seal freezes this analyzer's labels for this pass. It does not delete PCM or speaker memory.
Each component has its own seal. The offline pass starts independent timelines.

`AudioProcessor` owns one serial CPU slot through the existing runtime `SyncTaskPool`.
It acquires the model guard before submission. Cancellation cannot create an unbounded native-work queue.
`AudioReaderTask` owns its progress cursor and one queued output batch.
It consumes coalesced watch notifications without losing samples. It checks the final extent after writer closure.
The I/O batch limit does not select model inference windows.
`Acoustics` owns Whisper weights, LID progress and the observation cache.
It implements the LID lifecycle directly. LID and fixed-text alignment share one model and CPU slot.
LID advances on the configured sample grid, including when context is zero.
Only stopped input permits a short final cell. Notification granularity does not select analysis ranges.
VAD and diarization have separate slots. The session event loop executes no model inference.

`AudioStream` owns admitted asynchronous Qwen tasks, cancellation, final ordering and overlap history.
`AudioSegmenter` owns active/pending ranges, endpoint decisions and request revisions.
A revision is allocated at submission. An old partial cannot replace an accepted newer revision or final.

```text
PCM log R[0:N), 16000 Hz, mono i16
  VAD reader        -> speech spans, sealed prefix V
  LID reader        -> language/unknown spans, sealed prefix L
  diarizer reader   -> speaker spans, independently revisable prefix D
          |                    |                        |
          +---------- AudioStream timelines ------------+
                                |
                 AudioSegmenter selects [start,end)
                                |
       AudioSegmentTranscribeReq(segment,revision,immutable PCM,language evidence)
                                |
                  Qwen RPC -> fixed-text alignment
                                |
              ordered final or current partial
                                |
       latest speaker timeline -> turns -> JSONL / terminal / Document

speaker Replace after text final -> speakers event for the same text revision
                                    -> update turns, preserve manual edits
```

Sample fields use `start` / `end` in Rust. Existing transcript JSON keeps `start_sample` / `end_sample`.
Task metrics count Qwen calls, submitted audio, result reuse and alignment.
Continuous LID work is shared across tasks and is not counted again per task.

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
The writer appends complete byte pairs directly. Model readers convert only their requested ranges to `i16`.
No Python interpreter participates in online audio processing or transcription state.

```text
48000-Hz stereo input -> FFmpeg -> mono i16 PCM at 16000 Hz
  AudioCapture owns the only AudioRecordingWriter
    append R[0:N) to a temporary PCM file, including silence
    publish N after the complete write
      +-> archive reader -> FFmpeg FLAC -> atomic publication
      +-> VAD reader -> speech spans
      +-> LID reader -> language spans, including unknown
      +-> speaker reader -> revisable speaker spans
              timelines -> segmenter -> immutable ASR range snapshots
                Qwen service / shared CPU alignment -> live JSONL and terminal
capture EOF -> sync and seal R -> archive completes independently of ASR
            -> full-recording analysis with new state -> offline JSONL
```

`FfmpegSource` owns the child, pipes, incomplete frame and bounded stderr diagnostics.
`AudioRecordingWriter` owns append order and the published sample extent. Readers use positional reads and independent notification cursors.
The operating-system page cache supplies shared recent data. Each admitted model job owns a bounded contiguous snapshot.
The PCM spool and compressed output coexist during capture. Analysis components do not retain separate complete audio copies.
A slow analysis reader stops advancing when its bounded queue is full. Capture continues writing the same spool.
An inference error stops that analysis pass without stopping capture.
Archive publication runs independently. A slow or failed archive does not delay live finals or the offline pass.
Publication follows a complete write. It does not imply `fsync` per frame or power-loss durability.
Capture completion synchronizes and seals the PCM file. A failed archive keeps the spool and reports its path.
Successful archive publication, or explicit discard, permits spool deletion after analysis readers finish.

Enter stops capture and drains analysis. Ctrl-C, SIGTERM, and SIGHUP cancel analysis, then finish capture and archive publication.
Default files use the capture-start timestamp in the starting directory: `.flac` and one `.<pass>.jsonl` per analysis pass.
`--save` selects a new WAV or FLAC destination. `--discard-recording` suppresses the archive and deletes the temporary recording.
Recording has no default duration limit. `--duration` limits microphone or file capture in either mode.
Each ASR request still fits the service's 30-second limit.
The second pass uses the complete recording, bounded larger-context requests, and no partials.
`--no-reprocess` disables it. `--mode offline --input PATH` runs this pass without real-time pacing.
The two passes use the same sample clock but have independent segment and speaker IDs.
Live pause and speaker cuts are candidates in the offline window lookback. The offline pass uses the same bounded LID observation cache.
ASR regenerates text from the recording. Original live events and manual edits remain separate.
`--mode offline`, including the default invocation, waits for capture to finish and runs the same offline pipeline.
It uses VAD, LID, language-aware requests, Mandarin recovery, word alignment, and speaker attribution.
All modes emit the same JSONL event schema. Offline output contains final and speaker updates, without partials.
The `.offline.jsonl` file contains the final result. Speaker updates replace attribution for the referenced segment revision.
The client no longer sends a separate unary whole-recording request or retains a separate collected-output path.

```text
source samples [0, N) -> one writer -> PCM log + compressed archive
  --mode stream: readers advance during capture -> live events + boundary hints
  capture ends: seal [0, N)
    default / --mode offline / live reprocess
      -> fresh offline readers over [0, N)
      -> bounded language-aware ASR ranges -> aligned text + speaker turns
      -> final / speakers events in .offline.jsonl
```

`--no-reprocess` suppresses only the second pass after live analysis. It does not disable offline-only analysis.

### Speaker reader protocol

`--speaker-command PATH` (`models.speaker_command`) selects the reader for both passes.
The default path is `target/asr-readers/community-1/reader` under the repository root.
The client starts one process per pass with `--recording PATH --pass live|offline`.
The reader opens the canonical PCM log read-only. It must read only announced sample ranges.
Standard input and output carry newline-delimited JSON. Standard error carries diagnostics.
Before accepting input, the reader must send `{"protocol":1}` on success or `{"error":"reason"}` on initialization failure.
An initialization failure terminates the reader. The client reports its reason without treating it as a truncated activity response.

```text
reader -> client: {"protocol":1}
client -> reader: {"start":0,"end":16000,"stopped":false}
reader -> client: []                         (insufficient evidence)
client -> reader: {"start":16000,"end":32000,"stopped":false}
reader -> client: [{"type":"replace","range":{"start":0,"end":32000},"spans":[
                     {"start_sample":8000,"end_sample":24000,"speaker":0}]}]
client -> reader: {"start":32000,"end":32000,"stopped":true}
reader -> client: [{"type":"seal","range":{"start":0,"end":32000}}]
```

Each request must receive one response. The reader must flush each response.
Replacements must stay inside accepted audio, cover a continuous timeline and preserve sealed prefixes.
Spans can overlap when speakers overlap. An empty replacement is an authoritative empty result.
Seals must advance a continuous covered prefix. The stop response must seal all accepted input.
The process adapter validates these external contracts before passing updates to the internal timeline.
Cancellation terminates the child and reaps it outside the session event loop.
The archive remains independent of process success.

The [Community-1 reader](../crates/inference-asr-client/examples/readers/community_1.rs) is an experimental backend.
Build it in the pinned [speakrs checkout](https://github.com/avencera/speakrs/tree/5a8f5804c189defe597e8aebb6a82bbfedde9f48).
It adds no dependency to the main Cargo workspace.

Set `SPEAKRS` to that checkout directory. Place its prepared native CPU checkpoint beside the installed reader in `models/`.
The checkpoint must contain segmentation, WeSpeaker embedding and PLDA files, as documented by speakrs.

```sh
cp crates/inference-asr-client/examples/readers/community_1.rs "$SPEAKRS/examples/khalamari_reader.rs"
cp crates/inference-asr-client/examples/readers/community_1_test.rs "$SPEAKRS/examples/community_1_test.rs"
cargo build --release --manifest-path "$SPEAKRS/Cargo.toml" --features cpu --example khalamari_reader
mkdir -p target/asr-readers/community-1/models
cp "$SPEAKRS/target/release/examples/khalamari_reader" target/asr-readers/community-1/reader
cp -R /absolute/path/to/community-1/models/. target/asr-readers/community-1/models/
target/release/asr --mode stream
```

The backend reads native CPU weights and PLDA files from `models/` beside its executable, independent of the working directory.
`--speaker-model-dir PATH` (`models.speaker_model_dir`) passes an explicit `--model-dir PATH` to the reader.
An invalid override fails instead of loading another checkpoint. No runtime environment variable selects the checkpoint.
Client setup does not download these files or build the reader.
The previous offline-only RTTM process contract and native diarizer are removed.

### Speaker attribution

The Community-1 experiment uses overlapping bounded windows for live processing and the complete recording offline.
The candidate live settings are a 30-second window, a 2-second update interval and a 10-second revision tail.
Shared-audio overlap establishes candidate speaker correspondence. Any positive overlap can preserve a short activity without an embedding.
One-to-one assignment keeps different local voices distinct. Unmatched voices use the configured cosine threshold.
Each identity retains its first valid embedding and its latest valid embedding. Matching uses the better cosine score.
The first reference remains fixed. Repeated overlapping windows do not average it into a different voice.
An unmatched cluster that exists only before the seal does not enroll a new identity.
Only clusters that reach the mutable output range refresh their references.
The offline pass can revise the complete recording and uses independent speaker IDs.
These policies belong to the backend. They do not appear in `AudioDiarizer`.

```text
single writer -> PCM log [0, N) -> Community-1 reader owns model and identity state
  local model activity + embeddings for [window_start, N)
    -> shared sample intervals -> one-to-one existing IDs
    -> unmatched: compare fixed and recent references -> existing ID or new ID
       sealed-only unmatched activity -> no enrollment
    -> Replace [sealed, N) -> Seal next prefix -> ASR attribution

each ID: first valid reference + latest valid reference
PCM, labels and identity references have separate owners and lifetimes
```

The reader uses a fixed 30-second window, 2-second update interval, 10-second revision tail and 0.7 similarity threshold.
The former experimental environment controls are removed. Alternate historical policies below are measurements from retained experiment binaries.
The reader processes coalesced progress in order. It does not skip unprocessed audio when the window advances.
These experiment controls do not establish a quality or latency guarantee.

Each transcript span includes speaker activity and approximate word times.
`transcript.rs` owns these records and their attribution rules. `transcribe.rs` owns inference execution.
Rust `Span.speaker_activity` retains raw activity. Its JSON key remains `spans[].speakers`.
Word and turn `speakers` fields contain attribution IDs.
A timed word uses the speaker with the most covered samples across its complete interval.
Coverage uses the union of each speaker's activity intervals. Overlapping rows for one speaker do not count twice.
Equal coverage retains multiple IDs. Raw activity retains every observed ID, including brief secondary activity.
This separates activity detection from text attribution, as in [WhisperX](https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py).
Zero-duration words retain unresolved word attribution and unchanged times.

Turn projection uses the common speaker IDs on both sides of unaligned text.
At a span edge, it uses the available neighboring attribution.
Conflicting activity inside the unaligned range prevents this inference. Different neighboring speakers remain distinct.
If no words have timing, one observed ID can support span-level attribution; multiple IDs remain unresolved.
Adjacent turns with the same IDs merge. Contiguous internal language spans do not create an extra turn for the same voice.
A single resulting turn uses its transcript span's range. Its times have span precision, not exact speech-boundary precision.
These rules do not change word times or claim recovery of every overlapping voice.
The terminal, editor and JSONL show `[start, end)` speaker text turns.

A `speakers` event revises attribution for an existing final text revision.
It does not repeat transcription or allocate a new text revision.
Raw activity boundaries can change without changing the displayed turns. JSONL retains those changes; terminal views do not repeat the same document.

```text
word "前" [18.308, 18.488): 2880 samples
  speaker_0 covers [18.308, 18.488): 2880 samples
  speaker_1 covers [18.374125, 18.488): 1822 samples
                      |
                      v
  text attribution: speaker_0; raw activity still contains both IDs
  unaligned text:   speaker_0 ... 蒙(?) ... speaker_0
                      |
                      v
  turn projection: speaker_0 ... 蒙 ...; original word times stay unresolved
  Document: changed turns -> terminal/editor; unchanged turns -> no redraw
```
The editor applies it only to matching text and revision. A manual correction retains precedence.
A replacement that leaves the clipped speaker labels unchanged emits no transcript update.
An empty replacement still clears prior attribution. Label timelines stay ordered by start sample.
Scheduling and attribution stop scanning at the requested end, without traversing future labels.
Overlapping spans of the same speaker represent one active voice.
Only finals whose speaker ranges remain unsealed need retained attribution state.
Speaker revisions do not extend the lifetime of language observations; each timeline retains only its own consumers' ranges.
VAD-negative audio stays in the recording. Positive speaker evidence can preserve speech for segmentation.
Late speaker revisions update attribution. They do not re-run completed ASR ranges or backfill old VAD-negative ranges.
The offline pass can choose new ranges from the complete recording.

### Community-1 streaming experiment

This is a matched comparison of the same native CPU Community-1 pipeline with different context policies.
It does not compare different ASR models or claim equivalence to every pyannote configuration.

```text
same 16-kHz mono PCM, same model files, one process at a time
  progress every 1 s -> reader inference -> Replace / Seal
    first labels and revised labels -> 10-ms reference grid -> DER
    measured call durations + source arrival times -> simulated FIFO latency
```

The three far-field inputs are AMI ES2004a, IS1009a and TS3003a, each restricted to `[50,150)` seconds.
DER includes overlap, uses no collar, and permits one file-global speaker permutation.
The aggregate weights errors by reference speaker time. Lower is better.
The latency column gives the range of per-recording P95 values. It is simulated source-time-to-first-label latency,
not measured microphone-to-text latency. Offline waits for EOF.

```text
Policy                  First DER  Revised DER  CPU work / 300 s  Label P95
----------------------  ---------  -----------  ----------------  -----------
Full recording offline     14.12%       14.12%             12.11 s  EOF only
Growing prefix, every 2 s  25.64%       14.12%            290.16 s  24.20-24.76 s
30 s window, 10 s tail     30.80%       18.84%            122.82 s   2.84-2.85 s
30 s window, 20 s tail     30.80%       19.11%            122.21 s   2.84-2.85 s
10 s window, 5 s tail      43.34%       41.05%             16.77 s   2.01-2.02 s
```

The 30-second/10-second candidate uses about 327 MiB peak process RSS.
Its simulated seal P95 is 12.84-12.85 seconds. Provisional labels and text do not wait for that seal.
In a separate 30-second two-speaker recording, its first/revised DER is 26.04%/8.91%; offline DER is 5.71%.
The initial four-speaker IS1009a excerpt exposes identity fragmentation: the live reader produces six IDs and 24.82% revised DER;
the offline pass produces four IDs and 11.04% DER.
Keep 30/2/10 as the experiment default. Longer revision time did not improve aggregate accuracy.
Full-prefix recomputation accumulates work. Short windows lose speaker context.
The backend remains experimental; the offline pass remains useful.

Evidence was recorded on Apple M3 Max, 48 GiB, macOS 27.2, with no ASR service running.
The base commit was `3d21fa70c4aa6eff40c7934675d800abf2f15529` with the reader-contract rewrite dirty.
The reference checkout was `5a8f5804c189defe597e8aebb6a82bbfedde9f48`.
The measured reader SHA-256 was `23bac6f6171d9736efb12514681831cf75e56f862bcf74463bcd0895f1b25399`.
Raw commands, PCM hashes, dirty-tree hashes, updates and scores are in
`target/asr-community-stream-eval/report.json` and `target/asr-community-hold-eval/report.json`.
The initial reader removes per-call diagnostics and handles coalesced progress without changing the tested one-second inference windows.
Its SHA-256 is `0f7718a10554e61625dc34471675a9bdecf7b3b66296b92e91382386506e1892`.
A paced 30-second replay measured first-label P95 at 2.54 seconds, seal P95 at 12.54 seconds, and EOF drain at 1.04 seconds.
It produced the same first/revised DER as the unpaced two-speaker case.
A single coalesced notification for all 100 seconds of ES2004a preserved the incremental run's final DER of 18.30%.
These checks are recorded in `target/asr-community-paced-eval/report.json` and `target/asr-community-coalesced-eval/report.json`.

#### Identity-state audit

The matched audit uses the same four recordings, CPU checkpoints, 30/2/10 policy, cosine threshold and scoring rules.
The reader now accepts short shared-audio correspondence, retains fixed/recent references, and avoids enrolling sealed-only activity.
The model and its offline clustering are unchanged.

```text
Three far-field recordings, 300 s total       Before       Current      Offline
------------------------------------------  -----------  -----------  -----------
First-label DER                              30.80%       29.54%       EOF only
Revised DER                                  18.84%       18.79%       14.12%
IS1009a hypothesis IDs, four reference voices  6            5            4
CPU work                                     122.37 s     122.11 s     12.21 s
Simulated first-label P95                     2.84-2.85 s  2.84 s       EOF only
```

IS1009a first/revised DER changes from 41.70%/24.82% to 38.19%/24.67%.
Both other far-field recordings and the separate two-speaker recording retain their previous DER.
The CPU work difference is small. This measurement does not establish a throughput improvement.
Peak RSS remains about 329-335 MiB. Each identity retains one additional embedding vector.
Five live IDs for four reference voices still show fragmentation. The current matcher does not solve global clustering.
ES2004a and TS3003a retain zero identity-confusion time. Most remaining errors there are missed speech, not fragmented IDs.
Dropping unmatched activities with no embedding regressed the two-speaker recording and was rejected.
Global overlap assignment did not improve this comparison, so the reader retains the simpler matching procedure.
The separate paced 30-second replay measures first-label P95 at 2.54 seconds, seal P95 at 12.54 seconds, and EOF drain at 0.99 seconds.
Production HTTP/gRPC acceptance retains two voices in both passes, bilingual text, and an archive that decodes to the exact source PCM.

The three focused reader regressions run in the pinned checkout after copying both reader files:

```sh
cargo test --manifest-path "$SPEAKRS/Cargo.toml" --features cpu --example khalamari_reader
```

The base commit is `80eea6a854390cbb29fbb129980df2362b1d69a6` with the reader audit dirty.
The machine is Apple M3 Max, 48 GiB, macOS 27.2. No ASR service runs during these reader measurements.
The reference checkout remains `5a8f5804c189defe597e8aebb6a82bbfedde9f48`.
The current reader SHA-256 is `12569c63886524640d61b52329252cfddabe1238500a09b301328a0f26944bcd`.
`target/asr-speaker-identity-audit/` retains paired commands, binary/source/checkpoint hashes, updates, scores and rejected candidate replays.

Real Qwen acceptance passed eight HTTP cases and three gRPC cases, including bilingual speech and repeated words.
After the pass-ownership audit, four focused cases passed again, including default LID and timestamp-constrained overlap.
Finals remained in audio order. Speaker-only events preserved the finalized text and revision.
A separate live/offline replay completed both passes. Its FLAC decoded to the exact source PCM.
An invalid reader response terminated that reader while capture saved all 48,000 requested samples.
Cancellation during reader readiness reaped the child and saved the recorded prefix in 0.55 seconds.
The E2E reports are under `target/asr-community-e2e/` and `target/asr-community-final/`.

### Native model setup

Build the Rust client and prepare Whisper assets once.
Also complete the [speaker reader setup](#speaker-reader-protocol) before starting ASR:

```sh
cargo build --release --bin asr
target/release/asr --setup-models
target/release/asr --mode stream
```

CMake and the Xcode command-line tools are required to build the macOS client.
Cargo downloads the checksum-pinned ONNX Runtime 1.31 static archive supplied by `ort-sys` during the first build.
The macOS archive also contains CoreML support. The embedded Silero model selects CPU execution.
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

`--setup-models` prepares the pinned multilingual CT2 Whisper-tiny checkpoint.
It checks each file's byte count and SHA-256 before it replaces the destination.
It validates cached files before reuse and replaces corrupt entries from the pinned source.
A cancelled or failed preparation removes its temporary file.
`--whisper-source` copies the same pinned files from a local snapshot instead of downloading them.
The client does not download models during recording.
If a checkpoint cannot load, analysis reports the error. Capture continues and retains the recording.
`--model-cache` selects the shared external model directory.
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

Whisper uses CPU int8 computation for LID and fixed-text alignment.
Qwen3-ASR remains the only transcript generator.
An explicit `--language` bypasses LID. Speaker attribution still needs Whisper alignment.
Recommendation: Leave `--language` unset for English/Chinese mixtures.
See [service commands](service.md#microphone-and-file-client).

### Continuous client

| Owner | Contract |
| --- | --- |
| `AudioRecordingWriter` | Append-only samples, publication extent and temporary file |
| `AudioRecordingReader` | Independent notifications and range reads |
| `AudioSegmenter` | Analysis cursor, active range, pending finals and revision state |
| Submitted `AudioSegmentTranscribeReq` | One shared immutable job owns bounded audio, language evidence and source coordinates |
| `AudioTranscriber` | Shared alignment model, concurrent service calls and last completed result |
| `AudioStream` | Independent label timelines, segmenter, task admission, final ordering and overlap history |
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
  complete PCM stays in the recording spool
  jobs own bounded Arc snapshots; segments retain ranges, not another PCM queue
```

Sample positions start at recording start and include silence.
They do not reset at segment boundaries or between analysis passes.
Right context and endpoint padding clamp to captured audio, within the validated file-offset domain.
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
`streaming.silence_seconds` and `streaming.sentence_silence_seconds` configure these heuristics.
These thresholds are policy choices, not calibrated model requirements.

The maximum owned range defaults to 24 seconds.
A duration cut searches the preceding three seconds for at least 80 ms of sustained quiet.
It retains the complete suffix. A forced cut can split a word without a suitable pause.
`streaming.window_seconds` and `streaming.lookback_seconds` configure this policy.

LID uses 0.5-second cells with up to 0.5 seconds of context on each side.
The default Whisper encoder pads each observation to its 30-second reference input.
`--fast-lid` (`language.fast`) uses a five-second minimum encoder extent and requires a language score of at least `max(0.9, language.confidence)` to set a hint.
Longer observations keep all their frames. This option changes the LID compute and confidence policy. It preserves the ASR window and the recording clock.
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

The LID reader and alignment tasks share one Whisper model. The LID reader owns its bounded observation cache.
It acquires the CPU-model guard asynchronously before it submits work to the runtime core `SyncTaskPool`.
The fixed pool has one runner. Whisper retains its existing internal thread setting.
A canceled queued task is skipped. A running native call finishes without publishing its canceled result.
Cancellation cannot accumulate blocking calls behind that guard.
Tasks share one HTTP WebSocket or gRPC audio connection. Each segment request retains its own metrics.
Only ordered final commits publish overlap words to later jobs.

Final work has priority. Unsubmitted partials coalesce into the latest snapshot.
The final queue capacity is `max_pending_finals` in both modes. Published context remains available independently of this limit.
Queue backpressure suspends segmentation. Analysis readers have separate bounded output queues. Capture continues into the complete recording.
At EOF, the final active range waits for a queue slot.
Graceful flush waits for pending inference results.
Jobs reuse the last completed Qwen result only for the same recording sample range and language.
This does not reuse encoder computation or decoder KV.
Events report wait/processing durations, request count, submitted audio, cache hits and alignment calls.
These metrics describe emitted jobs. They exclude discarded or cancelled jobs.

### Optional ASR overlap

`streaming.overlap_seconds` defaults to zero.
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

The reader-contract audit compared the same Qwen3-ASR 1.7B 8-bit service, Whisper-tiny, Community-1 reader and PCM inputs
on Apple M3 Max, 48 GiB, macOS 27.2. The baseline was `2ce30814`; the candidate included the local audit changes.
The comparison records dirty state, binary hashes, commands and model hashes under `target/asr-agreement-audit/`.

```text
Same published PCM and model outputs
  before: notifications select short LID cells; each speaker Replace rewrites output
  after:  configured sample grid selects LID cells; changed owned labels rewrite output
          capture/input/completion stay serviced while analysis awaits readiness
```

| Concrete risk | Before | After |
| --- | --- | --- |
| Zero LID context, English then Chinese | Chinese audio was transcribed as English; Chinese evidence was absent | Chinese evidence reaches the existing recovery path; bilingual reference passes |
| Speaker updates, human English/Chinese | 11 events | 7 events; identical final text and latest turns |
| Speaker updates, repeated mixed-language words | 8 events | 0 events; identical final text and latest turns |
| Enter during a controlled four-second reader startup | Capture continued to 4.544 seconds | Capture ended at 1.024 seconds; analysis still drained |

The startup check used a terminal and a file-backed capture substitute. It did not record a live microphone.
Both archives decoded to the exact source prefix. These measurements concern control and redundant output;
they do not establish improved Community-1 diarization accuracy or a general latency bound.


Run the core client contracts:

```sh
cargo test -p inference-asr-client
```

These tests cover publication/read ownership, cancellation, revisions, endpointing, suffix retention, overlap ambiguity and VAD recurrence.
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
Reports retain its hash, audio hashes, settings, evaluator source hash/Git state, offline baselines and raw streaming events.
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
