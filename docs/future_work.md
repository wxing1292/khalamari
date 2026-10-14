# Future Work

This file tracks unresolved implementation work and bounded investigations.
Current behavior and completed designs belong in the [owning documents](README.md).
[Engineering conventions](engineering_conventions.md) own shared optimization, validation, and replay rules.

## Agent Integration

- Separate turn interruption from session shutdown in the Pi provider. Its `AbortSignal` currently closes the complete generation stream.
- Add a Codex-compatible adapter above the generation RPC. The adapter must own durable threads, turns, items, approvals, and tool execution.
- Add reconnect and event replay only when an agent requires recovery from a broken stream. Keep durable identity in the adapter.
- Add resident-session branches with shared immutable token/cache prefixes and independent mutable GDN state.
  Define GDN state copying before allowing prefix rewrite.

## Runtime Lifecycle

- Connect request-slot and page pressure to `Sessions::evict_one_req`. Add periodic `Sessions::evict_expired` invocation outside the session manager.
- Add partial-eviction recovery at `Sessions::resume`. Preserve request identity, token metadata, trie ownership, and cache-lane page identity.
  Restore KV/GDN state before scheduling. A fully evicted request returns `Evicted` and requires complete input.
- Measure repeated-media workloads before adding cross-request resource sharing.
  Reuse the runtime pin-cache ownership model. Shared payload must be immutable and retain one RAII allocation owner.
- Add bounded retries for retryable resource-materialization failures. Preserve assertions for programming errors.
- Replace the `ResourceMaterializationResp` downcast before adding a second async-task response type.
  Keep typed dispatch independent of `UserRequest` and `AsyncTaskPool`.
- Redesign host-pinned segment ownership before offload. Require unique ownership for mutation/free and explicit cross-thread rules for read-only views.
- Add per-request and trie-block onload/offload through the [model-state I/O lifecycle](model_state_io.md).
  Keep `AwaitReservation` scheduler-owned. Reserve `Swapped` for nonresident model state.
- Measure full/selected snapshot I/O: bitmap scanning, fragmented-range Metal submission, vectored writes, and peak host-memory overhead.
  Evaluate integrity hashing without a second SSD pass, including the CPU scan after a load.
  Preserve synchronous lifecycle APIs, FIFO Stop, atomic publication, and the existing schema/resource validation.
  Add a worker only if measurements justify it.

## Pipeline Parallelism

- Complete per-request readiness for overlapping Prefill/Decode with `max_compute_slots > 1`.
  An earlier response must not release a request whose later Decode remains in flight.
- Add the unused-sticky-budget `FIFOBatcher` regression case: an ID-map-only in-flight request must not block the next runnable request.
- Audit the Trie Decode fallback after overlapping MTP Prefill. Account for the scheduled range before runtime commit.
- Enforce request-local FIFO commit and LIFO cancellation before scheduler mutation.
- Validate compute-sequence FIFO across slots, requests, and responses. An out-of-order response is a contract violation.
- Define `PipelineStageResult` with sequence, stage, request-slot identity, and model-specific payload.
  Preserve ragged hidden-row association in the Qwen payload.
- Add Qwen stage configuration and transport. Non-first stages ingest hidden states; non-last stages emit them.
  The last stage owns final norm, unembedding, sampling, rejection, and MTP proposals.
- Send final decisions to every stage. Each stage must commit its pending state transaction before runtime receives the response.
- Permit overlap only after transport, pending transactions, and cache publication have bounded in-flight ownership and causal order.

## Confidence-aware Scheduling

Complete the PP>1 request lifecycle before these changes. See [token-budget allocation](token_budget_allocator.md).

- Add shadow telemetry for proposal position, confidence bin, selected prefix, validation result, request, and model.
- Add runtime-owned confidence calibration only when telemetry supports it. Do not select an absolute threshold without calibration.
- Add measured cost estimates for fixed cost, Main rows, padding, replay transitions, acceptance, and admission.
- Add page-feasibility feedback independently of the token budget.
- Verify mixed Prefill/Decode, request-local proposal trimming, probability/confidence lengths, visibility, and pipeline fill/drain.
  Keep proposal runs at configured `num_spec_tokens` until these gates pass.

## Prefill

Revisit high-throughput long Prefill after the current Decode work.
Select changes from end-to-end TTFT and prompt throughput.
Preserve ragged batches, mixed Prefill/Decode, and cache/state lifecycle.

## Multimodal Input

- Measure resource-level early unload: peak arena bytes, rematerialization, and repeated audio/image/video placements.
- Add long-audio chunking/streaming after collected Qwen3-ASR parity. Define timestamps, overlap, context carry, and output merging first.
- Add model-owned image processing, Vision Tower, spatial merger, and true T/H/W M-RoPE.
  Reuse runtime resources and `ResourceEmbed` replacement.
- Add video after image support. Define demux, sampling, timestamps, and segment mapping from reference fixtures.
- Measure first-Prefill materialization and batch interference before adding separate resource scheduling.
- Compare tower attention with decoder bidirectional GQA after both tower baselines are stable.
  Share lower-level SDPA only when contiguous tower tensors and paged decoder state retain their distinct contracts and show a measured benefit.

## Replay Evolution

- Revisit backend-neutral replay before adding another backend.
  Keep shared execution lifecycle neutral and Metal replay/fusion backend-owned.
- Add a bounded replay catalog only when measured memory growth requires it.
  Eviction must cover all stage caches and retain in-flight resources through completion.
- Design multi-command-buffer chains around MTP rejection readback, allocator-reset ownership, and proposal dependencies.

## Performance Investigations

### Benchmark coverage

Extend the [existing component matrix](executor_benchmarks.md#component-latency-matrix) and benchmark owners rather than adding production benchmark paths.

- Extend real-weight coverage beyond the current fixtures. Add ragged GQA/GDN segments and controlled sparse expert occupancy.
  Keep sparse-expert-only latency separate from complete MoE routing, layout, shared experts, and combine.
  Keep measurement, reference checks, and dataset orchestration separate from policy fitting.
- Compare oMLX/TensorFold only with matching inputs, weights, precision, state writes, and operation boundaries.
  Separate sparse expert compute from complete router/layout/combine work.
- Extend sparse-model E2E coverage separately from the current dense serving matrix.
  Keep cold initialization, retained-prefix preparation, full Prefill, and steady-state Decode as separate metrics.

### GQA

- Extend [selection calibration](executor_benchmarks.md#generic-gqa-matrix) across Apple GPUs, ragged requests, cache dtypes, and visibility windows.
  Evaluate complete route latency and expensive-case regret before reducing criteria.
  Withhold context lengths when fitting both geometry and partition criteria.
  Record grid size, GPU cores, occupancy, and reduction traffic.
- Verify recorded capacities `64,128,256,384` and shared scratch through production service execution.
  CPU policy/capacity coverage reaches 256K. Selection calibration and numerical acceptance above 32K remain outstanding.
  The native-cache component latency matrix also covers uniform 64K ctx. This measurement is not policy or numerical acceptance.
- Tighten the initialized partial-state envelope from compatible query geometry and Map bounds.
  Do not multiply independent worst-case capacities. Preserve the shader u32 element-index check.
  The 27B GQA fixture with a 131072-token capacity currently reserves 6442450944 partial-output elements and fails this check before GPU work.
  A shared maximum capacity can also reject smaller benchmark shapes. Initialize each full-Prefill shape at its own capacity.
- Compare a complete base-2 softmax family across SlicedKv, SlicedQ, bidirectional SDPA, and reducers.
  Preserve one compatible partial-state ABI and require parity plus a measured gain.
- Evaluate pipelined K/V loads after checking an unpadded layout.
  The padded `Tkv=16`, `D=256` double buffer requires 33 KiB; an exact-32-KiB design needs separate correctness and bank-conflict evidence.
- Isolate short-context FP8 conversion, page lookup, reduction, and launch cost within the same cache contract.
- Re-probe native E4M3 conversion only when the device and compiler expose the required operation.
  Tensor storage support alone is insufficient. Compare generated instructions, traffic, occupancy, parity, and E2E latency.

### GDN

- Evaluate V-row/token-staging variants using request lengths/counts, head geometry, and state-write requirements.
  Preserve F32 live state, logs, and accepted-prefix reconstruction.
  A dynamic compute choice must enter GDN topology before composing with GQA in the existing Main replay key.
- Compare request buckets with `min(max_requests, num_total_tokens)` using a shared recorded token capacity.
  Measure key count, first-record time, steady-state latency, and inactive dispatches.
  Verify outputs, logs, accepted states, reload, and shared scratch before changing the default.
- Revisit TensorOps commit only if arbitrary accepted prefixes retain vectorized recurrence's short-prefix performance.
  Cover zero-length copies, partial dimensions, active jobs, and CPU-reference states.
- Validate BF16 operands on long real trajectories and correlated-key stress cases.
  Keep F32 state/accumulation while investigating the [documented error bounds](executor_gdn.md).
- Measure short-input routing and post-forward commit with controlled old/new runs.
  Keep materialized-state capacity bounded by accepted final states and cache boundaries.

### Affine matmul and MoE

- Extend [Steel/sliced-K calibration](executor_dense_mlp.md#backend-selection) across dimensions, dtypes, groups, devices, and row counts.
  Separate projection execution from caller overhead. Use the shared affine selector for MLP, GQA/GDN projections, and LM head.
- Evaluate cross-threadgroup split-K for small-M, long-K projections with insufficient output parallelism.
  Compare BM16/BN32 and wider row tiles, including partial storage, reduction cost, scratch ownership, and projection rounding.
- Measure two-row weight reuse independently. Check packed-dot group accumulation and input magnitudes against the scalar affine reference.
- Extend sparse expert calibration across devices, dimensions, and natural routing distributions.
  Preserve the [current per-expert QMV/QMM strategy](executor_moe.md) until a measured alternative improves mixed segments.
- Reduce empty complementary dispatch and padded-candidate cost without a CPU routing readback.
  Whole-batch selection regressed on the measured mixed segments. Preserve compact routing, fused gate/up/SiLU, and replay dependencies.

### Whole-model measurement

- Diagnose eight-request memory before enlarging arenas.
  Separate weights, arena pages, request-local state, scratch, and replay resources; compare independent and shared prefixes.
  Distinguish physical Metal allocation failure from logical page exhaustion.
- Attribute speculative-round time through production Main, GQA/GDN, projections, sampling/rejection, retained-feature/history updates, draft, and commit.
  Compare fixed proposal windows before default engine policies. Record proposal sources, candidate vocabulary, quantization, and sampling contracts.
- Add an opt-in Metal timeline profiler and correct replay/ICB timestamp ordering.
  Validate samples against completion feedback in relaxed/precise modes.
  Preserve encoder dependencies and output/acceptance/state parity. Keep profiling separate from normal throughput.
- Measure cold model/device setup, pipeline compilation, replay recording, and first distinct shapes separately from steady-state requests.
- Evaluate BF16 radix Top-K for sample-only and distribution operations.
  Preserve token-ID ties, nonfinite inputs, temperature/nucleus filtering, and rejection probabilities.

## Model and Backend Investigations

- Validate GQA, GDN, affine QMM, and grouped expert TensorOps on M5 before acceleration claims.
  Compare operands, tiles, thresholds, and full-model Prefill/Decode with cache, visibility, snapshot, and commit parity.
  Re-test GQA PV width 256 on target hardware; current [GQA](executor_gqa.md) retains width 128.
- Audit per-tensor quantization outside MoE routing against supported checkpoint contracts.
  Resolve exact binding subtrees when required. Share metadata/page tables/scratch only across compatible layouts.
- Add Main-only warm-up for MTP requests shorter than `num_spec_tokens`.
  Runtime core must initialize verified history and cache lanes before switching mode.
- Isolate one-row/multi-row Qwen3 Main divergence through GQA, dense MLP, and unembedding.
  Keep exact batch-shape reproducibility separate from rejection correctness.
- Integrate [gated residual](executor_gated_residual.md) through a supported model checkpoint contract.
- Add gated DSpark attention or new checkpoint variants only from real checkpoint requirements.
  Keep gate-neutral history/block/reducer/scratch contracts; preserve official target layers and proposal layouts.
- Permit overlapping DSpark batches only after scratch, outputs, replay arguments, and probability stores have bounded in-flight ownership.
- Compare a DFlash2 ring cache with paged history for ephemeral requests, window overflow, prefix reuse, forks, eviction, and fragmentation.
  Measure retained bytes, page-table work, copies, and E2E latency before changing the owner-local default.
- Evaluate grouped draft Prefill projections only for a measured bottleneck.
  Group compatible Wk and Wv layouts separately and preserve direct paged output without aggregation copies.
- Define backend-neutral immutable `Weight` / `Tensor` / `Storage` ownership.
  Keep file/mapping lifetime in checkpoint readers, semantic identity in model planning, and buffer/view materialization in each backend.
  Preserve initialization-only conversion and keep Metal buffers out of executor-core.
