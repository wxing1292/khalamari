# Executor Tests, Benchmarks, and Profiling

This document owns the executor verification ladder, benchmark surfaces, metrics, profiling conventions, and performance
evidence requirements. Component math and source layouts remain in the matching `executor_*.md`.
[`service.md`](service.md) contains the service lifecycle and end-to-end commands.

## Verification ladder

Use the smallest production owner that can prove the changed contract, then compose upward:

1. Compare each optimized backend component with a slow or CPU reference.
2. Exercise the real production component API and the owner of its metadata and buffer.
3. Verify one real-weight layer path.
4. Scale Qwen layer ranges through `layer0`, `layer4`, `first4`, and `all_layers`.
5. Run the complete Main owner with all transformer layers and final norm.
6. Add Main embedding, gather and unembedding, and ordinary sampling.
7. Add MTP proposal, sparse distributions, rejection, and commit last.

Component gains do not prove end-to-end performance.

Recommendation: Use the same ladder to isolate end-to-end regressions.

Establish correctness and workload identity before timing.

Run Metal and GPU commands strictly serially across tests, benchmarks, services, and worktrees. Use `--test-threads=1`
for a Metal test command.

Keep expected-panic contract tests on the host. This placement prevents one deliberate panic from obscuring later GPU
results.

## Benchmark layers

There are two benchmark levels:

```text
inference-backend-metal
  synthetic, model-independent kernel/component questions

inference-executor-metal
  real checkpoint weights and production component/layer/executor ownership
```

For each executor component, include one-layer production forward perf and configurable setup/kernel/sub-op perf.

Backend Criterion targets:

```text
dense_mlp  sparse_mlp  moe  gqa_split_kv  gqa_bidi_block_sdpa  gdn_attn  gdn_state_io
embedding  unembedding  norm  gated_residual  affine_quantized_matmul  matmul_bf16  buffer_io
```

`rejection_sampling` is the backend custom-CLI target. Model-executor targets:

```text
qwen3_gqa              qwen3_dspark           qwen3_dspark_forward
qwen3_dspark_unembedding                       qwen3_dspark_sampling
qwen35_dense_mlp       qwen35_moe             qwen35_gqa
qwen35_gdn             qwen35_main_layers     qwen35_main
qwen35_main_text_embed qwen35_main_gather_unembed
qwen35_main_sampling   qwen35_vanilla_prefill_decode
```

Real-weight targets accept the model paths that their production owner needs.
`qwen3_dspark_sampling` needs only `--dspark-model-dir`.
`qwen35_main_sampling` does not load model weights.
These targets share `--iters`, `--warmup-iters`, and `--runs`.

Production `src` must not gain benchmark-only state, feature paths, or environment controls.

For the Qwen3x GPU-prepared DSpark path or the Qwen3.5 DFlash2 path, compare a baseline commit or binary with the
current commit or binary.
Do not add a production executor mode, service option, or configuration key to select the old CPU read boundary.
Use the same process shape, model, request batch, seeds, and sampler parameters for both runs.
Record CPU preparation/read time, combined Main-and-Spec wall time, replay cache hit/build state, and proposal
throughput.
The combined submission duration is not a Main-only metric.

For full-cycle measurements, start the Qwen3 or Qwen3.5 DSpark service, or the Qwen3.5 DFlash2 service, with the
production commands in [`service.md`](service.md), and drive it with the production `decode` client. This path
exercises the combined Main-and-Spec submission and reports
`main_spec_replay_elapsed` through the executor timing data. Use identical server and client arguments for the
baseline and current binaries.

Run the initial measurements with the serial Metal compute pass.
Treat a concurrent compute dispatch and a multi-queue overlap as separate measured follow-ups.

## Target meanings

- `gated_residual` measures GR read, write, composition, and fused/unfused dependent chains with synthetic BF16 weights.
  Each submission records 32 invocations at four branches, hidden dimension 2560, and low-rank dimension 320.
  See the [GR contract](executor_gated_residual.md#verification-and-measurement) for shapes and measurement scope.
- The backend `moe` target includes `route_replay128`. The `gdn_attn` target includes `replay128` cases.
  These cases record 128 component invocations per submission.
  This case amortizes submission overhead when single-invocation wall-time samples are noisy.
  Compare the same repeat count and token shape. Keep this diagnostic separate from full MoE forward measurements.
- `norm` measures standalone RMSNorm and residual-add RMSNorm variants.
  The `rms-only/replay64` cases record 64 standalone BF16 RMSNorm commands in one replay.
  This method amortizes command submission and wait overhead for kernel-level comparisons.
- `affine_quantized_matmul` compares automatic and forced QMV/QMM routes.
  Its BF16 profiles cover square 4K/8K and wide 12K/14K/16K/32K outputs, plus a 5120 projection and a mixed-dtype projection.
  Row counts cover 1–8 and include 16, 17, 24, 32, 33, 40, 48, 56, 64, and 65 to expose QMV and QMM routing boundaries.
- `matmul_bf16` measures the production adaptive BF16 matmul owner. It covers the `M = 1` GEMV path and the Steel
  GEMM path with Qwen3-ASR hidden, FFN-up, and convolution-output matrix shapes.
- `buffer_io` measures the production Metal `BufferIO` API at `4`, `64`, and `128 MiB`.

  | Benchmark key | Direction | macOS data cache | Timed completion |
  | --- | --- | --- | --- |
  | `file_to_buffer_uncached` | File to shared Metal buffer | Disabled | `BufferIO::file_to_buffer` returns |
  | `buffer_to_file_uncached_sync` | Shared Metal buffer to file | Disabled | `BufferIOFile::sync_all` returns |

  It excludes snapshot metadata, rename, and parent-directory sync.
  Set `PSI_DEC_BUFFER_IO_BENCH_DIR` to select the storage volume.
  The default directory is the operating-system temporary directory.

  Each size uses this untimed setup:

  ```text
  allocate one shared source Metal buffer
    -> fill it with a repeated 1 MiB byte pattern
    -> allocate one shared destination Metal buffer
    -> create the benchmark file with F_NOCACHE and F_GLOBAL_NOCACHE
    -> buffer_to_file the complete range
    -> sync_all the file
    -> open a separate uncached read handle
    -> file_to_buffer the complete range
    -> validate every destination byte in 1 MiB chunks
  ```

  The timed file-to-buffer case repeatedly reads the same file and range into the same destination buffer.
  `F_GLOBAL_NOCACHE` keeps the Metal URL-backed file handle outside the macOS data cache.
  This condition does not bypass an SSD controller cache.

  The timed buffer-to-file case overwrites the same range with `F_NOCACHE` enabled and calls `sync_all` during each
  iteration.
  File creation, file opening, pattern generation, correctness validation, and cleanup remain outside all timed cases.
- `qwen35_gqa` selects `--gqa-model 27b|35b` and accepts the SplitKV `single_q` or `tiled_q` variant. It derives FP8
  E4M3FN KV tokens per page from the production 32 KiB page size and the selected model profile. It can run an explicit
  untimed `--validate-split-kv-tiled-q` comparison. `--max-tokens` fixes the token and Map task-template capacities.
  Each candidate derives its partial-state capacity from its registered Q-range width.
  The default is the server default of 128. Each case reports the active KV splits, fixed-TQ reserved partial slots,
  active partial states, and segment distribution.
- `qwen3_gqa` loads real Qwen3 ungated-GQA weights. It measures full replay, SplitKV-only variants, and exact QKV/output
  projection kernels.
- `qwen3_gqa` exposes static SplitKV tile geometry as CLI arguments. It can validate SingleQ output against TiledQ
  output.
  Its projection probes compare QMV, QMM BM8/BN32, and QMM BM16/BN32. These forced paths are benchmark-only.
- `gqa_bidi_block_sdpa` measures the model-independent bidirectional local-block SDPA map component.
  It accepts block size, request count, head geometry, partial-state Q width, and dtype as CLI arguments.
  The default `max_q_tokens = 8` matches the current production TiledQ partial-state layout.
  The backend owns its 32-thread threadgroup geometry.
- `qwen3_dspark` loads real Main and DSpark checkpoints.
  It runs the public executor lifecycle for `main` or `dspark`.
  It uses the checkpoint `block_size` as the DSpark proposal length.
  It reports each record, submit/wait, read, and commit boundary.
  It reports `num_spec_tokens`, proposed tokens, accepted tokens, generated proposals, and acceptance.
- `qwen3_dspark_forward` loads real Main and DSpark weights.
  It compares the complete `MainTextEmbed` and Main forward with `DSparkEmbed` plus the complete DSpark backbone and
  final norm.
  The Main and DSpark depth-comparison cases use the same request count, seven rows per request, and history length.
  The result reports total and per-layer time.
  Use the per-layer value to compare the 40-layer Main stack with the five-layer DSpark stack.
  The additional `main-verification` result uses eight rows per request and a DSpark-enabled Main executor.
  It includes `MainTextEmbed`, all Main layers, Main residual capture, and DSpark context projection.
  It excludes Main `GatherUnembed` and rejection sampling.
- `qwen3_dspark_unembedding` loads the production `Qwen3xDSparkGatherUnembed` component.
  It uses DSpark-owned unembed weights when they exist.
  Otherwise, it uses the Main unembed weights, as the production executor does.
  It measures gather and unembed together for `block_size` rows per request.
- `qwen3_dspark_sampling` loads real DSpark Markov weights.
  It measures the complete sequential fused Markov-map, sampling, and write-distribution replay.
  Each proposal step uses one fused W1, W2, base-logit-add, and tile-Top-K map.
  It then uses the generic sample-and-write-distribution reducer.
  It accepts `--temperature`, `--top-k`, `--top-p`, and `--seed`.
  It prints proposal token IDs, exact proposal probability bits, and a stable fingerprint of the complete sparse draft
  distribution.
  `qwen3_dspark_forward`, `qwen3_dspark_unembedding`, and `qwen3_dspark_sampling` isolate the three ordered proposal
  segments:

  ```text
  DSparkEmbed + DSpark forward + final norm
    -> GatherUnembed
    -> Markov correction + sampling
  ```

- `qwen35_gdn` measures the production chunkwise/replay GDN forward with the 35B-A3B profile.
  It excludes accepted-state commit and cache publication.
  `--tokens-per-req` selects a ragged request layout. `--prefill-requests` selects the chunkwise prefix.
  The prefix writes final states. The suffix writes per-token F32 replay logs.
  `--compare-recurrent` compares an all-recurrent forward with the same full-state write map and no replay log.
  This reference is not the previous complete model lifecycle. Both graphs share weights, inputs, and initial state.
  The untimed check compares outputs and requested snapshots. Timed iterations alternate their order.
  `--subcomponents` uses the same forward contract.
- `qwen35_moe` compares token-major and expert-major policies for real sparse-model weights.
- `qwen35_main_layers` records only selected `Qwen35MainLayer` owners.
  It accepts `layer0`, `layer3`, `layer4`, `first4`, `all_layers`, or `layersSTART-END`.
  `layersSTART-END` selects the half-open layer range `[START, END)`.
  `all_layers` does not include final norm and does not measure the complete Main stage.
  For the 27B schedule, `layer3` is the first GQA layer. The bench records fixed active extents and does not submit
  dynamic replay arguments. `--max-tokens` defaults to 128. It fixes the scratch and metadata capacity and the current
  Map task-template scheduling budget. GQA cases report the selected variant, padded replay extent, and materialized
  segment distribution.
- `qwen35_main` records the production `Qwen35Main` owner.
  Its implicit `main` case includes all transformer layers and final norm.
  It uses the production replay key and active-token arguments.
  It does not include MainTextEmbed, GatherUnembed, sampling, or readback.
- `qwen35_main` and `qwen35_main_layers` use synthetic hidden inputs and initialized component metadata.
  They validate `context + num_tokens` against the model position capacity.
  Use `qwen35_vanilla_prefill_decode` when the measurement requires a real committed context.
- `qwen35_main_text_embed` records the production `Qwen35MainTextEmbed` owner with real checkpoint weights.
  It accepts `--tokens` and `--max-tokens`.
- `qwen35_main_gather_unembed` records the production `Qwen35GatherUnembed` owner with real checkpoint weights.
  Its `gather_unembed` case measures gather and unembedding together.
  It accepts `--rows` and `--max-rows`.
  Standalone Gather and Unembed measurements remain backend component benchmarks.
- `qwen35_main_sampling` records the production ordinary `Sampling` replay owner.
  It accepts `sample` and `sample_readback` cases.
  It uses synthetic BF16 logits and does not load model weights.
  It does not include draft write-distribution or rejection sampling.
  The default sampling values are seed 42, temperature 0.7, Top-K 20, and Top-P 0.8.
  It accepts the production greedy values `temperature=0` and `top_p=0`.
- The Qwen3.5 Main component targets run `--warmup-iters` before each result sample.
  They sort numeric shape lists and reject duplicate numeric shapes.
- `qwen35_vanilla_prefill_decode` measures context-aware Prefill and Decode through the public
  `ReplayableDecoderModel`
  lifecycle.
  It resets request slots and rebuilds each starting context with committed, chunked Prefill work before each
  trajectory.
  The operation timer excludes the reset and context rebuild.
  Prefill skips gather, unembedding, sampling, and readback because it has no Main output rows.
  Decode commits one visible sampled token per request and uses that token as the next input.
  `--prefill-tokens` and `--decode-tokens` are per-request operation totals.
  Prefill splits totals that exceed the active batch capacity.
  The active per-request chunk is `min(max_tokens_per_request, max_tokens / num_reqs)`.
  The fixture allocates unique Main KV and GDN state page IDs from one shared page-ID domain.
  It sends only newly materialized cache blocks for each request.

  The target accepts these workload and capacity options:

  ```text
  --cases prefill,decode
  --contexts N[,N...]
  --prefill-tokens N[,N...]
  --decode-tokens N[,N...]
  --num-reqs N
  --max-tokens N
  --max-tokens-per-request N
  --num-tokens-per-block N
  --num-cache-pages N
  --seed N
  --temperature F
  --top-k N
  --top-p F
  --warmup-iters N
  --iters N
  --runs N
  ```

  The default cases are `prefill,decode`.
  The default contexts are `0,1024,4096,8192`.
  The default Prefill totals are `64,128`.
  The default Decode total is `32`.
  The default capacities are one request, 128 batch tokens, 128 request tokens, 2,048 block tokens, and 393,216 pages.
  The default sampling values are seed 42, temperature 0.7, Top-K 20, and Top-P 0.8.
  The default timing controls are two warmup iterations, five measured iterations, and five runs.

  The target sorts list values and rejects duplicate shapes.
  It generates deterministic model-valid token IDs without a tokenizer.
  Each result includes input and output FNV-1a fingerprints.
  A Prefill result uses an untimed deterministic Decode probe for its output fingerprint.
  The target verifies case-order independence and Prefill chunk decomposition before measurement.
  It also compares a single-step Decode probe after alternate context chunking when the capacity permits the
  comparison. The single step isolates context construction from autoregressive sampling feedback.
  Repeated trajectories and case-order checks require exact token and probability bits.
  Chunk-decomposition checks use an untimed greedy Decode probe with `temperature=0` and `top_p=0`.
  They require exact next-token identity.
  This probe separates execution chunking from stochastic sampling sensitivity.
  Measured trajectories and exact repeat checks continue to use the requested sampling configuration.
  Each result reports the replay-cache-cold operation separately from steady-state samples.
  Result timing separates context rebuild, prepare, record, Main replay, Main sampling replay, Spec replay, finish, and
  commit feedback.
  Context rebuild timing includes the request-slot reset.
  Provenance reports the commit, dirty state, model path, machine, operating system, architecture, and relevant
  environment.

Representative smoke commands:

```text
cargo bench --bench gqa_bidi_block_sdpa -- \
  --block-sizes 7 --num-requests 1 \
  --max-q-tokens 8 \
  --iters 1 --warmup-iters 0 --runs 1

PSI_DEC_BUFFER_IO_BENCH_DIR=<storage-directory> \
  cargo bench --bench buffer_io

cargo bench --bench rejection_sampling -- \
  --mode dspark-markov-top-k-map --rows 1 --top-k 20 --vocab 151936 \
  --markov-rank 256 --markov-w1-group-size 64 --markov-w1-bits 4 \
  --markov-w2-group-size 64 --markov-w2-bits 8 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench --bench qwen3_gqa -- \
  --model-dir <qwen3-model-dir> --tokens-per-req 16 --contexts 128 \
  --iters 1 --warmup-iters 0 --runs 1 --validate

cargo bench --bench qwen3_dspark -- \
  --model-dir <qwen3-model-dir> --dspark-model-dir <dspark-model-dir> \
  --cases dspark --num-requests 1 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench --bench qwen3_dspark_forward -- \
  --model-dir <qwen3-model-dir> --dspark-model-dir <dspark-model-dir> \
  --num-requests 1 --context 128 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench --bench qwen3_dspark_unembedding -- \
  --model-dir <qwen3-model-dir> --dspark-model-dir <dspark-model-dir> \
  --num-requests 1 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench --bench qwen3_dspark_sampling -- \
  --dspark-model-dir <dspark-model-dir> --num-requests 1 --top-k 1 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench --bench qwen35_gqa -- \
  --model-dir <27b-model-dir> --gqa-model 27b --tokens 1 \
  --contexts 0 --num-reqs 1 --gqa-split-kv-variants single_q \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench --bench qwen35_gqa -- \
  --model-dir <27b-model-dir> --gqa-model 27b \
  --gqa-tokens-per-req 64,1 --gqa-contexts-per-req 1024,65536 \
  --max-tokens 128 \
  --gqa-split-kv-variants single_q,tiled_q \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench --bench qwen35_main_layers -- \
  --model-dir <27b-model-dir> --cases layer0 --tokens 1 --contexts 0 \
  --max-tokens 128 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench --bench qwen35_main -- \
  --model-dir <27b-model-dir> --tokens 1 --contexts 0 \
  --max-tokens 128 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench --bench qwen35_main_text_embed -- \
  --model-dir <27b-model-dir> --tokens 1 --max-tokens 128 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench --bench qwen35_main_gather_unembed -- \
  --model-dir <27b-model-dir> --rows 1 --max-rows 128 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench --bench qwen35_main_sampling -- \
  --cases sample,sample_readback --rows 1 --vocab-size 151936 --top-k 20 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench -p inference-executor-metal --bench qwen35_vanilla_prefill_decode -- \
  --model-dir <model-dir> --cases prefill,decode \
  --contexts 7 --prefill-tokens 2 --decode-tokens 2 \
  --num-reqs 1 --max-tokens 2 --max-tokens-per-request 2 \
  --num-tokens-per-block 2048 --num-cache-pages 16384 \
  --iters 1 --warmup-iters 1 --runs 1
```

Run one performance command at a time. List the planned cases first. GPU contention and memory pressure invalidate
comparisons.

## Metrics

`setup_us` includes model loading and fixture construction.
The Qwen3 DSpark executor target uses `cache_miss_*` for its first complete execution.
`qwen35_vanilla_prefill_decode` uses `replay_cache_cold_*` for the first operation after `clear_replay_cache()`.

Executor trajectory targets report wall time and their production lifecycle boundaries.
The Vanilla target reports the Main, Main-with-sampling, and Spec replay values from `ModelOutputTiming`.
It reports prepare, record, finish, and feedback/commit as distinct host boundaries.

Force-sync and profile-summary measurements are diagnostic metrics. They are not normal wall-clock throughput. Never
compare the two measurement types as equivalent workloads.

Benchmark keys contain only comparison dimensions. Useful dimensions include:

- Model
- Storage and backend
- Operation
- Batch and tokens
- A meaningful context or state coordinate

Do not include these values:

- Default layer 0
- Generic `detail` or `sub-op`
- Metadata that output already shows elsewhere

```text
gqa/qwen36-27b/metal/full-forward/b1t64-c1024
gdn/qwen36-35b-a3b/metal/full-forward/with-state/b1t1-s1
moe/qwen36-35b-a3b/metal/token-major/b1t64
```

## Profiling and logging

The service uses static `profiling::span(...)` names for its coarse CPU tree. Put dynamic request and shape values in
structured logs.

The `--profile component|operation` modes currently produce the same CPU tree.
`MetalRuntime` enables relaxed Metal 4 GPU stage timestamps by default.
The backend Stream API supports precise timestamps for explicit diagnostics. Precise timestamps can change execution performance.
Use service `executor_cpu_ms`, not `main_cpu_ms`, to compare a split Main-to-Spec lifecycle with an integrated
lifecycle.
[`service.md`](service.md) documents the GPU timing fields and defaults.

## Performance evidence

A performance claim records all of:

```text
commit and dirty state
machine, OS, architecture, and relevant environment
model/checkpoint and command
sampling config and deterministic seed
metric and workload/trajectory fields
baseline and current samples
verdict
```

For speculative decoding, also record these values:

- Proposals
- Sampled tokens
- Accepted tokens and chunks
- Acceptance efficiency

A throughput change with a different deterministic acceptance trajectory is not a pure executor or kernel comparison.

A checked-in end-to-end reference is an observed result, not a threshold. The owning helper must record its complete
provenance. It must suppress throughput deltas when the current provenance or deterministic trajectory does not match
the reference.
