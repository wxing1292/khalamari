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
dense_mlp  sparse_mlp  moe  gqa_split_kv  gqa_bidi_block_sdpa  qsa_indexer  gdn_attn  gdn_state_io
embedding  unembedding  norm  gated_residual  affine_quantized_matmul  matmul_bf16  buffer_io
```

`rejection_sampling` and `gqa_sdpa_matrix` are backend custom-CLI targets. Model-executor targets:

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

The `ngram_embedding` backend benchmark records production H1/H4 lookup and hash+lookup with 32 invocations per replay.
Its [component guide](executor_ngram_embedding.md#verification-and-benchmark) defines the resident layout, selection, and measurement scope.

Production `src` must not gain benchmark-only state, feature paths, or environment controls.

The [QSA indexer guide](executor_qsa_indexer.md#benchmarks) separates the production replay benchmark from
the cross-framework scorer study. Its [result snapshot](qsa_indexer_performance.json) preserves the measured candidate medians and source identity.

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

## Component latency matrix

`component-latency-bench` plans, resumes, and reports real-weight component measurements.
The controller uses existing executor benchmark workers. It does not change production kernel selection.

```text
CLI shape product                         Component measurement key
requests x contexts x queries  -- GQA --> request segments, ctx, Q, recorded capacity
                               -- GDN --> request segments, zero/nonzero state, write mode
                               -- MLP --> total rows = requests * queries
                               -- sparse MLP / MoE --> total rows = requests * queries

full Prefill: requests x full  ----------> ctx = 0, Q = full input per request

plan -> canonical cases -> worker -> durable JSONL -> grouped Markdown/JSON report
                           |
                           +-- no new swap -> measured -> skip on resume
                           +-- swap/error -> unusable/failed -> stop -> retry on resume
```

| Component | Worker | Fixture | Timed path |
| --- | --- | --- | --- |
| GQA | `qwen35_gqa` | 27B or 35B-A3B | Full production replay with the automatic lookup |
| GDN | `qwen35_gdn` | 27B or 35B-A3B | Full chunkwise/replay forward |
| Dense MLP | `qwen35_dense_mlp` | 27B | `full_auto` |
| Sparse MLP | `qwen35_moe --sparse-only` | 35B-A3B, layer 0 | Expert-major gate/up/SiLU and down projections |
| MoE | `qwen35_moe` | 35B-A3B, layer 0 | Complete token-major and expert-major graphs |

MoE includes routing, layouts, sparse expert MLPs, shared experts, and combine.
It is not a sparse-expert-only latency.
The sparse MLP worker prepares routing and the packed input before timing.
It excludes routing, packing, shared experts, and combine.
Use `--routing-output <prefix>` with one `--tokens` value and `--impls expert_major` to export route IDs and packed order.
The worker writes `<prefix>.indices` and `<prefix>.routes` as little-endian `u32` arrays.
These files let a peer benchmark use the same expert assignments and input order.
The backend `sparse_mlp` target also isolates sparse expert computation with generated weights.

| Axis | Default values |
| --- | --- |
| Requests | `1,2,4,8` |
| Existing ctx per request | `0,1024,2048,4096,8192,16384,32768,65536` |
| New input per request | `1,2,3,4,5,6,7,8,12,16,32,64,128,256` |
| Full Prefill per request | `2048,4096,8192,16384` |

The default product contains 464 logical cases per component.
The controller measures equivalent component inputs once. Each report row names that measurement.

GQA uses real projection weights, synthetic hidden inputs, and initially zero FP8 cached K/V.
The ctx axis changes paged visibility, positions, and allocated cache capacity.
It does not construct a real model prefix.
`--max-tokens` defaults to 256. Each GQA case raises this recorded capacity when its active rows require more space.

GDN uses zero state at ctx=0. All positive contexts use the same fixed nonzero state fixture.
Their report rows therefore reuse the same materialized-state measurement.
The default path writes final Prefill snapshots.
`--gdn-replay` makes append cases write per-token replay logs. Full Prefill cases still write final snapshots.
The measurement excludes accepted-state reconstruction and cache publication.

Dense MLP, sparse MLP, and MoE receive flat input rows. They have no request segmentation or ctx input.
Cases with the same total row count reuse one measurement, including equivalent full-Prefill rows.

Latency is the median replay submission-and-wait wall time in milliseconds.
Raw logs retain each run sample. Setup, loading, compilation, and prefix construction are outside the timing.
Full Prefill here means one component forward from zero history. It is not complete model Prefill or E2E latency.
Use the [fixed-work E2E matrix](#cross-engine-serving-matrix) for serving comparisons.

Build the controller without a GPU run:

```sh
cargo build --release -p inference-executor-metal --example component-latency-bench
```

Generate all five plans without loading a model or initializing Metal:

```sh
for component in gqa gdn dense-mlp sparse-mlp moe; do
  target/release/examples/component-latency-bench \
    --component "$component" --mode plan --out "target/perf/components/$component"
done
```

Run a component after GPU measurements are authorized:

```sh
caffeinate -i target/release/examples/component-latency-bench \
  --component gdn --mode run --model-dir <27b-model-dir> \
  --requests 1,2,4,8 --contexts 0,1024,2048,4096,8192,16384,32768 \
  --queries 1,2,4,8,16,32,64,128,256 --full 2048,4096,8192,16384 \
  --hours 2 --out target/perf/components/gdn
```

`run` builds only the selected optimized worker with `cargo bench --no-run`.
The controller reads checkpoint metadata to select the 27B or 35B-A3B fixture.
The worker validates the actual tensors. Dense MLP requires 27B. Sparse MLP and MoE require 35B-A3B.

Repeat the same command to resume. `--hours 0` removes the deadline.
A positive deadline starts after worker build and identity checks. The current case completes before the controller stops.
Measured cases are skipped. Failed or swap-affected cases are retried with a new raw log.
Any new swap-in or swap-out invalidates the case and stops the run.
Existing swap occupancy alone does not invalidate a new zero-activity measurement.
Run GPU workers serially across all processes and worktrees.

`plan.json` contains logical cases and canonical measurement keys.
`identity.json` binds results to checkpoint metadata, shard file metadata, measurement source, worker binary, hardware, and timing options.
The controller must use a new output directory when that identity changes.
Documentation-only edits and Git history rewrites do not change measurement identity.
Each JSONL record also includes the commit, dirty state, controller command, worker arguments, and swap counters.

`results.jsonl` records each completed attempt. Resume removes only an incomplete final record.
`report.md` groups and sorts rows by append/full Prefill, requests, ctx, and input length.
`report.json` retains the same groups as structured rows.
`--mode report` rebuilds reports without GPU work. Supply the same shape axes and GDN write mode.

The controller does not run oMLX or TensorFold workers.
An external comparison must match weights, input tensors, state contents, precision, state writes, and operation boundaries.

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
- `qwen35_gqa` selects `--gqa-model 27b|35b` and accepts SplitKV `auto`, `sliced_kv`, and `sliced_q` variants. It derives FP8
  E4M3FN KV tokens per page from the production 32 KiB page size and the selected model profile. It can run an explicit
  untimed `--validate-sdpa-sliced-q` comparison. `--max-tokens` fixes the token and Map task-template capacities.
  Each candidate derives its partial-state capacity from its registered Q-range width.
  The default is the server default of 128. Each case reports the active KV splits, fixed-TQ reserved partial slots,
  active partial states, and segment distribution.
- `qwen3_gqa` loads real Qwen3 ungated-GQA weights. It measures full replay, SplitKV-only variants, and exact QKV/output
  projection kernels.
- `qwen3_gqa` exposes static SplitKV tile geometry as CLI arguments. It can validate SlicedKv output against SlicedQ
  output.
  Its projection probes compare QMV, QMM BM8/BN32, and QMM BM16/BN32. These forced paths are benchmark-only.
- `gqa_bidi_block_sdpa` measures the model-independent bidirectional local-block SDPA map component.
  It accepts block size, request count, head geometry, partial-state Q width, and dtype as CLI arguments.
  The default `max_q_tokens = 8` matches the current production SlicedQ partial-state layout.
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

- `qwen35_gdn` measures the production chunkwise/replay GDN forward with the 27B or 35B-A3B profile.
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
  `--reuse-context` selects a Prefill-only matrix with retained prefix pages.
  Each context must end at a `--num-tokens-per-block` boundary.
  This mode builds the prefix once per context.
  Each sample resets its request slot and restores the retained state through `prepare_batch`.
  Its wall timer includes metadata preparation, state restoration, Main execution, and the final state commit wait.
  It emits `matrix_result` with cold, warm wall, submission, and preparation times.
  Its untimed warmup probes verify that restoration preserves the output tokens.
  At one publication boundary, the 8-token case also checks a recomputed prefix.
  `PSI_PREFILL_SKIP` contains completed `context:tokens` pairs, separated by commas.
  The comparison runner sets this benchmark-only variable when it resumes a matrix.
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
  --contexts 0 --num-reqs 1 --gqa-split-kv-variants sliced_kv \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench --bench qwen35_gqa -- \
  --model-dir <27b-model-dir> --gqa-model 27b \
  --gqa-tokens-per-req 64,1 --gqa-contexts-per-req 1024,65536 \
  --max-tokens 128 \
  --gqa-split-kv-variants sliced_kv,sliced_q \
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

Run one performance command at a time. List the planned cases first. Concurrent GPU benchmarks and memory pressure
invalidate comparisons.

When the desktop is in use, alternate baseline and candidate measurements in the same session.
Compare paired latency ratios. Discard unstable pairs.
An absolute difference from an idle historical run does not establish a regression.

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
An ICB stage timing claim must validate the timestamp boundaries against GPU completion feedback.
Monotonic encoder timestamps alone do not prove that a sample includes the preceding ICB work.
The local M3 Max audit observed unavailable relaxed samples and implausibly short precise Main samples.
Use command-buffer completion times for an independent check.
A diagnostic that submits and waits for each stage must preserve the output and acceptance trajectory.
Keep its added synchronization cost separate from normal service throughput.
Use service `executor_cpu_ms`, not `main_cpu_ms`, to compare a split Main-to-Spec lifecycle with an integrated
lifecycle.
[`service.md`](service.md) documents the GPU timing fields and defaults.

## Performance evidence

[Component latency data](component_performance.md) records the existing five-component, three-engine snapshot.
It keeps complete MoE, sparse expert compute, and separate policy probes distinct.

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

Cross-framework tables must identify each framework's measured version or source commit and execution path.
Group results by component and workload shape. Show each framework's latency and the PSI relative latency difference.

For speculative decoding, also record these values:

- Proposals
- Sampled tokens
- Accepted tokens and chunks
- Acceptance efficiency

A throughput change with a different deterministic acceptance trajectory is not a pure executor or kernel comparison.

A checked-in end-to-end reference is an observed result, not a threshold. The owning helper must record its complete
provenance. It must suppress throughput deltas when the current provenance or deterministic trajectory does not match
the reference.

### Main-only serving results, 2026-10-14

The user supplied these eight helper-summary results after the short-context GQA selection fix.
Local `main` at receipt was `7c5d4356e561a6f49a8f002b256756bbbaebd77c`.
That association does not replace the producing run's missing CONFIG record.
The session machine is Apple M3 Max with 40 GPU cores and 48 GiB unified memory.
The model names follow the helper cases: `Qwen3.6-35B-A3B-4bit` and `Qwen3.8-27B-4bit`.
Run count, variance, exact output hashes, OS/configuration records, and memory telemetry were not supplied with the summary.

```text
helper case: one model, one prompt, one output limit
    -> complete serving path with Spec disabled
    -> sampled output tokens
    -> observed decode tokens/s

This metric includes the serving decode lifecycle.
It is separate from precise GPU intervals and isolated component latency.
```

| Case | Prompt | Output limit | Output tokens | Decode tokens/s | Tokens/chunk |
| --- | --- | ---: | ---: | ---: | ---: |
| 35b_off | GSM8K | 256 | 256 | 96.146 | 1.000 |
| 35b_off | GSM8K | 1024 | 1024 | 94.874 | 1.000 |
| 35b_off | Chat | 256 | 256 | 97.135 | 1.000 |
| 35b_off | Chat | 1024 | 1024 | 94.985 | 1.000 |
| 27b_off | GSM8K | 256 | 256 | 22.730 | 1.000 |
| 27b_off | GSM8K | 1024 | 279 | 22.760 | 1.000 |
| 27b_off | Chat | 256 | 256 | 22.834 | 1.000 |
| 27b_off | Chat | 1024 | 1024 | 22.236 | 1.000 |

The observed Off throughput is about 95–97 tokens/s for 35B and 22–23 tokens/s for 27B.
The 27B GSM8K case at limit 1024 ended at 279 tokens, so the limit does not establish equal work.
This summary is an observed serving result. It is not a controlled kernel-speedup claim or a pass/fail threshold.
No new MTP, DSpark, DFlash2, or cross-engine result is implied.
The helper retains its separate 2026-09-19 comparison baseline.
The [component snapshot](component_performance.md) retains its original measurement identities and values.

## Generic GQA matrix

The standalone matrix keeps measurement, dataset control, policy fitting, and reporting in their existing benchmark owners:

```text
crates/inference-backend-metal/benches/gqa/
  matrix.rs                    Metal worker and CPU-reference measurement
  matrix/
    controller.rs              CLI, resume identity, durable results, and worker lifecycle
    controller_test.rs         host dataset and process-contract tests
    policy.rs                  benchmark policy candidates
    report.rs                  quality filtering, coverage, fitting, and exports
    report_test.rs              host report and calibration-contract tests
```

`gqa-sdpa-bench` is a standalone backend benchmark executable.
It owns matrix planning, serial measurements, SQLite resume, reports, and table generation.
The original `gqa_sdpa_matrix` custom bench remains the JSON measurement worker.
Both entry points use the same worker source.
`scripts/gqa_sdpa_matrix.py` only builds and starts the standalone executable.

Build the executable before the scan:

```sh
cargo build --release -p inference-backend-metal --example gqa-sdpa-bench
```

The executable is `target/release/examples/gqa-sdpa-bench`.
Measurement commands require a release build and the build checkout for source provenance.
The controller uses the macOS `/usr/bin/sqlite3` and `/usr/bin/shasum` tools.
It does not require Python or a new database library.

### Matrix and measurement contract

The runner uses nonzero BF16 Q, FP8 K/V, per-token causal masks, and reversed physical page IDs.
V has a positive offset. A zero-output kernel therefore fails the numerical gate even at long contexts.
It compares every measured output with SlicedKv output.
It also checks CPU attention for one head at the first, middle, and last Q tokens.
The focused backend test covers sliding and empty visibility plus split/direct parity.

| Input axis | Default values |
| --- | --- |
| History tokens | `0, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768` |
| Queries per request | `1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128, 256` |
| Request count | `1, 2, 4, 8` |
| Q/KV head ratio | `6, 8, 12` |
| Head dimension | `128, 256` |
| KV head count | `2, 8` |
| KV tokens per page | `16` |

`--history`, `--queries`, `--requests`, and `--ratios` select workload axes.
These axes can contain other nonnegative history lengths and positive counts.
`--dims`, `--kv-heads`, and `--pages` configure operator profiles.
The implemented cache profiles are `D128/P8,16` and `D256/P8,16,32`.
`D128/P32` is excluded because the backend does not implement it.

The full candidate product contains SlicedKv plus three Q-token/head assignments.
The assignments are QTokensPerHead, QHeadsPerToken, and QTokensAndHeads.
Token tiles are `1,2,4,8,16,32`.
Head extents are `1,ceil(ratio/2),ratio`.
KV staging lengths are `8,16,32,64,128`.
KV slice rules are `64,128,256,512,1024,2048,4096,8192,full`.
Direct output requires a full KV slice.
Slice rules that resolve to the same partition share one case key.
Distinct static row geometries remain separate candidates.

The default full product contains 6048 workload profiles and 11727168 resolved candidate tuples.
Resource filtering occurs after enumeration.
Queries and histories follow argument order.
The traversal order is query count, history length, then operator profile.
The runner visits all profiles at one history before it advances to the next history.
It visits every history before it advances to the next query count.
Other profile axes advance by increasing sums of their value indices.
Recommendation: Put the highest-priority query and history values first for a limited run.

The screen contains QHeadsPerToken and QTokensAndHeads with token tiles `1,2,4,8`.
It tests half/full head extents, staging lengths `16,64`, and slices `64,128,256,512,2048,full`.
Existing SlicedKv and QTokensPerHead kernel geometries use the same screen slice rules.
`--mode smoke` uses history `64`, query count `5`, two requests, and ratios `6,12`.
`--mode plan` enumerates the full product without GPU work.
With `--policy`, plan mode enumerates a focused comparison set instead.
`--mode policy --policy <path>` measures that focused set.
Frozen policies use schema 4 and criterion `shared-partitioned-grid-v2`.
Older policies must be regenerated from recorded measurements because single-query short-context routing changed.
Saved measurement keys retain their existing candidate and coefficient field tags.
The set contains the frozen route, full/same-slice SlicedKv, Q-token/head assignment controls, and an unsplit Map/Reduce control.
It pairs each comparison geometry with adjacent KV slices.
This pairing supports complete geometry/partition comparisons.
It contains at most 16 distinct candidates per workload.
The added comparisons retain the original canonical case keys.
Resume skips the original completed comparisons and measures only newly requested keys.
The controls depend on the workload and frozen rule. They do not depend on measured winners.
The plan retains the frozen coefficients and source document.
The shared criterion code belongs to the controller. Planning and reporting use that code.

The metric is amortized replay wall time in milliseconds.
The default replay contains eight ordered SDPA invocations.
The time includes submission and completion wait.
Compilation, allocation, recording, and numerical checks occur outside the measured interval.
Each round alternates baseline/candidate/baseline and candidate/baseline/candidate.
The fixed reference is full-slice SlicedKv.
The policy-loss reference uses the best stable compatible recorded candidate for each input workload.
The report retains compatible records outside the current candidate shortlist.
The existing-kernel comparison separately uses the best stable existing kernel geometry across measured KV slices.
Selected coverage describes only the current plan.
A complete screen or focused policy scan does not establish the full candidate optimum.
The forced candidates do not exercise production selection.
These measurements do not report isolated GPU stage time or model throughput.

### Run, stop, and resume

Run a plan without GPU work:

```sh
target/release/examples/gqa-sdpa-bench --mode plan --out target/perf/gqa-night
```

Recommendation: Start with this screen across all default history lengths:

```sh
caffeinate -i target/release/examples/gqa-sdpa-bench \
  --mode screen --hours 2 --queries 2 4 8 --kv-heads 4 \
  --out target/perf/gqa-balanced
```

Repeat the command to continue for another two hours.
The default `--hours 0` has no deadline.
`--limit` limits new measurements. Its default `0` has no limit.
A deadline or Ctrl-C finishes and commits the current case before it stops.
`status.json` distinguishes `complete`, `time_budget`, `measurement_limit`, `interrupted`, and `runner_error`.
A worker error retains committed cases and generates a report before the controller returns the error.
`complete` means the traversal finished. The report separately states whether every selected candidate was measured.

Continue with the full candidate product:

```sh
caffeinate -i target/release/examples/gqa-sdpa-bench \
  --mode full --hours 2 --queries 2 4 8 --kv-heads 4 \
  --out target/perf/gqa-balanced
```

Inspect a frozen policy against saved records without GPU work:

```sh
cp target/perf/gqa-balanced/policy.json target/perf/gqa-policy.json

target/release/examples/gqa-sdpa-bench \
  --mode plan --policy target/perf/gqa-policy.json \
  --history 8192 --queries 1 2 3 4 5 6 7 8 --kv-heads 2 4 8 \
  --out target/perf/gqa-night
```

The focused plan reports recorded candidate counts and retryable deferred candidates.
It does not create a worker or a Metal device.
Measure the remaining cases with the same axes and output directory:

```sh
caffeinate -i target/release/examples/gqa-sdpa-bench \
  --mode policy --policy target/perf/gqa-policy.json --hours 2 \
  --history 8192 --queries 1 2 3 4 5 6 7 8 --kv-heads 2 4 8 \
  --out target/perf/gqa-night
```

The frozen input must differ from the generated local and frozen policy exports in the output directory.
The runner rejects that collision before it replaces report artifacts.
The report evaluates the saved frozen snapshot alongside the local fit.
It exports the snapshot as `frozen-policy.json` and `frozen-policy.rs`.
It does not replace or refit the frozen input.
Report mode uses the saved snapshot when `--policy` is omitted.

The capacity report groups uniform workloads by operator geometry, query length, and request count.
It inspects context lengths from the query length through that profile's largest visible interval in the plan.
It evaluates both sides of geometry, general slice, slice-floor, adaptive slice, and decode-grid boundaries.
Within one geometry and slice interval, causal Map counts are nondecreasing in visibility.
The report therefore retains each geometry's peak active Map count, split count, reserved partial groups, and a witness.
A maximum-context endpoint alone can miss a peak before the policy selects a larger slice.
The analysis runs on the CPU. It adds no GPU measurements.
These uniform active bounds do not include production replay padding or prove ragged lifecycle allocations.
The plan endpoint is an analysis envelope. It is not a runtime context cap.
The report also groups the supplied uniform profiles by operator configuration.
Each group uses one shared Map allocation ceiling and the production `ReplayBucketPolicy`.
The report pads each geometry's Map bound before it multiplies by that geometry's token width.
Direct replay uses equal Map and Q-range extents and reserves no partial-state groups.
These envelopes cover the supplied query/request profiles. Their displayed bounds do not imply complete combination coverage.
Configured lifecycle limits and ragged routing still require separate integration.

Recorded measurements, failures, and hardware exclusions are skipped by canonical case key.
A missing record remains unmeasured.
An allocation above `--max-scratch-mib` is recorded as `deferred`.
Increasing that budget retries newly eligible deferred cases without repeating measured cases.
A physical mapping above the thread limit has `hardware_ineligible` status.
A compilation or numerical error has `failed` status.

Resume identity contains measurement source, GPU chip/core count, memory, OS, execution environment, `inner`, and `repeats`.
Axis subsets, mode, time budget, and report settings do not change measurement identity.
The binary refuses changed measurement identity and refuses measurement source newer than the executable.
Use a new output directory after a kernel or measurement change.
Old datasets require a verified measurement-source migration record before they can resume through the new controller.
Report mode can still read an old dataset without migration or current hardware checks.

Run Metal and GPU commands one at a time.
A checkout GPU lock prevents concurrent matrix runners in that checkout.
A dataset lock prevents concurrent writers of the same report set.
These locks do not control inference services or other GPU tools.

### Reports and direct lookup tables

Regenerate reports without GPU work:

```sh
target/release/examples/gqa-sdpa-bench \
  --mode report --out target/perf/gqa-balanced \
  --max-regret 0.05 --max-noise 0.10
```

Report mode uses the saved plan and measurement configuration.
It does not create a Metal device.

| Artifact | Contract |
| --- | --- |
| `plan.json` | Selected axes, mode, resource budget, metric, and optional frozen policy snapshot. A focused plan also reports recorded candidate counts. |
| `results.sqlite3` | Durable case keys, status, commands, raw timing samples, and numerical evidence. |
| `identity.json` | Measurement identity. A migrated dataset preserves its original identity. |
| `controller-identity.json` | Validated controller identity for a migrated dataset. |
| `measurement-identity.json` | Migration evidence that matches the original source fingerprint. |
| `runs.jsonl` | Commit, dirty state, executable hash, compiler, device, power, command, and environment. |
| `status.json` | Stop reason, new measurement count, skipped case count, and elapsed scan time. |
| `best.json` | Raw best measured candidate per selected workload. |
| `comparisons.json` | Raw winner, stable winner, and stable existing-kernel envelope per workload. |
| `report.md`, `report.json` | Selected coverage, numerical results, quality exclusions, and table size. |
| `selection.json`, `selection.rs` | Exact-grid diagnostic: candidate dictionaries and direct-index decisions. |
| `policy.json`, `policy.rs` | Local fitted lookup: 24 geometry strategy slots, general KV slice thresholds, and the single-query partition budgets. |
| `frozen-policy.json`, `frozen-policy.rs` | Evaluated frozen coefficients with the dataset identity. The lookup returns `None` when no frozen input exists. |
| `compression.json` | Exploratory dictionary sizes under per-point relative or absolute loss bounds. |
| `policy-evaluation.json` | Local calibration, context holdouts, and optional frozen-policy evaluation with timing and reference coverage. |
| `policy-capacity.json` | Continuous-context active bounds and shared uniform replay envelopes for local and frozen policies. Includes per-geometry padding, peak witnesses, and unavailable routes. |

The quality filter checks baseline drift, candidate bracket drift, and normalized sample range.
`--max-noise` defaults to `0.10` for each check.
Raw samples and excluded winners remain in the result set.

Each table profile contains actual Q heads, KV heads, head dimension, and page length.
Its calibration identity contains the hardware and operator contract.
The profile does not use a model name or a head ratio alone.
The operator contract is BF16 Q/output, FP8 E4M3 cache, and uniform per-request causal history.

The generator reuses a candidate across points only when each point satisfies `--max-regret`.
The default `0.05` permits at most five percent observed relative-time loss from that point's best stable candidate.
`comparisons.json` records candidate coverage and scan completeness for each measured workload in the current mode.
The reference includes all compatible recorded candidates for each input workload.
Scan completeness covers only the selected mode. A focused scan does not identify the full candidate optimum.
`compression.json` compares relative bounds and absolute-time allowances.
It counts kernel geometries with output mode and KV slice parameters separately.
Its geometry cover can choose a different measured KV slice for each point.
This ablation does not define a runtime policy or prove the minimum geometry count.
An absolute allowance dominates the relative bound below `absolute_bound / relative_bound` reference time.
Use the reference-time buckets to check its effect on expensive points.
Estimated extra time equals normalized relative loss multiplied by the best stable candidate wall time.
Each alternative reports maximum loss for reference-time buckets at `0,0.05,0.2,1,2` ms.
These overlapping buckets expose expensive-point regressions separately from tiny-point tradeoffs.
They describe measurements. They do not define a production performance policy.
It is not a new timing measurement or a model-level latency result.
The greedy alternatives do not prove the minimum dictionary size.
They do not change `selection.json`.

`policy-evaluation.json` evaluates four geometry families and integer KV slice thresholds.
The generic geometry criteria use a Q-token/head dimension budget and a cooperative KV-loader allowance.
The Q-token/head capacity is `q_token_head_values / head_dimension`.
QTokensAndHeads uses `ceil(token_tile * head_extent / 8)` SIMDgroups.
QHeadsPerToken uses `token_tile * ceil(head_extent / 8)` SIMDgroups.
QHeadsPerToken must add SIMDgroups before the criterion spends padded Q-token/head positions to reduce cooperative KV-loader work.
The loader-work proxy is `ceil(KV_stage / QTokensAndHeads_SIMDgroups)`.
These coefficients are empirical proxies. They are not compiler register limits or measured residency.

The report fits nine pairs from `q_token_head_values=8192,16384,32768` and `loader_tokens=8,16,32`.
Each fit compares complete selected routes with stable references.
The training set contains only inputs with stable measured selected routes for all nine parameter pairs.
The fit must compare every pair on the same input set.
Missingness must not act as a timing signal.
The report retains the available, common, and excluded reference counts.
It also counts training inputs where the parameter sets select different complete routes.
A stage with no route differences retains its prior coefficients.
The fit ranks worst relative loss and worst estimated extra time at `>=2`, `>=1`, then `>=0.2` ms.
The fit does not use winner labels or a partition oracle.
The second stage compares 195 base-grid and partition parameter sets.
The base grid is `requests * KV_heads * ceil(queries / token_tile) * ceil(head_ratio / head_extent)`.
A small base grid can select padded QHeadsPerToken or impose a minimum KV slice.
The floor feature is `visible_KV_tokens * base_grid * SIMDgroups_per_threadblock`.
Each parameter set must have stable selected timings at every common training point.
The second stage must not worsen the first-stage maximum relative loss or maximum estimated extra time in any cost tier.
This guard prevents a gain at slow inputs from hiding a large regression at medium-cost inputs.
The third stage compares the fixed single-query prior with six shared grid policies.
It fits a minimum partitioned-grid budget and a partition target.
The tested budgets are `2,4,8` and `16,32` times the calibration device compute-unit count.
The export stores the two calibrated budgets. Runtime lookup does not query hardware again.
The third stage must not worsen the preceding cost-tier bounds on its common stable support.
The report refits all three stages after withholding each history length.
The holdouts do not validate the prior kernel repertoire, D128/head-tile eligibility, slice envelope, or first two general slice thresholds.
The report returns no fitted policy without common stable references at `>=0.2` ms.
It also returns no fitted policy when no stage changes a complete route on its common training set.
Single-query measurements can identify the third-stage coefficients when the resource and general partition stages remain invariant.

The single-query strategy uses the existing SlicedKv and QTokensAndHeads implementations.
SlicedKv processes at most eight logical heads in one head tile.
Let `L` be visible KV tokens and `G` be the GQA ratio.
Let `target_groups` and `min_grid_groups` be the calibrated budgets.
The adaptive criterion uses these quantities:

```text
matrix_base_groups = requests * KV_heads * ceil(G / 8)
matrix_slice = clamp(floor_power_of_two(L * matrix_base_groups / target_groups), 256, 512)
matrix_grid = matrix_base_groups * ceil(L / matrix_slice)
SlicedKv_base_tasks = requests * KV_heads * ceil(G / min(G, 8))
SlicedKv_slice = clamp(floor_power_of_two(L * SlicedKv_base_tasks * 4 / target_groups), 256, 1024)
```

The slice calculation clamps its positive work input to at least one before rounding.
D256 groups with `G > 8` use QTokensAndHeads when `matrix_grid >= min_grid_groups`.
Other inputs retain SlicedKv.
QTokensAndHeads uses one query token, the complete logical head group, and KV stage 16.
The two paths share `target_groups`. The criterion does not fit separate slice budgets per path.
The short-KV direct-output limit is `L <= 32` for one Q token per request and `L <= 512` for multiple Q tokens.
Above the single-query limit, selection retains its KV partition budget.
These quantities are empirical parallelism proxies. They do not establish compiler residency.
Frozen context-transfer comparisons must verify complete-route loss and coverage before promotion.
The criterion uses operator dimensions. It does not use model names.

The schema-3 export records either an adaptive policy or the fixed prior.
An invariant third stage retains the fixed prior of a 512-token slice and an eight-task base-grid threshold.
The reader accepts schema-2 frozen exports and saved plans with their original fixed routing.
It must not reinterpret an old frozen artifact as the adaptive policy.
Unsupported schemas, invalid budgets, and different calibration devices return an input error.

The short-KV alternatives use row capacities `16,32,64`.
The criteria use rounded query capacity, head ratio, head dimension, visible KV length, request count, KV-head count, and SIMDgroup count.
The integer feature replaces floating-point geometric slice rounding. It does not estimate kernel residency.
The family repertoire and slice thresholds use M3 Max 40-core calibration through 32K.
The report retains the saved identity of the evaluated dataset.
`report.json` and `policy-evaluation.json` also record the generator command and executable SHA-256.
It does not claim calibration quality on another device.

Use `--policy` to evaluate a frozen exported policy alongside the local fit:

```sh
target/release/examples/gqa-sdpa-bench \
  --mode report \
  --out target/perf/gqa-night \
  --policy target/perf/gqa-balanced-20261006/policy.json
```

This command does not create a Metal device or repeat measurements.
The frozen comparison does not refit coefficients, select a per-point winner, or replace the local export.
`policy-evaluation.json` retains the source path, source document, and complete evaluation.
`report.md` includes a separate frozen-policy cost table.
`frozen-policy.json` and `frozen-policy.rs` export exactly the evaluated frozen coefficients.
The exporter uses the same criterion implementation for the local and frozen policies.
Both exports record the evaluated dataset identity.
The source path and original source document remain in `policy-evaluation.json`.
The input must use schema 2 or 3, the matching decode criterion, the current strategy table, and the dataset's device.
The input coefficients must belong to the supported calibration domain.
Invalid or unavailable input policy data returns a typed error before report artifacts are replaced.
This check prevents an old criterion from being interpreted as the current criterion.

Each point retains its proposed complete candidate.
It also retains selected and reference timing summaries, including bracket drift and normalized sample range.
The report does not replace a missing or unstable candidate with a recorded winner.
Its reference-time buckets count verified and unverified selected timings separately.
Workloads without a stable reference have a separate count.
The proposals export `policy.json` and `policy.rs`. They do not change the exact-grid diagnostic or production routing.
They require independent coverage and expensive-point validation before promotion.

Each dictionary value contains layout, token tile, head extent, KV stage, KV slice, and direct-output mode.
A null KV slice means a full slice.

The decision grid stores one `u16` per input point.
Zero means the point has no eligible stable candidate.
Other values are one-based dictionary indices.
The generator verifies the measured loss bound and one decision per covered point.
The bound describes these measurements. It is not a statistical confidence interval.

The exact-grid diagnostic lookup has three steps:

1. Match hardware and static operator profile.
2. Find exact history, query, and request indices in the recorded axes.
3. Read `decisions[(history_index * query_count + query_index) * request_count + request_index]` and its dictionary value.

The exact-grid lookup returns `None` for unmeasured shapes, excluded points, and different hardware.
It does not match continuous intervals or run a cost model.
The factored export has a different contract:

1. Match hardware and validate operator dimensions. The caller must establish uniform BF16/FP8 causal inputs.
2. Index one of four static profile rows and six rounded-query columns.
3. Derive the selected row geometry from shared resource and base-grid criteria.
4. Read the general KV slice from integer threshold tables, or derive the single-query slice from its shared partition budget.

The strategy slots contain four shared strategy values. They are not 24 independently fitted winners.
The exporter embeds the same private criterion implementation that the report evaluates.
It performs no runtime search, cost-model evaluation, or per-point winner lookup.

The generated artifact also exports `lookup_batch`.
Each request shape is `(history_tokens, query_tokens)`. Query counts must be positive.
The entry point validates the same shader and host-allocation domains as uniform lookup.
Geometry uses the longest query and maximum visible KV endpoint.
The base grid uses `sum_requests(ceil(query_tokens / token_tile))`.
Adaptive single-query partitioning uses the sum of request endpoints.
Its reuse grid uses `sum_requests(ceil(visible_KV_tokens / KV_slice))`.
Uniform lookup retains constant-time calculations and selects the same routes.
Batch lookup extends the uniform calibration. Existing performance reports evaluate uniform inputs only.
Ragged numerical, performance, and configured lifecycle validation remain required before production promotion.
The factored export accepts legal intermediate contexts. It does not clamp them to the measured grid.
It returns `None` for unavailable calibration, unsupported operator inputs, or different hardware.
A completed report must replace a prior export when fitting becomes unavailable.
Off-grid selection is not numerical, scratch, replay, or performance validation.
Performance evidence remains limited to stable measured selected routes.

The generated table is a calibration artifact.
It does not change the production registry, selector, replay capacity, or scratch lifecycle.
See [future work](future_work.md#performance-investigations) for production promotion.

## Cross-engine serving matrix

`scripts/qwen35_e2e_matrix.py` compares a fixed raw-token workload across psi-dec, oMLX, and TensorFold.
It complements the representative text prompts in `qwen35_e2e_decode_perf.sh`.
It does not replace their output-quality checks.

The default matrix has 232 shapes:

- Request counts: `1,2,4,8`.
- Retained context: `0,2048,4096,8192,16384,32768`.
- Appended input tokens per request: `1,2,4,8,16,32,64,128,256`.
- From-empty input tokens per request: `2048,4096,8192,16384`.

Each request generates exactly 256 tokens by default. Use `--output-tokens` to change this fixed count.
The default prefill chunk is 128 tokens per request.
Append cases use one warm-up and two measured samples by default.
Full-prefill cases run after the append matrix has exercised the operator paths.
They use no additional full-prompt warm-up and one measured sample by default.
Use `--full-repeats` for more full-prompt samples.
The modes are Main-only, MTP proposal counts 1 through 4, and DFlash2 with seven proposals.
The oMLX worker uses the official Lightning MTP path with `mtp_fixed_depth`.
Its metadata view binds the supplied standalone head as `language_model.mtp.*`.
Native MTP uses `model-native-mtp-no-eos`. Other modes use `model-no-eos`.
The view retains the original shard files and the head's quantization metadata.
The loader renames parameter keys. It does not requantize or copy checkpoint values.
Initialization rejects a legacy norm repair that would change the shared head weights.
The native path supports concurrent requests. Earlier external-drafter fallback results remain separate.
The adapter disables native context-copy proposals. A copied window can exceed `mtp_fixed_depth`.
It records every MTP verify width and accepted count at the existing chain owner.
The native summary log counts deeper positions only until the first rejection.
That conditional denominator does not count all generated proposals.
A measured MTP sample without proposal statistics is also excluded as `N/A: MTP fallback`.
The TensorFold Qwen dense family has no MTP mode. The report marks those entries `N/A`.

The controller starts one engine at a time. It initializes each request capacity separately.
The psi-dec worker calls the production raw-token gRPC API from a separate process.
The oMLX worker uses the production VLM serving engine and its in-memory prefix cache.
The TensorFold worker uses the maintained Python `0.6` LaneEngine path.
The native TensorFold `1.0.4` server has a different fixed tree policy and no raw-token completion API.
Do not label the Python fixed-chain results as native-server performance.

Use the same local Main, MTP, and affine DFlash2 checkpoint paths for all workers.
The workers do not download checkpoints or compare weight values.
The TensorFold adapter reads the affine metadata before loading the existing draft.
It expands quantized selector embeddings for TensorFold's CPU selector.
It retains the production draft kernels and fixes the maximum chain at seven candidates.
It records the actual generated and accepted candidate counts.
A shared TensorFold forward retains its published row capacity.
The fixed-chain request cohort is limited to `family.batch_rows // 8` streams.
It disables adaptive chain allocation and selects the full vocabulary through `TF_DRAFT_VOCAB=full`.
The oMLX DFlash2 adapter records actual verify widths through the existing chain owner.
Its conditional log denominator is not the complete generated-proposal count.

The benchmark creates a local metadata view with EOS stopping disabled.
All weight and tokenizer files remain linked to their original files.
Greedy sampling uses temperature 0 and seed 42.
Synthetic tokens establish input identity and shape. Their acceptance rate does not describe natural conversation.
Default KV precision remains engine-specific: psi-dec uses FP8; the MLX engines use BF16.
Different numerical paths can produce different greedy output hashes.
The report retains the hashes and fixed output counts.

Prefix setup is outside timing. Each measured request must resume at the exact requested context.
The default fixture has a distinct prefix for each request.
`--shared-prefix` prepares one identical prefix for all requests.
The suffix after the four fixed lookahead positions remains distinct for each request.
The psi-dec and oMLX workers use their native prefix caches to share the prepared prefix.
TensorFold shares immutable baseline arrays and uses `LaneEngine.copy_single_cache` for each working cache.
The TensorFold worker extends a matching baseline as context grows.
Changed or shorter prefixes require fresh preparation.
A shorter GDN prefix requires its recurrent state at that position. Truncating a longer KV cache is insufficient.
Recommendation: List contexts in increasing order to reuse incremental preparation.
Shared and distinct prefixes are different workloads. A changed prefix policy requires a new output directory.
MLX prefix setup uses `--prefix-chunk` (default 2048). Timed input uses `--chunk` (default 128).
The psi-dec server uses `--chunk` for setup and timing because its request capacity is static.
The setup chunk and cache budgets are part of provenance.
Both MLX engines limit their allocator's retained free buffers with `--mlx-cache-gib` (default 1).
This limit does not change active model tensors or the retained context.
It prevents unused GPU buffers from competing with oMLX's CPU hot-cache data for unified memory.
The allocator cache limit is part of provenance. A changed limit requires a new output directory.
The default oMLX hot-cache budget is 20 GiB. The psi-dec page budget is 98304 pages per request.
The psi-dec server allocates and zeros the full page arena during initialization.
Shared prefixes reduce occupied pages. They do not reduce that configured allocation.
Recommendation: Size `--cache-pages-per-request` for the unique retained blocks and concurrent mutable tails.
The page budget must include Main and shifted MTP lanes, GDN snapshots, and pending output blocks.
Full-prefill inputs remain distinct for each request. The shared-prefix policy does not reduce their required page budget.
Up to four suffix tokens remain fixed as MTP lookahead.
The remaining suffix tokens vary before they can publish another complete cache block.
The psi-dec MTP depth `K` needs `K` exact suffix tokens as lookahead during untimed prefix setup.
A retained-context shape with fewer than `K` input tokens cannot reuse a complete shifted cache column.
At zero context, the runtime also requires at least `K` initial input tokens.
The report marks it `N/A: MTP lookahead`. It must not time history recomputation as a hot-prefix result.
The script verifies psi-dec's first submitted token index and oMLX's reported cached-token count.
The reported cached-token count is authoritative.
A sample with fewer cached tokens is recorded as `N/A: prefix not retained`.
Its excluded timing remains in JSONL. It must not produce a hot-prefix comparison ratio.
TensorFold preserves its prepared cache as an immutable baseline.
Each sample copies its cache containers through the native copy API.
The native API can share contiguous immutable arrays. It materializes array views when required.

TTFT ends when every request has its first token. Decode ends when every request finishes.
The TensorFold adapter observes `LaneStream.emitted` after each engine step.
It records the first token time when that list first becomes nonempty.
An early request can decode before the last request has its first token.
For concurrent requests, decode latency measures the remaining batch phase, not a separate kernel interval.
Total includes request dispatch, prefix restoration, model execution, sampling, and completion delivery.
The psi-dec transport is gRPC. The MLX adapters enter their serving engines in process.
Record this transport difference when interpreting small latency differences.
These are ordinary wall-clock timings. They are not force-sync GPU profile timings.

Install the engine environments separately. The psi-dec worker also needs `grpcio` and `grpcio-tools`.
Build the production release launcher before timing:

```sh
cargo build --release -p inference-runtime-launcher --bin qwen3_5_dense
python3 scripts/qwen35_e2e_matrix.py \
  --engine all \
  --omlx-python /path/to/omlx-env/bin/python \
  --tensorfold-python /path/to/tensorfold-env/bin/python \
  --out target/perf/qwen35-e2e-matrix
```

`results.jsonl` stores engine, mode, and shape keys. Resume uses the latest result for each key.
It skips valid completed results and fixed capability exclusions. It retries OOM and swap-invalidated results.
Each worker checks its source, script, checkpoint metadata, version, and timing configuration before appending.
GPU provenance records the GPU model and core count. Display state and resolution do not change the benchmark identity.
A Metal out-of-memory failure records the affected keys as `N/A: Metal OOM`.
The psi-dec adapter classifies an RPC failure as Metal OOM only when the new server log segment contains `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
The worker closes its GPU owner and returns exit code 86.
The controller stops. It does not retry OOM automatically.
Correct the memory budget before an explicit resume.
Each GPU worker runs under an external supervisor.
The supervisor samples macOS `Swapins` and `Swapouts` counters every 0.5 seconds.
Existing swap usage alone does not reject a run. Any new swap-in or swap-out stops the worker and controller.
The supervisor records counter observations in `<engine>-<mode>-memory.jsonl`.
It preserves raw results and excludes newly completed rows in the affected observation interval through `timing-exclusions.jsonl`.
The reports label those timings `unusable: swap activity` and omit their comparison ratios.
Each new result has a durable ledger offset. A timing exclusion applies to that result, not to a later clean retry.
Legacy key-only exclusions apply to legacy results that have no ledger offset.
An interrupted, unrecorded shape remains pending. Earlier clean observations remain available on resume.
Use a new output directory when those facts change.
`report.md` groups rows by mode, operation, request count, context, and input length.
`report.md`, `ttft.md`, and `decode.md` contain separate total, TTFT, and decode tables.
Each table shows all engine values and psi-dec latency differences.
Results with different input hashes must not produce a comparison ratio.

Use `--engine plan` to inspect the matrix without loading a model.
Use `--engine report` to regenerate the table without GPU work.
Use `--requests`, `--contexts`, `--tokens`, `--full`, and `--modes` for a subset.
Use `--engine psi-dec --mode mtp2` to run one worker with its engine Python environment.
Run `python3 scripts/qwen35_e2e_matrix_test.py` for CPU-only workload-identity checks.
