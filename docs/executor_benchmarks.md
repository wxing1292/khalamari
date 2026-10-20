# Executor Tests, Benchmarks, and Profiling

This document owns the executor verification ladder, benchmark surfaces, metrics, profiling conventions, and performance
evidence requirements. Component math and source layouts remain in the matching `executor_*.md`.
[`service.md`](service.md) contains the service lifecycle and end-to-end commands.

## Qwen4x correctness

Run `bash scripts/qwen4x_validate.sh` for the serial fixed/random CPU-reference and GPU lifecycle suite.
The script saves complete logs, commit, dirty state, and hardware under `target/validation/qwen4x`.
An optional checkpoint argument captures the first eight or all Main layers.
The capture checks finite output and does not establish cross-framework numerical parity.
See [Qwen4x verification](executor_qwen4x.md#verification).
This script does not measure performance.

## Verification ladder

Select the smallest production owner that proves the changed contract:

```text
CPU/slow reference -> backend component -> production component + metadata
  -> real-weight layer -> layer range -> Main + final norm
  -> embedding + unembedding + sampling -> Spec proposal + rejection + commit
```

Establish correctness and workload identity before timing.
Component gains do not prove end-to-end gains.
Run GPU commands one at a time across processes and worktrees.
Use `--test-threads=1` for Metal tests. Keep expected-panic contract tests on the host.

## Benchmark layers

| Layer | Input | Purpose |
| --- | --- | --- |
| `inference-backend-metal` | Synthetic tensors | Model-independent kernels and components |
| `inference-executor-metal` | Real weights and production owners | Component, layer, and executor replay |
| Service | External requests | Complete runtime and transport lifecycle |

Find registered targets in each crate's `Cargo.toml`.
Executor targets share `--iters`, `--warmup-iters`, and `--runs`.
`qwen35_main_sampling` uses synthetic logits and does not load weights.

| Specialized measurement | Owner |
| --- | --- |
| H1/H4 lookup and hash+lookup, 32 invocations per replay | [N-gram embedding](executor_ngram_embedding.md#verification-and-benchmark) |
| Production replay versus cross-framework scorer study | [QSA indexer](executor_qsa_indexer.md#benchmarks) and [result snapshot](qsa_indexer_performance.json) |
| DFlash2 combined Main-and-Spec lifecycle | [Service](service.md), `decode` client, `main_spec_replay_elapsed` |

Compare baseline and candidate binaries with identical models, batches, seeds, and sampler settings.
Keep CPU preparation/read time, replay-cache state, and combined Main-and-Spec wall time distinct.
Production `src` must not gain benchmark-only state or switches.

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

| Target | Timed path | Contract or options |
| --- | --- | --- |
| `gated_residual` | GR read/write, composition, fused/unfused chains | [GR](executor_gated_residual.md#verification-and-measurement) |
| `moe` / `gdn_attn` replay128 | 128 invocations per submission | Diagnostic submission amortization, separate from full forward |
| `norm` `rms-only/replay64` | 64 BF16 RMSNorm commands per replay | Kernel comparison with amortized submission overhead |
| `affine_quantized_matmul` | Automatic and forced QMV/QMM | [Dense MLP](executor_dense_mlp.md) |
| `matmul_bf16` | Production GEMV and Steel GEMM | Qwen3-ASR matrix shapes |
| `buffer_io` | Uncached file/buffer I/O at `4`, `64`, `128 MiB` | Read returns or write `sync_all` completes |
| `gqa_bidi_block_sdpa` | Bidirectional local-block SDPA | Custom CLI, `--max-q-tokens 8` by default |
| `qwen3_gqa` | Ungated GQA replay and projection probes | Real Qwen3 weights, optional SlicedKv/SlicedQ validation |
| `qwen35_gqa` | Full GQA replay or selected subcomponents | `--gqa-model 27b\|35b`, `--max-tokens 128` |
| `qwen35_gdn` | Chunkwise/replay forward | Excludes accepted-state commit and publication |
| `qwen35_moe` | Token-major/expert-major forward | Real sparse weights |
| `qwen35_main_layers` | Selected `Qwen35MainLayer` owners | `layer0`, `layer3`, `layer4`, `first4`, `all_layers`, `layersSTART-END` |
| `qwen35_main` | All transformer layers and final norm | Excludes embedding, unembedding, sampling, and readback |
| `qwen35_main_text_embed` | Production text embedding | `--tokens`, `--max-tokens` |
| `qwen35_main_gather_unembed` | Gather plus unembedding | `--rows`, `--max-rows` |
| `qwen35_main_sampling` | `sample` or `sample_readback` | Excludes draft distributions and rejection |
| `qwen35_vanilla_prefill_decode` | Committed Prefill/Decode lifecycle | Real context through `ReplayableDecoderModel` |

`layersSTART-END` is the half-open range `[START, END)`.
`all_layers` excludes final norm. For 27B, `layer3` is the first GQA layer.
Main and Main-layer targets use synthetic hidden inputs and initialized metadata.
Use the Vanilla target when the measurement requires committed context.

For `buffer_io`, set `PSI_DEC_BUFFER_IO_BENCH_DIR` to select the volume.
The default is the operating-system temporary directory.
Setup validates a repeated 1 MiB pattern before timing.
Reads use `F_GLOBAL_NOCACHE`. Writes use `F_NOCACHE` and `sync_all` each iteration.
File setup, metadata, rename, directory sync, validation, and cleanup are outside timing.
These flags do not bypass SSD controller caches.

### Vanilla lifecycle

```text
reset slot -> build committed context in chunks                 untimed
  -> prepare -> Main -> optional sampling -> finish -> commit   wall timer
  -> deterministic output fingerprint + repeat/chunk checks     untimed
```

| Option or result | Meaning |
| --- | --- |
| `--cases prefill,decode` | Default cases |
| `--contexts 0,1024,4096,8192` | Default starting contexts |
| `--prefill-tokens 64,128` / `--decode-tokens 32` | Default per-request operation totals |
| `--num-reqs 1` | Default request count |
| `--max-tokens 128` / `--max-tokens-per-request 128` | Default batch/request capacities |
| `--num-tokens-per-block 2048` / `--num-cache-pages 393216` | Default block/page capacities |
| `--warmup-iters 2 --iters 5 --runs 5` | Default timing controls |
| `--reuse-context` | Prefill only, retained block-aligned prefix, restoration inside timing |
| `PSI_PREFILL_SKIP` | Benchmark-only completed `context:tokens` pairs |
| `matrix_result` | Retained-context cold/warm wall, submission, and preparation times |

The active chunk per request is `min(max_tokens_per_request, max_tokens / num_reqs)`.
Prefill skips gather, unembedding, sampling, and readback.
Decode commits one sampled token per request and feeds it into the next step.
Main KV and GDN pages use one page-ID domain.

The sampling defaults are seed 42, temperature 0.7, Top-K 20, and Top-P 0.8.
Repeated trajectories require exact token and probability bits.
Chunk checks use greedy untimed probes with `temperature=0` and `top_p=0` and require exact next-token identity.
Each result includes input/output FNV-1a fingerprints and separates replay-cache-cold time from steady-state time.

Run a small committed-context check:

```sh
cargo bench -p inference-executor-metal --bench qwen35_vanilla_prefill_decode -- \
  --model-dir <model-dir> --cases prefill,decode \
  --contexts 7 --prefill-tokens 2 --decode-tokens 2 \
  --num-reqs 1 --max-tokens 2 --max-tokens-per-request 2 \
  --num-tokens-per-block 2048 --num-cache-pages 16384 \
  --iters 1 --warmup-iters 1 --runs 1
```

List cases before a performance run.
On an active desktop, alternate baseline/candidate measurements and compare stable paired latency ratios.

## Metrics

`setup_us` includes model loading and fixture construction.
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

| Measurement | Contract |
| --- | --- |
| `--profile component\|operation` | Same coarse CPU tree, static `profiling::span(...)` names |
| Dynamic request/shape values | Structured logs |
| Relaxed Metal 4 GPU timestamps | Enabled by default, can be unavailable |
| Precise Stream timestamps | Explicit diagnostics, can change performance |
| ICB stage timing | Must agree with completion feedback, monotonic encoder values alone are insufficient |
| `executor_cpu_ms` | Complete CPU-observed lifecycle, including integrated Spec work |

The M3 Max audit found unavailable relaxed samples and implausibly short precise Main samples.
Check stage boundaries against command-buffer completion times.
Keep diagnostic synchronization cost separate from normal throughput.
See [service timing fields](service.md).

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
No new MTP, DFlash2, or cross-engine result is implied.
The helper retains its separate 2026-09-19 comparison baseline.
The [component snapshot](component_performance.md) retains its original measurement identities and values.

## Generic GQA matrix

`gqa-sdpa-bench` plans serial measurements, resumes a SQLite dataset, and exports reports and lookup policies.
`gqa_sdpa_matrix` remains its JSON worker. Both use `benches/gqa/matrix.rs`.
`scripts/gqa_sdpa_matrix.py` builds and starts the standalone executable.

```text
axes -> canonical candidate keys -> numerical gate -> paired replay timing
  -> results.sqlite3 -> stable coverage -> report + diagnostic/fitted exports
```

### Matrix and measurement contract

| Axis | Default values |
| --- | --- |
| History tokens | `0, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768` |
| Queries/request | `1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128, 256` |
| Requests | `1, 2, 4, 8` |
| Q/KV head ratio | `6, 8, 12` |
| Head dimension | `128, 256` |
| KV heads | `2, 8` |
| KV tokens/page | `16` |

`--history`, `--queries`, `--requests`, `--ratios`, `--dims`, `--kv-heads`, and `--pages` select axes.
Supported cache profiles are `D128/P8,16` and `D256/P8,16,32`.
Traversal follows query count, history, then operator profile.
Recommendation: Put priority query/history values first for a limited run.

| Boundary | Contract |
| --- | --- |
| Input | Nonzero BF16 Q, FP8 K/V, causal masks, reversed page IDs, positive-offset V |
| Numerical gate | Every output versus SlicedKv, CPU attention at first/middle/last Q for one head |
| Metric | Replay submission-and-wait wall milliseconds, eight ordered invocations by default |
| Excluded | Compilation, allocation, recording, numerical checks |
| Pairing | Baseline/candidate/baseline alternates with candidate/baseline/candidate |
| Fixed reference | Full-slice SlicedKv |
| Policy-loss reference | Best stable compatible recorded candidate for that workload |

A screen or focused policy scan does not establish the full candidate optimum.
Forced candidates do not exercise production selection or measure model throughput.

### Run, stop, and resume

Build and plan without GPU work:

```sh
cargo build --release -p inference-backend-metal --example gqa-sdpa-bench
target/release/examples/gqa-sdpa-bench --mode plan --out target/perf/gqa-night
```

Recommendation: Start with a limited screen:

```sh
caffeinate -i target/release/examples/gqa-sdpa-bench \
  --mode screen --hours 2 --queries 2 4 8 --kv-heads 4 \
  --out target/perf/gqa-balanced
```

Repeat the command to resume. Use `--mode full` for the full candidate product.
Use `--mode smoke` for history `64`, query count `5`, two requests, and ratios `6,12`.
`--hours 0` and `--limit 0` remove the deadline and new-measurement limit.
A deadline or Ctrl-C completes and commits the current case before stopping.

| State | Resume behavior |
| --- | --- |
| Measured, failed, hardware-ineligible | Skip canonical key |
| Unmeasured | Measure |
| `deferred` by `--max-scratch-mib` | Retry when the budget permits |
| Changed measurement identity or newer measurement source | Reject, use a new directory or verified source migration |

Identity includes measurement source, chip/core count, memory, OS, environment, `inner`, and `repeats`.
Axis subsets, mode, time budget, and report options do not change it.
A checkout GPU lock and dataset writer lock do not control other GPU tools.
Measurement needs a release executable, its build checkout, `/usr/bin/sqlite3`, and `/usr/bin/shasum`.

Copy a policy outside its generated output paths before focused measurement:

```sh
cp target/perf/gqa-balanced/policy.json target/perf/gqa-policy.json
caffeinate -i target/release/examples/gqa-sdpa-bench \
  --mode policy --policy target/perf/gqa-policy.json --hours 2 \
  --history 8192 --queries 1 2 3 4 5 6 7 8 --kv-heads 2 4 8 \
  --out target/perf/gqa-night
```

`--mode plan --policy <path>` inspects this focused set without Metal.
The frozen input must use schema 4 and criterion `shared-partitioned-grid-v2`.
It must match the operator, strategies, and calibration device.
Older frozen policies require regeneration from saved measurements.
Report mode can read old datasets without current-hardware or measurement-source migration checks.

### Reports and direct lookup tables

Regenerate without GPU work:

```sh
target/release/examples/gqa-sdpa-bench \
  --mode report --out target/perf/gqa-balanced \
  --max-regret 0.05 --max-noise 0.10
```

| Artifact | Purpose |
| --- | --- |
| `plan.json`, `identity.json`, `runs.jsonl`, `status.json` | Plan, measurement identity, run provenance, stop reason |
| `results.sqlite3` | Durable keys, statuses, raw timings, numerical evidence |
| `best.json`, `comparisons.json` | Raw/stable winners, existing-kernel envelope, coverage |
| `report.md`, `report.json` | Coverage, numerical results, quality exclusions |
| `selection.json`, `selection.rs` | `u16` exact-grid decisions, zero unknown, positive values index the dictionary from one |
| `policy.json`, `policy.rs` | Locally fitted geometry/slice policy, schema 4 |
| `frozen-policy.json`, `frozen-policy.rs` | Evaluated frozen coefficients, without refitting |
| `policy-evaluation.json` | Calibration, holdouts, frozen evaluation, source identity |
| `policy-capacity.json` | Continuous-context active bounds and uniform replay envelopes |
| `compression.json` | Exploratory dictionary-size/loss tradeoffs |
| `controller-identity.json`, `measurement-identity.json` | Verified dataset-migration evidence |

`--max-noise 0.10` filters baseline drift, bracket drift, and normalized sample range.
`--max-regret 0.05` bounds observed relative-time loss against each point's best stable reference.
These bounds describe measurements, not statistical confidence or production guarantees.
Excluded and raw results remain available.

```text
exact-grid lookup: hardware/profile -> exact H/Q/B indices -> u16 decision -> candidate
fitted lookup:     hardware/profile -> geometry + slice criteria -> candidate
ragged lookup:     request (history, query) pairs -> shared geometry + partition bounds
```

The exact grid returns `None` for unmeasured shapes and different hardware.
The fitted export supports intermediate contexts and returns `None` for unavailable calibration, unsupported inputs, or different hardware.
Off-grid selection is not numerical or lifecycle validation.
Calibration uses uniform BF16/FP8 causal workloads on M3 Max through 32K.
Uniform capacity envelopes exclude ragged allocation and production replay padding guarantees.
Exports do not modify production routing. See [production selection](gqa_sdpa_selection.md) and [active validation](future_work.md#performance-investigations).

The implementation owners are `matrix/controller.rs`, `matrix/policy.rs`, and `matrix/report.rs`.
Their host tests cover process, report, policy, and calibration contracts.

## Cross-engine serving matrix

`scripts/qwen35_e2e_matrix.py` compares fixed raw-token work across psi-dec, oMLX, and TensorFold.
Use [service text-prompt checks](service.md#end-to-end-performance-helper) for output quality.

| Axis or control | Default |
| --- | --- |
| Requests | `1,2,4,8` |
| Retained context | `0,2048,4096,8192,16384,32768` |
| Appended tokens/request | `1,2,4,8,16,32,64,128,256` |
| From-empty tokens/request | `2048,4096,8192,16384` |
| Output tokens/request | Exactly 256, `--output-tokens` |
| Timed Prefill chunk | 128/request, `--chunk` |
| Append warmup / measurements | 1 / 2 |
| Full-Prefill measurements | 1, `--full-repeats` |
| Modes | Off, MTP1 through MTP4, DFlash2 K7 |
| Sampling | Greedy, temperature 0, seed 42 |
| Stopping | Local metadata view disables EOS, every request emits the fixed count |

The matrix has 232 shapes. Full-Prefill cases follow append cases without an additional full-prompt warmup.

```text
example: two requests, retained prefix C=2048, new T=8, output=256/request
prefix token IDs -> committed KV/GDN state                          untimed

time        t0          tA          tB          tC          tD
request 0   start------>first------decode------>done
request 1   start------------------>first------decode------>done
TTFT        |-----------------------|
Decode                              |-----------------------|
total       |-----------------------------------------------|

start includes dispatch, prefix restoration, and Prefill; first/done are tokens 1/256
TTFT = tB-t0, Decode = tD-tB, total = tD-t0
request 0 can decode before request 1 emits token 1
```

TTFT ends when every request has its first token. Decode ends when every request finishes.
Concurrent decode latency covers the remaining batch phase.

| Engine | Path and comparison limits |
| --- | --- |
| psi-dec | Production raw-token gRPC from a separate process, FP8 KV |
| oMLX | Production VLM engine in process, BF16 KV, native fixed-depth Lightning MTP |
| TensorFold | Python `0.6` LaneEngine in process, BF16 KV, fixed-chain DFlash2 |

TensorFold native `1.0.4` has a different tree policy and no raw-token completion API.
Dense TensorFold MTP is N/A.
MTP measurements need actual proposal statistics. Fallback runs are excluded.
DFlash2 uses the same affine checkpoint, full draft vocabulary, and seven-candidate maximum.
Workers use the supplied checkpoint files and metadata views without requantizing weights.
They do not compare weight payload values.

| Adapter | Workload constraint |
| --- | --- |
| oMLX MTP | Standalone head bound as `language_model.mtp.*`, `mtp_fixed_depth`, context-copy proposals disabled |
| TensorFold DFlash2 | CPU selector expands affine codebooks before timing, `TF_DRAFT_VOCAB=full`, fixed chain, at most `family.batch_rows // 8` streams |
Numerical paths can produce different greedy hashes. Synthetic acceptance does not describe natural conversation.

| Prefix or memory contract | Behavior |
| --- | --- |
| Prefix identity | Distinct/request, or `--shared-prefix` with distinct timed suffixes |
| Cache validation | Exact first PSI token index and authoritative oMLX cached-token count |
| Missing retained prefix | `N/A: prefix not retained`, no comparison ratio |
| MTP depth `K` | Requires `K` suffix lookahead tokens, otherwise `N/A: MTP lookahead` |
| MLX prefix chunk | `--prefix-chunk 2048`, PSI uses its static `--chunk` |
| MLX allocator free-buffer limit | `--mlx-cache-gib 1`, does not limit live KV |
| oMLX hot cache | 20 GiB |
| PSI page arena | 98304 pages/request, allocated and zeroed at initialization |

TensorFold copies working cache containers from an immutable baseline through its native API.
Shorter GDN prefixes need their own recurrent state. Truncating KV alone is insufficient.
Recommendation: Order contexts increasingly for incremental preparation.
Prefix policy, setup chunk, cache budgets, source, checkpoints, and timing options are measurement identity.
Use a new output directory when they change.

Build the release launcher and supply separate engine environments:

```sh
cargo build --release -p inference-runtime-launcher --bin qwen3_5_dense
python3 scripts/qwen35_e2e_matrix.py \
  --engine all \
  --omlx-python /path/to/omlx-env/bin/python \
  --tensorfold-python /path/to/tensorfold-env/bin/python \
  --out target/perf/qwen35-e2e-matrix
```

The PSI environment also needs `grpcio` and `grpcio-tools`.

| Operation | Result |
| --- | --- |
| Repeat command | Skip valid completed results/capability exclusions, retry OOM/swap exclusions |
| `--engine plan` | Inspect without loading a model |
| `--engine report` | Regenerate without GPU work |
| `--requests`, `--contexts`, `--tokens`, `--full`, `--modes` | Select a subset |
| `--engine psi-dec --mode mtp2` | One worker in its engine environment |
| Metal OOM | Record `N/A: Metal OOM`, release GPU owner, exit 86, stop controller |
| New swap-in/out | Supervisor stops worker/controller, retains raw results, excludes affected observations |

The supervisor samples swap counters every 0.5 seconds. Existing swap occupancy alone does not reject a run.
Correct an OOM budget before explicit resume. Interrupted unrecorded shapes remain pending.
`results.jsonl` and ledger offsets bind exclusions to attempts, so a clean retry remains usable.
Reports are `report.md`, `ttft.md`, and `decode.md`.
Different input hashes must not produce comparison ratios.

Run CPU-only identity checks with `python3 scripts/qwen35_e2e_matrix_test.py`.
