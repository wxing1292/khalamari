# QSA indexer

The indexer prepares pooled keys and selects logical QSA groups.
[QSA SDPA](executor_gqa.md#qsa-sdpa) consumes the selected IDs and reads the main K/V cache.
The indexer does not read main K/V or allocate runtime pages.

## Source owners

```text
inference-executor-core/src/attn/qsa_indexer/
  core.rs              model dimensions and projection contract
  reference.rs         independent CPU math and deterministic top-k
inference-executor-metal/src/attn/qsa_indexer/
  backend.rs           projection, preparation, and selection recording
  batch_metadata.rs    request-local token records and score task templates
  scratch.rs           reusable QK, Q, and FP32 scores
inference-backend-metal/src/components/
  qsa_indexer.rs                    kernel selection and recording
  metal/qsa_indexer.metal           preparation, generic scoring, exact radix
  metal/qsa_indexer_vector.metal    register-based SIMD scoring
  metal/qsa_indexer_steel.metal     flat and head-loop Steel scoring
```

## Contract and lifetime

`T` counts flattened query tokens. `Hq` counts indexer Q heads. `D` is the head dimension.
`G` is the score-column capacity. `K` is the selected-group budget.
The model geometry uses BF16, `Hq=4`, `D=128`, compression ratio `4`, and `K=512`.

```text
hidden [T,H] -- quantized affine --> projected QK [T,(Hq+1)*D]
                                      |
                +---------------------+----------------------+
                |                                            |
          raw Q [T,Hq,D]                               raw K [T,D]
                |                                            |
          norm + RoPE                      source carry + complete groups of 4
                |                                            |
       Q [T,Hq,D], token-major                       BF16 mean -> norm + RoPE
                |                                            |
                |                                     paged K [groups,D]
                +----------------------+---------------------+
                                       |
                              FP32 scores [T,G]
                                       |
                            exact per-query top-512
                                       |
                         group IDs [T,512], ascending
```

A `Token` defines the visible interval
`[group_start, (token_index + 1) / compress_ratio)`.
The score at column `g` belongs to logical group `group_start + g`.
A sparse query has more than `top_k` complete visible groups.
Selection writes exactly `top_k` distinct IDs for that query.
Higher scores win. Lower group IDs win ties. Output IDs are chronological.
Dense queries and inactive destinations remain unchanged.

The caller owns physical pages, page IDs, and state versions.
Preparation keeps at most three raw BF16 keys per state slot.
Each active token has a distinct candidate destination slot.
Candidate slots must not overlap active source slots.
A completed group publishes one pooled key. Its RoPE position is the group's first token.
The caller can restore an accepted prefix by retaining its candidate state and cache extent.

For query token `t` and complete group `g`, the score is
`sum_h ReLU(dot(Q[t,h,:], K[g,:])) / sqrt(D)`.
ReLU applies after each complete dot product and before the head sum.
The indexer has four Q heads and one pooled K head. These are separate from the main GQA heads.
An incomplete group has no index key. QSA SDPA handles its visible tail tokens.

```text
One request: existing tokens [0,6), new tokens [6,8), ratio 4

source raw carry [2,128] = raw K for tokens 4,5
projected raw K [2,128]  = raw K for tokens 6,7
    |
    +-- token 6 -> candidate state [3,128] = raw K for 4,5,6
    |
    `-- token 7 -> FP32 mean(raw K[4..8,:]) -> BF16 [128]
                   -> norm -> RoPE(position 4)
                   -> publish logical index group 1
                   -> candidate carry length 0

Each nonempty carry writes its own candidate state slot.
Both read the unchanged source slot and projected batch, not another candidate.
```

## Shape selection

Selection uses workload facts rather than model names or prefill/decode labels.
| Record-time `Shape` field | Capacity |
| --- | --- |
| `num_total_tokens` | Flattened query tokens |
| `num_total_sparse_queries` | Sparse queries |
| `num_total_request_sparse_queries` | Maximum sparse queries in one request segment |
| `num_total_groups` | Score columns relative to each query's `group_start` |
| `num_total_score_task_templates` | Request-local score worklist |

For a request with history 2048 and eight input tokens that crosses the dense budget:

```text
Q [8,4,128] -> dense prefix [0,3), sparse suffix [3,8)
shape: total tokens 8, sparse queries 5, request sparse queries 5
score template: query_start=3, num_queries=5
    -> read Q [3..8,4,128]
    -> write scores [3..8,G], IDs [3..8,512]
```

The replay key must include the complete shape, cache layout, and component configuration.
The replay owner must omit selection and its score-task parameter when `num_total_groups <= top_k`.
When selection is recorded, submission supplies `num_active_tokens` and `num_active_score_task_templates` independently.
The active template count can be zero when all queries use dense attention.
Task-template capacity is independent of token capacity.
A request with 32 sparse queries uses one head-loop task. A smaller request with 31 sparse queries uses four flat tasks.
The replay owner must bucket task capacity separately. Metadata update enforces that bound.
A capacity can select a wider kernel than the active work requires. Metadata must follow the recorded geometry.

For BF16 `[T,4,128]`, pages must contain a multiple of 64 keys to use the optimized scorers.
The vector scorer also requires a power-of-two key count per page.
Other supported configurations use the generic SIMD scorer.
Let `N` be sparse-query capacity and `Q` be per-request sparse-query capacity.

| Condition | Score path |
| --- | --- |
| `Q == 1`, or `N <= 4`, or `G < 2048 && (N <= 8 \|\| Q <= 2)` | Vector SIMD; up to 8 queries per template |
| Other optimized shapes, request has fewer than 32 sparse queries | Steel flat; 8 queries × 4 heads per template |
| Other optimized shapes, request has at least 32 sparse queries | Steel head-loop; up to 32 queries per template |

A recording includes the head-loop path when `Q >= 32`.
There is no score-work threshold. Metadata uses each request's sparse-query count to select the task width.
The flat kernel handles templates with at most eight queries in that recording.
The head-loop kernel handles the other templates, including partial final tiles.
Each template stays inside one request and contains only sparse queries.
Flat and head-loop commands declare disjoint score writes before radix reads them.

```text
One batch: r0 has 33 sparse Q tokens, r1 has 9; Q [42,4,128]
Recorded geometry: Steel with head-loop, task capacity >= 4

task y    source Q tokens    num_queries    writer
   0      r0 [ 0,32)             32        head-loop
   1      r0 [32,33)              1        flat
   2      r1 [ 0, 8)              8        flat
   3      r1 [ 8, 9)              1        flat
                 |
         +-------+--------+
         |                |
      flat grid       head-loop grid     same (key tiles, task capacity, 1)
      y=1,2,3         y=0                other tasks return before loads
         |                |
         +-------+--------+
                 |
      disjoint scores [42,G] -> barrier -> per-query radix -> IDs [42,512]

Submission can activate fewer tasks without recording another program.
No request union, score reduction between tasks, or CPU score readback is needed.
```

The vector scorer uses one key per SIMDgroup when `G < 4096`.
Otherwise it uses the largest power of two at most `min(8, G / 2048)`.
A single-query specialization bounds the query loop at one.
It keeps all four Q-head slices in registers across the key loop.

## Score data flow

For two requests with 128 Q tokens each, `Q [256,4,128]` is token-major.
One flat task reads eight tokens from one request and 64 keys from one physical page.
The grid is `(aligned key tiles, recorded task-template capacity, 1)`.
There are 128 threads and four SIMDgroups per Steel threadgroup.

For a 32 KiB index page, one BF16 key occupies `128 * 2 = 256` bytes.
One page holds 128 keys, or 512 original tokens. A 64-key tile stays inside one page.
The scorer resolves the tile's page once. The runtime page ID does not encode a QSA group.

```text
r0 has H=4096 and Q=128; r1 has the same lengths
Q [256,4,128], score capacity G=(4096+128)/4=1056
page IDs [2,9], each physical page stores K [128,128] BF16

Head-loop task y=1 -> r0 Q tokens [32,64)
grid x=3          -> logical groups [192,256)
page_ids[r0,1]    -> physical page p, key offsets [64,128)
                         |
             Q [32,4,128] and K [64,128]
                         |
               four warps, WM=2 / WN=2

               group columns, each cell is 8 columns
             0    8   16   24   32   40   48   56
task Q    +----+----+----+----+----+----+----+----+
 [ 0, 8)  |SG0 |SG1 |SG0 |SG1 |SG0 |SG1 |SG0 |SG1 |
 [ 8,16)  |SG2 |SG3 |SG2 |SG3 |SG2 |SG3 |SG2 |SG3 |
 [16,24)  |SG0 |SG1 |SG0 |SG1 |SG0 |SG1 |SG0 |SG1 |
 [24,32)  |SG2 |SG3 |SG2 |SG3 |SG2 |SG3 |SG2 |SG3 |
          +----+----+----+----+----+----+----+----+

SG0 reads 16 Q tokens and 32 keys from staged SMEM per head.
Each warp retains eight FP32 [8,8] C output tiles and eight S output tiles.
No cross-warp score reduction: each warp writes its own output positions.
Result -> scores [r0 tokens 32..64, logical groups 192..256]
```

```text
Flat Steel: one task, Q [8,4,128], K [64,128]

register C [32,64] FP32 = 0            // M = 8 Q tokens * 4 heads
for d in 0..128 step 16:
    Q [32,16] BF16 -> SMEM A
    K [64,16] BF16 -> SMEM B           // reused across all 32 matrix inputs
    barrier
    register C += A * transpose(B)     // four SIMDgroups own distinct output positions
    barrier before SMEM reuse
SIMD shuffles sum ReLU(C) across 4 heads
write scores [8,64] FP32               // only each query's visible groups
```

A head-loop task reads up to 32 Q tokens from one request and the same 64-key tile.
Its four SIMDgroups partition a `[32,64]` output tile.
Each SIMDgroup owns a logical `[16,32]` output tile with interleaved positions.

```text
Head-loop Steel: one task, Q [32,4,128], K [64,128]

register S [32,64] FP32 = 0             // S/C are distributed across 128 threads
for h in 0..4:
    register C [32,64] FP32 = 0
    for d in 0..128 step 16:
        Q_h [32,16] BF16 -> SMEM A
        K   [64,16] BF16 -> SMEM B     // loaded again for each head
        barrier
        register C += A * transpose(B)
        barrier before SMEM reuse
    register S += ReLU(C)
write scores [32,64] = S / sqrt(128)   // each query has its own visible interval
```

For two requests with one Q token each, a vector task reads only that request's query.
The grid is `(aligned key tiles, task-template capacity, 1)`.
There are 256 threads and eight SIMDgroups per threadgroup.

```text
Vector SIMD: one task, Q [1,4,128], K [8,128]

resolve one physical page with the recorded page shift
for each SIMDgroup s in 0..8:          // independent key ownership, B=1 here
    Q [1,4,128] -> registers           // single-query specialization only
    K_s [128] -> register key          // lane l owns dimensions [4*l,4*l+4)
    for q in this request's query template:
        register score = 0
        for h in 0..4:
            use cached Q, or load Q[q,h,:] for a multi-query template
            register dot = sum(local Q * key)
            score += ReLU(simd_sum(dot))
        lane 0 writes scores[q,s] / sqrt(128)
```

The vector path uses no SMEM. Each key stays in registers across heads and queries.
Key tiles align to their own width. An unaligned `group_start` can require one additional grid slot.
The page size remains independent of the four-token QSA grouping.

## Exact radix

Generic configurations use 256 threads per query.
For optimized `top_k=512` shapes, `G < 8192 && N > 8` also uses 256 threads per query.
The other optimized `top_k=512` shapes use 1024 threads per query.
The latter path samples only when that query has at least 4096 visible groups.

```text
scores[q,visible G] FP32, GMEM
    |
    +-- full radix: four byte passes over visible scores
    |
    `-- sampled radix:
          1024 samples -> approximate cutoff
          scan ALL visible scores -> candidate count C
          if 512 <= C <= 2048:
              compact ALL qualifying keys/IDs -> SMEM [C]
              four exact radix passes over SMEM candidates
          else:
              four exact radix passes over ALL visible scores
    |
    exact cutoff + remaining tie budget, registers
    |
    per-lane contiguous intervals -> SIMD prefix scans -> threadgroup prefix scan
    |
    IDs [q,512], increasing logical group ID
```

Sampling does not change membership. The full-range path is required for valid inputs such as all-equal scores.
SIMDgroup 0 scans the 256 histogram bins. Each lane owns eight bins.
Only visible score columns are read. Stale columns cannot enter selection after a context shrink.
There are no global candidate buffers or merge dispatches.

## Score study, 2026-10-16

The complete prefill study contains 1560 timed sparse shapes, one fixed preflight, and 128 dense-only skips.
Each timed shape has 22 samples per arm and metric. The study contains 1,990,560 timing records.
The workload-balanced selection fit chooses one head-loop BM32/BN64/BK16 arm for both score and score-plus-radix.
No tested shape threshold meets the required 3% training improvement.
Training uses 1024 shapes. Validation holds out 536 complete context-bucket shapes.

The measurement source is commit `0752151867c827aa08b9c86937cf1fa631c41c05` with uncommitted QSA work.
[The checked-in result](qsa_indexer_performance.json) retains the dirty state, source hashes, environment, command, and all 29 candidate medians.
Each result case stores `score` and `score_radix` arrays in the declared `arms` order, in microseconds.
The machine is Apple M3 Max with 48 GiB on macOS 27.0.1. The Python environment uses MLX 0.32.3 and TensorFold 0.6.6.
The oMLX source commit is `64bf5d1ba6336b45955c00f7fe0e46acd5c85095`.
Inputs are synthetic BF16 Q and pooled K. No model weights are involved.

The runner command is:

```sh
caffeinate -i /private/tmp/psi-dec-prefill-compare-20261008/.venv-tensorfold/bin/python \
  scripts/qsa_indexer_score_matrix.py --mode run --suite prefill \
  --out target/perf/qsa-indexer-score-night-v2
```

The metric is common MLX compiled-graph wall-clock latency while the desktop can be in use.
Every completed case reports no new swap. Failed attempts do not enter the report.
The measured paths adapt scorer source bodies. They do not reproduce complete framework runtimes or production Metal ICB.
All score-plus-radix arms use the same exact full256 selector. Production sampled/full radix selection is unchanged.
Q/K projection, pooling, page construction, and layout preparation are excluded. FP32 casts and matmul intermediates are included.
KHALAMARI uses paged K and token-major Q. The oMLX and TensorFold FP32 peers use prepared contiguous K.

The following changes are KHALAMARI latency relative to each peer. Negative values mean lower latency.
Each matched shape has equal weight in these geometric means.

| Score plus common radix | Shapes | oMLX native | oMLX FP32 | TensorFold FP32 | TensorFold SIMD B4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| All unique shapes | 1560 | -2.8% | -36.4% | -45.5% | -57.5% |
| Append | 144 | -4.4% | -36.2% | -45.0% | -54.6% |
| One-shot | 8 | -24.2% | -66.9% | -73.3% | -73.5% |
| Growth chunks | 1408 | -2.5% | -36.2% | -45.4% | -57.7% |

Score latency is 10.7% lower than the flat baseline. Score-plus-common-radix latency is 9.2% lower.
The held-out selected/baseline ratios are 0.888 and 0.905, respectively.
Even an optimistic per-shape oracle reduces held-out score-plus-radix latency by only another 0.26% relative to fixed BM32.
These are aggregate results. BM32 is not fastest at every shape, and its small advantage over oMLX native varies by shape.
Growth sums cover independent synthetic sparse chunks. They are not complete prefill latency or TTFT.

Representative absolute latencies use score plus the common radix. All times are microseconds.

| Requests | History | Q tokens/request | KHALAMARI BM32 | oMLX native | oMLX FP32 | TensorFold FP32 | TensorFold SIMD B4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 128 | 277.23 | 296.11 | 305.42 | 307.39 | 316.52 |
| 1 | 16384 | 128 | 325.78 | 325.26 | 394.07 | 406.16 | 524.64 |
| 1 | 65536 | 256 | 819.82 | 833.95 | 1508.86 | 1823.44 | 2425.77 |
| 4 | 16384 | 256 | 790.23 | 806.10 | 1487.21 | 1799.36 | 2397.21 |
| 8 | 65536 | 256 | 4434.90 | 4476.39 | 11569.60 | 12912.29 | 17393.89 |
| 1 | 0 | 4096 | 486.03 | 517.42 | 864.00 | 1041.91 | 1145.85 |
| 1 | 0 | 16384 | 4026.27 | 6180.22 | 16184.11 | 20332.42 | 18796.23 |
| 2 | 0 | 16384 | 7697.91 | 12087.29 | 32104.54 | 40290.22 | 37318.47 |

The following comparison uses the same input and makes the source-level resource change explicit:

```text
Input: Q [64,4,128] BF16, K tile [64,128] BF16 from a history with G > 512
Both: BK16, WM2/WN2, 128 threads per block, four warps

Before BM64:
    one block, Q [0,64), groups [0,64)
    register C/S [64,64] each, distributed across 128 threads
    each thread retains 64 FP32 C+S values

After BM32:
    block 0: Q [ 0,32), groups [0,64)
    block 1: Q [32,64), groups [0,64)
    register C/S [32,64] each, distributed across each block's 128 threads
    each thread retains 32 FP32 C+S values

Both: loop over four heads, stage Q/K slices in SMEM, retain C/S in registers
      -> score tile [64,64] FP32 -> complete scores [64,G]
      -> common exact radix -> IDs [64,512]
```

| Interpretation | Limit |
| --- | --- |
| Reduced C/S storage and more blocks | Compiler register allocation/occupancy unmeasured |
| KHALAMARI skips wholly future tiles before loads, oMLX masks at store | One-shot differences do not isolate BM32 |
| Resident-K/Q alternatives lose | Fewer explicit reads alone do not establish lower latency |
| Q=1..16 and mixed-tail production composition | Remain unmeasured, existing choices unchanged |


## Benchmarks

| Production Criterion replay | Contract |
| --- | --- |
| Timed | `Compute::select`, scoring and full/sampled selector |
| Excluded | Preparation and main QSA SDPA |
| Requests / Q tokens | `1/2/4/8` / `1/8/32/128/256` |
| History / one-shot | `4096/16384/65536` / `4096/16384` |
| Invocations | 32/replay, divide latency by 32 for per-invocation latency |

```sh
cargo bench -p inference-backend-metal --bench qsa_indexer
```

The Python study compares forced scorer variants under a common MLX timing envelope.
It supports full prefill and small-Q suites, validates against CPU math, and saves resumable per-case results.
Run it with a Python environment that contains MLX, NumPy, and TensorFold.
Use `--omlx-dir` to select the oMLX source checkout.
There is no default time limit. Repeat the same command to resume unchanged inputs and sources.
The supervisor rejects new system swap, worker failure, or source changes.

```sh
python scripts/qsa_indexer_score_matrix.py --mode plan --suite prefill
python scripts/qsa_indexer_score_matrix.py --mode validate --suite smoke --out target/perf/qsa-indexer-smoke
python scripts/qsa_indexer_score_matrix.py --mode run --suite prefill --out target/perf/qsa-indexer-prefill
python scripts/qsa_indexer_score_matrix.py --mode report --out target/perf/qsa-indexer-prefill
```

`summary.csv` retains all per-shape medians. `selection.json` retains the held-out selection result.
`report.md` contains per-arm comparisons and synthetic growth sums.
The common selector isolates scorer differences. It cannot establish production replay or complete-engine speedups.

## Verification

Backend tests compare scores, selected membership, and preparation with independent CPU math.
They cover fixed ties, seeded random BF16 inputs, generic F32 geometry, unaligned group starts,
reversed physical pages, short query templates, tails, and non-monotonic replay counts.
Executor tests cover metadata, candidate-state restoration, and batched indexer plus SDPA composition.
Run Metal commands serially as specified in [executor verification](executor_benchmarks.md).
