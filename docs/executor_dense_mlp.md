# Dense MLP Executor

This document describes the current dense gated-MLP implementation.
It covers semantic shapes, scratch ownership, Metal replay, tests, and production benchmarks.

## Source layout

`crates/inference-executor-core` intentionally has no MLX or Metal dependency.
It owns backend-neutral dense MLP layer metadata.
`crates/inference-executor-metal` owns the current Metal replay backend:

```text
crates/inference-executor-core/src/mlp/dense/
  mod.rs
  core.rs      DenseMLPCore

crates/inference-executor-metal/src/mlp/dense/
  mod.rs
  backend.rs   DenseMLPMetalConfig + DenseMLP
  scratch.rs   reusable dense MLP scratch allocation owner and borrowed replay bindings

crates/inference-executor-metal/src/model/qwen/
  v3_x/layer/dense_mlp.rs  Qwen3xDenseMLP, private checkpoint weights, load, and record
  v3/main/layer.rs         fixed Qwen3 Main GQA + dense-MLP layer composition
  v3/main/component_config.rs
                           Qwen3 Main dense-MLP geometry/config builder
  v3_5/main/layer.rs       Qwen3.5 Main dense-MLP/MoE layer variants
  v3_5/mtp/layer.rs        Qwen3.5 MTP dense-MLP/MoE layer variants
  v3_5/component_config.rs Qwen3.5 dense-MLP geometry/config builder

crates/inference-executor-core/src/def/
  DenseLinearShape
  SparseLinearShape

crates/inference-executor-metal/src/def/
  ReplayLayer              typed semantic replay input/output and record contract
crates/inference-backend-metal/src/components/replay.rs
  ReplayRecorder           concrete Metal recording and fusion
```

The current runtime path is the Metal replay path in
`crates/inference-executor-metal`.

Reusable Metal dense MLP kernels live in:

```text
crates/inference-backend-metal/src/components/dense_mlp.rs
crates/inference-backend-metal/src/components/dense_mlp_test.rs
crates/inference-backend-metal/src/operators/affine_quantized.rs
crates/inference-backend-metal/src/operators/metal/affine_quantized_gate_up_swiglu.metal
crates/inference-backend-metal/src/operators/metal/affine_quantized_qmm_steel.metal
crates/inference-backend-metal/src/operators/metal/affine_quantized_gate_up_swiglu_steel.metal
crates/inference-backend-metal/src/operators/metal/affine_quantized_small_qmm.metal
```

## Shape model

`DenseMLPCore` owns immutable layer metadata:

```text
model_layer_index
hidden_dim
intermediate_dim
```

It derives dense MLP projection shapes:

```text
linear_shape
gate_up_shape
down_shape
```

`DenseMLP` connects model-level dense MLP metadata to `inference-backend-metal` kernels.
It owns the full `gate_up_swiglu -> down` backend path.
It does not own tensor storage, runtime scheduling, or page allocation.

The backend implements `ReplayLayer`.
Qwen model and layer code use the backend `ReplayRecorder` to append dense MLP work to a larger whole-layer or whole-model replay.
Focused tests and benches build replay programs from the same recorder path.
The internal order is `gate_up_swiglu -> down [barrier before]`.
Model and layer wiring own barriers on the first consumer command and downstream residual consumers.

## Replay contract

`DenseMLP` records one dense gated MLP forward into a caller-owned `ReplayRecorder`.
It does not submit commands.
It does not own tensor storage or request lifecycle.
The semantic layer input is
`DenseMLPInput { num_total_tokens, num_active_tokens, hidden_state, next_hidden_state, scratch, weights }`.
Replay returns the caller-owned `next_hidden_state` buffer directly.
The immutable core does not wrap replay counts in a separate shape. Dense MLP and MoE pass the recorded capacity and
active-token parameter directly to their backend invocations.

The replay order is:

```text
hidden_state
  -> fused gate/up quantized projection and SwiGLU
  -> down quantized projection
  -> next_hidden_state
```

`DenseMLPInput.num_total_tokens` is the recorded row capacity. `num_active_tokens` accepts
`ReplayU32::Fixed(value)` or `ReplayU32::Parameter(key)`.
Production callers allocate scratch for model capacity.
A fixed invocation requires the active and total token counts to match. Every invocation validates all buffers against
`num_total_tokens`.
All buffers and weights must match the configured dimensions, group size, bit width, and dtype.
This requirement covers hidden buffers, SwiGLU scratch, and immutable weights.

Qwen model replay keeps dense MLP scratch in one model-owned `DenseMLPScratch`.
Its `bindings()` method exposes borrowed `DenseMLPScratchBindings` during replay recording.
Scratch allocation geometry consists of `max_tokens`, `intermediate_dim`, and `io_dtype`.
It does not accept quantization group size or bit width because those weight facts do not affect scratch layout.
The model stream serializes Main and MTP execution.
Thus, layers can reuse the SwiGLU scratch.

The shared `Qwen3xDenseMLP` leaf owns per-layer immutable weights and a shared scratch handle.
The caller owns hidden input and output buffers.
It retains the core and Metal configuration that created its backend.
Weight reload uses these retained values.
`Qwen3MainLayer` and the dense variants of `Qwen35MainLayer` and `Qwen35MTPLayer` compose that leaf.
Each composition uses a separate role-specific layer and scratch type.
Their model-specific binding trees contain `Qwen3xDenseMLPWeightBindings` at the leaf boundary.
The weight owner loads one bounded `TensorMap` from that exact gate/up/down binding subtree.
It removes every tensor and materializes fused gate-up buffers while it keeps the down projection separate.
The map must be empty after construction.
When Main and MTP both use dense MLP, Qwen3.5 initialization requires their `intermediate_size` values to match.
Both users share the executor `max_tokens` value and the current BF16 model boundary.
If only one graph uses dense MLP, the loader derives the scratch geometry from that graph.

### Bucketed replay

`dense_mlp::Compute` exposes one composite topology identity:

```text
dense_mlp::ReplayTopology
  gate_up_swiglu_affine
  down_affine
```

It also exposes the sorted union of the `gate_up_swiglu` and `down` affine topology boundaries.
The owner of a larger replay stage must union these boundaries with the boundaries from all other token-domain components.
The larger replay stage then selects one shared `num_total_tokens` capacity.
Dense MLP does not own that final policy.

The replay key must contain `num_total_tokens` and the composite dense MLP topology.
The key must not contain `num_active_tokens`.
The submission supplies `num_active_tokens` through the caller-owned replay parameter key.

The two dense MLP stages bind the same `u32` key with the same `1..=num_total_tokens` domain:

```text
gate_up_swiglu affine
down affine
```

Thus, a parameterized dense MLP replay declares one parameter. A fixed active count declares no parameters.

Each stage records work for `num_total_tokens` rows.
Affine QMV returns for each inactive row before it reads input or writes output.
Affine QMM skips fully inactive tiles and masks inactive rows in a partially active tile.
The fused epilogue writes only active SwiGLU rows.

Inactive scratch rows can contain poison, output from an earlier full submission, or other stale values.
The implementation does not clear these rows.
The guards ensure that later stages do not consume or overwrite them.

`Qwen3xDenseMLP` exposes one `record(...)` API and topology accessors.
Qwen3.5 Main calls this API with the Main stage-owned total token count and active-token value.
Qwen3.5 MTP calls this API with its separate body stage-owned token capacity and active-token key when the physical MTP
layer uses dense MLP.

## Data flow and backend stages

Dense MLP is a pure hidden-state transform with no request page/state side effects:

```text
hidden_state[num_tokens, hidden_dim]
  -> fused gate/up quantized affine
  -> swiglu[row, intermediate] = SiLU(gate[row, col]) * up[row, col]
  -> down quantized affine
  -> next_hidden_state[num_tokens, hidden_dim]
```

Gate and up weights remain stacked along the output dimension. `config.n` is `2 * intermediate_dim`.
The fused affine operator produces `intermediate_dim` output columns.
QMV assigns gate and up to separate SIMDgroups.
QMM keeps separate gate and up accumulators in each SIMDgroup.
Steel QMM loads the input tile once and stages the gate and up weights in sequence.
Register QMM reads packed weights directly into the SIMDgroup operands or lane-local dot products.
QMV transfers rounded projection values through threadgroup memory.
Steel and sliced-K MMA QMM keep both projection accumulators in registers.
Scalar register QMM assigns gate and up to separate SIMDgroups.
Both paths round gate and up to the output dtype, compute
`(gate / (1 + exp(-gate))) * up` in F32, and round the result to the output dtype.
This sequence preserves the dense MLP reference contract.
Sparse MLP uses its own intermediate-rounding contract.

The fused operator writes one `swiglu[num_tokens, intermediate_dim]` scratch buffer.
It does not materialize the stacked gate/up projection.
The down projection reads that scratch and immutable down weights.
The model-boundary input, scratch, and output use BF16.

Resource flow is:

```text
gate_up_swiglu affine
  reads hidden_state + gate/up weights/scales/biases
  writes swiglu scratch

down affine
  reads swiglu scratch + down weights/scales/biases
  writes next_hidden_state
```

The component records barriers between these stages.
Each stage consumes scratch from the previous stage.
Model replay records additional layer-level barriers around residual and norm consumers.
It does not put these barriers inside the dense MLP component.

Dense MLP has no expert-major policy.
Every active token row runs the same dense expert.
The backend exposes one `invoke(...)` API.
It accepts `Shape { num_total_tokens }` and one `ReplayU32` active-token value.
`ReplayU32::Fixed(value)` requires `value == num_total_tokens`.
`ReplayU32::Parameter(key)` supplies the active prefix at submission.
Capacity buffers can be larger than the active prefix.

## Execution hierarchy

Dense MLP is a semantic command graph. It is not one kernel launch:

```text
DenseMLPExecution
├── gate_up_swiglu affine_quantized::Matmul
└── down affine_quantized::Matmul
```

Each affine owner defines its QMV or QMM thread-block task, tile geometry, and layout. The dense MLP owner supplies the
projection geometry and runtime row count. It does not select the affine kernel again.

Dense MLP does not use a component-level registry, selector, or planner. The command graph does not change with the
runtime row count. The
two affine owners make independent row-dependent kernel choices. `dense_mlp::ReplayTopology` records both affine
choices so that a replay bucket cannot cross either topology boundary.

## Backend selection

`dense_mlp::Compute` owns two adaptive `affine_quantized::Matmul` objects.
It constructs the gate/up/SwiGLU operator with `new_gate_up_swiglu` and the down operator with `new`.
Each `affine_quantized::Matmul` owns the QMV/QMM candidates and selects its kernel from the fixed epilogue and total
row count. Selection and topology boundaries use the same selector. They do not depend on registry entry order.
`operators/metal/affine_quantized_small_qmm.metal` owns the register-based small-row kernels.
The shared affine selector uses dimensions, quantization, and dtypes. Model code does not arm a separate route.
For eligible Q4 FP16 or BF16 operands, rows 4 through 8 use `QmmSlicedK`.
sliced-K MMA constructs exact `128 + q` matrix operands in registers. It applies scale and bias after each group dot product.
The kernel retains F32 accumulators and the storage-dtype SwiGLU intermediate rounding.
It does not round each dequantized weight to BF16 or FP16 before the dot product.
This reduction and weight-rounding sequence can differ numerically from Steel QMM.

sliced-K MMA uses sliced-K: it assigns K partitions to SIMDgroups in one threadgroup.
The family name does not require eight SIMDgroups. The geometry selects four or eight.
`n` is the output width of one projection. For fused gate/up, it is `config.n / 2`.
The K partition count stays fixed across row tiles.

| Projection shape | SIMDgroups per threadgroup | Default CT for 4–8 rows |
| --- | ---: | ---: |
| `n >= 65536` | 4 | 4 |
| `n < 65536` and `k >= 16384` | 8 | 2 |
| Other shapes | 8 | 1 |

Eligibility requires one shared operand dtype, Q4, `n >= 1024`, complete output tiles, and
`k % (group_size * num_simdgroups) == 0`.
The existing affine contract supports group sizes 32, 64, and 128. It does not add Q5 support.
Rows 4–8 use RT1 and the default CT above.
The 9–64-row policy requires BF16, group64, `n >= 4096`, and `k >= 4096`.
It rounds the recorded row capacity up to a multiple of eight before it selects RT and CT.

| Rounded row capacity | Ordinary RT / initial CT | Fused SwiGLU RT / CT |
| ---: | --- | --- |
| 16 | 2 / 4 | 2 / 1 |
| 24 | 3 / 4 | 3 / 1 |
| 32 | 2 / 4 | 4 / 1 |
| 40 | 1 / 2 | 1 / 2 |
| 48 | 3 / 4 | 3 / 1 |
| 56 | 1 / 2 | 1 / 2 |
| 64 | 2 / 4 | Steel |

For `n >= 65536`, rounded capacities 32, 56, and 64 retain Steel.
For the other entries, CT halves until `n % (8 * CT) == 0`.
Ordinary entries with RT greater than one also reduce CT until the grid has at least 256 threadblocks or CT reaches one.
This condition preserves output parallelism for smaller projection widths.
The selector uses the recorded row capacity. It does not use the active row count.

```text
one threadgroup:
  output tile = (8 * RT) rows x (8 * CT) columns
  threads     = 32 * num_simdgroups
  grid        = ceil(num_total_rows / (8 * RT)) x (n / (8 * CT))

  packed weight fragment --+--> row tile 0
                           +--> row tile 1
                           +--> row tile 2
                           +--> row tile 3

  K partitions -> F32 partials in threadgroup memory -> one reduction
  threadgroup memory bytes = matrices * RT * CT * num_simdgroups * 32 * 8
  matrices = 1 for ordinary projection; 2 for fused gate/up
```

RT reuses each packed weight fragment across 8-row matrix tiles.
It preserves each row's group accumulation order, K partitions, and reduction order.
It also preserves storage-dtype gate/up rounding before SwiGLU.
The kernel does not allocate a global partial buffer or use a second reduction dispatch.
Active rows are submission parameters. A threadgroup outside the active row range returns before it loads input.
Padded lanes in the last active tile read the last active input row and never write inactive output rows.
Initialization registers only variants that fit device threadgroup memory and pipeline thread limits.
An unavailable sliced-K MMA geometry uses the existing base policy.
Topology boundaries use the actual registered selection.
`KernelKind::QmmSlicedK` records RT and CT. Active row changes do not compile a kernel.
MLP, GQA projections, GDN projections, and the LM head use this same affine selector.

Three-row projections use lane-strided QMM when all operands share FP16 or BF16,
weights use Q4 or Q8, `n >= 4096`, `n % 4 == 0`, and `k % 64 == 0`.
For `n >= 100000` with `n % 32 == 0`, `QmmSimdgroupsBm4Bn32` assigns four columns to each of eight SIMDgroups.
Otherwise, `QmmSlicedKBm3Bn4` assigns two K partitions to each four-column threadgroup.
This scalar kernel partitions K across SIMDgroups in one threadgroup.
The K-partition reduction stays inside that threadgroup. It needs no global partial buffer or second dispatch.
Both variants support stacked gate/up weights and the storage-dtype SwiGLU epilogue.
Each SIMDgroup keeps one projection's accumulator array. Fused gate/up doubles the SIMDgroup count.
The threadgroup transfers both projection results before the fused activation. It does not allocate a device scratch buffer.
The compiler emits literal row and column accumulator indices. The column kernel unrolls each packed-value loop.
Weight, scale, and bias row bases use 64-bit offsets before the K loop. The loop uses row-local indices.

Other shapes retain the existing QMV and Steel QMM policy.
Steel QMM uses `BlockMMA` with F32 accumulators and 8x8 SIMDgroup matrix operations.
`operators/metal/affine_quantized_qmm_steel.metal` owns ordinary QMM.
`operators/metal/affine_quantized_gate_up_swiglu_steel.metal` owns fused gate/up/SwiGLU.
Each block loads an input tile and dequantizes a weight tile into SMEM.
BM8 and BM16 use BK64 when all operands use BF16, the quantization group is at least 64, and `n < 65536`.
The other configurations use BK32.
Initialization requires `k % group_size == 0`. Eligible BK64 configurations therefore have no K tail.
The shader specialization and pipeline-memory validation use the same BK value.
The fused kernel reuses the input tile across gate and up. The projections reuse one weight staging region.
Both projection accumulators remain in REGISTER until the fused activation.
The kernel rounds each projection to its storage dtype before it computes SwiGLU in F32.
Steel same-dtype operands retain storage-dtype weight rounding. Mixed-dtype operands use F32.
QMV remains the one- and two-row path and serves shapes outside the register-QMM domain.
Full tiles use unchecked loaders. Partial tiles retain active-row and output-column masks.
The model and executor provide the complete dense-MLP dimensions and active row count.
They do not select a kernel or tile.

After the eligible register-QMM rules above, large dense MLPs use this base policy when
`hidden_dim > 4096` or `intermediate_dim > 4096`:

| Recorded row capacity | Gate/up/SwiGLU | Down |
| ---: | --- | --- |
| 1–5 | QMV BN4 | QMV BN8 |
| 6–8 | QMM BM8/BN16 | QMM BM8/BN32 |
| 9–16 | QMM BM16/BN16 | QMM BM16/BN32 |
| 17 or more | QMM BM32/BN16 | QMM BM32/BN32 |

The following rule overrides rows above 16 for each eligible affine operator.
Input, output, and scale/bias must all use BF16. Weights must use Q4/group64.
The affine config must have `n >= 4096` and `k >= 4096`.

| Recorded row capacity | Gate/up/SwiGLU | Ordinary projection, including down |
| ---: | --- | --- |
| 17–2047 | QMM BM32/BN32/BK32 | QMM BM32/BN64/BK32 |
| 2048 or more | QMM BM64/BN32/BK32 | QMM BM64/BN64/BK32 |

An eligible ordinary projection keeps BM16/BN32 for 17–32 rows when
`ceil(n / 64) < 256`.
This rule applies when the BM32/BN64 grid has fewer than 256 threadblocks.
BM16/BN32 launches `2 * ceil(n / 32)` threadblocks and uses BK64 under the existing BK rule.
Each threadblock accumulates the complete K dimension. This path does not use split-K.
This base rule applies after the sliced-K MMA rules above.

The selector exposes each selected topology boundary to replay bucketing.
sliced-K MMA exposes changes at 9, 17, 25, 33, 41, 49, 57, and 65 when they change the selected RT, CT, or family.
The base low-parallelism ordinary projection transitions to BM32 at 33 rows instead of 17.
The 2048-row boundary remains unchanged.
Gate/up and down evaluate eligibility independently.

Smaller dense MLPs outside the register-QMM domain keep QMV for a longer range.
The first QMM row count is 18 when both dimensions are at most 2048.
The first QMM row count is 12 for the remaining smaller shapes.
After this crossover, the backend uses BM16 through 16 rows and BM32 for larger row counts.
The output tile remains BN16 for fused gate/up/SwiGLU and BN32 for down.

Gate/up and down apply the same backend selector independently.
They can share a family when their dimensions select the same candidate.
Ordinary BM8 and BM16 QMM use four SIMDgroups across the output columns.
Fused BM8 and BM16 QMM use two SIMDgroups.
BM32 and BM64 use four SIMDgroups in both paths.
For group size 64, the BF16 BM8/BN32/BK64 kernel uses 128 threads and 5760 bytes of static threadblock memory.
The memory contains the `8 × 72` input tile and the `32 × 72` weight tile.
The stride is `BK` plus 16 bytes of padding.
The fused BM8/BN16/BK64 kernel uses 64 threads and 3456 bytes of static threadblock memory.
It shares one input tile between gate and up and reuses one weight staging region.
Each SIMDgroup keeps two F32 projection accumulators live.
Each BK iteration has two threadgroup barriers in ordinary QMM and four in fused QMM.
The fused kernel rounds both projections to the output dtype before it computes SwiGLU in F32.
It needs no separate projection allocation.
In fused QMV, one SIMDgroup computes all gate columns in the output tile. The other computes the matching up columns.
Each thread keeps one weight/scales/biases pointer set and one accumulator array.
Initialization selects an aligned specialization when both output columns and the K dimension contain complete tiles.
That specialization omits tail checks from the reduction loop.
Kernel initialization checks the SIMD width, pipeline thread limit, calculated threadblock memory, reported static
threadblock memory, and device threadblock-memory limit.
Initialization also validates the kernel/epilogue pairing. Tile width does not imply an activation or scratch layout.

Benchmark-only QMV/QMM probes select an affine kernel policy for measurement.
The semantic data flow stays the same.

### Steel and sliced-K

Steel names a compute implementation. Sliced-K names a work partition.
The current Steel implementation partitions output fragments across SIMDgroups and stages operands in SMEM.
The current sliced-K MMA implementation partitions K across SIMDgroups and constructs operands in registers.
Other implementations can combine these choices.
The terms follow [CUTLASS parallelized reductions](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#parallelized-reductions).

The following diagram defines the partition and storage terms:

```text
Metal term       Corresponding term in these diagrams
SIMDgroup (SG)   warp: 32 lanes
threadgroup (TG) threadblock: one or more SIMDgroups
SMEM            threadgroup memory
GMEM            device memory

BK        sequential K-stage length
sliced-K  K partitions across SIMDgroups inside one threadgroup
split-K   K partitions across threadgroups

Current paths:
  Steel        -> different output coordinates per SG -> each SG covers K [0, K)
  sliced-K MMA -> same output coordinates per SG      -> each SG covers a K slice
                                                     -> merge partials in SMEM
```

### Steel output tiles

This ordinary Steel example uses BM16/BN32, WM1/WN4, and four SIMDgroups:

```text
X [16, K]       W_Q4 [32, K]       Y [16, 32] = X * W^T

                         BN = 32
             +--------+--------+--------+--------+
BM = 16      | SG0    | SG1    | SG2    | SG3    |
             | 16x8   | 16x8   | 16x8   | 16x8   |
             +--------+--------+--------+--------+
columns      [0,8)    [8,16)   [16,24)  [24,32)

SG0: rows [0,16), columns [ 0, 8), K [0,K)
SG1: rows [0,16), columns [ 8,16), K [0,K)
SG2: rows [0,16), columns [16,24), K [0,K)
SG3: rows [0,16), columns [24,32), K [0,K)

threadgroup = WM * WN * 32 = 128 threads

K traversal for every SG:
  [0, BK) -> [BK, 2*BK) -> ... -> [K-BK, K)
```

The following forced example uses the same full tensor dimensions as the sliced-K example below.
The adaptive selector can choose a different implementation.

```text
M=16 / N=5120 / K=17408, BF16 Q4/group64
BM=16 / BN=32 / BK=64 / WM=1 / WN=4

X [16, 17408]         W_Q4 [5120, 17408]
      |                        |
      +-- 1 M tile             +-- 160 N tiles
                 \             /
                  Y [16, 5120]
        +--------+--------+--------+-----+--------+
        | TG 0   | TG 1   | TG 2   | ... | TG 159 |
        | 16x32  | 16x32  | 16x32  |     | 16x32  |
        +--------+--------+--------+-----+--------+

dispatch:
    grid        = (160 N-tiles, 1 M-tile, 1)
    threadgroup = (32 threads, 4, 1) = 128 threads

One TG, one of 272 sequential K stages:

    GMEM X [16x64] ----> SMEM Xs [16x72] BF16 --+
                                               +--> four SGs' MMA
    GMEM W_Q4 [32x64] -> dequantize              |
    scales/biases ----> SMEM Ws [32x72] BF16 ----+

    72 = BK64 + 16-byte padding / 2 bytes
    SMEM = (16 + 32) * 72 * 2 = 6912 bytes

    SG0 [16x8]    SG1 [16x8]    SG2 [16x8]    SG3 [16x8]
    columns 0..8  columns 8..16 columns 16..24 columns 24..32
        |             |             |             |
        +---- each retains F32 output fragments across all 272 stages
        +---- each covers the full K range
        +---- each stores its own output columns

    One SG: two 8x8 output fragments
    Per lane: 2 * (64 / 32) = 4 persistent F32 output values
    No K-partial merge. No global partial-output buffer.
```

Each threadgroup stages one BK interval at a time. It does not load the complete K range at once.
The ordinary shader uses this loop order:

```text
for k = 0; k < K; k += BK:
    threadgroup barrier             // before reusing the previous stage's SMEM
    load X[BM, BK] into SMEM
    dequantize W[BN, BK] into SMEM    // storage dtype for same-dtype operands
    threadgroup barrier             // operands are ready for all SGs
    each SG updates its own output fragments

after the final stage:
    each SG stores its output fragments
    no K-partial merge
```

The fused Steel shader uses WN2 for all BM values.
For BM16/BN16, both SIMDgroups compute gate and up for their own output region:

```text
                   BN = 16
             +----------+----------+
BM = 16      | SG0      | SG1      |
             | gate/up  | gate/up  |
             | 16x8     | 16x8     |
             +----------+----------+
columns      [0,8)      [8,16)

SG0: rows [0,16), columns [0, 8), K [0,K), gate AND up
SG1: rows [0,16), columns [8,16), K [0,K), gate AND up
threadgroup = WM1 * WN2 * 32 = 64 threads

for each BK stage:
    barrier -> load X + dequantize gate -> barrier -> gate MMA
    barrier -> reuse X + dequantize up -> barrier -> up MMA

after all K stages:
    round gate and up to storage dtype -> F32 SwiGLU -> store
```

Larger Steel tiles distribute 8x8 fragments with the `BlockMMA` strides:

```text
                       BM8 / BM16       BM32 / BM64
ordinary WM / WN       1 / 4            2 / 2
fused WM / WN          1 / 2            2 / 2

For SIMDgroup sg:
  warp_row = sg / WN
  warp_col = sg % WN
  row-fragment starts = 8 * (warp_row + i * WM), i = 0..BM/(8*WM)-1
  col-fragment starts = 8 * (warp_col + j * WN), j = 0..BN/(8*WN)-1
  every output fragment traverses K [0, K)
```

### Sliced-K tiles

This forced geometry shows RT2/CT4 and eight SIMDgroups.
It is a valid kernel geometry. It is not the adaptive default for this shape.
The default reduces CT to two to preserve at least 256 output threadgroups.

The following example shows the logical tensor partition:

```text
Example: ordinary projection, M=16 / N=5120 / K=17408
Geometry: RT=2 / CT=4 / S=8
S = number of SIMDgroups per threadgroup

X [16, 17408]       W_Q4 [5120, 17408]
       |                    |
       |                    +-- N: 160 tiles, 32 output columns per tile
       |
       +-- M: 1 tile, 16 rows per tile

Y [16, 5120]
   +--------+--------+--------+-----+----------+
   | TG 0   | TG 1   | TG 2   | ... | TG 159   |
   | 16x32  | 16x32  | 16x32  |     | 16x32    |
   +--------+--------+--------+-----+----------+

dispatch:
    grid        = (1 M-tile, 160 N-tiles, 1)
    threadgroup = (256 threads, 1, 1)
    total       = 160 threadgroups

One TG covers:
    X_tile = X[16 rows, entire K]
    W_tile = W[32 output columns, entire K]

    K=17408
    +------+------+------+------+------+------+------+------+
    | K0   | K1   | K2   | K3   | K4   | K5   | K6   | K7   |
    |2176  |2176  |2176  |2176  |2176  |2176  |2176  |2176  |
    +------+------+------+------+------+------+------+------+
       |      |      |      |      |      |      |      |
      SG0    SG1    SG2    SG3    SG4    SG5    SG6    SG7
       |      |      |      |      |      |      |      |
      P0     P1     P2     P3     P4     P5     P6     P7
    [16x32][16x32][16x32][16x32][16x32][16x32][16x32][16x32]
       \______\______\______\______\______\______\______/
                              |
                         SUM over K
                              |
                         Y_tile[16x32]

    SG0: K [    0,  2176)
    SG1: K [ 2176,  4352)
    SG2: K [ 4352,  6528)
    SG3: K [ 6528,  8704)
    SG4: K [ 8704, 10880)
    SG5: K [10880, 13056)
    SG6: K [13056, 15232)
    SG7: K [15232, 17408)

    Each SG covers the same 16x32 output coordinates.
    SGs partition K, not output rows or columns.

Actual adaptive selection for these M/N/K values:
                    forced example       adaptive default
    RT / CT / S     2 / 4 / 8            2 / 2 / 8
    output tile     16x32                 16x16
    grid            (1, 160, 1)          (1, 320, 1)
    SMEM partials   16 KiB                8 KiB
```

The K ranges show coverage. Each SIMDgroup streams quantization groups into registers.
It does not keep its complete K slice in registers.
The diagrams use `Y = X * W^T`. The shader matrix operands compute the transposed tile, `Y^T = W * X^T`.

The following diagram shows one SIMDgroup's distributed output partial:

```text
One SG's 16x32 partial:

                    output columns
                [0,8) [8,16) [16,24) [24,32)
              +------+------+------+------+
rows [0,8)    | C00  | C01  | C02  | C03  |
              +------+------+------+------+
rows [8,16)   | C10  | C11  | C12  | C13  |
              +------+------+------+------+
                 each cell = 8x8 MMA accumulator

    RT * CT = 2 * 4 = 8 accumulator fragments
    each fragment: 64 F32 values distributed across 32 lanes
    each lane: 8 * (64 / 32) = 8 * 2 = 16 F32 partial-output values
               + input, weight, and group-dot temporaries

    8 = number of output fragments, not the MMA K dimension
    2 = output values per fragment per lane

Lifetimes:
    acc0 / acc1  -> reset for each quantization group
    out0 / out1  -> retain corrected group sums for one SG's K slice
    part         -> SMEM [projection][RT][CT][S][lane], float2 per lane
    SG0          -> sum the S partials in order -> epilogue -> GMEM output
```

The per-lane count covers one projection's persistent partial output. It is not the total register count.
Fused gate/up keeps separate partials for both projections.
Each encoded weight fragment serves all RT row tiles in its SIMDgroup.
Each input fragment serves all CT column tiles in that SIMDgroup.
The final merge uses one threadgroup barrier and no second dispatch.

### Packed operands

The following encoding converts a Q4 code into an exact BF16 or FP16 matrix operand:

```text
q = Q4 code in [0, 15]

BF16 normal value = 2^(exponent - 127) * (1 + fraction / 128)
exponent = 134    -> 128 * (1 + fraction / 128) = 128 + fraction

                 sign   exponent   fraction (7 bits)
BF16(128)          0     10000110   0000000  = 0x4300
q = 5                             0000101
0x4300 | q         0     10000110   0000101  = BF16(133)

BF16 operand bits = 0x4300 | q
FP16 operand bits = 0x5800 | (q << 3)
FP16 fraction unit at 128 = 1/8, so the shift gives exactly 128 + q

packed Q4 load -> register bit encoding -> MMA with input fragments
                                          |
original scale/bias -> F32 correction -----+-> F32 group contribution
```

The bit encoding does not add 128 numerically to each code.
The kernel loads the original scale and quantization bias. It computes the offset correction in F32.
It does not allocate a correction tensor.
This bias is a quantization parameter, not a linear layer's output bias.

Sliced-K describes the K partition. It does not require a specific weight bit width.
The `128 + q` encoding can represent unsigned Q1 through Q7 codes exactly.
The current MMA kernel implements Q4 unpacking only.
The global affine operator retains its existing Q8 and other supported bit widths.
The following initialization paths keep these contracts separate:

```text
Config::validate()
    supported bits: 2, 3, 4, 6, 8
           |
           +--> QMV / Steel
           |      retain the global bit-width domain
           |
           +--> scalar small QMM
           |      uses_small_qmm(): Q4 or Q8
           |      three-row sliced-K OR SIMDgroup-column partition
           |
           +--> sliced-K MMA
                  uses_sliced_k(): bits == 4
                         |
                         +--> selector candidate
                         +--> registry initialization
                         +--> forced Kernel initialization validation
                                      |
                                      +--> Q4 shader unpack
                                           8 codes per uint32
                                           nibble mask 0x000F000F
```

The theoretical encoding range does not add Q1, Q5, or Q7 support to `Config`.
The existing eligibility check owns the Q4 restriction. The shader adds no duplicate bit-width check.

For one input row, output column, and quantization group, the correction is:

```text
w_i  = s * q_i + bias
a2   = sum_i(x_i * (128 + q_i))
xsum = sum_i(x_i)

s * a2 + (bias - 128 * s) * xsum
  = s * sum_i(x_i * (128 + q_i)) + bias * sum_i(x_i) - 128 * s * sum_i(x_i)
  = s * sum_i(x_i * q_i) + bias * sum_i(x_i)
  = sum_i(x_i * (s * q_i + bias))
  = sum_i(x_i * w_i)
```

The scale and bias are constant within this group. They can differ across groups and output columns.
The identity holds in real arithmetic.
F32 cancellation, accumulation order, and weight rounding can change the result bits.
Steel rounds each same-dtype dequantized weight before its dot product. sliced-K MMA applies the correction after the group dot product.
Both fused paths retain storage-dtype projection rounding before F32 SwiGLU.

## Tests and benchmarks

| Test owner | Numerical/replay contract |
| --- | --- |
| Dense MLP | Gate/up -> `SiLU(gate) * up` -> down, production Q4/Q4 BF16-affine and mixed Q4/Q6 F32-affine |
| Cached replay | One total/topology, non-monotonic active prefixes, both sides of affine topology boundaries |
| Register QMM | FP16/BF16, supported groups, row/column geometries, ordinary/fused epilogues, unchanged inactive outputs |
| Steel/register references | Fixed/seeded random codes, scales, biases, scalar F64 accumulation, no shader fragments/K partitions |
| Same-dtype Steel | Materialize dequantized weights at operand dtype before dot product |
| Mixed Steel/register QMM | F32 dequantized weights |
| Fused reference | Round gate/up, F32 SwiGLU, round output |
| Host selector | Selection/topology sweep, separate from numerical tests |

Each fixture computes its CPU reference once and compares active prefixes.
Error budgets cover operand precision, output rounding, and F32 group correction.
Dense MLP tests do not use inactive scratch/output rows as an oracle.

Current Metal component bench:

```text
cargo bench --bench dense_mlp -- --profile-time 1 --noplot
```

Current Metal real-weight comparison bench:

```text
cargo bench --bench qwen35_dense_mlp -- \
  --model-dir <27b-model-dir> --tokens 1 --cases full_auto \
  --iters 1 --warmup-iters 0 --runs 1
```

The `--dump-output PATH` option writes each case output as raw BF16 after timing.
It can supply numerical evidence for kernel-policy comparisons.
Each case initializes its own fixture. Case order does not change the timed input.

The bench covers the 27B dense profile.
CLI arguments select the model path, token list, case list, iteration count, warmup count, and run count.
The bench can run the automatic full dense MLP path or focused shape-policy probes:

```text
full_auto
full_qmv_bn8_bk32
full_qmm_bm8
full_qmm_bm16
full_qmm_bm32
gate_up_swiglu_auto
gate_up_swiglu_qmv_bn8_bk32
gate_up_swiglu_qmm_bm8_bn16
gate_up_swiglu_qmm_bm16_bn16
gate_up_swiglu_qmm_bm32_bn16
down_auto
down_qmv_bn8_bk32
down_qmm_bm8_bn32
down_qmm_bm16_bn32
down_qmm_bm32_bn32
```

The default forward path is the real-weight replay path:

```text
gate_up_swiglu -> down
```

The fused stage computes `SiLU(gate) * up` without a stacked projection buffer.
The component exposes `invoke_gate_up_swiglu` and `invoke_down` for focused measurement.

The real-weight `*_auto` cases use `DenseMLP` and its normal shape-dependent policy.
`qmv_bn8_bk32` means the forced QMV BN8/BK32 kernel.
Each stage-specific `qmm` case includes its complete BM/BN tile.
A `full_qmm_bm*` case uses that BM with BN16 for fused gate/up/SwiGLU and BN32 for down.
Forced qmv/qmm cases are benchmark-only operator-policy probes.

Forced probes measure crossover policy. Production uses fused gate/up/SwiGLU.
Each row reports `backend=metal`, `command_count`, `retained_buffers`, `retained_pipelines`, and `constant_bytes`.

Recommendation: Compare the backend component bench first.
Then compare the real-weight dense MLP wrapper and the layer/layer-ladder bench.
Dense MLP scratch is reusable at model scope.
The caller must preserve the layer-boundary hidden buffer until downstream residual consumers finish.

[`executor_benchmarks.md`](executor_benchmarks.md) defines shared GPU serialization, benchmark metrics, and
performance-evidence rules.

The [component latency matrix](executor_benchmarks.md#component-latency-matrix) provides resumable `full_auto` measurements and grouped reports.
The measured input is the total row count. Request and ctx labels do not change this component input.
