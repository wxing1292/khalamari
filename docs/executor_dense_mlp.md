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
  core.rs      DenseMLPCore + DenseMLPReplayShape

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
crates/inference-executor-core/src/backend/
  Recorder
```

The current runtime path is the Metal replay path in
`crates/inference-executor-metal`.

Reusable Metal dense MLP kernels live in:

```text
crates/inference-backend-metal/src/components/dense_mlp.rs
crates/inference-backend-metal/src/components/dense_mlp_test.rs
crates/inference-backend-metal/src/operators/affine_quantized.rs
crates/inference-backend-metal/src/operators/metal/affine_quantized_gate_up_swiglu.metal
crates/inference-backend-metal/src/operators/metal/affine_quantized_gate_up_swiglu_steel.metal
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
Qwen model and layer code use `Recorder` to append dense MLP work to a larger whole-layer or whole-model replay.
Focused tests and benches build replay programs from the same recorder path.
The internal order is `gate_up_swiglu -> down [barrier before]`.
Model and layer wiring own barriers on the first consumer command and downstream residual consumers.

## Replay contract

`DenseMLP` records one dense gated MLP forward into a caller-owned `Recorder`.
It does not submit commands.
It does not own tensor storage or request lifecycle.
The semantic layer input is
`DenseMLPInput { num_total_tokens, num_active_tokens, hidden_state, next_hidden_state, scratch, weights }`.
Replay returns the caller-owned `next_hidden_state` buffer directly.

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

The shared `Qwen3xDenseMLP` leaf directly owns immutable weights and per-layer output buffers.
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
The DSpark path supplies equal active and total counts.

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
QMV and QMM assign gate and up to separate SIMDgroups.
Each SIMDgroup loads weights and accumulates only its assigned projection.
Both paths transfer their rounded projection values through threadgroup memory.
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
QMM uses Steel `BlockMMA` with F32 accumulators and 8x8 SIMDgroup matrix operations.
`operators/metal/affine_quantized_qmm_steel.metal` owns ordinary QMM.
`operators/metal/affine_quantized_gate_up_swiglu_steel.metal` owns fused gate/up/SwiGLU.
Both use BK32. Each block loads an input tile and dequantizes a weight tile into SMEM.
The fused kernel reuses the input tile across gate and up. The projections reuse one weight staging region.
Both projection accumulators remain in REGISTER until the fused activation.
The kernel rounds each projection to its storage dtype before it computes SwiGLU in F32.
Same-dtype operands retain storage-dtype rounding. Mixed-dtype operands use F32. QMV remains the small-row path.
Full tiles use unchecked loaders. Partial tiles retain active-row and output-column masks.
The model and executor provide the complete dense-MLP dimensions and active row count.
They do not select a kernel or tile.

Large dense MLPs use this base policy when `hidden_dim > 4096` or `intermediate_dim > 4096`:

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

The selector exposes the 2048-row topology boundary to replay bucketing.
Gate/up and down evaluate eligibility independently.

Smaller dense MLPs keep QMV for a longer range.
The first QMM row count is 18 when both dimensions are at most 2048.
The first QMM row count is 12 for the remaining smaller shapes.
After this crossover, the backend uses BM16 through 16 rows and BM32 for larger row counts.
The output tile remains BN16 for fused gate/up/SwiGLU and BN32 for down.

Gate/up and down apply the same backend selector independently.
They can share a family when their dimensions select the same candidate.
For group size 64, the 8-row BF16 BM8/BN32 kernel uses 64 threads and 5760 bytes of static threadblock memory.
The memory contains the `8 × 72` input tile and the `32 × 72` weight tile.
The stride is `BK` plus 16 bytes of padding. BF16 and F16 operands normally use `BK=min(group_size, 64)`.
For BM8 and BM16 with at least 65,536 weight rows, the backend selects BK32 at initialization.
F32 operands always use BK32. The same selection sets the shader constant and validates pipeline scratch.
The 27B dense MLP retains BK64 for group size 64. Accumulators remain F32 in each case.
The fused QMM shares one input tile between gate and up in each K iteration.
Each SIMDgroup keeps one F32 cooperative accumulator live.
The fused QMM reuses its weight scratch for output-dtype projection values after the K loop.
A threadgroup barrier separates the last matrix read from the scratch reuse. Another barrier separates projection
stores from SwiGLU reads. It needs no separate projection allocation.
The fused BM8/BN16 kernel uses 5760 bytes for the group-size-64 BF16 layout.
In fused QMV, one SIMDgroup computes all gate columns in the output tile. The other computes the matching up columns.
Each thread keeps one weight/scales/biases pointer set and one accumulator array.
Initialization selects an aligned specialization when both output columns and the K dimension contain complete tiles.
That specialization omits tail checks from the reduction loop.
Kernel initialization checks the SIMD width, pipeline thread limit, calculated threadblock memory, reported static
threadblock memory, and device threadblock-memory limit.
Initialization also validates the kernel/epilogue pairing. Tile width does not imply an activation or scratch layout.

Benchmark-only QMV/QMM probes select an affine kernel policy for measurement.
The semantic data flow stays the same.

## Tests and benchmarks

The focused backend test records each dense MLP topology into an isolated test cache.
It replays non-monotonic active-token sequences and compares each active output prefix with the CPU quantized
dense-MLP reference.
The reference covers the gate/up projection, `SiLU(gate) * up`, and the down projection as one numerical contract.
It accepts separate group sizes and bit widths for the gate/up and down projections.
The test covers the production Q4/Q4 BF16-affine layout and a mixed Q4/Q6 F32-affine layout.
It records capacities on each side of the affine topology boundaries.
It does not use inactive output or scratch rows as an oracle.
Affine operator tests own kernel selection and topology-boundary contracts.

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

They help select the correct production threshold.
They are not separate production paths.
The full production path uses the fused gate/up/SwiGLU operator.

The real-weight bench prints replay metadata with each perf row:

```text
backend
command_count
retained_buffers
retained_pipelines
constant_bytes
```

The Metal stream backend name supplies `backend`.
The expected value is `backend=metal`.

Recommendation: Compare the backend component bench first.
Then compare the real-weight dense MLP wrapper and the layer/layer-ladder bench.
Dense MLP scratch is reusable at model scope.
The caller must preserve the layer-boundary hidden buffer until downstream residual consumers finish.

[`executor_benchmarks.md`](executor_benchmarks.md) defines shared GPU serialization, benchmark metrics, and
performance-evidence rules.
