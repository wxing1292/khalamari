# Gated Residual

Gated residual (GR) maintains several residual channels for each token.
`read` produces one hidden row for attention or MLP computation.
For a paired write, read also produces one write coefficient per channel.
`write` adds the component output to each original channel with a learned scalar coefficient.
The two operations record through the executor `Recorder`. They do not submit work or own request state.

## Source layout

| Owner | Source | Responsibility |
| --- | --- | --- |
| Executor core | `crates/inference-executor-core/src/gated_residual/` | Semantic geometry and CPU references. |
| Metal executor | `crates/inference-executor-metal/src/gated_residual/` | `GatedResidual`, typed read/write inputs, and reusable scratch ownership. |
| Metal backend | `crates/inference-backend-metal/src/components/gated_residual.rs` | Buffer validation, compiled kernels, and dispatch recording. |
| Metal shaders | `crates/inference-backend-metal/src/components/metal/gated_residual.metal` | Normalization, gated read, and weighted residual write. |

The module is independent of model layer order, runtime scheduling, page allocation, and cache lifecycle.
Qwen3.8 model integration remains in [future work](future_work.md).

## Computation

Let `T` be the token count, `C` the residual channel count, `H` the hidden dimension, and `r` the low-rank dimension.
`C` corresponds to `num_branches` in the current API. Each channel contains one `H`-dimensional hidden row.
`GatedResidualCore` configures `C`, `H`, `r`, and the RMSNorm epsilon during initialization.
The current Qwen3.8 reference configuration uses `C=4`, `H=2560`, and `r=320`.
The module also supports other positive dimensions, including unaligned dimensions.

All matrices use contiguous row-major storage. Weight shapes follow checkpoint output-by-input order.
The equations below use one token and column vectors for the flattened projections.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `R` | `[C,H]` | Original residual. |
| `N` | `[C,H]` | Independently normalized channels. |
| `norm` | `[C,H]` | Raw Gemma RMSNorm affine offsets. The multiplier is `1 + norm`. |
| `down` | `[r,CH]` | Read gate down projection. |
| `up` | `[CH,r]` | Read gate up projection. |
| `write_weight` | `[C,CH]` | Write coefficient projection. |
| `x`, `y` | `[H]` | Component input and component output. |

The arithmetic is:

```text
N[c,h] = R[c,h] / sqrt(mean_h(R[c,h]^2) + eps) * (1 + norm[c,h])
z      = SiLU(down @ flatten(N) / C)
G      = reshape(sigmoid(up @ z), [C,H])
x[h]   = sum_c(G[c,h] * N[c,h]) / C
a      = 2 * sigmoid(write_weight @ flatten(N) / C)

y      = component(x)
R'[c,h] = R[c,h] + a[c] * y[h]
```

The read gate has one coefficient per channel element.
The write gate has one coefficient per channel and broadcasts across `H`.
Read divides by `C`, not by the sum of the gates.
Write preserves the original, unnormalized residual.

Here, `norm` is Gemma RMSNorm applied independently to each residual channel.
Qwen3.8 uses GR read in place of separate attention and MLP input RMSNorms.
Read output goes directly into the attention or MLP component.
The pinned [SGLang model setup](https://github.com/sgl-project/sglang/blob/954458567e10257f1d8d5ff808d2dbc74fc26967/python/sglang/srt/models/qwen4_exp.py#L1285-L1291)
removes `input_layernorm` and `post_attention_layernorm`.
Internal attention norms, such as Q/K normalization and GDN output norm, keep their separate roles.

Projection arrows use checkpoint weights `[output, input]`: `X @ weight.T`.

```text
READ: normalize each channel
  R [T,C,H] -- RMSNorm over H, norm [C,H] --> N [T,C,H]

READ: compute gates from N_flat [T,C*H], a view of N
  N_flat [T,C*H] --+-- down [r,C*H] --> d [T,r]
                  |                       |
                  |                   /C then SiLU
                  |                       v
                  |                    z [T,r]
                  |                       |
                  |                  up [C*H,r]
                  |                       v
                  |                   u [T,C*H] -- sigmoid/view --> G [T,C,H]
                  |
                  +-- write_weight [C,C*H] --> w [T,C]
                                                  |
                                              /C then 2*sigmoid
                                                  v
                                               a [T,C]

READ: mix the channels
  N [T,C,H] --+
             +-- multiply --> NG [T,C,H] -- mean over C --> x [T,H]
  G [T,C,H] --+

COMPONENT
  x [T,H] -- attention / MoE --> y [T,H]

WRITE: broadcast a across H and y across C
  a [T,C] -- view [T,C,1] --+
                          +-- multiply --> update [T,C,H] --+
  y [T,H] -- view [T,1,H] --+                                |
                                                           +-- add --> R' [T,C,H]
  R [T,C,H] -----------------------------------------------+           (may overwrite R)
```

Shapes describe mathematical values. `d`, `u`, `w`, `G`, `NG`, and `update` stay inside kernels.
Views and broadcasts do not allocate buffers. `N [T,C,H]` and `z [T,r]` are read scratch.
Both projections from `N` execute in one dispatch.
Only `a`, `R`, and the component output must remain live for write.
Read releases `N` and `z` for subsequent scratch use.

The reference sources call these operations `mix` and `combine`.
The local API uses `read` and `write`.
See the pinned [vLLM implementation](https://github.com/vllm-project/vllm/blob/97b1b121176d405bcec861960dc605e9e41227e2/vllm/models/qwen4_exp/common/hyperconnection.py)
and [SGLang implementation](https://github.com/sgl-project/sglang/blob/954458567e10257f1d8d5ff808d2dbc74fc26967/python/sglang/srt/layers/hyperconnection.py).

## Numeric contract

The executor uses BF16 inputs, weights, outputs, and scratch.
The backend also supports F32 for component use and numerical verification.
Normalization, dot products, and channel sums accumulate in F32.

BF16 execution rounds at the eager tensor-operation boundaries.
These boundaries include projection outputs, division by `C`, activations, gate products, and final outputs.
Norm computes `1 + norm` in F32 before it stores its result.
The CPU reference models these rounding points independently of the Metal dispatch structure.
Parallel reduction order can still produce small numerical differences.

Checkpoint weights use their original layout and affine offsets.
This module does not own checkpoint file loading or quantization conversion.

## Ownership and replay

`GatedResidual` owns immutable compiled computation.
The caller owns weights, residual buffers, hidden buffers, and `GatedResidualScratch`.
Read returns the hidden output buffer.
A paired read binds `Some(WriteGate { weight, output })` to store coefficients in a caller-owned `[T,C]` buffer.
The write input can use an independently computed hidden output from attention or MLP.

Read requires its outputs and scratch buffers to be disjoint from each other and from its inputs and weights.
Write can overwrite its original residual in place.
Its output must not alias the gate or component output.
Each invocation validates buffer lengths against its recorded capacity.

Scratch contains `[max_tokens,C,H]` normalized residual and `[max_tokens,r]` low-rank activations.
Both scratch buffers can be reused after read completes.
The write gate must remain unchanged until the paired write completes.
A final read-only mixer binds `write_gate: None` and requires no write weights or gate output.
This selects a read-only down-projection kernel at record time.

`num_total_tokens` fixes the recorded grid and required buffer capacity.
`num_active_tokens` is a `ReplayU32` submission parameter or fixed value in `1..=num_total_tokens`.
Inactive token threadgroups return before buffer access or barriers.
Read and write use one static topology. They have no token-count kernel crossover.
The containing replay owner keys its program by total capacity and other static composition facts.

Read records three commands: channel normalization, down projection with SiLU and optional write coefficients, and up projection with gated reduction.
Write records one command: weighted residual addition.

The backend recorder fuses an adjacent write and the next read's channel normalization.
The fused command stores both the updated residual and the next normalized residual.
It rounds the residual to its storage dtype before RMSNorm, as the separate commands do.
The next read supplies its own norm weights and epsilon. Its projection rank can differ.
Fusion requires equal channel count, hidden dimension, dtype, total capacity, and active binding.
The write output must be the next read input.
The next normalized scratch must not alias the write gate or component output.
The next norm weights must not alias the write output.
These checks prevent cross-threadgroup races when the separate operations reuse dead inputs.
Incompatible operations remain separate. This follows the existing residual-add/RMSNorm recorder contract.

```text
read_i                        component       write_i + read_(i+1)
---------------------------   ---------       --------------------------------
norm -> down/SiLU + a -> read -> y -> [weighted add + next norm] -> down -> read
          kernel 2          kernel 3          one fused kernel
```

Each compatible boundary removes one dispatch and one consumer barrier.
Fusion belongs to the backend recorder. The executor has no fusion policy or tuning switch.
Projections load four contiguous elements when the reduction dimension is divisible by four.
Other dimensions use scalar loads. This choice specializes during initialization and does not change replay topology.
Consumer barriers separate dependent commands and scratch reuse.
All weights and referenced buffers remain resident through replay completion under the backend replay contract.

## Verification and measurement

CPU tests use hand-derived read and write examples.
Metal tests use seeded fixtures and compare F32 and BF16 results with the CPU reference.
The generator and maximum/mean error checks follow the existing dense-MLP and GDN tests.
F32 uses a maximum absolute tolerance of `2e-5`. BF16 uses `0.03125`.
Mean absolute error must remain below one tenth of the maximum tolerance.
This accounts for intermediate BF16 rounding and cancellation while limiting aggregate error.
They cover non-aligned dimensions, checkpoint geometry, in-place write, and inactive residual preservation.
Fusion tests compare fused and separate commands, then check the next read against the CPU reference.
They cover different next-read rank and epsilon.
Two separate-operation execution cases cover different active counts and reuse of the component-output buffer as normalized scratch.
A separate test reuses normalized and low-rank scratch before the paired write.
Replay tests use total capacity four and active counts `[1,4,3,2]`.
The executor test composes read, write, and a final read with shared scratch through the production recorder.

Run focused checks from the repository root:

```sh
cargo test -p inference-executor-core --all-features --lib gated_residual
cargo test -p inference-backend-metal --all-features --lib gated_residual -- --test-threads=1
cargo test -p inference-executor-metal --all-features --lib gated_residual -- --test-threads=1
```

The backend `gated_residual` benchmark measures read, write, their composition, and dependent chains at `C=4`, `H=2560`, and `r=320`.
It uses synthetic BF16 weights and token counts `1`, `4`, `16`, and `64`.
Each submission contains 32 invocations. Divide the reported submission time by 32 for the amortized component time.
The write benchmark initializes its inputs through read before measurement.
The composition benchmark uses the read output directly as the write input and excludes attention or MLP computation.
The `read_write` case measures 32 independent pairs and retains the pre-fusion comparison workload.
The chain cases record 32 dependent pairs and one final read, with shared scratch and a shared gate buffer.
The first pair reads an immutable residual, so each submission starts from the same input.
The raw chain uses 131 commands. Recorder fusion reduces this to 99 commands.
`chain_unfused` and `chain_fused` compare identical arithmetic and outputs.
Their amortized times include one thirty-second of the final read.
Follow the shared [verification and performance rules](executor_benchmarks.md).

```sh
cargo bench -p inference-backend-metal --bench gated_residual -- \
  --warm-up-time 1 --measurement-time 2 --sample-size 20 --noplot
```

This module does not establish whole-model correctness, real-checkpoint parity, or model throughput.
