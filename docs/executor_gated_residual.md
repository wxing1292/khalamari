# Gated Residual

Gated residual (GR) maintains several residual channels for each token.
`read` produces one hidden row for attention or MLP computation.
For a paired write, read also produces one write coefficient per channel.
`write` adds the component output to each original channel with a learned scalar coefficient.
The two operations record through the backend `ReplayRecorder`. They do not submit work or own request state.

## Source layout

| Owner | Source | Responsibility |
| --- | --- | --- |
| Executor core | `crates/inference-executor-core/src/gated_residual/` | Semantic geometry and CPU references. |
| Metal executor | `crates/inference-executor-metal/src/gated_residual/` | `GatedResidual`, typed read/write inputs, and reusable scratch ownership. |
| Metal backend | `crates/inference-backend-metal/src/components/gated_residual.rs` | Buffer validation, compiled kernels, and dispatch recording. |
| Metal shaders | `crates/inference-backend-metal/src/components/metal/gated_residual.metal` | General geometry, normalization, and residual write. |
| Small-row shaders | `crates/inference-backend-metal/src/components/metal/gated_residual_small.metal` | Fused norm/down and gated read for up to four recorded token rows. |
| Matrix shaders | `crates/inference-backend-metal/src/components/metal/gated_residual_matrix.metal` | BF16 matrix projections with weight reuse across token rows. |

[Qwen4x](executor_qwen4x.md) owns model composition.
GR remains independent of layer order, scheduling, page allocation, and cache lifecycle.
[Future work](future_work.md) tracks remaining model acceptance.

## Computation

Let `T` be the token count, `C` the residual channel count, `H` the hidden dimension, and `r` the low-rank dimension.
`C` corresponds to `num_branches` in the current API. Each channel contains one `H`-dimensional hidden row.
`GatedResidualCore` configures `C`, `H`, `r`, and the RMSNorm epsilon during initialization.
The worked example uses `C=4`, `H=2560`, and `r=320`.
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
Views and broadcasts do not allocate buffers. `N [T,C,H]` and projection intermediates are read scratch.
The down projection and write coefficient projection share a dispatch.
Only `a`, `R`, and the component output must remain live for write.
Read releases both scratch buffers for subsequent use.

The reference sources call these operations `mix` and `combine`.
The local API uses `read` and `write`.
See the pinned [vLLM implementation](https://github.com/vllm-project/vllm/blob/97b1b121176d405bcec861960dc605e9e41227e2/vllm/models/qwen4_exp/common/hyperconnection.py)
and [SGLang implementation](https://github.com/sgl-project/sglang/blob/954458567e10257f1d8d5ff808d2dbc74fc26967/python/sglang/srt/layers/hyperconnection.py).

## Numeric contract

The executor uses BF16 inputs, weights, outputs, and normalized residual scratch.
Small-row projection partials use F32 scratch. Matrix low-rank activations use BF16 scratch.
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

Scratch contains `[max_tokens,C,H]` normalized residual and one projection buffer.
`Config::projection_scratch_bytes(max_tokens)` sizes that buffer for all recorded shapes up to the capacity.
The buffer holds either BF16 low-rank activations or F32 channel partials and saved write gates.
For `C=4`, `H=2560`, `r=320`, capacity one needs 5200 projection bytes; capacity four needs 20800 bytes.
Both scratch buffers can be reused after read completes.
The write gate must remain unchanged until the paired write completes.
A final read-only mixer binds `write_gate: None` and requires no write weights or gate output.
This selects a read-only projection variant at record time.

`num_total_tokens` fixes the recorded grid, kernel selection, and required buffer capacity.
`num_active_tokens` is a `ReplayU32` submission parameter or fixed value in `1..=num_total_tokens`.
Inactive token groups return before buffer access or barriers. Partial matrix tiles mask inactive rows.
The containing replay owner keys its program by total capacity and other static composition facts.
Selection uses token capacity and geometry. Request roles such as prefill and decode do not select kernels.

### Projection selection

BF16 matrix projections support `C=4`, `H%32=0`, and `r%8=0`.
The small-row path also requires `H%256=0` and `r%16=0`.
Its normalized channel and low-rank SMEM allocations must each fit 32 KiB.
Its read-only down grid must have at least `H/256` groups to store all normalized stripes.
The backend establishes these capabilities during initialization.
F32 and other geometries use the general scalar projection path.

| Recorded token capacity | Eligible path | Token rows per matrix tile | Read commands |
| --- | --- | ---: | ---: |
| `1..=4` | Small-row fusion | N/A | 2 |
| `5..=16` | Matrix | 8 | 3 |
| `17..=128` | Matrix | 16 | 3 |
| `129..` | Matrix | 32 | 3 |

If small-row fusion is ineligible, matrix projections can also serve `1..=4`.
These defaults come from relative measurements on an M3 Max. They are workload selections, not model identifiers.
Checkpoint weights retain their original row-major layout.

For `T=64`, `C=4`, `H=2560`, and `r=320`, the projection paths use this data flow.
Normalization remains visible because both paths consume the same `N`.

```text
Both paths:
  R [64,4,2560] BF16 + norm [4,2560]
      -> RMSNorm over H -> N [64,10240] BF16 in GMEM

General scalar path:
  down grid [81,64], 128 threads / 4 SIMDgroups per group
    group [g,t], SIMDgroup s owns output row 4*g+s
    for k in 10240 columns:
        load N[t,k] and weight[row,k] from GMEM
        register dot -> SIMD sum -> SiLU or write gate
    write z[t,:] [320] and a[t,:] [4] BF16
    weight rows are read again for each token t

  read grid [640,64], 128 threads / 4 SIMDgroups per group
    group [g,t], SIMDgroup s owns hidden column h=4*g+s
    register mixed = 0
    for c in 4 channels:
        GMEM z[t,:] [320] and up[c,h,:] [320] -> register dot
        SIMD sum -> sigmoid * N[t,c,h] -> mixed
    write x[t,h] = mixed/4

Matrix path, token tile = 16 rows:
  down grid [41,4], 256 threads / 8 SIMDgroups per group
    group [g,m] owns N[16*m:16*m+16,:] and projection rows [8*g,8*g+8)
    SIMDgroup s owns K [1280*s,1280*s+1280)
    register accum [2,8,8] FP32 = 0 per SIMDgroup
    for k in its K range, step 8:
        GMEM weight [8,8] -> registers once
        for t in 2 token fragments:
            GMEM N [8,8] -> registers
            BF16 MMA -> accum[t]                 // reuse the weight fragment
    SMEM partial [8,2,8,8] FP32 -> sum over 8 SIMDgroups
        -> SiLU / write gate -> z [64,320], a [64,4] BF16 in GMEM

  read grid [80,4], 128 threads / 4 SIMDgroups per group
    group [g,m] owns tokens [16*m,16*m+16), hidden columns [32*g,32*g+32)
    SIMDgroup s owns 8 hidden columns across all 4 channels
    register accum [2,4,8,8] FP32 = 0 per SIMDgroup
    for k in 320 rank columns, step 8:
        GMEM z [2,8,8] -> registers
        for output fragment in 4 fragments:
            GMEM up [8,8] -> registers once
            for t in 2 token fragments:
                BF16 MMA -> accum[t,output]       // reuse z and weight fragments
    sigmoid * N -> SIMD channel sum -> x [64,2560] BF16 in GMEM
```

The matrix path loads each weight tile for 16 token rows instead of one row.
The down kernel partitions the long reduction across eight SIMDgroups and merges their partials in SMEM.
The read kernel keeps its partials in registers and does not stage weights in SMEM.

Small-row fusion removes a dispatch and exposes channel-level parallel work.
For `T=1` with the same geometry, it uses this data flow:

```text
down grid [41,4,1], 256 threads / 8 SIMDgroups per group
  group [g,c,0]:
    R[0,c,:] [2560] -> register values[10] per thread
    RMSNorm -> SMEM N_c [2560] BF16
    SIMDgroup s reads down/write_weight[8*g+s,c,:] [2560]
    register dot -> SIMD sum -> partial[c,0,8*g+s] FP32 in GMEM
    group g<10 stores a disjoint 256-column stripe of N[0,c,:]

read grid [160,1], 256 threads / 8 SIMDgroups per group
  group [g,0]:
    GMEM partial[:,0,:] [4,320] -> sum channels -> SiLU -> SMEM z [320] BF16
    for each of 16 hidden columns h owned by the group:
        four channels, four rank slices per channel
        SMEM z and GMEM up[c,h,:] -> register dot -> SIMD rank sum
        sigmoid * N[0,c,h] -> SIMD channel sum -> x[0,h] BF16 in GMEM
  group g=0 also merges four write-gate columns -> a[0,:] [4] BF16
```

Norm work is repeated across the down output groups. This avoids a separate norm launch.
The read groups reuse their local `z` across 16 hidden columns.

### Adjacent write/read fusion

Write alone records one weighted residual-add command.
The backend recorder can fuse an adjacent write with the next read.
For scalar and matrix projections, one command stores the updated residual and normalizes it.
For small-row projections, down computes the updated values before norm but does not store the residual.
It saves the old write gate in projection scratch. The read command then stores the updated residual.
This barrier between commands lets every down group finish reading the old residual and component output.
Both variants preserve the BF16 storage boundary before RMSNorm.

The next read supplies its own norm weights, epsilon, and projection rank.
Fusion requires equal channel count, hidden dimension, dtype, total capacity, and active binding.
The write output must be the next read input.
The backend rejects aliases that would overwrite live write inputs or next-read weights before their last use.
For small rows, this includes projection-scratch reuse and deferred residual writes.
Incompatible operations remain separate. These checks run when replay is recorded.

```text
Scalar / matrix:
  read_i -> component y -> [weighted add + next norm] -> next down -> next read

Small rows:
  read_i -> component y -> [next norm + down, save old gate]
                              -> [next read + deferred weighted add]
```

Each compatible boundary removes one dispatch and one consumer barrier.
Fusion belongs to the backend recorder. The executor has no fusion policy or tuning switch.
Projections load four contiguous elements when the reduction dimension is divisible by four.
Other dimensions use scalar loads. This choice specializes during initialization and does not change replay topology.
Consumer barriers separate dependent commands and scratch reuse.
All weights and referenced buffers remain resident through replay completion under the backend replay contract.

## Verification and measurement

| Check | Coverage |
| --- | --- |
| CPU | Hand-derived read/write examples |
| Metal parity | Seeded F32/BF16, unaligned/checkpoint geometry, in-place write, inactive residuals |
| Error bounds | F32 max `2e-5`, BF16 max `0.03125`, mean below one tenth of max tolerance |
| Fusion | Same arithmetic/output, next-read CPU parity, different rank/epsilon |
| Scratch lifetime | Different active counts, output-buffer reuse as norm scratch, both buffers reused before write |
| Tiled kernels | Capacities `1`, `4`, `5`, `16`, `17`, `128`, `129`, partial active rows |
| Replay | Total four, active `[1,4,3,2]` |
| Executor | Read -> write -> final read, shared scratch, production recorder |

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
| Chain | Unfused commands | Fused commands |
| --- | ---: | ---: |
| Small rows | 98 | 66 |
| Scalar/matrix | 131 | 99 |

`chain_unfused` and `chain_fused` have identical arithmetic/outputs.
Amortized time includes one thirty-second of the final read.
Follow the shared [verification and performance rules](executor_benchmarks.md).

```sh
cargo bench -p inference-backend-metal --bench gated_residual -- \
  --warm-up-time 1 --measurement-time 2 --sample-size 20 --noplot
```

This module does not establish whole-model correctness, real-checkpoint parity, or model throughput.
