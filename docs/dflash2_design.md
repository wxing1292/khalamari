# Qwen3x DFlash2 Design

DFlash2 produces a fixed block of proposals with one transformer forward, then samples a path through candidate tokens.
Its layers use dynamic convolution. Its selector conditions each proposal on the previously selected token.
This document describes the current Qwen3.5 implementation.

Use [Proposal block and attention](#proposal-block-and-attention) for row counts and [End-to-end flow](#end-to-end-flow) for execution order.
[`executor_gqa.md`](executor_gqa.md) owns shared GQA details. [`executor_sampling.md`](executor_sampling.md) owns shared sampling contracts.

## Scope and ownership

DFlash2 owns an independent Spec model role.

| Owner | Responsibility |
| --- | --- |
| Runtime core | Scheduling, request lifecycle, physical pages, and page IDs |
| Main | Token embedding, transformer execution, selected-layer residual capture, unembedding, Main sampling, and rejection |
| DFlash2 | Checkpoint, history page table, replay caches, workspaces, dynamic convolution, and selector |
| Qwen executor | Compose Main and DFlash2 recordings, submit the ordered sequence, wait, and adapt results |

DFlash2 reuses Main Embed, Main Unembed, GQA, dense MLP, raw Top-K, and sparse rejection.
It has no confidence head, so the response adapter returns `1.0` for each proposal token.

## Proposal block and attention

The runtime proposal length `N` is fixed at startup.
The service uses checkpoint `block_size - 1` as the default. `--num-spec-tokens N` overrides this default.
The executor keeps the checkpoint unchanged.

| Symbol | Meaning in this document |
| --- | --- |
| `N` | Proposals per Decode request |
| `p` | Absolute position of the newly sampled Main anchor |
| `t` | Query row index, from `0` through `N`. Proposal positions use `1` through `N`. |
| `x_0` | Main anchor at position `p` |
| `x_t` | Proposal token at position `p + t`, for `1 <= t <= N` |

The query block has one anchor row and `N` MASK rows.
Only MASK rows produce proposals. For `N = 3`, the row-to-token relation is:

```text
+--------------------+----------+----------+----------+----------+
| Query row t        | 0        | 1        | 2        | 3        |
+--------------------+----------+----------+----------+----------+
| Query position     | p        | p + 1    | p + 2    | p + 3    |
| Input token        | x_0      | MASK     | MASK     | MASK     |
| Gathered for output| no       | yes      | yes      | yes      |
| Produced proposal  | -        | x_1      | x_2      | x_3      |
| Proposal position  | -        | p + 1    | p + 2    | p + 3    |
+--------------------+----------+----------+----------+----------+
```

| Work domain | Rows per request |
| --- | --- |
| Embedding, attention, and dynamic convolution | `N + 1` |
| Output gathering, selection, and draft distributions | `N` |

The DFlash2 query block must be smaller than `sliding_window`.
Each layer reduces SplitKV history partials with bidirectional local-block SDPA partials.

DFlash2 stores all committed history and reads this half-open range for each query:

```text
[max(0, query_position + 1 - sliding_window), anchor_position)
```

Here, `anchor_position = p` and `query_position = p + t`.
The history start can differ between rows in the same block.
The local block contains the anchor and all MASK K/V with bidirectional attention.
Its K/V is temporary.

The Spec Decode replay key contains padded history TaskTemplate capacity.
The active count remains a submission argument, so matching padded capacities reuse one replay.

## End-to-end flow

Spec Prefill converts captured Main features into persistent history K/V.
Spec Decode uses that history and the anchor-plus-MASK block to produce new proposals.
The two stages use independent replay recordings within one ordered GPU submission:

```text
Main Embed -> Main forward and selected-layer capture
  -> GatherUnembed -> RejectionSampling
  -> Spec Decode prepare: write anchor, positions, and visible history ranges
  -> DFlash2 Prefill: project captured Main rows -> history K/V pages
  -> DFlash2 Embed -> all DFlash2 layers -> final RMSNorm
  -> gather MASK rows -> Main Unembed -> raw Top-K
  -> score candidate edges -> sequential path sampling

CPU boundary: submit sequence -> wait for completion -> read decision + proposals

Prefill-only: Main Embed -> Main forward/capture -> DFlash2 Prefill
```

The Qwen executor owns recording, submission, and wait boundaries.
There is no CPU rejection read between Main verification and Spec Decode.
A prefill-only batch creates history but has no sampled anchor and produces no proposals.

Each selected Main layer writes every Main row directly into its assigned capture columns.
The Main-feature projector stores the column range for each selected layer at initialization.
Main replay recording uses this mapping through `MainResidualCapture`.
Spec Prefill borrows the active token count, request slots, and flat token indices from the current Main GQA metadata.
It persists every captured Main row, including the rejected physical suffix.
Logical commit exposes only fixed Main rows and the accepted speculative prefix.

For example, Main verifies one known token `w` and three old drafts `d1, d2, d3` at positions `b` through `b + 3`.
Rejection sampling returns a new anchor `y`:

| Accepted old drafts | New anchor position `p` | Newly committed history rows |
| --- | --- | --- |
| None | `b + 1` | `w` |
| `d1` | `b + 2` | `w, d1` |
| `d1, d2, d3` | `b + 4` | `w, d1, d2, d3` |

In each case, DFlash2 starts its new block with `x_0 = y` at `p`.
The history upper bound excludes the rejected suffix. Each query's sliding lower bound can exclude older committed rows.

## Layer composition

Each layer applies the attention branch, then the dense MLP branch.
Each branch normalizes its residual input and derives both convolution kernels from that normalized input.
The post-convolution reuses those coefficients after attention or MLP completes.

The data and coefficient paths are:

```text
Data:
  residual input -> RMSNorm -> h_t -> pre-convolution
    -> attention or dense MLP -> post-convolution -> add residual input
    -> residual output

Coefficients:
  h_t -> kernel projection -> pre and post coefficients
    + BF16 base kernels -> K_pre(t), K_post(t)

Layer order:
  attention branch -> MLP branch -> next layer
```

Each branch has one kernel projection and BF16 base kernel.
For branch side `s` in `{pre, post}`, the effective kernel is:

```text
K_s(t) = K_base_s + Delta K_s(h_t)
```

In this section, `h_t` is the normalized input row of the current branch.
The convolution reads current and earlier rows in one request-local block.
It never combines requests.
The model applies final RMSNorm after all DFlash2 layers.

## Candidate selection

Main Unembed produces unary logits for each MASK row.
Raw Top-K selects `selector_top_k` candidates `C_t` per position.
All candidate sets and edge scores exist before the sequential path walk starts.
The walk does not run the DFlash2 transformer again for each token.

| Symbol | Value at proposal position `t`, from `1` through `N` |
| --- | --- |
| `h_t` | Final DFlash2 hidden output for MASK row `t` |
| `a` | Previously selected token. At `t = 1`, this token is anchor `x_0`. |
| `b` | Candidate token in `C_t` |
| `A`, `B` | Predecessor and successor codebooks |
| `H` | Hidden projection |
| `U_t(b)` | Unary logit for candidate `b` |
| `S_t(a,b)` | Selector score for the edge from `a` to `b` |

The selector adds the learned edge correction to each unary logit:

```text
S_t(a,b) = U_t(b) + <A(a) * H(h_t), B(b)>
```

It scores the anchor-to-candidate edges for the first position and all top-k-to-top-k edges between later adjacent positions.
The selected predecessor determines which score row the next step uses:

```text
x_0 --[scores S_1(x_0, b), b in C_1]--> sample x_1
x_1 --[scores S_2(x_1, b), b in C_2]--> sample x_2
x_2 --[scores S_3(x_2, b), b in C_3]--> sample x_3
```

Each step applies request temperature and samples its conditional distribution `q_t(. | x_{t-1})`.
It applies no post-selection top-p.
The output contains the selected tokens, their probabilities, and sparse draft distributions for Main rejection sampling.

## Cache and lifecycle

Runtime core supplies one page-ID list for each logical cache block.
Main and DFlash2 history share cache lane 0, which the executor splits once:

```text
[Main page IDs | DFlash2 history page IDs]
```

Main and DFlash2 use separate page tables with one request-slot lifecycle.
A reset clears both bindings, while runtime core retains physical-page ownership.

Persistent state contains the request page table and history K/V pages.
Snapshots use `dflash2-gqa-request-page-table`.
Local Q/K/V, attention partials, convolution coefficients, candidates, and selector output are ephemeral.

`Qwen3xDFlash2Prefill` and `Qwen3xDFlash2Body` share one model through `load_model` and `unload_model`.
`Qwen3xDFlash2Execution::unload_state` collects these handles before it releases the model's state bindings.
After state restoration, `load_state` restores the bindings and shares the model with both replay owners.

## Checkpoint contract

`Qwen3xDFlash2Config` adapts `DFlash2DraftModel` to the flat canonical schema.
It validates Main compatibility, selected layers, block and window sizes, attention and convolution geometry, selector
rank, and candidate count.
The checkpoint boundary rejects unknown architectures and nested fields.

`Qwen3xDFlash2WeightBindings` accepts only the exact source or affine manifest.
The manifest cannot replace reused Main embedding or unembedding.
Each projection resolves its exact affine layout.

The shared Spec converter writes packed `U32` matrices and BF16 affine parameters.
It keeps RMSNorm weights and convolution base kernels as BF16.
The loader also accepts uniform F32 affine scales and biases.
The default conversion uses group size 64 and 4-bit matrices.
Layer 2 and layer 4 `v_proj` and `down_proj` use 6-bit matrices.
Use [`service.md`](service.md) for conversion commands.

## Key source layout

```text
crates/inference-executor-core/src/model/qwen/v3_x/dflash2/
  config.rs                  checkpoint schema and validation
  weight_layout.rs           exact source and affine manifests

crates/inference-executor-core/src/bin/qwen3x_spec_quantize/
  dflash2.rs                 DFlash2 conversion policy

crates/inference-executor-metal/src/model/qwen/v3_x/dflash2/
  execution.rs               Prefill and Decode orchestration
  main_feature.rs            all-row projection from selected Main layers
  attention.rs               history-plus-block attention
  conv.rs                    dynamic grouped convolution
  layer.rs                   DFlash2 layer composition
  model.rs                   model and replay owners
  output.rs                  Top-K, selector, and draft distributions

crates/inference-executor-metal/src/model/qwen/v3_5/executor/
  dflash2.rs                 Qwen3.5 outer integration

crates/inference-backend-metal/src/components/
  dynamic_grouped_conv.rs
  sampling/dflash2_selector.rs
```

## Verification

Focused tests cover config adaptation, manifests, affine layouts, sliding ranges, convolution and selector parity,
replay active counts, service modes, and speculator mutual exclusion.

Use [`service.md`](service.md) for end-to-end commands.
Use [`executor_benchmarks.md`](executor_benchmarks.md) before a performance claim.
