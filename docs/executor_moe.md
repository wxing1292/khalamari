# MoE Executor

This document describes the current MoE implementation.
It covers routing, execution-variant selection, expert execution, shared-expert composition, tests, and production
benchmarks.

## Source layout

`crates/inference-executor-core` intentionally has no MLX or Metal dependency.
It owns backend-neutral MoE layer metadata.
`crates/inference-executor-metal` owns the current Metal replay backend.
MoE is the public semantic boundary.
The low-level sparse expert MLP remains an inner component of the MoE path.

```text
crates/inference-executor-core/src/mlp/moe/
  mod.rs      semantic MoE component boundary exports
  core.rs      GatedMoECore + GatedMoEReplayShape

crates/inference-executor-metal/src/mlp/moe/
  mod.rs      Metal MoE module root
  backend.rs   GatedMoEMetalConfig + GatedMoE
  backend_test.rs
               MoE owner API-set and replay-path tests
  scratch.rs   routing, top-k-expert, and optional shared-expert scratch ownership and bindings

crates/inference-executor-metal/src/model/qwen/
  v3_x/layer/moe.rs  Qwen3xMoE, private checkpoint weights, load, and record
  v3_5/main/layer.rs Qwen3.5 Main dense-MLP/MoE layer variants
  v3_5/mtp/layer.rs  Qwen3.5 MTP dense-MLP/MoE layer variants
  v3_5/component_config.rs
                      Qwen3.5 MoE geometry/config builder

crates/inference-executor-core/src/def/
  DenseLinearShape
  SparseLinearShape
```

The current runtime path is the Metal replay path in
`crates/inference-executor-metal`.

Reusable Metal MoE / sparse expert kernels live in:

```text
crates/inference-backend-metal/src/components/
  moe/
    mod.rs
    routing.rs
    routing_test.rs
    expert_major.rs
    expert_major_test.rs
    combine.rs
    combine_test.rs
  sparse_mlp.rs
  sparse_mlp_test.rs
```

## Shape model

`GatedMoECore` owns immutable layer metadata:

```text
model_layer_index
hidden_dim
intermediate_dim
shared_experts_intermediate_dim (optional and independent from routed expert intermediate_dim)
num_experts
num_experts_per_token
norm_topk_prob
```

It derives MoE / sparse expert shapes:

```text
router_shape
gate_shape
up_shape
down_shape
```

`GatedMoE` owns the semantic MoE replay contract:

```text
router projection -> route top-k -> dispatch -> sparse expert MLP -> combine/scatter
```

`GatedMoE` owns the token-major and compact expert-major implementations.
It selects one implementation from the active replay shape.
The backend implements the executor `ReplayLayer` contract.
Qwen model and layer code use this contract to append MoE work to a larger e2e replay.
The contract has one semantic input, one output, and a caller-owned recorder.
The semantic replay input can include a shared-expert branch:

```text
router projection -> route top-k -> dispatch -> sparse expert MLP
shared expert + shared gate
combine/scatter with shared contribution
```

Current MoE replay records dispatch/layout as part of the selected token-major or expert-major execution variant.
The scheduler does not own a separate dispatch phase.
Model and layer wiring treat the full MoE MLP as one component boundary in a larger layer/model ICB.
The MoE backend records internal barriers where commands have RAW dependencies.
These barriers separate router projection and routing, dispatch and layout, expert compute, shared-expert work, and
combine/scatter.

`GatedMoEMetalConfig` keeps expert quantization bits separate from router and shared-gate
quantization bits:

```text
bits              top-k expert and shared-expert MLP projections
router_bits       MoE router projection
shared_expert_gate_bits  shared-expert gate projection
```

Qwen3.6-35B-A3B uses 4-bit expert MLPs and 8-bit router/shared-gate tensors through config quantization overrides.
The Qwen component geometry helper resolves these overrides during semantic load.
Benches must not assume one global bit width for all projections in a MoE layer.

The router and shared-gate projections each use one adaptive affine operator.
`GatedMoE` provides the fixed projection geometry and quantization layout when it creates each operator.
It provides the current active token count when it records each projection.
The affine operator selects QMV or a QMM tile.
`GatedMoE` does not store separate QMV/QMM kernels or a projection threshold.

`GatedMoECore::shared_experts_intermediate_dim` is the only semantic source for shared-expert presence and shape.
Code must not infer this information from `intermediate_dim`.
Routed and shared experts can use different intermediate widths.

Weight loading derives from `shared_experts_intermediate_dim`.
Shared-expert MLP construction and optional scratch allocation also derive from it.
The Qwen weight owner groups the shared gate and dense expert under one optional owner.
Thus, a partially populated shared-expert branch is not representable.

The Qwen MoE weight owner loads one bounded `TensorMap` from the exact MoE binding subtree.
It retains the core and Metal configuration that created its backend.
Weight reload uses these retained values.
It removes router, top-k expert, and optional shared-expert tensors from that map.
The private sparse-expert owner consumes its gate/up/down subset without creating a second checkpoint owner.
Each owner validates and materializes the backend-required persistent layout during initialization.
The map must be empty after the complete MoE owner is constructed.

`components::sparse_mlp::Compute` remains a lower-level expert compute component.
It exposes token-major and expert-major sparse expert MLP compute.
It does not own routing, dispatch, combine, shared-expert work, or execution-variant selection.

Its token-major shape is `{ num_routes, num_tokens }`.
Its expert-major shape is `{ num_routes }`.
`components::sparse_mlp::Config` owns `num_experts` for both paths.
Raw gather-matmul operators use semantic gather axes `{ num_routes, num_input_vectors }`.
Only their true matrix axes retain `n` and `k`.

## Replay contract

`GatedMoE` records one MoE MLP forward through `ReplayLayer::record(...)` and a caller-owned `Recorder`.
The full MoE path is the component boundary.
The sparse expert MLP kernel alone is not that boundary.

The semantic replay input is:

```text
GatedMoEInput
  shape    GatedMoEReplayShape
  hidden_state      &Buffer
  next_hidden_state &Buffer
  scratch  MoEScratchBindings
  weights  GatedMoEWeights
  shared_experts optional GatedMoESharedExpertsInput
```

Replay returns `next_hidden_state` directly.
It does not wrap the caller-owned buffer in a one-field output object.

Focused production-path tests use the same `ReplayLayer::record(...)` entrypoint as model replay.
The full-forward MoE benchmark composes the lower-level backend components directly.
This benchmark-only composition can force each implementation without adding a force option to the production API.

The no-shared-expert replay order is:

```text
hidden_state
  -> router quantized projection
  -> router bf16 softmax
  -> MoE routing
  -> token-major or expert-major sparse expert dispatch/compute
  -> top-k weighted combine
  -> next_hidden_state
```

The shared-expert replay order adds the shared branch before the final combine:

```text
hidden_state
  -> shared dense expert
  -> shared gate projection
  -> combine routed contribution + shared contribution
```

MoE routing is a two-stage contract.
The router projection writes bf16 logits.
The softmax operator writes a bf16 `router_probs` buffer over all experts.
`moe::routing::Compute` selects top-k experts from `router_probs`.
The routing kernel does not read router logits.
Only the softmax operator reads the logits.

This contract aligns replay resource dependencies with the data flow.
The routing kernel renormalizes selected probabilities only when `norm_topk_prob=true`.
`expert_indices` and `expert_probs` are route-major with `num_tokens * num_experts_per_token` entries.

`ReplayLayer::record(...)` uses one `GatedMoEInput` contract:

```text
GatedMoEInput
  num_total_tokens       recorded token capacity
  num_active_tokens      ReplayU32
  hidden_state           &Buffer
  next_hidden_state      &Buffer
  scratch                MoEScratchBindings
  weights                GatedMoEWeights
  shared_experts         optional GatedMoESharedExpertsInput
```

`GatedMoE::record(...)` selects the token-major or expert-major path from `num_total_tokens`.
It records all dispatch grids and validates all buffers at this total capacity.
`ReplayU32::Fixed(value)` requires `value == num_total_tokens`. `ReplayU32::Parameter(key)` supplies active work at
submission. The submitted value must be in `[1, num_total_tokens]`.

The routing, sparse MLP, expert-major, combine, optional shared dense MLP, and optional shared-gate commands use
one caller-owned replay parameter key.
They use the same type and the same domain.
The replay therefore declares one parameter.
No component declares a separate active-route parameter.
Each route-based command derives
`num_active_routes = num_active_tokens * num_experts_per_token` from the active token count and fixed top-k value.

The fixed total token count determines the full command topology.
The active token count does not select an execution variant or affine kernel.
`GatedMoEReplayTopology` identifies the selected variant key, router affine topology, optional shared-gate affine
topology, and optional shared dense MLP gate-up/down topology.
`GatedMoE::replay_topology_boundaries()` returns the union of:

- the token-major/expert-major boundary at token count `5`;
- router affine boundaries;
- optional shared-gate affine boundaries; and
- optional shared dense MLP gate-up/down affine boundaries.

A caller must construct `ReplayBucketPolicy` with these boundaries.
The policy must not pad an active token count across a topology boundary.
Sparse MLP, expert-major layout/pack/scatter, and combine keep one topology inside their selected path and add no
token-count boundary.

One `ReplayLayer::record(...)` path returns `next_hidden_state`. A fixed active value declares no replay parameter. A
parameter active value declares one replay parameter. Qwen3.5 Main calls this API through `Qwen3xMoE` with the Main
stage-owned token capacity and active-token key. Qwen3.5 MTP calls the same API with its separate body stage-owned token
capacity and active-token key
when the physical MTP layer uses MoE.

The backend retains a routing-only API.
This API records this chain:

```text
router affine -> router softmax -> top-k routing
```

`GatedMoE::record_routing(...)` accepts the same total-token and `ReplayU32` active-token representation as full MoE.
The router affine, softmax, and top-k routing commands use the same key and the same domain.
A fixed active value declares no replay parameter.
A parameter active value declares one replay parameter.

The fixed total token count determines the recorded affine kernel and all dispatch grids.
The active token count does not determine topology or replay identity.
All routing-chain buffers must cover `num_total_tokens`.
The router affine uses its active-row replay ABI.
The softmax kernel returns uniformly for an inactive row before it reads logits or reaches a threadgroup barrier.
The top-k routing kernel returns uniformly for an inactive token before it reads probabilities, writes outputs, or
reaches a threadgroup barrier.

`GatedMoE::routing_replay_topology(...)` reports only `{ router_affine }`.
`GatedMoE::routing_replay_topology_boundaries()` reports only router affine topology changes.
It does not add the full MoE token-major/expert-major boundary at token count `5`.
An affine topology change can independently occur at token count `5`.

The routing-only API does not change the `ReplayLayer` contract.
The full composition uses this routing chain with the same active-token value.

The ragged expert-major leaf uses these APIs:

```text
moe::expert_major::Compute::invoke_layout(...)
moe::expert_major::Compute::invoke_pack_input(...)
moe::expert_major::Compute::invoke_scatter_without_shared_experts(...)
moe::expert_major::Compute::invoke_scatter_with_shared_experts(...)
```

Each API accepts one `Shape { num_total_tokens }` and one `ReplayU32` active-token value.
The layout, pack, and scatter commands use the same active-token source and domain.
The Metal layout-count, layout-scatter, and pack commands derive
`num_active_routes = num_active_tokens * num_experts_per_token`.
The leaf does not declare an active-route parameter.

The total token capacity determines the count/scatter, pack, and output dispatch grids.
It also determines all token-major and route-major buffer validation.
Layout clear and prefix always process the fixed expert domain.
Layout count and layout scatter return before they read an inactive `expert_indices` route.
Pack returns before it reads an inactive route map or input row.
Both scatter variants return before they read inactive route maps, probabilities, packed output, shared output, or
shared-gate logits.
They also return before they write the inactive output tail.

A fixed active value requires `num_active_tokens == num_total_tokens` and declares no replay parameter.
A parameter active value declares one replay parameter for the composite expert-major execution.
These kernels do not change topology with token count.
They add no token-count boundary to a composite policy.
The full MoE owner must still keep the token-major/expert-major path boundary at token count `5`.
The full `GatedMoE` execution composes these leaf APIs on the expert-major path.

The backend sparse MLP leaf uses these algorithm-specific APIs:

```text
sparse_mlp::Compute::invoke_token_major(...)
sparse_mlp::Compute::invoke_expert_major(...)
```

Both APIs accept `num_total_tokens`, fixed `num_experts_per_token`, and one `ReplayU32` active-token value.
The component validates `num_total_routes = num_total_tokens * num_experts_per_token`.
It uses this total route count for dispatch and for all route, input, output, and SwiGLU scratch buffer validation.
All four expert affine commands use the same active-token source and domain.
Each Metal command derives `num_active_routes = num_active_tokens * num_experts_per_token`.
The leaf does not declare a second active-route parameter.

The token-major gate/up command returns before it reads `token_indices` or `expert_indices` for an inactive route.
The token-major down command returns before the gather code reads `route_indices` or `expert_indices`.
The expert-major gate/up and down commands return before they read `experts_by_route`.
Inactive commands do not read route inputs or write SwiGLU and output tails.

A fixed active value declares no replay parameter.
A parameter active value declares one replay parameter.
Token-major uses QMV. Expert-major records QMV alone below the affine batch crossover.
At larger recorded capacities, it records complementary QMV and QMM commands. The active expert segment lengths
select which command writes each segment. The total token capacity fixes this command topology.
The two explicit layouts also have different command topology.
The full MoE owner must keep the token-major/expert-major path boundary at token count `5` in its composite policy.
The sparse leaf does not select the path.
The full `GatedMoE` execution composes the selected leaf path.

The token-major combine leaf uses these APIs:

```text
moe::combine::Compute::invoke_without_shared_experts(...)
moe::combine::Compute::invoke_with_shared_experts(...)
```

Both APIs accept a fixed `num_total_tokens` capacity in `moe::combine::Shape` and one `ReplayU32` active-token value.
The total token count determines the dispatch grid and the required routed, probability, shared-branch, and output
buffer capacities.
The active token count does not change command topology.

Each combine kernel returns for an inactive output element before it reads routed data, route probabilities, shared
hidden state, or shared-gate logits.
It also returns before it writes the output tail.
A fixed active value declares no replay parameter.
A parameter active value declares one replay parameter.
The combine kernels do not change topology with token count, so they add no token-count boundary to a composite
policy.
The parameterized `GatedMoE` replay composes these leaf APIs on the token-major path.

Routing fixes `num_experts`, `num_experts_per_token`, and `norm_topk_prob` during initialization.
The generated Metal source specializes these values. Submission supplies only the active token count.
The 256-thread reduction scans expert groups before each reduction, so expert count can exceed 256.
Selected-expert arrays use the configured top-k. They have no fixed top-16 capacity.
Initialization requires `0 < num_experts_per_token <= num_experts` and checks the device shared-memory limit.
The executor softmax still limits `num_experts` to 4096. Shape validation also checks the shader index domain.
Invalid internal configurations panic at the owning boundary.
Ties select the smaller expert ID. Routing preserves BF16 probability and accumulation semantics.
It renormalizes only when `norm_topk_prob` is enabled, top-k exceeds one, and the selected sum is positive.

Production callers may allocate scratch with the executor's maximum token capacity.
The private `Selector::select(&Registry, ...)` operation selects a registered execution variant from the current
`num_tokens`.
Capacity-recorded replay selects it from `num_total_tokens`.
Each path validates that buffers cover its recorded total route, input, scratch, and output shapes.
The token-major path consumes `token_indices` and `route_indices` directly.

The expert-major path builds compact expert-grouped routes and packs hidden rows.
It then runs route-major sparse expert compute.
Finally, it inverse-scatters and combines the results.
Route order inside an expert group is not a semantic contract.
The inverse route map is part of the contract.

Qwen model replay keeps MoE scratch in one model-owned `MoEScratch`.
It owns three explicit regions.
They are routing, `topk_experts`, and optional `shared_experts`.
`bindings()` exposes routing and top-k-expert scratch.
`shared_experts_bindings()` exposes the optional shared-expert branch.

The model stream serializes Main and MTP execution.
Thus, MoE layers can reuse router logits/probs, route metadata, sparse swiglu, expert-major packing, and optional
shared-expert scratch.
When Main and MTP both use MoE, the Qwen3.5 loader rejects incompatible scratch geometry before allocation.
The shared geometry requires the same hidden dimension, expert count, selected experts per token, routed-expert
intermediate dimension, and optional shared-expert intermediate dimension.
A zero shared-expert intermediate dimension means that the branch is absent.
Both models must have the same shared-expert presence, and a present branch must have the same dimension.
The executor provides one maximum token capacity. The current Qwen3.5 component config fixes the scratch dtype to
BF16.
`norm_topk_prob` changes routing semantics but does not change scratch geometry.
If only Main or only MTP uses MoE, the scratch derives from that one consumer.

The shared `Qwen3xMoE` leaf owns per-layer router, top-k, and shared-expert weights.
The role-specific layer and scratch types own output buffers and composition.
These types are `Qwen35MainLayer`/`Qwen35MainLayerScratch` and `Qwen35MTPLayer`/`Qwen35MTPLayerScratch`.

Token-major `token_indices` and identity `route_indices` are capacity metadata.
They are not request metadata.
Qwen initializes them once in `MoEScratch`.
Each replay consumes the prefix for the current route count.

When a shared expert is present, the routed and shared-expert branches form a fork/join data flow.
Both branches read the same normalized hidden input.
They write disjoint scratch buffers and join only at final combine/scatter.
Recommendation: Do not insert a barrier between these branches unless they share a buffer.

`moe::expert_major::Compute` represents ragged expert-major MoE dispatch.
It groups routes by expert in a compact route buffer.
It packs token hidden states and runs route-major sparse expert compute.
It then inverse-scatters the results to token-major output.
The current implementation uses a parallel histogram/counting-sort layout:

```text
expert_indices
  -> expert_counts
  -> expert_offsets / expert_cursors
  -> routes_by_expert
  -> routes_by_token
  -> experts_by_route
```

`routes_by_expert[expert_route]` maps compact expert-major row to the original token-major route.
`routes_by_token[token_route]` maps the original token-major route back to its compact expert-major row.
`expert_offsets[e]..expert_offsets[e + 1]` is the ragged route segment for expert `e`.
Active expert IDs must be in `0..num_experts`. The routing owner establishes this invariant. The affine kernels
trust these IDs after they exclude inactive routes. Weight and row address offsets use 64-bit arithmetic.
MoE semantics require compact expert grouping and the inverse route map.
Original route order inside each expert is not a forward contract.

Expert-major sparse compute uses ragged rows with shape `{ num_experts, num_routes }`.
For each `expert_route` in `0..num_routes`, `experts_by_route[expert_route]` selects the expert.
It does not allocate `num_experts * routes_per_expert` rows.
Each active QMM thread block covers up to eight rows of one expert and 32 output columns.
The recorded grid contains one candidate per route and output-column tile. `expert_offsets` identifies the starts
of eight-row tiles within each expert. Other candidates return before loading operands or reaching a barrier.
Different expert tiles can execute independently.
The gate/up and down stages reuse `affine_qmm_tile` from `operators/metal/affine_quantized_tensor_ops.metal`.
Their manual accumulator access follows the shared
[cooperative tensor access contract](engineering_conventions.md#metal-cooperative-tensor-access).
F32 operands use a 32-element K tile. For group size 64, each thread block uses 5760 bytes: an `8 × 36` input tile
and a `32 × 36` weight tile. Gate/up reuses this scratch across its two projections and keeps both F32 cooperative
accumulators for the fused activation. Short-segment QMV retains its paired gate/up dot products within each SIMDgroup.
The private `ExpertMajorKernels` owner compiles and selects both families for gate/up and down. Each stage retains
its own buffer bindings.
One weight tile is dequantized once for all rows in that tile.
The affine batch crossover keeps small recorded batches on direct QMV. This avoids complementary dispatch overhead.
At larger capacities, expert segments with at least four rows use QMM. Shorter segments use QMV.
This per-expert crossover controls weight reuse inside the ragged eight-row tile.
QMV and QMM commands declare disjoint output segments and share one parent replay submission.
The complementary QMV command derives its row-task count from a 1024-threadblock target and the column-tile count.
Each task walks multiple routes and skips QMM segments. Both families apply SwiGLU before storing the gate/up result.
The existing compact route maps and pack/scatter stages do not change.

Token-major and expert-major replay paths remain explicit benchmark probes.
The production Metal backend uses token-major for `num_tokens <= 4`.
It uses expert-major for larger microbatches.

## Data flow and backend stages

MoE starts as a hidden-state transform and introduces a route-major side stream:

```text
hidden_state[tokens, hidden_dim]
  -> router projection
  -> softmax over num_experts
  -> routing top-k
       expert_indices[routes]
       expert_probs[routes]
  -> selected sparse expert MLP path
  -> combine/scatter
  -> next_hidden_state[tokens, hidden_dim]
```

where:

```text
num_routes = num_tokens * num_experts_per_token
route_index = token_index * num_experts_per_token + expert_slot_index
```

Router projection writes bf16 logits.
Softmax writes bf16 probabilities over all experts.
Routing reads probabilities, not logits.
It writes route-major `expert_indices` and f32 `expert_probs`.
When `norm_topk_prob` is enabled, routing renormalizes selected top-k probabilities over the selected experts.
Otherwise, the values remain the selected softmax probabilities.

### Token-Major Sparse MLP

Token-major sparse MLP keeps routes in original token-major order:

```text
input hidden[tokens, hidden_dim]
token_indices[routes]       token row for each route
expert_indices[routes]      selected expert for each route
route_indices[routes]       swiglu row used by down projection

fused gate/up/silu
  reads input[token_indices[route]]
  reads expert_indices[route] expert weights
  writes swiglu[route, intermediate_dim]

down
  reads swiglu[route_indices[route]]
  reads expert_indices[route] expert weights
  writes routed_hidden[route, hidden_dim]
```

In the shared Qwen `MoEScratch`, `token_indices` and identity `route_indices` are capacity metadata.
Qwen initializes them once.
Current `tokens * topk` determines the active route prefix.

### Ragged Expert-Major Sparse MLP

Expert-major first converts token-major routes into compact expert-major rows:

```text
expert_indices[token_route]
  -> expert_counts[num_experts]
  -> expert_offsets[num_experts + 1]
  -> routes_by_expert[expert_route]
  -> routes_by_token[token_route]
  -> experts_by_route[expert_route]
```

The layout kernels are:

```text
layout_clear    zero expert_counts and expert_cursors
layout_count    count routes per expert from expert_indices
layout_prefix   prefix-sum expert_counts into expert_offsets and reset cursors
layout_scatter  assign each token_route to a compact expert_route
```

`expert_offsets[e]..expert_offsets[e + 1]` is the ragged segment for expert `e`.
`routes_by_expert` maps each compact expert-major row to its original token-major route.
`routes_by_token` is the inverse map for final scatter.
Ragged expert-major affine kernels use `experts_by_route`.
`experts_by_route` selects the expert for each compact route row.

After layout:

```text
pack_input
  reads hidden_state[token]
  reads routes_by_expert
  writes packed_input[expert_route, hidden_dim]

ragged sparse expert MLP
  reads packed_input + experts_by_route + expert weights
  writes packed_output[expert_route, hidden_dim]

scatter/combine
  reads packed_output via routes_by_token
  reads expert_probs[token_route]
  writes next_hidden_state[token, hidden_dim]
```

Route order inside one expert segment is not semantic.
The contract requires compact expert grouping and a correct inverse map for scatter.
Expert-major affine kernels can process the resulting contiguous ragged segments.
They do not require a rectangular `experts * routes_per_expert` layout.

### Combine and Shared Expert

Without a shared expert, combine computes:

```text
next_hidden[token, dim] =
  sum_{slot in topk} expert_probs[token_route] * routed_hidden[token_route, dim]
```

With a shared expert, the routed branch and shared branch are a fork/join:

```text
routed branch: hidden -> routing -> sparse expert MLP -> routed contribution
shared branch: hidden -> shared dense expert, hidden -> shared gate projection

next_hidden[token, dim] =
  routed_sum[token, dim] + sigmoid(shared_expert_gate[token]) * shared_hidden[token, dim]
```

The branches read the same normalized hidden input and write disjoint scratch.
Recommendation: Add replay barriers only at these actual dependencies:

- Router logits before softmax.
- Probabilities before routing.
- Routing/layout before sparse expert compute.
- Expert output and shared branch before final combine or scatter.

### Execution hierarchy

MoE has one semantic owner and two complete command-graph decompositions:

```text
Registry
  Vec<(VariantKey, Variant)>
    TokenMajor  -> owns moe::combine::Compute
    ExpertMajor -> owns moe::expert_major::Compute

Selector::select(&Registry, GatedMoEReplayShape)
    -> (VariantKey::TokenMajor, &Variant::TokenMajor)
    |      routing -> token-major sparse expert MLP -> combine
    |
    -> (VariantKey::ExpertMajor, &Variant::ExpertMajor)
           routing -> layout -> pack -> expert-major sparse MLP -> scatter
```

`VariantKey` is the replay-topology identity because it names a different command graph. `Variant` owns the
compiled resources that are unique to that graph. The private `Selector` is the only owner of the token-count
threshold. Topology reporting and recording call the same selector with the total token count. Repeating this
inexpensive calculation is intentional. The component does not materialize a plan object.

Each non-persistent leaf kernel defines its own task:

| Kernel phase | One thread-block task |
| --- | --- |
| Routing Top-K | One token and its complete expert-probability row. |
| Expert-major layout clear | A bounded expert-index range. |
| Expert-major layout count | A bounded token-route range. |
| Expert-major layout prefix | The complete expert-count array. |
| Expert-major layout scatter | A bounded token-route range. |
| Expert-major pack | A bounded `(expert route, hidden)` value range. |
| Token-major combine | A bounded `(token, hidden)` output range. |
| Expert-major scatter | A bounded `(token, hidden)` output range. |

The current routing kernel constants require 256 threads. The expert-major clear, count, scatter, pack, and output
phases also require 256 threads. The current prefix phase requires one thread. These values belong to phase-scoped
thread-block constants. They are not dynamic workload fields.

`sparse_mlp::Compute` exposes token-major and expert-major entry points. These entry points are explicit layouts chosen
by the MoE owner. The sparse MLP leaf does not run another path selector. Its gather-affine and ragged expert-major
affine kernels own their matrix tiles and thread ownership.

### Backend selection boundary

`GatedMoE` owns execution-variant selection through its private `Registry` and `Selector`.
`sparse_mlp::Compute` owns only expert inner compute.
The current production selector is:

```text
tokens <= 4  -> token-major
tokens > 4   -> ragged expert-major
```

The runtime core and Qwen model config do not contain this threshold.
They also do not expose a forced implementation.
The full-forward benchmark directly composes the same lower-level backend components to force token-major or
expert-major execution.
Changing the selector is a Metal backend performance change.
Recommendation: Justify that change with full MoE wrapper numbers, not only isolated sparse MLP kernel timings.

Focused leaf tests record one total capacity into an isolated test cache.
They replay all legal active-token counts in non-monotonic order and compare the active logical domain with CPU
references.
The routing test covers normalized and non-normalized top-k probabilities, including 512 experts with top-10 and top-17.
It also covers non-power-of-two expert counts, top-1, top-k equal to expert count, and high expert IDs.
Explicit fixtures check cross-group ties, zero probabilities, and unchanged inactive output rows.
The sparse MLP test covers token-major and expert-major layouts, including the token-major fast-down topology.
The combine test covers paths with and without shared experts.
The expert-major layout test compares expert counts, expert offsets, route-map invariants, packed input, and both
scatter paths with CPU references.
It does not require a stable route order inside one expert group.
The routing boundary test checks inactive output rows separately from active-output correctness.
The full `GatedMoE` replay test uses isolated test caches and CPU composition references. It covers token-major capacity
`4` and expert-major capacity `8`. It replays every legal active token count in non-monotonic order. It runs with and
without shared experts, including 512-expert top-10 and top-17 configurations.
Router and expert weights vary so the complete output depends on expert selection and expert strides.
The full-output check uses `1e-6 + 0.02 * abs(reference)` tolerance for BF16 rounding against F32 reference intermediates.
It also verifies that experts in the upper half of the configured range participate.
A separate routing replay test uses non-uniform router weights and checks active logits,
probabilities, expert indices, and expert probabilities against the CPU routing reference.
Replay infrastructure tests own generic replay-parameter validation.

## Tests and benchmarks

Current Metal component benches:

```text
cargo bench --bench moe
cargo bench --bench sparse_mlp
```

The benches include Metal replay/ICB cases for MoE routing/combine.
`route_replay128` records 128 routing invocations in one submission to amortize host submission and wait overhead.
Its reported time covers the full submission. Divide by 128 for the mean time per routing invocation.
This diagnostic does not represent complete model throughput.
They include token-major sparse expert forward paths.
Synthetic forward cases record router projection, routing, sparse expert MLP, shared expert MLP, shared gate, and
scatter/combine.
They record these operations in one batch.
Token-major and expert-major implementations remain explicit replay cases.
They use the 35B-A3B MoE profile:

```text
hidden_dim=2048
moe_intermediate_dim=512
num_experts=256
topk_experts=8
tokens=1,2,4,8,16,32,64
```

Current Metal real-weight MoE full-forward bench:

```text
cargo bench --bench qwen35_moe -- \
  --model-dir <35b-a3b-model-dir> --layer 0 --tokens 1 \
  --impls token_major --iters 1 --warmup-iters 0 --runs 1
```

`qwen35_moe` loads a Qwen3.6-35B-A3B 4-bit checkpoint.
It composes one sparse layer from the production backend components.
It runs explicit token-major and compact expert-major replay without a production force flag.
CLI arguments select the model path, layer, token list, iteration counts, implementation, and parity checking:

```text
cargo bench --bench qwen35_moe -- \
  --model-dir <35b-a3b-model-dir> --layer 0 \
  --tokens 1,2,4,8,16,32,64 \
  --impls token_major,expert_major \
  --check-parity
```

`--check-parity` is an explicit non-timed diagnostic.
It is off by default.
Keep it separate from pure timing after correctness verification.

`sparse_mlp` includes replay-backed token-major sparse expert fused gate/up/silu probes.
It includes Metal replay/ICB sparse expert forward paths over the same token counts.
New sparse/MoE Metal code does not keep direct-submit component or forward paths.
The 27B dense checkpoint has no MoE routing, combine, or sparse-MLP path.
Thus, there is no meaningful 27B MoE bench.

Recommendation: Bisect MoE performance by contract boundary in this order:

1. Router projection, softmax, and routing.
2. Sparse expert MLP.
3. Combine.
4. Shared branch.
5. Full MoE wrapper.

Routing and combine belong to MoE.
`sparse_mlp::Compute` owns only expert inner compute.
Top-k ordering differences alone are not a performance target.
Route probability semantics must identify logits or already-softmaxed probabilities.
They must also identify whether routing renormalizes selected top-k probabilities.

Token-major QMV and expert-major TensorOps QMM must match the sparse CPU reference within its numerical tolerance.
Expert QMM uses F32 dequantized operands and accumulators. QMV applies scale and bias to packed-weight dot products.
The different F32 reduction orders can change output bits.
The sparse activation retains its existing storage-dtype rounding stages.
The current production backend uses token-major for `num_tokens <= 4`.
It uses expert-major for larger microbatches.

[`executor_benchmarks.md`](executor_benchmarks.md) defines shared GPU serialization, benchmark metrics, and
performance-evidence rules.
