# Engineering Conventions

This document gives repository-wide engineering conventions:

- Naming
- Runtime shapes and storage layouts
- Replay and resource safety
- Optimization
- Public APIs
- Test style

[`high_level.md`](high_level.md) defines architecture and ownership boundaries. Each component document defines its
tensor terms, source owners, and execution paths. [`gpu_execution.md`](gpu_execution.md) defines the shared GPU launch,
specialization, task, tile, layout, registry, and selection vocabulary.

## Design decision order

Use production ownership as the first design input. Do not start from a helper, branch, test fixture, or benchmark.

Use this order for a source or test design:

1. Identify the owner, caller-visible inputs, outputs, resources, and lifecycle.
2. Identify the invariant that the owner must enforce.
3. Compare peer components that have the same contract.
4. Apply the same owner boundary, API vocabulary, and structure where the contracts match.
5. Keep a real contract difference. Do not add a lifecycle operation only to complete a symmetric API.
6. Add an entity only when it owns a concept, invariant, resource, lifecycle, or reusable operation.
7. Validate external input and configuration at the owning boundary.
8. Use direct data flow and ordinary arithmetic in the same owner's private path after validation proves the result.
9. Select the narrowest test layer that can express the contract.

Transferable symmetry is a goal. Zero duplication is not a goal. Visual identity is not a goal.

Apply strong symmetry to these peer groups when their contracts match:

- GQA and GDN
- dense MLP, sparse MLP, and MoE
- Main, MTP, and DFlash2 model roles at shared role boundaries

Compare public APIs, owner structure, source layout, replay policy, and test classification. Keep component-specific
state transitions and compute paths when the contracts differ.

Production source may change when the change clarifies a real owner, API, resource, or lifecycle. Production source
must not change only to reduce test setup or expose an implementation detail.

Runtime core must not own model-specific GQA, GDN, MLP, MoE, sampling, MTP, or DFlash2 policy. The model executor owns
these semantics. The Metal backend owns their reusable kernels and dispatch implementation.

## Rust semantic workflow

Use rust-analyzer semantic operations when applicable: definitions, references, types, diagnostics, rename, and refactoring.
Use `rg` for textual discovery. Do not substitute text matching for symbol identity or binding analysis.
Rename each binding or item independently when the same spelling occurs in multiple scopes.

Audit Metal, generated source strings, documentation, inactive configurations, and host↔shader ABI correspondence separately.
Rust-analyzer does not cover these boundaries. Check shadowing, stale references, and semantic-equivalence regressions.

## Naming and coordinate domains

Recommendation: Names show their domain semantics. Use established component abbreviations as type prefixes. Examples
include `GQA*` and `GDN*`.

Use semantic names at model, layer, component, and non-matmul operator boundaries. Examples include:

- `num_tokens`
- `num_routes`
- `num_input_vectors`
- `num_experts_per_token`

Use `m`, `n`, and `k` only for a low-level matrix-multiplication shape. Keep semantic names for routing, gather, page,
and state coordinates. Keep these names when the implementation sends the values to a matmul.

Use `num_*` for a valid typed work count. Use `total_*` for a padded dispatch, replay, scratch, or capacity extent.
Name the item that the value counts.
For cached replay domains, use the specific `num_active_<domain>` and `num_total_<domain>` names defined under
[Runtime shapes and persistent layouts](#runtime-shapes-and-persistent-layouts).

Do not use `element` or an unqualified `size` for a tensor or domain. Use `*_bytes` for these byte values:

- Raw allocation lengths
- Binding offsets
- Address arithmetic
- Byte copies

Use unsigned integers for counts, slots, and IDs. Use signed integers for real negative values, sentinels, or imported
ABI contracts.

Use half-open intervals `[start, end)` by default. Rust source must use `start..end` for this interval. An owning
contract may use a different interval representation when that representation is necessary. In that case, document
the endpoint semantics at the owning type or function. Do not infer inclusive or exclusive endpoints from context.

An API that accepts a valid or visible interval must expose both bounds. Do not expose only an exclusive upper bound
and assume that the inclusive lower bound is `0`. Pass `[start, end)` explicitly, including when `start == 0`. Do not
use `Option` for one bound when the interval itself is required.

Use checked `u64` values for host byte offsets and address arithmetic. Use `ulong` for flattened Metal addresses.
Convert to `usize` only at a Rust slice, pointer, or Objective-C API boundary.

Keep bounded local counts and indices as Rust `u32` or Metal `uint`. A large allocation does not require 64-bit local
loops or tensor coordinates.

Configuration names identify their owner:

```text
*ExecutorConfig   model-executor initialization
*RuntimeConfig    runtime core
*ServerConfig     service/application initialization
```

Checkpoint schemas identify their format. For example, use `HFGenerationConfig` instead of an unqualified subsystem
name.

Recommendation: Use precise coordinates:

```text
token_index
block_index
base_token_index + token_offset
base_block_index + block_offset
state_version
candidate_state_versions
verified_state_version
replay_source_state_version
candidate_start_state_version
candidate_end_state_version
```

Recommendation: Do not use `absolute_*` unless another coordinate system exists. Do not use an unqualified `offset` or
`context`. Name the owner or unit.

This rule also applies to names such as `candidate_lens` and `commit_len`.

Use `flat` only for a tensor or coordinate from flattened per-request token sequences. Per-request `token_indices` can
expand into `flat_token_indices`.

Keep request-slot IDs as `req_slots` when they repeat for each token. `cu_tokens` already identifies flattened segment
boundaries. It does not need a `flat` prefix.

Use `cu_<items>` only for a monotonic cumulative-count table with `N + 1` entries. Adjacent values select this
half-open segment for logical owner `i`: `cu_items[i]..cu_items[i + 1]`.

The owning comment names the owner and the counted item. Do not use `cu_` for these values:

- Ordinary coordinates
- Byte offsets
- Capacities
- Non-cumulative metadata

Domain-standard abbreviations are appropriate when the owning boundary makes them unambiguous. Attention code may use
`q`, `k`, `v`, `qk`, and `kv`.

Attention code may also use projection names such as `qkv`, `qgkv`, and `qkvabz`. Do not reuse these names for unrelated
concepts.

Recommendation: Outside an established domain, use the complete semantic noun.

Let the type and the enclosing owner establish context. A field name must distinguish the field from its peers. Do not
repeat the entity kind when the type already supplies it.

Use `qgkv`, `qkvabz`, `gate_up`, `router`, and `output` for affine stages in a typed component. Do not add
`projection`, `proj`, `kernel`, or `buffer` only to repeat the field type. Keep exact checkpoint tensor names such as
`q_proj` and `down_proj` at the checkpoint boundary.

Name a pure transform from its input to its outputs. Examples include:

```text
qgkv_to_q_g_k_v
qkv_to_q_k_v
qkvabz_to_qkv_a_b_z
bf16_to_f32
f32_to_bf16
```

Do not add a tensor role to a pure data-conversion name. For example, use `bf16_to_f32`, not
`hidden_state_to_f32`.

Use the exact operation name when it is part of the contract. Use `swiglu` for `SiLU(gate) * up`. Do not call this
operation only `activation` or `silu`.

Use `shared_experts` and `topk_experts` for MoE branches. Use `shared_expert_gate` for the one gate that controls the
aggregate shared-expert branch.

Model and executor component configs must state their hidden-state boundary dtype. Use `io_dtype` when input and output
must use one dtype. Use `input_dtype` and `output_dtype` when the two boundaries are independent. These fields describe
workload facts. They do not select a kernel dtype specialization.

Current Metal model components accept BF16 hidden-state boundaries. A recognized F32 boundary must fail with an explicit
future-work `todo!` until the complete component path supports it. Low-level operators may support additional dtype
combinations.

Use one symbolic convention for attention tensor and tile comments:

```text
Q: [Tq,  Hq,  D]    Tq  = Q tokens       Hq  = Q heads
K: [Tkv, Hkv, D]    Tkv = KV tokens      Hkv = KV heads
V: [Tkv, Hkv, D]    D   = head dimension
O: [Tq,  Hq,  D]

Q tile: [Tq_tile, Hq_tile, D]
K tile: [Tkv_tile, D]  // one fixed KV head
V tile: [Tkv_tile, D]
```

`T` identifies a token dimension. `H` identifies a head dimension. Add `_tile` to the tiled axis.

Describe GQA work with Q tokens, Q heads, KV tokens, and head dimensions.
Do not replace these coordinates with an unqualified `row` in GQA names or explanations.

Do not introduce the ambiguous names `Bq` or `Bkv`. Outside the SDPA tensor or kernel boundary, use `token_tile`.
Use `q_*` only for the Q tensor and its dimensions.

## GPU work vocabulary

Keep mathematical decomposition separate from launch topology:

- A `*Tile` is the smallest named matmul-like unit at that component boundary. It is not a launch object.
- A `*Task` is the full logical work for one threadblock. A Task and a threadblock have a 1:1 relationship.
- A `*TaskTemplate` is an optional stored subset of Task fields. Regular grid coordinates reuse this subset.
- A threadblock is one cooperating group of GPU threads. Metal calls it a threadgroup.
- A warp is a hardware execution subgroup within one threadblock. Metal calls it a SIMDgroup.
- A grid is all threadblocks launched by one kernel dispatch.

`Task` and `TaskTemplate` identify logical work. They are not CUDA or Metal launch primitives.

Use the CUDA execution terms in backend-independent documentation and source:

| Project term    | CUDA term      | Metal term         |
| --------------- | -------------- | ------------------ |
| `threadblock`   | thread block   | threadgroup        |
| `warp`          | warp           | SIMDgroup          |
| `shared memory` | shared memory  | threadgroup memory |

Keep the exact backend term in a backend API, shader attribute, intrinsic, identifier, or quotation. Examples include
`threadgroup_barrier`, `simdgroup_index_in_threadgroup`, and `thread_index_in_simdgroup`.

A Task can run one or more Tile steps. It can move one tensor tile repeatedly along an ordered axis.

Put each path-specific Tile, Task, and Grid contract beside its Rust recorder or source owner. Do not put this contract
on generic model metadata that multiple paths share.

Do not add Rust or MSL items only to represent a threadblock, grid, or fully derived Task. This rule applies to structs
and variables.

At the owning Rust or MSL boundary, comments list each logical Task coordinate and its source. Identify grid-derived
coordinates explicitly. Do not store a Task, TaskTemplate, or ABI buffer when all fields are derived.

Name a Task only when a threadblock owns one semantic work unit. Some flat elementwise or map dispatches use
threadblock grouping only for tuning.

For these dispatches, describe the tensor map and grid. Do not add a `*Task` name.

For irregular work, store only fields that regular rules cannot derive. One TaskTemplate and its grid coordinates must
produce one Task for one threadblock.

For a map/reduce pipeline, use `*PartialOutput` for the map result. Use `*Output` for the fully reduced result.

Reduce metadata identifies its selected partial outputs. It does not identify the TaskTemplate that produced them.
Put component coordinates, ABI records, and cumulative-offset examples in the owning component document and source.

## Runtime shapes and persistent layouts

Keep runtime or replay shapes separate from initialization capacity and storage layouts.

`*ReplayShape` contains only values that define one recording. These values include recorded capacities, topology, and
static geometry. Submission input contains the active counts and other dynamic values.
The replay shape does not contain initialization capacities, persistent-buffer strides, or storage coordinates.
A replay capacity can equal an initialization limit, but it has a different owner and meaning.

A reusable leaf component may use one `*Shape` for exact and bucketed invocations. Keep this shape when it owns shape
validation or derived execution extents, or when it preserves the contract of peer components. Do not remove it only
because it contains one field.

For each cached replay work domain, use `num_total_<domain>` for the recorded grid or capacity. Use
`num_active_<domain>` for the logical work in one submission. Bind `num_active_<domain>` as a replay parameter. Keep
the two values separate even when they are equal. Validate
`0 < num_active_<domain> <= num_total_<domain>` before submission for required work domains.
An optional branch in a fixed graph can have zero active work. Its owner must validate
`0 <= num_active_<domain> <= num_total_<domain>` and its relation to the containing active domain.
Every inactive kernel must return before it accesses metadata or state, writes output, or enters a threadblock barrier.
The recorded capacity remains positive. A derived suffix can use the containing active count and the prefix count.

The capacity policy can select `num_total_<domain> == num_active_<domain>`. This identity policy is necessary when the
recorded commands cannot execute inactive lanes safely. It does not change the replay parameter or cache-key contract.
Use `num_<domain>` only when the component has no replayed active and total distinction.

Remove an exact API or a shared shape only after a repository-wide reference audit confirms that production does not
use it. Tests and benchmarks are not sufficient evidence of production ownership.

Use `*Layout` for an object that primarily describes persistent tensor dimensions. Do not name this object `*Shape`.

For example, a GQA page-ID table stores
`[num_req_slots, num_gqa_layers, num_blocks, num_page_ids_per_block]`. It exposes exactly these fields:

```text
num_req_slots
num_gqa_layers
num_blocks
num_page_ids_per_block
```

Backend command shapes may contain a nested Layout. This permission applies when lowering needs persistent dimensions
as command constants or source specialization.

Layer-local coordinates use the matching name:

- Use `model_layer_index` for the full model stack.
- Use `gqa_layer_index` for the compact GQA table.

Recommendation: Do not encode one caller's tensor-axis interpretation in a name. For example, do not use
`num_page_table_layers` when the stored dimension contains layers.

Layouts and persistent state store independent model inputs, resource handles, and lifecycle data. Do not cache a count
that adjacent dimensions, a data type, or `page_bytes` uniquely determine.

Derive that count at the typed-index or raw-byte boundary that needs it. Forward paths may borrow initialized layouts.
They must not derive capacity from the current batch.

Logical structure has priority. Stable component identity and explicit data-type or tuning choices are not redundant.
Meaningful resource views are also not redundant because one model uses one value.

Use transferable symmetry across peer components. Use the same owner boundary, API vocabulary, and structure when the
contracts match. Symmetry does not require identical lifecycle operations or zero duplication. Do not add a type, trait,
wrapper, field, or operation only to create visual symmetry.

Remove a wrapper only when it forwards one-to-one. The wrapper must own no independent lifetime, slice, resource, or
semantic branch.

### Validation and arithmetic

The trust boundary determines validation behavior.

Outside the trust boundary, input is not trusted. An explicit input validator must validate user input, external batch
metadata, checkpoint data, file data, and other externally controlled values. A validation failure must return the
shared typed `Error` or the component-specific recoverable error. The caller-visible semantics select the error
variant. For example, invalid RPC input must return `InvalidArgument`.

A constructor or initialization path establishes the trusted domain. It must validate static configuration, topology,
capacity, layout, narrowing, allocation size, and shader-domain limits that its private paths require. It may use
checked arithmetic to prove these invariants once.

Inside the trust boundary, code manages invariants. Private paths must use ordinary arithmetic and direct lossless
casts. An invariant violation is a code bug. Use `panic!`, `assert!`, or `debug_assert!` according to the lifecycle and
cost rules. Do not convert an invariant violation to a recoverable error. Do not repeat checked operations,
`assert!`, or `debug_assert!` for a shape or range that an owning boundary already proved.

Each checked addition, subtraction, multiplication, or conversion must identify one named real boundary through its
owner API, value name, or failure message. Allowed boundaries are allocation or byte sizing, external or runtime counts
and indices, narrowing, shader counts and element indices, file or snapshot data, state versions, and real overflow.

Outside a constructor, initialization path, explicit input validator, or test, an assertion must protect either a
complex state or resource transition, or a non-local invariant that no existing owner has proved. Add a concise
adjacent source comment that names the invariant and explains why the assertion is necessary. Do not add defensive
assertions to a trusted private path.

Keep checked arithmetic at these boundaries:

- Allocation and byte sizing
- External or runtime counts, indices, IDs, and capacities
- Narrowing or target-width-dependent conversions
- Shader count and element-index domains
- File and snapshot offsets or lengths
- State-version conversion and shift semantics
- Real overflow boundaries

Use direct casts for conversions that are lossless on all supported targets. For example, Apple Metal code can use
`u32 as u64` and `usize as u64`. Do not mechanically remove checked arithmetic. Keep it when the operation still owns
one of the listed boundaries. Do not keep it after the same owner already proved the complete domain.

## Replay and asynchronous resource safety

Replay keys contain only facts that change these items:

- Recorded command structure
- Dispatch topology
- Static geometry
- Scratch extent
- A necessary algorithm choice

Within one replay owner, the cache key must contain each `num_total_<domain>`, the selected topology, and all other
record-time static facts. It must not contain `num_active_<domain>`. A cache entry must be reusable for each legal
active count in its recorded domain.

Each submission must supply `num_active_<domain>` through a typed replay parameter. Changing only an active count must
not record a new program. Changing a total count, topology, or other record-time static fact must select or record a
different program. These requirements also apply when an identity capacity policy makes the active and total values
equal for one submission.

A reusable leaf component must expose its topology identity and topology boundaries for each bucketed work domain. The
owner of a composite replay stage must union the boundaries from all participating leaf components before it selects a
capacity. A component-local policy is not the final policy for a larger replay stage.

The semantic replay-stage owner must construct and apply the final `ReplayBucketPolicy`. Model-specific callers must
request a prepared replay shape from that owner. They must not duplicate a bucket-selection algorithm.

Validate all invariants that are necessary for safe replay reuse before the cache lookup. Key derivation can perform
this validation. An assertion that runs only while a program records does not protect a cache hit.

Replay bucket capacities are an explicit exception to the default half-open interval convention. Each stored bucket is
a positive inclusive upper capacity. A topology boundary `b` is the exclusive upper boundary of the preceding topology.
It separates the half-open topology domains `[.., b)` and `[b, ..)`. The policy must add `b - 1` as the final inclusive
capacity for the preceding topology. Zero means that the work domain is absent. It must not be a replay bucket capacity.

Two bindings may use the same replay parameter key only when they use the same scalar type, active work domain, and
validated range. A selected topology must declare each active work domain that its recorded commands consume. It must
submit each declared parameter exactly once.

When one replay stage is the only consumer of a component, `Replay<T>` must be its single owner and access path.
Use `Replay::component()` for prepare, replay-argument, and read operations that belong to that stage.
Do not keep a sibling `Rc<T>` only to bypass the replay owner.

A separate shared handle is valid only when multiple independent stages or semantic owners consume the same resource.
Examples include one sampler shared by Main, MTP, and rejection, or one GDN state table shared by layers, restore, and
publish.

Put request slots and dynamic values in batch metadata or submission arguments. Dynamic values include these examples:

- Valid counts
- Page IDs and offsets
- Temperature, top-p, and seed
- Sample position and sampling domain

Do not expand the cache key to prevent the implementation of a typed dynamic input.

Put synchronization at the exact consumer dependency. A missing barrier is a correctness bug. A global barrier for
each command is not an acceptable replacement.

Identify each RAW, WAR, WAW, aliasing, or semantic-phase boundary.

Keep asynchronous backend resources alive and resident until host-visible completion proves that the submission
finished. An in-flight submission can still refer to these resources.

During this time, do not reset or reuse these resources:

- A command allocator
- A scratch owner
- A parameter buffer
- A replay-local resource

Put backend-specific completion and residency mechanisms in the backend documentation and implementation.

[`executor.md`](executor.md) defines detailed replay composition. The Metal backend
[`README`](../crates/inference-backend-metal/README.md) defines the Metal object model, completion mechanism, and
residency ownership.

## Metal cooperative tensor access

Cooperative tensor storage depends on the GPU and the TensorOps layout.
`begin()`/`end()` and `get_capacity()` do not establish element validity.
Before scalar access or coordinate extraction, kernels must check the owning tensor's `is_valid_element` for that iterator or index.
This rule includes accumulator initialization, arithmetic, mapped reduction reads, and stores to ordinary arrays.
An invalid storage slot is not a logical tensor element. Kernels must skip it before deriving an address.

For a valid element, use the tensor's logical extents as the coordinate invariant.
Do not repeat fixed tile bounds that this invariant already proves.
Keep checks for active tokens, heads, routes, visibility, and partial tiles. Storage validity does not establish these runtime bounds.

One validity check can cover tensors with the same layout, such as a copy or the same operation factory specialization.
Paired iterators must advance together, including when the loop skips an invalid slot.
Equal dimensions alone do not establish layout compatibility.
Use `map_iterator` only between compatible layouts. Exchange values through logical coordinates when the layouts differ.
Keep collective TensorOps operations and barriers outside per-element validity branches.

See the [Metal Shading Language specification, section 2.22.3](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
It defines access to an invalid cooperative element as undefined behavior.
Recommendation: Review every manual cooperative access when adding or changing a TensorOps kernel.
Run CPU-reference checks for partial tiles and padded active counts on each available GPU generation.
A passing test on one GPU does not establish the storage layout of another GPU.

## Optimization correctness

An optimization must preserve the semantic boundary that it replaces:

- Tensor outputs
- Routing decisions
- Probability distributions
- State versions
- Lifecycle effects

A speculative path must not publish unverified future state. An alternative sampler must use the same transformed
distribution as the production path.

Each current component document defines its exact sampling, state, and component-path contracts.

Kernel and composition policies depend on shape. Keep strong paths for small and large shapes. Do not replace all paths
with one locally faster specialization.

A primitive microbenchmark is not sufficient evidence for a model-layer or end-to-end change. At representative
shapes, validate these items:

- Production composition
- Metadata updates and barriers
- Scratch lifetime
- Output ownership

Establish output, route, probability, or state parity before you make a performance claim. For speculative execution,
keep the deterministic workload trajectory fixed.

If the trajectory changes, report these values:

- Proposals
- Sampled tokens
- Accepted tokens and chunks
- Acceptance efficiency

Throughput from a different trajectory is not a like-for-like executor or kernel comparison. See
[`executor_benchmarks.md`](executor_benchmarks.md) for the verification and performance-evidence rules.

Keep the README performance summary limited to PSI results. Identify each measurement date and mode.
Keep historical results separate from current results. Put cross-framework comparisons in the owning performance report.

## Public API

Keep the public surface small. Items remain private unless an external caller needs them. Do not use `pub(crate)` or
`pub(super)`.

Treat a model role as a structural ownership boundary. Main, MTP, and DFlash2 use different concrete layer types. This
rule applies when they use some of the same operators.

Share true leaf components and utilities across roles. Do not share a structural layer facade that mixes their
execution graphs.

Production source and benchmarks use crate-absolute paths for sibling and ancestor imports. Use `use super::...` only
in tests.

A trait can be the production interface for a concrete type. In this case, put the method implementation directly in
the trait implementation.

Keep constructors and other operations in the inherent implementation. Do not duplicate trait methods as inherent
methods. Do not forward between duplicate methods.

Do not export these items only for convenience:

- Planning structs
- Kernel metadata
- Workaround modules
- Scratch internals
- Backend-local tables

A benchmark can need private internals. Discuss a benchmark helper API before you widen the production API. Do not
change production `src` only to make a benchmark easier.

Keep backend details behind backend APIs. Metal components own kernels, dispatch parameters, tile scratch, and runtime
resource bindings.

Model executors own model semantics, persistent model or request buffers, and component wiring. They do not expose
backend tiling or kernel-local tables in model APIs.

Model and executor components must provide the complete semantic workload facts that a backend operator needs. These
facts include tensor shapes, data types, storage layouts, quantization parameters, and buffers.

The backend operator must select its algorithm, kernel family, dispatch geometry, and tile configuration. Model and
executor components must not select or name backend kernels or tile configurations.

A backend benchmark may force an exact backend path to measure a crossover. This benchmark control must remain at the
backend boundary. It must not enter a model configuration or executor API.

Use a component-scoped `ExecutionVariant` for a complete selectable combination of algorithm, command graph, kernel
families, and compile-time constants. Use a component-scoped `KernelKind` only when a lower-level kernel identity has a
real consumer, such as replay topology, logging, or a backend benchmark. Do not use an unqualified `Path`, `Kind`, or
`Algorithm` for these concepts.

Production code must select one complete execution variant from complete workload facts. A benchmark may force a
concrete implementation at the owning backend boundary. Do not add an execution-variant enum when only one
implementation exists. Do not add separate algorithm and kernel selector layers for one decision.

An adaptive affine quantized matmul uses these ownership boundaries:

- `affine_quantized::Config` contains fixed workload facts. These facts are `N`, `K`, quantization parameters, and
  data types. It does not contain `M`.
- `affine_quantized::Matmul` owns the candidate kernels. It selects one kernel from the runtime `M` and the fixed config.
- `affine_quantized::Matmul` exposes the selected `affine_quantized::KernelKind` as a stable topology identity. It also
  exposes the first `M` for each topology change. A replay bucket policy must use these boundaries. Model code must not
  duplicate the selector thresholds.
- `affine_quantized::Kernel` owns one compiled specialization. Its `affine_quantized::KernelKind` fixes the
  QMV or QMM family and its tile dimensions.
- An exact `affine_quantized::Invocation` contains one fixed `M`, buffers, and byte offsets.
- A bucketed `affine_quantized::Invocation` contains `num_total_rows` and a `u32` replay parameter key for
  `num_active_rows`. Kernel selection, dispatch, and buffer validation use `num_total_rows`.

An inactive QMV row must return before it reads the input or writes the output. An inactive QMM row threadgroup must
return before it derives input pointers or reaches a threadgroup barrier. The Metal entry points and replay parameter
table use `u32` for the active row count. The backend rejects total row counts above the positive `i32` range because
the internal MLX matrix dimensions use `int`.

`affine_quantized::Matmul` must support each combination of F32, F16, and BF16 input, scale/bias, and output data types.
QMV BN8/BK32 and QMM BM8/BN32, BM16/BN32, and BM32/BN32 provide this complete capability set.
QMV Quad BN64 is an optional specialization for its supported same-dtype shapes.
The adaptive owner must select QMV BN8/BK32 when QMV Quad BN64 does not support the workload.

Production model and executor components must use `affine_quantized::Matmul`. They must not select an
`affine_quantized::KernelKind`.

## Commits

Recommendation: One commit owns one coherent architectural idea. While history is local, amend or rebase a fix into
its owning commit.

After you publish or share a commit, preserve the history. Use a focused follow-up commit.

Use a concise subject that starts with a lowercase verb. Focus the subject on the main purpose. Keep standard acronyms:

```text
add HTTP chat completions
clean tokenizer streaming
```

Do not list related changes in the subject. When a body is useful, give a few high-level points. Do not restate the
per-file diff.

Recommendation: Use reviewable and compilable boundaries when practical. An intentional local history split can
temporarily fail to compile.

Delivered history must be self-consistent.

## Test style

Each Rust test function starts with `test_`. It protects one behavior, correctness, ownership, or lifecycle contract.

Use runtime-core unit tests as the primary naming and case-design reference. Name tests `test_<api>_<case>` or
`test_<api_sequence>_<case>`. Use short case names such as `success`, `fail`, `invalid_shape`, and `bucketing`. Examples
include `test_replay_success`, `test_replay_invalid_shape`, and `test_replay_bucketing`.

Let the module path identify the component. Do not put backend, `matches_reference`, poison value, canary, fixture,
capacity, or another internal mechanism in a test name. An actual named production or reference API can appear. Use one
representative fixed case, mixed case, or compact table for one contract. Do not add one test for each branch,
assertion, or helper.

Organize component tests by owner API set and lifecycle scenario. Do not create one test for each internal helper or
branch. One mixed-request scenario may assign a different case to each request when one owner prepares the full group.

Use the same test classification across peer components. Do not give a stateless component a transaction lifecycle
only to match a stateful component.

Keep these test responsibilities separate:

- A fixed CPU-reference test proves the reference implementation.
- A Metal parity test proves kernel math, dispatch, ABI, padding, aliasing, and canaries.
- An executor component test proves metadata, resource ownership, and component-local lifecycle.
- An integration test proves model composition, weight residency, and full executor lifecycle.

Central replay infrastructure tests prove cache lookup, parameter transport, and parameter-range validation. A central
test uses a small deterministic component and an exact reference. It must not use output inequality as a correctness
oracle.

Each replayable component must have an execution test for each independently variable active work domain. The test must
use the production component record API through an isolated test replay cache. It must record one total capacity and
topology, then replay a non-monotonic sequence of legal active counts. Use a recorded total capacity of `4` and the
representative active sequence `[1, 4, 3, 2]` for a true active/total bucketing contract.

Test peer components independently even when a production model owner records them in one replay program and stores
that program in one cache. The component test cache is test-only orchestration. It must not add a production replay
owner, wrapper, trait, or lifecycle operation. A separate model wiring test must prove the composite cache key, cache
reuse, and shared replay-parameter binding.

The owner test must prove these contracts:

- The first use of the isolated test cache records one program. Later submissions with the same total capacity,
  topology, and static facts hit the same cache entry.
- Each submission produces the exact reference result for its active logical domain.
- A change to total capacity, topology, or another record-time static fact selects or records a different entry.

The test can ignore an ordinary inactive output or scratch tail. It must not fill that tail with NaN as poison. If the
component writes persistent or scatter-addressed semantic destinations, the test must also prove that one submission
does not change its inactive destinations. Snapshot the current inactive destinations before each partial submission
when an earlier full replay can have changed them.

Keep the component reference and active-domain projection in the component test. Centralize only repeated replay
orchestration. Do not add a production trait, wrapper, or lifecycle operation for test reuse.

Do not combine a CPU oracle, Metal parity, and a lifecycle scenario in one test.

Let the test target and module path identify the subject. Use the shortest unambiguous case name.

Exercise the production owner API and its real contract boundary. A helper-only test does not protect owner wiring for
setters, recording, or submission.

Keep a helper-only test only when the helper owns reusable semantics that production tests cannot cover clearly.

Use `fixture_*` for structured test data or builders. Use `reference_*` for slow or CPU oracles.

Use a simple `new_<noun>` name for a small test-local mock constructor. For example, use `new_task`. Use `test_*` only
for test entry points.

List test entries before local helper functions. Keep the helpers at the end of the test module.

Do not keep constructor-only tests when stronger execution tests cover them. Do not test derived Rust behavior such as
`PartialEq`.

Do not keep a test that only inspects derived kernel constants, thread-block fields, or an unimplemented future-work
branch. Prove a supported kernel configuration through execution. Keep a selector-boundary test when the selection
changes replay topology or preserves a measured production policy.

Keep unit tests beside the production logic that they validate. Keep a small and focused test module inline. Put longer
unit-test coverage in a sibling `*_test.rs` file and declare it under `cfg(test)`.

Put integration tests in the crate `tests/` directory. Put benchmarks and benchmark-only support in the crate
`benches/` directory.

Tests and benchmarks exercise production APIs. Do not add test-only or benchmark-only abstractions to production code.

Use `unwrap()` for test setup and coordination failures that are not the tested behavior. Do not use verbose
`expect(...)` messages for these failures.

Use these Mockall call counts:

- `.once()` for exactly one call
- `.times(n)` for repeated calls
- `.never()` for forbidden calls

Omit the call count only when it is intentionally unrestricted. Use `return_const` for a cloneable fixed result. Use
`return_once` for a one-use or move-capturing result.

Use `returning` for a repeatable computed result.

Recommendation: Optimized numerical paths have concise fixed-input and random-input tests against a CPU or slow
reference.

Recommendation: Reference implementations make correctness clear and keep Clippy output clean.

Use a named input struct instead of a long argument list.

Recommendation: Use iterators and `enumerate` unless index arithmetic is the tested behavior.

Test-only constructors and fixtures may remain simple when production construction enforces the contract. Do not change
them only to mirror production.

Put backend execution constraints in the matching backend or executor document. Serial Metal testing is one example.

## General source clarity

Use explicit tracing macro paths for INFO and DEBUG events:

```rust
let _span = tracing::info_span!("runtime").entered();
tracing::info!("started");
tracing::debug!(request_id, "request prepared");
tracing::info!("stopped");
```

Do not import or invoke bare `info!` or `debug!` macros. Give each long-running component a short kebab-case span name.
Put stable component fields on the span. Emit only `started` and `stopped` at INFO for its lifetime. Do not repeat the
component name, a `component` field, or a `phase` field in these events. Instrument an async future with its span. Do not
hold an entered span guard across `.await`. Do not emit an additional INFO event only to report receipt of the shared
shutdown signal. Individual worker or runner lifecycle events must not be INFO.

Keep operation transactions and machine-consumed telemetry separate from component lifetime events. They may use
structured phase, result, and timing fields.

Recommendation: Use concrete nouns such as `file_name_for`, `file_path`, and `mapped_files`. Do not use overloaded
shorthand unless it is a stable module contract.

A callback can keep, transform, or remove an item. For this behavior, use an explicit action enum. Do not encode the
control flow in `Option<T>`.

Use `Option<T>` only when absence is a valid domain state. Do not use it to select an implicit default value. For a
required bounded interval, pass both bounds without `Option`.

Comments explain ownership, ordering, units, or backend constraints that are not obvious. They do not describe
mechanics that the code shows.
