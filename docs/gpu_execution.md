# GPU Execution

This document defines the shared GPU execution model for the repository. It defines names, ownership, and lifecycle.
Component documents define component-specific algorithms, tasks, tiles, arguments, and layouts.

The model follows the hierarchy and scope concepts in
[Modern GPU Programming for MLSys](https://mlc.ai/modern-gpu-programming-for-mlsys/). That material is a conceptual
reference. It uses NVIDIA Blackwell as its main target. Do not copy Blackwell-specific CTA, warpgroup, TMA, TMEM, or
tile-size names into a backend-independent API without a matching project contract.

The project uses `threadblock` in backend-independent identifiers. CUDA calls this object a thread block. Metal calls
it a threadgroup. Metal source must use Metal terms at the backend boundary.

## Mental model

Keep compilation, launch, semantic work, and runtime selection separate:

```text
Compilation
    KernelSource + KernelConstants
        -> CompiledKernel

Launch
    CompiledKernel
    + KernelArguments
    + KernelExecutionConfiguration
        -> KernelLaunch

Hardware execution
    KernelLaunch
        -> Grid
            -> ThreadBlocks
                -> Threads

Semantic work
    one current non-persistent ThreadBlock
        -> zero or one component-defined ThreadBlockTask
            -> component-defined ThreadTask per Thread

Runtime implementation choice
    StaticConfig
        -> Registry of legal Variants
    DynamicWorkload + Registry
        -> Selector::select(...)
            -> Selection
                -> Invocation
                    -> record(...)
```

These relations define project vocabulary. They do not require one generic Rust type for each box.

## Compilation and launch

Use these project relations:

```text
CompiledKernel
    = compile(KernelSource, KernelConstants)

KernelLaunch
    = CompiledKernel
    + KernelArguments
    + KernelExecutionConfiguration
```

The concrete Metal backend type is `metal::CompiledKernel`. It owns one compiled compute pipeline. A
component-specific type can wrap a `CompiledKernel` with operation constants or variant identity.

`KernelConstants` contains values that are fixed when the project compiles or looks up a `CompiledKernel`.
`KernelArguments` contains launch data such as buffers, metadata, and scalar values.
`KernelExecutionConfiguration` contains `grid_dimensions` and `thread_block_dimensions`.

Do not put an executed Grid object in `KernelExecutionConfiguration`. A launch creates the Grid.

All threads in one launch execute the same `CompiledKernel`. All threads can read the same `KernelArguments`. Each
thread has a different `thread_block_index` and `thread_index`.

Do not copy a compile-time constant into `KernelArguments` only because the shader uses the value.

## Constants and specialization

Group compile-time constants by the scope that owns them:

```text
ComponentKernelConstants
├── model or storage geometry
└── kernels
    └── one semantic kernel phase
        ├── thread_block
        │   └── collective thread-block properties
        └── optional lower scope
            └── simdgroup properties
```

Use nesting to show scope. Do not repeat the scope in each nested field name.

- `max_*` is a compile-time upper bound.
- `num_*` is an exact count in its stated scope.
- `required_threads` is a compile-time launch requirement.
- `KernelExecutionConfiguration.thread_block_dimensions` contains the actual launch dimensions.

The required and actual thread counts must match.

Use `specialization` for the act or result of compiling an implementation for fixed constants. Use
`KernelConstants` for the value that contains those constants. Do not name that value `KernelSpecialization`.

The MLC reference uses compile-time constants such as `BLK_M`. It describes the resulting kernels as specialized for
fixed tile shapes. This project uses the same distinction.

## Tasks, tiles, and layouts

A task describes semantic work. For a current non-persistent kernel that has threadblock-level semantic work, use this
model:

```text
ThreadBlockTask
    = derive(
        KernelArguments,
        thread_block_index,
        KernelConstants,
      )

ThreadTask
    = derive(
        ThreadBlockTask,
        thread_index,
        KernelConstants,
      )
```

The `1:1` relation between a ThreadBlock and a ThreadBlockTask is a kernel contract. It is not a global GPU hardware
rule. Each component must state its relation.

`ThreadBlockTask` and `ThreadTask` are semantic definitions. The CPU does not have to construct or upload task
objects. A kernel can derive task coordinates from grid indices and compact metadata.

Do not add a Task name to a flat elementwise dispatch only for symmetry. Describe its tensor map and Grid when a
threadblock does not own a distinct semantic work unit.

A tile is a bounded tensor subregion that an implementation reuses or computes as one unit. A tile is not a launch,
an execution configuration, or a generic task object.

For GEMM, `BM`, `BN`, and `BK` describe operand and output subregions. `BK` is an implementation loop extent. For SDPA
or recurrence, use the real coordinate domains, such as Q tokens, KV tokens, heads, or state rows. Do not force these
domains into GEMM names.

A layout maps logical coordinates to storage locations or thread ownership. State its coordinate domains and owning
scope. Do not use `Tile`, `Task`, or `Layout` as an unqualified substitute for this information.

## Execution variants

A `Variant` is the complete unit that a runtime selector can choose:

```text
Variant
├── algorithm or execution-path identity
├── one or more kernel-family identities
└── matching KernelConstants
```

Do not create separate selector layers for algorithm choice and shape-specific kernel choice. One registry entry must
identify the complete selectable combination. GQA SlicedKv and SlicedQ entries each identify the Map and Reduce kernels
and their matching constants.

`Variant` can own ready `CompiledKernel` values. It can also contain stable descriptions that another owner uses to
compile or look up those kernels. The component contract determines this ownership.

Use `VariantKey` only when the key has a real consumer. Valid consumers include replay identity, logging, cache lookup,
and deterministic test assertions.

## Registry and selector ownership

Use this component-local owner structure when the component has runtime-selectable variants:

```text
Compute
└── Selector
    └── Registry
        └── Vec<(VariantKey, Variant)>
```

`Compute` represents the component execution owner in this diagram. Keep a more precise existing owner name such as
`Matmul` or `GatedMoE` when that name describes the operation better.

This structure is the target convention for new and refactored source. The current component table records selection
semantics. It does not claim that each current owner already has this exact field structure.

The module path supplies the component scope. Use the short private names `Registry`, `Selector`, `VariantKey`,
`Variant`, `Workload`, and `Selection`. Do not repeat the component name in each private item. A public cross-module API
can use a longer name such as `ExecutionVariant` when `Variant` is not clear at the caller.

`Registry` owns the legal variants for one static component configuration. `Selector` owns the `Registry` and any
static limits or tuning data that selection needs. The component execution owner owns the `Selector`.

Use a `Vec<(VariantKey, Variant)>` by default. Current registries are small, selectors scan candidates, and deterministic
order is useful. Use another data structure only when measured scale or lookup behavior requires it.

## Registry lifecycle

Use this local API vocabulary:

```rust
struct Registry {
    variants: Vec<(VariantKey, Variant)>,
}

impl Registry {
    fn new(/* static inputs */) -> Self;
    fn get(&self, key: VariantKey) -> &Variant;

    // Optional private construction helper.
    fn insert(&mut self, key: VariantKey, variant: Variant);
}
```

The snippet defines names and ownership. It is not a generic Rust interface.

`Registry::new(...)` must construct the complete legal variant set. Static capability checks determine which variants
are legal. These checks can use dtype, model geometry, storage geometry, GPU capability, and required feature support.
They must not use dynamic workload performance thresholds.

A private `Registry::insert(...)` can build the Registry incrementally. It must reject a duplicate `VariantKey` at
initialization. The Registry must be immutable when `Registry::new(...)` returns.

Do not add `Registry::set(...)`. A `set` operation would imply that invocation or recording code can replace a
registered variant. Runtime mutation would invalidate selection, replay identity, and compiled-kernel ownership.

Do not expose `insert(...)` to invocation or recording code.

## Selection lifecycle

Use this local API vocabulary:

```rust
struct Selector {
    registry: Registry,
    // Optional static limits or tuning data.
}

impl Selector {
    fn new(registry: Registry /* optional selector inputs */) -> Self;
    fn select(&self, workload: Workload<'_>) -> Selection;
}

impl Compute {
    fn invoke(&self, /* dynamic inputs */) -> Invocation<'_>;
}

impl Invocation<'_> {
    fn record(&self, /* recorder inputs */);
}
```

Use the same method names where these contracts match:

- `Registry::new(...)`
- `Registry::get(...)`
- `Registry::insert(...)` for optional private construction
- `Selector::new(...)`
- `Selector::select(...)`
- `Compute::invoke(...)`
- `Invocation::record(...)`

`Selector::select(...)` is the component's dynamic selection operation. It must evaluate only legal variants. It must
use the dynamic workload, candidate costs, and selector-owned tuning data.

The selector must compare the complete candidate cost. If a candidate needs task partitioning, scratch extents, replay
extents, or metadata extents, the selector must derive these coupled values before it compares candidates. Do not add a
separate generic Planner layer.

`Compute::invoke(...)` must select before it creates an Invocation. The Invocation must retain the selected Variant or
all identity and derived values that recording needs. `Invocation::record(...)` must use this frozen result. It must
not make a different choice.

A component can call the same pure selector again at a separate topology boundary when it cannot carry the original
selection across that boundary. Both calls must use equivalent inputs. Do not copy the decision logic into another
helper.

## Selection results

Use the smallest result that preserves one decision:

```text
Simple result
    (VariantKey, &Variant)

Coupled result
    component-local Selection
    ├── VariantKey or Variant
    ├── derived task partitioning
    ├── scratch and metadata extents
    └── replay-topology identity
```

The key is optional when no consumer needs it.

Use a component-local `Selection` when metadata, replay, and recording must consume multiple coupled values. GQA uses
this form because the chosen SDPA variant changes Q ranges, KV ranges, partial-state groups, and replay extents.

Do not add a generic `Plan` type. `ExecutorHibernationPlan` and `StateSnapshotPlan` describe state-persistence requests.
They are not GPU execution selections.

Do not add a cross-component `Registry`, `Selector`, `Variant`, or `Selection` trait. The repository has no generic
caller that can use such a trait. A trait would enforce method spelling, but it would not enforce component capability,
cost, metadata, replay, or recording invariants. Use the same local structure and audit each component at its owner.

## Active work and recorded capacity

Kernel-variant selection and replay-capacity selection are different concepts:

```text
dynamic semantic shape
    -> Variant selection
    -> recorded command topology

active work count
    -> replay bucket policy
    -> recorded capacity
```

A replay key must identify each choice that changes a compiled kernel or recorded command graph. Do not add an active
count to the key when a replay parameter can provide that count without changing topology.

A component must use one operation name for both cases. The API must carry `num_active_*` and `num_total_*` values when
the distinction is relevant. A caller without capacity padding sets `num_active_* == num_total_*`. Do not encode the
capacity policy in names such as `select_exact(...)`, `select_bucketed(...)`, or `*_with_token_capacity(...)`.

## Fixed components and delegated selection

A component with one fixed implementation can omit `Registry`, `Selector`, and `Selection`. It must still use clear
`KernelConstants`, task, argument, and execution-configuration names. Add selection objects when a second real runtime
variant appears. Do not add an empty framework only for visual symmetry.

A compound component can delegate a leaf decision to the leaf owner. It must not select the same leaf variant again.
For example, dense MLP and unembedding delegate the row-dependent QMV/QMM choice to `affine_quantized::Matmul`.

## Current component mapping

This table identifies the current selection owner. Component documents contain the detailed source and task contracts.

| Component | Selectable unit | Selection owner and result |
| --- | --- | --- |
| GQA SDPA | Complete SplitKV Map/Reduce execution variant. | `gqa::sdpa::Selector` returns a rich component-local `Selection`. |
| BiDiBlockGQA history SDPA | Complete SplitKV Map/Reduce execution variant plus fixed proposal capacity. | `bidi_block_gqa::sdpa::Selector` returns a component-local `Selection` with the variant and `BiDiBlockGQACapacity`. |
| GDN | Fixed mixed graph with chunkwise and replay kernels. The request split is a submission parameter. | `gdn::compute::Compute` owns kernels and initialization constants directly. |
| Quantized affine | QMV or QMM kernel for the runtime row count. | `affine_quantized::Selector` returns the selected kernel entry. |
| BF16 matmul | GEMV or Steel GEMM kernel for the runtime row count. | `matmul_bf16::Selector` returns the selected Variant. |
| Dense MLP | No independent outer variant. | Each affine owner selects QMV or QMM. |
| Unembedding | No independent outer variant. | Its affine owner selects QMV or QMM. |
| MoE | Complete token-major or expert-major command graph. | The MoE `Selector` returns `(VariantKey, &Variant)`. |
| Sparse expert MLP | No independent outer command-graph variant. | MoE selects the outer graph. The sparse leaf records expert compute. |
| Top-K sampling | Map implementation for dtype, output contract, and Top-K width. | The Top-K `Selector` returns the selected Map Variant. |
| DFlash2 candidate selector | One fixed predecessor-map, edge-score, and path-walk graph. | No runtime Registry or Selector. |
| Dynamic grouped convolution, embedding, row gather, normalization, and residual operations | One current fixed implementation per initialized configuration. | No runtime Registry or Selector. |
| Sparse rejection sampling | One current fixed request kernel. | Replay buckets select capacities, not kernels. |
| DSpark Markov sampling Map | One current fixed Map implementation. | No runtime Registry or Selector. |

Keep these intentional differences:

- Embedding is a quantized row lookup. Unembedding is an adaptive quantized matrix multiplication. They do not share a
  selector.
- Dense MLP applies shared weights to all rows. Sparse expert MLP applies expert-indexed weights to routed rows. MoE
  owns the outer routing and command-graph choice.
- Top-K sampling partitions a vocabulary row. Rejection sampling processes one ordered request. They do not share a
  ThreadBlockTask.
- GQA selection materializes coupled SplitKV work. GDN records a fixed graph and selects active work at submission.
  Do not add GQA partial-state or SplitKV concepts to GDN.

See these component documents:

- [GQA Executor](executor_gqa.md)
- [GQA SDPA selection](gqa_sdpa_selection.md)
- [GDN Executor](executor_gdn.md)
- [Dense MLP Executor](executor_dense_mlp.md)
- [MoE Executor](executor_moe.md)
- [Sampling Executor](executor_sampling.md)
- [Model primitives](executor_model_primitives.md)

## Review checklist

For each new runtime-selectable component, document and review these items:

1. Name the static configuration and legal capability checks.
2. Name the complete Variant that the selector chooses.
3. Name the dynamic Workload fields and their coordinate domains.
4. Put all dynamic choice logic in `Selector::select(...)`.
5. State whether the result is a tuple or a component-local `Selection`.
6. Freeze the choice before recording, or repeat the same pure selector with equivalent inputs.
7. Include each topology-changing choice in the replay identity.
8. State the ThreadBlock-to-ThreadBlockTask relation for each kernel that has semantic tasks.
9. Keep the Registry immutable after initialization.
10. Update the matching component document with the source change.
