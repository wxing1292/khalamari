# High-Level Engineering Guidance

This document defines shared engineering rules and the boundary between the runtime core, model executor, and Metal backend.
[`AGENTS.md`](../AGENTS.md) gives the agent workflow and verification commands.
The [documentation index](README.md) routes readers to component and workflow documents.

## Doc style

Use [ASD-STE100-informed English](technical_english.md). Prefer ASCII tables and data-flow diagrams over repeated prose.

| Document | Owns |
| --- | --- |
| `AGENTS.md` | Agent workflow, repository map, verification entry points |
| Root README | Project model and first request |
| Documentation index | Task routing and current/planned status |
| Architecture guide | Ownership and data flow |
| Component guide | Current `src`, contracts, validation, findings |
| Workflow guide | Shared commands and operations |
| `future_work.md` | Active investigations and incomplete work |
| `engineering_conventions.md` | Durable engineering rules |

Link to each contract's owner. Do not copy it or create broad historical/performance archaeology directories.
Update the owning component guide when source layout or default paths change.

## Core vs executor

| Layer        | Owns                                                                                                                            | Must not own                                                     |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Runtime core  | scheduling, request lifecycle, token/block metadata, KV/state page allocation/free, page ownership, cache/state notifications                    | model tensor layout, backend details, GQA/GDN/MLP internals      |
| Model executor | model layout parsing, backend tensor/state objects, GQA, Gated DeltaNet, dense MLP, MoE, component-local page interpretation              | scheduling policy, global lifecycle, page allocation/free policy |
| Metal backend  | Metal FFI, device/buffer/kernel/stream/runtime primitives, Apple Silicon component kernels consumed by the model executor                        | request scheduling, global lifecycle, page allocation/free policy |

The runtime core provides stable metadata and page IDs. Executor and backend components use this contract to run the
model.

## Shared hard constraints

| Boundary | Required behavior |
| --- | --- |
| Recoverable failure | Shared typed `Error`, variant selected by caller-visible semantics |
| Internal invariant violation | `panic!`, `assert!`, or `debug_assert!` |
| Release `assert!` | Initialization, one-time structure/ownership checks, or a release-enforced contract |
| Repeated hot-path invariant | `debug_assert!`, without rechecking a fact the owner already proved |
| Visibility | Private by default, plain `pub` for intentional APIs, no `pub(crate)` or `pub(super)` |
| Rust formatting | `cargo +nightly fmt` |
| Module roots | Normal `use` imports before module declarations |

```text
focused owner tests -> relevant package tests + compile gates -> external production caller
                                                              -> expensive GPU acceptance
```

Use the [verification commands](../AGENTS.md#verify-the-changed-contract) that match the changed contract.
Run Metal/GPU commands one at a time. Parallel workspace tests are not a GPU gate.
Run expensive GPU checks at final acceptance, or earlier to resolve a correctness question.
Recommendation: For RPC acceptance, run a release service, exercise its protocol externally, and verify clean shutdown.

Production source defines ownership and lifecycle. Tests and benchmarks must not add production state, compatibility branches, or abstractions only for convenience.
Start from production contracts. Add boundary tests for real structural risks.

Recommendation: Expose child modules with `pub mod`. Avoid parent re-exports without an API need.

## Delegated work

| Owner | Responsibility |
| --- | --- |
| Parent | Scope, integration state, shared-resource serialization, conflicts, status, final verification |
| Child | Its assigned work, progress, questions, and results |

The parent must not duplicate a child's work. Keep scopes separate and remain available for reports.
Do not use busy waiting or block polling.

## Ownership style

Name the managed object before adding APIs.
An entity must own a concept, invariant, resource, lifecycle, reusable operation, or material clarity.
Remove redundant derived state and one-use forwarding helpers.
Keep stable identity, explicit dtype/tuning choices, and meaningful resource views.
Derive counts and capacities from the dimensions that own them.
Recommendation: Use direct data flow. Do not minimize entity count mechanically.

```text
external input/configuration
  -> validate owning boundary once
     positive dimensions -> checked derived counts -> shader domain -> buffer ranges
  -> trusted private path: ordinary arithmetic + lossless casts -> dispatch
```

An upstream initialization boundary can establish the same domain.
The private backend path must not repeat its checks.
Peer components must use the same owning boundary when their contracts match.

| Domain | Boundary |
| --- | --- |
| `u32` shader count | Reject `2^32` |
| `u32` element index | Exactly `2^32` elements are valid, maximum index `u32::MAX` |

Name assertions/tests for the count or index domain they protect.
Keep checked arithmetic at real allocation, file, snapshot, narrowing, shader, runtime, and state-version boundaries.

| API | Meaning |
| --- | --- |
| `get / get_ref / get_mut`, `set`, `push / pop` | Small object-owned operations |
| `reset` | Reset the full managed object at that scope |
| `set_<field>` | Change only that field |
| `new` | Construct an owner, external access only if its contract requires it |
| `load` / `unload` | Acquire/release the same resource shape |
| `load_all` / `unload_all` | Acquire/release every resource in the current domain |

Recommendation: Keep lifecycle pairs symmetric within one owner.
Do not add hot-path cleanup operations to compensate for unclear ownership.

A mapping/index owns data relationships. The store owns path resolution, mapped-file caches, base directory, and resource lifetime.
Use `load(model_dir)` or `from_model_dir(model_dir)` for fixed checkpoint layouts.
Recommendation: Keep `new(model_dir, index)` when callers also need explicit composition.
Do not hide another resource lifecycle behind the same load/unload pair.

An impossible lifecycle state is an invariant bug, not a recoverable error.
Change the structure only when valid inputs can break the data flow or the owner cannot enforce the contract.

## Design style

```text
caller inputs/outputs + ownership + state transitions
  -> compare peer contracts -> select reusable backend implementation
  -> verify intended default -> remove obsolete implementation
```

Recommendation: Use established libraries when they reduce maintenance or correctness risk.
Keep project semantics at their owner, such as Qwen response grammar above a generic incremental tokenizer.

| Peer contracts match | Real semantic/lifecycle difference |
| --- | --- |
| Parallel names, APIs, ownership, replay and tests | Keep the difference explicit at its owner |
| Small symmetric entity when it clarifies the contract | No entity only for visual symmetry |

Peer examples include GQA/GDN state tables, dense MLP/MoE, and sampling/rejection sampling.

Reusable backend operators/components own performance optimizations.
Selection uses dimensions, dtype, quantization, topology, and device capabilities.
It must not use model family names, checkpoint names, or checkpoint paths.
Model implementations adapt checkpoints and compose components from validated semantic inputs.
Model-specific fixtures may validate an optimization but do not justify model-specific routing.

## Model / layer / operator boundary

```text
Model                         Which whole-model stage runs next?
  embedding -> Main layers -> final norm/unembedding -> sampling/rejection
  MTP logical steps, DFlash2 proposal graph, request state, replay caches
    |
Layer / Component             Which semantic computation and state does this own?
  GQA, GDN, MLP, MoE, embedding, sampling, MTP/DFlash2 layer semantics
  typed weights, requests, pages, inputs, outputs, record support
    |
Operator                      How does this operation run on the backend?
  kernels, scratch/bindings, resource access, barriers, replay recording
    |
Backend Command               One dispatch
  one pipeline/function + bindings + geometry + consumer barrier
```

Metal uses one compute-pipeline dispatch per ICB slot.
One component can emit several operators. One operator can emit several commands.
Model code must not bind backend resources when a semantic layer can own that boundary.
Backend operators must not contain model request semantics that a component can translate.

| Boundary trait | Contract |
| --- | --- |
| `ReplayLayer` | Typed model-executor semantic replay |
| `Operator` | Backend-recordable execution |

Recommendation: Use concrete types for clear implementations and traits where boundaries can drift.
Metal semantic layers record directly through `inference_backend_metal::components::ReplayRecorder`.
The model replay-cache owner creates this recorder from its concrete `Stream`.
Executor core owns model execution and submission contracts, model geometry, and CPU references.
Backend executors own recording, replay caches, resource layouts, and capacity choices.
Device, stream, buffer, and recorder types do not require shared backend traits.
Do not expose lower-level builders in component inputs or public executor APIs.
Do not add a wrapper that only repackages arguments.

Recommendation: Separate initialization from runtime work:

```text
initialization/cache build                 runtime hot path
weights + kernels + reusable scratch       current token/page/state metadata
relayout + path resolution                 submit cached/recorded work
kernel-selection tables                    read required small CPU outputs
```

Reuse resources at the narrowest safe owner.
The producer must write directly to a consumer's persistent destination when an intermediate GPU copy is unnecessary.

## Engineering conventions

[Engineering conventions](engineering_conventions.md) own naming, coordinate/numeric domains, Tile/Task vocabulary,
shape/layout separation, public APIs, replay safety, and test style.

## Performance evidence

Use the [complete evidence contract](executor_benchmarks.md#performance-evidence):

```text
commit + dirty state + machine/environment + model/command
  + baseline/current samples + workload/metric + verdict
```

Compare the same work. For Spec, include sampled tokens, chunks/decisions, and acceptance efficiency.
A changed acceptance trajectory alone does not establish an executor/kernel regression.

## Definition of done

Apply the criteria that match the change. A change is done only when:

```text
target behavior is implemented
default path uses the intended logic
obsolete fallback or experimental path is removed or isolated
slow/reference correctness coverage exists where applicable
relevant cargo check/test issues are fixed
relevant benches are runnable
profile keys are coherent
public API did not grow unnecessarily
runtime core vs executor ownership is still clean
current docs are updated if behavior/layout changes
```

## Common anti-patterns

| Pattern | Correct owner or contract |
| --- | --- |
| Scheduler logic in executor components | Runtime core |
| Model tensor-layout parsing in runtime core | Model executor |
| Model-specific reusable kernel policy | Backend operator/component |
| Bench-driven production abstractions | Production ownership and lifecycle |
| Defensive recovery for impossible states | Owning invariant |
| Obsolete paths behind feature/env switches | Verified intended default |
| Extra GPU copies | Producer writes final destination |
| Helper buckets, one-field forwarding wrappers, broad exports | Explicit owner with a real consumer |
| Mechanical tests, large inline modules | Focused owner contracts and sibling test files |
| Premature completion claims | [Definition of done](#definition-of-done) |
