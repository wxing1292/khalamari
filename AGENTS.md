# Repository Instructions

## Read the owning guidance

Read [high-level guidance](docs/high_level.md) first. It owns shared rules and architecture boundaries.
Use the [README](README.md) for setup and the crate map.
Use the [documentation index](docs/README.md) to find the affected component. Read only the focused documents needed for the task.

| Task | Required guidance |
| --- | --- |
| Source or API change | [Engineering conventions](docs/engineering_conventions.md) and the owning component document |
| Rust navigation or refactoring | [Rust semantic workflow](docs/engineering_conventions.md#rust-semantic-workflow) |
| Server, client, or logging change | [Service](docs/service.md) |
| Metal correctness or performance | [Executor verification](docs/executor_benchmarks.md) |
| Documentation change | [Technical English](docs/technical_english.md) |

## Design from contracts

Identify required behavior, inputs, outputs, invariants, and lifecycle before selecting an implementation.

```text
caller inputs/outputs + invariants + lifecycle
  -> runtime core: scheduling, request/cache lifecycle, token/block/page metadata
  -> model executor: model layout, computation, local page interpretation
  -> backend: kernels, resources, dispatch
```

- Use the simplest design that satisfies the contract. Remove redundant state, branches, wrappers, and abstractions.
- Follow existing naming, structure, APIs, and code style. Match peer contracts and keep real semantic differences explicit.
- Do not reshape production `src` only to simplify tests or benchmarks. Do not add entities only for visual symmetry.
- Keep items private unless an intentional API needs `pub`. Do not use `pub(crate)` or `pub(super)`.
- Cached replay keys contain `num_total_*`, topology, and record-time facts. Submission supplies separate `num_active_*`, even when equal.

## Establish trust, then use invariants

| Boundary | Required behavior |
| --- | --- |
| External input/configuration | Validate once at its owner, return shared typed `Error` for recoverable failures |
| Trusted private path | Ordinary arithmetic and lossless casts, no repeated checks or fallback recovery |
| Runtime/allocation/file/snapshot/narrowing/shader/state-version boundary | Keep required checked arithmetic until the owner proves the domain |
| Internal invariant bug | Assertion or panic, never a recoverable error |
| Initialization/one-time ownership/release contract | Release `assert!` permitted |
| Repeated internal check | `debug_assert!`, no release hot-path noise or owner-proved rechecks |

## Verify the changed contract

Use the existing test methodology. Test changed behavior and meaningful boundaries. Do not add tests only to mirror the implementation.
Run commands from the repository root. Match checks to the changed contract:

| Change | Verification |
| --- | --- |
| Documentation only | Check links, source references, command accuracy, and `git diff --check` |
| Source behavior | Run focused owner tests, then relevant package tests |
| Broad Rust change | Run all compile gates below before handoff |
| Runtime, executor, or RPC acceptance | Exercise the production path through an external caller, following the service and executor guides |

Format Rust with `cargo +nightly fmt`. The compile gates are:

```sh
cargo +nightly fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo +nightly clippy --workspace --all-targets --all-features -- -D warnings
git diff --check
```

Run Metal/GPU and perf/bench commands one at a time across agents and processes.
Do not use parallel workspace tests as a GPU gate.
Before a performance claim, record commit, dirty state, model, command, environment, metric, baseline, current result, and verdict.
Keep force-sync/profile-summary results separate from normal wall-clock throughput.

## Explain with data-flow diagrams

Prefer ASCII tables/graphs to prose. Technical explanations must include ASCII data flow.
Use [Steel output tiles](docs/executor_dense_mlp.md#steel-output-tiles) and
[sliced-K tiles](docs/executor_dense_mlp.md#sliced-k-tiles) as references.

| Explanation | Required diagram detail |
| --- | --- |
| Computation | One concrete input, axes/shapes/layouts, worker reads, intermediate computation, output writes, reuse/reduction |
| Before/after change | Same input, unchanged stages visible, changed movement/computation/ownership marked |
| GPU kernel | Tensor partition, grid, threadgroup size, SIMDgroup ownership, concrete coordinates, final reduction |
| Routing/packing | Concrete source/packed rows, source token and destination, partial tiles/inactive work |

Explain counts and latency after data flow. Operation names connected by arrows alone are insufficient.

## Document and deliver

Use ASD-STE100-informed prose. Preserve technical meaning, requirement strength, and exact technical text.
Update the owning component document in the same change when source layout or default paths change.
This includes GQA, GDN, dense MLP, MoE, sampling, and MTP.
Update `docs/service.md` when service commands or logging change.
Current-component documents describe current `src`. Put active follow-up work in `docs/future_work.md`.

Preserve unrelated work. Keep delegated scopes separate and coordinate shared resources.
Follow the [definition of done](docs/high_level.md#definition-of-done).
Report the change, verification results, and remaining limitations.
