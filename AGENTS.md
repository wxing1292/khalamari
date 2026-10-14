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

- Start from first principles. Identify required behavior, inputs, outputs, invariants, and lifecycle. Put each responsibility in its owning component.
- Runtime core owns scheduling, requests, token/block metadata, page allocation/free, and cache lifecycle.
  The model executor consumes metadata and page IDs. It owns model layout, computation, and component-local page interpretation.
- Apply Occam's razor. Choose the simplest design that satisfies the contract. Remove redundant state, branches, wrappers, and abstractions.
  Do not reshape production `src` only to make benchmarks or tests easier.
- Follow existing naming, structure, APIs, and code style. Match peer components when their contracts match.
  Keep real semantic differences explicit. Do not add entities only for visual symmetry.
- Keep items private unless an intentional API needs `pub`. Do not use `pub(crate)` or `pub(super)`.
- Cached replay takes `num_active_*` at submission. Its key contains `num_total_*`, topology, and other record-time static facts.
  Keep active and total counts separate even when equal.

## Establish trust, then use invariants

- Validate external inputs and configuration at the owning input or initialization boundary.
  Return the shared typed `Error` for recoverable failures, with caller-visible semantics.
- Inside the validated domain, use ordinary arithmetic and direct lossless casts.
  Do not repeat checked operations or add fallback behavior to hide broken invariants.
- Keep checked arithmetic at real runtime, allocation, file, snapshot, narrowing, shader-domain, and state-version boundaries.
  Do not remove a check until the owning boundary proves the required domain.
- Treat an internal invariant violation as a code bug. Use assertions or panics, not recoverable errors.
  Release `assert!` is limited to initialization, one-time structural/ownership boundaries, or contracts that release code must enforce.
  Use `debug_assert!` for repeated internal checks that add release hot-path noise. Do not recheck facts that the owner already proved.

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

- Technical explanations in chat and documentation must use ASCII diagrams. Code snippets can supplement the diagrams.
  Use the [Steel output tiles](docs/executor_dense_mlp.md#steel-output-tiles) and
  [sliced-K tiles](docs/executor_dense_mlp.md#sliced-k-tiles) as reference examples.
- Start with one concrete input. Label tensor axes, shapes, and layouts. Show which input slices each worker reads,
  how it computes intermediate results, and where it writes the output. Show data reuse and reductions when they apply.
- For a change, use the same input in the before and after diagrams. Mark the changed data movement, computation,
  or ownership at the relevant stage. Keep unchanged stages visible so the reader can compare the two paths.
- For a GPU kernel, show the tensor partition, dispatch grid, threadgroup size, and SIMDgroup ownership.
  Map concrete grid coordinates to row, column, K, or expert ranges. Show how partial results form the final output.
- For routing or packing, show concrete rows before and after the mapping. Connect each packed row to its source
  token and destination output. Show partial tiles and inactive work when they affect the design.
- Explain counts and latency after the reader can follow the data. A list of numbers or arrows between operation
  names does not explain a design or its differences.

## Document and deliver

Use ASD-STE100-informed prose. Preserve technical meaning, requirement strength, and exact technical text.
Update the owning component document in the same change when source layout or default paths change.
This includes GQA, GDN, dense MLP, MoE, sampling, and MTP.
Update `docs/service.md` when service commands or logging change.
Current-component documents describe current `src`. Put active follow-up work in `docs/future_work.md`.

Preserve unrelated work. Keep delegated scopes separate and coordinate shared resources.
Follow the [definition of done](docs/high_level.md#definition-of-done).
Report the change, verification results, and remaining limitations.
