# Documentation Guide

Start with the [project README](../README.md) to run a first request.
Use this index to find the owner of a question. Read only the documents relevant to the task.
Agents start with [`AGENTS.md`](../AGENTS.md) for repository instructions.
For source changes, read [high-level guidance](high_level.md) and [engineering conventions](engineering_conventions.md).
For documentation changes, use [technical English](technical_english.md).

## Choose a task

| Task | Start here | Add when relevant |
| --- | --- | --- |
| Run or integrate the service | [Service](service.md) | [Pi provider](../agent-plugins/pi/README.md) |
| Change scheduling or cache lifecycle | [Runtime core](core.md) | [Token-budget allocation](token_budget_allocator.md) |
| Change model execution or replay composition | [Executor architecture](executor.md) | [Qwen executor](executor_qwen.md) and the component below |
| Change GPU recording or a kernel | [GPU execution](gpu_execution.md) | [Metal backend](../crates/inference-backend-metal/README.md) and the component below |
| Verify correctness or investigate performance | [Verification and benchmarks](executor_benchmarks.md) | [Component latency data](component_performance.md), the affected component, and [active work](future_work.md) |

For one request, follow the ownership chain:

```text
service API -> runtime core -> model executor -> Metal backend
               scheduling     model semantics   GPU execution
```

## Find a component

Each guide identifies its current source and contracts.
The Qwen guide owns whole-model composition. The other guides own the named computation or lifecycle.

| Component or lifecycle | Guide |
| --- | --- |
| Qwen model composition | [Qwen executor](executor_qwen.md), [Qwen4x](executor_qwen4x.md) |
| Embedding, normalization, and unembedding | [Model primitives](executor_model_primitives.md) |
| GQA / QSA | [GQA](executor_gqa.md), [QSA SDPA](executor_gqa.md#qsa-sdpa), [QSA indexer](executor_qsa_indexer.md), [kernel selection](gqa_sdpa_selection.md) |
| Gated DeltaNet | [GDN](executor_gdn.md) |
| Gated residual | [GR](executor_gated_residual.md) |
| N-gram embedding | [N-gram embedding](executor_ngram_embedding.md) |
| PLE | [Per-layer embedding](executor_ple.md) |
| Dense MLP | [Dense MLP](executor_dense_mlp.md) |
| MoE | [MoE](executor_moe.md) |
| Sampling and rejection sampling | [Sampling](executor_sampling.md) |
| Qwen3.5 MTP | [Scheduler/executor protocol](mtp_design.md#schedulerexecutor-protocol) |
| Qwen3x DSpark | [DSpark](dspark_design.md) |
| Qwen3x DFlash2 | [DFlash2](dflash2_design.md) |
| Qwen3-ASR | [Audio transcription](qwen3_asr.md) |
| Whole-model Stop/Start | [Executor hibernation](executor_hibernation.md) |
| Snapshot I/O and request/cache movement | [Model state I/O](model_state_io.md) |

Current component documents describe current `src`, including the MTP, DSpark, and DFlash2 `*_design.md` files.
`model_state_io.md` marks implemented and planned work separately.
[Future work](future_work.md) tracks active investigations and incomplete work.

The Firecracker [setup](firecracker/setup.md) and [commands](firecracker/commands.md) are unsupported legacy references.

## Maintenance rules

Give each document one primary purpose:

- Put durable repository rules in `engineering_conventions.md`.
- Put component contracts, source paths, and findings in the guide for that component.
- Put shared verification and benchmark commands in the workflow that owns them.
- Put active follow-up work in `future_work.md`. Link to a focused design document when an unresolved contract needs more detail.

Recommendation: Link to the owner with a one-sentence description instead of duplicating its contract.
Keep a component command only when it explains the production path or its flags.
Keep headings navigable and source paths current.
Consolidate or delete stale prose when a stable rule moves to an owning document.
Do not describe desired future state as current API or source.
Do not add broad historical note directories.
Apply [technical English](technical_english.md) to new and revised English documentation.

Follow the [performance evidence rules](executor_benchmarks.md#performance-evidence) for measurements.
Run GPU and performance commands one at a time.
