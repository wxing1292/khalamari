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
| Qwen3x DFlash2 | [DFlash2](dflash2_design.md) |
| Qwen3-ASR | [Audio transcription](qwen3_asr.md) |
| Whole-model Stop/Start | [Executor hibernation](executor_hibernation.md) |
| Snapshot I/O and request/cache movement | [Model state I/O](model_state_io.md) |

Current component documents describe current `src`, including the MTP and DFlash2 `*_design.md` files.
`model_state_io.md` marks implemented and planned work separately.
[Future work](future_work.md) tracks active investigations and incomplete work.

The Firecracker [setup](firecracker/setup.md) and [commands](firecracker/commands.md) are unsupported legacy references.

## Maintenance rules

| Content | Owner |
| --- | --- |
| Durable rules | [Engineering conventions](engineering_conventions.md) |
| Current contracts, source paths, findings | Component guide |
| Shared verification/benchmark commands | Workflow guide |
| Active investigation or incomplete work | [Future work](future_work.md) |
| English style | [Technical English](technical_english.md) |
| Measurement validity | [Performance evidence](executor_benchmarks.md#performance-evidence) |

Link to the owner. Delete stale duplication instead of creating historical note directories.
Keep headings, source references, and commands current.
Planned work must not appear as current API.
Run GPU/performance commands one at a time.
