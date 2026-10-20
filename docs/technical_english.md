# Technical English Guide

This guide adapts ASD-STE100 Issue 9 for software documentation in this repository.
It is ASD-STE100-informed.
Do not claim certification or full ASD-STE100 compliance without an authorized review.

ASD published Issue 9 on 2025-01-15.
Use the [official Issue 9 PDF](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf) as the primary reference.
Use the [official downloads page](https://www.asd-ste100.org/STE_downloads.html) to request the current standard.
`ASD-STE100` is a registered ASD trademark.
ASD owns the copyright in the standard.

Use this guide with [`high_level.md`](high_level.md) and [`engineering_conventions.md`](engineering_conventions.md).
Those documents continue to own architecture and engineering rules.
This guide does not replace the standard, its controlled dictionary, trained review, or applicable project rules.
Link to official copies of the standard.
Do not copy or redistribute the standard or its dictionary in this repository.

## Protected technical text

| Preserve exactly | Examples |
| --- | --- |
| Executable text | Code, commands, exact output, equations, numeric values |
| Identifiers | Paths, types, macros, crates, features, APIs, fields, environment variables, flags |
| External text | URLs, link destinations, model/product names, licenses, trademarks, quotations |

Use code formatting for protected technical text. Rewrite only the surrounding prose.
Do not shorten a command or alter its meaning to meet a word limit.
Count each inline identifier, path, link, or quotation as one word.
Link labels may improve when their meaning and destination remain unchanged.

## Project terminology

ASD-STE100 permits subject-specific technical nouns and technical verbs.
This repository uses software and model terms as project terminology.
Use each term with one meaning and one grammatical function.

| Term | Project meaning |
| --- | --- |
| runtime core | The layer that owns scheduling, request lifecycle, token/block metadata, KV/state page allocation/free, page ownership, and cache/state notifications. |
| model executor | The layer that owns model layout parsing, backend tensor/state objects, components, page interpretation, sampling, profiling, benchmarking, and replay composition. |
| Qwen executor | The Qwen implementation of the model executor. |
| Metal backend | The layer that owns Metal resources, kernels, recording, and submission. |
| request | One runtime-managed inference lifecycle. |
| page | A runtime-owned unit of KV-cache or model-state storage. |
| replay | Recorded backend work that the executor can submit again. |
| component | A model computation with a defined semantic owner, inputs, outputs, and state. |
| operator | A backend operation that lowers one tensor computation. |
| command | One concrete backend dispatch. |

Keep established terms such as GQA, GDN, dense MLP, MoE, MTP, KV cache, and ICB.
Define an abbreviation at first use if the intended readers are not familiar with it.
Do not invent a synonym for variation.
Do not use `runtime`, `executor`, or `cache` alone when the shorter term creates ambiguity.

Use a direct, approved word when it keeps the technical meaning.
Keep an established software verb when a general replacement would change that meaning.
Examples include `decode`, `debug`, `download`, `install`, `record`, `replay`, and `submit`.

## Normative wording

| Wording | Meaning |
| --- | --- |
| `must` / `must not` | Requirement / prohibition |
| `may` | Permission |
| `can` | Capability or possibility |
| `Recommendation:` | Preferred non-binding guidance |

Preserve requirement strength, owner, exceptions, conditions, defaults, and performance claims.
Use `must` instead of `shall`.
Replace `should` with `Recommendation:` for advice. Do not turn advice into a requirement.
Preserve assertion lifecycle and strength, including `assert!` versus `debug_assert!`.

## Sentence and word rules

| Rule | Limit or exception |
| --- | --- |
| Active voice, explicit actor, simple tense | Passive voice only when the actor is unknown or unimportant |
| One topic/sentence | One action/procedure sentence, unless actions must coincide |
| Descriptive sentence | At most 25 words when practical |
| Procedure sentence | At most 20 words when practical |
| Conditions | Before the action |
| Multi-word noun | At most three words when practical, preserve official names |
| Vocabulary | Established project terms, American English, no synonyms for variety |
| Grammar | No contractions or semicolons, keep necessary articles/subjects/verbs |
| Hyphens | Only for a direct relation or protected name |

Use approved STE words when they preserve software meaning.
Do not alter protected technical text to meet a length limit.

## Paragraph and Markdown rules

```text
problem + owner + contract -> concrete input -> ASCII data flow/table
  -> necessary detail -> commands or source references
```

Prefer an ASCII table or graph when it conveys the same information more directly than prose.
Do not repeat a figure in prose or add a diagram without a reader question.
Use the [MTP protocol](mtp_design.md) as a design-document reference.

| Structure | Use |
| --- | --- |
| Paragraph | One topic, at most six sentences |
| Heading | Short, one identifiable topic |
| Numbered list | Mandatory order, one action per step |
| Bullet list | Parallel items without mandatory order |
| Table | Mappings and repeated field comparisons |
| Code block | Exact commands/code, introduced by a short purpose statement |
| ASCII diagram | Ownership, data movement, token positions, counts, state transitions |

Define symbols beside figures. Distinguish example inputs from production defaults.
Keep diagrams narrow. Label arrow payloads, shapes, dtypes, workers, and reductions.
For kernels, show loop scope, reuse, synchronization, loads/stores, and storage lifetime.
Use `GMEM`, `SMEM`, and `register` for device, threadgroup, and thread-local storage.
Distinguish distributed register tensor shapes from per-thread storage.

Notes contain information only. Put required actions in procedure steps.
Preserve warning/caution risk levels. State the condition or command before its consequence.
ASD-STE100 controls language. Repository conventions control Markdown, abbreviations, and units.

## Review checklist

| Check | Required result |
| --- | --- |
| Technical text | Commands, identifiers, values, code, links, and quotations preserve meaning |
| Contracts | Owners, boundaries, requirements, permissions, and exceptions are unchanged |
| Terminology | One consistent term per concept |
| Prose | Short active sentences, explicit actors, one topic/action |
| Layout | ASCII tables/graphs replace repetitive descriptions |
| Procedures | Conditions precede actions, notes contain no requirements |
| Claims | No certification claim or redistribution of the standard/dictionary |
