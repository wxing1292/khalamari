# Technical English Guide

This guide adapts ASD-STE100 Issue 9 for software documentation in this repository.
It is ASD-STE100-informed.
ASD has not certified this guide.
This guide does not claim full ASD-STE100 compliance.
Do not claim certified or full ASD-STE100 compliance without an authorized review.

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

Protected technical text has spelling or syntax that a tool, protocol, source file, or external project controls.
It includes:

- Fenced code and inline code.
- Shell commands and their exact output.
- File names, paths, and module paths.
- Identifiers, types, functions, macros, crate names, and feature names.
- API routes, protocol fields, configuration keys, environment variables, and command-line flags.
- Exact log messages, error messages, user-interface text, and quoted output.
- URLs, link targets, model names, product names, license names, and trademarks.

Do not rewrite protected technical text.
Use code spans or code fences when Markdown supports that format.
Rewrite the surrounding prose to make its purpose clear.
Preserve the destination of every link.
You may improve a link label when the new label keeps the same meaning.

Do not change a command to make its prose shorter.
Split or shorten the text that introduces the command.
During a length review, count each inline identifier, path, link, or quoted string as one word.

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

Keep the original requirement strength during every rewrite.
Do not weaken a requirement or make a recommendation mandatory.

| Wording | Meaning |
| --- | --- |
| `must` | The reader has to satisfy the requirement. |
| `must not` | The requirement prohibits the action or state. |
| `may` | The reader has permission or an option. |
| `can` | The text describes a capability or possibility. |
| `Recommendation:` | The text gives preferred guidance and permits a justified exception. |

Use `must` instead of `shall`.
Use `can` only for capability, not permission.
`should` is not an approved STE word.
The Issue 9 recurring-error table gives `must` as its alternative.
Do not make this replacement when it would strengthen a repository recommendation.
Use `Recommendation:` to identify optional guidance.

Keep assertion strength and lifecycle scope.
For example, do not replace `assert!` with `debug_assert!` during a prose rewrite.
Do not change an owner, exception, condition, default, or performance claim.

## Sentence and word rules

Use approved STE words when they preserve the software meaning.
Use project terminology for subject-specific concepts.
Use American English spelling unless protected text or another repository rule requires different spelling.

Use active voice and name the owner of each action.
Use passive voice only when the actor is unknown.
Use simple verb forms and simple tenses.
Do not use progressive, perfect, or complex auxiliary constructions when a simple form is accurate.

Write a descriptive sentence with no more than 25 words when practical.
Write a procedure sentence with no more than 20 words when practical.
Do not alter protected technical text only to meet a length limit.
Split the surrounding prose instead.

Give one topic in each sentence.
Give one instruction in each procedure sentence.
You may combine actions only when they must occur at the same time.
Write procedure steps in the imperative form.
Put a required condition before its action.

Do not use contractions.
Do not use semicolons.
Use two sentences instead of a semicolon.
Do not omit necessary articles, subjects, or verbs to make a sentence shorter.

Do not use regional words, slang, or unnecessary jargon.
Replace an unnecessary phrasal verb with a direct verb.
Keep a phrasal verb when it is an established technical verb and no direct replacement is accurate.

Keep a multi-word noun to three words when practical.
Write an official technical name in full, even when it has more than three words.
Then define a clear short form if later text needs it.
Use hyphens only when they show a direct relation or preserve an official name.

## Paragraph and Markdown rules

Each paragraph must have one topic.
Limit each paragraph to six sentences.
Start a paragraph with its topic when the connection is not already clear.
Give information in a logical order.

ASD-STE100 controls language, not Markdown formatting.
Repository documents control heading levels, links, tables, code blocks, and source paths.
Repository conventions also control abbreviations and units of measurement.
Use these Markdown rules:

- Use short headings that identify one topic.
- Introduce a code block with one sentence that states its purpose.
- Use a numbered list when order is mandatory.
- Use a bullet list when item order does not affect the meaning.
- Keep list items grammatically parallel.
- Put one procedure action in each numbered item.
- Use a table for exact mappings or repeated field comparisons.
- Keep prose out of a table when paragraphs are clearer.
- Use descriptive link labels and preserve link targets.
- Keep protected technical text in code formatting.

Recommendation: Use the [MTP protocol](mtp_design.md) as the design-document example for people and LLM agents.
Lead with the problem, owner, and contract. Follow with a concrete example before detailed source layout or API signatures.
Prefer ASCII timelines and tables when token positions, counts, ownership, or state transitions are easier to inspect visually.
Define symbols beside the figure. State the example inputs and distinguish them from production defaults.
Keep diagrams narrow and label arrows with the data or event they carry.
Do not repeat the complete diagram in prose or add a diagram that does not answer a reader's question.

Recommendation: Use compact loop pseudocode to explain kernel data flow, one implementation difference at a time.
Label tensor shapes, data types, and storage locations in place.
Use `GMEM` for device memory, `SMEM` for threadgroup memory, and `register` for thread-local values.
Show which threads or groups use each buffer and accumulator.
Put loads, computation, synchronization, and stores at their actual loop scope to show reuse and lifetime.
Distinguish distributed register tensor shapes from per-thread storage.

A note gives information only.
Do not put a required action in a note.
Put the action in the applicable procedure step.

Use an applicable label, such as Warning or Caution, to identify the risk level.
Start safety text with a clear command or condition.
Then identify the risk or possible result.
Do not reduce the stated risk during a rewrite.

## Review checklist

Check all revised documentation before handoff:

- [ ] Protected technical text is unchanged.
- [ ] Commands, paths, identifiers, links, and code examples keep their exact meaning.
- [ ] Architecture owners and boundaries are explicit and unchanged.
- [ ] Requirements, recommendations, permissions, and capabilities keep their original strength.
- [ ] Each concept uses one project term.
- [ ] Descriptive sentences have 25 words or fewer when practical.
- [ ] Procedure sentences have 20 words or fewer when practical.
- [ ] Sentences use active voice unless the actor is unknown or unimportant.
- [ ] Each sentence has one topic.
- [ ] Each procedure sentence has one action unless actions occur at the same time.
- [ ] Required conditions occur before their actions.
- [ ] Paragraphs have one topic and no more than six sentences.
- [ ] Complex information uses a list or table when that structure improves clarity.
- [ ] The prose has no contractions.
- [ ] The prose has no semicolons.
- [ ] Notes contain information only.
- [ ] Warnings and cautions identify the action, risk, and possible result.
- [ ] The text does not claim ASD certification or full ASD-STE100 compliance.
- [ ] The text links the official standard instead of reproducing it.
