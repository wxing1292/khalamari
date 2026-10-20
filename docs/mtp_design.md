# Qwen3.5 MTP Protocol

This document defines the current scheduler/executor protocol.
Examples use three logical MTP modules, `K = 3`, and two tokens per cache block, `N = 2`.
The block size is illustrative, not a production default.

Use these sections for specific questions:

| Question | Section |
| --- | --- |
| What do runtime core and the model executor exchange? | [Scheduler/executor protocol](#schedulerexecutor-protocol) |
| Which hidden outputs survive a step? | [Hidden-state buffer](#hidden-state-buffer) |
| How do Prefill and Decode transitions work? | [Request transitions](#request-transitions) |
| Which formulas determine tokens and retained rows? | [Calculation summary](#calculation-summary) |
| Which checkpoint, replay, and workspace contracts apply? | [Execution and checkpoint contract](#execution-and-checkpoint-contract) |
| Where are the implementation and verification paths? | [Source and verification](#source-and-verification) |
| How does runtime core commit CPU token metadata? | [Scheduler cache commit](#scheduler-cache-commit) |
| How is MTP GPU latency measured? | [GPU timing](#gpu-timing) |

## GPU timing

| Timing field/boundary | Contract |
| --- | --- |
| Logical MTP interval | Hidden transfer, embed/body layers, gather/unembed, sampling |
| `Qwen35MTPExecution` | Sum valid intervals after existing waits, no added synchronization |
| Excluded | Host recording, proposal readback, submission gaps |
| Missing interval | Complete batch total unavailable |
| `ModelOutputTiming::spec_replay_gpu_elapsed` | Existing `spec_gpu_ms` metric, scheduler `mtp` GPU latency row |

`MetalRuntime` enables relaxed timestamps by default. [Service logging](service.md) owns controls and Main boundaries.

After Main completes, rejection decisions fix the accepted GDN prefix.
Before MTP embedding, the executor submits [GDN state commit](executor_gdn.md#state-data-flow) on a separate Metal stream.
MTP can run while this stream reconstructs Main state and publishes accepted snapshots.
The next Main preparation waits before it reuses GDN logs, state slots, or page staging.
The scheduler response remains after the MTP proposal. This change does not advance the runtime core's token commit.

## Hidden-state buffer

After Decode, the executor retains these BF16 hidden outputs for each request:

```text
+-------+----+----+-----+----+
| Index | 0  | 1  | 2   | 3  |
+-------+----+----+-----+----+
| Block | 0  | 0  | 1   | 1  |
+-------+----+----+-----+----+
| Cache | C  | C  | C   | P  |
+-------+----+----+-----+----+
| Main  | t0 | t1 | t2  | w  |
| MTP0  | t1 | t2 | w*  | x1 |
| MTP1  | t2 | w* | x1* | x2 |
| MTP2  | w  | x1 | x2  | x3 |
+-------+----+----+-----+----+
```

| Mark | Meaning                                                                      |
| ---- | ---------------------------------------------------------------------------- |
| `C`  | Every lane has cached KV at this index. Some MTP token IDs can still change. |
| `P`  | Next input diagonal. These tokens have no KV at this column.                 |
| `w*` | The module's final hidden output for `w` is also saved across steps.         |
| `-`  | No cached KV before this step.                                             |
| `.`  | No input row executed at this position.                                    |

`w` is the last Main sampled token. `x1, x2, x3` are its proposed drafts.
The table separates KV storage from the additional hidden buffer:

| Producer | Saved hidden outputs per request | Consumer                           |
| -------- | -------------------------------- | ---------------------------------- |
| Main     | None in the MTP hidden buffer    | MTP0 reads this step's Main output |
| MTP0     | `w`: 1 BF16 row                  | MTP1                               |
| MTP1     | `w, x1`: 2 BF16 rows             | MTP2                               |
| MTP2     | None                             | No next MTP module                 |

The final draft `x3` has no cached KV slot in any lane.
The executor stores `Qwen35MTPCacheState::Decode { token_index: 3, token_ids: [w, x1, x2] }`.
Its `token_index` is the next Main input position. This metadata is not a query.

An MTP input uses the previous module's hidden output at the **same column**.
For example, MTP1 uses MTP0's hidden for `w` to process a replacement for `x1` at index 2.

Ordinary Prefill saves token metadata, but no persistent hidden outputs.
A Prefill that replaces old drafts can read the preceding Decode hidden buffer.
The transition examples below show both cases.

### Read before write

The current implementation uses one reusable buffer, not ping-pong buffers:

```text
Before MTP1 executes:

old MTP0 hidden cache ----+
                          +--> gather --> previous_hidden scratch
this step's MTP0 output --+
          |
          +-----------------> scatter --> MTP0 hidden cache
                                          (Decode requests only)

previous_hidden scratch ------------> RMSNorm --+
                                                +--> concat --> project --> MTP1
input token IDs ----------> embed --> RMSNorm --+

Required order: gather old rows -> scatter new rows -> execute MTP1
```

MTP2 follows the same order for the MTP1 cache.
The executor replaces request metadata only after all MTP steps complete.
Turn completion must not clear this metadata before the next query consumes it.
Request-slot reset changes the metadata to `Empty`.

## Scheduler/executor protocol

The scheduler selects **Main input tokens**. The executor prepares MTP inputs after rejection sampling.

```text
Scheduler                       Model executor
---------                       --------------
Main query + page IDs ---------> Main forward
                                      |
                                      v
                                rejection sampling
                                      |
                                      v
                                MTP0 -> z1 -> MTP1 -> z2 -> MTP2 -> z3
                                      |
accepted tokens + y + drafts <---------+
```

## Request transitions

### Prefill to Prefill

Before: a Prefill processed Main `t0, t1, t2`, with known lookahead `t3, t4, t5`.
Every cell below has KV. No hidden row is retained.

```text
+-------+----+----+----+
| Index | 0  | 1  | 2  |
+-------+----+----+----+
| Block | 0  | 0  | 1  |
+-------+----+----+----+
| Cache | C  | C  | C  |
+-------+----+----+----+
| Main  | t0 | t1 | t2 |
| MTP0  | t1 | t2 | t3 |
| MTP1  | t2 | t3 | t4 |
| MTP2  | t3 | t4 | t5 |
+-------+----+----+----+
```

Stored state: `Qwen35MTPCacheState::Prefill { token_index: 3, token_ids: [t3, t4, t5] }`.
Those three IDs describe the completed cache tail, not the next Prefill query.
The next Prefill needs its Main window **plus three known lookahead tokens**:

```text
QueryTokens::Prefill {
    token_index: 3,
    window: 2,
    tokens: [t3, t4, t5, t6, t7],
}
```

Executed inputs: every module starts at index 3 and runs two rows.

```text
+-------+---+---+---+----+----+
| Index | 0 | 1 | 2 | 3  | 4  |
+-------+---+---+---+----+----+
| Block | 0 | 0 | 1 | 1  | 2  |
+-------+---+---+---+----+----+
| Cache | C | C | C | -  | -  |
+-------+---+---+---+----+----+
| Main  | . | . | . | t3 | t4 |
| MTP0  | . | . | . | t4 | t5 |
| MTP1  | . | . | . | t5 | t6 |
| MTP2  | . | . | . | t6 | t7 |
+-------+---+---+---+----+----+
```

After: every lane has KV through index 4. No token is sampled.

```text
+-------+----+----+----+----+----+
| Index | 0  | 1  | 2  | 3  | 4  |
+-------+----+----+----+----+----+
| Block | 0  | 0  | 1  | 1  | 2  |
+-------+----+----+----+----+----+
| Cache | C  | C  | C  | C  | C  |
+-------+----+----+----+----+----+
| Main  | t0 | t1 | t2 | t3 | t4 |
| MTP0  | t1 | t2 | t3 | t4 | t5 |
| MTP1  | t2 | t3 | t4 | t5 | t6 |
| MTP2  | t3 | t4 | t5 | t6 | t7 |
+-------+----+----+----+----+----+
```

Stored state: `Qwen35MTPCacheState::Prefill { token_index: 5, token_ids: [t5, t6, t7] }`.
No persistent hidden rows are written. Every previous-hidden input comes from this step.

### Prefill to Decode

Before: a Prefill processed Main `t0, t1, t2`, with known lookahead `t3, t4, t5`.

```text
+-------+----+----+----+
| Index | 0  | 1  | 2  |
+-------+----+----+----+
| Block | 0  | 0  | 1  |
+-------+----+----+----+
| Cache | C  | C  | C  |
+-------+----+----+----+
| Main  | t0 | t1 | t2 |
| MTP0  | t1 | t2 | t3 |
| MTP1  | t2 | t3 | t4 |
| MTP2  | t3 | t4 | t5 |
+-------+----+----+----+
```

Stored state: `Qwen35MTPCacheState::Prefill { token_index: 3, token_ids: [t3, t4, t5] }`.
Only `t3, t4, t5` remain. They cannot form a Prefill window with three lookahead tokens.
The scheduler sends all three unprocessed Main tokens as Decode:

```text
QueryTokens::Decode {
    token_index: 3,
    tokens: [t3, t4, t5],
    spec_tokens: [],
}
Result: validated_tokens = [], sampled_token = y, spec_tokens = [z1, z2, z3]
```

Executed inputs: every module starts at index 3 and runs three rows.

```text
+-------+---+---+---+----+----+----+
| Index | 0 | 1 | 2 | 3  | 4  | 5  |
+-------+---+---+---+----+----+----+
| Block | 0 | 0 | 1 | 1  | 2  | 2  |
+-------+---+---+---+----+----+----+
| Cache | C | C | C | -  | -  | -  |
+-------+---+---+---+----+----+----+
| Main  | . | . | . | t3 | t4 | t5 |
| MTP0  | . | . | . | t4 | t5 | y  |
| MTP1  | . | . | . | t5 | y  | z1 |
| MTP2  | . | . | . | y  | z1 | z2 |
+-------+---+---+---+----+----+----+
```

After: every lane has KV through index 5. The stars show the new hidden cache.

```text
+-------+----+----+----+----+----+-----+----+
| Index | 0  | 1  | 2  | 3  | 4  | 5   | 6  |
+-------+----+----+----+----+----+-----+----+
| Block | 0  | 0  | 1  | 1  | 2  | 2   | 3  |
+-------+----+----+----+----+----+-----+----+
| Cache | C  | C  | C  | C  | C  | C   | P  |
+-------+----+----+----+----+----+-----+----+
| Main  | t0 | t1 | t2 | t3 | t4 | t5  | y  |
| MTP0  | t1 | t2 | t3 | t4 | t5 | y*  | z1 |
| MTP1  | t2 | t3 | t4 | t5 | y* | z1* | z2 |
| MTP2  | t3 | t4 | t5 | y  | z1 | z2  | z3 |
+-------+----+----+----+----+----+-----+----+
```

Stored state: `Qwen35MTPCacheState::Decode { token_index: 6, token_ids: [y, z1, z2] }`.
Every previous-hidden input comes from this step. This transition does not need a Prefill hidden cache.
Admission and prefix-cache hits must also leave at least `K` known Main tokens for the first Decode.

### Decode to Decode

#### All reject

Before: Main sampled `w`. MTP proposed `x1, x2, x3`.

```text
+-------+----+----+-----+----+
| Index | 0  | 1  | 2   | 3  |
+-------+----+----+-----+----+
| Block | 0  | 0  | 1   | 1  |
+-------+----+----+-----+----+
| Cache | C  | C  | C   | P  |
+-------+----+----+-----+----+
| Main  | t0 | t1 | t2  | w  |
| MTP0  | t1 | t2 | w*  | x1 |
| MTP1  | t2 | w* | x1* | x2 |
| MTP2  | w  | x1 | x2  | x3 |
+-------+----+----+-----+----+
```

The scheduler submits:

```text
QueryTokens::Decode {
    token_index: 3,
    tokens: [w],
    spec_tokens: [x1, x2, x3],
}
Result: validated_tokens = [], sampled_token = y, spec_tokens = [z1, z2, z3]
```

Executed inputs: Main runs before rejection sampling. MTP runs after the decision.

```text
+-------+---+---+----+----+----+----+----+
| Index | 0 | 1 | 2  | 3  | 4  | 5  | 6  |
+-------+---+---+----+----+----+----+----+
| Block | 0 | 0 | 1  | 1  | 2  | 2  | 3  |
+-------+---+---+----+----+----+----+----+
| Cache | C | C | C  | -  | -  | -  | -  |
+-------+---+---+----+----+----+----+----+
| Main  | . | . | .  | w  | x1 | x2 | x3 |
| MTP0  | . | . | .  | y  | .  | .  | .  |
| MTP1  | . | . | y  | z1 | .  | .  | .  |
| MTP2  | . | y | z1 | z2 | .  | .  | .  |
+-------+---+---+----+----+----+----+----+
```

MTP1 reads cached MTP0 hidden for `w`. MTP2 reads cached MTP1 hidden for `w`.
All other previous-hidden inputs come from this step.

After: all lanes have cached KV through index `3`.

```text
+-------+----+----+----+-----+----+
| Index | 0  | 1  | 2  | 3   | 4  |
+-------+----+----+----+-----+----+
| Block | 0  | 0  | 1  | 1   | 2  |
+-------+----+----+----+-----+----+
| Cache | C  | C  | C  | C   | P  |
+-------+----+----+----+-----+----+
| Main  | t0 | t1 | t2 | w   | y  |
| MTP0  | t1 | t2 | w  | y*  | z1 |
| MTP1  | t2 | w  | y* | z1* | z2 |
| MTP2  | w  | y  | z1 | z2  | z3 |
+-------+----+----+----+-----+----+
```

`Qwen35MTPCacheState::Decode { token_index: 4, token_ids: [y, z1, z2] }` records this completed step.
The next Main query starts at index `4` with `y`.

#### Partial accept

Before: Main sampled `w`. MTP proposed `x1, x2, x3`.

```text
+-------+----+----+-----+----+
| Index | 0  | 1  | 2   | 3  |
+-------+----+----+-----+----+
| Block | 0  | 0  | 1   | 1  |
+-------+----+----+-----+----+
| Cache | C  | C  | C   | P  |
+-------+----+----+-----+----+
| Main  | t0 | t1 | t2  | w  |
| MTP0  | t1 | t2 | w*  | x1 |
| MTP1  | t2 | w* | x1* | x2 |
| MTP2  | w  | x1 | x2  | x3 |
+-------+----+----+-----+----+
```

The scheduler submits:

```text
QueryTokens::Decode {
    token_index: 3,
    tokens: [w],
    spec_tokens: [x1, x2, x3],
}
Result: validated_tokens = [x1], sampled_token = y, spec_tokens = [z1, z2, z3]
```

Executed inputs: Main runs before rejection sampling. MTP runs after the decision.

```text
+-------+---+---+---+----+----+----+----+
| Index | 0 | 1 | 2 | 3  | 4  | 5  | 6  |
+-------+---+---+---+----+----+----+----+
| Block | 0 | 0 | 1 | 1  | 2  | 2  | 3  |
+-------+---+---+---+----+----+----+----+
| Cache | C | C | C | -  | -  | -  | -  |
+-------+---+---+---+----+----+----+----+
| Main  | . | . | . | w  | x1 | x2 | x3 |
| MTP0  | . | . | . | x1 | y  | .  | .  |
| MTP1  | . | . | . | y  | z1 | .  | .  |
| MTP2  | . | . | y | z1 | z2 | .  | .  |
+-------+---+---+---+----+----+----+----+
```

MTP1 keeps its cached `x1` at index 2. MTP2 keeps its cached `x1` at index 1.
Neither module executes those rows again. MTP2 reads cached MTP1 hidden for `x1` at index 2.
All other previous-hidden inputs come from this step.

After: all lanes have cached KV through index `4`.

```text
+-------+----+----+----+----+-----+----+
| Index | 0  | 1  | 2  | 3  | 4   | 5  |
+-------+----+----+----+----+-----+----+
| Block | 0  | 0  | 1  | 1  | 2   | 2  |
+-------+----+----+----+----+-----+----+
| Cache | C  | C  | C  | C  | C   | P  |
+-------+----+----+----+----+-----+----+
| Main  | t0 | t1 | t2 | w  | x1  | y  |
| MTP0  | t1 | t2 | w  | x1 | y*  | z1 |
| MTP1  | t2 | w  | x1 | y* | z1* | z2 |
| MTP2  | w  | x1 | y  | z1 | z2  | z3 |
+-------+----+----+----+----+-----+----+
```

`Qwen35MTPCacheState::Decode { token_index: 5, token_ids: [y, z1, z2] }` records this completed step.
The next Main query starts at index `5` with `y`.

#### All accept

Before: Main sampled `w`. MTP proposed `x1, x2, x3`.

```text
+-------+----+----+-----+----+
| Index | 0  | 1  | 2   | 3  |
+-------+----+----+-----+----+
| Block | 0  | 0  | 1   | 1  |
+-------+----+----+-----+----+
| Cache | C  | C  | C   | P  |
+-------+----+----+-----+----+
| Main  | t0 | t1 | t2  | w  |
| MTP0  | t1 | t2 | w*  | x1 |
| MTP1  | t2 | w* | x1* | x2 |
| MTP2  | w  | x1 | x2  | x3 |
+-------+----+----+-----+----+
```

The scheduler submits:

```text
QueryTokens::Decode {
    token_index: 3,
    tokens: [w],
    spec_tokens: [x1, x2, x3],
}
Result: validated_tokens = [x1, x2, x3], sampled_token = y, spec_tokens = [z1, z2, z3]
```

Executed inputs: Main runs before rejection sampling. MTP runs after the decision.

```text
+-------+---+---+---+----+----+----+----+
| Index | 0 | 1 | 2 | 3  | 4  | 5  | 6  |
+-------+---+---+---+----+----+----+----+
| Block | 0 | 0 | 1 | 1  | 2  | 2  | 3  |
+-------+---+---+---+----+----+----+----+
| Cache | C | C | C | -  | -  | -  | -  |
+-------+---+---+---+----+----+----+----+
| Main  | . | . | . | w  | x1 | x2 | x3 |
| MTP0  | . | . | . | x1 | x2 | x3 | y  |
| MTP1  | . | . | . | x2 | x3 | y  | z1 |
| MTP2  | . | . | . | x3 | y  | z1 | z2 |
+-------+---+---+---+----+----+----+----+
```

Every previous-hidden input comes from this step. The executor does not read old hidden rows.

After: all lanes have cached KV through index `6`.

```text
+-------+----+----+----+----+----+----+-----+----+
| Index | 0  | 1  | 2  | 3  | 4  | 5  | 6   | 7  |
+-------+----+----+----+----+----+----+-----+----+
| Block | 0  | 0  | 1  | 1  | 2  | 2  | 3   | 3  |
+-------+----+----+----+----+----+----+-----+----+
| Cache | C  | C  | C  | C  | C  | C  | C   | P  |
+-------+----+----+----+----+----+----+-----+----+
| Main  | t0 | t1 | t2 | w  | x1 | x2 | x3  | y  |
| MTP0  | t1 | t2 | w  | x1 | x2 | x3 | y*  | z1 |
| MTP1  | t2 | w  | x1 | x2 | x3 | y* | z1* | z2 |
| MTP2  | w  | x1 | x2 | x3 | y  | z1 | z2  | z3 |
+-------+----+----+----+----+----+----+-----+----+
```

`Qwen35MTPCacheState::Decode { token_index: 7, token_ids: [y, z1, z2] }` records this completed step.
The next Main query starts at index `7` with `y`.

### Decode to Prefill

Before: Main sampled `w`. MTP proposed `x1, x2, x3`.

```text
+-------+----+----+-----+----+
| Index | 0  | 1  | 2   | 3  |
+-------+----+----+-----+----+
| Block | 0  | 0  | 1   | 1  |
+-------+----+----+-----+----+
| Cache | C  | C  | C   | P  |
+-------+----+----+-----+----+
| Main  | t0 | t1 | t2  | w  |
| MTP0  | t1 | t2 | w*  | x1 |
| MTP1  | t2 | w* | x1* | x2 |
| MTP2  | w  | x1 | x2  | x3 |
+-------+----+----+-----+----+
```

A new turn or guided auto-completion supplies canonical `w, a, b, c`.
At least one cached identity changes: `a != x1` or `b != x2`.
The scheduler keeps Main's input index and anchor:

```text
QueryTokens::Prefill {
    token_index: 3,
    window: 1,
    tokens: [w, a, b, c],
}
```

Executed inputs: Main runs only `w`. MTP replaces the old speculative tail from the first draft position.

```text
+-------+---+---+---+---+
| Index | 0 | 1 | 2 | 3 |
+-------+---+---+---+---+
| Block | 0 | 0 | 1 | 1 |
+-------+---+---+---+---+
| Cache | C | C | C | - |
+-------+---+---+---+---+
| Main  | . | . | . | w |
| MTP0  | . | . | . | a |
| MTP1  | . | . | a | b |
| MTP2  | . | a | b | c |
+-------+---+---+---+---+
```

MTP1 reads cached MTP0 hidden for `w`. MTP2 reads cached MTP1 hidden for `w`.
All other previous-hidden inputs come from this step.

After: every lane has KV through index 3. No token is sampled.

```text
+-------+----+----+----+---+
| Index | 0  | 1  | 2  | 3 |
+-------+----+----+----+---+
| Block | 0  | 0  | 1  | 1 |
+-------+----+----+----+---+
| Cache | C  | C  | C  | C |
+-------+----+----+----+---+
| Main  | t0 | t1 | t2 | w |
| MTP0  | t1 | t2 | w  | a |
| MTP1  | t2 | w  | a  | b |
| MTP2  | w  | a  | b  | c |
+-------+----+----+----+---+
```

Stored state: `Qwen35MTPCacheState::Prefill { token_index: 4, token_ids: [a, b, c] }`.
The old Decode hidden rows are no longer valid. Prefill writes no new persistent hidden rows.
Main and its GDN state do not rewind. The next query uses the ordinary Prefill or first-Decode rules above.

#### New canonical input

A new turn or auto-completion can also use a **Decode query** when Main processes the whole known suffix.
The query variant does not determine whether cached MTP token IDs match.

Before: Main sampled `w`. MTP proposed `x1, x2, x3`.

```text
+-------+----+----+-----+----+
| Index | 0  | 1  | 2   | 3  |
+-------+----+----+-----+----+
| Block | 0  | 0  | 1   | 1  |
+-------+----+----+-----+----+
| Cache | C  | C  | C   | P  |
+-------+----+----+-----+----+
| Main  | t0 | t1 | t2  | w  |
| MTP0  | t1 | t2 | w*  | x1 |
| MTP1  | t2 | w* | x1* | x2 |
| MTP2  | w  | x1 | x2  | x3 |
+-------+----+----+-----+----+
```

Again, `a != x1` or `b != x2`. This time Main processes all four canonical tokens:

```text
QueryTokens::Decode {
    token_index: 3,
    tokens: [w, a, b, c],
    spec_tokens: [],
}
Result: validated_tokens = [], sampled_token = y, spec_tokens = [z1, z2, z3]
```

Executed inputs: Main does not repeat `t0, t1, t2`. MTP replaces the old speculative tail.

```text
+-------+---+---+---+---+---+----+----+
| Index | 0 | 1 | 2 | 3 | 4 | 5  | 6  |
+-------+---+---+---+---+---+----+----+
| Block | 0 | 0 | 1 | 1 | 2 | 2  | 3  |
+-------+---+---+---+---+---+----+----+
| Cache | C | C | C | - | - | -  | -  |
+-------+---+---+---+---+---+----+----+
| Main  | . | . | . | w | a | b  | c  |
| MTP0  | . | . | . | a | b | c  | y  |
| MTP1  | . | . | a | b | c | y  | z1 |
| MTP2  | . | a | b | c | y | z1 | z2 |
+-------+---+---+---+---+---+----+----+
```

MTP1 reads cached MTP0 hidden for `w`. MTP2 reads cached MTP1 hidden for `w`.
All other previous-hidden inputs come from this step.

After: every lane has KV through index 6. Decode saves the starred hidden outputs.

```text
+-------+----+----+----+---+---+----+-----+----+
| Index | 0  | 1  | 2  | 3 | 4 | 5  | 6   | 7  |
+-------+----+----+----+---+---+----+-----+----+
| Block | 0  | 0  | 1  | 1 | 2 | 2  | 3   | 3  |
+-------+----+----+----+---+---+----+-----+----+
| Cache | C  | C  | C  | C | C | C  | C   | P  |
+-------+----+----+----+---+---+----+-----+----+
| Main  | t0 | t1 | t2 | w | a | b  | c   | y  |
| MTP0  | t1 | t2 | w  | a | b | c  | y*  | z1 |
| MTP1  | t2 | w  | a  | b | c | y* | z1* | z2 |
| MTP2  | w  | a  | b  | c | y | z1 | z2  | z3 |
+-------+----+----+----+---+---+----+-----+----+
```

Stored state: `Qwen35MTPCacheState::Decode { token_index: 7, token_ids: [y, z1, z2] }`.

## Calculation summary

### Input indices and tokens

| Symbol | Existing query/result field                                            |
| ------ | ---------------------------------------------------------------------- |
| `P`    | `query.token_index`                                                    |
| `T`    | Decode `query.tokens.len()`, with `T > 0`                              |
| `D`    | Decode `query.spec_tokens.len()`, with `0 <= D <= K`                   |
| `A`    | Decode `result.validated_tokens.len()`, with `0 <= A <= D`             |
| `W`    | Prefill `query.window`, with `W > 0`                                   |
| `m`    | Zero-based MTP module index, `0 <= m < K`                              |
| `P'`   | Cached end after the step: `P + T + A` for Decode, `P + W` for Prefill |

Use the query and result fields directly. The following slices use exclusive end indices:

```text
Main Decode input:
    start  = P
    tokens = query.tokens + query.spec_tokens
    rows   = T + D

Matching MTP Decode input, module m:
    tokens = (query.tokens + result.validated_tokens).skip(m + 1)
             + [result.sampled_token]
             + result.spec_tokens.take(m)
    rows   = max(T + A, m + 1)
    start  = P' - rows

Replacement MTP Decode input, module m:
    tokens = query.tokens.skip(1) + result.validated_tokens
             + [result.sampled_token]
             + result.spec_tokens.take(m)
    rows   = T + A + m
    start  = P - m

Main Prefill input:
    start  = P
    tokens = query.tokens[..W]

Matching MTP Prefill input, module m:
    start  = P
    tokens = query.tokens[m + 1..m + 1 + W]
    rows   = W

Replacement MTP Prefill input, module m:
    start  = P - m
    tokens = query.tokens[1..1 + W + m]
    rows   = W + m

Prefill query.tokens.len() = W + K
Every MTP module ends at P'.
```

Matching Decode reuses cached MTP rows. Replacement starts from the old first-draft position for simplicity.
Neither rule repeats a previously committed Main row.

### Hidden rows read, computed, and retained

Each newly executed MTP input produces one new hidden row.
Each input reads one previous-hidden row from the preceding module.
The cells below list counts in module order: `MTP0 / MTP1 / MTP2`.

| Example above                 | New MTP hidden rows | Previous-hidden inputs from this step | Previous-hidden inputs from old cache | Retained after step |
| ----------------------------- | ------------------- | ------------------------------------- | ------------------------------------- | ------------------- |
| Prefill to Prefill            | `2 / 2 / 2`         | `2 / 2 / 2`                           | `0 / 0 / 0`                           | `0 / 0 / 0`         |
| Prefill to Decode             | `3 / 3 / 3`         | `3 / 3 / 3`                           | `0 / 0 / 0`                           | `1 / 2 / 0`         |
| All reject                    | `1 / 2 / 3`         | `1 / 1 / 2`                           | `0 / 1 / 1`                           | `1 / 2 / 0`         |
| Partial accept                | `2 / 2 / 3`         | `2 / 2 / 2`                           | `0 / 0 / 1`                           | `1 / 2 / 0`         |
| All accept                    | `4 / 4 / 4`         | `4 / 4 / 4`                           | `0 / 0 / 0`                           | `1 / 2 / 0`         |
| Decode to Prefill replacement | `1 / 2 / 3`         | `1 / 1 / 2`                           | `0 / 1 / 1`                           | `0 / 0 / 0`         |
| New canonical Decode input    | `4 / 5 / 6`         | `4 / 4 / 5`                           | `0 / 1 / 1`                           | `1 / 2 / 0`         |

For a consumer module `m > 0`, matching Decode reads one old producer row only when `m >= T + A`.
Its offset in the producer's saved rows is `T + A - 1`.
Replacement reads offset `0`. All remaining hidden inputs come from this step.

The flat buffer is module-major, then request-slot-major. Each non-final module `m` retains its last `m + 1` output rows:

```text
logical cache-index range = [P' - (m + 1), P')

R = request-slot capacity
H = hidden dimension

physical row start = R * m * (m + 1) / 2 + req_slot * (m + 1)
physical row end   = physical row start + m + 1

total rows  = R * K * (K - 1) / 2
total bytes = total rows * H * 2          // BF16, one buffer
```

The final module retains no hidden rows. `K = 1` needs no hidden-buffer allocation.

### Cached token identities

The [first table](#hidden-state-buffer) shows stored Decode `[w, x1, x2]` and the pending index 3.
The executor compares only overlapping IDs:

| Incoming IDs at Main index 3    | Repair cached MTP KV?                   |
| ------------------------------- | --------------------------------------- |
| `[w]`, `[w, x1]`, `[w, x1, x2]` | No. A missing suffix is not a mismatch. |
| `[w, x1, x2, x3]`               | No. The cached prefix matches.          |
| Canonical `[w, x1, x2, c]`      | No. Old `x3` has no cached KV slot.     |
| Canonical `[w, a, b, c]`        | Yes, if `a != x1` or `b != x2`.         |
| Different anchor or input index | Protocol violation.                     |

Stored Prefill IDs are canonical. Their overlapping prefix must match.
Stored Decode IDs can contain speculative tokens. A mismatch selects the replacement input rules.
Both metadata variants store `K` IDs, not a Prefill execution window.

Submitted speculative IDs must still match their stored draft distributions.
Submitting fewer drafts shortens verification. It does not itself reject a token or change the prefix-match rule.

## Execution and checkpoint contract

MTP reuses one physical GQA/MLP body layer for `K` dependent logical steps:

```text
Main hidden + Main sampled token
          |
          v
   MTP0 -> sample z1 -> MTP1 -> sample z2 -> ... -> sample zK
          ^ each sample uses shared Main unembedding and draft sampling
```

Only the last input row of each Decode request samples a token. Prefill does not sample.
Earlier input rows update MTP context and provide hidden inputs to the next module.

| Area                      | Current contract                                                                                                                        |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Mode                      | Qwen3.5 MTP and DFlash2 are mutually exclusive                                                                                 |
| Checkpoint                | Exactly one physical GQA body layer, with dense MLP or MoE                                                                              |
| Shared weights            | Main token embedding and unembedding. No dedicated MTP embedding.                                                                       |
| Validation                | Main-compatible hidden/attention/expert geometry, shared scratch geometry, exact tensor manifest, and quantized MTPEmbed projection     |
| CLI                       | `--num-spec-tokens K`, `K > 0`, `--max-tokens-per-request >= K`                                                                         |
| GQA pages                 | One Main lane and `K` MTP lanes. Runtime core owns physical pages.                                                                      |
| Page-table snapshot       | `mtp-gqa-request-page-table`                                                                                                            |
| GDN                       | Main only. State versions have no MTP shift. See [`executor_gdn.md`](executor_gdn.md).                                                  |
| Hidden/workspace lifetime | Hidden cache persists as described above. Current-step token/hidden/logit/sampling scratch is reusable workspace.                       |
| Shared workspace capacity | `B + min(R, B) * (K - 1)` rows for Main budget `B` and request capacity `R`. Covers Decode and Prefill repair, including mixed batches. |
| Replays                   | HiddenStateTransfer, MTPEmbed, body, GatherUnembed, DraftSampling                                                                       |
| Body replay key           | Token capacity, GQA capacity/topology, MLP topology                                                                                     |
| Submission parameters     | Active counts and logical GQA layer index                                                                                               |
| Sequential readback       | Wait/read after each non-final step before preparing the next step                                                                      |
| External lifecycle        | One `submit_spec -> wait -> read_spec` transaction, not the fixed-block DFlash2 lifecycle                                        |
| Draft distribution row    | `req_slot * K + step_index`, containing the exact sampled distribution                                                                  |
| Confidence                | No confidence head. Each returned speculative confidence is `1.0`.                                                                      |

## Source and verification

| Root under `crates/` | Owner files |
| --- | --- |
| `inference-runtime-core/src/compute/request/decoder/` | `query_tokens.rs`, query fields/lane slices |
| `inference-runtime-core/src/runtime/decoder/trie_cache/blocks/trie/` | `api.rs`, MTP/collision tests, token-window/placeholder sanity |
| `inference-executor-core/src/model/qwen/v3_5/` | `config.rs`, `weight_layout.rs` |
| `inference-executor-metal/src/model/qwen/v3_5/` | `component_config.rs`, `executor/mtp.rs` |
| `inference-executor-metal/src/model/qwen/v3_5/mtp/` | `decode_plan.rs`, `hidden_state_cache.rs`, `hidden_state_transfer.rs`, `embed.rs`, `layer.rs`, `mod.rs` |
| `inference-backend-metal/src/operators/` | `row_route.rs`, `row_scatter.rs` |
| `inference-executor-metal/src/sampling/` | `top_k_replay.rs`, `spec_probs.rs` |

[Service](service.md) owns external commands.
[Sampling](executor_sampling.md), [GQA](executor_gqa.md), and [verification](executor_benchmarks.md) own component checks and measurement rules.

## Scheduler cache commit

This section describes **CPU token metadata**, not extra Main or MTP execution.
Allocation, lookup, reservation, and publication operate on whole block columns.
All lanes hit the prefix cache together, or none do.

### Prepare with placeholders

Before: Main has one queued token `w`. The last completed Decode produced `x1, x2, x3`.

```text
+-------+----+----+----+----+
| Index | 0  | 1  | 2  | 3  |
+-------+----+----+----+----+
| Block | 0  | 0  | 1  | 1  |
+-------+----+----+----+----+
| Cache | C  | C  | C  | P  |
+-------+----+----+----+----+
| Main  | t0 | t1 | t2 | w  |
| MTP0  | t1 | t2 | w  | x1 |
| MTP1  | t2 | w  | x1 | x2 |
| MTP2  | w  | x1 | x2 | x3 |
+-------+----+----+----+----+
```

Decode preparation assigns `w` to a Main block slot.
Unknown shifted inputs use `? = Token::default() = u32::MAX` in mutable MTP slots:

```text
+-------+----+----+----+---+
| Index | 0  | 1  | 2  | 3 |
+-------+----+----+----+---+
| Block | 0  | 0  | 1  | 1 |
+-------+----+----+----+---+
| Cache | C  | C  | C  | - |
+-------+----+----+----+---+
| Main  | t0 | t1 | t2 | w |
| MTP0  | t1 | t2 | w  | ? |
| MTP1  | t2 | w  | x1 | ? |
| MTP2  | w  | x1 | x2 | ? |
+-------+----+----+----+---+

queued Main source: [w]
padded source:      [w, ?, ?, ?]
lane windows:      [w]  [?]  [?]  [?]
```

`prepare` reads ready and queued tokens, then pads missing lookahead.
`write_tokens` shifts the source and splits the interval at block boundaries.
The same writer handles Prefill preparation and both commit variants.
The scheduler never sends a placeholder as an executor token input.

The caller supplies the complete interval. Fixed blocks check equal IDs; mutable blocks copy them.
For example, this Prefill spans a semi-immutable block (`S`) and a mutable block (`M`):

```text
write_tokens(0, 3, [t0, t1, t2, t3, t4, t5])

+-------+-------+-------+------+
| Index | 0     | 1     | 2    |
+-------+-------+-------+------+
| Block | 0     | 0     | 1    |
| Type  | S     | S     | M    |
| Cache | -     | -     | -    |
| Write | check | check | copy |
+-------+-------+-------+------+
| Main  | t0    | t1    | t2   |
| MTP0  | t1    | t2    | t3   |
| MTP1  | t2    | t3    | t4   |
| MTP2  | t3    | t4    | t5   |
+-------+-------+-------+------+
```

Immutable blocks also check equal IDs. A different fixed ID or an interval outside allocated blocks is a protocol error.
The interval must be non-empty. The writer does not clip it or advance scheduled/cached progress.

### Commit with sliding overwrite

Main accepts `x1`, rejects `x2, x3`, and samples `y`. MTP proposes `z1, z2, z3`.
Core uses the returned tokens to replace the mutable metadata:

```text
source = [t0, t1, t2, w, x1, y, z1, z2, z3]

write real token IDs -> advance cached progress -> publish eligible block columns
```

After commit: every cached cell has a real token ID. The new pending diagonal is not included in this cached rectangle.

```text
+-------+----+----+----+----+----+
| Index | 0  | 1  | 2  | 3  | 4  |
+-------+----+----+----+----+----+
| Block | 0  | 0  | 1  | 1  | 2  |
+-------+----+----+----+----+----+
| Cache | C  | C  | C  | C  | C  |
+-------+----+----+----+----+----+
| Main  | t0 | t1 | t2 | w  | x1 |
| MTP0  | t1 | t2 | w  | x1 | y  |
| MTP1  | t2 | w  | x1 | y  | z1 |
| MTP2  | w  | x1 | y  | z1 | z2 |
+-------+----+----+----+----+----+
```

Core queues only `y` for Main. It stores `z1, z2, z3` separately for later verification.
The writer uses the same source for every lane:

```text
P = previous Main input index
K = number of MTP modules
N = tokens per block

start  = P.saturating_sub(K)
end    = new cached end
lane j = source[start + j..end + j]
```

Here all three block columns were mutable before the step. The write range is `[0, 5)`:

```text
+-------+------+------+------+------+------+
| Index | 0    | 1    | 2    | 3    | 4    |
+-------+------+------+------+------+------+
| Block | 0    | 0    | 1    | 1    | 2    |
+-------+------+------+------+------+------+
| Cache | C    | C    | C    | -    | -    |
+-------+------+------+------+------+------+
| Write | copy | copy | copy | copy | copy |
+-------+------+------+------+------+------+
| Main  | t0   | t1   | t2   | w    | x1   |
| MTP0  | t1   | t2   | w    | x1   | y    |
| MTP1  | t2   | w    | x1   | y    | z1   |
| MTP2  | w    | x1   | y    | z1   | z2   |
+-------+------+------+------+------+------+
```

The writer splits `[0, 5)` into `[0, 2)`, `[2, 4)`, and `[4, 5)`.
It handles every crossed block, including `K > N`.
Copying an unchanged token ID does not repeat GPU work.

### Storage and publication

| Item                                | Contract                                                                                                   |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `queued_tokens`                     | Canonical Main tokens not yet assigned to block token slots                                                |
| Mutable storage                     | `[Token; N]` plus logical `num_tokens`                                                                     |
| `MutableBlock::write_tokens(token_index, tokens)` | Overwrite or append. Do not advance scheduled/cached progress.                                  |
| Active placeholder                  | Allowed only in noncached mutable MTP slots                                                                |
| Immutable/semi-immutable IDs        | Check that supplied IDs match. Never overwrite fixed IDs.                                                  |
| Proposal truncation                 | Commit the full returned proposal first. Limit future verification without erasing cached tail identities. |

A column can be published only when every lane has cached, canonical token IDs:

```text
column end E must satisfy:
    E <= cached_end
    E + K <= canonical_total

canonical_total includes y, but excludes new drafts.
```

For the partial-accept case above, `cached_end = 5` and `canonical_total = 6`:

```text
+---------+-----+-----+----+----+----+
| Index   | 0   | 1   | 2  | 3  | 4  |
+---------+-----+-----+----+----+----+
| Block   | 0   | 0   | 1  | 1  | 2  |
+---------+-----+-----+----+----+----+
| Cache   | C   | C   | C  | C  | C  |
+---------+-----+-----+----+----+----+
| Publish | yes | yes | no | no | no |
+---------+-----+-----+----+----+----+
| Main    | t0  | t1  | t2 | w  | x1 |
| MTP0    | t1  | t2  | w  | x1 | y  |
| MTP1    | t2  | w   | x1 | y  | z1 |
| MTP2    | w   | x1  | y  | z1 | z2 |
+---------+-----+-----+----+----+----+
```

Block 1 cannot publish because MTP2 still contains speculative `z1`.
Block 2 is incomplete and also contains speculative IDs.
The completed step still has one common cached end across all lanes.
Runtime core does not need persistent per-lane cursors or the executor's MTP input plan.

Before commit, cached mutable IDs can still describe the old proposal while canonical input describes its replacement.
The debug sanity check validates the old cached rectangle separately from fixed and uncached token windows.
