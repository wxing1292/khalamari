# GQA SDPA Selection

This document describes the current GQA SDPA selection model.
It separates conceptual task vocabulary from the implemented kernel constants, selection policy, and GPU ABI.
[`executor_gqa.md`](executor_gqa.md) owns the complete GQA component and source layout.

## Reference vocabulary

Use [GPU execution vocabulary](gpu_execution.md) for the shared launch, constant, variant, task, tile, layout, and
selection terms. This section defines the SDPA-specific mapping.

### SDPA kernel constants

The Map constants use this hierarchy:

```text
MapKernelConstants
├── kind
├── thread_block
│   ├── max_q_tokens
│   ├── max_q_heads
│   ├── kv_tokens_per_iteration
│   └── required_threads
└── kv_cache
    └── tokens_per_page
```

The Reduce constants do not describe the KV cache because Reduce does not read it:

```text
ReduceKernelConstants
└── thread_block
    ├── max_q_tokens
    ├── max_q_heads
    └── required_threads
```

All fields under `thread_block` describe collective properties of one threadblock. A `max_*` field is an upper bound.
`kv_tokens_per_iteration` gives the KV interval used to partition Map tasks.
For sliced-Q, it also gives the shared KV staging width.
For sliced-KV, each SIMDgroup processes 16 KV tokens per step independently of this interval.
`required_threads` is a compile-time requirement. It must match the actual threadblock dimensions in the
execution configuration.

`tokens_per_page` belongs to `kv_cache` because the kernel uses it for paged KV address translation. It is not a work
tile extent.

The project does not define a generic `Tile` Rust type for this hierarchy. A concrete kernel can use internal tensor
tiles and data layouts. These implementation details do not change the semantic task boundary.

### SDPA tasks

For the current non-persistent SDPA kernels, one threadblock executes one task. This relation is a kernel contract. It
is not a global GPU programming rule.

```text
one Map ThreadBlock -> one MapThreadBlockTask
one Reduce ThreadBlock -> one ReduceThreadBlockTask
```

Use these semantic derivations:

```text
MapThreadBlockTask
    = derive(
        MapKernelArguments,
        thread_block_index,
        MapKernelConstants,
      )

MapThreadTask
    = derive(
        MapThreadBlockTask,
        thread_index,
        MapKernelConstants,
      )
```

A conceptual Map task has these fields:

```rust
struct MapThreadBlockTask {
    request_index: u32,

    flat_q_token_indices: Range<u32>,
    q_head_indices: Range<u32>,

    kv_head_index: u32,
    request_local_kv_token_indices: Range<u32>,

    partial_state_group_index: u32,
}
```

One Map task defines this Cartesian product:

```text
flat Q-token range
x Q-head range
x one KV head
x request-local KV-token range
```

The Map task produces one `PartialAttentionState` for each `(flat Q token, Q head)` coordinate. The scalar
`partial_state_group_index` identifies the base of this group. It does not identify one scalar partial state.

`MapThreadTask` is shader-derived. The CPU does not have to construct or upload it. The kernel defines the exact thread
and data layout.

The Reduce task follows the same rule:

```rust
struct ReduceThreadBlockTask {
    request_index: u32,

    flat_q_token_indices: Range<u32>,
    q_head_indices: Range<u32>,

    partial_state_group_indices: Range<u32>,
}
```

```text
ReduceThreadTask
    = derive(
        ReduceThreadBlockTask,
        thread_index,
        ReduceKernelConstants,
      )
```

One Reduce task merges one or more `PartialAttentionState` values for each output coordinate. It writes one final
Q-token and Q-head output range. `ReduceThreadTask` is shader-derived.

### SDPA execution

One Map/Reduce SDPA execution contains two ordered launches:

```text
SDPAExecution
├── Map KernelLaunch
│   └── produces PartialAttentionState values
└── Reduce KernelLaunch
    └── merges PartialAttentionState values and writes final output
```

The Reduce launch depends on Map completion. The default lookup uses direct output for unsplit Q-token/head tiles.
An explicit unsplit tiled route can instead write final output in one Map launch.

The generic `PartialAttentionState` boundary is representation-independent. It does not require separate numerator or
denominator buffers.

## Current implementation facts

### Ownership

The GQA component supplies semantic request facts. The Metal backend supplies legal execution variants. The Metal
executor selector owns dynamic work partitioning and complete-candidate comparison.

```text
backend_sdpa::Config
    -> backend_sdpa::Registry
    -> gqa::sdpa::Selector

gqa::sdpa::RequestShape[]
    -> Q-range density and Q-head policy
    -> complete gqa::sdpa::Selection candidates still needed by the policy
    -> D=256 page-16 cost comparison, when required
    -> selected gqa::sdpa::Selection
    -> GQAMetadataBuffers::update(...)
    -> Map kernel
    -> Reduce kernel
```

The runtime core does not select an execution variant. It continues to own request scheduling, page allocation,
and cache lifecycle.

### Static variants

`backend_sdpa::Config` contains `io_dtype`, `num_q_heads`, `num_kv_heads`, `head_dim`, and `tokens_per_page`.

`backend_sdpa::Registry` performs static capability filtering. It does not use batch lengths or performance
thresholds. Each legal `backend_sdpa::ExecutionVariant` contains one compatible Map and Reduce pair.

The current Rust constant fields follow the reference hierarchy:

```text
backend_sdpa::ExecutionVariant
├── map
│   ├── kind
│   ├── thread_block
│   │   ├── max_q_tokens
│   │   ├── max_q_heads
│   │   ├── kv_tokens_per_iteration
│   │   └── required_threads
│   └── kv_cache
│       └── tokens_per_page
└── reduce
    └── thread_block
        ├── max_q_tokens
        ├── max_q_heads
        └── required_threads
```

`map.kind` identifies `SlicedKv`, `QTokensPerHead`, `QHeadsPerToken`, or `QTokensAndHeads`.
The token capacity does not identify the algorithm.
A sliced-Q variant can also have `max_q_tokens = 1`.
The complete variant remains part of the replay topology.

`Selector::for_lookup` registers the matrix and SlicedKv geometries reachable from the compact lookup.
`backend_sdpa::Registry::new` retains the original candidate set for explicit and composite callers.
`ExecutionVariant::sliced_q` exposes additional generic Q-token/head assignments for explicit callers.
These variants support token capacities `1, 2, 4, 8, 16, 32` and KV staging `8, 16, 32, 64, 128`.
The physical mapping must fit 1024 threads and the compiled pipeline limit.
Their BF16/FP8 cache profiles remain `D128/P8,16` and `D256/P8,16,32`.
The additional mappings support GQA ratios above eight through explicit head ranges.

`QHeadsPerToken` pads each token's head extent to a multiple of eight.
Six heads therefore use eight physical head positions.
`QTokensAndHeads` rounds the complete token/head product to a multiple of eight.
A `4 x 6` tile uses 24 Q-token/head coordinates in three SIMDgroups.
Each coordinate retains its original token, head, visibility mask, and output coordinate.

The generic shader chunks the shared K/V dimension when a complete staging tile exceeds 32 KiB.
Q and the output numerator retain the full head dimension in registers.
Existing QTokensPerHead geometries retain their original source and memory reservation.

`sliced_q::Compute::new_direct` records an unsplit final-output Map.
Each Q range must have one unique template that covers all its visible KV positions.
The caller must supply matching active Q-range and Map counts.
This path does not allocate partial statistics or record Reduce.
`Selector::select_partitioned` materializes an explicit fixed-slice route, including direct output.
The default lookup selects direct output through 32 visible KV tokens for one query token per request.
It uses a 512-token limit when a request has multiple query tokens.
Explicit registries retain their existing Map/Reduce allocation policy.
Both gated and ungated replay owners can record the explicit direct route.

`GQAMetadataBuffers` retains the selected `backend_sdpa::ExecutionVariant`. The recording path passes this exact value
to the low-level SplitKV `Compute` constructor. It does not reconstruct a variant or copy Map fields into the low-level
model and storage `Config`. Each SplitKV module uses a private `KernelConstants` value to bind that `Config` to the
selected Map and Reduce constants.

The backend chooses the original QTokensPerHead source for its existing geometries.
It chooses the generic sliced-Q source for QHeadsPerToken, QTokensAndHeads, extended QTokensPerHead, and direct output.
Both split sources use the same metadata and natural-log partial-state ABI.

### Default lookup

`GQA::new` and `UngatedGQA::new` pass their existing `GQAPageTableLayout` to `Selector::for_lookup`.
The selector registers reachable geometries at initialization.
The runtime selects one strategy from 24 slots.
It does not compare measured point winners or materialize competing candidates.

| Head dimension / Q heads per KV head | Rounded Q counts `1,2,4,8,16,32` |
| --- | --- |
| `128 / <=8` | single query, small query, shared Q-token/head tiles, shared Q-token/head tiles, shared Q-token/head tiles, shared Q-token/head tiles |
| `128 / >8` | single query, shared Q-token/head tiles, shared Q-token/head tiles, shared Q-token/head tiles, shared Q-token/head tiles, shared Q-token/head tiles |
| `256 / <=8` | single query, shared Q-token/head tiles, shared Q-token/head tiles, QTokensPerHead, QTokensPerHead, QTokensPerHead |
| `256 / >8` | single query, shared Q-token/head tiles, shared Q-token/head tiles, shared Q-token/head tiles, shared Q-token/head tiles, shared Q-token/head tiles |

The Q count is the largest request Q count, rounded up to a power of two and capped at 32.
The QTokensPerHead strategy uses `T8/KV16`.
The shared Q-token/head strategy limits the logical token/head product to `16384 / head_dim` coordinates.
It uses `KV64` for D128 and `KV16` for D256.
It selects QHeadsPerToken when head padding provides more loader groups and QTokensAndHeads would load more than 16 KV tokens per group.
Otherwise, it selects QTokensAndHeads.
The D128 small-query strategy uses QHeadsPerToken through 4096 visible KV tokens and SlicedKv above that length.

The short-context rule uses QTokensAndHeads, `KV16`, at most 32 logical Q-token/head coordinates, and direct output.
Its visible-KV limit is 32 for single-query workloads and 512 for multi-query workloads.
The single-query limit keeps at most two sequential KV16 iterations before the Reduce saving stops paying for lost KV parallelism.
These are measured crossover criteria, not correctness limits.
Single-query D256 groups larger than eight heads use QTokensAndHeads when the proposed split grid has at least 160 SIMDgroups.
Other single-query workloads use SlicedKv above 32 visible KV tokens.
They retain intra-threadblock KV parallelism across the previous 512-token boundary.
Their KV slice is a bounded power of two derived from summed visible KV work and a target of 1280 SIMDgroups.

Other workloads use `end^2 * requests * SIMDgroups_per_Map^2` as the KV-slice feature.
The thresholds `2^26, 2^30, 2^34` select `64,128,256`; larger features select `512`.
These coefficients are empirical operator criteria from the M3 Max matrix.
They are not compiler limits or a claim about other hardware.
They do not contain model names or fixed sequence-length limits.

Initialization derives the maximum visible context from the request page-table extent.
It bounds Map capacity by the larger of token capacity and request capacity times the single-query split envelope.
It reserves partial groups for that Map capacity and the largest reachable token tile.
The runtime can increase the selected KV slice to satisfy this fixed resource contract.
For `n` Q ranges and Map capacity `B`, it requires `slice >= ceil(sum_visible / (B - n + 1))`.
The identity `sum(ceil(visible_i / slice)) <= ceil(sum_visible / slice) + n - 1` proves the bound.
This operation preserves full causal KV coverage and permits arbitrary intermediate contexts.
CPU tests cover policy transitions and lengths through 256K.
GPU measurements above 32K remain deferred.

### Explicit and composite selection

`gqa::sdpa::RequestShape` contains `num_history_tokens` and `num_q_tokens` for one request. For a causal Q-token offset:

```text
num_visible_kv_tokens = num_history_tokens + q_token_offset + 1
```

The selector materializes a Map task range that reaches the maximum visible KV token in the Q-token range. Metadata
also uploads one required half-open `visible_kv_token_ranges` entry per flat Q token. SlicedQ computes the intersection
of the Q token's range and the Map TaskTemplate range. It does not assume `begin = 0` or infer an endpoint from
`flat_token_indices`. A fully masked segment produces `exp_sum = 0` and `max_logit = -infinity`; Reduce gives that
partial zero weight.

Range construction validates that the KV endpoint, including iteration padding, fits `u32`.
Task materialization uses this validated domain for ordinary arithmetic and casts.

The selector creates request-local `gqa::sdpa::QTokenRange` values. A range contains `request_index`,
`flat_q_token_indices`, and `num_visible_kv_tokens`. This token count is the KV extent visible to the last Q token
in the causal range. A `QTokenRange` is dynamic task metadata. It is not a kernel tile or a complete Map task.

The selector calculates the number of KV iterations:

```text
num_kv_iterations
    = ceil(num_visible_kv_tokens / map.thread_block.kv_tokens_per_iteration)
```

`num_kv_iterations` counts selector partition intervals. It does not count context tokens.
For sliced-Q, these intervals match the KV staging width.
For sliced-KV, they do not count the physical 16-token SIMDgroup steps.
`num_kv_splits` counts the KV segments assigned to one Q-token range. Each segment produces one Map task template.
`num_map_task_templates` counts these templates across the complete batch.

The selector starts with one split per Q-token range. A max-heap selects the range with the largest
`ceil(num_kv_iterations / num_kv_splits)`. Earlier Q-token ranges win ties. The selector increments that range's
split count and updates its heap priority. It stops when the batch reaches the Map task-template capacity or every
split contains one KV iteration. A batch that already fills the capacity does not build the heap.
A single Q-token range directly receives `min(num_kv_iterations, max_map_task_templates)` splits.

An explicit fixed-slice route does not use the heap. For each Q range, it generates:

```text
num_kv_splits = ceil(num_visible_kv_tokens / kv_slice)
KV range(s) = [s * kv_slice, min((s + 1) * kv_slice, num_visible_kv_tokens))
```

A fixed KV slice can be smaller than or unaligned to the selected staging width.
Each Map starts at its assigned KV endpoint and masks the final staging tile.
The SlicedKv, original QTokensPerHead, and generic sliced-Q sources support this contract.
Each request retains its own query tails and visible KV extent.
A direct route requires a tiled variant and one complete KV range per Q range.
Its replay uses the same padded Q-range and Map-template extent.
It produces no partial-state groups and records no Reduce command.
The existing `reduce_sdpa_partial_outputs` field records this semantic difference.
`Selection` and metadata derive `direct_output` from that field and the variant. They do not store duplicate mode state.
The replay topology stores `sdpa_direct_output` because the two output paths can have identical recorded capacities and kernel geometry.

The Map task-template ceiling is independent of the token ceiling.
`GQAReplayBucketPolicy` uses the token ceiling for token and Q-range buckets.
It uses the Map ceiling for the KV-split bucket.
A Map/Reduce selection must satisfy `num_total_sdpa_map_task_templates * max_q_tokens <= partial_state_group_capacity`.
The partial-state bound includes replay padding. Active Map counts alone do not establish that bound.
`GQAMetadataBuffers` allocates and validates token and Map domains separately.
`Selector::for_partitions` derives an explicit-route envelope from token and request limits and each variant's split bound.
See [resource initialization](executor_gqa.md#resource-initialization).

The automatic path distributes consecutive KV iterations across Map tasks. A tail task can contain fewer KV iterations. Adjacent
task ranges do not overlap and do not leave gaps.

`gqa::sdpa::MapTaskTemplate` stores only the fields that the regular grid cannot derive:

```text
q_token_range_index
request_local_kv_token_indices
```

The current GPU ABI flattens one template to three `u32` values:

```text
[q_token_range_index, request_local_kv_token_begin, request_local_kv_token_end]
```

The grid derives the Q-head range and KV-head index. The template and grid coordinates together derive one semantic
`MapThreadBlockTask`. The CPU does not upload a fully materialized `MapThreadBlockTask[]` array.

`cu_sdpa_partial_outputs` selects the partial-state groups that Reduce merges for each Q-token range. Reduce metadata
names the partial states. It does not name the Map template that produced them.

### Selection

`gqa::sdpa::Selection` contains one complete dynamic decision:

```text
variant
q_token_ranges
map_task_templates
cu_partial_outputs_by_q_token_range
replay_shape
metrics
```

The selection does not store a duplicate registry index. The complete execution variant identifies the kernel geometry.
The replay topology also identifies direct output. Recording consumes both facts.

`GQAMetadataBuffers::update(...)` uploads the selection. It does not select another variant or recompute KV
partitioning.

The materialized selection is necessary because the selected variant, Q-token ranges, Map task templates,
partial-state offsets, replay shape, and metrics are one coupled result. A kernel-kind enum or threshold helper cannot
represent this result. Moving candidate construction into `GQAMetadataBuffers` would mix selection with GPU ABI upload.
Returning unrelated parallel values would weaken the boundary and permit mismatched variant and task metadata.

`gqa::sdpa::Selector` is not a stateless wrapper. It owns the legal variant registry and the allocation limits.
For explicit and composite registries, it applies the Q-range density and Q-head rules before materialization.
These rules need only request lengths and variant constants. Most profiles then construct one complete candidate.
The D=256 page-16 cost comparison still constructs both SlicedKv and the selected SlicedQ candidate.
Each constructed candidate includes KV splits, metadata, replay capacities, and complete metrics.
This work allocates and fills request-local vectors. Callers must not repeat it only to recover one field.

The selector includes candidate materialization. GQA does not have a separate `Planner` type or a `Plan` type. The
`Selection` name describes the returned value and does not introduce a second decision layer.

### Metrics and selection

`gqa::sdpa::SelectionMetrics` names each counted unit:

```text
num_scheduled_qk_token_pairs
num_active_qk_token_pairs
num_map_threadblocks_per_kv_head
num_map_simdgroup_waves_per_kv_head
num_active_partial_state_groups
num_active_partial_states
num_reserved_partial_state_groups
num_replay_reserved_partial_state_groups
max_kv_iterations_per_split
num_logical_qk_token_pairs
```

Scheduled QK token pairs round each Map's KV extent to its partition interval and use the complete token-tile width.
Active QK token pairs use the same rounded KV count and the actual Q-range width.
For sliced-KV, this interval is coarser than the physical 16-token SIMDgroup steps.
These counts are selection estimates. They do not measure exact shader work.
Logical QK token pairs count only causal query/key coordinates.
A fixed split can add interval padding. The metrics sum the materialized Maps, not one unsplit range.

The original D=256, 16-token-page candidate policy keeps the measured crossover threshold. It rejects a SlicedQ candidate when active QK
token pairs use less than half of its scheduled QK token pairs. It then applies the existing selection score. The score
is a tuning heuristic. It is not a FLOP, byte, token, or elapsed-time unit.

The selector evaluates request-local Q-token ranges. It does not use only aggregate token and context counts. This rule applies to the original candidate policy. The default lookup can use QTokensAndHeads for these requests.

### Current partial-state ABI

The current Map ABI stores natural-log-domain `partial_max_logits`, `partial_exp_sums`, and normalized
`partial_output`.
`partial_exp_sums` is the semantic denominator. The current ABI does not store the numerator.

For partial states `a` and `b`:

```text
max = max(max_a, max_b)
denom = exp(max_a - max) * denom_a
      + exp(max_b - max) * denom_b
numerator = exp(max_a - max) * denom_a * output_a
          + exp(max_b - max) * denom_b * output_b
output = numerator / denom
```

A possible future ABI can store `max`, `denom`, and `numerator`. The current implementation does not use that representation.

### Replay and resource contracts

The current implementation uses these contracts:

- `GQAReplayShape` count domains and exact or bucketed meanings.
- Replay parameter keys.
- Partial-state scratch allocation and layout.
- Page-table layout and KV-cache ownership.
- Map-before-Reduce dependency.
- SlicedKv, SlicedQ, bidirectional block SDPA, and their reducers share one natural-log partial-state ABI.

Replay padding can change the recorded Map grid. Therefore, the selector includes the selected replay capacity when it
materializes and compares candidates.

## Design boundary

`gqa::sdpa::Selector` owns dynamic selection and complete-candidate construction.
One `backend_sdpa::ExecutionVariant` identifies the selected Map and Reduce pair.
Q-token ranges, Map task templates, and kernel constant fields remain distinct concepts.

The selector consumes the existing partial-state ABI, replay key, resource contract, KV-segment allocation policy, and
Metal numerical kernels.
See the [selector source](../crates/inference-executor-metal/src/attn/gqa/sdpa.rs) for the implemented policy.

## Validation

The implementation must keep registry capability tests, selector crossover tests, ragged-tail tests, metadata ABI
tests, SlicedKv and SlicedQ Metal parity tests, the Qwen3.8 D=256 page-16 parity test, and representative one-layer
production performance checks.
