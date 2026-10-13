# GQA Executor

This document describes the current GQA implementation. It covers tensor geometry, batch metadata, Metal replay,
KV-page interpretation, and correctness tests.

Use these sections for common tasks:

| Task | Section |
| --- | --- |
| Locate implementation files and resource owners | [Source layout](#source-layout) and [Ownership](#ownership) |
| Understand kernel selection and replay capacity | [Backend constants and variants](#backend-constants-and-variants) and [Replay contract](#replay-contract) |
| Trace tensors and page bindings | [Data flow and bindings](#data-flow-and-bindings) |
| Select correctness checks and benchmarks | [Tests and benches](#tests-and-benches) |
| Check KV storage and numerical bounds | [FP8 KV-cache storage and validation](#fp8-kv-cache-storage-and-validation) |

## Source layout

```text
crates/inference-executor-core/src/attn/
  mod.rs                    MLX-free attention module exports
  gqa/
    mod.rs                  GQA module root
    core.rs                 gated QGKV GQACore metadata and projection shapes
    ungated_core.rs         ungated QKV UngatedGQACore metadata and projection shapes
    bidi_block_gqa_core.rs  backend-neutral BiDiBlockGQA geometry and per-query history ranges
    reference.rs            CPU projected-GQA correctness oracle

crates/inference-executor-metal/src/attn/
  mod.rs                    Metal attention module exports
  gqa/
    mod.rs                  GQA Metal module root
    sdpa.rs                 executor-owned candidate construction and execution selection
    sdpa_test.rs            host selection, partition coverage, and replay-envelope contracts
    batch_metadata.rs       capacity-sized GPU upload target for one SDPA selection
    backend.rs              gated QGKV Metal replay wiring
    backend_full_test.rs    gated GQA owner replay and CPU parity
    scratch.rs              gated QGKV scratch allocation and borrowed replay bindings
    ungated_backend.rs      ungated QKV Metal replay wiring
    ungated_scratch.rs      ungated QKV scratch allocation and borrowed replay bindings
    request_page_table.rs   per-request, per-layer KV page table for runtime-supplied page IDs
    request_page_table/
      file_io.rs            symmetric full and selected state file I/O
  bidi_block_gqa/
    mod.rs                  paged-history plus bidirectional local-block GQA module root
    backend.rs              ungated paged-history plus bidirectional local-block replay graph
    capacity.rs             Metal BiDiBlockGQA partial-output capacity
    kv_cache_write.rs       Main-feature projection and persistent history K/V cache write
    metadata.rs             per-query history ranges and block metadata
    sdpa.rs                 fixed-proposal history execution selection
    scratch.rs              fixed-capacity local Q/K/V and attention partials
    state.rs                page-table and proposal metadata lifecycle
    state/
      file_io.rs            full and selected state file I/O

crates/inference-executor-metal/src/model/qwen/
  v3_x/
    dflash2/                Qwen3x DFlash2 attention, layer, and model
    dspark/                 Qwen3x DSpark attention, layer, and model
    layer/bidi_block_gqa_weights.rs
                            independent Q/K/V/output affine layouts and borrowed offsets
    layer/gqa.rs            Qwen3xGQA, private checkpoint weights, load, and record
    state/gqa.rs            Qwen3xGQAState page/metadata/reset lifecycle grouping
    state/gqa/file_io.rs    Qwen3xGQAState full and selected state file I/O
  v3/
    main/gqa.rs             Qwen3 Main ungated GQA weights, state, load, and record
    main/gqa/file_io.rs     Qwen3 Main GQA full and selected state file I/O
    main/layer.rs           fixed QKV Qwen3MainLayer composition
    main/component_config.rs
                            Qwen3 Main QKV GQA geometry/config builder
  v3_5/
    main/layer.rs           Qwen3.5 Main QGKV-GQA/GDN layer variants
    mtp/layer.rs            Qwen3.5 MTP GQA layer composition
    component_config.rs     Qwen3.5 QGKV GQA geometry/config builder

crates/inference-backend-metal/src/components/
  gqa/
    mod.rs                  backend GQA component module root
    activation_gate.rs      attention-output gate component
    bidi_block_sdpa.rs      dense bidirectional local-block SDPA partial component
    bidi_block_sdpa_test.rs bidirectional local-block SDPA Metal parity contracts
    kv_page_write.rs        KV page-write component
    qgkv_split.rs           gated QGKV split component
    qkv_split.rs            ungated QKV split component
    sdpa.rs                 backend-owned SDPA variant registry, constants, and capability checks
    sdpa/
      sliced_kv.rs           SplitKV SlicedKv Map/Reduce component
      sliced_kv_test.rs      SplitKV SlicedKv Metal parity and replay contracts
      sliced_q.rs            Q-token/head Map/Reduce and unsplit direct-output component
      sliced_q_test.rs       SplitKV SlicedQ Metal parity and padded replay contracts
  rms_norm_rope.rs          reusable Metal token/head RMSNorm/RoPE component
  metal/
    gqa_qgkv_split.metal       gated QGKV split source
    gqa_qkv_split.metal        ungated QKV split source
    rms_norm_rope.metal         Metal token/head RMSNorm/RoPE source
    gqa_kv_page_write.metal     Metal KV page-write source
    gqa_fp8.metal                      shared packed FP8-to-BF16 decode
    gqa_sdpa_sliced_kv_map.metal     Metal SplitKV SlicedKv map source
    gqa_sdpa_sliced_kv_reduce.metal  Metal SplitKV SlicedKv reduce source
    gqa_sdpa_sliced_q_per_head_map.metal      Steel SplitKV SlicedQ map source
    gqa_sdpa_sliced_q_map.metal   generic Q-token/head map source
    gqa_sdpa_sliced_q_reduce.metal   Metal SplitKV SlicedQ reduce source
    gqa_bidi_block_sdpa.metal         Metal bidirectional local-block SDPA partial-output source
    gqa_activation_gate.metal    Metal attention-output gate source
```

`crates/inference-executor-core` owns the backend-neutral GQA semantic metadata and replay shape.
`crates/inference-executor-metal` owns the Metal replay wiring and request page table.

The Metal GQA executor backend implements the executor `ReplayLayer` contract. Qwen model and layer code can use this
contract to append GQA work to a larger replay. A semantic layer input and output connect the work to a caller-owned
`Recorder`.

`request_page_table.rs` owns the executor request-slot KV page table. This table accumulates page IDs between reset
notifications. The runtime core owns physical page allocation and release. The model executor converts runtime cache
lanes into model-role and GQA-layer coordinates before it writes the table.

The gated, ungated, and BiDiBlockGQA backends give each quantized projection to one adaptive affine operator.
The caller provides the fixed projection dimensions, quantization layout, and dtype when it creates the operator.
It provides the recorded token capacity and the submission-time active token count.
The affine operator selects QMV or a QMM tile.
The shared [affine selection policy](executor_dense_mlp.md#backend-selection) also owns sliced-K MMA token and output-feature tiling.
GQA code does not store separate QMV/QMM kernels or a projection threshold.
This contract applies to fused Main QGKV/QKV projections and output projections.
BiDiBlockGQA gives Q, K, V, and output independent affine layouts.
Its checkpoint owner may store Q/K/V in one physical buffer and supply explicit byte offsets.
It may also supply independent buffers.
This layout supports mixed projection bit widths without a transpose or aggregation copy.
The Metal backend registry derives legal SDPA execution variants from the static attention and KV-cache shape.
The Metal executor selector chooses a complete execution from the prepared batch shape.
Model plans do not specify Metal SDPA tiles or threadblock sizes.
An exact backend microbenchmark may construct a low-level SDPA component with explicit geometry.

SplitKV is the local KV-range map/reduce computation on one Metal device. It partitions each visible KV range into KV
segments. It maps the segments and reduces the partial outputs on the same device. This document does not call this
computation ContextParallel. ContextParallel is reserved for KV or context sharding across devices or ranks.

`SlicedKv` and `SlicedQ` describe the current concrete kernel families and Q tiling. These names are orthogonal to
SplitKV. The selector identifies the selected execution by its complete variant.
`MapKernelKind` identifies the algorithm and Q-token/head assignment independently of token capacity.

## Ownership

Each derived `GQACore` carries its source model coordinate plus the common GQA dimensions:

```text
model_layer_index
hidden_dim
head_dim
num_q_heads / num_kv_heads
attention scale
```

`GQACore` defines the gated contract. Its fused projection is always QGKV. Its replay always applies the trained
attention-output gate.

`UngatedGQACore` defines the separate ungated contract. Its fused projection is always QKV. It has no gate resources
or gate recording step.

Both contracts derive query, key, and value widths from the head counts and `head_dim`. Neither contract uses a layout
enum or gate flag.

GQA tensor and tile comments use one symbolic convention:

```text
Q: [Tq,  Hq,  D]       Q tile: [Tq_tile, Hq_tile, D]
K: [Tkv, Hkv, D]       K tile: [Tkv_tile, D]  // one fixed KV head
V: [Tkv, Hkv, D]       V tile: [Tkv_tile, D]
O: [Tq,  Hq,  D]

Q tile [Tq_tile, Hq_tile, D] x K tile^T [D, Tkv_tile]
  -> scores [Tq_tile, Hq_tile, Tkv_tile]
scores [Tq_tile, Hq_tile, Tkv_tile] x V tile [Tkv_tile, D]
  -> output partial [Tq_tile, Hq_tile, D]
```

`T` identifies token dimensions. `H` identifies head dimensions. `D` is `head_dim`.

These equations describe tensor tiles inside one concrete kernel. They do not define `MapThreadBlockTask`,
`gqa::sdpa::QTokenRange`, or `gqa::sdpa::MapTaskTemplate`.

Only `model_layer_index` is per layer. Qwen validates the remaining fields once and uses them through the shared backend.

Qwen3 Main owns `Qwen3MainGQAState`, `UngatedGQA`, and `UngatedGQAScratch`.

Qwen3.5 owns `Qwen3xGQAState`. This state holds one shared gated `GQA` backend and one `GQAScratch` for compatible
invocations. It also holds one `Rc<GQARequestPageTable>` and reusable `GQAMetadataBuffers`.

Each `Qwen3xGQA` layer component retains clones of the backend, scratch, and page-table handles. It also owns its
weights and compact layer coordinate. Qwen3 Main uses its model-owned ungated leaf.

`Qwen35MainLayer` and `Qwen35MTPLayer` use the gated shared leaf in separate role-specific owners.

The gated and ungated executors share the lower-level norm/RoPE, KV update, SplitKV, and output-projection
components. Their projection, scratch, weights, and replay graphs remain concrete and separate.

This structure removes mode checks from the gated QGKV command sequence. It also makes the missing gate structural in
the ungated QKV path.

Qwen3 Main constructs `UngatedGQACore` and `UngatedGQA`.
Qwen3.5 Main and MTP construct `GQACore` and gated `GQA`.
Qwen3x DSpark constructs `BiDiBlockGQACore` and `BiDiBlockGQA`.
Its QKV attention graph is independent from Main and MTP.
The shared BiDiBlockGQA owner also supports per-query half-open history ranges.
Qwen3x DFlash2 constructs its own `Qwen3xDFlash2Attention` owner around the same low-level `BiDiBlockGQA` contract.
It supplies query-position-relative sliding-history ranges and its own layer composition.

Init-time component constants supply the head dimensions, head counts, RoPE constants, and page geometry. A
model-specific runtime branch does not supply these values.

The `rms_norm_rope` component operates on (token, head) coordinates. It does not own query grouping, KV grouping, or attention state.

`RopeScaling` selects unscaled default RoPE or Yarn RoPE.
Initialization resolves the correction range and the complete inverse-frequency table.
Yarn constants blend extrapolated and interpolated inverse frequencies before Metal library compilation.
The RMSNorm/RoPE kernel reads the immutable inverse-frequency table for each rotary dimension.
It applies the configured Yarn attention factor to the rotated Q and K values.

`GQARequestPageTable` stores executor request-slot KV page IDs in a fixed-stride GPU buffer. It retains the IDs between
runtime reset and update notifications.

```text
page_ids[req_slot, gqa_layer_index, block_index, page_id_index] -> runtime KV page ID
```

The runtime still owns physical page allocation and release.

Construction validates the complete flat table capacity. Entry updates and request-slot resets use direct flat-index
arithmetic after this proof. Debug bounds protect private invocation errors.

`GQAMetadataBuffers` stores the GPU arrays shared by every GQA layer in one model replay:

```text
req_slots[num_tokens]
flat_token_indices[num_tokens]
visible_kv_token_ranges[num_tokens][request_local_kv_token_begin, request_local_kv_token_end]
q_token_ranges[num_q_token_tiles][flat_q_token_begin/flat_q_token_end]  // SlicedQ
sdpa_map_task_templates[num_total_sdpa_map_task_templates]
    [q_token_range_index/request_local_kv_token_begin/request_local_kv_token_end]
cu_sdpa_partial_outputs[num_tokens + 1]                                  // SlicedKv
cu_sdpa_partial_outputs[num_q_token_tiles + 1]                           // SlicedQ
```

`gqa::sdpa::Selector` validates cumulative-token monotonicity and request context ranges. It materializes the Q-token
ranges, Map task templates, cumulative partial-output offsets, replay extents, and selection metrics.
`GQAMetadataBuffers::update(...)` uploads that complete selection. The metadata owner does not select an execution or divide
KV work again.

Each three-`u32` `sdpa_map_task_templates` entry materializes one compact `SDPAMapTaskTemplate`:
`{q_token_range_index, kv_token_begin, kv_token_end}`. It assigns one GQA Map threadblock the half-open request-local
persistent-history range `[kv_token_begin, kv_token_end)` for one Q-token range. One Map threadblock can loop over
multiple K/V iterations in that range. The shared replay shape retains the generic TaskTemplate names because the
BiDiBlockGQA composite contract can also reserve a bidirectional local-block partial-output slot.

The grid supplies `kv_head_index` and `q_head_range_index`. These coordinates combine with the template to produce one
logical `MapThreadBlockTask`. One threadblock owns each task in a `1:1` relation. The buffer does not duplicate the
grid-derived coordinates.

The sliced-KV variant uses one-token Q ranges. The sliced-Q variants build request-local
Q-token ranges.

The default causal selector uses the compact geometry and KV-slice lookup described in [SDPA selection](gqa_sdpa_selection.md#default-lookup).
Explicit and composite registries assign additional KV splits to the Q-token range with the largest KV-iteration count per split.
A max-heap preserves this greedy policy and selects the earlier Q-token range on ties. KV splits for one Q-token
range are contiguous. See [SDPA selection](gqa_sdpa_selection.md) for the count units and allocation sequence.

`SelectorLimits` separates token capacity, Map task-template capacity, and partial-state-group capacity.
`Selector::new` accepts all three capacities from its initialization owner.
Token and Q-range metadata use the token capacity. Map metadata uses the Map task-template capacity.
Replay uses separate token and Map bucket ceilings.
The selector verifies that the padded Map extent times the selected Q-range width fits the partial-state capacity.

### Resource initialization

`Selector::for_partitions` initializes resources for an explicit-route policy.
Its caller supplies the token limit `M`, request limit `R`, and each variant's maximum splits per Q range.
The caller must bound splits over the complete configured context domain, including policy transitions.
The caller must honor the request limit when it selects work.
The envelope does not derive split bounds from a measured context grid.
It is independent of the default lookup resource contract.

For positive request lengths and token-tile width `T`, the maximum number of Q ranges is:

```text
r = min(R, M)
max_q_token_ranges(T) = r + floor((M - r) / T)
max_maps(variant) = max_q_token_ranges(T) * max_kv_splits(variant)
map_capacity = max_variants(max_maps(variant))
partial_state_group_capacity = max_variants(map_bucket(max_maps(variant)) * T)
```

`map_bucket` uses the actual shared replay buckets at `map_capacity`.
The calculation keeps each variant's Map bound and token width together.
It must not multiply the largest Map bound by the largest token width from another variant.
`GQA::new_scratch`, `UngatedGQA::new_scratch`, metadata factories, and replay policy use the resulting selector limits.

The Main and MTP constructors use `Selector::for_lookup` for supported D128/D256 cache profiles.
They derive request and context capacities from the existing page-table layout.
The lookup bounds shared Map and partial storage independently of the Q-token/context product.
It increases a KV slice when the proposed partitions exceed that initialized capacity.
Both gated and ungated components use this contract.
Other static profiles retain their original legal variants and allocation contract.
See [default lookup](gqa_sdpa_selection.md#default-lookup) for the capacity proof and calibration limits.

For a fixed Q-token/head output coordinate, adjacent `cu_sdpa_partial_outputs` values select the
`SDPAPartialOutput`s for the reducer.

`num_total_sdpa_map_task_templates` is the recorded replay capacity in the shared shape. The selector uses the shared
capacity policy. Unused tail Map task templates contain an invalid Q-token-range index and do not write a map result.
BiDiBlockGQA uses this padded count as the replay-cache shape. It supplies `num_sdpa_map_task_templates` as a
submission-time replay argument. Thus, different active history lengths can share one recorded capacity.
BiDiBlockGQA token and Q-token-range totals cover the materialized request slots.
Active counts can be smaller than those totals when Spec Decode uses a padded request bucket.
The recording still dispatches the total Q-token-range extent and receives the active Q-token-range count at
submission.

The SplitKV `SlicedKv` map also permits an invalid-Q-token-range `SDPAMapTaskTemplate` in one token's generic composite
range. This template does not write a history partial output for that slot.

A caller may populate the reserved max-logit, exp-sum, and normalized `SDPAPartialOutput` through
`gqa::bidi_block_sdpa::Compute`. It does this before it invokes the unchanged partial-output reducer.

This generic composition supports an attention connection that combines SplitKV history with a dense bidirectional
local block. The backend component does not own model-specific proposal or cache semantics.

Replay recording borrows `&GQAMetadataBuffers` directly. It does not use a duplicate bindings wrapper.

The CPU uses `cu_tokens`, per-request `req_slots`, and per-request starting `token_indices` to build these token-major
arrays. GQA kernels do not consume `cu_tokens`. Therefore, `GQAMetadataBuffers` does not retain a GPU copy.

The model-level GQA storage shape is:

```text
pages[num_cache_pages][page_bytes]

main_page_ids[num_req_slots][num_gqa_layers][num_blocks][num_page_ids_per_block]
optional_mtp_page_ids[num_req_slots][1][num_blocks][num_page_ids_per_block]
optional_bidi_block_gqa_page_ids[num_req_slots][num_spec_layers][num_blocks][num_page_ids_per_block]

one KV page, viewed as FP8 E4M3FN:
  [K/V][num_kv_heads][num_tokens_per_page][head_dim]
```

The Metal config stores `page_bytes` and the BF16 activation dtype. It derives `num_tokens_per_page` from `page_bytes`,
`num_kv_heads`, `head_dim`, and the fixed one-byte FP8 cache element. It does not store that derived value or a
configurable KV-cache dtype.

For a flat token, the page lookup is exactly:

```text
req_slot -> gqa_layer_index -> block_index -> page_id_index -> page_id -> KV page
```

The main page-ID table uses a fixed compact main `gqa_layer_index`.
The optional MTP has an independent table with `K` GQA layer entries.
Runtime cache lane `step_index + 1` supplies the page IDs for MTP table entry `step_index`.
`Qwen35MTP` performs this model-specific lane-to-layer-index conversion through the generic page-table write API.

Qwen3 DSpark and Qwen3x DFlash2 have separate role-local page tables.
Runtime cache lane 0 contains one flat page-ID list for each block: `[Main IDs | Spec IDs]` in a persistent BiDiBlockGQA
mode.
The Qwen executor validates the exact combined length and splits this list once at the model prepare boundary.
Vanilla and MTP modes require an empty Spec remainder.
Both page tables keep their own layer and page-stride geometry.
The executor sends the same runtime request-slot reset notification to both tables before it reuses a slot.
Both tables use `GQARequestPageTable::reset_req_slots`.

`GQARequestPageTable` does not parse runtime cache lanes or model-role composition. It exposes symmetric per-entry
access:

```text
write_page_ids(req_slot, layer_index, block_index, page_ids)
read_page_ids(req_slot, layer_index, block_index)
```

Main and the selected BiDiBlockGQA state owner accept one complete role-local block.
They validate the exact role-local length and the cache-page ID domain.
They then write one table entry for each GQA layer.

The layout type uses generic `num_gqa_layers` for both tables. Each table instance owns its capacity and can use a
different GQA configuration.

The Qwen executor updates Main state once for each Main batch.
It updates optional MTP metadata once for the MTP stage.
It updates DSpark or DFlash2 proposal metadata once for the selected BiDiBlockGQA stage.

Main GQA layers borrow the Main state domain. The optional MTP owns its backend, scratch, and logical-lane page-ID table.

Layer owners retain immutable weights and a `gqa_layer_index`. A main layer does not retain model-level GQA
configuration or batch metadata.
The physical MTP layer binds its logical `gqa_layer_index` as a submission-time replay parameter.
MTP step `step_index` binds GQA layer entry `step_index`.
Page preparation maps runtime cache lane `step_index + 1` to that entry.
The step index does not enter the replay key.

## Backend constants and variants

The GQA benchmark uses these subcomponent names:

```text
qgkv-proj
split
q-norm-rope
k-norm-rope
kv-page-write
sdpa-sliced-kv
sdpa-sliced-q
gate
output-proj
```

GQA owns KV page-table and cache interpretation inside the executor. The runtime core owns physical page allocation and
release. It also provides page IDs.

The replay SplitKV path reads the shared KV page arena through token metadata and the executor GPU page table. It does
not materialize a forward-local dense context window.

The path does not upload per-forward block tables before it launches the selected Metal attention kernels.

`SlicedKv` and `SlicedQ` map/reduce generate Metal source from the exact selected component geometry.
Immutable head, dtype, page, scale, and backend-selected variant values become source constants.

Replay work determines the cached recorded variant. This work includes `num_tokens`, Q-token ranges, the total
KV-split extent, and the selected `max_q_heads` value.

SplitKV partial-output reduce also generates source for stable Q-head and head-dimension geometry. It keeps the active
token count as a replay argument.

The common kernel source-hash cache reuses identical generated pipelines. This compilation does not introduce
model-specific backend types or names.

Recording materializes the replay-specific `gqa::sdpa::sliced_kv::Compute` or
`gqa::sdpa::sliced_q::Compute`. The recorded invocation retains its Metal pipelines. The GQA owner must not add a
second pipeline cache.

SplitKV `SlicedKv` and `SlicedQ` separate model and storage facts from selected kernel constants and dynamic replay work:

```text
sliced_kv::Config or sliced_q::Config    backend_sdpa::ExecutionVariant
  num_q_heads                           map KernelConstants
  num_kv_heads                          reduce KernelConstants
  head_dim
  scale                                sliced_kv::Shape or sliced_q::Shape
  page_bytes                            replay capacities
  page_table_layout
  dtype
```

The selected `backend_sdpa::ExecutionVariant` is the only source for `max_q_tokens`, `max_q_heads`,
`kv_tokens_per_iteration`, and Map and Reduce `required_threads`. `GQAMetadataBuffers` retains this exact value.
Recording passes it to `sliced_kv::Compute::new(...)`, `sliced_q::Compute::new(...)`, or `sliced_q::Compute::new_direct(...)`.
Recording does not copy variant fields into `Config`.

Both SplitKV compute owners validate immutable configuration and shape at construction.
Map and Reduce recording validate the newly supplied buffer ranges and replay arguments.

Each low-level SplitKV module binds `Config` and the selected variant in one private `KernelConstants` value. Source
generation, scratch validation, and Map and Reduce dispatch use that value. The Map and Reduce thread requirements
remain distinct fields even when a current variant gives them the same value.

`num_total_sdpa_map_task_templates` is the shared shape field for the padded KV-split extent. It is not the number of
KV iterations.

One KV split can cover several consecutive KV iterations. The replay bucket policy rounds up `num_kv_splits` to
produce the total replay dispatch and scratch extent. The replay field retains its established name.

The backend configuration is model-independent.
`model/qwen/v3/main/component_config.rs` builds the Qwen3 Main ungated core.
Each `model/qwen/v3_x/dspark/attention.rs` layer derives its ungated core and Metal configuration from the normalized
DSpark config and its exact attention binding subtree.
Each `model/qwen/v3_x/dflash2/attention.rs` layer derives its independent core and Metal configuration from the
normalized DFlash2 config and its exact attention binding subtree.
`model/qwen/v3_5/component_config.rs` builds gated Main and MTP cores.

Each DSpark or DFlash2 layer owns its weight-dependent `BiDiBlockGQA` and `BiDiBlockGQAKVCacheWriter`.
`BiDiBlockGQAState` owns the shared page table, metadata, scratch, and frozen SplitKV history selection.
Quantization layout is not part of the shared-state compatibility contract.
The state receives static attention and KV-cache facts through `backend_sdpa::Config`.
`backend_sdpa::Registry::new(...)` provides the legal history variants. `bidi_block_gqa::sdpa::Selector` derives the maximum
Q-token-range extent, scratch extent, replay extent, and launch-cost metrics for each candidate. Its `Selection`
contains the execution variant and `BiDiBlockGQACapacity`. The selector first minimizes how many times one history K/V
token must be loaded for the fixed proposal. It then compares the kernel KV-iteration width, padded Q-token/head coordinates, scratch
extent, and Q-head coverage. The state freezes this result at initialization. The runtime metadata builder then
materializes the current history and block-partial composition.

Each concrete backend converts its core and `GQAMetalConfig` into a projection split. `GQAMetalConfig` contains only
model, quantization, storage, and RoPE facts. The backend derives Metal SDPA tuning from `head_dim`. It then constructs
the shared norm/RoPE, KV-update, SplitKV, and output-projection components.

The Qwen GQA weight owner loads one bounded `TensorMap` from its exact GQA binding subtree.
It retains the core and `GQAMetalConfig` values that created the backend.
Weight reload uses these retained values.
It removes Q/K/V, norm, and output tensors from that map.
It then materializes the fused QGKV or QKV buffers required by the selected backend ABI.
The map must be empty after construction.

Only gated `GQA` constructs the activation-gate component. Backend source and APIs contain no Qwen model names or Qwen
configuration types.

`gqa::qgkv_split`, `gqa::qkv_split`, and `gqa::activation_gate` each own private `KernelConstants`. Their
`thread_block.required_threads` value is 256. The Q/K/V head geometry is also a compile-time source constant for the
projection-split and activation-gate kernels. Runtime token counts remain invocation data.

## Replay contract

`GQA` records one GQA layer forward through `ReplayLayer::record(...)` and a caller-owned `Recorder`. It does not
submit commands. It does not own request scheduling or page allocation.

The semantic replay input is:

```text
GQAInput
  page_table_layout GQAPageTableLayout
  gqa_layer_index   compact coordinate into the bound page table
  batch_metadata     &GQAMetadataBuffers
  hidden_state      &Buffer
  next_hidden_state &Buffer
  kv_cache          GQAKVCacheBindings
  weights           GQAWeights
  scratch           GQAScratchBindings
```

`GQAOutput<'a>` is the named alias for the returned `&'a Buffer`. It is the caller-owned `next_hidden_state` buffer.
The alias does not allocate or add a wrapper.

Focused tests and benches use the same `ReplayLayer::record(...)` entrypoint as model replay. The data-flow section
defines the stage order and buffer dependencies.

KV page write reads BF16 projected K/V scratch and writes FP8 E4M3FN cache elements. SplitKV reads FP8 cache elements
and converts them to BF16 before F32 dot-product and online-softmax computation. The component derives its page stride
and `num_tokens_per_page` with the fixed one-byte cache element.

The Metal component uses only `gqa_kv_page_write_fp8_e4m3fn`. `gqa::kv_page_write::Config` owns stable
`num_kv_heads`, `head_dim`, `page_bytes`, the BF16 source dtype, and derived tokens-per-page. It does not select a
cache-format kernel.

The cache conversion uses fixed unit scale and IEEE-style round-to-nearest, ties-to-even. It saturates finite overflow
and infinity to `±448`. It preserves the sign of zero. It encodes NaN with an E4M3FN NaN code. The cache does not own a
scale buffer, scale metadata, or a scale lifecycle. `SlicedKv` and `SlicedQ` decode
eight adjacent cache bytes directly to packed BF16 bit patterns for threadgroup K/V storage. Matrix operations use
the configured query dtype, BF16 K/V, and F32 accumulation.

This format is the only production GQA cache contract. The implementation does not keep a BF16 compatibility cache,
an alternate writer, a runtime fallback, or a feature flag.

The private `gqa::kv_page_write::KernelConstants` contains this compile-time config and thread-block constants that require
256 threads. One thread owns one flattened `(token write, KV head, head-dimension)` value. The dispatch grid and active
token-write count remain invocation data.

`num_token_writes`, `gqa_layer_index`, and page-table coordinates remain invocation data.
The Metal replay core provides symmetric fixed-or-parameter scalar sources for `u32`, `u64`, `i32`, `i64`, and `f32`.
`ReplayArguments`, `CommandParameterLayoutBuilder`, and `CommandRecorder` support the same scalar set. GQA uses
`ReplayU32::Fixed(value)` or `ReplayU32::Parameter(key)` through the same kernel and invocation path. The GQA component
does not define a replay-indexed constructor, kernel variant, or model-specific flag.

The replay shape separates fixed page-table layout from active work and recorded capacity. It retains the established
Q-token-tile identifiers for replay ABI stability. These identifiers represent Q-token ranges:

```text
num_tokens                         active flat Q tokens in the microbatch
num_total_tokens                   recorded flat-Q-token capacity
num_q_token_tiles                  active request-local Q-token ranges
num_total_q_token_tiles            recorded Q-token-range capacity
num_sdpa_map_task_templates        active SDPA map TaskTemplates
num_total_sdpa_map_task_templates  recorded SDPA map TaskTemplate capacity
reduce_sdpa_partial_outputs        whether the active batch selection semantically requires partial reduction
```

For `SlicedKv`, `num_q_token_tiles` equals `num_tokens`. The variant does not consume the active-Q-token-range replay
parameter.

`SlicedKv` replay always records the reduce command. It records this command even when each token has only one map
KV split.

This rule lets both SplitKV variants share one recorded program for the same Q-token-range and KV-split geometry. The
flag remains batch-selection metadata and does not enter the replay key.

`gqa::sdpa::Selector::select(...)` accepts request shapes, one capacity policy, and `num_total_tokens`. It selects the
execution variant and derives the complete active and recorded SDPA extents. `GQAMetadataBuffers::update(...)` stores the
selected execution and replay shape from this selection. GQA does not apply a second token-capacity policy.

The caller-owned token capacity must satisfy `num_tokens <= num_total_tokens <= max_tokens`. It must also preserve the
QGKV and output affine topologies selected for `num_tokens`. GQA validates these topologies during preparation and
recording. The recording check prevents a direct metadata update from bypassing the topology contract.

The GQA token policy includes topology boundaries from both affine projections. The Q-token-range and KV-split
policies use the shared default buckets. `GQA::replay_token_topology_boundaries()` exposes the union to composite
stage policies. `backend_sdpa::ExecutionVariant` remains an explicit topology identity in the replay key.

Parameterized GQA execution consumes these submission values:

```text
gqa.num_active_tokens                    qgkv/output affine, KV write, and attention token guards
gqa.num_active_q_token_tiles             SlicedQ map and reduce guards
gqa.num_active_kv_splits                  SplitKV map guard
```

Projection split, norm/RoPE, KV write, activation gate, and the qgkv/output affine kernels return before an inactive
token reads input or metadata, mutates a page, or writes output. SplitKV SlicedKv and SlicedQ also return before inactive
KV splits or Q-token ranges read their metadata. All token-domain commands use the same active-token parameter key
and range.
The ungated QKV split follows the same active/total contract and uses the caller-owned active-token key.

`GQAInput::num_active_tokens` accepts `ReplayU32::Fixed(value)` or `ReplayU32::Parameter(key)`. A fixed value records
active work without replay padding. A parameter key records a fixed total capacity and supplies the active count at
submission. The supplied key must differ from the private Q-token-range and KV-split keys.

`add_gqa_replay_arguments(...)` supplies all default GQA arguments. A composite stage supplies its active-token
argument once. It then calls `add_gqa_private_replay_arguments(...)` for the Q-token-range and KV-split arguments.

`GQAInput` borrows the metadata object instead of carrying a duplicate shape. Backend recording and replay-key
construction both read the stored shape. Therefore, a batch selection cannot use a different dispatch shape.

The fixed page-table layout is separate init-time state:

```text
num_req_slots               request-slot dimension of the bound page table
num_gqa_layers              GQA-layer dimension of the bound page-ID table
num_blocks                  block dimension of the bound page table
num_page_ids_per_block      physical page IDs assigned to one cache block
```

Qwen3.5 service replay uses a 2048-token logical cache block. Physical KV pages remain 32 KiB.

The model's tokens-per-physical-page and GQA-layer count determine the page-ID count. The 27B model uses 2,048 GQA
pages per logical block (16 layers × 2048/16). The 35B-A3B model uses 640 pages (10 × 2048/32).

The runtime trie and GDN state table use this same logical boundary.

Qwen3 has no GDN snapshot boundary. Its service uses a 16-token logical cache block. Qwen3-14B stores 16 tokens in each
32 KiB physical page.

One logical block therefore owns one page per GQA layer. It owns 40 pages across all 40 layers.

For Qwen3.5 model replay, the Qwen executor validates runtime cache lane 0 and writes the Main page table. DSpark or
DFlash2 mode also writes its independent role-local page table. `Qwen35MTP` separately maps runtime cache lane
`step_index + 1` to MTP
GQA layer entry `step_index`. `GQA::prepare(...)` selects the SDPA execution and builds the complete batch selection once.
Every GQA layer reuses this selection.

`SlicedKv` replay always records partial-output reduction. This rule also applies when each batch token has one
KV split.

The shared `Qwen35GQAReplayKey` contains the three recorded capacities. It also contains the complete
`backend_sdpa::ExecutionVariant`, `sdpa_direct_output`, qgkv affine topology, and output affine topology.
Active counts remain submission values and do not enter this GQA subkey.

`Qwen35MainReplayKey` and `Qwen35MTPReplayKey` use this shared GQA subkey. Qwen3.5 Main selects one composite token
capacity and forces GQA metadata to use it. All Main token commands use the caller-owned Main active-token key.
The Q-token-range and KV-split counts remain private GQA replay dimensions. The Main key also contains the
non-optional GDN request-count subkey. Qwen3.5 MTP independently selects one composite body token capacity and forces
its GQA metadata to use it. All MTP body token commands use the caller-owned MTP active-token key. MTP does not
declare the component-local GQA active-token parameter.

MTP keeps its separate GQA and MLP composite key.
All MTP steps in one batch have the same token and attention shape, so they reuse one recorded program.
The replay argument selects the logical MTP GQA layer at execution time.
Main recording supplies each physical layer's fixed index through the same kernel ABI.

Qwen3 uses its separate ungated GQA implementation.
DSpark and DFlash2 keep separate model-level replay keys and submission ABIs.
Both use the BiDiBlockGQA replay shape with materialized token and Q-token-range capacities and shared TaskTemplate padding.
The active token, Q-token-range, and TaskTemplate counts remain submission values.
The generic composite map includes one block partial-output slot.
All SplitKV and bidirectional local-block SDPA partial producers store `partial_max_logits` as natural logits.
SlicedKv, SlicedQ, and bidirectional local-block SDPA use the natural exponential for online softmax and partial reduction.
This rule keeps one log domain across every producer and consumer of the shared partial-state ABI.

### Execution strategy

See [GQA SDPA selection](gqa_sdpa_selection.md) for the reference execution, constants, task, and selector hierarchy.

`backend_sdpa::Config` contains static workload facts: I/O dtype, Q-head count, KV-head count, head dimension, and
tokens per KV page. `backend_sdpa::Registry::new(...)` derives legal `backend_sdpa::ExecutionVariant` values. Each value
contains one compatible Map and Reduce constant set. `supports(...)` checks only static capability and correctness
conditions.

`gqa::sdpa::Selector` owns the dynamic workload and replay capacity.
The default `Selector::for_lookup(...)` uses the [compact lookup criteria](gqa_sdpa_selection.md#default-lookup) for supported BF16 geometry.
It selects one execution variant, KV slice, and output mode. It constructs metadata only for that selection.
Other inputs and `Selector::new(...)` retain the original Q-range density rules and candidate cost comparison.
Each original candidate uses greedy KV-segment allocation and complete metrics.
The selected `gqa::sdpa::Selection` contains the variant,
the materialized Q-token ranges and Map task templates, cumulative partial-output offsets, replay shape, and metrics.

The selector owns both execution selection and metadata construction. GQA does not have a separate planner layer.
It returns a rich selection because the variant, request-local work partition, and replay extents form one coupled result.
The selection is the atomic boundary between dynamic choice and metadata upload.

`GQAMetadataBuffers::update(...)` uploads the selection. Recording executes the stored variant and does not
select again. Both current concrete kernel families partition a long visible KV range into independent KV segments.

Each BiDiBlockGQA owner selects one variant from `backend_sdpa::Registry::new(...)` at initialization.
`bidi_block_gqa::sdpa::Selector` compares
the complete fixed-proposal candidates. Its primary cost is the K/V load multiplicity for one history token. It then
compares the kernel KV-iteration width, padded Q-token/head coordinates, reserved partial-state groups, and Q-head coverage. Its
`Selection` freezes the variant and the coupled capacity. The SplitKV history map and bidirectional local-block map produce
that physical partial layout for one shared Reduce.
DSpark and DFlash2 keep model-specific metadata construction.
They do not use the general GQA request selector.

BiDiBlockGQA reserves one local-block partial for each Q range before it assigns history KV splits.
A max-heap assigns each additional history split to the range with the largest remaining work per split.
It preserves the existing tie order: later Q ranges win. A single Q range receives its split count directly.
Metadata construction validates the padded history endpoint before it creates individual task templates.
This check includes the nonzero start of a sliding history range.

For Qwen3x DSpark and Qwen3.5 DFlash2 GPU-prepared Spec Decode, `BiDiBlockGQAState` keeps ownership of the same metadata
buffers.
The complete replay sequence is CPU-recorded before submission. `SpecDecodeInput` rewrites accepted-dependent flat query
positions, visible history ranges, and every active history TaskTemplate before GQA reads them. It cannot change the
later replay key or grid, the Q-range and cumulative-partial structure, or the number of Map TaskTemplates.
CPU preparation uses the maximum accepted prefix to retain the metadata allocator's worst-case recorded split count and
parallelism. Lower acceptance maps excess splits to canonical empty ranges. A smaller seed, including all-reject, can
remain correct if each Q range retains at least one history task and its block partial. However, that seed can
under-partition a later longer history range and change performance.
The transform does not replace the final bidirectional local-block partial at `cu_sdpa_partial_outputs[r + 1] - 1`.
Padded TaskTemplates remain outside the active cumulative extent.
An explicit barrier separates these writes from the Spec Decode consumers.

The fixed quota can contain an empty history range after rejection changes the anchor.
Both selected Map implementations write the canonical empty partial for an empty range.
Reduce therefore combines the empty history contribution with the preserved bidirectional local-block partial.

SlicedQ requires one half-open `visible_kv_token_ranges` entry for each flat Q token. The Map kernel computes the
intersection of this Q token's range and the Map TaskTemplate K/V range. It does not infer a missing lower bound or causal
upper bound. If the intersection is empty, SlicedQ writes the empty partial state `exp_sum = 0` and
`max_logit = -infinity`. Reduce ignores that partial.

Ordinary causal GQA uploads `[0, q_position + 1)` for each Q token. DSpark uploads `[0, anchor)` for every anchor and mask
token in one proposal block. Therefore, every DSpark Q token sees the same complete persistent history.
The shared BiDiBlockGQA metadata also accepts a distinct half-open history range for each Q token.
A tiled task covers the union of its Q tokens' ranges. The SlicedQ kernel applies each Q token's exact range mask.

The same backend SlicedQ component represents DFlash2 query-position-relative sliding history.
For a window of `W` tokens and a separate local block, Q-token position `q` uploads
`[max(0, q + 1 - W), anchor)` for persistent history.
The composite parity test covers different explicit Q-token bounds, a fully masked history segment, and one full local
block.
The DFlash2 owner derives and validates the exact window and Spec-block sizes.
It does not add a DFlash mode flag to `BiDiBlockGQA`.

`BiDiBlockGQA::record` computes proposal-block Q/K/V in `BiDiBlockGQAScratch`.
It reads persistent history K/V only through the accepted anchor.
It does not write proposal-block K/V to the paged cache.
`BiDiBlockGQAKVCacheWriter` is the persistent K/V writer used by Spec Prefill.
Qwen3.5 Spec Prefill writes every Main token, and runtime already owns the physical blocks for those tokens.
The GPU-prepared Decode path therefore does not require scheduler-budget, logical-history, context-window, or trie-page
reservation changes.

The two SDPA implementations assign work within a Map threadblock in different ways:

```text
                                  one Map threadblock
                          +--------------------------------+
sliced_kv                 | same Q token and Q-head tile    |
                          | SG0 -> KV chunk 0, 4, 8, ...    |
                          | SG1 -> KV chunk 1, 5, 9, ...    |
                          | SG2 -> KV chunk 2, 6, 10, ...   |
                          | SG3 -> KV chunk 3, 7, 11, ...   |
                          |        merge SG partial states |
                          +--------------------------------+

sliced_q                  +--------------------------------+
                          | SG0 -> Q-token/head subset 0   |
                          | SG1 -> Q-token/head subset 1   |
                          | SG2 -> Q-token/head subset 2   |
                          | all SGs reuse the same KV tile |
                          | each SG owns distinct outputs |
                          +--------------------------------+
```

Both implementations use GQA head sharing. If `G = Hq / Hkv`, KV head `k` supplies K/V to Q heads
`[k * G, (k + 1) * G)`.
Both can split the visible KV range across multiple threadblocks. That split is separate from the SIMDgroup
assignment inside a threadblock.

A buffered Map task template supplies a Q-token-range index and a half-open request-local KV-token range.
The grid supplies the KV head and Q-head range:

```text
Map task template                                grid coordinates
[q_token_range, kv_begin, kv_end)           +   [kv_head, q_head_range]
                    \                                /
                     +------ one Map threadblock

visible KV range [0, N)
  -> one or more Map tasks
  -> partial output + maximum + exponential sum for each (Q token, Q head)
  -> Reduce -> final output[Q token, Q head, D]
```

`cu_sdpa_partial_outputs` selects consecutive partials for each Q-token range.
Padded replay templates use `q_token_range_index = u32::MAX`. Their threadblocks return without writing.
An unsplit `sliced_q` execution can write final output directly and omit Reduce.

The backend registry owns legal static geometries. The executor selector owns workload selection.
The [selection document](gqa_sdpa_selection.md#default-lookup) defines the current default lookup and the explicit
candidate policy. Token capacity alone does not identify the implementation: `sliced_q` can process one Q token.

The original `Registry::new` candidate set supports up to eight Q heads per KV head for `sliced_q`.
`Selector::for_lookup` also registers the generic Q-token/head assignments and legal head ranges for larger ratios.
Supported BF16/FP8 profiles are `D128/P8,16` and `D256/P8,16,32`.
Generic `sliced_q` variants permit up to 1024 threads, subject to the compiled pipeline limit.
The selected token/head assignment determines the thread count.

Both Map kernels use MLX Steel 8x8 fragments and native SIMDgroup matrix operations.
QK multiplies BF16 or F32 Q by BF16 K and accumulates F32 scores.
PV multiplies F32 softmax weights by BF16 V and accumulates an F32 output numerator.
The Steel fragment layout supplies the lane coordinates for masking, per-Q-token/head reductions, and stores.
There is no TensorOps layout probe, architecture-name check, or alternate Map implementation.

Kernel initialization validates shape, SIMD width, thread limits, and SMEM capacity.
Replay recording validates borrowed buffer extents and parameter bounds once for that recorded graph.
Cached replay submission supplies active counts without repeating static shape checks.
Active head, token, and KV-range masks remain part of attention semantics.

#### Short-context selection

The selector uses visible KV work and the query count per request.
For one Q token per request, direct output is useful only through 32 visible KV tokens.
Above that limit, the normal single-query policy retains KV parallelism.
The multi-query direct-output limit remains 512 visible KV tokens.
The [selection document](gqa_sdpa_selection.md#default-lookup) owns the complete criteria.

The following input shows the corrected single-query routing. It uses the same tensors in both paths.

```text
Input: B=1 request, Q[1 token, 16 heads, D=256]
       paged K/V[385 KV tokens, 2 heads, D=256]
       history=384; each KV head serves 8 Q heads

BEFORE: short-context override -> sliced_q, QTokensAndHeads
  grid=(2 KV heads, 1 Map task, 1), threadgroup=(32 threads, 1, 1)
  TG(head=0):
    Q[t0, heads 0..8, D] -> SG0 registers
    KV head 0: [0,16) -> [16,32) -> ... -> [384,385)
                  all 25 sequential KV iterations belong to SG0
      -> QK / online softmax / PV -> final output[t0, heads 0..8, D]
  TG(head=1) does the same for Q heads 8..16.

AFTER: normal single-query policy -> sliced_kv, KV slice=256
  grid=(2 Map tasks, 2 KV heads, 1), threadgroup=(128 threads, 1, 1)
  TG(task=0, head=0): KV[0,256)
    SG0: KV[0,16), [64,80), [128,144), [192,208)
    SG1: KV[16,32), [80,96), [144,160), [208,224)
    SG2: KV[32,48), [96,112), [160,176), [224,240)
    SG3: KV[48,64), [112,128), [176,192), [240,256)
      each SG reads the SAME Q[t0, heads 0..8, D]
      -> QK / online softmax / PV -> four states for the SAME outputs
      -> merge inside TG -> partial state A[t0, heads 0..8, D]
  TG(task=1, head=0): KV[256,385), same division and final visibility mask
      -> partial state B[t0, heads 0..8, D]
  Reduce: merge A and B -> final output[t0, heads 0..8, D]
  The other two TGs supply Q heads 8..16 through KV head 1.

Changed: sequential KV work -> parallel KV work + softmax-aware merges
Same:    tensors, causal visibility, FP8 cache, BF16 output coordinates
```

The cutoff is an empirical M3 Max crossover. It is not a semantic limit on context length.
One-query and multi-query workloads retain the same registry, scratch ownership, and replay-key contract.

#### `SlicedKv`

`SlicedKv` always uses `max_q_tokens=1`.
One block owns one Q token, one KV head, up to `max_q_heads` Q heads, and one KV split.
The dispatch grid uses X for Map task templates and Y for KV heads and Q-head ranges.
Task capacity belongs to the dispatch shape. It does not specialize the shader.
A new split capacity can reuse the compiled pipeline while it records a new replay.
`kv_tokens_per_iteration` remains a planning input. The kernel processes its assigned range in 16-token tiles.

Each SIMDgroup takes a disjoint sequence of 16-token tiles from the split.
It loads one 8x8 Q fragment from GMEM for each QK dimension step.
It does not retain the complete Q tile across KV iterations.
It retains F32 online-softmax statistics and the output numerator in REGISTER.
The Q-head extent is padded to eight.
Each QK or PV operation consists of 8x8 SIMDgroup matrix operations.

```text
for each assigned 16-token KV tile:
    GMEM Q -> REGISTER Q[8 Q heads, 8 dimensions], one fragment at a time
    paged FP8 K -> SMEM BF16 K[16, 64], one dimension slice at a time
    REGISTER Q x SMEM K^T -> REGISTER F32 scores[8, 16]
    online softmax -> REGISTER F32 E, maximum, denominator
    paged FP8 V -> the same SMEM region, one dimension slice at a time
    REGISTER E x SMEM V -> REGISTER F32 output numerator[8, D]

merge SIMDgroup states -> one partial state per Q head
```

The final dimension fragment can be incomplete. Query loads and K/V staging pad it with zero.
Output stores use the actual head width. Inactive heads do not write partial states.
Queries retain the configured F32 or BF16 dtype.
Probabilities, statistics, and accumulators remain F32.

Each SIMDgroup reuses one 2 KiB SMEM region for K, V, and its final 8x64 F32 output tile.
The block merges local maxima, exponential sums, and numerators before it writes its partial state.
This merge adds no GPU command or CPU synchronization.

A block needs `(num_simdgroups * (8*64 + 8*2) + 8*2) * sizeof(float)` SMEM bytes.
The 128-thread production variant uses 8,512 bytes. The 256-thread variant uses 16,960 bytes.
This SMEM footprint does not grow with head width. The output REGISTER payload grows with head width.
Q loads retain only one fragment at a time to limit the live REGISTER payload.

#### `SlicedQ`

The request-shape selector chooses SlicedKv or SlicedQ from the prepared batch.
Token, Q-token-range, and Map task capacities belong to the dispatch shape and replay bindings.
They do not specialize Map or Reduce shaders. New replay capacities reuse compiled pipelines.
Map and Reduce compile independently. Only Map includes the Steel fragment definitions and FP8 decoder.

Each SIMDgroup owns up to eight `(Q token, Q head)` coordinates per fragment.
`MapKernelKind` defines that assignment:

```text
Input for all three assignments:
  Q[4 Q tokens, 24 Q heads, D=256]
  K/V[256 KV tokens, 4 KV heads, D=256]
  G = 24 / 4 = 6 Q heads per KV head
  history = 252 tokens; causal ends = 253, 254, 255, 256

For KV head 2, local head h=0..5 means global Q head 12+h.
Each entry below is (Q token t, local Q head h).

QTokensPerHead: one fixed Q head per SIMDgroup
  SG0: (t0,h0) (t1,h0) (t2,h0) (t3,h0) [4 masked token positions]
  SG1: (t0,h1) (t1,h1) (t2,h1) (t3,h1) [4 masked token positions]
  ...
  SG5: (t0,h5) (t1,h5) (t2,h5) (t3,h5) [4 masked token positions]
  6 SGs = 192 threads; token capacity is padded to 8 per head.

QHeadsPerToken: one fixed Q token per SIMDgroup
  SG0: (t0,h0..5) [2 masked head positions]
  SG1: (t1,h0..5) [2 masked head positions]
  SG2: (t2,h0..5) [2 masked head positions]
  SG3: (t3,h0..5) [2 masked head positions]
  4 SGs = 128 threads; six Q heads use eight physical head positions.

QTokensAndHeads: consecutive token/head coordinates
  SG0: (t0,h0..5) (t1,h0..1)
  SG1: (t1,h2..5) (t2,h0..3)
  SG2: (t2,h4..5) (t3,h0..5)
  3 SGs = 96 threads; 4 * 6 = 24 useful coordinates.

All three assignments read Q at its original [token, head, D] address.
They do not allocate a packed Q buffer.
```

The following comparison uses the same input and one unsplit `[0, 256)` KV range.
It shows the intra-threadblock work division. Multiple KV ranges add Map tasks and a subsequent Reduce.

```text
sliced_kv: 4 SGs process the SAME output tile

  grid = (4 Q-token tasks, 4 KV heads, 1)
  threadgroup = (128 threads, 1, 1); total = 16 threadblocks

  TG(task=t2, head=2): Q[t2, heads 12..18, D]
    K/V chunks are 16 KV tokens each:
    SG0: KV[  0..16) [ 64..80) [128..144) [192..208)
    SG1: KV[ 16..32) [ 80..96) [144..160) [208..224)
    SG2: KV[ 32..48) [ 96..112)[160..176) [224..240)
    SG3: KV[ 48..64) [112..128)[176..192) [240..256)
      |         each SG: QK -> online softmax -> PV
      v
    four partial states for the SAME [t2, heads 12..18, D]
      -> softmax-aware merge inside TG
      -> GMEM partial[t2, heads 12..18, D] + statistics
      -> Reduce -> output[t2, heads 12..18, D]

sliced_q, QTokensAndHeads: 3 SGs process DISTINCT output coordinates

  grid = (4 KV heads, 1 Q-token-range task, 1)
  threadgroup = (96 threads, 1, 1); total = 4 threadblocks

  TG(head=2, task=t0..t4):
    Q[t0..t4, heads 12..18, D] -> SG registers as shown above
    KV[0..16, head 2, D] -> shared SMEM tile
       +----------------+----------------+----------------+
       | SG0 reads tile | SG1 reads tile | SG2 reads tile |
       | 8 coordinates  | 8 coordinates  | 8 coordinates  |
       +----------------+----------------+----------------+
    repeat for KV[16..32), ..., KV[240..256)
    each coordinate applies its own causal end
       -> QK -> online softmax -> PV, independently in each SG
       -> output[t0..t4, heads 12..18, D], direct unsplit stores
    no merge between SGs: they own different output coordinates
```

Each SIMDgroup loads its Q fragments once before the KV loop.
It retains the complete F32 output numerator in registers.
The threadblock stages K and V into the same shared-memory region in sequence.
All SIMDgroups reuse that region. Barriers separate staging and use.

```text
GMEM Q[token, head, D] -> REGISTER Q fragments

for each assigned KV tile:
    paged FP8 K -> shared SMEM BF16 K
    Q x K^T -> REGISTER F32 scores
    visible-range mask + online softmax -> REGISTER probabilities and statistics
    paged FP8 V -> the same SMEM region
    probabilities x V -> REGISTER F32 output numerator

split execution: normalize -> partial output + statistics -> Reduce
unsplit direct:  normalize -> final output, no Reduce
```

The generic source supports KV iteration widths `8, 16, 32, 64, 128`.
It chunks the head dimension when a complete shared-memory tile would exceed 32 KiB.
The specialized `QTokensPerHead` source retains its original `8, 16, 32` KV widths and memory reservation.
The backend uses the generic source for the other assignments, extended geometries, and direct output.

The shared FP8 decoder converts eight cache bytes to packed BF16 values.
Probabilities, online-softmax statistics, and output accumulators remain F32.
Cached prefixes, per-Q-token visibility, causal masks, and the partial-state ABI retain their contracts.
Map adds no CPU wait or separate submission.

Both implementations merge KV partial states by rescaling them to one global maximum.
`sliced_kv` reduces flat `[Q token, Q head, D]` elements.
`sliced_q` launches one Reduce block per `(Q head, Q-token range)`.
`sliced_kv` retains Reduce for one split to preserve its replay topology.
The direct `sliced_q` path omits Reduce; its split path retains Reduce.

Focused fixed, request-tail, multi-range, and ragged cases compare both variants with the CPU reference.

Qwen3.5 keeps reusable gated-GQA scratch in the directly owned `Qwen3xGQAState`. The `GQA` backend creates this scratch
from its selector limits. Ordinary activation buffers use `max_tokens`.
Partial buffers use `partial_state_group_capacity`. Individual GQA layers do not own this scratch.

The executor owns one Main `GQAScratch`. The optional MTP owns one matching scratch because its GQA configuration can
differ. All logical MTP steps reuse this one MTP scratch owner.

This scratch contains the buffers for QGKV projection, the projected gate, gated output, norm/RoPE, and SplitKV. The fixed
gated graph requires these buffers.

Qwen3 Main owns `UngatedGQAScratch`. The `UngatedGQA` backend creates this scratch from the same selector limits. The
scratch contains the QKV projection, norm/RoPE, and SDPA buffers for the fixed ungated graph. It has no gate buffers.

Both backends create metadata with `new_metadata()` and replay buckets with `replay_bucket_policy()`.
These methods use the same selector limits as `new_scratch()`.
State allocation and reallocation obtain all three resources from that backend owner.
The state refreshes its replay bucket policy during reallocation.

Both scratch types expose matching borrowed replay bindings. The model stream serializes Main and MTP execution.
Therefore, submissions reuse their buffers without per-layer allocation.

Qwen3 DSpark and Qwen3x DFlash2 each own a role-local `BiDiBlockGQAScratch`.
This scratch contains proposal-local Q/K/V, SplitKV history partials, and bidirectional local-block partials.
Define these capacities:

```text
T_capacity = max_requests * block_size
Q_capacity = max_requests * ceil(block_size / selected_max_q_tokens)
P_capacity = next_power_of_two(max(T_capacity, 2 * Q_capacity))
```

`P_capacity` is the Map task-template capacity. It retains enough history task parallelism without multiplying the
legacy SlicedKv capacity by the selected Q-tile width.
It does not depend on `max_position_embeddings`.
`BiDiBlockGQACapacity` owns this Metal resource rule.
The backend-neutral `BiDiBlockCapacity` contains only request and block geometry.

The bound for SDPA partial scratch is `Map task-template capacity * selected max_q_tokens * num_q_heads`.
It is independent of `max_position_embeddings`.

`GQAMetadataBuffers` owns the matching submission metadata. The owner receives its capacity once and updates its data
for each submission. Its buffers are read-only during a recorded GQA layer forward.

The buffer contract is:

```text
hidden_state / next_hidden_state     bf16 model boundary buffers shaped [num_tokens, hidden_dim]
req_slots                            request slot repeated per flat token
flat_token_indices                   request-absolute token index per flat token; used for RoPE and KV-page writes
visible_kv_token_ranges              request-local [begin, end) visible K/V range per flat Q token
q_token_ranges                      request-local flat-Q-token ranges consumed by SplitKV SlicedQ
sdpa_map_task_templates             materialized Q-token-range index and request-local KV-token range for Map tasks
cu_sdpa_partial_outputs             cumulative partial-output counts selected per Q-token range by SplitKV Reduce
page_ids                             fixed-stride [req_slot, gqa_layer_index, block_index, page_id_index]
kv_pages                             shared runtime-provided FP8 E4M3FN KV arena backing
scratch                              caller-owned capacity buffers, used only up to current replay shape
weights                              immutable fused projection, q/k norm, and output projection buffers
```

Q and K norm weights keep the checkpoint BF16 storage type. The norm/RoPE kernel reads them directly. It preserves the
configured activation-type arithmetic and rounding order. RMS reduction and RoPE trigonometry use F32.
Default and Yarn RoPE use the same component and replay boundary.

The recording marks the KV arena as both a write and read resource. KV update writes the current tokens. SplitKV
reads the request-visible pages.

KV update calculates the cache block and physical page position from the absolute token index. It then finds the page
through the same page-ID table that SDPA uses.

The runtime core owns page IDs and their reset and release lifecycle. The executor owns model-layer page-ID
interpretation. It validates shape-local buffer capacities.

The executor rejects a runtime page ID outside the bound global KV page arena.

KV update validates the current-token flat K/V input buffers and token metadata buffers. It also validates the
fixed-stride page table capacity.

The update checks each supplied page ID against the model's global `kv_pages` capacity. Page IDs remain runtime-owned
global cache identifiers. This check enforces the runtime/executor ownership contract at the ingestion boundary.

## Data flow and bindings

Both GQA data flows are a single hidden-state stream plus side effects into the runtime-owned KV arena. The gated graph
is:

```text
hidden_state[num_tokens, hidden_dim]
  -> QGKV projection
  -> split q / gate / k / v
  -> q norm + RoPE, k norm + RoPE
  -> write current k/v tokens to KV pages
  -> SplitKV map reads visible KV pages through the selected block lowering
  -> activation gate
  -> output projection
  -> next_hidden_state[num_tokens, hidden_dim]
```

The separate ungated graph is:

```text
hidden_state[num_tokens, hidden_dim]
  -> QKV projection
  -> split q / k / v
  -> q norm + RoPE, k norm + RoPE
  -> write current k/v tokens to KV pages
  -> SplitKV map reads visible KV pages through the selected block lowering
  -> output projection
  -> next_hidden_state[num_tokens, hidden_dim]
```

The CPU inputs are `cu_tokens`, per-request `req_slots`, and per-request starting `token_indices`.
`GQAMetadataBuffers` expands them once into the token-major arrays in the ownership section.

All GQA layers in the replay borrow that selection. The kernels do not consume `cu_tokens` directly.

The page table is storage layout, not a replay shape:

```text
page_ids[req_slot, gqa_layer_index, block_index, page_id_index] -> runtime KV page ID
```

KV page writes occur before SplitKV. Writes and reads use the same page-table interpretation. For each write token, the KV
update kernel calculates:

```text
block_index      = flat_token_index / (num_tokens_per_page * num_page_ids_per_block)
page_id_index    = (flat_token_index / num_tokens_per_page) % num_page_ids_per_block
page_token_index = flat_token_index % num_tokens_per_page
page_id          = page_ids[req_slot, gqa_layer_index, block_index, page_id_index]
```

The kernel then writes the projected K and V token to the shared page arena. It uses the selected page and token
offset.

Within one page, the logical view is `[K/V, kv_head, page_token_index, head_dim]`. Its exact byte footprint must equal
`page_bytes`. The runtime owns the physical page ID and page lifetime.

SplitKV has map and reduce stages:

```text
map(map_task_template_index, kv_head, q_head_range)
  reads one Q-token range
  derives one MapThreadBlockTask from the template, grid coordinates, and MapKernelConstants
  walks the task's KV-token range in fixed-size KV iterations
  merges iteration partials with online softmax
  resolves each KV token through page IDs + KV page arena
  writes partial_max_logits, partial_exp_sums, and SDPAPartialOutput

reduce(flat_token, q_head)
  uses cu_sdpa_partial_outputs to read that token's partial outputs
  combines stable online-softmax partials
  writes the final attention output token
```

Reduce uses per-partial-output max logits to combine exp sums and weighted outputs. It does not materialize a dense
context window.

`SlicedKv` replay records this reduce even when each token has one KV split. This rule keeps the replay topology
stable.

Qwen3-14B `SlicedKv` uses `kv_tokens_per_iteration=128`, `required_threads=128`, and `max_q_heads=5`.

Qwen3.5 uses `kv_tokens_per_iteration=256`, `required_threads=128`, and `max_q_heads <= 8`.
The planning grain stays at 256 KV tokens. The independent thread count limits the block to four SIMDgroups.

The layout groups Q heads by KV head. Each `MapThreadBlockTask` handles one KV head and one Q-head range.

The common resource dependencies are explicit:

```text
q/k norm+RoPE reads q/k scratch and flat_token_indices, writes normalized q/k scratch
KV update reads k/v scratch + flat token metadata + page IDs, writes KV pages
SplitKV map reads q scratch + KV pages + sdpa_map_task_templates + page IDs, writes SDPA scratch
SplitKV reduce reads SDPA scratch + cu_sdpa_partial_outputs, writes attention output
```

Gated `GQA` prefixes these stages with QGKV projection and a split into q/g/k/v scratch. It adds the activation gate
and an output projection after gated attention.

`UngatedGQA` prefixes the stages with QKV projection and a split into q/k/v scratch. It sends attention output directly
to the output projection.

Record barriers only between stages with real dependencies.

Recommendation: The model/layer boundary does not add an implicit every-command barrier around GQA.

Explicit component barriers and backend-inferred buffer hazards provide the required internal order.

Both gated and ungated replay owners support explicit fixed-slice and direct-output selections.
The automatic Main/MTP selector retains its existing Map/Reduce policy.
Direct output uses one Map template per Q range and equal padded extents for both domains.
Its replay omits partial-output bindings, Reduce, and the private active-KV-split parameter.
The output mode enters the replay topology, even when both paths have equal recorded capacities.
See [explicit route materialization](gqa_sdpa_selection.md#explicit-and-composite-selection) for the partition contract.

## Tests and benches

Focused backend and component tests provide part of the correctness coverage. Qwen wiring and model tests provide the
remaining coverage.

The executor selector tests cover fixed slices across ragged causal requests and independent direct replay padding.
The gated full-replay test compares direct and Map/Reduce output with the CPU reference across changing active counts.
It verifies separate cached replays for both output modes with equal recorded extents.
The ungated SDPA-stage test covers six logical query heads, causal output, and unchanged inactive Q tokens for both modes.

`gqa/sdpa/sliced_kv_test.rs` compares SplitKV SlicedKv with the CPU projected-GQA reference. Its main replay test uses
one isolated cache and the complete active-count sequence for capacity `8`. Focused cases cover a ragged batch and a
logical cache block that spans multiple physical page IDs, BF16 and F32 activations, and head dimensions up to 256.
The cases validate compact KV-split indexing,
online-softmax iteration merging, request slots, page-table lookup, and causal visibility.

The SlicedQ tests exercise the Steel Map with 8-, 16-, and 32-token KV iterations.
The generic Q-token/head test also checks 64- and 128-token staging, head-dimension chunking,
6/8/12 GQA ratios, request tails, sliding masks, empty visibility, and direct final output.
It verifies the padded-head and joint `4 x 6` mappings against the CPU reference.
They use the production recorder and preserve one partial-state ABI.
They cover all five supported `(D, tokens_per_page)` profiles, non-contiguous pages, a cached prefix, multiple KV
heads, Q-head tails, multiple Q fragments, sliding visibility, fully masked Q tokens, and padded replay capacity.

`gqa/sdpa/sliced_q_test.rs` compares the BF16-output SplitKV SlicedQ map and reduce variant, with an FP8 E4M3FN KV
cache, against the same CPU reference. The replay test uses one isolated cache with a total token capacity
of `4` and the active-count sequence `1, 4, 3, 2`. The Q-token-tile and map-task capacities stay fixed.
The paged-range case also verifies that inactive output Q tokens retain their sentinel values.

`gqa/bidi_block_sdpa_test.rs` records one total Q-token-range capacity of `8` and replays
`1, 8, 3, 7, 2, 6, 4, 5` active ranges. It compares each active partial state and output with a CPU softmax reference.
This test protects the BiDiBlockGQA total-dispatch and active-guard contract.

`gqa/qkv_split.rs` records one total token capacity of `8` and uses the same non-monotonic active-count sequence.
It compares active Q, K, and V tokens with the exact CPU split reference.

`gqa/qgkv_split.rs` and `gqa/activation_gate.rs` use the same isolated-cache sequence. They compare the active Q/G/K/V
tokens and gated attention values with exact CPU references. `gqa/kv_page_write.rs` compares the complete affected FP8
page arena with a CPU page-table, page-offset, and E4M3FN conversion reference because page writes update persistent
state.

`attn/gqa/backend_full_test.rs` records the real gated `GQA` owner. It replays the non-monotonic active-token sequence
through the production multi-domain cache keys. It compares the active output and the persistent KV page with an
independent CPU reconstruction of QGKV affine, split, RMSNorm/RoPE, causal GQA, activation gate, and output affine.
The test ignores stateless scratch and output tails.

Metal backend component replay sanity lives in:

```text
cargo bench --bench gqa_split_kv -- --profile-time 1 --noplot

cargo bench --bench gqa_bidi_block_sdpa -- \
  --block-sizes 7 --num-requests 1,4 \
  --max-q-tokens 8 \
  --iters 1 --warmup-iters 0 --runs 1
```

`gqa-sdpa-bench` compares generic SlicedKv, QTokensPerHead, QHeadsPerToken, and QTokensAndHeads candidates.
It uses forced KV slices and tests split and unsplit output through the same backend APIs.
It generates coverage reports and calibration lookup tables. The production selector does not consume these tables.
[`executor_benchmarks.md`](executor_benchmarks.md#generic-gqa-matrix) contains the serial scan commands.

The `gqa_split_kv` backend bench records SplitKV SlicedKv building blocks only in Metal replay/ICB paths. GQA Metal code does not
benchmark or expose direct-submit component or forward wiring.

`gqa_bidi_block_sdpa` records only `gqa::bidi_block_sdpa::Compute`.
It measures the dense bidirectional local-block map contribution used by DSpark.
It does not measure history attention, partial reduction, projections, or a DSpark layer.
Its default `max_q_tokens = 8` uses the production SlicedQ partial-state layout. The option can select a different legal
partial-state width for an exact comparison.

One bidirectional local-block SDPA Task owns one Q token and one Q head. The grid supplies the Q-head index,
Q-token-range index, and
range-local Q-token offset. `q_token_ranges` derives the flat Q-token index. The end of the matching
`cu_sdpa_partial_outputs` range derives the block partial-output slot.
The backend uses one 32-thread SIMDgroup for the Task.
The private `gqa::bidi_block_sdpa::KernelConstants` contains the stable SDPA config and its thread-block constants. The
constants require 32
threads and a 32-thread SIMDgroup. The generated source no longer declares an unused
`num_threads_per_threadblock` constant.
For `head_dim=128`, each thread keeps four F32 Q values and one dot-product accumulator.
The logical Q register payload is 16 bytes per thread.
The SIMDgroup computes each Q/K dot product across the head dimension.
It then computes four output dimensions per thread.

For the seven-token DSpark block, the kernel uses 28 bytes of static shared memory for seven F32 logits.
It does not allocate a shared reduction array.
Kernel construction checks the pipeline SIMD width, the pipeline thread limit, and the device shared-memory limit.
Metal does not expose the compiler register allocation.
The backend therefore uses the source-level live-value count and the production-shape benchmark to validate register
pressure.

Metal backend real full-forward replay bench lives in:

```text
cargo bench --bench qwen3_gqa -- \
  --model-dir <qwen3-model-dir> --tokens-per-req 16 --contexts 0,128,1024 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench --bench qwen3_dspark -- \
  --model-dir <qwen3-model-dir> --dspark-model-dir <dspark-model-dir> \
  --cases dspark --num-requests 1 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench --bench qwen35_gqa -- \
  --model-dir <27b-model-dir> --gqa-model 27b --tokens 1 \
  --contexts 0 --num-reqs 1 --gqa-split-kv-variants sliced_kv \
  --iters 1 --warmup-iters 0 --runs 1
```

These benches use CLI arguments instead of environment variables.

`qwen3_dspark` runs the public Main and Spec executor hooks.
It measures the complete DSpark graph.
Use it for DSpark composition and lifecycle costs.
Do not compare its timing directly with `gqa_bidi_block_sdpa`.

`qwen3_gqa` loads the Qwen3 model config and first-layer ungated weights. It accepts explicit per-request token counts
and context lengths. It reports full-replay and SplitKV-only measurements.
It also reports exact QKV and output-projection measurements for QMV, QMM BM8/BN32, and QMM BM16/BN32.
These forced projection paths are benchmark-only.
Production GQA continues to use `affine_quantized::Matmul` selection from the complete shape and dtype.

Its SplitKV SlicedKv and SlicedQ constant arguments are configurable. `--validate` compares full SlicedKv and
SlicedQ outputs for a workload where production selects the SlicedQ variant.

Use `--sdpa-sliced-kv-kv-tokens-per-iteration`, `--sdpa-sliced-kv-required-threads`, and
`--sdpa-sliced-kv-max-q-heads` to configure SlicedKv. Use `--sdpa-sliced-q-max-q-tokens`,
`--sdpa-sliced-q-kv-tokens-per-iteration`, and `--sdpa-sliced-q-max-q-heads` to configure SlicedQ.

The validation also prints threadgroup memory and logical accumulator counts. These counts do not measure compiler register allocation.

For `qwen35_gqa`, `--gqa-model 27b|35b` selects the real-weight layer profile. Pass the matching model directory with
`--model-dir`.

The bench uses the production 32 KiB physical page size. It derives the KV tokens per page from the selected model
profile and the one-byte FP8 K/V element size. The 27B profile uses 16 tokens per page. The 35B profile uses 32 tokens
per page.

For GQA, `--tokens` is the total current flat-token count. `--num-reqs` is the number of request segments in that
microbatch. `--contexts` is the existing context length for each request before its measured tokens.

The bench distributes tokens as evenly as possible across requests. It builds `req_slots`, `flat_token_indices`,
`visible_kv_token_ranges`, and a fixed-stride request page table from these options.

Recommendation: For a single-request decode-style context sweep, use `--tokens 1 --num-reqs 1` and vary `--contexts`.

For a multi-request decode batch, you may use `--tokens 8 --num-reqs 8`.

For a prefill/suffix sweep, you may use `--tokens 64 --num-reqs 1 --contexts 0,2048,4096`.

Without an explicit context list, the bench uses existing context length zero. `--gqa-tokens-per-req` supplies explicit
ragged per-request token counts.

`--gqa-contexts-per-req` supplies the matching existing context length for each ragged request. It requires
`--gqa-tokens-per-req` and cannot be combined with `--contexts`.

`--max-tokens` fixes the token and Map task-template capacities for both forced SplitKV candidates.
Each candidate derives its partial-state capacity from its registered Q-range width.
The default is 128, which matches the server default. Each case reports Q-token
ranges, Map task templates, active and reserved partial-state groups, replay slots, and KV iterations per Map
task template.

The comparison replay reports `split_kv_variant=sliced_kv` or `split_kv_variant=sliced_q`. Model execution uses the
automatic selector policy described above.

`--gqa-sdpa-sliced-kv-kv-tokens-per-iteration`, `--gqa-sdpa-sliced-kv-required-threads`, and
`--gqa-sdpa-sliced-kv-max-q-heads` override the `SlicedKv` defaults.

`--gqa-sdpa-sliced-q-max-q-tokens`, `--gqa-sdpa-sliced-q-kv-tokens-per-iteration`, and
`--gqa-sdpa-sliced-q-max-q-heads` configure the `SlicedQ` comparison variant.

When the Q-head override is absent, the bench uses the production half/full Q/KV-group rule. The benchmark CLI and
output use the production constant terms `max_q_tokens`, `kv_tokens_per_iteration`, `max_q_heads`, and
`required_threads`.

`--print-limits` prints the device threadblock-memory limit. It also prints the derived `SlicedKv`
threadblock-memory footprint.

The current backend records explicit data-dependency barriers. The replay layer also infers hazards from declared
buffer usage. It does not add a conservative every-command fallback.

This bench loads real Qwen3.6 layer weights. It measures the full replay path: qgkv projection, projection split, q/k
norm+RoPE, KV page write, SplitKV, activation gate, and output projection.

Do not compare component-only SplitKV timings with full-forward numbers.

Subcomponent probes use the same request-slot/page-table capacity contract as full-forward replay. Multi-request
`kv-page-write` probes must pass the true `num_req_slots` through the page-table layout in
`gqa::kv_page_write::Shape`.

Do not hard-code one request slot. That value under-validates the page-table contract, even if the kernel reads the
larger backing buffer.

Production Qwen GQA keeps Q, gate, projected K/V, attention output, and projection scratch in BF16. Its persistent KV
pages use FP8 E4M3FN. The page writer converts projected BF16 K/V into those pages. Gated, ungated, BiDiBlockGQA,
DSpark, and DFlash2 history reads use the same page format and lifecycle.

The Metal executor keeps forward wiring out of `components/`. Gated `GQA` composes QGKV projection, the gate, and the
shared attention building blocks.

`UngatedGQA` composes the fixed QKV graph without gate resources.

`ReplayLayer::record(...)` appends to a caller-owned whole-layer/model replay recorder. Focused tests and benches build
replay programs through the same recorder path.

Each model state binds one shared scratch allocation for compatible layers. Therefore, decode replay reuse does not
multiply scratch by the layer count.

During `unload_state`, each layer first drops its shared GQA backend, scratch, and request-page-table references.
The model state owner then releases the final references and its batch metadata.
State load rebuilds these transient resources before it restores the full `GQARequestPageTable` payload.
Each GQA state owner implements `FullStateIO` with `GQAStateSnapshotFiles`. `read_full_state` requires mutable state and
a unique `Rc<GQARequestPageTable>`. This contract prevents restore from overwriting a page table that is still attached
to an execution graph.

The core `scale` is part of both attention contracts. The executor passes it to SplitKV kernels. Kernels must not
silently substitute `1 / sqrt(head_dim)`.

Q and K norm/RoPE use separately specialized commands because their stable head counts differ.

Use typed indices for typed buffer offsets. Use byte offsets for raw Metal buffer bindings.

Use 64-bit Metal address math when page or stride arithmetic multiplies large strides. Resource usage marks must cover
KV page writes and reads exactly.

Record dynamic threadblock memory in both direct and ICB paths. Replay-cache shape keys must distinguish fixed
page-table stride, execution variant, and replay extents.

Debug in this order:

1. Component primitive
2. Real-weight GQA wrapper
3. Attention slice in a layer
4. Layer ladder

## FP8 KV-cache storage and validation

FP8 E4M3FN uses half the persistent GQA KV-cache bytes of BF16. It stores twice as many tokens per 32 KiB page.
The page-ID table, page ownership, cache lifecycle, and replay topology use the same contracts.
BF16 activation scratch and F32 arithmetic storage retain their own formats.

The host conversion tests cover fixed encodings, saturation, ties-to-even, exhaustive finite-code round trips, and quantization error.
For subnormal inputs, the absolute-error bound is `2^-10`. For normal inputs, the relative-error bound is `0.0625`.
A Metal page-write test covers negative zero, subnormal values, tie values, finite maximum, finite overflow, and infinity.

The deterministic 64-token projected-GQA quality fixture compares FP8-cache output with a BF16-cache oracle.
The fixture requires no saturated values, maximum absolute error `<= 0.025`, and mean absolute error `<= 0.005`.
SlicedKv, SlicedQ, BiDiBlockGQA, and the complete GQA owner also retain their CPU-oracle parity tests.

Shared GPU serialization and benchmark methods are in [`executor_benchmarks.md`](executor_benchmarks.md).
