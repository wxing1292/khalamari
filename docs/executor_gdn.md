# GDN Executor

This document describes the current GDN forward, state commit, and cache-page lifecycle.
Historical measurements appear after the current implementation sections.

## Source layout

| Owner | Source |
| --- | --- |
| Backend-neutral geometry and decode prefix range | `crates/inference-executor-core/src/attn/gdn/{core,state}.rs` |
| CPU correctness references | `crates/inference-executor-core/src/attn/gdn/reference.rs` |
| Qwen request packing and sampling metadata | `crates/inference-executor-core/src/model/qwen/v3_5/batch.rs` |
| GDN projections and mixed forward | `crates/inference-executor-metal/src/attn/gdn/backend.rs` |
| Batch metadata and shared layer scratch | `crates/inference-executor-metal/src/attn/gdn/{batch_metadata,scratch}.rs` |
| Materialized state slots and pending cache pages | `crates/inference-executor-metal/src/attn/gdn/request_slots.rs` |
| State/log resources, preparation, commit, restore, and publish | `crates/inference-executor-metal/src/attn/gdn/state_table.rs` |
| Full and selected snapshot I/O | `crates/inference-executor-metal/src/attn/gdn/state_table/file_io.rs` |
| Shared model state lifecycle | `crates/inference-executor-metal/src/model/qwen/v3_x/state/gdn.rs` |
| Per-layer weights and log bindings | `crates/inference-executor-metal/src/model/qwen/v3_x/layer/gdn.rs` |
| Forward geometry, kernel ownership, and buffer contract | `crates/inference-backend-metal/src/components/gdn/compute.rs` |
| Chunkwise, reference recurrent, and replay recording | `crates/inference-backend-metal/src/components/gdn/compute/{chunkwise,recurrent,replay}.rs` |
| All-layer replay commit | `crates/inference-backend-metal/src/components/gdn/state_replay.rs` |
| State-page kernels | `crates/inference-backend-metal/src/components/gdn/state_pages.rs` |

Metal sources live in `crates/inference-backend-metal/src/components/metal/`:

- `gdn_compute.metal`: short convolution, convolution snapshots, and output norm/gate.
- `gdn_compute_chunkwise.metal`: fused sequential TensorOps computation with sixteen-token tiles.
- `gdn_compute_recurrent.metal`: final/candidate recurrent reference paths.
- `gdn_compute_replay.metal`: recurrent outputs and the F32 replay log.
- `gdn_state_replay.metal`: accepted-prefix recurrent and convolution commit.
- `gdn_qkvabz_split.metal`: projection split.
- `gdn_state_page_read.metal` and `gdn_state_page_write.metal`: immutable page I/O.

Source separation does not add per-component replay caches or per-layer submissions.

## Tensor and axis vocabulary

| Axis | Meaning |
| --- | --- |
| `R` | Requests |
| `T` | Flat input tokens |
| `Hqk`, `Dqk` | Q/K heads and head width |
| `Hv`, `Dv` | V/state heads and head width |
| `Cqkv` | `2 * Hqk * Dqk + Hv * Dv` |
| `Kc`, `Ks` | Convolution kernel width and history length, `Ks = Kc - 1` |
| `S` | Physical state slots |
| `L` | GDN layers |

Forward uses these BF16 tensors:

```text
qkv                       [T, Cqkv]
a, b                      [T, Hv]
z, recurrent_output       [T, Hv, Dv]
conv_qkv                  [T, Cqkv]
norm_gated_output         [T, Hv, Dv]
recurrent_states          [L, S, Hv, Dv, Dqk]
conv_states               [L, S, Cqkv, Ks]
```

`GDNCore.output_gate` selects `GDNOutputGate::Silu` or `GDNOutputGate::Sigmoid` during initialization.
The executor lowers this choice into backend `compute::Config.output_gate`.
The generated Metal source fixes the activation for the component lifetime.
Output normalization computes `RMSNorm(recurrent_output) * gate(z)` before the output projection.
The Qwen35 configuration builder selects SiLU explicitly.
The choice does not change recurrence gates, state storage, or replay submission arguments.
Both activations have CPU-reference and full-component replay coverage for Prefill and Decode.
The backend `gdn_attn` benchmark includes `replay128` cases to amortize submission overhead.
These cases repeat the same forward from immutable source-state slots 128 times per submission.
Their reported time covers all 128 invocations.

`conv_qkv` contains the short-convolution output after SiLU. It is not the raw convolution accumulation.
The state arena stores BF16. State arithmetic uses F32.

The transient replay log uses these layouts:

```text
alpha                     F32  [L, max_replay_tokens, Hv]
k                         F32  [L, max_replay_tokens, Hqk, Dqk]
u                         F32  [L, max_replay_tokens, Hv, Dv]
qkv                       BF16 [L, max_replay_tokens, Cqkv]
```

Each layer binds `token_offset = layer_index * max_replay_tokens`.
A log token index is the forward flat token index minus `cu_tokens[num_active_chunkwise_requests]`.
Only replay requests write the log. Chunkwise prefix rows do not occupy log storage.
The record-time log capacity is independent of the full-forward token capacity.
`GDNRequestStateResources` owns the log buffers beside the recurrent and convolution arenas.
Initialization, resource release, and resource reload cover all six state/log buffers together.
The log lives until this forward's commit finishes. It is not a circular buffer or a cache checkpoint.

## Execution paths, kernel constants, and tasks

Qwen stably orders complete requests before it packs tokens, sampler configuration, GQA metadata, or GDN metadata:

```text
Prefill -> long non-spec Decode -> remaining Decode
```

The request enum retains its original meaning. A long Decode still produces a sampled response.
Routing uses actual submitted speculative tokens, not the configured speculator type.

| Input | GDN core |
| --- | --- |
| Prefill | Chunkwise |
| Decode with speculative tokens | Replay |
| Decode without speculative tokens, more than 8 input tokens | Chunkwise |
| Decode without speculative tokens, at most 8 input tokens | Replay |

The fixed-input replay limit is 8 tokens. Speculative inputs must have at most 8 fixed tokens plus their actual
speculative tokens. Qwen validates this contract at its input boundary.
The initial short-input threshold is independent of the chunk tile width. It has not been calibrated by new performance measurements.
The scheduler owns input construction. The executor validates request and token capacities at its input boundary.

The production graph is fixed:

```text
QKVABZ projection -> split -> short convolution
                              |             |
                   chunkwise prefix     replay suffix
                              |             |
                              +------v------+
                             output norm/gate
                                     |
                              output projection
```

Both branches retain the full recorded request capacity. Their active counts select work at submission.
An empty branch performs no work. The two cores access disjoint output rows and request-owned state slots.
The recorder declares that disjoint access. Output norm/gate waits for both cores through buffer dependencies.

Short convolution also copies raw QKV for replay requests in the same dispatch.
Its compile-time `SaveReplay` specialization keeps the old standalone recurrent APIs independent of the log.

Chunkwise uses Metal 4 TensorOps on all supported devices, including devices without GPU Neural Accelerators.
`Compute` directly owns the kernels for the fixed graph. It has no runtime registry or selector.
`Compute::new` validates model geometry and selects the V-row and Q/K state-update tiles once.
The shader source renders these constants. Recording validates invocation shapes and buffer ranges.
Each tile contains sixteen tokens and at most 64 V rows per thread block.
The selected V-row count must divide `Dv` and fit the device shared-memory budget.
Wider Q/K heads use smaller V-row tiles with the same TensorOps implementation.
The kernel keeps F32 state in cooperative tensors. Each state-update tensor contains eight V rows and at most 128 Q/K columns.
Wider Q/K heads retain multiple state-update tensors per SIMDgroup.
The backend emits named tensors because MSL does not allow arrays of cooperative tensors.
Manual state, transform, and output access follows the shared
[cooperative tensor access contract](engineering_conventions.md#metal-cooperative-tensor-access).
State projections consume a BF16 operand copy in
threadgroup memory. The state update accumulates into the original F32 tensor.
The compiler controls register allocation.
The kernel ends a tile early at a requested cache-boundary snapshot row.
It writes BF16 state only at requested boundaries and at the request end. A snapshot does not round the live F32 state.

The kernel expresses the Gram matrix, WY transforms, state projections, output, and state update as
`mpp::tensor_ops::matmul2d` operations. Matrix operands use BF16. Accumulation, normalization, gates, and the triangular
inverse use F32 with `relaxed_precision = false`. Two independent eight-row solves and F32 matrix products form the
inverse. SIMDgroups distribute the W columns. The output combines F32 state and local projections in cooperative tensors.
Static threadgroup arrays need `(64 + 2 * R) * Dqk + 32 * R + 2752` bytes before compiler padding.
Here, `R` is the V-row count rounded up to eight rows for SIMDgroup storage.
With a 32 KiB budget and `Dv=128`, `Dqk=128` uses 64 V rows and needs 29,376 bytes.
`Dqk=256` uses 16 V rows and needs 27,840 bytes.
Initialization rejects a configuration when no tile fits.
Pipeline construction also checks the reported storage and thread-count limits.
This kernel uses the existing bindings, dispatch metadata, and parent ICB. It adds no submission or CPU wait.
Tile selection does not change the mixed graph or the ReplaySSM suffix.

The 128-column state-update tile also avoids an observed M3 Max TensorOps failure.
A standalone BF16 `8 x 256` multiply-accumulate probe computes incorrect values in columns 128 through 255.
The tiled implementation passes the CPU-reference checks for `Dqk=256`, including multiple V blocks and requested state snapshots.
These checks do not establish a performance gain on M5.

BF16 matrix operands change numerical results relative to the F32 SIMDgroup implementation.
A synthetic 2,048-token test with correlated K, weak decay, and beta near one reports 0.893% output relative L2 error.
Its maximum output error is 3.315% of the reference peak. This exceeds the experiment's 3% normalized maximum bound.
The final-state relative L2 error is 1.533%, above the experiment's 1.5% bound.
This result does not establish model-level quality parity with vLLM or SGLang.

Replay uses sequential register recurrence for all suffix requests, with or without speculative tokens.
Each SIMDgroup owns two V rows. Two SIMDgroups form one thread block. Dqk is distributed across 32 lanes.
The replay path requires `Dqk % 32 == 0` and `Dv % 4 == 0`, as validated at initialization.
It does not use a chunkwise triangular solve for short verification windows.

## Canonical metadata and host/Metal ABI

`GDNMetadataBuffers` owns capacity-sized GPU metadata:

| Field | Layout | Meaning |
| --- | --- | --- |
| `cu_tokens` | `u32[R + 1]` | Half-open request ranges in flat tokens |
| `src_recurrent_state_slots` | `u32[R]` | Current recurrent slots |
| `src_conv_state_slots` | `u32[R]` | Current convolution slots |
| `flat_recurrent_state_write_slots` | `u32[T]` | Chunkwise recurrent snapshot destinations |
| `flat_conv_state_write_slots` | `u32[T]` | Chunkwise convolution snapshot destinations |

`u32::MAX` means that forward must not write a full state for that row.
All replay-request entries use this sentinel. Inactive capacity must not access request metadata or log rows.
Recurrent and convolution slot IDs are independent. Equal numeric IDs are not an ownership contract.

The active replay parameters are:

```text
gdn.num_active_requests
gdn.num_active_tokens
gdn.num_active_chunkwise_requests
```

Active counts are submission data. Recorded total request/token counts determine dispatch capacity.
The chunkwise count may be zero or equal to the active request count.

A commit job contains six `u32` fields in this exact ABI order:

```text
src_recurrent_state_slot
src_conv_state_slot
dst_recurrent_state_slot
dst_conv_state_slot
replay_token_begin
num_tokens
```

`state_replay::write_jobs` packs that order. Each job materializes one selected version across all GDN layers.
Sources must remain live and distinct from every destination until the submission completes.

## Ownership

Runtime core owns scheduling, request identities, token versions, page allocation/free, and cache lifecycle.
The executor consumes those versions and page IDs. It owns GDN state layout and slot interpretation.
Backend components own reusable computation and dispatch.

Qwen owns one shared GDN backend, scratch allocation, batch metadata allocation, and request-state table.
Each layer owns its weights and binds its own region of the all-layer arenas and replay log.
Shared projection scratch can be reused between layers. Replay logs cannot be reused until commit finishes.

The state table stores current recurrent/conv slots, their common state version, materialized destinations, and pending page mappings.
It uses the same destination versions for recurrent and convolution state.

For `B = ceil(max_tokens_per_request / num_tokens_per_block)`, initialization reserves:

```text
max_publish_jobs_per_req          = B
max_materialized_states_per_req   = B + 1
num_state_slots_per_req           = B + 2
max_replay_tokens_per_req         = 8 + max_spec_tokens
max_replay_tokens                 = min(max_tokens, max_requests * max_replay_tokens_per_req)
```

The full-state bound includes current state, a final state, and crossed boundaries. It does not depend on the number of speculative tokens.
The log bound reserves a replay window per request. Prefill capacity does not increase that window.
Buffer allocations and shader-domain conversions are validated at their owning construction boundaries.

Only replay requests create `GDNStateTxn`, which defines the selectable accepted-prefix range.
Chunkwise requests store their final version directly. Qwen microbatch metadata does not duplicate GDN transactions.

For source version `P`, `F` fixed inputs, and `D` speculative inputs, replay commit may select:

```text
[P + F, P + F + D + 1)
```

The selected version counts processed Main inputs. It does not include a newly sampled token that Main has not processed.
A generic zero-fixed-input range may select the source version and need no materialization job.
Qwen's sampled input contract requires at least one fixed input.

## Replay contract

The model retains the Main replay cache. GDN does not own a separate component replay cache.
`GDNReplayTopology` contains the QKVABZ and output affine topologies. Total capacities and model topology are record-time facts.
The prefill/decode mixture is submission data and does not create a new Main key by itself.

### Execution strategy

Replay forward loads the source BF16 state into F32 registers and processes tokens in order:

```text
k_t       = normalize(k_t)
decayed   = alpha_t * S_(t-1)
u_t       = beta_t * (v_t - decayed * k_t)
S_t       = decayed + u_t * transpose(k_t)
y_t       = S_t * q_t
```

The log stores `alpha_t`, normalized `k_t`, and `u_t` in F32. `u_t` already includes `beta_t`.
Forward writes normal BF16 output, but it does not materialize the replay request's recurrent or convolution state.

After verification, commit reloads the same source state and applies the logged updates for each accepted target.
It writes one BF16 state per target. It does not recompute projection, convolution, gates, normalization, or outputs.
Convolution commit takes the final `Ks` raw inputs from the old history and this forward's QKV log.

All requests use the same commit mechanism. Short no-spec Decode is not a direct-state-write special case.
The commit kernels dispatch across jobs and layers. They do not require a CPU loop that submits each layer separately.
The recurrent commit assigns contiguous Q/K values from one V row to each thread.
At initialization, the backend selects 16, 4, or 1 values per thread. The selected count must divide `Dqk`.
Four-value vectors share each token's alpha and u loads. State updates remain F32; stored state remains BF16.

## State data flow

`GDNRequestStateTable` prepares metadata directly in its request table. It borrows the batch inputs and retains only
the page mappings and commit records needed after prepare. Page registration does not allocate state slots.

Prepare performs these steps:

1. Resolve runtime-provided restore pages and validate the source version.
2. Retain future cache-page mappings, including mappings that an earlier batch supplied.
3. Allocate chunkwise final and cache-boundary destinations.
4. Leave replay full-state destinations unallocated and set their forward maps to `u32::MAX`.

Commit performs these steps:

1. Validate the chosen versions against chunkwise final versions or replay selectable ranges.
2. Allocate replay final and accepted-boundary destinations for the entire batch.
3. Build jobs while every source slot remains live.
4. Promote the chosen destinations and release CPU slot mappings that are no longer needed.
5. Submit cached all-layer recurrent/conv reconstruction and page publication.

Allocation must precede promotion for the entire batch. Otherwise, a later request could reuse a source slot that an earlier GPU job still reads.
Final and boundary targets share one destination when their versions match.
Rejected boundaries are not materialized or published. Their pending page mappings remain available for a later accepted forward.
Pending page mappings stay sorted by state version. Commit drains only the accepted prefix and retains the allocation
for future mappings. It also consumes the batch commit records and reuses their allocation from initialization.

`Qwen3xGDNState` retains the pending commit submission. `finish_commit` waits for reconstruction and publication.
It owns the commit replay cache and a separate Metal stream.
The cache key contains the total reconstruction-job and publication-request counts.
Every commit refreshes job and publication metadata, including on a cache hit.
Restore and publication validate the page-buffer capacity and prepare their metadata before cache lookup.
Batch preparation validates page IDs once against that capacity.
Their prepared active counts supply both the identity-capacity replay key and the submission arguments.
A resource reload clears both restore and commit replay caches before old buffers are released.
Prepare, reset, replay-cache clear, and snapshot preparation wait before they reuse logs, slots, pages, or metadata.
Full and selected snapshot I/O require a completed commit. They serialize durable slot metadata and state arenas, not transient replay logs.
Resource reload recreates the logs. Restore writes the runtime-selected page snapshot into the request's current slots.

Commit kernels and page I/O declare their buffer accesses. The replay dependency analysis orders publish after both state writers.
There is one post-forward submission and one wait at the next state-use boundary, not one wait per layer or per component.
Main must finish before the commit stream reads its logs.
With MTP, the executor submits GDN commit after Main rejection readback and before MTP embedding.
MTP uses separate state and page regions, so it can overlap reconstruction and publication.
Other speculator modes submit GDN commit at the existing model response boundary.

## Profile keys

The full-forward GDN benchmark reports `mixed_replay`. Its subcomponent keys are:

```text
qkvabz
qkvabz-to-qkv-a-b-z
compute_mixed_replay
output
```

The executor state wait trace uses `gdn_commit_wait`. It includes pending reconstruction and publication.
Do not add dynamic values to profile paths.

## GDN kernel family

The production path calls `compute::Compute::invoke_with_replay`.
Standalone final/candidate recurrent APIs remain available for component references and isolated kernel comparisons.
They are not runtime fallbacks for Qwen.

`state_replay::Commit` consumes an accepted prefix and produces full recurrent/conv state.
`state_pages::Read` and `state_pages::Write` handle the independent runtime page representation.
An empty job set records no commit operation. A publish-only batch still submits its page writes.

## Tests and benches

Focused correctness coverage includes:

- CPU parity for recurrence, convolution, output gating, and both quantized projections.
- Mixed chunkwise/replay outputs with independent active request, token, and prefix counts.
- Default kernel selection for padded V rows, multiple V blocks, and wider Q/K heads.
  Tests use the production recorder and cover sixteen-token chunks, ragged tails, boundary snapshots,
  serial/concurrent parity, and selected-prefix commits across layers.
- Per-layer logs with grouped K heads, disjoint state slots, and inactive capacity.
- Reconstructed states for selected prefixes, including zero and complete acceptance.
- Chunkwise final/boundary maps and deferred replay allocation.
- No-spec replay commits, accepted boundary publication, and rejected page canaries.
- Vectorized commit against an F64 recurrence, including zero and long accepted prefixes, scalar and vector storage, and replayed active job counts.
- Restore/reset isolation and full/selected state snapshots.

Run Metal tests serially:

```sh
cargo test -p inference-backend-metal --all-features --lib components::gdn -- --test-threads=1
cargo test -p inference-executor-metal --all-features --lib attn::gdn -- --test-threads=1
```

`qwen35_gdn` measures one GDN forward with real 35B-A3B weights. It excludes accepted-state commit and cache publication.
`--tokens-per-req` sets ragged input lengths. `--prefill-requests` selects the chunkwise prefix.
The prefix writes final snapshots; the suffix writes replay logs. `--subcomponents` uses the same forward contract.

`--compare-recurrent` compares that forward with all-recurrent computation using the same state-write map.
The reference does not write a replay log or full decode states. It is not a comparison with the previous complete model lifecycle.
Its untimed checks compare outputs, source preservation, and requested snapshots.

Use the production service and external caller for model-level acceptance checks. Follow
[`executor_benchmarks.md`](executor_benchmarks.md) and [`service.md`](service.md).
Do not compare component forward timings with complete forward/commit timings.

## BF16 storage conversion evidence

The BF16 conversion reduces the byte size of every GDN global transient tensor and persistent state tensor by exactly
50% relative to F32. F32 register and threadgroup arithmetic does not change this global-memory calculation.

The component quality fixture quantizes each global stage before it runs the unchanged CPU recurrent oracle. It covers
final and candidate recurrent-state materialization. The gate requires maximum absolute error `<= 0.005` and mean
absolute error `<= 0.0005`. The observed maximum absolute error was `0.00390625`. The largest observed mean absolute
error was `0.00038775802`. The full GDN owner also checks maximum and mean errors for the output and complete persistent
state arenas.

The performance baseline was clean commit `192ceaec3b8baa941b90dea3bba943b2c259642d`. The converted result used clean
commit `f9831607a4be3bc03e364447c10468c90c0b9caf`. The converted benchmark writes valid BF16 fixture values into every
BF16 input and state buffer. Both measurements used macOS 27.0 build 26A5425a on an arm64 Apple M3 Max with 48 GB
memory. No `PSI_*` environment variables were set. These are normal wall-clock replay measurements. They do not use
force-sync or profile-summary mode.

The backend Criterion medians were:

| Tokens | Component | F32 baseline | BF16 storage | Delta |
| ---: | --- | ---: | ---: | ---: |
| 1 | recurrent | 297.78 µs | 290.40 µs | -2.48% |
| 16 | recurrent | 323.91 µs | 347.93 µs | +7.42% |

The one-token result improves by 2.48%. The 16-token result regresses by 7.42%. The conversion adds BF16 global-storage
conversion and writeback work, but it halves global-memory traffic and storage. The dominant effect changes with the
workload. Do not classify the conversion as an unconditional latency gain or regression.

A load-width audit found no GQA-style scalar staging defect in GDN state access. State-page read and write copy raw
BF16 payloads as 16-byte `uint4` units. Each recurrent-state fragment round assigns one Dqk value to each of 32 lanes.
The lanes access one contiguous 64-byte BF16 span and promote their owned values to F32 registers. A packed per-lane
load would cross the recurrent reduction's lane ownership and would require a data redistribution. Convolution-state
and output paths also consume their loaded value in the thread that owns its arithmetic. Keep these access patterns
unless a separate profile identifies a bottleneck.

The exact commands were:

```text
cargo bench -p inference-backend-metal --bench gdn_attn -- 'metal/gdn-attn/core-ragged_recurrent/replay/batch1/tokens(1|16)$' --warm-up-time 1 --measurement-time 2 --sample-size 20 --noplot
```

Shared GPU serialization, benchmark metrics, and performance-evidence rules are in
[`executor_benchmarks.md`](executor_benchmarks.md).


## Fixed mixed replay verification: 2026-09-07

This comparison measures one complete GDN layer with real Qwen3.6-35B-A3B weights.
It does not measure a whole-model speedup.

Provenance:

- Measured base: `45c86d920cbd0d74d10c36c9aa22257a720bc5b1`, with dirty implementation and benchmark changes.
  The tested source is preserved by `abd44a31` and `87a51b8c`.
- Machine: Mac15,9, 16 CPU cores, 48 GiB, arm64, macOS 27.0 (`26A5425a`).
- Power: Battery Power, 89%, discharging. Codex was active and the user reported concurrent GPU use.
  Treat these as relative results under that shared workload.
- Model: `/Users/wenquanxing/Workspace/models/Qwen3.6-35B-A3B-4bit`.
- Environment: `RUST_LOG=warn`. No GPU timestamps or force-sync profiling.
- Release benchmark SHA256: `dac338c62a910be36bddd44621f73ddf76703ca7210f20f67ab948741ba6a6c6`.
- Local artifacts: `/private/tmp/psi-gdn-mixed`. This directory contains `provenance.json`, `source.patch`,
  `commands.json`, `results.json`, and the raw logs. Temporary artifacts are local to this machine.

The baseline records the previous full recurrent graph through the current legacy invocation APIs.
It has seven commands and no empty chunk dispatch. It is not a historical executable.
The candidate records the production fixed mixed graph with eight commands.
Both use one replay execution and one completion wait per layer.
Both use the same weights, input buffers, and immutable source-state slots.
The candidate graph saves each Decode candidate row and only the final Prefill row in these fixtures.

Each iteration alternates which graph runs first. Each run has ten warmup pairs and 100 measured pairs.
Five runs produce five paired ratios. The metric is ordinary replay wall time, including submit and wait.
The percentage column is the median of the five candidate/baseline ratios, minus one.
It is not the ratio of the two independently calculated median-time columns.

```sh
RUST_LOG=warn cargo bench -p inference-executor-metal --bench qwen35_gdn -- \
  --model-dir /Users/wenquanxing/Workspace/models/Qwen3.6-35B-A3B-4bit \
  --tokens-per-req 128,128,1,4 --prefill-requests 2 --contexts 32 \
  --candidate-states --compare-recurrent --warmup-iters 10 --iters 100 --runs 5
```

The measurements invoked the saved release executable directly with the same arguments.
The table gives the request lengths and prefill-prefix count for each command variant.
All GPU commands ran serially.

| Tokens per request | Prefill requests | Recurrent median (us) | Mixed median (us) | Paired wall-time change |
| --- | ---: | ---: | ---: | ---: |
| `1` | 0 | 308.554 | 309.882 | -0.17% |
| `1,1,1,1` | 0 | 399.914 | 402.548 | +0.47% |
| `4,4,4,4` | 0 | 597.977 | 598.691 | +1.67% |
| `7,8,9` | 3 | 690.446 | 704.000 | +0.82% |
| `128` | 1 | 1594.107 | 1482.272 | -6.93% |
| `9,7,1,4` | 2 | 697.025 | 712.408 | +1.94% |
| `128,128,1,4` | 2 | 2961.659 | 2781.420 | -6.05% |

Paired candidate/baseline ratios:

| Tokens per request | Ratios by run |
| --- | --- |
| `1` | `0.982180, 0.993354, 1.000208, 1.004304, 0.998305` |
| `1,1,1,1` | `1.002889, 0.976741, 1.006196, 1.021392, 1.004709` |
| `4,4,4,4` | `1.023441, 0.996890, 1.016720, 0.989838, 1.026010` |
| `7,8,9` | `1.028615, 1.003400, 1.008192, 1.007636, 1.020598` |
| `128` | `0.931097, 0.930660, 0.933896, 0.927438, 0.925235` |
| `9,7,1,4` | `1.019437, 1.020138, 1.022431, 1.005030, 1.018201` |
| `128,128,1,4` | `0.939455, 0.935506, 0.935535, 0.939861, 0.941235` |

Verdict: the 128-token Prefill and long mixed fixtures reduce full-layer wall time by about 6–7%.
Decode and short fixtures range from -0.17% to +1.94%. These small differences do not establish a general Decode gain.
Some short fixtures are slightly slower. These measurements do not isolate empty-branch cost from timing variation
or chunkwise work. The result supports this fixed graph as a starting point for the measured workloads.

All seven fixtures pass the untimed comparison. Decode hidden outputs and recurrent states match exactly.
For fixtures with Prefill, the largest hidden-output absolute error is `0.00024414` and the largest recurrent-state
absolute error is `0.00006104`. The largest relative L2 errors are `0.00021685` and `0.00006020`, respectively.
The benchmark also checks convolution results and source-state preservation.
Chunkwise arithmetic reorders F32 operations before BF16 output rounding, so Prefill is not required to match bitwise.

Correctness checks passed:

- 19 Metal backend GDN tests.
- 16 executor GDN owner, metadata, and state-lifecycle tests.
- 129 executor-core tests and 301 runtime-core tests. Two shared-memory tests needed normal host permissions.
- The final mixed backend regression also verifies that final-only execution ignores valid nonfinal state slots.
- Real Main Prefill/Decode case-order, repeatability, and chunk-decomposition checks at two requests and contexts `0,32`.
- `cargo +nightly fmt --all -- --check`, workspace check, workspace Clippy with `-D warnings`, and `git diff --check`.

The real Main lifecycle check used this command:

```sh
RUST_LOG=warn cargo bench -p inference-executor-metal --bench qwen35_vanilla_prefill_decode -- \
  --model-dir /Users/wenquanxing/Workspace/models/Qwen3.6-35B-A3B-4bit \
  --cases prefill,decode --contexts 0,32 --prefill-tokens 32 --decode-tokens 2 \
  --num-reqs 2 --max-tokens 64 --max-tokens-per-request 32 \
  --num-tokens-per-block 2048 --num-cache-pages 16384 \
  --temperature 0 --top-p 0 --warmup-iters 1 --iters 1 --runs 1
```

This Main run is a correctness and lifecycle check. Its absolute times are not a matched performance comparison.
ReplaySSM and sequence-parallel chunkwise execution are not part of this change.

## Concurrent mixed replay verification: 2026-09-07

The GDN core commands now declare disjoint access to request-owned output rows and recurrent state slots.
The recorded graph retains the common-input barrier and the output normalization barrier.
Main still owns one complete replay. Both full-layer comparison graphs contain eight commands.

### Provenance and method

- Source baseline: `d9692e52bfb54574df6d4b4e90a0f108a3c71834`, with no production source changes at task start.
  Unrelated documentation edits were present and remained separate from this change.
- Measured current source: the same commit with the disjoint-access and barrier-encoding changes in the working tree.
- Baseline binary SHA-256: `dac338c62a910be36bddd44621f73ddf76703ca7210f20f67ab948741ba6a6c6`.
  This is the preserved serial mixed binary from the preceding verification.
- Current binary SHA-256: `7f3b7ee3e4694f65c5fa06dc20db56e7fec526fb6322409f5f0ea70b4a05a75a`.
- Model: `/Users/wenquanxing/Workspace/models/Qwen3.6-35B-A3B-4bit`.
- Machine: Mac15,9, arm64, 16 CPU cores, 48 GiB memory, macOS 27.0 build 26A5425a.
- Power: battery, 81% during screening and 80% to 79% during confirmation.
  The user reported concurrent Codex GPU use and requested relative comparisons.
- Environment: `RUST_LOG=warn`. GPU timestamps, force-sync, Metal debug layer, and shader validation were unset.
- Artifacts: `/private/tmp/psi-gdn-concurrent` contains build logs, source snapshots, binary hashes, commands, and raw samples.
  The `relative` and `relative-confirm` directories contain the two measurement sets.

The baseline and current builds use the same chunkwise and recurrent kernels.
The current build also calls `setBarrier()` before encoding the consumer dispatch, as the Apple API directs.
The comparison therefore measures both synchronization changes together.
It does not isolate the effect of the barrier API call order.

The driver launches the two binaries serially in ABBA blocks. A is the serial baseline. B is the current build.
This is a comparison between processes, not an interleaved comparison within one process.
Each process uses the same deterministic inputs, real weights, request lengths, and initial state.
The fixture uses context 32 and the production candidate-state mask: final Prefill row and every Decode row.
Each internal timer includes replay submission and wait. It excludes model load, compilation, recording, and warmup.
No production benchmark switch was added.

For each ABBA block, the driver averages the two baseline process means and the two current process means.
The reported change is the median of the block ratios minus one. Negative values mean less wall time.
The screening run used two ABBA blocks, 10 warmup iterations, and 100 measured iterations per process.
The confirmation run used three ABBA blocks, 50 warmup iterations, and 500 measured iterations per process.
Confirmation focused on the variable Decode controls and the mixed case with an observed compute reduction.

```sh
python3 /private/tmp/psi-gdn-concurrent/run_relative.py \
  --baseline /private/tmp/psi-gdn-concurrent/qwen35_gdn-serial \
  --current /private/tmp/psi-gdn-concurrent/qwen35_gdn-concurrent \
  --blocks 2 --output-dir /private/tmp/psi-gdn-concurrent/relative

python3 /private/tmp/psi-gdn-concurrent/run_relative.py \
  --baseline /private/tmp/psi-gdn-concurrent/qwen35_gdn-serial \
  --current /private/tmp/psi-gdn-concurrent/qwen35_gdn-concurrent \
  --blocks 3 --iters 500 --warmup-iters 50 \
  --cases decode-b1,decode-b4,mixed-balanced --subcomponent-cases mixed-balanced \
  --output-dir /private/tmp/psi-gdn-concurrent/relative-confirm
```

### Relative observations

The screening results measure the complete GDN layer, including both projections.

| Request lengths | Prefill requests | Median wall-time change | Block change range |
| --- | ---: | ---: | ---: |
| `1` | 0 | -8.95% | -13.83% to -4.07% |
| `1,1,1,1` | 0 | +3.58% | +3.47% to +3.68% |
| `9,7,1,4` | 2 | -1.89% | -4.21% to +0.42% |
| `128,128,1,4` | 2 | -0.07% | -0.57% to +0.42% |
| `128,4,4,4` | 1 | -3.64% | -6.16% to -1.12% |
| `128` | 1 | -0.90% | -1.41% to -0.39% |

The longer confirmation produced these results.

| Request lengths | Measured operation | Median wall-time change | Block change range |
| --- | --- | ---: | ---: |
| `1` | Complete GDN layer | -2.70% | -8.18% to +1.17% |
| `1,1,1,1` | Complete GDN layer | -0.94% | -1.19% to +0.40% |
| `128,4,4,4` | Complete GDN layer | -2.86% | -4.02% to -2.59% |
| `128,4,4,4` | GDN compute | -6.05% | -6.16% to -5.38% |
| `128,4,4,4` | QKVABZ projection | +0.32% | +0.13% to +0.49% |
| `128,4,4,4` | QKVABZ split | +0.65% | +0.10% to +1.05% |
| `128,4,4,4` | Output projection | +0.18% | -0.32% to +4.00% |

GDN compute includes short convolution, state materialization, both core branches, and output normalization/gating.
Its result is not a measurement of the fused chunk kernel alone.
The complete `128,4,4,4` layer used baseline block means of 1788.348, 1772.829, and 1759.244 microseconds.
Its current block means were 1716.411, 1722.102, and 1713.677 microseconds.

Verdict: the confirmed mixed fixture reduces full-layer wall time by about 2.9% under these conditions.
The long mixed screening fixture is approximately unchanged.
The Decode control changes vary across runs, including a direction change for four requests.
These observations do not establish a universal speedup or a Decode regression.
The tests establish that the graph permits overlap. No GPU timeline was captured to quantify physical overlap.
No whole-model performance comparison was made.

### Correctness and integration

- The mixed backend test reuses both graphs across dynamic Prefill counts, pure phases, empty branches, and ragged token counts.
  The concurrent graph and the same kernels with serialized Decode produce identical bits in all five output and state buffers.
  The CPU oracle, sparse state destinations, final-only state contract, and inactive canaries remain covered.
- Four dependency tests verify entry and join hazards, partition history, scope isolation, unselected resources, and explicit barriers.
- A GPU fork/join test executes two consecutive groups that reuse storage.
  Eight active-count and split combinations include empty branches, padded capacity, and full capacity.
  All outputs and canaries match the CPU result.
- All 161 backend tests passed, including 19 GDN tests and 24 shared-stream tests.
  All 16 executor GDN tests also passed.
- The real 35B Main lifecycle passed case-order and chunk-consistency checks for two requests at contexts 0 and 32.
  It used 32 Prefill tokens per request and two Decode tokens per request with greedy sampling.
  This run verifies integration only. Its timings are not part of the relative comparison.
- Formatting, workspace check, and workspace Clippy with `-D warnings` passed.
