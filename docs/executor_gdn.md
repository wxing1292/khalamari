# GDN Executor

This document describes the current GDN forward, state commit, and cache-page lifecycle.

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
| Recurrent, reference snapshot, and replay recording | `crates/inference-backend-metal/src/components/gdn/compute/{recurrent,recurrent_snapshot,replay}.rs` |
| All-layer replay commit | `crates/inference-backend-metal/src/components/gdn/state_replay.rs` |
| State-page kernels | `crates/inference-backend-metal/src/components/gdn/state_pages.rs` |

Metal sources live in `crates/inference-backend-metal/src/components/metal/`:

- `gdn_compute.metal`: short convolution, convolution snapshots, and output norm/gate.
- `gdn_compute_recurrent.metal`: tiled F32 register recurrence and requested state snapshots.
- `gdn_compute_recurrent_snapshot.metal`: standalone final/candidate reference paths.
- `gdn_compute_replay.metal`: recurrent outputs and the F32 replay log.
- `gdn_state_replay.metal`: accepted-prefix recurrent and convolution commit.
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

Production projections store one packed `[T, QKV + A + B + Z]` buffer.
Short convolution, recurrence gates, and output gating read column offsets with the packed row stride.
They do not copy projection outputs into separate tensors.
`ProjectionLayout` fixes these offsets during initialization. Separate bindings remain available to standalone component callers.

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
A log token index is the forward flat token index minus `cu_tokens[num_active_recurrent_requests]`.
Only replay requests write the log. Recurrent prefix rows do not occupy log storage.
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
| Prefill | Recurrent |
| Decode with speculative tokens | Replay |
| Decode without speculative tokens, more than 8 input tokens | Recurrent |
| Decode without speculative tokens, at most 8 input tokens | Replay |

The fixed-input replay limit is 8 tokens. Speculative inputs must have at most 8 fixed tokens plus their actual
speculative tokens. Qwen validates this contract at its input boundary.
The short-input routing threshold is independent of the token staging width.
The scheduler owns input construction. The executor validates request and token capacities at its input boundary.

The production graph is fixed:

```text
QKVABZ projection -> strided views -> short convolution
                              |             |
                   recurrent prefix     replay suffix
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

Recurrent and replay use F32 register recurrence. Neither path uses TensorOps or a triangular chunkwise solve.
`Compute` owns both kernels. Initialization selects their static geometry and validates the pipeline and SMEM limits.
Recording validates invocation shapes and buffer ranges. Cached submissions supply active counts.

A recurrent threadblock owns one request, one V head, and one V-row slice of its state.
For `Dqk=128` and `Dv=128`, it owns 32 V rows and uses eight lanes per row.
Each lane retains 16 F32 state values. Four threadblocks cover the full V head.
Different V rows have independent state updates. They need no cross-threadblock synchronization.

The block stages up to 16 tokens of normalized Q/K, gates, and V in SMEM.
It processes those tokens in order while the state remains in REGISTER.
One token reads the previous state, computes U, updates the state, and computes its output.
The block then reuses the staging memory for the next token tile.
It writes BF16 state at requested cache boundaries and at the request end.
A snapshot does not round the live F32 state.

Initialization prefers 32 V rows, then 16, 8, or 4 rows when divisibility or SMEM capacity requires a smaller tile.
It uses eight-token staging for Q/K heads wider than 128.
The recurrent and replay kernels keep their geometry choices at the backend owner.
They add no per-token submission or CPU synchronization.

Replay uses the same recurrence with its own source and output contract.
It prefers 32 V rows and eight lanes per row. It stages up to 16 tokens, or eight for wider Q/K heads.
Initialization reduces the replay token tile when the SMEM budget requires it.
Replay writes F32 alpha, normalized K, and U to GMEM for accepted-prefix reconstruction.
It does not write full per-token candidate states.
The path requires `Dqk % 32 == 0` and `Dv % 4 == 0`, as validated at initialization.

## Canonical metadata and host/Metal ABI

`GDNMetadataBuffers` owns capacity-sized GPU metadata:

| Field | Layout | Meaning |
| --- | --- | --- |
| `cu_tokens` | `u32[R + 1]` | Half-open request ranges in flat tokens |
| `src_recurrent_state_slots` | `u32[R]` | Current recurrent slots |
| `src_conv_state_slots` | `u32[R]` | Current convolution slots |
| `flat_recurrent_state_write_slots` | `u32[T]` | Recurrent state snapshot destinations |
| `flat_conv_state_write_slots` | `u32[T]` | Recurrent convolution snapshot destinations |

`u32::MAX` means that forward must not write a full state for that row.
All replay-request entries use this sentinel. Inactive capacity must not access request metadata or log rows.
Recurrent and convolution slot IDs are independent. Equal numeric IDs are not an ownership contract.

The active replay parameters are:

```text
gdn.num_active_requests
gdn.num_active_tokens
gdn.num_active_recurrent_requests
```

Active counts are submission data. Recorded total request/token counts determine dispatch capacity.
The recurrent count may be zero or equal to the active request count.

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
Recurrent requests store their final version directly. Qwen microbatch metadata does not duplicate GDN transactions.

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

All replay requests use the same commit mechanism. Short no-spec Decode is not a direct-state-write special case.
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
3. Allocate recurrent final and cache-boundary destinations.
4. Leave replay full-state destinations unallocated and set their forward maps to `u32::MAX`.

Commit performs these steps:

1. Validate the chosen versions against recurrent final versions or replay selectable ranges.
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
- Mixed recurrent/replay outputs with independent active request, token, and prefix counts.
- Default kernel selection for padded V rows, multiple V blocks, and wider Q/K heads.
  Tests use the production recorder and cover sixteen-token chunks, ragged tails, boundary snapshots,
  serial/concurrent parity, and selected-prefix commits across layers.
- Per-layer logs with grouped K heads, disjoint state slots, and inactive capacity.
- Reconstructed states for selected prefixes, including zero and complete acceptance.
- Recurrent final/boundary maps and deferred replay allocation.
- No-spec replay commits, accepted boundary publication, and rejected page canaries.
- Vectorized commit against an F64 recurrence, including zero and long accepted prefixes, scalar and vector storage, and replayed active job counts.
- Restore/reset isolation and full/selected state snapshots.

Run Metal tests serially:

```sh
cargo test -p inference-backend-metal --all-features --lib components::gdn -- --test-threads=1
cargo test -p inference-executor-metal --all-features --lib attn::gdn -- --test-threads=1
```

`qwen35_gdn` measures one GDN forward with real 35B-A3B weights. It excludes accepted-state commit and cache publication.
`--tokens-per-req` sets ragged input lengths. `--prefill-requests` selects the recurrent prefix.
The prefix writes final snapshots; the suffix writes replay logs. `--subcomponents` uses the same forward contract.

`--compare-recurrent` compares that forward with all-recurrent computation using the same state-write map.
The reference does not write a replay log or full decode states. It is not a comparison with the previous complete model lifecycle.
Its untimed checks compare outputs, source preservation, and requested snapshots.

Use the production service and external caller for model-level acceptance checks. Follow
[`executor_benchmarks.md`](executor_benchmarks.md) and [`service.md`](service.md).
Do not compare component forward timings with complete forward/commit timings.
