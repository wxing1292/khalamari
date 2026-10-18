# Executor Hibernation

Executor hibernation releases model resources while the service process and runtime cache identity remain live.
The next batch restores model residency before execution.
This document describes the current idle detection, ordered executor protocol, resource operations, and verification.

[`model_state_io.md`](model_state_io.md) defines the implemented full and selected snapshot paths and the planned
request-mobility design.

## Lifecycle requirements

The service must release model weights and backend state after an idle period.
The service process and its RPC listeners must remain active.
The first new request must load the model before model execution resumes.

The implementation must preserve runtime cache identity across a successful stop and start.
It must discard runtime cache metadata after any snapshot or restore failure.

## Lifecycle flow

The table shows a successful cycle with no intervening request release.
Runtime identity stays live while model resources leave backend memory:

```text
+-----------------------+------------------+------------------+--------------------+
| Resource              | Started          | Stopped          | Started again      |
+-----------------------+------------------+------------------+--------------------+
| Runtime cache and IDs | Live             | Retained         | Same identities    |
| Model weights         | Loaded           | Unloaded         | Checkpoint reload  |
| Mutable backend state | Resident         | Snapshot on disk | Snapshot restore   |
| Replay programs       | Recorded as used | Cleared          | Recorded as used   |
+-----------------------+------------------+------------------+--------------------+
```

If work becomes runnable after runtime core queues `Stop`, one request channel preserves this order:

```text
Batch A -> Stop(plan) -> Start(plan) -> Batch B
```

The stop boundary must preserve any completed request state that runtime core still owns.
The model must not serialize an in-flight GPU job.
Runtime core and the model executor must agree on a completed batch boundary before `Stop` runs.
Runtime core can retain live requests across this boundary.

The event loop uses this operation order:

```text
Stop:
  stop registered encoder executors
  drain request-slot resets
  clear_replay_cache
  unload_state(snapshot path, plan)
  unload_weights

Start or Batch while stopped:
  load_weights
  load_state(snapshot path, plan)
  start registered encoder executors
  remove snapshot
  drain deferred request-slot resets
```

Each event-loop instance uses a unique process-local path in the system temporary directory.
The event loop removes the snapshot after a successful start and when the event loop exits.

## Current implementation boundary

`inference-executor-core` owns [`ReplayableDecoderModel`](../crates/inference-executor-core/src/model/replayable_decoder_model.rs).
The trait defines batch execution and these synchronous resource operations:

```rust
trait ReplayableDecoderModel {
    fn clear_replay_cache(&mut self);
    fn unload_state(
        &mut self,
        snapshot_path: &Path,
        plan: &ExecutorHibernationPlan,
    ) -> Result<(), ModelExecutorError>;
    fn unload_weights(&mut self);
    fn load_weights(&mut self) -> Result<(), ModelExecutorError>;
    fn load_state(
        &mut self,
        snapshot_path: &Path,
        plan: &ExecutorHibernationPlan,
    ) -> Result<(), ModelExecutorError>;
}
```

Qwen3 and Qwen3.5 implement these operations.
GQA, GDN, MTP, DFlash2, MLP, embed, unembed, and sampling owners participate in symmetric resource traversal.
The operations run synchronously on the model executor thread.
The event loop also stops and starts registered encoder executors around decoder state unload and load.

[`ReplayableDecoderModelEventLoop`](../crates/inference-runtime-service/src/executor/replayable_decoder_model_event_loop.rs)
owns the loaded model, its stable `Started` or `Stopped` state, and one state snapshot path.
It handles `Batch`, `Start`, and `Stop` requests synchronously on the executor thread.
`Batch` starts a stopped model before it executes the batch.

The event loop defers request-slot resets while the model is stopped.
It applies all deferred resets after state loading and before it acknowledges `Start` or executes a batch.

The executor hibernation timeout defaults to 300 seconds.
`--executor-hibernation-timeout-secs` accepts a positive integer.
`--executor-hibernation-mode` accepts `all` or `selected`. It defaults to `selected`.
The service does not have a lifecycle status API or status route.

## Current resource order

`clear_replay_cache` removes recorded programs that retain Metal resources.
`unload_state` writes the snapshot required by the hibernation plan before it releases state buffers.
`unload_weights` removes all shared weight owners before it drops the final owner.

Load and unload traversal must remain logically symmetric.
Names and ownership levels must also remain symmetric.

## State snapshot

`RuntimeConfig::executor_hibernation_mode` fixes the Stop/Start plan policy at runtime construction.
The Qwen services default to `ExecutorHibernationMode::Selected`.
Use `--executor-hibernation-mode all` to write every state entry.
The direct model integration tests can use `ExecutorHibernationPlan::All`.

The selected snapshot contains these resources:

- The `PageArena` payload for every allocated runtime page ID.
- Main GQA request rows for every allocated request slot.
- MTP or DFlash2 GQA request rows for every allocated request slot when configured.
- The current GDN recurrent state for every allocated request slot.
- The current GDN convolution state for every allocated request slot.
- The durable GDN request state table.

The page-ID selection includes active requests and reusable trie cache blocks.
The shared runtime page allocator assigns unique IDs across Main and configured Spec KV pages and GDN state pages.
Runtime core scans this allocator bitmap and converts the allocated IDs directly to canonical ranges.
It scans the request-slot allocator bitmap in the same way.

The durable GDN request state table contains these values:

- The current recurrent state slot for each request slot.
- The current convolution state slot for each request slot.
- The current state version for each request slot.
- Free recurrent state slots in recurrent allocator order.
- Free convolution state slots in convolution allocator order.
- Future state versions that must publish to runtime-owned page IDs.
- The page IDs for each future publish.

The snapshot does not store submitted restore jobs, submitted publish jobs, or current batch transactions.
The model must finish or clear this transient work before it writes the snapshot.

The snapshot is one directory with a `manifest` file and one file for each semantic state item.
The [snapshot format](model_state_io.md#snapshot-format-and-publication) defines encoding, resource files, validation, and atomic publication.
The [component interfaces](model_state_io.md#component-state-interfaces) define the symmetric full and selected I/O operations.

## Ownership boundary

Runtime core owns these resources:

- Request lifecycle and request admission.
- Request-slot identity.
- Trie cache metadata.
- Physical page allocation and ownership.
- Runtime page IDs and their logical cache identity.

The model executor owns these resources:

- Model weights.
- `PageArena` payload data.
- GQA request page tables.
- GDN recurrent and convolution state.
- GDN request state versions and future publish mappings.
- Component-local state interpretation.
- Replay programs and retained Metal resources.

Runtime core owns idle detection and executor protocol state.
The service owns the configured executor hibernation timeout.
The model event loop owns the resource operation order.
Runtime core must not parse model-specific state.
The model executor must not allocate or free runtime-owned page IDs.

## Runtime-to-device protocol

The protocol is symmetric:

```rust
pub enum ReplayableModelExecutorRequest {
    Batch(BatchDeviceRequest),
    Start(ExecutorHibernationPlan),
    Stop(ExecutorHibernationPlan),
}

pub enum ReplayableModelExecutorResponse {
    Batch(BatchDeviceResponse),
    Started,
    Stopped,
}
```

The request-response pairs are:

```text
Batch(request) -> Batch(response)
Start(plan)   -> Started
Stop(plan)    -> Stopped
```

`Start` and `Stop` are idempotent when repeated with the same hibernation plan.
Runtime core stores the Stop plan and reuses it for Start.
The [runtime hibernation state](../crates/inference-runtime-core/src/runtime/scheduler/executor_hibernate.rs)
tracks the commanded `Started` or `Stopped(ExecutorHibernationPlan)` state.
Runtime core sends `Start` when a stopped executor has work to flush.
One request channel orders `Batch`, `Start`, and `Stop`.
One response channel preserves the matching response order.
The model event loop processes one request at a time.
Each channel reserves one entry in addition to the compute-slot capacity.
This entry preserves capacity for lifecycle commands alongside submitted batches.

Runtime core sends and receives all protocol variants.
It consumes `Started` and `Stopped` as acknowledgements and then attempts the next flush.
It does not maintain separate transition states for these acknowledgements.
After sending `Start`, runtime core disables the idle timer until it receives `Started`.
Submitting a batch also disables the idle timer.
Runtime core starts a new timeout interval after the last in-flight batch response commits.
The `Started` acknowledgement starts a timeout interval only when no batch remains in flight.
Before it blocks in `select`, runtime core attempts the next flush and then decides whether to start that interval.
A new batch keeps the timer disabled without creating a temporary timer channel.
Other events preserve the existing idle deadline.
Weight loading, state loading, and batch execution do not count as idle time.

```text
event       submit A, B    commit A       commit B -> flush C    commit C -> no work    timeout
in flight        2             1                   1                       0               0
idle timer  disabled      disabled            disabled                  starts        Stop(plan)
```

Future wiring may expose the tracked residency through a status API.

## Failure contract

Snapshot and checkpoint I/O failures are recoverable operation errors.
Lifecycle misuse is an internal invariant violation.

The event-loop wiring fails closed after these errors:

- Snapshot write failure.
- Snapshot read failure.
- Weight load failure.
- Encoder start failure.

The current failure path invokes global shutdown and does not send a success response.
Process shutdown discards the full runtime cache metadata.
The service must not continue with runtime page IDs that refer to invalid executor state.

## Integration verification

The ignored model state I/O integration tests cover this sequence:

```text
decode
  -> clear_replay_cache
  -> unload_state(first snapshot)
  -> unload_weights
  -> load_weights
  -> load_state(first snapshot)
  -> unload_state(second snapshot)
  -> compare snapshot file sets and bytes
  -> load_state(second snapshot)
  -> reset request slot
  -> decode
```

The two semantic snapshot directories must be byte-for-byte equal.
The final decode verifies the reloaded checkpoint weights and the second state restore.
Separate tests run `ExecutorHibernationPlan::All` and `ExecutorHibernationPlan::Selected` for each model configuration.
The selected plan uses request-slot and page-ID ranges from the synthetic decode.
The test matrix contains four cases: Qwen3.6 27B and 35B with MTP, and full and selected state.
Each test requires the matching Main and speculative checkpoint environment variables.

Run the matrix with this command:

```sh
cargo test --release \
  --test model_state_io -- --ignored --nocapture --test-threads=1
```
