# Runtime Core Guidance

Runtime core owns scheduling, request lifecycle, and page and cache ownership.

`inference-error` owns the shared `Error`, `Result`, and error logging macros.
Runtime core re-exports these items to preserve its caller API.
The launcher uses the same error contract without depending on runtime core.

Recommendation: Keep runtime core model-agnostic and separate from the model executor.

## Selection

Core may know requests, batches, token/block indices, page IDs, lifecycle state, and allocation ownership.
Recommendation: Keep model-specific layout and computation behind the executor boundary.

`RuntimeConfig::num_tokens_per_cache_block` is model-neutral cache metadata. It gives the token extent of one trie
block and its attached opaque KV and state-page vectors.

The service compiles this extent as const `N` for each production artifact. Production selects the artifact and its
configuration at build and deployment time. Runtime initialization requires
`RuntimeConfig::num_tokens_per_cache_block == N`. One live runtime does not change this extent.

This extent differs from the token capacity of one physical KV page. The executor interprets each vector according to
its model layout. Core only allocates, shares, and reports the vector as one logical block.

## Core-owned responsibilities

```text
request -> slot + lifecycle -> schedule batch -> executor metadata
                    |                 |
                    |                 +-> batch row: token_index / block_index / page IDs
                    +-> trie block: opaque KV/state page vectors
                              -> shared physical page allocator
                              -> reuse, eviction, accepted/rejected state notifications
```

Core owns Prefill/Decode/Spec scheduling, slot reset/drop, page allocation/free, and cache lifecycle.
The executor interprets payloads and component layouts.

## Scheduler and trie cache object model

The scheduler owns request placement, budgets, and compute-slot ordering. A request owns its decoder-block view.
Preparation can reserve or reuse trie-cache blocks and physical pages before preparation produces executor metadata.

A full block is eligible for trie lookup only when at least one request token remains after the block. This rule keeps
a real forward suffix for logits and recurrent state. This design does not need a model-specific full-prefix replay
operation.

Earlier blocks still reuse trie entries. Core commits a completed terminal mutable block normally. Later requests can
then use that block as a non-terminal prefix.

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ Sessions::create                                                           │
│ validate input -> allocate RequestSlot -> create running InternalRequest   │
│ -> submit through the request channel                                      │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │ event-loop enqueue
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ ScheduleQueue                                                              │
│ new_queue | unique run_queue | ID -> request map                           │
│ unordered AwaitReservation futures | async-task request sender             │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ InstrumentedScheduler<SimpleScheduler>                                     │
│ periodical/lifetime API and Spec stats + hard scheduler limits             │
│ free/used compute slots + ordered compute-slot sequence                    │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │ runnable work + free slot
                                    │ -> allocate ordered compute slot
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ FIFOBatcher::prepare                                                       │
│ plan slot-local sticky requests without changing request state             │
│ allocate minimum validated, maximum validated, then speculative budgets    │
│ use remaining request and token budgets for FIFO preparation               │
│ call InternalRequest::prepare with one absolute token budget per request   │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ InternalRequest::prepare                                                   │
│ initialize/reserve decoder blocks -> schedule query tokens                 │
│ -> build DecoderSyncBlocks -> DeviceRequest                                │
└──────────┬───────────────┬────────────────┬──────────────────┬─────────────┘
           │ Continue      │ Skip           │ Await            │ async/term
           v               v                v                  v
┌──────────────────┐  ┌───────────────┐  ┌────────────────┐  ┌───────────────┐
│ BatchDeviceReq   │  │ ID map only   │  │ waiting_reqs   │  │ task pool or  │
│ Prefill: requeue │  │ await response│  │ wait future    │  │ preempt/drop  │
│ Decode: map only │  └───────────────┘  └────────────────┘  └───────────────┘
└────────┬─────────┘
         │ executor submission
         v
┌──────────────────┐     ┌───────────────────────────────────────────────────┐
│ BatchDeviceResp  │────>│ SimpleScheduler/FIFOBatcher::commit               │
│ same slot order  │     │ match response by ID and oldest compute-slot seq  │
└──────────────────┘     └────────────────────────┬──────────────────────────┘
                                                  v
                               ┌──────────────────────────────────────┐
                               │ InternalRequest::commit              │
                               │ commit decoder tokens/cache state    │
                               │ emit visible output + stop handling  │
                               └──────┬──────────────┬───────────────┘
                                      │ Continue     │ TurnCompleted       │ Terminal
                                      v              v                     v
                               ┌─────────────┐ ┌──────────────────┐ ┌──────────────┐
                               │ run_queue   │ │ completed channel│ │ drop request │
                               │ push front  │ │ -> Sessions      │ │ release slot │
                               └─────────────┘ │ -> pending       │ │ and pages    │
                                               │ -> publish event │ └──────────────┘
                                               └────────┬─────────┘
                                                        │ start turn with new tokens
                                                        │ request channel
                                                        v
                                               ┌──────────────────┐
                                               │ event-loop enqueue│
                                               │ same request/slot │
                                               └──────────────────┘
```

`PrepareResult::Skip` deliberately leaves the request in the ID map. It does not put the request on `run_queue`.
Previously submitted work owns the next transition. The executor response returns the request through `commit`.

`PrepareResult::Continue` carries a `ComputePhase`. A Prefill query returns to `run_queue` after preparation so that
another Prefill query can enter the pipeline. A Decode query stays only in the ID map until a response commits or a
cancellation restores it.

`run_queue` stores each request ID at most once. `push_front` moves an existing ID to the front. `push_back` keeps an
existing ID at its current position. Commit and cancellation can therefore restore a request without creating a
second runnable occurrence.

Each compute slot keeps the request ID order selected for its most recent device batch. The next use of that slot treats these
IDs as its sticky working set. Different compute slots can contain the same request ID during pipelined Prefill. One
device batch must contain each request ID at most once.

`BatchDeviceRequest::new` preserves the scheduler's request order. Core does not sort requests for model execution.
The model executor can reorder complete requests during `prepare_batch`, before it constructs component metadata.
This packing order does not change the scheduler's request selection or sticky order.
The same order reaches model commit and response construction. Runtime core continues to resolve responses by request ID.

`SimpleScheduler` resolves retained sticky IDs through the request-ID map. It skips absent IDs and
creates immutable `ReqTokenInventory` values. It then allocates minimum validated, maximum validated, and
speculative token budgets. The speculative phase uses a request-local causal candidate heap. The allocator returns a
request-ID-to-token-budget map. `FIFOBatcher` consumes this map before it uses the remaining hard request and token
budgets for the normal FIFO queue.

An assigned token budget is an upper bound. Block initialization can reuse a cached prefix and leave less work than
the budget. The request selects only the remaining validated tokens and available drafts.

Runtime admission and scheduler batching have separate limits. `RuntimeConfig::max_running_requests` bounds the
request-slot domain, the request channel, and the async-task request and response channels. Every request-channel
entry owns one request slot, so this limit is also a strict channel-capacity bound.

`SchedulerConfig::max_requests`, `max_tokens`, and `max_tokens_per_request` remain per-batch limits.
`max_tokens_per_request` must not exceed `max_tokens`.
`DevReq` and `BatchDevReq` expose `request_cost` and `token_cost` for prepared work.
`SchedulerConfig::max_compute_slots` bounds outstanding batches. It therefore also bounds the batch-request and
batch-response channels.

Runtime initialization requires the per-batch request limit to be no larger than the running-request limit.
Request admission requires at least `max(1, L - 1)` initial input tokens. This rule gives each configured cache lane
its required initial token. The runtime returns `InvalidArgument` before it constructs Trie blocks when the request is
too short.

`RuntimeConfig::context_window` is the effective model-input token limit. Request initialization validates
`history.len() + prompt.len() + sampled.len() < context_window` before it constructs Trie blocks. It also requires the
initial sampled-token count to be less than `max_sampled_tokens`.

Request initialization validates all caller input before it allocates a request slot. It then constructs a running
`InternalRequest` and submits that request to the synchronous event loop. If no slot is available, initialization
returns `Unavailable` before it creates the internal or external request.

The same slot follows the request until request drop. It passes through run queues, device-pending work, the
scheduler-owned reservation-wait collection, and service-owned pending session storage.

`CommitResult::TurnCompleted(reason)` transfers the complete `InternalRequest` and completion reason out of the
scheduler. The event loop sends both values through the completed-turn channel. The service session manager stores
the request before it publishes `RequestEvent::TurnCompleted` to the caller. Ownership location represents the wait
between turns. The request does not need a separate waiting state.

The completed-turn channel transfers request ownership to `Sessions`. The request-event channel sends token
probabilities and turn completion to the caller. These channels have different receivers and ownership contracts.
`InternalRequest` stores only the request-event sender. It does not store turn-completion state.

The `Sessions` task is the only owner of the pending-session map. Resume, single-request eviction, and idle GC
use an unbounded command channel. Each command includes a oneshot response sender. The session manager does not share
the pending-session map through a mutex. If eviction precedes resume in the command order, resume observes
that the resident request is absent and returns `Evicted`.

Turn completion makes all current tokens history. It resets current-turn prompt and sampled-token metadata and clears
unused speculative proposals. The next turn adds prompt tokens to the retained `TrieDecoderBlocks` and makes the new
suffix ready for scheduling. The service returns the same `InternalRequest` through the request channel. Event-loop
admission enqueues this request without allocating a new slot and without changing its `Running` status.

The session manager checks the request status before resume.
External cancellation can make the request terminal while turn metadata changes.
This race must not panic. Scheduler admission and preparation retire terminal requests.

The retained blocks preserve committed KV and state progress. Scheduling starts at the retained cursor. It includes
the prior turn's last sampled token and the new prompt because these tokens have no committed model state yet. It does
not include earlier committed tokens. Initial and resumed turns use the same request channel.

Context exhaustion is terminal. A stop sequence or the per-turn sampled-token limit completes only the current turn
when the context still has capacity.

Request-slot allocator usage is therefore the single admission count. No separate running-request counter exists.

`Sessions::evict_one_req` removes the oldest request that waits between turns. `Sessions::evict_expired`
removes all requests that exceed the specified idle duration. Each API transitions an active request to
`RequestStatus::Evicted` before it drops the retained request. It preserves an existing terminal status. Request drop
releases the request slot and removes the request's cache pins. The trie cache can retain reusable blocks until its
cache policy evicts them. The service returns `Error::Evicted`. A gRPC streaming RPC maps this error to `ABORTED`
because the caller must restart the complete session.

Both APIs send commands to the session manager, which scans its pending-session map. Resource-pressure and periodic-GC
callers own the policy and invocation order.

`PrepareResult::Await` transfers the request into the `ScheduleQueue` reservation-wait collection. The collection is a
`FuturesUnordered<AwaitReservation<...>>`. It does not produce a device request for that request.

`AwaitReservation` is not a general asynchronous task. Its wait future must retain completion until the scheduler
polls it. It must not require an independent timer or I/O wake to make the synchronous event loop poll the collection.

After preparation starts, the scheduler still allocates a compute slot and submits the resulting batch. It does this
even when the batch contains zero device requests.

Empty batches preserve executor participation for backends such as distributed expert parallelism. The empty response
releases the compute slot in normal sequence order.

The request remains `Running` while it waits for the reservation. A reservation wait does not change model-state
residency.

`SimpleScheduler::commit`, `cancel`, and `handle_async_task_resp` update their work.
They then call `ScheduleQueue::handle_ready_waits`.
This method polls `ScheduleQueue::pop_ready_reqs` until it returns `None`.
Each poll returns at most one ready request from the unordered wait collection.

A request can become terminal before or during the wait. The request then stays terminal.
`handle_ready_waits` drops terminal requests and appends other ready requests to the back of `run_queue`.

Shutdown cancels reservation waits when it drops the scheduler and its wait collection. The scheduler event loop
remains a synchronous thread.

This is the complete lifecycle for cache-reservation waits. This lifecycle is not KV and state offload.
Backing storage remains unchanged. Future onload and offload require an additional explicit ownership design.

### Async tasks and input resources

`AsyncTaskPool<T>` executes request-associated work outside the synchronous scheduler event loop.
It is generic over `AsyncTaskReq`.
It does not depend on `UserRequest`.

`PrepareResult::BlockingAsyncTask` retains the internal request only in the request-ID map.
It prevents further request preparation until the response updates the request.
`PrepareResult::NonblockingAsyncTask` keeps the internal request runnable.
Both task types prevent terminal request release until their responses arrive.

Runtime core currently uses this path for input-resource materialization.
A request owns each `Resource` and its stable `ResourcePlacement`.
A symbolic resource contains a typed UUID and an opaque `ResourceURI`.
A concrete resource also owns an arena `OffsetAllocation` and its logical resource-token count.

`ResourcePlacement` maps absolute initial-request token ranges to logical resource-token ranges.
Runtime intersects that mapping with each cache block.
It stores the result as a `ResourceSegment` trie annotation.
The segment contains the resource ID, block-local token index, logical resource index, and segment length.

Cache-block initialization applies this sequence:

```text
trie block hit
  -> do not materialize annotated resources

trie block miss
  -> all referenced resources are concrete
       -> initialize the block and continue

  -> one referenced resource is symbolic
       -> return ResourceNotFound
       -> create ResourceMaterializationReq outside init_block_once
       -> dispatch through the registered ResourceProcessor
       -> apply ResourceMaterializationResp to the retained internal request
       -> initialize the block again
```

`ResourceProcessors` maps each `ResourceTypeID` to one registered `ResourceProcessor`.
Runtime core dispatches by `ResourceTypeID`.
It does not interpret media types, checkpoint metadata, or tower inputs.
The registered processor owns model-specific materialization.
For a local Metal implementation, the processor resolves its source and forwards it to a standalone encoder executor.
For a remote implementation, the processor can call a remote encoder and install the validated result in the local
resource arena.
Runtime core does not distinguish these implementations.

The materialization task does not hold the internal request.
The task carries the request ID and required resource IDs.
The response returns concrete resources with the same identities.
The internal request changes the matching symbolic resources to concrete resources.

Before a blocking resource task starts, runtime releases the reservation for the block that requires materialization.
Earlier unfinished blocks can remain reserved.
A concrete resource must remain alive until all device work that reads its allocation is complete.
After commit caches all placement ranges for one resource, runtime changes that resource back to symbolic.
The transition can occur while the last cache block is still mutable.
The allocation returns its byte range through RAII release.

The current runtime has one async response type.
`InternalRequest` therefore downcasts `AsyncTaskResp` to `ResourceMaterializationResp`.
[`future_work.md`](future_work.md) records the required typed dispatch work before a second response type is added.

The event loop can submit after these request-work events:

- It receives a user request.
- It applies an async-task response.
- It commits a device response.

Device-response commits and async-task responses also discover completed reservation waits.
The event loop then prepares the next batch.

The event loop submits immediately when the scheduler has runnable token work and a free compute slot.

Request and token budgets remain hard per-batch limits in `FIFOBatcher::prepare`. They are not mutable aggregation
thresholds. The current path has no scheduler flush timer.

The scheduler stats timer fires every 30 seconds. It does not flush scheduler work. `InstrumentedScheduler` prints and
resets non-empty periodical stats. It retains separate lifetime stats and always prints them during shutdown.

Both scopes contain scheduler API counts/latencies, batch-execution latency, and speculative-token counts.
`InstrumentedScheduler` starts a batch-execution timer after `Scheduler::prepare` returns. It stops the timer when the
matching response reaches `Scheduler::commit`. The metric includes executor work and time in the model-executor request
and response channels. It does not measure GPU kernel time only. The speculative table counts every proposal position
that the Spec forward produced. It counts an accepted position when `validated_tokens` contains that position. The rate
is `accepted / proposed` at the same index.

The trie decoder keeps proposal tokens, probabilities, and confidence values in one request-local proposal state.
These vectors must have the same length. A verification-prefix trim changes all three vectors. Proposal confidence
does not change rejection sampling.

The initial scheduler policy applies an identity transform to proposal confidence. It uses cumulative confidence as a
ranking score. The runtime does not yet have evidence that these values are calibrated or comparable across requests.
The policy does not use an absolute confidence threshold.

The trie cache is the storage and reuse subsystem reached through each
request's `TrieDecoderBlocks`:

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ TrieDecoderBlocks for one request                                          │
│ queued/ready/spec tokens + epoch + per-lane block sequences                │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    v
          ┌──────────────────────────────────────────────────────┐
          │ Request block lifecycle                              │
          │ Mutable -> SemiImmutable -> Immutable                │
          │ each logical block contains one block per cache lane │
          └─────────────────────────┬────────────────────────────┘
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ MultiLaneBlockCache                                                        │
│ reserve/commit/free all configured cache lanes as one request operation    │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    v
┌────────────────────────────────────────────────────────────────────────────┐
│ Per-lane block cache                                                       │
│ block metadata + token/resource annotations + physical page ownership      │
└───────────────────┬──────────────────────────────┬─────────────────────────┘
                    │ immutable identity           │ allocate/free
                    v                              v
┌──────────────────────────────────────┐  ┌──────────────────────────────────┐
│ Trie                                 │  │ Physical page allocators         │
│ roots + token/resource edges         │  │ KV pages + GDN state pages       │
│ partitioned TrieNodeStore            │  │ globally unique page IDs         │
│ external/child pin counts            │  └─────────────────┬────────────────┘
└───────────────────┬──────────────────┘                    │
                    │ unpinned leaf candidates              │
                    v                                       │
┌──────────────────────────────────────┐                    │
│ EvictionPolicy (default: S3FIFO)     │                    │
│ select and reserve candidates        │                    │
│ accept/reject after trie decision    │                    │
└───────────────────┬──────────────────┘                    │
                    │ successful eviction                   │
                    └──────────────────────┬────────────────┘
                                           v
                         ┌────────────────────────────────┐
                         │ Page IDs may be reused         │
                         │ after ownership is released    │
                         └────────────────────────────────┘

TrieDecoderBlocks::prepare_blocks()
  -> DecoderSyncBlocks
     block_index + lane/layer page IDs
  -> DeviceRequest
  -> model executor page/state tables
```

The runtime core carries the compile-time cache-lane count as const `L`.
`MultiLaneTrieBlockCache<P, L, ...>`, runtime requests, and service runtime types preserve this const through scheduling.
`RuntimeConfig::cache_lanes` must contain exactly `L` lane configurations.
Each multi-lane allocation, reserve, commit, and free operation requires one entry for every lane.
Qwen3.5 MTP configures one Main lane and one lane for each logical MTP step.
Thus, Qwen3.5 uses `L = num_spec_tokens + 1`.

Mutable and semi-immutable blocks are request-local lifecycle objects. Trie nodes represent immutable identity. Pin
counts protect reusable nodes from eviction. S3FIFO tracks eligible unpinned leaves.

The cache and trie use `EvictionPolicy` for candidate selection and resolution.
The policy declares its concrete handle as `EvictionPolicy::Entry: EvictionEntry`.
`EvictionPolicy::new_entry` creates an unregistered pinned entry.
Each node holds `Arc<E>` for that entry type and uses static dispatch for registration, eligibility transitions, and reuse notifications.
The policy object remains `Arc<dyn EvictionPolicy<TrieNodeKey, Entry = E>>`.
The trie owns pin counts, checks eviction safety, and releases nodes and pages.

S3FIFO implements the policy with S and M queues.
Its associated entry type is `S3FIFOEntry`.
`SingleLaneTrieBlockCache::new` selects S3FIFO.
`SingleLaneTrieBlockCache::with_eviction_policy` accepts another policy without changing the trie or page allocators.
The cache traits pass the associated entry type to `TrieDecoderBlocks` through its existing cache parameter.

[`s3_fifo/entry.rs`](../crates/inference-runtime-core/src/runtime/decoder/trie_cache/s3_fifo/entry.rs) owns the concrete `S3FIFOEntry` notification handle.
[`s3_fifo/queue_entry.rs`](../crates/inference-runtime-core/src/runtime/decoder/trie_cache/s3_fifo/queue_entry.rs) owns the worker's queue links and candidate state.

The trie stores logical identity and ownership links. Physical allocators own page IDs and backing storage.

All cache lanes allocate page IDs from one shared allocator.
If a per-lane allocation or reservation exhausts the pool, `MultiLaneTrieBlockCache` tries eviction across all lanes and retries the operation.
It reports `ResourceLimitExceeded` only when the operation fails and no lane can evict a block.

## Page and cache model

Separate logical executor metadata from physical storage.

Executor-facing KV metadata is logical and lane-first:

```text
lane -> kv_block -> layer -> page_id
```

Physical storage is allocator-facing and flat:

```text
page_id -> page buffer
```

`page_id` is globally allocated. It is not scoped by lane, block, or layer.

Runtime core does not define a GPU KV page tensor layout. It allocates opaque physical pages. It reports their IDs in
logical cache-block metadata.

The executor interprets each page according to the active model. [`executor_gqa.md`](executor_gqa.md) documents the
current Qwen GQA layout.

Recommendation: Use the same logical metadata and physical page IDs for CPU, onload, and offload paths.

Do not create a separate cache object hierarchy unless scheduling or placement policy needs it.

## Managed objects

| Owner domain | Objects |
| --- | --- |
| Scheduling | Request, request slot, lifecycle table, queue, batch, batch row |
| Storage | KV/state page allocators, ownership table, accepted token/state metadata |

Recommendation: Use small lifecycle APIs: `get / get_ref / get_mut`, `set`, `allocate / free`, `push / pop`, `reset`.
Recommendation: Use distinct verbs such as `clear`, `zero`, `load`, `insert`, and `publish` together only for clearly different objects.

## Executor notification contract

Runtime core and the model executor use one ordered request-response protocol:

```text
Batch(request) -> Batch(response)
Start(plan)   -> Started
Stop(plan)    -> Stopped
```

The runtime event loop tracks the commanded `Started` or `Stopped(plan)` state.
It appends `Stop` to the same ordered channel after all batches that it has already sent.
The executor completes those batches before it handles `Stop`.
It sends `Start` with the stored Stop plan when a stopped executor has work to flush.
It can append a batch after `Start` without a separate transition state.
The ordered channel guarantees that the executor handles `Start` before that batch.
The runtime consumes `Started` and `Stopped` as acknowledgements and then attempts the next flush.
The runtime disables the idle timer while Start awaits `Started` or any batch remains in flight.
`Scheduler::num_in_flight_compute_slots()` reports the number of occupied compute slots.
After a batch-response commit or `Started`, the runtime attempts the next flush before it blocks in `select`.
It starts a new timeout interval only when no batch remains in flight after that flush.
Continuous decode does not create an idle timer between batches.
Other events preserve the existing idle deadline.

Each protocol channel reserves one entry in addition to the compute-slot capacity.
This entry preserves capacity for lifecycle commands alongside submitted batches.

The idle condition does not require zero live requests.
A request can remain in runtime core while it waits for resources.
The model state snapshot preserves executor state for that request across stop and start.

`RuntimeConfig::executor_hibernation_mode` fixes the Stop/Start plan policy at runtime construction.
For `Selected`, runtime core scans the allocation bitmaps for request slots and page IDs.
It converts set bits directly to sorted, disjoint, and nonadjacent ID ranges.
The page ranges include active requests, reusable trie cache blocks, and resources held by runtime tasks.
For `All`, runtime core sends `ExecutorHibernationPlan::All`.

The bitmap scan is not linearizable.
An allocation that occurs after its bitmap word is read cannot publish executor-visible state before the queued `Stop`.
A free that occurs after its bitmap word is read can add unused state to the snapshot without changing correctness.
Runtime core must quiesce future independent per-request I/O tasks before it scans the bitmaps.

Recommendation: Send core-owned lifecycle changes from core to the executor:

```text
request slot initialized
request slot completed/dropped/reset
prefix/cache hit selected
batch metadata constructed
accepted/rejected tokens finalized
KV/state pages allocated or freed
```

Recommendation: Do not infer global lifecycle state from a forward call. Initiate a required global request-slot reset
from core lifecycle.

If one forward must replace only executor-local state, use a specific set or update operation. Do not reset the whole
slot.

## Sync CPU task pool

[`SyncTaskPool`](../crates/inference-runtime-core/src/runtime/tasks/sync_task_pool.rs) runs synchronous CPU tasks on a
dedicated Rayon pool. It is standalone. Requests, scheduling, and guided decoding do not use it yet.

`SyncTaskPool::new(shutdown, num_runners)` creates the worker threads. `num_runners` must be positive.
Thread creation failures return `Error::Unavailable`. Cloned handles share the same workers.
Rayon owns the task queue and worker lifecycle. This pool does not need an `event_loop`.

`spawn` takes an owned `FnOnce() -> Result<T>` closure and submits it before returning.
It returns a `tokio::sync::oneshot::Receiver<Result<T>>` directly, with one channel per task.
The caller does not need to receive or poll for the task to run.
The worker moves the result to the receiver. It does not clone task inputs or results.

The receiver supports `try_recv(&mut self)`, `blocking_recv(self)`, and `.await`.
`try_recv` returns `Empty` while the task is pending and `Closed` if the sender disappears without a result.
`blocking_recv` must run outside asynchronous execution contexts. `.await` also works with non-Tokio executors.
A successful receive returns the task's `Result<T>`. A disconnected sender produces an outer oneshot `RecvError`.
The pool adds no boxed future or async wrapper around the receiver.

This example submits a state snapshot after token ID `11`:

```text
caller owns SM after token 11
  |
  +--clone--> owned snapshot --> spawn closure --> Rayon worker
  |                                                  |
  |                                            compute mask
  |                                                  |
  +--retain oneshot receiver <---- send(mask) <-------+
          |
          +--try_recv / blocking_recv / .await --> owned mask
```

Dropping the receiver closes its channel. A worker skips the task if it observes the closed channel before execution.
A running synchronous task continues. Its result is discarded when the receiver is gone.

Shared shutdown rejects new tasks and skips queued tasks with `Error::Unavailable`.
Running tasks finish normally. Queued tasks report their errors when workers reach them.
Dropping the last pool handle lets Rayon finish submitted work and stop its threads.
Dropping a handle does not join the workers.
The queue has no admission limit. Callers must bound their outstanding tasks.

A caught task panic returns `Error::Internal` with the panic message when available.
It does not trigger shared shutdown or resume the panic in the caller. The pool can execute later tasks.
A returned task error also leaves the pool running.
The Rayon panic handler triggers shutdown for uncaught panics outside the task guard, such as a panicking destructor.

## Guided decode state machine

[`GuidedDecodeSM`](../crates/inference-runtime-core/src/runtime/guided_decode.rs) defines a standalone CPU state machine.
Requests, scheduling, sampling, and service APIs do not use it yet. `DefaultStateMachine` permits every token and never
terminates. Its status is always `StateMachineStatus::Accepting`.

`try_apply` consumes one `Token` value and returns `StateMachineTransition::Allow`, `Reject`, or `Error`.
A rejection includes the token ID, a byte offset when available, and a reason. A rejection or error may change the
receiver. A caller that needs to preserve its state must clone it before applying a token.
Implementations derive or implement `Clone`. `dyn-clone` also makes `Box<dyn GuidedDecodeSM>` implement `Clone`.
Clones must share immutable data and keep mutable state inexpensive to copy or clone on write.

`fill_mask(self: Box<Self>)` consumes an owned SM snapshot and returns one `MaskFuture`.
The caller clones its SM before calling `fill_mask` when it needs to retain the original.
An implementation must submit any background work before it returns. The task owns the consumed snapshot and the
mask buffer. The future does not borrow the caller's SM. The state machine does not use `SyncTaskPool` yet.

`Mask::AllowAll` permits every token without allocating a bitset. `Mask::Bitset` owns packed `u32` words.
Bit `token_id % 32` in word `token_id / 32` permits that token when set. The bitset must cover the vocabulary.
The consumer must ignore unused bits in the final word. Mask generation must not change the logical state.
Mask bits must agree with the `Allow` and `Reject` results of `try_apply` from the same state.
`DefaultStateMachine` returns a ready future containing `Mask::AllowAll`.

`fill_spec_masks` generates one mask for the current state and each grammar-valid draft prefix.
The current state must already include the anchor token. The function accepts an iterator of `Token` values.
It clones the SM, applies drafts in order, and submits each snapshot through `fill_mask`.
It stops at the first rejected draft and returns transition errors to the caller.
The original SM remains unchanged, including when a rejected transition changes its temporary copy.
The function does not modify proposals or mark drafts as accepted by the target model.
If a draft is rejected, the final returned mask still describes the state before that draft.
Verification can use this mask to give the rejected draft zero target probability.
The consumer must preserve rejection sampling at that position. It must not treat the shorter mask list as an all-accepted proposal.

This standalone example uses anchor token `10` and drafts `35, 68`:

```text
original SM: includes anchor 10; remains owned by caller
  |
  +--clone--> trial SM
                |
                +--clone--> fill_mask --> future[0]: predicts token after 10
                |
                +--try_apply(35): Allow
                +--clone--> fill_mask --> future[1]: predicts token after 35
                |
                +--try_apply(68): Allow
                +--clone--> fill_mask --> future[2]: predicts token after 68

Main input rows:          [10,         35,         68]
Next-token mask rows:     [future[0],  future[1],  future[2]]
Target distribution for: [draft 35,   draft 68,   continuation]
```

Each snapshot has one independent future. Background work starts before the caller polls these futures.
With no drafts, the function still returns the anchor mask.
Callers can supply only the proposal prefix selected for a query. The function does not require a configured draft width.

The module has no activation controller or runtime token cursor.
Jump-forward completion is not supported. Applying one token must not append, replace, or promote other tokens.

## Stop and EOS handling

Runtime core owns token-ID stop and EOS completion at the request commit boundary. The executor reports sampled tokens.

Core checks the configured stop token sequences. It commits the exact sampled prefix through the first completion
boundary. Core then completes the turn or terminates the request.

A one-token stop sequence represents EOS.

The successful status records one of these completion reasons:

- `CompletionReason::StopSequence` when the stop matcher observed a match
- `LengthLimit` when caller-visible output reaches its limit
- `ContextLimit` when caller-visible output reaches the effective context window

`ContextLimit` wins any tie because context exhaustion terminates the request. `StopSequence` wins a tie with
`LengthLimit`. Service layers map the recorded reason. They must not infer it from the emitted token count.

Model executors may provide model-specific default stop sequences, such as Qwen
EOS token IDs. The service merges caller-provided token sequences with model
defaults and de-duplicates them before submitting the request to core.

Per-request token and probability delivery must not silently drop committed output. The current runtime uses an
unbounded internal channel between request commit and the transport-neutral `ExternalRequest`. This channel keeps
synchronous request commit non-blocking and preserves every committed output. `max_running_requests` bounds the number
of admitted request channels. `max_sampled_tokens` and `context_window` bound the caller-visible output of each request.
The runtime does not terminate a request only because its transport consumer is slow.

Dropping that response causes three actions:

- It drops the external request.
- It closes the receiver.
- It cancels the request lifecycle.

`max_sampled_tokens` and `context_window` bound the committed output. Core reserves one token of remaining
sampled-token and context capacity for the next decode response. It limits each new speculative proposal to the
capacity before that reserved token. Core also finds the earliest stop sequence that the proposal can complete. It
retains only the proper stop-sequence prefix. The final stop token remains available for the next sampled token.

The executor validates a prefix of the submitted speculative tokens. Its committed KV/GDN state ends after that
validated prefix. Each decode response also returns one final sampled token. That token is not materialized in
KV/GDN state.

Core commits every validated token and the final sampled token. A completion boundary must end at the final sampled
token. It must not end inside the validated prefix. Core applies this contract when it limits the preceding speculative
proposal. Core must not shorten the validated prefix or remove the final sampled token after executor commit.

`SampledTokens::Decode::sampled_token` and `sampled_prob` are required values. `DecoderBlocks::commit` receives the
validated prefix and the final sampled token without a completion-specific branch. It does not receive separate
completion-boundary metadata. `TokenProbs` contains the same committed tokens.

Request-slot drop adds its slot to a deduplicated reset set. It does this before it returns the slot to the allocator.

A capacity-one channel only wakes the executor. It never carries slot IDs. A full wake channel means that a wake is
already pending.

The reset notifier uses the runtime's `Shutdown` handle.
Only a disconnected wake send checks this handle.
After shutdown, a disconnected receiver does not prevent slot release.
A disconnected receiver during normal operation remains an internal invariant violation.

The executor event loop selects over wake notifications and device batches. It can therefore reset dropped slots while
the executor is otherwise idle.

When a batch arrives, the executor drains the reset set before it prepares the batch. A reused slot cannot execute
before its prior reset notification.

## Scheduler contracts and invariants

### Compute-slot queries

`Scheduler::next_compute_slot_seq()` returns `ComputeSlotSeq::Ready` when a compute slot is available.
It returns `ComputeSlotSeq::Pending` when all compute slots are occupied.
Both variants carry the next `seq_id` from one scheduler-wide counter.
Queries do not consume a sequence. Each `prepare` consumes one sequence, including an empty batch.
`cancel` releases the slot without reversing the sequence counter.
Slot availability does not imply runnable token work. `can_flush` checks both conditions.

`Scheduler::num_in_flight_compute_slots()` returns the number of prepared batches pending `commit` or `cancel`.
`SimpleScheduler` derives this count from its occupied compute slots.
`InstrumentedScheduler` forwards both queries to the wrapped scheduler.

```text
operation       slot A    slot B    slot C    next_compute_slot_seq()       in flight
initial           -         -         -      Ready   { seq_id: 1 }              0
prepare           1         -         -      Ready   { seq_id: 2 }              1
prepare           1         2         -      Ready   { seq_id: 3 }              2
prepare           1         2         3      Pending { seq_id: 4 }              3
commit 1          -         2         3      Ready   { seq_id: 4 }              2
prepare           4         2         3      Pending { seq_id: 5 }              3
cancel 4          -         2         3      Ready   { seq_id: 5 }              2
```

### Main inputs and multi-lane cache tokens

[`mtp_design.md`](mtp_design.md#schedulerexecutor-protocol) owns the shared scheduler/executor MTP protocol.
It uses before/input/after tables for Prefill and Decode transitions, including all three speculative acceptance outcomes.
See [scheduler cache commit](mtp_design.md#scheduler-cache-commit) for token storage, mutable overwrite, block crossings, and publication.
The source owners are `TrieDecoderBlocks::prepare`, `commit`, and `write_tokens`.
Vanilla and DFlash2 use the same Main progress and block commit operations without shifted MTP lanes.

If the scheduler guarantees a condition, rely on it and assert at the boundary where violations become visible.

Examples:

```text
page IDs are allocated before executor sees them
batch rows have valid token_index/block_index metadata
accepted/rejected token metadata is internally consistent
request slot lifecycle notifications are ordered
```

The scheduler forms FIFO ragged batches from runtime readiness and token-budget constraints. It does not group requests
by these model-executor sampling parameters:

- Top-k
- Top-p
- Temperature
- Seed
- Speculative stage

A model executor must accept the resulting mixed batch. It must compact or partition the batch, or select replay
geometry internally. It must not change scheduler policy.

Request status tracks lifecycle ownership, not scheduler placement:

```text
Initialized -> transient status during request construction
Running     -> request owns a request slot in the normal runtime path
Swapped     -> reserved for a request whose model state is not device-resident
terminal    -> Cancelled, TimedOut, Aborted, Evicted, or Completed(CompletionReason)
```

Scheduler-internal locations such as new queue, run queue, pending device work,
and response commit do not create additional request statuses.

Runtime-critical background threads hold a `ShutdownGuard` for their full lifetime. Normal return and panic unwind both
drop the guard. Both paths notify the other runtime loops to stop.

A failed worker must not leave a partially live service waiting indefinitely.

`PrepareResult::Skip` means a request has no additional runnable query while
previous scheduled work is still in flight. The batcher keeps it in the
request-ID map. It does not put the request back on the run queue. The model executor response returns it through
`commit`.

Do not requeue a skipped request until its in-flight work makes it runnable. Device batch responses must commit in compute-sequence order because decoder
scheduled token ranges are FIFO-owned. Within a batch, commit and cancellation resolve each request by request ID.
Request order within a batch does not have to match the scheduler's selection order.

Commit and cancellation put a continuing request at the front of the run queue. The queue deduplicates an ID that is
already runnable.

Pipeline stages can overlap. Their final batch responses must preserve compute-sequence order until core owns an explicit epoch or
reorder buffer.

An out-of-order response is an internal contract violation.

Recommendation: Fail fast on this response until a real transport reorder contract exists.

`ScheduleQueue` owns reservation waits in `waiting_reqs`.
`RuntimeConfig::max_running_requests` bounds this collection through the request-slot limit. Every admitted request
owns exactly one slot while it waits.

The async-task channels carry resource materialization work separately from reservation waits. Their capacities equal
`RuntimeConfig::max_running_requests`.

The run queue has priority over the new queue:

- A continued request goes to the front.
- A completed reservation wait goes to the back.
- A new request enters the new queue.

Actual KV and state onload and offload also remain future work. This work must not use the reservation-wait name as if
data movement already existed.

Do not add executor-side recovery for impossible scheduler states unless there is a real runtime recovery path.

## Core exclusions

Recommendation: Keep tensor/checkpoint/Metal layouts, GQA/GDN kernel semantics, MLP routing/fusion, and component profiling in the executor/backend.
See the [ownership boundary](high_level.md#core-vs-executor).

## Verification

Run the runtime-core owner tests with `cargo test -p inference-runtime-core --lib`.
`blocks/trie/api_test_single_lane.rs` covers one cache lane with and without speculative proposals.
`blocks/trie/api_test_w_mtp.rs` covers shifted multi-lane token and cache progress.
Each module uses the production trie cache and page allocators.
The service runtime tests cover admission, completion, and hibernation through the scheduler event loop.
