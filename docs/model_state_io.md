# Model State I/O Design

Model-state I/O moves mutable executor payload between backend memory and files.
Whole-model Stop/Start preserves the live runtime object graph and its page IDs.
Planned request and cache mobility can change physical placement.
This document defines their separate lifecycles and shared I/O contracts.

[`high_level.md`](high_level.md) defines the runtime-core and model-executor boundary.
[`executor_hibernation.md`](executor_hibernation.md) describes the current whole-model Stop/Start path.

This document uses `checkpoint` only for the original immutable external model weights.
It uses `snapshot` only for executor-generated mutable state from the same-executor Stop/Start lifecycle.

## Status

The repository has two model-state I/O projects.

| Project | Workstream | Status |
| --- | --- | --- |
| Whole-model residency | Full-state Stop/Start | Implemented with the v3 directory snapshot path |
| Whole-model residency | Selected-state Stop/Start | Implemented with the v3 directory snapshot path |
| Request and cache mobility | Per-request swap | Planned |
| Request and cache mobility | Trie cache-block I/O | Planned |

## Shared model

Whole-model I/O and request mobility share state descriptions and byte-range I/O.
They do not share one lifecycle contract.

```text
                                shared metadata and I/O
                               request slots + page IDs
                                          |
                    +---------------------+---------------------+
                    |                                           |
                    v                                           v
       whole-model selected                            request and block I/O
       physical identity is stable                      placement can change
       synchronous Stop/Start                           asynchronous tasks
       runtime metadata is unchanged                    runtime updates placement
```

Full-state and selected-state I/O use the same whole-model lifecycle.
They differ in which state entries the snapshot contains.

## Ownership

Runtime core owns these concepts:

- Request lifecycle and request slots.
- Trie metadata and logical cache blocks.
- KV and state page allocation.
- Physical page IDs and block placement.
- Cache eviction policy.

The model executor owns these concepts:

- `PageArena` payload interpretation.
- GQA page-table interpretation.
- GDN recurrent and convolution state.
- GDN request state and future-publish mappings.
- Model-specific gather and scatter operations.

The Metal backend owns these concepts:

- `Buffer`, `Device`, and compute `Stream`.
- The `BufferIO` Metal I/O queue.
- The `BufferIOFile` POSIX and Metal file handles.
- File-to-buffer and buffer-to-file byte-range transfers.

## Whole-model full state

The current full-state lifecycle is synchronous.
[`executor_hibernation.md`](executor_hibernation.md#lifecycle-flow) defines the surrounding encoder and request-slot reset order.

```text
runtime Stop
  -> complete each earlier batch in the ordered executor queue
  -> clear executor replay state
  -> write all mutable model state
  -> publish the snapshot
  -> release mutable Metal state
  -> unload weights
  -> Stopped

runtime Start
  -> reload weights from the original checkpoint
  -> allocate mutable Metal state
  -> read all mutable model state
  -> attach the restored state
  -> remove the snapshot
  -> Started
```

Weights are not snapshot data.
`unload_weights()` drops Metal weight residency.
`load_weights()` reloads the original checkpoint.

The v3 snapshot uses one directory and one file for each logical resource.
Metal buffer files use `BufferIOFileCacheMode::Uncached`.
The writer does not allocate an application staging buffer.
The reader transfers each file directly into its destination Metal buffer.
The snapshot layer uses symmetric buffer APIs:

```text
write_full_buffer / read_full_buffer
write_selected_buffer / read_selected_buffer
```

Local payload files do not contain checksums.

## Whole-model selected state

Selected Stop/Start persists all runtime-valid state, not only active-request state.
This path is implemented.

The runtime core supplies this semantic selection as canonical ID ranges:

```text
allocated request-slot ranges
+
allocated page-ID ranges
```

The page set includes reusable trie blocks with no active request reference.
Omission of these pages leaves valid runtime metadata with missing executor payload.

Runtime core scans the shared page allocator bitmap.
This set includes pages for active requests, unpinned reusable cache nodes, and runtime tasks.
The scan converts set bits directly to canonical ranges.
The executor converts those ID ranges to byte ranges:

```text
canonical page-ID ranges
  -> scale by PageArena entry bytes
  -> direct file ranges
```

Runtime core scans the request-slot allocator bitmap in the same way.
The bitmap scan can include a newly allocated but not yet initialized resource or a resource that is freed during the
scan. Saving that unused state is safe.
No omitted ID can publish executor-visible payload ahead of the queued `Stop`.
Future independent per-request I/O tasks must be quiescent before this scan.
`ExecutorHibernationPlan::selected(...)` enforces the canonical range contracts.

The protocol carries one shared hibernation plan:

```text
ExecutorHibernationPlan::All
ExecutorHibernationPlan::Selected { request_slot_ranges, page_id_ranges }

ReplayableModelExecutorRequest::Stop(ExecutorHibernationPlan)
ReplayableModelExecutorRequest::Start(ExecutorHibernationPlan)
```

`RuntimeConfig::executor_hibernation_mode` fixes the policy when the runtime starts:

```text
ExecutorHibernationMode::All      -> ExecutorHibernationPlan::All
ExecutorHibernationMode::Selected -> scan allocated request slots and page IDs
```

The Qwen services default to `ExecutorHibernationMode::Selected`. Use `--executor-hibernation-mode all` to write every
state entry. Runtime core creates the concrete plan at Stop. The executor consumes that plan and does not own a second
policy setting.

`Stop` and `Start` must use the same plan variant and fields.
Runtime core stores the exact Stop plan and supplies it again on Start.
The executor compares the Start plan with the snapshot manifest before it allocates restore resources.

The selected resource mapping is:

| Resource | Selection | File order |
| --- | --- | --- |
| `PageArena` | Allocated page-ID ranges | Increasing page ID |
| Main GQA request page table | Allocated request-slot ranges | Increasing request slot |
| MTP, DSpark, or DFlash2 GQA request page table | Allocated request-slot ranges | Increasing request slot |
| GDN recurrent state | Current recurrent slot for each allocated request slot | Layer, then increasing recurrent slot |
| GDN convolution state | Current convolution slot for each allocated request slot | Layer, then increasing convolution slot |
| GDN request state table | Complete durable table | Native `GDNRequestSlots` order |

The GDN request-state table remains complete because it owns independent recurrent and convolution slot allocators and
their free-slot orders.
It also contains current versions and future-publish mappings. The recurrent and convolution payload files contain only
the current slots for selected requests.

Selected restore allocates fresh GQA and GDN buffers before it reads payload files. Their unselected entries keep the
zero-initialized state for free request slots. Unselected `PageArena` entries are unspecified because runtime core does
not own those page IDs.

Each selected Metal file packs its selected entries without padding. The hibernation plan and component layout derive
every file range. The format does not need a second per-entry index or resource-coordinate graph.

For example, let the allocated `PageArena` IDs be `1`, `2`, and `5`.
Let each page entry occupy `E` bytes.
The canonical ID ranges are `[1, 3)` and `[5, 6)`.
All byte ranges below have exclusive ends:

```text
                     Stop                         Start
               device -> file                 file -> device

+---------+----------------+----------------+----------------+
| Page ID | Source bytes   | Snapshot bytes | Restored bytes |
+---------+----------------+----------------+----------------+
| 1       | [E, 2E)        | [0, E)         | [E, 2E)        |
| 2       | [2E, 3E)       | [E, 2E)        | [2E, 3E)       |
| 5       | [5E, 6E)       | [2E, 3E)       | [5E, 6E)       |
+---------+----------------+----------------+----------------+
```

The snapshot packs the payload. Restore returns each entry to its original page ID.
The snapshot includes page `5` even when only a reusable trie block owns it.

Selected Stop/Start does not change runtime state:

```text
trie metadata       unchanged
DeviceBlock         unchanged
placement           unchanged
page IDs            unchanged
request slots       unchanged
num_in_sync_blocks  unchanged
```

The design does not need a cache-generation ID.
The synchronous lifecycle keeps the same runtime object graph alive.

## Snapshot format and publication

The v3 snapshot uses this flat directory layout:

```text
snapshot/
  manifest
  page-arena
  main-gqa-request-page-table
  main-gdn-request-state-table
  main-gdn-recurrent-state
  main-gdn-conv-state
  mtp-gqa-request-page-table
  dspark-gqa-request-page-table
  dflash2-gqa-request-page-table
```

`manifest` uses `wincode` with native byte order, fixed-width integers, `u32` sequence lengths, and `u8` enum tags.
It stores the format magic, format version, hibernation plan, file kind, and exact file length.
The GDN request-state table uses the same `wincode` configuration. The writer serializes `GDNRequestSlots` directly.
It does not create a snapshot DTO or clone page mappings into an intermediate metadata graph.
The snapshot codec disables the `wincode` preallocation limit. The local snapshot is a trusted artifact from the same
executor instance. GDN does not calculate or supply a serialized-size limit. The reader streams the manifest-sized
metadata file into the decoder.

The manifest and metadata writers stream `wincode` output into their files.
The metadata writer records the actual file position after encoding. It does not run a separate serialized-size pass.
Their readers stream each file into its owned Rust value and reject trailing bytes.
These paths do not allocate a second full encoded byte buffer.

Each Metal buffer has one semantic file name.
The writer opens these files with `BufferIOFileCacheMode::Uncached`.
The MTP, DSpark, and DFlash2 GQA files are topology-dependent.
A snapshot contains at most one of them.

The writer and reader validate the same topology-specific expected file set.
The writer rejects an incomplete set before it publishes the snapshot.
The reader validates the manifest, sorted unique file set, semantic file kind, exact directory contents, regular-file
type, and manifest file lengths before the model allocates restore buffers.
Each component validates its expected buffer length before it transfers file data.
The local format does not add a payload checksum.
The manifest does not contain a model fingerprint or a model-instance nonce.
The live executor shell and its unique event-loop snapshot path establish identity for same-process Stop/Start.
The reader must not use this format to restore another executor instance, model setup, or process.

Publish a new snapshot in this order:

```text
create snapshot.tmp-<id>/
  -> write each semantic state file
  -> sync each state file
  -> write manifest
  -> sync manifest
  -> sync snapshot.tmp-<id>/
  -> rename snapshot.tmp-<id>/ to snapshot/
  -> sync the parent directory
```

The reader must only open the final `snapshot/` name.
An incomplete temporary directory is not a valid snapshot.

Cross-node transfer can calculate SHA-256 while it reads the published state files.

This whole-model lifecycle is not a process-restart checkpoint.
It keeps runtime metadata in the live process while model residency is stopped.

## Component state interfaces

The Metal backend implements the standalone `BufferIO` primitive.
The v3 model snapshot format uses `BufferIO` for each Metal buffer resource.

The model executor uses one `FullStateIO` trait for component state:

```rust
pub trait FullStateIO {
    type Files: Copy;

    fn write_full_state(
        &self,
        writer: &mut StateSnapshotWriter,
        files: Self::Files,
    ) -> Result<(), ModelExecutorError>;

    fn read_full_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
    ) -> Result<(), ModelExecutorError>;
}
```

`PageArenaStateSnapshotFiles`, `GQAStateSnapshotFiles`, and `GDNStateSnapshotFiles` identify the files for each
component. The same GQA implementation can serve Main, MTP, DSpark, or DFlash2. The executor therefore supplies the
semantic file set at the model-role boundary.

Each component keeps its `FullStateIO` implementation in an adjacent `file_io.rs` file. This layout keeps storage logic
separate from forward execution and resource allocation.

The same components implement the symmetric `SelectedStateIO` trait:

```rust
use std::ops::Range;

pub trait SelectedStateIO: FullStateIO {
    type ID;

    fn write_selected_state(
        &self,
        writer: &mut StateSnapshotWriter,
        files: Self::Files,
        id_ranges: &[Range<Self::ID>],
    ) -> Result<(), ModelExecutorError>;

    fn read_selected_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
        id_ranges: &[Range<Self::ID>],
    ) -> Result<(), ModelExecutorError>;
}
```

The supertrait enforces one file identity for all-state and selected-state I/O. `PageArena` uses `RawPageID` as its
`ID`. GQA and GDN owners use `RawRequestSlot`. These aliases expose the ID domain at each component API. The
model owner passes each field to the component that owns its interpretation.

## `BufferIO`

`BufferIO` is a concrete Metal backend component.
It is not an executor-core trait.

`BufferIO::new(&Device)` creates one serial `MTLIOCommandQueue`.
This queue is independent from the `MTL4CommandQueue` in the compute `Stream`.
`MetalRuntime` owns one compute `Stream` and one `BufferIO`.

The I/O queue sets `maxCommandsInFlight` to `1`.
Each command buffer contains one load, and the caller waits for completion before it submits the next chunk.
The serial queue type controls command ordering; it does not replace this concurrency limit.
The limit bounds I/O worker allocation when multiple `BufferIO` owners coexist.
The regression test keeps 32 independent queues alive and verifies a read through each queue.

`BufferIO::create` creates a new output file with an explicit `BufferIOFileCacheMode`.
`BufferIO::open` opens an existing input file with the same explicit mode.
Both methods return one `BufferIOFile`.
`BufferIOFile` retains one POSIX file handle and one `MTLIOFileHandle` for the same file.
This owner prevents each range transfer from reopening the file.
The public API does not accept `OpenOptions`.
This restriction prevents append mode from breaking positional file offsets.

`BufferIOFileCacheMode::Cached` uses the default macOS data cache.
`BufferIOFileCacheMode::Uncached` applies `F_NOCACHE` and `F_GLOBAL_NOCACHE` before it creates the Metal file handle.
`F_NOCACHE` selects the uncached positional-I/O path used by `buffer_to_file`.
`F_GLOBAL_NOCACHE` applies the same cache policy to the Metal URL-backed file handle used by `file_to_buffer`.
This mode bypasses the macOS data cache.
It does not guarantee bypass of an SSD controller cache.

The API has direction-based names:

```text
file_to_buffer(BufferIOFile, file_offset_bytes, Buffer, buffer_offset_bytes, len_bytes)
buffer_to_file(Buffer, buffer_offset_bytes, BufferIOFile, file_offset_bytes, len_bytes)
```

Each method lists the source and its byte offset before the destination and its byte offset.
Both methods use checked `u64` byte coordinates.
Both methods return `std::io::Result`.
The model executor must map a failure to `ModelExecutorError` at its boundary.

`file_to_buffer` uses `MTLIOCommandBuffer::loadBuffer`. It divides ranges larger than 1 GiB into serial commands.
Metal I/O rejects one command when its size reaches 2 GiB on the supported Apple Silicon path.
The method waits for Metal I/O completion before it returns.

`buffer_to_file` writes from `Buffer::contents()` with positional file I/O.
The method does not allocate an application staging buffer.
The method does not sync or publish the file.
`BufferIOFile::sync_all` syncs the file when the snapshot owner requests it.

The caller must complete earlier GPU access before either method starts.
The caller owns later GPU synchronization, file sync, and snapshot publication.

The two directions use different platform mechanisms:

```text
file -> shared Metal buffer     MTLIOCommandQueue
shared Metal buffer -> file     positional file I/O
```

`MTL4CommandQueue` cannot use a filesystem file as a command source or destination.

## Per-request swap

Status: Planned. This section defines the intended contract, not a current service capability.

Per-request swap is separate from whole-model residency.
It changes runtime-visible placement and physical identity.

Swap-out can start only at this scheduler boundary:

```text
the request's last prepared or in-flight batch commits
  -> no later batch is prepared
  -> submit request swap-out
```

Swap-in can start only for a request whose state is `Swapped`.
The scheduler must not prepare a batch before swap-in completes.

Successful swap-out releases request-slot executor resources.
It also releases request-private device pages.
Swap-in can allocate a different request slot and different page IDs.

The scheduler owns the asynchronous task.
The executor uses a separate I/O execution path.
The request task submits the operation and waits for completion.

The current reservation-wait task must not use `Swapped` as its status.
The request remains `Running` while it waits for a reservation.
Reserve `Swapped` for state that is not device-resident.

## Trie cache-block I/O

Status: Planned. This section defines the intended contract, not a current service capability.

The trie is one logical cache across Device, Host, and Disk placement.

```text
logical trie block
  -> Device(page IDs)
  -> Host(...)
  -> Disk(...)
```

Cache eviction policy owns cache-block unload decisions.
Request swap-out releases its references and private resources.
It does not independently evict shared trie blocks.

Use one `BlockIOTask` type with Load and Unload operations.
The task owns its temporary pin and any S3FIFO reservation.
Cancellation or failure must release the pin and reject an unfinished reservation.

An unload completion must check the pin count and replace placement under one trie-node mutation lock.

```text
only the task pin remains
  -> replace Device with Disk
  -> release DeviceBlock

a request pin exists
  -> keep Device
  -> optionally keep the Disk replica
```

This lock prevents a request from observing released page IDs.

New request initialization runs in its existing Tokio task.
It walks and pins the unified trie before scheduler enqueue.
It submits `BlockIOTask::Load` for each non-device block and waits in place.
The request enters the scheduler only after all selected blocks have Device placement.

## Failure behavior

A whole-model snapshot failure fails the Stop or Start operation.
The current service can use global shutdown because it has no rollback response protocol.

A failed Stop write leaves model state and weights resident.
The executor releases state only after snapshot publication succeeds.

A failed Start must release provisional allocations.
It must not attach partial state.

A request or block I/O failure must remain request-scoped or block-scoped.
It must not publish a partial placement change.

## Implementation order

1. Measure standalone `BufferIO` file-to-buffer and buffer-to-file throughput. Complete.
2. Add the v2 semantic-resource directory snapshot. Complete.
3. Migrate full-state Stop/Start from v1 staging to `BufferIO`. Complete.
4. Add `ExecutorHibernationPlan` and selected component I/O. Complete.
5. Measure full-state and selected-state throughput and peak memory with production model layouts.
6. Add per-request swap at the scheduler commit boundary.
7. Add unified trie placement and `BlockIOTask`.
8. Add cross-node transfer only after local I/O is correct.

## Deferred decisions

The following details remain open:

- The concurrency and queue-count policy for future asynchronous `BufferIO` transfers.
- Batched Metal command submission for many noncontiguous selected read ranges.
- Vectored positional writes for many noncontiguous selected write ranges.
- Uncached range alignment and other filesystem tuning.
- Trie insertion and duplicate-load coordination.
- Disk replica retention after a concurrent request pin.
- Prefill/decode transfer identity and remote publication.
