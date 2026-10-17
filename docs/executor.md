# Model Executor Architecture

The model executor turns runtime-owned batch metadata and page IDs into model execution. It owns model structure,
weights, component state, replay composition, and sampling. It does not schedule requests or allocate globally owned
cache pages.

`ReplayableDecoderModel::prepare_batch` can reorder complete requests before it constructs model and component metadata.
The caller retains that order through `commit_batch`. Request IDs preserve the response identity across this reorder.

Read this document after the top-level [README](../README.md) and [`core.md`](core.md). It explains the stable executor
mental model. Component documents describe current source and algorithms. Workflow documents contain shared commands
and cross-component measurement rules.

## Boundary and ownership

```text
runtime core
  schedule requests
  own request/cache lifecycle and physical page allocation
  produce batch metadata and page IDs
          |
          v
model executor
  bind model weights and component state
  interpret page IDs for GQA/GDN
  compose model stages and replay programs
  return sampled decisions and component results
          |
          v
Metal backend
  own device, buffers, kernels, command recording, and replay submission
```

The executor consumes runtime decisions. It must not recreate scheduler policy.

The runtime transports model inputs and outputs. It must not parse model tensor layouts or component-local page
contents.

The source dependency direction puts model semantics at the top and Metal execution at the bottom:

```text
runtime core ---------------- batch, page IDs, lifecycle ------------------+
                                                                          |
executor core ---------------- Core, ReplayShape, CPU reference ----------+
                                                                          v
                                                               Qwen model executor
                                                               stage order and roles
                                                                          |
                                                                          v
                                                               executor component adapter
                                                               weights/state/metadata/scratch
                                                                          |
                                                                          v
                                                               backend Metal component
                                                               reusable compute and tuning
                                                                          |
                                                                          v
                                                               backend Operator / Invocation
                                                               bindings, barriers, dispatch
                                                                          |
                                                                          v
                                                               Metal runtime
                                                               buffers, kernels, replay, submit
```

The arrows show use and lowering. They do not transfer semantic ownership to a lower layer.

## Shared component model

GQA, GDN, dense MLP, MoE, and sampling use the same ownership pattern:

```text
backend-neutral core/config and shape contract
  -> backend component with immutable kernels/tuning
  -> model-executor adapter with weights, state, metadata, and scratch
  -> typed record input/output at the model boundary
  -> CPU/reference oracle and focused production-owner tests
```

Component-specific complexity stays behind that boundary. Backend APIs remain model-independent. Qwen adapters supply
model dimensions, weights, and measured defaults.

Validation follows the same ownership:

- Config loading returns errors when it cannot parse or normalize checkpoint data.
- Component `Core` and backend constructors assert their static geometry once.
- Replay recording checks only current-batch shape, capacity, and binding contracts.

Do not restate component geometry during model normalization. Do not revalidate immutable core and config state on each
record.

Do not add wrappers only to align names. A type is useful when it owns a semantic boundary, invariant, resource, or
lifecycle.

`GQAMetadataBuffers` and `GDNMetadataBuffers` own reusable GPU metadata buffers for one batch. `GQAInput` and `GDNInput`
borrow the record-time tensors and component metadata.

`GQAOutput` and `GDNOutput` name the corresponding component outputs. They do not introduce another allocation owner.

## Common entities

The same entity vocabulary applies across attention, MLP, sampling, and model composition. A component can omit an
entity when its contract does not need it.

| Layer | Common entity | Primary responsibility |
| --- | --- | --- |
| Executor core | `*Core` | Backend-neutral geometry and semantic relationships |
| Executor core | `*ReplayShape` | One recorded work shape and its submission capacity |
| Executor core | `ReplayBucketPolicy` | Shared capacity algorithm with executor-supplied topology boundaries |
| Executor core | `*_reference` | CPU oracle with no Metal dependency |
| Model executor | `Qwen*`, Main, MTP, DSpark, DFlash2 | Model role, stage order, checkpoint bindings, and persistent state |
| Model executor | `Replay<T>`, `ReplayComponent` | One semantic replay-stage owner and its topology cache |
| Model executor | `FullStateIO`, `SelectedStateIO` | Full or selected state transfer at the state owner |
| Executor component | `*Input`, `*Output` | Typed record-time boundary with borrowed resources |
| Executor component | `*MetadataBuffers`, `*Scratch` | Reusable batch metadata and temporary workspace |
| Executor component | `*StateTable`, `*RequestSlots` | Component-local persistent state and request-slot interpretation |
| Executor component | `ReplayLayer` | Typed lowering from one semantic component to `ReplayOp` values |
| Executor component | `Recorder<ReplayOp>` | Backend-neutral stage composition through Metal replay operations |
| Backend component | `*Config`, `*Shape` | Static workload facts and one backend invocation shape |
| Backend component | `*Buffers`, `*Weights`, `*Scratch` | Borrowed backend resource groups for one invocation |
| Backend component | `ExecutionVariant`, `VariantKey`, `KernelConstants`, `*KernelKind` | Selectable execution, replay identity, compile-time constants, and low-level kernel identity |
| Backend component | `*Invocation` | One recordable backend operation or command sequence |
| Metal resource | `Device`, `Buffer`, `BufferView`, `Kernel`, `Stream` | Metal resource and submission ownership |
| Metal resource | `BufferIO` | Direct file and shared-buffer range transfer |
| Metal runtime | `Operator`, `CommandRecorder` | Kernel, resource, constant, barrier, and dispatch recording |
| Metal runtime | `ReplayProgramBuilder`, `ReplayProgram` | Recorded command construction and stable resource retention |
| Metal runtime | `ReplayArguments`, `ReplaySubmission` | Submission-time values and in-flight completion ownership |

`Stream::new_on_same_queue` creates an independent submission owner on an existing Metal command queue.
Each stream owns its allocator, completion handler, and replay residency set.
A replay must return to the stream that recorded it, even when another stream shares that queue.
Sharing a queue does not establish resource dependencies. Owners must retain their synchronization boundaries.

`*Buffers`, `*Weights`, and `*Scratch` group borrowed resources. Their names do not imply allocation ownership.
The containing model or component owns each persistent resource.

Use these source files as concrete examples:

- [`def/layer.rs`](../crates/inference-executor-metal/src/def/layer.rs) defines `ReplayLayer`.
- [`replay.rs`](../crates/inference-executor-metal/src/replay.rs) defines `Replay<T>` and `ReplayComponent`.
- [`mlp/dense/backend.rs`](../crates/inference-executor-metal/src/mlp/dense/backend.rs) is an executor component adapter.
- [`layer/dense_mlp.rs`](../crates/inference-executor-metal/src/model/qwen/v3_x/layer/dense_mlp.rs) is a Qwen model owner.
- [`dense_mlp.rs`](../crates/inference-backend-metal/src/components/dense_mlp.rs) is a backend component.
- [`operation.rs`](../crates/inference-backend-metal/src/metal/stream/operation.rs) defines `Operator` and `CommandRecorder`.
- [`stream/replay.rs`](../crates/inference-backend-metal/src/metal/stream/replay.rs) defines the Metal replay program lifecycle.

### Backend operator pattern

This generic pattern shows a replay-recorded backend operator at the lowest component boundary. The names are
placeholders. Its invocation constructor validates the shape, buffer extents, and shader domain before recording.

```rust
pub struct ComponentInvocation<'a> {
    kernel: &'a ComponentKernel,
    shape: ComponentShape,
    buffers: ComponentBuffers<'a>,
    num_active_items_key: ReplayParameterKey,
}

impl Operator for ComponentInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(&self.kernel.kernel);
        recorder.set_buffer_read(0, self.buffers.input, 0);
        recorder.set_buffer_write(1, self.buffers.output, 0);

        recorder.bind_u32(2, self.num_active_items_key, 1, self.shape.num_total_items);

        recorder.dispatch_threadblocks(self.shape.grid(), self.kernel.threads_per_threadblock());
    }
}
```

The backend operator selects kernels and dispatch geometry when it creates the invocation.
Recording uses that validated selection and declares the submission-time parameter domains.
It does not own model stage order, request lifecycle, or checkpoint names.

### Executor component pattern

This generic pattern shows how an executor component lowers typed semantic input to a backend invocation.

```rust
pub struct Component {
    compute: BackendComponent,
}

pub struct ComponentInput<'a> {
    pub shape: ComponentReplayShape,
    pub input: &'a Buffer,
    pub output: &'a Buffer,
    pub scratch: ComponentScratchBindings<'a>,
    pub weights: ComponentWeights<'a>,
}

impl ReplayLayer for Component {
    type Input<'a> = ComponentInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        input.shape.validate();
        recorder.record_with_barrier_before(ReplayOp::opaque(self.compute.invoke(
            backend_shape(input.shape),
            input.input,
            input.output,
            input.scratch,
            input.weights,
        )));
        input.output
    }
}
```

The executor component owns the semantic input and output contract. It also owns component metadata and scratch
interpretation. The backend invocation stays responsible for Metal binding and dispatch.

### Model owner pattern

This generic pattern shows a model owner with symmetric weight residency and a direct record path.

```rust
pub struct ModelComponent {
    core: ComponentCore,
    backend: Component,
    weights: Option<ComponentWeightBuffers>,
    scratch: Rc<ComponentScratch>,
}

impl ModelComponent {
    pub fn load_weights(&mut self, store: &mut SafeTensorStore) -> Result<(), ModelExecutorError> {
        assert!(self.weights.is_none(), "component weights are already loaded");
        self.weights = Some(ComponentWeightBuffers::load(store, &self.core)?);
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        assert!(self.weights.is_some(), "component weights are not loaded");
        self.weights.take();
    }

    fn weights(&self) -> &ComponentWeightBuffers {
        self.weights
            .as_ref()
            .expect("component weights must be loaded before execution")
    }

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        shape: ComponentReplayShape,
        input: &'a Buffer,
        output: &'a Buffer,
    )
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let _ = <Component as ReplayLayer>::record(
            &self.backend,
            recorder,
            ComponentInput {
                shape,
                input,
                output,
                scratch: self.scratch.bindings(),
                weights: self.weights().as_borrowed(),
            },
        );
    }
}
```

The model owner resolves checkpoint bindings and persistent resources. It invokes its layers in semantic order. It
does not select a backend kernel or duplicate backend topology thresholds.

### Replay lifecycle pattern

Recording fixes command topology and capacity. Submission supplies dynamic values within that recorded domain.

```rust
let (key, _cache_hit) = replay.record(&runtime, &input);
let program = replay.replay(&key);

let arguments = ReplayArguments::new()
    .with_u32(NUM_ACTIVE_ITEMS, input.num_active_items);
let submission = runtime.submit_replay_with_arguments(program, &arguments);
submission.wait();
```

`_cache_hit` reports whether the replay already existed. It does not change component semantics.

A stateful component can add `prepare`, `commit`, `cancel`, `publish`, or `restore` when its real contract needs those
operations. A stateless component must not add them for symmetry.

## Current source areas

```text
crates/inference-executor-core/src/
  attn/gqa       backend-neutral GQA metadata and shapes
  attn/gdn       backend-neutral GDN metadata and shapes
  mlp/dense      backend-neutral dense gated-MLP metadata
  mlp/moe        backend-neutral MoE metadata and execution policy
  model/qwen     Qwen config, microbatch, and pending-transaction contracts
  sampling       sampling config, RNG domains, shapes, and CPU references

crates/inference-executor-metal/src/
  attn           GQA/GDN adapters, batch metadata, page/state tables, scratch
  mlp            dense-MLP and MoE adapters
  model/qwen     semantic model/layer components, weights, replay stages, MTP, DSpark, DFlash2
  sampling       top-k/top-p, DSpark Markov, DFlash2 selection, and sparse rejection replay owners

crates/inference-backend-metal/src/
  metal          reusable Metal device/buffer/kernel/stream/replay runtime
  operators      recordable backend operations without model semantics
  components     reusable GQA, GDN, MLP, sampling, norm, embedding, and page-I/O kernels
  **/*_test.rs   longer unit-test suites for the adjacent production owner
```

For exact files and current paths, use the component documents:

- [`executor_qwen.md`](executor_qwen.md): Qwen semantic model loading, request state, replay stages, MTP, DSpark, and
  DFlash2.
- [`mtp_design.md`](mtp_design.md): MTP input composition, sequential proposals, cache lanes, and sampling.
- [`dspark_design.md`](dspark_design.md): DSpark block attention, Markov sampling, confidence, and persistent context.
- [`dflash2_design.md`](dflash2_design.md): DFlash2 sliding attention, dynamic convolution, and proposal selection.
- [`executor_gqa.md`](executor_gqa.md): GQA projection, KV pages, attention map and reduce, and outputs.
- [`executor_gdn.md`](executor_gdn.md): GDN projection, short convolution, recurrence, and state pages.
- [`executor_dense_mlp.md`](executor_dense_mlp.md): dense gated MLP.
- [`executor_moe.md`](executor_moe.md): routing and sparse expert execution.
- [`executor_sampling.md`](executor_sampling.md): ordinary sampling and sparse rejection.
- [`executor_model_primitives.md`](executor_model_primitives.md): embedding, unembedding, normalization, residual, and
  fused RoPE components.

## Model composition

Model-specific code connects reusable components. It does not absorb their implementations. A Qwen main forward has
this conceptual flow:

```text
token IDs
  -> embedding
  -> repeated transformer layers
       input norm
       GQA or GDN
       residual / post-attention norm
       dense MLP or MoE
       residual
  -> final norm
  -> unembedding
  -> ordinary sampling or Main verification distributions
  -> optional MTP, DSpark, or DFlash2 proposal and rejection flow
```

Normalized model configuration selects each layer variant. Exact typed binding subtrees identify the weights.

Semantic layer and component `load` functions consume those inputs directly.
Each model role owns only the plan needed to convert its configuration to reusable components.

The layer owns stage ordering and scratch handoff. The component owns reusable math, backend dispatch, and
component-local state interpretation.

Backend-neutral core types name component geometry and runtime replay shapes. Metal `ReplayLayer` names the typed
record input and output and the record operation.

`ReplayLayer` is intentionally lightweight. Page tables, routing keys, state transactions, and other component metadata
remain explicit typed input. An artificial tensor-to-tensor API does not hide this metadata.

## Weight contract

Model weights are immutable while loaded.
Each later load must materialize the same checkpoint representation.

- Load only the tensor set for the current semantic owner or layer into a `TensorMap`.
- Preserve each tensor key, storage dtype, shape, and bytes in the map value.
- Let the semantic owner remove its exact tensors from the map.
- Treat a nonempty map after owner construction as an incomplete ownership transfer.
- Parse model layout and validate shapes while loading.
- Complete required relayout, slicing, head reordering, and byte-level fusion during initialization.
- Materialize backend-owned immutable buffers and views, then release checkpoint mmap and file ownership when possible.
- Do not rewrite, relayout, or fuse model weights per request or token.
- Do not silently dequantize a full unsupported quantized weight. Fail explicitly when no runtime kernel supports it.

Persistent model parameters must preserve the checkpoint storage dtype and quantized representation. Kernels perform
dequantization, precision promotion, and model-defined numeric transforms during execution.

Runtime numerical state and scratch may use the compute dtype required for stability. These buffers are not model
weights.

If measured execution requires a precomputed parameter transform, give the derived execution resource an explicit name
and owner. Do not expose it as the loaded checkpoint weight. Record the supporting performance evidence.

Recommendation: Do not use hot-path `contiguous` calls. If execution requires a layout, prepare it during
initialization.

`SafeTensorStore` reads safetensors into a bounded `TensorMap` and releases the mapped shards after the copy.
It does not materialize the full checkpoint at one time.
Exact binding trees contain keys only.
They do not own tensor bytes or duplicate checkpoint values.
Generic embedding and unembedding owners apply the same contract to their exact quantized tensor binding.

## Metal lowering boundary

Executor code lowers semantic components into backend recordable operations:

```text
ReplayLayer::record(typed input)
  -> Recorder<ReplayOp>
  -> backend Operator::record
  -> pipeline + buffer/constant bindings + dispatch
  -> ReplayProgram
  -> submission-time ReplayArguments
```

The executor owns semantic stage order and the buffers exchanged between stages. Backend operators own kernel binding,
resource usage, dispatch, and internal phase barriers.

The backend [`README`](../crates/inference-backend-metal/README.md) contains these details:

- The complete Metal object model
- Residency rules
- The stream lifecycle
- The minimal Add One example

This document does not duplicate those details.

### Buffer and scratch ownership

Keep these domains distinct:

| Object                  | Owner                                      | Meaning                                                             |
| ----------------------- | ------------------------------------------ | ------------------------------------------------------------------- |
| Immutable weight buffer | model or component                         | initialized once and shared across replays                          |
| Runtime page buffer     | runtime allocates and component interprets | persistent KV or GDN state addressed by runtime page IDs            |
| Batch metadata buffer   | component batch-metadata owner             | current batch's offsets, page IDs, or state slots                   |
| Scratch buffer          | component, layer, or model scratch owner   | temporary partials and intermediates with explicit reuse boundaries |
| Replay parameter buffer | backend replay program                     | submission-time scalar values for one recorded program              |

A `Buffer` is raw storage. A tensor or weight view adds dtype, shape, layout, and byte offset.

Different views may intentionally alias one buffer. Scratch reuse is correct only when the next writer cannot destroy
data that a later stage still consumes.

### Barriers

A barrier belongs to the consumer command that must wait. Layer entry barriers protect cross-component dependencies.
Backend components retain their internal phase barriers.

Do not infer barriers from method order. Do not duplicate them at both layers. Do not add barriers to independent
reads.

## Replay composition

Recording is expensive relative to replay. The executor therefore caches stable command topology.

A replay key contains only values that change recorded commands, dispatch topology, static geometry, or scratch layout.
Metadata buffers or `ReplayArguments` contain dynamic values that fit an existing recording.

```text
static / replay-defining
  component geometry and tuning
  total capacity for each replayed work domain
  command topology and scratch extent

dynamic / submission-scoped
  valid request, token, row, task-template, or partial-output counts
  page IDs, offsets, state slots, sampling parameters
  other values consumed through current batch metadata
```

Capacity-bucket replay is safe only when each participating kernel causes inactive lanes to return before these
actions:

- Reading input
- Mutating state
- Advancing RNG
- Writing output

Otherwise, the capacity policy selects the active count as the total capacity. This identity policy can record more
programs, but it preserves the same architecture: the total capacity remains in the key, and the active count remains
a submission parameter. Padding is a dispatch property. It does not permit changes to valid work or semantic
descriptor counts.

The shared default capacity policy starts with these buckets:

```text
1 2 4 6 8 12 16 20 24 32 40 48 56 64
```

Above `64`, the policy divides each power-of-two interval into quarters. For example, the next buckets are `80`,
`96`, `112`, and `128`. The configured capacity is always a terminal bucket. A bucket must not exceed allocated
scratch or buffer capacity.

A component must insert a terminal bucket before each recorded-topology boundary. The component must keep categorical
topology choices in its replay key. Capacity bucketing must not combine different kernel selections, command counts,
or dispatch structures. A topology boundary identifies the first active count for the new topology. The policy ignores
a boundary above the configured capacity. The backend component that selects the topology must own these boundaries.

Some work domains permit zero active work. A policy result of zero means that the domain does not record or dispatch
work. Zero is not a replay capacity. A fixed graph can instead retain an optional branch at a positive recorded
capacity and submit zero active work. The branch must satisfy the inactive-kernel contract above. This form preserves
command topology but retains dispatch and barrier costs.

`ReplayArguments` contain keyed submission values that recording declares. Each cached replay work domain declares its
active count as one of these values. Submission validates that the caller provides each declared value exactly once
and within its recorded bounds. Active and total counts remain separate when their values are equal.

These arguments prevent a program rebuild for scalar activity changes. They do not replace component batch metadata.

### Qwen replay stages

Qwen keeps separate replay caches for these semantically separate stages:

- Main forward
- Main output
- GDN state restore
- Ordinary sampling
- MTP proposal
- DSpark context append
- DSpark proposal
- DFlash2 history append
- DFlash2 proposal
- Spec sampling
- Rejection sampling

A cache boundary exists when command topology, lifecycle, or CPU dependency differs. A component name alone does not
create a cache boundary.

Normal forward, output, and sampling commands can share one ordered command buffer when they share one dependency chain.

Qwen3 and Qwen3.5 DSpark, and Qwen3.5 DFlash2, record Main, rejection sampling, Spec Decode prepare, Spec
Prefill, Spec Decode, and proposal sampling in one ordered GPU submission. Qwen3.5 MTP retains its established
Spec lifecycle boundary. GDN chunkwise forward writes final and cache-boundary states. GDN replay Decode retains
per-forward deltas and reconstructs the accepted state after sampling. Reconstruction and page publication share one
post-forward submission. See [`executor_gdn.md`](executor_gdn.md) for the state lifecycle.

The generic executor lifecycle uses role-qualified Main and Spec hooks:

```text
embed_main -> forward_main -> unembed_main -> sample_main
submit_main -> wait -> read_main

run_spec
embed_spec -> forward_spec -> unembed_spec -> sample_spec
submit_spec -> wait -> read_spec

run_spec_prefill -> prefill_spec                  when Spec Prefill work exists
run_spec_decode -> decode_spec                    when Spec Decode work exists
submit_spec -> wait -> read_spec
```

The first Spec form is one combined invocation.
Qwen3.5 MTP uses this form and keeps its existing `embed_spec`, `forward_spec`, `unembed_spec`, and `sample_spec` order.

The second Spec form is the generic interface vocabulary for independent Prefill and Decode invocations.
Qwen3 and Qwen3.5 DSpark, and Qwen3.5 DFlash2, do not select this later fixed-block hook lifecycle.
They record the fixed-block sequence as part of Main recording:

```text
Main -> rejection sampling -> Spec Decode prepare -> Spec Prefill -> Spec Decode -> proposal sampling
submit_main -> wait -> read_main
```

Their `run_spec_prefill` and `run_spec_decode` hooks return false because the integrated sequence already contains this
work.
The corresponding later fixed-block hook methods are invalid for these executor paths.
One model mode must not select both forms for the same batch.

One model-specific Spec Decode owner can compose embedding, model layers, output, and sampling.
This composition keeps the complete Spec Decode lifecycle at the model role boundary.

Detailed keys, stage order, and request lifecycle are in [`executor_qwen.md`](executor_qwen.md). Sampling and rejection RNG
and write-distribution contracts are in [`executor_sampling.md`](executor_sampling.md).

## Concurrency and lifecycle

The current decoder service path uses one decoder executor synchronously.
The decoder executor prepares a batch, executes it, obtains the result, and commits before the next decoder batch.
Audio and Vision encoder executors use independent workers and Metal streams.
They can execute while the decoder worker processes unrelated requests.

Replay caches, scratch owners, request-slot state, and pending GDN commits therefore remain executor-owned. They
remain confined to one thread unless an API explicitly states otherwise.

Runtime core still owns the durable request and cache lifecycle. The executor reports sampled decisions and component
results. It does not free globally owned pages or commit scheduler state independently.

`inference-executor-core` owns `ReplayableDecoderModel`, `ReplayableEncoderModel`, executor timing, submission, and page
interpretation contracts.
`ReplayableDecoderModel` also defines synchronous model residency operations.
`model_name()` reports model identity. `model_mode()` reports the executor-owned `vanilla`, `mtp`, `dspark`, or
`dflash2` composition mode for service telemetry.
All recoverable model operations return `ModelExecutorError`.
Current Qwen model executors support this order:

```text
stop: clear_replay_cache -> unload_state -> unload_weights
start: load_weights -> load_state
```

Audio and Vision use standalone encoder executors.
`AudioEncoderExecutor` and `VisionEncoderExecutor` each own one worker thread and one Metal stream.
The worker owns a private model that implements `ReplayableEncoderModel`.
This trait defines `prepare`, `record`, `submit`, and `complete` without decoder request slots, sampling, KV state, or
snapshot operations.
The encoder records one replay program for each current request.
Replay reuse requires a future submission-argument contract for source and arena ranges.

The service owns the heterogeneous `EncoderExecutorLifecycle` collection.
`ReplayableModelExecutors<M>` groups one decoder with this encoder collection at the service boundary.
It does not add execution or scheduling policy.
It stops each encoder before it unloads decoder state and weights.
It starts each encoder after it loads decoder state and before it accepts the next decoder batch.
Encoder Stop drains earlier jobs on the encoder FIFO before it unloads weights.
Encoder Start reloads weights on the same worker and stream.
An encode job that races after Stop reloads the encoder before it records work.
This rule prevents the scheduler hibernation timer from invalidating an in-flight resource task.

The model shell remains allocated while its resources are unloaded.
Weight-bearing component shells remain allocated and fail fast if execution accesses missing weights.
The unload traversal must remove all shared `Rc` owners.
The final owner releases the shared Metal resource.

`unload_state` writes the selected `PageArena`, GQA, and GDN payloads to SSD before it releases their buffers.
Direct callers can still select `ExecutorHibernationPlan::All`.
The snapshot also stores durable GDN request state and future publish page IDs.
The executor finishes pending GDN reconstruction and publication before it writes the snapshot. Transient replay logs
are not part of the snapshot.

State-bearing components implement the symmetric `FullStateIO` and `SelectedStateIO` traits.
`PageArenaStateSnapshotFiles`, `GQAStateSnapshotFiles`, and `GDNStateSnapshotFiles` identify their semantic files.
Read methods require mutable component state.

The writer and reader validate the same topology-specific semantic file set.
`load_state` validates the snapshot container before it allocates state resources.
Each component validates its resource length before it reads the resource.
It attaches consumers only after all state reads succeed.
It releases all new resources after a read failure.

The Metal backend also provides the standalone `BufferIO` component.
It transfers byte ranges between files and shared Metal buffers without an application staging buffer.
`BufferIOFile` owns the POSIX and Metal handles for one file.
`BufferIOFileCacheMode::Uncached` bypasses the macOS data cache for positional and Metal file I/O.
The current v3 directory snapshot uses `BufferIO` for each Metal buffer resource.
It uses native-endian `wincode` for the manifest. It streams direct `GDNRequestSlots` metadata with the same
configuration. It does not use a GDN snapshot DTO.
The manifest stores the exact `ExecutorHibernationPlan` used by Stop and Start.
See [`model_state_io.md`](model_state_io.md) for the current format and remaining work.

`ReplayableDecoderModelEventLoop` invokes these operations for idempotent `Start` and `Stop` commands.
It coordinates the registered encoder executor lifecycles at the same command boundary.
It also starts a stopped model before it executes a batch.
Runtime core tracks executor residency and sends ordered lifecycle commands after the configured idle period.
See [`executor_hibernation.md`](executor_hibernation.md) for the current lifecycle and remaining work.

## Verification boundary

Recommendation: Use the narrowest production owner that can express the invariant:

- CPU references prove math.
- Backend tests prove shader build, dispatch, ABI, and parity.
- Executor component tests prove real metadata, state, and scratch ownership.
- Layer and end-to-end tests prove composition and lifecycle.

Component unit tests use the same classification model across attention, MLP, sampling, and model I/O. They do not
force the same lifecycle on components that have different ownership contracts.

- GDN metadata tests exercise exact, bucketed, and caller-owned token-capacity APIs. The GDN numerical replay test
  sweeps all active counts for a total capacity of `8` and checks persistent state.
- GDN state tests exercise mixed chunkwise/replay commits, accepted-prefix reconstruction, deferred publish, restore, and selective reset.
- GQA metadata tests exercise sliced-KV and sliced-Q paths with exact, bucketed, and caller-owned capacity APIs.
- GQA page-table tests exercise selected state I/O and selective reset without reproducing KV-kernel math.
- MoE component tests protect execution-variant selection. Isolated replay tests compare token-major, expert-major,
  shared-expert, and non-shared-expert active outputs with CPU references.
- Sampling tests sweep all active rows in one recorded capacity. They protect mixed request configurations and Target
  and Draft runtime-parameter domains.
- Rejection tests sweep all active requests in one recorded capacity. They protect mixed ragged requests, zero-draft
  requests, prepared inputs, and result prefixes.
- Dense MLP, Embed, and RowGather isolated replay tests sweep active counts and compare active outputs with CPU
  references.
- Quantized affine, RMSNorm, Softmax, and BF16 row-concat replay tests use caller-provided active parameters. They
  compare only the active logical output with CPU references.
- The Qwen3.5 MTPEmbed component test uses nonzero weights. It compares previous-hidden gather, token embedding, and
  the complete norm-concat-projection output with CPU references.
- The Unembed test protects numerical replay parity. Qwen3, Qwen3.5, and DSpark GatherUnembed tests compose gather and
  affine-unembed CPU references across all active counts for one recorded capacity. The DSpark test also protects the
  request-major to step-major row conversion.

Do not add a component test that only inspects replay keys, arguments, capacity identities, or record wiring. A replay
contract test must execute the production owner, prove cache reuse across active counts, and compare the active logical
result with an exact reference. The central replay infrastructure test does not replace this owner test.

Keep CPU-reference tests in `inference-executor-core`. Keep Metal parity, ABI, padding, and canary tests in their Metal
owner. Keep lifecycle scenarios independent of both layers.

Do not reshape production source for test construction. Do not add naming-only tests. Run Metal tests serially.

[`executor_benchmarks.md`](executor_benchmarks.md) defines these shared topics:

- The full verification ladder
- Benchmark targets
- Profiling vocabulary
- Performance-evidence rules

## Operational workflows

- Model download, server and client commands, logging, cold-start separation, and end-to-end helpers:
  [`service.md`](service.md).
- Tests, benchmarks, profiling, and performance claims: [`executor_benchmarks.md`](executor_benchmarks.md).
- Shared naming, API, ownership, and definition-of-done rules: [`engineering_conventions.md`](engineering_conventions.md)
  and [`high_level.md`](high_level.md).
- Active investigations: [`future_work.md`](future_work.md).

The Metal backend embeds MLX-derived headers at build time through `build.rs`. The repository
[`NOTICE`](../NOTICE) retains their MIT attribution.
