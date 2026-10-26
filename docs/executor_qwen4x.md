# Qwen4x executor

Qwen4x has an independent model graph in `model/qwen/v4_x`.
It composes the existing GDN, QSA, gated residual, dense MLP/MoE, and resident n-gram components.
The text service supports Main-only and native Main+MTP execution.
External speculators, vision, and SSD table loading are not wired.

## One concrete forward

Use two tokens from one request, hidden width `H=2560`, and residual channels `C=4`.
`T` is the flattened token count across requests.
Activations use contiguous BF16 storage.
Weights and state remain at their owning component.

```text
token IDs [T=2] + request-local previous IDs [2]
    |
    +--> n-gram hash IDs [2,16] U32
    |      resident Q4 table lookup -> E [2,2560]
    |      key/value projections -> K [2,4,2560], V [2,2560]
    |      key norm --------------------------------------------+
    |                                                          |
    +--> token embedding [2,2560]                               |
           repeat across channels -> X [2,4,2560]              |
                    |                                         |
                    v                                         |
       for each decoder layer:                                |
           if this is a configured PLE layer:                 |
               X + prepared K/V + conv history <--------------+
                   -> PLE -> X [2,4,2560]
           GR read(X) -> hidden [2,2560], write gate [2,4]
               -> GDN or QSA -> branch [2,2560]
           GR write(X, gate, branch) -> X [2,4,2560]
           GR read(X) -> hidden [2,2560], write gate [2,4]
               -> dense MLP or MoE -> branch [2,2560]
           GR write(X, gate, branch) -> X [2,4,2560]
                    |
       final GR read -> hidden [2,2560]
                    |
       gather requested tokens -> unembed -> sample
```

The default layer schedule repeats three GDN layers and one QSA layer.
`ple_layer_ids` selects one-indexed decoder layers.
The table hash uses the zero-based ordinal within that list.
A configured PLE layer must use GDN.
The final GR mixer has no write projection or separate final RMSNorm.

Token-only PLE preparation runs before layer 0.
The current single-queue schedule executes preparation serially.
It does not overlap preparation with layer 0.
The residual-dependent PLE gate and convolution run at the configured layer.

## Checkpoint boundary

| Boundary | Contract |
| --- | --- |
| `Qwen4xModelConfig::normalize` | Validate model relationships and shared CPU reference integer domains |
| `Qwen4xExecutor::new` | Validate Metal geometry and shader integer domains before weight I/O or GPU allocation |
| Metal Main QSA | 24 Q heads, two KV heads, head dimension 256 |
| Metal indexer | Four Q heads, one K head, head dimension 128, budget 2048, compression ratio four |
| Metal MoE | At most 512 experts |
| Rotary geometry | Both use 64 dimensions, default partial factor `0.25`, indexer uses main rotary width |
| Binding resolver | One complete prefix, reject mixed/incomplete trees |
| Loader | Validate dtype/shape/affine layout before recording |
| GR/PLE normalization | Raw BF16 offsets |
| QSA normalization | Convert offsets to effective FP32 scales at load |
| GDN | Existing norm contract, configured SiLU/sigmoid output gate |
| N-gram table | Stream shards into at most eight final resident groups, stable global IDs, one affine layout/group |

The complete table must fit memory.
The n-gram backend selects H1/H4/general lookup without allocating a second complete table.

Executor core does not impose Metal's fixed head geometry or expert limit.
The PLE CPU reference uses `u32` dimensions, residual width, and convolution history length.
Core validation retains these shared limits when PLE is enabled.

```text
checkpoint Q/KV heads = 12/2, head dimension = 256
    -> core: positive dimensions, integral head ratio, valid rotary width
    -> Metal executor: reject unsupported QSA geometry
       weights unopened; GPU resources unallocated

checkpoint Q/KV heads = 24/2, head dimension = 256
    -> same core semantic checks
    -> Metal executor: validate fixed geometry and shader domains
    -> existing binding resolver -> weight loading -> GPU execution
```

## Dense and MoE selection

The checkpoint configuration selects the MLP at initialization.
`num_experts = 0` requires a positive `intermediate_size` and selects dense SwiGLU.
A positive `num_experts` requires complete MoE dimensions.
Main and MTP use the same existing MLP kernels and shape policies.
The loader allocates only the selected scratch type.
Parameter counts such as 27B or 35B do not select a graph.
Metal's QSA geometry limits still apply.

## Native MTP

`init_qwen4x_model_with_mtp` takes Main and head checkpoint directories plus a positive proposal depth `K`.
The Main configuration must declare one trained QSA head in `text_config.mtp`.
The head uses the final four-channel Main state and the shared token embedding and unembedding.
Dedicated embeddings and intermediate-layer hidden sources are rejected at this boundary.
Main-only initialization does not require a usable MTP head.
The binding resolver checks a complete native head before GPU allocation.
Supported head prefixes are `language_model.mtp.`, `mtp.`, and the unprefixed head layout.
The head uses Main's affine layout.
Optional `mtp_num_experts` and `mtp_num_experts_per_tok` override the head's expert geometry at initialization.
`Qwen4xExecutor::new_with_mtp` also accepts an explicit normalized model configuration, like the Main-only constructor.

```text
Main final channels R [T,C=4,H], contiguous BF16
  | remain live throughout the proposal
  |                               retained hidden tail [module,request,tail,C,H]
  |                                    |
  +------------------+-----------------+--- previous module channels [T',C,H]
                     |
     routes [Tq,2] U32: (source, source token)
     one GPU gather -> previous [Tq,C,H]
                     |
     scatter the previous module's retained tail after the gather
                     |
     RMSNorm over C*H -> normalized [Tq,C,H]
     view [Tq*C,H] -> shared fc_hidden [H,H] -> projected [Tq,C,H]
                                                        |
next token IDs [Tq] -> shared embedding [Tq,H]            |
  RMSNorm over H -> fc_embedding [H,H] -> [Tq,H]         |
  broadcast over C ---------------------------------- add
                                                        v
                                               channels [Tq,C,H]
    GR read -> QSA -> GR write -> GR read -> dense/MoE -> GR write
                                                        |
    final GR read -> [Tq,H] -> gather/unembed -> draft distribution
                                                        |
    channels [Tq,C,H] + sampled draft -------------------+-> next logical module
```

All logical modules reuse one trained head and its scratch.
Each module has independent KV, pooled index keys, and carry history.
The executor retains `K*(K-1)/2` wide hidden rows per request.
These rows repair the boundary after a rejected or replaced draft.
A single-token input is not sufficient to repair every logical module.

For example, `K=3`, Main input starts at `p=9`, and all drafts are rejected:

```text
module   input position   tokens          previous-hidden reads
MTP0     9                y               Main output for anchor
MTP1     8                y,z1            old MTP0 boundary, fresh MTP0 output
MTP2     7                y,z1,z2         old MTP1 boundary, fresh MTP1 outputs

Every module ends at version 10.
Gather old boundary rows before writing the new retained tail.
```

Main commits the accepted GDN/QSA/PLE prefix once before the proposal.
The runtime receives its response after the proposal finishes.
Each logical module runs its native indexer against its own cache.
Internal proposal steps use the existing combined Spec lifecycle.
A sampled draft becomes an input to the next step, so that dependency requires a submit, wait, and read.
Main-only does not allocate proposal resources.

A head replay key includes its logical module, total capacities, topology, and bucketed tail-write capacity.
Submission supplies active token, channel, and tail-write counts separately.
The module identity protects the recorded state-buffer bindings.
Hidden gather and tail scatter use the existing row operators, as Qwen3.5 MTP does.
The route buffers cover the recorded capacities. Inactive scatter work returns before reading a route.

## Pages and state

Runtime core owns allocation, page IDs, block notifications, and prefix-cache lifetime.
`Qwen4xState` interprets each model-owned page span when it receives a batch.
No QSA layout enters the scheduler.

The service uses 2048-token runtime cache blocks and 32 KiB physical pages.
These units differ from the QSA indexer's four-token groups.
For one QSA layer:

```text
runtime KV block: logical token interval [0,2048)
    page IDs supplied by runtime
        |
        +--> main KV IDs [64] -> 64 pages x 32 tokens/page
        |      each token: 2 KV heads x 256 dims x K/V x FP8
        |
        `--> index K IDs [4] -> 4 pages x 128 group keys/page
               each key: 128 dims x BF16
               each group key represents 4 tokens

runtime state block: state at version 2048
    [ existing GDN state pages | packed Qwen4x auxiliary pages ]
                                  |
                                  +--> raw indexer carry, for each QSA layer
                                  +--> PLE convolution history, for each PLE layer
                                  `--> previous token IDs for n-gram hashing
```

Each QSA layer retains up to three raw indexer keys in `[3,128]` BF16 carry.
Those keys have not formed a complete four-token group.
The published pooled key cannot reconstruct them.
Each candidate token writes a distinct carry slot.
The commit selects the accepted candidate and publishes runtime block boundaries.

For default PLE geometry, convolution history has nine normalized inputs per residual channel.
N-gram hashing retains two token IDs per request.
These histories cover Main-only chunking and prefix restoration.
MTP uses the same accepted-prefix Main commit and adds per-module retained carry.

```text
current request state at version p
    | read-only during forward
    +--> QSA candidate carry slots [flat input token]
    +--> PLE normalized inputs [T,C,H]
                |
         commit accepted prefix
                +--> state for each crossed runtime block boundary
                |       -> runtime-owned physical pages
                `--> current request state at accepted version
```

Commit work runs on the existing Metal queue.
The next prepare, reset, cache clear, or snapshot waits for pending commit work.
Prefix restore loads the latest synchronized boundary before the next forward.
Full and selected hibernation save page contents, page tables, auxiliary state, and request metadata.
Reload restores resources before the model records another replay.

The Main owner uses `load_state` and `unload_state` for its GDN resource handles.
`Qwen4xState` uses `allocate_resources` and `release_resources` for its auxiliary buffers, like the GQA and GDN state owners.

## Replay and reusable storage

The executor uses the existing embed, gather/unembed, sampler, and GDN transaction owners.
Main supplies one active-token replay parameter to GR, MLP, PLE, and GDN.
GDN adds its private request counts. QSA supplies its own metadata counts.
Main owns one GR workspace, one shared MLP workspace, and one shared QSA workspace.
Each PLE layer owns preparation outputs because those outputs must survive until that layer runs.

`Qwen4xMainArgs` supplies the Main replay bindings. `replay_arguments` supplies the active counts.
GR read emits a write gate when its loaded weights contain a write projection.
The final mixer has no write projection and emits only the final hidden state.

A replay key contains padded capacities, component topologies, and static buffer layouts.
Submission arguments contain active token and task counts.
QSA query and group capacities use the existing replay bucket policy.
The backend still computes per-query sparse selections.
It routes short queries to dense GQA and long queries to indexed sparse GQA.
Different requests can use different paths in one batch.

QSA metadata buffers retain stable identities across record-time shapes.
`Qwen4xExecutorConfig.max_tokens` bounds the flattened query-token count in one batch.
Initialization allocates shared scratch for every supported shape within this limit.
Dense Map-task capacity stays fixed at initialization. Token replay buckets do not reduce it.
Shared scratch covers the peak SplitKV partial count across supported shapes.
The largest token shape can select S=1 while a smaller shape requires more partial storage.

Two small backend operators support model composition:

```text
repeat_rows:
    input [T,H] -> output [T,C,H], BF16 bit copies
    grid = (ceil(C*H/256), T, 1), 256 threads/threadgroup
    threadgroup (feature tile, token) owns disjoint output features

state_pages:
    jobs [J,3] U64 = (source byte offset, destination byte offset, byte count)
    grid = (J,1,1), 256 threads/threadgroup
    threadgroup j copies one disjoint destination range
    thread tid copies uint lanes tid, tid+256, ...
```

For native MTP, carry history stays in the existing indexer state pool:

```text
indexer state pool, BF16 [slot,3,128]
  current request slots [R]
  candidate token slots [T_capacity]
  retained slots [R,K+1]
      |
prepare: source slot = current, or retained[request,version % (K+1)]
      | no restore copy
forward: one candidate carry per input token
      |
commit: existing batched range copy writes current + retained versions
```

The state owner completes its previous commit before it prepares another batch.
Each owner uses a separate command allocator on the existing queue, as GDN does.
MTP adds one runtime cache lane per logical module.
Each lane supplies both KV/index pages and auxiliary-state pages.
The scheduler continues to treat these page spans as opaque.

Full and selected snapshots save each module's page table, auxiliary state, metadata, and retained carry ranges.
The snapshot writer transfers these buffers directly to files.
The proposal's small hidden/probability workspaces and CPU identity metadata stay resident across same-executor Stop/Start, like the existing MTP path.

## Verification

Configuration and complete-layout tests run on the CPU.
Synthetic first8 tests use nonzero fixed and random affine checkpoints.
An independent eager CPU graph checks final hidden states for mixed requests.
Lifecycle tests cover chunking, decode, prefix restoration, and selected hibernation.
Native-head tests compare fixed/random Dense and MoE outputs with the eager CPU graph.
Combined Spec tests cover depth 1/3/4, two requests, accepted prefixes, replacement inputs, mixed Prefill/Decode, and full/selected Stop/Start.
Depth four checks three active tail writes in a four-row scatter recording.
They recompute the complete Main-to-MTP hidden-state chain on the CPU after each wave.
The HTTP acceptance check uses a small fixed checkpoint with native MTP.
It covers continuous decode, two requests, a 2056-token prompt, prefix reuse, and clean shutdown.
This fixture check does not establish trained-head numerical parity or throughput.
The eager graph preserves BF16 storage boundaries for GDN output, SwiGLU, and attention gating.
Component QSA tests check shared scratch across SIMD, MMA, SplitKV, and S=1 recordings.

Run the serial GPU correctness suite when the GPU is available:

```sh
bash scripts/qwen4x_validate.sh
```

An optional real checkpoint capture uses the production executor:

```sh
bash scripts/qwen4x_validate.sh /path/to/Qwen4x-affine 8 1,2,3,4,5,6,7,8
```

A native MTP capture also exercises the public combined Spec hooks:

```sh
bash scripts/qwen4x_validate.sh /path/to/Qwen4x-affine 8 1,2,3,4,5,6,7,8 /path/to/native-head 3
```

The last three input tokens provide MTP lookahead in this example.
The capture checks finite Main and MTP output. It does not time steady-state performance.

The capture writes final hidden states and checks finite values.
It is not a numerical comparison against oMLX or TensorFold.
Use `all` instead of `8` to capture the full Main graph.
The first8 synthetic tests do not establish real-checkpoint acceptance.
See [follow-up work](future_work.md#qwen4x-integration) for the remaining acceptance and PLE work.
