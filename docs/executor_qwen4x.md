# Qwen4x executor

Qwen4x has an independent model graph in `model/qwen/v4_x`.
It composes the existing GDN, QSA, gated residual, MoE, and resident n-gram components.
The text service supports Main-only execution.
MTP, external speculators, vision, and SSD table loading are not wired.

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
               -> MoE -> branch [2,2560]
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
| `Qwen4xModelConfig::normalize` | Validate supported architecture once |
| Main QSA | 24 Q heads, two KV heads, head dimension 256 |
| Indexer | Four Q heads, one K head, head dimension 128 |
| Rotary geometry | Both use 64 dimensions, default partial factor `0.25`, indexer uses main rotary width |
| Binding resolver | One complete prefix, reject mixed/incomplete trees |
| Loader | Validate dtype/shape/affine layout before recording |
| GR/PLE normalization | Raw BF16 offsets |
| QSA normalization | Convert offsets to effective FP32 scales at load |
| GDN | Existing norm contract, configured SiLU/sigmoid output gate |
| N-gram table | Stream shards into at most eight final resident groups, stable global IDs, one affine layout/group |

The complete table must fit memory.
The n-gram backend selects H1/H4/general lookup without allocating a second complete table.

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
They do not define an MTP acceptance protocol.

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
Main supplies one active-token replay parameter to GR, MoE, PLE, and GDN.
GDN adds its private request counts. QSA supplies its own metadata counts.
Main owns one GR workspace, one shared MoE workspace, and one shared QSA workspace.
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

## Verification

Configuration and complete-layout tests run on the CPU.
Synthetic first8 tests use nonzero fixed and random affine checkpoints.
An independent eager CPU graph checks final hidden states for mixed requests.
Lifecycle tests cover chunking, decode, prefix restoration, and selected hibernation.
Component QSA tests check shared scratch across SIMD, MMA, SplitKV, and S=1 recordings.

Run the serial GPU correctness suite when the GPU is available:

```sh
bash scripts/qwen4x_validate.sh
```

An optional real checkpoint capture uses the production executor:

```sh
bash scripts/qwen4x_validate.sh /path/to/Qwen4x-affine 8 1,2,3,4,5,6,7,8
```

The capture writes final hidden states and checks finite values.
It is not a numerical comparison against oMLX or TensorFold.
Use `all` instead of `8` to capture the full Main graph.
The first8 synthetic tests do not establish real-checkpoint or service acceptance.
See [follow-up work](future_work.md#qwen4x-integration) for the remaining acceptance and PLE work.
