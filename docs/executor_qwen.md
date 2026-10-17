# Qwen Executor

This document describes the current Qwen3, Qwen3-ASR, and Qwen3.5/Qwen3.6/Qwen3.8 Metal executors.
The document covers checkpoint configuration, top-down loading, state preparation, cached replay, and sampling.
Qwen3 supports separate Vanilla and fixed-block DSpark modes.
Qwen3.5 supports separate Vanilla, reusable-layer MTP, fixed-block DSpark, and DFlash2 modes.
The `v3_x` directories contain version-neutral leaf components, utilities, and the Qwen3x DSpark and DFlash2 models.
Each model owns its structural contracts and execution graph.

Qwen4x has an independent [Main-only executor](executor_qwen4x.md).
It reuses the generic leaf components and keeps its GR, QSA, and PLE graph in `v4_x`.

## Numerical checkpoints

Set `PSI_QWEN35_CHECK_FINITE=1` before starting the process to enable numerical checkpoints.
The setting is read once. It is disabled by default.
The backend `components/check_finite.rs` checks active F32 or BF16 rows and implements the same `SubmissionCheck`
contract as sampling. It records one non-finite value with its row, column, dtype, and F32 bit pattern.

Qwen3.5 Main and MTP record checks at layer inputs, input normalization, GQA or GDN outputs, attention residuals,
post-attention normalization, MLP outputs, layer outputs, and final normalization.
The shared GatherUnembed owner checks gathered hidden rows and logits.
Main labels include the zero-based layer index. MTP labels refer to its shared trained layer.
GatherUnembed labels refer to the shared Main/MTP output head.

Only active rows are checked. Padding and unused scratch rows are excluded.
Each checkpoint saves its failure on the GPU before later operations can overwrite the buffer.
After successful GPU completion, the CPU reports the first failed checkpoint in recorded model order.
This is the first observed failing boundary. It does not identify an instruction inside a fused kernel, and it does not
inspect persistent cache contents or every intermediate tensor inside GQA, GDN, or MLP.
The check rejects NaN, positive infinity, and negative infinity.

The setting adds GPU scans and synchronization dependencies. It can prevent adjacent residual-add/RMSNorm fusion.
Use this mode for correctness diagnosis, not throughput measurements.
A Metal submission error takes precedence over checkpoint readback. If the GPU cannot complete the submission, the
checkpoint diagnostic is not guaranteed to be available.
See [service diagnostics](service.md#numerical-failure-diagnostics) for a complete capture command.

## Source layout

```text
crates/inference-executor-core/src/model/qwen/v3/
  config.rs                 Qwen3ModelConfig/Qwen3TextConfig, strict parsing, and EOS fallback
  batch.rs                  Qwen3Microbatch, request, response, and sampled-decision contracts
  weight_layout.rs          exact Qwen3 Main/unembed binding tree

crates/inference-executor-core/src/model/qwen/v3_asr/
  config.rs                 strict Qwen3-ASR checkpoint and preprocessor contract
  input.rs                  prepared audio and Audio Tower output-row geometry
  weight_layout.rs          exact Audio Tower and shared Qwen3 text binding tree

crates/inference-executor-core/src/model/qwen/v3_x/
  config.rs                 shared quantization, RoPE, and tensor-path value utilities
  weight_layout.rs          shared GQA/GDN/dense-MLP/MoE leaf binding types and helpers
  dflash2/
    config.rs               official DFlash2 schema adapter and canonical configuration contract
    weight_layout.rs        exact source and affine DFlash2 binding trees
  dspark/
    config.rs               official Qwen3 DSpark configuration contract
    weight_layout.rs        exact source and affine DSpark binding trees

crates/inference-executor-core/src/model/qwen/v3_5/
  config.rs                 Qwen35ModelConfig/Qwen35TextConfig parsing and normalization
  batch.rs                  Qwen35Microbatch, request, response, and sampled-decision contracts
  pending_transactions.rs   Qwen35 sequence-ordered pending transactions
  weight_layout.rs          exact Qwen35 Main/unembed/MTP binding trees

crates/inference-executor-core/src/bin/qwen3x_spec_quantize/
  main.rs                   shared DSpark/DFlash2 converter CLI
  checkpoint.rs             shared safetensors and BF16 affine conversion
  dspark.rs                 DSpark tensor and bit policy
  dflash2.rs                DFlash2 tensor and bit policy

crates/inference-executor-metal/src/
  replay.rs                 generic Replay<T> component/cache owner
  model/
    embedding.rs            shared weight-bearing Embed component
    unembedding.rs          shared weight-bearing Unembed component and QMV/QMM selection
    gather.rs               shared row-gather model component
    main_residual_capture.rs
                            shared Main residual-capture contract
    page_arena.rs           shared physical page buffer
    residual_add.rs         shared residual-add/capture model component
    rms_norm.rs             shared weight-bearing RMS-normalization component
    qwen/v3_x/
      dflash2/
        embed.rs            Main embedding view and anchor-plus-MASK replay
        execution.rs        Spec Prefill and Spec Decode recording, submission, and proposal readback
        load.rs             exact affine checkpoint and resource load
        main_feature.rs     all-row projection from selected Main layers and direct capture owner
        attention.rs        sliding-history plus bidirectional local-block attention
        conv.rs             dynamic grouped-convolution weight owner
        layer.rs            independent Qwen3xDFlash2Layer
        model.rs            Prefill and body replay owners
        output.rs           raw Top-K, candidate lattice, path selection, and sparse output
      dspark/
        embed.rs            Qwen3x DSpark embedding replay
        execution.rs        shared execution resources and per-batch recording
        load.rs             shared checkpoint and resource load
        main_feature.rs     all-row projection from selected Main layers
        attention.rs        paged-history plus bidirectional local-block attention
        layer.rs            independent Qwen3xDSparkLayer
        model.rs            Prefill and body replay owners
        output.rs           gather/unembed and Markov sampling
        sampling.rs         Qwen3x Markov checkpoint weights and generic backend adapter
      layer/
        gqa.rs              shared Qwen3xGQA leaf load and record
        gdn.rs              shared Qwen3xGDN leaf load and record
        dense_mlp.rs        shared Qwen3xDenseMLP leaf load and record
        moe.rs              shared Qwen3xMoE leaf load and record
      state/
        gqa.rs              Qwen3xGQAState page/metadata/reset lifecycle
        gdn.rs              Qwen3xGDNState prepare/restore/commit/publish/reset lifecycle
      weight.rs             shared Qwen checkpoint decoding/validation helpers
    qwen/v3/
      main/
        mod.rs              Qwen3 Main owner and replay key
        text_embed.rs       Qwen3 Main text embedding component and replay key
        gqa.rs              Qwen3 Main ungated GQA weights, state, load, and record
        layer.rs            fixed Qwen3MainLayer and Qwen3MainLayerScratch
        output.rs           Qwen3 gather/unembed component and replay key
        component_config.rs Qwen3 QKV GQA and dense-MLP core/Metal configuration
      executor/
        mod.rs              Qwen3Executor, private pending transactions, and runtime integration
        load.rs             separate Vanilla and DSpark top-down load
        batch.rs            validation, prepare, reset, and commit lifecycle
        recording.rs        recorder lifecycle and Main replay submission
        main.rs             MainTextEmbed, MainResourceEmbed, Main, and GatherUnembed orchestration
        sampling.rs         ordinary or rejection sampling and readback
        dspark.rs           DSpark Spec proposal orchestration
    qwen/v3_asr/
      audio.rs              Audio Tower load, model graph, and execution
      resource.rs           prepared audio registration and async materialization
    qwen/v3_5/
      main/
        mod.rs              Qwen35 Main owner and replay key
        text_embed.rs       Qwen35 Main text embedding component and replay key
        layer.rs            Qwen35MainLayer variants and role-specific scratch
        output.rs           Qwen35 gather/unembed component and replay key
      mtp/
        decode_plan.rs        acceptance-aware Decode token/index plan
        hidden_state_cache.rs per-request BF16 hidden cache and tail metadata
        hidden_state_transfer.rs previous-hidden gather and retained-tail scatter
        mod.rs              supported one-layer Qwen35MTP owner and replay key
        embed.rs            Qwen35MTPEmbed and its replay key
        layer.rs            Qwen35MTPLayer and role-specific scratch
      component_config.rs   Qwen35 component configuration and MTP validation
      executor/
        mod.rs              Qwen35Executor fields and ReplayableDecoderModel integration
        load.rs             layer count pass and separate Vanilla/MTP/DSpark/DFlash2 top-down load
        batch.rs            validation, prepare, reset, and commit lifecycle
        recording.rs        recorder lifecycle and common replay submission
        main.rs             MainTextEmbed, MainResourceEmbed, Main, and GatherUnembed orchestration
        sampling.rs         Main/Spec/rejection orchestration and readback
        mtp.rs              MTP request, proposal-batch, and proposal flow
        dspark.rs           DSpark Spec proposal orchestration
        dflash2.rs          DFlash2 Spec proposal orchestration

crates/inference-executor-metal/src/sampling/
  sampling_params.rs        request-slot SamplingParamsStore shared by all sampling modes
  top_k_sampling.rs         TopKSampling and TopKSamplingOutputBuffers
  top_k_replay.rs           Sampling/DraftSampling replay components
  rejection_replay.rs       generic sparse rejection replay owner
  dspark_markov.rs          DSpark Markov correction and sequential sampling
  spec_probs.rs             SpecProbsStore sparse Spec/Main probability workspace

crates/inference-backend-metal/src/components/
  dynamic_grouped_conv.rs   request-local DFlash2 prepare/finish convolution leaf
  metal/dynamic_grouped_conv.metal
  sampling/dflash2_selector.rs
                            DFlash2 candidate-lattice scoring and probabilistic path walk
  metal/dflash2_selector.metal
```

Runtime core owns scheduling, request lifecycle, physical page allocation/free, and page IDs.
The executor owns model-specific interpretation of page IDs, trained tensors, backend state, replay caches, and
submission order.
Metal kernels remain backend components.

The sharing boundary is below the model execution graph.
Qwen3 and Qwen3.5 each own these model-level objects:

- A model configuration.
- Batch and response contracts.
- A pending-transaction owner.
- A model binding tree.
- Layer and scratch types.
- A plan and configuration builder.
- Main components.
- An executor and recorder.
- Replay keys.

Structural APIs stay in their model directories.

`qwen/v3_x` contains shared leaf components, values, and the reusable Qwen3x DSpark and DFlash2 models.
These leaves include quantization, RoPE, tensor-path values, weight bindings, checkpoint helpers, and GQA/GDN state
owners.
They also include `Qwen3xGQA`, `Qwen3xGDN`, `Qwen3xDenseMLP`, and `Qwen3xMoE`.
The DSpark and DFlash2 models compose compatible generic leaf owners into independent model roles.
Qwen3 has no GDN transaction, state-page metadata, MTP lane, or Qwen3.5 replay key.

## Semantic object tree

```text
Qwen3Executor
  main_gqa_state: Qwen3MainGQAState
  main_text_embed: Replay<Qwen3MainTextEmbed>
  main_resource_embed: Option<MainResourceEmbed>
  main: Replay<Qwen3Main>
  gather_unembed: Replay<Qwen3GatherUnembed>
  sampling: Replay<Sampling>
  speculator: Qwen3Speculator
    Vanilla
    DSpark
      execution: Qwen3xDSparkExecution
      rejection_sampling: Replay<RejectionSampling>
      spec_probs: SpecProbsStore
  pages: PageArena

Qwen3-ASR service composition
  resource_arena: MetalResourceArena
  audio_processor: Qwen3ASRAudioProcessor
    audio_worker: AudioTower
  executor: Qwen3Executor
    main_resource_embed: MainResourceEmbed
      resource_embed: Replay<ResourceEmbed>
    shared Qwen3 Main text decoder

Qwen35Executor
  main_gqa_state: Qwen3xGQAState
  main_gdn_state: Qwen3xGDNState
  main_text_embed: Replay<Qwen35MainTextEmbed>
  main_resource_embed: Option<MainResourceEmbed>
  main: Replay<Qwen35Main>
  gather_unembed: Replay<Qwen35GatherUnembed>
  sampling: Replay<Sampling>
  speculator: Qwen35Speculator
    Vanilla
    MTP
      num_spec_tokens: usize
      gqa_state: Qwen3xGQAState
      embed: Replay<Qwen35MTPEmbed>
      body: Replay<Qwen35MTP>
      sampling: Replay<DraftSampling>
      execution: Qwen35MTPExecution
      common: Qwen35SpeculativeResources
    DSpark
      execution: Qwen3xDSparkExecution
      common: Qwen35SpeculativeResources
    DFlash2
      execution: Qwen3xDFlash2Execution
      common: Qwen35SpeculativeResources
  pages: PageArena

Qwen35SpeculativeResources
  rejection_sampling: Replay<RejectionSampling>
  spec_probs: SpecProbsStore

Qwen3xDSparkExecution
  prefill: Replay<Qwen3xDSparkPrefill>
  gqa_state: BiDiBlockGQAState
  embed: Replay<Qwen3xDSparkEmbed>
  body: Replay<Qwen3xDSparkBody>
  gather_unembed: Replay<Qwen3xDSparkGatherUnembed>
  sampling: Replay<Qwen3xDSparkSampling>
    markov: Qwen3xDSparkMarkov
  decode_input: Replay<SpecDecodeInput>
  reusable hidden/logit workspaces
  page layout and block geometry

Qwen3xDFlash2Execution
  prefill: Replay<Qwen3xDFlash2Prefill>
  gqa_state: BiDiBlockGQAState
  embed: Replay<Qwen3xDFlash2Embed>
  body: Replay<Qwen3xDFlash2Body>
  output: Replay<Qwen3xDFlash2Output>
    Main Unembed view
    raw Top-K merge
    candidate lattice and probabilistic path walk
  decode_input: Replay<SpecDecodeInput>
  reusable hidden, convolution, selector, and attention workspaces
  page layout, query-block geometry, and sliding-window contract

Qwen3ModelOpsRecorder / Qwen35ModelOpsRecorder
  dspark_spec_prefill: Option<Qwen3xDSparkPrefillRecording>
  dspark_spec_decode: Option<Qwen3xDSparkDecodeRecording>
  dflash2_spec_prefill: Option<Qwen3xDFlash2PrefillRecording>
  dflash2_spec_decode: Option<Qwen3xDFlash2DecodeRecording>

Qwen3xDSparkPrefillRecording
  Prefill replay key

Qwen3xDSparkDecodeRecording
  optional Spec Decode prepare replay key and arguments
  embed/body/gather/sampling replay keys
  sampling arguments
  request slots

Qwen3xDFlash2PrefillRecording
  Prefill replay key

Qwen3xDFlash2DecodeRecording
  Spec Decode prepare replay key and arguments
  embed/body/output replay keys
  output arguments
  request slots
```

The shared recorder keeps the Spec Prefill and Spec Decode fields for each fixed-block mode optional. Vanilla and MTP
use `None/None`. A fixed-block Prefill-only batch uses `Some/None`. A fixed-block Spec Decode batch uses `Some/Some`.
This state table does not require a separate enum because the selected outer model mode already owns the valid
transition.

### Mode architecture

Vanilla executes Main embedding, Main, output projection, and target sampling.
MTP adds its separate embedding, reusable physical body layer, and draft sampling owner.
DSpark and DFlash2 remain peer model roles with independent Spec Prefill and Spec Decode recordings.
They share only compatible lower-level BiDiBlockGQA components.

[`dspark_design.md`](dspark_design.md) defines DSpark fixed-block attention, Markov sampling, state, and lifecycle.
[`dflash2_design.md`](dflash2_design.md) defines DFlash2 persistent history, sliding attention, dynamic convolution,
candidate selection, state, and lifecycle.

Semantic components own weights, static configuration, and `load + record`.
Each weight-bearing leaf retains the core and Metal configuration that created its backend.
Weight reload uses this retained contract and does not derive the backend configuration again.
`Replay<T>` owns the related replay cache.
Each executor owns its dynamic workspaces, lifecycle order, and submissions.
Each executor stores one closed speculator enum.
The enum cannot represent a partial resource set or simultaneous MTP, DSpark, and DFlash2 resources.
Vanilla executors do not allocate rejection or speculative-probability resources.
Each initialized executor is one outer model composition: Vanilla, MTP, DSpark, or DFlash2.
The selected composition owns its lifecycle contract.
The common Main implementation does not merge MTP, DSpark, or DFlash2 into one high-level model role.
Per-batch recorders retain optional replay keys because a batch can omit output or Spec stages.
These lifecycle keys do not own initialized model resources.

Embedding, unembedding, row gather, residual add/capture, and RMS normalization remain shared model components.
These components hide backend kernel and invocation details.
Concrete Main and MTP compositions own graph order.

`Embed` and `Unembed` use the same weight-bearing owner spine.
Each owner provides one `ReplayLayer` input with `num_total_*` and `ReplayU32` `num_active_*` fields.
Fixed active values equal the total capacity.
Parameterized active values use the same immutable compute object and loaded weights.
Only `Unembed` exposes replay topology because its affine operator selects QMV or QMM.

`Qwen3Microbatch` records decode requests and an optional speculative suffix.
It identifies the Main rows that require unembedding.
It implements the backend-neutral executor-core `SpecMicrobatch` rejection-input contract.
The Qwen3 executor rejects speculative input when DSpark is disabled.

The model role is also a structural boundary.
`Qwen3MainLayer` owns the fixed-QKV `Qwen3MainGQA` and `Qwen3xDenseMLP` topology for Qwen3 Main.
`Qwen35MainLayer` owns the QGKV-GQA/GDN and dense-MLP/MoE variants for Qwen3.5 Main.
`Qwen35MTPLayer` independently owns the MTP decoder-layer graph.
The MTP embed, layer norms, and final norm load bounded tensor maps for their exact binding subtrees.
Each MTP owner removes its tensors before it creates Metal buffers.
`Qwen3xDSparkLayer` independently owns the DSpark decoder-layer graph.
`Qwen3xDFlash2Layer` independently owns the DFlash2 decoder-layer graph.
These role-specific layers can compose the same leaf components.
They do not share a structural layer type.

`Qwen3xDSparkLayer` composes ungated DSpark GQA, RMSNorm, dense MLP, and residual components.
It does not extend `Qwen3MainLayer`.
It does not add a variant to `Qwen35MTPLayer`.

`Qwen3xDFlash2Layer` composes its own attention, dynamic grouped-convolution, RMSNorm, dense MLP, and residual
components.
It is not a DSpark flag or a `Qwen3xDSparkLayer` variant.

Qwen3-ASR composes a separate Audio Tower with the Qwen3 text decoder.
The CPU audio preprocessor and Audio Tower are model-owned components.
The runtime sees only resources, placement spans, and async materialization tasks.
The executor embeds vocabulary tokens first and then replaces active audio placeholder rows with Audio Tower output.
Text-only Qwen3 retains its original embedding path.
[`qwen3_asr.md`](qwen3_asr.md) defines the complete current Qwen3-ASR contract.

`Qwen3xGQA` and `Qwen3xGDN` store compact per-kind layer indices, not model-layer indices, for page-table and
state-arena addressing.

## Configuration, bindings, and load

`Qwen3ModelConfig` strictly parses the flat Hugging Face Qwen3 schema.
It rejects unsupported GDN, MoE, MTP, sliding-window, and RoPE-scaling variants.
Its EOS token IDs provide a Qwen3 fallback when `generation_config.json` supplies none.
`Qwen3xDSparkConfig` independently validates the repository's flat canonical DSpark schema.
The checkpoint boundary uses an architecture adapter table to convert supported external schemas before canonical
deserialization.
It validates Main compatibility, `target_layer_ids`, fixed-block geometry, ungated GQA, default or Yarn RoPE, and the
`vanilla` Markov head.
`Qwen3xDFlash2Config` independently adapts `DFlash2DraftModel` configuration to the repository's flat canonical
schema.
It validates Main compatibility, sliding-window attention, query-block geometry, dynamic-convolution geometry, and
candidate-selector geometry.
`Qwen3xDFlash2WeightBindings` requires the exact published source manifest or the exact affine manifest.
It does not add optional embedding or unembedding tensors because DFlash2 reuses the Main owners.
`Qwen35ModelConfig` independently parses and normalizes the Qwen3.5/Qwen3.6/Qwen3.8 schema.
That schema includes layer-kind, MoE, MTP, and partial-RoPE fields.
`Qwen3ExecutorConfig` and `Qwen35ExecutorConfig` keep runtime capacities model-specific.

CLI parser structs accept `Option` values for checkpoint arguments.
The service validates these values once and converts them to `Qwen3ModelMode` or `Qwen35ModelMode`.
The normalized service configuration stores only the closed model mode.
The executor public API has one initializer for each supported mode.
The private loader receives one closed init enum.
The constructed executor receives one closed speculator enum.

`Qwen35ExecutorConfig::max_requests` is the executor request-slot capacity.
`Qwen35Config` derives it from the same `--max-requests` value as the scheduler per-batch request capacity.
The service passes that value to `Qwen35ExecutorConfig::max_requests`, `RuntimeConfig::max_running_requests`, and
`SchedulerConfig::max_requests`.
GQA page tables, GDN request state, sampling state, and request-indexed workspaces share this slot domain.
Main, MTP, DSpark, and DFlash2 use this same configuration flow.

Each model resolves its own exact typed binding tree before real tensor reads:

```text
Qwen3ModelWeightBindings
  embed: QuantizedTensorBindings
  main:
    final_norm_weight
    layers: Vec<Qwen3LayerWeightBindings>
      gqa: Qwen3xGQAWeightBindings
      mlp: Qwen3xDenseMLPWeightBindings
  unembed: QuantizedTensorBindings

Qwen3xDSparkWeightBindings
  optional embed: QuantizedTensorBindings
  main_feature:
    hidden_norm_weight
    fc: QuantizedTensorBindings
  layers: Vec<Qwen3xDSparkLayerWeightBindings>
    gqa: Qwen3xGQAWeightBindings
    mlp: Qwen3xDenseMLPWeightBindings
    input_layernorm_weight
    post_attention_layernorm_weight
  final_norm_weight
  optional unembed: QuantizedTensorBindings
  markov: Qwen3xDSparkMarkovWeightBindings
  confidence: Qwen3xDSparkConfidenceWeightBindings

Qwen3xDFlash2WeightBindings
  main_feature:
    hidden_norm_weight
    fc: QuantizedTensorBindings
  layers: Vec<Qwen3xDFlash2LayerWeightBindings>
    attention: independent Q/K/V/output affine bindings
    mlp: independent gate/up/down affine bindings
    input_layernorm_weight
    post_attention_layernorm_weight
    attention_conv: projection plus BF16 base kernel
    mlp_conv: projection plus BF16 base kernel
  final_norm_weight
  selector:
    hidden_projection
    predecessor_codebook
    successor_codebook

Qwen35ModelWeightBindings
  embed: QuantizedTensorBindings
  main:
    final_norm_weight
    layers: Vec<Qwen35LayerWeightBindings>
      attention: Qwen35AttentionWeightBindings::{GQA,GDN}
      mlp: Qwen35MLPWeightBindings::{Dense,MoE}
  unembed: QuantizedTensorBindings

Qwen35MTPWeightBindings
  embed:
    prev_hidden_norm_weight
    token_hidden_norm_weight
    projection: QuantizedTensorBindings
  body: Qwen35LayerWeightBindings
  final_norm_weight
```

Initialization is top-down:

1. Select one closed Vanilla, MTP, DSpark, or DFlash2 init mode.
2. Parse and validate the Main configuration and the selected Spec configuration.
3. Count Main GQA/GDN layers and Dense/MoE scratch requirements.
4. Construct Main state domains and the selected Spec state domain.
5. Construct the model-local layer scratch, component scratch, and token `Embed`.
6. Move each exact binding subtree to its semantic owner. Each owner reads and validates its own real weights.
7. Construct Main and the selected Spec stages.
8. Aggregate all selected Spec resources into one enum variant.
9. Construct `PageArena` and wrap each cached stage in `Replay<T>`.

Qwen3 follows the same ownership order with separate Vanilla and DSpark graphs.
It parses its flat Main configuration and resolves its Main binding tree.
When configured, it parses the DSpark configuration and passes it to the shared DSpark loader.
It constructs one QKV GQA state domain and dense scratch.
It constructs a second ungated GQA state domain and `BiDiBlockGQAScratch` when DSpark is enabled.
It loads Main and Main output for both modes.
The DSpark mode also loads the shared DSpark execution owner and rejection resources.

Qwen3.5 loads Vanilla, its MTP graph, the reusable Qwen3x DSpark graph, or the reusable Qwen3x DFlash2 graph.
The loader cannot receive or construct more than one speculator.
When Main and MTP both use MoE, they reuse one model-owned `MoEScratch`.
The MTP compatibility check requires exact matches for `num_experts_per_tok`, `moe_intermediate_size`, and
`shared_expert_intermediate_size` before it allocates this scratch.
For `shared_expert_intermediate_size`, `0` means that the shared-expert branch is absent.
A positive value means that the branch is present, and both models must specify the same value.
`norm_topk_prob` does not determine scratch geometry and may differ.
If only Main or only MTP uses MoE, the unused model-side MoE geometry does not constrain the scratch allocation.
An incompatible shared geometry returns a recoverable model initialization error.
For DSpark, the loader validates Main hidden width, layer count, vocabulary, position limit, and RoPE values.
It permits the DSpark query projection width to differ from the Main hidden width.
The checkpoint `block_size` is the DSpark proposal count.
The loader does not accept a separate proposal-length override.
The DSpark proposal length is independent of the Main per-request verification budget.
The scheduler may verify only a proposal prefix.
DSpark derives its row capacity as `max_requests * block_size`.
If the checkpoint omits embedding or unembedding weights, DSpark creates a
caller-capacity view that shares the immutable Main kernel and weights.
It adds DSpark context K/V pages to the Main cache lane and retains the Main GDN state domain.

For DFlash2, the loader validates the Main hidden width, selected residual layers, vocabulary, position limit, RoPE,
query-block limit, sliding window, convolution geometry, and selector geometry.
The checkpoint `block_size` is the complete Decode query-block size.
The block has one anchor row and `block_size - 1` MASK proposal rows.
The loader does not accept a separate proposal-length override.
DFlash2 derives its row capacity as `max_requests * block_size`.
It reuses the immutable Main embedding and unembedding owners.
It adds persistent DFlash2 history K/V pages to the Main cache lane.
It retains the Main GDN state domain.

Qwen3 has one runtime cache lane and allocates no GDN state domain.
The executor splits each runtime block between Main K/V and persistent DSpark context K/V.

There is no Main/MTP plan object tree or aggregate component-weight owner.
Qwen3 Main owns QKV GQA and dense-MLP geometry conversion in `qwen/v3/main/component_config.rs`.
Qwen3.5 owns QGKV GQA, GDN, dense-MLP, MoE, and MTP validation in `qwen/v3_5/component_config.rs`.
Qwen3x DSpark has no plan object or plan source file.
Each DSpark semantic owner derives its proposal geometry from `Qwen3xDSparkConfig::block_size` and resolves its affine
layout from the exact binding subtree that it consumes.
Each owner loads a bounded `TensorMap`, removes its tensors, performs its required fusion, and requires an empty map.
Each DSpark layer owns its weight-dependent GQA and dense-MLP backend.
The DSpark state domain shares only page tables, metadata, scratch, and geometry-dependent compute selection.

Qwen3x DFlash2 has no plan object or plan source file.
Each DFlash2 semantic owner derives its geometry from `Qwen3xDFlash2Config` and resolves its affine layout from its
exact binding subtree.
Q, K, V, output, gate, up, and down projections can use independent affine layouts.
The loader does not assume that Wk and Wv have the same dtype or quantization layout.
The loader accepts BF16 or F32 affine scales and biases.
It selects the Metal affine parameter dtype from the checkpoint and requires one dtype for the full DFlash2 model.

## Replay ownership

`Replay<T>::record(runtime, input)` derives the component key and returns immediately on a hit.
On a miss, it records, builds, and inserts exactly once.
It returns `(key, cache_hit)`.
`Replay<T>::replay(key)` is a strict lookup.
It panics if record did not establish the key.
`Replay<T>` exposes `component()` explicitly and does not implement `Deref`.

The independent cached graphs are:

```text
Replay<Qwen3MainTextEmbed>   Qwen3 token embedding
Replay<Qwen3Main>            Qwen3 dense full-attention layers -> final norm
Replay<Qwen3xDSparkPrefill>   all captured rows from selected Main layers -> persistent DSpark context K/V
Replay<Qwen3GatherUnembed>   Qwen3 gather -> unembed
Replay<Sampling>             ordinary Main sampling
Replay<RejectionSampling>    Main sparse distributions -> speculative rejection
Replay<Qwen3xDSparkEmbed>     anchor + MASK block embedding
Replay<Qwen3xDSparkBody>      fixed DSpark layers -> final norm
Replay<Qwen3xDSparkGatherUnembed>
                             request-major hidden -> step-major logits
Replay<Qwen3xDSparkSampling>
                             Markov correction, sampling, and sparse draft storage
Replay<SpecDecodeInput>      Qwen3/Qwen3.5 sparse rejection output -> fixed DSpark Spec Decode input and sampling positions
Replay<Qwen3xDFlash2Prefill>
                             all captured rows from selected Main layers -> persistent DFlash2 history K/V
Replay<Qwen3xDFlash2Embed>    anchor + MASK block embedding through the Main Embed view
Replay<Qwen3xDFlash2Body>     fixed DFlash2 layers -> final norm
Replay<Qwen3xDFlash2Output>   gather MASK rows -> Main Unembed -> selector -> sparse draft storage
Replay<SpecDecodeInput>       sparse rejection output -> fixed DFlash2 Spec Decode input and sampling position

Replay<Qwen35MainTextEmbed>  token embedding
Replay<Qwen35Main>           all Main layers -> final norm
Replay<Qwen35GatherUnembed>  gather -> unembed
Replay<Sampling>             ordinary Main sampling
Replay<Qwen35MTPEmbed>       previous-hidden gather + token embed + input projection
Replay<Qwen35MTP>            one physical GQA body layer -> final norm
Replay<DraftSampling>        draft sampling + sparse draft distribution
Replay<RejectionSampling>    Main sparse distribution + rejection
Replay<GDNStateRestore>
                              snapshot restore into live GDN state
```

MainTextEmbed and MTPEmbed are separate replay boundaries with their own keys.

The shared quantized embedding leaf has one active/total recording API.
`num_total_tokens` defines the recorded grid and buffer extent.
`num_active_tokens` is `ReplayU32::Fixed(num_total_tokens)` or a caller-provided replay parameter.
A replay parameter has the range `1..=num_total_tokens`.
The kernel checks the active token count before it reads `token_ids` or writes the output row.
Qwen3.5 MainTextEmbed reads the configured token-row capacity from `Embed::max_tokens()`.
It owns a base `ReplayBucketPolicy` capped by this capacity.
It records the bucket capacity in `Qwen35MainTextEmbedReplayKey` and never records the active token count in the key.
It uses the stage-owned `qwen3.5.main_text_embed.num_active_tokens` replay parameter for submission.
The executor stores this argument with the prepared key and submits both to the same replay program.
Qwen3 MainTextEmbed uses its stage-owned active-token parameter with identity capacity.
Qwen3.5 MTPEmbed uses a parameter active count as part of its composed replay.

The shared row-gather leaf has one active/total recording API.
`row_gather::Shape::num_total_rows` is the recorded grid extent.
`num_active_rows` is fixed to this extent or supplied at submission.
It validates the row-index and output buffers and dispatches the grid for that capacity.
It binds the caller-provided active-row key with the range `1..=capacity`.
The kernel checks the active row count before it reads an inactive row index or input value and before it writes an
inactive output value.
Qwen3.5 MTPEmbed and GatherUnembed use a parameter active count.
Qwen3 and DSpark GatherUnembed use stage-owned active-row parameters with identity capacity.

The shared unembedding leaf has one active/total recording API.
It uses the caller-provided total row count and `ReplayU32` active row count.
It validates that the total row capacity is in `1..=UnembedConfig::max_tokens`.
It validates the hidden input and logits output ranges against this total row capacity.
The affine replay parameter validates the submitted active row count in `1..=capacity`.
The leaf exposes the affine kernel topology for a row capacity and every row count that changes this topology.
The stage bucket policy must include these topology boundaries.
This rule lets Gather and Unembed use one active-row key without padding across an affine kernel change.
Qwen3.5 GatherUnembed uses a parameter active count.
Qwen3 and DSpark GatherUnembed use parameter active counts with identity capacity.

The shared BF16 row-concat leaf has one active/total recording API.
The API names the recorded row count `num_total_rows` and the active row count `num_active_rows`.
The configuration names the logical input row width `num_columns`.
`num_columns` must be divisible by four.
The three buffer base addresses must be 8-byte aligned.
The kernel copies one `bfloat4` vector per thread.
It validates both input buffers and the output buffer against `num_total_rows`.
It binds the caller-provided active-row key with the range `1..=capacity`.
The kernel checks the active row count before it reads an input value or writes an output value.
The row-concat leaf has one fixed topology and adds no replay bucket boundary.
Qwen3.5 MTPEmbed uses a parameter active count.

The shared RMS-normalization leaf has one active/total recording API.
It dispatches `num_total_tokens` and uses a fixed or parameter active token count.
The RMS-normalization kernel checks the active token count before it reads or writes a row.

Replay recording can fuse a residual add with the immediately following RMS normalization.
Fusion is an optional recorder optimization.
The recorder fuses the operations only when their buffer, dtype, capacity, hidden dimension, and replay parameter
domains match.
An intervening operation or an incompatible RMS normalization disables fusion and does not fail replay construction.
Each dependent RMS normalization records its own consumer barrier.
Fusion preserves a barrier requested by either source operation.
Residual recording and residual capture use `residual_add::RowShape { num_total_rows, num_columns }`.
Both fields count tensor elements, not bytes.
The active row count is fixed or supplied by the caller.
The fused command uses the shared active-token key and token capacity.
It does not declare a separate active-value parameter.
The standalone residual kernel and both fused kernels check the active token count before they read or
write a value or row.
Residual capture is also independently recordable.
Its BF16 vec4 kernel writes the residual output and the selected capture columns only for active rows.
An adjacent compatible RMS normalization can replace it with the fused capture kernel.
Both paths validate the capture destination for the recorded capacity.
RMS normalization and residual/RMS-normalization fusion have fixed token-count topology and add no replay bucket
boundary.
Qwen3.5 MTPEmbed, Main, and MTP use parameter active counts for normalization and residual recording.

Qwen3 MainTextEmbed, Main, and GatherUnembed use the same active/total architecture.
Their current capacity policy is identity.
The total capacity remains in each key, and the active count remains a submission parameter.
Qwen3 Main also submits the active Q-token-range and KV-split counts that its ungated GQA consumes.

Qwen3.5 Main owns one token-capacity replay domain.
The executor selects this capacity before it prepares Main attention metadata.
The stage replay policy uses the shared base bucket ladder.
It also includes every GQA and GDN token-topology boundary and every actual Main layer MLP topology boundary.
The MLP boundary union includes the full-MoE compute-path boundary between four and five tokens.
It also includes dense MLP, router, and optional shared-expert affine and dense boundaries.
The executor token workspace capacity caps this policy.

The executor forces both GQA and GDN metadata to use the selected Main token capacity.
GQA continues to bucket Q-token tiles and SDPA map TaskTemplates independently.
GDN continues to bucket requests independently.
These dimensions use private replay arguments.
All Main token-row commands use the single `qwen3.5.main.num_active_tokens` parameter.
The Main replay does not declare the GQA or GDN component-local active-token parameters.
A sliced-KV Main replay declares three parameters: the Main token count, the GQA TaskTemplate count, and the GDN
request count.
A sliced-Q Main replay also declares the GQA Q-token-tile count.

The production Main replay key records the selected token capacity, all GQA and GDN capacities and categorical
topologies, and the ordered MLP topology for every model layer.
The active token count does not enter this key.
`Qwen35Main::record(...)` receives `num_total_tokens` and `ReplayU32`.
Production `Replay<Qwen35Main>` supplies the stage active-token parameter.

The recorder normally fuses each Main attention residual with the following post-attention RMS normalization.
It also normally fuses each Main MLP residual with the next layer input normalization or the final normalization.
These adjacencies are optimization opportunities, not correctness requirements.
The Main residual-capture path has the same optional fusion opportunity.
Both the standalone and fused capture commands write only active rows.
DSpark or DFlash2 can consume this Main capture.
Each Spec owner retains its own replay keys, arguments, and recording policies.

Qwen3.5 MTPEmbed owns one token-capacity replay domain.
Its replay policy uses the shared base bucket ladder and the input-projection FC topology boundaries.
The policy is capped by the executor token workspace capacity.
The selected capacity and FC topology identify a production replay key.
The active token count does not enter this key.
Production recording uses `Qwen35MTPEmbed::prepare_replay(...)`.
The composed Gather, Embed, both RMS normalizations, BF16 concat, and FC commands use the same
`qwen3.5.mtp_embed.num_active_tokens` parameter.
The composed replay declares exactly one parameter.
The executor stores this argument with the MTPEmbed key and reuses it for every MTP step.

Qwen3.5 MTP owns a separate body token-capacity replay domain.
Its policy uses the shared base bucket ladder.
It also includes every GQA token-topology boundary and the actual physical MTP layer MLP topology boundaries.
The MLP is either dense MLP or full MoE.
The full-MoE boundaries include the token-major/expert-major boundary between four and five tokens.
The executor token workspace capacity caps this policy.

The executor selects the MTP body capacity before it prepares MTP GQA metadata.
It forces the GQA token capacity to the selected stage capacity.
GQA continues to bucket Q-token tiles and SDPA map TaskTemplates independently as private replay dimensions.
All MTP body token-row commands use the single `qwen3.5.mtp.num_active_tokens` parameter.
The body does not declare the component-local `gqa.num_active_tokens` parameter.
Each MTP step also supplies `qwen3.5.mtp.gqa_layer_index`.
A sliced-KV MTP replay declares three parameters: the MTP token count, the GQA TaskTemplate count, and the dynamic
GQA layer index.
A sliced-Q MTP replay also declares the GQA Q-token-tile count.

The production MTP replay key records the selected token capacity, all GQA capacities and categorical topology, and
the categorical dense-MLP or full-MoE topology.
The active token count and logical MTP step index do not enter this key.
`Qwen35MTP::record(...)` receives `num_total_tokens` and `ReplayU32`.
Production `Replay<Qwen35MTP>` supplies the stage active-token parameter.

The recorder normally fuses the attention residual with the post-attention normalization.
It also normally fuses the final MLP residual with the output normalization.
The residual kernels remain correct if either fusion opportunity is unavailable.
Logical MTP steps can use different active token counts and selected capacities.
Each step prepares its own metadata and reuses the program for its selected topology.
MTPEmbed, MTP body, and GatherUnembed retain separate replay parameter domains.

Qwen3.5 GatherUnembed owns one output-row-capacity replay domain.
Its replay policy combines the shared base bucket ladder with every unembed affine topology boundary.
`UnembedConfig::max_tokens` caps this policy.
The loader sets this cap to `min(max_tokens, max_requests * (num_spec_tokens + 1))`.
Vanilla uses `num_spec_tokens = 0`.
Main and MTP output rows, sampling scratch, and sparse target distributions use the same capacity.
The full Prefill token capacity still applies to model body workspaces.
This cap counts output rows, not requests.
One speculative request can produce more than one output row.
`Qwen35GatherUnembed::prepare_replay(...)` maps each nonzero active row count to one recorded capacity.
The production replay key records `num_total_rows` and the categorical unembed topology.
The active row count does not enter this key.
The composed Gather and Unembed commands use the same `qwen3.5.gather_unembed.num_active_rows` parameter.
The composed replay declares exactly one parameter.
Main and MTP use the same replay cache because they bind the same stable buffers.
The recorder stores separate Main and MTP `ReplayArguments` because their active row counts can differ.
An active row count of zero omits GatherUnembed replay.

Qwen3 defines separate replay keys for MainTextEmbed, Main, and GatherUnembed.
Its Main key owns the total token capacity, every layer MLP topology, and the ungated GQA total capacities and topology.
Active token, Q-token-range, and KV-split counts do not enter the key.
It never aliases a Qwen3.5 key or stores an optional GDN key.

DSpark and DFlash2 use identity capacity for Prefill rows, Embed rows, fixed-block token rows, and output rows.
Each replay key still stores a `num_total_*` value, and each submission still supplies the matching `num_active_*`
parameter.
The BiDiBlockGQA body independently pads the history Map TaskTemplate domain to a power-of-two total capacity.
Its active Map count and active Q-token-range count are submission parameters.
DFlash2 also supplies its active query-block count to dynamic grouped convolution.
The DFlash2 body owns this replay parameter key. The convolution leaf binds the caller-owned key.
The DFlash2 output owner similarly supplies one active-request key to all selector commands.
The body and output owners submit these parameters when the recorded total capacity is `1`.
Thus, one body cache entry can reuse compatible history capacity without retaining active work from the recording that
created it.

## Main data flow and workspace ownership

```text
token_ids
  -> MainTextEmbed
token_hidden_input
  -> Main layers using model-local residual_stream[2] ping-pong
hidden_output
  -> GatherUnembed(row_indices)
unembed_hidden
  -> unembed
unembed_logits
  -> Sampling or RejectionSampling
```

`token_hidden_input` is the embedding destination and layer-0 input.
`hidden_output` is the final RMSNorm destination.
The executor owns both `Rc<Buffer>` workspace slots.
Runtime stages pass them without `Option`, hidden handles, or hidden-source enums.
The final layer residual is only the current local ping-pong buffer.
There is no `final_residual` field, accessor, or allocation.

Qwen3 Main constructs `Qwen3MainLayerScratch`.
Qwen3.5 Main constructs `Qwen35MainLayerScratch`.
Qwen3.5 MTP owns separate layer scratch.
All configured logical MTP steps reuse this scratch and the same physical weights.
Similar workspace roles do not imply shared structural ownership.

`Qwen3Main` and `Qwen35Main` accept the shared `MainResidualCapture` boundary.
Main queries the capture owner immediately before each layer's final post-MLP residual add.
The capture owner returns an optional opaque `residual_add::CaptureTarget`.
The destination selects a stable BF16 column range that the capture owner owns.
Each selected Main layer writes directly into its assigned range in one prearranged buffer.
The capture path does not run a concatenate kernel.
Spec Prefill projects the complete captured Main-row prefix.
`None` records the ordinary residual add.

The object-safe capture contract returns only this descriptor.
It never receives a recorder.
Both Main record methods remain generic over `Recorder<Operator = ReplayOp>`.

The Qwen3 and Qwen3.5 loaders supply no capture owner when fixed-block Spec Prefill is disabled.
When DSpark is enabled for Qwen3 or Qwen3.5, `Qwen3xDSparkMainFeatureProjector` owns the capture destinations.
Both executors record `Qwen3xDSparkPrefill` before the combined Main and Spec submission.
When DFlash2 is enabled for Qwen3.5, `Qwen3xDFlash2MainFeatureProjector` owns its separate capture destinations.
Qwen3.5 records `Qwen3xDFlash2Prefill` before the combined Main and Spec submission.
Main depends only on `MainResidualCapture`.

`Qwen35GatherUnembedArgs` has a flat structure.
It binds the final-normalized hidden source, row indices, gather destination, and logits destination.
Gathered hidden and logits remain executor workspaces.

## Batch execution lifecycle

The service calls the model hooks in a fixed order.
One outer model mode selects only its applicable lifecycle.
An empty component input omits that component from its model sequence.
The executor does not store a separate submitted-state flag.

`embed_main` materializes MainTextEmbed and optional MainResourceEmbed work.
`forward_main` materializes Main.
It registers the pending model transaction.
It does not submit backend work.

`unembed_main` materializes GatherUnembed when the batch has sampled rows.
It returns immediately when the batch has no sampled rows.

`sample_main` materializes Sampling or RejectionSampling when the batch has sampled rows.
For Qwen3 and Qwen3.5 DSpark, and for Qwen3.5 DFlash2, it also calls `record_spec` before submission.
Spec Prefill borrows the active token count, request slots, and flat token indices from the current Main GQA metadata.
These buffers contain the complete captured Main-row prefix and remain valid through recording and submission.
`record_spec_prefill` records physical Spec Prefill for every Main row.
When the batch contains Decode requests, `record_decode_prepare` and `record_spec_decode` record the dependent Spec
Decode path.
These methods do not submit backend work or read backend output.

`submit_main` submits one Main sequence.
For a batch with no sampled rows, the sequence is:

```text
MainTextEmbed -> MainResourceEmbed? -> Main
```

For ordinary sampling, the sequence is:

```text
MainTextEmbed -> MainResourceEmbed? -> Main -> GatherUnembed -> Sampling
```

When a speculator is enabled, `sample_main` materializes RejectionSampling for both initial and speculative input.
RejectionSampling supports a ragged `0..N` speculative-token count per request.
For MTP, `submit_main` submits this sequence:

```text
MainTextEmbed -> MainResourceEmbed? -> Main -> GatherUnembed -> RejectionSampling
```

For Qwen3 or Qwen3.5 DSpark, it submits one ordered serial sequence:

```text
MainTextEmbed -> MainResourceEmbed? -> Main -> GatherUnembed -> RejectionSampling
  -> SpecDecodeInput -> DSparkPrefill
  -> DSparkEmbed -> DSpark -> DSparkGatherUnembed -> DSparkSampling
```

For Qwen3.5 DFlash2, it submits one ordered serial sequence:

```text
MainTextEmbed -> MainResourceEmbed? -> Main -> GatherUnembed -> RejectionSampling
  -> SpecDecodeInput -> DFlash2Prefill
  -> DFlash2Embed -> DFlash2 -> DFlash2Output
```

A prefill-only batch omits GatherUnembed, RejectionSampling, SpecDecodeInput, and Spec Decode.
It still appends the applicable Spec Prefill replay after Main.

The service waits once for this combined submission.
`read_main` then reads the sparse rejection result and the final proposal result.
It does not read either result before the combined wait completes.
`main_spec_replay_elapsed` reports the combined Main and Spec submission duration.
It does not report Main-only duration.

Relaxed Metal 4 GPU timestamps are enabled by default and split this sequence at five stages:

```text
Main
  -> RejectionSampling
  -> SpecDecodeInput
  -> Spec Prefill
  -> Spec Decode + proposal sampling
```

The executor maps the ordered stage intervals to `main_gpu_elapsed`, `rejection_gpu_elapsed`,
`spec_prepare_gpu_elapsed`, `spec_prefill_gpu_elapsed`, and `spec_decode_gpu_elapsed`.
It derives total Spec GPU time from the three Spec intervals.
This data does not change `main_spec_replay_elapsed`, `main_cpu_ms`, or `spec_cpu_ms`.
Service telemetry reports `executor_cpu_ms = main_cpu_ms + spec_cpu_ms` for comparisons across the old split
fixed-block Spec lifecycle and the integrated lifecycle. Old and integrated `main_cpu_ms` values do not cover
equivalent work.

Qwen3.5 MTP uses the existing combined `run_spec` lifecycle.
Its `embed_spec`, `forward_spec`, `unembed_spec`, and `sample_spec` hooks materialize MTPEmbed, MTP, GatherUnembed, and
DraftSampling.
MTP does not use the fixed-block Prefill or Decode hooks.

For Qwen3 and Qwen3.5 DSpark, and for Qwen3.5 DFlash2, `run_spec_prefill` and `run_spec_decode` return false because the
fixed-block path is already part of `submit_main`.
The later fixed-block hooks are impossible for these executor paths.

Fixed-block Spec Prefill writes physical Spec history for every Main row, including a rejected Decode suffix.
Runtime commit keeps logical history at the fixed Main rows plus the accepted speculative prefix.
The rejected physical tail is not logically visible.
The next transaction overwrites that tail before a later logical range can expose it.

`SpecDecodeInput` owns the static and GPU-written Decode-input lifecycle.
Its `prepare` method fills its reusable resources for the current batch.
`Qwen3xDSparkExecution` constructs this resource for both Qwen3 and Qwen3.5.
`Qwen3xDFlash2Execution` constructs the same owner for Qwen3.5 DFlash2.
`record_decode_prepare` reads `SparseRejectionSamplingOutput` and writes accepted-dependent anchor positions, anchor
tokens, block token IDs, GQA metadata, and sampling positions.
`record_spec_decode` binds the same physical buffers without an intermediate copy.
Spec Decode keeps its proposal-block K/V in `BiDiBlockGQAScratch`.
History SDPA reads persistent K/V only through the accepted anchor.
Spec Decode does not write trie-backed K/V pages.
Spec Prefill is the only fixed-block stage that writes persistent K/V, and it writes only Main rows whose blocks
runtime already owns.
Therefore, this path does not change scheduler budgets, logical token accounting, service context sizing, or runtime
trie allocation.

For Qwen3.5 MTP, `submit_spec` starts one model-specific Spec transaction.
For `--num-spec-tokens K`, the Qwen3.5 MTP owner executes this dependent sequence K times:

```text
for step_index in 0..K:
    MTPEmbed -> MTP -> GatherUnembed -> DraftSampling
    sampled token becomes the next step's token input
```

The physical weights, scratch, buffer bindings, and replay programs remain the same for every step.
The MTP output overwrites the stable previous-hidden source buffer, so the next replay consumes the prior MTP output
without changing a buffer binding.
The step index is a replay argument for GQA page selection and distribution-row selection.
It is not part of an MTP replay key.

The current implementation waits and reads after each non-final MTP pass.
`submit_spec` returns the final submission, and the service performs the final wait and `read_spec` call.
This preserves one external Spec lifecycle while the MTP owner controls K internal passes.

The complete service order is:

```text
embed_main -> forward_main -> unembed_main -> sample_main
submit_main -> wait -> read_main

if run_spec(model_batch_req, sampled_output):
    embed_spec -> forward_spec -> unembed_spec -> sample_spec
    submit_spec -> wait -> read_spec

# Generic fixed-block interface vocabulary. Current Qwen DSpark and DFlash2 paths do not select these hooks.
if run_spec_prefill(model_batch_req):
    prefill_spec
if run_spec_decode(model_batch_req, sampled_output):
    decode_spec
if separate Spec Prefill or Spec Decode was recorded:
    submit_spec -> wait
if Spec Decode was recorded:
    read_spec
commit
```

Qwen3 and Qwen3.5 DSpark, and Qwen3.5 DFlash2, complete during `submit_main -> wait -> read_main`.
The MTP, DSpark, and DFlash2 branches are mutually exclusive.

## GQA/GDN lifecycle

`Qwen3MainGQAState` groups the Qwen3 ungated backend, scratch, request page table, and metadata buffers.
It accepts only complete Qwen3 Main page-ID blocks. It does not interpret runtime cache lanes.
Qwen3 has zero state pages.
It does not construct, restore, publish, commit, or reset a GDN state table.
`BiDiBlockGQAState` owns a separate DSpark or DFlash2 page table, metadata buffers, backend, and block scratch.
In a fixed-block Spec mode, runtime cache lane 0 stores `[Main page IDs | Spec page IDs]` for each logical block.
`prepare_batch` validates the exact combined length and splits this list once.
It sends each complete role-local list to the applicable state owner.
Qwen3.5 Main and MTP own distinct gated `Qwen3xGQAState` domains.
The Main and selected BiDiBlockGQA state owners expose symmetric role-local state access:

```text
write_page_ids(req_slot, block_index, page_ids)
read_page_ids(req_slot, block_index)
prepare_metadata(req_slots, token_indices, cu_tokens)
reset_req_slots(runtime_notification)
```

The generic `GQARequestPageTable` exposes only per-layer entry access for page IDs.
`Qwen35MTP` keeps its separate runtime lane-to-GQA-layer mapping.

`Qwen3xGDNState` groups a backend, scratch, request state table, metadata, cached restore replay, and one optional
asynchronous state commit and publication.
The current Qwen3.5 executor must own one `Qwen3xGDNState`.
The executor thread prepares it synchronously:

```text
Main GQA write_page_ids
Main GQA prepare_metadata
Main GDN prepare_states(req_slots, block_indices, token_indices, cu_tokens, num_spec_tokens, num_chunkwise_requests, state_page_ids_by_req)
Main GDN prepare_metadata(cu_tokens, prepared_states)
Qwen3.5 MTP page-table prepare
optional GDN restore + wait
```

The shared GDN state leaf receives only these component inputs.
Qwen3.5 extracts them from its model-owned microbatch before it calls the leaf.
`Qwen3xGDNState` does not depend on the Qwen3.5 batch type.

No prepare worker, channel, or receiver exists.
GDN restore refreshes page-I/O staging on every batch.
It records only on a replay miss and waits before Main.
Commit selects verified state versions.
It reconstructs replay states and publishes cache pages in one uncached submission when jobs exist.

MTP keeps the GDN current version aligned with the Main runtime cache frontier.
For one request, Main calculates `num_fixed_tokens = num_total_tokens - num_spec_tokens`.
It commits `input_state_version + num_fixed_tokens + num_accepted_tokens`.
`num_spec_tokens` does not directly adjust this state version.
The next Main call starts at that selected state version. It does not replay verified tail tokens.
A replay transaction permits `S + 1` accepted-prefix choices for `S` speculative tokens. It retains deltas until commit.
DSpark, DFlash2, and MTP use the same unshifted Main state-version domain.
Cache-boundary versions are an independent materialization requirement.
Commit materializes the selected final state and accepted cache boundaries. It does not materialize rejected prefixes.
The next newly sampled token index remains the verified state version.
Runtime can represent a full prompt as `QueryTokens::Decode` when the prompt fits the token budget, so this rule also
applies to that zero-spec warm-up.
`QueryTokens::Prefill` commits its full window.

Runtime core can receive the response while state reconstruction and publication continue.
The next prepare or reset waits before it reuses replay logs, page-I/O staging, or live-state resources.

Whole-request reset enters through `Qwen35Executor::reset_req_slots`.
It fans out to sampling, Main GQA, the selected Spec owner, and Main GDN.
Inner state tables do not infer reset from token indices.
A state version ahead of its token index is a lifecycle invariant violation and panics.

## Supported MTP

MTP reuses one physical GQA body layer for `K` dependent logical steps.
It shares the Main token embedding and owns its input projection, GQA page table, scratch, and proposal loop.
Runtime cache lanes `1..=K` map to the matching logical MTP steps.

Each logical step uses the same composed sequence:

```text
MTPEmbed -> MTP -> GatherUnembed -> DraftSampling
```

Each non-final step waits for its sampled token before the next step starts.
The public Spec lifecycle remains one transaction.

The request-local cache metadata is `Empty`, `Prefill`, or `Decode`.
Each nonempty state stores the pending Main index and `K` cached tail token IDs, not the uncached final draft.
Decode metadata and retained BF16 hidden rows survive turn completion.
The next input must keep the anchor and index. A changed cached continuation prefix repairs MTP from old `x1`.
Main does not replay tokens. Recurrent GDN keeps the same unshifted transaction contract.
Only Decode writes the retained hidden rows. Metadata changes after the full MTP wave completes.

[`mtp_design.md`](mtp_design.md) documents the complete current component contract.

## Supported DSpark

DSpark support is experimental.
The Qwen3 and Qwen3.5 DSpark modes support one fixed-block DSpark checkpoint.
The checkpoint `block_size` defines the fixed proposal count.
`--num-spec-tokens` is an MTP-only service option.
Qwen3.5 MTP, DSpark, and DFlash2 are mutually exclusive.

Qwen3 and Qwen3.5 record Spec Decode prepare, DSpark Prefill, Spec Decode, and proposal sampling before submission.
The GPU Spec Decode prepare stage reads rejection-sampling output directly.
A decode-ready Qwen3 or Qwen3.5 batch uses this single submission:

```text
MainTextEmbed -> MainResourceEmbed? -> Main -> GatherUnembed -> RejectionSampling
  -> SpecDecodeInput -> DSparkPrefill
  -> DSparkEmbed -> DSpark -> DSparkGatherUnembed -> DSparkSampling
```

DSpark input is request-major.
`DSparkGatherUnembed` converts body output to the step-major order required by sequential Markov sampling.
It validates the maximum request-by-step row domain at construction.
DSpark reads stable temperature, top-p, seed, and top-k values from the shared request-slot `SamplingParamsStore`.
Spec Decode prepare writes one first sample position for each request. DSpark step `i` adds `i` to that position.
The draft probability store uses request-slot identity because these rows cross a batch boundary.
Main verification distributions use compact active-row identity because they exist only in one submission.

Main K/V and persistent DSpark context K/V share one runtime cache-block lifecycle.
The executor owns separate page tables and splits each runtime page span.
Proposal-local Q/K/V and attention partials remain in executor-owned `BiDiBlockGQAScratch`.

Qwen3.5 GDN uses the same accepted-prefix replay commit for MTP, DSpark, and DFlash2.
It retains per-forward deltas and materializes only the accepted final state and crossed cache boundaries.
The state-slot capacity does not grow with speculative width. See [`executor_gdn.md`](executor_gdn.md).
The Qwen3.5 service sets the running-slot capacity from `--max-requests` for Main, MTP, DSpark, and DFlash2.
These state buffers remain allocated and reusable while model state is loaded.
`unload_state` writes persistent cache state to SSD and releases its loaded resources.
`clear_replay_cache` and `unload_weights` release replay resources and model weights.

[`dspark_design.md`](dspark_design.md) documents the complete current component contract.

## Supported DFlash2

DFlash2 support is experimental.
Qwen3.5-family executors support the affine Qwen3x DFlash2 checkpoint contract.
The DFlash2 owner is a peer of the DSpark owner.
It is not a mode flag inside DSpark.

The checkpoint `block_size` defines the complete Decode query block.
The Decode block contains one anchor row followed by `block_size - 1` MASK rows.
`--num-spec-tokens` is an MTP-only service option.
Prefill and Decode are independent replay recordings:

```text
MainTextEmbed -> MainResourceEmbed? -> Main -> GatherUnembed -> RejectionSampling
  -> SpecDecodeInput -> DFlash2Prefill
  -> DFlash2Embed -> DFlash2 -> DFlash2Output
```

The executor records both independent recordings before one ordered submission.
Prefill depends on Main capture.
Spec Decode prepare depends on rejection sampling.
The serial sequence emits Spec Decode prepare first, then Prefill, and then the remaining Spec Decode work.
This order makes the acceptance-dependent chain ready first and preserves both dependencies without a CPU boundary.

DFlash2 reads stable temperature and seed values from the same request-slot `SamplingParamsStore` as Main, MTP, and
DSpark. The sequential path walk adds each proposal step to the first sample position from Spec Decode prepare.

Prefill projects every physical Main residual row once.
Logical history contains the fixed Main rows and the accepted speculative prefix.
It does not expose the rejected physical suffix or the newly sampled anchor.
Each DFlash2 layer derives persistent K/V from that projected Main feature and writes its own paged history cache.
The persistent cache stores all history tokens.
Decode limits reads, not writes, with one explicit half-open history range for each query row:

```text
[max(0, query_position + 1 - sliding_window), anchor_position)
```

The anchor and MASK rows form one query tile or part of a query tile.
Each layer computes one SplitKV sliding-history partial and one dense bidirectional block partial.
The shared reducer combines those partials.
The layer then applies the DFlash2 prepare/finish dynamic convolution around attention and MLP.

The output owner gathers only MASK rows.
It uses the shared Main Unembed owner to produce unary logits.
It selects the fixed `selector_top_k` candidate set, builds the DFlash2 edge lattice, walks one request-local
probabilistic path, and writes the exact sparse draft distributions that rejection sampling consumes.

Main K/V and persistent DFlash2 history K/V share one runtime cache-block lifecycle.
The executor owns a separate DFlash2 page table and snapshot file.
Proposal-local Q/K/V, attention partials, convolution coefficients, candidate tensors, and selector outputs remain
ephemeral executor workspaces.

[`dflash2_design.md`](dflash2_design.md) documents the complete current component contract.

## Verification

Unit tests cover:

- Qwen3-ASR checkpoint parsing, audio preparation, Audio Tower execution, resource materialization, and
  text-and-audio composition.
- The strict flat Qwen3 adapter.
- Model-specific Main batch contracts.
- Normalized Qwen3.5 configuration and exact bindings.
- GQA/GDN state and page overwrite/reset.
- GDN transactions and snapshot I/O.
- Generic replay idempotence and strict lookup.
- MTP cache-lane mapping and MTP, DSpark, and DFlash2 sparse rejection.
- DSpark configuration, bindings, attention, Markov sampling, and page splitting.
- DFlash2 configuration adaptation, exact bindings, per-row sliding ranges, convolution, selector, and page splitting.
- Embed, RowGather, and Unembed active-row replay against CPU references.
- Qwen3 and Qwen3.5 GatherUnembed composition for every active row count in one recorded capacity.
- DSpark GatherUnembed request-major to step-major conversion and affine unembedding for every active request count.
- Qwen3.5 MTPEmbed previous-hidden gather, token embedding, both norms, concatenation, and input projection with nonzero
  weights for every active token count.

The Qwen3, Qwen3.5, DSpark, and DFlash2 Embed owners only bind model-local replay keys around the shared `Embed` leaf.
They do not duplicate its numerical test. DFlash2 Output has a candidate-selection contract instead of the
GatherUnembed contract. Its Gather, Unembed, Top-K, codebook Embed, and selector leaves retain independent numerical
coverage.

End-to-end verification exercises Qwen3-ASR transcription through the service endpoint.
End-to-end tests exercise Qwen3 Main-only and Qwen3 DSpark through server/decode.
They also exercise Qwen3.5 Vanilla, MTP, and DSpark modes.
Focused DFlash2 unit tests cover its new model and backend contracts.
The tests inspect generated text.
Performance evidence follows [`executor_benchmarks.md`](executor_benchmarks.md).
Collect that evidence serially.
