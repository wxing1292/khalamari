//! Native Qwen4x MTP head. The caller owns request state and proposal steps.
//!
//! ```text
//! previous streams [T,C,H] -> RMSNorm over C*H -> [T*C,H] -> fc_hidden -> channels [T,C,H]
//! token embedding  [T,H]   -> RMSNorm over H   -> [T,H]   -> fc_embedding --+ (broadcast C)
//!                                                                      |
//! channels [T,C,H] <--------------------------------------------------- add
//!      | GR read [T,H] -> QSA -> GR write [T,C,H]
//!      | GR read [T,H] -> Dense/MoE -> GR write [T,C,H]
//!      +-> final GR read -> mixed [T,H] for shared unembedding
//!      +-> retained streams [T,C,H] for the next proposal step
//! ```
use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::gated_residual;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_backend_metal::operators::row_route;
use inference_backend_metal::operators::row_scatter;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v4_x::Qwen4xModelConfig;
use inference_executor_core::model::qwen::v4_x::weight_layout::Qwen4xAttentionWeightBindings;
use inference_executor_core::model::qwen::v4_x::weight_layout::Qwen4xMTPWeightBindings;
use inference_executor_core::replay::ReplayBucketPolicy;

use crate::attn::qsa::backend::QSA;
use crate::attn::qsa::backend::QSAInput;
use crate::attn::qsa::backend::QSAKVCacheBindings;
use crate::attn::qsa::backend::QSAReplay;
use crate::attn::qsa::backend::QSAReplayTopology;
use crate::attn::qsa::scratch::QSAScratch;
use crate::attn::qsa_indexer::backend::QSAIndexerCacheBindings;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::gated_residual::backend::GatedResidual;
use crate::gated_residual::backend::ReadInput;
use crate::gated_residual::backend::WriteInput;
use crate::gated_residual::scratch::GatedResidualScratch;
use crate::model::qwen::v3_x::weight::validate_shape;
use crate::model::qwen::v4_x::component_config::gr_core;
use crate::model::qwen::v4_x::component_config::qsa_configs;
use crate::model::qwen::v4_x::component_config::uniform_layout;
use crate::model::qwen::v4_x::mlp::Qwen4xMLP;
use crate::model::qwen::v4_x::mlp::Qwen4xMLPReplayTopology;
use crate::model::qwen::v4_x::mlp::Qwen4xMLPScratch;
use crate::model::qwen::v4_x::state::Qwen4xState;
use crate::model::qwen::v4_x::weight::AffineWeights;
use crate::model::qwen::v4_x::weight::GRWeights;
use crate::model::qwen::v4_x::weight::QSAWeights;
use crate::model::rms_norm::RMSNorm;
use crate::replay::ReplayComponent;

pub mod plan;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("qwen4x.mtp.num_active_tokens");
const NUM_ACTIVE_CHANNELS: ReplayParameterKey = ReplayParameterKey::new("qwen4x.mtp.num_active_channels");
const NUM_ACTIVE_WRITE_ROWS: ReplayParameterKey = ReplayParameterKey::new("qwen4x.mtp.num_active_write_rows");

pub struct Qwen4xMTP {
    num_channels: u32,
    hidden_norm: RMSNorm,
    token_norm: RMSNorm,
    hidden_projection: affine_quantized::Matmul,
    token_projection: affine_quantized::Matmul,
    projections: Option<(AffineWeights, AffineWeights)>,
    residual: GatedResidual,
    attention_residual: Option<GRWeights>,
    mlp_residual: Option<GRWeights>,
    mixer: Option<GRWeights>,
    attention: QSA,
    attention_weights: Option<QSAWeights>,
    mlp: Qwen4xMLP,
    route: row_route::Kernel,
    scatter: row_scatter::Kernel,
    previous_hidden: Buffer,
    normalized_hidden: Buffer,
    normalized_token: Buffer,
    projected_token: Buffer,
    channels: Buffer,
    mixed: Buffer,
    branch: Buffer,
    gate: Buffer,
    unit_gate: Buffer,
    gr_scratch: GatedResidualScratch,
    qsa_scratch: QSAScratch,
    bucket_policy: ReplayBucketPolicy,
}

pub struct Qwen4xMTPArgs<'a> {
    pub module_index: usize,
    pub previous_hidden: &'a Buffer,
    pub hidden_cache: &'a Buffer,
    pub hidden_routes: &'a Buffer,
    pub write_routes: &'a Buffer,
    pub num_write_rows: u32,
    pub token_embedding: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub pages: &'a Buffer,
    pub state: &'a Qwen4xState,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Qwen4xMTPReplayKey {
    module_index: usize,
    num_total_write_rows: u32,
    sdpa: crate::attn::qsa::sdpa::QSASDPAShape,
    indexer: inference_backend_metal::components::qsa_indexer::Shape,
    attention: QSAReplayTopology,
    hidden_projection: affine_quantized::KernelKind,
    token_projection: affine_quantized::KernelKind,
    mlp: Qwen4xMLPReplayTopology,
}

impl Qwen4xMTP {
    pub fn new(
        device: &Device,
        config: &Qwen4xModelConfig,
        bindings: &Qwen4xMTPWeightBindings,
        max_tokens: u32,
        state: &Qwen4xState,
    ) -> Result<Self, ModelExecutorError> {
        let core = gr_core(config);
        max_tokens
            .checked_mul(core.num_branches as u32)
            .and_then(|n| n.checked_mul(core.hidden_dim as u32))
            .expect("MTP flattened channel indices must fit u32");
        let Qwen4xAttentionWeightBindings::QSA { main, indexer } = &bindings.layer.attention else {
            unreachable!("native MTP uses a QSA layer")
        };
        let (qsa_core, qsa_metal, indexer_core, indexer_metal) = qsa_configs(config, 0, main, indexer)?;
        let attention = QSA::new(
            device,
            qsa_core,
            qsa_metal,
            indexer_core,
            indexer_metal,
            state.page_table_layout(),
        );
        let hidden_projection = affine_quantized::Matmul::new(
            device,
            uniform_layout(config, &[&bindings.hidden.weight], "Qwen4x MTP hidden")?.config(
                core.hidden_dim,
                core.hidden_dim,
                Dtype::Bfloat16,
            ),
        );
        let token_projection = affine_quantized::Matmul::new(
            device,
            uniform_layout(config, &[&bindings.token.weight], "Qwen4x MTP token")?.config(
                core.hidden_dim,
                core.hidden_dim,
                Dtype::Bfloat16,
            ),
        );
        let scratch = Qwen4xMLPScratch::new(device, config, &bindings.layer.mlp, max_tokens as usize)?;
        let mlp = Qwen4xMLP::new(device, config, 0, &bindings.layer.mlp, &scratch)?;
        let mut boundaries = attention.replay_token_topology_boundaries().into_vec();
        boundaries.extend(mlp.replay_topology_boundaries());
        boundaries.extend(token_projection.topology_boundaries());
        boundaries.extend(
            hidden_projection
                .topology_boundaries()
                .iter()
                .map(|&n| n.div_ceil(core.num_branches as u32)),
        );
        let bf16 = |width: usize| {
            Buffer::new_zeroed_elements(
                device,
                (max_tokens as usize)
                    .checked_mul(width)
                    .expect("MTP scratch elements must fit usize"),
                Dtype::Bfloat16,
            )
        };
        Ok(Self {
            num_channels: core.num_branches as u32,
            hidden_norm: RMSNorm::new_with_weight_dtype(device, core.residual_dim(), core.eps, Dtype::Float32),
            token_norm: RMSNorm::new_with_weight_dtype(device, core.hidden_dim, core.eps, Dtype::Float32),
            hidden_projection,
            token_projection,
            projections: None,
            residual: GatedResidual::new(device, core),
            attention_residual: None,
            mlp_residual: None,
            mixer: None,
            qsa_scratch: attention.new_scratch(state.max_sdpa_shape(), state.max_indexer_shape()),
            attention,
            attention_weights: None,
            mlp,
            route: row_route::Kernel::new(
                device,
                row_route::Config {
                    num_cols: core.residual_dim() as u32,
                    dtype: Dtype::Bfloat16,
                },
            ),
            scatter: row_scatter::Kernel::new(
                device,
                row_scatter::Config {
                    num_cols: core.residual_dim() as u32,
                    dtype: Dtype::Bfloat16,
                },
            ),
            previous_hidden: bf16(core.residual_dim()),
            normalized_hidden: bf16(core.residual_dim()),
            normalized_token: bf16(core.hidden_dim),
            projected_token: bf16(core.hidden_dim),
            channels: bf16(core.residual_dim()),
            mixed: bf16(core.hidden_dim),
            branch: bf16(core.hidden_dim),
            gate: bf16(core.num_branches),
            unit_gate: Buffer::from_slice(
                device,
                &vec![half::bf16::ONE.to_bits(); max_tokens as usize * core.num_branches],
            ),
            gr_scratch: GatedResidualScratch::new(device, core, max_tokens as usize),
            bucket_policy: ReplayBucketPolicy::with_topology_boundaries(max_tokens, &boundaries),
        })
    }
    pub fn capacity(&self, num_active_tokens: u32) -> u32 {
        self.bucket_policy.capacity(num_active_tokens)
    }
    pub fn channels(&self) -> &Buffer {
        &self.channels
    }
    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen4xModelConfig,
        bindings: &Qwen4xMTPWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        let core = gr_core(config);
        let norm = |store: &mut SafeTensorStore, name: &str, width: usize| -> Result<Buffer, ModelExecutorError> {
            let tensor = store.tensor_bytes(name, safetensors::Dtype::BF16)?;
            validate_shape(name, tensor.shape(), &[width])?;
            let effective: Vec<f32> = tensor
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::bf16::from_bits(u16::from_ne_bytes(*b)).to_f32() + 1.0)
                .collect();
            Ok(Buffer::from_slice(device, &effective))
        };
        self.hidden_norm
            .load_weights(norm(store, &bindings.hidden_norm, core.residual_dim())?);
        self.token_norm
            .load_weights(norm(store, &bindings.token_norm, core.hidden_dim)?);
        self.projections = Some((
            AffineWeights::load(
                device,
                store,
                &bindings.hidden,
                uniform_layout(config, &[&bindings.hidden.weight], "Qwen4x MTP hidden")?,
                core.hidden_dim,
                core.hidden_dim,
            )?,
            AffineWeights::load(
                device,
                store,
                &bindings.token,
                uniform_layout(config, &[&bindings.token.weight], "Qwen4x MTP token")?,
                core.hidden_dim,
                core.hidden_dim,
            )?,
        ));
        self.attention_residual = Some(GRWeights::load(
            device,
            store,
            config,
            &bindings.layer.attention_residual,
        )?);
        self.mlp_residual = Some(GRWeights::load(device, store, config, &bindings.layer.mlp_residual)?);
        self.mixer = Some(GRWeights::load(device, store, config, &bindings.mixer)?);
        self.mlp.load_weights(device, store, bindings.layer.mlp.clone())?;
        let Qwen4xAttentionWeightBindings::QSA { main, indexer } = &bindings.layer.attention else {
            unreachable!()
        };
        let (core, _, indexer_core, _) = qsa_configs(config, 0, main, indexer)?;
        self.attention_weights = Some(QSAWeights::load(
            device,
            store,
            config,
            &core,
            indexer_core,
            main,
            indexer,
        )?);
        store.unload_all();
        Ok(())
    }
    pub fn unload_weights(&mut self) {
        self.hidden_norm.unload_weights();
        self.token_norm.unload_weights();
        self.projections.take();
        self.attention_residual.take();
        self.mlp_residual.take();
        self.mixer.take();
        self.attention_weights.take();
        self.mlp.unload_weights();
    }
    pub fn replay_arguments(&self, input: &Qwen4xMTPArgs<'_>) -> ReplayArguments {
        let batch = input.state.batch();
        let count = batch.active.sdpa.num_active_tokens;
        let mut args = ReplayArguments::new()
            .with_u32(NUM_ACTIVE_TOKENS, count)
            .with_u32(NUM_ACTIVE_CHANNELS, count * self.num_channels);
        if input.num_write_rows > 0 {
            args = args.with_u32(NUM_ACTIVE_WRITE_ROWS, input.num_write_rows);
        }
        batch
            .active
            .add_replay_arguments(batch.indexer_shape, batch.sdpa.config().top_k, &mut args);
        args
    }
    fn read<'a>(&'a self, recorder: &mut ReplayRecorder, weights: &'a GRWeights, total: u32, output: &'a Buffer) {
        self.residual.read(
            recorder,
            ReadInput {
                num_total_tokens: total,
                num_active_tokens: ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
                residual: &self.channels,
                hidden_state: output,
                scratch: &self.gr_scratch,
                weights: weights.read(),
                write_gate: weights.write.as_ref().map(|weight| {
                    gated_residual::WriteGate {
                        weight,
                        output: &self.gate,
                    }
                }),
            },
        );
    }
    fn write(&self, recorder: &mut ReplayRecorder, total: u32) {
        self.residual.write(
            recorder,
            WriteInput {
                num_total_tokens: total,
                num_active_tokens: ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
                residual: &self.channels,
                gate: &self.gate,
                hidden_state: &self.branch,
                output: &self.channels,
            },
        );
    }
}
impl ReplayComponent for Qwen4xMTP {
    type Key = Qwen4xMTPReplayKey;
    type Input<'a> = Qwen4xMTPArgs<'a>;
    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        let batch = input.state.batch();
        let total = batch.sdpa.shape().num_total_tokens;
        Qwen4xMTPReplayKey {
            module_index: input.module_index,
            num_total_write_rows: self.bucket_policy.capacity_allow_zero(input.num_write_rows),
            sdpa: batch.sdpa.shape(),
            indexer: batch.indexer_shape,
            attention: self.attention.replay_topology(total),
            hidden_projection: self.hidden_projection.topology(total * self.num_channels),
            token_projection: self.token_projection.topology(total),
            mlp: self.mlp.replay_topology(total),
        }
    }
    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let batch = input.state.batch();
        let total = batch.sdpa.shape().num_total_tokens;
        let active = ReplayU32::Parameter(NUM_ACTIVE_TOKENS);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.route.invoke(
            row_route::Shape { num_total_rows: total },
            active,
            row_route::Buffers {
                first_input: input.previous_hidden,
                second_input: &self.channels,
                third_input: input.hidden_cache,
                routes: input.hidden_routes,
                output: &self.previous_hidden,
            },
        )));
        if input.num_write_rows > 0 {
            recorder.record_with_barrier_before(ReplayOp::opaque(self.scatter.invoke(
                row_scatter::Shape {
                    num_total_rows: self.bucket_policy.capacity(input.num_write_rows),
                },
                ReplayU32::Parameter(NUM_ACTIVE_WRITE_ROWS),
                row_scatter::Buffers {
                    input: &self.channels,
                    routes: input.write_routes,
                    output: input.hidden_cache,
                },
            )));
        }
        self.hidden_norm
            .record_with_barrier(recorder, total, active, &self.previous_hidden, &self.normalized_hidden);
        self.token_norm
            .record(recorder, total, active, input.token_embedding, &self.normalized_token);
        let (hidden, token) = self.projections.as_ref().expect("MTP projections must be loaded");
        recorder.record_with_barrier_before(ReplayOp::opaque(self.hidden_projection.invoke(
            total * self.num_channels,
            ReplayU32::Parameter(NUM_ACTIVE_CHANNELS),
            &self.channels,
            0,
            &self.normalized_hidden,
            0,
            &hidden.weight,
            0,
            &hidden.scales,
            0,
            &hidden.biases,
            0,
        )));
        recorder.record(ReplayOp::opaque(self.token_projection.invoke(
            total,
            active,
            &self.projected_token,
            0,
            &self.normalized_token,
            0,
            &token.weight,
            0,
            &token.scales,
            0,
            &token.biases,
            0,
        )));
        self.residual.write(
            recorder,
            WriteInput {
                num_total_tokens: total,
                num_active_tokens: active,
                residual: &self.channels,
                gate: &self.unit_gate,
                hidden_state: &self.projected_token,
                output: &self.channels,
            },
        );
        self.read(
            recorder,
            self.attention_residual.as_ref().expect("MTP GR weights must be loaded"),
            total,
            &self.mixed,
        );
        let mut scratch = self.qsa_scratch.bindings();
        scratch.sdpa_shape = batch.sdpa.shape();
        scratch.indexer_shape = batch.indexer_shape;
        self.attention.record(
            recorder,
            QSAInput {
                batch_metadata: &batch.sdpa,
                indexer_metadata: &batch.indexer,
                page_table_index: ReplayU32::Fixed(0),
                hidden_state: &self.mixed,
                next_hidden_state: &self.branch,
                kv_cache: QSAKVCacheBindings {
                    pages: input.pages,
                    page_ids: input.state.page_ids(),
                },
                indexer_cache: QSAIndexerCacheBindings {
                    pages: input.pages,
                    page_ids: input.state.index_page_ids(0),
                    raw_states: input.state.index_states(0),
                },
                weights: self
                    .attention_weights
                    .as_ref()
                    .expect("MTP QSA weights must be loaded")
                    .bindings(),
                scratch,
                active: QSAReplay::parameters(),
            },
        );
        self.write(recorder, total);
        self.read(
            recorder,
            self.mlp_residual.as_ref().expect("MTP GR weights must be loaded"),
            total,
            &self.mixed,
        );
        self.mlp.record(recorder, &self.mixed, &self.branch, total, active);
        self.write(recorder, total);
        self.read(
            recorder,
            self.mixer.as_ref().expect("MTP mixer weights must be loaded"),
            total,
            input.hidden_output,
        );
    }
}
