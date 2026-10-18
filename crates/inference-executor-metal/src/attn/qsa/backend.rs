use inference_backend_metal::components::gqa::activation_gate as backend_activation_gate;
use inference_backend_metal::components::gqa::kv_page_write as backend_kv_page_write;
use inference_backend_metal::components::gqa::qgkv_split as backend_qgkv_split;
use inference_backend_metal::components::qsa_indexer;
use inference_backend_metal::components::rms_norm_rope;
use inference_backend_metal::components::rms_norm_rope::RopeScaling;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::attn::GQACore;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::qsa_indexer::core::QSAIndexerCore;
use inference_executor_core::backend::recorder::Recorder;

use crate::attn::qsa::batch_metadata::QSASDPAMetadataBuffers;
use crate::attn::qsa::scratch::QSAScratch;
use crate::attn::qsa::scratch::QSAScratchBindings;
use crate::attn::qsa::sdpa::QSASDPA;
use crate::attn::qsa::sdpa::QSASDPAActive;
use crate::attn::qsa::sdpa::QSASDPAConfig;
use crate::attn::qsa::sdpa::QSASDPAInput;
use crate::attn::qsa::sdpa::QSASDPAReplay;
use crate::attn::qsa::sdpa::QSASDPARequest;
use crate::attn::qsa::sdpa::QSASDPAShape;
use crate::attn::qsa_indexer::backend::QSAIndexer;
use crate::attn::qsa_indexer::backend::QSAIndexerCacheBindings;
use crate::attn::qsa_indexer::backend::QSAIndexerMetalConfig;
use crate::attn::qsa_indexer::backend::QSAIndexerPrepareInput;
use crate::attn::qsa_indexer::backend::QSAIndexerSelectInput;
use crate::attn::qsa_indexer::backend::QSAIndexerWeights;
use crate::attn::qsa_indexer::batch_metadata::QSAIndexerMetadataBuffers;
use crate::attn::qsa_indexer::batch_metadata::QSAIndexerRequest;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QSAMetalConfig {
    pub group_size: u32,
    pub bits: u32,
    pub page_bytes: u32,
    pub rope_dim: u32,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: RopeScaling,
    pub io_dtype: Dtype,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct QSAReplayTopology {
    pub qgkv_affine: affine_quantized::KernelKind,
    pub indexer_affine: affine_quantized::KernelKind,
    pub output_affine: affine_quantized::KernelKind,
}

impl QSAMetalConfig {
    pub fn validate(self, core: &GQACore) {
        core.validate();
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert!(self.page_bytes > 0);
        let head_dim: u32 = core.head_dim.try_into().expect("QSA head_dim must fit u32");
        assert!(self.rope_dim > 0 && self.rope_dim <= head_dim);
        assert_eq!(self.rope_dim % 2, 0);
        assert!(self.norm_eps.is_finite() && self.norm_eps > 0.0);
        assert!(self.rope_theta.is_finite() && self.rope_theta > 0.0);
        self.rope_scaling.validate();
        assert!(
            !matches!(self.rope_scaling, RopeScaling::Yarn { .. }) || self.rope_theta > 1.0,
            "Yarn rope_theta must be greater than 1"
        );
        match self.io_dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 QSA model boundary is not supported"),
            dtype => panic!("unsupported QSA model boundary dtype {dtype:?}"),
        }
        assert!(kv_page_write_config(core, self).num_tokens_per_page() > 0);
    }
}

#[derive(Clone, Copy)]
pub struct QSAKVCacheBindings<'a> {
    pub pages: &'a Buffer,
    pub page_ids: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct QSAWeights<'a> {
    pub qgkv_weight: &'a Buffer,
    pub qgkv_scales: &'a Buffer,
    pub qgkv_biases: &'a Buffer,
    /// Effective F32 RMSNorm scales, including the checkpoint's `1 + weight` transform.
    pub q_norm_scale: &'a Buffer,
    /// Effective F32 RMSNorm scales, including the checkpoint's `1 + weight` transform.
    pub k_norm_scale: &'a Buffer,
    pub output_weight: &'a Buffer,
    pub output_scales: &'a Buffer,
    pub output_biases: &'a Buffer,
    pub indexer: QSAIndexerWeights<'a>,
}

#[derive(Clone, Copy)]
pub struct QSAInput<'a> {
    pub batch_metadata: &'a QSASDPAMetadataBuffers,
    pub indexer_metadata: &'a QSAIndexerMetadataBuffers,
    pub page_table_index: ReplayU32,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub kv_cache: QSAKVCacheBindings<'a>,
    pub indexer_cache: QSAIndexerCacheBindings<'a>,
    pub weights: QSAWeights<'a>,
    pub scratch: QSAScratchBindings<'a>,
    pub active: QSAReplay,
}

/// Submission counts for the two independently prepared QSA components.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QSAActive {
    pub sdpa: QSASDPAActive,
    pub num_active_score_task_templates: u32,
}

impl QSAActive {
    pub fn fixed_replay(self) -> QSAReplay {
        QSAReplay {
            sdpa: self.sdpa.fixed_replay(),
            num_active_score_task_templates: ReplayU32::Fixed(self.num_active_score_task_templates),
        }
    }

    pub fn add_replay_arguments(self, indexer_shape: qsa_indexer::Shape, top_k: u32, arguments: &mut ReplayArguments) {
        self.sdpa.add_replay_arguments(arguments);
        if indexer_shape.num_total_groups > top_k {
            arguments.set_u32(
                QSA_NUM_ACTIVE_SCORE_TASK_TEMPLATES,
                self.num_active_score_task_templates,
            );
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QSAReplay {
    pub sdpa: QSASDPAReplay,
    pub num_active_score_task_templates: ReplayU32,
}

impl QSAReplay {
    pub fn parameters() -> Self {
        Self {
            sdpa: QSASDPAReplay::parameters(),
            num_active_score_task_templates: ReplayU32::Parameter(QSA_NUM_ACTIVE_SCORE_TASK_TEMPLATES),
        }
    }
}

const QSA_NUM_ACTIVE_SCORE_TASK_TEMPLATES: ReplayParameterKey =
    ReplayParameterKey::new("qsa.num_active_score_task_templates");

pub type QSAOutput<'a> = &'a Buffer;

/// Full QSA attention path.
///
/// ```text
/// hidden [T,H] -> QGKV affine/split
///   Q/G [T,24,256] -> Q: RMSNorm/RoPE -> prepared Q [T,24,256]
///   K/V [T,2,256]  -> K: RMSNorm/RoPE -> caller-owned main KV pages
///   hidden [T,H]   -> indexer prepare/select -> IDs [T,512]
///   prepared Q + main pages + IDs -> dense/sparse SDPA [T,24,256]
///   SDPA + G -> sigmoid gate -> output affine -> hidden [T,H]
/// ```
///
/// The caller owns main pages, indexer pages, carry state, and state-slot lifecycle.
pub struct QSA {
    device: Device,
    core: GQACore,
    qgkv: affine_quantized::Matmul,
    qgkv_to_q_g_k_v: backend_qgkv_split::Compute,
    q_norm_rope: rms_norm_rope::Compute,
    k_norm_rope: rms_norm_rope::Compute,
    kv_page_write: backend_kv_page_write::Compute,
    gate: backend_activation_gate::Compute,
    output: affine_quantized::Matmul,
    indexer: QSAIndexer,
    sdpa: QSASDPA,
}

impl QSA {
    pub fn new(
        device: &Device,
        core: GQACore,
        config: QSAMetalConfig,
        indexer_core: QSAIndexerCore,
        indexer_config: QSAIndexerMetalConfig,
        page_table_layout: GQAPageTableLayout,
    ) -> Self {
        config.validate(&core);
        indexer_core.validate();
        assert_eq!(
            indexer_core.hidden_dim, core.hidden_dim,
            "QSA main and indexer projections must consume the same hidden dimension"
        );
        let qgkv = core.qgkv_shape();
        let output = core.output_shape();
        let indexer = QSAIndexer::new(device, indexer_core, indexer_config);
        let sdpa = QSASDPA::new(
            device,
            QSASDPAConfig {
                num_q_heads: core.num_q_heads.try_into().expect("QSA Q-head count must fit u32"),
                num_kv_heads: core.num_kv_heads.try_into().expect("QSA KV-head count must fit u32"),
                head_dim: core.head_dim.try_into().expect("QSA head_dim must fit u32"),
                scale: core.scale,
                page_bytes: config.page_bytes,
                page_table_layout,
                compress_ratio: indexer_core
                    .compress_ratio
                    .try_into()
                    .expect("QSA indexer compression ratio must fit u32"),
                top_k: indexer_core.top_k.try_into().expect("QSA indexer top-k must fit u32"),
            },
        );
        Self {
            device: device.clone(),
            core: core.clone(),
            qgkv: affine_quantized::Matmul::new(device, affine_config(qgkv.out_dim, qgkv.in_dim, config)),
            qgkv_to_q_g_k_v: backend_qgkv_split::Compute::new(device, qgkv_split_config(&core)),
            q_norm_rope: rms_norm_rope::Compute::new(device, norm_rope_config(&core, config, core.num_q_heads)),
            k_norm_rope: rms_norm_rope::Compute::new(device, norm_rope_config(&core, config, core.num_kv_heads)),
            kv_page_write: backend_kv_page_write::Compute::new(device, kv_page_write_config(&core, config)),
            gate: backend_activation_gate::Compute::new(device, gate_config(&core)),
            output: affine_quantized::Matmul::new(device, affine_config(output.out_dim, output.in_dim, config)),
            indexer,
            sdpa,
        }
    }

    pub fn replay_topology(&self, num_total_tokens: u32) -> QSAReplayTopology {
        QSAReplayTopology {
            qgkv_affine: self.qgkv.topology(num_total_tokens),
            indexer_affine: self.indexer.replay_topology(num_total_tokens),
            output_affine: self.output.topology(num_total_tokens),
        }
    }

    pub fn replay_token_topology_boundaries(&self) -> Box<[u32]> {
        let mut boundaries = self.qgkv.topology_boundaries().into_vec();
        boundaries.extend(self.indexer.replay_token_topology_boundaries());
        boundaries.extend(self.output.topology_boundaries());
        boundaries.sort_unstable();
        boundaries.dedup();
        boundaries.into_boxed_slice()
    }

    pub fn replay_sparse_query_topology_boundaries(&self) -> Box<[u32]> {
        self.sdpa.replay_sparse_query_topology_boundaries()
    }

    pub fn new_scratch(&self, sdpa_shape: QSASDPAShape, indexer_shape: qsa_indexer::Shape) -> QSAScratch {
        QSAScratch::new(
            &self.device,
            &self.core,
            &self.indexer,
            &self.sdpa,
            sdpa_shape,
            indexer_shape,
        )
    }

    /// Updates both attention metadata owners from one request list.
    pub fn update_batch_metadata(
        &self,
        sdpa_metadata: &QSASDPAMetadataBuffers,
        indexer_metadata: &QSAIndexerMetadataBuffers,
        requests: &[QSAIndexerRequest],
        dst_state_slots: &[u32],
        scratch: &QSAScratch,
    ) -> QSAActive {
        let sdpa_requests: Vec<_> = requests
            .iter()
            .map(|request| {
                QSASDPARequest {
                    req_slot: request.req_slot,
                    token_index: request.token_index,
                    num_tokens: request.num_tokens,
                    group_start: request.group_start,
                }
            })
            .collect();
        let active = sdpa_metadata.update(&sdpa_requests);
        let indexer_active = indexer_metadata.update(requests, dst_state_slots, scratch.indexer_shape());
        assert_eq!(active.num_active_tokens, indexer_active.num_active_tokens);
        QSAActive {
            sdpa: active,
            num_active_score_task_templates: indexer_active.num_active_score_task_templates,
        }
    }

    fn validate_input(&self, input: &QSAInput<'_>) {
        let sdpa_shape = input.batch_metadata.shape();
        assert_eq!(sdpa_shape, input.scratch.sdpa_shape);
        let indexer_shape = input.scratch.indexer_shape;
        assert_eq!(sdpa_shape.num_total_tokens, indexer_shape.num_total_tokens);
        if let ReplayU32::Fixed(index) = input.page_table_index {
            assert!(index < self.sdpa.page_table_layout().num_gqa_layers);
        }
    }
}

impl ReplayLayer for QSA {
    type Input<'a> = QSAInput<'a>;
    type Output<'a> = QSAOutput<'a>;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.validate_input(&input);
        let shape = input.batch_metadata.shape();
        let scratch = input.scratch;
        let active_tokens = input.active.sdpa.num_active_tokens;
        let qgkv = self.qgkv.invoke(
            shape.num_total_tokens,
            active_tokens,
            scratch.qgkv,
            0,
            input.hidden_state,
            0,
            input.weights.qgkv_weight,
            0,
            input.weights.qgkv_scales,
            0,
            input.weights.qgkv_biases,
            0,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(qgkv));
        let split = self.qgkv_to_q_g_k_v.invoke(
            backend_qgkv_split::Shape {
                num_total_tokens: shape.num_total_tokens,
            },
            backend_qgkv_split::Buffers {
                qgkv: scratch.qgkv,
                q: scratch.q,
                g: scratch.g,
                k: scratch.k,
                v: scratch.v,
            },
            active_tokens,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(split));
        let q_norm_rope = self.q_norm_rope.invoke(
            rms_norm_rope::Shape {
                num_total_tokens: shape.num_total_tokens,
            },
            rms_norm_rope::Buffers {
                input: scratch.q,
                norm_weight: input.weights.q_norm_scale,
                flat_token_indices: input.batch_metadata.query_positions(),
                output: scratch.q_norm_rope,
            },
            active_tokens,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(q_norm_rope));
        let k_norm_rope = self.k_norm_rope.invoke(
            rms_norm_rope::Shape {
                num_total_tokens: shape.num_total_tokens,
            },
            rms_norm_rope::Buffers {
                input: scratch.k,
                norm_weight: input.weights.k_norm_scale,
                flat_token_indices: input.batch_metadata.query_positions(),
                output: scratch.k_norm_rope,
            },
            active_tokens,
        );
        recorder.record(ReplayOp::opaque(k_norm_rope));
        let page_write = self.kv_page_write.invoke(
            backend_kv_page_write::Shape {
                num_total_token_writes: shape.num_total_tokens,
                page_table_layout: backend_page_table_layout(self.sdpa.page_table_layout()),
            },
            backend_kv_page_write::Buffers {
                pages: input.kv_cache.pages,
                flat_k: scratch.k_norm_rope,
                flat_v: scratch.v,
                req_slots: input.batch_metadata.req_slots(),
                flat_token_indices: input.batch_metadata.query_positions(),
                page_ids: input.kv_cache.page_ids,
            },
            active_tokens,
            input.page_table_index,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(page_write));
        self.indexer.prepare(
            recorder,
            QSAIndexerPrepareInput {
                num_total_tokens: shape.num_total_tokens,
                num_active_tokens: active_tokens,
                hidden_state: input.hidden_state,
                batch_metadata: input.indexer_metadata,
                cache: input.indexer_cache,
                weights: input.weights.indexer,
                scratch: scratch.indexer,
            },
        );
        self.indexer.select(
            recorder,
            QSAIndexerSelectInput {
                num_active_score_task_templates: input.active.num_active_score_task_templates,
                shape: scratch.indexer_shape,
                num_active_tokens: active_tokens,
                batch_metadata: input.indexer_metadata,
                pages: input.indexer_cache.pages,
                page_ids: input.indexer_cache.page_ids,
                scratch: scratch.indexer,
                group_ids: scratch.selected_group_ids,
            },
        );
        self.sdpa.record(
            recorder,
            QSASDPAInput {
                q: scratch.q_norm_rope,
                kv_pages: input.kv_cache.pages,
                page_ids: input.kv_cache.page_ids,
                selected_group_ids: scratch.selected_group_ids,
                output: scratch.attention_output,
                metadata: input.batch_metadata,
                scratch: scratch.sdpa,
                page_table_index: input.page_table_index,
            },
            input.active.sdpa,
        );
        let gate = self.gate.invoke(
            backend_activation_gate::Shape {
                num_total_tokens: shape.num_total_tokens,
            },
            backend_activation_gate::Buffers {
                attention_output: scratch.attention_output,
                g: scratch.g,
                output: scratch.gated_attention_output,
            },
            active_tokens,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(gate));
        let output = self.output.invoke(
            shape.num_total_tokens,
            active_tokens,
            input.next_hidden_state,
            0,
            scratch.gated_attention_output,
            0,
            input.weights.output_weight,
            0,
            input.weights.output_scales,
            0,
            input.weights.output_biases,
            0,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(output));
        input.next_hidden_state
    }
}

fn backend_page_table_layout(layout: GQAPageTableLayout) -> backend_kv_page_write::PageTableLayout {
    backend_kv_page_write::PageTableLayout {
        num_req_slots: layout.num_req_slots,
        num_blocks: layout.num_blocks,
        num_gqa_layers: layout.num_gqa_layers,
        num_page_ids_per_block: layout.num_page_ids_per_block,
    }
}

fn qgkv_split_config(core: &GQACore) -> backend_qgkv_split::Config {
    backend_qgkv_split::Config::bf16(
        core.num_q_heads.try_into().expect("QSA Q-head count must fit u32"),
        core.num_kv_heads.try_into().expect("QSA KV-head count must fit u32"),
        core.head_dim.try_into().expect("QSA head_dim must fit u32"),
    )
}

fn norm_rope_config(core: &GQACore, config: QSAMetalConfig, num_heads: usize) -> rms_norm_rope::Config {
    rms_norm_rope::Config::bf16(
        num_heads.try_into().expect("QSA head count must fit u32"),
        core.head_dim.try_into().expect("QSA head_dim must fit u32"),
        config.rope_dim,
        config.norm_eps,
        config.rope_theta,
    )
    .with_rope_scaling(config.rope_scaling)
    .with_norm_weight_dtype(Dtype::Float32)
}

fn kv_page_write_config(core: &GQACore, config: QSAMetalConfig) -> backend_kv_page_write::Config {
    backend_kv_page_write::Config {
        num_kv_heads: core.num_kv_heads.try_into().expect("QSA KV-head count must fit u32"),
        head_dim: core.head_dim.try_into().expect("QSA head_dim must fit u32"),
        page_bytes: config.page_bytes,
        dtype: config.io_dtype,
    }
}

fn gate_config(core: &GQACore) -> backend_activation_gate::Config {
    backend_activation_gate::Config::bf16(
        core.num_q_heads.try_into().expect("QSA Q-head count must fit u32"),
        core.head_dim.try_into().expect("QSA head_dim must fit u32"),
    )
}

fn affine_config(n: usize, k: usize, config: QSAMetalConfig) -> affine_quantized::Config {
    affine_quantized::Config {
        n: n.try_into().expect("QSA affine n must fit i32"),
        k: k.try_into().expect("QSA affine k must fit i32"),
        group_size: config.group_size.try_into().expect("QSA group_size must fit i32"),
        bits: config.bits.try_into().expect("QSA bits must fit i32"),
        input_dtype: config.io_dtype,
        output_dtype: config.io_dtype,
        scale_bias_dtype: config.io_dtype,
    }
}

#[cfg(test)]
#[path = "backend_test.rs"]
mod tests;
