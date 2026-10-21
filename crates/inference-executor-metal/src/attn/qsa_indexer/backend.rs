use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::qsa_indexer;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::attn::qsa_indexer::core::QSAIndexerCore;

use crate::attn::qsa_indexer::batch_metadata::QSAIndexerMetadataBuffers;
use crate::attn::qsa_indexer::scratch::QSAIndexerScratch;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QSAIndexerMetalConfig {
    pub group_size: u32,
    pub bits: u32,
    pub io_dtype: Dtype,
}

/// Independent indexer. The caller owns pages, state versions, weights, and scratch.
/// `T` counts query tokens, `Hq` indexer heads, `D` head dimensions, and `G` score columns.
///
/// ```text
/// Projection:
///   hidden [T,H] -- Wqk [(Hq+1)*D,H] --> qk [T,(Hq+1)*D]
///   qk [T,(Hq+1)*D] --+--> raw Q [T,Hq,D] -- norm/RoPE --> Q [T,Hq,D]
///                     `--> raw K [T,D]
///
/// Key preparation:
///   raw K [T,D] + source carry [R,ratio-1,D]
///     |-- mean over ratio -- norm/RoPE --> paged keys [cached_groups,D]
///     `-- incomplete groups ------------> candidate carry [T,ratio-1,D]
///
/// Selection:
///   Q [T,Hq,D] + paged keys [cached_groups,D] --> scores [T,G]
///     --> per-query causal top-k --> group IDs [T,K] for sparse queries
/// ```
/// `H` is the hidden dimension, `R` counts requests, and `K` is the group budget.
/// `ratio` is the compression ratio. Each query has its own visible group range within `G` score columns.
/// State-slot and page-table bindings replace the logical state/cache shapes above.
/// Selected groups are shared across the main attention heads. No main K/V is read here.
/// Preparation updates keys and carry for every active token. Selection writes only
/// queries with more than `K` visible groups. Those rows contain exactly `K` IDs.
/// The caller uses dense attention for the other rows and must not consume their IDs.
pub struct QSAIndexer {
    core: QSAIndexerCore,
    qk: affine_quantized::Matmul,
    compute: qsa_indexer::Compute,
}

#[derive(Clone, Copy)]
pub struct QSAIndexerWeights<'a> {
    pub qk_weight: &'a Buffer,
    pub qk_scales: &'a Buffer,
    pub qk_biases: &'a Buffer,
    pub q_norm: &'a Buffer,
    pub k_norm: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct QSAIndexerCacheBindings<'a> {
    pub pages: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub raw_states: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct QSAIndexerPrepareInput<'a> {
    pub num_total_tokens: u32,
    pub num_active_tokens: ReplayU32,
    pub hidden_state: &'a Buffer,
    pub batch_metadata: &'a QSAIndexerMetadataBuffers,
    pub cache: QSAIndexerCacheBindings<'a>,
    pub weights: QSAIndexerWeights<'a>,
    pub scratch: &'a QSAIndexerScratch,
}

#[derive(Clone, Copy)]
pub struct QSAIndexerSelectInput<'a> {
    pub shape: qsa_indexer::Shape,
    pub num_active_tokens: ReplayU32,
    pub num_active_score_task_templates: ReplayU32,
    pub batch_metadata: &'a QSAIndexerMetadataBuffers,
    pub pages: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub scratch: &'a QSAIndexerScratch,
    pub group_ids: &'a Buffer,
}

impl QSAIndexer {
    pub fn new(device: &Device, core: QSAIndexerCore, config: QSAIndexerMetalConfig) -> Self {
        core.validate();
        match config.io_dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 QSA indexer model boundary is not supported"),
            dtype => panic!("unsupported QSA indexer model boundary dtype {dtype:?}"),
        }
        let compute = qsa_indexer::Compute::new(
            device,
            qsa_indexer::Config {
                num_q_heads: core.num_q_heads.try_into().expect("QSA indexer heads must fit u32"),
                head_dim: core
                    .head_dim
                    .try_into()
                    .expect("QSA indexer head dimension must fit u32"),
                compress_ratio: core
                    .compress_ratio
                    .try_into()
                    .expect("QSA indexer compression ratio must fit u32"),
                top_k: core.top_k.try_into().expect("QSA indexer top-k must fit u32"),
                rope_dim: core
                    .rope_dim
                    .try_into()
                    .expect("QSA indexer rotary dimension must fit u32"),
                rope_theta: core.rope_theta,
                norm_eps: core.norm_eps,
                dtype: config.io_dtype,
            },
        );
        let qk = affine_quantized::Matmul::new(
            device,
            affine_quantized::Config {
                n: core
                    .qk_dim()
                    .try_into()
                    .expect("QSA indexer projection width must fit i32"),
                k: core
                    .hidden_dim
                    .try_into()
                    .expect("QSA indexer hidden dimension must fit i32"),
                group_size: config
                    .group_size
                    .try_into()
                    .expect("QSA indexer quantization group must fit i32"),
                bits: config
                    .bits
                    .try_into()
                    .expect("QSA indexer quantization bits must fit i32"),
                input_dtype: config.io_dtype,
                output_dtype: config.io_dtype,
                scale_bias_dtype: config.io_dtype,
            },
        );
        Self { core, qk, compute }
    }

    pub fn new_scratch(&self, device: &Device, max_tokens: u32, max_groups: u32) -> QSAIndexerScratch {
        QSAIndexerScratch::new(device, self.core, max_tokens, max_groups)
    }

    pub fn replay_token_topology_boundaries(&self) -> Box<[u32]> {
        self.qk.topology_boundaries()
    }

    pub fn replay_topology(&self, num_total_tokens: u32) -> affine_quantized::KernelKind {
        self.qk.topology(num_total_tokens)
    }

    pub fn prepare<'a>(&'a self, recorder: &mut ReplayRecorder, input: QSAIndexerPrepareInput<'a>) {
        recorder.record_with_barrier_before(ReplayOp::opaque(self.qk.invoke(
            input.num_total_tokens,
            input.num_active_tokens,
            input.scratch.qk(),
            0,
            input.hidden_state,
            0,
            input.weights.qk_weight,
            0,
            input.weights.qk_scales,
            0,
            input.weights.qk_biases,
            0,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.compute.prepare(
            input.num_total_tokens,
            input.batch_metadata.layout(),
            input.num_active_tokens,
            qsa_indexer::PrepareBuffers {
                qk: input.scratch.qk(),
                tokens: input.batch_metadata.tokens(),
                q_norm: input.weights.q_norm,
                k_norm: input.weights.k_norm,
                pages: input.cache.pages,
                page_ids: input.cache.page_ids,
                raw_states: input.cache.raw_states,
                q: input.scratch.q(),
            },
        )));
    }

    pub fn select<'a>(&'a self, recorder: &mut ReplayRecorder, input: QSAIndexerSelectInput<'a>) -> &'a Buffer {
        if input.shape.num_total_groups as usize <= self.core.top_k {
            return input.group_ids;
        }
        recorder.record_with_barrier_before(ReplayOp::opaque(self.compute.select(
            input.shape,
            input.batch_metadata.layout(),
            input.num_active_tokens,
            input.num_active_score_task_templates,
            qsa_indexer::SelectBuffers {
                q: input.scratch.q(),
                tokens: input.batch_metadata.tokens(),
                score_task_templates: input.batch_metadata.score_task_templates(),
                pages: input.pages,
                page_ids: input.page_ids,
                group_ids: input.group_ids,
            },
            input.scratch.scores(),
        )));
        input.group_ids
    }
}

#[cfg(test)]
#[path = "backend_test.rs"]
mod tests;
