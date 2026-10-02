use inference_backend_metal::components::qsa_indexer::Shape as QSAIndexerShape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GQACore;

use crate::attn::qsa::sdpa::QSASDPA;
use crate::attn::qsa::sdpa::QSASDPAScratch;
use crate::attn::qsa::sdpa::QSASDPAShape;
use crate::attn::qsa_indexer::backend::QSAIndexer;
use crate::attn::qsa_indexer::scratch::QSAIndexerScratch;

pub struct QSAScratch {
    qgkv: Buffer,
    q: Buffer,
    g: Buffer,
    k: Buffer,
    v: Buffer,
    q_norm_rope: Buffer,
    k_norm_rope: Buffer,
    attention_output: Buffer,
    gated_attention_output: Buffer,
    selected_group_ids: Buffer,
    indexer: QSAIndexerScratch,
    sdpa: QSASDPAScratch,
    sdpa_shape: QSASDPAShape,
    indexer_shape: QSAIndexerShape,
}

#[derive(Clone, Copy)]
pub struct QSAScratchBindings<'a> {
    pub qgkv: &'a Buffer,
    pub q: &'a Buffer,
    pub g: &'a Buffer,
    pub k: &'a Buffer,
    pub v: &'a Buffer,
    pub q_norm_rope: &'a Buffer,
    pub k_norm_rope: &'a Buffer,
    pub attention_output: &'a Buffer,
    pub gated_attention_output: &'a Buffer,
    pub selected_group_ids: &'a Buffer,
    pub indexer: &'a QSAIndexerScratch,
    pub sdpa: &'a QSASDPAScratch,
    pub sdpa_shape: QSASDPAShape,
    pub indexer_shape: QSAIndexerShape,
}

impl QSAScratch {
    pub fn new(
        device: &Device,
        core: &GQACore,
        indexer: &QSAIndexer,
        sdpa: &QSASDPA,
        sdpa_shape: QSASDPAShape,
        indexer_shape: QSAIndexerShape,
    ) -> Self {
        sdpa_shape.validate();
        assert!(indexer_shape.num_total_tokens > 0 && indexer_shape.num_total_groups > 0);
        assert_eq!(sdpa_shape.num_total_tokens, indexer_shape.num_total_tokens);
        let max_tokens = sdpa_shape.num_total_tokens as usize;
        let tensor_elements = |dim: usize| {
            let num_elements = max_tokens
                .checked_mul(dim)
                .expect("QSA scratch tensor element count must fit usize");
            u32::try_from(num_elements).expect("QSA scratch tensor element count must fit shader u32 domain");
            num_elements
        };
        let q_values = tensor_elements(core.q_dim());
        let selected_id_values = tensor_elements(sdpa.top_k() as usize);
        Self {
            qgkv: Buffer::new_zeroed_elements(device, tensor_elements(core.qgkv_dim()), Dtype::Bfloat16),
            q: Buffer::new_zeroed_elements(device, q_values, Dtype::Bfloat16),
            g: Buffer::new_zeroed_elements(device, q_values, Dtype::Bfloat16),
            k: Buffer::new_zeroed_elements(device, tensor_elements(core.k_dim()), Dtype::Bfloat16),
            v: Buffer::new_zeroed_elements(device, tensor_elements(core.v_dim()), Dtype::Bfloat16),
            q_norm_rope: Buffer::new_zeroed_elements(device, q_values, Dtype::Bfloat16),
            k_norm_rope: Buffer::new_zeroed_elements(device, tensor_elements(core.k_dim()), Dtype::Bfloat16),
            attention_output: Buffer::new_zeroed_elements(device, q_values, Dtype::Bfloat16),
            gated_attention_output: Buffer::new_zeroed_elements(device, q_values, Dtype::Bfloat16),
            selected_group_ids: Buffer::new_zeroed_elements(device, selected_id_values, Dtype::Uint32),
            indexer: indexer.new_scratch(device, indexer_shape.num_total_tokens, indexer_shape.num_total_groups),
            sdpa: sdpa.new_scratch(sdpa_shape),
            sdpa_shape,
            indexer_shape,
        }
    }

    pub fn sdpa_shape(&self) -> QSASDPAShape {
        self.sdpa_shape
    }

    pub fn indexer_shape(&self) -> QSAIndexerShape {
        self.indexer_shape
    }

    pub fn bindings(&self) -> QSAScratchBindings<'_> {
        QSAScratchBindings {
            qgkv: &self.qgkv,
            q: &self.q,
            g: &self.g,
            k: &self.k,
            v: &self.v,
            q_norm_rope: &self.q_norm_rope,
            k_norm_rope: &self.k_norm_rope,
            attention_output: &self.attention_output,
            gated_attention_output: &self.gated_attention_output,
            selected_group_ids: &self.selected_group_ids,
            indexer: &self.indexer,
            sdpa: &self.sdpa,
            sdpa_shape: self.sdpa_shape,
            indexer_shape: self.indexer_shape,
        }
    }
}
