use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::qsa_indexer::core::QSAIndexerCore;

/// Reusable intermediates. Selection and candidate state live in caller-owned outputs.
pub struct QSAIndexerScratch {
    qk: Buffer,
    q: Buffer,
    scores: Buffer,
}

impl QSAIndexerScratch {
    pub fn new(device: &Device, core: QSAIndexerCore, max_tokens: u32, max_groups: u32) -> Self {
        core.validate();
        assert!(max_tokens > 0 && max_groups > 0);
        let elements = |columns: usize| {
            (max_tokens as usize)
                .checked_mul(columns)
                .expect("QSA indexer scratch capacity must fit usize")
        };
        Self {
            qk: Buffer::new_zeroed_elements(device, elements(core.qk_dim()), Dtype::Bfloat16),
            q: Buffer::new_zeroed_elements(device, elements(core.num_q_heads * core.head_dim), Dtype::Bfloat16),
            scores: Buffer::new_zeroed_elements(device, elements(max_groups as usize), Dtype::Float32),
        }
    }

    pub fn qk(&self) -> &Buffer {
        &self.qk
    }

    pub fn q(&self) -> &Buffer {
        &self.q
    }

    pub fn scores(&self) -> &Buffer {
        &self.scores
    }
}
