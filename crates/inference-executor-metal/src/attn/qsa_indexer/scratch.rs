use inference_backend_metal::components::qsa_indexer::Compute;
use inference_backend_metal::components::qsa_indexer::SelectScratch;
use inference_backend_metal::components::qsa_indexer::Shape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::qsa_indexer::core::QSAIndexerCore;

/// Reusable intermediates. Selection and candidate state live in caller-owned outputs.
pub struct QSAIndexerScratch {
    qk: Buffer,
    q: Buffer,
    scores: Buffer,
    candidates: [Buffer; 2],
}

impl QSAIndexerScratch {
    pub fn new(device: &Device, core: QSAIndexerCore, compute: &Compute, shape: Shape) -> Self {
        core.validate();
        assert!(shape.num_total_tokens > 0 && shape.num_total_groups > 0);
        let elements = |columns: usize| {
            (shape.num_total_tokens as usize)
                .checked_mul(columns)
                .expect("QSA indexer scratch capacity must fit usize")
        };
        Self {
            qk: Buffer::new_zeroed_elements(device, elements(core.qk_dim()), Dtype::Bfloat16),
            q: Buffer::new_zeroed_elements(device, elements(core.num_q_heads * core.head_dim), Dtype::Bfloat16),
            scores: Buffer::new_zeroed_elements(device, elements(shape.num_total_groups as usize), Dtype::Float32),
            candidates: std::array::from_fn(|_| Buffer::new_zeroed(device, compute.candidate_bytes(shape))),
        }
    }

    pub fn qk(&self) -> &Buffer {
        &self.qk
    }

    pub fn q(&self) -> &Buffer {
        &self.q
    }

    pub fn selection(&self) -> SelectScratch<'_> {
        SelectScratch {
            scores: &self.scores,
            candidates: [&self.candidates[0], &self.candidates[1]],
        }
    }
}
