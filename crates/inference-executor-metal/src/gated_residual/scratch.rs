use inference_backend_metal::components::gated_residual::ReadScratch;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::gated_residual::core::GatedResidualCore;

/// Reusable read intermediates. Both buffers can be reused after read completes.
pub struct GatedResidualScratch {
    normalized_residual: Buffer,
    low_rank: Buffer,
}

impl GatedResidualScratch {
    pub fn new(device: &Device, core: GatedResidualCore, max_tokens: usize) -> Self {
        core.validate();
        assert!(max_tokens > 0);
        let normalized_elements = max_tokens
            .checked_mul(core.residual_dim())
            .expect("GR normalized residual capacity must fit usize");
        let low_rank_elements = max_tokens
            .checked_mul(core.low_rank_dim)
            .expect("GR low-rank capacity must fit usize");
        Self {
            normalized_residual: Buffer::new_zeroed_elements(device, normalized_elements, Dtype::Bfloat16),
            low_rank: Buffer::new_zeroed_elements(device, low_rank_elements, Dtype::Bfloat16),
        }
    }

    pub fn bindings(&self) -> ReadScratch<'_> {
        ReadScratch {
            normalized_residual: &self.normalized_residual,
            low_rank: &self.low_rank,
        }
    }
}
