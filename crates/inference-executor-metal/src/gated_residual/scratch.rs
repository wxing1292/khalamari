use inference_backend_metal::components::gated_residual::ReadScratch;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::gated_residual::core::GatedResidualCore;

/// Reusable read intermediates. Both buffers can be reused after read completes.
pub struct GatedResidualScratch {
    normalized_residual: Buffer,
    projection: Buffer,
}

impl GatedResidualScratch {
    pub fn new(device: &Device, core: GatedResidualCore, max_tokens: usize) -> Self {
        let config = super::config(core);
        let normalized_elements = max_tokens
            .checked_mul(core.residual_dim())
            .expect("GR normalized residual capacity must fit usize");
        let projection_bytes = config.projection_scratch_bytes(max_tokens);
        Self {
            normalized_residual: Buffer::new_zeroed_elements(device, normalized_elements, Dtype::Bfloat16),
            projection: Buffer::new_zeroed(device, projection_bytes),
        }
    }

    pub fn bindings(&self) -> ReadScratch<'_> {
        ReadScratch {
            normalized_residual: &self.normalized_residual,
            projection: &self.projection,
        }
    }
}
