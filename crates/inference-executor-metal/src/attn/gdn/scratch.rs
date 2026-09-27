use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GDNCore;

pub struct GDNScratch {
    qkvabz: Buffer,
    conv_qkv: Buffer,
    recurrent_output: Buffer,
    norm_gated_output: Buffer,
}

#[derive(Clone, Copy)]
pub struct GDNScratchBindings<'a> {
    pub qkvabz: &'a Buffer,
    pub conv_qkv: &'a Buffer,
    pub recurrent_output: &'a Buffer,
    pub norm_gated_output: &'a Buffer,
}

impl GDNScratch {
    pub fn new(device: &Device, core: &GDNCore, max_tokens: usize) -> Self {
        core.validate();
        assert!(max_tokens > 0);
        let tensor_elements = |dim: usize| {
            max_tokens
                .checked_mul(dim)
                .expect("GDN scratch tensor element count must fit usize")
        };

        Self {
            qkvabz: Buffer::new_zeroed_elements(device, tensor_elements(core.qkvabz_dim()), Dtype::Bfloat16),
            conv_qkv: Buffer::new_zeroed_elements(device, tensor_elements(core.qkv_dim()), Dtype::Bfloat16),
            recurrent_output: Buffer::new_zeroed_elements(device, tensor_elements(core.v_dim()), Dtype::Bfloat16),
            norm_gated_output: Buffer::new_zeroed_elements(device, tensor_elements(core.v_dim()), Dtype::Bfloat16),
        }
    }

    pub fn bindings(&self) -> GDNScratchBindings<'_> {
        GDNScratchBindings {
            qkvabz: &self.qkvabz,
            conv_qkv: &self.conv_qkv,
            recurrent_output: &self.recurrent_output,
            norm_gated_output: &self.norm_gated_output,
        }
    }
}
