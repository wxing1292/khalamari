use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GQACore;

use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::sdpa::Selector;

fn assert_u32_element_index_domain(num_elements: usize, name: &str) {
    assert!(num_elements > 0, "{name} must contain elements");
    assert!(
        u32::try_from(num_elements - 1).is_ok(),
        "{name} exceeds the shader u32 element-index domain: num_elements={num_elements}"
    );
}

pub struct GQAScratch {
    qgkv: Buffer,
    q: Buffer,
    g: Buffer,
    k: Buffer,
    v: Buffer,
    q_norm_rope: Buffer,
    k_norm_rope: Buffer,
    sdpa_partial_exp_sums: Buffer,
    sdpa_partial_max_logits: Buffer,
    sdpa_partial_output: Buffer,
    attention_output: Buffer,
    gated_attention_output: Buffer,
}

#[derive(Clone, Copy)]
pub struct GQAScratchBindings<'a> {
    pub qgkv: &'a Buffer,
    pub q: &'a Buffer,
    pub g: &'a Buffer,
    pub k: &'a Buffer,
    pub v: &'a Buffer,
    pub q_norm_rope: &'a Buffer,
    pub k_norm_rope: &'a Buffer,
    pub sdpa_partial_exp_sums: &'a Buffer,
    pub sdpa_partial_max_logits: &'a Buffer,
    pub sdpa_partial_output: &'a Buffer,
    pub attention_output: &'a Buffer,
    pub gated_attention_output: &'a Buffer,
}

impl GQAScratch {
    pub fn new(device: &Device, core: &GQACore, config: GQAMetalConfig, sdpa_selector: &Selector) -> Self {
        core.validate();
        config.validate();
        let max_tokens = sdpa_selector.limits().max_tokens as usize;
        let num_sdpa_partial_state_groups = sdpa_selector.limits().partial_state_group_capacity;
        let num_sdpa_partial_states = num_sdpa_partial_state_groups
            .checked_mul(core.num_q_heads)
            .expect("GQA scratch partial-output statistic count must fit usize");
        let tensor_elements = |dim: usize| {
            let elements = max_tokens
                .checked_mul(dim)
                .expect("GQA scratch tensor element count must fit usize");
            u32::try_from(elements).expect("GQA scratch tensor element count must fit the shader u32 count domain");
            elements
        };
        assert_u32_element_index_domain(num_sdpa_partial_states, "GQA SDPA partial-output statistics");
        let num_sdpa_partial_output_values = num_sdpa_partial_states
            .checked_mul(core.head_dim)
            .expect("GQA SDPA partial output element count must fit usize");
        assert_u32_element_index_domain(num_sdpa_partial_output_values, "GQA SDPA partial output");
        Self {
            qgkv: Buffer::new_zeroed_elements(device, tensor_elements(core.qgkv_dim()), config.io_dtype),
            q: Buffer::new_zeroed_elements(device, tensor_elements(core.q_dim()), config.io_dtype),
            g: Buffer::new_zeroed_elements(device, tensor_elements(core.g_dim()), config.io_dtype),
            k: Buffer::new_zeroed_elements(device, tensor_elements(core.k_dim()), config.io_dtype),
            v: Buffer::new_zeroed_elements(device, tensor_elements(core.v_dim()), config.io_dtype),
            q_norm_rope: Buffer::new_zeroed_elements(device, tensor_elements(core.q_dim()), config.io_dtype),
            k_norm_rope: Buffer::new_zeroed_elements(device, tensor_elements(core.k_dim()), config.io_dtype),
            sdpa_partial_exp_sums: Buffer::new_zeroed_elements(device, num_sdpa_partial_states, Dtype::Float32),
            sdpa_partial_max_logits: Buffer::new_zeroed_elements(device, num_sdpa_partial_states, Dtype::Float32),
            sdpa_partial_output: Buffer::new_zeroed_elements(device, num_sdpa_partial_output_values, config.io_dtype),
            attention_output: Buffer::new_zeroed_elements(device, tensor_elements(core.q_dim()), config.io_dtype),
            gated_attention_output: Buffer::new_zeroed_elements(device, tensor_elements(core.q_dim()), config.io_dtype),
        }
    }

    pub fn bindings(&self) -> GQAScratchBindings<'_> {
        GQAScratchBindings {
            qgkv: &self.qgkv,
            q: &self.q,
            g: &self.g,
            k: &self.k,
            v: &self.v,
            q_norm_rope: &self.q_norm_rope,
            k_norm_rope: &self.k_norm_rope,
            sdpa_partial_exp_sums: &self.sdpa_partial_exp_sums,
            sdpa_partial_max_logits: &self.sdpa_partial_max_logits,
            sdpa_partial_output: &self.sdpa_partial_output,
            attention_output: &self.attention_output,
            gated_attention_output: &self.gated_attention_output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::assert_u32_element_index_domain;

    #[test]
    fn test_gated_and_ungated_activations_retain_token_capacity() {
        use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
        use inference_backend_metal::components::rms_norm_rope::RopeScaling;
        use inference_backend_metal::metal::Device;
        use inference_backend_metal::metal::Dtype;
        use inference_executor_core::attn::GQACore;
        use inference_executor_core::attn::UngatedGQACore;

        use super::GQAScratch;
        use crate::attn::gqa::backend::GQAMetalConfig;
        use crate::attn::gqa::sdpa::Selector;
        use crate::attn::gqa::ungated_scratch::UngatedGQAScratch;

        let device = Device::system_default();
        let config = GQAMetalConfig {
            group_size: 32,
            bits: 8,
            page_bytes: 512,
            rope_dim: 32,
            norm_eps: 1.0e-6,
            rope_theta: 10_000.0,
            rope_scaling: RopeScaling::Default,
            io_dtype: Dtype::Bfloat16,
        };
        let registry = backend_sdpa::Registry::new(backend_sdpa::Config {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: 32,
            tokens_per_page: 8,
        });
        let selector = Selector::new(registry, 4, 32, 32);
        let core = GQACore::new(0, 32, 32, 1, 1, 1.0);
        let gated = GQAScratch::new(&device, &core, config, &selector);
        let ungated_core = UngatedGQACore::new(0, 32, 32, 1, 1, 1.0);
        let ungated = UngatedGQAScratch::new(&device, &ungated_core, config, &selector);
        let gated = gated.bindings();
        let ungated = ungated.bindings();
        assert_eq!(gated.qgkv.len_bytes(), 4 * 128 * 2);
        assert_eq!(ungated.qkv.len_bytes(), 4 * 96 * 2);
        assert_eq!(gated.attention_output.len_bytes(), 4 * 32 * 2);
        assert_eq!(ungated.attention_output.len_bytes(), 4 * 32 * 2);
        assert_eq!(gated.sdpa_partial_exp_sums.len_bytes(), 32 * 4);
        assert_eq!(ungated.sdpa_partial_exp_sums.len_bytes(), 32 * 4);
        assert_eq!(gated.sdpa_partial_output.len_bytes(), 32 * 32 * 2);
        assert_eq!(ungated.sdpa_partial_output.len_bytes(), 32 * 32 * 2);
    }

    #[test]
    fn test_u32_element_index_domain_accepts_two_to_32_elements() {
        assert_u32_element_index_domain(u32::MAX as usize + 1, "test GQA scratch");
    }
}
