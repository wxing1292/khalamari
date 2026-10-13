use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::UngatedGQACore;

use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::sdpa::Selector;

fn assert_u32_element_index_domain(num_elements: usize, name: &str) {
    assert!(num_elements > 0, "{name} must contain elements");
    assert!(
        u32::try_from(num_elements - 1).is_ok(),
        "{name} exceeds the shader u32 element-index domain: num_elements={num_elements}"
    );
}

pub struct UngatedGQAScratch {
    qkv: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    q_norm_rope: Buffer,
    k_norm_rope: Buffer,
    sdpa_partial_exp_sums: Buffer,
    sdpa_partial_max_logits: Buffer,
    sdpa_partial_output: Buffer,
    attention_output: Buffer,
}

#[derive(Clone, Copy)]
pub struct UngatedGQAScratchBindings<'a> {
    pub qkv: &'a Buffer,
    pub q: &'a Buffer,
    pub k: &'a Buffer,
    pub v: &'a Buffer,
    pub q_norm_rope: &'a Buffer,
    pub k_norm_rope: &'a Buffer,
    pub sdpa_partial_exp_sums: &'a Buffer,
    pub sdpa_partial_max_logits: &'a Buffer,
    pub sdpa_partial_output: &'a Buffer,
    pub attention_output: &'a Buffer,
}

impl UngatedGQAScratch {
    pub fn new(device: &Device, core: &UngatedGQACore, config: GQAMetalConfig, sdpa_selector: &Selector) -> Self {
        core.validate();
        config.validate();
        let max_tokens = sdpa_selector.limits().max_tokens as usize;
        let num_sdpa_partial_state_groups = sdpa_selector.limits().partial_state_group_capacity;
        let num_sdpa_partial_states = num_sdpa_partial_state_groups
            .checked_mul(core.num_q_heads)
            .expect("ungated GQA scratch partial-output statistic count must fit usize");
        let tensor_elements = |dim: usize| {
            let elements = max_tokens
                .checked_mul(dim)
                .expect("ungated GQA scratch tensor element count must fit usize");
            u32::try_from(elements)
                .expect("ungated GQA scratch tensor element count must fit the shader u32 count domain");
            elements
        };
        assert_u32_element_index_domain(num_sdpa_partial_states, "ungated GQA SDPA partial-output statistics");
        let num_sdpa_partial_output_values = num_sdpa_partial_states
            .checked_mul(core.head_dim)
            .expect("ungated GQA SDPA partial output element count must fit usize");
        assert_u32_element_index_domain(num_sdpa_partial_output_values, "ungated GQA SDPA partial output");
        Self {
            qkv: Buffer::new_zeroed_elements(device, tensor_elements(core.qkv_dim()), config.io_dtype),
            q: Buffer::new_zeroed_elements(device, tensor_elements(core.q_dim()), config.io_dtype),
            k: Buffer::new_zeroed_elements(device, tensor_elements(core.k_dim()), config.io_dtype),
            v: Buffer::new_zeroed_elements(device, tensor_elements(core.v_dim()), config.io_dtype),
            q_norm_rope: Buffer::new_zeroed_elements(device, tensor_elements(core.q_dim()), config.io_dtype),
            k_norm_rope: Buffer::new_zeroed_elements(device, tensor_elements(core.k_dim()), config.io_dtype),
            sdpa_partial_exp_sums: Buffer::new_zeroed_elements(device, num_sdpa_partial_states, Dtype::Float32),
            sdpa_partial_max_logits: Buffer::new_zeroed_elements(device, num_sdpa_partial_states, Dtype::Float32),
            sdpa_partial_output: Buffer::new_zeroed_elements(device, num_sdpa_partial_output_values, config.io_dtype),
            attention_output: Buffer::new_zeroed_elements(device, tensor_elements(core.q_dim()), config.io_dtype),
        }
    }

    pub fn bindings(&self) -> UngatedGQAScratchBindings<'_> {
        UngatedGQAScratchBindings {
            qkv: &self.qkv,
            q: &self.q,
            k: &self.k,
            v: &self.v,
            q_norm_rope: &self.q_norm_rope,
            k_norm_rope: &self.k_norm_rope,
            sdpa_partial_exp_sums: &self.sdpa_partial_exp_sums,
            sdpa_partial_max_logits: &self.sdpa_partial_max_logits,
            sdpa_partial_output: &self.sdpa_partial_output,
            attention_output: &self.attention_output,
        }
    }
}
