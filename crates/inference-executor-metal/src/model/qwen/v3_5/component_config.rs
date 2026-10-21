//! Derives Metal component configuration from a Qwen3.5 checkpoint configuration.

use inference_backend_metal::components::rms_norm_rope::RopeScaling;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::GQACore;
use inference_executor_core::attn::gdn::GDNOutputGate;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::mlp::moe::GatedMoECore;
use inference_executor_core::model::qwen::v3_5::LayerType;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::Qwen35TextConfig;
use inference_executor_core::model::qwen::v3_x::QuantizationConfig;

use crate::attn::gdn::backend::GDNMetalConfig;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::def::quantized_affine::QuantizedAffineLayout;
use crate::mlp::dense::backend::DenseMLPMetalConfig;
use crate::mlp::moe::backend::GatedMoEMetalConfig;
use crate::model::qwen::v3_5::QWEN35_PAGE_SIZE_BYTES;
use crate::model::qwen::v3_x::weight::to_u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35MetalDefaults {
    pub group_size: u32,
    pub bits: u32,
    pub hidden_dtype: Dtype,
}

impl Default for Qwen35MetalDefaults {
    fn default() -> Self {
        Self {
            group_size: 64,
            bits: 4,
            hidden_dtype: Dtype::Bfloat16,
        }
    }
}

impl Qwen35MetalDefaults {
    pub fn from_quantization(quantization: Option<&QuantizationConfig>) -> Result<Self, ModelExecutorError> {
        let mut defaults = Self::default();
        if let Some(quantization) = quantization {
            defaults.group_size = to_u32("Qwen3.5 quantization group_size", quantization.group_size)?;
            defaults.bits = to_u32("Qwen3.5 quantization bits", quantization.bits)?;
        }
        Ok(defaults)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35LayerCounts {
    pub gqa: usize,
    pub gdn: usize,
    pub has_dense_mlp: bool,
    pub has_moe: bool,
}

pub fn qwen35_layer_counts(config: &Qwen35ModelConfig) -> Result<Qwen35LayerCounts, ModelExecutorError> {
    let mut counts = Qwen35LayerCounts {
        gqa: 0,
        gdn: 0,
        has_dense_mlp: false,
        has_moe: false,
    };
    for model_layer_index in 0..config.text_config.num_hidden_layers {
        match config.layer_type_at(model_layer_index)? {
            LayerType::GDN => counts.gdn += 1,
            LayerType::FullAttention => counts.gqa += 1,
        }
        if config.layer_uses_moe(model_layer_index) {
            counts.has_moe = true;
        } else {
            counts.has_dense_mlp = true;
        }
    }
    Ok(counts)
}

pub fn derive_qwen35_gdn_configs(
    model_layer_index: usize,
    text: &Qwen35TextConfig,
    defaults: Qwen35MetalDefaults,
) -> Result<(GDNCore, GDNMetalConfig), ModelExecutorError> {
    let core = GDNCore {
        model_layer_index,
        hidden_dim: text.hidden_size,
        num_qk_heads: text.linear_num_key_heads,
        qk_head_dim: text.linear_key_head_dim,
        num_v_heads: text.linear_num_value_heads,
        v_head_dim: text.linear_value_head_dim,
        conv_kernel_size: text.linear_conv_kernel_dim,
        q_scale: (text.linear_key_head_dim as f32).sqrt().recip(),
        output_gate: GDNOutputGate::Silu,
    };
    core.validate();
    let metal = GDNMetalConfig {
        group_size: defaults.group_size,
        bits: defaults.bits,
        norm_eps: text.rms_norm_eps,
        input_dtype: defaults.hidden_dtype,
        output_dtype: defaults.hidden_dtype,
        qkvabz_scale_bias_dtype: Dtype::Bfloat16,
        output_scale_bias_dtype: Dtype::Bfloat16,
    };
    metal.validate();
    Ok((core, metal))
}

pub fn derive_qwen35_gqa_configs(
    model_layer_index: usize,
    text: &Qwen35TextConfig,
    defaults: Qwen35MetalDefaults,
) -> Result<(GQACore, GQAMetalConfig), ModelExecutorError> {
    let core = GQACore::new(
        model_layer_index,
        text.hidden_size,
        text.head_dim,
        text.num_attention_heads,
        text.num_key_value_heads,
        text.scale,
    );
    core.validate();
    let metal = GQAMetalConfig {
        group_size: defaults.group_size,
        bits: defaults.bits,
        page_bytes: to_u32("Qwen3.5 GQA page_bytes", QWEN35_PAGE_SIZE_BYTES)?,
        rope_dim: to_u32("Qwen3.5 GQA rope_dim", text.rope_dim)?,
        norm_eps: text.rms_norm_eps,
        rope_theta: text.rope_theta,
        rope_scaling: RopeScaling::Default,
        io_dtype: defaults.hidden_dtype,
    };
    metal.validate();
    assert!(metal.num_tokens_per_page(&core) > 0);
    Ok((core, metal))
}

pub fn derive_qwen35_dense_mlp_configs(
    model_layer_index: usize,
    text: &Qwen35TextConfig,
    defaults: Qwen35MetalDefaults,
) -> Result<(DenseMLPCore, DenseMLPMetalConfig), ModelExecutorError> {
    if text.intermediate_size == 0 {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3.5 layer {model_layer_index} uses dense MLP but intermediate_size is zero"
        )));
    }
    let core = DenseMLPCore {
        model_layer_index,
        hidden_dim: text.hidden_size,
        intermediate_dim: text.intermediate_size,
    };
    core.validate();
    let metal = DenseMLPMetalConfig {
        gate_up: QuantizedAffineLayout {
            group_size: defaults.group_size,
            bits: defaults.bits,
            scale_bias_dtype: defaults.hidden_dtype,
        },
        down: QuantizedAffineLayout {
            group_size: defaults.group_size,
            bits: defaults.bits,
            scale_bias_dtype: defaults.hidden_dtype,
        },
        io_dtype: defaults.hidden_dtype,
    };
    metal.validate();
    Ok((core, metal))
}

pub fn derive_qwen35_moe_configs(
    layer_prefix: &str,
    model_layer_index: usize,
    config: &Qwen35ModelConfig,
    defaults: Qwen35MetalDefaults,
) -> Result<(GatedMoECore, GatedMoEMetalConfig), ModelExecutorError> {
    let text = &config.text_config;
    let core = GatedMoECore {
        model_layer_index,
        hidden_dim: text.hidden_size,
        intermediate_dim: text.moe_intermediate_size,
        shared_experts_intermediate_dim: (text.shared_expert_intermediate_size > 0)
            .then_some(text.shared_expert_intermediate_size),
        num_experts: text.num_experts,
        num_experts_per_token: text.num_experts_per_tok,
        norm_topk_prob: text.norm_topk_prob,
    };
    core.validate();
    let metal = GatedMoEMetalConfig {
        group_size: defaults.group_size,
        bits: defaults.bits,
        router_bits: quant_bits_for(config, &format!("{layer_prefix}.mlp.gate.weight"), defaults.bits)?,
        shared_expert_gate_bits: quant_bits_for(
            config,
            &format!("{layer_prefix}.mlp.shared_expert_gate.weight"),
            defaults.bits,
        )?,
        io_dtype: defaults.hidden_dtype,
    };
    metal.validate();
    Ok((core, metal))
}

pub fn validate_qwen35_mtp_config(
    main_model_config: &Qwen35ModelConfig,
    mtp_model_config: &Qwen35ModelConfig,
) -> Result<(), ModelExecutorError> {
    let main = &main_model_config.text_config;
    let mtp = &mtp_model_config.text_config;
    if main.hidden_size != mtp.hidden_size
        || main.num_attention_heads != mtp.num_attention_heads
        || main.num_key_value_heads != mtp.num_key_value_heads
        || main.head_dim != mtp.head_dim
        || main.num_experts != mtp.num_experts
    {
        return Err(ModelExecutorError::custom(format!(
            "qwen3.5 MTP config must match main model dimensions: main hidden={} q_heads={} kv_heads={} head_dim={} \
             experts={} mtp hidden={} q_heads={} kv_heads={} head_dim={} experts={}",
            main.hidden_size,
            main.num_attention_heads,
            main.num_key_value_heads,
            main.head_dim,
            main.num_experts,
            mtp.hidden_size,
            mtp.num_attention_heads,
            mtp.num_key_value_heads,
            mtp.head_dim,
            mtp.num_experts
        )));
    }
    let main_uses_dense_mlp =
        (0..main.num_hidden_layers).any(|model_layer_index| !main_model_config.layer_uses_moe(model_layer_index));
    let mtp_uses_dense_mlp = !mtp_model_config.layer_uses_moe(0);
    if main_uses_dense_mlp && mtp_uses_dense_mlp && main.intermediate_size != mtp.intermediate_size {
        return Err(ModelExecutorError::custom(format!(
            "qwen3.5 shared dense MLP scratch requires matching intermediate_size: main={} mtp={}",
            main.intermediate_size, mtp.intermediate_size
        )));
    }
    let main_uses_moe =
        (0..main.num_hidden_layers).any(|model_layer_index| main_model_config.layer_uses_moe(model_layer_index));
    let mtp_uses_moe = mtp_model_config.layer_uses_moe(0);
    if main_uses_moe
        && mtp_uses_moe
        && (main.num_experts_per_tok != mtp.num_experts_per_tok
            || main.moe_intermediate_size != mtp.moe_intermediate_size
            || main.shared_expert_intermediate_size != mtp.shared_expert_intermediate_size)
    {
        return Err(ModelExecutorError::custom(format!(
            "qwen3.5 shared MoE scratch requires matching geometry: main num_experts_per_tok={} \
             moe_intermediate_size={} shared_expert_intermediate_size={} mtp num_experts_per_tok={} \
             moe_intermediate_size={} shared_expert_intermediate_size={}",
            main.num_experts_per_tok,
            main.moe_intermediate_size,
            main.shared_expert_intermediate_size,
            mtp.num_experts_per_tok,
            mtp.moe_intermediate_size,
            mtp.shared_expert_intermediate_size
        )));
    }
    assert_eq!(
        mtp.mtp_num_hidden_layers, 1,
        "qwen3.5 MTP checkpoint must contain exactly one physical body layer"
    );
    if mtp.mtp_use_dedicated_embeddings {
        return Err(ModelExecutorError::custom(
            "qwen3.5 MTP checkpoint must share the Main token embedding",
        ));
    }
    Ok(())
}

fn quant_bits_for(config: &Qwen35ModelConfig, tensor_name: &str, default_bits: u32) -> Result<u32, ModelExecutorError> {
    let bits = config
        .quantization
        .as_ref()
        .map(|quantization| quantization.resolve_for_tensor(tensor_name).bits)
        .unwrap_or(default_bits as usize);
    to_u32("Qwen3.5 quantization bits", bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_dense_scratch_accepts_matching_intermediate_size() {
        let main = model_config(16, 0, 1);
        let mtp = model_config(16, 0, 1);

        assert_eq!(validate_qwen35_mtp_config(&main, &mtp), Ok(()));
    }

    #[test]
    fn shared_dense_scratch_rejects_mismatched_intermediate_size() {
        let main = model_config(16, 0, 1);
        let mtp = model_config(32, 0, 1);

        let error = validate_qwen35_mtp_config(&main, &mtp).unwrap_err();

        assert_eq!(
            error.to_string(),
            "qwen3.5 shared dense MLP scratch requires matching intermediate_size: main=16 mtp=32"
        );
    }

    #[test]
    fn dense_scratch_geometry_does_not_constrain_a_single_dense_user() {
        let main_dense = model_config(16, 8, 2);
        let mtp_moe = model_config(32, 8, 1);
        assert!(!main_dense.layer_uses_moe(0));
        assert!(mtp_moe.layer_uses_moe(0));
        assert_eq!(validate_qwen35_mtp_config(&main_dense, &mtp_moe), Ok(()));

        let main_moe = model_config(16, 8, 1);
        let mtp_dense = model_config(32, 8, 2);
        assert!(main_moe.layer_uses_moe(0));
        assert!(!mtp_dense.layer_uses_moe(0));
        assert_eq!(validate_qwen35_mtp_config(&main_moe, &mtp_dense), Ok(()));
    }

    #[test]
    fn shared_moe_scratch_accepts_matching_geometry_and_non_geometry_difference() {
        let main = model_config(16, 8, 1);
        let mut mtp = model_config(16, 8, 1);
        mtp.text_config.norm_topk_prob = false;

        assert_eq!(validate_qwen35_mtp_config(&main, &mtp), Ok(()));
    }

    #[test]
    fn shared_moe_scratch_rejects_each_mismatched_geometry_field() {
        let base_main = model_config(16, 8, 1);
        let base_mtp = model_config(16, 8, 1);

        let mut topk_mismatch = base_mtp.clone();
        topk_mismatch.text_config.num_experts_per_tok = 3;
        let mut routed_intermediate_mismatch = base_mtp.clone();
        routed_intermediate_mismatch.text_config.moe_intermediate_size = 12;
        let mut shared_intermediate_mismatch = base_mtp.clone();
        shared_intermediate_mismatch.text_config.shared_expert_intermediate_size = 24;
        let mut shared_absent_main = base_main.clone();
        shared_absent_main.text_config.shared_expert_intermediate_size = 0;
        let mut shared_absent_mtp = base_mtp.clone();
        shared_absent_mtp.text_config.shared_expert_intermediate_size = 0;

        let cases = [
            ("num_experts_per_tok", base_main.clone(), topk_mismatch),
            ("moe_intermediate_size", base_main.clone(), routed_intermediate_mismatch),
            (
                "shared_expert_intermediate_size value",
                base_main.clone(),
                shared_intermediate_mismatch,
            ),
            ("shared expert absent in Main", shared_absent_main, base_mtp.clone()),
            ("shared expert absent in MTP", base_main, shared_absent_mtp),
        ];

        for (case, main, mtp) in cases {
            let error = validate_qwen35_mtp_config(&main, &mtp).unwrap_err();
            let message = error.to_string();
            assert!(
                message.contains("qwen3.5 shared MoE scratch requires matching geometry"),
                "unexpected error for {case}: {message}"
            );
            assert!(
                message.contains("num_experts_per_tok="),
                "missing top-k values for {case}"
            );
            assert!(
                message.contains("moe_intermediate_size="),
                "missing routed intermediate values for {case}"
            );
            assert!(
                message.contains("shared_expert_intermediate_size="),
                "missing shared-expert presence/value for {case}"
            );
        }
    }

    #[test]
    fn moe_scratch_geometry_does_not_constrain_a_single_moe_user() {
        let mut main_dense = model_config(16, 8, 2);
        main_dense.text_config.num_experts_per_tok = 3;
        main_dense.text_config.moe_intermediate_size = 12;
        main_dense.text_config.shared_expert_intermediate_size = 0;
        let mtp_moe = model_config(32, 8, 1);
        assert!(!main_dense.layer_uses_moe(0));
        assert!(mtp_moe.layer_uses_moe(0));
        assert_eq!(validate_qwen35_mtp_config(&main_dense, &mtp_moe), Ok(()));

        let main_moe = model_config(16, 8, 1);
        let mut mtp_dense = model_config(32, 8, 2);
        mtp_dense.text_config.num_experts_per_tok = 3;
        mtp_dense.text_config.moe_intermediate_size = 12;
        mtp_dense.text_config.shared_expert_intermediate_size = 0;
        assert!(main_moe.layer_uses_moe(0));
        assert!(!mtp_dense.layer_uses_moe(0));
        assert_eq!(validate_qwen35_mtp_config(&main_moe, &mtp_dense), Ok(()));
    }

    fn model_config(intermediate_size: usize, num_experts: usize, decoder_sparse_step: usize) -> Qwen35ModelConfig {
        Qwen35ModelConfig {
            model_type: "qwen3_5".to_string(),
            tie_word_embeddings: false,
            text_config: Qwen35TextConfig {
                model_type: "qwen3_5_text".to_string(),
                hidden_size: 8,
                hidden_act: "silu".to_string(),
                intermediate_size,
                num_hidden_layers: 1,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                head_dim: 4,
                rms_norm_eps: 1e-6,
                vocab_size: 16,
                max_position_embeddings: 32,
                attention_bias: false,
                tie_word_embeddings: false,
                layer_types: vec!["full_attention".to_string()],
                full_attention_interval: 1,
                linear_num_value_heads: 2,
                linear_num_key_heads: 1,
                linear_key_head_dim: 4,
                linear_value_head_dim: 4,
                linear_conv_kernel_dim: 4,
                decoder_sparse_step,
                num_experts,
                num_experts_per_tok: usize::from(num_experts > 0) * 2,
                shared_expert_intermediate_size: usize::from(num_experts > 0) * 16,
                moe_intermediate_size: usize::from(num_experts > 0) * 8,
                norm_topk_prob: true,
                mtp_num_hidden_layers: 1,
                mtp_use_dedicated_embeddings: false,
                rope_theta: 10_000.0,
                partial_rotary_factor: 1.0,
                rope_parameters: None,
                use_cache: true,
                dtype: None,
                scale: 0.5,
                rope_dim: 4,
            },
            quantization: None,
        }
    }
}
