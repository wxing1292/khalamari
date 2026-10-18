fn gr_core(config: &Qwen4xModelConfig) -> GatedResidualCore {
    let text = &config.text_config;
    GatedResidualCore {
        num_branches: text.hc_count,
        hidden_dim: text.hidden_size,
        low_rank_dim: text.hc_lowrank,
        eps: text.rms_norm_eps,
    }
}

pub fn gdn_configs(
    config: &Qwen4xModelConfig,
    model_layer_index: usize,
    bindings: &inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGDNWeightBindings,
) -> Result<(GDNCore, GDNMetalConfig), ModelExecutorError> {
    let text = &config.text_config;
    let layout = uniform_layout(
        config,
        &[
            &bindings.qkv.weight,
            &bindings.a.weight,
            &bindings.b.weight,
            &bindings.z.weight,
            &bindings.output.weight,
        ],
        "Qwen4x GDN",
    )?;
    Ok((
        GDNCore {
            model_layer_index,
            hidden_dim: text.hidden_size,
            num_qk_heads: text.linear_num_key_heads,
            qk_head_dim: text.linear_key_head_dim,
            num_v_heads: text.linear_num_value_heads,
            v_head_dim: text.linear_value_head_dim,
            conv_kernel_size: text.linear_conv_kernel_dim,
            q_scale: (text.linear_key_head_dim as f32).sqrt().recip(),
            output_gate: config.output_gate(),
        },
        GDNMetalConfig {
            group_size: layout.group_size,
            bits: layout.bits,
            norm_eps: text.rms_norm_eps,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            qkvabz_scale_bias_dtype: Dtype::Bfloat16,
            output_scale_bias_dtype: Dtype::Bfloat16,
        },
    ))
}

fn qsa_configs(
    config: &Qwen4xModelConfig,
    model_layer_index: usize,
    main: &inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings,
    indexer: &Qwen4xIndexerWeightBindings,
) -> Result<(GQACore, QSAMetalConfig, QSAIndexerCore, QSAIndexerMetalConfig), ModelExecutorError> {
    let text = &config.text_config;
    let main_layout = uniform_layout(
        config,
        &[&main.q.weight, &main.k.weight, &main.v.weight, &main.output.weight],
        "Qwen4x QSA",
    )?;
    let index_layout = uniform_layout(config, &[&indexer.qk.weight], "Qwen4x indexer")?;
    let rope_theta = text
        .rope_parameters
        .rope_theta
        .ok_or_else(|| ModelExecutorError::custom("Qwen4x requires rope_theta"))?;
    let rotary = text
        .rope_parameters
        .partial_rotary_factor
        .ok_or_else(|| ModelExecutorError::custom("Qwen4x requires partial_rotary_factor"))?;
    let rope_dim = (text.head_dim as f32 * rotary) as u32;
    let index_rope_dim = rope_dim as usize;
    Ok((
        GQACore::new(
            model_layer_index,
            text.hidden_size,
            text.head_dim,
            text.num_attention_heads,
            text.num_key_value_heads,
            (text.head_dim as f32).sqrt().recip(),
        ),
        QSAMetalConfig {
            group_size: main_layout.group_size,
            bits: main_layout.bits,
            page_bytes: 32 * 1024,
            rope_dim,
            norm_eps: text.rms_norm_eps,
            rope_theta,
            rope_scaling: RopeScaling::Default,
            io_dtype: Dtype::Bfloat16,
        },
        QSAIndexerCore {
            hidden_dim: text.hidden_size,
            num_q_heads: text.indexer_n_heads,
            head_dim: text.indexer_head_dim,
            compress_ratio: text.indexer_compress_ratio,
            top_k: text.indexer_budget / text.indexer_compress_ratio,
            rope_dim: index_rope_dim,
            rope_theta,
            norm_eps: text.rms_norm_eps,
        },
        QSAIndexerMetalConfig {
            group_size: index_layout.group_size,
            bits: index_layout.bits,
            io_dtype: Dtype::Bfloat16,
        },
    ))
}

fn moe_configs(
    config: &Qwen4xModelConfig,
    model_layer_index: usize,
    bindings: &inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xMoEWeightBindings,
) -> Result<(GatedMoECore, GatedMoEMetalConfig), ModelExecutorError> {
    let text = &config.text_config;
    let mut names = vec![
        bindings.experts.gate.weight.as_str(),
        bindings.experts.up.weight.as_str(),
        bindings.experts.down.weight.as_str(),
    ];
    if let Some(shared) = &bindings.shared_expert {
        names.extend([
            shared.gate.weight.as_str(),
            shared.up.weight.as_str(),
            shared.down.weight.as_str(),
        ]);
    }
    let layout = uniform_layout(config, &names, "Qwen4x experts")?;
    let router = uniform_layout(config, &[&bindings.router.weight], "Qwen4x router")?;
    let shared_gate = bindings
        .shared_expert_gate
        .as_ref()
        .map(|b| uniform_layout(config, &[&b.weight], "Qwen4x shared gate"))
        .transpose()?;
    if router.group_size != layout.group_size || shared_gate.is_some_and(|g| g.group_size != layout.group_size) {
        return Err(ModelExecutorError::custom(
            "Qwen4x MoE projections must share a quantization group size",
        ));
    }
    Ok((
        GatedMoECore {
            model_layer_index,
            hidden_dim: text.hidden_size,
            intermediate_dim: text.moe_intermediate_size,
            shared_experts_intermediate_dim: (text.shared_expert_intermediate_size > 0)
                .then_some(text.shared_expert_intermediate_size),
            num_experts: text.num_experts,
            num_experts_per_token: text.num_experts_per_tok,
            norm_topk_prob: text.norm_topk_prob,
        },
        GatedMoEMetalConfig {
            group_size: layout.group_size,
            bits: layout.bits,
            router_bits: router.bits,
            shared_expert_gate_bits: shared_gate.map_or(layout.bits, |g| g.bits),
            io_dtype: Dtype::Bfloat16,
        },
    ))
}

pub fn uniform_layout(
    config: &Qwen4xModelConfig,
    names: &[&str],
    component: &str,
) -> Result<QuantizedAffineLayout, ModelExecutorError> {
    let quant = config
        .quantization
        .as_ref()
        .ok_or_else(|| ModelExecutorError::custom("Qwen4x requires an affine-quantized checkpoint"))?;
    let layout = resolve_uniform_quantization(quant, names, component)?;
    if !matches!(layout.group_size, 32 | 64 | 128)
        || !matches!(layout.bits, 2 | 3 | 4 | 6 | 8)
        || layout.mode.as_deref().is_some_and(|m| m != "affine")
    {
        return Err(ModelExecutorError::custom(format!(
            "{component} has an unsupported affine layout"
        )));
    }
    Ok(QuantizedAffineLayout {
        group_size: layout.group_size as u32,
        bits: layout.bits as u32,
        scale_bias_dtype: Dtype::Bfloat16,
    })
}
