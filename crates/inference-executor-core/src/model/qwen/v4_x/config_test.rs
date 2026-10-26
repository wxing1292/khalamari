use super::*;

fn fixture() -> Qwen4xModelConfig {
    let mut config: Qwen4xModelConfig = serde_json::from_str(include_str!("config_fixture.json")).unwrap();
    config.normalize().unwrap();
    config
}
#[test]
fn test_qwen4x_first8_layout() {
    let config = fixture();
    let kinds: Vec<_> = (0..8).map(|i| config.layer_type_at(i)).collect();
    assert_eq!(
        kinds,
        vec![
            LayerType::GDN,
            LayerType::GDN,
            LayerType::GDN,
            LayerType::QSA,
            LayerType::GDN,
            LayerType::GDN,
            LayerType::GDN,
            LayerType::QSA
        ]
    );
    assert_eq!(config.output_gate(), GDNOutputGate::Sigmoid);
    assert_eq!(config.text_config.ple_layer_ids, [2]);
}
#[test]
fn test_rejects_unsupported_architecture_at_config_boundary() {
    for invalid in [
        "layer_types",
        "ple_layer_ids",
        "output_gate_type",
        "indexer_budget",
        "head_ratio",
        "compression_ratio",
        "rope_parameters",
    ] {
        let mut config = fixture();
        match invalid {
            "layer_types" => config.text_config.layer_types[0] = "unknown".to_string(),
            "ple_layer_ids" => config.text_config.ple_layer_ids = vec![4],
            "output_gate_type" => config.text_config.output_gate_type = "relu".to_string(),
            "indexer_budget" => config.text_config.indexer_budget = 2047,
            "head_ratio" => config.text_config.num_attention_heads = 23,
            "compression_ratio" => config.text_config.indexer_compress_ratio = 1,
            "rope_parameters" => config.text_config.rope_parameters.rope_type = Some("yarn".to_string()),
            _ => unreachable!(),
        }
        assert!(config.normalize().is_err(), "{invalid}");
    }
}

#[test]
fn test_model_geometry_does_not_require_metal_capabilities() {
    let mut config = fixture();
    config.text_config.num_attention_heads = 12;
    config.text_config.num_experts = 1024;
    config.text_config.max_position_embeddings = i32::MAX as usize + 1;
    config.normalize().unwrap();

    config.text_config.ple_layer_ids.clear();
    config.text_config.hidden_size = i32::MAX as usize + 1;
    config.normalize().unwrap();
}

#[test]
fn test_ple_reference_integer_domains_remain_validated() {
    for invalid in ["residual_width", "vocab_size", "history_length"] {
        let mut config = fixture();
        match invalid {
            "residual_width" => config.text_config.hidden_size = 1 << 30,
            "vocab_size" => config.text_config.vocab_size = u32::MAX as usize + 1,
            "history_length" => config.text_config.ple_conv_kernel_size = 1 << 31,
            _ => unreachable!(),
        }
        assert!(config.normalize().is_err(), "{invalid}");
    }
}

#[test]
fn test_checkpoint_defaults_and_rope_alias() {
    let mut value: serde_json::Value = serde_json::from_str(include_str!("config_fixture.json")).unwrap();
    let text = value["text_config"].as_object_mut().unwrap();
    for field in [
        "output_gate_type",
        "full_attention_interval",
        "indexer_n_heads",
        "indexer_kv_heads",
        "indexer_head_dim",
        "indexer_budget",
        "indexer_compress_ratio",
        "rope_parameters",
    ] {
        text.remove(field);
    }
    let mut config: Qwen4xModelConfig = serde_json::from_value(value).unwrap();
    config.normalize().unwrap();
    assert_eq!(config.output_gate(), GDNOutputGate::Sigmoid);
    assert_eq!(config.text_config.indexer_budget, 2048);
    assert_eq!(config.text_config.rope_parameters.partial_rotary_factor, Some(0.25));
    let rope: RopeParameters = serde_json::from_value(serde_json::json!({"type": "default"})).unwrap();
    assert_eq!(rope.rope_type.as_deref(), Some("default"));
}

#[test]
fn test_eos_list_retains_all_stops_and_rejects_invalid_ids() {
    let mut config = fixture();
    config.text_config.eos_token_id = Some(serde_json::json!([0, 7]));
    assert_eq!(config.eos_token_ids().unwrap(), [0, 7]);
    assert_eq!(config.eos_token_id().unwrap(), 0);
    config.text_config.eos_token_id = Some(serde_json::json!([0, 256]));
    assert!(config.normalize().is_err());
}

#[test]
fn test_dense_checkpoint_and_native_mtp_boundaries() {
    let mut value: serde_json::Value = serde_json::from_str(include_str!("config_fixture.json")).unwrap();
    let text = value["text_config"].as_object_mut().unwrap();
    for field in [
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    ] {
        text.remove(field);
    }
    text.insert("intermediate_size".to_owned(), 64.into());
    let mut config: Qwen4xModelConfig = serde_json::from_value(value).unwrap();
    config.normalize().unwrap();
    assert_eq!(config.text_config.num_experts, 0);
    assert!(config.mtp_config().is_err());
    config.text_config.mtp = Some(Qwen4xMTPConfig {
        num_hidden_layers: 1,
        rope_theta: Some(500_000.0),
        ..Default::default()
    });
    let head = config.mtp_config().unwrap();
    assert_eq!(head.text_config.num_hidden_layers, 1);
    assert_eq!(head.layer_type_at(0), LayerType::QSA);
    assert!(head.text_config.ple_layer_ids.is_empty());
    assert_eq!(head.text_config.rope_parameters.rope_theta, Some(500_000.0));
    for invalid in ["layer_count", "hidden_source", "embedding_source", "rotary"] {
        let mut model = config.clone();
        let mtp = model.text_config.mtp.as_mut().unwrap();
        match invalid {
            "layer_count" => mtp.num_hidden_layers = 2,
            "hidden_source" => mtp.mtp_use_hidden_state_from_layer = Some(3),
            "embedding_source" => model.text_config.mtp_use_dedicated_embeddings = true,
            "rotary" => mtp.rope_theta = Some(0.0),
            _ => unreachable!(),
        }
        // Main-only can use the checkpoint independently of optional Spec capabilities.
        model.normalize().unwrap();
        assert!(model.mtp_config().is_err(), "{invalid}");
    }
    let mut pruned = fixture();
    pruned.text_config.mtp = config.text_config.mtp.clone();
    pruned.text_config.mtp_num_experts = Some(2);
    pruned.text_config.mtp_num_experts_per_tok = Some(1);
    let head = pruned.mtp_config().unwrap();
    assert_eq!(head.text_config.num_experts, 2);
    assert_eq!(head.text_config.num_experts_per_tok, 1);
    assert_eq!(pruned.text_config.num_experts, 4);
    pruned.text_config.mtp_num_experts_per_tok = Some(3);
    pruned.normalize().unwrap();
    assert!(pruned.mtp_config().is_err());
    config.text_config.intermediate_size = 0;
    assert!(config.normalize().is_err());
}
