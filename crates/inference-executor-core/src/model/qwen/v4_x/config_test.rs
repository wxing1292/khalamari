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
        "rope_parameters",
    ] {
        let mut config = fixture();
        match invalid {
            "layer_types" => config.text_config.layer_types[0] = "unknown".to_string(),
            "ple_layer_ids" => config.text_config.ple_layer_ids = vec![4],
            "output_gate_type" => config.text_config.output_gate_type = "relu".to_string(),
            "indexer_budget" => config.text_config.indexer_budget = 2047,
            "rope_parameters" => config.text_config.rope_parameters.rope_type = Some("yarn".to_string()),
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
