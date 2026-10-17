use super::*;
use crate::model::qwen::v3_x::TensorPathLayout;
fn fixture() -> Qwen4xModelConfig {
    let mut config: Qwen4xModelConfig = serde_json::from_str(include_str!("config_fixture.json")).unwrap();
    config.normalize().unwrap();
    config
}
#[test]
fn test_qwen4x_weight_layout() {
    let config = fixture();
    let bindings = bind(
        &config,
        TensorPathLayout {
            container_prefix: "model.language_model.",
            model_prefix: "",
        },
        "linear_attn",
    );
    let resolved = resolve_qwen4x_model_weight_bindings(&config, bindings.tensor_names()).unwrap();
    assert!(resolved.layers[0].ple.is_none());
    assert_eq!(
        resolved.layers[1].ple.as_ref().unwrap().table[0].weight,
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight"
    );
    assert!(resolved.mixer.write.is_none());
    assert_eq!(
        resolved.layers[7].attention_residual.write.as_deref(),
        Some("model.language_model.layers.7.attn_hyper_connection.block_inject_weight.weight")
    );
}
#[test]
fn test_incomplete_or_mixed_checkpoint_layout() {
    let config = fixture();
    let a = bind(
        &config,
        TensorPathLayout {
            container_prefix: "",
            model_prefix: "model.",
        },
        "linear_attn",
    );
    let b = bind(
        &config,
        TensorPathLayout {
            container_prefix: "language_model.",
            model_prefix: "model.",
        },
        "linear_attn",
    );
    let mut names = a.tensor_names();
    names.pop();
    assert!(resolve_qwen4x_model_weight_bindings(&config, names.iter().copied()).is_err());
    names.push(b.mixer.up.as_str());
    assert!(resolve_qwen4x_model_weight_bindings(&config, names.iter().copied()).is_err());
}
