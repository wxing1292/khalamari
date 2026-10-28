use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use super::resolve_qwen3_model_weight_bindings;
use crate::model::qwen::v3::Qwen3ModelConfig;
use crate::model::qwen::v3::init_qwen3_model_config;

static NEXT_MODEL_DIR: AtomicU64 = AtomicU64::new(0);

#[test]
fn test_resolves_supported_exact_tensor_path_layout() {
    let model_config = model_config();
    let tensor_names = supported_tensor_names();

    let bindings = resolve_qwen3_model_weight_bindings(&model_config, tensor_names.iter().map(String::as_str)).unwrap();

    assert_eq!(bindings.embed.weight, "language_model.model.embed_tokens.weight");
    assert_eq!(
        bindings.main.layers[0].gqa.q.weight,
        "language_model.model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(bindings.unembed.weight, "language_model.lm_head.weight");
}

#[test]
fn test_rejects_missing_required_tensor() {
    let model_config = model_config();
    let missing = "language_model.model.layers.0.self_attn.q_norm.weight";
    let tensor_names = supported_tensor_names()
        .into_iter()
        .filter(|name| name != missing)
        .collect::<Vec<_>>();

    let error =
        resolve_qwen3_model_weight_bindings(&model_config, tensor_names.iter().map(String::as_str)).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("does not match a supported exact weight layout")
    );
    assert!(error.to_string().contains(missing));
}

fn model_config() -> Qwen3ModelConfig {
    let test_id = NEXT_MODEL_DIR.fetch_add(1, Ordering::Relaxed);
    let model_dir = std::env::temp_dir().join(format!(
        "khalamari-qwen3-weight-layout-{}-{test_id}",
        std::process::id()
    ));
    std::fs::create_dir(&model_dir).unwrap();
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"],
            "attention_bias": false,
            "eos_token_id": 15,
            "head_dim": 4,
            "hidden_act": "silu",
            "hidden_size": 8,
            "intermediate_size": 16,
            "max_position_embeddings": 32,
            "model_type": "qwen3",
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "quantization": {
                "group_size": 64,
                "bits": 4
            },
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16",
            "use_cache": true,
            "vocab_size": 16
        }))
        .unwrap(),
    )
    .unwrap();
    let model_config = init_qwen3_model_config(&model_dir);
    std::fs::remove_dir_all(model_dir).unwrap();
    model_config.unwrap()
}

fn supported_tensor_names() -> Vec<String> {
    let container_prefix = "language_model.";
    let model_prefix = "language_model.model.";
    let layer_prefix = "language_model.model.layers.0";
    let attention_prefix = format!("{layer_prefix}.self_attn");
    let mlp_prefix = format!("{layer_prefix}.mlp");
    let mut names = Vec::new();

    push_quantized_names(&mut names, &format!("{model_prefix}embed_tokens"));
    names.push(format!("{model_prefix}norm.weight"));
    push_quantized_names(&mut names, &format!("{container_prefix}lm_head"));
    names.extend([
        format!("{layer_prefix}.input_layernorm.weight"),
        format!("{layer_prefix}.post_attention_layernorm.weight"),
    ]);
    for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        push_quantized_names(&mut names, &format!("{attention_prefix}.{projection}"));
    }
    names.extend([
        format!("{attention_prefix}.q_norm.weight"),
        format!("{attention_prefix}.k_norm.weight"),
    ]);
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        push_quantized_names(&mut names, &format!("{mlp_prefix}.{projection}"));
    }
    names
}

fn push_quantized_names(names: &mut Vec<String>, prefix: &str) {
    for suffix in ["weight", "scales", "biases"] {
        names.push(format!("{prefix}.{suffix}"));
    }
}
