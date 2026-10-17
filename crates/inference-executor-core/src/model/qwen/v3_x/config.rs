use std::collections::HashMap;

use serde::Deserialize;
use serde::Serialize;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct QuantizationConfig {
    pub group_size: usize,
    pub bits: usize,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(flatten, default)]
    pub tensor_overrides: HashMap<String, TensorQuantizationOverride>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct TensorQuantizationOverride {
    #[serde(default)]
    pub group_size: Option<usize>,
    #[serde(default)]
    pub bits: Option<usize>,
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedQuantizationConfig {
    pub group_size: usize,
    pub bits: usize,
    pub mode: Option<String>,
}

impl TensorQuantizationOverride {
    fn resolve_with_defaults(&self, defaults: &QuantizationConfig) -> ResolvedQuantizationConfig {
        ResolvedQuantizationConfig {
            group_size: self.group_size.unwrap_or(defaults.group_size),
            bits: self.bits.unwrap_or(defaults.bits),
            mode: self.mode.clone().or_else(|| defaults.mode.clone()),
        }
    }
}

impl QuantizationConfig {
    pub fn resolve_for_tensor(&self, tensor_name: &str) -> ResolvedQuantizationConfig {
        let tensor_base = tensor_name.strip_suffix(".weight").unwrap_or(tensor_name);
        let internal_name = normalize_qwen_name(tensor_name);
        let internal_base = normalize_qwen_name(tensor_base);
        self.tensor_overrides
            .get(tensor_name)
            .or_else(|| self.tensor_overrides.get(tensor_base))
            .or_else(|| self.tensor_overrides.get(&internal_name))
            .or_else(|| self.tensor_overrides.get(&internal_base))
            .map(|tensor_override| tensor_override.resolve_with_defaults(self))
            .unwrap_or_else(|| {
                ResolvedQuantizationConfig {
                    group_size: self.group_size,
                    bits: self.bits,
                    mode: self.mode.clone(),
                }
            })
    }

    pub fn normalize_tensor_overrides(&mut self) {
        if self.tensor_overrides.is_empty() {
            return;
        }

        let explicit_overrides = std::mem::take(&mut self.tensor_overrides);
        for (name, tensor_override) in &explicit_overrides {
            self.tensor_overrides.insert(name.clone(), tensor_override.clone());
        }
        for (name, tensor_override) in explicit_overrides {
            for alias in quant_override_aliases(&name) {
                self.tensor_overrides
                    .entry(alias)
                    .or_insert_with(|| tensor_override.clone());
            }
        }
    }
}

fn quant_override_aliases(name: &str) -> [String; 3] {
    let base = name.strip_suffix(".weight").unwrap_or(name);
    [base.to_string(), normalize_qwen_name(name), normalize_qwen_name(base)]
}

fn normalize_qwen_name(name: &str) -> String {
    let mut normalized = name;
    for prefix in [
        "model.language_model.model.",
        "model.language_model.",
        "language_model.model.",
        "language_model.",
        "model.",
    ] {
        if let Some(stripped) = normalized.strip_prefix(prefix) {
            normalized = stripped;
            break;
        }
    }
    if let Some(suffix) = normalized.strip_prefix("lm_head") {
        return format!("unembed{suffix}");
    }
    normalized.to_string()
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct RopeParameters {
    #[serde(default, alias = "type")]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub attention_factor: Option<f32>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
    #[serde(default)]
    pub mscale: Option<f32>,
    #[serde(default)]
    pub mscale_all_dim: Option<f32>,
    #[serde(default)]
    pub truncate: Option<bool>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorPathLayout {
    pub container_prefix: &'static str,
    pub model_prefix: &'static str,
}

impl TensorPathLayout {
    pub fn model_path(&self, suffix: &str) -> String {
        format!("{}{}{}", self.container_prefix, self.model_prefix, suffix)
    }

    pub fn container_path(&self, suffix: &str) -> String {
        format!("{}{}", self.container_prefix, suffix)
    }
}

pub fn tensor_path_layout_candidates() -> [TensorPathLayout; 5] {
    [
        TensorPathLayout {
            container_prefix: "",
            model_prefix: "model.",
        },
        TensorPathLayout {
            container_prefix: "language_model.",
            model_prefix: "model.",
        },
        TensorPathLayout {
            container_prefix: "language_model.",
            model_prefix: "",
        },
        TensorPathLayout {
            container_prefix: "model.language_model.",
            model_prefix: "model.",
        },
        TensorPathLayout {
            container_prefix: "model.language_model.",
            model_prefix: "",
        },
    ]
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::normalize_qwen_name;
    use crate::model::qwen::v3_x::QuantizationConfig;

    #[test]
    fn test_resolves_quantization_override_aliases() {
        let mut quantization = serde_json::from_value::<QuantizationConfig>(json!({
            "group_size": 64,
            "bits": 4,
            "model.layers.0.mlp.gate.weight": {
                "group_size": 128,
                "bits": 3
            }
        }))
        .unwrap();
        quantization.normalize_tensor_overrides();

        let direct = quantization.resolve_for_tensor("model.layers.0.mlp.gate.weight");
        let normalized = quantization.resolve_for_tensor("layers.0.mlp.gate.weight");
        let fallback = quantization.resolve_for_tensor("model.layers.0.mlp.up_proj.weight");

        assert_eq!(direct.group_size, 128);
        assert_eq!(normalized.bits, 3);
        assert_eq!(fallback.group_size, 64);
        assert_eq!(fallback.bits, 4);
    }

    #[test]
    fn test_normalizes_common_qwen_container_prefixes() {
        assert_eq!(
            normalize_qwen_name("model.layers.0.mlp.gate_proj.weight"),
            "layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            normalize_qwen_name("language_model.model.layers.1.self_attn.q_proj.weight"),
            "layers.1.self_attn.q_proj.weight"
        );
        assert_eq!(normalize_qwen_name("lm_head.weight"), "unembed.weight");
    }
}
