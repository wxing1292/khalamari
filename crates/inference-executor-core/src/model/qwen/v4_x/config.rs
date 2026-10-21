use std::path::Path;

use serde::Deserialize;
use serde::Serialize;

use crate::attn::gdn::GDNOutputGate;
use crate::def::ModelExecutorError;
use crate::model::qwen::v3_x::QuantizationConfig;
use crate::model::qwen::v3_x::RopeParameters;

/// Text-only Qwen4x checkpoint schema.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Qwen4xModelConfig {
    pub model_type: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub text_config: Qwen4xTextConfig,
    #[serde(default, alias = "quantization_config")]
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Qwen4xTextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: usize,
    pub linear_num_value_heads: usize,
    pub linear_num_key_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub shared_expert_intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_hc_count")]
    pub hc_count: usize,
    #[serde(default = "default_hc_lowrank")]
    pub hc_lowrank: usize,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: usize,
    #[serde(default)]
    pub ple_layer_ids: Vec<usize>,
    #[serde(default)]
    pub ple_embed_dim: usize,
    #[serde(default = "default_ple_conv_kernel_size")]
    pub ple_conv_kernel_size: usize,
    #[serde(default = "default_ngram_size")]
    pub ngram_size: usize,
    #[serde(default = "default_heads_per_ngram")]
    pub heads_per_ngram: usize,
    #[serde(default = "default_ngram_vocab_size_base")]
    pub ngram_vocab_size_base: u32,
    #[serde(default = "default_make_ngram_vocab_size_divisible_by")]
    pub make_ngram_vocab_size_divisible_by: usize,
    #[serde(default = "default_split_ngram_parts")]
    pub split_ngram_parts: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_indexer_n_heads")]
    pub indexer_n_heads: usize,
    #[serde(default = "default_indexer_kv_heads")]
    pub indexer_kv_heads: usize,
    #[serde(default = "default_indexer_head_dim")]
    pub indexer_head_dim: usize,
    #[serde(default = "default_indexer_budget")]
    pub indexer_budget: usize,
    #[serde(default = "default_indexer_compress_ratio")]
    pub indexer_compress_ratio: usize,
    #[serde(default = "default_output_gate_type")]
    pub output_gate_type: String,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_rope_parameters")]
    pub rope_parameters: RopeParameters,
}

fn default_hc_count() -> usize {
    4
}

fn default_hc_lowrank() -> usize {
    320
}

fn default_full_attention_interval() -> usize {
    4
}

fn default_ple_conv_kernel_size() -> usize {
    4
}

fn default_ngram_size() -> usize {
    3
}

fn default_heads_per_ngram() -> usize {
    8
}

fn default_ngram_vocab_size_base() -> u32 {
    20_000_000
}

fn default_make_ngram_vocab_size_divisible_by() -> usize {
    128
}

fn default_split_ngram_parts() -> usize {
    128
}

fn default_seed() -> u64 {
    1234
}

fn default_indexer_n_heads() -> usize {
    4
}

fn default_indexer_kv_heads() -> usize {
    1
}

fn default_indexer_head_dim() -> usize {
    128
}

fn default_indexer_budget() -> usize {
    2048
}

fn default_indexer_compress_ratio() -> usize {
    4
}

fn default_output_gate_type() -> String {
    "sigmoid".to_owned()
}

fn default_hidden_act() -> String {
    "silu".to_owned()
}

fn default_norm_topk_prob() -> bool {
    true
}

fn default_rope_parameters() -> RopeParameters {
    RopeParameters {
        rope_type: Some("default".to_owned()),
        rope_theta: Some(10_000_000.0),
        partial_rotary_factor: Some(0.25),
        ..Default::default()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerType {
    GDN,
    QSA,
}

impl Qwen4xModelConfig {
    pub fn normalize(&mut self) -> Result<(), ModelExecutorError> {
        let text = &mut self.text_config;
        if text.hidden_size == 0
            || text.num_hidden_layers == 0
            || text.vocab_size == 0
            || text.max_position_embeddings == 0
            || text.full_attention_interval == 0
        {
            return Err(ModelExecutorError::custom(
                "Qwen4x model dimensions and attention interval must be positive",
            ));
        }
        if text.head_dim == 0 {
            if text.num_attention_heads == 0 || !text.hidden_size.is_multiple_of(text.num_attention_heads) {
                return Err(ModelExecutorError::custom(
                    "Qwen4x cannot infer an integral head dimension",
                ));
            }
            text.head_dim = text.hidden_size / text.num_attention_heads;
        }
        if text.ple_embed_dim == 0 {
            text.ple_embed_dim = text.hidden_size;
        }
        let dimensions = [
            text.hidden_size,
            text.num_hidden_layers,
            text.num_attention_heads,
            text.num_key_value_heads,
            text.head_dim,
            text.linear_num_value_heads,
            text.linear_num_key_heads,
            text.linear_key_head_dim,
            text.linear_value_head_dim,
            text.linear_conv_kernel_dim,
            text.num_experts,
            text.num_experts_per_tok,
            text.moe_intermediate_size,
            text.vocab_size,
            text.max_position_embeddings,
            text.hc_count,
            text.hc_lowrank,
            text.indexer_n_heads,
            text.indexer_head_dim,
            text.indexer_budget,
        ];
        if dimensions.contains(&0)
            || text.hc_count < 2
            || text.num_experts_per_tok > text.num_experts
            || !text.num_attention_heads.is_multiple_of(text.num_key_value_heads)
            || !text.linear_num_value_heads.is_multiple_of(text.linear_num_key_heads)
            || text.hc_count.checked_mul(text.hidden_size).is_none()
        {
            return Err(ModelExecutorError::custom(
                "Qwen4x has invalid model dimensions or head/expert relationships",
            ));
        }
        let theta = text.rope_parameters.rope_theta.unwrap_or(0.0);
        let rotary = text.rope_parameters.partial_rotary_factor.unwrap_or(0.0);
        let rope_dim = text.head_dim as f32 * rotary;
        if !text.rms_norm_eps.is_finite()
            || text.rms_norm_eps <= 0.0
            || !theta.is_finite()
            || theta <= 0.0
            || !rotary.is_finite()
            || rotary <= 0.0
            || rope_dim <= 0.0
            || rope_dim.fract() != 0.0
            || rope_dim as usize > text.head_dim
            || rope_dim as usize > text.indexer_head_dim
            || !(rope_dim as usize).is_multiple_of(2)
            || text
                .rope_parameters
                .rope_type
                .as_deref()
                .is_some_and(|t| t != "default")
        {
            return Err(ModelExecutorError::custom(
                "Qwen4x requires positive RMSNorm epsilon and supported default partial RoPE",
            ));
        }
        if text.hidden_act != "silu" || text.attention_bias || text.indexer_kv_heads != 1 {
            return Err(ModelExecutorError::custom(
                "Qwen4x requires SiLU MLP, bias-free attention, and one indexer KV head",
            ));
        }
        if text.indexer_compress_ratio <= 1 || !text.indexer_budget.is_multiple_of(text.indexer_compress_ratio) {
            return Err(ModelExecutorError::custom(
                "Qwen4x indexer compression ratio must exceed one and divide its budget",
            ));
        }
        if !matches!(text.output_gate_type.as_str(), "silu" | "sigmoid") {
            return Err(ModelExecutorError::custom(
                "Qwen4x output_gate_type must be silu or sigmoid",
            ));
        }
        if text.layer_types.is_empty() {
            text.layer_types = (0..text.num_hidden_layers)
                .map(|i| {
                    if (i + 1).is_multiple_of(text.full_attention_interval) {
                        "qwen_sparse_attention"
                    } else {
                        "linear_attention"
                    }
                    .to_owned()
                })
                .collect();
        }
        if text.layer_types.len() != text.num_hidden_layers
            || text.layer_types.iter().any(|kind| {
                !matches!(
                    kind.as_str(),
                    "linear_attention" | "qwen_sparse_attention" | "full_attention"
                )
            })
        {
            return Err(ModelExecutorError::custom(
                "Qwen4x layer_types must define one supported attention type per layer",
            ));
        }
        if !text.layer_types.iter().any(|kind| kind == "linear_attention")
            || !text
                .layer_types
                .iter()
                .any(|kind| matches!(kind.as_str(), "qwen_sparse_attention" | "full_attention"))
        {
            return Err(ModelExecutorError::custom("Qwen4x requires both GDN and QSA layers"));
        }
        text.ple_layer_ids.sort_unstable();
        text.ple_layer_ids.dedup();
        if text
            .ple_layer_ids
            .iter()
            .any(|&id| id == 0 || id > text.num_hidden_layers || text.layer_types[id - 1] != "linear_attention")
        {
            return Err(ModelExecutorError::custom(
                "Qwen4x PLE layer IDs must select one-indexed GDN layers",
            ));
        }
        if !text.ple_layer_ids.is_empty() {
            let ngram_heads = text
                .ngram_size
                .checked_sub(1)
                .and_then(|n| n.checked_mul(text.heads_per_ngram));
            if text.ngram_size < 2
                || text.heads_per_ngram == 0
                || text.split_ngram_parts == 0
                || text.ple_conv_kernel_size < 2
                || text.ple_embed_dim == 0
                || text.ngram_vocab_size_base < 2
                || text.make_ngram_vocab_size_divisible_by == 0
                || ngram_heads.is_none_or(|heads| heads == 0 || !text.ple_embed_dim.is_multiple_of(heads))
                || u32::try_from(text.hc_count * text.hidden_size).is_err()
                || text
                    .ple_conv_kernel_size
                    .checked_sub(1)
                    .and_then(|n| n.checked_mul(text.ngram_size))
                    .is_none_or(|n| u32::try_from(n).is_err())
                || [
                    text.vocab_size,
                    text.hidden_size,
                    text.hc_count,
                    text.ngram_size,
                    text.heads_per_ngram,
                    text.ple_embed_dim,
                    text.ple_conv_kernel_size,
                    text.split_ngram_parts,
                    text.make_ngram_vocab_size_divisible_by,
                ]
                .iter()
                .any(|&n| u32::try_from(n).is_err())
            {
                return Err(ModelExecutorError::custom(
                    "Qwen4x PLE has invalid n-gram/table geometry",
                ));
            }
            self.eos_token_id()?;
        }
        if let Some(quantization) = &mut self.quantization {
            quantization.normalize_tensor_overrides();
        }
        Ok(())
    }

    pub fn layer_type_at(&self, model_layer_index: usize) -> LayerType {
        match self.text_config.layer_types[model_layer_index].as_str() {
            "linear_attention" => LayerType::GDN,
            "qwen_sparse_attention" | "full_attention" => LayerType::QSA,
            _ => unreachable!("Qwen4x layer types are normalized at initialization"),
        }
    }

    pub fn output_gate(&self) -> GDNOutputGate {
        match self.text_config.output_gate_type.as_str() {
            "silu" => GDNOutputGate::Silu,
            "sigmoid" => GDNOutputGate::Sigmoid,
            _ => unreachable!("Qwen4x output gate is validated at initialization"),
        }
    }

    pub fn eos_token_id(&self) -> Result<u32, ModelExecutorError> {
        Ok(self.eos_token_ids()?[0])
    }

    pub fn eos_token_ids(&self) -> Result<Vec<u32>, ModelExecutorError> {
        let error = || ModelExecutorError::custom("Qwen4x requires valid EOS token IDs");
        let parse = |value: &serde_json::Value| {
            value
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .filter(|&id| (id as usize) < self.text_config.vocab_size)
                .ok_or_else(error)
        };
        match self.text_config.eos_token_id.as_ref() {
            Some(serde_json::Value::Array(values)) if !values.is_empty() => values.iter().map(parse).collect(),
            Some(value) => Ok(vec![parse(value)?]),
            None => Err(error()),
        }
    }
}

pub fn init_qwen4x_model_config(model_dir: impl AsRef<Path>) -> Result<Qwen4xModelConfig, ModelExecutorError> {
    let path = model_dir.as_ref().join("config.json");
    let data =
        std::fs::read(&path).map_err(|err| ModelExecutorError::custom(format!("unable to read {path:?}: {err}")))?;
    let mut envelope: serde_json::Value = serde_json::from_slice(&data)
        .map_err(|err| ModelExecutorError::custom(format!("unable to parse {path:?}: {err}")))?;
    if !envelope.is_object() || !envelope.get("text_config").is_some_and(serde_json::Value::is_object) {
        return Err(ModelExecutorError::custom("Qwen4x requires an object with text_config"));
    }
    // Both spellings occur in MLX checkpoints. Resolve once before typed parsing.
    if let Some(nested) = envelope.get("quantization_config").cloned() {
        if let Some(primary) = envelope.get("quantization")
            && primary != &nested
        {
            return Err(ModelExecutorError::custom("Qwen4x quantization layouts disagree"));
        }
        envelope
            .as_object_mut()
            .expect("checkpoint envelope is an object")
            .remove("quantization_config");
        envelope["quantization"] = nested;
    }
    if envelope["text_config"]
        .get("eos_token_id")
        .is_none_or(serde_json::Value::is_null)
        && envelope.get("eos_token_id").is_some()
    {
        envelope["text_config"]["eos_token_id"] = envelope["eos_token_id"].clone();
    }
    let mut config: Qwen4xModelConfig = serde_json::from_value(envelope)
        .map_err(|err| ModelExecutorError::custom(format!("unable to parse Qwen4x checkpoint config: {err}")))?;
    config.normalize()?;
    Ok(config)
}

#[cfg(test)]
#[path = "config_test.rs"]
mod tests;
