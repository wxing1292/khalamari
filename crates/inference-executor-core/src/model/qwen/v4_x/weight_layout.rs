use std::collections::HashSet;

use super::LayerType;
use super::Qwen4xModelConfig;
use crate::checkpoint::QuantizedTensorBindings;
use crate::def::ModelExecutorError;
use crate::model::qwen::v3_x::TensorPathLayout;
use crate::model::qwen::v3_x::tensor_path_layout_candidates;
use crate::model::qwen::v3_x::weight_layout::Qwen3xGDNWeightBindings;
use crate::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;
use crate::model::qwen::v3_x::weight_layout::Qwen3xMoEWeightBindings;
use crate::model::qwen::v3_x::weight_layout::Qwen3xSparseExpertWeightBindings;
use crate::model::qwen::v3_x::weight_layout::dense_mlp_bindings;
use crate::model::qwen::v3_x::weight_layout::push_quantized_tensor_names;
use crate::model::qwen::v3_x::weight_layout::quantized;
use crate::model::qwen::v3_x::weight_layout::quantized_path;

#[derive(Clone, Debug)]
pub struct Qwen4xModelWeightBindings {
    pub embed: QuantizedTensorBindings,
    pub layers: Vec<Qwen4xLayerWeightBindings>,
    pub mixer: Qwen4xGRWeightBindings,
    pub unembed: QuantizedTensorBindings,
}

#[derive(Clone, Debug)]
pub struct Qwen4xGRWeightBindings {
    pub norm: String,
    pub down: String,
    pub up: String,
    pub write: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Qwen4xLayerWeightBindings {
    pub attention: Qwen4xAttentionWeightBindings,
    pub attention_residual: Qwen4xGRWeightBindings,
    pub mlp: Qwen3xMoEWeightBindings,
    pub mlp_residual: Qwen4xGRWeightBindings,
    pub ple: Option<Qwen4xPLEWeightBindings>,
}

#[derive(Clone, Debug)]
pub enum Qwen4xAttentionWeightBindings {
    GDN(Qwen3xGDNWeightBindings),
    QSA {
        main: Qwen3xGQAWeightBindings,
        indexer: Qwen4xIndexerWeightBindings,
    },
}

#[derive(Clone, Debug)]
pub struct Qwen4xIndexerWeightBindings {
    pub qk: QuantizedTensorBindings,
    pub q_norm: String,
    pub k_norm: String,
}

#[derive(Clone, Debug)]
pub struct Qwen4xPLEWeightBindings {
    pub key: QuantizedTensorBindings,
    pub value: QuantizedTensorBindings,
    pub key_norm: String,
    pub query_norm: String,
    pub conv_norm: String,
    pub conv: String,
    pub table: Vec<QuantizedTensorBindings>,
}

impl Qwen4xGRWeightBindings {
    fn new(prefix: &str, write: bool) -> Self {
        Self {
            norm: format!("{prefix}.hc_norm.weight"),
            down: format!("{prefix}.input_mix_weight_down.weight"),
            up: format!("{prefix}.input_mix_weight_up.weight"),
            write: write.then(|| format!("{prefix}.block_inject_weight.weight")),
        }
    }
    pub fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        names.extend([self.norm.as_str(), self.down.as_str(), self.up.as_str()]);
        if let Some(write) = &self.write {
            names.push(write);
        }
    }
}

impl Qwen4xPLEWeightBindings {
    pub fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        push_quantized_tensor_names(&self.key, names);
        push_quantized_tensor_names(&self.value, names);
        names.extend([
            self.key_norm.as_str(),
            self.query_norm.as_str(),
            self.conv_norm.as_str(),
            self.conv.as_str(),
        ]);
        for shard in &self.table {
            push_quantized_tensor_names(shard, names);
        }
    }
}

impl Qwen4xModelWeightBindings {
    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        push_quantized_tensor_names(&self.embed, &mut names);
        push_quantized_tensor_names(&self.unembed, &mut names);
        self.mixer.push_tensor_names(&mut names);
        for layer in &self.layers {
            layer.attention_residual.push_tensor_names(&mut names);
            layer.mlp_residual.push_tensor_names(&mut names);
            layer.mlp.push_tensor_names(&mut names);
            match &layer.attention {
                Qwen4xAttentionWeightBindings::GDN(bindings) => bindings.push_tensor_names(&mut names),
                Qwen4xAttentionWeightBindings::QSA { main, indexer } => {
                    main.push_tensor_names(&mut names);
                    push_quantized_tensor_names(&indexer.qk, &mut names);
                    names.extend([indexer.q_norm.as_str(), indexer.k_norm.as_str()]);
                },
            }
            if let Some(ple) = &layer.ple {
                ple.push_tensor_names(&mut names);
            }
        }
        names
    }
}

fn bind(config: &Qwen4xModelConfig, layout: TensorPathLayout, gdn_component: &str) -> Qwen4xModelWeightBindings {
    let text = &config.text_config;
    let embed = quantized_path(layout.model_path("embed_tokens"));
    let unembed = if config.tie_word_embeddings || text.tie_word_embeddings {
        embed.clone()
    } else {
        quantized_path(layout.container_path("lm_head"))
    };
    let layers = (0..text.num_hidden_layers)
        .map(|index| {
            let prefix = layout.model_path(&format!("layers.{index}"));
            let attention = match config.layer_type_at(index) {
                LayerType::GDN => {
                    let prefix = format!("{prefix}.{gdn_component}");
                    Qwen4xAttentionWeightBindings::GDN(Qwen3xGDNWeightBindings {
                        qkv: quantized(&prefix, "in_proj_qkv"),
                        a: quantized(&prefix, "in_proj_a"),
                        b: quantized(&prefix, "in_proj_b"),
                        z: quantized(&prefix, "in_proj_z"),
                        conv_weight: format!("{prefix}.conv1d.weight"),
                        norm_weight: format!("{prefix}.norm.weight"),
                        a_log: format!("{prefix}.A_log"),
                        dt_bias: format!("{prefix}.dt_bias"),
                        output: quantized(&prefix, "out_proj"),
                    })
                },
                LayerType::QSA => {
                    let prefix = format!("{prefix}.self_attn");
                    Qwen4xAttentionWeightBindings::QSA {
                        main: Qwen3xGQAWeightBindings {
                            q: quantized(&prefix, "q_proj"),
                            k: quantized(&prefix, "k_proj"),
                            v: quantized(&prefix, "v_proj"),
                            q_norm_weight: format!("{prefix}.q_norm.weight"),
                            k_norm_weight: format!("{prefix}.k_norm.weight"),
                            output: quantized(&prefix, "o_proj"),
                        },
                        indexer: Qwen4xIndexerWeightBindings {
                            qk: quantized(&prefix, "indexer.index_qk_proj"),
                            q_norm: format!("{prefix}.indexer.q_layernorm.weight"),
                            k_norm: format!("{prefix}.indexer.k_layernorm.weight"),
                        },
                    }
                },
            };
            let mlp = format!("{prefix}.mlp");
            let ple = text.ple_layer_ids.contains(&(index + 1)).then(|| {
                let prefix = format!("{prefix}.ple");
                Qwen4xPLEWeightBindings {
                    key: quantized(&prefix, "key_proj"),
                    value: quantized(&prefix, "value_proj"),
                    key_norm: format!("{prefix}.norm_key.weight"),
                    query_norm: format!("{prefix}.norm_query.weight"),
                    conv_norm: format!("{prefix}.norm_conv.weight"),
                    conv: format!("{prefix}.conv1d.weight"),
                    table: (0..text.split_ngram_parts)
                        .map(|shard| quantized(&prefix, &format!("ple_embedding.ngram_embedding.shards.{shard}")))
                        .collect(),
                }
            });
            Qwen4xLayerWeightBindings {
                attention,
                attention_residual: Qwen4xGRWeightBindings::new(&format!("{prefix}.attn_hyper_connection"), true),
                mlp_residual: Qwen4xGRWeightBindings::new(&format!("{prefix}.mlp_hyper_connection"), true),
                mlp: Qwen3xMoEWeightBindings {
                    router: quantized(&mlp, "gate"),
                    experts: Qwen3xSparseExpertWeightBindings {
                        gate: quantized(&mlp, "switch_mlp.gate_proj"),
                        up: quantized(&mlp, "switch_mlp.up_proj"),
                        down: quantized(&mlp, "switch_mlp.down_proj"),
                    },
                    shared_expert_gate: (text.shared_expert_intermediate_size > 0)
                        .then(|| quantized(&mlp, "shared_expert_gate")),
                    shared_expert: (text.shared_expert_intermediate_size > 0)
                        .then(|| dense_mlp_bindings(&format!("{mlp}.shared_expert"))),
                },
                ple,
            }
        })
        .collect();
    Qwen4xModelWeightBindings {
        embed,
        layers,
        mixer: Qwen4xGRWeightBindings::new(&layout.model_path("hyper_connection_mixer"), false),
        unembed,
    }
}

pub fn resolve_qwen4x_model_weight_bindings<'a>(
    config: &Qwen4xModelConfig,
    tensor_names: impl IntoIterator<Item = &'a str>,
) -> Result<Qwen4xModelWeightBindings, ModelExecutorError> {
    let available: HashSet<_> = tensor_names.into_iter().collect();
    for layout in tensor_path_layout_candidates() {
        for gdn_component in ["linear_attn", "gated_delta_net"] {
            let bindings = bind(config, layout, gdn_component);
            if bindings.tensor_names().iter().all(|name| available.contains(name)) {
                return Ok(bindings);
            }
        }
    }
    Err(ModelExecutorError::custom(
        "Qwen4x checkpoint does not contain one complete supported tensor layout",
    ))
}

#[cfg(test)]
#[path = "weight_layout_test.rs"]
mod tests;
