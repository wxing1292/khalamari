use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::path::Path;

use inference_executor_core::model::qwen::v3_x::dflash2::init_qwen3x_dflash2_config;
use inference_executor_core::model::qwen::v3_x::dflash2::resolve_qwen3x_dflash2_source_weight_bindings;

use crate::checkpoint::ConversionPlan;
use crate::checkpoint::Result;
use crate::checkpoint::convert_checkpoint;
use crate::checkpoint::error;

const HIGH_BIT_TENSORS: [&str; 4] = [
    "layers.2.self_attn.v_proj.weight",
    "layers.2.mlp.down_proj.weight",
    "layers.4.self_attn.v_proj.weight",
    "layers.4.mlp.down_proj.weight",
];

pub fn convert(input_dir: &Path, output_dir: &Path, group_size: usize, bits: usize, high_bits: usize) -> Result<()> {
    convert_checkpoint(input_dir, output_dir, |input_dir, header, mut config_value| {
        let config_path = input_dir.join("config.json");
        let config = init_qwen3x_dflash2_config(input_dir)
            .map_err(|err| error(format!("invalid Qwen3x DFlash2 config {config_path:?}: {err}")))?;
        let bindings =
            resolve_qwen3x_dflash2_source_weight_bindings(&config, header.tensors.keys().map(String::as_str))
                .map_err(|err| error(format!("Qwen3x DFlash2 tensor set mismatch: {err}")))?;

        let mut bit_overrides = BTreeMap::new();
        let mut quantization = serde_json::Map::from_iter([
            ("group_size".to_string(), serde_json::Value::from(group_size)),
            ("bits".to_string(), serde_json::Value::from(bits)),
            ("mode".to_string(), serde_json::Value::from("affine")),
        ]);
        if high_bits != bits {
            for name in HIGH_BIT_TENSORS
                .into_iter()
                .filter(|name| high_bit_layer_index(name) < config.num_hidden_layers)
            {
                bit_overrides.insert(name.to_string(), high_bits);
                let config_name = name.strip_suffix(".weight").expect("high-bit tensor must be a weight");
                quantization.insert(config_name.to_string(), serde_json::json!({ "bits": high_bits }));
            }
        }
        config_value
            .as_object_mut()
            .ok_or_else(|| error("DFlash2 config root must be a JSON object"))?
            .insert("quantization".to_string(), serde_json::Value::Object(quantization));

        Ok(ConversionPlan {
            model_name: "DFlash2",
            format: "khalamari-dflash2-affine",
            source_max_rank: 3,
            group_size,
            bits,
            bit_overrides,
            unquantized_matrices: BTreeSet::new(),
            renamed_tensors: BTreeMap::from([
                (
                    "candidate_selector.predecessor_codebook".to_string(),
                    "candidate_selector.predecessor_codebook.weight".to_string(),
                ),
                (
                    "candidate_selector.successor_codebook".to_string(),
                    "candidate_selector.successor_codebook.weight".to_string(),
                ),
            ]),
            expected_output_names: bindings.tensor_names().into_iter().map(str::to_string).collect(),
            metadata: HashMap::from([("high_bits".to_string(), high_bits.to_string())]),
            config: config_value,
        })
    })
}

fn high_bit_layer_index(name: &str) -> usize {
    name.strip_prefix("layers.")
        .and_then(|name| name.split_once('.'))
        .and_then(|(layer, _)| layer.parse().ok())
        .expect("high-bit tensor must start with a numeric layer index")
}
