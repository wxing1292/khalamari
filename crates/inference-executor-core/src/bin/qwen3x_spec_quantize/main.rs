use std::error::Error;
use std::ffi::OsString;
use std::path::PathBuf;

mod checkpoint;
mod dflash2;

use checkpoint::Result;
use checkpoint::error;

#[derive(Debug)]
struct Args {
    input_dir: PathBuf,
    output_dir: PathBuf,
    group_size: usize,
    bits: usize,
    model_bits: usize,
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!("{}", usage());
        return Ok(());
    }
    let args = Args::parse(arguments)?;
    dflash2::convert(
        &args.input_dir,
        &args.output_dir,
        args.group_size,
        args.bits,
        args.model_bits,
    )?;
    Ok(())
}

impl Args {
    fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self> {
        let mut args = args.into_iter();
        match args
            .next()
            .ok_or_else(|| error(format!("missing model subcommand\n{}", usage())))?
            .to_str()
        {
            Some("dflash2") => {},
            Some(name) => return Err(error(format!("unknown model subcommand {name:?}\n{}", usage()))),
            None => return Err(error("model subcommand is not valid UTF-8")),
        };
        let mut input_dir = None;
        let mut output_dir = None;
        let mut group_size = 64;
        let mut bits = 4;
        let mut model_bits = 6;
        while let Some(argument) = args.next() {
            let name = argument
                .to_str()
                .ok_or_else(|| error(format!("argument {argument:?} is not valid UTF-8")))?;
            let value = args
                .next()
                .ok_or_else(|| error(format!("missing value for {name}\n{}", usage())))?;
            match name {
                "--input-dir" => input_dir = Some(PathBuf::from(value)),
                "--output-dir" => output_dir = Some(PathBuf::from(value)),
                "--group-size" => group_size = parse_usize(name, &value)?,
                "--bits" => bits = parse_usize(name, &value)?,
                "--high-bits" => model_bits = parse_usize(name, &value)?,
                _ => return Err(error(format!("unknown argument {name:?}\n{}", usage()))),
            }
        }
        Ok(Self {
            input_dir: input_dir.ok_or_else(|| error(format!("missing --input-dir\n{}", usage())))?,
            output_dir: output_dir.ok_or_else(|| error(format!("missing --output-dir\n{}", usage())))?,
            group_size,
            bits,
            model_bits,
        })
    }
}

fn parse_usize(name: &str, value: &std::ffi::OsStr) -> Result<usize> {
    value
        .to_str()
        .ok_or_else(|| error(format!("value for {name} is not valid UTF-8")))?
        .parse::<usize>()
        .map_err(|err| error(format!("invalid integer for {name}: {err}")))
}

fn usage() -> &'static str {
    "usage:\n  qwen3x_spec_quantize dflash2 --input-dir DIR --output-dir DIR [--group-size 64] [--bits 4] [--high-bits \
     6]"
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;

    use half::bf16;
    use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2Config;
    use inference_executor_core::model::qwen::v3_x::dflash2::init_qwen3x_dflash2_config;
    use safetensors::Dtype;
    use safetensors::SafeTensors;
    use safetensors::tensor::View;
    use safetensors::tensor::serialize_to_file;

    use super::*;

    struct OwnedTensor {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for &OwnedTensor {
        fn dtype(&self) -> Dtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    #[test]
    fn test_dflash2_checkpoint_uses_shared_bf16_affine_contract() {
        let root = test_root("dflash2");
        let input_dir = root.join("input");
        let output_dir = root.join("output");
        std::fs::create_dir_all(&input_dir).unwrap();
        write_input(&input_dir, dflash2_config(), |input_dir| {
            let config = init_qwen3x_dflash2_config(input_dir).unwrap();
            dflash2_tensors(&config)
        });

        dflash2::convert(&input_dir, &output_dir, 64, 4, 6).unwrap();

        let bytes = std::fs::read(output_dir.join("model.safetensors")).unwrap();
        let checkpoint = SafeTensors::deserialize(&bytes).unwrap();
        assert_index_matches(&output_dir, &checkpoint);
        assert_eq!(checkpoint.tensor("fc.weight").unwrap().dtype(), Dtype::U32);
        assert_eq!(checkpoint.tensor("fc.scales").unwrap().dtype(), Dtype::BF16);
        assert_eq!(checkpoint.tensor("fc.biases").unwrap().dtype(), Dtype::BF16);
        assert_eq!(checkpoint.tensor("fc.weight").unwrap().shape(), [64, 40]);
        assert_eq!(checkpoint.tensor("fc.scales").unwrap().shape(), [64, 5]);
        assert_eq!(
            checkpoint.tensor("layers.2.self_attn.v_proj.weight").unwrap().shape(),
            [32, 12]
        );
        assert_eq!(
            checkpoint.tensor("layers.4.mlp.down_proj.weight").unwrap().shape(),
            [64, 24]
        );
        assert_eq!(
            checkpoint.tensor("layers.0.self_attn.v_proj.weight").unwrap().shape(),
            [32, 8]
        );
        assert_eq!(
            checkpoint
                .tensor("candidate_selector.predecessor_codebook.weight")
                .unwrap()
                .dtype(),
            Dtype::U32
        );
        assert_eq!(
            checkpoint
                .tensor("layers.0.attention_conv.base_kernel")
                .unwrap()
                .dtype(),
            Dtype::BF16
        );
        let output_config = read_output_config(&output_dir);
        for name in [
            "layers.2.self_attn.v_proj",
            "layers.2.mlp.down_proj",
            "layers.4.self_attn.v_proj",
            "layers.4.mlp.down_proj",
        ] {
            assert_eq!(output_config["quantization"][name]["bits"], 6);
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    fn test_root(model: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "khalamari-qwen3x-spec-quantize-{model}-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ))
    }

    fn write_input(
        input_dir: &Path,
        config: serde_json::Value,
        tensors: impl FnOnce(&Path) -> BTreeMap<String, OwnedTensor>,
    ) {
        std::fs::write(
            input_dir.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let tensors = tensors(input_dir);
        serialize_to_file(
            tensors.iter().map(|(name, tensor)| (name.as_str(), tensor)),
            None,
            &input_dir.join("model.safetensors"),
        )
        .unwrap();
    }

    fn assert_index_matches(output_dir: &Path, checkpoint: &SafeTensors<'_>) {
        let index = serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(output_dir.join("model.safetensors.index.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(index["weight_map"].as_object().unwrap().len(), checkpoint.names().len());
        assert!(
            index["weight_map"]
                .as_object()
                .unwrap()
                .values()
                .all(|file_name| file_name == "model.safetensors")
        );
    }

    fn read_output_config(output_dir: &Path) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(output_dir.join("config.json")).unwrap()).unwrap()
    }

    fn dflash2_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Qwen3DFlash2Model"],
            "model_type": "qwen3",
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 127,
            "selector_rank": 64,
            "selector_top_k": 16,
            "target_layer_ids": [0, 1, 2, 3, 4],
            "dtype": "bfloat16",
            "attention_bias": false,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 5,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_target_layers": 6,
            "head_dim": 16,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000000.0,
            "max_position_embeddings": 8192,
            "vocab_size": 128,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
            "max_window_layers": 5,
            "use_cache": true,
            "use_sliding_window": true,
            "sliding_window": 64,
            "tie_word_embeddings": false,
            "is_causal": false,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "rope_type": "default"
            }
        })
    }

    fn dflash2_tensors(config: &Qwen3xDFlash2Config) -> BTreeMap<String, OwnedTensor> {
        let mut tensors = BTreeMap::new();
        insert_matrix(
            &mut tensors,
            "fc.weight",
            config.hidden_size,
            config.hidden_size * config.num_hidden_layers,
        );
        insert_vector(&mut tensors, "hidden_norm.weight", config.hidden_size);
        insert_vector(&mut tensors, "norm.weight", config.hidden_size);
        insert_matrix(
            &mut tensors,
            "candidate_selector.hidden_projection.weight",
            config.selector_rank,
            config.hidden_size,
        );
        insert_matrix(
            &mut tensors,
            "candidate_selector.predecessor_codebook",
            config.vocab_size,
            config.selector_rank,
        );
        insert_matrix(
            &mut tensors,
            "candidate_selector.successor_codebook",
            config.vocab_size,
            config.selector_rank,
        );
        insert_layers(
            &mut tensors,
            config.num_hidden_layers,
            config.hidden_size,
            config.intermediate_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        );
        for layer in 0..config.num_hidden_layers {
            for conv in ["attention_conv", "mlp_conv"] {
                let prefix = format!("layers.{layer}.{conv}");
                insert_base_kernel(
                    &mut tensors,
                    &format!("{prefix}.base_kernel"),
                    config.conv_kernel_size,
                    config.hidden_size,
                );
                insert_matrix(
                    &mut tensors,
                    &format!("{prefix}.kernel_projection.weight"),
                    2 * config.conv_kernel_size * config.hidden_size / config.conv_group_size,
                    config.hidden_size,
                );
            }
        }
        tensors
    }

    fn insert_layers(
        tensors: &mut BTreeMap<String, OwnedTensor>,
        num_layers: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
    ) {
        for layer in 0..num_layers {
            let prefix = format!("layers.{layer}");
            insert_vector(tensors, &format!("{prefix}.input_layernorm.weight"), hidden_size);
            insert_vector(
                tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden_size,
            );
            insert_matrix(
                tensors,
                &format!("{prefix}.self_attn.q_proj.weight"),
                num_attention_heads * head_dim,
                hidden_size,
            );
            for projection in ["k_proj", "v_proj"] {
                insert_matrix(
                    tensors,
                    &format!("{prefix}.self_attn.{projection}.weight"),
                    num_key_value_heads * head_dim,
                    hidden_size,
                );
            }
            insert_matrix(
                tensors,
                &format!("{prefix}.self_attn.o_proj.weight"),
                hidden_size,
                num_attention_heads * head_dim,
            );
            insert_vector(tensors, &format!("{prefix}.self_attn.q_norm.weight"), head_dim);
            insert_vector(tensors, &format!("{prefix}.self_attn.k_norm.weight"), head_dim);
            for projection in ["gate_proj", "up_proj"] {
                insert_matrix(
                    tensors,
                    &format!("{prefix}.mlp.{projection}.weight"),
                    intermediate_size,
                    hidden_size,
                );
            }
            insert_matrix(
                tensors,
                &format!("{prefix}.mlp.down_proj.weight"),
                hidden_size,
                intermediate_size,
            );
        }
    }

    fn insert_matrix(tensors: &mut BTreeMap<String, OwnedTensor>, name: &str, rows: usize, columns: usize) {
        let values = (0..rows * columns)
            .map(|index| ((index % 97) as f32 - 48.0) / 37.0)
            .collect::<Vec<_>>();
        tensors.insert(
            name.to_string(),
            OwnedTensor {
                dtype: Dtype::BF16,
                shape: vec![rows, columns],
                data: bf16_bytes(&values),
            },
        );
    }

    fn insert_vector(tensors: &mut BTreeMap<String, OwnedTensor>, name: &str, dimension: usize) {
        tensors.insert(
            name.to_string(),
            OwnedTensor {
                dtype: Dtype::BF16,
                shape: vec![dimension],
                data: bf16_bytes(&vec![1.0; dimension]),
            },
        );
    }

    fn insert_base_kernel(tensors: &mut BTreeMap<String, OwnedTensor>, name: &str, taps: usize, hidden_size: usize) {
        tensors.insert(
            name.to_string(),
            OwnedTensor {
                dtype: Dtype::BF16,
                shape: vec![2, taps, hidden_size],
                data: bf16_bytes(&vec![0.5; 2 * taps * hidden_size]),
            },
        );
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&value| bf16::from_f32(value).to_bits().to_le_bytes())
            .collect()
    }
}
