use inference_backend_metal::components::gated_residual;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GQACore;
use inference_executor_core::attn::qsa_indexer::core::QSAIndexerCore;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v4_x::Qwen4xModelConfig;
use inference_executor_core::model::qwen::v4_x::weight_layout::Qwen4xGRWeightBindings;
use inference_executor_core::model::qwen::v4_x::weight_layout::Qwen4xIndexerWeightBindings;

use crate::attn::qsa_indexer::backend::QSAIndexerWeights;
use crate::checkpoint::SafeTensorStore;
use crate::def::quantized_affine::QuantizedAffineLayout;
use crate::def::quantized_affine::QuantizedAffineWeights;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::qwen::v3_x::weight::validate_len;
use crate::model::qwen::v3_x::weight::validate_shape;
use crate::model::qwen::v4_x::component_config::gr_core;
use crate::model::qwen::v4_x::component_config::uniform_layout;

pub struct AffineWeights {
    pub weight: Buffer,
    pub scales: Buffer,
    pub biases: Buffer,
}

impl AffineWeights {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &QuantizedTensorBindings,
        layout: QuantizedAffineLayout,
        output_dim: usize,
        input_dim: usize,
    ) -> Result<Self, ModelExecutorError> {
        let config = layout.config(output_dim, input_dim, Dtype::Bfloat16);
        config.validate();
        let load = |store: &mut SafeTensorStore, name: &str, dtype, expected| -> Result<Buffer, ModelExecutorError> {
            let tensor = store.tensor_bytes(name, dtype)?;
            validate_len(name, tensor.data().len(), expected)?;
            Ok(Buffer::from_slice(device, tensor.data()))
        };
        Ok(Self {
            weight: load(store, &bindings.weight, safetensors::Dtype::U32, config.weight_bytes())?,
            scales: load(
                store,
                &bindings.scales,
                safetensors::Dtype::BF16,
                config.scale_or_bias_bytes(),
            )?,
            biases: load(
                store,
                &bindings.biases,
                safetensors::Dtype::BF16,
                config.scale_or_bias_bytes(),
            )?,
        })
    }
    pub fn bindings(&self) -> QuantizedAffineWeights<'_> {
        QuantizedAffineWeights::new(&self.weight, &self.scales, &self.biases)
    }
}

pub struct GRWeights {
    norm: Buffer,
    down: Buffer,
    up: Buffer,
    pub write: Option<Buffer>,
}
impl GRWeights {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen4xModelConfig,
        bindings: &Qwen4xGRWeightBindings,
    ) -> Result<Self, ModelExecutorError> {
        let core = gr_core(config);
        let load = |store: &mut SafeTensorStore, name: &str, shape: &[usize]| {
            load_qwen3x_norm_weight(device, store, name, shape)
        };
        Ok(Self {
            norm: load(store, &bindings.norm, &[core.residual_dim()])?,
            down: load(store, &bindings.down, &[core.low_rank_dim, core.residual_dim()])?,
            up: load(store, &bindings.up, &[core.residual_dim(), core.low_rank_dim])?,
            write: bindings
                .write
                .as_ref()
                .map(|name| load(store, name, &[core.num_branches, core.residual_dim()]))
                .transpose()?,
        })
    }
    pub fn read(&self) -> gated_residual::ReadWeights<'_> {
        gated_residual::ReadWeights {
            norm: &self.norm,
            down: &self.down,
            up: &self.up,
        }
    }
}

pub struct QSAWeights {
    qgkv: AffineWeights,
    output: AffineWeights,
    indexer: AffineWeights,
    q_norm: Buffer,
    k_norm: Buffer,
    indexer_q_norm: Buffer,
    indexer_k_norm: Buffer,
}
impl QSAWeights {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen4xModelConfig,
        core: &GQACore,
        index_core: QSAIndexerCore,
        bindings: &inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings,
        index: &Qwen4xIndexerWeightBindings,
    ) -> Result<Self, ModelExecutorError> {
        let layout = uniform_layout(
            config,
            &[
                &bindings.q.weight,
                &bindings.k.weight,
                &bindings.v.weight,
                &bindings.output.weight,
            ],
            "Qwen4x QSA",
        )?;
        let projections = [
            (
                &bindings.q,
                layout.config(core.q_dim() * 2, core.hidden_dim, Dtype::Bfloat16),
            ),
            (
                &bindings.k,
                layout.config(core.k_dim(), core.hidden_dim, Dtype::Bfloat16),
            ),
            (
                &bindings.v,
                layout.config(core.v_dim(), core.hidden_dim, Dtype::Bfloat16),
            ),
        ];
        for (_, config) in &projections {
            config.validate();
        }
        let qgkv = AffineWeights {
            weight: join_tensors(
                device,
                store,
                safetensors::Dtype::U32,
                projections.iter().map(|(b, c)| (b.weight.as_str(), c.weight_bytes())),
            )?,
            scales: join_tensors(
                device,
                store,
                safetensors::Dtype::BF16,
                projections
                    .iter()
                    .map(|(b, c)| (b.scales.as_str(), c.scale_or_bias_bytes())),
            )?,
            biases: join_tensors(
                device,
                store,
                safetensors::Dtype::BF16,
                projections
                    .iter()
                    .map(|(b, c)| (b.biases.as_str(), c.scale_or_bias_bytes())),
            )?,
        };
        let norm = |store: &mut SafeTensorStore, name: &str, width: usize| -> Result<Buffer, ModelExecutorError> {
            let data = store.tensor_bytes(name, safetensors::Dtype::BF16)?;
            validate_shape(name, data.shape(), &[width])?;
            let values: Vec<f32> = data
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from(half::bf16::from_bits(u16::from_ne_bytes([b[0], b[1]]))) + 1.0)
                .collect();
            Ok(Buffer::from_slice(device, &values))
        };
        // Main RMSNorm/RoPE consumes effective F32 scales. The indexer consumes raw BF16 offsets.
        Ok(Self {
            qgkv,
            output: AffineWeights::load(device, store, &bindings.output, layout, core.hidden_dim, core.q_dim())?,
            indexer: AffineWeights::load(
                device,
                store,
                &index.qk,
                uniform_layout(config, &[&index.qk.weight], "Qwen4x indexer")?,
                index_core.qk_dim(),
                index_core.hidden_dim,
            )?,
            q_norm: norm(store, &bindings.q_norm_weight, core.head_dim)?,
            k_norm: norm(store, &bindings.k_norm_weight, core.head_dim)?,
            indexer_q_norm: load_qwen3x_norm_weight(device, store, &index.q_norm, &[index_core.head_dim])?,
            indexer_k_norm: load_qwen3x_norm_weight(device, store, &index.k_norm, &[index_core.head_dim])?,
        })
    }
    pub fn bindings(&self) -> crate::attn::qsa::backend::QSAWeights<'_> {
        crate::attn::qsa::backend::QSAWeights {
            qgkv_weight: &self.qgkv.weight,
            qgkv_scales: &self.qgkv.scales,
            qgkv_biases: &self.qgkv.biases,
            q_norm_scale: &self.q_norm,
            k_norm_scale: &self.k_norm,
            output_weight: &self.output.weight,
            output_scales: &self.output.scales,
            output_biases: &self.output.biases,
            indexer: QSAIndexerWeights {
                qk_weight: &self.indexer.weight,
                qk_scales: &self.indexer.scales,
                qk_biases: &self.indexer.biases,
                q_norm: &self.indexer_q_norm,
                k_norm: &self.indexer_k_norm,
            },
        }
    }
}

fn join_tensors<'a>(
    device: &Device,
    store: &mut SafeTensorStore,
    dtype: safetensors::Dtype,
    parts: impl Iterator<Item = (&'a str, usize)> + Clone,
) -> Result<Buffer, ModelExecutorError> {
    let len_bytes = parts
        .clone()
        .try_fold(0usize, |total_bytes, (_, part_bytes)| {
            total_bytes.checked_add(part_bytes)
        })
        .expect("joined weight bytes must fit usize");
    let output = Buffer::new_uninit(device, len_bytes);
    let mut offset_bytes = 0;
    for (name, part_bytes) in parts {
        let tensor = store.tensor_bytes(name, dtype)?;
        validate_len(name, tensor.data().len(), part_bytes)?;
        output.write_bytes(offset_bytes, tensor.data());
        offset_bytes += part_bytes;
    }
    Ok(output)
}
