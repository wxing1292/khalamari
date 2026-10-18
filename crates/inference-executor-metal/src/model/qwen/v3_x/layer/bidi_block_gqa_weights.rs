use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::checkpoint::TensorMap;
use inference_executor_core::checkpoint::remove_tensor;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;

use crate::attn::bidi_block_gqa::backend::BiDiBlockGQAMetalConfig;
use crate::attn::bidi_block_gqa::backend::BiDiBlockGQAWeights;
use crate::checkpoint::SafeTensorStore;
use crate::def::quantized_affine::QuantizedAffineLayout;
use crate::def::quantized_affine::QuantizedAffineWeights;
use crate::model::qwen::v3_x::weight::affine_parameter_safetensors_dtype;
use crate::model::qwen::v3_x::weight::concat_bytes;
use crate::model::qwen::v3_x::weight::remove_norm_weight;
use crate::model::qwen::v3_x::weight::remove_quant_weight;
use crate::model::qwen::v3_x::weight::validate_len;

pub struct Qwen3xBiDiBlockGQAWeightBuffers {
    qkv_weight: Buffer,
    qkv_scales: Buffer,
    qkv_biases: Buffer,
    q_offsets: AffineOffsets,
    k_offsets: AffineOffsets,
    v_offsets: AffineOffsets,
    q_norm_weight: Buffer,
    k_norm_weight: Buffer,
    output_weight: Buffer,
    output_scales: Buffer,
    output_biases: Buffer,
}

#[derive(Clone, Copy)]
struct AffineOffsets {
    weight_offset_bytes: usize,
    scales_offset_bytes: usize,
    biases_offset_bytes: usize,
}

struct AffineBytes {
    weight: Vec<u8>,
    scales: Vec<u8>,
    biases: Vec<u8>,
}

impl Qwen3xBiDiBlockGQAWeightBuffers {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen3xGQAWeightBindings,
        core: &UngatedGQACore,
        metal: BiDiBlockGQAMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        let mut tensor_names = Vec::new();
        bindings.push_tensor_names(&mut tensor_names);
        let mut tensors = store.load_tensors(tensor_names)?;
        let weights = Self::from_tensors(device, &mut tensors, bindings, core, metal)?;
        assert!(tensors.is_empty(), "BiDiBlockGQA must consume its tensor map");
        Ok(weights)
    }

    fn from_tensors(
        device: &Device,
        tensors: &mut TensorMap,
        bindings: &Qwen3xGQAWeightBindings,
        core: &UngatedGQACore,
        metal: BiDiBlockGQAMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        core.validate();
        metal.validate();
        let q = remove_affine(tensors, &bindings.q, metal.q, core.q_dim(), core.hidden_dim)?;
        let k = remove_affine(tensors, &bindings.k, metal.k, core.k_dim(), core.hidden_dim)?;
        let v = remove_affine(tensors, &bindings.v, metal.v, core.v_dim(), core.hidden_dim)?;
        let q_offsets = AffineOffsets {
            weight_offset_bytes: 0,
            scales_offset_bytes: 0,
            biases_offset_bytes: 0,
        };
        let k_offsets = AffineOffsets {
            weight_offset_bytes: q.weight.len(),
            scales_offset_bytes: q.scales.len(),
            biases_offset_bytes: q.biases.len(),
        };
        let v_offsets = AffineOffsets {
            weight_offset_bytes: q
                .weight
                .len()
                .checked_add(k.weight.len())
                .expect("BiDiBlockGQA V weight offset must fit usize"),
            scales_offset_bytes: q
                .scales
                .len()
                .checked_add(k.scales.len())
                .expect("BiDiBlockGQA V scale offset must fit usize"),
            biases_offset_bytes: q
                .biases
                .len()
                .checked_add(k.biases.len())
                .expect("BiDiBlockGQA V bias offset must fit usize"),
        };
        let qkv_weight = concat_bytes(&[&q.weight, &k.weight, &v.weight]);
        let qkv_scales = concat_bytes(&[&q.scales, &k.scales, &v.scales]);
        let qkv_biases = concat_bytes(&[&q.biases, &k.biases, &v.biases]);
        let output = remove_affine(tensors, &bindings.output, metal.output, core.hidden_dim, core.q_dim())?;

        Ok(Self {
            qkv_weight: Buffer::from_slice(device, &qkv_weight),
            qkv_scales: Buffer::from_slice(device, &qkv_scales),
            qkv_biases: Buffer::from_slice(device, &qkv_biases),
            q_offsets,
            k_offsets,
            v_offsets,
            q_norm_weight: remove_norm_weight(
                device,
                tensors,
                &bindings.q_norm_weight,
                &[core.head_dim],
                metal.norm_weight_dtype,
            )?,
            k_norm_weight: remove_norm_weight(
                device,
                tensors,
                &bindings.k_norm_weight,
                &[core.head_dim],
                metal.norm_weight_dtype,
            )?,
            output_weight: Buffer::from_slice(device, &output.weight),
            output_scales: Buffer::from_slice(device, &output.scales),
            output_biases: Buffer::from_slice(device, &output.biases),
        })
    }

    pub fn as_borrowed(&self) -> BiDiBlockGQAWeights<'_> {
        BiDiBlockGQAWeights {
            q: self.qkv_weights(self.q_offsets),
            k: self.qkv_weights(self.k_offsets),
            v: self.qkv_weights(self.v_offsets),
            q_norm_weight: &self.q_norm_weight,
            k_norm_weight: &self.k_norm_weight,
            output: QuantizedAffineWeights::new(&self.output_weight, &self.output_scales, &self.output_biases),
        }
    }

    fn qkv_weights(&self, offsets: AffineOffsets) -> QuantizedAffineWeights<'_> {
        QuantizedAffineWeights {
            weight: &self.qkv_weight,
            weight_offset_bytes: offsets.weight_offset_bytes,
            scales: &self.qkv_scales,
            scales_offset_bytes: offsets.scales_offset_bytes,
            biases: &self.qkv_biases,
            biases_offset_bytes: offsets.biases_offset_bytes,
        }
    }
}

fn remove_affine(
    tensors: &mut TensorMap,
    bindings: &QuantizedTensorBindings,
    layout: QuantizedAffineLayout,
    output_dim: usize,
    input_dim: usize,
) -> Result<AffineBytes, ModelExecutorError> {
    let config = layout.config(output_dim, input_dim, Dtype::Bfloat16);
    let weight = remove_quant_weight(tensors, &bindings.weight)?;
    let tensor_dtype = affine_parameter_safetensors_dtype(layout.scale_bias_dtype);
    let scales = remove_tensor(tensors, &bindings.scales, tensor_dtype)?.into_data();
    let biases = remove_tensor(tensors, &bindings.biases, tensor_dtype)?.into_data();
    validate_len(
        &format!("{} weight", bindings.weight),
        weight.len(),
        config.weight_bytes(),
    )?;
    validate_len(
        &format!("{} scales", bindings.scales),
        scales.len(),
        config.scale_or_bias_bytes(),
    )?;
    validate_len(
        &format!("{} biases", bindings.biases),
        biases.len(),
        config.scale_or_bias_bytes(),
    )?;
    Ok(AffineBytes { weight, scales, biases })
}
