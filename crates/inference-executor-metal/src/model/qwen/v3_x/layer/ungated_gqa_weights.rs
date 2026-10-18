use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::checkpoint::TensorMap;
use inference_executor_core::checkpoint::remove_tensor;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;

use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::ungated_backend::UngatedGQAWeights;
use crate::checkpoint::SafeTensorStore;
use crate::model::qwen::v3_x::weight::affine_config;
use crate::model::qwen::v3_x::weight::concat_bytes;
use crate::model::qwen::v3_x::weight::remove_quant_weight;
use crate::model::qwen::v3_x::weight::remove_qwen3x_norm_weight;
use crate::model::qwen::v3_x::weight::validate_len;

pub struct Qwen3xUngatedGQAWeightBuffers {
    qkv_weight: Buffer,
    qkv_scales: Buffer,
    qkv_biases: Buffer,
    q_norm_weight: Buffer,
    k_norm_weight: Buffer,
    output_weight: Buffer,
    output_scales: Buffer,
    output_biases: Buffer,
}

impl Qwen3xUngatedGQAWeightBuffers {
    pub fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen3xGQAWeightBindings,
        core: &UngatedGQACore,
        metal: GQAMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        let mut tensor_names = Vec::new();
        bindings.push_tensor_names(&mut tensor_names);
        let mut tensors = store.load_tensors(tensor_names)?;
        let weights = Self::from_tensors(device, &mut tensors, bindings, core, metal)?;
        assert!(tensors.is_empty(), "ungated GQA must consume its tensor map");
        Ok(weights)
    }

    fn from_tensors(
        device: &Device,
        tensors: &mut TensorMap,
        bindings: &Qwen3xGQAWeightBindings,
        core: &UngatedGQACore,
        metal: GQAMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        core.validate();
        metal.validate();
        let q_weight = remove_quant_weight(tensors, &bindings.q.weight)?;
        let k_weight = remove_quant_weight(tensors, &bindings.k.weight)?;
        let v_weight = remove_quant_weight(tensors, &bindings.v.weight)?;
        let q_scales = remove_tensor(tensors, &bindings.q.scales, safetensors::Dtype::BF16)?.into_data();
        let k_scales = remove_tensor(tensors, &bindings.k.scales, safetensors::Dtype::BF16)?.into_data();
        let v_scales = remove_tensor(tensors, &bindings.v.scales, safetensors::Dtype::BF16)?.into_data();
        let q_biases = remove_tensor(tensors, &bindings.q.biases, safetensors::Dtype::BF16)?.into_data();
        let k_biases = remove_tensor(tensors, &bindings.k.biases, safetensors::Dtype::BF16)?.into_data();
        let v_biases = remove_tensor(tensors, &bindings.v.biases, safetensors::Dtype::BF16)?.into_data();
        let qkv_weight = concat_bytes(&[&q_weight, &k_weight, &v_weight]);
        let qkv_scales = concat_bytes(&[&q_scales, &k_scales, &v_scales]);
        let qkv_biases = concat_bytes(&[&q_biases, &k_biases, &v_biases]);
        let qkv_config = affine_config(
            core.qkv_dim(),
            core.hidden_dim,
            metal.group_size,
            metal.bits,
            metal.io_dtype,
            metal.io_dtype,
            metal.io_dtype,
        );
        validate_len("ungated GQA qkv weight", qkv_weight.len(), qkv_config.weight_bytes())?;
        validate_len(
            "ungated GQA qkv scales",
            qkv_scales.len(),
            qkv_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "ungated GQA qkv biases",
            qkv_biases.len(),
            qkv_config.scale_or_bias_bytes(),
        )?;

        let output_config = affine_config(
            core.hidden_dim,
            core.q_dim(),
            metal.group_size,
            metal.bits,
            metal.io_dtype,
            metal.io_dtype,
            metal.io_dtype,
        );
        let output_weight = remove_quant_weight(tensors, &bindings.output.weight)?;
        let output_scales = remove_tensor(tensors, &bindings.output.scales, safetensors::Dtype::BF16)?.into_data();
        let output_biases = remove_tensor(tensors, &bindings.output.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len(
            "ungated GQA output weight",
            output_weight.len(),
            output_config.weight_bytes(),
        )?;
        validate_len(
            "ungated GQA output scales",
            output_scales.len(),
            output_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "ungated GQA output biases",
            output_biases.len(),
            output_config.scale_or_bias_bytes(),
        )?;

        Ok(Self {
            qkv_weight: Buffer::from_slice(device, &qkv_weight),
            qkv_scales: Buffer::from_slice(device, &qkv_scales),
            qkv_biases: Buffer::from_slice(device, &qkv_biases),
            q_norm_weight: remove_qwen3x_norm_weight(device, tensors, &bindings.q_norm_weight, &[core.head_dim])?,
            k_norm_weight: remove_qwen3x_norm_weight(device, tensors, &bindings.k_norm_weight, &[core.head_dim])?,
            output_weight: Buffer::from_slice(device, &output_weight),
            output_scales: Buffer::from_slice(device, &output_scales),
            output_biases: Buffer::from_slice(device, &output_biases),
        })
    }

    pub fn as_borrowed(&self) -> UngatedGQAWeights<'_> {
        UngatedGQAWeights {
            qkv_weight: &self.qkv_weight,
            qkv_scales: &self.qkv_scales,
            qkv_biases: &self.qkv_biases,
            q_norm_weight: &self.q_norm_weight,
            k_norm_weight: &self.k_norm_weight,
            output_weight: &self.output_weight,
            output_scales: &self.output_scales,
            output_biases: &self.output_biases,
        }
    }
}
