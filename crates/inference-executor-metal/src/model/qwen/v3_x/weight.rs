use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::checkpoint::TensorMap;
use inference_executor_core::checkpoint::remove_tensor;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::QuantizationConfig;
use inference_executor_core::model::qwen::v3_x::ResolvedQuantizationConfig;

use crate::checkpoint::SafeTensorStore;
use crate::mlp::moe::backend::GatedMoEMetalConfig;

pub fn quant_weight(store: &mut SafeTensorStore, name: &str) -> Result<Vec<u8>, ModelExecutorError> {
    Ok(store.tensor_bytes(name, safetensors::Dtype::U32)?.into_data())
}

pub fn remove_quant_weight(tensors: &mut TensorMap, name: &str) -> Result<Vec<u8>, ModelExecutorError> {
    Ok(remove_tensor(tensors, name, safetensors::Dtype::U32)?.into_data())
}

pub fn affine_parameter_safetensors_dtype(dtype: Dtype) -> safetensors::Dtype {
    match dtype {
        Dtype::Bfloat16 => safetensors::Dtype::BF16,
        Dtype::Float32 => safetensors::Dtype::F32,
        dtype => panic!("unsupported affine parameter dtype {dtype:?}"),
    }
}

pub fn load_qwen3x_norm_weight(
    device: &Device,
    store: &mut SafeTensorStore,
    name: &str,
    expected_shape: &[usize],
) -> Result<Buffer, ModelExecutorError> {
    let data = store.tensor_bytes(name, safetensors::Dtype::BF16)?;
    validate_shape(name, data.shape(), expected_shape)?;
    Ok(Buffer::from_slice(device, data.data()))
}

pub fn remove_qwen3x_norm_weight(
    device: &Device,
    tensors: &mut TensorMap,
    name: &str,
    expected_shape: &[usize],
) -> Result<Buffer, ModelExecutorError> {
    remove_norm_weight(device, tensors, name, expected_shape, Dtype::Bfloat16)
}

pub fn remove_norm_weight(
    device: &Device,
    tensors: &mut TensorMap,
    name: &str,
    expected_shape: &[usize],
    dtype: Dtype,
) -> Result<Buffer, ModelExecutorError> {
    let dtype = match dtype {
        Dtype::Bfloat16 => safetensors::Dtype::BF16,
        Dtype::Float32 => safetensors::Dtype::F32,
        dtype => panic!("unsupported norm weight dtype {dtype:?}"),
    };
    let data = remove_tensor(tensors, name, dtype)?;
    validate_shape(name, data.shape(), expected_shape)?;
    Ok(Buffer::from_slice(device, data.data()))
}

pub fn concat_bytes(parts: &[&[u8]]) -> Vec<u8> {
    let len_bytes = parts.iter().map(|part| part.len()).sum();
    let mut out = Vec::with_capacity(len_bytes);
    for part in parts {
        out.extend_from_slice(part);
    }
    out
}

pub fn resolve_uniform_quantization(
    config: &QuantizationConfig,
    tensor_names: &[&str],
    component: &str,
) -> Result<ResolvedQuantizationConfig, ModelExecutorError> {
    let first_name = tensor_names
        .first()
        .expect("uniform quantization requires tensor names");
    let first = config.resolve_for_tensor(first_name);
    if let Some((tensor_name, resolved)) = tensor_names.iter().skip(1).find_map(|&tensor_name| {
        let resolved = config.resolve_for_tensor(tensor_name);
        (resolved != first).then_some((tensor_name, resolved))
    }) {
        return Err(ModelExecutorError::custom(format!(
            "{component} requires one affine layout, but {first_name:?} uses group_size={} bits={} mode={:?} and \
             {tensor_name:?} uses group_size={} bits={} mode={:?}",
            first.group_size, first.bits, first.mode, resolved.group_size, resolved.bits, resolved.mode,
        )));
    }
    Ok(first)
}

pub fn affine_config(
    n: usize,
    k: usize,
    group_size: u32,
    bits: u32,
    input_dtype: Dtype,
    output_dtype: Dtype,
    scale_bias_dtype: Dtype,
) -> affine_quantized::Config {
    affine_quantized::Config {
        n: n.try_into().expect("affine n must fit i32"),
        k: k.try_into().expect("affine k must fit i32"),
        group_size: group_size.try_into().expect("affine group_size must fit i32"),
        bits: bits.try_into().expect("affine bits must fit i32"),
        input_dtype,
        output_dtype,
        scale_bias_dtype,
    }
}

pub fn sparse_affine_layout(
    experts: usize,
    output_dim: usize,
    input_dim: usize,
    metal: GatedMoEMetalConfig,
) -> SparseAffineLayout {
    SparseAffineLayout {
        experts,
        output_dim,
        input_dim,
        group_size: metal.group_size as usize,
        bits: metal.bits as usize,
        scale_bias_dtype: metal.io_dtype,
    }
}

pub struct SparseAffineLayout {
    experts: usize,
    output_dim: usize,
    input_dim: usize,
    group_size: usize,
    bits: usize,
    scale_bias_dtype: Dtype,
}

impl SparseAffineLayout {
    pub fn weight_bytes(&self) -> usize {
        self.experts * self.output_dim * (self.input_dim * self.bits / 32) * std::mem::size_of::<u32>()
    }

    pub fn scale_or_bias_bytes(&self) -> usize {
        self.experts * self.output_dim * (self.input_dim / self.group_size) * self.scale_bias_dtype.item_size()
    }
}

pub fn validate_len(name: &str, actual: usize, expected: usize) -> Result<(), ModelExecutorError> {
    if actual != expected {
        return Err(ModelExecutorError::custom(format!(
            "{name} byte length mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

pub fn validate_shape(name: &str, actual: &[usize], expected: &[usize]) -> Result<(), ModelExecutorError> {
    if actual != expected {
        return Err(ModelExecutorError::custom(format!(
            "{name} shape mismatch: expected {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

pub fn to_u32(name: &str, value: usize) -> Result<u32, ModelExecutorError> {
    value
        .try_into()
        .map_err(|_| ModelExecutorError::custom(format!("{name}={value} must fit u32")))
}
