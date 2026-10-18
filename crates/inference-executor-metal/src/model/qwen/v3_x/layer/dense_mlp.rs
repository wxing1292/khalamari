use std::rc::Rc;

use inference_backend_metal::components::dense_mlp;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::TensorMap;
use inference_executor_core::checkpoint::remove_tensor;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xDenseMLPWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::mlp::dense::backend::DenseMLP;
use crate::mlp::dense::backend::DenseMLPInput;
use crate::mlp::dense::backend::DenseMLPMetalConfig;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::qwen::v3_x::weight::concat_bytes;
use crate::model::qwen::v3_x::weight::remove_quant_weight;
use crate::model::qwen::v3_x::weight::to_u32;
use crate::model::qwen::v3_x::weight::validate_len;

pub struct Qwen3xDenseMLP {
    core: DenseMLPCore,
    metal: DenseMLPMetalConfig,
    backend: DenseMLP,
    weights: Option<DenseMLPWeightBuffers>,
    scratch: Rc<DenseMLPScratch>,
}

impl Qwen3xDenseMLP {
    pub fn new(device: &Device, core: DenseMLPCore, metal: DenseMLPMetalConfig, scratch: Rc<DenseMLPScratch>) -> Self {
        Self {
            backend: DenseMLP::new(device, core.clone(), metal),
            core,
            metal,
            weights: None,
            scratch,
        }
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: Qwen3xDenseMLPWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        assert!(self.weights.is_none(), "Qwen3.x dense MLP weights are already loaded");
        self.weights = Some(DenseMLPWeightBuffers::load(
            device, store, &bindings, &self.core, self.metal,
        )?);
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        assert!(self.weights.is_some(), "Qwen3.x dense MLP weights are not loaded");
        self.weights.take();
    }

    fn weights(&self) -> &DenseMLPWeightBuffers {
        self.weights
            .as_ref()
            .expect("Qwen3.x dense MLP weights must be loaded before execution")
    }

    pub fn replay_topology(&self, num_total_tokens: u32) -> dense_mlp::ReplayTopology {
        self.backend.replay_topology(num_total_tokens)
    }

    pub fn replay_topology_boundaries(&self) -> Box<[u32]> {
        self.backend.replay_topology_boundaries()
    }

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        input: &'a Buffer,
        output: &'a Buffer,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let _ = <DenseMLP as ReplayLayer>::record(
            &self.backend,
            recorder,
            DenseMLPInput {
                num_total_tokens,
                num_active_tokens,
                hidden_state: input,
                next_hidden_state: output,
                scratch: self.scratch.bindings(),
                weights: self.weights().as_borrowed(),
            },
        );
    }
}

// Public only inside the private `dense_mlp` module path so the sibling MoE
// owner can reuse the identical shared-expert tensor layout.
pub struct DenseMLPWeightBuffers {
    gate_up_weight: Buffer,
    gate_up_scales: Buffer,
    gate_up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

impl DenseMLPWeightBuffers {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen3xDenseMLPWeightBindings,
        core: &DenseMLPCore,
        metal: DenseMLPMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        let mut tensor_names = Vec::new();
        bindings.push_tensor_names(&mut tensor_names);
        let mut tensors = store.load_tensors(tensor_names)?;
        let weights = Self::from_tensors(device, &mut tensors, bindings, core, metal)?;
        assert!(tensors.is_empty(), "dense MLP must consume its tensor map");
        Ok(weights)
    }

    pub fn from_tensors(
        device: &Device,
        tensors: &mut TensorMap,
        bindings: &Qwen3xDenseMLPWeightBindings,
        core: &DenseMLPCore,
        metal: DenseMLPMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        core.validate();
        metal.validate();
        let gate_weight = remove_quant_weight(tensors, &bindings.gate.weight)?;
        let up_weight = remove_quant_weight(tensors, &bindings.up.weight)?;
        let gate_up_dtype = safetensors_dtype(metal.gate_up.scale_bias_dtype);
        let down_dtype = safetensors_dtype(metal.down.scale_bias_dtype);
        let gate_scales = remove_tensor(tensors, &bindings.gate.scales, gate_up_dtype)?.into_data();
        let up_scales = remove_tensor(tensors, &bindings.up.scales, gate_up_dtype)?.into_data();
        let gate_biases = remove_tensor(tensors, &bindings.gate.biases, gate_up_dtype)?.into_data();
        let up_biases = remove_tensor(tensors, &bindings.up.biases, gate_up_dtype)?.into_data();
        let down_weight = remove_quant_weight(tensors, &bindings.down.weight)?;
        let down_scales = remove_tensor(tensors, &bindings.down.scales, down_dtype)?.into_data();
        let down_biases = remove_tensor(tensors, &bindings.down.biases, down_dtype)?.into_data();

        let config = inference_backend_metal::components::dense_mlp::Config {
            hidden_dim: to_u32("dense hidden_dim", core.hidden_dim)?,
            intermediate_dim: to_u32("dense intermediate_dim", core.intermediate_dim)?,
            gate_up_group_size: metal.gate_up.group_size,
            gate_up_bits: metal.gate_up.bits,
            gate_up_scale_bias_dtype: metal.gate_up.scale_bias_dtype,
            down_group_size: metal.down.group_size,
            down_bits: metal.down.bits,
            down_scale_bias_dtype: metal.down.scale_bias_dtype,
            dtype: metal.io_dtype,
        };
        let gate_up_config = config.gate_up_config();
        let down_config = config.down_config();
        let gate_up_weight = concat_bytes(&[&gate_weight, &up_weight]);
        let gate_up_scales = concat_bytes(&[&gate_scales, &up_scales]);
        let gate_up_biases = concat_bytes(&[&gate_biases, &up_biases]);
        validate_len(
            "dense gate_up weight",
            gate_up_weight.len(),
            gate_up_config.weight_bytes(),
        )?;
        validate_len(
            "dense gate_up scales",
            gate_up_scales.len(),
            gate_up_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "dense gate_up biases",
            gate_up_biases.len(),
            gate_up_config.scale_or_bias_bytes(),
        )?;
        validate_len("dense down weight", down_weight.len(), down_config.weight_bytes())?;
        validate_len(
            "dense down scales",
            down_scales.len(),
            down_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "dense down biases",
            down_biases.len(),
            down_config.scale_or_bias_bytes(),
        )?;
        Ok(Self {
            gate_up_weight: Buffer::from_slice(device, &gate_up_weight),
            gate_up_scales: Buffer::from_slice(device, &gate_up_scales),
            gate_up_biases: Buffer::from_slice(device, &gate_up_biases),
            down_weight: Buffer::from_slice(device, &down_weight),
            down_scales: Buffer::from_slice(device, &down_scales),
            down_biases: Buffer::from_slice(device, &down_biases),
        })
    }

    pub fn as_borrowed(&self) -> dense_mlp::Weights<'_> {
        dense_mlp::Weights {
            gate_up_weight: &self.gate_up_weight,
            gate_up_scales: &self.gate_up_scales,
            gate_up_biases: &self.gate_up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }
}

fn safetensors_dtype(dtype: inference_backend_metal::metal::Dtype) -> safetensors::Dtype {
    match dtype {
        inference_backend_metal::metal::Dtype::Bfloat16 => safetensors::Dtype::BF16,
        inference_backend_metal::metal::Dtype::Float32 => safetensors::Dtype::F32,
        dtype => panic!("unsupported dense MLP affine parameter dtype {dtype:?}"),
    }
}
