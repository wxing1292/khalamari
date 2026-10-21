use std::rc::Rc;

use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::sparse_mlp;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::checkpoint::TensorMap;
use inference_executor_core::checkpoint::remove_tensor;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::mlp::moe::GatedMoECore;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xMoEWeightBindings;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xSparseExpertWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::quantized_affine::QuantizedAffineLayout;
use crate::mlp::dense::backend::DenseMLPMetalConfig;
use crate::mlp::moe::backend::GatedMoE;
use crate::mlp::moe::backend::GatedMoEInput;
use crate::mlp::moe::backend::GatedMoEMetalConfig;
use crate::mlp::moe::backend::GatedMoEReplayTopology;
use crate::mlp::moe::backend::GatedMoESharedExpertsInput;
use crate::mlp::moe::backend::GatedMoESharedExpertsWeights;
use crate::mlp::moe::backend::GatedMoEWeights;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::qwen::v3_x::layer::dense_mlp::DenseMLPWeightBuffers;
use crate::model::qwen::v3_x::weight::affine_config;
use crate::model::qwen::v3_x::weight::remove_quant_weight;
use crate::model::qwen::v3_x::weight::sparse_affine_layout;
use crate::model::qwen::v3_x::weight::validate_len;

pub struct Qwen3xMoE {
    core: GatedMoECore,
    metal: GatedMoEMetalConfig,
    backend: GatedMoE,
    weights: Option<Box<Qwen3xMoEWeights>>,
    scratch: Rc<MoEScratch>,
}

impl Qwen3xMoE {
    pub fn new(device: &Device, core: GatedMoECore, metal: GatedMoEMetalConfig, scratch: Rc<MoEScratch>) -> Self {
        Self {
            backend: GatedMoE::new(device, core.clone(), metal),
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
        bindings: Qwen3xMoEWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        assert!(self.weights.is_none(), "Qwen3.x MoE weights are already loaded");
        self.weights = Some(Box::new(Qwen3xMoEWeights::load(
            device, store, &bindings, &self.core, self.metal,
        )?));
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        assert!(self.weights.is_some(), "Qwen3.x MoE weights are not loaded");
        self.weights.take();
    }

    fn weights(&self) -> &Qwen3xMoEWeights {
        self.weights
            .as_deref()
            .expect("Qwen3.x MoE weights must be loaded before execution")
    }

    pub fn replay_topology(&self, num_total_tokens: u32) -> GatedMoEReplayTopology {
        self.backend.replay_topology(num_total_tokens)
    }

    pub fn replay_topology_boundaries(&self) -> Box<[u32]> {
        self.backend.replay_topology_boundaries()
    }

    pub fn record<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        input: &'a Buffer,
        output: &'a Buffer,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
    ) {
        let weights = self.weights();
        let shared_experts = weights.shared_experts.as_ref().map(|weights| {
            GatedMoESharedExpertsInput {
                scratch: self
                    .scratch
                    .shared_experts_bindings()
                    .expect("qwen3.x shared-expert weights require shared-expert scratch"),
                weights: GatedMoESharedExpertsWeights {
                    shared_expert_gate_weight: &weights.gate_weight,
                    shared_expert_gate_scales: &weights.gate_scales,
                    shared_expert_gate_biases: &weights.gate_biases,
                    shared_experts: weights.mlp.as_borrowed(),
                },
            }
        });
        let _ = <GatedMoE as ReplayLayer>::record(
            &self.backend,
            recorder,
            GatedMoEInput {
                num_total_tokens,
                num_active_tokens,
                hidden_state: input,
                next_hidden_state: output,
                scratch: self.scratch.bindings(),
                weights: GatedMoEWeights {
                    router_weight: &weights.router_weight,
                    router_scales: &weights.router_scales,
                    router_biases: &weights.router_biases,
                    topk_experts: weights.experts.as_borrowed(),
                },
                shared_experts,
            },
        );
    }
}

struct Qwen3xMoEWeights {
    router_weight: Buffer,
    router_scales: Buffer,
    router_biases: Buffer,
    experts: Qwen3xSparseExpertWeights,
    shared_experts: Option<Qwen3xSharedExpertsWeightBuffers>,
}

struct Qwen3xSparseExpertWeights {
    gate_weight: Buffer,
    gate_scales: Buffer,
    gate_biases: Buffer,
    up_weight: Buffer,
    up_scales: Buffer,
    up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

struct Qwen3xSharedExpertsWeightBuffers {
    gate_weight: Buffer,
    gate_scales: Buffer,
    gate_biases: Buffer,
    mlp: DenseMLPWeightBuffers,
}

impl Qwen3xMoEWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen3xMoEWeightBindings,
        core: &GatedMoECore,
        metal: GatedMoEMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        let mut tensor_names = Vec::new();
        bindings.push_tensor_names(&mut tensor_names);
        let mut tensors = store.load_tensors(tensor_names)?;
        let weights = Self::from_tensors(device, &mut tensors, bindings, core, metal)?;
        assert!(tensors.is_empty(), "MoE must consume its tensor map");
        Ok(weights)
    }

    fn from_tensors(
        device: &Device,
        tensors: &mut TensorMap,
        bindings: &Qwen3xMoEWeightBindings,
        core: &GatedMoECore,
        metal: GatedMoEMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        core.validate();
        metal.validate();
        let router_config = affine_config(
            core.num_experts,
            core.hidden_dim,
            metal.group_size,
            metal.router_bits,
            metal.io_dtype,
            metal.io_dtype,
            metal.io_dtype,
        );
        let router_weight = remove_quant_weight(tensors, &bindings.router.weight)?;
        let router_scales = remove_tensor(tensors, &bindings.router.scales, safetensors::Dtype::BF16)?.into_data();
        let router_biases = remove_tensor(tensors, &bindings.router.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len(
            "sparse router weight",
            router_weight.len(),
            router_config.weight_bytes(),
        )?;
        validate_len(
            "sparse router scales",
            router_scales.len(),
            router_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "sparse router biases",
            router_biases.len(),
            router_config.scale_or_bias_bytes(),
        )?;

        let experts = Qwen3xSparseExpertWeights::from_tensors(device, tensors, &bindings.experts, core, metal)?;
        let shared_experts = if let Some(shared_experts_intermediate_dim) = core.shared_experts_intermediate_dim {
            let gate_bindings = bindings
                .shared_expert_gate
                .as_ref()
                .expect("qwen3.x shared expert geometry requires shared expert gate bindings");
            let expert_bindings = bindings
                .shared_expert
                .as_ref()
                .expect("qwen3.x shared expert geometry requires shared expert bindings");
            let gate_config = affine_config(
                1,
                core.hidden_dim,
                metal.group_size,
                metal.shared_expert_gate_bits,
                metal.io_dtype,
                metal.io_dtype,
                metal.io_dtype,
            );
            let gate_weight = remove_quant_weight(tensors, &gate_bindings.weight)?;
            let gate_scales = remove_tensor(tensors, &gate_bindings.scales, safetensors::Dtype::BF16)?.into_data();
            let gate_biases = remove_tensor(tensors, &gate_bindings.biases, safetensors::Dtype::BF16)?.into_data();
            validate_len(
                "sparse shared expert gate weight",
                gate_weight.len(),
                gate_config.weight_bytes(),
            )?;
            validate_len(
                "sparse shared expert gate scales",
                gate_scales.len(),
                gate_config.scale_or_bias_bytes(),
            )?;
            validate_len(
                "sparse shared expert gate biases",
                gate_biases.len(),
                gate_config.scale_or_bias_bytes(),
            )?;
            let shared_experts_core = DenseMLPCore {
                model_layer_index: core.model_layer_index,
                hidden_dim: core.hidden_dim,
                intermediate_dim: shared_experts_intermediate_dim,
            };
            let shared_experts_metal = DenseMLPMetalConfig {
                gate_up: QuantizedAffineLayout {
                    group_size: metal.group_size,
                    bits: metal.bits,
                    scale_bias_dtype: metal.io_dtype,
                },
                down: QuantizedAffineLayout {
                    group_size: metal.group_size,
                    bits: metal.bits,
                    scale_bias_dtype: metal.io_dtype,
                },
                io_dtype: metal.io_dtype,
            };
            Some(Qwen3xSharedExpertsWeightBuffers {
                gate_weight: Buffer::from_slice(device, &gate_weight),
                gate_scales: Buffer::from_slice(device, &gate_scales),
                gate_biases: Buffer::from_slice(device, &gate_biases),
                mlp: DenseMLPWeightBuffers::from_tensors(
                    device,
                    tensors,
                    expert_bindings,
                    &shared_experts_core,
                    shared_experts_metal,
                )?,
            })
        } else {
            assert!(
                bindings.shared_expert_gate.is_none() && bindings.shared_expert.is_none(),
                "qwen3.x MoE without shared expert geometry must not bind shared expert tensors"
            );
            None
        };
        Ok(Self {
            router_weight: Buffer::from_slice(device, &router_weight),
            router_scales: Buffer::from_slice(device, &router_scales),
            router_biases: Buffer::from_slice(device, &router_biases),
            experts,
            shared_experts,
        })
    }
}

impl Qwen3xSparseExpertWeights {
    fn from_tensors(
        device: &Device,
        tensors: &mut TensorMap,
        bindings: &Qwen3xSparseExpertWeightBindings,
        core: &GatedMoECore,
        metal: GatedMoEMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        let expert_gate_layout = sparse_affine_layout(core.num_experts, core.intermediate_dim, core.hidden_dim, metal);
        let expert_down_layout = sparse_affine_layout(core.num_experts, core.hidden_dim, core.intermediate_dim, metal);
        let gate_weight = remove_quant_weight(tensors, &bindings.gate.weight)?;
        let gate_scales = remove_tensor(tensors, &bindings.gate.scales, safetensors::Dtype::BF16)?.into_data();
        let gate_biases = remove_tensor(tensors, &bindings.gate.biases, safetensors::Dtype::BF16)?.into_data();
        let up_weight = remove_quant_weight(tensors, &bindings.up.weight)?;
        let up_scales = remove_tensor(tensors, &bindings.up.scales, safetensors::Dtype::BF16)?.into_data();
        let up_biases = remove_tensor(tensors, &bindings.up.biases, safetensors::Dtype::BF16)?.into_data();
        let down_weight = remove_quant_weight(tensors, &bindings.down.weight)?;
        let down_scales = remove_tensor(tensors, &bindings.down.scales, safetensors::Dtype::BF16)?.into_data();
        let down_biases = remove_tensor(tensors, &bindings.down.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len(
            "sparse expert gate weight",
            gate_weight.len(),
            expert_gate_layout.weight_bytes(),
        )?;
        validate_len(
            "sparse expert gate scales",
            gate_scales.len(),
            expert_gate_layout.scale_or_bias_bytes(),
        )?;
        validate_len(
            "sparse expert gate biases",
            gate_biases.len(),
            expert_gate_layout.scale_or_bias_bytes(),
        )?;
        validate_len(
            "sparse expert up weight",
            up_weight.len(),
            expert_gate_layout.weight_bytes(),
        )?;
        validate_len(
            "sparse expert up scales",
            up_scales.len(),
            expert_gate_layout.scale_or_bias_bytes(),
        )?;
        validate_len(
            "sparse expert up biases",
            up_biases.len(),
            expert_gate_layout.scale_or_bias_bytes(),
        )?;
        validate_len(
            "sparse expert down weight",
            down_weight.len(),
            expert_down_layout.weight_bytes(),
        )?;
        validate_len(
            "sparse expert down scales",
            down_scales.len(),
            expert_down_layout.scale_or_bias_bytes(),
        )?;
        validate_len(
            "sparse expert down biases",
            down_biases.len(),
            expert_down_layout.scale_or_bias_bytes(),
        )?;
        Ok(Self {
            gate_weight: Buffer::from_slice(device, &gate_weight),
            gate_scales: Buffer::from_slice(device, &gate_scales),
            gate_biases: Buffer::from_slice(device, &gate_biases),
            up_weight: Buffer::from_slice(device, &up_weight),
            up_scales: Buffer::from_slice(device, &up_scales),
            up_biases: Buffer::from_slice(device, &up_biases),
            down_weight: Buffer::from_slice(device, &down_weight),
            down_scales: Buffer::from_slice(device, &down_scales),
            down_biases: Buffer::from_slice(device, &down_biases),
        })
    }

    fn as_borrowed(&self) -> sparse_mlp::Weights<'_> {
        sparse_mlp::Weights {
            gate_weight: &self.gate_weight,
            gate_scales: &self.gate_scales,
            gate_biases: &self.gate_biases,
            up_weight: &self.up_weight,
            up_scales: &self.up_scales,
            up_biases: &self.up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }
}
