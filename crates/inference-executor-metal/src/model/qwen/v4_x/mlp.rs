use std::rc::Rc;

use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::dense_mlp;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v4_x::Qwen4xModelConfig;
use inference_executor_core::model::qwen::v4_x::weight_layout::Qwen4xMLPWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::moe::backend::GatedMoEReplayTopology;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::qwen::v3_x::layer::Qwen3xDenseMLP;
use crate::model::qwen::v3_x::layer::Qwen3xMoE;
use crate::model::qwen::v4_x::component_config::dense_configs;
use crate::model::qwen::v4_x::component_config::moe_configs;

#[allow(clippy::large_enum_variant)]
pub enum Qwen4xMLP {
    Dense(Qwen3xDenseMLP),
    MoE(Qwen3xMoE),
}
pub enum Qwen4xMLPScratch {
    Dense(Rc<DenseMLPScratch>),
    MoE(Rc<MoEScratch>),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Qwen4xMLPReplayTopology {
    Dense(dense_mlp::ReplayTopology),
    MoE(GatedMoEReplayTopology),
}
impl Qwen4xMLPScratch {
    pub fn new(
        device: &Device,
        config: &Qwen4xModelConfig,
        bindings: &Qwen4xMLPWeightBindings,
        max_tokens: usize,
    ) -> Result<Self, ModelExecutorError> {
        Ok(match bindings {
            Qwen4xMLPWeightBindings::Dense(bindings) => {
                let (core, _) = dense_configs(config, 0, bindings)?;
                Self::Dense(Rc::new(DenseMLPScratch::new(
                    device,
                    &core,
                    Dtype::Bfloat16,
                    max_tokens,
                )))
            },
            Qwen4xMLPWeightBindings::MoE(bindings) => {
                let (core, metal) = moe_configs(config, 0, bindings)?;
                Self::MoE(Rc::new(MoEScratch::new(device, &core, metal, max_tokens)))
            },
        })
    }
}
impl Qwen4xMLP {
    pub fn new(
        device: &Device,
        config: &Qwen4xModelConfig,
        model_layer_index: usize,
        bindings: &Qwen4xMLPWeightBindings,
        scratch: &Qwen4xMLPScratch,
    ) -> Result<Self, ModelExecutorError> {
        Ok(match (bindings, scratch) {
            (Qwen4xMLPWeightBindings::Dense(bindings), Qwen4xMLPScratch::Dense(scratch)) => {
                let (core, metal) = dense_configs(config, model_layer_index, bindings)?;
                Self::Dense(Qwen3xDenseMLP::new(device, core, metal, Rc::clone(scratch)))
            },
            (Qwen4xMLPWeightBindings::MoE(bindings), Qwen4xMLPScratch::MoE(scratch)) => {
                let (core, metal) = moe_configs(config, model_layer_index, bindings)?;
                Self::MoE(Qwen3xMoE::new(device, core, metal, Rc::clone(scratch)))
            },
            _ => unreachable!("MLP scratch matches validated model configuration"),
        })
    }
    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: Qwen4xMLPWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        match (self, bindings) {
            (Self::Dense(component), Qwen4xMLPWeightBindings::Dense(bindings)) => {
                component.load_weights(device, store, *bindings)
            },
            (Self::MoE(component), Qwen4xMLPWeightBindings::MoE(bindings)) => {
                component.load_weights(device, store, *bindings)
            },
            _ => unreachable!("MLP weights match initialized component"),
        }
    }
    pub fn unload_weights(&mut self) {
        match self {
            Self::Dense(component) => component.unload_weights(),
            Self::MoE(component) => component.unload_weights(),
        }
    }
    pub fn record<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        input: &'a Buffer,
        output: &'a Buffer,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
    ) {
        match self {
            Self::Dense(component) => component.record(recorder, input, output, num_total_tokens, num_active_tokens),
            Self::MoE(component) => component.record(recorder, input, output, num_total_tokens, num_active_tokens),
        }
    }
    pub fn replay_topology(&self, num_total_tokens: u32) -> Qwen4xMLPReplayTopology {
        match self {
            Self::Dense(component) => Qwen4xMLPReplayTopology::Dense(component.replay_topology(num_total_tokens)),
            Self::MoE(component) => Qwen4xMLPReplayTopology::MoE(component.replay_topology(num_total_tokens)),
        }
    }
    pub fn replay_topology_boundaries(&self) -> Box<[u32]> {
        match self {
            Self::Dense(component) => component.replay_topology_boundaries(),
            Self::MoE(component) => component.replay_topology_boundaries(),
        }
    }
}
