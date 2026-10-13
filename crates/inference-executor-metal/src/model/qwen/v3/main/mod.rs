use std::rc::Rc;

use inference_backend_metal::components::dense_mlp;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3::Qwen3TextConfig;
use inference_executor_core::model::qwen::v3::weight_layout::Qwen3MainWeightBindings;

use self::component_config::Qwen3MainConfig;
use crate::attn::gqa::ungated_backend::UngatedGQAReplayTopology;
use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::main_residual_capture::MainResidualCapture;
use crate::model::qwen::v3::main::gqa::Qwen3MainGQAState;
use crate::model::qwen::v3::main::layer::Qwen3MainLayer;
use crate::model::qwen::v3::main::layer::Qwen3MainLayerInput;
use crate::model::qwen::v3::main::layer::Qwen3MainLayerScratch;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::rms_norm::RMSNorm;
use crate::replay::ReplayComponent;

const QWEN3_MAIN_NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("qwen3.main.num_active_tokens");

pub mod text_embed;
pub mod gqa;
pub mod layer;
pub mod output;
pub mod component_config;

pub struct Qwen3Main {
    layers: Vec<Qwen3MainLayer>,
    final_norm: RMSNorm,
    residual_capture: Option<Rc<dyn MainResidualCapture>>,
}

#[derive(Clone, Copy)]
pub struct Qwen3MainArgs<'a> {
    pub num_tokens: u32,
    pub hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub gqa: &'a crate::attn::gqa::batch_metadata::GQAMetadataBuffers,
    pub gqa_replay_topology: UngatedGQAReplayTopology,
    pub pages: &'a Buffer,
}

impl Qwen3Main {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        config: Qwen3MainConfig<'_>,
        gqa_state: &Qwen3MainGQAState,
        residual_capture: Option<Rc<dyn MainResidualCapture>>,
        layer_scratch: Rc<Qwen3MainLayerScratch>,
        dense_scratch: &Rc<DenseMLPScratch>,
    ) -> Result<Self, ModelExecutorError> {
        let text = config.text;
        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        for model_layer_index in 0..text.num_hidden_layers {
            layers.push(Qwen3MainLayer::new(
                device,
                config,
                model_layer_index,
                gqa_state,
                Rc::clone(&layer_scratch),
                Rc::clone(dense_scratch),
            )?);
        }
        Ok(Self {
            layers,
            final_norm: RMSNorm::new(device, text.hidden_size, text.rms_norm_eps),
            residual_capture,
        })
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        text: &Qwen3TextConfig,
        bindings: Qwen3MainWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        let Qwen3MainWeightBindings {
            final_norm_weight,
            layers: layer_bindings,
        } = bindings;
        assert_eq!(
            layer_bindings.len(),
            text.num_hidden_layers,
            "qwen3 Main config and checkpoint binding layer counts must match"
        );
        assert_eq!(
            self.layers.len(),
            layer_bindings.len(),
            "qwen3 Main component and checkpoint binding layer counts must match"
        );
        for (layer, bindings) in self.layers.iter_mut().zip(layer_bindings) {
            layer.load_weights(device, store, text, bindings)?;
            store.unload_all();
        }
        let final_norm_weight = load_qwen3x_norm_weight(device, store, &final_norm_weight, &[text.hidden_size])?;
        self.final_norm.load_weights(final_norm_weight);
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        assert!(
            self.residual_capture.is_none(),
            "qwen3 Main residual capture must be detached before weight unloading"
        );
        self.final_norm.unload_weights();
        for layer in self.layers.iter_mut().rev() {
            layer.unload_weights();
        }
    }

    pub fn set_residual_capture(&mut self, residual_capture: Option<Rc<dyn MainResidualCapture>>) {
        assert!(
            self.residual_capture.is_none(),
            "qwen3 Main residual capture is already attached"
        );
        self.residual_capture = residual_capture;
    }

    pub fn unset_residual_capture(&mut self) {
        self.residual_capture.take();
    }

    pub fn unload_state(&mut self) {
        for layer in self.layers.iter_mut().rev() {
            layer.unload_state();
        }
    }

    pub fn load_state(&mut self, state: &Qwen3MainGQAState) {
        for layer in &mut self.layers {
            layer.load_state(state);
        }
    }

    pub fn prepare_replay(
        &self,
        num_active_tokens: u32,
        gqa_shape: GQAReplayShape,
        gqa_topology: UngatedGQAReplayTopology,
    ) -> (Qwen3MainReplayKey, ReplayArguments) {
        let key = self.replay_key_for(num_active_tokens, gqa_shape, gqa_topology);
        let arguments = ReplayArguments::new().with_u32(QWEN3_MAIN_NUM_ACTIVE_TOKENS, num_active_tokens);
        (key, arguments)
    }

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        args: Qwen3MainArgs<'a>,
    ) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        match num_active_tokens {
            ReplayU32::Fixed(value) => {
                assert_eq!(value, args.num_tokens);
                assert_eq!(value, num_total_tokens);
            },
            ReplayU32::Parameter(_) => self.validate_capacity(args.num_tokens, num_total_tokens),
        }
        let mut hidden = args.hidden_input;
        for layer in &self.layers {
            let residual_output = layer.residual_output();
            hidden = layer.record(
                recorder,
                num_total_tokens,
                num_active_tokens,
                Qwen3MainLayerInput {
                    gqa: args.gqa,
                    num_tokens: args.num_tokens,
                    pages: args.pages,
                    residual_input: hidden,
                    residual_output,
                    residual_capture_dest: self
                        .residual_capture
                        .as_ref()
                        .and_then(|capture| capture.capture_for_model_layer(layer.layer_index())),
                },
            );
        }
        self.final_norm.record_with_barrier(
            recorder,
            num_total_tokens,
            num_active_tokens,
            hidden,
            args.hidden_output,
        );
        args.hidden_output
    }

    fn replay_key_for(
        &self,
        num_active_tokens: u32,
        gqa_shape: GQAReplayShape,
        gqa_topology: UngatedGQAReplayTopology,
    ) -> Qwen3MainReplayKey {
        gqa_shape.validate();
        assert_eq!(
            gqa_shape.num_tokens, num_active_tokens,
            "qwen3 Main GQA active tokens must match the stage"
        );
        let num_total_tokens = gqa_shape.num_total_tokens;
        self.validate_capacity(num_active_tokens, num_total_tokens);
        let mlp_topologies = self
            .layers
            .iter()
            .map(|layer| layer.mlp_replay_topology(num_total_tokens))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Qwen3MainReplayKey::new(num_total_tokens, gqa_shape, gqa_topology, mlp_topologies)
    }

    fn validate_capacity(&self, num_active_tokens: u32, num_total_tokens: u32) {
        assert!(num_active_tokens > 0);
        assert!(num_active_tokens <= num_total_tokens);
        for layer in &self.layers {
            assert_eq!(
                layer.mlp_replay_topology(num_active_tokens),
                layer.mlp_replay_topology(num_total_tokens),
                "qwen3 Main replay token capacity must preserve every layer MLP topology"
            );
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Qwen3MainGQAReplayKey {
    num_total_q_token_tiles: u32,
    num_total_sdpa_map_task_templates: u32,
    topology: UngatedGQAReplayTopology,
}

impl Qwen3MainGQAReplayKey {
    fn new(gqa_shape: GQAReplayShape, topology: UngatedGQAReplayTopology) -> Self {
        gqa_shape.validate();
        Self {
            num_total_q_token_tiles: gqa_shape.num_total_q_token_tiles,
            num_total_sdpa_map_task_templates: gqa_shape.num_total_sdpa_map_task_templates,
            topology,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3MainReplayKey {
    num_total_tokens: u32,
    mlp_topologies: Box<[dense_mlp::ReplayTopology]>,
    gqa: Qwen3MainGQAReplayKey,
}

impl Qwen3MainReplayKey {
    fn new(
        num_total_tokens: u32,
        gqa_shape: GQAReplayShape,
        gqa_topology: UngatedGQAReplayTopology,
        mlp_topologies: Box<[dense_mlp::ReplayTopology]>,
    ) -> Self {
        gqa_shape.validate();
        assert_eq!(gqa_shape.num_total_tokens, num_total_tokens);
        Self {
            num_total_tokens,
            mlp_topologies,
            gqa: Qwen3MainGQAReplayKey::new(gqa_shape, gqa_topology),
        }
    }

    fn num_total_tokens(&self) -> u32 {
        self.num_total_tokens
    }
}

impl ReplayComponent for Qwen3Main {
    type Key = Qwen3MainReplayKey;
    type Input<'a> = Qwen3MainArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        self.replay_key_for(input.num_tokens, input.gqa.replay_shape(), input.gqa_replay_topology)
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let key = self.replay_key(input);
        Qwen3Main::record(
            self,
            recorder,
            key.num_total_tokens(),
            ReplayU32::Parameter(QWEN3_MAIN_NUM_ACTIVE_TOKENS),
            *input,
        );
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::gqa::sdpa;
    use inference_backend_metal::metal::Dtype;
    use inference_backend_metal::metal::ReplayArguments;
    use inference_backend_metal::operators::affine_quantized;

    use super::*;
    use crate::attn::gqa::ungated_backend::UNGATED_GQA_NUM_ACTIVE_KV_SPLITS;
    use crate::attn::gqa::ungated_backend::UNGATED_GQA_NUM_ACTIVE_Q_TOKEN_TILES;
    use crate::attn::gqa::ungated_backend::add_ungated_gqa_private_replay_arguments;

    #[test]
    fn test_main_replay_reuses_capacity_and_submits_active_gqa_work() {
        let sdpa_config = sdpa::Config {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: 8,
            num_kv_heads: 1,
            head_dim: 128,
            tokens_per_page: 8,
        };
        let topology = UngatedGQAReplayTopology {
            sdpa_execution: sdpa::ExecutionVariant::sliced_q_per_head(sdpa_config, 8, 16, 8),
            sdpa_direct_output: false,
            qkv_affine: affine_quantized::KernelKind::QmvBn8Bk32,
            output_affine: affine_quantized::KernelKind::QmvBn8Bk32,
        };
        let shorter_history = GQAReplayShape {
            num_tokens: 8,
            num_total_tokens: 8,
            num_q_token_tiles: 1,
            num_total_q_token_tiles: 1,
            num_sdpa_map_task_templates: 7,
            num_total_sdpa_map_task_templates: 8,
            reduce_sdpa_partial_outputs: true,
        };
        let mut longer_history = shorter_history;
        longer_history.num_sdpa_map_task_templates = 8;

        assert_eq!(
            Qwen3MainGQAReplayKey::new(shorter_history, topology),
            Qwen3MainGQAReplayKey::new(longer_history, topology)
        );

        let mut shorter_arguments = ReplayArguments::new();
        add_ungated_gqa_private_replay_arguments(shorter_history, topology, &mut shorter_arguments);
        let mut longer_arguments = ReplayArguments::new();
        add_ungated_gqa_private_replay_arguments(longer_history, topology, &mut longer_arguments);
        assert_eq!(
            shorter_arguments,
            ReplayArguments::new()
                .with_u32(UNGATED_GQA_NUM_ACTIVE_Q_TOKEN_TILES, 1)
                .with_u32(UNGATED_GQA_NUM_ACTIVE_KV_SPLITS, 7)
        );
        assert_eq!(
            longer_arguments,
            ReplayArguments::new()
                .with_u32(UNGATED_GQA_NUM_ACTIVE_Q_TOKEN_TILES, 1)
                .with_u32(UNGATED_GQA_NUM_ACTIVE_KV_SPLITS, 8)
        );
    }
}
