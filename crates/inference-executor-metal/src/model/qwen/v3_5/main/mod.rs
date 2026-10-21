use std::rc::Rc;

use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::check_finite;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::LayerType;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MainWeightBindings;
use inference_executor_core::replay::ReplayBucketPolicy;

use crate::attn::gdn::backend::GDNReplayTopology;
use crate::attn::gdn::batch_metadata::GDNReplayShape;
use crate::attn::gqa::backend::GQAReplayTopology;
use crate::attn::gqa::batch_metadata::GQAReplayShape;
use crate::checkpoint::SafeTensorStore;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::main_residual_capture::MainResidualCapture;
use crate::model::qwen::v3_5::Qwen35GQAReplayKey;
use crate::model::qwen::v3_5::component_config::Qwen35MetalDefaults;
use crate::model::qwen::v3_5::main::layer::Qwen35MainLayer;
use crate::model::qwen::v3_5::main::layer::Qwen35MainLayerInput;
use crate::model::qwen::v3_5::main::layer::Qwen35MainLayerScratch;
use crate::model::qwen::v3_5::main::layer::Qwen35MainMLPReplayTopology;
use crate::model::qwen::v3_x::state::Qwen3xGDNState;
use crate::model::qwen::v3_x::state::Qwen3xGQAState;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::rms_norm::RMSNorm;
use crate::replay::ReplayComponent;

pub mod text_embed;
pub mod layer;
pub mod output;

const QWEN35_MAIN_NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("qwen3.5.main.num_active_tokens");

pub struct Qwen35Main {
    finite_check: Option<check_finite::Compute>,
    layers: Vec<Qwen35MainLayer>,
    final_norm: RMSNorm,
    residual_capture: Option<Rc<dyn MainResidualCapture>>,
    replay_bucket_policy: ReplayBucketPolicy,
}

#[derive(Clone, Copy)]
pub struct Qwen35MainArgs<'a> {
    pub num_tokens: u32,
    pub hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub gqa: &'a crate::attn::gqa::batch_metadata::GQAMetadataBuffers,
    pub gqa_replay_topology: GQAReplayTopology,
    pub gdn: &'a crate::attn::gdn::batch_metadata::GDNMetadataBuffers,
    pub gdn_replay_topology: GDNReplayTopology,
    pub pages: &'a Buffer,
}

impl Qwen35Main {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        config: &Qwen35ModelConfig,
        max_tokens: usize,
        defaults: Qwen35MetalDefaults,
        gqa_state: &Qwen3xGQAState,
        gdn_state: &Qwen3xGDNState,
        residual_capture: Option<Rc<dyn MainResidualCapture>>,
        layer_scratch: Rc<Qwen35MainLayerScratch>,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        let mut num_loaded_gqa_layers = 0;
        let mut num_loaded_gdn_layers = 0;
        let mut layers = Vec::with_capacity(config.text_config.num_hidden_layers);
        for layer_index in 0..config.text_config.num_hidden_layers {
            let layer_type = config.layer_type_at(layer_index)?;
            let attn_layer_index = match layer_type {
                LayerType::FullAttention => num_loaded_gqa_layers,
                LayerType::GDN => num_loaded_gdn_layers,
            };
            layers.push(Qwen35MainLayer::new(
                device,
                config,
                defaults,
                layer_index,
                attn_layer_index,
                gqa_state,
                gdn_state,
                Rc::clone(&layer_scratch),
                dense_scratch,
                moe_scratch,
            )?);
            match layer_type {
                LayerType::FullAttention => num_loaded_gqa_layers += 1,
                LayerType::GDN => num_loaded_gdn_layers += 1,
            }
        }
        let max_tokens = max_tokens
            .try_into()
            .expect("qwen3.5 Main replay token capacity must fit u32");
        let mut topology_boundaries = gqa_state.replay_token_topology_boundaries().into_vec();
        topology_boundaries.extend(gdn_state.replay_token_topology_boundaries());
        for layer in &layers {
            topology_boundaries.extend(layer.mlp_replay_topology_boundaries());
        }
        Ok(Self {
            finite_check: crate::trace::qwen35_check_finite().then(|| {
                check_finite::Compute::new(
                    device,
                    Dtype::Bfloat16,
                    config
                        .text_config
                        .hidden_size
                        .try_into()
                        .expect("hidden dim must fit u32"),
                )
            }),
            layers,
            final_norm: RMSNorm::new(device, config.text_config.hidden_size, config.text_config.rms_norm_eps),
            residual_capture,
            replay_bucket_policy: ReplayBucketPolicy::with_topology_boundaries(max_tokens, &topology_boundaries),
        })
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        bindings: Qwen35MainWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        let Qwen35MainWeightBindings {
            final_norm_weight,
            layers: layer_bindings,
        } = bindings;
        assert_eq!(
            self.layers.len(),
            layer_bindings.len(),
            "qwen3.5 Main component and checkpoint binding layer counts must match"
        );
        for (layer, bindings) in self.layers.iter_mut().zip(layer_bindings) {
            layer.load_weights(device, store, config, bindings)?;
            store.unload_all();
        }
        self.final_norm.load_weights(load_qwen3x_norm_weight(
            device,
            store,
            &final_norm_weight,
            &[config.text_config.hidden_size],
        )?);
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        assert!(
            self.residual_capture.is_none(),
            "qwen3.5 Main residual capture must be detached before weight unloading"
        );
        self.final_norm.unload_weights();
        for layer in self.layers.iter_mut().rev() {
            layer.unload_weights();
        }
    }

    pub fn set_residual_capture(&mut self, residual_capture: Option<Rc<dyn MainResidualCapture>>) {
        assert!(
            self.residual_capture.is_none(),
            "qwen3.5 Main residual capture is already attached"
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

    pub fn load_state(&mut self, gqa_state: &Qwen3xGQAState, gdn_state: &Qwen3xGDNState) {
        for layer in &mut self.layers {
            layer.load_state(gqa_state, gdn_state);
        }
    }

    pub fn capacity(&self, num_active_tokens: u32) -> u32 {
        self.replay_bucket_policy.capacity(num_active_tokens)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_replay(
        &self,
        num_active_tokens: u32,
        gqa_shape: GQAReplayShape,
        gqa_topology: GQAReplayTopology,
        gdn_shape: GDNReplayShape,
        gdn_topology: GDNReplayTopology,
    ) -> (Qwen35MainReplayKey, ReplayArguments) {
        let key = self.replay_key_for(num_active_tokens, gqa_shape, gqa_topology, gdn_shape, gdn_topology);
        let arguments = main_replay_arguments(num_active_tokens);
        (key, arguments)
    }

    pub fn record<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        args: Qwen35MainArgs<'a>,
    ) -> &'a Buffer {
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
                Qwen35MainLayerInput {
                    gdn: args.gdn,
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
        if let Some(check) = &self.finite_check {
            recorder.record(ReplayOp::opaque(check.invoke(
                check_finite::Input {
                    buffer: args.hidden_output,
                    num_total_rows: num_total_tokens,
                    num_active_rows: num_active_tokens,
                },
                "Main final_norm.output".to_owned(),
            )));
        }
        args.hidden_output
    }

    #[allow(clippy::too_many_arguments)]
    fn replay_key_for(
        &self,
        num_active_tokens: u32,
        gqa_shape: GQAReplayShape,
        gqa_topology: GQAReplayTopology,
        gdn_shape: GDNReplayShape,
        gdn_topology: GDNReplayTopology,
    ) -> Qwen35MainReplayKey {
        gqa_shape.validate();
        gdn_shape.validate();
        assert_eq!(
            gqa_shape.num_tokens, num_active_tokens,
            "qwen3.5 Main GQA active tokens must match the stage"
        );
        assert_eq!(
            gdn_shape.num_tokens, num_active_tokens,
            "qwen3.5 Main GDN active tokens must match the stage"
        );
        assert_eq!(
            gqa_shape.num_total_tokens, gdn_shape.num_total_tokens,
            "qwen3.5 Main GQA and GDN token capacities must match"
        );
        let num_total_tokens = gqa_shape.num_total_tokens;
        self.validate_capacity(num_active_tokens, num_total_tokens);
        let mlp_topologies = self
            .layers
            .iter()
            .map(|layer| layer.mlp_replay_topology(num_total_tokens))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Qwen35MainReplayKey::new(
            num_total_tokens,
            gqa_shape,
            gqa_topology,
            gdn_shape,
            gdn_topology,
            mlp_topologies,
        )
    }

    fn validate_capacity(&self, num_active_tokens: u32, num_total_tokens: u32) {
        assert_eq!(
            self.capacity(num_active_tokens),
            num_total_tokens,
            "qwen3.5 Main metadata token capacity must match the stage policy"
        );
        for layer in &self.layers {
            assert_eq!(
                layer.mlp_replay_topology(num_active_tokens),
                layer.mlp_replay_topology(num_total_tokens),
                "qwen3.5 Main replay token capacity must preserve every layer MLP topology"
            );
        }
    }
}

fn main_replay_arguments(num_active_tokens: u32) -> ReplayArguments {
    ReplayArguments::new().with_u32(QWEN35_MAIN_NUM_ACTIVE_TOKENS, num_active_tokens)
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35MainReplayKey {
    num_total_tokens: u32,
    mlp_topologies: Box<[Qwen35MainMLPReplayTopology]>,
    gqa: Qwen35GQAReplayKey,
    gdn: Qwen35MainGDNReplayKey,
}

impl Qwen35MainReplayKey {
    fn new(
        num_total_tokens: u32,
        gqa_shape: GQAReplayShape,
        gqa_topology: GQAReplayTopology,
        gdn_shape: GDNReplayShape,
        gdn_topology: GDNReplayTopology,
        mlp_topologies: Box<[Qwen35MainMLPReplayTopology]>,
    ) -> Self {
        gqa_shape.validate();
        gdn_shape.validate();
        assert_eq!(
            gqa_shape.num_total_tokens, num_total_tokens,
            "qwen3.5 Main GQA key capacity must match the stage"
        );
        assert_eq!(
            gdn_shape.num_total_tokens, num_total_tokens,
            "qwen3.5 Main GDN key capacity must match the stage"
        );
        Self {
            num_total_tokens,
            mlp_topologies,
            gqa: Qwen35GQAReplayKey::new(gqa_shape, gqa_topology),
            gdn: Qwen35MainGDNReplayKey::new(gdn_shape, gdn_topology),
        }
    }

    fn num_total_tokens(&self) -> u32 {
        self.num_total_tokens
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Qwen35MainGDNReplayKey {
    num_total_reqs: u32,
    num_total_tokens: u32,
    topology: GDNReplayTopology,
}

impl Qwen35MainGDNReplayKey {
    fn new(gdn_shape: GDNReplayShape, topology: GDNReplayTopology) -> Self {
        gdn_shape.validate();
        Self {
            num_total_reqs: gdn_shape.num_total_reqs,
            num_total_tokens: gdn_shape.num_total_tokens,
            topology,
        }
    }
}

impl ReplayComponent for Qwen35Main {
    type Key = Qwen35MainReplayKey;
    type Input<'a> = Qwen35MainArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        self.replay_key_for(
            input.num_tokens,
            input.gqa.replay_shape(),
            input.gqa_replay_topology,
            input.gdn.replay_shape(),
            input.gdn_replay_topology,
        )
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let key = self.replay_key(input);
        Qwen35Main::record(
            self,
            recorder,
            key.num_total_tokens(),
            ReplayU32::Parameter(QWEN35_MAIN_NUM_ACTIVE_TOKENS),
            *input,
        );
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::residual_add;
    use inference_backend_metal::metal::Dtype;

    use super::*;

    struct SelectedLayerCapture {
        model_layer_index: usize,
        capture_output: Buffer,
    }

    impl MainResidualCapture for SelectedLayerCapture {
        fn capture_for_model_layer(&self, model_layer_index: usize) -> Option<residual_add::CaptureTarget<'_>> {
            (model_layer_index == self.model_layer_index)
                .then(|| residual_add::CaptureTarget::columns(&self.capture_output, 16, 4..12))
        }
    }

    fn assert_replay_component<T: ReplayComponent>() {}

    #[test]
    fn test_residual_capture_selects_only_the_configured_model_layer() {
        let device = Device::system_default();
        let capture = Rc::new(SelectedLayerCapture {
            model_layer_index: 10,
            capture_output: Buffer::new_zeroed_elements(&device, 64, Dtype::Bfloat16),
        });
        let erased_capture: Rc<dyn MainResidualCapture> = capture.clone();

        assert!(erased_capture.capture_for_model_layer(9).is_none());
        assert!(erased_capture.capture_for_model_layer(10).is_some());
        assert!(erased_capture.capture_for_model_layer(11).is_none());

        assert_replay_component::<Qwen35Main>();
    }

    #[test]
    fn test_absent_main_capture_never_captures() {
        let capture: Option<Rc<dyn MainResidualCapture>> = None;
        for model_layer_index in [0, 1, 10, usize::MAX] {
            assert!(
                capture
                    .as_ref()
                    .and_then(|capture| capture.capture_for_model_layer(model_layer_index))
                    .is_none()
            );
        }

        assert_replay_component::<Qwen35Main>();
    }
}
