use std::rc::Rc;

use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::check_finite;
use inference_backend_metal::components::dense_mlp;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35AttentionWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35LayerWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MLPWeightBindings;

use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::moe::backend::GatedMoEReplayTopology;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::qwen::v3_5::component_config::Qwen35MetalDefaults;
use crate::model::qwen::v3_5::component_config::derive_qwen35_dense_mlp_configs;
use crate::model::qwen::v3_5::component_config::derive_qwen35_gqa_configs;
use crate::model::qwen::v3_5::component_config::derive_qwen35_moe_configs;
use crate::model::qwen::v3_5::mtp::QWEN35_MTP_GQA_LAYER_INDEX;
use crate::model::qwen::v3_x::layer::Qwen3xDenseMLP;
use crate::model::qwen::v3_x::layer::Qwen3xGQA;
use crate::model::qwen::v3_x::layer::Qwen3xMoE;
use crate::model::qwen::v3_x::state::Qwen3xGQAState;
use crate::model::qwen::v3_x::weight::remove_qwen3x_norm_weight;
use crate::model::residual_add::ResidualAdd;
use crate::model::rms_norm::RMSNorm;

pub struct Qwen35MTPLayer {
    finite_check: Option<check_finite::Compute>,
    input_norm: RMSNorm,
    attention: Qwen3xGQA,
    residual_add: ResidualAdd,
    post_attention_norm: RMSNorm,
    mlp: Qwen35MTPMLP,
    scratch: Rc<Qwen35MTPLayerScratch>,
}

pub struct Qwen35MTPLayerScratch {
    max_tokens: u32,
    hidden_dim: u32,
    residual_output: Buffer,
    normalized_hidden: Buffer,
    branch_output: Buffer,
    post_attention_hidden: Buffer,
}

#[allow(clippy::large_enum_variant)]
enum Qwen35MTPMLP {
    Dense(Qwen3xDenseMLP),
    MoE(Qwen3xMoE),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Qwen35MTPMLPReplayTopology {
    Dense(dense_mlp::ReplayTopology),
    MoE(GatedMoEReplayTopology),
}

#[derive(Clone, Copy)]
pub struct Qwen35MTPLayerInput<'a> {
    pub gqa: &'a GQAMetadataBuffers,
    pub num_tokens: u32,
    pub pages: &'a Buffer,
    pub residual_input: &'a Buffer,
}

impl Qwen35MTPLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        model_layer_index: usize,
        gqa_state: &Qwen3xGQAState,
        scratch: Rc<Qwen35MTPLayerScratch>,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        let (attention_core, attention_metal) =
            derive_qwen35_gqa_configs(model_layer_index, &config.text_config, defaults)?;
        let attention = Qwen3xGQA::new(
            inference_backend_metal::metal::ReplayU32::Parameter(QWEN35_MTP_GQA_LAYER_INDEX),
            attention_core,
            attention_metal,
            Rc::clone(gqa_state.backend()),
            Rc::clone(gqa_state.scratch()),
            Rc::clone(gqa_state.request_page_table()),
        );
        let mlp = Qwen35MTPMLP::new(device, config, defaults, dense_scratch, moe_scratch)?;
        let hidden_dim = config.text_config.hidden_size;
        let eps = config.text_config.rms_norm_eps;
        Ok(Self {
            finite_check: crate::trace::qwen35_check_finite().then(|| {
                check_finite::Compute::new(
                    device,
                    Dtype::Bfloat16,
                    hidden_dim.try_into().expect("hidden dim must fit u32"),
                )
            }),
            input_norm: RMSNorm::new(device, hidden_dim, eps),
            attention,
            residual_add: ResidualAdd::new(device),
            post_attention_norm: RMSNorm::new(device, hidden_dim, eps),
            mlp,
            scratch,
        })
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        bindings: Qwen35LayerWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        let Qwen35LayerWeightBindings {
            input_norm_weight,
            post_attention_norm_weight,
            attention,
            mlp,
        } = bindings;
        let attention = match attention {
            Qwen35AttentionWeightBindings::GQA(bindings) => bindings,
            Qwen35AttentionWeightBindings::GDN(_) => panic!("qwen3.5 MTP layer bindings must contain GQA attention"),
        };
        self.attention.load_weights(device, store, attention)?;
        self.mlp.load_weights(device, store, mlp)?;
        let hidden_dim = config.text_config.hidden_size;
        let mut tensors = store.load_tensors([input_norm_weight.as_str(), post_attention_norm_weight.as_str()])?;
        self.input_norm.load_weights(remove_qwen3x_norm_weight(
            device,
            &mut tensors,
            &input_norm_weight,
            &[hidden_dim],
        )?);
        self.post_attention_norm.load_weights(remove_qwen3x_norm_weight(
            device,
            &mut tensors,
            &post_attention_norm_weight,
            &[hidden_dim],
        )?);
        assert!(tensors.is_empty(), "qwen3.5 MTP layer must consume its norm tensor map");
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        self.post_attention_norm.unload_weights();
        self.input_norm.unload_weights();
        self.mlp.unload_weights();
        self.attention.unload_weights();
    }

    pub fn unload_state(&mut self) {
        self.attention.unload_state();
    }

    pub fn load_state(&mut self, state: &Qwen3xGQAState) {
        self.attention.load_state(state);
    }

    pub fn gqa_tokens_per_page(&self) -> usize {
        self.attention.num_tokens_per_page()
    }

    pub fn mlp_replay_topology(&self, num_total_tokens: u32) -> Qwen35MTPMLPReplayTopology {
        self.mlp.replay_topology(num_total_tokens)
    }

    pub fn mlp_replay_topology_boundaries(&self) -> Box<[u32]> {
        self.mlp.replay_topology_boundaries()
    }

    /// Records the MTP layer at one caller-selected token capacity.
    ///
    /// An adjacent compatible RMS normalization can fuse with the final
    /// residual add. The residual add remains valid when fusion is unavailable.
    pub fn record<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        input: Qwen35MTPLayerInput<'a>,
    ) -> &'a Buffer {
        assert!(input.num_tokens > 0, "qwen3.5 MTP replay requires active tokens");
        assert!(
            input.num_tokens <= num_total_tokens,
            "qwen3.5 MTP active tokens must not exceed the replay capacity"
        );
        let hidden_dim = self.scratch.hidden_dim;
        let check = |recorder: &mut ReplayRecorder, stage: &str, buffer: &'a Buffer| {
            if let Some(check) = &self.finite_check {
                recorder.record(ReplayOp::opaque(check.invoke(
                    check_finite::Input {
                        buffer,
                        num_total_rows: num_total_tokens,
                        num_active_rows: num_active_tokens,
                    },
                    format!("MTP {stage}"),
                )));
            }
        };
        check(recorder, "input", input.residual_input);
        self.input_norm.record_with_barrier(
            recorder,
            num_total_tokens,
            num_active_tokens,
            input.residual_input,
            &self.scratch.normalized_hidden,
        );
        check(recorder, "input_norm.output", &self.scratch.normalized_hidden);
        self.attention.record(
            recorder,
            &self.scratch.normalized_hidden,
            &self.scratch.branch_output,
            input.pages,
            input.gqa,
            num_active_tokens,
        );
        check(recorder, "GQA.output", &self.scratch.branch_output);
        self.residual_add.record(
            recorder,
            num_total_tokens,
            hidden_dim,
            num_active_tokens,
            input.residual_input,
            &self.scratch.branch_output,
            &self.scratch.post_attention_hidden,
        );
        check(
            recorder,
            "attention_residual.output",
            &self.scratch.post_attention_hidden,
        );
        self.post_attention_norm.record_with_barrier(
            recorder,
            num_total_tokens,
            num_active_tokens,
            &self.scratch.post_attention_hidden,
            &self.scratch.normalized_hidden,
        );
        check(recorder, "post_attention_norm.output", &self.scratch.normalized_hidden);
        self.mlp.record(
            recorder,
            &self.scratch.normalized_hidden,
            &self.scratch.branch_output,
            num_total_tokens,
            num_active_tokens,
        );
        check(recorder, "MLP.output", &self.scratch.branch_output);
        self.residual_add.record(
            recorder,
            num_total_tokens,
            hidden_dim,
            num_active_tokens,
            &self.scratch.post_attention_hidden,
            &self.scratch.branch_output,
            &self.scratch.residual_output,
        );
        check(recorder, "output", &self.scratch.residual_output);
        &self.scratch.residual_output
    }
}

impl ReplayLayer for Qwen35MTPLayer {
    type Input<'a> = Qwen35MTPLayerInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: Self::Input<'a>) -> Self::Output<'a> {
        Qwen35MTPLayer::record(
            self,
            recorder,
            input.num_tokens,
            ReplayU32::Fixed(input.num_tokens),
            input,
        )
    }
}

impl Qwen35MTPMLP {
    fn new(
        device: &Device,
        config: &Qwen35ModelConfig,
        defaults: Qwen35MetalDefaults,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        if config.layer_uses_moe(0) {
            let (core, metal) = derive_qwen35_moe_configs("layers.0", 0, config, defaults)?;
            Ok(Self::MoE(Qwen3xMoE::new(
                device,
                core,
                metal,
                Rc::clone(moe_scratch.expect("qwen3.5 MTP MoE layer requires shared MoE scratch")),
            )))
        } else {
            let (core, metal) = derive_qwen35_dense_mlp_configs(0, &config.text_config, defaults)?;
            Ok(Self::Dense(Qwen3xDenseMLP::new(
                device,
                core,
                metal,
                Rc::clone(dense_scratch.expect("qwen3.5 MTP dense layer requires shared dense scratch")),
            )))
        }
    }

    fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: Qwen35MLPWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        match (self, bindings) {
            (Self::Dense(component), Qwen35MLPWeightBindings::Dense(bindings)) => {
                component.load_weights(device, store, *bindings)
            },
            (Self::MoE(component), Qwen35MLPWeightBindings::MoE(bindings)) => {
                component.load_weights(device, store, *bindings)
            },
            _ => panic!("qwen3.5 MTP layer MLP config and checkpoint bindings must have the same kind"),
        }
    }

    fn unload_weights(&mut self) {
        match self {
            Self::Dense(component) => component.unload_weights(),
            Self::MoE(component) => component.unload_weights(),
        }
    }

    fn record<'a>(
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

    fn replay_topology(&self, num_total_tokens: u32) -> Qwen35MTPMLPReplayTopology {
        match self {
            Self::Dense(component) => Qwen35MTPMLPReplayTopology::Dense(component.replay_topology(num_total_tokens)),
            Self::MoE(component) => Qwen35MTPMLPReplayTopology::MoE(component.replay_topology(num_total_tokens)),
        }
    }

    fn replay_topology_boundaries(&self) -> Box<[u32]> {
        match self {
            Self::Dense(component) => component.replay_topology_boundaries(),
            Self::MoE(component) => component.replay_topology_boundaries(),
        }
    }
}

impl Qwen35MTPLayerScratch {
    pub fn new(device: &Device, max_tokens: usize, hidden_dim: usize) -> Self {
        assert!(max_tokens > 0);
        assert!(hidden_dim > 0);
        let max_tokens_u32 = u32::try_from(max_tokens).expect("qwen3.5 MTP layer token capacity must fit u32");
        let hidden_dim_u32 = u32::try_from(hidden_dim).expect("qwen3.5 MTP hidden dimension must fit u32");
        let hidden_elements = max_tokens
            .checked_mul(hidden_dim)
            .expect("qwen3.5 MTP layer scratch element count must fit usize");
        u32::try_from(hidden_elements).expect("qwen3.5 MTP layer scratch must fit the shader u32 element-count domain");
        Self {
            max_tokens: max_tokens_u32,
            hidden_dim: hidden_dim_u32,
            residual_output: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            normalized_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            branch_output: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            post_attention_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
        }
    }

    fn residual_values(&self, num_tokens: u32) -> u32 {
        debug_assert!(num_tokens > 0 && num_tokens <= self.max_tokens);
        num_tokens * self.hidden_dim
    }
}
