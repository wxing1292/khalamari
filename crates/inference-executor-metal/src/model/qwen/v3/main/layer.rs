use std::rc::Rc;

use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::dense_mlp;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3::Qwen3TextConfig;
use inference_executor_core::model::qwen::v3::weight_layout::Qwen3LayerWeightBindings;

use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::qwen::v3::main::component_config::Qwen3MainConfig;
use crate::model::qwen::v3::main::component_config::derive_qwen3_dense_mlp_configs;
use crate::model::qwen::v3::main::component_config::derive_qwen3_gqa_configs;
use crate::model::qwen::v3::main::gqa::Qwen3MainGQA;
use crate::model::qwen::v3::main::gqa::Qwen3MainGQAState;
use crate::model::qwen::v3_x::layer::Qwen3xDenseMLP;
use crate::model::qwen::v3_x::weight::load_qwen3x_norm_weight;
use crate::model::residual_add::ResidualAdd;
use crate::model::rms_norm::RMSNorm;

pub struct Qwen3MainLayer {
    layer_index: usize,
    input_norm: RMSNorm,
    attention: Qwen3MainGQA,
    residual_add: ResidualAdd,
    post_attention_norm: RMSNorm,
    mlp: Qwen3xDenseMLP,
    scratch: Rc<Qwen3MainLayerScratch>,
}

pub struct Qwen3MainLayerScratch {
    max_tokens: u32,
    hidden_dim: u32,
    residual_stream: [Buffer; 2],
    normalized_hidden: Buffer,
    branch_output: Buffer,
    post_attention_hidden: Buffer,
}

#[derive(Clone, Copy)]
pub struct Qwen3MainLayerInput<'a> {
    pub gqa: &'a GQAMetadataBuffers,
    pub num_tokens: u32,
    pub pages: &'a Buffer,
    pub residual_input: &'a Buffer,
    pub residual_output: &'a Buffer,
}

impl Qwen3MainLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        config: Qwen3MainConfig<'_>,
        layer_index: usize,
        gqa_state: &Qwen3MainGQAState,
        scratch: Rc<Qwen3MainLayerScratch>,
        dense_scratch: Rc<DenseMLPScratch>,
    ) -> Result<Self, ModelExecutorError> {
        let hidden_dim = config.text.hidden_size;
        let eps = config.text.rms_norm_eps;
        let (gqa_core, gqa_metal) = derive_qwen3_gqa_configs(layer_index, config)?;
        let attention = Qwen3MainGQA::new(gqa_core, gqa_metal, gqa_state);
        let (mlp_core, mlp_metal) = derive_qwen3_dense_mlp_configs(layer_index, config)?;
        let mlp = Qwen3xDenseMLP::new(device, mlp_core, mlp_metal, dense_scratch);
        Ok(Self {
            layer_index,
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
        text: &Qwen3TextConfig,
        bindings: Qwen3LayerWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        let Qwen3LayerWeightBindings {
            input_norm_weight,
            post_attention_norm_weight,
            gqa: attention_bindings,
            mlp: mlp_bindings,
        } = bindings;
        let hidden_dim = text.hidden_size;
        self.attention.load_weights(device, store, attention_bindings)?;
        self.mlp.load_weights(device, store, mlp_bindings)?;
        self.input_norm.load_weights(load_qwen3x_norm_weight(
            device,
            store,
            &input_norm_weight,
            &[hidden_dim],
        )?);
        self.post_attention_norm.load_weights(load_qwen3x_norm_weight(
            device,
            store,
            &post_attention_norm_weight,
            &[hidden_dim],
        )?);
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

    pub fn load_state(&mut self, state: &Qwen3MainGQAState) {
        self.attention.load_state(state);
    }

    pub fn layer_index(&self) -> usize {
        self.layer_index
    }

    pub fn residual_output(&self) -> &Buffer {
        self.scratch.residual_stream(self.layer_index)
    }

    pub fn mlp_replay_topology(&self, num_total_tokens: u32) -> dense_mlp::ReplayTopology {
        self.mlp.replay_topology(num_total_tokens)
    }

    pub fn record<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        input: Qwen3MainLayerInput<'a>,
    ) -> &'a Buffer {
        assert!(input.num_tokens > 0, "qwen3 Main replay requires active tokens");
        assert!(
            input.num_tokens <= num_total_tokens,
            "qwen3 Main active tokens must not exceed the replay capacity"
        );
        let hidden_dim = self.scratch.hidden_dim();
        self.input_norm.record_with_barrier(
            recorder,
            num_total_tokens,
            num_active_tokens,
            input.residual_input,
            &self.scratch.normalized_hidden,
        );
        self.attention.record(
            recorder,
            &self.scratch.normalized_hidden,
            &self.scratch.branch_output,
            input.pages,
            input.gqa,
            num_active_tokens,
        );
        self.residual_add.record(
            recorder,
            num_total_tokens,
            hidden_dim,
            num_active_tokens,
            input.residual_input,
            &self.scratch.branch_output,
            &self.scratch.post_attention_hidden,
        );
        self.post_attention_norm.record_with_barrier(
            recorder,
            num_total_tokens,
            num_active_tokens,
            &self.scratch.post_attention_hidden,
            &self.scratch.normalized_hidden,
        );
        self.mlp.record(
            recorder,
            &self.scratch.normalized_hidden,
            &self.scratch.branch_output,
            num_total_tokens,
            num_active_tokens,
        );

        self.residual_add.record(
            recorder,
            num_total_tokens,
            hidden_dim,
            num_active_tokens,
            &self.scratch.post_attention_hidden,
            &self.scratch.branch_output,
            input.residual_output,
        );
        input.residual_output
    }
}

impl ReplayLayer for Qwen3MainLayer {
    type Input<'a> = Qwen3MainLayerInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: Self::Input<'a>) -> Self::Output<'a> {
        Qwen3MainLayer::record(
            self,
            recorder,
            input.num_tokens,
            ReplayU32::Fixed(input.num_tokens),
            input,
        )
    }
}

impl Qwen3MainLayerScratch {
    pub fn new(device: &Device, max_tokens: usize, hidden_dim: usize) -> Self {
        assert!(max_tokens > 0);
        assert!(hidden_dim > 0);
        let max_tokens_u32 = u32::try_from(max_tokens).expect("qwen3 layer token capacity must fit u32");
        let hidden_dim_u32 = u32::try_from(hidden_dim).expect("qwen3 hidden dimension must fit u32");
        let hidden_elements = max_tokens
            .checked_mul(hidden_dim)
            .expect("qwen3 layer scratch element count must fit usize");
        u32::try_from(hidden_elements).expect("qwen3 layer scratch must fit the shader u32 element-count domain");
        Self {
            max_tokens: max_tokens_u32,
            hidden_dim: hidden_dim_u32,
            residual_stream: [
                Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
                Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            ],
            normalized_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            branch_output: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            post_attention_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
        }
    }

    fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }

    fn residual_values(&self, num_tokens: u32) -> u32 {
        debug_assert!(num_tokens > 0 && num_tokens <= self.max_tokens);
        num_tokens * self.hidden_dim
    }

    fn residual_stream(&self, model_layer_index: usize) -> &Buffer {
        &self.residual_stream[model_layer_index % self.residual_stream.len()]
    }
}
