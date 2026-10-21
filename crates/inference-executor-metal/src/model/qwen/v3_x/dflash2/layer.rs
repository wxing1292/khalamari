use std::rc::Rc;

use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2Config;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2LayerWeightBindings;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xDenseMLPWeightBindings;

use crate::attn::bidi_block_gqa::metadata::BiDiBlockGQAMetadataBuffers;
use crate::attn::bidi_block_gqa::state::BiDiBlockGQAState;
use crate::checkpoint::SafeTensorStore;
use crate::def::quantized_affine::QuantizedAffineLayout;
use crate::mlp::dense::backend::DenseMLPMetalConfig;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::qwen::v3_x::dflash2::attention::Qwen3xDFlash2Attention;
use crate::model::qwen::v3_x::dflash2::attention::derive_qwen3x_dflash2_gqa_configs;
use crate::model::qwen::v3_x::dflash2::conv::Qwen3xDFlash2Conv;
use crate::model::qwen::v3_x::layer::Qwen3xDenseMLP;
use crate::model::qwen::v3_x::weight::remove_qwen3x_norm_weight;
use crate::model::qwen::v3_x::weight::resolve_uniform_quantization;
use crate::model::qwen::v3_x::weight::to_u32;
use crate::model::residual_add::ResidualAdd;
use crate::model::rms_norm::RMSNorm;

pub struct Qwen3xDFlash2Layer {
    dflash2_layer_index: usize,
    input_norm: RMSNorm,
    attention_conv: Qwen3xDFlash2Conv,
    attention: Qwen3xDFlash2Attention,
    residual_add: ResidualAdd,
    post_attention_norm: RMSNorm,
    mlp_conv: Qwen3xDFlash2Conv,
    mlp: Qwen3xDenseMLP,
    scratch: Rc<Qwen3xDFlash2LayerScratch>,
}

pub struct Qwen3xDFlash2LayerScratch {
    max_tokens: u32,
    hidden_dim: u32,
    residual_stream: [Buffer; 2],
    normalized_hidden: Buffer,
    prepared_hidden: Buffer,
    branch_output: Buffer,
    convolved_output: Buffer,
    post_attention_hidden: Buffer,
}

#[derive(Clone, Copy)]
pub struct Qwen3xDFlash2LayerInput<'a> {
    pub metadata: &'a BiDiBlockGQAMetadataBuffers,
    pub pages: &'a Buffer,
    pub residual_input: &'a Buffer,
    pub residual_output: &'a Buffer,
}

impl Qwen3xDFlash2Layer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        config: &Qwen3xDFlash2Config,
        num_spec_tokens: usize,
        max_requests: usize,
        dflash2_layer_index: usize,
        page_bytes: usize,
        bindings: &Qwen3xDFlash2LayerWeightBindings,
        gqa_state: &BiDiBlockGQAState,
        scratch: Rc<Qwen3xDFlash2LayerScratch>,
        dense_scratch: Rc<DenseMLPScratch>,
        scale_bias_dtype: Dtype,
    ) -> Result<Self, ModelExecutorError> {
        let (attention_core, attention_metal) = derive_qwen3x_dflash2_gqa_configs(
            config,
            num_spec_tokens,
            dflash2_layer_index,
            &bindings.gqa,
            page_bytes,
            scale_bias_dtype,
        )?;
        let (mlp_core, mlp_metal) =
            derive_qwen3x_dflash2_dense_mlp_configs(config, dflash2_layer_index, &bindings.mlp, scale_bias_dtype)?;
        let hidden_dim = attention_core.attention.hidden_dim;
        assert_eq!(hidden_dim, mlp_core.hidden_dim);
        Ok(Self {
            dflash2_layer_index,
            input_norm: RMSNorm::new(device, hidden_dim, config.rms_norm_eps),
            attention_conv: Qwen3xDFlash2Conv::new(
                device,
                config,
                num_spec_tokens,
                max_requests,
                &bindings.attention_conv,
                scale_bias_dtype,
            )?,
            attention: Qwen3xDFlash2Attention::new(
                device,
                attention_core,
                attention_metal,
                dflash2_layer_index,
                gqa_state,
            ),
            residual_add: ResidualAdd::new(device),
            post_attention_norm: RMSNorm::new(device, hidden_dim, config.rms_norm_eps),
            mlp_conv: Qwen3xDFlash2Conv::new(
                device,
                config,
                num_spec_tokens,
                max_requests,
                &bindings.mlp_conv,
                scale_bias_dtype,
            )?,
            mlp: Qwen3xDenseMLP::new(device, mlp_core, mlp_metal, dense_scratch),
            scratch,
        })
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen3xDFlash2Config,
        bindings: Qwen3xDFlash2LayerWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        let Qwen3xDFlash2LayerWeightBindings {
            input_norm_weight,
            attention_conv,
            gqa,
            post_attention_norm_weight,
            mlp_conv,
            mlp,
        } = bindings;
        self.attention_conv.load_weights(device, store, attention_conv)?;
        self.attention.load_weights(device, store, gqa)?;
        self.mlp_conv.load_weights(device, store, mlp_conv)?;
        self.mlp.load_weights(device, store, mlp)?;
        let mut tensors = store.load_tensors([input_norm_weight.as_str(), post_attention_norm_weight.as_str()])?;
        self.input_norm.load_weights(remove_qwen3x_norm_weight(
            device,
            &mut tensors,
            &input_norm_weight,
            &[config.hidden_size],
        )?);
        self.post_attention_norm.load_weights(remove_qwen3x_norm_weight(
            device,
            &mut tensors,
            &post_attention_norm_weight,
            &[config.hidden_size],
        )?);
        assert!(
            tensors.is_empty(),
            "Qwen3x DFlash2 layer must consume its norm tensor map"
        );
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        self.mlp.unload_weights();
        self.mlp_conv.unload_weights();
        self.post_attention_norm.unload_weights();
        self.attention.unload_weights();
        self.attention_conv.unload_weights();
        self.input_norm.unload_weights();
    }

    pub fn unload_state(&mut self) {
        self.attention.unload_state();
    }

    pub fn load_state(&mut self, state: &BiDiBlockGQAState) {
        self.attention.load_state(state);
    }

    pub fn residual_output(&self) -> &Buffer {
        self.scratch.residual_stream(self.dflash2_layer_index)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_prefill<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        main_feature: &'a Buffer,
        req_slots: &'a Buffer,
        flat_token_indices: &'a Buffer,
        pages: &'a Buffer,
    ) {
        self.attention.record_prefill(
            recorder,
            num_total_tokens,
            num_active_tokens,
            main_feature,
            req_slots,
            flat_token_indices,
            pages,
        );
    }

    pub fn record_bidi_block<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        num_active_query_blocks: ReplayU32,
        input: Qwen3xDFlash2LayerInput<'a>,
    ) -> &'a Buffer {
        debug_assert!(num_total_tokens <= self.scratch.max_tokens);
        self.input_norm.record_with_barrier(
            recorder,
            num_total_tokens,
            num_active_tokens,
            input.residual_input,
            &self.scratch.normalized_hidden,
        );
        self.attention_conv.record_prepare(
            recorder,
            num_total_tokens,
            num_active_tokens,
            num_active_query_blocks,
            &self.scratch.normalized_hidden,
            &self.scratch.prepared_hidden,
        );
        self.attention.record_bidi_block(
            recorder,
            num_active_tokens,
            input.metadata,
            &self.scratch.prepared_hidden,
            &self.scratch.branch_output,
            input.pages,
        );
        self.attention_conv.record_finish(
            recorder,
            num_total_tokens,
            num_active_query_blocks,
            &self.scratch.branch_output,
            &self.scratch.convolved_output,
        );
        self.residual_add.record(
            recorder,
            num_total_tokens,
            self.scratch.hidden_dim,
            num_active_tokens,
            input.residual_input,
            &self.scratch.convolved_output,
            &self.scratch.post_attention_hidden,
        );
        self.post_attention_norm.record_with_barrier(
            recorder,
            num_total_tokens,
            num_active_tokens,
            &self.scratch.post_attention_hidden,
            &self.scratch.normalized_hidden,
        );
        self.mlp_conv.record_prepare(
            recorder,
            num_total_tokens,
            num_active_tokens,
            num_active_query_blocks,
            &self.scratch.normalized_hidden,
            &self.scratch.prepared_hidden,
        );
        self.mlp.record(
            recorder,
            &self.scratch.prepared_hidden,
            &self.scratch.branch_output,
            num_total_tokens,
            num_active_tokens,
        );
        self.mlp_conv.record_finish(
            recorder,
            num_total_tokens,
            num_active_query_blocks,
            &self.scratch.branch_output,
            &self.scratch.convolved_output,
        );
        self.residual_add.record(
            recorder,
            num_total_tokens,
            self.scratch.hidden_dim,
            num_active_tokens,
            &self.scratch.post_attention_hidden,
            &self.scratch.convolved_output,
            input.residual_output,
        );
        input.residual_output
    }
}

fn derive_qwen3x_dflash2_dense_mlp_configs(
    config: &Qwen3xDFlash2Config,
    dflash2_layer_index: usize,
    bindings: &Qwen3xDenseMLPWeightBindings,
    scale_bias_dtype: Dtype,
) -> Result<(DenseMLPCore, DenseMLPMetalConfig), ModelExecutorError> {
    let quantization = config
        .quantization
        .as_ref()
        .ok_or_else(|| ModelExecutorError::custom("Qwen3x DFlash2 dense MLP requires quantization config"))?;
    let gate_up = resolve_uniform_quantization(
        quantization,
        &[bindings.gate.weight.as_str(), bindings.up.weight.as_str()],
        "Qwen3x DFlash2 dense MLP gate/up",
    )?;
    let down = quantization.resolve_for_tensor(&bindings.down.weight);
    for (component, resolved) in [("gate/up", &gate_up), ("down", &down)] {
        if !matches!(resolved.mode.as_deref(), None | Some("affine")) {
            return Err(ModelExecutorError::custom(format!(
                "Qwen3x DFlash2 dense MLP {component} requires affine quantization, got mode={:?}",
                resolved.mode
            )));
        }
    }
    let core = DenseMLPCore {
        model_layer_index: dflash2_layer_index,
        hidden_dim: config.hidden_size,
        intermediate_dim: config.intermediate_size,
    };
    core.validate();
    let metal = DenseMLPMetalConfig {
        gate_up: QuantizedAffineLayout {
            group_size: to_u32("Qwen3x DFlash2 dense MLP gate/up group_size", gate_up.group_size)?,
            bits: to_u32("Qwen3x DFlash2 dense MLP gate/up bits", gate_up.bits)?,
            scale_bias_dtype,
        },
        down: QuantizedAffineLayout {
            group_size: to_u32("Qwen3x DFlash2 dense MLP down group_size", down.group_size)?,
            bits: to_u32("Qwen3x DFlash2 dense MLP down bits", down.bits)?,
            scale_bias_dtype,
        },
        io_dtype: Dtype::Bfloat16,
    };
    metal.validate();
    Ok((core, metal))
}

impl Qwen3xDFlash2LayerScratch {
    pub fn new(device: &Device, max_tokens: usize, hidden_dim: usize) -> Self {
        assert!(max_tokens > 0 && hidden_dim > 0);
        let max_tokens_u32 = u32::try_from(max_tokens).expect("Qwen3x DFlash2 layer token capacity must fit u32");
        let hidden_dim_u32 = u32::try_from(hidden_dim).expect("Qwen3x DFlash2 hidden dimension must fit u32");
        let hidden_elements = max_tokens
            .checked_mul(hidden_dim)
            .expect("Qwen3x DFlash2 layer scratch element count must fit usize");
        u32::try_from(hidden_elements)
            .expect("Qwen3x DFlash2 layer scratch must fit the shader u32 element-count domain");
        let hidden = || Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16);
        Self {
            max_tokens: max_tokens_u32,
            hidden_dim: hidden_dim_u32,
            residual_stream: [hidden(), hidden()],
            normalized_hidden: hidden(),
            prepared_hidden: hidden(),
            branch_output: hidden(),
            convolved_output: hidden(),
            post_attention_hidden: hidden(),
        }
    }

    fn residual_stream(&self, dflash2_layer_index: usize) -> &Buffer {
        &self.residual_stream[dflash2_layer_index % self.residual_stream.len()]
    }
}
