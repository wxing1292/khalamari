use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::TensorMap;
use inference_executor_core::checkpoint::remove_tensor;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGDNWeightBindings;

use crate::attn::gdn::backend::GDN;
use crate::attn::gdn::backend::GDNInput;
use crate::attn::gdn::backend::GDNLayerStateBindings;
use crate::attn::gdn::backend::GDNMetalConfig;
use crate::attn::gdn::backend::GDNWeights;
use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;
use crate::attn::gdn::scratch::GDNScratch;
use crate::attn::gdn::state_table::GDNRequestStateResources;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_x::state::Qwen3xGDNState;
use crate::model::qwen::v3_x::weight::affine_config;
use crate::model::qwen::v3_x::weight::concat_bytes;
use crate::model::qwen::v3_x::weight::remove_quant_weight;
use crate::model::qwen::v3_x::weight::validate_len;

pub struct Qwen3xGDN {
    gdn_layer_index: usize,
    core: GDNCore,
    metal: GDNMetalConfig,
    weights: Option<Qwen3xGDNWeights>,
    backend: Option<Rc<GDN>>,
    scratch: Option<Rc<GDNScratch>>,
    request_state_resources: Option<Rc<GDNRequestStateResources>>,
}

impl Qwen3xGDN {
    pub fn new(
        gdn_layer_index: usize,
        core: GDNCore,
        metal: GDNMetalConfig,
        backend: Rc<GDN>,
        scratch: Rc<GDNScratch>,
        request_state_resources: Rc<GDNRequestStateResources>,
    ) -> Self {
        Self {
            gdn_layer_index,
            core,
            metal,
            weights: None,
            backend: Some(backend),
            scratch: Some(scratch),
            request_state_resources: Some(request_state_resources),
        }
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: Qwen3xGDNWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        assert!(self.weights.is_none(), "Qwen3.x GDN weights are already loaded");
        self.weights = Some(Qwen3xGDNWeights::load(
            device, store, &bindings, &self.core, self.metal,
        )?);
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        assert!(self.weights.is_some(), "Qwen3.x GDN weights are not loaded");
        self.weights.take();
    }

    pub fn unload_state(&mut self) {
        assert!(
            self.backend.is_some() && self.scratch.is_some() && self.request_state_resources.is_some(),
            "Qwen3.x GDN state is not loaded"
        );
        self.request_state_resources.take();
        self.scratch.take();
        self.backend.take();
    }

    pub fn load_state(&mut self, state: &Qwen3xGDNState) {
        assert!(
            self.backend.is_none() && self.scratch.is_none() && self.request_state_resources.is_none(),
            "Qwen3.x GDN state is already loaded"
        );
        self.backend = Some(Rc::clone(state.backend()));
        self.scratch = Some(Rc::clone(state.scratch()));
        self.request_state_resources = Some(Rc::clone(state.request_state_resources()));
    }

    fn weights(&self) -> &Qwen3xGDNWeights {
        self.weights
            .as_ref()
            .expect("Qwen3.x GDN weights must be loaded before execution")
    }

    pub fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        input: &'a Buffer,
        output: &'a Buffer,
        metadata: &'a GDNMetadataBuffers,
        num_active_tokens: ReplayU32,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let state = self.request_state_resources().layer_bindings(self.gdn_layer_index);
        let _ = <GDN as ReplayLayer>::record(
            self.backend(),
            recorder,
            GDNInput {
                hidden_state: input,
                next_hidden_state: output,
                scratch: self.scratch().bindings(),
                batch_metadata: metadata,
                state: GDNLayerStateBindings {
                    conv_state: state.conv_states,
                    conv_state_offset_bytes: state.conv_layer_offset_bytes,
                    next_conv_state: state.conv_states,
                    next_conv_state_offset_bytes: state.conv_layer_offset_bytes,
                    recurrent_state_arena: state.recurrent_states,
                    recurrent_state_arena_offset_bytes: state.recurrent_layer_offset_bytes,
                    replay: state.replay,
                },
                weights: self.weights().as_borrowed(),
                num_active_tokens,
            },
        );
    }

    fn backend(&self) -> &GDN {
        self.backend
            .as_deref()
            .expect("Qwen3.x GDN state must be loaded before execution")
    }

    fn scratch(&self) -> &GDNScratch {
        self.scratch
            .as_deref()
            .expect("Qwen3.x GDN state must be loaded before execution")
    }

    fn request_state_resources(&self) -> &GDNRequestStateResources {
        self.request_state_resources
            .as_deref()
            .expect("Qwen3.x GDN state must be loaded before execution")
    }
}

struct Qwen3xGDNWeights {
    qkvabz_weight: Buffer,
    qkvabz_scales: Buffer,
    qkvabz_biases: Buffer,
    conv_weight: Buffer,
    norm_weight: Buffer,
    a_log: Buffer,
    dt_bias: Buffer,
    output_weight: Buffer,
    output_scales: Buffer,
    output_biases: Buffer,
}

impl Qwen3xGDNWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen3xGDNWeightBindings,
        core: &GDNCore,
        metal: crate::attn::gdn::backend::GDNMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        let mut tensor_names = Vec::new();
        bindings.push_tensor_names(&mut tensor_names);
        let mut tensors = store.load_tensors(tensor_names)?;
        let weights = Self::from_tensors(device, &mut tensors, bindings, core, metal)?;
        assert!(tensors.is_empty(), "GDN must consume its tensor map");
        Ok(weights)
    }

    fn from_tensors(
        device: &Device,
        tensors: &mut TensorMap,
        bindings: &Qwen3xGDNWeightBindings,
        core: &GDNCore,
        metal: crate::attn::gdn::backend::GDNMetalConfig,
    ) -> Result<Self, ModelExecutorError> {
        core.validate();
        metal.validate();
        let qkv_weight = remove_quant_weight(tensors, &bindings.qkv.weight)?;
        let a_weight = remove_quant_weight(tensors, &bindings.a.weight)?;
        let b_weight = remove_quant_weight(tensors, &bindings.b.weight)?;
        let z_weight = remove_quant_weight(tensors, &bindings.z.weight)?;
        let qkv_scales = remove_tensor(tensors, &bindings.qkv.scales, safetensors::Dtype::BF16)?.into_data();
        let a_scales = remove_tensor(tensors, &bindings.a.scales, safetensors::Dtype::BF16)?.into_data();
        let b_scales = remove_tensor(tensors, &bindings.b.scales, safetensors::Dtype::BF16)?.into_data();
        let z_scales = remove_tensor(tensors, &bindings.z.scales, safetensors::Dtype::BF16)?.into_data();
        let qkv_biases = remove_tensor(tensors, &bindings.qkv.biases, safetensors::Dtype::BF16)?.into_data();
        let a_biases = remove_tensor(tensors, &bindings.a.biases, safetensors::Dtype::BF16)?.into_data();
        let b_biases = remove_tensor(tensors, &bindings.b.biases, safetensors::Dtype::BF16)?.into_data();
        let z_biases = remove_tensor(tensors, &bindings.z.biases, safetensors::Dtype::BF16)?.into_data();
        let qkvabz_weight = concat_bytes(&[&qkv_weight, &a_weight, &b_weight, &z_weight]);
        let qkvabz_scales = concat_bytes(&[&qkv_scales, &a_scales, &b_scales, &z_scales]);
        let qkvabz_biases = concat_bytes(&[&qkv_biases, &a_biases, &b_biases, &z_biases]);

        let qkvabz_config = affine_config(
            core.qkvabz_dim(),
            core.hidden_dim,
            metal.group_size,
            metal.bits,
            metal.input_dtype,
            Dtype::Bfloat16,
            metal.qkvabz_scale_bias_dtype,
        );
        validate_len("GDN qkvabz weight", qkvabz_weight.len(), qkvabz_config.weight_bytes())?;
        validate_len(
            "GDN qkvabz scales",
            qkvabz_scales.len(),
            qkvabz_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "GDN qkvabz biases",
            qkvabz_biases.len(),
            qkvabz_config.scale_or_bias_bytes(),
        )?;

        let output_config = affine_config(
            core.hidden_dim,
            core.v_dim(),
            metal.group_size,
            metal.bits,
            Dtype::Bfloat16,
            metal.output_dtype,
            metal.output_scale_bias_dtype,
        );
        let output_weight = remove_quant_weight(tensors, &bindings.output.weight)?;
        let output_scales = remove_tensor(tensors, &bindings.output.scales, safetensors::Dtype::BF16)?.into_data();
        let output_biases = remove_tensor(tensors, &bindings.output.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len("GDN output weight", output_weight.len(), output_config.weight_bytes())?;
        validate_len(
            "GDN output scales",
            output_scales.len(),
            output_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "GDN output biases",
            output_biases.len(),
            output_config.scale_or_bias_bytes(),
        )?;

        let conv_weight = remove_tensor(tensors, &bindings.conv_weight, safetensors::Dtype::BF16)?.into_data();
        validate_len(
            "GDN conv weight",
            conv_weight.len(),
            core.qkv_dim() * core.conv_kernel_size * Dtype::Bfloat16.item_size(),
        )?;
        let norm_weight = remove_tensor(tensors, &bindings.norm_weight, safetensors::Dtype::BF16)?.into_data();
        validate_len(
            "GDN norm weight",
            norm_weight.len(),
            core.v_head_dim * Dtype::Bfloat16.item_size(),
        )?;
        let a_log = remove_tensor(tensors, &bindings.a_log, safetensors::Dtype::BF16)?.into_data();
        let dt_bias = remove_tensor(tensors, &bindings.dt_bias, safetensors::Dtype::BF16)?.into_data();
        validate_len("GDN A_log", a_log.len(), core.num_v_heads * Dtype::Bfloat16.item_size())?;
        validate_len(
            "GDN dt_bias",
            dt_bias.len(),
            core.num_v_heads * Dtype::Bfloat16.item_size(),
        )?;

        Ok(Self {
            qkvabz_weight: Buffer::from_slice(device, &qkvabz_weight),
            qkvabz_scales: Buffer::from_slice(device, &qkvabz_scales),
            qkvabz_biases: Buffer::from_slice(device, &qkvabz_biases),
            conv_weight: Buffer::from_slice(device, &conv_weight),
            norm_weight: Buffer::from_slice(device, &norm_weight),
            a_log: Buffer::from_slice(device, &a_log),
            dt_bias: Buffer::from_slice(device, &dt_bias),
            output_weight: Buffer::from_slice(device, &output_weight),
            output_scales: Buffer::from_slice(device, &output_scales),
            output_biases: Buffer::from_slice(device, &output_biases),
        })
    }

    fn as_borrowed(&self) -> GDNWeights<'_> {
        GDNWeights {
            qkvabz_weight: &self.qkvabz_weight,
            qkvabz_scales: &self.qkvabz_scales,
            qkvabz_biases: &self.qkvabz_biases,
            conv_weight: &self.conv_weight,
            norm_weight: &self.norm_weight,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            output_weight: &self.output_weight,
            output_scales: &self.output_scales,
            output_biases: &self.output_biases,
        }
    }
}
