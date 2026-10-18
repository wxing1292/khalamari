use std::ops::Range;

use inference_backend_metal::components::residual_add;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::remove_tensor;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2Config;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2MainFeatureWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::ReplayOp;
use crate::model::main_residual_capture::MainResidualCapture;
use crate::model::qwen::v3_x::weight::affine_parameter_safetensors_dtype;
use crate::model::qwen::v3_x::weight::remove_quant_weight;
use crate::model::qwen::v3_x::weight::remove_qwen3x_norm_weight;
use crate::model::qwen::v3_x::weight::validate_len;
use crate::model::rms_norm::RMSNorm;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Qwen3xDFlash2MainFeatureLayout {
    max_tokens: u32,
    hidden_dim: u32,
    selected_hidden_dim: u32,
}

impl Qwen3xDFlash2MainFeatureLayout {
    fn new(config: &Qwen3xDFlash2Config, max_tokens: usize) -> Self {
        assert!(max_tokens > 0, "Qwen3 DFlash2 Main-feature workspace requires tokens");
        assert!(
            !config.target_layer_ids.is_empty(),
            "Qwen3 DFlash2 Main-feature workspace requires selected decoder outputs"
        );
        let max_tokens = max_tokens
            .try_into()
            .expect("Qwen3 DFlash2 Main-feature max_tokens must fit u32");
        let num_selected_residuals = config
            .target_layer_ids
            .len()
            .try_into()
            .expect("Qwen3 DFlash2 selected decoder-output count must fit u32");
        let hidden_dim: u32 = config
            .hidden_size
            .try_into()
            .expect("Qwen3 DFlash2 Main hidden dimension must fit u32");
        let selected_hidden_dim = hidden_dim
            .checked_mul(num_selected_residuals)
            .expect("Qwen3 DFlash2 selected Main width must fit u32");
        Self {
            max_tokens,
            hidden_dim,
            selected_hidden_dim,
        }
    }

    fn main_residual_elements(self) -> usize {
        (self.max_tokens as usize)
            .checked_mul(self.selected_hidden_dim as usize)
            .expect("Qwen3 DFlash2 Main residual workspace must fit usize")
    }

    fn main_feature_elements(self) -> usize {
        (self.max_tokens as usize)
            .checked_mul(self.hidden_dim as usize)
            .expect("Qwen3 DFlash2 Main-feature workspace must fit usize")
    }
}

struct Qwen3xDFlash2MainFeatureWeights {
    fc_weight: Buffer,
    fc_scales: Buffer,
    fc_biases: Buffer,
}

pub struct Qwen3xDFlash2MainFeatureProjector {
    layout: Qwen3xDFlash2MainFeatureLayout,
    residual_columns: Vec<Option<Range<u32>>>,
    fc_config: affine_quantized::Config,
    fc: affine_quantized::Matmul,
    hidden_norm: RMSNorm,
    weights: Option<Qwen3xDFlash2MainFeatureWeights>,
    main_residuals: Buffer,
    main_feature: Buffer,
}

impl MainResidualCapture for Qwen3xDFlash2MainFeatureProjector {
    fn capture_for_model_layer(&self, model_layer_index: usize) -> Option<residual_add::CaptureTarget<'_>> {
        self.residual_columns
            .get(model_layer_index)
            .and_then(Option::as_ref)
            .map(|columns| {
                residual_add::CaptureTarget::columns(
                    &self.main_residuals,
                    self.layout.selected_hidden_dim,
                    columns.clone(),
                )
            })
    }
}

impl Qwen3xDFlash2MainFeatureProjector {
    pub fn new(
        device: &Device,
        config: &Qwen3xDFlash2Config,
        bindings: &Qwen3xDFlash2MainFeatureWeightBindings,
        max_tokens: usize,
        scale_bias_dtype: Dtype,
    ) -> Result<Self, ModelExecutorError> {
        let layout = Qwen3xDFlash2MainFeatureLayout::new(config, max_tokens);
        let quantization = config
            .quantization
            .as_ref()
            .ok_or_else(|| ModelExecutorError::custom("Qwen3x DFlash2 Main feature requires quantization config"))?
            .resolve_for_tensor(&bindings.fc.weight);
        require_affine_quantization(&quantization, &bindings.fc.weight)?;
        let fc_config = dflash2_fc_config(layout, &quantization, scale_bias_dtype);
        let num_main_layers = config.target_layer_ids.iter().copied().max().unwrap() + 1;
        let mut residual_columns = vec![None; num_main_layers];
        for (residual_slice_index, &model_layer_index) in config.target_layer_ids.iter().enumerate() {
            let columns = &mut residual_columns[model_layer_index];
            assert!(
                columns.is_none(),
                "Qwen3 DFlash2 selected Main decoder layers must be unique"
            );
            let start = residual_slice_index as u32 * layout.hidden_dim;
            *columns = Some(start..start + layout.hidden_dim);
        }
        Ok(Self {
            layout,
            residual_columns,
            fc_config,
            fc: affine_quantized::Matmul::new(device, fc_config),
            hidden_norm: RMSNorm::new(device, layout.hidden_dim as usize, config.rms_norm_eps),
            weights: None,
            main_residuals: Buffer::new_zeroed_elements(device, layout.main_residual_elements(), Dtype::Bfloat16),
            main_feature: Buffer::new_zeroed_elements(device, layout.main_feature_elements(), Dtype::Bfloat16),
        })
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen3xDFlash2Config,
        bindings: &Qwen3xDFlash2MainFeatureWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        assert!(
            self.weights.is_none(),
            "Qwen3.x DFlash2 Main-feature weights are already loaded"
        );
        let mut tensor_names = Vec::new();
        bindings.push_tensor_names(&mut tensor_names);
        let mut tensors = store.load_tensors(tensor_names)?;
        let weight = remove_quant_weight(&mut tensors, &bindings.fc.weight)?;
        let scale_bias_dtype = affine_parameter_safetensors_dtype(self.fc_config.scale_bias_dtype);
        let scales = remove_tensor(&mut tensors, &bindings.fc.scales, scale_bias_dtype)?.into_data();
        let biases = remove_tensor(&mut tensors, &bindings.fc.biases, scale_bias_dtype)?.into_data();
        validate_len(
            "Qwen3 DFlash2 Main FC weight",
            weight.len(),
            self.fc_config.weight_bytes(),
        )?;
        validate_len(
            "Qwen3 DFlash2 Main FC scales",
            scales.len(),
            self.fc_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "Qwen3 DFlash2 Main FC biases",
            biases.len(),
            self.fc_config.scale_or_bias_bytes(),
        )?;
        self.hidden_norm.load_weights(remove_qwen3x_norm_weight(
            device,
            &mut tensors,
            &bindings.hidden_norm_weight,
            &[self.layout.hidden_dim as usize],
        )?);
        self.weights = Some(Qwen3xDFlash2MainFeatureWeights {
            fc_weight: Buffer::from_slice(device, &weight),
            fc_scales: Buffer::from_slice(device, &scales),
            fc_biases: Buffer::from_slice(device, &biases),
        });
        assert!(
            tensors.is_empty(),
            "Qwen3x DFlash2 Main-feature projector must consume its tensor map"
        );
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        assert!(
            self.weights.is_some(),
            "Qwen3.x DFlash2 Main-feature weights are not loaded"
        );
        self.weights.take();
        self.hidden_norm.unload_weights();
    }

    pub fn record<'a, R>(&'a self, recorder: &mut R, num_total_tokens: u32, num_active_tokens: ReplayU32) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        assert!(num_total_tokens > 0, "Qwen3 DFlash2 Main projection requires tokens");
        assert!(
            num_total_tokens <= self.layout.max_tokens,
            "Qwen3 DFlash2 Main projection exceeds capacity"
        );
        let weights = self
            .weights
            .as_ref()
            .expect("Qwen3.x DFlash2 Main-feature weights must be loaded before execution");
        recorder.record_with_barrier_before(ReplayOp::opaque(self.fc.invoke(
            num_total_tokens,
            num_active_tokens,
            &self.main_feature,
            0,
            &self.main_residuals,
            0,
            &weights.fc_weight,
            0,
            &weights.fc_scales,
            0,
            &weights.fc_biases,
            0,
        )));
        self.hidden_norm.record_with_barrier(
            recorder,
            num_total_tokens,
            num_active_tokens,
            &self.main_feature,
            &self.main_feature,
        );
        &self.main_feature
    }
}

fn dflash2_fc_config(
    layout: Qwen3xDFlash2MainFeatureLayout,
    quantization: &inference_executor_core::model::qwen::v3_x::ResolvedQuantizationConfig,
    scale_bias_dtype: Dtype,
) -> affine_quantized::Config {
    affine_quantized::Config {
        n: layout
            .hidden_dim
            .try_into()
            .expect("Qwen3 DFlash2 Main hidden_dim must fit i32"),
        k: layout
            .selected_hidden_dim
            .try_into()
            .expect("Qwen3 DFlash2 selected Main width must fit i32"),
        group_size: quantization
            .group_size
            .try_into()
            .expect("Qwen3 DFlash2 Main FC group_size must fit i32"),
        bits: quantization
            .bits
            .try_into()
            .expect("Qwen3 DFlash2 Main FC bits must fit i32"),
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype,
    }
}

fn require_affine_quantization(
    quantization: &inference_executor_core::model::qwen::v3_x::ResolvedQuantizationConfig,
    tensor_name: &str,
) -> Result<(), ModelExecutorError> {
    if !matches!(quantization.mode.as_deref(), None | Some("affine")) {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3x DFlash2 Main-feature tensor {tensor_name:?} requires affine quantization, got mode={:?}",
            quantization.mode
        )));
    }
    Ok(())
}
