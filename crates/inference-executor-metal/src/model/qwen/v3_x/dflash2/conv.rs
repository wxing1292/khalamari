use inference_backend_metal::components::dynamic_grouped_conv;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::remove_tensor;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2Config;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2ConvWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::replay_op::ReplayOp;
use crate::model::qwen::v3_x::weight::affine_parameter_safetensors_dtype;
use crate::model::qwen::v3_x::weight::remove_quant_weight;
use crate::model::qwen::v3_x::weight::to_u32;
use crate::model::qwen::v3_x::weight::validate_len;

struct Qwen3xDFlash2ConvWeights {
    base: Buffer,
    projection_weight: Buffer,
    projection_scales: Buffer,
    projection_biases: Buffer,
}

pub struct Qwen3xDFlash2Conv {
    spec_block_size: u32,
    max_requests: u32,
    projection_config: affine_quantized::Config,
    projection: affine_quantized::Matmul,
    conv_config: dynamic_grouped_conv::Config,
    conv: dynamic_grouped_conv::Compute,
    weights: Option<Qwen3xDFlash2ConvWeights>,
    projected_coefficients: Buffer,
}

impl Qwen3xDFlash2Conv {
    pub fn new(
        device: &Device,
        config: &Qwen3xDFlash2Config,
        num_spec_tokens: usize,
        max_requests: usize,
        bindings: &Qwen3xDFlash2ConvWeightBindings,
        scale_bias_dtype: Dtype,
    ) -> Result<Self, ModelExecutorError> {
        let spec_block_size = num_spec_tokens
            .checked_add(1)
            .expect("Qwen3x DFlash2 query block size must fit usize");
        let quantization = config
            .quantization
            .as_ref()
            .ok_or_else(|| ModelExecutorError::custom("Qwen3x DFlash2 convolution requires quantization config"))?
            .resolve_for_tensor(&bindings.kernel_projection.weight);
        if !matches!(quantization.mode.as_deref(), None | Some("affine")) {
            return Err(ModelExecutorError::custom(format!(
                "Qwen3x DFlash2 convolution projection requires affine quantization, got mode={:?}",
                quantization.mode
            )));
        }
        let conv_config = dynamic_grouped_conv::Config {
            hidden_dim: to_u32("Qwen3x DFlash2 convolution hidden dimension", config.hidden_size)?,
            group_size: to_u32("Qwen3x DFlash2 convolution group size", config.conv_group_size)?,
            kernel_size: to_u32("Qwen3x DFlash2 convolution kernel size", config.conv_kernel_size)?,
            io_dtype: Dtype::Bfloat16,
            base_dtype: Dtype::Bfloat16,
        };
        conv_config.validate();
        let projection_config = affine_quantized::Config {
            n: conv_config
                .projection_dim()
                .try_into()
                .expect("Qwen3x DFlash2 convolution projection width must fit i32"),
            k: config
                .hidden_size
                .try_into()
                .expect("Qwen3x DFlash2 convolution hidden dimension must fit i32"),
            group_size: quantization
                .group_size
                .try_into()
                .expect("Qwen3x DFlash2 convolution affine group size must fit i32"),
            bits: quantization
                .bits
                .try_into()
                .expect("Qwen3x DFlash2 convolution affine bits must fit i32"),
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            scale_bias_dtype,
        };
        projection_config.validate();
        let shape = dynamic_grouped_conv::Shape {
            num_total_query_blocks: max_requests
                .try_into()
                .expect("Qwen3x DFlash2 convolution request capacity must fit u32"),
            query_block_size: spec_block_size
                .try_into()
                .expect("Qwen3x DFlash2 convolution Spec block size must fit u32"),
        };
        shape.validate();
        Ok(Self {
            spec_block_size: shape.query_block_size,
            max_requests: shape.num_total_query_blocks,
            projection_config,
            projection: affine_quantized::Matmul::new(device, projection_config),
            conv_config,
            conv: dynamic_grouped_conv::Compute::new(device, conv_config),
            weights: None,
            projected_coefficients: Buffer::new_zeroed(device, conv_config.projected_coefficients_bytes(shape)),
        })
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: Qwen3xDFlash2ConvWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        assert!(
            self.weights.is_none(),
            "Qwen3x DFlash2 convolution weights are already loaded"
        );
        let mut names = Vec::new();
        bindings.push_tensor_names(&mut names);
        let mut tensors = store.load_tensors(names)?;
        let base = remove_tensor(&mut tensors, &bindings.base_kernel, safetensors::Dtype::BF16)?.into_data();
        let projection_weight = remove_quant_weight(&mut tensors, &bindings.kernel_projection.weight)?;
        let scale_bias_dtype = affine_parameter_safetensors_dtype(self.projection_config.scale_bias_dtype);
        let projection_scales =
            remove_tensor(&mut tensors, &bindings.kernel_projection.scales, scale_bias_dtype)?.into_data();
        let projection_biases =
            remove_tensor(&mut tensors, &bindings.kernel_projection.biases, scale_bias_dtype)?.into_data();
        validate_len(
            "Qwen3x DFlash2 convolution base",
            base.len(),
            self.conv_config.base_bytes(),
        )?;
        validate_len(
            "Qwen3x DFlash2 convolution projection weight",
            projection_weight.len(),
            self.projection_config.weight_bytes(),
        )?;
        validate_len(
            "Qwen3x DFlash2 convolution projection scales",
            projection_scales.len(),
            self.projection_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "Qwen3x DFlash2 convolution projection biases",
            projection_biases.len(),
            self.projection_config.scale_or_bias_bytes(),
        )?;
        self.weights = Some(Qwen3xDFlash2ConvWeights {
            base: Buffer::from_slice(device, &base),
            projection_weight: Buffer::from_slice(device, &projection_weight),
            projection_scales: Buffer::from_slice(device, &projection_scales),
            projection_biases: Buffer::from_slice(device, &projection_biases),
        });
        assert!(
            tensors.is_empty(),
            "Qwen3x DFlash2 convolution must consume its tensor map"
        );
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        assert!(
            self.weights.is_some(),
            "Qwen3x DFlash2 convolution weights are not loaded"
        );
        self.weights.take();
    }

    pub fn record_prepare<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        num_active_query_blocks: ReplayU32,
        hidden: &'a Buffer,
        output: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let (shape, weights) = self.shape_and_weights(num_total_tokens);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.projection.invoke(
            num_total_tokens,
            num_active_tokens,
            &self.projected_coefficients,
            0,
            hidden,
            0,
            &weights.projection_weight,
            0,
            &weights.projection_scales,
            0,
            &weights.projection_biases,
            0,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.conv.invoke(
            shape,
            num_active_query_blocks,
            dynamic_grouped_conv::Side::Prepare,
            dynamic_grouped_conv::Buffers {
                hidden,
                projected_coefficients: &self.projected_coefficients,
                base: &weights.base,
                output,
            },
        )));
    }

    pub fn record_finish<'a, R>(
        &'a self,
        recorder: &mut R,
        num_total_tokens: u32,
        num_active_query_blocks: ReplayU32,
        hidden: &'a Buffer,
        output: &'a Buffer,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let (shape, weights) = self.shape_and_weights(num_total_tokens);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.conv.invoke(
            shape,
            num_active_query_blocks,
            dynamic_grouped_conv::Side::Finish,
            dynamic_grouped_conv::Buffers {
                hidden,
                projected_coefficients: &self.projected_coefficients,
                base: &weights.base,
                output,
            },
        )));
    }

    fn shape_and_weights(&self, num_tokens: u32) -> (dynamic_grouped_conv::Shape, &Qwen3xDFlash2ConvWeights) {
        assert!(num_tokens > 0 && num_tokens.is_multiple_of(self.spec_block_size));
        let num_requests = num_tokens / self.spec_block_size;
        assert!(num_requests <= self.max_requests);
        let shape = dynamic_grouped_conv::Shape {
            num_total_query_blocks: num_requests,
            query_block_size: self.spec_block_size,
        };
        let weights = self
            .weights
            .as_ref()
            .expect("Qwen3x DFlash2 convolution weights must be loaded before execution");
        (shape, weights)
    }
}
