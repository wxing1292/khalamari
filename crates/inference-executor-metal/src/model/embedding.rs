use std::rc::Rc;

use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::embedding;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::checkpoint::remove_tensor;
use inference_executor_core::def::ModelExecutorError;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbedConfig {
    pub max_tokens: u32,
    pub vocab_size: u32,
    pub hidden_dim: u32,
    pub group_size: u32,
    pub bits: u32,
    pub scale_bias_dtype: Dtype,
    pub output_dtype: Dtype,
}

impl EmbedConfig {
    pub fn validate(self) {
        assert!(self.max_tokens > 0);
        self.compute_config().validate();
    }

    fn compute_config(self) -> embedding::Config {
        embedding::Config {
            vocab_size: self.vocab_size,
            hidden_dim: self.hidden_dim,
            group_size: self.group_size,
            bits: self.bits,
            scale_bias_dtype: self.scale_bias_dtype,
            output_dtype: self.output_dtype,
        }
    }
}

pub struct Embed {
    config: EmbedConfig,
    compute: Rc<embedding::Compute>,
    weights: Option<Rc<QuantizedEmbeddingWeights>>,
}

pub struct QuantizedEmbeddingWeights {
    weight: Buffer,
    scales: Buffer,
    biases: Buffer,
}

#[derive(Clone, Copy)]
pub struct EmbedInput<'a> {
    pub num_total_tokens: u32,
    pub num_active_tokens: ReplayU32,
    pub token_ids: &'a Buffer,
    pub output_hidden: &'a Buffer,
}

impl Embed {
    /// Returns the token-row capacity of the embedding buffers.
    pub fn max_tokens(&self) -> u32 {
        self.config.max_tokens
    }

    /// Returns a token-capacity view that shares the immutable kernel and weights.
    pub fn with_max_tokens(&self, max_tokens: u32) -> Self {
        let config = EmbedConfig {
            max_tokens,
            ..self.config
        };
        config.validate();
        Self {
            config,
            compute: Rc::clone(&self.compute),
            weights: self.weights.as_ref().map(Rc::clone),
        }
    }

    pub fn new(device: &Device, config: EmbedConfig) -> Self {
        config.validate();
        Self {
            config,
            compute: Rc::new(embedding::Compute::new(device, config.compute_config())),
            weights: None,
        }
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: QuantizedTensorBindings,
    ) -> Result<(), ModelExecutorError> {
        assert!(self.weights.is_none(), "embedding weights are already loaded");
        self.weights = Some(Rc::new(QuantizedEmbeddingWeights::load(
            device,
            store,
            self.config.compute_config(),
            bindings,
        )?));
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        assert!(self.weights.is_some(), "embedding weights are not loaded");
        self.weights.take();
    }

    pub fn tied_weights(&self) -> Rc<QuantizedEmbeddingWeights> {
        Rc::clone(
            self.weights
                .as_ref()
                .expect("embedding weights must be loaded before they are tied"),
        )
    }

    fn weights(&self) -> &QuantizedEmbeddingWeights {
        self.weights
            .as_deref()
            .expect("embedding weights must be loaded before execution")
    }

    fn validate_num_tokens(&self, num_tokens: u32) {
        assert!(num_tokens > 0, "embedding requires at least one token");
        assert!(
            num_tokens <= self.config.max_tokens,
            "embedding num_tokens={} exceed max_tokens={}",
            num_tokens,
            self.config.max_tokens
        );
    }
}

impl ReplayLayer for Embed {
    type Input<'a> = EmbedInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: Self::Input<'a>) -> Self::Output<'a> {
        self.validate_num_tokens(input.num_total_tokens);
        let weights = self.weights();
        let shape = embedding::Shape {
            num_total_tokens: input.num_total_tokens,
        };
        let buffers = embedding::Buffers {
            token_ids: input.token_ids,
            weight: &weights.weight,
            scales: &weights.scales,
            biases: &weights.biases,
            output: input.output_hidden,
        };
        let invocation = self.compute.invoke(shape, input.num_active_tokens, buffers);
        recorder.record_with_barrier_before(ReplayOp::opaque(invocation));
        input.output_hidden
    }
}

impl QuantizedEmbeddingWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: embedding::Config,
        bindings: QuantizedTensorBindings,
    ) -> Result<Self, ModelExecutorError> {
        let mut tensors = store.load_tensors([
            bindings.weight.as_str(),
            bindings.scales.as_str(),
            bindings.biases.as_str(),
        ])?;
        let weight = remove_tensor(&mut tensors, &bindings.weight, safetensors::Dtype::U32)?.into_data();
        let scale_bias_dtype = safetensors_dtype(config.scale_bias_dtype);
        let scales = remove_tensor(&mut tensors, &bindings.scales, scale_bias_dtype)?.into_data();
        let biases = remove_tensor(&mut tensors, &bindings.biases, scale_bias_dtype)?.into_data();
        validate_len("embed weight", weight.len(), config.weight_bytes())?;
        validate_len(
            "embed scales",
            scales.len(),
            config.num_affine_params() * config.scale_bias_dtype.item_size(),
        )?;
        validate_len("embed biases", biases.len(), scales.len())?;
        let weights = Self {
            weight: Buffer::from_slice(device, &weight),
            scales: Buffer::from_slice(device, &scales),
            biases: Buffer::from_slice(device, &biases),
        };
        assert!(tensors.is_empty(), "embed must consume its tensor map");
        Ok(weights)
    }

    pub fn buffers(&self) -> (&Buffer, &Buffer, &Buffer) {
        (&self.weight, &self.scales, &self.biases)
    }
}

fn safetensors_dtype(dtype: Dtype) -> safetensors::Dtype {
    match dtype {
        Dtype::Bfloat16 => safetensors::Dtype::BF16,
        Dtype::Float32 => safetensors::Dtype::F32,
        dtype => panic!("unsupported quantized embedding scale/bias dtype {dtype:?}"),
    }
}

fn validate_len(name: &str, actual: usize, expected: usize) -> Result<(), ModelExecutorError> {
    if actual != expected {
        return Err(ModelExecutorError::custom(format!(
            "{name} byte length mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}
