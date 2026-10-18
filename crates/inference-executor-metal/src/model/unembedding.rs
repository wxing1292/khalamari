use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::checkpoint::remove_tensor;
use inference_executor_core::def::ModelExecutorError;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::model::embedding::QuantizedEmbeddingWeights;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnembedConfig {
    pub max_tokens: u32,
    pub vocab_size: u32,
    pub hidden_dim: u32,
    pub group_size: u32,
    pub bits: u32,
    pub input_dtype: Dtype,
    pub output_dtype: Dtype,
    pub scale_bias_dtype: Dtype,
}

impl UnembedConfig {
    pub fn validate(self) {
        assert!(self.max_tokens > 0);
        self.compute_config().validate();
        assert_eq!(self.input_dtype, Dtype::Bfloat16);
        assert_eq!(self.output_dtype, Dtype::Bfloat16);
        assert_eq!(
            self.scale_bias_dtype,
            Dtype::Bfloat16,
            "unembed requires BF16 affine parameters"
        );
    }

    pub fn logits_bytes(self) -> usize {
        self.validate();
        self.compute_config()
            .output_bytes(self.max_tokens.try_into().expect("unembed max_tokens must fit i32"))
    }

    fn compute_config(self) -> affine_quantized::Config {
        affine_quantized::Config {
            n: self.vocab_size.try_into().expect("unembed vocab_size must fit i32"),
            k: self.hidden_dim.try_into().expect("unembed hidden_dim must fit i32"),
            group_size: self.group_size.try_into().expect("unembed group_size must fit i32"),
            bits: self.bits.try_into().expect("unembed bits must fit i32"),
            input_dtype: self.input_dtype,
            output_dtype: self.output_dtype,
            scale_bias_dtype: self.scale_bias_dtype,
        }
    }
}

pub struct Unembed {
    config: UnembedConfig,
    compute: Rc<affine_quantized::Matmul>,
    weights: Option<UnembedWeightOwner>,
}

#[derive(Clone)]
enum UnembedWeightOwner {
    Independent(Rc<UnembedWeights>),
    Tied(Rc<QuantizedEmbeddingWeights>),
}

struct UnembedWeights {
    weight: Buffer,
    scales: Buffer,
    biases: Buffer,
}

#[derive(Clone, Copy)]
pub struct UnembedInput<'a> {
    pub num_total_rows: u32,
    pub num_active_rows: ReplayU32,
    pub hidden: &'a Buffer,
    pub logits: &'a Buffer,
}

impl Unembed {
    pub fn new(device: &Device, config: UnembedConfig) -> Self {
        config.validate();
        let affine_config = config.compute_config();
        Self {
            config,
            compute: Rc::new(affine_quantized::Matmul::new(device, affine_config)),
            weights: None,
        }
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: QuantizedTensorBindings,
    ) -> Result<(), ModelExecutorError> {
        assert!(self.weights.is_none(), "unembed weights are already loaded");
        self.weights = Some(UnembedWeightOwner::Independent(Rc::new(UnembedWeights::load(
            device,
            store,
            self.config.compute_config(),
            bindings,
        )?)));
        Ok(())
    }

    pub fn load_tied_weights(&mut self, weights: Rc<QuantizedEmbeddingWeights>) {
        assert!(self.weights.is_none(), "unembed weights are already loaded");
        self.weights = Some(UnembedWeightOwner::Tied(weights));
        self.validate_weights();
    }

    pub fn unload_weights(&mut self) {
        assert!(self.weights.is_some(), "unembed weights are not loaded");
        self.weights.take();
    }

    fn validate_weights(&self) {
        let config = self.config.compute_config();
        let (weight, scales, biases) = self.weight_buffers();
        assert_eq!(weight.len_bytes(), config.weight_bytes());
        assert_eq!(scales.len_bytes(), config.scale_or_bias_bytes());
        assert_eq!(biases.len_bytes(), scales.len_bytes());
    }

    fn weight_buffers(&self) -> (&Buffer, &Buffer, &Buffer) {
        match self
            .weights
            .as_ref()
            .expect("unembed weights must be loaded before execution")
        {
            UnembedWeightOwner::Independent(weights) => (&weights.weight, &weights.scales, &weights.biases),
            UnembedWeightOwner::Tied(weights) => weights.buffers(),
        }
    }

    pub fn vocab_size(&self) -> u32 {
        self.config.vocab_size
    }

    pub fn max_tokens(&self) -> u32 {
        self.config.max_tokens
    }

    /// Returns a row-capacity view that shares the immutable matmul and weights.
    pub fn with_max_tokens(&self, max_tokens: u32) -> Self {
        let config = UnembedConfig {
            max_tokens,
            ..self.config
        };
        config.validate();
        Self {
            config,
            compute: Rc::clone(&self.compute),
            weights: self.weights.clone(),
        }
    }

    pub fn replay_topology(&self, num_total_rows: u32) -> affine_quantized::KernelKind {
        self.validate_num_rows(num_total_rows);
        self.compute.topology(num_total_rows)
    }

    pub fn replay_topology_boundaries(&self) -> Box<[u32]> {
        self.compute.topology_boundaries()
    }

    fn validate_num_rows(&self, num_rows: u32) {
        assert!(num_rows > 0, "unembed requires at least one row");
        assert!(
            num_rows <= self.config.max_tokens,
            "unembed num_rows={} exceed max_tokens={}",
            num_rows,
            self.config.max_tokens
        );
    }
}

impl ReplayLayer for Unembed {
    type Input<'a> = UnembedInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.validate_num_rows(input.num_total_rows);
        let (weight, scales, biases) = self.weight_buffers();
        let invocation = self.compute.invoke(
            input.num_total_rows,
            input.num_active_rows,
            input.logits,
            0,
            input.hidden,
            0,
            weight,
            0,
            scales,
            0,
            biases,
            0,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(invocation));
        input.logits
    }
}

impl UnembedWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: affine_quantized::Config,
        bindings: QuantizedTensorBindings,
    ) -> Result<Self, ModelExecutorError> {
        let mut tensors = store.load_tensors([
            bindings.weight.as_str(),
            bindings.scales.as_str(),
            bindings.biases.as_str(),
        ])?;
        let weight = remove_tensor(&mut tensors, &bindings.weight, safetensors::Dtype::U32)?.into_data();
        let scales = remove_tensor(&mut tensors, &bindings.scales, safetensors::Dtype::BF16)?.into_data();
        let biases = remove_tensor(&mut tensors, &bindings.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len("unembed weight", weight.len(), config.weight_bytes())?;
        validate_len("unembed scales", scales.len(), config.scale_or_bias_bytes())?;
        validate_len("unembed biases", biases.len(), config.scale_or_bias_bytes())?;
        let weights = Self {
            weight: Buffer::from_slice(device, &weight),
            scales: Buffer::from_slice(device, &scales),
            biases: Buffer::from_slice(device, &biases),
        };
        assert!(tensors.is_empty(), "unembed must consume its tensor map");
        Ok(weights)
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

#[cfg(test)]
pub fn fixture_unembed(device: &Device, config: UnembedConfig) -> (Unembed, Vec<u8>, Vec<f32>, Vec<f32>) {
    use half::bf16;

    config.validate();
    let affine_config = config.compute_config();
    let weight_values = (0..affine_config.weight_bytes())
        .map(|index| ((index * 17 + 3) % 31) as u8)
        .collect::<Vec<_>>();
    let num_affine_values = affine_config.scale_or_bias_bytes() / size_of::<u16>();
    let scale_values = (0..num_affine_values)
        .map(|index| bf16::from_f32(0.000_244_140_63 + (index % 5) as f32 * 0.000_030_517_578).to_f32())
        .collect::<Vec<_>>();
    let bias_values = (0..num_affine_values)
        .map(|index| bf16::from_f32(-0.000_122_070_31 + (index % 3) as f32 * 0.000_030_517_578).to_f32())
        .collect::<Vec<_>>();
    let bf16_buffer = |values: &[f32]| {
        let bits = values
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        Buffer::from_slice(device, &bits)
    };
    let unembed = Unembed {
        config,
        compute: Rc::new(affine_quantized::Matmul::new(device, affine_config)),
        weights: Some(UnembedWeightOwner::Independent(Rc::new(UnembedWeights {
            weight: Buffer::from_slice(device, &weight_values),
            scales: bf16_buffer(&scale_values),
            biases: bf16_buffer(&bias_values),
        }))),
    };
    unembed.validate_weights();
    (unembed, weight_values, scale_values, bias_values)
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use inference_backend_metal::metal::Buffer;
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::metal::ReplayArguments;
    use inference_backend_metal::metal::ReplayParameterKey;
    use inference_backend_metal::metal::ReplayU32;
    use inference_backend_metal::metal::Stream;
    use inference_executor_core::mlp::dense::reference::QuantizedAffineReferenceShape;
    use inference_executor_core::mlp::dense::reference::quantized_affine_reference;

    use super::Dtype;
    use super::Unembed;
    use super::UnembedConfig;
    use super::UnembedInput;
    use super::fixture_unembed;
    use crate::def::layer::ReplayLayer;
    use crate::def::replay_op::MetalReplayRuntime;
    use crate::def::replay_op::ReplayRecorder;
    use crate::replay::Replay;
    use crate::replay::ReplayComponent;

    const NUM_ACTIVE_ROWS: ReplayParameterKey = ReplayParameterKey::new("test.unembed.num_active_rows");

    #[test]
    fn test_replay_matches_cpu_reference_across_active_counts() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let runtime = MetalReplayRuntime::new(&stream);
        let config = fixture_config(8);
        let (unembed, weight_values, scale_values, bias_values) = fixture_unembed(&device, config);
        let num_values_per_hidden_row = config.hidden_dim as usize;
        let num_values_per_logits_row = config.vocab_size as usize;
        let num_total_hidden_values = config.max_tokens as usize * num_values_per_hidden_row;
        let num_total_logits_values = config.max_tokens as usize * num_values_per_logits_row;
        let hidden_values = (0..num_total_hidden_values)
            .map(|index| bf16::from_f32(((index * 17 + 5) % 41) as f32 * 0.03125 - 0.625).to_f32())
            .collect::<Vec<_>>();
        let hidden = bf16_buffer(&device, &hidden_values);
        let logits = Buffer::new_zeroed_elements(&device, num_total_logits_values, Dtype::Bfloat16);
        let input = UnembedInput {
            num_total_rows: config.max_tokens,
            num_active_rows: ReplayU32::Parameter(NUM_ACTIVE_ROWS),
            hidden: &hidden,
            logits: &logits,
        };
        let mut replay = Replay::new("test Unembed", TestUnembed(unembed));
        let (key, cache_hit) = replay.record(&runtime, &input);
        assert!(!cache_hit);

        for num_active_rows in [1_usize, 8, 3, 7, 2, 6, 4, 5] {
            assert_eq!(replay.record(&runtime, &input), (key, true));
            runtime
                .submit_replay_with_arguments(
                    replay.replay(&key),
                    &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, num_active_rows as u32),
                )
                .wait();
            let expected = quantized_affine_reference(
                QuantizedAffineReferenceShape {
                    num_rows: num_active_rows,
                    output_dim: num_values_per_logits_row,
                    input_dim: num_values_per_hidden_row,
                    group_size: config.group_size as usize,
                    bits: config.bits as usize,
                },
                &hidden_values[..num_active_rows * num_values_per_hidden_row],
                &weight_values,
                &scale_values,
                &bias_values,
            );
            let actual = read_bf16_values(&logits, num_active_rows * num_values_per_logits_row);
            assert_close(&actual, &expected, 1.0e-2);
        }
    }

    struct TestUnembed(Unembed);

    impl ReplayComponent for TestUnembed {
        type Key = (u32, super::affine_quantized::KernelKind);
        type Input<'a> = UnembedInput<'a>;

        fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
            (input.num_total_rows, self.0.replay_topology(input.num_total_rows))
        }

        fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
            <Unembed as ReplayLayer>::record(&self.0, recorder, *input);
        }
    }

    fn fixture_config(max_tokens: u32) -> UnembedConfig {
        UnembedConfig {
            max_tokens,
            vocab_size: 32,
            hidden_dim: 32,
            group_size: 32,
            bits: 8,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            scale_bias_dtype: Dtype::Bfloat16,
        }
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bits = values
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        Buffer::from_slice(device, &bits)
    }

    fn read_bf16_values(buffer: &Buffer, len: usize) -> Vec<f32> {
        buffer
            .read_typed::<u16>(0, len)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let difference = (actual - expected).abs();
            assert!(
                difference <= tolerance,
                "unembed mismatch at {index}: actual={actual} expected={expected} difference={difference} \
                 tolerance={tolerance}"
            );
        }
    }
}
