use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLComputePipelineState;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayU32;

const RMS_NORM_SOURCE: &str = include_str!("metal/rms_norm.metal");

const BF16_VECTOR_WIDTH: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KernelKind {
    F32,
    Bf16Vectorized,
    Bf16VectorizedF32Weight,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants {
    kind: KernelKind,
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn new(config: Config) -> Self {
        let constants = Self {
            kind: match (config.io_dtype, config.weight_dtype) {
                (Dtype::Float32, Dtype::Float32) => KernelKind::F32,
                (Dtype::Bfloat16, Dtype::Bfloat16) => KernelKind::Bf16Vectorized,
                (Dtype::Bfloat16, Dtype::Float32) => KernelKind::Bf16VectorizedF32Weight,
                dtypes => panic!("unsupported RMSNorm IO/weight dtypes {dtypes:?}"),
            },
            thread_block: ThreadBlockConstants { required_threads: 1024 },
        };
        if constants.kind == KernelKind::F32 {
            // The scalar shader advances uint r by four values per thread.
            let loop_width = constants.thread_block.required_threads as u64 * 4;
            let padded_hidden_dim = (config.hidden_dim as u64).div_ceil(loop_width) * loop_width;
            assert!(
                padded_hidden_dim <= u32::MAX as u64,
                "F32 RMSNorm padded hidden dimension must fit the shader u32 loop domain"
            );
        }
        constants
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub hidden_dim: u32,
    pub eps: f32,
    pub io_dtype: Dtype,
    pub weight_dtype: Dtype,
}

impl Config {
    pub fn f32(hidden_dim: u32, eps: f32) -> Self {
        Self {
            hidden_dim,
            eps,
            io_dtype: Dtype::Float32,
            weight_dtype: Dtype::Float32,
        }
    }

    /// Creates a BF16 configuration. `hidden_dim` must be divisible by 4.
    pub fn bf16(hidden_dim: u32, eps: f32) -> Self {
        Self {
            hidden_dim,
            eps,
            io_dtype: Dtype::Bfloat16,
            weight_dtype: Dtype::Bfloat16,
        }
    }

    /// Creates a BF16 input/output configuration with F32 scale weights.
    pub fn bf16_with_f32_weight(hidden_dim: u32, eps: f32) -> Self {
        Self {
            hidden_dim,
            eps,
            io_dtype: Dtype::Bfloat16,
            weight_dtype: Dtype::Float32,
        }
    }

    pub fn validate(self) {
        assert!(self.hidden_dim > 0);
        assert!(self.eps.is_finite() && self.eps > 0.0);
        assert!(matches!(self.io_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert!(matches!(self.weight_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert!(
            self.io_dtype == Dtype::Bfloat16 || self.weight_dtype == Dtype::Float32,
            "F32 RMSNorm IO requires F32 weights"
        );
        assert!(
            self.io_dtype != Dtype::Bfloat16 || self.hidden_dim.is_multiple_of(BF16_VECTOR_WIDTH),
            "BF16 RMSNorm hidden_dim must be divisible by {BF16_VECTOR_WIDTH}"
        );
    }

    pub fn num_values(self, shape: Shape) -> usize {
        self.validate();
        shape.validate();
        (shape.num_total_tokens as usize)
            .checked_mul(self.hidden_dim as usize)
            .expect("RMSNorm value count must fit usize")
    }

    pub fn bytes(self, shape: Shape) -> usize {
        self.num_values(shape)
            .checked_mul(self.io_dtype.item_size())
            .expect("RMSNorm byte length must fit usize")
    }

    pub fn weight_bytes(self) -> usize {
        self.validate();
        (self.hidden_dim as usize)
            .checked_mul(self.weight_dtype.item_size())
            .expect("RMSNorm weight byte length must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_tokens: u32,
}

impl Shape {
    pub fn validate(self) {
        assert!(self.num_total_tokens > 0);
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub input: &'a Buffer,
    pub weight: &'a Buffer,
    pub output: &'a Buffer,
}

/// RMSNorm data flow:
///
/// ```text
/// buffers.input ----\
///                    +--> RMSNorm --> buffers.output
/// buffers.weight ---/
/// ```
pub struct Compute {
    config: Config,
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        let constants = KernelConstants::new(config);
        Self {
            config,
            constants,
            kernel: CompiledKernel::new(device, RMS_NORM_SOURCE, rms_norm_function_name(constants)),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, num_active_tokens: ReplayU32, buffers: Buffers<'a>) -> Invocation<'a> {
        Invocation {
            kernel: &self.kernel,
            config: self.config,
            constants: self.constants,
            shape,
            buffers,
            num_active_tokens_key: match num_active_tokens {
                ReplayU32::Fixed(value) => {
                    assert_eq!(value, shape.num_total_tokens);
                    None
                },
                ReplayU32::Parameter(key) => Some(key),
            },
        }
    }
}

pub struct Invocation<'a> {
    kernel: &'a CompiledKernel,
    config: Config,
    constants: KernelConstants,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

pub struct ReplayInvocation {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    config: Config,
    constants: KernelConstants,
    shape: Shape,
    buffers: OwnedBuffers,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

pub struct ReplayOp {
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub config: Config,
    constants: KernelConstants,
    pub shape: Shape,
    pub buffers: OwnedBuffers,
    pub num_active_tokens_key: Option<ReplayParameterKey>,
}

#[derive(Clone)]
pub struct OwnedBuffers {
    pub input: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub input_len_bytes: usize,
    pub weight: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub weight_len_bytes: usize,
    pub output: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub output_len_bytes: usize,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.input, 0);
        recorder.set_buffer_read(1, self.buffers.weight, 0);
        recorder.set_buffer_write(2, self.buffers.output, 0);
        record_num_active_tokens(recorder, 3, self.shape.num_total_tokens, self.num_active_tokens_key);
        recorder.set_u32(4, self.config.hidden_dim);
        recorder.set_f32(5, self.config.eps);
        recorder.dispatch_threadblocks(
            (self.shape.num_total_tokens as usize, 1, 1),
            (self.constants.thread_block.required_threads as usize, 1, 1),
        );
    }
}

impl Operator for ReplayInvocation {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        recorder.set_retained_pipeline_state(&self.pipeline);
        recorder.set_retained_buffer_read(0, &self.buffers.input, 0);
        recorder.set_retained_buffer_read(1, &self.buffers.weight, 0);
        recorder.set_retained_buffer_write(2, &self.buffers.output, 0);
        record_num_active_tokens(recorder, 3, self.shape.num_total_tokens, self.num_active_tokens_key);
        recorder.set_u32(4, self.config.hidden_dim);
        recorder.set_f32(5, self.config.eps);
        recorder.dispatch_threadblocks(
            (self.shape.num_total_tokens as usize, 1, 1),
            (self.constants.thread_block.required_threads as usize, 1, 1),
        );
    }
}

impl Invocation<'_> {
    pub fn into_replay_op(self) -> ReplayOp {
        ReplayOp {
            pipeline: self.kernel.as_raw_retained(),
            config: self.config,
            constants: self.constants,
            shape: self.shape,
            buffers: OwnedBuffers {
                input: self.buffers.input.as_raw_retained(),
                input_len_bytes: self.buffers.input.len_bytes(),
                weight: self.buffers.weight.as_raw_retained(),
                weight_len_bytes: self.buffers.weight.len_bytes(),
                output: self.buffers.output.as_raw_retained(),
                output_len_bytes: self.buffers.output.len_bytes(),
            },
            num_active_tokens_key: self.num_active_tokens_key,
        }
    }

    fn validate(&self) {
        assert!(self.buffers.input.len_bytes() >= self.config.bytes(self.shape));
        assert!(self.buffers.weight.len_bytes() >= self.config.weight_bytes());
        assert!(self.buffers.output.len_bytes() >= self.config.bytes(self.shape));
    }
}

impl ReplayInvocation {
    fn validate(&self) {
        assert!(self.buffers.input_len_bytes >= self.config.bytes(self.shape));
        assert!(self.buffers.weight_len_bytes >= self.config.weight_bytes());
        assert!(self.buffers.output_len_bytes >= self.config.bytes(self.shape));
    }
}

impl ReplayOp {
    pub fn into_replay(self) -> ReplayInvocation {
        ReplayInvocation {
            pipeline: self.pipeline,
            config: self.config,
            constants: self.constants,
            shape: self.shape,
            buffers: self.buffers,
            num_active_tokens_key: self.num_active_tokens_key,
        }
    }
}

fn record_num_active_tokens(
    recorder: &CommandRecorder,
    binding_index: usize,
    num_total_tokens: u32,
    key: Option<ReplayParameterKey>,
) {
    match key {
        Some(key) => recorder.bind_u32(binding_index, key, 1, num_total_tokens),
        None => recorder.set_u32(binding_index, num_total_tokens),
    }
}

fn rms_norm_function_name(constants: KernelConstants) -> &'static str {
    match constants.kind {
        KernelKind::F32 => "rms_norm_f32",
        KernelKind::Bf16Vectorized => "rms_norm_bf16_vec4",
        KernelKind::Bf16VectorizedF32Weight => "rms_norm_bf16_vec4_weight_f32",
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use inference_executor_core::reference::rms_norm_reference;

    use super::*;
    use crate::metal::ReplayArguments;
    use crate::metal::Stream;

    const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.rms_norm.num_active_tokens");

    #[test]
    fn test_bf16_io_with_f32_weight_replay_matches_reference_across_active_counts() {
        assert_replay_matches_reference(Config::bf16_with_f32_weight(64, 1.0e-6));
    }

    #[test]
    fn test_f32_replay_matches_reference_across_active_counts() {
        assert_replay_matches_reference(Config::f32(64, 1.0e-6));
    }

    #[test]
    fn test_bf16_replay_matches_reference_across_active_counts() {
        assert_replay_matches_reference(Config::bf16(64, 1.0e-6));
    }

    #[test]
    #[should_panic(expected = "BF16 RMSNorm hidden_dim must be divisible by 4")]
    fn test_bf16_rejects_non_vector_width() {
        Config::bf16(3, 1.0e-6).validate();
    }

    fn assert_replay_matches_reference(config: Config) {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = Shape { num_total_tokens: 8 };
        let kernel = Compute::new(&device, config);
        let input_values = (0..config.num_values(shape))
            .map(|index| ((index * 37) % 251) as f32 * 0.03125 - 3.5)
            .collect::<Vec<_>>();
        let weight_values = (0..config.hidden_dim)
            .map(|index| 0.5 + ((index * 19) % 31) as f32 * 0.03125)
            .collect::<Vec<_>>();
        let stored_input_values = round_values(&input_values, config.io_dtype);
        let stored_weight_values = round_values(&weight_values, config.weight_dtype);
        let input = buffer_from_values(&device, &stored_input_values, config.io_dtype);
        let weight = buffer_from_values(&device, &stored_weight_values, config.weight_dtype);
        let output = Buffer::new_zeroed(&device, config.bytes(shape));
        let replay = {
            let mut recorder = stream.create_replay_program();
            recorder.record(kernel.invoke(
                shape,
                ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
                Buffers {
                    input: &input,
                    weight: &weight,
                    output: &output,
                },
            ));
            recorder.build()
        };

        for num_active_tokens in [1_usize, 8, 3, 7, 2, 6, 4, 5] {
            stream
                .submit_replay_with_arguments(
                    &replay,
                    &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens as u32),
                )
                .wait();
            let active_values = num_active_tokens * config.hidden_dim as usize;
            let expected = rms_norm_reference(
                &stored_input_values[..active_values],
                &stored_weight_values,
                None,
                num_active_tokens,
                config.hidden_dim as usize,
                config.eps,
            );
            let actual = read_values(&output, active_values, config.io_dtype);
            let tolerance = if config.io_dtype == Dtype::Float32 {
                1.0e-5
            } else {
                0.02
            };
            assert_close(&actual, &expected, tolerance);
        }
    }

    fn round_values(values: &[f32], dtype: Dtype) -> Vec<f32> {
        match dtype {
            Dtype::Float32 => values.to_vec(),
            Dtype::Bfloat16 => values.iter().map(|value| bf16::from_f32(*value).to_f32()).collect(),
            _ => panic!("unsupported RMSNorm test dtype {dtype:?}"),
        }
    }

    fn buffer_from_values(device: &Device, values: &[f32], dtype: Dtype) -> Buffer {
        match dtype {
            Dtype::Float32 => Buffer::from_slice(device, values),
            Dtype::Bfloat16 => {
                Buffer::from_slice(
                    device,
                    &values
                        .iter()
                        .map(|value| bf16::from_f32(*value).to_bits())
                        .collect::<Vec<_>>(),
                )
            },
            _ => panic!("unsupported RMSNorm test dtype {dtype:?}"),
        }
    }

    fn read_values(buffer: &Buffer, len: usize, dtype: Dtype) -> Vec<f32> {
        match dtype {
            Dtype::Float32 => buffer.read_typed(0, len),
            Dtype::Bfloat16 => {
                buffer
                    .read_typed::<u16>(0, len)
                    .into_iter()
                    .map(bf16::from_bits)
                    .map(bf16::to_f32)
                    .collect()
            },
            _ => panic!("unsupported RMSNorm test dtype {dtype:?}"),
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "RMSNorm mismatch at {index}: expected={expected} actual={actual} tolerance={tolerance}"
            );
        }
    }
}
