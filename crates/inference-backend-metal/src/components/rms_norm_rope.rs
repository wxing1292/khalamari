use crate::components::assert_u32_count_domain;
use crate::components::assert_u32_index_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const RMS_NORM_ROPE_SOURCE: &str = include_str!("metal/rms_norm_rope.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct KernelConstants {
    config: Config,
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn new(config: Config) -> Self {
        config.validate();
        let constants = Self {
            config,
            thread_block: ThreadBlockConstants { required_threads: 128 },
        };
        // The reduction shader advances uint r by four values per thread.
        let loop_width = constants.thread_block.required_threads as u64 * 4;
        let padded_head_dim = (config.head_dim as u64).div_ceil(loop_width) * loop_width;
        assert!(
            padded_head_dim <= u32::MAX as u64,
            "RMSNorm/RoPE padded head dimension must fit the shader u32 loop domain"
        );
        constants
    }

    fn num_threads(self, shape: Shape) -> usize {
        checked_product(
            "RMSNorm/RoPE thread count",
            &[
                self.config.num_token_heads(shape),
                self.thread_block.required_threads as usize,
            ],
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeScaling {
    Default,
    Yarn {
        factor: f32,
        attention_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        original_max_position_embeddings: u32,
        truncate: bool,
    },
}

struct YarnKernelParameters {
    factor: f32,
    attention_factor: f32,
    correction_low: f32,
    correction_high: f32,
}

struct RopeKernelParameters {
    attention_factor: f32,
    inverse_frequencies: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub num_heads: u32,
    pub head_dim: u32,
    pub rope_dim: u32,
    pub eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: RopeScaling,
    pub dtype: Dtype,
    pub norm_weight_dtype: Dtype,
}

impl Config {
    pub fn f32(num_heads: u32, head_dim: u32, rope_dim: u32, eps: f32, rope_theta: f32) -> Self {
        Self {
            num_heads,
            head_dim,
            rope_dim,
            eps,
            rope_theta,
            rope_scaling: RopeScaling::Default,
            dtype: Dtype::Float32,
            norm_weight_dtype: Dtype::Bfloat16,
        }
    }

    pub fn bf16(num_heads: u32, head_dim: u32, rope_dim: u32, eps: f32, rope_theta: f32) -> Self {
        Self {
            num_heads,
            head_dim,
            rope_dim,
            eps,
            rope_theta,
            rope_scaling: RopeScaling::Default,
            dtype: Dtype::Bfloat16,
            norm_weight_dtype: Dtype::Bfloat16,
        }
    }

    pub fn with_rope_scaling(mut self, rope_scaling: RopeScaling) -> Self {
        self.rope_scaling = rope_scaling;
        self
    }

    pub fn with_norm_weight_dtype(mut self, norm_weight_dtype: Dtype) -> Self {
        self.norm_weight_dtype = norm_weight_dtype;
        self
    }

    pub fn validate(self) {
        assert!(self.num_heads > 0);
        assert!(self.head_dim > 0);
        assert!(self.rope_dim > 0);
        assert!(self.rope_dim <= self.head_dim);
        assert_eq!(self.rope_dim % 2, 0);
        assert!(self.eps.is_finite() && self.eps > 0.0);
        assert!(self.rope_theta.is_finite() && self.rope_theta > 0.0);
        self.rope_scaling.validate();
        assert!(
            !matches!(self.rope_scaling, RopeScaling::Yarn { .. }) || self.rope_theta > 1.0,
            "Yarn rope_theta must be greater than 1"
        );
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert!(matches!(self.norm_weight_dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn num_slots(self, shape: Shape) -> usize {
        checked_product(
            "RMSNorm/RoPE element count",
            &[
                shape.num_total_tokens as usize,
                self.num_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    pub fn bytes(self, shape: Shape) -> usize {
        checked_product(
            "RMSNorm/RoPE byte length",
            &[self.num_slots(shape), self.dtype.item_size()],
        )
    }

    pub fn norm_weight_bytes(self) -> usize {
        checked_product(
            "RMSNorm/RoPE weight byte length",
            &[self.head_dim as usize, self.norm_weight_dtype.item_size()],
        )
    }

    pub fn flat_token_indices_bytes(self, shape: Shape) -> usize {
        checked_product(
            "RMSNorm/RoPE token-index byte length",
            &[shape.num_total_tokens as usize, size_of::<u32>()],
        )
    }

    fn num_token_heads(self, shape: Shape) -> usize {
        checked_product(
            "RMSNorm/RoPE token-head row count",
            &[shape.num_total_tokens as usize, self.num_heads as usize],
        )
    }
}

impl RopeScaling {
    pub fn validate(self) {
        match self {
            Self::Default => {},
            Self::Yarn {
                factor,
                attention_factor,
                beta_fast,
                beta_slow,
                original_max_position_embeddings,
                ..
            } => {
                assert!(factor.is_finite() && factor >= 1.0);
                assert!(attention_factor.is_finite() && attention_factor > 0.0);
                assert!(beta_fast.is_finite() && beta_fast > 0.0);
                assert!(beta_slow.is_finite() && beta_slow > 0.0);
                assert!(beta_fast >= beta_slow);
                assert!(original_max_position_embeddings > 0);
            },
        }
    }

    fn yarn_kernel_parameters(self, rope_dim: u32, rope_theta: f32) -> Option<YarnKernelParameters> {
        match self {
            Self::Default => None,
            Self::Yarn {
                factor,
                attention_factor,
                beta_fast,
                beta_slow,
                original_max_position_embeddings,
                truncate,
            } => {
                let correction = |rotations: f32| {
                    let dimension = f64::from(rope_dim);
                    let base = f64::from(rope_theta);
                    let original_length = f64::from(original_max_position_embeddings);
                    let rotations = f64::from(rotations);
                    (dimension * (original_length / (rotations * std::f64::consts::TAU)).ln() / (2.0 * base.ln()))
                        as f32
                };
                let mut low = correction(beta_fast).max(0.0);
                let mut high = correction(beta_slow).min(rope_dim.saturating_sub(1) as f32);
                if truncate {
                    low = low.floor();
                    high = high.ceil();
                }
                assert!(low <= high, "Yarn correction range must be ordered");
                if low == high {
                    high += 0.001;
                }
                Some(YarnKernelParameters {
                    factor,
                    attention_factor,
                    correction_low: low,
                    correction_high: high,
                })
            },
        }
    }

    fn kernel_parameters(self, rope_dim: u32, rope_theta: f32) -> RopeKernelParameters {
        let yarn = self.yarn_kernel_parameters(rope_dim, rope_theta);
        let rope_half = rope_dim / 2;
        let inverse_frequencies = (0..rope_half)
            .map(|dimension| {
                let mut inverse_frequency = rope_theta.powf(-(dimension as f32 / rope_half as f32));
                if let Some(parameters) = &yarn {
                    let ramp = ((dimension as f32 - parameters.correction_low)
                        / (parameters.correction_high - parameters.correction_low))
                        .clamp(0.0, 1.0);
                    inverse_frequency *= 1.0 + ramp * (1.0 / parameters.factor - 1.0);
                }
                assert!(
                    inverse_frequency.is_finite() && inverse_frequency > 0.0,
                    "RMSNorm/RoPE inverse frequency must be finite and positive"
                );
                inverse_frequency
            })
            .collect();
        RopeKernelParameters {
            attention_factor: yarn.map_or(1.0, |parameters| parameters.attention_factor),
            inverse_frequencies,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub num_total_tokens: u32,
}

impl Shape {
    pub fn validate(self, config: Config) {
        assert!(self.num_total_tokens > 0);
        assert_u32_count_domain(config.num_token_heads(self), "RMSNorm/RoPE token-head rows");
        assert_u32_index_domain(config.num_slots(self), "RMSNorm/RoPE elements");
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub input: &'a Buffer,
    pub norm_weight: &'a Buffer,
    pub flat_token_indices: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct Compute {
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        let constants = KernelConstants::new(config);
        let source = rms_norm_rope_source(constants);
        let function_name = match (constants.config.dtype, constants.config.norm_weight_dtype) {
            (Dtype::Float32, Dtype::Bfloat16) => "rms_norm_rope_f32",
            (Dtype::Float32, Dtype::Float32) => "rms_norm_rope_f32_weight_f32",
            (Dtype::Bfloat16, Dtype::Bfloat16) => "rms_norm_rope_bf16",
            (Dtype::Bfloat16, Dtype::Float32) => "rms_norm_rope_bf16_weight_f32",
            dtypes => panic!("unsupported RMSNorm/RoPE IO/weight dtypes {dtypes:?}"),
        };
        Self {
            constants,
            kernel: CompiledKernel::new(device, &source, function_name),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, buffers: Buffers<'a>, num_active_tokens: ReplayU32) -> Invocation<'a> {
        let invocation = Invocation {
            constants: self.constants,
            kernel: &self.kernel,
            shape,
            buffers,
            num_active_tokens,
        };
        invocation.validate();
        invocation
    }
}

fn rms_norm_rope_source(constants: KernelConstants) -> String {
    let config = constants.config;
    let parameters = config
        .rope_scaling
        .kernel_parameters(config.rope_dim, config.rope_theta);
    let inverse_frequencies = parameters
        .inverse_frequencies
        .iter()
        .map(|value| format!("{value:.9e}f"))
        .collect::<Vec<_>>()
        .join(", ");
    let constants = format!(
        "using namespace metal;\n\nconstant uint num_heads = {}u;\nconstant uint head_dim = {}u;\nconstant uint \
         rope_dim = {}u;\nconstant float eps = {:.9e}f;\nconstant float rope_attention_factor = {:.9e}f;\nconstant \
         float rope_inverse_frequencies[{}] = {{ {} }};",
        config.num_heads,
        config.head_dim,
        config.rope_dim,
        config.eps,
        parameters.attention_factor,
        parameters.inverse_frequencies.len(),
        inverse_frequencies,
    );
    RMS_NORM_ROPE_SOURCE.replacen("using namespace metal;", &constants, 1)
}

pub struct Invocation<'a> {
    constants: KernelConstants,
    kernel: &'a CompiledKernel,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_tokens: ReplayU32,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let constants = self.constants;
        let shape = self.shape;
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.input, 0);
        recorder.set_buffer_read(1, self.buffers.norm_weight, 0);
        recorder.set_buffer_read(2, self.buffers.flat_token_indices, 0);
        recorder.set_buffer_write(3, self.buffers.output, 0);
        match self.num_active_tokens {
            ReplayU32::Fixed(value) => recorder.set_u32(4, value),
            ReplayU32::Parameter(key) => recorder.bind_u32(4, key, 1, shape.num_total_tokens),
        }
        recorder.dispatch_1d(
            constants.num_threads(shape),
            constants.thread_block.required_threads as usize,
        );
    }
}

impl Invocation<'_> {
    fn validate(&self) {
        let config = self.constants.config;
        self.shape.validate(config);
        self.num_active_tokens.validate_range(1, self.shape.num_total_tokens);
        assert!(self.buffers.input.len_bytes() >= config.bytes(self.shape));
        assert!(self.buffers.norm_weight.len_bytes() >= config.norm_weight_bytes());
        assert!(self.buffers.flat_token_indices.len_bytes() >= config.flat_token_indices_bytes(self.shape));
        assert!(self.buffers.output.len_bytes() >= config.bytes(self.shape));
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::Buffers;
    use super::Compute;
    use super::Config;
    use super::RopeScaling;
    use super::Shape;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Dtype;
    use crate::metal::ReplayU32;
    use crate::metal::Stream;

    #[test]
    fn test_yarn_correction_range() {
        let scaling = RopeScaling::Yarn {
            factor: 32.0,
            attention_factor: 1.0 + 0.1 * 32.0_f32.ln(),
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position_embeddings: 8192,
            truncate: true,
        };

        let parameters = scaling.yarn_kernel_parameters(128, 10_000_000.0).unwrap();

        assert_eq!(parameters.factor, 32.0);
        assert_eq!(parameters.attention_factor, 1.0 + 0.1 * 32.0_f32.ln());
        assert_eq!(parameters.correction_low, 14.0);
        assert_eq!(parameters.correction_high, 29.0);
    }

    #[test]
    fn test_f32_yarn_matches_cpu_reference_at_long_context() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let scaling = RopeScaling::Yarn {
            factor: 32.0,
            attention_factor: 1.0 + 0.1 * 32.0_f32.ln(),
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position_embeddings: 8192,
            truncate: true,
        };
        let config = Config::f32(1, 128, 128, 1e-6, 10_000_000.0).with_rope_scaling(scaling);
        let shape = Shape { num_total_tokens: 1 };
        let input = std::array::from_fn::<_, 128, _>(|index| ((index * 37 % 29) as f32 - 14.0) / 8.0);
        let norm_weight =
            std::array::from_fn::<_, 128, _>(|index| bf16::from_f32(0.75 + (index % 11) as f32 / 20.0).to_bits());
        let position = 262_143_u32;
        let input_buffer = Buffer::from_slice(&device, &input);
        let norm_weight_buffer = Buffer::from_slice(&device, &norm_weight);
        let flat_token_indices = Buffer::from_slice(&device, &[position]);
        let output = Buffer::new_zeroed_elements(&device, input.len(), Dtype::Float32);
        let kernel = Compute::new(&device, config);
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            shape,
            Buffers {
                input: &input_buffer,
                norm_weight: &norm_weight_buffer,
                flat_token_indices: &flat_token_indices,
                output: &output,
            },
            ReplayU32::Fixed(shape.num_total_tokens),
        ));
        stream.submit_replay(&builder.build()).wait();

        let inv_rms = (input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32 + 1e-6)
            .sqrt()
            .recip();
        let parameters = scaling.yarn_kernel_parameters(128, 10_000_000.0).unwrap();
        let mut expected = [0.0_f32; 128];
        for dimension in 0..64 {
            let ramp = ((dimension as f32 - parameters.correction_low)
                / (parameters.correction_high - parameters.correction_low))
                .clamp(0.0, 1.0);
            let yarn_scale = 1.0 + ramp * (1.0 / 32.0 - 1.0);
            let inv_freq = 10_000_000.0_f32.powf(-(dimension as f32 / 64.0)) * yarn_scale;
            let theta = position as f32 * inv_freq;
            let (sin, cos) = theta.sin_cos();
            let first = bf16::from_bits(norm_weight[dimension]).to_f32() * input[dimension] * inv_rms;
            let second = bf16::from_bits(norm_weight[dimension + 64]).to_f32() * input[dimension + 64] * inv_rms;
            expected[dimension] = parameters.attention_factor * (first * cos - second * sin);
            expected[dimension + 64] = parameters.attention_factor * (first * sin + second * cos);
        }
        let actual = output.read_typed::<f32>(0, input.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-3,
                "index={index} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn test_bf16_preserves_norm_rounding_order() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config::bf16(1, 4, 2, 1e-6, 1_000_000.0);
        let shape = Shape { num_total_tokens: 1 };
        let input = [0.73046875, -1.171875, 0.439_453_13, 2.03125].map(|value| bf16::from_f32(value).to_bits());
        let norm_weight = [1.296875, 0.8984375, 1.1015625, 0.703125].map(|value| bf16::from_f32(value).to_bits());
        let input_buffer = Buffer::from_slice(&device, &input);
        let norm_weight_buffer = Buffer::from_slice(&device, &norm_weight);
        let flat_token_indices = Buffer::from_slice(&device, &[0_u32]);
        let output = Buffer::new_zeroed_elements(&device, input.len(), Dtype::Bfloat16);
        let kernel = Compute::new(&device, config);
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            shape,
            Buffers {
                input: &input_buffer,
                norm_weight: &norm_weight_buffer,
                flat_token_indices: &flat_token_indices,
                output: &output,
            },
            ReplayU32::Fixed(shape.num_total_tokens),
        ));
        stream.submit_replay(&builder.build()).wait();

        let square_sum = input
            .iter()
            .map(|bits| {
                let value = bf16::from_bits(*bits).to_f32();
                value * value
            })
            .sum::<f32>();
        let inv_rms = (square_sum / input.len() as f32 + 1e-6).sqrt().recip();
        let expected = input
            .iter()
            .zip(norm_weight)
            .map(|(input_bits, weight_bits)| {
                let normalized = bf16::from_f32(bf16::from_bits(*input_bits).to_f32() * inv_rms);
                bf16::from_f32(bf16::from_bits(weight_bits).to_f32() * normalized.to_f32()).to_bits()
            })
            .collect::<Vec<_>>();

        assert_eq!(output.read_typed::<u16>(0, input.len()), expected);
    }

    #[test]
    fn test_bf16_io_with_f32_norm_weight_matches_cpu_reference() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config::bf16(1, 4, 2, 1e-6, 1_000_000.0).with_norm_weight_dtype(Dtype::Float32);
        let shape = Shape { num_total_tokens: 1 };
        let input = [0.75, -1.25, 0.5, 2.0].map(|value| bf16::from_f32(value).to_bits());
        let norm_weight = [1.3_f32, 0.9, 1.1, 0.7];
        let input_buffer = Buffer::from_slice(&device, &input);
        let norm_weight_buffer = Buffer::from_slice(&device, &norm_weight);
        let flat_token_indices = Buffer::from_slice(&device, &[0_u32]);
        let output = Buffer::new_zeroed_elements(&device, input.len(), Dtype::Bfloat16);
        let kernel = Compute::new(&device, config);
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke(
            shape,
            Buffers {
                input: &input_buffer,
                norm_weight: &norm_weight_buffer,
                flat_token_indices: &flat_token_indices,
                output: &output,
            },
            ReplayU32::Fixed(shape.num_total_tokens),
        ));
        stream.submit_replay(&builder.build()).wait();

        let input_f32 = input.map(|bits| bf16::from_bits(bits).to_f32());
        let inv_rms = (input_f32.iter().map(|value| value * value).sum::<f32>() / input.len() as f32 + 1e-6)
            .sqrt()
            .recip();
        let expected = input_f32
            .iter()
            .zip(norm_weight)
            .map(|(input, weight)| bf16::from_f32(input * inv_rms * weight).to_bits())
            .collect::<Vec<_>>();

        assert_eq!(output.read_typed::<u16>(0, input.len()), expected);
    }

    #[test]
    #[should_panic(expected = "RMSNorm/RoPE elements exceeds the shader u32 element-index domain")]
    fn test_shape_rejects_shader_index_overflow() {
        Shape {
            num_total_tokens: 1 << 30,
        }
        .validate(Config::bf16(2, 4, 4, 1e-6, 1_000_000.0));
    }
}
