use std::mem::size_of;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const QUANTIZED_EMBEDDING_SOURCE: &str = include_str!("metal/quantized_embedding.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants {
    scale_bias_dtype: Dtype,
    output_dtype: Dtype,
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn new(config: Config) -> Self {
        let constants = Self {
            scale_bias_dtype: config.scale_bias_dtype,
            output_dtype: config.output_dtype,
            thread_block: ThreadBlockConstants { required_threads: 256 },
        };
        constants.validate();
        constants
    }

    fn validate(self) {
        assert!(matches!(self.scale_bias_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert_eq!(self.output_dtype, Dtype::Bfloat16);
        assert!(self.thread_block.required_threads > 0);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub vocab_size: u32,
    pub hidden_dim: u32,
    pub group_size: u32,
    pub bits: u32,
    pub scale_bias_dtype: Dtype,
    pub output_dtype: Dtype,
}

impl Config {
    pub fn validate(self) {
        assert!(self.vocab_size > 0);
        assert!(self.hidden_dim > 0);
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.hidden_dim % self.group_size, 0);
        assert!(matches!(self.scale_bias_dtype, Dtype::Float32 | Dtype::Bfloat16));
        match self.output_dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 quantized embedding output is not implemented"),
            dtype => panic!("unsupported quantized embedding output dtype {dtype:?}"),
        }
        let _ = self.weight_bytes_unchecked();
        let _ = self.num_affine_params_unchecked();
    }

    fn packed_cols_unchecked(self) -> usize {
        (self.hidden_dim as usize)
            .checked_mul(self.bits as usize)
            .expect("quantized embedding packed width must fit usize")
            / 32
    }

    pub fn weight_bytes(self) -> usize {
        self.validate();
        self.weight_bytes_unchecked()
    }

    fn weight_bytes_unchecked(self) -> usize {
        (self.vocab_size as usize)
            .checked_mul(self.packed_cols_unchecked())
            .and_then(|count| count.checked_mul(size_of::<u32>()))
            .expect("quantized embedding weight byte length must fit usize")
    }

    pub fn num_affine_params(self) -> usize {
        self.validate();
        self.num_affine_params_unchecked()
    }

    fn num_affine_params_unchecked(self) -> usize {
        (self.vocab_size as usize)
            .checked_mul((self.hidden_dim / self.group_size) as usize)
            .expect("quantized embedding affine parameter count must fit usize")
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

    pub fn num_output_values(self, config: Config) -> usize {
        self.validate();
        self.num_total_tokens
            .checked_mul(config.hidden_dim)
            .expect("quantized embedding output index must fit the shader u32 domain") as usize
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub token_ids: &'a Buffer,
    pub weight: &'a Buffer,
    pub scales: &'a Buffer,
    pub biases: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct Compute {
    config: Config,
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        let constants = KernelConstants::new(config);
        let function_name = match constants.scale_bias_dtype {
            Dtype::Float32 => "quantized_embedding_f32_to_bf16",
            Dtype::Bfloat16 => "quantized_embedding_bf16_to_bf16",
            _ => unreachable!("validated quantized embedding scale/bias dtype"),
        };
        Self {
            config,
            constants,
            kernel: CompiledKernel::new(device, QUANTIZED_EMBEDDING_SOURCE, function_name),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, num_active_tokens: ReplayU32, buffers: Buffers<'a>) -> Invocation<'a> {
        if let ReplayU32::Fixed(value) = num_active_tokens {
            assert_eq!(value, shape.num_total_tokens);
        }
        validate_buffers(self.config, shape, &buffers);
        Invocation {
            kernel: self,
            shape,
            buffers,
            num_active_tokens,
        }
    }
}

pub struct Invocation<'a> {
    kernel: &'a Compute,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_tokens: ReplayU32,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.kernel.config;
        recorder.set_kernel(&self.kernel.kernel);
        recorder.set_buffer_read(0, self.buffers.token_ids, 0);
        recorder.set_buffer_read(1, self.buffers.weight, 0);
        recorder.set_buffer_read(2, self.buffers.scales, 0);
        recorder.set_buffer_read(3, self.buffers.biases, 0);
        recorder.set_buffer_write(4, self.buffers.output, 0);
        match self.num_active_tokens {
            ReplayU32::Fixed(value) => recorder.set_u32(5, value),
            ReplayU32::Parameter(key) => recorder.bind_u32(5, key, 1, self.shape.num_total_tokens),
        }
        recorder.set_u32(6, config.vocab_size);
        recorder.set_u32(7, config.hidden_dim);
        recorder.set_u32(8, config.group_size);
        recorder.set_u32(9, config.bits);
        let required_threads = self.kernel.constants.thread_block.required_threads as usize;
        recorder.dispatch_threadblocks(
            (self.shape.num_output_values(config).div_ceil(required_threads), 1, 1),
            (required_threads, 1, 1),
        );
    }
}

fn validate_buffers(config: Config, shape: Shape, buffers: &Buffers<'_>) {
    shape.validate();
    let affine_param_bytes = config
        .num_affine_params_unchecked()
        .checked_mul(config.scale_bias_dtype.item_size())
        .expect("quantized embedding affine parameter bytes must fit usize");
    let output_bytes = shape
        .num_output_values(config)
        .checked_mul(config.output_dtype.item_size())
        .expect("quantized embedding output bytes must fit usize");
    assert!(buffers.token_ids.len_bytes() >= shape.num_total_tokens as usize * size_of::<i32>());
    assert_eq!(buffers.weight.len_bytes(), config.weight_bytes_unchecked());
    assert_eq!(buffers.scales.len_bytes(), affine_param_bytes);
    assert_eq!(buffers.biases.len_bytes(), affine_param_bytes);
    assert!(buffers.output.len_bytes() >= output_bytes);
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::Buffers;
    use super::Compute;
    use super::Config;
    use super::Shape;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Dtype;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::ReplayU32;
    use crate::metal::Stream;

    const VOCAB_SIZE: u32 = 2;
    const HIDDEN_DIM: u32 = 32;
    const GROUP_SIZE: u32 = 32;
    const BITS: u32 = 4;
    const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.quantized_embedding.num_active_tokens");

    #[test]
    fn test_replay_matches_reference_across_active_counts_and_affine_dtypes() {
        for scale_bias_dtype in [Dtype::Float32, Dtype::Bfloat16] {
            let device = Device::system_default();
            let stream = Stream::new(&device);
            let raw_scales = [0.5_f32, 0.25];
            let raw_biases = [-1.0_f32, 2.0];
            let scales = stored_affine_values(raw_scales, scale_bias_dtype);
            let biases = stored_affine_values(raw_biases, scale_bias_dtype);
            let config = Config {
                vocab_size: VOCAB_SIZE,
                hidden_dim: HIDDEN_DIM,
                group_size: GROUP_SIZE,
                bits: BITS,
                scale_bias_dtype,
                output_dtype: Dtype::Bfloat16,
            };
            let shape = Shape { num_total_tokens: 8 };
            let token_ids_values = [0_i32, 1, -1, 2, 1, 0, 1, 0];
            let token_ids = Buffer::from_slice(&device, &token_ids_values);
            let weight = Buffer::from_slice(&device, &packed_q4_rows());
            let scales_buffer = affine_buffer(&device, &scales, scale_bias_dtype);
            let biases_buffer = affine_buffer(&device, &biases, scale_bias_dtype);
            let output = Buffer::new_zeroed_elements(&device, shape.num_output_values(config), Dtype::Bfloat16);
            let kernel = Compute::new(&device, config);
            let replay = {
                let mut recorder = stream.create_replay_program();
                recorder.record(kernel.invoke(
                    shape,
                    ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
                    Buffers {
                        token_ids: &token_ids,
                        weight: &weight,
                        scales: &scales_buffer,
                        biases: &biases_buffer,
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
                let expected = token_ids_reference(&token_ids_values[..num_active_tokens], &scales, &biases);
                assert_eq!(
                    output.read_typed::<u16>(0, num_active_tokens * HIDDEN_DIM as usize,),
                    expected
                );
            }
        }
    }

    fn packed_q4_rows() -> Vec<u8> {
        (0..VOCAB_SIZE)
            .flat_map(|row| {
                (0..HIDDEN_DIM / 2).map(move |column| {
                    let lo = (row * 3 + column * 2) & 0x0f;
                    let hi = (row * 3 + column * 2 + 1) & 0x0f;
                    (lo | (hi << 4)) as u8
                })
            })
            .collect()
    }

    fn token_ids_reference(token_ids: &[i32], scales: &[f32; 2], biases: &[f32; 2]) -> Vec<u16> {
        token_ids
            .iter()
            .flat_map(|&token_id| {
                (0..HIDDEN_DIM).map(move |hidden_index| {
                    if !(0..VOCAB_SIZE as i32).contains(&token_id) {
                        return bf16::ZERO.to_bits();
                    }
                    let quantized = (token_id as u32 * 3 + hidden_index) & 0x0f;
                    bf16::from_f32(quantized as f32 * scales[token_id as usize] + biases[token_id as usize]).to_bits()
                })
            })
            .collect()
    }

    fn stored_affine_values(values: [f32; 2], dtype: Dtype) -> [f32; 2] {
        match dtype {
            Dtype::Float32 => values,
            Dtype::Bfloat16 => values.map(|value| bf16::from_f32(value).to_f32()),
            _ => panic!("unsupported embedding test affine dtype {dtype:?}"),
        }
    }

    fn affine_buffer(device: &Device, values: &[f32], dtype: Dtype) -> Buffer {
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
            _ => panic!("unsupported embedding test affine dtype {dtype:?}"),
        }
    }
}
