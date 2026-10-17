//! Bidirectional attention over contiguous token blocks.

use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;

const SOURCE: &str = include_str!("metal/tower_block_attention.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub num_heads: u32,
    pub head_dim: u32,
}

impl Config {
    fn validate(self) {
        assert!(self.num_heads > 0, "tower block attention head count must be positive");
        assert!(
            self.head_dim > 0 && self.head_dim <= 128,
            "tower block attention supports head_dim <= 128"
        );
    }

    fn scale(self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    fn tensor_bytes(self, shape: Shape) -> usize {
        shape.num_rows as usize * self.num_heads as usize * self.head_dim as usize * Dtype::Bfloat16.item_size()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_rows: u32,
    pub block_size: u32,
}

impl Shape {
    fn validate(self, config: Config) {
        assert!(self.num_rows > 0, "tower block attention row count must be positive");
        assert!(self.block_size > 0, "tower block attention block size must be positive");
        let last_window_start = ((self.num_rows - 1) / self.block_size) as u64 * self.block_size as u64;
        assert!(
            last_window_start + self.block_size as u64 <= u32::MAX as u64,
            "tower block attention window end must fit the shader u32 domain"
        );
        checked_product(
            "tower block attention tensor byte length",
            &[
                self.num_rows as usize,
                config.num_heads as usize,
                config.head_dim as usize,
                Dtype::Bfloat16.item_size(),
            ],
        );
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub query: &'a Buffer,
    pub key: &'a Buffer,
    pub value: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct Compute {
    config: Config,
    kernel: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            kernel: CompiledKernel::new(device, SOURCE, "tower_block_attention_bf16"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, buffers: Buffers<'a>) -> Invocation<'a> {
        shape.validate(self.config);
        let tensor_bytes = self.config.tensor_bytes(shape);
        for (name, buffer) in [
            ("query", buffers.query),
            ("key", buffers.key),
            ("value", buffers.value),
            ("output", buffers.output),
        ] {
            assert!(
                buffer.len_bytes() >= tensor_bytes,
                "tower block attention {name} buffer is too small"
            );
        }
        Invocation {
            compute: self,
            shape,
            buffers,
        }
    }
}

pub struct Invocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    buffers: Buffers<'a>,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.compute.config;
        recorder.set_kernel(&self.compute.kernel);
        recorder.set_buffer_read(0, self.buffers.query, 0);
        recorder.set_buffer_read(1, self.buffers.key, 0);
        recorder.set_buffer_read(2, self.buffers.value, 0);
        recorder.set_buffer_write(3, self.buffers.output, 0);
        recorder.set_u32(4, self.shape.num_rows);
        recorder.set_u32(5, config.num_heads);
        recorder.set_u32(6, config.head_dim);
        recorder.set_u32(7, self.shape.block_size);
        recorder.set_f32(8, config.scale());
        recorder.dispatch_threadblocks(
            (config.num_heads as usize, self.shape.num_rows as usize, 1),
            (config.head_dim.next_multiple_of(32) as usize, 1, 1),
        );
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use inference_executor_core::reference::softmax_reference;

    use super::*;
    use crate::metal::ReplayArguments;
    use crate::metal::Stream;

    #[test]
    fn test_invoke_multiple_blocks() {
        assert_invoke(
            Config {
                num_heads: 2,
                head_dim: 64,
            },
            Shape {
                num_rows: 5,
                block_size: 3,
            },
        );
    }

    #[test]
    fn test_invoke_head_dim_72() {
        assert_invoke(
            Config {
                num_heads: 1,
                head_dim: 72,
            },
            Shape {
                num_rows: 3,
                block_size: 3,
            },
        );
    }

    fn assert_invoke(config: Config, shape: Shape) {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let num_values = shape.num_rows * config.num_heads * config.head_dim;
        let query_values = (0..num_values)
            .map(|index| bf16::from_f32((index % 17) as f32 * 0.015625 - 0.125))
            .collect::<Vec<_>>();
        let key_values = (0..num_values)
            .map(|index| bf16::from_f32((index % 19) as f32 * -0.01171875 + 0.09375))
            .collect::<Vec<_>>();
        let value_values = (0..num_values)
            .map(|index| bf16::from_f32((index % 23) as f32 * 0.0078125 - 0.0625))
            .collect::<Vec<_>>();
        let bf16_buffer = |values: &[bf16]| {
            Buffer::from_slice(&device, &values.iter().map(|value| value.to_bits()).collect::<Vec<_>>())
        };
        let query = bf16_buffer(&query_values);
        let key = bf16_buffer(&key_values);
        let value = bf16_buffer(&value_values);
        let output = Buffer::new_zeroed_elements(&device, num_values, Dtype::Bfloat16);
        let compute = Compute::new(&device, config);
        let mut recorder = stream.create_replay_program();
        recorder.record(compute.invoke(
            shape,
            Buffers {
                query: &query,
                key: &key,
                value: &value,
                output: &output,
            },
        ));
        stream
            .submit_replay_with_arguments(&recorder.build(), &ReplayArguments::new())
            .wait();

        let actual = output
            .read_typed::<u16>(0, num_values as usize)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        let expected = reference_attention(shape, config, &query_values, &key_values, &value_values);
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 0.01,
                "index={index} actual={actual} expected={expected}"
            );
        }
    }

    fn reference_attention(shape: Shape, config: Config, query: &[bf16], key: &[bf16], value: &[bf16]) -> Vec<f32> {
        let mut output = vec![0.0; query.len()];
        let head_dim = config.head_dim as usize;
        let num_heads = config.num_heads as usize;
        for row in 0..shape.num_rows as usize {
            let window_start = row / shape.block_size as usize * shape.block_size as usize;
            let window_end = (window_start + shape.block_size as usize).min(shape.num_rows as usize);
            for head in 0..num_heads {
                let query_start = (row * num_heads + head) * head_dim;
                let scores = (window_start..window_end)
                    .map(|key_row| {
                        let key_start = (key_row * num_heads + head) * head_dim;
                        (0..head_dim)
                            .map(|dim| query[query_start + dim].to_f32() * key[key_start + dim].to_f32())
                            .sum::<f32>()
                            * config.scale()
                    })
                    .collect::<Vec<_>>();
                let probabilities = softmax_reference(&scores);
                for dim in 0..head_dim {
                    output[query_start + dim] = (window_start..window_end)
                        .zip(&probabilities)
                        .map(|(value_row, probability)| {
                            let value_index = (value_row * num_heads + head) * head_dim + dim;
                            probability * value[value_index].to_f32()
                        })
                        .sum();
                }
            }
        }
        output
    }
}
