//! BF16 LayerNorm with learned scale and bias.

use std::collections::HashSet;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::mlx_headers::find_mlx_metal_header_root;
use crate::mlx_headers::read_mlx_metal_header;

const THREADS_PER_THREADBLOCK: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub hidden_dim: u32,
    pub eps: f32,
}

impl Config {
    fn validate(self) {
        assert!(self.hidden_dim > 0, "LayerNorm hidden dimension must be positive");
        // MLX reads four values per thread and advances its uint loop by this tile width.
        let loop_width = THREADS_PER_THREADBLOCK as u64 * 4;
        let padded_hidden_dim = (self.hidden_dim as u64).div_ceil(loop_width) * loop_width;
        assert!(
            padded_hidden_dim <= u32::MAX as u64,
            "LayerNorm padded hidden dimension must fit the shader u32 loop domain"
        );
        assert!(
            self.eps.is_finite() && self.eps > 0.0,
            "LayerNorm epsilon must be finite and positive"
        );
    }

    fn row_bytes(self) -> usize {
        self.hidden_dim as usize * Dtype::Bfloat16.item_size()
    }

    fn tensor_bytes(self, shape: Shape) -> usize {
        shape.num_rows as usize * self.row_bytes()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_rows: u32,
}

impl Shape {
    fn validate(self, config: Config) {
        assert!(self.num_rows > 0, "LayerNorm row count must be positive");
        (self.num_rows as usize)
            .checked_mul(config.row_bytes())
            .expect("LayerNorm tensor byte length must fit usize");
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub input: &'a Buffer,
    pub weight: &'a Buffer,
    pub bias: &'a Buffer,
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
            kernel: CompiledKernel::new(device, source(), "layer_norm_loopedbfloat16"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, buffers: Buffers<'a>) -> Invocation<'a> {
        shape.validate(self.config);
        let tensor_bytes = self.config.tensor_bytes(shape);
        let row_bytes = self.config.row_bytes();
        assert!(
            buffers.input.len_bytes() >= tensor_bytes,
            "LayerNorm input buffer is too small"
        );
        assert!(
            buffers.output.len_bytes() >= tensor_bytes,
            "LayerNorm output buffer is too small"
        );
        assert!(
            buffers.weight.len_bytes() >= row_bytes,
            "LayerNorm weight buffer is too small"
        );
        assert!(
            buffers.bias.len_bytes() >= row_bytes,
            "LayerNorm bias buffer is too small"
        );
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
        recorder.set_buffer_read(0, self.buffers.input, 0);
        recorder.set_buffer_read(1, self.buffers.weight, 0);
        recorder.set_buffer_read(2, self.buffers.bias, 0);
        recorder.set_buffer_write(3, self.buffers.output, 0);
        recorder.set_f32(4, config.eps);
        recorder.set_u32(5, config.hidden_dim);
        recorder.set_u32(6, 1);
        recorder.set_u32(7, 1);
        recorder.dispatch_threadblocks((self.shape.num_rows as usize, 1, 1), (THREADS_PER_THREADBLOCK, 1, 1));
    }
}

fn source() -> &'static str {
    static EXPANDED_SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    EXPANDED_SOURCE.get_or_init(|| {
        let root = find_mlx_metal_header_root("layer_norm.metal", |_| true, "LayerNorm");
        let mut included = HashSet::new();
        let mut source = read_mlx_metal_header(&root, "mlx/backend/metal/kernels/layer_norm.metal", &mut included);
        let declaration = "constant bool has_w [[function_constant(20)]];";
        let declaration_start = source
            .find(declaration)
            .unwrap_or_else(|| panic!("LayerNorm MLX source is missing {declaration:?}"));
        source.replace_range(
            declaration_start..declaration_start + declaration.len(),
            "constant bool has_w = true;",
        );
        source
    })
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;
    use crate::metal::ReplayArguments;
    use crate::metal::Stream;

    #[test]
    fn test_invoke_fixed() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config {
            hidden_dim: 7,
            eps: 1.0e-5,
        };
        let shape = Shape { num_rows: 3 };
        let input_values = (0..shape.num_rows * config.hidden_dim)
            .map(|index| bf16::from_f32(index as f32 * 0.125 - 1.0))
            .collect::<Vec<_>>();
        let weight_values = (0..config.hidden_dim)
            .map(|index| bf16::from_f32(0.75 + index as f32 * 0.03125))
            .collect::<Vec<_>>();
        let bias_values = (0..config.hidden_dim)
            .map(|index| bf16::from_f32(index as f32 * -0.015625))
            .collect::<Vec<_>>();
        let bf16_buffer = |values: &[bf16]| {
            Buffer::from_slice(&device, &values.iter().map(|value| value.to_bits()).collect::<Vec<_>>())
        };
        let input = bf16_buffer(&input_values);
        let weight = bf16_buffer(&weight_values);
        let bias = bf16_buffer(&bias_values);
        let output = Buffer::new_zeroed_elements(&device, input_values.len(), Dtype::Bfloat16);
        let compute = Compute::new(&device, config);
        let mut recorder = stream.create_replay_program();
        recorder.record(compute.invoke(
            shape,
            Buffers {
                input: &input,
                weight: &weight,
                bias: &bias,
                output: &output,
            },
        ));
        stream
            .submit_replay_with_arguments(&recorder.build(), &ReplayArguments::new())
            .wait();

        let actual = output
            .read_typed::<u16>(0, input_values.len())
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        let expected = reference_layer_norm(
            &input_values,
            &weight_values,
            &bias_values,
            shape.num_rows as usize,
            config,
        );
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 0.02,
                "index={index} actual={actual} expected={expected}"
            );
        }
    }

    fn reference_layer_norm(
        input: &[bf16],
        weight: &[bf16],
        bias: &[bf16],
        num_rows: usize,
        config: Config,
    ) -> Vec<f32> {
        let hidden_dim = config.hidden_dim as usize;
        let mut output = vec![0.0; input.len()];
        for row in 0..num_rows {
            let values = &input[row * hidden_dim..(row + 1) * hidden_dim];
            let mean = values.iter().map(|value| value.to_f32()).sum::<f32>() / hidden_dim as f32;
            let variance = values.iter().map(|value| (value.to_f32() - mean).powi(2)).sum::<f32>() / hidden_dim as f32;
            let inverse_stddev = (variance + config.eps).sqrt().recip();
            for column in 0..hidden_dim {
                output[row * hidden_dim + column] =
                    (values[column].to_f32() - mean) * inverse_stddev * weight[column].to_f32() + bias[column].to_f32();
            }
        }
        output
    }
}
