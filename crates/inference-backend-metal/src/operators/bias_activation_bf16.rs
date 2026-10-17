//! Row-wise BF16 bias and activation.

use std::collections::HashSet;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::mlx_headers::find_mlx_metal_header_root;
use crate::mlx_headers::read_mlx_metal_header;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float precise_gelu(float value) {
  return 0.5f * value * (1.0f + erf(value * 0.7071067811865475f));
}

kernel void bias_activation_bf16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* bias [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant uint& num_values [[buffer(3)]],
    constant uint& num_columns [[buffer(4)]],
    constant uint& activation [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= num_values) return;
  float value = float(input[index]) + float(bias[index % num_columns]);
  if (activation == 1u) value = precise_gelu(value);
  output[index] = bfloat(value);
}
"#;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Activation {
    Identity,
    Gelu,
}

impl Activation {
    const fn shader_value(self) -> u32 {
        match self {
            Self::Identity => 0,
            Self::Gelu => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_rows: u32,
    pub num_columns: u32,
}

impl Shape {
    fn validate(self) {
        assert!(self.num_rows > 0, "BF16 bias activation row count must be positive");
        assert!(
            self.num_columns > 0,
            "BF16 bias activation column count must be positive"
        );
        assert!(
            self.num_rows <= u32::MAX / self.num_columns,
            "BF16 bias activation value count must fit the shader u32 count domain"
        );
    }

    fn num_values(self) -> u32 {
        self.num_rows * self.num_columns
    }
}

pub struct Kernel {
    kernel: CompiledKernel,
}

impl Kernel {
    pub fn new(device: &Device) -> Self {
        Self {
            kernel: CompiledKernel::new(device, source(), "bias_activation_bf16"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        shape: Shape,
        activation: Activation,
        input: &'a Buffer,
        input_offset_bytes: usize,
        bias: &'a Buffer,
        output: &'a Buffer,
        output_offset_bytes: usize,
    ) -> Invocation<'a> {
        shape.validate();
        let tensor_bytes = shape.num_values() as usize * Dtype::Bfloat16.item_size();
        let bias_bytes = shape.num_columns as usize * Dtype::Bfloat16.item_size();
        assert_range(input, input_offset_bytes, tensor_bytes, "BF16 bias activation input");
        assert_range(output, output_offset_bytes, tensor_bytes, "BF16 bias activation output");
        assert!(
            bias.len_bytes() >= bias_bytes,
            "BF16 bias activation bias buffer is too small"
        );
        Invocation {
            kernel: &self.kernel,
            shape,
            activation,
            input,
            input_offset_bytes,
            bias,
            output,
            output_offset_bytes,
        }
    }
}

pub struct Invocation<'a> {
    kernel: &'a CompiledKernel,
    shape: Shape,
    activation: Activation,
    input: &'a Buffer,
    input_offset_bytes: usize,
    bias: &'a Buffer,
    output: &'a Buffer,
    output_offset_bytes: usize,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let num_values = self.shape.num_values();
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.input, self.input_offset_bytes);
        recorder.set_buffer_read(1, self.bias, 0);
        recorder.set_buffer_write(2, self.output, self.output_offset_bytes);
        recorder.set_u32(3, num_values);
        recorder.set_u32(4, self.shape.num_columns);
        recorder.set_u32(5, self.activation.shader_value());
        recorder.dispatch_1d(num_values as usize, 256);
    }
}

fn assert_range(buffer: &Buffer, offset_bytes: usize, len_bytes: usize, name: &str) {
    assert!(
        offset_bytes.is_multiple_of(Dtype::Bfloat16.item_size()),
        "{name} byte offset must be BF16-aligned"
    );
    let buffer_bytes = buffer.len_bytes();
    assert!(
        offset_bytes <= buffer_bytes && len_bytes <= buffer_bytes - offset_bytes,
        "{name} byte range exceeds its buffer"
    );
}

fn source() -> &'static str {
    static EXPANDED_SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    EXPANDED_SOURCE.get_or_init(|| {
        let root = find_mlx_metal_header_root("erf.h", |_| true, "BF16 GELU");
        let mut included = HashSet::new();
        let mut source = read_mlx_metal_header(&root, "mlx/backend/metal/kernels/erf.h", &mut included);
        source.push_str(SOURCE);
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
    fn test_invoke_identity_and_gelu() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = Shape {
            num_rows: 3,
            num_columns: 7,
        };
        let offset_values = 8_usize;
        let input_values = (0..shape.num_values())
            .map(|index| bf16::from_f32(index as f32 * 0.125 - 1.0))
            .collect::<Vec<_>>();
        let bias_values = (0..shape.num_columns)
            .map(|index| bf16::from_f32(index as f32 * -0.0625 + 0.125))
            .collect::<Vec<_>>();
        let mut padded_input = vec![0; offset_values];
        padded_input.extend(input_values.iter().map(|value| value.to_bits()));
        let input = Buffer::from_slice(&device, &padded_input);
        let bias = Buffer::from_slice(
            &device,
            &bias_values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        );
        let kernel = Kernel::new(&device);
        for activation in [Activation::Identity, Activation::Gelu] {
            let output =
                Buffer::new_zeroed_elements(&device, offset_values + shape.num_values() as usize, Dtype::Bfloat16);
            let mut recorder = stream.create_replay_program();
            recorder.record(kernel.invoke(
                shape,
                activation,
                &input,
                offset_values * size_of::<u16>(),
                &bias,
                &output,
                offset_values * size_of::<u16>(),
            ));
            stream
                .submit_replay_with_arguments(&recorder.build(), &ReplayArguments::new())
                .wait();

            assert_eq!(output.read_typed::<u16>(0, offset_values), vec![0; offset_values]);
            let actual = output
                .read_typed::<u16>(offset_values, shape.num_values() as usize)
                .into_iter()
                .map(|bits| bf16::from_bits(bits).to_f32())
                .collect::<Vec<_>>();
            for (index, actual) in actual.into_iter().enumerate() {
                let value = input_values[index].to_f32() + bias_values[index % shape.num_columns as usize].to_f32();
                let expected = match activation {
                    Activation::Identity => value,
                    Activation::Gelu => reference_gelu(value),
                };
                assert!(
                    (actual - expected).abs() < 0.01,
                    "index={index} actual={actual} expected={expected}"
                );
            }
        }
    }

    fn reference_gelu(value: f32) -> f32 {
        let inner = (2.0 / std::f32::consts::PI).sqrt() * (value + 0.044_715 * value.powi(3));
        0.5 * value * (1.0 + inner.tanh())
    }
}
