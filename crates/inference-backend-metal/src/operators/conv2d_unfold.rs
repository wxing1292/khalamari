//! BF16 im2col transform for a 3x3 stride-2 Conv2D with unit padding.

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;

const SOURCE: &str = include_str!("metal/conv2d_unfold.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub batch: u32,
    pub input_height: u32,
    pub input_width: u32,
    pub channels: u32,
}

impl Shape {
    pub const fn output_height(self) -> u32 {
        self.input_height.div_ceil(2)
    }

    pub const fn output_width(self) -> u32 {
        self.input_width.div_ceil(2)
    }

    pub fn patch_values(self) -> u32 {
        (self.channels * 9 + 15) & !15
    }

    pub fn output_rows(self) -> u32 {
        self.batch * self.output_height() * self.output_width()
    }

    pub fn output_values(self) -> u32 {
        self.output_rows() * self.patch_values()
    }

    fn validate(self) {
        assert!(self.batch > 0, "Conv2D unfold batch size must be positive");
        assert!(self.input_height > 0, "Conv2D unfold input height must be positive");
        assert!(self.input_width > 0, "Conv2D unfold input width must be positive");
        assert!(self.channels > 0, "Conv2D unfold channel count must be positive");
        let patch_values = self
            .channels
            .checked_mul(9)
            .and_then(|values| values.checked_add(15))
            .expect("Conv2D unfold aligned patch length must fit the shader u32 domain")
            & !15;
        self.batch
            .checked_mul(self.output_height())
            .and_then(|rows| rows.checked_mul(self.output_width()))
            .and_then(|rows| rows.checked_mul(patch_values))
            .expect("Conv2D unfold output value count must fit the shader u32 count domain");
        // Each patch has at least 16 values, so this count bound also keeps padded coordinates within i32.
    }

    fn input_bytes(self) -> usize {
        self.batch as usize
            * self.input_height as usize
            * self.input_width as usize
            * self.channels as usize
            * Dtype::Bfloat16.item_size()
    }
}

pub struct Kernel {
    kernel: CompiledKernel,
}

impl Kernel {
    pub fn new(device: &Device) -> Self {
        Self {
            kernel: CompiledKernel::new(device, SOURCE, "conv2d_unfold_3x3_stride2_bf16"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, input: &'a Buffer, output: &'a Buffer) -> Invocation<'a> {
        shape.validate();
        assert!(
            input.len_bytes() >= shape.input_bytes(),
            "Conv2D unfold input buffer is too small"
        );
        assert!(
            output.len_bytes() >= shape.output_values() as usize * Dtype::Bfloat16.item_size(),
            "Conv2D unfold output buffer is too small"
        );
        assert_ne!(
            input.as_raw_ptr(),
            output.as_raw_ptr(),
            "Conv2D unfold output must not alias input"
        );
        Invocation {
            kernel: &self.kernel,
            shape,
            input,
            output,
        }
    }
}

pub struct Invocation<'a> {
    kernel: &'a CompiledKernel,
    shape: Shape,
    input: &'a Buffer,
    output: &'a Buffer,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let output_values = self.shape.output_values();
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.input, 0);
        recorder.set_buffer_write(1, self.output, 0);
        recorder.set_u32(2, self.shape.batch);
        recorder.set_u32(3, self.shape.input_height);
        recorder.set_u32(4, self.shape.input_width);
        recorder.set_u32(5, self.shape.channels);
        recorder.set_u32(6, self.shape.output_height());
        recorder.set_u32(7, self.shape.output_width());
        recorder.dispatch_1d(output_values as usize, 256);
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;
    use crate::metal::ReplayArguments;
    use crate::metal::Stream;

    #[test]
    fn test_invoke_nhwc() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = Shape {
            batch: 2,
            input_height: 3,
            input_width: 4,
            channels: 2,
        };
        let input_values = (0..shape.batch * shape.input_height * shape.input_width * shape.channels)
            .map(|index| bf16::from_f32(index as f32 + 1.0))
            .collect::<Vec<_>>();
        let input = Buffer::from_slice(
            &device,
            &input_values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        );
        let output = Buffer::new_zeroed_elements(&device, shape.output_values(), Dtype::Bfloat16);
        let kernel = Kernel::new(&device);
        let mut recorder = stream.create_replay_program();
        recorder.record(kernel.invoke(shape, &input, &output));
        stream
            .submit_replay_with_arguments(&recorder.build(), &ReplayArguments::new())
            .wait();

        let actual = output
            .read_typed::<u16>(0, shape.output_values() as usize)
            .into_iter()
            .map(bf16::from_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual, reference_unfold(shape, &input_values));
    }

    fn reference_unfold(shape: Shape, input: &[bf16]) -> Vec<bf16> {
        let patch_values = shape.patch_values() as usize;
        let mut output = vec![bf16::ZERO; shape.output_values() as usize];
        for batch_index in 0..shape.batch {
            for output_y in 0..shape.output_height() {
                for output_x in 0..shape.output_width() {
                    let output_row =
                        ((batch_index * shape.output_height() + output_y) * shape.output_width() + output_x) as usize;
                    for kernel_y in 0..3_i32 {
                        for kernel_x in 0..3_i32 {
                            let input_y = (output_y * 2) as i32 + kernel_y - 1;
                            let input_x = (output_x * 2) as i32 + kernel_x - 1;
                            if input_y < 0
                                || input_y >= shape.input_height as i32
                                || input_x < 0
                                || input_x >= shape.input_width as i32
                            {
                                continue;
                            }
                            for channel in 0..shape.channels {
                                let patch_index =
                                    ((kernel_y * 3 + kernel_x) as u32 * shape.channels + channel) as usize;
                                let input_index = ((((batch_index * shape.input_height + input_y as u32)
                                    * shape.input_width
                                    + input_x as u32)
                                    * shape.channels)
                                    + channel) as usize;
                                output[output_row * patch_values + patch_index] = input[input_index];
                            }
                        }
                    }
                }
            }
        }
        output
    }
}
