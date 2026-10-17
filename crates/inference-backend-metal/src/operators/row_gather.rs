//! Indexed row gather for dense Metal buffers.

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("metal/row_gather.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants {
    dtype: Dtype,
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn new(config: Config) -> Self {
        Self {
            dtype: config.dtype,
            thread_block: ThreadBlockConstants { required_threads: 256 },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub num_cols: u32,
    pub dtype: Dtype,
}

impl Config {
    fn validate(self) {
        assert!(self.num_cols > 0);
        assert!(matches!(self.dtype, Dtype::Bfloat16 | Dtype::Float32));
    }

    fn row_bytes(self) -> usize {
        (self.num_cols as usize)
            .checked_mul(self.dtype.item_size())
            .expect("row gather byte length per row must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_rows: u32,
}

impl Shape {
    fn validate(self, config: Config) {
        assert!(self.num_total_rows > 0);
        self.num_values(config);
    }

    fn row_indices_bytes(self) -> usize {
        (self.num_total_rows as usize)
            .checked_mul(size_of::<u32>())
            .expect("row gather index byte length must fit usize")
    }

    fn output_bytes(self, config: Config) -> usize {
        (self.num_values(config) as usize)
            .checked_mul(config.dtype.item_size())
            .expect("row gather output byte length must fit usize")
    }

    fn num_values(self, config: Config) -> u32 {
        self.num_total_rows
            .checked_mul(config.num_cols)
            .expect("row gather value count must fit the shader u32 index domain")
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub input: &'a Buffer,
    /// Each active index must select a complete row from `input`.
    pub row_indices: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct Kernel {
    config: Config,
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Kernel {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        let constants = KernelConstants::new(config);
        let function_name = match constants.dtype {
            Dtype::Bfloat16 => "row_gather_bf16",
            Dtype::Float32 => "row_gather_f32",
            _ => unreachable!("validated row gather dtype"),
        };
        Self {
            config,
            constants,
            kernel: CompiledKernel::new(device, SOURCE, function_name),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, num_active_rows: ReplayU32, buffers: Buffers<'a>) -> Invocation<'a> {
        let invocation = Invocation {
            config: self.config,
            constants: self.constants,
            kernel: &self.kernel,
            shape,
            buffers,
            num_active_rows,
        };
        invocation.validate();
        invocation
    }
}

pub struct Invocation<'a> {
    config: Config,
    constants: KernelConstants,
    kernel: &'a CompiledKernel,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_rows: ReplayU32,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.input, 0);
        recorder.set_buffer_read(1, self.buffers.row_indices, 0);
        recorder.set_buffer_write(2, self.buffers.output, 0);
        recorder.set_u32(3, self.config.num_cols);
        match self.num_active_rows {
            ReplayU32::Fixed(value) => recorder.set_u32(4, value),
            ReplayU32::Parameter(key) => recorder.bind_u32(4, key, 1, self.shape.num_total_rows),
        }
        recorder.dispatch_1d(
            self.shape.num_values(self.config) as usize,
            self.constants.thread_block.required_threads as usize,
        );
    }
}

impl Invocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        if let ReplayU32::Fixed(value) = self.num_active_rows {
            assert_eq!(value, self.shape.num_total_rows);
        }
        assert!(self.buffers.input.len_bytes() >= self.config.row_bytes());
        assert!(self.buffers.row_indices.len_bytes() >= self.shape.row_indices_bytes());
        assert!(self.buffers.output.len_bytes() >= self.shape.output_bytes(self.config));
        assert_ne!(
            self.buffers.output.as_raw_ptr(),
            self.buffers.input.as_raw_ptr(),
            "row gather output must not alias input"
        );
        assert_ne!(
            self.buffers.output.as_raw_ptr(),
            self.buffers.row_indices.as_raw_ptr(),
            "row gather output must not alias row indices"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::Stream;

    const NUM_ACTIVE_ROWS: ReplayParameterKey = ReplayParameterKey::new("test.row_gather.num_active_rows");

    #[test]
    fn test_replay_matches_reference_across_active_counts_and_dtypes() {
        for dtype in [Dtype::Bfloat16, Dtype::Float32] {
            let device = Device::system_default();
            let stream = Stream::new(&device);
            let config = Config { num_cols: 7, dtype };
            let shape = Shape { num_total_rows: 8 };
            let row_bytes = config.row_bytes();
            let input_values = (0..16 * row_bytes)
                .map(|index| ((index * 29 + 7) % 251) as u8)
                .collect::<Vec<_>>();
            let row_indices_values = [15_u32, 0, 7, 3, 12, 1, 9, 5];
            let input = Buffer::from_slice(&device, &input_values);
            let row_indices = Buffer::from_slice(&device, &row_indices_values);
            let output = Buffer::new_zeroed(&device, shape.output_bytes(config));
            let kernel = Kernel::new(&device, config);
            let replay = {
                let mut builder = stream.create_replay_program();
                builder.record(kernel.invoke(
                    shape,
                    ReplayU32::Parameter(NUM_ACTIVE_ROWS),
                    Buffers {
                        input: &input,
                        row_indices: &row_indices,
                        output: &output,
                    },
                ));
                builder.build()
            };

            for num_active_rows in [1_usize, 8, 3, 7, 2, 6, 4, 5] {
                let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, num_active_rows as u32);
                stream.submit_replay_with_arguments(&replay, &arguments).wait();
                let expected = gathered_bytes(&input_values, row_bytes, &row_indices_values[..num_active_rows]);
                assert_eq!(output.read_typed::<u8>(0, num_active_rows * row_bytes), expected);
            }
        }
    }

    fn gathered_bytes(input: &[u8], row_bytes: usize, row_indices: &[u32]) -> Vec<u8> {
        row_indices
            .iter()
            .flat_map(|&row_index| {
                let row_index = row_index as usize;
                input[row_index * row_bytes..(row_index + 1) * row_bytes]
                    .iter()
                    .copied()
            })
            .collect()
    }
}
