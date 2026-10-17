//! Copy indexed dense rows into indexed destination rows.

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("metal/row_scatter.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub num_cols: u32,
    pub dtype: Dtype,
}

impl Config {
    fn validate(self) {
        assert!(self.num_cols > 0, "row scatter requires columns");
        assert!(
            matches!(self.dtype, Dtype::Bfloat16 | Dtype::Float32),
            "row scatter supports BF16 and F32"
        );
    }

    fn row_bytes(self) -> usize {
        (self.num_cols as usize)
            .checked_mul(self.dtype.item_size())
            .expect("row scatter row byte length must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_rows: u32,
}

impl Shape {
    fn validate(self, config: Config) {
        assert!(self.num_total_rows > 0, "row scatter requires rows");
        self.num_values(config);
    }

    fn num_values(self, config: Config) -> u32 {
        self.num_total_rows
            .checked_mul(config.num_cols)
            .expect("row scatter value count must fit the shader u32 index domain")
    }

    fn routes_bytes(self) -> usize {
        (self.num_total_rows as usize)
            .checked_mul(size_of::<[u32; 2]>())
            .expect("row scatter table byte length must fit usize")
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub input: &'a Buffer,
    /// One `[input_row, output_row]` pair per copied row.
    /// Active routes must select complete rows. Output rows must be unique.
    pub routes: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct Kernel {
    config: Config,
    kernel: CompiledKernel,
}

impl Kernel {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        let function_name = match config.dtype {
            Dtype::Bfloat16 => "row_scatter_bf16",
            Dtype::Float32 => "row_scatter_f32",
            _ => unreachable!("validated row scatter dtype"),
        };
        Self {
            config,
            kernel: CompiledKernel::new(device, SOURCE, function_name),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, num_active_rows: ReplayU32, buffers: Buffers<'a>) -> Invocation<'a> {
        let invocation = Invocation {
            config: self.config,
            kernel: &self.kernel,
            shape,
            num_active_rows,
            buffers,
        };
        invocation.validate();
        invocation
    }
}

pub struct Invocation<'a> {
    config: Config,
    kernel: &'a CompiledKernel,
    shape: Shape,
    num_active_rows: ReplayU32,
    buffers: Buffers<'a>,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.input, 0);
        recorder.set_buffer_read(1, self.buffers.routes, 0);
        recorder.set_buffer_write(2, self.buffers.output, 0);
        recorder.set_u32(3, self.config.num_cols);
        match self.num_active_rows {
            ReplayU32::Fixed(value) => recorder.set_u32(4, value),
            ReplayU32::Parameter(key) => recorder.bind_u32(4, key, 1, self.shape.num_total_rows),
        }
        recorder.dispatch_1d(self.shape.num_values(self.config) as usize, 256);
    }
}

impl Invocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.config);
        if let ReplayU32::Fixed(value) = self.num_active_rows {
            assert_eq!(value, self.shape.num_total_rows);
        }
        assert!(self.buffers.input.len_bytes() >= self.config.row_bytes());
        assert!(self.buffers.routes.len_bytes() >= self.shape.routes_bytes());
        assert!(self.buffers.output.len_bytes() >= self.config.row_bytes());
        assert_ne!(
            self.buffers.output.as_raw_ptr(),
            self.buffers.input.as_raw_ptr(),
            "row scatter output must not alias input"
        );
        assert_ne!(
            self.buffers.output.as_raw_ptr(),
            self.buffers.routes.as_raw_ptr(),
            "row scatter output must not alias routes"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::Stream;

    const NUM_ACTIVE_ROWS: ReplayParameterKey = ReplayParameterKey::new("test.row_scatter.num_active_rows");

    #[test]
    fn test_replay_copies_bf16_rows_to_discontiguous_destinations() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config {
            num_cols: 2,
            dtype: Dtype::Bfloat16,
        };
        let shape = Shape { num_total_rows: 5 };
        let input = Buffer::from_slice(&device, &(0_u16..24).collect::<Vec<_>>());
        let routes = [[7, 1], [0, 8], [3, 4], [9, 0], [5, 6]];
        let route_values = routes.iter().flatten().copied().collect::<Vec<_>>();
        let routes_buffer = Buffer::from_slice(&device, &route_values);
        let output = Buffer::new_zeroed_elements(&device, 10 * config.num_cols as usize, config.dtype);
        let kernel = Kernel::new(&device, config);
        let replay = {
            let mut recorder = stream.create_replay_program();
            recorder.record(kernel.invoke(
                shape,
                ReplayU32::Parameter(NUM_ACTIVE_ROWS),
                Buffers {
                    input: &input,
                    routes: &routes_buffer,
                    output: &output,
                },
            ));
            recorder.build()
        };

        stream
            .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, 3))
            .wait();
        assert_eq!(output.read_typed::<u16>(2, 2), vec![14, 15]);
        assert_eq!(output.read_typed::<u16>(16, 2), vec![0, 1]);
        assert_eq!(output.read_typed::<u16>(8, 2), vec![6, 7]);
        assert_eq!(output.read_typed::<u16>(0, 2), vec![0, 0]);
    }

    #[test]
    fn test_copies_f32_rows() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let input = Buffer::from_slice(&device, &[1.0_f32, 2.0, 3.0]);
        let routes = Buffer::from_slice(&device, &[0_u32, 2, 1, 0, 2, 1]);
        let output = Buffer::new_zeroed_elements(&device, 3, Dtype::Float32);
        let kernel = Kernel::new(
            &device,
            Config {
                num_cols: 1,
                dtype: Dtype::Float32,
            },
        );
        let mut recorder = stream.create_replay_program();
        recorder.record(kernel.invoke(
            Shape { num_total_rows: 3 },
            ReplayU32::Fixed(3),
            Buffers {
                input: &input,
                routes: &routes,
                output: &output,
            },
        ));
        stream.submit_replay(&recorder.build()).wait();
        assert_eq!(output.read_typed::<f32>(0, 3), vec![2.0, 3.0, 1.0]);
    }
}
