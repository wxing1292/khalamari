//! Route dense rows from one of three source buffers.

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("metal/row_route.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub num_cols: u32,
    pub dtype: Dtype,
}

impl Config {
    fn validate(self) {
        assert!(self.num_cols > 0, "row route requires columns");
        assert!(
            matches!(self.dtype, Dtype::Bfloat16 | Dtype::Float32),
            "row route supports BF16 and F32"
        );
    }

    fn row_bytes(self) -> usize {
        (self.num_cols as usize)
            .checked_mul(self.dtype.item_size())
            .expect("row route row byte length must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_rows: u32,
}

impl Shape {
    fn validate(self, config: Config) {
        assert!(self.num_total_rows > 0, "row route requires rows");
        self.num_values(config);
    }

    fn num_values(self, config: Config) -> u32 {
        self.num_total_rows
            .checked_mul(config.num_cols)
            .expect("row route value count must fit the shader u32 index domain")
    }

    fn routes_bytes(self) -> usize {
        (self.num_total_rows as usize)
            .checked_mul(size_of::<[u32; 2]>())
            .expect("row route table byte length must fit usize")
    }

    fn output_bytes(self, config: Config) -> usize {
        (self.num_values(config) as usize)
            .checked_mul(config.dtype.item_size())
            .expect("row route output byte length must fit usize")
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub first_input: &'a Buffer,
    pub second_input: &'a Buffer,
    pub third_input: &'a Buffer,
    /// One `[source_index, source_row]` pair per output row. Source indices are `0`, `1`, and `2`.
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
            Dtype::Bfloat16 => "row_route_bf16",
            Dtype::Float32 => "row_route_f32",
            _ => unreachable!("validated row route dtype"),
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
        recorder.set_buffer_read(0, self.buffers.first_input, 0);
        recorder.set_buffer_read(1, self.buffers.second_input, 0);
        recorder.set_buffer_read(2, self.buffers.third_input, 0);
        recorder.set_buffer_read(3, self.buffers.routes, 0);
        recorder.set_buffer_write(4, self.buffers.output, 0);
        recorder.set_u32(5, self.config.num_cols);
        match self.num_active_rows {
            ReplayU32::Fixed(value) => recorder.set_u32(6, value),
            ReplayU32::Parameter(key) => recorder.bind_u32(6, key, 1, self.shape.num_total_rows),
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
        let row_bytes = self.config.row_bytes();
        assert!(self.buffers.first_input.len_bytes() >= row_bytes);
        assert!(self.buffers.second_input.len_bytes() >= row_bytes);
        assert!(self.buffers.third_input.len_bytes() >= row_bytes);
        assert!(self.buffers.routes.len_bytes() >= self.shape.routes_bytes());
        assert!(self.buffers.output.len_bytes() >= self.shape.output_bytes(self.config));
        for (name, input) in [
            ("first input", self.buffers.first_input),
            ("second input", self.buffers.second_input),
            ("third input", self.buffers.third_input),
            ("routes", self.buffers.routes),
        ] {
            assert_ne!(
                self.buffers.output.as_raw_ptr(),
                input.as_raw_ptr(),
                "row route output must not alias {name}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::Stream;

    const NUM_ACTIVE_ROWS: ReplayParameterKey = ReplayParameterKey::new("test.row_route.num_active_rows");

    #[test]
    fn test_replay_routes_three_bf16_sources_across_active_counts() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = Shape { num_total_rows: 8 };
        let config = Config {
            num_cols: 3,
            dtype: Dtype::Bfloat16,
        };
        let first = Buffer::from_slice(&device, &(0_u16..48).collect::<Vec<_>>());
        let second = Buffer::from_slice(&device, &(100_u16..148).collect::<Vec<_>>());
        let third = Buffer::from_slice(&device, &(200_u16..248).collect::<Vec<_>>());
        let routes = [[0, 7], [1, 0], [2, 6], [1, 5], [0, 3], [2, 1], [0, 2], [1, 4]];
        let route_values = routes.iter().flatten().copied().collect::<Vec<_>>();
        let routes_buffer = Buffer::from_slice(&device, &route_values);
        let output = Buffer::new_zeroed(&device, shape.output_bytes(config));
        let kernel = Kernel::new(&device, config);
        let replay = {
            let mut recorder = stream.create_replay_program();
            recorder.record(kernel.invoke(
                shape,
                ReplayU32::Parameter(NUM_ACTIVE_ROWS),
                Buffers {
                    first_input: &first,
                    second_input: &second,
                    third_input: &third,
                    routes: &routes_buffer,
                    output: &output,
                },
            ));
            recorder.build()
        };

        for num_active_rows in [1, 8, 3, 7, 2, 6, 4, 5] {
            stream
                .submit_replay_with_arguments(
                    &replay,
                    &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, num_active_rows),
                )
                .wait();
            let expected = routes[..num_active_rows as usize]
                .iter()
                .flat_map(|&[source, row]| {
                    let base = match source {
                        0 => 0,
                        1 => 100,
                        2 => 200,
                        _ => unreachable!(),
                    };
                    (0..config.num_cols).map(move |column| base + row * config.num_cols + column)
                })
                .map(|value| value as u16)
                .collect::<Vec<_>>();
            assert_eq!(
                output.read_typed::<u16>(0, expected.len()),
                expected,
                "active rows={num_active_rows}"
            );
        }
    }

    #[test]
    fn test_routes_f32_sources() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let first = Buffer::from_slice(&device, &[1.0_f32]);
        let second = Buffer::from_slice(&device, &[2.0_f32]);
        let third = Buffer::from_slice(&device, &[3.0_f32]);
        let routes = Buffer::from_slice(&device, &[0_u32, 0, 1, 0, 2, 0]);
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
                first_input: &first,
                second_input: &second,
                third_input: &third,
                routes: &routes,
                output: &output,
            },
        ));
        stream.submit_replay(&recorder.build()).wait();
        assert_eq!(output.read_typed::<f32>(0, 3), vec![1.0, 2.0, 3.0]);
    }
}
