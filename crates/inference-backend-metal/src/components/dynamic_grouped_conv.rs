use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("metal/dynamic_grouped_conv.metal");
const REQUIRED_THREADS: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub hidden_dim: u32,
    pub group_size: u32,
    pub kernel_size: u32,
    pub io_dtype: Dtype,
    pub base_dtype: Dtype,
}

impl Config {
    pub fn validate(self) {
        assert!(self.hidden_dim > 0);
        assert!(self.group_size > 0);
        assert!(self.kernel_size > 0);
        assert!(self.hidden_dim.is_multiple_of(self.group_size));
        assert_eq!(self.io_dtype, Dtype::Bfloat16);
        assert_eq!(self.base_dtype, Dtype::Bfloat16);
        let _ = self.num_groups();
        let _ = self.projection_dim();
        let _ = self.base_bytes();
    }

    pub fn num_groups(self) -> u32 {
        self.validate_dimensions();
        self.hidden_dim / self.group_size
    }

    pub fn projection_dim(self) -> u32 {
        self.validate_dimensions();
        self.num_groups()
            .checked_mul(self.kernel_size)
            .and_then(|value| value.checked_mul(2))
            .expect("dynamic grouped-convolution projection dimension must fit u32")
    }

    pub fn base_bytes(self) -> usize {
        self.validate_dimensions();
        checked_product(
            "dynamic grouped-convolution base byte length",
            &[
                2,
                self.kernel_size as usize,
                self.hidden_dim as usize,
                self.base_dtype.item_size(),
            ],
        )
    }

    pub fn hidden_bytes(self, shape: Shape) -> usize {
        shape.validate();
        checked_product(
            "dynamic grouped-convolution hidden byte length",
            &[
                shape.num_total_query_blocks as usize,
                shape.query_block_size as usize,
                self.hidden_dim as usize,
                self.io_dtype.item_size(),
            ],
        )
    }

    pub fn projected_coefficients_bytes(self, shape: Shape) -> usize {
        shape.validate();
        checked_product(
            "dynamic grouped-convolution coefficient byte length",
            &[
                shape.num_total_query_blocks as usize,
                shape.query_block_size as usize,
                self.projection_dim() as usize,
                self.io_dtype.item_size(),
            ],
        )
    }

    fn validate_dimensions(self) {
        assert!(self.hidden_dim > 0);
        assert!(self.group_size > 0);
        assert!(self.kernel_size > 0);
        assert!(self.hidden_dim.is_multiple_of(self.group_size));
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_query_blocks: u32,
    pub query_block_size: u32,
}

impl Shape {
    pub fn validate(self) {
        assert!(self.num_total_query_blocks > 0);
        assert!(self.query_block_size > 0);
        let _ = self
            .num_total_query_blocks
            .checked_mul(self.query_block_size)
            .expect("dynamic grouped-convolution token count must fit u32");
    }

    fn num_total_tokens(self) -> u32 {
        self.num_total_query_blocks * self.query_block_size
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Side {
    Prepare,
    Finish,
}

impl Side {
    const fn index(self) -> u32 {
        match self {
            Self::Prepare => 0,
            Self::Finish => 1,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub hidden: &'a Buffer,
    pub projected_coefficients: &'a Buffer,
    pub base: &'a Buffer,
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
            kernel: CompiledKernel::new(device, SOURCE, "dynamic_grouped_conv_bf16"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: Shape,
        num_active_query_blocks: ReplayU32,
        side: Side,
        buffers: Buffers<'a>,
    ) -> Invocation<'a> {
        let invocation = Invocation {
            compute: self,
            shape,
            num_active_query_blocks,
            side,
            buffers,
        };
        invocation.validate();
        invocation
    }
}

pub struct Invocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    num_active_query_blocks: ReplayU32,
    side: Side,
    buffers: Buffers<'a>,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.compute.config;
        recorder.set_kernel(&self.compute.kernel);
        recorder.set_buffer_read(0, self.buffers.hidden, 0);
        recorder.set_buffer_read(1, self.buffers.projected_coefficients, 0);
        recorder.set_buffer_read(2, self.buffers.base, 0);
        recorder.set_buffer_write(3, self.buffers.output, 0);
        match self.num_active_query_blocks {
            ReplayU32::Fixed(value) => recorder.set_u32(4, value),
            ReplayU32::Parameter(key) => {
                recorder.bind_u32(4, key, 1, self.shape.num_total_query_blocks);
            },
        }
        recorder.set_u32(5, self.shape.query_block_size);
        recorder.set_u32(6, config.hidden_dim);
        recorder.set_u32(7, config.group_size);
        recorder.set_u32(8, config.kernel_size);
        recorder.set_u32(9, self.side.index());
        let num_values = self.shape.num_total_tokens() as usize * config.hidden_dim as usize;
        recorder.dispatch_1d(num_values, REQUIRED_THREADS);
    }
}

impl Invocation<'_> {
    fn validate(&self) {
        let config = self.compute.config;
        if let ReplayU32::Fixed(value) = self.num_active_query_blocks {
            assert_eq!(value, self.shape.num_total_query_blocks);
        }
        assert!(self.buffers.hidden.len_bytes() >= config.hidden_bytes(self.shape));
        assert!(self.buffers.projected_coefficients.len_bytes() >= config.projected_coefficients_bytes(self.shape));
        assert_eq!(self.buffers.base.len_bytes(), config.base_bytes());
        assert!(self.buffers.output.len_bytes() >= config.hidden_bytes(self.shape));
        let num_values = self.shape.num_total_tokens() as usize * config.hidden_dim as usize;
        assert!(
            u32::try_from(num_values - 1).is_ok(),
            "dynamic grouped-convolution output index exceeds the shader u32 domain"
        );
    }
}

fn checked_product(name: &str, factors: &[usize]) -> usize {
    factors
        .iter()
        .try_fold(1usize, |product, &factor| product.checked_mul(factor))
        .unwrap_or_else(|| panic!("{name} must fit usize"))
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::Stream;

    const TEST_NUM_ACTIVE_QUERY_BLOCKS: ReplayParameterKey =
        ReplayParameterKey::new("test.dynamic_grouped_conv.num_active_query_blocks");

    #[test]
    fn test_metal_matches_request_local_reference_for_both_sides() {
        assert_metal_matches_request_local_reference(8, &[1, 8, 3, 7, 2, 6, 4, 5]);
        assert_metal_matches_request_local_reference(1, &[1]);
    }

    fn assert_metal_matches_request_local_reference(num_total_query_blocks: u32, active_query_block_counts: &[u32]) {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config {
            hidden_dim: 8,
            group_size: 4,
            kernel_size: 3,
            io_dtype: Dtype::Bfloat16,
            base_dtype: Dtype::Bfloat16,
        };
        let shape = Shape {
            num_total_query_blocks,
            query_block_size: 4,
        };
        let num_tokens = shape.num_total_tokens() as usize;
        let hidden = (0..num_tokens * config.hidden_dim as usize)
            .map(|index| bf16::from_f32((index as f32 % 17.0 - 8.0) / 11.0).to_bits())
            .collect::<Vec<_>>();
        let coefficients = (0..num_tokens * config.projection_dim() as usize)
            .map(|index| bf16::from_f32((index as f32 % 13.0 - 6.0) / 23.0).to_bits())
            .collect::<Vec<_>>();
        let base = (0..2 * config.kernel_size as usize * config.hidden_dim as usize)
            .map(|index| bf16::from_f32((index as f32 % 7.0 - 3.0) / 19.0).to_bits())
            .collect::<Vec<_>>();
        let hidden_buffer = Buffer::from_slice(&device, &hidden);
        let coefficient_buffer = Buffer::from_slice(&device, &coefficients);
        let base_buffer = Buffer::from_slice(&device, &base);
        let output = Buffer::new_zeroed_elements(&device, hidden.len(), Dtype::Bfloat16);
        let compute = Compute::new(&device, config);

        for side in [Side::Prepare, Side::Finish] {
            let replay = {
                let mut builder = stream.create_replay_program();
                builder.record(compute.invoke(
                    shape,
                    ReplayU32::Parameter(TEST_NUM_ACTIVE_QUERY_BLOCKS),
                    side,
                    Buffers {
                        hidden: &hidden_buffer,
                        projected_coefficients: &coefficient_buffer,
                        base: &base_buffer,
                        output: &output,
                    },
                ));
                builder.build()
            };
            for &num_active_query_blocks in active_query_block_counts {
                let arguments = ReplayArguments::new().with_u32(TEST_NUM_ACTIVE_QUERY_BLOCKS, num_active_query_blocks);
                stream.submit_replay_with_arguments(&replay, &arguments).wait();

                let expected = reference(
                    config,
                    shape,
                    num_active_query_blocks as usize,
                    side,
                    &hidden,
                    &coefficients,
                    &base,
                );
                let actual = output.read_typed::<u16>(0, expected.len());
                assert_eq!(actual, expected);
            }
        }
    }

    fn reference(
        config: Config,
        shape: Shape,
        num_active_query_blocks: usize,
        side: Side,
        hidden: &[u16],
        coefficients: &[u16],
        base: &[u16],
    ) -> Vec<u16> {
        let hidden_dim = config.hidden_dim as usize;
        let group_size = config.group_size as usize;
        let groups = config.num_groups() as usize;
        let taps = config.kernel_size as usize;
        let block_size = shape.query_block_size as usize;
        let mut output = vec![0u16; num_active_query_blocks * block_size * hidden_dim];
        for block in 0..num_active_query_blocks {
            for row in 0..block_size {
                let token = block * block_size + row;
                for hidden_index in 0..hidden_dim {
                    let group = hidden_index / group_size;
                    let mut value = 0.0f32;
                    for tap in 0..taps.min(row + 1) {
                        let source_token = token - tap;
                        let coefficient_index = (((token * 2 + side.index() as usize) * taps + tap) * groups) + group;
                        let base_index = (side.index() as usize * taps + tap) * hidden_dim + hidden_index;
                        value += (bf16::from_bits(base[base_index]).to_f32()
                            + bf16::from_bits(coefficients[coefficient_index]).to_f32())
                            * bf16::from_bits(hidden[source_token * hidden_dim + hidden_index]).to_f32();
                    }
                    output[token * hidden_dim + hidden_index] = bf16::from_f32(value).to_bits();
                }
            }
        }
        output
    }
}
