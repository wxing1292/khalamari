use std::collections::HashSet;
use std::path::PathBuf;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayU32;
use crate::mlx_headers::find_mlx_metal_header_root;
use crate::mlx_headers::read_mlx_metal_header;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants {
    config: Config,
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn current(config: Config) -> Self {
        config.validate();
        let num_threads_needed = config.num_values_per_row.div_ceil(4);
        let required_threads = num_threads_needed.div_ceil(32) * 32;
        Self {
            config,
            thread_block: ThreadBlockConstants { required_threads },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub num_values_per_row: u32,
    pub dtype: Dtype,
}

impl Config {
    fn validate(self) {
        assert!(self.num_values_per_row > 0);
        assert!(self.num_values_per_row <= 4096);
        match self.dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 softmax is not implemented"),
            dtype => panic!("unsupported softmax dtype {dtype:?}"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_rows: u32,
}

impl Shape {
    fn validate(self) {
        assert!(self.num_total_rows > 0);
    }

    fn bytes(self, config: Config) -> usize {
        (self.num_total_rows as usize)
            .checked_mul(config.num_values_per_row as usize)
            .and_then(|num_values| num_values.checked_mul(config.dtype.item_size()))
            .expect("softmax byte length must fit usize")
    }
}

pub struct Kernel {
    constants: KernelConstants,
    kernel: CompiledKernel,
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub input: &'a Buffer,
    pub output: &'a Buffer,
}

impl Kernel {
    pub fn new(device: &Device, config: Config) -> Self {
        let constants = KernelConstants::current(config);
        let source = softmax_source();
        Self {
            constants,
            kernel: CompiledKernel::new(device, source, "block_softmax_bfloat16"),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, num_active_rows: ReplayU32, buffers: Buffers<'a>) -> Invocation<'a> {
        shape.validate();
        let bytes = shape.bytes(self.constants.config);
        assert!(bytes <= buffers.input.len_bytes());
        assert!(bytes <= buffers.output.len_bytes());
        Invocation {
            kernel: self,
            shape,
            buffers,
            num_active_rows_key: active_key(shape.num_total_rows, num_active_rows),
        }
    }
}

fn active_key(num_total_rows: u32, num_active_rows: ReplayU32) -> Option<ReplayParameterKey> {
    match num_active_rows {
        ReplayU32::Fixed(num_active_rows) => {
            assert_eq!(num_active_rows, num_total_rows);
            None
        },
        ReplayU32::Parameter(key) => Some(key),
    }
}

pub struct Invocation<'a> {
    kernel: &'a Kernel,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_rows_key: Option<ReplayParameterKey>,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let shape = self.shape;
        let constants = self.kernel.constants;
        let config = constants.config;
        recorder.set_kernel(&self.kernel.kernel);
        recorder.set_buffer_read(0, self.buffers.input, 0);
        recorder.set_buffer_write(1, self.buffers.output, 0);
        recorder.set_i32(2, config.num_values_per_row as i32);
        match self.num_active_rows_key {
            Some(key) => recorder.bind_u32(3, key, 1, shape.num_total_rows),
            None => recorder.set_u32(3, shape.num_total_rows),
        }

        let required_threads = constants.thread_block.required_threads as usize;
        recorder.dispatch_1d(shape.num_total_rows as usize * required_threads, required_threads);
    }
}

fn softmax_source() -> &'static str {
    static EXPANDED_SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    EXPANDED_SOURCE.get_or_init(|| {
        let root = mlx_metal_header_root();
        let mut included = HashSet::new();
        let mut source = String::new();
        source.push_str(
            "#include <metal_stdlib>\n#include <metal_common>\n#include <metal_simdgroup>\nusing namespace metal;\n",
        );
        source.push_str(&read_mlx_metal_header(
            &root,
            "mlx/backend/metal/kernels/defines.h",
            &mut included,
        ));
        source.push_str(&read_mlx_metal_header(
            &root,
            "mlx/backend/metal/kernels/utils.h",
            &mut included,
        ));
        source.push_str(&read_mlx_metal_header(
            &root,
            "mlx/backend/metal/kernels/softmax.h",
            &mut included,
        ));
        source.push_str(PARAMETERIZED_SOFTMAX_SOURCE);
        source
    })
}

// This kernel is the MLX single-row softmax with one fixed-capacity replay guard.
// The guard is uniform across the whole threadgroup and occurs before all input
// reads and threadgroup barriers.
const PARAMETERIZED_SOFTMAX_SOURCE: &str = r#"
template <typename T, typename AccT = T, int N_READS = SOFTMAX_N_READS>
[[kernel]] void softmax_single_row_bucketed(
    const device T* in,
    device T* out,
    constant int& axis_size,
    constant uint& num_active_rows,
    uint gid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  if (gid >= num_active_rows) {
    return;
  }

  int lid = _lid;
  constexpr int SIMD_SIZE = 32;
  threadgroup AccT local_max[SIMD_SIZE];
  threadgroup AccT local_normalizer[SIMD_SIZE];
  AccT ld[N_READS];

  in += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      ld[i] = AccT(in[i]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      ld[i] =
          ((lid * N_READS + i) < axis_size) ? AccT(in[i]) : Limits<AccT>::min;
    }
  }
  if (simd_group_id == 0) {
    local_max[simd_lane_id] = Limits<AccT>::min;
    local_normalizer[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  AccT maxval = Limits<AccT>::finite_min;
  for (int i = 0; i < N_READS; i++) {
    maxval = (maxval < ld[i]) ? ld[i] : maxval;
  }
  maxval = simd_max(maxval);
  if (simd_lane_id == 0) {
    local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    maxval = simd_max(local_max[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_max[0] = maxval;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = local_max[0];

  AccT normalizer = 0;
  for (int i = 0; i < N_READS; i++) {
    AccT exp_x = softmax_exp(ld[i] - maxval);
    ld[i] = exp_x;
    normalizer += exp_x;
  }
  normalizer = simd_sum(normalizer);
  if (simd_lane_id == 0) {
    local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    normalizer = simd_sum(local_normalizer[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_normalizer[0] = normalizer;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = 1 / local_normalizer[0];

  out += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out[i] = T(ld[i] * normalizer);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] = T(ld[i] * normalizer);
      }
    }
  }
}

template [[host_name("block_softmax_bfloat16")]] [[kernel]]
decltype(softmax_single_row_bucketed<bfloat16_t, bfloat16_t>)
softmax_single_row_bucketed<bfloat16_t, bfloat16_t>;
"#;

fn mlx_metal_header_root() -> PathBuf {
    find_mlx_metal_header_root("softmax.h", |_| true, "softmax")
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::Buffers;
    use super::Config;
    use super::Kernel;
    use super::Shape;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Dtype;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::ReplayU32;
    use crate::metal::Stream;

    const NUM_ACTIVE_ROWS: ReplayParameterKey = ReplayParameterKey::new("test.softmax.num_active_rows");

    #[test]
    fn test_replay_matches_reference_across_active_counts() {
        let config = Config {
            num_values_per_row: 4,
            dtype: Dtype::Bfloat16,
        };
        let shape = Shape { num_total_rows: 8 };
        let (device, kernel) = create_softmax_kernel(config);
        let stream = Stream::new(&device);
        let input_values = (0..shape.num_total_rows as usize * config.num_values_per_row as usize)
            .map(|index| ((index * 11 + 3) % 29) as f32 * 0.25 - 3.0)
            .collect::<Vec<_>>();
        let input = bf16_buffer(&device, &input_values);
        let output = Buffer::new_zeroed(&device, shape.bytes(config));
        let replay = {
            let mut recorder = stream.create_replay_program();
            recorder.record(kernel.invoke(
                shape,
                ReplayU32::Parameter(NUM_ACTIVE_ROWS),
                Buffers {
                    input: &input,
                    output: &output,
                },
            ));
            recorder.build()
        };

        for num_active_rows in [1_usize, 8, 3, 7, 2, 6, 4, 5] {
            stream
                .submit_replay_with_arguments(
                    &replay,
                    &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, num_active_rows as u32),
                )
                .wait();
            let num_active_values = num_active_rows * config.num_values_per_row as usize;
            let actual = read_bf16_values(&output, num_active_values);
            let expected = cpu_softmax_bf16_rows(
                &input_values[..num_active_values],
                num_active_rows,
                config.num_values_per_row as usize,
            );
            assert_close(&actual, &expected, 0.01);
        }
    }

    fn create_softmax_kernel(config: Config) -> (Device, Kernel) {
        let device = Device::system_default();
        let kernel = Kernel::new(&device, config);
        (device, kernel)
    }

    fn cpu_softmax_bf16_rows(values: &[f32], num_rows: usize, num_values_per_row: usize) -> Vec<f32> {
        assert_eq!(values.len(), num_rows * num_values_per_row);
        let mut output = Vec::with_capacity(values.len());
        for row in values.chunks_exact(num_values_per_row) {
            let row: Vec<f32> = row.iter().map(|value| bf16::from_f32(*value).to_f32()).collect();
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|value| (*value - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            output.extend(exps.into_iter().map(|value| bf16::from_f32(value / sum).to_f32()));
        }
        output
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bits: Vec<u16> = values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect();
        Buffer::from_slice(device, &bits)
    }

    fn read_bf16_values(buffer: &Buffer, len: usize) -> Vec<f32> {
        buffer
            .read_typed::<u16>(0, len)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "value mismatch at {index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}
