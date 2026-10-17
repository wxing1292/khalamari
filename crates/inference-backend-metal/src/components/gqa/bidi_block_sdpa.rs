use crate::components::assert_u32_count_domain;
use crate::components::assert_u32_index_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("../metal/gqa_bidi_block_sdpa.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
    simdgroup_width: u32,
}

impl ThreadBlockConstants {
    fn current() -> Self {
        Self {
            required_threads: 32,
            simdgroup_width: 32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct KernelConstants {
    config: Config,
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn current(config: Config) -> Self {
        Self {
            config,
            thread_block: ThreadBlockConstants::current(),
        }
    }
}

/// Dense request-block SDPA that writes into a selected SplitKV partial-output
/// layout.
///
/// The grid supplies one Q-head index, Q-token-range index, and range-local
/// Q-token offset for each threadblock. `q_token_ranges` derives the flat Q-token
/// index. The end of the matching cumulative partial-output range identifies
/// the block partial. One active threadblock owns one Q-token/Q-head bidirectional block SDPA
/// task. The selected SplitKV reducer later combines the bidirectional block
/// and persistent-history partial outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub block_size: u32,
    pub max_q_tokens: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub dtype: Dtype,
}

impl Config {
    pub fn validate(self) {
        let thread_block = ThreadBlockConstants::current();
        assert!(self.block_size > 0);
        assert!(self.max_q_tokens > 0);
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert_eq!(self.num_q_heads % self.num_kv_heads, 0);
        assert!(self.head_dim > 0);
        assert!(self.scale.is_finite() && self.scale > 0.0);
        assert_eq!(
            self.head_dim % thread_block.simdgroup_width,
            0,
            "bidirectional block SDPA head_dim must be divisible by the SIMD width"
        );
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    fn q_elements(self, shape: Shape) -> usize {
        checked_product(
            "bidirectional block SDPA Q element count",
            &[
                shape.num_total_tokens as usize,
                self.num_q_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    fn kv_elements(self, shape: Shape) -> usize {
        checked_product(
            "bidirectional block SDPA K/V element count",
            &[
                shape.num_total_tokens as usize,
                self.num_kv_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    fn partial_output_stat_elements(self, shape: Shape) -> usize {
        checked_product(
            "bidirectional block SDPA partial-output statistic element count",
            &[
                shape.num_total_partial_output_slots as usize,
                self.num_q_heads as usize,
                self.max_q_tokens as usize,
            ],
        )
    }

    fn partial_output_values(self, shape: Shape) -> usize {
        self.partial_output_stat_elements(shape)
            .checked_mul(self.head_dim as usize)
            .expect("bidirectional block SDPA partial-output element count must fit usize")
    }

    fn dispatch_threads(self, shape: Shape, thread_block: ThreadBlockConstants) -> usize {
        checked_product(
            "bidirectional block SDPA thread count",
            &[
                shape.num_total_q_token_ranges as usize,
                self.num_q_heads as usize,
                self.max_q_tokens as usize,
                thread_block.required_threads as usize,
            ],
        )
    }

    fn thread_block_memory_bytes(self) -> usize {
        (self.block_size as usize)
            .checked_mul(size_of::<f32>())
            .expect("bidirectional block SDPA thread-block memory must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_tokens: u32,
    pub num_total_q_token_ranges: u32,
    pub num_total_partial_output_slots: u32,
}

impl Shape {
    pub fn validate(self, config: Config) {
        config.validate();
        assert!(self.num_total_tokens > 0);
        assert!(self.num_total_q_token_ranges > 0 && self.num_total_q_token_ranges <= self.num_total_tokens);
        assert!(self.num_total_partial_output_slots >= self.num_total_q_token_ranges);
        assert_eq!(
            self.num_total_tokens % config.block_size,
            0,
            "bidirectional block SDPA tokens must contain complete request blocks"
        );
        assert_u32_count_domain(config.q_elements(self), "bidirectional block SDPA Q");
        assert_u32_count_domain(config.kv_elements(self), "bidirectional block SDPA K/V");
        assert_u32_index_domain(
            config.partial_output_stat_elements(self),
            "bidirectional block SDPA partial-output statistics",
        );
        assert_u32_index_domain(
            config.partial_output_values(self),
            "bidirectional block SDPA partial output",
        );
        assert_u32_count_domain(
            config.dispatch_threads(self, ThreadBlockConstants::current()),
            "bidirectional block SDPA threads",
        );
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub q: &'a Buffer,
    pub local_k: &'a Buffer,
    pub local_v: &'a Buffer,
    pub q_token_ranges: &'a Buffer,
    pub cu_sdpa_partial_outputs: &'a Buffer,
    pub partial_exp_sums: &'a Buffer,
    pub partial_max_logits: &'a Buffer,
    pub partial_output: &'a Buffer,
}

pub struct Compute {
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        let constants = KernelConstants::current(config);
        let function_name = match config.dtype {
            Dtype::Float32 => "gqa_bidi_block_sdpa_f32",
            Dtype::Bfloat16 => "gqa_bidi_block_sdpa_bf16",
            dtype => panic!("unsupported bidirectional block SDPA dtype {dtype:?}"),
        };
        let kernel = CompiledKernel::new(device, &bidi_block_sdpa_source(constants), function_name);
        assert_eq!(
            kernel.thread_execution_width(),
            constants.thread_block.simdgroup_width as usize,
            "bidirectional block SDPA requires a 32-thread SIMDgroup"
        );
        assert!(
            constants.thread_block.required_threads as usize <= kernel.max_total_threads_per_threadblock(),
            "bidirectional block SDPA requires {} threads per thread block but the pipeline supports {}",
            constants.thread_block.required_threads,
            kernel.max_total_threads_per_threadblock()
        );
        let max_thread_block_memory_length = device.max_threadblock_memory_length();
        assert!(
            config.thread_block_memory_bytes() <= max_thread_block_memory_length,
            "bidirectional block SDPA requires {} bytes of thread-block memory but the device supports {}",
            config.thread_block_memory_bytes(),
            max_thread_block_memory_length
        );
        assert!(
            kernel.static_threadblock_memory_length() <= max_thread_block_memory_length,
            "bidirectional block SDPA pipeline uses {} bytes of static thread-block memory but the device supports {}",
            kernel.static_threadblock_memory_length(),
            max_thread_block_memory_length
        );
        Self { constants, kernel }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: Shape,
        num_active_q_token_ranges: ReplayU32,
        buffers: Buffers<'a>,
    ) -> Invocation<'a> {
        let invocation = Invocation {
            constants: self.constants,
            kernel: &self.kernel,
            shape,
            num_active_q_token_ranges,
            buffers,
        };
        invocation.validate();
        invocation
    }
}

fn bidi_block_sdpa_source(kernel_constants: KernelConstants) -> String {
    let config = kernel_constants.config;
    let source_constants = format!(
        "using namespace metal;\n\nconstant uint block_size = {}u;\nconstant uint num_q_heads = {}u;\nconstant uint \
         num_kv_heads = {}u;\nconstant uint head_dim = {}u;\nconstant uint max_q_tokens = {}u;\nconstant float \
         attention_scale = {:.9e}f;\nconstant uint simd_width = {}u;\nconstant uint q_values_per_thread = head_dim / \
         simd_width;",
        config.block_size,
        config.num_q_heads,
        config.num_kv_heads,
        config.head_dim,
        config.max_q_tokens,
        config.scale,
        kernel_constants.thread_block.simdgroup_width,
    );
    SOURCE.replacen("using namespace metal;", &source_constants, 1)
}

pub struct Invocation<'a> {
    constants: KernelConstants,
    kernel: &'a CompiledKernel,
    shape: Shape,
    num_active_q_token_ranges: ReplayU32,
    buffers: Buffers<'a>,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.constants.config;
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.q, 0);
        recorder.set_buffer_read(1, self.buffers.local_k, 0);
        recorder.set_buffer_read(2, self.buffers.local_v, 0);
        recorder.set_buffer_read(3, self.buffers.q_token_ranges, 0);
        recorder.set_buffer_read(4, self.buffers.cu_sdpa_partial_outputs, 0);
        recorder.set_buffer_write(5, self.buffers.partial_exp_sums, 0);
        recorder.set_buffer_write(6, self.buffers.partial_max_logits, 0);
        recorder.set_buffer_write(7, self.buffers.partial_output, 0);
        match self.num_active_q_token_ranges {
            ReplayU32::Fixed(value) => recorder.set_u32(8, value),
            ReplayU32::Parameter(key) => {
                recorder.bind_u32(8, key, 1, self.shape.num_total_q_token_ranges);
            },
        }
        recorder.dispatch_threadblocks(
            (
                config.num_q_heads as usize,
                self.shape.num_total_q_token_ranges as usize,
                config.max_q_tokens as usize,
            ),
            (self.constants.thread_block.required_threads as usize, 1, 1),
        );
    }
}

impl Invocation<'_> {
    fn validate(&self) {
        let config = self.constants.config;
        self.shape.validate(config);
        if let ReplayU32::Fixed(value) = self.num_active_q_token_ranges {
            assert_eq!(value, self.shape.num_total_q_token_ranges);
        }
        assert!(self.buffers.q.len_bytes() >= bytes(config.q_elements(self.shape), config.dtype));
        assert!(self.buffers.local_k.len_bytes() >= bytes(config.kv_elements(self.shape), config.dtype));
        assert!(self.buffers.local_v.len_bytes() >= bytes(config.kv_elements(self.shape), config.dtype));
        assert!(
            self.buffers.q_token_ranges.len_bytes()
                >= (self.shape.num_total_q_token_ranges as usize)
                    .checked_mul(2 * size_of::<u32>())
                    .expect("bidirectional block SDPA Q-token-range bytes must fit usize")
        );
        assert!(
            self.buffers.cu_sdpa_partial_outputs.len_bytes()
                >= (self.shape.num_total_q_token_ranges as usize)
                    .checked_add(1)
                    .and_then(|count| count.checked_mul(size_of::<u32>()))
                    .expect("bidirectional block SDPA cumulative partial-output bytes must fit usize")
        );
        let partial_output_stat_bytes = self
            .constants
            .config
            .partial_output_stat_elements(self.shape)
            .checked_mul(size_of::<f32>())
            .expect("bidirectional block SDPA partial-output statistic bytes must fit usize");
        assert!(self.buffers.partial_exp_sums.len_bytes() >= partial_output_stat_bytes);
        assert!(self.buffers.partial_max_logits.len_bytes() >= partial_output_stat_bytes);
        assert!(
            self.buffers.partial_output.len_bytes() >= bytes(config.partial_output_values(self.shape), config.dtype,)
        );
    }
}

fn bytes(num_elements: usize, dtype: Dtype) -> usize {
    num_elements
        .checked_mul(dtype.item_size())
        .expect("bidirectional block SDPA buffer byte length must fit usize")
}

#[cfg(test)]
#[path = "bidi_block_sdpa_test.rs"]
mod tests;
