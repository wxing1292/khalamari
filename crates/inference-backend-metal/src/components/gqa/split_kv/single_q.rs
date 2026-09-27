use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

use crate::components::assert_u32_count_domain;
use crate::components::assert_u32_index_domain;
use crate::components::checked_product;
use crate::components::gqa::kv_page_write::PageTableLayout;
use crate::components::gqa::sdpa;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;
use crate::mlx_headers::steel_mma_source;

const FP8_SOURCE: &str = include_str!("../../metal/gqa_fp8.metal");
const MAP_SOURCE: &str = include_str!("../../metal/gqa_split_kv_single_q_map.metal");
const REDUCE_SOURCE: &str = include_str!("../../metal/gqa_split_kv_single_q_reduce.metal");

#[derive(Clone, Copy, Debug, PartialEq)]
struct KernelConstants {
    config: Config,
    map: sdpa::MapKernelConstants,
    reduce: sdpa::ReduceKernelConstants,
}

impl KernelConstants {
    fn new(config: Config, execution: sdpa::ExecutionVariant) -> Self {
        config.validate();
        assert!(execution.supports(config.sdpa_config()));
        assert_eq!(execution.map.thread_block.max_q_tokens, 1);
        Self {
            config,
            map: execution.map,
            reduce: execution.reduce,
        }
    }

    fn map_threadblock_memory_bytes(self) -> usize {
        let map = self.map.thread_block;
        (map.required_threads as usize / 32 * (8 * 64 + 8 * 2) + 8 * 2) * size_of::<f32>()
    }

    fn num_q_head_ranges_per_kv_head(self) -> u32 {
        self.config
            .q_heads_per_kv_head()
            .div_ceil(self.map.thread_block.max_q_heads)
    }

    fn map_threads(self, shape: Shape) -> usize {
        checked_product(
            "GQA SDPA map thread count",
            &[
                shape.num_total_sdpa_map_task_templates as usize,
                self.config.num_kv_heads as usize,
                self.num_q_head_ranges_per_kv_head() as usize,
                self.map.thread_block.required_threads as usize,
            ],
        )
    }
}

/// SplitKV SingleQ SDPA (`T` = tokens, `H` = heads, `D` = head width):
///
/// ```text
/// Q: [Tq, Hq, D]       Q tile: [Tq_tile, Hq_tile, D]
/// K: [Tkv, Hkv, D]     K tile: [Tkv_tile, D]  (one fixed KV head)
/// V: [Tkv, Hkv, D]     V tile: [Tkv_tile, D]
/// O: [Tq, Hq, D]
/// Q tile [Tq_tile, Hq_tile, D] x K tile^T [D, Tkv_tile]
///   -> scores -> x V tile [Tkv_tile, D]
///   -> SDPAPartialOutput [Tq_tile, Hq_tile, D]
/// Map task template: { q_token_range_index, kv_token_begin, kv_token_end }
/// MapThreadBlockTask:
///   { q_token_range_index, kv_token_begin, kv_token_end } from the template
///   + { kv_head_index, q_head_range_index } from the grid
/// grid: (total Map task templates, Hkv * Q-head ranges, 1)
/// threadblock: (configured width, 1, 1)
/// parallel: Map tasks, KV heads, Q-head ranges, SIMDgroup KV ranges
/// ordered/reduce: online softmax per SIMDgroup, then threadblock reduction
/// produces: SDPAPartialOutput + statistics -> final reduce -> SDPAOutput
/// ```
///
/// This variant uses `Tq_tile = 1`. Only the Map task template is materialized.
/// It uses the shared three-`u32` TaskTemplate ABI. The complete task is
/// derived from the template, grid coordinates, and specialization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub page_bytes: u32,
    pub page_table_layout: PageTableLayout,
    pub dtype: Dtype,
}

impl Config {
    pub fn validate(self) {
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert!(self.head_dim > 0);
        assert!(self.scale > 0.0);
        assert_eq!(self.num_q_heads % self.num_kv_heads, 0);
        assert!(self.num_tokens_per_page() > 0);
        self.page_table_layout.validate();
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn num_tokens_per_page(self) -> u32 {
        let kv_bytes_per_token = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| bytes.checked_mul(size_of::<u8>().try_into().expect("FP8 size must fit u32")))
            .expect("GQA SDPA K/V bytes per token must fit u32");
        assert!(
            self.page_bytes.is_multiple_of(kv_bytes_per_token),
            "GQA page_bytes must be divisible by the K/V bytes per token"
        );
        self.page_bytes / kv_bytes_per_token
    }

    pub fn q_heads_per_kv_head(self) -> u32 {
        self.num_q_heads / self.num_kv_heads
    }

    fn sdpa_config(self) -> sdpa::Config {
        sdpa::Config {
            io_dtype: self.dtype,
            num_q_heads: self.num_q_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            tokens_per_page: self.num_tokens_per_page(),
        }
    }

    pub fn num_output_values(self, shape: Shape) -> usize {
        checked_product(
            "GQA SDPA output element count",
            &[
                shape.num_total_tokens as usize,
                self.num_q_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    pub fn num_sdpa_partial_output_stats(self, shape: Shape) -> usize {
        checked_product(
            "GQA SDPA partial-output statistic count",
            &[
                shape.num_total_sdpa_map_task_templates as usize,
                self.num_q_heads as usize,
            ],
        )
    }

    pub fn num_partial_output_values(self, shape: Shape) -> usize {
        self.num_sdpa_partial_output_stats(shape)
            .checked_mul(self.head_dim as usize)
            .expect("GQA SDPA partial output element count must fit usize")
    }

    pub fn q_bytes(self, shape: Shape) -> u64 {
        (self.num_output_values(shape) as u64)
            .checked_mul(self.dtype.item_size() as u64)
            .expect("GQA SDPA query byte length must fit u64")
    }

    pub fn kv_pages_bytes(self) -> usize {
        self.page_bytes as usize
    }

    pub fn req_slots_bytes(self, shape: Shape) -> u64 {
        (shape.num_total_tokens as u64)
            .checked_mul(size_of::<u32>() as u64)
            .expect("GQA SDPA request-slot byte length must fit u64")
    }

    pub fn page_ids_bytes(self) -> u64 {
        self.page_table_layout.bytes() as u64
    }

    pub fn sdpa_map_task_templates_bytes(self, shape: Shape) -> u64 {
        (shape.num_total_sdpa_map_task_templates as u64)
            .checked_mul(3)
            .and_then(|count| count.checked_mul(size_of::<u32>() as u64))
            .expect("GQA SDPA map TaskTemplate byte length must fit u64")
    }

    pub fn cu_sdpa_partial_outputs_bytes(self, shape: Shape) -> u64 {
        (shape.num_total_tokens as u64)
            .checked_add(1)
            .and_then(|count| count.checked_mul(size_of::<u32>() as u64))
            .expect("GQA SDPA partial-output cumulative-count byte length must fit u64")
    }

    pub fn partial_output_stats_bytes(self, shape: Shape) -> u64 {
        (self.num_sdpa_partial_output_stats(shape) as u64)
            .checked_mul(size_of::<f32>() as u64)
            .expect("GQA SDPA statistic byte length must fit u64")
    }

    pub fn partial_output_bytes(self, shape: Shape) -> u64 {
        (self.num_partial_output_values(shape) as u64)
            .checked_mul(self.dtype.item_size() as u64)
            .expect("GQA SDPA partial output byte length must fit u64")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_tokens: u32,
    pub num_total_sdpa_map_task_templates: u32,
}

impl Shape {
    pub fn validate(self, config: Config) {
        config.validate();
        assert!(self.num_total_tokens > 0);
        assert!(self.num_total_sdpa_map_task_templates > 0);
        assert_u32_count_domain(config.num_output_values(self), "GQA SDPA query/output");
        assert_u32_index_domain(
            config.num_sdpa_partial_output_stats(self),
            "GQA SDPA partial-output stats",
        );
        assert_u32_index_domain(config.num_partial_output_values(self), "GQA SDPA partial output");
    }
}

#[derive(Clone, Copy)]
pub struct MapBuffers<'a> {
    pub q: &'a Buffer,
    pub kv_pages: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub sdpa_map_task_templates: &'a Buffer,
    pub partial_exp_sums: &'a Buffer,
    pub partial_max_logits: &'a Buffer,
    pub partial_output: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct ReduceBuffers<'a> {
    pub partial_exp_sums: &'a Buffer,
    pub partial_max_logits: &'a Buffer,
    pub partial_output: &'a Buffer,
    pub cu_sdpa_partial_outputs: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct Scratch {
    pub partial_exp_sums: Buffer,
    pub partial_max_logits: Buffer,
    pub partial_output: Buffer,
}

impl Scratch {
    pub fn new(device: &Device, config: Config, shape: Shape) -> Self {
        shape.validate(config);
        Self {
            partial_exp_sums: Buffer::new_zeroed(device, config.partial_output_stats_bytes(shape)),
            partial_max_logits: Buffer::new_zeroed(device, config.partial_output_stats_bytes(shape)),
            partial_output: Buffer::new_zeroed(device, config.partial_output_bytes(shape)),
        }
    }

    pub fn map_buffers<'a>(
        &'a self,
        q: &'a Buffer,
        kv_pages: &'a Buffer,
        req_slots: &'a Buffer,
        page_ids: &'a Buffer,
        sdpa_map_task_templates: &'a Buffer,
    ) -> MapBuffers<'a> {
        MapBuffers {
            q,
            kv_pages,
            req_slots,
            page_ids,
            sdpa_map_task_templates,
            partial_exp_sums: &self.partial_exp_sums,
            partial_max_logits: &self.partial_max_logits,
            partial_output: &self.partial_output,
        }
    }

    pub fn reduce_buffers<'a>(&'a self, cu_sdpa_partial_outputs: &'a Buffer, output: &'a Buffer) -> ReduceBuffers<'a> {
        ReduceBuffers {
            partial_exp_sums: &self.partial_exp_sums,
            partial_max_logits: &self.partial_max_logits,
            partial_output: &self.partial_output,
            cu_sdpa_partial_outputs,
            output,
        }
    }
}

pub struct Compute {
    constants: KernelConstants,
    shape: Shape,
    map: CompiledKernel,
    reduce: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config, execution: sdpa::ExecutionVariant, shape: Shape) -> Self {
        let constants = KernelConstants::new(config, execution);
        shape.validate(config);
        assert_u32_count_domain(constants.map_threads(shape), "GQA SDPA map threads");
        assert!(
            constants.map_threadblock_memory_bytes() <= device.max_threadblock_memory_length(),
            "GQA SplitKV SingleQ shape needs {} bytes of threadblock memory but device only supports {}",
            constants.map_threadblock_memory_bytes(),
            device.max_threadblock_memory_length()
        );
        let reduce_function_name = match config.dtype {
            Dtype::Float32 => "gqa_split_kv_single_q_reduce_f32",
            Dtype::Bfloat16 => "gqa_split_kv_single_q_reduce_bf16",
            dtype => panic!("unsupported GQA SplitKV SingleQ reduce dtype {dtype:?}"),
        };
        let map = CompiledKernel::new(
            device,
            &gqa_split_kv_single_q_map_source(constants),
            "gqa_split_kv_single_q_map",
        );
        assert!(map.static_threadblock_memory_length() <= device.max_threadblock_memory_length());
        assert!(constants.map.thread_block.required_threads as usize <= map.max_total_threads_per_threadblock());
        assert_eq!(map.thread_execution_width(), 32);
        Self {
            constants,
            shape,
            map,
            reduce: CompiledKernel::new(
                device,
                &gqa_split_kv_single_q_reduce_source(constants),
                reduce_function_name,
            ),
        }
    }

    pub fn invoke_map<'a>(
        &self,
        buffers: MapBuffers<'a>,
        page_table_index: ReplayU32,
        num_active_tokens: ReplayU32,
        num_active_kv_splits: ReplayU32,
    ) -> MapInvocation<'a> {
        MapInvocation {
            pipeline: self.map.as_raw_retained(),
            constants: self.constants,
            shape: self.shape,
            buffers,
            page_table_index,
            num_active_tokens,
            num_active_kv_splits,
        }
    }

    pub fn invoke_reduce<'a>(&self, buffers: ReduceBuffers<'a>, num_active_tokens: ReplayU32) -> ReduceInvocation<'a> {
        ReduceInvocation {
            pipeline: self.reduce.as_raw_retained(),
            constants: self.constants,
            shape: self.shape,
            buffers,
            num_active_tokens,
        }
    }
}

fn gqa_split_kv_single_q_map_source(constants: KernelConstants) -> String {
    let config = constants.config;
    let map = constants.map.thread_block;
    let dtype = metal_dtype_name(config.dtype);
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
{steel}
{FP8_SOURCE}
using T = {dtype};
#define NUM_Q_HEADS {num_q_heads}
#define NUM_KV_HEADS {num_kv_heads}
#define KV_HEAD_DIM {head_dim}
#define ATTENTION_SCALE {scale}
#define Q_HEADS_PER_KV_HEAD {q_heads_per_kv_head}
#define MAX_Q_HEADS {max_q_heads}
#define NUM_Q_HEAD_RANGES_PER_KV_HEAD {num_q_head_ranges_per_kv_head}
#define NUM_TOKENS {num_tokens}
#define PAGE_BYTES {page_bytes}
#define REQUIRED_THREADS {required_threads}
#define NUM_GQA_LAYERS {num_gqa_layers}
#define NUM_BLOCKS {num_blocks}
#define NUM_PAGE_IDS_PER_BLOCK {num_page_ids_per_block}

{MAP_SOURCE}
"#,
        steel = steel_mma_source(),
        dtype = dtype,
        head_dim = config.head_dim,
        scale = config.scale,
        q_heads_per_kv_head = config.q_heads_per_kv_head(),
        max_q_heads = map.max_q_heads,
        num_q_head_ranges_per_kv_head = constants.num_q_head_ranges_per_kv_head(),
        num_kv_heads = config.num_kv_heads,
        num_tokens = config.num_tokens_per_page(),
        num_q_heads = config.num_q_heads,
        page_bytes = config.page_bytes,
        required_threads = map.required_threads,
        num_gqa_layers = config.page_table_layout.num_gqa_layers,
        num_blocks = config.page_table_layout.num_blocks,
        num_page_ids_per_block = config.page_table_layout.num_page_ids_per_block,
    )
}

fn metal_dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Float32 => "float",
        Dtype::Bfloat16 => "bfloat16_t",
        unsupported_dtype => panic!("unsupported GQA SplitKV SingleQ map dtype {unsupported_dtype:?}"),
    }
}

pub struct MapInvocation<'a> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    constants: KernelConstants,
    shape: Shape,
    buffers: MapBuffers<'a>,
    page_table_index: ReplayU32,
    num_active_tokens: ReplayU32,
    num_active_kv_splits: ReplayU32,
}

impl Operator for MapInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        let shape = self.shape;
        recorder.set_retained_pipeline_state(&self.pipeline);
        recorder.set_buffer_read(0, self.buffers.q, 0);
        recorder.set_buffer_read(1, self.buffers.kv_pages, 0);
        recorder.set_buffer_read(2, self.buffers.req_slots, 0);
        recorder.set_buffer_read(3, self.buffers.page_ids, 0);
        recorder.set_buffer_read(4, self.buffers.sdpa_map_task_templates, 0);
        recorder.set_buffer_write(5, self.buffers.partial_exp_sums, 0);
        recorder.set_buffer_write(6, self.buffers.partial_max_logits, 0);
        recorder.set_buffer_write(7, self.buffers.partial_output, 0);
        let max_page_table_index = self.constants.config.page_table_layout.num_gqa_layers - 1;
        match self.page_table_index {
            ReplayU32::Fixed(page_table_index) => {
                assert!(
                    page_table_index <= max_page_table_index,
                    "GQA page-table index exceeds layer count"
                );
                recorder.set_u32(8, page_table_index);
            },
            ReplayU32::Parameter(key) => recorder.bind_u32(8, key, 0, max_page_table_index),
        }
        set_replay_u32(
            recorder,
            9,
            self.num_active_tokens,
            shape.num_total_tokens,
            "GQA SplitKV SingleQ active token count",
        );
        set_replay_u32(
            recorder,
            10,
            self.num_active_kv_splits,
            shape.num_total_sdpa_map_task_templates,
            "GQA SplitKV SingleQ active KV split count",
        );
        recorder.dispatch_threadblocks(
            (
                shape.num_total_sdpa_map_task_templates as usize,
                (self.constants.config.num_kv_heads * self.constants.num_q_head_ranges_per_kv_head()) as usize,
                1,
            ),
            (self.constants.map.thread_block.required_threads as usize, 1, 1),
        );
    }
}

impl MapInvocation<'_> {
    fn validate(&self) {
        let config = self.constants.config;
        assert!(self.buffers.q.len_bytes_u64() >= config.q_bytes(self.shape));
        assert!(self.buffers.kv_pages.len_bytes_u64() >= config.page_bytes as u64);
        assert!(self.buffers.req_slots.len_bytes_u64() >= config.req_slots_bytes(self.shape));
        assert!(self.buffers.page_ids.len_bytes_u64() >= config.page_ids_bytes());
        assert!(
            self.buffers.sdpa_map_task_templates.len_bytes_u64() >= config.sdpa_map_task_templates_bytes(self.shape)
        );
        assert!(self.buffers.partial_exp_sums.len_bytes_u64() >= config.partial_output_stats_bytes(self.shape));
        assert!(self.buffers.partial_max_logits.len_bytes_u64() >= config.partial_output_stats_bytes(self.shape));
        assert!(self.buffers.partial_output.len_bytes_u64() >= config.partial_output_bytes(self.shape));
    }
}

pub struct ReduceInvocation<'a> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    constants: KernelConstants,
    shape: Shape,
    buffers: ReduceBuffers<'a>,
    num_active_tokens: ReplayU32,
}

impl Operator for ReduceInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        let shape = self.shape;
        recorder.set_retained_pipeline_state(&self.pipeline);
        recorder.set_buffer_read(0, self.buffers.partial_exp_sums, 0);
        recorder.set_buffer_read(1, self.buffers.partial_max_logits, 0);
        recorder.set_buffer_read(2, self.buffers.partial_output, 0);
        recorder.set_buffer_read(3, self.buffers.cu_sdpa_partial_outputs, 0);
        recorder.set_buffer_write(4, self.buffers.output, 0);
        set_replay_u32(
            recorder,
            5,
            self.num_active_tokens,
            shape.num_total_tokens,
            "GQA SplitKV SingleQ active token count",
        );
        recorder.dispatch_1d(
            self.constants.config.num_output_values(shape),
            self.constants.reduce.thread_block.required_threads as usize,
        );
    }
}

fn set_replay_u32(recorder: &CommandRecorder<'_>, index: usize, value: ReplayU32, max_value: u32, name: &str) {
    match value {
        ReplayU32::Fixed(value) => {
            assert!(value > 0, "{name} must be positive");
            assert!(value <= max_value, "{name} exceeds recorded capacity");
            recorder.set_u32(index, value);
        },
        ReplayU32::Parameter(key) => recorder.bind_u32(index, key, 1, max_value),
    }
}

impl ReduceInvocation<'_> {
    fn validate(&self) {
        let config = self.constants.config;
        assert!(self.buffers.partial_exp_sums.len_bytes_u64() >= config.partial_output_stats_bytes(self.shape));
        assert!(self.buffers.partial_max_logits.len_bytes_u64() >= config.partial_output_stats_bytes(self.shape));
        assert!(self.buffers.partial_output.len_bytes_u64() >= config.partial_output_bytes(self.shape));
        assert!(
            self.buffers.cu_sdpa_partial_outputs.len_bytes_u64() >= config.cu_sdpa_partial_outputs_bytes(self.shape)
        );
        assert!(self.buffers.output.len_bytes_u64() >= config.q_bytes(self.shape));
    }
}

fn gqa_split_kv_single_q_reduce_source(constants: KernelConstants) -> String {
    let config = constants.config;
    let constants = format!(
        "using namespace metal;\n\nconstant uint num_q_heads = {}u;\nconstant uint head_dim = {}u;",
        config.num_q_heads, config.head_dim,
    );
    REDUCE_SOURCE.replacen("using namespace metal;", &constants, 1)
}

#[cfg(test)]
#[path = "single_q_test.rs"]
mod tests;
