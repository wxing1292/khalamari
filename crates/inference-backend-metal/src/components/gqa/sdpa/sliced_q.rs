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
const REDUCE_SOURCE: &str = include_str!("../../metal/gqa_sdpa_sliced_q_reduce.metal");
const GENERIC_MAP_SOURCE: &str = include_str!("../../metal/gqa_sdpa_sliced_q_map.metal");
const MAP_SOURCE: &str = include_str!("../../metal/gqa_sdpa_sliced_q_per_head_map.metal");

#[derive(Clone, Copy, Debug, PartialEq)]
struct KernelConstants {
    config: Config,
    map: sdpa::MapKernelConstants,
    reduce: sdpa::ReduceKernelConstants,
    kv_dim_tile: u32,
    direct_output: bool,
}

impl KernelConstants {
    fn new(config: Config, execution: sdpa::ExecutionVariant) -> Self {
        config.validate();
        assert!(execution.supports(config.sdpa_config()));
        assert_ne!(execution.map.kind, sdpa::MapKernelKind::SlicedKv);
        let mut constants = Self {
            config,
            map: execution.map,
            reduce: execution.reduce,
            kv_dim_tile: config.head_dim,
            direct_output: false,
        };
        // Keep a complete head dimension in registers. Chunk only the shared K/V tile.
        // The portable Metal threadgroup-memory budget is 32 KiB.
        while !constants.legacy_map() && constants.map_threadblock_memory_bytes() > 32 * 1024 {
            constants.kv_dim_tile /= 2;
        }
        assert_u32_count_domain(constants.num_head_groups(), "GQA SplitKV SlicedQ head groups");
        constants
    }

    fn legacy_map(self) -> bool {
        !self.direct_output
            && self.map.kind == sdpa::MapKernelKind::QTokensPerHead
            && matches!(self.map.thread_block.max_q_tokens, 8 | 16)
            && self.map.thread_block.kv_tokens_per_iteration <= 32
            && self.config.q_heads_per_kv_head() <= 8
    }

    fn num_q_head_ranges_per_kv_head(self) -> u32 {
        self.config
            .q_heads_per_kv_head()
            .div_ceil(self.map.thread_block.max_q_heads)
    }

    fn num_head_groups(self) -> usize {
        checked_product(
            "GQA SplitKV SlicedQ head-group count",
            &[
                self.config.num_kv_heads as usize,
                self.num_q_head_ranges_per_kv_head() as usize,
            ],
        )
    }

    fn partial_output_bytes(self, shape: Shape) -> u64 {
        checked_product_u64(
            "GQA SplitKV SlicedQ partial output byte length",
            &[
                shape.num_total_sdpa_map_task_templates as u64,
                self.config.num_q_heads as u64,
                self.map.thread_block.max_q_tokens as u64,
                self.config.head_dim as u64,
                self.config.dtype.item_size() as u64,
            ],
        )
    }

    fn partial_output_stats_bytes(self, shape: Shape) -> u64 {
        checked_product_u64(
            "GQA SplitKV SlicedQ partial statistic byte length",
            &[
                shape.num_total_sdpa_map_task_templates as u64,
                self.config.num_q_heads as u64,
                self.map.thread_block.max_q_tokens as u64,
                size_of::<f32>() as u64,
            ],
        )
    }

    fn map_threadblock_memory_bytes(self) -> usize {
        if !self.legacy_map() {
            return checked_product(
                "GQA sliced-Q K/V staging byte length",
                &[
                    self.map.thread_block.kv_tokens_per_iteration as usize,
                    self.kv_dim_tile as usize + 8,
                    2,
                ],
            );
        }
        let padded_head_dim = self.config.head_dim as usize + 16 / self.config.dtype.item_size();
        // K and V reuse one tile. Keep the measured reservation after moving
        // probabilities to registers. KV32 uses the same reservation as KV16.
        let reserved_kv_tokens = self.map.thread_block.kv_tokens_per_iteration.max(32);
        checked_product(
            "GQA SplitKV SlicedQ threadgroup memory byte length",
            &[
                reserved_kv_tokens as usize,
                padded_head_dim,
                self.config.dtype.item_size(),
            ],
        ) + checked_product(
            "GQA SplitKV SlicedQ probability reservation byte length",
            &[
                self.map.thread_block.max_q_tokens as usize,
                self.map.thread_block.max_q_heads as usize,
                16,
                size_of::<f32>(),
            ],
        ) + checked_product(
            "GQA SplitKV SlicedQ Q-token/head statistic reservation byte length",
            &[
                self.map.thread_block.max_q_tokens as usize,
                self.map.thread_block.max_q_heads as usize,
                3,
                size_of::<f32>(),
            ],
        )
    }
}

/// SplitKV SlicedQ SDPA (`T` = tokens, `H` = heads, `D` = head width):
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
/// grid: (Hkv * Q-head ranges, Map task templates, 1)
/// threadblock: (Q-token/head fragment count * 32, 1, 1); eight coordinates per fragment
/// parallel: Q-token ranges, KV heads, Q-head ranges, Q-token/head fragments
/// ordered/reduce: consecutive KV iterations merged with online softmax
/// produces: SDPAPartialOutput + statistics -> final reduce -> SDPAOutput
/// ```
///
/// Only the Map task template is materialized. It uses the shared three-`u32`
/// TaskTemplate ABI. The complete task is derived from the template, grid
/// coordinates, and specialization. A Q-token range never crosses a request
/// boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub page_bytes: u32,
    pub dtype: Dtype,
    pub page_table_layout: PageTableLayout,
}

impl Config {
    pub fn validate(self) {
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert_eq!(self.num_q_heads % self.num_kv_heads, 0);
        let sliced_q_profile = (self.head_dim, self.num_tokens_per_page());
        assert!(
            matches!(sliced_q_profile, (128, 8 | 16) | (256, 8 | 16 | 32)),
            "GQA SplitKV SlicedQ supports only (head_dim, tokens_per_page) profiles (128, 8), (128, 16), (256, 8), \
             (256, 16), and (256, 32), got {sliced_q_profile:?}"
        );
        assert!(self.scale > 0.0);
        assert_eq!(self.dtype, Dtype::Bfloat16, "GQA SplitKV SlicedQ specializes bf16");
        self.page_table_layout.validate();
    }

    pub fn num_tokens_per_page(self) -> u32 {
        let kv_bytes_per_token = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| bytes.checked_mul(size_of::<u8>().try_into().expect("FP8 size must fit u32")))
            .expect("GQA SplitKV SlicedQ K/V bytes per token must fit u32");
        assert!(self.page_bytes.is_multiple_of(kv_bytes_per_token));
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

    pub fn q_bytes(self, shape: Shape) -> u64 {
        checked_product_u64(
            "GQA SplitKV SlicedQ query byte length",
            &[
                shape.num_total_tokens as u64,
                self.num_q_heads as u64,
                self.head_dim as u64,
                self.dtype.item_size() as u64,
            ],
        )
    }

    pub fn partial_output_bytes(self, execution: sdpa::ExecutionVariant, shape: Shape) -> u64 {
        KernelConstants::new(self, execution).partial_output_bytes(shape)
    }

    pub fn partial_output_stats_bytes(self, execution: sdpa::ExecutionVariant, shape: Shape) -> u64 {
        KernelConstants::new(self, execution).partial_output_stats_bytes(shape)
    }

    /// Total SMEM reservation, including the compiled Map kernel's static memory.
    pub fn map_threadblock_memory_bytes(self, execution: sdpa::ExecutionVariant) -> usize {
        KernelConstants::new(self, execution).map_threadblock_memory_bytes()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_tokens: u32,
    pub num_total_q_token_tiles: u32,
    pub num_total_sdpa_map_task_templates: u32,
}

impl Shape {
    pub fn validate(self, config: Config) {
        config.validate();
        assert!(self.num_total_tokens > 0);
        assert!(self.num_total_q_token_tiles > 0 && self.num_total_q_token_tiles <= self.num_total_tokens);
        assert!(self.num_total_sdpa_map_task_templates >= self.num_total_q_token_tiles);
        assert_u32_index_domain(
            self.num_q_token_range_values(),
            "GQA SplitKV SlicedQ Q-token-range metadata",
        );
        assert_u32_index_domain(
            self.num_visible_kv_token_range_values(),
            "GQA SplitKV SlicedQ visible K/V-token-range metadata",
        );
        assert_u32_index_domain(
            self.num_sdpa_map_task_template_values(),
            "GQA SplitKV SlicedQ map TaskTemplate metadata",
        );
    }

    fn num_q_token_range_values(self) -> usize {
        checked_product(
            "GQA SplitKV SlicedQ Q-token-range metadata element count",
            &[self.num_total_q_token_tiles as usize, 2],
        )
    }

    fn num_visible_kv_token_range_values(self) -> usize {
        checked_product(
            "GQA SplitKV SlicedQ visible K/V-token-range metadata element count",
            &[self.num_total_tokens as usize, 2],
        )
    }

    fn num_sdpa_map_task_template_values(self) -> usize {
        checked_product(
            "GQA SplitKV SlicedQ map TaskTemplate metadata element count",
            &[self.num_total_sdpa_map_task_templates as usize, 3],
        )
    }

    fn num_cu_sdpa_partial_output_values(self) -> usize {
        self.num_total_q_token_tiles as usize + 1
    }
}

fn checked_product_u64(name: &str, factors: &[u64]) -> u64 {
    factors
        .iter()
        .try_fold(1u64, |product, &factor| product.checked_mul(factor))
        .unwrap_or_else(|| panic!("{name} must fit u64"))
}

#[derive(Clone, Copy)]
pub struct MapBuffers<'a> {
    pub q: &'a Buffer,
    pub kv_pages: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub page_ids: &'a Buffer,
    /// Per-flat-Q-token half-open request-local visible K/V ranges.
    pub visible_kv_token_ranges: &'a Buffer,
    pub q_token_ranges: &'a Buffer,
    pub sdpa_map_task_templates: &'a Buffer,
    pub partial_output: &'a Buffer,
    pub partial_exp_sums: &'a Buffer,
    pub partial_max_logits: &'a Buffer,
}

/// Unsplit SDPA inputs and final output. Each Q range must have exactly one
/// Map template. That template must cover the union of its visible KV ranges.
#[derive(Clone, Copy)]
pub struct DirectBuffers<'a> {
    pub q: &'a Buffer,
    pub kv_pages: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub visible_kv_token_ranges: &'a Buffer,
    pub q_token_ranges: &'a Buffer,
    pub sdpa_map_task_templates: &'a Buffer,
    pub output: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct ReduceBuffers<'a> {
    pub partial_output: &'a Buffer,
    pub partial_exp_sums: &'a Buffer,
    pub partial_max_logits: &'a Buffer,
    pub q_token_ranges: &'a Buffer,
    pub cu_sdpa_partial_outputs: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct Compute {
    constants: KernelConstants,
    shape: Shape,
    map: CompiledKernel,
    map_dynamic_memory_bytes: usize,
    reduce: Option<CompiledKernel>,
}

impl Compute {
    pub fn new(device: &Device, config: Config, execution: sdpa::ExecutionVariant, shape: Shape) -> Self {
        Self::create(device, config, execution, shape, false)
    }

    pub fn new_direct(device: &Device, config: Config, execution: sdpa::ExecutionVariant, shape: Shape) -> Self {
        assert_eq!(shape.num_total_sdpa_map_task_templates, shape.num_total_q_token_tiles);
        Self::create(device, config, execution, shape, true)
    }

    fn create(
        device: &Device,
        config: Config,
        execution: sdpa::ExecutionVariant,
        shape: Shape,
        direct_output: bool,
    ) -> Self {
        let mut constants = KernelConstants::new(config, execution);
        constants.direct_output = direct_output;
        shape.validate(config);
        let map_memory_bytes = constants.map_threadblock_memory_bytes();
        assert!(
            map_memory_bytes <= device.max_threadblock_memory_length(),
            "GQA SplitKV SlicedQ shape needs {} bytes of threadblock memory but device only supports {}",
            map_memory_bytes,
            device.max_threadblock_memory_length()
        );
        let source = source(constants);
        let (map_source, name) = if constants.legacy_map() {
            (MAP_SOURCE, "gqa_sdpa_sliced_q_per_head_map")
        } else {
            (GENERIC_MAP_SOURCE, "gqa_sdpa_sliced_q_map")
        };
        let map = CompiledKernel::new(
            device,
            &format!("{}\n{source}\n{FP8_SOURCE}\n{map_source}", steel_mma_source()),
            name,
        );
        assert!(
            map.static_threadblock_memory_length() <= map_memory_bytes,
            "GQA SplitKV SlicedQ static memory exceeds its reservation"
        );
        assert!(
            constants.map.thread_block.required_threads as usize <= map.max_total_threads_per_threadblock(),
            "GQA SplitKV SlicedQ threadblock exceeds the Map pipeline thread limit"
        );
        assert_eq!(map.thread_execution_width(), 32);
        Self {
            constants,
            shape,
            map_dynamic_memory_bytes: map_memory_bytes - map.static_threadblock_memory_length(),
            map,
            reduce: (!direct_output).then(|| {
                CompiledKernel::new(
                    device,
                    &format!("{source}\n{REDUCE_SOURCE}"),
                    "gqa_sdpa_sliced_q_reduce",
                )
            }),
        }
    }

    pub fn invoke_map<'a>(
        &self,
        buffers: MapBuffers<'a>,
        page_table_index: ReplayU32,
        num_active_tokens: ReplayU32,
        num_active_q_token_tiles: ReplayU32,
        num_active_kv_splits: ReplayU32,
    ) -> MapInvocation<'a> {
        assert!(!self.constants.direct_output);
        page_table_index.validate_range(0, self.constants.config.page_table_layout.num_gqa_layers - 1);
        num_active_tokens.validate_range(1, self.shape.num_total_tokens);
        num_active_q_token_tiles.validate_range(1, self.shape.num_total_q_token_tiles);
        num_active_kv_splits.validate_range(1, self.shape.num_total_sdpa_map_task_templates);
        let inputs = DirectBuffers {
            q: buffers.q,
            kv_pages: buffers.kv_pages,
            req_slots: buffers.req_slots,
            page_ids: buffers.page_ids,
            visible_kv_token_ranges: buffers.visible_kv_token_ranges,
            q_token_ranges: buffers.q_token_ranges,
            sdpa_map_task_templates: buffers.sdpa_map_task_templates,
            output: buffers.partial_output,
        };
        let invocation = MapInvocation {
            pipeline: self.map.as_raw_retained(),
            dynamic_memory_bytes: self.map_dynamic_memory_bytes,
            constants: self.constants,
            shape: self.shape,
            buffers: inputs,
            partial_stats: Some((buffers.partial_exp_sums, buffers.partial_max_logits)),
            page_table_index,
            num_active_tokens,
            num_active_q_token_tiles,
            num_active_kv_splits,
        };
        invocation.validate();
        invocation
    }

    pub fn invoke_direct<'a>(
        &self,
        buffers: DirectBuffers<'a>,
        page_table_index: ReplayU32,
        num_active_tokens: ReplayU32,
        num_active_q_token_tiles: ReplayU32,
    ) -> MapInvocation<'a> {
        assert!(self.constants.direct_output);
        page_table_index.validate_range(0, self.constants.config.page_table_layout.num_gqa_layers - 1);
        num_active_tokens.validate_range(1, self.shape.num_total_tokens);
        num_active_q_token_tiles.validate_range(1, self.shape.num_total_q_token_tiles);
        num_active_q_token_tiles.validate_range(1, self.shape.num_total_sdpa_map_task_templates);
        let invocation = MapInvocation {
            pipeline: self.map.as_raw_retained(),
            dynamic_memory_bytes: self.map_dynamic_memory_bytes,
            constants: self.constants,
            shape: self.shape,
            buffers,
            partial_stats: None,
            page_table_index,
            num_active_tokens,
            num_active_q_token_tiles,
            num_active_kv_splits: num_active_q_token_tiles,
        };
        invocation.validate();
        invocation
    }

    pub fn invoke_reduce<'a>(
        &self,
        buffers: ReduceBuffers<'a>,
        num_active_q_token_tiles: ReplayU32,
    ) -> ReduceInvocation<'a> {
        num_active_q_token_tiles.validate_range(1, self.shape.num_total_q_token_tiles);
        let invocation = ReduceInvocation {
            pipeline: self
                .reduce
                .as_ref()
                .expect("direct SDPA has no Reduce stage")
                .as_raw_retained(),
            constants: self.constants,
            shape: self.shape,
            buffers,
            num_active_q_token_tiles,
        };
        invocation.validate();
        invocation
    }
}

pub struct MapInvocation<'a> {
    dynamic_memory_bytes: usize,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    constants: KernelConstants,
    shape: Shape,
    buffers: DirectBuffers<'a>,
    partial_stats: Option<(&'a Buffer, &'a Buffer)>,
    page_table_index: ReplayU32,
    num_active_tokens: ReplayU32,
    num_active_q_token_tiles: ReplayU32,
    num_active_kv_splits: ReplayU32,
}

impl MapInvocation<'_> {
    fn validate(&self) {
        let shape = self.shape;
        let constants = self.constants;
        let config = constants.config;
        assert!(self.buffers.q.len_bytes_u64() >= config.q_bytes(shape));
        assert!(self.buffers.kv_pages.len_bytes() >= config.page_bytes as usize);
        assert!(self.buffers.req_slots.len_bytes() >= shape.num_total_tokens as usize * size_of::<u32>());
        assert!(self.buffers.page_ids.len_bytes() >= config.page_table_layout.bytes());
        assert!(
            self.buffers.visible_kv_token_ranges.len_bytes()
                >= shape.num_visible_kv_token_range_values() * size_of::<u32>()
        );
        assert!(self.buffers.q_token_ranges.len_bytes() >= shape.num_q_token_range_values() * size_of::<u32>());
        assert!(
            self.buffers.sdpa_map_task_templates.len_bytes()
                >= shape.num_sdpa_map_task_template_values() * size_of::<u32>()
        );
        if let Some((sums, maxs)) = self.partial_stats {
            assert!(self.buffers.output.len_bytes_u64() >= constants.partial_output_bytes(shape));
            assert!(sums.len_bytes_u64() >= constants.partial_output_stats_bytes(shape));
            assert!(maxs.len_bytes_u64() >= constants.partial_output_stats_bytes(shape));
        } else {
            assert!(self.buffers.output.len_bytes_u64() >= config.q_bytes(shape));
        }
    }
}

impl Operator for MapInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let shape = self.shape;
        let constants = self.constants;
        let config = constants.config;
        recorder.set_retained_pipeline_state(&self.pipeline);
        recorder.set_buffer_read(0, self.buffers.q, 0);
        recorder.set_buffer_read(1, self.buffers.kv_pages, 0);
        recorder.set_buffer_read(2, self.buffers.req_slots, 0);
        recorder.set_buffer_read(3, self.buffers.page_ids, 0);
        recorder.set_buffer_read(4, self.buffers.visible_kv_token_ranges, 0);
        recorder.set_buffer_read(5, self.buffers.q_token_ranges, 0);
        recorder.set_buffer_read(6, self.buffers.sdpa_map_task_templates, 0);
        recorder.set_buffer_write(7, self.buffers.output, 0);
        if let Some((sums, maxs)) = self.partial_stats {
            recorder.set_buffer_write(8, sums, 0);
            recorder.set_buffer_write(9, maxs, 0);
        }
        let max_page_table_index = config.page_table_layout.num_gqa_layers - 1;
        match self.page_table_index {
            ReplayU32::Fixed(page_table_index) => recorder.set_u32(10, page_table_index),
            ReplayU32::Parameter(key) => recorder.bind_u32(10, key, 0, max_page_table_index),
        }
        set_replay_u32(
            recorder,
            11,
            self.num_active_q_token_tiles,
            shape.num_total_q_token_tiles,
        );
        set_replay_u32(
            recorder,
            12,
            self.num_active_kv_splits,
            shape.num_total_sdpa_map_task_templates,
        );
        set_replay_u32(recorder, 13, self.num_active_tokens, shape.num_total_tokens);
        recorder.set_threadblock_memory_length(0, self.dynamic_memory_bytes);
        recorder.dispatch_threadblocks(
            (
                constants.num_head_groups(),
                shape.num_total_sdpa_map_task_templates as usize,
                1,
            ),
            (constants.map.thread_block.required_threads as usize, 1, 1),
        );
    }
}

pub struct ReduceInvocation<'a> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    constants: KernelConstants,
    shape: Shape,
    buffers: ReduceBuffers<'a>,
    num_active_q_token_tiles: ReplayU32,
}

impl ReduceInvocation<'_> {
    fn validate(&self) {
        let shape = self.shape;
        let constants = self.constants;
        let config = constants.config;
        assert!(self.buffers.partial_output.len_bytes_u64() >= constants.partial_output_bytes(shape));
        assert!(self.buffers.partial_exp_sums.len_bytes_u64() >= constants.partial_output_stats_bytes(shape));
        assert!(self.buffers.partial_max_logits.len_bytes_u64() >= constants.partial_output_stats_bytes(shape));
        assert!(self.buffers.q_token_ranges.len_bytes() >= shape.num_q_token_range_values() * size_of::<u32>());
        assert!(
            self.buffers.cu_sdpa_partial_outputs.len_bytes()
                >= shape.num_cu_sdpa_partial_output_values() * size_of::<u32>()
        );
        assert!(self.buffers.output.len_bytes_u64() >= config.q_bytes(shape));
    }
}

impl Operator for ReduceInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let shape = self.shape;
        let constants = self.constants;
        let config = constants.config;
        recorder.set_retained_pipeline_state(&self.pipeline);
        recorder.set_buffer_read(0, self.buffers.partial_output, 0);
        recorder.set_buffer_read(1, self.buffers.partial_exp_sums, 0);
        recorder.set_buffer_read(2, self.buffers.partial_max_logits, 0);
        recorder.set_buffer_read(3, self.buffers.q_token_ranges, 0);
        recorder.set_buffer_read(4, self.buffers.cu_sdpa_partial_outputs, 0);
        recorder.set_buffer_write(5, self.buffers.output, 0);
        set_replay_u32(
            recorder,
            6,
            self.num_active_q_token_tiles,
            shape.num_total_q_token_tiles,
        );
        recorder.dispatch_threadblocks(
            (config.num_q_heads as usize, shape.num_total_q_token_tiles as usize, 1),
            (constants.reduce.thread_block.required_threads as usize, 1, 1),
        );
    }
}

fn source(constants: KernelConstants) -> String {
    let config = constants.config;
    let map = constants.map.thread_block;
    let reduce = constants.reduce.thread_block;
    format!(
        r#"
#define Q_TOKEN_HEAD_LAYOUT {q_token_head_layout}
#define KV_DIM_TILE {kv_dim_tile}
#define DIRECT_OUTPUT {direct_output}
#define NUM_Q_HEADS {num_q_heads}
#define NUM_KV_HEADS {num_kv_heads}
#define MAX_Q_HEADS {max_q_heads}
#define NUM_Q_HEAD_RANGES_PER_KV_HEAD {num_q_head_ranges_per_kv_head}
#define HEAD_DIM {head_dim}
#define ATTENTION_SCALE {scale}
#define PAGE_BYTES {page_bytes}
#define NUM_TOKENS_PER_PAGE {num_tokens_per_page}
#define NUM_GQA_LAYERS {num_gqa_layers}
#define NUM_BLOCKS {num_blocks}
#define NUM_PAGE_IDS_PER_BLOCK {num_page_ids_per_block}
#define MAX_Q_TOKENS {max_q_tokens}
#define KV_TOKENS_PER_ITERATION {kv_tokens_per_iteration}
#define MAP_REQUIRED_THREADS {map_required_threads}
#define REDUCE_REQUIRED_THREADS {reduce_required_threads}
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
"#,
        q_token_head_layout = match constants.map.kind {
            sdpa::MapKernelKind::QTokensPerHead => 0,
            sdpa::MapKernelKind::QHeadsPerToken => 1,
            sdpa::MapKernelKind::QTokensAndHeads => 2,
            sdpa::MapKernelKind::SlicedKv => unreachable!(),
        },
        kv_dim_tile = constants.kv_dim_tile,
        direct_output = u32::from(constants.direct_output),
        num_q_heads = config.num_q_heads,
        num_kv_heads = config.num_kv_heads,
        max_q_heads = map.max_q_heads,
        num_q_head_ranges_per_kv_head = constants.num_q_head_ranges_per_kv_head(),
        head_dim = config.head_dim,
        scale = config.scale,
        page_bytes = config.page_bytes,
        num_tokens_per_page = config.num_tokens_per_page(),
        num_gqa_layers = config.page_table_layout.num_gqa_layers,
        num_blocks = config.page_table_layout.num_blocks,
        num_page_ids_per_block = config.page_table_layout.num_page_ids_per_block,
        max_q_tokens = map.max_q_tokens,
        kv_tokens_per_iteration = map.kv_tokens_per_iteration,
        map_required_threads = map.required_threads,
        reduce_required_threads = reduce.required_threads,
    )
}

fn set_replay_u32(recorder: &CommandRecorder<'_>, index: usize, value: ReplayU32, max_value: u32) {
    match value {
        ReplayU32::Fixed(value) => recorder.set_u32(index, value),
        ReplayU32::Parameter(key) => recorder.bind_u32(index, key, 1, max_value),
    }
}

#[cfg(test)]
#[path = "sliced_q_test.rs"]
mod tests;
