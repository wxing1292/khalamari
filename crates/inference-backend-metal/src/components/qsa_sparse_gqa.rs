use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;

const FP8_SOURCE: &str = include_str!("metal/gqa_fp8.metal");
const MAP_SOURCE: &str = include_str!("metal/qsa_sparse_gqa.metal");

// Sparse map for the production QSA SDPA path.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub page_bytes: u32,
    pub page_table_layout: crate::components::gqa::kv_page_write::PageTableLayout,
    pub compress_ratio: u32,
    pub top_k: u32,
}

impl Config {
    pub fn validate(self) {
        assert_eq!(self.num_q_heads, 24);
        assert_eq!(self.num_kv_heads, 2);
        assert_eq!(self.head_dim, 256);
        assert!(self.scale.is_finite() && self.scale > 0.0);
        assert_eq!(self.page_bytes, 32_768);
        assert_eq!(self.compress_ratio, 4);
        assert_eq!(self.top_k, 512);
        self.page_table_layout.validate();
        assert_eq!(self.tokens_per_page(), 32);
        self.page_table_layout
            .num_blocks
            .checked_mul(self.page_table_layout.num_page_ids_per_block)
            .and_then(|pages| pages.checked_mul(self.tokens_per_page()))
            .expect("QSA sparse page-table token capacity must fit u32");
    }

    pub fn tokens_per_page(self) -> u32 {
        let kv_bytes_per_token = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .and_then(|count| count.checked_mul(2))
            .expect("QSA sparse K/V bytes per token must fit u32");
        assert!(self.page_bytes.is_multiple_of(kv_bytes_per_token));
        self.page_bytes / kv_bytes_per_token
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_tokens: u32,
    pub num_total_sparse_queries: u32,
}

impl Shape {
    pub fn validate(self) {
        assert!(self.num_total_tokens > 0);
        assert!(self.num_total_sparse_queries > 0);
        assert!(self.num_total_sparse_queries <= self.num_total_tokens);
        crate::components::assert_u32_index_domain(
            crate::components::checked_product(
                "QSA selected-group element count",
                &[self.num_total_tokens as usize, 512],
            ),
            "QSA selected-group indexing",
        );
    }

    fn num_total_map_tasks(self, num_splits: u32) -> usize {
        (self.num_total_sparse_queries * num_splits) as usize
    }
}

struct Selector;

impl Selector {
    const QUERY_CAPACITIES: [(u32, u32); 6] = [(2, 64), (4, 32), (8, 16), (64, 8), (256, 4), (512, 2)];

    fn select(num_total_sparse_queries: u32) -> u32 {
        assert!(
            num_total_sparse_queries > 0,
            "QSA sparse-query capacity must be positive"
        );
        for (capacity, num_splits) in Self::QUERY_CAPACITIES {
            if num_total_sparse_queries <= capacity {
                return num_splits;
            }
        }
        1
    }
}

#[derive(Clone, Copy)]
pub enum MapDestination<'a> {
    Direct(&'a Buffer),
    Partials {
        exp_sums: &'a Buffer,
        max_logits: &'a Buffer,
        output: &'a Buffer,
    },
}

#[derive(Clone, Copy)]
pub struct MapBuffers<'a> {
    pub q: &'a Buffer,
    pub pages: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub selected_groups: &'a Buffer,
    pub query_positions: &'a Buffer,
    pub sparse_query_rows: &'a Buffer,
    pub destination: MapDestination<'a>,
}

/// Sparse QSA Map. S=1 writes normalized BF16 output directly. S>1 writes
/// SplitKV partials for the shared SlicedKv reducer.
///
/// ```text
/// Q [Tq,24,256] + group IDs [Tq,512] + tail [0..3 tokens]
///   -> indexed K/V tiles [Tkv_tile,256] for one KV head
///   -> online softmax -> output [Tq,24,256]
/// Task: { sparse query, KV split } from grid.x, KV head from grid.y.
/// Grid: (num_total_sparse_queries * S, 2, 1), 64 threads per threadblock.
/// The two SIMDgroups share K/V loads for all 12 Q heads (padded to 16).
/// S=1 writes final BF16 rows. S>1 writes F32 partials and softmax statistics.
/// ```
pub struct Compute {
    config: Config,
    shape: Shape,
    num_splits: u32,
    map: CompiledKernel,
}

impl Compute {
    pub fn selected_split_count(num_total_sparse_queries: u32) -> u32 {
        Selector::select(num_total_sparse_queries)
    }

    /// Exclusive upper bounds where the sparse-query capacity changes SplitKV topology.
    pub fn split_topology_boundaries() -> Box<[u32]> {
        Selector::QUERY_CAPACITIES.map(|(capacity, _)| capacity + 1).into()
    }

    pub fn new(device: &Device, config: Config, shape: Shape, num_splits: u32) -> Self {
        config.validate();
        shape.validate();
        assert!(num_splits > 0);
        assert!(num_splits <= config.top_k * config.compress_ratio);
        let tasks = shape
            .num_total_sparse_queries
            .checked_mul(num_splits)
            .expect("QSA sparse map task count must fit u32") as usize;
        crate::components::assert_u32_index_domain(
            crate::components::checked_product(
                "QSA sparse partial-stat element count",
                &[tasks, config.num_q_heads as usize],
            ),
            "QSA sparse partial stats",
        );
        crate::components::assert_u32_count_domain(
            checked_product("QSA sparse map thread count", &[tasks, 2, 64]),
            "QSA sparse map thread count",
        );
        let map = CompiledKernel::new(device, &production_map_source(config, num_splits), "qsa_sparse_gqa_map");
        assert_eq!(map.thread_execution_width(), 32);
        Self {
            config,
            shape,
            num_splits,
            map,
        }
    }

    pub fn map<'a>(
        &self,
        buffers: MapBuffers<'a>,
        num_active_sparse_queries: crate::metal::ReplayU32,
        page_table_index: crate::metal::ReplayU32,
    ) -> MapInvocation<'a> {
        MapInvocation {
            pipeline: self.map.as_raw_retained(),
            config: self.config,
            shape: self.shape,
            num_splits: self.num_splits,
            buffers,
            num_active_sparse_queries,
            page_table_index,
        }
    }
}

pub struct MapInvocation<'a> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    config: Config,
    shape: Shape,
    num_splits: u32,
    buffers: MapBuffers<'a>,
    num_active_sparse_queries: crate::metal::ReplayU32,
    page_table_index: crate::metal::ReplayU32,
}

impl Operator for MapInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        let layout = self.config.page_table_layout;
        recorder.set_retained_pipeline_state(&self.pipeline);
        recorder.set_buffer_read(0, self.buffers.q, 0);
        recorder.set_buffer_read(1, self.buffers.pages, 0);
        recorder.set_buffer_read(2, self.buffers.req_slots, 0);
        recorder.set_buffer_read(3, self.buffers.page_ids, 0);
        recorder.set_buffer_read(4, self.buffers.selected_groups, 0);
        recorder.set_buffer_read(5, self.buffers.query_positions, 0);
        recorder.set_buffer_read(6, self.buffers.sparse_query_rows, 0);
        if self.num_splits == 1 {
            let MapDestination::Direct(output) = self.buffers.destination else {
                panic!("QSA S=1 map requires direct BF16 output");
            };
            recorder.set_buffer_write(7, output, 0);
            set_replay_u32_min(
                recorder,
                8,
                self.num_active_sparse_queries,
                0,
                self.shape.num_total_sparse_queries,
                "QSA active sparse query count",
            );
            set_replay_u32_min(
                recorder,
                9,
                self.page_table_index,
                0,
                layout.num_gqa_layers - 1,
                "QSA page-table layer index",
            );
        } else {
            let MapDestination::Partials {
                exp_sums,
                max_logits,
                output,
            } = self.buffers.destination
            else {
                panic!("QSA S>1 map requires SplitKV partial buffers");
            };
            recorder.set_buffer_write(7, exp_sums, 0);
            recorder.set_buffer_write(8, max_logits, 0);
            recorder.set_buffer_write(9, output, 0);
            set_replay_u32_min(
                recorder,
                10,
                self.num_active_sparse_queries,
                0,
                self.shape.num_total_sparse_queries,
                "QSA active sparse query count",
            );
            set_replay_u32_min(
                recorder,
                11,
                self.page_table_index,
                0,
                layout.num_gqa_layers - 1,
                "QSA page-table layer index",
            );
        }
        recorder.dispatch_threadblocks((self.shape.num_total_map_tasks(self.num_splits), 2, 1), (64, 1, 1));
    }
}

impl MapInvocation<'_> {
    fn validate(&self) {
        let config = self.config;
        let shape = self.shape;
        let tasks = shape.num_total_map_tasks(self.num_splits);
        let require = |buffer: &Buffer, bytes: usize, name: &str| {
            assert!(buffer.len_bytes() >= bytes, "QSA sparse {name} buffer is too small");
        };
        let tensor_bytes = crate::components::checked_product(
            "QSA sparse query/output bytes",
            &[
                shape.num_total_tokens as usize,
                config.num_q_heads as usize,
                config.head_dim as usize,
                Dtype::Bfloat16.item_size(),
            ],
        );
        require(self.buffers.q, tensor_bytes, "Q");
        require(self.buffers.pages, config.page_bytes as usize, "KV pages");
        require(
            self.buffers.req_slots,
            shape.num_total_tokens as usize * size_of::<u32>(),
            "request slots",
        );
        require(self.buffers.page_ids, config.page_table_layout.bytes(), "page IDs");
        require(
            self.buffers.selected_groups,
            shape.num_total_tokens as usize * config.top_k as usize * size_of::<u32>(),
            "selected groups",
        );
        require(
            self.buffers.query_positions,
            shape.num_total_tokens as usize * size_of::<u32>(),
            "query positions",
        );
        require(
            self.buffers.sparse_query_rows,
            shape.num_total_sparse_queries as usize * size_of::<u32>(),
            "sparse query rows",
        );
        match (self.num_splits == 1, self.buffers.destination) {
            (true, MapDestination::Direct(output)) => require(output, tensor_bytes, "output"),
            (
                false,
                MapDestination::Partials {
                    exp_sums,
                    max_logits,
                    output,
                },
            ) => {
                let stats_bytes = tasks * config.num_q_heads as usize * size_of::<f32>();
                require(exp_sums, stats_bytes, "partial exp sums");
                require(max_logits, stats_bytes, "partial max logits");
                require(output, stats_bytes * config.head_dim as usize, "partial output");
            },
            (true, MapDestination::Partials { .. }) => panic!("QSA S=1 map cannot write partial buffers"),
            (false, MapDestination::Direct(_)) => panic!("QSA S>1 map cannot write final output"),
        }
    }
}

fn set_replay_u32_min(
    recorder: &CommandRecorder<'_>,
    index: usize,
    value: crate::metal::ReplayU32,
    min_value: u32,
    max_value: u32,
    name: &str,
) {
    match value {
        crate::metal::ReplayU32::Fixed(value) => {
            assert!(value >= min_value, "{name} is below its minimum");
            assert!(value <= max_value, "{name} exceeds recorded capacity");
            recorder.set_u32(index, value);
        },
        crate::metal::ReplayU32::Parameter(key) => recorder.bind_u32(index, key, min_value, max_value),
    }
}

fn production_map_source(config: Config, num_splits: u32) -> String {
    let layout = config.page_table_layout;
    let token_tile = if num_splits == 1 { 32 } else { 16 };
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}\n#define QSA_TOKEN_TILE {}\n#define QSA_NUM_SPLITS \
         {}\n#define QSA_DIRECT_OUTPUT {}\n#define QSA_ATTENTION_SCALE {:.9e}f\n#define QSA_NUM_PAGE_IDS_PER_BLOCK \
         {}\n#define QSA_NUM_BLOCKS {}\n#define QSA_NUM_GQA_LAYERS {}\n{}\n{}",
        crate::mlx_headers::steel_mma_source(),
        token_tile,
        num_splits,
        u8::from(num_splits == 1),
        config.scale * std::f32::consts::LOG2_E,
        layout.num_page_ids_per_block,
        layout.num_blocks,
        layout.num_gqa_layers,
        FP8_SOURCE,
        MAP_SOURCE,
    )
}
