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
const SIMD_SOURCE: &str = include_str!("metal/qsa_sparse_gqa_simd.metal");
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KernelKind {
    Simd { num_splits: u32, num_simdgroups: u32 },
    Mma { num_splits: u32 },
}

impl KernelKind {
    fn num_splits(self) -> u32 {
        match self {
            Self::Simd { num_splits, .. } | Self::Mma { num_splits } => num_splits,
        }
    }

    fn required_threads(self) -> u32 {
        match self {
            Self::Simd { num_simdgroups, .. } => num_simdgroups * 32,
            Self::Mma { .. } => 64,
        }
    }

    fn num_head_tiles(self) -> u32 {
        match self {
            Self::Simd { .. } => 8,
            Self::Mma { .. } => 2,
        }
    }
}

struct Selector;

impl Selector {
    const QUERY_CAPACITIES: [(u32, KernelKind); 7] = [
        (
            1,
            KernelKind::Simd {
                num_splits: 8,
                num_simdgroups: 8,
            },
        ),
        (
            2,
            KernelKind::Simd {
                num_splits: 32,
                num_simdgroups: 2,
            },
        ),
        (4, KernelKind::Mma { num_splits: 32 }),
        (8, KernelKind::Mma { num_splits: 16 }),
        (64, KernelKind::Mma { num_splits: 8 }),
        (256, KernelKind::Mma { num_splits: 4 }),
        (512, KernelKind::Mma { num_splits: 2 }),
    ];

    fn key(num_total_sparse_queries: u32) -> KernelKind {
        assert!(
            num_total_sparse_queries > 0,
            "QSA sparse-query capacity must be positive"
        );
        Self::QUERY_CAPACITIES
            .iter()
            .find(|(capacity, _)| num_total_sparse_queries <= *capacity)
            .map_or(KernelKind::Mma { num_splits: 1 }, |&(_, kind)| kind)
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

/// Indexed FP8 QSA Map with shape-selected SIMD or register-Q MMA.
/// Both paths write F32 SplitKV partials for the shared SlicedKv reducer.
/// The unsplit MMA path writes final BF16 output directly.
///
/// ```text
/// Q [Tq,24,256] BF16 + group IDs [Tq,512] U32 + tail [0..3 tokens]
///     recorded sparse-query capacity -> Selector -> one Map pipeline
///       <=2: SIMD, 3 Q heads/warp; Q/O [3,8] F32 registers per lane
///             indexed K/V -> registers -> local warp partials -> SMEM merge
///        >2: MMA, 12 active Q heads/threadblock; Q stays in registers
///             indexed K/V -> shared BF16 SMEM -> matrix fragments
///     -> F32 partials + max/sum -> shared Reduce -> O [Tq,24,256] BF16
/// Grid: [sparse-query capacity * splits, head tiles], threads = warps * 32.
/// Submission supplies the active sparse-query count. Inactive tasks return.
/// ```
pub struct Compute {
    config: Config,
    shape: Shape,
    kind: KernelKind,
    map: CompiledKernel,
}

impl Compute {
    pub fn selected_split_count(num_total_sparse_queries: u32) -> u32 {
        Selector::key(num_total_sparse_queries).num_splits()
    }

    /// First capacities where the kernel or its SplitKV topology changes.
    pub fn topology_boundaries() -> Box<[u32]> {
        Selector::QUERY_CAPACITIES.map(|(capacity, _)| capacity + 1).into()
    }

    pub fn new(device: &Device, config: Config, shape: Shape) -> Self {
        config.validate();
        shape.validate();
        let kind = Selector::key(shape.num_total_sparse_queries);
        let num_splits = kind.num_splits();
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
            checked_product(
                "QSA sparse map thread count",
                &[tasks, kind.num_head_tiles() as usize, kind.required_threads() as usize],
            ),
            "QSA sparse map thread count",
        );
        let map = CompiledKernel::new(device, &production_map_source(config, kind), "qsa_sparse_gqa_map");
        assert_eq!(map.thread_execution_width(), 32);
        assert!(kind.required_threads() as usize <= map.max_total_threads_per_threadblock());
        assert!(map.static_threadblock_memory_length() <= device.max_threadblock_memory_length());
        Self {
            config,
            shape,
            kind,
            map,
        }
    }

    pub fn map<'a>(
        &self,
        buffers: MapBuffers<'a>,
        num_active_sparse_queries: crate::metal::ReplayU32,
        page_table_index: crate::metal::ReplayU32,
    ) -> MapInvocation<'a> {
        let invocation = MapInvocation {
            pipeline: self.map.as_raw_retained(),
            config: self.config,
            shape: self.shape,
            kind: self.kind,
            buffers,
            num_active_sparse_queries,
            page_table_index,
        };
        invocation.validate();
        invocation
    }
}

pub struct MapInvocation<'a> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    config: Config,
    shape: Shape,
    kind: KernelKind,
    buffers: MapBuffers<'a>,
    num_active_sparse_queries: crate::metal::ReplayU32,
    page_table_index: crate::metal::ReplayU32,
}

impl Operator for MapInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let layout = self.config.page_table_layout;
        recorder.set_retained_pipeline_state(&self.pipeline);
        recorder.set_buffer_read(0, self.buffers.q, 0);
        recorder.set_buffer_read(1, self.buffers.pages, 0);
        recorder.set_buffer_read(2, self.buffers.req_slots, 0);
        recorder.set_buffer_read(3, self.buffers.page_ids, 0);
        recorder.set_buffer_read(4, self.buffers.selected_groups, 0);
        recorder.set_buffer_read(5, self.buffers.query_positions, 0);
        recorder.set_buffer_read(6, self.buffers.sparse_query_rows, 0);
        if self.kind.num_splits() == 1 {
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
            );
            set_replay_u32_min(recorder, 9, self.page_table_index, 0, layout.num_gqa_layers - 1);
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
            );
            set_replay_u32_min(recorder, 11, self.page_table_index, 0, layout.num_gqa_layers - 1);
        }
        recorder.dispatch_threadblocks(
            (
                self.shape.num_total_map_tasks(self.kind.num_splits()),
                self.kind.num_head_tiles() as usize,
                1,
            ),
            (self.kind.required_threads() as usize, 1, 1),
        );
    }
}

impl MapInvocation<'_> {
    fn validate(&self) {
        let config = self.config;
        let shape = self.shape;
        self.num_active_sparse_queries
            .validate_range(0, shape.num_total_sparse_queries);
        self.page_table_index
            .validate_range(0, config.page_table_layout.num_gqa_layers - 1);
        let tasks = shape.num_total_map_tasks(self.kind.num_splits());
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
        match (self.kind.num_splits() == 1, self.buffers.destination) {
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
) {
    match value {
        crate::metal::ReplayU32::Fixed(value) => recorder.set_u32(index, value),
        crate::metal::ReplayU32::Parameter(key) => recorder.bind_u32(index, key, min_value, max_value),
    }
}

fn production_map_source(config: Config, kind: KernelKind) -> String {
    let layout = config.page_table_layout;
    let num_splits = kind.num_splits();
    let common = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n#define QSA_NUM_SPLITS {num_splits}\n#define \
         QSA_DIRECT_OUTPUT {}\n#define QSA_NUM_PAGE_IDS_PER_BLOCK {}\n#define QSA_NUM_BLOCKS {}\n#define \
         QSA_NUM_GQA_LAYERS {}\n",
        u8::from(num_splits == 1),
        layout.num_page_ids_per_block,
        layout.num_blocks,
        layout.num_gqa_layers,
    );
    match kind {
        KernelKind::Simd { num_simdgroups, .. } => {
            let scale = config.scale * 256.0 * std::f32::consts::LOG2_E;
            format!(
                "{common}\n#define QSA_NUM_SIMDGROUPS {num_simdgroups}\n#define QSA_ATTENTION_SCALE \
                 {scale:.9e}f\n{SIMD_SOURCE}"
            )
        },
        KernelKind::Mma { .. } => {
            let token_tile = 16;
            format!(
                "{common}\n{}\n#define QSA_TOKEN_TILE {token_tile}\n#define QSA_ATTENTION_SCALE \
                 {:.9e}f\n{FP8_SOURCE}\n{MAP_SOURCE}",
                crate::mlx_headers::steel_mma_source(),
                config.scale * std::f32::consts::LOG2_E
            )
        },
    }
}

#[cfg(test)]
#[path = "qsa_sparse_gqa_test.rs"]
mod tests;
