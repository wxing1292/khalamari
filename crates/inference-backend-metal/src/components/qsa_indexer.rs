//! Pooled-key preparation, paged scoring, and exact group top-k.

use std::collections::HashSet;

use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("metal/qsa_indexer.metal");
const THREADS: usize = 128;
const SELECTION_THREADS: usize = 256;
const SAMPLED_SELECTION_THREADS: usize = 1024;
const SCORE_TILE_GROUPS: u32 = 64;
const FLAT_SCORE_QUERIES: u32 = 8;
const HEAD_LOOP_SCORE_QUERIES: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub num_q_heads: u32,
    pub head_dim: u32,
    pub compress_ratio: u32,
    pub top_k: u32,
    pub rope_dim: u32,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub dtype: Dtype,
}

impl Config {
    pub fn validate(self) {
        assert!(self.num_q_heads > 0 && self.head_dim > 0);
        assert!(self.compress_ratio > 1 && self.top_k > 0);
        assert!(self.rope_dim > 0 && self.rope_dim <= self.head_dim && self.rope_dim.is_multiple_of(2));
        assert!(self.rope_theta.is_finite() && self.rope_theta > 0.0);
        assert!(self.norm_eps.is_finite() && self.norm_eps > 0.0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
        self.num_q_heads
            .checked_add(1)
            .and_then(|heads| heads.checked_mul(self.head_dim))
            .expect("QSA indexer projection width must fit u32");
        self.compress_ratio
            .checked_mul(self.head_dim)
            .expect("QSA indexer group width must fit u32");
    }
    fn qk_dim(self) -> u32 {
        (self.num_q_heads + 1) * self.head_dim
    }
    fn state_dim(self) -> u32 {
        (self.compress_ratio - 1) * self.head_dim
    }
}

/// Metadata for one projected token. Token positions and group IDs are request-local.
/// `batch_start` addresses flattened query rows. Slot IDs address their respective storage domains.
/// Source state is the incomplete group at the start of this request's batch segment.
/// Active destination slots must be unique and disjoint from every active source slot.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Token {
    pub batch_start: u32,
    pub token_index: u32,
    pub req_slot: u32,
    pub src_state_slot: u32,
    pub dst_state_slot: u32,
    /// Visible complete groups: `[group_start, (token_index + 1) / compress_ratio)`.
    pub group_start: u32,
}

impl Token {
    pub fn to_array(self) -> [u32; 6] {
        [
            self.batch_start,
            self.token_index,
            self.req_slot,
            self.src_state_slot,
            self.dst_state_slot,
            self.group_start,
        ]
    }
}

/// One request-local range of consecutive sparse queries.
///
/// The metadata owner must cover every sparse query exactly once. A task must not include dense rows or cross a
/// request boundary. The scorer geometry sets the maximum query-template width to one, eight, or thirty-two.
/// The final task can be shorter.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ScoreTaskTemplate {
    pub query_start: u32,
    pub num_queries: u32,
}

impl ScoreTaskTemplate {
    pub fn to_array(self) -> [u32; 2] {
        [self.query_start, self.num_queries]
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CacheLayout {
    pub num_req_slots: u32,
    pub num_pages_per_request: u32,
    pub num_state_slots: u32,
    pub page_bytes: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Shape {
    pub num_total_tokens: u32,
    /// Sparse-query capacity. Dense-prefix tokens do not contribute to score work.
    pub num_total_sparse_queries: u32,
    /// Maximum sparse-query count in one request segment.
    pub num_total_request_sparse_queries: u32,
    /// Score columns and dispatch capacity, relative to each query's `group_start`.
    pub num_total_groups: u32,
    /// Capacity-sized metadata worklist. Active tasks are a submission value.
    pub num_total_score_task_templates: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScoreGeometry {
    Generic,
    Simd { groups_per_simd: u32 },
    Steel { records_head_loop: bool },
}

impl ScoreGeometry {
    pub fn for_shape(num_q_heads: u32, head_dim: u32, dtype: Dtype, shape: Shape, layout: CacheLayout) -> Self {
        assert!(shape.num_total_sparse_queries > 0 && shape.num_total_sparse_queries <= shape.num_total_tokens);
        assert!(shape.num_total_groups > 0);
        assert!(head_dim > 0);
        assert!(
            shape.num_total_request_sparse_queries > 0
                && shape.num_total_request_sparse_queries <= shape.num_total_sparse_queries
        );
        let key_bytes = head_dim as usize * dtype.item_size();
        let groups_per_page = layout.page_bytes as usize / key_bytes;
        if dtype != Dtype::Bfloat16 || num_q_heads != 4 || head_dim != 128 || !groups_per_page.is_multiple_of(64) {
            return Self::Generic;
        }
        if shape.num_total_request_sparse_queries == 1
            || shape.num_total_sparse_queries <= 4
            || (shape.num_total_groups < 2048
                && (shape.num_total_sparse_queries <= 8 || shape.num_total_request_sparse_queries <= 2))
        {
            if !groups_per_page.is_power_of_two() {
                return Self::Generic;
            }
            let groups_per_simd = if shape.num_total_groups < 4096 {
                1
            } else {
                (1 << (shape.num_total_groups / 2048).ilog2()).min(8)
            };
            return Self::Simd { groups_per_simd };
        }
        let records_head_loop = shape.num_total_request_sparse_queries >= HEAD_LOOP_SCORE_QUERIES;
        Self::Steel { records_head_loop }
    }

    pub fn num_queries_per_task(self, request_sparse_queries: u32) -> u32 {
        match self {
            Self::Generic => 1,
            Self::Simd { .. } => FLAT_SCORE_QUERIES,
            Self::Steel {
                records_head_loop: true,
            } if request_sparse_queries >= HEAD_LOOP_SCORE_QUERIES => HEAD_LOOP_SCORE_QUERIES,
            Self::Steel { .. } => FLAT_SCORE_QUERIES,
        }
    }
}

#[derive(Clone, Copy)]
pub struct PrepareBuffers<'a> {
    /// `[T, (Hq + 1) * D]`, projected before this operation.
    pub qk: &'a Buffer,
    pub tokens: &'a Buffer,
    /// Raw Gemma affine offsets, `[D]` each.
    pub q_norm: &'a Buffer,
    pub k_norm: &'a Buffer,
    /// One layer's page pool and `[num_req_slots, num_pages_per_request]` page table.
    pub pages: &'a Buffer,
    pub page_ids: &'a Buffer,
    /// `[num_state_slots, compress_ratio - 1, D]`, with caller-owned versions.
    pub raw_states: &'a Buffer,
    /// Normalized and rotated index queries, `[T, Hq, D]`.
    pub q: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct SelectBuffers<'a> {
    pub q: &'a Buffer,
    pub tokens: &'a Buffer,
    pub score_task_templates: &'a Buffer,
    pub pages: &'a Buffer,
    pub page_ids: &'a Buffer,
    /// `[T, top_k]` logical group IDs. Only sparse query rows are written.
    /// Each sparse row has exactly `top_k` distinct, visible IDs in ascending group order.
    pub group_ids: &'a Buffer,
}

pub struct Compute {
    config: Config,
    prepare: CompiledKernel,
    score: CompiledKernel,
    vector_score: Option<[[CompiledKernel; 4]; 2]>,
    flat_score: Option<CompiledKernel>,
    head_loop_score: Option<CompiledKernel>,
    select: CompiledKernel,
    sampled_select: Option<CompiledKernel>,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        let norm_bytes = (config.head_dim as usize + THREADS / 32) * 4;
        let sampled_bytes = if config.top_k == 512 {
            2 * 2048 * 4 + 256 * 4 + 66 * 4
        } else {
            0
        };
        assert!(
            norm_bytes
                .max(sampled_bytes)
                .max((256 + 2 + 2 * (SELECTION_THREADS / 32)) * 4)
                <= device.max_threadblock_memory_length(),
            "QSA indexer shared memory exceeds device capacity"
        );
        let scalar = match config.dtype {
            Dtype::Float32 => "float",
            Dtype::Bfloat16 => "bfloat",
            _ => unreachable!(),
        };
        let source = SOURCE.replacen(
            "using namespace metal;",
            &format!(
                "using namespace metal;\nusing scalar_t = {scalar};\nconstant uint heads = {}u, dim = {}u, ratio = \
                 {}u, top_k = {}u, rope_dim = {}u;\nconstant uint threads = {THREADS}u, selection_threads = \
                 {SELECTION_THREADS}u;\nconstant float norm_eps = {:.9e}f, rope_theta = {:.9e}f;",
                config.num_q_heads,
                config.head_dim,
                config.compress_ratio,
                config.top_k,
                config.rope_dim,
                config.norm_eps,
                config.rope_theta
            ),
            1,
        );
        let mut compute = Self {
            config,
            prepare: CompiledKernel::new(device, &source, "qsa_indexer_prepare"),
            score: CompiledKernel::new(device, &source, "qsa_indexer_score"),
            vector_score: None,
            flat_score: None,
            head_loop_score: None,
            select: CompiledKernel::new(device, &source, "qsa_indexer_select"),
            sampled_select: None,
        };
        if config.dtype == Dtype::Bfloat16 && config.num_q_heads == 4 && config.head_dim == 128 {
            compute.vector_score = Some(std::array::from_fn(|queries| {
                std::array::from_fn(|groups| {
                    let vector_source = format!(
                        "{source}\nconstant uint groups_per_simd = {}u, max_queries = {}u;\n{}",
                        1 << groups,
                        if queries == 0 { 1 } else { 8 },
                        include_str!("metal/qsa_indexer_vector.metal")
                    );
                    CompiledKernel::new(device, &vector_source, "qsa_indexer_vector_score")
                })
            }));
            if config.top_k == 512 {
                let sampled_source = source.replace("selection_threads = 256u", "selection_threads = 1024u");
                compute.sampled_select = Some(CompiledKernel::new(
                    device,
                    &sampled_source,
                    "qsa_indexer_sampled_select",
                ));
            }
            let steel_source = steel_score_source(&source);
            compute.flat_score = Some(CompiledKernel::new(device, &steel_source, "qsa_indexer_flat_bm32"));
            compute.head_loop_score = Some(CompiledKernel::new(device, &steel_source, "qsa_indexer_head_loop_bm32"));
        }
        compute
    }

    /// Prepares queries, complete group keys, and candidate raw-key carry.
    /// The caller owns token metadata, physical pages, and exclusive cache writes.
    pub fn prepare<'a>(
        &'a self,
        num_total_tokens: u32,
        layout: CacheLayout,
        num_active_tokens: ReplayU32,
        buffers: PrepareBuffers<'a>,
    ) -> PrepareInvocation<'a> {
        validate_tokens(num_total_tokens, num_active_tokens, buffers.tokens);
        self.validate_cache(layout, buffers.pages, buffers.page_ids);
        let config = self.config;
        let key_bytes = config.head_dim as usize * config.dtype.item_size();
        assert!(buffers.qk.len_bytes() >= tensor_bytes(num_total_tokens, config.qk_dim(), config.dtype.item_size()));
        assert!(buffers.q_norm.len_bytes() >= key_bytes && buffers.k_norm.len_bytes() >= key_bytes);
        assert!(
            buffers.raw_states.len_bytes()
                >= tensor_bytes(layout.num_state_slots, config.state_dim(), config.dtype.item_size())
        );
        assert!(
            buffers.q.len_bytes()
                >= tensor_bytes(
                    num_total_tokens,
                    config.num_q_heads * config.head_dim,
                    config.dtype.item_size()
                )
        );
        PrepareInvocation {
            compute: self,
            num_total_tokens,
            layout,
            num_active_tokens,
            buffers,
        }
    }

    /// Selects exactly `top_k` IDs for each query with more than `top_k` visible groups.
    /// Dense query rows and inactive rows leave their output and scratch unchanged.
    /// Preparation must finish before selection reads the queries and key cache.
    /// The caller must omit selection when the total group capacity fits `top_k`.
    pub fn select<'a>(
        &'a self,
        shape: Shape,
        layout: CacheLayout,
        num_active_tokens: ReplayU32,
        num_active_score_task_templates: ReplayU32,
        buffers: SelectBuffers<'a>,
        scores: &'a Buffer,
    ) -> SelectInvocation<'a> {
        validate_tokens(shape.num_total_tokens, num_active_tokens, buffers.tokens);
        assert!(
            shape.num_total_request_sparse_queries > 0
                && shape.num_total_request_sparse_queries <= shape.num_total_sparse_queries
        );
        self.validate_cache(layout, buffers.pages, buffers.page_ids);
        let config = self.config;
        assert!(
            shape.num_total_score_task_templates > 0 && shape.num_total_score_task_templates <= shape.num_total_tokens
        );
        assert!(
            buffers.score_task_templates.len_bytes()
                >= tensor_bytes(shape.num_total_score_task_templates, 1, size_of::<ScoreTaskTemplate>())
        );
        let groups_per_page = layout.page_bytes as usize / (config.head_dim as usize * config.dtype.item_size());
        assert!(shape.num_total_groups > config.top_k);
        assert!(shape.num_total_groups as u64 <= layout.num_pages_per_request as u64 * groups_per_page as u64);
        if let ReplayU32::Fixed(active) = num_active_score_task_templates {
            assert!(active <= shape.num_total_score_task_templates);
        }
        let max_aligned_tiles = aligned_score_tile_count(shape.num_total_groups, SCORE_TILE_GROUPS);
        assert!(
            u64::from(max_aligned_tiles) * u64::from(SCORE_TILE_GROUPS) <= u64::from(u32::MAX),
            "QSA indexer aligned group span must fit u32"
        );
        assert!(
            buffers.q.len_bytes()
                >= tensor_bytes(
                    shape.num_total_tokens,
                    config.num_q_heads * config.head_dim,
                    config.dtype.item_size()
                )
        );
        assert!(buffers.group_ids.len_bytes() >= tensor_bytes(shape.num_total_tokens, config.top_k, 4));
        assert!(scores.len_bytes() >= tensor_bytes(shape.num_total_tokens, shape.num_total_groups, 4));
        SelectInvocation {
            compute: self,
            shape,
            layout,
            num_active_tokens,
            num_active_score_task_templates,
            geometry: ScoreGeometry::for_shape(config.num_q_heads, config.head_dim, config.dtype, shape, layout),
            buffers,
            scores,
        }
    }

    fn validate_cache(&self, layout: CacheLayout, pages: &Buffer, page_ids: &Buffer) {
        assert!(layout.num_req_slots > 0 && layout.num_pages_per_request > 0 && layout.num_state_slots > 0);
        let key_bytes = self.config.head_dim as usize * self.config.dtype.item_size();
        assert!(layout.page_bytes as usize >= key_bytes && (layout.page_bytes as usize).is_multiple_of(key_bytes));
        assert!(pages.len_bytes().is_multiple_of(layout.page_bytes as usize));
        let groups_per_page = layout.page_bytes as usize / key_bytes;
        assert!(layout.num_pages_per_request as u64 * groups_per_page as u64 <= u32::MAX as u64);
        assert!(page_ids.len_bytes() >= tensor_bytes(layout.num_req_slots, layout.num_pages_per_request, 4));
    }
}

pub struct PrepareInvocation<'a> {
    compute: &'a Compute,
    num_total_tokens: u32,
    layout: CacheLayout,
    num_active_tokens: ReplayU32,
    buffers: PrepareBuffers<'a>,
}

pub struct SelectInvocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    layout: CacheLayout,
    num_active_tokens: ReplayU32,
    num_active_score_task_templates: ReplayU32,
    geometry: ScoreGeometry,
    buffers: SelectBuffers<'a>,
    scores: &'a Buffer,
}

impl Operator for SelectInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        match self.geometry {
            ScoreGeometry::Generic => {
                Score {
                    invocation: &self,
                    kernel: &self.compute.score,
                }
                .record(recorder)
            },
            ScoreGeometry::Simd { groups_per_simd } => {
                Score {
                    invocation: &self,
                    kernel: &self.compute.vector_score.as_ref().unwrap()
                        [usize::from(self.shape.num_total_request_sparse_queries != 1)]
                        [groups_per_simd.ilog2() as usize],
                }
                .record(recorder)
            },
            ScoreGeometry::Steel { records_head_loop } => {
                let flat = self
                    .compute
                    .flat_score
                    .as_ref()
                    .expect("matrix QSA scorer must be compiled for its geometry");
                if records_head_loop {
                    let head_loop = self
                        .compute
                        .head_loop_score
                        .as_ref()
                        .expect("QSA head-loop scorer must be compiled for its geometry");
                    recorder.record_disjoint_buffers(&[self.scores], || {
                        Score {
                            invocation: &self,
                            kernel: flat,
                        }
                        .record(recorder);
                        Score {
                            invocation: &self,
                            kernel: head_loop,
                        }
                        .record(recorder);
                    });
                } else {
                    Score {
                        invocation: &self,
                        kernel: flat,
                    }
                    .record(recorder);
                }
            },
        }
        recorder.record_with_barrier_before(Select { invocation: &self });
    }
}

impl SelectInvocation<'_> {
    fn bind_active(&self, recorder: &CommandRecorder<'_>, index: usize) {
        match self.num_active_tokens {
            ReplayU32::Fixed(active) => recorder.set_u32(index, active),
            ReplayU32::Parameter(key) => recorder.bind_u32(index, key, 1, self.shape.num_total_tokens),
        }
    }

    fn bind_active_score_tasks(&self, recorder: &CommandRecorder<'_>, index: usize) {
        match self.num_active_score_task_templates {
            ReplayU32::Fixed(active) => recorder.set_u32(index, active),
            ReplayU32::Parameter(key) => recorder.bind_u32(index, key, 0, self.shape.num_total_score_task_templates),
        }
    }

    fn groups_per_page(&self) -> u32 {
        self.layout.page_bytes / (self.compute.config.head_dim * self.compute.config.dtype.item_size() as u32)
    }
}

impl Operator for PrepareInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let i = self;
        recorder.set_kernel(&i.compute.prepare);
        recorder.set_buffer_read(0, i.buffers.qk, 0);
        recorder.set_buffer_read(1, i.buffers.tokens, 0);
        recorder.set_buffer_read(2, i.buffers.q_norm, 0);
        recorder.set_buffer_read(3, i.buffers.k_norm, 0);
        recorder.set_buffer_read_write(4, i.buffers.raw_states, 0);
        recorder.set_buffer_write(5, i.buffers.pages, 0);
        recorder.set_buffer_read(6, i.buffers.page_ids, 0);
        recorder.set_buffer_write(7, i.buffers.q, 0);
        match i.num_active_tokens {
            ReplayU32::Fixed(active) => recorder.set_u32(8, active),
            ReplayU32::Parameter(key) => recorder.bind_u32(8, key, 1, i.num_total_tokens),
        }
        recorder.set_u32(9, i.layout.num_pages_per_request);
        recorder.set_u32(
            10,
            i.layout.page_bytes / (i.compute.config.head_dim * i.compute.config.dtype.item_size() as u32),
        );
        recorder.dispatch_threadblocks(
            (
                i.compute.config.num_q_heads as usize + 1,
                i.num_total_tokens as usize,
                1,
            ),
            (THREADS, 1, 1),
        );
    }
}

// A template supplies the request-local query range. Grid x supplies the key tile.
// SIMD: 1 Q x 4 keys, 128 threads. Vector: <=8 Q x (8*B) keys, 256 threads.
// Steel: <=8 Q x 4 heads or <=32 Q per head, 64 keys, 128 threads.
struct Score<'a, 'b> {
    invocation: &'a SelectInvocation<'b>,
    kernel: &'a CompiledKernel,
}

impl Operator for Score<'_, '_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let i = self.invocation;
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, i.buffers.q, 0);
        recorder.set_buffer_read(1, i.buffers.tokens, 0);
        recorder.set_buffer_read(2, i.buffers.score_task_templates, 0);
        recorder.set_buffer_read(3, i.buffers.pages, 0);
        recorder.set_buffer_read(4, i.buffers.page_ids, 0);
        recorder.set_buffer_write(5, i.scores, 0);
        i.bind_active_score_tasks(recorder, 6);
        recorder.set_u32(7, i.layout.num_pages_per_request);
        recorder.set_u32(8, i.groups_per_page());
        recorder.set_u32(9, i.shape.num_total_groups);
        if matches!(i.geometry, ScoreGeometry::Simd { .. }) {
            recorder.set_u32(10, i.groups_per_page().ilog2());
        }
        let (tiles, threads) = match i.geometry {
            ScoreGeometry::Generic => (i.shape.num_total_groups.div_ceil(4), 128),
            ScoreGeometry::Simd { groups_per_simd } => {
                (
                    aligned_score_tile_count(i.shape.num_total_groups, 8 * groups_per_simd),
                    256,
                )
            },
            ScoreGeometry::Steel { .. } => {
                (
                    aligned_score_tile_count(i.shape.num_total_groups, SCORE_TILE_GROUPS),
                    128,
                )
            },
        };
        recorder.dispatch_threadblocks(
            (tiles as usize, i.shape.num_total_score_task_templates as usize, 1),
            (threads, 1, 1),
        );
    }
}

struct Select<'a, 'b> {
    invocation: &'a SelectInvocation<'b>,
}
impl Operator for Select<'_, '_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let i = self.invocation;
        let sampled = i.compute.sampled_select.is_some()
            && (i.shape.num_total_groups >= 8192 || i.shape.num_total_sparse_queries <= 8);
        recorder.set_kernel(if sampled {
            i.compute.sampled_select.as_ref().unwrap()
        } else {
            &i.compute.select
        });
        recorder.set_buffer_read(0, i.scores, 0);
        recorder.set_buffer_read(1, i.buffers.tokens, 0);
        recorder.set_buffer_write(2, i.buffers.group_ids, 0);
        i.bind_active(recorder, 3);
        recorder.set_u32(4, i.shape.num_total_groups);
        recorder.dispatch_threadblocks(
            (i.shape.num_total_tokens as usize, 1, 1),
            (
                if sampled {
                    SAMPLED_SELECTION_THREADS
                } else {
                    SELECTION_THREADS
                },
                1,
                1,
            ),
        );
    }
}

fn validate_tokens(num_total_tokens: u32, num_active_tokens: ReplayU32, tokens: &Buffer) {
    assert!(num_total_tokens > 0);
    if let ReplayU32::Fixed(active) = num_active_tokens {
        assert!(active > 0 && active <= num_total_tokens);
    }
    assert!(tokens.len_bytes() >= tensor_bytes(num_total_tokens, 1, size_of::<Token>()));
}

fn tensor_bytes(rows: u32, width: u32, bytes: usize) -> usize {
    checked_product("QSA indexer buffer bytes", &[rows as usize, width as usize, bytes])
}

fn aligned_score_tile_count(num_total_groups: u32, groups_per_tile: u32) -> u32 {
    ((u64::from(num_total_groups) + u64::from(groups_per_tile) - 1).div_ceil(u64::from(groups_per_tile)))
        .try_into()
        .expect("QSA indexer aligned score tile count must fit u32")
}

fn steel_score_source(source: &str) -> String {
    let root = crate::mlx_headers::find_mlx_metal_header_root(
        "utils.h",
        |root| root.join("mlx/backend/metal/kernels/steel/gemm/gemm.h").is_file(),
        "QSA indexer Steel scorer",
    );
    let mut included = HashSet::new();
    let utils = crate::mlx_headers::read_mlx_metal_header(&root, "mlx/backend/metal/kernels/utils.h", &mut included);
    let gemm =
        crate::mlx_headers::read_mlx_metal_header(&root, "mlx/backend/metal/kernels/steel/gemm/gemm.h", &mut included);
    format!(
        "{source}\n{utils}\n{gemm}\n{}",
        include_str!("metal/qsa_indexer_steel.metal")
    )
}

#[cfg(test)]
#[path = "qsa_indexer_test.rs"]
mod tests;
