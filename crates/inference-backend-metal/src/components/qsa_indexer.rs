//! Pooled-key preparation, paged scoring, and exact group top-k.

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
        self.top_k
            .checked_next_power_of_two()
            .and_then(|count| count.checked_mul(2))
            .expect("QSA indexer selection tile must fit u32");
    }

    fn candidate_capacity(self) -> u32 {
        self.top_k.next_power_of_two()
    }
    fn selection_tile(self) -> u32 {
        (2 * self.candidate_capacity()).max(256)
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
    /// Score columns and dispatch capacity, relative to each query's `group_start`.
    pub num_total_groups: u32,
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
    pub pages: &'a Buffer,
    pub page_ids: &'a Buffer,
    /// `[T, top_k]` logical group IDs. Only sparse query rows are written.
    /// Each sparse row has exactly `top_k` distinct, visible IDs in score order.
    pub group_ids: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct SelectScratch<'a> {
    pub scores: &'a Buffer,
    pub candidates: [&'a Buffer; 2],
}

pub struct Compute {
    config: Config,
    prepare: CompiledKernel,
    score: CompiledKernel,
    select: CompiledKernel,
    merge: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        let selection_bytes = config.selection_tile() as usize * 8;
        let norm_bytes = (config.head_dim as usize + THREADS / 32) * 4;
        assert!(
            selection_bytes.max(norm_bytes) <= device.max_threadblock_memory_length(),
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
                 {}u, top_k = {}u, rope_dim = {}u;\nconstant uint threads = {THREADS}u, candidate_capacity = {}u, \
                 selection_tile = {}u;\nconstant float norm_eps = {:.9e}f, rope_theta = {:.9e}f;",
                config.num_q_heads,
                config.head_dim,
                config.compress_ratio,
                config.top_k,
                config.rope_dim,
                config.candidate_capacity(),
                config.selection_tile(),
                config.norm_eps,
                config.rope_theta
            ),
            1,
        );
        Self {
            config,
            prepare: CompiledKernel::new(device, &source, "qsa_indexer_prepare"),
            score: CompiledKernel::new(device, &source, "qsa_indexer_score"),
            select: CompiledKernel::new(device, &source, "qsa_indexer_select"),
            merge: CompiledKernel::new(device, &source, "qsa_indexer_merge"),
        }
    }

    pub fn candidate_bytes(&self, shape: Shape) -> usize {
        checked_product(
            "QSA indexer candidate bytes",
            &[
                shape.num_total_tokens as usize,
                shape.num_total_groups.div_ceil(self.config.selection_tile()) as usize,
                self.config.candidate_capacity() as usize,
                8,
            ],
        )
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
        buffers: SelectBuffers<'a>,
        scratch: SelectScratch<'a>,
    ) -> SelectInvocation<'a> {
        validate_tokens(shape.num_total_tokens, num_active_tokens, buffers.tokens);
        self.validate_cache(layout, buffers.pages, buffers.page_ids);
        let config = self.config;
        let groups_per_page = layout.page_bytes as usize / (config.head_dim as usize * config.dtype.item_size());
        assert!(shape.num_total_groups > config.top_k);
        assert!(shape.num_total_groups as u64 <= layout.num_pages_per_request as u64 * groups_per_page as u64);
        assert!(
            buffers.q.len_bytes()
                >= tensor_bytes(
                    shape.num_total_tokens,
                    config.num_q_heads * config.head_dim,
                    config.dtype.item_size()
                )
        );
        assert!(buffers.group_ids.len_bytes() >= tensor_bytes(shape.num_total_tokens, config.top_k, 4));
        assert!(scratch.scores.len_bytes() >= tensor_bytes(shape.num_total_tokens, shape.num_total_groups, 4));
        for candidates in scratch.candidates {
            assert!(candidates.len_bytes() >= self.candidate_bytes(shape));
        }
        SelectInvocation {
            compute: self,
            shape,
            layout,
            num_active_tokens,
            buffers,
            scratch,
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
    buffers: SelectBuffers<'a>,
    scratch: SelectScratch<'a>,
}

impl Operator for SelectInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        Score { invocation: &self }.record(recorder);
        recorder.record_with_barrier_before(Select { invocation: &self });
        let mut num_partitions = self
            .shape
            .num_total_groups
            .div_ceil(self.compute.config.selection_tile());
        let mut source = 0;
        while num_partitions > 1 {
            recorder.record_with_barrier_before(Merge {
                invocation: &self,
                num_partitions,
                source,
            });
            num_partitions = num_partitions.div_ceil(2);
            source ^= 1;
        }
    }
}

impl SelectInvocation<'_> {
    fn bind_active(&self, recorder: &CommandRecorder<'_>, index: usize) {
        match self.num_active_tokens {
            ReplayU32::Fixed(active) => recorder.set_u32(index, active),
            ReplayU32::Parameter(key) => recorder.bind_u32(index, key, 1, self.shape.num_total_tokens),
        }
    }

    fn groups_per_page(&self) -> u32 {
        self.layout.page_bytes / (self.compute.config.head_dim * self.compute.config.dtype.item_size() as u32)
    }
    fn partitions(&self) -> u32 {
        self.shape
            .num_total_groups
            .div_ceil(self.compute.config.selection_tile())
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

struct Score<'a, 'b> {
    invocation: &'a SelectInvocation<'b>,
}
impl Operator for Score<'_, '_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let i = self.invocation;
        recorder.set_kernel(&i.compute.score);
        recorder.set_buffer_read(0, i.buffers.q, 0);
        recorder.set_buffer_read(1, i.buffers.tokens, 0);
        recorder.set_buffer_read(2, i.buffers.pages, 0);
        recorder.set_buffer_read(3, i.buffers.page_ids, 0);
        recorder.set_buffer_write(4, i.scratch.scores, 0);
        i.bind_active(recorder, 5);
        recorder.set_u32(6, i.layout.num_pages_per_request);
        recorder.set_u32(7, i.groups_per_page());
        recorder.set_u32(8, i.shape.num_total_groups);
        recorder.dispatch_threadblocks(
            (
                i.shape.num_total_groups.div_ceil((THREADS / 32) as u32) as usize,
                i.shape.num_total_tokens as usize,
                1,
            ),
            (THREADS, 1, 1),
        );
    }
}

struct Select<'a, 'b> {
    invocation: &'a SelectInvocation<'b>,
}
impl Operator for Select<'_, '_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let i = self.invocation;
        recorder.set_kernel(&i.compute.select);
        recorder.set_buffer_read(0, i.scratch.scores, 0);
        recorder.set_buffer_read(1, i.buffers.tokens, 0);
        recorder.set_buffer_write(2, i.scratch.candidates[0], 0);
        recorder.set_buffer_write(3, i.buffers.group_ids, 0);
        i.bind_active(recorder, 4);
        recorder.set_u32(5, i.shape.num_total_groups);
        recorder.set_u32(6, i.partitions());
        recorder.dispatch_threadblocks(
            (i.partitions() as usize, i.shape.num_total_tokens as usize, 1),
            (THREADS, 1, 1),
        );
    }
}

struct Merge<'a, 'b> {
    invocation: &'a SelectInvocation<'b>,
    num_partitions: u32,
    source: usize,
}
impl Operator for Merge<'_, '_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let i = self.invocation;
        recorder.set_kernel(&i.compute.merge);
        recorder.set_buffer_read(0, i.scratch.candidates[self.source], 0);
        recorder.set_buffer_write(1, i.scratch.candidates[self.source ^ 1], 0);
        recorder.set_buffer_read(2, i.buffers.tokens, 0);
        recorder.set_buffer_write(3, i.buffers.group_ids, 0);
        i.bind_active(recorder, 4);
        recorder.set_u32(5, self.num_partitions);
        recorder.set_u32(6, i.partitions());
        recorder.dispatch_threadblocks(
            (
                self.num_partitions.div_ceil(2) as usize,
                i.shape.num_total_tokens as usize,
                1,
            ),
            (THREADS, 1, 1),
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

#[cfg(test)]
#[path = "qsa_indexer_test.rs"]
mod tests;
