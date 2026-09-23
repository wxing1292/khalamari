use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLComputePipelineState;

use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("metal/gated_residual.metal");
const THREADS: usize = 128;
const SIMD_GROUPS: usize = THREADS / 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub num_branches: u32,
    pub hidden_dim: u32,
    pub low_rank_dim: u32,
    pub eps: f32,
    pub dtype: Dtype,
}

impl Config {
    pub fn validate(self) {
        assert!(self.num_branches > 0);
        assert!(self.hidden_dim > 0);
        assert!(self.low_rank_dim > 0);
        assert!(self.eps.is_finite() && self.eps > 0.0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
        self.num_branches
            .checked_mul(self.hidden_dim)
            .expect("GR residual dimension must fit u32");
        self.bytes(self.low_rank_dim, self.residual_dim());
        self.bytes(self.num_branches, self.residual_dim());
    }

    fn residual_dim(self) -> u32 {
        self.num_branches * self.hidden_dim
    }

    fn bytes(self, rows: u32, columns: u32) -> usize {
        checked_product(
            "GR tensor byte length",
            &[rows as usize, columns as usize, self.dtype.item_size()],
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Shape {
    pub num_total_tokens: u32,
}

#[derive(Clone, Copy)]
pub struct ReadWeights<'a> {
    /// Raw Gemma RMSNorm offsets, `[num_branches, hidden_dim]`.
    pub norm: &'a Buffer,
    /// `[low_rank_dim, num_branches * hidden_dim]`, row-major.
    pub down: &'a Buffer,
    /// `[num_branches * hidden_dim, low_rank_dim]`, row-major.
    pub up: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct ReadBuffers<'a> {
    pub residual: &'a Buffer,
    pub hidden_state: &'a Buffer,
    /// Omit for a final read-only mixer.
    pub write_gate: Option<WriteGate<'a>>,
}

#[derive(Clone, Copy)]
pub struct WriteGate<'a> {
    /// `[num_branches, num_branches * hidden_dim]`, row-major.
    pub weight: &'a Buffer,
    /// `[num_total_tokens, num_branches]`. Must live through the paired write.
    pub output: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct ReadScratch<'a> {
    pub normalized_residual: &'a Buffer,
    pub low_rank: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct WriteBuffers<'a> {
    pub residual: &'a Buffer,
    pub gate: &'a Buffer,
    pub hidden_state: &'a Buffer,
    /// May alias `residual`. Must not alias another input.
    pub output: &'a Buffer,
}

/// Stateless GR read/write operators. The caller owns weights, scratch, and replay lifetime.
pub struct Compute {
    config: Config,
    normalize: CompiledKernel,
    down: CompiledKernel,
    down_gate: CompiledKernel,
    read: CompiledKernel,
    write: CompiledKernel,
    write_norm: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        let scalar = match config.dtype {
            Dtype::Float32 => "float",
            Dtype::Bfloat16 => "bfloat",
            _ => unreachable!(),
        };
        let source = SOURCE.replacen(
            "using namespace metal;",
            &format!(
                "using namespace metal;\nusing scalar_t = {scalar};\nconstant uint branches = {}u;\nconstant uint \
                 hidden = {}u;\nconstant uint rank = {}u;\nconstant uint wide = {}u;\nconstant uint threads = \
                 {THREADS}u;\nconstant uint simd_groups = {SIMD_GROUPS}u;",
                config.num_branches,
                config.hidden_dim,
                config.low_rank_dim,
                config.residual_dim(),
            ),
            1,
        );
        Self {
            config,
            normalize: CompiledKernel::new(device, &source, "gr_normalize"),
            down: CompiledKernel::new(device, &source, "gr_down"),
            down_gate: CompiledKernel::new(device, &source, "gr_down_gate"),
            read: CompiledKernel::new(device, &source, "gr_read"),
            write: CompiledKernel::new(device, &source, "gr_write"),
            write_norm: CompiledKernel::new(device, &source, "gr_write_norm"),
        }
    }

    pub fn read<'a>(
        &'a self,
        shape: Shape,
        num_active_tokens: ReplayU32,
        buffers: ReadBuffers<'a>,
        scratch: ReadScratch<'a>,
        weights: ReadWeights<'a>,
    ) -> ReadInvocation<'a> {
        validate_shape(shape, num_active_tokens);
        let config = self.config;
        let wide = config.residual_dim();
        assert!(buffers.residual.len_bytes() >= config.bytes(shape.num_total_tokens, wide));
        assert!(buffers.hidden_state.len_bytes() >= config.bytes(shape.num_total_tokens, config.hidden_dim));
        assert!(scratch.normalized_residual.len_bytes() >= config.bytes(shape.num_total_tokens, wide));
        assert!(scratch.low_rank.len_bytes() >= config.bytes(shape.num_total_tokens, config.low_rank_dim));
        assert!(weights.norm.len_bytes() >= config.bytes(1, wide));
        assert!(weights.down.len_bytes() >= config.bytes(config.low_rank_dim, wide));
        assert!(weights.up.len_bytes() >= config.bytes(wide, config.low_rank_dim));
        if let Some(gate) = buffers.write_gate {
            assert!(gate.weight.len_bytes() >= config.bytes(config.num_branches, wide));
            assert!(gate.output.len_bytes() >= config.bytes(shape.num_total_tokens, config.num_branches));
        }
        ReadInvocation {
            compute: self,
            shape,
            num_active_tokens,
            buffers,
            scratch,
            weights,
        }
    }

    pub fn write<'a>(
        &'a self,
        shape: Shape,
        num_active_tokens: ReplayU32,
        buffers: WriteBuffers<'a>,
    ) -> WriteInvocation<'a> {
        validate_shape(shape, num_active_tokens);
        let config = self.config;
        let wide = config.residual_dim();
        let residual_bytes = config.bytes(shape.num_total_tokens, wide);
        assert!(buffers.residual.len_bytes() >= residual_bytes);
        assert!(buffers.gate.len_bytes() >= config.bytes(shape.num_total_tokens, config.num_branches));
        assert!(buffers.output.len_bytes() >= residual_bytes);
        assert!(buffers.hidden_state.len_bytes() >= config.bytes(shape.num_total_tokens, config.hidden_dim));
        WriteInvocation {
            compute: self,
            shape,
            num_active_tokens,
            buffers,
        }
    }
}

pub struct ReadInvocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    num_active_tokens: ReplayU32,
    buffers: ReadBuffers<'a>,
    scratch: ReadScratch<'a>,
    weights: ReadWeights<'a>,
}

impl Operator for ReadInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        Normalize { read: &self }.record(recorder);
        self.record_projections(recorder);
    }
}

impl<'a> ReadInvocation<'a> {
    fn record_projections(&self, recorder: &CommandRecorder<'_>) {
        recorder.record_with_barrier_before(Down { read: self });
        recorder.record_with_barrier_before(Read { read: self });
    }

    pub fn with_write(self, write: WriteReplayOp) -> impl Operator + 'a {
        assert!(
            write.can_fuse(&self),
            "GR write/read fusion requires compatible bindings"
        );
        WriteRead { write, read: self }
    }
}

struct WriteRead<'a> {
    write: WriteReplayOp,
    read: ReadInvocation<'a>,
}

impl Operator for WriteRead<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        WriteNorm {
            write: self.write,
            read: &self.read,
        }
        .record(recorder);
        self.read.record_projections(recorder);
    }
}

struct Normalize<'a, 'b> {
    read: &'a ReadInvocation<'b>,
}

impl Operator for Normalize<'_, '_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let read = self.read;
        recorder.set_kernel(&read.compute.normalize);
        recorder.set_buffer_read(0, read.buffers.residual, 0);
        recorder.set_buffer_read(1, read.weights.norm, 0);
        recorder.set_buffer_write(2, read.scratch.normalized_residual, 0);
        bind_active(recorder, 3, read.shape, read.num_active_tokens);
        recorder.set_f32(4, read.compute.config.eps);
        recorder.dispatch_threadblocks(
            (
                read.compute.config.num_branches as usize,
                read.shape.num_total_tokens as usize,
                1,
            ),
            (THREADS, 1, 1),
        );
    }
}

struct Down<'a, 'b> {
    read: &'a ReadInvocation<'b>,
}

impl Operator for Down<'_, '_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let read = self.read;
        let mut rows = read.compute.config.low_rank_dim as usize;
        if let Some(gate) = read.buffers.write_gate {
            recorder.set_kernel(&read.compute.down_gate);
            recorder.set_buffer_read(4, gate.weight, 0);
            recorder.set_buffer_write(5, gate.output, 0);
            rows += read.compute.config.num_branches as usize;
        } else {
            recorder.set_kernel(&read.compute.down);
        }
        recorder.set_buffer_read(0, read.scratch.normalized_residual, 0);
        recorder.set_buffer_read(1, read.weights.down, 0);
        recorder.set_buffer_write(2, read.scratch.low_rank, 0);
        bind_active(recorder, 3, read.shape, read.num_active_tokens);
        recorder.dispatch_threadblocks(
            (rows.div_ceil(SIMD_GROUPS), read.shape.num_total_tokens as usize, 1),
            (THREADS, 1, 1),
        );
    }
}

struct Read<'a, 'b> {
    read: &'a ReadInvocation<'b>,
}

impl Operator for Read<'_, '_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let read = self.read;
        recorder.set_kernel(&read.compute.read);
        recorder.set_buffer_read(0, read.scratch.normalized_residual, 0);
        recorder.set_buffer_read(1, read.scratch.low_rank, 0);
        recorder.set_buffer_read(2, read.weights.up, 0);
        recorder.set_buffer_write(3, read.buffers.hidden_state, 0);
        bind_active(recorder, 4, read.shape, read.num_active_tokens);
        recorder.dispatch_threadblocks(
            (
                (read.compute.config.hidden_dim as usize).div_ceil(SIMD_GROUPS),
                read.shape.num_total_tokens as usize,
                1,
            ),
            (THREADS, 1, 1),
        );
    }
}

pub struct WriteInvocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    num_active_tokens: ReplayU32,
    buffers: WriteBuffers<'a>,
}

impl WriteInvocation<'_> {
    pub fn into_replay_op(self) -> WriteReplayOp {
        WriteReplayOp {
            config: self.compute.config,
            shape: self.shape,
            num_active_tokens: self.num_active_tokens,
            pipeline: self.compute.write.as_raw_retained(),
            norm_pipeline: self.compute.write_norm.as_raw_retained(),
            residual: self.buffers.residual.as_raw_retained(),
            gate: self.buffers.gate.as_raw_retained(),
            hidden_state: self.buffers.hidden_state.as_raw_retained(),
            output: self.buffers.output.as_raw_retained(),
        }
    }
}

impl Operator for WriteInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.into_replay_op().record(recorder);
    }
}

/// Retains a pending write until the recorder can inspect the next operation.
pub struct WriteReplayOp {
    config: Config,
    shape: Shape,
    num_active_tokens: ReplayU32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    norm_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    residual: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_state: Retained<ProtocolObject<dyn MTLBuffer>>,
    output: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl WriteReplayOp {
    pub fn can_fuse(&self, read: &ReadInvocation<'_>) -> bool {
        let next = read.compute.config;
        self.config.num_branches == next.num_branches
            && self.config.hidden_dim == next.hidden_dim
            && self.config.dtype == next.dtype
            && self.shape == read.shape
            && self.num_active_tokens == read.num_active_tokens
            && std::ptr::eq(&*self.output, read.buffers.residual.as_raw())
            // The separate read can reuse dead write inputs. Fusion cannot overwrite
            // inputs still read by another branch threadgroup.
            && !std::ptr::eq(&*self.gate, read.scratch.normalized_residual.as_raw())
            && !std::ptr::eq(&*self.hidden_state, read.scratch.normalized_residual.as_raw())
            && !std::ptr::eq(&*self.output, read.weights.norm.as_raw())
    }

    fn bind(&self, recorder: &CommandRecorder<'_>) {
        recorder.set_retained_buffer_read(0, &self.residual, 0);
        recorder.set_retained_buffer_read(1, &self.gate, 0);
        recorder.set_retained_buffer_read(2, &self.hidden_state, 0);
        recorder.set_retained_buffer_write(3, &self.output, 0);
        bind_active(recorder, 4, self.shape, self.num_active_tokens);
    }

    fn dispatch(&self, recorder: &CommandRecorder<'_>) {
        recorder.dispatch_threadblocks(
            (
                self.config.num_branches as usize,
                self.shape.num_total_tokens as usize,
                1,
            ),
            (THREADS, 1, 1),
        );
    }
}

impl Operator for WriteReplayOp {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_retained_pipeline_state(&self.pipeline);
        self.bind(recorder);
        self.dispatch(recorder);
    }
}

struct WriteNorm<'a, 'b> {
    write: WriteReplayOp,
    read: &'a ReadInvocation<'b>,
}

impl Operator for WriteNorm<'_, '_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_retained_pipeline_state(&self.write.norm_pipeline);
        self.write.bind(recorder);
        recorder.set_buffer_read(5, self.read.weights.norm, 0);
        recorder.set_buffer_write(6, self.read.scratch.normalized_residual, 0);
        recorder.set_f32(7, self.read.compute.config.eps);
        self.write.dispatch(recorder);
    }
}

fn validate_shape(shape: Shape, active: ReplayU32) {
    assert!(shape.num_total_tokens > 0);
    if let ReplayU32::Fixed(value) = active {
        assert!(value > 0 && value <= shape.num_total_tokens);
    }
}

fn bind_active(recorder: &CommandRecorder<'_>, index: usize, shape: Shape, active: ReplayU32) {
    match active {
        ReplayU32::Fixed(value) => recorder.set_u32(index, value),
        ReplayU32::Parameter(key) => recorder.bind_u32(index, key, 1, shape.num_total_tokens),
    }
}

#[cfg(test)]
#[path = "gated_residual_test.rs"]
mod tests;
