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
        self.low_rank_dim
            .checked_add(self.num_branches)
            .expect("GR projection row count must fit u32");
        self.bytes(self.low_rank_dim, self.residual_dim());
        self.bytes(self.num_branches, self.residual_dim());
    }

    /// Capacity for any recorded shape up to `max_tokens`, including small-row partials.
    pub fn projection_scratch_bytes(self, max_tokens: usize) -> usize {
        self.validate();
        let tokens = max_tokens.try_into().expect("GR token capacity must fit u32");
        assert!(tokens > 0);
        self.projection_bytes(tokens)
    }

    fn projection_bytes(self, tokens: u32) -> usize {
        let activations = self.bytes(tokens, self.low_rank_dim);
        if self.supports_small() {
            let partials = tokens.min(4) as usize
                * (self.num_branches * (self.low_rank_dim + self.num_branches) + self.num_branches) as usize
                * size_of::<f32>();
            activations.max(partials)
        } else {
            activations
        }
    }

    fn supports_matrix(self) -> bool {
        self.dtype == Dtype::Bfloat16
            && self.num_branches == 4
            && self.hidden_dim.is_multiple_of(32)
            && self.low_rank_dim.is_multiple_of(8)
    }

    fn supports_small(self) -> bool {
        self.supports_matrix()
            && self.hidden_dim.is_multiple_of(256)
            && self.low_rank_dim.is_multiple_of(16)
            // Each SMEM allocation fits 32 KiB. Read-only down groups cover N's stripes.
            && self.hidden_dim <= (32 * 1024 - 8 * 4) / 2
            && self.low_rank_dim <= 32 * 1024 / 2
            && self.low_rank_dim / 8 >= self.hidden_dim / 256
    }

    fn shader_source(self, source: &str, extra: &str) -> String {
        let scalar = if self.dtype == Dtype::Bfloat16 {
            "bfloat"
        } else {
            "float"
        };
        source.replacen(
            "using namespace metal;",
            &format!(
                "using namespace metal;\nusing scalar_t = {scalar};\nconstant uint branches = {}u;\nconstant uint \
                 hidden = {}u;\nconstant uint rank = {}u;\nconstant uint wide = {}u;\nconstant uint threads = \
                 {THREADS}u;\nconstant uint simd_groups = {SIMD_GROUPS}u;\n{extra}",
                self.num_branches,
                self.hidden_dim,
                self.low_rank_dim,
                self.residual_dim(),
            ),
            1,
        )
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
    /// Allocate with `Config::projection_scratch_bytes`. Contents are kernel-local scratch.
    pub projection: &'a Buffer,
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
    small: Option<SmallKernels>,
    matrix: Option<[MatrixKernels; 3]>,
}

struct SmallKernels {
    // [pending_write][with_gate], selected when replay is recorded.
    down: [[CompiledKernel; 2]; 2],
    read: [[CompiledKernel; 2]; 2],
}

struct MatrixKernels {
    down: [CompiledKernel; 2],
    read: CompiledKernel,
    token_rows: usize,
}

#[derive(Clone, Copy)]
enum ReadKernels<'a> {
    Scalar,
    Small(&'a SmallKernels),
    Matrix(&'a MatrixKernels),
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        let source = config.shader_source(SOURCE, "");
        let small = config.supports_small().then(|| {
            let compile = |name| {
                std::array::from_fn(|pending| {
                    std::array::from_fn(|gate| {
                        let source = config.shader_source(
                            include_str!("metal/gated_residual_small.metal"),
                            &format!(
                                "constant bool pending_write = {};\nconstant bool with_gate = {};",
                                pending != 0,
                                gate != 0
                            ),
                        );
                        CompiledKernel::new(device, &source, name)
                    })
                })
            };
            SmallKernels {
                down: compile("gr_small_down"),
                read: compile("gr_small_read"),
            }
        });
        let matrix = config.supports_matrix().then(|| {
            std::array::from_fn(|index| {
                let token_tiles = 1 << index;
                let source = |with_gate| {
                    config.shader_source(
                        include_str!("metal/gated_residual_matrix.metal"),
                        &format!("constant uint token_tiles = {token_tiles}u;\nconstant bool with_gate = {with_gate};"),
                    )
                };
                MatrixKernels {
                    down: std::array::from_fn(|gate| CompiledKernel::new(device, &source(gate != 0), "gr_matrix_down")),
                    read: CompiledKernel::new(device, &source(false), "gr_matrix_read"),
                    token_rows: 8 * token_tiles,
                }
            })
        });
        Self {
            config,
            small,
            matrix,
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
        assert!(scratch.projection.len_bytes() >= config.projection_bytes(shape.num_total_tokens));
        assert!(weights.norm.len_bytes() >= config.bytes(1, wide));
        assert!(weights.down.len_bytes() >= config.bytes(config.low_rank_dim, wide));
        assert!(weights.up.len_bytes() >= config.bytes(wide, config.low_rank_dim));
        if let Some(gate) = buffers.write_gate {
            assert!(gate.weight.len_bytes() >= config.bytes(config.num_branches, wide));
            assert!(gate.output.len_bytes() >= config.bytes(shape.num_total_tokens, config.num_branches));
        }
        ReadInvocation {
            compute: self,
            kernels: self.select(shape.num_total_tokens),
            shape,
            num_active_tokens,
            buffers,
            scratch,
            weights,
        }
    }

    fn select(&self, num_total_tokens: u32) -> ReadKernels<'_> {
        if num_total_tokens <= 4
            && let Some(small) = &self.small
        {
            return ReadKernels::Small(small);
        }
        if let Some(matrix) = &self.matrix {
            let index = if num_total_tokens <= 16 {
                0
            } else if num_total_tokens <= 128 {
                1
            } else {
                2
            };
            ReadKernels::Matrix(&matrix[index])
        } else {
            ReadKernels::Scalar
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
    kernels: ReadKernels<'a>,
    shape: Shape,
    num_active_tokens: ReplayU32,
    buffers: ReadBuffers<'a>,
    scratch: ReadScratch<'a>,
    weights: ReadWeights<'a>,
}

impl Operator for ReadInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        if let ReadKernels::Small(kernels) = self.kernels {
            self.record_small(recorder, kernels, None);
        } else {
            Normalize { read: &self }.record(recorder);
            self.record_projections(recorder);
        }
    }
}

impl<'a> ReadInvocation<'a> {
    fn record_small(&self, recorder: &CommandRecorder<'_>, kernels: &SmallKernels, write: Option<&WriteReplayOp>) {
        let pending = usize::from(write.is_some());
        let gated = usize::from(self.buffers.write_gate.is_some());
        let config = self.compute.config;
        let columns = config.low_rank_dim + if gated != 0 { config.num_branches } else { 0 };
        recorder.set_kernel(&kernels.down[pending][gated]);
        if let Some(write) = write {
            recorder.set_retained_buffer_read(0, &write.residual, 0);
            recorder.set_retained_buffer_read(8, &write.gate, 0);
            recorder.set_retained_buffer_read(9, &write.hidden_state, 0);
        } else {
            recorder.set_buffer_read(0, self.buffers.residual, 0);
        }
        recorder.set_buffer_read(1, self.weights.norm, 0);
        recorder.set_buffer_read(2, self.weights.down, 0);
        if let Some(gate) = self.buffers.write_gate {
            recorder.set_buffer_read(3, gate.weight, 0);
        }
        recorder.set_buffer_write(4, self.scratch.projection, 0);
        recorder.set_buffer_write(5, self.scratch.normalized_residual, 0);
        bind_active(recorder, 6, self.shape, self.num_active_tokens);
        recorder.set_u32(7, self.shape.num_total_tokens);
        recorder.set_f32(10, config.eps);
        recorder.dispatch_threadblocks(
            (
                (columns as usize).div_ceil(8),
                config.num_branches as usize,
                self.shape.num_total_tokens as usize,
            ),
            (256, 1, 1),
        );

        recorder.set_kernel(&kernels.read[pending][gated]);
        recorder.set_barrier_before();
        recorder.set_buffer_read(0, self.scratch.normalized_residual, 0);
        recorder.set_buffer_read(1, self.scratch.projection, 0);
        recorder.set_buffer_read(2, self.weights.up, 0);
        recorder.set_buffer_write(3, self.buffers.hidden_state, 0);
        if let Some(gate) = self.buffers.write_gate {
            recorder.set_buffer_write(4, gate.output, 0);
        }
        bind_active(recorder, 5, self.shape, self.num_active_tokens);
        recorder.set_u32(6, self.shape.num_total_tokens);
        if let Some(write) = write {
            recorder.set_retained_buffer_read(7, &write.residual, 0);
            recorder.set_retained_buffer_write(8, &write.output, 0);
            recorder.set_retained_buffer_read(9, &write.hidden_state, 0);
        }
        recorder.dispatch_threadblocks(
            (config.hidden_dim as usize / 16, self.shape.num_total_tokens as usize, 1),
            (256, 1, 1),
        );
    }

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
        if let ReadKernels::Small(kernels) = self.read.kernels {
            self.read.record_small(recorder, kernels, Some(&self.write));
        } else {
            WriteNorm {
                write: self.write,
                read: &self.read,
            }
            .record(recorder);
            self.read.record_projections(recorder);
        }
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
        if let ReadKernels::Matrix(kernels) = read.kernels {
            recorder.set_kernel(&kernels.down[usize::from(read.buffers.write_gate.is_some())]);
            recorder.set_buffer_read(0, read.scratch.normalized_residual, 0);
            recorder.set_buffer_read(1, read.weights.down, 0);
            if let Some(gate) = read.buffers.write_gate {
                recorder.set_buffer_read(2, gate.weight, 0);
                recorder.set_buffer_write(4, gate.output, 0);
                rows += read.compute.config.num_branches as usize;
            }
            recorder.set_buffer_write(3, read.scratch.projection, 0);
            bind_active(recorder, 5, read.shape, read.num_active_tokens);
            recorder.dispatch_threadblocks(
                (
                    rows.div_ceil(8),
                    (read.shape.num_total_tokens as usize).div_ceil(kernels.token_rows),
                    1,
                ),
                (256, 1, 1),
            );
            return;
        }
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
        recorder.set_buffer_write(2, read.scratch.projection, 0);
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
        let (kernel, hidden_columns, token_rows) = match read.kernels {
            ReadKernels::Matrix(kernels) => (&kernels.read, 32, kernels.token_rows),
            ReadKernels::Scalar => (&read.compute.read, SIMD_GROUPS, 1),
            ReadKernels::Small(_) => unreachable!(),
        };
        recorder.set_kernel(kernel);
        recorder.set_buffer_read(0, read.scratch.normalized_residual, 0);
        recorder.set_buffer_read(1, read.scratch.projection, 0);
        recorder.set_buffer_read(2, read.weights.up, 0);
        recorder.set_buffer_write(3, read.buffers.hidden_state, 0);
        bind_active(recorder, 4, read.shape, read.num_active_tokens);
        recorder.dispatch_threadblocks(
            (
                (read.compute.config.hidden_dim as usize).div_ceil(hidden_columns),
                (read.shape.num_total_tokens as usize).div_ceil(token_rows),
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
            && (!matches!(read.kernels, ReadKernels::Small(_)) || (
                !std::ptr::eq(&*self.residual, read.scratch.normalized_residual.as_raw())
                && !std::ptr::eq(&*self.residual, read.scratch.projection.as_raw())
                && !std::ptr::eq(&*self.gate, read.scratch.projection.as_raw())
                && !std::ptr::eq(&*self.hidden_state, read.scratch.projection.as_raw())
                && !std::ptr::eq(&*self.output, read.weights.down.as_raw())
                && !std::ptr::eq(&*self.output, read.weights.up.as_raw())
                && read.buffers.write_gate.is_none_or(|gate| {
                    !std::ptr::eq(&*self.hidden_state, gate.output.as_raw())
                        && !std::ptr::eq(&*self.output, gate.weight.as_raw())
                })
            ))
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
