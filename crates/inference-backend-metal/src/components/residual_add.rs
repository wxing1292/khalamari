use std::ops::Range;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLComputePipelineState;
use objc2_metal::MTLResource;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayU32;

const RESIDUAL_ADD_SOURCE: &str = include_str!("metal/residual_add.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants {
    thread_block: ThreadBlockConstants,
}

const KERNEL_CONSTANTS: KernelConstants = KernelConstants {
    thread_block: ThreadBlockConstants { required_threads: 256 },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub lhs_dtype: Dtype,
    pub rhs_dtype: Dtype,
    pub output_dtype: Dtype,
}

impl Config {
    pub fn f32() -> Self {
        Self {
            lhs_dtype: Dtype::Float32,
            rhs_dtype: Dtype::Float32,
            output_dtype: Dtype::Float32,
        }
    }

    pub fn bf16() -> Self {
        Self {
            lhs_dtype: Dtype::Bfloat16,
            rhs_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
        }
    }

    pub fn bf16_f32_to_bf16() -> Self {
        Self {
            lhs_dtype: Dtype::Bfloat16,
            rhs_dtype: Dtype::Float32,
            output_dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(
            matches!(
                (self.lhs_dtype, self.rhs_dtype, self.output_dtype),
                (Dtype::Float32, Dtype::Float32, Dtype::Float32)
                    | (Dtype::Bfloat16, Dtype::Bfloat16, Dtype::Bfloat16)
                    | (Dtype::Bfloat16, Dtype::Float32, Dtype::Bfloat16)
            ),
            "unsupported residual-add dtype combination: lhs={:?}, rhs={:?}, output={:?}",
            self.lhs_dtype,
            self.rhs_dtype,
            self.output_dtype
        );
    }

    pub fn lhs_bytes(self, shape: Shape) -> usize {
        self.validate();
        shape.validate();
        (shape.num_values as usize)
            .checked_mul(self.lhs_dtype.item_size())
            .expect("residual-add lhs byte length must fit usize")
    }

    pub fn rhs_bytes(self, shape: Shape) -> usize {
        self.validate();
        shape.validate();
        (shape.num_values as usize)
            .checked_mul(self.rhs_dtype.item_size())
            .expect("residual-add rhs byte length must fit usize")
    }

    pub fn output_bytes(self, shape: Shape) -> usize {
        self.validate();
        shape.validate();
        (shape.num_values as usize)
            .checked_mul(self.output_dtype.item_size())
            .expect("residual-add output byte length must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_values: u32,
}

impl Shape {
    pub fn validate(self) {
        assert!(self.num_values > 0);
    }
}

/// Two-dimensional row-major residual shape.
///
/// Both fields count elements. They do not count bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowShape {
    pub num_total_rows: u32,
    pub num_columns: u32,
}

impl RowShape {
    pub fn validate(self) {
        assert!(self.num_total_rows > 0);
        assert!(self.num_columns > 0);
        self.num_total_rows
            .checked_mul(self.num_columns)
            .expect("residual-add row value count must fit u32");
    }

    pub fn num_values(self) -> u32 {
        self.validate();
        self.num_total_rows * self.num_columns
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub lhs: &'a Buffer,
    pub rhs: &'a Buffer,
    pub output: &'a Buffer,
}

/// Elementwise residual-add data flow:
///
/// ```text
/// buffers.lhs ----\
///                  +--> output = lhs + rhs --> buffers.output
/// buffers.rhs ----/
/// ```
pub struct Compute {
    config: Config,
    kernel: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            kernel: CompiledKernel::new(device, RESIDUAL_ADD_SOURCE, residual_add_function_name(config)),
        }
    }

    pub fn invoke_values<'a>(&'a self, shape: Shape, buffers: Buffers<'a>) -> Invocation<'a> {
        Invocation {
            kernel: &self.kernel,
            config: self.config,
            shape,
            buffers,
            row_shape: None,
            num_active_rows_key: None,
        }
    }

    pub fn invoke_rows<'a>(
        &'a self,
        shape: RowShape,
        num_active_rows: ReplayU32,
        buffers: Buffers<'a>,
    ) -> Invocation<'a> {
        Invocation {
            kernel: &self.kernel,
            config: self.config,
            shape: Shape {
                num_values: shape.num_values(),
            },
            buffers,
            row_shape: Some(shape),
            num_active_rows_key: match num_active_rows {
                ReplayU32::Fixed(value) => {
                    assert_eq!(value, shape.num_total_rows);
                    None
                },
                ReplayU32::Parameter(key) => Some(key),
            },
        }
    }
}

pub struct Invocation<'a> {
    kernel: &'a CompiledKernel,
    config: Config,
    shape: Shape,
    buffers: Buffers<'a>,
    row_shape: Option<RowShape>,
    num_active_rows_key: Option<ReplayParameterKey>,
}

pub struct ReplayInvocation {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    config: Config,
    shape: Shape,
    buffers: OwnedBuffers,
    row_shape: Option<RowShape>,
    num_active_rows_key: Option<ReplayParameterKey>,
}

pub struct ReplayOp {
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub config: Config,
    pub shape: Shape,
    pub buffers: OwnedBuffers,
    pub row_shape: Option<RowShape>,
    pub num_active_rows_key: Option<ReplayParameterKey>,
}

pub struct CaptureReplayOp {
    pub residual: ReplayOp,
    pub capture: OwnedCaptureTarget,
}

pub struct CaptureReplayInvocation {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    residual: ReplayOp,
    capture: OwnedCaptureTarget,
}

/// Destination for capturing every complete row produced by a residual add.
///
/// The target is supported for BF16 replay. Each complete residual row is
/// written into the selected destination columns. An adjacent compatible
/// RMSNorm can fuse with this operation, but fusion is not required.
///
/// The BF16 capture kernel uses vec4 loads and stores. The source column count,
/// destination range width, destination column count, and destination column
/// start must therefore be divisible by four.
///
/// The residual invocation must carry an explicit `RowShape`.
/// Recording checks that its column count equals the selected column count.
/// Recording also checks destination capacity and buffer aliasing.
#[derive(Clone, Copy)]
pub struct CaptureTarget<'a> {
    buffer: &'a Buffer,
    num_destination_columns: u32,
    column_start: u32,
    column_end: u32,
}

pub struct OwnedCaptureTarget {
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub buffer_len_bytes: usize,
    pub num_destination_columns: u32,
    pub column_start: u32,
    pub column_end: u32,
}

#[derive(Clone)]
pub struct OwnedBuffers {
    pub lhs: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub lhs_len_bytes: usize,
    pub rhs: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub rhs_len_bytes: usize,
    pub output: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub output_len_bytes: usize,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.lhs, 0);
        recorder.set_buffer_read(1, self.buffers.rhs, 0);
        recorder.set_buffer_write(2, self.buffers.output, 0);
        record_shape(recorder, self.shape, self.row_shape, self.num_active_rows_key);
        dispatch_values(recorder, self.shape.num_values);
    }
}

impl Operator for ReplayInvocation {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        recorder.set_retained_pipeline_state(&self.pipeline);
        recorder.set_retained_buffer_read(0, &self.buffers.lhs, 0);
        recorder.set_retained_buffer_read(1, &self.buffers.rhs, 0);
        recorder.set_retained_buffer_write(2, &self.buffers.output, 0);
        record_shape(recorder, self.shape, self.row_shape, self.num_active_rows_key);
        dispatch_values(recorder, self.shape.num_values);
    }
}

impl Operator for CaptureReplayInvocation {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        recorder.set_retained_pipeline_state(&self.pipeline);
        recorder.set_retained_buffer_read(0, &self.residual.buffers.lhs, 0);
        recorder.set_retained_buffer_read(1, &self.residual.buffers.rhs, 0);
        recorder.set_retained_buffer_write(2, &self.residual.buffers.output, 0);
        recorder.set_retained_buffer_write(3, &self.capture.buffer, 0);
        let row_shape = self.row_shape();
        match self.residual.num_active_rows_key {
            Some(key) => recorder.bind_u32(4, key, 1, row_shape.num_total_rows),
            None => recorder.set_u32(4, row_shape.num_total_rows),
        }
        recorder.set_u32(5, row_shape.num_columns / 4);
        recorder.set_u32(6, self.capture.num_destination_columns / 4);
        recorder.set_u32(7, self.capture.column_start / 4);
        let num_vectors = self.residual.shape.num_values as usize / 4;
        let required_threads = KERNEL_CONSTANTS.thread_block.required_threads as usize;
        recorder.dispatch_threadblocks((num_vectors.div_ceil(required_threads), 1, 1), (required_threads, 1, 1));
    }
}

impl<'a> CaptureTarget<'a> {
    /// Selects the destination columns for every complete residual row.
    ///
    /// `num_destination_columns` and `columns` are element counts and tensor
    /// coordinates, not byte offsets.
    /// This constructor verifies that the range is non-empty and contained in
    /// a destination row. The range width, destination column count, and column
    /// start must be multiples of four. The range width must equal the
    /// residual source column count. Recording asserts this invariant together with the
    /// no-alias and capacity contracts.
    pub fn columns(buffer: &'a Buffer, num_destination_columns: u32, columns: Range<u32>) -> Self {
        assert!(
            num_destination_columns > 0,
            "residual-add capture destination must have columns"
        );
        assert!(
            columns.start < columns.end,
            "residual-add capture column range must be non-empty"
        );
        assert!(
            columns.end <= num_destination_columns,
            "residual-add capture columns must be within the row"
        );
        let column_width = columns.end - columns.start;
        assert!(
            column_width.is_multiple_of(4),
            "unsupported residual-add capture layout: BF16 capture width must be divisible by four"
        );
        assert!(
            num_destination_columns.is_multiple_of(4) && columns.start.is_multiple_of(4),
            "unsupported residual-add capture layout: BF16 capture requires aligned column count and column start"
        );
        Self {
            buffer,
            num_destination_columns,
            column_start: columns.start,
            column_end: columns.end,
        }
    }
}

impl Invocation<'_> {
    pub fn into_replay_op(self) -> ReplayOp {
        ReplayOp {
            pipeline: self.kernel.as_raw_retained(),
            config: self.config,
            shape: self.shape,
            buffers: OwnedBuffers {
                lhs: self.buffers.lhs.as_raw_retained(),
                lhs_len_bytes: self.buffers.lhs.len_bytes(),
                rhs: self.buffers.rhs.as_raw_retained(),
                rhs_len_bytes: self.buffers.rhs.len_bytes(),
                output: self.buffers.output.as_raw_retained(),
                output_len_bytes: self.buffers.output.len_bytes(),
            },
            row_shape: self.row_shape,
            num_active_rows_key: self.num_active_rows_key,
        }
    }

    pub fn into_capture_replay_op(self, capture: CaptureTarget<'_>) -> CaptureReplayOp {
        assert!(
            self.row_shape.is_some(),
            "residual-add capture requires an explicit row shape"
        );
        CaptureReplayOp {
            residual: self.into_replay_op(),
            capture: OwnedCaptureTarget {
                buffer: capture.buffer.as_raw_retained(),
                buffer_len_bytes: capture.buffer.len_bytes(),
                num_destination_columns: capture.num_destination_columns,
                column_start: capture.column_start,
                column_end: capture.column_end,
            },
        }
    }

    fn validate(&self) {
        if let Some(row_shape) = self.row_shape {
            assert_eq!(self.shape.num_values, row_shape.num_values());
        }
        assert!(self.num_active_rows_key.is_none() || self.row_shape.is_some());
        assert!(self.buffers.lhs.len_bytes() >= self.config.lhs_bytes(self.shape));
        assert!(self.buffers.rhs.len_bytes() >= self.config.rhs_bytes(self.shape));
        assert!(self.buffers.output.len_bytes() >= self.config.output_bytes(self.shape));
    }
}

impl ReplayOp {
    pub fn into_replay(self) -> ReplayInvocation {
        ReplayInvocation {
            pipeline: self.pipeline,
            config: self.config,
            shape: self.shape,
            buffers: self.buffers,
            row_shape: self.row_shape,
            num_active_rows_key: self.num_active_rows_key,
        }
    }
}

impl CaptureReplayOp {
    pub fn into_replay(self) -> CaptureReplayInvocation {
        let device = Device::from_raw_retained(self.residual.buffers.lhs.device());
        CaptureReplayInvocation {
            pipeline: CompiledKernel::new(&device, RESIDUAL_ADD_SOURCE, "residual_add_capture_bf16_vec4")
                .as_raw_retained(),
            residual: self.residual,
            capture: self.capture,
        }
    }
}

impl ReplayInvocation {
    fn validate(&self) {
        if let Some(row_shape) = self.row_shape {
            assert_eq!(self.shape.num_values, row_shape.num_values());
        }
        assert!(self.num_active_rows_key.is_none() || self.row_shape.is_some());
        assert!(self.buffers.lhs_len_bytes >= self.config.lhs_bytes(self.shape));
        assert!(self.buffers.rhs_len_bytes >= self.config.rhs_bytes(self.shape));
        assert!(self.buffers.output_len_bytes >= self.config.output_bytes(self.shape));
    }
}

impl CaptureReplayInvocation {
    fn row_shape(&self) -> RowShape {
        self.residual
            .row_shape
            .expect("residual-add capture requires an explicit row shape")
    }

    fn validate(&self) {
        self.residual.config.validate();
        self.residual.shape.validate();
        let row_shape = self.row_shape();
        assert_eq!(self.residual.shape.num_values, row_shape.num_values());
        assert_eq!(self.residual.config, Config::bf16());
        assert_eq!(
            self.capture.column_end - self.capture.column_start,
            row_shape.num_columns,
            "residual-add capture column count must match the residual source column count"
        );
        assert!(row_shape.num_columns.is_multiple_of(4));
        let required_values = (row_shape.num_total_rows as usize - 1)
            .checked_mul(self.capture.num_destination_columns as usize)
            .and_then(|value| value.checked_add(self.capture.column_end as usize))
            .expect("residual-add capture value count must fit usize");
        let required_bytes = required_values
            .checked_mul(Dtype::Bfloat16.item_size())
            .expect("residual-add capture byte count must fit usize");
        assert!(self.capture.buffer_len_bytes >= required_bytes);
        assert!(self.residual.buffers.lhs_len_bytes >= self.residual.config.lhs_bytes(self.residual.shape));
        assert!(self.residual.buffers.rhs_len_bytes >= self.residual.config.rhs_bytes(self.residual.shape));
        assert!(self.residual.buffers.output_len_bytes >= self.residual.config.output_bytes(self.residual.shape));
        for other in [
            &self.residual.buffers.lhs,
            &self.residual.buffers.rhs,
            &self.residual.buffers.output,
        ] {
            assert!(
                !std::ptr::eq(Retained::as_ptr(&self.capture.buffer), Retained::as_ptr(other)),
                "residual-add capture output must not alias a residual-add buffer"
            );
        }
    }
}

fn record_shape(
    recorder: &CommandRecorder,
    shape: Shape,
    row_shape: Option<RowShape>,
    num_active_rows_key: Option<ReplayParameterKey>,
) {
    match (row_shape, num_active_rows_key) {
        (Some(row_shape), Some(key)) => {
            recorder.bind_u32(3, key, 1, row_shape.num_total_rows);
            recorder.set_u32(4, row_shape.num_columns);
        },
        (Some(row_shape), None) => {
            recorder.set_u32(3, row_shape.num_total_rows);
            recorder.set_u32(4, row_shape.num_columns);
        },
        (None, None) => {
            recorder.set_u32(3, shape.num_values);
            recorder.set_u32(4, 1);
        },
        (None, Some(_)) => panic!("residual-add replay parameter requires a row shape"),
    }
}

fn dispatch_values(recorder: &CommandRecorder, num_values: u32) {
    let required_threads = KERNEL_CONSTANTS.thread_block.required_threads as usize;
    recorder.dispatch_threadblocks(
        ((num_values as usize).div_ceil(required_threads), 1, 1),
        (required_threads, 1, 1),
    );
}

fn residual_add_function_name(config: Config) -> &'static str {
    match (config.lhs_dtype, config.rhs_dtype, config.output_dtype) {
        (Dtype::Float32, Dtype::Float32, Dtype::Float32) => "residual_add_f32",
        (Dtype::Bfloat16, Dtype::Bfloat16, Dtype::Bfloat16) => "residual_add_bf16",
        (Dtype::Bfloat16, Dtype::Float32, Dtype::Bfloat16) => "residual_add_bf16_f32_to_bf16",
        (lhs_dtype, rhs_dtype, output_dtype) => {
            panic!(
                "unsupported residual-add dtype combination: lhs={lhs_dtype:?}, rhs={rhs_dtype:?}, \
                 output={output_dtype:?}"
            )
        },
    }
}
