use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLComputePipelineState;
use objc2_metal::MTLResource;

use crate::components::residual_add;
use crate::components::rms_norm;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayU32;

const RESIDUAL_ADD_RMS_NORM_SOURCE: &str = include_str!("metal/residual_add_rms_norm.metal");

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub hidden_dim: u32,
    pub eps: f32,
    pub io_dtype: Dtype,
}

impl Config {
    pub fn f32(hidden_dim: u32, eps: f32) -> Self {
        Self {
            hidden_dim,
            eps,
            io_dtype: Dtype::Float32,
        }
    }

    /// Creates a BF16 configuration. `hidden_dim` must be divisible by 4.
    pub fn bf16(hidden_dim: u32, eps: f32) -> Self {
        Self {
            hidden_dim,
            eps,
            io_dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.hidden_dim > 0);
        assert!(self.eps.is_finite() && self.eps > 0.0);
        assert!(matches!(self.io_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert!(
            self.io_dtype != Dtype::Bfloat16 || self.hidden_dim.is_multiple_of(4),
            "BF16 residual-add RMSNorm hidden_dim must be divisible by 4"
        );
    }

    pub fn num_values(self, shape: Shape) -> usize {
        self.validate();
        shape.validate();
        (shape.num_total_tokens as usize)
            .checked_mul(self.hidden_dim as usize)
            .expect("residual-add RMSNorm value count must fit usize")
    }

    pub fn bytes(self, shape: Shape) -> usize {
        self.num_values(shape)
            .checked_mul(self.io_dtype.item_size())
            .expect("residual-add RMSNorm byte length must fit usize")
    }

    pub fn weight_bytes(self) -> usize {
        self.validate();
        (self.hidden_dim as usize)
            .checked_mul(self.io_dtype.item_size())
            .expect("residual-add RMSNorm weight byte length must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_tokens: u32,
}

impl Shape {
    pub fn validate(self) {
        assert!(self.num_total_tokens > 0);
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub lhs: &'a Buffer,
    pub rhs: &'a Buffer,
    pub weight: &'a Buffer,
    pub residual_output: &'a Buffer,
    pub norm_output: &'a Buffer,
}

/// Fused residual-add and RMSNorm data flow:
///
/// ```text
/// buffers.lhs ----\
///                  +--> residual add --> buffers.residual_output --\
/// buffers.rhs ----/                                                 +--> RMSNorm --> buffers.norm_output
/// buffers.weight ---------------------------------------------------/
/// ```
///
/// The capture variant also writes the residual-add result to the selected
/// columns in `capture_output`.
pub struct Compute {
    config: Config,
    constants: KernelConstants,
    kernel: CompiledKernel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelKind {
    Scalar,
    Bf16Vectorized,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants {
    kind: KernelKind,
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn new(config: Config, kind: KernelKind) -> Self {
        validate_kernel_kind(config, kind);
        let constants = Self {
            kind,
            thread_block: ThreadBlockConstants { required_threads: 1024 },
        };
        if kind == KernelKind::Scalar {
            // The scalar shader advances uint r by four values per thread.
            let loop_width = constants.thread_block.required_threads as u64 * 4;
            let padded_hidden_dim = (config.hidden_dim as u64).div_ceil(loop_width) * loop_width;
            assert!(
                padded_hidden_dim <= u32::MAX as u64,
                "scalar residual-add RMSNorm padded hidden dimension must fit the shader u32 loop domain"
            );
        }
        constants
    }
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        Self::new_with_kind(device, config, selected_kernel_kind(config))
    }

    /// Creates one exact backend path for backend tests and benchmarks.
    ///
    /// Model and executor code must use [`Self::new`].
    pub fn new_with_kind(device: &Device, config: Config, kind: KernelKind) -> Self {
        config.validate();
        let constants = KernelConstants::new(config, kind);
        Self {
            config,
            constants,
            kernel: CompiledKernel::new(
                device,
                RESIDUAL_ADD_RMS_NORM_SOURCE,
                residual_add_rms_norm_function_name(config, constants.kind),
            ),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, num_active_tokens: ReplayU32, buffers: Buffers<'a>) -> Invocation<'a> {
        shape.validate();
        Invocation {
            kernel: &self.kernel,
            config: self.config,
            constants: self.constants,
            shape,
            buffers,
            num_active_tokens_key: active_key(shape.num_total_tokens, num_active_tokens),
        }
    }
}

fn active_key(num_total_tokens: u32, num_active_tokens: ReplayU32) -> Option<ReplayParameterKey> {
    match num_active_tokens {
        ReplayU32::Fixed(num_active_tokens) => {
            assert_eq!(num_active_tokens, num_total_tokens);
            None
        },
        ReplayU32::Parameter(key) => Some(key),
    }
}

pub struct Invocation<'a> {
    kernel: &'a CompiledKernel,
    config: Config,
    constants: KernelConstants,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

pub struct ReplayInvocation {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    config: Config,
    constants: KernelConstants,
    shape: Shape,
    buffers: OwnedBuffers,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

pub struct CaptureReplayInvocation {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    config: Config,
    constants: KernelConstants,
    shape: Shape,
    buffers: CaptureOwnedBuffers,
    capture_num_columns: u32,
    capture_column_start: u32,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

#[derive(Clone)]
struct OwnedBuffers {
    lhs: Retained<ProtocolObject<dyn MTLBuffer>>,
    lhs_len_bytes: usize,
    rhs: Retained<ProtocolObject<dyn MTLBuffer>>,
    rhs_len_bytes: usize,
    weight: Retained<ProtocolObject<dyn MTLBuffer>>,
    weight_len_bytes: usize,
    residual_output: Retained<ProtocolObject<dyn MTLBuffer>>,
    residual_output_len_bytes: usize,
    norm_output: Retained<ProtocolObject<dyn MTLBuffer>>,
    norm_output_len_bytes: usize,
}

struct CaptureOwnedBuffers {
    lhs: Retained<ProtocolObject<dyn MTLBuffer>>,
    lhs_len_bytes: usize,
    rhs: Retained<ProtocolObject<dyn MTLBuffer>>,
    rhs_len_bytes: usize,
    weight: Retained<ProtocolObject<dyn MTLBuffer>>,
    weight_len_bytes: usize,
    residual_output: Retained<ProtocolObject<dyn MTLBuffer>>,
    residual_output_len_bytes: usize,
    capture_output: Retained<ProtocolObject<dyn MTLBuffer>>,
    capture_output_len_bytes: usize,
    norm_output: Retained<ProtocolObject<dyn MTLBuffer>>,
    norm_output_len_bytes: usize,
}

impl OwnedBuffers {
    #[allow(clippy::too_many_arguments)]
    fn new(
        lhs: Retained<ProtocolObject<dyn MTLBuffer>>,
        lhs_len_bytes: usize,
        rhs: Retained<ProtocolObject<dyn MTLBuffer>>,
        rhs_len_bytes: usize,
        weight: Retained<ProtocolObject<dyn MTLBuffer>>,
        weight_len_bytes: usize,
        residual_output: Retained<ProtocolObject<dyn MTLBuffer>>,
        residual_output_len_bytes: usize,
        norm_output: Retained<ProtocolObject<dyn MTLBuffer>>,
        norm_output_len_bytes: usize,
    ) -> Self {
        Self {
            lhs,
            lhs_len_bytes,
            rhs,
            rhs_len_bytes,
            weight,
            weight_len_bytes,
            residual_output,
            residual_output_len_bytes,
            norm_output,
            norm_output_len_bytes,
        }
    }
}

impl CaptureOwnedBuffers {
    #[allow(clippy::too_many_arguments)]
    fn new(
        lhs: Retained<ProtocolObject<dyn MTLBuffer>>,
        lhs_len_bytes: usize,
        rhs: Retained<ProtocolObject<dyn MTLBuffer>>,
        rhs_len_bytes: usize,
        weight: Retained<ProtocolObject<dyn MTLBuffer>>,
        weight_len_bytes: usize,
        residual_output: Retained<ProtocolObject<dyn MTLBuffer>>,
        residual_output_len_bytes: usize,
        capture_output: Retained<ProtocolObject<dyn MTLBuffer>>,
        capture_output_len_bytes: usize,
        norm_output: Retained<ProtocolObject<dyn MTLBuffer>>,
        norm_output_len_bytes: usize,
    ) -> Self {
        Self {
            lhs,
            lhs_len_bytes,
            rhs,
            rhs_len_bytes,
            weight,
            weight_len_bytes,
            residual_output,
            residual_output_len_bytes,
            capture_output,
            capture_output_len_bytes,
            norm_output,
            norm_output_len_bytes,
        }
    }
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.lhs, 0);
        recorder.set_buffer_read(1, self.buffers.rhs, 0);
        recorder.set_buffer_read(2, self.buffers.weight, 0);
        recorder.set_buffer_write(3, self.buffers.residual_output, 0);
        recorder.set_buffer_write(4, self.buffers.norm_output, 0);
        record_num_active_tokens(recorder, 5, self.shape.num_total_tokens, self.num_active_tokens_key);
        recorder.set_u32(6, self.config.hidden_dim);
        recorder.set_f32(7, self.config.eps);
        recorder.dispatch_threadblocks(
            (self.shape.num_total_tokens as usize, 1, 1),
            (self.constants.thread_block.required_threads as usize, 1, 1),
        );
    }
}

impl Operator for ReplayInvocation {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        recorder.set_retained_pipeline_state(&self.pipeline);
        recorder.set_retained_buffer_read(0, &self.buffers.lhs, 0);
        recorder.set_retained_buffer_read(1, &self.buffers.rhs, 0);
        recorder.set_retained_buffer_read(2, &self.buffers.weight, 0);
        recorder.set_retained_buffer_write(3, &self.buffers.residual_output, 0);
        recorder.set_retained_buffer_write(4, &self.buffers.norm_output, 0);
        record_num_active_tokens(recorder, 5, self.shape.num_total_tokens, self.num_active_tokens_key);
        recorder.set_u32(6, self.config.hidden_dim);
        recorder.set_f32(7, self.config.eps);
        recorder.dispatch_threadblocks(
            (self.shape.num_total_tokens as usize, 1, 1),
            (self.constants.thread_block.required_threads as usize, 1, 1),
        );
    }
}

impl Operator for CaptureReplayInvocation {
    fn record(self, recorder: &CommandRecorder<'_>) {
        self.validate();
        recorder.set_retained_pipeline_state(&self.pipeline);
        recorder.set_retained_buffer_read(0, &self.buffers.lhs, 0);
        recorder.set_retained_buffer_read(1, &self.buffers.rhs, 0);
        recorder.set_retained_buffer_read(2, &self.buffers.weight, 0);
        recorder.set_retained_buffer_write(3, &self.buffers.residual_output, 0);
        recorder.set_retained_buffer_write(4, &self.buffers.capture_output, 0);
        recorder.set_retained_buffer_write(5, &self.buffers.norm_output, 0);
        record_num_active_tokens(recorder, 6, self.shape.num_total_tokens, self.num_active_tokens_key);
        recorder.set_u32(7, self.config.hidden_dim);
        recorder.set_u32(8, self.capture_num_columns / 4);
        recorder.set_u32(9, self.capture_column_start / 4);
        recorder.set_f32(10, self.config.eps);
        recorder.dispatch_threadblocks(
            (self.shape.num_total_tokens as usize, 1, 1),
            (self.constants.thread_block.required_threads as usize, 1, 1),
        );
    }
}

impl Invocation<'_> {
    fn validate(&self) {
        assert!(self.buffers.lhs.len_bytes() >= self.config.bytes(self.shape));
        assert!(self.buffers.rhs.len_bytes() >= self.config.bytes(self.shape));
        assert!(self.buffers.weight.len_bytes() >= self.config.weight_bytes());
        assert!(self.buffers.residual_output.len_bytes() >= self.config.bytes(self.shape));
        assert!(self.buffers.norm_output.len_bytes() >= self.config.bytes(self.shape));
    }
}

impl ReplayInvocation {
    pub fn is_residual_add_rms_norm_fusion_compatible(
        residual_add: &residual_add::ReplayOp,
        rms_norm: &rms_norm::ReplayOp,
    ) -> bool {
        let Some(residual_values) = rms_norm.shape.num_total_tokens.checked_mul(rms_norm.config.hidden_dim) else {
            return false;
        };
        let row_shape_matches = residual_add.row_shape.is_none_or(|shape| {
            shape.num_total_rows == rms_norm.shape.num_total_tokens && shape.num_columns == rms_norm.config.hidden_dim
        });
        let replay_domain_matches = match (residual_add.num_active_rows_key, rms_norm.num_active_tokens_key) {
            (None, None) => row_shape_matches,
            (Some(residual_key), Some(rms_norm_key)) => residual_key == rms_norm_key && row_shape_matches,
            _ => false,
        };
        replay_domain_matches
            && residual_add.shape.num_values == residual_values
            && rms_norm.config.weight_dtype == rms_norm.config.io_dtype
            && residual_add.config.lhs_dtype == rms_norm.config.io_dtype
            && residual_add.config.rhs_dtype == rms_norm.config.io_dtype
            && residual_add.config.output_dtype == rms_norm.config.io_dtype
            && std::ptr::eq(
                Retained::as_ptr(&residual_add.buffers.output),
                Retained::as_ptr(&rms_norm.buffers.input),
            )
    }

    pub fn fuse_residual_add_rms_norm(residual_add: residual_add::ReplayOp, rms_norm: rms_norm::ReplayOp) -> Self {
        assert!(
            Self::is_residual_add_rms_norm_fusion_compatible(&residual_add, &rms_norm),
            "residual-add output must be the fused RMSNorm input"
        );
        let config = Config {
            hidden_dim: rms_norm.config.hidden_dim,
            eps: rms_norm.config.eps,
            io_dtype: rms_norm.config.io_dtype,
        };
        let shape = Shape {
            num_total_tokens: rms_norm.shape.num_total_tokens,
        };
        let buffers = OwnedBuffers::new(
            residual_add.buffers.lhs,
            residual_add.buffers.lhs_len_bytes,
            residual_add.buffers.rhs,
            residual_add.buffers.rhs_len_bytes,
            rms_norm.buffers.weight,
            rms_norm.buffers.weight_len_bytes,
            residual_add.buffers.output,
            residual_add.buffers.output_len_bytes,
            rms_norm.buffers.output,
            rms_norm.buffers.output_len_bytes,
        );
        Self::new(config, shape, buffers, rms_norm.num_active_tokens_key)
    }

    fn new(
        config: Config,
        shape: Shape,
        buffers: OwnedBuffers,
        num_active_tokens_key: Option<ReplayParameterKey>,
    ) -> Self {
        let device = Device::from_raw_retained(buffers.lhs.device());
        let constants = KernelConstants::new(config, selected_kernel_kind(config));
        Self {
            pipeline: CompiledKernel::new(
                &device,
                RESIDUAL_ADD_RMS_NORM_SOURCE,
                residual_add_rms_norm_function_name(config, constants.kind),
            )
            .as_raw_retained(),
            config,
            constants,
            shape,
            buffers,
            num_active_tokens_key,
        }
    }

    fn validate(&self) {
        assert!(self.buffers.lhs_len_bytes >= self.config.bytes(self.shape));
        assert!(self.buffers.rhs_len_bytes >= self.config.bytes(self.shape));
        assert!(self.buffers.weight_len_bytes >= self.config.weight_bytes());
        assert!(self.buffers.residual_output_len_bytes >= self.config.bytes(self.shape));
        assert!(self.buffers.norm_output_len_bytes >= self.config.bytes(self.shape));
    }
}

impl CaptureReplayInvocation {
    pub fn is_residual_add_capture_rms_norm_fusion_compatible(
        residual_add: &residual_add::CaptureReplayOp,
        rms_norm: &rms_norm::ReplayOp,
    ) -> bool {
        residual_add.residual.config == residual_add::Config::bf16()
            && rms_norm.config.io_dtype == Dtype::Bfloat16
            && residual_add.residual.row_shape.is_some()
            && residual_add.capture.column_end - residual_add.capture.column_start == rms_norm.config.hidden_dim
            && ReplayInvocation::is_residual_add_rms_norm_fusion_compatible(&residual_add.residual, rms_norm)
    }

    pub fn fuse_residual_add_capture_rms_norm(
        residual_add: residual_add::CaptureReplayOp,
        rms_norm: rms_norm::ReplayOp,
    ) -> Self {
        assert!(
            Self::is_residual_add_capture_rms_norm_fusion_compatible(&residual_add, &rms_norm),
            "residual-add capture and RMSNorm must have compatible buffers, shapes, dtypes, and replay domains"
        );

        let config = Config {
            hidden_dim: rms_norm.config.hidden_dim,
            eps: rms_norm.config.eps,
            io_dtype: rms_norm.config.io_dtype,
        };
        let shape = Shape {
            num_total_tokens: rms_norm.shape.num_total_tokens,
        };
        let capture_num_columns = residual_add.capture.num_destination_columns;
        let capture_column_start = residual_add.capture.column_start;
        let buffers = CaptureOwnedBuffers::new(
            residual_add.residual.buffers.lhs,
            residual_add.residual.buffers.lhs_len_bytes,
            residual_add.residual.buffers.rhs,
            residual_add.residual.buffers.rhs_len_bytes,
            rms_norm.buffers.weight,
            rms_norm.buffers.weight_len_bytes,
            residual_add.residual.buffers.output,
            residual_add.residual.buffers.output_len_bytes,
            residual_add.capture.buffer,
            residual_add.capture.buffer_len_bytes,
            rms_norm.buffers.output,
            rms_norm.buffers.output_len_bytes,
        );
        Self::new(
            config,
            shape,
            buffers,
            capture_num_columns,
            capture_column_start,
            rms_norm.num_active_tokens_key,
        )
    }

    fn new(
        config: Config,
        shape: Shape,
        buffers: CaptureOwnedBuffers,
        capture_num_columns: u32,
        capture_column_start: u32,
        num_active_tokens_key: Option<ReplayParameterKey>,
    ) -> Self {
        let device = Device::from_raw_retained(buffers.lhs.device());
        let constants = KernelConstants::new(config, KernelKind::Bf16Vectorized);
        Self {
            pipeline: CompiledKernel::new(
                &device,
                RESIDUAL_ADD_RMS_NORM_SOURCE,
                residual_add_capture_rms_norm_function_name(config),
            )
            .as_raw_retained(),
            config,
            constants,
            shape,
            buffers,
            capture_num_columns,
            capture_column_start,
            num_active_tokens_key,
        }
    }

    fn validate(&self) {
        self.config.validate();
        self.shape.validate();
        assert_eq!(self.config.io_dtype, Dtype::Bfloat16);
        assert!(self.capture_num_columns >= self.config.hidden_dim);
        assert!(self.capture_column_start <= self.capture_num_columns - self.config.hidden_dim);
        assert!(
            self.capture_num_columns.is_multiple_of(4),
            "unsupported residual-add capture layout: BF16 capture column count must be divisible by four"
        );
        assert!(
            self.capture_column_start.is_multiple_of(4),
            "unsupported residual-add capture layout: BF16 capture column start must be divisible by four"
        );
        assert!(self.buffers.lhs_len_bytes >= self.config.bytes(self.shape));
        assert!(self.buffers.rhs_len_bytes >= self.config.bytes(self.shape));
        assert!(self.buffers.weight_len_bytes >= self.config.weight_bytes());
        assert!(self.buffers.residual_output_len_bytes >= self.config.bytes(self.shape));
        assert!(self.buffers.norm_output_len_bytes >= self.config.bytes(self.shape));
        let last_row_start = (self.shape.num_total_tokens as usize - 1)
            .checked_mul(self.capture_num_columns as usize)
            .expect("residual-add capture last-row offset must fit usize");
        let required_values = last_row_start
            .checked_add(self.capture_column_start as usize)
            .and_then(|value| value.checked_add(self.config.hidden_dim as usize))
            .expect("residual-add capture value count must fit usize");
        let required_bytes = required_values
            .checked_mul(Dtype::Bfloat16.item_size())
            .expect("residual-add capture byte count must fit usize");
        assert!(self.buffers.capture_output_len_bytes >= required_bytes);
        for other in [
            &self.buffers.lhs,
            &self.buffers.rhs,
            &self.buffers.weight,
            &self.buffers.residual_output,
            &self.buffers.norm_output,
        ] {
            assert!(
                !std::ptr::eq(Retained::as_ptr(&self.buffers.capture_output), Retained::as_ptr(other),),
                "residual-add capture output must not alias another fused residual/RMSNorm buffer"
            );
        }
    }
}

fn record_num_active_tokens(
    recorder: &CommandRecorder,
    binding_index: usize,
    num_total_tokens: u32,
    key: Option<ReplayParameterKey>,
) {
    match key {
        Some(key) => recorder.bind_u32(binding_index, key, 1, num_total_tokens),
        None => recorder.set_u32(binding_index, num_total_tokens),
    }
}

fn selected_kernel_kind(config: Config) -> KernelKind {
    match config.io_dtype {
        Dtype::Float32 => KernelKind::Scalar,
        Dtype::Bfloat16 => KernelKind::Bf16Vectorized,
        dtype => panic!("unsupported residual-add RMSNorm dtype {dtype:?}"),
    }
}

fn validate_kernel_kind(config: Config, kind: KernelKind) {
    if kind == KernelKind::Bf16Vectorized {
        assert_eq!(config.io_dtype, Dtype::Bfloat16);
        assert!(
            config.hidden_dim.is_multiple_of(4),
            "vectorized residual-add RMSNorm requires hidden_dim divisible by 4"
        );
    }
}

fn residual_add_rms_norm_function_name(config: Config, kind: KernelKind) -> &'static str {
    match (config.io_dtype, kind) {
        (Dtype::Float32, KernelKind::Scalar) => "residual_add_rms_norm_f32",
        (Dtype::Bfloat16, KernelKind::Scalar) => "residual_add_rms_norm_bf16",
        (Dtype::Bfloat16, KernelKind::Bf16Vectorized) => "residual_add_rms_norm_bf16_vec4",
        (dtype, kind) => panic!("unsupported residual-add RMSNorm kernel: dtype={dtype:?} kind={kind:?}"),
    }
}

fn residual_add_capture_rms_norm_function_name(config: Config) -> &'static str {
    match config.io_dtype {
        Dtype::Bfloat16 => "residual_add_capture_rms_norm_bf16_vec4",
        dtype => panic!("unsupported residual-add capture RMSNorm dtype {dtype:?}"),
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use half::bf16;

    use super::*;
    use crate::metal::ReplayU32;
    use crate::metal::Stream;

    #[test]
    #[should_panic(expected = "BF16 residual-add RMSNorm hidden_dim must be divisible by 4")]
    fn test_bf16_rejects_non_vector_width() {
        Config::bf16(6, 1.0e-6).validate();
    }

    #[test]
    fn test_bf16_fusion() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let tokens = 3;
        let hidden_dim = 128;
        let residual_add = residual_add::Compute::new(&device, residual_add::Config::bf16());
        let rms_norm = rms_norm::Compute::new(&device, rms_norm::Config::bf16(hidden_dim as u32, 1.0e-6));
        let config = Config::bf16(hidden_dim as u32, 1.0e-6);
        let fused_scalar = Compute::new_with_kind(&device, config, KernelKind::Scalar);
        let fused = Compute::new(&device, config);
        let shape = Shape {
            num_total_tokens: tokens as u32,
        };
        let num_values = tokens * hidden_dim;
        let lhs = bf16_buffer(&device, num_values, 13, -0.75);
        let rhs = bf16_buffer(&device, num_values, 17, -0.25);
        let weight = bf16_buffer(&device, hidden_dim, 5, 0.001);
        let unfused_residual = Buffer::new_zeroed(&device, num_values * size_of::<u16>());
        let unfused_norm = Buffer::new_zeroed(&device, num_values * size_of::<u16>());
        let fused_scalar_residual = Buffer::new_zeroed(&device, num_values * size_of::<u16>());
        let fused_scalar_norm = Buffer::new_zeroed(&device, num_values * size_of::<u16>());
        let fused_vec4_residual = Buffer::new_zeroed(&device, num_values * size_of::<u16>());
        let fused_vec4_norm = Buffer::new_zeroed(&device, num_values * size_of::<u16>());

        let mut builder = stream.create_replay_program();
        builder.record(residual_add.invoke_values(
            residual_add::Shape {
                num_values: num_values as u32,
            },
            residual_add::Buffers {
                lhs: &lhs,
                rhs: &rhs,
                output: &unfused_residual,
            },
        ));
        builder.record_with_barrier_before(rms_norm.invoke(
            rms_norm::Shape {
                num_total_tokens: tokens as u32,
            },
            ReplayU32::Fixed(tokens as u32),
            rms_norm::Buffers {
                input: &unfused_residual,
                weight: &weight,
                output: &unfused_norm,
            },
        ));
        stream.submit_replay(&builder.build()).wait();

        let mut builder = stream.create_replay_program();
        builder.record(fused_scalar.invoke(
            shape,
            ReplayU32::Fixed(shape.num_total_tokens),
            Buffers {
                lhs: &lhs,
                rhs: &rhs,
                weight: &weight,
                residual_output: &fused_scalar_residual,
                norm_output: &fused_scalar_norm,
            },
        ));
        stream.submit_replay(&builder.build()).wait();

        assert_eq!(
            unfused_residual.read_typed::<u16>(0, num_values),
            fused_scalar_residual.read_typed::<u16>(0, num_values)
        );
        assert_eq!(
            unfused_norm.read_typed::<u16>(0, num_values),
            fused_scalar_norm.read_typed::<u16>(0, num_values)
        );

        let mut builder = stream.create_replay_program();
        builder.record(fused.invoke(
            shape,
            ReplayU32::Fixed(shape.num_total_tokens),
            Buffers {
                lhs: &lhs,
                rhs: &rhs,
                weight: &weight,
                residual_output: &fused_vec4_residual,
                norm_output: &fused_vec4_norm,
            },
        ));
        stream.submit_replay(&builder.build()).wait();

        assert_eq!(
            unfused_residual.read_typed::<u16>(0, num_values),
            fused_vec4_residual.read_typed::<u16>(0, num_values)
        );
        assert_eq!(
            unfused_norm.read_typed::<u16>(0, num_values),
            fused_vec4_norm.read_typed::<u16>(0, num_values)
        );
    }

    fn bf16_buffer(device: &Device, len: usize, step: usize, base: f32) -> Buffer {
        let values = (0..len)
            .map(|index| bf16::from_f32(base + ((index * step) % 23) as f32 * 0.03125).to_bits())
            .collect::<Vec<_>>();
        Buffer::from_slice(device, &values)
    }
}
