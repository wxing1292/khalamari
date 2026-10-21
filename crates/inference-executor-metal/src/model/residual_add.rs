use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::residual_add;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayU32;

pub struct ResidualAdd {
    compute: residual_add::Compute,
}

impl ResidualAdd {
    pub fn new(device: &Device) -> Self {
        Self {
            compute: residual_add::Compute::new(device, residual_add::Config::bf16()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        num_total_rows: u32,
        num_columns: u32,
        num_active_rows: ReplayU32,
        lhs: &'a Buffer,
        rhs: &'a Buffer,
        residual_output: &'a Buffer,
    ) {
        let shape = residual_add::RowShape {
            num_total_rows,
            num_columns,
        };
        let buffers = residual_add::Buffers {
            lhs,
            rhs,
            output: residual_output,
        };
        let invocation = self.compute.invoke_rows(shape, num_active_rows, buffers);
        recorder.record_with_barrier_before(ReplayOp::residual_add(invocation));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_with_capture<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        num_total_rows: u32,
        num_columns: u32,
        num_active_rows: ReplayU32,
        lhs: &'a Buffer,
        rhs: &'a Buffer,
        residual_output: &'a Buffer,
        capture: residual_add::CaptureTarget<'a>,
    ) {
        let shape = residual_add::RowShape {
            num_total_rows,
            num_columns,
        };
        let buffers = residual_add::Buffers {
            lhs,
            rhs,
            output: residual_output,
        };
        let invocation = self.compute.invoke_rows(shape, num_active_rows, buffers);
        recorder.record_with_barrier_before(ReplayOp::residual_add_with_capture(invocation, capture));
    }
}
