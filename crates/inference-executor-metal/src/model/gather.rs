use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::row_gather;

pub struct Gather {
    compute: row_gather::Kernel,
}

impl Gather {
    pub fn new(device: &Device, hidden_dim: u32) -> Self {
        Self {
            compute: row_gather::Kernel::new(
                device,
                row_gather::Config {
                    num_cols: hidden_dim,
                    dtype: Dtype::Bfloat16,
                },
            ),
        }
    }

    pub fn record<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        num_total_rows: u32,
        num_active_rows: ReplayU32,
        input: &'a Buffer,
        row_indices: &'a Buffer,
        output: &'a Buffer,
    ) {
        let shape = row_gather::Shape { num_total_rows };
        let buffers = row_gather::Buffers {
            input,
            row_indices,
            output,
        };
        let invocation = self.compute.invoke(shape, num_active_rows, buffers);
        recorder.record_with_barrier_before(ReplayOp::opaque(invocation));
    }
}
