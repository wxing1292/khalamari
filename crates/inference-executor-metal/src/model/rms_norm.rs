use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::rms_norm;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;

pub struct RMSNorm {
    weight: Option<Buffer>,
    compute: rms_norm::Compute,
}

impl RMSNorm {
    pub fn new(device: &Device, hidden_dim: usize, eps: f32) -> Self {
        Self::new_with_weight_dtype(device, hidden_dim, eps, Dtype::Bfloat16)
    }

    pub fn new_with_weight_dtype(device: &Device, hidden_dim: usize, eps: f32, weight_dtype: Dtype) -> Self {
        assert!(hidden_dim > 0, "RMS norm hidden dimension must be positive");
        assert!(eps.is_finite() && eps > 0.0, "RMS norm epsilon must be positive");
        let hidden_dim = hidden_dim.try_into().expect("RMS norm hidden dimension must fit u32");
        let config = match weight_dtype {
            Dtype::Bfloat16 => rms_norm::Config::bf16(hidden_dim, eps),
            Dtype::Float32 => rms_norm::Config::bf16_with_f32_weight(hidden_dim, eps),
            dtype => panic!("unsupported RMS norm weight dtype {dtype:?}"),
        };
        Self {
            weight: None,
            compute: rms_norm::Compute::new(device, config),
        }
    }

    pub fn load_weights(&mut self, weight: Buffer) {
        assert!(self.weight.is_none(), "RMS norm weights are already loaded");
        self.weight = Some(weight);
    }

    pub fn unload_weights(&mut self) {
        assert!(self.weight.is_some(), "RMS norm weights are not loaded");
        self.weight.take();
    }

    fn weight(&self) -> &Buffer {
        self.weight
            .as_ref()
            .expect("RMS norm weights must be loaded before execution")
    }

    fn invocation<'a>(
        &'a self,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        input: &'a Buffer,
        output: &'a Buffer,
    ) -> rms_norm::Invocation<'a> {
        let shape = rms_norm::Shape { num_total_tokens };
        let buffers = rms_norm::Buffers {
            input,
            weight: self.weight(),
            output,
        };
        self.compute.invoke(shape, num_active_tokens, buffers)
    }

    pub fn record<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        input: &'a Buffer,
        output: &'a Buffer,
    ) {
        recorder.record(ReplayOp::rms_norm(self.invocation(
            num_total_tokens,
            num_active_tokens,
            input,
            output,
        )));
    }

    pub fn record_with_barrier<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        input: &'a Buffer,
        output: &'a Buffer,
    ) {
        recorder.record_with_barrier_before(ReplayOp::rms_norm(self.invocation(
            num_total_tokens,
            num_active_tokens,
            input,
            output,
        )));
    }
}
