use inference_backend_metal::components::dense_mlp;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::mlp::dense::DenseMLPCore;

use crate::def::layer::ReplayLayer;
use crate::def::quantized_affine::QuantizedAffineLayout;
use crate::def::replay_op::ReplayOp;
use crate::mlp::dense::scratch::DenseMLPScratchBindings;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseMLPMetalConfig {
    pub gate_up: QuantizedAffineLayout,
    pub down: QuantizedAffineLayout,
    pub io_dtype: Dtype,
}

impl DenseMLPMetalConfig {
    pub fn validate(self) {
        self.gate_up.validate();
        self.down.validate();
        match self.io_dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 dense MLP model boundary is not supported"),
            dtype => panic!("unsupported dense MLP model boundary dtype {dtype:?}"),
        }
    }
}

pub struct DenseMLP {
    compute: dense_mlp::Compute,
}

#[derive(Clone, Copy)]
pub struct DenseMLPInput<'a> {
    pub num_total_tokens: u32,
    pub num_active_tokens: ReplayU32,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub scratch: DenseMLPScratchBindings<'a>,
    pub weights: dense_mlp::Weights<'a>,
}

impl DenseMLP {
    pub fn new(device: &Device, core: DenseMLPCore, config: DenseMLPMetalConfig) -> Self {
        core.validate();
        config.validate();
        Self {
            compute: dense_mlp::Compute::new(device, compute_config(&core, config)),
        }
    }

    pub fn replay_topology(&self, num_total_tokens: u32) -> dense_mlp::ReplayTopology {
        self.compute.topology(num_total_tokens)
    }

    pub fn replay_topology_boundaries(&self) -> Box<[u32]> {
        self.compute.topology_boundaries()
    }
}

impl ReplayLayer for DenseMLP {
    type Input<'a> = DenseMLPInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let buffers = dense_mlp::Buffers {
            hidden_state: input.hidden_state,
            next_hidden_state: input.next_hidden_state,
        };
        let scratch = dense_mlp::Scratch {
            swiglu: input.scratch.swiglu,
        };
        let invocation = self.compute.invoke(
            dense_mlp::Shape {
                num_total_tokens: input.num_total_tokens,
            },
            input.num_active_tokens,
            buffers,
            scratch,
            input.weights,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(invocation));
        input.next_hidden_state
    }
}

fn compute_config(core: &DenseMLPCore, config: DenseMLPMetalConfig) -> dense_mlp::Config {
    dense_mlp::Config {
        hidden_dim: core.hidden_dim.try_into().expect("dense MLP hidden_dim must fit u32"),
        intermediate_dim: core
            .intermediate_dim
            .try_into()
            .expect("dense MLP intermediate_dim must fit u32"),
        gate_up_group_size: config.gate_up.group_size,
        gate_up_bits: config.gate_up.bits,
        gate_up_scale_bias_dtype: config.gate_up.scale_bias_dtype,
        down_group_size: config.down.group_size,
        down_bits: config.down.bits,
        down_scale_bias_dtype: config.down.scale_bias_dtype,
        dtype: config.io_dtype,
    }
}
