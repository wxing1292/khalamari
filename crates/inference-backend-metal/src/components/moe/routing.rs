use std::mem::size_of;

use crate::components::assert_u32_index_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayU32;

const MOE_ROUTING_SOURCE: &str = include_str!("../metal/moe_routing.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants {
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn from_config(config: Config, max_threadblock_memory_bytes: usize) -> Self {
        config.validate();
        let constants = Self {
            thread_block: ThreadBlockConstants { required_threads: 256 },
        };
        // Candidate and selected arrays each store one F32 probability and one u32 expert ID.
        let memory_bytes = (u64::from(constants.thread_block.required_threads)
            + u64::from(config.num_experts_per_token))
            * (size_of::<f32>() + size_of::<u32>()) as u64;
        assert!(
            memory_bytes <= max_threadblock_memory_bytes as u64,
            "MoE routing top-k arrays must fit device shared memory"
        );
        constants
    }
}

/// Routes each token to its top-k experts from router probabilities.
///
/// The caller owns the preceding softmax stage. This kernel selects top-k
/// experts by the bf16 softmax probabilities and optionally renormalizes the
/// selected probabilities across the top-k set.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub num_experts: u32,
    pub num_experts_per_token: u32,
    pub norm_topk_prob: bool,
}

impl Config {
    pub fn validate(self) {
        assert!(self.num_experts > 0);
        assert!(self.num_experts_per_token > 0);
        assert!(self.num_experts_per_token <= self.num_experts);
    }

    pub fn num_routes(self, shape: Shape) -> usize {
        self.validate();
        shape.validate();
        checked_product(
            "MoE routing route count",
            &[shape.num_total_tokens as usize, self.num_experts_per_token as usize],
        )
    }

    fn num_router_prob_elements(self, shape: Shape) -> usize {
        self.validate();
        shape.validate();
        checked_product(
            "MoE routing probability element count",
            &[shape.num_total_tokens as usize, self.num_experts as usize],
        )
    }

    pub fn validate_shape(self, shape: Shape) {
        self.validate();
        shape.validate();
        assert_u32_index_domain(self.num_router_prob_elements(shape), "MoE routing probability elements");
        assert_u32_index_domain(self.num_routes(shape), "MoE routing routes");
    }

    pub fn router_probs_bytes(self, shape: Shape) -> usize {
        checked_product(
            "MoE routing probability byte length",
            &[self.num_router_prob_elements(shape), size_of::<u16>()],
        )
    }

    pub fn expert_indices_bytes(self, shape: Shape) -> usize {
        checked_product(
            "MoE routing expert-index byte length",
            &[self.num_routes(shape), size_of::<u32>()],
        )
    }

    pub fn expert_probs_bytes(self, shape: Shape) -> usize {
        checked_product(
            "MoE routing expert-probability byte length",
            &[self.num_routes(shape), size_of::<f32>()],
        )
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

pub struct Buffers<'a> {
    pub router_probs: &'a Buffer,
    pub expert_indices: &'a Buffer,
    pub expert_probs: &'a Buffer,
}

pub struct Compute {
    config: Config,
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Compute {
    pub fn new(device: &crate::metal::Device, config: Config) -> Self {
        let constants = KernelConstants::from_config(config, device.max_threadblock_memory_length());
        let source_constants = format!(
            "using namespace metal;\nconstant uint num_experts = {num_experts}u;\nconstant uint num_experts_per_token \
             = {num_experts_per_token}u;\nconstant bool norm_topk_prob = {norm_topk_prob};\nconstant uint \
             required_threads = {required_threads}u;\nconstant uint num_expert_groups = {num_expert_groups}u;",
            num_experts = config.num_experts,
            num_experts_per_token = config.num_experts_per_token,
            norm_topk_prob = config.norm_topk_prob,
            required_threads = constants.thread_block.required_threads,
            num_expert_groups = config.num_experts.div_ceil(constants.thread_block.required_threads),
        );
        let source = MOE_ROUTING_SOURCE.replacen("using namespace metal;", &source_constants, 1);
        let kernel = CompiledKernel::new(device, &source, "moe_route_topk");
        assert!(kernel.static_threadblock_memory_length() <= device.max_threadblock_memory_length());
        Self {
            config,
            constants,
            kernel,
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, num_active_tokens: ReplayU32, buffers: Buffers<'a>) -> Invocation<'a> {
        self.config.validate_shape(shape);
        validate_buffers(self.config, shape, &buffers);
        Invocation {
            kernel: self,
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
    kernel: &'a Compute,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(&self.kernel.kernel);
        recorder.set_buffer_read(0, self.buffers.router_probs, 0);
        recorder.set_buffer_write(1, self.buffers.expert_indices, 0);
        recorder.set_buffer_write(2, self.buffers.expert_probs, 0);
        match self.num_active_tokens_key {
            Some(key) => recorder.bind_u32(3, key, 1, self.shape.num_total_tokens),
            None => recorder.set_u32(3, self.shape.num_total_tokens),
        }
        recorder.dispatch_threadblocks(
            (self.shape.num_total_tokens as usize, 1, 1),
            (self.kernel.constants.thread_block.required_threads as usize, 1, 1),
        );
    }
}

fn validate_buffers(config: Config, shape: Shape, buffers: &Buffers<'_>) {
    let router_probs_bytes = config.router_probs_bytes(shape);
    let expert_indices_bytes = config.expert_indices_bytes(shape);
    let expert_probs_bytes = config.expert_probs_bytes(shape);
    assert!(
        buffers.router_probs.len_bytes() >= router_probs_bytes,
        "MoE routing router_probs buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        router_probs_bytes,
        buffers.router_probs.len_bytes()
    );
    assert!(
        buffers.expert_indices.len_bytes() >= expert_indices_bytes,
        "MoE routing expert_indices buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        expert_indices_bytes,
        buffers.expert_indices.len_bytes()
    );
    assert!(
        buffers.expert_probs.len_bytes() >= expert_probs_bytes,
        "MoE routing expert_probs buffer too short: shape={shape:?} required_bytes={} buffer_bytes={}",
        expert_probs_bytes,
        buffers.expert_probs.len_bytes()
    );
}

#[cfg(test)]
#[path = "routing_test.rs"]
mod tests;
