use std::mem::size_of;

use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayU32;

const MOE_EXPERT_MAJOR_SOURCE: &str = include_str!("../metal/moe_expert_major.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants {
    layout_clear: ThreadBlockConstants,
    layout_count: ThreadBlockConstants,
    layout_prefix: ThreadBlockConstants,
    layout_scatter: ThreadBlockConstants,
    pack_input: ThreadBlockConstants,
    scatter_output: ThreadBlockConstants,
}

impl KernelConstants {
    fn current() -> Self {
        let elementwise = ThreadBlockConstants { required_threads: 256 };
        Self {
            layout_clear: elementwise,
            layout_count: elementwise,
            layout_prefix: ThreadBlockConstants { required_threads: 1 },
            layout_scatter: elementwise,
            pack_input: elementwise,
            scatter_output: elementwise,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub num_experts: u32,
    pub num_experts_per_token: u32,
    pub hidden_dim: u32,
    pub dtype: Dtype,
}

impl Config {
    pub fn bf16(num_experts: u32, num_experts_per_token: u32, hidden_dim: u32) -> Self {
        Self {
            num_experts,
            num_experts_per_token,
            hidden_dim,
            dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.num_experts > 0);
        assert!(self.num_experts_per_token > 0);
        assert!(self.num_experts_per_token <= self.num_experts);
        assert!(self.hidden_dim > 0);
        assert_eq!(self.dtype, Dtype::Bfloat16);
    }

    pub fn validate_shape(self, shape: Shape) {
        self.validate();
        shape.validate();
        self.num_routes(shape);
        assert_u32_count_domain(
            self.num_route_hidden_elements(shape),
            "MoE expert-major routed-hidden elements",
        );
        assert_u32_count_domain(
            self.num_token_hidden_elements(shape),
            "MoE expert-major token-hidden elements",
        );
    }

    pub fn num_routes(self, shape: Shape) -> u32 {
        self.validate();
        shape.validate();
        shape
            .num_total_tokens
            .checked_mul(self.num_experts_per_token)
            .expect("MoE expert-major route count must fit u32")
    }

    fn num_route_hidden_elements(self, shape: Shape) -> usize {
        checked_product(
            "MoE expert-major routed-hidden element count",
            &[self.num_routes(shape) as usize, self.hidden_dim as usize],
        )
    }

    fn num_token_hidden_elements(self, shape: Shape) -> usize {
        checked_product(
            "MoE expert-major token-hidden element count",
            &[shape.num_total_tokens as usize, self.hidden_dim as usize],
        )
    }

    pub fn route_indices_bytes(self, shape: Shape) -> usize {
        checked_product(
            "MoE expert-major route-index byte length",
            &[self.num_routes(shape) as usize, size_of::<u32>()],
        )
    }

    pub fn expert_counts_bytes(self) -> usize {
        checked_product(
            "MoE expert-major expert-count byte length",
            &[self.num_experts as usize, size_of::<u32>()],
        )
    }

    pub fn expert_offsets_bytes(self) -> usize {
        checked_product(
            "MoE expert-major expert-offset byte length",
            &[self.num_experts as usize + 1, size_of::<u32>()],
        )
    }

    pub fn route_probs_bytes(self, shape: Shape) -> usize {
        checked_product(
            "MoE expert-major route-probability byte length",
            &[self.num_routes(shape) as usize, size_of::<f32>()],
        )
    }

    pub fn route_hidden_bytes(self, shape: Shape) -> usize {
        checked_product(
            "MoE expert-major routed-hidden byte length",
            &[self.num_route_hidden_elements(shape), self.dtype.item_size()],
        )
    }

    pub fn token_hidden_bytes(self, shape: Shape) -> usize {
        checked_product(
            "MoE expert-major token-hidden byte length",
            &[self.num_token_hidden_elements(shape), self.dtype.item_size()],
        )
    }

    pub fn shared_expert_gate_logits_bytes(self, shape: Shape) -> usize {
        checked_product(
            "MoE expert-major shared-gate byte length",
            &[shape.num_total_tokens as usize, self.dtype.item_size()],
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub num_total_tokens: u32,
}

impl Shape {
    pub fn validate(self) {
        assert!(self.num_total_tokens > 0);
    }
}

pub struct LayoutBuffers<'a> {
    pub expert_indices: &'a Buffer,
    pub expert_counts: &'a Buffer,
    pub expert_offsets: &'a Buffer,
    pub expert_cursors: &'a Buffer,
    pub routes_by_expert: &'a Buffer,
    pub routes_by_token: &'a Buffer,
    pub experts_by_route: &'a Buffer,
}

pub struct PackInputBuffers<'a> {
    pub input: &'a Buffer,
    pub routes_by_expert: &'a Buffer,
    pub packed_input: &'a Buffer,
}

pub struct ScatterWithoutSharedExpertsBuffers<'a> {
    pub packed_output: &'a Buffer,
    pub routes_by_token: &'a Buffer,
    pub routed_probs: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct ScatterWithSharedExpertsBuffers<'a> {
    pub packed_output: &'a Buffer,
    pub routes_by_token: &'a Buffer,
    pub routed_probs: &'a Buffer,
    pub shared_hidden: &'a Buffer,
    pub shared_expert_gate_logits: &'a Buffer,
    pub output: &'a Buffer,
}

pub struct Compute {
    config: Config,
    constants: KernelConstants,
    layout_clear: CompiledKernel,
    layout_count: CompiledKernel,
    layout_prefix: CompiledKernel,
    layout_scatter: CompiledKernel,
    pack_input: CompiledKernel,
    scatter_without_shared_experts: CompiledKernel,
    scatter_with_shared_experts: CompiledKernel,
}

impl Compute {
    pub fn new(device: &crate::metal::Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            constants: KernelConstants::current(),
            layout_clear: CompiledKernel::new(device, MOE_EXPERT_MAJOR_SOURCE, "moe_expert_major_layout_clear"),
            layout_count: CompiledKernel::new(device, MOE_EXPERT_MAJOR_SOURCE, "moe_expert_major_layout_count"),
            layout_prefix: CompiledKernel::new(device, MOE_EXPERT_MAJOR_SOURCE, "moe_expert_major_layout_prefix"),
            layout_scatter: CompiledKernel::new(device, MOE_EXPERT_MAJOR_SOURCE, "moe_expert_major_layout_scatter"),
            pack_input: CompiledKernel::new(device, MOE_EXPERT_MAJOR_SOURCE, "moe_expert_major_pack_input"),
            scatter_without_shared_experts: CompiledKernel::new(
                device,
                MOE_EXPERT_MAJOR_SOURCE,
                "moe_expert_major_scatter_without_shared_experts",
            ),
            scatter_with_shared_experts: CompiledKernel::new(
                device,
                MOE_EXPERT_MAJOR_SOURCE,
                "moe_expert_major_scatter_with_shared_experts",
            ),
        }
    }

    pub fn invoke_layout<'a>(
        &'a self,
        shape: Shape,
        num_active_tokens: ReplayU32,
        buffers: LayoutBuffers<'a>,
    ) -> LayoutInvocation<'a> {
        self.config.validate_shape(shape);
        validate_layout_buffers(self.config, shape, &buffers);
        LayoutInvocation {
            kernels: self,
            shape,
            buffers,
            num_active_tokens_key: active_key(shape.num_total_tokens, num_active_tokens),
        }
    }

    pub fn invoke_pack_input<'a>(
        &'a self,
        shape: Shape,
        num_active_tokens: ReplayU32,
        buffers: PackInputBuffers<'a>,
    ) -> PackInputInvocation<'a> {
        self.config.validate_shape(shape);
        validate_pack_input_buffers(self.config, shape, &buffers);
        PackInputInvocation {
            kernels: self,
            shape,
            buffers,
            num_active_tokens_key: active_key(shape.num_total_tokens, num_active_tokens),
        }
    }

    pub fn invoke_scatter_without_shared_experts<'a>(
        &'a self,
        shape: Shape,
        num_active_tokens: ReplayU32,
        buffers: ScatterWithoutSharedExpertsBuffers<'a>,
    ) -> ScatterWithoutSharedExpertsInvocation<'a> {
        self.config.validate_shape(shape);
        validate_scatter_without_shared_experts_buffers(self.config, shape, &buffers);
        ScatterWithoutSharedExpertsInvocation {
            kernels: self,
            shape,
            buffers,
            num_active_tokens_key: active_key(shape.num_total_tokens, num_active_tokens),
        }
    }

    pub fn invoke_scatter_with_shared_experts<'a>(
        &'a self,
        shape: Shape,
        num_active_tokens: ReplayU32,
        buffers: ScatterWithSharedExpertsBuffers<'a>,
    ) -> ScatterWithSharedExpertsInvocation<'a> {
        self.config.validate_shape(shape);
        validate_scatter_with_shared_experts_buffers(self.config, shape, &buffers);
        ScatterWithSharedExpertsInvocation {
            kernels: self,
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

pub struct LayoutInvocation<'a> {
    kernels: &'a Compute,
    shape: Shape,
    buffers: LayoutBuffers<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for LayoutInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.kernels.config;
        recorder.set_kernel(&self.kernels.layout_clear);
        recorder.set_buffer_write(0, self.buffers.expert_counts, 0);
        recorder.set_buffer_write(1, self.buffers.expert_cursors, 0);
        recorder.set_u32(2, config.num_experts);
        recorder.dispatch_1d(
            config.num_experts as usize,
            self.kernels.constants.layout_clear.required_threads as usize,
        );

        recorder.set_kernel(&self.kernels.layout_count);
        recorder.set_barrier_before();
        recorder.set_buffer_read(0, self.buffers.expert_indices, 0);
        recorder.set_buffer_read_write(1, self.buffers.expert_counts, 0);
        record_num_active_tokens(recorder, 2, self.shape.num_total_tokens, self.num_active_tokens_key);
        recorder.set_u32(3, config.num_experts_per_token);
        recorder.set_u32(4, config.num_experts);
        recorder.dispatch_1d(
            config.num_routes(self.shape) as usize,
            self.kernels.constants.layout_count.required_threads as usize,
        );

        recorder.set_kernel(&self.kernels.layout_prefix);
        recorder.set_barrier_before();
        recorder.set_buffer_read(0, self.buffers.expert_counts, 0);
        recorder.set_buffer_write(1, self.buffers.expert_offsets, 0);
        recorder.set_buffer_write(2, self.buffers.expert_cursors, 0);
        recorder.set_u32(3, config.num_experts);
        recorder.dispatch_1d(1, self.kernels.constants.layout_prefix.required_threads as usize);

        recorder.set_kernel(&self.kernels.layout_scatter);
        recorder.set_barrier_before();
        recorder.set_buffer_read(0, self.buffers.expert_indices, 0);
        recorder.set_buffer_read_write(1, self.buffers.expert_cursors, 0);
        recorder.set_buffer_write(2, self.buffers.routes_by_expert, 0);
        recorder.set_buffer_write(3, self.buffers.routes_by_token, 0);
        recorder.set_buffer_write(4, self.buffers.experts_by_route, 0);
        record_num_active_tokens(recorder, 5, self.shape.num_total_tokens, self.num_active_tokens_key);
        recorder.set_u32(6, config.num_experts_per_token);
        recorder.set_u32(7, config.num_experts);
        recorder.dispatch_1d(
            config.num_routes(self.shape) as usize,
            self.kernels.constants.layout_scatter.required_threads as usize,
        );
    }
}

pub struct PackInputInvocation<'a> {
    kernels: &'a Compute,
    shape: Shape,
    buffers: PackInputBuffers<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for PackInputInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.kernels.config;
        recorder.set_kernel(&self.kernels.pack_input);
        recorder.set_buffer_read(0, self.buffers.input, 0);
        recorder.set_buffer_read(1, self.buffers.routes_by_expert, 0);
        recorder.set_buffer_write(2, self.buffers.packed_input, 0);
        record_num_active_tokens(recorder, 3, self.shape.num_total_tokens, self.num_active_tokens_key);
        recorder.set_u32(4, config.num_experts_per_token);
        recorder.set_u32(5, config.hidden_dim);
        recorder.dispatch_1d(
            config.num_route_hidden_elements(self.shape),
            self.kernels.constants.pack_input.required_threads as usize,
        );
    }
}

pub struct ScatterWithoutSharedExpertsInvocation<'a> {
    kernels: &'a Compute,
    shape: Shape,
    buffers: ScatterWithoutSharedExpertsBuffers<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for ScatterWithoutSharedExpertsInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.kernels.config;
        recorder.set_kernel(&self.kernels.scatter_without_shared_experts);
        recorder.set_buffer_read(0, self.buffers.packed_output, 0);
        recorder.set_buffer_read(1, self.buffers.routes_by_token, 0);
        recorder.set_buffer_read(2, self.buffers.routed_probs, 0);
        recorder.set_buffer_write(3, self.buffers.output, 0);
        record_num_active_tokens(recorder, 4, self.shape.num_total_tokens, self.num_active_tokens_key);
        recorder.set_u32(5, config.num_experts_per_token);
        recorder.set_u32(6, config.hidden_dim);
        recorder.dispatch_1d(
            config.num_token_hidden_elements(self.shape),
            self.kernels.constants.scatter_output.required_threads as usize,
        );
    }
}

pub struct ScatterWithSharedExpertsInvocation<'a> {
    kernels: &'a Compute,
    shape: Shape,
    buffers: ScatterWithSharedExpertsBuffers<'a>,
    num_active_tokens_key: Option<ReplayParameterKey>,
}

impl Operator for ScatterWithSharedExpertsInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.kernels.config;
        recorder.set_kernel(&self.kernels.scatter_with_shared_experts);
        recorder.set_buffer_read(0, self.buffers.packed_output, 0);
        recorder.set_buffer_read(1, self.buffers.routes_by_token, 0);
        recorder.set_buffer_read(2, self.buffers.routed_probs, 0);
        recorder.set_buffer_read(3, self.buffers.shared_hidden, 0);
        recorder.set_buffer_read(4, self.buffers.shared_expert_gate_logits, 0);
        recorder.set_buffer_write(5, self.buffers.output, 0);
        record_num_active_tokens(recorder, 6, self.shape.num_total_tokens, self.num_active_tokens_key);
        recorder.set_u32(7, config.num_experts_per_token);
        recorder.set_u32(8, config.hidden_dim);
        recorder.dispatch_1d(
            config.num_token_hidden_elements(self.shape),
            self.kernels.constants.scatter_output.required_threads as usize,
        );
    }
}

fn record_num_active_tokens(
    recorder: &CommandRecorder<'_>,
    binding_index: usize,
    num_total_tokens: u32,
    key: Option<ReplayParameterKey>,
) {
    match key {
        Some(key) => recorder.bind_u32(binding_index, key, 1, num_total_tokens),
        None => recorder.set_u32(binding_index, num_total_tokens),
    }
}

fn validate_layout_buffers(config: Config, shape: Shape, buffers: &LayoutBuffers<'_>) {
    let bytes = config.route_indices_bytes(shape);
    assert!(buffers.expert_indices.len_bytes() >= bytes);
    assert!(buffers.expert_counts.len_bytes() >= config.expert_counts_bytes());
    assert!(buffers.expert_offsets.len_bytes() >= config.expert_offsets_bytes());
    assert!(buffers.expert_cursors.len_bytes() >= config.expert_counts_bytes());
    assert!(buffers.routes_by_expert.len_bytes() >= bytes);
    assert!(buffers.routes_by_token.len_bytes() >= bytes);
    assert!(buffers.experts_by_route.len_bytes() >= bytes);
}

fn validate_pack_input_buffers(config: Config, shape: Shape, buffers: &PackInputBuffers<'_>) {
    assert!(buffers.input.len_bytes() >= config.token_hidden_bytes(shape));
    assert!(buffers.routes_by_expert.len_bytes() >= config.route_indices_bytes(shape));
    assert!(buffers.packed_input.len_bytes() >= config.route_hidden_bytes(shape));
}

fn validate_scatter_without_shared_experts_buffers(
    config: Config,
    shape: Shape,
    buffers: &ScatterWithoutSharedExpertsBuffers<'_>,
) {
    assert!(buffers.packed_output.len_bytes() >= config.route_hidden_bytes(shape));
    assert!(buffers.routes_by_token.len_bytes() >= config.route_indices_bytes(shape));
    assert!(buffers.routed_probs.len_bytes() >= config.route_probs_bytes(shape));
    assert!(buffers.output.len_bytes() >= config.token_hidden_bytes(shape));
}

fn validate_scatter_with_shared_experts_buffers(
    config: Config,
    shape: Shape,
    buffers: &ScatterWithSharedExpertsBuffers<'_>,
) {
    assert!(buffers.packed_output.len_bytes() >= config.route_hidden_bytes(shape));
    assert!(buffers.routes_by_token.len_bytes() >= config.route_indices_bytes(shape));
    assert!(buffers.routed_probs.len_bytes() >= config.route_probs_bytes(shape));
    assert!(buffers.shared_hidden.len_bytes() >= config.token_hidden_bytes(shape));
    assert!(buffers.shared_expert_gate_logits.len_bytes() >= config.shared_expert_gate_logits_bytes(shape));
    assert!(buffers.output.len_bytes() >= config.token_hidden_bytes(shape));
}

#[cfg(test)]
#[path = "expert_major_test.rs"]
mod tests;
