use std::mem::size_of;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;
use crate::operators::affine_quantized;

fn to_i32(value: u32, name: &str) -> i32 {
    value.try_into().unwrap_or_else(|_| panic!("{name} must fit i32"))
}

fn checked_bytes(name: &str, dimensions: &[usize], dtype: Dtype) -> usize {
    dimensions
        .iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
        .and_then(|elements| elements.checked_mul(dtype.item_size()))
        .unwrap_or_else(|| panic!("{name} byte length must fit usize"))
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub num_experts: u32,
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub group_size: u32,
    pub bits: u32,
    pub dtype: Dtype,
}

impl Config {
    pub fn validate(self) {
        assert!(self.num_experts > 0);
        assert!(self.hidden_dim > 0);
        assert!(self.intermediate_dim > 0);
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(
            self.hidden_dim % self.group_size,
            0,
            "sparse MLP hidden_dim must be group aligned"
        );
        assert_eq!(
            self.intermediate_dim % self.group_size,
            0,
            "sparse MLP intermediate_dim must be group aligned"
        );
        assert_eq!(self.dtype, Dtype::Bfloat16, "sparse MLP currently supports bf16 only");
        i32::try_from(self.hidden_dim).expect("sparse MLP hidden_dim must fit i32");
        i32::try_from(self.num_experts).expect("sparse MLP expert count must fit i32");
        i32::try_from(self.intermediate_dim).expect("sparse MLP intermediate_dim must fit i32");
    }

    pub fn gate_up_config(self) -> affine_quantized::ExpertConfig {
        self.expert_affine_config(self.intermediate_dim, self.hidden_dim)
    }

    pub fn down_config(self) -> affine_quantized::ExpertConfig {
        self.expert_affine_config(self.hidden_dim, self.intermediate_dim)
    }

    fn expert_affine_config(self, n: u32, k: u32) -> affine_quantized::ExpertConfig {
        self.validate();
        affine_quantized::ExpertConfig {
            num_experts: self.num_experts as i32,
            matmul: affine_quantized::Config::same_dtype(
                n as i32,
                k as i32,
                self.group_size as i32,
                self.bits as i32,
                self.dtype,
            ),
        }
    }

    fn token_major_gate_up_shape(self, shape: TokenMajorShape) -> affine_quantized::GatherShape {
        shape.validate();
        affine_quantized::GatherShape {
            num_routes: to_i32(shape.num_total_routes, "sparse MLP route count"),
            num_input_vectors: to_i32(shape.num_total_tokens, "sparse MLP token count"),
        }
    }

    fn token_major_down_shape(self, shape: TokenMajorShape) -> affine_quantized::GatherShape {
        shape.validate();
        affine_quantized::GatherShape {
            num_routes: to_i32(shape.num_total_routes, "sparse MLP route count"),
            num_input_vectors: to_i32(shape.num_total_routes, "sparse MLP route count"),
        }
    }

    fn expert_major_affine_shape(self, shape: ExpertMajorShape) -> affine_quantized::RaggedExpertMajorShape {
        shape.validate();
        affine_quantized::RaggedExpertMajorShape {
            num_routes: to_i32(shape.num_total_routes, "sparse MLP route count"),
        }
    }

    pub fn token_major_input_bytes(self, shape: TokenMajorShape) -> usize {
        self.validate();
        shape.validate();
        self.token_major_input_bytes_unchecked(shape)
    }

    fn token_major_input_bytes_unchecked(self, shape: TokenMajorShape) -> usize {
        checked_bytes(
            "sparse MLP token-major input",
            &[shape.num_total_tokens as usize, self.hidden_dim as usize],
            self.dtype,
        )
    }

    pub fn token_major_route_indices_bytes(self, shape: TokenMajorShape) -> usize {
        shape.validate();
        self.token_major_route_indices_bytes_unchecked(shape)
    }

    fn token_major_route_indices_bytes_unchecked(self, shape: TokenMajorShape) -> usize {
        (shape.num_total_routes as usize)
            .checked_mul(size_of::<u32>())
            .expect("sparse MLP route-index byte length must fit usize")
    }

    pub fn swiglu_bytes(self, num_routes: u32) -> usize {
        self.validate();
        assert!(num_routes > 0);
        self.swiglu_bytes_unchecked(num_routes)
    }

    fn swiglu_bytes_unchecked(self, num_routes: u32) -> usize {
        checked_bytes(
            "sparse MLP swiglu",
            &[num_routes as usize, self.intermediate_dim as usize],
            self.dtype,
        )
    }

    pub fn token_major_output_bytes(self, shape: TokenMajorShape) -> usize {
        self.validate();
        shape.validate();
        self.token_major_output_bytes_unchecked(shape)
    }

    pub fn expert_major_input_bytes(self, shape: ExpertMajorShape) -> usize {
        self.validate();
        shape.validate();
        checked_bytes(
            "sparse MLP expert-major input",
            &[shape.num_total_routes as usize, self.hidden_dim as usize],
            self.dtype,
        )
    }

    pub fn expert_major_output_bytes(self, shape: ExpertMajorShape) -> usize {
        self.validate();
        shape.validate();
        checked_bytes(
            "sparse MLP expert-major output",
            &[shape.num_total_routes as usize, self.hidden_dim as usize],
            self.dtype,
        )
    }

    pub fn expert_major_route_indices_bytes(self, shape: ExpertMajorShape) -> usize {
        shape.validate();
        (shape.num_total_routes as usize)
            .checked_mul(size_of::<u32>())
            .expect("sparse MLP expert-major route-index byte length must fit usize")
    }

    fn token_major_output_bytes_unchecked(self, shape: TokenMajorShape) -> usize {
        checked_bytes(
            "sparse MLP token-major output",
            &[shape.num_total_routes as usize, self.hidden_dim as usize],
            self.dtype,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TokenMajorShape {
    pub num_total_routes: u32,
    pub num_total_tokens: u32,
}

impl TokenMajorShape {
    pub fn validate(self) {
        assert!(self.num_total_routes > 0);
        assert!(self.num_total_tokens > 0);
        to_i32(self.num_total_routes, "sparse MLP route count");
        to_i32(self.num_total_tokens, "sparse MLP token count");
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExpertMajorShape {
    pub num_total_routes: u32,
    pub num_total_tokens: u32,
    pub num_experts_per_token: u32,
}

impl ExpertMajorShape {
    pub fn validate(self) {
        to_i32(self.num_total_routes, "sparse MLP route count");
        assert!(self.num_total_routes > 0);
        assert!(self.num_total_tokens > 0);
        assert!(self.num_experts_per_token > 0);
        assert_eq!(
            self.num_total_routes,
            self.num_total_tokens
                .checked_mul(self.num_experts_per_token)
                .expect("sparse MLP total route count must fit u32")
        );
    }
}

#[derive(Clone, Copy)]
struct RouteActivity {
    num_total_tokens: u32,
    num_experts_per_token: u32,
    num_active_tokens: ReplayU32,
}

impl RouteActivity {
    fn new(config: Config, num_total_tokens: u32, num_experts_per_token: u32, num_active_tokens: ReplayU32) -> Self {
        config.validate();
        assert!(num_total_tokens > 0);
        assert!(num_experts_per_token > 0);
        assert!(
            num_experts_per_token <= config.num_experts,
            "sparse MLP experts per token must not exceed expert count"
        );
        let num_total_routes = num_total_tokens
            .checked_mul(num_experts_per_token)
            .expect("sparse MLP total route count must fit u32");
        i32::try_from(num_total_routes).expect("sparse MLP total route count must fit i32");
        if let ReplayU32::Fixed(num_active_tokens) = num_active_tokens {
            assert_eq!(num_active_tokens, num_total_tokens);
        }
        Self {
            num_total_tokens,
            num_experts_per_token,
            num_active_tokens,
        }
    }

    fn num_total_routes(self) -> u32 {
        self.num_total_tokens * self.num_experts_per_token
    }
}

#[derive(Clone, Copy)]
pub struct TokenMajorBuffers<'a> {
    pub input: &'a Buffer,
    pub token_indices: &'a Buffer,
    pub expert_indices: &'a Buffer,
    pub route_indices: &'a Buffer,
    pub routed_hidden: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct ExpertMajorBuffers<'a> {
    pub packed_input: &'a Buffer,
    pub experts_by_route: &'a Buffer,
    pub expert_offsets: &'a Buffer,
    pub packed_output: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct Weights<'a> {
    pub gate_weight: &'a Buffer,
    pub gate_scales: &'a Buffer,
    pub gate_biases: &'a Buffer,
    pub up_weight: &'a Buffer,
    pub up_scales: &'a Buffer,
    pub up_biases: &'a Buffer,
    pub down_weight: &'a Buffer,
    pub down_scales: &'a Buffer,
    pub down_biases: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct Scratch<'a> {
    pub swiglu: &'a Buffer,
}

/// Records one quantized sparse MLP through either supported route layout:
///
/// ```text
/// TokenMajor
///
/// input [T, H] --(token_indices, expert_indices)--> gate_up_swiglu
///                                                       |
///                                                       v
///                                                swiglu [R, I]
///                                                       |
///                              (route_indices, expert_indices)
///                                                       |
///                                                       v
///                                             routed_hidden [R, H]
///
/// ExpertMajor
///
/// packed_input [R, H] --(experts_by_route)--> gate_up_swiglu
///                                                   |
///                                                   v
///                                            swiglu [R, I]
///                                                   |
///                                           experts_by_route
///                                                   |
///                                                   v
///                                         packed_output [R, H]
/// ```
pub struct Compute {
    token_major: TokenMajorKernels,
    expert_major: ExpertMajorKernels,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        Self {
            token_major: TokenMajorKernels::new(device, config),
            expert_major: ExpertMajorKernels::new(device, config),
        }
    }

    pub fn invoke_token_major<'a>(
        &'a self,
        shape: TokenMajorShape,
        num_experts_per_token: u32,
        num_active_tokens: ReplayU32,
        buffers: TokenMajorBuffers<'a>,
        scratch: Scratch<'a>,
        weights: Weights<'a>,
    ) -> TokenMajorInvocation<'a> {
        self.token_major.invoke(
            shape,
            num_experts_per_token,
            num_active_tokens,
            buffers,
            scratch,
            weights,
        )
    }

    pub fn invoke_expert_major<'a>(
        &'a self,
        shape: ExpertMajorShape,
        num_active_tokens: ReplayU32,
        buffers: ExpertMajorBuffers<'a>,
        scratch: Scratch<'a>,
        weights: Weights<'a>,
    ) -> ExpertMajorInvocation<'a> {
        let route_activity = RouteActivity::new(
            self.expert_major.config,
            shape.num_total_tokens,
            shape.num_experts_per_token,
            num_active_tokens,
        );
        assert_eq!(shape.num_total_routes, route_activity.num_total_routes());
        self.expert_major
            .invoke(shape, route_activity, buffers, scratch, weights)
    }
}

pub struct TokenMajorKernels {
    config: Config,
    gate_up_swiglu: affine_quantized::GatherGateUpSwiGLUKernel,
    down: affine_quantized::GatherMatmulKernel,
}

pub struct ExpertMajorKernels {
    config: Config,
    gate_up_swiglu: affine_quantized::RaggedExpertMajorGateUpSwiGLUKernel,
    down: affine_quantized::RaggedExpertMajorMatmulKernel,
}

impl TokenMajorKernels {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            gate_up_swiglu: affine_quantized::GatherGateUpSwiGLUKernel::new(device, config.gate_up_config()),
            down: affine_quantized::GatherMatmulKernel::new(device, config.down_config()),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: TokenMajorShape,
        num_experts_per_token: u32,
        num_active_tokens: ReplayU32,
        buffers: TokenMajorBuffers<'a>,
        scratch: Scratch<'a>,
        weights: Weights<'a>,
    ) -> TokenMajorInvocation<'a> {
        let route_activity = RouteActivity::new(
            self.config,
            shape.num_total_tokens,
            num_experts_per_token,
            num_active_tokens,
        );
        assert_eq!(shape.num_total_routes, route_activity.num_total_routes());
        TokenMajorInvocation {
            kernels: self,
            shape,
            buffers,
            scratch,
            weights,
            route_activity,
        }
    }

    pub fn invoke_gate_up_swiglu<'a>(
        &'a self,
        shape: TokenMajorShape,
        num_active_tokens: ReplayU32,
        buffers: TokenMajorBuffers<'a>,
        scratch: Scratch<'a>,
        weights: Weights<'a>,
    ) -> TokenMajorGateUpSwiGLUInvocation<'a> {
        assert_eq!(shape.num_total_routes % shape.num_total_tokens, 0);
        let route_activity = RouteActivity::new(
            self.config,
            shape.num_total_tokens,
            shape.num_total_routes / shape.num_total_tokens,
            num_active_tokens,
        );
        TokenMajorGateUpSwiGLUInvocation {
            kernels: self,
            shape,
            buffers,
            scratch,
            weights,
            route_activity,
        }
    }

    pub fn invoke_down<'a>(
        &'a self,
        shape: TokenMajorShape,
        num_active_tokens: ReplayU32,
        buffers: TokenMajorBuffers<'a>,
        scratch: Scratch<'a>,
        weights: Weights<'a>,
    ) -> TokenMajorDownInvocation<'a> {
        assert_eq!(shape.num_total_routes % shape.num_total_tokens, 0);
        let route_activity = RouteActivity::new(
            self.config,
            shape.num_total_tokens,
            shape.num_total_routes / shape.num_total_tokens,
            num_active_tokens,
        );
        TokenMajorDownInvocation {
            kernels: self,
            shape,
            buffers,
            scratch,
            weights,
            route_activity,
        }
    }
}

impl ExpertMajorKernels {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            gate_up_swiglu: affine_quantized::RaggedExpertMajorGateUpSwiGLUKernel::new(device, config.gate_up_config()),
            down: affine_quantized::RaggedExpertMajorMatmulKernel::new(device, config.down_config()),
        }
    }

    fn invoke<'a>(
        &'a self,
        shape: ExpertMajorShape,
        route_activity: RouteActivity,
        buffers: ExpertMajorBuffers<'a>,
        scratch: Scratch<'a>,
        weights: Weights<'a>,
    ) -> ExpertMajorInvocation<'a> {
        ExpertMajorInvocation {
            kernels: self,
            shape,
            buffers,
            scratch,
            weights,
            route_activity,
        }
    }
}

pub struct TokenMajorInvocation<'a> {
    kernels: &'a TokenMajorKernels,
    shape: TokenMajorShape,
    buffers: TokenMajorBuffers<'a>,
    scratch: Scratch<'a>,
    weights: Weights<'a>,
    route_activity: RouteActivity,
}

impl Operator for TokenMajorInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        TokenMajorGateUpSwiGLUInvocation {
            kernels: self.kernels,
            shape: self.shape,
            buffers: self.buffers,
            scratch: self.scratch,
            weights: self.weights,
            route_activity: self.route_activity,
        }
        .record(recorder);
        recorder.record_with_barrier_before(TokenMajorDownInvocation {
            kernels: self.kernels,
            shape: self.shape,
            buffers: self.buffers,
            scratch: self.scratch,
            weights: self.weights,
            route_activity: self.route_activity,
        });
    }
}

pub struct TokenMajorGateUpSwiGLUInvocation<'a> {
    kernels: &'a TokenMajorKernels,
    shape: TokenMajorShape,
    buffers: TokenMajorBuffers<'a>,
    scratch: Scratch<'a>,
    weights: Weights<'a>,
    route_activity: RouteActivity,
}

impl Operator for TokenMajorGateUpSwiGLUInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let invocation = self.kernels.gate_up_swiglu.invoke(
            self.kernels.config.token_major_gate_up_shape(self.shape),
            self.route_activity.num_total_tokens,
            self.route_activity.num_experts_per_token,
            self.route_activity.num_active_tokens,
            self.scratch.swiglu,
            self.buffers.input,
            self.weights.gate_weight,
            self.weights.gate_scales,
            self.weights.gate_biases,
            self.weights.up_weight,
            self.weights.up_scales,
            self.weights.up_biases,
            self.buffers.token_indices,
            self.buffers.expert_indices,
        );
        invocation.record(recorder);
    }
}

pub struct TokenMajorDownInvocation<'a> {
    kernels: &'a TokenMajorKernels,
    shape: TokenMajorShape,
    buffers: TokenMajorBuffers<'a>,
    scratch: Scratch<'a>,
    weights: Weights<'a>,
    route_activity: RouteActivity,
}

impl Operator for TokenMajorDownInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let invocation = self.kernels.down.invoke(
            self.kernels.config.token_major_down_shape(self.shape),
            self.route_activity.num_total_tokens,
            self.route_activity.num_experts_per_token,
            self.route_activity.num_active_tokens,
            self.buffers.routed_hidden,
            self.scratch.swiglu,
            self.weights.down_weight,
            self.weights.down_scales,
            self.weights.down_biases,
            self.buffers.route_indices,
            self.buffers.expert_indices,
        );
        invocation.record(recorder);
    }
}

pub struct ExpertMajorInvocation<'a> {
    kernels: &'a ExpertMajorKernels,
    shape: ExpertMajorShape,
    buffers: ExpertMajorBuffers<'a>,
    scratch: Scratch<'a>,
    weights: Weights<'a>,
    route_activity: RouteActivity,
}

impl Operator for ExpertMajorInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let gate_up_swiglu = self.kernels.gate_up_swiglu.invoke(
            self.kernels.config.expert_major_affine_shape(self.shape),
            self.route_activity.num_total_tokens,
            self.route_activity.num_experts_per_token,
            self.route_activity.num_active_tokens,
            self.scratch.swiglu,
            self.buffers.packed_input,
            self.weights.gate_weight,
            self.weights.gate_scales,
            self.weights.gate_biases,
            self.weights.up_weight,
            self.weights.up_scales,
            self.weights.up_biases,
            self.buffers.experts_by_route,
            self.buffers.expert_offsets,
        );
        gate_up_swiglu.record(recorder);
        let down = self.kernels.down.invoke(
            self.kernels.config.expert_major_affine_shape(self.shape),
            self.route_activity.num_total_tokens,
            self.route_activity.num_experts_per_token,
            self.route_activity.num_active_tokens,
            self.buffers.packed_output,
            self.scratch.swiglu,
            self.weights.down_weight,
            self.weights.down_scales,
            self.weights.down_biases,
            self.buffers.experts_by_route,
            self.buffers.expert_offsets,
        );
        recorder.record_with_barrier_before(down);
    }
}

#[cfg(test)]
#[path = "sparse_mlp_test.rs"]
mod tests;
