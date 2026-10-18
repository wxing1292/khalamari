use inference_backend_metal::components::dense_mlp;
use inference_backend_metal::components::moe::combine;
use inference_backend_metal::components::moe::expert_major;
use inference_backend_metal::components::moe::routing;
use inference_backend_metal::components::sparse_mlp;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_backend_metal::operators::softmax;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::mlp::moe::GatedMoECore;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::mlp::moe::scratch::MoERoutingScratchBindings;
use crate::mlp::moe::scratch::MoEScratchBindings;
use crate::mlp::moe::scratch::SharedExpertsScratchBindings;

const TOKEN_MAJOR_MAX_TOKENS: u32 = 4;
const EXPERT_MAJOR_MIN_TOKENS: u32 = TOKEN_MAJOR_MAX_TOKENS + 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatedMoEMetalConfig {
    pub group_size: u32,
    pub bits: u32,
    pub router_bits: u32,
    pub shared_expert_gate_bits: u32,
    pub io_dtype: Dtype,
}

impl GatedMoEMetalConfig {
    pub fn validate(self) {
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert!(matches!(self.router_bits, 2 | 3 | 4 | 6 | 8));
        assert!(matches!(self.shared_expert_gate_bits, 2 | 3 | 4 | 6 | 8));
        match self.io_dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 gated MoE model boundary is not supported"),
            dtype => panic!("unsupported gated MoE model boundary dtype {dtype:?}"),
        }
    }
}

#[derive(Clone, Copy)]
pub struct GatedMoEWeights<'a> {
    pub router_weight: &'a Buffer,
    pub router_scales: &'a Buffer,
    pub router_biases: &'a Buffer,
    pub topk_experts: sparse_mlp::Weights<'a>,
}

#[derive(Clone, Copy)]
pub struct GatedMoERoutingWeights<'a> {
    pub router_weight: &'a Buffer,
    pub router_scales: &'a Buffer,
    pub router_biases: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct GatedMoERoutingInput<'a> {
    pub num_total_tokens: u32,
    pub num_active_tokens: ReplayU32,
    pub hidden_state: &'a Buffer,
    pub scratch: MoERoutingScratchBindings<'a>,
    pub weights: GatedMoERoutingWeights<'a>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GatedMoERoutingReplayTopology {
    pub router_affine: affine_quantized::KernelKind,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GatedMoEReplayTopology {
    variant_key: VariantKey,
    pub router_affine: affine_quantized::KernelKind,
    pub shared_expert_gate_affine: Option<affine_quantized::KernelKind>,
    pub shared_experts_dense: Option<dense_mlp::ReplayTopology>,
}

impl<'a> GatedMoEWeights<'a> {
    fn routing(self) -> GatedMoERoutingWeights<'a> {
        GatedMoERoutingWeights {
            router_weight: self.router_weight,
            router_scales: self.router_scales,
            router_biases: self.router_biases,
        }
    }
}

#[derive(Clone, Copy)]
pub struct GatedMoESharedExpertsWeights<'a> {
    pub shared_expert_gate_weight: &'a Buffer,
    pub shared_expert_gate_scales: &'a Buffer,
    pub shared_expert_gate_biases: &'a Buffer,
    pub shared_experts: dense_mlp::Weights<'a>,
}

#[derive(Clone, Copy)]
pub struct GatedMoESharedExpertsInput<'a> {
    pub scratch: SharedExpertsScratchBindings<'a>,
    pub weights: GatedMoESharedExpertsWeights<'a>,
}

#[derive(Clone, Copy)]
pub struct GatedMoEInput<'a> {
    pub num_total_tokens: u32,
    pub num_active_tokens: ReplayU32,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub scratch: MoEScratchBindings<'a>,
    pub weights: GatedMoEWeights<'a>,
    pub shared_experts: Option<GatedMoESharedExpertsInput<'a>>,
}

/// Records one gated MoE layer:
///
/// ```text
/// hidden_state
///      |
///      v
/// router -> router_logits -> router_softmax -> router_probs
///                                               |
///                                               v
///                                            routing
///                                               |
///                                expert_indices + expert_probs
///
/// TokenMajor
///
/// hidden_state + routing
///      -> topk_experts.invoke_token_major
///      -> routed_hidden
///      -> combine
///      -> next_hidden_state
///
/// ExpertMajor
///
/// routing
///      -> expert_major.layout
/// hidden_state + layout
///      -> expert_major.pack_input
///      -> topk_experts.invoke_expert_major
///      -> packed_output
///      -> expert_major.scatter
///      -> next_hidden_state
///
/// Optional shared branch
///
/// hidden_state -> shared_experts.mlp --------------> shared_hidden
/// hidden_state -> shared_experts.shared_expert_gate -> gate_logits
///
/// combine and expert_major.scatter consume both shared outputs.
/// ```
pub struct GatedMoE {
    core: GatedMoECore,
    registry: Registry,
    router: affine_quantized::Matmul,
    router_softmax: softmax::Kernel,
    routing: routing::Compute,
    topk_experts: sparse_mlp::Compute,
    shared_experts: Option<GatedMoESharedExperts>,
}

struct GatedMoESharedExperts {
    shared_expert_gate: affine_quantized::Matmul,
    mlp: dense_mlp::Compute,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum VariantKey {
    TokenMajor,
    ExpertMajor,
}

enum Variant {
    TokenMajor { combine: combine::Compute },
    ExpertMajor { expert_major: expert_major::Compute },
}

struct Registry {
    entries: Vec<(VariantKey, Variant)>,
}

impl Registry {
    fn new(device: &Device, core: &GatedMoECore) -> Self {
        let combine = combine::Compute::new(
            device,
            combine::Config::bf16(
                core.num_experts_per_token.try_into().expect("MoE top-k must fit u32"),
                core.hidden_dim.try_into().expect("MoE hidden_dim must fit u32"),
            ),
        );
        let expert_major = expert_major::Compute::new(
            device,
            expert_major::Config::bf16(
                core.num_experts.try_into().expect("MoE expert count must fit u32"),
                core.num_experts_per_token.try_into().expect("MoE top-k must fit u32"),
                core.hidden_dim.try_into().expect("MoE hidden_dim must fit u32"),
            ),
        );
        Self {
            entries: vec![
                (VariantKey::TokenMajor, Variant::TokenMajor { combine }),
                (VariantKey::ExpertMajor, Variant::ExpertMajor { expert_major }),
            ],
        }
    }

    fn get(&self, key: VariantKey) -> (VariantKey, &Variant) {
        let (key, variant) = self
            .entries
            .iter()
            .find(|(candidate_key, _)| *candidate_key == key)
            .expect("MoE registry requires each selectable execution variant");
        (*key, variant)
    }
}

struct Selector;

impl Selector {
    fn select(registry: &Registry, num_total_tokens: u32) -> (VariantKey, &Variant) {
        let key = if num_total_tokens <= TOKEN_MAJOR_MAX_TOKENS {
            VariantKey::TokenMajor
        } else {
            VariantKey::ExpertMajor
        };
        registry.get(key)
    }
}

impl GatedMoE {
    fn validate_input(&self, input: &GatedMoEInput<'_>) {
        if let ReplayU32::Fixed(num_active_tokens) = input.num_active_tokens {
            assert_eq!(num_active_tokens, input.num_total_tokens);
        }
        assert!(
            shared_experts_input_matches_core(&self.core, input.shared_experts.is_some()),
            "gated MoE replay shared expert must match core configuration"
        );
    }

    pub fn new(device: &Device, core: GatedMoECore, config: GatedMoEMetalConfig) -> Self {
        core.validate();
        config.validate();
        let router_shape = core.router_shape();
        let registry = Registry::new(device, &core);
        Self {
            registry,
            router: affine_quantized::Matmul::new(
                device,
                affine_config_with_bits(router_shape.out_dim, router_shape.in_dim, config.router_bits, config),
            ),
            router_softmax: softmax::Kernel::new(
                device,
                softmax::Config {
                    num_values_per_row: core.num_experts.try_into().expect("MoE expert count must fit u32"),
                    dtype: config.io_dtype,
                },
            ),
            routing: routing::Compute::new(
                device,
                routing::Config {
                    num_experts: core.num_experts.try_into().expect("MoE expert count must fit u32"),
                    num_experts_per_token: core.num_experts_per_token.try_into().expect("MoE top-k must fit u32"),
                    norm_topk_prob: core.norm_topk_prob,
                },
            ),
            topk_experts: sparse_mlp::Compute::new(device, topk_experts_config(&core, config)),
            shared_experts: core.shared_experts_core().map(|shared_core| {
                let gate_shape = core
                    .shared_expert_gate_shape()
                    .expect("shared-expert MLP requires a shared-expert gate shape");
                GatedMoESharedExperts {
                    shared_expert_gate: affine_quantized::Matmul::new(
                        device,
                        affine_config_with_bits(
                            gate_shape.out_dim,
                            gate_shape.in_dim,
                            config.shared_expert_gate_bits,
                            config,
                        ),
                    ),
                    mlp: dense_mlp::Compute::new(device, shared_experts_config(&shared_core, config)),
                }
            }),
            core,
        }
    }

    /// Returns the recorded topology of the routing-only replay chain.
    pub fn routing_replay_topology(&self, num_total_tokens: u32) -> GatedMoERoutingReplayTopology {
        GatedMoERoutingReplayTopology {
            router_affine: self.router.topology(num_total_tokens),
        }
    }

    /// Returns the first token count for each routing-only topology change.
    pub fn routing_replay_topology_boundaries(&self) -> Box<[u32]> {
        self.router.topology_boundaries()
    }

    /// Returns the complete command topology for one full-MoE replay capacity.
    pub fn replay_topology(&self, num_total_tokens: u32) -> GatedMoEReplayTopology {
        let (variant_key, _) = Selector::select(&self.registry, num_total_tokens);
        GatedMoEReplayTopology {
            variant_key,
            router_affine: self.router.topology(num_total_tokens),
            shared_expert_gate_affine: self
                .shared_experts
                .as_ref()
                .map(|shared_experts| shared_experts.shared_expert_gate.topology(num_total_tokens)),
            shared_experts_dense: self
                .shared_experts
                .as_ref()
                .map(|shared_experts| shared_experts.mlp.topology(num_total_tokens)),
        }
    }

    /// Returns the first token count for each full-MoE command-topology change.
    pub fn replay_topology_boundaries(&self) -> Box<[u32]> {
        let mut boundaries = self.router.topology_boundaries().into_vec();
        boundaries.push(EXPERT_MAJOR_MIN_TOKENS);
        if let Some(shared_experts) = &self.shared_experts {
            boundaries.extend(shared_experts.shared_expert_gate.topology_boundaries());
            boundaries.extend(shared_experts.mlp.topology_boundaries());
        }
        boundaries.sort_unstable();
        boundaries.dedup();
        boundaries.into_boxed_slice()
    }

    /// Records only `router affine -> softmax -> top-k routing`.
    pub fn record_routing<'a, R>(&'a self, recorder: &mut R, input: GatedMoERoutingInput<'a>)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_total_tokens = input.num_total_tokens;
        let scratch = input.scratch;
        let weights = input.weights;
        let num_active_tokens = input.num_active_tokens;
        recorder.record_with_barrier_before(ReplayOp::opaque(self.router.invoke(
            num_total_tokens,
            num_active_tokens,
            scratch.router_logits,
            0,
            input.hidden_state,
            0,
            weights.router_weight,
            0,
            weights.router_scales,
            0,
            weights.router_biases,
            0,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.router_softmax.invoke(
            softmax::Shape {
                num_total_rows: num_total_tokens,
            },
            num_active_tokens,
            softmax::Buffers {
                input: scratch.router_logits,
                output: scratch.router_probs,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.routing.invoke(
            routing::Shape { num_total_tokens },
            num_active_tokens,
            routing::Buffers {
                router_probs: scratch.router_probs,
                expert_indices: scratch.expert_indices,
                expert_probs: scratch.expert_probs,
            },
        )));
    }

    fn record_token_major<'a>(
        &'a self,
        combine: &'a combine::Compute,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        input: GatedMoEInput<'a>,
    ) {
        let num_total_tokens = input.num_total_tokens;
        let num_active_tokens = input.num_active_tokens;
        let hidden_state = input.hidden_state;
        let next_hidden_state = input.next_hidden_state;
        let scratch = input.scratch;
        let weights = input.weights;
        let shared_experts = input.shared_experts;
        self.record_routing(
            recorder,
            GatedMoERoutingInput {
                num_total_tokens,
                num_active_tokens: input.num_active_tokens,
                hidden_state,
                scratch: scratch.routing,
                weights: weights.routing(),
            },
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(self.topk_experts.invoke_token_major(
            sparse_mlp::TokenMajorShape {
                num_total_routes: self.num_total_routes(num_total_tokens),
                num_total_tokens,
            },
            self.num_experts_per_token(),
            num_active_tokens,
            sparse_mlp::TokenMajorBuffers {
                input: hidden_state,
                token_indices: scratch.topk_experts.token_indices,
                expert_indices: scratch.routing.expert_indices,
                route_indices: scratch.topk_experts.route_indices,
                routed_hidden: scratch.topk_experts.routed_hidden,
            },
            sparse_mlp::Scratch {
                swiglu: scratch.topk_experts.sparse_swiglu,
            },
            weights.topk_experts,
        )));
        match shared_experts {
            None => {
                recorder.record_with_barrier_before(ReplayOp::opaque(combine.invoke_without_shared_experts(
                    combine::Shape { num_total_tokens },
                    num_active_tokens,
                    combine::WithoutSharedExpertsBuffers {
                        routed_hidden: scratch.topk_experts.routed_hidden,
                        routed_probs: scratch.routing.expert_probs,
                        output: next_hidden_state,
                    },
                )));
            },
            Some(shared_experts) => {
                self.record_shared_experts(
                    recorder,
                    num_total_tokens,
                    num_active_tokens,
                    hidden_state,
                    shared_experts.scratch,
                    shared_experts.weights,
                );
                recorder.record_with_barrier_before(ReplayOp::opaque(combine.invoke_with_shared_experts(
                    combine::Shape { num_total_tokens },
                    num_active_tokens,
                    combine::WithSharedExpertsBuffers {
                        routed_hidden: scratch.topk_experts.routed_hidden,
                        routed_probs: scratch.routing.expert_probs,
                        shared_hidden: shared_experts.scratch.hidden,
                        shared_expert_gate_logits: shared_experts.scratch.gate_logits,
                        output: next_hidden_state,
                    },
                )));
            },
        }
    }

    fn record_expert_major<'a>(
        &'a self,
        expert_major: &'a expert_major::Compute,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        input: GatedMoEInput<'a>,
    ) {
        let num_total_tokens = input.num_total_tokens;
        let num_active_tokens = input.num_active_tokens;
        let hidden_state = input.hidden_state;
        let next_hidden_state = input.next_hidden_state;
        let scratch = input.scratch;
        let weights = input.weights;
        let shared_experts = input.shared_experts;
        let expert_major_shape = expert_major::Shape { num_total_tokens };
        self.record_routing(
            recorder,
            GatedMoERoutingInput {
                num_total_tokens,
                num_active_tokens: input.num_active_tokens,
                hidden_state,
                scratch: scratch.routing,
                weights: weights.routing(),
            },
        );
        if let Some(shared_experts) = shared_experts {
            self.record_shared_experts(
                recorder,
                num_total_tokens,
                num_active_tokens,
                hidden_state,
                shared_experts.scratch,
                shared_experts.weights,
            );
        }
        let layout = ReplayOp::opaque(expert_major.invoke_layout(
            expert_major_shape,
            num_active_tokens,
            expert_major::LayoutBuffers {
                expert_indices: scratch.routing.expert_indices,
                expert_counts: scratch.topk_experts.expert_counts,
                expert_offsets: scratch.topk_experts.expert_offsets,
                expert_cursors: scratch.topk_experts.expert_cursors,
                routes_by_expert: scratch.topk_experts.routes_by_expert,
                routes_by_token: scratch.topk_experts.routes_by_token,
                experts_by_route: scratch.topk_experts.experts_by_route,
            },
        ));
        if shared_experts.is_some() {
            recorder.record(layout);
        } else {
            recorder.record_with_barrier_before(layout);
        }
        recorder.record_with_barrier_before(ReplayOp::opaque(expert_major.invoke_pack_input(
            expert_major_shape,
            num_active_tokens,
            expert_major::PackInputBuffers {
                input: hidden_state,
                routes_by_expert: scratch.topk_experts.routes_by_expert,
                packed_input: scratch.topk_experts.packed_input,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.topk_experts.invoke_expert_major(
            sparse_mlp::ExpertMajorShape {
                num_total_routes: self.num_total_routes(num_total_tokens),
                num_total_tokens,
                num_experts_per_token: self.num_experts_per_token(),
            },
            num_active_tokens,
            sparse_mlp::ExpertMajorBuffers {
                packed_input: scratch.topk_experts.packed_input,
                experts_by_route: scratch.topk_experts.experts_by_route,
                expert_offsets: scratch.topk_experts.expert_offsets,
                packed_output: scratch.topk_experts.routed_hidden,
            },
            sparse_mlp::Scratch {
                swiglu: scratch.topk_experts.sparse_swiglu,
            },
            weights.topk_experts,
        )));
        match shared_experts {
            None => {
                recorder.record_with_barrier_before(ReplayOp::opaque(
                    expert_major.invoke_scatter_without_shared_experts(
                        expert_major_shape,
                        num_active_tokens,
                        expert_major::ScatterWithoutSharedExpertsBuffers {
                            packed_output: scratch.topk_experts.routed_hidden,
                            routes_by_token: scratch.topk_experts.routes_by_token,
                            routed_probs: scratch.routing.expert_probs,
                            output: next_hidden_state,
                        },
                    ),
                ));
            },
            Some(shared_experts) => {
                recorder.record_with_barrier_before(ReplayOp::opaque(expert_major.invoke_scatter_with_shared_experts(
                    expert_major_shape,
                    num_active_tokens,
                    expert_major::ScatterWithSharedExpertsBuffers {
                        packed_output: scratch.topk_experts.routed_hidden,
                        routes_by_token: scratch.topk_experts.routes_by_token,
                        routed_probs: scratch.routing.expert_probs,
                        shared_hidden: shared_experts.scratch.hidden,
                        shared_expert_gate_logits: shared_experts.scratch.gate_logits,
                        output: next_hidden_state,
                    },
                )));
            },
        }
    }

    fn record_shared_experts<'a, I>(
        &'a self,
        recorder: &mut I,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        input: &'a Buffer,
        scratch: SharedExpertsScratchBindings<'a>,
        weights: GatedMoESharedExpertsWeights<'a>,
    ) where
        I: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let shared_experts = self
            .shared_experts
            .as_ref()
            .expect("gated MoE shared replay requires a configured shared expert");
        recorder.record(ReplayOp::opaque(shared_experts.mlp.invoke(
            dense_mlp::Shape { num_total_tokens },
            num_active_tokens,
            dense_mlp::Buffers {
                hidden_state: input,
                next_hidden_state: scratch.hidden,
            },
            dense_mlp::Scratch {
                swiglu: scratch.dense_mlp.swiglu,
            },
            weights.shared_experts,
        )));
        recorder.record(ReplayOp::opaque(shared_experts.shared_expert_gate.invoke(
            num_total_tokens,
            num_active_tokens,
            scratch.gate_logits,
            0,
            input,
            0,
            weights.shared_expert_gate_weight,
            0,
            weights.shared_expert_gate_scales,
            0,
            weights.shared_expert_gate_biases,
            0,
        )));
    }

    fn num_total_routes(&self, num_total_tokens: u32) -> u32 {
        num_total_tokens
            .checked_mul(self.num_experts_per_token())
            .expect("MoE route count must fit u32")
    }

    fn num_experts_per_token(&self) -> u32 {
        self.core.num_experts_per_token as u32
    }
}

impl ReplayLayer for GatedMoE {
    type Input<'a> = GatedMoEInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        self.validate_input(&input);
        let next_hidden_state = input.next_hidden_state;
        let (_, variant) = Selector::select(&self.registry, input.num_total_tokens);
        match variant {
            Variant::TokenMajor { combine } => self.record_token_major(combine, recorder, input),
            Variant::ExpertMajor { expert_major } => self.record_expert_major(expert_major, recorder, input),
        }
        next_hidden_state
    }
}

fn topk_experts_config(core: &GatedMoECore, config: GatedMoEMetalConfig) -> sparse_mlp::Config {
    sparse_mlp::Config {
        num_experts: core
            .num_experts
            .try_into()
            .expect("MoE sparse expert count must fit u32"),
        hidden_dim: core
            .hidden_dim
            .try_into()
            .expect("MoE sparse expert hidden_dim must fit u32"),
        intermediate_dim: core
            .intermediate_dim
            .try_into()
            .expect("MoE sparse expert intermediate_dim must fit u32"),
        group_size: config.group_size,
        bits: config.bits,
        dtype: config.io_dtype,
    }
}

fn shared_experts_config(core: &DenseMLPCore, config: GatedMoEMetalConfig) -> dense_mlp::Config {
    dense_mlp::Config {
        hidden_dim: core
            .hidden_dim
            .try_into()
            .expect("shared expert hidden_dim must fit u32"),
        intermediate_dim: core
            .intermediate_dim
            .try_into()
            .expect("shared expert intermediate_dim must fit u32"),
        gate_up_group_size: config.group_size,
        gate_up_bits: config.bits,
        gate_up_scale_bias_dtype: config.io_dtype,
        down_group_size: config.group_size,
        down_bits: config.bits,
        down_scale_bias_dtype: config.io_dtype,
        dtype: config.io_dtype,
    }
}

fn affine_config_with_bits(n: usize, k: usize, bits: u32, config: GatedMoEMetalConfig) -> affine_quantized::Config {
    affine_quantized::Config {
        n: n.try_into().expect("MoE affine n must fit i32"),
        k: k.try_into().expect("MoE affine k must fit i32"),
        group_size: config.group_size.try_into().expect("MoE group size must fit i32"),
        bits: bits.try_into().expect("MoE bits must fit i32"),
        input_dtype: config.io_dtype,
        output_dtype: config.io_dtype,
        scale_bias_dtype: config.io_dtype,
    }
}

fn shared_experts_input_matches_core(core: &GatedMoECore, input_has_shared_experts: bool) -> bool {
    core.has_shared_experts() == input_has_shared_experts
}

#[cfg(test)]
#[path = "backend_test.rs"]
mod tests;
