use std::hint::black_box;
use std::mem::size_of;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use inference_backend_metal::components::dense_mlp;
use inference_backend_metal::components::moe::combine;
use inference_backend_metal::components::moe::expert_major;
use inference_backend_metal::components::moe::routing;
use inference_backend_metal::components::sparse_mlp;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::ReplayProgramBuilder;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::affine_quantized;
use inference_backend_metal::operators::softmax;

#[path = "../support.rs"]
mod support;
use support::affine_param_fixture;
use support::bf16_buffer;
use support::gate_fixture;
use support::hidden_fixture;
use support::identity_indices;
use support::quantized_weight;
use support::quantized_weight_stack_for_experts;
use support::route_probs_fixture;
use support::token_route_indices;
use support::zero_fixture;

const NUM_EXPERTS: u32 = 256;
const TOPK_EXPERTS: u32 = 8;
const HIDDEN_DIM: u32 = 2048;
const INTERMEDIATE_DIM: u32 = 512;
const GROUP_SIZE: u32 = 64;
const EXPERT_BITS: u32 = 4;
const ROUTER_BITS: u32 = 8;
const MOE_PROFILE: &str = "qwen36-35b-a3b";
const BENCH_TOKENS: [u32; 7] = [1, 2, 4, 8, 16, 32, 64];

fn affine_config(n: u32, k: u32, bits: u32) -> affine_quantized::Config {
    affine_quantized::Config::same_dtype(
        n.try_into().expect("MoE affine output dimension must fit i32"),
        k.try_into().expect("MoE affine input dimension must fit i32"),
        GROUP_SIZE.try_into().expect("MoE affine group size must fit i32"),
        bits.try_into().expect("MoE affine bits must fit i32"),
        Dtype::Bfloat16,
    )
}

fn bench_moe(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/moe");

    for num_tokens in BENCH_TOKENS {
        for num_invocations in [1, 128] {
            let routing = RoutingFixture::new(&device, num_tokens, num_invocations);
            let path = if num_invocations == 1 {
                "route".to_owned()
            } else {
                format!("route_replay{num_invocations}")
            };
            group.throughput(Throughput::Elements(num_tokens as u64 * num_invocations as u64));
            group.bench_function(
                format!("{MOE_PROFILE}/{path}/num_tokens{num_tokens}/experts{NUM_EXPERTS}/topk{TOPK_EXPERTS}"),
                |b| {
                    b.iter(|| {
                        routing.replay();
                        black_box(&routing.expert_probs);
                    });
                },
            );
        }

        let combine = CombineFixture::new(&device, num_tokens);
        group.throughput(Throughput::Elements(num_tokens as u64 * HIDDEN_DIM as u64));
        group.bench_function(
            format!(
                "{MOE_PROFILE}/combine/without_shared_experts/num_tokens{num_tokens}/topk{TOPK_EXPERTS}/\
                 hidden{HIDDEN_DIM}"
            ),
            |b| {
                b.iter(|| {
                    combine.replay_without_shared_experts();
                    black_box(&combine.output);
                });
            },
        );
        group.bench_function(
            format!(
                "{MOE_PROFILE}/combine/with_shared_experts/num_tokens{num_tokens}/topk{TOPK_EXPERTS}/\
                 hidden{HIDDEN_DIM}"
            ),
            |b| {
                b.iter(|| {
                    combine.replay_with_shared_experts();
                    black_box(&combine.output);
                });
            },
        );
    }

    for num_tokens in BENCH_TOKENS {
        let forward = MoEForwardFixture::new(&device, num_tokens);
        group.throughput(Throughput::Elements(num_tokens as u64 * HIDDEN_DIM as u64));
        group.bench_function(
            format!(
                "{MOE_PROFILE}/forward/token_major/num_tokens{num_tokens}/experts{NUM_EXPERTS}/topk{TOPK_EXPERTS}/\
                 hidden{HIDDEN_DIM}/intermediate{INTERMEDIATE_DIM}"
            ),
            |b| {
                b.iter(|| {
                    forward.run_token_major_replay();
                    black_box(&forward.replay_output);
                });
            },
        );
        group.bench_function(
            format!(
                "{MOE_PROFILE}/forward/expert_major/num_tokens{num_tokens}/experts{NUM_EXPERTS}/topk{TOPK_EXPERTS}/\
                 hidden{HIDDEN_DIM}/intermediate{INTERMEDIATE_DIM}"
            ),
            |b| {
                b.iter(|| {
                    forward.run_expert_major_replay();
                    black_box(&forward.expert_major_output);
                });
            },
        );
    }

    group.finish();
}

struct RoutingFixture {
    stream: Stream,
    kernel: routing::Compute,
    shape: routing::Shape,
    router_probs: Buffer,
    expert_indices: Buffer,
    expert_probs: Buffer,
    replay: ReplayProgram,
}

impl RoutingFixture {
    fn new(device: &Device, num_tokens: u32, num_invocations: usize) -> Self {
        let config = routing::Config {
            num_experts: NUM_EXPERTS,
            num_experts_per_token: TOPK_EXPERTS,
            norm_topk_prob: true,
        };
        let shape = routing::Shape {
            num_total_tokens: num_tokens,
        };
        let router_probs = bf16_buffer(device, &route_probs_fixture(num_tokens as usize, NUM_EXPERTS as usize));
        let expert_indices = Buffer::new_zeroed(device, config.expert_indices_bytes(shape));
        let expert_probs = Buffer::new_zeroed(device, config.expert_probs_bytes(shape));
        let stream = Stream::new(device);
        let kernel = routing::Compute::new(device, config);
        let replay = build_routing_replay(
            &stream,
            &kernel,
            shape,
            &router_probs,
            &expert_indices,
            &expert_probs,
            num_invocations,
        );
        let fixture = Self {
            stream,
            kernel,
            shape,
            router_probs,
            expert_indices,
            expert_probs,
            replay,
        };
        fixture.replay();
        fixture
    }

    fn replay(&self) {
        self.stream.submit_replay(&self.replay).wait();
    }
}

struct CombineFixture {
    stream: Stream,
    kernels: combine::Compute,
    shape: combine::Shape,
    routed_hidden: Buffer,
    routed_probs: Buffer,
    shared_hidden: Buffer,
    shared_expert_gate_logits: Buffer,
    output: Buffer,
    without_shared_experts_replay: ReplayProgram,
    with_shared_experts_replay: ReplayProgram,
}

impl CombineFixture {
    fn new(device: &Device, num_tokens: u32) -> Self {
        let config = combine::Config::bf16(TOPK_EXPERTS, HIDDEN_DIM);
        let shape = combine::Shape {
            num_total_tokens: num_tokens,
        };
        let routed_hidden = bf16_buffer(
            device,
            &hidden_fixture(num_tokens as usize * TOPK_EXPERTS as usize, HIDDEN_DIM as usize),
        );
        let routed_probs = Buffer::from_slice(device, &route_probs_fixture(num_tokens as usize, TOPK_EXPERTS as usize));
        let shared_hidden = bf16_buffer(device, &hidden_fixture(num_tokens as usize, HIDDEN_DIM as usize));
        let shared_expert_gate_logits = bf16_buffer(device, &gate_fixture(num_tokens as usize));
        let output = Buffer::new_zeroed(device, num_tokens as usize * HIDDEN_DIM as usize * size_of::<u16>());
        let stream = Stream::new(device);
        let kernels = combine::Compute::new(device, config);
        let without_shared_experts_replay = build_combine_without_shared_experts_replay(
            &stream,
            &kernels,
            shape,
            &routed_hidden,
            &routed_probs,
            &output,
        );
        let with_shared_experts_replay = build_combine_with_shared_experts_replay(
            &stream,
            &kernels,
            shape,
            combine::WithSharedExpertsBuffers {
                routed_hidden: &routed_hidden,
                routed_probs: &routed_probs,
                shared_hidden: &shared_hidden,
                shared_expert_gate_logits: &shared_expert_gate_logits,
                output: &output,
            },
        );
        let fixture = Self {
            stream,
            kernels,
            shape,
            routed_hidden,
            routed_probs,
            shared_hidden,
            shared_expert_gate_logits,
            output,
            without_shared_experts_replay,
            with_shared_experts_replay,
        };
        fixture.replay_without_shared_experts();
        fixture.replay_with_shared_experts();
        fixture
    }

    fn replay_without_shared_experts(&self) {
        self.stream.submit_replay(&self.without_shared_experts_replay).wait();
    }

    fn replay_with_shared_experts(&self) {
        self.stream.submit_replay(&self.with_shared_experts_replay).wait();
    }
}

fn build_routing_replay(
    stream: &Stream,
    kernel: &routing::Compute,
    shape: routing::Shape,
    router_probs: &Buffer,
    expert_indices: &Buffer,
    expert_probs: &Buffer,
    num_invocations: usize,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    for _ in 0..num_invocations {
        builder.record(kernel.invoke(
            shape,
            ReplayU32::Fixed(shape.num_total_tokens),
            routing::Buffers {
                router_probs,
                expert_indices,
                expert_probs,
            },
        ));
    }
    builder.build()
}

fn build_combine_without_shared_experts_replay(
    stream: &Stream,
    kernel: &combine::Compute,
    shape: combine::Shape,
    routed_hidden: &Buffer,
    routed_probs: &Buffer,
    output: &Buffer,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    builder.record(kernel.invoke_without_shared_experts(
        shape,
        inference_backend_metal::metal::ReplayU32::Fixed(shape.num_total_tokens),
        combine::WithoutSharedExpertsBuffers {
            routed_hidden,
            routed_probs,
            output,
        },
    ));
    builder.build()
}

fn build_combine_with_shared_experts_replay(
    stream: &Stream,
    kernel: &combine::Compute,
    shape: combine::Shape,
    buffers: combine::WithSharedExpertsBuffers<'_>,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    builder.record(kernel.invoke_with_shared_experts(
        shape,
        inference_backend_metal::metal::ReplayU32::Fixed(shape.num_total_tokens),
        buffers,
    ));
    builder.build()
}

struct MoEForwardFixture {
    stream: Stream,
    routing_shape: routing::Shape,
    sparse_shape: sparse_mlp::TokenMajorShape,
    expert_major_config: expert_major::Config,
    expert_major_shape: expert_major::Shape,
    dense_shape: dense_mlp::Shape,
    combine_shape: combine::Shape,
    router: affine_quantized::Matmul,
    router_softmax: softmax::Kernel,
    shared_expert_gate: affine_quantized::Matmul,
    routing: routing::Compute,
    expert_major: expert_major::Compute,
    sparse_mlp: sparse_mlp::Compute,
    shared_experts: dense_mlp::Compute,
    combine: combine::Compute,
    input: Buffer,
    router_logits: Buffer,
    router_probs: Buffer,
    expert_indices: Buffer,
    expert_probs: Buffer,
    token_indices: Buffer,
    route_indices: Buffer,
    expert_counts: Buffer,
    expert_offsets: Buffer,
    expert_cursors: Buffer,
    routes_by_expert: Buffer,
    routes_by_token: Buffer,
    experts_by_route: Buffer,
    packed_input: Buffer,
    routed_hidden: Buffer,
    expert_major_routed_hidden: Buffer,
    sparse_swiglu: Buffer,
    expert_major_swiglu: Buffer,
    shared_hidden: Buffer,
    shared_expert_gate_logits: Buffer,
    shared_expert_gate_weight: Buffer,
    shared_expert_gate_scales: Buffer,
    shared_expert_gate_biases: Buffer,
    router_weight: Buffer,
    router_scales: Buffer,
    router_biases: Buffer,
    sparse_weights: SparseMLPWeights,
    shared_weights: DenseMLPWeights,
    shared_scratch: DenseMLPScratch,
    replay_output: Buffer,
    expert_major_output: Buffer,
    replay: ReplayProgram,
    expert_major_replay: ReplayProgram,
}

impl MoEForwardFixture {
    fn new(device: &Device, num_tokens: u32) -> Self {
        let stream = Stream::new(device);
        let router_config = affine_config(NUM_EXPERTS, HIDDEN_DIM, ROUTER_BITS);
        let shared_expert_gate_config = affine_config(1, HIDDEN_DIM, ROUTER_BITS);
        let num_tokens_i32 = num_tokens.try_into().expect("MoE token count must fit i32");
        let routing_config = routing::Config {
            num_experts: NUM_EXPERTS,
            num_experts_per_token: TOPK_EXPERTS,
            norm_topk_prob: true,
        };
        let routing_shape = routing::Shape {
            num_total_tokens: num_tokens,
        };
        let sparse_config = sparse_mlp::Config {
            num_experts: NUM_EXPERTS,
            hidden_dim: HIDDEN_DIM,
            intermediate_dim: INTERMEDIATE_DIM,
            group_size: GROUP_SIZE,
            bits: EXPERT_BITS,
            dtype: Dtype::Bfloat16,
        };
        let sparse_shape = sparse_mlp::TokenMajorShape {
            num_total_routes: num_tokens * TOPK_EXPERTS,
            num_total_tokens: num_tokens,
        };
        let expert_major_config = expert_major::Config::bf16(NUM_EXPERTS, TOPK_EXPERTS, HIDDEN_DIM);
        let expert_major_shape = expert_major::Shape {
            num_total_tokens: num_tokens,
        };
        let expert_major_sparse_shape = sparse_mlp::TokenMajorShape {
            num_total_routes: expert_major_config.num_routes(expert_major_shape),
            num_total_tokens: expert_major_config.num_routes(expert_major_shape),
        };
        let dense_config = dense_mlp::Config {
            hidden_dim: HIDDEN_DIM,
            intermediate_dim: INTERMEDIATE_DIM,
            gate_up_group_size: GROUP_SIZE,
            gate_up_bits: EXPERT_BITS,
            gate_up_scale_bias_dtype: Dtype::Bfloat16,
            down_group_size: GROUP_SIZE,
            down_bits: EXPERT_BITS,
            down_scale_bias_dtype: Dtype::Bfloat16,
            dtype: Dtype::Bfloat16,
        };
        let dense_shape = dense_mlp::Shape {
            num_total_tokens: num_tokens,
        };
        let combine_config = combine::Config::bf16(TOPK_EXPERTS, HIDDEN_DIM);
        let combine_shape = combine::Shape {
            num_total_tokens: num_tokens,
        };
        let sparse_gate_up_config = sparse_config.gate_up_config();
        let sparse_down_config = sparse_config.down_config();
        let dense_gate_up_config = dense_config.gate_up_config();
        let dense_down_config = dense_config.down_config();
        let num_routes = num_tokens as usize * TOPK_EXPERTS as usize;
        let input = bf16_buffer(device, &hidden_fixture(num_tokens as usize, HIDDEN_DIM as usize));
        let router_logits = Buffer::new_zeroed(device, router_config.output_bytes(num_tokens_i32));
        let router_probs = Buffer::new_zeroed(device, router_config.output_bytes(num_tokens_i32));
        let expert_indices = Buffer::new_zeroed(device, routing_config.expert_indices_bytes(routing_shape));
        let expert_probs = Buffer::new_zeroed(device, routing_config.expert_probs_bytes(routing_shape));
        let token_indices =
            Buffer::from_slice(device, &token_route_indices(num_tokens as usize, TOPK_EXPERTS as usize));
        let route_indices = Buffer::from_slice(device, &identity_indices(num_routes));
        let expert_counts = Buffer::new_zeroed(device, expert_major_config.expert_counts_bytes());
        let expert_offsets = Buffer::new_zeroed(device, expert_major_config.expert_offsets_bytes());
        let expert_cursors = Buffer::new_zeroed(device, expert_major_config.expert_counts_bytes());
        let routes_by_expert = Buffer::new_zeroed(device, expert_major_config.route_indices_bytes(expert_major_shape));
        let routes_by_token = Buffer::new_zeroed(device, expert_major_config.route_indices_bytes(expert_major_shape));
        let experts_by_route = Buffer::new_zeroed(device, expert_major_config.route_indices_bytes(expert_major_shape));
        let packed_input = Buffer::new_zeroed(device, expert_major_config.route_hidden_bytes(expert_major_shape));
        let routed_hidden = Buffer::new_zeroed(device, sparse_config.token_major_output_bytes(sparse_shape));
        let expert_major_routed_hidden = Buffer::new_zeroed(
            device,
            sparse_config.token_major_output_bytes(expert_major_sparse_shape),
        );
        let sparse_swiglu = Buffer::new_zeroed(device, sparse_config.swiglu_bytes(sparse_shape.num_total_routes));
        let expert_major_swiglu = Buffer::new_zeroed(
            device,
            sparse_config.swiglu_bytes(expert_major_sparse_shape.num_total_routes),
        );
        let shared_hidden = Buffer::new_zeroed(device, dense_down_config.output_bytes(num_tokens_i32));
        let shared_expert_gate_logits =
            Buffer::new_zeroed(device, shared_expert_gate_config.output_bytes(num_tokens_i32));
        let replay_output = Buffer::new_zeroed(device, combine_config.output_bytes(combine_shape));
        let expert_major_output = Buffer::new_zeroed(device, combine_config.output_bytes(combine_shape));
        let router = affine_quantized::Matmul::new(device, router_config);
        let router_softmax = softmax::Kernel::new(
            device,
            softmax::Config {
                num_values_per_row: NUM_EXPERTS,
                dtype: Dtype::Bfloat16,
            },
        );
        let shared_expert_gate = affine_quantized::Matmul::new(device, shared_expert_gate_config);
        let routing = routing::Compute::new(device, routing_config);
        let expert_major = expert_major::Compute::new(device, expert_major_config);
        let sparse_mlp = sparse_mlp::Compute::new(device, sparse_config);
        let shared_experts = dense_mlp::Compute::new(device, dense_config);
        let combine = combine::Compute::new(device, combine_config);
        let router_weight = quantized_weight(device, router_config.weight_bytes());
        let router_scales = bf16_buffer(
            device,
            &affine_param_fixture(router_config.scale_or_bias_bytes() / size_of::<u16>()),
        );
        let router_biases = bf16_buffer(
            device,
            &zero_fixture(router_config.scale_or_bias_bytes() / size_of::<u16>()),
        );
        let shared_expert_gate_weight = quantized_weight(device, shared_expert_gate_config.weight_bytes());
        let shared_expert_gate_scales = bf16_buffer(
            device,
            &affine_param_fixture(shared_expert_gate_config.scale_or_bias_bytes() / size_of::<u16>()),
        );
        let shared_expert_gate_biases = bf16_buffer(
            device,
            &zero_fixture(shared_expert_gate_config.scale_or_bias_bytes() / size_of::<u16>()),
        );
        let sparse_weights = SparseMLPWeights {
            gate_weight: quantized_weight_stack_for_experts(
                device,
                NUM_EXPERTS as usize,
                sparse_gate_up_config.weight_bytes_per_expert(),
            ),
            gate_scales: bf16_buffer(
                device,
                &affine_param_fixture(
                    NUM_EXPERTS as usize * sparse_gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>(),
                ),
            ),
            gate_biases: bf16_buffer(
                device,
                &zero_fixture(
                    NUM_EXPERTS as usize * sparse_gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>(),
                ),
            ),
            up_weight: quantized_weight_stack_for_experts(
                device,
                NUM_EXPERTS as usize,
                sparse_gate_up_config.weight_bytes_per_expert(),
            ),
            up_scales: bf16_buffer(
                device,
                &affine_param_fixture(
                    NUM_EXPERTS as usize * sparse_gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>(),
                ),
            ),
            up_biases: bf16_buffer(
                device,
                &zero_fixture(
                    NUM_EXPERTS as usize * sparse_gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>(),
                ),
            ),
            down_weight: quantized_weight_stack_for_experts(
                device,
                NUM_EXPERTS as usize,
                sparse_down_config.weight_bytes_per_expert(),
            ),
            down_scales: bf16_buffer(
                device,
                &affine_param_fixture(
                    NUM_EXPERTS as usize * sparse_down_config.affine_param_bytes_per_expert() / size_of::<u16>(),
                ),
            ),
            down_biases: bf16_buffer(
                device,
                &zero_fixture(
                    NUM_EXPERTS as usize * sparse_down_config.affine_param_bytes_per_expert() / size_of::<u16>(),
                ),
            ),
        };
        let shared_weights = DenseMLPWeights {
            gate_up_weight: quantized_weight(device, dense_gate_up_config.weight_bytes()),
            gate_up_scales: bf16_buffer(
                device,
                &affine_param_fixture(dense_gate_up_config.scale_or_bias_bytes() / size_of::<u16>()),
            ),
            gate_up_biases: bf16_buffer(
                device,
                &zero_fixture(dense_gate_up_config.scale_or_bias_bytes() / size_of::<u16>()),
            ),
            down_weight: quantized_weight(device, dense_down_config.weight_bytes()),
            down_scales: bf16_buffer(
                device,
                &affine_param_fixture(dense_down_config.scale_or_bias_bytes() / size_of::<u16>()),
            ),
            down_biases: bf16_buffer(
                device,
                &zero_fixture(dense_down_config.scale_or_bias_bytes() / size_of::<u16>()),
            ),
        };
        let shared_scratch = DenseMLPScratch {
            swiglu: Buffer::new_zeroed(device, dense_config.swiglu_bytes(dense_shape)),
        };
        let replay = build_moe_forward_replay(
            &stream,
            MoEForwardRecord {
                routing_shape,
                sparse_shape,
                dense_shape,
                combine_shape,
                router: &router,
                router_softmax: &router_softmax,
                shared_expert_gate: &shared_expert_gate,
                routing: &routing,
                sparse_mlp: &sparse_mlp,
                shared_experts: &shared_experts,
                combine: &combine,
                input: &input,
                router_logits: &router_logits,
                router_probs: &router_probs,
                expert_indices: &expert_indices,
                expert_probs: &expert_probs,
                token_indices: &token_indices,
                route_indices: &route_indices,
                routed_hidden: &routed_hidden,
                sparse_swiglu: &sparse_swiglu,
                shared_hidden: &shared_hidden,
                shared_expert_gate_logits: &shared_expert_gate_logits,
                output: &replay_output,
                router_weight: &router_weight,
                router_scales: &router_scales,
                router_biases: &router_biases,
                shared_expert_gate_weight: &shared_expert_gate_weight,
                shared_expert_gate_scales: &shared_expert_gate_scales,
                shared_expert_gate_biases: &shared_expert_gate_biases,
                sparse_weights: sparse_weights.as_borrowed(),
                shared_weights: shared_weights.as_borrowed(),
                shared_scratch: shared_scratch.as_borrowed(),
            },
        );
        let expert_major_replay = build_moe_expert_major_forward_replay(
            &stream,
            MoEExpertMajorForwardRecord {
                routing_shape,
                sparse_shape: expert_major_sparse_shape,
                expert_major_config,
                expert_major_shape,
                dense_shape,
                router: &router,
                router_softmax: &router_softmax,
                shared_expert_gate: &shared_expert_gate,
                routing: &routing,
                expert_major: &expert_major,
                sparse_mlp: &sparse_mlp,
                shared_experts: &shared_experts,
                input: &input,
                router_logits: &router_logits,
                router_probs: &router_probs,
                expert_indices: &expert_indices,
                expert_probs: &expert_probs,
                route_indices: &route_indices,
                expert_counts: &expert_counts,
                expert_offsets: &expert_offsets,
                expert_cursors: &expert_cursors,
                routes_by_expert: &routes_by_expert,
                routes_by_token: &routes_by_token,
                experts_by_route: &experts_by_route,
                packed_input: &packed_input,
                routed_hidden: &expert_major_routed_hidden,
                sparse_swiglu: &expert_major_swiglu,
                shared_hidden: &shared_hidden,
                shared_expert_gate_logits: &shared_expert_gate_logits,
                output: &expert_major_output,
                router_weight: &router_weight,
                router_scales: &router_scales,
                router_biases: &router_biases,
                shared_expert_gate_weight: &shared_expert_gate_weight,
                shared_expert_gate_scales: &shared_expert_gate_scales,
                shared_expert_gate_biases: &shared_expert_gate_biases,
                sparse_weights: sparse_weights.as_borrowed(),
                shared_weights: shared_weights.as_borrowed(),
                shared_scratch: shared_scratch.as_borrowed(),
            },
        );
        let fixture = Self {
            stream,
            routing_shape,
            sparse_shape,
            expert_major_config,
            expert_major_shape,
            dense_shape,
            combine_shape,
            router,
            router_softmax,
            shared_expert_gate,
            routing,
            expert_major,
            sparse_mlp,
            shared_experts,
            combine,
            input,
            router_logits,
            router_probs,
            expert_indices,
            expert_probs,
            token_indices,
            route_indices,
            expert_counts,
            expert_offsets,
            expert_cursors,
            routes_by_expert,
            routes_by_token,
            experts_by_route,
            packed_input,
            routed_hidden,
            expert_major_routed_hidden,
            sparse_swiglu,
            expert_major_swiglu,
            shared_hidden,
            shared_expert_gate_logits,
            shared_expert_gate_weight,
            shared_expert_gate_scales,
            shared_expert_gate_biases,
            router_weight,
            router_scales,
            router_biases,
            sparse_weights,
            shared_weights,
            shared_scratch,
            replay_output,
            expert_major_output,
            replay,
            expert_major_replay,
        };
        fixture.assert_token_major_and_expert_major_replay_match();
        let counts = fixture.expert_counts.read_typed::<u32>(0, NUM_EXPERTS as usize);
        let active = counts.iter().filter(|&&count| count > 0).count();
        let max_rows = counts.iter().copied().max().unwrap();
        eprintln!("MoE fixture tokens={num_tokens} active_experts={active} max_rows_per_expert={max_rows}");
        fixture
    }

    fn run_token_major_replay(&self) {
        self.stream.submit_replay(&self.replay).wait();
    }

    fn run_expert_major_replay(&self) {
        self.stream.submit_replay(&self.expert_major_replay).wait();
    }

    fn record<'a>(&'a self, output: &'a Buffer) -> MoEForwardRecord<'a> {
        MoEForwardRecord {
            routing_shape: self.routing_shape,
            sparse_shape: self.sparse_shape,
            dense_shape: self.dense_shape,
            combine_shape: self.combine_shape,
            router: &self.router,
            router_softmax: &self.router_softmax,
            shared_expert_gate: &self.shared_expert_gate,
            routing: &self.routing,
            sparse_mlp: &self.sparse_mlp,
            shared_experts: &self.shared_experts,
            combine: &self.combine,
            input: &self.input,
            router_logits: &self.router_logits,
            router_probs: &self.router_probs,
            expert_indices: &self.expert_indices,
            expert_probs: &self.expert_probs,
            token_indices: &self.token_indices,
            route_indices: &self.route_indices,
            routed_hidden: &self.routed_hidden,
            sparse_swiglu: &self.sparse_swiglu,
            shared_hidden: &self.shared_hidden,
            shared_expert_gate_logits: &self.shared_expert_gate_logits,
            output,
            router_weight: &self.router_weight,
            router_scales: &self.router_scales,
            router_biases: &self.router_biases,
            shared_expert_gate_weight: &self.shared_expert_gate_weight,
            shared_expert_gate_scales: &self.shared_expert_gate_scales,
            shared_expert_gate_biases: &self.shared_expert_gate_biases,
            sparse_weights: self.sparse_weights.as_borrowed(),
            shared_weights: self.shared_weights.as_borrowed(),
            shared_scratch: self.shared_scratch.as_borrowed(),
        }
    }

    fn expert_major_record<'a>(&'a self, output: &'a Buffer) -> MoEExpertMajorForwardRecord<'a> {
        MoEExpertMajorForwardRecord {
            routing_shape: self.routing_shape,
            sparse_shape: sparse_mlp::TokenMajorShape {
                num_total_routes: self.expert_major_config.num_routes(self.expert_major_shape),
                num_total_tokens: self.expert_major_config.num_routes(self.expert_major_shape),
            },
            expert_major_config: self.expert_major_config,
            expert_major_shape: self.expert_major_shape,
            dense_shape: self.dense_shape,
            router: &self.router,
            router_softmax: &self.router_softmax,
            shared_expert_gate: &self.shared_expert_gate,
            routing: &self.routing,
            expert_major: &self.expert_major,
            sparse_mlp: &self.sparse_mlp,
            shared_experts: &self.shared_experts,
            input: &self.input,
            router_logits: &self.router_logits,
            router_probs: &self.router_probs,
            expert_indices: &self.expert_indices,
            expert_probs: &self.expert_probs,
            route_indices: &self.route_indices,
            expert_counts: &self.expert_counts,
            expert_offsets: &self.expert_offsets,
            expert_cursors: &self.expert_cursors,
            routes_by_expert: &self.routes_by_expert,
            routes_by_token: &self.routes_by_token,
            experts_by_route: &self.experts_by_route,
            packed_input: &self.packed_input,
            routed_hidden: &self.expert_major_routed_hidden,
            sparse_swiglu: &self.expert_major_swiglu,
            shared_hidden: &self.shared_hidden,
            shared_expert_gate_logits: &self.shared_expert_gate_logits,
            output,
            router_weight: &self.router_weight,
            router_scales: &self.router_scales,
            router_biases: &self.router_biases,
            shared_expert_gate_weight: &self.shared_expert_gate_weight,
            shared_expert_gate_scales: &self.shared_expert_gate_scales,
            shared_expert_gate_biases: &self.shared_expert_gate_biases,
            sparse_weights: self.sparse_weights.as_borrowed(),
            shared_weights: self.shared_weights.as_borrowed(),
            shared_scratch: self.shared_scratch.as_borrowed(),
        }
    }

    fn assert_token_major_and_expert_major_replay_match(&self) {
        self.run_token_major_replay();
        self.run_expert_major_replay();
        let replay = self
            .replay_output
            .read_typed::<u16>(0, self.replay_output.len_bytes() / size_of::<u16>());
        let expert_major = self
            .expert_major_output
            .read_typed::<u16>(0, self.expert_major_output.len_bytes() / size_of::<u16>());
        // QMM and QMV use different F32 reduction orders. Use the sparse
        // owner's CPU-reference tolerance instead of requiring identical bits.
        for (index, (&expected, &actual)) in replay.iter().zip(&expert_major).enumerate() {
            let expected = half::bf16::from_bits(expected).to_f32();
            let actual = half::bf16::from_bits(actual).to_f32();
            assert!(
                (actual - expected).abs() <= 2.0e-5 + 8.0e-3 * expected.abs(),
                "MoE token-major/expert-major mismatch at {index}: expected={expected}, actual={actual}"
            );
        }
    }
}

struct MoEForwardRecord<'a> {
    routing_shape: routing::Shape,
    sparse_shape: sparse_mlp::TokenMajorShape,
    dense_shape: dense_mlp::Shape,
    combine_shape: combine::Shape,
    router: &'a affine_quantized::Matmul,
    router_softmax: &'a softmax::Kernel,
    shared_expert_gate: &'a affine_quantized::Matmul,
    routing: &'a routing::Compute,
    sparse_mlp: &'a sparse_mlp::Compute,
    shared_experts: &'a dense_mlp::Compute,
    combine: &'a combine::Compute,
    input: &'a Buffer,
    router_logits: &'a Buffer,
    router_probs: &'a Buffer,
    expert_indices: &'a Buffer,
    expert_probs: &'a Buffer,
    token_indices: &'a Buffer,
    route_indices: &'a Buffer,
    routed_hidden: &'a Buffer,
    sparse_swiglu: &'a Buffer,
    shared_hidden: &'a Buffer,
    shared_expert_gate_logits: &'a Buffer,
    output: &'a Buffer,
    router_weight: &'a Buffer,
    router_scales: &'a Buffer,
    router_biases: &'a Buffer,
    shared_expert_gate_weight: &'a Buffer,
    shared_expert_gate_scales: &'a Buffer,
    shared_expert_gate_biases: &'a Buffer,
    sparse_weights: sparse_mlp::Weights<'a>,
    shared_weights: dense_mlp::Weights<'a>,
    shared_scratch: dense_mlp::Scratch<'a>,
}

fn build_moe_forward_replay(stream: &Stream, record: MoEForwardRecord<'_>) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    record_moe_forward(&mut builder, record);
    builder.build()
}

struct MoEExpertMajorForwardRecord<'a> {
    routing_shape: routing::Shape,
    sparse_shape: sparse_mlp::TokenMajorShape,
    expert_major_config: expert_major::Config,
    expert_major_shape: expert_major::Shape,
    dense_shape: dense_mlp::Shape,
    router: &'a affine_quantized::Matmul,
    router_softmax: &'a softmax::Kernel,
    shared_expert_gate: &'a affine_quantized::Matmul,
    routing: &'a routing::Compute,
    expert_major: &'a expert_major::Compute,
    sparse_mlp: &'a sparse_mlp::Compute,
    shared_experts: &'a dense_mlp::Compute,
    input: &'a Buffer,
    router_logits: &'a Buffer,
    router_probs: &'a Buffer,
    expert_indices: &'a Buffer,
    expert_probs: &'a Buffer,
    route_indices: &'a Buffer,
    expert_counts: &'a Buffer,
    expert_offsets: &'a Buffer,
    expert_cursors: &'a Buffer,
    routes_by_expert: &'a Buffer,
    routes_by_token: &'a Buffer,
    experts_by_route: &'a Buffer,
    packed_input: &'a Buffer,
    routed_hidden: &'a Buffer,
    sparse_swiglu: &'a Buffer,
    shared_hidden: &'a Buffer,
    shared_expert_gate_logits: &'a Buffer,
    output: &'a Buffer,
    router_weight: &'a Buffer,
    router_scales: &'a Buffer,
    router_biases: &'a Buffer,
    shared_expert_gate_weight: &'a Buffer,
    shared_expert_gate_scales: &'a Buffer,
    shared_expert_gate_biases: &'a Buffer,
    sparse_weights: sparse_mlp::Weights<'a>,
    shared_weights: dense_mlp::Weights<'a>,
    shared_scratch: dense_mlp::Scratch<'a>,
}

fn build_moe_expert_major_forward_replay(stream: &Stream, record: MoEExpertMajorForwardRecord<'_>) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    record_moe_expert_major_forward(&mut builder, record);
    builder.build()
}

fn record_moe_forward<I>(builder: &mut I, record: MoEForwardRecord<'_>)
where
    I: MoEForwardBuilder,
{
    let num_total_tokens = record.routing_shape.num_total_tokens;
    let num_active_tokens = inference_backend_metal::metal::ReplayU32::Fixed(num_total_tokens);
    builder.record(record.router.invoke(
        num_total_tokens,
        num_active_tokens,
        record.router_logits,
        0,
        record.input,
        0,
        record.router_weight,
        0,
        record.router_scales,
        0,
        record.router_biases,
        0,
    ));
    builder.record_with_barrier_before(record.router_softmax.invoke(
        softmax::Shape {
            num_total_rows: num_total_tokens,
        },
        num_active_tokens,
        softmax::Buffers {
            input: record.router_logits,
            output: record.router_probs,
        },
    ));
    builder.record_with_barrier_before(record.routing.invoke(
        record.routing_shape,
        num_active_tokens,
        routing::Buffers {
            router_probs: record.router_probs,
            expert_indices: record.expert_indices,
            expert_probs: record.expert_probs,
        },
    ));
    builder.record_with_barrier_before(record.sparse_mlp.invoke_token_major(
        record.sparse_shape,
        record.sparse_shape.num_total_routes / num_total_tokens,
        num_active_tokens,
        sparse_mlp::TokenMajorBuffers {
            input: record.input,
            token_indices: record.token_indices,
            expert_indices: record.expert_indices,
            route_indices: record.route_indices,
            routed_hidden: record.routed_hidden,
        },
        sparse_mlp::Scratch {
            swiglu: record.sparse_swiglu,
        },
        record.sparse_weights,
    ));
    builder.record(record.shared_experts.invoke(
        record.dense_shape,
        num_active_tokens,
        dense_mlp::Buffers {
            hidden_state: record.input,
            next_hidden_state: record.shared_hidden,
        },
        record.shared_scratch,
        record.shared_weights,
    ));
    builder.record(record.shared_expert_gate.invoke(
        num_total_tokens,
        num_active_tokens,
        record.shared_expert_gate_logits,
        0,
        record.input,
        0,
        record.shared_expert_gate_weight,
        0,
        record.shared_expert_gate_scales,
        0,
        record.shared_expert_gate_biases,
        0,
    ));
    builder.record_with_barrier_before(record.combine.invoke_with_shared_experts(
        record.combine_shape,
        num_active_tokens,
        combine::WithSharedExpertsBuffers {
            routed_hidden: record.routed_hidden,
            routed_probs: record.expert_probs,
            shared_hidden: record.shared_hidden,
            shared_expert_gate_logits: record.shared_expert_gate_logits,
            output: record.output,
        },
    ));
}

fn record_moe_expert_major_forward<I>(builder: &mut I, record: MoEExpertMajorForwardRecord<'_>)
where
    I: MoEForwardBuilder,
{
    let num_total_tokens = record.routing_shape.num_total_tokens;
    let num_active_tokens = inference_backend_metal::metal::ReplayU32::Fixed(num_total_tokens);
    builder.record(record.router.invoke(
        num_total_tokens,
        num_active_tokens,
        record.router_logits,
        0,
        record.input,
        0,
        record.router_weight,
        0,
        record.router_scales,
        0,
        record.router_biases,
        0,
    ));
    builder.record_with_barrier_before(record.router_softmax.invoke(
        softmax::Shape {
            num_total_rows: num_total_tokens,
        },
        num_active_tokens,
        softmax::Buffers {
            input: record.router_logits,
            output: record.router_probs,
        },
    ));
    builder.record_with_barrier_before(record.routing.invoke(
        record.routing_shape,
        num_active_tokens,
        routing::Buffers {
            router_probs: record.router_probs,
            expert_indices: record.expert_indices,
            expert_probs: record.expert_probs,
        },
    ));
    builder.record_with_barrier_before(record.expert_major.invoke_layout(
        record.expert_major_shape,
        num_active_tokens,
        expert_major::LayoutBuffers {
            expert_indices: record.expert_indices,
            expert_counts: record.expert_counts,
            expert_offsets: record.expert_offsets,
            expert_cursors: record.expert_cursors,
            routes_by_expert: record.routes_by_expert,
            routes_by_token: record.routes_by_token,
            experts_by_route: record.experts_by_route,
        },
    ));
    builder.record_with_barrier_before(record.expert_major.invoke_pack_input(
        record.expert_major_shape,
        num_active_tokens,
        expert_major::PackInputBuffers {
            input: record.input,
            routes_by_expert: record.routes_by_expert,
            packed_input: record.packed_input,
        },
    ));
    builder.record_with_barrier_before(record.sparse_mlp.invoke_expert_major(
        sparse_mlp::ExpertMajorShape {
            num_total_routes: record.expert_major_config.num_routes(record.expert_major_shape),
            num_total_tokens,
            num_experts_per_token: record.expert_major_config.num_experts_per_token,
        },
        num_active_tokens,
        sparse_mlp::ExpertMajorBuffers {
            packed_input: record.packed_input,
            experts_by_route: record.experts_by_route,
            expert_offsets: record.expert_offsets,
            packed_output: record.routed_hidden,
        },
        sparse_mlp::Scratch {
            swiglu: record.sparse_swiglu,
        },
        record.sparse_weights,
    ));
    builder.record(record.shared_experts.invoke(
        record.dense_shape,
        num_active_tokens,
        dense_mlp::Buffers {
            hidden_state: record.input,
            next_hidden_state: record.shared_hidden,
        },
        record.shared_scratch,
        record.shared_weights,
    ));
    builder.record(record.shared_expert_gate.invoke(
        num_total_tokens,
        num_active_tokens,
        record.shared_expert_gate_logits,
        0,
        record.input,
        0,
        record.shared_expert_gate_weight,
        0,
        record.shared_expert_gate_scales,
        0,
        record.shared_expert_gate_biases,
        0,
    ));
    builder.record_with_barrier_before(record.expert_major.invoke_scatter_with_shared_experts(
        record.expert_major_shape,
        num_active_tokens,
        expert_major::ScatterWithSharedExpertsBuffers {
            packed_output: record.routed_hidden,
            routes_by_token: record.routes_by_token,
            routed_probs: record.expert_probs,
            shared_hidden: record.shared_hidden,
            shared_expert_gate_logits: record.shared_expert_gate_logits,
            output: record.output,
        },
    ));
}

trait MoEForwardBuilder {
    fn record<T: inference_backend_metal::metal::Operator>(&mut self, invocation: T);
    fn record_with_barrier_before<T: inference_backend_metal::metal::Operator>(&mut self, invocation: T);
}

impl MoEForwardBuilder for inference_backend_metal::metal::ReplayProgramBuilder {
    fn record<T: inference_backend_metal::metal::Operator>(&mut self, invocation: T) {
        ReplayProgramBuilder::record(self, invocation);
    }

    fn record_with_barrier_before<T: inference_backend_metal::metal::Operator>(&mut self, invocation: T) {
        ReplayProgramBuilder::record_with_barrier_before(self, invocation);
    }
}

struct SparseMLPWeights {
    gate_weight: Buffer,
    gate_scales: Buffer,
    gate_biases: Buffer,
    up_weight: Buffer,
    up_scales: Buffer,
    up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

impl SparseMLPWeights {
    fn as_borrowed(&self) -> sparse_mlp::Weights<'_> {
        sparse_mlp::Weights {
            gate_weight: &self.gate_weight,
            gate_scales: &self.gate_scales,
            gate_biases: &self.gate_biases,
            up_weight: &self.up_weight,
            up_scales: &self.up_scales,
            up_biases: &self.up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }
}

struct DenseMLPWeights {
    gate_up_weight: Buffer,
    gate_up_scales: Buffer,
    gate_up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

impl DenseMLPWeights {
    fn as_borrowed(&self) -> dense_mlp::Weights<'_> {
        dense_mlp::Weights {
            gate_up_weight: &self.gate_up_weight,
            gate_up_scales: &self.gate_up_scales,
            gate_up_biases: &self.gate_up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }
}

struct DenseMLPScratch {
    swiglu: Buffer,
}

impl DenseMLPScratch {
    fn as_borrowed(&self) -> dense_mlp::Scratch<'_> {
        dense_mlp::Scratch { swiglu: &self.swiglu }
    }
}

criterion_group!(benches, bench_moe);
criterion_main!(benches);
