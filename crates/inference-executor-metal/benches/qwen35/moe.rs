use std::fs::File;
use std::mem::size_of;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::components::dense_mlp;
use inference_backend_metal::components::moe::combine;
use inference_backend_metal::components::moe::expert_major;
use inference_backend_metal::components::moe::routing;
use inference_backend_metal::components::sparse_mlp;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::affine_quantized;
use inference_backend_metal::operators::softmax;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_metal::def::replay_op::MetalReplayRuntime;
use inference_executor_metal::def::replay_op::ReplayOp;
use inference_executor_metal::mlp::dense::scratch::DenseMLPScratchBindings;
use inference_executor_metal::mlp::moe::scratch::MoERoutingScratchBindings;
use inference_executor_metal::mlp::moe::scratch::MoEScratchBindings;
use inference_executor_metal::mlp::moe::scratch::SharedExpertsScratchBindings;
use inference_executor_metal::mlp::moe::scratch::TopKExpertsScratchBindings;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

const SHARD: &str = "model-00001-of-00004.safetensors";
const NUM_EXPERTS: u32 = 256;
const TOPK_EXPERTS: u32 = 8;
const HIDDEN_DIM: u32 = 2048;
const INTERMEDIATE_DIM: u32 = 512;
const GROUP_SIZE: u32 = 64;
const EXPERT_BITS: u32 = 4;
const ROUTER_BITS: u32 = 8;

fn main() {
    let args = Args::parse();

    let device = Device::system_default();
    let weights = RealMoEWeights::load(&device, &args.model_dir, args.model_layer_index);
    for num_tokens in args.num_tokens {
        if args.check_parity {
            let token_fixture = RealMoEFixture::new(&device, num_tokens, &weights, MoERealImpl::TokenMajor);
            let expert_fixture = RealMoEFixture::new(&device, num_tokens, &weights, MoERealImpl::ExpertMajor);
            token_fixture.run_replay();
            expert_fixture.run_replay();
            let token_bits = token_fixture.output_bits();
            let expert_bits = expert_fixture.output_bits();
            print_bitwise(
                args.model_layer_index,
                num_tokens,
                "token_major/expert_major",
                &token_bits,
                &expert_bits,
            );
        }

        for implementation in &args.implementations {
            let fixture = RealMoEFixture::new(&device, num_tokens, &weights, *implementation);
            let samples = fixture.measure(args.sparse_only, args.warmup_iters, args.iters, args.runs);
            if let Some(prefix) = &args.routing_output {
                fixture.write_routing(prefix);
            }
            print_perf(
                if args.sparse_only {
                    "sparse-mlp-real"
                } else {
                    "moe-real"
                },
                args.model_layer_index,
                implementation.key(),
                num_tokens,
                args.iters,
                &samples,
            );
        }
    }
}

struct Args {
    model_dir: PathBuf,
    model_layer_index: usize,
    num_tokens: Vec<u32>,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
    implementations: Vec<MoERealImpl>,
    check_parity: bool,
    sparse_only: bool,
    routing_output: Option<PathBuf>,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            model_layer_index: 0,
            num_tokens: vec![1, 2, 4, 8, 16, 32, 64],
            iters: 50,
            warmup_iters: 20,
            runs: 1,
            implementations: vec![MoERealImpl::TokenMajor, MoERealImpl::ExpertMajor],
            check_parity: false,
            sparse_only: false,
            routing_output: None,
        };
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut iter, &arg)),
                "--layer" => args.model_layer_index = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--tokens" => args.num_tokens = parse_u32_list(&next_arg(&mut iter, &arg), &arg),
                "--iters" => args.iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--impls" => args.implementations = parse_implementations(&next_arg(&mut iter, &arg)),
                "--check-parity" => args.check_parity = true,
                "--sparse-only" => args.sparse_only = true,
                "--routing-output" => args.routing_output = Some(PathBuf::from(next_arg(&mut iter, &arg))),
                "--bench" => {},
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                },
                _ => panic!("unknown argument {arg:?}; pass --help for usage"),
            }
        }
        assert!(!args.num_tokens.is_empty(), "--tokens must include at least one value");
        assert!(
            !args.implementations.is_empty(),
            "--impls must select token_major, expert_major, or both"
        );
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        assert!(
            args.routing_output.is_none()
                || (args.implementations == [MoERealImpl::ExpertMajor] && args.num_tokens.len() == 1),
            "--routing-output requires --impls expert_major and one --tokens value"
        );
        args
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MoERealImpl {
    TokenMajor,
    ExpertMajor,
}

impl MoERealImpl {
    fn key(self) -> &'static str {
        match self {
            Self::TokenMajor => "token_major",
            Self::ExpertMajor => "expert_major",
        }
    }
}

struct ForcedMoEKernels {
    router: affine_quantized::Matmul,
    router_softmax: softmax::Kernel,
    routing: routing::Compute,
    expert_major: expert_major::Compute,
    experts: sparse_mlp::Compute,
    shared_experts: dense_mlp::Compute,
    shared_expert_gate: affine_quantized::Matmul,
    combine: combine::Compute,
}

#[derive(Clone, Copy)]
struct ForcedMoEWeights<'a> {
    router_weight: &'a Buffer,
    router_scales: &'a Buffer,
    router_biases: &'a Buffer,
    experts: sparse_mlp::Weights<'a>,
    shared_experts: dense_mlp::Weights<'a>,
    shared_expert_gate_weight: &'a Buffer,
    shared_expert_gate_scales: &'a Buffer,
    shared_expert_gate_biases: &'a Buffer,
}

impl ForcedMoEKernels {
    fn new(device: &Device) -> Self {
        Self {
            router: affine_quantized::Matmul::new(device, affine_config(NUM_EXPERTS, HIDDEN_DIM, ROUTER_BITS)),
            router_softmax: softmax::Kernel::new(
                device,
                softmax::Config {
                    num_values_per_row: NUM_EXPERTS,
                    dtype: Dtype::Bfloat16,
                },
            ),
            routing: routing::Compute::new(device, routing_config()),
            expert_major: expert_major::Compute::new(device, expert_major_config()),
            experts: sparse_mlp::Compute::new(device, sparse_config()),
            shared_experts: dense_mlp::Compute::new(device, dense_config()),
            shared_expert_gate: affine_quantized::Matmul::new(device, affine_config(1, HIDDEN_DIM, ROUTER_BITS)),
            combine: combine::Compute::new(device, combine::Config::bf16(TOPK_EXPERTS, HIDDEN_DIM)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record<'a, R>(
        &'a self,
        recorder: &mut R,
        implementation: MoERealImpl,
        num_tokens: u32,
        input: &'a Buffer,
        output: &'a Buffer,
        scratch: MoEScratchBindings<'a>,
        shared_scratch: SharedExpertsScratchBindings<'a>,
        weights: ForcedMoEWeights<'a>,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let routing_shape = routing_shape(num_tokens);
        let num_active_tokens = ReplayU32::Fixed(num_tokens);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.router.invoke(
            num_tokens,
            num_active_tokens,
            scratch.routing.router_logits,
            0,
            input,
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
                num_total_rows: num_tokens,
            },
            num_active_tokens,
            softmax::Buffers {
                input: scratch.routing.router_logits,
                output: scratch.routing.router_probs,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.routing.invoke(
            routing_shape,
            num_active_tokens,
            routing::Buffers {
                router_probs: scratch.routing.router_probs,
                expert_indices: scratch.routing.expert_indices,
                expert_probs: scratch.routing.expert_probs,
            },
        )));

        match implementation {
            MoERealImpl::TokenMajor => {
                recorder.record_with_barrier_before(ReplayOp::opaque(self.experts.invoke_token_major(
                    sparse_token_major_shape(num_tokens),
                    TOPK_EXPERTS,
                    num_active_tokens,
                    sparse_mlp::TokenMajorBuffers {
                        input,
                        token_indices: scratch.topk_experts.token_indices,
                        expert_indices: scratch.routing.expert_indices,
                        route_indices: scratch.topk_experts.route_indices,
                        routed_hidden: scratch.topk_experts.routed_hidden,
                    },
                    sparse_mlp::Scratch {
                        swiglu: scratch.topk_experts.sparse_swiglu,
                    },
                    weights.experts,
                )));
                self.record_shared(recorder, num_tokens, input, shared_scratch, weights);
                recorder.record_with_barrier_before(ReplayOp::opaque(self.combine.invoke_with_shared_experts(
                    combine::Shape {
                        num_total_tokens: num_tokens,
                    },
                    num_active_tokens,
                    combine::WithSharedExpertsBuffers {
                        routed_hidden: scratch.topk_experts.routed_hidden,
                        routed_probs: scratch.routing.expert_probs,
                        shared_hidden: shared_scratch.hidden,
                        shared_expert_gate_logits: shared_scratch.gate_logits,
                        output,
                    },
                )));
            },
            MoERealImpl::ExpertMajor => {
                let shape = expert_major::Shape {
                    num_total_tokens: num_tokens,
                };
                self.record_shared(recorder, num_tokens, input, shared_scratch, weights);
                recorder.record(ReplayOp::opaque(self.expert_major.invoke_layout(
                    shape,
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
                )));
                recorder.record_with_barrier_before(ReplayOp::opaque(self.expert_major.invoke_pack_input(
                    shape,
                    num_active_tokens,
                    expert_major::PackInputBuffers {
                        input,
                        routes_by_expert: scratch.topk_experts.routes_by_expert,
                        packed_input: scratch.topk_experts.packed_input,
                    },
                )));
                recorder.record_with_barrier_before(ReplayOp::opaque(
                    self.experts.invoke_expert_major(
                        sparse_mlp::ExpertMajorShape {
                            num_total_routes: num_tokens
                                .checked_mul(TOPK_EXPERTS)
                                .expect("forced MoE route count must fit u32"),
                            num_total_tokens: num_tokens,
                            num_experts_per_token: TOPK_EXPERTS,
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
                        weights.experts,
                    ),
                ));
                recorder.record_with_barrier_before(ReplayOp::opaque(
                    self.expert_major.invoke_scatter_with_shared_experts(
                        shape,
                        num_active_tokens,
                        expert_major::ScatterWithSharedExpertsBuffers {
                            packed_output: scratch.topk_experts.routed_hidden,
                            routes_by_token: scratch.topk_experts.routes_by_token,
                            routed_probs: scratch.routing.expert_probs,
                            shared_hidden: shared_scratch.hidden,
                            shared_expert_gate_logits: shared_scratch.gate_logits,
                            output,
                        },
                    ),
                ));
            },
        }
    }

    fn record_shared<'a, R>(
        &'a self,
        recorder: &mut R,
        num_tokens: u32,
        input: &'a Buffer,
        scratch: SharedExpertsScratchBindings<'a>,
        weights: ForcedMoEWeights<'a>,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        recorder.record(ReplayOp::opaque(self.shared_experts.invoke(
            dense_mlp::Shape {
                num_total_tokens: num_tokens,
            },
            ReplayU32::Fixed(num_tokens),
            dense_mlp::Buffers {
                hidden_state: input,
                next_hidden_state: scratch.hidden,
            },
            dense_mlp::Scratch {
                swiglu: scratch.dense_mlp.swiglu,
            },
            weights.shared_experts,
        )));
        recorder.record(ReplayOp::opaque(self.shared_expert_gate.invoke(
            num_tokens,
            ReplayU32::Fixed(num_tokens),
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
}

struct RealMoEFixture<'a> {
    stream: Stream,
    output: Buffer,
    replay: ReplayProgram,
    sparse_replay: ReplayProgram,
    _input: Buffer,
    _router_logits: Buffer,
    _router_probs: Buffer,
    _expert_indices: Buffer,
    _expert_probs: Buffer,
    _token_indices: Buffer,
    _route_indices: Buffer,
    _expert_counts: Buffer,
    _expert_offsets: Buffer,
    _expert_cursors: Buffer,
    _routes_by_expert: Buffer,
    _routes_by_token: Buffer,
    _experts_by_route: Buffer,
    _packed_input: Buffer,
    _routed_hidden: Buffer,
    _sparse_swiglu: Buffer,
    _shared_hidden: Buffer,
    _shared_expert_gate_logits: Buffer,
    _shared_scratch: DenseMLPScratch,
    _weights: &'a RealMoEWeights,
}

impl<'a> RealMoEFixture<'a> {
    fn new(device: &Device, num_tokens: u32, weights: &'a RealMoEWeights, implementation: MoERealImpl) -> Self {
        let stream = Stream::new(device);
        let router_config = affine_config(NUM_EXPERTS, HIDDEN_DIM, ROUTER_BITS);
        let shared_expert_gate_config = affine_config(1, HIDDEN_DIM, ROUTER_BITS);
        let num_tokens_i32 = num_tokens.try_into().expect("MoE token count must fit i32");
        let routing_shape = routing_shape(num_tokens);
        let sparse_config = sparse_config();
        let dense_config = dense_config();
        let sparse_shape = sparse_token_major_shape(num_tokens);
        let expert_major_config = expert_major_config();
        let expert_major_shape = expert_major::Shape {
            num_total_tokens: num_tokens,
        };
        let expert_major_sparse_shape = sparse_mlp::TokenMajorShape {
            num_total_routes: expert_major_config.num_routes(expert_major_shape),
            num_total_tokens: expert_major_config.num_routes(expert_major_shape),
        };
        let selected_sparse_shape = match implementation {
            MoERealImpl::TokenMajor => sparse_shape,
            MoERealImpl::ExpertMajor => expert_major_sparse_shape,
        };
        let dense_shape = dense_mlp::Shape {
            num_total_tokens: num_tokens,
        };
        let combine_config = combine::Config::bf16(TOPK_EXPERTS, HIDDEN_DIM);
        let combine_shape = combine::Shape {
            num_total_tokens: num_tokens,
        };
        let num_routes = num_tokens as usize * TOPK_EXPERTS as usize;
        let input = Buffer::from_slice(device, &hidden_fixture(num_tokens as usize, HIDDEN_DIM as usize));
        let router_logits = Buffer::new_zeroed(device, router_config.output_bytes(num_tokens_i32));
        let router_probs = Buffer::new_zeroed(device, router_config.output_bytes(num_tokens_i32));
        let routing_config = routing_config();
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
        let routed_hidden = Buffer::new_zeroed(device, sparse_config.token_major_output_bytes(selected_sparse_shape));
        let sparse_swiglu = Buffer::new_zeroed(
            device,
            sparse_config.swiglu_bytes(selected_sparse_shape.num_total_routes),
        );
        let shared_hidden = Buffer::new_zeroed(device, dense_config.down_config().output_bytes(num_tokens_i32));
        let shared_expert_gate_logits =
            Buffer::new_zeroed(device, shared_expert_gate_config.output_bytes(num_tokens_i32));
        let output = Buffer::new_zeroed(device, combine_config.output_bytes(combine_shape));
        let shared_scratch = DenseMLPScratch {
            swiglu: Buffer::new_zeroed(device, dense_config.swiglu_bytes(dense_shape)),
        };
        let kernels = ForcedMoEKernels::new(device);
        let mut recorder = MetalReplayRuntime::new(&stream).create_recorder();
        kernels.record(
            &mut recorder,
            implementation,
            num_tokens,
            &input,
            &output,
            MoEScratchBindings {
                routing: MoERoutingScratchBindings {
                    router_logits: &router_logits,
                    router_probs: &router_probs,
                    expert_indices: &expert_indices,
                    expert_probs: &expert_probs,
                },
                topk_experts: TopKExpertsScratchBindings {
                    token_indices: &token_indices,
                    route_indices: &route_indices,
                    routed_hidden: &routed_hidden,
                    sparse_swiglu: &sparse_swiglu,
                    expert_counts: &expert_counts,
                    expert_offsets: &expert_offsets,
                    expert_cursors: &expert_cursors,
                    routes_by_expert: &routes_by_expert,
                    routes_by_token: &routes_by_token,
                    experts_by_route: &experts_by_route,
                    packed_input: &packed_input,
                },
            },
            shared_scratch.as_shared_scratch(&shared_hidden, &shared_expert_gate_logits),
            ForcedMoEWeights {
                router_weight: &weights.router_weight,
                router_scales: &weights.router_scales,
                router_biases: &weights.router_biases,
                experts: weights.sparse.as_borrowed(),
                shared_experts: weights.shared.as_borrowed(),
                shared_expert_gate_weight: &weights.shared_expert_gate_weight,
                shared_expert_gate_scales: &weights.shared_expert_gate_scales,
                shared_expert_gate_biases: &weights.shared_expert_gate_biases,
            },
        );
        let replay = recorder.build();
        let mut recorder = MetalReplayRuntime::new(&stream).create_recorder();
        let scratch = sparse_mlp::Scratch { swiglu: &sparse_swiglu };
        let num_active_tokens = ReplayU32::Fixed(num_tokens);
        match implementation {
            MoERealImpl::TokenMajor => {
                recorder.record(ReplayOp::opaque(kernels.experts.invoke_token_major(
                    sparse_shape,
                    TOPK_EXPERTS,
                    num_active_tokens,
                    sparse_mlp::TokenMajorBuffers {
                        input: &input,
                        token_indices: &token_indices,
                        expert_indices: &expert_indices,
                        route_indices: &route_indices,
                        routed_hidden: &routed_hidden,
                    },
                    scratch,
                    weights.sparse.as_borrowed(),
                )))
            },
            MoERealImpl::ExpertMajor => {
                recorder.record(ReplayOp::opaque(kernels.experts.invoke_expert_major(
                    sparse_mlp::ExpertMajorShape {
                        num_total_routes: expert_major_config.num_routes(expert_major_shape),
                        num_total_tokens: num_tokens,
                        num_experts_per_token: TOPK_EXPERTS,
                    },
                    num_active_tokens,
                    sparse_mlp::ExpertMajorBuffers {
                        packed_input: &packed_input,
                        experts_by_route: &experts_by_route,
                        expert_offsets: &expert_offsets,
                        packed_output: &routed_hidden,
                    },
                    scratch,
                    weights.sparse.as_borrowed(),
                )))
            },
        }
        let sparse_replay = recorder.build();
        Self {
            stream,
            output,
            replay,
            sparse_replay,
            _input: input,
            _router_logits: router_logits,
            _router_probs: router_probs,
            _expert_indices: expert_indices,
            _expert_probs: expert_probs,
            _token_indices: token_indices,
            _route_indices: route_indices,
            _expert_counts: expert_counts,
            _expert_offsets: expert_offsets,
            _expert_cursors: expert_cursors,
            _routes_by_expert: routes_by_expert,
            _routes_by_token: routes_by_token,
            _experts_by_route: experts_by_route,
            _packed_input: packed_input,
            _routed_hidden: routed_hidden,
            _sparse_swiglu: sparse_swiglu,
            _shared_hidden: shared_hidden,
            _shared_expert_gate_logits: shared_expert_gate_logits,
            _shared_scratch: shared_scratch,
            _weights: weights,
        }
    }

    fn run_replay(&self) {
        MetalReplayRuntime::new(&self.stream).submit_replay(&self.replay).wait();
    }

    fn measure(&self, sparse_only: bool, warmup_iters: usize, iters: usize, runs: usize) -> Vec<f64> {
        let replay = if sparse_only {
            self.run_replay(); // Prepare routing and packed input outside sparse-expert timing.
            &self.sparse_replay
        } else {
            &self.replay
        };
        measure_runs(runs, warmup_iters, iters, || {
            MetalReplayRuntime::new(&self.stream).submit_replay(replay).wait();
        })
    }

    fn write_routing(&self, prefix: &Path) {
        for (suffix, buffer) in [("indices", &self._expert_indices), ("routes", &self._routes_by_expert)] {
            let values = buffer.read_typed::<u32>(0, buffer.len_bytes() / size_of::<u32>());
            let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
            std::fs::write(prefix.with_added_extension(suffix), bytes).expect("write benchmark route witness");
        }
    }

    fn output_bits(&self) -> Vec<u16> {
        self.output
            .read_typed::<u16>(0, self.output.len_bytes() / size_of::<u16>())
    }
}

struct RealMoEWeights {
    router_weight: Buffer,
    router_scales: Buffer,
    router_biases: Buffer,
    shared_expert_gate_weight: Buffer,
    shared_expert_gate_scales: Buffer,
    shared_expert_gate_biases: Buffer,
    sparse: SparseMLPWeights,
    shared: DenseMLPWeights,
}

impl RealMoEWeights {
    fn load(device: &Device, model_dir: &Path, model_layer_index: usize) -> Self {
        let shard_path = model_dir.join(SHARD);
        let mapped = MappedFile::open(&shard_path);
        let tensors = SafeTensors::deserialize(mapped.as_bytes()).unwrap_or_else(|err| {
            panic!(
                "unable to deserialize safetensors shard {}: {err:?}",
                shard_path.display()
            )
        });
        let router_weight = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "gate.weight"),
            safetensors::Dtype::U32,
        );
        let router_scales = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "gate.scales"),
            safetensors::Dtype::BF16,
        );
        let router_biases = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "gate.biases"),
            safetensors::Dtype::BF16,
        );
        let shared_expert_gate_weight = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert_gate.weight"),
            safetensors::Dtype::U32,
        );
        let shared_expert_gate_scales = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert_gate.scales"),
            safetensors::Dtype::BF16,
        );
        let shared_expert_gate_biases = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert_gate.biases"),
            safetensors::Dtype::BF16,
        );
        let sparse = SparseMLPWeights {
            gate_weight: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.gate_proj.weight"),
                    safetensors::Dtype::U32,
                ),
            ),
            gate_scales: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.gate_proj.scales"),
                    safetensors::Dtype::BF16,
                ),
            ),
            gate_biases: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.gate_proj.biases"),
                    safetensors::Dtype::BF16,
                ),
            ),
            up_weight: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.up_proj.weight"),
                    safetensors::Dtype::U32,
                ),
            ),
            up_scales: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.up_proj.scales"),
                    safetensors::Dtype::BF16,
                ),
            ),
            up_biases: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.up_proj.biases"),
                    safetensors::Dtype::BF16,
                ),
            ),
            down_weight: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.down_proj.weight"),
                    safetensors::Dtype::U32,
                ),
            ),
            down_scales: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.down_proj.scales"),
                    safetensors::Dtype::BF16,
                ),
            ),
            down_biases: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "switch_mlp.down_proj.biases"),
                    safetensors::Dtype::BF16,
                ),
            ),
        };
        let shared_expert_gate_weight_dense = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert.gate_proj.weight"),
            safetensors::Dtype::U32,
        );
        let shared_experts_up_weight = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert.up_proj.weight"),
            safetensors::Dtype::U32,
        );
        let shared_expert_gate_scales_dense = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert.gate_proj.scales"),
            safetensors::Dtype::BF16,
        );
        let shared_experts_up_scales = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert.up_proj.scales"),
            safetensors::Dtype::BF16,
        );
        let shared_expert_gate_biases_dense = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert.gate_proj.biases"),
            safetensors::Dtype::BF16,
        );
        let shared_experts_up_biases = tensor_bytes(
            &tensors,
            &tensor_name(model_layer_index, "shared_expert.up_proj.biases"),
            safetensors::Dtype::BF16,
        );
        let shared = DenseMLPWeights {
            gate_up_weight: Buffer::from_slice(
                device,
                &concat_bytes(&shared_expert_gate_weight_dense, &shared_experts_up_weight),
            ),
            gate_up_scales: Buffer::from_slice(
                device,
                &concat_bytes(&shared_expert_gate_scales_dense, &shared_experts_up_scales),
            ),
            gate_up_biases: Buffer::from_slice(
                device,
                &concat_bytes(&shared_expert_gate_biases_dense, &shared_experts_up_biases),
            ),
            down_weight: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "shared_expert.down_proj.weight"),
                    safetensors::Dtype::U32,
                ),
            ),
            down_scales: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "shared_expert.down_proj.scales"),
                    safetensors::Dtype::BF16,
                ),
            ),
            down_biases: Buffer::from_slice(
                device,
                &tensor_bytes(
                    &tensors,
                    &tensor_name(model_layer_index, "shared_expert.down_proj.biases"),
                    safetensors::Dtype::BF16,
                ),
            ),
        };
        validate_weight_sizes(
            &router_weight,
            &router_scales,
            &router_biases,
            &shared_expert_gate_weight,
            &shared_expert_gate_scales,
            &shared_expert_gate_biases,
            &sparse,
            &shared,
        );
        Self {
            router_weight: Buffer::from_slice(device, &router_weight),
            router_scales: Buffer::from_slice(device, &router_scales),
            router_biases: Buffer::from_slice(device, &router_biases),
            shared_expert_gate_weight: Buffer::from_slice(device, &shared_expert_gate_weight),
            shared_expert_gate_scales: Buffer::from_slice(device, &shared_expert_gate_scales),
            shared_expert_gate_biases: Buffer::from_slice(device, &shared_expert_gate_biases),
            sparse,
            shared,
        }
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

    fn as_shared_scratch<'a>(
        &'a self,
        shared_hidden: &'a Buffer,
        shared_expert_gate_logits: &'a Buffer,
    ) -> SharedExpertsScratchBindings<'a> {
        SharedExpertsScratchBindings {
            hidden: shared_hidden,
            gate_logits: shared_expert_gate_logits,
            dense_mlp: DenseMLPScratchBindings { swiglu: &self.swiglu },
        }
    }
}

struct MappedFile {
    ptr: *mut libc::c_void,
    len: usize,
}

impl MappedFile {
    fn open(path: &Path) -> Self {
        let file = File::open(path).unwrap_or_else(|err| panic!("unable to open {}: {err}", path.display()));
        let len = file
            .metadata()
            .unwrap_or_else(|err| panic!("unable to stat {}: {err}", path.display()))
            .len() as usize;
        assert!(len > 0, "safetensors shard must not be empty");
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            panic!("unable to mmap {}: {}", path.display(), std::io::Error::last_os_error());
        }
        unsafe {
            let _ = libc::madvise(ptr, len, libc::MADV_RANDOM);
        }
        Self { ptr, len }
    }

    fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr, self.len);
        }
    }
}

fn tensor_name(model_layer_index: usize, suffix: &str) -> String {
    format!("language_model.model.layers.{model_layer_index}.mlp.{suffix}")
}

fn tensor_bytes(tensors: &SafeTensors<'_>, name: &str, dtype: safetensors::Dtype) -> Vec<u8> {
    let view = tensors
        .tensor(name)
        .unwrap_or_else(|err| panic!("missing safetensor {name}: {err:?}"));
    assert_eq!(view.dtype(), dtype, "unexpected dtype for tensor {name}");
    validate_tensor_shape(name, &view);
    view.data().to_vec()
}

fn validate_tensor_shape(name: &str, view: &TensorView<'_>) {
    let shape = view.shape();
    if name.ends_with("mlp.gate.weight") {
        assert_eq!(shape, &[NUM_EXPERTS as usize, packed_k_words(HIDDEN_DIM, ROUTER_BITS)]);
    } else if name.ends_with("mlp.gate.scales") || name.ends_with("mlp.gate.biases") {
        assert_eq!(shape, &[NUM_EXPERTS as usize, groups(HIDDEN_DIM)]);
    } else if name.ends_with("shared_expert_gate.weight") {
        assert_eq!(shape, &[1, packed_k_words(HIDDEN_DIM, ROUTER_BITS)]);
    } else if name.ends_with("shared_expert_gate.scales") || name.ends_with("shared_expert_gate.biases") {
        assert_eq!(shape, &[1, groups(HIDDEN_DIM)]);
    } else if name.ends_with("switch_mlp.gate_proj.weight") || name.ends_with("switch_mlp.up_proj.weight") {
        assert_eq!(
            shape,
            &[
                NUM_EXPERTS as usize,
                INTERMEDIATE_DIM as usize,
                packed_k_words(HIDDEN_DIM, EXPERT_BITS)
            ]
        );
    } else if name.ends_with("switch_mlp.gate_proj.scales")
        || name.ends_with("switch_mlp.gate_proj.biases")
        || name.ends_with("switch_mlp.up_proj.scales")
        || name.ends_with("switch_mlp.up_proj.biases")
    {
        assert_eq!(
            shape,
            &[NUM_EXPERTS as usize, INTERMEDIATE_DIM as usize, groups(HIDDEN_DIM)]
        );
    } else if name.ends_with("switch_mlp.down_proj.weight") {
        assert_eq!(
            shape,
            &[
                NUM_EXPERTS as usize,
                HIDDEN_DIM as usize,
                packed_k_words(INTERMEDIATE_DIM, EXPERT_BITS)
            ]
        );
    } else if name.ends_with("switch_mlp.down_proj.scales") || name.ends_with("switch_mlp.down_proj.biases") {
        assert_eq!(
            shape,
            &[NUM_EXPERTS as usize, HIDDEN_DIM as usize, groups(INTERMEDIATE_DIM)]
        );
    } else if name.ends_with("shared_expert.gate_proj.weight") || name.ends_with("shared_expert.up_proj.weight") {
        assert_eq!(
            shape,
            &[INTERMEDIATE_DIM as usize, packed_k_words(HIDDEN_DIM, EXPERT_BITS)]
        );
    } else if name.ends_with("shared_expert.gate_proj.scales")
        || name.ends_with("shared_expert.gate_proj.biases")
        || name.ends_with("shared_expert.up_proj.scales")
        || name.ends_with("shared_expert.up_proj.biases")
    {
        assert_eq!(shape, &[INTERMEDIATE_DIM as usize, groups(HIDDEN_DIM)]);
    } else if name.ends_with("shared_expert.down_proj.weight") {
        assert_eq!(
            shape,
            &[HIDDEN_DIM as usize, packed_k_words(INTERMEDIATE_DIM, EXPERT_BITS)]
        );
    } else if name.ends_with("shared_expert.down_proj.scales") || name.ends_with("shared_expert.down_proj.biases") {
        assert_eq!(shape, &[HIDDEN_DIM as usize, groups(INTERMEDIATE_DIM)]);
    } else {
        panic!("unexpected MoE tensor name {name}");
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_weight_sizes(
    router_weight: &[u8],
    router_scales: &[u8],
    router_biases: &[u8],
    shared_expert_gate_weight: &[u8],
    shared_expert_gate_scales: &[u8],
    shared_expert_gate_biases: &[u8],
    sparse: &SparseMLPWeights,
    shared: &DenseMLPWeights,
) {
    let router_config = affine_config(NUM_EXPERTS, HIDDEN_DIM, ROUTER_BITS);
    let shared_expert_gate_config = affine_config(1, HIDDEN_DIM, ROUTER_BITS);
    assert_eq!(router_weight.len(), router_config.weight_bytes());
    assert_eq!(router_scales.len(), router_config.scale_or_bias_bytes());
    assert_eq!(router_biases.len(), router_config.scale_or_bias_bytes());
    assert_eq!(
        shared_expert_gate_weight.len(),
        shared_expert_gate_config.weight_bytes()
    );
    assert_eq!(
        shared_expert_gate_scales.len(),
        shared_expert_gate_config.scale_or_bias_bytes()
    );
    assert_eq!(
        shared_expert_gate_biases.len(),
        shared_expert_gate_config.scale_or_bias_bytes()
    );
    let sparse_config = sparse_config();
    let sparse_shape = sparse_mlp::TokenMajorShape {
        num_total_routes: TOPK_EXPERTS,
        num_total_tokens: 1,
    };
    let gate_up_config = sparse_config.gate_up_config();
    let down_config = sparse_config.down_config();
    assert_eq!(
        sparse.gate_weight.len_bytes(),
        NUM_EXPERTS as usize * gate_up_config.weight_bytes_per_expert()
    );
    assert_eq!(
        sparse.gate_scales.len_bytes(),
        NUM_EXPERTS as usize * gate_up_config.affine_param_bytes_per_expert()
    );
    assert_eq!(
        sparse.gate_biases.len_bytes(),
        NUM_EXPERTS as usize * gate_up_config.affine_param_bytes_per_expert()
    );
    assert_eq!(sparse.up_weight.len_bytes(), sparse.gate_weight.len_bytes());
    assert_eq!(sparse.up_scales.len_bytes(), sparse.gate_scales.len_bytes());
    assert_eq!(sparse.up_biases.len_bytes(), sparse.gate_biases.len_bytes());
    assert_eq!(
        sparse.down_weight.len_bytes(),
        NUM_EXPERTS as usize * down_config.weight_bytes_per_expert()
    );
    assert_eq!(
        sparse.down_scales.len_bytes(),
        NUM_EXPERTS as usize * down_config.affine_param_bytes_per_expert()
    );
    assert_eq!(
        sparse.down_biases.len_bytes(),
        NUM_EXPERTS as usize * down_config.affine_param_bytes_per_expert()
    );
    let dense_config = dense_config();
    let dense_gate_up_config = dense_config.gate_up_config();
    let dense_down_config = dense_config.down_config();
    assert_eq!(shared.gate_up_weight.len_bytes(), dense_gate_up_config.weight_bytes());
    assert_eq!(
        shared.gate_up_scales.len_bytes(),
        dense_gate_up_config.scale_or_bias_bytes()
    );
    assert_eq!(
        shared.gate_up_biases.len_bytes(),
        dense_gate_up_config.scale_or_bias_bytes()
    );
    assert_eq!(shared.down_weight.len_bytes(), dense_down_config.weight_bytes());
    assert_eq!(shared.down_scales.len_bytes(), dense_down_config.scale_or_bias_bytes());
    assert_eq!(shared.down_biases.len_bytes(), dense_down_config.scale_or_bias_bytes());
}

fn sparse_config() -> sparse_mlp::Config {
    sparse_mlp::Config {
        num_experts: NUM_EXPERTS,
        hidden_dim: HIDDEN_DIM,
        intermediate_dim: INTERMEDIATE_DIM,
        group_size: GROUP_SIZE,
        bits: EXPERT_BITS,
        dtype: Dtype::Bfloat16,
    }
}

fn dense_config() -> dense_mlp::Config {
    dense_mlp::Config {
        hidden_dim: HIDDEN_DIM,
        intermediate_dim: INTERMEDIATE_DIM,
        gate_up_group_size: GROUP_SIZE,
        gate_up_bits: EXPERT_BITS,
        gate_up_scale_bias_dtype: Dtype::Bfloat16,
        down_group_size: GROUP_SIZE,
        down_bits: EXPERT_BITS,
        down_scale_bias_dtype: Dtype::Bfloat16,
        dtype: Dtype::Bfloat16,
    }
}

fn affine_config(n: u32, k: u32, bits: u32) -> affine_quantized::Config {
    affine_quantized::Config {
        n: n.try_into().expect("MoE affine output dimension must fit i32"),
        k: k.try_into().expect("MoE affine input dimension must fit i32"),
        group_size: GROUP_SIZE.try_into().expect("MoE affine group size must fit i32"),
        bits: bits.try_into().expect("MoE affine bits must fit i32"),
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
    }
}

fn routing_shape(num_tokens: u32) -> routing::Shape {
    routing::Shape {
        num_total_tokens: num_tokens,
    }
}

fn routing_config() -> routing::Config {
    routing::Config {
        num_experts: NUM_EXPERTS,
        num_experts_per_token: TOPK_EXPERTS,
        norm_topk_prob: true,
    }
}

fn expert_major_config() -> expert_major::Config {
    expert_major::Config::bf16(NUM_EXPERTS, TOPK_EXPERTS, HIDDEN_DIM)
}

fn sparse_token_major_shape(num_tokens: u32) -> sparse_mlp::TokenMajorShape {
    sparse_mlp::TokenMajorShape {
        num_total_routes: num_tokens
            .checked_mul(TOPK_EXPERTS)
            .expect("forced MoE route count must fit u32"),
        num_total_tokens: num_tokens,
    }
}

fn concat_bytes(left: &[u8], right: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    out.extend_from_slice(left);
    out.extend_from_slice(right);
    out
}

fn hidden_fixture(num_tokens: usize, hidden_dim: usize) -> Vec<u16> {
    (0..num_tokens * hidden_dim)
        .map(|index| bf16::from_f32(((index % 23) as f32 - 11.0) * 0.03125).to_bits())
        .collect()
}

fn token_route_indices(num_tokens: usize, topk_experts: usize) -> Vec<u32> {
    (0..num_tokens * topk_experts)
        .map(|route| u32::try_from(route / topk_experts).expect("token route index must fit u32"))
        .collect()
}

fn identity_indices(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| u32::try_from(index).expect("identity index must fit u32"))
        .collect()
}

fn packed_k_words(k: u32, bits: u32) -> usize {
    (k as usize * bits as usize) / 32
}

fn groups(k: u32) -> usize {
    (k / GROUP_SIZE) as usize
}

fn next_arg(iter: &mut impl Iterator<Item = String>, flag: &str) -> String {
    iter.next()
        .unwrap_or_else(|| panic!("{flag} requires a value; pass --help for usage"))
}

fn parse_u32_list(value: &str, flag: &str) -> Vec<u32> {
    parse_list(value)
        .into_iter()
        .map(|part| {
            part.parse::<u32>()
                .unwrap_or_else(|err| panic!("invalid {flag} entry {part:?}: {err}"))
        })
        .collect()
}

fn parse_usize(value: &str, flag: &str) -> usize {
    value
        .parse()
        .unwrap_or_else(|err| panic!("invalid {flag} value {value:?}: {err}"))
}

fn parse_implementations(value: &str) -> Vec<MoERealImpl> {
    let mut impls = Vec::new();
    for part in value
        .split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
    {
        match part {
            "both" => {
                impls.push(MoERealImpl::TokenMajor);
                impls.push(MoERealImpl::ExpertMajor);
            },
            "token_major" => impls.push(MoERealImpl::TokenMajor),
            "expert_major" => impls.push(MoERealImpl::ExpertMajor),
            other => panic!("unknown --impls entry {other:?}; use token_major, expert_major, or both"),
        }
    }
    impls.dedup();
    assert!(
        !impls.is_empty(),
        "--impls must select token_major, expert_major, or both"
    );
    impls
}

fn parse_list(value: &str) -> Vec<&str> {
    let values = value
        .split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    assert!(!values.is_empty(), "list argument must contain at least one value");
    values
}

fn print_help() {
    println!(
        "\
moe real-weight replay bench

Options:
  --model-dir PATH
  --layer N
  --tokens 1,2,4,8,16,32,64
  --impls token_major,expert_major,both
  --sparse-only           time only expert compute; prepare routing/layout before measurement
  --routing-output <path> write route IDs and packed order for cross-engine comparisons
  --check-parity
  --iters N
  --warmup-iters N
  --runs N
"
    );
}

fn print_bitwise(model_layer_index: usize, num_tokens: u32, implementation: &str, baseline: &[u16], actual: &[u16]) {
    let first_mismatch = baseline
        .iter()
        .zip(actual.iter())
        .position(|(left, right)| left != right)
        .unwrap_or(0);
    let (lhs, rhs) = baseline
        .get(first_mismatch)
        .zip(actual.get(first_mismatch))
        .map(|(left, right)| (*left, *right))
        .unwrap_or((0, 0));
    println!(
        "bitwise component=moe-real impl={implementation} layer={model_layer_index} num_tokens={num_tokens} \
         num_values={} equal={} first_mismatch={first_mismatch} lhs=0x{lhs:04x} rhs=0x{rhs:04x}",
        baseline.len(),
        baseline == actual,
    );
}

fn measure_runs(runs: usize, warmup_iters: usize, iters: usize, mut run: impl FnMut()) -> Vec<f64> {
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        for _ in 0..warmup_iters {
            run();
        }
        let mut duration = Duration::ZERO;
        for _ in 0..iters {
            let start = Instant::now();
            run();
            duration += start.elapsed();
        }
        samples.push(duration.as_secs_f64() * 1_000_000.0 / iters as f64);
    }
    samples
}

fn print_perf(
    component: &str,
    model_layer_index: usize,
    implementation: &str,
    num_tokens: u32,
    iters: usize,
    samples: &[f64],
) {
    let median_us = median(samples);
    let sample_text = samples
        .iter()
        .map(|sample| format!("{sample:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "perf component={component} impl={implementation} layer={model_layer_index} num_tokens={num_tokens} \
         iters={iters} runs={} median_us={median_us:.3} samples_us=[{sample_text}]",
        samples.len()
    );
}

fn median(samples: &[f64]) -> f64 {
    assert!(!samples.is_empty());
    let mut sorted = samples.to_vec();
    sorted.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}
