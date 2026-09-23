use std::mem::size_of;

use half::bf16;
use inference_backend_metal::components::dense_mlp;
use inference_backend_metal::components::sparse_mlp;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_executor_core::mlp::dense::reference::QuantizedAffineReferenceShape;
use inference_executor_core::mlp::dense::reference::QuantizedDenseMLPReferenceGeometry;
use inference_executor_core::mlp::dense::reference::QuantizedDenseMLPReferenceWeights;
use inference_executor_core::mlp::dense::reference::quantized_affine_reference;
use inference_executor_core::mlp::dense::reference::quantized_dense_mlp_reference;
use inference_executor_core::mlp::moe::reference::QuantizedSparseMLPReferenceInput;
use inference_executor_core::mlp::moe::reference::QuantizedSparseMLPReferenceWeights;
use inference_executor_core::mlp::moe::reference::moe_combine_with_shared_experts_bf16_reference;
use inference_executor_core::mlp::moe::reference::moe_combine_without_shared_experts_bf16_reference;
use inference_executor_core::mlp::moe::reference::moe_routing_from_bf16_probs_reference;
use inference_executor_core::mlp::moe::reference::quantized_sparse_mlp_reference;

use super::*;
use crate::def::replay_op::MetalReplayRuntime;
use crate::def::replay_op::ReplayRecorder;
use crate::mlp::moe::scratch::MoEScratch;
use crate::replay::Replay;
use crate::replay::ReplayComponent;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.gated_moe.num_active_tokens");

#[test]
fn test_selector_returns_registered_variant_at_crossover() {
    let device = Device::system_default();
    let (core, _) = routing_test_config(true);
    let registry = Registry::new(&device, &core);
    assert_eq!(
        Selector::select(&registry, GatedMoEReplayShape { num_tokens: 1 }).0,
        VariantKey::TokenMajor
    );
    assert_eq!(
        Selector::select(&registry, GatedMoEReplayShape { num_tokens: 4 }).0,
        VariantKey::TokenMajor
    );
    assert_eq!(
        Selector::select(&registry, GatedMoEReplayShape { num_tokens: 5 }).0,
        VariantKey::ExpertMajor
    );
}

#[test]
fn test_routing_replay_matches_reference_across_active_counts() {
    assert_routing_replay_matches_reference(8, 3);
    assert_routing_replay_matches_reference(512, 10);
}

fn assert_routing_replay_matches_reference(num_experts: usize, num_experts_per_token: usize) {
    let device = Device::system_default();
    let stream = inference_backend_metal::metal::Stream::new(&device);
    let (mut core, metal) = routing_test_config(true);
    core.num_experts = num_experts;
    core.num_experts_per_token = num_experts_per_token;
    let moe = GatedMoE::new(&device, core.clone(), metal);
    let router_config = affine_config_with_bits(core.num_experts, core.hidden_dim, metal.router_bits, metal);
    let num_total_tokens = 8_u32;
    let hidden_dim = core.hidden_dim;
    let num_experts = core.num_experts;
    let topk = core.num_experts_per_token;
    let all_hidden = bf16_values(
        &(0..num_total_tokens as usize * hidden_dim)
            .map(|index| ((index * 17 + 5) % 41) as f32 * 0.03125 - 0.625)
            .collect::<Vec<_>>(),
    );
    let router_weight_values = (0..num_experts * hidden_dim)
        .map(|index| ((index * 29 + 11) % 251) as u8)
        .collect::<Vec<_>>();
    let router_scale_values = bf16_values(
        &(0..num_experts)
            .map(|expert| 0.000_976_562_5 + expert as f32 * 0.000_061_035_156)
            .collect::<Vec<_>>(),
    );
    let router_bias_values = bf16_values(
        &(0..num_experts)
            .map(|expert| -0.011_718_75 + expert as f32 * 0.003_906_25)
            .collect::<Vec<_>>(),
    );
    assert_eq!(router_weight_values.len(), router_config.weight_bytes());
    let hidden_state = bf16_buffer(&device, &all_hidden);
    let router_weight = Buffer::from_slice(&device, &router_weight_values);
    let router_scales = bf16_buffer(&device, &router_scale_values);
    let router_biases = bf16_buffer(&device, &router_bias_values);
    let num_router_values = num_total_tokens as usize * num_experts;
    let num_routes = num_total_tokens as usize * topk;
    let router_logits = Buffer::new_zeroed_elements(&device, num_router_values, Dtype::Bfloat16);
    let router_probs = Buffer::new_zeroed_elements(&device, num_router_values, Dtype::Bfloat16);
    let expert_indices = Buffer::new_zeroed_elements(&device, num_routes, Dtype::Uint32);
    let expert_probs = Buffer::new_zeroed_elements(&device, num_routes, Dtype::Float32);
    let scratch = MoERoutingScratchBindings {
        router_logits: &router_logits,
        router_probs: &router_probs,
        expert_indices: &expert_indices,
        expert_probs: &expert_probs,
    };
    let weights = GatedMoERoutingWeights {
        router_weight: &router_weight,
        router_scales: &router_scales,
        router_biases: &router_biases,
    };

    let expected_logits = cpu_router_logits(
        router_config,
        num_total_tokens as usize,
        &all_hidden,
        &router_weight_values,
        &router_scale_values,
        &router_bias_values,
    );
    let expected_probs = cpu_softmax_bf16_rows(&expected_logits, num_total_tokens as usize, num_experts);
    let expected_routes = moe_routing_from_bf16_probs_reference(
        &expected_probs,
        num_total_tokens as usize,
        num_experts,
        topk,
        core.norm_topk_prob,
    );

    let runtime = MetalReplayRuntime::new(&stream);
    let mut replay = Replay::new("test.gated_moe.routing", TestGatedMoERouting(moe));
    let input = GatedMoERoutingInput {
        num_total_tokens,
        num_active_tokens: ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
        hidden_state: &hidden_state,
        scratch,
        weights,
    };
    let (key, cache_hit) = replay.record(&runtime, &input);
    assert!(!cache_hit);

    for num_active_tokens in [1_u32, 8, 3, 7, 2, 6, 4, 5] {
        assert_eq!(replay.record(&runtime, &input), (key, true));
        runtime
            .submit_replay_with_arguments(
                replay.replay(&key),
                &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens),
            )
            .wait();

        let active_router_values = num_active_tokens as usize * num_experts;
        let active_routes = num_active_tokens as usize * topk;
        assert_close(
            &read_bf16_values(&router_logits, num_router_values)[..active_router_values],
            &expected_logits[..active_router_values],
            0.25,
        );
        assert_close(
            &read_bf16_values(&router_probs, num_router_values)[..active_router_values],
            &expected_probs[..active_router_values],
            0.02,
        );
        assert_eq!(
            &expert_indices.read_typed::<u32>(0, num_routes)[..active_routes],
            &expected_routes.expert_indices[..active_routes]
        );
        assert_close(
            &expert_probs.read_typed::<f32>(0, num_routes)[..active_routes],
            &expected_routes.expert_probs[..active_routes],
            0.02,
        );
    }
}

#[test]
fn test_full_replay_matches_reference_across_active_counts_and_topologies() {
    assert_full_replay_matches_reference(8, 2);
    assert_full_replay_matches_reference(512, 10);
    assert_full_replay_matches_reference(512, 17);
}

fn assert_full_replay_matches_reference(num_experts: usize, num_experts_per_token: usize) {
    let device = Device::system_default();
    let stream = inference_backend_metal::metal::Stream::new(&device);
    let runtime = MetalReplayRuntime::new(&stream);

    for with_shared_experts in [false, true] {
        let (mut core, metal) = full_test_config(with_shared_experts);
        core.num_experts = num_experts;
        core.num_experts_per_token = num_experts_per_token;
        let moe = GatedMoE::new(&device, core.clone(), metal);
        let weights = FullMoETestWeights::new(&device, &core, metal);
        let max_tokens = 8_u32;
        let scratch = MoEScratch::new(&device, &core, metal, max_tokens as usize);
        let all_hidden = bf16_values(
            &(0..max_tokens as usize * core.hidden_dim)
                .map(|index| ((index * 23 + 7) % 67) as f32 * 0.015_625 - 0.5)
                .collect::<Vec<_>>(),
        );
        let hidden_state = bf16_buffer(&device, &all_hidden);
        let output_elements = max_tokens as usize * core.hidden_dim;
        let next_hidden_state = Buffer::new_zeroed_elements(&device, output_elements, Dtype::Bfloat16);
        let mut replay = Replay::new("test.gated_moe.full", TestGatedMoE(moe));

        for (num_total_tokens, active_counts) in [
            (4_u32, [1_u32, 4, 2, 3, 0, 0, 0, 0]),
            (8_u32, [1_u32, 8, 3, 7, 2, 6, 4, 5]),
        ] {
            let input = weights.input(
                &scratch,
                &hidden_state,
                &next_hidden_state,
                num_total_tokens,
                ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            );
            let (key, cache_hit) = replay.record(&runtime, &input);
            assert!(!cache_hit);

            for num_active_tokens in active_counts.into_iter().take(num_total_tokens as usize) {
                assert_eq!(replay.record(&runtime, &input), (key, true));
                runtime
                    .submit_replay_with_arguments(
                        replay.replay(&key),
                        &ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens),
                    )
                    .wait();

                let active_values = num_active_tokens as usize * core.hidden_dim;
                let actual = read_bf16_values(&next_hidden_state, active_values);
                let expected =
                    full_moe_reference(&core, metal, &all_hidden[..active_values], num_active_tokens as usize);
                assert_eq!(actual.len(), expected.len());
                // The reference keeps F32 intermediates. Allow BF16 rounding without hiding small outputs.
                for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                    let tolerance = 1.0e-6 + 0.02 * expected.abs();
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "full MoE mismatch at {index}: actual={actual} expected={expected} tolerance={tolerance}"
                    );
                }
                if num_active_tokens == max_tokens {
                    let expert_indices = scratch
                        .bindings()
                        .routing
                        .expert_indices
                        .read_typed::<u32>(0, num_active_tokens as usize * core.num_experts_per_token);
                    assert!(
                        expert_indices
                            .iter()
                            .any(|&expert| expert as usize >= core.num_experts / 2)
                    );
                }
            }
        }
    }
}

struct TestGatedMoERouting(GatedMoE);

impl ReplayComponent for TestGatedMoERouting {
    type Key = (u32, GatedMoERoutingReplayTopology);
    type Input<'a> = GatedMoERoutingInput<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        (
            input.num_total_tokens,
            self.0.routing_replay_topology(input.num_total_tokens),
        )
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.0.record_routing(recorder, *input);
    }
}

struct TestGatedMoE(GatedMoE);

impl ReplayComponent for TestGatedMoE {
    type Key = (u32, GatedMoEReplayTopology);
    type Input<'a> = GatedMoEInput<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        (input.num_total_tokens, self.0.replay_topology(input.num_total_tokens))
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let _ = <GatedMoE as ReplayLayer>::record(&self.0, recorder, *input);
    }
}

struct FullMoETestWeights {
    router_weight: Buffer,
    router_scales: Buffer,
    router_biases: Buffer,
    topk_experts: FullMoESparseTestWeights,
    shared_experts: Option<FullMoESharedTestWeights>,
}

impl FullMoETestWeights {
    fn new(device: &Device, core: &GatedMoECore, metal: GatedMoEMetalConfig) -> Self {
        let router_shape = core.router_shape();
        let router_config =
            affine_config_with_bits(router_shape.out_dim, router_shape.in_dim, metal.router_bits, metal);
        Self {
            router_weight: Buffer::from_slice(device, &full_router_weights(router_config)),
            router_scales: filled_bf16_buffer(device, router_config.scale_or_bias_bytes(), 0.000_976_562_5),
            router_biases: Buffer::new_zeroed(device, router_config.scale_or_bias_bytes()),
            topk_experts: FullMoESparseTestWeights::new(device, topk_experts_config(core, metal)),
            shared_experts: core
                .shared_experts_core()
                .map(|shared_core| FullMoESharedTestWeights::new(device, core, &shared_core, metal)),
        }
    }

    fn input<'a>(
        &'a self,
        scratch: &'a MoEScratch,
        hidden_state: &'a Buffer,
        next_hidden_state: &'a Buffer,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
    ) -> GatedMoEInput<'a> {
        GatedMoEInput {
            num_total_tokens,
            num_active_tokens,
            hidden_state,
            next_hidden_state,
            scratch: scratch.bindings(),
            weights: self.as_borrowed(),
            shared_experts: self.shared_experts_input(scratch),
        }
    }

    fn as_borrowed(&self) -> GatedMoEWeights<'_> {
        GatedMoEWeights {
            router_weight: &self.router_weight,
            router_scales: &self.router_scales,
            router_biases: &self.router_biases,
            topk_experts: self.topk_experts.as_borrowed(),
        }
    }

    fn shared_experts_input<'a>(&'a self, scratch: &'a MoEScratch) -> Option<GatedMoESharedExpertsInput<'a>> {
        self.shared_experts.as_ref().map(|weights| {
            GatedMoESharedExpertsInput {
                scratch: scratch
                    .shared_experts_bindings()
                    .expect("full MoE shared test weights require shared scratch"),
                weights: weights.as_borrowed(),
            }
        })
    }
}

struct FullMoESparseTestWeights {
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

impl FullMoESparseTestWeights {
    fn new(device: &Device, config: sparse_mlp::Config) -> Self {
        let num_experts = config.num_experts as usize;
        let gate_up = config.gate_up_config();
        let down = config.down_config();
        let gate_up_weight_bytes = num_experts
            .checked_mul(gate_up.weight_bytes_per_expert())
            .expect("full MoE test sparse gate/up weight bytes must fit usize");
        let gate_up_param_bytes = num_experts
            .checked_mul(gate_up.affine_param_bytes_per_expert())
            .expect("full MoE test sparse gate/up parameter bytes must fit usize");
        let down_weight_bytes = num_experts
            .checked_mul(down.weight_bytes_per_expert())
            .expect("full MoE test sparse down weight bytes must fit usize");
        let down_param_bytes = num_experts
            .checked_mul(down.affine_param_bytes_per_expert())
            .expect("full MoE test sparse down parameter bytes must fit usize");
        Self {
            gate_weight: filled_u8_buffer(device, gate_up_weight_bytes, 7),
            gate_scales: filled_bf16_buffer(device, gate_up_param_bytes, 0.000_976_562_5),
            gate_biases: Buffer::new_zeroed(device, gate_up_param_bytes),
            up_weight: filled_u8_buffer(device, gate_up_weight_bytes, 11),
            up_scales: filled_bf16_buffer(device, gate_up_param_bytes, 0.000_976_562_5),
            up_biases: Buffer::new_zeroed(device, gate_up_param_bytes),
            down_weight: Buffer::from_slice(device, &full_down_weights(num_experts, down_weight_bytes)),
            down_scales: filled_bf16_buffer(device, down_param_bytes, 0.000_976_562_5),
            down_biases: Buffer::new_zeroed(device, down_param_bytes),
        }
    }

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

struct FullMoESharedTestWeights {
    gate_weight: Buffer,
    gate_scales: Buffer,
    gate_biases: Buffer,
    dense: FullMoEDenseTestWeights,
}

impl FullMoESharedTestWeights {
    fn new(device: &Device, core: &GatedMoECore, shared_core: &DenseMLPCore, metal: GatedMoEMetalConfig) -> Self {
        let gate_shape = core
            .shared_expert_gate_shape()
            .expect("full MoE shared test requires a shared gate shape");
        let gate_config = affine_config_with_bits(
            gate_shape.out_dim,
            gate_shape.in_dim,
            metal.shared_expert_gate_bits,
            metal,
        );
        Self {
            gate_weight: filled_u8_buffer(device, gate_config.weight_bytes(), 17),
            gate_scales: filled_bf16_buffer(device, gate_config.scale_or_bias_bytes(), 0.000_976_562_5),
            gate_biases: Buffer::new_zeroed(device, gate_config.scale_or_bias_bytes()),
            dense: FullMoEDenseTestWeights::new(device, shared_experts_config(shared_core, metal)),
        }
    }

    fn as_borrowed(&self) -> GatedMoESharedExpertsWeights<'_> {
        GatedMoESharedExpertsWeights {
            shared_expert_gate_weight: &self.gate_weight,
            shared_expert_gate_scales: &self.gate_scales,
            shared_expert_gate_biases: &self.gate_biases,
            shared_experts: self.dense.as_borrowed(),
        }
    }
}

struct FullMoEDenseTestWeights {
    gate_up_weight: Buffer,
    gate_up_scales: Buffer,
    gate_up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

impl FullMoEDenseTestWeights {
    fn new(device: &Device, config: dense_mlp::Config) -> Self {
        let gate_up = config.gate_up_config();
        let down = config.down_config();
        Self {
            gate_up_weight: filled_u8_buffer(device, gate_up.weight_bytes(), 19),
            gate_up_scales: filled_bf16_buffer(device, gate_up.scale_or_bias_bytes(), 0.000_976_562_5),
            gate_up_biases: Buffer::new_zeroed(device, gate_up.scale_or_bias_bytes()),
            down_weight: filled_u8_buffer(device, down.weight_bytes(), 23),
            down_scales: filled_bf16_buffer(device, down.scale_or_bias_bytes(), 0.000_976_562_5),
            down_biases: Buffer::new_zeroed(device, down.scale_or_bias_bytes()),
        }
    }

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

fn full_test_config(with_shared_experts: bool) -> (GatedMoECore, GatedMoEMetalConfig) {
    (
        GatedMoECore {
            model_layer_index: 0,
            hidden_dim: 32,
            intermediate_dim: 32,
            shared_experts_intermediate_dim: with_shared_experts.then_some(32),
            num_experts: 8,
            num_experts_per_token: 2,
            norm_topk_prob: true,
        },
        GatedMoEMetalConfig {
            group_size: 32,
            bits: 8,
            router_bits: 8,
            shared_expert_gate_bits: 8,
            io_dtype: Dtype::Bfloat16,
        },
    )
}

fn routing_test_config(norm_topk_prob: bool) -> (GatedMoECore, GatedMoEMetalConfig) {
    (
        GatedMoECore {
            model_layer_index: 0,
            hidden_dim: 32,
            intermediate_dim: 32,
            shared_experts_intermediate_dim: None,
            num_experts: 8,
            num_experts_per_token: 3,
            norm_topk_prob,
        },
        GatedMoEMetalConfig {
            group_size: 32,
            bits: 8,
            router_bits: 8,
            shared_expert_gate_bits: 8,
            io_dtype: Dtype::Bfloat16,
        },
    )
}

fn full_router_weights(config: affine_quantized::Config) -> Vec<u8> {
    let bytes_per_expert = config.weight_bytes() / config.n as usize;
    (0..config.n)
        .flat_map(|expert| std::iter::repeat_n(if expert >= config.n / 2 { 18 } else { 9 }, bytes_per_expert))
        .collect()
}

fn full_down_weights(num_experts: usize, num_weight_bytes: usize) -> Vec<u8> {
    (0..num_experts)
        .flat_map(|expert| std::iter::repeat_n(13 + (expert % 7) as u8, num_weight_bytes / num_experts))
        .collect()
}

fn full_moe_reference(core: &GatedMoECore, metal: GatedMoEMetalConfig, hidden: &[f32], num_tokens: usize) -> Vec<f32> {
    assert_eq!(hidden.len(), num_tokens * core.hidden_dim);
    let router_config = affine_config_with_bits(core.num_experts, core.hidden_dim, metal.router_bits, metal);
    let router_weights = full_router_weights(router_config);
    let router_scales = vec![0.000_976_562_5; router_config.scale_or_bias_bytes() / size_of::<u16>()];
    let router_biases = vec![0.0; router_scales.len()];
    let router_logits = cpu_router_logits(
        router_config,
        num_tokens,
        hidden,
        &router_weights,
        &router_scales,
        &router_biases,
    );
    let router_probs = cpu_softmax_bf16_rows(&router_logits, num_tokens, core.num_experts);
    let routes = moe_routing_from_bf16_probs_reference(
        &router_probs,
        num_tokens,
        core.num_experts,
        core.num_experts_per_token,
        core.norm_topk_prob,
    );

    let sparse_config = topk_experts_config(core, metal);
    let gate_up_config = sparse_config.gate_up_config();
    let down_config = sparse_config.down_config();
    let gate_up_weight_len = core.num_experts * gate_up_config.weight_bytes_per_expert();
    let gate_up_param_len = core.num_experts * gate_up_config.affine_param_bytes_per_expert() / size_of::<u16>();
    let down_weight_len = core.num_experts * down_config.weight_bytes_per_expert();
    let down_param_len = core.num_experts * down_config.affine_param_bytes_per_expert() / size_of::<u16>();
    let token_indices = (0..num_tokens)
        .flat_map(|token| (0..core.num_experts_per_token).map(move |_| token as u32))
        .collect::<Vec<_>>();
    let swiglu_indices = (0..token_indices.len() as u32).collect::<Vec<_>>();
    let routed_hidden = quantized_sparse_mlp_reference(QuantizedSparseMLPReferenceInput {
        hidden,
        token_indices: &token_indices,
        expert_indices: &routes.expert_indices,
        swiglu_indices: &swiglu_indices,
        hidden_dim: core.hidden_dim,
        intermediate_dim: core.intermediate_dim,
        group_size: metal.group_size as usize,
        bits: metal.bits as usize,
        num_experts: core.num_experts,
        weights: QuantizedSparseMLPReferenceWeights {
            gate_weight: &vec![7_u8; gate_up_weight_len],
            gate_scales: &vec![0.000_976_562_5; gate_up_param_len],
            gate_biases: &vec![0.0; gate_up_param_len],
            up_weight: &vec![11_u8; gate_up_weight_len],
            up_scales: &vec![0.000_976_562_5; gate_up_param_len],
            up_biases: &vec![0.0; gate_up_param_len],
            down_weight: &full_down_weights(core.num_experts, down_weight_len),
            down_scales: &vec![0.000_976_562_5; down_param_len],
            down_biases: &vec![0.0; down_param_len],
        },
    });
    let routed_output = moe_combine_without_shared_experts_bf16_reference(
        &routed_hidden,
        &routes.expert_probs,
        num_tokens,
        core.num_experts_per_token,
        core.hidden_dim,
    );

    let output = match core.shared_experts_core() {
        None => routed_output,
        Some(shared_core) => {
            let dense_config = shared_experts_config(&shared_core, metal);
            let gate_up = dense_config.gate_up_config();
            let down = dense_config.down_config();
            let shared_hidden = quantized_dense_mlp_reference(
                &shared_core,
                hidden,
                num_tokens,
                QuantizedDenseMLPReferenceGeometry {
                    gate_up_group_size: metal.group_size as usize,
                    gate_up_bits: metal.bits as usize,
                    down_group_size: metal.group_size as usize,
                    down_bits: metal.bits as usize,
                },
                QuantizedDenseMLPReferenceWeights {
                    gate_up_weight: &vec![19_u8; gate_up.weight_bytes()],
                    gate_up_scales: &vec![0.000_976_562_5; gate_up.scale_or_bias_bytes() / size_of::<u16>()],
                    gate_up_biases: &vec![0.0; gate_up.scale_or_bias_bytes() / size_of::<u16>()],
                    down_weight: &vec![23_u8; down.weight_bytes()],
                    down_scales: &vec![0.000_976_562_5; down.scale_or_bias_bytes() / size_of::<u16>()],
                    down_biases: &vec![0.0; down.scale_or_bias_bytes() / size_of::<u16>()],
                },
            );
            let gate_shape = core
                .shared_expert_gate_shape()
                .expect("shared MoE reference requires a shared gate shape");
            let gate_config = affine_config_with_bits(
                gate_shape.out_dim,
                gate_shape.in_dim,
                metal.shared_expert_gate_bits,
                metal,
            );
            let shared_gate_logits = quantized_affine_reference(
                QuantizedAffineReferenceShape {
                    num_rows: num_tokens,
                    output_dim: gate_shape.out_dim,
                    input_dim: gate_shape.in_dim,
                    group_size: metal.group_size as usize,
                    bits: metal.shared_expert_gate_bits as usize,
                },
                hidden,
                &vec![17_u8; gate_config.weight_bytes()],
                &vec![0.000_976_562_5; gate_config.scale_or_bias_bytes() / size_of::<u16>()],
                &vec![0.0; gate_config.scale_or_bias_bytes() / size_of::<u16>()],
            );
            moe_combine_with_shared_experts_bf16_reference(
                &routed_output,
                &shared_hidden,
                &shared_gate_logits,
                num_tokens,
                core.hidden_dim,
            )
        },
    };
    output.into_iter().map(|bits| bf16::from_bits(bits).to_f32()).collect()
}

fn cpu_router_logits(
    config: affine_quantized::Config,
    num_tokens: usize,
    hidden: &[f32],
    weights: &[u8],
    scales: &[f32],
    biases: &[f32],
) -> Vec<f32> {
    assert_eq!(config.bits, 8);
    let num_experts = config.n as usize;
    let hidden_dim = config.k as usize;
    let group_size = config.group_size as usize;
    let num_groups = hidden_dim / group_size;
    assert_eq!(hidden.len(), num_tokens * hidden_dim);
    assert_eq!(weights.len(), num_experts * hidden_dim);
    assert_eq!(scales.len(), num_experts * num_groups);
    assert_eq!(biases.len(), num_experts * num_groups);
    let mut logits = Vec::with_capacity(num_tokens * num_experts);
    for token in 0..num_tokens {
        let hidden_row = &hidden[token * hidden_dim..(token + 1) * hidden_dim];
        for expert in 0..num_experts {
            let weight_row = &weights[expert * hidden_dim..(expert + 1) * hidden_dim];
            let mut value = 0.0_f32;
            for group in 0..num_groups {
                let start = group * group_size;
                let end = start + group_size;
                let input_sum = hidden_row[start..end].iter().sum::<f32>();
                let dot = hidden_row[start..end]
                    .iter()
                    .zip(&weight_row[start..end])
                    .map(|(input, weight)| *input * f32::from(*weight))
                    .sum::<f32>();
                let affine_index = expert * num_groups + group;
                value += scales[affine_index] * dot + input_sum * biases[affine_index];
            }
            logits.push(bf16::from_f32(value).to_f32());
        }
    }
    logits
}

fn cpu_softmax_bf16_rows(values: &[f32], num_rows: usize, num_values_per_row: usize) -> Vec<f32> {
    assert_eq!(values.len(), num_rows * num_values_per_row);
    let mut output = Vec::with_capacity(values.len());
    for row in values.chunks_exact(num_values_per_row) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps = row.iter().map(|value| (*value - max).exp()).collect::<Vec<_>>();
        let sum = exps.iter().sum::<f32>();
        output.extend(exps.into_iter().map(|value| bf16::from_f32(value / sum).to_f32()));
    }
    output
}

fn bf16_values(values: &[f32]) -> Vec<f32> {
    values.iter().map(|value| bf16::from_f32(*value).to_f32()).collect()
}

fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    let bits = values
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    Buffer::from_slice(device, &bits)
}

fn filled_u8_buffer(device: &Device, len: usize, value: u8) -> Buffer {
    Buffer::from_slice(device, &vec![value; len])
}

fn filled_bf16_buffer(device: &Device, bytes: usize, value: f32) -> Buffer {
    assert_eq!(bytes % size_of::<u16>(), 0);
    Buffer::from_slice(device, &vec![bf16::from_f32(value).to_bits(); bytes / size_of::<u16>()])
}

fn read_bf16_values(buffer: &Buffer, len: usize) -> Vec<f32> {
    buffer
        .read_typed::<u16>(0, len)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value mismatch at index={index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}
