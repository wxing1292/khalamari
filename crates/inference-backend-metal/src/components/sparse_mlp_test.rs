use std::mem::size_of;

use half::bf16;
use inference_executor_core::mlp::moe::reference::QuantizedSparseMLPReferenceInput;
use inference_executor_core::mlp::moe::reference::QuantizedSparseMLPReferenceWeights;
use inference_executor_core::mlp::moe::reference::quantized_sparse_mlp_reference;

use super::*;
use crate::metal::Buffer;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayProgram;
use crate::metal::ReplayU32;
use crate::metal::Stream;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.sparse_mlp.num_active_tokens");

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Layout {
    TokenMajor,
    ExpertMajor,
}

#[test]
fn test_replay_active_counts() {
    for (num_total_tokens, intermediate_dim, group_size, num_experts, active_sequence) in [
        (4, 64, 32, 5, [1, 4, 3, 2]),
        (17, 512, 32, 5, [1, 17, 16, 3]),
        // Nine tokens put three-row QMV and four-row QMM segments in one replay.
        (49, 64, 32, 5, [1, 49, 9, 10]),
        // Two K tiles share each quantization group, including mixed QMV/QMM segments.
        (49, 64, 64, 5, [1, 49, 9, 10]),
        // Sparse occupancy crosses the 32-expert SIMD batch while active replay counts change.
        (64, 64, 64, 33, [1, 64, 16, 17]),
        // Cross the 32-expert SIMD batch with empty experts and mixed QMV/QMM segments.
        (97, 64, 64, 33, [1, 97, 16, 17]),
        // Dense occupancy records the wider expert tile and replays partial active batches.
        (161, 64, 64, 5, [1, 161, 16, 17]),
    ] {
        let fixture = SparseMLPFixture::new(num_total_tokens, 2, intermediate_dim, group_size, num_experts);
        for layout in [Layout::TokenMajor, Layout::ExpertMajor] {
            let replay = fixture.replay(layout);
            for (case_index, num_active_tokens) in active_sequence.into_iter().enumerate() {
                let work = fixture.write_work(
                    num_active_tokens,
                    0x8100_0000_u32
                        .wrapping_add(intermediate_dim)
                        .wrapping_add(case_index as u32),
                );
                fixture.submit(&replay, num_active_tokens);
                fixture.assert_active_output(layout, &work);
            }
        }
    }
}

struct SparseMLPFixture {
    stream: Stream,
    config: Config,
    compute: Compute,
    num_total_tokens: u32,
    num_experts_per_token: u32,
    token_major_input: Buffer,
    token_indices: Buffer,
    expert_indices: Buffer,
    route_indices: Buffer,
    token_major_output: Buffer,
    token_major_swiglu: Buffer,
    expert_major_input: Buffer,
    experts_by_route: Buffer,
    expert_offsets: Buffer,
    expert_major_output: Buffer,
    expert_major_swiglu: Buffer,
    weights: SparseMLPWeights,
}

impl SparseMLPFixture {
    fn new(
        num_total_tokens: u32,
        num_experts_per_token: u32,
        intermediate_dim: u32,
        group_size: u32,
        num_experts: u32,
    ) -> Self {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config {
            num_experts,
            hidden_dim: 64,
            intermediate_dim,
            group_size,
            bits: 4,
            dtype: Dtype::Bfloat16,
        };
        let num_total_routes = num_total_tokens * num_experts_per_token;
        let token_shape = TokenMajorShape {
            num_total_routes,
            num_total_tokens,
        };
        let expert_shape = ExpertMajorShape {
            num_total_routes,
            num_total_tokens,
            num_experts_per_token,
        };
        let route_index_bytes = num_total_routes as usize * size_of::<u32>();
        let compute = Compute::new(&device, config);
        let weights = SparseMLPWeights::new(&device, config);
        Self {
            token_major_input: Buffer::new_zeroed(&device, config.token_major_input_bytes(token_shape)),
            token_indices: Buffer::new_zeroed(&device, route_index_bytes),
            expert_indices: Buffer::new_zeroed(&device, route_index_bytes),
            route_indices: Buffer::new_zeroed(&device, route_index_bytes),
            token_major_output: Buffer::new_zeroed(&device, config.token_major_output_bytes(token_shape)),
            token_major_swiglu: Buffer::new_zeroed(&device, config.swiglu_bytes(num_total_routes)),
            expert_major_input: Buffer::new_zeroed(&device, config.expert_major_input_bytes(expert_shape)),
            experts_by_route: Buffer::new_zeroed(&device, route_index_bytes),
            expert_offsets: Buffer::new_zeroed(&device, (config.num_experts as usize + 1) * size_of::<u32>()),
            expert_major_output: Buffer::new_zeroed(&device, config.expert_major_output_bytes(expert_shape)),
            expert_major_swiglu: Buffer::new_zeroed(&device, config.swiglu_bytes(num_total_routes)),
            stream,
            config,
            compute,
            num_total_tokens,
            num_experts_per_token,
            weights,
        }
    }

    fn num_total_routes(&self) -> u32 {
        self.num_total_tokens * self.num_experts_per_token
    }

    fn num_active_routes(&self, num_active_tokens: u32) -> u32 {
        assert!(num_active_tokens <= self.num_total_tokens);
        num_active_tokens * self.num_experts_per_token
    }

    fn token_major_shape(&self) -> TokenMajorShape {
        TokenMajorShape {
            num_total_routes: self.num_total_routes(),
            num_total_tokens: self.num_total_tokens,
        }
    }

    fn expert_major_shape(&self) -> ExpertMajorShape {
        ExpertMajorShape {
            num_total_routes: self.num_total_routes(),
            num_total_tokens: self.num_total_tokens,
            num_experts_per_token: self.num_experts_per_token,
        }
    }

    fn replay(&self, layout: Layout) -> ReplayProgram {
        let mut builder = self.stream.create_replay_program();
        match layout {
            Layout::TokenMajor => {
                builder.record(self.compute.invoke_token_major(
                    self.token_major_shape(),
                    self.num_experts_per_token,
                    ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
                    TokenMajorBuffers {
                        input: &self.token_major_input,
                        token_indices: &self.token_indices,
                        expert_indices: &self.expert_indices,
                        route_indices: &self.route_indices,
                        routed_hidden: &self.token_major_output,
                    },
                    Scratch {
                        swiglu: &self.token_major_swiglu,
                    },
                    self.weights.bindings(),
                ))
            },
            Layout::ExpertMajor => {
                builder.record(self.compute.invoke_expert_major(
                    self.expert_major_shape(),
                    ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
                    ExpertMajorBuffers {
                        packed_input: &self.expert_major_input,
                        experts_by_route: &self.experts_by_route,
                        expert_offsets: &self.expert_offsets,
                        packed_output: &self.expert_major_output,
                    },
                    Scratch {
                        swiglu: &self.expert_major_swiglu,
                    },
                    self.weights.bindings(),
                ))
            },
        }
        builder.build()
    }

    fn write_work(&self, num_active_tokens: u32, seed: u32) -> ActiveSparseMLPInput {
        let hidden_dim = self.config.hidden_dim as usize;
        let num_total_routes = self.num_total_routes() as usize;
        let num_active_routes = self.num_active_routes(num_active_tokens) as usize;

        let token_hidden = bf16_values(&generated_values(self.num_total_tokens as usize * hidden_dim, seed));
        self.token_major_input.write_typed(0, &bf16_bits(&token_hidden));

        let packed_hidden = bf16_values(&generated_values(num_total_routes * hidden_dim, seed.wrapping_add(1)));
        self.expert_major_input.write_typed(0, &bf16_bits(&packed_hidden));

        let token_indices = (0..num_total_routes)
            .map(|route| route as u32 / self.num_experts_per_token)
            .collect::<Vec<_>>();
        let expert_indices = (0..num_total_routes)
            .map(|route| (route as u32 * 3 + seed) % self.config.num_experts)
            .collect::<Vec<_>>();
        let route_indices = (0..num_total_routes as u32).collect::<Vec<_>>();
        self.token_indices.write_typed(0, &token_indices);
        self.expert_indices.write_typed(0, &expert_indices);
        self.route_indices.write_typed(0, &route_indices);
        let mut experts_by_route = expert_indices.clone();
        experts_by_route[..num_active_routes].sort_unstable();
        self.experts_by_route.write_typed(0, &experts_by_route);
        let offsets = (0..=self.config.num_experts)
            .map(|expert| experts_by_route[..num_active_routes].partition_point(|&value| value < expert) as u32)
            .collect::<Vec<_>>();
        self.expert_offsets.write_typed(0, &offsets);

        ActiveSparseMLPInput {
            token_hidden: token_hidden[..num_active_tokens as usize * hidden_dim].to_vec(),
            packed_hidden: packed_hidden[..num_active_routes * hidden_dim].to_vec(),
            token_indices: token_indices[..num_active_routes].to_vec(),
            expert_indices: expert_indices[..num_active_routes].to_vec(),
            experts_by_route: experts_by_route[..num_active_routes].to_vec(),
            route_indices: route_indices[..num_active_routes].to_vec(),
        }
    }

    fn submit(&self, replay: &ReplayProgram, num_active_tokens: u32) {
        let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens);
        self.stream.submit_replay_with_arguments(replay, &arguments).wait();
    }

    fn assert_active_output(&self, layout: Layout, input: &ActiveSparseMLPInput) {
        let (hidden, token_indices, output) = match layout {
            Layout::TokenMajor => {
                (
                    input.token_hidden.as_slice(),
                    input.token_indices.as_slice(),
                    &self.token_major_output,
                )
            },
            Layout::ExpertMajor => {
                (
                    input.packed_hidden.as_slice(),
                    input.route_indices.as_slice(),
                    &self.expert_major_output,
                )
            },
        };
        let expected = quantized_sparse_mlp_reference(QuantizedSparseMLPReferenceInput {
            hidden,
            token_indices,
            expert_indices: match layout {
                Layout::TokenMajor => &input.expert_indices,
                Layout::ExpertMajor => &input.experts_by_route,
            },
            swiglu_indices: &input.route_indices,
            hidden_dim: self.config.hidden_dim as usize,
            intermediate_dim: self.config.intermediate_dim as usize,
            group_size: self.config.group_size as usize,
            bits: self.config.bits as usize,
            num_experts: self.config.num_experts as usize,
            weights: self.weights.reference(),
        })
        .into_iter()
        .map(|value| bf16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
        let actual = output
            .read_typed::<u16>(0, expected.len())
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        assert_close_rel(&actual, &expected, 2.0e-5, 8.0e-3);
    }
}

struct ActiveSparseMLPInput {
    token_hidden: Vec<f32>,
    packed_hidden: Vec<f32>,
    token_indices: Vec<u32>,
    expert_indices: Vec<u32>,
    experts_by_route: Vec<u32>,
    route_indices: Vec<u32>,
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
    gate_weight_values: Vec<u8>,
    gate_scale_values: Vec<f32>,
    gate_bias_values: Vec<f32>,
    up_weight_values: Vec<u8>,
    up_scale_values: Vec<f32>,
    up_bias_values: Vec<f32>,
    down_weight_values: Vec<u8>,
    down_scale_values: Vec<f32>,
    down_bias_values: Vec<f32>,
}

impl SparseMLPWeights {
    fn new(device: &Device, config: Config) -> Self {
        let num_experts = config.num_experts as usize;
        let gate_up = config.gate_up_config();
        let down = config.down_config();
        let gate_weight_values = generated_bytes(num_experts * gate_up.weight_bytes_per_expert(), 0x8300_0001);
        let gate_scale_values = bf16_values(&generated_scales(
            num_experts * gate_up.affine_param_bytes_per_expert() / size_of::<u16>(),
            0x8300_0002,
        ));
        let gate_bias_values = bf16_values(&generated_biases(
            num_experts * gate_up.affine_param_bytes_per_expert() / size_of::<u16>(),
            0x8300_0003,
        ));
        let up_weight_values = generated_bytes(num_experts * gate_up.weight_bytes_per_expert(), 0x8300_0004);
        let up_scale_values = bf16_values(&generated_scales(
            num_experts * gate_up.affine_param_bytes_per_expert() / size_of::<u16>(),
            0x8300_0005,
        ));
        let up_bias_values = bf16_values(&generated_biases(
            num_experts * gate_up.affine_param_bytes_per_expert() / size_of::<u16>(),
            0x8300_0006,
        ));
        let down_weight_values = generated_bytes(num_experts * down.weight_bytes_per_expert(), 0x8300_0007);
        let down_scale_values = bf16_values(&generated_scales(
            num_experts * down.affine_param_bytes_per_expert() / size_of::<u16>(),
            0x8300_0008,
        ));
        let down_bias_values = bf16_values(&generated_biases(
            num_experts * down.affine_param_bytes_per_expert() / size_of::<u16>(),
            0x8300_0009,
        ));
        Self {
            gate_weight: Buffer::from_slice(device, &gate_weight_values),
            gate_scales: bf16_buffer(device, &gate_scale_values),
            gate_biases: bf16_buffer(device, &gate_bias_values),
            up_weight: Buffer::from_slice(device, &up_weight_values),
            up_scales: bf16_buffer(device, &up_scale_values),
            up_biases: bf16_buffer(device, &up_bias_values),
            down_weight: Buffer::from_slice(device, &down_weight_values),
            down_scales: bf16_buffer(device, &down_scale_values),
            down_biases: bf16_buffer(device, &down_bias_values),
            gate_weight_values,
            gate_scale_values,
            gate_bias_values,
            up_weight_values,
            up_scale_values,
            up_bias_values,
            down_weight_values,
            down_scale_values,
            down_bias_values,
        }
    }

    fn bindings(&self) -> Weights<'_> {
        Weights {
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

    fn reference(&self) -> QuantizedSparseMLPReferenceWeights<'_> {
        QuantizedSparseMLPReferenceWeights {
            gate_weight: &self.gate_weight_values,
            gate_scales: &self.gate_scale_values,
            gate_biases: &self.gate_bias_values,
            up_weight: &self.up_weight_values,
            up_scales: &self.up_scale_values,
            up_biases: &self.up_bias_values,
            down_weight: &self.down_weight_values,
            down_scales: &self.down_scale_values,
            down_biases: &self.down_bias_values,
        }
    }
}

fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    Buffer::from_slice(device, &bf16_bits(values))
}

fn bf16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect()
}

fn bf16_values(values: &[f32]) -> Vec<f32> {
    values.iter().map(|value| bf16::from_f32(*value).to_f32()).collect()
}

fn generated_values(count: usize, mut state: u32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
        })
        .collect()
}

fn generated_scales(count: usize, random_seed: u32) -> Vec<f32> {
    generated_values(count, random_seed)
        .into_iter()
        .map(|value| 0.0005 + value.abs() * 0.001)
        .collect()
}

fn generated_biases(count: usize, random_seed: u32) -> Vec<f32> {
    generated_values(count, random_seed)
        .into_iter()
        .map(|value| value * 0.0002)
        .collect()
}

fn generated_bytes(count: usize, mut state: u32) -> Vec<u8> {
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) as u8
        })
        .collect()
}

fn assert_close_rel(actual: &[f32], expected: &[f32], abs_tolerance: f32, rel_tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        let tolerance = abs_tolerance.max(expected.abs() * rel_tolerance);
        assert!(
            diff <= tolerance,
            "sparse MLP output mismatch at {index}: expected={expected} actual={actual} diff={diff} \
             tolerance={tolerance}"
        );
    }
}
