use half::bf16;
use inference_executor_core::mlp::moe::reference::moe_combine_with_shared_experts_bf16_reference;
use inference_executor_core::mlp::moe::reference::moe_combine_without_shared_experts_bf16_reference;

use super::*;
use crate::metal::Buffer;
use crate::metal::Device;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayProgram;
use crate::metal::Stream;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.moe.expert_major.num_active_tokens");
const ACTIVE_SEQUENCE: [u32; 8] = [1, 8, 3, 7, 2, 6, 4, 5];

#[test]
fn test_replay_matches_reference_across_active_counts() {
    let fixture = ExpertMajorFixture::new();
    let replay = fixture.replay();
    for (case_index, num_active_tokens) in ACTIVE_SEQUENCE.into_iter().enumerate() {
        let work = fixture.write_work(0x7148_9200_u32.wrapping_add(case_index as u32));
        fixture.submit(&replay, num_active_tokens);
        fixture.assert_active_work(&work, num_active_tokens);
    }
}

struct ExpertMajorWork {
    input: Vec<f32>,
    expert_indices: Vec<u32>,
    routed_probs: Vec<f32>,
    shared_hidden: Vec<f32>,
    shared_expert_gate_logits: Vec<f32>,
}

struct ExpertMajorFixture {
    stream: Stream,
    config: Config,
    shape: Shape,
    compute: Compute,
    input: Buffer,
    expert_indices: Buffer,
    routed_probs: Buffer,
    shared_hidden: Buffer,
    shared_expert_gate_logits: Buffer,
    expert_counts: Buffer,
    expert_offsets: Buffer,
    expert_cursors: Buffer,
    routes_by_expert: Buffer,
    routes_by_token: Buffer,
    experts_by_route: Buffer,
    packed_input: Buffer,
    output: Buffer,
    output_with_shared_experts: Buffer,
}

impl ExpertMajorFixture {
    fn new() -> Self {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config::bf16(6, 3, 16);
        let shape = Shape { num_total_tokens: 8 };
        Self {
            input: Buffer::new_zeroed(&device, config.token_hidden_bytes(shape)),
            expert_indices: Buffer::new_zeroed(&device, config.route_indices_bytes(shape)),
            routed_probs: Buffer::new_zeroed(&device, config.route_probs_bytes(shape)),
            shared_hidden: Buffer::new_zeroed(&device, config.token_hidden_bytes(shape)),
            shared_expert_gate_logits: Buffer::new_zeroed(&device, config.shared_expert_gate_logits_bytes(shape)),
            expert_counts: Buffer::new_zeroed(&device, config.expert_counts_bytes()),
            expert_offsets: Buffer::new_zeroed(&device, config.expert_offsets_bytes()),
            expert_cursors: Buffer::new_zeroed(&device, config.expert_counts_bytes()),
            routes_by_expert: Buffer::new_zeroed(&device, config.route_indices_bytes(shape)),
            routes_by_token: Buffer::new_zeroed(&device, config.route_indices_bytes(shape)),
            experts_by_route: Buffer::new_zeroed(&device, config.route_indices_bytes(shape)),
            packed_input: Buffer::new_zeroed(&device, config.route_hidden_bytes(shape)),
            output: Buffer::new_zeroed(&device, config.token_hidden_bytes(shape)),
            output_with_shared_experts: Buffer::new_zeroed(&device, config.token_hidden_bytes(shape)),
            compute: Compute::new(&device, config),
            stream,
            config,
            shape,
        }
    }

    fn replay(&self) -> ReplayProgram {
        let mut builder = self.stream.create_replay_program();
        builder.record(self.compute.invoke_layout(
            self.shape,
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            LayoutBuffers {
                expert_indices: &self.expert_indices,
                expert_counts: &self.expert_counts,
                expert_offsets: &self.expert_offsets,
                expert_cursors: &self.expert_cursors,
                routes_by_expert: &self.routes_by_expert,
                routes_by_token: &self.routes_by_token,
                experts_by_route: &self.experts_by_route,
            },
        ));
        builder.record_with_barrier_before(self.compute.invoke_pack_input(
            self.shape,
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            PackInputBuffers {
                input: &self.input,
                routes_by_expert: &self.routes_by_expert,
                packed_input: &self.packed_input,
            },
        ));
        builder.record_with_barrier_before(self.compute.invoke_scatter_without_shared_experts(
            self.shape,
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            ScatterWithoutSharedExpertsBuffers {
                packed_output: &self.packed_input,
                routes_by_token: &self.routes_by_token,
                routed_probs: &self.routed_probs,
                output: &self.output,
            },
        ));
        builder.record_with_barrier_before(self.compute.invoke_scatter_with_shared_experts(
            self.shape,
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            ScatterWithSharedExpertsBuffers {
                packed_output: &self.packed_input,
                routes_by_token: &self.routes_by_token,
                routed_probs: &self.routed_probs,
                shared_hidden: &self.shared_hidden,
                shared_expert_gate_logits: &self.shared_expert_gate_logits,
                output: &self.output_with_shared_experts,
            },
        ));
        builder.build()
    }

    fn write_work(&self, seed: u32) -> ExpertMajorWork {
        let num_tokens = self.shape.num_total_tokens as usize;
        let num_routes = self.config.num_routes(self.shape) as usize;
        let hidden_dim = self.config.hidden_dim as usize;
        let input = bf16_values(&generated_values(num_tokens * hidden_dim, seed));
        let expert_indices = (0..num_routes)
            .map(|route| (route as u32 * 5 + seed) % self.config.num_experts)
            .collect::<Vec<_>>();
        let routed_probs = generated_probs(
            num_tokens,
            self.config.num_experts_per_token as usize,
            seed.wrapping_add(1),
        );
        let shared_hidden = bf16_values(&generated_values(num_tokens * hidden_dim, seed.wrapping_add(2)));
        let shared_expert_gate_logits = bf16_values(&generated_values(num_tokens, seed.wrapping_add(3)));
        self.input.write_typed(0, &bf16_bits(&input));
        self.expert_indices.write_typed(0, &expert_indices);
        self.routed_probs.write_typed(0, &routed_probs);
        self.shared_hidden.write_typed(0, &bf16_bits(&shared_hidden));
        self.shared_expert_gate_logits
            .write_typed(0, &bf16_bits(&shared_expert_gate_logits));
        ExpertMajorWork {
            input,
            expert_indices,
            routed_probs,
            shared_hidden,
            shared_expert_gate_logits,
        }
    }

    fn submit(&self, replay: &ReplayProgram, num_active_tokens: u32) {
        let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens);
        self.stream.submit_replay_with_arguments(replay, &arguments).wait();
    }

    fn assert_active_work(&self, work: &ExpertMajorWork, num_active_tokens: u32) {
        let num_active_tokens = num_active_tokens as usize;
        let num_active_routes = num_active_tokens * self.config.num_experts_per_token as usize;
        let hidden_dim = self.config.hidden_dim as usize;
        let expert_indices = &work.expert_indices[..num_active_routes];

        let expected_counts = expert_counts_reference(expert_indices, self.config.num_experts as usize);
        let expected_offsets = expert_offsets_reference(&expected_counts);
        assert_eq!(
            self.expert_counts
                .read_typed::<u32>(0, self.config.num_experts as usize),
            expected_counts
        );
        assert_eq!(
            self.expert_offsets
                .read_typed::<u32>(0, self.config.num_experts as usize + 1),
            expected_offsets
        );

        let routes_by_expert = self.routes_by_expert.read_typed::<u32>(0, num_active_routes);
        let routes_by_token = self.routes_by_token.read_typed::<u32>(0, num_active_routes);
        let experts_by_route = self.experts_by_route.read_typed::<u32>(0, num_active_routes);
        assert_expert_major_maps(
            expert_indices,
            &expected_offsets,
            &routes_by_expert,
            &routes_by_token,
            &experts_by_route,
        );
        assert_packed_input_matches_routes(
            &work.input,
            &self.packed_input.read_typed::<u16>(0, num_active_routes * hidden_dim),
            &routes_by_expert,
            self.config.num_experts_per_token as usize,
            hidden_dim,
        );

        let routed_hidden = repeated_route_hidden(
            &work.input,
            num_active_tokens,
            self.config.num_experts_per_token as usize,
            hidden_dim,
        );
        let routed = moe_combine_without_shared_experts_bf16_reference(
            &routed_hidden,
            &work.routed_probs[..num_active_routes],
            num_active_tokens,
            self.config.num_experts_per_token as usize,
            hidden_dim,
        );
        assert_eq!(self.output.read_typed::<u16>(0, routed.len()), routed);
        let expected_with_shared_experts = moe_combine_with_shared_experts_bf16_reference(
            &routed,
            &work.shared_hidden[..num_active_tokens * hidden_dim],
            &work.shared_expert_gate_logits[..num_active_tokens],
            num_active_tokens,
            hidden_dim,
        );
        assert_eq!(
            self.output_with_shared_experts
                .read_typed::<u16>(0, expected_with_shared_experts.len()),
            expected_with_shared_experts
        );
    }
}

fn expert_counts_reference(expert_indices: &[u32], num_experts: usize) -> Vec<u32> {
    let mut counts = vec![0_u32; num_experts];
    for &expert_index in expert_indices {
        counts[expert_index as usize] += 1;
    }
    counts
}

fn expert_offsets_reference(expert_counts: &[u32]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(expert_counts.len() + 1);
    offsets.push(0);
    for &count in expert_counts {
        offsets.push(offsets.last().copied().unwrap() + count);
    }
    offsets
}

fn assert_expert_major_maps(
    expert_indices: &[u32],
    expert_offsets: &[u32],
    routes_by_expert: &[u32],
    routes_by_token: &[u32],
    experts_by_route: &[u32],
) {
    let mut seen_routes = vec![false; expert_indices.len()];
    for (expert_route, &original_route) in routes_by_expert.iter().enumerate() {
        let original_route = original_route as usize;
        assert!(original_route < expert_indices.len());
        assert!(!seen_routes[original_route]);
        seen_routes[original_route] = true;
        assert_eq!(routes_by_token[original_route] as usize, expert_route);
        assert_eq!(experts_by_route[expert_route], expert_indices[original_route]);
    }
    assert!(seen_routes.into_iter().all(|seen| seen));
    for expert in 0..expert_offsets.len() - 1 {
        for &actual_expert in &experts_by_route[expert_offsets[expert] as usize..expert_offsets[expert + 1] as usize] {
            assert_eq!(actual_expert as usize, expert);
        }
    }
}

fn assert_packed_input_matches_routes(
    input: &[f32],
    packed_input: &[u16],
    routes_by_expert: &[u32],
    num_experts_per_token: usize,
    hidden_dim: usize,
) {
    for (expert_route, &original_route) in routes_by_expert.iter().enumerate() {
        let token = original_route as usize / num_experts_per_token;
        for hidden_index in 0..hidden_dim {
            assert_eq!(
                packed_input[expert_route * hidden_dim + hidden_index],
                bf16::from_f32(input[token * hidden_dim + hidden_index]).to_bits()
            );
        }
    }
}

fn repeated_route_hidden(
    input: &[f32],
    num_tokens: usize,
    num_experts_per_token: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(num_tokens * num_experts_per_token * hidden_dim);
    for token in 0..num_tokens {
        let hidden = &input[token * hidden_dim..(token + 1) * hidden_dim];
        for _ in 0..num_experts_per_token {
            output.extend_from_slice(hidden);
        }
    }
    output
}

fn generated_values(count: usize, mut state: u32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
        })
        .collect()
}

fn generated_probs(num_tokens: usize, num_routes: usize, mut state: u32) -> Vec<f32> {
    let mut values = Vec::with_capacity(num_tokens * num_routes);
    for _ in 0..num_tokens {
        let row = (0..num_routes)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / 16_777_216.0) + 0.01
            })
            .collect::<Vec<_>>();
        let sum = row.iter().sum::<f32>();
        values.extend(row.into_iter().map(|value| value / sum));
    }
    values
}

fn bf16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect()
}

fn bf16_values(values: &[f32]) -> Vec<f32> {
    values.iter().map(|value| bf16::from_f32(*value).to_f32()).collect()
}
