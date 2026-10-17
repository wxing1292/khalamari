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

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.moe.combine.num_active_tokens");
const ACTIVE_SEQUENCE: [u32; 8] = [1, 8, 3, 7, 2, 6, 4, 5];

#[test]
fn test_replay_matches_reference_across_active_counts_and_shared_expert_modes() {
    let fixture = CombineFixture::new();
    for with_shared_experts in [false, true] {
        let replay = fixture.replay(with_shared_experts);
        for (case_index, num_active_tokens) in ACTIVE_SEQUENCE.into_iter().enumerate() {
            let work = fixture.write_work(0x2148_937A_u32.wrapping_add(case_index as u32));
            fixture.submit(&replay, num_active_tokens);
            fixture.assert_active_output(&work, num_active_tokens, with_shared_experts);
        }
    }
}

struct CombineFixture {
    stream: Stream,
    config: Config,
    shape: Shape,
    compute: Compute,
    routed_hidden: Buffer,
    routed_probs: Buffer,
    shared_hidden: Buffer,
    shared_expert_gate_logits: Buffer,
    output: Buffer,
}

impl CombineFixture {
    fn new() -> Self {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config::bf16(3, 64);
        let shape = Shape { num_total_tokens: 8 };
        Self {
            routed_hidden: Buffer::new_zeroed(&device, config.routed_output_bytes(shape)),
            routed_probs: Buffer::new_zeroed(&device, config.routed_probs_bytes(shape)),
            shared_hidden: Buffer::new_zeroed(&device, config.output_bytes(shape)),
            shared_expert_gate_logits: Buffer::new_zeroed(&device, config.shared_expert_gate_logits_bytes(shape)),
            output: Buffer::new_zeroed(&device, config.output_bytes(shape)),
            compute: Compute::new(&device, config),
            stream,
            config,
            shape,
        }
    }

    fn replay(&self, with_shared_experts: bool) -> ReplayProgram {
        let mut builder = self.stream.create_replay_program();
        if with_shared_experts {
            builder.record(self.compute.invoke_with_shared_experts(
                self.shape,
                ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
                WithSharedExpertsBuffers {
                    routed_hidden: &self.routed_hidden,
                    routed_probs: &self.routed_probs,
                    shared_hidden: &self.shared_hidden,
                    shared_expert_gate_logits: &self.shared_expert_gate_logits,
                    output: &self.output,
                },
            ));
        } else {
            builder.record(self.compute.invoke_without_shared_experts(
                self.shape,
                ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
                WithoutSharedExpertsBuffers {
                    routed_hidden: &self.routed_hidden,
                    routed_probs: &self.routed_probs,
                    output: &self.output,
                },
            ));
        }
        builder.build()
    }

    fn write_work(&self, seed: u32) -> CombineWork {
        let num_total_tokens = self.shape.num_total_tokens as usize;
        let hidden_dim = self.config.hidden_dim as usize;
        let num_experts_per_token = self.config.num_experts_per_token as usize;
        let routed_hidden = bf16_values(&generated_values(
            num_total_tokens * num_experts_per_token * hidden_dim,
            seed,
        ));
        let routed_probs = generated_probs(num_total_tokens, num_experts_per_token, seed.wrapping_add(1));
        let shared_hidden = bf16_values(&generated_values(num_total_tokens * hidden_dim, seed.wrapping_add(2)));
        let shared_expert_gate_logits = bf16_values(&generated_values(num_total_tokens, seed.wrapping_add(3)));
        self.routed_hidden.write_typed(0, &bf16_bits(&routed_hidden));
        self.routed_probs.write_typed(0, &routed_probs);
        self.shared_hidden.write_typed(0, &bf16_bits(&shared_hidden));
        self.shared_expert_gate_logits
            .write_typed(0, &bf16_bits(&shared_expert_gate_logits));
        CombineWork {
            routed_hidden,
            routed_probs,
            shared_hidden,
            shared_expert_gate_logits,
        }
    }

    fn submit(&self, replay: &ReplayProgram, num_active_tokens: u32) {
        let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens);
        self.stream.submit_replay_with_arguments(replay, &arguments).wait();
    }

    fn assert_active_output(&self, work: &CombineWork, num_active_tokens: u32, with_shared_experts: bool) {
        let num_active_tokens = num_active_tokens as usize;
        let hidden_dim = self.config.hidden_dim as usize;
        let num_experts_per_token = self.config.num_experts_per_token as usize;
        let num_active_routes = num_active_tokens * num_experts_per_token;
        let routed = moe_combine_without_shared_experts_bf16_reference(
            &work.routed_hidden[..num_active_routes * hidden_dim],
            &work.routed_probs[..num_active_routes],
            num_active_tokens,
            num_experts_per_token,
            hidden_dim,
        );
        let expected = if with_shared_experts {
            moe_combine_with_shared_experts_bf16_reference(
                &routed,
                &work.shared_hidden[..num_active_tokens * hidden_dim],
                &work.shared_expert_gate_logits[..num_active_tokens],
                num_active_tokens,
                hidden_dim,
            )
        } else {
            routed
        };
        let actual = self.output.read_typed::<u16>(0, expected.len());
        assert_close_bits(&actual, &expected, 1.0e-3);
    }
}

struct CombineWork {
    routed_hidden: Vec<f32>,
    routed_probs: Vec<f32>,
    shared_hidden: Vec<f32>,
    shared_expert_gate_logits: Vec<f32>,
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

fn assert_close_bits(actual: &[u16], expected: &[u16], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let actual = bf16::from_bits(actual).to_f32();
        let expected = bf16::from_bits(expected).to_f32();
        assert!(
            (actual - expected).abs() <= tolerance,
            "value mismatch at index={index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}
