use half::bf16;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::Stream;
use inference_executor_core::gated_residual::reference;

use super::*;
use crate::def::replay_op::MetalReplayRuntime;
use crate::def::replay_op::ReplayRecorder;
use crate::replay::Replay;
use crate::replay::ReplayComponent;

const ACTIVE: ReplayParameterKey = ReplayParameterKey::new("test.gr.num_active_tokens");

#[test]
fn test_read_write_replay() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let runtime = MetalReplayRuntime::new(&stream);
    for (hidden, rank, total) in [(37, 17, 4), (256, 32, 4), (256, 32, 17)] {
        let core = GatedResidualCore {
            num_branches: 4,
            hidden_dim: hidden,
            low_rank_dim: rank,
            eps: 1.0e-6,
        };
        let wide = core.residual_dim();
        let mut replay = Replay::new(
            "test-gr",
            Fixture {
                gr: GatedResidual::new(&device, core),
                scratch: GatedResidualScratch::new(&device, core, total),
                residual: fixture_buffer(&device, total * wide, 0.05, 11),
                hidden: Buffer::new_zeroed_elements(&device, total * core.hidden_dim, Dtype::Bfloat16),
                output: Buffer::new_zeroed_elements(&device, total * wide, Dtype::Bfloat16),
                final_hidden: Buffer::new_zeroed_elements(&device, total * core.hidden_dim, Dtype::Bfloat16),
                norm: fixture_buffer(&device, wide, 0.01, 7),
                down: fixture_buffer(&device, wide * core.low_rank_dim, 0.005, 13),
                up: fixture_buffer(&device, wide * core.low_rank_dim, 0.02, 19),
                gate: Buffer::new_zeroed_elements(&device, total * core.num_branches, Dtype::Bfloat16),
                write: fixture_buffer(&device, wide * core.num_branches, 0.02, 23),
            },
        );
        let (key, hit) = replay.record(&runtime, &(total as u32));
        assert!(!hit);
        for active in [1, total, total - 1, 2] {
            assert!(replay.record(&runtime, &(total as u32)).1);
            runtime
                .submit_replay_with_arguments(
                    replay.replay(&key),
                    &ReplayArguments::new().with_u32(ACTIVE, active as u32),
                )
                .wait();
            let fixture = replay.component();
            let residual = values(&fixture.residual, active * wide);
            let norm = values(&fixture.norm, wide);
            let down = values(&fixture.down, wide * core.low_rank_dim);
            let up = values(&fixture.up, wide * core.low_rank_dim);
            let write = values(&fixture.write, wide * core.num_branches);
            let weights = reference::ReadWeights {
                norm: &norm,
                down: &down,
                up: &up,
            };
            let read = reference::read(reference::ReadInput {
                core,
                precision: reference::Precision::Bf16,
                num_tokens: active,
                residual: &residual,
                weights,
            });
            let output = reference::write(reference::WriteInput {
                core,
                precision: reference::Precision::Bf16,
                num_tokens: active,
                residual: &residual,
                normalized_residual: &read.normalized_residual,
                hidden_state: &read.hidden_state,
                weight: &write,
            });
            assert_close(&values(&fixture.output, active * wide), &output);
            let final_read = reference::read(reference::ReadInput {
                core,
                precision: reference::Precision::Bf16,
                num_tokens: active,
                residual: &output,
                weights,
            });
            assert_close(
                &values(&fixture.final_hidden, active * core.hidden_dim),
                &final_read.hidden_state,
            );
        }
        assert!(!replay.record(&runtime, &2).1);
    }
}

struct Fixture {
    gr: GatedResidual,
    scratch: GatedResidualScratch,
    residual: Buffer,
    hidden: Buffer,
    output: Buffer,
    final_hidden: Buffer,
    norm: Buffer,
    down: Buffer,
    up: Buffer,
    write: Buffer,
    gate: Buffer,
}

impl ReplayComponent for Fixture {
    type Key = u32;
    type Input<'a> = u32;

    fn replay_key(&self, input: &u32) -> u32 {
        *input
    }

    fn record(&self, recorder: &mut ReplayRecorder, input: &u32) {
        let weights = gated_residual::ReadWeights {
            norm: &self.norm,
            down: &self.down,
            up: &self.up,
        };
        let read = self.gr.read(
            recorder,
            ReadInput {
                num_total_tokens: *input,
                num_active_tokens: ReplayU32::Parameter(ACTIVE),
                residual: &self.residual,
                hidden_state: &self.hidden,
                scratch: &self.scratch,
                weights,
                write_gate: Some(gated_residual::WriteGate {
                    weight: &self.write,
                    output: &self.gate,
                }),
            },
        );
        self.gr.write(
            recorder,
            WriteInput {
                num_total_tokens: *input,
                num_active_tokens: ReplayU32::Parameter(ACTIVE),
                residual: &self.residual,
                gate: &self.gate,
                hidden_state: read,
                output: &self.output,
            },
        );
        // The recorder fuses write and the final read normalization with shared scratch.
        self.gr.read(
            recorder,
            ReadInput {
                num_total_tokens: *input,
                num_active_tokens: ReplayU32::Parameter(ACTIVE),
                residual: &self.output,
                hidden_state: &self.final_hidden,
                scratch: &self.scratch,
                weights,
                write_gate: None,
            },
        );
    }
}

fn fixture_buffer(device: &Device, len: usize, scale: f32, seed: usize) -> Buffer {
    Buffer::from_slice(
        device,
        &(0..len)
            .map(|index| bf16::from_f32((((index * seed + index / 11) % 61) as f32 - 30.0) * scale).to_bits())
            .collect::<Vec<_>>(),
    )
}

fn values(buffer: &Buffer, len: usize) -> Vec<f32> {
    buffer
        .read_typed::<u16>(0, len)
        .into_iter()
        .map(|value| bf16::from_bits(value).to_f32())
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    // Match the GR backend: BF16 cancellation needs an absolute bound and a
    // separate aggregate bound, rather than relative error near zero.
    let mut max_error = 0.0_f32;
    let mut error_sum = 0.0;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let error = (actual - expected).abs();
        max_error = max_error.max(error);
        error_sum += error;
    }
    let mean_error = error_sum / actual.len() as f32;
    assert!(
        max_error <= 0.03125 && mean_error <= 0.003125,
        "max_error={max_error}, mean_error={mean_error}",
    );
}
