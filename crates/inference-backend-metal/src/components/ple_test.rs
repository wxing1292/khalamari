use half::bf16;
use inference_executor_core::ple::core::PLECore;
use inference_executor_core::ple::reference;

use super::*;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::Stream;

const TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.ple.tokens");
const REQUESTS: ReplayParameterKey = ReplayParameterKey::new("test.ple.requests");

#[test]
fn test_apply_commit_replay() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    for (channels, hidden, fixed) in [(3, 37, false), (4, 2560, false), (4, 32, true)] {
        let core = PLECore {
            num_channels: channels,
            hidden_dim: hidden,
            embedding_dim: 128,
            conv_kernel_size: 4,
            conv_dilation: 3,
            norm_eps: 1e-6,
        };
        let config = Config {
            num_channels: channels,
            hidden_dim: hidden,
            conv_kernel_size: 4,
            conv_dilation: 3,
            norm_eps: 1e-6,
        };
        let compute = Compute::new(&device, config);
        let wide = core.residual_dim() as usize;
        let tail = core.state_len() as usize;
        let mut seed = 0x8173_32A1_u32;
        let mut fixture = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    if fixed {
                        0.0
                    } else {
                        bf16::from_f32(((seed % 127) as i32 - 63) as f32 / 128.0).to_f32()
                    }
                })
                .collect()
        };
        let residual = fixture(24 * wide);
        let keys = fixture(24 * wide);
        let values = fixture(24 * hidden as usize);
        let key_norm = fixture(wide);
        let query_norm = fixture(wide);
        let conv_norm = fixture(wide);
        let conv = fixture(wide * 4);
        let initial = fixture(4 * tail * wide);
        let r = buffer(&device, &residual);
        let k = buffer(&device, &keys);
        let v = buffer(&device, &values);
        let kn = buffer(&device, &key_norm);
        let qn = buffer(&device, &query_norm);
        let cn = buffer(&device, &conv_norm);
        let cw = buffer(&device, &conv);
        let states = buffer(&device, &initial);
        let nk = Buffer::new_zeroed(&device, 24 * wide * 2);
        let nq = Buffer::new_zeroed(&device, 24 * wide * 2);
        let gated = Buffer::new_zeroed(&device, 24 * wide * 2);
        let normalized = Buffer::new_zeroed(&device, 24 * wide * 2);
        let output = Buffer::new_zeroed(&device, 24 * wide * 2);
        let meta = Buffer::from_slice(
            &device,
            &(0..24)
                .flat_map(|t| if t < 3 { [0_u32, 0] } else { [3, 1] })
                .collect::<Vec<_>>(),
        );
        let requests = Buffer::from_slice(&device, &[0_u32, 0, 2, 3, 1, 3]);
        let accepted = Buffer::from_slice(&device, &[3_u32, 14]);
        let mut b = stream.create_replay_program();
        b.record(compute.norm(
            24,
            ReplayU32::Parameter(TOKENS),
            NormBuffers {
                input: &k,
                weight: &kn,
                output: &nk,
            },
        ));
        b.record(compute.norm(
            24,
            ReplayU32::Parameter(TOKENS),
            NormBuffers {
                input: &r,
                weight: &qn,
                output: &nq,
            },
        ));
        b.record_with_barrier_before(compute.gate(
            24,
            ReplayU32::Parameter(TOKENS),
            GateBuffers {
                key: &nk,
                query: &nq,
                value: &v,
                output: &gated,
            },
        ));
        b.record_with_barrier_before(compute.norm(
            24,
            ReplayU32::Parameter(TOKENS),
            NormBuffers {
                input: &gated,
                weight: &cn,
                output: &normalized,
            },
        ));
        b.record_with_barrier_before(compute.conv(
            24,
            ReplayU32::Parameter(TOKENS),
            4,
            ConvBuffers {
                normalized: &normalized,
                gated: &gated,
                residual: &r,
                weight: &cw,
                tokens: &meta,
                states: &states,
                output: &output,
            },
        ));
        let replay = b.build();
        let mut b = stream.create_replay_program();
        b.record(compute.commit(
            24,
            2,
            ReplayU32::Parameter(REQUESTS),
            4,
            CommitBuffers {
                normalized: &normalized,
                requests: &requests,
                accepted_tokens: &accepted,
                states: &states,
            },
        ));
        let commit = b.build();
        for (active, active_requests) in [(17, 2), (3, 1), (17, 2)] {
            states.write_typed(
                0,
                &initial.iter().map(|&v| bf16::from_f32(v).to_bits()).collect::<Vec<_>>(),
            );
            output.write_typed(0, &vec![bf16::from_f32(-99.0).to_bits(); 24 * wide]);
            stream
                .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(TOKENS, active))
                .wait();
            let actual = output.read_typed::<u16>(0, 24 * wide);
            for (request, (start, count)) in [(0_usize, 3_usize), (3, 14)]
                .into_iter()
                .take(active_requests)
                .enumerate()
            {
                let result = reference::apply(reference::Input {
                    core,
                    num_tokens: count,
                    residual: &residual[start * wide..(start + count) * wide],
                    keys: &keys[start * wide..(start + count) * wide],
                    values: &values[start * hidden as usize..(start + count) * hidden as usize],
                    state: &initial[request * tail * wide..(request + 1) * tail * wide],
                    weights: reference::Weights {
                        key_norm: &key_norm,
                        query_norm: &query_norm,
                        conv_norm: &conv_norm,
                        conv: &conv,
                    },
                });
                for (i, &expected) in result.residual.iter().enumerate() {
                    let got = bf16::from_bits(actual[start * wide + i]).to_f32();
                    assert!(
                        (got - expected).abs() <= 0.03125,
                        "H={hidden}, token={start}, index={i}: {got} != {expected}"
                    );
                }
                for (buffer, expected) in [(&gated, &result.gated), (&normalized, &result.normalized)] {
                    let actual = buffer.read_typed::<u16>(start * wide, count * wide);
                    for (&a, &e) in actual.iter().zip(expected) {
                        assert!((bf16::from_bits(a).to_f32() - e).abs() <= 0.03125);
                    }
                }
            }
            assert!(
                actual[active as usize * wide..]
                    .iter()
                    .all(|&v| v == bf16::from_f32(-99.0).to_bits())
            );
            for counts in [[0_u32, 14], [2, 1], [3, 14]] {
                accepted.write_typed(0, &counts);
                stream
                    .submit_replay_with_arguments(
                        &commit,
                        &ReplayArguments::new().with_u32(REQUESTS, active_requests as u32),
                    )
                    .wait();
                let after = states.read_typed::<u16>(0, initial.len());
                assert_eq!(
                    &after[..2 * tail * wide],
                    &initial[..2 * tail * wide]
                        .iter()
                        .map(|&v| bf16::from_f32(v).to_bits())
                        .collect::<Vec<_>>()
                );
                let gpu_normalized = normalized.read_typed::<u16>(0, 17 * wide);
                for request in 0..active_requests {
                    let start = if request == 0 { 0 } else { 3 };
                    let n = counts[request] as usize;
                    // State copies are exact. Numerical reference parity belongs to the normalized producer.
                    let history = [
                        initial[request * tail * wide..(request + 1) * tail * wide]
                            .iter()
                            .map(|&v| bf16::from_f32(v).to_bits())
                            .collect::<Vec<_>>(),
                        gpu_normalized[start * wide..(start + n) * wide].to_vec(),
                    ]
                    .concat();
                    assert_eq!(
                        &after[(request + 2) * tail * wide..(request + 3) * tail * wide],
                        &history[history.len() - tail * wide..]
                    );
                }
                if active_requests == 1 {
                    assert_eq!(
                        &after[3 * tail * wide..],
                        &initial[3 * tail * wide..]
                            .iter()
                            .map(|&v| bf16::from_f32(v).to_bits())
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}
fn buffer(device: &Device, values: &[f32]) -> Buffer {
    Buffer::from_slice(
        device,
        &values.iter().map(|&v| bf16::from_f32(v).to_bits()).collect::<Vec<_>>(),
    )
}
