use std::hint::black_box;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use half::bf16;
use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::gated_residual;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;

const NUM_INVOCATIONS: usize = 32;
const BRANCHES: u32 = 4;
const HIDDEN: u32 = 2560;
const RANK: u32 = 320;

fn bench_gated_residual(c: &mut Criterion) {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = gated_residual::Config {
        num_branches: BRANCHES,
        hidden_dim: HIDDEN,
        low_rank_dim: RANK,
        eps: 1.0e-6,
        dtype: Dtype::Bfloat16,
    };
    let compute = gated_residual::Compute::new(&device, config);
    let wide = (BRANCHES * HIDDEN) as usize;
    let norm = fixture_buffer(&device, wide, 0.002, 11);
    let down = fixture_buffer(&device, wide * RANK as usize, 0.0005, 13);
    let up = fixture_buffer(&device, wide * RANK as usize, 0.002, 17);
    let write_weight = fixture_buffer(&device, wide * BRANCHES as usize, 0.0005, 23);
    let weights = gated_residual::ReadWeights {
        norm: &norm,
        down: &down,
        up: &up,
    };
    let mut group = c.benchmark_group("metal/gr");
    for tokens in [1, 4, 16, 64] {
        let shape = gated_residual::Shape {
            num_total_tokens: tokens,
        };
        let active = ReplayU32::Fixed(tokens);
        let residual = fixture_buffer(&device, tokens as usize * wide, 0.025, 7);
        let hidden = Buffer::new_zeroed_elements(&device, tokens as usize * HIDDEN as usize, Dtype::Bfloat16);
        let normalized = Buffer::new_zeroed_elements(&device, tokens as usize * wide, Dtype::Bfloat16);
        let projection = Buffer::new_zeroed(&device, config.projection_scratch_bytes(tokens as usize));
        let output = Buffer::new_zeroed_elements(&device, tokens as usize * wide, Dtype::Bfloat16);
        let gate = Buffer::new_zeroed_elements(&device, tokens as usize * BRANCHES as usize, Dtype::Bfloat16);
        let read_buffers = gated_residual::ReadBuffers {
            residual: &residual,
            hidden_state: &hidden,
            write_gate: Some(gated_residual::WriteGate {
                weight: &write_weight,
                output: &gate,
            }),
        };
        let scratch = gated_residual::ReadScratch {
            normalized_residual: &normalized,
            projection: &projection,
        };
        let write_buffers = gated_residual::WriteBuffers {
            residual: &residual,
            gate: &gate,
            hidden_state: &hidden,
            output: &output,
        };

        // Initialize write inputs through the production read outside the measurements.
        let mut builder = stream.create_replay_program();
        builder.record(compute.read(shape, active, read_buffers, scratch, weights));
        stream.submit_replay(&builder.build()).wait();
        for operation in ["read", "write", "read_write"] {
            let mut builder = stream.create_replay_program();
            for _ in 0..NUM_INVOCATIONS {
                if operation != "write" {
                    builder.record_with_barrier_before(compute.read(shape, active, read_buffers, scratch, weights));
                }
                if operation != "read" {
                    builder.record_with_barrier_before(compute.write(shape, active, write_buffers));
                }
            }
            let replay = builder.build();
            group.throughput(Throughput::Elements(tokens as u64 * NUM_INVOCATIONS as u64));
            group.bench_function(
                format!(
                    "{operation}/replay{NUM_INVOCATIONS}/tokens{tokens}/branches{BRANCHES}/hidden{HIDDEN}/rank{RANK}"
                ),
                |b| {
                    b.iter(|| {
                        stream.submit_replay(&replay).wait();
                        black_box((&hidden, &output));
                    });
                },
            );
        }
        let mut unfused_commands = 0;
        for fuse in [false, true] {
            let mut recorder = ReplayRecorder::new(stream.create_replay_program());
            for index in 0..NUM_INVOCATIONS {
                let input = if index == 0 { &residual } else { &output };
                let read = compute.read(
                    shape,
                    active,
                    gated_residual::ReadBuffers {
                        residual: input,
                        ..read_buffers
                    },
                    scratch,
                    weights,
                );
                recorder.record_with_barrier_before(if fuse {
                    ReplayOp::gated_residual_read(read)
                } else {
                    ReplayOp::opaque(read)
                });
                let write = compute.write(
                    shape,
                    active,
                    gated_residual::WriteBuffers {
                        residual: input,
                        ..write_buffers
                    },
                );
                recorder.record_with_barrier_before(if fuse {
                    ReplayOp::gated_residual_write(write)
                } else {
                    ReplayOp::opaque(write)
                });
            }
            let read = compute.read(
                shape,
                active,
                gated_residual::ReadBuffers {
                    residual: &output,
                    write_gate: None,
                    ..read_buffers
                },
                scratch,
                weights,
            );
            recorder.record_with_barrier_before(if fuse {
                ReplayOp::gated_residual_read(read)
            } else {
                ReplayOp::opaque(read)
            });
            let replay = recorder.build();
            if fuse {
                assert_eq!(replay.command_count() + NUM_INVOCATIONS, unfused_commands);
            } else {
                unfused_commands = replay.command_count();
            }
            let operation = if fuse { "chain_fused" } else { "chain_unfused" };
            group.bench_function(
                format!(
                    "{operation}/replay{NUM_INVOCATIONS}/tokens{tokens}/branches{BRANCHES}/hidden{HIDDEN}/rank{RANK}"
                ),
                |b| {
                    b.iter(|| {
                        stream.submit_replay(&replay).wait();
                        black_box((&hidden, &output));
                    })
                },
            );
        }
    }
    group.finish();
}

fn fixture_buffer(device: &Device, len: usize, scale: f32, seed: usize) -> Buffer {
    Buffer::from_slice(
        device,
        &(0..len)
            .map(|index| bf16::from_f32((((index * seed + index / 11) % 127) as f32 - 63.0) * scale).to_bits())
            .collect::<Vec<_>>(),
    )
}

criterion_group!(benches, bench_gated_residual);
criterion_main!(benches);
