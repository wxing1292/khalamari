use half::bf16;
use inference_executor_core::gated_residual::core::GatedResidualCore;
use inference_executor_core::gated_residual::reference;

use super::*;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::Stream;

const ACTIVE: ReplayParameterKey = ReplayParameterKey::new("test.gr.num_active_tokens");

#[test]
fn test_read_write_bucketing() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    // Odd dimensions cover every SIMD tail. The last case uses checkpoint geometry.
    for (branches, hidden, rank, dtype, in_place) in [
        (1, 1, 1, Dtype::Float32, false),
        (3, 37, 7, Dtype::Float32, false),
        (3, 37, 7, Dtype::Bfloat16, false),
        (4, 64, 16, Dtype::Float32, false),
        (4, 64, 17, Dtype::Bfloat16, true),
        (4, 2560, 320, Dtype::Float32, false),
        (4, 2560, 320, Dtype::Bfloat16, false),
    ] {
        let fixture = Fixture::new(&device, branches, hidden, rank, dtype, 4);
        let output = if in_place { &fixture.residual } else { &fixture.output };
        let shape = Shape { num_total_tokens: 4 };
        let replay = {
            let mut builder = stream.create_replay_program();
            builder.record(fixture.compute.read(
                shape,
                ReplayU32::Parameter(ACTIVE),
                fixture.read_buffers(),
                fixture.scratch(),
                fixture.weights(),
            ));
            builder.record_with_barrier_before(fixture.compute.write(
                shape,
                ReplayU32::Parameter(ACTIVE),
                fixture.write_buffers(output),
            ));
            builder.build()
        };
        for active in [1, 4, 3, 2] {
            let residual = values(&fixture.residual, active * branches * hidden, dtype);
            let before = values(output, 4 * branches * hidden, dtype);
            stream
                .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(ACTIVE, active as u32))
                .wait();
            fixture.check(&residual, output, active);
            let after = values(output, 4 * branches * hidden, dtype);
            assert_eq!(
                &after[active * branches * hidden..],
                &before[active * branches * hidden..]
            );
        }
    }
}

#[test]
fn test_tiled_read_write_bucketing() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    // Small rows, matrix tile tails, and both row-tile crossovers use the CPU oracle.
    for (hidden, rank, total) in [
        (256, 32, 1),
        (256, 32, 4),
        (96, 24, 4),
        (256, 32, 5),
        (256, 32, 16),
        (256, 32, 17),
        (256, 32, 128),
        (256, 32, 129),
    ] {
        let fixture = Fixture::new(&device, 4, hidden, rank, Dtype::Bfloat16, total);
        let shape = Shape {
            num_total_tokens: total as u32,
        };
        let mut builder = stream.create_replay_program();
        builder.record(fixture.compute.read(
            shape,
            ReplayU32::Parameter(ACTIVE),
            fixture.read_buffers(),
            fixture.scratch(),
            fixture.weights(),
        ));
        builder.record_with_barrier_before(fixture.compute.write(
            shape,
            ReplayU32::Parameter(ACTIVE),
            fixture.write_buffers(&fixture.output),
        ));
        let replay = builder.build();
        for active in [total, 1, total.saturating_sub(1).max(1)] {
            let before = values(&fixture.output, total * 4 * hidden, Dtype::Bfloat16);
            stream
                .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(ACTIVE, active as u32))
                .wait();
            fixture.check(
                &values(&fixture.residual, active * 4 * hidden, Dtype::Bfloat16),
                &fixture.output,
                active,
            );
            let after = values(&fixture.output, total * 4 * hidden, Dtype::Bfloat16);
            assert_eq!(&after[active * 4 * hidden..], &before[active * 4 * hidden..]);
        }
    }
}

#[test]
fn test_tiled_write_read_fusion() {
    use crate::components::ReplayOp;
    use crate::components::ReplayRecorder;

    let device = Device::system_default();
    let stream = Stream::new(&device);
    for total in [1, 4, 5, 17, 129] {
        // Reuse both hidden and gate outputs across reads, as a model layer chain does.
        for in_place in [false, true] {
            for with_gate in [false, true] {
                let plain = Fixture::new(&device, 4, 256, 32, Dtype::Bfloat16, total);
                let fused = Fixture::new(&device, 4, 256, 32, Dtype::Bfloat16, total);
                let shape = Shape {
                    num_total_tokens: total as u32,
                };
                let record = |fixture: &Fixture, fuse| {
                    let output = if in_place { &fixture.residual } else { &fixture.output };
                    let mut recorder = ReplayRecorder::new(stream.create_replay_program());
                    recorder.record(ReplayOp::gated_residual_read(fixture.compute.read(
                        shape,
                        ReplayU32::Parameter(ACTIVE),
                        fixture.read_buffers(),
                        fixture.scratch(),
                        fixture.weights(),
                    )));
                    let write = fixture.compute.write(
                        shape,
                        ReplayU32::Parameter(ACTIVE),
                        WriteBuffers {
                            hidden_state: &fixture.hidden_state,
                            ..fixture.write_buffers(output)
                        },
                    );
                    recorder.record_with_barrier_before(if fuse {
                        ReplayOp::gated_residual_write(write)
                    } else {
                        ReplayOp::opaque(write)
                    });
                    recorder.record_with_barrier_before(ReplayOp::gated_residual_read(fixture.compute.read(
                        shape,
                        ReplayU32::Parameter(ACTIVE),
                        ReadBuffers {
                            residual: output,
                            write_gate: if with_gate {
                                fixture.read_buffers().write_gate
                            } else {
                                None
                            },
                            ..fixture.read_buffers()
                        },
                        fixture.scratch(),
                        fixture.weights(),
                    )));
                    recorder.build()
                };
                let unfused = record(&plain, false);
                let optimized = record(&fused, true);
                assert_eq!(optimized.command_count() + 1, unfused.command_count());
                for active in [total, 1] {
                    let output = if in_place { &fused.residual } else { &fused.output };
                    let before = values(output, total * 1024, Dtype::Bfloat16);
                    let args = ReplayArguments::new().with_u32(ACTIVE, active as u32);
                    stream.submit_replay_with_arguments(&unfused, &args).wait();
                    stream.submit_replay_with_arguments(&optimized, &args).wait();
                    let residual = values(output, total * 1024, Dtype::Bfloat16);
                    let plain_output = if in_place { &plain.residual } else { &plain.output };
                    assert_eq!(residual, values(plain_output, total * 1024, Dtype::Bfloat16));
                    assert_eq!(&residual[active * 1024..], &before[active * 1024..]);
                    let norm = values(&fused.norm, 1024, Dtype::Bfloat16);
                    let down = values(&fused.down, 1024 * 32, Dtype::Bfloat16);
                    let up = values(&fused.up, 1024 * 32, Dtype::Bfloat16);
                    let expected = reference::read(reference::ReadInput {
                        core: fused.core,
                        precision: reference::Precision::Bf16,
                        num_tokens: active,
                        residual: &residual[..active * 1024],
                        weights: reference::ReadWeights {
                            norm: &norm,
                            down: &down,
                            up: &up,
                        },
                    });
                    assert_eq!(
                        values(&fused.hidden_state, total * 256, Dtype::Bfloat16),
                        values(&plain.hidden_state, total * 256, Dtype::Bfloat16)
                    );
                    assert_close(
                        &values(&fused.hidden_state, active * 256, Dtype::Bfloat16),
                        &expected.hidden_state,
                        Dtype::Bfloat16,
                    );
                }
            }
        }
    }
}

#[test]
fn test_write_read_fusion() {
    use crate::components::ReplayOp;
    use crate::components::ReplayRecorder;

    let device = Device::system_default();
    let stream = Stream::new(&device);
    for (branches, hidden, rank, dtype) in [
        (1, 1, 1, Dtype::Float32),
        (3, 37, 7, Dtype::Float32),
        (4, 65, 33, Dtype::Bfloat16),
        (4, 2560, 320, Dtype::Bfloat16),
    ] {
        for in_place in [false, true] {
            let plain = Fixture::new(&device, branches, hidden, rank, dtype, 4);
            let fused = Fixture::new(&device, branches, hidden, rank, dtype, 4);
            // The next read owns its own norm epsilon and projection rank.
            let next_core = GatedResidualCore {
                low_rank_dim: rank + 1,
                eps: 0.25,
                ..plain.core
            };
            let next = Compute::new(
                &device,
                Config {
                    low_rank_dim: next_core.low_rank_dim as u32,
                    eps: next_core.eps,
                    ..plain.compute.config
                },
            );
            let wide = branches * hidden;
            let next_norm = fixture_buffer(&device, wide, 0.00390625, 31, dtype);
            let next_down = fixture_buffer(&device, wide * (rank + 1), 0.03125 / (wide as f32).sqrt(), 17, dtype);
            let next_up = fixture_buffer(
                &device,
                wide * (rank + 1),
                0.03125 / ((rank + 1) as f32).sqrt(),
                37,
                dtype,
            );
            let next_low_rank = Buffer::new_zeroed_elements(&device, 4 * (rank + 1), dtype);
            let weights = ReadWeights {
                norm: &next_norm,
                down: &next_down,
                up: &next_up,
            };
            let shape = Shape { num_total_tokens: 4 };
            let active = ReplayU32::Parameter(ACTIVE);
            let record = |fixture: &Fixture, fuse: bool| {
                let output = if in_place { &fixture.residual } else { &fixture.output };
                let mut recorder = ReplayRecorder::new(stream.create_replay_program());
                recorder.record(ReplayOp::gated_residual_read(fixture.compute.read(
                    shape,
                    active,
                    fixture.read_buffers(),
                    fixture.scratch(),
                    fixture.weights(),
                )));
                let write = fixture.compute.write(shape, active, fixture.write_buffers(output));
                recorder.record_with_barrier_before(if fuse {
                    ReplayOp::gated_residual_write(write)
                } else {
                    ReplayOp::opaque(write)
                });
                recorder.record_with_barrier_before(ReplayOp::gated_residual_read(next.read(
                    shape,
                    active,
                    ReadBuffers {
                        residual: output,
                        hidden_state: &fixture.hidden_state,
                        write_gate: None,
                    },
                    ReadScratch {
                        normalized_residual: &fixture.normalized_residual,
                        projection: &next_low_rank,
                    },
                    weights,
                )));
                recorder.build()
            };
            let unfused_replay = record(&plain, false);
            let fused_replay = record(&fused, true);
            assert_eq!(fused_replay.command_count() + 1, unfused_replay.command_count());
            for active in [1, 4, 3, 2] {
                let output = if in_place { &fused.residual } else { &fused.output };
                let before_output = values(output, 4 * wide, dtype);
                let args = ReplayArguments::new().with_u32(ACTIVE, active as u32);
                stream.submit_replay_with_arguments(&unfused_replay, &args).wait();
                stream.submit_replay_with_arguments(&fused_replay, &args).wait();
                let expected_output = if in_place { &plain.residual } else { &plain.output };
                let result = values(output, 4 * wide, dtype);
                assert_eq!(result, values(expected_output, 4 * wide, dtype));
                assert_eq!(&result[active * wide..], &before_output[active * wide..]);
                let norm = values(&next_norm, wide, dtype);
                let down = values(&next_down, wide * (rank + 1), dtype);
                let up = values(&next_up, wide * (rank + 1), dtype);
                let expected = reference::read(reference::ReadInput {
                    core: next_core,
                    precision: if dtype == Dtype::Float32 {
                        reference::Precision::F32
                    } else {
                        reference::Precision::Bf16
                    },
                    num_tokens: active,
                    residual: &result[..active * wide],
                    weights: reference::ReadWeights {
                        norm: &norm,
                        down: &down,
                        up: &up,
                    },
                });
                let hidden_result = values(&fused.hidden_state, 4 * hidden, dtype);
                assert_eq!(hidden_result, values(&plain.hidden_state, 4 * hidden, dtype));
                assert_close(&hidden_result[..active * hidden], &expected.hidden_state, dtype);
                let norm_result = values(&fused.normalized_residual, 4 * wide, dtype);
                assert_eq!(norm_result, values(&plain.normalized_residual, 4 * wide, dtype));
                assert_close(&norm_result[..active * wide], &expected.normalized_residual, dtype);
            }
        }
    }
}

#[test]
fn test_write_read_separate() {
    use crate::components::ReplayOp;
    use crate::components::ReplayRecorder;

    let device = Device::system_default();
    let stream = Stream::new(&device);
    // Both sequences are valid as separate operations. Fusion would change the
    // active domain or overwrite y while another branch still reads it.
    for reuse_hidden in [false, true] {
        let fixture = Fixture::new(&device, 4, 37, 17, Dtype::Bfloat16, 4);
        let reused = fixture_buffer(&device, 4 * 4 * 37, 0.015625, 0x5A17_920B, Dtype::Bfloat16);
        let shape = Shape { num_total_tokens: 4 };
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());
        recorder.record(ReplayOp::gated_residual_read(fixture.compute.read(
            shape,
            ReplayU32::Fixed(4),
            fixture.read_buffers(),
            fixture.scratch(),
            fixture.weights(),
        )));
        let write_active = if reuse_hidden { 4 } else { 1 };
        recorder.record_with_barrier_before(ReplayOp::gated_residual_write(fixture.compute.write(
            shape,
            ReplayU32::Fixed(write_active),
            WriteBuffers {
                hidden_state: &reused,
                ..fixture.write_buffers(&fixture.output)
            },
        )));
        let scratch = ReadScratch {
            normalized_residual: if reuse_hidden {
                &reused
            } else {
                &fixture.normalized_residual
            },
            projection: &fixture.projection,
        };
        recorder.record_with_barrier_before(ReplayOp::gated_residual_read(fixture.compute.read(
            shape,
            ReplayU32::Fixed(4),
            ReadBuffers {
                residual: &fixture.output,
                hidden_state: &fixture.hidden_state,
                write_gate: None,
            },
            scratch,
            fixture.weights(),
        )));
        let replay = recorder.build();
        assert_eq!(replay.command_count(), 7);
        let norm = values(&fixture.norm, 4 * 37, Dtype::Bfloat16);
        let down = values(&fixture.down, 4 * 37 * 17, Dtype::Bfloat16);
        let up = values(&fixture.up, 4 * 37 * 17, Dtype::Bfloat16);
        let weights = reference::ReadWeights {
            norm: &norm,
            down: &down,
            up: &up,
        };
        let input = values(&fixture.residual, write_active as usize * 4 * 37, Dtype::Bfloat16);
        let first = reference::read(reference::ReadInput {
            core: fixture.core,
            precision: reference::Precision::Bf16,
            num_tokens: write_active as usize,
            residual: &input,
            weights,
        });
        let hidden = values(&reused, write_active as usize * 37, Dtype::Bfloat16);
        let write_weight = values(&fixture.write_weight, 4 * 4 * 37, Dtype::Bfloat16);
        let mut expected_output = reference::write(reference::WriteInput {
            core: fixture.core,
            precision: reference::Precision::Bf16,
            num_tokens: write_active as usize,
            residual: &input,
            normalized_residual: &first.normalized_residual,
            hidden_state: &hidden,
            weight: &write_weight,
        });
        expected_output.resize(4 * 4 * 37, 0.0);
        let expected = reference::read(reference::ReadInput {
            core: fixture.core,
            precision: reference::Precision::Bf16,
            num_tokens: 4,
            residual: &expected_output,
            weights,
        });
        stream.submit_replay(&replay).wait();
        assert_close(
            &values(&fixture.output, 4 * 4 * 37, Dtype::Bfloat16),
            &expected_output,
            Dtype::Bfloat16,
        );
        assert_close(
            &values(&fixture.hidden_state, 4 * 37, Dtype::Bfloat16),
            &expected.hidden_state,
            Dtype::Bfloat16,
        );
    }
}

#[test]
fn test_small_read_reuses_dead_residual() {
    use crate::components::ReplayOp;
    use crate::components::ReplayRecorder;

    let device = Device::system_default();
    let stream = Stream::new(&device);
    // After a separate write, its old residual is dead. Either next-read scratch
    // buffer may reuse it. A fused down must not overwrite it while peers read it.
    for reuse_projection in [false, true] {
        let fixture = Fixture::new(&device, 4, 256, 32, Dtype::Bfloat16, 4);
        let residual = values(&fixture.residual, 4096, Dtype::Bfloat16);
        let norm = values(&fixture.norm, 1024, Dtype::Bfloat16);
        let down = values(&fixture.down, 1024 * 32, Dtype::Bfloat16);
        let up = values(&fixture.up, 1024 * 32, Dtype::Bfloat16);
        let write = values(&fixture.write_weight, 4 * 1024, Dtype::Bfloat16);
        let module_output = values(&fixture.module_output, 4 * 256, Dtype::Bfloat16);
        let weights = reference::ReadWeights {
            norm: &norm,
            down: &down,
            up: &up,
        };
        let first = reference::read(reference::ReadInput {
            core: fixture.core,
            precision: reference::Precision::Bf16,
            num_tokens: 4,
            residual: &residual,
            weights,
        });
        let written = reference::write(reference::WriteInput {
            core: fixture.core,
            precision: reference::Precision::Bf16,
            num_tokens: 4,
            residual: &residual,
            normalized_residual: &first.normalized_residual,
            hidden_state: &module_output,
            weight: &write,
        });
        let expected = reference::read(reference::ReadInput {
            core: fixture.core,
            precision: reference::Precision::Bf16,
            num_tokens: 4,
            residual: &written,
            weights,
        });
        let shape = Shape { num_total_tokens: 4 };
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());
        recorder.record(ReplayOp::gated_residual_read(fixture.compute.read(
            shape,
            ReplayU32::Fixed(4),
            fixture.read_buffers(),
            fixture.scratch(),
            fixture.weights(),
        )));
        recorder.record_with_barrier_before(ReplayOp::gated_residual_write(fixture.compute.write(
            shape,
            ReplayU32::Fixed(4),
            fixture.write_buffers(&fixture.output),
        )));
        recorder.record_with_barrier_before(ReplayOp::gated_residual_read(fixture.compute.read(
            shape,
            ReplayU32::Fixed(4),
            ReadBuffers {
                residual: &fixture.output,
                write_gate: None,
                ..fixture.read_buffers()
            },
            ReadScratch {
                normalized_residual: if reuse_projection {
                    &fixture.normalized_residual
                } else {
                    &fixture.residual
                },
                projection: if reuse_projection {
                    &fixture.residual
                } else {
                    &fixture.projection
                },
            },
            fixture.weights(),
        )));
        stream.submit_replay(&recorder.build()).wait();
        assert_close(
            &values(&fixture.output, 4096, Dtype::Bfloat16),
            &written,
            Dtype::Bfloat16,
        );
        assert_close(
            &values(&fixture.hidden_state, 1024, Dtype::Bfloat16),
            &expected.hidden_state,
            Dtype::Bfloat16,
        );
    }
}

#[test]
fn test_scratch_reuse_before_write() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let fixture = Fixture::new(&device, 4, 37, 17, Dtype::Bfloat16, 2);
    let shape = Shape { num_total_tokens: 2 };
    let active = ReplayU32::Fixed(2);
    let mut expected = Vec::new();
    for reuse in [false, true] {
        let mut builder = stream.create_replay_program();
        builder.record(fixture.compute.read(
            shape,
            active,
            fixture.read_buffers(),
            fixture.scratch(),
            fixture.weights(),
        ));
        if reuse {
            // A second read overwrites N and z, but keeps the first read's write gate.
            builder.record_with_barrier_before(fixture.compute.read(
                shape,
                active,
                ReadBuffers {
                    residual: &fixture.output,
                    hidden_state: &fixture.hidden_state,
                    write_gate: None,
                },
                fixture.scratch(),
                fixture.weights(),
            ));
        }
        builder.record_with_barrier_before(fixture.compute.write(
            shape,
            active,
            fixture.write_buffers(&fixture.output),
        ));
        stream.submit_replay(&builder.build()).wait();
        let actual = values(&fixture.output, 2 * 4 * 37, Dtype::Bfloat16);
        if reuse {
            assert_eq!(actual, expected);
        } else {
            expected = actual;
        }
    }
}
struct Fixture {
    core: GatedResidualCore,
    dtype: Dtype,
    compute: Compute,
    residual: Buffer,
    hidden_state: Buffer,
    normalized_residual: Buffer,
    projection: Buffer,
    module_output: Buffer,
    output: Buffer,
    norm: Buffer,
    down: Buffer,
    up: Buffer,
    write_weight: Buffer,
    gate: Buffer,
}

impl Fixture {
    fn new(device: &Device, branches: usize, hidden: usize, rank: usize, dtype: Dtype, tokens: usize) -> Self {
        let wide = branches * hidden;
        let core = GatedResidualCore {
            num_branches: branches,
            hidden_dim: hidden,
            low_rank_dim: rank,
            eps: 1.0e-6,
        };
        let config = Config {
            num_branches: branches as u32,
            hidden_dim: hidden as u32,
            low_rank_dim: rank as u32,
            eps: core.eps,
            dtype,
        };
        Self {
            core,
            dtype,
            compute: Compute::new(device, config),
            residual: fixture_buffer(device, tokens * wide, 0.03125, 11, dtype),
            hidden_state: Buffer::new_zeroed_elements(device, tokens * hidden, dtype),
            normalized_residual: Buffer::new_zeroed_elements(device, tokens * wide, dtype),
            projection: Buffer::new_zeroed(device, config.projection_scratch_bytes(tokens)),
            module_output: fixture_buffer(device, tokens * hidden, 0.015625, 23, dtype),
            output: Buffer::new_zeroed_elements(device, tokens * wide, dtype),
            norm: fixture_buffer(device, wide, 0.0078125, 7, dtype),
            down: fixture_buffer(device, wide * rank, 0.125 / (wide as f32).sqrt(), 13, dtype),
            up: fixture_buffer(device, wide * rank, 0.125 / (rank as f32).sqrt(), 19, dtype),
            gate: Buffer::new_zeroed_elements(device, tokens * branches, dtype),
            write_weight: fixture_buffer(device, branches * wide, 0.125 / (wide as f32).sqrt(), 29, dtype),
        }
    }

    fn read_buffers(&self) -> ReadBuffers<'_> {
        ReadBuffers {
            residual: &self.residual,
            hidden_state: &self.hidden_state,
            write_gate: Some(WriteGate {
                weight: &self.write_weight,
                output: &self.gate,
            }),
        }
    }

    fn scratch(&self) -> ReadScratch<'_> {
        ReadScratch {
            normalized_residual: &self.normalized_residual,
            projection: &self.projection,
        }
    }

    fn weights(&self) -> ReadWeights<'_> {
        ReadWeights {
            norm: &self.norm,
            down: &self.down,
            up: &self.up,
        }
    }

    fn write_buffers<'a>(&'a self, output: &'a Buffer) -> WriteBuffers<'a> {
        WriteBuffers {
            residual: &self.residual,
            gate: &self.gate,
            hidden_state: &self.module_output,
            output,
        }
    }

    fn check(&self, residual: &[f32], output_buffer: &Buffer, active: usize) {
        let core = self.core;
        let wide = core.residual_dim();
        let norm = values(&self.norm, wide, self.dtype);
        let down = values(&self.down, wide * core.low_rank_dim, self.dtype);
        let up = values(&self.up, wide * core.low_rank_dim, self.dtype);
        let write_weight = values(&self.write_weight, wide * core.num_branches, self.dtype);
        let module_output = values(&self.module_output, active * core.hidden_dim, self.dtype);
        let precision = if self.dtype == Dtype::Float32 {
            reference::Precision::F32
        } else {
            reference::Precision::Bf16
        };
        let expected = reference::read(reference::ReadInput {
            core,
            precision,
            num_tokens: active,
            residual,
            weights: reference::ReadWeights {
                norm: &norm,
                down: &down,
                up: &up,
            },
        });
        assert_close(
            &values(&self.normalized_residual, active * wide, self.dtype),
            &expected.normalized_residual,
            self.dtype,
        );
        assert_close(
            &values(&self.hidden_state, active * core.hidden_dim, self.dtype),
            &expected.hidden_state,
            self.dtype,
        );
        let output = reference::write(reference::WriteInput {
            core,
            precision,
            num_tokens: active,
            residual,
            normalized_residual: &expected.normalized_residual,
            hidden_state: &module_output,
            weight: &write_weight,
        });
        assert_close(&values(output_buffer, active * wide, self.dtype), &output, self.dtype);
    }
}

fn fixture_buffer(device: &Device, len: usize, scale: f32, mut seed: u32, dtype: Dtype) -> Buffer {
    let values = (0..len)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (((seed >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0) * 63.0 * scale
        })
        .collect::<Vec<_>>();
    match dtype {
        Dtype::Float32 => Buffer::from_slice(device, &values),
        Dtype::Bfloat16 => {
            Buffer::from_slice(
                device,
                &values
                    .iter()
                    .map(|&value| bf16::from_f32(value).to_bits())
                    .collect::<Vec<_>>(),
            )
        },
        _ => unreachable!(),
    }
}

fn values(buffer: &Buffer, len: usize, dtype: Dtype) -> Vec<f32> {
    match dtype {
        Dtype::Float32 => buffer.read_typed(0, len),
        Dtype::Bfloat16 => {
            buffer
                .read_typed::<u16>(0, len)
                .into_iter()
                .map(|value| bf16::from_bits(value).to_f32())
                .collect()
        },
        _ => unreachable!(),
    }
}

fn assert_close(actual: &[f32], expected: &[f32], dtype: Dtype) {
    assert_eq!(actual.len(), expected.len());
    // BF16 normalization and projection rounding can move a small mixed output
    // across zero. Bound both worst-case and aggregate error, as in GDN tests.
    let tolerance = if dtype == Dtype::Float32 { 2.0e-5 } else { 0.03125 };
    let mut max_error = 0.0_f32;
    let mut error_sum = 0.0;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let error = (actual - expected).abs();
        max_error = max_error.max(error);
        error_sum += error;
    }
    let mean_error = error_sum / actual.len() as f32;
    assert!(
        max_error <= tolerance && mean_error <= tolerance * 0.1,
        "max_error={max_error}, mean_error={mean_error}, tolerance={tolerance}"
    );
}
