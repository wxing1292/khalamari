use half::bf16;
use half::f16;
use inference_executor_core::replay::ReplayBucketPolicy;

use super::*;
use crate::metal::ReplayArguments;
use crate::metal::Stream;
use crate::test_support::ReplayTestCache;

const NUM_ACTIVE_ROWS: ReplayParameterKey = ReplayParameterKey::new("test.affine.num_active_rows");

fn adaptive_config(n: i32, k: i32, dtype: Dtype) -> Config {
    Config::same_dtype(n, k, 64, 4, dtype)
}

#[test]
fn test_adaptive_large_vocabulary_qmm_crossover() {
    assert_eq!(
        Selector::key(adaptive_config(151_936, 2048, Dtype::Bfloat16), 4, Epilogue::Identity),
        KernelKind::QmvBn8Bk32
    );
    assert_eq!(
        Selector::key(adaptive_config(151_936, 2048, Dtype::Bfloat16), 5, Epilogue::Identity),
        KernelKind::QmmBm8Bn32
    );
    assert_eq!(
        Selector::key(adaptive_config(151_936, 5120, Dtype::Bfloat16), 5, Epilogue::Identity),
        KernelKind::QmvBn8Bk32
    );
    assert_eq!(
        Selector::key(adaptive_config(151_936, 5120, Dtype::Bfloat16), 6, Epilogue::Identity),
        KernelKind::QmmBm8Bn32
    );
}

#[test]
fn test_adaptive_qmm_tile_crossover() {
    assert_eq!(
        Selector::key(adaptive_config(151_936, 5120, Dtype::Bfloat16), 16, Epilogue::Identity),
        KernelKind::QmmBm16Bn32
    );
    assert_eq!(
        Selector::key(adaptive_config(151_936, 5120, Dtype::Bfloat16), 17, Epilogue::Identity),
        KernelKind::QmmBm32Bn64
    );
}

#[test]
fn test_adaptive_dense_projection_crossover() {
    let large_projection = adaptive_config(34_816, 5120, Dtype::Bfloat16);
    assert_eq!(
        Selector::key(large_projection, 5, Epilogue::Identity),
        KernelKind::QmvBn8Bk32
    );
    assert_eq!(
        Selector::key(large_projection, 6, Epilogue::Identity),
        KernelKind::QmmBm8Bn32
    );
    assert_eq!(
        Selector::key(large_projection, 8, Epilogue::Identity),
        KernelKind::QmmBm8Bn32
    );
    assert_eq!(
        Selector::key(large_projection, 9, Epilogue::Identity),
        KernelKind::QmmBm16Bn32
    );

    let common_projection = adaptive_config(1024, 2048, Dtype::Bfloat16);
    assert_eq!(
        Selector::key(common_projection, 8, Epilogue::Identity),
        KernelKind::QmvBn8Bk32
    );
    assert_eq!(
        Selector::key(common_projection, 18, Epilogue::Identity),
        KernelKind::QmmBm32Bn32
    );
}

#[test]
fn test_adaptive_topology_boundaries_follow_selector() {
    let cases = [
        (adaptive_config(151_936, 2048, Dtype::Bfloat16), &[5, 9, 17][..], false),
        (adaptive_config(151_936, 5120, Dtype::Bfloat16), &[6, 9, 17][..], true),
        (adaptive_config(248_320, 2048, Dtype::Bfloat16), &[5, 9, 17][..], false),
        (adaptive_config(248_320, 5120, Dtype::Bfloat16), &[6, 9, 17][..], true),
        (adaptive_config(34_816, 5120, Dtype::Bfloat16), &[6, 9, 17][..], true),
        (adaptive_config(4096, 4096, Dtype::Bfloat16), &[12, 17][..], true),
        (adaptive_config(1024, 2048, Dtype::Bfloat16), &[18][..], false),
    ];

    for epilogue in [Epilogue::Identity, Epilogue::SwiGLU] {
        for (config, expected, large_projection) in cases {
            let mut expected = expected.to_vec();
            if large_projection {
                expected.push(2048);
            }
            assert_eq!(
                &*adaptive_topology_boundaries(config, epilogue),
                expected,
                "config={config:?}"
            );

            let boundaries = adaptive_topology_boundaries(config, epilogue);
            let policy = ReplayBucketPolicy::with_topology_boundaries(4096, &boundaries);
            for num_active_rows in 1..=4096 {
                let num_total_rows = policy.capacity(num_active_rows);
                assert_eq!(
                    Selector::key(config, num_active_rows as i32, epilogue),
                    Selector::key(config, num_total_rows as i32, epilogue),
                    "config={config:?} num_active_rows={num_active_rows} num_total_rows={num_total_rows}"
                );
            }
        }
    }
}

#[test]
fn test_large_projection_selection_preserves_other_contracts() {
    let config = adaptive_config(16_480, 5120, Dtype::Bfloat16);
    for rows in [128, 256, 1024, 2047] {
        assert_eq!(Selector::key(config, rows, Epilogue::Identity), KernelKind::QmmBm32Bn64);
    }
    for rows in [2048, 4096, 16384] {
        assert_eq!(Selector::key(config, rows, Epilogue::Identity), KernelKind::QmmBm64Bn64);
        assert_eq!(Selector::key(config, rows, Epilogue::SwiGLU), KernelKind::QmmBm64Bn32);
    }
    for other in [
        Config::same_dtype(16_480, 5120, 64, 4, Dtype::Float16),
        Config { bits: 8, ..config },
        Config {
            group_size: 128,
            ..config
        },
        Config {
            input_dtype: Dtype::Float32,
            ..config
        },
        Config {
            output_dtype: Dtype::Float32,
            ..config
        },
        Config {
            scale_bias_dtype: Dtype::Float32,
            ..config
        },
        Config { n: 48, ..config },
        Config { k: 2048, ..config },
    ] {
        assert_eq!(Selector::key(other, 4096, Epilogue::Identity), KernelKind::QmmBm32Bn32);
    }
}

#[test]
fn test_large_qmm_replay_preserves_partial_tiles_and_inactive_rows() {
    for n in [1, 31, 32, 33, 63, 64, 65, 96] {
        assert_bucketed_replay_matches_reference(
            Config::same_dtype(n, 128, 64, 4, Dtype::Bfloat16),
            KernelKind::QmmBm64Bn32,
            128,
        );
    }
    for config in [
        Config::same_dtype(65, 64, 32, 8, Dtype::Float32),
        Config::same_dtype(65, 128, 128, 4, Dtype::Float16),
        Config {
            output_dtype: Dtype::Float32,
            ..Config::same_dtype(65, 64, 64, 4, Dtype::Bfloat16)
        },
    ] {
        assert_bucketed_replay_matches_reference(config, KernelKind::QmmBm64Bn32, 128);
    }
}

#[test]
fn test_bucketed_qmv_variants_match_reference_across_active_counts() {
    let same_qmv = Config::same_dtype(4, 32, 32, 8, Dtype::Float32);
    let same_qmv_fast = Config::same_dtype(8, 512, 64, 8, Dtype::Float32);
    let same_qmv_quad = Config::same_dtype(9, 64, 64, 8, Dtype::Float32);
    let mixed_qmv = Config {
        n: 4,
        k: 32,
        group_size: 32,
        bits: 8,
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Float32,
        scale_bias_dtype: Dtype::Float32,
    };
    let mixed_qmv_fast = Config {
        n: 8,
        k: 512,
        group_size: 64,
        bits: 8,
        input_dtype: Dtype::Float16,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Float32,
    };

    for (config, kind) in [
        (same_qmv, KernelKind::QmvBn8Bk32),
        (same_qmv_fast, KernelKind::QmvBn8Bk32),
        (same_qmv_quad, KernelKind::QmvQuadBn64),
        (mixed_qmv, KernelKind::QmvBn8Bk32),
        (mixed_qmv_fast, KernelKind::QmvBn8Bk32),
    ] {
        assert_bucketed_replay_matches_reference(config, kind, 4);
    }
}

#[test]
fn test_bucketed_qmm_variants_match_reference_across_active_counts() {
    let same = Config::same_dtype(32, 32, 32, 8, Dtype::Float32);
    let same_unaligned = Config::same_dtype(1, 32, 32, 8, Dtype::Float32);
    let same_q4_bf16 = Config::same_dtype(32, 64, 64, 4, Dtype::Bfloat16);
    let mixed_unaligned = Config {
        n: 3,
        k: 32,
        group_size: 32,
        bits: 8,
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Float32,
        scale_bias_dtype: Dtype::Float32,
    };

    for (config, kind, num_total_rows) in [
        (same, KernelKind::QmmBm8Bn32, 16),
        (same, KernelKind::QmmBm16Bn32, 32),
        (same, KernelKind::QmmBm32Bn32, 64),
        (same_q4_bf16, KernelKind::QmmBm16Bn32, 32),
        (same_q4_bf16, KernelKind::QmmBm32Bn32, 64),
        (same_unaligned, KernelKind::QmmBm32Bn32, 64),
        (mixed_unaligned, KernelKind::QmmBm8Bn32, 16),
        (mixed_unaligned, KernelKind::QmmBm16Bn32, 32),
        (mixed_unaligned, KernelKind::QmmBm32Bn32, 64),
    ] {
        assert_bucketed_replay_matches_reference(config, kind, num_total_rows);
    }
}

fn assert_bucketed_replay_matches_reference(config: Config, kind: KernelKind, num_total_rows: i32) {
    assert_epilogue_replay_matches_reference(config, kind, num_total_rows, Epilogue::Identity);
}

#[test]
fn test_steel_wide_tiles_preserve_replay_rows_and_projection_rounding() {
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        for mixed in [false, true] {
            let mut config = Config::same_dtype(130, 128, 64, 4, dtype);
            if mixed {
                config.scale_bias_dtype = Dtype::Float32;
            }
            for kind in [KernelKind::QmmBm32Bn64, KernelKind::QmmBm64Bn64] {
                assert_epilogue_replay_matches_reference(config, kind, 128, Epilogue::Identity);
            }
            for kind in [KernelKind::QmmBm32Bn32, KernelKind::QmmBm64Bn32] {
                assert_epilogue_replay_matches_reference(config, kind, 128, Epilogue::SwiGLU);
            }
        }
    }
}

fn assert_epilogue_replay_matches_reference(config: Config, kind: KernelKind, num_total_rows: i32, epilogue: Epilogue) {
    assert!(matches!(config.bits, 4 | 8));
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let input_source = fixture_values(num_total_rows as usize * config.k as usize, 0.00390625);
    let input_values = round_values_to_dtype(&input_source, config.input_dtype);
    let num_weight_values = config.n as usize * config.k as usize;
    let weight_values = if config.bits == 4 {
        fixture_q4_values(num_weight_values)
    } else {
        fixture_weight_bytes(num_weight_values)
    };
    let num_affine_values = config.n as usize * (config.k / config.group_size) as usize;
    let scales = round_values_to_dtype(&fixture_values(num_affine_values, 0.001953125), config.scale_bias_dtype);
    let biases = round_values_to_dtype(
        &fixture_values(num_affine_values, -0.0009765625),
        config.scale_bias_dtype,
    );
    let input = buffer_from_f32(&device, &input_values, config.input_dtype);
    let weight = if config.bits == 4 {
        Buffer::from_slice(&device, &pack_q4(&weight_values))
    } else {
        Buffer::from_slice(&device, &weight_values)
    };
    let scales_buffer = buffer_from_f32(&device, &scales, config.scale_bias_dtype);
    let biases_buffer = buffer_from_f32(&device, &biases, config.scale_bias_dtype);
    let output_bytes = |rows| config.output_bytes(rows) / epilogue.output_divisor() as usize;
    let output = Buffer::new_zeroed(&device, output_bytes(num_total_rows));
    let kernel = Kernel::new_with_epilogue(&device, config, kind, epilogue);
    let mut cache = ReplayTestCache::new();
    let (replay, cache_hit) = cache.record(num_total_rows, || {
        let mut recorder = stream.create_replay_program();
        recorder.record(kernel.invoke(
            num_total_rows as u32,
            ReplayU32::Parameter(NUM_ACTIVE_ROWS),
            &output,
            0,
            &input,
            0,
            &weight,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
        ));
        recorder.build()
    });
    assert!(!cache_hit);

    let mut active_counts = vec![
        1,
        num_total_rows,
        3,
        num_total_rows - 1,
        2,
        num_total_rows / 2,
        num_total_rows / 2 + 1,
    ];
    active_counts.retain(|&active| active > 0 && active <= num_total_rows);
    active_counts.dedup();
    for num_active_rows in active_counts {
        let (replay, cache_hit) = cache.record(num_total_rows, || unreachable!());
        assert!(cache_hit);
        let before = output.read_typed::<u8>(0, output_bytes(num_total_rows)).to_vec();
        stream
            .submit_replay_with_arguments(
                replay,
                &ReplayArguments::new().with_u32(NUM_ACTIVE_ROWS, num_active_rows as u32),
            )
            .wait();
        let after = output.read_typed::<u8>(0, output_bytes(num_total_rows));
        let active_bytes = output_bytes(num_active_rows);
        assert_eq!(
            &after[active_bytes..],
            &before[active_bytes..],
            "inactive output rows changed"
        );
        let expected = round_values_to_dtype(
            &cpu_affine_reference(
                config,
                num_active_rows,
                &input_values[..num_active_rows as usize * config.k as usize],
                &weight_values,
                &scales,
                &biases,
            ),
            config.output_dtype,
        );
        let expected = if epilogue == Epilogue::SwiGLU {
            let width = config.n as usize / 2;
            let values: Vec<f32> = expected
                .chunks_exact(config.n as usize)
                .flat_map(|row| (0..width).map(move |i| (row[i] / (1.0 + (-row[i]).exp())) * row[width + i]))
                .collect();
            round_values_to_dtype(&values, config.output_dtype)
        } else {
            expected
        };
        let actual = read_f32(
            &output,
            num_active_rows as usize * config.n as usize / epilogue.output_divisor() as usize,
            config.output_dtype,
        );
        assert_close_case(&actual, &expected, output_tolerance(config.output_dtype), config, kind);
    }
}

fn output_tolerance(dtype: Dtype) -> f32 {
    match dtype {
        Dtype::Float32 => 1.0e-3,
        Dtype::Float16 => 0.02,
        Dtype::Bfloat16 => 0.125,
        _ => unreachable!(),
    }
}

fn execute_matmul(stream: &Stream, invocation: Invocation<'_>) {
    let mut builder = stream.create_replay_program();
    builder.record(invocation);
    let replay = builder.build();
    stream.submit_replay(&replay).wait();
}

#[test]
fn test_adaptive_matmul_supports_all_float_dtype_combinations() {
    const DTYPES: [Dtype; 3] = [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16];

    let device = Device::system_default();
    let stream = Stream::new(&device);
    let max_m = 31;
    let input_source = fixture_values(max_m * 32, 0.00390625);
    let weight = fixture_weight_bytes(8 * 32);
    let scales_source = fixture_values(8, 0.001953125);
    let biases_source = fixture_values(8, -0.0009765625);
    let weight_buffer = Buffer::from_slice(&device, &weight);

    for input_dtype in DTYPES {
        for scale_bias_dtype in DTYPES {
            for output_dtype in DTYPES {
                let config = Config {
                    n: 8,
                    k: 32,
                    group_size: 32,
                    bits: 8,
                    input_dtype,
                    output_dtype,
                    scale_bias_dtype,
                };
                let input_values = round_values_to_dtype(&input_source, input_dtype);
                let scales = round_values_to_dtype(&scales_source, scale_bias_dtype);
                let biases = round_values_to_dtype(&biases_source, scale_bias_dtype);
                let input = buffer_from_f32(&device, &input_values, input_dtype);
                let scales_buffer = buffer_from_f32(&device, &scales, scale_bias_dtype);
                let biases_buffer = buffer_from_f32(&device, &biases, scale_bias_dtype);
                let matmul = Matmul::new(&device, config);

                let cases = [
                    (matmul.registry.get(Selector::qmv_key(config)), 2),
                    (matmul.registry.get(KernelKind::QmmBm8Bn32), 7),
                    (matmul.registry.get(KernelKind::QmmBm16Bn32), 15),
                    (matmul.registry.get(KernelKind::QmmBm32Bn32), 31),
                ];
                for (kernel, m) in cases {
                    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
                    execute_matmul(
                        &stream,
                        kernel.invoke(
                            m as u32,
                            ReplayU32::Fixed(m as u32),
                            &output,
                            0,
                            &input,
                            0,
                            &weight_buffer,
                            0,
                            &scales_buffer,
                            0,
                            &biases_buffer,
                            0,
                        ),
                    );

                    let actual = read_f32(&output, m as usize * config.n as usize, output_dtype);
                    let expected = round_values_to_dtype(
                        &cpu_affine_reference(config, m, &input_values, &weight, &scales, &biases),
                        output_dtype,
                    );
                    let tolerance = match output_dtype {
                        Dtype::Float32 => 1.0e-3,
                        Dtype::Float16 => 0.02,
                        Dtype::Bfloat16 => 0.125,
                        _ => unreachable!(),
                    };
                    assert_close_case(&actual, &expected, tolerance, config, kernel.kind());
                }
            }
        }
    }
}

#[test]
fn test_qmv_fast_supports_all_float_dtype_combinations() {
    const DTYPES: [Dtype; 3] = [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16];

    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 2;
    let input_source = fixture_values(m as usize * 512, 0.00390625);
    let weight = fixture_weight_bytes(8 * 512);
    let scales_source = fixture_values(8 * (512 / 64), 0.001953125);
    let biases_source = fixture_values(8 * (512 / 64), -0.0009765625);
    let weight_buffer = Buffer::from_slice(&device, &weight);

    for input_dtype in DTYPES {
        for scale_bias_dtype in DTYPES {
            for output_dtype in DTYPES {
                let config = Config {
                    n: 8,
                    k: 512,
                    group_size: 64,
                    bits: 8,
                    input_dtype,
                    output_dtype,
                    scale_bias_dtype,
                };
                let input_values = round_values_to_dtype(&input_source, input_dtype);
                let scales = round_values_to_dtype(&scales_source, scale_bias_dtype);
                let biases = round_values_to_dtype(&biases_source, scale_bias_dtype);
                let input = buffer_from_f32(&device, &input_values, input_dtype);
                let scales_buffer = buffer_from_f32(&device, &scales, scale_bias_dtype);
                let biases_buffer = buffer_from_f32(&device, &biases, scale_bias_dtype);
                let output = Buffer::new_zeroed(&device, config.output_bytes(m));
                let kernel = Kernel::new(&device, config, KernelKind::QmvBn8Bk32);
                execute_matmul(
                    &stream,
                    kernel.invoke(
                        m as u32,
                        ReplayU32::Fixed(m as u32),
                        &output,
                        0,
                        &input,
                        0,
                        &weight_buffer,
                        0,
                        &scales_buffer,
                        0,
                        &biases_buffer,
                        0,
                    ),
                );

                let actual = read_f32(&output, m as usize * config.n as usize, output_dtype);
                let expected = round_values_to_dtype(
                    &cpu_affine_reference(config, m, &input_values, &weight, &scales, &biases),
                    output_dtype,
                );
                let tolerance = match output_dtype {
                    Dtype::Float32 => 1.0e-3,
                    Dtype::Float16 => 0.02,
                    Dtype::Bfloat16 => 0.125,
                    _ => unreachable!(),
                };
                assert_close_case(&actual, &expected, tolerance, config, kernel.kind());
            }
        }
    }
}

#[test]
fn test_qmv_reference() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 2;
    let config = Config {
        n: 4,
        k: 32,
        group_size: 32,
        bits: 8,
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Float32,
        scale_bias_dtype: Dtype::Float32,
    };
    let input_f32 = fixture_values(m as usize * config.k as usize, 0.03125);
    let input_bf16 = input_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let weight = fixture_weight_bytes(config.n as usize * config.k as usize);
    let scales = fixture_values(config.n as usize, 0.015625);
    let biases = fixture_values(config.n as usize, -0.0078125);
    let input = Buffer::from_slice(&device, &input_bf16);
    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
    let weight_buffer = Buffer::from_slice(&device, &weight);
    let scales_buffer = Buffer::from_slice(&device, &scales);
    let biases_buffer = Buffer::from_slice(&device, &biases);

    execute_matmul(
        &stream,
        Kernel::new(&device, config, Selector::key(config, m, Epilogue::Identity)).invoke(
            m as u32,
            ReplayU32::Fixed(m as u32),
            &output,
            0,
            &input,
            0,
            &weight_buffer,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
        ),
    );

    let actual = output.read_typed::<f32>(0, m as usize * config.n as usize);
    let expected = cpu_affine_reference(
        config,
        m,
        &input_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &weight,
        &scales,
        &biases,
    );
    assert_close(&actual, &expected, 1.0e-4);
}

#[test]
fn test_qmv_fast_reference() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 2;
    let config = Config {
        n: 8,
        k: 512,
        group_size: 64,
        bits: 8,
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Float32,
        scale_bias_dtype: Dtype::Float32,
    };
    let input_f32 = fixture_values(m as usize * config.k as usize, 0.00390625);
    let input_bf16 = input_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let weight = fixture_weight_bytes(config.n as usize * config.k as usize);
    let scales = fixture_values(config.n as usize * (config.k / config.group_size) as usize, 0.001953125);
    let biases = fixture_values(
        config.n as usize * (config.k / config.group_size) as usize,
        -0.0009765625,
    );
    let input = Buffer::from_slice(&device, &input_bf16);
    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
    let weight_buffer = Buffer::from_slice(&device, &weight);
    let scales_buffer = Buffer::from_slice(&device, &scales);
    let biases_buffer = Buffer::from_slice(&device, &biases);

    execute_matmul(
        &stream,
        Kernel::new(&device, config, Selector::key(config, m, Epilogue::Identity)).invoke(
            m as u32,
            ReplayU32::Fixed(m as u32),
            &output,
            0,
            &input,
            0,
            &weight_buffer,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
        ),
    );

    let actual = output.read_typed::<f32>(0, m as usize * config.n as usize);
    let expected = cpu_affine_reference(
        config,
        m,
        &input_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &weight,
        &scales,
        &biases,
    );
    assert_close(&actual, &expected, 1.0e-3);
}

#[test]
fn test_qmm_reference() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 18;
    let config = Config {
        n: 4,
        k: 32,
        group_size: 32,
        bits: 8,
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Float32,
        scale_bias_dtype: Dtype::Float32,
    };
    let input_f32 = fixture_values(m as usize * config.k as usize, 0.03125);
    let input_bf16 = input_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let weight = fixture_weight_bytes(config.n as usize * config.k as usize);
    let scales = fixture_values(config.n as usize, 0.015625);
    let biases = fixture_values(config.n as usize, -0.0078125);
    let input = Buffer::from_slice(&device, &input_bf16);
    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
    let weight_buffer = Buffer::from_slice(&device, &weight);
    let scales_buffer = Buffer::from_slice(&device, &scales);
    let biases_buffer = Buffer::from_slice(&device, &biases);

    execute_matmul(
        &stream,
        Kernel::new(&device, config, Selector::key(config, m, Epilogue::Identity)).invoke(
            m as u32,
            ReplayU32::Fixed(m as u32),
            &output,
            0,
            &input,
            0,
            &weight_buffer,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
        ),
    );

    let actual = output.read_typed::<f32>(0, m as usize * config.n as usize);
    let expected = cpu_affine_reference(
        config,
        m,
        &input_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &weight,
        &scales,
        &biases,
    );
    assert_close(&actual, &expected, 1.0e-4);
}

#[test]
fn test_qmm_bm8_bn32_q4_bf16_reference() {
    for n in [32, 65_537] {
        assert_qmm_bm8_bm16_bn32_q4_bf16_reference(8, n);
    }
}

#[test]
fn test_qmm_bm16_bn32_q4_bf16_reference() {
    for n in [32, 65_537] {
        assert_qmm_bm8_bm16_bn32_q4_bf16_reference(16, n);
    }
}

fn assert_qmm_bm8_bm16_bn32_q4_bf16_reference(bm: usize, n: i32) {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 7;
    let config = Config {
        n,
        k: 64,
        group_size: 64,
        bits: 4,
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
    };
    let input_f32 = fixture_values(m as usize * config.k as usize, 0.03125);
    let input_bf16 = input_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let weight_values = fixture_q4_values(config.n as usize * config.k as usize);
    let weight = pack_q4(&weight_values);
    let scales_f32 = fixture_values(config.n as usize, 0.015625);
    let biases_f32 = fixture_values(config.n as usize, -0.0078125);
    let scales_bf16 = scales_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let biases_bf16 = biases_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let input = Buffer::from_slice(&device, &input_bf16);
    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
    let weight_buffer = Buffer::from_slice(&device, &weight);
    let scales_buffer = Buffer::from_slice(&device, &scales_bf16);
    let biases_buffer = Buffer::from_slice(&device, &biases_bf16);

    let kernel = match bm {
        8 => Kernel::new(&device, config, KernelKind::QmmBm8Bn32),
        16 => Kernel::new(&device, config, KernelKind::QmmBm16Bn32),
        _ => panic!("QMM BM=8/16 BN=32 reference requires BM=8 or BM=16"),
    };
    execute_matmul(
        &stream,
        kernel.invoke(
            m as u32,
            ReplayU32::Fixed(m as u32),
            &output,
            0,
            &input,
            0,
            &weight_buffer,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
        ),
    );

    let actual = output
        .read_typed::<u16>(0, m as usize * config.n as usize)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let expected = cpu_affine_reference(
        config,
        m,
        &input_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &weight_values,
        &scales_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &biases_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
    )
    .into_iter()
    .map(|value| bf16::from_f32(value).to_f32())
    .collect::<Vec<_>>();
    assert_close(&actual, &expected, 0.125);
}

#[test]
fn test_qmv_bf16() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 1;
    let config = Config {
        n: 4,
        k: 32,
        group_size: 32,
        bits: 8,
        input_dtype: Dtype::Float32,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
    };
    let input = fixture_values(m as usize * config.k as usize, 0.03125);
    let weight = fixture_weight_bytes(config.n as usize * config.k as usize);
    let scales_f32 = fixture_values(config.n as usize, 0.015625);
    let biases_f32 = fixture_values(config.n as usize, -0.0078125);
    let scales_bf16 = scales_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let biases_bf16 = biases_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let input_buffer = Buffer::from_slice(&device, &input);
    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
    let weight_buffer = Buffer::from_slice(&device, &weight);
    let scales_buffer = Buffer::from_slice(&device, &scales_bf16);
    let biases_buffer = Buffer::from_slice(&device, &biases_bf16);

    execute_matmul(
        &stream,
        Kernel::new(&device, config, Selector::key(config, m, Epilogue::Identity)).invoke(
            m as u32,
            ReplayU32::Fixed(m as u32),
            &output,
            0,
            &input_buffer,
            0,
            &weight_buffer,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
        ),
    );

    let actual = output
        .read_typed::<u16>(0, m as usize * config.n as usize)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let expected = cpu_affine_reference(
        config,
        m,
        &input,
        &weight,
        &scales_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &biases_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
    )
    .into_iter()
    .map(|value| bf16::from_f32(value).to_f32())
    .collect::<Vec<_>>();
    assert_close(&actual, &expected, 1.0e-4);
}

#[test]
fn test_qmv_fast_bf16() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 2;
    let config = Config {
        n: 8,
        k: 512,
        group_size: 64,
        bits: 8,
        input_dtype: Dtype::Float32,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
    };
    let input = fixture_values(m as usize * config.k as usize, 0.00390625);
    let weight = fixture_weight_bytes(config.n as usize * config.k as usize);
    let scales_f32 = fixture_values(config.n as usize * (config.k / config.group_size) as usize, 0.001953125);
    let biases_f32 = fixture_values(
        config.n as usize * (config.k / config.group_size) as usize,
        -0.0009765625,
    );
    let scales_bf16 = scales_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let biases_bf16 = biases_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let input_buffer = Buffer::from_slice(&device, &input);
    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
    let weight_buffer = Buffer::from_slice(&device, &weight);
    let scales_buffer = Buffer::from_slice(&device, &scales_bf16);
    let biases_buffer = Buffer::from_slice(&device, &biases_bf16);

    execute_matmul(
        &stream,
        Kernel::new(&device, config, Selector::key(config, m, Epilogue::Identity)).invoke(
            m as u32,
            ReplayU32::Fixed(m as u32),
            &output,
            0,
            &input_buffer,
            0,
            &weight_buffer,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
        ),
    );

    let actual = output
        .read_typed::<u16>(0, m as usize * config.n as usize)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let expected = cpu_affine_reference(
        config,
        m,
        &input,
        &weight,
        &scales_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &biases_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
    )
    .into_iter()
    .map(|value| bf16::from_f32(value).to_f32())
    .collect::<Vec<_>>();
    assert_close(&actual, &expected, 1.0e-3);
}

#[test]
fn test_qmm_bf16() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let m = 18;
    let config = Config {
        n: 4,
        k: 32,
        group_size: 32,
        bits: 8,
        input_dtype: Dtype::Float32,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
    };
    let input = fixture_values(m as usize * config.k as usize, 0.03125);
    let weight = fixture_weight_bytes(config.n as usize * config.k as usize);
    let scales_f32 = fixture_values(config.n as usize, 0.015625);
    let biases_f32 = fixture_values(config.n as usize, -0.0078125);
    let scales_bf16 = scales_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let biases_bf16 = biases_f32
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect::<Vec<_>>();
    let input_buffer = Buffer::from_slice(&device, &input);
    let output = Buffer::new_zeroed(&device, config.output_bytes(m));
    let weight_buffer = Buffer::from_slice(&device, &weight);
    let scales_buffer = Buffer::from_slice(&device, &scales_bf16);
    let biases_buffer = Buffer::from_slice(&device, &biases_bf16);

    execute_matmul(
        &stream,
        Kernel::new(&device, config, Selector::key(config, m, Epilogue::Identity)).invoke(
            m as u32,
            ReplayU32::Fixed(m as u32),
            &output,
            0,
            &input_buffer,
            0,
            &weight_buffer,
            0,
            &scales_buffer,
            0,
            &biases_buffer,
            0,
        ),
    );

    let actual = output
        .read_typed::<u16>(0, m as usize * config.n as usize)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let expected = cpu_affine_reference(
        config,
        m,
        &input,
        &weight,
        &scales_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
        &biases_bf16
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>(),
    )
    .into_iter()
    .map(|value| bf16::from_f32(value).to_f32())
    .collect::<Vec<_>>();
    assert_close(&actual, &expected, 1.0e-4);
}

fn cpu_affine_reference(
    config: Config,
    m: i32,
    input: &[f32],
    weight: &[u8],
    scales: &[f32],
    biases: &[f32],
) -> Vec<f32> {
    let m = m as usize;
    let n = config.n as usize;
    let k = config.k as usize;
    let mut output = vec![0.0_f32; m * n];
    for row in 0..m {
        let input_row = &input[row * k..(row + 1) * k];
        for col in 0..n {
            let weight_row = &weight[col * k..(col + 1) * k];
            let mut value = 0.0_f32;
            for group in 0..(k / config.group_size as usize) {
                let group_start = group * config.group_size as usize;
                let group_end = group_start + config.group_size as usize;
                let input_group = &input_row[group_start..group_end];
                let weight_group = &weight_row[group_start..group_end];
                let input_sum = input_group.iter().copied().sum::<f32>();
                let dot = input_group
                    .iter()
                    .zip(weight_group)
                    .map(|(x, w)| *x * f32::from(*w))
                    .sum::<f32>();
                let affine_index = col * (k / config.group_size as usize) + group;
                value += scales[affine_index] * dot + input_sum * biases[affine_index];
            }
            output[row * n + col] = value;
        }
    }
    output
}

fn fixture_values(len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|index| ((index % 17) as f32 - 8.0) * scale).collect()
}

fn round_values_to_dtype(values: &[f32], dtype: Dtype) -> Vec<f32> {
    match dtype {
        Dtype::Float32 => values.to_vec(),
        Dtype::Float16 => values.iter().map(|&value| f16::from_f32(value).to_f32()).collect(),
        Dtype::Bfloat16 => values.iter().map(|&value| bf16::from_f32(value).to_f32()).collect(),
        _ => panic!("affine dtype test requires f32, f16, or bf16"),
    }
}

fn buffer_from_f32(device: &Device, values: &[f32], dtype: Dtype) -> Buffer {
    match dtype {
        Dtype::Float32 => Buffer::from_slice(device, values),
        Dtype::Float16 => {
            Buffer::from_slice(
                device,
                &values
                    .iter()
                    .map(|&value| f16::from_f32(value).to_bits())
                    .collect::<Vec<_>>(),
            )
        },
        Dtype::Bfloat16 => {
            Buffer::from_slice(
                device,
                &values
                    .iter()
                    .map(|&value| bf16::from_f32(value).to_bits())
                    .collect::<Vec<_>>(),
            )
        },
        _ => panic!("affine dtype test requires f32, f16, or bf16"),
    }
}

fn read_f32(buffer: &Buffer, len: usize, dtype: Dtype) -> Vec<f32> {
    match dtype {
        Dtype::Float32 => buffer.read_typed::<f32>(0, len),
        Dtype::Float16 => {
            buffer
                .read_typed::<u16>(0, len)
                .into_iter()
                .map(|bits| f16::from_bits(bits).to_f32())
                .collect()
        },
        Dtype::Bfloat16 => {
            buffer
                .read_typed::<u16>(0, len)
                .into_iter()
                .map(|bits| bf16::from_bits(bits).to_f32())
                .collect()
        },
        _ => panic!("affine dtype test requires f32, f16, or bf16"),
    }
}

fn fixture_weight_bytes(len: usize) -> Vec<u8> {
    (0..len).map(|index| ((index * 7 + 3) % 251) as u8).collect()
}

fn fixture_q4_values(len: usize) -> Vec<u8> {
    (0..len).map(|index| ((index * 7 + 3) % 16) as u8).collect()
}

fn pack_q4(values: &[u8]) -> Vec<u32> {
    assert!(values.len().is_multiple_of(8));
    values
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0u32, |word, (index, &value)| word | u32::from(value) << (index * 4))
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "mixed affine mismatch at {index}: actual={actual} expected={expected} diff={diff}"
        );
    }
}

fn assert_close_case(actual: &[f32], expected: &[f32], tolerance: f32, config: Config, kind: KernelKind) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "affine dtype combination mismatch: config={config:?} kind={kind:?} index={index} actual={actual} \
             expected={expected} diff={diff} tolerance={tolerance}"
        );
    }
}
