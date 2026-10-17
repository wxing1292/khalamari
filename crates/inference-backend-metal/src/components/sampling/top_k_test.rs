use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use half::bf16;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::SamplingDomain;
use inference_executor_core::sampling::reference::ReferenceSampleRow;
use inference_executor_core::sampling::reference::sparse_sample_row_with_domain_reference;

use super::*;
use crate::metal::Buffer;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::ReplayArguments;
use crate::metal::ReplayProgram;
use crate::metal::Stream;

const ACTIVE_SEQUENCE: [u32; 4] = [1, 4, 3, 2];
const SAMPLE_POSITIONS: [u32; 4] = [11, 29, 41, 53];
const OUTPUT_DISTRIBUTION_INDICES: [u32; 4] = [1, 3, 0, 2];

#[test]
fn test_map_variant_selection() {
    let reduction_shape = Shape {
        num_total_sampling_inputs: 1,
        vocab_size: 256,
        top_k: 20,
    };
    let bitonic_shape = Shape {
        top_k: 64,
        ..reduction_shape
    };
    assert_eq!(
        Selector::key(reduction_shape, Dtype::Float32, Operation::Sample),
        VariantKey::F32Reduction
    );
    assert_eq!(
        Selector::key(bitonic_shape, Dtype::Float32, Operation::Sample),
        VariantKey::F32Bitonic
    );
    assert_eq!(
        Selector::key(reduction_shape, Dtype::Bfloat16, Operation::Sample),
        VariantKey::Bf16Reduction
    );
    assert_eq!(
        Selector::key(bitonic_shape, Dtype::Bfloat16, Operation::Sample),
        VariantKey::Bf16Bitonic
    );
    for operation in [Operation::WriteDistribution, Operation::SampleAndWriteDistribution] {
        assert_eq!(
            Selector::key(reduction_shape, Dtype::Float32, operation),
            VariantKey::F32Bitonic
        );
        assert_eq!(
            Selector::key(reduction_shape, Dtype::Bfloat16, operation),
            VariantKey::Bf16Bitonic
        );
    }
}

#[test]
fn test_merge_replay_bucketing() {
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        for top_k in [4, 64] {
            run_merge_case(dtype, top_k);
        }
    }
}

#[test]
fn test_sample_replay_bucketing() {
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        for top_k in [20, 64] {
            run_sample_case(dtype, top_k);
        }
    }
}

#[test]
fn test_write_distribution_replay_bucketing() {
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        run_distribution_case(dtype, false);
    }
}

#[test]
fn test_sample_write_distribution_replay_bucketing() {
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        run_distribution_case(dtype, true);
    }
}

fn run_merge_case(dtype: Dtype, top_k: u32) {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let shape = Shape {
        num_total_sampling_inputs: 4,
        vocab_size: 257,
        top_k,
    };
    let (logits, reference_logits, logits_offset_bytes) = logits_fixture(&device, dtype, shape, 0x62D0_14A9);
    let map = MapCompute::new(&device);
    let reduce = ReduceCompute::new(&device);
    let candidate_count = map.candidate_count(shape);
    let tile_token_ids = Buffer::new_zeroed_elements(&device, candidate_count, Dtype::Int32);
    let tile_logits = Buffer::new_zeroed_elements(&device, candidate_count, Dtype::Float32);
    let token_ids = Buffer::new_zeroed_elements(
        &device,
        shape.num_total_sampling_inputs as usize * shape.top_k as usize,
        Dtype::Int32,
    );
    let merged_logits = Buffer::new_zeroed_elements(
        &device,
        shape.num_total_sampling_inputs as usize * shape.top_k as usize,
        Dtype::Float32,
    );
    let replay = {
        let mut builder = stream.create_replay_program();
        builder.record(map.invoke_replay(
            shape,
            dtype,
            Operation::Merge,
            MapBuffers {
                logits: &logits,
                logits_offset_bytes,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        builder.record_with_barrier_before(reduce.invoke_merge(
            shape,
            MergeBuffers {
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
                token_ids: &token_ids,
                logits: &merged_logits,
            },
        ));
        builder.build()
    };

    for num_active_rows in ACTIVE_SEQUENCE {
        let mut arguments = ReplayArguments::new();
        map.add_replay_arguments(shape, num_active_rows, &mut arguments);
        reduce.add_replay_arguments(shape, num_active_rows, &mut arguments);
        stream.submit_replay_with_arguments(&replay, &arguments).wait();

        let num_values = num_active_rows as usize * shape.top_k as usize;
        let actual_ids = token_ids.read_typed::<i32>(0, num_values);
        let actual_logits = merged_logits.read_typed::<f32>(0, num_values);
        for row in 0..num_active_rows as usize {
            let mut expected = reference_logits[row * shape.vocab_size as usize..(row + 1) * shape.vocab_size as usize]
                .iter()
                .copied()
                .enumerate()
                .collect::<Vec<_>>();
            expected.sort_by(|(left_id, left), (right_id, right)| {
                right.total_cmp(left).then_with(|| left_id.cmp(right_id))
            });
            for (slot, &(token_id, logit)) in expected.iter().take(shape.top_k as usize).enumerate() {
                let output = row * shape.top_k as usize + slot;
                assert_eq!(actual_ids[output], token_id as i32);
                assert_eq!(actual_logits[output], logit);
            }
        }
    }
    let mut arguments = ReplayArguments::new();
    map.add_replay_arguments(shape, 1, &mut arguments);
    reduce.add_replay_arguments(shape, 1, &mut arguments);
    assert_invalid_logits_rejected(&stream, &replay, &arguments, &logits, dtype, shape.vocab_size as usize);
}

fn run_sample_case(dtype: Dtype, top_k: u32) {
    let sampling_domain = SamplingDomain::Target;
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let shape = Shape {
        num_total_sampling_inputs: 4,
        vocab_size: 257,
        top_k,
    };
    let (logits, reference_logits, logits_offset_bytes) = logits_fixture(&device, dtype, shape, 0x19A2_7C4D);
    let configs = sampler_configs(top_k as usize);
    let params = sampling_params(&device, &configs);
    let req_slots = Buffer::from_slice(&device, &(0..shape.num_total_sampling_inputs).collect::<Vec<_>>());
    let sample_positions = Buffer::from_slice(&device, &SAMPLE_POSITIONS);
    let map = MapCompute::new(&device);
    let reduce = ReduceCompute::new(&device);
    let candidate_count = map.candidate_count(shape);
    let tile_token_ids = Buffer::new_zeroed_elements(&device, candidate_count, Dtype::Int32);
    let tile_logits = Buffer::new_zeroed_elements(&device, candidate_count, Dtype::Float32);
    let sampled_token_ids =
        Buffer::new_zeroed_elements(&device, shape.num_total_sampling_inputs as usize, Dtype::Int32);
    let sampled_token_probs =
        Buffer::new_zeroed_elements(&device, shape.num_total_sampling_inputs as usize, Dtype::Float32);
    let replay = {
        let mut builder = stream.create_replay_program();
        builder.record(map.invoke_replay(
            shape,
            dtype,
            Operation::Sample,
            MapBuffers {
                logits: &logits,
                logits_offset_bytes,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        builder.record_with_barrier_before(reduce.invoke_sample(
            shape,
            SampleBuffers {
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
                token_ids: &sampled_token_ids,
                token_probs: &sampled_token_probs,
                params: &params,
                req_slots: &req_slots,
                sample_positions: &sample_positions,
                sample_position_increment: 0,
                sampling_domain: u32::from(sampling_domain),
            },
        ));
        builder.build()
    };

    for num_active_rows in ACTIVE_SEQUENCE {
        let mut arguments = ReplayArguments::new();
        map.add_replay_arguments(shape, num_active_rows, &mut arguments);
        reduce.add_replay_arguments(shape, num_active_rows, &mut arguments);
        stream.submit_replay_with_arguments(&replay, &arguments).wait();
        assert_active_samples(
            shape,
            &reference_logits,
            &configs,
            sampling_domain,
            num_active_rows as usize,
            &sampled_token_ids,
            &sampled_token_probs,
        );
    }
    let mut arguments = ReplayArguments::new();
    map.add_replay_arguments(shape, 1, &mut arguments);
    reduce.add_replay_arguments(shape, 1, &mut arguments);
    assert_invalid_logits_rejected(&stream, &replay, &arguments, &logits, dtype, shape.vocab_size as usize);
    if dtype == Dtype::Float32 && top_k > 1 {
        req_slots.write_typed(0, &[2_u32]);
        sample_positions.write_typed(0, &[394_u32]);
        write_sampling_params(
            &params,
            2,
            &SamplerConfig {
                temperature: 0.1,
                top_k: 2,
                top_p: 1.0,
                seed: 42,
            },
        );
        logits.write_typed(shape.vocab_size as usize, &[f32::MAX, f32::MAX / 2.0]);
        let failure = catch_unwind(AssertUnwindSafe(|| {
            stream.submit_replay_with_arguments(&replay, &arguments).wait();
        }))
        .expect_err("normalization overflow must fail");
        let message = failure.downcast_ref::<String>().unwrap();
        for expected in [
            "invalid probability normalization; row=0 code=2",
            "req_slot=2 sample_position=394 domain=Target (Main/bonus)",
            "top_k=2 temperature=0.1 top_p=1 seed=42",
            "diagnostic_value=NaN",
            "candidate_token=0",
        ] {
            assert!(message.contains(expected), "{message}");
        }
    }
}

fn run_distribution_case(dtype: Dtype, sample: bool) {
    let sampling_domain = SamplingDomain::Draft;
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let shape = Shape {
        num_total_sampling_inputs: 4,
        vocab_size: 257,
        top_k: 20,
    };
    let (logits, reference_logits, logits_offset_bytes) = logits_fixture(&device, dtype, shape, 0xE6B4_2A17);
    let configs = sampler_configs(shape.top_k as usize);
    let params = sampling_params(&device, &configs);
    let req_slots = Buffer::from_slice(&device, &(0..shape.num_total_sampling_inputs).collect::<Vec<_>>());
    let sample_positions = Buffer::from_slice(&device, &SAMPLE_POSITIONS);
    let output_distribution_indices = Buffer::from_slice(&device, &OUTPUT_DISTRIBUTION_INDICES);
    let max_k = 24_u32;
    let num_output_distributions = 4_u32;
    let map = MapCompute::new(&device);
    let reduce = ReduceCompute::new(&device);
    let candidate_count = map.candidate_count(shape);
    let tile_token_ids = Buffer::new_zeroed_elements(&device, candidate_count, Dtype::Int32);
    let tile_logits = Buffer::new_zeroed_elements(&device, candidate_count, Dtype::Float32);
    let sampled_token_ids =
        Buffer::new_zeroed_elements(&device, shape.num_total_sampling_inputs as usize, Dtype::Int32);
    let sampled_token_probs =
        Buffer::new_zeroed_elements(&device, shape.num_total_sampling_inputs as usize, Dtype::Float32);
    let distribution_token_ids = Buffer::new_zeroed_elements(
        &device,
        num_output_distributions as usize * max_k as usize,
        Dtype::Int32,
    );
    let distribution_probs = Buffer::new_zeroed_elements(
        &device,
        num_output_distributions as usize * max_k as usize,
        Dtype::Float32,
    );
    let operation = if sample {
        Operation::SampleAndWriteDistribution
    } else {
        Operation::WriteDistribution
    };
    let replay = {
        let mut builder = stream.create_replay_program();
        builder.record(map.invoke_replay(
            shape,
            dtype,
            operation,
            MapBuffers {
                logits: &logits,
                logits_offset_bytes,
                tile_token_ids: &tile_token_ids,
                tile_logits: &tile_logits,
            },
        ));
        if sample {
            builder.record_with_barrier_before(reduce.invoke_sample_and_write_distribution(
                shape,
                SampleAndWriteDistributionBuffers {
                    tile_token_ids: &tile_token_ids,
                    tile_logits: &tile_logits,
                    sampled_token_ids: &sampled_token_ids,
                    sampled_token_probs: &sampled_token_probs,
                    distribution_token_ids: &distribution_token_ids,
                    distribution_probs: &distribution_probs,
                    params: &params,
                    req_slots: &req_slots,
                    sample_positions: &sample_positions,
                    sample_position_increment: 0,
                    sampling_domain: u32::from(sampling_domain),
                    output_distribution_indices: &output_distribution_indices,
                    max_k,
                    num_output_distributions,
                },
            ));
        } else {
            builder.record_with_barrier_before(reduce.invoke_write_distribution(
                shape,
                WriteDistributionBuffers {
                    tile_token_ids: &tile_token_ids,
                    tile_logits: &tile_logits,
                    distribution_token_ids: &distribution_token_ids,
                    distribution_probs: &distribution_probs,
                    params: &params,
                    req_slots: &req_slots,
                    output_distribution_indices: &output_distribution_indices,
                    max_k,
                    num_output_distributions,
                },
            ));
        }
        builder.build()
    };

    for num_active_rows in ACTIVE_SEQUENCE {
        let inactive_distributions = OUTPUT_DISTRIBUTION_INDICES[num_active_rows as usize..]
            .iter()
            .map(|&distribution| {
                let offset = distribution as usize * max_k as usize;
                (
                    distribution,
                    distribution_token_ids.read_typed::<i32>(offset, max_k as usize),
                    distribution_probs.read_typed::<f32>(offset, max_k as usize),
                )
            })
            .collect::<Vec<_>>();
        let mut arguments = ReplayArguments::new();
        map.add_replay_arguments(shape, num_active_rows, &mut arguments);
        reduce.add_replay_arguments(shape, num_active_rows, &mut arguments);
        stream.submit_replay_with_arguments(&replay, &arguments).wait();

        if sample {
            assert_active_samples(
                shape,
                &reference_logits,
                &configs,
                sampling_domain,
                num_active_rows as usize,
                &sampled_token_ids,
                &sampled_token_probs,
            );
        }
        for (row, &distribution) in OUTPUT_DISTRIBUTION_INDICES
            .iter()
            .take(num_active_rows as usize)
            .enumerate()
        {
            let expected = sample_reference(shape, &reference_logits, &configs, sampling_domain, row);
            let offset = distribution as usize * max_k as usize;
            assert_eq!(
                distribution_token_ids.read_typed::<i32>(offset, expected.prob_token_ids.len()),
                expected.prob_token_ids
            );
            assert_close(
                &distribution_probs.read_typed::<f32>(offset, expected.prob_values.len()),
                &expected.prob_values,
                1.0e-5,
            );
        }
        for (distribution, token_ids, probs) in inactive_distributions {
            let offset = distribution as usize * max_k as usize;
            assert_eq!(
                distribution_token_ids.read_typed::<i32>(offset, max_k as usize),
                token_ids
            );
            assert_eq!(distribution_probs.read_typed::<f32>(offset, max_k as usize), probs);
        }
    }
    let mut arguments = ReplayArguments::new();
    map.add_replay_arguments(shape, 1, &mut arguments);
    reduce.add_replay_arguments(shape, 1, &mut arguments);
    assert_invalid_logits_rejected(&stream, &replay, &arguments, &logits, dtype, shape.vocab_size as usize);
}

// The fixture has one prefix row. Only the first active row is invalidated.
// Panics occur on the CPU after GPU completion, so no GPU fault is injected.
fn assert_invalid_logits_rejected(
    stream: &Stream,
    replay: &ReplayProgram,
    arguments: &ReplayArguments,
    logits: &Buffer,
    dtype: Dtype,
    vocab_size: usize,
) {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        match dtype {
            Dtype::Float32 => logits.write_typed(vocab_size, &vec![invalid; vocab_size]),
            Dtype::Bfloat16 => logits.write_typed(vocab_size, &vec![bf16::from_f32(invalid).to_bits(); vocab_size]),
            _ => unreachable!(),
        }
        let failure = catch_unwind(AssertUnwindSafe(|| {
            stream.submit_replay_with_arguments(replay, arguments).wait();
        }))
        .expect_err("an all-nonfinite logits row must fail its submission");
        let message = failure
            .downcast_ref::<String>()
            .expect("shader panic must report its reason");
        assert!(
            message.contains("no finite logit candidates; row=0 code=1"),
            "{message}"
        );

        // One finite candidate is valid, including token 0. Reuse also proves
        // that the completed submission releases its allocator and error state.
        match dtype {
            Dtype::Float32 => logits.write_typed(vocab_size, &[1.0_f32]),
            Dtype::Bfloat16 => logits.write_typed(vocab_size, &[bf16::from_f32(1.0).to_bits()]),
            _ => unreachable!(),
        }
        stream.submit_replay_with_arguments(replay, arguments).wait();
    }
}

fn assert_active_samples(
    shape: Shape,
    logits: &[f32],
    configs: &[SamplerConfig],
    sampling_domain: SamplingDomain,
    num_active_rows: usize,
    sampled_token_ids: &Buffer,
    sampled_token_probs: &Buffer,
) {
    let actual_tokens = sampled_token_ids.read_typed::<i32>(0, num_active_rows);
    let actual_probs = sampled_token_probs.read_typed::<f32>(0, num_active_rows);
    for row in 0..num_active_rows {
        let expected = sample_reference(shape, logits, configs, sampling_domain, row);
        assert_eq!(actual_tokens[row], expected.sampled_token as i32, "row={row}");
        assert_close(&actual_probs[row..row + 1], &[expected.sampled_prob], 1.0e-5);
    }
}

fn sample_reference(
    shape: Shape,
    logits: &[f32],
    configs: &[SamplerConfig],
    sampling_domain: SamplingDomain,
    row: usize,
) -> ReferenceSampleRow {
    sparse_sample_row_with_domain_reference(
        &configs[row],
        &logits[row * shape.vocab_size as usize..(row + 1) * shape.vocab_size as usize],
        configs[row].top_k,
        SAMPLE_POSITIONS[row],
        sampling_domain,
    )
}

fn sampler_configs(max_top_k: usize) -> Vec<SamplerConfig> {
    let requested_top_k = [1, max_top_k, 3, 7];
    let temperatures = [0.0, 0.9, 0.7, 0.8];
    let top_ps = [1.0, 0.82, 0.9, 0.85];
    let seeds = [7, 19, 31, 43];
    (0..4)
        .map(|row| {
            SamplerConfig {
                temperature: temperatures[row],
                top_k: requested_top_k[row].min(max_top_k),
                top_p: top_ps[row],
                seed: seeds[row],
            }
        })
        .collect()
}

fn sampling_params(device: &Device, configs: &[SamplerConfig]) -> Buffer {
    let params = Buffer::new_zeroed(device, configs.len() * 4 * size_of::<u32>());
    for (row, config) in configs.iter().enumerate() {
        write_sampling_params(&params, row, config);
    }
    params
}

fn write_sampling_params(params: &Buffer, row: usize, config: &SamplerConfig) {
    let offset = row * 4;
    params.write_typed(offset, &[config.temperature, config.top_p]);
    params.write_typed(offset + 2, &[config.seed(), config.top_k as u32]);
}

fn logits_fixture(device: &Device, dtype: Dtype, shape: Shape, seed: u32) -> (Buffer, Vec<f32>, usize) {
    let row_values = shape.vocab_size as usize;
    let active_values = generated_logits(shape.num_total_sampling_inputs as usize * row_values, seed);
    let mut stored_values = generated_logits(row_values, seed.wrapping_add(1));
    stored_values.extend_from_slice(&active_values);
    match dtype {
        Dtype::Float32 => {
            (
                Buffer::from_slice(device, &stored_values),
                active_values,
                row_values * size_of::<f32>(),
            )
        },
        Dtype::Bfloat16 => {
            let rounded = active_values
                .iter()
                .map(|value| bf16::from_f32(*value).to_f32())
                .collect::<Vec<_>>();
            (
                Buffer::from_slice(
                    device,
                    &stored_values
                        .iter()
                        .map(|value| bf16::from_f32(*value).to_bits())
                        .collect::<Vec<_>>(),
                ),
                rounded,
                row_values * size_of::<u16>(),
            )
        },
        _ => panic!("unsupported top-k test logits dtype {dtype:?}"),
    }
}

fn generated_logits(count: usize, mut state: u32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 4_194_304.0) - 2.0
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value mismatch at index={index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}
