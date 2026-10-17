use inference_executor_core::sampling::reference::rejection_sample_reference;

use super::Buffers;
use super::Compute;
use super::Shape;
use crate::metal::Buffer;
use crate::metal::Device;
use crate::metal::ReplayArguments;
use crate::metal::Stream;

#[test]
fn test_replay_active_counts() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let shape = Shape {
        num_total_reqs: 4,
        num_total_target_distributions: 7,
        num_total_draft_distributions: 5,
        top_k: 4,
        max_target_k: 4,
        max_draft_k: 4,
    };
    let target_rows = vec![
        vec![0.0, 0.50, 0.20, 0.30, 0.0, 0.0],
        vec![0.0, 0.10, 0.55, 0.35, 0.0, 0.0],
        vec![0.0, 0.05, 0.10, 0.85, 0.0, 0.0],
        vec![0.0, 1.00, 0.00, 0.00, 0.0, 0.0],
        vec![0.0, 0.00, 1.00, 0.00, 0.0, 0.0],
        vec![0.0, 0.00, 0.00, 0.00, 1.0, 0.0],
        vec![0.0, 0.00, 0.00, 0.00, 0.0, 1.0],
        vec![1.00, 0.00, 0.00, 0.00, 0.0, 0.0],
        vec![0.00, 1.00, 0.00, 0.00, 0.0, 0.0],
        vec![0.00, 0.00, 1.00, 0.00, 0.0, 0.0],
        vec![0.00, 0.00, 0.00, 1.00, 0.0, 0.0],
    ];
    let draft_rows = vec![
        vec![0.0, 0.25, 0.50, 0.25, 0.0, 0.0],
        vec![0.0, 0.10, 0.20, 0.70, 0.0, 0.0],
        vec![0.0, 1.00, 0.00, 0.00, 0.0, 0.0],
    ];
    let draft_tokens = [2_u32, 3, 1];
    let target_distribution = write_distributions_from_dense(
        &target_rows[..shape.num_total_target_distributions as usize],
        shape.max_target_k as usize,
    );
    let draft_distribution = write_distributions_from_dense(&draft_rows, shape.max_draft_k as usize);
    let draft_distribution_indices = [2_u32, 0, 4];
    let mut mapped_draft_token_ids = vec![-1_i32; shape.num_total_draft_distributions as usize * 4];
    let mut mapped_draft_probs = vec![0.0_f32; shape.num_total_draft_distributions as usize * 4];
    for (draft_row, &distribution_row) in draft_distribution_indices.iter().enumerate() {
        let source = draft_row * shape.max_draft_k as usize;
        let destination = distribution_row as usize * shape.max_draft_k as usize;
        mapped_draft_token_ids[destination..destination + shape.max_draft_k as usize]
            .copy_from_slice(&draft_distribution.0[source..source + shape.max_draft_k as usize]);
        mapped_draft_probs[destination..destination + shape.max_draft_k as usize]
            .copy_from_slice(&draft_distribution.1[source..source + shape.max_draft_k as usize]);
    }

    let mut padded_target_token_ids = vec![-1_i32; shape.num_total_target_distributions as usize * 4];
    let mut padded_target_probs = vec![0.0_f32; shape.num_total_target_distributions as usize * 4];
    padded_target_token_ids[..target_distribution.0.len()].copy_from_slice(&target_distribution.0);
    padded_target_probs[..target_distribution.1.len()].copy_from_slice(&target_distribution.1);
    let target_distribution_token_ids = Buffer::from_slice(&device, &padded_target_token_ids);
    let target_distribution_probs = Buffer::from_slice(&device, &padded_target_probs);
    let draft_distribution_token_ids = Buffer::from_slice(&device, &mapped_draft_token_ids);
    let draft_distribution_probs = Buffer::from_slice(&device, &mapped_draft_probs);
    let flat_draft_token_ids = Buffer::from_slice(&device, &[2_i32, 3, 1, 0, 0]);
    let cu_target_values = [0_u32, 3, 5, 6, 7];
    let cu_draft_values = [0_u32, 2, 3, 3, 3];
    let cu_target_distributions = Buffer::from_slice(&device, &cu_target_values);
    let cu_draft_distributions = Buffer::from_slice(&device, &cu_draft_values);
    let flat_draft_distribution_indices = Buffer::from_slice(&device, &[2_u32, 0, 4, 0, 0]);
    let flat_accepted_token_ids = Buffer::new_zeroed_elements(&device, 5, crate::metal::Dtype::Int32);
    let flat_accepted_probs = Buffer::new_zeroed_elements(&device, 5, crate::metal::Dtype::Float32);
    let num_accepted_tokens =
        Buffer::new_zeroed_elements(&device, shape.num_total_reqs as usize, crate::metal::Dtype::Uint32);
    let sampled_token_ids =
        Buffer::new_zeroed_elements(&device, shape.num_total_reqs as usize, crate::metal::Dtype::Int32);
    let sampled_token_probs =
        Buffer::new_zeroed_elements(&device, shape.num_total_reqs as usize, crate::metal::Dtype::Float32);
    let runtime_params_values = [
        7_u32, 19, 4, 0, // reject path
        11, 23, 1, 0, // all-accept path
        13, 29, 1, 0, // zero-draft path
        17, 31, 1, 0,
    ];
    let runtime_params = Buffer::from_slice(&device, &runtime_params_values);
    let kernel = Compute::new(&device);
    let replay = {
        let mut builder = stream.create_replay_program();
        builder.record(kernel.invoke_replay(
            shape,
            Buffers {
                target_distribution_token_ids: &target_distribution_token_ids,
                target_distribution_probs: &target_distribution_probs,
                draft_distribution_token_ids: &draft_distribution_token_ids,
                draft_distribution_probs: &draft_distribution_probs,
                flat_draft_token_ids: &flat_draft_token_ids,
                cu_target_distributions: &cu_target_distributions,
                cu_draft_distributions: &cu_draft_distributions,
                flat_draft_distribution_indices: &flat_draft_distribution_indices,
                flat_accepted_token_ids: &flat_accepted_token_ids,
                flat_accepted_probs: &flat_accepted_probs,
                num_accepted_tokens: &num_accepted_tokens,
                sampled_token_ids: &sampled_token_ids,
                sampled_token_probs: &sampled_token_probs,
                runtime_params: &runtime_params,
            },
        ));
        builder.build()
    };

    let expected = [
        rejection_sample_reference(
            &draft_tokens[..2],
            &target_rows[..3],
            &draft_rows[..2],
            runtime_params_values[0],
            runtime_params_values[1],
        ),
        rejection_sample_reference(
            &draft_tokens[2..3],
            &target_rows[3..5],
            &draft_rows[2..3],
            runtime_params_values[4],
            runtime_params_values[5],
        ),
        rejection_sample_reference(
            &[],
            &target_rows[5..6],
            &[],
            runtime_params_values[8],
            runtime_params_values[9],
        ),
        rejection_sample_reference(
            &[],
            &target_rows[6..7],
            &[],
            runtime_params_values[12],
            runtime_params_values[13],
        ),
    ];
    let assert_active_outputs = |num_active_reqs: usize| {
        let actual_counts = num_accepted_tokens.read_typed::<u32>(0, num_active_reqs);
        let actual_sampled_tokens = sampled_token_ids.read_typed::<i32>(0, num_active_reqs);
        let actual_sampled_probs = sampled_token_probs.read_typed::<f32>(0, num_active_reqs);
        for req in 0..num_active_reqs {
            assert_eq!(
                actual_counts[req] as usize,
                expected[req].accepted_tokens.len(),
                "req={req}"
            );
            assert_eq!(
                actual_sampled_tokens[req], expected[req].sampled_token as i32,
                "req={req}"
            );
            assert_close(
                &actual_sampled_probs[req..req + 1],
                &[expected[req].sampled_prob],
                1.0e-5,
            );
            let start = cu_draft_values[req] as usize;
            let num_accepted = expected[req].accepted_tokens.len();
            if num_accepted > 0 {
                assert_eq!(
                    flat_accepted_token_ids.read_typed::<i32>(start, num_accepted),
                    expected[req]
                        .accepted_tokens
                        .iter()
                        .map(|&token| token as i32)
                        .collect::<Vec<_>>()
                );
                assert_close(
                    &flat_accepted_probs.read_typed::<f32>(start, num_accepted),
                    &expected[req].accepted_probs,
                    1.0e-5,
                );
            }
        }
    };
    let submit = |num_active_reqs: usize| {
        let mut arguments = ReplayArguments::new();
        kernel.add_replay_arguments(
            shape,
            num_active_reqs as u32,
            cu_target_values[num_active_reqs],
            cu_draft_values[num_active_reqs],
            &mut arguments,
        );
        stream.submit_replay_with_arguments(&replay, &arguments).wait();
    };

    for num_active_reqs in [1_usize, 4, 3, 2] {
        submit(num_active_reqs);
        assert_active_outputs(num_active_reqs);
    }

    // The last active request has no drafts. An empty target distribution
    // must fail instead of returning token 0 with probability 0.
    let final_row = 6 * shape.max_target_k as usize;
    target_distribution_probs.write_typed(final_row, &[0.0_f32; 4]);
    submit(3); // The invalid row is outside the active request prefix.
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| submit(4)))
        .expect_err("zero rejection probability mass must fail");
    let message = failure.downcast_ref::<String>().unwrap();
    assert!(
        message.contains("invalid rejection probability mass; row=3 code=4"),
        "{message}"
    );
    assert!(message.contains("req_slot=unavailable"), "{message}");
    assert!(message.contains("distribution_index=6"), "{message}");
    assert!(message.contains("diagnostic_value=0 bits=0x00000000"), "{message}");
    assert!(!message.contains("temperature="), "{message}");
    assert!(!message.contains("top_p="), "{message}");
    target_distribution_probs.write_typed(final_row, &padded_target_probs[final_row..final_row + 4]);
    submit(4);
    assert_active_outputs(4);
}

fn write_distributions_from_dense(rows: &[Vec<f32>], row_stride: usize) -> (Vec<i32>, Vec<f32>) {
    let mut token_ids = vec![-1; rows.len() * row_stride];
    let mut probs = vec![0.0; rows.len() * row_stride];
    for (row_index, row) in rows.iter().enumerate() {
        let mut distribution = row
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, prob)| *prob > 0.0)
            .map(|(token, prob)| (token as i32, prob))
            .collect::<Vec<_>>();
        distribution.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap().then_with(|| left.0.cmp(&right.0)));
        assert!(distribution.len() <= row_stride);
        let base = row_index * row_stride;
        for (slot, (token, prob)) in distribution.into_iter().enumerate() {
            token_ids[base + slot] = token;
            probs[base + slot] = prob;
        }
    }
    (token_ids, probs)
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
