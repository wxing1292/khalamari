use std::rc::Rc;

use half::bf16;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::Stream;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::model::qwen::v3_5::sample_req_slots;
use inference_executor_core::model::qwen::v3_5::sample_sampler_configs;
use inference_executor_core::model::qwen::v3_5::sample_token_positions;
use inference_executor_core::sampling::SamplerConfig;

use super::rejection_replay::RejectionSampler;
use super::rejection_replay::RejectionSamplerInput;
use super::rejection_replay::RejectionSampling;
use super::rejection_replay::RejectionSamplingInput;
use super::rejection_sampling::SparseRejectionSamplingReqParams;
use super::sampling_params::SamplingParamsStore;
use super::spec_probs::SpecProbsStore;
use super::top_k_replay::DraftSampling;
use super::top_k_replay::DraftSamplingInput;
use super::top_k_replay::Sampling;
use super::top_k_replay::SamplingInput;
use super::top_k_sampling::TopKSampling;
use super::top_k_sampling::TopKSamplingInputs;
use super::top_k_sampling::TopKSamplingOutputBuffers;
use super::top_k_sampling::TopKSamplingWriteDistributionOutput;
use super::top_k_sampling::config::TopKSamplingBounds;
use crate::replay::Replay;

#[test]
fn test_sampling_and_draft_mask_replay() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let bounds = TopKSamplingBounds {
        max_sampling_inputs: 1,
        vocab_size: 65,
        top_k: 1,
    };
    let sampler = Rc::new(TopKSampling::new(
        &device,
        Rc::new(SamplingParamsStore::new(&device, bounds, 1)),
    ));
    let config = SamplerConfig {
        temperature: 0.0,
        top_k: 1,
        ..SamplerConfig::default()
    };
    sampler.set_params(&[0], &[config]);
    sampler.prepare(&[0], &[8]);
    let output = TopKSamplingOutputBuffers::new(&device, bounds);
    let logits = Buffer::from_slice(
        &device,
        &std::iter::repeat_n(bf16::from_f32(1000.0).to_bits(), 65)
            .chain((0..65).map(|token| bf16::from_f32(token as f32).to_bits()))
            .collect::<Vec<_>>(),
    );
    let mask = Buffer::from_slice(&device, &[0_u32, 1, 0]);
    let mut replay = Replay::new("test masked sampling", Sampling::new(sampler.clone()));
    let input = SamplingInput {
        shape: replay.component().prepare_shape(&[config]),
        inputs: TopKSamplingInputs {
            logits: &logits,
            logits_offset_bytes: 65 * size_of::<u16>(),
            token_mask: None,
        },
        output: output.as_output(),
    };
    let mut arguments = ReplayArguments::new();
    sampler.add_replay_arguments(input.shape, &mut arguments);
    let (unmasked_key, hit) = replay.record(&stream, &input);
    assert!(!hit);
    stream
        .submit_replay_with_arguments(replay.replay(&unmasked_key), &arguments)
        .wait();
    assert_eq!(output.token_ids.read_typed::<i32>(0, 1), [64]);

    let masked_input = SamplingInput {
        inputs: TopKSamplingInputs {
            token_mask: Some(&mask),
            ..input.inputs
        },
        ..input
    };
    let (masked_key, hit) = replay.record(&stream, &masked_input);
    assert!(!hit);
    assert_ne!(unmasked_key, masked_key);
    stream
        .submit_replay_with_arguments(replay.replay(&masked_key), &arguments)
        .wait();
    assert_eq!(output.token_ids.read_typed::<i32>(0, 1), [32]);

    mask.write_typed(0, &[1_u32 << 31, 0, 0]);
    assert!(replay.record(&stream, &masked_input).1);
    stream
        .submit_replay_with_arguments(replay.replay(&masked_key), &arguments)
        .wait();
    assert_eq!(output.token_ids.read_typed::<i32>(0, 1), [31]);
    assert!(replay.record(&stream, &input).1);
    stream
        .submit_replay_with_arguments(replay.replay(&unmasked_key), &arguments)
        .wait();
    assert_eq!(output.token_ids.read_typed::<i32>(0, 1), [64]);

    let distributions = SpecProbsStore::new(&device, 1, 1, 1, 1);
    let distribution_indices = Buffer::from_slice(&device, &[0_u32]);
    let mut draft_replay = Replay::new("test masked draft sampling", DraftSampling::new(sampler, 1));
    let draft_input = DraftSamplingInput {
        shape: input.shape,
        inputs: input.inputs,
        output: input.output,
        sparse: TopKSamplingWriteDistributionOutput {
            token_ids: distributions.draft_token_ids(),
            probs: distributions.draft_probs(),
            output_distribution_indices: &distribution_indices,
            max_k: 1,
            num_output_distributions: 1,
        },
    };
    for (inputs, expected_token, expected_hit) in [
        (input.inputs, 64, false),
        (masked_input.inputs, 31, false),
        (input.inputs, 64, true),
        (masked_input.inputs, 31, true),
    ] {
        let (key, hit) = draft_replay.record(&stream, &DraftSamplingInput { inputs, ..draft_input });
        assert_eq!(hit, expected_hit);
        stream
            .submit_replay_with_arguments(draft_replay.replay(&key), &arguments)
            .wait();
        assert_eq!(output.token_ids.read_typed::<i32>(0, 1), [expected_token]);
        assert_eq!(
            distributions.draft_token_ids().read_typed::<i32>(0, 1),
            [expected_token]
        );
        assert_eq!(distributions.draft_probs().read_typed::<f32>(0, 1), [1.0]);
    }
}

#[test]
fn test_rejection_uses_masked_target_and_original_draft() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let bounds = TopKSamplingBounds {
        max_sampling_inputs: 8,
        vocab_size: 4,
        top_k: 2,
    };
    let sampler = Rc::new(TopKSampling::new(
        &device,
        Rc::new(SamplingParamsStore::new(&device, bounds, 3)),
    ));
    let config = SamplerConfig {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 2,
        seed: 7,
    };
    // Requests: accept draft 1 then reject draft 2, accept draft 1 then sample bonus 3, no drafts.
    let batch = Qwen35Microbatch::new(
        vec![2, 0, 1],
        vec![0; 3],
        vec![7, 9, 11],
        vec![0, 1, 2, 0, 1, 0],
        vec![0, 3, 5, 6],
        vec![Vec::new(); 3],
        vec![config; 3],
        vec![true; 6],
    );
    let rejector = Rc::new(RejectionSampler::new(&device, 2, 3, 8, 2));
    let distributions = SpecProbsStore::new(&device, 2, 3, 8, 2);
    let draft_indices = [
        distributions.draft_distribution_index(2, 0),
        distributions.draft_distribution_index(2, 1),
        distributions.draft_distribution_index(0, 0),
    ];
    distributions
        .draft_token_ids()
        .write_typed(0, &[1_i32, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, 3]);
    distributions
        .draft_probs()
        .write_typed(0, &[1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5]);
    let prepared = rejector.prepare_inputs(&batch, &draft_indices);
    sampler.set_params(batch.req_slots(), batch.sampler_configs());
    sampler.prepare(&sample_req_slots(&batch), &sample_token_positions(&batch));
    let active_shape = sampler.active_shape(&sample_sampler_configs(&batch));
    let rejection_shape = rejector.prepare_replay_shape(&prepared, active_shape.top_k);
    let target_shape = active_shape.with_num_total_sampling_inputs(rejection_shape.num_total_target_distributions);
    let logits = Buffer::new_zeroed_elements(&device, 32, Dtype::Bfloat16);
    let mask = Buffer::from_slice(&device, &[2_u32, 8, 1, 2, 8, 1, 0, 0]);
    let target_indices = Buffer::from_slice(&device, &[0_u32, 1, 2, 3, 4, 5, 6, 7]);
    let rejection = RejectionSamplerInput {
        shape: rejection_shape,
        target_token_ids: distributions.target_token_ids(),
        target_probs: distributions.target_probs(),
        draft_token_ids: distributions.draft_token_ids(),
        draft_probs: distributions.draft_probs(),
    };
    let input = RejectionSamplingInput {
        target_shape,
        inputs: TopKSamplingInputs {
            logits: &logits,
            logits_offset_bytes: 0,
            token_mask: Some(&mask),
        },
        target_sparse: TopKSamplingWriteDistributionOutput {
            token_ids: distributions.target_token_ids(),
            probs: distributions.target_probs(),
            output_distribution_indices: &target_indices,
            max_k: 2,
            num_output_distributions: 8,
        },
        rejection,
    };
    let mut replay = Replay::new(
        "test masked rejection",
        RejectionSampling::new(sampler.clone(), rejector.clone()),
    );
    let (key, hit) = replay.record(&stream, &input);
    assert!(!hit);
    let params = [8, 10, 12].map(|sample_position| {
        SparseRejectionSamplingReqParams {
            seed: config.seed(),
            sample_position,
            top_k: 2,
        }
    });
    rejector.set_runtime_params(&params);
    let mut arguments = ReplayArguments::new();
    sampler.add_replay_arguments(target_shape, &mut arguments);
    rejector.add_replay_arguments(rejection, &mut arguments);
    stream
        .submit_replay_with_arguments(replay.replay(&key), &arguments)
        .wait();
    let results = rejector.read_results(3, 3);
    for (req, (accepted, sampled)) in [(1, 3), (1, 3), (0, 0)].into_iter().enumerate() {
        assert_eq!(results.num_accepted_tokens(req), accepted);
        assert_eq!(results.sampled_token_id(req), sampled);
        assert_eq!(results.sampled_prob(req), 1.0);
    }
    assert_eq!(results.accepted_token_ids(0, 1), [1]);
    assert_eq!(results.accepted_token_ids(2, 1), [1]);
    // Renormalizing the rejected proposal after masking would erase its positive residual mass.
    assert_eq!(
        distributions.draft_probs().read_typed::<f32>(0, 12),
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5]
    );
    assert_eq!(
        distributions.draft_token_ids().read_typed::<i32>(0, 12),
        [1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, 3]
    );
}
