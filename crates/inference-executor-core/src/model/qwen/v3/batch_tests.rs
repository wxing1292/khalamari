use inference_runtime_core::compute::DecoderSyncBlocks;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::runtime::RawRequestSlot;

use super::*;

#[test]
fn test_builds_main_microbatch() {
    let requests = vec![
        device_request(
            10,
            0,
            QueryTokens::Prefill {
                epoch: 1,
                token_index: 4,
                tokens: tokens(&[101, 102, 103]),
                window: 3,
            },
            2,
        ),
        device_request(
            11,
            1,
            QueryTokens::Decode {
                epoch: 1,
                token_index: 7,
                tokens: tokens(&[201, 202, 203]),
                spec_tokens: vec![],
            },
            3,
        ),
    ];

    let batch = Qwen3Microbatch::from_requests(&requests, vec![SamplerConfig::default(); 2]);

    assert_eq!(batch.req_slots(), &[0, 1]);
    assert_eq!(batch.token_indices(), &[4, 7]);
    assert_eq!(batch.flat_token_ids(), &[101, 102, 103, 201, 202, 203]);
    assert_eq!(batch.cu_tokens(), &[0, 3, 6]);
    assert_eq!(gather_flat_indices(&batch), vec![5]);
    assert_eq!(num_main_output_rows(&batch), 1);
    assert_eq!(sample_token_positions(&batch), vec![10]);
}

#[test]
fn test_batch_request_preserves_compute_sequence() {
    let core_batch = BatchDeviceRequest::new(
        17,
        [device_request(
            10,
            0,
            QueryTokens::Decode {
                epoch: 1,
                token_index: 4,
                tokens: tokens(&[101]),
                spec_tokens: vec![],
            },
            2,
        )],
    );

    let batch = Qwen3ModelBatchRequest::from_core_batch(&core_batch, vec![SamplerConfig::default()]);

    assert_eq!(batch.compute_seq(), 17);
    assert_eq!(batch.microbatch().total_tokens(), 1);
}

#[test]
fn test_main_batch_accepts_speculative_input_suffix() {
    let requests = [device_request(
        10,
        0,
        QueryTokens::Decode {
            epoch: 1,
            token_index: 7,
            tokens: tokens(&[201]),
            spec_tokens: tokens(&[202]),
        },
        3,
    )];

    let batch = Qwen3Microbatch::from_requests(&requests, vec![SamplerConfig::default()]);

    assert_eq!(batch.flat_token_ids(), &[201, 202]);
    assert_eq!(batch.num_spec_tokens(0), 1);
    assert_eq!(batch.flat_sample_mask(), &[true, true]);
    assert_eq!(gather_flat_indices(&batch), vec![0, 1]);
    assert_eq!(sample_token_positions(&batch), vec![8, 9]);
}

#[test]
fn test_sampling_helpers_follow_decode_requests() {
    let first = SamplerConfig {
        temperature: 0.7,
        top_k: 64,
        top_p: 0.8,
        seed: 7,
    };
    let second = SamplerConfig {
        temperature: 1.1,
        top_k: 64,
        top_p: 0.9,
        seed: 99,
    };
    let third = SamplerConfig {
        temperature: 0.5,
        top_k: 32,
        top_p: 0.6,
        seed: 123,
    };
    let batch = Qwen3Microbatch::new(
        vec![0, 1, 2],
        vec![4, 10, 20],
        vec![11, 12, 13, 21, 22, 31],
        vec![0, 3, 5, 6],
        vec![first, second, third],
        vec![0, 0, 0],
        vec![false, false, true, false, false, true],
    );

    assert_eq!(gather_flat_indices(&batch), vec![2, 5]);
    assert_eq!(num_main_output_rows(&batch), 2);
    assert_eq!(sample_token_positions(&batch), vec![7, 21]);
    assert_eq!(sample_sampler_configs(&batch), vec![first, third]);
}

#[test]
fn test_sampled_tokens_become_decode_decisions() {
    let sampled_tokens = Qwen3SampledTokens::new(vec![31, 32], vec![0.5, 0.25]);

    let decisions = sample_decisions_from_sampled_tokens(&sampled_tokens);

    assert_eq!(
        decisions,
        vec![
            Qwen3DecodeDecision {
                sampled_token: 31,
                sampled_prob: 0.5,
                ..Qwen3DecodeDecision::default()
            },
            Qwen3DecodeDecision {
                sampled_token: 32,
                sampled_prob: 0.25,
                ..Qwen3DecodeDecision::default()
            },
        ]
    );
}

#[test]
fn test_converts_main_decisions_to_core_response() {
    let core = BatchDeviceRequest::new(
        7,
        vec![
            device_request(
                10,
                0,
                QueryTokens::Prefill {
                    epoch: 1,
                    token_index: 0,
                    tokens: tokens(&[1, 2]),
                    window: 2,
                },
                0,
            ),
            device_request(
                11,
                1,
                QueryTokens::Decode {
                    epoch: 2,
                    token_index: 2,
                    tokens: tokens(&[3]),
                    spec_tokens: vec![],
                },
                0,
            ),
        ],
    );
    let decision = Qwen3DecodeDecision {
        sampled_token: 5,
        sampled_prob: 0.3,
        spec_tokens: vec![6, 7],
        spec_probs: vec![0.2, 0.1],
        spec_confidences: vec![0.9, 0.8],
        ..Default::default()
    };

    let response = to_core_batch_resp(core, vec![decision]);

    assert_eq!(response.seq, 7);
    assert!(matches!(
        response.dev_resps[0].sampled_tokens,
        SampledTokens::Prefill { epoch: 1 }
    ));
    match &response.dev_resps[1].sampled_tokens {
        SampledTokens::Decode {
            epoch,
            validated_tokens,
            validated_probs,
            sampled_token,
            spec_tokens,
            spec_probs,
            spec_confidences,
            ..
        } => {
            assert_eq!(*epoch, 2);
            assert!(validated_tokens.is_empty());
            assert!(validated_probs.is_empty());
            assert_eq!(sampled_token.value(), 5);
            assert_eq!(
                spec_tokens.iter().map(|token| token.value()).collect::<Vec<_>>(),
                [6, 7]
            );
            assert_eq!(
                spec_probs.iter().map(|value| value.into_inner()).collect::<Vec<_>>(),
                [0.2, 0.1]
            );
            assert_eq!(
                spec_confidences
                    .iter()
                    .map(|value| value.into_inner())
                    .collect::<Vec<_>>(),
                [0.9, 0.8]
            );
        },
        SampledTokens::Prefill { .. } => panic!("expected decode sampled tokens"),
    }
}

fn device_request(req_id: usize, req_slot: RawRequestSlot, tokens: QueryTokens, block_index: usize) -> DeviceRequest {
    DeviceRequest::new(
        req_id,
        req_slot,
        tokens,
        DecoderSyncBlocks::new(block_index, vec![], vec![]),
        vec![],
        SamplingConfig::default(),
    )
}

fn tokens(values: &[u32]) -> Vec<Token> {
    values.iter().copied().map(Token::new).collect()
}
