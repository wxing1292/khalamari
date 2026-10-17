use half::bf16;

use super::*;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::Stream;

const TEST_NUM_ACTIVE_REQUESTS: ReplayParameterKey =
    ReplayParameterKey::new("test.dflash2_selector.num_active_requests");
const OUTPUT_DISTRIBUTION_INDICES: [u32; 8] = [3, 7, 1, 5, 0, 6, 2, 4];

#[test]
fn test_replay_active_counts() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config {
        rank: 4,
        top_k: 3,
        embedding_dtype: Dtype::Bfloat16,
    };
    let shape = Shape {
        num_total_requests: 4,
        num_steps: 2,
    };
    let candidates = (0..config.candidate_count(shape))
        .map(|index| 1_000 + index as i32)
        .collect::<Vec<_>>();
    let anchors = (0..shape.num_total_requests)
        .map(|request| 10_000 + request as i32)
        .collect::<Vec<_>>();
    let expected_predecessors = predecessor_reference(shape, config.top_k, &anchors, &candidates);
    let predecessor_embeddings = embedding_rows(&expected_predecessors, config.rank as usize, 0.013);
    let successor_embeddings = embedding_rows(&candidates, config.rank as usize, -0.009);
    let projected_hidden = (0..shape.proposal_count() * config.rank as usize)
        .map(|index| bf16::from_f32((index as f32 % 9.0 - 4.0) / 7.0).to_bits())
        .collect::<Vec<_>>();
    let candidate_logits = (0..config.candidate_count(shape))
        .map(|index| (index as f32 % 7.0 - 3.0) / 5.0)
        .collect::<Vec<_>>();
    let anchor_buffer = Buffer::from_slice(&device, &anchors);
    let candidate_buffer = Buffer::from_slice(&device, &candidates);
    let predecessor_id_buffer = Buffer::new_zeroed(&device, candidates.len() * size_of::<i32>());
    let predecessor_embedding_buffer = Buffer::from_slice(&device, &predecessor_embeddings);
    let successor_embedding_buffer = Buffer::from_slice(&device, &successor_embeddings);
    let hidden_buffer = Buffer::from_slice(&device, &projected_hidden);
    let logit_buffer = Buffer::from_slice(&device, &candidate_logits);
    let score_buffer = Buffer::new_zeroed(&device, config.score_count(shape) * size_of::<f32>());
    let params = Buffer::new_zeroed(&device, shape.num_total_requests as usize * 4 * size_of::<u32>());
    for request in 0..shape.num_total_requests as usize {
        params.write_typed(request * 4, &[0.8f32, 1.0]);
        params.write_typed(request * 4 + 2, &[17 + request as u32, config.top_k]);
    }
    let req_slots = Buffer::from_slice(&device, &(0..shape.num_total_requests).collect::<Vec<_>>());
    let sample_positions = Buffer::from_slice(
        &device,
        &(0..shape.num_total_requests)
            .map(|request| 29 + request)
            .collect::<Vec<_>>(),
    );
    let proposal_count = shape.proposal_count();
    let distribution_indices = Buffer::from_slice(&device, &OUTPUT_DISTRIBUTION_INDICES);
    let proposal_token_ids = Buffer::new_zeroed(&device, proposal_count * size_of::<i32>());
    let proposal_probs = Buffer::new_zeroed(&device, proposal_count * size_of::<f32>());
    let distribution_token_ids = Buffer::new_zeroed(&device, proposal_count * 4 * size_of::<i32>());
    let distribution_probs = Buffer::new_zeroed(&device, proposal_count * 4 * size_of::<f32>());
    let compute = Compute::new(&device, config);
    let active = ReplayU32::Parameter(TEST_NUM_ACTIVE_REQUESTS);
    let replay = {
        let mut builder = stream.create_replay_program();
        builder.record(compute.invoke_predecessor_ids(
            shape,
            active,
            PredecessorIdBuffers {
                anchor_token_ids: &anchor_buffer,
                candidate_token_ids: &candidate_buffer,
                predecessor_token_ids: &predecessor_id_buffer,
            },
        ));
        builder.record_with_barrier_before(compute.invoke_scores(
            shape,
            active,
            ScoreBuffers {
                candidate_logits: &logit_buffer,
                projected_hidden: &hidden_buffer,
                predecessor_embeddings: &predecessor_embedding_buffer,
                successor_embeddings: &successor_embedding_buffer,
                scores: &score_buffer,
            },
        ));
        builder.record_with_barrier_before(compute.invoke_walk(
            shape,
            active,
            WalkBuffers {
                candidate_token_ids: &candidate_buffer,
                scores: &score_buffer,
                params: &params,
                req_slots: &req_slots,
                sample_positions: &sample_positions,
                sampling_domain: 0xD1A5_0001,
                output_distribution_indices: &distribution_indices,
                proposal_token_ids: &proposal_token_ids,
                proposal_probs: &proposal_probs,
                distribution_token_ids: &distribution_token_ids,
                distribution_probs: &distribution_probs,
                max_distribution_k: 4,
                num_output_distributions: proposal_count as u32,
            },
        ));
        builder.build()
    };
    let expected_scores = score_reference(
        config,
        shape,
        &candidate_logits,
        &projected_hidden,
        &predecessor_embeddings,
        &successor_embeddings,
    );

    for num_active_requests in [1, 4, 3, 2] {
        let inactive_distributions = OUTPUT_DISTRIBUTION_INDICES
            [num_active_requests as usize * shape.num_steps as usize..]
            .iter()
            .map(|&distribution| {
                let offset = distribution as usize * 4;
                (
                    distribution,
                    distribution_token_ids.read_typed::<i32>(offset, 4),
                    distribution_probs.read_typed::<f32>(offset, 4),
                )
            })
            .collect::<Vec<_>>();
        let arguments = ReplayArguments::new().with_u32(TEST_NUM_ACTIVE_REQUESTS, num_active_requests);
        stream.submit_replay_with_arguments(&replay, &arguments).wait();

        let active_candidate_count = num_active_requests as usize * shape.num_steps as usize * config.top_k as usize;
        assert_eq!(
            predecessor_id_buffer.read_typed::<i32>(0, active_candidate_count),
            expected_predecessors[..active_candidate_count]
        );
        let active_score_count = active_candidate_count * config.top_k as usize;
        let actual_scores = score_buffer.read_typed::<f32>(0, active_score_count);
        for (actual, expected) in actual_scores.iter().zip(&expected_scores[..active_score_count]) {
            assert!(
                (actual - expected).abs() < 2.0e-4,
                "actual={actual} expected={expected}"
            );
        }

        for request in 0..num_active_requests as usize {
            let candidate_begin = request * shape.num_steps as usize * config.top_k as usize;
            let candidate_end = candidate_begin + shape.num_steps as usize * config.top_k as usize;
            let score_begin = request * shape.num_steps as usize * (config.top_k * config.top_k) as usize;
            let score_end = score_begin + shape.num_steps as usize * (config.top_k * config.top_k) as usize;
            let (expected_tokens, expected_probs, expected_distributions) = walk_reference(
                config,
                &candidates[candidate_begin..candidate_end],
                &actual_scores[score_begin..score_end],
                0.8,
                17 + request as u32,
                29 + request as u32,
                0xD1A5_0001,
            );
            let proposal_begin = request * shape.num_steps as usize;
            assert_eq!(
                proposal_token_ids.read_typed::<i32>(proposal_begin, shape.num_steps as usize),
                expected_tokens
            );
            for (actual, expected) in proposal_probs
                .read_typed::<f32>(proposal_begin, shape.num_steps as usize)
                .iter()
                .zip(expected_probs)
            {
                assert!((actual - expected).abs() < 1.0e-6);
            }
            for (step, expected_distribution) in expected_distributions.iter().enumerate() {
                let distribution = OUTPUT_DISTRIBUTION_INDICES[proposal_begin + step] as usize;
                let ids = distribution_token_ids.read_typed::<i32>(distribution * 4, 4);
                let probs = distribution_probs.read_typed::<f32>(distribution * 4, 4);
                let candidate_step_begin = candidate_begin + step * config.top_k as usize;
                assert_eq!(
                    &ids[..config.top_k as usize],
                    &candidates[candidate_step_begin..candidate_step_begin + config.top_k as usize]
                );
                assert_eq!(ids[config.top_k as usize], -1);
                for candidate in 0..config.top_k as usize {
                    assert!((probs[candidate] - expected_distribution[candidate]).abs() < 1.0e-6);
                }
                assert_eq!(probs[config.top_k as usize], 0.0);
            }
        }
        for (distribution, token_ids, probs) in inactive_distributions {
            let offset = distribution as usize * 4;
            assert_eq!(distribution_token_ids.read_typed::<i32>(offset, 4), token_ids);
            assert_eq!(distribution_probs.read_typed::<f32>(offset, 4), probs);
        }
    }
}

fn predecessor_reference(shape: Shape, top_k: u32, anchors: &[i32], candidates: &[i32]) -> Vec<i32> {
    let mut predecessors = Vec::with_capacity(candidates.len());
    for (request, &anchor) in anchors.iter().enumerate() {
        for step in 0..shape.num_steps as usize {
            if step == 0 {
                predecessors.extend(std::iter::repeat_n(anchor, top_k as usize));
            } else {
                let begin = (request * shape.num_steps as usize + step - 1) * top_k as usize;
                predecessors.extend_from_slice(&candidates[begin..begin + top_k as usize]);
            }
        }
    }
    predecessors
}

fn embedding_rows(token_ids: &[i32], rank: usize, scale: f32) -> Vec<u16> {
    token_ids
        .iter()
        .flat_map(|&token| {
            (0..rank).map(move |index| bf16::from_f32((token as f32 + index as f32 + 1.0) * scale).to_bits())
        })
        .collect()
}

fn score_reference(
    config: Config,
    shape: Shape,
    unary: &[f32],
    hidden: &[u16],
    predecessors: &[u16],
    successors: &[u16],
) -> Vec<f32> {
    let rank = config.rank as usize;
    let top_k = config.top_k as usize;
    let mut scores = vec![0.0; config.score_count(shape)];
    for request in 0..shape.num_total_requests as usize {
        for step in 0..shape.num_steps as usize {
            let candidate_base = (request * shape.num_steps as usize + step) * top_k;
            let hidden_base = (request * shape.num_steps as usize + step) * rank;
            for predecessor in 0..top_k {
                for successor in 0..top_k {
                    let mut score = unary[candidate_base + successor];
                    for index in 0..rank {
                        score += bf16::from_bits(predecessors[(candidate_base + predecessor) * rank + index]).to_f32()
                            * bf16::from_bits(hidden[hidden_base + index]).to_f32()
                            * bf16::from_bits(successors[(candidate_base + successor) * rank + index]).to_f32();
                    }
                    let edge = ((candidate_base + predecessor) * top_k) + successor;
                    scores[edge] = score;
                }
            }
        }
    }
    scores
}

fn walk_reference(
    config: Config,
    candidates: &[i32],
    scores: &[f32],
    temperature: f32,
    seed: u32,
    sample_position: u32,
    domain: u32,
) -> (Vec<i32>, Vec<f32>, Vec<Vec<f32>>) {
    let top_k = config.top_k as usize;
    let mut previous = 0;
    let mut tokens = Vec::new();
    let mut selected_probs = Vec::new();
    let mut distributions = Vec::new();
    for step in 0..2 {
        let score_base = (step * top_k + previous) * top_k;
        let maximum = scores[score_base..score_base + top_k]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities = scores[score_base..score_base + top_k]
            .iter()
            .map(|&score| ((score - maximum) / temperature).exp())
            .collect::<Vec<_>>();
        let total = probabilities.iter().sum::<f32>();
        for probability in &mut probabilities {
            *probability /= total;
        }
        let draw = uniform01(sampling_random(seed, sample_position + step as u32, domain));
        let mut cumulative = 0.0;
        let mut index = top_k - 1;
        for (candidate, &probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if cumulative >= draw {
                index = candidate;
                break;
            }
        }
        tokens.push(candidates[step * top_k + index]);
        selected_probs.push(probabilities[index]);
        distributions.push(probabilities);
        previous = index;
    }
    (tokens, selected_probs, distributions)
}

fn sampling_random(seed: u32, sample_position: u32, domain: u32) -> u32 {
    mix(seed ^ mix(sample_position.wrapping_add(0x9e37_79b9)) ^ domain)
}

fn mix(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn uniform01(random: u32) -> f32 {
    ((random & 0x00ff_ffff) as f32 + 0.5) / 16_777_216.0
}
