#include <metal_stdlib>
using namespace metal;

struct sampling_params {
    float temperature;
    float top_p;
    uint seed;
    uint top_k;
};

static inline uint khalamari_mix(uint h) {
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

static inline uint khalamari_sampling_random(uint seed, uint sample_position, uint sampling_domain) {
    return khalamari_mix(seed ^ khalamari_mix(sample_position + 0x9e3779b9u) ^ sampling_domain);
}

static inline float khalamari_uniform01(uint random) {
    return (float(random & 0x00ffffffu) + 0.5f) * (1.0f / 16777216.0f);
}

kernel void dflash2_selector_predecessor_ids(
    device const int* anchor_token_ids [[buffer(0)]],
    device const int* candidate_token_ids [[buffer(1)]],
    device int* predecessor_token_ids [[buffer(2)]],
    constant uint& num_active_requests [[buffer(3)]],
    constant uint& num_steps [[buffer(4)]],
    constant uint& top_k [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint candidates_per_request = num_steps * top_k;
    const uint request = gid / candidates_per_request;
    if (request >= num_active_requests) {
        return;
    }
    const uint request_offset = gid - request * candidates_per_request;
    const uint step = request_offset / top_k;
    const uint predecessor = request_offset - step * top_k;
    predecessor_token_ids[gid] = step == 0u
        ? anchor_token_ids[request]
        : candidate_token_ids[gid - top_k];
}

kernel void dflash2_selector_scores_bf16(
    device const float* candidate_logits [[buffer(0)]],
    device const bfloat* projected_hidden [[buffer(1)]],
    device const bfloat* predecessor_embeddings [[buffer(2)]],
    device const bfloat* successor_embeddings [[buffer(3)]],
    device float* scores [[buffer(4)]],
    constant uint& num_active_requests [[buffer(5)]],
    constant uint& num_steps [[buffer(6)]],
    constant uint& top_k [[buffer(7)]],
    constant uint& rank [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint scores_per_step = top_k * top_k;
    const uint scores_per_request = num_steps * scores_per_step;
    const uint request = gid / scores_per_request;
    if (request >= num_active_requests) {
        return;
    }
    const uint request_offset = gid - request * scores_per_request;
    const uint step = request_offset / scores_per_step;
    const uint edge = request_offset - step * scores_per_step;
    const uint predecessor = edge / top_k;
    const uint successor = edge - predecessor * top_k;
    const ulong candidate_base = ((ulong)request * (ulong)num_steps + (ulong)step) * (ulong)top_k;
    const ulong predecessor_base = (candidate_base + (ulong)predecessor) * (ulong)rank;
    const ulong successor_base = (candidate_base + (ulong)successor) * (ulong)rank;
    const ulong hidden_base = ((ulong)request * (ulong)num_steps + (ulong)step) * (ulong)rank;
    float score = candidate_logits[candidate_base + (ulong)successor];
    for (uint index = 0u; index < rank; ++index) {
        score += float(predecessor_embeddings[predecessor_base + (ulong)index])
            * float(projected_hidden[hidden_base + (ulong)index])
            * float(successor_embeddings[successor_base + (ulong)index]);
    }
    scores[gid] = score;
}

kernel void dflash2_selector_walk(
    device const int* candidate_token_ids [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device const sampling_params* params [[buffer(2)]],
    device const uint* req_slots [[buffer(3)]],
    device const uint* sample_positions [[buffer(4)]],
    device const uint* output_distribution_indices [[buffer(5)]],
    device int* proposal_token_ids [[buffer(6)]],
    device float* proposal_probs [[buffer(7)]],
    device int* distribution_token_ids [[buffer(8)]],
    device float* distribution_probs [[buffer(9)]],
    constant uint& num_active_requests [[buffer(10)]],
    constant uint& num_steps [[buffer(11)]],
    constant uint& top_k [[buffer(12)]],
    constant uint& max_distribution_k [[buffer(13)]],
    constant uint& sampling_domain [[buffer(14)]],
    uint request [[thread_position_in_grid]]
) {
    if (request >= num_active_requests) {
        return;
    }
    const sampling_params request_params = params[req_slots[request]];
    uint previous = 0u;
    for (uint step = 0u; step < num_steps; ++step) {
        const ulong proposal = (ulong)request * (ulong)num_steps + (ulong)step;
        const ulong candidate_base = proposal * (ulong)top_k;
        const ulong score_base = (candidate_base + (ulong)previous) * (ulong)top_k;
        float maximum = -INFINITY;
        uint maximum_index = 0u;
        for (uint candidate = 0u; candidate < top_k; ++candidate) {
            const float score = scores[score_base + (ulong)candidate];
            if (score > maximum || (score == maximum && candidate < maximum_index)) {
                maximum = score;
                maximum_index = candidate;
            }
        }
        const uint distribution = output_distribution_indices[proposal];
        const ulong distribution_base = (ulong)distribution * (ulong)max_distribution_k;
        const bool greedy = request_params.temperature == 0.0f;
        float total = 0.0f;
        const float inverse_temperature = greedy ? 0.0f : 1.0f / request_params.temperature;
        for (uint candidate = 0u; candidate < top_k; ++candidate) {
            const int token = candidate_token_ids[candidate_base + (ulong)candidate];
            const float score = scores[score_base + (ulong)candidate];
            const float probability = greedy
                ? (candidate == maximum_index ? 1.0f : 0.0f)
                : metal::exp((score - maximum) * inverse_temperature);
            distribution_token_ids[distribution_base + (ulong)candidate] = token;
            distribution_probs[distribution_base + (ulong)candidate] = probability;
            total += probability;
        }
        for (uint candidate = top_k; candidate < max_distribution_k; ++candidate) {
            distribution_token_ids[distribution_base + (ulong)candidate] = -1;
            distribution_probs[distribution_base + (ulong)candidate] = 0.0f;
        }
        uint selected = greedy ? maximum_index : top_k - 1u;
        if (!greedy) {
            const uint random = khalamari_sampling_random(
                request_params.seed, sample_positions[request] + step, sampling_domain);
            const float draw = khalamari_uniform01(random);
            float cumulative = 0.0f;
            bool has_selected = false;
            for (uint candidate = 0u; candidate < top_k; ++candidate) {
                const ulong probability_index = distribution_base + (ulong)candidate;
                const float probability = distribution_probs[probability_index] / total;
                distribution_probs[probability_index] = probability;
                cumulative += probability;
                if (!has_selected && cumulative >= draw) {
                    selected = candidate;
                    has_selected = true;
                }
            }
        }
        proposal_token_ids[proposal] = candidate_token_ids[candidate_base + (ulong)selected];
        proposal_probs[proposal] = distribution_probs[distribution_base + (ulong)selected];
        previous = selected;
    }
}
