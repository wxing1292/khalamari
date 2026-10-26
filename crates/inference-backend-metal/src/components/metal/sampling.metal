#include <metal_stdlib>
using namespace metal;

#define THREADGROUP_SIZE 256
#define TOP_K_MAX 256
#define TILE_CURSOR_MAX 1024
#define NEG_INF -3.4028234663852886e38f

// The component owns this ABI and its CPU diagnostic formatter in failure.rs.
enum sampling_error_code : uint {
    sampling_success = 0,
    sampling_no_finite_candidates = 1,
    sampling_invalid_normalization = 2,
    sampling_invalid_parameters = 3,
    sampling_invalid_rejection_mass = 4,
};

struct sampling_status {
    uint code;
    float value;
    int token;
};

struct sampling_failure_info {
    uint row;
    uint req_slot;
    uint sample_position;
    uint sampling_domain;
    uint distribution_index;
    uint seed;
    uint top_k;
    float temperature;
    float top_p;
    float value;
    int token;
    uint target_start;
    uint target_end;
    uint draft_start;
    uint draft_end;
};

struct sampling_failure {
    atomic_uint code;
    sampling_failure_info info;
};
static_assert(sizeof(sampling_failure) == 64, "sampling failure ABI");

static inline sampling_failure_info failure_info(uint row) {
    sampling_failure_info info = {};
    info.row = row;
    info.req_slot = UINT_MAX;
    info.sample_position = UINT_MAX;
    info.distribution_index = UINT_MAX;
    info.token = -1;
    return info;
}

// No successful lane writes this record. The first failure remains intact
// even when later commands reuse the input and scratch buffers.
static inline void record_sampling_failure(
    device sampling_failure* failure, uint code, sampling_failure_info info
) {
    uint expected = 0;
    while (expected == 0) {
        if (atomic_compare_exchange_weak_explicit(
                &failure->code, &expected, code, memory_order_relaxed, memory_order_relaxed)) {
            failure->info = info;
            return;
        }
    }
}

struct sampling_params {
    float temperature;
    float top_p;
    uint seed;
    uint top_k;
};

static inline void record_top_k_failure(
    device sampling_failure* failure, sampling_status status, uint row,
    sampling_params params, uint req_slot, uint position, uint domain, uint distribution
) {
    sampling_failure_info info = failure_info(row);
    info.req_slot = req_slot;
    info.sample_position = position;
    info.sampling_domain = domain;
    info.distribution_index = distribution;
    info.seed = params.seed;
    info.top_k = params.top_k;
    info.temperature = params.temperature;
    info.top_p = params.top_p;
    info.value = status.value;
    info.token = status.token;
    record_sampling_failure(failure, status.code, info);
}

struct rejection_runtime_params {
    uint seed;
    uint sample_position;
    uint top_k;
    // Keeps each request row 16-byte aligned.
    uint padding;
};

// Keep reduction and bitonic in separate entry points. Static threadgroup
// allocations are pipeline-wide, so a runtime branch would charge the small-k
// path for both algorithms even when only reduction executes.
static inline bool rank_before(float left_logit, int left_token, float right_logit, int right_token) {
    if (left_token < 0) {
        return false;
    }
    if (right_token < 0) {
        return true;
    }
    return left_logit > right_logit || (left_logit == right_logit && left_token < right_token);
}

static inline uint psi_mix(uint h) {
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

constant uint SAMPLING_DOMAIN_TARGET = 0x243f6a88u;
constant uint SAMPLING_DOMAIN_ACCEPT = 0x13198a2eu;
constant uint SAMPLING_DOMAIN_RESAMPLE = 0x03707344u;

static inline uint psi_sampling_random(uint seed, uint sample_position, uint sampling_domain) {
    return psi_mix(seed ^ psi_mix(sample_position + 0x9e3779b9u) ^ sampling_domain);
}

static inline float psi_uniform01(uint random) {
    return (float(random & 0x00ffffffu) + 0.5f) * (1.0f / 16777216.0f);
}

template <bool MASKED>
static inline bool token_allowed(
    device const uint* token_mask,
    uint mask_words_per_row,
    uint row,
    uint token
) {
    if constexpr (!MASKED) {
        return true;
    }
    const ulong word = (ulong)row * mask_words_per_row + token / 32;
    return (token_mask[word] & (1u << (token % 32))) != 0;
}

template <typename T, bool MASKED>
static inline void top_k_logits_tile_reduction(
    device const T* logits,
    device const uint* token_mask,
    uint mask_words_per_row,
    device int* tile_token_ids,
    device float* tile_logits,
    uint vocab,
    uint top_k,
    uint vocab_tile_size,
    uint num_tiles,
    uint lane,
    uint row,
    uint tile,
    threadgroup float* top_logits,
    threadgroup int* top_tokens,
    threadgroup float* reduce_values,
    threadgroup int* reduce_tokens
) {
    const uint tile_start = tile * vocab_tile_size;
    const uint tile_end = metal::min(tile_start + vocab_tile_size, vocab);
    const ulong base = (ulong)row * (ulong)vocab;
    const ulong out_base = ((ulong)row * (ulong)num_tiles + (ulong)tile) * (ulong)top_k;
    for (uint slot = lane; slot < top_k; slot += THREADGROUP_SIZE) {
        top_logits[slot] = NEG_INF;
        top_tokens[slot] = -1;
        tile_token_ids[out_base + slot] = -1;
        tile_logits[out_base + slot] = NEG_INF;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One lane owns one vocabulary token in the fixed 256-token partition.
    // Keep its mask decision and logit in registers across the top-k selections.
    const uint token_index = tile_start + lane;
    float candidate_logit = NEG_INF;
    int candidate_token = -1;
    if (token_index < tile_end && token_allowed<MASKED>(token_mask, mask_words_per_row, row, token_index)) {
        float logit = float(logits[base + (ulong)token_index]);
        if (metal::isfinite(logit)) {
            candidate_logit = logit;
            candidate_token = int(token_index);
        }
    }
    for (uint out_slot = 0; out_slot < top_k; ++out_slot) {
        bool already_selected = false;
        for (uint previous = 0; previous < out_slot; ++previous) {
            if (top_tokens[previous] == candidate_token) {
                already_selected = true;
                break;
            }
        }
        float local_best_logit = already_selected ? NEG_INF : candidate_logit;
        int local_best_token = already_selected ? -1 : candidate_token;
        reduce_values[lane] = local_best_logit;
        reduce_tokens[lane] = local_best_token;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = THREADGROUP_SIZE / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                float other_logit = reduce_values[lane + stride];
                int other_token = reduce_tokens[lane + stride];
                float current_logit = reduce_values[lane];
                int current_token = reduce_tokens[lane];
                if (other_logit > current_logit ||
                    (other_logit == current_logit && other_token >= 0 &&
                     (current_token < 0 || other_token < current_token))) {
                    reduce_values[lane] = other_logit;
                    reduce_tokens[lane] = other_token;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            top_logits[out_slot] = reduce_values[0];
            top_tokens[out_slot] = reduce_tokens[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint slot = lane; slot < top_k; slot += THREADGROUP_SIZE) {
        tile_token_ids[out_base + slot] = top_tokens[slot];
        tile_logits[out_base + slot] = top_logits[slot];
    }
}

template <typename T, bool MASKED>
static inline void top_k_logits_tile_bitonic(
    device const T* logits,
    device const uint* token_mask,
    uint mask_words_per_row,
    device int* tile_token_ids,
    device float* tile_logits,
    uint vocab,
    uint top_k,
    uint vocab_tile_size,
    uint num_tiles,
    uint lane,
    uint row,
    uint tile,
    threadgroup float* values,
    threadgroup int* tokens
) {
    const uint tile_start = tile * vocab_tile_size;
    const uint tile_end = metal::min(tile_start + vocab_tile_size, vocab);
    const ulong base = (ulong)row * (ulong)vocab;
    const ulong out_base = ((ulong)row * (ulong)num_tiles + (ulong)tile) * (ulong)top_k;
    const uint token_index = tile_start + lane;
    if (token_index < tile_end && token_allowed<MASKED>(token_mask, mask_words_per_row, row, token_index)) {
        float logit = float(logits[base + (ulong)token_index]);
        values[lane] = metal::isfinite(logit) ? logit : NEG_INF;
        tokens[lane] = metal::isfinite(logit) ? int(token_index) : -1;
    } else {
        values[lane] = NEG_INF;
        tokens[lane] = -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= THREADGROUP_SIZE; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            uint ixj = lane ^ j;
            if (ixj > lane) {
                float left_value = values[lane];
                int left_token = tokens[lane];
                float right_value = values[ixj];
                int right_token = tokens[ixj];
                bool descending = (lane & k) == 0;
                bool should_swap = descending
                    ? rank_before(right_value, right_token, left_value, left_token)
                    : rank_before(left_value, left_token, right_value, right_token);
                if (should_swap) {
                    values[lane] = right_value;
                    tokens[lane] = right_token;
                    values[ixj] = left_value;
                    tokens[ixj] = left_token;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint slot = lane; slot < top_k; slot += THREADGROUP_SIZE) {
        tile_token_ids[out_base + slot] = tokens[slot];
        tile_logits[out_base + slot] = values[slot];
    }
}

kernel void top_k_logits_tiles(
    device const float* logits [[buffer(0)]],
    device int* tile_token_ids [[buffer(1)]],
    device float* tile_logits [[buffer(2)]],
    constant uint& num_active_threads_u [[buffer(3)]],
    constant uint& vocab_u [[buffer(4)]],
    constant uint& top_k_u [[buffer(5)]],
    constant uint& vocab_tile_size_u [[buffer(6)]],
    constant uint& num_tiles_u [[buffer(7)]],
    uint global_thread_id [[thread_position_in_grid]]
) {
    if (global_thread_id >= num_active_threads_u) {
        return;
    }
    uint lane = global_thread_id % THREADGROUP_SIZE;
    uint group = global_thread_id / THREADGROUP_SIZE;
    uint row = group / num_tiles_u;
    uint tile = group % num_tiles_u;
    threadgroup float top_logits[TOP_K_MAX];
    threadgroup int top_tokens[TOP_K_MAX];
    threadgroup float reduce_values[THREADGROUP_SIZE];
    threadgroup int reduce_tokens[THREADGROUP_SIZE];
    top_k_logits_tile_reduction<float, false>(
        logits, nullptr, 0, tile_token_ids, tile_logits, vocab_u, top_k_u, vocab_tile_size_u, num_tiles_u,
        lane, row, tile, top_logits, top_tokens, reduce_values, reduce_tokens);
}

kernel void top_k_logits_tiles_masked(
    device const float* logits [[buffer(0)]],
    device int* tile_token_ids [[buffer(1)]],
    device float* tile_logits [[buffer(2)]],
    constant uint& num_active_threads_u [[buffer(3)]],
    constant uint& vocab_u [[buffer(4)]],
    constant uint& top_k_u [[buffer(5)]],
    constant uint& vocab_tile_size_u [[buffer(6)]],
    constant uint& num_tiles_u [[buffer(7)]],
    device const uint* token_mask [[buffer(8)]],
    constant uint& mask_words_per_row [[buffer(9)]],
    uint global_thread_id [[thread_position_in_grid]]
) {
    if (global_thread_id >= num_active_threads_u) {
        return;
    }
    uint lane = global_thread_id % THREADGROUP_SIZE;
    uint group = global_thread_id / THREADGROUP_SIZE;
    uint row = group / num_tiles_u;
    uint tile = group % num_tiles_u;
    threadgroup float top_logits[TOP_K_MAX];
    threadgroup int top_tokens[TOP_K_MAX];
    threadgroup float reduce_values[THREADGROUP_SIZE];
    threadgroup int reduce_tokens[THREADGROUP_SIZE];
    top_k_logits_tile_reduction<float, true>(
        logits, token_mask, mask_words_per_row, tile_token_ids, tile_logits, vocab_u, top_k_u, vocab_tile_size_u, num_tiles_u,
        lane, row, tile, top_logits, top_tokens, reduce_values, reduce_tokens);
}

kernel void top_k_logits_tiles_bitonic(
    device const float* logits [[buffer(0)]],
    device int* tile_token_ids [[buffer(1)]],
    device float* tile_logits [[buffer(2)]],
    constant uint& num_active_threads_u [[buffer(3)]],
    constant uint& vocab_u [[buffer(4)]],
    constant uint& top_k_u [[buffer(5)]],
    constant uint& vocab_tile_size_u [[buffer(6)]],
    constant uint& num_tiles_u [[buffer(7)]],
    uint global_thread_id [[thread_position_in_grid]]
) {
    if (global_thread_id >= num_active_threads_u) {
        return;
    }
    uint lane = global_thread_id % THREADGROUP_SIZE;
    uint group = global_thread_id / THREADGROUP_SIZE;
    uint row = group / num_tiles_u;
    uint tile = group % num_tiles_u;
    threadgroup float values[THREADGROUP_SIZE];
    threadgroup int tokens[THREADGROUP_SIZE];
    top_k_logits_tile_bitonic<float, false>(
        logits, nullptr, 0, tile_token_ids, tile_logits, vocab_u, top_k_u, vocab_tile_size_u, num_tiles_u,
        lane, row, tile, values, tokens);
}

kernel void top_k_logits_tiles_bitonic_masked(
    device const float* logits [[buffer(0)]],
    device int* tile_token_ids [[buffer(1)]],
    device float* tile_logits [[buffer(2)]],
    constant uint& num_active_threads_u [[buffer(3)]],
    constant uint& vocab_u [[buffer(4)]],
    constant uint& top_k_u [[buffer(5)]],
    constant uint& vocab_tile_size_u [[buffer(6)]],
    constant uint& num_tiles_u [[buffer(7)]],
    device const uint* token_mask [[buffer(8)]],
    constant uint& mask_words_per_row [[buffer(9)]],
    uint global_thread_id [[thread_position_in_grid]]
) {
    if (global_thread_id >= num_active_threads_u) {
        return;
    }
    uint lane = global_thread_id % THREADGROUP_SIZE;
    uint group = global_thread_id / THREADGROUP_SIZE;
    uint row = group / num_tiles_u;
    uint tile = group % num_tiles_u;
    threadgroup float values[THREADGROUP_SIZE];
    threadgroup int tokens[THREADGROUP_SIZE];
    top_k_logits_tile_bitonic<float, true>(
        logits, token_mask, mask_words_per_row, tile_token_ids, tile_logits, vocab_u, top_k_u, vocab_tile_size_u, num_tiles_u,
        lane, row, tile, values, tokens);
}

kernel void top_k_logits_tiles_bf16(
    device const bfloat* logits [[buffer(0)]],
    device int* tile_token_ids [[buffer(1)]],
    device float* tile_logits [[buffer(2)]],
    constant uint& num_active_threads_u [[buffer(3)]],
    constant uint& vocab_u [[buffer(4)]],
    constant uint& top_k_u [[buffer(5)]],
    constant uint& vocab_tile_size_u [[buffer(6)]],
    constant uint& num_tiles_u [[buffer(7)]],
    uint global_thread_id [[thread_position_in_grid]]
) {
    if (global_thread_id >= num_active_threads_u) {
        return;
    }
    uint lane = global_thread_id % THREADGROUP_SIZE;
    uint group = global_thread_id / THREADGROUP_SIZE;
    uint row = group / num_tiles_u;
    uint tile = group % num_tiles_u;
    threadgroup float top_logits[TOP_K_MAX];
    threadgroup int top_tokens[TOP_K_MAX];
    threadgroup float reduce_values[THREADGROUP_SIZE];
    threadgroup int reduce_tokens[THREADGROUP_SIZE];
    top_k_logits_tile_reduction<bfloat, false>(
        logits, nullptr, 0, tile_token_ids, tile_logits, vocab_u, top_k_u, vocab_tile_size_u, num_tiles_u,
        lane, row, tile, top_logits, top_tokens, reduce_values, reduce_tokens);
}

kernel void top_k_logits_tiles_bf16_masked(
    device const bfloat* logits [[buffer(0)]],
    device int* tile_token_ids [[buffer(1)]],
    device float* tile_logits [[buffer(2)]],
    constant uint& num_active_threads_u [[buffer(3)]],
    constant uint& vocab_u [[buffer(4)]],
    constant uint& top_k_u [[buffer(5)]],
    constant uint& vocab_tile_size_u [[buffer(6)]],
    constant uint& num_tiles_u [[buffer(7)]],
    device const uint* token_mask [[buffer(8)]],
    constant uint& mask_words_per_row [[buffer(9)]],
    uint global_thread_id [[thread_position_in_grid]]
) {
    if (global_thread_id >= num_active_threads_u) {
        return;
    }
    uint lane = global_thread_id % THREADGROUP_SIZE;
    uint group = global_thread_id / THREADGROUP_SIZE;
    uint row = group / num_tiles_u;
    uint tile = group % num_tiles_u;
    threadgroup float top_logits[TOP_K_MAX];
    threadgroup int top_tokens[TOP_K_MAX];
    threadgroup float reduce_values[THREADGROUP_SIZE];
    threadgroup int reduce_tokens[THREADGROUP_SIZE];
    top_k_logits_tile_reduction<bfloat, true>(
        logits, token_mask, mask_words_per_row, tile_token_ids, tile_logits, vocab_u, top_k_u, vocab_tile_size_u, num_tiles_u,
        lane, row, tile, top_logits, top_tokens, reduce_values, reduce_tokens);
}

kernel void top_k_logits_tiles_bf16_bitonic(
    device const bfloat* logits [[buffer(0)]],
    device int* tile_token_ids [[buffer(1)]],
    device float* tile_logits [[buffer(2)]],
    constant uint& num_active_threads_u [[buffer(3)]],
    constant uint& vocab_u [[buffer(4)]],
    constant uint& top_k_u [[buffer(5)]],
    constant uint& vocab_tile_size_u [[buffer(6)]],
    constant uint& num_tiles_u [[buffer(7)]],
    uint global_thread_id [[thread_position_in_grid]]
) {
    if (global_thread_id >= num_active_threads_u) {
        return;
    }
    uint lane = global_thread_id % THREADGROUP_SIZE;
    uint group = global_thread_id / THREADGROUP_SIZE;
    uint row = group / num_tiles_u;
    uint tile = group % num_tiles_u;
    threadgroup float values[THREADGROUP_SIZE];
    threadgroup int tokens[THREADGROUP_SIZE];
    top_k_logits_tile_bitonic<bfloat, false>(
        logits, nullptr, 0, tile_token_ids, tile_logits, vocab_u, top_k_u, vocab_tile_size_u, num_tiles_u,
        lane, row, tile, values, tokens);
}

kernel void top_k_logits_tiles_bf16_bitonic_masked(
    device const bfloat* logits [[buffer(0)]],
    device int* tile_token_ids [[buffer(1)]],
    device float* tile_logits [[buffer(2)]],
    constant uint& num_active_threads_u [[buffer(3)]],
    constant uint& vocab_u [[buffer(4)]],
    constant uint& top_k_u [[buffer(5)]],
    constant uint& vocab_tile_size_u [[buffer(6)]],
    constant uint& num_tiles_u [[buffer(7)]],
    device const uint* token_mask [[buffer(8)]],
    constant uint& mask_words_per_row [[buffer(9)]],
    uint global_thread_id [[thread_position_in_grid]]
) {
    if (global_thread_id >= num_active_threads_u) {
        return;
    }
    uint lane = global_thread_id % THREADGROUP_SIZE;
    uint group = global_thread_id / THREADGROUP_SIZE;
    uint row = group / num_tiles_u;
    uint tile = group % num_tiles_u;
    threadgroup float values[THREADGROUP_SIZE];
    threadgroup int tokens[THREADGROUP_SIZE];
    top_k_logits_tile_bitonic<bfloat, true>(
        logits, token_mask, mask_words_per_row, tile_token_ids, tile_logits, vocab_u, top_k_u, vocab_tile_size_u, num_tiles_u,
        lane, row, tile, values, tokens);
}

static inline void merge_distribution(
    uint lane,
    uint row,
    uint top_k,
    uint num_tiles,
    uint tile_top_k,
    uint vocab_tile_size,
    float temperature,
    float top_p,
    device const int* tile_token_ids,
    device const float* tile_logits,
    threadgroup float* top_logits,
    threadgroup int* top_tokens,
    threadgroup float* weights,
    threadgroup float* reduce_values,
    threadgroup int* reduce_tokens,
    threadgroup ushort* tile_cursors,
    threadgroup sampling_status& status
) {
    if (lane == 0) {
        status = {sampling_success, 0.0f, -1};
    }
    for (uint slot = lane; slot < top_k; slot += THREADGROUP_SIZE) {
        top_logits[slot] = NEG_INF;
        top_tokens[slot] = -1;
        weights[slot] = 0.0f;
    }
    bool use_tile_cursors = num_tiles <= TILE_CURSOR_MAX;
    if (use_tile_cursors) {
        for (uint tile = lane; tile < num_tiles; tile += THREADGROUP_SIZE) {
            tile_cursors[tile] = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong candidate_base = (ulong)row * (ulong)num_tiles * (ulong)tile_top_k;
    for (uint out_slot = 0; out_slot < top_k; ++out_slot) {
        float local_best_logit = NEG_INF;
        int local_best_token = -1;
        for (uint tile = lane; tile < num_tiles; tile += THREADGROUP_SIZE) {
            uint cursor;
            if (use_tile_cursors) {
                cursor = uint(tile_cursors[tile]);
            } else {
                cursor = 0;
                for (uint previous = 0; previous < out_slot; ++previous) {
                    int previous_token = top_tokens[previous];
                    if (previous_token >= 0 && uint(previous_token) / vocab_tile_size == tile) {
                        cursor += 1;
                    }
                }
            }
            if (cursor >= tile_top_k) {
                continue;
            }
            const ulong candidate_index =
                candidate_base + (ulong)tile * (ulong)tile_top_k + (ulong)cursor;
            int token = tile_token_ids[candidate_index];
            float logit = tile_logits[candidate_index];
            if (token < 0 || !metal::isfinite(logit)) {
                continue;
            }
            if (logit > local_best_logit ||
                (logit == local_best_logit && (local_best_token < 0 || token < local_best_token))) {
                local_best_logit = logit;
                local_best_token = token;
            }
        }
        reduce_values[lane] = local_best_logit;
        reduce_tokens[lane] = local_best_token;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = THREADGROUP_SIZE / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                float other_logit = reduce_values[lane + stride];
                int other_token = reduce_tokens[lane + stride];
                float current_logit = reduce_values[lane];
                int current_token = reduce_tokens[lane];
                if (other_logit > current_logit ||
                    (other_logit == current_logit && other_token >= 0 &&
                     (current_token < 0 || other_token < current_token))) {
                    reduce_values[lane] = other_logit;
                    reduce_tokens[lane] = other_token;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            top_logits[out_slot] = reduce_values[0];
            top_tokens[out_slot] = reduce_tokens[0];
            if (use_tile_cursors && reduce_tokens[0] >= 0) {
                const uint selected_tile = uint(reduce_tokens[0]) / vocab_tile_size;
                if (selected_tile < num_tiles) {
                    tile_cursors[selected_tile] = ushort(metal::min(uint(tile_cursors[selected_tile]) + 1, tile_top_k));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        if (top_tokens[0] < 0 || !metal::isfinite(top_logits[0])) {
            status = {sampling_no_finite_candidates, top_logits[0], top_tokens[0]};
            return;
        }
        bool greedy = (temperature == 0.0f) || (top_k == 1) || (top_p == 0.0f);
        if (greedy) {
            weights[0] = 1.0f;
            return;
        }
        float temp = metal::max(temperature, 1.0e-6f);
        float max_scaled = top_logits[0] / temp;
        float total = 0.0f;
        for (uint slot = 0; slot < top_k; ++slot) {
            weights[slot] = (top_tokens[slot] >= 0 && metal::isfinite(top_logits[slot]))
                ? metal::exp((top_logits[slot] / temp) - max_scaled)
                : 0.0f;
            total += weights[slot];
        }
        if (!(total > 0.0f) || !metal::isfinite(total)) {
            status = {sampling_invalid_normalization, total, top_tokens[0]};
            return;
        }
        float kept_total = 0.0f;
        uint kept_count = 0;
        for (uint slot = 0; slot < top_k; ++slot) {
            kept_total += weights[slot];
            kept_count = slot + 1;
            if (top_p < 1.0f && kept_total >= top_p * total) {
                break;
            }
        }
        for (uint slot = 0; slot < kept_count; ++slot) {
            weights[slot] = weights[slot] / kept_total;
        }
        for (uint slot = kept_count; slot < top_k; ++slot) {
            top_tokens[slot] = -1;
            weights[slot] = 0.0f;
        }
    }
}

static inline void sample_merged_distribution(
    uint row,
    uint top_k,
    sampling_params params,
    uint sample_position,
    uint sampling_domain,
    threadgroup const int* top_tokens,
    threadgroup const float* weights,
    device int* sampled_token_ids,
    device float* sampled_token_probs
) {
    sampled_token_ids[row] = top_tokens[0];
    sampled_token_probs[row] = weights[0];
    if (params.temperature == 0.0f || top_k == 1 || params.top_p == 0.0f) {
        return;
    }
    uint kept_count = 0;
    float kept_total = 0.0f;
    for (uint slot = 0; slot < top_k; ++slot) {
        if (top_tokens[slot] < 0 || weights[slot] <= 0.0f) {
            break;
        }
        kept_total += weights[slot];
        kept_count += 1;
    }
    if (kept_count <= 0 || kept_total <= 0.0f) {
        return;
    }
    uint random = psi_sampling_random(params.seed, sample_position, sampling_domain);
    float draw = psi_uniform01(random) * kept_total;
    float cumulative = 0.0f;
    for (uint slot = 0; slot < kept_count; ++slot) {
        cumulative += weights[slot];
        if (cumulative >= draw) {
            sampled_token_ids[row] = top_tokens[slot];
            sampled_token_probs[row] = weights[slot];
            return;
        }
    }
    sampled_token_ids[row] = top_tokens[kept_count - 1];
    sampled_token_probs[row] = weights[kept_count - 1];
}

kernel void top_k_merge_tiles(
    device const int* tile_token_ids [[buffer(0)]],
    device const float* tile_logits [[buffer(1)]],
    device int* token_ids [[buffer(2)]],
    device float* logits [[buffer(3)]],
    constant uint& num_active_threads [[buffer(4)]],
    constant uint& top_k [[buffer(5)]],
    constant uint& num_tiles [[buffer(6)]],
    constant uint& tile_top_k [[buffer(7)]],
    constant uint& vocab_tile_size [[buffer(8)]],
    device sampling_failure* failure [[buffer(30)]],
    uint global_thread_id [[thread_position_in_grid]]
) {
    if (global_thread_id >= num_active_threads || top_k == 0u || top_k > TOP_K_MAX) {
        return;
    }
    const uint lane = global_thread_id % THREADGROUP_SIZE;
    const uint row = global_thread_id / THREADGROUP_SIZE;
    threadgroup float top_logits[TOP_K_MAX];
    threadgroup int top_tokens[TOP_K_MAX];
    threadgroup float weights[TOP_K_MAX];
    threadgroup float reduce_values[THREADGROUP_SIZE];
    threadgroup int reduce_tokens[THREADGROUP_SIZE];
    threadgroup ushort tile_cursors[TILE_CURSOR_MAX];
    threadgroup sampling_status status;
    merge_distribution(
        lane, row, top_k, num_tiles, tile_top_k, vocab_tile_size, 1.0f, 1.0f,
        tile_token_ids, tile_logits, top_logits, top_tokens, weights,
        reduce_values, reduce_tokens, tile_cursors, status);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (status.code != sampling_success) {
        if (lane == 0) {
            record_top_k_failure(failure, status, row, sampling_params{1.0f, 1.0f, 0, top_k}, UINT_MAX, UINT_MAX, 0, UINT_MAX);
        }
        return;
    }
    const ulong output_base = (ulong)row * (ulong)top_k;
    for (uint slot = lane; slot < top_k; slot += THREADGROUP_SIZE) {
        token_ids[output_base + (ulong)slot] = top_tokens[slot];
        logits[output_base + (ulong)slot] = top_logits[slot];
    }
}

kernel void top_k_sample_tiles(
    device const int* tile_token_ids [[buffer(0)]],
    device const float* tile_logits [[buffer(1)]],
    device int* token_ids [[buffer(2)]],
    device float* token_probs [[buffer(3)]],
    device const sampling_params* params [[buffer(4)]],
    device const uint* req_slots [[buffer(5)]],
    device const uint* sample_positions [[buffer(6)]],
    constant uint& num_active_threads_u [[buffer(7)]],
    constant uint& sample_position_increment [[buffer(8)]],
    constant uint& sampling_domain [[buffer(9)]],
    constant uint& top_k_u [[buffer(10)]],
    constant uint& num_tiles_u [[buffer(11)]],
    constant uint& tile_top_k_u [[buffer(12)]],
    constant uint& vocab_tile_size_u [[buffer(13)]],
    device sampling_failure* failure [[buffer(30)]],
    uint global_thread_id [[thread_position_in_grid]]
) {
    if (global_thread_id >= num_active_threads_u) {
        return;
    }
    uint lane = global_thread_id % THREADGROUP_SIZE;
    uint row = global_thread_id / THREADGROUP_SIZE;
    threadgroup float top_logits[TOP_K_MAX];
    threadgroup int top_tokens[TOP_K_MAX];
    threadgroup float weights[TOP_K_MAX];
    threadgroup float reduce_values[THREADGROUP_SIZE];
    threadgroup int reduce_tokens[THREADGROUP_SIZE];
    threadgroup ushort tile_cursors[TILE_CURSOR_MAX];
    threadgroup sampling_status status;
    sampling_params request_params = params[req_slots[row]];
    uint row_top_k = request_params.top_k;
    if (row_top_k == 0 || row_top_k > top_k_u) {
        if (lane == 0) {
            record_top_k_failure(failure, {sampling_invalid_parameters, 0.0f, -1}, row, request_params, req_slots[row], sample_positions[row] + sample_position_increment, sampling_domain, UINT_MAX);
        }
        return;
    }
    merge_distribution(lane, row, row_top_k, num_tiles_u, tile_top_k_u,
        vocab_tile_size_u, request_params.temperature, request_params.top_p, tile_token_ids, tile_logits, top_logits, top_tokens, weights,
        reduce_values, reduce_tokens, tile_cursors, status);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (status.code != sampling_success) {
        if (lane == 0) {
            record_top_k_failure(failure, status, row, request_params, req_slots[row], sample_positions[row] + sample_position_increment, sampling_domain, UINT_MAX);
        }
        return;
    }
    if (lane == 0) {
        sample_merged_distribution(
            row, row_top_k, request_params, sample_positions[row] + sample_position_increment,
            sampling_domain, top_tokens, weights, token_ids, token_probs);
    }
}

kernel void top_k_write_distribution_tiles(
    device const int* tile_token_ids [[buffer(0)]],
    device const float* tile_logits [[buffer(1)]],
    device int* token_ids [[buffer(2)]],
    device float* token_probs [[buffer(3)]],
    device const sampling_params* params [[buffer(4)]],
    device const uint* req_slots [[buffer(5)]],
    device const uint* output_distribution_indices [[buffer(6)]],
    constant uint& num_active_threads_u [[buffer(7)]],
    constant uint& top_k_u [[buffer(8)]],
    constant uint& num_tiles_u [[buffer(9)]],
    constant uint& tile_top_k_u [[buffer(10)]],
    constant uint& vocab_tile_size_u [[buffer(11)]],
    constant uint& max_k_u [[buffer(12)]],
    constant uint& num_output_distributions_u [[buffer(13)]],
    device sampling_failure* failure [[buffer(30)]],
    uint global_thread_id [[thread_position_in_grid]]
) {
    if (global_thread_id >= num_active_threads_u) {
        return;
    }
    uint lane = global_thread_id % THREADGROUP_SIZE;
    uint row = global_thread_id / THREADGROUP_SIZE;

    threadgroup float top_logits[TOP_K_MAX];
    threadgroup int top_tokens[TOP_K_MAX];
    threadgroup float weights[TOP_K_MAX];
    threadgroup float reduce_values[THREADGROUP_SIZE];
    threadgroup int reduce_tokens[THREADGROUP_SIZE];
    threadgroup ushort tile_cursors[TILE_CURSOR_MAX];
    threadgroup sampling_status status;
    sampling_params request_params = params[req_slots[row]];
    uint row_top_k = request_params.top_k;
    if (row_top_k == 0 || row_top_k > top_k_u) {
        if (lane == 0) {
            record_top_k_failure(failure, {sampling_invalid_parameters, 0.0f, -1}, row, request_params, req_slots[row], UINT_MAX, 0, output_distribution_indices[row]);
        }
        return;
    }
    merge_distribution(lane, row, row_top_k, num_tiles_u, tile_top_k_u,
        vocab_tile_size_u, request_params.temperature, request_params.top_p, tile_token_ids, tile_logits, top_logits, top_tokens, weights,
        reduce_values, reduce_tokens, tile_cursors, status);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (status.code != sampling_success) {
        if (lane == 0) {
            record_top_k_failure(failure, status, row, request_params, req_slots[row], UINT_MAX, 0, output_distribution_indices[row]);
        }
        return;
    }

    uint output_distribution_index = output_distribution_indices[row];
    if (output_distribution_index >= num_output_distributions_u) {
        return;
    }
    const ulong output_base =
        (ulong)output_distribution_index * (ulong)max_k_u;
    for (uint slot = lane; slot < row_top_k; slot += THREADGROUP_SIZE) {
        token_ids[output_base + slot] = top_tokens[slot];
        token_probs[output_base + slot] = weights[slot];
    }
}

kernel void top_k_sample_and_write_distribution_tiles(
    device const int* tile_token_ids [[buffer(0)]],
    device const float* tile_logits [[buffer(1)]],
    device int* sampled_token_ids [[buffer(2)]],
    device float* sampled_token_probs [[buffer(3)]],
    device int* distribution_token_ids [[buffer(4)]],
    device float* distribution_probs [[buffer(5)]],
    device const sampling_params* params [[buffer(6)]],
    device const uint* req_slots [[buffer(7)]],
    device const uint* sample_positions [[buffer(8)]],
    device const uint* output_distribution_indices [[buffer(9)]],
    constant uint& num_active_threads_u [[buffer(10)]],
    constant uint& sample_position_increment [[buffer(11)]],
    constant uint& sampling_domain [[buffer(12)]],
    constant uint& top_k_u [[buffer(13)]],
    constant uint& num_tiles_u [[buffer(14)]],
    constant uint& tile_top_k_u [[buffer(15)]],
    constant uint& vocab_tile_size_u [[buffer(16)]],
    constant uint& max_k_u [[buffer(17)]],
    constant uint& num_output_distributions_u [[buffer(18)]],
    device sampling_failure* failure [[buffer(30)]],
    uint global_thread_id [[thread_position_in_grid]]
) {
    if (global_thread_id >= num_active_threads_u) {
        return;
    }
    uint lane = global_thread_id % THREADGROUP_SIZE;
    uint row = global_thread_id / THREADGROUP_SIZE;

    threadgroup float top_logits[TOP_K_MAX];
    threadgroup int top_tokens[TOP_K_MAX];
    threadgroup float weights[TOP_K_MAX];
    threadgroup float reduce_values[THREADGROUP_SIZE];
    threadgroup int reduce_tokens[THREADGROUP_SIZE];
    threadgroup ushort tile_cursors[TILE_CURSOR_MAX];
    threadgroup sampling_status status;
    sampling_params request_params = params[req_slots[row]];
    uint row_top_k = request_params.top_k;
    if (row_top_k == 0 || row_top_k > top_k_u) {
        if (lane == 0) {
            record_top_k_failure(failure, {sampling_invalid_parameters, 0.0f, -1}, row, request_params, req_slots[row], sample_positions[row] + sample_position_increment, sampling_domain, output_distribution_indices[row]);
        }
        return;
    }
    merge_distribution(lane, row, row_top_k, num_tiles_u, tile_top_k_u,
        vocab_tile_size_u, request_params.temperature, request_params.top_p, tile_token_ids, tile_logits, top_logits, top_tokens, weights,
        reduce_values, reduce_tokens, tile_cursors, status);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (status.code != sampling_success) {
        if (lane == 0) {
            record_top_k_failure(failure, status, row, request_params, req_slots[row], sample_positions[row] + sample_position_increment, sampling_domain, output_distribution_indices[row]);
        }
        return;
    }

    uint output_distribution_index = output_distribution_indices[row];
    if (output_distribution_index < num_output_distributions_u) {
        const ulong output_base =
            (ulong)output_distribution_index * (ulong)max_k_u;
        for (uint slot = lane; slot < row_top_k; slot += THREADGROUP_SIZE) {
            distribution_token_ids[output_base + slot] = top_tokens[slot];
            distribution_probs[output_base + slot] = weights[slot];
        }
    }

    if (lane == 0) {
        sample_merged_distribution(
            row, row_top_k, request_params, sample_positions[row] + sample_position_increment,
            sampling_domain, top_tokens, weights, sampled_token_ids, sampled_token_probs);
    }
}

static inline float write_distribution_prob(
    device const int* token_ids,
    device const float* token_probs,
    uint distribution_index,
    bool has_distribution,
    uint max_k,
    uint top_k,
    int token
) {
    if (!has_distribution || token < 0) {
        return 0.0f;
    }
    const ulong base = (ulong)distribution_index * (ulong)max_k;
    for (uint slot = 0; slot < top_k; ++slot) {
        if (token_ids[base + slot] == token) {
            return metal::max(token_probs[base + slot], 0.0f);
        }
    }
    return 0.0f;
}

static inline void sparse_sample_residual(
    device const int* target_token_ids,
    device const float* target_probs,
    device const int* flat_draft_token_ids,
    device const float* draft_probs,
    device int* sampled_token_ids,
    device float* sampled_token_probs,
    uint req,
    uint target_distribution_index,
    uint draft_distribution_index,
    bool has_draft_distribution,
    uint max_target_k,
    uint max_draft_k,
    uint top_k,
    float uniform,
    device sampling_failure* failure,
    sampling_failure_info info
) {
    sampled_token_ids[req] = 0;
    sampled_token_probs[req] = 0.0f;
    const ulong target_base =
        (ulong)target_distribution_index * (ulong)max_target_k;
    float total = 0.0f;
    for (uint slot = 0; slot < top_k; ++slot) {
        int token = target_token_ids[target_base + slot];
        if (token < 0) {
            continue;
        }
        float q = metal::max(target_probs[target_base + slot], 0.0f);
        float p = write_distribution_prob(
            flat_draft_token_ids, draft_probs, draft_distribution_index, has_draft_distribution,
            max_draft_k, top_k, token);
        total += metal::max(q - p, 0.0f);
    }
    if (!(total > 0.0f) || !metal::isfinite(total)) {
        info.distribution_index = target_distribution_index;
        info.value = total;
        record_sampling_failure(failure, sampling_invalid_rejection_mass, info);
        return;
    }

    float draw = uniform * total;
    float cumulative = 0.0f;
    int fallback_token = 0;
    float fallback_prob = 0.0f;
    for (uint slot = 0; slot < top_k; ++slot) {
        int token = target_token_ids[target_base + slot];
        if (token < 0) {
            continue;
        }
        float q = metal::max(target_probs[target_base + slot], 0.0f);
        float p = write_distribution_prob(
            flat_draft_token_ids, draft_probs, draft_distribution_index, has_draft_distribution,
            max_draft_k, top_k, token);
        float mass = metal::max(q - p, 0.0f);
        if (mass <= 0.0f) {
            continue;
        }
        fallback_token = token;
        fallback_prob = q;
        cumulative += mass;
        if (cumulative >= draw) {
            sampled_token_ids[req] = token;
            sampled_token_probs[req] = q;
            return;
        }
    }
    sampled_token_ids[req] = fallback_token;
    sampled_token_probs[req] = fallback_prob;
}

kernel void rejection_sparse_sample(
    device const int* target_distribution_token_ids [[buffer(0)]],
    device const float* target_distribution_probs [[buffer(1)]],
    device const int* draft_distribution_token_ids [[buffer(2)]],
    device const float* draft_distribution_probs [[buffer(3)]],
    device const int* flat_draft_token_ids [[buffer(4)]],
    device const uint* cu_target_distributions [[buffer(5)]],
    device const uint* cu_draft_distributions [[buffer(6)]],
    device int* flat_accepted_token_ids [[buffer(7)]],
    device float* flat_accepted_probs [[buffer(8)]],
    device uint* num_accepted_tokens [[buffer(9)]],
    device int* sampled_token_ids [[buffer(10)]],
    device float* sampled_token_probs [[buffer(11)]],
    device const rejection_runtime_params* runtime_params [[buffer(12)]],
    device const uint* flat_draft_distribution_indices [[buffer(13)]],
    constant uint& num_active_threads_u [[buffer(14)]],
    constant uint& num_target_distributions_u [[buffer(15)]],
    constant uint& num_draft_distributions_u [[buffer(16)]],
    constant uint& top_k_u [[buffer(17)]],
    constant uint& max_target_k_u [[buffer(18)]],
    constant uint& max_draft_k_u [[buffer(19)]],
    device sampling_failure* failure [[buffer(30)]],
    uint global_thread_id [[thread_position_in_grid]]
) {
    if (global_thread_id >= num_active_threads_u) {
        return;
    }
    uint lane = global_thread_id % THREADGROUP_SIZE;
    uint req = global_thread_id / THREADGROUP_SIZE;
    if (lane != 0) {
        return;
    }

    rejection_runtime_params params = runtime_params[req];
    const uint top_k = params.top_k;
    const uint target_start = cu_target_distributions[req];
    const uint target_end = cu_target_distributions[req + 1];
    const uint draft_start = cu_draft_distributions[req];
    const uint draft_end = cu_draft_distributions[req + 1];
    sampling_failure_info info = failure_info(req);
    info.seed = params.seed;
    info.sample_position = params.sample_position;
    info.top_k = top_k;
    info.target_start = target_start;
    info.target_end = target_end;
    info.draft_start = draft_start;
    info.draft_end = draft_end;

    num_accepted_tokens[req] = 0;
    sampled_token_ids[req] = 0;
    sampled_token_probs[req] = 0.0f;
    if (top_k == 0 || top_k > top_k_u || top_k > TOP_K_MAX || target_start > target_end ||
        draft_start > draft_end || target_end > num_target_distributions_u ||
        draft_end > num_draft_distributions_u || target_end - target_start != draft_end - draft_start + 1) {
        record_sampling_failure(failure, sampling_invalid_parameters, info);
        return;
    }
    const uint draft_len = draft_end - draft_start;

    for (uint offset = 0; offset < draft_len; ++offset) {
        const uint flat_draft_index = draft_start + offset;
        const uint draft_distribution_index = flat_draft_distribution_indices[flat_draft_index];
        const uint target_distribution_index = target_start + offset;
        int draft_token = flat_draft_token_ids[flat_draft_index];
        info.sample_position = params.sample_position + offset;
        info.distribution_index = target_distribution_index;
        info.token = draft_token;
        if (draft_token < 0) {
            record_sampling_failure(failure, sampling_invalid_parameters, info);
            return;
        }

        float q = write_distribution_prob(
            target_distribution_token_ids, target_distribution_probs, target_distribution_index, true,
            max_target_k_u, top_k, draft_token);
        float p = write_distribution_prob(
            draft_distribution_token_ids, draft_distribution_probs, draft_distribution_index, true,
            max_draft_k_u, top_k, draft_token);
        float accept_prob = p > 0.0f ? metal::min(q / p, 1.0f) : 0.0f;
        uint sample_position = params.sample_position + offset;
        float u = psi_uniform01(psi_sampling_random(params.seed, sample_position, SAMPLING_DOMAIN_ACCEPT));
        if (u <= accept_prob) {
            flat_accepted_token_ids[flat_draft_index] = draft_token;
            flat_accepted_probs[flat_draft_index] = q;
            num_accepted_tokens[req] = offset + 1;
            continue;
        }

        num_accepted_tokens[req] = offset;
        info.sampling_domain = SAMPLING_DOMAIN_RESAMPLE;
        float residual_uniform =
            psi_uniform01(psi_sampling_random(params.seed, sample_position, SAMPLING_DOMAIN_RESAMPLE));
        sparse_sample_residual(
            target_distribution_token_ids, target_distribution_probs, draft_distribution_token_ids, draft_distribution_probs,
            sampled_token_ids, sampled_token_probs, req, target_distribution_index, draft_distribution_index, true,
            max_target_k_u, max_draft_k_u, top_k, residual_uniform, failure, info);
        return;
    }

    const uint final_target_distribution_index = target_start + draft_len;
    uint final_sample_position = params.sample_position + draft_len;
    info.sample_position = final_sample_position;
    info.sampling_domain = SAMPLING_DOMAIN_TARGET;
    float final_uniform =
        psi_uniform01(psi_sampling_random(params.seed, final_sample_position, SAMPLING_DOMAIN_TARGET));
    sparse_sample_residual(
        target_distribution_token_ids, target_distribution_probs, draft_distribution_token_ids, draft_distribution_probs,
        sampled_token_ids, sampled_token_probs, req, final_target_distribution_index, 0, false,
        max_target_k_u, max_draft_k_u, top_k, final_uniform, failure, info);
}
