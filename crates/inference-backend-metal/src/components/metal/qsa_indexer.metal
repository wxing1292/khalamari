#include <metal_stdlib>
using namespace metal;

// Host ABI: qsa_indexer::Token. Ranges are half-open.
struct Token {
    uint batch_start, token_index, req_slot, src_state_slot, dst_state_slot;
    uint group_start;
};
struct Candidate { float score; uint group; };

inline uint num_visible_groups(Token token) {
    return (token.token_index + 1) / ratio - token.group_start;
}

inline ulong key_offset(device const uint* page_ids, Token token, uint group,
                        uint num_pages_per_request, uint groups_per_page) {
    const uint page = page_ids[ulong(token.req_slot) * num_pages_per_request + group / groups_per_page];
    return (ulong(page) * groups_per_page + group % groups_per_page) * dim;
}

inline float raw_key(device const scalar_t* qk, device const scalar_t* states,
                     Token token, uint flat_token, uint position, uint column) {
    const uint batch_position = token.token_index - (flat_token - token.batch_start);
    if (position < batch_position) {
        return float(states[(ulong(token.src_state_slot) * (ratio - 1) + position % ratio) * dim + column]);
    }
    const uint row = token.batch_start + (position - batch_position);
    return float(qk[(ulong(row) * (heads + 1) + heads) * dim + column]);
}

// Task: one query head or pooled-key row, derived from grid (head, flat token).
// The key task also stores the incomplete-group state after this token.
kernel void qsa_indexer_prepare(
    device const scalar_t* qk [[buffer(0)]],
    device const Token* tokens [[buffer(1)]],
    device const scalar_t* q_norm [[buffer(2)]],
    device const scalar_t* k_norm [[buffer(3)]],
    device scalar_t* states [[buffer(4)]],
    device scalar_t* pages [[buffer(5)]],
    device const uint* page_ids [[buffer(6)]],
    device scalar_t* q [[buffer(7)]],
    constant uint& num_active_tokens [[buffer(8)]],
    constant uint& num_pages_per_request [[buffer(9)]],
    constant uint& groups_per_page [[buffer(10)]],
    uint2 task [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    if (task.y >= num_active_tokens) return;
    const Token token = tokens[task.y];
    const bool is_key = task.x == heads;
    const uint group_start = token.token_index / ratio * ratio;
    const uint carry = (token.token_index + 1) % ratio;
    if (is_key && carry != 0) {
        for (uint column = tid; column < carry * dim; column += threads) {
            states[ulong(token.dst_state_slot) * (ratio - 1) * dim + column] =
                scalar_t(raw_key(qk, states, token, task.y, group_start + column / dim, column % dim));
        }
        return;
    }
    threadgroup float row[dim];
    threadgroup float sums[threads / 32];
    float square_sum = 0.0f;
    for (uint column = tid; column < dim; column += threads) {
        float value;
        if (is_key) {
            float sum = 0.0f;
            for (uint j = 0; j < ratio; ++j) sum += raw_key(qk, states, token, task.y, group_start + j, column);
            value = float(scalar_t(sum / float(ratio)));
        } else {
            value = float(qk[(ulong(task.y) * (heads + 1) + task.x) * dim + column]);
        }
        row[column] = value;
        square_sum += value * value;
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0) sums[simd] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < threads / 32 ? sums[lane] : 0.0f;
    const float inv_rms = precise::rsqrt(simd_sum(total) / float(dim) + norm_eps);
    device const scalar_t* weight = is_key ? k_norm : q_norm;
    for (uint column = tid; column < dim; column += threads) {
        row[column] = float(scalar_t(row[column] * inv_rms * (1.0f + float(weight[column]))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint position = is_key ? group_start : token.token_index;
    device scalar_t* output = is_key
        ? pages + key_offset(page_ids, token, token.token_index / ratio, num_pages_per_request, groups_per_page)
        : q + (ulong(task.y) * heads + task.x) * dim;
    for (uint column = tid; column < dim; column += threads) {
        if (column < rope_dim / 2) {
            const uint other = column + rope_dim / 2;
            const float angle = float(position) * pow(rope_theta, -2.0f * float(column) / float(rope_dim));
            const float c = float(scalar_t(cos(angle))), s = float(scalar_t(sin(angle)));
            const float x = row[column], y = row[other];
            output[column] = scalar_t(float(scalar_t(x * c)) - float(scalar_t(y * s)));
            output[other] = scalar_t(float(scalar_t(y * c)) + float(scalar_t(x * s)));
        } else if (column >= rope_dim) {
            output[column] = scalar_t(row[column]);
        }
    }
}

// Task: four index groups for one query, with one SIMDgroup per key.
// Each SIMDgroup reuses its key across all indexer query heads.
kernel void qsa_indexer_score(
    device const scalar_t* q [[buffer(0)]],
    device const Token* tokens [[buffer(1)]],
    device const scalar_t* pages [[buffer(2)]],
    device const uint* page_ids [[buffer(3)]],
    device float* scores [[buffer(4)]],
    constant uint& num_active_tokens [[buffer(5)]],
    constant uint& num_pages_per_request [[buffer(6)]],
    constant uint& groups_per_page [[buffer(7)]],
    constant uint& num_total_groups [[buffer(8)]],
    uint2 task [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    if (task.y >= num_active_tokens) return;
    const Token token = tokens[task.y];
    const uint relative_group = task.x * (threads / 32) + simd;
    const uint num_groups = num_visible_groups(token);
    if (num_groups <= top_k || relative_group >= num_groups) return;
    const uint group = token.group_start + relative_group;
    const ulong offset = key_offset(page_ids, token, group, num_pages_per_request, groups_per_page);
    float dots[heads];
    for (uint head = 0; head < heads; ++head) dots[head] = 0.0f;
    for (uint column = lane; column < dim; column += 32) {
        const float key = float(pages[offset + column]);
        for (uint head = 0; head < heads; ++head) dots[head] += key * float(q[(ulong(task.y) * heads + head) * dim + column]);
    }
    float score = 0.0f;
    for (uint head = 0; head < heads; ++head) score += max(simd_sum(dots[head]), 0.0f);
    if (lane == 0) scores[ulong(task.y) * num_total_groups + relative_group] = score * rsqrt(float(dim));
}

inline bool better(Candidate a, Candidate b) {
    return a.score > b.score || (a.score == b.score && a.group < b.group);
}

inline void compare_pairs(threadgroup Candidate* rows, uint count, uint distance, uint stage, uint tid) {
    for (uint index = tid; index < count; index += threads) {
        const uint other = index ^ distance;
        if (other > index) {
            const Candidate a = rows[index], b = rows[other];
            const bool descending = (index & stage) == 0;
            if (descending ? better(b, a) : better(a, b)) {
                rows[index] = b;
                rows[other] = a;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void store_selection(threadgroup const Candidate* rows, device Candidate* partials,
                            device uint* ids, uint2 task,
                            uint partition_stride, bool final, uint tid) {
    if (final) {
        for (uint index = tid; index < top_k; index += threads) ids[ulong(task.y) * top_k + index] = rows[index].group;
    } else {
        for (uint index = tid; index < candidate_capacity; index += threads)
            partials[(ulong(task.y) * partition_stride + task.x) * candidate_capacity + index] = rows[index];
    }
}

// Task: sort one contiguous score tile for one query. Keep enough local candidates
// to contain the global top-k. Candidate.group preserves the logical group ID.
kernel void qsa_indexer_select(
    device const float* scores [[buffer(0)]], device const Token* tokens [[buffer(1)]],
    device Candidate* partials [[buffer(2)]], device uint* ids [[buffer(3)]],
    constant uint& num_active_tokens [[buffer(4)]], constant uint& num_total_groups [[buffer(5)]],
    constant uint& num_partitions [[buffer(6)]],
    uint2 task [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (task.y >= num_active_tokens) return;
    const Token token = tokens[task.y];
    const uint num_groups = num_visible_groups(token);
    if (num_groups <= top_k) return;
    threadgroup Candidate rows[selection_tile];
    for (uint index = tid; index < selection_tile; index += threads) {
        const uint group = task.x * selection_tile + index;
        rows[index] = group < num_groups
            ? Candidate{scores[ulong(task.y) * num_total_groups + group], token.group_start + group}
            : Candidate{-INFINITY, UINT_MAX};
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stage = 2; stage <= selection_tile; stage *= 2)
        for (uint distance = stage / 2; distance > 0; distance /= 2)
            compare_pairs(rows, selection_tile, distance, stage, tid);
    store_selection(rows, partials, ids, task, num_partitions, num_partitions == 1, tid);
}

// Task: merge two sorted candidate lists. Reversing the second list makes the
// concatenation bitonic, so only log2(2 * candidate_capacity) compare stages remain.
kernel void qsa_indexer_merge(
    device const Candidate* input [[buffer(0)]], device Candidate* output [[buffer(1)]],
    device const Token* tokens [[buffer(2)]], device uint* ids [[buffer(3)]],
    constant uint& num_active_tokens [[buffer(4)]], constant uint& num_partitions [[buffer(5)]],
    constant uint& partition_stride [[buffer(6)]],
    uint2 task [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (task.y >= num_active_tokens) return;
    const Token token = tokens[task.y];
    const uint num_groups = num_visible_groups(token);
    if (num_groups <= top_k) return;
    threadgroup Candidate rows[2 * candidate_capacity];
    for (uint index = tid; index < 2 * candidate_capacity; index += threads) {
        const bool second = index >= candidate_capacity;
        const uint partition = task.x * 2 + uint(second);
        const uint source = second ? 2 * candidate_capacity - index - 1 : index;
        rows[index] = partition < num_partitions
            ? input[(ulong(task.y) * partition_stride + partition) * candidate_capacity + source]
            : Candidate{-INFINITY, UINT_MAX};
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint distance = candidate_capacity; distance > 0; distance /= 2)
        compare_pairs(rows, 2 * candidate_capacity, distance, 2 * candidate_capacity, tid);
    store_selection(rows, output, ids, task, partition_stride, num_partitions <= 2, tid);
}
