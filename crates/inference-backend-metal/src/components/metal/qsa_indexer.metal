#include <metal_stdlib>
using namespace metal;

// Host ABI: qsa_indexer::Token. Ranges are half-open.
struct Token {
    uint batch_start, token_index, req_slot, src_state_slot, dst_state_slot;
    uint group_start;
};
struct ScoreTaskTemplate {
    uint query_start, num_queries;
};

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
    device const ScoreTaskTemplate* score_tasks [[buffer(2)]],
    device const scalar_t* pages [[buffer(3)]],
    device const uint* page_ids [[buffer(4)]],
    device float* scores [[buffer(5)]],
    constant uint& num_active_score_task_templates [[buffer(6)]],
    constant uint& num_pages_per_request [[buffer(7)]],
    constant uint& groups_per_page [[buffer(8)]],
    constant uint& num_total_groups [[buffer(9)]],
    uint2 task [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    if (task.y >= num_active_score_task_templates) return;
    const ScoreTaskTemplate score_task = score_tasks[task.y];
    const uint query = score_task.query_start;
    const Token token = tokens[query];
    const uint relative_group = task.x * (threads / 32) + simd;
    const uint num_groups = num_visible_groups(token);
    if (relative_group >= num_groups) return;
    const uint group = token.group_start + relative_group;
    const ulong offset = key_offset(page_ids, token, group, num_pages_per_request, groups_per_page);
    float dots[heads];
    for (uint head = 0; head < heads; ++head) dots[head] = 0.0f;
    for (uint column = lane; column < dim; column += 32) {
        const float key = float(pages[offset + column]);
        for (uint head = 0; head < heads; ++head) dots[head] += key * float(q[(ulong(query) * heads + head) * dim + column]);
    }
    float score = 0.0f;
    for (uint head = 0; head < heads; ++head) score += max(simd_sum(dots[head]), 0.0f);
    if (lane == 0) scores[ulong(query) * num_total_groups + relative_group] = score * rsqrt(float(dim));
}

// Copyright 2026 TensorFold contributors.
// Adapted from TensorFold 0.6.6 block_select.py (Apache-2.0).
// Scores are nonnegative FP32. Unsigned bit patterns preserve their order.
// SIMDgroup 0 scans 256 histogram bins; each lane owns eight descending bins.
inline void radix_cut(threadgroup atomic_uint* hist, uint lane, uint prefix, int shift, uint need,
                      threadgroup uint& cut, threadgroup uint& rest) {
    uint counts[8], count = 0;
    for (uint i = 0; i < 8; ++i) {
        counts[i] = atomic_load_explicit(&hist[255u - 8u * lane - i], memory_order_relaxed);
        count += counts[i];
    }
    uint above = simd_prefix_exclusive_sum(count);
    if (above < need && above + count >= need) {
        for (uint i = 0; i < 8; ++i) {
            if (above + counts[i] >= need) {
                cut = prefix | ((255u - 8u * lane - i) << uint(shift));
                rest = need - above;
                break;
            }
            above += counts[i];
        }
    }
}

// A lane owns a contiguous interval, so the two-level scan preserves group order.
// COMPACT reads only SMEM candidates. Otherwise every pass reads visible GMEM scores.
template <bool COMPACT>
inline void radix_select(device const float* scores, threadgroup const uint* compact_keys,
                         threadgroup const uint* compact_ids, uint count, uint group_start,
                         device uint* ids, uint tid, threadgroup atomic_uint* hist,
                         threadgroup uint* state, threadgroup uint* greater, threadgroup uint* equal) {
    const uint lane = tid % 32u, simd = tid / 32u;
    constexpr uint simdgroups = selection_threads / 32u;
    uint prefix = 0, mask = 0, need = top_k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < count; i += selection_threads) {
            const uint key = COMPACT ? compact_keys[i] : as_type<uint>(scores[i]);
            if ((key & mask) == prefix)
                atomic_fetch_add_explicit(&hist[(key >> uint(shift)) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) radix_cut(hist, lane, prefix, shift, need, state[0], state[1]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = state[0];
        need = state[1];
        mask |= 255u << uint(shift);
    }
    const uint width = count / selection_threads + uint(count % selection_threads != 0u);
    const uint start = uint(min(ulong(count), ulong(tid) * width)), end = start + min(width, count - start);
    uint num_greater = 0, num_equal = 0;
    for (uint i = start; i < end; ++i) {
        const uint key = COMPACT ? compact_keys[i] : as_type<uint>(scores[i]);
        num_greater += uint(key > prefix);
        num_equal += uint(key == prefix);
    }
    uint pre_greater = simd_prefix_exclusive_sum(num_greater);
    uint pre_equal = simd_prefix_exclusive_sum(num_equal);
    if (lane == 31) {
        greater[simd] = pre_greater + num_greater;
        equal[simd] = pre_equal + num_equal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        const uint g = lane < simdgroups ? greater[lane] : 0u;
        const uint e = lane < simdgroups ? equal[lane] : 0u;
        const uint pg = simd_prefix_exclusive_sum(g), pe = simd_prefix_exclusive_sum(e);
        if (lane < simdgroups) { greater[lane] = pg; equal[lane] = pe; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pre_greater += greater[simd];
    pre_equal += equal[simd];
    uint output = pre_greater + min(pre_equal, need);
    for (uint i = start; i < end; ++i) {
        const uint key = COMPACT ? compact_keys[i] : as_type<uint>(scores[i]);
        bool take = key > prefix;
        if (key == prefix) { take = pre_equal < need; ++pre_equal; }
        if (take) ids[output++] = group_start + (COMPACT ? compact_ids[i] : i);
    }
}

// Task: one query's exact top-k, grid(query), 256 threads / 8 SIMDgroups.
kernel void qsa_indexer_select(
    device const float* scores [[buffer(0)]], device const Token* tokens [[buffer(1)]],
    device uint* ids [[buffer(2)]], constant uint& num_active_tokens [[buffer(3)]],
    constant uint& num_total_groups [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
    uint query [[threadgroup_position_in_grid]]) {
    if (query >= num_active_tokens) return;
    const Token token = tokens[query];
    const uint count = num_visible_groups(token);
    if (count <= top_k) return;
    threadgroup atomic_uint hist[256];
    threadgroup uint state[2], greater[selection_threads / 32], equal[selection_threads / 32];
    radix_select<false>(scores + ulong(query) * num_total_groups, nullptr, nullptr, count, token.group_start,
                        ids + ulong(query) * top_k, tid, hist, state, greater, equal);
}

// Task: one query, 1024 threads / 32 SIMDgroups. Sampling only narrows exact selection.
// The complete visible range is counted. Too few or too many candidates use the full range.
kernel void qsa_indexer_sampled_select(
    device const float* scores [[buffer(0)]], device const Token* tokens [[buffer(1)]],
    device uint* ids [[buffer(2)]], constant uint& num_active_tokens [[buffer(3)]],
    constant uint& num_total_groups [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
    uint query [[threadgroup_position_in_grid]]) {
    if (query >= num_active_tokens) return;
    const Token token = tokens[query];
    const uint count = num_visible_groups(token);
    if (count <= top_k) return;
    constexpr uint capacity = 2048, simdgroups = selection_threads / 32;
    threadgroup atomic_uint hist[256];
    threadgroup uint state[2], greater[simdgroups], equal[simdgroups];
    threadgroup uint compact_keys[capacity], compact_ids[capacity];
    const uint lane = tid % 32u, simd = tid / 32u;
    device const float* row = scores + ulong(query) * num_total_groups;
    device uint* output = ids + ulong(query) * top_k;
    uint threshold = 0, offset = 0, candidates = 0;
    const uint width = count / selection_threads + uint(count % selection_threads != 0u);
    const uint start = uint(min(ulong(count), ulong(tid) * width)), end = start + min(width, count - start);
    if (count >= 4096) {
        const uint sample = as_type<uint>(row[ulong(tid) * count / selection_threads]);
        uint prefix = 0, mask = 0, need = clamp(2u * top_k * selection_threads / count, 1u, selection_threads);
        for (int shift = 24; shift >= 0; shift -= 8) {
            if (tid < 256) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if ((sample & mask) == prefix)
                atomic_fetch_add_explicit(&hist[(sample >> uint(shift)) & 255u], 1u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd == 0) radix_cut(hist, lane, prefix, shift, need, state[0], state[1]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix = state[0]; need = state[1]; mask |= 255u << uint(shift);
        }
        threshold = prefix;
        uint local_count = 0;
        for (uint i = start; i < end; ++i) local_count += uint(as_type<uint>(row[i]) >= threshold);
        offset = simd_prefix_exclusive_sum(local_count);
        if (lane == 31) greater[simd] = offset + local_count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            const uint n = lane < simdgroups ? greater[lane] : 0u;
            const uint pre = simd_prefix_exclusive_sum(n);
            if (lane < simdgroups) greater[lane] = pre;
            if (lane == 31) state[0] = pre + n;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        offset += greater[simd];
        candidates = state[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (candidates >= top_k && candidates <= capacity) {
        for (uint i = start; i < end; ++i) {
            const uint key = as_type<uint>(row[i]);
            if (key >= threshold) { compact_keys[offset] = key; compact_ids[offset] = i; ++offset; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        radix_select<true>(row, compact_keys, compact_ids, candidates, token.group_start, output,
                           tid, hist, state, greater, equal);
    } else {
        radix_select<false>(row, nullptr, nullptr, count, token.group_start, output,
                            tid, hist, state, greater, equal);
    }
}
