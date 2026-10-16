// Copyright 2026 TensorFold contributors.
// Adapted from TensorFold 0.6.6 attention.py (Apache-2.0).
// Q [T,4,128] is token-major. Keys [G,128] are paged BF16.
// Grid (key tile, request-local query template), 256 threads / 8 SIMDgroups.
// SIMDgroup s owns groups [(8*x+s)*B, (8*x+s+1)*B), B=groups_per_simd.
// Each lane keeps four contiguous K dimensions in registers, reused across Q and heads.
kernel void qsa_indexer_vector_score(
    device const scalar_t* q [[buffer(0)]], device const Token* tokens [[buffer(1)]],
    device const ScoreTaskTemplate* score_tasks [[buffer(2)]], device const scalar_t* pages [[buffer(3)]],
    device const uint* page_ids [[buffer(4)]], device float* scores [[buffer(5)]],
    constant uint& num_active_score_task_templates [[buffer(6)]],
    constant uint& num_pages_per_request [[buffer(7)]], constant uint& groups_per_page [[buffer(8)]],
    constant uint& num_total_groups [[buffer(9)]], constant uint& page_shift [[buffer(10)]], uint2 task [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    if (task.y >= num_active_score_task_templates) return;
    const ScoreTaskTemplate queries = score_tasks[task.y];
    const Token first = tokens[queries.query_start];
    const uint tile_width = 8u * groups_per_simd;
    const uint aligned_start = first.group_start - first.group_start % tile_width;
    const uint tile_start = aligned_start + task.x * tile_width;
    const uint group_end = (first.token_index + queries.num_queries) / ratio;
    if (tile_start >= group_end) return;
    const uint physical_page = page_ids[ulong(first.req_slot) * num_pages_per_request + (tile_start >> page_shift)];
    const ulong page_base = ulong(physical_page) * groups_per_page * 128u;
    const uint start = tile_start + simd * groups_per_simd;
    float cached_q[4][4];
    if (max_queries == 1u) {
        for (uint head = 0; head < 4; ++head)
            for (uint i = 0; i < 4; ++i)
                cached_q[head][i] = float(q[(ulong(queries.query_start) * 4u + head) * 128u + lane * 4u + i]);
    }
    for (uint group = start; group < start + groups_per_simd && group < group_end; ++group) {
        if (group < first.group_start) continue;
        const ulong offset = page_base + ulong((group & (groups_per_page - 1u))) * 128u + lane * 4u;
        float key[4];
        for (uint i = 0; i < 4; ++i) key[i] = float(pages[offset + i]);
        for (uint r = 0; r < min(max_queries, queries.num_queries); ++r) {
            if (group >= (first.token_index + r + 1u) / ratio) continue;
            const uint query = queries.query_start + r;
            float score = 0.0f;
            for (uint head = 0; head < 4; ++head) {
                const ulong q_offset = (ulong(query) * 4u + head) * 128u + lane * 4u;
                float dot = 0.0f;
                for (uint i = 0; i < 4; ++i) dot = fma(max_queries == 1u ? cached_q[head][i] : float(q[q_offset + i]), key[i], dot);
                score += max(simd_sum(dot), 0.0f);
            }
            if (lane == 0)
                scores[ulong(query) * num_total_groups + group - first.group_start] = score / precise::sqrt(128.0f);
        }
    }
}
