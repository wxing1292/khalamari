// SplitKV TiledQ SDPA map. One MapThreadBlockTask maps 1:1 to one threadblock. Its
// fields are sourced as follows:
//
// Map task template {     // materialized; three u32 fields
//   q_token_range_index,  // sdpa_map_task_templates[map_task_template_index, 0]
//   kv_token_begin,       // sdpa_map_task_templates[map_task_template_index, 1]
//   kv_token_end,         // sdpa_map_task_templates[map_task_template_index, 2]
// }
// MapThreadBlockTask {
//   q_token_range_index,  // from the Map task template
//   kv_token_begin,       // from the Map task template
//   kv_token_end,         // from the Map task template
//   kv_head_index,        // grid-derived from threadblock_position.x
//   q_head_range_index,   // grid-derived from threadblock_position.x
// }
//
// visible_kv_token_ranges stores one request-local half-open
// [kv_token_begin, kv_token_end) range for each flat Q token. Each row computes
// the intersection of this range and its Map task-template range.
//
// A sentinel Map task template returns without writing any partial output or
// statistics.

// Steel fragments keep Q, softmax weights, and the output numerator in REGISTER.
kernel void gqa_split_kv_tiled_q_map(
    device const bfloat16_t* q [[buffer(0)]],
    device const uchar* kv_pages [[buffer(1)]],
    device const uint* req_slots [[buffer(2)]],
    device const uint* page_ids [[buffer(3)]],
    device const uint* visible_kv_token_ranges [[buffer(4)]],
    device const uint* q_token_ranges [[buffer(5)]],
    device const uint* sdpa_map_task_templates [[buffer(6)]],
    device bfloat16_t* partial_output [[buffer(7)]],
    device float* partial_exp_sums [[buffer(8)]],
    device float* partial_max_logits [[buffer(9)]],
    constant uint& gqa_layer_index [[buffer(10)]],
    constant uint& num_active_q_token_tiles [[buffer(11)]],
    constant uint& num_active_kv_splits [[buffer(12)]],
    constant uint& num_active_tokens [[buffer(13)]],
    threadgroup char* shared_mem [[threadgroup(0)]],
    uint3 threadblock_position [[threadgroup_position_in_grid]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr int NUM_SIMD_LANES = 32;
    constexpr int NUM_SIMDGROUPS = MAP_REQUIRED_THREADS / NUM_SIMD_LANES;
    constexpr int NUM_SIMDGROUPS_PER_Q_HEAD = MAX_Q_TOKENS / 8;
    constexpr int NUM_HEAD_FRAGMENTS = HEAD_DIM / 8;
    constexpr int NUM_KV_TOKEN_FRAGMENTS = KV_TOKENS_PER_ITERATION / 8;
    constexpr int Q_HEADS_PER_KV_HEAD = NUM_Q_HEADS / NUM_KV_HEADS;
    static_assert(NUM_SIMDGROUPS == NUM_SIMDGROUPS_PER_Q_HEAD * MAX_Q_HEADS);

    const uint map_task_template_index = threadblock_position.y;
    if (map_task_template_index >= num_active_kv_splits) {
        return;
    }
    const uint q_token_range_index = sdpa_map_task_templates[map_task_template_index * 3];
    if (q_token_range_index >= num_active_q_token_tiles) {
        return;
    }
    const uint flat_token_start = q_token_ranges[q_token_range_index * 2];
    const uint flat_token_end = q_token_ranges[q_token_range_index * 2 + 1];
    if (flat_token_start >= num_active_tokens || flat_token_end > num_active_tokens) {
        return;
    }
    const uint num_q_tokens_in_range = flat_token_end - flat_token_start;
    const uint kv_token_begin = sdpa_map_task_templates[map_task_template_index * 3 + 1];
    const uint kv_token_end = sdpa_map_task_templates[map_task_template_index * 3 + 2];
    const uint req_slot = req_slots[flat_token_start];

    const uint head_group_index = threadblock_position.x;
    const uint q_head_range_index = head_group_index % uint(NUM_Q_HEAD_RANGES_PER_KV_HEAD);
    const uint kv_head_index = head_group_index / uint(NUM_Q_HEAD_RANGES_PER_KV_HEAD);
    const uint q_head_range_begin = q_head_range_index * uint(MAX_Q_HEADS);
    const uint num_active_q_heads = min(
        uint(MAX_Q_HEADS), uint(Q_HEADS_PER_KV_HEAD) - q_head_range_begin);
    const uint local_q_head_index = simdgroup_index / uint(NUM_SIMDGROUPS_PER_Q_HEAD);
    const uint token_fragment_index = simdgroup_index % uint(NUM_SIMDGROUPS_PER_Q_HEAD);
    const bool active_q_head = local_q_head_index < num_active_q_heads;
    const uint q_head_index =
        kv_head_index * uint(Q_HEADS_PER_KV_HEAD) + q_head_range_begin + local_q_head_index;

    constexpr int PAD = 16 / int(sizeof(bfloat16_t));
    constexpr int LEADING_DIM = HEAD_DIM + PAD;
    threadgroup bfloat16_t* k_shared = reinterpret_cast<threadgroup bfloat16_t*>(shared_mem);
    threadgroup bfloat16_t* v_shared = k_shared;

    const short2 coordinate = mlx::steel::BaseMMAFrag<float, 8, 8>::get_coord(ushort(lane));
    const ushort token_fragment_offset = coordinate.y;
    const ushort fragment_col = coordinate.x;
    const uint token_offset = token_fragment_index * 8 + uint(token_fragment_offset);
    const bool inactive_token = !active_q_head || token_offset >= num_q_tokens_in_range;
    const uint flat_q_token_index = flat_token_start + token_offset;
    const uint visible_kv_token_begin = inactive_token ? 0 : visible_kv_token_ranges[flat_q_token_index * 2];
    const uint visible_kv_token_end = inactive_token ? 0 : visible_kv_token_ranges[flat_q_token_index * 2 + 1];

    using QueryTile = mlx::steel::MMATile<bfloat16_t, 1, NUM_HEAD_FRAGMENTS>;
    using ScoreTile = mlx::steel::MMATile<float, 1, NUM_KV_TOKEN_FRAGMENTS>;
    using OutputTile = mlx::steel::MMATile<float, 1, NUM_HEAD_FRAGMENTS>;
    QueryTile query;
    OutputTile output;
    if (!inactive_token) {
        const device bfloat16_t* row = q + ((ulong)flat_q_token_index * NUM_Q_HEADS + q_head_index) * HEAD_DIM;
        #pragma unroll
        for (int f = 0; f < NUM_HEAD_FRAGMENTS; ++f) {
            query.frag_at(0, f) = bfloat2(row[f * 8 + fragment_col], row[f * 8 + fragment_col + 1]);
        }
    }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    const uint num_kv_iterations =
        (kv_token_end - kv_token_begin + uint(KV_TOKENS_PER_ITERATION - 1)) / uint(KV_TOKENS_PER_ITERATION);
    for (uint kv_iteration_index = 0; kv_iteration_index < num_kv_iterations; ++kv_iteration_index) {
        const uint kv_iteration_begin = kv_token_begin + kv_iteration_index * uint(KV_TOKENS_PER_ITERATION);
        for (uint kv_token_offset = simdgroup_index; kv_token_offset < uint(KV_TOKENS_PER_ITERATION);
             kv_token_offset += uint(NUM_SIMDGROUPS)) {
            const uint kv_token_index = kv_iteration_begin + kv_token_offset;
            if (kv_token_index < kv_token_end) {
                const uint block_index = kv_token_index / uint(NUM_TOKENS_PER_PAGE * NUM_PAGE_IDS_PER_BLOCK);
                const uint page_id_index =
                    (kv_token_index / uint(NUM_TOKENS_PER_PAGE)) % uint(NUM_PAGE_IDS_PER_BLOCK);
                const uint page_token_index = kv_token_index % uint(NUM_TOKENS_PER_PAGE);
                const ulong page_table_index =
                    ((((ulong)req_slot * (ulong)NUM_GQA_LAYERS + (ulong)gqa_layer_index)
                      * (ulong)NUM_BLOCKS + (ulong)block_index)
                     * (ulong)NUM_PAGE_IDS_PER_BLOCK) + (ulong)page_id_index;
                const ulong page_id = (ulong)page_ids[page_table_index];
                const ulong page_base = page_id * (ulong)PAGE_BYTES;
                const device uchar* k = kv_pages + page_base
                    + (ulong)(((0 * NUM_KV_HEADS + kv_head_index) * NUM_TOKENS_PER_PAGE + page_token_index) * HEAD_DIM);
                constexpr uint LOAD_VALUES = 8;
                #pragma unroll
                for (uint dim = lane * LOAD_VALUES; dim < uint(HEAD_DIM);
                     dim += uint(NUM_SIMD_LANES) * LOAD_VALUES) {
                    const uint2 k_fp8 = *reinterpret_cast<device const uint2*>(k + dim);
                    *reinterpret_cast<threadgroup uint4*>(
                        k_shared + kv_token_offset * LEADING_DIM + dim) = fp8_e4m3x8_to_bf16x8(k_fp8);
                }
            } else {
                constexpr uint LOAD_VALUES = 8;
                #pragma unroll
                for (uint dim = lane * LOAD_VALUES; dim < uint(HEAD_DIM);
                     dim += uint(NUM_SIMD_LANES) * LOAD_VALUES) {
                    *reinterpret_cast<threadgroup uint4*>(
                        k_shared + kv_token_offset * LEADING_DIM + dim) = uint4(0);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        ScoreTile scores;
        #pragma unroll
        for (int dim_fragment = 0; dim_fragment < NUM_HEAD_FRAGMENTS; ++dim_fragment) {
            auto q_fragment = query.mat_at(0, dim_fragment);
            #pragma unroll
            for (int f = 0; f < NUM_KV_TOKEN_FRAGMENTS; ++f) {
                simdgroup_matrix<bfloat16_t, 8, 8> k_fragment;
                simdgroup_load(k_fragment, k_shared + f * 8 * LEADING_DIM + dim_fragment * 8,
                              LEADING_DIM, ulong2(0), true);
                auto score_fragment = scores.mat_at(0, f);
                simdgroup_multiply_accumulate(score_fragment, q_fragment, k_fragment, score_fragment);
                scores.frag_at(0, f) = reinterpret_cast<thread float2&>(score_fragment.thread_elements());
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kv_token_offset = simdgroup_index; kv_token_offset < uint(KV_TOKENS_PER_ITERATION);
             kv_token_offset += uint(NUM_SIMDGROUPS)) {
            const uint kv_token_index = kv_iteration_begin + kv_token_offset;
            if (kv_token_index < kv_token_end) {
                const uint block_index = kv_token_index / uint(NUM_TOKENS_PER_PAGE * NUM_PAGE_IDS_PER_BLOCK);
                const uint page_id_index =
                    (kv_token_index / uint(NUM_TOKENS_PER_PAGE)) % uint(NUM_PAGE_IDS_PER_BLOCK);
                const uint page_token_index = kv_token_index % uint(NUM_TOKENS_PER_PAGE);
                const ulong page_table_index =
                    ((((ulong)req_slot * (ulong)NUM_GQA_LAYERS + (ulong)gqa_layer_index)
                      * (ulong)NUM_BLOCKS + (ulong)block_index)
                     * (ulong)NUM_PAGE_IDS_PER_BLOCK) + (ulong)page_id_index;
                const ulong page_id = (ulong)page_ids[page_table_index];
                const ulong page_base = page_id * (ulong)PAGE_BYTES;
                const device uchar* v = kv_pages + page_base
                    + (ulong)(((1 * NUM_KV_HEADS + kv_head_index) * NUM_TOKENS_PER_PAGE + page_token_index) * HEAD_DIM);
                constexpr uint LOAD_VALUES = 8;
                #pragma unroll
                for (uint dim = lane * LOAD_VALUES; dim < uint(HEAD_DIM);
                     dim += uint(NUM_SIMD_LANES) * LOAD_VALUES) {
                    const uint2 v_fp8 = *reinterpret_cast<device const uint2*>(v + dim);
                    *reinterpret_cast<threadgroup uint4*>(
                        v_shared + kv_token_offset * LEADING_DIM + dim) = fp8_e4m3x8_to_bf16x8(v_fp8);
                }
            } else {
                constexpr uint LOAD_VALUES = 8;
                #pragma unroll
                for (uint dim = lane * LOAD_VALUES; dim < uint(HEAD_DIM);
                     dim += uint(NUM_SIMD_LANES) * LOAD_VALUES) {
                    *reinterpret_cast<threadgroup uint4*>(
                        v_shared + kv_token_offset * LEADING_DIM + dim) = uint4(0);
                }
            }
        }

        const bool fully_visible = !inactive_token && kv_iteration_begin >= visible_kv_token_begin
            && kv_iteration_begin + uint(KV_TOKENS_PER_ITERATION) <= min(visible_kv_token_end, kv_token_end);
        if (fully_visible) {
            #pragma unroll
            for (int f = 0; f < NUM_KV_TOKEN_FRAGMENTS; ++f) scores.frag_at(0, f) *= ATTENTION_SCALE;
        } else {
            #pragma unroll
            for (int kv_token_fragment = 0; kv_token_fragment < NUM_KV_TOKEN_FRAGMENTS; ++kv_token_fragment) {
                #pragma unroll
                for (int fragment_slot = 0; fragment_slot < 2; ++fragment_slot) {
                    const uint kv_token_index = kv_iteration_begin + uint(kv_token_fragment * 8)
                        + uint(fragment_col) + uint(fragment_slot);
                    const float score = scores.frag_at(0, kv_token_fragment)[fragment_slot] * ATTENTION_SCALE;
                    scores.frag_at(0, kv_token_fragment)[fragment_slot] =
                        inactive_token || kv_token_index < visible_kv_token_begin
                            || kv_token_index >= visible_kv_token_end || kv_token_index >= kv_token_end
                        ? -INFINITY
                        : score;
                }
            }
        }

        float local_max = -INFINITY;
        #pragma unroll
        for (int kv_token_fragment = 0; kv_token_fragment < NUM_KV_TOKEN_FRAGMENTS; ++kv_token_fragment) {
            local_max = max(local_max, max(scores.frag_at(0, kv_token_fragment)[0], scores.frag_at(0, kv_token_fragment)[1]));
        }
        local_max = max(local_max, simd_shuffle_xor(local_max, ushort(1)));
        const float iteration_max = max(local_max, simd_shuffle_xor(local_max, ushort(8)));
        float next_max = max(running_max, iteration_max);
        const float running_scale = running_max == -INFINITY ? 0.0f : metal::exp(running_max - next_max);
        if (running_max == -INFINITY && next_max == -INFINITY) {
            next_max = 0.0f;
        }
        running_max = next_max;

        float local_sum = 0.0f;
        #pragma unroll
        for (int kv_token_fragment = 0; kv_token_fragment < NUM_KV_TOKEN_FRAGMENTS; ++kv_token_fragment) {
            #pragma unroll
            for (int fragment_slot = 0; fragment_slot < 2; ++fragment_slot) {
                const float probability = scores.frag_at(0, kv_token_fragment)[fragment_slot] == -INFINITY
                    ? 0.0f
                    : metal::exp(scores.frag_at(0, kv_token_fragment)[fragment_slot] - next_max);
                scores.frag_at(0, kv_token_fragment)[fragment_slot] = probability;
                local_sum += probability;
            }
        }
        local_sum += simd_shuffle_xor(local_sum, ushort(1));
        local_sum += simd_shuffle_xor(local_sum, ushort(8));
        running_sum = running_sum * running_scale + local_sum;
        #pragma unroll
        for (int dim_fragment = 0; dim_fragment < NUM_HEAD_FRAGMENTS; ++dim_fragment) {
            output.frag_at(0, dim_fragment) *= running_scale;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int kv_token_fragment = 0; kv_token_fragment < NUM_KV_TOKEN_FRAGMENTS; ++kv_token_fragment) {
            auto probability_fragment = scores.mat_at(0, kv_token_fragment);
            #pragma unroll
            for (int dim_fragment = 0; dim_fragment < NUM_HEAD_FRAGMENTS; ++dim_fragment) {
                simdgroup_matrix<bfloat16_t, 8, 8> v_fragment;
                simdgroup_load(
                    v_fragment,
                    v_shared + kv_token_fragment * 8 * LEADING_DIM + dim_fragment * 8,
                    LEADING_DIM);
                auto output_fragment = output.mat_at(0, dim_fragment);
                simdgroup_multiply_accumulate(output_fragment, probability_fragment, v_fragment, output_fragment);
                output.frag_at(0, dim_fragment) =
                    reinterpret_cast<thread float2&>(output_fragment.thread_elements());
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!inactive_token) {
        const ulong partial_output_index =
            ((ulong)map_task_template_index * NUM_Q_HEADS + (ulong)q_head_index) * MAX_Q_TOKENS
            + (ulong)token_offset;
        if (fragment_col == 0) {
            partial_exp_sums[partial_output_index] = running_sum;
            partial_max_logits[partial_output_index] =
                running_sum == 0.0f ? -INFINITY : running_max;
        }
        const float inverse_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
        #pragma unroll
        for (int dim_fragment = 0; dim_fragment < NUM_HEAD_FRAGMENTS; ++dim_fragment) {
            #pragma unroll
            for (int fragment_slot = 0; fragment_slot < 2; ++fragment_slot) {
                const int dim = dim_fragment * 8 + int(fragment_col) + fragment_slot;
                partial_output[partial_output_index * HEAD_DIM + (ulong)dim] =
                    bfloat16_t(output.frag_at(0, dim_fragment)[fragment_slot] * inverse_sum);
            }
        }
    }
}
