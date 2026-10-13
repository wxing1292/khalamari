// Each SIMDgroup stages one 16-token x 64-dimension K or V tile. K and V
// reuse this storage; the final cross-SIMDgroup reduction reuses it as F32.
inline void load_kv_tile(
    device const uchar* pages, device const uint* page_ids,
    uint req_slot, uint layer_index, uint kv_head_index,
    uint token_begin, uint token_end, uint dim_begin, uint value_plane,
    threadgroup bfloat16_t* tile, uint lane)
{
    for (uint index = lane * 8; index < 16 * 64; index += 32 * 8) {
        const ulong token = (ulong)token_begin + index / 64;
        const uint dim = dim_begin + index % 64;
        if (token < token_end && dim < KV_HEAD_DIM) {
            const uint block = token / (NUM_TOKENS * NUM_PAGE_IDS_PER_BLOCK);
            const uint page_index = token / NUM_TOKENS % NUM_PAGE_IDS_PER_BLOCK;
            const ulong address = ((((ulong)req_slot * NUM_GQA_LAYERS + layer_index) * NUM_BLOCKS + block)
                                   * NUM_PAGE_IDS_PER_BLOCK) + page_index;
            const ulong page = page_ids[address];
            const device uchar* source = pages + page * PAGE_BYTES
                + ((ulong)(value_plane * NUM_KV_HEADS + kv_head_index) * NUM_TOKENS + token % NUM_TOKENS) * KV_HEAD_DIM + dim;
            if constexpr (KV_HEAD_DIM % 8 == 0) {
                *reinterpret_cast<threadgroup uint4*>(tile + index) =
                    fp8_e4m3x8_to_bf16x8(*reinterpret_cast<device const uint2*>(source));
            } else {
                for (uint part = 0; part < 8; ++part) {
                    tile[index + part] = dim + part < KV_HEAD_DIM
                        ? as_type<bfloat16_t>(fp8_e4m3_to_bf16_bits(source[part])) : bfloat16_t(0.0f);
                }
            }
        } else {
            *reinterpret_cast<threadgroup uint4*>(tile + index) = uint4(0);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void gqa_sdpa_sliced_kv_map(
    device const T* q [[buffer(0)]],
    device const uchar* kv_pages [[buffer(1)]],
    device const uint* req_slots [[buffer(2)]],
    device const uint* page_ids [[buffer(3)]],
    device const uint* sdpa_map_task_templates [[buffer(4)]],
    device float* partial_exp_sums [[buffer(5)]],
    device float* partial_max_logits [[buffer(6)]],
    device T* partial_output [[buffer(7)]],
    constant uint& gqa_layer_index [[buffer(8)]],
    constant uint& num_active_tokens [[buffer(9)]],
    constant uint& num_active_kv_splits [[buffer(10)]],
    uint3 block [[threadgroup_position_in_grid]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint num_simdgroups = REQUIRED_THREADS / 32;
    constexpr uint num_head_fragments = (KV_HEAD_DIM + 7) / 8;
    constexpr uint num_output_tiles = (KV_HEAD_DIM + 63) / 64;
    const uint task = block.x;
    if (task >= num_active_kv_splits) return;
    const uint q_token = sdpa_map_task_templates[task * 3];
    if (q_token >= num_active_tokens) return;
    const uint head_group = block.y;
    const uint head_range = head_group % NUM_Q_HEAD_RANGES_PER_KV_HEAD;
    const uint kv_head = head_group / NUM_Q_HEAD_RANGES_PER_KV_HEAD;
    const uint head_begin = kv_head * Q_HEADS_PER_KV_HEAD + head_range * MAX_Q_HEADS;
    const uint num_heads = min(uint(MAX_Q_HEADS), uint(Q_HEADS_PER_KV_HEAD) - head_range * MAX_Q_HEADS);
    const uint kv_begin = sdpa_map_task_templates[task * 3 + 1];
    const uint kv_end = sdpa_map_task_templates[task * 3 + 2];
    const uint req_slot = req_slots[q_token];

    // Each SIMDgroup owns disjoint KV iterations. Reuse its KV workspace for
    // the final cross-SIMDgroup merge. Keep E and N in REGISTER; load Q by fragment.
    threadgroup float workspace[num_simdgroups * 8 * 64];
    threadgroup float stats[num_simdgroups * 8 * 2];
    threadgroup float global_stats[8 * 2];
    threadgroup bfloat16_t* kv = reinterpret_cast<threadgroup bfloat16_t*>(workspace + simdgroup_index * 8 * 64);
    threadgroup float* q_head_exp_sums = stats + simdgroup_index * 16;
    threadgroup float* q_head_max_logits = q_head_exp_sums + 8;
    const short2 coord = mlx::steel::BaseMMAFrag<float, 8, 8>::get_coord(ushort(lane));
    const uint q_head_offset = uint(coord.y);
    const uint col = uint(coord.x);
    mlx::steel::MMATile<float, 1, num_head_fragments> output;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (ulong begin = (ulong)kv_begin + simdgroup_index * 16; begin < kv_end; begin += num_simdgroups * 16) {
        mlx::steel::MMATile<float, 1, 2> scores;
        #pragma unroll
        for (uint tile = 0; tile < num_output_tiles; ++tile) {
            load_kv_tile(kv_pages, page_ids, req_slot, gqa_layer_index, kv_head, uint(begin), kv_end, tile * 64, 0, kv, lane);
            #pragma unroll
            for (uint d = 0; d < 8; ++d) {
                if (tile * 8 + d < num_head_fragments) {
                    simdgroup_matrix<T, 8, 8> q_fragment;
                    vec<T, 2> q_values(T(0));
                    #pragma unroll
                    for (uint i = 0; i < 2; ++i) {
                        const uint dim = tile * 64 + d * 8 + col + i;
                        if (q_head_offset < num_heads && dim < KV_HEAD_DIM) {
                            q_values[i] = q[((ulong)q_token * NUM_Q_HEADS + head_begin + q_head_offset) * KV_HEAD_DIM + dim];
                        }
                    }
                    reinterpret_cast<thread vec<T, 2>&>(q_fragment.thread_elements()) = q_values;
                    #pragma unroll
                    for (uint f = 0; f < 2; ++f) {
                        simdgroup_matrix<bfloat16_t, 8, 8> k_fragment;
                        simdgroup_load(k_fragment, kv + f * 8 * 64 + d * 8, 64, ulong2(0), true);
                        auto score_fragment = scores.mat_at(0, f);
                        simdgroup_multiply_accumulate(score_fragment, q_fragment, k_fragment, score_fragment);
                        scores.frag_at(0, f) = reinterpret_cast<thread float2&>(score_fragment.thread_elements());
                    }
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        float local_max = -INFINITY;
        #pragma unroll
        for (uint f = 0; f < 2; ++f) {
            #pragma unroll
            for (uint i = 0; i < 2; ++i) {
                float score = scores.frag_at(0, f)[i];
                score = q_head_offset < num_heads && begin + f * 8 + col + i < kv_end ? score * ATTENTION_SCALE : -INFINITY;
                scores.frag_at(0, f)[i] = score;
                local_max = max(local_max, score);
            }
        }
        local_max = max(local_max, simd_shuffle_xor(local_max, ushort(1)));
        local_max = max(local_max, simd_shuffle_xor(local_max, ushort(8)));
        float next_max = max(running_max, local_max);
        const float scale = running_max == -INFINITY ? 0.0f : metal::exp(running_max - next_max);
        if (next_max == -INFINITY) next_max = 0.0f;
        running_max = next_max;
        float local_sum = 0.0f;
        #pragma unroll
        for (uint f = 0; f < 2; ++f) {
            #pragma unroll
            for (uint i = 0; i < 2; ++i) {
                float score = scores.frag_at(0, f)[i];
                score = score == -INFINITY ? 0.0f : metal::exp(score - next_max);
                scores.frag_at(0, f)[i] = score;
                local_sum += score;
            }
        }
        local_sum += simd_shuffle_xor(local_sum, ushort(1));
        local_sum += simd_shuffle_xor(local_sum, ushort(8));
        running_sum = running_sum * scale + local_sum;
        #pragma unroll
        for (uint f = 0; f < num_head_fragments; ++f) output.frag_at(0, f) *= scale;
        #pragma unroll
        for (uint tile = 0; tile < num_output_tiles; ++tile) {
            load_kv_tile(kv_pages, page_ids, req_slot, gqa_layer_index, kv_head, uint(begin), kv_end, tile * 64, 1, kv, lane);
            #pragma unroll
            for (uint f = 0; f < 2; ++f) {
                auto probability_fragment = scores.mat_at(0, f);
                #pragma unroll
                for (uint d = 0; d < 8; ++d) {
                    if (tile * 8 + d < num_head_fragments) {
                        simdgroup_matrix<bfloat16_t, 8, 8> v_fragment;
                        simdgroup_load(v_fragment, kv + f * 8 * 64 + d * 8, 64);
                        auto output_fragment = output.mat_at(0, tile * 8 + d);
                        simdgroup_multiply_accumulate(output_fragment, probability_fragment, v_fragment, output_fragment);
                        output.frag_at(0, tile * 8 + d) = reinterpret_cast<thread float2&>(output_fragment.thread_elements());
                    }
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (col == 0) { q_head_exp_sums[q_head_offset] = running_sum; q_head_max_logits[q_head_offset] = running_sum > 0.0f ? running_max : -INFINITY; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index == 0 && lane < num_heads) {
        float maximum = -INFINITY;
        for (uint sg = 0; sg < num_simdgroups; ++sg) maximum = max(maximum, stats[sg * 16 + 8 + lane]);
        float sum = 0.0f;
        for (uint sg = 0; sg < num_simdgroups; ++sg) {
            const float local_sum = stats[sg * 16 + lane];
            if (local_sum > 0.0f) sum += local_sum * metal::exp(stats[sg * 16 + 8 + lane] - maximum);
        }
        global_stats[lane] = sum;
        global_stats[8 + lane] = maximum;
        const ulong partial = (ulong)task * NUM_Q_HEADS + head_begin + lane;
        partial_exp_sums[partial] = sum;
        partial_max_logits[partial] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tile = 0; tile < num_output_tiles; ++tile) {
        #pragma unroll
        for (uint d = 0; d < 8; ++d) {
            if (tile * 8 + d < num_head_fragments) {
                workspace[simdgroup_index * 8 * 64 + q_head_offset * 64 + d * 8 + col] = output.frag_at(0, tile * 8 + d)[0];
                workspace[simdgroup_index * 8 * 64 + q_head_offset * 64 + d * 8 + col + 1] = output.frag_at(0, tile * 8 + d)[1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup_index == 0 && q_head_offset < num_heads) {
            #pragma unroll
            for (uint d = 0; d < 8; ++d) {
                #pragma unroll
                for (uint i = 0; i < 2; ++i) {
                    const uint dim = tile * 64 + d * 8 + col + i;
                    if (dim < KV_HEAD_DIM) {
                        float value = 0.0f;
                        for (uint sg = 0; sg < num_simdgroups; ++sg) {
                            const float scale = stats[sg * 16 + q_head_offset] > 0.0f ? metal::exp(stats[sg * 16 + 8 + q_head_offset] - global_stats[8 + q_head_offset]) : 0.0f;
                            value += workspace[sg * 8 * 64 + q_head_offset * 64 + d * 8 + col + i] * scale;
                        }
                        const ulong partial = (ulong)task * NUM_Q_HEADS + head_begin + q_head_offset;
                        partial_output[partial * KV_HEAD_DIM + dim] = T(global_stats[q_head_offset] > 0.0f ? value / global_stats[q_head_offset] : 0.0f);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
