#include <metal_stdlib>
using namespace metal;
using namespace mlx::steel;
typedef bfloat bfloat16_t;

// Sparse QSA map over the request's paged FP8 KV cache.

constant uint qsa_group_size = 4;
constant uint qsa_num_groups = 512;
constant uint qsa_selected_tokens = qsa_group_size * qsa_num_groups;
constant uint qsa_head_dim = 256;
constant uint qsa_num_q_heads = 24;
constant uint qsa_num_kv_heads = 2;
constant uint qsa_q_heads_per_kv_head = 12;
constant uint qsa_tokens_per_page = 32;
constant uint qsa_page_bytes = 32768;
constant uint qsa_token_tile = QSA_TOKEN_TILE;
constant uint qsa_dimension_chunk = 256;
constant float qsa_ln2 = 0.6931471805599453f;

constant uint qsa_kv_workspace_elements = qsa_token_tile * (qsa_dimension_chunk + 8);

inline void qsa_resolve_tile(
    device const uint* selected_groups,
    uint req_slot,
    device const uint* page_ids,
    uint page_table_index,
    uint query,
    uint query_position,
    uint selected_begin,
    uint selected_end,
    threadgroup uint* page,
    threadgroup uint* page_token,
    uint lane)
{
    const uint group_begin = selected_begin / qsa_group_size;
    const uint group_end = (selected_end + qsa_group_size - 1) / qsa_group_size;
    for (uint group_offset = lane; group_offset < group_end - group_begin; group_offset += 64) {
        const uint group_position = (group_begin + group_offset) * qsa_group_size;
        uint source_token;
        if (group_position < qsa_selected_tokens) {
            const uint selected_group = selected_groups[query * qsa_num_groups + group_position / qsa_group_size];
            source_token = selected_group * qsa_group_size;
        } else {
            const uint complete_groups = (query_position + 1) / qsa_group_size;
            source_token = complete_groups * qsa_group_size + group_position - qsa_selected_tokens;
        }
        const uint page_slot = source_token / qsa_tokens_per_page;
        const uint block = page_slot / QSA_NUM_PAGE_IDS_PER_BLOCK;
        const uint page_index = page_slot % QSA_NUM_PAGE_IDS_PER_BLOCK;
        const ulong table_index = ((((ulong)req_slot * QSA_NUM_GQA_LAYERS + page_table_index)
            * QSA_NUM_BLOCKS + block) * QSA_NUM_PAGE_IDS_PER_BLOCK) + page_index;
        const uint page_id = page_ids[table_index];
        const uint page_token_begin = source_token % qsa_tokens_per_page;
        for (uint offset = 0; offset < qsa_group_size; ++offset) {
            const uint selected_position = group_position + offset;
            if (selected_position >= selected_begin && selected_position < selected_end) {
                const uint index = selected_position - selected_begin;
                page_token[index] = page_token_begin + offset;
                page[index] = page_id;
            }
        }
    }
    for (uint index = lane; index < qsa_token_tile; index += 64) {
        if (selected_begin + index >= selected_end) {
            page[index] = 0;
            page_token[index] = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <uint plane>
inline void qsa_load_kv_chunk(
    device const uchar* pages,
    uint kv_head,
    uint dimension_begin,
    uint selected_begin,
    uint selected_end,
    threadgroup const uint* page,
    threadgroup const uint* page_token,
    threadgroup bfloat16_t* tile,
    uint lane)
{
    constexpr uint leading_v = qsa_dimension_chunk + 8;
    for (uint element = lane * 8; element < qsa_token_tile * qsa_dimension_chunk;
         element += 64 * 8) {
        const uint row = element / qsa_dimension_chunk;
        const uint dim = element % qsa_dimension_chunk;
        const uint selected_position = selected_begin + row;
        uint4 decoded = uint4(0);
        if (selected_position < selected_end) {
            const uint64_t row_offset =
                (uint64_t(plane * qsa_num_kv_heads + kv_head) * qsa_tokens_per_page
                 + page_token[row]) * qsa_head_dim + dimension_begin + dim;
            const device uchar* source = pages + uint64_t(page[row]) * qsa_page_bytes + row_offset;
            const uint2 packed = *reinterpret_cast<device const uint2*>(source);
            decoded = fp8_e4m3x8_to_bf16x8(packed);
        }
        *reinterpret_cast<threadgroup uint4*>(tile + row * leading_v + dim) = decoded;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Task: one sparse query and KV split from grid.x, one KV head from grid.y.
// Q tile [1,16,256] holds 12 active Q heads. K/V tiles [Tkv_tile,256] share
// indexed loads across both SIMDgroups. Each SIMDgroup owns up to eight Q heads.
// Q stays in distributed BF16 registers across the complete selected-KV scan.
kernel void qsa_sparse_gqa_map(
    device const bfloat16_t* q [[buffer(0)]],
    device const uchar* pages [[buffer(1)]],
    device const uint* req_slots [[buffer(2)]],
    device const uint* page_ids [[buffer(3)]],
    device const uint* selected_groups [[buffer(4)]],
    device const uint* query_positions [[buffer(5)]],
    device const uint* sparse_query_rows [[buffer(6)]],
#if QSA_DIRECT_OUTPUT
    device bfloat16_t* direct_output [[buffer(7)]],
    constant uint& num_active_sparse_queries [[buffer(8)]],
    constant uint& page_table_index [[buffer(9)]],
#else
    device float* partial_exp_sums [[buffer(7)]],
    device float* partial_max_logits [[buffer(8)]],
    device float* partial_output [[buffer(9)]],
    constant uint& num_active_sparse_queries [[buffer(10)]],
    constant uint& page_table_index [[buffer(11)]],
#endif
    uint3 block [[threadgroup_position_in_grid]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr short frag = 8;
    constexpr short leading_v = qsa_dimension_chunk + 8;
    const uint task = block.x;
    if (task >= num_active_sparse_queries * QSA_NUM_SPLITS) return;
    const uint sparse_query = task / QSA_NUM_SPLITS;
    const uint split = task % QSA_NUM_SPLITS;
    const uint query = sparse_query_rows[sparse_query];
    const uint query_position = query_positions[query];
    const uint selected_length = qsa_selected_tokens + (query_position + 1) % qsa_group_size;
    const uint selected_begin = uint(uint64_t(selected_length) * split / QSA_NUM_SPLITS);
    const uint selected_end = uint(uint64_t(selected_length) * (split + 1) / QSA_NUM_SPLITS);
    const uint req_slot = req_slots[query];
    const uint kv_head = block.y;
    const uint query_head_begin = kv_head * qsa_q_heads_per_kv_head;
    const uint simd_lane = lane;
    const uint global_lane = simdgroup_index * 32 + simd_lane;
    const short2 coord = mlx::steel::BaseMMAFrag<float, 8, 8>::get_coord(ushort(simd_lane));
    const uint row = uint(coord.y);
    const uint col = uint(coord.x);

    threadgroup bfloat16_t kv_tile[qsa_kv_workspace_elements];
    threadgroup uint selected_page[qsa_token_tile];
    threadgroup uint selected_page_token[qsa_token_tile];

    using Frag = BaseMMAFrag<float, 8, 8>;
    MMATile<float, 1, qsa_token_tile / 8, Frag> scores;
    MMATile<float, 1, qsa_head_dim / frag, Frag> output;
    output.clear();
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const float scale = QSA_ATTENTION_SCALE;
    const uint q_head_start = query_head_begin + simdgroup_index * 8;
    const uint num_heads = min(8u, query_head_begin + qsa_q_heads_per_kv_head - q_head_start);

    MMATile<bfloat16_t, 1, qsa_head_dim / frag> query_tile;
    #pragma unroll
    for (uint f = 0; f < qsa_head_dim / frag; ++f) {
        bfloat2 value(0);
        if (row < num_heads) {
            const device bfloat16_t* source = q + (ulong(query) * qsa_num_q_heads + q_head_start + row) * qsa_head_dim;
            value = bfloat2(source[f * frag + col], source[f * frag + col + 1]);
        }
        query_tile.frag_at(0, f) = value;
    }

    for (uint begin = selected_begin; begin < selected_end; begin += qsa_token_tile) {
        const uint end = min(begin + qsa_token_tile, selected_end);
        qsa_resolve_tile(selected_groups, req_slot, page_ids, page_table_index,
            query, query_position, begin, end,
            selected_page, selected_page_token, global_lane);
        scores.clear();
        #pragma unroll
        for (uint chunk = 0; chunk < qsa_head_dim / qsa_dimension_chunk; ++chunk) {
            const uint dimension_begin = chunk * qsa_dimension_chunk;
            qsa_load_kv_chunk<0>(pages, kv_head, dimension_begin,
                begin, end, selected_page, selected_page_token, kv_tile, global_lane);
            #pragma unroll
            for (uint sub = 0; sub < qsa_dimension_chunk / frag; ++sub) {
                simdgroup_barrier(mem_flags::mem_none);
                auto q_frag_index = query_tile.mat_at(0, dimension_begin / frag + sub);
                simdgroup_matrix<bfloat16_t, 8, 8> k_frag_index;
                #pragma unroll
                for (uint frag_index = 0; frag_index < qsa_token_tile / 8; ++frag_index) {
                    simdgroup_load(k_frag_index,
                        kv_tile + frag_index * frag * leading_v + sub * frag,
                        leading_v, ulong2(0), true);
                    auto score_matrix = scores.mat_at(0, frag_index);
                    simdgroup_multiply_accumulate(score_matrix,
                        q_frag_index, k_frag_index, score_matrix);
                    scores.frag_at(0, frag_index)
                        = reinterpret_cast<thread float2&>(score_matrix.thread_elements());
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float tile_max = -INFINITY;
        #pragma unroll
        for (uint frag_index = 0; frag_index < qsa_token_tile / 8; ++frag_index) {
            #pragma unroll
            for (uint item = 0; item < 2; ++item) {
                const uint selected_position = begin + frag_index * 8 + col + item;
                float score = scores.frag_at(0, frag_index)[item] * scale;
                score = row < num_heads && selected_position < end
                    ? score : -INFINITY;
                scores.frag_at(0, frag_index)[item] = score;
                tile_max = max(tile_max, score);
            }
        }
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, ushort(1)));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, ushort(8)));
        float next_max = max(running_max, tile_max);
        const float old_scale = running_max == -INFINITY ? 0.0f : fast::exp2(running_max - next_max);
        if (next_max == -INFINITY) next_max = 0.0f;
        running_max = next_max;
        float tile_sum = 0.0f;
        #pragma unroll
        for (uint frag_index = 0; frag_index < qsa_token_tile / 8; ++frag_index) {
            #pragma unroll
            for (uint item = 0; item < 2; ++item) {
                float score = scores.frag_at(0, frag_index)[item];
                score = score == -INFINITY ? 0.0f : fast::exp2(score - next_max);
                scores.frag_at(0, frag_index)[item] = score;
                tile_sum += score;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, ushort(1));
        tile_sum += simd_shuffle_xor(tile_sum, ushort(8));
        running_sum = running_sum * old_scale + tile_sum;
        #pragma unroll
        for (uint frag_index = 0; frag_index < qsa_head_dim / frag; ++frag_index)
            output.frag_at(0, frag_index) *= old_scale;

        #pragma unroll
        for (uint chunk = 0; chunk < qsa_head_dim / qsa_dimension_chunk; ++chunk) {
            qsa_load_kv_chunk<1>(pages, kv_head, chunk * qsa_dimension_chunk,
                begin, end, selected_page, selected_page_token, kv_tile, global_lane);
            #pragma unroll
            for (uint frag_index = 0; frag_index < qsa_token_tile / 8; ++frag_index) {
                #pragma unroll
                for (uint sub = 0; sub < qsa_dimension_chunk / frag; ++sub) {
                    simdgroup_matrix<bfloat16_t, 8, 8> v_frag_index;
                    simdgroup_load(v_frag_index,
                        kv_tile + frag_index * frag * leading_v + sub * frag, leading_v);
                    const uint output_index = chunk * (qsa_dimension_chunk / frag) + sub;
                    auto output_matrix = output.mat_at(0, output_index);
                    simdgroup_multiply_accumulate(output_matrix,
                        scores.mat_at(0, frag_index), v_frag_index, output_matrix);
                    output.frag_at(0, output_index)
                        = reinterpret_cast<thread float2&>(output_matrix.thread_elements());
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint partial_head_base = task * qsa_num_q_heads + q_head_start;
#if !QSA_DIRECT_OUTPUT
    if (col == 0 && row < num_heads) {
        partial_exp_sums[partial_head_base + row] = running_sum;
        partial_max_logits[partial_head_base + row] =
            running_sum > 0.0f ? running_max * qsa_ln2 : -INFINITY;
    }
#endif
    const float reciprocal_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    #pragma unroll
    for (uint frag_index = 0; frag_index < qsa_head_dim / frag; ++frag_index)
        output.frag_at(0, frag_index) *= reciprocal_sum;
    if (row < num_heads) {
#if QSA_DIRECT_OUTPUT
        device bfloat16_t* out = direct_output
            + (uint64_t(query) * qsa_num_q_heads + q_head_start + row) * qsa_head_dim;
        #pragma unroll
        for (uint frag_index = 0; frag_index < qsa_head_dim / frag; ++frag_index) {
            const float2 values = output.frag_at(0, frag_index);
            const uint dim = frag_index * frag + col;
            out[dim] = bfloat16_t(values[0]);
            out[dim + 1] = bfloat16_t(values[1]);
        }
#else
        const uint rows_left = num_heads - row;
        device float* out = partial_output + ulong(partial_head_base + row) * qsa_head_dim + col;
        output.template store_safe<float, 1, 1>(out, qsa_head_dim, short2(qsa_head_dim - col, rows_left));
#endif
    }
}
