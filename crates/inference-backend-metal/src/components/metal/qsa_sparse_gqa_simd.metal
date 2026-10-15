#include <metal_stdlib>
using namespace metal;

// Rebase E4M3FN into FP16 / 256, including zeros, subnormals, and signed NaNs.
// Q absorbs the K scale before the scan. The output restores the V scale.
inline uint qsa_fp8x2_to_half(uint bits) {
    uint result = ((bits & 0x7fu) << 7) | ((bits & 0x80u) << 8)
        | ((bits & 0x7f00u) << 15) | ((bits & 0x8000u) << 16);
    result |= select(0u, 0x7c00u, (bits & 0x7fu) == 0x7fu);
    result |= select(0u, 0x7c000000u, (bits & 0x7f00u) == 0x7f00u);
    return result;
}

inline uint4 qsa_fp8x8_to_half(uint2 bits) {
    return uint4(qsa_fp8x2_to_half(bits.x), qsa_fp8x2_to_half(bits.x >> 16),
        qsa_fp8x2_to_half(bits.y), qsa_fp8x2_to_half(bits.y >> 16));
}

inline float qsa_exp(float value) {
    return fast::exp2(value);
}

constant float qsa_max_to_ln = 0.6931471805599453f;

// Q [Tq,24,256] BF16, group IDs [Tq,512] U32, paged K/V FP8 GMEM.
// Grid [sparse-query capacity * S,8 head tiles], num_simdgroups*32 threads/threadblock.
// Example head tile 0: KV head 0, Q heads [0,3). Warp g scans every Gth group.
//
// Q -> registers [3,8] F32 per lane, once; numerator [3,8] F32 = 0
// for assigned four-token group:
//     resolve group/page once
//     K -> registers [8 dims/lane]; reuse across 3 Q heads -> scores [3,4]
//     update register max/sum and rescale numerator
//     V -> registers [8 dims/lane]; reuse across 3 Q heads -> numerator
// warp partials -> SMEM [num_simdgroups,3,256] F32; one barrier; warp 0 merges
// -> global normalized partials [sparse query,S,24,256] F32
// -> shared Reduce -> O [Tq,24,256] BF16
kernel void qsa_sparse_gqa_map(
    device const bfloat* q [[buffer(0)]],
    device const uchar* pages [[buffer(1)]],
    device const uint* req_slots [[buffer(2)]],
    device const uint* page_ids [[buffer(3)]],
    device const uint* selected_groups [[buffer(4)]],
    device const uint* query_positions [[buffer(5)]],
    device const uint* sparse_query_rows [[buffer(6)]],
    device float* partial_exp_sums [[buffer(7)]],
    device float* partial_max_logits [[buffer(8)]],
    device float* partial_output [[buffer(9)]],
    constant uint& num_active_sparse_queries [[buffer(10)]],
    constant uint& page_table_index [[buffer(11)]],
    uint3 block [[threadgroup_position_in_grid]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    constexpr uint q_heads_per_simdgroup = 3;
    constexpr uint num_simdgroups = QSA_NUM_SIMDGROUPS;
    const uint lane = lane_id;
    const uint sg = simdgroup_index;
    const uint task = block.x;
    if (task >= num_active_sparse_queries * QSA_NUM_SPLITS) return;
    const uint sparse_query = task / QSA_NUM_SPLITS;
    const uint query = sparse_query_rows[sparse_query];
    const uint split = task % QSA_NUM_SPLITS;
    const uint kv_head = block.y / 4;
    const uint head_begin = kv_head * 12 + (block.y % 4) * q_heads_per_simdgroup;
    const uint req_slot = req_slots[query];
    const uint query_position = query_positions[query];
    const uint tail = (query_position + 1) % 4;
    const uint begin = 512 * split / QSA_NUM_SPLITS;
    const uint end = 512 * (split + 1) / QSA_NUM_SPLITS + uint(split == QSA_NUM_SPLITS - 1 && tail > 0);
    float4 q_values[q_heads_per_simdgroup][2];
    float4 output_accumulator[q_heads_per_simdgroup][2];
    float maxima[q_heads_per_simdgroup];
    float sums[q_heads_per_simdgroup];
    #pragma unroll
    for (uint h = 0; h < q_heads_per_simdgroup; ++h) {
        const device bfloat4* src = reinterpret_cast<device const bfloat4*>(q + (ulong(query) * 24 + head_begin + h) * 256 + lane * 8);
        q_values[h][0] = float4(src[0]) * QSA_ATTENTION_SCALE;
        q_values[h][1] = float4(src[1]) * QSA_ATTENTION_SCALE;
        output_accumulator[h][0] = float4(0);
        output_accumulator[h][1] = float4(0);
        maxima[h] = -INFINITY;
        sums[h] = 0;
    }
    for (uint group = begin + sg; group < end; group += num_simdgroups) {
        const uint source = group < 512 ? selected_groups[query * 512 + group] * 4 : (query_position + 1) / 4 * 4;
        const uint num_group_tokens = group < 512 ? 4 : tail;
        const uint page_slot = source / 32;
        const uint page_block = page_slot / QSA_NUM_PAGE_IDS_PER_BLOCK;
        const uint page_index = page_slot % QSA_NUM_PAGE_IDS_PER_BLOCK;
        const ulong table_index = (((ulong(req_slot) * QSA_NUM_GQA_LAYERS + page_table_index)
            * QSA_NUM_BLOCKS + page_block) * QSA_NUM_PAGE_IDS_PER_BLOCK) + page_index;
        const uint page = page_ids[table_index];
        const ulong base = ulong(page) * 32768 + (kv_head * 32 + source % 32) * 256 + lane * 8;
        float scores[q_heads_per_simdgroup][4];
        #pragma unroll
        for (uint t = 0; t < 4; ++t) {
            float4 k0(0), k1(0);
            if (t < num_group_tokens) {
                const uint4 bits = qsa_fp8x8_to_half(*reinterpret_cast<device const uint2*>(pages + base + t * 256));
                k0 = float4(as_type<half4>(bits.xy));
                k1 = float4(as_type<half4>(bits.zw));
            }
            #pragma unroll
            for (uint h = 0; h < q_heads_per_simdgroup; ++h) {
                const float sc = simd_sum(dot(q_values[h][0], k0) + dot(q_values[h][1], k1));
                scores[h][t] = t < num_group_tokens ? sc : -INFINITY;
            }
        }
        #pragma unroll
        for (uint h = 0; h < q_heads_per_simdgroup; ++h) {
            const float m = max(maxima[h], max(max(scores[h][0], scores[h][1]), max(scores[h][2], scores[h][3])));
            const float factor = qsa_exp(maxima[h] - m);
            float total = 0;
            #pragma unroll
            for (uint t = 0; t < 4; ++t) {
                scores[h][t] = qsa_exp(scores[h][t] - m);
                total += scores[h][t];
            }
            sums[h] = sums[h] * factor + total;
            output_accumulator[h][0] *= factor;
            output_accumulator[h][1] *= factor;
            maxima[h] = m;
        }
        #pragma unroll
        for (uint t = 0; t < 4; ++t) {
            if (t < num_group_tokens) {
                const uint4 bits = qsa_fp8x8_to_half(*reinterpret_cast<device const uint2*>(pages + base + 16384 + t * 256));
                const float4 v0 = float4(as_type<half4>(bits.xy)), v1 = float4(as_type<half4>(bits.zw));
                #pragma unroll
                for (uint h = 0; h < q_heads_per_simdgroup; ++h) {
                    output_accumulator[h][0] = fma(float4(scores[h][t]), v0, output_accumulator[h][0]);
                    output_accumulator[h][1] = fma(float4(scores[h][t]), v1, output_accumulator[h][1]);
                }
            }
        }
    }
    threadgroup float local_max[num_simdgroups][q_heads_per_simdgroup], local_sum[num_simdgroups][q_heads_per_simdgroup];
    threadgroup float4 local_output[num_simdgroups][q_heads_per_simdgroup][64];
    #pragma unroll
    for (uint h = 0; h < q_heads_per_simdgroup; ++h) {
        if (lane == 0) {
            local_max[sg][h] = maxima[h];
            local_sum[sg][h] = sums[h];
        }
        local_output[sg][h][lane * 2] = output_accumulator[h][0];
        local_output[sg][h][lane * 2 + 1] = output_accumulator[h][1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    #pragma unroll
    for (uint h = 0; h < q_heads_per_simdgroup; ++h) {
        float m = -INFINITY;
        #pragma unroll
        for (uint g = 0; g < num_simdgroups; ++g) m = max(m, local_max[g][h]);
        float l = 0;
        float4 a(0), b(0);
        #pragma unroll
        for (uint g = 0; g < num_simdgroups; ++g) {
            const float w = qsa_exp(local_max[g][h] - m);
            l += local_sum[g][h] * w;
            a += local_output[g][h][lane * 2] * w;
            b += local_output[g][h][lane * 2 + 1] * w;
        }
        sums[h] = l;
        maxima[h] = m;
        output_accumulator[h][0] = a;
        output_accumulator[h][1] = b;
    }
    #pragma unroll
    for (uint h = 0; h < q_heads_per_simdgroup; ++h) {
        const uint at = (task * 24 + head_begin + h);
        if (lane == 0) {
            partial_exp_sums[at] = sums[h];
            partial_max_logits[at] = maxima[h] * qsa_max_to_ln;
        }
        device float4* dst = reinterpret_cast<device float4*>(partial_output + at * 256 + lane * 8);
        dst[0] = output_accumulator[h][0] * (256.0f / sums[h]);
        dst[1] = output_accumulator[h][1] * (256.0f / sums[h]);
    }
}
