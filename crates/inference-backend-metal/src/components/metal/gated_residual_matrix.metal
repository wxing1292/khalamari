#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

float stored(float x) { return float(bfloat(x)); }
float sigmoid(float x) { return 1.0f / (1.0f + metal::exp(-x)); }

// Injected constants: branches=4, hidden%32=0, rank%8=0, token_tiles, with_gate.
// Example: T=64, H=2560, r=320, token_tiles=2 (16 token rows per tile).
//
// down grid [41,4], 256 threads = 8 SIMDgroups per threadgroup:
//   group [n,m] owns tokens [16*m,16*m+16), projection rows [8*n,8*n+8).
//   SIMDgroup s owns K [1280*s,1280*s+1280) of N [T,10240].
//   for each 8-wide K slice:
//     load weight [8,8] into registers once, reuse across two token fragments.
//     register BF16 MMA -> register FP32 accumulators [2,8,8].
//   SMEM partial [8,2,8,8] FP32 -> sum over SIMDgroups -> SiLU / write gate.
//
// read grid [80,4], 128 threads = 4 SIMDgroups per threadgroup:
//   group [h,m] owns tokens [16*m,16*m+16), hidden columns [32*h,32*h+32).
//   SIMDgroup s owns eight hidden columns, all four channels.
//   for each 8-wide rank slice:
//     register z [2,8,8] and weight [4,8,8], reused across token/output tiles.
//     register FP32 accumulators [2,4,8,8]. No SMEM staging.
//   sigmoid * N -> SIMD shuffle across four channels -> x [16,32].

kernel void gr_matrix_down(
    device const bfloat* normalized [[buffer(0)]],
    device const bfloat* down [[buffer(1)]],
    device const bfloat* write_weight [[buffer(2)]],
    device bfloat* low_rank [[buffer(3)]],
    device bfloat* gates [[buffer(4)]],
    constant uint& num_active_tokens [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint columns = rank + (with_gate ? branches : 0);
    const uint token_base = group.y * 8 * token_tiles, column_base = group.x * 8;
    if (token_base >= num_active_tokens) return;
    // Native 8x8 fragments: each lane owns [fragment_row,fragment_column: +2].
    const uint quad = lane >> 2;
    const uint fragment_row = (quad & 4) | ((lane >> 1) & 3);
    const uint fragment_column = ((quad & 2) << 1) | ((lane & 1) << 1);
    simdgroup_matrix<float, 8, 8> accum[token_tiles];
    for (uint t = 0; t < token_tiles; ++t) accum[t] = simdgroup_matrix<float, 8, 8>(0.0f);
    const ulong column0 = ulong(column_base) + fragment_column, column1 = column0 + 1;
    const uint safe0 = uint(min(column0, ulong(columns - 1))), safe1 = uint(min(column1, ulong(columns - 1)));
    const device bfloat* weight0 = safe0 < rank ? down + ulong(safe0) * wide : write_weight + ulong(safe0 - rank) * wide;
    const device bfloat* weight1 = safe1 < rank ? down + ulong(safe1) * wide : write_weight + ulong(safe1 - rank) * wide;
    for (uint k = simd_group * (wide / 8); k < (simd_group + 1) * (wide / 8); k += 8) {
        simdgroup_matrix<bfloat, 8, 8> weight;
        weight.thread_elements()[0] = column0 < columns ? weight0[k + fragment_row] : bfloat(0);
        weight.thread_elements()[1] = column1 < columns ? weight1[k + fragment_row] : bfloat(0);
        for (uint t = 0; t < token_tiles; ++t) {
            const uint token = uint(min(ulong(token_base) + t * 8 + fragment_row, ulong(num_active_tokens - 1)));
            simdgroup_matrix<bfloat, 8, 8> input;
            input.thread_elements()[0] = normalized[ulong(token) * wide + k + fragment_column];
            input.thread_elements()[1] = normalized[ulong(token) * wide + k + fragment_column + 1];
            simdgroup_multiply_accumulate(accum[t], input, weight, accum[t]);
        }
    }
    threadgroup float partials[8 * token_tiles * 64];
    for (uint t = 0; t < token_tiles; ++t) {
        partials[(simd_group * token_tiles + t) * 64 + fragment_row * 8 + fragment_column] = accum[t].thread_elements()[0];
        partials[(simd_group * token_tiles + t) * 64 + fragment_row * 8 + fragment_column + 1] = accum[t].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint cell = tid; cell < token_tiles * 64; cell += 256) {
        const ulong token = ulong(token_base) + cell / 64 * 8 + (cell % 64) / 8;
        const ulong column = ulong(column_base) + cell % 8;
        if (token >= num_active_tokens || column >= columns) continue;
        float sum = 0.0f;
        for (uint s = 0; s < 8; ++s) sum += partials[s * token_tiles * 64 + cell];
        const float value = stored(stored(sum) / float(branches));
        if (column < rank) low_rank[ulong(token) * rank + column] = bfloat(value * sigmoid(value));
        else gates[ulong(token) * branches + column - rank] = bfloat(2.0f * stored(sigmoid(value)));
    }
}

kernel void gr_matrix_read(
    device const bfloat* normalized [[buffer(0)]],
    device const bfloat* low_rank [[buffer(1)]],
    device const bfloat* up [[buffer(2)]],
    device bfloat* hidden_state [[buffer(3)]],
    constant uint& num_active_tokens [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint token_base = group.y * 8 * token_tiles, hidden_base = group.x * 32;
    if (token_base >= num_active_tokens) return;
    const uint quad = lane >> 2;
    const uint fragment_row = (quad & 4) | ((lane >> 1) & 3);
    const uint fragment_column = ((quad & 2) << 1) | ((lane & 1) << 1);
    simdgroup_matrix<float, 8, 8> accum[token_tiles][4];
    for (uint t = 0; t < token_tiles; ++t) {
        for (uint c = 0; c < 4; ++c) accum[t][c] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (uint k = 0; k < rank; k += 8) {
        simdgroup_matrix<bfloat, 8, 8> input[token_tiles];
        for (uint t = 0; t < token_tiles; ++t) {
            const uint token = uint(min(ulong(token_base) + t * 8 + fragment_row, ulong(num_active_tokens - 1)));
            input[t].thread_elements()[0] = low_rank[ulong(token) * rank + k + fragment_column];
            input[t].thread_elements()[1] = low_rank[ulong(token) * rank + k + fragment_column + 1];
        }
        for (uint c = 0; c < 4; ++c) {
            const uint column = simd_group * 32 + c * 8 + fragment_column;
            const uint h = hidden_base + column / branches, channel = column % branches;
            simdgroup_matrix<bfloat, 8, 8> weight;
            weight.thread_elements()[0] = up[ulong(channel * hidden + h) * rank + k + fragment_row];
            weight.thread_elements()[1] = up[ulong((channel + 1) * hidden + h) * rank + k + fragment_row];
            for (uint t = 0; t < token_tiles; ++t) simdgroup_multiply_accumulate(accum[t][c], input[t], weight, accum[t][c]);
        }
    }
    for (uint t = 0; t < token_tiles; ++t) {
        const ulong token = ulong(token_base) + t * 8 + fragment_row;
        const uint safe_token = uint(min(token, ulong(num_active_tokens - 1)));
        for (uint c = 0; c < 4; ++c) {
            const uint column = simd_group * 32 + c * 8 + fragment_column;
            const uint h = hidden_base + column / branches, channel = column % branches;
            const float p0 = stored(stored(sigmoid(stored(accum[t][c].thread_elements()[0])))
                                    * float(normalized[ulong(safe_token) * wide + channel * hidden + h]));
            const float p1 = stored(stored(sigmoid(stored(accum[t][c].thread_elements()[1])))
                                    * float(normalized[ulong(safe_token) * wide + (channel + 1) * hidden + h]));
            const float p2 = simd_shuffle_xor(p0, 1), p3 = simd_shuffle_xor(p1, 1);
            if ((lane & 1) == 0 && token < num_active_tokens) {
                hidden_state[ulong(token) * hidden + h] = bfloat(((p0 + p1) + p2 + p3) / float(branches));
            }
        }
    }
}
