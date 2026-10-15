#include <metal_stdlib>
using namespace metal;

float stored(float x) { return float(bfloat(x)); }
float sigmoid(float x) { return 1.0f / (1.0f + metal::exp(-x)); }

// Injected constants: branches=4, hidden%256=0, rank%16=0, with_gate, pending_write.
// Example: T=1, H=2560, r=320, with_gate=true. All tensors are row-major.
//
// down grid [41,4,T], 256 threads = 8 SIMDgroups per threadgroup:
//   group [g,c,t] reads R[t,c,:] [2560] and norm[c,:] [2560].
//   register values[10] per thread -> RMSNorm -> SMEM N_c [2560] BF16.
//   SIMDgroup s reads down/write_weight[8*g+s,c,:] [2560].
//   register dot + SIMD sum -> GMEM partial[c,t,8*g+s] FP32.
//   groups g<10 each store a disjoint 256-element stripe of N[t,c,:].
//
// read grid [160,T], 256 threads per threadgroup:
//   merge partial[:,t,:] [4,320] -> SMEM z [320] BF16.
//   each 16-lane group owns one h, four channels and four rank slices.
//   register up dots -> SIMD rank sum -> sigmoid * N -> SIMD channel sum.
//   one lane writes x[t,h]. Groups reuse z across 16 hidden columns.
//
// A pending write is evaluated before norm but stored only in the read kernel.
// This leaves old R and y intact until all down threadgroups finish reading.

kernel void gr_small_down(
    device const bfloat* residual [[buffer(0)]],
    device const bfloat* norm [[buffer(1)]],
    device const bfloat* down [[buffer(2)]],
    device const bfloat* write_weight [[buffer(3)]],
    device float* partials [[buffer(4)]],
    device bfloat* normalized [[buffer(5)]],
    constant uint& num_active_tokens [[buffer(6)]],
    constant uint& num_total_tokens [[buffer(7)]],
    device const bfloat* old_gate [[buffer(8)]],
    device const bfloat* old_hidden [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint token = group.z, channel = group.y;
    if (token >= num_active_tokens) return;
    constexpr uint columns = rank + (with_gate ? branches : 0);
    const ulong base = ulong(token) * wide + channel * hidden;
    const float gate = pending_write ? float(old_gate[ulong(token) * branches + channel]) : 0.0f;
    float values[hidden / 256], sum = 0.0f;
    for (uint i = 0; i < hidden / 256; ++i) {
        const uint h = tid + i * 256;
        float value = float(residual[base + h]);
        if (pending_write) value = stored(value + stored(gate * float(old_hidden[ulong(token) * hidden + h])));
        values[i] = value;
        sum += value * value;
    }
    sum = simd_sum(sum);
    threadgroup float sums[8];
    threadgroup bfloat4 normalized_vectors[hidden / 4];
    threadgroup bfloat* n = reinterpret_cast<threadgroup bfloat*>(normalized_vectors);
    if (lane == 0) sums[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.0f;
    for (uint s = 0; s < 8; ++s) sum += sums[s];
    const float inv_rms = metal::precise::rsqrt(sum / float(hidden) + eps);
    for (uint i = 0; i < hidden / 256; ++i) {
        const uint h = tid + i * 256;
        n[h] = bfloat((values[i] * inv_rms) * (1.0f + float(norm[channel * hidden + h])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group.x < hidden / 256) {
        const uint h = group.x * 256 + tid;
        normalized[base + h] = n[h];
    }
    if (pending_write && group.x == 0 && tid == 0) {
        partials[ulong(branches) * num_total_tokens * columns + ulong(token) * branches + channel] = gate;
    }
    const uint row = group.x * 8 + simd_group;
    if (row >= columns) return;
    const device bfloat* weight = row < rank
        ? down + ulong(row) * wide + channel * hidden
        : write_weight + ulong(row - rank) * wide + channel * hidden;
    float dot_product = 0.0f;
    for (uint k = lane; k < hidden / 4; k += 32) {
        dot_product += dot(float4(normalized_vectors[k]),
                           float4(reinterpret_cast<device const bfloat4*>(weight)[k]));
    }
    dot_product = simd_sum(dot_product);
    if (lane == 0) partials[(ulong(channel) * num_total_tokens + token) * columns + row] = dot_product;
}

kernel void gr_small_read(
    device const bfloat* normalized [[buffer(0)]],
    device const float* partials [[buffer(1)]],
    device const bfloat* up [[buffer(2)]],
    device bfloat* hidden_state [[buffer(3)]],
    device bfloat* gates [[buffer(4)]],
    constant uint& num_active_tokens [[buffer(5)]],
    constant uint& num_total_tokens [[buffer(6)]],
    device const bfloat* old_residual [[buffer(7)]],
    device bfloat* written [[buffer(8)]],
    device const bfloat* old_hidden [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint token = group.y;
    if (token >= num_active_tokens) return;
    constexpr uint columns = rank + (with_gate ? branches : 0);
    threadgroup bfloat4 low_rank_vectors[rank / 4];
    threadgroup bfloat* z = reinterpret_cast<threadgroup bfloat*>(low_rank_vectors);
    for (uint r = tid; r < rank; r += 256) {
        float value = 0.0f;
        for (uint c = 0; c < branches; ++c) value += partials[(ulong(c) * num_total_tokens + token) * columns + r];
        value = stored(stored(value) / float(branches));
        z[r] = bfloat(value * sigmoid(value));
    }
    if (with_gate && group.x == 0 && tid < branches) {
        float value = 0.0f;
        for (uint c = 0; c < branches; ++c) value += partials[(ulong(c) * num_total_tokens + token) * columns + rank + tid];
        value = stored(stored(value) / float(branches));
        gates[ulong(token) * branches + tid] = bfloat(2.0f * stored(sigmoid(value)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint chunk = tid % 4, channel = (tid / 4) % branches;
    const uint h = group.x * 16 + tid / 16;
    const device bfloat4* weight = reinterpret_cast<device const bfloat4*>(up + ulong(channel * hidden + h) * rank);
    float dot_product = 0.0f;
    for (uint k = chunk * (rank / 16); k < (chunk + 1) * (rank / 16); ++k) {
        dot_product += dot(float4(low_rank_vectors[k]), float4(weight[k]));
    }
    dot_product += simd_shuffle_xor(dot_product, 1);
    dot_product += simd_shuffle_xor(dot_product, 2);
    const float product = stored(stored(sigmoid(stored(dot_product)))
                                 * float(normalized[ulong(token) * wide + channel * hidden + h]));
    const float p1 = simd_shuffle_xor(product, 4);
    const float p2 = simd_shuffle_xor(product, 8);
    const float p3 = simd_shuffle_xor(product, 12);
    if (tid % 16 == 0) {
        if (pending_write) {
            // Each h has one writer. Read y before overwriting a possibly shared x/y buffer.
            const float y = float(old_hidden[ulong(token) * hidden + h]);
            for (uint c = 0; c < branches; ++c) {
                const float gate = partials[ulong(branches) * num_total_tokens * columns + ulong(token) * branches + c];
                const ulong index = ulong(token) * wide + c * hidden + h;
                written[index] = bfloat(float(old_residual[index]) + stored(gate * y));
            }
        }
        hidden_state[ulong(token) * hidden + h] = bfloat(((product + p1) + p2 + p3) / float(branches));
    }
}
