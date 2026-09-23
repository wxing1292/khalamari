#include <metal_stdlib>
using namespace metal;

// Round at eager tensor boundaries, including register-only BF16 intermediates.
float stored(float x) { return float(scalar_t(x)); }
float sigmoid(float x) { return 1.0f / (1.0f + metal::exp(-x)); }

template <uint width>
float dot_lane(device const scalar_t* input, device const scalar_t* weight, uint lane, uint stride) {
    float sum = 0.0f;
    if (width % 4 == 0) {
        // Row starts are vector-aligned because each row has exactly `width` elements.
        const device vec<scalar_t, 4>* input4 = reinterpret_cast<const device vec<scalar_t, 4>*>(input);
        const device vec<scalar_t, 4>* weight4 = reinterpret_cast<const device vec<scalar_t, 4>*>(weight);
        for (size_t column = lane; column < width / 4; column += stride) {
            sum += dot(float4(input4[column]), float4(weight4[column]));
        }
    } else {
        for (size_t column = lane; column < width; column += stride) {
            sum += float(input[column]) * float(weight[column]);
        }
    }
    return sum;
}

float block_sum(float value, threadgroup float* sums, uint lane, uint simd_group) {
    value = simd_sum(value);
    if (lane == 0) sums[simd_group] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = lane < simd_groups ? sums[lane] : 0.0f;
    return simd_sum(value);
}

kernel void gr_normalize(
    device const scalar_t* residual [[buffer(0)]],
    device const scalar_t* weight [[buffer(1)]],
    device scalar_t* normalized [[buffer(2)]],
    constant uint& num_active_tokens [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (group.y >= num_active_tokens) return;
    const size_t base = size_t(group.y) * wide + size_t(group.x) * hidden;
    float sum = 0.0f;
    for (size_t column = tid.x; column < hidden; column += threads) {
        const float value = float(residual[base + column]);
        sum += value * value;
    }
    threadgroup float sums[simd_groups];
    sum = block_sum(sum, sums, lane, simd_group);
    const float inv_rms = metal::precise::rsqrt(sum / float(hidden) + eps);
    for (size_t column = tid.x; column < hidden; column += threads) {
        const float scale = 1.0f + float(weight[size_t(group.x) * hidden + column]);
        normalized[base + column] = scalar_t((float(residual[base + column]) * inv_rms) * scale);
    }
}

void down_row(device const scalar_t* normalized, device const scalar_t* weight,
              device scalar_t* low_rank, uint token, size_t row, uint lane) {
    float sum = dot_lane<wide>(normalized + size_t(token) * wide, weight + row * wide, lane, 32);
    sum = simd_sum(sum);
    if (lane == 0) {
        const float scaled = stored(stored(sum) / float(branches));
        low_rank[size_t(token) * rank + row] = scalar_t(scaled * sigmoid(scaled));
    }
}

kernel void gr_down(
    device const scalar_t* normalized [[buffer(0)]],
    device const scalar_t* weight [[buffer(1)]],
    device scalar_t* low_rank [[buffer(2)]],
    constant uint& num_active_tokens [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (group.y >= num_active_tokens) return;
    const size_t row = size_t(group.x) * simd_groups + simd_group;
    if (row < rank) down_row(normalized, weight, low_rank, group.y, row, lane);
}

kernel void gr_down_gate(
    device const scalar_t* normalized [[buffer(0)]],
    device const scalar_t* weight [[buffer(1)]],
    device scalar_t* low_rank [[buffer(2)]],
    constant uint& num_active_tokens [[buffer(3)]],
    device const scalar_t* write_weight [[buffer(4)]],
    device scalar_t* gate [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (group.y >= num_active_tokens) return;
    const size_t row = size_t(group.x) * simd_groups + simd_group;
    if (row < rank) {
        down_row(normalized, weight, low_rank, group.y, row, lane);
    } else if (row - rank < branches) {
        float sum = dot_lane<wide>(normalized + size_t(group.y) * wide,
                                  write_weight + (row - rank) * wide, lane, 32);
        sum = simd_sum(sum);
        if (lane == 0) {
            const float scaled = stored(stored(sum) / float(branches));
            gate[size_t(group.y) * branches + row - rank] = scalar_t(2.0f * stored(sigmoid(scaled)));
        }
    }
}

kernel void gr_read(
    device const scalar_t* normalized [[buffer(0)]],
    device const scalar_t* low_rank [[buffer(1)]],
    device const scalar_t* weight [[buffer(2)]],
    device scalar_t* output [[buffer(3)]],
    constant uint& num_active_tokens [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (group.y >= num_active_tokens) return;
    const size_t column = size_t(group.x) * simd_groups + simd_group;
    if (column >= hidden) return;
    float mixed = 0.0f;
    for (uint branch = 0; branch < branches; ++branch) {
        const size_t row = size_t(branch) * hidden + column;
        float sum = dot_lane<rank>(low_rank + size_t(group.y) * rank, weight + row * rank, lane, 32);
        sum = simd_sum(sum);
        if (lane == 0) {
            const float gate = stored(sigmoid(stored(sum)));
            mixed += stored(gate * float(normalized[size_t(group.y) * wide + row]));
        }
    }
    if (lane == 0) output[size_t(group.y) * hidden + column] = scalar_t(mixed / float(branches));
}

kernel void gr_write(
    device const scalar_t* residual [[buffer(0)]],
    device const scalar_t* gates [[buffer(1)]],
    device const scalar_t* hidden_state [[buffer(2)]],
    device scalar_t* output [[buffer(3)]],
    constant uint& num_active_tokens [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    if (group.y >= num_active_tokens) return;
    const float gate = float(gates[size_t(group.y) * branches + group.x]);
    const size_t base = size_t(group.y) * wide + size_t(group.x) * hidden;
    for (size_t column = tid.x; column < hidden; column += threads) {
        const float update = stored(gate * float(hidden_state[size_t(group.y) * hidden + column]));
        output[base + column] = scalar_t(float(residual[base + column]) + update);
    }
}

kernel void gr_write_norm(
    device const scalar_t* residual [[buffer(0)]],
    device const scalar_t* gates [[buffer(1)]],
    device const scalar_t* hidden_state [[buffer(2)]],
    device scalar_t* output [[buffer(3)]],
    constant uint& num_active_tokens [[buffer(4)]],
    device const scalar_t* weight [[buffer(5)]],
    device scalar_t* normalized [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (group.y >= num_active_tokens) return;
    const float gate = float(gates[size_t(group.y) * branches + group.x]);
    const size_t base = size_t(group.y) * wide + size_t(group.x) * hidden;
    float sum = 0.0f;
    for (size_t column = tid.x; column < hidden; column += threads) {
        const float update = stored(gate * float(hidden_state[size_t(group.y) * hidden + column]));
        // Preserve the unfused residual storage boundary before normalization.
        const float value = stored(float(residual[base + column]) + update);
        output[base + column] = scalar_t(value);
        sum += value * value;
    }
    threadgroup float sums[simd_groups];
    sum = block_sum(sum, sums, lane, simd_group);
    const float inv_rms = metal::precise::rsqrt(sum / float(hidden) + eps);
    for (size_t column = tid.x; column < hidden; column += threads) {
        const float scale = 1.0f + float(weight[size_t(group.x) * hidden + column]);
        normalized[base + column] = scalar_t((float(output[base + column]) * inv_rms) * scale);
    }
}
