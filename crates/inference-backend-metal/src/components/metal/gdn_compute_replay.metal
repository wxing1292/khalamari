// SPDX-License-Identifier: Apache-2.0
// Fused F32 normalization/gates, request metadata, BF16 state
// arenas, and per-token F32 alpha/K/U logs for accepted-prefix commit.
// Blocked sequential recurrence. F32 inputs and register state.
kernel void gdn_compute_replay_bf16(
    device bfloat16_t* recurrent_output [[buffer(0)]],
    device const bfloat16_t* recurrent_state_arena [[buffer(1)]],
    device const bfloat16_t* conv_qkv [[buffer(2)]],
    device const bfloat16_t* a [[buffer(3)]],
    device const bfloat16_t* b [[buffer(4)]],
    device const bfloat16_t* a_log [[buffer(5)]],
    device const bfloat16_t* dt_bias [[buffer(6)]],
    device const uint* src_recurrent_state_slots [[buffer(7)]],
    device float* replay_u [[buffer(8)]],
    device const uint* cu_tokens [[buffer(9)]],
    constant float& q_scale [[buffer(10)]],
    constant uint& num_active_reqs [[buffer(11)]],
    constant ulong& recurrent_state_offset_bytes [[buffer(12)]],
    constant uint& num_active_recurrent_requests [[buffer(13)]],
    device float* replay_k [[buffer(14)]],
    device float* replay_alpha [[buffer(15)]],
    constant ulong& replay_token_offset [[buffer(16)]],
    uint3 threadblock_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    // Each lane group owns one V row. State stays F32 for the whole request.
    constexpr uint lanes_per_row = replay_lanes_per_row;
    constexpr uint values_per_lane = qk_head_dim / lanes_per_row;
    constexpr uint vectors_per_lane = values_per_lane / 4;
    constexpr uint token_tile = replay_token_tile;
    constexpr uint qk_stride = qk_head_dim + 4;
    constexpr uint storage_rows = replay_num_simdgroups * 32 / lanes_per_row;
    constexpr uint value_stride = storage_rows + 8;
    constexpr uint num_threads = replay_num_simdgroups * 32;
    const uint req = threadblock_position.y / num_v_heads;
    if (req < num_active_recurrent_requests || req >= num_active_reqs) return;
    const uint head = threadblock_position.y % num_v_heads;
    const uint qk_head = head / (num_v_heads / num_qk_heads);
    const uint row = thread_index / lanes_per_row;
    const uint segment = thread_index % lanes_per_row;
    const uint column = segment * values_per_lane;
    const uint value_base = threadblock_position.x * replay_num_v_rows;
    const uint begin = cu_tokens[req];
    const uint end = cu_tokens[req + 1];
    const uint replay_begin = cu_tokens[num_active_recurrent_requests];
    const ulong arena_base = recurrent_state_offset_bytes / sizeof(bfloat16_t);
    const ulong state_stride = (ulong)num_v_heads * v_head_dim * qk_head_dim;
    const ulong state_row = ((ulong)head * v_head_dim + value_base + row) * qk_head_dim + column;
    const uint key_base = num_qk_heads * qk_head_dim;
    const uint value_channel_base = 2 * key_base + head * v_head_dim + value_base;
    const bool valid_row = row < replay_num_v_rows;
    threadgroup float shared_q[token_tile * qk_stride];
    threadgroup float shared_k[token_tile * qk_stride];
    threadgroup bfloat shared_v[token_tile * value_stride];
    threadgroup float shared_decay[token_tile];
    threadgroup float shared_beta[token_tile];
    float4 state[vectors_per_lane];
    #pragma unroll
    for (uint i = 0; i < vectors_per_lane; ++i) {
        state[i] = valid_row ? float4(*reinterpret_cast<device const bfloat4*>(
            recurrent_state_arena + arena_base + (ulong)src_recurrent_state_slots[req] * state_stride
                + state_row + 4 * i)) : float4(0.0f);
    }
    const float decay_rate = -metal::exp(float(a_log[head]));
    const float bias = float(dt_bias[head]);
    for (uint start = begin; start < end; start += token_tile) {
        const uint count = min(token_tile, end - start);
        // One SIMDgroup normalizes one token. All V rows reuse these F32 inputs.
        for (uint t = simdgroup_index; t < count; t += replay_num_simdgroups) {
            float q2 = 0.0f, k2 = 0.0f;
            float q_values[qk_head_dim / 32];
            float k_values[qk_head_dim / 32];
            #pragma unroll
            for (uint i = 0; i < qk_head_dim / 32; ++i) {
                const uint d = lane + i * 32;
                const ulong address = (ulong)(start + t) * qkv_dim + qk_head * qk_head_dim + d;
                const float q = float(conv_qkv[address]);
                const float k = float(conv_qkv[address + key_base]);
                q_values[i] = q;
                k_values[i] = k;
                q2 += q * q;
                k2 += k * k;
            }
            const float q_norm = metal::rsqrt(simd_sum(q2) + 1.0e-6f) * q_scale;
            const float k_norm = metal::rsqrt(simd_sum(k2) + 1.0e-6f);
            #pragma unroll
            for (uint i = 0; i < qk_head_dim / 32; ++i) {
                const uint d = lane + i * 32;
                shared_q[t * qk_stride + d] = q_values[i] * q_norm;
                const float normalized_k = k_values[i] * k_norm;
                shared_k[t * qk_stride + d] = normalized_k;
                if (threadblock_position.x == 0 && head % (num_v_heads / num_qk_heads) == 0) {
                    replay_k[((replay_token_offset + start + t - replay_begin) * num_qk_heads + qk_head) * qk_head_dim + d] = normalized_k;
                }
            }
            if (lane == 0) {
                const ulong address = (ulong)(start + t) * input_gate_stride + head;
                const float dt = float(a[input_a_offset + address]) + bias;
                const float softplus = dt > 20.0f ? dt : metal::log(1.0f + metal::exp(dt));
                shared_decay[t] = metal::exp(decay_rate * softplus);
                if (threadblock_position.x == 0) {
                    replay_alpha[(replay_token_offset + start + t - replay_begin) * num_v_heads + head] = shared_decay[t];
                }
                shared_beta[t] = 1.0f / (1.0f + metal::exp(-float(b[input_b_offset + address])));
            }
        }
        for (uint i = thread_index; i < count * storage_rows; i += num_threads) {
            const uint t = i / storage_rows;
            const uint r = i % storage_rows;
            shared_v[t * value_stride + r] = r < replay_num_v_rows
                ? conv_qkv[(ulong)(start + t) * qkv_dim + value_channel_base + r] : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < count; ++t) {
            float4 keys[vectors_per_lane];
            float4 partial = 0.0f;
            const float decay = shared_decay[t];
            #pragma unroll
            for (uint i = 0; i < vectors_per_lane; ++i) {
                keys[i] = *reinterpret_cast<threadgroup const float4*>(shared_k + t * qk_stride + column + i * 4);
                state[i] *= decay;
                partial += state[i] * keys[i];
            }
            float dot = partial.x + partial.y + partial.z + partial.w;
            #pragma unroll
            for (uint offset = lanes_per_row / 2; offset > 0; offset /= 2) dot += simd_shuffle_down(dot, offset);
            dot = simd_shuffle(dot, lane / lanes_per_row * lanes_per_row);
            const float delta = (float(shared_v[t * value_stride + row]) - dot) * shared_beta[t];
            if (segment == 0 && valid_row) {
                replay_u[((replay_token_offset + start + t - replay_begin) * num_v_heads + head) * v_head_dim + value_base + row] = delta;
            }
            float4 output = 0.0f;
            #pragma unroll
            for (uint i = 0; i < vectors_per_lane; ++i) {
                state[i] += keys[i] * delta;
                output += state[i] * *reinterpret_cast<threadgroup const float4*>(
                    shared_q + t * qk_stride + column + i * 4);
            }
            float result = output.x + output.y + output.z + output.w;
            #pragma unroll
            for (uint offset = lanes_per_row / 2; offset > 0; offset /= 2) result += simd_shuffle_down(result, offset);
            if (segment == 0 && valid_row) {
                recurrent_output[((ulong)(start + t) * num_v_heads + head) * v_head_dim + value_base + row] = bfloat(result);
            }

        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
