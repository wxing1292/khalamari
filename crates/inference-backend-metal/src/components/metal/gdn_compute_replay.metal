// Forward produces outputs and a per-token F32 replay log. It does not write full states.
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
    constant uint& num_active_chunkwise_requests [[buffer(13)]],
    device float* replay_k [[buffer(14)]],
    device float* replay_alpha [[buffer(15)]],
    constant ulong& replay_token_offset [[buffer(16)]],
    uint3 threadblock_position [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadblock [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_q[qk_head_dim];
    threadgroup float shared_k[qk_head_dim];
    threadgroup float shared_gate[2];

    const uint qk_dim_thread_index = thread_position_in_threadblock.x;
    const uint local_simdgroup_index = thread_position_in_threadblock.y;
    const uint v_row_range_index =
        threadblock_position.x * replay_num_simdgroups + local_simdgroup_index;
    const uint req_v_head_linear_index = threadblock_position.y;
    const uint v_head_index = req_v_head_linear_index % num_v_heads;
    const uint req_index = req_v_head_linear_index / num_v_heads;
    const uint v_dim_base =
        v_row_range_index * replay_num_v_rows_per_simdgroup;
    if (req_index < num_active_chunkwise_requests || req_index >= num_active_reqs
        || v_dim_base + replay_num_v_rows_per_simdgroup > v_head_dim) {
        return;
    }

    const uint num_qk_dim_threads = replay_num_qk_dim_threads;
    const uint num_state_fragments = (qk_head_dim + num_qk_dim_threads - 1) / num_qk_dim_threads;
    const uint num_v_heads_per_qk_head = num_v_heads / num_qk_heads;
    const uint qk_head_index = v_head_index / num_v_heads_per_qk_head;
    const uint flat_token_begin = cu_tokens[req_index];
    const uint flat_token_end = cu_tokens[req_index + 1];
    const uint replay_flat_token_begin = cu_tokens[num_active_chunkwise_requests];
    const uint q_base = 0;
    const uint k_base = num_qk_heads * qk_head_dim;
    const uint v_base = k_base + num_qk_heads * qk_head_dim;
    const uint recurrent_state_stride = num_v_heads * v_head_dim * qk_head_dim;
    const uint src_state_slot = src_recurrent_state_slots[req_index];
    const ulong recurrent_state_base = recurrent_state_offset_bytes / sizeof(bfloat16_t);

    thread float state_fragments[
        replay_num_v_rows_per_simdgroup * num_state_fragments];
    for (uint v_row_index_in_range = 0;
         v_row_index_in_range < replay_num_v_rows_per_simdgroup;
         ++v_row_index_in_range) {
        const uint v_dim_index = v_dim_base + v_row_index_in_range;
        const ulong state_row_offset = ((ulong)v_head_index * v_head_dim + v_dim_index) * qk_head_dim;
        const ulong source_state_base =
            recurrent_state_base + (ulong)src_state_slot * recurrent_state_stride + state_row_offset;
        for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments;
             ++state_fragment_index) {
            const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
            state_fragments[v_row_index_in_range * num_state_fragments + state_fragment_index] =
                qk_dim_index < qk_head_dim ? recurrent_state_arena[source_state_base + qk_dim_index] : 0.0f;
        }
    }

    for (uint flat_token_index = flat_token_begin; flat_token_index < flat_token_end; ++flat_token_index) {
        float q_square_sum_partial = 0.0f;
        float k_square_sum_partial = 0.0f;
        thread float q_fragments[num_state_fragments];
        thread float k_fragments[num_state_fragments];
        for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments;
             ++state_fragment_index) {
            q_fragments[state_fragment_index] = 0.0f;
            k_fragments[state_fragment_index] = 0.0f;
        }
        if (local_simdgroup_index == 0) {
            for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments;
                 ++state_fragment_index) {
                const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
                const ulong q_value_index =
                    (ulong)flat_token_index * qkv_dim + q_base + qk_head_index * qk_head_dim + qk_dim_index;
                const ulong k_value_index =
                    (ulong)flat_token_index * qkv_dim + k_base + qk_head_index * qk_head_dim + qk_dim_index;
                const float q_value = conv_qkv[q_value_index];
                const float k_value = conv_qkv[k_value_index];
                q_fragments[state_fragment_index] = q_value;
                k_fragments[state_fragment_index] = k_value;
                q_square_sum_partial += q_value * q_value;
                k_square_sum_partial += k_value * k_value;
            }
            const float q_square_sum = simd_broadcast(simd_sum(q_square_sum_partial), 0);
            const float k_square_sum = simd_broadcast(simd_sum(k_square_sum_partial), 0);
            const float q_inv_norm = metal::rsqrt(q_square_sum + 1.0e-6f) * q_scale;
            const float k_inv_norm = metal::rsqrt(k_square_sum + 1.0e-6f);
            for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments;
                 ++state_fragment_index) {
                const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
                q_fragments[state_fragment_index] *= q_inv_norm;
                k_fragments[state_fragment_index] *= k_inv_norm;
                if (qk_dim_index < qk_head_dim) {
                    shared_q[qk_dim_index] = q_fragments[state_fragment_index];
                    shared_k[qk_dim_index] = k_fragments[state_fragment_index];
                    if (threadblock_position.x == 0 && v_head_index % num_v_heads_per_qk_head == 0) {
                        replay_k[((replay_token_offset + (flat_token_index - replay_flat_token_begin)) * num_qk_heads + qk_head_index)
                            * qk_head_dim + qk_dim_index] = k_fragments[state_fragment_index];
                    }
                }
            }
        }

        if (local_simdgroup_index == 0 && qk_dim_thread_index == 0) {
            const ulong gate_index = (ulong)flat_token_index * input_gate_stride + v_head_index;
            const float beta = 1.0f / (1.0f + metal::exp(-b[input_b_offset + gate_index]));
            const float dt = a[input_a_offset + gate_index] + float(dt_bias[v_head_index]);
            const float sp = dt > 20.0f ? dt : metal::log(1.0f + metal::exp(dt));
            const float decay_rate = -metal::exp(float(a_log[v_head_index]));
            shared_gate[0] = beta;
            shared_gate[1] = metal::exp(decay_rate * sp);
            if (threadblock_position.x == 0) {
                replay_alpha[(replay_token_offset + (flat_token_index - replay_flat_token_begin)) * num_v_heads + v_head_index] = shared_gate[1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments;
             ++state_fragment_index) {
            const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
            q_fragments[state_fragment_index] = shared_q[qk_dim_index];
            k_fragments[state_fragment_index] = shared_k[qk_dim_index];
        }
        const float beta = shared_gate[0];
        const float decay = shared_gate[1];

        float v_lane = 0.0f;
        if (qk_dim_thread_index < replay_num_v_rows_per_simdgroup) {
            const uint v_dim_index = v_dim_base + qk_dim_thread_index;
            v_lane = conv_qkv[(ulong)flat_token_index * qkv_dim + v_base
                + (ulong)v_head_index * v_head_dim + v_dim_index];
        }
        for (uint v_row_index_in_range = 0;
             v_row_index_in_range < replay_num_v_rows_per_simdgroup;
             ++v_row_index_in_range) {
            const uint v_dim_index = v_dim_base + v_row_index_in_range;
            float state_k_partial = 0.0f;
            for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments;
                 ++state_fragment_index) {
                const uint fragment_offset = v_row_index_in_range * num_state_fragments + state_fragment_index;
                const float decayed_state = state_fragments[fragment_offset] * decay;
                state_fragments[fragment_offset] = decayed_state;
                state_k_partial += decayed_state * k_fragments[state_fragment_index];
            }
            const float v_value = simd_broadcast(v_lane, v_row_index_in_range);
            const float state_k_dot = simd_broadcast(simd_sum(state_k_partial), 0);
            const float delta = (v_value - state_k_dot) * beta;

            if (qk_dim_thread_index == 0) {
                replay_u[((replay_token_offset + (flat_token_index - replay_flat_token_begin)) * num_v_heads + v_head_index)
                    * v_head_dim + v_dim_index] = delta;
            }
            float state_q_partial = 0.0f;
            for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments;
                 ++state_fragment_index) {
                const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
                const uint fragment_offset = v_row_index_in_range * num_state_fragments + state_fragment_index;
                const float updated_state =
                    state_fragments[fragment_offset] + k_fragments[state_fragment_index] * delta;
                state_fragments[fragment_offset] = updated_state;
                state_q_partial += updated_state * q_fragments[state_fragment_index];
            }
            const float recurrent_output_value = simd_broadcast(simd_sum(state_q_partial), 0);
            if (qk_dim_thread_index == v_row_index_in_range) {
                recurrent_output[
                    ((ulong)flat_token_index * num_v_heads + v_head_index) * v_head_dim + v_dim_index] =
                    bfloat16_t(recurrent_output_value);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
