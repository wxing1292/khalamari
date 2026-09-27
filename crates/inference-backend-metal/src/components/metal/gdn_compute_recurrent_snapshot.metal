// One logical FinalRecurrentStateThreadBlockTask maps 1:1 to one
// threadblock.
// It owns recurrent_state[slot, v_head_index, v_dim_indices, 0..Dqk] and
// advances that state over flat_token_indices in order. The kernel derives the
// task from its arguments, thread-block index, and constants. It does not
// require a materialized task buffer:
//
// FinalRecurrentStateThreadBlockTask {
//   request_index,      // grid-derived from threadblock_position.y / Hv
//   v_head_index,       // grid-derived from threadblock_position.y % Hv
//   v_dim_indices,      // grid-derived half-open range
//   flat_token_indices, // cu_tokens[request_index]..cu_tokens[request_index + 1]
// }
kernel void gdn_compute_final_recurrent_state_bf16(
    device bfloat16_t* recurrent_output [[buffer(0)]],
    device bfloat16_t* recurrent_state_arena [[buffer(1)]],
    device const bfloat16_t* conv_qkv [[buffer(2)]],
    device const bfloat16_t* a [[buffer(3)]],
    device const bfloat16_t* b [[buffer(4)]],
    device const bfloat16_t* a_log [[buffer(5)]],
    device const bfloat16_t* dt_bias [[buffer(6)]],
    device const uint* src_recurrent_state_slots [[buffer(7)]],
    device const uint* flat_recurrent_state_write_slots [[buffer(8)]],
    device const uint* cu_tokens [[buffer(9)]],
    constant float& q_scale [[buffer(10)]],
    constant uint& num_active_reqs [[buffer(11)]],
    constant ulong& recurrent_state_offset_bytes [[buffer(12)]],
    constant uint& num_active_recurrent_requests [[buffer(13)]],
    uint3 threadblock_position [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadblock [[thread_position_in_threadgroup]]
) {
    const ulong recurrent_state_base = recurrent_state_offset_bytes / sizeof(bfloat16_t);
    const uint qk_dim_thread_index = thread_position_in_threadblock.x;
    const uint v_row_index_in_range = thread_position_in_threadblock.y;
    const uint num_qk_dim_threads = final_recurrent_state_num_qk_dim_threads;
    const uint num_state_fragments = (qk_head_dim + num_qk_dim_threads - 1) / num_qk_dim_threads;
    const uint v_row_range_index = threadblock_position.x;
    const uint req_v_head_linear_index = threadblock_position.y;
    const uint v_head_index = req_v_head_linear_index % num_v_heads;
    const uint req_index = req_v_head_linear_index / num_v_heads;
    const uint v_dim_index =
        v_row_range_index * final_recurrent_state_num_v_rows + v_row_index_in_range;
    if (req_index < num_active_recurrent_requests || req_index >= num_active_reqs) {
        return;
    }

    const uint num_v_heads_per_qk_head = num_v_heads / num_qk_heads;
    const uint qk_head_index = v_head_index / num_v_heads_per_qk_head;
    const uint flat_token_begin = cu_tokens[req_index];
    const uint flat_token_end = cu_tokens[req_index + 1];
    const uint q_base = 0;
    const uint k_base = num_qk_heads * qk_head_dim;
    const uint v_base = k_base + num_qk_heads * qk_head_dim;
    const uint recurrent_state_stride = num_v_heads * v_head_dim * qk_head_dim;
    const uint src_state_slot = src_recurrent_state_slots[req_index];
    // An invalid slot keeps the row output but does not materialize its state.
    const uint state_slot = flat_recurrent_state_write_slots[flat_token_end - 1];

    threadgroup float q_inv_norm_shared;
    threadgroup float k_inv_norm_shared;
    threadgroup float decay_shared;
    threadgroup float beta_shared;

    if (v_dim_index >= v_head_dim) {
        return;
    }

    // Each thread owns one strided Dqk fragment. Keep it thread-local across
    // the request's ordered token loop and publish the final state slice once.
    const uint state_row_offset = (v_head_index * v_head_dim + v_dim_index) * qk_head_dim;
    thread float state_fragments[num_state_fragments];
    for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments; ++state_fragment_index) {
        const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
        state_fragments[state_fragment_index] =
            qk_dim_index < qk_head_dim
                ? recurrent_state_arena[
                      recurrent_state_base
                      + (ulong)(src_state_slot * recurrent_state_stride + state_row_offset + qk_dim_index)]
                : 0.0f;
    }

    for (uint flat_token_index = flat_token_begin; flat_token_index < flat_token_end; ++flat_token_index) {
        if (v_row_index_in_range == 0) {
            float q_square_sum_partial = 0.0f;
            float k_square_sum_partial = 0.0f;
            for (uint qk_dim_index = qk_dim_thread_index; qk_dim_index < qk_head_dim; qk_dim_index += num_qk_dim_threads) {
                const uint q_value_index = flat_token_index * qkv_dim + q_base + qk_head_index * qk_head_dim + qk_dim_index;
                const uint k_value_index = flat_token_index * qkv_dim + k_base + qk_head_index * qk_head_dim + qk_dim_index;
                const float q = conv_qkv[q_value_index];
                const float k = conv_qkv[k_value_index];
                q_square_sum_partial += q * q;
                k_square_sum_partial += k * k;
            }
            const float q_square_sum = simd_sum(q_square_sum_partial);
            const float k_square_sum = simd_sum(k_square_sum_partial);
            if (qk_dim_thread_index == 0) {
                const ulong gate_index = (ulong)flat_token_index * input_gate_stride + v_head_index;
                const float beta_t = 1.0f / (1.0f + metal::exp(-b[input_b_offset + gate_index]));
                const float dt = a[input_a_offset + gate_index] + float(dt_bias[v_head_index]);
                const float sp = dt > 20.0f ? dt : metal::log(1.0f + metal::exp(dt));
                const float decay_rate = -metal::exp(float(a_log[v_head_index]));
                q_inv_norm_shared = metal::rsqrt(q_square_sum + 1.0e-6f) * q_scale;
                k_inv_norm_shared = metal::rsqrt(k_square_sum + 1.0e-6f);
                beta_shared = beta_t;
                decay_shared = metal::exp(decay_rate * sp);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float v_t = conv_qkv[flat_token_index * qkv_dim + v_base + v_head_index * v_head_dim + v_dim_index];
        float state_k_partial = 0.0f;
        for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments; ++state_fragment_index) {
            const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
            if (qk_dim_index >= qk_head_dim) {
                continue;
            }
            const uint k_value_index = flat_token_index * qkv_dim + k_base + qk_head_index * qk_head_dim + qk_dim_index;
            const float k_norm = conv_qkv[k_value_index] * k_inv_norm_shared;
            const float decayed_state = state_fragments[state_fragment_index] * decay_shared;
            state_fragments[state_fragment_index] = decayed_state;
            state_k_partial += decayed_state * k_norm;
        }
        const float state_k_dot = simd_sum(state_k_partial);
        const float delta = (v_t - simd_broadcast(state_k_dot, 0)) * beta_shared;

        float state_q_partial = 0.0f;
        for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments; ++state_fragment_index) {
            const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
            if (qk_dim_index >= qk_head_dim) {
                continue;
            }
            const uint q_value_index = flat_token_index * qkv_dim + q_base + qk_head_index * qk_head_dim + qk_dim_index;
            const uint k_value_index = flat_token_index * qkv_dim + k_base + qk_head_index * qk_head_dim + qk_dim_index;
            const float k_norm = conv_qkv[k_value_index] * k_inv_norm_shared;
            const float q_norm = conv_qkv[q_value_index] * q_inv_norm_shared;
            const float updated_state = state_fragments[state_fragment_index] + k_norm * delta;
            state_fragments[state_fragment_index] = updated_state;
            state_q_partial += updated_state * q_norm;
        }
        const float recurrent_output_value = simd_sum(state_q_partial);
        if (qk_dim_thread_index == 0) {
            recurrent_output[(flat_token_index * num_v_heads + v_head_index) * v_head_dim + v_dim_index] =
                bfloat16_t(recurrent_output_value);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (state_slot != GDN_INVALID_STATE_SLOT_ID) {
        for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments; ++state_fragment_index) {
            const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
            if (qk_dim_index < qk_head_dim) {
                recurrent_state_arena[
                    recurrent_state_base
                    + (ulong)(state_slot * recurrent_state_stride + state_row_offset + qk_dim_index)] =
                    bfloat16_t(state_fragments[state_fragment_index]);
            }
        }
    }
}

// One logical CandidateRecurrentStateThreadBlockTask maps 1:1 to one
// threadblock. It owns
// recurrent_state[slot, v_head_index, v_dim_indices, 0..Dqk], advances
// flat_token_indices in order, and can materialize state after each token.
// Grid x derives v_dim_indices. Grid y derives request_index and v_head_index.
// Each local SIMDgroup owns
// candidate_recurrent_state_num_v_rows_per_simdgroup V
// rows. The kernel derives the task from its arguments, thread-block index,
// and constants.
// It does not require a materialized task buffer.
kernel void gdn_compute_candidate_recurrent_state_bf16(
    device bfloat16_t* recurrent_output [[buffer(0)]],
    device bfloat16_t* recurrent_state_arena [[buffer(1)]],
    device const bfloat16_t* conv_qkv [[buffer(2)]],
    device const bfloat16_t* a [[buffer(3)]],
    device const bfloat16_t* b [[buffer(4)]],
    device const bfloat16_t* a_log [[buffer(5)]],
    device const bfloat16_t* dt_bias [[buffer(6)]],
    device const uint* src_recurrent_state_slots [[buffer(7)]],
    device const uint* flat_recurrent_state_write_slots [[buffer(8)]],
    device const uint* cu_tokens [[buffer(9)]],
    constant float& q_scale [[buffer(10)]],
    constant uint& num_active_reqs [[buffer(11)]],
    constant ulong& recurrent_state_offset_bytes [[buffer(12)]],
    constant uint& num_active_recurrent_requests [[buffer(13)]],
    uint3 threadblock_position [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadblock [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_q[qk_head_dim];
    threadgroup float shared_k[qk_head_dim];
    threadgroup float shared_gate[2];

    const uint qk_dim_thread_index = thread_position_in_threadblock.x;
    const uint local_simdgroup_index = thread_position_in_threadblock.y;
    const uint v_row_range_index =
        threadblock_position.x * candidate_recurrent_state_num_simdgroups + local_simdgroup_index;
    const uint req_v_head_linear_index = threadblock_position.y;
    const uint v_head_index = req_v_head_linear_index % num_v_heads;
    const uint req_index = req_v_head_linear_index / num_v_heads;
    const uint v_dim_base =
        v_row_range_index * candidate_recurrent_state_num_v_rows_per_simdgroup;
    if (req_index < num_active_recurrent_requests || req_index >= num_active_reqs
        || v_dim_base + candidate_recurrent_state_num_v_rows_per_simdgroup > v_head_dim) {
        return;
    }

    const uint num_qk_dim_threads = candidate_recurrent_state_num_qk_dim_threads;
    const uint num_state_fragments = (qk_head_dim + num_qk_dim_threads - 1) / num_qk_dim_threads;
    const uint num_v_heads_per_qk_head = num_v_heads / num_qk_heads;
    const uint qk_head_index = v_head_index / num_v_heads_per_qk_head;
    const uint flat_token_begin = cu_tokens[req_index];
    const uint flat_token_end = cu_tokens[req_index + 1];
    const uint q_base = 0;
    const uint k_base = num_qk_heads * qk_head_dim;
    const uint v_base = k_base + num_qk_heads * qk_head_dim;
    const uint recurrent_state_stride = num_v_heads * v_head_dim * qk_head_dim;
    const uint src_state_slot = src_recurrent_state_slots[req_index];
    const ulong recurrent_state_base = recurrent_state_offset_bytes / sizeof(bfloat16_t);

    thread float state_fragments[
        candidate_recurrent_state_num_v_rows_per_simdgroup * num_state_fragments];
    for (uint v_row_index_in_range = 0;
         v_row_index_in_range < candidate_recurrent_state_num_v_rows_per_simdgroup;
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
        if (qk_dim_thread_index < candidate_recurrent_state_num_v_rows_per_simdgroup) {
            const uint v_dim_index = v_dim_base + qk_dim_thread_index;
            v_lane = conv_qkv[(ulong)flat_token_index * qkv_dim + v_base
                + (ulong)v_head_index * v_head_dim + v_dim_index];
        }
        const uint candidate_state_slot = flat_recurrent_state_write_slots[flat_token_index];
        for (uint v_row_index_in_range = 0;
             v_row_index_in_range < candidate_recurrent_state_num_v_rows_per_simdgroup;
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

            float state_q_partial = 0.0f;
            const ulong state_row_offset = ((ulong)v_head_index * v_head_dim + v_dim_index) * qk_head_dim;
            for (uint state_fragment_index = 0; state_fragment_index < num_state_fragments;
                 ++state_fragment_index) {
                const uint qk_dim_index = qk_dim_thread_index + state_fragment_index * num_qk_dim_threads;
                const uint fragment_offset = v_row_index_in_range * num_state_fragments + state_fragment_index;
                const float updated_state =
                    state_fragments[fragment_offset] + k_fragments[state_fragment_index] * delta;
                state_fragments[fragment_offset] = updated_state;
                if (candidate_state_slot != GDN_INVALID_STATE_SLOT_ID && qk_dim_index < qk_head_dim) {
                    recurrent_state_arena[
                        recurrent_state_base + (ulong)candidate_state_slot * recurrent_state_stride
                        + state_row_offset + qk_dim_index] = bfloat16_t(updated_state);
                }
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
