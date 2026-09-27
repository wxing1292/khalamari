#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// Sixteen-token gated WY chunks. BF16 matrix operands with F32 accumulation.
// Live state, normalization, gates, and the triangular inverse remain F32.
// Each SIMDgroup owns eight V rows in persistent cooperative state tiles.
kernel void gdn_compute_chunkwise_state_bf16(
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
    constant uint& num_active_chunkwise_requests [[buffer(11)]],
    constant ulong& recurrent_state_offset_bytes [[buffer(12)]],
    constant uint& write_candidate_states [[buffer(13)]],
    uint3 threadblock_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint matrix_size = 8;
    // Keep each state-update destination within 128 Q/K columns. Wider heads
    // retain multiple F32 cooperative tensors and use the same TensorOps path.
    constexpr uint state_qk_tile_size = chunkwise_state_qk_tile_size;
    constexpr uint num_state_tiles = qk_head_dim / state_qk_tile_size;
    const uint num_simdgroups = chunkwise_state_num_simdgroups;
    const uint num_threads = chunkwise_state_num_qk_dim_threads * num_simdgroups;
    const uint v_tile_storage_rows = num_simdgroups * matrix_size;
    const uint v_row_range_index = threadblock_position.x;
    const uint req_v_head_linear_index = threadblock_position.y;
    const uint v_head_index = req_v_head_linear_index % num_v_heads;
    const uint req_index = req_v_head_linear_index / num_v_heads;
    if (req_index >= num_active_chunkwise_requests) {
        return;
    }

    const uint num_v_heads_per_qk_head = num_v_heads / num_qk_heads;
    const uint qk_head_index = v_head_index / num_v_heads_per_qk_head;
    const uint flat_token_begin = cu_tokens[req_index];
    const uint flat_token_end = cu_tokens[req_index + 1];
    const uint q_base = 0;
    const uint k_base = num_qk_heads * qk_head_dim;
    const uint v_base = k_base + num_qk_heads * qk_head_dim;
    const ulong recurrent_state_stride = (ulong)num_v_heads * v_head_dim * qk_head_dim;
    const uint src_state_slot = src_recurrent_state_slots[req_index];
    const ulong recurrent_state_base = recurrent_state_offset_bytes / sizeof(bfloat16_t);
    const uint v_dim_base = v_row_range_index * chunkwise_state_num_v_rows;

    threadgroup bfloat normalized_k[chunkwise_token_chunk_size * qk_head_dim];
    threadgroup bfloat transformed_vectors[chunkwise_token_chunk_size * qk_head_dim];
    threadgroup float transform[chunkwise_token_chunk_size * chunkwise_token_chunk_size];
    threadgroup float inverse_diagonal[chunkwise_token_chunk_size * chunkwise_token_chunk_size];
    threadgroup bfloat shared_v[chunkwise_token_chunk_size * chunkwise_state_num_simdgroups * matrix_size];
    threadgroup bfloat state_operands[chunkwise_state_num_simdgroups * matrix_size * qk_head_dim];
    threadgroup bfloat weighted_transform[chunkwise_token_chunk_size * chunkwise_token_chunk_size];
    threadgroup float q_inv_norm[chunkwise_token_chunk_size];
    threadgroup float cumulative_log_decay[chunkwise_token_chunk_size];
    threadgroup float beta[chunkwise_token_chunk_size];

    static_assert(chunkwise_token_chunk_size == 16);
    static_assert(chunkwise_state_num_v_rows <= 64);
    tensor<threadgroup bfloat, extents<int, qk_head_dim, 16>, tensor_inline> k_tile(
        normalized_k, extents<int, qk_head_dim, 16>{});
    tensor<threadgroup bfloat, extents<int, qk_head_dim, 16>, tensor_inline> vectors(
        transformed_vectors, extents<int, qk_head_dim, 16>{});
    tensor<threadgroup float, extents<int, 16, 16>, tensor_inline> transform_tile(
        transform, extents<int, 16, 16>{});
    tensor<threadgroup float, extents<int, 16, 16>, tensor_inline> inverse_diagonal_tile(
        inverse_diagonal, extents<int, 16, 16>{});
    tensor<threadgroup bfloat, extents<int, 8, 16>, tensor_inline> v_tile(
        shared_v + simdgroup_index * 8, extents<int, 8, 16>{}, array<int, 2>{1, int(v_tile_storage_rows)});

    tensor<threadgroup bfloat, extents<int, 16, 16>, tensor_inline> weighted_tile(
        weighted_transform, extents<int, 16, 16>{});

    constexpr auto gram_descriptor = matmul2d_descriptor(16, 16, qk_head_dim, false, true, false);
    constexpr auto vectors_descriptor = matmul2d_descriptor(16, 16, 16, false, false, false);
    constexpr auto state_vectors_descriptor = matmul2d_descriptor(8, 16, qk_head_dim, false, true, false);
    constexpr auto values_descriptor = matmul2d_descriptor(8, 16, 16, true, true, false);
    constexpr auto output_descriptor = matmul2d_descriptor(8, 16, 16, true, true, false);
    constexpr auto update_descriptor = matmul2d_descriptor(
        8, state_qk_tile_size, 16, true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<gram_descriptor, execution_simdgroup> gram_op;
    matmul2d<vectors_descriptor, execution_simdgroup> vectors_op;
    matmul2d<state_vectors_descriptor, execution_simdgroup> state_vectors_op;
    matmul2d<values_descriptor, execution_simdgroup> values_op;
    matmul2d<output_descriptor, execution_simdgroup> output_op;
    matmul2d<update_descriptor, execution_simdgroup> update_op;
    using State = decltype(update_op.get_destination_cooperative_tensor<decltype(v_tile), decltype(k_tile), float>());
    GDN_DECLARE_STATE_TILES
    #pragma unroll
    for (uint state_tile_index = 0; state_tile_index < num_state_tiles; ++state_tile_index) {
        thread auto& state = *states[state_tile_index];
        for (auto it = state.begin(); it != state.end(); ++it) {
            if (!state.is_valid_element(it)) continue;
            const auto coordinate = it.get_multidimensional_index();
            const uint row = simdgroup_index * matrix_size + uint(coordinate[1]);
            const uint col = state_tile_index * state_qk_tile_size + uint(coordinate[0]);
            const ulong address = recurrent_state_base + (ulong)src_state_slot * recurrent_state_stride
                + ((ulong)v_head_index * v_head_dim + v_dim_base + row) * qk_head_dim + col;
            *it = row < chunkwise_state_num_v_rows ? float(recurrent_state_arena[address]) : 0.0f;
        }
    }

    for (uint chunk_start = flat_token_begin; chunk_start < flat_token_end;) {
        uint num_chunk_tokens = min(chunkwise_token_chunk_size, flat_token_end - chunk_start);
        // A materialized row ends the chunk. The next chunk retains the F32
        // cooperative state and does not reload the rounded BF16 checkpoint.
        if (write_candidate_states != 0) {
            for (uint token_index = 0; token_index < num_chunk_tokens; ++token_index) {
                if (flat_recurrent_state_write_slots[chunk_start + token_index] != GDN_INVALID_STATE_SLOT_ID) {
                    num_chunk_tokens = token_index + 1;
                    break;
                }
            }
        }

        for (uint value_index = thread_index;
             value_index < chunkwise_token_chunk_size * qk_head_dim;
             value_index += num_threads) {
            const uint token_index_in_chunk = value_index / qk_head_dim;
            if (token_index_in_chunk >= num_chunk_tokens) {
                normalized_k[value_index] = bfloat(0.0f);
            }
        }
        for (uint value_index = thread_index;
             value_index < chunkwise_token_chunk_size * v_tile_storage_rows;
             value_index += num_threads) {
            const uint token_index_in_chunk = value_index / v_tile_storage_rows;
            const uint v_row_index = value_index - token_index_in_chunk * v_tile_storage_rows;
            shared_v[value_index] = bfloat(token_index_in_chunk < num_chunk_tokens
                    && v_row_index < chunkwise_state_num_v_rows
                ? float(conv_qkv[
                      (ulong)(chunk_start + token_index_in_chunk) * qkv_dim + v_base
                      + v_head_index * v_head_dim + v_dim_base + v_row_index])
                : 0.0f);
        }
        for (uint token_index_in_chunk = simdgroup_index;
             token_index_in_chunk < num_chunk_tokens;
             token_index_in_chunk += num_simdgroups) {
            const uint flat_token_index = chunk_start + token_index_in_chunk;
            float q_square_sum_partial = 0.0f;
            float k_square_sum_partial = 0.0f;
            for (uint qk_dim_index = lane; qk_dim_index < qk_head_dim;
                 qk_dim_index += chunkwise_state_num_qk_dim_threads) {
                const ulong q_value_index =
                    (ulong)flat_token_index * qkv_dim + q_base + qk_head_index * qk_head_dim + qk_dim_index;
                const ulong k_value_index =
                    (ulong)flat_token_index * qkv_dim + k_base + qk_head_index * qk_head_dim + qk_dim_index;
                const float q_value = float(conv_qkv[q_value_index]);
                const float k_value = float(conv_qkv[k_value_index]);
                q_square_sum_partial += q_value * q_value;
                k_square_sum_partial += k_value * k_value;
            }
            const float q_square_sum = simd_sum(q_square_sum_partial);
            const float k_square_sum = simd_sum(k_square_sum_partial);
            const float k_inverse_norm = metal::rsqrt(k_square_sum + 1.0e-6f);
            for (uint qk_dim_index = lane; qk_dim_index < qk_head_dim;
                 qk_dim_index += chunkwise_state_num_qk_dim_threads) {
                const ulong k_value_index =
                    (ulong)flat_token_index * qkv_dim + k_base + qk_head_index * qk_head_dim + qk_dim_index;
                normalized_k[token_index_in_chunk * qk_head_dim + qk_dim_index] =
                    bfloat(float(conv_qkv[k_value_index]) * k_inverse_norm);
            }
            if (lane == 0) {
                const ulong gate_index = (ulong)flat_token_index * input_gate_stride + v_head_index;
                const float dt = float(a[input_a_offset + gate_index]) + float(dt_bias[v_head_index]);
                const float softplus = dt > 20.0f ? dt : metal::log(1.0f + metal::exp(dt));
                const float decay_rate = -metal::exp(float(a_log[v_head_index]));
                q_inv_norm[token_index_in_chunk] = metal::rsqrt(q_square_sum + 1.0e-6f) * q_scale;
                cumulative_log_decay[token_index_in_chunk] = decay_rate * softplus;
                beta[token_index_in_chunk] = 1.0f / (1.0f + metal::exp(-float(b[input_b_offset + gate_index])));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_index == 0) {
            float log_decay_sum = 0.0f;
            for (uint token_index_in_chunk = 0; token_index_in_chunk < num_chunk_tokens;
                 ++token_index_in_chunk) {
                log_decay_sum += cumulative_log_decay[token_index_in_chunk];
                cumulative_log_decay[token_index_in_chunk] = log_decay_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simdgroup_index == 0) {
            gram_op.run(k_tile, k_tile, transform_tile);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint matrix_index = thread_index;
             matrix_index < chunkwise_token_chunk_size * chunkwise_token_chunk_size;
             matrix_index += num_threads) {
            const uint row = matrix_index / chunkwise_token_chunk_size;
            const uint col = matrix_index - row * chunkwise_token_chunk_size;
            if (row >= num_chunk_tokens || col >= num_chunk_tokens || row < col) {
                transform[matrix_index] = 0.0f;
            } else if (row == col) {
                transform[matrix_index] = 1.0f;
            } else {
                transform[matrix_index] = -beta[row]
                    * metal::exp(cumulative_log_decay[row] - cumulative_log_decay[col])
                    * transform[matrix_index];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Invert independent 8x8 diagonal blocks with F32 forward substitution.
        // Merge the off-diagonal block with matrix products.
        for (uint matrix_index = thread_index;
             matrix_index < chunkwise_token_chunk_size * chunkwise_token_chunk_size;
             matrix_index += num_threads) {
            inverse_diagonal[matrix_index] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint block_index = simdgroup_index; block_index < 2; block_index += num_simdgroups) {
            const uint block_base = block_index * 8;
            float column[8];
            if (lane < 8) {
                for (uint row = 0; row < 8; ++row) {
                    column[row] = transform[(block_base + row) * 16 + block_base + lane];
                }
                for (uint row = lane + 1; row < 8; ++row) {
                    float value = column[row];
                    for (uint inner = lane + 1; inner < row; ++inner) {
                        value += transform[(block_base + row) * 16 + block_base + inner] * column[inner];
                    }
                    column[row] = value;
                }
                for (uint row = 0; row < 8; ++row) {
                    inverse_diagonal[(block_base + row) * 16 + block_base + lane] = column[row];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint matrix_index = thread_index;
             matrix_index < chunkwise_token_chunk_size * chunkwise_token_chunk_size;
             matrix_index += num_threads) {
            if (matrix_index / 16 < 8 || matrix_index % 16 >= 8) {
                transform[matrix_index] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup_index == 0) {
            constexpr auto inverse_descriptor = matmul2d_descriptor(16, 16, 16, false, false, false);
            matmul2d<inverse_descriptor, execution_simdgroup> inverse_op;
            auto product = inverse_op.get_destination_cooperative_tensor<decltype(inverse_diagonal_tile), decltype(transform_tile), float>();
            inverse_op.run(inverse_diagonal_tile, transform_tile, product);
            auto left = inverse_op.get_left_input_cooperative_tensor<float, float, float>(product);
            auto merged = inverse_op.get_destination_cooperative_tensor<decltype(left), decltype(inverse_diagonal_tile), float>();
            inverse_op.run(left, inverse_diagonal_tile, merged);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (auto it = merged.begin(); it != merged.end(); ++it) {
                if (!merged.is_valid_element(it)) continue;
                const auto coordinate = it.get_multidimensional_index();
                const uint matrix_index = uint(coordinate[1]) * 16 + uint(coordinate[0]);
                transform[matrix_index] = inverse_diagonal[matrix_index] + *it;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint matrix_index = thread_index;
             matrix_index < chunkwise_token_chunk_size * chunkwise_token_chunk_size;
             matrix_index += num_threads) {
            const uint row = matrix_index / chunkwise_token_chunk_size;
            const uint col = matrix_index - row * chunkwise_token_chunk_size;
            weighted_transform[matrix_index] = bfloat(row < num_chunk_tokens && col <= row
                ? transform[matrix_index] * beta[col]
                : 0.0f);
        }
        for (uint value_index = thread_index;
             value_index < chunkwise_token_chunk_size * qk_head_dim;
             value_index += num_threads) {
            const uint row = value_index / qk_head_dim;
            transformed_vectors[value_index] = bfloat(row < num_chunk_tokens
                ? metal::exp(cumulative_log_decay[row]) * float(normalized_k[value_index])
                : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint col = simdgroup_index * 16; col < qk_head_dim; col += num_simdgroups * 16) {
            auto vector_columns = vectors.slice<16, 16>(col, 0);
            auto w = vectors_op.get_destination_cooperative_tensor<decltype(weighted_tile), decltype(vector_columns), float>();
            vectors_op.run(weighted_tile, vector_columns, w);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (auto it = w.begin(); it != w.end(); ++it) {
                if (!w.is_valid_element(it)) continue;
                const auto coordinate = it.get_multidimensional_index();
                transformed_vectors[uint(coordinate[1]) * qk_head_dim + col + uint(coordinate[0])] = bfloat(*it);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto values = values_op.get_destination_cooperative_tensor<decltype(v_tile), decltype(weighted_tile), float>();
        values_op.run(v_tile, weighted_tile, values);
        // The V input and transposed U output share storage within this
        // SIMDgroup's disjoint V-row range. Finish all reads before stores.
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (auto it = values.begin(); it != values.end(); ++it) {
            if (!values.is_valid_element(it)) continue;
            const auto coordinate = it.get_multidimensional_index();
            shared_v[uint(coordinate[0]) * v_tile_storage_rows + simdgroup_index * 8 + uint(coordinate[1])] = bfloat(*it);
        }
        // The live state stays F32. Only its matrix operand is rounded.
        // Each SIMDgroup owns a disjoint eight-row region of the operand tile.
        #pragma unroll
        for (uint state_tile_index = 0; state_tile_index < num_state_tiles; ++state_tile_index) {
            thread auto& state = *states[state_tile_index];
            for (auto it = state.begin(); it != state.end(); ++it) {
                if (!state.is_valid_element(it)) continue;
                const auto coordinate = it.get_multidimensional_index();
                const uint row = simdgroup_index * matrix_size + uint(coordinate[1]);
                const uint col = state_tile_index * state_qk_tile_size + uint(coordinate[0]);
                state_operands[row * qk_head_dim + col] = bfloat(*it);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tensor<threadgroup bfloat, extents<int, qk_head_dim, 8>, tensor_inline> state_input(
            state_operands + simdgroup_index * matrix_size * qk_head_dim,
            extents<int, qk_head_dim, 8>{});
        auto state_w = state_vectors_op.get_destination_cooperative_tensor<decltype(state_input), decltype(vectors), float>();
        state_vectors_op.run(state_input, vectors, state_w);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (auto it = state_w.begin(); it != state_w.end(); ++it) {
            if (!state_w.is_valid_element(it)) continue;
            const auto coordinate = it.get_multidimensional_index();
            const uint token = uint(coordinate[0]);
            const uint row = simdgroup_index * 8 + uint(coordinate[1]);
            shared_v[token * v_tile_storage_rows + row] = bfloat(float(shared_v[token * v_tile_storage_rows + row]) - *it);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint value_index = thread_index;
             value_index < chunkwise_token_chunk_size * qk_head_dim;
             value_index += num_threads) {
            const uint row = value_index / qk_head_dim;
            const uint qk_dim_index = value_index - row * qk_head_dim;
            if (row < num_chunk_tokens) {
                const uint flat_token_index = chunk_start + row;
                const ulong q_value_index =
                    (ulong)flat_token_index * qkv_dim + q_base + qk_head_index * qk_head_dim + qk_dim_index;
                transformed_vectors[value_index] = bfloat(float(conv_qkv[q_value_index]) * q_inv_norm[row]);
            } else {
                transformed_vectors[value_index] = bfloat(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simdgroup_index == 0) {
            gram_op.run(vectors, k_tile, transform_tile);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint matrix_index = thread_index;
             matrix_index < chunkwise_token_chunk_size * chunkwise_token_chunk_size;
             matrix_index += num_threads) {
            const uint row = matrix_index / chunkwise_token_chunk_size;
            const uint col = matrix_index - row * chunkwise_token_chunk_size;
            weighted_transform[matrix_index] = bfloat(row < num_chunk_tokens && col <= row
                ? metal::exp(cumulative_log_decay[row] - cumulative_log_decay[col])
                    * transform[matrix_index]
                : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Use one destination layout for both terms. Keep the state projection
        // in a cooperative tensor until the local output is ready.
        auto state_q = output_op.get_destination_cooperative_tensor<decltype(v_tile), decltype(weighted_tile), float>();
        state_vectors_op.run(state_input, vectors, state_q);
        for (auto it = state_q.begin(); it != state_q.end(); ++it) {
            if (!state_q.is_valid_element(it)) continue;
            const auto coordinate = it.get_multidimensional_index();
            const uint token = uint(coordinate[0]);
            *it = token < num_chunk_tokens
                ? *it * metal::exp(cumulative_log_decay[token]) : 0.0f;
        }
        auto local_output = output_op.get_destination_cooperative_tensor<decltype(v_tile), decltype(weighted_tile), float>();
        output_op.run(v_tile, weighted_tile, local_output);
        auto state_q_it = state_q.begin();
        for (auto it = local_output.begin(); it != local_output.end(); ++it, ++state_q_it) {
            if (!local_output.is_valid_element(it)) continue;
            const auto coordinate = it.get_multidimensional_index();
            const uint token = uint(coordinate[0]);
            const uint row = simdgroup_index * 8 + uint(coordinate[1]);
            if (token < num_chunk_tokens && row < chunkwise_state_num_v_rows) {
                recurrent_output[((ulong)(chunk_start + token) * num_v_heads + v_head_index) * v_head_dim
                    + v_dim_base + row] = bfloat16_t(*it + *state_q_it);
            }
        }
        // QK readers must finish before K is changed for the state update.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float final_decay = metal::exp(cumulative_log_decay[num_chunk_tokens - 1]);
        for (uint index = thread_index; index < 16 * qk_head_dim; index += num_threads) {
            const uint token = index / qk_head_dim;
            normalized_k[index] = bfloat(float(normalized_k[index]) * (token < num_chunk_tokens
                ? metal::exp(cumulative_log_decay[num_chunk_tokens - 1] - cumulative_log_decay[token])
                : 0.0f));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint state_slot = write_candidate_states != 0 || chunk_start + num_chunk_tokens == flat_token_end
            ? flat_recurrent_state_write_slots[chunk_start + num_chunk_tokens - 1]
            : GDN_INVALID_STATE_SLOT_ID;
        #pragma unroll
        for (uint state_tile_index = 0; state_tile_index < num_state_tiles; ++state_tile_index) {
            thread auto& state = *states[state_tile_index];
            for (auto it = state.begin(); it != state.end(); ++it) {
                if (state.is_valid_element(it)) *it *= final_decay;
            }
            auto keys = k_tile.slice<state_qk_tile_size, 16>(state_tile_index * state_qk_tile_size, 0);
            update_op.run(v_tile, keys, state);
            if (state_slot != GDN_INVALID_STATE_SLOT_ID) {
                for (auto it = state.begin(); it != state.end(); ++it) {
                    if (!state.is_valid_element(it)) continue;
                    const auto coordinate = it.get_multidimensional_index();
                    const uint row = simdgroup_index * matrix_size + uint(coordinate[1]);
                    const uint col = state_tile_index * state_qk_tile_size + uint(coordinate[0]);
                    if (row < chunkwise_state_num_v_rows) {
                        const ulong address = recurrent_state_base + (ulong)state_slot * recurrent_state_stride
                            + ((ulong)v_head_index * v_head_dim + v_dim_base + row) * qk_head_dim + col;
                        recurrent_state_arena[address] = bfloat16_t(*it);
                    }
                }
            }
        }
        chunk_start += num_chunk_tokens;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
