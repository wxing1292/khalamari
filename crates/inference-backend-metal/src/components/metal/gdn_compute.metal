
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

constant uint GDN_INVALID_STATE_SLOT_ID = 0xffffffffu;

// Tensor axes: T = flat tokens, Hqk/Dqk = Q/K heads and head width,
// Hv/Dv = V heads and head width, and
// Cqkv = 2 * Hqk * Dqk + Hv * Dv = concatenated Q/K/V channel width.
// Short convolution operates independently along Cqkv; conv_kernel_size is
// its temporal kernel extent, not a tensor-channel dimension.

template <bool SaveReplay>
kernel void gdn_compute_short_conv(
    device bfloat16_t* conv_qkv [[buffer(0)]],
    device bfloat16_t* next_conv_state [[buffer(1)]],
    device const bfloat16_t* qkv [[buffer(2)]],
    device const bfloat16_t* conv_state [[buffer(3)]],
    device const bfloat16_t* conv_weight [[buffer(4)]],
    device const uint* src_conv_state_slots [[buffer(5)]],
    device const uint* flat_conv_state_write_slots [[buffer(6)]],
    device const uint* cu_tokens [[buffer(7)]],
    constant uint& num_active_reqs [[buffer(8)]],
    constant uint& num_active_tokens [[buffer(9)]],
    constant ulong& conv_state_offset_bytes [[buffer(10)]],
    constant ulong& next_conv_state_offset_bytes [[buffer(11)]],
    constant uint& write_final_conv_state [[buffer(12)]],
    device bfloat16_t* replay_qkv [[buffer(13)]],
    constant ulong& replay_token_offset [[buffer(14)]],
    constant uint& num_active_recurrent_requests [[buffer(15)]],
    uint global_linear_index [[thread_position_in_grid]]
) {
    if (global_linear_index >= num_active_tokens * qkv_dim) {
        return;
    }
    const ulong conv_state_base = conv_state_offset_bytes / sizeof(bfloat16_t);
    const ulong next_conv_state_base = next_conv_state_offset_bytes / sizeof(bfloat16_t);
    const uint channel_index = global_linear_index % qkv_dim;
    const uint flat_token_index = global_linear_index / qkv_dim;
    uint req_index = 0;
    for (uint candidate_req_index = 0; candidate_req_index < num_active_reqs; ++candidate_req_index) {
        if (flat_token_index < cu_tokens[candidate_req_index + 1]) {
            req_index = candidate_req_index;
            break;
        }
    }
    const uint flat_token_begin = cu_tokens[req_index];
    const uint token_index_in_req = flat_token_index - flat_token_begin;
    const uint src_state_slot = src_conv_state_slots[req_index];
    const uint state_slot = flat_conv_state_write_slots[flat_token_index];
    const bool write_state = state_slot != GDN_INVALID_STATE_SLOT_ID
        && (write_final_conv_state == 0 || flat_token_index + 1 == cu_tokens[req_index + 1]);

    float acc = 0.0f;
    for (uint kernel_index = 0; kernel_index < conv_kernel_size; ++kernel_index) {
        const long sequence_index =
            (long)token_index_in_req + (long)kernel_index - (long)conv_state_len;
        float x = 0.0f;
        if (sequence_index < 0) {
            const uint state_index = uint(sequence_index + (long)conv_state_len);
            const uint state_offset = (src_state_slot * qkv_dim + channel_index) * conv_state_len + state_index;
            x = conv_state[conv_state_base + (ulong)state_offset];
        } else {
            const ulong input_offset = (ulong)(flat_token_begin + uint(sequence_index)) * input_qkv_stride + channel_index;
            x = qkv[input_offset];
        }
        // Taps 1..Kc retain the raw history after this token.
        if (write_state && kernel_index > 0) {
            const ulong dst_offset = ((ulong)state_slot * qkv_dim + channel_index) * conv_state_len + kernel_index - 1;
            next_conv_state[next_conv_state_base + dst_offset] = bfloat16_t(x);
        }
        const uint weight_offset = channel_index * conv_kernel_size + kernel_index;
        acc += x * float(conv_weight[weight_offset]);
    }
    if constexpr (SaveReplay) {
        if (req_index >= num_active_recurrent_requests) {
            const uint replay_token_index = flat_token_index - cu_tokens[num_active_recurrent_requests];
            replay_qkv[(replay_token_offset + replay_token_index) * qkv_dim + channel_index] = qkv[(ulong)flat_token_index * input_qkv_stride + channel_index];
        }
    }
    conv_qkv[global_linear_index] = bfloat16_t(acc / (1.0f + metal::exp(-acc)));
}

template [[host_name("gdn_compute_short_conv_bf16")]] [[kernel]]
decltype(gdn_compute_short_conv<false>) gdn_compute_short_conv<false>;
template [[host_name("gdn_compute_short_conv_replay_bf16")]] [[kernel]]
decltype(gdn_compute_short_conv<true>) gdn_compute_short_conv<true>;

// One logical OutputNormGateThreadBlockTask maps 1:1 to one 128-thread
// threadblock. It RMS-normalizes and gates one [Dv] recurrent-output vector.
// The kernel derives the task from its arguments, thread-block index, and
// constants. It does not require a materialized task buffer:
//
// OutputNormGateThreadBlockTask {
//   flat_token_index,  // grid-derived from threadblock linear index / Hv
//   v_head_index,      // grid-derived from threadblock linear index % Hv
// }
kernel void gdn_compute_output_norm_gate_bf16(
    device bfloat16_t* norm_gated_output [[buffer(0)]],
    device const bfloat16_t* recurrent_output [[buffer(1)]],
    device const bfloat16_t* z [[buffer(2)]],
    device const bfloat16_t* norm_weight [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& num_active_tokens [[buffer(5)]],
    uint global_thread_index [[thread_position_in_grid]]
) {
    const uint reduction_thread_index =
        global_thread_index % output_norm_gate_required_threads;
    const uint token_head_index =
        global_thread_index / output_norm_gate_required_threads;
    const uint num_token_heads = num_active_tokens * num_v_heads;
    if (token_head_index >= num_token_heads) {
        return;
    }
    const uint flat_token_index = token_head_index / num_v_heads;
    const uint v_head_index = token_head_index % num_v_heads;
    const uint token_head_base = flat_token_index * num_v_heads * v_head_dim + v_head_index * v_head_dim;
    threadgroup float square_sum_partials[output_norm_gate_required_threads];

    float square_sum_partial = 0.0f;
    for (uint v_dim_index = reduction_thread_index;
         v_dim_index < v_head_dim;
         v_dim_index += output_norm_gate_required_threads) {
        const float x = recurrent_output[token_head_base + v_dim_index];
        square_sum_partial += x * x;
    }
    square_sum_partials[reduction_thread_index] = square_sum_partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = output_norm_gate_required_threads / 2; stride > 0; stride >>= 1) {
        if (reduction_thread_index < stride) {
            square_sum_partials[reduction_thread_index] += square_sum_partials[reduction_thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = metal::rsqrt(square_sum_partials[0] / float(v_head_dim) + eps);
    for (uint v_dim_index = reduction_thread_index;
         v_dim_index < v_head_dim;
         v_dim_index += output_norm_gate_required_threads) {
        const uint output_index = token_head_base + v_dim_index;
        const float z_value = z[input_z_offset + (ulong)flat_token_index * input_z_stride + v_head_index * v_head_dim + v_dim_index];
        const float denominator = 1.0f + metal::exp(-z_value);
        const float gate = output_gate_is_sigmoid ? 1.0f / denominator : z_value / denominator;
        const float normalized_value = recurrent_output[output_index] * inv_rms * float(norm_weight[v_dim_index]);
        norm_gated_output[output_index] = bfloat16_t(normalized_value * gate);
    }
}
