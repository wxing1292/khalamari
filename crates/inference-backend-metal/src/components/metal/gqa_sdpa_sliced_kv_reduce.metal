#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

// SplitKV SlicedKv reduce. For one flat-token/Q-head output coordinate, adjacent
// cu_sdpa_partial_outputs values select the leading partial-output dimension to
// merge. The cumulative values do not count scalar tensor elements.

template <bool Mapped, typename PartialT, typename OutputT>
void gqa_sdpa_sliced_kv_reduce_impl(
    device const float* partial_exp_sums,
    device const float* partial_max_logits,
    device const PartialT* partial_output,
    device const uint* cu_sdpa_partial_outputs,
    device const uint* output_query_indices,
    device OutputT* output,
    constant uint& num_active_tokens,
    uint gid
) {
    const uint total = num_active_tokens * num_q_heads * head_dim;
    if (gid >= total) return;
    const uint dim_index = gid % head_dim;
    const uint q_head_index = (gid / head_dim) % num_q_heads;
    const uint flat_token_index = gid / (head_dim * num_q_heads);
    uint output_query_index = flat_token_index;
    if constexpr (Mapped) output_query_index = output_query_indices[flat_token_index];
    const uint partial_output_begin = cu_sdpa_partial_outputs[flat_token_index];
    const uint partial_output_end = cu_sdpa_partial_outputs[flat_token_index + 1];

    float global_max = -INFINITY;
    for (uint partial_output_index = partial_output_begin;
         partial_output_index < partial_output_end;
         ++partial_output_index) {
        const uint partial_output_stats_index = partial_output_index * num_q_heads + q_head_index;
        global_max = metal::max(global_max, partial_max_logits[partial_output_stats_index]);
    }

    float global_exp_sum = 0.0f;
    float output_accumulator = 0.0f;
    for (uint partial_output_index = partial_output_begin;
         partial_output_index < partial_output_end;
         ++partial_output_index) {
        const uint partial_output_stats_index = partial_output_index * num_q_heads + q_head_index;
        const float partial_exp_sum = partial_exp_sums[partial_output_stats_index];
        const float weight = isfinite(partial_max_logits[partial_output_stats_index])
            ? metal::exp(partial_max_logits[partial_output_stats_index] - global_max) * partial_exp_sum
            : 0.0f;
        global_exp_sum += weight;
        output_accumulator += weight * float(partial_output[partial_output_stats_index * head_dim + dim_index]);
    }
    ulong output_index = gid;
    if constexpr (Mapped) {
        output_index = (ulong(output_query_index) * num_q_heads + q_head_index) * head_dim + dim_index;
    }
    output[output_index] = OutputT(global_exp_sum > 0.0f ? output_accumulator / global_exp_sum : 0.0f);
}

kernel void gqa_sdpa_sliced_kv_reduce_f32(
    device const float* partial_exp_sums [[buffer(0)]],
    device const float* partial_max_logits [[buffer(1)]],
    device const float* partial_output [[buffer(2)]],
    device const uint* cu_sdpa_partial_outputs [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& num_active_tokens [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    gqa_sdpa_sliced_kv_reduce_impl<false, float, float>(
        partial_exp_sums, partial_max_logits, partial_output, cu_sdpa_partial_outputs,
        cu_sdpa_partial_outputs, output,
        num_active_tokens, gid);
}

kernel void gqa_sdpa_sliced_kv_reduce_bf16(
    device const float* partial_exp_sums [[buffer(0)]],
    device const float* partial_max_logits [[buffer(1)]],
    device const bfloat16_t* partial_output [[buffer(2)]],
    device const uint* cu_sdpa_partial_outputs [[buffer(3)]],
    device bfloat16_t* output [[buffer(4)]],
    constant uint& num_active_tokens [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    gqa_sdpa_sliced_kv_reduce_impl<false, bfloat16_t, bfloat16_t>(
        partial_exp_sums, partial_max_logits, partial_output, cu_sdpa_partial_outputs,
        cu_sdpa_partial_outputs, output,
        num_active_tokens, gid);
}

kernel void gqa_sdpa_sliced_kv_reduce_mapped_bf16_bf16(
    device const float* partial_exp_sums [[buffer(0)]],
    device const float* partial_max_logits [[buffer(1)]],
    device const bfloat16_t* partial_output [[buffer(2)]],
    device const uint* cu_sdpa_partial_outputs [[buffer(3)]],
    device const uint* output_query_indices [[buffer(4)]],
    device bfloat16_t* output [[buffer(5)]],
    constant uint& num_active_queries [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    gqa_sdpa_sliced_kv_reduce_impl<true, bfloat16_t, bfloat16_t>(
        partial_exp_sums, partial_max_logits, partial_output, cu_sdpa_partial_outputs,
        output_query_indices, output, num_active_queries, gid);
}

kernel void gqa_sdpa_sliced_kv_reduce_mapped_f32_bf16(
    device const float* partial_exp_sums [[buffer(0)]],
    device const float* partial_max_logits [[buffer(1)]],
    device const float* partial_output [[buffer(2)]],
    device const uint* cu_sdpa_partial_outputs [[buffer(3)]],
    device const uint* output_query_indices [[buffer(4)]],
    device bfloat16_t* output [[buffer(5)]],
    constant uint& num_active_queries [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    gqa_sdpa_sliced_kv_reduce_impl<true, float, bfloat16_t>(
        partial_exp_sums, partial_max_logits, partial_output, cu_sdpa_partial_outputs,
        output_query_indices, output, num_active_queries, gid);
}
