// Generic indexed row gather.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

template <typename T>
void row_gather_impl(
    device const T* input,
    device const uint* row_indices,
    device T* output,
    constant uint& num_cols,
    constant uint& num_active_rows,
    uint gid
) {
    uint num_values = num_active_rows * num_cols;
    if (gid >= num_values) return;
    uint output_row = gid / num_cols;
    uint col = gid - output_row * num_cols;
    uint input_row = row_indices[output_row];
    output[gid] = input[ulong(input_row) * num_cols + col];
}

kernel void row_gather_bf16(
    device const bfloat16_t* input [[buffer(0)]],
    device const uint* row_indices [[buffer(1)]],
    device bfloat16_t* output [[buffer(2)]],
    constant uint& num_cols [[buffer(3)]],
    constant uint& num_active_rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    row_gather_impl<bfloat16_t>(input, row_indices, output, num_cols, num_active_rows, gid);
}

kernel void row_gather_f32(
    device const float* input [[buffer(0)]],
    device const uint* row_indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& num_cols [[buffer(3)]],
    constant uint& num_active_rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    row_gather_impl<float>(input, row_indices, output, num_cols, num_active_rows, gid);
}
