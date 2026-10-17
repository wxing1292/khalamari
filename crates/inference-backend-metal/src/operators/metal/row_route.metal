// Route each output row from one of three dense source buffers.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

template <typename T>
void row_route_impl(
    device const T* first_input,
    device const T* second_input,
    device const T* third_input,
    device const uint2* routes,
    device T* output,
    constant uint& num_cols,
    constant uint& num_active_rows,
    uint gid
) {
    uint num_values = num_active_rows * num_cols;
    if (gid >= num_values) return;
    uint output_row = gid / num_cols;
    uint col = gid - output_row * num_cols;
    uint2 route = routes[output_row];
    device const T* input = route.x == 0 ? first_input : (route.x == 1 ? second_input : third_input);
    output[gid] = input[ulong(route.y) * num_cols + col];
}

kernel void row_route_bf16(
    device const bfloat16_t* first_input [[buffer(0)]],
    device const bfloat16_t* second_input [[buffer(1)]],
    device const bfloat16_t* third_input [[buffer(2)]],
    device const uint2* routes [[buffer(3)]],
    device bfloat16_t* output [[buffer(4)]],
    constant uint& num_cols [[buffer(5)]],
    constant uint& num_active_rows [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    row_route_impl<bfloat16_t>(first_input, second_input, third_input, routes, output, num_cols, num_active_rows, gid);
}

kernel void row_route_f32(
    device const float* first_input [[buffer(0)]],
    device const float* second_input [[buffer(1)]],
    device const float* third_input [[buffer(2)]],
    device const uint2* routes [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& num_cols [[buffer(5)]],
    constant uint& num_active_rows [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    row_route_impl<float>(first_input, second_input, third_input, routes, output, num_cols, num_active_rows, gid);
}
