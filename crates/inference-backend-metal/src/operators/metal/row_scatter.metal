// Copy indexed input rows to indexed output rows.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

template <typename T>
void row_scatter_impl(
    device const T* input,
    device const uint2* routes,
    device T* output,
    constant uint& num_cols,
    constant uint& num_active_rows,
    uint gid
) {
    uint num_values = num_active_rows * num_cols;
    if (gid >= num_values) return;
    uint route_index = gid / num_cols;
    uint col = gid - route_index * num_cols;
    uint2 route = routes[route_index];
    output[ulong(route.y) * num_cols + col] = input[ulong(route.x) * num_cols + col];
}

kernel void row_scatter_bf16(
    device const bfloat16_t* input [[buffer(0)]],
    device const uint2* routes [[buffer(1)]],
    device bfloat16_t* output [[buffer(2)]],
    constant uint& num_cols [[buffer(3)]],
    constant uint& num_active_rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    row_scatter_impl<bfloat16_t>(input, routes, output, num_cols, num_active_rows, gid);
}

kernel void row_scatter_f32(
    device const float* input [[buffer(0)]],
    device const uint2* routes [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& num_cols [[buffer(3)]],
    constant uint& num_active_rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    row_scatter_impl<float>(input, routes, output, num_cols, num_active_rows, gid);
}
