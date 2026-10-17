#include <metal_stdlib>

using namespace metal;

// Grid (feature tile, input token). Each channel copies the same BF16 input row.
kernel void repeat_rows(
    device const ushort* input [[buffer(0)]],
    device ushort* output [[buffer(1)]],
    constant uint& num_columns [[buffer(2)]],
    constant uint& num_repeats [[buffer(3)]],
    constant uint& num_active_tokens [[buffer(4)]],
    uint2 block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint column = block.x * 256 + tid;
    if (block.y >= num_active_tokens || column >= num_columns * num_repeats) return;
    const ulong dst = ulong(block.y) * num_columns * num_repeats + column;
    const ulong src = ulong(block.y) * num_columns + column % num_columns;
    output[dst] = input[src];
}
