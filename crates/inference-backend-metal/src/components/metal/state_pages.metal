#include <metal_stdlib>

using namespace metal;

struct CopyJob {
    ulong src_offset_bytes;
    ulong dst_offset_bytes;
    ulong len_bytes;
};

// Grid (job). Each threadgroup owns one disjoint destination range.
kernel void copy_state_ranges(
    device const uint* src [[buffer(0)]],
    device uint* dst [[buffer(1)]],
    device const CopyJob* jobs [[buffer(2)]],
    uint job_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const CopyJob job = jobs[job_index];
    for (ulong i = tid; i < job.len_bytes / 4; i += 256) {
        dst[job.dst_offset_bytes / 4 + i] = src[job.src_offset_bytes / 4 + i];
    }
}
