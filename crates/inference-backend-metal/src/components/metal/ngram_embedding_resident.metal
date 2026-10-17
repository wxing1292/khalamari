// IDs [T,16] U32 -> E [T,16,160] BF16, contiguous token/head/dim.
// Eight or fewer contiguous resident groups: W [N,20] U32, S/B [N,5] BF16.
//
// H1: grid [T,16,1], H4: grid [T,4,1]. Each threadblock has 160 threads (5 SIMDgroups).
// Threadblock (t,j) owns token t and heads [j*H,(j+1)*H). Thread d owns dimension d.
//
// for h in [j*H,(j+1)*H):
//     register id <- GMEM IDs[t,h]
//     select group g from constant row starts
//     register code/scale/bias <- GMEM W_g/S_g/B_g[row,d]
//     register decoded = BF16(BF16(scale*code)+bias)
//     GMEM E[t,h,d] <- BF16(decoded*table_scale)
//
// No shared memory or reduction. Inactive token threadblocks return before reading IDs.
kernel void ngram_lookup_resident(
    device const uint* w0 [[buffer(0)]],
    device const uint* w1 [[buffer(1)]],
    device const uint* w2 [[buffer(2)]],
    device const uint* w3 [[buffer(3)]],
    device const uint* w4 [[buffer(4)]],
    device const uint* w5 [[buffer(5)]],
    device const uint* w6 [[buffer(6)]],
    device const uint* w7 [[buffer(7)]],
    device const bfloat* s0 [[buffer(8)]],
    device const bfloat* s1 [[buffer(9)]],
    device const bfloat* s2 [[buffer(10)]],
    device const bfloat* s3 [[buffer(11)]],
    device const bfloat* s4 [[buffer(12)]],
    device const bfloat* s5 [[buffer(13)]],
    device const bfloat* s6 [[buffer(14)]],
    device const bfloat* s7 [[buffer(15)]],
    device const bfloat* b0 [[buffer(16)]],
    device const bfloat* b1 [[buffer(17)]],
    device const bfloat* b2 [[buffer(18)]],
    device const bfloat* b3 [[buffer(19)]],
    device const bfloat* b4 [[buffer(20)]],
    device const bfloat* b5 [[buffer(21)]],
    device const bfloat* b6 [[buffer(22)]],
    device const bfloat* b7 [[buffer(23)]],
    device const uint* ids [[buffer(24)]],
    constant uint* starts [[buffer(25)]],
    device bfloat* output [[buffer(26)]],
    constant float& table_scale [[buffer(27)]],
    constant uint& num_active_tokens [[buffer(28)]],
    uint2 block [[threadgroup_position_in_grid]],
    uint d [[thread_index_in_threadgroup]]
) {
    const uint token = block.x;
    if (token >= num_active_tokens) return;
    #pragma unroll
    for (uint h = 0; h < heads_per_threadblock; ++h) {
        const uint head = block.y * heads_per_threadblock + h;
        const uint id = ids[ulong(token) * 16u + head];
        uint group = id >= starts[4] ? 4u : 0u;
        group += id >= starts[group + 2u] ? 2u : 0u;
        group += id >= starts[group + 1u] ? 1u : 0u;
        device const uint* weight = w0;
        device const bfloat* scales = s0;
        device const bfloat* biases = b0;
        switch (group) {
            case 1: weight = w1; scales = s1; biases = b1; break;
            case 2: weight = w2; scales = s2; biases = b2; break;
            case 3: weight = w3; scales = s3; biases = b3; break;
            case 4: weight = w4; scales = s4; biases = b4; break;
            case 5: weight = w5; scales = s5; biases = b5; break;
            case 6: weight = w6; scales = s6; biases = b6; break;
            case 7: weight = w7; scales = s7; biases = b7; break;
        }
        const ulong row = ulong(id - starts[group]);
        const uint packed = weight[row * (160u / 8u) + d / 8u];
        const uint code = (packed >> ((d % 8u) * 4u)) & 15u;
        const ulong affine = row * (160u / 32u) + d / 32u;
        const bfloat product = bfloat(float(scales[affine]) * float(code));
        const bfloat value = bfloat(float(product) + float(biases[affine]));
        output[(ulong(token) * 16u + head) * 160u + d] = bfloat(float(value) * table_scale);
    }
}
