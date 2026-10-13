// SPDX-License-Identifier: Apache-2.0

// Sliced-K: NSG SIMDgroups partition K inside one threadgroup.
// Each SIMDgroup covers the same (8 * RT) by (8 * CT) logical output tile.
// Each 8x8 matrix distributes two elements per lane across 32 lanes.
template <typename T, int K, int N, int GS, int NSG, int CT, bool SWIGLU, int RT>
[[kernel]] void affine_qmm_sliced_k(
    const device uint* w_q [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]], device T* y [[buffer(4)]],
    const constant uint& num_active_rows [[buffer(7)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
    const uint row_base = threadgroup_position_in_grid.x * 8 * RT;
    if (row_base >= num_active_rows) return;
    const device uint* w_q2 = w_q + ulong(N) * K / 8;
    const device T* scales2 = scales + ulong(N) * (K / GS);
    const device T* biases2 = biases + ulong(N) * (K / GS);
    const uint sg = thread_position_in_threadgroup.x / 32;
    const uint lane = thread_position_in_threadgroup.x % 32;
    const uint qid = lane >> 2;
    const uint fm = (qid & 4) | ((lane >> 1) & 3);
    const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
    constexpr int GROUPS = K / GS, PER = GROUPS / NSG, WB = GS / 2;
    constexpr ulong ROW_BYTES = ulong(K) / 2;
    const int g_begin = int(sg) * PER;
    const int nbase = int(threadgroup_position_in_grid.y) * 8 * CT;
    const uint kx = (fm >> 1) * 8 + (fm & 1u) * 4;
    const device T* x0[RT];
    const device T* x1[RT];
    // Persistent K partials. The group-dot accumulators below reset for each group.
    float2 out0[RT][CT], out1[RT][CT];
    for (int rt = 0; rt < RT; ++rt) {
        const uint row = row_base + rt * 8 + fn;
        x0[rt] = x + ulong(min(row, num_active_rows - 1u)) * K + kx;
        x1[rt] = x + ulong(min(row + 1u, num_active_rows - 1u)) * K + kx;
        for (int c = 0; c < CT; ++c) out0[rt][c] = out1[rt][c] = float2(0.0f);
    }
    for (int g = g_begin; g < g_begin + PER; ++g) {
        simdgroup_matrix<float, 8, 8> acc0[RT][CT], acc1[RT][CT];
        float2 xsum[RT];
        for (int rt = 0; rt < RT; ++rt) {
            xsum[rt] = float2(0.0f);
            for (int c = 0; c < CT; ++c) {
                acc0[rt][c] = simdgroup_matrix<float, 8, 8>(0.0f);
                acc1[rt][c] = simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
        for (int qd = 0; qd < GS / 32; ++qd) {
            const int kq = g * GS + qd * 32;
            T xa[RT][4], xb[RT][4];
            for (int rt = 0; rt < RT; ++rt) {
                const vec<T, 4> va = *((const device vec<T, 4>*)(x0[rt] + kq));
                const vec<T, 4> vb = *((const device vec<T, 4>*)(x1[rt] + kq));
                for (int j = 0; j < 4; ++j) {
                    xa[rt][j] = va[j];
                    xb[rt][j] = vb[j];
                    xsum[rt] += float2(float(xa[rt][j]), float(xb[rt][j]));
                }
            }
            for (int c = 0; c < CT; ++c) {
                const int n = nbase + c * 8 + int(fm);
                const device uchar* wg = (const device uchar*)w_q + ulong(n) * ROW_BYTES + ulong(g) * WB;
                const uint word = ((const device uint*)(wg + qd * 16))[fn >> 1];
                for (int j = 0; j < 4; ++j) {
                    const uint codes = (word >> (4 * j)) & 0x000F000Fu;
                    // Exact 128 + q operands: BF16 fraction units are 1; FP16 units are 1/8.
                    const uint pair = is_same_v<T, half> ? ((codes << 3) | 0x58005800u) : (codes | 0x43004300u);
                    simdgroup_matrix<T, 8, 8> a;
                    a.thread_elements()[0] = as_type<T>(ushort(pair));
                    a.thread_elements()[1] = as_type<T>(ushort(pair >> 16));
                    for (int rt = 0; rt < RT; ++rt) {
                        simdgroup_matrix<T, 8, 8> b;
                        b.thread_elements()[0] = xa[rt][j];
                        b.thread_elements()[1] = xb[rt][j];
                        simdgroup_multiply_accumulate(acc0[rt][c], a, b, acc0[rt][c]);
                    }
                }
                if (SWIGLU) {
                    wg = (const device uchar*)w_q2 + ulong(n) * ROW_BYTES + ulong(g) * WB;
                    const uint word_up = ((const device uint*)(wg + qd * 16))[fn >> 1];
                    for (int j = 0; j < 4; ++j) {
                        const uint codes = (word_up >> (4 * j)) & 0x000F000Fu;
                        const uint pair = is_same_v<T, half> ? ((codes << 3) | 0x58005800u) : (codes | 0x43004300u);
                        simdgroup_matrix<T, 8, 8> a;
                        a.thread_elements()[0] = as_type<T>(ushort(pair));
                        a.thread_elements()[1] = as_type<T>(ushort(pair >> 16));
                        for (int rt = 0; rt < RT; ++rt) {
                            simdgroup_matrix<T, 8, 8> b;
                            b.thread_elements()[0] = xa[rt][j];
                            b.thread_elements()[1] = xb[rt][j];
                            simdgroup_multiply_accumulate(acc1[rt][c], a, b, acc1[rt][c]);
                        }
                    }
                }
            }
        }
        for (int rt = 0; rt < RT; ++rt) {
            xsum[rt] += float2(simd_shuffle_xor(xsum[rt].x, 2), simd_shuffle_xor(xsum[rt].y, 2));
            xsum[rt] += float2(simd_shuffle_xor(xsum[rt].x, 4), simd_shuffle_xor(xsum[rt].y, 4));
            xsum[rt] += float2(simd_shuffle_xor(xsum[rt].x, 16), simd_shuffle_xor(xsum[rt].y, 16));
            for (int c = 0; c < CT; ++c) {
                const ulong n = nbase + c * 8 + int(fm);
                const float s = float(scales[n * GROUPS + g]), bi = float(biases[n * GROUPS + g]);
                // Use the original quantization bias. Compute its offset correction in F32.
                out0[rt][c] += s * float2(acc0[rt][c].thread_elements()[0], acc0[rt][c].thread_elements()[1])
                    + (bi - 128.0f * s) * xsum[rt];
                if (SWIGLU) {
                    const float s2 = float(scales2[n * GROUPS + g]), bi2 = float(biases2[n * GROUPS + g]);
                    out1[rt][c] += s2 * float2(acc1[rt][c].thread_elements()[0], acc1[rt][c].thread_elements()[1])
                        + (bi2 - 128.0f * s2) * xsum[rt];
                }
            }
        }
    }
    // SMEM layout: [projection][RT][CT][NSG][lane], one float2 per lane.
    threadgroup float2 part[(SWIGLU ? 2 : 1) * RT * NSG * CT * 32];
    for (int rt = 0; rt < RT; ++rt) {
        for (int c = 0; c < CT; ++c) {
            part[((rt * CT + c) * NSG + sg) * 32 + lane] = out0[rt][c];
            if (SWIGLU) part[(((RT + rt) * CT + c) * NSG + sg) * 32 + lane] = out1[rt][c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (int rt = 0; rt < RT; ++rt) {
            for (int c = 0; c < CT; ++c) {
                float2 v = float2(0.0f), u = float2(0.0f);
                for (int i = 0; i < NSG; ++i) {
                    v += part[((rt * CT + c) * NSG + i) * 32 + lane];
                    if (SWIGLU) u += part[(((RT + rt) * CT + c) * NSG + i) * 32 + lane];
                }
                if (SWIGLU) {
                    const float2 gv = float2(T(v.x), T(v.y)), uv = float2(T(u.x), T(u.y));
                    v = gv / (1.0f + exp(-gv)) * uv;
                }
                const uint row = row_base + rt * 8 + fn;
                const int n = nbase + c * 8 + int(fm);
                if (row < num_active_rows) y[ulong(row) * N + n] = T(v.x);
                if (row + 1u < num_active_rows) y[ulong(row + 1u) * N + n] = T(v.y);
            }
        }
    }
}

// SLICED_K partitions K across SIMDgroups. The reduction stays inside one threadgroup.
template <typename T, int GS, int BITS, int BM, int NSG, bool SLICED_K, bool SWIGLU>
[[kernel]] void affine_qmm_small(
    const device uint* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]], device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]], const constant int& N [[buffer(6)]],
    const constant uint& num_active_rows [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int NA = 4 * BM;
    uint part = SWIGLU ? sg % NSG : sg;
    uint matrix = SWIGLU ? sg / NSG : 0;
    int n0 = (SLICED_K ? int(tid.y) : int(tid.y) * NSG + int(part)) * 4;
    int packs = K / 8, groups = K / GS;
    int per_part = packs / NSG;
    int p_start = SLICED_K ? int(part) * per_part : 0;
    int p_end = SLICED_K && int(part) != NSG - 1 ? p_start + per_part : packs;
    w += ulong(matrix) * N * K * BITS / 32;
    scales += ulong(matrix) * N * groups;
    biases += ulong(matrix) * N * groups;
    float acc[NA];
    #pragma unroll
    for (int i = 0; i < NA; ++i) acc[i] = 0.0f;
    using Vec8 = vec<T, 8>;
    const device Vec8* xv = (const device Vec8*)x;
    const device Vec8* x0 = xv;
    const device Vec8* x1 = xv + ulong(min(1u, num_active_rows - 1u)) * packs;
    const device Vec8* x2 = xv + ulong(min(2u, num_active_rows - 1u)) * packs;
    const device Vec8* x3 = xv + ulong(min(3u, num_active_rows - 1u)) * packs;
    /* SMALL_QMM_COLUMN_POINTERS */
    for (int pack = p_start + int(lane); pack < p_end; pack += 32) {
        int gi = pack * 8 / GS;
        /* SMALL_QMM_PACKED_DOT */
    }
    #pragma unroll
    for (int i = 0; i < NA; ++i) acc[i] = simd_sum(acc[i]);
    if (SLICED_K || SWIGLU) {
        threadgroup float partials[(SWIGLU ? 2 : 1) * NSG * NA];
        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < NA; ++i) partials[sg * NA + i] = acc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((SLICED_K ? sg == 0 : sg < NSG) && lane < NA) {
            float g = 0.0f, u = 0.0f;
            if (SLICED_K) {
                #pragma unroll
                for (int p = 0; p < NSG; ++p) {
                    g += partials[p * NA + lane];
                    if (SWIGLU) u += partials[(NSG + p) * NA + lane];
                }
            } else {
                g = partials[part * NA + lane];
                u = partials[(NSG + part) * NA + lane];
            }
            uint row = lane % BM, col = lane / BM;
            if (SWIGLU) {
                g = float(T(g)); u = float(T(u));
                g = g / (1.0f + exp(-g)) * u;
            }
            if (row < num_active_rows) y[ulong(row) * N + n0 + col] = T(g);
        }
    } else if (lane < NA) {
        uint row = lane % BM, col = lane / BM;
        if (row < num_active_rows) y[ulong(row) * N + n0 + col] = T(acc[lane]);
    }
}
