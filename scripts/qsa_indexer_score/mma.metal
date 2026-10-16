// Copyright © 2026 Apple Inc.
// Benchmark only. Adapted from oMLX's Apache-2.0 DSA MMA fragment mapping.
// See licenses/omlx-Apache-2.0.txt.
// Q [tokens,4,128] and pooled K [groups,128] are BF16. C/S are FP32.
// One TG: QT tokens x BN groups. Four SGs partition the output, WM=WN=2.
// FLAT flattens token/head before MMA. Otherwise the outer loop visits heads.
// QRES holds all Q fragments in registers. Otherwise each step loads Q directly.
// K [128,BN] stays in transposed SMEM across all heads and dimension steps.
constexpr int D = 128, MT = (FLAT ? QT * 4 : QT) / 16, NT = BN / 16;
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const uint tid = thread_position_in_threadgroup.x;
const uint2 grid = threadgroup_position_in_grid.xy;
if (grid.y >= PARAMS[0]) return;
const uint2 task = uint2(TASKS[2 * grid.y], TASKS[2 * grid.y + 1]);
const uint position = TOKENS[6 * task.x + 1], req = TOKENS[6 * task.x + 2];
const auto* Q = QHEADMAJOR ? QHEAD : Q_ALL;
const uint group0 = grid.x * BN, end = (position + task.y) / 4;
if (group0 >= end) return;
const uint valid = min(uint(BN), end - group0);
const uint physical = PAGE_IDS[req * PARAMS[1] + group0 / 128];
const device bfloat* keys = KPAGED
    ? PAGES + (ulong(physical) * 128 + group0 % 128) * D
    : CONTIG + (ulong(req) * PARAMS[3] + group0) * D;
threadgroup bfloat kt[D * BN];
// Match the blocked transpose load in the oMLX resident-K reference.
// A dimension-striped load would send a warp's SMEM stores to the same bank.
for (uint j = 0; j < BN * D / 128; ++j) {
    const uint i = tid * (BN * D / 128) + j;
    const uint g = i / D, d = i % D;
    kt[d * BN + g] = g < valid ? keys[i] : bfloat(0.0f);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const short quad = short(lane) / 4;
const short fm = (quad & 4) + ((short(lane) / 2) % 4);
const short fn = (quad & 2) * 2 + (short(lane) % 2) * 2;
const int sm = 8 * int(sg / 2) + fm, sn = 8 * int(sg % 2) + fn;
simdgroup_matrix<bfloat,8,8> qr[QRES ? (FLAT ? 1 : 4) * MT * 16 : 1];
if constexpr (QRES) {
    for (int h = 0; h < (FLAT ? 1 : 4); ++h)
        for (int m = 0; m < MT; ++m)
            for (int step = 0; step < 16; ++step) {
                const uint mr = uint(sm + m * 16);
                const uint tr = FLAT ? mr / 4 : mr, head = FLAT ? mr % 4 : uint(h);
                const ulong off = (QHEADMAJOR
                    ? (ulong(req) * 4 + head) * PARAMS[6] + task.x % PARAMS[6] + tr
                    : (ulong(task.x) + tr) * 4 + head) * D + step * 8 + fn;
                const vec<bfloat,2> v = tr < task.y ? vec<bfloat,2>(Q[off],Q[off+1]) : vec<bfloat,2>(0);
                reinterpret_cast<thread vec<bfloat,2>&>(qr[(h * MT + m) * 16 + step].thread_elements()) = v;
            }
}
float sums[MT * NT * 2];
for (int i = 0; i < MT * NT * 2; ++i) sums[i] = 0.0f;
for (int h = 0; h < (FLAT ? 1 : 4); ++h) {
    simdgroup_matrix<float,8,8> c[MT * NT];
    for (int step = 0; step < 16; ++step) {
        simdgroup_matrix<bfloat,8,8> a[MT], b[NT];
        for (int m = 0; m < MT; ++m) {
            if constexpr (QRES) a[m] = qr[(h * MT + m) * 16 + step];
            else {
                const uint mr = uint(sm + m * 16);
                const uint tr = FLAT ? mr / 4 : mr, head = FLAT ? mr % 4 : uint(h);
                const ulong off = (QHEADMAJOR
                    ? (ulong(req) * 4 + head) * PARAMS[6] + task.x % PARAMS[6] + tr
                    : (ulong(task.x) + tr) * 4 + head) * D + step * 8 + fn;
                const vec<bfloat,2> v = tr < task.y ? vec<bfloat,2>(Q[off],Q[off+1]) : vec<bfloat,2>(0);
                reinterpret_cast<thread vec<bfloat,2>&>(a[m].thread_elements()) = v;
            }
        }
        for (int n = 0; n < NT; ++n) {
            const threadgroup bfloat* p = kt + (step * 8 + fm) * BN + sn + n * 16;
            reinterpret_cast<thread vec<bfloat,2>&>(b[n].thread_elements()) = vec<bfloat,2>(p[0],p[1]);
        }
        for (int m = 0; m < MT; ++m)
            for (int n = 0; n < NT; ++n) {
                if (step == 0) simdgroup_multiply(c[m*NT+n],a[m],b[n]);
                else simdgroup_multiply_accumulate(c[m*NT+n],a[m],b[n],c[m*NT+n]);
            }
    }
    for (int i = 0; i < MT * NT; ++i) {
        const vec<float,2> v = reinterpret_cast<thread vec<float,2>&>(c[i].thread_elements());
        sums[2*i] += max(v[0],0.0f); sums[2*i+1] += max(v[1],0.0f);
    }
}
for (int m = 0; m < MT; ++m)
    for (int n = 0; n < NT; ++n)
        for (int e = 0; e < 2; ++e) {
            const uint mr = uint(sm + m * 16), tr = FLAT ? mr / 4 : mr;
            const uint g = group0 + uint(sn + n * 16 + e);
            float score = sums[2*(m*NT+n)+e];
            if constexpr (FLAT) {
                const ushort base = ushort(lane & ~6u);
                score = simd_shuffle(score,base) + simd_shuffle(score,ushort(base|2))
                      + simd_shuffle(score,ushort(base|4)) + simd_shuffle(score,ushort(base|6));
            }
            if ((!FLAT || mr % 4 == 0) && tr < task.y && g < (position+tr+1)/4)
                O[(ulong(task.x)+tr)*PARAMS[3]+g] = score * rsqrt(128.0f);
        }
