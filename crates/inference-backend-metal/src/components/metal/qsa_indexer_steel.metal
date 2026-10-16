// Copyright © 2026 Apple Inc.
// Adapted from the Steel QSA score kernels in oMLX commit
// d6b2b92b11ebcbf4a4c1f002b0d590a3117212a8 (Apache-2.0).
// This source keeps request-local task bounds and relative score columns.
using namespace mlx::steel;
typedef bfloat bfloat16_t;

template <typename T, int BM, int BN, int BK, int WM, int WN, bool FLAT>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
qsa_indexer_steel_score(
    const device T* Q [[buffer(0)]],
    const device Token* tokens [[buffer(1)]],
    const device ScoreTaskTemplate* score_tasks [[buffer(2)]],
    const device T* Pages [[buffer(3)]],
    const device uint* page_ids [[buffer(4)]],
    device float* O [[buffer(5)]],
    const constant uint& num_active_score_task_templates [[buffer(6)]],
    const constant uint& pages_per_request [[buffer(7)]],
    const constant uint& groups_per_page [[buffer(8)]],
    const constant uint& num_total_groups [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 grid [[threadgroup_position_in_grid]]) {
  using gemm_kernel = GEMMKernel<T, T, BM, BN, BK, WM, WN, false, true, true, true, float>;
  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  if (grid.y >= num_active_score_task_templates) return;
  const ScoreTaskTemplate score_task = score_tasks[grid.y];
  if constexpr (FLAT) {
    if (score_task.num_queries > 8u) return;
  } else {
    if (score_task.num_queries <= 8u) return;
  }

  const Token token = tokens[score_task.query_start];
  const uint aligned_group_start = token.group_start - token.group_start % BN;
  const uint last_group_end = (token.token_index + score_task.num_queries) / ratio;
  const uint visible_span = last_group_end - aligned_group_start;
  const uint tile_offset = grid.x * BN;
  if (tile_offset >= visible_span) return;
  const uint tile_group_start = aligned_group_start + tile_offset;
  const uint tile_groups = min(uint(BN), visible_span - tile_offset);
  const uint logical_page = tile_group_start / groups_per_page;
  const uint physical_page = page_ids[ulong(token.req_slot) * pages_per_request + logical_page];
  const device T* B = Pages +
      (ulong(physical_page) * groups_per_page + tile_group_start % groups_per_page) * 128u;
  const short tgp_bm = short(FLAT ? score_task.num_queries * 4u : score_task.num_queries);
  const short tgp_bn = short(tile_groups);

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];
  thread mma_t mma_op(simd_group_id, simd_lane_id);
  const device T* task_q = Q + ulong(score_task.query_start) * 4u * 128u;

  if constexpr (FLAT) {
    mma_op.Ctile.clear();
    thread loader_a_t loader_a(task_q, 128, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(B, 128, Bs, simd_group_id, simd_lane_id);
    for (int k = 0; k < 8; ++k) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tgp_bm == BM) loader_a.load_unsafe();
      else loader_a.load_safe(short2(BK, tgp_bm));
      if (tgp_bn == BN) loader_b.load_unsafe();
      else loader_b.load_safe(short2(BK, tgp_bn));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
    threadgroup_barrier(mem_flags::mem_none);

    const uint lane_base = simd_lane_id & ~6u;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
      const int matrix_row = mma_op.sm + i * mma_t::TM_stride;
      const bool row_valid = matrix_row < int(score_task.num_queries * 4u);
      const uint query_offset = uint(matrix_row) / 4u;
      const uint query_index = score_task.query_start + query_offset;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
        const uint tile_column = uint(mma_op.sn) + uint(j) * uint(mma_t::TN_stride);
        thread const auto& frag = mma_op.Ctile.frag_at(i, j);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
          const uint column = tile_column + uint(e);
          const float positive_head_score = metal::max(frag[e], 0.0f);
          const float head0 = simd_shuffle(positive_head_score, ushort(lane_base | 0u));
          const float head1 = simd_shuffle(positive_head_score, ushort(lane_base | 2u));
          const float head2 = simd_shuffle(positive_head_score, ushort(lane_base | 4u));
          const float head3 = simd_shuffle(positive_head_score, ushort(lane_base | 6u));
          const float score = (head0 + head1 + head2 + head3) * rsqrt(128.0f);
          if (row_valid && (matrix_row % 4) == 0 && column < tile_groups) {
            const uint absolute_group = tile_group_start + column;
            const uint row_group_end = (token.token_index + query_offset + 1u) / ratio;
            if (absolute_group >= token.group_start && absolute_group < row_group_end) {
              const uint relative_group = absolute_group - token.group_start;
              O[ulong(query_index) * num_total_groups + relative_group] = score;
            }
          }
        }
      }
    }
  } else {
    float accum[decltype(mma_op.Ctile)::kElemsPerTile];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(mma_op.Ctile)::kElemsPerTile; ++i) accum[i] = 0.0f;

    STEEL_PRAGMA_UNROLL
    for (short head = 0; head < 4; ++head) {
      mma_op.Ctile.clear();
      thread loader_a_t loader_a(task_q + ulong(head) * 128u, 512, As, simd_group_id, simd_lane_id);
      thread loader_b_t loader_b(B, 128, Bs, simd_group_id, simd_lane_id);
      for (int k = 0; k < 8; ++k) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tgp_bm == BM) loader_a.load_unsafe();
        else loader_a.load_safe(short2(BK, tgp_bm));
        if (tgp_bn == BN) loader_b.load_unsafe();
        else loader_b.load_safe(short2(BK, tgp_bn));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(As, Bs);
        loader_a.next();
        loader_b.next();
      }
      threadgroup_barrier(mem_flags::mem_none);

      short accumulator_index = 0;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
          thread const auto& frag = mma_op.Ctile.frag_at(i, j);
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
            accum[accumulator_index++] += metal::max(frag[e], 0.0f);
          }
        }
      }
    }

    short accumulator_index = 0;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
      const int query_offset = mma_op.sm + i * mma_t::TM_stride;
      const bool row_valid = query_offset < int(score_task.num_queries);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
        const uint tile_column = uint(mma_op.sn) + uint(j) * uint(mma_t::TN_stride);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
          const uint column = tile_column + uint(e);
          const float score = accum[accumulator_index++] * rsqrt(128.0f);
          if (row_valid && column < tile_groups) {
            const uint absolute_group = tile_group_start + column;
            const uint row_group_end = (token.token_index + uint(query_offset) + 1u) / ratio;
            if (absolute_group >= token.group_start && absolute_group < row_group_end) {
              const uint query_index = score_task.query_start + uint(query_offset);
              const uint relative_group = absolute_group - token.group_start;
              O[ulong(query_index) * num_total_groups + relative_group] = score;
            }
          }
        }
      }
    }
  }
}

instantiate_kernel("qsa_indexer_flat_bm32", qsa_indexer_steel_score, bfloat16_t, 32, 64, 16, 2, 2, true);
instantiate_kernel("qsa_indexer_head_loop_bm32", qsa_indexer_steel_score, bfloat16_t, 32, 64, 16, 2, 2, false);
