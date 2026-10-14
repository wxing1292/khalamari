// Map one expert-row tile to its compact route interval.
template <int BM, int min_rows, int num_experts>
inline bool expert_row_tile(
    const device uint* offsets, uint num_active_routes, uint row_tile,
    uint lane, thread uint& expert, thread uint& route, thread uint& end) {
  const uint num_active_row_tiles = min(
      num_active_routes,
      (num_active_routes + BM - 1) / BM + uint(num_experts) - 1);
  if (row_tile >= num_active_row_tiles) {
    return false;
  }
  uint base = 0;
  for (uint first = 0; first < uint(num_experts); first += 32) {
    const uint index = first + lane;
    const uint begin = index < uint(num_experts) ? offsets[index] : num_active_routes;
    const uint limit = index < uint(num_experts) ? offsets[index + 1] : num_active_routes;
    const uint rows = limit - begin;
    const uint tiles = rows >= uint(min_rows) ? (rows + BM - 1) / BM : 0;
    const uint preceding = simd_prefix_exclusive_sum(tiles);
    const uint local = row_tile - base;
    const uint owner = simd_min(
        local >= preceding && local < preceding + tiles ? lane : 32u);
    if (owner < 32) {
      expert = first + owner;
      route = simd_shuffle(begin, ushort(owner)) +
              (local - simd_shuffle(preceding, ushort(owner))) * BM;
      end = min(route + BM, simd_shuffle(limit, ushort(owner)));
      return true;
    }
    base += simd_sum(tiles);
  }
  return false;
}

// Each active block owns one row tile within one expert segment.
template <typename T, int group_size, int bits, int qmm_min_rows,
          int num_experts, int BM>
[[kernel]] void expert_major_down_qmm(
    const device uint32_t *w [[buffer(0)]],
    const device T *scales [[buffer(1)]], const device T *biases [[buffer(2)]],
    const device T *x [[buffer(3)]],
    const device uint *experts_by_route [[buffer(4)]],
    device T *y [[buffer(5)]], const constant int &K [[buffer(6)]],
    const constant int &N [[buffer(7)]],
    const constant uint &num_active_tokens [[buffer(8)]],
    const constant uint &num_experts_per_token [[buffer(9)]],
    const device uint *expert_offsets [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int LD = BK + 16 / sizeof(T);
  const uint num_active_routes = num_active_tokens * num_experts_per_token;
  uint expert, route, end;
  if (!expert_row_tile<BM, qmm_min_rows, num_experts>(
          expert_offsets, num_active_routes, tid.x, simd_lid, expert, route,
          end)) {
    return;
  }
  const int col = tid.y * BN;
  const short num_cols = min(BN, N - col);
  const long weight_row_bytes =
      long(K) * get_bytes_per_pack<bits>() / get_pack_factor<bits, 8>();
  threadgroup T Xs[BM * LD];
  threadgroup T Ws[BN * LD];
  const long weight_row = long(expert) * N + col;
  auto result = affine_qmm_tile<T, T, T, group_size, bits, BM, BK>(
      reinterpret_cast<const device uint8_t *>(w) +
          weight_row * weight_row_bytes,
      scales + weight_row * (K / group_size),
      biases + weight_row * (K / group_size), x + long(route) * K, Xs, Ws, K,
      short(end - route), num_cols, simd_gid, simd_lid);
  result.store_result_safe(y + long(route) * N + col, N, short2(num_cols, end - route));
}

template <typename T, int group_size, int bits, int qmm_min_rows,
          int num_experts, int BM>
[[kernel]] void expert_major_gate_up_swiglu_qmm(
    const device uint32_t *gate_w [[buffer(0)]],
    const device T *gate_scales [[buffer(1)]],
    const device T *gate_biases [[buffer(2)]],
    const device uint32_t *up_w [[buffer(3)]],
    const device T *up_scales [[buffer(4)]],
    const device T *up_biases [[buffer(5)]], const device T *x [[buffer(6)]],
    const device uint *experts_by_route [[buffer(7)]],
    device T *y [[buffer(8)]], const constant int &K [[buffer(9)]],
    const constant int &N [[buffer(10)]],
    const constant uint &num_active_tokens [[buffer(11)]],
    const constant uint &num_experts_per_token [[buffer(12)]],
    const device uint *expert_offsets [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int LD = BK + 16 / sizeof(T);
  const uint num_active_routes = num_active_tokens * num_experts_per_token;
  uint expert, route, end;
  if (!expert_row_tile<BM, qmm_min_rows, num_experts>(
          expert_offsets, num_active_routes, tid.x, simd_lid, expert, route,
          end)) {
    return;
  }
  const int col = tid.y * BN;
  const short num_cols = min(BN, N - col);
  const long weight_row_bytes =
      long(K) * get_bytes_per_pack<bits>() / get_pack_factor<bits, 8>();
  threadgroup T Xs[BM * LD];
  threadgroup T Ws[BN * LD];
  const long weight_row = long(expert) * N + col;
  auto gate = affine_qmm_tile<T, T, T, group_size, bits, BM, BK>(
      reinterpret_cast<const device uint8_t *>(gate_w) +
          weight_row * weight_row_bytes,
      gate_scales + weight_row * (K / group_size),
      gate_biases + weight_row * (K / group_size), x + long(route) * K, Xs, Ws,
      K, short(end - route), num_cols, simd_gid, simd_lid);
  auto up = affine_qmm_tile<T, T, T, group_size, bits, BM, BK>(
      reinterpret_cast<const device uint8_t *>(up_w) +
          weight_row * weight_row_bytes,
      up_scales + weight_row * (K / group_size),
      up_biases + weight_row * (K / group_size), x + long(route) * K, Xs, Ws, K,
      short(end - route), num_cols, simd_gid, simd_lid);
  // Preserve the sparse MLP activation's storage-dtype rounding stages.
  for (short i = 0; i < decltype(gate.Ctile)::kElemsPerTile; ++i) {
    const T g = T(gate.Ctile.elems()[i]);
    const T u = T(up.Ctile.elems()[i]);
    const T sigmoid = T(1.0f / (1.0f + exp(-float(g))));
    const T silu = T(float(g) * float(sigmoid));
    gate.Ctile.elems()[i] = float(T(float(silu) * float(u)));
  }
  gate.store_result_safe(y + long(route) * N + col, N, short2(num_cols, end - route));
}
