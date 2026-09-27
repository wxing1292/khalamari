// Each active block starts an eight-row tile within one expert segment.
// Interior route positions return before loading operands or reaching barriers.
template <typename T, int group_size, int bits, int qmm_min_rows>
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
  constexpr int BM = 8;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int LD = BK + 16 / sizeof(float);
  const uint num_active_routes = num_active_tokens * num_experts_per_token;
  const uint route = tid.x;
  if (route >= num_active_routes) {
    return;
  }
  const uint expert = experts_by_route[route];
  if (expert_offsets[expert + 1] - expert_offsets[expert] <
      uint(qmm_min_rows)) {
    return;
  }
  if ((route - expert_offsets[expert]) % BM != 0) {
    return;
  }
  const uint end = min(route + BM, expert_offsets[expert + 1]);
  const int col = tid.y * BN;
  const short num_cols = min(BN, N - col);
  const long weight_row_bytes =
      long(K) * get_bytes_per_pack<bits>() / get_pack_factor<bits, 8>();
  threadgroup float Xs[BM * LD];
  threadgroup float Ws[BN * LD];
  const long weight_row = long(expert) * N + col;
  auto result = affine_qmm_tile<T, T, float, group_size, bits, BM, BK>(
      reinterpret_cast<const device uint8_t *>(w) +
          weight_row * weight_row_bytes,
      scales + weight_row * (K / group_size),
      biases + weight_row * (K / group_size), x + long(route) * K, Xs, Ws, K,
      short(end - route), num_cols, simd_gid, simd_lid);
  result.store_result_safe(y + long(route) * N + col, N, short2(num_cols, end - route));
}

template <typename T, int group_size, int bits, int qmm_min_rows>
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
  constexpr int BM = 8;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int LD = BK + 16 / sizeof(float);
  const uint num_active_routes = num_active_tokens * num_experts_per_token;
  const uint route = tid.x;
  if (route >= num_active_routes) {
    return;
  }
  const uint expert = experts_by_route[route];
  if (expert_offsets[expert + 1] - expert_offsets[expert] <
      uint(qmm_min_rows)) {
    return;
  }
  if ((route - expert_offsets[expert]) % BM != 0) {
    return;
  }
  const uint end = min(route + BM, expert_offsets[expert + 1]);
  const int col = tid.y * BN;
  const short num_cols = min(BN, N - col);
  const long weight_row_bytes =
      long(K) * get_bytes_per_pack<bits>() / get_pack_factor<bits, 8>();
  threadgroup float Xs[BM * LD];
  threadgroup float Ws[BN * LD];
  const long weight_row = long(expert) * N + col;
  auto gate = affine_qmm_tile<T, T, float, group_size, bits, BM, BK>(
      reinterpret_cast<const device uint8_t *>(gate_w) +
          weight_row * weight_row_bytes,
      gate_scales + weight_row * (K / group_size),
      gate_biases + weight_row * (K / group_size), x + long(route) * K, Xs, Ws,
      K, short(end - route), num_cols, simd_gid, simd_lid);
  auto up = affine_qmm_tile<T, T, float, group_size, bits, BM, BK>(
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
