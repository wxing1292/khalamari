// Gate and up reuse X in SMEM. Keep both F32 accumulators in REGISTER.
template <typename InT, typename ParamT, typename OutT,
          int group_size, int bits, int BM, int BN, int BK>
[[kernel]] void affine_qmm_gate_up_swiglu(
    const device uint32_t* w [[buffer(0)]],
    const device ParamT* scales [[buffer(1)]],
    const device ParamT* biases [[buffer(2)]],
    const device InT* x [[buffer(3)]], device OutT* y [[buffer(4)]],
    const constant int& K [[buffer(5)]], const constant int& N [[buffer(6)]],
    const constant uint& num_active_rows [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (tid.y * BM >= num_active_rows) return;
  constexpr bool same_dtype = is_same_v<InT, ParamT> && is_same_v<InT, OutT>;
  using T = conditional_t<same_dtype, InT, float>;
  constexpr int LD = BK + 16 / sizeof(T);
  constexpr int WM = BM >= 32 ? 2 : 1, WN = 2;
  threadgroup T Xs[BM * LD];
  threadgroup T Ws[BN * LD];
  using mma_t = mlx::steel::BlockMMA<T, OutT, BM, BN, BK, WM, WN, false, true, LD, LD>;
  using loader_x_t = conditional_t<same_dtype,
      mlx::steel::BlockLoader<T, BM, BK, LD, 1, WM * WN * SIMD_SIZE>,
      PsiDecMixedBlockLoader<InT, T, BM, BK, LD, 1, WM * WN * SIMD_SIZE>>;
  using loader_w_t = PsiDecMixedQuantizedBlockLoader<ParamT, T, BN, BK, LD, 1,
      WM * WN * SIMD_SIZE, group_size, bits>;
  const int y_row = tid.y * BM, y_col = tid.x * BN;
  const short rows = min(BM, int(num_active_rows) - y_row), cols = min(BN, N - y_col);
  const long weight_row_bytes = long(K) * get_bytes_per_pack<bits>() / get_pack_factor<bits, 8>();
  loader_x_t loader_x(x + long(y_row) * K, K, Xs, simd_gid, simd_lid);
  const device uint8_t* weights = reinterpret_cast<const device uint8_t*>(w);
  loader_w_t loader_gate(weights + long(y_col) * weight_row_bytes,
      scales + long(y_col) * (K / group_size), biases + long(y_col) * (K / group_size), K, Ws, simd_gid, simd_lid);
  loader_w_t loader_up(weights + long(N + y_col) * weight_row_bytes,
      scales + long(N + y_col) * (K / group_size), biases + long(N + y_col) * (K / group_size), K, Ws, simd_gid, simd_lid);
  mma_t gate(simd_gid, simd_lid), up(simd_gid, simd_lid);
  const bool full = rows == BM && cols == BN;
  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (full) { loader_x.load_unsafe(); loader_gate.load_unsafe(); }
    else { loader_x.load_safe(short2(BK, rows)); loader_gate.load_safe(short2(BK, cols)); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gate.mma(Xs, Ws);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (full) loader_up.load_unsafe(); else loader_up.load_safe(short2(BK, cols));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    up.mma(Xs, Ws);
    loader_x.next(); loader_gate.next(); loader_up.next();
  }
  // Match materialized projections: round each projection once, then do SwiGLU in F32.
  for (short i = 0; i < decltype(gate.Ctile)::kElemsPerTile; ++i) {
    const float g = float(OutT(gate.Ctile.elems()[i]));
    const float u = float(OutT(up.Ctile.elems()[i]));
    gate.Ctile.elems()[i] = (g / (1.0f + exp(-g))) * u;
  }
  y += long(y_row) * N + y_col;
  if (full) gate.store_result(y, N);
  else gate.store_result_safe(y, N, short2(cols, rows));
}
