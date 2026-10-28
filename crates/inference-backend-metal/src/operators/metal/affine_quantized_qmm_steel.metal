// Each block dequantizes one weight tile and reuses it across BM input rows.
template <typename InT, typename ParamT, typename OutT,
          int group_size, int bits, int BM, int BN, int BK>
[[kernel]] void affine_qmm_steel(
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
  // WM/WN partition output fragments. Each SIMDgroup traverses all K through sequential BK stages.
  constexpr int WM = BM >= 32 ? 2 : 1;
  constexpr int WN = BM < 32 ? 4 : 2;
  threadgroup T Xs[BM * LD];
  threadgroup T Ws[BN * LD];
  using mma_t = mlx::steel::BlockMMA<T, OutT, BM, BN, BK, WM, WN, false, true, LD, LD>;
  using loader_x_t = conditional_t<same_dtype,
      mlx::steel::BlockLoader<T, BM, BK, LD, 1, WM * WN * SIMD_SIZE>,
      KhalamariMixedBlockLoader<InT, T, BM, BK, LD, 1, WM * WN * SIMD_SIZE>>;
  using loader_w_t = KhalamariMixedQuantizedBlockLoader<ParamT, T, BN, BK, LD, 1,
      WM * WN * SIMD_SIZE, group_size, bits>;
  const int y_row = tid.y * BM, y_col = tid.x * BN;
  const short rows = min(BM, int(num_active_rows) - y_row);
  const short cols = min(BN, N - y_col);
  const long weight_row_bytes = long(K) * get_bytes_per_pack<bits>() / get_pack_factor<bits, 8>();
  loader_x_t loader_x(x + long(y_row) * K, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(reinterpret_cast<const device uint8_t*>(w) + y_col * weight_row_bytes,
      scales + long(y_col) * (K / group_size), biases + long(y_col) * (K / group_size), K, Ws, simd_gid, simd_lid);
  mma_t mma(simd_gid, simd_lid);
  // The active count may change on replay; choose the full-tile loop per block.
  if (rows == BM && cols == BN) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe(); loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma.mma(Xs, Ws);
      loader_x.next(); loader_w.next();
    }
  } else {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, rows)); loader_w.load_safe(short2(BK, cols));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma.mma(Xs, Ws);
      loader_x.next(); loader_w.next();
    }
  }
  y += long(y_row) * N + y_col;
  if (rows == BM && cols == BN) mma.store_result(y, N);
  else mma.store_result_safe(y, N, short2(cols, rows));
}
