template <typename T, int group_size, int bits, int D, bool batched>
[[kernel]] void psi_dec_qmv_quad(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant uint& num_active_rows [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  static_assert(!batched, "bucketed affine QMV supports non-batched input");
  if (tid.x >= num_active_rows) {
    return;
  }

  constexpr int quads_per_simd = SIMD_SIZE / QUAD_SIZE;
  constexpr int pack_factor = 32 / bits;
  constexpr int values_per_thread = D / QUAD_SIZE;
  constexpr int packs_per_thread = values_per_thread / pack_factor;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int results_per_quadgroup = 8;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_quadgroup] = {0};

  const ulong in_vec_size_w = ulong(in_vec_size) / pack_factor;
  const ulong in_vec_size_g = ulong(in_vec_size) / group_size;
  const int out_row = tid.y * quads_per_simd * results_per_quadgroup + quad_gid;
  if (out_row >= out_vec_size) {
    return;
  }

  w += out_row * in_vec_size_w + quad_lid * packs_per_thread;
  scales += out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
  x += ulong(tid.x) * in_vec_size + quad_lid * values_per_thread;
  y += ulong(tid.x) * out_vec_size + out_row;

  U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

  for (int row = 0; row < results_per_quadgroup; row++) {
    const int output_row = row * quads_per_simd + out_row;
    if (output_row < out_vec_size) {
      const device uint8_t* wl =
          reinterpret_cast<const device uint8_t*>(w + row * in_vec_size_w * quads_per_simd);
      const device T* sl = scales + row * in_vec_size_g * quads_per_simd;
      const device T* bl = biases + row * in_vec_size_g * quads_per_simd;
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, sl[0], bl[0], sum);
    }
  }

  for (int row = 0; row < results_per_quadgroup; row++) {
    result[row] = quad_sum(result[row]);
    const int output_row = row * quads_per_simd + out_row;
    if (quad_lid == 0 && output_row < out_vec_size) {
      y[row * quads_per_simd] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void psi_dec_qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant uint& num_active_rows [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(!batched, "bucketed affine QMV supports non-batched input");
  if (tid.x >= num_active_rows) {
    return;
  }
  qmv_fast_impl<T, group_size, bits>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void psi_dec_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant uint& num_active_rows [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(!batched, "bucketed affine QMV supports non-batched input");
  if (tid.x >= num_active_rows) {
    return;
  }
  qmv_impl<T, group_size, bits>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}
