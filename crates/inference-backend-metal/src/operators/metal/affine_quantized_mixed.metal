template <typename InT, typename ParamT, typename OutT, int group_size, int bits>
[[kernel]] void mixed_qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device ParamT* scales [[buffer(1)]],
    const device ParamT* biases [[buffer(2)]],
    const device InT* x [[buffer(3)]],
    device OutT* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant uint& num_active_rows [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (tid.x >= num_active_rows) {
    return;
  }

  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits;
  constexpr int bytes_per_pack = power_of_2_bits ? 4 : 3;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const ulong in_vec_size_w = ulong(in_vec_size) * bytes_per_pack / pack_factor;
  const ulong in_vec_size_g = ulong(in_vec_size) / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * static_cast<int64_t>(in_vec_size) + simd_lid * values_per_thread;
  y += tid.x * static_cast<int64_t>(out_vec_size) + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<InT, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device ParamT* sl = scales + row * in_vec_size_g;
      const device ParamT* bl = biases + row * in_vec_size_g;

      U s = static_cast<U>(sl[0]);
      U b = static_cast<U>(bl[0]);
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<OutT>(result[row]);
    }
  }
}

template <typename InT, typename ParamT, typename OutT, int group_size, int bits>
[[kernel]] void mixed_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device ParamT* scales [[buffer(1)]],
    const device ParamT* biases [[buffer(2)]],
    const device InT* x [[buffer(3)]],
    device OutT* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant uint& num_active_rows [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (tid.x >= num_active_rows) {
    return;
  }

  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 2;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits;
  constexpr int bytes_per_pack = power_of_2_bits ? 4 : 3;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const ulong in_vec_size_w = ulong(in_vec_size) * bytes_per_pack / pack_factor;
  const ulong in_vec_size_g = ulong(in_vec_size) / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * static_cast<int64_t>(in_vec_size) + simd_lid * values_per_thread;
  y += tid.x * static_cast<int64_t>(out_vec_size) + out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    U sum = load_vector<InT, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device ParamT* sl = scales + row * in_vec_size_g;
        const device ParamT* bl = biases + row * in_vec_size_g;
        result[row] += qdot<U, values_per_thread, bits>(
            wl, x_thread, static_cast<U>(sl[0]), static_cast<U>(bl[0]), sum);
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    U sum = load_vector_safe<InT, U, values_per_thread, bits>(x, x_thread, remaining);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device ParamT* sl = scales + row * in_vec_size_g;
        const device ParamT* bl = biases + row * in_vec_size_g;
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, static_cast<U>(sl[0]), static_cast<U>(bl[0]), sum, remaining);
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    if (out_row + row < out_vec_size) {
      U value = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<OutT>(value);
      }
    }
  }
}

template <
    typename InT,
    typename OutT,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short alignment = 1,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct PsiDecMixedBlockLoader {
  STEEL_CONST short vec_size = n_reads;

  const ulong src_ld;
  const ulong tile_stride;
  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup OutT* dst;
  const device InT* src;

  PsiDecMixedBlockLoader(
      const device InT* src_,
      const int src_ld_,
      threadgroup OutT* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj) {}

  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = static_cast<OutT>(src[i * src_ld + j]);
      }
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = OutT(0);
        }
      }
      return;
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        const bool valid = (i < src_tile_dim.y) && (j < src_tile_dim.x);
        dst[i * dst_ld + j] = valid ? static_cast<OutT>(src[i * src_ld + j]) : OutT(0);
      }
    }
  }

  METAL_FUNC void next() {
    src += tile_stride;
  }
};

template <
    typename ParamT,
    typename OutT,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct PsiDecMixedQuantizedBlockLoader {
  static_assert(
      BCOLS <= group_size,
      "The group size should be larger than the columns");
  static_assert(
      group_size % BCOLS == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 6 || bits == 8,
      "Template undefined for bits not in {2, 3, 4, 6, 8}");

  MLX_MTL_CONST short pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 8 / bits;
  MLX_MTL_CONST short bytes_per_pack = (bits == 3 || bits == 6) ? 3 : 1;
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size / BCOLS;

  const ulong src_ld;
  const ulong tile_stride;
  short group_step_cnt;
  const ulong group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup OutT* dst;
  const device uint8_t* src;
  const device ParamT* scales;
  const device ParamT* biases;

  PsiDecMixedQuantizedBlockLoader(
      const device uint8_t* src_,
      const device ParamT* scales_,
      const device ParamT* biases_,
      const int src_ld_,
      threadgroup OutT* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  METAL_FUNC void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    OutT scale = static_cast<OutT>(*scales);
    OutT bias = static_cast<OutT>(*biases);
    for (int i = 0; i < n_reads; i++) {
      dequantize<OutT, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = OutT(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = OutT(0);
      }
      return;
    }

    OutT scale = static_cast<OutT>(*scales);
    OutT bias = static_cast<OutT>(*biases);
    for (int i = 0; i < n_reads; i++) {
      dequantize<OutT, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  METAL_FUNC void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};
