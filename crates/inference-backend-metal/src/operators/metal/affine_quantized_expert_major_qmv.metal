template <typename T, int group_size, int bits, int qmm_min_rows>
[[kernel]] void expert_major_down_qmv(
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
    uint3 grid [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  for (uint route = tid.x; route < num_active_tokens * num_experts_per_token;
       route += grid.x) {
    const uint expert = experts_by_route[route];
    const uint end = expert_offsets[expert + 1];
    if (end - expert_offsets[expert] >= uint(qmm_min_rows)) {
      // Skip this QMM segment without changing the block's route stride.
      route += ((end - route - 1) / grid.x) * grid.x;
      continue;
    }
    const long weight_bytes =
        long(N) * K * get_bytes_per_pack<bits>() / get_pack_factor<bits, 8>();
    const long affine_values = long(N) * (K / group_size);
    qmv_impl<T, group_size, bits>(
        reinterpret_cast<const device uint32_t *>(
            reinterpret_cast<const device uint8_t *>(w) +
            expert * weight_bytes),
        scales + expert * affine_values, biases + expert * affine_values,
        x + long(route) * K, y + long(route) * N, K, N, tid.y, simd_gid,
        simd_lid);
  }
}

template <typename T, int group_size, int bits, int qmm_min_rows>
[[kernel]] void expert_major_gate_up_swiglu_qmv(
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
    uint3 grid [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  for (uint route = tid.x; route < num_active_tokens * num_experts_per_token;
       route += grid.x) {
    const uint expert = experts_by_route[route];
    const uint end = expert_offsets[expert + 1];
    if (end - expert_offsets[expert] >= uint(qmm_min_rows)) {
      // Skip this QMM segment without changing the block's route stride.
      route += ((end - route - 1) / grid.x) * grid.x;
      continue;
    }
    const long weight_bytes =
        long(N) * K * get_bytes_per_pack<bits>() / get_pack_factor<bits, 8>();
    const long affine_values = long(N) * (K / group_size);
    qmv_gate_up_swiglu<T, group_size, bits>(
        reinterpret_cast<const device uint32_t *>(
            reinterpret_cast<const device uint8_t *>(gate_w) +
            expert * weight_bytes),
        gate_scales + expert * affine_values,
        gate_biases + expert * affine_values,
        reinterpret_cast<const device uint32_t *>(
            reinterpret_cast<const device uint8_t *>(up_w) +
            expert * weight_bytes),
        up_scales + expert * affine_values, up_biases + expert * affine_values,
        x + long(route) * K, y + long(route) * N, K, N, tid.y, simd_gid,
        simd_lid);
  }
}

template <typename T, const int group_size, const int bits>
[[kernel]] void expert_major_down_qmv_all(
    const device uint32_t *w [[buffer(0)]],
    const device T *scales [[buffer(1)]], const device T *biases [[buffer(2)]],
    const device T *x [[buffer(3)]],
    const device uint32_t *experts_by_route [[buffer(4)]],
    device T *y [[buffer(5)]], const constant int &in_vec_size [[buffer(6)]],
    const constant int &out_vec_size [[buffer(7)]],
    const constant uint &num_active_tokens [[buffer(8)]],
    const constant uint &num_experts_per_token [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint route = tid.x;
  const uint num_active_routes = num_active_tokens * num_experts_per_token;
  if (route >= num_active_routes) {
    return;
  }
  const uint expert = experts_by_route[route];
  const long in_vec_size_w = long(in_vec_size) *
                             (bits == 3 || bits == 6 ? 3 : 4) /
                             (bits == 3   ? 8
                              : bits == 6 ? 4
                                          : 32 / bits);
  const int in_vec_size_g = in_vec_size / group_size;
  const long expert_weight_stride = long(out_vec_size) * in_vec_size_w;
  const long expert_affine_stride = long(out_vec_size) * in_vec_size_g;

  qmv_impl<T, group_size, bits>(
      (const device uint32_t *)((const device uint8_t *)w +
                                expert * expert_weight_stride),
      scales + expert * expert_affine_stride,
      biases + expert * expert_affine_stride, x + long(route) * in_vec_size,
      y + long(route) * out_vec_size, in_vec_size, out_vec_size, tid.y,
      simd_gid, simd_lid);
}

template <typename T, const int group_size, const int bits>
[[kernel]] void expert_major_gate_up_swiglu_qmv_all(
    const device uint32_t *gate_w [[buffer(0)]],
    const device T *gate_scales [[buffer(1)]],
    const device T *gate_biases [[buffer(2)]],
    const device uint32_t *up_w [[buffer(3)]],
    const device T *up_scales [[buffer(4)]],
    const device T *up_biases [[buffer(5)]], const device T *x [[buffer(6)]],
    const device uint32_t *experts_by_route [[buffer(7)]],
    device T *y [[buffer(8)]], const constant int &in_vec_size [[buffer(9)]],
    const constant int &out_vec_size [[buffer(10)]],
    const constant uint &num_active_tokens [[buffer(11)]],
    const constant uint &num_experts_per_token [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint route = tid.x;
  const uint num_active_routes = num_active_tokens * num_experts_per_token;
  if (route >= num_active_routes) {
    return;
  }
  const uint expert = experts_by_route[route];
  const long in_vec_size_w = long(in_vec_size) *
                             (bits == 3 || bits == 6 ? 3 : 4) /
                             (bits == 3   ? 8
                              : bits == 6 ? 4
                                          : 32 / bits);
  const int in_vec_size_g = in_vec_size / group_size;
  const long expert_weight_stride = long(out_vec_size) * in_vec_size_w;
  const long expert_affine_stride = long(out_vec_size) * in_vec_size_g;

  qmv_gate_up_swiglu<T, group_size, bits>(
      (const device uint32_t *)((const device uint8_t *)gate_w +
                                expert * expert_weight_stride),
      gate_scales + expert * expert_affine_stride,
      gate_biases + expert * expert_affine_stride,
      (const device uint32_t *)((const device uint8_t *)up_w +
                                expert * expert_weight_stride),
      up_scales + expert * expert_affine_stride,
      up_biases + expert * expert_affine_stride, x + long(route) * in_vec_size,
      y + long(route) * out_vec_size, in_vec_size, out_vec_size, tid.y,
      simd_gid, simd_lid);
}
