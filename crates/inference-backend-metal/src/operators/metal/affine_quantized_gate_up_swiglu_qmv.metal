template <typename T, int group_size, int bits>
METAL_FUNC void qmv_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint column_group,
    uint simd_gid,
    uint simd_lid) {
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
  const int out_row = column_group * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;
  y += out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;
        result[row] += qdot<U, values_per_thread, bits>(
            wl, x_thread, sl[0], bl[0], sum);
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
    U sum = load_vector_safe<T, U, values_per_thread, bits>(x, x_thread, remaining);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, sl[0], bl[0], sum, remaining);
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    if (out_row + row < out_vec_size) {
      U value = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(value);
      }
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_gate_up_swiglu(
    const device uint32_t* gate_w,
    const device T* gate_scales,
    const device T* gate_biases,
    const device uint32_t* up_w,
    const device T* up_scales,
    const device T* up_biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint column_group,
    uint simd_gid,
    uint simd_lid) {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 2;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits;
  constexpr int bytes_per_pack = power_of_2_bits ? 4 : 3;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* gate_ws = (const device uint8_t*)gate_w;
  const device uint8_t* up_ws = (const device uint8_t*)up_w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U gate_result[results_per_simdgroup] = {0};
  thread U up_result[results_per_simdgroup] = {0};

  const ulong in_vec_size_w = ulong(in_vec_size) * bytes_per_pack / pack_factor;
  const ulong in_vec_size_g = ulong(in_vec_size) / group_size;
  const int out_row = column_group * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }

  gate_ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  up_ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  gate_scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  gate_biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  up_scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  up_biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;
  y += out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* gate_wl = gate_ws + row * in_vec_size_w;
        const device T* gate_sl = gate_scales + row * in_vec_size_g;
        const device T* gate_bl = gate_biases + row * in_vec_size_g;
        const device uint8_t* up_wl = up_ws + row * in_vec_size_w;
        const device T* up_sl = up_scales + row * in_vec_size_g;
        const device T* up_bl = up_biases + row * in_vec_size_g;
        gate_result[row] += qdot<U, values_per_thread, bits>(
            gate_wl, x_thread, gate_sl[0], gate_bl[0], sum);
        up_result[row] += qdot<U, values_per_thread, bits>(
            up_wl, x_thread, up_sl[0], up_bl[0], sum);
      }
    }

    gate_ws += block_size * bytes_per_pack / pack_factor;
    up_ws += block_size * bytes_per_pack / pack_factor;
    gate_scales += block_size / group_size;
    gate_biases += block_size / group_size;
    up_scales += block_size / group_size;
    up_biases += block_size / group_size;
    x += block_size;
  }

  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    U sum = load_vector_safe<T, U, values_per_thread, bits>(x, x_thread, remaining);

    for (int row = 0; row < results_per_simdgroup; row++) {
      if (out_row + row < out_vec_size) {
        const device uint8_t* gate_wl = gate_ws + row * in_vec_size_w;
        const device T* gate_sl = gate_scales + row * in_vec_size_g;
        const device T* gate_bl = gate_biases + row * in_vec_size_g;
        const device uint8_t* up_wl = up_ws + row * in_vec_size_w;
        const device T* up_sl = up_scales + row * in_vec_size_g;
        const device T* up_bl = up_biases + row * in_vec_size_g;
        gate_result[row] += qdot_safe<U, values_per_thread, bits>(
            gate_wl, x_thread, gate_sl[0], gate_bl[0], sum, remaining);
        up_result[row] += qdot_safe<U, values_per_thread, bits>(
            up_wl, x_thread, up_sl[0], up_bl[0], sum, remaining);
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    if (out_row + row < out_vec_size) {
      U gate = simd_sum(gate_result[row]);
      U up = simd_sum(up_result[row]);
      if (simd_lid == 0) {
        T gate_t = static_cast<T>(gate);
        T up_t = static_cast<T>(up);
        T sigmoid = static_cast<T>(1.0f / (1.0f + exp(-static_cast<float>(gate_t))));
        T silu = static_cast<T>(static_cast<float>(gate_t) * static_cast<float>(sigmoid));
        y[row] = static_cast<T>(static_cast<float>(silu) * static_cast<float>(up_t));
      }
    }
  }
}

template <typename T, const int group_size, const int bits>
[[kernel]] void
dense_gate_up_swiglu(const device uint32_t *w [[buffer(0)]], const device T *scales [[buffer(1)]],
                     const device T *biases [[buffer(2)]], const device T *x [[buffer(3)]], device T *y [[buffer(4)]],
                     const constant int &in_vec_size [[buffer(5)]], const constant int &out_vec_size [[buffer(6)]],
                     uint3 tid [[threadgroup_position_in_grid]], uint simd_gid [[simdgroup_index_in_threadgroup]],
                     uint simd_lid [[thread_index_in_simdgroup]]) {
  const ulong in_vec_size_w = ulong(in_vec_size) * (bits == 3 || bits == 6 ? 3 : 4) /
      (bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits);
  const ulong in_vec_size_g = ulong(in_vec_size) / group_size;
  const device uint32_t* gate_w = w;
  const device uint32_t* up_w = (const device uint32_t*)((const device uint8_t*)w + out_vec_size * in_vec_size_w);
  const device T* gate_scales = scales;
  const device T* gate_biases = biases;
  const device T* up_scales = scales + out_vec_size * in_vec_size_g;
  const device T* up_biases = biases + out_vec_size * in_vec_size_g;
  qmv_gate_up_swiglu<T, group_size, bits>(
      gate_w,
      gate_scales,
      gate_biases,
      up_w,
      up_scales,
      up_biases,
      x + ulong(tid.x) * in_vec_size,
      y + ulong(tid.x) * out_vec_size,
      in_vec_size,
      out_vec_size,
      tid.y,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits>
[[kernel]] void split_gate_up_swiglu(
    const device uint32_t* gate_w [[buffer(0)]],
    const device T* gate_scales [[buffer(1)]],
    const device T* gate_biases [[buffer(2)]],
    const device uint32_t* up_w [[buffer(3)]],
    const device T* up_scales [[buffer(4)]],
    const device T* up_biases [[buffer(5)]],
    const device T* x [[buffer(6)]],
    device T* y [[buffer(7)]],
    const constant int& in_vec_size [[buffer(8)]],
    const constant int& out_vec_size [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  qmv_gate_up_swiglu<T, group_size, bits>(
      gate_w,
      gate_scales,
      gate_biases,
      up_w,
      up_scales,
      up_biases,
      x + ulong(tid.x) * in_vec_size,
      y + ulong(tid.x) * out_vec_size,
      in_vec_size,
      out_vec_size,
      tid.y,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits>
[[kernel]] void token_major_gate_up_swiglu(
    const device uint32_t* gate_w [[buffer(0)]],
    const device T* gate_scales [[buffer(1)]],
    const device T* gate_biases [[buffer(2)]],
    const device uint32_t* up_w [[buffer(3)]],
    const device T* up_scales [[buffer(4)]],
    const device T* up_biases [[buffer(5)]],
    const device T* x [[buffer(6)]],
    const device uint32_t* lhs_indices [[buffer(7)]],
    const device uint32_t* rhs_indices [[buffer(8)]],
    device T* y [[buffer(9)]],
    const constant int& in_vec_size [[buffer(10)]],
    const constant int& out_vec_size [[buffer(11)]],
    const constant int& num_experts [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint route = tid.z;
  const uint input_row = lhs_indices[route];
  const uint expert = rhs_indices[route];
  if (expert >= uint(num_experts)) {
    return;
  }
  const ulong in_vec_size_w = ulong(in_vec_size) * (bits == 3 || bits == 6 ? 3 : 4) /
      (bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits);
  const ulong in_vec_size_g = ulong(in_vec_size) / group_size;
  const ulong expert_weight_stride = out_vec_size * in_vec_size_w;
  const ulong expert_affine_stride = out_vec_size * in_vec_size_g;
  qmv_gate_up_swiglu<T, group_size, bits>(
      (const device uint32_t*)((const device uint8_t*)gate_w + expert * expert_weight_stride),
      gate_scales + expert * expert_affine_stride,
      gate_biases + expert * expert_affine_stride,
      (const device uint32_t*)((const device uint8_t*)up_w + expert * expert_weight_stride),
      up_scales + expert * expert_affine_stride,
      up_biases + expert * expert_affine_stride,
      x + ulong(input_row) * in_vec_size,
      y + ulong(route) * out_vec_size,
      in_vec_size,
      out_vec_size,
      tid.y,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits>
[[kernel]] void token_major_gate_up_swiglu_bucketed(
    const device uint32_t* gate_w [[buffer(0)]],
    const device T* gate_scales [[buffer(1)]],
    const device T* gate_biases [[buffer(2)]],
    const device uint32_t* up_w [[buffer(3)]],
    const device T* up_scales [[buffer(4)]],
    const device T* up_biases [[buffer(5)]],
    const device T* x [[buffer(6)]],
    const device uint32_t* lhs_indices [[buffer(7)]],
    const device uint32_t* rhs_indices [[buffer(8)]],
    device T* y [[buffer(9)]],
    const constant int& in_vec_size [[buffer(10)]],
    const constant int& out_vec_size [[buffer(11)]],
    const constant int& num_experts [[buffer(12)]],
    const constant uint& num_active_tokens [[buffer(13)]],
    const constant uint& num_experts_per_token [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint route = tid.z;
  const uint num_active_routes = num_active_tokens * num_experts_per_token;
  if (route >= num_active_routes) {
    return;
  }
  const uint input_row = lhs_indices[route];
  const uint expert = rhs_indices[route];
  if (expert >= uint(num_experts)) {
    return;
  }
  const ulong in_vec_size_w = ulong(in_vec_size) * (bits == 3 || bits == 6 ? 3 : 4) /
      (bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits);
  const ulong in_vec_size_g = ulong(in_vec_size) / group_size;
  const ulong expert_weight_stride = out_vec_size * in_vec_size_w;
  const ulong expert_affine_stride = out_vec_size * in_vec_size_g;
  qmv_gate_up_swiglu<T, group_size, bits>(
      (const device uint32_t*)((const device uint8_t*)gate_w + expert * expert_weight_stride),
      gate_scales + expert * expert_affine_stride,
      gate_biases + expert * expert_affine_stride,
      (const device uint32_t*)((const device uint8_t*)up_w + expert * expert_weight_stride),
      up_scales + expert * expert_affine_stride,
      up_biases + expert * expert_affine_stride,
      x + ulong(input_row) * in_vec_size,
      y + ulong(route) * out_vec_size,
      in_vec_size,
      out_vec_size,
      tid.y,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits, const bool fast>
[[kernel]] void token_major_down_matmul_bucketed(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    const constant uint& num_active_tokens [[buffer(21)]],
    const constant uint& num_experts_per_token [[buffer(22)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint route = tid.z;
  const uint num_active_routes = num_active_tokens * num_experts_per_token;
  if (route >= num_active_routes) {
    return;
  }

  const int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      out_vec_size * M,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  if constexpr (fast) {
    qmv_fast_impl<T, group_size, bits>(
        w,
        scales,
        biases,
        x,
        y,
        in_vec_size,
        out_vec_size,
        tid,
        simd_gid,
        simd_lid);
  } else {
    qmv_impl<T, group_size, bits>(
        w,
        scales,
        biases,
        x,
        y,
        in_vec_size,
        out_vec_size,
        tid,
        simd_gid,
        simd_lid);
  }
}
