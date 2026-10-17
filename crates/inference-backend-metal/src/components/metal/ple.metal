struct PLEToken { uint batch_start, src_state_slot; };
struct PLERequest { uint query_start, src_state_slot, dst_state_slot; };

// Preserve eager BF16 storage boundaries for register-only intermediates.
float stored(float value) { return float(bfloat(value)); }

float block_sum(float value, threadgroup float* sums, uint lane, uint simd_group) {
    value = simd_sum(value);
    if (lane == 0) sums[simd_group] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return simd_sum(lane < threads / 32u ? sums[lane] : 0.0f);
}

// Block (channel, token), 128 threads. FP32 register partials reduce through SMEM [4].
kernel void ple_norm(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant uint& num_active_tokens [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (group.y >= num_active_tokens) return;
    const ulong base = ulong(group.y) * wide + ulong(group.x) * hidden;
    float sum = 0.0f;
    for (ulong d = tid; d < hidden; d += threads) {
        const float value = float(input[base + d]);
        sum += value * value;
    }
    threadgroup float sums[threads / 32];
    const float inv_rms = precise::rsqrt(block_sum(sum, sums, lane, simd_group) / float(hidden) + eps);
    for (ulong d = tid; d < hidden; d += threads) {
        const float scale = 1.0f + float(weight[ulong(group.x) * hidden + d]);
        output[base + d] = bfloat(float(input[base + d]) * inv_rms * scale);
    }
}

// Block (channel, token). Q/K products reduce to one gate; V is shared across channels.
kernel void ple_gate(
    device const bfloat* key [[buffer(0)]],
    device const bfloat* query [[buffer(1)]],
    device const bfloat* value [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& num_active_tokens [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (group.y >= num_active_tokens) return;
    const ulong base = ulong(group.y) * wide + ulong(group.x) * hidden;
    float dot = 0.0f;
    for (ulong d = tid; d < hidden; d += threads) {
        dot += stored(float(key[base + d]) * float(query[base + d]));
    }
    threadgroup float sums[threads / 32];
    const float score = stored(stored(block_sum(dot, sums, lane, simd_group)) / stored(precise::sqrt(float(hidden))));
    const float root = stored(precise::sqrt(max(abs(score), 1e-6f)));
    const float gate = stored(1.0f / (1.0f + exp(-stored(sign(score) * root))));
    for (ulong d = tid; d < hidden; d += threads) {
        output[base + d] = bfloat(gate * float(value[ulong(group.y) * hidden + d]));
    }
}

// Thread (feature, token). Historical taps read state; current taps read normalized Z.
kernel void ple_conv(
    device const bfloat* normalized [[buffer(0)]],
    device const bfloat* gated [[buffer(1)]],
    device const bfloat* residual [[buffer(2)]],
    device const bfloat* weight [[buffer(3)]],
    device const PLEToken* tokens [[buffer(4)]],
    device const bfloat* states [[buffer(5)]],
    device bfloat* output [[buffer(6)]],
    constant uint& num_active_tokens [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint d = group.x * 256u + tid, token = group.y;
    if (token >= num_active_tokens || d >= wide) return;
    const PLEToken meta = tokens[token];
    const uint token_offset = token - meta.batch_start;
    float sum = 0.0f;
    for (uint j = 0; j < taps; ++j) {
        const uint lag = (taps - 1u - j) * dilation;
        const float value = lag <= token_offset
            ? float(normalized[ulong(token - lag) * wide + d])
            : float(states[(ulong(meta.src_state_slot) * tail + tail + token_offset - lag) * wide + d]);
        sum += value * float(weight[ulong(d) * taps + j]);
    }
    const float value = stored(sum);
    const float silu = stored(value * stored(1.0f / (1.0f + exp(-value))));
    const ulong at = ulong(token) * wide + d;
    output[at] = bfloat(float(residual[at]) + stored(float(gated[at]) + silu));
}

// One thread copies one accepted-prefix history value. Destinations and sources are disjoint.
kernel void ple_commit(
    device const bfloat* normalized [[buffer(0)]],
    device const PLERequest* requests [[buffer(1)]],
    device const uint* accepted [[buffer(2)]],
    device bfloat* states [[buffer(3)]],
    constant uint& num_active_requests [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint column = group.x * 256u + tid;
    if (group.y >= num_active_requests || ulong(column) >= ulong(tail) * wide) return;
    const PLERequest request = requests[group.y];
    const uint j = column / wide, d = column % wide;
    const ulong end = ulong(accepted[group.y]) + j;
    const bfloat value = end >= tail
        ? normalized[(ulong(request.query_start) + end - tail) * wide + d]
        : states[(ulong(request.src_state_slot) * tail + end) * wide + d];
    states[ulong(request.dst_state_slot) * tail * wide + column] = value;
}
