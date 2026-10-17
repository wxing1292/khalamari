struct NGramToken { uint batch_start, history_slot; };

// One thread owns one (token, hash head). No hidden state or table data is read.
kernel void ngram_hash(
    device const uint* input [[buffer(0)]],
    device const uint* history [[buffer(1)]],
    device const NGramToken* tokens [[buffer(2)]],
    device const ulong* multipliers [[buffer(3)]],
    device const uint* sizes [[buffer(4)]],
    device const uint* offsets [[buffer(5)]],
    device uint* ids [[buffer(6)]],
    constant uint& num_active_tokens [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const ulong query = ulong(gid) / num_heads;
    if (query >= num_active_tokens) return;
    const uint head = gid % num_heads;
    const NGramToken token = tokens[query];
    const uint offset = uint(query) - token.batch_start;
    ulong mixed = 0;
    bool cut = false;
    for (uint lag = 0; lag < head / heads_per_ngram + 2u; ++lag) {
        const uint id = lag <= offset ? input[query - lag]
            : history[ulong(token.history_slot) * (ngram_size - 1u) + ngram_size - 1u + offset - lag];
        mixed ^= ulong(cut ? eos_token_id : id) * multipliers[lag];
        if (lag > 0 && id == eos_token_id) cut = true;
    }
    long row = as_type<long>(mixed) % long(sizes[head]);
    if (row < 0) row += long(sizes[head]);
    ids[ulong(query) * num_heads + head] = uint(row) + offsets[head];
}

kernel void ngram_lookup_bf16(
    device const uint* ids [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant uint& num_active_tokens [[buffer(3)]],
    constant uint& start [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (ulong(gid) >= ulong(num_active_tokens) * num_heads * head_dim) return;
    const uint id = ids[gid / head_dim];
    if (id < start || id - start >= count) return;
    output[gid] = bfloat(float(weight[ulong(id - start) * head_dim + gid % head_dim]) * scale);
}

kernel void ngram_lookup_affine(
    device const uint* ids [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant uint& num_active_tokens [[buffer(3)]],
    constant uint& start [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    device const bfloat* scales [[buffer(7)]],
    device const bfloat* biases [[buffer(8)]],
    constant uint& bits [[buffer(9)]],
    constant uint& group_size [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (ulong(gid) >= ulong(num_active_tokens) * num_heads * head_dim) return;
    const uint id = ids[gid / head_dim];
    if (id < start || id - start >= count) return;
    const uint d = gid % head_dim;
    const ulong row = ulong(id - start), affine = row * (head_dim / group_size) + d / group_size;
    const uint code = unpack_affine_value(weight, row * (ulong(head_dim) * bits / 8u), d, bits);
    const bfloat product = bfloat(float(scales[affine]) * float(code));
    const bfloat decoded = bfloat(float(product) + float(biases[affine]));
    output[gid] = bfloat(float(decoded) * scale);
}
