
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

static inline float read_bf16(device const bfloat16_t* values, uint index) {
    return float(values[index]);
}

static inline void write_bf16(device bfloat16_t* values, uint index, float value) {
    values[index] = bfloat16_t(value);
}

static inline float combine_topk(
    device const bfloat16_t* routed_hidden,
    device const float* routed_probs,
    uint token,
    uint dim,
    uint num_experts_per_token,
    uint hidden_dim
) {
    float acc = 0.0f;
    for (uint slot = 0; slot < num_experts_per_token; ++slot) {
        const uint route = token * num_experts_per_token + slot;
        const float route_weight = float(bfloat16_t(routed_probs[route]));
        acc = fma(route_weight, read_bf16(routed_hidden, route * hidden_dim + dim), acc);
    }
    return acc;
}

kernel void moe_combine_without_shared_experts(
    device const bfloat16_t* routed_hidden [[buffer(0)]],
    device const float* routed_probs [[buffer(1)]],
    device bfloat16_t* output [[buffer(2)]],
    constant uint& num_active_tokens [[buffer(3)]],
    constant uint& num_experts_per_token [[buffer(4)]],
    constant uint& hidden_dim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_active_tokens * hidden_dim) return;
    const uint token = gid / hidden_dim;
    const uint dim = gid - token * hidden_dim;
    write_bf16(output, gid, combine_topk(routed_hidden, routed_probs, token, dim, num_experts_per_token, hidden_dim));
}

kernel void moe_combine_with_shared_experts(
    device const bfloat16_t* routed_hidden [[buffer(0)]],
    device const float* routed_probs [[buffer(1)]],
    device const bfloat16_t* shared_hidden [[buffer(2)]],
    device const bfloat16_t* shared_expert_gate_logits [[buffer(3)]],
    device bfloat16_t* output [[buffer(4)]],
    constant uint& num_active_tokens [[buffer(5)]],
    constant uint& num_experts_per_token [[buffer(6)]],
    constant uint& hidden_dim [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_active_tokens * hidden_dim) return;
    const uint token = gid / hidden_dim;
    const uint dim = gid - token * hidden_dim;

    const float routed_output = float(bfloat16_t(combine_topk(
        routed_hidden,
        routed_probs,
        token,
        dim,
        num_experts_per_token,
        hidden_dim
    )));
    const float shared_expert_gate = 1.0f / (1.0f + metal::exp(-read_bf16(shared_expert_gate_logits, token)));
    const float value = routed_output + shared_expert_gate * read_bf16(shared_hidden, gid);
    write_bf16(output, gid, value);
}
