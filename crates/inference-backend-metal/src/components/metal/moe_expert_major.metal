
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

static inline float read_bf16(device const bfloat16_t* values, uint index) {
    return float(values[index]);
}

static inline void write_bf16(device bfloat16_t* values, uint index, float value) {
    values[index] = bfloat16_t(value);
}

kernel void moe_expert_major_layout_clear(
    device uint* expert_counts [[buffer(0)]],
    device uint* expert_cursors [[buffer(1)]],
    constant uint& num_experts [[buffer(2)]],
    uint expert [[thread_position_in_grid]]
) {
    if (expert >= num_experts) return;
    expert_counts[expert] = 0;
    expert_cursors[expert] = 0;
}

kernel void moe_expert_major_layout_count(
    device const uint* expert_indices [[buffer(0)]],
    device atomic_uint* expert_counts [[buffer(1)]],
    constant uint& num_active_tokens [[buffer(2)]],
    constant uint& num_experts_per_token [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    uint route [[thread_position_in_grid]]
) {
    const uint num_active_routes = num_active_tokens * num_experts_per_token;
    if (route >= num_active_routes) return;
    const uint expert = expert_indices[route];
    if (expert >= num_experts) return;
    atomic_fetch_add_explicit(expert_counts + expert, 1, memory_order_relaxed);
}

kernel void moe_expert_major_layout_prefix(
    device const uint* expert_counts [[buffer(0)]],
    device uint* expert_offsets [[buffer(1)]],
    device uint* expert_cursors [[buffer(2)]],
    constant uint& num_experts [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    uint offset = 0;
    for (uint expert = 0; expert < num_experts; ++expert) {
        expert_offsets[expert] = offset;
        expert_cursors[expert] = offset;
        const uint count = expert_counts[expert];
        offset += count;
    }
    expert_offsets[num_experts] = offset;
}

kernel void moe_expert_major_layout_scatter(
    device const uint* expert_indices [[buffer(0)]],
    device atomic_uint* expert_cursors [[buffer(1)]],
    device uint* routes_by_expert [[buffer(2)]],
    device uint* routes_by_token [[buffer(3)]],
    device uint* experts_by_route [[buffer(4)]],
    constant uint& num_active_tokens [[buffer(5)]],
    constant uint& num_experts_per_token [[buffer(6)]],
    constant uint& num_experts [[buffer(7)]],
    uint route [[thread_position_in_grid]]
) {
    const uint num_active_routes = num_active_tokens * num_experts_per_token;
    if (route >= num_active_routes) return;
    const uint expert = expert_indices[route];
    if (expert >= num_experts) return;
    const uint expert_route = atomic_fetch_add_explicit(expert_cursors + expert, 1, memory_order_relaxed);
    routes_by_expert[expert_route] = route;
    routes_by_token[route] = expert_route;
    experts_by_route[expert_route] = expert;
}

kernel void moe_expert_major_pack_input(
    device const bfloat16_t* input [[buffer(0)]],
    device const uint* routes_by_expert [[buffer(1)]],
    device bfloat16_t* packed_input [[buffer(2)]],
    constant uint& num_active_tokens [[buffer(3)]],
    constant uint& num_experts_per_token [[buffer(4)]],
    constant uint& hidden_dim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint route = gid / hidden_dim;
    const uint num_active_routes = num_active_tokens * num_experts_per_token;
    if (route >= num_active_routes) return;
    const uint dim = gid - route * hidden_dim;
    const uint original_route = routes_by_expert[route];
    const uint token = original_route / num_experts_per_token;
    packed_input[gid] = input[token * hidden_dim + dim];
}

static inline float scatter_topk(
    device const bfloat16_t* packed_output,
    device const uint* routes_by_token,
    device const float* routed_probs,
    uint token,
    uint dim,
    uint num_experts_per_token,
    uint hidden_dim
) {
    float acc = 0.0f;
    for (uint slot = 0; slot < num_experts_per_token; ++slot) {
        const uint original_route = token * num_experts_per_token + slot;
        const uint packed_route = routes_by_token[original_route];
        const float route_weight = float(bfloat16_t(routed_probs[original_route]));
        acc = fma(route_weight, read_bf16(packed_output, packed_route * hidden_dim + dim), acc);
    }
    return acc;
}

kernel void moe_expert_major_scatter_without_shared_experts(
    device const bfloat16_t* packed_output [[buffer(0)]],
    device const uint* routes_by_token [[buffer(1)]],
    device const float* routed_probs [[buffer(2)]],
    device bfloat16_t* output [[buffer(3)]],
    constant uint& num_active_tokens [[buffer(4)]],
    constant uint& num_experts_per_token [[buffer(5)]],
    constant uint& hidden_dim [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_active_tokens * hidden_dim) return;
    const uint token = gid / hidden_dim;
    const uint dim = gid - token * hidden_dim;
    write_bf16(output, gid, scatter_topk(packed_output, routes_by_token, routed_probs, token, dim, num_experts_per_token, hidden_dim));
}

kernel void moe_expert_major_scatter_with_shared_experts(
    device const bfloat16_t* packed_output [[buffer(0)]],
    device const uint* routes_by_token [[buffer(1)]],
    device const float* routed_probs [[buffer(2)]],
    device const bfloat16_t* shared_hidden [[buffer(3)]],
    device const bfloat16_t* shared_expert_gate_logits [[buffer(4)]],
    device bfloat16_t* output [[buffer(5)]],
    constant uint& num_active_tokens [[buffer(6)]],
    constant uint& num_experts_per_token [[buffer(7)]],
    constant uint& hidden_dim [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_active_tokens * hidden_dim) return;
    const uint token = gid / hidden_dim;
    const uint dim = gid - token * hidden_dim;
    const float routed_output = float(bfloat16_t(scatter_topk(
        packed_output,
        routes_by_token,
        routed_probs,
        token,
        dim,
        num_experts_per_token,
        hidden_dim
    )));
    const float shared_expert_gate = 1.0f / (1.0f + metal::exp(-read_bf16(shared_expert_gate_logits, token)));
    write_bf16(output, gid, routed_output + shared_expert_gate * read_bf16(shared_hidden, gid));
}
