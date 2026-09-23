//! CPU reference implementation for MoE routing, sparse expert MLP, and combine tests.

use half::bf16;

use crate::mlp::dense::reference::QuantizedAffineReferenceShape;
use crate::mlp::dense::reference::quantized_affine_reference;
use crate::reference::sigmoid_reference;
use crate::reference::silu_reference;
use crate::reference::softmax_reference;

#[derive(Clone, Debug, PartialEq)]
pub struct MoERoutingReference {
    pub expert_indices: Vec<u32>,
    pub expert_probs: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct QuantizedSparseMLPReferenceWeights<'a> {
    pub gate_weight: &'a [u8],
    pub gate_scales: &'a [f32],
    pub gate_biases: &'a [f32],
    pub up_weight: &'a [u8],
    pub up_scales: &'a [f32],
    pub up_biases: &'a [f32],
    pub down_weight: &'a [u8],
    pub down_scales: &'a [f32],
    pub down_biases: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
pub struct QuantizedSparseMLPReferenceInput<'a> {
    pub hidden: &'a [f32],
    pub token_indices: &'a [u32],
    pub expert_indices: &'a [u32],
    pub swiglu_indices: &'a [u32],
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub group_size: usize,
    pub bits: usize,
    pub num_experts: usize,
    pub weights: QuantizedSparseMLPReferenceWeights<'a>,
}

pub fn moe_routing_reference(
    router_logits: &[f32],
    num_tokens: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    norm_topk_prob: bool,
) -> MoERoutingReference {
    assert_eq!(router_logits.len(), num_tokens * num_experts);
    let mut router_probs = Vec::with_capacity(router_logits.len());
    for token_index in 0..num_tokens {
        router_probs.extend(softmax_reference(
            &router_logits[token_index * num_experts..(token_index + 1) * num_experts],
        ));
    }
    moe_routing_from_probs_reference(
        &router_probs,
        num_tokens,
        num_experts,
        num_experts_per_token,
        norm_topk_prob,
    )
}

pub fn moe_routing_from_probs_reference(
    router_probs: &[f32],
    num_tokens: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    norm_topk_prob: bool,
) -> MoERoutingReference {
    assert_eq!(router_probs.len(), num_tokens * num_experts);
    assert!(num_experts_per_token > 0);
    assert!(num_experts_per_token <= num_experts);
    assert!(u32::try_from(num_experts).is_ok());

    let mut expert_indices = Vec::with_capacity(num_tokens * num_experts_per_token);
    let mut expert_probs = Vec::with_capacity(num_tokens * num_experts_per_token);
    for token_index in 0..num_tokens {
        let mut ranked = router_probs[token_index * num_experts..(token_index + 1) * num_experts]
            .iter()
            .enumerate()
            .map(|(expert_index, &probability)| (expert_index as u32, probability))
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap().then_with(|| left.0.cmp(&right.0)));
        ranked.truncate(num_experts_per_token);
        let selected_total = ranked.iter().map(|(_, probability)| *probability).sum::<f32>();
        for (expert_index, probability) in ranked {
            expert_indices.push(expert_index);
            expert_probs.push(if norm_topk_prob {
                probability / selected_total
            } else {
                probability
            });
        }
    }
    MoERoutingReference {
        expert_indices,
        expert_probs,
    }
}

pub fn moe_routing_from_bf16_probs_reference(
    router_probs: &[f32],
    num_tokens: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    norm_topk_prob: bool,
) -> MoERoutingReference {
    assert_eq!(router_probs.len(), num_tokens * num_experts);
    assert!(num_experts_per_token > 0);
    assert!(num_experts_per_token <= num_experts);
    assert!(u32::try_from(num_experts).is_ok());

    let router_probs = router_probs
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect::<Vec<_>>();
    let mut expert_indices = Vec::with_capacity(num_tokens * num_experts_per_token);
    let mut expert_probs = Vec::with_capacity(num_tokens * num_experts_per_token);
    for token_index in 0..num_tokens {
        let mut selected = Vec::<(u32, f32)>::with_capacity(num_experts_per_token);
        for _ in 0..num_experts_per_token {
            let mut best = (u32::MAX, -1.0f32);
            for expert_index in 0..num_experts {
                if selected
                    .iter()
                    .any(|(selected_expert, _)| *selected_expert == expert_index as u32)
                {
                    continue;
                }
                let probability = router_probs[token_index * num_experts + expert_index];
                if probability > best.1 || (probability == best.1 && (expert_index as u32) < best.0) {
                    best = (expert_index as u32, probability);
                }
            }
            selected.push(best);
        }

        let mut topk_sum = 0.0f32;
        for (_, probability) in &selected {
            topk_sum = bf16::from_f32(topk_sum + *probability).to_f32();
        }
        for (expert_index, mut probability) in selected {
            if norm_topk_prob && num_experts_per_token > 1 && topk_sum > 0.0 {
                probability = bf16::from_f32(probability / topk_sum).to_f32();
            }
            expert_indices.push(expert_index);
            expert_probs.push(bf16::from_f32(probability).to_f32());
        }
    }
    MoERoutingReference {
        expert_indices,
        expert_probs,
    }
}

pub fn quantized_sparse_mlp_reference(input: QuantizedSparseMLPReferenceInput<'_>) -> Vec<f32> {
    let QuantizedSparseMLPReferenceInput {
        hidden,
        token_indices,
        expert_indices,
        swiglu_indices,
        hidden_dim,
        intermediate_dim,
        group_size,
        bits,
        num_experts,
        weights,
    } = input;
    assert_eq!(token_indices.len(), expert_indices.len());
    assert_eq!(token_indices.len(), swiglu_indices.len());
    assert!(num_experts > 0);
    let input_projection_shape = QuantizedAffineReferenceShape {
        num_rows: 1,
        output_dim: intermediate_dim,
        input_dim: hidden_dim,
        group_size,
        bits,
    };
    let output_projection_shape = QuantizedAffineReferenceShape {
        num_rows: 1,
        output_dim: hidden_dim,
        input_dim: intermediate_dim,
        group_size,
        bits,
    };
    input_projection_shape.validate();
    output_projection_shape.validate();
    assert_eq!(hidden.len() % hidden_dim, 0);
    let num_input_rows = hidden.len() / hidden_dim;
    let num_routes = token_indices.len();
    let input_projection_weight_bytes = input_projection_shape.weight_bytes();
    let input_projection_param_count = input_projection_shape.affine_param_len();
    let output_projection_weight_bytes = output_projection_shape.weight_bytes();
    let output_projection_param_count = output_projection_shape.affine_param_len();
    assert_eq!(weights.gate_weight.len(), num_experts * input_projection_weight_bytes);
    assert_eq!(weights.up_weight.len(), num_experts * input_projection_weight_bytes);
    assert_eq!(weights.down_weight.len(), num_experts * output_projection_weight_bytes);
    assert_eq!(weights.gate_scales.len(), num_experts * input_projection_param_count);
    assert_eq!(weights.gate_biases.len(), num_experts * input_projection_param_count);
    assert_eq!(weights.up_scales.len(), num_experts * input_projection_param_count);
    assert_eq!(weights.up_biases.len(), num_experts * input_projection_param_count);
    assert_eq!(weights.down_scales.len(), num_experts * output_projection_param_count);
    assert_eq!(weights.down_biases.len(), num_experts * output_projection_param_count);

    let mut swiglu_by_route = vec![0.0_f32; num_routes * intermediate_dim];
    for route_index in 0..num_routes {
        let token_index = token_indices[route_index] as usize;
        let expert_index = expert_indices[route_index] as usize;
        assert!(token_index < num_input_rows);
        assert!(expert_index < num_experts);

        let input_row = &hidden[token_index * hidden_dim..(token_index + 1) * hidden_dim];
        let gate = quantized_affine_reference(
            input_projection_shape,
            input_row,
            expert_slice(weights.gate_weight, expert_index, input_projection_weight_bytes),
            expert_slice(weights.gate_scales, expert_index, input_projection_param_count),
            expert_slice(weights.gate_biases, expert_index, input_projection_param_count),
        );
        let up = quantized_affine_reference(
            input_projection_shape,
            input_row,
            expert_slice(weights.up_weight, expert_index, input_projection_weight_bytes),
            expert_slice(weights.up_scales, expert_index, input_projection_param_count),
            expert_slice(weights.up_biases, expert_index, input_projection_param_count),
        );
        let activated_hidden = gate
            .iter()
            .zip(up.iter())
            .map(|(&gate_value, &up_value)| silu_reference(gate_value) * up_value)
            .collect::<Vec<_>>();
        swiglu_by_route[route_index * intermediate_dim..(route_index + 1) * intermediate_dim]
            .copy_from_slice(&activated_hidden);
    }

    let mut output = vec![0.0_f32; num_routes * hidden_dim];
    for route_index in 0..num_routes {
        let swiglu_index = swiglu_indices[route_index] as usize;
        let expert_index = expert_indices[route_index] as usize;
        assert!(swiglu_index < num_routes);
        assert!(expert_index < num_experts);
        let route_output = quantized_affine_reference(
            output_projection_shape,
            &swiglu_by_route[swiglu_index * intermediate_dim..(swiglu_index + 1) * intermediate_dim],
            expert_slice(weights.down_weight, expert_index, output_projection_weight_bytes),
            expert_slice(weights.down_scales, expert_index, output_projection_param_count),
            expert_slice(weights.down_biases, expert_index, output_projection_param_count),
        );
        output[route_index * hidden_dim..(route_index + 1) * hidden_dim].copy_from_slice(&route_output);
    }
    output
}

pub fn moe_combine_without_shared_experts_reference(
    routed_hidden: &[f32],
    routed_probs: &[f32],
    num_tokens: usize,
    num_experts_per_token: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    assert_eq!(routed_hidden.len(), num_tokens * num_experts_per_token * hidden_dim);
    assert_eq!(routed_probs.len(), num_tokens * num_experts_per_token);
    let mut output = vec![0.0; num_tokens * hidden_dim];
    for token_index in 0..num_tokens {
        for route_offset in 0..num_experts_per_token {
            let route_index = token_index * num_experts_per_token + route_offset;
            let route_weight = routed_probs[route_index];
            for hidden_index in 0..hidden_dim {
                output[token_index * hidden_dim + hidden_index] +=
                    route_weight * routed_hidden[route_index * hidden_dim + hidden_index];
            }
        }
    }
    output
}

pub fn moe_combine_with_shared_experts_reference(
    routed_output: &[f32],
    shared_hidden: &[f32],
    shared_expert_gate_logits: &[f32],
    num_tokens: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    assert_eq!(routed_output.len(), num_tokens * hidden_dim);
    assert_eq!(shared_hidden.len(), num_tokens * hidden_dim);
    assert_eq!(shared_expert_gate_logits.len(), num_tokens);
    let mut output = routed_output.to_vec();
    for token_index in 0..num_tokens {
        let gate = sigmoid_reference(shared_expert_gate_logits[token_index]);
        for hidden_index in 0..hidden_dim {
            output[token_index * hidden_dim + hidden_index] +=
                gate * shared_hidden[token_index * hidden_dim + hidden_index];
        }
    }
    output
}

pub fn moe_combine_without_shared_experts_bf16_reference(
    routed_hidden: &[f32],
    routed_probs: &[f32],
    num_tokens: usize,
    num_experts_per_token: usize,
    hidden_dim: usize,
) -> Vec<u16> {
    assert_eq!(routed_hidden.len(), num_tokens * num_experts_per_token * hidden_dim);
    assert_eq!(routed_probs.len(), num_tokens * num_experts_per_token);
    let mut output = Vec::with_capacity(num_tokens * hidden_dim);
    for token_index in 0..num_tokens {
        for hidden_index in 0..hidden_dim {
            let mut combined = 0.0f32;
            for route_offset in 0..num_experts_per_token {
                let route_index = token_index * num_experts_per_token + route_offset;
                let route_weight = bf16::from_f32(routed_probs[route_index]).to_f32();
                let hidden = bf16::from_f32(routed_hidden[route_index * hidden_dim + hidden_index]).to_f32();
                let weighted = bf16::from_f32(route_weight * hidden).to_f32();
                combined = bf16::from_f32(combined + weighted).to_f32();
            }
            output.push(bf16::from_f32(combined).to_bits());
        }
    }
    output
}

pub fn moe_combine_with_shared_experts_bf16_reference(
    routed_output: &[u16],
    shared_hidden: &[f32],
    shared_expert_gate_logits: &[f32],
    num_tokens: usize,
    hidden_dim: usize,
) -> Vec<u16> {
    assert_eq!(routed_output.len(), num_tokens * hidden_dim);
    assert_eq!(shared_hidden.len(), num_tokens * hidden_dim);
    assert_eq!(shared_expert_gate_logits.len(), num_tokens);
    let mut output = Vec::with_capacity(num_tokens * hidden_dim);
    for (token_index, &shared_expert_gate_logit) in shared_expert_gate_logits.iter().enumerate().take(num_tokens) {
        let gate_logit = bf16::from_f32(shared_expert_gate_logit).to_f32();
        let gate = sigmoid_reference(gate_logit);
        for hidden_index in 0..hidden_dim {
            let hidden_offset = token_index * hidden_dim + hidden_index;
            let routed = bf16::from_bits(routed_output[hidden_offset]).to_f32();
            let shared = bf16::from_f32(shared_hidden[hidden_offset]).to_f32();
            output.push(bf16::from_f32(routed + gate * shared).to_bits());
        }
    }
    output
}

fn expert_slice<T>(values: &[T], expert_index: usize, values_per_expert: usize) -> &[T] {
    &values[expert_index * values_per_expert..(expert_index + 1) * values_per_expert]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_routing() {
        let routed = moe_routing_reference(&[0.25, 2.0, -1.0, 1.0, 3.0, 3.0, 0.5, -2.0], 2, 4, 2, true);

        assert_eq!(routed.expert_indices, vec![1, 3, 0, 1]);
        assert!((routed.expert_probs[0] + routed.expert_probs[1] - 1.0).abs() < 1.0e-6);
        assert!((routed.expert_probs[2] + routed.expert_probs[3] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn test_quantized_sparse_mlp() {
        const DIM: usize = 32;
        const NUM_EXPERTS: usize = 2;
        let input_projection_shape = QuantizedAffineReferenceShape {
            num_rows: 1,
            output_dim: DIM,
            input_dim: DIM,
            group_size: DIM,
            bits: 8,
        };
        let output_projection_shape = input_projection_shape;
        let input_projection_params = input_projection_shape.affine_param_len();
        let output_projection_params = output_projection_shape.affine_param_len();
        let mut hidden = vec![0.0; DIM * 2];
        hidden[0] = 1.0;
        hidden[DIM] = 2.0;
        let mut gate_biases = vec![1.0; NUM_EXPERTS * input_projection_params];
        gate_biases[input_projection_params..].fill(2.0);
        let mut up_biases = vec![2.0; NUM_EXPERTS * input_projection_params];
        up_biases[input_projection_params..].fill(3.0);
        let mut down_biases = vec![1.0 / DIM as f32; NUM_EXPERTS * output_projection_params];
        down_biases[output_projection_params..].fill(2.0 / DIM as f32);
        let input_weight_bytes = NUM_EXPERTS * input_projection_shape.weight_bytes();
        let output_weight_bytes = NUM_EXPERTS * output_projection_shape.weight_bytes();
        let input_scales = vec![0.0; NUM_EXPERTS * input_projection_params];
        let output_scales = vec![0.0; NUM_EXPERTS * output_projection_params];
        let gate_weight = vec![0; input_weight_bytes];
        let up_weight = vec![0; input_weight_bytes];
        let down_weight = vec![0; output_weight_bytes];

        let output = quantized_sparse_mlp_reference(QuantizedSparseMLPReferenceInput {
            hidden: &hidden,
            token_indices: &[0, 1, 0],
            expert_indices: &[0, 0, 1],
            swiglu_indices: &[1, 0, 2],
            hidden_dim: DIM,
            intermediate_dim: DIM,
            group_size: DIM,
            bits: 8,
            num_experts: NUM_EXPERTS,
            weights: QuantizedSparseMLPReferenceWeights {
                gate_weight: &gate_weight,
                gate_scales: &input_scales,
                gate_biases: &gate_biases,
                up_weight: &up_weight,
                up_scales: &input_scales,
                up_biases: &up_biases,
                down_weight: &down_weight,
                down_scales: &output_scales,
                down_biases: &down_biases,
            },
        });

        let expected_by_route = [
            4.0 * silu_reference(2.0),
            2.0 * silu_reference(1.0),
            6.0 * silu_reference(2.0),
        ];
        for (route_index, &expected) in expected_by_route.iter().enumerate() {
            assert!(
                output[route_index * DIM..(route_index + 1) * DIM]
                    .iter()
                    .all(|&value| (value - expected).abs() < 1.0e-5)
            );
        }
    }

    #[test]
    fn test_moe_combine() {
        let routed = moe_combine_without_shared_experts_reference(&[1.0, 2.0, 3.0, 4.0], &[0.25, 0.75], 1, 2, 2);
        assert_eq!(routed, vec![2.5, 3.5]);

        let output = moe_combine_with_shared_experts_reference(&routed, &[10.0, -2.0], &[0.0], 1, 2);
        assert_eq!(output, vec![7.5, 2.5]);
    }
}
