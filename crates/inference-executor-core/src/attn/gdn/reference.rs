//! CPU reference implementation for Gated DeltaNet primitive tests.

use crate::attn::gdn::GDNCore;
use crate::attn::gdn::GDNOutputGate;
use crate::reference::sigmoid_reference;
use crate::reference::silu_reference;
use crate::reference::softplus_reference;

#[derive(Clone, Debug, PartialEq)]
pub struct GDNShortConvReference {
    pub conv_qkv: Vec<f32>,
    pub next_conv_state: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GDNRecurrentReference {
    pub recurrent_output: Vec<f32>,
    pub next_recurrent_state: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct GDNRecurrentReferenceInput<'a> {
    pub cu_tokens: &'a [u32],
    pub source_recurrent_state: &'a [f32],
    pub conv_qkv: &'a [f32],
    pub a: &'a [f32],
    pub b: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
}

pub fn gdn_short_conv_reference(
    core: &GDNCore,
    cu_tokens: &[u32],
    source_conv_state: &[f32],
    qkv: &[f32],
    conv_weight: &[f32],
) -> GDNShortConvReference {
    core.validate();
    assert_eq!(cu_tokens.first(), Some(&0));
    assert!(cu_tokens.windows(2).all(|tokens| tokens[0] <= tokens[1]));
    let num_reqs = cu_tokens.len() - 1;
    let num_tokens = *cu_tokens.last().unwrap() as usize;
    let conv_state_len = core.conv_state_len();
    assert_eq!(source_conv_state.len(), num_reqs * core.qkv_dim() * conv_state_len);
    assert_eq!(qkv.len(), num_tokens * core.qkv_dim());
    assert_eq!(conv_weight.len(), core.qkv_dim() * core.conv_kernel_size);

    let mut conv_qkv = vec![0.0; num_tokens * core.qkv_dim()];
    let mut next_conv_state = vec![0.0; num_reqs * core.qkv_dim() * conv_state_len];
    for req_index in 0..num_reqs {
        let req_start = cu_tokens[req_index] as usize;
        let req_end = cu_tokens[req_index + 1] as usize;
        let num_req_tokens = req_end - req_start;
        for channel_index in 0..core.qkv_dim() {
            for token_index_in_req in 0..num_req_tokens {
                let mut convolution = 0.0;
                for kernel_index in 0..core.conv_kernel_size {
                    let sequence_index = token_index_in_req as isize + kernel_index as isize - conv_state_len as isize;
                    let value = if sequence_index < 0 {
                        let state_index = (sequence_index + conv_state_len as isize) as usize;
                        source_conv_state[(req_index * core.qkv_dim() + channel_index) * conv_state_len + state_index]
                    } else {
                        qkv[((req_start + sequence_index as usize) * core.qkv_dim()) + channel_index]
                    };
                    convolution += value * conv_weight[channel_index * core.conv_kernel_size + kernel_index];
                }
                conv_qkv[((req_start + token_index_in_req) * core.qkv_dim()) + channel_index] =
                    silu_reference(convolution);
            }

            for state_index in 0..conv_state_len {
                let sequence_index = num_req_tokens as isize + state_index as isize - conv_state_len as isize;
                let value = if sequence_index < 0 {
                    let source_state_index = state_index + num_req_tokens;
                    source_conv_state
                        [(req_index * core.qkv_dim() + channel_index) * conv_state_len + source_state_index]
                } else {
                    qkv[((req_start + sequence_index as usize) * core.qkv_dim()) + channel_index]
                };
                next_conv_state[(req_index * core.qkv_dim() + channel_index) * conv_state_len + state_index] = value;
            }
        }
    }

    GDNShortConvReference {
        conv_qkv,
        next_conv_state,
    }
}

pub fn gdn_recurrent_reference(core: &GDNCore, input: GDNRecurrentReferenceInput<'_>) -> GDNRecurrentReference {
    core.validate();
    let cu_tokens = input.cu_tokens;
    assert_eq!(cu_tokens.first(), Some(&0));
    assert!(cu_tokens.windows(2).all(|tokens| tokens[0] <= tokens[1]));
    let num_reqs = cu_tokens.len() - 1;
    let num_tokens = *cu_tokens.last().unwrap() as usize;
    let recurrent_state_stride = core.num_v_heads * core.v_head_dim * core.qk_head_dim;
    assert_eq!(input.source_recurrent_state.len(), num_reqs * recurrent_state_stride);
    assert_eq!(input.conv_qkv.len(), num_tokens * core.qkv_dim());
    assert_eq!(input.a.len(), num_tokens * core.num_v_heads);
    assert_eq!(input.b.len(), num_tokens * core.num_v_heads);
    assert_eq!(input.a_log.len(), core.num_v_heads);
    assert_eq!(input.dt_bias.len(), core.num_v_heads);

    let mut recurrent_output = vec![0.0; num_tokens * core.v_dim()];
    let mut next_recurrent_state = input.source_recurrent_state.to_vec();
    let num_v_heads_per_qk_head = core.num_v_heads / core.num_qk_heads;
    let k_base = core.qk_dim();
    let v_base = k_base + core.qk_dim();

    for req_index in 0..num_reqs {
        for token_index in cu_tokens[req_index] as usize..cu_tokens[req_index + 1] as usize {
            for v_head_index in 0..core.num_v_heads {
                let qk_head_index = v_head_index / num_v_heads_per_qk_head;
                let query_inverse_norm = inverse_l2_norm(
                    input.conv_qkv,
                    token_index,
                    core.qkv_dim(),
                    qk_head_index,
                    core.qk_head_dim,
                    0,
                ) * core.q_scale;
                let key_inverse_norm = inverse_l2_norm(
                    input.conv_qkv,
                    token_index,
                    core.qkv_dim(),
                    qk_head_index,
                    core.qk_head_dim,
                    k_base,
                );
                let gate_offset = token_index * core.num_v_heads + v_head_index;
                let beta = sigmoid_reference(input.b[gate_offset]);
                let time_step = input.a[gate_offset] + input.dt_bias[v_head_index];
                let decay_rate = -input.a_log[v_head_index].exp();
                let decay = (decay_rate * softplus_reference(time_step)).exp();
                for value_dim_index in 0..core.v_head_dim {
                    let value = input.conv_qkv
                        [token_index * core.qkv_dim() + v_base + v_head_index * core.v_head_dim + value_dim_index];
                    let state_base = req_index * recurrent_state_stride
                        + (v_head_index * core.v_head_dim + value_dim_index) * core.qk_head_dim;
                    let mut state_key_dot_product = 0.0;
                    for qk_dim_index in 0..core.qk_head_dim {
                        let normalized_key = input.conv_qkv
                            [token_index * core.qkv_dim() + k_base + qk_head_index * core.qk_head_dim + qk_dim_index]
                            * key_inverse_norm;
                        let state_index = state_base + qk_dim_index;
                        let decayed_state = next_recurrent_state[state_index] * decay;
                        next_recurrent_state[state_index] = decayed_state;
                        state_key_dot_product += decayed_state * normalized_key;
                    }
                    let delta = (value - state_key_dot_product) * beta;
                    let mut output_value = 0.0;
                    for qk_dim_index in 0..core.qk_head_dim {
                        let normalized_query = input.conv_qkv
                            [token_index * core.qkv_dim() + qk_head_index * core.qk_head_dim + qk_dim_index]
                            * query_inverse_norm;
                        let normalized_key = input.conv_qkv
                            [token_index * core.qkv_dim() + k_base + qk_head_index * core.qk_head_dim + qk_dim_index]
                            * key_inverse_norm;
                        let state_index = state_base + qk_dim_index;
                        next_recurrent_state[state_index] += normalized_key * delta;
                        output_value += next_recurrent_state[state_index] * normalized_query;
                    }
                    recurrent_output[token_index * core.v_dim() + v_head_index * core.v_head_dim + value_dim_index] =
                        output_value;
                }
            }
        }
    }

    GDNRecurrentReference {
        recurrent_output,
        next_recurrent_state,
    }
}

pub fn gdn_output_norm_gate_reference(
    core: &GDNCore,
    recurrent_output: &[f32],
    z: &[f32],
    norm_weight: &[f32],
    norm_eps: f32,
) -> Vec<f32> {
    core.validate();
    assert!(norm_eps.is_finite() && norm_eps > 0.0);
    assert_eq!(recurrent_output.len(), z.len());
    assert_eq!(recurrent_output.len() % core.v_head_dim, 0);
    assert_eq!(norm_weight.len(), core.v_head_dim);

    let mut output = Vec::with_capacity(recurrent_output.len());
    for (recurrent_row, z_row) in recurrent_output
        .chunks_exact(core.v_head_dim)
        .zip(z.chunks_exact(core.v_head_dim))
    {
        let inv_rms = (recurrent_row.iter().map(|value| value * value).sum::<f32>() / core.v_head_dim as f32
            + norm_eps)
            .sqrt()
            .recip();
        output.extend(
            recurrent_row
                .iter()
                .zip(z_row)
                .zip(norm_weight)
                .map(|((&value, &gate), &weight)| {
                    let gate = match core.output_gate {
                        GDNOutputGate::Silu => silu_reference(gate),
                        GDNOutputGate::Sigmoid => sigmoid_reference(gate),
                    };
                    value * inv_rms * weight * gate
                }),
        );
    }
    output
}

fn inverse_l2_norm(
    conv_qkv: &[f32],
    token_index: usize,
    qkv_dim: usize,
    qk_head_index: usize,
    qk_head_dim: usize,
    vector_offset: usize,
) -> f32 {
    let mut sum_squares = 0.0;
    for qk_dim_index in 0..qk_head_dim {
        let value = conv_qkv[token_index * qkv_dim + vector_offset + qk_head_index * qk_head_dim + qk_dim_index];
        sum_squares += value * value;
    }
    (sum_squares + 1.0e-6).sqrt().recip()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn core() -> GDNCore {
        GDNCore {
            model_layer_index: 0,
            hidden_dim: 4,
            num_qk_heads: 1,
            qk_head_dim: 2,
            num_v_heads: 1,
            v_head_dim: 2,
            conv_kernel_size: 3,
            q_scale: 1.0,
            output_gate: GDNOutputGate::Silu,
        }
    }

    #[test]
    fn test_output_gate_activation_after_norm() {
        let mut core = core();
        let recurrent = [1.0; 6];
        let z = [-2.0, -1.0, 0.0, 0.0, 1.0, 2.0];
        let inv_rms = (1.0_f32 + 1.0e-6).sqrt().recip();
        for output_gate in [GDNOutputGate::Silu, GDNOutputGate::Sigmoid] {
            core.output_gate = output_gate;
            let actual = gdn_output_norm_gate_reference(&core, &recurrent, &z, &[1.0; 2], 1.0e-6);
            for (&value, &z) in actual.iter().zip(&z) {
                let numerator = match output_gate {
                    GDNOutputGate::Silu => z,
                    GDNOutputGate::Sigmoid => 1.0,
                };
                let expected = inv_rms * numerator / (1.0 + (-z).exp());
                assert!((value - expected).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn test_short_convolution() {
        let core = core();
        let output = gdn_short_conv_reference(&core, &[0, 1], &[1.0; 12], &[0.5; 6], &[1.0; 18]);

        assert_eq!(output.conv_qkv, vec![silu_reference(2.5); 6]);
        assert_eq!(output.next_conv_state, [1.0, 0.5].repeat(6));
    }

    #[test]
    fn test_recurrent_update() {
        let core = core();
        let output = gdn_recurrent_reference(
            &core,
            GDNRecurrentReferenceInput {
                cu_tokens: &[0, 1],
                source_recurrent_state: &[0.0; 4],
                conv_qkv: &[0.0, 1.0, 0.0, 1.0, 2.0, -1.0],
                a: &[0.0],
                b: &[0.0],
                a_log: &[0.0],
                dt_bias: &[0.0],
            },
        );

        let inverse_norm = (1.0_f32 + 1.0e-6).sqrt().recip();
        let normalized_dot_product = inverse_norm * inverse_norm;
        let expected_output = [normalized_dot_product, -0.5 * normalized_dot_product];
        let expected_state = [0.0, inverse_norm, 0.0, -0.5 * inverse_norm];
        for (&actual, expected) in output.recurrent_output.iter().zip(expected_output) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        for (&actual, expected) in output.next_recurrent_state.iter().zip(expected_state) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }
}
