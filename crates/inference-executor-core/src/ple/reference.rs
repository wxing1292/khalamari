use half::bf16;

use super::core::PLECore;

pub struct Weights<'a> {
    pub key_norm: &'a [f32],
    pub query_norm: &'a [f32],
    pub conv_norm: &'a [f32],
    pub conv: &'a [f32],
}
pub struct Input<'a> {
    pub core: PLECore,
    pub num_tokens: usize,
    pub residual: &'a [f32],
    pub keys: &'a [f32],
    pub values: &'a [f32],
    pub state: &'a [f32],
    pub weights: Weights<'a>,
}
pub struct Output {
    pub residual: Vec<f32>,
    pub gated: Vec<f32>,
    pub normalized: Vec<f32>,
}

/// Eager BF16 reference for one request. It does not use GPU metadata or state-slot layouts.
/// Inputs and weights contain BF16-rounded values represented as FP32.
pub fn apply(input: Input<'_>) -> Output {
    let Input {
        core,
        num_tokens,
        residual,
        keys,
        values,
        state,
        weights,
    } = input;
    core.validate();
    let wide = core.residual_dim() as usize;
    let hidden = core.hidden_dim as usize;
    assert_eq!(residual.len(), num_tokens * wide);
    assert_eq!(keys.len(), residual.len());
    assert_eq!(values.len(), num_tokens * hidden);
    assert_eq!(state.len(), core.state_len() as usize * wide);
    assert_eq!(weights.key_norm.len(), wide);
    assert_eq!(weights.query_norm.len(), wide);
    assert_eq!(weights.conv_norm.len(), wide);
    assert_eq!(weights.conv.len(), wide * core.conv_kernel_size as usize);
    let mut gated = vec![0.0; residual.len()];
    let mut normalized = vec![0.0; residual.len()];
    for token in 0..num_tokens {
        for channel in 0..core.num_channels as usize {
            let start = token * wide + channel * hidden;
            let end = start + hidden;
            let k = norm(
                &keys[start..end],
                &weights.key_norm[channel * hidden..(channel + 1) * hidden],
                core.norm_eps,
            );
            let q = norm(
                &residual[start..end],
                &weights.query_norm[channel * hidden..(channel + 1) * hidden],
                core.norm_eps,
            );
            let dot = round(k.iter().zip(q).map(|(k, q)| round(k * q)).sum());
            let score = round(dot / round((hidden as f32).sqrt()));
            let root = round(score.abs().max(1e-6).sqrt());
            let signed_root = if score == 0.0 { 0.0 } else { score.signum() * root };
            let gate = round(sigmoid(round(signed_root)));
            for d in 0..hidden {
                gated[start + d] = round(gate * values[token * hidden + d]);
            }
            normalized[start..end].copy_from_slice(&norm(
                &gated[start..end],
                &weights.conv_norm[channel * hidden..(channel + 1) * hidden],
                core.norm_eps,
            ));
        }
    }
    let history: Vec<_> = state.iter().chain(&normalized).copied().collect();
    let mut output = vec![0.0; residual.len()];
    for token in 0..num_tokens {
        for d in 0..wide {
            let mut sum = 0.0;
            for j in 0..core.conv_kernel_size as usize {
                sum += history[(token + j * core.conv_dilation as usize) * wide + d]
                    * weights.conv[d * core.conv_kernel_size as usize + j];
            }
            let v = round(sum);
            let silu = round(v * round(sigmoid(v)));
            output[token * wide + d] = round(residual[token * wide + d] + round(gated[token * wide + d] + silu));
        }
    }
    Output {
        residual: output,
        gated,
        normalized,
    }
}

fn norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let inv = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps)
        .sqrt()
        .recip();
    x.iter()
        .zip(weight)
        .map(|(&x, &w)| round(x * inv * (1.0 + w)))
        .collect()
}
fn round(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
