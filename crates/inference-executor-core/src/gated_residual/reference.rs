use half::bf16;

use crate::gated_residual::core::GatedResidualCore;

/// Rounds at the tensor-operation boundaries of the eager reference.
#[derive(Clone, Copy)]
pub enum Precision {
    F32,
    Bf16,
}

impl Precision {
    fn round(self, value: f32) -> f32 {
        match self {
            Self::F32 => value,
            Self::Bf16 => bf16::from_f32(value).to_f32(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct ReadWeights<'a> {
    /// Checkpoint Gemma RMSNorm offsets: the affine multiplier is `1 + norm`.
    pub norm: &'a [f32],
    /// Row-major `[low_rank_dim, num_branches * hidden_dim]`.
    pub down: &'a [f32],
    /// Row-major `[num_branches * hidden_dim, low_rank_dim]`.
    pub up: &'a [f32],
}

pub struct ReadInput<'a> {
    pub core: GatedResidualCore,
    pub precision: Precision,
    pub num_tokens: usize,
    pub residual: &'a [f32],
    pub weights: ReadWeights<'a>,
}

pub struct ReadOutput {
    pub hidden_state: Vec<f32>,
    pub normalized_residual: Vec<f32>,
}

pub fn read(input: ReadInput<'_>) -> ReadOutput {
    let ReadInput {
        core,
        precision,
        num_tokens,
        residual,
        weights,
    } = input;
    core.validate();
    let wide = core.residual_dim();
    assert_eq!(residual.len(), num_tokens * wide);
    assert_eq!(weights.norm.len(), wide);
    assert_eq!(weights.down.len(), core.low_rank_dim * wide);
    assert_eq!(weights.up.len(), wide * core.low_rank_dim);
    let round = |value| precision.round(value);
    let mut normalized_residual = vec![0.0; residual.len()];
    let mut hidden_state = vec![0.0; num_tokens * core.hidden_dim];
    for token in 0..num_tokens {
        let residual = &residual[token * wide..(token + 1) * wide];
        let normed = &mut normalized_residual[token * wide..(token + 1) * wide];
        for (branch, row) in residual.chunks_exact(core.hidden_dim).enumerate() {
            let inv_rms = (row.iter().map(|value| value * value).sum::<f32>() / core.hidden_dim as f32 + core.eps)
                .sqrt()
                .recip();
            for (column, value) in row.iter().enumerate() {
                let index = branch * core.hidden_dim + column;
                normed[index] = round(value * inv_rms * (1.0 + weights.norm[index]));
            }
        }
        let low_rank = weights
            .down
            .chunks_exact(wide)
            .map(|weight| {
                let projected = round(dot(normed, weight));
                let scaled = round(projected / core.num_branches as f32);
                round(scaled * sigmoid(scaled))
            })
            .collect::<Vec<_>>();
        let mut mixed = vec![0.0; core.hidden_dim];
        for (index, weight) in weights.up.chunks_exact(core.low_rank_dim).enumerate() {
            let gate = round(sigmoid(round(dot(&low_rank, weight))));
            mixed[index % core.hidden_dim] += round(gate * normed[index]);
        }
        for (output, value) in hidden_state[token * core.hidden_dim..(token + 1) * core.hidden_dim]
            .iter_mut()
            .zip(mixed)
        {
            *output = round(value / core.num_branches as f32);
        }
    }
    ReadOutput {
        hidden_state,
        normalized_residual,
    }
}

pub struct WriteInput<'a> {
    pub core: GatedResidualCore,
    pub precision: Precision,
    pub num_tokens: usize,
    pub residual: &'a [f32],
    pub normalized_residual: &'a [f32],
    pub hidden_state: &'a [f32],
    /// Row-major `[num_branches, num_branches * hidden_dim]`.
    pub weight: &'a [f32],
}

pub fn write(input: WriteInput<'_>) -> Vec<f32> {
    let WriteInput {
        core,
        precision,
        num_tokens,
        residual,
        normalized_residual,
        hidden_state,
        weight,
    } = input;
    core.validate();
    let wide = core.residual_dim();
    assert_eq!(residual.len(), num_tokens * wide);
    assert_eq!(normalized_residual.len(), residual.len());
    assert_eq!(hidden_state.len(), num_tokens * core.hidden_dim);
    assert_eq!(weight.len(), core.num_branches * wide);
    let round = |value| precision.round(value);
    let mut output = residual.to_vec();
    for token in 0..num_tokens {
        let normed = &normalized_residual[token * wide..(token + 1) * wide];
        let hidden = &hidden_state[token * core.hidden_dim..(token + 1) * core.hidden_dim];
        for (branch, weight) in weight.chunks_exact(wide).enumerate() {
            let projected = round(dot(normed, weight));
            let scaled = round(projected / core.num_branches as f32);
            let gate = round(2.0 * round(sigmoid(scaled)));
            let base = token * wide + branch * core.hidden_dim;
            for (column, value) in hidden.iter().enumerate() {
                output[base + column] = round(residual[base + column] + round(gate * value));
            }
        }
    }
    output
}

fn dot(input: &[f32], weight: &[f32]) -> f32 {
    input.iter().zip(weight).map(|(input, weight)| input * weight).sum()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
#[path = "reference_test.rs"]
mod tests;
