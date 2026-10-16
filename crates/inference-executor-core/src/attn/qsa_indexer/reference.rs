//! Eager CPU reference. History contains raw keys, independently of GPU page/state layouts.

use half::bf16;

use crate::attn::qsa_indexer::core::QSAIndexerCore;

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

pub struct Input<'a> {
    pub core: QSAIndexerCore,
    pub precision: Precision,
    /// Packed projection `[T, (Hq + 1) * D]` for one request.
    pub qk: &'a [f32],
    /// Raw keys before this call, `[token_index, D]`.
    pub raw_history: &'a [f32],
    /// Visible history starts at this complete index group.
    pub group_start: usize,
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
}

pub struct Output {
    pub q: Vec<f32>,
    pub keys: Vec<f32>,
    pub raw_keys: Vec<f32>,
    pub scores: Vec<Vec<f32>>,
    /// Chronological IDs. Higher scores win; lower group IDs win cutoff ties.
    pub group_ids: Vec<Vec<u32>>,
}

pub fn projected_indexer(input: Input<'_>) -> Output {
    let Input {
        core,
        precision,
        qk,
        raw_history,
        group_start,
        q_norm,
        k_norm,
    } = input;
    core.validate();
    assert_eq!(q_norm.len(), core.head_dim);
    assert_eq!(k_norm.len(), core.head_dim);
    assert!(qk.len().is_multiple_of(core.qk_dim()));
    assert!(raw_history.len().is_multiple_of(core.head_dim));
    let token_index = raw_history.len() / core.head_dim;
    let num_tokens = qk.len() / core.qk_dim();
    assert!(group_start <= token_index / core.compress_ratio);
    let mut raw_keys = raw_history.to_vec();
    let mut q = Vec::new();
    for (token, row) in qk.chunks_exact(core.qk_dim()).enumerate() {
        for head in row[..core.num_q_heads * core.head_dim].chunks_exact(core.head_dim) {
            q.extend(norm_rope(core, precision, head, q_norm, token_index + token));
        }
        raw_keys.extend_from_slice(&row[core.num_q_heads * core.head_dim..]);
    }
    let mut keys = Vec::new();
    for (group, rows) in raw_keys.chunks_exact(core.compress_ratio * core.head_dim).enumerate() {
        let pooled: Vec<_> = (0..core.head_dim)
            .map(|dim| {
                precision.round(
                    rows.chunks_exact(core.head_dim).map(|row| row[dim]).sum::<f32>() / core.compress_ratio as f32,
                )
            })
            .collect();
        keys.extend(norm_rope(core, precision, &pooled, k_norm, group * core.compress_ratio));
    }
    let mut scores = Vec::with_capacity(num_tokens);
    let mut group_ids = Vec::with_capacity(num_tokens);
    for token in 0..num_tokens {
        let group_end = (token_index + token + 1) / core.compress_ratio;
        let query = &q[token * core.num_q_heads * core.head_dim..(token + 1) * core.num_q_heads * core.head_dim];
        let row: Vec<f32> = (group_start..group_end)
            .map(|group| {
                let key = &keys[group * core.head_dim..(group + 1) * core.head_dim];
                query
                    .chunks_exact(core.head_dim)
                    .map(|head| head.iter().zip(key).map(|(q, k)| q * k).sum::<f32>().max(0.0))
                    .sum::<f32>()
                    / (core.head_dim as f32).sqrt()
            })
            .collect();
        let mut ids: Vec<_> = (group_start..group_end).map(|group| group as u32).collect();
        if ids.len() > core.top_k {
            ids.sort_by(|&left, &right| {
                row[right as usize - group_start]
                    .total_cmp(&row[left as usize - group_start])
                    .then(left.cmp(&right))
            });
        }
        ids.truncate(core.top_k);
        ids.sort_unstable();
        scores.push(row);
        group_ids.push(ids);
    }
    Output {
        q,
        keys,
        raw_keys,
        scores,
        group_ids,
    }
}

fn norm_rope(core: QSAIndexerCore, precision: Precision, row: &[f32], weights: &[f32], position: usize) -> Vec<f32> {
    let inv_rms = (row.iter().map(|x| x * x).sum::<f32>() / core.head_dim as f32 + core.norm_eps)
        .sqrt()
        .recip();
    let mut output: Vec<_> = row
        .iter()
        .zip(weights)
        .map(|(&x, &w)| precision.round(x * inv_rms * (1.0 + w)))
        .collect();
    for dim in 0..core.rope_dim / 2 {
        let theta = position as f32 * core.rope_theta.powf(-2.0 * dim as f32 / core.rope_dim as f32);
        // The eager model stores cos/sin and intermediate products in the tensor dtype.
        let cos = precision.round(theta.cos());
        let sin = precision.round(theta.sin());
        let other = dim + core.rope_dim / 2;
        let x = output[dim];
        let y = output[other];
        output[dim] = precision.round(precision.round(x * cos) - precision.round(y * sin));
        output[other] = precision.round(precision.round(y * cos) + precision.round(x * sin));
    }
    output
}

#[cfg(test)]
#[path = "reference_test.rs"]
mod tests;
