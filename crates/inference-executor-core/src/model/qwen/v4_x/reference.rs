//! Eager BF16 model reference. Dense checkpoint matrices use row-major `[output,input]` values.
//! This path has no Metal pages, worklists, replay capacities, or state-slot addressing.
use std::collections::HashMap;

use half::bf16;

use super::Qwen4xModelConfig;
use super::weight_layout::*;
use crate::attn::gdn::GDNCore;
use crate::attn::gdn::reference as gdn;
use crate::attn::qsa_indexer::core::QSAIndexerCore;
use crate::attn::qsa_indexer::reference as indexer;
use crate::gated_residual::core::GatedResidualCore;
use crate::gated_residual::reference as gr;
use crate::model::qwen::v3_x::weight_layout::Qwen3xMoEWeightBindings;
use crate::ngram_embedding::core::HashParameters;
use crate::ngram_embedding::core::NGramEmbeddingCore;
use crate::ngram_embedding::reference as ngram;
use crate::ple::core::PLECore;
use crate::ple::reference as ple;
use crate::reference::sigmoid_reference as sigmoid;
use crate::reference::softmax_reference;

/// One complete request from position zero. Call independently for each batched request.
pub fn forward(
    config: &Qwen4xModelConfig,
    bindings: &Qwen4xModelWeightBindings,
    weights: &HashMap<String, Vec<f32>>,
    tokens: &[u32],
) -> Vec<f32> {
    forward_with_channels(config, bindings, weights, tokens).0
}

/// Main output and pre-mixer `[T,C,H]` streams.
pub fn forward_with_channels(
    config: &Qwen4xModelConfig,
    bindings: &Qwen4xModelWeightBindings,
    weights: &HashMap<String, Vec<f32>>,
    tokens: &[u32],
) -> (Vec<f32>, Vec<f32>) {
    let t = &config.text_config;
    let embed = &weights[&bindings.embed.weight];
    let residual = tokens
        .iter()
        .flat_map(|&id| {
            (0..t.hc_count).flat_map(move |_| {
                embed[id as usize * t.hidden_size..(id as usize + 1) * t.hidden_size]
                    .iter()
                    .copied()
            })
        })
        .collect();
    graph(config, &bindings.layers, &bindings.mixer, weights, tokens, residual)
}

/// Native MTP output and streams. Inputs use decoded BF16 values.
pub fn mtp(
    config: &Qwen4xModelConfig,
    bindings: &Qwen4xMTPWeightBindings,
    weights: &HashMap<String, Vec<f32>>,
    previous_hidden: &[f32],
    token_embedding: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let t = &config.text_config;
    let count = token_embedding.len() / t.hidden_size;
    let normalize = |input: &[f32], norm: &[f32], width: usize| -> Vec<f32> {
        input
            .chunks_exact(width)
            .flat_map(|row| {
                let inv = (row.iter().map(|x| x * x).sum::<f32>() / width as f32 + t.rms_norm_eps)
                    .sqrt()
                    .recip();
                row.iter().zip(norm).map(move |(&x, &w)| round(x * inv * (1.0 + w)))
            })
            .collect()
    };
    let hidden = normalize(
        previous_hidden,
        &weights[&bindings.hidden_norm],
        t.hidden_size * t.hc_count,
    );
    let token = normalize(token_embedding, &weights[&bindings.token_norm], t.hidden_size);
    let hidden = linear(
        &hidden,
        &weights[&bindings.hidden.weight],
        count * t.hc_count,
        t.hidden_size,
        t.hidden_size,
    );
    let token = linear(
        &token,
        &weights[&bindings.token.weight],
        count,
        t.hidden_size,
        t.hidden_size,
    );
    let residual = hidden
        .iter()
        .enumerate()
        .map(|(i, &x)| round(x + token[i / (t.hc_count * t.hidden_size) * t.hidden_size + i % t.hidden_size]))
        .collect();
    graph(
        config,
        std::slice::from_ref(&bindings.layer),
        &bindings.mixer,
        weights,
        &vec![0; count],
        residual,
    )
}

fn graph(
    config: &Qwen4xModelConfig,
    layers: &[Qwen4xLayerWeightBindings],
    mixer: &Qwen4xGRWeightBindings,
    weights: &HashMap<String, Vec<f32>>,
    tokens: &[u32],
    mut residual: Vec<f32>,
) -> (Vec<f32>, Vec<f32>) {
    let t = &config.text_config;
    let hidden = t.hidden_size;
    let count = tokens.len();
    let get = |name: &str| {
        weights
            .get(name)
            .expect("reference checkpoint tensor must exist")
            .as_slice()
    };
    let core = GatedResidualCore {
        num_branches: t.hc_count,
        hidden_dim: hidden,
        low_rank_dim: t.hc_lowrank,
        eps: t.rms_norm_eps,
    };
    let gr_read = |residual: &[f32], b: &Qwen4xGRWeightBindings| {
        gr::read(gr::ReadInput {
            core,
            precision: gr::Precision::Bf16,
            num_tokens: count,
            residual,
            weights: gr::ReadWeights {
                norm: get(&b.norm),
                down: get(&b.down),
                up: get(&b.up),
            },
        })
    };
    let gr_write = |residual: &[f32], normalized: &[f32], branch: &[f32], b: &Qwen4xGRWeightBindings| {
        gr::write(gr::WriteInput {
            core,
            precision: gr::Precision::Bf16,
            num_tokens: count,
            residual,
            normalized_residual: normalized,
            hidden_state: branch,
            weight: get(b.write.as_ref().unwrap()),
        })
    };
    let mut ple_index = 0;
    for (layer_index, layer) in layers.iter().enumerate() {
        if let Some(p) = &layer.ple {
            let nc = NGramEmbeddingCore {
                vocab_size: t.vocab_size as u32,
                eos_token_id: config.eos_token_id().unwrap(),
                ngram_size: t.ngram_size as u32,
                heads_per_ngram: t.heads_per_ngram as u32,
                embedding_dim: t.ple_embed_dim as u32,
            };
            let parameters = HashParameters::new(nc, t.ngram_vocab_size_base, ple_index, t.seed);
            ple_index += 1;
            let ids = ngram::row_ids(
                nc,
                &parameters,
                &vec![nc.eos_token_id; nc.ngram_size as usize - 1],
                tokens,
            );
            let table: Vec<f32> = p.table.iter().flat_map(|b| get(&b.weight).iter().copied()).collect();
            let embedding: Vec<_> = ids
                .iter()
                .flat_map(|&id| {
                    table[id as usize * nc.head_dim() as usize..(id as usize + 1) * nc.head_dim() as usize]
                        .iter()
                        .copied()
                })
                .collect();
            let keys = linear(
                &embedding,
                get(&p.key.weight),
                count,
                t.ple_embed_dim,
                hidden * t.hc_count,
            );
            let values = linear(&embedding, get(&p.value.weight), count, t.ple_embed_dim, hidden);
            let pc = PLECore {
                num_channels: t.hc_count as u32,
                hidden_dim: hidden as u32,
                embedding_dim: t.ple_embed_dim as u32,
                conv_kernel_size: t.ple_conv_kernel_size as u32,
                conv_dilation: t.ngram_size as u32,
                norm_eps: t.rms_norm_eps,
            };
            residual = ple::apply(ple::Input {
                core: pc,
                num_tokens: count,
                residual: &residual,
                keys: &keys,
                values: &values,
                state: &vec![0.0; pc.state_len() as usize * pc.residual_dim() as usize],
                weights: ple::Weights {
                    key_norm: get(&p.key_norm),
                    query_norm: get(&p.query_norm),
                    conv_norm: get(&p.conv_norm),
                    conv: get(&p.conv),
                },
            })
            .residual;
        }
        let read = gr_read(&residual, &layer.attention_residual);
        let input = &read.hidden_state;
        let branch = match &layer.attention {
            Qwen4xAttentionWeightBindings::GDN(b) => {
                let c = GDNCore {
                    model_layer_index: layer_index,
                    hidden_dim: hidden,
                    num_qk_heads: t.linear_num_key_heads,
                    qk_head_dim: t.linear_key_head_dim,
                    num_v_heads: t.linear_num_value_heads,
                    v_head_dim: t.linear_value_head_dim,
                    conv_kernel_size: t.linear_conv_kernel_dim,
                    q_scale: (t.linear_key_head_dim as f32).sqrt().recip(),
                    output_gate: config.output_gate(),
                };
                let qkv = linear(input, get(&b.qkv.weight), count, hidden, c.qkv_dim());
                let conv = gdn::gdn_short_conv_reference(
                    &c,
                    &[0, count as u32],
                    &vec![0.0; c.qkv_dim() * c.conv_state_len()],
                    &qkv,
                    get(&b.conv_weight),
                );
                let conv_values: Vec<_> = conv.conv_qkv.into_iter().map(round).collect();
                let a = linear(input, get(&b.a.weight), count, hidden, c.num_v_heads);
                let beta = linear(input, get(&b.b.weight), count, hidden, c.num_v_heads);
                let z = linear(input, get(&b.z.weight), count, hidden, c.v_dim());
                let recurrent = gdn::gdn_recurrent_reference(
                    &c,
                    gdn::GDNRecurrentReferenceInput {
                        cu_tokens: &[0, count as u32],
                        source_recurrent_state: &vec![0.0; c.num_v_heads * c.qk_head_dim * c.v_head_dim],
                        conv_qkv: &conv_values,
                        a: &a,
                        b: &beta,
                        a_log: get(&b.a_log),
                        dt_bias: get(&b.dt_bias),
                    },
                );
                let value: Vec<_> = gdn::gdn_output_norm_gate_reference(
                    &c,
                    &recurrent
                        .recurrent_output
                        .iter()
                        .copied()
                        .map(round)
                        .collect::<Vec<_>>(),
                    &z,
                    get(&b.norm_weight),
                    t.rms_norm_eps,
                )
                .into_iter()
                .map(round)
                .collect();
                linear(&value, get(&b.output.weight), count, c.v_dim(), hidden)
            },
            Qwen4xAttentionWeightBindings::QSA { main, indexer: b } => {
                let qg = linear(
                    input,
                    get(&main.q.weight),
                    count,
                    hidden,
                    t.num_attention_heads * t.head_dim * 2,
                );
                let raw_k = linear(
                    input,
                    get(&main.k.weight),
                    count,
                    hidden,
                    t.num_key_value_heads * t.head_dim,
                );
                let v: Vec<_> = linear(
                    input,
                    get(&main.v.weight),
                    count,
                    hidden,
                    t.num_key_value_heads * t.head_dim,
                )
                .into_iter()
                .map(fp8)
                .collect();
                let mut q = Vec::new();
                let mut gate = Vec::new();
                for pair in qg.chunks_exact(2 * t.head_dim) {
                    q.extend_from_slice(&pair[..t.head_dim]);
                    gate.extend_from_slice(&pair[t.head_dim..]);
                }
                let rope_dim = (t.head_dim as f32 * t.rope_parameters.partial_rotary_factor.unwrap()) as usize;
                let theta = t.rope_parameters.rope_theta.unwrap();
                let q = norm_rope(
                    &q,
                    t.num_attention_heads,
                    t.head_dim,
                    get(&main.q_norm_weight),
                    rope_dim,
                    theta,
                    t.rms_norm_eps,
                );
                let k: Vec<_> = norm_rope(
                    &raw_k,
                    t.num_key_value_heads,
                    t.head_dim,
                    get(&main.k_norm_weight),
                    rope_dim,
                    theta,
                    t.rms_norm_eps,
                )
                .into_iter()
                .map(fp8)
                .collect();
                let ic = QSAIndexerCore {
                    hidden_dim: hidden,
                    num_q_heads: t.indexer_n_heads,
                    head_dim: t.indexer_head_dim,
                    compress_ratio: t.indexer_compress_ratio,
                    top_k: t.indexer_budget / t.indexer_compress_ratio,
                    rope_dim,
                    rope_theta: theta,
                    norm_eps: t.rms_norm_eps,
                };
                let projected = linear(input, get(&b.qk.weight), count, hidden, ic.qk_dim());
                let index = indexer::projected_indexer(indexer::Input {
                    core: ic,
                    precision: indexer::Precision::Bf16,
                    qk: &projected,
                    raw_history: &[],
                    group_start: 0,
                    q_norm: get(&b.q_norm),
                    k_norm: get(&b.k_norm),
                });
                let mut output = vec![0.0; count * t.num_attention_heads * t.head_dim];
                for token in 0..count {
                    let visible: Vec<_> = if (token + 1) / ic.compress_ratio <= ic.top_k {
                        (0..=token).collect()
                    } else {
                        index.group_ids[token]
                            .iter()
                            .flat_map(|&g| g as usize * ic.compress_ratio..(g as usize + 1) * ic.compress_ratio)
                            .chain((token + 1) / ic.compress_ratio * ic.compress_ratio..=token)
                            .collect()
                    };
                    for head in 0..t.num_attention_heads {
                        let kv = head / (t.num_attention_heads / t.num_key_value_heads);
                        let query = &q[(token * t.num_attention_heads + head) * t.head_dim
                            ..(token * t.num_attention_heads + head + 1) * t.head_dim];
                        let logits: Vec<_> = visible
                            .iter()
                            .map(|&p| {
                                query
                                    .iter()
                                    .zip(
                                        &k[(p * t.num_key_value_heads + kv) * t.head_dim
                                            ..(p * t.num_key_value_heads + kv + 1) * t.head_dim],
                                    )
                                    .map(|(q, k)| q * k)
                                    .sum::<f32>()
                                    / (t.head_dim as f32).sqrt()
                            })
                            .collect();
                        let probabilities = softmax_reference(&logits);
                        for d in 0..t.head_dim {
                            let at = (token * t.num_attention_heads + head) * t.head_dim + d;
                            let value = round(
                                visible
                                    .iter()
                                    .zip(&probabilities)
                                    .map(|(&p, &prob)| prob * v[(p * t.num_key_value_heads + kv) * t.head_dim + d])
                                    .sum(),
                            );
                            output[at] = round(value * attention_sigmoid(gate[at]));
                        }
                    }
                }
                linear(
                    &output,
                    get(&main.output.weight),
                    count,
                    t.num_attention_heads * t.head_dim,
                    hidden,
                )
            },
        };
        residual = gr_write(&residual, &read.normalized_residual, &branch, &layer.attention_residual);
        let read = gr_read(&residual, &layer.mlp_residual);
        let branch = match &layer.mlp {
            Qwen4xMLPWeightBindings::MoE(b) => moe(config, b, &get, &read.hidden_state, count),
            Qwen4xMLPWeightBindings::Dense(b) => {
                let gate = linear(
                    &read.hidden_state,
                    get(&b.gate.weight),
                    count,
                    hidden,
                    t.intermediate_size,
                );
                let up = linear(
                    &read.hidden_state,
                    get(&b.up.weight),
                    count,
                    hidden,
                    t.intermediate_size,
                );
                let value: Vec<_> = gate
                    .iter()
                    .zip(up)
                    .map(|(&g, u)| round(round(g * round(sigmoid(g))) * u))
                    .collect();
                linear(&value, get(&b.down.weight), count, t.intermediate_size, hidden)
            },
        };
        residual = gr_write(&residual, &read.normalized_residual, &branch, &layer.mlp_residual);
    }
    (gr_read(&residual, mixer).hidden_state, residual)
}

fn linear(input: &[f32], weight: &[f32], tokens: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(weight.len(), k * n);
    input
        .chunks_exact(k)
        .take(tokens)
        .flat_map(|row| {
            weight
                .chunks_exact(k)
                .map(move |w| round(row.iter().zip(w).map(|(x, w)| x * w).sum()))
        })
        .collect()
}
fn moe<'a>(
    config: &Qwen4xModelConfig,
    b: &Qwen3xMoEWeightBindings,
    get: &impl Fn(&str) -> &'a [f32],
    input: &[f32],
    tokens: usize,
) -> Vec<f32> {
    let t = &config.text_config;
    let h = t.hidden_size;
    let n = t.moe_intermediate_size;
    let logits = linear(input, get(&b.router.weight), tokens, h, t.num_experts);
    let probs: Vec<_> = logits
        .chunks_exact(t.num_experts)
        .flat_map(|r| softmax_reference(r).into_iter().map(round))
        .collect();
    let routes = crate::mlp::moe::reference::moe_routing_from_bf16_probs_reference(
        &probs,
        tokens,
        t.num_experts,
        t.num_experts_per_tok,
        t.norm_topk_prob,
    );
    let mut output = vec![0.0; tokens * h];
    for token in 0..tokens {
        for j in 0..t.num_experts_per_tok {
            let route = token * t.num_experts_per_tok + j;
            let expert = routes.expert_indices[route] as usize;
            let w = |name: &str, k, n| &get(name)[expert * k * n..(expert + 1) * k * n];
            let gate = linear(
                &input[token * h..(token + 1) * h],
                w(&b.experts.gate.weight, h, n),
                1,
                h,
                n,
            );
            let up = linear(
                &input[token * h..(token + 1) * h],
                w(&b.experts.up.weight, h, n),
                1,
                h,
                n,
            );
            let value: Vec<_> = gate
                .iter()
                .zip(up)
                .map(|(&g, u)| round(round(g * round(sigmoid(g))) * u))
                .collect();
            let value = linear(&value, w(&b.experts.down.weight, n, h), 1, n, h);
            for d in 0..h {
                output[token * h + d] += round(routes.expert_probs[route]) * value[d];
            }
        }
    }
    output.iter_mut().for_each(|value| *value = round(*value));
    if let Some(shared) = &b.shared_expert {
        let n = t.shared_expert_intermediate_size;
        let gate = linear(input, get(&shared.gate.weight), tokens, h, n);
        let up = linear(input, get(&shared.up.weight), tokens, h, n);
        let value: Vec<_> = gate
            .iter()
            .zip(up)
            .map(|(&g, u)| round(round(g * round(sigmoid(g))) * u))
            .collect();
        let value = linear(&value, get(&shared.down.weight), tokens, n, h);
        let gate = linear(input, get(&b.shared_expert_gate.as_ref().unwrap().weight), tokens, h, 1);
        for (token, &gate) in gate.iter().enumerate() {
            for d in 0..h {
                let at = token * h + d;
                output[at] = round(output[at] + sigmoid(gate) * value[at]);
            }
        }
    }
    output
}
// The BF16 attention gate stores exp, denominator, reciprocal, and the negative branch.
fn attention_sigmoid(value: f32) -> f32 {
    let probability = round(round(1.0 + round((-value.abs()).exp())).recip());
    if value < 0.0 {
        round(1.0 - probability)
    } else {
        probability
    }
}
fn norm_rope(input: &[f32], heads: usize, dim: usize, norm: &[f32], rotary: usize, theta: f32, eps: f32) -> Vec<f32> {
    let mut output = input.to_vec();
    for (index, (row, dst)) in input.chunks_exact(dim).zip(output.chunks_exact_mut(dim)).enumerate() {
        let inv = (row.iter().map(|x| x * x).sum::<f32>() / dim as f32 + eps)
            .sqrt()
            .recip();
        for d in 0..dim {
            dst[d] = row[d] * inv * (1.0 + norm[d]);
        }
        for d in 0..rotary / 2 {
            let (sin, cos) = ((index / heads) as f32 * theta.powf(-2.0 * d as f32 / rotary as f32)).sin_cos();
            let a = dst[d];
            let b = dst[d + rotary / 2];
            dst[d] = a * cos - b * sin;
            dst[d + rotary / 2] = a * sin + b * cos;
        }
        dst.iter_mut().for_each(|v| *v = round(*v));
    }
    output
}
fn round(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}
fn fp8(value: f32) -> f32 {
    // Enumerate the E4M3FN lattice. This oracle does not use the backend's converter.
    let magnitude = value.abs().min(448.0);
    let (_, closest) = (0u32..127)
        .map(|bits| {
            let e = bits >> 3;
            let m = bits & 7;
            let v = if e == 0 {
                m as f32 / 512.0
            } else {
                (1.0 + m as f32 / 8.0) * 2.0f32.powi(e as i32 - 7)
            };
            (bits, v)
        })
        .min_by(|(a, x), (b, y)| {
            (x - magnitude)
                .abs()
                .total_cmp(&(y - magnitude).abs())
                .then((a & 1).cmp(&(b & 1)))
        })
        .unwrap();
    closest.copysign(value)
}
