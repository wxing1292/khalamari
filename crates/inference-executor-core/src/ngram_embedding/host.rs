use super::core::HashParameters;
use super::core::NGramEmbeddingCore;

/// Token-only hashing for storage reads before GPU execution. Each order shares one XOR across its heads.
pub fn row_ids(core: NGramEmbeddingCore, parameters: &HashParameters, history: &[u32], tokens: &[u32]) -> Vec<u32> {
    core.validate();
    assert_eq!(history.len(), core.ngram_size as usize - 1);
    assert!(history.iter().chain(tokens).all(|&id| id < core.vocab_size));
    let heads = core.num_heads() as usize;
    let mut ids = vec![0; tokens.len().checked_mul(heads).expect("n-gram ID count must fit usize")];
    for (token, output) in ids.chunks_exact_mut(heads).enumerate() {
        let mut mixed = u64::from(tokens[token]).wrapping_mul(parameters.multipliers[0]);
        let mut cut = false;
        for lag in 1..core.ngram_size as usize {
            let id = if lag <= token {
                tokens[token - lag]
            } else {
                history[history.len() + token - lag]
            };
            mixed ^= u64::from(if cut { core.eos_token_id } else { id }).wrapping_mul(parameters.multipliers[lag]);
            cut |= id == core.eos_token_id;
            let first = (lag - 1) * core.heads_per_ngram as usize;
            for head in first..first + core.heads_per_ngram as usize {
                output[head] = (mixed as i64).rem_euclid(i64::from(parameters.head_vocab_sizes[head])) as u32
                    + parameters.head_offsets[head];
            }
        }
    }
    ids
}

#[cfg(test)]
#[path = "host_test.rs"]
mod tests;
