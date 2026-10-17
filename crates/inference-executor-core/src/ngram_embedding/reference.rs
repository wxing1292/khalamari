use super::core::HashParameters;
use super::core::NGramEmbeddingCore;

/// Eager token-only reference for one request. History contains exactly `ngram_size - 1` IDs.
pub fn row_ids(core: NGramEmbeddingCore, parameters: &HashParameters, history: &[u32], tokens: &[u32]) -> Vec<u32> {
    core.validate();
    assert_eq!(history.len(), core.ngram_size as usize - 1);
    assert!(history.iter().chain(tokens).all(|&id| id < core.vocab_size));
    let sequence: Vec<_> = history.iter().chain(tokens).copied().collect();
    let mut output = Vec::new();
    for token in 0..tokens.len() {
        let position = history.len() + token;
        for head in 0..core.num_heads() as usize {
            let order = head / core.heads_per_ngram as usize + 2;
            let mut mixed = 0_u64;
            let mut cut = false;
            for lag in 0..order {
                let id = sequence[position - lag];
                mixed ^= u64::from(if cut { core.eos_token_id } else { id }).wrapping_mul(parameters.multipliers[lag]);
                if lag > 0 && id == core.eos_token_id {
                    cut = true;
                }
            }
            let size = i64::from(parameters.head_vocab_sizes[head]);
            output.push((mixed as i64).rem_euclid(size) as u32 + parameters.head_offsets[head]);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_ids_fixture() {
        let core = NGramEmbeddingCore {
            vocab_size: 32,
            eos_token_id: 0,
            ngram_size: 3,
            heads_per_ngram: 2,
            embedding_dim: 128,
        };
        let p = HashParameters::new(core, 11, 0, 1234);
        // Independent Python integer fixture from the pinned oMLX SplitMix/XOR/modulo equations.
        assert_eq!(
            p.multipliers,
            [256496738022509279, 251627398771002343, 55314113489879221]
        );
        assert_eq!(p.head_vocab_sizes, [11, 13, 17, 19]);
        assert_eq!(
            row_ids(core, &p, &[0, 0], &[1, 2, 0, 3, 4]),
            [
                9, 20, 26, 52, 2, 12, 37, 47, 2, 20, 30, 54, 5, 12, 30, 55, 3, 23, 27, 47
            ]
        );
        let first = row_ids(core, &p, &[0, 0], &[1, 2]);
        let rest = row_ids(core, &p, &[1, 2], &[0, 3, 4]);
        assert_eq!([first, rest].concat(), row_ids(core, &p, &[0, 0], &[1, 2, 0, 3, 4]));
    }
}
