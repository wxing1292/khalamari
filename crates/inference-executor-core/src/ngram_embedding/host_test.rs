use super::*;
use crate::ngram_embedding::reference;

#[test]
fn test_fixed_and_random() {
    for (ngram_size, heads_per_ngram, eos_token_id) in [(3, 2, 0), (5, 8, 7)] {
        let core = NGramEmbeddingCore {
            vocab_size: 32,
            eos_token_id,
            ngram_size,
            heads_per_ngram,
            embedding_dim: (ngram_size - 1) * heads_per_ngram * 32,
        };
        let parameters = HashParameters::new(core, 11, 1, 1234);
        let history = vec![eos_token_id; ngram_size as usize - 1];
        let fixed = vec![1, 2, eos_token_id, 3, 4, eos_token_id, 5];
        let mut seed = 0x8411_u32;
        let random: Vec<_> = (0..137)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed % 32
            })
            .collect();
        for tokens in [fixed, random] {
            assert_eq!(
                row_ids(core, &parameters, &history, &tokens),
                reference::row_ids(core, &parameters, &history, &tokens)
            );
            let split = 5;
            let prefix = row_ids(core, &parameters, &history, &tokens[..split]);
            let previous = &tokens[split - history.len()..split];
            let rest = row_ids(core, &parameters, previous, &tokens[split..]);
            assert_eq!(
                [prefix, rest].concat(),
                reference::row_ids(core, &parameters, &history, &tokens)
            );
        }
    }
}
