/// Token-only n-gram embedding geometry. Hash heads cover orders `2..=ngram_size`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NGramEmbeddingCore {
    pub vocab_size: u32,
    pub eos_token_id: u32,
    pub ngram_size: u32,
    pub heads_per_ngram: u32,
    pub embedding_dim: u32,
}

impl NGramEmbeddingCore {
    pub fn validate(self) {
        assert!(self.vocab_size > 0 && self.eos_token_id < self.vocab_size);
        assert!(self.ngram_size >= 2 && self.heads_per_ngram > 0);
        let heads = (self.ngram_size - 1)
            .checked_mul(self.heads_per_ngram)
            .expect("n-gram head count must fit u32");
        assert!(self.embedding_dim > 0 && self.embedding_dim.is_multiple_of(heads));
    }

    pub fn num_heads(self) -> u32 {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    pub fn head_dim(self) -> u32 {
        self.embedding_dim / self.num_heads()
    }
}

/// Immutable checkpoint hash parameters. Row offsets precede storage padding and sharding.
pub struct HashParameters {
    pub multipliers: Vec<u64>,
    pub head_vocab_sizes: Vec<u32>,
    pub head_offsets: Vec<u32>,
}

impl HashParameters {
    /// `ple_layer_index` is the zero-based ordinal in `ple_layer_ids`, not a decoder layer index.
    pub fn new(core: NGramEmbeddingCore, vocab_size_base: u32, ple_layer_index: u32, seed: u64) -> Self {
        core.validate();
        assert!(vocab_size_base >= 2);
        let half_bound = (i64::MAX as u64 / u64::from(core.vocab_size)) / 2;
        let base_seed = seed.wrapping_add(10007 * u64::from(ple_layer_index));
        let multipliers = (0..core.ngram_size)
            .map(|i| {
                let value = base_seed.wrapping_add(0x9E3779B97F4A7C15_u64.wrapping_mul(u64::from(i) + 1));
                2 * (splitmix64(value) % half_bound) + 1
            })
            .collect();
        let skip = ple_layer_index
            .checked_mul(core.num_heads())
            .expect("n-gram layer head offset must fit u32");
        let count = skip
            .checked_add(core.num_heads())
            .expect("n-gram head interval must fit u32");
        let mut prime = vocab_size_base - 1;
        let mut offset = 0_u32;
        let mut head_vocab_sizes = Vec::new();
        let mut head_offsets = Vec::new();
        for i in 0..count {
            loop {
                prime = prime.checked_add(1).expect("n-gram vocabulary prime must fit u32");
                if is_prime(prime) {
                    break;
                }
            }
            if i >= skip {
                head_offsets.push(offset);
                head_vocab_sizes.push(prime);
                offset = offset.checked_add(prime).expect("n-gram table row count must fit u32");
            }
        }
        Self {
            multipliers,
            head_vocab_sizes,
            head_offsets,
        }
    }

    pub fn num_rows(&self) -> u32 {
        self.head_offsets.last().unwrap() + self.head_vocab_sizes.last().unwrap()
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E3779B97F4A7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^ (value >> 31)
}

fn is_prime(value: u32) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor <= value / divisor {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}
