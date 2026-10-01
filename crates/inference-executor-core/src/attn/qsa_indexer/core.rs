/// QSA's independent index queries and one pooled key per index group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QSAIndexerCore {
    pub hidden_dim: usize,
    pub num_q_heads: usize,
    pub head_dim: usize,
    pub compress_ratio: usize,
    pub top_k: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
}

impl QSAIndexerCore {
    pub fn validate(self) {
        assert!(self.hidden_dim > 0);
        assert!(self.num_q_heads > 0);
        assert!(self.head_dim > 0);
        assert!(self.compress_ratio > 1);
        assert!(self.top_k > 0);
        assert!(self.rope_dim > 0 && self.rope_dim <= self.head_dim);
        assert!(self.rope_dim.is_multiple_of(2));
        assert!(self.rope_theta.is_finite() && self.rope_theta > 0.0);
        assert!(self.norm_eps.is_finite() && self.norm_eps > 0.0);
        self.qk_dim();
        self.state_dim();
    }

    pub fn qk_dim(self) -> usize {
        self.num_q_heads
            .checked_add(1)
            .and_then(|heads| heads.checked_mul(self.head_dim))
            .expect("QSA indexer projection dimension must fit usize")
    }

    pub fn state_dim(self) -> usize {
        (self.compress_ratio - 1)
            .checked_mul(self.head_dim)
            .expect("QSA indexer state dimension must fit usize")
    }
}
