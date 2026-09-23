/// Gated residual geometry. Each token carries `num_branches` independent hidden rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedResidualCore {
    pub num_branches: usize,
    pub hidden_dim: usize,
    pub low_rank_dim: usize,
    pub eps: f32,
}

impl GatedResidualCore {
    pub fn validate(self) {
        assert!(self.num_branches > 0);
        assert!(self.hidden_dim > 0);
        assert!(self.low_rank_dim > 0);
        assert!(self.eps.is_finite() && self.eps > 0.0);
        self.residual_dim();
    }

    pub fn residual_dim(self) -> usize {
        self.num_branches
            .checked_mul(self.hidden_dim)
            .expect("GR residual dimension must fit usize")
    }
}
