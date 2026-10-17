/// Per-layer embedding transforms all residual channels before GR read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PLECore {
    pub num_channels: u32,
    pub hidden_dim: u32,
    pub embedding_dim: u32,
    pub conv_kernel_size: u32,
    pub conv_dilation: u32,
    pub norm_eps: f32,
}

impl PLECore {
    pub fn validate(self) {
        assert!(self.num_channels > 0 && self.hidden_dim > 0 && self.embedding_dim > 0);
        assert!(self.conv_kernel_size > 1 && self.conv_dilation > 0);
        assert!(self.norm_eps.is_finite() && self.norm_eps > 0.0);
        self.residual_dim();
        self.state_len();
    }

    pub fn residual_dim(self) -> u32 {
        self.num_channels
            .checked_mul(self.hidden_dim)
            .expect("PLE residual width must fit u32")
    }

    pub fn state_len(self) -> u32 {
        self.conv_kernel_size
            .checked_sub(1)
            .and_then(|n| n.checked_mul(self.conv_dilation))
            .expect("PLE convolution history length must fit u32")
    }
}
