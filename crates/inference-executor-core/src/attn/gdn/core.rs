/// Activation applied to the projected output gate after RMS normalization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GDNOutputGate {
    Silu,
    Sigmoid,
}

/// Immutable GDN semantics and tensor geometry in canonical ABI order.
///
/// ```text
/// projected/conv Q: [T, Hqk, Dqk]
/// projected/conv K: [T, Hqk, Dqk]
/// projected/conv V: [T, Hv,  Dv]
/// recurrent state: [state_slot, Hv, Dv, Dqk]
/// ```
///
/// `T` is the flattened token axis, `Hqk`/`Hv` are head axes, and
/// `Dqk`/`Dv` are within-head dimensions. `Cqkv` is the concatenated Q/K/V
/// channel width at the projection and short-convolution boundaries:
/// `Cqkv = 2 * Hqk * Dqk + Hv * Dv`. `C` is not a head axis, head width, or
/// convolution-kernel extent.
#[derive(Clone, Debug, PartialEq)]
pub struct GDNCore {
    pub model_layer_index: usize,
    pub hidden_dim: usize,
    pub num_qk_heads: usize,
    pub qk_head_dim: usize,
    pub num_v_heads: usize,
    pub v_head_dim: usize,
    pub conv_kernel_size: usize,
    pub q_scale: f32,
    pub output_gate: GDNOutputGate,
}

impl GDNCore {
    pub fn validate(&self) {
        assert!(self.hidden_dim > 0);
        assert!(self.num_qk_heads > 0);
        assert!(self.qk_head_dim > 0);
        assert!(self.num_v_heads > 0);
        assert!(self.v_head_dim > 0);
        assert_eq!(self.num_v_heads % self.num_qk_heads, 0);
        assert!(self.conv_kernel_size > 1);
        assert!(self.q_scale > 0.0);
        let _ = self.qkvabz_dim();
    }

    pub fn qk_dim(&self) -> usize {
        self.num_qk_heads
            .checked_mul(self.qk_head_dim)
            .expect("GDN query/key dimension must fit usize")
    }

    pub fn v_dim(&self) -> usize {
        self.num_v_heads
            .checked_mul(self.v_head_dim)
            .expect("GDN value dimension must fit usize")
    }

    pub fn qkv_dim(&self) -> usize {
        self.qk_dim()
            .checked_mul(2)
            .and_then(|dim| dim.checked_add(self.v_dim()))
            .expect("GDN concatenated Q/K/V dimension must fit usize")
    }

    pub fn conv_state_len(&self) -> usize {
        self.conv_kernel_size - 1
    }

    pub fn qkvabz_dim(&self) -> usize {
        self.num_v_heads
            .checked_mul(2)
            .and_then(|dim| dim.checked_add(self.qkv_dim()))
            .and_then(|dim| dim.checked_add(self.v_dim()))
            .expect("GDN fused projection dimension must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GDNReplayShape {
    /// Current valid request count.
    pub num_reqs: u32,
    /// Recorded request capacity.
    pub num_total_reqs: u32,
    /// Current valid flat-token count.
    pub num_tokens: u32,
    /// Recorded flat-token capacity.
    pub num_total_tokens: u32,
}

impl GDNReplayShape {
    pub fn new(num_reqs: u32, num_total_reqs: u32, num_tokens: u32, num_total_tokens: u32) -> Self {
        let shape = Self {
            num_reqs,
            num_total_reqs,
            num_tokens,
            num_total_tokens,
        };
        shape.validate();
        shape
    }

    pub fn validate(self) {
        assert!(self.num_reqs > 0);
        assert!(self.num_reqs <= self.num_total_reqs);
        assert!(self.num_tokens > 0);
        assert!(self.num_tokens <= self.num_total_tokens);
        assert!(
            self.num_reqs <= self.num_tokens,
            "GDN replay requires at least one active token per request"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::GDNCore;
    use super::GDNReplayShape;

    #[test]
    fn test_replay_shape_allows_independent_capacities() {
        GDNReplayShape::new(2, 4, 2, 2);
    }

    #[test]
    #[should_panic(expected = "GDN replay requires at least one active token per request")]
    fn test_replay_shape_rejects_more_active_requests_than_tokens() {
        GDNReplayShape::new(3, 4, 2, 4);
    }

    #[test]
    #[should_panic(expected = "GDN fused projection dimension must fit usize")]
    fn test_dimension_overflow_panics() {
        GDNCore {
            model_layer_index: 0,
            hidden_dim: 1,
            num_qk_heads: 1,
            qk_head_dim: 1,
            num_v_heads: usize::MAX,
            v_head_dim: 2,
            conv_kernel_size: 2,
            q_scale: 1.0,
            output_gate: super::GDNOutputGate::Silu,
        }
        .validate();
    }
}
