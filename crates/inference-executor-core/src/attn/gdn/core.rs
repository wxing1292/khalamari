/// Activation applied to the projected output gate after RMS normalization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GDNOutputGate {
    Silu,
    Sigmoid,
}

/// Immutable GDN semantics and tensor geometry in logical reference order.
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

#[cfg(test)]
mod tests {
    use super::GDNCore;

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
