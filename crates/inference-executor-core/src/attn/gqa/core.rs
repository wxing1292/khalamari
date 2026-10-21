use crate::def::DenseLinearShape;

#[derive(Clone, Debug, PartialEq)]
pub struct GQACore {
    pub model_layer_index: usize,
    pub hidden_dim: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub scale: f32,
}

impl GQACore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_layer_index: usize,
        hidden_dim: usize,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        scale: f32,
    ) -> Self {
        Self {
            model_layer_index,
            hidden_dim,
            head_dim,
            num_q_heads,
            num_kv_heads,
            scale,
        }
    }

    pub fn validate(&self) {
        assert!(self.hidden_dim > 0);
        assert!(self.head_dim > 0);
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert_eq!(self.num_q_heads % self.num_kv_heads, 0);
        assert!(self.scale > 0.0);
        let _ = self.qgkv_dim();
    }

    pub fn q_dim(&self) -> usize {
        self.num_q_heads
            .checked_mul(self.head_dim)
            .expect("GQA query dimension must fit usize")
    }

    pub fn g_dim(&self) -> usize {
        self.q_dim()
    }

    pub fn k_dim(&self) -> usize {
        self.num_kv_heads
            .checked_mul(self.head_dim)
            .expect("GQA key dimension must fit usize")
    }

    pub fn v_dim(&self) -> usize {
        self.k_dim()
    }

    pub fn qg_dim(&self) -> usize {
        self.q_dim()
            .checked_add(self.g_dim())
            .expect("GQA query/gate dimension must fit usize")
    }

    pub fn qgkv_dim(&self) -> usize {
        self.qg_dim()
            .checked_add(self.k_dim())
            .and_then(|dim| dim.checked_add(self.v_dim()))
            .expect("GQA fused projection dimension must fit usize")
    }

    pub fn qgkv_shape(&self) -> DenseLinearShape {
        self.validate();
        DenseLinearShape {
            out_dim: self.qgkv_dim(),
            in_dim: self.hidden_dim,
        }
    }

    pub fn output_shape(&self) -> DenseLinearShape {
        self.validate();
        DenseLinearShape {
            out_dim: self.hidden_dim,
            in_dim: self.q_dim(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GQACore;

    #[test]
    #[should_panic(expected = "GQA query dimension must fit usize")]
    fn test_dimension_overflow_panics() {
        let core = GQACore::new(0, 1, 2, usize::MAX, 1, 1.0);
        core.validate();
    }
}
