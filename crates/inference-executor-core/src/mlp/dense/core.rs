use crate::def::DenseLinearShape;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseMLPCore {
    pub model_layer_index: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
}

impl DenseMLPCore {
    pub fn validate(&self) {
        assert!(self.hidden_dim > 0);
        assert!(self.intermediate_dim > 0);
        let _ = self.gate_up_shape();
    }

    pub fn linear_shape(&self) -> DenseLinearShape {
        DenseLinearShape {
            out_dim: self.hidden_dim,
            in_dim: self.hidden_dim,
        }
    }

    pub fn gate_up_shape(&self) -> DenseLinearShape {
        DenseLinearShape {
            out_dim: self
                .intermediate_dim
                .checked_mul(2)
                .expect("dense MLP gate/up dimension must fit usize"),
            in_dim: self.hidden_dim,
        }
    }

    pub fn down_shape(&self) -> DenseLinearShape {
        DenseLinearShape {
            out_dim: self.hidden_dim,
            in_dim: self.intermediate_dim,
        }
    }
}
