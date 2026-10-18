use crate::def::DenseLinearShape;
use crate::def::SparseLinearShape;
use crate::mlp::dense::DenseMLPCore;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GatedMoECore {
    pub model_layer_index: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub shared_experts_intermediate_dim: Option<usize>,
    pub num_experts: usize,
    pub num_experts_per_token: usize,
    pub norm_topk_prob: bool,
}

impl GatedMoECore {
    pub fn validate(&self) {
        assert!(self.hidden_dim > 0);
        assert!(self.intermediate_dim > 0);
        assert!(
            self.shared_experts_intermediate_dim.is_none_or(|dim| dim > 0),
            "shared expert intermediate_dim must be positive when present"
        );
        assert!(self.num_experts > 0);
        assert!(self.num_experts_per_token > 0);
        assert!(
            self.num_experts_per_token <= self.num_experts,
            "num_experts_per_token={} must be <= num_experts={}",
            self.num_experts_per_token,
            self.num_experts
        );
    }

    pub fn router_shape(&self) -> DenseLinearShape {
        DenseLinearShape {
            out_dim: self.num_experts,
            in_dim: self.hidden_dim,
        }
    }

    pub fn gate_shape(&self) -> SparseLinearShape {
        SparseLinearShape {
            num_experts: self.num_experts,
            out_dim: self.intermediate_dim,
            in_dim: self.hidden_dim,
        }
    }

    pub fn up_shape(&self) -> SparseLinearShape {
        self.gate_shape()
    }

    pub fn down_shape(&self) -> SparseLinearShape {
        SparseLinearShape {
            num_experts: self.num_experts,
            out_dim: self.hidden_dim,
            in_dim: self.intermediate_dim,
        }
    }

    pub fn has_shared_experts(&self) -> bool {
        self.shared_experts_intermediate_dim.is_some()
    }

    pub fn shared_expert_gate_shape(&self) -> Option<DenseLinearShape> {
        self.shared_experts_intermediate_dim.map(|_| {
            DenseLinearShape {
                out_dim: 1,
                in_dim: self.hidden_dim,
            }
        })
    }

    pub fn shared_experts_core(&self) -> Option<DenseMLPCore> {
        self.shared_experts_intermediate_dim.map(|intermediate_dim| {
            DenseMLPCore {
                model_layer_index: self.model_layer_index,
                hidden_dim: self.hidden_dim,
                intermediate_dim,
            }
        })
    }
}
