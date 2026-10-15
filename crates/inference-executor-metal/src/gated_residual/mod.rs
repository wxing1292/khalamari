use inference_backend_metal::components::gated_residual;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::gated_residual::core::GatedResidualCore;

pub mod backend;
pub mod scratch;

fn config(core: GatedResidualCore) -> gated_residual::Config {
    core.validate();
    gated_residual::Config {
        num_branches: core.num_branches.try_into().expect("GR branch count must fit u32"),
        hidden_dim: core.hidden_dim.try_into().expect("GR hidden dimension must fit u32"),
        low_rank_dim: core
            .low_rank_dim
            .try_into()
            .expect("GR low-rank dimension must fit u32"),
        eps: core.eps,
        dtype: Dtype::Bfloat16,
    }
}
