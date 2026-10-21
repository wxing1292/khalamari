use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::gated_residual;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::gated_residual::core::GatedResidualCore;

use crate::gated_residual::scratch::GatedResidualScratch;

/// Records independent GR read and write phases into a caller-owned replay.
///
/// `T` counts tokens, `C = num_branches` counts residual channels, `H` is the
/// hidden dimension, and `r` is the gate projection rank.
/// Qwen3.8 uses `C = 4`, `H = 2560`, and `r = 320`.
/// Projection arrows use checkpoint weights `[output, input]`: `X @ weight.T`.
/// RMSNorm uses the affine multiplier `1 + norm`.
///
/// ```text
/// READ: normalize each channel
///   R [T,C,H] -- RMSNorm over H, norm [C,H] --> N [T,C,H]
///
/// READ: compute gates from N_flat [T,C*H], a view of N
///   N_flat [T,C*H] --+-- down [r,C*H] --> d [T,r]
///                   |                       |
///                   |                   /C then SiLU
///                   |                       v
///                   |                    z [T,r]
///                   |                       |
///                   |                  up [C*H,r]
///                   |                       v
///                   |                   u [T,C*H] -- sigmoid/view --> G [T,C,H]
///                   |
///                   +-- write_weight [C,C*H] --> w [T,C]
///                                                   |
///                                               /C then 2*sigmoid
///                                                   v
///                                                a [T,C]
///
/// READ: mix the channels
///   N [T,C,H] --+
///              +-- multiply --> NG [T,C,H] -- mean over C --> x [T,H]
///   G [T,C,H] --+
///
/// COMPONENT
///   x [T,H] -- attention / MoE --> y [T,H]
///
/// WRITE: broadcast a across H and y across C
///   a [T,C] -- view [T,C,1] --+
///                           +-- multiply --> update [T,C,H] --+
///   y [T,H] -- view [T,1,H] --+                                |
///                                                            +-- add --> R' [T,C,H]
///   R [T,C,H] -----------------------------------------------+           (may overwrite R)
/// ```
///
/// Shapes describe the mathematical values. `d`, `u`, `w`, `G`, `NG`, and
/// `update` stay inside kernels. Views and broadcasts do not allocate buffers.
/// `N [T,C,H]` and projection intermediates are scratch and can be reused after read.
/// `R [T,C,H]` and `a [T,C]` remain live until write. `a` does not depend on `y`.
///
/// The backend selects projections by recorded token capacity and geometry.
/// It fuses a compatible write with the next read and preserves BF16 boundaries.
/// Scratch holds normalized residual and kernel-local projection intermediates.
/// Qwen3.8 uses GR normalization in place of separate attention/MLP input RMSNorms.
pub struct GatedResidual {
    compute: gated_residual::Compute,
}

pub struct ReadInput<'a> {
    pub num_total_tokens: u32,
    pub num_active_tokens: ReplayU32,
    pub residual: &'a Buffer,
    pub hidden_state: &'a Buffer,
    pub scratch: &'a GatedResidualScratch,
    pub weights: gated_residual::ReadWeights<'a>,
    pub write_gate: Option<gated_residual::WriteGate<'a>>,
}

pub struct WriteInput<'a> {
    pub num_total_tokens: u32,
    pub num_active_tokens: ReplayU32,
    pub residual: &'a Buffer,
    pub gate: &'a Buffer,
    pub hidden_state: &'a Buffer,
    pub output: &'a Buffer,
}

impl GatedResidual {
    pub fn new(device: &Device, core: GatedResidualCore) -> Self {
        Self {
            compute: gated_residual::Compute::new(device, super::config(core)),
        }
    }

    pub fn read<'a>(&'a self, recorder: &mut ReplayRecorder, input: ReadInput<'a>) -> &'a Buffer {
        let scratch = input.scratch.bindings();
        recorder.record_with_barrier_before(ReplayOp::gated_residual_read(self.compute.read(
            gated_residual::Shape {
                num_total_tokens: input.num_total_tokens,
            },
            input.num_active_tokens,
            gated_residual::ReadBuffers {
                residual: input.residual,
                hidden_state: input.hidden_state,
                write_gate: input.write_gate,
            },
            scratch,
            input.weights,
        )));
        input.hidden_state
    }

    pub fn write<'a>(&'a self, recorder: &mut ReplayRecorder, input: WriteInput<'a>) -> &'a Buffer {
        recorder.record_with_barrier_before(ReplayOp::gated_residual_write(self.compute.write(
            gated_residual::Shape {
                num_total_tokens: input.num_total_tokens,
            },
            input.num_active_tokens,
            gated_residual::WriteBuffers {
                residual: input.residual,
                gate: input.gate,
                hidden_state: input.hidden_state,
                output: input.output,
            },
        )));
        input.output
    }
}

#[cfg(test)]
#[path = "backend_test.rs"]
mod tests;
