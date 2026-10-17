//! Reusable Metal-backed operators without model component semantics.
//!
//! Each public operator has a scoped module. The module path identifies the
//! operation. Short leaf names identify API roles.
//!
//! A fixed operator separates fixed workload facts from invocation-time extents:
//!
//! ```text
//! Config + Shape + Buffers
//!            |
//!            v
//!          Kernel
//!            |
//!            v
//!        Invocation
//!            |
//!            v
//!     CommandRecorder
//! ```
//!
//! `affine_quantized::Matmul` is an adaptive owner. It selects one
//! `affine_quantized::Kernel` from its private registry for the runtime row
//! count.
//!
//! `matmul_bf16::Matmul` selects GEMV or GEMM.
//! `bias_activation_bf16::Kernel` is the separate row-wise epilogue used by
//! model-level affine owners.

pub mod affine_quantized;
pub mod bf16_concat_rows;
pub mod bias_activation_bf16;
pub mod conv2d_unfold;
pub mod matmul_bf16;
pub mod repeat_rows;
pub mod row_gather;
pub mod row_route;
pub mod row_scatter;
pub mod softmax;
