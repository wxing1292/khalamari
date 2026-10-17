//! Metal backend building blocks.
//!
//! `metal` contains the low-level raw Metal Buffer / BufferView / Stream /
//! Kernel API. `operators` contains reusable backend operators without model
//! component semantics. `components` contains buffer-first operators used by
//! model executors.

pub mod components;
pub mod metal;
pub mod operators;
mod mlx_headers;
mod runtime;

pub use runtime::MetalRuntime;
