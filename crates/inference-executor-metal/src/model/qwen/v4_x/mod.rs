//! Text-only Qwen4x model composition. Runtime page ownership remains model-independent.

mod executor;
pub use executor::Qwen4xExecutor;
pub use executor::Qwen4xExecutorConfig;
pub use executor::init_qwen4x_model;
pub use executor::init_qwen4x_model_with_mtp;

mod main;
mod state;

mod component_config;
mod weight;
mod mlp;

mod mtp;
