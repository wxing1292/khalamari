//! Text-only Qwen4x model composition. Runtime page ownership remains model-independent.

mod executor;
pub use executor::Qwen4xExecutor;
pub use executor::Qwen4xExecutorConfig;
pub use executor::init_qwen4x_model;

mod main;
mod state;
