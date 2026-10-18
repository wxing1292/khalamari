//! Sampling compute building blocks.
//!
//! The backend owns kernel selection and dispatch geometry. Executor components
//! provide shapes, runtime parameters, buffers, and output requirements.
//!
//! ```text
//! logits
//!   |
//!   v
//! top_k::MapCompute
//!   |
//!   +--> partial token IDs
//!   +--> partial logits
//!            |
//!            v
//!      top_k::ReduceCompute
//!            |
//!            +--> sampled token and probability
//!            +--> sparse distribution
//!
//! target sparse distributions + draft sparse distributions
//!            |
//!            v
//! rejection::Compute
//!            |
//!            +--> accepted draft tokens
//!            +--> sampled bonus token
//! ```

const SAMPLING_SOURCE: &str = include_str!("../metal/sampling.metal");
const MAX_TOP_K: u32 = 256;
mod failure;

fn checked_num_threads(num_work_items: u32, num_threads_per_work_item: u32) -> u32 {
    num_work_items
        .checked_mul(num_threads_per_work_item)
        .expect("Metal sampling thread count must fit u32")
}

fn checked_product(name: &str, factors: &[usize]) -> usize {
    factors
        .iter()
        .try_fold(1usize, |product, &factor| product.checked_mul(factor))
        .unwrap_or_else(|| panic!("{name} must fit usize"))
}

fn checked_bytes(name: &str, num_elements: usize, item_size: usize) -> usize {
    num_elements
        .checked_mul(item_size)
        .unwrap_or_else(|| panic!("{name} byte length must fit usize"))
}

pub mod dflash2_selector;
pub mod rejection;
pub mod spec_decode_input;
pub mod top_k;
