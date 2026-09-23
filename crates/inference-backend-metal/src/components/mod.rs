//! Reusable buffer-first Metal model execution components.
//!
//! These operators are implemented with the low-level Metal API, but their
//! contracts are component semantics rather than generic Metal primitives.
//! They should not own model layer order, scheduler policy, or runtime page
//! allocation.
//!
//! Each public component module uses short, scoped names. `Config` owns fixed
//! workload facts. These facts include dtype, model dimensions, and fixed
//! scalar parameters. `Shape` owns invocation-time extents and runtime
//! metadata. `Buffers`, `Weights`, and `Scratch` bind storage. `Compute` owns
//! reusable compiled Metal execution state. `Invocation` records one
//! execution into a stream batch. Private `KernelConstants` values describe
//! compile-time kernel constants.

pub mod audio_encoder_layout;
pub mod check_finite;
pub mod tower_block_attention;

pub mod dense_mlp;
pub mod gated_residual;
pub mod sparse_mlp;

pub mod dynamic_grouped_conv;

pub mod embedding;
pub mod layer_norm;
pub mod resource_embed;

pub mod gdn;
pub mod gqa;
pub mod moe;

pub mod residual_add;
pub mod residual_add_rms_norm;

pub mod rms_norm;
pub mod rms_norm_rope;

pub mod sampling;

mod replay;
pub use replay::ReplayOp;
pub use replay::ReplayRecorder;

fn checked_product(name: &str, factors: &[usize]) -> usize {
    factors
        .iter()
        .try_fold(1usize, |product, &factor| product.checked_mul(factor))
        .unwrap_or_else(|| panic!("{name} must fit usize"))
}

fn assert_u32_count_domain(count: usize, name: &str) {
    assert!(count > 0, "{name} must be positive");
    assert!(
        u32::try_from(count).is_ok(),
        "{name} exceeds the shader u32 count domain: count={count}"
    );
}

fn assert_u32_index_domain(num_elements: usize, name: &str) {
    assert!(num_elements > 0, "{name} must contain elements");
    assert!(
        u32::try_from(num_elements - 1).is_ok(),
        "{name} exceeds the shader u32 element-index domain: num_elements={num_elements}"
    );
}

#[cfg(test)]
mod tests {
    use super::assert_u32_count_domain;
    use super::assert_u32_index_domain;

    #[test]
    #[should_panic(expected = "exceeds the shader u32 count domain")]
    fn test_u32_count_domain_rejects_two_to_32() {
        assert_u32_count_domain(u32::MAX as usize + 1, "test count");
    }

    #[test]
    fn test_u32_index_domain_accepts_two_to_32_elements() {
        assert_u32_index_domain(u32::MAX as usize + 1, "test elements");
    }

    #[test]
    #[should_panic(expected = "exceeds the shader u32 element-index domain")]
    fn test_u32_index_domain_rejects_more_than_two_to_32_elements() {
        assert_u32_index_domain(u32::MAX as usize + 2, "test elements");
    }
}
