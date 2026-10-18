use std::cell::Cell;

pub mod rejection_sampling;
pub mod rejection_replay;
pub mod sampling_params;
pub mod spec_probs;
pub mod top_k_replay;
pub mod top_k_sampling;

#[derive(Default)]
struct RuntimeParamRows {
    configured: Cell<Option<u32>>,
}

impl RuntimeParamRows {
    fn set(&self, num_rows: u32) {
        self.configured.set(Some(num_rows));
    }

    fn consume(&self, num_active_rows: u32, name: &str) {
        assert_eq!(
            self.configured.take(),
            Some(num_active_rows),
            "{name} runtime parameter rows must be freshly configured for the active replay rows"
        );
    }
}
