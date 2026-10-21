use std::time::Duration;

use inference_backend_metal::metal::ReplaySubmission;
use inference_executor_core::model::ExecutionSubmission;

/// Adapts Metal completion to the shared model execution contract.
pub struct MetalReplaySubmission {
    inner: ReplaySubmission,
}

impl ExecutionSubmission for MetalReplaySubmission {
    fn wait(&self) {
        self.inner.wait();
    }

    fn gpu_timestamp_durations(&self) -> Option<Vec<Duration>> {
        self.inner.gpu_timestamp_durations()
    }
}

impl From<ReplaySubmission> for MetalReplaySubmission {
    fn from(inner: ReplaySubmission) -> Self {
        Self { inner }
    }
}
