use std::time::Duration;

pub use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayExecution;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::ReplaySubmission;
use inference_backend_metal::metal::Stream;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::backend::runtime::Runtime;
use inference_executor_core::model::ExecutionSubmission;

pub struct MetalReplayRuntime<'a> {
    stream: &'a Stream,
}

impl<'a> MetalReplayRuntime<'a> {
    pub fn new(stream: &'a Stream) -> Self {
        Self { stream }
    }

    pub fn create_recorder(&self) -> ReplayRecorder {
        ReplayRecorder::new(self.stream.create_replay_program())
    }

    pub fn gpu_timestamps_enabled(&self) -> bool {
        self.stream.gpu_timestamps_enabled()
    }

    pub fn submit_replay(&self, replay: &ReplayProgram) -> MetalReplaySubmission {
        MetalReplaySubmission::new(self.stream.submit_replay(replay))
    }

    pub fn submit_replay_with_arguments(
        &self,
        replay: &ReplayProgram,
        arguments: &ReplayArguments,
    ) -> MetalReplaySubmission {
        MetalReplaySubmission::new(self.stream.submit_replay_with_arguments(replay, arguments))
    }

    pub fn submit_replay_sequence(&self, executions: &[ReplayExecution<'_>]) -> MetalReplaySubmission {
        MetalReplaySubmission::new(self.stream.submit_replay_sequence(executions))
    }

    pub fn submit_replay_sequence_with_gpu_timestamps(
        &self,
        executions: &[ReplayExecution<'_>],
        stage_end_indices: &[usize],
    ) -> MetalReplaySubmission {
        MetalReplaySubmission::new(
            self.stream
                .submit_replay_sequence_with_gpu_timestamps(executions, stage_end_indices),
        )
    }
}

impl Runtime for MetalReplayRuntime<'_> {
    type Replay = ReplayProgram;
    type Submission = MetalReplaySubmission;
    type Recorder<'a>
        = ReplayRecorder
    where
        Self: 'a;

    fn create_recorder(&self) -> Self::Recorder<'_> {
        MetalReplayRuntime::create_recorder(self)
    }

    fn submit_replay(&self, replay: &Self::Replay) -> Self::Submission {
        MetalReplayRuntime::submit_replay(self, replay)
    }
}

pub struct ReplayRecorder {
    inner: inference_backend_metal::components::ReplayRecorder,
}

impl ReplayRecorder {
    fn new(inner: inference_backend_metal::metal::ReplayProgramBuilder) -> Self {
        Self {
            inner: inference_backend_metal::components::ReplayRecorder::new(inner),
        }
    }

    pub fn build(self) -> ReplayProgram {
        self.inner.build()
    }
}

impl<'a> Recorder<'a> for ReplayRecorder {
    type Operator = ReplayOp<'a>;
    type Replay = ReplayProgram;

    fn record(&mut self, operator: Self::Operator) {
        self.inner.record(operator);
    }

    fn record_with_barrier_before(&mut self, operator: Self::Operator) {
        self.inner.record_with_barrier_before(operator);
    }

    fn build(self) -> Self::Replay {
        ReplayRecorder::build(self)
    }
}

pub struct MetalReplaySubmission {
    inner: ReplaySubmission,
}

impl MetalReplaySubmission {
    fn new(inner: ReplaySubmission) -> Self {
        Self { inner }
    }

    pub fn wait(&self) {
        self.inner.wait();
    }

    pub fn gpu_timestamp_durations(&self) -> Option<Vec<Duration>> {
        self.inner.gpu_timestamp_durations()
    }
}

impl ExecutionSubmission for MetalReplaySubmission {
    fn wait(&self) {
        MetalReplaySubmission::wait(self);
    }

    fn gpu_timestamp_durations(&self) -> Option<Vec<Duration>> {
        MetalReplaySubmission::gpu_timestamp_durations(self)
    }
}
