use std::mem::size_of;

use super::MAX_TOP_K;
use super::SAMPLING_SOURCE;
use super::checked_bytes;
use super::checked_num_threads;
use super::checked_product;
use super::failure::SamplingFailure;
use super::failure::SamplingOperation;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Operator;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants {
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn current() -> Self {
        Self {
            thread_block: ThreadBlockConstants { required_threads: 256 },
        }
    }
}

const NUM_ACTIVE_THREADS_KEY: ReplayParameterKey = ReplayParameterKey::new("rejection_sampling.num_active_threads");
const NUM_TARGET_DISTRIBUTIONS_KEY: ReplayParameterKey =
    ReplayParameterKey::new("rejection_sampling.num_active_target_distributions");
const NUM_DRAFT_DISTRIBUTIONS_KEY: ReplayParameterKey =
    ReplayParameterKey::new("rejection_sampling.num_active_draft_distributions");

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub num_total_reqs: u32,
    pub num_total_draft_distributions: u32,
    pub num_total_target_distributions: u32,
    pub top_k: u32,
    pub max_target_k: u32,
    pub max_draft_k: u32,
}

impl Shape {
    pub fn validate(self) {
        assert!(self.num_total_reqs > 0);
        assert!(self.num_total_target_distributions > 0);
        assert!(self.top_k > 0);
        assert!(self.top_k <= MAX_TOP_K);
        assert!(self.max_target_k >= self.top_k);
        assert!(self.max_draft_k >= self.top_k);
    }

    pub fn num_accepted_token_slots(self) -> usize {
        self.num_total_draft_distributions.max(1) as usize
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub target_distribution_token_ids: &'a Buffer,
    pub target_distribution_probs: &'a Buffer,
    pub draft_distribution_token_ids: &'a Buffer,
    pub draft_distribution_probs: &'a Buffer,
    pub flat_draft_token_ids: &'a Buffer,
    pub cu_target_distributions: &'a Buffer,
    pub cu_draft_distributions: &'a Buffer,
    pub flat_draft_distribution_indices: &'a Buffer,
    pub flat_accepted_token_ids: &'a Buffer,
    pub flat_accepted_probs: &'a Buffer,
    pub num_accepted_tokens: &'a Buffer,
    pub sampled_token_ids: &'a Buffer,
    pub sampled_token_probs: &'a Buffer,
    pub runtime_params: &'a Buffer,
}

pub struct Compute {
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Compute {
    pub fn new(device: &crate::metal::Device) -> Self {
        Self {
            constants: KernelConstants::current(),
            kernel: CompiledKernel::new(device, SAMPLING_SOURCE, "rejection_sparse_sample"),
        }
    }

    pub fn invoke_replay<'a>(&'a self, shape: Shape, buffers: Buffers<'a>) -> Invocation<'a> {
        shape.validate();
        let num_target_slots = checked_product(
            "sparse rejection target-distribution slot count",
            &[
                shape.num_total_target_distributions as usize,
                shape.max_target_k as usize,
            ],
        );
        let num_draft_slots = checked_product(
            "sparse rejection draft-distribution slot count",
            &[shape.num_total_draft_distributions as usize, shape.max_draft_k as usize],
        );
        assert!(
            buffers.target_distribution_token_ids.len_bytes()
                >= checked_bytes("sparse rejection target token", num_target_slots, size_of::<i32>()),
            "sparse rejection target-distribution token buffer too short"
        );
        assert!(
            buffers.target_distribution_probs.len_bytes()
                >= checked_bytes(
                    "sparse rejection target probability",
                    num_target_slots,
                    size_of::<f32>()
                ),
            "sparse rejection target-distribution probability buffer too short"
        );
        assert_eq!(
            buffers.draft_distribution_token_ids.len_bytes() / size_of::<i32>(),
            buffers.draft_distribution_probs.len_bytes() / size_of::<f32>(),
            "sparse rejection draft-distribution token/probability buffers must have equal element counts"
        );
        if shape.num_total_draft_distributions > 0 {
            assert!(
                buffers.draft_distribution_token_ids.len_bytes()
                    >= checked_bytes("sparse rejection draft token", num_draft_slots, size_of::<i32>()),
                "sparse rejection draft-distribution token buffer too short"
            );
            assert!(
                buffers.draft_distribution_probs.len_bytes()
                    >= checked_bytes("sparse rejection draft probability", num_draft_slots, size_of::<f32>()),
                "sparse rejection draft-distribution probability buffer too short"
            );
        }
        assert!(
            buffers.flat_draft_token_ids.len_bytes()
                >= checked_bytes(
                    "sparse rejection flat draft token",
                    shape.num_total_draft_distributions as usize,
                    size_of::<i32>(),
                ),
            "sparse rejection draft-token buffer is too short"
        );
        assert!(
            buffers.cu_target_distributions.len_bytes()
                >= checked_bytes(
                    "sparse rejection cumulative target distribution",
                    (shape.num_total_reqs as usize)
                        .checked_add(1)
                        .expect("sparse rejection request count must fit usize"),
                    size_of::<u32>(),
                ),
            "sparse rejection target CU-distribution buffer is too short"
        );
        assert!(
            buffers.cu_draft_distributions.len_bytes()
                >= checked_bytes(
                    "sparse rejection cumulative draft distribution",
                    (shape.num_total_reqs as usize)
                        .checked_add(1)
                        .expect("sparse rejection request count must fit usize"),
                    size_of::<u32>(),
                ),
            "sparse rejection draft CU-distribution buffer is too short"
        );
        assert!(
            buffers.flat_draft_distribution_indices.len_bytes()
                >= checked_bytes(
                    "sparse rejection flat draft-distribution index",
                    shape.num_total_draft_distributions as usize,
                    size_of::<u32>(),
                ),
            "sparse rejection flat draft-distribution index buffer too short"
        );
        assert!(
            buffers.flat_accepted_token_ids.len_bytes()
                >= checked_bytes(
                    "sparse rejection accepted token",
                    shape.num_accepted_token_slots(),
                    size_of::<i32>(),
                ),
            "sparse rejection accepted-token buffer is too short"
        );
        assert!(
            buffers.flat_accepted_probs.len_bytes()
                >= checked_bytes(
                    "sparse rejection accepted probability",
                    shape.num_accepted_token_slots(),
                    size_of::<f32>(),
                ),
            "sparse rejection accepted-probability buffer is too short"
        );
        assert!(
            buffers.num_accepted_tokens.len_bytes()
                >= checked_bytes(
                    "sparse rejection accepted-token count",
                    shape.num_total_reqs as usize,
                    size_of::<u32>(),
                ),
            "sparse rejection accepted-token-count buffer is too short"
        );
        assert!(
            buffers.sampled_token_ids.len_bytes()
                >= checked_bytes(
                    "sparse rejection sampled token",
                    shape.num_total_reqs as usize,
                    size_of::<i32>(),
                ),
            "sparse rejection sampled-token buffer is too short"
        );
        assert!(
            buffers.sampled_token_probs.len_bytes()
                >= checked_bytes(
                    "sparse rejection sampled probability",
                    shape.num_total_reqs as usize,
                    size_of::<f32>(),
                ),
            "sparse rejection sampled-probability buffer is too short"
        );
        assert!(
            buffers.runtime_params.len_bytes()
                >= checked_product(
                    "sparse rejection runtime parameter byte length",
                    &[shape.num_total_reqs as usize, 4, size_of::<u32>()],
                ),
            "sparse rejection runtime parameter buffer is too short"
        );
        checked_num_threads(shape.num_total_reqs, self.constants.thread_block.required_threads);
        Invocation {
            kernel: self,
            shape,
            buffers,
        }
    }

    /// Adds active values for the same shape used to record the replay.
    pub fn add_replay_arguments(
        &self,
        shape: Shape,
        num_active_reqs: u32,
        num_active_target_distributions: u32,
        num_active_draft_distributions: u32,
        arguments: &mut ReplayArguments,
    ) {
        assert!(
            num_active_reqs > 0 && num_active_reqs <= shape.num_total_reqs,
            "sparse rejection active requests must fit the recorded capacity"
        );
        assert!(
            num_active_target_distributions > 0
                && num_active_target_distributions <= shape.num_total_target_distributions,
            "sparse rejection active target distributions must fit the recorded capacity"
        );
        assert!(
            num_active_draft_distributions <= shape.num_total_draft_distributions,
            "sparse rejection active draft distributions must fit the recorded capacity"
        );
        arguments.set_u32(
            NUM_ACTIVE_THREADS_KEY,
            checked_num_threads(num_active_reqs, self.constants.thread_block.required_threads),
        );
        arguments.set_u32(NUM_TARGET_DISTRIBUTIONS_KEY, num_active_target_distributions);
        if shape.num_total_draft_distributions > 0 {
            arguments.set_u32(NUM_DRAFT_DISTRIBUTIONS_KEY, num_active_draft_distributions);
        }
    }
}

pub struct Invocation<'a> {
    kernel: &'a Compute,
    shape: Shape,
    buffers: Buffers<'a>,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(&self.kernel.kernel);
        SamplingFailure::record(
            recorder,
            &self.kernel.kernel,
            SamplingOperation::Rejection,
            format!("{:?}", self.shape),
        );
        recorder.set_buffer_read(0, self.buffers.target_distribution_token_ids, 0);
        recorder.set_buffer_read(1, self.buffers.target_distribution_probs, 0);
        recorder.set_buffer_read(2, self.buffers.draft_distribution_token_ids, 0);
        recorder.set_buffer_read(3, self.buffers.draft_distribution_probs, 0);
        recorder.set_buffer_read(4, self.buffers.flat_draft_token_ids, 0);
        recorder.set_buffer_read(5, self.buffers.cu_target_distributions, 0);
        recorder.set_buffer_read(6, self.buffers.cu_draft_distributions, 0);
        recorder.set_buffer_write(7, self.buffers.flat_accepted_token_ids, 0);
        recorder.set_buffer_write(8, self.buffers.flat_accepted_probs, 0);
        recorder.set_buffer_write(9, self.buffers.num_accepted_tokens, 0);
        recorder.set_buffer_write(10, self.buffers.sampled_token_ids, 0);
        recorder.set_buffer_write(11, self.buffers.sampled_token_probs, 0);
        recorder.set_buffer_read(12, self.buffers.runtime_params, 0);
        recorder.set_buffer_read(13, self.buffers.flat_draft_distribution_indices, 0);
        recorder.set_u32(17, self.shape.top_k);
        recorder.set_u32(18, self.shape.max_target_k);
        recorder.set_u32(19, self.shape.max_draft_k);
        let num_threads_per_req = self.kernel.constants.thread_block.required_threads;
        let num_total_threads = self.shape.num_total_reqs * num_threads_per_req;
        recorder.bind_u32(14, NUM_ACTIVE_THREADS_KEY, num_threads_per_req, num_total_threads);
        recorder.bind_u32(
            15,
            NUM_TARGET_DISTRIBUTIONS_KEY,
            1,
            self.shape.num_total_target_distributions,
        );
        if self.shape.num_total_draft_distributions == 0 {
            recorder.set_u32(16, 0);
        } else {
            recorder.bind_u32(
                16,
                NUM_DRAFT_DISTRIBUTIONS_KEY,
                0,
                self.shape.num_total_draft_distributions,
            );
        }
        recorder.dispatch_1d(num_total_threads as usize, num_threads_per_req as usize);
    }
}

#[cfg(test)]
#[path = "rejection_test.rs"]
mod tests;
