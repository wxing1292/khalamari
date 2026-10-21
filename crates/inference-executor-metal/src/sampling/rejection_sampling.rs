use std::cell::Cell;

use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::sampling::rejection;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseRejectionSamplingBounds {
    pub max_reqs: u32,
    pub max_draft_distributions: u32,
    pub max_target_distributions: u32,
    pub max_k: u32,
}

impl SparseRejectionSamplingBounds {
    pub fn validate(self) {
        assert!(self.max_reqs > 0, "sparse rejection sampling requires request capacity");
        assert!(
            self.max_draft_distributions > 0,
            "sparse rejection sampling requires draft-distribution capacity"
        );
        assert!(
            self.max_target_distributions > 0,
            "sparse rejection sampling requires target-distribution capacity"
        );
        assert!(self.max_k > 0, "sparse rejection sampling requires top_k capacity");
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseRejectionSamplingShape {
    pub num_active_reqs: u32,
    pub num_total_reqs: u32,
    pub num_active_draft_distributions: u32,
    pub num_total_draft_distributions: u32,
    pub num_active_target_distributions: u32,
    pub num_total_target_distributions: u32,
    pub top_k: u32,
    pub max_target_k: u32,
    pub max_draft_k: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseRejectionSamplingReqParams {
    pub seed: u32,
    pub sample_position: u32,
    pub top_k: u32,
}

impl SparseRejectionSamplingShape {
    pub fn validate(self) {
        assert!(self.num_active_reqs > 0, "sparse rejection sampling requires reqs");
        assert!(self.num_active_reqs <= self.num_total_reqs);
        assert!(self.num_active_draft_distributions <= self.num_total_draft_distributions);
        assert!(self.num_active_target_distributions <= self.num_total_target_distributions);
        assert!(self.top_k > 0, "sparse rejection sampling requires top_k");
        assert!(
            self.max_target_k >= self.top_k,
            "sparse rejection target distribution must cover top_k"
        );
        assert!(
            self.max_draft_k >= self.top_k,
            "sparse rejection draft distribution must cover top_k"
        );
        let expected_num_target_distributions = self
            .num_active_draft_distributions
            .checked_add(self.num_active_reqs)
            .expect("sparse rejection target-distribution count must fit u32");
        assert_eq!(
            self.num_active_target_distributions, expected_num_target_distributions,
            "sparse rejection requires one target distribution per draft plus one final distribution per request"
        );
    }
}

#[derive(Clone, Copy)]
pub struct SparseRejectionSamplingOutput<'a> {
    pub flat_accepted_token_ids: &'a Buffer,
    pub flat_accepted_probs: &'a Buffer,
    pub num_accepted_tokens: &'a Buffer,
    pub sampled_token_ids: &'a Buffer,
    pub sampled_token_probs: &'a Buffer,
}

pub struct SparseRejectionSampling {
    kernel: rejection::Compute,
    bounds: SparseRejectionSamplingBounds,
    runtime_params: Buffer,
    runtime_param_rows: Cell<Option<u32>>,
}

impl SparseRejectionSampling {
    pub fn new(device: &Device, bounds: SparseRejectionSamplingBounds) -> Self {
        bounds.validate();
        Self {
            kernel: rejection::Compute::new(device),
            bounds,
            runtime_params: Buffer::new_zeroed_elements(
                device,
                (bounds.max_reqs as usize)
                    .checked_mul(4)
                    .expect("sparse rejection runtime parameter capacity must fit usize"),
                Dtype::Uint32,
            ),
            runtime_param_rows: Cell::new(None),
        }
    }

    pub fn set_runtime_params(&self, params: &[SparseRejectionSamplingReqParams]) {
        assert!(
            params.len() <= self.bounds.max_reqs as usize,
            "sparse rejection sampling runtime request params exceed total requests"
        );
        for (req_index, params) in params.iter().enumerate() {
            assert!(
                params.top_k > 0 && params.top_k <= self.bounds.max_k,
                "sparse rejection sampling request top_k must fit capacity"
            );
            self.runtime_params.write_typed(
                req_index * 4,
                &[
                    params.seed,
                    params.sample_position,
                    params.top_k,
                    0, // Padding keeps each request entry 16-byte aligned.
                ],
            );
        }
        self.runtime_param_rows.set(Some(params.len() as u32));
    }

    pub fn record<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        shape: SparseRejectionSamplingShape,
        inputs: SparseRejectionSamplingInputs<'a>,
        output: SparseRejectionSamplingOutput<'a>,
    ) {
        self.validate(shape);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.kernel.invoke_replay(
            component_shape(shape),
            rejection::Buffers {
                target_distribution_token_ids: inputs.target_distribution_token_ids,
                target_distribution_probs: inputs.target_distribution_probs,
                draft_distribution_token_ids: inputs.draft_distribution_token_ids,
                draft_distribution_probs: inputs.draft_distribution_probs,
                flat_draft_token_ids: inputs.flat_draft_token_ids,
                cu_target_distributions: inputs.cu_target_distributions,
                cu_draft_distributions: inputs.cu_draft_distributions,
                flat_draft_distribution_indices: inputs.flat_draft_distribution_indices,
                flat_accepted_token_ids: output.flat_accepted_token_ids,
                flat_accepted_probs: output.flat_accepted_probs,
                num_accepted_tokens: output.num_accepted_tokens,
                sampled_token_ids: output.sampled_token_ids,
                sampled_token_probs: output.sampled_token_probs,
                runtime_params: &self.runtime_params,
            },
        )));
    }

    /// Adds active counts for the same shape used to record the replay.
    pub fn add_replay_arguments(&self, shape: SparseRejectionSamplingShape, arguments: &mut ReplayArguments) {
        assert_eq!(
            self.runtime_param_rows.take(),
            Some(shape.num_active_reqs),
            "sparse rejection sampling runtime parameter rows must be freshly configured for the active replay rows"
        );
        self.kernel.add_replay_arguments(
            component_shape(shape),
            shape.num_active_reqs,
            shape.num_active_target_distributions,
            shape.num_active_draft_distributions,
            arguments,
        );
    }

    fn validate(&self, shape: SparseRejectionSamplingShape) {
        shape.validate();
        assert!(
            shape.num_total_reqs <= self.bounds.max_reqs,
            "sparse rejection sampling total requests={} exceeds max={}",
            shape.num_total_reqs,
            self.bounds.max_reqs
        );
        assert!(
            shape.num_total_draft_distributions <= self.bounds.max_draft_distributions,
            "sparse rejection sampling total draft distributions={} exceeds max={}",
            shape.num_total_draft_distributions,
            self.bounds.max_draft_distributions
        );
        assert!(
            shape.num_total_target_distributions <= self.bounds.max_target_distributions,
            "sparse rejection sampling total target distributions={} exceeds max={}",
            shape.num_total_target_distributions,
            self.bounds.max_target_distributions
        );
        assert!(
            shape.top_k <= self.bounds.max_k,
            "sparse rejection sampling top_k={} exceed capacity={}",
            shape.top_k,
            self.bounds.max_k
        );
        assert_eq!(
            shape.max_target_k, self.bounds.max_k,
            "sparse rejection target distribution slots must match capacity"
        );
        assert_eq!(
            shape.max_draft_k, self.bounds.max_k,
            "sparse rejection draft distribution slots must match capacity"
        );
    }
}

fn component_shape(shape: SparseRejectionSamplingShape) -> rejection::Shape {
    rejection::Shape {
        num_total_reqs: shape.num_total_reqs,
        num_total_draft_distributions: shape.num_total_draft_distributions,
        num_total_target_distributions: shape.num_total_target_distributions,
        top_k: shape.top_k,
        max_target_k: shape.max_target_k,
        max_draft_k: shape.max_draft_k,
    }
}

#[derive(Clone, Copy)]
pub struct SparseRejectionSamplingInputs<'a> {
    pub target_distribution_token_ids: &'a Buffer,
    pub target_distribution_probs: &'a Buffer,
    pub draft_distribution_token_ids: &'a Buffer,
    pub draft_distribution_probs: &'a Buffer,
    pub flat_draft_token_ids: &'a Buffer,
    pub cu_target_distributions: &'a Buffer,
    pub cu_draft_distributions: &'a Buffer,
    pub flat_draft_distribution_indices: &'a Buffer,
}
