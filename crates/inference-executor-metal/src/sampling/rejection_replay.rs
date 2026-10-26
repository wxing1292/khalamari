use std::rc::Rc;

use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_executor_core::replay::ReplayBucketPolicy;
use inference_executor_core::sampling::SpecMicrobatch;

use crate::replay::ReplayComponent;
use crate::sampling::rejection_sampling::SparseRejectionSampling;
use crate::sampling::rejection_sampling::SparseRejectionSamplingBounds;
use crate::sampling::rejection_sampling::SparseRejectionSamplingInputs;
use crate::sampling::rejection_sampling::SparseRejectionSamplingOutput;
use crate::sampling::rejection_sampling::SparseRejectionSamplingReqParams;
use crate::sampling::rejection_sampling::SparseRejectionSamplingShape;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingInputs;
use crate::sampling::top_k_sampling::TopKSamplingWriteDistributionOutput;
use crate::sampling::top_k_sampling::config::TopKSamplingLogitsDtype;
use crate::sampling::top_k_sampling::config::TopKSamplingShape;

pub struct RejectionSampler {
    sparse_sampler: SparseRejectionSampling,
    max_requests: u32,
    max_spec_tokens: u32,
    max_target_distributions: u32,
    max_k: u32,
    request_bucket_policy: ReplayBucketPolicy,
    draft_distribution_bucket_policy: ReplayBucketPolicy,
    target_distribution_bucket_policy: ReplayBucketPolicy,
    cu_target_distributions: Buffer,
    cu_draft_distributions: Buffer,
    flat_draft_token_ids: Buffer,
    flat_draft_distribution_indices: Buffer,
    flat_accepted_token_ids: Buffer,
    flat_accepted_probs: Buffer,
    num_accepted_tokens: Buffer,
    sampled_token_ids: Buffer,
    sampled_token_probs: Buffer,
}

#[derive(Clone, Copy)]
pub struct RejectionSamplerInput<'a> {
    pub shape: SparseRejectionSamplingShape,
    pub target_token_ids: &'a Buffer,
    pub target_probs: &'a Buffer,
    pub draft_token_ids: &'a Buffer,
    pub draft_probs: &'a Buffer,
}

#[derive(Debug, Eq, PartialEq)]
pub struct PreparedRejection {
    pub decode_req_indices: Vec<usize>,
    pub num_active_draft_distributions: usize,
}

pub struct RejectionResults {
    flat_accepted_token_ids: Vec<i32>,
    flat_accepted_probs: Vec<f32>,
    num_accepted_tokens: Vec<u32>,
    sampled_token_ids: Vec<i32>,
    sampled_token_probs: Vec<f32>,
}

impl RejectionResults {
    pub fn num_accepted_tokens(&self, decode_req_index: usize) -> usize {
        self.num_accepted_tokens[decode_req_index] as usize
    }

    pub fn accepted_token_ids(&self, flat_draft_index: usize, num_tokens: usize) -> &[i32] {
        &self.flat_accepted_token_ids[flat_draft_index..flat_draft_index + num_tokens]
    }

    pub fn accepted_probs(&self, flat_draft_index: usize, num_tokens: usize) -> &[f32] {
        &self.flat_accepted_probs[flat_draft_index..flat_draft_index + num_tokens]
    }

    pub fn sampled_token_id(&self, decode_req_index: usize) -> i32 {
        self.sampled_token_ids[decode_req_index]
    }

    pub fn sampled_prob(&self, decode_req_index: usize) -> f32 {
        self.sampled_token_probs[decode_req_index]
    }
}

impl PreparedRejection {
    pub fn num_active_decode_reqs(&self) -> usize {
        self.decode_req_indices.len()
    }

    pub fn num_active_target_distributions(&self) -> usize {
        self.num_active_draft_distributions + self.num_active_decode_reqs()
    }
}

impl RejectionSampler {
    pub fn new(device: &Device, max_spec_tokens: usize, max_requests: usize, max_tokens: usize, max_k: u32) -> Self {
        assert!(max_spec_tokens > 0, "rejection sampling requires speculative tokens");
        assert!(max_requests > 0, "rejection sampling requires requests");
        assert!(max_tokens > 0, "rejection sampling requires target distributions");
        let max_draft_distributions: u32 = max_requests
            .checked_mul(max_spec_tokens)
            .expect("rejection sampling draft distributions overflow")
            .try_into()
            .expect("rejection sampling draft-distribution count must fit u32");
        let max_requests_u32: u32 = max_requests
            .try_into()
            .expect("rejection sampling request capacity must fit u32");
        let max_spec_tokens_u32 =
            u32::try_from(max_spec_tokens).expect("rejection sampling spec-token capacity must fit u32");
        let max_target_distributions =
            u32::try_from(max_tokens).expect("rejection sampling target-distribution capacity must fit u32");
        let bounds = SparseRejectionSamplingBounds {
            max_reqs: max_requests_u32,
            max_draft_distributions,
            max_target_distributions,
            max_k,
        };
        Self {
            sparse_sampler: SparseRejectionSampling::new(device, bounds),
            max_requests: max_requests_u32,
            max_spec_tokens: max_spec_tokens_u32,
            max_target_distributions,
            max_k,
            request_bucket_policy: ReplayBucketPolicy::new(max_requests_u32),
            draft_distribution_bucket_policy: ReplayBucketPolicy::new(max_draft_distributions),
            target_distribution_bucket_policy: ReplayBucketPolicy::new(max_target_distributions),
            cu_target_distributions: Buffer::new_zeroed_elements(
                device,
                max_requests
                    .checked_add(1)
                    .expect("rejection sampling cumulative target length overflow"),
                Dtype::Uint32,
            ),
            cu_draft_distributions: Buffer::new_zeroed_elements(
                device,
                max_requests
                    .checked_add(1)
                    .expect("rejection sampling cumulative draft length overflow"),
                Dtype::Uint32,
            ),
            flat_draft_token_ids: Buffer::new_zeroed_elements(device, max_draft_distributions as usize, Dtype::Int32),
            flat_draft_distribution_indices: Buffer::new_zeroed_elements(
                device,
                max_draft_distributions as usize,
                Dtype::Uint32,
            ),
            flat_accepted_token_ids: Buffer::new_zeroed_elements(
                device,
                max_draft_distributions as usize,
                Dtype::Int32,
            ),
            flat_accepted_probs: Buffer::new_zeroed_elements(device, max_draft_distributions as usize, Dtype::Float32),
            num_accepted_tokens: Buffer::new_zeroed_elements(device, max_requests, Dtype::Uint32),
            sampled_token_ids: Buffer::new_zeroed_elements(device, max_requests, Dtype::Int32),
            sampled_token_probs: Buffer::new_zeroed_elements(device, max_requests, Dtype::Float32),
        }
    }

    pub fn prepare_inputs<B>(&self, microbatch: &B, flat_draft_distribution_indices: &[u32]) -> PreparedRejection
    where
        B: SpecMicrobatch,
    {
        let decode_req_indices = (0..microbatch.num_reqs())
            .filter(|&req_index| microbatch.is_decode_req(req_index))
            .collect::<Vec<_>>();
        let num_decode_reqs = decode_req_indices.len();
        assert!(num_decode_reqs > 0, "rejection sampling requires decode requests");
        assert!(
            num_decode_reqs <= self.max_requests as usize,
            "rejection sampling requests exceed sampler capacity"
        );
        let num_draft_distributions = decode_req_indices
            .iter()
            .map(|&req_index| microbatch.num_spec_tokens(req_index) as usize)
            .sum::<usize>();
        assert!(
            num_draft_distributions <= (self.max_requests * self.max_spec_tokens) as usize,
            "rejection sampling draft distributions exceed sampler capacity"
        );
        assert!(
            num_draft_distributions + num_decode_reqs <= self.max_target_distributions as usize,
            "rejection sampling target distributions exceed max_tokens"
        );
        assert_eq!(
            flat_draft_distribution_indices.len(),
            num_draft_distributions,
            "rejection sampling draft-distribution indices must match flat drafts"
        );
        let draft_capacity = (self.max_requests * self.max_spec_tokens) as usize;
        assert!(
            flat_draft_distribution_indices
                .iter()
                .all(|&index| (index as usize) < draft_capacity),
            "rejection sampling draft-distribution index exceeds sampler capacity"
        );
        let mut cu_target = Vec::with_capacity(num_decode_reqs + 1);
        let mut cu_draft = Vec::with_capacity(num_decode_reqs + 1);
        let mut flat_draft_tokens = Vec::with_capacity(num_draft_distributions);
        cu_target.push(0_u32);
        cu_draft.push(0_u32);
        for &req_index in &decode_req_indices {
            let draft_len = microbatch.num_spec_tokens(req_index) as usize;
            assert!(
                draft_len <= self.max_spec_tokens as usize,
                "rejection sampling num_spec_tokens exceeds sampler capacity"
            );
            let q_end = microbatch.cu_tokens()[req_index + 1] as usize;
            let q_start = q_end
                .checked_sub(draft_len)
                .expect("rejection sampling draft suffix exceeds request flat num_tokens");
            flat_draft_tokens.extend_from_slice(&microbatch.flat_token_ids()[q_start..q_end]);
            cu_draft.push(flat_draft_tokens.len() as u32);
            cu_target.push((flat_draft_tokens.len() + cu_target.len()) as u32);
        }
        self.cu_target_distributions.write_typed(0, &cu_target);
        self.cu_draft_distributions.write_typed(0, &cu_draft);
        self.flat_draft_token_ids.write_typed(0, &flat_draft_tokens);
        self.flat_draft_distribution_indices
            .write_typed(0, flat_draft_distribution_indices);
        PreparedRejection {
            decode_req_indices,
            num_active_draft_distributions: num_draft_distributions,
        }
    }

    pub fn prepare_replay_shape(&self, prepared: &PreparedRejection, top_k: u32) -> SparseRejectionSamplingShape {
        let num_active_reqs = prepared.num_active_decode_reqs();
        let num_active_draft_distributions = prepared.num_active_draft_distributions;
        assert!(
            num_active_reqs > 0 && num_active_reqs <= self.max_requests as usize,
            "rejection sampling prepared requests exceed sampler capacity"
        );
        assert!(
            num_active_draft_distributions <= (self.max_requests * self.max_spec_tokens) as usize,
            "rejection sampling prepared draft distributions exceed sampler capacity"
        );
        let num_active_target_distributions = num_active_draft_distributions + num_active_reqs;
        assert!(
            num_active_target_distributions <= self.max_target_distributions as usize,
            "rejection sampling prepared target distributions exceed sampler capacity"
        );
        assert!(
            top_k > 0 && top_k <= self.max_k,
            "rejection sampling top_k exceeds capacity"
        );
        let num_active_reqs = num_active_reqs as u32;
        let num_active_draft_distributions = num_active_draft_distributions as u32;
        let num_active_target_distributions = num_active_target_distributions as u32;
        let shape = SparseRejectionSamplingShape {
            num_active_reqs,
            num_total_reqs: self.request_bucket_policy.capacity(num_active_reqs),
            num_active_draft_distributions,
            num_total_draft_distributions: self
                .draft_distribution_bucket_policy
                .capacity_allow_zero(num_active_draft_distributions),
            num_active_target_distributions,
            num_total_target_distributions: self
                .target_distribution_bucket_policy
                .capacity(num_active_target_distributions),
            top_k,
            max_target_k: self.max_k,
            max_draft_k: self.max_k,
        };
        shape.validate();
        shape
    }

    pub fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: RejectionSamplerInput<'a>) {
        self.sparse_sampler
            .record(recorder, input.shape, self.input(input), self.output());
    }

    pub fn add_replay_arguments(&self, input: RejectionSamplerInput<'_>, arguments: &mut ReplayArguments) {
        self.sparse_sampler.add_replay_arguments(input.shape, arguments);
    }

    pub fn set_runtime_params(&self, params: &[SparseRejectionSamplingReqParams]) {
        self.sparse_sampler.set_runtime_params(params);
    }

    fn input<'a>(&'a self, input: RejectionSamplerInput<'a>) -> SparseRejectionSamplingInputs<'a> {
        SparseRejectionSamplingInputs {
            target_distribution_token_ids: input.target_token_ids,
            target_distribution_probs: input.target_probs,
            draft_distribution_token_ids: input.draft_token_ids,
            draft_distribution_probs: input.draft_probs,
            flat_draft_token_ids: &self.flat_draft_token_ids,
            cu_target_distributions: &self.cu_target_distributions,
            cu_draft_distributions: &self.cu_draft_distributions,
            flat_draft_distribution_indices: &self.flat_draft_distribution_indices,
        }
    }

    pub fn output(&self) -> SparseRejectionSamplingOutput<'_> {
        SparseRejectionSamplingOutput {
            flat_accepted_token_ids: &self.flat_accepted_token_ids,
            flat_accepted_probs: &self.flat_accepted_probs,
            num_accepted_tokens: &self.num_accepted_tokens,
            sampled_token_ids: &self.sampled_token_ids,
            sampled_token_probs: &self.sampled_token_probs,
        }
    }

    pub fn read_results(&self, num_decode_reqs: usize, num_draft_distributions: usize) -> RejectionResults {
        debug_assert!(num_decode_reqs <= self.max_requests as usize);
        debug_assert!(num_draft_distributions <= (self.max_requests * self.max_spec_tokens) as usize);
        RejectionResults {
            flat_accepted_token_ids: self
                .flat_accepted_token_ids
                .read_typed::<i32>(0, num_draft_distributions),
            flat_accepted_probs: self.flat_accepted_probs.read_typed::<f32>(0, num_draft_distributions),
            num_accepted_tokens: self.num_accepted_tokens.read_typed::<u32>(0, num_decode_reqs),
            sampled_token_ids: self.sampled_token_ids.read_typed::<i32>(0, num_decode_reqs),
            sampled_token_probs: self.sampled_token_probs.read_typed::<f32>(0, num_decode_reqs),
        }
    }
}

pub struct RejectionSampling {
    sampler: Rc<TopKSampling>,
    rejector: Rc<RejectionSampler>,
}

impl RejectionSampling {
    pub fn new(sampler: Rc<TopKSampling>, rejector: Rc<RejectionSampler>) -> Self {
        Self { sampler, rejector }
    }

    pub fn rejector(&self) -> &Rc<RejectionSampler> {
        &self.rejector
    }

    fn validate_replay_input(input: &RejectionSamplingInput<'_>) {
        assert_eq!(
            input.target_shape.num_active_sampling_inputs, input.rejection.shape.num_active_target_distributions,
            "rejection target sampling rows must match active target distributions"
        );
        assert_eq!(
            input.target_shape.num_total_sampling_inputs, input.rejection.shape.num_total_target_distributions,
            "rejection target sampling capacity must match target-distribution capacity"
        );
        assert_eq!(
            input.target_shape.top_k, input.rejection.shape.top_k,
            "rejection target sampling top_k must match rejection top_k"
        );
    }
}

#[derive(Clone, Copy)]
pub struct RejectionSamplingInput<'a> {
    pub target_shape: TopKSamplingShape,
    /// Target logits and mask. Stored draft probabilities remain unchanged.
    pub inputs: TopKSamplingInputs<'a>,
    pub target_sparse: TopKSamplingWriteDistributionOutput<'a>,
    pub rejection: RejectionSamplerInput<'a>,
}

impl ReplayComponent for RejectionSampling {
    type Key = RejectionReplayKey;
    type Input<'a> = RejectionSamplingInput<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Self::validate_replay_input(input);
        RejectionReplayKey {
            num_total_reqs: input.rejection.shape.num_total_reqs,
            num_total_target_distributions: input.rejection.shape.num_total_target_distributions,
            num_total_draft_distributions: input.rejection.shape.num_total_draft_distributions,
            top_k: input.rejection.shape.top_k,
            has_token_mask: input.inputs.token_mask.is_some(),
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.sampler.record_write_distribution(
            recorder,
            input.target_shape,
            TopKSamplingLogitsDtype::Bfloat16,
            input.inputs,
            input.target_sparse,
        );
        self.rejector.record(recorder, input.rejection);
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RejectionReplayKey {
    num_total_reqs: u32,
    num_total_target_distributions: u32,
    num_total_draft_distributions: u32,
    top_k: u32,
    has_token_mask: bool,
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use inference_backend_metal::metal::Buffer;
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::metal::Dtype;
    use inference_backend_metal::metal::Stream;
    use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
    use inference_executor_core::sampling::SamplerConfig;

    use super::PreparedRejection;
    use super::RejectionSampler;
    use super::RejectionSamplerInput;
    use super::RejectionSampling;
    use super::RejectionSamplingInput;
    use crate::replay::Replay;
    use crate::sampling::rejection_sampling::SparseRejectionSamplingShape;
    use crate::sampling::sampling_params::SamplingParamsStore;
    use crate::sampling::spec_probs::SpecProbsStore;
    use crate::sampling::top_k_sampling::TopKSampling;
    use crate::sampling::top_k_sampling::TopKSamplingInputs;
    use crate::sampling::top_k_sampling::TopKSamplingWriteDistributionOutput;
    use crate::sampling::top_k_sampling::config::TopKSamplingBounds;
    use crate::sampling::top_k_sampling::config::TopKSamplingShape;

    #[test]
    fn test_prepare_inputs_success() {
        let device = Device::system_default();
        let sampler = RejectionSampler::new(&device, 3, 5, 6, 4);
        let distributions = SpecProbsStore::new(&device, 3, 5, 6, 4);
        let batch = Qwen35Microbatch::new(
            vec![4, 0, 3, 1],
            vec![0, 0, 0, 0],
            vec![5, 8, 11, 15],
            vec![10, 11, 20, 30, 31, 32, 40, 41],
            vec![0, 2, 3, 6, 8],
            vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            vec![SamplerConfig::default(); 4],
            vec![false, false, true, true, true, true, true, true],
        );
        let draft_distribution_indices = [
            distributions.draft_distribution_index(3, 0),
            distributions.draft_distribution_index(3, 1),
            distributions.draft_distribution_index(1, 0),
        ];

        let prepared = sampler.prepare_inputs(&batch, &draft_distribution_indices);
        let shape = sampler.prepare_replay_shape(&prepared, 4);
        assert_eq!(prepared.decode_req_indices, vec![1, 2, 3]);
        assert_eq!(prepared.num_active_draft_distributions, 3);
        assert_eq!(prepared.num_active_target_distributions(), 6);
        assert_eq!(shape.num_active_reqs, 3);
        assert_eq!(shape.num_total_reqs, 4);
        assert_eq!(shape.num_active_draft_distributions, 3);
        assert_eq!(shape.num_total_draft_distributions, 4);
        assert_eq!(shape.num_active_target_distributions, 6);
        assert_eq!(shape.num_total_target_distributions, 6);
        assert_eq!(sampler.cu_draft_distributions.read_typed::<u32>(0, 4), vec![0, 0, 2, 3]);
        assert_eq!(
            sampler.cu_target_distributions.read_typed::<u32>(0, 4),
            vec![0, 1, 4, 6]
        );
        assert_eq!(sampler.flat_draft_token_ids.read_typed::<i32>(0, 3), vec![31, 32, 41]);
        assert_eq!(
            sampler.flat_draft_distribution_indices.read_typed::<u32>(0, 3),
            vec![9, 10, 3]
        );
    }

    #[test]
    fn test_read_results_success() {
        let device = Device::system_default();
        let sampler = RejectionSampler::new(&device, 2, 4, 12, 4);
        sampler.flat_accepted_token_ids.write_typed(0, &[11_i32, 12, 21]);
        sampler.flat_accepted_probs.write_typed(0, &[0.1_f32, 0.2, 0.3]);
        sampler.num_accepted_tokens.write_typed(0, &[2_u32, 1]);
        sampler.sampled_token_ids.write_typed(0, &[13_i32, 22]);
        sampler.sampled_token_probs.write_typed(0, &[0.4_f32, 0.5]);

        let results = sampler.read_results(2, 3);

        assert_eq!(results.num_accepted_tokens(0), 2);
        assert_eq!(results.num_accepted_tokens(1), 1);
        assert_eq!(results.accepted_token_ids(0, 2), &[11, 12]);
        assert_eq!(results.accepted_token_ids(2, 1), &[21]);
        assert_eq!(results.accepted_probs(0, 3), &[0.1, 0.2, 0.3]);
        assert_eq!(results.sampled_token_id(0), 13);
        assert_eq!(results.sampled_token_id(1), 22);
        assert_eq!(results.sampled_prob(0), 0.4);
        assert_eq!(results.sampled_prob(1), 0.5);
    }

    #[test]
    #[should_panic(expected = "prepared draft distributions exceed sampler capacity")]
    fn test_prepare_replay_shape_invalid_shape() {
        let device = Device::system_default();
        let sampler = RejectionSampler::new(&device, 2, 2, 6, 4);
        let prepared = PreparedRejection {
            decode_req_indices: vec![0],
            num_active_draft_distributions: 5,
        };

        let _ = sampler.prepare_replay_shape(&prepared, 4);
    }

    #[test]
    #[should_panic(expected = "target sampling capacity must match target-distribution capacity")]
    fn test_replay_invalid_shape() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let bounds = TopKSamplingBounds {
            max_sampling_inputs: 8,
            vocab_size: 16,
            top_k: 4,
        };
        let sampler = Rc::new(TopKSampling::new(
            &device,
            Rc::new(SamplingParamsStore::new(&device, bounds, 4)),
        ));
        let rejector = Rc::new(RejectionSampler::new(&device, 2, 4, 8, 4));
        let mut replay = Replay::new("test rejection", RejectionSampling::new(sampler, rejector));
        let runtime = &stream;
        let spec_probs = SpecProbsStore::new(&device, 2, 4, 8, 4);
        let logits = Buffer::new_zeroed_elements(&device, 16, Dtype::Bfloat16);
        let target_distribution_indices = Buffer::new_zeroed_elements(&device, 8, Dtype::Uint32);
        let input = RejectionSamplingInput {
            target_shape: TopKSamplingShape {
                num_active_sampling_inputs: 1,
                num_total_sampling_inputs: 1,
                vocab_size: 16,
                top_k: 4,
            },
            inputs: TopKSamplingInputs {
                logits: &logits,
                logits_offset_bytes: 0,
                token_mask: None,
            },
            target_sparse: TopKSamplingWriteDistributionOutput {
                token_ids: spec_probs.target_token_ids(),
                probs: spec_probs.target_probs(),
                output_distribution_indices: &target_distribution_indices,
                max_k: 4,
                num_output_distributions: 8,
            },
            rejection: RejectionSamplerInput {
                shape: SparseRejectionSamplingShape {
                    num_active_reqs: 1,
                    num_total_reqs: 1,
                    num_active_draft_distributions: 0,
                    num_total_draft_distributions: 0,
                    num_active_target_distributions: 1,
                    num_total_target_distributions: 1,
                    top_k: 4,
                    max_target_k: 4,
                    max_draft_k: 4,
                },
                target_token_ids: spec_probs.target_token_ids(),
                target_probs: spec_probs.target_probs(),
                draft_token_ids: spec_probs.draft_token_ids(),
                draft_probs: spec_probs.draft_probs(),
            },
        };
        let (_, cache_hit) = replay.record(runtime, &input);
        assert!(!cache_hit);
        let mismatched_input = RejectionSamplingInput {
            target_shape: TopKSamplingShape {
                num_total_sampling_inputs: 2,
                ..input.target_shape
            },
            ..input
        };

        let _ = replay.record(runtime, &mismatched_input);
    }
}
