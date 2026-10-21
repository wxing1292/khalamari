use std::rc::Rc;

use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::metal::Buffer;
use inference_executor_core::replay::ReplayBucketPolicy;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::SamplingDomain;

use crate::replay::ReplayComponent;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingDraw;
use crate::sampling::top_k_sampling::TopKSamplingInputs;
use crate::sampling::top_k_sampling::TopKSamplingOutput;
use crate::sampling::top_k_sampling::TopKSamplingSampleAndWriteDistributionOutput;
use crate::sampling::top_k_sampling::TopKSamplingWriteDistributionOutput;
use crate::sampling::top_k_sampling::config::TopKSamplingLogitsDtype;
use crate::sampling::top_k_sampling::config::TopKSamplingShape;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct TopKSamplingReplayKey {
    pub num_total_sampling_inputs: u32,
    pub top_k: u32,
}

pub struct Sampling {
    pub sampler: Rc<TopKSampling>,
    bucket_policy: ReplayBucketPolicy,
}

pub struct DraftSampling {
    pub sampler: Rc<TopKSampling>,
    bucket_policy: ReplayBucketPolicy,
}

#[derive(Clone, Copy)]
pub struct SamplingInput<'a> {
    pub shape: TopKSamplingShape,
    pub logits: &'a Buffer,
    pub output: TopKSamplingOutput<'a>,
}

#[derive(Clone, Copy)]
pub struct DraftSamplingInput<'a> {
    pub shape: TopKSamplingShape,
    pub logits: &'a Buffer,
    pub output: TopKSamplingOutput<'a>,
    pub sparse: TopKSamplingWriteDistributionOutput<'a>,
}

fn sampling_key(shape: TopKSamplingShape) -> TopKSamplingReplayKey {
    TopKSamplingReplayKey {
        num_total_sampling_inputs: shape.num_total_sampling_inputs,
        top_k: shape.top_k,
    }
}

impl Sampling {
    pub fn new(sampler: Rc<TopKSampling>) -> Self {
        let bucket_policy = ReplayBucketPolicy::new(sampler.max_sampling_inputs());
        Self { sampler, bucket_policy }
    }

    pub fn prepare_shape(&self, configs: &[SamplerConfig]) -> TopKSamplingShape {
        let active = self.sampler.active_shape(configs);
        active.with_num_total_sampling_inputs(self.bucket_policy.capacity(active.num_active_sampling_inputs))
    }
}

impl DraftSampling {
    pub fn new(sampler: Rc<TopKSampling>, max_sampling_inputs: u32) -> Self {
        assert!(
            max_sampling_inputs <= sampler.max_sampling_inputs(),
            "draft sampling replay capacity exceeds sampler capacity"
        );
        Self {
            sampler,
            bucket_policy: ReplayBucketPolicy::new(max_sampling_inputs),
        }
    }

    pub fn prepare_shape(&self, configs: &[SamplerConfig]) -> TopKSamplingShape {
        let active = self.sampler.active_shape(configs);
        active.with_num_total_sampling_inputs(self.bucket_policy.capacity(active.num_active_sampling_inputs))
    }
}

impl ReplayComponent for Sampling {
    type Key = TopKSamplingReplayKey;
    type Input<'a> = SamplingInput<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        sampling_key(input.shape)
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.sampler.record(
            recorder,
            input.shape,
            TopKSamplingLogitsDtype::Bfloat16,
            SamplingDomain::Target,
            TopKSamplingInputs {
                logits: input.logits,
                logits_offset_bytes: 0,
            },
            input.output,
        );
    }
}

impl ReplayComponent for DraftSampling {
    type Key = TopKSamplingReplayKey;
    type Input<'a> = DraftSamplingInput<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        sampling_key(input.shape)
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.sampler.record_with_write_distribution(
            recorder,
            input.shape,
            TopKSamplingLogitsDtype::Bfloat16,
            TopKSamplingDraw {
                sample_position_increment: 0,
                sampling_domain: SamplingDomain::Draft,
            },
            TopKSamplingInputs {
                logits: input.logits,
                logits_offset_bytes: 0,
            },
            TopKSamplingSampleAndWriteDistributionOutput {
                sampled: input.output,
                distribution: input.sparse,
            },
        );
    }
}
