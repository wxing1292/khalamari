use std::mem::size_of;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("../metal/dflash2_selector.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub rank: u32,
    pub top_k: u32,
    pub embedding_dtype: Dtype,
}

impl Config {
    pub fn validate(self) {
        assert!(self.rank > 0);
        assert!(self.top_k > 0 && self.top_k <= 256);
        assert_eq!(self.embedding_dtype, Dtype::Bfloat16);
    }

    pub fn candidate_count(self, shape: Shape) -> usize {
        self.validate();
        shape.validate();
        checked_product(
            "DFlash2 selector candidate count",
            &[
                shape.num_total_requests as usize,
                shape.num_steps as usize,
                self.top_k as usize,
            ],
        )
    }

    pub fn score_count(self, shape: Shape) -> usize {
        self.candidate_count(shape)
            .checked_mul(self.top_k as usize)
            .expect("DFlash2 selector score count must fit usize")
    }

    pub fn embedding_bytes(self, shape: Shape) -> usize {
        self.candidate_count(shape)
            .checked_mul(self.rank as usize)
            .and_then(|count| count.checked_mul(self.embedding_dtype.item_size()))
            .expect("DFlash2 selector embedding byte length must fit usize")
    }

    pub fn projected_hidden_bytes(self, shape: Shape) -> usize {
        checked_product(
            "DFlash2 selector projected-hidden byte length",
            &[
                shape.num_total_requests as usize,
                shape.num_steps as usize,
                self.rank as usize,
                self.embedding_dtype.item_size(),
            ],
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_requests: u32,
    pub num_steps: u32,
}

impl Shape {
    pub fn validate(self) {
        assert!(self.num_total_requests > 0);
        assert!(self.num_steps > 0);
        let _ = self
            .num_total_requests
            .checked_mul(self.num_steps)
            .expect("DFlash2 selector step count must fit u32");
    }

    fn proposal_count(self) -> usize {
        self.validate();
        self.num_total_requests as usize * self.num_steps as usize
    }
}

#[derive(Clone, Copy)]
pub struct PredecessorIdBuffers<'a> {
    pub anchor_token_ids: &'a Buffer,
    pub candidate_token_ids: &'a Buffer,
    pub predecessor_token_ids: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct ScoreBuffers<'a> {
    pub candidate_logits: &'a Buffer,
    pub projected_hidden: &'a Buffer,
    pub predecessor_embeddings: &'a Buffer,
    pub successor_embeddings: &'a Buffer,
    pub scores: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct WalkBuffers<'a> {
    pub candidate_token_ids: &'a Buffer,
    pub scores: &'a Buffer,
    pub params: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub sample_positions: &'a Buffer,
    pub sampling_domain: u32,
    pub output_distribution_indices: &'a Buffer,
    pub proposal_token_ids: &'a Buffer,
    pub proposal_probs: &'a Buffer,
    pub distribution_token_ids: &'a Buffer,
    pub distribution_probs: &'a Buffer,
    pub max_distribution_k: u32,
    pub num_output_distributions: u32,
}

pub struct Compute {
    config: Config,
    predecessor_ids: CompiledKernel,
    scores: CompiledKernel,
    walk: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            predecessor_ids: CompiledKernel::new(device, SOURCE, "dflash2_selector_predecessor_ids"),
            scores: CompiledKernel::new(device, SOURCE, "dflash2_selector_scores_bf16"),
            walk: CompiledKernel::new(device, SOURCE, "dflash2_selector_walk"),
        }
    }

    pub fn invoke_predecessor_ids<'a>(
        &'a self,
        shape: Shape,
        num_active_requests: ReplayU32,
        buffers: PredecessorIdBuffers<'a>,
    ) -> PredecessorIdInvocation<'a> {
        if let ReplayU32::Fixed(value) = num_active_requests {
            assert_eq!(value, shape.num_total_requests);
        }
        let invocation = PredecessorIdInvocation {
            compute: self,
            shape,
            num_active_requests,
            buffers,
        };
        invocation.validate();
        invocation
    }

    pub fn invoke_scores<'a>(
        &'a self,
        shape: Shape,
        num_active_requests: ReplayU32,
        buffers: ScoreBuffers<'a>,
    ) -> ScoreInvocation<'a> {
        if let ReplayU32::Fixed(value) = num_active_requests {
            assert_eq!(value, shape.num_total_requests);
        }
        let invocation = ScoreInvocation {
            compute: self,
            shape,
            num_active_requests,
            buffers,
        };
        invocation.validate();
        invocation
    }

    pub fn invoke_walk<'a>(
        &'a self,
        shape: Shape,
        num_active_requests: ReplayU32,
        buffers: WalkBuffers<'a>,
    ) -> WalkInvocation<'a> {
        if let ReplayU32::Fixed(value) = num_active_requests {
            assert_eq!(value, shape.num_total_requests);
        }
        let invocation = WalkInvocation {
            compute: self,
            shape,
            num_active_requests,
            buffers,
        };
        invocation.validate();
        invocation
    }
}

pub struct PredecessorIdInvocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    num_active_requests: ReplayU32,
    buffers: PredecessorIdBuffers<'a>,
}

pub struct ScoreInvocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    num_active_requests: ReplayU32,
    buffers: ScoreBuffers<'a>,
}

pub struct WalkInvocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    num_active_requests: ReplayU32,
    buffers: WalkBuffers<'a>,
}

impl Operator for PredecessorIdInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(&self.compute.predecessor_ids);
        recorder.set_buffer_read(0, self.buffers.anchor_token_ids, 0);
        recorder.set_buffer_read(1, self.buffers.candidate_token_ids, 0);
        recorder.set_buffer_write(2, self.buffers.predecessor_token_ids, 0);
        bind_active_requests(recorder, 3, self.shape, self.num_active_requests);
        recorder.set_u32(4, self.shape.num_steps);
        recorder.set_u32(5, self.compute.config.top_k);
        recorder.dispatch_1d(self.compute.config.candidate_count(self.shape), 256);
    }
}

impl PredecessorIdInvocation<'_> {
    fn validate(&self) {
        let candidates = self.compute.config.candidate_count(self.shape);
        assert!(self.buffers.anchor_token_ids.len_bytes() >= self.shape.num_total_requests as usize * size_of::<i32>());
        assert!(self.buffers.candidate_token_ids.len_bytes() >= candidates * size_of::<i32>());
        assert!(self.buffers.predecessor_token_ids.len_bytes() >= candidates * size_of::<i32>());
        assert!(
            u32::try_from(candidates - 1).is_ok(),
            "DFlash2 selector candidate index exceeds shader u32"
        );
    }
}

impl Operator for ScoreInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(&self.compute.scores);
        recorder.set_buffer_read(0, self.buffers.candidate_logits, 0);
        recorder.set_buffer_read(1, self.buffers.projected_hidden, 0);
        recorder.set_buffer_read(2, self.buffers.predecessor_embeddings, 0);
        recorder.set_buffer_read(3, self.buffers.successor_embeddings, 0);
        recorder.set_buffer_write(4, self.buffers.scores, 0);
        bind_active_requests(recorder, 5, self.shape, self.num_active_requests);
        recorder.set_u32(6, self.shape.num_steps);
        recorder.set_u32(7, self.compute.config.top_k);
        recorder.set_u32(8, self.compute.config.rank);
        recorder.dispatch_1d(self.compute.config.score_count(self.shape), 256);
    }
}

impl ScoreInvocation<'_> {
    fn validate(&self) {
        let config = self.compute.config;
        let candidates = config.candidate_count(self.shape);
        assert!(self.buffers.candidate_logits.len_bytes() >= candidates * size_of::<f32>());
        assert!(self.buffers.projected_hidden.len_bytes() >= config.projected_hidden_bytes(self.shape));
        assert!(self.buffers.predecessor_embeddings.len_bytes() >= config.embedding_bytes(self.shape));
        assert!(self.buffers.successor_embeddings.len_bytes() >= config.embedding_bytes(self.shape));
        assert!(self.buffers.scores.len_bytes() >= config.score_count(self.shape) * size_of::<f32>());
        assert!(
            u32::try_from(config.score_count(self.shape) - 1).is_ok(),
            "DFlash2 selector score index exceeds shader u32"
        );
    }
}

impl Operator for WalkInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(&self.compute.walk);
        recorder.set_buffer_read(0, self.buffers.candidate_token_ids, 0);
        recorder.set_buffer_read(1, self.buffers.scores, 0);
        recorder.set_buffer_read(2, self.buffers.params, 0);
        recorder.set_buffer_read(3, self.buffers.req_slots, 0);
        recorder.set_buffer_read(4, self.buffers.sample_positions, 0);
        recorder.set_buffer_read(5, self.buffers.output_distribution_indices, 0);
        recorder.set_buffer_write(6, self.buffers.proposal_token_ids, 0);
        recorder.set_buffer_write(7, self.buffers.proposal_probs, 0);
        recorder.set_buffer_write(8, self.buffers.distribution_token_ids, 0);
        recorder.set_buffer_write(9, self.buffers.distribution_probs, 0);
        bind_active_requests(recorder, 10, self.shape, self.num_active_requests);
        recorder.set_u32(11, self.shape.num_steps);
        recorder.set_u32(12, self.compute.config.top_k);
        recorder.set_u32(13, self.buffers.max_distribution_k);
        recorder.set_u32(14, self.buffers.sampling_domain);
        recorder.dispatch_threadblocks((self.shape.num_total_requests as usize, 1, 1), (1, 1, 1));
    }
}

impl WalkInvocation<'_> {
    fn validate(&self) {
        let config = self.compute.config;
        let proposals = self.shape.proposal_count();
        let candidates = config.candidate_count(self.shape);
        assert!(self.buffers.max_distribution_k >= config.top_k);
        assert!(self.buffers.num_output_distributions > 0);
        assert!(self.buffers.candidate_token_ids.len_bytes() >= candidates * size_of::<i32>());
        assert!(self.buffers.scores.len_bytes() >= config.score_count(self.shape) * size_of::<f32>());
        assert!(self.buffers.params.len_bytes() >= 4 * size_of::<u32>());
        assert!(self.buffers.req_slots.len_bytes() >= self.shape.num_total_requests as usize * size_of::<u32>());
        assert!(self.buffers.sample_positions.len_bytes() >= self.shape.num_total_requests as usize * size_of::<u32>());
        assert!(self.buffers.output_distribution_indices.len_bytes() >= proposals * size_of::<u32>());
        assert!(self.buffers.proposal_token_ids.len_bytes() >= proposals * size_of::<i32>());
        assert!(self.buffers.proposal_probs.len_bytes() >= proposals * size_of::<f32>());
        let distribution_slots = checked_product(
            "DFlash2 selector distribution slot count",
            &[
                self.buffers.num_output_distributions as usize,
                self.buffers.max_distribution_k as usize,
            ],
        );
        assert!(self.buffers.distribution_token_ids.len_bytes() >= distribution_slots * size_of::<i32>());
        assert!(self.buffers.distribution_probs.len_bytes() >= distribution_slots * size_of::<f32>());
    }
}

fn bind_active_requests(recorder: &CommandRecorder<'_>, index: usize, shape: Shape, active: ReplayU32) {
    match active {
        ReplayU32::Fixed(value) => recorder.set_u32(index, value),
        ReplayU32::Parameter(key) => {
            recorder.bind_u32(index, key, 1, shape.num_total_requests);
        },
    }
}

fn checked_product(name: &str, factors: &[usize]) -> usize {
    factors
        .iter()
        .try_fold(1usize, |product, &factor| product.checked_mul(factor))
        .unwrap_or_else(|| panic!("{name} must fit usize"))
}

#[cfg(test)]
#[path = "dflash2_selector_test.rs"]
mod tests;
