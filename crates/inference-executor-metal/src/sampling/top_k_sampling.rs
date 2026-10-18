use std::rc::Rc;

use inference_backend_metal::components::sampling::top_k;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_core::sampling::SamplingDomain;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_executor_core::sampling::TopKSamplingLogitsDtype;
use inference_executor_core::sampling::TopKSamplingShape;

use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;
use crate::sampling::sampling_params::SamplingParamsStore;

struct TopKSamplingCompute {
    map: top_k::MapCompute,
    reduce: top_k::ReduceCompute,
}

impl TopKSamplingCompute {
    fn new(device: &Device) -> Self {
        Self {
            map: top_k::MapCompute::new(device),
            reduce: top_k::ReduceCompute::new(device),
        }
    }

    fn record_sample<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        logits_dtype: Dtype,
        buffers: TopKSamplingComputeBuffers<'a>,
        draw: TopKSamplingDraw,
        output: TopKSamplingOutput<'a>,
    ) {
        let shape = component_shape(shape);
        let map_buffers = top_k::MapBuffers {
            logits: buffers.logits,
            logits_offset_bytes: buffers.logits_offset_bytes,
            tile_token_ids: buffers.partial_token_ids,
            tile_logits: buffers.partial_logits,
        };
        recorder.record_with_barrier_before(ReplayOp::opaque(self.map.invoke_replay(
            shape,
            logits_dtype,
            top_k::Operation::Sample,
            map_buffers,
        )));
        self.record_reduce_sample(recorder, shape, buffers, draw, output);
    }

    fn record_write_distribution<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        logits_dtype: Dtype,
        buffers: TopKSamplingComputeBuffers<'a>,
        output: TopKSamplingWriteDistributionOutput<'a>,
    ) {
        let shape = component_shape(shape);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.map.invoke_replay(
            shape,
            logits_dtype,
            top_k::Operation::WriteDistribution,
            top_k::MapBuffers {
                logits: buffers.logits,
                logits_offset_bytes: buffers.logits_offset_bytes,
                tile_token_ids: buffers.partial_token_ids,
                tile_logits: buffers.partial_logits,
            },
        )));
        self.record_reduce_write_distribution(recorder, shape, buffers, output);
    }

    fn record_sample_and_write_distribution<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        logits_dtype: Dtype,
        buffers: TopKSamplingComputeBuffers<'a>,
        draw: TopKSamplingDraw,
        output: TopKSamplingSampleAndWriteDistributionOutput<'a>,
    ) {
        let shape = component_shape(shape);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.map.invoke_replay(
            shape,
            logits_dtype,
            top_k::Operation::SampleAndWriteDistribution,
            top_k::MapBuffers {
                logits: buffers.logits,
                logits_offset_bytes: buffers.logits_offset_bytes,
                tile_token_ids: buffers.partial_token_ids,
                tile_logits: buffers.partial_logits,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(
            self.reduce.invoke_sample_and_write_distribution_with_layout(
                shape,
                top_k::SampleAndWriteDistributionBuffers {
                    tile_token_ids: buffers.partial_token_ids,
                    tile_logits: buffers.partial_logits,
                    sampled_token_ids: output.sampled.sampled_token_ids,
                    sampled_token_probs: output.sampled.sampled_token_probs,
                    distribution_token_ids: output.distribution.token_ids,
                    distribution_probs: output.distribution.probs,
                    params: buffers.params,
                    req_slots: buffers.req_slots,
                    sample_positions: buffers.sample_positions,
                    sample_position_increment: draw.sample_position_increment,
                    sampling_domain: u32::from(draw.sampling_domain),
                    output_distribution_indices: output.distribution.output_distribution_indices,
                    max_k: output.distribution.max_k,
                    num_output_distributions: output.distribution.num_output_distributions,
                },
                self.map.partial_candidate_layout(),
            ),
        ));
    }

    fn record_reduce_sample<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: top_k::Shape,
        buffers: TopKSamplingComputeBuffers<'a>,
        draw: TopKSamplingDraw,
        output: TopKSamplingOutput<'a>,
    ) {
        recorder.record_with_barrier_before(ReplayOp::opaque(self.reduce.invoke_sample_with_layout(
            shape,
            top_k::SampleBuffers {
                tile_token_ids: buffers.partial_token_ids,
                tile_logits: buffers.partial_logits,
                token_ids: output.sampled_token_ids,
                token_probs: output.sampled_token_probs,
                params: buffers.params,
                req_slots: buffers.req_slots,
                sample_positions: buffers.sample_positions,
                sample_position_increment: draw.sample_position_increment,
                sampling_domain: u32::from(draw.sampling_domain),
            },
            self.map.partial_candidate_layout(),
        )));
    }

    fn record_reduce_write_distribution<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: top_k::Shape,
        buffers: TopKSamplingComputeBuffers<'a>,
        output: TopKSamplingWriteDistributionOutput<'a>,
    ) {
        recorder.record_with_barrier_before(ReplayOp::opaque(self.reduce.invoke_write_distribution_with_layout(
            shape,
            top_k::WriteDistributionBuffers {
                tile_token_ids: buffers.partial_token_ids,
                tile_logits: buffers.partial_logits,
                distribution_token_ids: output.token_ids,
                distribution_probs: output.probs,
                params: buffers.params,
                req_slots: buffers.req_slots,
                output_distribution_indices: output.output_distribution_indices,
                max_k: output.max_k,
                num_output_distributions: output.num_output_distributions,
            },
            self.map.partial_candidate_layout(),
        )));
    }
}

#[derive(Clone, Copy)]
struct TopKSamplingComputeBuffers<'a> {
    logits: &'a Buffer,
    logits_offset_bytes: usize,
    partial_token_ids: &'a Buffer,
    partial_logits: &'a Buffer,
    params: &'a Buffer,
    req_slots: &'a Buffer,
    sample_positions: &'a Buffer,
}

struct TopKSamplingScratch {
    partial_token_ids: Buffer,
    partial_logits: Buffer,
}

impl TopKSamplingScratch {
    fn new(device: &Device, bounds: TopKSamplingBounds, map: &top_k::MapCompute) -> Self {
        let max_shape = bounds.max_shape();
        let candidate_count = map.candidate_count(component_shape(max_shape));
        Self {
            partial_token_ids: Buffer::new_zeroed_elements(device, candidate_count, Dtype::Int32),
            partial_logits: Buffer::new_zeroed_elements(device, candidate_count, Dtype::Float32),
        }
    }
}

pub struct TopKSamplingOutputBuffers {
    pub token_ids: Buffer,
    pub token_probs: Buffer,
}

impl TopKSamplingOutputBuffers {
    pub fn new(device: &Device, bounds: TopKSamplingBounds) -> Self {
        let max_shape = bounds.max_shape();
        Self {
            token_ids: Buffer::new_zeroed_elements(device, max_shape.num_total_sampling_inputs as usize, Dtype::Int32),
            token_probs: Buffer::new_zeroed_elements(
                device,
                max_shape.num_total_sampling_inputs as usize,
                Dtype::Float32,
            ),
        }
    }

    pub fn as_output(&self) -> TopKSamplingOutput<'_> {
        TopKSamplingOutput {
            sampled_token_ids: &self.token_ids,
            sampled_token_probs: &self.token_probs,
        }
    }
}

#[derive(Clone, Copy)]
pub struct TopKSamplingWriteDistributionOutput<'a> {
    pub token_ids: &'a Buffer,
    pub probs: &'a Buffer,
    pub output_distribution_indices: &'a Buffer,
    pub max_k: u32,
    pub num_output_distributions: u32,
}

#[derive(Clone, Copy)]
pub struct TopKSamplingDraw {
    pub sample_position_increment: u32,
    pub sampling_domain: SamplingDomain,
}

#[derive(Clone, Copy)]
pub struct TopKSamplingSampleAndWriteDistributionOutput<'a> {
    pub sampled: TopKSamplingOutput<'a>,
    pub distribution: TopKSamplingWriteDistributionOutput<'a>,
}

pub struct TopKSampling {
    compute: TopKSamplingCompute,
    params: Rc<SamplingParamsStore>,
    req_slots: Buffer,
    sample_positions: Buffer,
    scratch: TopKSamplingScratch,
}

impl TopKSampling {
    pub fn new(device: &Device, params: Rc<SamplingParamsStore>) -> Self {
        let bounds = params.bounds();
        let compute = TopKSamplingCompute::new(device);
        let scratch = TopKSamplingScratch::new(device, bounds, &compute.map);
        Self {
            compute,
            params,
            req_slots: Buffer::new_zeroed_elements(device, bounds.max_sampling_inputs as usize, Dtype::Uint32),
            sample_positions: Buffer::new_zeroed_elements(device, bounds.max_sampling_inputs as usize, Dtype::Uint32),
            scratch,
        }
    }

    pub fn prepare(&self, req_slots: &[u32], sample_positions: &[u32]) {
        assert_eq!(
            req_slots.len(),
            sample_positions.len(),
            "top-k sampling requires one request slot per position"
        );
        assert!(
            req_slots.len() <= self.params.bounds().max_sampling_inputs as usize,
            "top-k sampling rows exceed capacity"
        );
        assert!(
            req_slots.iter().all(|&req_slot| req_slot < self.params.num_req_slots()),
            "top-k sampling request slot exceeds parameter capacity"
        );
        self.req_slots.write_typed(0, req_slots);
        self.sample_positions.write_typed(0, sample_positions);
    }

    pub fn set_params(&self, req_slots: &[u32], configs: &[SamplerConfig]) {
        self.params.set(req_slots, configs);
    }

    pub fn record<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        logits_dtype: TopKSamplingLogitsDtype,
        sampling_domain: SamplingDomain,
        inputs: TopKSamplingInputs<'a>,
        output: TopKSamplingOutput<'a>,
    ) {
        self.validate(shape);
        self.compute.record_sample(
            recorder,
            shape,
            component_logits_dtype(logits_dtype),
            TopKSamplingComputeBuffers {
                logits: inputs.logits,
                logits_offset_bytes: inputs.logits_offset_bytes,
                partial_token_ids: &self.scratch.partial_token_ids,
                partial_logits: &self.scratch.partial_logits,
                params: self.params.buffer(),
                req_slots: &self.req_slots,
                sample_positions: &self.sample_positions,
            },
            TopKSamplingDraw {
                sample_position_increment: 0,
                sampling_domain,
            },
            output,
        );
    }

    pub fn record_write_distribution<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        logits_dtype: TopKSamplingLogitsDtype,
        inputs: TopKSamplingInputs<'a>,
        output: TopKSamplingWriteDistributionOutput<'a>,
    ) {
        self.validate(shape);
        self.compute.record_write_distribution(
            recorder,
            shape,
            component_logits_dtype(logits_dtype),
            TopKSamplingComputeBuffers {
                logits: inputs.logits,
                logits_offset_bytes: inputs.logits_offset_bytes,
                partial_token_ids: &self.scratch.partial_token_ids,
                partial_logits: &self.scratch.partial_logits,
                params: self.params.buffer(),
                req_slots: &self.req_slots,
                sample_positions: &self.sample_positions,
            },
            output,
        );
    }

    pub fn record_with_write_distribution<'a>(
        &'a self,
        recorder: &mut impl Recorder<'a, Operator = ReplayOp<'a>>,
        shape: TopKSamplingShape,
        logits_dtype: TopKSamplingLogitsDtype,
        draw: TopKSamplingDraw,
        inputs: TopKSamplingInputs<'a>,
        output: TopKSamplingSampleAndWriteDistributionOutput<'a>,
    ) {
        self.validate(shape);
        self.compute.record_sample_and_write_distribution(
            recorder,
            shape,
            component_logits_dtype(logits_dtype),
            TopKSamplingComputeBuffers {
                logits: inputs.logits,
                logits_offset_bytes: inputs.logits_offset_bytes,
                partial_token_ids: &self.scratch.partial_token_ids,
                partial_logits: &self.scratch.partial_logits,
                params: self.params.buffer(),
                req_slots: &self.req_slots,
                sample_positions: &self.sample_positions,
            },
            draw,
            output,
        );
    }

    pub fn active_shape(&self, configs: &[SamplerConfig]) -> TopKSamplingShape {
        self.params.active_shape(configs)
    }

    pub fn max_sampling_inputs(&self) -> u32 {
        self.params.bounds().max_sampling_inputs
    }

    /// Adds active counts for the same shape used to record the replay.
    pub fn add_replay_arguments(&self, shape: TopKSamplingShape, arguments: &mut ReplayArguments) {
        let component_shape = component_shape(shape);
        self.compute
            .map
            .add_replay_arguments(component_shape, shape.num_active_sampling_inputs, arguments);
        self.compute
            .reduce
            .add_replay_arguments(component_shape, shape.num_active_sampling_inputs, arguments);
    }

    fn validate(&self, shape: TopKSamplingShape) {
        let bounds = self.params.bounds();
        assert!(shape.num_active_sampling_inputs > 0);
        assert!(shape.num_active_sampling_inputs <= shape.num_total_sampling_inputs);
        assert!(shape.num_total_sampling_inputs <= bounds.max_sampling_inputs);
        assert_eq!(shape.vocab_size, bounds.vocab_size);
        assert!(shape.top_k > 0 && shape.top_k <= bounds.top_k);
    }
}

#[derive(Clone, Copy)]
pub struct TopKSamplingInput<'a> {
    pub shape: TopKSamplingShape,
    pub logits_dtype: TopKSamplingLogitsDtype,
    pub sampling_domain: SamplingDomain,
    pub inputs: TopKSamplingInputs<'a>,
    pub output: TopKSamplingOutput<'a>,
}

impl ReplayLayer for TopKSampling {
    type Input<'a> = TopKSamplingInput<'a>;
    type Output<'a> = TopKSamplingOutput<'a>;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        Self::record(
            self,
            recorder,
            input.shape,
            input.logits_dtype,
            input.sampling_domain,
            input.inputs,
            input.output,
        );
        input.output
    }
}

#[derive(Clone, Copy)]
pub struct TopKSamplingInputs<'a> {
    pub logits: &'a Buffer,
    pub logits_offset_bytes: usize,
}

#[derive(Clone, Copy)]
pub struct TopKSamplingOutput<'a> {
    pub sampled_token_ids: &'a Buffer,
    pub sampled_token_probs: &'a Buffer,
}

fn component_shape(shape: TopKSamplingShape) -> top_k::Shape {
    top_k::Shape {
        num_total_sampling_inputs: shape.num_total_sampling_inputs,
        vocab_size: shape.vocab_size,
        top_k: shape.top_k,
    }
}

fn component_logits_dtype(dtype: TopKSamplingLogitsDtype) -> Dtype {
    match dtype {
        TopKSamplingLogitsDtype::Float32 => Dtype::Float32,
        TopKSamplingLogitsDtype::Bfloat16 => Dtype::Bfloat16,
    }
}
