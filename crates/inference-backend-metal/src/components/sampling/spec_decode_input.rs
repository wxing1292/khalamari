//! Rejection-sampling output to speculative Decode input preparation.
//!
//! One 32-thread threadgroup owns one active Decode request. This is one Apple
//! SIMDgroup on current targets, but the kernel does not use SIMDgroup
//! intrinsics. Thread 0 writes request scalars. The threadgroup strides the
//! fixed Spec block and history-Task quotas. Acceptance changes token positions
//! and range endpoints. It does not change the dispatched grid or work count.

use std::mem::size_of;

use super::super::assert_u32_count_domain;
use super::super::assert_u32_index_domain;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Operator;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;

const SOURCE: &str = include_str!("../metal/spec_decode_input.metal");
const NUM_ACTIVE_REQUESTS: ReplayParameterKey = ReplayParameterKey::new("spec_decode_input.num_active_requests");
const THREADS_PER_THREADGROUP: usize = 32;

const NUM_VISIBLE_RANGE_FIELDS: usize = 2;
const NUM_Q_TOKEN_RANGE_FIELDS: usize = 2;
const NUM_TASK_TEMPLATE_FIELDS: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub spec_block_size: u32,
    pub num_q_ranges_per_request: u32,
    pub kv_tokens_per_iteration: u32,
    pub history_window: u32,
    pub max_anchor_position: u32,
    pub max_task_templates: u32,
    pub mask_token_id: i32,
}

impl Config {
    pub fn validate(self) {
        assert!(self.spec_block_size > 0, "Spec Decode block must contain tokens");
        assert!(
            self.num_q_ranges_per_request > 0,
            "Spec Decode input must contain Q-token ranges"
        );
        assert!(
            self.num_q_ranges_per_request <= self.spec_block_size,
            "Spec Decode Q-token-range count cannot exceed its Spec block"
        );
        assert!(
            self.kv_tokens_per_iteration > 0,
            "Spec Decode history iteration must contain tokens"
        );
        assert!(
            self.history_window > self.spec_block_size,
            "Spec Decode history window must keep the anchor visible for every query row"
        );
        assert!(
            self.max_anchor_position > 0,
            "Spec Decode maximum anchor must be positive"
        );
        assert!(
            self.max_task_templates > 0,
            "Spec Decode TaskTemplate capacity must be positive"
        );
        let max_anchor_position = u64::from(self.max_anchor_position);
        let kv_tokens_per_iteration = u64::from(self.kv_tokens_per_iteration);
        let max_task_templates = u64::from(self.max_task_templates);
        // Prove every uint expression in rejection_to_spec_decode_input from
        // the owner maxima. Private batch preparation can then use direct
        // arithmetic for actual anchors and fixed Task quotas.
        let max_num_iterations = max_anchor_position / kv_tokens_per_iteration
            + u64::from(!max_anchor_position.is_multiple_of(kv_tokens_per_iteration));
        assert!(
            max_num_iterations * max_task_templates <= u64::from(u32::MAX),
            "Spec Decode history partition products must fit u32"
        );
        assert!(
            max_anchor_position + kv_tokens_per_iteration - 1 <= u64::from(u32::MAX),
            "Spec Decode history partition endpoints must fit u32"
        );
        assert!(
            max_anchor_position + u64::from(self.spec_block_size) <= u64::from(u32::MAX),
            "Spec Decode query and sampling positions must fit u32"
        );
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_requests: u32,
    pub num_total_q_token_ranges: u32,
    pub num_total_task_templates: u32,
}

impl Shape {
    pub fn validate(self, config: Config) {
        assert!(
            self.num_total_requests > 0,
            "Spec Decode input requires request capacity"
        );
        assert_eq!(
            self.num_total_q_token_ranges,
            self.num_total_requests
                .checked_mul(config.num_q_ranges_per_request)
                .expect("Spec Decode Q-token-range count must fit u32"),
            "Spec Decode Q-token-range capacity must match the request topology"
        );
        assert!(
            self.num_total_task_templates
                >= self
                    .num_total_q_token_ranges
                    .checked_mul(2)
                    .expect("Spec Decode minimum TaskTemplate count must fit u32"),
            "Spec Decode input requires a history and block TaskTemplate for every Q-token range"
        );
        assert!(
            self.num_total_task_templates <= config.max_task_templates,
            "Spec Decode TaskTemplate count exceeds the validated shader capacity"
        );
        let requests = self.num_total_requests as usize;
        let query_tokens = requests
            .checked_mul(config.spec_block_size as usize)
            .expect("Spec Decode query-token count must fit usize");
        let q_token_ranges = self.num_total_q_token_ranges as usize;
        let task_templates = self.num_total_task_templates as usize;
        assert_u32_count_domain(requests, "Spec Decode request count");
        assert_u32_index_domain(query_tokens, "Spec Decode query-token metadata");
        assert_u32_index_domain(
            query_tokens
                .checked_mul(NUM_VISIBLE_RANGE_FIELDS)
                .expect("Spec Decode visible-range field count must fit usize"),
            "Spec Decode visible history-token-range metadata",
        );
        assert_u32_index_domain(
            q_token_ranges
                .checked_mul(NUM_Q_TOKEN_RANGE_FIELDS)
                .expect("Spec Decode Q-token-range field count must fit usize"),
            "Spec Decode Q-token-range metadata",
        );
        assert_u32_index_domain(
            task_templates
                .checked_mul(NUM_TASK_TEMPLATE_FIELDS)
                .expect("Spec Decode TaskTemplate field count must fit usize"),
            "Spec Decode TaskTemplate metadata",
        );
        assert_u32_index_domain(
            q_token_ranges
                .checked_add(1)
                .expect("Spec Decode cumulative partial-output count must fit usize"),
            "Spec Decode cumulative partial-output metadata",
        );
    }

    fn num_total_query_tokens(self, config: Config) -> usize {
        self.num_total_requests as usize * config.spec_block_size as usize
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub num_accepted_tokens: &'a Buffer,
    pub sampled_token_ids: &'a Buffer,
    pub anchor_indices: &'a Buffer,
    pub anchor_token_ids: &'a Buffer,
    pub sample_positions: &'a Buffer,
    pub block_token_ids: &'a Buffer,
    pub flat_query_token_indices: &'a Buffer,
    pub visible_history_token_ranges: &'a Buffer,
    pub q_token_ranges: &'a Buffer,
    pub cu_sdpa_partial_outputs: &'a Buffer,
    pub sdpa_map_task_templates: &'a Buffer,
}

pub struct Compute {
    config: Config,
    prepare: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            prepare: CompiledKernel::new(device, SOURCE, "rejection_to_spec_decode_input"),
        }
    }

    pub fn invoke_replay<'a>(&'a self, shape: Shape, buffers: Buffers<'a>) -> Invocation<'a> {
        let invocation = Invocation {
            compute: self,
            shape,
            buffers,
        };
        invocation.validate();
        invocation
    }

    /// Adds active values for the same shape used to record the replay.
    pub fn add_replay_arguments(&self, shape: Shape, num_active_requests: u32, arguments: &mut ReplayArguments) {
        assert!(
            num_active_requests > 0 && num_active_requests <= shape.num_total_requests,
            "active Spec Decode requests must fit the recorded capacity"
        );
        arguments.set_u32(NUM_ACTIVE_REQUESTS, num_active_requests);
    }
}

pub struct Invocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    buffers: Buffers<'a>,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.compute.config;
        recorder.set_kernel(&self.compute.prepare);
        recorder.set_buffer_read(0, self.buffers.num_accepted_tokens, 0);
        recorder.set_buffer_read(1, self.buffers.sampled_token_ids, 0);
        recorder.set_buffer_read(2, self.buffers.anchor_indices, 0);
        recorder.set_buffer_write(3, self.buffers.anchor_token_ids, 0);
        recorder.set_buffer_write(4, self.buffers.sample_positions, 0);
        recorder.set_buffer_write(5, self.buffers.block_token_ids, 0);
        recorder.set_buffer_write(6, self.buffers.flat_query_token_indices, 0);
        recorder.set_buffer_write(7, self.buffers.visible_history_token_ranges, 0);
        recorder.set_buffer_read(8, self.buffers.q_token_ranges, 0);
        recorder.set_buffer_read(9, self.buffers.cu_sdpa_partial_outputs, 0);
        recorder.set_buffer_write(10, self.buffers.sdpa_map_task_templates, 0);
        recorder.bind_u32(11, NUM_ACTIVE_REQUESTS, 1, self.shape.num_total_requests);
        recorder.set_u32(12, config.spec_block_size);
        recorder.set_u32(13, config.num_q_ranges_per_request);
        recorder.set_u32(14, config.kv_tokens_per_iteration);
        recorder.set_u32(15, config.history_window);
        recorder.set_i32(16, config.mask_token_id);
        recorder.dispatch_1d(
            self.shape.num_total_requests as usize * THREADS_PER_THREADGROUP,
            THREADS_PER_THREADGROUP,
        );
    }
}

impl Invocation<'_> {
    fn validate(&self) {
        self.shape.validate(self.compute.config);
        let query_tokens = self.shape.num_total_query_tokens(self.compute.config);
        assert_buffer_elements::<u32>(
            self.buffers.num_accepted_tokens,
            self.shape.num_total_requests as usize,
            "accepted-token count",
        );
        assert_buffer_elements::<i32>(
            self.buffers.sampled_token_ids,
            self.shape.num_total_requests as usize,
            "sampled token ID",
        );
        assert_buffer_elements::<u32>(
            self.buffers.anchor_indices,
            self.shape.num_total_requests as usize,
            "anchor index",
        );
        assert_buffer_elements::<i32>(
            self.buffers.anchor_token_ids,
            self.shape.num_total_requests as usize,
            "anchor token ID",
        );
        assert_buffer_elements::<u32>(
            self.buffers.sample_positions,
            self.shape.num_total_requests as usize,
            "sample position",
        );
        assert_buffer_elements::<i32>(self.buffers.block_token_ids, query_tokens, "block token ID");
        assert_buffer_elements::<u32>(
            self.buffers.flat_query_token_indices,
            query_tokens,
            "flat query-token index",
        );
        assert_buffer_elements::<u32>(
            self.buffers.visible_history_token_ranges,
            query_tokens * NUM_VISIBLE_RANGE_FIELDS,
            "visible history-token range",
        );
        assert_buffer_elements::<u32>(
            self.buffers.q_token_ranges,
            self.shape.num_total_q_token_ranges as usize * NUM_Q_TOKEN_RANGE_FIELDS,
            "Q-token range",
        );
        assert_buffer_elements::<u32>(
            self.buffers.cu_sdpa_partial_outputs,
            self.shape.num_total_q_token_ranges as usize + 1,
            "cumulative SDPA partial output",
        );
        assert_buffer_elements::<u32>(
            self.buffers.sdpa_map_task_templates,
            self.shape.num_total_task_templates as usize * NUM_TASK_TEMPLATE_FIELDS,
            "SDPA Map TaskTemplate",
        );
    }
}

fn assert_buffer_elements<T>(buffer: &Buffer, elements: usize, name: &str) {
    let bytes = elements
        .checked_mul(size_of::<T>())
        .unwrap_or_else(|| panic!("Spec Decode {name} buffer length must fit usize"));
    assert!(buffer.len_bytes() >= bytes, "Spec Decode {name} buffer is too short");
}

#[cfg(test)]
#[derive(Debug, Eq, PartialEq)]
struct ReferenceOutput {
    anchors: Vec<u32>,
    block_token_ids: Vec<i32>,
    flat_query_token_indices: Vec<u32>,
    visible_history_token_ranges: Vec<u32>,
    task_templates: Vec<u32>,
}

#[cfg(test)]
fn reference_transform(
    config: Config,
    anchor_indices: &[u32],
    num_accepted_tokens: &[u32],
    sampled_token_ids: &[i32],
    q_token_ranges: &[u32],
    cu_sdpa_partial_outputs: &[u32],
) -> ReferenceOutput {
    config.validate();
    assert_eq!(anchor_indices.len(), num_accepted_tokens.len());
    assert_eq!(anchor_indices.len(), sampled_token_ids.len());
    let mut output = ReferenceOutput {
        anchors: Vec::new(),
        block_token_ids: Vec::new(),
        flat_query_token_indices: Vec::new(),
        visible_history_token_ranges: Vec::new(),
        task_templates: Vec::new(),
    };
    for request_index in 0..anchor_indices.len() {
        let anchor = anchor_indices[request_index] + num_accepted_tokens[request_index];
        output.anchors.push(anchor);
        for offset in 0..config.spec_block_size {
            output.block_token_ids.push(if offset == 0 {
                sampled_token_ids[request_index]
            } else {
                config.mask_token_id
            });
            let query_position = anchor + offset;
            output.flat_query_token_indices.push(query_position);
            output
                .visible_history_token_ranges
                .extend_from_slice(&[(query_position + 1).saturating_sub(config.history_window), anchor]);
        }
        for local_range_index in 0..config.num_q_ranges_per_request {
            let q_range_index = request_index as u32 * config.num_q_ranges_per_request + local_range_index;
            let query_begin = q_token_ranges[q_range_index as usize * 2];
            let block_begin = request_index as u32 * config.spec_block_size;
            let first_query_position = anchor + query_begin - block_begin;
            let history_begin = (first_query_position + 1).saturating_sub(config.history_window);
            let task_begin = cu_sdpa_partial_outputs[q_range_index as usize];
            let task_end = cu_sdpa_partial_outputs[q_range_index as usize + 1] - 1;
            let num_tasks = task_end - task_begin;
            let num_iterations = (anchor - history_begin).div_ceil(config.kv_tokens_per_iteration);
            for task_offset in 0..num_tasks {
                let iteration_begin = num_iterations * task_offset / num_tasks;
                let iteration_end = num_iterations * (task_offset + 1) / num_tasks;
                output.task_templates.extend_from_slice(&[
                    q_range_index,
                    history_begin + iteration_begin * config.kv_tokens_per_iteration,
                    anchor.min(history_begin + iteration_end * config.kv_tokens_per_iteration),
                ]);
            }
            output.task_templates.extend_from_slice(&[u32::MAX; 3]);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::Buffers;
    use super::Compute;
    use super::Config;
    use super::ReferenceOutput;
    use super::Shape;
    use super::reference_transform;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::ReplayArguments;
    use crate::metal::Stream;

    #[test]
    fn test_reference_success() {
        let config = Config {
            spec_block_size: 3,
            num_q_ranges_per_request: 1,
            kv_tokens_per_iteration: 4,
            history_window: u32::MAX,
            max_anchor_position: 64,
            max_task_templates: 6,
            mask_token_id: -7,
        };
        let actual = reference_transform(config, &[10, 20], &[2, 0], &[71, 72], &[0, 3, 3, 6], &[0, 3, 6]);
        assert_eq!(
            actual,
            ReferenceOutput {
                anchors: vec![12, 20],
                block_token_ids: vec![71, -7, -7, 72, -7, -7],
                flat_query_token_indices: vec![12, 13, 14, 20, 21, 22],
                visible_history_token_ranges: vec![0, 12, 0, 12, 0, 12, 0, 20, 0, 20, 0, 20],
                task_templates: vec![
                    0,
                    0,
                    4,
                    0,
                    4,
                    12,
                    u32::MAX,
                    u32::MAX,
                    u32::MAX,
                    1,
                    0,
                    8,
                    1,
                    8,
                    20,
                    u32::MAX,
                    u32::MAX,
                    u32::MAX
                ],
            }
        );
    }

    #[test]
    fn test_replay_active_counts() {
        const POISON_U32: u32 = 0xdead_beef;
        const POISON_I32: i32 = i32::MIN + 17;

        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config {
            spec_block_size: 3,
            num_q_ranges_per_request: 1,
            kv_tokens_per_iteration: 4,
            history_window: u32::MAX,
            max_anchor_position: 64,
            max_task_templates: 12,
            mask_token_id: -7,
        };
        let shape = Shape {
            num_total_requests: 4,
            num_total_q_token_ranges: 4,
            num_total_task_templates: 12,
        };
        let num_accepted_tokens = Buffer::from_slice(&device, &[2_u32, 0, 1, 3]);
        let sampled_token_ids = Buffer::from_slice(&device, &[71_i32, 72, 73, 74]);
        let anchor_indices = Buffer::from_slice(&device, &[10_u32, 20, 30, 40]);
        let anchor_token_ids = Buffer::from_slice(&device, &[POISON_I32; 4]);
        let sample_positions = Buffer::from_slice(&device, &[POISON_U32; 4]);
        let block_token_ids = Buffer::from_slice(&device, &[POISON_I32; 12]);
        let flat_query_token_indices = Buffer::from_slice(&device, &[POISON_U32; 12]);
        let visible_history_token_ranges = Buffer::from_slice(&device, &[POISON_U32; 24]);
        let q_token_ranges = Buffer::from_slice(&device, &[0_u32, 3, 3, 6, 6, 9, 9, 12]);
        let cu_sdpa_partial_outputs = Buffer::from_slice(&device, &[0_u32, 3, 6, 9, 12]);
        let mut initial_task_templates = [POISON_U32; 36];
        for block_task in [2_usize, 5, 8, 11] {
            initial_task_templates[block_task * 3..block_task * 3 + 3].fill(u32::MAX);
        }
        let sdpa_map_task_templates = Buffer::from_slice(&device, &initial_task_templates);
        let compute = Compute::new(&device, config);
        let replay = {
            let mut builder = stream.create_replay_program();
            builder.record(compute.invoke_replay(
                shape,
                Buffers {
                    num_accepted_tokens: &num_accepted_tokens,
                    sampled_token_ids: &sampled_token_ids,
                    anchor_indices: &anchor_indices,
                    anchor_token_ids: &anchor_token_ids,
                    sample_positions: &sample_positions,
                    block_token_ids: &block_token_ids,
                    flat_query_token_indices: &flat_query_token_indices,
                    visible_history_token_ranges: &visible_history_token_ranges,
                    q_token_ranges: &q_token_ranges,
                    cu_sdpa_partial_outputs: &cu_sdpa_partial_outputs,
                    sdpa_map_task_templates: &sdpa_map_task_templates,
                },
            ));
            builder.build()
        };
        for num_active_requests in [1_u32, 4, 3, 2] {
            let num_active_requests = num_active_requests as usize;
            let inactive_requests = num_active_requests..4;
            let inactive_tokens = num_active_requests * 3..12;
            let inactive_ranges = num_active_requests * 6..24;
            let inactive_tasks = num_active_requests * 9..36;
            let inactive_anchor_token_ids =
                anchor_token_ids.read_typed::<i32>(inactive_requests.start, inactive_requests.len());
            let inactive_sample_positions =
                sample_positions.read_typed::<u32>(inactive_requests.start, inactive_requests.len());
            let inactive_block_token_ids =
                block_token_ids.read_typed::<i32>(inactive_tokens.start, inactive_tokens.len());
            let inactive_flat_query_token_indices =
                flat_query_token_indices.read_typed::<u32>(inactive_tokens.start, inactive_tokens.len());
            let inactive_visible_history_token_ranges =
                visible_history_token_ranges.read_typed::<u32>(inactive_ranges.start, inactive_ranges.len());
            let inactive_task_templates =
                sdpa_map_task_templates.read_typed::<u32>(inactive_tasks.start, inactive_tasks.len());
            let expected = reference_transform(
                config,
                &anchor_indices.read_typed::<u32>(0, num_active_requests),
                &num_accepted_tokens.read_typed::<u32>(0, num_active_requests),
                &sampled_token_ids.read_typed::<i32>(0, num_active_requests),
                &q_token_ranges.read_typed::<u32>(0, 8),
                &cu_sdpa_partial_outputs.read_typed::<u32>(0, 5),
            );
            let mut arguments = ReplayArguments::new();
            compute.add_replay_arguments(shape, num_active_requests as u32, &mut arguments);

            stream.submit_replay_with_arguments(&replay, &arguments).wait();

            assert_eq!(
                anchor_token_ids.read_typed::<i32>(0, num_active_requests),
                sampled_token_ids.read_typed::<i32>(0, num_active_requests)
            );
            assert_eq!(
                sample_positions.read_typed::<u32>(0, num_active_requests),
                expected.anchors.iter().map(|anchor| anchor + 1).collect::<Vec<_>>()
            );
            assert_eq!(
                block_token_ids.read_typed::<i32>(0, expected.block_token_ids.len()),
                expected.block_token_ids
            );
            assert_eq!(
                flat_query_token_indices.read_typed::<u32>(0, expected.flat_query_token_indices.len()),
                expected.flat_query_token_indices
            );
            assert_eq!(
                visible_history_token_ranges.read_typed::<u32>(0, expected.visible_history_token_ranges.len()),
                expected.visible_history_token_ranges
            );
            assert_eq!(
                sdpa_map_task_templates.read_typed::<u32>(0, expected.task_templates.len()),
                expected.task_templates
            );
            assert_eq!(
                anchor_token_ids.read_typed::<i32>(inactive_requests.start, inactive_requests.len()),
                inactive_anchor_token_ids
            );
            assert_eq!(
                sample_positions.read_typed::<u32>(inactive_requests.start, inactive_requests.len()),
                inactive_sample_positions
            );
            assert_eq!(
                block_token_ids.read_typed::<i32>(inactive_tokens.start, inactive_tokens.len()),
                inactive_block_token_ids
            );
            assert_eq!(
                flat_query_token_indices.read_typed::<u32>(inactive_tokens.start, inactive_tokens.len()),
                inactive_flat_query_token_indices
            );
            assert_eq!(
                visible_history_token_ranges.read_typed::<u32>(inactive_ranges.start, inactive_ranges.len()),
                inactive_visible_history_token_ranges
            );
            assert_eq!(
                sdpa_map_task_templates.read_typed::<u32>(inactive_tasks.start, inactive_tasks.len()),
                inactive_task_templates
            );
        }
    }
}
