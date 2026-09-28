use std::cell::Cell;
use std::cell::RefCell;
use std::ptr::NonNull;
use std::rc::Rc;
use std::time::Duration;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTL4CommandAllocator;
use objc2_metal::MTL4CommandBuffer;
use objc2_metal::MTL4CommandEncoder;
use objc2_metal::MTL4CommandQueue;
use objc2_metal::MTL4ComputeCommandEncoder;
use objc2_metal::MTL4VisibilityOptions;
use objc2_metal::MTLDevice;
use objc2_metal::MTLStages;

use crate::metal::stream::CommitCompletion;
use crate::metal::stream::ReplayArguments;
use crate::metal::stream::ReplayProgram;
use crate::metal::stream::Stream;
use crate::metal::stream::replay::ReplayResources;
use crate::metal::stream::replay::assert_replay_submission_stream;
use crate::metal::stream::replay::encode_replay;
use crate::metal::stream::replay::validate_replay_arguments;
use crate::metal::stream::timestamp::SubmissionTimestamps;

#[derive(Clone, Copy, Debug)]
pub struct ReplayExecution<'a> {
    program: &'a ReplayProgram,
    arguments: &'a ReplayArguments,
}

impl<'a> ReplayExecution<'a> {
    pub fn new(program: &'a ReplayProgram, arguments: &'a ReplayArguments) -> Self {
        Self { program, arguments }
    }
}

#[derive(Debug)]
pub struct ReplaySubmission {
    allocator_in_flight: Rc<Cell<bool>>,
    completion: Rc<CommitCompletion>,
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    resources: Vec<Rc<ReplayResources>>,
    _queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    _command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    timestamps: Option<SubmissionTimestamps>,
    timestamp_durations: RefCell<Option<Vec<Duration>>>,
    waited: Cell<bool>,
}

impl ReplaySubmission {
    fn submit(
        stream: &Stream,
        command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
        resources: Vec<Rc<ReplayResources>>,
        timestamps: Option<SubmissionTimestamps>,
    ) -> Self {
        command_buffer.endCommandBuffer();
        let mut command_buffer_ptr = NonNull::new(Retained::as_ptr(&command_buffer).cast_mut())
            .expect("MTL4CommandBuffer pointer must not be null");
        unsafe {
            // Metal consumes this registration with the commit. CommitCompletion retains the block and options.
            stream
                .completion
                .options
                .addFeedbackHandler(RcBlock::as_ptr(&stream.completion.handler));
            stream
                .queue
                .commit_count_options(NonNull::from(&mut command_buffer_ptr), 1, &stream.completion.options);
        }

        Self {
            allocator_in_flight: stream.allocator_in_flight.clone(),
            completion: stream.completion.clone(),
            allocator: stream.allocator.clone(),
            resources,
            _queue: stream.queue.clone(),
            _command_buffer: command_buffer,
            timestamps,
            timestamp_durations: RefCell::new(None),
            waited: Cell::new(false),
        }
    }

    pub fn wait(&self) {
        if self.waited.replace(true) {
            return;
        }
        self.completion.wait();
        if let Some(timestamps) = &self.timestamps {
            *self.timestamp_durations.borrow_mut() = timestamps.resolve();
        }
        self.allocator.reset();
        self.allocator_in_flight.set(false);
        for resources in &self.resources {
            resources.assert_success();
        }
    }

    pub fn gpu_timestamp_durations(&self) -> Option<Vec<Duration>> {
        self.timestamp_durations.borrow().clone()
    }
}

impl Drop for ReplaySubmission {
    fn drop(&mut self) {
        self.wait();
    }
}

pub fn submit_replay_sequence(
    stream: &Stream,
    executions: &[ReplayExecution<'_>],
    timestamp_stage_end_indices: Option<&[usize]>,
) -> ReplaySubmission {
    assert!(
        !executions.is_empty(),
        "Metal replay sequence requires at least one execution"
    );
    for (index, execution) in executions.iter().enumerate() {
        assert_replay_submission_stream(execution.program, stream);
        validate_replay_arguments(execution.program, execution.arguments);
        for previous in &executions[..index] {
            if std::ptr::eq(previous.program, execution.program) {
                assert_eq!(
                    previous.arguments, execution.arguments,
                    "Metal replay sequence cannot execute the same program with conflicting arguments because replay \
                     arguments are stored in the program-owned parameter buffer"
                );
            }
        }
    }
    assert!(
        !stream.allocator_in_flight.replace(true),
        "Metal stream command allocator already has an in-flight submission; wait before submitting again"
    );
    let command_buffer = stream
        .device
        .newCommandBuffer()
        .expect("MTL4CommandBuffer allocation failed");
    command_buffer.beginCommandBufferWithAllocator(&stream.allocator);
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("MTL4ComputeCommandEncoder allocation failed");
    let timestamps = timestamp_stage_end_indices.and_then(|stage_end_indices| {
        stream
            .timestamp_profiler
            .as_ref()
            .and_then(|profiler| profiler.begin(stage_end_indices.len() + 1))
    });
    if let Some(timestamps) = &timestamps {
        timestamps.write(&encoder, 0);
    }

    let resources = if let Some(timestamps) = &timestamps {
        let stage_end_indices = timestamp_stage_end_indices.expect("timestamp submission requires stage ends");
        let mut resources = Vec::with_capacity(executions.len());
        let mut next_timestamp_stage = 0usize;
        for (index, execution) in executions.iter().enumerate() {
            resources.push(encode_execution(&encoder, index, execution));
            if stage_end_indices.get(next_timestamp_stage).copied() == Some(index + 1) {
                timestamps.write(&encoder, next_timestamp_stage + 1);
                next_timestamp_stage += 1;
            }
        }
        resources
    } else {
        executions
            .iter()
            .enumerate()
            .map(|(index, execution)| encode_execution(&encoder, index, execution))
            .collect()
    };
    encoder.endEncoding();
    ReplaySubmission::submit(stream, command_buffer, resources, timestamps)
}

fn encode_execution(
    encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    index: usize,
    execution: &ReplayExecution<'_>,
) -> Rc<ReplayResources> {
    if index > 0 {
        encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
            MTLStages::Dispatch,
            MTLStages::Dispatch,
            MTL4VisibilityOptions::None,
        );
    }
    encode_replay(execution.program, encoder, execution.arguments)
}
