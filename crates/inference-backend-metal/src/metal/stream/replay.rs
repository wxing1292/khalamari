use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::MTL4ComputeCommandEncoder;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLComputePipelineState;
use objc2_metal::MTLDevice;
use objc2_metal::MTLIndirectCommandBuffer;
use objc2_metal::MTLIndirectCommandBufferDescriptor;
use objc2_metal::MTLIndirectCommandType;
use objc2_metal::MTLIndirectComputeCommand;
use objc2_metal::MTLResourceOptions;
use objc2_metal::MTLSize;

use crate::metal::record_gpu_buffer_alloc;
use crate::metal::stream::MAX_BUFFER_BINDINGS;
use crate::metal::stream::Operator;
use crate::metal::stream::Stream;
use crate::metal::stream::TrackedGpuAllocation;
use crate::metal::stream::check::SubmissionCheck;
use crate::metal::stream::dependency::CommandDependencyTracker;
use crate::metal::stream::operation::CommandBinding;
use crate::metal::stream::operation::CommandDispatch;
use crate::metal::stream::operation::CommandMetadata;
use crate::metal::stream::operation::record_operator;
use crate::metal::stream::operation::record_operator_with_barrier_before;
use crate::metal::stream::parameter::CommandParameterLayoutBuilder;
use crate::metal::stream::parameter::ReplayArguments;
use crate::metal::stream::parameter::ReplayParameterTable;
use crate::metal::stream::parameter::allocate_parameter_buffer;
use crate::metal::stream::residency::Residency;
use crate::metal::stream::residency::ResidencySet;
use crate::metal::stream::residency::retain_buffer_once;
use crate::metal::stream::residency::retain_pipeline_once;

#[derive(Debug)]
pub struct ReplayProgramBuilder {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    residency_set: Rc<ResidencySet>,
    parameters: CommandParameterLayoutBuilder,
    commands: Vec<CommandMetadata>,
}

impl ReplayProgramBuilder {
    pub fn new(stream: &Stream) -> Self {
        Self {
            device: stream.device.clone(),
            residency_set: stream.residency_set.clone(),
            parameters: CommandParameterLayoutBuilder::default(),
            commands: Vec::new(),
        }
    }

    pub fn record<I: Operator>(&mut self, operator: I) {
        let recorded = record_operator(&self.parameters, operator);
        self.commands.extend(recorded.commands);
    }

    /// Records a consumer whose ICB command waits for all prior commands.
    pub fn record_with_barrier_before<I: Operator>(&mut self, operator: I) {
        if self.commands.is_empty() {
            self.record(operator);
            return;
        }
        let recorded = record_operator_with_barrier_before(&self.parameters, operator);
        self.commands.extend(recorded.commands);
    }

    pub fn build(self) -> ReplayProgram {
        let parameters = self.parameters.build();
        ReplayProgram::new(
            &self.device,
            &self.residency_set,
            self.commands,
            parameters.bytes,
            parameters.replay_parameter_table,
        )
    }
}

#[derive(Debug)]
pub struct ReplayProgram {
    stats: ReplayProgramStats,
    parameter_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    parameter_table: ReplayParameterTable,
    resources: Rc<ReplayResources>,
}

#[derive(Debug)]
pub struct ReplayResources {
    // Fields drop in declaration order: unregister before releasing the last Metal handles.
    residency: Rc<Residency>,
    icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    retained_buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    retained_pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    parameter_buffer_allocation: Option<TrackedGpuAllocation>,
    icb_allocation: TrackedGpuAllocation,
    checks: Vec<Rc<dyn SubmissionCheck>>,
}

impl ReplayResources {
    pub fn assert_success(&self) {
        for check in &self.checks {
            check.assert_success();
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplayProgramStats {
    pub command_count: usize,
    pub retained_buffer_count: usize,
    pub retained_pipeline_count: usize,
    pub parameter_buffer_bytes: usize,
    pub parameter_count: usize,
}

impl ReplayProgram {
    fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        residency_set: &Rc<ResidencySet>,
        commands: Vec<CommandMetadata>,
        parameter_bytes: Vec<u8>,
        parameter_table: ReplayParameterTable,
    ) -> Self {
        assert!(
            !commands.is_empty(),
            "cannot build Metal replay program without commands"
        );
        let parameter_buffer = allocate_parameter_buffer(device, &parameter_bytes, "replay_parameter_buffer");
        let (parameter_buffer, parameter_buffer_allocation) = match parameter_buffer {
            Some((buffer, allocation)) => (Some(buffer), Some(allocation)),
            None => (None, None),
        };

        let descriptor = MTLIndirectCommandBufferDescriptor::new();
        descriptor.setCommandTypes(
            MTLIndirectCommandType::ConcurrentDispatch | MTLIndirectCommandType::ConcurrentDispatchThreads,
        );
        descriptor.setInheritPipelineState(false);
        descriptor.setInheritBuffers(false);
        descriptor.setMaxKernelBufferBindCount(MAX_BUFFER_BINDINGS);
        unsafe {
            descriptor.setMaxKernelThreadgroupMemoryBindCount(MAX_BUFFER_BINDINGS);
        }
        let icb = unsafe {
            device
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &descriptor,
                    commands.len(),
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModePrivate,
                )
                .expect("MTLIndirectCommandBuffer allocation failed")
        };
        let icb_allocation_site = record_gpu_buffer_alloc("indirect_command_buffer", 0);
        unsafe {
            icb.resetWithRange(NSRange {
                location: 0,
                length: commands.len(),
            });
        }

        let mut retained_buffers = Vec::new();
        let mut retained_pipelines = Vec::new();
        let mut dependencies = CommandDependencyTracker::default();
        let mut checks = Vec::new();
        for (command_index, command_metadata) in commands.iter().enumerate() {
            if let Some(check) = &command_metadata.submission_check {
                checks.push(check.clone());
            }
            let command = unsafe { icb.indirectComputeCommandAtIndex(command_index) };
            command.setComputePipelineState(&command_metadata.pipeline);
            retain_pipeline_once(&mut retained_pipelines, &command_metadata.pipeline);
            for (binding_index, binding) in command_metadata.bindings.iter().enumerate() {
                let Some(binding) = binding else {
                    continue;
                };
                match binding {
                    CommandBinding::Buffer {
                        buffer, offset_bytes, ..
                    } => unsafe {
                        command.setKernelBuffer_offset_atIndex(buffer, *offset_bytes, binding_index);
                        retain_buffer_once(&mut retained_buffers, buffer);
                    },
                    CommandBinding::Parameter { offset_bytes } => {
                        let buffer = parameter_buffer
                            .as_ref()
                            .expect("parameter binding requires replay parameter buffer");
                        unsafe {
                            command.setKernelBuffer_offset_atIndex(buffer, *offset_bytes, binding_index);
                        }
                    },
                }
            }
            for (index, len_bytes) in command_metadata.threadblock_memory_lengths.iter().enumerate() {
                let Some(len_bytes) = len_bytes else {
                    continue;
                };
                unsafe {
                    command.setThreadgroupMemoryLength_atIndex(*len_bytes, index);
                }
            }
            if dependencies.barrier_before(command_metadata) {
                command.setBarrier();
            }
            match command_metadata.dispatch {
                CommandDispatch::Threads {
                    num_total_threads,
                    num_threads_per_threadblock,
                } => {
                    command.concurrentDispatchThreads_threadsPerThreadgroup(
                        MTLSize {
                            width: num_total_threads.0,
                            height: num_total_threads.1,
                            depth: num_total_threads.2,
                        },
                        MTLSize {
                            width: num_threads_per_threadblock.0,
                            height: num_threads_per_threadblock.1,
                            depth: num_threads_per_threadblock.2,
                        },
                    );
                },
                CommandDispatch::Threadblocks {
                    num_threadblocks,
                    num_threads_per_threadblock,
                } => {
                    command.concurrentDispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: num_threadblocks.0,
                            height: num_threadblocks.1,
                            depth: num_threadblocks.2,
                        },
                        MTLSize {
                            width: num_threads_per_threadblock.0,
                            height: num_threads_per_threadblock.1,
                            depth: num_threads_per_threadblock.2,
                        },
                    )
                },
            }
        }

        let stats = ReplayProgramStats {
            command_count: commands.len(),
            retained_buffer_count: retained_buffers.len(),
            retained_pipeline_count: retained_pipelines.len(),
            parameter_buffer_bytes: parameter_buffer.as_ref().map(|buffer| buffer.length()).unwrap_or(0),
            parameter_count: parameter_table.len(),
        };
        let retained_parameter_buffer = parameter_buffer.clone();
        if let Some(buffer) = parameter_buffer {
            retained_buffers.push(buffer);
        }
        let residency = residency_set.register(&retained_buffers, &retained_pipelines, &icb);
        let resources = Rc::new(ReplayResources {
            residency,
            icb,
            retained_buffers,
            retained_pipelines,
            parameter_buffer_allocation,
            checks,
            icb_allocation: TrackedGpuAllocation::new(icb_allocation_site, 0),
        });

        Self {
            stats,
            parameter_buffer: retained_parameter_buffer,
            parameter_table,
            resources,
        }
    }

    pub fn command_count(&self) -> usize {
        self.stats.command_count
    }

    pub fn stats(&self) -> ReplayProgramStats {
        self.stats
    }

    fn write_arguments(&self, arguments: &ReplayArguments) {
        if self.parameter_table.is_empty() {
            return;
        }
        let buffer = self
            .parameter_buffer
            .as_ref()
            .expect("Metal replay arguments require a parameter buffer");
        self.parameter_table.write(buffer, arguments);
    }
}

pub fn validate_replay_arguments(program: &ReplayProgram, arguments: &ReplayArguments) {
    program.parameter_table.validate(arguments);
}

pub fn encode_replay(
    program: &ReplayProgram,
    encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    arguments: &ReplayArguments,
) -> Rc<ReplayResources> {
    program.write_arguments(arguments);
    // Encoding finishes before commit. Errors remain sticky across repeated
    // executions of this program in the same submission.
    for check in &program.resources.checks {
        check.reset();
    }
    // MTL4 executeCommandsInBuffer needs a current compute pipeline on the
    // encoder even though the ICB commands carry their own pipelines. The ICB
    // descriptor still keeps inheritPipelineState=false; this encoder state
    // only enables the MTL4 ICB execution path.
    encoder.setComputePipelineState(&program.resources.retained_pipelines[0]);
    unsafe {
        encoder.executeCommandsInBuffer_withRange(
            &program.resources.icb,
            NSRange {
                location: 0,
                length: program.stats.command_count,
            },
        );
    }
    program.resources.clone()
}

pub fn assert_replay_submission_stream(program: &ReplayProgram, stream: &Stream) {
    assert!(
        program.resources.residency.belongs_to(&stream.residency_set),
        "Metal replay program must be submitted on the stream that recorded it"
    );
}
