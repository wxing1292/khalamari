use std::cell::RefCell;
use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLComputePipelineState;
use objc2_metal::MTLResourceUsage;

use crate::metal::Buffer;
use crate::metal::CompiledKernel;
use crate::metal::stream::MAX_BUFFER_BINDINGS;
use crate::metal::stream::check::SubmissionCheck;
use crate::metal::stream::parameter::CommandParameterLayoutBuilder;
use crate::metal::stream::parameter::ReplayParameterKey;

/// Recordable backend execution unit.
///
/// Operators bind kernels, buffers, constants, resource usage, and dispatch
/// shape into a `CommandRecorder`. They do not own model stage order or
/// request semantics.
pub trait Operator {
    fn record(self, recorder: &CommandRecorder<'_>);
}

#[derive(Debug)]
pub struct CommandRecorder<'a> {
    parameters: &'a CommandParameterLayoutBuilder,
    active: RefCell<Option<CommandMetadataBuilder>>,
    completed: RefCell<Vec<CommandMetadata>>,
}

impl<'a> CommandRecorder<'a> {
    fn new(parameters: &'a CommandParameterLayoutBuilder) -> Self {
        CommandRecorder {
            parameters,
            active: RefCell::new(None),
            completed: RefCell::new(Vec::new()),
        }
    }

    pub fn set_kernel(&self, kernel: &CompiledKernel) {
        assert!(
            self.active.borrow().is_none(),
            "previous Metal command must dispatch before setting another kernel"
        );
        *self.active.borrow_mut() = Some(CommandMetadataBuilder::new(kernel.as_raw_retained()));
    }

    pub fn set_retained_pipeline_state(&self, pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>) {
        assert!(
            self.active.borrow().is_none(),
            "previous Metal command must dispatch before setting another kernel"
        );
        *self.active.borrow_mut() = Some(CommandMetadataBuilder::new(pipeline.clone()));
    }

    /// Records an operator whose first command waits for all earlier buffer accesses.
    pub fn record_with_barrier_before<I: Operator>(&self, operator: I) {
        assert!(
            self.active.borrow().is_none(),
            "cannot record a barrier consumer while another Metal command is active"
        );
        let first_command_index = self.completed.borrow().len();
        operator.record(self);
        assert!(
            self.completed.borrow().len() > first_command_index,
            "Metal barrier consumer must record at least one command"
        );
        self.completed.borrow_mut()[first_command_index].barrier_before = true;
    }

    /// Declares disjoint accesses to the selected buffers across the recorded commands.
    ///
    /// For every submission, an element written by one command must not be read
    /// or written by another command in this scope. The regions may be dynamic
    /// or scattered. Other buffers and explicit barriers retain their dependencies.
    /// The declaration does not extend to commands outside this recording scope.
    pub fn record_disjoint_buffers(&self, buffers: &[&Buffer], record: impl FnOnce()) {
        assert!(
            self.active.borrow().is_none(),
            "disjoint buffer recording must start between commands"
        );
        let first_command_index = self.completed.borrow().len();
        record();
        assert!(
            self.active.borrow().is_none(),
            "disjoint buffer recording must finish between commands"
        );

        let scope = Rc::new(());
        for (command_index, command) in self.completed.borrow_mut()[first_command_index..]
            .iter_mut()
            .enumerate()
        {
            for binding in command.bindings.iter_mut().flatten() {
                let CommandBinding::Buffer { buffer, partition, .. } = binding else {
                    continue;
                };
                if buffers
                    .iter()
                    .any(|selected| std::ptr::eq(&**buffer, selected.as_raw()))
                {
                    assert!(partition.is_none(), "disjoint buffer recording scopes must not overlap");
                    *partition = Some(BufferAccessPartition {
                        scope: scope.clone(),
                        command_index,
                    });
                }
            }
        }
    }

    /// Makes the active consumer command wait for all earlier buffer accesses.
    pub fn set_barrier_before(&self) {
        let mut active = self.active.borrow_mut();
        let command = active
            .as_mut()
            .expect("Metal command must set a kernel before setting its barrier");
        assert!(!command.barrier_before, "Metal command barrier was set twice");
        command.barrier_before = true;
    }

    /// Retains the component's invariant check until submission completion.
    pub fn set_submission_check(&self, check: Rc<dyn SubmissionCheck>) {
        let mut active = self.active.borrow_mut();
        let active = active.as_mut().expect("submission check requires an active command");
        assert!(
            active.submission_check.is_none(),
            "Metal command already has a submission check"
        );
        active.submission_check = Some(check);
    }

    pub fn set_buffer_read(&self, index: usize, buffer: &Buffer, offset_bytes: usize) {
        self.set_buffer_with_usage(index, buffer, offset_bytes, MTLResourceUsage::Read);
    }

    pub fn set_buffer_write(&self, index: usize, buffer: &Buffer, offset_bytes: usize) {
        self.set_buffer_with_usage(index, buffer, offset_bytes, MTLResourceUsage::Write);
    }

    pub fn set_buffer_read_write(&self, index: usize, buffer: &Buffer, offset_bytes: usize) {
        self.set_buffer_with_usage(
            index,
            buffer,
            offset_bytes,
            MTLResourceUsage::Read | MTLResourceUsage::Write,
        );
    }

    pub fn set_retained_buffer_read(
        &self,
        index: usize,
        buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        offset_bytes: usize,
    ) {
        self.set_retained_buffer_with_usage(index, buffer, offset_bytes, MTLResourceUsage::Read);
    }

    pub fn set_retained_buffer_write(
        &self,
        index: usize,
        buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        offset_bytes: usize,
    ) {
        self.set_retained_buffer_with_usage(index, buffer, offset_bytes, MTLResourceUsage::Write);
    }

    fn set_buffer_with_usage(&self, index: usize, buffer: &Buffer, offset_bytes: usize, usage: MTLResourceUsage) {
        assert!(index < MAX_BUFFER_BINDINGS);
        assert_icb_buffer_binding(buffer.len_bytes_u64(), offset_bytes as u64);
        let mut active = self.active.borrow_mut();
        let active = active
            .as_mut()
            .expect("Metal command must set a kernel before binding buffers");
        active.set_binding(
            index,
            CommandBinding::Buffer {
                buffer: buffer.as_raw_retained(),
                offset_bytes,
                usage,
                partition: None,
            },
        );
    }

    fn set_retained_buffer_with_usage(
        &self,
        index: usize,
        buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        offset_bytes: usize,
        usage: MTLResourceUsage,
    ) {
        assert!(index < MAX_BUFFER_BINDINGS);
        assert_icb_buffer_binding(buffer.length() as u64, offset_bytes as u64);
        let mut active = self.active.borrow_mut();
        let active = active
            .as_mut()
            .expect("Metal command must set a kernel before binding buffers");
        active.set_binding(
            index,
            CommandBinding::Buffer {
                buffer: buffer.clone(),
                offset_bytes,
                usage,
                partition: None,
            },
        );
    }

    /// Sets a `u32` kernel argument to a value fixed while recording.
    pub fn set_u32(&self, index: usize, value: u32) {
        self.set_bytes(index, std::slice::from_ref(&value));
    }

    /// Sets a `u64`/Metal `ulong` kernel argument fixed while recording.
    pub fn set_u64(&self, index: usize, value: u64) {
        self.set_bytes(index, std::slice::from_ref(&value));
    }

    /// Binds a `u32` kernel argument to a replay parameter key supplied at submission.
    pub fn bind_u32(&self, index: usize, key: ReplayParameterKey, min_value: u32, max_value: u32) {
        self.bind_parameter(index, self.parameters.bind_u32(key, min_value, max_value));
    }

    /// Binds a `u64` kernel argument to a replay parameter key supplied at submission.
    pub fn bind_u64(&self, index: usize, key: ReplayParameterKey, min_value: u64, max_value: u64) {
        self.bind_parameter(index, self.parameters.bind_u64(key, min_value, max_value));
    }

    /// Binds an `i32` kernel argument to a replay parameter key supplied at submission.
    pub fn bind_i32(&self, index: usize, key: ReplayParameterKey, min_value: i32, max_value: i32) {
        self.bind_parameter(index, self.parameters.bind_i32(key, min_value, max_value));
    }

    /// Binds an `i64` kernel argument to a replay parameter key supplied at submission.
    pub fn bind_i64(&self, index: usize, key: ReplayParameterKey, min_value: i64, max_value: i64) {
        self.bind_parameter(index, self.parameters.bind_i64(key, min_value, max_value));
    }

    /// Binds an `f32` kernel argument to a replay parameter key supplied at submission.
    pub fn bind_f32(&self, index: usize, key: ReplayParameterKey, min_value: f32, max_value: f32) {
        self.bind_parameter(index, self.parameters.bind_f32(key, min_value, max_value));
    }

    fn bind_parameter(&self, index: usize, offset_bytes: usize) {
        assert!(index < MAX_BUFFER_BINDINGS);
        let mut command = self.active.borrow_mut();
        let command = command
            .as_mut()
            .expect("Metal command must set a kernel before binding replay parameters");
        command.set_binding(index, CommandBinding::Parameter { offset_bytes });
    }

    pub fn set_i32(&self, index: usize, value: i32) {
        self.set_bytes(index, std::slice::from_ref(&value));
    }

    pub fn set_i64(&self, index: usize, value: i64) {
        self.set_bytes(index, std::slice::from_ref(&value));
    }

    pub fn set_u32_slice(&self, index: usize, values: &[u32]) {
        assert!(!values.is_empty());
        self.set_bytes(index, values);
    }

    pub fn set_i32_slice(&self, index: usize, values: &[i32]) {
        assert!(!values.is_empty());
        self.set_bytes(index, values);
    }

    pub fn set_i64_slice(&self, index: usize, values: &[i64]) {
        assert!(!values.is_empty());
        self.set_bytes(index, values);
    }

    pub fn set_f32(&self, index: usize, value: f32) {
        self.set_bytes(index, std::slice::from_ref(&value));
    }

    fn set_bytes<T>(&self, index: usize, values: &[T]) {
        assert!(index < MAX_BUFFER_BINDINGS);
        let len_bytes = std::mem::size_of_val(values);
        assert!(len_bytes > 0);

        let mut active = self.active.borrow_mut();
        let active = active
            .as_mut()
            .expect("Metal command must set a kernel before binding constants");
        let offset_bytes = self.parameters.push_bytes(values);
        active.set_binding(index, CommandBinding::Parameter { offset_bytes });
    }

    /// Records a `dispatchThreads` grid. The fixed total may be smaller than one threadblock.
    pub fn dispatch_1d(&self, num_total_threads: usize, num_threads_per_threadblock: usize) {
        assert!(num_total_threads > 0);
        assert!(num_threads_per_threadblock > 0);
        self.dispatch(CommandDispatch::Threads {
            num_total_threads: (num_total_threads, 1, 1),
            num_threads_per_threadblock: (num_threads_per_threadblock, 1, 1),
        });
    }

    /// Records a `dispatchThreadgroups` grid using project-level threadblock terminology.
    pub fn dispatch_threadblocks(
        &self,
        num_threadblocks: (usize, usize, usize),
        num_threads_per_threadblock: (usize, usize, usize),
    ) {
        assert!(num_threadblocks.0 > 0);
        assert!(num_threadblocks.1 > 0);
        assert!(num_threadblocks.2 > 0);
        assert!(num_threads_per_threadblock.0 > 0);
        assert!(num_threads_per_threadblock.1 > 0);
        assert!(num_threads_per_threadblock.2 > 0);
        self.dispatch(CommandDispatch::Threadblocks {
            num_threadblocks,
            num_threads_per_threadblock,
        });
    }

    pub fn set_threadblock_memory_length(&self, index: usize, len_bytes: usize) {
        assert!(index < MAX_BUFFER_BINDINGS);
        assert!(len_bytes > 0);
        let mut active = self.active.borrow_mut();
        let active = active
            .as_mut()
            .expect("Metal command must set a kernel before binding threadblock memory");
        active.set_threadblock_memory_length(index, len_bytes);
    }

    fn dispatch(&self, dispatch: CommandDispatch) {
        self.active
            .borrow_mut()
            .as_mut()
            .expect("Metal command must set a kernel before dispatch")
            .dispatch = Some(dispatch);
        self.finish_active();
    }

    fn finish_active(&self) {
        let active = self
            .active
            .borrow_mut()
            .take()
            .expect("Metal command must be active before finishing");
        self.completed.borrow_mut().push(active.build());
    }

    fn finish(self) -> RecordedCommands {
        assert!(
            self.active.borrow().is_none(),
            "Metal command missing dispatch before recorder finish"
        );
        RecordedCommands {
            commands: self.completed.into_inner(),
        }
    }
}

fn assert_icb_buffer_binding(buffer_len_bytes: u64, offset_bytes: u64) {
    assert!(
        offset_bytes <= u32::MAX as u64,
        "Metal ICB kernel buffer binding offset exceeds the verified 32-bit range: offset_bytes={offset_bytes}; bind \
         a smaller resource or use a zero-offset resource view"
    );
    assert!(
        offset_bytes <= buffer_len_bytes,
        "Metal kernel buffer binding offset exceeds buffer length: offset_bytes={offset_bytes} \
         buffer_len_bytes={buffer_len_bytes}"
    );
}

pub fn record_operator<I: Operator>(parameters: &CommandParameterLayoutBuilder, operator: I) -> RecordedCommands {
    let recorder = CommandRecorder::new(parameters);
    operator.record(&recorder);
    recorder.finish()
}

pub fn record_operator_with_barrier_before<I: Operator>(
    parameters: &CommandParameterLayoutBuilder,
    operator: I,
) -> RecordedCommands {
    let recorder = CommandRecorder::new(parameters);
    recorder.record_with_barrier_before(operator);
    recorder.finish()
}

#[derive(Debug)]
pub struct RecordedCommands {
    pub commands: Vec<CommandMetadata>,
}

#[derive(Clone, Debug)]
struct CommandMetadataBuilder {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    bindings: Vec<Option<CommandBinding>>,
    threadblock_memory_lengths: Vec<Option<usize>>,
    dispatch: Option<CommandDispatch>,
    barrier_before: bool,
    submission_check: Option<Rc<dyn SubmissionCheck>>,
}

impl CommandMetadataBuilder {
    fn new(pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>) -> Self {
        Self {
            pipeline,
            bindings: vec![None; MAX_BUFFER_BINDINGS],
            threadblock_memory_lengths: vec![None; MAX_BUFFER_BINDINGS],
            dispatch: None,
            barrier_before: false,
            submission_check: None,
        }
    }

    fn set_binding(&mut self, index: usize, binding: CommandBinding) {
        assert!(index < MAX_BUFFER_BINDINGS);
        self.bindings[index] = Some(binding);
    }

    fn set_threadblock_memory_length(&mut self, index: usize, len_bytes: usize) {
        assert!(index < MAX_BUFFER_BINDINGS);
        assert!(len_bytes > 0);
        self.threadblock_memory_lengths[index] = Some(len_bytes);
    }

    fn build(self) -> CommandMetadata {
        CommandMetadata {
            pipeline: self.pipeline,
            bindings: self.bindings,
            threadblock_memory_lengths: self.threadblock_memory_lengths,
            dispatch: self.dispatch.expect("recorded Metal command missing dispatch"),
            barrier_before: self.barrier_before,
            submission_check: self.submission_check,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CommandMetadata {
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub bindings: Vec<Option<CommandBinding>>,
    pub threadblock_memory_lengths: Vec<Option<usize>>,
    pub dispatch: CommandDispatch,
    pub barrier_before: bool,
    pub submission_check: Option<Rc<dyn SubmissionCheck>>,
}

#[derive(Clone, Debug)]
pub enum CommandBinding {
    Buffer {
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        offset_bytes: usize,
        usage: MTLResourceUsage,
        partition: Option<BufferAccessPartition>,
    },
    Parameter {
        offset_bytes: usize,
    },
}

/// A command's access domain within one disjoint-buffer recording scope.
#[derive(Clone, Debug)]
pub struct BufferAccessPartition {
    scope: Rc<()>,
    command_index: usize,
}

impl BufferAccessPartition {
    pub fn is_disjoint_from(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.scope, &other.scope) && self.command_index != other.command_index
    }
}

impl PartialEq for BufferAccessPartition {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.scope, &other.scope) && self.command_index == other.command_index
    }
}

impl Eq for BufferAccessPartition {}

#[derive(Clone, Copy, Debug)]
pub enum CommandDispatch {
    Threads {
        num_total_threads: (usize, usize, usize),
        num_threads_per_threadblock: (usize, usize, usize),
    },
    Threadblocks {
        num_threadblocks: (usize, usize, usize),
        num_threads_per_threadblock: (usize, usize, usize),
    },
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::CommandParameterLayoutBuilder;
    use super::CommandRecorder;
    use super::Operator;
    use super::assert_icb_buffer_binding;
    use crate::metal::Buffer;
    use crate::metal::CompiledKernel;
    use crate::metal::Device;

    const NOOP_SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void noop(device uint *values [[buffer(0)]], uint id [[thread_position_in_grid]]) {
            values[id] = values[id];
        }
    "#;

    struct NoopInvocation<'a> {
        kernel: &'a CompiledKernel,
        values: &'a Buffer,
    }

    impl Operator for NoopInvocation<'_> {
        fn record(self, recorder: &CommandRecorder<'_>) {
            recorder.set_kernel(self.kernel);
            recorder.set_buffer_read_write(0, self.values, 0);
            recorder.dispatch_1d(1, 1);
        }
    }

    #[test]
    fn test_consumer_barrier() {
        let device = Device::system_default();
        let kernel = CompiledKernel::new(&device, NOOP_SOURCE, "noop");
        let values = Buffer::new_zeroed(&device, size_of::<u32>());

        let sequence_layout = CommandParameterLayoutBuilder::default();
        let sequence = CommandRecorder::new(&sequence_layout);
        NoopInvocation {
            kernel: &kernel,
            values: &values,
        }
        .record(&sequence);
        sequence.record_with_barrier_before(NoopInvocation {
            kernel: &kernel,
            values: &values,
        });
        let sequence = sequence.finish();
        assert_eq!(sequence.commands.len(), 2);
        assert!(!sequence.commands[0].barrier_before);
        assert!(sequence.commands[1].barrier_before);

        let consumer_layout = CommandParameterLayoutBuilder::default();
        let consumer = CommandRecorder::new(&consumer_layout);
        consumer.record_with_barrier_before(NoopInvocation {
            kernel: &kernel,
            values: &values,
        });
        let consumer = consumer.finish();
        assert_eq!(consumer.commands.len(), 1);
        assert!(consumer.commands[0].barrier_before);
    }

    #[test]
    #[should_panic(expected = "Metal ICB kernel buffer binding offset exceeds the verified 32-bit range")]
    fn test_kernel_buffer_binding_offset_domain() {
        assert_icb_buffer_binding(u64::MAX, u32::MAX as u64 + 1);
    }
}
