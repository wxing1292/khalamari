use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const GDN_STATE_PAGE_WRITE_SOURCE: &str = include_str!("../metal/gdn_state_page_write.metal");
const GDN_STATE_PAGE_READ_SOURCE: &str = include_str!("../metal/gdn_state_page_read.metal");

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

/// Static batch geometry for GDN state-page I/O.
///
/// Each state-I/O request selects one logical state version, one recurrent physical
/// slot, one convolution physical slot, and its page IDs across every GDN layer.
/// One `ReadThreadBlockTask` or `WriteThreadBlockTask` maps 1:1 to one
/// threadblock and has the complete logical coordinates
/// `{ state_io_request_index, gdn_layer_index, state_kind, page_index_in_state }`.
/// The grid and shape values derive every coordinate, so no Task value,
/// TaskTemplate, or ABI buffer is materialized. `page_id`,
/// `recurrent_state_slot`, and `conv_state_slot` remain data inputs rather than
/// Task coordinates. `state_kind` selects the applicable physical slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub num_gdn_layers: u32,
    pub num_state_slots: u32,
    pub recurrent_state_bytes: u32,
    pub conv_state_bytes: u32,
    pub page_bytes: u32,
}

impl Config {
    pub fn validate(self) {
        assert!(self.num_gdn_layers > 0);
        assert!(self.num_state_slots > 0);
        assert!(self.recurrent_state_bytes > 0);
        assert_eq!(self.recurrent_state_bytes % 16, 0);
        assert!(self.conv_state_bytes > 0);
        assert_eq!(self.conv_state_bytes % 16, 0);
        assert!(self.page_bytes > 0);
        assert_eq!(self.page_bytes % 16, 0);
    }

    pub fn validate_shape(self, shape: Shape) {
        self.validate();
        shape.validate();
        self.num_total_pages(shape);
    }

    pub fn state_slots_bytes(self, shape: Shape) -> usize {
        checked_product(
            "GDN state-slot metadata byte length",
            &[shape.num_total_state_io_requests as usize, size_of::<u32>()],
        )
    }

    fn state_arena_bytes(self, state_bytes: u32) -> usize {
        checked_product(
            "GDN state arena byte length",
            &[
                self.num_gdn_layers as usize,
                self.num_state_slots as usize,
                state_bytes as usize,
            ],
        )
    }

    fn num_total_pages(self, shape: Shape) -> usize {
        let pages_per_layer = self
            .recurrent_state_bytes
            .div_ceil(self.page_bytes)
            .checked_add(self.conv_state_bytes.div_ceil(self.page_bytes))
            .expect("GDN pages per layer must fit u32");
        let num_pages = checked_product(
            "GDN state-page batch count",
            &[
                self.num_gdn_layers as usize,
                shape.num_total_state_io_requests as usize,
                pages_per_layer as usize,
            ],
        );
        assert_u32_count_domain(num_pages, "GDN state-page batch pages");
        num_pages
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_state_io_requests: u32,
}

impl Shape {
    pub fn validate(self) {
        assert!(self.num_total_state_io_requests > 0);
    }
}

#[derive(Clone, Copy)]
pub struct WriteBuffers<'a> {
    pub pages: &'a Buffer,
    pub recurrent_states: &'a Buffer,
    pub conv_states: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub recurrent_state_slots: &'a Buffer,
    pub conv_state_slots: &'a Buffer,
}

pub struct Write {
    config: Config,
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Write {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            constants: KernelConstants::current(),
            kernel: CompiledKernel::new(device, GDN_STATE_PAGE_WRITE_SOURCE, "gdn_state_page_batch_write_bf16"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: Shape,
        num_active_state_io_requests: ReplayU32,
        buffers: WriteBuffers<'a>,
    ) -> WriteInvocation<'a> {
        let invocation = WriteInvocation {
            kernel: self,
            shape,
            num_active_state_io_requests,
            buffers,
        };
        invocation.validate();
        invocation
    }
}

pub struct WriteInvocation<'a> {
    kernel: &'a Write,
    shape: Shape,
    num_active_state_io_requests: ReplayU32,
    buffers: WriteBuffers<'a>,
}

impl Operator for WriteInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.kernel.config;
        recorder.set_kernel(&self.kernel.kernel);
        recorder.set_buffer_write(0, self.buffers.pages, 0);
        recorder.set_buffer_read(1, self.buffers.recurrent_states, 0);
        recorder.set_buffer_read(2, self.buffers.conv_states, 0);
        recorder.set_buffer_read(3, self.buffers.page_ids, 0);
        recorder.set_buffer_read(4, self.buffers.recurrent_state_slots, 0);
        recorder.set_buffer_read(5, self.buffers.conv_state_slots, 0);
        recorder.set_u32(6, config.num_gdn_layers);
        recorder.set_u32(7, config.num_state_slots);
        bind_active_state_io_requests(recorder, 8, self.shape, self.num_active_state_io_requests);
        recorder.set_u32(9, config.recurrent_state_bytes.div_ceil(config.page_bytes));
        recorder.set_u32(10, config.recurrent_state_bytes);
        recorder.set_u32(11, config.conv_state_bytes.div_ceil(config.page_bytes));
        recorder.set_u32(12, config.conv_state_bytes);
        recorder.set_u32(13, config.page_bytes);
        recorder.dispatch_threadblocks(
            (config.num_total_pages(self.shape), 1, 1),
            (self.kernel.constants.thread_block.required_threads as usize, 1, 1),
        );
    }
}

impl WriteInvocation<'_> {
    fn validate(&self) {
        let config = self.kernel.config;
        config.validate_shape(self.shape);
        if let ReplayU32::Fixed(value) = self.num_active_state_io_requests {
            assert_eq!(value, self.shape.num_total_state_io_requests);
        }
        assert!(self.buffers.page_ids.len_bytes() >= config.num_total_pages(self.shape) * size_of::<u32>());
        assert!(self.buffers.recurrent_states.len_bytes() >= config.state_arena_bytes(config.recurrent_state_bytes));
        assert!(self.buffers.conv_states.len_bytes() >= config.state_arena_bytes(config.conv_state_bytes));
        assert!(self.buffers.recurrent_state_slots.len_bytes() >= config.state_slots_bytes(self.shape));
        assert!(self.buffers.conv_state_slots.len_bytes() >= config.state_slots_bytes(self.shape));
    }
}

#[derive(Clone, Copy)]
pub struct ReadBuffers<'a> {
    pub pages: &'a Buffer,
    pub recurrent_states: &'a Buffer,
    pub conv_states: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub recurrent_state_slots: &'a Buffer,
    pub conv_state_slots: &'a Buffer,
}

pub struct Read {
    config: Config,
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Read {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            constants: KernelConstants::current(),
            kernel: CompiledKernel::new(device, GDN_STATE_PAGE_READ_SOURCE, "gdn_state_page_batch_read_bf16"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: Shape,
        num_active_state_io_requests: ReplayU32,
        buffers: ReadBuffers<'a>,
    ) -> ReadInvocation<'a> {
        let invocation = ReadInvocation {
            kernel: self,
            shape,
            num_active_state_io_requests,
            buffers,
        };
        invocation.validate();
        invocation
    }
}

pub struct ReadInvocation<'a> {
    kernel: &'a Read,
    shape: Shape,
    num_active_state_io_requests: ReplayU32,
    buffers: ReadBuffers<'a>,
}

impl Operator for ReadInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.kernel.config;
        recorder.set_kernel(&self.kernel.kernel);
        recorder.set_buffer_read(0, self.buffers.pages, 0);
        recorder.set_buffer_write(1, self.buffers.recurrent_states, 0);
        recorder.set_buffer_write(2, self.buffers.conv_states, 0);
        recorder.set_buffer_read(3, self.buffers.page_ids, 0);
        recorder.set_buffer_read(4, self.buffers.recurrent_state_slots, 0);
        recorder.set_buffer_read(5, self.buffers.conv_state_slots, 0);
        recorder.set_u32(6, config.num_gdn_layers);
        recorder.set_u32(7, config.num_state_slots);
        bind_active_state_io_requests(recorder, 8, self.shape, self.num_active_state_io_requests);
        recorder.set_u32(9, config.recurrent_state_bytes.div_ceil(config.page_bytes));
        recorder.set_u32(10, config.recurrent_state_bytes);
        recorder.set_u32(11, config.conv_state_bytes.div_ceil(config.page_bytes));
        recorder.set_u32(12, config.conv_state_bytes);
        recorder.set_u32(13, config.page_bytes);
        recorder.dispatch_threadblocks(
            (config.num_total_pages(self.shape), 1, 1),
            (self.kernel.constants.thread_block.required_threads as usize, 1, 1),
        );
    }
}

impl ReadInvocation<'_> {
    fn validate(&self) {
        let config = self.kernel.config;
        config.validate_shape(self.shape);
        if let ReplayU32::Fixed(value) = self.num_active_state_io_requests {
            assert_eq!(value, self.shape.num_total_state_io_requests);
        }
        assert!(self.buffers.page_ids.len_bytes() >= config.num_total_pages(self.shape) * size_of::<u32>());
        assert!(self.buffers.recurrent_states.len_bytes() >= config.state_arena_bytes(config.recurrent_state_bytes));
        assert!(self.buffers.conv_states.len_bytes() >= config.state_arena_bytes(config.conv_state_bytes));
        assert!(self.buffers.recurrent_state_slots.len_bytes() >= config.state_slots_bytes(self.shape));
        assert!(self.buffers.conv_state_slots.len_bytes() >= config.state_slots_bytes(self.shape));
    }
}

fn bind_active_state_io_requests(
    recorder: &CommandRecorder<'_>,
    index: usize,
    shape: Shape,
    num_active_state_io_requests: ReplayU32,
) {
    match num_active_state_io_requests {
        ReplayU32::Fixed(value) => recorder.set_u32(index, value),
        ReplayU32::Parameter(key) => {
            recorder.bind_u32(index, key, 1, shape.num_total_state_io_requests);
        },
    }
}

#[cfg(test)]
mod tests {
    use super::Config;
    use super::Read;
    use super::ReadBuffers;
    use super::Shape;
    use super::Write;
    use super::WriteBuffers;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::ReplayU32;
    use crate::metal::Stream;

    const NUM_ACTIVE_STATE_IO_REQUESTS: ReplayParameterKey =
        ReplayParameterKey::new("test.gdn_state_pages.num_active_state_io_requests");

    #[test]
    #[should_panic(expected = "GDN state-page batch pages exceeds the shader u32 count domain")]
    fn test_batch_shape_rejects_shader_count_overflow() {
        Config {
            num_gdn_layers: 1 << 30,
            num_state_slots: 1,
            recurrent_state_bytes: 16,
            conv_state_bytes: 48,
            page_bytes: 16,
        }
        .num_total_pages(Shape {
            num_total_state_io_requests: 1,
        });
    }

    #[test]
    fn test_multi_layer_write_read_preserves_page_layout_and_unselected_slots() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config {
            num_gdn_layers: 2,
            num_state_slots: 3,
            recurrent_state_bytes: 16 * size_of::<u16>() as u32,
            conv_state_bytes: 8 * size_of::<u16>() as u32,
            page_bytes: 32,
        };
        let page_read = Read::new(&device, config);
        let page_write = Write::new(&device, config);
        let shape = Shape {
            num_total_state_io_requests: 2,
        };
        let recurrent_values = (0..96).map(|value| value as u16 + 10).collect::<Vec<_>>();
        let conv_values = (0..48).map(|value| value as u16 + 100).collect::<Vec<_>>();
        let recurrent_source = Buffer::from_slice(&device, &recurrent_values);
        let conv_source = Buffer::from_slice(&device, &conv_values);
        let state_canary = 0x7bcd_u16;
        let page_canary = 0x7def_u16;
        let recurrent_target = Buffer::from_slice(&device, &vec![state_canary; recurrent_values.len()]);
        let conv_target = Buffer::from_slice(&device, &vec![state_canary; conv_values.len()]);
        let pages = Buffer::from_slice(&device, &[page_canary; 9 * 16]);
        let page_ids = Buffer::from_slice(&device, &[1_u32, 3, 5, 7, 2, 4, 6, 8]);
        let recurrent_state_slots = Buffer::from_slice(&device, &[2_u32, 0]);
        let conv_state_slots = Buffer::from_slice(&device, &[1_u32, 0]);

        let mut write = stream.create_replay_program();
        write.record(page_write.invoke(
            shape,
            ReplayU32::Fixed(shape.num_total_state_io_requests),
            WriteBuffers {
                pages: &pages,
                recurrent_states: &recurrent_source,
                conv_states: &conv_source,
                page_ids: &page_ids,
                recurrent_state_slots: &recurrent_state_slots,
                conv_state_slots: &conv_state_slots,
            },
        ));
        stream.submit_replay(&write.build()).wait();

        let expected_pages = [
            vec![page_canary; 16],
            recurrent_values[32..48].to_vec(),
            recurrent_values[0..16].to_vec(),
            [conv_values[8..16].to_vec(), vec![0; 8]].concat(),
            [conv_values[0..8].to_vec(), vec![0; 8]].concat(),
            recurrent_values[80..96].to_vec(),
            recurrent_values[48..64].to_vec(),
            [conv_values[32..40].to_vec(), vec![0; 8]].concat(),
            [conv_values[24..32].to_vec(), vec![0; 8]].concat(),
        ]
        .concat();
        assert_eq!(pages.read_typed::<u16>(0, expected_pages.len()), expected_pages);

        let mut read = stream.create_replay_program();
        read.record(page_read.invoke(
            shape,
            ReplayU32::Fixed(shape.num_total_state_io_requests),
            ReadBuffers {
                pages: &pages,
                recurrent_states: &recurrent_target,
                conv_states: &conv_target,
                page_ids: &page_ids,
                recurrent_state_slots: &recurrent_state_slots,
                conv_state_slots: &conv_state_slots,
            },
        ));
        stream.submit_replay(&read.build()).wait();

        for layer in 0..2 {
            for state_slot in 0..3 {
                let recurrent_start = (layer * 3 + state_slot) * 16;
                let expected = if matches!(state_slot, 0 | 2) {
                    &recurrent_values[recurrent_start..recurrent_start + 16]
                } else {
                    &[state_canary; 16]
                };
                assert_eq!(recurrent_target.read_typed::<u16>(recurrent_start, 16), expected);
            }
            for state_slot in 0..3 {
                let conv_start = (layer * 3 + state_slot) * 8;
                let expected = if matches!(state_slot, 0 | 1) {
                    &conv_values[conv_start..conv_start + 8]
                } else {
                    &[state_canary; 8]
                };
                assert_eq!(conv_target.read_typed::<u16>(conv_start, 8), expected);
            }
        }
    }

    #[test]
    fn test_write_replay_matches_page_reference_across_active_counts() {
        const NUM_TOTAL_REQUESTS: u32 = 8;
        const VALUES_PER_PAGE: usize = 8;
        const PAGE_CANARY: u16 = 0x7def;
        const ACTIVE_COUNTS: [u32; 8] = [1, 8, 3, 7, 2, 6, 4, 5];

        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config {
            num_gdn_layers: 1,
            num_state_slots: NUM_TOTAL_REQUESTS,
            recurrent_state_bytes: (VALUES_PER_PAGE * size_of::<u16>()) as u32,
            conv_state_bytes: (VALUES_PER_PAGE * size_of::<u16>()) as u32,
            page_bytes: (VALUES_PER_PAGE * size_of::<u16>()) as u32,
        };
        let shape = Shape {
            num_total_state_io_requests: NUM_TOTAL_REQUESTS,
        };
        let state_values = NUM_TOTAL_REQUESTS as usize * VALUES_PER_PAGE;
        let recurrent_values = (0..state_values).map(|index| index as u16 + 1).collect::<Vec<_>>();
        let conv_values = (0..state_values).map(|index| index as u16 + 101).collect::<Vec<_>>();
        let recurrent_states = Buffer::from_slice(&device, &recurrent_values);
        let conv_states = Buffer::from_slice(&device, &conv_values);
        let page_ids = Buffer::from_slice(&device, &(0..NUM_TOTAL_REQUESTS * 2).collect::<Vec<_>>());
        let state_slots = Buffer::from_slice(&device, &(0..NUM_TOTAL_REQUESTS).collect::<Vec<_>>());
        let page_values = NUM_TOTAL_REQUESTS as usize * 2 * VALUES_PER_PAGE;
        let pages = Buffer::from_slice(&device, &vec![PAGE_CANARY; page_values]);
        let write = Write::new(&device, config);
        let replay = {
            let mut builder = stream.create_replay_program();
            builder.record(write.invoke(
                shape,
                ReplayU32::Parameter(NUM_ACTIVE_STATE_IO_REQUESTS),
                WriteBuffers {
                    pages: &pages,
                    recurrent_states: &recurrent_states,
                    conv_states: &conv_states,
                    page_ids: &page_ids,
                    recurrent_state_slots: &state_slots,
                    conv_state_slots: &state_slots,
                },
            ));
            builder.build()
        };

        for num_active_requests in ACTIVE_COUNTS {
            pages.write_typed(0, &vec![PAGE_CANARY; page_values]);
            let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_STATE_IO_REQUESTS, num_active_requests);
            stream.submit_replay_with_arguments(&replay, &arguments).wait();

            let mut expected = vec![PAGE_CANARY; page_values];
            for request_index in 0..num_active_requests as usize {
                let state_begin = request_index * VALUES_PER_PAGE;
                let page_begin = request_index * 2 * VALUES_PER_PAGE;
                expected[page_begin..page_begin + VALUES_PER_PAGE]
                    .copy_from_slice(&recurrent_values[state_begin..state_begin + VALUES_PER_PAGE]);
                expected[page_begin + VALUES_PER_PAGE..page_begin + 2 * VALUES_PER_PAGE]
                    .copy_from_slice(&conv_values[state_begin..state_begin + VALUES_PER_PAGE]);
            }
            assert_eq!(pages.read_typed::<u16>(0, page_values), expected);
        }
    }

    #[test]
    fn test_read_replay_matches_state_reference_across_active_counts() {
        const NUM_TOTAL_REQUESTS: u32 = 8;
        const VALUES_PER_PAGE: usize = 8;
        const CANARY: u16 = 0x7bcd;
        const ACTIVE_COUNTS: [u32; 8] = [1, 8, 3, 7, 2, 6, 4, 5];

        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config {
            num_gdn_layers: 1,
            num_state_slots: NUM_TOTAL_REQUESTS,
            recurrent_state_bytes: (VALUES_PER_PAGE * size_of::<u16>()) as u32,
            conv_state_bytes: (VALUES_PER_PAGE * size_of::<u16>()) as u32,
            page_bytes: (VALUES_PER_PAGE * size_of::<u16>()) as u32,
        };
        let shape = Shape {
            num_total_state_io_requests: NUM_TOTAL_REQUESTS,
        };
        let page_values = (0..NUM_TOTAL_REQUESTS as usize * 2 * VALUES_PER_PAGE)
            .map(|index| index as u16 + 1)
            .collect::<Vec<_>>();
        let pages = Buffer::from_slice(&device, &page_values);
        let page_ids = Buffer::from_slice(&device, &(0..NUM_TOTAL_REQUESTS * 2).collect::<Vec<_>>());
        let state_slots = Buffer::from_slice(&device, &(0..NUM_TOTAL_REQUESTS).collect::<Vec<_>>());
        let state_values = NUM_TOTAL_REQUESTS as usize * VALUES_PER_PAGE;
        let recurrent_states = Buffer::from_slice(&device, &vec![CANARY; state_values]);
        let conv_states = Buffer::from_slice(&device, &vec![CANARY; state_values]);
        let read = Read::new(&device, config);
        let replay = {
            let mut builder = stream.create_replay_program();
            builder.record(read.invoke(
                shape,
                ReplayU32::Parameter(NUM_ACTIVE_STATE_IO_REQUESTS),
                ReadBuffers {
                    pages: &pages,
                    recurrent_states: &recurrent_states,
                    conv_states: &conv_states,
                    page_ids: &page_ids,
                    recurrent_state_slots: &state_slots,
                    conv_state_slots: &state_slots,
                },
            ));
            builder.build()
        };

        for num_active_requests in ACTIVE_COUNTS {
            recurrent_states.write_typed(0, &vec![CANARY; state_values]);
            conv_states.write_typed(0, &vec![CANARY; state_values]);
            let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_STATE_IO_REQUESTS, num_active_requests);
            stream.submit_replay_with_arguments(&replay, &arguments).wait();

            let mut expected_recurrent = vec![CANARY; state_values];
            let mut expected_conv = vec![CANARY; state_values];
            for request_index in 0..num_active_requests as usize {
                let state_begin = request_index * VALUES_PER_PAGE;
                let recurrent_page_begin = request_index * 2 * VALUES_PER_PAGE;
                let conv_page_begin = recurrent_page_begin + VALUES_PER_PAGE;
                expected_recurrent[state_begin..state_begin + VALUES_PER_PAGE]
                    .copy_from_slice(&page_values[recurrent_page_begin..recurrent_page_begin + VALUES_PER_PAGE]);
                expected_conv[state_begin..state_begin + VALUES_PER_PAGE]
                    .copy_from_slice(&page_values[conv_page_begin..conv_page_begin + VALUES_PER_PAGE]);
            }
            assert_eq!(recurrent_states.read_typed::<u16>(0, state_values), expected_recurrent);
            assert_eq!(conv_states.read_typed::<u16>(0, state_values), expected_conv);
        }
    }
}
