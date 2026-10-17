use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("../metal/gqa_kv_page_write.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants {
    config: Config,
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn current(config: Config) -> Self {
        Self {
            config,
            thread_block: ThreadBlockConstants { required_threads: 256 },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageTableLayout {
    pub num_req_slots: u32,
    pub num_gqa_layers: u32,
    pub num_blocks: u32,
    pub num_page_ids_per_block: u32,
}

impl PageTableLayout {
    pub fn validate(self) {
        assert!(self.num_req_slots > 0);
        assert!(self.num_blocks > 0);
        assert!(self.num_gqa_layers > 0);
        assert!(self.num_page_ids_per_block > 0);
    }

    pub fn bytes(self) -> usize {
        (self.num_req_slots as usize)
            .checked_mul(self.num_gqa_layers as usize)
            .and_then(|count| count.checked_mul(self.num_blocks as usize))
            .and_then(|count| count.checked_mul(self.num_page_ids_per_block as usize))
            .and_then(|count| count.checked_mul(size_of::<u32>()))
            .expect("GQA page-table byte length must fit usize")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_bytes: u32,
    pub dtype: Dtype,
}

impl Config {
    pub fn validate(self) {
        assert!(self.num_kv_heads > 0);
        assert!(self.num_tokens_per_page() > 0);
        assert!(self.head_dim > 0);
        assert_eq!(
            self.dtype,
            Dtype::Bfloat16,
            "GQA FP8 KV page write requires BF16 source tensors"
        );
    }

    pub fn num_tokens_per_page(self) -> u32 {
        // KV[2][num_kv_heads][num_tokens_per_page][head_dim], where 0 is K and 1 is V.
        let kv_bytes_per_token = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| bytes.checked_mul(size_of::<u8>().try_into().expect("FP8 size must fit u32")))
            .expect("GQA K/V bytes per token must fit u32");
        assert!(
            self.page_bytes.is_multiple_of(kv_bytes_per_token),
            "GQA page_bytes must be divisible by the K/V bytes per token"
        );
        self.page_bytes / kv_bytes_per_token
    }

    pub fn index_bytes(self, shape: Shape) -> usize {
        (shape.num_total_token_writes as usize)
            .checked_mul(size_of::<u32>())
            .expect("GQA KV page-write index bytes must fit usize")
    }

    pub fn flat_kv_bytes(self, shape: Shape) -> usize {
        self.num_total_threads(shape)
            .checked_mul(self.dtype.item_size())
            .expect("GQA flattened K/V byte length must fit usize")
    }

    pub fn num_total_threads(self, shape: Shape) -> usize {
        checked_product(
            "GQA KV page-write thread count",
            &[
                shape.num_total_token_writes as usize,
                self.num_kv_heads as usize,
                self.head_dim as usize,
            ],
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub num_total_token_writes: u32,
    pub page_table_layout: PageTableLayout,
}

impl Shape {
    pub fn validate(self, config: Config) {
        config.validate();
        assert!(self.num_total_token_writes > 0);
        self.page_table_layout.validate();
        assert_u32_count_domain(config.num_total_threads(self), "GQA KV page-write threads");
    }

    pub fn page_ids_bytes(self) -> usize {
        self.page_table_layout.bytes()
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub pages: &'a Buffer,
    pub flat_k: &'a Buffer,
    pub flat_v: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub flat_token_indices: &'a Buffer,
    pub page_ids: &'a Buffer,
}

pub struct Compute {
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        let constants = KernelConstants::current(config);
        let source = kv_page_write_source(constants);
        Self {
            constants,
            kernel: CompiledKernel::new(device, &source, "gqa_kv_page_write_fp8_e4m3fn"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: Shape,
        buffers: Buffers<'a>,
        num_active_token_writes: ReplayU32,
        page_table_index: ReplayU32,
    ) -> Invocation<'a> {
        let invocation = Invocation {
            constants: self.constants,
            kernel: &self.kernel,
            shape,
            buffers,
            num_active_token_writes,
            page_table_index,
        };
        invocation.validate();
        invocation
    }
}

fn kv_page_write_source(kernel_constants: KernelConstants) -> String {
    let config = kernel_constants.config;
    let source_constants = format!(
        "using namespace metal;\n\nconstant uint num_kv_heads = {}u;\nconstant uint head_dim = {}u;\nconstant uint \
         num_tokens_per_page = {}u;\nconstant uint page_bytes = {}u;",
        config.num_kv_heads,
        config.head_dim,
        config.num_tokens_per_page(),
        config.page_bytes,
    );
    SOURCE.replacen("using namespace metal;", &source_constants, 1)
}

pub struct Invocation<'a> {
    constants: KernelConstants,
    kernel: &'a CompiledKernel,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_token_writes: ReplayU32,
    page_table_index: ReplayU32,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.constants.config;
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_write(0, self.buffers.pages, 0);
        recorder.set_buffer_read(1, self.buffers.flat_k, 0);
        recorder.set_buffer_read(2, self.buffers.flat_v, 0);
        recorder.set_buffer_read(3, self.buffers.req_slots, 0);
        recorder.set_buffer_read(4, self.buffers.flat_token_indices, 0);
        recorder.set_buffer_read(5, self.buffers.page_ids, 0);
        match self.num_active_token_writes {
            ReplayU32::Fixed(num_active_token_writes) => recorder.set_u32(6, num_active_token_writes),
            ReplayU32::Parameter(key) => recorder.bind_u32(6, key, 1, self.shape.num_total_token_writes),
        }
        let max_page_table_index = self.shape.page_table_layout.num_gqa_layers - 1;
        match self.page_table_index {
            ReplayU32::Fixed(page_table_index) => recorder.set_u32(7, page_table_index),
            ReplayU32::Parameter(key) => recorder.bind_u32(7, key, 0, max_page_table_index),
        }
        recorder.set_u32(8, self.shape.page_table_layout.num_gqa_layers);
        recorder.set_u32(9, self.shape.page_table_layout.num_blocks);
        recorder.set_u32(10, self.shape.page_table_layout.num_page_ids_per_block);
        recorder.dispatch_1d(
            config.num_total_threads(self.shape),
            self.constants.thread_block.required_threads as usize,
        );
    }
}

impl Invocation<'_> {
    fn validate(&self) {
        let config = self.constants.config;
        self.shape.validate(config);
        self.num_active_token_writes
            .validate_range(1, self.shape.num_total_token_writes);
        self.page_table_index
            .validate_range(0, self.shape.page_table_layout.num_gqa_layers - 1);
        assert!(self.buffers.flat_k.len_bytes() >= config.flat_kv_bytes(self.shape));
        assert!(self.buffers.flat_v.len_bytes() >= config.flat_kv_bytes(self.shape));
        assert!(self.buffers.req_slots.len_bytes() >= config.index_bytes(self.shape));
        assert!(self.buffers.flat_token_indices.len_bytes() >= config.index_bytes(self.shape));
        assert!(self.buffers.page_ids.len_bytes() >= self.shape.page_ids_bytes());
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;
    use crate::components::gqa::fp8::bf16_to_fp8_e4m3;
    use crate::components::gqa::fp8::f32_to_bf16;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::Stream;

    const NUM_ACTIVE_WRITES: ReplayParameterKey = ReplayParameterKey::new("test.gqa.kv.active_writes");

    #[test]
    #[should_panic(expected = "GQA KV page-write threads exceeds the shader u32 count domain")]
    fn test_shape_rejects_shader_count_overflow() {
        let config = Config {
            num_kv_heads: 2,
            head_dim: 2,
            page_bytes: 16,
            dtype: Dtype::Bfloat16,
        };
        Shape {
            num_total_token_writes: 1 << 30,
            page_table_layout: PageTableLayout {
                num_req_slots: 1,
                num_gqa_layers: 1,
                num_blocks: 1,
                num_page_ids_per_block: 1,
            },
        }
        .validate(config);
    }

    #[test]
    fn test_replay_matches_page_reference_across_active_counts() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let page_values = 16_usize;
        let num_pages = 4_usize;
        let initial_page_value = 0x55_u8;
        let flat_k_values = (0..16).map(|index| 1.0 + index as f32 * 0.125).collect::<Vec<_>>();
        let flat_v_values = (0..16).map(|index| -1.0 - index as f32 * 0.125).collect::<Vec<_>>();
        let req_slot_values = [0_u32, 1, 0, 1, 0, 1, 0, 1];
        let token_index_values = [0_u32, 0, 1, 1, 2, 2, 3, 3];
        let page_id_values = [0_u32, 1, 2, 3];
        let pages = Buffer::from_slice(&device, &vec![initial_page_value; num_pages * page_values]);
        let flat_k = bf16_buffer(&device, &flat_k_values);
        let flat_v = bf16_buffer(&device, &flat_v_values);
        let req_slots = Buffer::from_slice(&device, &req_slot_values);
        let flat_token_indices = Buffer::from_slice(&device, &token_index_values);
        let page_ids = Buffer::from_slice(&device, &page_id_values);
        let config = Config {
            num_kv_heads: 1,
            head_dim: 2,
            page_bytes: page_values as u32,
            dtype: Dtype::Bfloat16,
        };
        let shape = Shape {
            num_total_token_writes: 8,
            page_table_layout: PageTableLayout {
                num_req_slots: 2,
                num_gqa_layers: 1,
                num_blocks: 2,
                num_page_ids_per_block: 1,
            },
        };
        let kernel = Compute::new(&device, config);
        let replay = {
            let mut builder = stream.create_replay_program();
            builder.record(kernel.invoke(
                shape,
                Buffers {
                    pages: &pages,
                    flat_k: &flat_k,
                    flat_v: &flat_v,
                    req_slots: &req_slots,
                    flat_token_indices: &flat_token_indices,
                    page_ids: &page_ids,
                },
                ReplayU32::Parameter(NUM_ACTIVE_WRITES),
                ReplayU32::Fixed(0),
            ));
            builder.build()
        };

        for num_active_writes in [1_usize, 8, 3, 7, 2, 6, 4, 5] {
            pages.write_typed(0, &vec![initial_page_value; num_pages * page_values]);
            stream
                .submit_replay_with_arguments(
                    &replay,
                    &ReplayArguments::new().with_u32(NUM_ACTIVE_WRITES, num_active_writes as u32),
                )
                .wait();

            let mut expected = vec![initial_page_value; num_pages * page_values];
            let num_tokens_per_page = config.num_tokens_per_page() as usize;
            let kv_values_per_page = config.num_kv_heads as usize * num_tokens_per_page * config.head_dim as usize;
            for write_index in 0..num_active_writes {
                let req_slot = req_slot_values[write_index] as usize;
                let token_index = token_index_values[write_index] as usize;
                let block_index = token_index / num_tokens_per_page;
                let page_id =
                    page_id_values[req_slot * shape.page_table_layout.num_blocks as usize + block_index] as usize;
                let page_token_index = token_index % num_tokens_per_page;
                for dim_index in 0..config.head_dim as usize {
                    let page_token_offset = page_token_index * config.head_dim as usize + dim_index;
                    let flat_offset = write_index * config.head_dim as usize + dim_index;
                    expected[page_id * page_values + page_token_offset] =
                        bf16_to_fp8_e4m3(f32_to_bf16(flat_k_values[flat_offset]));
                    expected[page_id * page_values + kv_values_per_page + page_token_offset] =
                        bf16_to_fp8_e4m3(f32_to_bf16(flat_v_values[flat_offset]));
                }
            }
            assert_eq!(pages.read_typed::<u8>(0, expected.len()), expected);
        }
    }

    #[test]
    fn test_fp8_replay_matches_page_reference() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let k_values = [
            -0.0,
            2.0_f32.powi(-10),
            2.0_f32.powi(-9),
            1.0625,
            1.1875,
            400.0,
            448.0,
            449.0,
            f32::INFINITY,
        ];
        let v_values = k_values.map(|value| -value);
        let mut expected = vec![0; 4 * k_values.len()];
        for (index, value) in k_values.into_iter().enumerate() {
            expected[k_values.len() + index] = bf16_to_fp8_e4m3(f32_to_bf16(value));
        }
        for (index, value) in v_values.into_iter().enumerate() {
            expected[3 * k_values.len() + index] = bf16_to_fp8_e4m3(f32_to_bf16(value));
        }
        let pages = Buffer::new_zeroed(&device, expected.len());
        let k = bf16_buffer(&device, &k_values);
        let v = bf16_buffer(&device, &v_values);
        run(&device, &stream, &pages, &k, &v);
        assert_eq!(pages.read_typed::<u8>(0, expected.len()), expected);
    }

    fn run(device: &Device, stream: &Stream, pages: &Buffer, k: &Buffer, v: &Buffer) {
        let req_slots = Buffer::from_slice(device, &[0u32]);
        let flat_token_indices = Buffer::from_slice(device, &[1u32]);
        let page_ids = Buffer::from_slice(device, &[0u32]);
        let config = Config {
            num_kv_heads: 1,
            head_dim: (k.len_bytes() / size_of::<u16>()).try_into().unwrap(),
            page_bytes: pages.len_bytes() as u32,
            dtype: Dtype::Bfloat16,
        };
        let shape = Shape {
            num_total_token_writes: 1,
            page_table_layout: PageTableLayout {
                num_req_slots: 1,
                num_gqa_layers: 1,
                num_blocks: 1,
                num_page_ids_per_block: 1,
            },
        };
        let kernel = Compute::new(device, config);
        let replay = {
            let mut recorder = stream.create_replay_program();
            recorder.record(kernel.invoke(
                shape,
                Buffers {
                    pages,
                    flat_k: k,
                    flat_v: v,
                    req_slots: &req_slots,
                    flat_token_indices: &flat_token_indices,
                    page_ids: &page_ids,
                },
                ReplayU32::Parameter(NUM_ACTIVE_WRITES),
                ReplayU32::Fixed(0),
            ));
            recorder.build()
        };
        stream
            .submit_replay_with_arguments(
                &replay,
                &ReplayArguments::new().with_u32(NUM_ACTIVE_WRITES, shape.num_total_token_writes),
            )
            .wait();
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        Buffer::from_slice(
            device,
            &values
                .iter()
                .map(|&value| bf16::from_f32(value).to_bits())
                .collect::<Vec<_>>(),
        )
    }
}
