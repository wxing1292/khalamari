use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("../metal/gqa_qkv_split.metal");

#[derive(Clone, Copy, Debug, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct KernelConstants {
    config: Config,
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn current(config: Config) -> Self {
        config.validate();
        Self {
            config,
            thread_block: ThreadBlockConstants { required_threads: 256 },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub dtype: Dtype,
}

impl Config {
    pub fn f32(num_q_heads: u32, num_kv_heads: u32, head_dim: u32) -> Self {
        Self {
            num_q_heads,
            num_kv_heads,
            head_dim,
            dtype: Dtype::Float32,
        }
    }

    pub fn bf16(num_q_heads: u32, num_kv_heads: u32, head_dim: u32) -> Self {
        Self {
            num_q_heads,
            num_kv_heads,
            head_dim,
            dtype: Dtype::Bfloat16,
        }
    }

    pub fn validate(self) {
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert!(self.head_dim > 0);
        assert!(matches!(self.dtype, Dtype::Float32 | Dtype::Bfloat16));
        let _ = self.qkv_width();
    }

    pub fn num_qkv_slots(self, shape: Shape) -> usize {
        checked_product(
            "ungated GQA projection element count",
            &[shape.num_total_tokens as usize, self.qkv_width()],
        )
    }

    pub fn num_q_slots(self, shape: Shape) -> usize {
        checked_product(
            "ungated GQA query element count",
            &[
                shape.num_total_tokens as usize,
                self.num_q_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    pub fn num_kv_slots(self, shape: Shape) -> usize {
        checked_product(
            "ungated GQA key/value element count",
            &[
                shape.num_total_tokens as usize,
                self.num_kv_heads as usize,
                self.head_dim as usize,
            ],
        )
    }

    pub fn qkv_bytes(self, shape: Shape) -> usize {
        checked_product(
            "ungated GQA projection byte length",
            &[self.num_qkv_slots(shape), self.dtype.item_size()],
        )
    }

    pub fn q_bytes(self, shape: Shape) -> usize {
        checked_product(
            "ungated GQA query byte length",
            &[self.num_q_slots(shape), self.dtype.item_size()],
        )
    }

    pub fn kv_bytes(self, shape: Shape) -> usize {
        checked_product(
            "ungated GQA key/value byte length",
            &[self.num_kv_slots(shape), self.dtype.item_size()],
        )
    }

    pub fn qkv_width(self) -> usize {
        let num_qkv_heads = self
            .num_kv_heads
            .checked_mul(2)
            .and_then(|num_kv_heads| self.num_q_heads.checked_add(num_kv_heads))
            .expect("ungated GQA fused projection head count must fit u32");
        checked_product(
            "ungated GQA fused projection width",
            &[num_qkv_heads as usize, self.head_dim as usize],
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub num_total_tokens: u32,
}

impl Shape {
    pub fn validate(self, config: Config) {
        config.validate();
        assert!(self.num_total_tokens > 0);
        assert_u32_count_domain(config.num_qkv_slots(self), "ungated GQA projection elements");
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub qkv: &'a Buffer,
    pub q: &'a Buffer,
    pub k: &'a Buffer,
    pub v: &'a Buffer,
}

pub struct Compute {
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        let constants = KernelConstants::current(config);
        let source = qkv_split_source(constants);
        let function_name = match config.dtype {
            Dtype::Float32 => "gqa_qkv_split_f32",
            Dtype::Bfloat16 => "gqa_qkv_split_bf16",
            dtype => panic!("unsupported ungated GQA projection split dtype {dtype:?}"),
        };
        Self {
            constants,
            kernel: CompiledKernel::new(device, &source, function_name),
        }
    }

    pub fn invoke<'a>(&'a self, shape: Shape, buffers: Buffers<'a>, num_active_tokens: ReplayU32) -> Invocation<'a> {
        let invocation = Invocation {
            constants: self.constants,
            kernel: &self.kernel,
            shape,
            buffers,
            num_active_tokens,
        };
        invocation.validate();
        invocation
    }
}

fn qkv_split_source(constants: KernelConstants) -> String {
    let config = constants.config;
    let constants = format!(
        "using namespace metal;\n\nconstant uint num_q_heads = {}u;\nconstant uint num_kv_heads = {}u;\nconstant uint \
         head_dim = {}u;",
        config.num_q_heads, config.num_kv_heads, config.head_dim,
    );
    SOURCE.replacen("using namespace metal;", &constants, 1)
}

pub struct Invocation<'a> {
    constants: KernelConstants,
    kernel: &'a CompiledKernel,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_tokens: ReplayU32,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let shape = self.shape;
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.buffers.qkv, 0);
        recorder.set_buffer_write(1, self.buffers.q, 0);
        recorder.set_buffer_write(2, self.buffers.k, 0);
        recorder.set_buffer_write(3, self.buffers.v, 0);
        match self.num_active_tokens {
            ReplayU32::Fixed(value) => recorder.set_u32(4, value),
            ReplayU32::Parameter(key) => recorder.bind_u32(4, key, 1, shape.num_total_tokens),
        }
        recorder.dispatch_1d(
            self.constants.config.num_qkv_slots(shape),
            self.constants.thread_block.required_threads as usize,
        );
    }
}

impl Invocation<'_> {
    fn validate(&self) {
        let config = self.constants.config;
        self.shape.validate(config);
        self.num_active_tokens.validate_range(1, self.shape.num_total_tokens);
        assert!(self.buffers.qkv.len_bytes() >= config.qkv_bytes(self.shape));
        assert!(self.buffers.q.len_bytes() >= config.q_bytes(self.shape));
        assert!(self.buffers.k.len_bytes() >= config.kv_bytes(self.shape));
        assert!(self.buffers.v.len_bytes() >= config.kv_bytes(self.shape));
    }
}

#[cfg(test)]
mod tests {
    use super::Buffers;
    use super::Compute;
    use super::Config;
    use super::Shape;
    use crate::metal::Buffer;
    use crate::metal::Device;
    use crate::metal::Dtype;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::ReplayU32;
    use crate::metal::Stream;

    const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.gqa_qkv_split.num_active_tokens");

    #[test]
    fn test_replay_matches_reference_across_active_counts() {
        const NUM_TOTAL_TOKENS: u32 = 8;
        const ACTIVE_COUNTS: [u32; 8] = [1, 8, 3, 7, 2, 6, 4, 5];

        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config::f32(2, 1, 2);
        let shape = Shape {
            num_total_tokens: NUM_TOTAL_TOKENS,
        };
        let qkv_width = config.qkv_width();
        let q_width = config.num_q_heads as usize * config.head_dim as usize;
        let kv_width = config.num_kv_heads as usize * config.head_dim as usize;
        let qkv_values = (0..NUM_TOTAL_TOKENS as usize * qkv_width)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let qkv = Buffer::from_slice(&device, &qkv_values);
        let q = Buffer::new_zeroed_elements(&device, config.num_q_slots(shape), Dtype::Float32);
        let k = Buffer::new_zeroed_elements(&device, config.num_kv_slots(shape), Dtype::Float32);
        let v = Buffer::new_zeroed_elements(&device, config.num_kv_slots(shape), Dtype::Float32);
        let kernel = Compute::new(&device, config);
        let replay = {
            let mut builder = stream.create_replay_program();
            builder.record(kernel.invoke(
                shape,
                Buffers {
                    qkv: &qkv,
                    q: &q,
                    k: &k,
                    v: &v,
                },
                ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            ));
            builder.build()
        };

        for num_active_tokens in ACTIVE_COUNTS {
            let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens);
            stream.submit_replay_with_arguments(&replay, &arguments).wait();
            let active_tokens = num_active_tokens as usize;
            let mut expected_q = Vec::with_capacity(active_tokens * q_width);
            let mut expected_k = Vec::with_capacity(active_tokens * kv_width);
            let mut expected_v = Vec::with_capacity(active_tokens * kv_width);
            for token_index in 0..active_tokens {
                let qkv_token = &qkv_values[token_index * qkv_width..(token_index + 1) * qkv_width];
                expected_q.extend_from_slice(&qkv_token[..q_width]);
                expected_k.extend_from_slice(&qkv_token[q_width..q_width + kv_width]);
                expected_v.extend_from_slice(&qkv_token[q_width + kv_width..]);
            }
            assert_eq!(q.read_typed::<f32>(0, expected_q.len()), expected_q);
            assert_eq!(k.read_typed::<f32>(0, expected_k.len()), expected_k);
            assert_eq!(v.read_typed::<f32>(0, expected_v.len()), expected_v);
        }
    }

    #[test]
    #[should_panic(expected = "ungated GQA projection elements exceeds the shader u32 count domain")]
    fn test_shape_rejects_shader_count_overflow() {
        Shape {
            num_total_tokens: 1 << 30,
        }
        .validate(Config::f32(2, 1, 1));
    }
}
