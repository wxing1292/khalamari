use std::hint::black_box;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::affine_quantized;

const VOCAB_SIZE: i32 = 151_936;
const TOKENS: [i32; 9] = [1, 5, 6, 7, 8, 14, 16, 21, 28];
const PROFILES: [Profile; 2] = [Profile::Qwen27, Profile::Qwen35];
const MATMUL_PATHS: [MatmulPath; 5] = [
    MatmulPath::Auto,
    MatmulPath::QmvBn8Bk32,
    MatmulPath::QmmBm8Bn32,
    MatmulPath::QmmBm16Bn32,
    MatmulPath::QmmBm32Bn32,
];

fn bench_unembedding(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/unembedding");
    for profile in PROFILES {
        for tokens in TOKENS {
            for path in MATMUL_PATHS {
                let fixture = UnembeddingFixture::new(&device, profile, tokens, path);
                group.throughput(Throughput::Elements(tokens as u64 * VOCAB_SIZE as u64));
                group.bench_function(format!("{}/{}/tokens{tokens}", profile.key(), path.key()), |b| {
                    b.iter(|| {
                        fixture.run();
                        black_box(&fixture.logits);
                    });
                });
            }
        }
    }
    group.finish();
}

#[derive(Clone, Copy)]
enum MatmulPath {
    Auto,
    QmvBn8Bk32,
    QmmBm8Bn32,
    QmmBm16Bn32,
    QmmBm32Bn32,
}

impl MatmulPath {
    fn key(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::QmvBn8Bk32 => "qmv-bn8-bk32",
            Self::QmmBm8Bn32 => "qmm-bm8-bn32",
            Self::QmmBm16Bn32 => "qmm-bm16-bn32",
            Self::QmmBm32Bn32 => "qmm-bm32-bn32",
        }
    }
}

#[derive(Clone, Copy)]
enum Profile {
    Qwen27,
    Qwen35,
}

impl Profile {
    fn key(self) -> &'static str {
        match self {
            Self::Qwen27 => "qwen27",
            Self::Qwen35 => "qwen35",
        }
    }

    fn hidden_dim(self) -> i32 {
        match self {
            Self::Qwen27 => 5120,
            Self::Qwen35 => 2048,
        }
    }

    fn bits(self) -> i32 {
        match self {
            Self::Qwen27 | Self::Qwen35 => 4,
        }
    }
}

struct UnembeddingFixture {
    stream: Stream,
    replay: ReplayProgram,
    logits: Buffer,
}

impl UnembeddingFixture {
    fn new(device: &Device, profile: Profile, tokens: i32, path: MatmulPath) -> Self {
        let config = affine_quantized::Config {
            n: VOCAB_SIZE,
            k: profile.hidden_dim(),
            group_size: 64,
            bits: profile.bits(),
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            scale_bias_dtype: Dtype::Bfloat16,
        };
        let hidden = Buffer::new_zeroed(device, config.input_bytes(tokens));
        let weight = Buffer::new_zeroed(device, config.weight_bytes());
        let scales = Buffer::new_zeroed(device, config.scale_or_bias_bytes());
        let biases = Buffer::new_zeroed(device, config.scale_or_bias_bytes());
        let logits = Buffer::new_zeroed(device, config.output_bytes(tokens));
        let stream = Stream::new(device);
        let mut builder = stream.create_replay_program();
        match path {
            MatmulPath::Auto => {
                let matmul = affine_quantized::Matmul::new(device, config);
                builder.record(matmul.invoke(
                    tokens as u32,
                    ReplayU32::Fixed(tokens as u32),
                    &logits,
                    0,
                    &hidden,
                    0,
                    &weight,
                    0,
                    &scales,
                    0,
                    &biases,
                    0,
                ));
            },
            MatmulPath::QmvBn8Bk32 => {
                let kernel = affine_quantized::Kernel::new(device, config, affine_quantized::KernelKind::QmvBn8Bk32);
                builder.record(kernel.invoke(
                    tokens as u32,
                    ReplayU32::Fixed(tokens as u32),
                    &logits,
                    0,
                    &hidden,
                    0,
                    &weight,
                    0,
                    &scales,
                    0,
                    &biases,
                    0,
                ));
            },
            MatmulPath::QmmBm8Bn32 => {
                let kernel = affine_quantized::Kernel::new(device, config, affine_quantized::KernelKind::QmmBm8Bn32);
                builder.record(kernel.invoke(
                    tokens as u32,
                    ReplayU32::Fixed(tokens as u32),
                    &logits,
                    0,
                    &hidden,
                    0,
                    &weight,
                    0,
                    &scales,
                    0,
                    &biases,
                    0,
                ));
            },
            MatmulPath::QmmBm16Bn32 => {
                let kernel = affine_quantized::Kernel::new(device, config, affine_quantized::KernelKind::QmmBm16Bn32);
                builder.record(kernel.invoke(
                    tokens as u32,
                    ReplayU32::Fixed(tokens as u32),
                    &logits,
                    0,
                    &hidden,
                    0,
                    &weight,
                    0,
                    &scales,
                    0,
                    &biases,
                    0,
                ));
            },
            MatmulPath::QmmBm32Bn32 => {
                let kernel = affine_quantized::Kernel::new(device, config, affine_quantized::KernelKind::QmmBm32Bn32);
                builder.record(kernel.invoke(
                    tokens as u32,
                    ReplayU32::Fixed(tokens as u32),
                    &logits,
                    0,
                    &hidden,
                    0,
                    &weight,
                    0,
                    &scales,
                    0,
                    &biases,
                    0,
                ));
            },
        }
        let replay = builder.build();
        let fixture = Self { stream, replay, logits };
        fixture.run();
        fixture
    }

    fn run(&self) {
        self.stream.submit_replay(&self.replay).wait();
    }
}

criterion_group!(benches, bench_unembedding);
criterion_main!(benches);
