use std::hint::black_box;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::affine_quantized;

const ROWS: [i32; 18] = [1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 24, 32, 33, 40, 48, 56, 64, 65];
const PROFILES: [Profile; 8] = [
    Profile {
        name: "bf16-square4k",
        n: 4_096,
        k: 4_096,
        input_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
    },
    Profile {
        name: "bf16-square8k",
        n: 8_192,
        k: 8_192,
        input_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
    },
    Profile {
        name: "bf16-wide12k",
        n: 12_288,
        k: 5_120,
        input_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
    },
    Profile {
        name: "bf16-wide14k",
        n: 14_336,
        k: 5_120,
        input_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
    },
    Profile {
        name: "bf16-wide16k",
        n: 16_384,
        k: 4_096,
        input_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
    },
    Profile {
        name: "bf16-wide32k",
        n: 32_768,
        k: 4_096,
        input_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
    },
    Profile {
        name: "bf16-bf16-bf16",
        n: 5_120,
        k: 5_120,
        input_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
    },
    Profile {
        name: "f32-bf16-f32",
        n: 6_144,
        k: 2_048,
        input_dtype: Dtype::Float32,
        scale_bias_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Float32,
    },
];
const KERNELS: [BenchKernel; 6] = [
    BenchKernel::Auto,
    BenchKernel::Exact(affine_quantized::KernelKind::QmvBn8Bk32),
    BenchKernel::Exact(affine_quantized::KernelKind::QmmBm8Bn32),
    BenchKernel::Exact(affine_quantized::KernelKind::QmmBm16Bn32),
    BenchKernel::Exact(affine_quantized::KernelKind::QmmBm32Bn32),
    BenchKernel::Exact(affine_quantized::KernelKind::QmmBm32Bn64),
];

fn bench_affine_quantized_matmul(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/affine-quantized-matmul");

    for profile in PROFILES {
        for rows in ROWS {
            for kernel in KERNELS {
                let fixture = MatmulFixture::new(&device, profile, rows, kernel);
                group.throughput(Throughput::Elements(rows as u64 * profile.n as u64));
                group.bench_function(format!("{}/rows{rows}/{}", profile.name, kernel.name()), |b| {
                    b.iter(|| {
                        fixture.run();
                        black_box(&fixture.output);
                    });
                });
            }
        }
    }

    group.finish();
}

#[derive(Clone, Copy)]
struct Profile {
    name: &'static str,
    n: i32,
    k: i32,
    input_dtype: Dtype,
    scale_bias_dtype: Dtype,
    output_dtype: Dtype,
}

impl Profile {
    fn config(self) -> affine_quantized::Config {
        affine_quantized::Config {
            n: self.n,
            k: self.k,
            group_size: 64,
            bits: 4,
            input_dtype: self.input_dtype,
            output_dtype: self.output_dtype,
            scale_bias_dtype: self.scale_bias_dtype,
        }
    }
}

#[derive(Clone, Copy)]
enum BenchKernel {
    Auto,
    Exact(affine_quantized::KernelKind),
}

impl BenchKernel {
    fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Exact(affine_quantized::KernelKind::QmvBn4Bk32) => "qmv-bn4-bk32",
            Self::Exact(affine_quantized::KernelKind::QmmBm8Bn16) => "qmm-bm8-bn16",
            Self::Exact(affine_quantized::KernelKind::QmmBm16Bn16) => "qmm-bm16-bn16",
            Self::Exact(affine_quantized::KernelKind::QmmBm32Bn16) => "qmm-bm32-bn16",
            Self::Exact(affine_quantized::KernelKind::QmvBn8Bk32) => "qmv-bn8-bk32",
            Self::Exact(affine_quantized::KernelKind::QmvQuadBn64) => "qmv-quad-bn64",
            Self::Exact(affine_quantized::KernelKind::QmmSlicedK { .. }) => "qmm-sliced-k",
            Self::Exact(affine_quantized::KernelKind::QmmSlicedKBm3Bn4) => "qmm-sliced-k-bm3-bn4",
            Self::Exact(affine_quantized::KernelKind::QmmSimdgroupsBm4Bn32) => "qmm-simdgroups-bm4-bn32",
            Self::Exact(affine_quantized::KernelKind::QmmBm8Bn32) => "qmm-bm8-bn32",
            Self::Exact(affine_quantized::KernelKind::QmmBm16Bn32) => "qmm-bm16-bn32",
            Self::Exact(affine_quantized::KernelKind::QmmBm32Bn32) => "qmm-bm32-bn32",
            Self::Exact(affine_quantized::KernelKind::QmmBm64Bn32) => "qmm-bm64-bn32",
            Self::Exact(affine_quantized::KernelKind::QmmBm32Bn64) => "qmm-bm32-bn64",
            Self::Exact(affine_quantized::KernelKind::QmmBm64Bn64) => "qmm-bm64-bn64",
        }
    }
}

struct MatmulFixture {
    stream: Stream,
    replay: ReplayProgram,
    output: Buffer,
}

impl MatmulFixture {
    fn new(device: &Device, profile: Profile, rows: i32, bench_kernel: BenchKernel) -> Self {
        let config = profile.config();
        let input = Buffer::new_zeroed(device, config.input_bytes(rows));
        let output = Buffer::new_zeroed(device, config.output_bytes(rows));
        let weight = Buffer::new_zeroed(device, config.weight_bytes());
        let scales = Buffer::new_zeroed(device, config.scale_or_bias_bytes());
        let biases = Buffer::new_zeroed(device, config.scale_or_bias_bytes());
        let stream = Stream::new(device);
        let mut recorder = stream.create_replay_program();
        match bench_kernel {
            BenchKernel::Auto => {
                let matmul = affine_quantized::Matmul::new(device, config);
                recorder.record(matmul.invoke(
                    rows as u32,
                    inference_backend_metal::metal::ReplayU32::Fixed(rows as u32),
                    &output,
                    0,
                    &input,
                    0,
                    &weight,
                    0,
                    &scales,
                    0,
                    &biases,
                    0,
                ));
            },
            BenchKernel::Exact(kind) => {
                let kernel = affine_quantized::Kernel::new(device, config, kind);
                recorder.record(kernel.invoke(
                    rows as u32,
                    inference_backend_metal::metal::ReplayU32::Fixed(rows as u32),
                    &output,
                    0,
                    &input,
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
        let replay = recorder.build();
        let fixture = Self { stream, replay, output };
        fixture.run();
        fixture
    }

    fn run(&self) {
        self.stream.submit_replay(&self.replay).wait();
    }
}

criterion_group!(benches, bench_affine_quantized_matmul);
criterion_main!(benches);
