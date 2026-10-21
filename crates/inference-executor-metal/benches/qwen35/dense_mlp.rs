use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::dense_mlp;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::Operator;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_metal::def::layer::ReplayLayer;
use inference_executor_metal::def::quantized_affine::QuantizedAffineLayout;
use inference_executor_metal::mlp::dense::backend::DenseMLP;
use inference_executor_metal::mlp::dense::backend::DenseMLPInput;
use inference_executor_metal::mlp::dense::backend::DenseMLPMetalConfig;
use inference_executor_metal::mlp::dense::scratch::DenseMLPScratchBindings;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

const SHARD: &str = "model-00001-of-00003.safetensors";
const LAYER0_MLP: &str = "language_model.model.layers.0.mlp";
const HIDDEN_DIM: u32 = 5120;
const INTERMEDIATE_DIM: u32 = 17_408;
const GROUP_SIZE: u32 = 64;
const BITS: u32 = 4;

fn main() {
    let args = Args::parse();
    if let Some(directory) = &args.dump_output {
        std::fs::create_dir_all(directory).expect("cannot create dense MLP output directory");
    }

    let device = Device::system_default();
    let weights = RealDenseMLPWeights::load(&device, &args.model_dir);
    for num_tokens in args.tokens {
        for case in &args.cases {
            let fixture = RealDenseMLPFixture::new(&device, num_tokens, &weights);
            let replay = fixture.replay(*case);
            let samples = measure_runs(args.runs, args.warmup_iters, args.iters, || {
                fixture.submit_replay(&replay)
            });
            if let Some(directory) = &args.dump_output {
                let buffer = match case {
                    DenseMLPBenchCase::GateUpSwiGLUAuto
                    | DenseMLPBenchCase::GateUpSwiGLUQmmBm8Bn16
                    | DenseMLPBenchCase::GateUpSwiGLUQmmBm16Bn16
                    | DenseMLPBenchCase::GateUpSwiGLUQmvBn8Bk32
                    | DenseMLPBenchCase::GateUpSwiGLUQmmBm32Bn16 => &fixture.swiglu,
                    DenseMLPBenchCase::FullAuto
                    | DenseMLPBenchCase::FullQmmBm8
                    | DenseMLPBenchCase::FullQmmBm16
                    | DenseMLPBenchCase::FullQmvBn8Bk32
                    | DenseMLPBenchCase::FullQmmBm32
                    | DenseMLPBenchCase::DownAuto
                    | DenseMLPBenchCase::DownQmmBm8Bn32
                    | DenseMLPBenchCase::DownQmmBm16Bn32
                    | DenseMLPBenchCase::DownQmvBn8Bk32
                    | DenseMLPBenchCase::DownQmmBm32Bn32 => &fixture.output,
                };
                let mut bytes = vec![0; buffer.len_bytes()];
                buffer.read_bytes(0, &mut bytes);
                let path = directory.join(format!("{}-rows{num_tokens}.bf16", case.key()));
                std::fs::write(path, bytes).expect("cannot write dense MLP output");
            }
            print_perf(
                fixture.stream.backend_name(),
                case.key(),
                num_tokens,
                args.iters,
                &samples,
                &replay,
            );
        }
    }
}

struct Args {
    model_dir: PathBuf,
    tokens: Vec<u32>,
    cases: Vec<DenseMLPBenchCase>,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
    dump_output: Option<PathBuf>,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            tokens: vec![1, 2, 4, 8, 10, 16, 32, 64],
            cases: vec![DenseMLPBenchCase::FullAuto],
            iters: 50,
            warmup_iters: 20,
            runs: 3,
            dump_output: None,
        };
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut iter, &arg)),
                "--dump-output" => args.dump_output = Some(PathBuf::from(next_arg(&mut iter, &arg))),
                "--tokens" => args.tokens = parse_u32_list(&next_arg(&mut iter, &arg), &arg),
                "--cases" => args.cases = parse_cases(&next_arg(&mut iter, &arg)),
                "--iters" => args.iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--bench" => {},
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                },
                _ => panic!("unknown argument {arg:?}; pass --help for usage"),
            }
        }
        assert!(!args.tokens.is_empty(), "--tokens must include at least one value");
        assert!(!args.cases.is_empty(), "--cases must include at least one case");
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        args
    }
}

#[derive(Clone, Copy, Debug)]
enum DenseMLPBenchCase {
    FullAuto,
    FullQmmBm8,
    FullQmmBm16,
    FullQmvBn8Bk32,
    FullQmmBm32,
    GateUpSwiGLUAuto,
    GateUpSwiGLUQmmBm8Bn16,
    GateUpSwiGLUQmmBm16Bn16,
    GateUpSwiGLUQmvBn8Bk32,
    GateUpSwiGLUQmmBm32Bn16,
    DownAuto,
    DownQmmBm8Bn32,
    DownQmmBm16Bn32,
    DownQmvBn8Bk32,
    DownQmmBm32Bn32,
}

impl DenseMLPBenchCase {
    fn parse(value: &str) -> Self {
        match value {
            "full_auto" => Self::FullAuto,
            "full_qmm_bm8" => Self::FullQmmBm8,
            "full_qmm_bm16" => Self::FullQmmBm16,
            "full_qmv_bn8_bk32" => Self::FullQmvBn8Bk32,
            "full_qmm_bm32" => Self::FullQmmBm32,
            "gate_up_swiglu_auto" => Self::GateUpSwiGLUAuto,
            "gate_up_swiglu_qmm_bm8_bn16" => Self::GateUpSwiGLUQmmBm8Bn16,
            "gate_up_swiglu_qmm_bm16_bn16" => Self::GateUpSwiGLUQmmBm16Bn16,
            "gate_up_swiglu_qmv_bn8_bk32" => Self::GateUpSwiGLUQmvBn8Bk32,
            "gate_up_swiglu_qmm_bm32_bn16" => Self::GateUpSwiGLUQmmBm32Bn16,
            "down_auto" => Self::DownAuto,
            "down_qmm_bm8_bn32" => Self::DownQmmBm8Bn32,
            "down_qmm_bm16_bn32" => Self::DownQmmBm16Bn32,
            "down_qmv_bn8_bk32" => Self::DownQmvBn8Bk32,
            "down_qmm_bm32_bn32" => Self::DownQmmBm32Bn32,
            _ => panic!("unknown dense MLP bench case {value:?}"),
        }
    }

    fn key(self) -> &'static str {
        match self {
            Self::FullAuto => "full_auto",
            Self::FullQmmBm8 => "full_qmm_bm8",
            Self::FullQmmBm16 => "full_qmm_bm16",
            Self::FullQmvBn8Bk32 => "full_qmv_bn8_bk32",
            Self::FullQmmBm32 => "full_qmm_bm32",
            Self::GateUpSwiGLUAuto => "gate_up_swiglu_auto",
            Self::GateUpSwiGLUQmmBm8Bn16 => "gate_up_swiglu_qmm_bm8_bn16",
            Self::GateUpSwiGLUQmmBm16Bn16 => "gate_up_swiglu_qmm_bm16_bn16",
            Self::GateUpSwiGLUQmvBn8Bk32 => "gate_up_swiglu_qmv_bn8_bk32",
            Self::GateUpSwiGLUQmmBm32Bn16 => "gate_up_swiglu_qmm_bm32_bn16",
            Self::DownAuto => "down_auto",
            Self::DownQmmBm8Bn32 => "down_qmm_bm8_bn32",
            Self::DownQmmBm16Bn32 => "down_qmm_bm16_bn32",
            Self::DownQmvBn8Bk32 => "down_qmv_bn8_bk32",
            Self::DownQmmBm32Bn32 => "down_qmm_bm32_bn32",
        }
    }
}

struct RealDenseMLPFixture<'a> {
    stream: Stream,
    shape: dense_mlp::Shape,
    backend: DenseMLP,
    compute: dense_mlp::Compute,
    hidden_state: Buffer,
    output: Buffer,
    swiglu: Buffer,
    gate_up_swiglu_qmv_bn8_bk32: affine_quantized::Kernel,
    gate_up_swiglu_qmm_bm8_bn16: affine_quantized::Kernel,
    gate_up_swiglu_qmm_bm16_bn16: affine_quantized::Kernel,
    gate_up_swiglu_qmm_bm32_bn16: affine_quantized::Kernel,
    down_qmv_bn8_bk32: affine_quantized::Kernel,
    down_qmm_bm8_bn32: affine_quantized::Kernel,
    down_qmm_bm16_bn32: affine_quantized::Kernel,
    down_qmm_bm32_bn32: affine_quantized::Kernel,
    weights: &'a RealDenseMLPWeights,
}

impl<'a> RealDenseMLPFixture<'a> {
    fn new(device: &Device, num_tokens: u32, weights: &'a RealDenseMLPWeights) -> Self {
        let config = dense_config();
        let backend = DenseMLP::new(
            device,
            DenseMLPCore {
                model_layer_index: 0,
                hidden_dim: HIDDEN_DIM as usize,
                intermediate_dim: INTERMEDIATE_DIM as usize,
            },
            DenseMLPMetalConfig {
                gate_up: QuantizedAffineLayout {
                    group_size: GROUP_SIZE,
                    bits: BITS,
                    scale_bias_dtype: Dtype::Bfloat16,
                },
                down: QuantizedAffineLayout {
                    group_size: GROUP_SIZE,
                    bits: BITS,
                    scale_bias_dtype: Dtype::Bfloat16,
                },
                io_dtype: Dtype::Bfloat16,
            },
        );
        let shape = dense_mlp::Shape {
            num_total_tokens: num_tokens,
        };
        let gate_up_config = gate_up_affine_config();
        let down_config = down_affine_config();
        let m = num_tokens.try_into().expect("dense MLP token count must fit i32");
        let hidden_state = Buffer::from_slice(device, &hidden_fixture(num_tokens as usize, HIDDEN_DIM as usize));
        Self {
            stream: Stream::new(device),
            shape,
            backend,
            compute: dense_mlp::Compute::new(device, config),
            hidden_state,
            output: Buffer::new_zeroed(device, down_config.output_bytes(m)),
            swiglu: Buffer::from_slice(device, &hidden_fixture(num_tokens as usize, INTERMEDIATE_DIM as usize)),
            gate_up_swiglu_qmv_bn8_bk32: affine_quantized::Kernel::new_gate_up_swiglu(
                device,
                gate_up_config,
                affine_quantized::KernelKind::QmvBn8Bk32,
            ),
            gate_up_swiglu_qmm_bm8_bn16: affine_quantized::Kernel::new_gate_up_swiglu(
                device,
                gate_up_config,
                affine_quantized::KernelKind::QmmBm8Bn16,
            ),
            gate_up_swiglu_qmm_bm16_bn16: affine_quantized::Kernel::new_gate_up_swiglu(
                device,
                gate_up_config,
                affine_quantized::KernelKind::QmmBm16Bn16,
            ),
            gate_up_swiglu_qmm_bm32_bn16: affine_quantized::Kernel::new_gate_up_swiglu(
                device,
                gate_up_config,
                affine_quantized::KernelKind::QmmBm32Bn16,
            ),
            down_qmv_bn8_bk32: affine_quantized::Kernel::new(
                device,
                down_config,
                affine_quantized::KernelKind::QmvBn8Bk32,
            ),
            down_qmm_bm8_bn32: affine_quantized::Kernel::new(
                device,
                down_config,
                affine_quantized::KernelKind::QmmBm8Bn32,
            ),
            down_qmm_bm16_bn32: affine_quantized::Kernel::new(
                device,
                down_config,
                affine_quantized::KernelKind::QmmBm16Bn32,
            ),
            down_qmm_bm32_bn32: affine_quantized::Kernel::new(
                device,
                down_config,
                affine_quantized::KernelKind::QmmBm32Bn32,
            ),
            weights,
        }
    }

    fn replay(&self, case: DenseMLPBenchCase) -> ReplayProgram {
        match case {
            DenseMLPBenchCase::FullAuto => self.forward_replay(),
            DenseMLPBenchCase::FullQmmBm8 => self.forced_full_replay(DenseAffinePolicy::QmmBm8Bn32),
            DenseMLPBenchCase::FullQmmBm16 => self.forced_full_replay(DenseAffinePolicy::QmmBm16Bn32),
            DenseMLPBenchCase::FullQmvBn8Bk32 => self.forced_full_replay(DenseAffinePolicy::QmvBn8Bk32),
            DenseMLPBenchCase::FullQmmBm32 => self.forced_full_replay(DenseAffinePolicy::QmmBm32Bn32),
            DenseMLPBenchCase::GateUpSwiGLUAuto => {
                build_single_invocation_replay(
                    &self.stream,
                    self.compute.invoke_gate_up_swiglu(
                        self.shape,
                        ReplayU32::Fixed(self.shape.num_total_tokens),
                        &self.hidden_state,
                        &self.swiglu,
                        self.weights.as_borrowed(),
                    ),
                )
            },
            DenseMLPBenchCase::GateUpSwiGLUQmmBm8Bn16 => self.forced_gate_up_replay(DenseAffinePolicy::QmmBm8Bn32),
            DenseMLPBenchCase::GateUpSwiGLUQmmBm16Bn16 => self.forced_gate_up_replay(DenseAffinePolicy::QmmBm16Bn32),
            DenseMLPBenchCase::GateUpSwiGLUQmvBn8Bk32 => self.forced_gate_up_replay(DenseAffinePolicy::QmvBn8Bk32),
            DenseMLPBenchCase::GateUpSwiGLUQmmBm32Bn16 => self.forced_gate_up_replay(DenseAffinePolicy::QmmBm32Bn32),
            DenseMLPBenchCase::DownAuto => {
                build_single_invocation_replay(
                    &self.stream,
                    self.compute.invoke_down(
                        self.shape,
                        ReplayU32::Fixed(self.shape.num_total_tokens),
                        &self.swiglu,
                        &self.output,
                        self.weights.as_borrowed(),
                    ),
                )
            },
            DenseMLPBenchCase::DownQmmBm8Bn32 => self.forced_down_replay(DenseAffinePolicy::QmmBm8Bn32),
            DenseMLPBenchCase::DownQmmBm16Bn32 => self.forced_down_replay(DenseAffinePolicy::QmmBm16Bn32),
            DenseMLPBenchCase::DownQmvBn8Bk32 => self.forced_down_replay(DenseAffinePolicy::QmvBn8Bk32),
            DenseMLPBenchCase::DownQmmBm32Bn32 => self.forced_down_replay(DenseAffinePolicy::QmmBm32Bn32),
        }
    }

    fn forward_replay(&self) -> ReplayProgram {
        let mut recorder = ReplayRecorder::new(self.stream.create_replay_program());
        let _ = <DenseMLP as ReplayLayer>::record(
            &self.backend,
            &mut recorder,
            DenseMLPInput {
                num_total_tokens: self.shape.num_total_tokens,
                num_active_tokens: ReplayU32::Fixed(self.shape.num_total_tokens),
                hidden_state: &self.hidden_state,
                next_hidden_state: &self.output,
                scratch: DenseMLPScratchBindings { swiglu: &self.swiglu },
                weights: self.weights.as_borrowed(),
            },
        );
        recorder.build()
    }

    fn submit_replay(&self, replay: &ReplayProgram) {
        self.stream.submit_replay(replay).wait();
    }

    fn forced_full_replay(&self, policy: DenseAffinePolicy) -> ReplayProgram {
        let mut recorder = ReplayRecorder::new(self.stream.create_replay_program());
        self.record_forced_gate_up(&mut recorder, policy);
        self.record_forced_down(&mut recorder, policy);
        recorder.build()
    }

    fn forced_gate_up_replay(&self, policy: DenseAffinePolicy) -> ReplayProgram {
        let mut recorder = ReplayRecorder::new(self.stream.create_replay_program());
        self.record_forced_gate_up(&mut recorder, policy);
        recorder.build()
    }

    fn forced_down_replay(&self, policy: DenseAffinePolicy) -> ReplayProgram {
        let mut recorder = ReplayRecorder::new(self.stream.create_replay_program());
        self.record_forced_down(&mut recorder, policy);
        recorder.build()
    }

    fn record_forced_gate_up(&self, recorder: &mut ReplayRecorder, policy: DenseAffinePolicy) {
        let kernel = match policy {
            DenseAffinePolicy::QmvBn8Bk32 => &self.gate_up_swiglu_qmv_bn8_bk32,
            DenseAffinePolicy::QmmBm8Bn32 => &self.gate_up_swiglu_qmm_bm8_bn16,
            DenseAffinePolicy::QmmBm16Bn32 => &self.gate_up_swiglu_qmm_bm16_bn16,
            DenseAffinePolicy::QmmBm32Bn32 => &self.gate_up_swiglu_qmm_bm32_bn16,
        };
        let weights = self.weights.as_borrowed();
        recorder.record_with_barrier_before(ReplayOp::opaque(kernel.invoke(
            self.shape.num_total_tokens,
            ReplayU32::Fixed(self.shape.num_total_tokens),
            &self.swiglu,
            0,
            &self.hidden_state,
            0,
            weights.gate_up_weight,
            0,
            weights.gate_up_scales,
            0,
            weights.gate_up_biases,
            0,
        )));
    }

    fn record_forced_down(&self, recorder: &mut ReplayRecorder, policy: DenseAffinePolicy) {
        let kernel = match policy {
            DenseAffinePolicy::QmvBn8Bk32 => &self.down_qmv_bn8_bk32,
            DenseAffinePolicy::QmmBm8Bn32 => &self.down_qmm_bm8_bn32,
            DenseAffinePolicy::QmmBm16Bn32 => &self.down_qmm_bm16_bn32,
            DenseAffinePolicy::QmmBm32Bn32 => &self.down_qmm_bm32_bn32,
        };
        let weights = self.weights.as_borrowed();
        recorder.record_with_barrier_before(ReplayOp::opaque(kernel.invoke(
            self.shape.num_total_tokens,
            ReplayU32::Fixed(self.shape.num_total_tokens),
            &self.output,
            0,
            &self.swiglu,
            0,
            weights.down_weight,
            0,
            weights.down_scales,
            0,
            weights.down_biases,
            0,
        )));
    }
}

#[derive(Clone, Copy)]
enum DenseAffinePolicy {
    QmvBn8Bk32,
    QmmBm8Bn32,
    QmmBm16Bn32,
    QmmBm32Bn32,
}

struct RealDenseMLPWeights {
    gate_up_weight: Buffer,
    gate_up_scales: Buffer,
    gate_up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
}

impl RealDenseMLPWeights {
    fn load(device: &Device, model_dir: &Path) -> Self {
        let shard_path = model_dir.join(SHARD);
        let mapped = MappedFile::open(&shard_path);
        let tensors = SafeTensors::deserialize(mapped.as_bytes()).unwrap_or_else(|err| {
            panic!(
                "unable to deserialize safetensors shard {}: {err:?}",
                shard_path.display()
            )
        });
        let gate_weight = tensor_bytes(&tensors, &tensor_name("gate_proj", "weight"), safetensors::Dtype::U32);
        let up_weight = tensor_bytes(&tensors, &tensor_name("up_proj", "weight"), safetensors::Dtype::U32);
        let gate_scales = tensor_bytes(&tensors, &tensor_name("gate_proj", "scales"), safetensors::Dtype::BF16);
        let up_scales = tensor_bytes(&tensors, &tensor_name("up_proj", "scales"), safetensors::Dtype::BF16);
        let gate_biases = tensor_bytes(&tensors, &tensor_name("gate_proj", "biases"), safetensors::Dtype::BF16);
        let up_biases = tensor_bytes(&tensors, &tensor_name("up_proj", "biases"), safetensors::Dtype::BF16);
        let down_weight = tensor_bytes(&tensors, &tensor_name("down_proj", "weight"), safetensors::Dtype::U32);
        let down_scales = tensor_bytes(&tensors, &tensor_name("down_proj", "scales"), safetensors::Dtype::BF16);
        let down_biases = tensor_bytes(&tensors, &tensor_name("down_proj", "biases"), safetensors::Dtype::BF16);

        let gate_up_weight = concat_bytes(&gate_weight, &up_weight);
        let gate_up_scales = concat_bytes(&gate_scales, &up_scales);
        let gate_up_biases = concat_bytes(&gate_biases, &up_biases);

        let config = dense_config();
        let gate_up_config = config.gate_up_config();
        let down_config = config.down_config();
        assert_eq!(gate_up_weight.len(), gate_up_config.weight_bytes());
        assert_eq!(gate_up_scales.len(), gate_up_config.scale_or_bias_bytes());
        assert_eq!(gate_up_biases.len(), gate_up_config.scale_or_bias_bytes());
        assert_eq!(down_weight.len(), down_config.weight_bytes());
        assert_eq!(down_scales.len(), down_config.scale_or_bias_bytes());
        assert_eq!(down_biases.len(), down_config.scale_or_bias_bytes());

        Self {
            gate_up_weight: Buffer::from_slice(device, &gate_up_weight),
            gate_up_scales: Buffer::from_slice(device, &gate_up_scales),
            gate_up_biases: Buffer::from_slice(device, &gate_up_biases),
            down_weight: Buffer::from_slice(device, &down_weight),
            down_scales: Buffer::from_slice(device, &down_scales),
            down_biases: Buffer::from_slice(device, &down_biases),
        }
    }

    fn as_borrowed(&self) -> dense_mlp::Weights<'_> {
        dense_mlp::Weights {
            gate_up_weight: &self.gate_up_weight,
            gate_up_scales: &self.gate_up_scales,
            gate_up_biases: &self.gate_up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }
}

struct MappedFile {
    ptr: *mut libc::c_void,
    len: usize,
}

impl MappedFile {
    fn open(path: &Path) -> Self {
        let file = File::open(path).unwrap_or_else(|err| panic!("unable to open {}: {err}", path.display()));
        let len = file
            .metadata()
            .unwrap_or_else(|err| panic!("unable to stat {}: {err}", path.display()))
            .len() as usize;
        assert!(len > 0, "safetensors shard must not be empty");
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            panic!("unable to mmap {}: {}", path.display(), std::io::Error::last_os_error());
        }
        unsafe {
            let _ = libc::madvise(ptr, len, libc::MADV_RANDOM);
        }
        Self { ptr, len }
    }

    fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr, self.len);
        }
    }
}

fn tensor_name(proj: &str, part: &str) -> String {
    format!("{LAYER0_MLP}.{proj}.{part}")
}

fn tensor_bytes(tensors: &SafeTensors<'_>, name: &str, dtype: safetensors::Dtype) -> Vec<u8> {
    let view = tensors
        .tensor(name)
        .unwrap_or_else(|err| panic!("missing safetensor {name}: {err:?}"));
    assert_eq!(view.dtype(), dtype, "unexpected dtype for tensor {name}");
    validate_tensor_shape(name, &view);
    view.data().to_vec()
}

fn validate_tensor_shape(name: &str, view: &TensorView<'_>) {
    let shape = view.shape();
    if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") {
        assert_eq!(shape, &[INTERMEDIATE_DIM as usize, packed_k_words(HIDDEN_DIM)]);
    } else if name.ends_with("down_proj.weight") {
        assert_eq!(shape, &[HIDDEN_DIM as usize, packed_k_words(INTERMEDIATE_DIM)]);
    } else if name.ends_with("gate_proj.scales")
        || name.ends_with("gate_proj.biases")
        || name.ends_with("up_proj.scales")
        || name.ends_with("up_proj.biases")
    {
        assert_eq!(shape, &[INTERMEDIATE_DIM as usize, groups(HIDDEN_DIM)]);
    } else if name.ends_with("down_proj.scales") || name.ends_with("down_proj.biases") {
        assert_eq!(shape, &[HIDDEN_DIM as usize, groups(INTERMEDIATE_DIM)]);
    } else {
        panic!("unexpected dense MLP tensor name {name}");
    }
}

fn concat_bytes(left: &[u8], right: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    out.extend_from_slice(left);
    out.extend_from_slice(right);
    out
}

fn dense_config() -> dense_mlp::Config {
    dense_mlp::Config {
        hidden_dim: HIDDEN_DIM,
        intermediate_dim: INTERMEDIATE_DIM,
        gate_up_group_size: GROUP_SIZE,
        gate_up_bits: BITS,
        gate_up_scale_bias_dtype: Dtype::Bfloat16,
        down_group_size: GROUP_SIZE,
        down_bits: BITS,
        down_scale_bias_dtype: Dtype::Bfloat16,
        dtype: Dtype::Bfloat16,
    }
}

fn gate_up_affine_config() -> affine_quantized::Config {
    dense_affine_config(INTERMEDIATE_DIM * 2, HIDDEN_DIM)
}

fn down_affine_config() -> affine_quantized::Config {
    dense_affine_config(HIDDEN_DIM, INTERMEDIATE_DIM)
}

fn dense_affine_config(n: u32, k: u32) -> affine_quantized::Config {
    affine_quantized::Config {
        n: n.try_into().expect("dense MLP affine n must fit i32"),
        k: k.try_into().expect("dense MLP affine k must fit i32"),
        group_size: GROUP_SIZE.try_into().expect("dense MLP group size must fit i32"),
        bits: BITS.try_into().expect("dense MLP bits must fit i32"),
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
    }
}

fn hidden_fixture(tokens: usize, hidden_dim: usize) -> Vec<u16> {
    (0..tokens * hidden_dim)
        .map(|index| bf16::from_f32(((index % 23) as f32 - 11.0) * 0.03125).to_bits())
        .collect()
}

fn packed_k_words(k: u32) -> usize {
    (k as usize * BITS as usize) / 32
}

fn groups(k: u32) -> usize {
    (k / GROUP_SIZE) as usize
}

fn build_single_invocation_replay<I>(stream: &Stream, invocation: I) -> ReplayProgram
where
    I: Operator,
{
    let mut recorder = ReplayRecorder::new(stream.create_replay_program());
    recorder.record(ReplayOp::opaque(invocation));
    recorder.build()
}

fn next_arg(iter: &mut impl Iterator<Item = String>, flag: &str) -> String {
    iter.next()
        .unwrap_or_else(|| panic!("{flag} requires a value; pass --help for usage"))
}

fn parse_u32_list(value: &str, flag: &str) -> Vec<u32> {
    parse_list(value)
        .into_iter()
        .map(|part| {
            part.parse::<u32>()
                .unwrap_or_else(|err| panic!("invalid {flag} entry {part:?}: {err}"))
        })
        .collect()
}

fn parse_cases(value: &str) -> Vec<DenseMLPBenchCase> {
    parse_list(value).into_iter().map(DenseMLPBenchCase::parse).collect()
}

fn parse_list(value: &str) -> Vec<&str> {
    let values = value
        .split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    assert!(!values.is_empty(), "list argument must contain at least one value");
    values
}

fn parse_usize(value: &str, flag: &str) -> usize {
    value
        .parse()
        .unwrap_or_else(|err| panic!("invalid {flag} value {value:?}: {err}"))
}

fn print_help() {
    println!(
        "\
dense_mlp real-weight replay bench

Options:
  --model-dir PATH
  --dump-output PATH  write each case output as raw BF16 after timing
  --tokens 1,2,4,8,10,16,32,64
  --cases full_auto,full_qmv_bn8_bk32,full_qmm_bm8,full_qmm_bm16,full_qmm_bm32,gate_up_swiglu_auto,\
         gate_up_swiglu_qmv_bn8_bk32,gate_up_swiglu_qmm_bm8_bn16,gate_up_swiglu_qmm_bm16_bn16,\
         gate_up_swiglu_qmm_bm32_bn16,down_auto,down_qmv_bn8_bk32,down_qmm_bm8_bn32,down_qmm_bm16_bn32,\
         down_qmm_bm32_bn32
  --iters N
  --warmup-iters N
  --runs N
"
    );
}

fn measure_runs(runs: usize, warmup_iters: usize, iters: usize, mut run: impl FnMut()) -> Vec<f64> {
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        for _ in 0..warmup_iters {
            run();
        }
        let mut duration = Duration::ZERO;
        for _ in 0..iters {
            let start = Instant::now();
            run();
            duration += start.elapsed();
        }
        samples.push(duration.as_secs_f64() * 1_000_000.0 / iters as f64);
    }
    samples
}

fn print_perf(
    backend: &str,
    implementation: &str,
    num_tokens: u32,
    iters: usize,
    samples: &[f64],
    replay: &ReplayProgram,
) {
    let median_us = median(samples);
    let stats = replay.stats();
    let sample_text = samples
        .iter()
        .map(|sample| format!("{sample:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "perf component=dense-mlp-real backend={backend} impl={implementation} tokens={num_tokens} iters={iters} \
         runs={} median_us={median_us:.3} command_count={} retained_buffers={} retained_pipelines={} \
         constant_bytes={} samples_us=[{sample_text}]",
        samples.len(),
        stats.command_count,
        stats.retained_buffer_count,
        stats.retained_pipeline_count,
        stats.parameter_buffer_bytes
    );
}

fn median(samples: &[f64]) -> f64 {
    assert!(!samples.is_empty());
    let mut sorted = samples.to_vec();
    sorted.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}
