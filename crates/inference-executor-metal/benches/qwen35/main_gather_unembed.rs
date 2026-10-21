use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use inference_executor_core::checkpoint::SafeTensorStore;
use inference_executor_core::model::qwen::v3_5::init_qwen35_model_config;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35ModelWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::resolve_qwen35_model_weight_bindings;
use inference_executor_metal::model::qwen::v3_5::main::output::Qwen35GatherUnembed;
use inference_executor_metal::model::qwen::v3_5::main::output::Qwen35GatherUnembedArgs;
use inference_executor_metal::model::qwen::v3_5::main::output::Qwen35GatherUnembedReplayKey;
use inference_executor_metal::model::unembedding::Unembed;
use inference_executor_metal::model::unembedding::UnembedConfig;
use inference_executor_metal::replay::Replay;

const DEFAULT_MAX_ROWS: u32 = 128;

fn main() {
    let args = Args::parse();
    let device = Device::system_default();
    let mut fixture = GatherUnembedFixture::new(&device, &args.model_dir, args.max_rows);

    for &num_rows in &args.rows {
        let prepared = fixture.prepare(num_rows);
        fixture.run(&prepared);
        let samples = measure_runs(args.runs, args.warmup_iters, args.iters, || fixture.run(&prepared));
        print_perf(&args, num_rows, fixture.replay.replay(&prepared.key), &samples);
    }
}

struct Args {
    model_dir: PathBuf,
    rows: Vec<u32>,
    max_rows: u32,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            rows: vec![1, 2, 4, 8],
            max_rows: DEFAULT_MAX_ROWS,
            iters: 200,
            warmup_iters: 50,
            runs: 7,
        };
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(),
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut iter, &arg)),
                "--rows" => args.rows = parse_u32_list(&next_arg(&mut iter, &arg), &arg),
                "--max-rows" => args.max_rows = parse_u32(&next_arg(&mut iter, &arg), &arg),
                "--iters" => args.iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(args.max_rows > 0, "--max-rows must be positive");
        assert!(!args.rows.is_empty(), "--rows must include at least one value");
        assert!(
            args.rows.iter().all(|&value| value > 0),
            "--rows entries must be positive"
        );
        assert!(
            args.rows.iter().all(|&value| value <= args.max_rows),
            "--rows entries must not exceed --max-rows"
        );
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        args.rows.sort_unstable();
        assert!(
            args.rows.windows(2).all(|pair| pair[0] != pair[1]),
            "--rows must not contain duplicates"
        );
        args
    }
}

struct GatherUnembedFixture {
    stream: Stream,
    hidden_input: Buffer,
    row_indices: Buffer,
    hidden_output: Buffer,
    logits: Buffer,
    replay: Replay<Qwen35GatherUnembed>,
}

struct PreparedReplay {
    key: Qwen35GatherUnembedReplayKey,
    arguments: ReplayArguments,
}

impl GatherUnembedFixture {
    fn new(device: &Device, model_dir: &std::path::Path, max_rows: u32) -> Self {
        let config = init_qwen35_model_config(model_dir).unwrap_or_else(|error| {
            panic!(
                "unable to initialize Qwen3.5 config from {}: {error}",
                model_dir.display()
            )
        });
        let quantization = config
            .quantization
            .as_ref()
            .expect("qwen3.5 GatherUnembed benchmark requires quantized weights");
        let vocab_size: u32 = config
            .text_config
            .vocab_size
            .try_into()
            .expect("Qwen3.5 vocabulary size must fit u32");
        let hidden_dim: u32 = config
            .text_config
            .hidden_size
            .try_into()
            .expect("Qwen3.5 hidden dimension must fit u32");
        let unembed_config = UnembedConfig {
            max_tokens: max_rows,
            vocab_size,
            hidden_dim,
            group_size: quantization
                .group_size
                .try_into()
                .expect("Qwen3.5 quantization group size must fit u32"),
            bits: quantization
                .bits
                .try_into()
                .expect("Qwen3.5 quantization bits must fit u32"),
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            scale_bias_dtype: Dtype::Bfloat16,
        };
        let mut store = SafeTensorStore::from_model_dir(model_dir)
            .unwrap_or_else(|error| panic!("unable to open Qwen3.5 weights from {}: {error}", model_dir.display()));
        let Qwen35ModelWeightBindings { unembed, .. } =
            resolve_qwen35_model_weight_bindings(&config, store.index().tensor_names()).unwrap_or_else(|error| {
                panic!(
                    "unable to resolve Qwen3.5 weight layout from {}: {error}",
                    model_dir.display()
                )
            });
        let mut unembed_component = Unembed::new(device, unembed_config);
        unembed_component
            .load_weights(device, &mut store, unembed)
            .unwrap_or_else(|error| panic!("unable to load Qwen3.5 unembed weights: {error}"));
        store.unload_all();
        let hidden = (0..max_rows as usize * hidden_dim as usize)
            .map(|index| bf16::from_f32(((index * 17 + 11) % 251) as f32 * 0.003_906_25 - 0.5).to_bits())
            .collect::<Vec<_>>();
        let row_indices = (0..max_rows).rev().collect::<Vec<_>>();
        let component = Qwen35GatherUnembed::new(device, hidden_dim, Rc::new(unembed_component));
        assert_eq!(
            component.max_rows(),
            max_rows,
            "GatherUnembed capacity must match --max-rows"
        );
        Self {
            stream: Stream::new(device),
            hidden_input: Buffer::from_slice(device, &hidden),
            row_indices: Buffer::from_slice(device, &row_indices),
            hidden_output: Buffer::new_zeroed_elements(
                device,
                max_rows as usize * hidden_dim as usize,
                Dtype::Bfloat16,
            ),
            logits: Buffer::new_zeroed(device, unembed_config.logits_bytes()),
            replay: Replay::new("qwen3.5 GatherUnembed benchmark", component),
        }
    }

    fn prepare(&mut self, num_rows: u32) -> PreparedReplay {
        let input = Qwen35GatherUnembedArgs {
            num_rows,
            hidden_input: &self.hidden_input,
            row_indices: &self.row_indices,
            hidden_output: &self.hidden_output,
            logits: &self.logits,
        };
        let (expected_key, arguments) = self.replay.component().prepare_replay(num_rows);
        let runtime = &self.stream;
        let (key, _) = self.replay.record(runtime, &input);
        assert_eq!(
            key, expected_key,
            "GatherUnembed prepared and recorded replay keys must match"
        );
        PreparedReplay { key, arguments }
    }

    fn run(&self, prepared: &PreparedReplay) {
        self.stream
            .submit_replay_with_arguments(self.replay.replay(&prepared.key), &prepared.arguments)
            .wait();
    }
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
        samples.push(duration.as_secs_f64() * 1.0e6 / iters as f64);
    }
    samples
}

fn print_perf(args: &Args, num_rows: u32, replay: &ReplayProgram, samples: &[f64]) {
    let median_us = median(samples);
    let stats = replay.stats();
    let samples_text = samples
        .iter()
        .map(|sample| format!("{sample:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "perf component=qwen35-main-gather-unembed model_dir={} case=gather_unembed num_rows={num_rows} max_rows={} \
         commands={} retained_buffers={} retained_pipelines={} constant_bytes={} iters={} runs={} \
         median_us={median_us:.3} samples_us=[{samples_text}]",
        args.model_dir.display(),
        args.max_rows,
        stats.command_count,
        stats.retained_buffer_count,
        stats.retained_pipeline_count,
        stats.parameter_buffer_bytes,
        args.iters,
        samples.len(),
    );
}

fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "benchmark requires timing samples");
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let midpoint = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[midpoint - 1] + values[midpoint]) * 0.5
    } else {
        values[midpoint]
    }
}

fn parse_u32_list(value: &str, name: &str) -> Vec<u32> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| parse_u32(part, name))
        .collect()
}

fn parse_u32(value: &str, name: &str) -> u32 {
    value
        .parse()
        .unwrap_or_else(|error| panic!("invalid {name} value {value:?}: {error}"))
}

fn parse_usize(value: &str, name: &str) -> usize {
    value
        .parse()
        .unwrap_or_else(|error| panic!("invalid {name} value {value:?}: {error}"))
}

fn next_arg(iter: &mut impl Iterator<Item = String>, name: &str) -> String {
    iter.next()
        .unwrap_or_else(|| panic!("{name} requires a value; pass --help for usage"))
}

fn print_help_and_exit() -> ! {
    println!(
        "qwen35_main_gather_unembed bench\n--model-dir PATH\n--rows 1,2,4,8\n--max-rows 128\n--iters \
         N\n--warmup-iters N\n--runs N"
    );
    std::process::exit(0);
}
