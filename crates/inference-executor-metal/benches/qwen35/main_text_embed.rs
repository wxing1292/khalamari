use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

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
use inference_executor_metal::model::embedding::Embed;
use inference_executor_metal::model::embedding::EmbedConfig;
use inference_executor_metal::model::qwen::v3_5::main::text_embed::Qwen35MainTextEmbed;
use inference_executor_metal::model::qwen::v3_5::main::text_embed::Qwen35MainTextEmbedArgs;
use inference_executor_metal::model::qwen::v3_5::main::text_embed::Qwen35MainTextEmbedReplayKey;
use inference_executor_metal::replay::Replay;

const DEFAULT_MAX_TOKENS: u32 = 128;

fn main() {
    let args = Args::parse();
    let device = Device::system_default();
    let mut fixture = MainTextEmbedFixture::new(&device, &args.model_dir, args.max_tokens);

    for &num_tokens in &args.tokens {
        let prepared = fixture.prepare(num_tokens);
        fixture.run(&prepared);
        let samples = measure_runs(args.runs, args.warmup_iters, args.iters, || fixture.run(&prepared));
        print_perf(&args, num_tokens, fixture.replay.replay(&prepared.key), &samples);
    }
}

struct Args {
    model_dir: PathBuf,
    tokens: Vec<u32>,
    max_tokens: u32,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            tokens: vec![1, 4, 16, 64, 128],
            max_tokens: DEFAULT_MAX_TOKENS,
            iters: 200,
            warmup_iters: 50,
            runs: 7,
        };
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(),
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut iter, &arg)),
                "--tokens" => args.tokens = parse_u32_list(&next_arg(&mut iter, &arg), &arg),
                "--max-tokens" => args.max_tokens = parse_u32(&next_arg(&mut iter, &arg), &arg),
                "--iters" => args.iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(args.max_tokens > 0, "--max-tokens must be positive");
        assert!(!args.tokens.is_empty(), "--tokens must include at least one value");
        assert!(
            args.tokens.iter().all(|&value| value > 0),
            "--tokens entries must be positive"
        );
        assert!(
            args.tokens.iter().all(|&value| value <= args.max_tokens),
            "--tokens entries must not exceed --max-tokens"
        );
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        args.tokens.sort_unstable();
        assert!(
            args.tokens.windows(2).all(|pair| pair[0] != pair[1]),
            "--tokens must not contain duplicates"
        );
        args
    }
}

struct MainTextEmbedFixture {
    stream: Stream,
    token_ids: Buffer,
    hidden_output: Buffer,
    replay: Replay<Qwen35MainTextEmbed>,
}

struct PreparedReplay {
    key: Qwen35MainTextEmbedReplayKey,
    arguments: ReplayArguments,
}

impl MainTextEmbedFixture {
    fn new(device: &Device, model_dir: &std::path::Path, max_tokens: u32) -> Self {
        let config = init_qwen35_model_config(model_dir).unwrap_or_else(|error| {
            panic!(
                "unable to initialize Qwen3.5 config from {}: {error}",
                model_dir.display()
            )
        });
        let quantization = config
            .quantization
            .as_ref()
            .expect("qwen3.5 MainTextEmbed benchmark requires quantized weights");
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
        let embed_config = EmbedConfig {
            max_tokens,
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
            scale_bias_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
        };
        let mut store = SafeTensorStore::from_model_dir(model_dir)
            .unwrap_or_else(|error| panic!("unable to open Qwen3.5 weights from {}: {error}", model_dir.display()));
        let Qwen35ModelWeightBindings { embed, .. } =
            resolve_qwen35_model_weight_bindings(&config, store.index().tensor_names()).unwrap_or_else(|error| {
                panic!(
                    "unable to resolve Qwen3.5 weight layout from {}: {error}",
                    model_dir.display()
                )
            });
        let mut embed_component = Embed::new(device, embed_config);
        embed_component
            .load_weights(device, &mut store, embed)
            .unwrap_or_else(|error| panic!("unable to load Qwen3.5 MainTextEmbed weights: {error}"));
        store.unload_all();
        let token_ids = (0..max_tokens)
            .map(|index| ((u64::from(index) * 65_537 + 17) % u64::from(vocab_size)) as i32)
            .collect::<Vec<_>>();
        Self {
            stream: Stream::new(device),
            token_ids: Buffer::from_slice(device, &token_ids),
            hidden_output: Buffer::new_zeroed_elements(
                device,
                max_tokens as usize * hidden_dim as usize,
                Dtype::Bfloat16,
            ),
            replay: Replay::new(
                "qwen3.5 MainTextEmbed benchmark",
                Qwen35MainTextEmbed::new(Rc::new(embed_component)),
            ),
        }
    }

    fn prepare(&mut self, num_tokens: u32) -> PreparedReplay {
        let input = Qwen35MainTextEmbedArgs {
            num_tokens,
            token_ids: &self.token_ids,
            hidden_output: &self.hidden_output,
        };
        let (expected_key, arguments) = self.replay.component().prepare_replay(num_tokens);
        let runtime = &self.stream;
        let (key, _) = self.replay.record(runtime, &input);
        assert_eq!(
            key, expected_key,
            "MainTextEmbed prepared and recorded replay keys must match"
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

fn print_perf(args: &Args, num_tokens: u32, replay: &ReplayProgram, samples: &[f64]) {
    let median_us = median(samples);
    let stats = replay.stats();
    let samples_text = samples
        .iter()
        .map(|sample| format!("{sample:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "perf component=qwen35-main-text-embed model_dir={} case=embed num_tokens={num_tokens} max_tokens={} \
         commands={} retained_buffers={} retained_pipelines={} constant_bytes={} iters={} runs={} \
         median_us={median_us:.3} samples_us=[{samples_text}]",
        args.model_dir.display(),
        args.max_tokens,
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
        "qwen35_main_text_embed bench\n--model-dir PATH\n--tokens 1,4,16,64,128\n--max-tokens 128\n--iters \
         N\n--warmup-iters N\n--runs N"
    );
    std::process::exit(0);
}
