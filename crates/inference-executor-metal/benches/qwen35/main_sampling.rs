use std::time::Duration;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use inference_executor_core::sampling::MAX_TOP_K;
use inference_executor_core::sampling::SamplerConfig;
use inference_executor_metal::replay::Replay;
use inference_executor_metal::sampling::sampling_params::SamplingParamsStore;
use inference_executor_metal::sampling::top_k_replay::Sampling;
use inference_executor_metal::sampling::top_k_replay::SamplingInput;
use inference_executor_metal::sampling::top_k_replay::TopKSamplingReplayKey;
use inference_executor_metal::sampling::top_k_sampling::TopKSampling;
use inference_executor_metal::sampling::top_k_sampling::TopKSamplingInputs;
use inference_executor_metal::sampling::top_k_sampling::TopKSamplingOutputBuffers;
use inference_executor_metal::sampling::top_k_sampling::config::TopKSamplingBounds;

const DEFAULT_VOCAB_SIZE: u32 = 151_936;

fn main() {
    let args = Args::parse();
    let device = Device::system_default();
    let max_rows = *args.rows.iter().max().expect("sampling benchmark requires rows");
    let mut fixture = MainSamplingFixture::new(&device, &args, max_rows);

    for &num_rows in &args.rows {
        let prepared = fixture.prepare(&args, num_rows);
        fixture.run(Case::SampleReadback, &prepared);
        for &case in &args.cases {
            let samples = measure_runs(args.runs, args.warmup_iters, args.iters, || {
                fixture.run(case, &prepared)
            });
            print_perf(&args, case, num_rows, fixture.replay.replay(&prepared.key), &samples);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Case {
    Sample,
    SampleReadback,
}

impl Case {
    fn key(self) -> &'static str {
        match self {
            Self::Sample => "sample",
            Self::SampleReadback => "sample_readback",
        }
    }
}

struct Args {
    cases: Vec<Case>,
    rows: Vec<u32>,
    vocab_size: u32,
    top_k: u32,
    temperature: f32,
    top_p: f32,
    seed: u32,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            cases: vec![Case::Sample, Case::SampleReadback],
            rows: vec![1, 2, 4, 8],
            vocab_size: DEFAULT_VOCAB_SIZE,
            top_k: 20,
            temperature: 0.7,
            top_p: 0.8,
            seed: 42,
            iters: 200,
            warmup_iters: 50,
            runs: 7,
        };
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(),
                "--cases" => args.cases = parse_cases(&next_arg(&mut iter, &arg)),
                "--rows" => args.rows = parse_u32_list(&next_arg(&mut iter, &arg), &arg),
                "--vocab-size" => args.vocab_size = parse_u32(&next_arg(&mut iter, &arg), &arg),
                "--top-k" => args.top_k = parse_u32(&next_arg(&mut iter, &arg), &arg),
                "--temperature" => args.temperature = parse_f32(&next_arg(&mut iter, &arg), &arg),
                "--top-p" => args.top_p = parse_f32(&next_arg(&mut iter, &arg), &arg),
                "--seed" => args.seed = parse_u32(&next_arg(&mut iter, &arg), &arg),
                "--iters" => args.iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut iter, &arg), &arg),
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        assert!(!args.cases.is_empty(), "--cases must include at least one case");
        assert!(!args.rows.is_empty(), "--rows must include at least one value");
        assert!(
            args.rows.iter().all(|&value| value > 0),
            "--rows entries must be positive"
        );
        assert!(args.vocab_size > 0, "--vocab-size must be positive");
        assert!(
            i32::try_from(args.vocab_size).is_ok(),
            "--vocab-size must fit the signed token-buffer ABI"
        );
        assert!(args.top_k > 0, "--top-k must be positive");
        assert!(args.top_k <= args.vocab_size, "--top-k must not exceed --vocab-size");
        assert!(args.top_k as usize <= MAX_TOP_K, "--top-k must not exceed {MAX_TOP_K}");
        assert!(
            args.temperature.is_finite() && args.temperature >= 0.0,
            "--temperature must be finite and non-negative"
        );
        assert!(
            args.top_p.is_finite() && (0.0..=1.0).contains(&args.top_p),
            "--top-p must be finite and in [0, 1]"
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

    fn sampler_config(&self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.temperature,
            top_k: self.top_k as usize,
            top_p: self.top_p,
            seed: self.seed,
        }
    }
}

struct MainSamplingFixture {
    stream: Stream,
    logits: Buffer,
    output: TopKSamplingOutputBuffers,
    replay: Replay<Sampling>,
}

struct PreparedReplay {
    key: TopKSamplingReplayKey,
    arguments: ReplayArguments,
    num_rows: u32,
}

impl MainSamplingFixture {
    fn new(device: &Device, args: &Args, max_rows: u32) -> Self {
        let bounds = TopKSamplingBounds::from_config(&args.sampler_config(), max_rows, args.vocab_size)
            .unwrap_or_else(|error| panic!("unable to initialize Qwen3.5 Main sampling bounds: {error}"));
        let params = std::rc::Rc::new(SamplingParamsStore::new(device, bounds, max_rows));
        let sampler = std::rc::Rc::new(TopKSampling::new(device, params));
        let logits = (0..max_rows as usize * args.vocab_size as usize)
            .map(|index| {
                let row = index / args.vocab_size as usize;
                let column = index % args.vocab_size as usize;
                bf16::from_f32(((column * 37 + row * 101) % 997) as f32 * 0.007_812_5 - 3.5).to_bits()
            })
            .collect::<Vec<_>>();
        Self {
            stream: Stream::new(device),
            logits: Buffer::from_slice(device, &logits),
            output: TopKSamplingOutputBuffers::new(device, bounds),
            replay: Replay::new("qwen3.5 Main sampling benchmark", Sampling::new(sampler)),
        }
    }

    fn prepare(&mut self, args: &Args, num_rows: u32) -> PreparedReplay {
        let configs = vec![args.sampler_config(); num_rows as usize];
        let req_slots = (0..num_rows).collect::<Vec<_>>();
        let sample_positions = (0..num_rows).collect::<Vec<_>>();
        let shape = self.replay.component().prepare_shape(&configs);
        self.replay.component().sampler.set_params(&req_slots, &configs);
        self.replay.component().sampler.prepare(&req_slots, &sample_positions);
        let mut arguments = ReplayArguments::new();
        self.replay
            .component()
            .sampler
            .add_replay_arguments(shape, &mut arguments);
        let input = SamplingInput {
            shape,
            inputs: TopKSamplingInputs {
                logits: &self.logits,
                logits_offset_bytes: 0,
                token_mask: None,
            },
            output: self.output.as_output(),
        };
        let runtime = &self.stream;
        let (key, _) = self.replay.record(runtime, &input);
        PreparedReplay {
            key,
            arguments,
            num_rows,
        }
    }

    fn run(&self, case: Case, prepared: &PreparedReplay) {
        self.stream
            .submit_replay_with_arguments(self.replay.replay(&prepared.key), &prepared.arguments)
            .wait();
        if case == Case::SampleReadback {
            let _ = self.output.token_ids.read_typed::<i32>(0, prepared.num_rows as usize);
            let _ = self.output.token_probs.read_typed::<f32>(0, prepared.num_rows as usize);
        }
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

fn print_perf(args: &Args, case: Case, num_rows: u32, replay: &ReplayProgram, samples: &[f64]) {
    let median_us = median(samples);
    let stats = replay.stats();
    let samples_text = samples
        .iter()
        .map(|sample| format!("{sample:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "perf component=qwen35-main-sampling case={} num_rows={num_rows} vocab_size={} top_k={} temperature={} \
         top_p={} seed={} commands={} retained_buffers={} retained_pipelines={} constant_bytes={} iters={} runs={} \
         median_us={median_us:.3} samples_us=[{samples_text}]",
        case.key(),
        args.vocab_size,
        args.top_k,
        args.temperature,
        args.top_p,
        args.seed,
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

fn parse_cases(value: &str) -> Vec<Case> {
    let cases = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            match part {
                "sample" => Case::Sample,
                "sample_readback" => Case::SampleReadback,
                other => panic!("invalid --cases value {other:?}; expected sample or sample_readback"),
            }
        })
        .collect::<Vec<_>>();
    assert!(!cases.is_empty(), "--cases must include at least one case");
    assert!(
        cases
            .iter()
            .enumerate()
            .all(|(index, case)| !cases[..index].contains(case)),
        "--cases must not contain duplicates"
    );
    cases
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

fn parse_f32(value: &str, name: &str) -> f32 {
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
        "qwen35_main_sampling bench\n--cases sample,sample_readback\n--rows 1,2,4,8\n--vocab-size 151936\n--top-k \
         20\n--temperature 0.7\n--top-p 0.8\n--seed 42\n--iters N\n--warmup-iters N\n--runs N"
    );
    std::process::exit(0);
}
