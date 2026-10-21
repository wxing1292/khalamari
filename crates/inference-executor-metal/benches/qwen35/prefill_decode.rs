use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;
use std::time::Instant;

use inference_executor_core::model::ExecutionSubmission;
use inference_executor_core::model::ModelOutputTiming;
use inference_executor_core::model::ReplayableDecoderModel;
use inference_executor_core::model::qwen::v3_5::init_qwen35_model_config;
use inference_executor_core::sampling::MAX_TOP_K;
use inference_executor_metal::model::qwen::v3_5::executor::Qwen35Executor;
use inference_executor_metal::model::qwen::v3_5::executor::Qwen35ExecutorConfig;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::DecoderSyncBlocks;
use inference_runtime_core::compute::DeviceRequest;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::compute::SampledTokens;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::runtime::RawRequestSlot;
use inference_runtime_core::runtime::Token;

const DEFAULT_CONTEXTS: &[usize] = &[0, 1024, 4096, 8192];
const DEFAULT_PREFILL_TOKENS: &[usize] = &[64, 128];
const DEFAULT_DECODE_TOKENS: &[usize] = &[32];
const DEFAULT_NUM_REQUESTS: usize = 1;
const DEFAULT_MAX_TOKENS: usize = 128;
const DEFAULT_MAX_TOKENS_PER_REQUEST: usize = 128;
const DEFAULT_NUM_TOKENS_PER_BLOCK: usize = 2048;
const DEFAULT_NUM_CACHE_PAGES: usize = 393_216;
const DEFAULT_SEED: u32 = 42;
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_TOP_K: usize = 20;
const DEFAULT_TOP_P: f32 = 0.8;
const DEFAULT_WARMUP_ITERS: usize = 2;
const DEFAULT_ITERS: usize = 5;
const DEFAULT_RUNS: usize = 5;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

pub fn run_vanilla() {
    let args = Args::parse();
    let model_config = init_qwen35_model_config(&args.model_dir)
        .unwrap_or_else(|error| panic!("unable to load Qwen3.5 model config: {error}"));
    args.validate_for_model(
        model_config.text_config.vocab_size,
        model_config.text_config.max_position_embeddings,
    );
    let shapes = args.shapes();
    let provenance = Provenance::collect();

    let setup_start = Instant::now();
    let mut fixture = VanillaFixture::new(&args, model_config.text_config.vocab_size, &shapes);
    let setup_elapsed = setup_start.elapsed();

    provenance.print(&args.model_dir);
    if args.reuse_context {
        run_reused_prefill(&args, &shapes, &mut fixture);
        return;
    }
    let expected = fixture.validate_cross_checks(&shapes);
    for shape in shapes {
        println!(
            "bench_start component=qwen35-vanilla-prefill-decode case={} num_reqs={} context={} prefill_tokens={} \
             decode_tokens={} model_dir={}",
            shape.case.key(),
            args.num_reqs,
            shape.context,
            shape.prefill_tokens(),
            shape.decode_tokens(),
            escape_field(&args.model_dir.display().to_string()),
        );

        let expected_shape = expected
            .get(&shape)
            .expect("cross-check result must exist for every benchmark shape");
        let cold = fixture.run_trajectory(shape, fixture.active_chunk_tokens, fixture.active_chunk_tokens, true);
        assert_trajectory_matches(&cold, expected_shape, "replay-cache-cold");

        for _ in 0..args.warmup_iters {
            let warmup = fixture.run_trajectory(shape, fixture.active_chunk_tokens, fixture.active_chunk_tokens, false);
            assert_trajectory_matches(&warmup, expected_shape, "warmup");
        }

        let samples = (0..args.runs)
            .map(|_| {
                let mut sample = RunSample::default();
                for _ in 0..args.iters {
                    let trajectory =
                        fixture.run_trajectory(shape, fixture.active_chunk_tokens, fixture.active_chunk_tokens, false);
                    assert_trajectory_matches(&trajectory, expected_shape, "measured");
                    sample.add_trajectory(&trajectory);
                }
                sample
            })
            .collect::<Vec<_>>();

        print_result(&args, shape, setup_elapsed, &cold, &samples);
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Case {
    Prefill,
    Decode,
}

impl Case {
    fn key(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Shape {
    case: Case,
    context: usize,
    operation_tokens: usize,
}

impl Shape {
    fn prefill_tokens(self) -> usize {
        match self.case {
            Case::Prefill => self.operation_tokens,
            Case::Decode => 0,
        }
    }

    fn decode_tokens(self) -> usize {
        match self.case {
            Case::Prefill => 0,
            Case::Decode => self.operation_tokens,
        }
    }

    fn final_context(self) -> usize {
        self.context
            .checked_add(self.operation_tokens)
            .expect("benchmark final context must fit usize")
    }

    fn result_identity_fields(self) -> (Case, usize, usize, usize, usize) {
        (
            self.case,
            self.context,
            self.prefill_tokens(),
            self.decode_tokens(),
            self.final_context(),
        )
    }
}

#[derive(Clone, Debug)]
struct Args {
    model_dir: PathBuf,
    cases: Vec<Case>,
    contexts: Vec<usize>,
    prefill_tokens: Vec<usize>,
    decode_tokens: Vec<usize>,
    num_reqs: usize,
    max_tokens: usize,
    max_tokens_per_request: usize,
    num_tokens_per_block: usize,
    num_cache_pages: usize,
    seed: u32,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    warmup_iters: usize,
    iters: usize,
    runs: usize,
    reuse_context: bool,
}

impl Args {
    fn parse() -> Self {
        Self::parse_from(std::env::args())
    }

    fn parse_from(values: impl IntoIterator<Item = String>) -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            cases: vec![Case::Prefill, Case::Decode],
            contexts: DEFAULT_CONTEXTS.to_vec(),
            prefill_tokens: DEFAULT_PREFILL_TOKENS.to_vec(),
            decode_tokens: DEFAULT_DECODE_TOKENS.to_vec(),
            num_reqs: DEFAULT_NUM_REQUESTS,
            max_tokens: DEFAULT_MAX_TOKENS,
            max_tokens_per_request: DEFAULT_MAX_TOKENS_PER_REQUEST,
            num_tokens_per_block: DEFAULT_NUM_TOKENS_PER_BLOCK,
            num_cache_pages: DEFAULT_NUM_CACHE_PAGES,
            seed: DEFAULT_SEED,
            temperature: DEFAULT_TEMPERATURE,
            top_k: DEFAULT_TOP_K,
            top_p: DEFAULT_TOP_P,
            warmup_iters: DEFAULT_WARMUP_ITERS,
            iters: DEFAULT_ITERS,
            runs: DEFAULT_RUNS,
            reuse_context: false,
        };
        let mut values = values.into_iter().skip(1);
        while let Some(arg) = values.next() {
            match arg.as_str() {
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut values, &arg)),
                "--cases" => args.cases = parse_cases(&next_arg(&mut values, &arg)),
                "--contexts" => args.contexts = parse_usize_list(&next_arg(&mut values, &arg), &arg, true),
                "--prefill-tokens" => args.prefill_tokens = parse_usize_list(&next_arg(&mut values, &arg), &arg, false),
                "--decode-tokens" => args.decode_tokens = parse_usize_list(&next_arg(&mut values, &arg), &arg, false),
                "--num-reqs" => args.num_reqs = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--max-tokens" => args.max_tokens = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--max-tokens-per-request" => {
                    args.max_tokens_per_request = parse_usize(&next_arg(&mut values, &arg), &arg)
                },
                "--num-tokens-per-block" => args.num_tokens_per_block = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--num-cache-pages" => args.num_cache_pages = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--seed" => args.seed = parse_u32(&next_arg(&mut values, &arg), &arg),
                "--temperature" => args.temperature = parse_f32(&next_arg(&mut values, &arg), &arg),
                "--top-k" => args.top_k = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--top-p" => args.top_p = parse_f32(&next_arg(&mut values, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--iters" => args.iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--reuse-context" => args.reuse_context = true,
                "--bench" => {},
                "--help" | "-h" => print_help_and_exit(),
                _ => panic!("unknown argument {arg:?}; pass --help for usage"),
            }
        }
        args.validate_counts();
        args
    }

    fn validate_counts(&self) {
        assert!(!self.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(!self.cases.is_empty(), "--cases must not be empty");
        assert!(!self.contexts.is_empty(), "--contexts must not be empty");
        assert!(!self.prefill_tokens.is_empty(), "--prefill-tokens must not be empty");
        assert!(!self.decode_tokens.is_empty(), "--decode-tokens must not be empty");
        assert!(self.num_reqs > 0, "--num-reqs must be positive");
        assert!(self.max_tokens > 0, "--max-tokens must be positive");
        assert!(
            self.max_tokens_per_request > 0,
            "--max-tokens-per-request must be positive"
        );
        assert!(self.num_tokens_per_block > 0, "--num-tokens-per-block must be positive");
        assert!(self.num_cache_pages > 0, "--num-cache-pages must be positive");
        assert!(self.top_k > 0, "--top-k must be positive");
        assert!(self.warmup_iters > 0, "--warmup-iters must be positive");
        assert!(self.iters > 0, "--iters must be positive");
        assert!(self.runs > 0, "--runs must be positive");
        assert!(
            self.temperature.is_finite() && self.temperature >= 0.0,
            "--temperature must be finite and non-negative"
        );
        assert!(
            self.top_p.is_finite() && (0.0..=1.0).contains(&self.top_p),
            "--top-p must be finite and in [0, 1]"
        );
        assert!(
            u32::try_from(self.num_reqs).is_ok(),
            "--num-reqs must fit the u32 request-slot domain"
        );
        assert!(
            u32::try_from(self.num_cache_pages - 1).is_ok(),
            "--num-cache-pages must fit the u32 page-ID domain"
        );
        assert!(
            self.max_tokens / self.num_reqs > 0,
            "--max-tokens must process at least one token for every active request"
        );
        if self.reuse_context {
            assert_eq!(
                self.cases,
                vec![Case::Prefill],
                "--reuse-context requires --cases prefill"
            );
            assert!(
                self.contexts
                    .iter()
                    .all(|context| context % self.num_tokens_per_block == 0),
                "--reuse-context requires contexts at state publication boundaries"
            );
        }
    }

    fn validate_for_model(&self, vocab_size: usize, max_position_embeddings: usize) {
        assert!(vocab_size > 0, "Qwen3.5 benchmark requires a positive vocabulary size");
        assert!(
            u32::try_from(vocab_size).is_ok() && i32::try_from(vocab_size).is_ok(),
            "Qwen3.5 vocabulary size must fit the benchmark token domains"
        );
        assert!(self.top_k <= vocab_size, "--top-k must not exceed the model vocabulary");
        assert!(self.top_k <= MAX_TOP_K, "--top-k must not exceed {MAX_TOP_K}");
        for shape in self.shapes() {
            let required_position = match shape.case {
                Case::Prefill => {
                    shape
                        .final_context()
                        .checked_add(1)
                        .expect("post-Prefill Decode probe position must fit usize")
                },
                Case::Decode => shape.final_context(),
            };
            assert!(
                required_position <= max_position_embeddings,
                "benchmark shape requires position {required_position}, but the model supports \
                 {max_position_embeddings}"
            );
            u32::try_from(required_position).expect("benchmark positions must fit the u32 model domain");
        }
    }

    fn active_chunk_tokens(&self) -> usize {
        self.max_tokens_per_request.min(self.max_tokens / self.num_reqs)
    }

    fn shapes(&self) -> Vec<Shape> {
        let mut shapes = Vec::new();
        for &case in &self.cases {
            for &context in &self.contexts {
                let operation_tokens = match case {
                    Case::Prefill => &self.prefill_tokens,
                    Case::Decode => &self.decode_tokens,
                };
                shapes.extend(operation_tokens.iter().copied().map(|operation_tokens| {
                    Shape {
                        case,
                        context,
                        operation_tokens,
                    }
                }));
            }
        }
        shapes
    }

    fn sampling_config(&self) -> SamplingConfig {
        SamplingConfig {
            max_sampled_tokens: 1,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            seed: Some(self.seed),
            stop_sequences: Vec::new(),
        }
    }
}

struct VanillaFixture {
    model: Qwen35Executor,
    pages: PageFixture,
    sampling_config: SamplingConfig,
    num_reqs: usize,
    vocab_size: u32,
    active_chunk_tokens: usize,
    next_sequence: u64,
    next_epoch: usize,
}

impl VanillaFixture {
    fn new(args: &Args, vocab_size: usize, shapes: &[Shape]) -> Self {
        let config = Qwen35ExecutorConfig {
            max_requests: args.num_reqs,
            max_tokens: args.max_tokens,
            max_tokens_per_request: args.max_tokens_per_request,
            num_cache_pages: args.num_cache_pages,
            num_tokens_per_block: args.num_tokens_per_block,
        };
        let model = init_qwen_3_5_model(&args.model_dir, config)
            .unwrap_or_else(|error| panic!("unable to initialize qwen35_vanilla_prefill_decode: {error}"));
        assert_eq!(model.model_mode(), "vanilla", "benchmark target requires Vanilla mode");
        assert!(
            model.num_mtp_gqa_page_ids_per_block().is_empty(),
            "Vanilla benchmark must not have MTP cache lanes"
        );
        let max_required_position = shapes
            .iter()
            .map(|shape| shape.final_context() + usize::from(shape.case == Case::Prefill))
            .max()
            .expect("benchmark requires at least one shape");
        let pages = PageFixture::new(
            args.num_reqs,
            max_required_position,
            args.num_tokens_per_block,
            args.num_cache_pages,
            &[model.num_main_lane_gqa_page_ids_per_block()],
            &[model.num_gdn_state_page_ids_per_block()],
        );
        Self {
            model,
            pages,
            sampling_config: args.sampling_config(),
            num_reqs: args.num_reqs,
            vocab_size: vocab_size
                .try_into()
                .expect("Qwen3.5 vocabulary size must fit the u32 token domain"),
            active_chunk_tokens: args.active_chunk_tokens(),
            next_sequence: 0,
            next_epoch: 0,
        }
    }

    fn validate_cross_checks(&mut self, shapes: &[Shape]) -> BTreeMap<Shape, ExpectedTrajectory> {
        let mut expected = BTreeMap::new();
        for &shape in shapes.iter().rev() {
            let standard = self.run_trajectory(shape, self.active_chunk_tokens, self.active_chunk_tokens, false);
            match shape.case {
                Case::Prefill => self.validate_prefill_decomposition(shape),
                Case::Decode => self.validate_decode_context_decomposition(shape),
            }
            println!(
                "bench_check component=qwen35-vanilla-prefill-decode check=case-order-and-chunk-consistency case={} \
                 context={} prefill_tokens={} decode_tokens={} input_fingerprint={:016x} output_fingerprint={:016x} \
                 status=pass",
                shape.case.key(),
                shape.context,
                shape.prefill_tokens(),
                shape.decode_tokens(),
                standard.identity.input_fingerprint,
                standard.identity.output_fingerprint,
            );
            assert!(
                expected
                    .insert(
                        shape,
                        ExpectedTrajectory {
                            identity: standard.identity,
                            outputs: standard.outputs,
                        },
                    )
                    .is_none(),
                "benchmark shapes must be unique"
            );
        }
        expected
    }

    fn validate_prefill_decomposition(&mut self, shape: Shape) {
        let standard = self.run_greedy_validation_trajectory(shape, self.active_chunk_tokens, self.active_chunk_tokens);
        let combined_shape = Shape {
            case: Case::Prefill,
            context: 0,
            operation_tokens: shape.final_context(),
        };
        let combined =
            self.run_greedy_validation_trajectory(combined_shape, self.active_chunk_tokens, self.active_chunk_tokens);
        assert_decomposition_output_tokens(
            &standard.outputs,
            &combined.outputs,
            "committed prefix plus suffix must match full Prefill at the deterministic Decode probe",
        );

        if shape.operation_tokens > 1 && shape.operation_tokens <= self.active_chunk_tokens {
            let smaller_chunk = shape.operation_tokens.div_ceil(2).min(shape.operation_tokens - 1);
            let split = self.run_greedy_validation_trajectory(shape, self.active_chunk_tokens, smaller_chunk);
            assert_decomposition_identity(
                standard.identity,
                split.identity,
                "one large Prefill chunk must preserve the Prefill input identity",
            );
            assert_decomposition_output_tokens(
                &standard.outputs,
                &split.outputs,
                "one large Prefill chunk must match multiple smaller chunks",
            );
        }
    }

    fn validate_decode_context_decomposition(&mut self, shape: Shape) {
        if shape.context > 1 && self.active_chunk_tokens > 1 {
            let probe_shape = Shape {
                operation_tokens: 1,
                ..shape
            };
            let standard =
                self.run_greedy_validation_trajectory(probe_shape, self.active_chunk_tokens, self.active_chunk_tokens);
            let smaller_chunk = (self.active_chunk_tokens / 2).max(1);
            let split = self.run_greedy_validation_trajectory(probe_shape, smaller_chunk, self.active_chunk_tokens);
            assert_decomposition_output_tokens(
                &standard.outputs,
                &split.outputs,
                "single-step Decode output must not depend on Prefill chunk decomposition",
            );
            assert_decomposition_identity(
                standard.identity,
                split.identity,
                "Decode context chunking must preserve the single-step Decode input identity",
            );
        }
    }

    fn run_greedy_validation_trajectory(
        &mut self,
        shape: Shape,
        context_chunk_tokens: usize,
        operation_chunk_tokens: usize,
    ) -> TrajectoryResult {
        let validation_sampling_config = greedy_validation_sampling_config(&self.sampling_config);
        let requested_sampling_config = std::mem::replace(&mut self.sampling_config, validation_sampling_config);
        let result = self.run_trajectory(shape, context_chunk_tokens, operation_chunk_tokens, false);
        self.sampling_config = requested_sampling_config;
        result
    }

    fn run_trajectory(
        &mut self,
        shape: Shape,
        context_chunk_tokens: usize,
        operation_chunk_tokens: usize,
        clear_replay_cache: bool,
    ) -> TrajectoryResult {
        assert!(context_chunk_tokens > 0 && operation_chunk_tokens > 0);
        let epoch = self.next_epoch;
        self.next_epoch = self
            .next_epoch
            .checked_add(1)
            .expect("benchmark request epoch must fit usize");

        let rebuild_start = Instant::now();
        let req_slots = (0..self.num_reqs)
            .map(|req_index| req_index.try_into().expect("benchmark request slot must fit u32"))
            .collect::<Vec<RawRequestSlot>>();
        self.model.reset_req_slots(&req_slots);
        self.pages.reset_sync();
        if shape.context > 0 {
            let (context_timing, consumed) = self.execute_prefill_range(0, shape.context, context_chunk_tokens, epoch);
            assert_eq!(consumed, shape.context);
            assert_stage_accounting(&context_timing);
        }
        let context_rebuild = rebuild_start.elapsed();

        if clear_replay_cache {
            self.model.clear_replay_cache();
        }

        let mut input_streams = self.deterministic_streams(0, shape.context);
        let (operation, outputs) = match shape.case {
            Case::Prefill => {
                for (req_index, stream) in input_streams.iter_mut().enumerate() {
                    stream.extend(
                        (shape.context..shape.final_context())
                            .map(|position| deterministic_token(req_index, position, self.vocab_size)),
                    );
                }
                let operation_start = Instant::now();
                let (mut timing, consumed) =
                    self.execute_prefill_range(shape.context, shape.operation_tokens, operation_chunk_tokens, epoch);
                timing.wall = operation_start.elapsed();
                let mut accounting = OperationAccounting::new(shape);
                accounting.commit(shape.operation_tokens);
                accounting.finish();
                assert_eq!(consumed, shape.operation_tokens);
                assert_stage_accounting(&timing);
                let outputs = self.execute_decode_probe(shape.final_context(), epoch);
                (timing, outputs)
            },
            Case::Decode => {
                let (timing, decode_inputs, outputs) =
                    self.execute_decode(shape.context, shape.operation_tokens, epoch);
                for (stream, decode_inputs) in input_streams.iter_mut().zip(decode_inputs) {
                    stream.extend(decode_inputs);
                }
                assert_stage_accounting(&timing);
                (timing, outputs)
            },
        };

        let identity = TrajectoryIdentity {
            result_fields: shape.result_identity_fields(),
            input_fingerprint: fingerprint_token_streams(&input_streams),
            output_fingerprint: fingerprint_output_streams(&outputs),
        };
        assert_eq!(identity.result_fields.4, shape.final_context());
        TrajectoryResult {
            identity,
            outputs,
            context_rebuild,
            operation,
        }
    }

    fn deterministic_streams(&self, start: usize, num_tokens: usize) -> Vec<Vec<u32>> {
        (0..self.num_reqs)
            .map(|req_index| {
                (start..start + num_tokens)
                    .map(|position| deterministic_token(req_index, position, self.vocab_size))
                    .collect()
            })
            .collect()
    }

    fn execute_prefill_range(
        &mut self,
        start_position: usize,
        num_tokens: usize,
        max_chunk_tokens: usize,
        epoch: usize,
    ) -> (ExecutionTiming, usize) {
        assert!(num_tokens > 0, "Prefill range must contain tokens");
        let chunks = plan_prefill_chunks(num_tokens, max_chunk_tokens);
        let mut timing = ExecutionTiming::default();
        let mut consumed = 0usize;
        for chunk_tokens in chunks {
            let token_index = start_position
                .checked_add(consumed)
                .expect("Prefill token index must fit usize");
            let final_context = token_index
                .checked_add(chunk_tokens)
                .expect("Prefill chunk end must fit usize");
            let sequence = self.next_sequence();
            let requests = (0..self.num_reqs)
                .map(|req_index| {
                    let tokens = (token_index..final_context)
                        .map(|position| Token::new(deterministic_token(req_index, position, self.vocab_size)))
                        .collect::<Vec<_>>();
                    DeviceRequest::new(
                        req_index,
                        req_index.try_into().expect("benchmark request slot must fit u32"),
                        QueryTokens::Prefill {
                            epoch,
                            token_index,
                            tokens,
                            window: chunk_tokens,
                        },
                        self.pages.sync_blocks(req_index, final_context),
                        None,
                        vec![],
                        self.sampling_config.clone(),
                    )
                })
                .collect::<Vec<_>>();
            let (batch_timing, response) =
                self.execute_batch(BatchDeviceRequest::new(sequence, requests), BatchKind::Prefill);
            self.verify_prefill_response(&response, sequence, epoch, token_index, chunk_tokens);
            timing.add_assign(batch_timing);
            consumed = consumed
                .checked_add(chunk_tokens)
                .expect("Prefill consumed-token count must fit usize");
        }
        assert_eq!(consumed, num_tokens, "Prefill must consume the exact requested total");
        (timing, consumed)
    }

    fn execute_decode(
        &mut self,
        context: usize,
        decode_tokens: usize,
        epoch: usize,
    ) -> (ExecutionTiming, Vec<Vec<u32>>, Vec<Vec<OutputSample>>) {
        let mut accounting = OperationAccounting::new(Shape {
            case: Case::Decode,
            context,
            operation_tokens: decode_tokens,
        });
        let mut input_streams = vec![Vec::with_capacity(decode_tokens); self.num_reqs];
        let mut output_streams = vec![Vec::with_capacity(decode_tokens); self.num_reqs];
        let mut current_tokens = (0..self.num_reqs)
            .map(|req_index| deterministic_token(req_index, context, self.vocab_size))
            .collect::<Vec<_>>();
        let operation_start = Instant::now();
        let mut timing = ExecutionTiming::default();
        for token_offset in 0..decode_tokens {
            let token_index = context
                .checked_add(token_offset)
                .expect("Decode token index must fit usize");
            let final_context = token_index.checked_add(1).expect("Decode context must fit usize");
            let sequence = self.next_sequence();
            let requests = current_tokens
                .iter()
                .copied()
                .enumerate()
                .map(|(req_index, token)| {
                    input_streams[req_index].push(token);
                    DeviceRequest::new(
                        req_index,
                        req_index.try_into().expect("benchmark request slot must fit u32"),
                        QueryTokens::Decode {
                            epoch,
                            token_index,
                            tokens: vec![Token::new(token)],
                            spec_tokens: Vec::new(),
                        },
                        self.pages.sync_blocks(req_index, final_context),
                        None,
                        vec![],
                        self.sampling_config.clone(),
                    )
                })
                .collect::<Vec<_>>();
            let (batch_timing, response) =
                self.execute_batch(BatchDeviceRequest::new(sequence, requests), BatchKind::Decode);
            current_tokens = self.verify_decode_response(&response, sequence, epoch, token_index, &mut output_streams);
            timing.add_assign(batch_timing);
            accounting.commit(1);
        }
        timing.wall = operation_start.elapsed();
        accounting.finish();
        assert_eq!(
            output_streams.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![decode_tokens; self.num_reqs],
            "Decode must commit the requested visible-token count for every request"
        );
        (timing, input_streams, output_streams)
    }

    fn execute_decode_probe(&mut self, token_index: usize, epoch: usize) -> Vec<Vec<OutputSample>> {
        let sequence = self.next_sequence();
        let final_context = token_index.checked_add(1).expect("Decode probe context must fit usize");
        let requests = (0..self.num_reqs)
            .map(|req_index| {
                let token = deterministic_token(req_index, token_index, self.vocab_size);
                DeviceRequest::new(
                    req_index,
                    req_index.try_into().expect("benchmark request slot must fit u32"),
                    QueryTokens::Decode {
                        epoch,
                        token_index,
                        tokens: vec![Token::new(token)],
                        spec_tokens: Vec::new(),
                    },
                    self.pages.sync_blocks(req_index, final_context),
                    None,
                    vec![],
                    self.sampling_config.clone(),
                )
            })
            .collect::<Vec<_>>();
        let (timing, response) = self.execute_batch(BatchDeviceRequest::new(sequence, requests), BatchKind::Decode);
        assert_stage_accounting(&timing);
        let mut outputs = vec![Vec::with_capacity(1); self.num_reqs];
        let _ = self.verify_decode_response(&response, sequence, epoch, token_index, &mut outputs);
        outputs
    }

    fn execute_batch(
        &mut self,
        mut core_batch_req: BatchDeviceRequest,
        kind: BatchKind,
    ) -> (ExecutionTiming, BatchDeviceResponse) {
        let wall_start = Instant::now();
        let prepare_start = Instant::now();
        let model_batch_req = self.model.prepare_batch(&mut core_batch_req);
        let prepare = prepare_start.elapsed();

        let record_start = Instant::now();
        let mut recorder = self.model.begin_ops_recording(&model_batch_req);
        let hidden = self.model.embed_main(&mut recorder, &model_batch_req);
        let hidden = self.model.forward_main(&mut recorder, &model_batch_req, hidden);
        if kind == BatchKind::Decode {
            let output = self.model.unembed_main(&mut recorder, &model_batch_req, &hidden);
            self.model.sample_main(&mut recorder, &model_batch_req, &output);
        }
        let record = record_start.elapsed();

        let finish_start = Instant::now();
        let replay_start = Instant::now();
        let submission = self.model.submit_main(&recorder);
        submission.wait();
        let main_submit_wait = replay_start.elapsed();
        let sampled = match kind {
            BatchKind::Prefill => self.model.empty_sampled_output(),
            BatchKind::Decode => {
                let gpu_timestamp_durations = submission.gpu_timestamp_durations();
                self.model.read_main(
                    &recorder,
                    &model_batch_req,
                    main_submit_wait,
                    gpu_timestamp_durations.as_deref(),
                )
            },
        };
        assert!(!self.model.run_spec(&model_batch_req, &sampled));
        assert!(!self.model.run_spec_prefill(&model_batch_req));
        assert!(!self.model.run_spec_decode(&model_batch_req, &sampled));
        drop(recorder);
        let finish = finish_start.elapsed();

        let mut stage = self.model.sampled_output_timing(&sampled).unwrap_or_default();
        match kind {
            BatchKind::Prefill => {
                assert!(
                    stage.is_zero(),
                    "Vanilla Prefill must not produce sampled-output timing"
                );
                stage.main_replay_elapsed = main_submit_wait;
            },
            BatchKind::Decode => {
                assert_eq!(
                    stage.main_replay_elapsed + stage.main_sample_replay_elapsed,
                    main_submit_wait,
                    "Vanilla Decode ModelOutputTiming must match the Main submit and wait"
                );
            },
        }

        let feedback_start = Instant::now();
        let response = self.model.commit_batch(core_batch_req, sampled);
        let feedback = feedback_start.elapsed();
        let wall = wall_start.elapsed();
        (
            ExecutionTiming {
                wall,
                batch_wall: wall,
                stage,
                prepare,
                record,
                finish,
                feedback,
                main_submit_wait,
            },
            response,
        )
    }

    fn verify_prefill_response(
        &self,
        response: &BatchDeviceResponse,
        sequence: u64,
        epoch: usize,
        token_index: usize,
        chunk_tokens: usize,
    ) {
        assert_eq!(response.seq, sequence, "Prefill response sequence must match request");
        assert_eq!(response.dev_resps.len(), self.num_reqs);
        for (req_index, response) in response.dev_resps.iter().enumerate() {
            assert_eq!(response.req_id, req_index);
            let QueryTokens::Prefill {
                epoch: response_epoch,
                token_index: response_token_index,
                window,
                ..
            } = &response.query_tokens
            else {
                panic!("Prefill response must preserve Prefill query tokens");
            };
            assert_eq!(*response_epoch, epoch);
            assert_eq!(*response_token_index, token_index);
            assert_eq!(*window, chunk_tokens);
            let SampledTokens::Prefill { epoch: sampled_epoch } = &response.sampled_tokens else {
                panic!("Prefill response must contain Prefill sampled tokens");
            };
            assert_eq!(*sampled_epoch, epoch, "Prefill response epoch must match request");
        }
    }

    fn verify_decode_response(
        &self,
        response: &BatchDeviceResponse,
        sequence: u64,
        epoch: usize,
        token_index: usize,
        output_streams: &mut [Vec<OutputSample>],
    ) -> Vec<u32> {
        assert_eq!(response.seq, sequence, "Decode response sequence must match request");
        assert_eq!(response.dev_resps.len(), self.num_reqs);
        assert_eq!(output_streams.len(), self.num_reqs);
        let mut next_tokens = Vec::with_capacity(self.num_reqs);
        for (req_index, response) in response.dev_resps.iter().enumerate() {
            assert_eq!(response.req_id, req_index);
            let QueryTokens::Decode {
                epoch: response_epoch,
                token_index: response_token_index,
                tokens,
                spec_tokens,
            } = &response.query_tokens
            else {
                panic!("Decode response must preserve Decode query tokens");
            };
            assert_eq!(*response_epoch, epoch);
            assert_eq!(*response_token_index, token_index);
            assert_eq!(tokens.len(), 1);
            assert!(spec_tokens.is_empty(), "Vanilla Decode must not use speculative input");
            let SampledTokens::Decode {
                epoch: sampled_epoch,
                validated_tokens,
                validated_probs,
                sampled_token,
                sampled_prob,
                spec_tokens,
                spec_probs,
                spec_confidences,
            } = &response.sampled_tokens
            else {
                panic!("Decode response must contain Decode sampled tokens");
            };
            assert_eq!(*sampled_epoch, epoch, "Decode response epoch must match request");
            assert!(validated_tokens.is_empty() && validated_probs.is_empty());
            assert!(spec_tokens.is_empty() && spec_probs.is_empty() && spec_confidences.is_empty());
            let token = sampled_token.value();
            let probability = sampled_prob.into_inner();
            assert!(probability.is_finite() && (0.0..=1.0).contains(&probability));
            output_streams[req_index].push(OutputSample {
                token,
                probability_bits: probability.to_bits(),
            });
            next_tokens.push(token);
        }
        next_tokens
    }

    fn next_sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .expect("benchmark request sequence must fit u64");
        sequence
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BatchKind {
    Prefill,
    Decode,
}

#[derive(Clone, Debug)]
struct PageFixture {
    requests: Vec<RequestPages>,
    num_tokens_per_block: usize,
    max_blocks: usize,
    num_cache_pages: usize,
}

#[derive(Clone, Debug)]
struct RequestPages {
    kv_by_lane: Vec<Vec<Vec<u32>>>,
    state_by_lane: Vec<Vec<Vec<u32>>>,
    num_in_sync_blocks: usize,
}

impl PageFixture {
    fn new(
        num_reqs: usize,
        max_position: usize,
        num_tokens_per_block: usize,
        num_cache_pages: usize,
        kv_page_ids_per_block_by_lane: &[usize],
        state_page_ids_per_block_by_lane: &[usize],
    ) -> Self {
        assert!(num_reqs > 0);
        assert!(max_position > 0);
        assert!(num_tokens_per_block > 0);
        assert!(num_cache_pages > 0);
        assert!(!kv_page_ids_per_block_by_lane.is_empty());
        assert!(!state_page_ids_per_block_by_lane.is_empty());
        assert!(kv_page_ids_per_block_by_lane.iter().all(|&count| count > 0));
        assert!(state_page_ids_per_block_by_lane.iter().all(|&count| count > 0));
        assert!(u32::try_from(num_cache_pages - 1).is_ok());
        let max_blocks = max_position.div_ceil(num_tokens_per_block);
        let pages_per_request = kv_page_ids_per_block_by_lane
            .iter()
            .chain(state_page_ids_per_block_by_lane)
            .try_fold(0usize, |sum, &page_ids_per_block| {
                page_ids_per_block
                    .checked_mul(max_blocks)
                    .and_then(|pages| sum.checked_add(pages))
            })
            .expect("benchmark page count must fit usize");
        let required_pages = pages_per_request
            .checked_mul(num_reqs)
            .expect("benchmark page count must fit usize");
        assert!(
            required_pages <= num_cache_pages,
            "benchmark fixture requires {required_pages} unique page IDs, but --num-cache-pages is {num_cache_pages}"
        );
        u32::try_from(required_pages).expect("benchmark allocated page-ID endpoint must fit u32");

        let mut next_page_id = 0_u32;
        let requests = (0..num_reqs)
            .map(|_| {
                let kv_by_lane = allocate_page_lanes(kv_page_ids_per_block_by_lane, max_blocks, &mut next_page_id);
                let state_by_lane =
                    allocate_page_lanes(state_page_ids_per_block_by_lane, max_blocks, &mut next_page_id);
                RequestPages {
                    kv_by_lane,
                    state_by_lane,
                    num_in_sync_blocks: 0,
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(next_page_id as usize, required_pages);
        let unique_page_ids = requests
            .iter()
            .flat_map(|request| request.kv_by_lane.iter().chain(&request.state_by_lane))
            .flat_map(|lane| lane.iter())
            .flat_map(|block| block.iter().copied())
            .collect::<BTreeSet<_>>();
        assert_eq!(unique_page_ids.len(), required_pages, "page IDs must be unique");

        Self {
            requests,
            num_tokens_per_block,
            max_blocks,
            num_cache_pages,
        }
    }

    fn reset_sync(&mut self) {
        for request in &mut self.requests {
            request.num_in_sync_blocks = 0;
        }
    }

    fn sync_blocks(&mut self, req_index: usize, final_context: usize) -> DecoderSyncBlocks {
        let request = self
            .requests
            .get_mut(req_index)
            .expect("benchmark request index must fit the page fixture");
        let required_blocks = final_context.div_ceil(self.num_tokens_per_block);
        assert!(
            required_blocks <= self.max_blocks,
            "benchmark page-table capacity exceeded"
        );
        let block_index = request.num_in_sync_blocks;
        assert!(block_index <= required_blocks, "page synchronization must be monotonic");
        let kv_page_ids = request
            .kv_by_lane
            .iter()
            .map(|lane| lane[block_index..required_blocks].to_vec())
            .collect::<Vec<_>>();
        let state_page_ids = request
            .state_by_lane
            .iter()
            .map(|lane| lane[block_index..required_blocks].to_vec())
            .collect::<Vec<_>>();
        request.num_in_sync_blocks = required_blocks;
        assert!(
            kv_page_ids
                .iter()
                .chain(&state_page_ids)
                .flat_map(|lane| lane.iter())
                .flat_map(|block| block.iter())
                .all(|&page_id| (page_id as usize) < self.num_cache_pages),
            "synced page IDs must fit the shared executor page arena"
        );
        DecoderSyncBlocks::new(block_index, kv_page_ids, state_page_ids)
    }
}

fn allocate_page_lanes(
    page_ids_per_block_by_lane: &[usize],
    num_blocks: usize,
    next_page_id: &mut u32,
) -> Vec<Vec<Vec<u32>>> {
    page_ids_per_block_by_lane
        .iter()
        .map(|&page_ids_per_block| {
            (0..num_blocks)
                .map(|_| next_page_ids(next_page_id, page_ids_per_block))
                .collect()
        })
        .collect()
}

fn next_page_ids(next_page_id: &mut u32, count: usize) -> Vec<u32> {
    let first = *next_page_id;
    *next_page_id = next_page_id
        .checked_add(count.try_into().expect("page-ID count must fit u32"))
        .expect("page-ID endpoint must fit u32");
    (first..*next_page_id).collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OperationAccounting {
    shape: Shape,
    committed_tokens: usize,
}

impl OperationAccounting {
    fn new(shape: Shape) -> Self {
        Self {
            shape,
            committed_tokens: 0,
        }
    }

    fn commit(&mut self, tokens: usize) {
        self.committed_tokens = self
            .committed_tokens
            .checked_add(tokens)
            .expect("committed-token count must fit usize");
        assert!(
            self.committed_tokens <= self.shape.operation_tokens,
            "operation committed more visible tokens than requested"
        );
    }

    fn finish(self) -> usize {
        assert_eq!(
            self.committed_tokens, self.shape.operation_tokens,
            "operation must commit the exact requested token count"
        );
        let final_context = self
            .shape
            .context
            .checked_add(self.committed_tokens)
            .expect("final committed context must fit usize");
        assert_eq!(final_context, self.shape.final_context());
        final_context
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ExecutionTiming {
    wall: Duration,
    batch_wall: Duration,
    stage: ModelOutputTiming,
    prepare: Duration,
    record: Duration,
    finish: Duration,
    feedback: Duration,
    main_submit_wait: Duration,
}

impl ExecutionTiming {
    fn add_assign(&mut self, other: Self) {
        self.wall += other.wall;
        self.batch_wall += other.batch_wall;
        self.stage.add_assign(other.stage);
        self.prepare += other.prepare;
        self.record += other.record;
        self.finish += other.finish;
        self.feedback += other.feedback;
        self.main_submit_wait += other.main_submit_wait;
    }

    fn finish_cpu_estimate(self) -> Duration {
        self.finish.saturating_sub(submit_wait_elapsed(&self.stage))
    }
}

fn submit_wait_elapsed(timing: &ModelOutputTiming) -> Duration {
    timing.main_replay_elapsed
        + timing.main_sample_replay_elapsed
        + timing.main_spec_replay_elapsed
        + timing.spec_replay_elapsed
}

fn assert_stage_accounting(timing: &ExecutionTiming) {
    assert_eq!(
        submit_wait_elapsed(&timing.stage),
        timing.main_submit_wait,
        "ModelOutputTiming replay totals must match committed batch submissions"
    );
    assert_eq!(timing.stage.spec_replay_elapsed, Duration::ZERO);
    assert_eq!(timing.stage.spec_passes, 0);
    assert!(timing.batch_wall <= timing.wall || timing.wall == Duration::ZERO);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OutputSample {
    token: u32,
    probability_bits: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TrajectoryIdentity {
    result_fields: (Case, usize, usize, usize, usize),
    input_fingerprint: u64,
    output_fingerprint: u64,
}

struct TrajectoryResult {
    identity: TrajectoryIdentity,
    outputs: Vec<Vec<OutputSample>>,
    context_rebuild: Duration,
    operation: ExecutionTiming,
}

struct ExpectedTrajectory {
    identity: TrajectoryIdentity,
    outputs: Vec<Vec<OutputSample>>,
}

fn assert_trajectory_matches(actual: &TrajectoryResult, expected: &ExpectedTrajectory, phase: &str) {
    assert_eq!(
        actual.identity, expected.identity,
        "{phase} trajectory identity must match the cross-check"
    );
    assert_eq!(
        actual.outputs, expected.outputs,
        "{phase} output tokens and probability bits must match the cross-check"
    );
}

fn assert_decomposition_identity(expected: TrajectoryIdentity, actual: TrajectoryIdentity, message: &str) {
    assert_eq!(expected.result_fields, actual.result_fields, "{message}");
    assert_eq!(expected.input_fingerprint, actual.input_fingerprint, "{message}");
}

fn assert_decomposition_output_tokens(expected: &[Vec<OutputSample>], actual: &[Vec<OutputSample>], message: &str) {
    assert_eq!(expected.len(), actual.len(), "{message}");
    for (request_index, (expected_request, actual_request)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(expected_request.len(), actual_request.len(), "{message}");
        for (token_index, (&expected, &actual)) in expected_request.iter().zip(actual_request).enumerate() {
            assert_eq!(
                expected.token, actual.token,
                "{message}: request_index={request_index} token_index={token_index}"
            );
        }
    }
}

fn greedy_validation_sampling_config(config: &SamplingConfig) -> SamplingConfig {
    SamplingConfig {
        temperature: 0.0,
        top_p: 0.0,
        ..config.clone()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct RunSample {
    operation: ExecutionTiming,
    context_rebuild: Duration,
}

impl RunSample {
    fn add_trajectory(&mut self, trajectory: &TrajectoryResult) {
        self.operation.add_assign(trajectory.operation);
        self.context_rebuild += trajectory.context_rebuild;
    }
}

fn plan_prefill_chunks(total_tokens: usize, max_chunk_tokens: usize) -> Vec<usize> {
    assert!(total_tokens > 0, "Prefill token total must be positive");
    assert!(max_chunk_tokens > 0, "Prefill chunk capacity must be positive");
    let mut remaining = total_tokens;
    let mut chunks = Vec::with_capacity(total_tokens.div_ceil(max_chunk_tokens));
    while remaining > 0 {
        let chunk = remaining.min(max_chunk_tokens);
        chunks.push(chunk);
        remaining -= chunk;
    }
    assert_eq!(chunks.iter().sum::<usize>(), total_tokens);
    chunks
}

fn deterministic_token(req_index: usize, position: usize, vocab_size: u32) -> u32 {
    assert!(vocab_size > 0);
    let req = u64::try_from(req_index).expect("request index must fit u64");
    let position = u64::try_from(position).expect("token position must fit u64");
    let mut value = position ^ req.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value % u64::from(vocab_size)) as u32
}

fn fingerprint_token_streams(streams: &[Vec<u32>]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    fnv_u64(&mut hash, streams.len() as u64);
    for (req_index, stream) in streams.iter().enumerate() {
        fnv_u64(&mut hash, req_index as u64);
        fnv_u64(&mut hash, stream.len() as u64);
        for &token in stream {
            fnv_u32(&mut hash, token);
        }
    }
    hash
}

fn fingerprint_output_streams(streams: &[Vec<OutputSample>]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    fnv_u64(&mut hash, streams.len() as u64);
    for (req_index, stream) in streams.iter().enumerate() {
        fnv_u64(&mut hash, req_index as u64);
        fnv_u64(&mut hash, stream.len() as u64);
        for output in stream {
            fnv_u32(&mut hash, output.token);
            fnv_u32(&mut hash, output.probability_bits);
        }
    }
    hash
}

fn fnv_u32(hash: &mut u64, value: u32) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn fnv_u64(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn print_result(args: &Args, shape: Shape, setup_elapsed: Duration, cold: &TrajectoryResult, samples: &[RunSample]) {
    let wall_us = sorted_metric(samples, args.iters, |sample| sample.operation.wall);
    let context_rebuild_us = sorted_metric(samples, args.iters, |sample| sample.context_rebuild);
    let main_us = sorted_metric(samples, args.iters, |sample| sample.operation.stage.main_replay_elapsed);
    let main_sample_us = sorted_metric(samples, args.iters, |sample| {
        sample.operation.stage.main_sample_replay_elapsed
    });
    let spec_us = sorted_metric(samples, args.iters, |sample| sample.operation.stage.spec_replay_elapsed);
    let prepare_us = sorted_metric(samples, args.iters, |sample| sample.operation.prepare);
    let record_us = sorted_metric(samples, args.iters, |sample| sample.operation.record);
    let finish_us = sorted_metric(samples, args.iters, |sample| sample.operation.finish_cpu_estimate());
    let feedback_us = sorted_metric(samples, args.iters, |sample| sample.operation.feedback);
    let wall_median_us = median_of_sorted(&wall_us);
    let metric_tokens = args
        .num_reqs
        .checked_mul(shape.operation_tokens)
        .expect("benchmark throughput numerator must fit usize");
    let tokens_per_s = metric_tokens as f64 * 1.0e6 / wall_median_us;
    let token_latency_us = match shape.case {
        Case::Prefill => 0.0,
        Case::Decode => wall_median_us / shape.decode_tokens() as f64,
    };
    let samples_field = wall_us
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "perf component=qwen35-vanilla-prefill-decode model_dir={} case={} num_reqs={} context={} prefill_tokens={} \
         decode_tokens={} final_context={} max_tokens={} max_tokens_per_request={} num_tokens_per_block={} \
         num_cache_pages={} input_fingerprint={:016x} output_fingerprint={:016x} seed={} temperature={} top_k={} \
         top_p={} iters={} runs={} setup_us={:.3} replay_cache_cold_wall_us={:.3} \
         replay_cache_cold_context_rebuild_us={:.3} wall_median_us={:.3} tokens_per_s={:.3} token_latency_us={:.3} \
         context_rebuild_median_us={:.3} main_median_us={:.3} main_sample_median_us={:.3} spec_median_us={:.3} \
         prepare_median_us={:.3} record_cpu_estimate_median_us={:.3} finish_cpu_estimate_median_us={:.3} \
         feedback_median_us={:.3} samples_us=[{}]",
        escape_field(&args.model_dir.display().to_string()),
        shape.case.key(),
        args.num_reqs,
        shape.context,
        shape.prefill_tokens(),
        shape.decode_tokens(),
        shape.final_context(),
        args.max_tokens,
        args.max_tokens_per_request,
        args.num_tokens_per_block,
        args.num_cache_pages,
        cold.identity.input_fingerprint,
        cold.identity.output_fingerprint,
        args.seed,
        args.temperature,
        args.top_k,
        args.top_p,
        args.iters,
        samples.len(),
        setup_elapsed.as_secs_f64() * 1.0e6,
        cold.operation.wall.as_secs_f64() * 1.0e6,
        cold.context_rebuild.as_secs_f64() * 1.0e6,
        wall_median_us,
        tokens_per_s,
        token_latency_us,
        median_of_sorted(&context_rebuild_us),
        median_of_sorted(&main_us),
        median_of_sorted(&main_sample_us),
        median_of_sorted(&spec_us),
        median_of_sorted(&prepare_us),
        median_of_sorted(&record_us),
        median_of_sorted(&finish_us),
        median_of_sorted(&feedback_us),
        samples_field,
    );
}

fn sorted_metric(samples: &[RunSample], iters: usize, metric: impl Fn(&RunSample) -> Duration) -> Vec<f64> {
    let mut values = samples
        .iter()
        .map(|sample| metric(sample).as_secs_f64() * 1.0e6 / iters as f64)
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values
}

fn median_of_sorted(samples: &[f64]) -> f64 {
    assert!(!samples.is_empty());
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) * 0.5
    } else {
        samples[mid]
    }
}

struct Provenance {
    commit: String,
    dirty: bool,
    machine: String,
    os: String,
    architecture: &'static str,
    environment: String,
}

impl Provenance {
    fn collect() -> Self {
        let commit = command_output("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
        let dirty = command_output("git", &["status", "--porcelain", "--untracked-files=normal"])
            .is_some_and(|output| !output.is_empty());
        let machine = command_output("sysctl", &["-n", "hw.model"])
            .or_else(|| std::env::var("HOSTNAME").ok())
            .unwrap_or_else(|| "unknown".into());
        let os = command_output("sw_vers", &["-productVersion"])
            .map(|version| format!("macOS-{version}"))
            .or_else(|| command_output("uname", &["-sr"]))
            .unwrap_or_else(|| "unknown".into());
        let mut environment = std::env::vars()
            .filter(|(key, _)| {
                key.starts_with("PSI_DEC_") || key.starts_with("MTL_") || matches!(key.as_str(), "RUST_LOG")
            })
            .map(|(key, value)| format!("{}={}", escape_field(&key), escape_field(&value)))
            .collect::<Vec<_>>();
        environment.sort();
        Self {
            commit,
            dirty,
            machine,
            os,
            architecture: std::env::consts::ARCH,
            environment: if environment.is_empty() {
                "none".into()
            } else {
                environment.join(",")
            },
        }
    }

    fn print(&self, model_dir: &std::path::Path) {
        println!(
            "provenance component=qwen35-vanilla-prefill-decode model_dir={} commit={} dirty={} machine={} os={} \
             architecture={} environment={}",
            escape_field(&model_dir.display().to_string()),
            escape_field(&self.commit),
            self.dirty,
            escape_field(&self.machine),
            escape_field(&self.os),
            self.architecture,
            self.environment,
        );
    }
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn escape_field(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'/' | b':' | b'=') {
            escaped.push(char::from(byte));
        } else {
            escaped.push_str(&format!("%{byte:02X}"));
        }
    }
    escaped
}

fn parse_cases(value: &str) -> Vec<Case> {
    let mut cases = value
        .split(',')
        .map(str::trim)
        .filter(|case| !case.is_empty())
        .map(|case| {
            match case {
                "prefill" => Case::Prefill,
                "decode" => Case::Decode,
                _ => panic!("unknown qwen35 prefill/decode case {case:?}"),
            }
        })
        .collect::<Vec<_>>();
    assert!(!cases.is_empty(), "--cases must not be empty");
    cases.sort_unstable();
    assert!(
        cases.windows(2).all(|pair| pair[0] != pair[1]),
        "--cases must not contain duplicates"
    );
    cases
}

fn parse_usize_list(value: &str, flag: &str, allow_zero: bool) -> Vec<usize> {
    let mut values = value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| parse_usize(value, flag))
        .collect::<Vec<_>>();
    assert!(!values.is_empty(), "{flag} must not be empty");
    if !allow_zero {
        assert!(values.iter().all(|&value| value > 0), "{flag} values must be positive");
    }
    values.sort_unstable();
    assert!(
        values.windows(2).all(|pair| pair[0] != pair[1]),
        "{flag} must not contain duplicates"
    );
    values
}

fn next_arg(values: &mut impl Iterator<Item = String>, flag: &str) -> String {
    values.next().unwrap_or_else(|| panic!("{flag} requires a value"))
}

fn parse_usize(value: &str, flag: &str) -> usize {
    value.parse().unwrap_or_else(|_| panic!("{flag} requires a usize"))
}

fn parse_u32(value: &str, flag: &str) -> u32 {
    value.parse().unwrap_or_else(|_| panic!("{flag} requires a u32"))
}

fn parse_f32(value: &str, flag: &str) -> f32 {
    value.parse().unwrap_or_else(|_| panic!("{flag} requires an f32"))
}

fn print_help_and_exit() -> ! {
    println!(
        "qwen35_vanilla_prefill_decode benchmark\n--model-dir PATH\n--cases prefill,decode\n--contexts \
         N[,N...]\n--prefill-tokens N[,N...]\n--decode-tokens N[,N...]\n--num-reqs N\n--max-tokens \
         N\n--max-tokens-per-request N\n--num-tokens-per-block N\n--num-cache-pages N\n--seed N\n--temperature \
         F\n--top-k N\n--top-p F\n--warmup-iters N\n--iters N\n--runs N\n--reuse-context"
    );
    std::process::exit(0);
}

// Reuse materialized prefix pages. All benchmark contexts must end at a state
// publication boundary. Resetting the request slot then restores the last prefix
// state through the normal prepare_batch path, without recomputing the prefix.
fn run_reused_prefill(args: &Args, shapes: &[Shape], fixture: &mut VanillaFixture) {
    use std::io::Write;

    let skipped = std::env::var("PSI_PREFILL_SKIP").unwrap_or_default();
    let skipped = skipped.split(',').collect::<BTreeSet<_>>();
    let req_slots = (0..args.num_reqs).map(|index| index as u32).collect::<Vec<_>>();
    for &context in &args.contexts {
        let pending = shapes
            .iter()
            .copied()
            .filter(|shape| {
                shape.context == context
                    && !skipped.contains(format!("{}:{}", context, shape.operation_tokens).as_str())
            })
            .collect::<Vec<_>>();
        if pending.is_empty() {
            continue;
        }
        fixture.model.reset_req_slots(&req_slots);
        fixture.pages.reset_sync();
        let epoch = fixture.next_epoch;
        fixture.next_epoch += 1;
        let prefix_start = Instant::now();
        if context > 0 {
            fixture.execute_prefill_range(0, context, fixture.active_chunk_tokens, epoch);
        }
        fixture.model.reset_req_slots(&req_slots);
        println!(
            "matrix_prefix num_reqs={} context={} elapsed_ms={:.6}",
            args.num_reqs,
            context,
            prefix_start.elapsed().as_secs_f64() * 1000.0
        );
        std::io::stdout().flush().unwrap();

        for shape in pending {
            let mut samples = Vec::new();
            let mut submit_samples = Vec::new();
            let mut prepare_samples = Vec::new();
            let mut cold_ms = 0.0;
            let mut expected_outputs: Option<Vec<Vec<OutputSample>>> = None;
            for iteration in 0..(args.warmup_iters + args.iters * args.runs) {
                fixture.model.reset_req_slots(&req_slots);
                fixture.pages.reset_sync();
                let epoch = fixture.next_epoch;
                fixture.next_epoch += 1;
                let start = Instant::now();
                let (timing, consumed) =
                    fixture.execute_prefill_range(context, shape.operation_tokens, fixture.active_chunk_tokens, epoch);
                assert_eq!(consumed, shape.operation_tokens);
                // commit_batch submits the recurrent-state commit asynchronously.
                // A warmup probe, or reset below, waits for that submission.
                if iteration < args.warmup_iters {
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    if iteration == 0 {
                        cold_ms = elapsed_ms;
                    }
                    let outputs = fixture.execute_decode_probe(shape.final_context(), epoch);
                    if let Some(expected) = &expected_outputs {
                        assert_decomposition_output_tokens(
                            expected,
                            &outputs,
                            "restored prefix must reproduce the warmup probe",
                        );
                    } else {
                        expected_outputs = Some(outputs);
                    }
                    fixture.model.reset_req_slots(&req_slots);
                } else {
                    fixture.model.reset_req_slots(&req_slots);
                    samples.push(start.elapsed().as_secs_f64() * 1000.0);
                    submit_samples.push(timing.main_submit_wait.as_secs_f64() * 1000.0);
                    prepare_samples.push(timing.prepare.as_secs_f64() * 1000.0);
                }
            }
            // Check page restoration against a fresh production-path prefix at a
            // representative publication boundary. This check is outside timing.
            if context == DEFAULT_NUM_TOKENS_PER_BLOCK && shape.operation_tokens == 8 {
                let fresh =
                    fixture.run_trajectory(shape, fixture.active_chunk_tokens, fixture.active_chunk_tokens, false);
                assert_decomposition_output_tokens(
                    expected_outputs.as_ref().unwrap(),
                    &fresh.outputs,
                    "restored state must match a recomputed prefix",
                );
                println!(
                    "matrix_check num_reqs={} context={} tokens={} status=pass",
                    args.num_reqs, context, shape.operation_tokens
                );
            }
            let format_values = |values: &[f64]| {
                values
                    .iter()
                    .map(|value| format!("{value:.6}"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            println!(
                "matrix_result num_reqs={} context={} tokens={} chunk_tokens={} cold_ms={:.6} samples_ms={} \
                 submit_ms={} prepare_ms={} status=measured",
                args.num_reqs,
                context,
                shape.operation_tokens,
                fixture.active_chunk_tokens,
                cold_ms,
                format_values(&samples),
                format_values(&submit_samples),
                format_values(&prepare_samples)
            );
            std::io::stdout().flush().unwrap();
        }
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use std::panic::AssertUnwindSafe;

    use super::*;

    #[test]
    fn test_cli_lists_are_sorted_and_validated() {
        let args = Args::parse_from(
            [
                "bench",
                "--model-dir",
                "/model",
                "--cases",
                "decode, prefill",
                "--contexts",
                "8, 0,4",
                "--prefill-tokens",
                "128,64",
                "--decode-tokens",
                "32,1",
            ]
            .into_iter()
            .map(String::from),
        );
        assert_eq!(args.cases, vec![Case::Prefill, Case::Decode]);
        assert_eq!(args.contexts, vec![0, 4, 8]);
        assert_eq!(args.prefill_tokens, vec![64, 128]);
        assert_eq!(args.decode_tokens, vec![1, 32]);

        let duplicate = std::panic::catch_unwind(|| parse_usize_list("1,1", "--contexts", true));
        assert!(duplicate.is_err());
        let zero = std::panic::catch_unwind(|| parse_usize_list("0", "--prefill-tokens", false));
        assert!(zero.is_err());
    }

    #[test]
    fn test_shape_cross_product_order() {
        let args = Args {
            model_dir: "/model".into(),
            cases: vec![Case::Prefill, Case::Decode],
            contexts: vec![0, 4],
            prefill_tokens: vec![2, 3],
            decode_tokens: vec![1],
            num_reqs: 1,
            max_tokens: 4,
            max_tokens_per_request: 4,
            num_tokens_per_block: 4,
            num_cache_pages: 64,
            seed: 42,
            temperature: 0.7,
            top_k: 1,
            top_p: 0.8,
            warmup_iters: 1,
            iters: 1,
            runs: 1,
            reuse_context: false,
        };
        assert_eq!(
            args.shapes(),
            vec![
                Shape {
                    case: Case::Prefill,
                    context: 0,
                    operation_tokens: 2,
                },
                Shape {
                    case: Case::Prefill,
                    context: 0,
                    operation_tokens: 3,
                },
                Shape {
                    case: Case::Prefill,
                    context: 4,
                    operation_tokens: 2,
                },
                Shape {
                    case: Case::Prefill,
                    context: 4,
                    operation_tokens: 3,
                },
                Shape {
                    case: Case::Decode,
                    context: 0,
                    operation_tokens: 1,
                },
                Shape {
                    case: Case::Decode,
                    context: 4,
                    operation_tokens: 1,
                },
            ]
        );
    }

    #[test]
    fn test_prefill_chunk_planning() {
        assert_eq!(plan_prefill_chunks(1, 4), vec![1]);
        assert_eq!(plan_prefill_chunks(8, 4), vec![4, 4]);
        assert_eq!(plan_prefill_chunks(10, 4), vec![4, 4, 2]);

        let args = Args::parse_from(
            [
                "bench",
                "--model-dir",
                "/model",
                "--num-reqs",
                "3",
                "--max-tokens",
                "10",
                "--max-tokens-per-request",
                "8",
            ]
            .into_iter()
            .map(String::from),
        );
        assert_eq!(args.active_chunk_tokens(), 3);
    }

    #[test]
    fn test_decode_token_accounting() {
        let shape = Shape {
            case: Case::Decode,
            context: 7,
            operation_tokens: 3,
        };
        let mut accounting = OperationAccounting::new(shape);
        accounting.commit(1);
        accounting.commit(1);
        accounting.commit(1);
        assert_eq!(accounting.finish(), 10);

        let mut overcommit = OperationAccounting::new(shape);
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| overcommit.commit(4)));
        assert!(result.is_err());
    }

    #[test]
    fn test_page_id_allocation_and_capacity_failure() {
        let fixture = PageFixture::new(2, 9, 4, 42, &[2], &[5]);
        let page_ids = fixture
            .requests
            .iter()
            .flat_map(|request| request.kv_by_lane.iter().chain(&request.state_by_lane))
            .flat_map(|lane| lane.iter())
            .flat_map(|block| block.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(page_ids.len(), 42);
        assert_eq!(page_ids.iter().copied().collect::<BTreeSet<_>>().len(), 42);
        assert_eq!(page_ids.iter().copied().max(), Some(41));

        let capacity_failure = std::panic::catch_unwind(|| PageFixture::new(2, 9, 4, 41, &[2], &[5]));
        assert!(capacity_failure.is_err());
    }

    #[test]
    fn test_decoder_sync_blocks_incremental_ranges() {
        let mut fixture = PageFixture::new(1, 13, 4, 32, &[2], &[3]);
        let first = fixture.sync_blocks(0, 1);
        assert_eq!(first.block_index(), 0);
        assert_eq!(first.kv_page_ids()[0].len(), 1);
        assert_eq!(first.state_page_ids()[0].len(), 1);

        let same_block = fixture.sync_blocks(0, 4);
        assert_eq!(same_block.block_index(), 1);
        assert!(same_block.kv_page_ids()[0].is_empty());
        assert!(same_block.state_page_ids()[0].is_empty());

        let crossing = fixture.sync_blocks(0, 9);
        assert_eq!(crossing.block_index(), 1);
        assert_eq!(crossing.kv_page_ids()[0].len(), 2);
        assert_eq!(crossing.state_page_ids()[0].len(), 2);

        fixture.reset_sync();
        let rebuilt = fixture.sync_blocks(0, 9);
        assert_eq!(rebuilt.block_index(), 0);
        assert_eq!(rebuilt.kv_page_ids()[0].len(), 3);
        assert_eq!(rebuilt.state_page_ids()[0].len(), 3);
    }

    #[test]
    fn test_greedy_validation_sampling_preserves_non_greedy_fields() {
        let requested = SamplingConfig {
            max_sampled_tokens: 3,
            temperature: 0.7,
            top_k: 17,
            top_p: 0.8,
            seed: Some(9),
            stop_sequences: vec![vec![Token::new(5)]],
        };
        let validation = greedy_validation_sampling_config(&requested);
        assert_eq!(validation.temperature, 0.0);
        assert_eq!(validation.top_p, 0.0);
        assert_eq!(validation.max_sampled_tokens, requested.max_sampled_tokens);
        assert_eq!(validation.top_k, requested.top_k);
        assert_eq!(validation.seed, requested.seed);
        assert_eq!(validation.stop_sequences, requested.stop_sequences);
    }

    #[test]
    fn test_deterministic_tokens_and_fingerprints() {
        let first = (0..8)
            .map(|position| deterministic_token(0, position, 1024))
            .collect::<Vec<_>>();
        let again = (0..8)
            .map(|position| deterministic_token(0, position, 1024))
            .collect::<Vec<_>>();
        let second_request = (0..8)
            .map(|position| deterministic_token(1, position, 1024))
            .collect::<Vec<_>>();
        assert_eq!(first, again);
        assert_ne!(first, second_request);
        assert!(first.iter().chain(&second_request).all(|&token| token < 1024));
        assert_eq!(
            fingerprint_token_streams(std::slice::from_ref(&first)),
            fingerprint_token_streams(std::slice::from_ref(&again))
        );
        assert_ne!(
            fingerprint_token_streams(std::slice::from_ref(&first)),
            fingerprint_token_streams(std::slice::from_ref(&second_request))
        );
    }

    #[test]
    fn test_result_identity_and_final_context() {
        let prefill = Shape {
            case: Case::Prefill,
            context: 8,
            operation_tokens: 4,
        };
        assert_eq!(prefill.result_identity_fields(), (Case::Prefill, 8, 4, 0, 12));
        let decode = Shape {
            case: Case::Decode,
            context: 8,
            operation_tokens: 3,
        };
        assert_eq!(decode.result_identity_fields(), (Case::Decode, 8, 0, 3, 11));

        let expected = vec![vec![OutputSample {
            token: 7,
            probability_bits: 0.20_f32.to_bits(),
        }]];
        let same_token = vec![vec![OutputSample {
            token: 7,
            probability_bits: 0.90_f32.to_bits(),
        }]];
        assert_decomposition_output_tokens(&expected, &same_token, "test parity");

        let different_token = vec![vec![OutputSample {
            token: 8,
            probability_bits: 0.20_f32.to_bits(),
        }]];
        let mismatch =
            std::panic::catch_unwind(|| assert_decomposition_output_tokens(&expected, &different_token, "test parity"));
        assert!(mismatch.is_err());
    }
}
