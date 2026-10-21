use std::mem::size_of;
use std::path::PathBuf;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::gqa::kv_page_write as backend_kv_page_write;
use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::components::gqa::sdpa::sliced_kv as backend_sliced_kv;
use inference_backend_metal::components::gqa::sdpa::sliced_q as backend_sliced_q;
use inference_backend_metal::components::rms_norm_rope::RopeScaling;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::checkpoint::SafeTensorStore;
use inference_executor_core::model::qwen::v3::Qwen3ModelConfig;
use inference_executor_core::model::qwen::v3::init_qwen3_model_config;
use inference_executor_core::model::qwen::v3::weight_layout::resolve_qwen3_model_weight_bindings;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;
use inference_executor_metal::attn::gqa::backend::GQAKVCacheBindings;
use inference_executor_metal::attn::gqa::backend::GQAMetalConfig;
use inference_executor_metal::attn::gqa::batch_metadata::GQAMetadataBuffers;
use inference_executor_metal::attn::gqa::batch_metadata::GQAReplayBucketPolicy;
use inference_executor_metal::attn::gqa::request_page_table::GQAPageTableLayout;
use inference_executor_metal::attn::gqa::sdpa::RequestShape;
use inference_executor_metal::attn::gqa::sdpa::Selector;
use inference_executor_metal::attn::gqa::ungated_backend::UngatedGQA;
use inference_executor_metal::attn::gqa::ungated_backend::UngatedGQAInput;
use inference_executor_metal::attn::gqa::ungated_backend::UngatedGQAWeights;
use inference_executor_metal::attn::gqa::ungated_scratch::UngatedGQAScratch;
use inference_executor_metal::def::layer::ReplayLayer;
use inference_executor_metal::model::qwen::v3::QWEN3_PAGE_SIZE_BYTES;

const MAX_TOKENS: usize = 64;

#[derive(Clone, Copy)]
struct BenchParams {
    sdpa_sliced_kv_kv_tokens_per_iteration: u32,
    sdpa_sliced_kv_required_threads: u32,
    sdpa_sliced_kv_max_q_heads: u32,
    sdpa_sliced_q_max_q_tokens: u32,
    sdpa_sliced_q_kv_tokens_per_iteration: u32,
    sdpa_sliced_q_max_q_heads: u32,
}

struct Args {
    model_dir: PathBuf,
    tokens_per_req: Vec<u32>,
    contexts: Vec<u32>,
    params: BenchParams,
    warmup_iters: usize,
    iters: usize,
    runs: usize,
    validate: bool,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            tokens_per_req: vec![16],
            contexts: vec![0, 128, 1024, 4096],
            params: BenchParams {
                sdpa_sliced_kv_kv_tokens_per_iteration: 128,
                sdpa_sliced_kv_required_threads: 128,
                sdpa_sliced_kv_max_q_heads: 5,
                sdpa_sliced_q_max_q_tokens: 8,
                sdpa_sliced_q_kv_tokens_per_iteration: 16,
                sdpa_sliced_q_max_q_heads: 0,
            },
            warmup_iters: 10,
            iters: 50,
            runs: 5,
            validate: false,
        };
        let mut values = std::env::args().skip(1);
        while let Some(arg) = values.next() {
            match arg.as_str() {
                "--help" | "-h" => help(),
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut values, &arg)),
                "--tokens-per-req" => args.tokens_per_req = parse_u32_list(&next_arg(&mut values, &arg), &arg),
                "--contexts" => args.contexts = parse_u32_list(&next_arg(&mut values, &arg), &arg),
                "--sdpa-sliced-kv-kv-tokens-per-iteration" => {
                    args.params.sdpa_sliced_kv_kv_tokens_per_iteration = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--sdpa-sliced-kv-required-threads" => {
                    args.params.sdpa_sliced_kv_required_threads = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--sdpa-sliced-kv-max-q-heads" => {
                    args.params.sdpa_sliced_kv_max_q_heads = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--sdpa-sliced-q-max-q-tokens" => {
                    args.params.sdpa_sliced_q_max_q_tokens = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--sdpa-sliced-q-kv-tokens-per-iteration" => {
                    args.params.sdpa_sliced_q_kv_tokens_per_iteration = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--sdpa-sliced-q-max-q-heads" => {
                    args.params.sdpa_sliced_q_max_q_heads = parse_u32(&next_arg(&mut values, &arg), &arg)
                },
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--iters" => args.iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--validate" => args.validate = true,
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(!args.tokens_per_req.is_empty());
        assert!(args.tokens_per_req.iter().all(|&tokens| tokens > 0));
        assert!(args.tokens_per_req.iter().sum::<u32>() as usize <= MAX_TOKENS);
        assert!(!args.contexts.is_empty());
        assert!(args.iters > 0 && args.runs > 0);
        args
    }
}

struct Weights {
    qkv_weight: Buffer,
    qkv_scales: Buffer,
    qkv_biases: Buffer,
    q_norm_weight: Buffer,
    k_norm_weight: Buffer,
    output_weight: Buffer,
    output_scales: Buffer,
    output_biases: Buffer,
}

impl Weights {
    fn load(device: &Device, store: &mut SafeTensorStore, bindings: &Qwen3xGQAWeightBindings, head_dim: usize) -> Self {
        let (q_weight, q_scales, q_biases) = load_quantized(store, &bindings.q);
        let (k_weight, k_scales, k_biases) = load_quantized(store, &bindings.k);
        let (v_weight, v_scales, v_biases) = load_quantized(store, &bindings.v);
        let qkv_weight = concat(&[&q_weight, &k_weight, &v_weight]);
        let qkv_scales = concat(&[&q_scales, &k_scales, &v_scales]);
        let qkv_biases = concat(&[&q_biases, &k_biases, &v_biases]);
        let (output_weight, output_scales, output_biases) = load_quantized(store, &bindings.output);
        Self {
            qkv_weight: Buffer::from_slice(device, &qkv_weight),
            qkv_scales: Buffer::from_slice(device, &qkv_scales),
            qkv_biases: Buffer::from_slice(device, &qkv_biases),
            q_norm_weight: Buffer::from_slice(device, &load_norm_bf16(store, &bindings.q_norm_weight, head_dim)),
            k_norm_weight: Buffer::from_slice(device, &load_norm_bf16(store, &bindings.k_norm_weight, head_dim)),
            output_weight: Buffer::from_slice(device, &output_weight),
            output_scales: Buffer::from_slice(device, &output_scales),
            output_biases: Buffer::from_slice(device, &output_biases),
        }
    }

    fn bindings(&self) -> UngatedGQAWeights<'_> {
        UngatedGQAWeights {
            qkv_weight: &self.qkv_weight,
            qkv_scales: &self.qkv_scales,
            qkv_biases: &self.qkv_biases,
            q_norm_weight: &self.q_norm_weight,
            k_norm_weight: &self.k_norm_weight,
            output_weight: &self.output_weight,
            output_scales: &self.output_scales,
            output_biases: &self.output_biases,
        }
    }
}

struct Fixture {
    device: Device,
    stream: Stream,
    core: UngatedGQACore,
    config: GQAMetalConfig,
    params: BenchParams,
    num_tokens: u32,
    num_reqs: u32,
    existing_context: u32,
    sdpa_sliced_kv_max_q_heads: u32,
    production_sliced_q_max_q_heads: u32,
    sdpa_sliced_q_max_q_heads: u32,
    uses_sliced_q_variant: bool,
    page_table_layout: GQAPageTableLayout,
    sdpa_sliced_kv_metadata: GQAMetadataBuffers,
    sdpa_sliced_q_metadata: GQAMetadataBuffers,
    full_sdpa_sliced_kv: ReplayProgram,
    full_sdpa_sliced_q: ReplayProgram,
    sdpa_sliced_kv: ReplayProgram,
    sdpa_sliced_q: ReplayProgram,
    affine_replays: Vec<AffineReplay>,
    sdpa_sliced_kv_output: Buffer,
    sdpa_sliced_q_output: Buffer,
    _backend: UngatedGQA,
    _scratch: UngatedGQAScratch,
    _hidden: Buffer,
    _pages: Buffer,
    _page_ids: Buffer,
}

impl Fixture {
    fn new(
        device: &Device,
        model: &Qwen3ModelConfig,
        weights: &Weights,
        tokens_per_req: &[u32],
        existing_context: u32,
        params: BenchParams,
    ) -> Self {
        let text = &model.text_config;
        let core = UngatedGQACore::new(
            0,
            text.hidden_size,
            text.head_dim,
            text.num_attention_heads,
            text.num_key_value_heads,
            (text.head_dim as f32).sqrt().recip(),
        );
        let quantization = model
            .quantization
            .as_ref()
            .expect("Qwen3 GQA bench requires quantization");
        let config = GQAMetalConfig {
            group_size: quantization.group_size.try_into().expect("group size must fit u32"),
            bits: quantization.bits.try_into().expect("bits must fit u32"),
            page_bytes: QWEN3_PAGE_SIZE_BYTES.try_into().expect("page bytes must fit u32"),
            rope_dim: text.head_dim.try_into().expect("RoPE dim must fit u32"),
            norm_eps: text.rms_norm_eps,
            rope_theta: text.rope_theta,
            rope_scaling: RopeScaling::Default,
            io_dtype: Dtype::Bfloat16,
        };
        config.validate();
        let num_tokens = tokens_per_req.iter().sum();
        let num_reqs = tokens_per_req.len().try_into().expect("request count must fit u32");
        let cu_tokens = cumulative_tokens(tokens_per_req);
        let req_slots = (0..num_reqs).collect::<Vec<_>>();
        let token_indices = vec![existing_context; num_reqs as usize];
        let tokens_per_page = config.num_ungated_tokens_per_page(&core);
        let max_req_tokens = tokens_per_req.iter().copied().max().unwrap();
        let end_context = existing_context + max_req_tokens;
        let num_blocks = end_context.div_ceil(tokens_per_page).max(1);
        let page_table_layout = GQAPageTableLayout {
            num_req_slots: num_reqs,
            num_blocks,
            num_gqa_layers: 1,
            num_page_ids_per_block: 1,
        };
        let q_heads_per_kv_head = core.num_q_heads / core.num_kv_heads;
        let sdpa_sliced_kv_max_q_heads = q_heads_per_kv_head.min(params.sdpa_sliced_kv_max_q_heads as usize) as u32;
        let sliced_q_max_supported_q_heads = 256 / (params.sdpa_sliced_q_max_q_tokens / 8 * 32);
        let production_sliced_q_max_q_heads = q_heads_per_kv_head.min(sliced_q_max_supported_q_heads as usize) as u32;
        let sdpa_sliced_q_max_q_heads = if params.sdpa_sliced_q_max_q_heads == 0 {
            production_sliced_q_max_q_heads
        } else {
            params.sdpa_sliced_q_max_q_heads
        };
        assert!(sdpa_sliced_q_max_q_heads > 0 && sdpa_sliced_q_max_q_heads <= q_heads_per_kv_head as u32);
        assert!(sdpa_sliced_q_max_q_heads <= sliced_q_max_supported_q_heads);
        let sdpa_config = backend_sdpa::Config {
            io_dtype: config.io_dtype,
            num_q_heads: core.num_q_heads.try_into().expect("GQA Q-head count must fit u32"),
            num_kv_heads: core.num_kv_heads.try_into().expect("GQA KV-head count must fit u32"),
            head_dim: core.head_dim.try_into().expect("GQA head_dim must fit u32"),
            tokens_per_page,
        };
        let request_shapes = RequestShape::from_batch(&token_indices, &cu_tokens);
        let replay_bucket_policy = GQAReplayBucketPolicy::new(MAX_TOKENS as u32, MAX_TOKENS as u32);
        let sliced_kv_variant = backend_sdpa::ExecutionVariant::sliced_kv(
            sdpa_config,
            params.sdpa_sliced_kv_kv_tokens_per_iteration,
            params.sdpa_sliced_kv_required_threads,
            sdpa_sliced_kv_max_q_heads,
        );
        let sliced_kv_selection = Selector::new(
            backend_sdpa::Registry::from_variants(sdpa_config, vec![sliced_kv_variant]),
            MAX_TOKENS,
            MAX_TOKENS,
            MAX_TOKENS
                .checked_mul(sliced_kv_variant.map.thread_block.max_q_tokens as usize)
                .expect("GQA SDPA partial-state-group capacity must fit usize"),
        )
        .select(&request_shapes, &replay_bucket_policy, num_tokens);
        let sdpa_sliced_kv_metadata = GQAMetadataBuffers::new(device, MAX_TOKENS, MAX_TOKENS);
        sdpa_sliced_kv_metadata.update(&req_slots, &token_indices, &cu_tokens, &sliced_kv_selection);
        let sliced_q_variant = backend_sdpa::ExecutionVariant::sliced_q_per_head(
            sdpa_config,
            params.sdpa_sliced_q_max_q_tokens,
            params.sdpa_sliced_q_kv_tokens_per_iteration,
            sdpa_sliced_q_max_q_heads,
        );
        let sliced_q_selection = Selector::new(
            backend_sdpa::Registry::from_variants(sdpa_config, vec![sliced_q_variant]),
            MAX_TOKENS,
            MAX_TOKENS,
            MAX_TOKENS
                .checked_mul(sliced_q_variant.map.thread_block.max_q_tokens as usize)
                .expect("GQA SDPA partial-state-group capacity must fit usize"),
        )
        .select(&request_shapes, &replay_bucket_policy, num_tokens);
        let sdpa_sliced_q_metadata = GQAMetadataBuffers::new(device, MAX_TOKENS, MAX_TOKENS);
        let sliced_q_shape = sdpa_sliced_q_metadata.update(&req_slots, &token_indices, &cu_tokens, &sliced_q_selection);

        let stream = Stream::new(device);
        let backend = UngatedGQA::new(device, core.clone(), config, MAX_TOKENS, page_table_layout);
        let scratch = backend.new_scratch();
        let hidden = Buffer::from_slice(
            device,
            &(0..num_tokens as usize * core.hidden_dim)
                .map(|index| bf16::from_f32(((index % 31) as f32 - 15.0) / 64.0).to_bits())
                .collect::<Vec<_>>(),
        );
        let sdpa_sliced_kv_output =
            Buffer::new_zeroed_elements(device, num_tokens as usize * core.hidden_dim, Dtype::Bfloat16);
        let sdpa_sliced_q_output =
            Buffer::new_zeroed_elements(device, num_tokens as usize * core.hidden_dim, Dtype::Bfloat16);
        let num_pages = num_reqs as usize * num_blocks as usize;
        let pages = Buffer::new_zeroed(
            device,
            num_pages
                .checked_mul(config.page_bytes as usize)
                .expect("GQA page arena size must fit usize"),
        );
        let page_ids = Buffer::from_slice(
            device,
            &(0..num_pages)
                .map(|page_id| page_id.try_into().expect("page ID must fit u32"))
                .collect::<Vec<u32>>(),
        );
        let full_sdpa_sliced_kv = full_replay(
            &stream,
            &backend,
            page_table_layout,
            &sdpa_sliced_kv_metadata,
            &hidden,
            &sdpa_sliced_kv_output,
            &pages,
            &page_ids,
            weights,
            &scratch,
        );
        let full_sdpa_sliced_q = full_replay(
            &stream,
            &backend,
            page_table_layout,
            &sdpa_sliced_q_metadata,
            &hidden,
            &sdpa_sliced_q_output,
            &pages,
            &page_ids,
            weights,
            &scratch,
        );
        let sdpa_sliced_kv = sdpa_sliced_kv_replay(
            device,
            &stream,
            &core,
            config,
            page_table_layout,
            &sdpa_sliced_kv_metadata,
            &pages,
            &page_ids,
            &scratch,
        );
        let sdpa_sliced_q = sdpa_sliced_q_replay(
            device,
            &stream,
            &core,
            config,
            page_table_layout,
            &sdpa_sliced_q_metadata,
            &pages,
            &page_ids,
            &scratch,
        );
        let affine_replays = affine_replays(
            device,
            &stream,
            &core,
            config,
            num_tokens,
            &hidden,
            &sdpa_sliced_q_output,
            weights,
            &scratch,
        );
        Self {
            device: device.clone(),
            stream,
            core,
            config,
            params,
            num_tokens,
            num_reqs,
            existing_context,
            sdpa_sliced_kv_max_q_heads,
            production_sliced_q_max_q_heads,
            sdpa_sliced_q_max_q_heads,
            uses_sliced_q_variant: (num_tokens as u64) >= 2 * sliced_q_shape.num_q_token_tiles as u64,
            page_table_layout,
            sdpa_sliced_kv_metadata,
            sdpa_sliced_q_metadata,
            full_sdpa_sliced_kv,
            full_sdpa_sliced_q,
            sdpa_sliced_kv,
            sdpa_sliced_q,
            affine_replays,
            sdpa_sliced_kv_output,
            sdpa_sliced_q_output,
            _backend: backend,
            _scratch: scratch,
            _hidden: hidden,
            _pages: pages,
            _page_ids: page_ids,
        }
    }

    fn validate(&self) {
        if !self.uses_sliced_q_variant {
            println!("validation skipped=sdpa_sliced_q_variant_not_selected");
            return;
        }
        self.stream.submit_replay(&self.full_sdpa_sliced_kv).wait();
        self.stream.submit_replay(&self.full_sdpa_sliced_q).wait();
        let sliced_kv = self
            .sdpa_sliced_kv_output
            .read_typed::<u16>(0, self.num_tokens as usize * self.core.hidden_dim);
        let sliced_q = self
            .sdpa_sliced_q_output
            .read_typed::<u16>(0, self.num_tokens as usize * self.core.hidden_dim);
        let max_abs_diff = sliced_kv
            .iter()
            .zip(sliced_q)
            .map(|(&lhs, rhs)| (bf16::from_bits(lhs).to_f32() - bf16::from_bits(rhs).to_f32()).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff <= 0.0625,
            "Qwen3 full GQA SplitKV variant mismatch: max_abs_diff={max_abs_diff}"
        );
        println!("validation max_abs_diff={max_abs_diff}");
    }

    fn print_resources(&self) {
        let num_simdgroups = self.params.sdpa_sliced_kv_required_threads as usize / 32;
        let sliced_kv_smem = (num_simdgroups * (8 * 64 + 8 * 16 + 8 * 3) + 8 * 2) * size_of::<f32>();
        let sliced_q_config = self.sliced_q_config();
        let sliced_q_execution = self.sdpa_sliced_q_metadata.variant();
        let sliced_q_smem = sliced_q_config.map_threadblock_memory_bytes(sliced_q_execution);
        let sliced_kv_output_accumulators = 8 * 64 / 32 * self.core.head_dim.div_ceil(64);
        println!(
            "resources device={} head_dim={} q_heads={} kv_heads={} q_heads_per_kv_head={} \
             sdpa_sliced_kv_max_q_heads={} sdpa_sliced_kv_required_threads={} \
             sdpa_sliced_kv_kv_tokens_per_iteration={} sdpa_sliced_kv_threadgroup_bytes={} \
             sdpa_sliced_kv_output_accumulators_per_thread={} production_sliced_q_max_q_heads={} \
             sdpa_sliced_q_max_q_heads={} sdpa_sliced_q_required_threads={} sdpa_sliced_q_max_q_tokens={} \
             sdpa_sliced_q_kv_tokens_per_iteration={} sdpa_sliced_q_threadgroup_bytes={} \
             sdpa_sliced_q_output_accumulators_per_thread={}",
            self.device.name(),
            self.core.head_dim,
            self.core.num_q_heads,
            self.core.num_kv_heads,
            self.core.num_q_heads / self.core.num_kv_heads,
            self.sdpa_sliced_kv_max_q_heads,
            self.params.sdpa_sliced_kv_required_threads,
            self.params.sdpa_sliced_kv_kv_tokens_per_iteration,
            sliced_kv_smem,
            sliced_kv_output_accumulators,
            self.production_sliced_q_max_q_heads,
            self.sdpa_sliced_q_max_q_heads,
            sliced_q_execution.map.thread_block.required_threads,
            self.params.sdpa_sliced_q_max_q_tokens,
            self.params.sdpa_sliced_q_kv_tokens_per_iteration,
            sliced_q_smem,
            8 * self.core.head_dim / 32,
        );
    }

    fn measure(&self, warmup_iters: usize, iters: usize, runs: usize) {
        let fields = format!(
            "num_tokens={} num_reqs={} context={} sdpa_sliced_kv_max_q_heads={} production_sliced_q_max_q_heads={}",
            self.num_tokens,
            self.num_reqs,
            self.existing_context,
            self.sdpa_sliced_kv_max_q_heads,
            self.production_sliced_q_max_q_heads,
        );
        print_measurement(
            "gqa.full.sdpa_sliced_kv",
            &fields,
            measure(&self.stream, &self.full_sdpa_sliced_kv, warmup_iters, iters, runs),
        );
        print_measurement(
            "gqa.split_kv.sliced_kv",
            &fields,
            measure(&self.stream, &self.sdpa_sliced_kv, warmup_iters, iters, runs),
        );
        if self.uses_sliced_q_variant {
            print_measurement(
                "gqa.full.sdpa_sliced_q",
                &fields,
                measure(&self.stream, &self.full_sdpa_sliced_q, warmup_iters, iters, runs),
            );
            let split_kv_fields = format!("{fields} sdpa_sliced_q_max_q_heads={}", self.sdpa_sliced_q_max_q_heads);
            print_measurement(
                "gqa.split_kv.sliced_q",
                &split_kv_fields,
                measure(&self.stream, &self.sdpa_sliced_q, warmup_iters, iters, runs),
            );
        }
        for affine in &self.affine_replays {
            print_measurement(
                affine.name,
                &fields,
                measure(&self.stream, &affine.replay, warmup_iters, iters, runs),
            );
        }
    }

    fn sliced_q_config(&self) -> backend_sliced_q::Config {
        backend_sliced_q::Config {
            num_q_heads: self.core.num_q_heads.try_into().expect("q heads must fit u32"),
            num_kv_heads: self.core.num_kv_heads.try_into().expect("KV heads must fit u32"),
            head_dim: self.core.head_dim.try_into().expect("head dim must fit u32"),
            scale: self.core.scale,
            page_bytes: self.config.page_bytes,
            dtype: self.config.io_dtype,
            page_table_layout: backend_kv_page_write::PageTableLayout {
                num_req_slots: self.page_table_layout.num_req_slots,
                num_blocks: self.page_table_layout.num_blocks,
                num_gqa_layers: self.page_table_layout.num_gqa_layers,
                num_page_ids_per_block: self.page_table_layout.num_page_ids_per_block,
            },
        }
    }
}

struct AffineReplay {
    name: &'static str,
    replay: ReplayProgram,
    _affine: affine_quantized::Kernel,
}

#[allow(clippy::too_many_arguments)]
fn affine_replays(
    device: &Device,
    stream: &Stream,
    core: &UngatedGQACore,
    config: GQAMetalConfig,
    num_tokens: u32,
    hidden: &Buffer,
    output: &Buffer,
    weights: &Weights,
    scratch: &UngatedGQAScratch,
) -> Vec<AffineReplay> {
    let scratch = scratch.bindings();
    let qkv = core.qkv_shape();
    let output_shape = core.output_shape();
    let qkv_config = affine_config(qkv.out_dim, qkv.in_dim, config);
    let output_config = affine_config(output_shape.out_dim, output_shape.in_dim, config);
    let kernels = [
        (
            "gqa.qkv.qmv_bn8_bk32",
            "gqa.output.qmv_bn8_bk32",
            affine_quantized::KernelKind::QmvBn8Bk32,
        ),
        (
            "gqa.qkv.qmm_bm8_bn32",
            "gqa.output.qmm_bm8_bn32",
            affine_quantized::KernelKind::QmmBm8Bn32,
        ),
        (
            "gqa.qkv.qmm_bm16_bn32",
            "gqa.output.qmm_bm16_bn32",
            affine_quantized::KernelKind::QmmBm16Bn32,
        ),
    ];
    let mut replays = Vec::with_capacity(kernels.len() * 2);
    for &(qkv_name, _, kind) in &kernels {
        replays.push(affine_replay(
            device,
            stream,
            qkv_name,
            qkv_config,
            kind,
            num_tokens,
            scratch.qkv,
            hidden,
            &weights.qkv_weight,
            &weights.qkv_scales,
            &weights.qkv_biases,
        ));
    }
    for &(_, output_name, kind) in &kernels {
        replays.push(affine_replay(
            device,
            stream,
            output_name,
            output_config,
            kind,
            num_tokens,
            output,
            scratch.attention_output,
            &weights.output_weight,
            &weights.output_scales,
            &weights.output_biases,
        ));
    }
    replays
}

#[allow(clippy::too_many_arguments)]
fn affine_replay(
    device: &Device,
    stream: &Stream,
    name: &'static str,
    config: affine_quantized::Config,
    kind: affine_quantized::KernelKind,
    num_tokens: u32,
    output: &Buffer,
    input: &Buffer,
    weight: &Buffer,
    scales: &Buffer,
    biases: &Buffer,
) -> AffineReplay {
    let kernel = affine_quantized::Kernel::new(device, config, kind);
    let mut recorder = stream.create_replay_program();
    recorder.record(kernel.invoke(
        num_tokens,
        ReplayU32::Fixed(num_tokens),
        output,
        0,
        input,
        0,
        weight,
        0,
        scales,
        0,
        biases,
        0,
    ));
    AffineReplay {
        name,
        replay: recorder.build(),
        _affine: kernel,
    }
}

fn affine_config(n: usize, k: usize, config: GQAMetalConfig) -> affine_quantized::Config {
    affine_quantized::Config {
        n: n.try_into().expect("GQA affine n must fit i32"),
        k: k.try_into().expect("GQA affine k must fit i32"),
        group_size: config
            .group_size
            .try_into()
            .expect("GQA affine group_size must fit i32"),
        bits: config.bits.try_into().expect("GQA affine bits must fit i32"),
        input_dtype: config.io_dtype,
        output_dtype: config.io_dtype,
        scale_bias_dtype: config.io_dtype,
    }
}

#[allow(clippy::too_many_arguments)]
fn full_replay(
    stream: &Stream,
    backend: &UngatedGQA,
    page_table_layout: GQAPageTableLayout,
    metadata: &GQAMetadataBuffers,
    hidden: &Buffer,
    output: &Buffer,
    pages: &Buffer,
    page_ids: &Buffer,
    weights: &Weights,
    scratch: &UngatedGQAScratch,
) -> ReplayProgram {
    let mut recorder = ReplayRecorder::new(stream.create_replay_program());
    let _ = <UngatedGQA as ReplayLayer>::record(
        backend,
        &mut recorder,
        UngatedGQAInput {
            page_table_layout,
            gqa_layer_index: ReplayU32::Fixed(0),
            batch_metadata: metadata,
            hidden_state: hidden,
            next_hidden_state: output,
            kv_cache: GQAKVCacheBindings {
                kv_pages: pages,
                page_ids,
            },
            weights: weights.bindings(),
            scratch: scratch.bindings(),
            num_active_tokens: ReplayU32::Fixed(metadata.replay_shape().num_tokens),
        },
    );
    recorder.build()
}

#[allow(clippy::too_many_arguments)]
fn sdpa_sliced_kv_replay(
    device: &Device,
    stream: &Stream,
    core: &UngatedGQACore,
    config: GQAMetalConfig,
    page_table_layout: GQAPageTableLayout,
    metadata: &GQAMetadataBuffers,
    pages: &Buffer,
    page_ids: &Buffer,
    scratch: &UngatedGQAScratch,
) -> ReplayProgram {
    let shape = metadata.replay_shape();
    let sdpa_shape = backend_sliced_kv::Shape {
        num_total_tokens: shape.num_tokens,
        num_total_sdpa_map_task_templates: shape.num_total_sdpa_map_task_templates,
    };
    let sdpa_config = backend_sliced_kv::Config {
        num_q_heads: core.num_q_heads.try_into().expect("q heads must fit u32"),
        num_kv_heads: core.num_kv_heads.try_into().expect("KV heads must fit u32"),
        head_dim: core.head_dim.try_into().expect("head dim must fit u32"),
        scale: core.scale,
        page_bytes: config.page_bytes,
        page_table_layout: backend_kv_page_write::PageTableLayout {
            num_req_slots: page_table_layout.num_req_slots,
            num_blocks: page_table_layout.num_blocks,
            num_gqa_layers: page_table_layout.num_gqa_layers,
            num_page_ids_per_block: page_table_layout.num_page_ids_per_block,
        },
        dtype: config.io_dtype,
    };
    let kernels = backend_sliced_kv::Compute::new(device, sdpa_config, metadata.variant(), sdpa_shape);
    let scratch = scratch.bindings();
    let mut recorder = ReplayRecorder::new(stream.create_replay_program());
    recorder.record(ReplayOp::opaque(kernels.invoke_map(
        backend_sliced_kv::MapBuffers {
            q: scratch.q_norm_rope,
            kv_pages: pages,
            req_slots: metadata.req_slots(),
            page_ids,
            sdpa_map_task_templates: metadata.sdpa_map_task_templates(),
            partial_exp_sums: scratch.sdpa_partial_exp_sums,
            partial_max_logits: scratch.sdpa_partial_max_logits,
            partial_output: scratch.sdpa_partial_output,
        },
        ReplayU32::Fixed(0),
        ReplayU32::Fixed(sdpa_shape.num_total_tokens),
        ReplayU32::Fixed(sdpa_shape.num_total_sdpa_map_task_templates),
    )));
    recorder.record_with_barrier_before(ReplayOp::opaque(kernels.invoke_reduce(
        backend_sliced_kv::ReduceBuffers {
            partial_exp_sums: scratch.sdpa_partial_exp_sums,
            partial_max_logits: scratch.sdpa_partial_max_logits,
            partial_output: scratch.sdpa_partial_output,
            cu_sdpa_partial_outputs: metadata.cu_sdpa_partial_outputs(),
            output: scratch.attention_output,
        },
        ReplayU32::Fixed(sdpa_shape.num_total_tokens),
    )));
    recorder.build()
}

#[allow(clippy::too_many_arguments)]
fn sdpa_sliced_q_replay(
    device: &Device,
    stream: &Stream,
    core: &UngatedGQACore,
    config: GQAMetalConfig,
    page_table_layout: GQAPageTableLayout,
    metadata: &GQAMetadataBuffers,
    pages: &Buffer,
    page_ids: &Buffer,
    scratch: &UngatedGQAScratch,
) -> ReplayProgram {
    let shape = metadata.replay_shape();
    let sliced_q_config = backend_sliced_q::Config {
        num_q_heads: core.num_q_heads.try_into().expect("q heads must fit u32"),
        num_kv_heads: core.num_kv_heads.try_into().expect("KV heads must fit u32"),
        head_dim: core.head_dim.try_into().expect("head dim must fit u32"),
        scale: core.scale,
        page_bytes: config.page_bytes,
        dtype: config.io_dtype,
        page_table_layout: backend_kv_page_write::PageTableLayout {
            num_req_slots: page_table_layout.num_req_slots,
            num_blocks: page_table_layout.num_blocks,
            num_gqa_layers: page_table_layout.num_gqa_layers,
            num_page_ids_per_block: page_table_layout.num_page_ids_per_block,
        },
    };
    let sliced_q_shape = backend_sliced_q::Shape {
        num_total_tokens: shape.num_tokens,
        num_total_q_token_tiles: shape.num_q_token_tiles,
        num_total_sdpa_map_task_templates: shape.num_total_sdpa_map_task_templates,
    };
    let kernels = backend_sliced_q::Compute::new(device, sliced_q_config, metadata.variant(), sliced_q_shape);
    let scratch = scratch.bindings();
    let mut recorder = ReplayRecorder::new(stream.create_replay_program());
    recorder.record(ReplayOp::opaque(kernels.invoke_map(
        backend_sliced_q::MapBuffers {
            q: scratch.q_norm_rope,
            kv_pages: pages,
            req_slots: metadata.req_slots(),
            page_ids,
            visible_kv_token_ranges: metadata.visible_kv_token_ranges(),
            q_token_ranges: metadata.q_token_ranges(),
            sdpa_map_task_templates: metadata.sdpa_map_task_templates(),
            partial_output: scratch.sdpa_partial_output,
            partial_exp_sums: scratch.sdpa_partial_exp_sums,
            partial_max_logits: scratch.sdpa_partial_max_logits,
        },
        ReplayU32::Fixed(0),
        ReplayU32::Fixed(sliced_q_shape.num_total_tokens),
        ReplayU32::Fixed(sliced_q_shape.num_total_q_token_tiles),
        ReplayU32::Fixed(sliced_q_shape.num_total_sdpa_map_task_templates),
    )));
    recorder.record_with_barrier_before(ReplayOp::opaque(kernels.invoke_reduce(
        backend_sliced_q::ReduceBuffers {
            partial_output: scratch.sdpa_partial_output,
            partial_exp_sums: scratch.sdpa_partial_exp_sums,
            partial_max_logits: scratch.sdpa_partial_max_logits,
            q_token_ranges: metadata.q_token_ranges(),
            cu_sdpa_partial_outputs: metadata.cu_sdpa_partial_outputs(),
            output: scratch.attention_output,
        },
        ReplayU32::Fixed(sliced_q_shape.num_total_q_token_tiles),
    )));
    recorder.build()
}

fn main() {
    let args = Args::parse();
    let model = init_qwen3_model_config(&args.model_dir).expect("unable to load Qwen3 model config");
    let device = Device::system_default();
    let mut store = SafeTensorStore::from_model_dir(&args.model_dir).expect("unable to open Qwen3 checkpoint");
    let bindings = resolve_qwen3_model_weight_bindings(&model, store.index().tensor_names())
        .expect("unable to resolve Qwen3 checkpoint layout");
    let layer = bindings
        .main
        .layers
        .first()
        .expect("Qwen3 checkpoint must contain a layer");
    let weights = Weights::load(&device, &mut store, &layer.gqa, model.text_config.head_dim);
    drop(store);

    for &context in &args.contexts {
        let fixture = Fixture::new(&device, &model, &weights, &args.tokens_per_req, context, args.params);
        fixture.print_resources();
        if args.validate {
            fixture.validate();
        }
        fixture.measure(args.warmup_iters, args.iters, args.runs);
    }
}

fn load_quantized(store: &mut SafeTensorStore, bindings: &QuantizedTensorBindings) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        store
            .tensor_bytes(&bindings.weight, safetensors::Dtype::U32)
            .expect("unable to load quantized weight")
            .into_data(),
        store
            .tensor_bytes(&bindings.scales, safetensors::Dtype::BF16)
            .expect("unable to load quantized scales")
            .into_data(),
        store
            .tensor_bytes(&bindings.biases, safetensors::Dtype::BF16)
            .expect("unable to load quantized biases")
            .into_data(),
    )
}

fn load_norm_bf16(store: &mut SafeTensorStore, name: &str, head_dim: usize) -> Vec<u8> {
    let tensor = store
        .tensor_bytes(name, safetensors::Dtype::BF16)
        .expect("unable to load norm weight");
    assert_eq!(tensor.shape(), &[head_dim]);
    tensor.into_data()
}

fn concat(parts: &[&[u8]]) -> Vec<u8> {
    let mut output = Vec::with_capacity(parts.iter().map(|part| part.len()).sum());
    for part in parts {
        output.extend_from_slice(part);
    }
    output
}

fn cumulative_tokens(tokens_per_req: &[u32]) -> Vec<u32> {
    let mut cumulative = Vec::with_capacity(tokens_per_req.len() + 1);
    cumulative.push(0);
    for &tokens in tokens_per_req {
        cumulative.push(cumulative.last().copied().unwrap() + tokens);
    }
    cumulative
}

fn measure(stream: &Stream, replay: &ReplayProgram, warmup_iters: usize, iters: usize, runs: usize) -> Vec<f64> {
    for _ in 0..warmup_iters {
        stream.submit_replay(replay).wait();
    }
    (0..runs)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..iters {
                stream.submit_replay(replay).wait();
            }
            start.elapsed().as_secs_f64() * 1000.0 / iters as f64
        })
        .collect()
}

fn print_measurement(name: &str, fields: &str, samples: Vec<f64>) {
    let mut sorted = samples;
    sorted.sort_by(f64::total_cmp);
    let median_ms = sorted[sorted.len() / 2];
    println!("perf component={name} {fields} median_ms={median_ms:.6} runs={sorted:?}");
}

fn next_arg(values: &mut impl Iterator<Item = String>, name: &str) -> String {
    values.next().unwrap_or_else(|| panic!("{name} requires a value"))
}

fn parse_u32_list(value: &str, name: &str) -> Vec<u32> {
    value.split(',').map(|part| parse_u32(part.trim(), name)).collect()
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

fn help() -> ! {
    println!(
        r#"qwen3_gqa bench
--model-dir PATH
--tokens-per-req 16
--contexts 0,128,1024,4096
--sdpa-sliced-kv-kv-tokens-per-iteration N
--sdpa-sliced-kv-required-threads N
--sdpa-sliced-kv-max-q-heads N
--sdpa-sliced-q-max-q-tokens 8|16
--sdpa-sliced-q-kv-tokens-per-iteration 8|16
--sdpa-sliced-q-max-q-heads N
--validate
--warmup-iters N
--iters N
--runs N"#
    );
    std::process::exit(0);
}
