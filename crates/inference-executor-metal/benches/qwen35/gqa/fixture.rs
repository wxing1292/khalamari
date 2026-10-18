use std::collections::BTreeMap;
use std::mem::size_of;

use inference_backend_metal::components::gqa::activation_gate as backend_activation_gate;
use inference_backend_metal::components::gqa::kv_page_write as backend_kv_page_write;
use inference_backend_metal::components::gqa::qgkv_split as backend_qgkv_split;
use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::components::gqa::sdpa::sliced_kv as backend_sliced_kv;
use inference_backend_metal::components::gqa::sdpa::sliced_q as backend_sliced_q;
use inference_backend_metal::components::rms_norm_rope;
use inference_backend_metal::components::rms_norm_rope::RopeScaling;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::attn::GQACore;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_metal::attn::gqa::backend::GQA;
use inference_executor_metal::attn::gqa::backend::GQAInput;
use inference_executor_metal::attn::gqa::backend::GQAKVCacheBindings;
use inference_executor_metal::attn::gqa::backend::GQAMetalConfig;
use inference_executor_metal::attn::gqa::backend::GQAWeights;
use inference_executor_metal::attn::gqa::batch_metadata::GQAMetadataBuffers;
use inference_executor_metal::attn::gqa::batch_metadata::GQAReplayBucketPolicy;
use inference_executor_metal::attn::gqa::scratch::GQAScratchBindings;
use inference_executor_metal::attn::gqa::sdpa::RequestShape;
use inference_executor_metal::attn::gqa::sdpa::Selector;
use inference_executor_metal::def::layer::ReplayLayer;
use inference_executor_metal::def::replay_op::MetalReplayRuntime;
use inference_executor_metal::def::replay_op::ReplayOp;
use safetensors::SafeTensors;

use crate::Args;
use crate::BITS;
use crate::GQA_NORM_EPS;
use crate::GQA_ROPE_DIM;
use crate::GQA_ROPE_THETA;
use crate::GQABenchParams;
use crate::GQAModelProfile;
use crate::GQASplitKVBenchVariant;
use crate::GROUP_SIZE;
use crate::assert_bf16_close;
use crate::build_single_invocation_replay;
use crate::cu_tokens;
use crate::gqa_attention_reference_at;
use crate::gqa_gate_config;
use crate::gqa_kv_page_write_config;
use crate::gqa_output_affine_config;
use crate::gqa_page_table_layout;
use crate::gqa_qgkv_affine_config;
use crate::gqa_qgkv_to_q_g_k_v_config;
use crate::gqa_sdpa_config;
use crate::gqa_sdpa_shape;
use crate::gqa_sdpa_sliced_q_config;
use crate::hidden_fixture;
use crate::max_bf16_diff;
use crate::measure_runs;
use crate::norm_rope_config;
use crate::norm_rope_shape;
use crate::page_table;
use crate::print_gqa_kernel_limits;
use crate::print_named_perf;
use crate::print_perf;
use crate::print_skip;
use crate::request_token_counts;
use crate::valid_num_reqs;

mod weights;
use weights::MappedFile;
use weights::RealGQAWeights;

pub fn run(args: Args) {
    let device = Device::system_default();
    if args.print_limits {
        print_gqa_kernel_limits(&device, args.params);
    }
    let mapped = MappedFile::open(&args.model_dir.join(args.model.shard));
    let tensors = SafeTensors::deserialize(mapped.as_bytes()).unwrap_or_else(|err| {
        panic!(
            "unable to deserialize safetensors shard {}: {err:?}",
            args.model_dir.join(args.model.shard).display()
        )
    });
    let weights = RealGQAWeights::load(&device, &tensors, args.model);
    let contexts = if args.contexts.is_empty() {
        vec![0]
    } else {
        args.contexts.clone()
    };
    if let Some(counts) = &args.tokens_per_req {
        if let Some(lengths) = &args.contexts_per_req {
            run_case(&device, &args, &weights, counts.clone(), lengths.clone());
        } else {
            for &context in &contexts {
                run_case(&device, &args, &weights, counts.clone(), vec![context; counts.len()]);
            }
        }
    } else {
        for &tokens in &args.tokens {
            for &requests in &args.num_reqs {
                if !valid_num_reqs(tokens, requests) {
                    print_skip(tokens, requests, None, None, "num_reqs_exceeds_tokens");
                    continue;
                }
                for &context in &contexts {
                    run_case(
                        &device,
                        &args,
                        &weights,
                        request_token_counts(tokens, requests),
                        vec![context; requests as usize],
                    );
                }
            }
        }
    }
}

fn measure_lookup(device: &Device, args: &Args, weights: &RealGQAWeights, counts: &[u32], contexts: &[u32]) {
    let model = args.model;
    let num_tokens = counts.iter().sum::<u32>();
    let num_reqs = counts.len() as u32;
    let end = contexts
        .iter()
        .zip(counts)
        .map(|(&context, &tokens)| context.checked_add(tokens).expect("GQA context endpoint must fit u32"))
        .max()
        .unwrap();
    let config = GQAMetalConfig {
        group_size: GROUP_SIZE,
        bits: BITS,
        page_bytes: model.page_bytes(),
        rope_dim: GQA_ROPE_DIM,
        norm_eps: GQA_NORM_EPS,
        rope_theta: GQA_ROPE_THETA,
        rope_scaling: RopeScaling::Default,
        io_dtype: Dtype::Bfloat16,
    };
    let core = GQACore::new(
        model.model_layer_index,
        model.hidden_dim,
        model.head_dim,
        model.num_q_heads,
        model.num_kv_heads,
        (model.head_dim as f32).sqrt().recip(),
    );
    let blocks = end.div_ceil(model.num_tokens_per_page());
    let layout = GQAPageTableLayout {
        num_req_slots: num_reqs,
        num_blocks: blocks,
        num_gqa_layers: 1,
        num_page_ids_per_block: 1,
    };
    let backend = GQA::new(device, core, config, args.max_tokens, layout);
    let metadata = backend.new_metadata();
    let slots = (0..num_reqs).collect::<Vec<_>>();
    let cu = cu_tokens(counts).into_iter().map(|n| n as u32).collect::<Vec<_>>();
    backend.prepare(
        &metadata,
        &slots,
        contexts,
        &cu,
        &backend.replay_bucket_policy(),
        num_tokens,
    );
    let scratch = backend.new_scratch();
    let hidden = Buffer::from_slice(device, &hidden_fixture(num_tokens as usize, model.hidden_dim));
    let output = Buffer::new_zeroed(device, hidden.len_bytes());
    let kv = Buffer::new_zeroed(device, num_reqs as usize * blocks as usize * config.page_bytes as usize);
    let pages = Buffer::from_slice(device, &page_table(num_reqs, blocks));
    let stream = Stream::new(device);
    let mut recorder = MetalReplayRuntime::new(&stream).create_recorder();
    <GQA as ReplayLayer>::record(
        &backend,
        &mut recorder,
        GQAInput {
            page_table_layout: layout,
            gqa_layer_index: ReplayU32::Fixed(0),
            batch_metadata: &metadata,
            hidden_state: &hidden,
            next_hidden_state: &output,
            kv_cache: GQAKVCacheBindings {
                kv_pages: &kv,
                page_ids: &pages,
            },
            weights: GQAWeights {
                qgkv_weight: &weights.qgkv_weight,
                qgkv_scales: &weights.qgkv_scales,
                qgkv_biases: &weights.qgkv_biases,
                q_norm_weight: &weights.q_norm_weight,
                k_norm_weight: &weights.k_norm_weight,
                output_weight: &weights.output_weight,
                output_scales: &weights.output_scales,
                output_biases: &weights.output_biases,
            },
            scratch: scratch.bindings(),
            num_active_tokens: ReplayU32::Fixed(num_tokens),
        },
    );
    let replay = recorder.build();
    print_split_selection(
        "auto",
        &metadata,
        metadata.variant().map.thread_block.max_q_tokens,
        metadata.variant().map.thread_block.kv_tokens_per_iteration,
        args.max_tokens,
    );
    let samples = measure_runs(args.runs, args.warmup_iters, args.iters, || {
        MetalReplayRuntime::new(&stream).submit_replay(&replay).wait();
    });
    let uniform = contexts.iter().all(|&n| n == contexts[0]).then_some(contexts[0]);
    print_perf(num_tokens, num_reqs, uniform, Some("auto"), args.iters, &samples);
}

fn run_case(device: &Device, args: &Args, weights: &RealGQAWeights, counts: Vec<u32>, contexts: Vec<u32>) {
    if args.split_kv_variants.contains(&GQASplitKVBenchVariant::Auto) {
        measure_lookup(device, args, weights, &counts, &contexts);
    }
    let variants = args
        .split_kv_variants
        .iter()
        .copied()
        .filter(|v| *v != GQASplitKVBenchVariant::Auto)
        .collect::<Vec<_>>();
    if variants.is_empty() && !args.validate_sdpa_sliced_q && !args.subcomponents {
        return;
    }
    let fixture = RealGQAFixture::new(
        device,
        counts,
        contexts,
        args.max_tokens,
        weights,
        args.params,
        args.model,
    );
    if args.validate_sdpa_sliced_q {
        fixture.validate_sdpa_sliced_q_attention();
    }
    fixture.measure(&variants, args.warmup_iters, args.iters, args.runs);
    if args.subcomponents {
        fixture.measure_subcomponents(&args.selected_subcomponents, args.warmup_iters, args.iters, args.runs);
    }
}

struct RealGQAFixture<'a> {
    device: Device,
    stream: Stream,
    model: GQAModelProfile,
    params: GQABenchParams,
    num_tokens: u32,
    num_reqs: u32,
    max_tokens: usize,
    num_tokens_per_req: Vec<u32>,
    existing_context_lens: Vec<u32>,
    uniform_context_len: Option<u32>,
    end_context_len: u32,
    next_hidden_state: Buffer,
    replay: ReplayProgram,
    sliced_q_next_hidden_state: Buffer,
    sliced_q_replay: ReplayProgram,
    _hidden_state: Buffer,
    _kv_pages: Buffer,
    batch_metadata: GQAMetadataBuffers,
    sliced_q_batch_metadata: GQAMetadataBuffers,
    _page_ids: Buffer,
    _sliced_q_partial_output: Buffer,
    _sliced_q_partial_exp_sums: Buffer,
    _sliced_q_partial_max_logits: Buffer,
    _qgkv: Buffer,
    _q: Buffer,
    _g: Buffer,
    _k: Buffer,
    _v: Buffer,
    _q_norm_rope: Buffer,
    _k_norm_rope: Buffer,
    _sdpa_partial_exp_sums: Buffer,
    _sdpa_partial_max_logits: Buffer,
    _sdpa_partial_output: Buffer,
    _attention_output: Buffer,
    _sliced_q_attention_output: Buffer,
    _gated_attention_output: Buffer,
    _weights: &'a RealGQAWeights,
}

impl<'a> RealGQAFixture<'a> {
    fn new(
        device: &Device,
        num_tokens_per_req: Vec<u32>,
        existing_context_lens: Vec<u32>,
        max_tokens: usize,
        weights: &'a RealGQAWeights,
        mut params: GQABenchParams,
        model: GQAModelProfile,
    ) -> Self {
        assert!(
            !num_tokens_per_req.is_empty(),
            "GQA bench requires at least one request"
        );
        assert!(
            num_tokens_per_req.iter().all(|&num_req_tokens| num_req_tokens > 0),
            "GQA bench token counts per request must be positive"
        );
        assert_eq!(
            existing_context_lens.len(),
            num_tokens_per_req.len(),
            "GQA bench context and token count lists must have the same length"
        );
        let num_tokens = num_tokens_per_req.iter().sum::<u32>();
        let num_reqs = num_tokens_per_req
            .len()
            .try_into()
            .expect("GQA request count must fit u32");
        assert!(num_tokens as usize <= max_tokens, "GQA bench token capacity exceeded");
        let stream = Stream::new(device);
        let core = GQACore::new(
            model.model_layer_index,
            model.hidden_dim,
            model.head_dim,
            model.num_q_heads,
            model.num_kv_heads,
            (model.head_dim as f32).sqrt().recip(),
        );
        let config = GQAMetalConfig {
            group_size: GROUP_SIZE,
            bits: BITS,
            page_bytes: model.page_bytes(),
            rope_dim: GQA_ROPE_DIM,
            norm_eps: GQA_NORM_EPS,
            rope_theta: GQA_ROPE_THETA,
            rope_scaling: RopeScaling::Default,
            io_dtype: Dtype::Bfloat16,
        };
        let end_context_len = existing_context_lens
            .iter()
            .zip(&num_tokens_per_req)
            .map(|(&context, &num_req_tokens)| {
                context
                    .checked_add(num_req_tokens)
                    .expect("GQA bench request context length must fit u32")
            })
            .max()
            .unwrap_or(0);
        let uniform_context_len = existing_context_lens
            .iter()
            .all(|&context| context == existing_context_lens[0])
            .then_some(existing_context_lens[0]);
        let num_blocks = end_context_len.div_ceil(model.num_tokens_per_page()).max(1);
        let page_table_layout = GQAPageTableLayout {
            num_req_slots: num_reqs,
            num_blocks,
            num_gqa_layers: 1,
            num_page_ids_per_block: 1,
        };
        let backend = GQA::new(device, core, config, max_tokens, page_table_layout);
        let hidden_state = Buffer::from_slice(device, &hidden_fixture(num_tokens as usize, model.hidden_dim));
        let kv_pages = Buffer::new_zeroed(
            device,
            num_reqs as usize * num_blocks as usize * config.page_bytes as usize,
        );
        assert!(num_tokens as usize <= max_tokens);
        let q_heads_per_kv_head = model.num_q_heads / model.num_kv_heads;
        let sliced_kv_max_q_heads = q_heads_per_kv_head
            .min(params.sdpa_sliced_kv_max_q_heads as usize)
            .try_into()
            .expect("GQA max Q-head count must fit u32");
        let num_q_token_ranges = num_tokens_per_req
            .iter()
            .map(|&count| count.div_ceil(params.sdpa_sliced_q_max_q_tokens))
            .sum::<u32>();
        if params.sdpa_sliced_q_max_q_heads == 0 {
            let desired_max_q_heads = if (num_tokens as u64) < 4 * num_q_token_ranges as u64 {
                q_heads_per_kv_head.div_ceil(2)
            } else {
                q_heads_per_kv_head
            };
            let max_supported_q_heads = 256 / (params.sdpa_sliced_q_max_q_tokens / 8 * 32);
            params.sdpa_sliced_q_max_q_heads = desired_max_q_heads
                .min(max_supported_q_heads as usize)
                .try_into()
                .expect("GQA max Q-head count must fit u32");
        }
        let req_slots = (0..num_reqs).collect::<Vec<_>>();
        let materialized_cu_tokens = cu_tokens(&num_tokens_per_req)
            .into_iter()
            .map(|value| value as u32)
            .collect::<Vec<_>>();
        let request_shapes = RequestShape::from_batch(&existing_context_lens, &materialized_cu_tokens);
        let replay_bucket_policy = GQAReplayBucketPolicy::new(
            max_tokens.try_into().expect("GQA max token count must fit u32"),
            max_tokens.try_into().expect("GQA max token count must fit u32"),
        );
        let sdpa_config = backend_sdpa::Config {
            io_dtype: config.io_dtype,
            num_q_heads: model.num_q_heads.try_into().expect("GQA Q-head count must fit u32"),
            num_kv_heads: model.num_kv_heads.try_into().expect("GQA KV-head count must fit u32"),
            head_dim: model.head_dim.try_into().expect("GQA head_dim must fit u32"),
            tokens_per_page: model.num_tokens_per_page(),
        };
        let sliced_kv_variant = backend_sdpa::ExecutionVariant::sliced_kv(
            sdpa_config,
            params.sdpa_sliced_kv_kv_tokens_per_iteration,
            params.sdpa_sliced_kv_required_threads,
            sliced_kv_max_q_heads,
        );
        let sliced_kv_selection = Selector::new(
            backend_sdpa::Registry::from_variants(sdpa_config, vec![sliced_kv_variant]),
            max_tokens,
            max_tokens,
            max_tokens
                .checked_mul(sliced_kv_variant.map.thread_block.max_q_tokens as usize)
                .expect("GQA SDPA partial-state-group capacity must fit usize"),
        )
        .select(&request_shapes, &replay_bucket_policy, num_tokens);
        let batch_metadata = GQAMetadataBuffers::new(device, max_tokens, max_tokens);
        let shape = batch_metadata.update(
            &req_slots,
            &existing_context_lens,
            &materialized_cu_tokens,
            &sliced_kv_selection,
        );
        let page_ids = Buffer::from_slice(device, &page_table(num_reqs, num_blocks));
        let sliced_q_variant = backend_sdpa::ExecutionVariant::sliced_q_per_head(
            sdpa_config,
            params.sdpa_sliced_q_max_q_tokens,
            params.sdpa_sliced_q_kv_tokens_per_iteration,
            params.sdpa_sliced_q_max_q_heads,
        );
        let sliced_q_selection = Selector::new(
            backend_sdpa::Registry::from_variants(sdpa_config, vec![sliced_q_variant]),
            max_tokens,
            max_tokens,
            max_tokens
                .checked_mul(sliced_q_variant.map.thread_block.max_q_tokens as usize)
                .expect("GQA SDPA partial-state-group capacity must fit usize"),
        )
        .select(&request_shapes, &replay_bucket_policy, num_tokens);
        let sliced_q_batch_metadata = GQAMetadataBuffers::new(device, max_tokens, max_tokens);
        let sliced_q_replay_shape = sliced_q_batch_metadata.update(
            &req_slots,
            &existing_context_lens,
            &materialized_cu_tokens,
            &sliced_q_selection,
        );
        let qgkv = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.qgkv_dim() * Dtype::Bfloat16.item_size(),
        );
        let q = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.q_dim() * Dtype::Bfloat16.item_size(),
        );
        let g = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.q_dim() * Dtype::Bfloat16.item_size(),
        );
        let k = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.kv_dim() * Dtype::Bfloat16.item_size(),
        );
        let v = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.kv_dim() * Dtype::Bfloat16.item_size(),
        );
        let q_norm_rope = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.q_dim() * Dtype::Bfloat16.item_size(),
        );
        let k_norm_rope = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.kv_dim() * Dtype::Bfloat16.item_size(),
        );
        let num_sdpa_partial_outputs = shape.num_total_sdpa_map_task_templates as usize * model.num_q_heads;
        let sdpa_partial_exp_sums = Buffer::new_zeroed(device, num_sdpa_partial_outputs * size_of::<f32>());
        let sdpa_partial_max_logits = Buffer::new_zeroed(device, num_sdpa_partial_outputs * size_of::<f32>());
        let sdpa_partial_output = Buffer::new_zeroed(
            device,
            num_sdpa_partial_outputs * model.head_dim * Dtype::Bfloat16.item_size(),
        );
        let attention_output = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.q_dim() * Dtype::Bfloat16.item_size(),
        );
        let sliced_q_attention_output = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.q_dim() * Dtype::Bfloat16.item_size(),
        );
        let num_reserved_sdpa_partial_output_tokens = sliced_q_replay_shape.num_total_sdpa_map_task_templates as usize
            * params.sdpa_sliced_q_max_q_tokens as usize;
        let sliced_q_partial_output = Buffer::new_zeroed(
            device,
            num_reserved_sdpa_partial_output_tokens * model.q_dim() * Dtype::Bfloat16.item_size(),
        );
        let sliced_q_partial_exp_sums = Buffer::new_zeroed(
            device,
            num_reserved_sdpa_partial_output_tokens * model.num_q_heads * size_of::<f32>(),
        );
        let sliced_q_partial_max_logits = Buffer::new_zeroed(
            device,
            num_reserved_sdpa_partial_output_tokens * model.num_q_heads * size_of::<f32>(),
        );
        let gated_attention_output = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.q_dim() * Dtype::Bfloat16.item_size(),
        );
        let next_hidden_state = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.hidden_dim * Dtype::Bfloat16.item_size(),
        );
        let sliced_q_next_hidden_state = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.hidden_dim * Dtype::Bfloat16.item_size(),
        );
        let mut recorder = MetalReplayRuntime::new(&stream).create_recorder();
        let _ = <GQA as ReplayLayer>::record(
            &backend,
            &mut recorder,
            GQAInput {
                page_table_layout,
                gqa_layer_index: ReplayU32::Fixed(0),
                batch_metadata: &batch_metadata,
                hidden_state: &hidden_state,
                next_hidden_state: &next_hidden_state,
                kv_cache: GQAKVCacheBindings {
                    kv_pages: &kv_pages,
                    page_ids: &page_ids,
                },
                weights: GQAWeights {
                    qgkv_weight: &weights.qgkv_weight,
                    qgkv_scales: &weights.qgkv_scales,
                    qgkv_biases: &weights.qgkv_biases,
                    q_norm_weight: &weights.q_norm_weight,
                    k_norm_weight: &weights.k_norm_weight,
                    output_weight: &weights.output_weight,
                    output_scales: &weights.output_scales,
                    output_biases: &weights.output_biases,
                },
                scratch: GQAScratchBindings {
                    qgkv: &qgkv,
                    q: &q,
                    g: &g,
                    k: &k,
                    v: &v,
                    q_norm_rope: &q_norm_rope,
                    k_norm_rope: &k_norm_rope,
                    sdpa_partial_exp_sums: &sdpa_partial_exp_sums,
                    sdpa_partial_max_logits: &sdpa_partial_max_logits,
                    sdpa_partial_output: &sdpa_partial_output,
                    attention_output: &attention_output,
                    gated_attention_output: &gated_attention_output,
                },
                num_active_tokens: ReplayU32::Fixed(shape.num_tokens),
            },
        );
        let replay = recorder.build();
        let qgkv_matmul = affine_quantized::Matmul::new(device, gqa_qgkv_affine_config(model));
        let qgkv_to_q_g_k_v = backend_qgkv_split::Compute::new(device, gqa_qgkv_to_q_g_k_v_config(model));
        let q_norm_rope_kernel = rms_norm_rope::Compute::new(device, norm_rope_config(model.num_q_heads, model));
        let k_norm_rope_kernel = rms_norm_rope::Compute::new(device, norm_rope_config(model.num_kv_heads, model));
        let kv_page_write =
            backend_kv_page_write::Compute::new(device, gqa_kv_page_write_config(model, config.page_bytes));
        let gate = backend_activation_gate::Compute::new(device, gqa_gate_config(model));
        let output = affine_quantized::Matmul::new(device, gqa_output_affine_config(model));
        let metal_page_table_layout = gqa_page_table_layout(num_reqs, end_context_len, model);
        let sliced_q_config = gqa_sdpa_sliced_q_config(metal_page_table_layout, model);
        let sliced_q_shape = backend_sliced_q::Shape {
            num_total_tokens: num_tokens,
            num_total_q_token_tiles: sliced_q_replay_shape.num_q_token_tiles,
            num_total_sdpa_map_task_templates: sliced_q_replay_shape.num_total_sdpa_map_task_templates,
        };
        let sliced_q_kernel = backend_sliced_q::Compute::new(
            device,
            sliced_q_config,
            sliced_q_batch_metadata.variant(),
            sliced_q_shape,
        );
        let mut sliced_q_builder = MetalReplayRuntime::new(&stream).create_recorder();
        sliced_q_builder.record_with_barrier_before(ReplayOp::opaque(qgkv_matmul.invoke(
            num_tokens,
            ReplayU32::Fixed(num_tokens),
            &qgkv,
            0,
            &hidden_state,
            0,
            &weights.qgkv_weight,
            0,
            &weights.qgkv_scales,
            0,
            &weights.qgkv_biases,
            0,
        )));
        sliced_q_builder.record_with_barrier_before(ReplayOp::opaque(qgkv_to_q_g_k_v.invoke(
            backend_qgkv_split::Shape {
                num_total_tokens: num_tokens,
            },
            backend_qgkv_split::Buffers {
                qgkv: &qgkv,
                q: &q,
                g: &g,
                k: &k,
                v: &v,
            },
            ReplayU32::Fixed(num_tokens),
        )));
        sliced_q_builder.record_with_barrier_before(ReplayOp::opaque(q_norm_rope_kernel.invoke(
            norm_rope_shape(num_tokens, model.num_q_heads, model),
            rms_norm_rope::Buffers {
                input: &q,
                norm_weight: &weights.q_norm_weight,
                flat_token_indices: batch_metadata.flat_token_indices(),
                output: &q_norm_rope,
            },
            ReplayU32::Fixed(num_tokens),
        )));
        sliced_q_builder.record(ReplayOp::opaque(k_norm_rope_kernel.invoke(
            norm_rope_shape(num_tokens, model.num_kv_heads, model),
            rms_norm_rope::Buffers {
                input: &k,
                norm_weight: &weights.k_norm_weight,
                flat_token_indices: batch_metadata.flat_token_indices(),
                output: &k_norm_rope,
            },
            ReplayU32::Fixed(num_tokens),
        )));
        sliced_q_builder.record_with_barrier_before(ReplayOp::opaque(kv_page_write.invoke(
            backend_kv_page_write::Shape {
                num_total_token_writes: num_tokens,
                page_table_layout: metal_page_table_layout,
            },
            backend_kv_page_write::Buffers {
                pages: &kv_pages,
                flat_k: &k_norm_rope,
                flat_v: &v,
                req_slots: sliced_q_batch_metadata.req_slots(),
                flat_token_indices: batch_metadata.flat_token_indices(),
                page_ids: &page_ids,
            },
            ReplayU32::Fixed(num_tokens),
            ReplayU32::Fixed(0),
        )));
        sliced_q_builder.record_with_barrier_before(ReplayOp::opaque(sliced_q_kernel.invoke_map(
            backend_sliced_q::MapBuffers {
                q: &q_norm_rope,
                kv_pages: &kv_pages,
                req_slots: batch_metadata.req_slots(),
                page_ids: &page_ids,
                visible_kv_token_ranges: sliced_q_batch_metadata.visible_kv_token_ranges(),
                q_token_ranges: sliced_q_batch_metadata.q_token_ranges(),
                sdpa_map_task_templates: sliced_q_batch_metadata.sdpa_map_task_templates(),
                partial_output: &sliced_q_partial_output,
                partial_exp_sums: &sliced_q_partial_exp_sums,
                partial_max_logits: &sliced_q_partial_max_logits,
            },
            ReplayU32::Fixed(0),
            ReplayU32::Fixed(num_tokens),
            ReplayU32::Fixed(sliced_q_shape.num_total_q_token_tiles),
            ReplayU32::Fixed(sliced_q_shape.num_total_sdpa_map_task_templates),
        )));
        sliced_q_builder.record_with_barrier_before(ReplayOp::opaque(sliced_q_kernel.invoke_reduce(
            backend_sliced_q::ReduceBuffers {
                partial_output: &sliced_q_partial_output,
                partial_exp_sums: &sliced_q_partial_exp_sums,
                partial_max_logits: &sliced_q_partial_max_logits,
                q_token_ranges: sliced_q_batch_metadata.q_token_ranges(),
                cu_sdpa_partial_outputs: sliced_q_batch_metadata.cu_sdpa_partial_outputs(),
                output: &sliced_q_attention_output,
            },
            ReplayU32::Fixed(sliced_q_shape.num_total_q_token_tiles),
        )));
        sliced_q_builder.record_with_barrier_before(ReplayOp::opaque(gate.invoke(
            backend_activation_gate::Shape {
                num_total_tokens: num_tokens,
            },
            backend_activation_gate::Buffers {
                attention_output: &sliced_q_attention_output,
                g: &g,
                output: &gated_attention_output,
            },
            ReplayU32::Fixed(num_tokens),
        )));
        sliced_q_builder.record_with_barrier_before(ReplayOp::opaque(output.invoke(
            num_tokens,
            ReplayU32::Fixed(num_tokens),
            &sliced_q_next_hidden_state,
            0,
            &gated_attention_output,
            0,
            &weights.output_weight,
            0,
            &weights.output_scales,
            0,
            &weights.output_biases,
            0,
        )));
        let sliced_q_replay = sliced_q_builder.build();
        Self {
            device: device.clone(),
            stream,
            model,
            params,
            num_tokens,
            num_reqs,
            max_tokens,
            num_tokens_per_req,
            existing_context_lens,
            uniform_context_len,
            end_context_len,
            next_hidden_state,
            replay,
            sliced_q_next_hidden_state,
            sliced_q_replay,
            _hidden_state: hidden_state,
            _kv_pages: kv_pages,
            batch_metadata,
            sliced_q_batch_metadata,
            _page_ids: page_ids,
            _sliced_q_partial_output: sliced_q_partial_output,
            _sliced_q_partial_exp_sums: sliced_q_partial_exp_sums,
            _sliced_q_partial_max_logits: sliced_q_partial_max_logits,
            _qgkv: qgkv,
            _q: q,
            _g: g,
            _k: k,
            _v: v,
            _q_norm_rope: q_norm_rope,
            _k_norm_rope: k_norm_rope,
            _sdpa_partial_exp_sums: sdpa_partial_exp_sums,
            _sdpa_partial_max_logits: sdpa_partial_max_logits,
            _sdpa_partial_output: sdpa_partial_output,
            _attention_output: attention_output,
            _sliced_q_attention_output: sliced_q_attention_output,
            _gated_attention_output: gated_attention_output,
            _weights: weights,
        }
    }

    fn run_sdpa_sliced_kv(&self) {
        MetalReplayRuntime::new(&self.stream).submit_replay(&self.replay).wait();
    }

    fn run_sdpa_sliced_q(&self) {
        MetalReplayRuntime::new(&self.stream)
            .submit_replay(&self.sliced_q_replay)
            .wait();
    }

    fn validate_sdpa_sliced_q_attention(&self) {
        let sdpa_config = gqa_sdpa_config(self.num_reqs, self.end_context_len, self.model);
        let sdpa_shape = gqa_sdpa_shape(self.batch_metadata.replay_shape());
        let sdpa =
            backend_sliced_kv::Compute::new(&self.device, sdpa_config, self.batch_metadata.variant(), sdpa_shape);
        let sdpa_sliced_kv_replay = {
            let mut recorder = MetalReplayRuntime::new(&self.stream).create_recorder();
            recorder.record(ReplayOp::opaque(sdpa.invoke_map(
                backend_sliced_kv::MapBuffers {
                    q: &self._q_norm_rope,
                    kv_pages: &self._kv_pages,
                    req_slots: self.batch_metadata.req_slots(),
                    page_ids: &self._page_ids,
                    sdpa_map_task_templates: self.batch_metadata.sdpa_map_task_templates(),
                    partial_exp_sums: &self._sdpa_partial_exp_sums,
                    partial_max_logits: &self._sdpa_partial_max_logits,
                    partial_output: &self._sdpa_partial_output,
                },
                ReplayU32::Fixed(0),
                ReplayU32::Fixed(sdpa_shape.num_total_tokens),
                ReplayU32::Fixed(sdpa_shape.num_total_sdpa_map_task_templates),
            )));
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_reduce(
                backend_sliced_kv::ReduceBuffers {
                    partial_exp_sums: &self._sdpa_partial_exp_sums,
                    partial_max_logits: &self._sdpa_partial_max_logits,
                    partial_output: &self._sdpa_partial_output,
                    cu_sdpa_partial_outputs: self.batch_metadata.cu_sdpa_partial_outputs(),
                    output: &self._attention_output,
                },
                ReplayU32::Fixed(sdpa_shape.num_total_tokens),
            )));
            recorder.build()
        };
        let sliced_q_config = gqa_sdpa_sliced_q_config(
            gqa_page_table_layout(self.num_reqs, self.end_context_len, self.model),
            self.model,
        );
        let sliced_q_shape = backend_sliced_q::Shape {
            num_total_tokens: self.num_tokens,
            num_total_q_token_tiles: self.sliced_q_batch_metadata.replay_shape().num_q_token_tiles,
            num_total_sdpa_map_task_templates: self
                .sliced_q_batch_metadata
                .replay_shape()
                .num_total_sdpa_map_task_templates,
        };
        let tiled = backend_sliced_q::Compute::new(
            &self.device,
            sliced_q_config,
            self.sliced_q_batch_metadata.variant(),
            sliced_q_shape,
        );
        let sliced_q_replay = {
            let mut recorder = MetalReplayRuntime::new(&self.stream).create_recorder();
            recorder.record(ReplayOp::opaque(tiled.invoke_map(
                backend_sliced_q::MapBuffers {
                    q: &self._q_norm_rope,
                    kv_pages: &self._kv_pages,
                    req_slots: self.sliced_q_batch_metadata.req_slots(),
                    page_ids: &self._page_ids,
                    visible_kv_token_ranges: self.sliced_q_batch_metadata.visible_kv_token_ranges(),
                    q_token_ranges: self.sliced_q_batch_metadata.q_token_ranges(),
                    sdpa_map_task_templates: self.sliced_q_batch_metadata.sdpa_map_task_templates(),
                    partial_output: &self._sliced_q_partial_output,
                    partial_exp_sums: &self._sliced_q_partial_exp_sums,
                    partial_max_logits: &self._sliced_q_partial_max_logits,
                },
                ReplayU32::Fixed(0),
                ReplayU32::Fixed(sliced_q_shape.num_total_tokens),
                ReplayU32::Fixed(sliced_q_shape.num_total_q_token_tiles),
                ReplayU32::Fixed(sliced_q_shape.num_total_sdpa_map_task_templates),
            )));
            recorder.record_with_barrier_before(ReplayOp::opaque(tiled.invoke_reduce(
                backend_sliced_q::ReduceBuffers {
                    partial_output: &self._sliced_q_partial_output,
                    partial_exp_sums: &self._sliced_q_partial_exp_sums,
                    partial_max_logits: &self._sliced_q_partial_max_logits,
                    q_token_ranges: self.sliced_q_batch_metadata.q_token_ranges(),
                    cu_sdpa_partial_outputs: self.sliced_q_batch_metadata.cu_sdpa_partial_outputs(),
                    output: &self._sliced_q_attention_output,
                },
                ReplayU32::Fixed(sliced_q_shape.num_total_q_token_tiles),
            )));
            recorder.build()
        };
        let runtime = MetalReplayRuntime::new(&self.stream);
        runtime.submit_replay(&sdpa_sliced_kv_replay).wait();
        runtime.submit_replay(&sliced_q_replay).wait();

        let sdpa_sliced_kv_output = &self._attention_output;
        let num_q_slots = self.num_tokens as usize * self.model.q_dim();
        let (max_diff_index, expected, actual, max_abs_diff) =
            max_bf16_diff(sdpa_sliced_kv_output, &self._sliced_q_attention_output, num_q_slots);
        if max_abs_diff > 0.0625 {
            let reference = gqa_attention_reference_at(
                &self._q_norm_rope,
                &self._kv_pages,
                &self._page_ids,
                &self.batch_metadata,
                self.end_context_len.div_ceil(self.model.num_tokens_per_page()).max(1),
                max_diff_index,
                self.model,
            );
            panic!(
                "GQA SplitKV SlicedQ mismatch at {max_diff_index}: sliced_kv={expected} sliced_q={actual} \
                 cpu_reference={reference} max_abs_diff={max_abs_diff} tolerance=0.0625"
            );
        }
    }

    fn measure(&self, variants: &[GQASplitKVBenchVariant], warmup_iters: usize, iters: usize, runs: usize) {
        let num_tokens_per_req = self
            .num_tokens_per_req
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let existing_context_lens = self
            .existing_context_lens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "case component=gqa model={} num_tokens_per_req=[{num_tokens_per_req}] \
             existing_context_lens=[{existing_context_lens}] max_tokens={} sdpa_sliced_q_max_q_tokens={} \
             sdpa_sliced_q_kv_tokens_per_iteration={} sdpa_sliced_q_max_q_heads={}",
            self.model.k,
            self.max_tokens,
            self.params.sdpa_sliced_q_max_q_tokens,
            self.params.sdpa_sliced_q_kv_tokens_per_iteration,
            self.params.sdpa_sliced_q_max_q_heads,
        );
        print_split_selection(
            "sliced_kv",
            &self.batch_metadata,
            1,
            self.params.sdpa_sliced_kv_kv_tokens_per_iteration,
            self.max_tokens,
        );
        print_split_selection(
            "sliced_q",
            &self.sliced_q_batch_metadata,
            self.params.sdpa_sliced_q_max_q_tokens,
            self.params.sdpa_sliced_q_kv_tokens_per_iteration,
            self.max_tokens,
        );
        if variants.contains(&GQASplitKVBenchVariant::SlicedKv) {
            let samples = measure_runs(runs, warmup_iters, iters, || self.run_sdpa_sliced_kv());
            let _ = self.next_hidden_state.len_bytes();
            print_perf(
                self.num_tokens,
                self.num_reqs,
                self.uniform_context_len,
                Some("sliced_kv"),
                iters,
                &samples,
            );
        }
        if variants.contains(&GQASplitKVBenchVariant::SlicedQ) {
            let sliced_q_samples = measure_runs(runs, warmup_iters, iters, || self.run_sdpa_sliced_q());
            let _ = self.sliced_q_next_hidden_state.len_bytes();
            print_perf(
                self.num_tokens,
                self.num_reqs,
                self.uniform_context_len,
                Some("sliced_q"),
                iters,
                &sliced_q_samples,
            );
        }
    }

    fn measure_subcomponents(&self, selected_subcomponents: &[String], warmup_iters: usize, iters: usize, runs: usize) {
        let qgkv_matmul = affine_quantized::Matmul::new(&self.device, gqa_qgkv_affine_config(self.model));
        let qgkv_to_q_g_k_v = backend_qgkv_split::Compute::new(&self.device, gqa_qgkv_to_q_g_k_v_config(self.model));
        let q_norm_rope =
            rms_norm_rope::Compute::new(&self.device, norm_rope_config(self.model.num_q_heads, self.model));
        let k_norm_rope =
            rms_norm_rope::Compute::new(&self.device, norm_rope_config(self.model.num_kv_heads, self.model));
        let kv_page_write = backend_kv_page_write::Compute::new(
            &self.device,
            gqa_kv_page_write_config(self.model, self.model.page_bytes()),
        );
        let gate = backend_activation_gate::Compute::new(&self.device, gqa_gate_config(self.model));
        let output = affine_quantized::Matmul::new(&self.device, gqa_output_affine_config(self.model));
        let sdpa_config = gqa_sdpa_config(self.num_reqs, self.end_context_len, self.model);
        let sdpa_shape = gqa_sdpa_shape(self.batch_metadata.replay_shape());
        let sdpa =
            backend_sliced_kv::Compute::new(&self.device, sdpa_config, self.batch_metadata.variant(), sdpa_shape);

        let qgkv_replay = build_single_invocation_replay(
            &self.stream,
            qgkv_matmul.invoke(
                self.num_tokens,
                ReplayU32::Fixed(self.num_tokens),
                &self._qgkv,
                0,
                &self._hidden_state,
                0,
                &self._weights.qgkv_weight,
                0,
                &self._weights.qgkv_scales,
                0,
                &self._weights.qgkv_biases,
                0,
            ),
        );
        let split_replay = build_single_invocation_replay(
            &self.stream,
            qgkv_to_q_g_k_v.invoke(
                backend_qgkv_split::Shape {
                    num_total_tokens: self.num_tokens,
                },
                backend_qgkv_split::Buffers {
                    qgkv: &self._qgkv,
                    q: &self._q,
                    g: &self._g,
                    k: &self._k,
                    v: &self._v,
                },
                ReplayU32::Fixed(self.num_tokens),
            ),
        );
        let q_norm_rope_replay = build_single_invocation_replay(
            &self.stream,
            q_norm_rope.invoke(
                norm_rope_shape(self.num_tokens, self.model.num_q_heads, self.model),
                rms_norm_rope::Buffers {
                    input: &self._q,
                    norm_weight: &self._weights.q_norm_weight,
                    flat_token_indices: self.batch_metadata.flat_token_indices(),
                    output: &self._q_norm_rope,
                },
                ReplayU32::Fixed(self.num_tokens),
            ),
        );
        let k_norm_rope_replay = build_single_invocation_replay(
            &self.stream,
            k_norm_rope.invoke(
                norm_rope_shape(self.num_tokens, self.model.num_kv_heads, self.model),
                rms_norm_rope::Buffers {
                    input: &self._k,
                    norm_weight: &self._weights.k_norm_weight,
                    flat_token_indices: self.batch_metadata.flat_token_indices(),
                    output: &self._k_norm_rope,
                },
                ReplayU32::Fixed(self.num_tokens),
            ),
        );
        let page_table_layout = gqa_page_table_layout(self.num_reqs, self.end_context_len, self.model);
        let kv_page_write_replay = build_single_invocation_replay(
            &self.stream,
            kv_page_write.invoke(
                backend_kv_page_write::Shape {
                    num_total_token_writes: self.num_tokens,
                    page_table_layout,
                },
                backend_kv_page_write::Buffers {
                    pages: &self._kv_pages,
                    flat_k: &self._k_norm_rope,
                    flat_v: &self._v,
                    req_slots: self.batch_metadata.req_slots(),
                    flat_token_indices: self.batch_metadata.flat_token_indices(),
                    page_ids: &self._page_ids,
                },
                ReplayU32::Fixed(self.num_tokens),
                ReplayU32::Fixed(0),
            ),
        );
        let sdpa_sliced_kv_replay = {
            let mut recorder = MetalReplayRuntime::new(&self.stream).create_recorder();
            recorder.record(ReplayOp::opaque(sdpa.invoke_map(
                backend_sliced_kv::MapBuffers {
                    q: &self._q_norm_rope,
                    kv_pages: &self._kv_pages,
                    req_slots: self.batch_metadata.req_slots(),
                    page_ids: &self._page_ids,
                    sdpa_map_task_templates: self.batch_metadata.sdpa_map_task_templates(),
                    partial_exp_sums: &self._sdpa_partial_exp_sums,
                    partial_max_logits: &self._sdpa_partial_max_logits,
                    partial_output: &self._sdpa_partial_output,
                },
                ReplayU32::Fixed(0),
                ReplayU32::Fixed(self.num_tokens),
                ReplayU32::Fixed(sdpa_shape.num_total_sdpa_map_task_templates),
            )));
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_reduce(
                backend_sliced_kv::ReduceBuffers {
                    partial_exp_sums: &self._sdpa_partial_exp_sums,
                    partial_max_logits: &self._sdpa_partial_max_logits,
                    partial_output: &self._sdpa_partial_output,
                    cu_sdpa_partial_outputs: self.batch_metadata.cu_sdpa_partial_outputs(),
                    output: &self._attention_output,
                },
                ReplayU32::Fixed(self.num_tokens),
            )));
            recorder.build()
        };
        let sliced_q_config = gqa_sdpa_sliced_q_config(page_table_layout, self.model);
        let sliced_q_shape = backend_sliced_q::Shape {
            num_total_tokens: self.num_tokens,
            num_total_q_token_tiles: self.sliced_q_batch_metadata.replay_shape().num_q_token_tiles,
            num_total_sdpa_map_task_templates: self
                .sliced_q_batch_metadata
                .replay_shape()
                .num_total_sdpa_map_task_templates,
        };
        let sliced_q_kernel = backend_sliced_q::Compute::new(
            &self.device,
            sliced_q_config,
            self.sliced_q_batch_metadata.variant(),
            sliced_q_shape,
        );
        let sliced_q_replay = {
            let mut recorder = MetalReplayRuntime::new(&self.stream).create_recorder();
            recorder.record(ReplayOp::opaque(sliced_q_kernel.invoke_map(
                backend_sliced_q::MapBuffers {
                    q: &self._q_norm_rope,
                    kv_pages: &self._kv_pages,
                    req_slots: self.sliced_q_batch_metadata.req_slots(),
                    page_ids: &self._page_ids,
                    visible_kv_token_ranges: self.sliced_q_batch_metadata.visible_kv_token_ranges(),
                    q_token_ranges: self.sliced_q_batch_metadata.q_token_ranges(),
                    sdpa_map_task_templates: self.sliced_q_batch_metadata.sdpa_map_task_templates(),
                    partial_output: &self._sliced_q_partial_output,
                    partial_exp_sums: &self._sliced_q_partial_exp_sums,
                    partial_max_logits: &self._sliced_q_partial_max_logits,
                },
                ReplayU32::Fixed(0),
                ReplayU32::Fixed(self.num_tokens),
                ReplayU32::Fixed(sliced_q_shape.num_total_q_token_tiles),
                ReplayU32::Fixed(sliced_q_shape.num_total_sdpa_map_task_templates),
            )));
            recorder.record_with_barrier_before(ReplayOp::opaque(sliced_q_kernel.invoke_reduce(
                backend_sliced_q::ReduceBuffers {
                    partial_output: &self._sliced_q_partial_output,
                    partial_exp_sums: &self._sliced_q_partial_exp_sums,
                    partial_max_logits: &self._sliced_q_partial_max_logits,
                    q_token_ranges: self.sliced_q_batch_metadata.q_token_ranges(),
                    cu_sdpa_partial_outputs: self.sliced_q_batch_metadata.cu_sdpa_partial_outputs(),
                    output: &self._sliced_q_attention_output,
                },
                ReplayU32::Fixed(sliced_q_shape.num_total_q_token_tiles),
            )));
            recorder.build()
        };
        let runtime = MetalReplayRuntime::new(&self.stream);
        runtime.submit_replay(&sdpa_sliced_kv_replay).wait();
        runtime.submit_replay(&sliced_q_replay).wait();
        let sdpa_sliced_kv_output = &self._attention_output;
        assert_bf16_close(
            sdpa_sliced_kv_output,
            &self._sliced_q_attention_output,
            self.num_tokens as usize * self.model.q_dim(),
            0.0625,
        );
        let gate_replay = build_single_invocation_replay(
            &self.stream,
            gate.invoke(
                backend_activation_gate::Shape {
                    num_total_tokens: self.num_tokens,
                },
                backend_activation_gate::Buffers {
                    attention_output: &self._attention_output,
                    g: &self._g,
                    output: &self._gated_attention_output,
                },
                ReplayU32::Fixed(self.num_tokens),
            ),
        );
        let output_replay = build_single_invocation_replay(
            &self.stream,
            output.invoke(
                self.num_tokens,
                ReplayU32::Fixed(self.num_tokens),
                &self.next_hidden_state,
                0,
                &self._gated_attention_output,
                0,
                &self._weights.output_weight,
                0,
                &self._weights.output_scales,
                0,
                &self._weights.output_biases,
                0,
            ),
        );

        self.measure_subcomponent(selected_subcomponents, "qgkv", &qgkv_replay, warmup_iters, iters, runs);
        self.measure_subcomponent(
            selected_subcomponents,
            "qgkv-to-q-g-k-v",
            &split_replay,
            warmup_iters,
            iters,
            runs,
        );
        self.measure_subcomponent(
            selected_subcomponents,
            "q-norm-rope",
            &q_norm_rope_replay,
            warmup_iters,
            iters,
            runs,
        );
        self.measure_subcomponent(
            selected_subcomponents,
            "k-norm-rope",
            &k_norm_rope_replay,
            warmup_iters,
            iters,
            runs,
        );
        self.measure_subcomponent(
            selected_subcomponents,
            "kv-page-write",
            &kv_page_write_replay,
            warmup_iters,
            iters,
            runs,
        );
        self.measure_subcomponent(
            selected_subcomponents,
            "sdpa-sliced-kv",
            &sdpa_sliced_kv_replay,
            warmup_iters,
            iters,
            runs,
        );
        self.measure_subcomponent(
            selected_subcomponents,
            "sdpa-sliced-q",
            &sliced_q_replay,
            warmup_iters,
            iters,
            runs,
        );
        self.measure_subcomponent(selected_subcomponents, "gate", &gate_replay, warmup_iters, iters, runs);
        self.measure_subcomponent(
            selected_subcomponents,
            "output",
            &output_replay,
            warmup_iters,
            iters,
            runs,
        );
    }

    fn measure_subcomponent(
        &self,
        selected_subcomponents: &[String],
        name: &str,
        replay: &ReplayProgram,
        warmup_iters: usize,
        iters: usize,
        runs: usize,
    ) {
        if !selected_subcomponents.iter().any(|selected| selected == name) {
            return;
        }
        let samples = measure_runs(runs, warmup_iters, iters, || {
            MetalReplayRuntime::new(&self.stream).submit_replay(replay).wait();
        });
        print_named_perf(
            &format!("gqa.{name}"),
            self.num_tokens,
            self.num_reqs,
            self.uniform_context_len,
            iters,
            &samples,
        );
    }
}

fn print_split_selection(
    variant: &str,
    metadata: &GQAMetadataBuffers,
    max_q_tokens: u32,
    kv_tokens_per_iteration: u32,
    max_tokens: usize,
) {
    let shape = metadata.replay_shape();
    let cu_kv_splits = metadata
        .cu_sdpa_partial_outputs()
        .read_typed::<u32>(0, shape.num_q_token_tiles as usize + 1);
    let num_map_task_templates_per_q_token_range = cu_kv_splits.windows(2).map(|cu| cu[1] - cu[0]).collect::<Vec<_>>();
    let num_active_q_tokens_per_q_token_range = if max_q_tokens == 1 {
        vec![1; shape.num_q_token_tiles as usize]
    } else {
        metadata
            .q_token_ranges()
            .read_typed::<u32>(0, shape.num_q_token_tiles as usize * 2)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|range| range[1] - range[0])
            .collect::<Vec<_>>()
    };
    let map_task_template_values = metadata
        .sdpa_map_task_templates()
        .read_typed::<u32>(0, shape.num_sdpa_map_task_templates as usize * 3);
    let mut kv_iterations_per_map_task_template_histogram = BTreeMap::new();
    for num_kv_iterations in map_task_template_values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|task| (task[2] - task[1]).div_ceil(kv_tokens_per_iteration))
    {
        *kv_iterations_per_map_task_template_histogram
            .entry(num_kv_iterations)
            .or_insert(0usize) += 1;
    }
    let kv_iterations_per_map_task_template_histogram = kv_iterations_per_map_task_template_histogram
        .into_iter()
        .map(|(num_kv_iterations, count)| format!("{num_kv_iterations}x{count}"))
        .collect::<Vec<_>>()
        .join(",");
    let num_active_partial_state_groups = num_active_q_tokens_per_q_token_range
        .iter()
        .zip(&num_map_task_templates_per_q_token_range)
        .map(|(&num_q_tokens, &num_templates)| u64::from(num_q_tokens) * u64::from(num_templates))
        .sum::<u64>();
    let reserved_partial_state_groups = shape
        .num_sdpa_map_task_templates
        .checked_mul(max_q_tokens)
        .expect("GQA reserved partial-state-group count must fit u32");
    let replay_reserved_partial_state_groups = shape
        .num_total_sdpa_map_task_templates
        .checked_mul(max_q_tokens)
        .expect("GQA replay reserved partial-state-group count must fit u32");
    println!(
        "split_selection component=gqa variant={variant} max_tokens={max_tokens} num_active_q_tokens={} \
         num_q_token_ranges={} num_active_q_tokens_per_q_token_range={num_active_q_tokens_per_q_token_range:?} \
         num_map_task_templates={} max_map_task_templates={max_tokens} \
         num_active_partial_state_groups={num_active_partial_state_groups} \
         num_reserved_partial_state_groups={reserved_partial_state_groups} num_replay_map_task_template_slots={} \
         num_replay_reserved_partial_state_groups={replay_reserved_partial_state_groups} \
         num_map_task_templates_per_q_token_range={num_map_task_templates_per_q_token_range:?} \
         num_kv_iterations_per_map_task_template_histogram=[{kv_iterations_per_map_task_template_histogram}]",
        shape.num_tokens,
        shape.num_q_token_tiles,
        shape.num_sdpa_map_task_templates,
        shape.num_total_sdpa_map_task_templates,
    );
}
