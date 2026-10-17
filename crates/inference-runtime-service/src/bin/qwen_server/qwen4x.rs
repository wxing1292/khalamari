use std::sync::Arc;

use clap::Parser;
use inference_executor_metal::model::qwen::v4_x::Qwen4xExecutorConfig;
use inference_executor_metal::model::qwen::v4_x::init_qwen4x_model;
use inference_runtime_core::Result;
use inference_runtime_core::config::CacheLaneRuntimeConfig;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::log_err_unavailable;
use inference_runtime_core::log_info_invalid_argument;
use inference_runtime_core::runtime::resource::processor::ResourceProcessors;
use inference_runtime_launcher::args::Qwen35Args;

use crate::codec::qwen::QwenCodec;
use crate::executor::ReplayableModelExecutors;
use crate::qwen_server::config::Qwen35Config;
use crate::qwen_server::sizing::block_cache_capacity;
use crate::qwen_server::sizing::context_window;
use crate::rpc::HTTPService;
use crate::runtime::serve_replay_model;

const TOKENS_PER_CACHE_BLOCK: usize = 2048;

pub fn run() {
    if let Err(error) = run_inner() {
        eprintln!("unable to start qwen4x: {error}");
        std::process::exit(1);
    }
}
fn run_inner() -> Result<()> {
    let args = Qwen35Args::parse();
    if args.spec.hf_spec_model_dir.is_some() || args.spec.spec_type.is_some() || args.spec.num_spec_tokens.is_some() {
        return Err(log_info_invalid_argument!(
            "Qwen4x currently supports Main-only execution"
        ));
    }
    // The shared hybrid-server CLI owns scheduler, page, transport, and telemetry options.
    let config = Qwen35Config::from_args(args)?;
    config.telemetry_config().init();
    let codec = Arc::new(
        QwenCodec::load(config.hf_model_dir())
            .map_err(|error| log_err_unavailable!("unable to load Qwen4x codec: {error}"))?,
    );
    let scheduler = config.scheduler_config();
    let model = init_qwen4x_model(
        config.hf_model_dir(),
        Qwen4xExecutorConfig {
            max_requests: config.max_running_requests(),
            max_tokens: scheduler.max_tokens,
            max_tokens_per_request: scheduler.max_tokens_per_request,
            num_cache_pages: config.num_cache_pages(),
            num_tokens_per_block: TOKENS_PER_CACHE_BLOCK,
        },
    )
    .map_err(|error| log_err_unavailable!("unable to initialize Qwen4x: {error}"))?;
    let kv = model.num_kv_page_ids_per_block();
    let state = model.num_state_page_ids_per_block();
    let runtime = RuntimeConfig {
        max_running_requests: config.max_running_requests(),
        executor_hibernation_timeout: config.executor_hibernation_timeout(),
        executor_hibernation_mode: config.executor_hibernation_mode(),
        context_window: context_window(model.model_config().text_config.max_position_embeddings, 0)?,
        num_tokens_per_cache_block: TOKENS_PER_CACHE_BLOCK,
        num_pages: config.num_cache_pages(),
        cache_lanes: vec![CacheLaneRuntimeConfig {
            num_pages_per_kv_block: kv,
            num_pages_per_state_block: state,
            block_cache_capacity: block_cache_capacity(config.num_cache_pages(), kv, state)?,
        }],
    };
    tracing::info!(
        model_mode = "vanilla",
        cache_block_tokens = TOKENS_PER_CACHE_BLOCK,
        num_kv_pages_per_block = kv,
        num_state_pages_per_block = state,
        max_requests = scheduler.max_requests,
        max_tokens = scheduler.max_tokens,
        "Qwen4x configured"
    );
    serve_replay_model::<TOKENS_PER_CACHE_BLOCK, 1, _>(
        config.grpc_listen_addr(),
        config.http_listen_addr(),
        HTTPService::ChatCompletions(codec),
        Arc::new(ResourceProcessors::new()),
        runtime,
        scheduler,
        ReplayableModelExecutors::new(model, vec![]),
    )
}
