use std::sync::Arc;

use clap::Parser;
use inference_executor_core::model::ReplayableDecoderModel;
use inference_executor_metal::model::qwen::v4_x::Qwen4xExecutorConfig;
use inference_executor_metal::model::qwen::v4_x::init_qwen4x_model;
use inference_executor_metal::model::qwen::v4_x::init_qwen4x_model_with_mtp;
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
use crate::qwen_server::config::Qwen35ModelMode;
use crate::qwen_server::sizing::block_cache_capacity;
use crate::qwen_server::sizing::context_window;
use crate::rpc::RPCService;
use crate::runtime::serve_replay_model;

const TOKENS_PER_CACHE_BLOCK: usize = 2048;

pub fn run<const L: usize>() {
    if let Err(error) = run_inner::<L>() {
        eprintln!("unable to start qwen4x: {error}");
        std::process::exit(1);
    }
}
fn run_inner<const L: usize>() -> Result<()> {
    let args = Qwen35Args::parse();
    // The shared hybrid-server CLI owns scheduler, page, transport, and telemetry options.
    let config = Qwen35Config::from_args(args)?;
    assert_eq!(
        L,
        config.num_cache_lanes(),
        "Qwen4x worker specialization must match the configured cache lanes"
    );
    config.telemetry_config().init();
    let codec = Arc::new(
        QwenCodec::load(config.hf_model_dir())
            .map_err(|error| log_err_unavailable!("unable to load Qwen4x codec: {error}"))?,
    );
    let scheduler = config.scheduler_config();
    let executor_config = Qwen4xExecutorConfig {
        max_requests: config.max_running_requests(),
        max_tokens: scheduler.max_tokens,
        max_tokens_per_request: scheduler.max_tokens_per_request,
        num_cache_pages: config.num_cache_pages(),
        num_tokens_per_block: TOKENS_PER_CACHE_BLOCK,
    };
    let model = match config.model_mode() {
        Qwen35ModelMode::Vanilla => init_qwen4x_model(config.hf_model_dir(), executor_config),
        Qwen35ModelMode::MTP {
            model_dir,
            num_spec_tokens,
        } => init_qwen4x_model_with_mtp(config.hf_model_dir(), model_dir, *num_spec_tokens, executor_config),
        Qwen35ModelMode::DFlash2 { .. } => {
            return Err(log_info_invalid_argument!("Qwen4x does not support DFlash2 yet"));
        },
    }
    .map_err(|error| log_err_unavailable!("unable to initialize Qwen4x: {error}"))?;
    let mut lane_pages = vec![(model.num_kv_page_ids_per_block(), model.num_state_page_ids_per_block())];
    lane_pages.extend(model.num_mtp_page_ids_per_block());
    let (kv, state) = lane_pages.iter().try_fold((0usize, 0usize), |(kv, state), &(k, s)| {
        Ok::<_, inference_runtime_core::Error>((
            kv.checked_add(k)
                .ok_or_else(|| log_info_invalid_argument!("Qwen4x KV page count overflow"))?,
            state
                .checked_add(s)
                .ok_or_else(|| log_info_invalid_argument!("Qwen4x state page count overflow"))?,
        ))
    })?;
    let capacity = block_cache_capacity(config.num_cache_pages(), kv, state)?;
    let cache_lanes = lane_pages
        .into_iter()
        .map(|(num_pages_per_kv_block, num_pages_per_state_block)| {
            CacheLaneRuntimeConfig {
                num_pages_per_kv_block,
                num_pages_per_state_block,
                block_cache_capacity: capacity,
            }
        })
        .collect();
    let runtime = RuntimeConfig {
        max_running_requests: config.max_running_requests(),
        executor_hibernation_timeout: config.executor_hibernation_timeout(),
        executor_hibernation_mode: config.executor_hibernation_mode(),
        context_window: context_window(model.model_config().text_config.max_position_embeddings, 0)?,
        num_tokens_per_cache_block: TOKENS_PER_CACHE_BLOCK,
        num_pages: config.num_cache_pages(),
        cache_lanes,
    };
    tracing::info!(
        model_mode = model.model_mode(),
        cache_block_tokens = TOKENS_PER_CACHE_BLOCK,
        num_kv_pages_per_block = kv,
        num_state_pages_per_block = state,
        max_requests = scheduler.max_requests,
        max_tokens = scheduler.max_tokens,
        "Qwen4x configured"
    );
    serve_replay_model::<TOKENS_PER_CACHE_BLOCK, L, _>(
        config.grpc_listen_addr(),
        config.http_listen_addr(),
        RPCService::Messages(codec),
        Arc::new(ResourceProcessors::new()),
        runtime,
        scheduler,
        ReplayableModelExecutors::new(model, vec![]),
    )
}
