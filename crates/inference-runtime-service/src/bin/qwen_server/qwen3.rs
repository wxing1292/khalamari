use std::sync::Arc;

use clap::Parser;
use inference_executor_core::model::ReplayableDecoderModel;
use inference_executor_metal::model::qwen::v3::executor::Qwen3Executor;
use inference_executor_metal::model::qwen::v3::executor::Qwen3ExecutorConfig;
use inference_executor_metal::model::qwen::v3::executor::init_qwen_3_model;
use inference_runtime_core::Result;
use inference_runtime_core::config::CacheLaneRuntimeConfig;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::log_err_internal;
use inference_runtime_core::log_err_unavailable;
use inference_runtime_core::runtime::resource::processor::ResourceProcessors;
use inference_runtime_launcher::args::Qwen3Args;

use crate::codec::qwen::QwenCodec;
use crate::executor::ReplayableModelExecutors;
use crate::qwen_server::config::Qwen3Config;
use crate::qwen_server::sizing::block_cache_capacity;
use crate::qwen_server::sizing::context_window;
use crate::rpc::RPCService;
use crate::runtime::serve_replay_model;

// Qwen3 has no GDN snapshot to amortize across a large logical block. One
// 16-token physical KV page per layer keeps trie granularity small.
const TOKENS_PER_CACHE_BLOCK: usize = 16;

pub fn run() {
    if let Err(error) = run_inner() {
        eprintln!("unable to start qwen3: {error}");
        std::process::exit(1);
    }
}

fn run_inner() -> Result<()> {
    let args = Qwen3Args::parse();
    let config = Qwen3Config::from_args(args)?;
    config.telemetry_config().init();
    let startup_span = tracing::info_span!("qwen-startup", model = "qwen3").entered();

    tracing::info!("loading Qwen codec");
    let qwen_codec = Arc::new(QwenCodec::load(config.hf_model_dir()).map_err(|error| {
        log_err_unavailable!("unable to load Qwen codec from {:?}: {error}", config.hf_model_dir())
    })?);
    tracing::info!("Qwen codec loaded");

    let scheduler_config = config.scheduler_config();
    let num_cache_pages = config.num_cache_pages();
    let max_running_requests = config.max_running_requests();
    tracing::info!("initializing model executor");
    let executor_config = Qwen3ExecutorConfig {
        max_requests: max_running_requests,
        max_tokens: scheduler_config.max_tokens,
        max_tokens_per_request: scheduler_config.max_tokens_per_request,
        num_cache_pages,
        num_tokens_per_block: TOKENS_PER_CACHE_BLOCK,
    };
    let model = init_qwen_3_model(config.hf_model_dir(), executor_config).map_err(|error| {
        log_err_internal!(
            "unable to initialize qwen3 Main model from {:?}: {error}",
            config.hf_model_dir()
        )
    })?;
    tracing::info!("model executor initialized");

    let runtime_config = build_runtime_config(&config, &model)?;
    let num_spec_tokens = model.num_spec_tokens();
    for cache_lane in 0..runtime_config.num_cache_lanes() {
        let lane = runtime_config.cache_lane(cache_lane);
        tracing::info!(
            cache_lane,
            mtp = cache_lane > 0,
            num_kv_pages_per_block = lane.num_pages_per_kv_block,
            num_state_pages_per_block = lane.num_pages_per_state_block,
            block_cache_capacity = lane.block_cache_capacity,
            "cache lane configured"
        );
    }

    tracing::info!(
        model_mode = model.model_mode(),
        num_spec_tokens,
        grpc_listen_addr = %config.grpc_listen_addr(),
        http_listen_addr = %config.http_listen_addr(),
        num_cache_pages,
        cache_block_tokens = TOKENS_PER_CACHE_BLOCK,
        max_running_requests,
        context_window = runtime_config.context_window,
        max_batch_requests = scheduler_config.max_requests,
        max_tokens = scheduler_config.max_tokens,
        max_tokens_per_request = scheduler_config.max_tokens_per_request,
        executor_hibernation_timeout_secs = config.executor_hibernation_timeout().as_secs(),
        executor_hibernation_mode = ?config.executor_hibernation_mode(),
        "configured"
    );
    drop(startup_span);

    serve_replay_model::<TOKENS_PER_CACHE_BLOCK, 1, _>(
        config.grpc_listen_addr(),
        config.http_listen_addr(),
        RPCService::Messages(qwen_codec),
        Arc::new(ResourceProcessors::new()),
        runtime_config,
        scheduler_config,
        ReplayableModelExecutors::new(model, vec![]),
    )
}

fn build_runtime_config(service_config: &Qwen3Config, model: &Qwen3Executor) -> Result<RuntimeConfig> {
    let num_cache_pages = service_config.num_cache_pages();
    let model_config = model.model_config();
    let text = &model_config.text_config;
    let num_pages_per_kv_block = model.num_kv_page_ids_per_block();
    let cache_lane = main_cache_lane(num_cache_pages, num_pages_per_kv_block)?;
    Ok(RuntimeConfig {
        max_running_requests: service_config.max_running_requests(),
        executor_hibernation_timeout: service_config.executor_hibernation_timeout(),
        executor_hibernation_mode: service_config.executor_hibernation_mode(),
        context_window: context_window(text.max_position_embeddings, 0)?,
        num_tokens_per_cache_block: TOKENS_PER_CACHE_BLOCK,
        num_pages: num_cache_pages,
        cache_lanes: vec![cache_lane],
    })
}

fn main_cache_lane(num_cache_pages: usize, num_pages_per_kv_block: usize) -> Result<CacheLaneRuntimeConfig> {
    Ok(CacheLaneRuntimeConfig {
        num_pages_per_kv_block,
        num_pages_per_state_block: 0,
        block_cache_capacity: block_cache_capacity(num_cache_pages, num_pages_per_kv_block, 0)?,
    })
}

#[cfg(test)]
mod tests {
    use inference_runtime_core::Error;

    use super::TOKENS_PER_CACHE_BLOCK;
    use super::main_cache_lane;

    #[test]
    fn test_qwen3_uses_small_gqa_logical_blocks() {
        assert_eq!(TOKENS_PER_CACHE_BLOCK, 16);
    }

    #[test]
    fn test_main_cache_lane_uses_no_state_pages() {
        let lane = main_cache_lane(40 * 1024, 40).unwrap();

        assert_eq!(lane.num_pages_per_kv_block, 40);
        assert_eq!(lane.num_pages_per_state_block, 0);
        assert_eq!(lane.block_cache_capacity, 1024);
    }

    #[test]
    fn test_main_cache_lane_reports_dynamic_minimum_pages() {
        assert!(matches!(
            main_cache_lane(39, 40),
            Err(Error::InvalidArgument(message))
                if message.contains("--num-cache-pages=39")
                    && message.contains("requiring 40 pages")
        ));
    }
}
