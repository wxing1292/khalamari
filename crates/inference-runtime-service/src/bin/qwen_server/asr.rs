use std::sync::Arc;

use clap::Parser;
use inference_executor_core::model::EncoderExecutorLifecycle;
use inference_executor_core::model::ReplayableDecoderModel;
use inference_executor_core::model::qwen::v3_asr::QWEN3_ASR_AUDIO_RESOURCE_TYPE;
use inference_executor_metal::model::qwen::v3::executor::Qwen3Executor;
use inference_executor_metal::model::qwen::v3::executor::Qwen3ExecutorConfig;
use inference_executor_metal::model::qwen::v3::executor::init_qwen3_asr_model;
use inference_runtime_core::Result;
use inference_runtime_core::config::CacheLaneRuntimeConfig;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::log_err_internal;
use inference_runtime_core::runtime::resource::processor::ResourceProcessors;
use inference_runtime_launcher::args::Qwen3ASRArgs;

use crate::asr::Qwen3ASRService;
use crate::executor::ReplayableModelExecutors;
use crate::qwen_server::config::Qwen3ASRConfig;
use crate::qwen_server::sizing::block_cache_capacity;
use crate::qwen_server::sizing::context_window;
use crate::rpc::RPCService;
use crate::runtime::serve_replay_model;

const TOKENS_PER_CACHE_BLOCK: usize = 16;

pub fn run() {
    if let Err(error) = run_inner() {
        eprintln!("unable to start qwen3-asr: {error}");
        std::process::exit(1);
    }
}

fn run_inner() -> Result<()> {
    let config = Qwen3ASRConfig::from_args(Qwen3ASRArgs::parse())?;
    config.telemetry_config().init();
    let startup_span = tracing::info_span!("qwen-startup", model = "qwen3-asr").entered();

    let scheduler_config = config.scheduler_config();
    let num_cache_pages = config.num_cache_pages();
    let max_running_requests = config.max_running_requests();
    tracing::info!("initializing model executor");
    let loaded = init_qwen3_asr_model(
        config.hf_model_dir(),
        Qwen3ExecutorConfig {
            max_requests: max_running_requests,
            max_tokens: scheduler_config.max_tokens,
            max_tokens_per_request: scheduler_config.max_tokens_per_request,
            num_cache_pages,
            num_tokens_per_block: TOKENS_PER_CACHE_BLOCK,
        },
    )
    .map_err(|error| {
        log_err_internal!(
            "unable to initialize Qwen3-ASR model from {:?}: {error}",
            config.hf_model_dir()
        )
    })?;
    tracing::info!("model executor initialized");

    let runtime_config = build_runtime_config(&config, &loaded.executor)?;
    let asr = Arc::new(Qwen3ASRService::load(
        config.hf_model_dir(),
        loaded.executor.asr_model_config().clone(),
        Arc::clone(&loaded.audio_processor),
    )?);
    let mut resource_processors = ResourceProcessors::new();
    let encoder_executors: Vec<Arc<dyn EncoderExecutorLifecycle>> = vec![loaded.audio_processor.encoder_executor()];
    resource_processors.register(QWEN3_ASR_AUDIO_RESOURCE_TYPE, loaded.audio_processor);
    let cache_lane = runtime_config.cache_lane(0);
    tracing::info!(
        model_mode = loaded.executor.model_mode(),
        grpc_listen_addr = %config.grpc_listen_addr(),
        http_listen_addr = %config.http_listen_addr(),
        num_cache_pages,
        cache_block_tokens = TOKENS_PER_CACHE_BLOCK,
        num_kv_pages_per_block = cache_lane.num_pages_per_kv_block,
        block_cache_capacity = cache_lane.block_cache_capacity,
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
        RPCService::Transcriptions(asr),
        Arc::new(resource_processors),
        runtime_config,
        scheduler_config,
        ReplayableModelExecutors::new(loaded.executor, encoder_executors),
    )
}

fn build_runtime_config(service_config: &Qwen3ASRConfig, model: &Qwen3Executor) -> Result<RuntimeConfig> {
    let num_cache_pages = service_config.num_cache_pages();
    let text = &model.asr_model_config().text;
    let num_pages_per_kv_block = model.num_kv_page_ids_per_block();
    Ok(RuntimeConfig {
        max_running_requests: service_config.max_running_requests(),
        executor_hibernation_timeout: service_config.executor_hibernation_timeout(),
        executor_hibernation_mode: service_config.executor_hibernation_mode(),
        context_window: context_window(text.max_position_embeddings, 0)?,
        num_tokens_per_cache_block: TOKENS_PER_CACHE_BLOCK,
        num_pages: num_cache_pages,
        cache_lanes: vec![CacheLaneRuntimeConfig {
            num_pages_per_kv_block,
            num_pages_per_state_block: 0,
            block_cache_capacity: block_cache_capacity(num_cache_pages, num_pages_per_kv_block, 0)?,
        }],
    })
}
