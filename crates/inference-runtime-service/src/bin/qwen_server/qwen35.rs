use std::sync::Arc;

use clap::Parser;
use inference_executor_core::model::ReplayableDecoderModel;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::init_qwen35_model_config;
use inference_executor_core::model::qwen::v3_x::dflash2::init_qwen3x_dflash2_config;
use inference_executor_metal::model::qwen::v3_5::executor::Qwen35Executor;
use inference_executor_metal::model::qwen::v3_5::executor::Qwen35ExecutorConfig;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model_with_dflash2;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model_with_mtp;
use inference_runtime_core::Result;
use inference_runtime_core::config::CacheLaneRuntimeConfig;
use inference_runtime_core::config::RuntimeConfig;
use inference_runtime_core::log_err_internal;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelKind {
    Dense,
    Sparse,
}

impl ModelKind {
    fn label(self) -> &'static str {
        match self {
            ModelKind::Dense => "qwen3.5 dense 27B",
            ModelKind::Sparse => "qwen3.5 sparse 35B-A3B",
        }
    }
}

pub fn run_dense<const L: usize>() {
    run_worker::<L>(ModelKind::Dense, Qwen35Args::parse());
}

pub fn run_sparse<const L: usize>() {
    run_worker::<L>(ModelKind::Sparse, Qwen35Args::parse());
}

fn run_worker<const L: usize>(kind: ModelKind, args: Qwen35Args) {
    if let Err(error) = run_service::<L>(kind, args) {
        eprintln!("unable to start {}: {error}", kind.label());
        std::process::exit(1);
    }
}

fn run_service<const L: usize>(kind: ModelKind, args: Qwen35Args) -> Result<()> {
    let config = Qwen35Config::from_args(args)?;
    assert!(L > 0, "qwen3.5 worker requires at least the Main cache lane");
    assert_eq!(
        L,
        config.num_cache_lanes(),
        "qwen3.5 worker const L must equal the configured cache-lane count"
    );
    config.telemetry_config().init();
    let startup_span = tracing::info_span!("qwen-startup", model = kind.label()).entered();

    let model_config = load_model_config(config.hf_model_dir())?;
    validate_checkpoint_kind(kind, &model_config)?;
    tracing::info!("loading Qwen codec");
    let qwen_codec = Arc::new(QwenCodec::load(config.hf_model_dir()).map_err(|error| {
        log_err_unavailable!("unable to load Qwen codec from {:?}: {error}", config.hf_model_dir())
    })?);
    tracing::info!("Qwen codec loaded");
    let scheduler_config = config.scheduler_config();
    let num_cache_pages = config.num_cache_pages();
    let max_running_requests = config.max_running_requests();

    tracing::info!("initializing model executor");
    let model = build_model(
        kind,
        &config,
        Qwen35ExecutorConfig {
            max_requests: max_running_requests,
            max_tokens: scheduler_config.max_tokens,
            max_tokens_per_request: scheduler_config.max_tokens_per_request,
            num_cache_pages,
            num_tokens_per_block: TOKENS_PER_CACHE_BLOCK,
        },
    )?;
    tracing::info!("model executor initialized");

    let runtime_config = build_runtime_config(&config, &model_config, &model)?;
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
        checkpoint_mtp_layers = model_config.text_config.mtp_num_hidden_layers,
        num_spec_tokens,
        model_mode = model.model_mode(),
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

    serve_replay_model::<TOKENS_PER_CACHE_BLOCK, L, _>(
        config.grpc_listen_addr(),
        config.http_listen_addr(),
        RPCService::Messages(qwen_codec),
        Arc::new(ResourceProcessors::new()),
        runtime_config,
        scheduler_config,
        ReplayableModelExecutors::new(model, vec![]),
    )
}

fn validate_checkpoint_kind(kind: ModelKind, model_config: &Qwen35ModelConfig) -> Result<()> {
    let num_experts = model_config.text_config.num_experts;
    match kind {
        ModelKind::Dense if num_experts != 0 => {
            Err(log_info_invalid_argument!(
                "qwen3_5_dense expects a dense checkpoint, got num_experts={num_experts}"
            ))
        },
        ModelKind::Sparse if num_experts == 0 => {
            Err(log_info_invalid_argument!(
                "qwen3_5_sparse expects a sparse checkpoint, got num_experts=0"
            ))
        },
        _ => Ok(()),
    }
}

fn all_lane_block_cache_capacity(
    num_cache_pages: usize,
    num_gqa_pages_per_main_block: usize,
    mtp_gqa_page_ids_per_block: &[usize],
    num_gdn_pages_per_main_block: usize,
) -> Result<usize> {
    let num_gqa_pages_per_cached_block = mtp_gqa_page_ids_per_block.iter().try_fold(
        u64::try_from(num_gqa_pages_per_main_block)
            .map_err(|_| log_err_internal!("qwen3.5 main GQA page count must fit u64"))?,
        |total, &num_pages| {
            total
                .checked_add(
                    u64::try_from(num_pages)
                        .map_err(|_| log_err_internal!("qwen3.5 MTP GQA page count must fit u64"))?,
                )
                .ok_or_else(|| log_err_internal!("qwen3.5 all-lane GQA page count overflow"))
        },
    )?;
    block_cache_capacity(
        num_cache_pages,
        usize::try_from(num_gqa_pages_per_cached_block)
            .map_err(|_| log_err_internal!("qwen3.5 all-lane GQA page count must fit usize"))?,
        num_gdn_pages_per_main_block,
    )
}

fn build_model(
    kind: ModelKind,
    config: &Qwen35Config,
    executor_config: Qwen35ExecutorConfig,
) -> Result<Qwen35Executor> {
    let hf_model_dir = config.hf_model_dir();
    let init_result = (|| {
        match config.model_mode() {
            Qwen35ModelMode::Vanilla => init_qwen_3_5_model(hf_model_dir, executor_config),
            Qwen35ModelMode::MTP {
                model_dir,
                num_spec_tokens,
            } => init_qwen_3_5_model_with_mtp(hf_model_dir, model_dir, *num_spec_tokens, executor_config),
            Qwen35ModelMode::DFlash2 {
                model_dir,
                num_spec_tokens,
            } => {
                let num_spec_tokens = match num_spec_tokens {
                    Some(count) => *count,
                    None => init_qwen3x_dflash2_config(model_dir)?.num_spec_tokens(),
                };
                init_qwen_3_5_model_with_dflash2(hf_model_dir, model_dir, num_spec_tokens, executor_config)
            },
        }
    })();
    init_result.map_err(|error| {
        log_err_internal!(
            "unable to initialize {} model from {hf_model_dir:?} in mode {:?}: {error}",
            kind.label(),
            config.model_mode(),
        )
    })
}

fn build_runtime_config(
    service_config: &Qwen35Config,
    model_config: &Qwen35ModelConfig,
    model: &Qwen35Executor,
) -> Result<RuntimeConfig> {
    let num_cache_pages = service_config.num_cache_pages();
    let text = &model_config.text_config;
    let num_gqa_pages_per_main_block = model.num_main_lane_gqa_page_ids_per_block();
    let num_gdn_pages_per_main_block = model.num_gdn_state_page_ids_per_block();
    let mtp_gqa_page_ids_per_block = model.num_mtp_gqa_page_ids_per_block();
    let block_cache_capacity = all_lane_block_cache_capacity(
        num_cache_pages,
        num_gqa_pages_per_main_block,
        &mtp_gqa_page_ids_per_block,
        num_gdn_pages_per_main_block,
    )?;
    let mut cache_lanes = vec![CacheLaneRuntimeConfig {
        num_pages_per_kv_block: num_gqa_pages_per_main_block,
        num_pages_per_state_block: num_gdn_pages_per_main_block,
        block_cache_capacity,
    }];
    for num_pages_per_kv_block in mtp_gqa_page_ids_per_block {
        cache_lanes.push(CacheLaneRuntimeConfig {
            num_pages_per_kv_block,
            num_pages_per_state_block: 0,
            block_cache_capacity,
        });
    }
    let block_spec_num_spec_tokens = match service_config.model_mode() {
        Qwen35ModelMode::DFlash2 { .. } => model.num_spec_tokens(),
        Qwen35ModelMode::Vanilla | Qwen35ModelMode::MTP { .. } => 0,
    };
    let runtime_config = RuntimeConfig {
        max_running_requests: service_config.max_running_requests(),
        executor_hibernation_timeout: service_config.executor_hibernation_timeout(),
        executor_hibernation_mode: service_config.executor_hibernation_mode(),
        context_window: context_window(text.max_position_embeddings, block_spec_num_spec_tokens)?,
        num_tokens_per_cache_block: TOKENS_PER_CACHE_BLOCK,
        num_pages: num_cache_pages,
        cache_lanes,
    };
    Ok(runtime_config)
}

fn load_model_config(hf_model_dir: &std::path::Path) -> Result<Qwen35ModelConfig> {
    init_qwen35_model_config(hf_model_dir)
        .map_err(|error| log_err_unavailable!("unable to read qwen3.5 model config from {hf_model_dir:?}: {error}"))
}

#[cfg(test)]
mod tests {
    use inference_runtime_core::Error;

    use super::all_lane_block_cache_capacity;

    #[test]
    fn test_all_lane_capacity_rejects_incomplete_block() {
        assert!(matches!(
            all_lane_block_cache_capacity(9151, 4096, &[256], 4800),
            Err(Error::InvalidArgument(message)) if message.contains("requiring 9152 pages")
        ));
    }

    #[test]
    fn test_all_lane_capacity_accepts_exact_block() {
        assert_eq!(all_lane_block_cache_capacity(9152, 4096, &[256], 4800).unwrap(), 1);
    }

    #[test]
    fn test_all_lane_capacity_counts_complete_blocks() {
        assert_eq!(all_lane_block_cache_capacity(29, 3, &[2, 1], 4).unwrap(), 2);
        assert_eq!(all_lane_block_cache_capacity(20, 10, &[], 0).unwrap(), 2);
    }
}
