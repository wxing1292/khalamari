use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use inference_runtime_core::Result;
use inference_runtime_core::config::ExecutorHibernationMode;
use inference_runtime_core::config::SchedulerConfig;
use inference_runtime_core::log_info_invalid_argument;
use inference_runtime_launcher::args::Qwen3ASRArgs;
use inference_runtime_launcher::args::Qwen3Args;
use inference_runtime_launcher::args::Qwen35Args;
use inference_runtime_launcher::args::QwenHibernationMode;
use inference_runtime_launcher::args::QwenLogLevel;
use inference_runtime_launcher::args::QwenProfileMode;
use inference_runtime_launcher::args::QwenSpecType;

use crate::telemetry::ProfileMode;
use crate::telemetry::ProfilingConfig;
use crate::telemetry::TelemetryConfig;

#[derive(Debug)]
struct SpecCheckpoint {
    model_dir: PathBuf,
    spec_type: QwenSpecType,
}

#[derive(Debug)]
pub struct Qwen3Config {
    grpc_listen_addr: SocketAddr,
    http_listen_addr: SocketAddr,
    hf_model_dir: PathBuf,
    telemetry: TelemetryConfig,
    executor_hibernation_timeout: Duration,
    executor_hibernation_mode: ExecutorHibernationMode,
    num_cache_pages: usize,
    scheduler_config: SchedulerConfig,
}

impl Qwen3Config {
    pub fn from_args(args: Qwen3Args) -> Result<Self> {
        validate_scheduler_token_capacity(args.max_tokens, args.max_tokens_per_request)?;
        if u32::try_from(args.max_requests.get()).is_err() {
            return Err(log_info_invalid_argument!(
                "--max-requests must fit the u32 request-slot domain"
            ));
        }
        if i32::try_from(args.max_tokens.get()).is_err() {
            return Err(log_info_invalid_argument!("--max-tokens must fit i32"));
        }
        if u32::try_from(args.max_tokens_per_request.get()).is_err() {
            return Err(log_info_invalid_argument!("--max-tokens-per-request must fit u32"));
        }
        if u32::try_from(args.num_cache_pages.get()).is_err() {
            return Err(log_info_invalid_argument!(
                "--num-cache-pages must fit the u32 page-ID domain"
            ));
        }

        Ok(Self {
            grpc_listen_addr: args.grpc_listen_addr,
            http_listen_addr: args.http_listen_addr,
            hf_model_dir: args.hf_model_dir,
            telemetry: TelemetryConfig {
                profiling: ProfilingConfig {
                    mode: args.profile.map(ProfileMode::from),
                    summary_every: args.profile.map_or(0, |_| 32),
                },
                debug_logging: matches!(args.logging, QwenLogLevel::Debug),
            },
            executor_hibernation_timeout: Duration::from_secs(args.executor_hibernation_timeout_secs.get()),
            executor_hibernation_mode: executor_hibernation_mode(args.executor_hibernation_mode),
            num_cache_pages: args.num_cache_pages.get(),
            scheduler_config: SchedulerConfig {
                max_requests: args.max_requests.get(),
                max_tokens: args.max_tokens.get(),
                max_tokens_per_request: args.max_tokens_per_request.get(),
                max_compute_slots: 1,
            },
        })
    }

    pub fn grpc_listen_addr(&self) -> SocketAddr {
        self.grpc_listen_addr
    }

    pub fn http_listen_addr(&self) -> SocketAddr {
        self.http_listen_addr
    }

    pub fn hf_model_dir(&self) -> &Path {
        self.hf_model_dir.as_path()
    }

    pub fn telemetry_config(&self) -> TelemetryConfig {
        self.telemetry
    }

    pub fn executor_hibernation_timeout(&self) -> Duration {
        self.executor_hibernation_timeout
    }

    pub fn executor_hibernation_mode(&self) -> ExecutorHibernationMode {
        self.executor_hibernation_mode
    }

    pub fn num_cache_pages(&self) -> usize {
        self.num_cache_pages
    }

    pub fn max_running_requests(&self) -> usize {
        self.scheduler_config.max_requests
    }

    pub fn scheduler_config(&self) -> SchedulerConfig {
        self.scheduler_config
    }
}

#[derive(Debug)]
pub struct Qwen3ASRConfig {
    grpc_listen_addr: SocketAddr,
    http_listen_addr: SocketAddr,
    hf_model_dir: PathBuf,
    telemetry: TelemetryConfig,
    executor_hibernation_timeout: Duration,
    executor_hibernation_mode: ExecutorHibernationMode,
    num_cache_pages: usize,
    scheduler_config: SchedulerConfig,
}

impl Qwen3ASRConfig {
    pub fn from_args(args: Qwen3ASRArgs) -> Result<Self> {
        validate_scheduler_token_capacity(args.max_tokens, args.max_tokens_per_request)?;
        if u32::try_from(args.max_requests.get()).is_err() {
            return Err(log_info_invalid_argument!(
                "--max-requests must fit the u32 request-slot domain"
            ));
        }
        if i32::try_from(args.max_tokens.get()).is_err() {
            return Err(log_info_invalid_argument!("--max-tokens must fit i32"));
        }
        if u32::try_from(args.max_tokens_per_request.get()).is_err() {
            return Err(log_info_invalid_argument!("--max-tokens-per-request must fit u32"));
        }
        if u32::try_from(args.num_cache_pages.get()).is_err() {
            return Err(log_info_invalid_argument!(
                "--num-cache-pages must fit the u32 page-ID domain"
            ));
        }

        Ok(Self {
            grpc_listen_addr: args.grpc_listen_addr,
            http_listen_addr: args.http_listen_addr,
            hf_model_dir: args.hf_model_dir,
            telemetry: TelemetryConfig {
                profiling: ProfilingConfig {
                    mode: args.profile.map(ProfileMode::from),
                    summary_every: args.profile.map_or(0, |_| 32),
                },
                debug_logging: matches!(args.logging, QwenLogLevel::Debug),
            },
            executor_hibernation_timeout: Duration::from_secs(args.executor_hibernation_timeout_secs.get()),
            executor_hibernation_mode: executor_hibernation_mode(args.executor_hibernation_mode),
            num_cache_pages: args.num_cache_pages.get(),
            scheduler_config: SchedulerConfig {
                max_requests: args.max_requests.get(),
                max_tokens: args.max_tokens.get(),
                max_tokens_per_request: args.max_tokens_per_request.get(),
                max_compute_slots: 1,
            },
        })
    }

    pub fn grpc_listen_addr(&self) -> SocketAddr {
        self.grpc_listen_addr
    }

    pub fn http_listen_addr(&self) -> SocketAddr {
        self.http_listen_addr
    }

    pub fn hf_model_dir(&self) -> &Path {
        self.hf_model_dir.as_path()
    }

    pub fn telemetry_config(&self) -> TelemetryConfig {
        self.telemetry
    }

    pub fn executor_hibernation_timeout(&self) -> Duration {
        self.executor_hibernation_timeout
    }

    pub fn executor_hibernation_mode(&self) -> ExecutorHibernationMode {
        self.executor_hibernation_mode
    }

    pub fn num_cache_pages(&self) -> usize {
        self.num_cache_pages
    }

    pub fn max_running_requests(&self) -> usize {
        self.scheduler_config.max_requests
    }

    pub fn scheduler_config(&self) -> SchedulerConfig {
        self.scheduler_config
    }
}

#[derive(Debug, Eq, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub enum Qwen35ModelMode {
    Vanilla,
    MTP {
        model_dir: PathBuf,
        num_spec_tokens: NonZeroUsize,
    },
    DFlash2 {
        model_dir: PathBuf,
        num_spec_tokens: Option<NonZeroUsize>,
    },
}

impl Qwen35ModelMode {
    pub fn num_cache_lanes(&self) -> usize {
        match self {
            Self::MTP { num_spec_tokens, .. } => {
                num_spec_tokens
                    .get()
                    .checked_add(1)
                    .expect("validated Qwen3.5 MTP cache-lane count must fit usize")
            },
            Self::Vanilla | Self::DFlash2 { .. } => 1,
        }
    }
}

#[derive(Debug)]
pub struct Qwen35Config {
    grpc_listen_addr: SocketAddr,
    http_listen_addr: SocketAddr,
    hf_model_dir: PathBuf,
    model_mode: Qwen35ModelMode,
    telemetry: TelemetryConfig,
    executor_hibernation_timeout: Duration,
    executor_hibernation_mode: ExecutorHibernationMode,
    num_cache_pages: usize,
    scheduler_config: SchedulerConfig,
}

impl Qwen35Config {
    pub fn from_args(args: Qwen35Args) -> Result<Self> {
        let num_cache_lanes = args.num_cache_lanes()?;
        let spec_checkpoint = normalize_spec_checkpoint(args.spec.hf_spec_model_dir, args.spec.spec_type)?;
        let num_mtp_tokens = NonZeroUsize::new(num_cache_lanes - 1);

        let model_mode = match spec_checkpoint {
            None => Qwen35ModelMode::Vanilla,
            Some(SpecCheckpoint {
                model_dir,
                spec_type: QwenSpecType::MTP,
            }) => {
                Qwen35ModelMode::MTP {
                    num_spec_tokens: num_mtp_tokens.expect("configured Qwen3.5 MTP must have a proposal length"),
                    model_dir,
                }
            },
            Some(SpecCheckpoint {
                model_dir,
                spec_type: QwenSpecType::DFlash2,
            }) => {
                Qwen35ModelMode::DFlash2 {
                    model_dir,
                    num_spec_tokens: args.spec.num_spec_tokens,
                }
            },
        };
        Ok(Self {
            grpc_listen_addr: args.grpc_listen_addr,
            http_listen_addr: args.http_listen_addr,
            hf_model_dir: args.hf_model_dir,
            model_mode,
            telemetry: TelemetryConfig {
                profiling: ProfilingConfig {
                    mode: args.profile.map(ProfileMode::from),
                    summary_every: args.profile.map_or(0, |_| 32),
                },
                debug_logging: matches!(args.logging, QwenLogLevel::Debug),
            },
            executor_hibernation_timeout: Duration::from_secs(args.executor_hibernation_timeout_secs.get()),
            executor_hibernation_mode: executor_hibernation_mode(args.executor_hibernation_mode),
            num_cache_pages: args.num_cache_pages.get(),
            scheduler_config: SchedulerConfig {
                max_requests: args.max_requests.get(),
                max_tokens: args.max_tokens.get(),
                max_tokens_per_request: args.max_tokens_per_request.get(),
                max_compute_slots: 1,
            },
        })
    }

    pub fn grpc_listen_addr(&self) -> SocketAddr {
        self.grpc_listen_addr
    }

    pub fn http_listen_addr(&self) -> SocketAddr {
        self.http_listen_addr
    }

    pub fn hf_model_dir(&self) -> &Path {
        self.hf_model_dir.as_path()
    }

    pub fn model_mode(&self) -> &Qwen35ModelMode {
        &self.model_mode
    }

    pub fn telemetry_config(&self) -> TelemetryConfig {
        self.telemetry
    }

    pub fn executor_hibernation_timeout(&self) -> Duration {
        self.executor_hibernation_timeout
    }

    pub fn executor_hibernation_mode(&self) -> ExecutorHibernationMode {
        self.executor_hibernation_mode
    }

    pub fn num_cache_lanes(&self) -> usize {
        self.model_mode.num_cache_lanes()
    }

    pub fn num_cache_pages(&self) -> usize {
        self.num_cache_pages
    }

    pub fn max_running_requests(&self) -> usize {
        self.scheduler_config.max_requests
    }

    pub fn scheduler_config(&self) -> SchedulerConfig {
        self.scheduler_config
    }
}

fn normalize_spec_checkpoint(
    hf_spec_model_dir: Option<PathBuf>,
    spec_type: Option<QwenSpecType>,
) -> Result<Option<SpecCheckpoint>> {
    match (hf_spec_model_dir, spec_type) {
        (None, None) => Ok(None),
        (Some(model_dir), Some(spec_type)) => Ok(Some(SpecCheckpoint { model_dir, spec_type })),
        _ => {
            Err(log_info_invalid_argument!(
                "--hf-spec-model-dir and --spec-type must be specified together"
            ))
        },
    }
}

fn validate_scheduler_token_capacity(max_tokens: NonZeroUsize, max_tokens_per_request: NonZeroUsize) -> Result<()> {
    if max_tokens_per_request.get() > max_tokens.get() {
        return Err(log_info_invalid_argument!(
            "--max-tokens-per-request={max_tokens_per_request} must not exceed --max-tokens={max_tokens}"
        ));
    }
    Ok(())
}

fn executor_hibernation_mode(mode: QwenHibernationMode) -> ExecutorHibernationMode {
    match mode {
        QwenHibernationMode::All => ExecutorHibernationMode::All,
        QwenHibernationMode::Selected => ExecutorHibernationMode::Selected,
    }
}

impl From<QwenProfileMode> for ProfileMode {
    fn from(value: QwenProfileMode) -> Self {
        match value {
            QwenProfileMode::Component => Self::Component,
            QwenProfileMode::Operation => Self::Operation,
        }
    }
}

#[cfg(test)]
#[path = "./config_test.rs"]
mod config_test;
