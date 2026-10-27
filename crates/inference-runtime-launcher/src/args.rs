use std::net::SocketAddr;
use std::num::NonZeroU64;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use clap::Args;
use clap::Parser;
use clap::ValueEnum;
use inference_error::Result;
use inference_error::log_info_invalid_argument;

#[derive(Args, Debug)]
pub struct QwenSpecArgs {
    #[arg(
        long,
        value_name = "DIR",
        help = "Speculative checkpoint directory (requires --spec-type)"
    )]
    pub hf_spec_model_dir: Option<PathBuf>,

    #[arg(
        long,
        value_enum,
        help = "Speculative checkpoint type (requires --hf-spec-model-dir)"
    )]
    pub spec_type: Option<QwenSpecType>,

    #[arg(
        long,
        help = "Speculative tokens generated per proposal; defaults to 1 for MTP or the checkpoint block geometry for \
                DFlash2"
    )]
    pub num_spec_tokens: Option<NonZeroUsize>,
}

#[derive(Debug, Parser)]
pub struct Qwen3Args {
    #[arg(long, default_value = "127.0.0.1:50051")]
    pub grpc_listen_addr: SocketAddr,

    #[arg(long, default_value = "127.0.0.1:8000")]
    pub http_listen_addr: SocketAddr,

    #[arg(long, value_name = "DIR")]
    pub hf_model_dir: PathBuf,

    #[arg(long, value_enum)]
    pub profile: Option<QwenProfileMode>,

    #[arg(long, value_enum, default_value_t = QwenLogLevel::Info)]
    pub logging: QwenLogLevel,

    #[arg(
        long,
        default_value = "300",
        help = "Seconds without model execution before state and weights unload"
    )]
    pub executor_hibernation_timeout_secs: NonZeroU64,

    #[arg(
        long,
        default_value = "selected",
        value_name = "MODE",
        help = "Executor hibernation state scope: all or selected"
    )]
    pub executor_hibernation_mode: QwenHibernationMode,

    #[arg(
        long,
        default_value = "393216",
        help = "Total shared cache pages used by GQA KV cache and GDN state cache"
    )]
    pub num_cache_pages: NonZeroUsize,

    #[arg(
        long,
        default_value = "2",
        help = "Maximum running requests and requests scheduled per batch"
    )]
    pub max_requests: NonZeroUsize,

    #[arg(long, default_value = "128", help = "Maximum flattened tokens scheduled per batch")]
    pub max_tokens: NonZeroUsize,

    #[arg(
        long,
        default_value = "64",
        help = "Maximum tokens from one request in one forward transaction"
    )]
    pub max_tokens_per_request: NonZeroUsize,
}

#[derive(Debug, Parser)]
pub struct Qwen3ASRArgs {
    #[arg(long, default_value = "127.0.0.1:50051")]
    pub grpc_listen_addr: SocketAddr,

    #[arg(long, default_value = "127.0.0.1:8000")]
    pub http_listen_addr: SocketAddr,

    #[arg(long, value_name = "DIR")]
    pub hf_model_dir: PathBuf,

    #[arg(long, value_enum)]
    pub profile: Option<QwenProfileMode>,

    #[arg(long, value_enum, default_value_t = QwenLogLevel::Info)]
    pub logging: QwenLogLevel,

    #[arg(
        long,
        default_value = "300",
        help = "Seconds without model execution before state and weights unload"
    )]
    pub executor_hibernation_timeout_secs: NonZeroU64,

    #[arg(
        long,
        default_value = "selected",
        value_name = "MODE",
        help = "Executor hibernation state scope: all or selected"
    )]
    pub executor_hibernation_mode: QwenHibernationMode,

    #[arg(long, default_value = "8192", help = "Total shared Qwen3-ASR KV-cache pages")]
    pub num_cache_pages: NonZeroUsize,

    #[arg(
        long,
        default_value = "8",
        help = "Maximum running requests and requests scheduled per batch"
    )]
    pub max_requests: NonZeroUsize,

    #[arg(long, default_value = "256", help = "Maximum flattened tokens scheduled per batch")]
    pub max_tokens: NonZeroUsize,

    #[arg(
        long,
        default_value = "64",
        help = "Maximum tokens from one request in one forward transaction"
    )]
    pub max_tokens_per_request: NonZeroUsize,
}

#[derive(Debug, Parser)]
pub struct Qwen35Args {
    #[arg(long, default_value = "127.0.0.1:50051")]
    pub grpc_listen_addr: SocketAddr,

    #[arg(long, default_value = "127.0.0.1:8000")]
    pub http_listen_addr: SocketAddr,

    #[arg(long, value_name = "DIR")]
    pub hf_model_dir: PathBuf,

    #[command(flatten)]
    pub spec: QwenSpecArgs,

    #[arg(long, value_enum)]
    pub profile: Option<QwenProfileMode>,

    #[arg(long, value_enum, default_value_t = QwenLogLevel::Info)]
    pub logging: QwenLogLevel,

    #[arg(
        long,
        default_value = "300",
        help = "Seconds without model execution before state and weights unload"
    )]
    pub executor_hibernation_timeout_secs: NonZeroU64,

    #[arg(
        long,
        default_value = "selected",
        value_name = "MODE",
        help = "Executor hibernation state scope: all or selected"
    )]
    pub executor_hibernation_mode: QwenHibernationMode,

    #[arg(
        long,
        default_value = "393216",
        help = "Total shared cache pages used by GQA KV cache and GDN state cache"
    )]
    pub num_cache_pages: NonZeroUsize,

    #[arg(
        long,
        default_value = "2",
        help = "Maximum running requests and requests scheduled per batch"
    )]
    pub max_requests: NonZeroUsize,

    #[arg(long, default_value = "128", help = "Maximum flattened tokens scheduled per batch")]
    pub max_tokens: NonZeroUsize,

    #[arg(
        long,
        default_value = "64",
        help = "Maximum tokens from one request in one forward transaction"
    )]
    pub max_tokens_per_request: NonZeroUsize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum QwenHibernationMode {
    All,
    Selected,
}

impl Qwen35Args {
    /// Validates the CLI capacities and returns the runtime cache-lane count.
    pub fn num_cache_lanes(&self) -> Result<usize> {
        if self.spec.hf_spec_model_dir.is_some() != self.spec.spec_type.is_some() {
            return Err(log_info_invalid_argument!(
                "--hf-spec-model-dir and --spec-type must be specified together"
            ));
        }
        if self.spec.num_spec_tokens.is_some() && self.spec.spec_type.is_none() {
            return Err(log_info_invalid_argument!("--num-spec-tokens requires --spec-type"));
        }
        if self.max_tokens_per_request > self.max_tokens {
            return Err(log_info_invalid_argument!(
                "--max-tokens-per-request={} must not exceed --max-tokens={}",
                self.max_tokens_per_request,
                self.max_tokens
            ));
        }
        if u32::try_from(self.max_requests.get()).is_err() {
            return Err(log_info_invalid_argument!(
                "--max-requests must fit the u32 request-slot domain"
            ));
        }
        if i32::try_from(self.max_tokens.get()).is_err() {
            return Err(log_info_invalid_argument!("--max-tokens must fit i32"));
        }
        if u32::try_from(self.max_tokens_per_request.get()).is_err() {
            return Err(log_info_invalid_argument!("--max-tokens-per-request must fit u32"));
        }
        if u32::try_from(self.num_cache_pages.get()).is_err() {
            return Err(log_info_invalid_argument!(
                "--num-cache-pages must fit the u32 page-ID domain"
            ));
        }
        if self.spec.spec_type != Some(QwenSpecType::MTP) {
            return Ok(1);
        }
        let tokens = self.spec.num_spec_tokens.unwrap_or(NonZeroUsize::MIN);
        if tokens > self.max_tokens_per_request {
            return Err(log_info_invalid_argument!(
                "--max-tokens-per-request={} must be at least --num-spec-tokens={tokens} for MTP cache-lane \
                 initialization",
                self.max_tokens_per_request
            ));
        }
        tokens
            .get()
            .checked_add(1)
            .ok_or_else(|| log_info_invalid_argument!("MTP cache-lane count must fit usize"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum QwenProfileMode {
    Component,
    Operation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[allow(clippy::upper_case_acronyms)]
pub enum QwenSpecType {
    #[value(name = "mtp")]
    MTP,
    #[value(name = "dflash2")]
    DFlash2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum QwenLogLevel {
    Info,
    Debug,
}
