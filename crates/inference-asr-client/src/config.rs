use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use clap::ArgMatches;
use clap::Args as ClapArgs;
use clap::CommandFactory;
use clap::FromArgMatches;
use clap::Parser;
use clap::parser::ValueSource;
use inference_error::Error;
use inference_error::Result;
use serde::Deserialize;
use serde::Serialize;

use crate::audio::SAMPLE_RATE;

#[derive(Clone, Debug, Default, Deserialize, Serialize, ClapArgs)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    #[command(flatten)]
    pub capture: CaptureConfig,
    #[command(flatten)]
    pub service: ServiceConfig,
    #[command(flatten)]
    pub streaming: StreamConfig,
    #[command(flatten)]
    pub language: LanguageConfig,
    #[command(flatten)]
    pub output: OutputConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize, ClapArgs)]
#[serde(default, deny_unknown_fields)]
pub struct CaptureConfig {
    #[arg(id = "capture.device", long = "device", default_value_t = Self::default().device)]
    pub device: String,
    #[arg(id = "capture.input", long = "input")]
    pub input: Option<PathBuf>,
    #[arg(id = "capture.duration", long = "duration")]
    pub duration: Option<f64>,
    #[arg(id = "capture.ffmpeg", long = "ffmpeg", default_value_t = Self::default().ffmpeg)]
    pub ffmpeg: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ClapArgs)]
#[serde(default, deny_unknown_fields)]
pub struct ServiceConfig {
    #[arg(id = "service.transport", long = "transport", value_enum, default_value_t = Self::default().transport)]
    pub transport: Transport,
    #[arg(id = "service.url", long = "url", default_value_t = Self::default().url)]
    pub url: String,
    #[arg(id = "service.model", long = "model", default_value_t = Self::default().model)]
    pub model: String,
    #[arg(id = "service.language", long = "language")]
    pub language: Option<String>,
    #[arg(id = "service.context", long = "context")]
    pub context: Option<String>,
    #[arg(id = "service.timeout", long = "timeout", default_value_t = Self::default().timeout)]
    pub timeout: f64,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    #[default]
    Http,
    Grpc,
}

impl std::fmt::Display for Transport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Http => "http",
            Self::Grpc => "grpc",
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ClapArgs)]
#[serde(default, deny_unknown_fields)]
pub struct StreamConfig {
    #[arg(id = "streaming.enabled", long = "stream", conflicts_with = "no_stream")]
    pub enabled: bool,
    #[arg(id = "streaming.window_seconds", long = "window-seconds", default_value_t = Self::default().window_seconds)]
    pub window_seconds: f64,
    #[arg(id = "streaming.interval", long = "interval", default_value_t = Self::default().interval)]
    pub interval: f64,
    #[arg(id = "streaming.silence_seconds", long = "silence-seconds", default_value_t = Self::default().silence_seconds)]
    pub silence_seconds: f64,
    #[arg(id = "streaming.sentence_silence_seconds", long = "sentence-silence-seconds", default_value_t = Self::default().sentence_silence_seconds)]
    pub sentence_silence_seconds: f64,
    #[arg(id = "streaming.lookback_seconds", long = "lookback-seconds", default_value_t = Self::default().lookback_seconds)]
    pub lookback_seconds: f64,
    #[arg(id = "streaming.overlap_seconds", long = "overlap-seconds", default_value_t = Self::default().overlap_seconds)]
    pub overlap_seconds: f64,
    #[arg(id = "streaming.speech_threshold", long = "speech-threshold", default_value_t = Self::default().speech_threshold)]
    pub speech_threshold: f32,
    #[arg(id = "streaming.padding_seconds", long = "padding-seconds", default_value_t = Self::default().padding_seconds)]
    pub padding_seconds: f64,
    #[arg(id = "streaming.min_partial_seconds", long = "min-partial-seconds", default_value_t = Self::default().min_partial_seconds)]
    pub min_partial_seconds: f64,
    #[arg(id = "streaming.max_in_flight", long = "max-in-flight", default_value_t = Self::default().max_in_flight)]
    pub max_in_flight: usize,
    #[arg(id = "streaming.max_pending_finals", long = "max-pending-finals", default_value_t = Self::default().max_pending_finals)]
    pub max_pending_finals: usize,
    #[arg(
        id = "streaming.reuse_identical_audio",
        long = "reuse-identical-audio",
        conflicts_with = "no_reuse_identical_audio"
    )]
    pub reuse_identical_audio: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, ClapArgs)]
#[serde(default, deny_unknown_fields)]
pub struct LanguageConfig {
    #[arg(id = "language.model", long = "lid-model", default_value_t = Self::default().model)]
    pub model: String,
    #[arg(id = "language.model_cache", long = "model-cache", default_value_os_t = Self::default().model_cache)]
    pub model_cache: PathBuf,
    #[arg(id = "language.confidence", long = "lid-confidence", default_value_t = Self::default().confidence)]
    pub confidence: f32,
    #[arg(id = "language.hop_seconds", long = "lid-hop-seconds", default_value_t = Self::default().hop_seconds)]
    pub hop_seconds: f64,
    #[arg(id = "language.context_seconds", long = "lid-context-seconds", default_value_t = Self::default().context_seconds)]
    pub context_seconds: f64,
    #[arg(id = "language.min_seconds", long = "lid-min-seconds", default_value_t = Self::default().min_seconds)]
    pub min_seconds: f64,
    #[arg(id = "language.cache_windows", long = "lid-cache-windows", default_value_t = Self::default().cache_windows)]
    pub cache_windows: usize,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, ClapArgs)]
#[serde(default, deny_unknown_fields)]
pub struct OutputConfig {
    #[arg(id = "output.json", long = "json", conflicts_with = "no_json")]
    pub json: bool,
    #[arg(id = "output.save", long = "save")]
    pub save: Option<PathBuf>,
    #[arg(
        id = "output.editor",
        long = "editor",
        help = "Open a revisable transcript view and save events/text in a new directory"
    )]
    pub editor: Option<PathBuf>,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            device: "default".into(),
            input: None,
            duration: None,
            ffmpeg: "ffmpeg".into(),
        }
    }
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            transport: Transport::default(),
            url: "http://127.0.0.1:8000/v1/audio/transcriptions".into(),
            model: "qwen3-asr-1.7b".into(),
            language: None,
            context: None,
            timeout: 120.0,
        }
    }
}

impl Default for StreamConfig {
    fn default() -> Self {
        let silence_seconds = 0.5;
        Self {
            enabled: false,
            window_seconds: 24.0,
            interval: 1.0,
            silence_seconds,
            sentence_silence_seconds: sentence_silence(silence_seconds),
            lookback_seconds: 3.0,
            overlap_seconds: 0.0,
            speech_threshold: 0.5,
            padding_seconds: 0.2,
            min_partial_seconds: 0.8,
            max_in_flight: 4,
            max_pending_finals: 2,
            reuse_identical_audio: true,
        }
    }
}

impl Default for LanguageConfig {
    fn default() -> Self {
        Self {
            model: "tiny".into(),
            model_cache: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/asr-models"),
            confidence: 0.7,
            hop_seconds: 0.5,
            context_seconds: 0.5,
            min_seconds: 0.8,
            cache_windows: 256,
        }
    }
}

#[derive(Parser)]
#[command(about = "Record or convert audio and transcribe through khalamari. All online processing runs in Rust.")]
pub struct Args {
    #[command(flatten)]
    pub config: Config,
    #[arg(long = "config")]
    config_file: Option<PathBuf>,
    #[arg(long)]
    pub print_config: bool,
    #[arg(long)]
    pub list_devices: bool,
    #[arg(long, help = "Prepare the pinned external Whisper-tiny checkpoint and exit")]
    pub setup_models: bool,
    #[arg(
        long,
        requires = "setup_models",
        help = "Copy the pinned checkpoint from a local CT2 snapshot"
    )]
    pub whisper_source: Option<PathBuf>,
    #[arg(long, conflicts_with_all = ["capture.input", "no_stream", "setup_models", "list_devices"])]
    pub replay: Option<PathBuf>,
    #[arg(long)]
    no_stream: bool,
    #[arg(long)]
    no_json: bool,
    #[arg(long)]
    no_reuse_identical_audio: bool,
}

impl Args {
    pub fn load() -> Result<Self> {
        Self::resolve(Self::command().get_matches())
    }

    fn resolve(matches: ArgMatches) -> Result<Self> {
        let mut args = Self::from_arg_matches(&matches).expect("clap validated the declared argument types");
        let mut sentence_explicit =
            matches.value_source("streaming.sentence_silence_seconds") == Some(ValueSource::CommandLine);
        let mut config = match &args.config_file {
            Some(path) => {
                let bytes = std::fs::read(path)
                    .map_err(|e| Error::invalid_argument(format!("configuration {}: {e}", path.display())))?;
                let value: serde_json::Value = serde_json::from_slice(&bytes)
                    .map_err(|e| Error::invalid_argument(format!("configuration: {e}")))?;
                sentence_explicit |= value
                    .get("streaming")
                    .is_some_and(|stream| stream.get("sentence_silence_seconds").is_some());
                // Validate the file schema before CLI overrides can hide invalid fields or types.
                let mut config: Config = serde_json::from_value(value)
                    .map_err(|e| Error::invalid_argument(format!("configuration: {e}")))?;
                let directory = path.parent().unwrap();
                for path in [
                    &mut config.capture.input,
                    &mut config.output.save,
                    &mut config.output.editor,
                ]
                .into_iter()
                .flatten()
                {
                    if path.is_relative() {
                        *path = directory.join(&*path);
                    }
                }
                if config.language.model_cache.is_relative() {
                    config.language.model_cache = directory.join(&config.language.model_cache);
                }
                if config.language.model.contains('/') && Path::new(&config.language.model).is_relative() {
                    config.language.model = directory.join(&config.language.model).to_string_lossy().into();
                }
                if config.capture.ffmpeg.contains('/') && Path::new(&config.capture.ffmpeg).is_relative() {
                    config.capture.ffmpeg = directory.join(&config.capture.ffmpeg).to_string_lossy().into();
                }
                config
            },
            None => Config::default(),
        };
        let mut value =
            serde_json::to_value(config).map_err(|e| Error::invalid_argument(format!("configuration: {e}")))?;
        let serde_json::Value::Object(sections) =
            serde_json::to_value(&args.config).map_err(|e| Error::invalid_argument(format!("configuration: {e}")))?
        else {
            unreachable!()
        };
        for (section, fields) in sections {
            let serde_json::Value::Object(fields) = fields else {
                unreachable!()
            };
            for (field, field_value) in fields {
                if matches.value_source(&format!("{section}.{field}")) == Some(ValueSource::CommandLine) {
                    value[&section][&field] = field_value;
                }
            }
        }
        config = serde_json::from_value(value).map_err(|e| Error::invalid_argument(format!("configuration: {e}")))?;
        if args.no_stream {
            config.streaming.enabled = false;
        }
        if args.no_json {
            config.output.json = false;
        }
        if args.no_reuse_identical_audio {
            config.streaming.reuse_identical_audio = false;
        }
        if args.replay.is_some() && config.output.editor.is_none() {
            return Err(Error::invalid_argument("replay requires an editor output directory"));
        }
        if args.replay.is_none() && !config.streaming.enabled && config.capture.duration.is_none() {
            config.capture.duration = Some(30.0);
        }
        if !sentence_explicit {
            config.streaming.sentence_silence_seconds = sentence_silence(config.streaming.silence_seconds);
        }
        config.service.language = config
            .service
            .language
            .take()
            .map(|language| language.trim().to_owned())
            .filter(|language| !language.is_empty());
        config.validate(args.replay.is_some())?;
        args.config = config;
        Ok(args)
    }
}

impl Config {
    fn validate(&self, replay: bool) -> Result<()> {
        let invalid = |message| Err(Error::invalid_argument(message));
        let endpoint =
            reqwest::Url::parse(&self.service.url).map_err(|e| Error::invalid_argument(format!("ASR URL: {e}")))?;
        if !matches!(endpoint.scheme(), "http" | "https")
            || endpoint.host().is_none()
            || !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.fragment().is_some()
        {
            return invalid("ASR URL must be HTTP(S), without credentials or fragment");
        }
        if matches!(self.service.transport, Transport::Grpc)
            && (endpoint.scheme() != "http" || endpoint.path() != "/" || endpoint.query().is_some())
        {
            return invalid("gRPC URL must be an HTTP origin without a path or query");
        }
        if self.service.model.trim().is_empty()
            || self.language.model.trim().is_empty()
            || self.capture.ffmpeg.trim().is_empty()
        {
            return invalid("model names and FFmpeg executable must not be empty");
        }
        let timeout = Duration::try_from_secs_f64(self.service.timeout)
            .map_err(|_| Error::invalid_argument("ASR timeout must be a finite positive duration"))?;
        if timeout.is_zero() || Instant::now().checked_add(timeout).is_none() {
            return invalid("ASR timeout must fit the monotonic clock");
        }
        if let Some(duration) = self.capture.duration
            && (!(0.01..=(u64::MAX / SAMPLE_RATE as u64) as f64).contains(&duration)
                || (!self.streaming.enabled && duration > 30.0))
        {
            return invalid("invalid duration; collected audio must fit 30 seconds");
        }
        let s = &self.streaming;
        if !(0.01..=s.window_seconds).contains(&s.interval)
            || !(0.01..=30.0).contains(&s.window_seconds)
            || (s.enabled && s.window_seconds < 2.0)
        {
            return invalid("invalid partial interval or window (streaming window: 2..30 seconds)");
        }
        if !(0.1..=5.0).contains(&s.silence_seconds) || !(0.1..=s.silence_seconds).contains(&s.sentence_silence_seconds)
        {
            return invalid("invalid silence or sentence endpoint threshold");
        }
        if !(0.1..=30.0).contains(&s.lookback_seconds)
            || !(0.0..=s.window_seconds / 2.0).contains(&s.overlap_seconds)
            || (s.overlap_seconds != 0.0
                && (s.overlap_seconds < 0.1 || s.window_seconds + 2.0 * s.overlap_seconds > 30.0))
        {
            return invalid("invalid lookback or overlap; window plus both contexts must fit 30 seconds");
        }
        if !(0.0 < s.speech_threshold && s.speech_threshold <= 1.0)
            || !(0.0..=1.0).contains(&s.padding_seconds)
            || !(0.1..=s.window_seconds).contains(&s.min_partial_seconds)
            || !(1..=64).contains(&s.max_in_flight)
            || !(1..=64).contains(&s.max_pending_finals)
        {
            return invalid("invalid VAD, padding, minimum partial duration, or inference/final capacity");
        }
        let l = &self.language;
        if !(0.0 < l.confidence && l.confidence <= 1.0)
            || !(0.1..=2.0).contains(&l.hop_seconds)
            || !(0.0..=5.0).contains(&l.context_seconds)
            || !(0.1..=l.hop_seconds + 2.0 * l.context_seconds).contains(&l.min_seconds)
            || !(1..=4096).contains(&l.cache_windows)
        {
            return invalid("invalid language observation settings");
        }
        if !replay && self.capture.input.as_ref().is_some_and(|p| !p.is_file()) {
            return invalid("input must name an existing audio file");
        }
        if !replay
            && self.output.save.as_ref().is_some_and(|p| {
                p.exists()
                    || !p
                        .parent()
                        .filter(|p| !p.as_os_str().is_empty())
                        .unwrap_or(Path::new("."))
                        .is_dir()
            })
        {
            return invalid("save must name a new file in an existing directory");
        }
        if let Some(path) = &self.output.editor {
            if (!s.enabled && !replay) || self.output.json {
                return invalid("editor requires streaming and conflicts with JSON stdout");
            }
            if path.exists()
                || !path
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."))
                    .is_dir()
            {
                return invalid("editor must name a new directory in an existing parent directory");
            }
        }
        Ok(())
    }
}

pub fn samples(seconds: f64) -> usize {
    (seconds * SAMPLE_RATE as f64) as usize
}

fn sentence_silence(silence_seconds: f64) -> f64 {
    (silence_seconds / 2.0).max(0.1)
}
