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
    #[arg(long, value_enum, default_value_t = Mode::default(), help_heading = "Workflow", help = "Stream live then reprocess, or record first and analyze offline")]
    pub mode: Mode,
    #[command(flatten, next_help_heading = "Input")]
    pub capture: CaptureConfig,
    #[command(flatten, next_help_heading = "Service")]
    pub service: ServiceConfig,
    #[command(flatten, next_help_heading = "Streaming")]
    pub streaming: StreamConfig,
    #[command(flatten, next_help_heading = "Language")]
    pub language: LanguageConfig,
    #[command(flatten, next_help_heading = "Models")]
    pub models: ModelConfig,
    #[command(flatten, next_help_heading = "Output")]
    pub output: OutputConfig,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Stream,
    #[default]
    Offline,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Stream => "stream",
            Self::Offline => "offline",
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ClapArgs)]
#[serde(default, deny_unknown_fields)]
pub struct CaptureConfig {
    #[arg(id = "capture.device", help = "Microphone device name or index", value_name = "DEVICE", long = "device", default_value_t = Self::default().device)]
    pub device: String,
    #[arg(
        id = "capture.input",
        help = "Read a recording instead of the microphone",
        value_name = "PATH",
        long = "input"
    )]
    pub input: Option<PathBuf>,
    #[arg(
        id = "capture.duration",
        help = "Stop recording after this many seconds",
        value_name = "SECONDS",
        long = "duration"
    )]
    pub duration: Option<f64>,
    #[arg(skip)]
    pub ffmpeg: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ClapArgs)]
#[serde(default, deny_unknown_fields)]
pub struct ServiceConfig {
    #[arg(id = "service.transport", help = "ASR service transport", value_name = "TRANSPORT", long = "transport", value_enum, default_value_t = Self::default().transport)]
    pub transport: Transport,
    #[arg(id = "service.url", help = "ASR service endpoint", value_name = "URL", long = "url", default_value_t = Self::default().url)]
    pub url: String,
    #[arg(skip)]
    pub model: String,
    #[arg(
        id = "service.language",
        help = "Force a language; omit for mixed speech",
        value_name = "LANGUAGE",
        long = "language"
    )]
    pub language: Option<String>,
    #[arg(
        id = "service.context",
        help = "Vocabulary or transcription context",
        value_name = "TEXT",
        long = "context"
    )]
    pub context: Option<String>,
    #[arg(skip)]
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
    #[arg(
        id = "streaming.reprocess",
        help = "Run offline analysis after streaming",
        long = "reprocess",
        conflicts_with = "no_reprocess"
    )]
    pub reprocess: bool,
    #[arg(skip)]
    pub window_seconds: f64,
    #[arg(skip)]
    pub interval: f64,
    #[arg(skip)]
    pub silence_seconds: f64,
    #[arg(skip)]
    pub sentence_silence_seconds: f64,
    #[arg(skip)]
    pub lookback_seconds: f64,
    #[arg(skip)]
    pub overlap_seconds: f64,
    #[arg(skip)]
    pub speech_threshold: f32,
    #[arg(skip)]
    pub padding_seconds: f64,
    #[arg(skip)]
    pub min_partial_seconds: f64,
    #[arg(id = "streaming.max_in_flight", help = "Maximum concurrent ASR requests", value_name = "COUNT", long = "max-in-flight", default_value_t = Self::default().max_in_flight)]
    pub max_in_flight: usize,
    #[arg(skip)]
    pub max_pending_finals: usize,
    #[arg(skip)]
    pub reuse_identical_audio: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, ClapArgs)]
#[serde(default, deny_unknown_fields)]
pub struct LanguageConfig {
    #[arg(
        id = "language.fast",
        long = "fast-lid",
        help = "Use a five-second minimum Whisper input and require at least 0.9 confidence for a language hint"
    )]
    pub fast: bool,
    #[arg(skip)]
    pub confidence: f32,
    #[arg(skip)]
    pub hop_seconds: f64,
    #[arg(skip)]
    pub context_seconds: f64,
    #[arg(skip)]
    pub min_seconds: f64,
    #[arg(skip)]
    pub cache_windows: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, ClapArgs)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    #[arg(id = "models.whisper", help = "Whisper checkpoint name or directory", value_name = "NAME_OR_PATH", long = "lid-model", default_value_t = Self::default().whisper)]
    pub whisper: String,
    #[arg(id = "models.speaker_command", hide_default_value = true, value_name = "PATH", long = "speaker-command", default_value_os_t = Self::default().speaker_command,
        help = "Speaker reader executable")]
    pub speaker_command: PathBuf,
    #[arg(
        id = "models.speaker_model_dir",
        long = "speaker-model-dir",
        value_name = "PATH",
        help = "Override the speaker reader checkpoint directory"
    )]
    pub speaker_model_dir: Option<PathBuf>,
    #[arg(id = "models.cache", hide_default_value = true, help = "Directory for prepared Whisper checkpoints", value_name = "PATH", long = "model-cache", default_value_os_t = Self::default().cache)]
    pub cache: PathBuf,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            whisper: "tiny".into(),
            speaker_command: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/asr-readers/community-1/reader"),
            speaker_model_dir: None,
            cache: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/asr-models"),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, ClapArgs)]
#[serde(default, deny_unknown_fields)]
pub struct OutputConfig {
    #[arg(
        id = "output.json",
        help = "Write transcript events as JSON Lines",
        long = "json",
        conflicts_with = "no_json"
    )]
    pub json: bool,
    #[arg(
        id = "output.save",
        help = "Save complete audio to a new WAV or FLAC file",
        value_name = "PATH",
        long = "save"
    )]
    pub save: Option<PathBuf>,
    #[arg(
        id = "output.discard_recording",
        help = "Discard complete audio after processing",
        long = "discard-recording",
        conflicts_with = "output.save"
    )]
    pub discard_recording: bool,
    #[arg(
        id = "output.editor",
        value_name = "PATH",
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
            reprocess: true,
            window_seconds: 24.0,
            interval: 0.5,
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
            fast: false,
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
    #[arg(
        long = "config",
        value_name = "PATH",
        help_heading = "Configuration",
        help = "Read advanced settings from a JSON file"
    )]
    config_file: Option<PathBuf>,
    #[arg(long, help_heading = "Configuration", help = "Print resolved settings and exit")]
    pub print_config: bool,
    #[arg(long, help_heading = "Setup", help = "List microphone devices and exit")]
    pub list_devices: bool,
    #[arg(
        long,
        help_heading = "Setup",
        help = "Prepare the pinned Whisper-tiny checkpoint and exit"
    )]
    pub setup_models: bool,
    #[arg(
        long,
        requires = "setup_models",
        value_name = "PATH",
        help_heading = "Setup",
        help = "Copy the pinned checkpoint from a local CT2 snapshot"
    )]
    pub whisper_source: Option<PathBuf>,
    #[arg(long, value_name = "PATH", help_heading = "Output", conflicts_with_all = ["capture.input", "mode", "setup_models", "list_devices"])]
    pub replay: Option<PathBuf>,
    #[arg(
        long,
        help_heading = "Output",
        help = "Override JSON output from the configuration file"
    )]
    no_json: bool,
    #[arg(long, help_heading = "Streaming", help = "Keep live results without an offline pass")]
    no_reprocess: bool,
}

impl Args {
    pub fn load() -> Result<Self> {
        Self::resolve(Self::command().get_matches())
    }

    fn resolve(matches: ArgMatches) -> Result<Self> {
        let mut args = Self::from_arg_matches(&matches).expect("clap validated the declared argument types");
        let mut sentence_explicit = false;
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
                    &mut config.models.speaker_model_dir,
                ]
                .into_iter()
                .flatten()
                {
                    if !path.as_os_str().is_empty() && path.is_relative() {
                        *path = directory.join(&*path);
                    }
                }
                if config.models.cache.is_relative() {
                    config.models.cache = directory.join(&config.models.cache);
                }
                if config.models.speaker_command.is_relative() {
                    config.models.speaker_command = directory.join(&config.models.speaker_command);
                }
                let model = &mut config.models.whisper;
                if model.contains('/') && Path::new(model).is_relative() {
                    *model = directory.join(&*model).to_string_lossy().into();
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
        let cli =
            serde_json::to_value(&args.config).map_err(|e| Error::invalid_argument(format!("configuration: {e}")))?;
        for id in matches.ids() {
            let id = id.as_str();
            if matches.value_source(id) != Some(ValueSource::CommandLine) {
                continue;
            }
            if let Some((section, field)) = id.split_once('.') {
                value[section][field] = cli[section][field].clone();
            } else if let Some(setting) = cli.get(id) {
                value[id] = setting.clone();
            }
        }
        config = serde_json::from_value(value).map_err(|e| Error::invalid_argument(format!("configuration: {e}")))?;
        if args.no_reprocess {
            config.streaming.reprocess = false;
        }
        if args.no_json {
            config.output.json = false;
        }
        if args.replay.is_some() && config.output.editor.is_none() {
            return Err(Error::invalid_argument("replay requires an editor output directory"));
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
        if config.language.fast {
            config.language.confidence = config.language.confidence.max(0.9);
        }
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
            || self.models.whisper.trim().is_empty()
            || self.models.speaker_command.as_os_str().is_empty()
            || self.capture.ffmpeg.trim().is_empty()
        {
            return invalid("model names and executable paths must not be empty");
        }
        if self
            .models
            .speaker_model_dir
            .as_ref()
            .is_some_and(|path| path.as_os_str().is_empty())
        {
            return invalid("speaker-model-dir must not be empty");
        }
        let timeout = Duration::try_from_secs_f64(self.service.timeout)
            .map_err(|_| Error::invalid_argument("ASR timeout must be a finite positive duration"))?;
        if timeout.is_zero() || Instant::now().checked_add(timeout).is_none() {
            return invalid("ASR timeout must fit the monotonic clock");
        }
        if let Some(duration) = self.capture.duration
            && !(0.01..=(u64::MAX / SAMPLE_RATE as u64) as f64).contains(&duration)
        {
            return invalid("duration must be at least 10 ms and fit the recording sample clock");
        }
        let s = &self.streaming;
        if self.output.discard_recording && self.output.save.is_some() {
            return invalid("discard-recording conflicts with save");
        }
        if self
            .output
            .save
            .as_ref()
            .is_some_and(|p| !p.extension().is_some_and(|e| e == "flac" || e == "wav"))
        {
            return invalid("save must use a .flac or .wav extension");
        }
        if !(0.01..=s.window_seconds).contains(&s.interval) || !(2.0..=30.0).contains(&s.window_seconds) {
            return invalid("invalid partial interval or window (analysis window: 2..30 seconds)");
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
            if (self.mode != Mode::Stream && !replay) || self.output.json {
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
