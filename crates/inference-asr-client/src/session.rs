use std::collections::VecDeque;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use inference_error::Error;
use inference_error::Result;

use crate::assets::SpeechAssets;
use crate::audio::AudioCapture;
use crate::audio::FfmpegSource;
use crate::config::Config;
use crate::config::Mode;
use crate::output::Output;
use crate::output::write_json;
use crate::recording::AudioRecordingReader;
use crate::recording::AudioRecordingWriter;
use crate::recording::archive;
use crate::recording::file_stem;
use crate::segment::AnalysisPass;
use crate::stream::AudioStream;
use crate::stream::SpeechModels;
use crate::terminal::TerminalInput;

/// Capture, analysis, and archive have independent lifetimes and share one recording clock.
pub struct AudioSession {
    config: Config,
    assets: SpeechAssets,
    output: Output,
    input: Option<TerminalInput>,
    recording: AudioRecordingReader,
    directory: PathBuf,
    capture: Option<AudioCapture>,
}

impl AudioSession {
    pub async fn run(
        config: Config,
        assets: SpeechAssets,
        cancelled: impl std::future::Future<Output = ()>,
    ) -> Result<()> {
        let output = Output::open(&config.output)?;
        let (recording, spool) = AudioRecordingWriter::create()?;
        let reader = recording.reader();
        let input = TerminalInput::new(config.capture.input.is_none() || output.is_editor(), output.is_editor())?;
        let directory = std::env::current_dir().map_err(session_error)?.join(file_stem()?);
        std::fs::create_dir(&directory).map_err(session_error)?;
        let destination = if config.output.discard_recording {
            None
        } else {
            Some(
                config
                    .output
                    .save
                    .clone()
                    .unwrap_or_else(|| directory.join("recording.m4a")),
            )
        };
        let source = FfmpegSource::open(&config)?;
        let archive = destination.map(|destination| {
            let recording = reader.clone();
            let ffmpeg = config.capture.ffmpeg.clone();
            tokio::spawn(async move {
                archive(recording, &ffmpeg, &destination).await?;
                Ok(destination)
            })
        });
        let capture = AudioCapture::start(source, recording);
        let mut session = Self {
            config,
            assets,
            output,
            input,
            recording: reader,
            directory,
            capture: Some(capture),
        };
        let result = tokio::select! {
            result = session.process() => result,
            _ = cancelled => Err(Error::cancelled("analysis cancelled")),
        };
        let captured = session.stop_capture().await;
        let mut result = captured.and(result);
        if let Some(task) = archive {
            match task.await.expect("recording archive task must not panic") {
                Ok(destination) => {
                    result = result.and(session.status(&format!("Recording saved: {}", destination.display())));
                },
                Err(error) if session.recording.end() == 0 => result = result.and(Err(error)),
                Err(error) => {
                    result = spool.keep().map_err(session_error).and_then(|path| {
                        Err(Error::unavailable(format!(
                            "{error}. Recover 16 kHz mono s16le audio at {}",
                            path.display()
                        )))
                    });
                },
            }
        }
        result.and(session.output.finish())
    }

    async fn process(&mut self) -> Result<()> {
        self.status(&format!(
            "Recording recovery file: {} (16 kHz, mono, s16le)",
            self.recording.path().display()
        ))?;
        self.status(if self.config.capture.input.is_some() {
            "Reading audio file. Ctrl-C cancels analysis and saves the recording."
        } else {
            "Recording. Press Enter to stop and flush; Ctrl-C cancels analysis and saves the recording."
        })?;
        if self.output.is_editor() {
            self.status("Recording. :replace ID TEXT | :sub OLD => NEW | Enter: stop | Ctrl-C: cancel")?;
        }
        let pass = if self.config.mode == Mode::Stream {
            AnalysisPass::Live
        } else {
            self.wait_capture().await?;
            self.status("Processing the complete recording.")?;
            AnalysisPass::Offline
        };
        self.status("Loading native CPU speech models.")?;
        let models = {
            let config = self.config.clone();
            let whisper = self.assets.whisper.clone();
            self.wait_analysis(SpeechModels::load(&config, &whisper)).await
        };
        let mut models = match models {
            Ok(models) => models,
            Err(error) => {
                self.status(&format!("Analysis could not start: {error}."))?;
                self.wait_capture().await?;
                return Err(error);
            },
        };
        let first = self.analyze(&mut models, pass, VecDeque::new()).await;
        if let Err(error) = &first {
            self.status(&format!("{} analysis stopped: {error}.", pass.name()))?;
        }
        self.wait_capture().await?;
        let (live_cuts, first) = match first {
            Ok(cuts) => (cuts, Ok(())),
            Err(error) => (VecDeque::new(), Err(error)),
        };
        if pass == AnalysisPass::Live && self.config.streaming.reprocess {
            self.status("Processing the complete recording.")?;
            self.analyze(&mut models, AnalysisPass::Offline, live_cuts).await?;
        }
        first?;
        if self.output.is_editor()
            && let Some(input) = &mut self.input
        {
            self.output.edit(input).await?;
        }
        Ok(())
    }

    async fn analyze(
        &mut self,
        models: &mut SpeechModels,
        pass: AnalysisPass,
        live_cuts: VecDeque<u64>,
    ) -> Result<VecDeque<u64>> {
        let name = if pass == AnalysisPass::Live {
            "live.txt"
        } else {
            "transcript.txt"
        };
        let transcript = self.directory.join(name);
        self.output.start_pass(pass, &transcript)?;
        let result: Result<_> = async {
            self.status(&format!("Transcript: {}", transcript.display()))?;
            let mut log = self
                .config
                .output
                .json
                .then(|| {
                    File::create_new(self.directory.join(format!("{}.jsonl", pass.name())))
                        .map(BufWriter::new)
                        .map_err(session_error)
                })
                .transpose()?;
            let mut stream = {
                let config = self.config.clone();
                let assets = self.assets.clone();
                self.wait_analysis(AudioStream::open(
                    &config,
                    self.recording.clone(),
                    pass,
                    models,
                    &assets,
                    live_cuts,
                ))
                .await?
            };
            let mut cuts = VecDeque::new();
            while let Some(events) = self.wait_analysis(stream.next()).await? {
                for event in events {
                    if pass == AnalysisPass::Live
                        && self.config.streaming.reprocess
                        && event.kind == "final"
                        && matches!(event.reason, "silence" | "sentence" | "speaker")
                    {
                        cuts.push_back(event.end);
                    }
                    if let Some(log) = &mut log {
                        write_json(log, &event)?;
                    }
                    self.output
                        .emit(&event, &self.input.as_ref().map_or(String::new(), TerminalInput::line))?;
                }
            }
            let stats = self.wait_analysis(stream.stop()).await?;
            if self.config.output.json {
                eprintln!("Audio transport: {}", serde_json::to_string(&stats).unwrap());
            }
            Ok(cuts)
        }
        .await;
        let saved = self.output.stop_pass();
        let cuts = result?;
        saved?;
        self.status(&format!("Transcript saved: {}", transcript.display()))?;
        Ok(cuts)
    }

    /// Keep capture and terminal control responsive during initialization, inference and connection drain.
    async fn wait_analysis<T>(&mut self, operation: impl std::future::Future<Output = Result<T>>) -> Result<T> {
        let mut operation = std::pin::pin!(operation);
        loop {
            tokio::select! {
                result = &mut operation => return result,
                captured = async { self.capture.as_mut().unwrap().wait().await }, if self.capture.is_some() => {
                    self.capture.take();
                    captured?;
                    self.status("Capture ended. Processing retained audio.")?;
                },
                command = async { self.input.as_mut().unwrap().wait().await }, if self.input.is_some() => {
                    self.handle_input(command?)?;
                },
            }
        }
    }

    fn handle_input(&mut self, command: Option<String>) -> Result<()> {
        let line = self.input.as_ref().unwrap().line();
        if self.output.input(command, &line)? {
            if let Some(capture) = &mut self.capture {
                capture.stop();
            }
            self.input.take();
        }
        Ok(())
    }

    async fn wait_capture(&mut self) -> Result<()> {
        while self.capture.is_some() {
            tokio::select! {
                captured = self.capture.as_mut().unwrap().wait() => {
                    self.capture.take();
                    captured?;
                },
                command = async { self.input.as_mut().unwrap().wait().await }, if self.input.is_some() => {
                    self.handle_input(command?)?;
                },
            }
        }
        Ok(())
    }

    async fn stop_capture(&mut self) -> Result<()> {
        let Some(capture) = &mut self.capture else {
            return Ok(());
        };
        capture.stop();
        let result = capture.wait().await;
        self.capture.take();
        result
    }

    fn status(&mut self, message: &str) -> Result<()> {
        self.output
            .status(message, &self.input.as_ref().map_or(String::new(), TerminalInput::line))
    }
}

fn session_error(error: impl std::fmt::Display) -> Error {
    Error::unavailable(format!("audio session: {error}"))
}
