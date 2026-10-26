use std::time::Duration;
use std::time::Instant;

use inference_error::Error;
use inference_error::Result;

use crate::audio::FfmpegSource;
use crate::audio::WavRecording;
use crate::audio::retain_recording;
use crate::config::Config;
use crate::config::samples;
use crate::output::Output;
use crate::service::AsrClient;
use crate::service::AudioTranscribeReq;
use crate::stream::AudioStream;
use crate::terminal::TerminalInput;

pub struct AudioSession {
    config: Config,
    output: Output,
    stream: Option<AudioStream>,
    input: Option<TerminalInput>,
    saved: Option<WavRecording>,
    capture: FfmpegSource,
}

impl AudioSession {
    pub fn open(config: Config) -> Result<Self> {
        let output = Output::open(&config.output)?;
        let stream = if config.streaming.enabled {
            eprintln!("Loading native CPU speech models.");
            Some(tokio::task::block_in_place(|| AudioStream::load(&config))?)
        } else {
            None
        };
        let input = TerminalInput::new(config.capture.input.is_none() || output.is_editor(), output.is_editor())?;
        let saved = config.output.save.as_deref().map(WavRecording::create).transpose()?;
        let capture = FfmpegSource::open(&config.capture, stream.is_some())?;

        Ok(Self {
            config,
            output,
            stream,
            input,
            saved,
            capture,
        })
    }

    pub async fn run(self) -> Result<()> {
        let Self {
            config,
            mut output,
            mut stream,
            mut input,
            mut saved,
            mut capture,
        } = self;
        eprintln!(
            "{}",
            if stream.is_some() {
                "Streaming. Press Enter to stop and flush; Ctrl-C cancels."
            } else {
                "Recording. Press Enter to stop; Ctrl-C cancels."
            }
        );
        output.status(
            "Recording. :replace ID TEXT | :sub OLD => NEW | Enter: stop/save | Ctrl-C: cancel",
            "",
        )?;
        let mut collected = Vec::new();
        let mut stopping = None;
        let mut last_audio = Instant::now();
        let mut capture_done = false;
        loop {
            if let Some(stream) = &mut stream {
                stream.schedule();
            }
            if capture_done && stream.as_ref().is_none_or(AudioStream::is_drained) {
                break;
            }
            let deadline = stopping.map_or(last_audio + Duration::from_secs(15), |started| {
                started + Duration::from_secs(5)
            });
            tokio::select! {
                frame = capture.next_frame(), if !capture_done => {
                    let Some(frame) = frame? else {
                        capture.finish().await?;
                        capture_done = true;
                        let samples = match &mut stream {
                            Some(stream) => stream.finish()?,
                            None => collected.len() as u64,
                        };
                        if samples == 0 {
                            return Err(Error::invalid_argument("recording contains no audio"));
                        }
                        output.status(
                            "Capture ended. Finals will finish here; edit text, or Enter to save/exit.",
                            &input.as_ref().map_or(String::new(), TerminalInput::line),
                        )?;
                        continue;
                    };
                    last_audio = Instant::now();
                    if let Some(writer) = &mut saved {
                        writer.write(&frame)?;
                    }
                    if let Some(stream) = &mut stream {
                        tokio::task::block_in_place(|| stream.feed(&frame))?;
                    } else {
                        if collected.len() + frame.len() > samples(config.capture.duration.unwrap()) {
                            return Err(Error::invalid_argument("audio exceeds the configured collected duration"));
                        }
                        collected.extend(frame);
                    }
                },
                events = async {
                    match &mut stream {
                        Some(stream) => stream.next().await,
                        None => std::future::pending().await,
                    }
                } => {
                    for event in events? {
                        output.emit(&event, &input.as_ref().map_or(String::new(), TerminalInput::line))?;
                    }
                },
                command = async {
                    match &mut input {
                        Some(input) => input.wait().await,
                        None => std::future::pending().await,
                    }
                }, if stopping.is_none() && (output.is_editor() || !capture_done) => {
                    if output.input(command?, &input.as_ref().unwrap().line())? {
                        if !capture_done {
                            capture.stop().await?;
                        }
                        stopping = Some(Instant::now());
                    }
                },
                _ = tokio::time::sleep_until(deadline.into()), if !capture_done => {
                    return Err(Error::deadline_exceeded(if stopping.is_some() {
                        "FFmpeg did not stop within 5 seconds"
                    } else { "no audio received for 15 seconds; check input device and microphone permission" }));
                },
            }
        }
        if let Some(saved) = saved {
            saved.finish()?;
        }
        if stream.is_none() {
            if collected.len() < 160 {
                return Err(Error::invalid_argument(
                    "recording must contain at least 10 ms of audio",
                ));
            }
            let client = AsrClient::new(config.service.clone())?;
            let result = match client
                .transcribe(AudioTranscribeReq {
                    audio: &collected,
                    language: config.service.language.as_deref().unwrap_or(""),
                })
                .await
            {
                Ok(result) => result,
                Err(error) => {
                    let path = retain_recording(config.output.save.as_deref(), &collected)?;
                    eprintln!("Recording retained at {}; retry with --input.", path.display());
                    return Err(error);
                },
            };
            output.collected(&result)?;
        }
        if output.is_editor() && stopping.is_none() {
            output.edit(input.as_mut().unwrap()).await?;
        }
        output.finish()
    }
}
