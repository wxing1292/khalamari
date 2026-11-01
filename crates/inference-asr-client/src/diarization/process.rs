use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::ops::Range;
use std::path::Path;
use std::process::Child;
use std::process::ChildStdin;
use std::process::ChildStdout;
use std::process::Command;
use std::process::Stdio;

use inference_error::Error;
use inference_error::Result;
use serde::Deserialize;
use serde::Serialize;

use crate::analysis::AudioAnalysisInput;
use crate::analysis::AudioAnalysisUpdate;
use crate::analysis::AudioProcessor;
use crate::analysis::AudioReaderTask;
use crate::diarization::AudioDiarizationUpdate;
use crate::diarization::AudioDiarizer;
use crate::recording::AudioRecordingReader;
use crate::segment::AnalysisPass;

/// The backend opens the same append-only PCM log read-only. Pipes carry progress and labels only.
pub struct SpeakerProcess {
    child: Option<Child>,
}

impl SpeakerProcess {
    pub async fn start(
        program: &Path,
        model_dir: Option<&Path>,
        pass: AnalysisPass,
        recording: AudioRecordingReader,
    ) -> Result<(Self, AudioReaderTask<AudioDiarizationUpdate>)> {
        let mut command = Command::new(program);
        command
            .arg("--recording")
            .arg(recording.path())
            .arg("--pass")
            .arg(pass.name());
        if let Some(model_dir) = model_dir {
            command.arg("--model-dir").arg(model_dir);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(process_error)?;
        let mut diarizer = ProcessDiarizer {
            stdin: child.stdin.take().unwrap(),
            stdout: BufReader::new(child.stdout.take().unwrap()),
            input: None,
            submitted: 0,
            covered: 0,
            sealed: 0,
        };
        // The process guard exists before awaiting native readiness. Cancellation always terminates the child.
        let process = Self { child: Some(child) };
        let input = recording.clone();
        let model = AudioProcessor::load(move || {
            diarizer.start(input);
            diarizer.ready()?;
            Ok(diarizer)
        })
        .await?;
        let reader = AudioReaderTask::start(recording, model, |model, range, stopped| {
            if !range.is_empty() {
                model.append(range);
            }
            if stopped {
                model.stop();
            }
            model.diarize()
        });
        Ok((process, reader))
    }
}

impl Drop for SpeakerProcess {
    fn drop(&mut self) {
        let mut child = self.child.take().unwrap();
        let _ = child.kill();
        // Reaping must not hold the session event loop or wait for a blocked pipe reader.
        tokio::task::spawn_blocking(move || {
            let _ = child.wait();
        });
    }
}

#[derive(Serialize)]
struct Request {
    start: u64,
    end: u64,
    stopped: bool,
}

#[derive(Deserialize)]
#[serde(untagged, deny_unknown_fields)]
enum Ready {
    Success { protocol: u32 },
    Failure { error: String },
}

struct ProcessDiarizer {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    input: Option<AudioAnalysisInput>,
    submitted: u64,
    covered: u64,
    sealed: u64,
}

impl ProcessDiarizer {
    fn ready(&mut self) -> Result<()> {
        match self.read()? {
            Ready::Success { protocol: 1 } => Ok(()),
            Ready::Success { .. } => Err(Error::invalid_argument("unsupported speaker reader protocol")),
            Ready::Failure { error } => Err(Error::unavailable(format!("speaker reader initialization: {error}"))),
        }
    }

    fn read<T: serde::de::DeserializeOwned>(&mut self) -> Result<T> {
        let mut line = Vec::new();
        // Bound an external response before allocating its parsed span vectors.
        std::io::Read::take(&mut self.stdout, 16 * 1024 * 1024)
            .read_until(b'\n', &mut line)
            .map_err(process_error)?;
        if line.is_empty() {
            return Err(Error::unavailable("speaker reader exited before sending a response"));
        }
        if line.last() != Some(&b'\n') {
            return Err(Error::unavailable(
                "speaker reader ended without a complete response or exceeded 16 MiB",
            ));
        }
        serde_json::from_slice(&line)
            .map_err(|error| Error::invalid_argument(format!("speaker reader response: {error}")))
    }
}

impl AudioDiarizer for ProcessDiarizer {
    fn start(&mut self, recording: AudioRecordingReader) {
        assert!(self.input.is_none(), "a speaker process owns one analysis pass");
        self.input = Some(AudioAnalysisInput::new(recording));
    }

    fn append(&mut self, range: Range<u64>) {
        self.input.as_mut().unwrap().append(range);
    }

    fn stop(&mut self) {
        self.input.as_mut().unwrap().stop();
    }

    fn diarize(&mut self) -> Result<Vec<AudioDiarizationUpdate>> {
        let input = self.input.as_ref().unwrap();
        let (end, stopped) = (input.end, input.stopped);
        serde_json::to_writer(
            &mut self.stdin,
            &Request {
                start: self.submitted,
                end,
                stopped,
            },
        )
        .map_err(process_error)?;
        self.stdin.write_all(b"\n").map_err(process_error)?;
        self.stdin.flush().map_err(process_error)?;
        let mut updates: Vec<AudioDiarizationUpdate> = self.read()?;
        for update in &mut updates {
            match update {
                AudioAnalysisUpdate::Replace { range, spans } => {
                    if range.start < self.sealed
                        || range.start > self.covered
                        || range.start >= range.end
                        || range.end > end
                        || spans
                            .iter()
                            .any(|span| span.start < range.start || span.start >= span.end || span.end > range.end)
                    {
                        return Err(Error::invalid_argument(
                            "speaker replacement must cover published, unsealed audio without holes",
                        ));
                    }
                    spans.sort_by_key(|span| (span.start, span.speaker));
                    self.covered = self.covered.max(range.end);
                },
                AudioAnalysisUpdate::Seal { range } => {
                    if range.start != self.sealed || range.start >= range.end || range.end > self.covered {
                        return Err(Error::invalid_argument("speaker seals must advance a covered prefix"));
                    }
                    self.sealed = range.end;
                },
            }
        }
        if stopped && self.sealed != end {
            return Err(Error::invalid_argument("speaker reader must seal all input after stop"));
        }
        self.submitted = end;
        Ok(updates)
    }
}

fn process_error(error: impl std::fmt::Display) -> Error {
    Error::unavailable(format!("speaker reader: {error}"))
}
