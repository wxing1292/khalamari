use std::fs::File;
use std::io::BufWriter;
use std::io::Cursor;
use std::io::Seek;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use inference_error::Error;
use inference_error::Result;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::process::Child;
use tokio::process::ChildStdout;
use tokio::process::Command;
use tokio::task::JoinHandle;

use crate::config::CaptureConfig;

pub const SAMPLE_RATE: usize = 16_000;
pub const FRAME_SAMPLES: usize = 512;
const WAV_SPEC: hound::WavSpec = hound::WavSpec {
    channels: 1,
    sample_rate: SAMPLE_RATE as u32,
    bits_per_sample: 16,
    sample_format: hound::SampleFormat::Int,
};

pub struct WavRecording(hound::WavWriter<BufWriter<File>>);
impl WavRecording {
    pub fn create(path: &Path) -> Result<Self> {
        let file = File::create_new(path).map_err(recording_error)?;
        Self::new(file)
    }

    fn new(file: File) -> Result<Self> {
        Ok(Self(
            hound::WavWriter::new(BufWriter::new(file), WAV_SPEC).map_err(recording_error)?,
        ))
    }

    pub fn write(&mut self, pcm: &[i16]) -> Result<()> {
        write_samples(&mut self.0, pcm)
    }

    pub fn finish(self) -> Result<()> {
        self.0.finalize().map_err(recording_error)
    }
}

pub fn retain_recording(saved: Option<&Path>, pcm: &[i16]) -> Result<PathBuf> {
    if let Some(path) = saved {
        return Ok(path.to_owned());
    }
    let (file, path) = tempfile::Builder::new()
        .prefix("psi-dec-asr-")
        .suffix(".wav")
        .tempfile()
        .map_err(recording_error)?
        .keep()
        .map_err(recording_error)?;
    let mut recording = WavRecording::new(file)?;
    recording.write(pcm)?;
    recording.finish()?;
    Ok(path)
}

fn write_samples(writer: &mut hound::WavWriter<impl Write + Seek>, pcm: &[i16]) -> Result<()> {
    // The fixed mono i16 header uses 36 RIFF bytes before the sample payload.
    let remaining = (u64::from(u32::MAX) - 36) / 2 - u64::from(writer.len());
    if pcm.len() as u64 > remaining {
        return Err(Error::resource_exhausted("saved WAV exceeds the RIFF size limit"));
    }
    let mut output = writer.get_i16_writer(pcm.len() as u32);
    for &sample in pcm {
        output.write_sample(sample);
    }
    output.flush().map_err(recording_error)
}

fn recording_error(error: impl std::fmt::Display) -> Error {
    Error::unavailable(format!("WAV output: {error}"))
}

/// Owns the FFmpeg process, both output pipes, and an incomplete PCM frame.
/// All producers normalize to this boundary: 16 kHz, mono, signed i16.
pub struct FfmpegSource {
    child: Child,
    stdout: ChildStdout,
    errors: Option<JoinHandle<Vec<u8>>>,
    pending: Vec<u8>,
}

impl FfmpegSource {
    pub fn open(config: &CaptureConfig, streaming: bool) -> Result<Self> {
        let mut command = Command::new(&config.ffmpeg);
        command.args(["-hide_banner", "-loglevel", "error", "-nostats"]);
        match &config.input {
            Some(path) => {
                if streaming {
                    command.arg("-re");
                }
                command.arg("-i").arg(path);
            },
            None => {
                if !cfg!(target_os = "macos") {
                    return Err(Error::invalid_argument(
                        "microphone capture requires macOS; use --input",
                    ));
                }
                command.args(["-f", "avfoundation", "-i", &format!("none:{}", config.device)]);
            },
        }
        if let Some(duration) = config.duration.filter(|_| config.input.is_none() || streaming) {
            command.arg("-t").arg(duration.to_string());
        }
        command.args([
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "s16le",
            "pipe:1",
        ]);
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| Error::unavailable(format!("unable to start FFmpeg: {e}")))?;
        let stdout = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();
        let errors = tokio::spawn(async move {
            let mut tail = Vec::new();
            let mut bytes = [0; 4096];
            while let Ok(count) = stderr.read(&mut bytes).await {
                if count == 0 {
                    break;
                }
                tail.extend_from_slice(&bytes[..count]);
                if tail.len() > 65536 {
                    tail.drain(..tail.len() - 65536);
                }
            }
            tail
        });
        Ok(Self {
            child,
            stdout,
            errors: Some(errors),
            pending: Vec::with_capacity(FRAME_SAMPLES * 2),
        })
    }

    /// The pending bytes belong to the source, so cancellation preserves a partial read.
    pub async fn next_frame(&mut self) -> Result<Option<Vec<i16>>> {
        while self.pending.len() < FRAME_SAMPLES * 2 {
            let mut bytes = [0; FRAME_SAMPLES * 2];
            let needed = FRAME_SAMPLES * 2 - self.pending.len();
            let count = self
                .stdout
                .read(&mut bytes[..needed])
                .await
                .map_err(|e| Error::unavailable(format!("FFmpeg audio read: {e}")))?;
            if count == 0 {
                break;
            }
            self.pending.extend_from_slice(&bytes[..count]);
        }
        if !self.pending.len().is_multiple_of(2) {
            return Err(Error::invalid_argument("FFmpeg returned an incomplete i16 sample"));
        }
        if self.pending.is_empty() {
            return Ok(None);
        }
        let frame = self
            .pending
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| i16::from_le_bytes([p[0], p[1]]))
            .collect();
        self.pending.clear();
        Ok(Some(frame))
    }

    pub async fn stop(&mut self) -> Result<()> {
        // The capture owner requests graceful shutdown once. Taking stdin closes that control path.
        let mut input = self.child.stdin.take().expect("capture stops once");
        match input.write_all(b"q\n").await {
            Ok(()) => {},
            Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => {},
            Err(error) => return Err(Error::unavailable(format!("FFmpeg stop: {error}"))),
        }
        Ok(())
    }

    pub async fn finish(&mut self) -> Result<()> {
        let status = tokio::time::timeout(Duration::from_secs(5), self.child.wait())
            .await
            .map_err(|_| Error::deadline_exceeded("FFmpeg did not exit within 5 seconds"))?
            .map_err(|e| Error::unavailable(format!("FFmpeg wait: {e}")))?;
        let errors = self
            .errors
            .take()
            .unwrap()
            .await
            .expect("FFmpeg stderr reader panicked");
        if !status.success() {
            return Err(Error::unavailable(format!(
                "FFmpeg exited with {status}: {}",
                String::from_utf8_lossy(&errors)
            )));
        }
        Ok(())
    }
}

impl Drop for FfmpegSource {
    fn drop(&mut self) {
        if let Some(errors) = self.errors.take() {
            errors.abort();
        }
    }
}

pub fn wav(samples: &[i16]) -> Vec<u8> {
    let mut buffer = Cursor::new(Vec::with_capacity(samples.len() * 2 + 44));
    let mut writer = hound::WavWriter::new(&mut buffer, WAV_SPEC).expect("in-memory WAV header");
    write_samples(&mut writer, samples).expect("bounded in-memory WAV samples");
    writer.finalize().expect("in-memory WAV finalize");
    buffer.into_inner()
}

/// Search 10-ms RMS cells. Prefer a sustained quiet interval over a consonant closure.
pub fn quiet_cut(pcm: &[i16], candidate: usize, previous: usize, radius: usize, minimum: usize) -> Option<usize> {
    let lo = (previous + minimum).max(candidate.saturating_sub(radius));
    let hi = pcm.len().saturating_sub(minimum).min(candidate + radius);
    let mut quiet_start = None;
    let mut best: Option<(usize, usize, usize)> = None;
    for start in (lo..=hi).step_by(160) {
        let quiet = start + 160 <= hi
            && pcm[start..start + 160]
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                < 160.0 * 160.0 * 160.0;
        match (quiet, quiet_start) {
            (true, None) => quiet_start = Some(start),
            (false, Some(begin)) => {
                if start - begin >= 1280 {
                    let cut = (begin + start) / 2;
                    let choice = (usize::MAX - (start - begin), cut.abs_diff(candidate), cut);
                    if best.is_none_or(|old| choice < old) {
                        best = Some(choice);
                    }
                }
                quiet_start = None;
            },
            _ => {},
        }
    }
    best.map(|choice| choice.2)
}
