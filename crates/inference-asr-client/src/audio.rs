use std::process::Stdio;
use std::time::Duration;

use inference_error::Error;
use inference_error::Result;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::process::Child;
use tokio::process::ChildStdout;
use tokio::process::Command;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::config::Config;
use crate::config::Mode;
use crate::recording::AudioRecordingWriter;

pub const SAMPLE_RATE: usize = 16_000;
pub const FRAME_SAMPLES: usize = 512;

/// Capture owns the only recording writer. Analysis and compression have independent readers.
pub struct AudioCapture {
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<Result<()>>,
}

impl AudioCapture {
    pub fn start(mut source: FfmpegSource, mut recording: AudioRecordingWriter) -> Self {
        let (stop, mut requested) = oneshot::channel();
        let task = tokio::spawn(async move {
            let result = async {
                let mut stopping = false;
                let mut deadline = tokio::time::Instant::now() + Duration::from_secs(15);
                loop {
                    tokio::select! {
                        frame = source.next_frame() => {
                            let Some(frame) = frame? else { break; };
                            tokio::task::block_in_place(|| recording.append(frame))?;
                            if !stopping {
                                deadline = tokio::time::Instant::now() + Duration::from_secs(15);
                            }
                        },
                        _ = &mut requested, if !stopping => {
                            stopping = true;
                            source.stop().await?;
                            deadline = tokio::time::Instant::now() + Duration::from_secs(5);
                        },
                        _ = tokio::time::sleep_until(deadline) => {
                            return Err(Error::deadline_exceeded(if stopping {
                                "FFmpeg did not stop within 5 seconds"
                            } else {
                                "no audio received for 15 seconds; check input device and microphone permission"
                            }));
                        },
                    }
                }
                source.finish().await
            }
            .await;
            let sealed = tokio::task::block_in_place(|| recording.stop());
            result.and(sealed)
        });
        Self { stop: Some(stop), task }
    }

    pub fn stop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
    }

    pub async fn wait(&mut self) -> Result<()> {
        (&mut self.task).await.expect("capture task must not panic")
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Owns the FFmpeg process, both output pipes, and an incomplete PCM frame.
/// All producers normalize to this boundary: 16 kHz, mono, signed i16.
pub struct FfmpegSource {
    child: Child,
    stdout: ChildStdout,
    errors: Option<JoinHandle<Vec<u8>>>,
    frame: [u8; FRAME_SAMPLES * 2],
    filled: usize,
}

impl FfmpegSource {
    pub fn open(settings: &Config) -> Result<Self> {
        let config = &settings.capture;
        let mut command = Command::new(&config.ffmpeg);
        command.args(["-hide_banner", "-loglevel", "error", "-nostats"]);
        match &config.input {
            Some(path) => {
                if settings.mode == Mode::Stream {
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
        if let Some(duration) = config.duration {
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
            frame: [0; FRAME_SAMPLES * 2],
            filled: 0,
        })
    }

    /// The pending bytes belong to the source, so cancellation preserves a partial read.
    pub async fn next_frame(&mut self) -> Result<Option<&[u8]>> {
        while self.filled < self.frame.len() {
            let count = self
                .stdout
                .read(&mut self.frame[self.filled..])
                .await
                .map_err(|e| Error::unavailable(format!("FFmpeg audio read: {e}")))?;
            if count == 0 {
                break;
            }
            self.filled += count;
        }
        if !self.filled.is_multiple_of(2) {
            return Err(Error::invalid_argument("FFmpeg returned an incomplete i16 sample"));
        }
        let len = std::mem::take(&mut self.filled);
        Ok((len != 0).then_some(&self.frame[..len]))
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
