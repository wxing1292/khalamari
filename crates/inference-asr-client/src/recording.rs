use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use inference_error::Error;
use inference_error::Result;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::watch;

/// The only writer of the recording sample clock. The temporary PCM file survives errors.
/// Publication follows the complete write. Dropping the writer seals the readable prefix.
pub struct AudioRecordingWriter {
    file: Arc<RecordingFile>,
    published: watch::Sender<u64>,
}

/// A reader has its own notification cursor. Positional reads never share a file seek position.
#[derive(Clone)]
pub struct AudioRecordingReader {
    file: Arc<RecordingFile>,
    published: watch::Receiver<u64>,
}

struct RecordingFile {
    pcm: File,
    path: PathBuf,
}

impl AudioRecordingWriter {
    pub fn create() -> Result<(Self, tempfile::TempPath)> {
        let (file, path) = tempfile::Builder::new()
            .prefix("khalamari-recording-")
            .suffix(".s16le")
            .tempfile()
            .map_err(recording_error)?
            .into_parts();
        Ok((
            Self {
                file: Arc::new(RecordingFile {
                    pcm: file,
                    path: path.to_path_buf(),
                }),
                published: watch::channel(0).0,
            },
            path,
        ))
    }

    pub fn reader(&self) -> AudioRecordingReader {
        AudioRecordingReader {
            file: self.file.clone(),
            published: self.published.subscribe(),
        }
    }

    /// FFmpeg has validated complete little-endian i16 samples.
    pub fn append(&mut self, bytes: &[u8]) -> Result<()> {
        let start = *self.published.borrow();
        let end = start
            .checked_add((bytes.len() / 2) as u64)
            .filter(|end| *end <= i64::MAX as u64 / 2)
            .ok_or_else(|| Error::resource_exhausted("recording exceeds the file offset domain"))?;
        self.file.pcm.write_all_at(bytes, start * 2).map_err(recording_error)?;
        self.published.send_replace(end);
        Ok(())
    }

    pub fn stop(self) -> Result<()> {
        if *self.published.borrow() == 0 {
            return Err(Error::invalid_argument("recording contains no audio"));
        }
        self.file.pcm.sync_all().map_err(recording_error)
    }
}

impl AudioRecordingReader {
    pub fn path(&self) -> &Path {
        &self.file.path
    }

    pub fn end(&self) -> u64 {
        *self.published.borrow()
    }

    pub fn is_sealed(&self) -> bool {
        self.published.has_changed().is_err()
    }

    fn bytes(&self, range: Range<u64>) -> Result<Vec<u8>> {
        // This boundary accepts independently scheduled readers, including retained older ranges.
        if range.start > range.end || range.end > self.end() {
            return Err(Error::invalid_argument(
                "recording read is outside the published sample range",
            ));
        }
        let num_bytes = usize::try_from((range.end - range.start) * 2)
            .map_err(|_| Error::resource_exhausted("recording read exceeds addressable memory"))?;
        let mut bytes = vec![0; num_bytes];
        self.file
            .pcm
            .read_exact_at(&mut bytes, range.start * 2)
            .map_err(recording_error)?;
        Ok(bytes)
    }

    pub fn read(&self, range: Range<u64>) -> Result<Vec<i16>> {
        Ok(self
            .bytes(range)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|sample| i16::from_le_bytes(*sample))
            .collect())
    }

    /// Returns the published extent. A shorter extent means the writer has sealed the file.
    /// The caller owns its sample cursor, including when it cancels a pending wait.
    pub async fn available(&mut self, until: u64) -> u64 {
        let _ = self.published.wait_for(|end| *end >= until).await;
        self.end()
    }
}

/// Compression follows the published prefix independently of capture and inference.
/// The destination becomes visible only after the encoder closes a complete file.
pub async fn archive(mut recording: AudioRecordingReader, ffmpeg: &str, destination: &Path) -> Result<()> {
    let directory = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let output = tempfile::Builder::new()
        .prefix(".khalamari-recording-")
        .tempfile_in(directory)
        .map_err(recording_error)?;
    let wav = destination.extension().is_some_and(|extension| extension == "wav");
    let mut command = Command::new(ffmpeg);
    command.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-c:a",
        if wav { "pcm_s16le" } else { "flac" },
        "-f",
        if wav { "wav" } else { "flac" },
    ]);
    if wav {
        command.args(["-rf64", "auto"]);
    }
    let mut child = command
        .arg(output.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .map_err(recording_error)?;
    let mut input = child.stdin.take().unwrap();
    let mut cursor = 0;
    loop {
        let end = recording.available(cursor + 16_384).await.min(cursor + 16_384);
        if end == cursor {
            break;
        }
        input
            .write_all(&recording.bytes(cursor..end)?)
            .await
            .map_err(recording_error)?;
        cursor = end;
    }
    drop(input);
    let status = child.wait().await.map_err(recording_error)?;
    if !status.success() {
        return Err(Error::unavailable(format!("recording encoder exited with {status}")));
    }
    if cursor == 0 {
        return Err(Error::invalid_argument("recording contains no audio"));
    }
    output.as_file().sync_all().map_err(recording_error)?;
    output.persist_noclobber(destination).map_err(recording_error)?;
    Ok(())
}

pub fn file_stem() -> Result<String> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(recording_error)?;
    let seconds = libc::time_t::try_from(now.as_secs()).map_err(recording_error)?;
    let mut calendar = std::mem::MaybeUninit::<libc::tm>::uninit();
    // localtime_r initializes the caller-owned value and does not use a shared tm buffer.
    let calendar = unsafe {
        if libc::localtime_r(&seconds, calendar.as_mut_ptr()).is_null() {
            return Err(Error::unavailable(
                "recording start time is outside the calendar domain",
            ));
        }
        calendar.assume_init()
    };
    Ok(format!(
        "asr-{:04}{:02}{:02}-{:02}{:02}{:02}-{:09}",
        calendar.tm_year + 1900,
        calendar.tm_mon + 1,
        calendar.tm_mday,
        calendar.tm_hour,
        calendar.tm_min,
        calendar.tm_sec,
        now.subsec_nanos()
    ))
}

fn recording_error(error: impl std::fmt::Display) -> Error {
    Error::unavailable(format!("recording: {error}"))
}

#[cfg(test)]
#[path = "recording_test.rs"]
mod tests;
