use std::collections::VecDeque;
use std::sync::Arc;

use bytes::Bytes;
use inference_runtime_core::Error;
use inference_runtime_core::Result;

use crate::api::Inference;
use crate::asr::Qwen3ASRService;
use crate::asr::Transcription;

pub const MAX_AUDIO_BODY_BYTES: usize = 32 * 1024 * 1024;
const MAX_OUTPUT_TOKENS: usize = 512;
const MAX_AUDIO_RANGE_SAMPLES: u64 = 30 * 16_000;

/// Connection-owned canonical PCM. Snapshots keep their byte storage alive after release.
#[derive(Default)]
pub struct AudioBuffer {
    chunks: VecDeque<(u64, Bytes)>,
    end: u64,
    released: u64,
    bytes: usize,
}

impl AudioBuffer {
    pub fn append(&mut self, start: u64, pcm: Bytes) -> Result<()> {
        if pcm.is_empty()
            || pcm.len() > MAX_AUDIO_RANGE_SAMPLES as usize * 2
            || !pcm.len().is_multiple_of(2)
            || start < self.end
        {
            return Err(Error::invalid_argument(
                "PCM appends must be nonempty, aligned, ordered and disjoint",
            ));
        }
        let end = start
            .checked_add((pcm.len() / 2) as u64)
            .ok_or_else(|| Error::invalid_argument("PCM sample range exceeds u64"))?;
        if pcm.len() > MAX_AUDIO_BODY_BYTES - self.bytes {
            return Err(Error::resource_exhausted(
                "audio stream PCM cache exceeds 32 MiB; release older ranges",
            ));
        }
        self.bytes += pcm.len();
        self.end = end;
        self.chunks.push_back((start, pcm));
        Ok(())
    }

    pub fn release(&mut self, before: u64) -> Result<()> {
        if before < self.released || before > self.end {
            return Err(Error::invalid_argument(
                "PCM release watermark must advance within uploaded audio",
            ));
        }
        while let Some((start, pcm)) = self.chunks.front_mut() {
            let end = *start + (pcm.len() / 2) as u64;
            if end <= before {
                self.bytes -= pcm.len();
                self.chunks.pop_front();
            } else {
                if *start < before {
                    let bytes = (before - *start) as usize * 2;
                    *pcm = pcm.slice(bytes..);
                    *start = before;
                    self.bytes -= bytes;
                }
                break;
            }
        }
        self.released = before;
        Ok(())
    }

    pub fn snapshot(&self, start: u64, end: u64) -> Result<Vec<Bytes>> {
        if start < self.released || start >= end || end - start > MAX_AUDIO_RANGE_SAMPLES {
            return Err(Error::invalid_argument(
                "ASR range must be retained, nonempty and at most 30 seconds",
            ));
        }
        let mut cursor = start;
        let mut result = Vec::new();
        for (base, pcm) in &self.chunks {
            let tail = *base + (pcm.len() / 2) as u64;
            if tail <= cursor {
                continue;
            }
            if *base > cursor {
                break;
            }
            let next = tail.min(end);
            result.push(pcm.slice((cursor - *base) as usize * 2..(next - *base) as usize * 2));
            cursor = next;
            if cursor == end {
                return Ok(result);
            }
        }
        Err(Error::invalid_argument("ASR range contains missing or unuploaded PCM"))
    }

    pub fn buffered_samples(&self) -> u64 {
        (self.bytes / 2) as u64
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
        self.bytes = 0;
        self.released = self.end;
    }
}

/// Owns one transport-independent WAV transcription request.
pub struct AudioTranscribeReq {
    wav: Vec<u8>,
    language: Option<String>,
    context: Option<String>,
}

impl AudioTranscribeReq {
    pub fn new(wav: Vec<u8>, model: &str, language: Option<String>, context: Option<String>) -> Result<Self> {
        if model.trim().is_empty() {
            return Err(Error::invalid_argument("model must not be empty"));
        }
        if wav.is_empty() || wav.len() > MAX_AUDIO_BODY_BYTES {
            return Err(Error::invalid_argument(
                "transcription requires a nonempty WAV of at most 32 MiB",
            ));
        }
        Ok(Self { wav, language, context })
    }
}

/// HTTP and gRPC share audio preparation, model submission, and response parsing.
#[derive(Clone)]
pub struct AudioTranscriber<const N: usize, const L: usize, const P: usize> {
    inference: Arc<Inference<N, L, P>>,
    asr: Arc<Qwen3ASRService>,
}

impl<const N: usize, const L: usize, const P: usize> AudioTranscriber<N, L, P> {
    pub fn new(inference: Arc<Inference<N, L, P>>, asr: Arc<Qwen3ASRService>) -> Self {
        Self { inference, asr }
    }

    pub async fn transcribe(&self, request: AudioTranscribeReq) -> Result<Transcription> {
        let asr = self.asr.clone();
        let prepared = tokio::task::spawn_blocking(move || {
            asr.prepare_wav(&request.wav, request.language.as_deref(), request.context.as_deref())
        })
        .await
        .expect("audio preparation task must not panic")?;
        self.asr.transcribe(&self.inference, prepared, MAX_OUTPUT_TOKENS).await
    }

    pub async fn transcribe_pcm(
        &self,
        chunks: Vec<Bytes>,
        language: Option<String>,
        context: Option<String>,
    ) -> Result<Transcription> {
        let asr = self.asr.clone();
        let prepared = tokio::task::spawn_blocking(move || {
            let samples = chunks
                .iter()
                .flat_map(|pcm| pcm.as_chunks::<2>().0.iter())
                .map(|bytes| i16::from_le_bytes(*bytes));
            asr.prepare_pcm(samples, language.as_deref(), context.as_deref())
        })
        .await
        .expect("audio preparation task must not panic")?;
        self.asr.transcribe(&self.inference, prepared, MAX_OUTPUT_TOKENS).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_buffer_ranges() {
        let mut buffer = AudioBuffer::default();
        let pcm = [1i16, 2, 3, 4]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect::<Vec<_>>();
        buffer.append(10, pcm.into()).unwrap();
        let snapshot = buffer.snapshot(11, 14).unwrap();
        buffer.release(12).unwrap();
        assert_eq!(buffer.buffered_samples(), 2);
        assert!(buffer.snapshot(11, 14).is_err());
        buffer.append(20, vec![5, 0, 6, 0].into()).unwrap();
        assert!(buffer.snapshot(13, 21).is_err());
        assert!(buffer.append(21, vec![0, 0].into()).is_err());
        buffer.release(22).unwrap();
        assert_eq!(buffer.buffered_samples(), 0);
        let actual: Vec<_> = snapshot
            .iter()
            .flat_map(|chunk| chunk.as_chunks::<2>().0.iter())
            .map(|bytes| i16::from_le_bytes(*bytes))
            .collect();
        assert_eq!(actual, [2, 3, 4]);
    }

    #[test]
    fn test_request_invalid_input() {
        for (wav, model) in [
            (vec![], "asr"),
            (vec![1], " "),
            (vec![0; MAX_AUDIO_BODY_BYTES + 1], "asr"),
        ] {
            assert!(matches!(
                AudioTranscribeReq::new(wav, model, None, None),
                Err(Error::InvalidArgument(_))
            ));
        }
    }
}
