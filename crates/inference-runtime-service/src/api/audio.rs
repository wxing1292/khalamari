use std::sync::Arc;

use inference_runtime_core::Error;
use inference_runtime_core::Result;

use crate::api::Inference;
use crate::asr::Qwen3ASRService;
use crate::asr::Transcription;

pub const MAX_AUDIO_BODY_BYTES: usize = 32 * 1024 * 1024;
const MAX_OUTPUT_TOKENS: usize = 512;

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
