mod audio;
mod tokenizer;

use std::path::Path;
use std::sync::Arc;

use inference_executor_core::model::qwen::v3_asr::QWEN3_ASR_AUDIO_RESOURCE_TYPE;
use inference_executor_core::model::qwen::v3_asr::Qwen3ASRModelConfig;
use inference_executor_metal::model::qwen::v3_asr::AudioSourceRegistration;
use inference_executor_metal::model::qwen::v3_asr::Qwen3ASRAudioProcessor;
use inference_runtime_core::Error;
use inference_runtime_core::Result;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::runtime::Resource;
use inference_runtime_core::runtime::ResourceID;
use inference_runtime_core::runtime::ResourcePlacement;
use inference_runtime_core::runtime::SymbolicResource;
use inference_runtime_core::runtime::Token;
use inference_runtime_core::tokenizer::Tokenizer;
use inference_runtime_core::tokenizer::huggingface::HFTokenizer;

use self::audio::Qwen3ASRAudioPreprocessor;
use crate::api::Inference;
use crate::api::decode::DecodeEvent;
use crate::api::decode::DecodeRequest;

const ASR_TEXT_TOKEN: &str = "<asr_text>";

pub struct Qwen3ASRService {
    config: Qwen3ASRModelConfig,
    tokenizer: HFTokenizer,
    audio_preprocessor: Qwen3ASRAudioPreprocessor,
    audio_processor: Arc<Qwen3ASRAudioProcessor>,
}

pub struct PreparedTranscription {
    tokens: Vec<Token>,
    resource: Resource,
    placement: ResourcePlacement,
    requested_language: Option<String>,
    source_registration: AudioSourceRegistration,
}

#[derive(Debug, Eq, PartialEq, serde::Serialize)]
pub struct Transcription {
    pub text: String,
    pub language: String,
}

impl Qwen3ASRService {
    pub fn load(
        model_dir: impl AsRef<Path>,
        config: Qwen3ASRModelConfig,
        audio_processor: Arc<Qwen3ASRAudioProcessor>,
    ) -> Result<Self> {
        let tokenizer = tokenizer::load(model_dir.as_ref())?;
        for (token, expected_id) in [
            ("<|audio_start|>", config.audio_start_token_id),
            ("<|audio_end|>", config.audio_end_token_id),
            ("<|audio_pad|>", config.audio_token_id),
        ] {
            let actual_id = tokenizer
                .token(token)
                .ok_or_else(|| Error::internal(format!("Qwen3-ASR tokenizer is missing {token:?}")))?
                .value();
            if actual_id != expected_id {
                return Err(Error::internal(format!(
                    "Qwen3-ASR tokenizer maps {token:?} to {actual_id}; expected {expected_id}"
                )));
            }
        }
        if tokenizer.token(ASR_TEXT_TOKEN).is_none() {
            return Err(Error::internal(format!(
                "Qwen3-ASR tokenizer is missing {ASR_TEXT_TOKEN:?}"
            )));
        }

        let audio_preprocessor = Qwen3ASRAudioPreprocessor::new(config.preprocessor.clone());
        Ok(Self {
            config,
            tokenizer,
            audio_preprocessor,
            audio_processor,
        })
    }

    pub fn prepare_wav(
        &self,
        wav_bytes: &[u8],
        language: Option<&str>,
        context: Option<&str>,
    ) -> Result<PreparedTranscription> {
        let requested_language = self.normalize_language(language)?;
        let source = self.audio_preprocessor.prepare_wav(wav_bytes)?;
        let num_resource_tokens = source.num_resource_tokens();
        let (tokens, audio_token_index) = tokenize_prompt(
            &self.tokenizer,
            &self.config,
            context.unwrap_or_default(),
            requested_language.as_deref(),
            num_resource_tokens,
        )?;

        let resource_id = ResourceID::new(QWEN3_ASR_AUDIO_RESOURCE_TYPE);
        let (uri, source_registration) = self.audio_processor.register_source(resource_id, source);
        let resource = Resource::Symbolic(SymbolicResource::new(resource_id, uri));
        let placement = ResourcePlacement::new(
            resource_id,
            vec![(audio_token_index, 0, num_resource_tokens)],
            tokens.len(),
        );
        Ok(PreparedTranscription {
            tokens,
            resource,
            placement,
            requested_language,
            source_registration,
        })
    }

    pub async fn transcribe<const N: usize, const L: usize, const P: usize>(
        &self,
        inference: &Inference<N, L, P>,
        prepared: PreparedTranscription,
        max_output_tokens: usize,
    ) -> Result<Transcription> {
        let PreparedTranscription {
            tokens,
            resource,
            placement,
            requested_language,
            source_registration: _source_registration,
        } = prepared;
        let sampling = SamplingConfig {
            max_sampled_tokens: max_output_tokens,
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            ..SamplingConfig::default()
        };
        let request = DecodeRequest::new(tokens, vec![(resource, placement)], sampling)?;
        let mut response = inference.create_session(request)?;
        let mut output_tokens = vec![];
        loop {
            match response.recv_event().await? {
                DecodeEvent::TokenProbs(token_probs) => output_tokens.extend(token_probs.tokens),
                DecodeEvent::Completed { .. } => {
                    let output = self.tokenizer.decode_without_special_tokens(&output_tokens)?;
                    return Ok(parse_transcription(output, requested_language));
                },
            }
        }
    }

    fn normalize_language(&self, language: Option<&str>) -> Result<Option<String>> {
        let Some(language) = language.map(str::trim).filter(|language| !language.is_empty()) else {
            return Ok(None);
        };
        self.config
            .support_languages
            .iter()
            .find(|supported| supported.eq_ignore_ascii_case(language))
            .cloned()
            .map(Some)
            .ok_or_else(|| Error::invalid_argument(format!("unsupported Qwen3-ASR language {language:?}")))
    }
}

fn tokenize_prompt(
    tokenizer: &HFTokenizer,
    config: &Qwen3ASRModelConfig,
    context: &str,
    language: Option<&str>,
    num_resource_tokens: usize,
) -> Result<(Vec<Token>, usize)> {
    let prefix = format!(
        concat!(
            "<|im_start|>system\n{}<|im_end|>\n",
            "<|im_start|>user\n<|audio_start|>"
        ),
        context
    );
    let mut tokens = tokenizer.encode(&prefix)?;
    let audio_token_index = tokens.len();
    tokens.resize(tokens.len() + num_resource_tokens, Token::new(config.audio_token_id));

    let assistant_prefix = language
        .map(|language| format!("language {language}{ASR_TEXT_TOKEN}"))
        .unwrap_or_default();
    let suffix = format!(
        concat!("<|audio_end|><|im_end|>\n", "<|im_start|>assistant\n{}"),
        assistant_prefix
    );
    tokens.extend(tokenizer.encode(&suffix)?);
    Ok((tokens, audio_token_index))
}

fn parse_transcription(output: String, requested_language: Option<String>) -> Transcription {
    let output = output.trim();
    if output.is_empty() {
        return Transcription {
            text: String::new(),
            language: String::new(),
        };
    }
    if let Some(language) = requested_language {
        return Transcription {
            text: output.to_string(),
            language,
        };
    }

    let Some((metadata, text)) = output.split_once(ASR_TEXT_TOKEN) else {
        return Transcription {
            text: output.to_string(),
            language: String::new(),
        };
    };
    let text = text.trim().to_string();
    if metadata.to_ascii_lowercase().contains("language none") {
        return Transcription {
            text,
            language: String::new(),
        };
    }
    let language = metadata
        .lines()
        .map(str::trim)
        .find_map(|line| {
            line.get(.."language ".len())
                .filter(|prefix| prefix.eq_ignore_ascii_case("language "))
                .map(|_| line["language ".len()..].trim())
                .filter(|language| !language.is_empty())
        })
        .map(normalize_detected_language)
        .unwrap_or_default();
    Transcription { text, language }
}

fn normalize_detected_language(language: &str) -> String {
    let mut characters = language.chars();
    let first = characters
        .next()
        .expect("a detected Qwen3-ASR language must not be empty");
    first
        .to_uppercase()
        .chain(characters.flat_map(char::to_lowercase))
        .collect()
}

#[cfg(test)]
#[path = "asr_test.rs"]
mod tests;
