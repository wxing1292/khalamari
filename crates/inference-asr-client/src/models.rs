use std::path::Path;
use std::sync::Arc;

use ct2rs::sys::ComputeType;
use ct2rs::sys::Config;
use ct2rs::sys::StorageView;
use ct2rs::sys::Whisper;
use inference_error::Error;
use inference_error::Result;
use ort::session::Session;
use ort::session::builder::AutoDevicePolicy;
use ort::value::Tensor;
use rustfft::Fft;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;
use tokenizers::Tokenizer;

use crate::audio::FRAME_SAMPLES;
use crate::audio::SAMPLE_RATE;
use crate::config::LanguageConfig;

pub const ALIGNMENT_STEP_SAMPLES: u64 = SAMPLE_RATE as u64 / 50;

fn model_error(error: impl std::fmt::Display) -> Error {
    Error::unavailable(format!("CPU speech model: {error}"))
}

/// Recurrent VAD state belongs to exactly one capture session.
pub struct Vad {
    session: Session,
    input: Tensor<f32>,
    h: Tensor<f32>,
    c: Tensor<f32>,
}

impl Vad {
    pub fn load() -> Result<Self> {
        let environment = ort::init().build().map_err(model_error)?;
        let mut session = Session::builder(&environment)
            .map_err(model_error)?
            .with_auto_device(AutoDevicePolicy::PreferCPU)
            .map_err(model_error)?
            .with_intra_threads(1)
            .map_err(model_error)?
            .commit_from_memory(include_bytes!("../assets/silero_vad_v6.onnx"))
            .map_err(model_error)?;
        let input =
            Tensor::from_array(([1, 64 + FRAME_SAMPLES], vec![0.0f32; 64 + FRAME_SAMPLES])).map_err(model_error)?;
        let h = Tensor::from_array(([1, 1, 128], vec![0.0f32; 128])).map_err(model_error)?;
        let c = Tensor::from_array(([1, 1, 128], vec![0.0f32; 128])).map_err(model_error)?;
        // Validate once. The input tensors stay zero; warmup outputs do not enter capture state.
        let outputs = session
            .run(ort::inputs![
                "input" => &input,
                "h" => &h,
                "c" => &c,
            ])
            .map_err(model_error)?;
        for (name, dimensions) in [
            ("speech_probs", &[1][..]),
            ("hn", &[1, 1, 128][..]),
            ("cn", &[1, 1, 128][..]),
        ] {
            let output = outputs
                .get(name)
                .expect("embedded Silero model must contain its declared outputs");
            let (shape, _) = output
                .try_extract_tensor::<f32>()
                .expect("embedded Silero outputs must be F32 tensors");
            assert_eq!(&shape[..], dimensions, "embedded Silero output {name}");
        }
        drop(outputs);
        Ok(Self { session, input, h, c })
    }

    pub fn probability(&mut self, pcm: &[i16]) -> Result<f32> {
        debug_assert!(pcm.len() <= FRAME_SAMPLES);
        let (_, input) = self.input.extract_tensor_mut();
        input.copy_within(FRAME_SAMPLES.., 0);
        input[64 + pcm.len()..].fill(0.0);
        for (output, &sample) in input[64..].iter_mut().zip(pcm) {
            *output = f32::from(sample) / 32768.0;
        }
        let outputs = self
            .session
            .run(ort::inputs![
                "input" => &self.input,
                "h" => &self.h,
                "c" => &self.c,
            ])
            .map_err(model_error)?;
        let (_, probability) = outputs["speech_probs"]
            .try_extract_tensor::<f32>()
            .expect("validated Silero probability output");
        let (_, h) = outputs["hn"]
            .try_extract_tensor::<f32>()
            .expect("validated Silero h output");
        let (_, c) = outputs["cn"]
            .try_extract_tensor::<f32>()
            .expect("validated Silero c output");
        self.h.extract_tensor_mut().1.copy_from_slice(h);
        self.c.extract_tensor_mut().1.copy_from_slice(c);
        Ok(probability[0])
    }
}

#[derive(Clone, Debug)]
pub struct AlignmentUnit {
    pub text: String,
    pub start_sample: usize,
    pub end_sample: usize,
    pub probability: f32,
}

/// CPU Whisper is an acoustic classifier and aligner. Qwen owns transcription text.
pub struct AcousticModel {
    whisper: Whisper,
    tokenizer: Tokenizer,
    mel: Mel,
    sot: usize,
    transcribe: usize,
}

impl AcousticModel {
    pub fn load(config: &LanguageConfig) -> Result<Self> {
        let path = if config.model.contains('/') {
            Path::new(&config.model).to_path_buf()
        } else {
            config.model_cache.join(format!("whisper-{}", config.model))
        };
        let whisper = Whisper::new(
            &path,
            Config {
                compute_type: ComputeType::INT8,
                num_threads_per_replica: 2,
                ..Config::default()
            },
        )
        .map_err(|error| {
            Error::unavailable(format!(
                "Whisper checkpoint `{}`: {error}; run `asr --setup-models` or select a prepared checkpoint with \
                 --lid-model",
                path.display()
            ))
        })?;
        if !whisper.is_multilingual() {
            return Err(Error::invalid_argument(
                "LID requires a multilingual Whisper checkpoint",
            ));
        }
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json")).map_err(model_error)?;
        let special = |text| {
            tokenizer
                .token_to_id(text)
                .map(|v| v as usize)
                .ok_or_else(|| Error::invalid_argument(format!("Whisper tokenizer lacks {text}")))
        };
        let sot = special("<|startoftranscript|>")?;
        let transcribe = special("<|transcribe|>")?;
        let mel = Mel::new(whisper.n_mels());
        Ok(Self {
            whisper,
            tokenizer,
            mel,
            sot,
            transcribe,
        })
    }

    pub fn language(&mut self, pcm: &[i16]) -> Result<(String, f32)> {
        let (features, frames) = self.mel.features(pcm);
        let mut padded = vec![0.0; self.whisper.n_mels() * 3000];
        let count = frames.min(3000);
        for bin in 0..self.whisper.n_mels() {
            padded[bin * 3000..bin * 3000 + count].copy_from_slice(&features[bin * frames..bin * frames + count]);
        }
        let (mut features, frames) = (padded, 3000);
        let view = StorageView::new(&[1, self.whisper.n_mels(), frames], &mut features, Default::default())
            .map_err(model_error)?;
        let detected = self.whisper.detect_language(&view).map_err(model_error)?;
        let best = &detected[0][0];
        let code = best.language.trim_start_matches("<|").trim_end_matches("|>");
        Ok((language_name(code).to_owned(), best.probability))
    }

    pub fn align(&mut self, pcm: &[i16], text: &str, language: &str) -> Result<Option<Vec<AlignmentUnit>>> {
        let tokens = self
            .tokenizer
            .encode(text, false)
            .map_err(model_error)?
            .get_ids()
            .to_vec();
        if tokens.len() + 3 >= 448 || pcm.len() < 1600 {
            return Ok(None);
        }
        let code = if text.chars().any(is_han) {
            "zh"
        } else {
            language_code(language)
        };
        let lang = self
            .tokenizer
            .token_to_id(&format!("<|{code}|>"))
            .ok_or_else(|| Error::invalid_argument("Whisper language token is missing"))? as usize;
        let (mut features, frames) = self.mel.features(pcm);
        let view = StorageView::new(&[1, self.whisper.n_mels(), frames], &mut features, Default::default())
            .map_err(model_error)?;
        let encoded = self.whisper.encode(&view, true).map_err(model_error)?;
        let token_ids = tokens.iter().map(|&id| id as usize).collect();
        let results = self
            .whisper
            .align(
                &encoded,
                &[self.sot, lang, self.transcribe],
                &[token_ids],
                &[pcm.len() / 160],
                7,
            )
            .map_err(model_error)?;
        let result = &results[0];
        let mut jumps = Vec::new();
        let mut previous = None;
        for step in &result.alignments {
            if previous != Some(step.token_x) {
                if step.token_x < 0 || step.frame_x < 0 {
                    return Err(Error::invalid_argument(
                        "Whisper returned a negative alignment coordinate",
                    ));
                }
                jumps.push(step.frame_x as usize * ALIGNMENT_STEP_SAMPLES as usize);
            }
            previous = Some(step.token_x);
        }
        if jumps.len() < tokens.len() + 1 || result.text_token_probs.len() < tokens.len() {
            return Err(Error::invalid_argument("Whisper returned incomplete token alignment"));
        }
        // Decode complete UTF-8 groups. Lexical splitting happens after alignment.
        let mut units = Vec::new();
        let mut begin = 0;
        let mut cumulative = String::new();
        for end in 1..=tokens.len() {
            let decoded = self.tokenizer.decode(&tokens[..end], false).map_err(model_error)?;
            if decoded.contains('\u{fffd}') {
                continue;
            }
            let Some(suffix) = decoded.strip_prefix(&cumulative) else {
                return Ok(None);
            };
            if suffix.is_empty() {
                continue;
            }
            let start_sample = jumps[begin];
            let end_sample = jumps[end];
            if end_sample > pcm.len() || start_sample > end_sample {
                return Ok(None);
            }
            let probability = result.text_token_probs[begin..end].iter().sum::<f32>() / (end - begin) as f32;
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                return Err(Error::invalid_argument("Whisper returned invalid token probabilities"));
            }
            units.push(AlignmentUnit {
                text: suffix.to_owned(),
                start_sample,
                end_sample,
                probability,
            });
            cumulative = decoded;
            begin = end;
        }
        if cumulative != text || begin != tokens.len() {
            return Ok(None);
        }
        Ok(Some(units))
    }
}

pub fn is_han(c: char) -> bool {
    ('\u{3400}'..='\u{9fff}').contains(&c)
}

const LANGUAGES: &[(&str, &str)] = &[
    ("zh", "Chinese"),
    ("en", "English"),
    ("yue", "Cantonese"),
    ("ar", "Arabic"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ko", "Korean"),
    ("ru", "Russian"),
    ("th", "Thai"),
    ("vi", "Vietnamese"),
    ("ja", "Japanese"),
    ("tr", "Turkish"),
    ("hi", "Hindi"),
    ("ms", "Malay"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("pl", "Polish"),
    ("cs", "Czech"),
    ("tl", "Filipino"),
    ("fa", "Persian"),
    ("el", "Greek"),
    ("ro", "Romanian"),
    ("hu", "Hungarian"),
    ("mk", "Macedonian"),
];
fn language_name(code: &str) -> &str {
    LANGUAGES.iter().find(|(c, _)| *c == code).map_or("", |(_, name)| name)
}

fn language_code(name: &str) -> &str {
    LANGUAGES
        .iter()
        .find(|(_, n)| *n == name)
        .map_or("en", |(code, _)| code)
}

/// Whisper's periodic Hann / reflected STFT / Slaney log-Mel contract.
/// Add 160 zero samples before the reflected STFT. LID pads the resulting features separately.
struct Mel {
    fft: Arc<dyn Fft<f32>>,
    scratch: Vec<Complex32>,
    filters: Vec<Vec<(usize, f32)>>,
    window: Vec<f32>,
}

impl Mel {
    fn new(bins: usize) -> Self {
        let hz = |mel: f64| {
            if mel < 15.0 {
                (200.0 / 3.0) * mel
            } else {
                1000.0 * ((6.4f64.ln() / 27.0) * (mel - 15.0)).exp()
            }
        };
        let max_mel = 15.0 + 8.0f64.ln() / (6.4f64.ln() / 27.0);
        let points: Vec<_> = (0..bins + 2)
            .map(|i| hz(max_mel * i as f64 / (bins + 1) as f64))
            .collect();
        let filters = (0..bins)
            .map(|i| {
                (0..201)
                    .filter_map(|j| {
                        let frequency = j as f64 * 40.0;
                        let weight = (((frequency - points[i]) / (points[i + 1] - points[i]))
                            .min((points[i + 2] - frequency) / (points[i + 2] - points[i + 1]))
                            .max(0.0)
                            * 2.0
                            / (points[i + 2] - points[i])) as f32;
                        (weight != 0.0).then_some((j, weight))
                    })
                    .collect()
            })
            .collect();
        let fft = FftPlanner::new().plan_fft_forward(400);
        let scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
        Self {
            fft,
            scratch,
            filters,
            window: (0..400)
                .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / 400.0).cos()) as f32)
                .collect(),
        }
    }

    fn features(&mut self, pcm: &[i16]) -> (Vec<f32>, usize) {
        let frames = (pcm.len() + 160) / 160;
        let mut features = vec![-10.0f32; self.filters.len() * frames];
        let mut frame = vec![Complex32::default(); 400];
        let mut power = [0.0; 201];
        for t in 0..frames {
            for (j, value) in frame.iter_mut().enumerate() {
                let index = (t * 160 + j) as isize - 200;
                let padded_samples = pcm.len() + 160;
                let period = 2 * (padded_samples - 1);
                let index = index.rem_euclid(period as isize) as usize;
                let index = if index < padded_samples { index } else { period - index };
                let sample = if index < pcm.len() {
                    f32::from(pcm[index]) / 32768.0
                } else {
                    0.0
                };
                *value = Complex32::new(sample * self.window[j], 0.0);
            }
            self.fft.process_with_scratch(&mut frame, &mut self.scratch);
            for (value, frequency) in power.iter_mut().zip(&frame[..201]) {
                *value = frequency.norm_sqr();
            }
            for (bin, filter) in self.filters.iter().enumerate() {
                features[bin * frames + t] = filter
                    .iter()
                    .map(|&(index, weight)| weight * power[index])
                    .sum::<f32>()
                    .max(1e-10)
                    .log10();
            }
        }
        let floor = features.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.0;
        for value in &mut features {
            *value = (value.max(floor) + 4.0) / 4.0;
        }
        // CT2's Whisper encoder consumes at most 3000 frames.
        if frames > 3000 {
            for bin in 0..self.filters.len() {
                features.copy_within(bin * frames..bin * frames + 3000, bin * 3000);
            }
            features.truncate(self.filters.len() * 3000);
            (features, 3000)
        } else {
            (features, frames)
        }
    }
}

#[cfg(test)]
#[path = "models_test.rs"]
mod tests;
