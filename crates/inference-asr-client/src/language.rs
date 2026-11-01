use std::collections::VecDeque;
use std::ops::Range;

use inference_error::Result;
use serde::Serialize;

use crate::analysis::AudioAnalysisInput;
use crate::analysis::AudioAnalysisUpdate;
use crate::analysis::AudioSpan;
use crate::audio::quiet_cut;
use crate::config::LanguageConfig;
use crate::config::ModelConfig;
use crate::config::samples;
use crate::models::AcousticModel;
use crate::models::timed_words;
use crate::recording::AudioRecordingReader;
use crate::transcript::Word;

pub trait AudioLanguageRecognizer: Send {
    fn start(&mut self, recording: AudioRecordingReader);

    fn append(&mut self, range: Range<u64>);

    fn recognize(&mut self) -> Result<Vec<AudioAnalysisUpdate<LanguageRange>>>;

    fn stop(&mut self);
}

#[derive(Clone, Debug, Serialize)]
pub struct LanguageRange {
    #[serde(rename = "start_sample")]
    pub start: u64,
    #[serde(rename = "end_sample")]
    pub end: u64,
    /// Empty means unknown. It selects automatic ASR language detection.
    pub language_hint: String,
    pub chinese_observed: bool,
}

impl AudioSpan for LanguageRange {
    fn range(&self) -> Range<u64> {
        self.start..self.end
    }

    fn set_range(&mut self, range: Range<u64>) {
        self.start = range.start;
        self.end = range.end;
    }
}

struct Observation {
    start: u64,
    end: u64,
    language: String,
    score: f32,
}

/// Shared Whisper weights, continuous LID state and fixed-text alignment for one recording.
pub struct Acoustics {
    model: AcousticModel,
    input: Option<AudioAnalysisInput>,
    processed: u64,
    hop: usize,
    context: usize,
    minimum: usize,
    confidence: f32,
    capacity: usize,
    observations: VecDeque<Observation>,
}

impl Acoustics {
    pub fn load(config: &LanguageConfig, models: &ModelConfig) -> Result<Self> {
        Ok(Self {
            model: AcousticModel::load(models, config)?,
            input: None,
            processed: 0,
            hop: samples(config.hop_seconds),
            context: samples(config.context_seconds),
            minimum: samples(config.min_seconds),
            confidence: config.confidence,
            capacity: config.cache_windows,
            observations: VecDeque::new(),
        })
    }

    pub fn align(&mut self, audio: &[i16], text: &str, language: &str, start: u64) -> Result<Option<Vec<Word>>> {
        if text.is_empty() {
            return Ok(Some(Vec::new()));
        }
        Ok(self
            .model
            .align(audio, text, language)?
            .and_then(|units| timed_words(text, &units, start)))
    }
}

impl AudioLanguageRecognizer for Acoustics {
    fn start(&mut self, recording: AudioRecordingReader) {
        self.input = Some(AudioAnalysisInput::new(recording));
        self.processed = 0;
    }

    fn append(&mut self, range: Range<u64>) {
        self.input.as_mut().unwrap().append(range);
    }

    fn stop(&mut self) {
        self.input.as_mut().unwrap().stop();
    }

    fn recognize(&mut self) -> Result<Vec<AudioAnalysisUpdate<LanguageRange>>> {
        let input = self.input.as_ref().unwrap();
        let start = self.processed;
        let mut spans = Vec::new();
        while self.processed < input.end {
            let end = self.processed + self.hop as u64;
            if !input.stopped && end + self.context as u64 > input.end {
                break;
            }
            let end = end.min(input.end);
            let lo = self.processed.saturating_sub(self.context as u64);
            let hi = (end + self.context as u64).min(input.end);
            let (language, score) = if hi - lo < self.minimum as u64 {
                (String::new(), 0.0)
            } else if let Some(old) = self.observations.iter().find(|old| old.start == lo && old.end == hi) {
                (old.language.clone(), old.score)
            } else {
                let audio = input.recording.read(lo..hi)?;
                let (language, score) = self.model.language(&audio)?;
                if self.observations.len() == self.capacity {
                    self.observations.pop_front();
                }
                self.observations.push_back(Observation {
                    start: lo,
                    end: hi,
                    language: language.clone(),
                    score,
                });
                (language, score)
            };
            let confident = score >= self.confidence && !language.is_empty();
            spans.push(LanguageRange {
                start: self.processed,
                end,
                chinese_observed: confident && language == "Chinese",
                language_hint: if confident { language } else { String::new() },
            });
            self.processed = end;
        }
        if start == self.processed {
            return Ok(Vec::new());
        }
        let range = start..self.processed;
        Ok(vec![
            AudioAnalysisUpdate::Replace {
                range: range.clone(),
                spans,
            },
            AudioAnalysisUpdate::Seal { range },
        ])
    }
}

/// Language evidence and ASR cuts have different ranges. Require a quiet cut before splitting audio.
pub fn plan(audio: &[i16], start: u64, observations: &[LanguageRange], minimum: usize) -> Vec<LanguageRange> {
    let mut cuts = vec![0];
    for pair in observations.windows(2) {
        if pair[0].language_hint != pair[1].language_hint
            && let Some(cut) = quiet_cut(
                audio,
                (pair[1].start - start) as usize,
                *cuts.last().unwrap(),
                12000,
                minimum,
            )
        {
            cuts.push(cut);
        }
    }
    cuts.push(audio.len());
    let mut ranges: Vec<LanguageRange> = Vec::new();
    for cut in cuts.windows(2) {
        let lo = start + cut[0] as u64;
        let hi = start + cut[1] as u64;
        let relevant = observations.iter().filter(|r| r.start < hi && lo < r.end);
        let first = relevant.clone().next().map_or("", |r| r.language_hint.as_str());
        let language = if !first.is_empty() && relevant.clone().all(|r| r.language_hint == first) {
            first
        } else {
            ""
        };
        let chinese = relevant.clone().any(|r| r.chinese_observed);
        if let Some(last) = ranges.last_mut().filter(|r| r.language_hint == language) {
            last.end = hi;
            last.chinese_observed |= chinese;
        } else {
            ranges.push(LanguageRange {
                start: lo,
                end: hi,
                language_hint: language.to_owned(),
                chinese_observed: chinese,
            });
        }
    }
    ranges
}
