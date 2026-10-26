use std::collections::VecDeque;

use inference_error::Result;
use serde::Serialize;

use crate::audio::quiet_cut;
use crate::config::LanguageConfig;
use crate::config::samples;
use crate::models::AcousticModel;

pub struct AudioLangRecognitionReq<'a> {
    pub segment: u64,
    pub start_sample: u64,
    pub end_sample: u64,
    /// The segment owner supplies exactly the samples in start_sample..end_sample.
    pub audio: &'a [i16],
}

pub struct AudioLangRecognitionResp {
    pub ranges: Vec<LanguageRange>,
    pub windows: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct LanguageRange {
    pub start_sample: u64,
    pub end_sample: u64,
    pub language_hint: String,
    pub chinese_observed: bool,
}

struct Observation {
    segment: u64,
    start: u64,
    end: u64,
    language: String,
    score: f32,
}

struct Cell {
    start: usize,
    end: usize,
    language: String,
    confident: bool,
}

pub struct LanguageAnalyzer {
    hop: usize,
    context: usize,
    minimum: usize,
    confidence: f32,
    capacity: usize,
    observations: VecDeque<Observation>,
}

impl LanguageAnalyzer {
    pub fn new(config: &LanguageConfig) -> Self {
        Self {
            hop: samples(config.hop_seconds),
            context: samples(config.context_seconds),
            minimum: samples(config.min_seconds),
            confidence: config.confidence,
            capacity: config.cache_windows,
            observations: VecDeque::new(),
        }
    }

    fn observe(
        &mut self,
        model: &mut AcousticModel,
        job: &AudioLangRecognitionReq<'_>,
        start: usize,
        end: usize,
        windows: &mut u64,
    ) -> Result<(String, f32)> {
        let a = job.start_sample + start as u64;
        let b = job.start_sample + end as u64;
        if let Some(old) = self
            .observations
            .iter()
            .find(|old| old.segment == job.segment && old.start == a && old.end == b)
        {
            return Ok((old.language.clone(), old.score));
        }
        *windows += 1;
        let (language, score) = model.language(&job.audio[start..end])?;
        if self.observations.len() == self.capacity {
            self.observations.pop_front();
        }
        self.observations.push_back(Observation {
            segment: job.segment,
            start: a,
            end: b,
            language: language.clone(),
            score,
        });
        Ok((language, score))
    }

    pub fn plan(
        &mut self,
        model: &mut AcousticModel,
        job: &AudioLangRecognitionReq<'_>,
    ) -> Result<AudioLangRecognitionResp> {
        let mut windows = 0;
        let count = (job.end_sample - job.start_sample) as usize;
        let hop = self.hop;
        let context = self.context;
        let minimum = self.minimum;
        let mut cells = Vec::new();
        for start in (0..count).step_by(hop) {
            let end = (start + hop).min(count);
            let lo = start.saturating_sub(context);
            let hi = (end + context).min(count);
            let (language, score) = if hi - lo < minimum {
                (String::new(), 0.0)
            } else {
                self.observe(model, job, lo, hi, &mut windows)?
            };
            cells.push(Cell {
                start,
                end,
                confident: score >= self.confidence && !language.is_empty(),
                language,
            });
        }
        let mut cuts = vec![0];
        for pair in cells.windows(2) {
            let previous = if pair[0].confident {
                pair[0].language.as_str()
            } else {
                ""
            };
            let next = if pair[1].confident {
                pair[1].language.as_str()
            } else {
                ""
            };
            if previous != next
                && let Some(cut) = quiet_cut(job.audio, pair[1].start, *cuts.last().unwrap(), 12000, minimum)
            {
                cuts.push(cut);
            }
        }
        cuts.push(count);
        let mut ranges: Vec<LanguageRange> = Vec::new();
        for cut in cuts.windows(2) {
            let mut relevant = cells.iter().filter(|c| c.start < cut[1] && c.end > cut[0]);
            let mut labels: Vec<_> = relevant
                .clone()
                .filter(|c| c.confident)
                .map(|c| c.language.as_str())
                .collect();
            labels.sort_unstable();
            labels.dedup();
            let uncertain = relevant.clone().any(|c| !c.confident);
            let mut language = if labels.len() == 1 && !uncertain {
                labels[0].to_owned()
            } else {
                String::new()
            };
            if labels.len() == 1 && uncertain && cut[1] - cut[0] >= minimum {
                let (candidate, score) = self.observe(model, job, cut[0], cut[1], &mut windows)?;
                if candidate == labels[0] && score >= self.confidence {
                    language = candidate;
                }
            }
            let chinese = relevant.any(|c| c.confident && c.language == "Chinese");
            let start = job.start_sample + cut[0] as u64;
            let end = job.start_sample + cut[1] as u64;
            if let Some(last) = ranges.last_mut().filter(|r| r.language_hint == language) {
                last.end_sample = end;
                last.chinese_observed |= chinese;
            } else {
                ranges.push(LanguageRange {
                    start_sample: start,
                    end_sample: end,
                    language_hint: language,
                    chinese_observed: chinese,
                });
            }
        }
        Ok(AudioLangRecognitionResp { ranges, windows })
    }
}
