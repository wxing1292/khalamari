use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use inference_error::Result;
use serde::Serialize;
use serde::Serializer;
use tokio::sync::watch;

use crate::analysis::AudioProcessor;
use crate::audio::SAMPLE_RATE;
use crate::config::Config;
use crate::config::samples;
use crate::language::Acoustics;
use crate::language::LanguageRange;
use crate::models::is_han;
use crate::overlap::stitch;
use crate::segment::AudioSegmentTranscribeReq;
use crate::service::AudioTranscribeResp;
use crate::service::stream::AudioStreamClient;
use crate::service::stream::AudioTransportStats;
use crate::transcript::AudioTranscript;
use crate::transcript::Overlap;
use crate::transcript::Span;
use crate::transcript::Word;

#[derive(Clone, Debug, Default, Serialize)]
pub struct Metrics {
    pub wait_seconds: f64,
    pub processing_seconds: f64,
    pub asr_requests: u64,
    #[serde(rename = "submitted_audio_seconds", serialize_with = "serialize_audio_seconds")]
    pub submitted_audio_samples: u64,
    pub result_cache_hits: u64,
    pub alignment_calls: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct AudioSegmentTranscribeResp {
    #[serde(flatten)]
    pub transcript: AudioTranscript,
    pub metrics: Metrics,
    pub transport: AudioTransportStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_ranges: Option<Vec<LanguageRange>>,
}

struct CachedRecognition {
    range: Range<u64>,
    language: String,
    result: AudioTranscribeResp,
}

/// Segment tasks share CPU speech models and a private handle to the stream's service connection.
pub struct AudioTranscriber {
    client: Arc<AudioStreamClient>,
    acoustics: AudioProcessor<Acoustics>,
    overlap: usize,
    language_minimum: usize,
    reuse: bool,
    last: Mutex<Option<CachedRecognition>>,
}

impl AudioTranscriber {
    pub fn new(config: &Config, client: Arc<AudioStreamClient>, acoustics: AudioProcessor<Acoustics>) -> Self {
        Self {
            client,
            acoustics,
            overlap: samples(config.streaming.overlap_seconds),
            language_minimum: samples(config.language.min_seconds),
            reuse: config.streaming.reuse_identical_audio,
            last: Mutex::new(None),
        }
    }

    async fn request(
        &self,
        job: &AudioSegmentTranscribeReq,
        range: Range<usize>,
        language: &str,
        metrics: &mut Metrics,
    ) -> Result<AudioTranscribeResp> {
        let range = job.audio_start + range.start as u64..job.audio_start + range.end as u64;
        if self.reuse
            && let Some(last) = self
                .last
                .lock()
                .unwrap()
                .as_ref()
                .filter(|last| last.range == range && last.language == language)
        {
            metrics.result_cache_hits += 1;
            return Ok(last.result.clone());
        }
        metrics.asr_requests += 1;
        metrics.submitted_audio_samples += (range.end - range.start).max(160);
        let result = self.client.transcribe(range.start, range.end, language).await?;
        if self.reuse {
            *self.last.lock().unwrap() = Some(CachedRecognition {
                range,
                language: language.to_owned(),
                result: result.clone(),
            });
        }
        Ok(result)
    }

    async fn recognize(
        &self,
        job: &AudioSegmentTranscribeReq,
        range: Range<usize>,
        language: &str,
        chinese: bool,
        metrics: &mut Metrics,
    ) -> Result<(AudioTranscribeResp, bool)> {
        let mut result = self.request(job, range.clone(), language, metrics).await?;
        let mut retry = false;
        if language.is_empty() && chinese && !result.text.is_empty() && !result.text.chars().any(is_han) {
            let candidate = self.request(job, range, "Chinese", metrics).await?;
            if candidate.text.chars().any(is_han) {
                result = candidate;
                retry = true;
            }
        }
        Ok((result, retry))
    }

    async fn align(
        &self,
        job: &Arc<AudioSegmentTranscribeReq>,
        range: Range<usize>,
        result: &AudioTranscribeResp,
        metrics: &mut Metrics,
    ) -> Result<Option<Vec<Word>>> {
        if result.text.is_empty() {
            return Ok(Some(Vec::new()));
        }
        metrics.alignment_calls += 1;
        let snapshot = job.clone();
        let start = job.audio_start + range.start as u64;
        let text = result.text.clone();
        let language = result.language.clone();
        self.acoustics
            .process(move |acoustics| acoustics.align(&snapshot.audio[range], &text, &language, start))
            .await
    }

    pub async fn process(
        &self,
        job: &Arc<AudioSegmentTranscribeReq>,
        mut committed: watch::Receiver<(u64, Vec<Word>)>,
    ) -> Result<AudioSegmentTranscribeResp> {
        let started = Instant::now();
        let mut metrics = Metrics::default();
        let ranges = if let Some(language) = self.client.language() {
            vec![LanguageRange {
                start: job.start,
                end: job.end,
                language_hint: language.to_owned(),
                chinese_observed: false,
            }]
        } else {
            crate::language::plan(job.owned(), job.start, &job.languages, self.language_minimum)
        };
        let combined;
        let plan = if self.overlap > 0 && ranges.len() > 1 {
            combined = vec![LanguageRange {
                start: job.start,
                end: job.end,
                language_hint: String::new(),
                chinese_observed: ranges.iter().any(|r| r.chinese_observed),
            }];
            &combined
        } else {
            &ranges
        };
        let mut spans = Vec::new();
        for range in plan {
            let start = (range.start - job.audio_start) as usize;
            let end = (range.end - job.audio_start) as usize;
            let read_start = start.saturating_sub(self.overlap);
            let read_end = (end + self.overlap).min(job.audio.len());
            let (mut result, mut retry) = self
                .recognize(
                    job,
                    read_start..read_end,
                    &range.language_hint,
                    range.chinese_observed,
                    &mut metrics,
                )
                .await?;
            let (metadata, words) = if self.overlap > 0 {
                let audio_start = job.audio_start + read_start as u64;
                let audio_end = job.audio_start + read_end as u64;
                let raw = result.text.clone();
                let aligned = self.align(job, read_start..read_end, &result, &mut metrics).await?;
                // Recognition and alignment can run ahead. Boundary ownership needs the preceding final.
                let history = committed
                    .wait_for(|(next_segment, _)| *next_segment >= job.segment)
                    .await
                    .expect("event loop owns committed overlap history")
                    .1
                    .clone();
                let stitched = aligned.and_then(|words| {
                    let quiet_end = job.is_final() && matches!(job.reason, "silence" | "sentence");
                    if words.iter().any(|word| {
                        word.start < range.end
                            && word.end > range.end
                            && (quiet_end || word.start < range.end.saturating_sub(self.overlap as u64))
                    }) {
                        return None;
                    }
                    stitch(&raw, &words, &history, range.start, range.end)
                });
                let (status, words, skipped, transcription_start, transcription_end) = match stitched {
                    Some((text, words, skipped)) => {
                        result.text = text;
                        ("aligned", words, skipped, audio_start, audio_end)
                    },
                    None => {
                        if read_start != start || read_end != end {
                            let (owned, owned_retry) = self
                                .recognize(
                                    job,
                                    start..end,
                                    &range.language_hint,
                                    range.chinese_observed,
                                    &mut metrics,
                                )
                                .await?;
                            result = owned;
                            retry |= owned_retry;
                        }
                        let words = self
                            .align(job, start..end, &result, &mut metrics)
                            .await?
                            .unwrap_or_default();
                        ("fallback", words, 0, range.start, range.end)
                    },
                };
                (
                    Some(Overlap {
                        audio_start_sample: audio_start,
                        audio_end_sample: audio_end,
                        raw_text: raw,
                        alignment_source: "whisper-attention",
                        alignment_status: status,
                        alignment_error: (status == "fallback")
                            .then_some("attention times do not establish boundary ownership; decoded owned audio"),
                        deduplicated_prefix_words: skipped,
                        transcription_audio_start_sample: transcription_start,
                        transcription_audio_end_sample: transcription_end,
                    }),
                    words,
                )
            } else {
                (
                    None,
                    self.align(job, start..end, &result, &mut metrics)
                        .await?
                        .unwrap_or_default(),
                )
            };
            let span = Span {
                start: range.start,
                end: range.end,
                language_hint: range.language_hint.clone(),
                language: result.language,
                chinese_audio_retry: retry,
                text: result.text,
                words,
                speaker_activity: Vec::new(),
                overlap: metadata,
            };
            spans.push(span);
        }
        Ok(AudioSegmentTranscribeResp {
            transcript: AudioTranscript::from_spans(spans),
            transport: self.client.stats(),
            language_ranges: (self.overlap > 0).then_some(ranges),
            metrics: Metrics {
                wait_seconds: started.duration_since(job.created).as_secs_f64(),
                processing_seconds: started.elapsed().as_secs_f64(),
                ..metrics
            },
        })
    }
}

fn serialize_audio_seconds<S: Serializer>(samples: &u64, serializer: S) -> std::result::Result<S::Ok, S::Error> {
    serializer.serialize_f64(*samples as f64 / SAMPLE_RATE as f64)
}
