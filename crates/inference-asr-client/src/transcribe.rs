use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use inference_error::Result;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::channel::ShutdownGuard;
use inference_runtime_core::runtime::tasks::SyncTaskPool;
use serde::Serialize;
use serde::Serializer;
use tokio::sync::watch;

use crate::audio::SAMPLE_RATE;
use crate::config::Config;
use crate::config::LanguageConfig;
use crate::config::samples;
use crate::language::AudioLangRecognitionReq;
use crate::language::LanguageAnalyzer;
use crate::language::LanguageRange;
use crate::models::AcousticModel;
use crate::models::is_han;
use crate::overlap::Word;
use crate::overlap::stitch;
use crate::overlap::timed_words;
use crate::segment::AudioSegmentReq;
use crate::service::AudioTranscribeResp;
use crate::service::stream::AudioStreamClient;
use crate::service::stream::AudioTransportStats;

#[derive(Debug, Default, Serialize)]
pub struct Metrics {
    pub wait_seconds: f64,
    pub processing_seconds: f64,
    pub asr_requests: u64,
    #[serde(rename = "submitted_audio_seconds", serialize_with = "serialize_audio_seconds")]
    pub submitted_audio_samples: u64,
    pub lid_windows: u64,
    pub result_cache_hits: u64,
    pub alignment_calls: u64,
}

#[derive(Debug, Serialize)]
pub struct AudioSegmentResp {
    pub text: String,
    pub language: String,
    pub spans: Vec<Span>,
    pub metrics: Metrics,
    pub transport: AudioTransportStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_ranges: Option<Vec<LanguageRange>>,
}

#[derive(Debug, Serialize)]
pub struct Span {
    pub start_sample: u64,
    pub end_sample: u64,
    pub language_hint: String,
    pub language: String,
    pub chinese_audio_retry: bool,
    pub text: String,
    #[serde(flatten)]
    pub overlap: Option<Overlap>,
}

#[derive(Debug, Serialize)]
pub struct Overlap {
    audio_start_sample: u64,
    audio_end_sample: u64,
    raw_text: String,
    alignment_source: &'static str,
    alignment_status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    alignment_error: Option<&'static str>,
    deduplicated_prefix_words: usize,
    transcription_audio_start_sample: u64,
    transcription_audio_end_sample: u64,
    pub words: Vec<Word>,
}

struct CachedRecognition {
    range: Range<u64>,
    language: String,
    result: AudioTranscribeResp,
}

struct Acoustics {
    model: AcousticModel,
    analyzer: LanguageAnalyzer,
    cpu: SyncTaskPool,
    _shutdown: ShutdownGuard,
}

impl Acoustics {
    fn load(config: &LanguageConfig) -> Result<Self> {
        let model = AcousticModel::load(config)?;
        let shutdown = Shutdown::new();
        let cpu = SyncTaskPool::new(shutdown.clone(), 1)?;
        Ok(Self {
            model,
            analyzer: LanguageAnalyzer::new(config),
            cpu,
            _shutdown: ShutdownGuard::new(shutdown),
        })
    }
}

/// Segment tasks share CPU speech models and a private handle to the stream's service connection.
pub struct AudioTranscriber {
    client: Arc<AudioStreamClient>,
    acoustics: Option<Arc<tokio::sync::Mutex<Acoustics>>>,
    overlap: usize,
    reuse: bool,
    last: Mutex<Option<CachedRecognition>>,
}

impl AudioTranscriber {
    pub fn load(config: &Config, client: Arc<AudioStreamClient>) -> Result<Self> {
        let acoustics = if config.service.language.is_none() || config.streaming.overlap_seconds > 0.0 {
            let acoustics = tokio::task::block_in_place(|| Acoustics::load(&config.language))?;
            Some(Arc::new(tokio::sync::Mutex::new(acoustics)))
        } else {
            None
        };
        Ok(Self {
            client,
            acoustics,
            overlap: samples(config.streaming.overlap_seconds),
            reuse: config.streaming.reuse_identical_audio,
            last: Mutex::new(None),
        })
    }

    async fn request(
        &self,
        job: &AudioSegmentReq,
        range: Range<usize>,
        language: &str,
        metrics: &mut Metrics,
    ) -> Result<AudioTranscribeResp> {
        let range = job.audio_start_sample + range.start as u64..job.audio_start_sample + range.end as u64;
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
        job: &AudioSegmentReq,
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
        job: &AudioSegmentReq,
        range: Range<usize>,
        result: &AudioTranscribeResp,
        metrics: &mut Metrics,
    ) -> Result<Option<Vec<Word>>> {
        if result.text.is_empty() {
            return Ok(Some(Vec::new()));
        }
        metrics.alignment_calls += 1;
        let mut acoustics = self.acoustics.as_ref().unwrap().clone().lock_owned().await;
        let cpu = acoustics.cpu.clone();
        let audio = job.audio.clone();
        let start = job.audio_start_sample + range.start as u64;
        let text = result.text.clone();
        let language = result.language.clone();
        cpu.spawn(move || {
            let units = acoustics.model.align(&audio[range], &text, &language)?;
            Ok(units.and_then(|units| timed_words(&text, &units, start)))
        })
        .await
        .expect("CPU pool returns every accepted task result")
    }

    pub async fn process(
        &self,
        job: &AudioSegmentReq,
        mut committed: watch::Receiver<(u64, Vec<Word>)>,
    ) -> Result<AudioSegmentResp> {
        let started = Instant::now();
        let mut metrics = Metrics::default();
        let ranges = if let Some(language) = self.client.language() {
            vec![LanguageRange {
                start_sample: job.start_sample,
                end_sample: job.end_sample,
                language_hint: language.to_owned(),
                chinese_observed: false,
            }]
        } else {
            // Acquire before spawning. Cancellation cannot leave an unbounded queue of blocking CPU calls.
            let mut acoustics = self.acoustics.as_ref().unwrap().clone().lock_owned().await;
            let cpu = acoustics.cpu.clone();
            let job = job.clone();
            let response = cpu
                .spawn(move || {
                    let Acoustics { model, analyzer, .. } = &mut *acoustics;
                    analyzer.plan(
                        model,
                        &AudioLangRecognitionReq {
                            segment: job.segment,
                            start_sample: job.start_sample,
                            end_sample: job.end_sample,
                            audio: job.owned(),
                        },
                    )
                })
                .await
                .expect("CPU pool returns every accepted task result")?;
            metrics.lid_windows = response.windows;
            response.ranges
        };
        let combined;
        let plan = if self.overlap > 0 && ranges.len() > 1 {
            combined = vec![LanguageRange {
                start_sample: job.start_sample,
                end_sample: job.end_sample,
                language_hint: String::new(),
                chinese_observed: ranges.iter().any(|r| r.chinese_observed),
            }];
            &combined
        } else {
            &ranges
        };
        let mut spans = Vec::new();
        for range in plan {
            let start = (range.start_sample - job.audio_start_sample) as usize;
            let end = (range.end_sample - job.audio_start_sample) as usize;
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
            let metadata = if self.overlap > 0 {
                let audio_start = job.audio_start_sample + read_start as u64;
                let audio_end = job.audio_start_sample + read_end as u64;
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
                        word.start_sample < range.end_sample
                            && word.end_sample > range.end_sample
                            && (quiet_end || word.start_sample < range.end_sample.saturating_sub(self.overlap as u64))
                    }) {
                        return None;
                    }
                    stitch(&raw, &words, &history, range.start_sample, range.end_sample)
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
                        ("fallback", words, 0, range.start_sample, range.end_sample)
                    },
                };
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
                    words,
                })
            } else {
                None
            };
            spans.push(Span {
                start_sample: range.start_sample,
                end_sample: range.end_sample,
                language_hint: range.language_hint.clone(),
                language: result.language,
                chinese_audio_retry: retry,
                text: result.text,
                overlap: metadata,
            });
        }
        let text = join_text(spans.iter().map(|span| span.text.as_str()));
        let mut languages = Vec::new();
        for span in &spans {
            if !span.language.is_empty() && !languages.contains(&span.language.as_str()) {
                languages.push(span.language.as_str());
            }
        }
        Ok(AudioSegmentResp {
            text,
            language: languages.join(","),
            spans,
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

fn join_text<'a>(parts: impl IntoIterator<Item = &'a str>) -> String {
    let mut result = String::new();
    for part in parts {
        let part = part.trim();
        if !result.is_empty()
            && !part.is_empty()
            && (result.chars().next_back().unwrap().is_ascii_alphanumeric()
                || part.chars().next().unwrap().is_ascii_alphanumeric())
        {
            result.push(' ');
        }
        result.push_str(part);
    }
    result
}
