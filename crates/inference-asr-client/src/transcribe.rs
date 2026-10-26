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
use crate::service::AsrClient;
use crate::service::AudioTranscribeReq;
use crate::service::AudioTranscribeResp;

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
    pcm: Vec<i16>,
    language: String,
    result: AudioTranscribeResp,
}

struct Acoustics {
    model: Option<AcousticModel>,
    analyzer: LanguageAnalyzer,
}

/// Segment tasks share one CPU model and service connection pool. Each task owns its work metrics.
pub struct AudioTranscriber {
    client: AsrClient,
    acoustics: Arc<tokio::sync::Mutex<Acoustics>>,
    overlap: usize,
    reuse: bool,
    last: Mutex<Option<CachedRecognition>>,
    cpu: SyncTaskPool,
    _shutdown: ShutdownGuard,
}

impl AudioTranscriber {
    pub fn load(config: &Config) -> Result<Self> {
        let shutdown = Shutdown::new();
        let cpu = SyncTaskPool::new(shutdown.clone(), 1)?;
        let model = if config.service.language.is_none() || config.streaming.overlap_seconds > 0.0 {
            Some(AcousticModel::load(&config.language)?)
        } else {
            None
        };
        Ok(Self {
            client: AsrClient::new(config.service.clone())?,
            acoustics: Arc::new(tokio::sync::Mutex::new(Acoustics {
                model,
                analyzer: LanguageAnalyzer::new(&config.language),
            })),
            overlap: samples(config.streaming.overlap_seconds),
            reuse: config.streaming.reuse_identical_audio,
            last: Mutex::new(None),
            cpu,
            _shutdown: ShutdownGuard::new(shutdown),
        })
    }

    async fn request(&self, pcm: &[i16], language: &str, metrics: &mut Metrics) -> Result<AudioTranscribeResp> {
        if self.reuse
            && let Some(last) = self
                .last
                .lock()
                .unwrap()
                .as_ref()
                .filter(|last| last.pcm == pcm && last.language == language)
        {
            metrics.result_cache_hits += 1;
            return Ok(last.result.clone());
        }
        let padded;
        let input = if pcm.len() < 160 {
            let mut input = vec![0; 160];
            input[..pcm.len()].copy_from_slice(pcm);
            padded = input;
            &padded
        } else {
            pcm
        };
        metrics.asr_requests += 1;
        metrics.submitted_audio_samples += input.len() as u64;
        let result = self
            .client
            .transcribe(AudioTranscribeReq { audio: input, language })
            .await?;
        if self.reuse {
            *self.last.lock().unwrap() = Some(CachedRecognition {
                pcm: pcm.to_vec(),
                language: language.to_owned(),
                result: result.clone(),
            });
        }
        Ok(result)
    }

    async fn recognize(
        &self,
        pcm: &[i16],
        language: &str,
        chinese: bool,
        metrics: &mut Metrics,
    ) -> Result<(AudioTranscribeResp, bool)> {
        let mut result = self.request(pcm, language, metrics).await?;
        let mut retry = false;
        if language.is_empty() && chinese && !result.text.is_empty() && !result.text.chars().any(is_han) {
            let candidate = self.request(pcm, "Chinese", metrics).await?;
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
        let mut acoustics = self.acoustics.clone().lock_owned().await;
        let audio = job.audio.clone();
        let start = job.audio_start_sample + range.start as u64;
        let text = result.text.clone();
        let language = result.language.clone();
        self.cpu
            .spawn(move || {
                let units = acoustics
                    .model
                    .as_mut()
                    .unwrap()
                    .align(&audio[range], &text, &language)?;
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
            let mut acoustics = self.acoustics.clone().lock_owned().await;
            let job = job.clone();
            let response = self
                .cpu
                .spawn(move || {
                    let Acoustics { model, analyzer } = &mut *acoustics;
                    analyzer.plan(
                        model.as_mut().unwrap(),
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
            let pcm = &job.audio[start..end];
            let request_pcm = &job.audio[read_start..read_end];
            let (mut result, mut retry) = self
                .recognize(request_pcm, &range.language_hint, range.chinese_observed, &mut metrics)
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
                                .recognize(pcm, &range.language_hint, range.chinese_observed, &mut metrics)
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
