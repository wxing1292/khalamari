use std::collections::VecDeque;
use std::ops::Range;
use std::time::Instant;

use inference_error::Result;
use serde::Serialize;

use crate::audio::SAMPLE_RATE;
use crate::audio::quiet_cut;
use crate::config::StreamConfig;
use crate::config::samples;
use crate::language::LanguageRange;
use crate::recording::AudioRecordingReader;
use crate::transcribe::AudioSegmentTranscribeResp;

const MIN_CUT_SAMPLES: usize = 12_800;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AnalysisPass {
    Live,
    Offline,
}

impl AnalysisPass {
    pub fn name(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Offline => "offline",
        }
    }
}

#[derive(Debug)]
pub struct AudioSegmentTranscribeReq {
    pub segment: u64,
    pub revision: u64,
    pub start: u64,
    pub end: u64,
    pub audio_start: u64,
    pub audio: Vec<i16>,
    pub languages: Vec<LanguageRange>,
    pub reason: &'static str,
    pub created: Instant,
}

impl AudioSegmentTranscribeReq {
    pub fn is_final(&self) -> bool {
        self.reason != "update"
    }

    pub fn owned(&self) -> &[i16] {
        let start = (self.start - self.audio_start) as usize;
        &self.audio[start..start + (self.end - self.start) as usize]
    }
}

struct AudioSegment {
    id: u64,
    start: u64,
    last_partial: u64,
    revision: u64,
    applied_revision: u64,
    previous_sentence: String,
    stable_end: u64,
}

struct FinalRange {
    segment: u64,
    start: u64,
    end: u64,
    reason: &'static str,
    created: Instant,
    revision: u64, // Last submitted partial revision. The final receives its revision at submission.
}

#[derive(Clone, Debug, Serialize)]
pub struct AudioEvent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub pass: AnalysisPass,
    pub segment: u64,
    pub revision: u64,
    #[serde(rename = "start_sample")]
    pub start: u64,
    #[serde(rename = "end_sample")]
    pub end: u64,
    pub start_ms: f64,
    pub end_ms: f64,
    pub reason: &'static str,
    #[serde(flatten)]
    pub result: AudioSegmentTranscribeResp,
}

/// Owns segmentation progress and every not-yet-submitted final range.
/// The stream owns submitted tasks and the ordered result queue.
/// Timing settings use samples; lookback is the search radius.
pub struct AudioSegmenter {
    pass: AnalysisPass,
    live_cuts: VecDeque<u64>,
    window: usize,
    lookback: usize,
    interval: u64,
    silence: u64,
    sentence_silence: u64,
    overlap: u64,
    padding: u64,
    min_partial: u64,
    capacity: usize,
    recording: AudioRecordingReader,
    end: u64,
    last_voice_end: u64,
    owned_until: u64,
    next_segment: u64,
    active: Option<AudioSegment>,
    pending: VecDeque<FinalRange>,
    stopped: bool,
}

impl AudioSegmenter {
    pub fn new(
        config: &StreamConfig,
        recording: AudioRecordingReader,
        pass: AnalysisPass,
        live_cuts: VecDeque<u64>,
    ) -> Self {
        Self {
            pass,
            live_cuts,
            window: samples(config.window_seconds),
            lookback: samples(config.lookback_seconds) / 2,
            interval: samples(config.interval) as u64,
            silence: samples(config.silence_seconds) as u64,
            sentence_silence: samples(config.sentence_silence_seconds) as u64,
            overlap: samples(config.overlap_seconds) as u64,
            padding: samples(config.padding_seconds) as u64,
            min_partial: samples(config.min_partial_seconds) as u64,
            capacity: config.max_pending_finals,
            recording,
            end: 0,
            last_voice_end: 0,
            owned_until: 0,
            next_segment: 0,
            active: None,
            pending: VecDeque::new(),
            stopped: false,
        }
    }

    pub fn feed(&mut self, end: u64, speech: bool, speaker_change: Option<u64>) -> Result<()> {
        debug_assert!(!self.stopped, "audio cannot arrive after stop");
        // The recording writer validated and published this range before VAD read it.
        let previous = self.end;
        self.end = end;
        if speech {
            self.last_voice_end = self.end;
            if self.active.is_none() {
                let start = previous.saturating_sub(self.padding);
                self.start_segment(start.max(self.owned_until));
            }
        }
        if let Some(segment) = &self.active {
            let window = self.window;
            if self.end - segment.start >= window as u64 {
                let start = segment.start;
                let limit = start + window as u64;
                let count = self.live_cuts.partition_point(|cut| *cut <= limit);
                let hint =
                    self.live_cuts.drain(..count).next_back().filter(|cut| {
                        *cut >= start + MIN_CUT_SAMPLES as u64 && limit - *cut <= (2 * self.lookback) as u64
                    });
                let limit = match hint {
                    Some(cut) => cut,
                    None => {
                        let radius = self.lookback;
                        // Keep the left margin used by quiet_cut. Earlier PCM cannot affect the cut.
                        let offset = window.saturating_sub(2 * radius + MIN_CUT_SAMPLES);
                        let search_start = start + offset as u64;
                        let pcm = self.recording.read(search_start..limit)?;
                        let target = window.saturating_sub(radius) - offset;
                        quiet_cut(&pcm, target, 0, radius, MIN_CUT_SAMPLES)
                            .map_or(limit, |cut| search_start + cut as u64)
                    },
                };
                // The offline reader already knows the speech tail of this window.
                let end = if self.pass == AnalysisPass::Offline {
                    limit.min(self.last_voice_end.saturating_add(self.padding))
                } else {
                    limit
                };
                self.finalize(end, "limit");
                if self.last_voice_end > limit {
                    self.start_segment(limit);
                }
            } else if self.pass == AnalysisPass::Live
                && let Some(cut) = speaker_change
                && cut >= segment.start + self.min_partial
                && cut < self.end
            {
                self.finalize(cut, "speaker");
                if self.last_voice_end > cut {
                    self.start_segment(cut);
                }
            } else if self.pass == AnalysisPass::Live {
                self.end_on_pause();
            }
        }
        Ok(())
    }

    fn start_segment(&mut self, start: u64) {
        self.active = Some(AudioSegment {
            id: self.next_segment,
            start,
            last_partial: start,
            revision: 0,
            applied_revision: 0,
            previous_sentence: String::new(),
            stable_end: 0,
        });
        self.next_segment += 1;
    }

    fn end_on_pause(&mut self) {
        let Some(segment) = &self.active else {
            return;
        };
        let silence = self.end - self.last_voice_end;
        let reason = if silence >= self.silence {
            "silence"
        } else if segment.stable_end >= self.last_voice_end && silence >= self.sentence_silence {
            "sentence"
        } else {
            return;
        };
        self.finalize(self.end.min(self.last_voice_end.saturating_add(self.padding)), reason)
    }

    fn finalize(&mut self, end: u64, reason: &'static str) {
        let segment = self.active.take().unwrap();
        self.owned_until = end;
        if end > segment.start {
            self.pending.push_back(FinalRange {
                segment: segment.id,
                start: segment.start,
                end,
                reason,
                created: Instant::now(),
                revision: segment.revision,
            });
        }
    }

    fn snapshot(
        &self,
        segment: u64,
        revision: u64,
        range: Range<u64>,
        reason: &'static str,
        created: Instant,
    ) -> Result<AudioSegmentTranscribeReq> {
        let (start, end) = (range.start, range.end);
        let overlap = self.overlap;
        let audio_start = start.saturating_sub(overlap);
        let available = if reason == "update" {
            self.end
        } else {
            self.recording.end()
        };
        let audio_end = available.min(end.saturating_add(overlap));
        Ok(AudioSegmentTranscribeReq {
            segment,
            revision,
            start,
            end,
            audio_start,
            audio: self.recording.read(audio_start..audio_end)?,
            languages: Vec::new(),
            reason,
            created,
        })
    }

    /// Allocate a revision only when a free inference slot accepts the job. Finals have priority.
    pub fn next_request(&mut self, language_end: u64) -> Result<Option<AudioSegmentTranscribeReq>> {
        if self.pending.front().is_some_and(|range| range.end > language_end) || self.waiting_for_context().is_some() {
            return Ok(None);
        }
        if let Some(range) = self.pending.pop_front() {
            return self
                .snapshot(
                    range.segment,
                    range.revision + 1,
                    range.start..range.end,
                    range.reason,
                    range.created,
                )
                .map(Some);
        }
        let end = self.end.min(language_end);
        if self.pass == AnalysisPass::Live
            && !self.stopped
            && let Some(segment) = &mut self.active
            && end >= segment.start + self.min_partial
            && self.last_voice_end > segment.last_partial
            && end >= segment.last_partial + self.interval
        {
            segment.last_partial = end;
            segment.revision += 1;
            let (id, revision, start) = (segment.id, segment.revision, segment.start);
            return self
                .snapshot(id, revision, start..end, "update", Instant::now())
                .map(Some);
        }
        Ok(None)
    }

    pub fn is_current(&self, job: &AudioSegmentTranscribeReq) -> bool {
        job.is_final()
            || self
                .active
                .as_ref()
                .is_some_and(|segment| segment.id == job.segment && job.revision > segment.applied_revision)
    }

    pub fn accept(
        &mut self,
        job: &AudioSegmentTranscribeReq,
        result: Result<AudioSegmentTranscribeResp>,
    ) -> Result<Option<AudioEvent>> {
        if !self.is_current(job) {
            return Ok(None);
        }
        let result = result?;
        if !job.is_final() {
            let segment = self.active.as_mut().unwrap();
            segment.applied_revision = job.revision;
            let text = result.transcript.text.split_whitespace().collect::<Vec<_>>().join(" ");
            let ending = text.trim_end_matches(['"', '\'', '”', '’', '」', '』', '）', '》', '】', ')', ']', '}']);
            let sentence = if ending.ends_with(['.', '!', '?', '。', '！', '？']) {
                text
            } else {
                String::new()
            };
            segment.stable_end = if !sentence.is_empty() && sentence == segment.previous_sentence {
                job.end
            } else {
                0
            };
            segment.previous_sentence = sentence;
            if self.can_feed() {
                self.end_on_pause();
            }
        }
        let event = AudioEvent {
            kind: if job.is_final() { "final" } else { "partial" },
            pass: self.pass,
            segment: job.segment,
            revision: job.revision,
            start: job.start,
            end: job.end,
            start_ms: job.start as f64 * 1000.0 / SAMPLE_RATE as f64,
            end_ms: job.end as f64 * 1000.0 / SAMPLE_RATE as f64,
            reason: job.reason,
            result,
        };
        Ok(Some(event))
    }

    pub fn end(&self) -> u64 {
        self.end
    }

    pub fn stop(&mut self) {
        assert!(!self.stopped, "segmenter stops once");
        self.stopped = true;
        if self.active.is_some() {
            self.finalize(self.end.min(self.last_voice_end.saturating_add(self.padding)), "end");
        }
    }

    pub fn retained_sample(&self) -> u64 {
        let overlap = self.overlap;
        let mut retain = self
            .active
            .as_ref()
            .map_or(self.end.saturating_sub(self.padding), |s| s.start)
            .saturating_sub(overlap);
        if let Some(first) = self.pending.front() {
            retain = retain.min(first.start.saturating_sub(overlap));
        }
        retain
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn can_feed(&self) -> bool {
        self.pending.len() < self.capacity
    }

    /// Right context belongs to the recording. A full queue need not advance the segmenter to obtain it.
    pub fn waiting_for_context(&self) -> Option<u64> {
        let until = self.pending.front()?.end + self.overlap;
        (self.recording.end() < until && !self.stopped && !self.recording.is_sealed()).then_some(until)
    }
}

#[cfg(test)]
#[path = "segment_test.rs"]
mod tests;
