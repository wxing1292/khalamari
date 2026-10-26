use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use inference_error::Error;
use inference_error::Result;
use serde::Serialize;

use crate::audio::SAMPLE_RATE;
use crate::audio::quiet_cut;
use crate::config::StreamConfig;
use crate::config::samples;
use crate::transcribe::AudioSegmentResp;

#[derive(Clone, Debug)]
pub struct AudioSegmentReq {
    pub segment: u64,
    pub revision: u64,
    pub start_sample: u64,
    pub end_sample: u64,
    pub audio_start_sample: u64,
    pub audio: Arc<[i16]>,
    pub reason: &'static str,
    pub created: Instant,
}

impl AudioSegmentReq {
    pub fn is_final(&self) -> bool {
        self.reason != "update"
    }

    pub fn owned(&self) -> &[i16] {
        let start = (self.start_sample - self.audio_start_sample) as usize;
        &self.audio[start..start + (self.end_sample - self.start_sample) as usize]
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

#[derive(Debug, Serialize)]
pub struct AudioEvent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub segment: u64,
    pub revision: u64,
    pub start_sample: u64,
    pub end_sample: u64,
    pub start_ms: f64,
    pub end_ms: f64,
    pub sample_rate: usize,
    pub reason: &'static str,
    #[serde(flatten)]
    pub transcript: AudioSegmentResp,
}

/// Owns the source timeline and every not-yet-submitted final range.
/// Submitted jobs own their audio. Completed finals retain their slots until ordered commit.
/// Timing settings use samples; lookback is the search radius.
pub struct AudioSegmenter {
    window: usize,
    lookback: usize,
    interval: u64,
    silence: u64,
    sentence_silence: u64,
    overlap: u64,
    padding: u64,
    min_partial: u64,
    capacity: usize,
    pcm: VecDeque<i16>,
    base: u64,
    end: u64,
    last_voice_end: u64,
    owned_until: u64,
    next_segment: u64,
    active: Option<AudioSegment>,
    pending: VecDeque<FinalRange>,
    completed: BTreeMap<u64, AudioEvent>,
    next_final: u64,
    finished: bool,
}

impl AudioSegmenter {
    pub fn new(config: &StreamConfig) -> Self {
        Self {
            window: samples(config.window_seconds),
            lookback: samples(config.lookback_seconds) / 2,
            interval: samples(config.interval) as u64,
            silence: samples(config.silence_seconds) as u64,
            sentence_silence: samples(config.sentence_silence_seconds) as u64,
            overlap: samples(config.overlap_seconds) as u64,
            padding: samples(config.padding_seconds) as u64,
            min_partial: samples(config.min_partial_seconds) as u64,
            capacity: config.max_pending_finals
                + (config.overlap_seconds / config.sentence_silence_seconds).ceil() as usize,
            pcm: VecDeque::new(),
            base: 0,
            end: 0,
            last_voice_end: 0,
            owned_until: 0,
            next_segment: 0,
            active: None,
            pending: VecDeque::new(),
            completed: BTreeMap::new(),
            next_final: 0,
            finished: false,
        }
    }

    pub fn feed(&mut self, frame: &[i16], speech: bool) -> Result<()> {
        debug_assert!(!self.finished, "audio cannot arrive after finish");
        self.end = self
            .end
            .checked_add(frame.len() as u64)
            .ok_or_else(|| Error::resource_exhausted("audio sample timeline exhausted"))?;
        self.pcm.extend(frame);
        if speech {
            self.last_voice_end = self.end;
            if self.active.is_none() {
                let start = self
                    .base
                    .max((self.end - frame.len() as u64).saturating_sub(self.padding));
                self.start_segment(start.max(self.owned_until));
            }
        }
        if let Some(segment) = &self.active {
            let window = self.window;
            if self.end - segment.start >= window as u64 {
                let radius = self.lookback;
                let pcm: Vec<_> = self
                    .range(segment.start, segment.start + window as u64)
                    .copied()
                    .collect();
                let cut = quiet_cut(&pcm, window.saturating_sub(radius), 0, radius, 12800).unwrap_or(window);
                let limit = segment.start + cut as u64;
                self.finalize(limit, "limit")?;
                if self.last_voice_end > limit {
                    self.start_segment(limit);
                }
            } else {
                self.end_on_pause()?;
            }
        }
        self.prune();
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

    fn end_on_pause(&mut self) -> Result<()> {
        let Some(segment) = &self.active else {
            return Ok(());
        };
        let silence = self.end - self.last_voice_end;
        let reason = if silence >= self.silence {
            "silence"
        } else if segment.stable_end >= self.last_voice_end && silence >= self.sentence_silence {
            "sentence"
        } else {
            return Ok(());
        };
        self.finalize(self.end.min(self.last_voice_end.saturating_add(self.padding)), reason)
    }

    fn finalize(&mut self, end: u64, reason: &'static str) -> Result<()> {
        let capacity = self.capacity;
        if reason != "end" && self.pending.len() >= capacity {
            return Err(Error::resource_exhausted(format!(
                "ASR cannot keep up: {capacity} final segments are pending; recording stopped without silent audio \
                 loss"
            )));
        }
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
        Ok(())
    }

    fn range(&self, start: u64, end: u64) -> impl Iterator<Item = &i16> {
        debug_assert!(self.base <= start && start <= end && end <= self.end);
        self.pcm.range((start - self.base) as usize..(end - self.base) as usize)
    }

    fn snapshot(
        &self,
        segment: u64,
        revision: u64,
        range: Range<u64>,
        reason: &'static str,
        created: Instant,
    ) -> AudioSegmentReq {
        let (start, end) = (range.start, range.end);
        let overlap = self.overlap;
        let audio_start = self.base.max(start.saturating_sub(overlap));
        let audio_end = self.end.min(end.saturating_add(overlap));
        AudioSegmentReq {
            segment,
            revision,
            start_sample: start,
            end_sample: end,
            audio_start_sample: audio_start,
            audio: self.range(audio_start, audio_end).copied().collect::<Vec<_>>().into(),
            reason,
            created,
        }
    }

    /// Allocate a revision only when a free inference slot accepts the job. Finals have priority.
    pub fn next_request(&mut self) -> Option<AudioSegmentReq> {
        if let Some(range) = self.pending.front() {
            if !self.finished && self.end - range.end < self.overlap {
                return None;
            }
            let range = self.pending.pop_front().unwrap();
            let job = self.snapshot(
                range.segment,
                range.revision + 1,
                range.start..range.end,
                range.reason,
                range.created,
            );
            self.prune();
            return Some(job);
        }
        if !self.finished
            && let Some(segment) = &mut self.active
            && self.end - segment.start >= self.min_partial
            && self.last_voice_end > segment.last_partial
            && self.end - segment.last_partial >= self.interval
        {
            segment.last_partial = self.end;
            segment.revision += 1;
            let (id, revision, start) = (segment.id, segment.revision, segment.start);
            return Some(self.snapshot(id, revision, start..self.end, "update", Instant::now()));
        }
        None
    }

    pub fn is_current(&self, job: &AudioSegmentReq) -> bool {
        job.is_final()
            || self
                .active
                .as_ref()
                .is_some_and(|segment| segment.id == job.segment && job.revision > segment.applied_revision)
    }

    pub fn accept(&mut self, job: &AudioSegmentReq, result: Result<AudioSegmentResp>) -> Result<Vec<AudioEvent>> {
        if !self.is_current(job) {
            return Ok(Vec::new());
        }
        let transcript = result?;
        if !job.is_final() {
            let segment = self.active.as_mut().unwrap();
            segment.applied_revision = job.revision;
            let text = transcript.text.split_whitespace().collect::<Vec<_>>().join(" ");
            let ending = text.trim_end_matches(['"', '\'', '”', '’', '」', '』', '）', '》', '】', ')', ']', '}']);
            let sentence = if ending.ends_with(['.', '!', '?', '。', '！', '？']) {
                text
            } else {
                String::new()
            };
            segment.stable_end = if !sentence.is_empty() && sentence == segment.previous_sentence {
                job.end_sample
            } else {
                0
            };
            segment.previous_sentence = sentence;
            self.end_on_pause()?;
        }
        let event = AudioEvent {
            kind: if job.is_final() { "final" } else { "partial" },
            segment: job.segment,
            revision: job.revision,
            start_sample: job.start_sample,
            end_sample: job.end_sample,
            start_ms: job.start_sample as f64 * 1000.0 / SAMPLE_RATE as f64,
            end_ms: job.end_sample as f64 * 1000.0 / SAMPLE_RATE as f64,
            sample_rate: SAMPLE_RATE,
            reason: job.reason,
            transcript,
        };
        if !job.is_final() {
            return Ok(vec![event]);
        }
        assert!(job.segment >= self.next_final, "final commits once");
        assert!(
            self.completed.insert(job.segment, event).is_none(),
            "final completes once"
        );
        let mut events = Vec::new();
        while let Some(event) = self.completed.remove(&self.next_final) {
            self.next_final += 1;
            events.push(event);
        }
        Ok(events)
    }

    pub fn finish(&mut self) -> Result<u64> {
        assert!(!self.finished, "capture finishes once");
        self.finished = true;
        if self.active.is_some() {
            self.finalize(self.end, "end")?;
        }
        Ok(self.end)
    }

    fn prune(&mut self) {
        let overlap = self.overlap;
        let mut retain = self
            .active
            .as_ref()
            .map_or(self.end.saturating_sub(self.padding), |s| s.start)
            .saturating_sub(overlap);
        if let Some(first) = self.pending.front() {
            retain = retain.min(first.start.saturating_sub(overlap));
        }
        let until = retain.max(self.base);
        self.pcm.drain(..(until - self.base) as usize);
        self.base = until;
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn num_completed(&self) -> usize {
        self.completed.len()
    }
}

#[cfg(test)]
#[path = "segment_test.rs"]
mod tests;
