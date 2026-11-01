use serde::Deserialize;
use serde::Serialize;

use crate::audio::SAMPLE_RATE;
use crate::diarization::SpeakerSpan;

#[derive(Clone, Debug, Serialize)]
pub struct AudioTranscript {
    pub sample_rate: usize,
    pub text: String,
    pub language: String,
    pub spans: Vec<Span>,
    pub turns: Vec<SpeakerTurn>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Word {
    pub text: String,
    #[serde(rename = "start_sample")]
    pub start: u64,
    #[serde(rename = "end_sample")]
    pub end: u64,
    pub alignment_score: f32,
    pub timing_status: &'static str,
    pub speakers: Vec<u32>,
    #[serde(skip)]
    pub byte_start: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Span {
    #[serde(rename = "start_sample")]
    pub start: u64,
    #[serde(rename = "end_sample")]
    pub end: u64,
    pub language_hint: String,
    pub language: String,
    pub chinese_audio_retry: bool,
    pub text: String,
    pub words: Vec<Word>,
    #[serde(rename = "speakers")]
    pub speaker_activity: Vec<SpeakerSpan>,
    #[serde(flatten)]
    pub overlap: Option<Overlap>,
}

/// The speaker-attributed view of a hypothesis, on the original sample clock.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SpeakerTurn {
    #[serde(rename = "start_sample")]
    pub start: u64,
    #[serde(rename = "end_sample")]
    pub end: u64,
    pub speakers: Vec<u32>,
    pub text: String,
}

impl SpeakerTurn {
    fn from_spans(spans: &[Span]) -> Vec<Self> {
        let mut turns: Vec<Self> = Vec::new();
        for turn in spans.iter().flat_map(Self::from_span) {
            if let Some(previous) = turns
                .last_mut()
                .filter(|previous| previous.end == turn.start && previous.speakers == turn.speakers)
            {
                previous.end = turn.end;
                previous.text = join_text([previous.text.as_str(), turn.text.as_str()]);
            } else {
                turns.push(turn);
            }
        }
        turns
    }

    fn from_span(span: &Span) -> Vec<Self> {
        if span.text.is_empty() {
            return Vec::new();
        }
        let single_speaker = || {
            span.speaker_activity
                .first()
                .filter(|speaker| {
                    span.speaker_activity
                        .iter()
                        .all(|other| other.speaker == speaker.speaker)
                })
                .map(|speaker| vec![speaker.speaker])
                .unwrap_or_default()
        };
        if span.words.is_empty() {
            return vec![Self {
                start: span.start,
                end: span.end,
                speakers: single_speaker(),
                text: span.text.clone(),
            }];
        }
        let mut groups = span.words.chunk_by(|a, b| a.speakers == b.speakers).peekable();
        let mut start = 0;
        let mut turns = Vec::new();
        while let Some(group) = groups.next() {
            let end = groups.peek().map_or(span.text.len(), |next| next[0].byte_start);
            turns.push(Self {
                start: group[0].start.clamp(span.start, span.end),
                end: group.last().unwrap().end.clamp(span.start, span.end),
                speakers: group[0].speakers.clone(),
                text: span.text[start..end].to_owned(),
            });
            start = end;
        }
        for i in 0..turns.len() {
            if !turns[i].speakers.is_empty() {
                continue;
            }
            let speakers = match (turns[..i].last(), turns[i + 1..].first()) {
                (Some(left), Some(right)) => {
                    left.speakers
                        .iter()
                        .filter(|speaker| right.speakers.contains(speaker))
                        .copied()
                        .collect()
                },
                (Some(neighbor), None) | (None, Some(neighbor)) => neighbor.speakers.clone(),
                (None, None) => single_speaker(),
            };
            let turn = &mut turns[i];
            // Infer continuity only when neighboring text agrees and the audio has no conflicting voice.
            let conflict = span.speaker_activity.iter().any(|speaker| {
                let intersects = if turn.start == turn.end {
                    speaker.start <= turn.start && turn.start < speaker.end
                } else {
                    speaker.start < turn.end && turn.start < speaker.end
                };
                intersects && !speakers.contains(&speaker.speaker)
            });
            if !conflict {
                turn.speakers = speakers;
            }
        }
        let mut merged: Vec<Self> = Vec::with_capacity(turns.len());
        for turn in turns {
            if let Some(previous) = merged.last_mut().filter(|previous| previous.speakers == turn.speakers) {
                previous.end = previous.end.max(turn.end);
                previous.text.push_str(&turn.text);
            } else {
                merged.push(turn);
            }
        }
        if let [turn] = merged.as_mut_slice() {
            turn.start = span.start;
            turn.end = span.end;
        }
        merged
    }

    pub fn display(&self) -> String {
        let who = if self.speakers.is_empty() {
            "unresolved".into()
        } else {
            self.speakers
                .iter()
                .map(|speaker| format!("speaker_{speaker}"))
                .collect::<Vec<_>>()
                .join("+")
        };
        format!(
            "[{:.2}, {:.2}) {who}  {}",
            self.start as f64 / SAMPLE_RATE as f64,
            self.end as f64 / SAMPLE_RATE as f64,
            self.text.trim()
        )
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Overlap {
    pub audio_start_sample: u64,
    pub audio_end_sample: u64,
    pub raw_text: String,
    pub alignment_source: &'static str,
    pub alignment_status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alignment_error: Option<&'static str>,
    pub deduplicated_prefix_words: usize,
    pub transcription_audio_start_sample: u64,
    pub transcription_audio_end_sample: u64,
}

impl Span {
    fn attribute(&mut self, speakers: &[SpeakerSpan]) -> bool {
        let (start, end) = (self.start, self.end);
        let labels = speakers
            .iter()
            .take_while(|speaker| speaker.start < end)
            .filter(|speaker| speaker.end > start)
            .map(|speaker| {
                SpeakerSpan {
                    start: speaker.start.max(start),
                    end: speaker.end.min(end),
                    speaker: speaker.speaker,
                }
            });
        if labels.clone().eq(self.speaker_activity.iter().cloned()) {
            return false;
        }
        self.speaker_activity = labels.collect();
        // Reuse per-word coverage storage. Same-speaker intervals can overlap after timeline replacement.
        let mut coverage: Vec<(u32, u64, u64)> = Vec::new();
        for word in &mut self.words {
            word.speakers.clear();
            if word.start == word.end {
                continue;
            }
            coverage.clear();
            for speaker in self
                .speaker_activity
                .iter()
                .take_while(|speaker| speaker.start < word.end)
            {
                let start = speaker.start.max(word.start);
                let end = speaker.end.min(word.end);
                if start >= end {
                    continue;
                }
                if let Some((_, num_samples, covered_end)) = coverage.iter_mut().find(|(id, ..)| *id == speaker.speaker)
                {
                    let start = start.max(*covered_end);
                    if start < end {
                        *num_samples += end - start;
                        *covered_end = end;
                    }
                } else {
                    coverage.push((speaker.speaker, end - start, end));
                }
            }
            let most = coverage
                .iter()
                .map(|(_, num_samples, _)| *num_samples)
                .max()
                .unwrap_or(0);
            word.speakers.extend(
                coverage
                    .iter()
                    .filter(|(_, num_samples, _)| *num_samples == most)
                    .map(|(speaker, ..)| *speaker),
            );
            word.speakers.sort_unstable();
        }
        true
    }
}

impl AudioTranscript {
    pub fn attribute(&mut self, speakers: &[SpeakerSpan]) -> bool {
        let mut changed = false;
        for span in &mut self.spans {
            changed |= span.attribute(speakers);
        }
        if changed {
            self.turns = SpeakerTurn::from_spans(&self.spans);
        }
        changed
    }

    pub fn from_spans(spans: Vec<Span>) -> Self {
        let text = join_text(spans.iter().map(|span| span.text.as_str()));
        let mut languages = Vec::new();
        for span in &spans {
            if !span.language.is_empty() && !languages.contains(&span.language.as_str()) {
                languages.push(span.language.as_str());
            }
        }
        let turns = SpeakerTurn::from_spans(&spans);
        Self {
            sample_rate: SAMPLE_RATE,
            text,
            language: languages.join(","),
            spans,
            turns,
        }
    }
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

#[cfg(test)]
#[path = "transcript_test.rs"]
mod tests;
