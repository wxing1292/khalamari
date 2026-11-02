use inference_error::Error;
use inference_error::Result;
use serde::Deserialize;
use serde::Serialize;

use crate::audio::SAMPLE_RATE;
use crate::segment::AudioEvent;
use crate::terminal::display_text;
use crate::transcript::SpeakerTurn;

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Update {
    Partial(Line),
    Final(Line),
    Speakers(Line),
    Correction(Correction),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Line {
    pub segment: u64,
    pub revision: u64,
    #[serde(rename = "start_sample")]
    pub start: u64,
    #[serde(rename = "end_sample")]
    pub end: u64,
    pub sample_rate: usize,
    pub text: String,
    #[serde(default)]
    pub turns: Vec<SpeakerTurn>,
}

impl Line {
    pub fn display(&self, separator: &str) -> String {
        if self.turns.is_empty() {
            return display_text(&self.text);
        }
        self.turns
            .iter()
            .map(|turn| display_text(&turn.display()))
            .collect::<Vec<_>>()
            .join(separator)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Correction {
    pub segment: u64,
    pub base_revision: u64,
    pub revision: u64,
    pub text: String,
}

/// Owns the current text. The event log separately retains unmodified recognition results.
#[derive(Default)]
pub struct Document {
    lines: Vec<Line>,
    draft: Option<Line>,
}

impl Document {
    pub fn lines(&self) -> &[Line] {
        &self.lines
    }

    pub fn draft(&self) -> Option<&Line> {
        self.draft.as_ref()
    }

    /// Returns whether trusted session output changes the document projection.
    pub fn observe(&mut self, event: &AudioEvent) -> bool {
        let line = Line {
            segment: event.segment,
            revision: event.revision,
            start: event.start,
            end: event.end,
            sample_rate: event.result.transcript.sample_rate,
            text: event.result.transcript.text.clone(),
            turns: event.result.transcript.turns.clone(),
        };
        self.commit(match event.kind {
            "partial" => Update::Partial(line),
            "final" => Update::Final(line),
            "speakers" => Update::Speakers(line),
            _ => unreachable!("speech session emits recognition events"),
        })
    }

    /// Validates replay and correction input. Final text accepts explicit corrections; speaker labels can be revised.
    pub fn apply(&mut self, update: Update) -> Result<()> {
        match &update {
            Update::Partial(line) | Update::Final(line) | Update::Speakers(line)
                if line.sample_rate != SAMPLE_RATE
                    || line.start >= line.end
                    || line.revision == 0
                    || line
                        .turns
                        .iter()
                        .any(|turn| turn.start < line.start || turn.start > turn.end || turn.end > line.end) =>
            {
                return Err(Error::invalid_argument(
                    "transcript event requires a nonempty 16 kHz sample range",
                ));
            },
            Update::Final(line) => {
                if line.segment >= self.lines.len() as u64
                    && (line.segment != self.lines.len() as u64
                        || self.lines.last().is_some_and(|previous| line.start < previous.end)
                        || self
                            .draft
                            .as_ref()
                            .is_some_and(|draft| draft.segment == line.segment && draft.revision >= line.revision))
                {
                    return Err(Error::invalid_argument(
                        "final transcript events must commit in audio and revision order",
                    ));
                }
            },
            Update::Correction(correction) => {
                if correction.segment >= self.lines.len() as u64 {
                    return Err(Error::invalid_argument("correction must name a finalized segment"));
                }
                let line = &self.lines[correction.segment as usize];
                if correction.base_revision != line.revision
                    || correction.base_revision.checked_add(1) != Some(correction.revision)
                {
                    return Err(Error::invalid_argument(
                        "correction must replace the current revision with its successor",
                    ));
                }
            },
            Update::Speakers(line) => {
                let current = self
                    .lines
                    .get(line.segment as usize)
                    .or_else(|| self.draft.as_ref().filter(|draft| draft.segment == line.segment));
                if let Some(current) = current {
                    if line.revision > current.revision
                        || (line.revision == current.revision
                            && (line.start != current.start || line.end != current.end || line.text != current.text))
                    {
                        return Err(Error::invalid_argument(
                            "speaker update must preserve the recognized text revision and range",
                        ));
                    }
                } else if self.draft.as_ref().is_none_or(|draft| line.segment >= draft.segment) {
                    return Err(Error::invalid_argument("speaker update precedes recognized text"));
                }
                // Older revisions and displaced drafts cannot change the current projection.
            },
            Update::Partial(_) => {},
        }
        self.commit(update);
        Ok(())
    }

    /// Applies trusted or owner-validated input. Returns whether the document projection changes.
    pub fn commit(&mut self, update: Update) -> bool {
        match update {
            Update::Partial(line) => {
                if line.segment >= self.lines.len() as u64
                    && self.draft.as_ref().is_none_or(|draft| {
                        line.segment > draft.segment
                            || (line.segment == draft.segment && line.revision > draft.revision)
                    })
                {
                    self.draft = Some(line);
                    return true;
                }
                false
            },
            Update::Final(line) => {
                if line.segment < self.lines.len() as u64 {
                    return false;
                }
                if self.draft.as_ref().is_some_and(|draft| draft.segment <= line.segment) {
                    self.draft = None;
                }
                self.lines.push(line);
                true
            },
            Update::Speakers(update) => {
                let line = self
                    .lines
                    .get_mut(update.segment as usize)
                    .or_else(|| self.draft.as_mut().filter(|draft| draft.segment == update.segment));
                if let Some(line) = line.filter(|line| {
                    line.revision == update.revision && line.text == update.text && line.turns != update.turns
                }) {
                    line.turns = update.turns;
                    return true;
                }
                false
            },
            Update::Correction(correction) => {
                let line = &mut self.lines[correction.segment as usize];
                line.text = correction.text;
                if let Some((turn, others)) = line.turns.split_first_mut()
                    && turn.speakers.len() == 1
                    && others.iter().all(|other| other.speakers == turn.speakers)
                {
                    turn.start = line.start;
                    turn.end = line.end;
                    turn.text.clone_from(&line.text);
                    line.turns.truncate(1);
                } else {
                    line.turns.clear();
                }
                line.revision = correction.revision;
                true
            },
        }
    }

    pub fn correction(&self, segment: u64, text: String) -> Result<Correction> {
        if segment >= self.lines.len() as u64 {
            return Err(Error::invalid_argument("correction must name a finalized segment"));
        }
        let line = &self.lines[segment as usize];
        Ok(Correction {
            segment,
            base_revision: line.revision,
            revision: line
                .revision
                .checked_add(1)
                .ok_or_else(|| Error::resource_exhausted("transcript revision exhausted"))?,
            text,
        })
    }
}

#[cfg(test)]
#[path = "document_test.rs"]
mod tests;
