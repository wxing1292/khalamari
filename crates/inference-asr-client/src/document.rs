use inference_error::Error;
use inference_error::Result;
use serde::Deserialize;
use serde::Serialize;

use crate::audio::SAMPLE_RATE;
use crate::segment::AudioEvent;

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Update {
    Partial(Line),
    Final(Line),
    Correction(Correction),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Line {
    pub segment: u64,
    pub revision: u64,
    pub start_sample: u64,
    pub end_sample: u64,
    pub sample_rate: usize,
    pub text: String,
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

    pub fn observe(&mut self, event: &AudioEvent) {
        let line = Line {
            segment: event.segment,
            revision: event.revision,
            start_sample: event.start_sample,
            end_sample: event.end_sample,
            sample_rate: event.sample_rate,
            text: event.transcript.text.clone(),
        };
        self.commit(match event.kind {
            "partial" => Update::Partial(line),
            "final" => Update::Final(line),
            _ => unreachable!("speech session emits recognition events"),
        });
    }

    /// Validates replay and correction input. A closed segment only accepts explicit corrections.
    pub fn apply(&mut self, update: Update) -> Result<()> {
        match &update {
            Update::Partial(line) | Update::Final(line)
                if line.sample_rate != SAMPLE_RATE || line.start_sample >= line.end_sample || line.revision == 0 =>
            {
                return Err(Error::invalid_argument(
                    "transcript event requires a nonempty 16 kHz sample range",
                ));
            },
            Update::Final(line) => {
                if line.segment >= self.lines.len() as u64
                    && (line.segment != self.lines.len() as u64
                        || self
                            .lines
                            .last()
                            .is_some_and(|previous| line.start_sample < previous.end_sample)
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
            Update::Partial(_) => {},
        }
        self.commit(update);
        Ok(())
    }

    /// Apply trusted session output or an update validated/prepared by this owner.
    pub fn commit(&mut self, update: Update) {
        match update {
            Update::Partial(line) => {
                if line.segment >= self.lines.len() as u64
                    && self.draft.as_ref().is_none_or(|draft| {
                        line.segment > draft.segment
                            || (line.segment == draft.segment && line.revision > draft.revision)
                    })
                {
                    self.draft = Some(line);
                }
            },
            Update::Final(line) => {
                if line.segment < self.lines.len() as u64 {
                    return;
                }
                if self.draft.as_ref().is_some_and(|draft| draft.segment <= line.segment) {
                    self.draft = None;
                }
                self.lines.push(line);
            },
            Update::Correction(correction) => {
                let line = &mut self.lines[correction.segment as usize];
                line.text = correction.text;
                line.revision = correction.revision;
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
