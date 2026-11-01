use std::ops::Range;

use inference_error::Result;
use serde::Deserialize;
use serde::Serialize;

use crate::analysis::AudioAnalysisUpdate;
use crate::analysis::AudioSpan;
use crate::recording::AudioRecordingReader;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SpeakerSpan {
    #[serde(rename = "start_sample")]
    pub start: u64,
    #[serde(rename = "end_sample")]
    pub end: u64,
    pub speaker: u32,
}

impl AudioSpan for SpeakerSpan {
    fn range(&self) -> Range<u64> {
        self.start..self.end
    }

    fn set_range(&mut self, range: Range<u64>) {
        self.start = range.start;
        self.end = range.end;
    }
}

pub type AudioDiarizationUpdate = AudioAnalysisUpdate<SpeakerSpan>;

/// One pass owns input, inference policy and speaker memory. VAD is an implementation detail.
/// Replace may revise an unsealed range. Stop followed by diarize flushes accepted input.
pub trait AudioDiarizer: Send {
    fn start(&mut self, recording: AudioRecordingReader);

    fn append(&mut self, range: Range<u64>);

    fn diarize(&mut self) -> Result<Vec<AudioDiarizationUpdate>>;

    fn stop(&mut self);
}
