use std::ops::Range;

use inference_error::Result;

use crate::analysis::AudioAnalysisInput;
use crate::analysis::AudioAnalysisUpdate;
use crate::audio::FRAME_SAMPLES;
use crate::models::Vad;
use crate::recording::AudioRecordingReader;

pub trait AudioSpeechDetector: Send {
    fn start(&mut self, recording: AudioRecordingReader);

    fn append(&mut self, range: Range<u64>);

    fn detect(&mut self) -> Result<Vec<AudioAnalysisUpdate<Range<u64>>>>;

    fn stop(&mut self);
}

pub struct SileroSpeechDetector {
    model: Vad,
    threshold: f32,
    input: Option<AudioAnalysisInput>,
    processed: u64,
}

impl SileroSpeechDetector {
    pub fn load(environment: &ort::environment::Environment, threshold: f32) -> Result<Self> {
        Ok(Self {
            model: Vad::load(environment)?,
            threshold,
            input: None,
            processed: 0,
        })
    }
}

impl AudioSpeechDetector for SileroSpeechDetector {
    fn start(&mut self, recording: AudioRecordingReader) {
        self.model.reset();
        self.input = Some(AudioAnalysisInput::new(recording));
        self.processed = 0;
    }

    fn append(&mut self, range: Range<u64>) {
        self.input.as_mut().unwrap().append(range);
    }

    fn stop(&mut self) {
        self.input.as_mut().unwrap().stop();
    }

    fn detect(&mut self) -> Result<Vec<AudioAnalysisUpdate<Range<u64>>>> {
        let input = self.input.as_ref().unwrap();
        let end = if input.stopped {
            input.end
        } else {
            input.end / FRAME_SAMPLES as u64 * FRAME_SAMPLES as u64
        };
        if end == self.processed {
            return Ok(Vec::new());
        }
        let range = self.processed..end;
        let audio = input.recording.read(range.clone())?;
        let mut spans = Vec::<Range<u64>>::new();
        for frame in audio.chunks(FRAME_SAMPLES) {
            let start = self.processed;
            self.processed += frame.len() as u64;
            if self.model.probability(frame)? >= self.threshold {
                if let Some(last) = spans.last_mut().filter(|last| last.end == start) {
                    last.end = self.processed;
                } else {
                    spans.push(start..self.processed);
                }
            }
        }
        Ok(vec![
            AudioAnalysisUpdate::Replace {
                range: range.clone(),
                spans,
            },
            AudioAnalysisUpdate::Seal { range },
        ])
    }
}
