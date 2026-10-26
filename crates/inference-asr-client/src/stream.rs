use std::collections::HashMap;
use std::sync::Arc;

use inference_error::Result;
use tokio::sync::watch;
use tokio::task::AbortHandle;
use tokio::task::Id;
use tokio::task::JoinSet;

use crate::config::Config;
use crate::config::samples;
use crate::models::Vad;
use crate::overlap::Word;
use crate::segment::AudioEvent;
use crate::segment::AudioSegmentReq;
use crate::segment::AudioSegmenter;
use crate::transcribe::AudioSegmentResp;
use crate::transcribe::AudioTranscriber;

/// Owns streaming VAD, segment state, bounded inference tasks, and committed overlap history.
pub struct AudioStream {
    vad: Vad,
    segmenter: AudioSegmenter,
    threshold: f32,
    capacity: usize,
    transcriber: Arc<AudioTranscriber>,
    tasks: JoinSet<Result<AudioSegmentResp>>,
    jobs: HashMap<Id, (AudioSegmentReq, AbortHandle)>,
    committed: watch::Sender<(u64, Vec<Word>)>,
    overlap: u64,
}

impl AudioStream {
    pub fn load(config: &Config) -> Result<Self> {
        Ok(Self {
            vad: Vad::load()?,
            segmenter: AudioSegmenter::new(&config.streaming),
            threshold: config.streaming.speech_threshold,
            capacity: config.streaming.max_in_flight,
            transcriber: Arc::new(AudioTranscriber::load(config)?),
            tasks: JoinSet::new(),
            jobs: HashMap::new(),
            committed: watch::channel((0, Vec::new())).0,
            overlap: samples(config.streaming.overlap_seconds) as u64,
        })
    }

    pub fn feed(&mut self, frame: &[i16]) -> Result<()> {
        let speech = self.vad.probability(frame)? >= self.threshold;
        self.segmenter.feed(frame, speech)
    }

    pub fn finish(&mut self) -> Result<u64> {
        self.segmenter.finish()
    }

    pub fn is_drained(&self) -> bool {
        self.tasks.is_empty() && !self.segmenter.has_pending() && self.segmenter.num_completed() == 0
    }

    pub fn schedule(&mut self) {
        for (job, task) in self.jobs.values() {
            if !self.segmenter.is_current(job) {
                task.abort();
            }
        }
        while self.tasks.len() + self.segmenter.num_completed() < self.capacity {
            let Some(job) = self.segmenter.next_request() else {
                break;
            };
            self.submit(job);
        }
    }

    fn submit(&mut self, job: AudioSegmentReq) {
        let transcriber = self.transcriber.clone();
        let committed = self.committed.subscribe();
        let snapshot = job.clone();
        let handle = self
            .tasks
            .spawn(async move { transcriber.process(&snapshot, committed).await });
        self.jobs.insert(handle.id(), (job, handle));
    }

    pub async fn next(&mut self) -> Result<Vec<AudioEvent>> {
        if self.tasks.is_empty() {
            return std::future::pending().await;
        }
        let Some((job, result)) = self.join().await else {
            return Ok(Vec::new());
        };
        let events = self.segmenter.accept(&job, result)?;
        for event in &events {
            self.commit(event);
        }
        Ok(events)
    }

    async fn join(&mut self) -> Option<(AudioSegmentReq, Result<AudioSegmentResp>)> {
        while let Some(completed) = self.tasks.join_next_with_id().await {
            let (id, result) = match completed {
                Ok((id, result)) => (id, Some(result)),
                Err(error) if error.is_cancelled() => (error.id(), None),
                Err(error) => panic!("transcription task must not panic: {error}"),
            };
            let (job, _) = self.jobs.remove(&id).expect("completed task owns a submitted job");
            if let Some(result) = result {
                return Some((job, result));
            }
            debug_assert!(!job.is_final(), "final tasks are not superseded");
        }
        None
    }

    fn commit(&self, event: &AudioEvent) {
        if event.kind != "final" || self.overlap == 0 {
            return;
        }
        let history = self
            .committed
            .borrow()
            .1
            .iter()
            .chain(
                event
                    .transcript
                    .spans
                    .iter()
                    .filter_map(|span| span.overlap.as_ref())
                    .flat_map(|span| &span.words),
            )
            .filter(|word| word.end_sample > event.end_sample.saturating_sub(self.overlap))
            .cloned()
            .collect();
        self.committed.send_replace((event.segment + 1, history));
    }
}
