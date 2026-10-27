use std::collections::HashMap;
use std::sync::Arc;

use inference_error::Error;
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
use crate::service::stream::AudioStreamClient;
use crate::service::stream::AudioTransportStats;
use crate::transcribe::AudioSegmentResp;
use crate::transcribe::AudioTranscriber;

/// Owns streaming VAD, segments, the service connection, bounded jobs, and committed overlap history.
pub struct AudioStream {
    vad: Vad,
    segmenter: AudioSegmenter,
    threshold: f32,
    capacity: usize,
    client: Arc<AudioStreamClient>,
    transcriber: Arc<AudioTranscriber>,
    tasks: JoinSet<Result<AudioSegmentResp>>,
    jobs: HashMap<Id, (AudioSegmentReq, AbortHandle)>,
    committed: watch::Sender<(u64, Vec<Word>)>,
    overlap: u64,
    uploaded_until: u64,
    released_until: u64,
}

impl AudioStream {
    pub async fn load(config: &Config) -> Result<Self> {
        let vad = tokio::task::block_in_place(Vad::load)?;
        let client =
            Arc::new(AudioStreamClient::connect(config.service.clone(), config.streaming.max_in_flight).await?);
        let transcriber = Arc::new(AudioTranscriber::load(config, client.clone())?);
        Ok(Self {
            vad,
            segmenter: AudioSegmenter::new(&config.streaming),
            threshold: config.streaming.speech_threshold,
            capacity: config.streaming.max_in_flight,
            client,
            transcriber,
            tasks: JoinSet::new(),
            jobs: HashMap::new(),
            committed: watch::channel((0, Vec::new())).0,
            overlap: samples(config.streaming.overlap_seconds) as u64,
            uploaded_until: 0,
            released_until: 0,
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

    pub async fn finish_transport(&mut self) -> Result<AudioTransportStats> {
        assert!(
            self.is_drained(),
            "transcription jobs finish before closing audio input"
        );
        self.client.finish().await
    }

    pub fn schedule(&mut self) -> Result<()> {
        for (job, task) in self.jobs.values() {
            if !self.segmenter.is_current(job) {
                task.abort();
            }
        }
        while self.tasks.len() + self.segmenter.num_completed() < self.capacity {
            let Some(job) = self.segmenter.next_request() else {
                break;
            };
            self.submit(job)?;
        }
        self.release(self.retained_sample())?;
        Ok(())
    }

    fn retained_sample(&self) -> u64 {
        self.jobs
            .values()
            .map(|(job, _)| job.audio_start_sample)
            .fold(self.segmenter.retained_sample(), u64::min)
    }

    fn release(&mut self, before: u64) -> Result<()> {
        let before = before.min(self.uploaded_until);
        if before > self.released_until {
            self.client.release(before)?;
            self.released_until = before;
        }
        Ok(())
    }

    fn submit(&mut self, job: AudioSegmentReq) -> Result<()> {
        self.release(self.retained_sample().min(job.audio_start_sample))?;
        let start = self.uploaded_until.max(job.audio_start_sample);
        let end = job.audio_start_sample + job.audio.len() as u64;
        if start < end {
            self.client
                .append(start, &job.audio[(start - job.audio_start_sample) as usize..])?;
            self.uploaded_until = end;
        }
        let transcriber = self.transcriber.clone();
        let committed = self.committed.subscribe();
        let snapshot = job.clone();
        let handle = self
            .tasks
            .spawn(async move { transcriber.process(&snapshot, committed).await });
        self.jobs.insert(handle.id(), (job, handle));
        Ok(())
    }

    pub async fn next(&mut self) -> Result<Vec<AudioEvent>> {
        let client = self.client.clone();
        let pending = !self.tasks.is_empty();
        let joined = tokio::select! {
            _ = client.closed() => return Err(Error::unavailable("audio connection closed")),
            result = self.join(), if pending => result,
        };
        let Some((job, result)) = joined else {
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
