use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use inference_error::Error;
use inference_error::Result;
use tokio::sync::watch;
use tokio::task::AbortHandle;
use tokio::task::Id;
use tokio::task::JoinSet;

use crate::analysis::AudioAnalysisUpdate;
use crate::analysis::AudioProcessor;
use crate::analysis::AudioReaderTask;
use crate::analysis::AudioTimeline;
use crate::audio::FRAME_SAMPLES;
use crate::config::Config;
use crate::config::samples;
use crate::diarization::AudioDiarizationUpdate;
use crate::diarization::SpeakerSpan;
use crate::diarization::process::SpeakerProcess;
use crate::language::Acoustics;
use crate::language::AudioLanguageRecognizer;
use crate::language::LanguageRange;
use crate::recording::AudioRecordingReader;
use crate::segment::AnalysisPass;
use crate::segment::AudioEvent;
use crate::segment::AudioSegmentTranscribeReq;
use crate::segment::AudioSegmenter;
use crate::service::stream::AudioStreamClient;
use crate::service::stream::AudioTransportStats;
use crate::transcribe::AudioSegmentTranscribeResp;
use crate::transcribe::AudioTranscriber;
use crate::transcript::Word;
use crate::vad::AudioSpeechDetector;
use crate::vad::SileroSpeechDetector;

/// Model weights belong to a recording. Each pass starts fresh recurrent and speaker state.
pub struct SpeechModels {
    vad: AudioProcessor<Box<dyn AudioSpeechDetector>>,
    acoustics: AudioProcessor<Acoustics>,
}

impl SpeechModels {
    pub async fn load(config: &Config) -> Result<Self> {
        let environment = ort::init()
            .build()
            .map_err(|e| Error::unavailable(format!("CPU speech environment: {e}")))?;
        let threshold = config.streaming.speech_threshold;
        let vad = AudioProcessor::load(move || {
            Ok(Box::new(SileroSpeechDetector::load(&environment, threshold)?) as Box<dyn AudioSpeechDetector>)
        })
        .await?;
        let language = config.language.clone();
        let models = config.models.clone();
        let acoustics = AudioProcessor::load(move || Acoustics::load(&language, &models)).await?;
        Ok(Self { vad, acoustics })
    }
}

/// One pass owns its connection, bounded jobs, speaker timeline and segment revisions.
pub struct AudioStream {
    vad: AudioReaderTask<AudioAnalysisUpdate<std::ops::Range<u64>>>,
    diarizer: AudioReaderTask<AudioDiarizationUpdate>,
    _speaker_process: SpeakerProcess,
    lid: Option<AudioReaderTask<AudioAnalysisUpdate<LanguageRange>>>,
    recording: AudioRecordingReader,
    segmenter: AudioSegmenter,
    capacity: usize,
    client: Arc<AudioStreamClient>,
    transcriber: Arc<AudioTranscriber>,
    tasks: JoinSet<Result<AudioSegmentTranscribeResp>>,
    jobs: HashMap<Id, (Arc<AudioSegmentTranscribeReq>, AbortHandle)>,
    completed: BTreeMap<u64, AudioEvent>,
    next_final: u64,
    committed: watch::Sender<(u64, Vec<Word>)>,
    overlap: u64,
    uploaded_until: u64,
    released_until: u64,
    voices: AudioTimeline<std::ops::Range<u64>>,
    speakers: AudioTimeline<SpeakerSpan>,
    languages: AudioTimeline<LanguageRange>,
    speaker: Option<u32>,
    pending_speakers: BTreeMap<u64, AudioEvent>,
}

impl AudioStream {
    pub async fn open(
        config: &Config,
        recording: AudioRecordingReader,
        pass: AnalysisPass,
        models: &mut SpeechModels,
        live_cuts: VecDeque<u64>,
    ) -> Result<Self> {
        let (speaker_process, diarizer) = SpeakerProcess::start(
            &config.models.speaker_command,
            config.models.speaker_model_dir.as_deref(),
            pass,
            recording.clone(),
        )
        .await?;
        let client =
            Arc::new(AudioStreamClient::connect(config.service.clone(), config.streaming.max_in_flight).await?);
        let reader = recording.clone();
        models
            .vad
            .process(move |vad| {
                vad.start(reader);
                Ok(())
            })
            .await?;
        let vad = AudioReaderTask::start(recording.clone(), models.vad.clone(), |vad, range, stopped| {
            if !range.is_empty() {
                vad.append(range);
            }
            if stopped {
                vad.stop();
            }
            vad.detect()
        });
        let lid = if config.service.language.is_none() {
            let reader = recording.clone();
            models
                .acoustics
                .process(move |acoustics| {
                    acoustics.start(reader);
                    Ok(())
                })
                .await?;
            Some(AudioReaderTask::start(
                recording.clone(),
                models.acoustics.clone(),
                |acoustics, range, stopped| {
                    if !range.is_empty() {
                        acoustics.append(range);
                    }
                    if stopped {
                        acoustics.stop();
                    }
                    acoustics.recognize()
                },
            ))
        } else {
            None
        };
        let transcriber = Arc::new(AudioTranscriber::new(config, client.clone(), models.acoustics.clone()));
        Ok(Self {
            vad,
            diarizer,
            _speaker_process: speaker_process,
            lid,
            segmenter: AudioSegmenter::new(&config.streaming, recording.clone(), pass, live_cuts),
            recording,
            capacity: config.streaming.max_in_flight,
            client,
            transcriber,
            tasks: JoinSet::new(),
            jobs: HashMap::new(),
            completed: BTreeMap::new(),
            next_final: 0,
            committed: watch::channel((0, Vec::new())).0,
            overlap: samples(config.streaming.overlap_seconds) as u64,
            uploaded_until: 0,
            released_until: 0,
            voices: AudioTimeline::new(),
            speakers: AudioTimeline::new(),
            languages: AudioTimeline::new(),
            speaker: None,
            pending_speakers: BTreeMap::new(),
        })
    }

    fn schedule(&mut self) -> Result<()> {
        let available = self.voices.sealed.min(self.speakers.end);
        let stopped = self.vad.is_stopped() && self.diarizer.is_stopped();
        while self.segmenter.can_feed() && self.segmenter.end() < available {
            let start = self.segmenter.end();
            let end = (start + FRAME_SAMPLES as u64).min(available);
            if !stopped && end - start < FRAME_SAMPLES as u64 {
                break;
            }
            let speech = self
                .voices
                .spans
                .iter()
                .take_while(|span| span.start < end)
                .any(|span| start < span.end);
            let midpoint = start.midpoint(end);
            let mut active = self
                .speakers
                .spans
                .iter()
                .take_while(|span| span.start <= midpoint)
                .filter(|span| midpoint < span.end);
            let first = active.next();
            let speech = speech
                || first.is_some()
                || self
                    .speakers
                    .spans
                    .iter()
                    .take_while(|span| span.start < end)
                    .any(|span| start < span.end);
            let current = first.filter(|first| active.all(|span| span.speaker == first.speaker));
            let boundary = current.and_then(|span| {
                let previous = self.speaker.replace(span.speaker);
                previous
                    .filter(|previous| *previous != span.speaker)
                    .map(|_| span.start.max(start))
            });
            self.segmenter.feed(end, speech, boundary)?;
        }
        if stopped && self.segmenter.end() == available && self.segmenter.can_feed() && !self.segmenter.is_stopped() {
            self.segmenter.stop();
        }
        for (job, task) in self.jobs.values() {
            if !self.segmenter.is_current(job) {
                task.abort();
            }
        }
        let mut speech = self.voices.spans.iter().filter(|span| span.end > self.segmenter.end());
        let preview = speech
            .clone()
            .next()
            .map(|first| first.start..speech.next_back().unwrap().end);
        while self.tasks.len() + self.completed.len() < self.capacity {
            let Some(job) = self.segmenter.next_request(
                if self.lid.is_some() {
                    self.languages.sealed
                } else {
                    u64::MAX
                },
                preview.clone(),
            )?
            else {
                break;
            };
            self.submit(job)?;
        }
        let retained = self.retained_sample();
        self.release(retained)?;
        self.languages.retain(retained);
        let retained = self
            .pending_speakers
            .values()
            .map(|event| event.start)
            .fold(retained, u64::min);
        self.speakers.retain(retained);
        self.voices.retain(self.segmenter.end());
        Ok(())
    }

    /// Each wait is cancellation safe. The session can handle input or publish its archive between events.
    pub async fn next(&mut self) -> Result<Option<Vec<AudioEvent>>> {
        loop {
            self.schedule()?;
            if self.segmenter.is_stopped()
                && self.diarizer.is_stopped()
                && self.lid.as_ref().is_none_or(AudioReaderTask::is_stopped)
                && self.tasks.is_empty()
                && !self.segmenter.has_pending()
                && self.completed.is_empty()
            {
                return Ok(None);
            }
            let context = self.segmenter.waiting_for_context();
            let client = self.client.clone();
            tokio::select! {
                _ = self.recording.available(context.unwrap_or(0)), if context.is_some() => {},
                updates = self.vad.next(), if !self.vad.is_stopped() => {
                    match updates {
                        Some(updates) => for update in updates? { self.voices.apply(update); },
                        None => assert_eq!(self.voices.sealed, self.recording.end(), "VAD must seal its stopped input"),
                    }
                },
                updates = self.diarizer.next(), if !self.diarizer.is_stopped() => {
                    match updates {
                        Some(updates) => {
                            let mut events = Vec::new();
                            for update in updates? {
                                if let Some(range) = self.speakers.apply(update) {
                                    for event in self.pending_speakers.values_mut().filter(|event| event.start < range.end && range.start < event.end) {
                                        if !event.result.transcript.attribute(&self.speakers.spans) {
                                            continue;
                                        }
                                        let mut revision = event.clone();
                                        revision.kind = "speakers";
                                        revision.reason = "speaker-update";
                                        revision.result.metrics = Default::default();
                                        events.push(revision);
                                    }
                                }
                            }
                            self.pending_speakers.retain(|_, event| event.end > self.speakers.sealed);
                            if !events.is_empty() { return Ok(Some(events)); }
                        },
                        None => assert_eq!(self.speakers.sealed, self.recording.end(), "diarizer must seal its stopped input"),
                    }
                },
                updates = async { self.lid.as_mut().unwrap().next().await }, if self.lid.as_ref().is_some_and(|lid| !lid.is_stopped()) => {
                    match updates {
                        Some(updates) => for update in updates? { self.languages.apply(update); },
                        None => assert_eq!(self.languages.sealed, self.recording.end(), "LID must seal its stopped input"),
                    }
                },
                _ = client.closed() => return Err(Error::unavailable("audio connection closed")),
                completed = self.tasks.join_next_with_id(), if !self.tasks.is_empty() => {
                    let completed = completed.unwrap();
                    let (id, result) = match completed {
                        Ok((id, result)) => (id, Some(result)),
                        Err(error) if error.is_cancelled() => (error.id(), None),
                        Err(error) => panic!("transcription task must not panic: {error}"),
                    };
                    let (job, _) = self.jobs.remove(&id).expect("completion owns its submitted job");
                    if let Some(result) = result {
                        let mut events = Vec::new();
                        if let Some(event) = self.segmenter.accept(&job, result)? {
                            if event.kind == "final" {
                                assert!(event.segment >= self.next_final, "final commits once");
                                assert!(self.completed.insert(event.segment, event).is_none(), "final completes once");
                                while let Some(event) = self.completed.remove(&self.next_final) {
                                    self.next_final += 1;
                                    events.push(event);
                                }
                            } else {
                                events.push(event);
                            }
                        }
                        for event in &mut events {
                            event.result.transcript.attribute(&self.speakers.spans);
                            self.commit(event);
                            if event.kind == "partial" {
                                self.pending_speakers.retain(|_, previous| previous.kind == "final");
                            }
                            if event.end > self.speakers.sealed {
                                self.pending_speakers.insert(event.segment, event.clone());
                            } else {
                                self.pending_speakers.remove(&event.segment);
                            }
                        }
                        return Ok(Some(events));
                    }
                },
            }
        }
    }

    pub async fn stop(self) -> Result<AudioTransportStats> {
        self.client.finish().await
    }

    fn retained_sample(&self) -> u64 {
        self.jobs
            .values()
            .map(|(job, _)| job.audio_start)
            .chain(self.completed.values().map(|event| event.start))
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

    fn submit(&mut self, mut job: AudioSegmentTranscribeReq) -> Result<()> {
        job.languages = self
            .languages
            .spans
            .iter()
            .filter(|range| range.start < job.end && job.start < range.end)
            .map(|range| {
                LanguageRange {
                    start: range.start.max(job.start),
                    end: range.end.min(job.end),
                    ..range.clone()
                }
            })
            .collect();
        // An unfinished LID prefix must not force its last language onto new mixed speech.
        let unknown = self.languages.sealed.max(job.start);
        if self.lid.is_some() && unknown < job.end {
            job.languages.push(LanguageRange {
                start: unknown,
                end: job.end,
                language_hint: String::new(),
                chinese_observed: false,
            });
        }
        debug_assert!(
            job.audio_start >= self.released_until,
            "a request retains all of its PCM"
        );
        self.release(self.retained_sample().min(job.audio_start))?;
        // Submitted jobs already own fully uploaded ranges. Only unsubmitted ranges
        // can need PCM between the previous upload and this job.
        let required = self.segmenter.retained_sample().min(job.audio_start);
        let start = self.uploaded_until.max(required);
        let end = job.audio_start + job.audio.len() as u64;
        if start < end {
            // Preserve all potentially needed PCM behind a leading draft. Confirmed,
            // unneeded prefixes can remain gaps, including arbitrarily long silence.
            self.client.append(&self.recording, start..end)?;
            self.uploaded_until = end;
        }
        let transcriber = self.transcriber.clone();
        let committed = self.committed.subscribe();
        let job = Arc::new(job);
        let snapshot = job.clone();
        let handle = self
            .tasks
            .spawn(async move { transcriber.process(&snapshot, committed).await });
        self.jobs.insert(handle.id(), (job, handle));
        Ok(())
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
            .chain(event.result.transcript.spans.iter().flat_map(|span| &span.words))
            .filter(|word| word.end > event.end.saturating_sub(self.overlap))
            .cloned()
            .collect();
        self.committed.send_replace((event.segment + 1, history));
    }
}
