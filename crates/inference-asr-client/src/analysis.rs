use std::ops::Range;
use std::sync::Arc;

use inference_error::Result;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::channel::ShutdownGuard;
use inference_runtime_core::runtime::tasks::SyncTaskPool;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::recording::AudioRecordingReader;

/// Replace is authoritative within its range, including an empty span list.
/// Seal freezes the next contiguous prefix for this analysis pass only.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AudioAnalysisUpdate<T> {
    Replace { range: Range<u64>, spans: Vec<T> },
    Seal { range: Range<u64> },
}

pub trait AudioSpan: Clone {
    fn range(&self) -> Range<u64>;

    fn set_range(&mut self, range: Range<u64>);
}

impl AudioSpan for Range<u64> {
    fn range(&self) -> Range<u64> {
        self.clone()
    }

    fn set_range(&mut self, range: Range<u64>) {
        *self = range;
    }
}

/// Stores labels independently of PCM and of downstream request lifetimes.
/// Analyzers supply replacements ordered by span start; application preserves that order.
pub struct AudioTimeline<T> {
    pub spans: Vec<T>,
    pub end: u64,
    pub sealed: u64,
}

impl<T: AudioSpan> AudioTimeline<T> {
    pub fn new() -> Self {
        Self {
            spans: Vec::new(),
            end: 0,
            sealed: 0,
        }
    }

    pub fn apply(&mut self, update: AudioAnalysisUpdate<T>) -> Option<Range<u64>> {
        match update {
            AudioAnalysisUpdate::Replace { range, spans } => {
                // Analyzer updates cover a continuous timeline and cannot revise a sealed prefix.
                debug_assert!(self.sealed <= range.start && range.start <= self.end && range.start < range.end);
                debug_assert!(spans.iter().all(|span| {
                    let r = span.range();
                    range.start <= r.start && r.start < r.end && r.end <= range.end
                }));
                if range.start == self.end {
                    self.end = range.end;
                    self.spans.extend(spans);
                    return Some(range);
                }
                let mut retained = Vec::with_capacity(self.spans.len() + spans.len());
                for span in self.spans.drain(..) {
                    let old = span.range();
                    if old.end <= range.start || old.start >= range.end {
                        retained.push(span);
                    } else {
                        if old.start < range.start {
                            let mut left = span.clone();
                            left.set_range(old.start..range.start);
                            retained.push(left);
                        }
                        if old.end > range.end {
                            let mut right = span;
                            right.set_range(range.end..old.end);
                            retained.push(right);
                        }
                    }
                }
                retained.extend(spans);
                retained.sort_by_key(|span| span.range().start);
                self.spans = retained;
                self.end = self.end.max(range.end);
                Some(range)
            },
            AudioAnalysisUpdate::Seal { range } => {
                debug_assert_eq!(range.start, self.sealed, "seals advance a contiguous prefix");
                debug_assert!(range.start < range.end && range.end <= self.end);
                self.sealed = range.end;
                None
            },
        }
    }

    pub fn retain(&mut self, start: u64) {
        self.spans.retain(|span| span.range().end > start);
    }
}

/// An analyzer sees only accepted input, even if the writer has published further ahead.
pub struct AudioAnalysisInput {
    pub recording: AudioRecordingReader,
    pub end: u64,
    pub stopped: bool,
}

impl AudioAnalysisInput {
    pub fn new(recording: AudioRecordingReader) -> Self {
        Self {
            recording,
            end: 0,
            stopped: false,
        }
    }

    pub fn append(&mut self, range: Range<u64>) {
        // A reader announces each published sample exactly once, without holes.
        debug_assert!(!self.stopped);
        debug_assert_eq!(range.start, self.end);
        debug_assert!(range.start < range.end && range.end <= self.recording.end());
        self.end = range.end;
    }

    pub fn stop(&mut self) {
        assert!(!self.stopped, "analysis input stops once");
        self.stopped = true;
    }
}

/// One mutable CPU model, one execution slot. Cancellation cannot queue additional native work.
pub struct AudioProcessor<T> {
    inner: Arc<Processor<T>>,
}

struct Processor<T> {
    model: Arc<Mutex<T>>,
    cpu: SyncTaskPool,
    _shutdown: ShutdownGuard,
}

impl<T> Clone for AudioProcessor<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Send + 'static> AudioProcessor<T> {
    pub async fn load(load: impl FnOnce() -> Result<T> + Send + 'static) -> Result<Self> {
        let shutdown = Shutdown::new();
        let cpu = SyncTaskPool::new(shutdown.clone(), 1)?;
        let guard = ShutdownGuard::new(shutdown);
        let model = cpu
            .spawn(load)
            .await
            .expect("CPU pool returns every accepted task result")?;
        Ok(Self {
            inner: Arc::new(Processor {
                model: Arc::new(Mutex::new(model)),
                cpu,
                _shutdown: guard,
            }),
        })
    }

    pub async fn process<R: Send + 'static>(
        &self,
        operation: impl FnOnce(&mut T) -> Result<R> + Send + 'static,
    ) -> Result<R> {
        let mut model = self.inner.model.clone().lock_owned().await;
        self.inner
            .cpu
            .spawn(move || operation(&mut model))
            .await
            .expect("CPU pool returns every accepted task result")
    }
}

/// Each reader owns its cursor. Progress notifications may coalesce; PCM remains in the log.
/// One queued result bounds reader lead without blocking capture or another reader.
pub struct AudioReaderTask<T> {
    updates: mpsc::Receiver<Result<Vec<T>>>,
    task: JoinHandle<()>,
    stopped: bool,
}

impl<T: Send + 'static> AudioReaderTask<T> {
    pub fn start<M: Send + 'static>(
        mut recording: AudioRecordingReader,
        model: AudioProcessor<M>,
        analyze: fn(&mut M, Range<u64>, bool) -> Result<Vec<T>>,
    ) -> Self {
        let (tx, updates) = mpsc::channel(1);
        let task = tokio::spawn(async move {
            let mut cursor = 0;
            loop {
                let available = recording.available(cursor + 1).await;
                // Bound I/O and result batches. Model window geometry stays inside each analyzer.
                let end = available.min(cursor + 16_384);
                // Read the final extent after closure. The writer may append between the wait and this check.
                let stopped = recording.is_sealed() && end == recording.end();
                let result = model.process(move |model| analyze(model, cursor..end, stopped)).await;
                let failed = result.is_err();
                if !matches!(&result, Ok(updates) if updates.is_empty()) && tx.send(result).await.is_err() {
                    break;
                }
                if failed || stopped {
                    break;
                }
                cursor = end;
            }
        });
        Self {
            updates,
            task,
            stopped: false,
        }
    }

    pub async fn next(&mut self) -> Option<Result<Vec<T>>> {
        let updates = self.updates.recv().await;
        self.stopped = updates.is_none();
        updates
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped
    }
}

impl<T> Drop for AudioReaderTask<T> {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[cfg(test)]
#[path = "analysis_test.rs"]
mod tests;
