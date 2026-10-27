use std::collections::HashMap;

use futures_util::Stream;
use futures_util::StreamExt;
use inference_runtime_core::Error;
use inference_runtime_core::Result;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_proto::inference_runtime_service::AudioStreamError;
use inference_runtime_proto::inference_runtime_service::AudioStreamReq;
use inference_runtime_proto::inference_runtime_service::AudioStreamResp;
use inference_runtime_proto::inference_runtime_service::AudioStreamStats;
use inference_runtime_proto::inference_runtime_service::AudioTranscribeResp;
use inference_runtime_proto::inference_runtime_service::audio_stream_req::Command;
use inference_runtime_proto::inference_runtime_service::audio_stream_resp::Result as AudioResult;
use tokio::task::AbortHandle;
use tokio::task::Id;
use tokio::task::JoinError;
use tokio::task::JoinSet;

use crate::api::audio::AudioBuffer;
use crate::api::audio::AudioTranscriber;
use crate::asr::Transcription;
use crate::rpc::error_code;

// Match the client's supported inference capacity. PCM and queued work have separate bounds.
const MAX_REQUESTS: usize = 64;

enum Event {
    Input(Option<Result<AudioStreamReq>>),
    Completed(std::result::Result<(Id, Result<Transcription>), JoinError>),
}

/// Both transports drive this connection owner. Dropping it releases audio and cancels jobs.
pub fn responses<const N: usize, const L: usize, const P: usize>(
    transcriber: AudioTranscriber<N, L, P>,
    requests: impl Stream<Item = Result<AudioStreamReq>> + Send + 'static,
    shutdown: Shutdown,
) -> impl Stream<Item = Result<AudioStreamResp>> + Send + 'static {
    async_stream::try_stream! {
        let mut requests = Box::pin(requests);
        let mut buffer = AudioBuffer::default();
        let mut stats = AudioStreamStats::default();
        let mut tasks = JoinSet::new();
        let mut jobs: HashMap<Id, (u64, AbortHandle)> = HashMap::new();
        let shutdown = shutdown.async_rx();
        let mut finishing = false;
        let mut input_closed = false;
        loop {
            let event = tokio::select! {
                _ = shutdown.recv() => break,
                request = requests.next(), if !input_closed => Event::Input(request),
                completed = tasks.join_next_with_id(), if !tasks.is_empty() => Event::Completed(completed.unwrap()),
            };
            match event {
                Event::Input(request) => {
                    let Some(request) = request else {
                        if !finishing { break; }
                        input_closed = true;
                        continue;
                    };
                    let command = request?.command.ok_or_else(|| Error::invalid_argument("audio command is required"))?;
                    if finishing && !matches!(&command, Command::CancelRequestId(_)) {
                        Err(Error::invalid_argument("audio input is sealed"))?;
                    }
                    match command {
                        Command::Append(append) => {
                            let samples = (append.pcm_s16le.len() / 2) as u64;
                            buffer.append(append.start_sample, append.pcm_s16le.into())?;
                            stats.uploaded_samples += samples;
                            stats.buffered_samples = buffer.buffered_samples();
                            stats.peak_buffered_samples = stats.peak_buffered_samples.max(buffer.buffered_samples());
                            yield AudioStreamResp { request_id: 0, result: None, stats: Some(stats), finished: false };
                        },
                        Command::ReleaseBeforeSample(before) => {
                            buffer.release(before)?;
                            stats.buffered_samples = buffer.buffered_samples();
                        },
                        Command::Finish(_) => {
                            finishing = true;
                            buffer.clear();
                            stats.buffered_samples = 0;
                        },
                        Command::CancelRequestId(request_id) => {
                            if let Some(task_id) = jobs.iter().find_map(|(task, (id, _))| (*id == request_id).then_some(*task)) {
                                // Cancellation returns admission capacity before the join event is polled.
                                jobs.remove(&task_id).unwrap().1.abort();
                            }
                        },
                        Command::Transcribe(request) => {
                            let id = request.request_id;
                            if id == 0 || jobs.values().any(|(active, _)| *active == id) {
                                Err(Error::invalid_argument("audio request ID must be nonzero and unique while active"))?;
                            }
                            if jobs.len() >= MAX_REQUESTS {
                                Err(Error::resource_exhausted("audio stream has 64 unfinished requests"))?;
                            }
                            let samples = if request.model.trim().is_empty() {
                                Err(Error::invalid_argument("model must not be empty"))
                            } else {
                                buffer.snapshot(request.start_sample, request.end_sample)
                            };
                            match samples {
                                Ok(samples) => {
                                    stats.requested_samples += request.end_sample - request.start_sample;
                                    let transcriber = transcriber.clone();
                                    let handle = tasks.spawn(async move {
                                        transcriber.transcribe_pcm(samples, request.language, request.context).await
                                    });
                                    jobs.insert(handle.id(), (id, handle));
                                },
                                Err(error) => yield error_response(id, error, Some(stats)),
                            }
                        },
                    }
                },
                Event::Completed(completed) => {
                    let (task_id, result) = match completed {
                        Ok((id, result)) => (id, Some(result)),
                        Err(error) if error.is_cancelled() => (error.id(), None),
                        Err(error) => panic!("audio transcription task must not panic: {error}"),
                    };
                    // Cancellation removes admission ownership, including an already-completed task's late result.
                    if let Some((request_id, _)) = jobs.remove(&task_id)
                        && let Some(result) = result {
                        yield match result {
                            Ok(result) => AudioStreamResp {
                                request_id,
                                result: Some(AudioResult::Transcription(AudioTranscribeResp {
                                    text: result.text, language: result.language,
                                })),
                                stats: Some(stats),
                                finished: false,
                            },
                            Err(error) => error_response(request_id, error, Some(stats)),
                        };
                    }
                },
            }
            if finishing && tasks.is_empty() {
                yield AudioStreamResp { request_id: 0, result: None, stats: Some(stats), finished: true };
                break;
            }
        }
    }
}

pub fn error_response(request_id: u64, error: Error, stats: Option<AudioStreamStats>) -> AudioStreamResp {
    AudioStreamResp {
        request_id,
        result: Some(AudioResult::Error(AudioStreamError {
            code: error_code(&error),
            message: error.to_string(),
        })),
        stats,
        finished: false,
    }
}
