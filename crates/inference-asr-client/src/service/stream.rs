use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

use futures_util::Sink;
use futures_util::SinkExt;
use futures_util::Stream;
use futures_util::StreamExt;
use inference_error::Error;
use inference_error::Result;
use inference_runtime_proto::inference_runtime_service::AudioAppend;
use inference_runtime_proto::inference_runtime_service::AudioFinish;
use inference_runtime_proto::inference_runtime_service::AudioRangeTranscribe;
use inference_runtime_proto::inference_runtime_service::AudioStreamReq;
use inference_runtime_proto::inference_runtime_service::AudioStreamResp;
use inference_runtime_proto::inference_runtime_service::audio_stream_req::Command as AudioCommand;
use inference_runtime_proto::inference_runtime_service::audio_stream_resp::Result as AudioResult;
use inference_runtime_proto::inference_runtime_service::inference_runtime_client::InferenceRuntimeClient;
use prost::Message as _;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::task::AbortHandle;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tonic::Status;
use tonic::transport::Endpoint;

use super::AudioTranscribeResp;
use super::grpc_error;
use crate::config::ServiceConfig;
use crate::config::Transport;

#[derive(Clone, Debug, Default, Serialize)]
pub struct AudioTransportStats {
    pub uploaded_samples: u64,
    pub requested_samples: u64,
    pub peak_buffered_samples: u64,
    pub buffered_samples: u64,
}

type Reply = oneshot::Sender<std::result::Result<AudioStreamResp, Status>>;

enum Command {
    Send(AudioStreamReq),
    Transcribe(AudioRangeTranscribe, Reply),
    Finish(Reply),
}

/// One recording owns one duplex connection. Jobs share its bounded command queue.
pub struct AudioStreamClient {
    input: mpsc::Sender<Command>,
    worker: JoinHandle<()>,
    config: ServiceConfig,
    next_id: AtomicU64,
    stats: Arc<Mutex<AudioTransportStats>>,
}

impl AudioStreamClient {
    pub async fn connect(config: ServiceConfig, capacity: usize) -> Result<Self> {
        let timeout = Duration::from_secs_f64(config.timeout);
        // At most one append, release, recognition and cancellation per admitted job.
        let capacity = capacity * 4 + 4;
        let (input, commands) = mpsc::channel(capacity);
        let stats = Arc::new(Mutex::new(AudioTransportStats::default()));
        let observed = stats.clone();
        let worker = match config.transport {
            Transport::Http => {
                let mut url = reqwest::Url::parse(&config.url).expect("validated ASR URL");
                url.set_path(&format!("{}/stream", url.path().trim_end_matches('/')));
                let scheme = if url.scheme() == "https" { "wss" } else { "ws" };
                url.set_scheme(scheme)
                    .expect("HTTP and WebSocket URL schemes share authority syntax");
                let mut limits = WebSocketConfig::default();
                limits.max_message_size = Some(65536);
                limits.max_frame_size = Some(65536);
                let (socket, _) = tokio::time::timeout(
                    timeout,
                    tokio_tungstenite::connect_async_with_config(url.as_str(), Some(limits), false),
                )
                .await
                .map_err(|_| Error::deadline_exceeded("audio WebSocket connection timed out"))?
                .map_err(|error| Error::unavailable(format!("audio WebSocket connection: {error}")))?;
                let (sink, responses) = socket.split();
                let sink = sink
                    .with(|request: AudioStreamReq| {
                        std::future::ready(Ok::<_, tokio_tungstenite::tungstenite::Error>(Message::Binary(
                            request.encode_to_vec().into(),
                        )))
                    })
                    .sink_map_err(|error| Status::unavailable(format!("audio WebSocket send: {error}")));
                let responses = responses.filter_map(|message| {
                    std::future::ready(match message {
                        Ok(Message::Binary(bytes)) => {
                            Some(
                                AudioStreamResp::decode(bytes)
                                    .map_err(|error| Status::invalid_argument(format!("audio response: {error}"))),
                            )
                        },
                        Ok(Message::Ping(_) | Message::Pong(_) | Message::Close(_)) => None,
                        Ok(_) => {
                            Some(Err(Status::invalid_argument(
                                "audio WebSocket requires protobuf binary messages",
                            )))
                        },
                        Err(error) => Some(Err(Status::unavailable(format!("audio WebSocket receive: {error}")))),
                    })
                });
                tokio::spawn(run(sink, responses, commands, observed, timeout))
            },
            Transport::Grpc => {
                let endpoint = Endpoint::from_shared(config.url.clone())
                    .map_err(|error| Error::invalid_argument(format!("gRPC endpoint: {error}")))?
                    .connect_timeout(timeout);
                let mut client = InferenceRuntimeClient::new(endpoint.connect_lazy()).max_decoding_message_size(65536);
                let (output, requests) = mpsc::channel(capacity);
                // The connection lasts for the recording. Only establishment and individual jobs have deadlines.
                let responses = tokio::time::timeout(timeout, client.stream_audio(ReceiverStream::new(requests)))
                    .await
                    .map_err(|_| Error::deadline_exceeded("audio gRPC connection timed out"))?
                    .map_err(grpc_error)?
                    .into_inner();
                let sink = futures_util::sink::unfold(output, |output, request| {
                    async move {
                        output
                            .send(request)
                            .await
                            .map_err(|_| Status::unavailable("audio gRPC request stream closed"))?;
                        Ok(output)
                    }
                });
                tokio::spawn(run(sink, responses, commands, observed, timeout))
            },
        };
        Ok(Self {
            input,
            worker,
            config,
            next_id: AtomicU64::new(1),
            stats,
        })
    }

    pub fn language(&self) -> Option<&str> {
        self.config.language.as_deref()
    }

    fn send(&self, command: AudioCommand) -> Result<()> {
        self.input
            .try_send(Command::Send(AudioStreamReq { command: Some(command) }))
            .map_err(|error| {
                match error {
                    mpsc::error::TrySendError::Full(_) => Error::resource_exhausted("audio transport cannot keep up"),
                    mpsc::error::TrySendError::Closed(_) => Error::unavailable("audio connection closed"),
                }
            })
    }

    pub fn append(&self, start_sample: u64, pcm: &[i16]) -> Result<()> {
        self.send(AudioCommand::Append(AudioAppend {
            start_sample,
            pcm_s16le: pcm.iter().flat_map(|sample| sample.to_le_bytes()).collect(),
        }))
    }

    pub fn release(&self, before: u64) -> Result<()> {
        self.send(AudioCommand::ReleaseBeforeSample(before))
    }

    pub fn stats(&self) -> AudioTransportStats {
        self.stats.lock().unwrap().clone()
    }

    pub async fn closed(&self) {
        self.input.closed().await;
    }

    pub async fn finish(&self) -> Result<AudioTransportStats> {
        tokio::time::timeout(Duration::from_secs_f64(self.config.timeout), async {
            let (reply, response) = oneshot::channel();
            self.input
                .send(Command::Finish(reply))
                .await
                .map_err(|_| Status::unavailable("audio connection closed"))?;
            response
                .await
                .map_err(|_| Status::unavailable("audio connection closed"))??;
            Ok::<_, Status>(self.stats())
        })
        .await
        .map_err(|_| Error::deadline_exceeded("audio stream finish timed out"))?
        .map_err(grpc_error)
    }

    pub async fn transcribe(&self, start_sample: u64, end_sample: u64, language: &str) -> Result<AudioTranscribeResp> {
        let request_id = self
            .next_id
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| Error::resource_exhausted("audio request IDs exhausted"))?;
        let (reply, response) = oneshot::channel();
        let request = AudioRangeTranscribe {
            request_id,
            start_sample,
            end_sample,
            model: self.config.model.clone(),
            language: (!language.is_empty()).then(|| language.to_owned()),
            context: self.config.context.clone(),
        };
        let result = tokio::time::timeout(Duration::from_secs_f64(self.config.timeout), async {
            self.input
                .send(Command::Transcribe(request, reply))
                .await
                .map_err(|_| Status::unavailable("audio connection closed"))?;
            let mut cancellation = Cancellation {
                request_id,
                input: self.input.clone(),
                worker: self.worker.abort_handle(),
                armed: true,
            };
            let result = response
                .await
                .map_err(|_| Status::unavailable("audio connection closed"))?;
            cancellation.armed = false;
            result
        })
        .await;
        match result {
            Ok(result) => {
                let response = result.map_err(grpc_error)?;
                let Some(AudioResult::Transcription(result)) = response.result else {
                    unreachable!("the worker publishes transcription replies for recognition IDs")
                };
                Ok(AudioTranscribeResp {
                    text: result.text,
                    language: result.language,
                })
            },
            Err(_) => Err(Error::deadline_exceeded("audio transcription timed out")),
        }
    }
}

impl Drop for AudioStreamClient {
    fn drop(&mut self) {
        self.worker.abort();
    }
}

struct Cancellation {
    request_id: u64,
    input: mpsc::Sender<Command>,
    worker: AbortHandle,
    armed: bool,
}

impl Drop for Cancellation {
    fn drop(&mut self) {
        if self.armed {
            let command = AudioStreamReq {
                command: Some(AudioCommand::CancelRequestId(self.request_id)),
            };
            if matches!(
                self.input.try_send(Command::Send(command)),
                Err(mpsc::error::TrySendError::Full(_))
            ) {
                // Close the connection rather than silently losing cancellation under backpressure.
                self.worker.abort();
            }
        }
    }
}

async fn run(
    sink: impl Sink<AudioStreamReq, Error = Status> + Send + 'static,
    responses: impl Stream<Item = std::result::Result<AudioStreamResp, Status>> + Send + 'static,
    mut commands: mpsc::Receiver<Command>,
    stats: Arc<Mutex<AudioTransportStats>>,
    timeout: Duration,
) {
    let pending: Mutex<HashMap<u64, Reply>> = Mutex::new(HashMap::new());
    let mut sink = Box::pin(sink);
    let mut responses = Box::pin(responses);
    let send = async {
        while let Some(command) = commands.recv().await {
            let request = match command {
                Command::Send(request) => {
                    if let Some(AudioCommand::CancelRequestId(id)) = &request.command {
                        pending.lock().unwrap().remove(id);
                    }
                    request
                },
                Command::Transcribe(request, reply) => {
                    if reply.is_closed() {
                        continue;
                    }
                    assert!(
                        pending.lock().unwrap().insert(request.request_id, reply).is_none(),
                        "request IDs are unique"
                    );
                    AudioStreamReq {
                        command: Some(AudioCommand::Transcribe(request)),
                    }
                },
                Command::Finish(reply) => {
                    assert!(
                        pending.lock().unwrap().insert(0, reply).is_none(),
                        "audio input closes once"
                    );
                    AudioStreamReq {
                        command: Some(AudioCommand::Finish(AudioFinish {})),
                    }
                },
            };
            tokio::time::timeout(timeout, sink.send(request))
                .await
                .map_err(|_| Status::deadline_exceeded("audio transport send timed out"))??;
        }
        Err::<(), _>(Status::unavailable("audio request queue closed"))
    };
    let receive = async {
        while let Some(response) = responses.next().await {
            let response = response?;
            if let Some(observed) = &response.stats {
                *stats.lock().unwrap() = AudioTransportStats {
                    uploaded_samples: observed.uploaded_samples,
                    requested_samples: observed.requested_samples,
                    peak_buffered_samples: observed.peak_buffered_samples,
                    buffered_samples: observed.buffered_samples,
                };
            }
            if response.finished {
                if response.request_id != 0 || response.result.is_some() || response.stats.is_none() {
                    return Err(Status::invalid_argument("invalid audio finish response"));
                }
                let mut pending = pending.lock().unwrap();
                if pending.len() != 1 || !pending.contains_key(&0) {
                    return Err(Status::invalid_argument(
                        "audio stream finished before its pending requests",
                    ));
                }
                let _ = pending.remove(&0).unwrap().send(Ok(response));
                return Ok(());
            }
            let request_id = response.request_id;
            let result = match &response.result {
                Some(AudioResult::Transcription(_)) => Ok(response),
                Some(AudioResult::Error(error)) => {
                    Err(Status::new(tonic::Code::from_i32(error.code), error.message.clone()))
                },
                None if response.request_id == 0 => continue,
                None => return Err(Status::invalid_argument("audio response is missing its result")),
            };
            if request_id == 0 {
                return Err(result
                    .err()
                    .unwrap_or_else(|| Status::invalid_argument("unexpected audio acknowledgement")));
            }
            if let Some(reply) = pending.lock().unwrap().remove(&request_id) {
                let _ = reply.send(result);
            }
        }
        Err::<(), _>(Status::unavailable("audio response stream closed"))
    };
    let result = tokio::select! { result = send => result, result = receive => result };
    let pending = pending.into_inner().unwrap();
    match result {
        Ok(()) => assert!(pending.is_empty(), "finished stream has no pending requests"),
        Err(status) => {
            for (_, reply) in pending {
                let _ = reply.send(Err(status.clone()));
            }
        },
    }
}
