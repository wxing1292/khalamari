use std::time::Duration;

use futures_util::StreamExt;
use inference_error::Error;
use inference_error::Result;
use inference_runtime_proto::inference_runtime_service::AudioTranscribeReq as ProtoAudioTranscribeReq;
use inference_runtime_proto::inference_runtime_service::inference_runtime_client::InferenceRuntimeClient;
use reqwest::multipart::Form;
use reqwest::multipart::Part;
use serde::Deserialize;
use serde::Serialize;
use tonic::transport::Channel;
use tonic::transport::Endpoint;

use crate::audio::wav;
use crate::config::ServiceConfig;
use crate::config::Transport;

pub struct AudioTranscribeReq<'a> {
    pub audio: &'a [i16],
    pub language: &'a str,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AudioTranscribeResp {
    pub text: String,
    #[serde(default)]
    pub language: String,
}

pub struct AsrClient {
    client: Client,
    config: ServiceConfig,
}

enum Client {
    Http(reqwest::Client),
    Grpc(InferenceRuntimeClient<Channel>),
}

impl AsrClient {
    pub fn language(&self) -> Option<&str> {
        self.config.language.as_deref()
    }

    pub fn new(config: ServiceConfig) -> Result<Self> {
        let timeout = Duration::from_secs_f64(config.timeout);
        let client = match config.transport {
            Transport::Http => {
                Client::Http(
                    reqwest::Client::builder()
                        .timeout(timeout)
                        .build()
                        .map_err(|e| Error::unavailable(format!("HTTP client: {e}")))?,
                )
            },
            Transport::Grpc => {
                let endpoint = Endpoint::from_shared(config.url.clone())
                    .map_err(|e| Error::invalid_argument(format!("gRPC endpoint: {e}")))?
                    .connect_timeout(timeout)
                    .timeout(timeout);
                Client::Grpc(InferenceRuntimeClient::new(endpoint.connect_lazy()).max_decoding_message_size(65536))
            },
        };
        Ok(Self { client, config })
    }

    pub async fn transcribe(&self, request: AudioTranscribeReq<'_>) -> Result<AudioTranscribeResp> {
        let bytes = wav(request.audio);
        match &self.client {
            Client::Http(client) => self.http(client, bytes, request.language).await,
            Client::Grpc(client) => {
                let mut request = tonic::Request::new(ProtoAudioTranscribeReq {
                    wav: bytes,
                    model: self.config.model.clone(),
                    language: (!request.language.is_empty()).then(|| request.language.to_owned()),
                    context: self.config.context.clone(),
                });
                request.set_timeout(Duration::from_secs_f64(self.config.timeout));
                let result = client
                    .clone()
                    .transcribe_audio(request)
                    .await
                    .map_err(grpc_error)?
                    .into_inner();
                Ok(AudioTranscribeResp {
                    text: result.text,
                    language: result.language,
                })
            },
        }
    }

    async fn http(&self, client: &reqwest::Client, bytes: Vec<u8>, language: &str) -> Result<AudioTranscribeResp> {
        let part = Part::bytes(bytes)
            .file_name("recording.wav")
            .mime_str("audio/wav")
            .unwrap();
        let mut form = Form::new()
            .text("model", self.config.model.clone())
            .text("response_format", "json")
            .part("file", part);
        if !language.is_empty() {
            form = form.text("language", language.to_owned());
        }
        if let Some(context) = &self.config.context {
            form = form.text("prompt", context.clone());
        }
        let response = client
            .post(&self.config.url)
            .multipart(form)
            .send()
            .await
            .map_err(http_error)?;
        let status = response.status();
        let mut stream = response.bytes_stream();
        let mut bytes = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(http_error)?;
            if chunk.len() > 65536 - bytes.len() {
                return Err(Error::resource_exhausted("ASR response exceeds 64 KiB"));
            }
            bytes.extend_from_slice(&chunk);
        }
        if !status.is_success() {
            let message = format!("ASR HTTP {status}: {}", String::from_utf8_lossy(&bytes));
            return Err(match status.as_u16() {
                400 => Error::invalid_argument(message),
                409 => Error::aborted(message),
                429 => Error::resource_exhausted(message),
                500 => Error::internal(message),
                504 => Error::deadline_exceeded(message),
                _ => Error::unavailable(message),
            });
        }
        serde_json::from_slice(&bytes).map_err(|e| Error::invalid_argument(format!("ASR response: {e}")))
    }
}

fn grpc_error(error: tonic::Status) -> Error {
    let message = format!("ASR request: {error}");
    match error.code() {
        tonic::Code::InvalidArgument => Error::invalid_argument(message),
        tonic::Code::ResourceExhausted => Error::resource_exhausted(message),
        tonic::Code::Cancelled => Error::cancelled(message),
        tonic::Code::DeadlineExceeded => Error::deadline_exceeded(message),
        tonic::Code::Aborted => Error::aborted(message),
        tonic::Code::Internal => Error::internal(message),
        _ => Error::unavailable(message),
    }
}

fn http_error(error: reqwest::Error) -> Error {
    if error.is_timeout() {
        Error::deadline_exceeded(format!("ASR request: {error}"))
    } else {
        Error::unavailable(format!("ASR request: {error}"))
    }
}
