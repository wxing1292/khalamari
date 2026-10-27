use axum::Extension;
use axum::extract::State;
use axum::extract::WebSocketUpgrade;
use axum::extract::ws::Message;
use axum::response::Response;
use futures_util::SinkExt;
use futures_util::StreamExt;
use inference_runtime_core::Error;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_proto::inference_runtime_service::AudioStreamReq;
use prost::Message as _;

use crate::api::audio::AudioTranscriber;
use crate::api::audio::MAX_AUDIO_BODY_BYTES;
use crate::rpc::audio_stream;

pub async fn handle<const N: usize, const L: usize, const P: usize>(
    State(transcriber): State<AudioTranscriber<N, L, P>>,
    Extension(shutdown): Extension<Shutdown>,
    upgrade: WebSocketUpgrade,
) -> Response {
    upgrade.max_message_size(MAX_AUDIO_BODY_BYTES).on_upgrade(move |socket| async move {
        let (mut output, mut input) = socket.split();
        let requests = async_stream::try_stream! {
            while let Some(message) = input.next().await {
                match message.map_err(|error| Error::unavailable(format!("audio WebSocket: {error}")))? {
                    Message::Binary(bytes) => yield AudioStreamReq::decode(bytes)
                        .map_err(|error| Error::invalid_argument(format!("audio command: {error}")))?,
                    Message::Ping(_) | Message::Pong(_) => {},
                    Message::Close(_) => Err(Error::cancelled("audio WebSocket disconnected"))?,
                    Message::Text(_) => Err(Error::invalid_argument("audio WebSocket requires protobuf binary messages"))?,
                }
            }
        };
        let mut responses = Box::pin(audio_stream::responses(transcriber, requests, shutdown));
        while let Some(response) = responses.next().await {
            let failed = response.is_err();
            let response = response.unwrap_or_else(|error| audio_stream::error_response(0, error, None));
            if output.send(Message::Binary(response.encode_to_vec().into())).await.is_err() || failed {
                break;
            }
        }
        let _ = output.close().await;
    })
}
