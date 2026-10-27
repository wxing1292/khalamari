use std::sync::Arc;

use axum::Extension;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::get;
use axum::routing::post;
use inference_runtime_core::channel::Shutdown;
use tracing::Instrument;

use crate::api::Inference;
use crate::api::audio::AudioTranscriber;
use crate::api::audio::MAX_AUDIO_BODY_BYTES;
use crate::api::messages::MessageGenerator;
use crate::codec::qwen::QwenCodec;
use crate::rpc::RPCService;

mod audio_stream;
mod chat_completions;
mod error;
mod transcriptions;

const MAX_CHAT_COMPLETIONS_BODY_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone)]
struct ChatCompletionsServer<const N: usize, const L: usize, const P: usize> {
    model_name: Arc<str>,
    message_generator: MessageGenerator<N, L, P>,
}

impl<const N: usize, const L: usize, const P: usize> ChatCompletionsServer<N, L, P> {
    fn new(model_name: String, inference: Arc<Inference<N, L, P>>, qwen_codec: Arc<QwenCodec>) -> Self {
        Self {
            model_name: model_name.into(),
            message_generator: MessageGenerator::new(inference, qwen_codec),
        }
    }

    fn router(self) -> Router {
        Router::<Self>::new()
            .route("/v1/chat/completions", post(chat_completions::handle::<N, L, P>))
            .layer(DefaultBodyLimit::max(MAX_CHAT_COMPLETIONS_BODY_BYTES))
            .with_state(self)
    }
}

pub async fn run<const N: usize, const L: usize, const P: usize>(
    listen_addr: std::net::SocketAddr,
    model_name: String,
    inference: Arc<Inference<N, L, P>>,
    service: RPCService,
    shutdown: Shutdown,
) -> Result<(), std::io::Error> {
    let listener = tokio::net::TcpListener::bind(listen_addr).await?;
    let router = match service {
        RPCService::Messages(qwen_codec) => ChatCompletionsServer::new(model_name, inference, qwen_codec).router(),
        RPCService::Transcriptions(asr) => {
            Router::<AudioTranscriber<N, L, P>>::new()
                .route("/v1/audio/transcriptions", post(transcriptions::handle::<N, L, P>))
                .route("/v1/audio/transcriptions/stream", get(audio_stream::handle::<N, L, P>))
                .layer(DefaultBodyLimit::max(MAX_AUDIO_BODY_BYTES))
                .layer(Extension(shutdown.clone()))
                .with_state(AudioTranscriber::new(inference, asr))
        },
    };
    async move {
        tracing::info!("started");
        let result = axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown.async_rx().recv().await;
            })
            .await;
        tracing::info!("stopped");
        result
    }
    .instrument(tracing::info_span!("http-server", %listen_addr))
    .await
}
