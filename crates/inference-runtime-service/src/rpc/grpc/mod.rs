use std::sync::Arc;

use inference_runtime_core::Error;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_proto::inference_runtime_service::inference_runtime_server::InferenceRuntimeServer;
use tonic::Status;
use tonic::transport::Server;
use tracing::Instrument;

use crate::api::Inference;
use crate::api::audio::AudioTranscriber;
use crate::api::audio::MAX_AUDIO_BODY_BYTES;
use crate::api::messages::MessageGenerator;
use crate::rpc::RPCService;

mod generation;

#[derive(Clone)]
struct GRPCServer<const N: usize, const L: usize, const P: usize> {
    inference: Arc<Inference<N, L, P>>,
    message_generator: Option<MessageGenerator<N, L, P>>,
    audio_transcriber: Option<AudioTranscriber<N, L, P>>,
}

impl<const N: usize, const L: usize, const P: usize> GRPCServer<N, L, P> {
    fn new(inference: Arc<Inference<N, L, P>>, service: RPCService) -> Self {
        let (message_generator, audio_transcriber) = match service {
            RPCService::Messages(codec) => (Some(MessageGenerator::new(inference.clone(), codec)), None),
            RPCService::Transcriptions(asr) => (None, Some(AudioTranscriber::new(inference.clone(), asr))),
        };
        Self {
            inference,
            message_generator,
            audio_transcriber,
        }
    }
}

pub async fn run<const N: usize, const L: usize, const P: usize>(
    listen_addr: std::net::SocketAddr,
    inference: Arc<Inference<N, L, P>>,
    service: RPCService,
    shutdown: Shutdown,
) -> Result<(), tonic::transport::Error> {
    async move {
        tracing::info!("started");
        let audio_enabled = matches!(&service, RPCService::Transcriptions(_));
        let service = InferenceRuntimeServer::new(GRPCServer::new(inference, service));
        let service = if audio_enabled {
            service.max_decoding_message_size(MAX_AUDIO_BODY_BYTES)
        } else {
            service
        };
        let result = Server::builder()
            .add_service(service)
            .serve_with_shutdown(listen_addr, async move {
                let _ = shutdown.async_rx().recv().await;
            })
            .await;
        tracing::info!("stopped");
        result
    }
    .instrument(tracing::info_span!("grpc-server", %listen_addr))
    .await
}

fn map_error(error: Error) -> Status {
    match error {
        Error::InvalidArgument(message) => Status::invalid_argument(message),
        Error::ResourceExhausted(message) => Status::resource_exhausted(message),
        Error::Cancelled(message) => Status::cancelled(message),
        Error::DeadlineExceeded(message) => Status::deadline_exceeded(message),
        Error::Aborted(message) => Status::aborted(message),
        Error::Evicted(message) => Status::aborted(message),
        Error::Unavailable(message) => Status::unavailable(message),
        Error::Internal(message) => Status::internal(message),
    }
}
