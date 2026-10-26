use std::net::SocketAddr;
use std::sync::Arc;

use inference_runtime_core::Result;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::log_err_unavailable;

use crate::api::Inference;
use crate::asr::Qwen3ASRService;
use crate::codec::qwen::QwenCodec;

mod grpc;
mod http;

#[derive(Clone)]
pub enum RPCService {
    Messages(Arc<QwenCodec>),
    Transcriptions(Arc<Qwen3ASRService>),
}

pub async fn run<const N: usize, const L: usize, const P: usize>(
    grpc_listen_addr: SocketAddr,
    http_listen_addr: SocketAddr,
    model_name: String,
    service: RPCService,
    inference: Arc<Inference<N, L, P>>,
    shutdown: Shutdown,
) -> Result<()> {
    let grpc_service = service.clone();
    let grpc_inference = inference.clone();
    let grpc_shutdown = shutdown.clone();
    let grpc_server = async move {
        let result = grpc::run(grpc_listen_addr, grpc_inference, grpc_service, grpc_shutdown.clone())
            .await
            .map_err(|error| log_err_unavailable!("gRPC server failed: {error}"));
        grpc_shutdown.shutdown();
        result
    };
    let http_shutdown = shutdown.clone();
    let http_server = async move {
        let result = http::run(http_listen_addr, model_name, inference, service, http_shutdown.clone())
            .await
            .map_err(|error| log_err_unavailable!("HTTP server failed: {error}"));
        http_shutdown.shutdown();
        result
    };

    let servers = async move { tokio::join!(grpc_server, http_server) };
    tokio::pin!(servers);
    let ((grpc_result, http_result), signal_result) = tokio::select! {
        results = &mut servers => (results, Ok(())),
        signal_result = shutdown_signal() => {
            shutdown.shutdown();
            (servers.await, signal_result)
        },
    };
    signal_result?;
    grpc_result?;
    http_result?;
    Ok(())
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::SignalKind;

        let mut terminate = tokio::signal::unix::signal(SignalKind::terminate())
            .map_err(|error| log_err_unavailable!("unable to install SIGTERM handler: {error}"))?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                result.map_err(|error| log_err_unavailable!("unable to install SIGINT handler: {error}"))
            },
            signal = terminate.recv() => {
                signal
                    .map(|_| ())
                    .ok_or_else(|| log_err_unavailable!("SIGTERM handler closed without a signal"))
            },
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c()
        .await
        .map_err(|error| log_err_unavailable!("unable to install shutdown handler: {error}"))
}
