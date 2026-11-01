use inference_error::Error;
use serde::Deserialize;
use serde::Serialize;

pub mod stream;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AudioTranscribeResp {
    pub text: String,
    #[serde(default)]
    pub language: String,
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
