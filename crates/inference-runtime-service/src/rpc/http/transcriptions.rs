use axum::Json;
use axum::extract::Multipart;
use axum::extract::State;
use axum::extract::multipart::Field;
use axum::extract::multipart::MultipartRejection;
use axum::response::IntoResponse;
use axum::response::Response;

use crate::api::audio::AudioTranscribeReq;
use crate::api::audio::AudioTranscriber;
use crate::rpc::http::error::HTTPError;
use crate::rpc::http::error::invalid_request;
use crate::rpc::http::error::map_error;

pub async fn handle<const N: usize, const L: usize, const P: usize>(
    State(transcriber): State<AudioTranscriber<N, L, P>>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Response, HTTPError> {
    let mut multipart =
        multipart.map_err(|error| invalid_request(format!("invalid multipart transcription request: {error}")))?;
    let mut file = None;
    let mut model = None;
    let mut language = None;
    let mut prompt = None;
    let mut response_format = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| invalid_request(format!("invalid multipart transcription request: {error}")))?
    {
        let name = field
            .name()
            .ok_or_else(|| invalid_request("transcription multipart field must have a name"))?
            .to_string();
        match name.as_str() {
            "file" => {
                if file.is_some() {
                    return Err(invalid_request("transcription request must contain one file field"));
                }
                file = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|error| invalid_request(format!("unable to read transcription file: {error}")))?
                        .to_vec(),
                );
            },
            "model" => {
                if model.is_some() {
                    return Err(invalid_request("transcription request must contain one model field"));
                }
                model = Some(field_text(field, "model").await?);
            },
            "language" => {
                if language.is_some() {
                    return Err(invalid_request(
                        "transcription request must contain at most one language field",
                    ));
                }
                language = Some(field_text(field, "language").await?);
            },
            "prompt" => {
                if prompt.is_some() {
                    return Err(invalid_request(
                        "transcription request must contain at most one prompt field",
                    ));
                }
                prompt = Some(field_text(field, "prompt").await?);
            },
            "response_format" => {
                if response_format.is_some() {
                    return Err(invalid_request(
                        "transcription request must contain at most one response_format field",
                    ));
                }
                response_format = Some(field_text(field, "response_format").await?);
            },
            _ => {
                return Err(invalid_request(format!("unsupported transcription field {name:?}")));
            },
        }
    }

    let file = file.ok_or_else(|| invalid_request("transcription request requires a WAV file field"))?;
    let model = model.ok_or_else(|| invalid_request("transcription request requires a model field"))?;
    let response_format = response_format.unwrap_or_else(|| "json".to_string());
    if !matches!(response_format.as_str(), "json" | "text") {
        return Err(invalid_request("Qwen3-ASR response_format must be json or text"));
    }
    let request = AudioTranscribeReq::new(file, &model, language, prompt).map_err(map_error)?;
    let transcription = transcriber.transcribe(request).await.map_err(map_error)?;
    if response_format == "text" {
        Ok(transcription.text.into_response())
    } else {
        Ok(Json(transcription).into_response())
    }
}

async fn field_text(field: Field<'_>, name: &str) -> Result<String, HTTPError> {
    field
        .text()
        .await
        .map_err(|error| invalid_request(format!("unable to read transcription {name}: {error}")))
}
