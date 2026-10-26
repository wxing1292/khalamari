use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

use futures_util::Stream;
use futures_util::StreamExt;
use inference_error::Error;
use inference_error::Result;
use sha2::Digest;
use sha2::Sha256;

const REVISION: &str = "d90ca5fe260221311c53c58e660288d3deb8d356";
const FILES: &[(&str, u64, &str)] = &[
    (
        "config.json",
        2249,
        "a73a28cdfe1c43ccc7202fa333d1f89c202477271407ae9a7f19afa52039cac8",
    ),
    (
        "model.bin",
        75538270,
        "dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1",
    ),
    (
        "tokenizer.json",
        2203239,
        "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
    ),
    (
        "vocabulary.txt",
        459861,
        "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
    ),
];

/// Prepare only Whisper. ORT and Silero already belong to the executable.
pub async fn prepare(cache: &Path, source: Option<&Path>) -> Result<()> {
    let directory = cache.join("whisper-tiny");
    std::fs::create_dir_all(&directory).map_err(asset_error)?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(asset_error)?;
    for &(name, bytes, digest) in FILES {
        let destination = directory.join(name);
        if source.is_none() && destination.is_file() {
            match copy_verified(
                file_chunks(File::open(&destination).map_err(asset_error)?),
                std::io::sink(),
                bytes,
                digest,
            )
            .await
            {
                Ok(()) => continue,
                Err(Error::InvalidArgument(error)) => {
                    eprintln!("Replacing invalid checkpoint {}: {error}", destination.display())
                },
                Err(error) => return Err(error),
            }
        }
        let mut temporary = tempfile::NamedTempFile::new_in(&directory).map_err(asset_error)?;
        match source {
            Some(source) => {
                copy_verified(
                    file_chunks(File::open(source.join(name)).map_err(asset_error)?),
                    &mut temporary,
                    bytes,
                    digest,
                )
                .await?;
            },
            None => {
                eprintln!("Downloading Whisper {name}.");
                let response = client
                    .get(format!(
                        "https://huggingface.co/Systran/faster-whisper-tiny/resolve/{REVISION}/{name}"
                    ))
                    .send()
                    .await
                    .and_then(reqwest::Response::error_for_status)
                    .map_err(asset_error)?;
                copy_verified(
                    response.bytes_stream().map(|chunk| chunk.map_err(asset_error)),
                    &mut temporary,
                    bytes,
                    digest,
                )
                .await?;
            },
        }
        temporary.persist(&destination).map_err(asset_error)?;
    }
    eprintln!("Whisper assets ready: {}", directory.display());
    Ok(())
}

async fn copy_verified<B: AsRef<[u8]>>(
    input: impl Stream<Item = Result<B>>,
    mut output: impl Write,
    bytes: u64,
    expected: &str,
) -> Result<()> {
    let mut input = std::pin::pin!(input);
    let mut digest = Sha256::new();
    let mut count = 0;
    while let Some(chunk) = input.next().await {
        let chunk = chunk?;
        let buffer = chunk.as_ref();
        let length = buffer.len();
        // An external reader must not exceed the pinned file's byte budget.
        if length as u64 > bytes - count {
            return Err(Error::invalid_argument("checkpoint exceeds its pinned byte count"));
        }
        output.write_all(buffer).map_err(asset_error)?;
        digest.update(buffer);
        count += length as u64;
    }
    let actual: String = digest.finalize().iter().map(|byte| format!("{byte:02x}")).collect();
    if count != bytes || actual != expected {
        return Err(Error::invalid_argument(
            "checkpoint does not match its pinned size and SHA-256",
        ));
    }
    Ok(())
}

fn file_chunks(file: File) -> impl Stream<Item = Result<Vec<u8>>> {
    futures_util::stream::try_unfold(file, |mut file| {
        async move {
            let mut buffer = vec![0; 65536];
            let length = file.read(&mut buffer).map_err(asset_error)?;
            buffer.truncate(length);
            Ok((length > 0).then_some((buffer, file)))
        }
    })
}

fn asset_error(error: impl std::fmt::Display) -> Error {
    Error::unavailable(format!("Whisper setup: {error}"))
}
