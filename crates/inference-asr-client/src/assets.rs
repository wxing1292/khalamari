use std::fs::File;
use std::fs::TryLockError;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use futures_util::StreamExt;
use inference_error::Error;
use inference_error::Result;
use sha2::Digest;
use sha2::Sha256;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;

use crate::config::ModelConfig;

mod reader;

const WHISPER_REVISION: &str = "d90ca5fe260221311c53c58e660288d3deb8d356";
const WHISPER_FILES: &[(&str, u64, &str)] = &[
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

const SPEAKER_REVISION: &str = "67f963b6a1d94fcd4678948237b3cc8bd69bf1d8";
const SPEAKER_FILES: &[(&str, u64, &str)] = &[
    (
        "plda_lda.npy",
        131200,
        "e20c9b012bebd1aabda5a38a127e63a43cf35debdc502715fc143e2fb6bc3c4b",
    ),
    (
        "plda_mean1.npy",
        2176,
        "e424c0c352182aa8e0f555dec1f3b30e29a20b9ed6b25d339f112af92e51e36f",
    ),
    (
        "plda_mean2.npy",
        640,
        "6f6fb708a2037197b5b84ffeaa8f140cb878088fbecd6ab042ad26a7691bd2cf",
    ),
    (
        "plda_mu.npy",
        1152,
        "d286d48acf99bbc1ed1502fed0a3e361ae5626ce1870c8be9f7397c5e47886c6",
    ),
    (
        "plda_phi.npy",
        1152,
        "1f403ada4969a5897b24912e5c25b81ed446b3a513a90135bef3c5c885992677",
    ),
    (
        "plda_psi.npy",
        1152,
        "d7128c9ed2f28a9781971805131129f077c04f948e2df12e52dcdb99f2b4e5f5",
    ),
    (
        "plda_tr.npy",
        131200,
        "e700b68cb319de3fafb5fa093eb9222c23c447084741f8d3a533640d425510ee",
    ),
    (
        "segmentation-3.0.safetensors",
        5897576,
        "88e7883cf56f679170e162419efddf970f929915281ebce80e351c42b5a7115d",
    ),
    (
        "wespeaker-multimask-tail.safetensors",
        26527656,
        "b97b090ceebcf10924732ad4552a2d867d76d38dcc1409a13687a6ba71ff1ef3",
    ),
    (
        "wespeaker-voxceleb-resnet34.min_num_samples.txt",
        4,
        "e4df891c484d7abb985dadf539fa1883a646dab6337af5cae4159c587b7050cc",
    ),
];

/// Prepared resource paths for one recording. Configuration remains caller-owned and unchanged.
#[derive(Clone)]
pub struct SpeechAssets {
    pub whisper: PathBuf,
    pub speaker_command: PathBuf,
    pub speaker_model_dir: Option<PathBuf>,
}

/// Prepare managed assets before capture. Explicit backend and checkpoint paths remain caller-owned.
pub async fn prepare(config: &ModelConfig) -> Result<SpeechAssets> {
    let whisper = whisper_path(config);
    let mut speaker_model_dir = config.speaker_model_dir.clone();
    if config.whisper != "tiny"
        && let Some(command) = &config.speaker_command
    {
        return Ok(SpeechAssets {
            whisper,
            speaker_command: command.clone(),
            speaker_model_dir,
        });
    }
    let cache = std::path::absolute(&config.cache).map_err(asset_error)?;
    tokio::fs::create_dir_all(&cache).await.map_err(asset_error)?;
    let _lock = lock(&cache).await?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(asset_error)?;
    if config.whisper == "tiny" {
        prepare_models(
            &client,
            &whisper,
            "Systran/faster-whisper-tiny",
            WHISPER_REVISION,
            WHISPER_FILES,
        )
        .await?;
    }
    let speaker_command = match &config.speaker_command {
        Some(command) => command.clone(),
        None => {
            if speaker_model_dir.is_none() {
                let models = cache.join("community-1/models").join(SPEAKER_REVISION);
                prepare_models(
                    &client,
                    &models,
                    "avencera/speakrs-models",
                    SPEAKER_REVISION,
                    SPEAKER_FILES,
                )
                .await?;
                speaker_model_dir = Some(models);
            }
            reader::prepare(&cache).await?
        },
    };
    Ok(SpeechAssets {
        whisper,
        speaker_command,
        speaker_model_dir,
    })
}

/// Managed checkpoints have revision-specific paths. Other names and explicit paths remain caller-owned.
fn whisper_path(config: &ModelConfig) -> PathBuf {
    if config.whisper.contains('/') {
        Path::new(&config.whisper).to_path_buf()
    } else if config.whisper == "tiny" {
        config.cache.join("whisper-tiny").join(WHISPER_REVISION)
    } else {
        config.cache.join(format!("whisper-{}", config.whisper))
    }
}

async fn lock(cache: &Path) -> Result<File> {
    let file = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(cache.join(".prepare.lock"))
        .map_err(asset_error)?;
    let mut waiting = false;
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(TryLockError::WouldBlock) => {
                if !waiting {
                    eprintln!("Waiting for another ASR process to prepare {}.", cache.display());
                    waiting = true;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            },
            Err(TryLockError::Error(error)) => return Err(asset_error(error)),
        }
    }
}

async fn prepare_models(
    client: &reqwest::Client,
    directory: &Path,
    repository: &str,
    revision: &str,
    files: &[(&str, u64, &str)],
) -> Result<()> {
    tokio::fs::create_dir_all(directory).await.map_err(asset_error)?;
    for &(name, bytes, digest) in files {
        prepare_file(
            client,
            &directory.join(name),
            &format!("https://huggingface.co/{repository}/resolve/{revision}/{name}"),
            bytes,
            digest,
        )
        .await?;
    }
    Ok(())
}

async fn prepare_file(
    client: &reqwest::Client,
    destination: &Path,
    url: &str,
    bytes: u64,
    expected: &str,
) -> Result<()> {
    match tokio::fs::metadata(destination).await {
        Ok(metadata) => {
            if metadata.len() == bytes && checksum(destination).await? == expected {
                return Ok(());
            }
            eprintln!("Replacing invalid checkpoint {}.", destination.display());
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
        Err(error) => return Err(asset_error(error)),
    }
    let temporary = tempfile::NamedTempFile::new_in(destination.parent().unwrap()).map_err(asset_error)?;
    let (file, temporary) = temporary.into_parts();
    let mut output = tokio::fs::File::from_std(file);
    eprintln!("Downloading {}.", destination.display());
    let response = client
        .get(url)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(asset_error)?;
    let mut input = response.bytes_stream();
    let mut digest = Sha256::new();
    let mut count = 0;
    while let Some(chunk) = input.next().await {
        let chunk = chunk.map_err(asset_error)?;
        // An external response must not exceed the pinned file's byte budget.
        if chunk.len() as u64 > bytes - count {
            return Err(Error::invalid_argument("asset exceeds its expected byte count"));
        }
        output.write_all(&chunk).await.map_err(asset_error)?;
        digest.update(&chunk);
        count += chunk.len() as u64;
    }
    if count != bytes || hex_digest(&digest.finalize()) != expected {
        return Err(Error::invalid_argument(
            "asset does not match its expected size and SHA-256",
        ));
    }
    // Tokio can buffer a file write. Publication must wait until the kernel receives every byte.
    output.flush().await.map_err(asset_error)?;
    drop(output);
    temporary.persist(destination).map_err(asset_error)?;
    Ok(())
}

async fn checksum(path: &Path) -> Result<String> {
    let mut file = tokio::fs::File::open(path).await.map_err(asset_error)?;
    let mut buffer = vec![0; 65536];
    let mut digest = Sha256::new();
    loop {
        let length = file.read(&mut buffer).await.map_err(asset_error)?;
        if length == 0 {
            return Ok(hex_digest(&digest.finalize()));
        }
        digest.update(&buffer[..length]);
    }
}

fn asset_error(error: impl std::fmt::Display) -> Error {
    Error::unavailable(format!("ASR assets: {error}"))
}

fn hex_digest(digest: &[u8]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
