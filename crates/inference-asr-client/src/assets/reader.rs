use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::path::PathBuf;
use std::process::Stdio;

use inference_error::Result;
use sha2::Digest;
use sha2::Sha256;
use tokio::process::Child;
use tokio::process::Command;

use crate::assets::asset_error;
use crate::assets::checksum;
use crate::assets::hex_digest;

const REVISION: &str = "5a8f5804c189defe597e8aebb6a82bbfedde9f48";
const SOURCE: &str = include_str!("../../examples/readers/community_1.rs");

/// The cache lock covers lookup, build and publication. Reader code and platform select one immutable entry.
pub async fn prepare(cache: &Path) -> Result<PathBuf> {
    let key = format!(
        "{REVISION}-{}-{}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        hex_digest(&Sha256::digest(SOURCE)),
    );
    let readers = cache.join("community-1/readers");
    let directory = readers.join(key);
    let executable = directory.join("reader");
    let checksum_file = directory.join("reader.sha256");
    if executable.is_file() && checksum_file.is_file() {
        let expected = tokio::fs::read_to_string(&checksum_file).await.map_err(asset_error)?;
        if checksum(&executable).await? == expected.trim() {
            let mut permissions = tokio::fs::metadata(&executable)
                .await
                .map_err(asset_error)?
                .permissions();
            if permissions.mode() & 0o111 == 0 {
                permissions.set_mode(permissions.mode() | 0o100);
                tokio::fs::set_permissions(&executable, permissions)
                    .await
                    .map_err(asset_error)?;
            }
            return Ok(executable);
        }
        eprintln!("Rebuilding invalid speaker reader {}.", executable.display());
    }
    eprintln!(
        "Building Community-1 CPU reader for first use. The result stays in {}.",
        cache.display()
    );
    tokio::fs::create_dir_all(&readers).await.map_err(asset_error)?;
    let staging = tempfile::Builder::new()
        .prefix(".reader-build-")
        .tempdir_in(&readers)
        .map_err(asset_error)?;
    let source = staging.path().join("source");
    run(Command::new("git").args(["init", "--quiet"]).arg(&source)).await?;
    run(Command::new("git")
        .arg("-C")
        .arg(&source)
        .args([
            "fetch",
            "--quiet",
            "--depth",
            "1",
            "https://github.com/avencera/speakrs.git",
            REVISION,
        ])
        .env("GIT_TERMINAL_PROMPT", "0"))
    .await?;
    run(Command::new("git")
        .arg("-C")
        .arg(&source)
        .args(["checkout", "--quiet", "--detach", "FETCH_HEAD"]))
    .await?;
    // Staging mutations must finish before an await can drop their directory on cancellation.
    std::fs::write(source.join("examples/khalamari_reader.rs"), SOURCE).map_err(asset_error)?;
    let build = cache.join("community-1/build");
    run(Command::new("cargo")
        .current_dir(&source)
        .args([
            "build",
            "--release",
            "--locked",
            "--no-default-features",
            "--features",
            "cpu",
            "--example",
            "khalamari_reader",
        ])
        .arg("--manifest-path")
        .arg(source.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(&build))
    .await?;
    let prepared = staging.path().join("prepared");
    std::fs::create_dir(&prepared).map_err(asset_error)?;
    std::fs::copy(build.join("release/examples/khalamari_reader"), prepared.join("reader")).map_err(asset_error)?;
    let digest = checksum(&prepared.join("reader")).await?;
    std::fs::write(prepared.join("reader.sha256"), format!("{digest}\n")).map_err(asset_error)?;
    if directory.exists() {
        std::fs::remove_dir_all(&directory).map_err(asset_error)?;
    }
    // Complete namespace changes before cancellation can release the cache lock.
    std::fs::rename(prepared, directory).map_err(asset_error)?;
    Ok(executable)
}

/// Cancellation must terminate Cargo's compiler children as well as the parent process.
struct PreparationProcess {
    child: Child,
}

impl Drop for PreparationProcess {
    fn drop(&mut self) {
        if let Some(id) = self.child.id() {
            // Each preparation command starts a separate process group owned by this guard.
            unsafe { libc::kill(-(id as i32), libc::SIGKILL) };
        }
    }
}

async fn run(command: &mut Command) -> Result<()> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    command.as_std_mut().process_group(0);
    let mut process = PreparationProcess {
        child: command.spawn().map_err(|error| {
            asset_error(format!(
                "cannot start {}: {error}",
                command.as_std().get_program().to_string_lossy()
            ))
        })?,
    };
    let status = process.child.wait().await.map_err(asset_error)?;
    if !status.success() {
        return Err(asset_error(format!(
            "{} exited with {status}",
            command.as_std().get_program().to_string_lossy()
        )));
    }
    Ok(())
}
