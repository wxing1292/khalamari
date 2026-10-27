use std::io::Write;
use std::process::ExitCode;
use std::process::Stdio;
use std::time::Duration;

use inference_error::Error;
use inference_error::Result;

use crate::config::Args;
use crate::output::Output;
use crate::session::AudioSession;
use crate::terminal::TerminalInput;

mod audio;
mod assets;
mod config;
mod document;
mod editor;
mod language;
mod models;
mod output;
mod overlap;
mod segment;
mod service;
mod session;
mod stream;
mod terminal;
mod transcribe;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> ExitCode {
    let result = match cancelled() {
        Ok(cancelled) => {
            tokio::select! {
                result = run() => result,
                _ = cancelled => Err(Error::cancelled("recording cancelled")),
            }
        },
        Err(error) => Err(error),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            if matches!(error, Error::Cancelled(_)) {
                ExitCode::from(130)
            } else {
                ExitCode::FAILURE
            }
        },
    }
}

fn cancelled() -> Result<impl std::future::Future<Output = ()>> {
    let install = |kind| {
        tokio::signal::unix::signal(kind).map_err(|e| Error::unavailable(format!("signal handler initialization: {e}")))
    };
    // Register synchronously before run can enter blocking native initialization.
    let mut interrupt = install(tokio::signal::unix::SignalKind::interrupt())?;
    let mut terminate = install(tokio::signal::unix::SignalKind::terminate())?;
    let mut hangup = install(tokio::signal::unix::SignalKind::hangup())?;
    Ok(async move {
        tokio::select! {
            _ = interrupt.recv() => {},
            _ = terminate.recv() => {},
            _ = hangup.recv() => {},
        }
    })
}

async fn run() -> Result<()> {
    let args = Args::load()?;
    let config = &args.config;
    if args.print_config {
        println!("{}", serde_json::to_string_pretty(&config).unwrap());
        return Ok(());
    }
    if args.setup_models {
        return assets::prepare(&config.language.model_cache, args.whisper_source.as_deref()).await;
    }
    if args.list_devices {
        let output = tokio::process::Command::new(&config.capture.ffmpeg)
            .args(["-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""])
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output();
        let output = tokio::time::timeout(Duration::from_secs(15), output)
            .await
            .map_err(|_| Error::deadline_exceeded("FFmpeg device listing timed out"))?
            .map_err(|e| Error::unavailable(format!("FFmpeg device listing: {e}")))?;
        std::io::stderr().write_all(&output.stderr).map_err(io_error)?;
        return if String::from_utf8_lossy(&output.stderr).contains("AVFoundation audio devices") {
            Ok(())
        } else {
            Err(Error::unavailable("FFmpeg did not return an audio device list"))
        };
    }
    if let Some(path) = &args.replay {
        let mut output = Output::open(&config.output)?;
        output.replay(path)?;
        let mut input = TerminalInput::new(true, true)?.unwrap();
        output.edit(&mut input).await?;
        return output.finish();
    }
    AudioSession::open(args.config).await?.run().await
}

fn io_error(error: std::io::Error) -> Error {
    Error::unavailable(format!("audio I/O: {error}"))
}
