use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
use std::process::Stdio;
use std::sync::LazyLock;
use std::time::Duration;
use std::time::Instant;

use clap::Parser;
use futures_util::StreamExt;
use inference_error::Error;
use inference_error::Result;
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use sha2::Digest;
use sha2::Sha256;
use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;
use tokio::process::Command;
use unicode_normalization::UnicodeNormalization;

#[path = "asr_e2e/fixtures.rs"]
mod fixtures;

static LEXEME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\x{3400}-\x{9fff}]|[A-Za-z0-9_]+(?:['’][A-Za-z0-9_]+)*|[^\W\d_]+").unwrap());

#[derive(Parser)]
#[command(about = "Replay real audio through a frozen ASR executable and a running Qwen service")]
struct Args {
    #[arg(long, default_value = "target/release/asr")]
    binary: PathBuf,
    #[arg(long, help = "Source commit of the supplied binary, if known")]
    binary_commit: Option<String>,
    #[arg(long, default_value = "http://127.0.0.1:8000/v1/audio/transcriptions")]
    url: String,
    #[arg(long, default_value = "target/asr-acceptance")]
    output: PathBuf,
    #[arg(long)]
    manifest: Option<PathBuf>,
    #[arg(long)]
    official: bool,
    #[arg(long)]
    paired_overlap: bool,
    #[arg(long, help = "Prepare recordings and manifest without running inference")]
    prepare_only: bool,
    #[arg(long)]
    model_dir: Option<PathBuf>,
}

#[derive(Deserialize, Serialize)]
struct Case {
    name: String,
    file: PathBuf,
    #[serde(default)]
    expected_fragments: Vec<String>,
    #[serde(default)]
    expected_occurrences: BTreeMap<String, usize>,
    reference_text: Option<String>,
    #[serde(default)]
    expected_final_reasons: Vec<String>,
    speech: Option<Vec<[u64; 2]>>,
    // The ASR CLI owns option validation. Unknown flags must fail there.
    #[serde(flatten)]
    flags: BTreeMap<String, Value>,
}
impl Case {
    fn arguments(&self, frames: u64) -> Result<Vec<String>> {
        check(
            self.expected_occurrences
                .keys()
                .all(|phrase| !tokens(phrase).is_empty()),
            "empty repetition reference",
        )?;
        let mut previous_end = 0;
        for &[start, end] in self.speech.iter().flatten() {
            check(
                previous_end <= start && start < end && end <= frames,
                "invalid speech reference interval",
            )?;
            previous_end = end;
        }
        let mut result = Vec::new();
        for (key, value) in &self.flags {
            let value = match value {
                Value::String(value) => value.clone(),
                Value::Number(value) => value.to_string(),
                _ => {
                    return Err(Error::invalid_argument(format!(
                        "case option {key} must be a string or number"
                    )));
                },
            };
            result.extend([format!("--{}", key.replace('_', "-")), value]);
        }
        Ok(result)
    }
}

struct Replay {
    events: Vec<Value>,
    first: Option<Value>,
    received: Vec<f64>,
    capture_start: Option<f64>,
    wall_seconds: f64,
    transport: Value,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Args::parse()).await {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        },
    }
}

async fn run(args: Args) -> Result<bool> {
    std::fs::create_dir_all(&args.output).map_err(external)?;
    let output = args.output.canonicalize().map_err(external)?;
    let mut cases: Vec<Case> = match &args.manifest {
        Some(path) => {
            let path = path.canonicalize().map_err(external)?;
            let directory = path.parent().unwrap();
            let mut cases: Vec<Case> =
                serde_json::from_slice(&std::fs::read(&path).map_err(external)?).map_err(external)?;
            for case in &mut cases {
                if case.file.is_relative() {
                    case.file = directory.join(&case.file);
                }
                if let Some(Value::String(config)) = case.flags.get_mut("config")
                    && Path::new(config).is_relative()
                {
                    *config = directory.join(&*config).to_string_lossy().into_owned();
                }
            }
            cases
        },
        None => fixtures::synthetic(&output, args.paired_overlap).await?,
    };
    if args.official {
        cases.extend(fixtures::official(&output).await?);
    }
    check(!cases.is_empty(), "manifest must contain at least one case")?;
    std::fs::write(output.join("manifest.json"), serde_json::to_vec_pretty(&cases).unwrap()).map_err(external)?;
    if args.prepare_only {
        println!("{}", output.join("manifest.json").display());
        return Ok(true);
    }
    let source_binary = args.binary.canonicalize().map_err(external)?;
    let binary = output.join("asr-replay");
    check(source_binary != binary, "output must not overwrite the source binary")?;
    std::fs::copy(&source_binary, &binary).map_err(external)?;
    let mut report = json!({
        "evaluator_commit": git(&["rev-parse", "HEAD"]).await?,
        "evaluator_dirty": git(&["status", "--short"]).await?,
        "binary_commit": args.binary_commit,
        "binary_source": source_binary, "binary": binary,
        "binary_sha256": file_digest(&binary)?,
        "evaluator_sha256": file_digest(Path::new(file!()))?,
        "cargo_lock_sha256": file_digest(Path::new("Cargo.lock"))?,
        "server_url":args.url,"fixture_source":args.manifest,
        "official_human_examples":args.official,"cases":[]
    });
    if let Some(path) = args.model_dir {
        report["model_config_sha256"] = json!(file_digest(&path.join("config.json"))?);
        report["model_dir"] = json!(path.canonicalize().map_err(external)?);
    }
    let mut passed = true;
    for case in cases {
        let pcm = fixtures::canonical(&case.file).await?;
        let frames = pcm.len() / 2;
        check(
            pcm.len().is_multiple_of(2) && (160..=480000).contains(&frames),
            "fixture must contain 10 ms through 30 seconds of mono i16 audio",
        )?;
        let mut flags = case.arguments(frames as u64)?;
        flags.extend([
            "--url".into(),
            args.url.clone(),
            "--input".into(),
            case.file.to_string_lossy().into_owned(),
        ]);
        let replay_flags = [flags.as_slice(), &["--stream".into(), "--json".into()]].concat();
        let settings: Value = serde_json::from_slice(
            &command(&binary, &[replay_flags.as_slice(), &["--print-config".into()]].concat()).await?,
        )
        .map_err(external)?;
        let baseline: Value = serde_json::from_slice(
            &command(
                &binary,
                &[flags.as_slice(), &["--no-stream".into(), "--json".into()]].concat(),
            )
            .await?,
        )
        .map_err(external)?;
        let replay = replay(&binary, &replay_flags, frames as f64 / 16000.0).await?;
        let mut row = json!({
            "name":case.name,"command":[binary.to_string_lossy().into_owned()],
            "arguments":replay_flags,"settings":settings,
            "audio_seconds":frames as f64 / 16000.0,"audio_sha256":digest(&pcm),
            "baseline":baseline,"events":replay.events,"first_partial":replay.first,
            "wall_seconds":replay.wall_seconds,
            "timing":{"capture_start_seconds":replay.capture_start,"event_receipts_seconds":replay.received}
        });
        if let Some(Value::String(path)) = case.flags.get("config") {
            row["config_sha256"] = json!(file_digest(Path::new(path))?);
        }
        match verify(&case, &replay, frames as u64) {
            Ok(text) => {
                row["protocol_verdict"] = json!("pass");
                row["text"] = json!(text);
                match verify_quality(&case, &text, &mut row) {
                    Ok(()) => row["verdict"] = json!("pass"),
                    Err(error) => {
                        row["verdict"] = json!("fail");
                        row["error"] = json!(error.to_string());
                        passed = false;
                    },
                }
                if let Some(speech) = &case.speech {
                    row["speech_reference"] = json!(speech);
                    row["final_latencies"] = final_latencies(&replay, speech);
                }
                let mut work = serde_json::Map::new();
                for key in [
                    "asr_requests",
                    "submitted_audio_seconds",
                    "lid_windows",
                    "result_cache_hits",
                    "alignment_calls",
                ] {
                    let total = if key == "submitted_audio_seconds" {
                        json!(
                            replay
                                .events
                                .iter()
                                .map(|event| event["metrics"][key].as_f64().unwrap())
                                .sum::<f64>()
                        )
                    } else {
                        json!(
                            replay
                                .events
                                .iter()
                                .map(|event| event["metrics"][key].as_u64().unwrap())
                                .sum::<u64>()
                        )
                    };
                    work.insert(key.into(), total);
                }
                row["work"] = Value::Object(work);
                row["work_scope"] = json!("emitted jobs; discarded and cancelled work is excluded");
                let transport = &replay.transport;
                row["transport"] = transport.clone();
                row["transport"]["uploaded_pcm_bytes"] = json!(transport["uploaded_samples"].as_u64().unwrap() * 2);
                row["transport"]["scope"] = json!("connection totals; include cancelled requests and uploads");
            },
            Err(error) => {
                row["protocol_verdict"] = json!("fail");
                row["verdict"] = json!("fail");
                row["error"] = json!(error.to_string());
                passed = false;
            },
        }
        println!(
            "{}",
            json!({"name":row["name"],"verdict":row["verdict"],"text":row.get("text").or(row.get("error"))})
        );
        report["cases"].as_array_mut().unwrap().push(row);
        std::fs::write(output.join("report.json"), serde_json::to_vec_pretty(&report).unwrap()).map_err(external)?;
    }
    Ok(passed)
}

async fn command(binary: &Path, flags: &[String]) -> Result<Vec<u8>> {
    let output = tokio::time::timeout(
        Duration::from_secs(150),
        Command::new(binary).args(flags).kill_on_drop(true).output(),
    )
    .await
    .map_err(|_| Error::deadline_exceeded("ASR command timed out"))?
    .map_err(external)?;
    if !output.status.success() {
        return Err(Error::unavailable(format!(
            "ASR {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(output.stdout)
}

async fn replay(binary: &Path, flags: &[String], seconds: f64) -> Result<Replay> {
    let started = Instant::now();
    let mut child = Command::new(binary)
        .args(flags)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(external)?;
    let mut out = BufReader::new(child.stdout.take().unwrap()).lines();
    let mut err = BufReader::new(child.stderr.take().unwrap()).lines();
    let mut output_done = false;
    let mut error_done = false;
    let mut diagnostics = String::new();
    let mut result = Replay {
        events: Vec::new(),
        first: None,
        received: Vec::new(),
        capture_start: None,
        wall_seconds: 0.0,
        transport: Value::Null,
    };
    let deadline = tokio::time::sleep(Duration::from_secs_f64(seconds * 3.0 + 60.0));
    tokio::pin!(deadline);
    let status = async {
        while !output_done || !error_done {
            tokio::select! {
                line = out.next_line(), if !output_done => match line.map_err(external)? {
                    Some(line) => {
                        let event: Value = serde_json::from_str(&line).map_err(external)?;
                        let received = started.elapsed().as_secs_f64();
                        if result.first.is_none() && event["type"] == "partial" {
                            result.first = Some(json!({"wall_seconds":received,"audio_end_sample":event["end_sample"],"process_running":child.try_wait().map_err(external)?.is_none()}));
                        }
                        result.received.push(received);
                        result.events.push(event);
                    },
                    None => output_done = true,
                },
                line = err.next_line(), if !error_done => match line.map_err(external)? {
                    Some(line) => {
                        if let Some(stats) = line.strip_prefix("Audio transport: ") {
                            result.transport = serde_json::from_str(stats).map_err(external)?;
                        }
                        if result.capture_start.is_none() && line.starts_with("Streaming. Press Enter to stop") {
                            result.capture_start = Some(started.elapsed().as_secs_f64());
                        }
                        diagnostics.push_str(&line);
                        diagnostics.push('\n');
                    },
                    None => error_done = true,
                },
                _ = &mut deadline => return Err(Error::deadline_exceeded("stream replay timed out")),
                _ = tokio::signal::ctrl_c() => return Err(Error::cancelled("stream replay cancelled")),
            }
        }
        tokio::select! {
            status = child.wait() => status.map_err(external),
            _ = &mut deadline => Err(Error::deadline_exceeded("stream replay did not exit")),
            _ = tokio::signal::ctrl_c() => Err(Error::cancelled("stream replay cancelled")),
        }
    }.await;
    match status {
        Ok(status) if status.success() => {},
        Ok(status) => return Err(Error::unavailable(format!("stream replay {status}: {diagnostics}"))),
        Err(error) => {
            if let Some(pid) = child.id() {
                unsafe {
                    libc::kill(pid as libc::pid_t, libc::SIGTERM);
                }
                if tokio::time::timeout(Duration::from_secs(5), child.wait())
                    .await
                    .is_err()
                {
                    child.kill().await.map_err(external)?;
                }
            }
            return Err(error);
        },
    }
    result.wall_seconds = started.elapsed().as_secs_f64();
    Ok(result)
}

fn verify(case: &Case, replay: &Replay, frames: u64) -> Result<String> {
    check(
        integer(&replay.transport["uploaded_samples"])? <= frames,
        "PCM was uploaded more than once",
    )?;
    check(
        integer(&replay.transport["buffered_samples"])? == 0,
        "finished connection retained PCM",
    )?;
    integer(&replay.transport["requested_samples"])?;
    let mut revisions = BTreeMap::new();
    let mut next_final = 0;
    let mut final_end = 0;
    for event in &replay.events {
        string(&event["text"])?;
        check(
            matches!(event["type"].as_str(), Some("partial" | "final")),
            "unknown event type",
        )?;
        let segment = integer(&event["segment"])?;
        let revision = integer(&event["revision"])?;
        let start = integer(&event["start_sample"])?;
        let end = integer(&event["end_sample"])?;
        check(segment >= next_final, "event after final")?;
        check(
            revision > revisions.get(&segment).copied().unwrap_or(0),
            "revision did not advance",
        )?;
        revisions.insert(segment, revision);
        check(start < end && end <= frames, "invalid audio range")?;
        let mut cursor = start;
        for span in array(&event["spans"])? {
            check(integer(&span["start_sample"])? == cursor, "ASR span gap or overlap")?;
            let span_end = integer(&span["end_sample"])?;
            check(cursor < span_end && span_end <= end, "invalid ASR span range")?;
            cursor = span_end;
            if let Some(words) = span.get("words") {
                let audio_start = integer(&span["audio_start_sample"])?;
                let audio_end = integer(&span["audio_end_sample"])?;
                check(
                    audio_start <= integer(&span["start_sample"])?
                        && cursor <= audio_end
                        && audio_end <= frames
                        && audio_end - audio_start <= 480000,
                    "invalid ASR context range",
                )?;
                let words = array(words)?;
                for word in words {
                    let a = integer(&word["start_sample"])?;
                    let b = integer(&word["end_sample"])?;
                    check(audio_start <= a && a <= b && b <= cursor, "invalid word time")?;
                }
                if words.is_empty() {
                    check(
                        normalize(string(&span["text"])?).is_empty() || span["alignment_status"] == "fallback",
                        "missing alignment without uncertainty",
                    )?;
                } else {
                    let text = words
                        .iter()
                        .map(|word| string(&word["text"]))
                        .collect::<Result<Vec<_>>>()?
                        .join(" ");
                    check(
                        normalize(&text) == normalize(string(&span["text"])?),
                        "word/text mismatch",
                    )?;
                }
            }
        }
        check(cursor == end, "incomplete ASR coverage")?;
        for key in ["asr_requests", "lid_windows", "result_cache_hits", "alignment_calls"] {
            integer(&event["metrics"][key])?;
        }
        let uploaded = integer(&event["transport"]["uploaded_samples"])?;
        check(uploaded <= frames, "PCM was uploaded more than once")?;
        integer(&event["transport"]["requested_samples"])?;
        check(
            integer(&event["transport"]["peak_buffered_samples"])? <= uploaded,
            "invalid PCM cache metric",
        )?;
        check(
            event["metrics"]["submitted_audio_seconds"]
                .as_f64()
                .is_some_and(|v| v.is_finite() && v >= 0.0),
            "invalid submitted audio metric",
        )?;
        if event["type"] == "final" {
            check(segment == next_final, "missing final segment")?;
            check(start >= final_end, "final audio overlap")?;
            final_end = end;
            next_final += 1;
        }
    }
    check(next_final > 0, "no final transcript")?;
    check(next_final as usize == revisions.len(), "unfinished transcript segment")?;
    check(replay.capture_start.is_some(), "missing capture-ready marker")?;
    check(
        replay.first.as_ref().is_some_and(|first| {
            first["process_running"] == true
                && first["audio_end_sample"].as_u64().unwrap() < frames
                && first["wall_seconds"].as_f64().unwrap() < frames as f64 / 16000.0
        }),
        "no live partial",
    )?;
    check(
        replay
            .events
            .iter()
            .any(|event| event["revision"].as_u64().unwrap() > 1),
        "no revisions",
    )?;
    for reason in &case.expected_final_reasons {
        check(
            replay
                .events
                .iter()
                .any(|event| event["type"] == "final" && event["reason"] == *reason),
            "expected endpoint was not exercised",
        )?;
    }
    Ok(final_text(&replay.events))
}

fn verify_quality(case: &Case, text: &str, row: &mut Value) -> Result<()> {
    if let Some(reference) = &case.reference_text {
        row["reference_text"] = json!(reference);
        row["quality"] = reference_error(text, reference);
    }
    let tokens = tokens(text);
    let mut counts = serde_json::Map::new();
    for (phrase, expected) in &case.expected_occurrences {
        let needle = self::tokens(phrase);
        let actual = tokens.windows(needle.len()).filter(|window| *window == needle).count();
        counts.insert(
            phrase.clone(),
            json!({"expected":expected,"actual":actual,"pass":actual==*expected}),
        );
    }
    row["repetition_checks"] = Value::Object(counts);
    for fragment in &case.expected_fragments {
        check(
            normalize(text).contains(&normalize(fragment)),
            &format!("missing spoken fragment {fragment:?}: {text}"),
        )?;
    }
    for (phrase, result) in row["repetition_checks"].as_object().unwrap() {
        check(
            result["pass"] == true,
            &format!("incorrect repetition count {phrase:?}: {result}"),
        )?;
    }
    Ok(())
}

fn final_text(events: &[Value]) -> String {
    events
        .iter()
        .filter(|event| event["type"] == "final")
        .map(|event| event["text"].as_str().unwrap())
        .collect::<Vec<_>>()
        .join(" ")
}
fn normalize(text: &str) -> String {
    text.nfkc()
        .flat_map(char::to_lowercase)
        .filter(|c| c.is_alphanumeric())
        .collect()
}
fn tokens(text: &str) -> Vec<String> {
    let text: String = text.nfkc().flat_map(char::to_lowercase).collect();
    LEXEME.find_iter(&text).map(|token| token.as_str().to_owned()).collect()
}
fn reference_error(text: &str, reference: &str) -> Value {
    let predicted = tokens(text);
    let expected = tokens(reference);
    let mut previous: Vec<_> = (0..=predicted.len()).collect();
    for (index, token) in expected.iter().enumerate() {
        let mut current = vec![index + 1];
        for (other, value) in predicted.iter().enumerate() {
            current.push(
                (current[other] + 1)
                    .min(previous[other + 1] + 1)
                    .min(previous[other] + usize::from(token != value)),
            );
        }
        previous = current;
    }
    json!({"mixed_unit_edits":previous[predicted.len()],"reference_units":expected.len(),"mixed_unit_error_rate":if expected.is_empty(){None}else{Some(previous[predicted.len()] as f64 / expected.len() as f64)}})
}
fn final_latencies(replay: &Replay, speech: &[[u64; 2]]) -> Value {
    let mut turns: Vec<[u64; 2]> = Vec::new();
    let mut previous = 0;
    for &[start, end] in speech {
        if let Some(last) = turns.last_mut().filter(|_| start - previous < 8000) {
            last[1] = end;
        } else {
            turns.push([start, end]);
        }
        previous = end;
    }
    let capture = replay.capture_start.expect("validated capture-ready marker");
    Value::Array(turns.iter().map(|&[start,end]| {
        match replay.events.iter().zip(&replay.received).rfind(|(event,_)| event["type"] == "final" && event["start_sample"].as_u64().unwrap() < end && event["end_sample"].as_u64().unwrap() > start) {
            Some((event,received)) => json!({"start_sample":start,"end_sample":end,"segment":event["segment"],"reason":event["reason"],"speech_to_final_seconds":received-capture-end as f64/16000.0}),
            None => json!({"start_sample":start,"end_sample":end,"missing_final":true}),
        }
    }).collect())
}
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}
fn file_digest(path: &Path) -> Result<String> {
    Ok(digest(&std::fs::read(path).map_err(external)?))
}
async fn git(flags: &[&str]) -> Result<String> {
    let result = Command::new("git").args(flags).output().await.map_err(external)?;
    check(result.status.success(), "Git provenance lookup failed")?;
    Ok(String::from_utf8(result.stdout).map_err(external)?.trim().into())
}
fn check(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::invalid_argument(message))
    }
}
fn integer(value: &Value) -> Result<u64> {
    value
        .as_u64()
        .ok_or_else(|| Error::invalid_argument("expected an unsigned event coordinate"))
}
fn string(value: &Value) -> Result<&str> {
    value
        .as_str()
        .ok_or_else(|| Error::invalid_argument("expected event text"))
}
fn array(value: &Value) -> Result<&Vec<Value>> {
    value
        .as_array()
        .ok_or_else(|| Error::invalid_argument("expected event array"))
}
fn external(error: impl std::fmt::Display) -> Error {
    Error::unavailable(format!("ASR evaluation: {error}"))
}
