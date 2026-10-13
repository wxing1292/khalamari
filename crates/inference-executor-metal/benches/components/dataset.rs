use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Child;
use std::process::Command;
use std::process::Stdio;
use std::thread;
use std::time::Duration;

use serde::Serialize;

use crate::matrix::*;

pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("tmp");
    let mut file = File::create(&temporary).map_err(unavailable)?;
    serde_json::to_writer_pretty(&mut file, value).map_err(unavailable)?;
    writeln!(file).map_err(unavailable)?;
    file.sync_all().map_err(unavailable)?;
    fs::rename(temporary, path).map_err(unavailable)
}

pub fn read_records(path: &Path) -> Result<BTreeMap<Case, Record>> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(error) => return Err(unavailable(error)),
    };
    // Only newline-terminated records were acknowledged. Recover an interrupted final write.
    let end = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |n| n + 1);
    if end != bytes.len() {
        File::options()
            .write(true)
            .open(path)
            .map_err(unavailable)?
            .set_len(end as u64)
            .map_err(unavailable)?;
    }
    let text = std::str::from_utf8(&bytes[..end]).map_err(unavailable)?;
    let mut records = BTreeMap::new();
    for line in text.lines() {
        let record: Record = serde_json::from_str(line).map_err(unavailable)?;
        records.insert(record.case, record);
    }
    Ok(records)
}

fn hash(bytes: &[u8]) -> Result<String> {
    let mut child = Command::new("/usr/bin/shasum")
        .args(["-a", "256"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(unavailable)?;
    let write = child.stdin.take().unwrap().write_all(bytes);
    let output = child.wait_with_output().map_err(unavailable)?;
    write.map_err(unavailable)?;
    if !output.status.success() {
        return Err(Error::unavailable("shasum failed"));
    }
    Ok(String::from_utf8(output.stdout)
        .map_err(unavailable)?
        .split_whitespace()
        .next()
        .unwrap()
        .to_string())
}

fn files(directory: &Path, names: &mut Vec<std::path::PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory).map_err(unavailable)? {
        let path = entry.map_err(unavailable)?.path();
        if path.is_dir() {
            files(&path, names)?;
        } else {
            names.push(path);
        }
    }
    Ok(())
}

pub fn check_identity(args: &Args, root: &Path, model: &Path, worker: &Path) -> Result<()> {
    let mut names = Vec::new();
    for package in [
        "inference-backend-metal",
        "inference-executor-core",
        "inference-executor-metal",
    ] {
        files(&root.join("crates").join(package).join("src"), &mut names)?;
        names.push(root.join("crates").join(package).join("Cargo.toml"));
    }
    names.push(root.join("Cargo.lock"));
    files(&root.join("crates/inference-executor-metal/benches"), &mut names)?;
    names.sort();
    let mut source = Vec::new();
    for path in names {
        source.extend_from_slice(path.strip_prefix(root).unwrap().to_str().unwrap().as_bytes());
        source.extend_from_slice(&fs::read(path).map_err(unavailable)?);
    }
    let mut metadata = Vec::new();
    let mut shards = Vec::new();
    for name in ["config.json", "model.safetensors.index.json"] {
        metadata.extend_from_slice(name.as_bytes());
        metadata.extend_from_slice(&fs::read(model.join(name)).map_err(unavailable)?);
    }
    for entry in fs::read_dir(model).map_err(unavailable)? {
        let path = entry.map_err(unavailable)?.path();
        if path.extension().is_some_and(|s| s == "safetensors") {
            let m = fs::metadata(&path).map_err(unavailable)?;
            shards.push(json!({"name":path.file_name().unwrap(),"bytes":m.len(),"modified":m.modified().map_err(unavailable)?.duration_since(std::time::UNIX_EPOCH).map_err(unavailable)?.as_nanos().to_string()}));
        }
    }
    shards.sort_by_key(|v| v["name"].as_str().unwrap().to_owned());
    let hardware = Command::new("/usr/sbin/system_profiler")
        .args(["-json", "SPDisplaysDataType"])
        .output()
        .map_err(unavailable)?;
    if !hardware.status.success() {
        return Err(Error::unavailable("cannot read GPU identity"));
    }
    let hardware: Value = serde_json::from_slice(&hardware.stdout).map_err(unavailable)?;
    let gpus = hardware["SPDisplaysDataType"]
        .as_array()
        .ok_or_else(|| Error::unavailable("GPU inventory is missing"))?
        .iter()
        .map(|gpu| json!({"name":gpu["sppci_model"],"cores":gpu["sppci_cores"]}))
        .collect::<Vec<_>>();
    let system = Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.model", "hw.memsize", "kern.osrelease"])
        .output()
        .map_err(unavailable)?;
    if !system.status.success() {
        return Err(Error::unavailable("cannot read system identity"));
    }
    let identity = json!({
        "schema":1,"component":args.component,"model":model,"metadata_sha256":hash(&metadata)?,"shards":shards,
        "measurement_sha256":hash(&source)?,"worker_sha256":hash(&fs::read(worker).map_err(unavailable)?)?,
        "gpu":gpus,"system":String::from_utf8(system.stdout).map_err(unavailable)?,
        "max_tokens":args.max_tokens,"gdn_replay":args.gdn_replay,
        "iters":args.iters,"warmup_iters":args.warmup_iters,"runs":args.runs,
    });
    ensure_identity(&args.out.join("identity.json"), &identity)
}

pub fn ensure_identity(path: &Path, identity: &Value) -> Result<()> {
    match fs::read(path) {
        Ok(bytes) => {
            let previous: Value = serde_json::from_slice(&bytes).map_err(unavailable)?;
            if previous != *identity {
                return Err(Error::invalid_argument(
                    "measurement identity changed; use a new output directory",
                ));
            }
            Ok(())
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => write_json(path, identity),
        Err(error) => Err(unavailable(error)),
    }
}

fn swap_counters() -> Result<[u64; 2]> {
    let output = Command::new("/usr/bin/vm_stat").output().map_err(unavailable)?;
    if !output.status.success() {
        return Err(Error::unavailable("vm_stat failed"));
    }
    let text = String::from_utf8(output.stdout).map_err(unavailable)?;
    let read = |name| {
        text.lines()
            .find_map(|line| line.strip_prefix(name))
            .ok_or_else(|| Error::unavailable("swap counter is missing"))?
            .trim()
            .trim_end_matches('.')
            .parse::<u64>()
            .map_err(unavailable)
    };
    Ok([read("Swapins:")?, read("Swapouts:")?])
}

struct Worker(Child);

impl Drop for Worker {
    fn drop(&mut self) {
        if self.0.try_wait().is_ok_and(|status| status.is_none()) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

pub fn provenance(root: &Path) -> Result<Value> {
    let read = |arguments: &[&str]| {
        let output = Command::new("git")
            .current_dir(root)
            .args(arguments)
            .output()
            .map_err(unavailable)?;
        if !output.status.success() {
            return Err(Error::unavailable(String::from_utf8_lossy(&output.stderr)));
        }
        String::from_utf8(output.stdout).map_err(unavailable)
    };
    Ok(json!({
        "commit":read(&["rev-parse", "HEAD"])?.trim(),
        "dirty":read(&["status", "--porcelain"])?,
        "controller_command":std::env::args().collect::<Vec<_>>(),
    }))
}

pub fn measure(args: &Args, executable: &Path, case: Case, command: Vec<String>, provenance: &Value) -> Result<Record> {
    let attempt = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(unavailable)?
        .as_nanos();
    let log = format!("{}-{}-{attempt}.log", args.component.worker(), case.label());
    let output = File::create(args.out.join(&log)).map_err(unavailable)?;
    let swap_before = swap_counters()?;
    let mut worker = Worker(
        Command::new(executable)
            .args(&command)
            .stdout(output.try_clone().map_err(unavailable)?)
            .stderr(output)
            .spawn()
            .map_err(unavailable)?,
    );
    let (status, swap_after, swapped) = loop {
        let completed = worker.0.try_wait().map_err(unavailable)?;
        let current = swap_counters()?;
        let swapped = current.iter().zip(swap_before).any(|(&new, old)| new > old);
        if swapped {
            if let Some(status) = completed {
                break (status, current, true);
            }
            worker.0.kill().map_err(unavailable)?;
            break (worker.0.wait().map_err(unavailable)?, current, true);
        }
        if let Some(status) = completed {
            break (status, current, false);
        }
        thread::sleep(Duration::from_millis(250));
    };
    let text = fs::read_to_string(args.out.join(&log)).map_err(unavailable)?;
    let (status, latencies) = if swapped {
        ("unusable: swap activity".to_owned(), Vec::new())
    } else if !status.success() {
        (format!("failed: {status}"), Vec::new())
    } else {
        match parse_latencies(&text, args.component, case, args.runs) {
            Ok(latencies) => ("measured".to_owned(), latencies),
            Err(error) => (format!("failed: {error}"), Vec::new()),
        }
    };
    Ok(Record {
        case,
        command,
        status,
        latencies,
        log,
        swap_before,
        swap_after,
        provenance: provenance.clone(),
    })
}

pub fn check_component(args: &Args) -> Result<()> {
    let identity = args.out.join("identity.json");
    if identity.exists() {
        let value: Value = serde_json::from_slice(&fs::read(identity).map_err(unavailable)?).map_err(unavailable)?;
        if value["component"] != serde_json::to_value(args.component).map_err(unavailable)?
            || value["gdn_replay"] != args.gdn_replay
        {
            return Err(Error::invalid_argument("output directory belongs to another component"));
        }
    }
    Ok(())
}

pub fn report(args: &Args, cases: &[Case], records: &BTreeMap<Case, Record>) -> Result<()> {
    let mut text = format!(
        "# {:?} component latency\n\nWall latency covers the component replay submission and wait. Setup, model \
         loading, and prefix construction are excluded.\n\n",
        args.component
    );
    text.push_str(
        "GDN uses zero state at ctx=0 and one fixed nonzero state fixture for all positive contexts. Dense MLP/MoE \
         depends on total rows. Equivalent rows reuse one acknowledged measurement.\n\n",
    );
    text.push_str(
        "Full Prefill means one component forward with zero history. It is not full-model Prefill or E2E latency. GDN \
         forward excludes accepted-state reconstruction and cache publication. GQA reports the production lookup; MoE \
         reports complete token-major/expert-major graphs.\n\n",
    );
    let mut report = Vec::new();
    for kind in [Kind::Append, Kind::FullPrefill] {
        for requests in cases
            .iter()
            .filter(|c| c.kind == kind)
            .map(|c| c.requests)
            .collect::<std::collections::BTreeSet<_>>()
        {
            text.push_str(&format!(
                "## {kind:?}, requests = {requests}\n\n| ctx | input/request | total rows | path | latency ms | \
                 status | measurement |\n| ---: | ---: | ---: | --- | ---: | --- | --- |\n"
            ));
            for case in cases.iter().filter(|c| c.kind == kind && c.requests == requests) {
                let key = case.measurement(args.component, args.gdn_replay);
                let record = records.get(&key);
                let status = record.map_or("pending", |r| r.status.as_str());
                report.push(
                    json!({"case":case,"measurement":key,"status":status,"latencies":record.map(|r|&r.latencies)}),
                );
                if let Some(record) = record.filter(|r| r.status == "measured") {
                    for latency in &record.latencies {
                        text.push_str(&format!(
                            "| {} | {} | {} | {} | {:.4} | measured | {} |\n",
                            case.context,
                            case.queries,
                            case.rows(),
                            latency.path,
                            latency.median_ms,
                            key.label()
                        ));
                    }
                } else {
                    text.push_str(&format!(
                        "| {} | {} | {} | — | — | {} | {} |\n",
                        case.context,
                        case.queries,
                        case.rows(),
                        status,
                        key.label()
                    ));
                }
            }
            text.push('\n');
        }
    }
    write_json(&args.out.join("report.json"), &report)?;
    let temporary = args.out.join("report.md.tmp");
    fs::write(&temporary, text).map_err(unavailable)?;
    fs::rename(temporary, args.out.join("report.md")).map_err(unavailable)
}
