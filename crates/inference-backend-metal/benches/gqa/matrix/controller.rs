//! Offline matrix planning, durable measurement records, and table generation.
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Display;
use std::fs::File;
use std::fs::OpenOptions;
use std::fs::{self};
use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::path::PathBuf;
use std::process::Child;
use std::process::ChildStdin;
use std::process::ChildStdout;
use std::process::Command as Process;
use std::process::Stdio;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Instant;

use clap::Parser;
use clap::ValueEnum;
use inference_error::Error;
use inference_error::Result;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;

mod report;

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Plan,
    Smoke,
    Screen,
    Full,
    /// Measure frozen-policy routes and a small, fixed comparison set. Requires --policy.
    Policy,
    Report,
}

#[derive(Parser)]
#[command(about = "Serial generic GQA matrix, resumable records, reports, and compressed lookup tables")]
struct Args {
    #[arg(long, value_enum, default_value = "plan")]
    mode: Mode,
    #[arg(long, default_value = "target/perf/gqa-matrix")]
    out: PathBuf,
    /// Run budget. Zero means no deadline. Finish the current case before stopping.
    #[arg(long, default_value_t = 0.0)]
    hours: f64,
    /// Maximum new measurements. Zero means no limit.
    #[arg(long, default_value_t = 0)]
    limit: usize,
    #[arg(long, num_args = 1.., default_values = ["0", "64", "256", "1024", "2048", "4096", "8192", "16384", "32768"])]
    history: Vec<u32>,
    #[arg(long, num_args = 1.., default_values = ["1", "2", "3", "4", "5", "6", "7", "8", "12", "16", "32", "64", "128", "256"])]
    queries: Vec<u32>,
    #[arg(long, num_args = 1.., default_values = ["1", "2", "4", "8"])]
    requests: Vec<u32>,
    #[arg(long, num_args = 1.., default_values = ["6", "8", "12"])]
    ratios: Vec<u32>,
    #[arg(long, num_args = 1.., default_values = ["128", "256"])]
    dims: Vec<u32>,
    #[arg(long, num_args = 1.., default_values = ["2", "8"])]
    kv_heads: Vec<u32>,
    #[arg(long, num_args = 1.., default_values = ["16"])]
    pages: Vec<u32>,
    #[arg(long, default_value_t = 8)]
    inner: u32,
    #[arg(long, default_value_t = 5)]
    repeats: u32,
    #[arg(long, default_value_t = 512)]
    max_scratch_mib: u64,
    /// Maximum observed relative-time loss from the best stable candidate.
    #[arg(long, default_value_t = 0.05)]
    max_regret: f64,
    /// Maximum bracket drift and normalized sample range for table candidates.
    #[arg(long, default_value_t = 0.10)]
    max_noise: f64,
    /// Evaluate a frozen export. With --mode policy or plan, use its focused comparison set.
    #[arg(long)]
    policy: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
struct Workload {
    history: u32,
    queries: u32,
    requests: u32,
    ratio: u32,
    dim: u32,
    kv_heads: u32,
    page: u32,
}

impl Workload {
    fn validate(self) -> Result<()> {
        if self.queries == 0
            || self.requests == 0
            || self.ratio == 0
            || self.kv_heads == 0
            || ![128, 256].contains(&self.dim)
            || ![8, 16, 32].contains(&self.page)
            || (self.dim == 128 && self.page == 32)
        {
            return Err(Error::invalid_argument(format!("unsupported workload: {self:?}")));
        }
        // Prove the shader, allocation, and metadata count domains at this input boundary.
        let end = self.history.checked_add(self.queries);
        let heads = self.ratio.checked_mul(self.kv_heads);
        let tokens = self.queries.checked_mul(self.requests);
        let page_bytes = self
            .kv_heads
            .checked_mul(self.page)
            .and_then(|n| n.checked_mul(self.dim))
            .and_then(|n| n.checked_mul(2));
        let pages = end.and_then(|n| n.div_ceil(self.page).checked_mul(self.requests));
        let q_bytes = heads
            .zip(tokens)
            .map(|(h, t)| h as u64 * t as u64)
            .and_then(|n| n.checked_mul(self.dim as u64 * 2));
        let kv_bytes = pages.zip(page_bytes).map(|(p, b)| p as u64 * b as u64);
        if end.is_none()
            || heads.is_none()
            || tokens.is_none()
            || page_bytes.is_none()
            || pages.is_none()
            || q_bytes.is_none_or(|n| n > isize::MAX as u64)
            || kv_bytes.is_none_or(|n| n > isize::MAX as u64)
        {
            return Err(Error::invalid_argument(
                "workload exceeds shader or host allocation domain",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum Layout {
    #[serde(rename = "single")]
    SlicedKv,
    #[serde(rename = "head")]
    QHeadsPerToken,
    #[serde(rename = "token")]
    QTokensPerHead,
    #[serde(rename = "joint")]
    QTokensAndHeads,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
struct Candidate {
    layout: Layout,
    tile: u32,
    heads: u32,
    stage: u32,
    slice: u32,
    direct: bool,
}

impl Candidate {
    fn threads(self) -> u64 {
        match self.layout {
            Layout::SlicedKv => 128,
            Layout::QTokensPerHead => self.tile.div_ceil(8) as u64 * self.heads as u64 * 32,
            Layout::QHeadsPerToken => self.tile as u64 * self.heads.div_ceil(8) as u64 * 32,
            Layout::QTokensAndHeads => (self.tile as u64 * self.heads as u64).div_ceil(8) * 32,
        }
    }
    fn scratch_bytes(self, w: Workload) -> u64 {
        if self.direct {
            return 0;
        }
        let tasks: u64 = (0..w.queries)
            .step_by(self.tile as usize)
            .map(|first| (w.history + first + self.tile.min(w.queries - first)).div_ceil(self.slice) as u64)
            .sum();
        tasks * w.requests as u64 * w.ratio as u64 * w.kv_heads as u64 * self.tile as u64 * (w.dim as u64 * 2 + 8)
    }
    fn baseline(self, w: Workload) -> bool {
        self.layout == Layout::SlicedKv
            || (self.layout == Layout::QTokensPerHead
                && self.tile == 8
                && !self.direct
                && w.ratio <= 8
                && ((self.stage == 16
                    && [w.ratio.div_ceil(2), w.ratio].contains(&self.heads)
                    && (w.dim == 256 || self.heads == w.ratio))
                    || (w.dim == 256 && self.stage == 32 && self.heads == w.ratio)))
    }
    fn screened(self, w: Workload) -> bool {
        let end = w.history + w.queries;
        let slice = self.slice == end || [64, 128, 256, 512, 2048].iter().any(|&n| n.min(end) == self.slice);
        slice
            && (self.baseline(w)
                || ([Layout::QHeadsPerToken, Layout::QTokensAndHeads].contains(&self.layout)
                    && [1, 2, 4, 8].contains(&self.tile)
                    && [w.ratio.div_ceil(2), w.ratio].contains(&self.heads)
                    && [16, 64].contains(&self.stage)))
    }
    fn validate(self, w: Workload) -> Result<()> {
        let valid = if self.layout == Layout::SlicedKv {
            self.tile == 1 && self.heads == w.ratio.min(8) && self.stage == w.dim && !self.direct
        } else {
            [1, 2, 4, 8, 16, 32].contains(&self.tile)
                && [8, 16, 32, 64, 128].contains(&self.stage)
                && [1, w.ratio.div_ceil(2), w.ratio].contains(&self.heads)
        };
        if !valid
            || self.slice == 0
            || self.slice > w.history + w.queries
            || (self.direct && self.slice != w.history + w.queries)
        {
            return Err(Error::invalid_argument(format!(
                "invalid persisted candidate: {self:?}"
            )));
        }
        Ok(())
    }
}

include!("policy.rs");

#[derive(Deserialize, Serialize)]
struct Command {
    workload: Workload,
    candidate: Candidate,
    inner: u32,
    repeats: u32,
}

fn variants(w: Workload) -> Vec<Candidate> {
    let end = w.history + w.queries;
    let slices: BTreeSet<_> = [64, 128, 256, 512, 1024, 2048, 4096, 8192, end]
        .map(|n| n.min(end))
        .into_iter()
        .collect();
    let mut candidates = Vec::new();
    let mut add = |layout, tile, heads, stage| {
        candidates.extend(slices.iter().map(|&slice| {
            Candidate {
                layout,
                tile,
                heads,
                stage,
                slice,
                direct: false,
            }
        }));
        if layout != Layout::SlicedKv {
            candidates.push(Candidate {
                layout,
                tile,
                heads,
                stage,
                slice: end,
                direct: true,
            });
        }
    };
    add(Layout::SlicedKv, 1, w.ratio.min(8), w.dim);
    for layout in [Layout::QHeadsPerToken, Layout::QTokensPerHead, Layout::QTokensAndHeads] {
        for tile in [1, 2, 4, 8, 16, 32] {
            for heads in BTreeSet::from([1, w.ratio.div_ceil(2), w.ratio]) {
                for stage in [8, 16, 32, 64, 128] {
                    add(layout, tile, heads, stage);
                }
            }
        }
    }
    candidates
}

#[derive(Deserialize, Serialize)]
struct Plan {
    input_axes: BTreeMap<String, Vec<u32>>,
    dims: Vec<u32>,
    kv_heads: Vec<u32>,
    pages: Vec<u32>,
    #[serde(default = "screen_mode")]
    mode: Mode,
    #[serde(default = "default_inner")]
    inner: u32,
    #[serde(default = "default_repeats")]
    repeats: u32,
    #[serde(default = "default_scratch")]
    max_scratch_mib: u64,
    #[serde(default)]
    frozen_policy: Option<(Policy, Value)>,
}
fn screen_mode() -> Mode {
    Mode::Screen
}
fn default_inner() -> u32 {
    8
}
fn default_repeats() -> u32 {
    5
}
fn default_scratch() -> u64 {
    512
}

impl Plan {
    fn new(a: &Args) -> Self {
        let axes = if a.mode == Mode::Smoke {
            [
                ("history", vec![64]),
                ("queries", vec![5]),
                ("requests", vec![2]),
                ("ratio", vec![6, 12]),
            ]
        } else {
            [
                ("history", a.history.clone()),
                ("queries", a.queries.clone()),
                ("requests", a.requests.clone()),
                ("ratio", a.ratios.clone()),
            ]
        };
        Self {
            input_axes: axes.into_iter().map(|(n, v)| (n.to_string(), v)).collect(),
            dims: a.dims.clone(),
            kv_heads: a.kv_heads.clone(),
            pages: a.pages.clone(),
            mode: a.mode,
            inner: a.inner,
            repeats: a.repeats,
            max_scratch_mib: a.max_scratch_mib,
            frozen_policy: None,
        }
    }
    fn workloads(&self) -> Vec<Workload> {
        let mut profiles = Vec::new();
        for (b, &requests) in self.input_axes["requests"].iter().enumerate() {
            for (g, &ratio) in self.input_axes["ratio"].iter().enumerate() {
                for (d, &dim) in self.dims.iter().enumerate() {
                    for (h, &kv_heads) in self.kv_heads.iter().enumerate() {
                        for (p, &page) in self.pages.iter().enumerate() {
                            if dim == 128 && page == 32 {
                                continue;
                            }
                            profiles.push((
                                [b, g, d, h, p],
                                Workload {
                                    requests,
                                    ratio,
                                    dim,
                                    kv_heads,
                                    page,
                                    history: 0,
                                    queries: 1,
                                },
                            ));
                        }
                    }
                }
            }
        }
        profiles.sort_by_key(|(i, _)| (i.iter().sum::<usize>(), *i));
        let mut workloads = Vec::new();
        for &queries in &self.input_axes["queries"] {
            for &history in &self.input_axes["history"] {
                for &(_, w) in &profiles {
                    workloads.push(Workload { history, queries, ..w });
                }
            }
        }
        workloads
    }
    fn validate(&self) -> Result<()> {
        if self.mode == Mode::Policy && self.frozen_policy.is_none() {
            return Err(Error::invalid_argument("policy mode requires --policy"));
        }
        if let Some((policy, source)) = &self.frozen_policy {
            let device = source["document"]["device"]
                .as_str()
                .ok_or_else(|| Error::invalid_argument("frozen policy device is missing"))?;
            let validated = decode_frozen_policy(&source["document"], device)?;
            if encoded(policy) != encoded(&validated) {
                return Err(Error::invalid_argument(
                    "saved policy coefficients differ from their source",
                ));
            }
        }
        for name in ["history", "queries", "requests", "ratio"] {
            let values = self
                .input_axes
                .get(name)
                .ok_or_else(|| Error::invalid_argument(format!("missing axis {name}")))?;
            distinct(values, name)?;
        }
        for (name, values) in [
            ("dims", &self.dims),
            ("kv_heads", &self.kv_heads),
            ("pages", &self.pages),
        ] {
            distinct(values, name)?;
        }
        if self.inner == 0
            || self.repeats == 0
            || self.max_scratch_mib == 0
            || self.max_scratch_mib > u64::MAX / 1024 / 1024
        {
            return Err(Error::invalid_argument(
                "inner, repeats, and scratch budget must be positive and fit their domains",
            ));
        }
        if self.dims.iter().any(|d| ![128, 256].contains(d))
            || self.pages.iter().any(|p| ![8, 16, 32].contains(p))
            || self.kv_heads.contains(&0)
        {
            return Err(Error::invalid_argument("unsupported profile axes"));
        }
        let workloads = self.workloads();
        if workloads.is_empty() {
            return Err(Error::invalid_argument("no implemented profile in this plan"));
        }
        for w in workloads {
            w.validate()?;
            // Split output capacities include padded records. Check the largest allocation once.
            let tasks = (w.history as u64 + w.queries as u64)
                .checked_mul(w.queries as u64)
                .and_then(|n| n.checked_mul(w.requests as u64));
            if tasks
                .and_then(|n| n.checked_mul(w.ratio as u64 * w.kv_heads as u64))
                .and_then(|n| n.checked_mul(32 * (w.dim as u64 * 2 + 8)))
                .is_none()
            {
                return Err(Error::invalid_argument("partial-output count exceeds host domain"));
            }
        }
        Ok(())
    }
    fn candidates(&self, w: Workload) -> Vec<Candidate> {
        if let Some((policy, _)) = &self.frozen_policy {
            return policy_candidates(w, *policy);
        }
        variants(w)
            .into_iter()
            .filter(|&c| !matches!(self.mode, Mode::Screen | Mode::Smoke) || c.screened(w))
            .collect()
    }
}
// Controls depend only on the frozen rule and workload, never on measured winners.
// Pair each comparison geometry with adjacent KV slices so geometry and partition stay comparable.
fn policy_candidates(w: Workload, policy: Policy) -> Vec<Candidate> {
    let end = w.history + w.queries;
    let single = Candidate {
        layout: Layout::SlicedKv,
        tile: 1,
        heads: w.ratio.min(8),
        stage: w.dim,
        slice: end,
        direct: false,
    };
    let mut candidates = BTreeSet::from([single]);
    if let Some(candidate) = proposed_candidate(w, policy) {
        candidates.insert(candidate);
        candidates.insert(Candidate {
            slice: candidate.slice,
            ..single
        });
        let tile = if candidate.layout == Layout::SlicedKv {
            w.queries.min(8).next_power_of_two()
        } else {
            candidate.tile
        };
        for layout in [Layout::QHeadsPerToken, Layout::QTokensAndHeads, Layout::QTokensPerHead] {
            candidates.insert(Candidate {
                layout,
                tile: if layout == Layout::QTokensPerHead { 8 } else { tile },
                heads: w.ratio,
                stage: 16,
                slice: candidate.slice,
                direct: candidate.slice == end,
            });
        }
        for comparison in candidates.clone().into_iter().filter(|c| c.slice == candidate.slice) {
            for slice in [(candidate.slice / 2).max(1), (candidate.slice * 2).min(end)] {
                candidates.insert(Candidate {
                    slice,
                    direct: comparison.layout != Layout::SlicedKv && slice == end,
                    ..comparison
                });
            }
        }
        if candidate.direct {
            candidates.insert(Candidate {
                direct: false,
                ..candidate
            });
        }
    }
    candidates.into_iter().collect()
}

fn distinct(values: &[u32], name: &str) -> Result<()> {
    if values.is_empty() || values.iter().collect::<HashSet<_>>().len() != values.len() {
        return Err(Error::invalid_argument(format!("{name} must contain distinct values")));
    }
    Ok(())
}
fn unavailable(e: impl Display) -> Error {
    Error::unavailable(e.to_string())
}
fn encoded(value: &impl Serialize) -> String {
    let mut value = serde_json::to_value(value).unwrap();
    value.sort_all_objects();
    value.to_string()
}
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_reader(File::open(path).map_err(unavailable)?)
        .map_err(|e| Error::invalid_argument(format!("{}: {e}", path.display())))
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    // Same-directory rename keeps report readers from seeing partial JSON.
    let tmp = path.with_extension("tmp");
    let mut file = File::create(&tmp).map_err(unavailable)?;
    serde_json::to_writer_pretty(&mut file, value).map_err(unavailable)?;
    writeln!(file).map_err(unavailable)?;
    file.sync_all().map_err(unavailable)?;
    fs::rename(tmp, path).map_err(unavailable)
}
fn output(root: &Path, program: &str, args: &[&str]) -> Result<String> {
    let out = Process::new(program)
        .args(args)
        .current_dir(root)
        .output()
        .map_err(unavailable)?;
    if !out.status.success() {
        return Err(Error::unavailable(format!(
            "{program}: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    String::from_utf8(out.stdout)
        .map(|s| s.trim().to_owned())
        .map_err(unavailable)
}

// SQLite owns each durable case. The command key uses the original sorted compact JSON format.
struct Database {
    process: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
}
impl Database {
    fn open(path: &Path) -> Result<Self> {
        let mut process = Process::new("/usr/bin/sqlite3")
            .args(["-batch"])
            .process_group(0)
            .arg(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(unavailable)?;
        let input = process.stdin.take().unwrap();
        let stdout = process.stdout.take().unwrap();
        let mut db = Self {
            process,
            input: Some(input),
            output: BufReader::new(stdout),
        };
        db.execute(
            ".bail on\n.headers off\n.mode list\nPRAGMA synchronous=FULL;\nCREATE TABLE IF NOT EXISTS results (key \
             text primary key, command text, status text, result text);",
        )?;
        Ok(db)
    }
    fn execute(&mut self, sql: &str) -> Result<()> {
        let input = self.input.as_mut().unwrap();
        writeln!(input, "{sql}\nSELECT 'committed';").map_err(unavailable)?;
        input.flush().map_err(unavailable)?;
        let mut line = String::new();
        self.output.read_line(&mut line).map_err(unavailable)?;
        if line.trim() != "committed" {
            return Err(Error::unavailable(
                "SQLite write failed; the last case was not acknowledged",
            ));
        }
        Ok(())
    }
    fn insert(&mut self, command: &Command, result: &Value) -> Result<()> {
        let cmd = encoded(command);
        let quote = |s: &str| format!("'{}'", s.replace('\'', "''"));
        self.execute(&format!(
            "INSERT INTO results VALUES({},{},{},{}) ON CONFLICT(key) DO UPDATE SET \
             status=excluded.status,result=excluded.result;",
            quote(&cmd),
            quote(&cmd),
            quote(result["status"].as_str().unwrap()),
            quote(&result.to_string())
        ))
    }
}
impl Drop for Database {
    fn drop(&mut self) {
        drop(self.input.take());
        let _ = self.process.wait();
    }
}

fn records(path: &Path, mut visit: impl FnMut(Value) -> Result<()>) -> Result<()> {
    let mut process = Process::new("/usr/bin/sqlite3")
        .args(["-readonly", "-batch"])
        .process_group(0)
        .arg(path)
        .arg(
            "SELECT json_object('key',key,'command',json(command),'status',status,'result',json(result)) FROM results;",
        )
        .stdout(Stdio::piped())
        .spawn()
        .map_err(unavailable)?;
    let result = (|| {
        for line in BufReader::new(process.stdout.take().unwrap()).lines() {
            let record = serde_json::from_str(&line.map_err(unavailable)?)
                .map_err(|e| Error::invalid_argument(format!("invalid result record: {e}")))?;
            visit(record)?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = process.kill();
    }
    let status = process.wait().map_err(unavailable)?;
    result?;
    if !status.success() {
        return Err(Error::unavailable("SQLite read failed"));
    }
    Ok(())
}

fn lock(path: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path)
        .map_err(unavailable)?;
    // SAFETY: file owns a valid descriptor. flock retains the lock until this owner closes it.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(Error::unavailable(format!(
            "{} is locked by another runner",
            path.display()
        )));
    }
    Ok(file)
}
extern "C" fn interrupt(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

fn measurement_hash(root: &Path) -> Result<String> {
    fn files(path: &Path, names: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(path).map_err(unavailable)? {
            let path = entry.map_err(unavailable)?.path();
            if path.is_dir() {
                files(&path, names)?;
            } else {
                names.push(path);
            }
        }
        Ok(())
    }
    let backend = root.join("crates/inference-backend-metal");
    let mut names = Vec::new();
    files(&backend.join("src"), &mut names)?;
    names.sort();
    names.push(backend.join("benches/gqa/matrix.rs"));
    names.push(backend.join("build.rs"));
    let built = fs::metadata(std::env::current_exe().map_err(unavailable)?)
        .map_err(unavailable)?
        .modified()
        .map_err(unavailable)?;
    let mut source = Vec::new();
    for path in names {
        if fs::metadata(&path)
            .map_err(unavailable)?
            .modified()
            .map_err(unavailable)?
            > built
        {
            return Err(Error::invalid_argument(
                "measurement source is newer than this binary; rebuild gqa-sdpa-bench",
            ));
        }
        source.extend_from_slice(path.strip_prefix(root).unwrap().to_str().unwrap().as_bytes());
        let mut bytes = Vec::new();
        File::open(&path)
            .map_err(unavailable)?
            .read_to_end(&mut bytes)
            .map_err(unavailable)?;
        if path.ends_with("benches/gqa/matrix.rs") {
            bytes = String::from_utf8(bytes)
                .map_err(unavailable)?
                .replace("pub fn main()", "fn main()")
                .into_bytes();
        }
        source.extend_from_slice(&bytes);
    }
    let mut child = Process::new("/usr/bin/shasum")
        .args(["-a", "256"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(unavailable)?;
    let write_result = child.stdin.take().unwrap().write_all(&source);
    let out = child.wait_with_output().map_err(unavailable)?;
    write_result.map_err(unavailable)?;
    if !out.status.success() {
        return Err(Error::unavailable("measurement source hash failed"));
    }
    Ok(String::from_utf8(out.stdout)
        .map_err(unavailable)?
        .split_whitespace()
        .next()
        .unwrap()
        .to_owned())
}

fn hardware_device(hardware: &str) -> Result<Value> {
    let graphics = hardware
        .split("Graphics/Displays:")
        .nth(1)
        .ok_or_else(|| Error::invalid_argument("GPU hardware description is missing"))?;
    let field = |text: &str, name: &str| {
        text.lines()
            .find_map(|line| line.trim().strip_prefix(name).map(str::trim).map(str::to_owned))
            .ok_or_else(|| Error::invalid_argument(format!("hardware field {name} is missing")))
    };
    Ok(
        json!({"chip":field(graphics,"Chipset Model:")?,"gpu_cores":field(graphics,"Total Number of Cores:")?,"memory":field(hardware,"Memory:")?}),
    )
}

fn identity(root: &Path, p: &Plan) -> Result<Value> {
    let hardware = output(root, "system_profiler", &["SPHardwareDataType", "SPDisplaysDataType"])?;
    let environment: BTreeMap<_, _> = [
        "RUSTFLAGS",
        "DYLD_INSERT_LIBRARIES",
        "INFERENCE_BACKEND_METAL_MLX_SOURCE_DIR",
        "INFERENCE_BACKEND_METAL_MLX_PREFIX",
    ]
    .into_iter()
    .filter_map(|name| std::env::var(name).ok().map(|v| (name, v)))
    .collect();
    Ok(
        json!({"measurement_sha256":measurement_hash(root)?, "device":hardware_device(&hardware)?,
        "os":output(root,"sw_vers",&[])?,"environment":environment,"inner":p.inner,"repeats":p.repeats}),
    )
}
fn check_identity(out: &Path, current: &Value) -> Result<()> {
    let path = out.join("identity.json");
    if !path.exists() {
        return write_json(&path, current);
    }
    let old: Value = read_json(&path)?;
    if old.get("measurement_sha256").is_some() {
        if old != *current {
            return Err(Error::invalid_argument(
                "measurement source, hardware, OS, or timing configuration changed; use a new --out directory",
            ));
        }
        return Ok(());
    }
    // Legacy migration requires the source proof captured before the controller change.
    let proof: Value = read_json(&out.join("measurement-identity.json"))?;
    let hardware = old["hardware"]
        .as_str()
        .ok_or_else(|| Error::invalid_argument("legacy hardware identity is missing"))?;
    let device = hardware_device(hardware)?;
    let mut environment = old["environment"].clone();
    if let Some(e) = environment.as_object_mut() {
        e.remove("CARGO_TARGET_DIR");
    }
    if proof["verified_source_sha256"] != old["source_sha256"]
        || proof["measurement_sha256"] != current["measurement_sha256"]
        || device != current["device"]
        || old["os"] != current["os"]
        || environment != current["environment"]
        || old["inner"] != current["inner"]
        || old["repeats"] != current["repeats"]
    {
        return Err(Error::invalid_argument(
            "legacy measurement identity does not match; use a new --out directory",
        ));
    }
    write_json(&out.join("controller-identity.json"), current)
}

// A frozen export is an external input. Prove its algorithm, device, and coefficient domains once.
fn read_frozen_policy(path: &Path, device: &str) -> Result<(Policy, Value)> {
    let data: Value = read_json(path)?;
    let policy = decode_frozen_policy(&data, device)?;
    Ok((
        policy,
        json!({"path":fs::canonicalize(path).map_err(unavailable)?.display().to_string(),"document":data}),
    ))
}

fn validate_policy_output(source: &Value, out: &Path) -> Result<()> {
    let input = source["path"]
        .as_str()
        .ok_or_else(|| Error::invalid_argument("frozen policy source path is missing"))?;
    let out = fs::canonicalize(out).map_err(unavailable)?;
    for name in ["policy.json", "policy.rs", "frozen-policy.json", "frozen-policy.rs"] {
        let path = out.join(name);
        let output = match fs::canonicalize(&path) {
            Ok(output) => output,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => path,
            Err(error) => return Err(unavailable(error)),
        };
        if Some(input) == output.to_str() {
            return Err(Error::invalid_argument(
                "the frozen input must differ from the generated policy exports; copy the input before using this \
                 output directory",
            ));
        }
    }
    Ok(())
}

fn decode_frozen_policy(data: &Value, device: &str) -> Result<Policy> {
    let operator = json!({"io_dtype":"bfloat16","cache_dtype":"fp8_e4m3","visibility":"uniform per-request causal history; no sliding window"});
    let schema = data["schema"].as_u64();
    let decode_matches = schema == Some(4)
        && data["decode_criteria"]["criterion"] == "shared-partitioned-grid-v2"
        && data["policy"]["decode"].is_object();
    if !decode_matches
        || data["device"] != device
        || data["operator"] != operator
        || data["strategies"] != json!(STRATEGIES)
    {
        return Err(Error::invalid_argument(
            "frozen policy does not match the current criterion schema, operator, or calibration device",
        ));
    }
    let policy: Policy = serde_json::from_value(data["policy"].clone())
        .map_err(|e| Error::invalid_argument(format!("invalid frozen policy coefficients: {e}")))?;
    let valid_decode = match policy.decode {
        DecodePolicy::Fixed => true,
        DecodePolicy::Adaptive {
            min_grid_groups,
            target_groups,
        } => {
            let cores = calibration_compute_units(device)?;
            [2, 4, 8].map(|n| n * cores).contains(&min_grid_groups)
                && [16, 32].map(|n| n * cores).contains(&target_groups)
        },
    };
    if !valid_decode
        || ![16, 32, 64].contains(&policy.short_q_token_heads)
        || ![8192, 16384, 32768].contains(&policy.q_token_head_values)
        || ![8, 16, 32].contains(&policy.loader_tokens)
        || ![0, 16, 32, 64, 128].contains(&policy.max_head_base_groups)
        || ![0, 16, 32, 64].contains(&policy.max_floor_base_groups)
        || ![0, 256, 512, 1024, 2048].contains(&policy.target_split_groups)
        || (policy.max_floor_base_groups == 0) != (policy.target_split_groups == 0)
        || policy.upper_slice_shift > 2
    {
        return Err(Error::invalid_argument(
            "frozen policy coefficients exceed the supported criterion domain",
        ));
    }
    Ok(policy)
}

fn calibration_compute_units(device: &str) -> Result<u32> {
    let cores: u32 = device
        .rsplit('/')
        .next()
        .unwrap()
        .parse()
        .map_err(|e| Error::invalid_argument(format!("invalid calibration compute-unit count: {e}")))?;
    if cores == 0 || cores > u32::MAX / 32 {
        return Err(Error::invalid_argument(
            "calibration compute-unit count exceeds the criterion domain",
        ));
    }
    Ok(cores)
}

fn calibration_device(identity: &Value) -> Result<String> {
    let device = if let Some(d) = identity.get("device") {
        d.clone()
    } else {
        hardware_device(
            identity["hardware"]
                .as_str()
                .ok_or_else(|| Error::invalid_argument("calibration hardware identity is missing"))?,
        )?
    };
    let chip = device["chip"]
        .as_str()
        .ok_or_else(|| Error::invalid_argument("calibration chip is missing"))?;
    let cores = device["gpu_cores"]
        .as_str()
        .ok_or_else(|| Error::invalid_argument("calibration GPU core count is missing"))?;
    Ok(format!("{chip}/{cores}"))
}

struct Worker {
    process: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
}
impl Worker {
    fn new(out: &Path) -> Result<Self> {
        let stderr = OpenOptions::new()
            .create(true)
            .append(true)
            .open(out.join("runner.stderr.log"))
            .map_err(unavailable)?;
        let mut process = Process::new(std::env::current_exe().map_err(unavailable)?)
            .arg("--worker")
            .process_group(0)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr)
            .spawn()
            .map_err(unavailable)?;
        Ok(Self {
            input: process.stdin.take(),
            output: BufReader::new(process.stdout.take().unwrap()),
            process,
        })
    }
    fn measure(&mut self, command: &Command) -> Result<Value> {
        let input = self.input.as_mut().unwrap();
        writeln!(input, "{}", encoded(command)).map_err(unavailable)?;
        input.flush().map_err(unavailable)?;
        let mut line = String::new();
        self.output.read_line(&mut line).map_err(unavailable)?;
        if line.is_empty() {
            return Err(Error::unavailable(
                "measurement worker exited; inspect runner.stderr.log",
            ));
        }
        serde_json::from_str(&line).map_err(unavailable)
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        drop(self.input.take());
        let _ = self.process.wait();
    }
}

pub fn main() -> Result<()> {
    let a = Args::parse();
    if !a.hours.is_finite()
        || a.hours < 0.0
        || !a.max_regret.is_finite()
        || !(0.0..=1.0).contains(&a.max_regret)
        || !a.max_noise.is_finite()
        || !(0.0..=1.0).contains(&a.max_noise)
    {
        return Err(Error::invalid_argument(
            "hours must be finite and nonnegative; max-regret and max-noise must be in [0,1]",
        ));
    }
    if cfg!(debug_assertions) && !matches!(a.mode, Mode::Plan | Mode::Report) {
        return Err(Error::invalid_argument(
            "GPU measurements require a release build of gqa-sdpa-bench",
        ));
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    fs::create_dir_all(&a.out).map_err(unavailable)?;
    let out = fs::canonicalize(&a.out).map_err(unavailable)?;
    let _dataset_lock = lock(&out.join("runner.lock"))?;
    if a.mode == Mode::Report {
        let plan: Value = read_json(&out.join("plan.json"))?;
        let mut p: Plan =
            serde_json::from_value(plan.clone()).map_err(|e| Error::invalid_argument(format!("invalid plan: {e}")))?;
        if plan.get("mode").is_none() && out.join("status.json").exists() {
            let status: Value = read_json(&out.join("status.json"))?;
            p.mode = serde_json::from_value(status["mode"].clone())
                .map_err(|e| Error::invalid_argument(format!("invalid saved mode: {e}")))?;
        }
        let old: Value = read_json(&out.join("identity.json"))?;
        p.inner = old["inner"]
            .as_u64()
            .ok_or_else(|| Error::invalid_argument("identity.inner is missing"))?
            .try_into()
            .map_err(unavailable)?;
        p.repeats = old["repeats"]
            .as_u64()
            .ok_or_else(|| Error::invalid_argument("identity.repeats is missing"))?
            .try_into()
            .map_err(unavailable)?;
        p.validate()?;
        return report::generate(&out, &p, a.max_noise, a.max_regret, a.policy.as_deref());
    }
    let mut p = Plan::new(&a);
    let current = identity(root, &p)?;
    if matches!(a.mode, Mode::Plan | Mode::Policy) {
        p.frozen_policy = a
            .policy
            .as_deref()
            .map(|path| read_frozen_policy(path, &calibration_device(&current)?))
            .transpose()?;
    }
    p.validate()?;
    if let Some((_, source)) = &p.frozen_policy {
        validate_policy_output(source, &out)?;
    }
    check_identity(&out, &current)?;
    let mut plan = serde_json::to_value(&p).unwrap();
    plan["candidate_axes"] = json!({"layouts":["single","head","token","joint"],"tiles":[1,2,4,8,16,32],"head_rules":["1","ceil(ratio/2)","ratio"],"stages":[8,16,32,64,128],"slices":[64,128,256,512,1024,2048,4096,8192,"full"],"outputs":["split_map_reduce","unsplit_direct"]});
    plan["candidate_selection"] = json!(if p.frozen_policy.is_some() {
        "Frozen route, full/same-slice SlicedKv, QHeadsPerToken/QTokensAndHeads/QTokensPerHead controls, adjacent \
         slices for each comparison geometry, and unsplit Map/Reduce control. Independent of measured winners."
    } else {
        "Full candidate product, filtered by the selected screen or smoke mode."
    });
    plan["metric"] = json!(
        "amortized replay wall ms including submit/wait; fixed full-slice SlicedKv reference; forced geometry, not \
         production selector"
    );
    plan["constraints"] = json!([
        "direct requires full slice",
        "one request and KV head per tile",
        "threads<=1024",
        "D128/P32 needs implementation",
        "missing record remains unmeasured"
    ]);
    write_json(&out.join("plan.json"), &plan)?;
    let db_path = out.join("results.sqlite3");
    let mut recorded = HashMap::new();
    if db_path.exists() {
        records(&db_path, |r| {
            let key = r["key"]
                .as_str()
                .ok_or_else(|| Error::invalid_argument("case key is missing"))?;
            let status = r["status"]
                .as_str()
                .ok_or_else(|| Error::invalid_argument("case status is missing"))?;
            recorded.insert(key.to_owned(), status.to_owned());
            Ok(())
        })?;
    }
    if a.mode == Mode::Plan {
        let workloads = p.workloads();
        let tuples: usize = workloads.iter().map(|&w| p.candidates(w).len()).sum();
        if p.frozen_policy.is_some() {
            let mut coverage = BTreeMap::<String, usize>::new();
            let mut retryable = 0;
            for &w in &workloads {
                for c in p.candidates(w) {
                    let command = Command {
                        workload: w,
                        candidate: c,
                        inner: p.inner,
                        repeats: p.repeats,
                    };
                    let status = recorded
                        .get(&encoded(&command))
                        .map(String::as_str)
                        .unwrap_or("missing");
                    *coverage.entry(status.to_owned()).or_default() += 1;
                    retryable +=
                        usize::from(status == "deferred" && c.scratch_bytes(w) <= p.max_scratch_mib * 1024 * 1024);
                }
            }
            plan["recorded_candidate_coverage"] = json!(coverage);
            plan["retryable_deferred_candidates"] = json!(retryable);
            write_json(&out.join("plan.json"), &plan)?;
            println!(
                "Recorded coverage: {}. Retryable deferred candidates: {retryable}.",
                encoded(&coverage)
            );
        }
        println!(
            "{} workload profiles; {tuples} distinct candidates before resource filtering. No GPU work. Plan: {}",
            workloads.len(),
            out.join("plan.json").display()
        );
        return Ok(());
    }
    let _gpu_lock = lock(&root.join("target/gqa-matrix.lock"))?;
    // SAFETY: this handler only sets an atomic flag. The owner drains and records the current case.
    unsafe {
        libc::signal(libc::SIGINT, interrupt as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, interrupt as *const () as libc::sighandler_t);
    }
    let prov = json!({"identity":current,"commit":output(root,"git",&["rev-parse","HEAD"])? ,"dirty":output(root,"git",&["status","--short"])? ,"power":output(root,"pmset",&["-g","batt"])? ,"rustc":output(root,"rustc",&["-Vv"])? ,"executable_sha256":output(root,"/usr/bin/shasum",&["-a","256",std::env::current_exe().map_err(unavailable)?.to_str().unwrap()])? ,"argv":std::env::args().collect::<Vec<_>>(),"plan":plan});
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out.join("runs.jsonl"))
        .map_err(unavailable)?;
    writeln!(log, "{prov}").map_err(unavailable)?;
    let mut db = Database::open(&db_path)?;
    let start = Instant::now();
    let mut worker = None;
    let mut measured = 0;
    let mut skipped = 0;
    let mut stop = "complete";
    let measurement_result: Result<()> = (|| {
        'workloads: for w in p.workloads() {
            for c in p.candidates(w) {
                let command = Command {
                    workload: w,
                    candidate: c,
                    inner: p.inner,
                    repeats: p.repeats,
                };
                if recorded
                    .get(&encoded(&command))
                    .is_some_and(|status| status != "deferred" || c.scratch_bytes(w) > p.max_scratch_mib * 1024 * 1024)
                {
                    skipped += 1;
                    continue;
                }
                stop = if INTERRUPTED.load(Ordering::Relaxed) {
                    "interrupted"
                } else if a.hours != 0.0 && start.elapsed().as_secs_f64() >= a.hours * 3600.0 {
                    "time_budget"
                } else if a.limit != 0 && measured >= a.limit {
                    "measurement_limit"
                } else {
                    "complete"
                };
                if stop != "complete" {
                    break 'workloads;
                }
                let mut result = if c.threads() > 1024 {
                    json!({"status":"hardware_ineligible","reason":"Q-token/head assignment exceeds 1024 threads"})
                } else if c.scratch_bytes(w) > p.max_scratch_mib * 1024 * 1024 {
                    json!({"status":"deferred","reason":"scratch allocation budget","bytes":c.scratch_bytes(w)})
                } else {
                    if worker.is_none() {
                        worker = Some(Worker::new(&out)?);
                    }
                    let mut r = worker.as_mut().unwrap().measure(&command)?;
                    if r["status"] == "failed"
                        && r["reason"]
                            .as_str()
                            .is_some_and(|s| s.contains("threadblock exceeds the Map pipeline thread limit"))
                    {
                        r["status"] = json!("hardware_ineligible");
                    }
                    measured += 1;
                    println!("{measured}: {w:?} {c:?} => {}", r["status"]);
                    r
                };
                result["physical_threads"] = json!(c.threads());
                result["logical_q_token_heads"] = json!(c.tile as u64 * c.heads as u64);
                result["physical_q_token_heads"] = if c.layout == Layout::SlicedKv {
                    Value::Null
                } else {
                    json!(c.threads() / 32 * 8)
                };
                db.insert(&command, &result)?;
            }
        }
        Ok(())
    })();
    drop(worker);
    drop(db);
    if measurement_result.is_err() {
        stop = "runner_error";
    }
    write_json(
        &out.join("status.json"),
        &json!({"mode":a.mode,"stop_reason":stop,"new_measurements":measured,"skipped_recorded":skipped,"elapsed_seconds":start.elapsed().as_secs_f64(),"error":measurement_result.as_ref().err().map(ToString::to_string)}),
    )?;
    let frozen_path = if p.mode == Mode::Policy {
        None
    } else {
        a.policy.as_deref()
    };
    let report_result = report::generate(&out, &p, a.max_noise, a.max_regret, frozen_path);
    measurement_result?;
    report_result?;
    println!("Stopped: {stop}. New measurements: {measured}; skipped recorded cases: {skipped}.");
    Ok(())
}

#[cfg(test)]
#[path = "controller_test.rs"]
mod tests;
