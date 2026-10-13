//! Host controller for the existing real-weight component measurement owners.
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use clap::Parser;
use clap::ValueEnum;
use inference_error::Error;
use inference_error::Result;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;

mod dataset;
#[cfg(test)]
#[path = "matrix_test.rs"]
mod tests;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum Component {
    Gqa,
    Gdn,
    DenseMlp,
    Moe,
}

impl Component {
    fn worker(self) -> &'static str {
        match self {
            Self::Gqa => "qwen35_gqa",
            Self::Gdn => "qwen35_gdn",
            Self::DenseMlp => "qwen35_dense_mlp",
            Self::Moe => "qwen35_moe",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Mode {
    Plan,
    Run,
    Report,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Kind {
    Append,
    FullPrefill,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct Case {
    kind: Kind,
    requests: u32,
    context: u32,
    queries: u32,
}

impl Case {
    fn rows(self) -> u32 {
        self.requests * self.queries
    }

    fn measurement(self, component: Component, gdn_replay: bool) -> Self {
        let mut case = Self {
            kind: Kind::Append,
            ..self
        };
        match component {
            Component::Gqa => {},
            Component::Gdn => {
                case.context = self.context.min(1);
                if gdn_replay {
                    case.kind = self.kind;
                }
            },
            Component::DenseMlp | Component::Moe => {
                case.requests = 1;
                case.context = 0;
                case.queries = self.rows();
            },
        }
        case
    }

    fn label(self) -> String {
        format!(
            "{:?}-r{}-ctx{}-q{}",
            self.kind, self.requests, self.context, self.queries
        )
    }
}

#[derive(Debug, Parser)]
#[command(about = "Plan, resume, and report real-weight component latency. Default mode does not use the GPU.")]
struct Args {
    #[arg(long, value_enum)]
    component: Component,
    #[arg(long, value_enum, default_value = "plan")]
    mode: Mode,
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, num_args = 1.., value_delimiter = ',', default_values_t = [1, 2, 4, 8])]
    requests: Vec<u32>,
    #[arg(long, num_args = 0.., value_delimiter = ',', default_values_t = [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536])]
    contexts: Vec<u32>,
    #[arg(long, num_args = 1.., value_delimiter = ',', default_values_t = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128, 256])]
    queries: Vec<u32>,
    #[arg(long, num_args = 0.., value_delimiter = ',', default_values_t = [2048, 4096, 8192, 16384])]
    full: Vec<u32>,
    /// GQA scratch/metadata capacity. Large cases raise it to their active row count.
    #[arg(long, default_value_t = 256)]
    max_tokens: u32,
    #[arg(long, default_value_t = 10)]
    iters: u32,
    #[arg(long, default_value_t = 3)]
    warmup_iters: u32,
    #[arg(long, default_value_t = 3)]
    runs: u32,
    /// Zero is unlimited. Finish the current case before stopping at the deadline.
    #[arg(long, default_value_t = 0.0)]
    hours: f64,
    /// GDN append cases can record replay logs instead of ordinary Prefill snapshots.
    #[arg(long)]
    gdn_replay: bool,
    #[arg(long)]
    out: PathBuf,
}

impl Args {
    fn cases(&self) -> Result<Vec<Case>> {
        if self.requests.is_empty()
            || self.requests.contains(&0)
            || self.queries.is_empty()
            || self.queries.iter().chain(&self.full).any(|&n| n == 0)
            || self.max_tokens == 0
            || self.max_tokens > i32::MAX as u32
            || self.iters == 0
            || self.runs == 0
            || !self.hours.is_finite()
            || self.hours < 0.0
        {
            return Err(Error::invalid_argument(
                "counts must be positive and hours must be finite and nonnegative",
            ));
        }
        if self.gdn_replay && self.component != Component::Gdn {
            return Err(Error::invalid_argument("--gdn-replay is specific to GDN"));
        }
        let mut cases = BTreeSet::new();
        for &requests in &self.requests {
            for (&context, &queries) in self
                .contexts
                .iter()
                .flat_map(|c| self.queries.iter().map(move |q| (c, q)))
            {
                cases.insert(Case {
                    kind: Kind::Append,
                    requests,
                    context,
                    queries,
                });
            }
            for &queries in &self.full {
                cases.insert(Case {
                    kind: Kind::FullPrefill,
                    requests,
                    context: 0,
                    queries,
                });
            }
        }
        for case in &cases {
            if case
                .requests
                .checked_mul(case.queries)
                .is_none_or(|n| n > i32::MAX as u32)
                || case.context.checked_add(case.queries).is_none()
            {
                return Err(Error::invalid_argument("case exceeds the worker row or context domain"));
            }
        }
        if cases.is_empty() {
            return Err(Error::invalid_argument("select append contexts or full-Prefill sizes"));
        }
        Ok(cases.into_iter().collect())
    }

    fn command(&self, case: Case, model: &Path, profile: &str) -> Vec<String> {
        let mut command = vec![
            "--model-dir".into(),
            model.display().to_string(),
            "--iters".into(),
            self.iters.to_string(),
            "--warmup-iters".into(),
            self.warmup_iters.to_string(),
            "--runs".into(),
            self.runs.to_string(),
        ];
        let segments = vec![case.queries.to_string(); case.requests as usize].join(",");
        match self.component {
            Component::Gqa => {
                command.extend([
                    "--gqa-split-kv-variants".into(),
                    "auto".into(),
                    "--gqa-model".into(),
                    profile.into(),
                    "--gqa-tokens-per-req".into(),
                    segments,
                    "--contexts".into(),
                    case.context.to_string(),
                    "--max-tokens".into(),
                    self.max_tokens.max(case.rows()).to_string(),
                ])
            },
            Component::Gdn => {
                command.extend([
                    "--gdn-model".into(),
                    profile.into(),
                    "--tokens-per-req".into(),
                    segments,
                    "--contexts".into(),
                    case.context.to_string(),
                    "--prefill-requests".into(),
                    if self.gdn_replay && case.kind == Kind::Append {
                        0
                    } else {
                        case.requests
                    }
                    .to_string(),
                ])
            },
            Component::DenseMlp => {
                command.extend([
                    "--tokens".into(),
                    case.rows().to_string(),
                    "--cases".into(),
                    "full_auto".into(),
                ])
            },
            Component::Moe => {
                command.extend([
                    "--tokens".into(),
                    case.rows().to_string(),
                    "--impls".into(),
                    "both".into(),
                ])
            },
        }
        command
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct Latency {
    path: String,
    median_ms: f64,
    samples_ms: Vec<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Record {
    case: Case,
    status: String,
    command: Vec<String>,
    latencies: Vec<Latency>,
    log: String,
    swap_before: [u64; 2],
    swap_after: [u64; 2],
    provenance: Value,
}

fn unavailable(error: impl std::fmt::Display) -> Error {
    Error::unavailable(error.to_string())
}

fn parse_latencies(text: &str, component: Component, case: Case, runs: u32) -> Result<Vec<Latency>> {
    let expected = match component {
        Component::Gqa => "gqa",
        Component::Gdn => "gdn",
        Component::DenseMlp => "dense-mlp-real",
        Component::Moe => "moe-real",
    };
    let mut results = Vec::new();
    for line in text.lines().filter(|line| line.starts_with("perf ")) {
        let fields = line
            .split_whitespace()
            .filter_map(|s| s.split_once('='))
            .collect::<std::collections::BTreeMap<_, _>>();
        if fields.get("component") != Some(&expected) {
            continue;
        }
        let field = |name| {
            fields
                .get(name)
                .copied()
                .ok_or_else(|| Error::invalid_argument(format!("worker omitted {name}")))
        };
        let count = field(if component == Component::Moe {
            "num_tokens"
        } else {
            "tokens"
        })?
        .parse::<u32>()
        .map_err(unavailable)?;
        if count != case.rows() {
            return Err(Error::invalid_argument("worker returned a different row count"));
        }
        if matches!(component, Component::Gqa | Component::Gdn) {
            let reqs = field("num_reqs")?.parse::<u32>().map_err(unavailable)?;
            let context = field("ctx")?.parse::<u32>().map_err(unavailable)?;
            if reqs != case.requests || context != case.context {
                return Err(Error::invalid_argument(
                    "worker returned a different request/state shape",
                ));
            }
        }
        let samples = field("samples_us")?;
        let samples_ms = samples
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .ok_or_else(|| Error::invalid_argument("invalid worker sample list"))?
            .split(',')
            .map(|s| s.parse::<f64>().map(|n| n / 1000.0).map_err(unavailable))
            .collect::<Result<Vec<_>>>()?;
        let median_ms = field("median_us")?.parse::<f64>().map_err(unavailable)? / 1000.0;
        if samples_ms.len() != runs as usize
            || samples_ms
                .iter()
                .chain([&median_ms])
                .any(|n| !n.is_finite() || *n < 0.0)
        {
            return Err(Error::invalid_argument("invalid worker latency samples"));
        }
        let path = match component {
            Component::Gqa => field("split_kv_variant")?,
            Component::Gdn => field("path")?,
            Component::DenseMlp | Component::Moe => field("impl")?,
        }
        .to_string();
        if results.iter().any(|r: &Latency| r.path == path) {
            return Err(Error::invalid_argument("duplicate worker path"));
        }
        results.push(Latency {
            path,
            median_ms,
            samples_ms,
        });
    }
    let paths = results.iter().map(|r| r.path.as_str()).collect::<BTreeSet<_>>();
    let expected_paths: BTreeSet<_> = match component {
        Component::Gqa => ["auto"].into_iter().collect(),
        Component::Gdn => ["mixed_replay"].into_iter().collect(),
        Component::DenseMlp => ["full_auto"].into_iter().collect(),
        Component::Moe => ["token_major", "expert_major"].into_iter().collect(),
    };
    if paths != expected_paths {
        return Err(Error::invalid_argument("worker did not return every requested path"));
    }
    Ok(results)
}

fn model_profile(model: &Path, component: Component) -> Result<&'static str> {
    let config: Value =
        serde_json::from_slice(&fs::read(model.join("config.json")).map_err(unavailable)?).map_err(unavailable)?;
    let profile = match config["text_config"]["hidden_size"].as_u64() {
        Some(5120) => "27b",
        Some(2048) => "35b",
        _ => {
            return Err(Error::invalid_argument(
                "the component workers require a 27B or 35B-A3B fixture",
            ));
        },
    };
    if (component == Component::DenseMlp && profile != "27b") || (component == Component::Moe && profile != "35b") {
        return Err(Error::invalid_argument(
            "the checkpoint does not match the dense MLP or MoE fixture",
        ));
    }
    Ok(profile)
}

fn build_worker(root: &Path, component: Component) -> Result<PathBuf> {
    let output = Command::new("cargo")
        .current_dir(root)
        .args([
            "bench",
            "--no-run",
            "--message-format=json",
            "-p",
            "inference-executor-metal",
            "--bench",
            component.worker(),
        ])
        .output()
        .map_err(unavailable)?;
    if !output.status.success() {
        return Err(Error::unavailable(String::from_utf8_lossy(&output.stderr)));
    }
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let value: Value = serde_json::from_str(line).map_err(unavailable)?;
        if value["reason"] == "compiler-artifact"
            && value["target"]["name"] == component.worker()
            && let Some(executable) = value["executable"].as_str()
        {
            return Ok(executable.into());
        }
    }
    Err(Error::unavailable(
        "Cargo did not return the component worker executable",
    ))
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    let cases = args.cases()?;
    fs::create_dir_all(&args.out).map_err(unavailable)?;
    let lock = fs::File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(args.out.join("run.lock"))
        .map_err(unavailable)?;
    lock.try_lock().map_err(unavailable)?;
    dataset::check_component(&args)?;
    dataset::write_json(
        &args.out.join("plan.json"),
        &json!({
            "component": args.component, "cases": cases,
            "measurements": cases.iter().map(|c| c.measurement(args.component, args.gdn_replay)).collect::<BTreeSet<_>>(),
            "metric": "component replay submit/wait wall latency; milliseconds",
            "gdn_state": "zero at ctx=0; one fixed nonzero fixture for every positive ctx",
            "mlp_shape": "requests * queries; independent of ctx",
            "full_prefill": "one component forward from zero history; excludes full-model prefix preparation",
        }),
    )?;
    let mut records = dataset::read_records(&args.out.join("results.jsonl"))?;
    if args.mode != Mode::Run {
        dataset::report(&args, &cases, &records)?;
        println!(
            "{} logical cases, {} unique component measurements. Report: {}",
            cases.len(),
            cases
                .iter()
                .map(|c| c.measurement(args.component, args.gdn_replay))
                .collect::<BTreeSet<_>>()
                .len(),
            args.out.join("report.md").display()
        );
        return Ok(());
    }
    let model = args
        .model_dir
        .as_ref()
        .ok_or_else(|| Error::invalid_argument("--model-dir is required for --mode run"))?
        .canonicalize()
        .map_err(unavailable)?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let profile = model_profile(&model, args.component)?;
    let worker = build_worker(root, args.component)?;
    dataset::check_identity(&args, root, &model, &worker)?;
    let provenance = dataset::provenance(root)?;
    let mut ledger = fs::File::options()
        .create(true)
        .append(true)
        .open(args.out.join("results.jsonl"))
        .map_err(unavailable)?;
    let started = Instant::now();
    for case in cases
        .iter()
        .map(|c| c.measurement(args.component, args.gdn_replay))
        .collect::<BTreeSet<_>>()
    {
        if records.get(&case).is_some_and(|r| r.status == "measured") {
            continue;
        }
        if args.hours > 0.0 && started.elapsed().as_secs_f64() >= args.hours * 3600.0 {
            break;
        }
        let command = args.command(case, &model, profile);
        let record = dataset::measure(&args, &worker, case, command, &provenance)?;
        let ok = record.status == "measured";
        serde_json::to_writer(&mut ledger, &record).map_err(unavailable)?;
        use std::io::Write;
        writeln!(ledger).map_err(unavailable)?;
        ledger.sync_data().map_err(unavailable)?;
        println!("{}: {}", case.label(), record.status);
        records.insert(case, record);
        dataset::report(&args, &cases, &records)?;
        if !ok {
            return Err(Error::unavailable(
                "component worker failed or had swap; see the recorded log. Resume after resolving it.",
            ));
        }
    }
    dataset::report(&args, &cases, &records)
}
