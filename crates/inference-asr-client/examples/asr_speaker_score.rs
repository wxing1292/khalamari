use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

use clap::Parser;
use inference_error::Error;
use inference_error::Result;
use serde_json::Value;
use serde_json::json;

#[derive(Parser)]
#[command(about = "Score one recording on a 10-ms grid, with overlap and no collar")]
struct Args {
    #[arg(long)]
    reference: PathBuf,
    #[arg(long, help = "Single-pass ASR JSONL or reference-system RTTM")]
    hypothesis: PathBuf,
    #[arg(
        long,
        default_value_t = 0.0,
        help = "Excerpt start in the reference recording, in seconds"
    )]
    start: f64,
    #[arg(long)]
    duration: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.start.is_finite() || args.start < 0.0 || !args.duration.is_finite() || args.duration <= 0.0 {
        return Err(Error::invalid_argument(
            "expected a nonnegative start and a positive finite duration",
        ));
    }
    let reference = intervals(&args.reference)?;
    let hypothesis = intervals(&args.hypothesis)?;
    let mut ids = [BTreeMap::new(), BTreeMap::new()];
    let mut edges = Vec::new();
    for (side, offset, intervals) in [(0, args.start, reference), (1, 0.0, hypothesis)] {
        for (speaker, start, end) in intervals {
            if !start.is_finite() || !end.is_finite() || start > end {
                return Err(Error::invalid_argument("invalid speaker interval"));
            }
            let frame = |time: f64| (((time - offset).clamp(0.0, args.duration) * 100.0) - 0.5).ceil() as u64;
            let (a, b) = (frame(start), frame(end));
            if a == b {
                continue;
            }
            let next = ids[side].len();
            let index = *ids[side].entry(speaker).or_insert(next);
            edges.push((a, side, index, 1i32));
            edges.push((b, side, index, -1i32));
        }
    }
    let n = ids[1].len();
    if n > 16 {
        return Err(Error::invalid_argument(
            "this exhaustive reference scorer accepts at most sixteen hypothesized speakers",
        ));
    }
    edges.sort_unstable();
    let mut active = [vec![0i32; ids[0].len()], vec![0i32; n]];
    let mut intersection = vec![vec![0u64; n]; ids[0].len()];
    let (mut speech, mut missed, mut false_alarm, mut shared, mut previous) = (0, 0, 0, 0, 0);
    for (frame, side, id, delta) in edges {
        let elapsed = frame - previous;
        if elapsed != 0 {
            let r = active[0].iter().filter(|v| **v > 0).count() as u64;
            let h = active[1].iter().filter(|v| **v > 0).count() as u64;
            speech += r * elapsed;
            missed += r.saturating_sub(h) * elapsed;
            false_alarm += h.saturating_sub(r) * elapsed;
            shared += r.min(h) * elapsed;
            for (i, _) in active[0].iter().enumerate().filter(|(_, v)| **v > 0) {
                for (j, _) in active[1].iter().enumerate().filter(|(_, v)| **v > 0) {
                    intersection[i][j] += elapsed;
                }
            }
        }
        active[side][id] += delta;
        previous = frame;
    }
    if speech == 0 {
        return Err(Error::invalid_argument("the reference excerpt has no labeled speech"));
    }
    // A recording-local speaker ID can match at most one reference speaker, across the entire file.
    let mut best = vec![0; 1 << n];
    for row in intersection {
        let mut next = best.clone();
        for (mask, &score) in best.iter().enumerate() {
            for (j, &matched) in row.iter().enumerate().filter(|(j, _)| mask & (1 << j) == 0) {
                let new = mask | (1 << j);
                next[new] = next[new].max(score + matched);
            }
        }
        best = next;
    }
    let confusion = shared - best.into_iter().max().unwrap();
    println!(
        "{}",
        json!({
            "reference_speakers": ids[0].len(), "hypothesis_speakers": n,
            "reference_speaker_seconds": speech as f64 / 100.0,
            "missed_seconds": missed as f64 / 100.0, "false_alarm_seconds": false_alarm as f64 / 100.0,
            "confusion_seconds": confusion as f64 / 100.0,
            "der": (missed + false_alarm + confusion) as f64 / speech as f64,
        })
    );
    Ok(())
}

fn intervals(path: &Path) -> Result<Vec<(String, f64, f64)>> {
    let file = File::open(path).map_err(|e| Error::unavailable(format!("{}: {e}", path.display())))?;
    let mut result = Vec::new();
    let mut transcripts = BTreeMap::<u64, (u64, Vec<(String, f64, f64)>)>::new();
    let mut pass = None;
    let mut recording = None;
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| Error::unavailable(e.to_string()))?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        if path.extension().is_some_and(|ext| ext == "rttm") {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.first() != Some(&"SPEAKER") || fields.len() < 8 {
                return Err(Error::invalid_argument("invalid RTTM row"));
            }
            if fields[2] != "1" || recording.get_or_insert_with(|| fields[1].to_owned()) != fields[1] {
                return Err(Error::invalid_argument("score one mono RTTM recording at a time"));
            }
            let seconds = |i: usize| {
                fields[i]
                    .parse::<f64>()
                    .map_err(|e| Error::invalid_argument(e.to_string()))
            };
            let start = seconds(3)?;
            result.push((fields[7].to_owned(), start, start + seconds(4)?));
        } else {
            let event: Value = serde_json::from_str(&line).map_err(|e| Error::invalid_argument(e.to_string()))?;
            let current = match event.get("type").and_then(Value::as_str) {
                Some("partial" | "correction") => continue,
                Some("final" | "speakers") => {
                    event["pass"]
                        .as_str()
                        .ok_or_else(|| Error::invalid_argument("missing ASR pass"))?
                },
                None if event.get("spans").is_some() => "collected",
                _ => return Err(Error::invalid_argument("expected a transcript or ASR event")),
            };
            if pass.get_or_insert_with(|| current.to_owned()) != current {
                return Err(Error::invalid_argument("score one pass at a time"));
            }
            let sample_rate = event["sample_rate"]
                .as_u64()
                .filter(|rate| *rate > 0)
                .ok_or_else(|| Error::invalid_argument("missing sample rate"))?;
            let mut activity_rows = Vec::new();
            for span in event["spans"]
                .as_array()
                .ok_or_else(|| Error::invalid_argument("missing ASR spans"))?
            {
                for activity in span["speakers"]
                    .as_array()
                    .ok_or_else(|| Error::invalid_argument("missing speaker activity"))?
                {
                    let sample = |key: &str| {
                        activity[key]
                            .as_u64()
                            .ok_or_else(|| Error::invalid_argument(format!("missing {key}")))
                    };
                    let start = sample("start_sample")?.max(
                        span["start_sample"]
                            .as_u64()
                            .ok_or_else(|| Error::invalid_argument("missing span start"))?,
                    );
                    let end = sample("end_sample")?.min(
                        span["end_sample"]
                            .as_u64()
                            .ok_or_else(|| Error::invalid_argument("missing span end"))?,
                    );
                    activity_rows.push((
                        sample("speaker")?.to_string(),
                        start as f64 / sample_rate as f64,
                        end as f64 / sample_rate as f64,
                    ));
                }
            }
            if current == "collected" {
                result.extend(activity_rows);
            } else {
                let segment = event["segment"]
                    .as_u64()
                    .ok_or_else(|| Error::invalid_argument("missing segment"))?;
                let revision = event["revision"]
                    .as_u64()
                    .ok_or_else(|| Error::invalid_argument("missing revision"))?;
                if event["type"] == "speakers"
                    && transcripts
                        .get(&segment)
                        .is_none_or(|(current, _)| *current != revision)
                {
                    return Err(Error::invalid_argument(
                        "speaker update must name an existing final revision",
                    ));
                }
                transcripts.insert(segment, (revision, activity_rows));
            }
        }
    }
    result.extend(transcripts.into_values().flat_map(|(_, activity)| activity));
    Ok(result)
}
