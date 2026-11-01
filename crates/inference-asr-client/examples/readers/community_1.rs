//! Experimental Community-1 reader. Build in the pinned speakrs checkout with --features cpu.
//! Pipes carry JSON progress/updates. PCM stays in the caller's append-only recording file.
use std::fs::File;
use std::io::BufRead;
use std::io::Write;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use serde_json::Value;
use serde_json::json;
use speakrs::DiarizationResult;
use speakrs::ExecutionMode;
use speakrs::OwnedDiarizationPipeline;

const WINDOW_SAMPLES: u64 = 30 * 16_000;
const UPDATE_SAMPLES: u64 = 2 * 16_000;
const REVISION_SAMPLES: u64 = 10 * 16_000;
const SPEAKER_SIMILARITY: f32 = 0.7;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Clone)]
struct Span {
    start: u64,
    end: u64,
    speaker: usize,
}

struct Speaker {
    anchor: Vec<f32>,
    recent: Vec<f32>,
}

struct Reader {
    file: File,
    pipeline: OwnedDiarizationPipeline,
    offline: bool,
    accepted: u64,
    processed: u64,
    sealed: u64,
    speakers: Vec<Speaker>,
    previous: Vec<Span>,
}

impl Reader {
    fn load() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut recording = None;
        let mut pass = None;
        let mut models = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--recording" => recording = args.next().map(PathBuf::from),
                "--pass" => pass = args.next(),
                "--model-dir" => models = Some(PathBuf::from(args.next().ok_or("--model-dir requires a path")?)),
                _ => return Err(format!("unknown argument {arg}").into()),
            }
        }
        let offline = match pass.as_deref() {
            Some("offline") => true,
            Some("live") => false,
            _ => return Err("--pass must be live or offline".into()),
        };
        let models = match models {
            Some(path) => path,
            None => std::env::current_exe()?.with_file_name("models"),
        };
        if models.as_os_str().is_empty() {
            return Err("--model-dir must not be empty".into());
        }
        Ok(Self {
            file: File::open(recording.ok_or("--recording is required")?)?,
            pipeline: OwnedDiarizationPipeline::from_dir(&models, ExecutionMode::Cpu)
                .map_err(|error| format!("Community-1 checkpoint {}: {error}", models.display()))?,
            offline,
            accepted: 0,
            processed: 0,
            sealed: 0,
            speakers: Vec::new(),
            previous: Vec::new(),
        })
    }

    fn read(&self, range: Range<u64>) -> Result<Vec<f32>> {
        let bytes = usize::try_from(
            (range.end - range.start)
                .checked_mul(2)
                .ok_or("audio byte extent overflow")?,
        )?;
        let mut pcm = vec![0; bytes];
        self.file
            .read_exact_at(&mut pcm, range.start.checked_mul(2).ok_or("audio offset overflow")?)?;
        Ok(pcm
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| f32::from(i16::from_le_bytes(*bytes)) / 32768.0)
            .collect())
    }

    fn process(&mut self, request: Value) -> Result<(Vec<Value>, bool)> {
        let start = request["start"].as_u64().ok_or("start must be a sample index")?;
        let end = request["end"].as_u64().ok_or("end must be a sample index")?;
        let stopped = request["stopped"].as_bool().ok_or("stopped must be a bool")?;
        if start != self.accepted || start > end || end > self.file.metadata()?.len() / 2 {
            return Err("input must append published PCM without holes".into());
        }
        self.accepted = end;
        if !stopped && (self.offline || end - self.processed < UPDATE_SAMPLES) {
            return Ok((Vec::new(), false));
        }
        let mut updates = Vec::new();
        while self.processed < end && (stopped || self.offline || end - self.processed >= UPDATE_SAMPLES) {
            let end = if self.offline {
                end
            } else {
                (self.processed + UPDATE_SAMPLES).min(end)
            };
            let start = if self.offline {
                0
            } else {
                end.saturating_sub(WINDOW_SAMPLES)
            };
            let audio = self.read(start..end)?;
            let result = self.pipeline.run(&audio)?;
            let mut spans = spans(&result, start..end)?;
            if !self.offline {
                let embeddings = embeddings(&result);
                identify(&mut spans, &embeddings, &self.previous, &mut self.speakers, self.sealed);
            }
            // Preserve the prefix outside this inference window until its labels are sealed.
            let mut replacement: Vec<_> = self
                .previous
                .iter()
                .filter_map(|span| {
                    let lo = span.start.max(self.sealed);
                    let hi = span.end.min(start);
                    (lo < hi).then_some(Span {
                        start: lo,
                        end: hi,
                        speaker: span.speaker,
                    })
                })
                .collect();
            replacement.extend(spans.iter().filter(|span| span.end > self.sealed).map(|span| {
                Span {
                    start: span.start.max(self.sealed),
                    end: span.end,
                    speaker: span.speaker,
                }
            }));
            replacement.sort_by_key(|span| (span.start, span.speaker));
            updates.push(json!({
                "type": "replace", "range": { "start": self.sealed, "end": end },
                "spans": replacement.iter().map(|span| json!({
                    "start_sample": span.start, "end_sample": span.end, "speaker": span.speaker,
                })).collect::<Vec<_>>(),
            }));
            self.previous = spans;
            self.processed = end;
            let seal = end.saturating_sub(REVISION_SAMPLES);
            if seal > self.sealed {
                updates.push(json!({ "type": "seal", "range": { "start": self.sealed, "end": seal } }));
                self.sealed = seal;
            }
        }
        let seal = if stopped { end } else { self.sealed };
        if seal > self.sealed {
            updates.push(json!({ "type": "seal", "range": { "start": self.sealed, "end": seal } }));
            self.sealed = seal;
        }
        Ok((updates, stopped))
    }
}

/// Prefer shared audio evidence. Use retained embeddings when a voice returns after leaving the window.
fn identify(
    spans: &mut Vec<Span>,
    embeddings: &[Vec<f32>],
    previous: &[Span],
    speakers: &mut Vec<Speaker>,
    sealed: u64,
) {
    let num_local = embeddings.len();
    let mut needed = vec![false; num_local];
    for span in spans.iter() {
        needed[span.speaker] |= span.end > sealed;
    }
    let mut mapping = vec![None; num_local];
    let mut used = vec![false; speakers.len()];
    let mut overlap = vec![vec![0u64; speakers.len()]; num_local];
    for span in spans.iter() {
        for old in previous {
            overlap[span.speaker][old.speaker] += span.end.min(old.end).saturating_sub(span.start.max(old.start));
        }
    }
    let mut edges: Vec<_> = overlap
        .iter()
        .enumerate()
        .flat_map(|(local, scores)| {
            scores
                .iter()
                .enumerate()
                .map(move |(global, &samples)| (samples, local, global))
        })
        .collect();
    edges.sort_by_key(|&(samples, local, global)| (std::cmp::Reverse(samples), local, global));
    for (samples, local, global) in edges {
        if samples != 0 && mapping[local].is_none() && !used[global] {
            mapping[local] = Some(global);
            used[global] = true;
        }
    }
    let mut edges: Vec<_> = embeddings
        .iter()
        .enumerate()
        .flat_map(|(local, embedding)| {
            speakers.iter().enumerate().map(move |(global, speaker)| {
                (
                    cosine(embedding, &speaker.anchor).max(cosine(embedding, &speaker.recent)),
                    local,
                    global,
                )
            })
        })
        .collect();
    edges.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    for (score, local, global) in edges {
        if score >= SPEAKER_SIMILARITY && mapping[local].is_none() && !used[global] {
            mapping[local] = Some(global);
            used[global] = true;
        }
    }
    spans.retain_mut(|span| {
        let local = span.speaker;
        if mapping[local].is_none() && !needed[local] {
            return false;
        }
        span.speaker = *mapping[local].get_or_insert_with(|| {
            let global = speakers.len();
            speakers.push(Speaker {
                anchor: embeddings[local].clone(),
                recent: embeddings[local].clone(),
            });
            global
        });
        true
    });
    for (local, global) in mapping.into_iter().enumerate() {
        if let Some(global) = global
            && needed[local]
            && !embeddings[local].is_empty()
        {
            let speaker = &mut speakers[global];
            if speaker.anchor.is_empty() {
                speaker.anchor.clone_from(&embeddings[local]);
            }
            speaker.recent.clone_from(&embeddings[local]);
        }
    }
}

fn spans(result: &DiarizationResult, range: Range<u64>) -> Result<Vec<Span>> {
    let mut spans = Vec::new();
    for segment in &result.segments {
        let speaker = segment
            .speaker
            .strip_prefix("SPEAKER_")
            .ok_or("unexpected Community-1 speaker label")?
            .parse()?;
        let sample = |seconds: f64| {
            range.start
                + (seconds * 16_000.0)
                    .round()
                    .clamp(0.0, (range.end - range.start) as f64) as u64
        };
        let (start, end) = (sample(segment.start), sample(segment.end));
        if start < end {
            spans.push(Span { start, end, speaker });
        }
    }
    Ok(spans)
}

fn embeddings(result: &DiarizationResult) -> Vec<Vec<f32>> {
    let (windows, local_speakers, width) = result.embeddings.dim();
    let mut sums = vec![vec![0.0; width]; result.discrete_diarization.ncols()];
    for window in 0..windows {
        for local in 0..local_speakers {
            if result.hard_clusters.is_empty() {
                continue;
            }
            let cluster = result.hard_clusters[[window, local]];
            if cluster < 0 {
                continue;
            }
            if (0..width).any(|i| !result.embeddings[[window, local, i]].is_finite()) {
                continue;
            }
            for (i, sum) in sums[cluster as usize].iter_mut().enumerate() {
                *sum += result.embeddings[[window, local, i]];
            }
        }
    }
    for sum in &mut sums {
        if !normalize(sum) {
            sum.clear();
        }
    }
    sums
}

fn normalize(values: &mut [f32]) -> bool {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        return false;
    }
    for value in values {
        *value /= norm;
    }
    true
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return -1.0;
    }
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn main() -> Result<()> {
    let mut output = std::io::stdout().lock();
    let mut reader = match Reader::load() {
        Ok(reader) => reader,
        Err(error) => {
            writeln!(output, "{}", json!({ "error": error.to_string() }))?;
            output.flush()?;
            return Err(error);
        },
    };
    writeln!(output, "{}", json!({ "protocol": 1 }))?;
    output.flush()?;
    for line in std::io::stdin().lock().lines() {
        let (updates, stopped) = reader.process(serde_json::from_str(&line?)?)?;
        serde_json::to_writer(&mut output, &updates)?;
        writeln!(output)?;
        output.flush()?;
        if stopped {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "community_1_test.rs"]
mod tests;
