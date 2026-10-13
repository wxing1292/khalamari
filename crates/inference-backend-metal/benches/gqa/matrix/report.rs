//! Quality filtering, coverage reporting, and compression of measured GQA candidates.
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use inference_error::Error;
use inference_error::Result;
use inference_executor_core::replay::ReplayBucketPolicy;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;

use crate::controller::Candidate;
use crate::controller::Command;
use crate::controller::DECODE_REUSE_GRID;
use crate::controller::DECODE_SLICE;
use crate::controller::DecodePolicy;
use crate::controller::Layout;
use crate::controller::Plan;
use crate::controller::Policy;
use crate::controller::STRATEGIES;
use crate::controller::Workload;
use crate::controller::calibration_compute_units;
use crate::controller::calibration_device;
use crate::controller::decode_frozen_policy;
use crate::controller::encoded;
use crate::controller::output;
use crate::controller::proposed_candidate;
use crate::controller::read_frozen_policy;
use crate::controller::read_json;
use crate::controller::records;
use crate::controller::unavailable;
use crate::controller::validate_policy_output;
use crate::controller::write_json;

#[derive(Deserialize)]
struct Sample {
    before_ms: f64,
    candidate_ms: f64,
    after_ms: f64,
    candidate_first_ms: Option<f64>,
    candidate_last_ms: Option<f64>,
}
#[derive(Deserialize)]
struct Measurement {
    samples: Vec<Sample>,
    max_abs_error: f64,
    mean_abs_error: f64,
    cpu_reference_values: usize,
}
#[derive(Clone, Serialize)]
struct Summary {
    workload: Workload,
    candidate: Candidate,
    median_relative_time: f64,
    median_wall_ms: f64,
    max_baseline_drift: f64,
    relative_sample_range: f64,
    stable: bool,
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}
fn summarize(command: &Command, result: Value, max_noise: f64) -> Result<Summary> {
    let m: Measurement =
        serde_json::from_value(result).map_err(|e| Error::invalid_argument(format!("invalid measurement: {e}")))?;
    if m.samples.len() != command.repeats as usize
        || m.cpu_reference_values == 0
        || !m.max_abs_error.is_finite()
        || !(0.0..=0.02).contains(&m.max_abs_error)
        || !m.mean_abs_error.is_finite()
        || !(0.0..=0.005).contains(&m.mean_abs_error)
    {
        return Err(Error::invalid_argument(
            "measurement does not satisfy numerical or repetition contract",
        ));
    }
    let mut ratios = Vec::new();
    let mut wall = Vec::new();
    let mut drift = 0.0_f64;
    for s in m.samples {
        if [s.before_ms, s.candidate_ms, s.after_ms]
            .into_iter()
            .chain(s.candidate_first_ms)
            .chain(s.candidate_last_ms)
            .any(|n| !n.is_finite() || n <= 0.0)
        {
            return Err(Error::invalid_argument("sample times must be finite and positive"));
        }
        ratios.push(s.candidate_ms / ((s.before_ms + s.after_ms) / 2.0));
        wall.push(s.candidate_ms);
        drift = drift.max(s.before_ms.max(s.after_ms) / s.before_ms.min(s.after_ms) - 1.0);
        if let (Some(first), Some(last)) = (s.candidate_first_ms, s.candidate_last_ms) {
            drift = drift.max(first.max(last) / first.min(last) - 1.0);
        }
    }
    let relative = median(&mut ratios);
    let range = (ratios[ratios.len() - 1] - ratios[0]) / relative;
    Ok(Summary {
        workload: command.workload,
        candidate: command.candidate,
        median_relative_time: relative,
        median_wall_ms: median(&mut wall),
        max_baseline_drift: drift,
        relative_sample_range: range,
        stable: drift <= max_noise && range <= max_noise,
    })
}

// None means one full KV slice. Some(n) means a fixed KV slice length.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct Route {
    layout: Layout,
    tile: u32,
    heads: u32,
    stage: u32,
    slice: Option<u32>,
    direct: bool,
}
impl Route {
    fn new(c: Candidate, w: Workload) -> Self {
        Self {
            layout: c.layout,
            tile: c.tile,
            heads: c.heads,
            stage: c.stage,
            slice: (c.slice != w.history + w.queries).then_some(c.slice),
            direct: c.direct,
        }
    }
}
#[derive(Serialize)]
struct Table {
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    tokens_per_page: u32,
    // These are exact measured axis values. The lookup does not interpolate shapes.
    axes: [Vec<u32>; 3],
    routes: Vec<Route>,
    // Zero is an unmeasured or unstable point. Other values are one-based route indices.
    decisions: Vec<u16>,
    unresolved: Vec<[u32; 3]>,
    max_observed_regret: f64,
}

// One measured input point owns its timing reference and eligible candidate set.
struct Cell {
    coordinate: [usize; 3],
    reference_wall_ms: f64,
    choices: Vec<(Route, f64)>,
}

// Reuse a complete candidate across points when each point satisfies the measured loss bound.
// Dictionary indices keep runtime lookup independent of this offline optimization.
fn compress(cells: &[Cell], max_regret: f64, max_extra_ms: f64) -> (Vec<Route>, Vec<u16>, f64, f64) {
    let mut allowed: BTreeMap<Route, Vec<usize>> = BTreeMap::new();
    for (index, cell) in cells.iter().enumerate() {
        let best = cell
            .choices
            .iter()
            .map(|(_, time)| *time)
            .min_by(f64::total_cmp)
            .unwrap();
        for &(route, time) in &cell.choices {
            let regret = time / best - 1.0;
            if regret <= max_regret || regret * cell.reference_wall_ms <= max_extra_ms {
                allowed.entry(route).or_default().push(index);
            }
        }
    }
    let (routes, decisions) = greedy_cover(&allowed, cells.len());
    let losses = selection_losses(cells, &routes, &decisions);
    for &(regret, extra) in &losses {
        assert!(
            regret <= max_regret + 1e-12 || extra <= max_extra_ms + 1e-12,
            "compressed table exceeds its measured bound"
        );
    }
    let max_observed = losses.iter().map(|&(regret, _)| regret).fold(0.0, f64::max);
    let max_estimated_extra_ms = losses.iter().map(|&(_, extra)| extra).fold(0.0, f64::max);
    (routes, decisions, max_observed, max_estimated_extra_ms)
}

// Every covered cell has at least one eligible key. Tie breaks keep the export deterministic.
fn greedy_cover<K: Copy + Ord>(allowed: &BTreeMap<K, Vec<usize>>, num_cells: usize) -> (Vec<K>, Vec<u16>) {
    let mut decisions = vec![0; num_cells];
    let mut keys = Vec::new();
    while decisions.contains(&0) {
        let (&key, covered) = allowed
            .iter()
            .max_by(|(ra, ca), (rb, cb)| {
                let a = ca.iter().filter(|&&i| decisions[i] == 0).count();
                let b = cb.iter().filter(|&&i| decisions[i] == 0).count();
                a.cmp(&b).then_with(|| rb.cmp(ra))
            })
            .unwrap();
        keys.push(key);
        // The fixed candidate product bounds each complete-route or geometry dictionary.
        let key_index = u16::try_from(keys.len()).expect("candidate dictionary must fit u16");
        for &cell in covered {
            if decisions[cell] == 0 {
                decisions[cell] = key_index;
            }
        }
    }
    (keys, decisions)
}

// A geometry can use a different measured KV slice at each point.
// This count isolates geometry reuse. It does not define a partition or runtime policy.
fn geometry_cover_size(cells: &[Cell], max_regret: f64, max_extra_ms: f64) -> usize {
    let mut allowed: BTreeMap<_, Vec<usize>> = BTreeMap::new();
    for (index, cell) in cells.iter().enumerate() {
        let best = cell
            .choices
            .iter()
            .map(|(_, time)| *time)
            .min_by(f64::total_cmp)
            .unwrap();
        let eligible: BTreeSet<_> = cell
            .choices
            .iter()
            .filter_map(|&(r, time)| {
                let regret = time / best - 1.0;
                (regret <= max_regret || regret * cell.reference_wall_ms <= max_extra_ms)
                    .then_some((r.layout, r.tile, r.heads, r.stage, r.direct))
            })
            .collect();
        for geometry in eligible {
            allowed.entry(geometry).or_default().push(index);
        }
    }
    greedy_cover(&allowed, cells.len()).0.len()
}

fn selection_losses(cells: &[Cell], routes: &[Route], decisions: &[u16]) -> Vec<(f64, f64)> {
    cells
        .iter()
        .zip(decisions)
        .map(|(cell, &decision)| {
            let route = routes[decision as usize - 1];
            let selected = cell.choices.iter().find(|(r, _)| *r == route).unwrap().1;
            let best = cell
                .choices
                .iter()
                .map(|(_, time)| *time)
                .min_by(f64::total_cmp)
                .unwrap();
            let regret = selected / best - 1.0;
            (regret, regret * cell.reference_wall_ms)
        })
        .collect()
}

fn compression_frontier(cells: &[Cell], max_regret: f64) -> Vec<Value> {
    let mut alternatives = Vec::new();
    for (relative_bound, absolute_bound_ms) in [
        (0.0, 0.0),
        (0.02, 0.0),
        (max_regret, 0.0),
        (0.10, 0.0),
        (max_regret, 0.005),
        (max_regret, 0.010),
        (max_regret, 0.020),
    ] {
        let (routes, decisions, regret, extra) = compress(cells, relative_bound, absolute_bound_ms);
        let losses = selection_losses(cells, &routes, &decisions);
        let reference_time_buckets: Vec<_> = [0.0, 0.05, 0.2, 1.0, 2.0]
            .into_iter()
            .map(|minimum| {
                let mut points = 0;
                let mut max_regret = 0.0_f64;
                let mut max_extra = 0.0_f64;
                for (cell, &(regret, extra)) in cells.iter().zip(&losses) {
                    if cell.reference_wall_ms >= minimum {
                        points += 1;
                        max_regret = max_regret.max(regret);
                        max_extra = max_extra.max(extra);
                    }
                }
                json!({"min_reference_wall_ms":minimum,"points":points,
                    "max_observed_regret":max_regret,"max_estimated_extra_ms":max_extra})
            })
            .collect();
        alternatives.push(
            json!({"relative_loss_bound":relative_bound,"absolute_loss_bound_ms":absolute_bound_ms,
            "candidate_dictionary_entries":routes.len(),
            "selected_kernel_geometries_with_output_mode":routes.iter().map(|r| (r.layout,r.tile,r.heads,r.stage,r.direct)).collect::<BTreeSet<_>>().len(),
            "selected_kv_slice_parameters":routes.iter().map(|r| r.slice).collect::<BTreeSet<_>>().len(),
            "geometry_cover_entries_with_partition_oracle":geometry_cover_size(cells,relative_bound,absolute_bound_ms),
            "covered_points":cells.len(),
            "max_observed_regret":regret,"max_estimated_extra_ms":extra,"reference_time_buckets":reference_time_buckets}),
        );
    }
    alternatives
}

const RESOURCE_POLICY: Policy = Policy {
    short_q_token_heads: 32,
    q_token_head_values: 16384,
    loader_tokens: 16,
    max_head_base_groups: 0,
    max_floor_base_groups: 0,
    target_split_groups: 0,
    upper_slice_shift: 0,
    decode: DecodePolicy::Fixed,
};

// Fit resource geometry first, then fit base-grid/partition criteria without worsening any cost tier.
fn fit_policy(
    workloads: &BTreeSet<Workload>,
    summaries: &BTreeMap<Workload, Vec<Summary>>,
    num_compute_units: u32,
) -> Option<(Policy, Value)> {
    let geometry_policies: Vec<_> = [8192, 16384, 32768]
        .into_iter()
        .flat_map(|q_token_head_values| {
            [8, 16, 32].map(|loader_tokens| {
                Policy {
                    q_token_head_values,
                    loader_tokens,
                    ..RESOURCE_POLICY
                }
            })
        })
        .collect();
    let (geometry, geometry_evidence, geometry_varies) =
        fit_candidates(workloads, summaries, &geometry_policies, None)?;
    let geometry = if geometry_varies { geometry } else { RESOURCE_POLICY };
    let policies = parallel_policies(geometry);
    let (policy, parallel_evidence, parallel_varies) = fit_candidates(workloads, summaries, &policies, Some(geometry))?;
    let policy = if parallel_varies { policy } else { geometry };
    let decode_policies: Vec<_> = std::iter::once(policy)
        .chain([2, 4, 8].into_iter().flat_map(|minimum| {
            [16, 32].map(move |target| {
                Policy {
                    decode: DecodePolicy::Adaptive {
                        min_grid_groups: minimum * num_compute_units,
                        target_groups: target * num_compute_units,
                    },
                    ..policy
                }
            })
        }))
        .collect();
    let (decode, decode_evidence, decode_varies) =
        fit_candidates(workloads, summaries, &decode_policies, Some(policy))?;
    if !geometry_varies && !parallel_varies && !decode_varies {
        return None;
    }
    Some((
        if decode_varies { decode } else { policy },
        json!({"geometry":geometry_evidence,"parallelism":parallel_evidence,"decode":decode_evidence}),
    ))
}

fn parallel_policies(geometry: Policy) -> Vec<Policy> {
    let mut policies = Vec::new();
    for max_head_base_groups in [0, 16, 32, 64, 128] {
        let floors = [(0, 0)].into_iter().chain(
            [16, 32, 64]
                .into_iter()
                .flat_map(|groups| [256, 512, 1024, 2048].map(move |target| (groups, target))),
        );
        for (max_floor_base_groups, target_split_groups) in floors {
            for upper_slice_shift in [0, 1, 2] {
                policies.push(Policy {
                    max_head_base_groups,
                    max_floor_base_groups,
                    target_split_groups,
                    upper_slice_shift,
                    ..geometry
                });
            }
        }
    }
    policies
}

fn policy_score(
    training: &[Workload],
    summaries: &BTreeMap<Workload, Vec<Summary>>,
    policy: Policy,
) -> [(f64, f64); 3] {
    let mut score = [(0.0_f64, 0.0_f64); 3];
    for &w in training {
        let records = &summaries[&w];
        let reference = records.iter().find(|r| r.stable).unwrap();
        let candidate = proposed_candidate(w, policy).unwrap();
        let selected = records.iter().find(|r| r.candidate == candidate && r.stable).unwrap();
        let regret = selected.median_relative_time / reference.median_relative_time - 1.0;
        for (i, minimum) in [2.0, 1.0, 0.2].into_iter().enumerate() {
            if reference.median_wall_ms < minimum {
                continue;
            }
            score[i].0 = score[i].0.max(regret);
            score[i].1 = score[i].1.max(regret * reference.median_wall_ms);
        }
    }
    score
}

fn fit_candidates(
    workloads: &BTreeSet<Workload>,
    summaries: &BTreeMap<Workload, Vec<Summary>>,
    policies: &[Policy],
    baseline: Option<Policy>,
) -> Option<(Policy, Value, bool)> {
    let references: Vec<_> = workloads
        .iter()
        .copied()
        .filter(|w| {
            summaries
                .get(w)
                .and_then(|records| records.iter().find(|r| r.stable))
                .is_some_and(|r| r.median_wall_ms >= 0.2)
        })
        .collect();
    // Every policy uses the same stable measured inputs. Missingness is not a timing signal.
    let training: Vec<_> = references
        .iter()
        .copied()
        .filter(|w| {
            policies.iter().all(|&p| {
                proposed_candidate(*w, p).is_some_and(|c| summaries[w].iter().any(|r| r.candidate == c && r.stable))
            })
        })
        .collect();
    if training.is_empty() {
        return None;
    }
    let varied_training_points = training
        .iter()
        .filter(|&&w| {
            let first = proposed_candidate(w, policies[0]);
            policies[1..].iter().any(|&p| proposed_candidate(w, p) != first)
        })
        .count();
    let bounds = baseline.map(|p| policy_score(&training, summaries, p));
    let mut best = None;
    let mut best_score = [(0.0_f64, 0.0_f64); 3];
    let mut frontier = Vec::new();
    for &policy in policies {
        let score = policy_score(&training, summaries, policy);
        let allowed = bounds.is_none_or(|limits| {
            score
                .iter()
                .zip(limits)
                .all(|(s, l)| s.0 <= l.0 + 1e-12 && s.1 <= l.1 + 1e-12)
        });
        frontier.push(json!({"policy":policy,"score":score,"within_baseline_bounds":allowed}));
        if !allowed {
            continue;
        }
        let improves = best.is_none()
            || score
                .iter()
                .zip(&best_score)
                .map(|(a, b)| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)))
                .find(|order| !order.is_eq())
                .is_some_and(|order| order.is_lt());
        if improves {
            best = Some(policy);
            best_score = score;
        }
    }
    Some((
        best.unwrap(),
        json!({"parameter_sets":policies.len(),"reference_points":references.len(),"training_points":training.len(),"varied_training_points":varied_training_points,"excluded_points":references.len()-training.len(),"quality_contract":"Every training input has a stable measured complete selected route for every parameter set.","score_cost_tiers_ms":[2,1,0.2],"score_fields":["max_observed_regret","max_estimated_extra_ms"],"baseline_bounds":bounds,"frontier":frontier}),
        varied_training_points > 0,
    ))
}

fn evaluate_policy(
    workloads: &BTreeSet<Workload>,
    summaries: &BTreeMap<Workload, Vec<Summary>>,
    recorded: &HashMap<(Workload, Candidate), String>,
    policy: Policy,
) -> Value {
    let mut points = Vec::new();
    let mut coverage: BTreeMap<&str, usize> = BTreeMap::new();
    let mut quality: BTreeMap<&str, usize> = BTreeMap::new();
    let mut losses = Vec::new();
    for &w in workloads {
        let candidate = proposed_candidate(w, policy);
        // generate() sorts summaries by normalized time before policy evaluation.
        let records = summaries.get(&w).map(Vec::as_slice).unwrap_or_default();
        let reference = records.iter().find(|r| r.stable);
        let selected = candidate.and_then(|c| records.iter().find(|r| r.candidate == c));
        let status = candidate.map_or("unsupported_geometry", |c| {
            recorded.get(&(w, c)).map(String::as_str).unwrap_or("missing")
        });
        let timing_quality = selected.map_or("unknown", |r| if r.stable { "stable" } else { "unstable" });
        *coverage.entry(status).or_default() += 1;
        *quality.entry(timing_quality).or_default() += 1;
        let regret = reference
            .zip(selected.filter(|r| r.stable))
            .map(|(r, s)| s.median_relative_time / r.median_relative_time - 1.0);
        if let Some(reference) = reference {
            losses.push((reference.median_wall_ms, regret));
        }
        points.push(json!({"workload":w,"candidate":candidate,"recorded_status":status,
            "timing_quality":timing_quality,"selected_timing_summary":selected,"reference_timing_summary":reference,
            "reference_wall_ms":reference.map(|r| r.median_wall_ms),
            "observed_regret":regret,"estimated_extra_ms":reference.zip(regret).map(|(r,v)| r.median_wall_ms*v)}));
    }
    let buckets: Vec<_> = [0.0, 0.05, 0.2, 1.0, 2.0]
        .into_iter()
        .map(|minimum| {
            let reference_points: Vec<_> = losses.iter().filter(|(wall, _)| *wall >= minimum).collect();
            let known: Vec<_> = reference_points
                .iter()
                .filter_map(|&&(wall, regret)| regret.map(|r| (r, wall * r)))
                .collect();
            json!({"min_reference_wall_ms":minimum,"reference_points":reference_points.len(),
            "verified_points":known.len(),"unverified_points":reference_points.len()-known.len(),
            "max_observed_regret":known.iter().map(|(r,_)| *r).max_by(f64::total_cmp),
            "max_estimated_extra_ms":known.iter().map(|(_,e)| *e).max_by(f64::total_cmp),
            "points_over_5_percent":known.iter().filter(|(r,_)| *r > 0.05).count(),
            "points_over_10_percent":known.iter().filter(|(r,_)| *r > 0.10).count()})
        })
        .collect();
    json!({"policy":policy,"geometry_families":4,
        "selected_coverage":coverage,"timing_quality":quality,"reference_unknown_points":workloads.len()-losses.len(),
        "reference_time_buckets":buckets,"points":points})
}

fn flat_index(index: [usize; 3], axes: &[Vec<u32>; 3]) -> usize {
    (index[0] * axes[1].len() + index[1]) * axes[2].len() + index[2]
}

fn candidate_scan_complete(coverage: &BTreeMap<String, usize>) -> bool {
    ["missing", "deferred", "failed"]
        .into_iter()
        .all(|status| coverage.get(status).copied().unwrap_or(0) == 0)
}

pub fn generate(out: &Path, p: &Plan, max_noise: f64, max_regret: f64, frozen_path: Option<&Path>) -> Result<()> {
    let workloads: BTreeSet<_> = p.workloads().into_iter().collect();
    let mut recorded = HashMap::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut summaries: BTreeMap<Workload, Vec<Summary>> = BTreeMap::new();
    let mut max_error = 0.0_f64;
    let mut mean_error = 0.0_f64;
    let db_path = out.join("results.sqlite3");
    if db_path.exists() {
        records(&db_path, |r| {
            let command: Command = serde_json::from_value(r["command"].clone())
                .map_err(|e| Error::invalid_argument(format!("invalid command record: {e}")))?;
            command.workload.validate()?;
            command.candidate.validate(command.workload)?;
            if r["key"] != encoded(&command) || r["status"] != r["result"]["status"] {
                return Err(Error::invalid_argument("result record key or status is inconsistent"));
            }
            let status = r["status"]
                .as_str()
                .ok_or_else(|| Error::invalid_argument("result status is missing"))?;
            if !["measured", "failed", "hardware_ineligible", "deferred"].contains(&status) {
                return Err(Error::invalid_argument("unknown result status"));
            }
            *counts.entry(status.to_owned()).or_default() += 1;
            let w = command.workload;
            let c = command.candidate;
            if command.inner != p.inner || command.repeats != p.repeats || !workloads.contains(&w) {
                return Ok(());
            }
            recorded.insert((w, c), status.to_owned());
            if status == "measured" {
                let record = summarize(&command, r["result"].clone(), max_noise)?;
                max_error = max_error.max(r["result"]["max_abs_error"].as_f64().unwrap());
                mean_error = mean_error.max(r["result"]["mean_abs_error"].as_f64().unwrap());
                summaries.entry(w).or_default().push(record);
            }
            Ok(())
        })?;
    }
    let mut coverage: BTreeMap<String, usize> = BTreeMap::new();
    let mut comparisons = Vec::new();
    let mut best = Vec::new();
    let mut profiles: BTreeMap<(u32, u32, u32, u32), Vec<Workload>> = BTreeMap::new();
    let mut unstable_winners = 0;
    for &w in &workloads {
        profiles
            .entry((w.ratio * w.kv_heads, w.kv_heads, w.dim, w.page))
            .or_default()
            .push(w);
        let mut workload_coverage = BTreeMap::new();
        for c in p.candidates(w) {
            let status = recorded.get(&(w, c)).map(String::as_str).unwrap_or("missing");
            *coverage.entry(status.to_owned()).or_default() += 1;
            *workload_coverage.entry(status.to_owned()).or_default() += 1;
        }
        if let Some(records) = summaries.get_mut(&w) {
            records.sort_by(|a, b| {
                a.median_relative_time
                    .total_cmp(&b.median_relative_time)
                    .then_with(|| a.candidate.cmp(&b.candidate))
            });
            best.push(records[0].clone());
            unstable_winners += usize::from(!records[0].stable);
            let stable = records.iter().find(|r| r.stable);
            let current = records.iter().find(|r| r.stable && r.candidate.baseline(w));
            comparisons.push(
                json!({"workload":w,"candidate_coverage":workload_coverage,"candidate_scan_complete":candidate_scan_complete(&workload_coverage),
                "raw_best":records[0],"stable_best":stable,"best_existing_kernel":current,
                "stable_speedup":stable.zip(current).map(|(s,c)| c.median_relative_time / s.median_relative_time)}),
            );
        }
    }
    let mut tables = Vec::new();
    let mut resolved = 0;
    let mut compression = Vec::new();
    for ((num_q_heads, num_kv_heads, head_dim, tokens_per_page), ws) in profiles {
        let axes: [Vec<u32>; 3] = [
            ws.iter()
                .map(|w| w.history)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            ws.iter()
                .map(|w| w.queries)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            ws.iter()
                .map(|w| w.requests)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
        ];
        let mut cells = Vec::new();
        let mut unresolved = Vec::new();
        for w in ws {
            let point = [w.history, w.queries, w.requests];
            let eligible: Vec<_> = summaries.get(&w).into_iter().flatten().filter(|r| r.stable).collect();
            // Summaries already follow normalized-time order. The first eligible record owns the reference.
            let Some(reference) = eligible.first() else {
                unresolved.push(point);
                continue;
            };
            cells.push(Cell {
                coordinate: std::array::from_fn(|i| axes[i].binary_search(&point[i]).unwrap()),
                reference_wall_ms: reference.median_wall_ms,
                choices: eligible
                    .iter()
                    .map(|r| (Route::new(r.candidate, w), r.median_relative_time))
                    .collect(),
            });
        }
        resolved += cells.len();
        compression.push(json!({"num_q_heads":num_q_heads,"num_kv_heads":num_kv_heads,
            "head_dim":head_dim,"tokens_per_page":tokens_per_page,
            "alternatives":compression_frontier(&cells,max_regret)}));
        let (routes, indices, max_observed_regret, _) = compress(&cells, max_regret, 0.0);
        let mut decisions = vec![0; axes.iter().map(Vec::len).product()];
        for (cell, route) in cells.iter().zip(indices) {
            let index = flat_index(cell.coordinate, &axes);
            assert_eq!(decisions[index], 0, "one decision per measured point");
            decisions[index] = route;
        }
        for point in &unresolved {
            let index = std::array::from_fn(|i| axes[i].binary_search(&point[i]).unwrap());
            assert_eq!(
                decisions[flat_index(index, &axes)],
                0,
                "missing point must not be exported"
            );
        }
        tables.push(Table {
            num_q_heads,
            num_kv_heads,
            head_dim,
            tokens_per_page,
            axes,
            routes,
            decisions,
            unresolved,
            max_observed_regret,
        });
    }
    let identity: Value = if out.join("controller-identity.json").exists() {
        read_json(&out.join("controller-identity.json"))?
    } else {
        read_json(&out.join("identity.json"))?
    };
    let device = calibration_device(&identity)?;
    let num_compute_units = calibration_compute_units(&device)?;
    let frozen = if let Some(path) = frozen_path {
        Some(read_frozen_policy(path, &device)?)
    } else {
        p.frozen_policy.clone()
    };
    if let Some((_, source)) = &frozen {
        decode_frozen_policy(&source["document"], &device)?;
        validate_policy_output(source, out)?;
    }
    let frozen_evaluation = frozen.as_ref().map(|(policy, source)| json!({
        "source":source,"contract":"Frozen parameters. No refit or winner substitution. Missing and unstable selected routes remain unverified.",
        "evaluation":evaluate_policy(&workloads, &summaries, &recorded, *policy)}));
    let complete = candidate_scan_complete(&coverage);
    let fitted = fit_policy(&workloads, &summaries, num_compute_units);
    let dictionary_entries: usize = tables.iter().map(|t| t.routes.len()).sum();
    let decision_bytes: usize = tables.iter().map(|t| t.decisions.len() * size_of::<u16>()).sum();
    let executable = std::env::current_exe().map_err(unavailable)?;
    let generator = json!({"argv":std::env::args().collect::<Vec<_>>(),
        "executable_sha256":output(Path::new("."), "/usr/bin/shasum", &["-a", "256", executable.to_str().unwrap()])?});
    let stats = json!({"generator":generator,"mode":p.mode,"metric":"amortized_replay_wall_ms","recorded_statuses":counts,"selected_coverage":coverage,"reference_scope":"all compatible recorded candidates for the input workloads; selected coverage describes only the current plan","workloads":workloads.len(),"measured_workloads":summaries.len(),"table_covered_points":resolved,"candidate_dictionary_entries":dictionary_entries,"decision_bytes":decision_bytes,"candidate_scan_complete":complete,
        "factored_policy":fitted.as_ref().map(|(policy,_)| json!({"geometry_strategy_slots":STRATEGIES.len()*STRATEGIES[0].len(),"slice_threshold_entries":8,"policy":policy})),
        "unstable_raw_winners":unstable_winners,"max_abs_error":max_error,"max_mean_abs_error":mean_error,"max_noise":max_noise,"max_regret":max_regret,"production_ready":false});
    write_json(&out.join("best.json"), &best)?;
    write_json(&out.join("comparisons.json"), &comparisons)?;
    write_json(&out.join("report.json"), &stats)?;
    write_json(
        &out.join("compression.json"),
        &json!({"contract":"Exploratory greedy reuse under a per-point relative OR absolute loss bound. Absolute loss is estimated from normalized ratios and reference wall time. Geometry counts separate kernel constants and output mode from KV slices. The geometry cover freely chooses a measured slice per point and does not define a runtime policy. These alternatives do not change selection.json.","profiles":compression}),
    )?;
    write_json(
        &out.join("selection.json"),
        &json!({"schema":1,"identity":identity,"operator":{"io_dtype":"bfloat16","cache_dtype":"fp8_e4m3","visibility":"uniform per-request causal history; no sliding window"},"axis_order":["history","queries_per_request","requests"],"lookup_contract":"exact axis values; flat index=(history_index*query_count+query_index)*request_count+request_index; decision 0 means unknown; other decisions are one-based route indices","max_regret":max_regret,"max_noise":max_noise,"profiles":tables,"production_ready":false}),
    )?;
    let proposals: Vec<_> = fitted
        .as_ref()
        .into_iter()
        .flat_map(|(policy, _)| {
            [16, 32, 64].map(|short_q_token_heads| {
                evaluate_policy(
                    &workloads,
                    &summaries,
                    &recorded,
                    Policy {
                        short_q_token_heads,
                        ..*policy
                    },
                )
            })
        })
        .collect();
    let mut holdouts = Vec::new();
    let histories: BTreeSet<_> = workloads.iter().map(|w| w.history).collect();
    for history in histories {
        let training: BTreeSet<_> = workloads.iter().copied().filter(|w| w.history != history).collect();
        let validation: BTreeSet<_> = workloads.iter().copied().filter(|w| w.history == history).collect();
        let fold = fit_policy(&training, &summaries, num_compute_units);
        let evaluation = fold
            .as_ref()
            .map(|(policy, _)| evaluate_policy(&validation, &summaries, &recorded, *policy));
        holdouts.push(
            json!({"withheld_history":history,"fit":fold.as_ref().map(|(_, data)| data),"evaluation":evaluation}),
        );
    }
    write_json(
        &out.join("policy-evaluation.json"),
        &json!({"contract":"Generic resource criteria fitted and evaluated against complete selected routes. No per-point candidate or partition oracle is used. Missing and unstable selected measurements remain unverified. The reference is the best stable compatible recorded candidate for each input workload; incomplete scans do not establish a complete optimum. These proposals do not change selection.json or production routing.",
            "generator":generator,"calibration":fitted.as_ref().map(|(_, data)| data),
            "calibration_contract":"Fit nine resource pairs, 195 base-grid/partition parameter sets, then seven single-query policies on common stable measured support for each stage. Later stages must not worsen the preceding maximum relative loss or estimated extra time in any cost tier. Rank costs >=2, >=1, then >=0.2 ms. Refit all stages for each context holdout. Missingness is not a timing signal. At least one stage must select different complete routes on its training support. A stage with no route differences retains its prior coefficients. The kernel repertoire, single-query D128/head-tile eligibility, slice envelope, and first two general slice thresholds remain prior calibration.",
            "geometry_criteria":{"q_token_head_capacity":"q_token_head_values / head_dimension","joint_SIMDgroups":"ceil(token_tile * head_extent / 8)","head_SIMDgroups":"token_tile * ceil(head_extent / 8)","q_heads_per_token":"The primary Q-token/head criterion uses QHeadsPerToken when it adds SIMDgroups and ceil(KV_stage / joint_SIMDgroups) exceeds loader_tokens. A calibrated small-base-grid criterion can also use QHeadsPerToken."},
            "context_holdouts":holdouts,"frozen_policy":frozen_evaluation,
            "evaluated_identity":identity,"production_ready":false,
            "partition_feature":"visible_KV_tokens^2 * requests * SIMDgroups_per_threadblock^2",
            "shared_slice_thresholds":[[1_u128<<26,64],[1_u128<<30,128]],"single_query_upper_threshold":1_u128<<34,"last_slice":512,
            "base_grid":"requests * KV_heads * ceil(queries / token_tile) * ceil(head_ratio / head_extent)",
            "slice_floor_feature":"visible_KV_tokens * base_grid * SIMDgroups_per_threadblock",
            "proposals":proposals}),
    )?;
    write_json(
        &out.join("policy-capacity.json"),
        &json!({"local":fitted.as_ref().map(|(policy,_)| policy_capacity(&workloads,*policy)),
            "frozen":frozen.as_ref().map(|(policy,source)| json!({"source":source,"capacity":policy_capacity(&workloads,*policy)}))}),
    )?;
    export_rust(out, &tables, &device)?;
    export_policy(
        out,
        "policy",
        fitted.as_ref().map(|(policy, _)| *policy),
        &identity,
        &device,
    )?;
    export_policy(
        out,
        "frozen-policy",
        frozen.as_ref().map(|(policy, _)| *policy),
        &identity,
        &device,
    )?;
    let policy_summary = if let Some((policy, _)) = &fitted {
        let mut summary = format!(
            "The fitted proposal uses `q_token_head_values={}`, `loader_tokens={}`, `max_head_base_groups={}`, \
             `max_floor_base_groups={}`, `target_split_groups={}`, `upper_slice_shift={}`, and `decode={:?}`.\n\nIt \
             exports 24 geometry strategy slots, two four-entry general slice tables, and the single-query partition \
             criterion.\n\n",
            policy.q_token_head_values,
            policy.loader_tokens,
            policy.max_head_base_groups,
            policy.max_floor_base_groups,
            policy.target_split_groups,
            policy.upper_slice_shift,
            policy.decode,
        );
        summary.push_str(&cost_table(&proposals[1]));
        summary
    } else {
        "Factored calibration is unavailable. It needs common stable complete-route support at >=0.2 ms and tunable \
         route differences.\n"
            .to_owned()
    };
    let frozen_summary = frozen_evaluation.as_ref().map_or_else(String::new, |evaluation| {
        format!(
            "## Frozen policy comparison\n\nSource: `{}`. The report does not refit these parameters.\n\n{}\n",
            evaluation["source"]["path"].as_str().unwrap(),
            cost_table(&evaluation["evaluation"])
        )
    });
    let markdown = format!(
        r#"# Generic GQA matrix report

The metric is amortized replay wall time. It includes submission and completion wait.
The timing baseline is full-slice SlicedKv.
Policy loss uses the best stable compatible recorded route for each input workload.
Selected coverage and scan completeness describe only the current plan.
A focused policy scan does not establish the full candidate optimum.
The comparison envelope contains stable measurements of existing kernel geometries and KV slices.
These measurements do not exercise production selection or model throughput.

| Item | Result |
| --- | ---: |
| Workloads | {} |
| Workloads with measurements | {} |
| Points covered by the table | {} |
| Candidate dictionary entries | {} |
| Decision storage bytes | {} |
| Unstable raw winners | {} |
| Maximum absolute output error | {} |
| Maximum mean output error | {} |
| Maximum allowed observed regret | {:.1}% |
| Maximum allowed noise | {:.1}% |
| Selected candidate scan complete | {} |

Selected coverage: `{}`.

The quality filter checks baseline brackets, candidate brackets, and the normalized sample range.
The table contains only measured candidates that pass numerical and quality checks.
Each profile contains a candidate dictionary and a dense decision grid.
The axes contain exact measured values. The grid does not cover intervening unmeasured shapes.
The generator verifies one decision per covered point and the measured regret bound at every point.
The regret bound is descriptive. It is not a statistical confidence interval.

[Comparisons](comparisons.json) retain raw winners, stable winners, and the stable existing-kernel envelope.
Each measured workload includes candidate coverage and scan completeness for the current mode.
An incomplete scan establishes only the best recorded candidate. It does not establish the complete candidate optimum.
[Policy evaluation](policy-evaluation.json) compares fixed generic criteria with actual complete measured routes.
It reports missing and unstable selected candidates without replacing them with a measured winner.
Four geometry families use a record/dimension budget and a cooperative KV-loader allowance.
The report fits nine resource pairs, 195 base-grid/partition sets, then seven single-query policies on common stable support.
All three stages are refitted for each context holdout.
A stage with no tunable route differences retains its prior coefficients.
Calibration is unavailable when no stage changes a route on common stable support.
Each later stage cannot worsen the preceding maximum loss or estimated extra time in any cost tier.
The kernel repertoire, single-query D128/head-tile eligibility, slice envelope, and first two general thresholds remain calibration priors.
These holdouts do not validate that prior family or slice selection.
They do not change the exact-grid diagnostic or production routing.

{policy_summary}
{frozen_summary}
[Compression tradeoffs](compression.json) compare shared candidates under per-point relative and absolute bounds.
These alternatives do not change either exported lookup. Absolute extra time is an estimate from normalized timing.
The geometry cover reuses kernel constants and output modes with a measured KV-slice oracle at each point.
It is an ablation. It does not define a runtime policy or prove a minimum cover size.
[Policy JSON](policy.json) and [policy Rust](policy.rs) export the local fitted lookup.
[Frozen policy JSON](frozen-policy.json) and [frozen policy Rust](frozen-policy.rs) export the evaluated frozen coefficients.
The frozen export is unavailable when the report has no frozen input.
The evaluation records the frozen source path and original document.
Both exports use the dataset identity and the same criterion implementation.
The lookup uses 24 geometry strategy slots, general KV slice tables, and one shared single-query partition criterion.
It accepts legal intermediate context lengths. The measured grid is not a context cap.
Selection outside the measured grid does not establish numerical, replay, scratch, or performance validation.
[Policy capacity](policy-capacity.json) checks continuous-context active Map and partial-state bounds at criterion boundaries.
It records a peak witness for each geometry and preserves unavailable routes.
Its endpoints come from this plan. They do not limit runtime context lengths.
These uniform active bounds do not include production replay padding or prove ragged lifecycle allocations.
This analysis uses no GPU measurement.
The export returns `None` when fitting is unavailable, the operator domain is unsupported, or hardware differs.
It contains no per-point winner lookup or runtime search.

[Selection JSON](selection.json) contains the exact-grid diagnostic and complete candidate parameters.
[Selection Rust](selection.rs) provides a direct lookup with explicit unknown-shape results.
[Raw records](results.sqlite3) retain samples and failures.
[Run provenance](runs.jsonl) records measurement source, power, command, and environment.
[Report provenance](report.json) records the generator command and executable SHA-256.

The table is a calibration artifact.
Production promotion still requires replay-capacity, ragged-batch, boundary, and service validation.
"#,
        workloads.len(),
        summaries.len(),
        resolved,
        dictionary_entries,
        decision_bytes,
        unstable_winners,
        max_error,
        mean_error,
        max_regret * 100.0,
        max_noise * 100.0,
        complete,
        encoded(&coverage)
    );
    fs::write(out.join("report.md"), markdown).map_err(unavailable)?;
    println!(
        "Selected coverage: {}. Exact-grid diagnostic: {resolved}/{} points, {dictionary_entries} dictionary entries, \
         {decision_bytes} decision bytes. Report: {}",
        encoded(&coverage),
        workloads.len(),
        out.join("report.md").display()
    );
    println!(
        "Factored policy: {}. Geometry strategy slots: {}. Export: {}",
        if fitted.is_some() { "fitted" } else { "unavailable" },
        STRATEGIES.len() * STRATEGIES[0].len(),
        out.join("policy.rs").display()
    );
    Ok(())
}

// Within a fixed geometry and slice interval, every causal Map count is nondecreasing in visibility.
// Include both sides of all geometry, partition, floor, and decode-grid boundaries.
fn policy_contexts(w: Workload, policy: Policy, max_visible_kv_tokens: u32) -> Vec<u32> {
    let mut contexts = vec![w.queries, max_visible_kv_tokens];
    let mut add = |boundary: u128| {
        let mut point = |value| {
            if (w.queries as u128..=max_visible_kv_tokens as u128).contains(&value) {
                contexts.push(value as u32);
            }
        };
        if boundary > 0 {
            point(boundary - 1);
        }
        point(boundary);
        point(boundary + 1);
    };
    add(512);
    add(1024);
    add(4096);
    for groups in 1_u128..=32 {
        let coefficient = w.requests as u128 * groups.pow(2);
        for limit in [1_u128 << 26, 1 << 30, 1 << 34, 1 << (34 + policy.upper_slice_shift)] {
            add((limit / coefficient).isqrt());
        }
        for tile in [1, 2, 4, 8, 16, 32] {
            let base_tasks = w.requests as u128 * w.kv_heads as u128 * w.queries.div_ceil(tile) as u128;
            for slice in [64_u128, 128, 256] {
                add(policy.target_split_groups as u128 * slice / (base_tasks * groups));
            }
        }
    }
    if let DecodePolicy::Adaptive {
        min_grid_groups,
        target_groups,
    } = policy.decode
    {
        let matrix_groups = w.requests as u128 * w.kv_heads as u128 * w.ratio.div_ceil(8) as u128;
        let single_groups = matrix_groups * 4;
        for (groups, max_slice) in [(matrix_groups, 512), (single_groups, 1024)] {
            for slice in [512_u128, 1024].into_iter().filter(|&slice| slice <= max_slice) {
                add((target_groups as u128 * slice).div_ceil(groups));
            }
        }
        let splits = (min_grid_groups as u128).div_ceil(matrix_groups);
        for slice in [256_u128, 512] {
            add((splits - 1) * slice + 1);
        }
    }
    contexts.sort_unstable();
    contexts.dedup();
    contexts
}

fn policy_capacity(workloads: &BTreeSet<Workload>, policy: Policy) -> Value {
    let mut envelopes = BTreeMap::new();
    let mut profiles = BTreeMap::<Workload, u32>::new();
    for &w in workloads {
        profiles
            .entry(Workload { history: 0, ..w })
            .and_modify(|end| *end = (*end).max(w.history + w.queries))
            .or_insert(w.history + w.queries);
    }
    let profiles = profiles
        .into_iter()
        .map(|(w, max_visible_kv_tokens)| {
            let contexts = policy_contexts(w, policy, max_visible_kv_tokens);
            let mut geometries = BTreeMap::new();
            let mut unavailable = Vec::new();
            for &end in &contexts {
                let workload = Workload { history: end - w.queries, ..w };
                let Some(candidate) = proposed_candidate(workload, policy) else {
                    unavailable.push(end);
                    continue;
                };
                let num_q_token_ranges = w.queries.div_ceil(candidate.tile) as u64 * w.requests as u64;
                let num_map_task_templates = (0..w.queries)
                    .step_by(candidate.tile as usize)
                    .map(|first| (workload.history + first + candidate.tile.min(w.queries - first)).div_ceil(candidate.slice) as u64)
                    .sum::<u64>() * w.requests as u64;
                let max_splits = end.div_ceil(candidate.slice);
                let key = (candidate.layout, candidate.tile, candidate.heads, candidate.stage, candidate.direct);
                let value = geometries.entry(key).or_insert((0, 0, end, candidate));
                value.1 = value.1.max(max_splits);
                if num_map_task_templates > value.0 {
                    value.0 = num_map_task_templates;
                    value.2 = end;
                    value.3 = candidate;
                }
                debug_assert!(num_map_task_templates >= num_q_token_ranges);
            }
            let (max_tokens, max_requests, max_visible, geometry_maps) = envelopes
                .entry((w.dim, w.ratio, w.kv_heads, w.page))
                .or_insert((0_u32, 0_u32, 0_u32, BTreeMap::new()));
            *max_tokens = (*max_tokens).max(w.queries * w.requests);
            *max_requests = (*max_requests).max(w.requests);
            *max_visible = (*max_visible).max(max_visible_kv_tokens);
            for (&geometry, &(maps, _, _, _)) in &geometries {
                geometry_maps
                    .entry(geometry)
                    .and_modify(|bound: &mut u64| *bound = (*bound).max(maps))
                    .or_insert(maps);
            }
            let geometries = geometries.into_iter().map(|((layout, tile, heads, stage, direct), (maps, splits, end, candidate))| {
                json!({"geometry":{"layout":layout,"tile":tile,"heads":heads,"stage":stage,"direct":direct},
                    "max_q_token_ranges":w.queries.div_ceil(tile) as u64*w.requests as u64,
                    "max_splits_per_q_range":splits,"max_active_map_task_templates":maps,
                    "max_active_reserved_partial_state_groups":if direct {0} else {maps*tile as u64},
                    "map_peak_visible_kv_tokens":end,"map_peak_candidate":candidate})
            }).collect::<Vec<_>>();
            json!({"workload":w,"min_visible_kv_tokens":w.queries,"max_visible_kv_tokens":max_visible_kv_tokens,
                "boundary_points_evaluated":contexts.len(),"unavailable_boundary_points":unavailable,"geometries":geometries})
        })
        .collect::<Vec<_>>();
    let envelopes = envelopes
        .into_iter()
        .map(|((dim, ratio, kv_heads, page), (max_tokens, max_requests, max_visible, geometry_maps))| {
            let operator = json!({"head_dim":dim,"q_heads_per_kv_head":ratio,"num_kv_heads":kv_heads,"tokens_per_page":page});
            let bounds = json!({"max_tokens":max_tokens,"max_requests":max_requests,"max_visible_kv_tokens":max_visible});
            let Some(max_active_maps) = geometry_maps.values().copied().max() else {
                return json!({"operator":operator,"analyzed_uniform_input_bounds":bounds,"available":false,
                    "reason":"No legal selected geometry in this input envelope."});
            };
            let Ok(max_active_maps) = u32::try_from(max_active_maps) else {
                return json!({"operator":operator,"analyzed_uniform_input_bounds":bounds,"available":false,
                    "reason":"Active Map task-template count exceeds the production u32 domain."});
            };
            let q_ranges = ReplayBucketPolicy::new(max_tokens);
            // Direct replay uses the same recorded Map and Q-range extents.
            let direct_maps = geometry_maps
                .iter()
                .filter(|((_, _, _, _, direct), _)| *direct)
                .map(|(_, &maps)| q_ranges.capacity(maps as u32))
                .max()
                .unwrap_or(0);
            let max_map_task_templates = max_active_maps.max(direct_maps);
            let map_buckets = ReplayBucketPolicy::new(max_map_task_templates);
            let mut partial_state_group_capacity = 0_u64;
            let geometries = geometry_maps
                .into_iter()
                .map(|((layout, tile, heads, stage, direct), maps)| {
                    let total_maps = if direct {
                        q_ranges.capacity(maps as u32)
                    } else {
                        map_buckets.capacity(maps as u32)
                    };
                    let partial_groups = if direct { 0 } else { total_maps as u64 * tile as u64 };
                    partial_state_group_capacity = partial_state_group_capacity.max(partial_groups);
                    json!({"geometry":{"layout":layout,"tile":tile,"heads":heads,"stage":stage,"direct":direct},
                        "max_active_map_task_templates":maps,"max_recorded_map_task_templates":total_maps,
                        "max_replay_reserved_partial_state_groups":partial_groups})
                })
                .collect::<Vec<_>>();
            json!({"operator":operator,"analyzed_uniform_input_bounds":bounds,"available":true,
                "max_map_task_templates":max_map_task_templates,"partial_state_group_capacity":partial_state_group_capacity,
                "geometries":geometries})
        })
        .collect::<Vec<_>>();
    json!({"contract":"CPU analysis of the complete selected route over each supplied uniform workload profile's continuous visible-context interval. Evaluate both sides of all possible criterion boundaries. Map counts are nondecreasing within a fixed geometry and slice interval. The shared envelopes use production replay buckets per geometry and preserve direct Map/Q-range extent equality. They cover the union of supplied query/request profiles, not every combination within the displayed input bounds. Configured lifecycle and ragged routing remain separate. The plan's visibility endpoint is an analysis envelope, not a runtime context cap. No GPU measurement is added.",
        "policy":policy,"profiles":profiles,"uniform_replay_envelopes":envelopes,"production_ready":false})
}

fn cost_table(evaluation: &Value) -> String {
    let mut summary = String::from(
        "| Reference cost | Verified / reference points | Maximum observed loss | Maximum estimated extra time |\n| \
         --- | ---: | ---: | ---: |\n",
    );
    for bucket in evaluation["reference_time_buckets"].as_array().unwrap() {
        let minimum = bucket["min_reference_wall_ms"].as_f64().unwrap();
        if minimum < 0.2 {
            continue;
        }
        let regret = bucket["max_observed_regret"]
            .as_f64()
            .map_or_else(|| "unknown".to_owned(), |n| format!("{:.2}%", n * 100.0));
        let extra = bucket["max_estimated_extra_ms"]
            .as_f64()
            .map_or_else(|| "unknown".to_owned(), |n| format!("{n:.3} ms"));
        summary.push_str(&format!(
            "| >={minimum} ms | {} / {} | {regret} | {extra} |\n",
            bucket["verified_points"], bucket["reference_points"]
        ));
    }
    summary
}

fn export_rust(out: &Path, tables: &[Table], device: &str) -> Result<()> {
    let mut code = String::from(
        "// Generated calibration artifact. BF16 Q/output, FP8 E4M3 cache, uniform causal requests.\n// Exact \
         measured axis values only. Validate production replay capacities before promotion.\n// Route fields: layout, \
         token tile, head extent, KV stage, KV slice (None=full), direct output.\npub type Route = (&'static str, \
         u32, u32, u32, Option<u32>, bool);\n#[allow(clippy::too_many_arguments)]\npub fn lookup(device: &str, \
         num_q_heads: u32, num_kv_heads: u32, head_dim: u32, tokens_per_page: u32, history: u32, queries_per_request: \
         u32, requests: u32) -> Option<Route> {\n",
    );
    code.push_str(&format!(
        "    if device != {device:?} {{ return None; }}\n    match (num_q_heads, num_kv_heads, head_dim, \
         tokens_per_page) {{\n"
    ));
    for t in tables {
        code.push_str(&format!(
            "        ({}, {}, {}, {}) => {{\n",
            t.num_q_heads, t.num_kv_heads, t.head_dim, t.tokens_per_page
        ));
        for (a, name) in ["history", "queries_per_request", "requests"].iter().enumerate() {
            code.push_str(&format!(
                "            let i{a} = {:?}.binary_search(&{name}).ok()?;\n",
                t.axes[a]
            ));
        }
        code.push_str("            let routes: &[Route] = &[\n");
        for r in &t.routes {
            let layout = encoded(&r.layout);
            code.push_str(&format!(
                "                ({layout}, {}, {}, {}, {:?}, {}),\n",
                r.tile, r.heads, r.stage, r.slice, r.direct
            ));
        }
        code.push_str("            ];\n");
        code.push_str(&format!("            let decisions: &[u16] = &{:?};\n", t.decisions));
        code.push_str(&format!(
            "            let index = (i0 * {} + i1) * {} + i2;\n",
            t.axes[1].len(),
            t.axes[2].len()
        ));
        code.push_str(
            "            let decision = decisions[index];\n            if decision == 0 { None } else { \
             Some(routes[decision as usize - 1]) }\n        }\n",
        );
    }
    code.push_str("        _ => None,\n    }\n}\n");
    fs::write(out.join("selection.rs"), code).map_err(unavailable)
}

// This export shares the criterion implementation with evaluation. It is not a production selector.
fn export_policy(out: &Path, name: &str, policy: Option<Policy>, identity: &Value, device: &str) -> Result<()> {
    let thresholds = policy.map(|p| {
        json!({
            "partition_feature": "visible_KV_tokens^2 * requests * SIMDgroups_per_threadblock^2",
            "slice": [[1_u128 << 26, 64], [1_u128 << 30, 128], [1_u128 << (34 + p.upper_slice_shift), 256]],
            "single_query_upper_threshold": 1_u128 << 34,
            "floor_feature": "visible_KV_tokens * base_grid * SIMDgroups_per_threadblock",
            "floor": [[p.target_split_groups as u128 * 64, 64], [p.target_split_groups as u128 * 128, 128], [p.target_split_groups as u128 * 256, 256]],
            "last_slice": 512
        })
    });
    write_json(
        &out.join(format!("{name}.json")),
        &json!({"schema":4,"identity":identity,"device":device,
            "operator":{"io_dtype":"bfloat16","cache_dtype":"fp8_e4m3","visibility":"uniform per-request causal history; no sliding window"},
            "contract":"Factored empirical criterion lookup. Geometry has four static profiles and six rounded-query columns. General KV slice lookup uses integer thresholds. Single-query lookup uses calibrated partition budgets. Legal context lengths are not restricted to measured coordinates. No runtime search or per-point winner lookup.",
            "strategy_profiles":["D128/G<=8","D128/G>8","D256/G<=8","D256/G>8"],
            "query_capacities":[1,2,4,8,16,32],"strategies":STRATEGIES,
            "strategy_values":{"0":"small-query D128","1":"resource Q-token/head positions","2":"legacy QTokensPerHead","3":"single-query KV reuse"},
            "short_context_visible_kv_limits":{"single_query":32,"multi_query":512},
            "decode_criteria":{"criterion":"shared-partitioned-grid-v2",
                "matrix_base_groups":"requests * KV_heads * ceil(GQA_ratio / 8)",
                "matrix_slice":"clamp(floor_power_of_two(visible_KV_tokens * matrix_base_groups / target_groups), 256, 512)",
                "matrix_grid":"matrix_base_groups * ceil(visible_KV_tokens / matrix_slice)",
                "q_tokens_and_heads":"D256 with more than eight heads and matrix_grid >= min_grid_groups uses full-head QTokensAndHeads with KV stage 16.",
                "single_query_slice":"clamp(floor_power_of_two(visible_KV_tokens * SlicedKv_base_tasks * 4 / target_groups), 256, 1024)",
                "fixed_prior":{"reuse_grid":DECODE_REUSE_GRID,"slice":DECODE_SLICE},
                "calibration":"The adaptive strategy shares one partition budget across both paths. Fixed retains prior routing when decode fitting is unavailable or invariant. Read complete-route loss and context holdouts before promotion."},
            "policy":policy,"thresholds":thresholds,"production_ready":false,
            "validation":"Read policy-evaluation.json for measured complete-route loss and coverage. Arbitrary-context selection does not establish numerical, scratch, replay, or performance validation outside measured support."}),
    )?;
    let coefficients = policy.map_or_else(
        || "None".to_owned(),
        |p| {
            format!(
                "Some(Policy {{ short_q_token_heads: {}, q_token_head_values: {}, loader_tokens: {}, \
                 max_head_base_groups: {}, max_floor_base_groups: {}, target_split_groups: {}, upper_slice_shift: {}, \
                 decode: DecodePolicy::{:?} }})",
                p.short_q_token_heads,
                p.q_token_head_values,
                p.loader_tokens,
                p.max_head_base_groups,
                p.max_floor_base_groups,
                p.target_split_groups,
                p.upper_slice_shift,
                p.decode,
            )
        },
    );
    let mut code = String::from(
        r#"// Generated empirical lookup. BF16 Q/output, FP8 E4M3 cache, uniform causal requests.
// This artifact does not change production selection. Read policy-evaluation.json before promotion.
// Route fields: layout, token tile, head extent, KV stage, KV slice (None=full), direct output.
pub type Route = (&'static str, u32, u32, u32, Option<u32>, bool);

fn supports_operator(num_q_heads: u32, num_kv_heads: u32, head_dim: u32, tokens_per_page: u32) -> bool {
    num_q_heads > 0 && num_kv_heads > 0 && num_q_heads.is_multiple_of(num_kv_heads)
        && [128, 256].contains(&head_dim) && [8, 16, 32].contains(&tokens_per_page)
        && (head_dim != 128 || tokens_per_page != 32)
}

#[derive(Clone, Copy)]
struct Workload {
    history: u32,
    queries: u32,
    requests: u32,
    ratio: u32,
    dim: u32,
    kv_heads: u32,
}
#[derive(Clone, Copy, PartialEq)]
enum Layout { SlicedKv, QHeadsPerToken, QTokensPerHead, QTokensAndHeads }
struct Candidate {
    layout: Layout,
    tile: u32,
    heads: u32,
    stage: u32,
    slice: u32,
    direct: bool,
}
impl Candidate {
    fn route(self, end: u32) -> Route {
        let layout = match self.layout {
            Layout::SlicedKv => "single", Layout::QHeadsPerToken => "head",
            Layout::QTokensPerHead => "token", Layout::QTokensAndHeads => "joint",
        };
        (layout, self.tile, self.heads, self.stage, (self.slice != end).then_some(self.slice), self.direct)
    }

    fn threads(&self) -> u64 {
        match self.layout {
            Layout::SlicedKv => 128,
            Layout::QTokensPerHead => self.tile.div_ceil(8) as u64 * self.heads as u64 * 32,
            Layout::QHeadsPerToken => self.tile as u64 * self.heads.div_ceil(8) as u64 * 32,
            Layout::QTokensAndHeads => (self.tile as u64 * self.heads as u64).div_ceil(8) * 32,
        }
    }
}

"#,
    );
    // Keep one authoritative implementation. The standalone artifact does not depend on serde.
    code.push_str(
        &include_str!("policy.rs")
            .replace(
                "#[derive(Clone, Copy, Deserialize, Serialize)]",
                "#[derive(Clone, Copy)]",
            )
            .replace(
                "#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]",
                "#[derive(Clone, Copy, Debug, Default)]",
            )
            // A generated artifact constructs one policy mode. Keep both template variants.
            .replace("enum DecodePolicy {", "#[allow(dead_code)]\nenum DecodePolicy {")
            .replace("#[serde(default)]", "")
            .replace("#[serde(rename = \"short_rows\")]", "")
            .replace("#[serde(rename = \"row_elements\")]", "")
            .replace("#[serde(tag = \"kind\", rename_all = \"snake_case\")]", ""),
    );
    code.push_str(&format!("\nconst POLICY: Option<Policy> = {coefficients};\n"));
    code.push_str(&format!(
        r#"
#[allow(clippy::too_many_arguments)]
pub fn lookup(
    device: &str, num_q_heads: u32, num_kv_heads: u32, head_dim: u32,
    tokens_per_page: u32, history: u32, queries_per_request: u32, requests: u32,
) -> Option<Route> {{
    if device != {device:?} {{ return None; }}
    let policy = POLICY?;
    if queries_per_request == 0 || requests == 0
        || !supports_operator(num_q_heads, num_kv_heads, head_dim, tokens_per_page) {{ return None; }}
    // Validate the same token, page, and host-allocation domains as the matrix input boundary.
    let end = history.checked_add(queries_per_request)?;
    let tokens = queries_per_request.checked_mul(requests)?;
    let page_bytes = num_kv_heads.checked_mul(tokens_per_page)?.checked_mul(head_dim)?.checked_mul(2)?;
    let pages = end.div_ceil(tokens_per_page).checked_mul(requests)?;
    let q_bytes = (num_q_heads as u64 * tokens as u64).checked_mul(head_dim as u64 * 2)?;
    let kv_bytes = pages as u64 * page_bytes as u64;
    if q_bytes > isize::MAX as u64 || kv_bytes > isize::MAX as u64 {{ return None; }}
    let w = Workload {{ history, queries: queries_per_request, requests,
        ratio: num_q_heads / num_kv_heads, dim: head_dim, kv_heads: num_kv_heads }};
    let c = proposed_candidate(w, policy)?;
    Some(c.route(end))
}}

/// Each request shape is (history tokens, positive query tokens).
/// Ragged routing extends the uniform calibration. It does not establish ragged performance or lifecycle validation.
pub fn lookup_batch(
    device: &str, num_q_heads: u32, num_kv_heads: u32, head_dim: u32,
    tokens_per_page: u32, request_shapes: &[(u32, u32)],
) -> Option<Route> {{
    if device != {device:?} {{ return None; }}
    let policy = POLICY?;
    if !supports_operator(num_q_heads, num_kv_heads, head_dim, tokens_per_page) {{ return None; }}
    let page_bytes = num_kv_heads.checked_mul(tokens_per_page)?.checked_mul(head_dim)?.checked_mul(2)?;
    let mut tokens = 0_u32;
    let mut pages = 0_u32;
    let mut batch = BatchShape {{ num_requests: 0, max_q_tokens: 0, max_visible_kv_tokens: 0, num_visible_kv_tokens: 0 }};
    for &(history, queries) in request_shapes {{
        if queries == 0 {{ return None; }}
        let end = history.checked_add(queries)?;
        tokens = tokens.checked_add(queries)?;
        pages = pages.checked_add(end.div_ceil(tokens_per_page))?;
        // Positive queries and the proved token sum also bound the request count.
        batch.num_requests += 1;
        batch.max_q_tokens = batch.max_q_tokens.max(queries);
        batch.max_visible_kv_tokens = batch.max_visible_kv_tokens.max(end);
        batch.num_visible_kv_tokens += end as u128;
    }}
    if batch.num_requests == 0 {{ return None; }}
    let q_bytes = (num_q_heads as u64 * tokens as u64).checked_mul(head_dim as u64 * 2)?;
    let kv_bytes = pages as u64 * page_bytes as u64;
    if q_bytes > isize::MAX as u64 || kv_bytes > isize::MAX as u64 {{ return None; }}
    let c = proposed_batch_candidate(
        head_dim, num_q_heads / num_kv_heads, num_kv_heads, batch,
        |tile| request_shapes.iter().map(|&(_, queries)| queries.div_ceil(tile) as u128).sum(),
        |slice| request_shapes.iter().map(|&(history, queries)| (history + queries).div_ceil(slice) as u128).sum(),
        policy,
    )?;
    Some(c.route(batch.max_visible_kv_tokens))
}}
"#
    ));
    fs::write(out.join(format!("{name}.rs")), code).map_err(unavailable)
}

#[cfg(test)]
#[path = "report_test.rs"]
mod tests;
