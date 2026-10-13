use super::*;
use crate::controller::BatchShape;
use crate::controller::proposed_batch_candidate;
use crate::controller::variants;
#[test]
fn test_dictionary_compression_preserves_measured_loss_bound() {
    let a = fixture_route(2);
    let b = fixture_route(4);
    let cells: Vec<_> = [vec![(a, 1.0), (b, 1.04)], vec![(a, 1.04), (b, 1.0)], vec![(b, 1.0)]]
        .into_iter()
        .zip([[0, 0, 0], [1, 0, 0], [3, 0, 0]])
        .map(|(choices, coordinate)| {
            Cell {
                choices,
                coordinate,
                reference_wall_ms: 1.0,
            }
        })
        .collect();
    let (routes, decisions, regret, _) = compress(&cells, 0.05, 0.0);
    assert_eq!(routes, [b]);
    assert_eq!(decisions, [1, 1, 1]);
    assert!(regret <= 0.05);
    let (routes, decisions, ..) = compress(&cells, 0.0, 0.0);
    assert_eq!(routes.len(), 2);
    assert_ne!(decisions[0], decisions[1]);
    let axes = [vec![0, 64, 256, 1024], vec![2], vec![1]];
    let mut grid = vec![0; 4];
    for (cell, route) in cells.iter().zip(decisions) {
        grid[flat_index(cell.coordinate, &axes)] = route;
    }
    assert_eq!(grid[2], 0);
}
#[test]
fn test_focused_policy_report_preserves_broader_recorded_reference() {
    use clap::Parser;

    use crate::controller::Args;
    use crate::controller::Database;
    use crate::controller::Mode;

    let out = std::env::temp_dir().join(format!("gqa-policy-report-{}", std::process::id()));
    fs::create_dir_all(&out).unwrap();
    let identity = json!({"device":{"chip":"test","gpu_cores":"40"}});
    write_json(&out.join("identity.json"), &identity).unwrap();
    let policy = Policy {
        max_head_base_groups: 16,
        max_floor_base_groups: 32,
        target_split_groups: 1024,
        upper_slice_shift: 1,
        ..RESOURCE_POLICY
    };
    export_policy(&out, "policy", Some(policy), &identity, "test/40").unwrap();
    fs::rename(out.join("policy.json"), out.join("frozen.json")).unwrap();
    let frozen = read_frozen_policy(&out.join("frozen.json"), "test/40").unwrap();
    let args = Args::parse_from([
        "bench",
        "--mode",
        "policy",
        "--history",
        "8192",
        "--queries",
        "4",
        "--requests",
        "1",
        "--ratios",
        "6",
        "--dims",
        "256",
        "--kv-heads",
        "4",
    ]);
    let mut plan = Plan::new(&args);
    plan.mode = Mode::Policy;
    plan.frozen_policy = Some(frozen);
    plan.validate().unwrap();
    let w = plan.workloads()[0];
    let selected = proposed_candidate(w, policy).unwrap();
    let faster = Candidate {
        layout: Layout::QTokensAndHeads,
        tile: 2,
        heads: 6,
        stage: 64,
        slice: 2048,
        direct: false,
    };
    assert!(!plan.candidates(w).contains(&faster));
    {
        let mut db = Database::open(&out.join("results.sqlite3")).unwrap();
        for (candidate, time) in [(selected, 12.0), (faster, 10.0)] {
            let command = Command {
                workload: w,
                candidate,
                inner: 8,
                repeats: 5,
            };
            let sample = json!({"before_ms":10.0,"candidate_ms":time,"after_ms":10.0,"candidate_first_ms":time,"candidate_last_ms":time});
            db.insert(&command, &json!({"status":"measured","samples":vec![sample;5],"max_abs_error":0.0,"mean_abs_error":0.0,"cpu_reference_values":3})).unwrap();
        }
    }
    generate(&out, &plan, 0.10, 0.05, None).unwrap();
    let result: Value = read_json(&out.join("policy-evaluation.json")).unwrap();
    let point = &result["frozen_policy"]["evaluation"]["points"][0];
    assert!((point["observed_regret"].as_f64().unwrap() - 0.2).abs() < 1e-12);
    let comparisons: Value = read_json(&out.join("comparisons.json")).unwrap();
    assert_eq!(
        comparisons[0]["stable_best"]["candidate"],
        serde_json::to_value(faster).unwrap()
    );
    let source: Value = read_json(&out.join("frozen.json")).unwrap();
    assert_eq!(source["policy"], serde_json::to_value(policy).unwrap());
    let local: Value = read_json(&out.join("policy.json")).unwrap();
    let exported: Value = read_json(&out.join("frozen-policy.json")).unwrap();
    assert!(local["policy"].is_null(), "one point cannot fit a tunable policy");
    assert_eq!(exported["policy"], source["policy"]);
    assert_eq!(exported["identity"], identity);
    assert!(
        fs::read_to_string(out.join("frozen-policy.rs"))
            .unwrap()
            .contains("Some(Policy")
    );
    // JSON and Rust exports are destinations. A collision must preserve all report artifacts.
    export_policy(&out, "policy", Some(policy), &identity, "test/40").unwrap();
    let before = fs::read(out.join("report.json")).unwrap();
    for name in ["policy.json", "policy.rs", "frozen-policy.json", "frozen-policy.rs"] {
        write_json(&out.join(name), &source).unwrap();
        assert!(generate(&out, &plan, 0.10, 0.05, Some(&out.join(name))).is_err());
        assert_eq!(fs::read(out.join("report.json")).unwrap(), before);
        assert_eq!(read_json::<Value>(&out.join(name)).unwrap(), source);
    }
    // A generated destination can also alias the input through a filesystem link.
    #[cfg(unix)]
    {
        let path = out.join("frozen-policy.rs");
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(out.join("frozen.json"), &path).unwrap();
        assert!(generate(&out, &plan, 0.10, 0.05, None).is_err());
        assert_eq!(fs::read(out.join("report.json")).unwrap(), before);
        assert_eq!(read_json::<Value>(&out.join("frozen.json")).unwrap(), source);
        fs::remove_file(path).unwrap();
    }
    // Removing the input must replace a previous frozen export with an unavailable lookup.
    plan.frozen_policy = None;
    generate(&out, &plan, 0.10, 0.05, None).unwrap();
    assert!(read_json::<Value>(&out.join("frozen-policy.json")).unwrap()["policy"].is_null());
    assert!(
        fs::read_to_string(out.join("frozen-policy.rs"))
            .unwrap()
            .contains("const POLICY: Option<Policy> = None;")
    );
    assert_eq!(read_json::<Value>(&out.join("frozen.json")).unwrap(), source);
    fs::remove_dir_all(out).unwrap();
}

#[test]
fn test_geometry_cover_keeps_partition_oracle_separate() {
    let a = Route {
        slice: Some(64),
        ..fixture_route(4)
    };
    let b = Route { slice: Some(128), ..a };
    let cells = [
        Cell {
            coordinate: [0, 0, 0],
            reference_wall_ms: 1.0,
            choices: vec![(a, 1.0), (b, 1.2)],
        },
        Cell {
            coordinate: [1, 0, 0],
            reference_wall_ms: 1.0,
            choices: vec![(a, 1.2), (b, 1.0)],
        },
    ];
    assert_eq!(compress(&cells, 0.05, 0.0).0.len(), 2);
    assert_eq!(geometry_cover_size(&cells, 0.05, 0.0), 1);
    let alternatives = compression_frontier(&cells, 0.05);
    assert_eq!(alternatives[2]["geometry_cover_entries_with_partition_oracle"], 1);
    assert_eq!(alternatives[2]["candidate_dictionary_entries"], 2);
    assert_eq!(alternatives[2]["selected_kv_slice_parameters"], 2);
}

#[test]
fn test_absolute_tolerance_does_not_hide_slow_point_regressions() {
    let a = fixture_route(2);
    let b = fixture_route(4);
    let mut cells = vec![
        Cell {
            coordinate: [0, 0, 0],
            reference_wall_ms: 0.01,
            choices: vec![(a, 1.0), (b, 1.2)],
        },
        Cell {
            coordinate: [1, 0, 0],
            reference_wall_ms: 1.0,
            choices: vec![(a, 1.2), (b, 1.0)],
        },
    ];
    let (routes, _, regret, extra) = compress(&cells, 0.05, 0.005);
    assert_eq!(routes, [b]);
    assert!(regret > 0.05 && extra < 0.005);
    let frontier = compression_frontier(&cells, 0.05);
    let alternative = frontier
        .iter()
        .find(|record| record["absolute_loss_bound_ms"] == 0.005)
        .unwrap();
    let slow = alternative["reference_time_buckets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|record| record["min_reference_wall_ms"] == 0.2)
        .unwrap();
    assert_eq!(slow["points"], 1);
    assert_eq!(slow["max_observed_regret"], 0.0);
    cells[0].reference_wall_ms = 1.0;
    let (routes, _, regret, _) = compress(&cells, 0.05, 0.005);
    assert_eq!(routes.len(), 2);
    assert_eq!(regret, 0.0);
}

#[test]
fn test_candidate_bracket_noise_cannot_enter_table() {
    let command = Command {
        workload: fixture_workload(),
        candidate: variants(fixture_workload())[0],
        inner: 8,
        repeats: 1,
    };
    let result = json!({"samples":[{"before_ms":1.,"after_ms":1.,"candidate_ms":1.,"candidate_first_ms":0.5,"candidate_last_ms":1.5}],"max_abs_error":0.,"mean_abs_error":0.,"cpu_reference_values":1});
    assert!(!summarize(&command, result, 0.10).unwrap().stable);
}
#[test]
fn test_non_decode_slice_lookup_matches_calibrated_geometric_rounding() {
    for end in [513, 1024, 2048, 4096, 8192, 16384, 32768] {
        for queries in [2, 3, 4, 5, 8, 16, 32, 128, 256] {
            for requests in [1, 2, 4, 8] {
                for ratio in [6, 8, 12] {
                    for dim in [128, 256] {
                        let w = Workload {
                            history: end - queries,
                            queries,
                            requests,
                            ratio,
                            dim,
                            ..fixture_workload()
                        };
                        w.validate().unwrap();
                        let c = proposed_candidate(
                            w,
                            Policy {
                                short_q_token_heads: 32,
                                q_token_head_values: 16384,
                                loader_tokens: 16,
                                ..RESOURCE_POLICY
                            },
                        )
                        .unwrap();
                        c.validate(w).unwrap();
                        let groups = (c.threads() / 32) as f64;
                        let ideal =
                            ((end as f64).sqrt() * (requests as f64).powf(0.25) * groups.sqrt()).clamp(64.0, 512.0);
                        let expected = [64_u32, 128, 256, 512]
                            .into_iter()
                            .min_by(|&a, &b| {
                                let da = (a as f64 / ideal).ln().abs();
                                let db = (b as f64 / ideal).ln().abs();
                                if (da - db).abs() < 1e-12 {
                                    a.cmp(&b)
                                } else {
                                    da.total_cmp(&db)
                                }
                            })
                            .unwrap();
                        assert_eq!(c.slice, expected, "{w:?}");
                    }
                }
            }
        }
    }
}

#[test]
fn test_policy_report_does_not_substitute_a_measured_winner() {
    let w = Workload {
        history: 1024,
        ..fixture_workload()
    };
    let selected = proposed_candidate(
        w,
        Policy {
            short_q_token_heads: 32,
            q_token_head_values: 16384,
            loader_tokens: 16,
            ..RESOURCE_POLICY
        },
    )
    .unwrap();
    let winner = Candidate {
        layout: Layout::QHeadsPerToken,
        ..selected
    };
    let workloads = BTreeSet::from([w]);
    let record = |candidate, time| {
        Summary {
            workload: w,
            candidate,
            median_relative_time: time,
            median_wall_ms: time,
            max_baseline_drift: 0.0,
            relative_sample_range: 0.0,
            stable: true,
        }
    };
    let mut summaries = BTreeMap::from([(w, vec![record(winner, 1.0), record(selected, 1.2)])]);
    let mut recorded = HashMap::from([
        ((w, winner), "measured".to_owned()),
        ((w, selected), "measured".to_owned()),
    ]);
    let report = evaluate_policy(
        &workloads,
        &summaries,
        &recorded,
        Policy {
            short_q_token_heads: 32,
            q_token_head_values: 16384,
            loader_tokens: 16,
            ..RESOURCE_POLICY
        },
    );
    assert_eq!(
        report["points"][0]["candidate"],
        serde_json::to_value(selected).unwrap()
    );
    assert!((report["points"][0]["observed_regret"].as_f64().unwrap() - 0.2).abs() < 1e-12);
    summaries.get_mut(&w).unwrap().pop();
    recorded.remove(&(w, selected));
    let report = evaluate_policy(
        &workloads,
        &summaries,
        &recorded,
        Policy {
            short_q_token_heads: 32,
            q_token_head_values: 16384,
            loader_tokens: 16,
            ..RESOURCE_POLICY
        },
    );
    assert_eq!(report["selected_coverage"]["missing"], 1);
    assert!(report["points"][0]["observed_regret"].is_null());
    assert_eq!(report["reference_time_buckets"][0]["verified_points"], 0);
    assert_eq!(report["reference_time_buckets"][0]["unverified_points"], 1);
    assert_eq!(
        report["points"][0]["candidate"],
        serde_json::to_value(selected).unwrap()
    );
}

#[test]
fn test_policy_fit_prioritizes_expensive_complete_routes() {
    let expensive = Workload {
        history: 32768,
        dim: 128,
        ..fixture_workload()
    };
    let wide = Workload {
        history: 8192,
        queries: 8,
        ratio: 12,
        dim: 128,
        ..fixture_workload()
    };
    let packed = Workload {
        history: 8192,
        queries: 8,
        dim: 128,
        ..fixture_workload()
    };
    let workloads = BTreeSet::from([expensive, wide, packed]);
    let mut summaries = BTreeMap::new();
    for &w in &workloads {
        let mut candidates = BTreeSet::new();
        for q_token_head_values in [8192, 16384, 32768] {
            for loader_tokens in [8, 16, 32] {
                let geometry = Policy {
                    short_q_token_heads: 32,
                    q_token_head_values,
                    loader_tokens,
                    ..RESOURCE_POLICY
                };
                for policy in parallel_policies(geometry) {
                    candidates.insert(proposed_candidate(w, policy).unwrap());
                }
            }
        }
        let mut records: Vec<_> = candidates
            .into_iter()
            .map(|candidate| {
                let best = if w == expensive {
                    candidate.layout == Layout::QHeadsPerToken
                } else if w == wide {
                    candidate.tile == 8
                } else {
                    candidate.layout == Layout::QTokensAndHeads
                };
                let relative = if best { 1.0 } else { 1.2 };
                Summary {
                    workload: w,
                    candidate,
                    median_relative_time: relative,
                    median_wall_ms: relative * if w == expensive { 2.0 } else { 1.0 },
                    max_baseline_drift: 0.0,
                    relative_sample_range: 0.0,
                    stable: true,
                }
            })
            .collect();
        records.sort_by(|a, b| a.median_relative_time.total_cmp(&b.median_relative_time));
        summaries.insert(w, records);
    }
    let (policy, evidence) = fit_policy(&workloads, &summaries, 40).unwrap();
    assert_eq!(evidence["geometry"]["training_points"], 3);
    assert_eq!(
        proposed_candidate(expensive, policy).unwrap().layout,
        Layout::QHeadsPerToken
    );
    assert_eq!(proposed_candidate(wide, policy).unwrap().tile, 8);
    assert_eq!(
        proposed_candidate(packed, policy).unwrap().layout,
        Layout::QTokensAndHeads
    );
    summaries
        .get_mut(&wide)
        .unwrap()
        .retain(|record| record.candidate.tile != 8);
    let (_, evidence) = fit_policy(&workloads, &summaries, 40).unwrap();
    assert_eq!(evidence["geometry"]["reference_points"], 3);
    assert_eq!(evidence["geometry"]["training_points"], 2);
    assert_eq!(evidence["geometry"]["excluded_points"], 1);
    assert!(fit_policy(&BTreeSet::from([wide]), &summaries, 40).is_none());
    assert!(fit_policy(&BTreeSet::new(), &summaries, 40).is_none());
    for records in summaries.values_mut() {
        for record in records {
            record.median_wall_ms = 0.1;
        }
    }
    assert!(fit_policy(&workloads, &summaries, 40).is_none());
}

fn fixture_batch_candidate(
    dim: u32,
    ratio: u32,
    kv_heads: u32,
    request_shapes: &[(u32, u32)],
    policy: Policy,
) -> Candidate {
    let batch = BatchShape {
        num_requests: request_shapes.len() as u32,
        max_q_tokens: request_shapes.iter().map(|&(_, q)| q).max().unwrap(),
        max_visible_kv_tokens: request_shapes.iter().map(|&(h, q)| h + q).max().unwrap(),
        num_visible_kv_tokens: request_shapes.iter().map(|&(h, q)| (h + q) as u128).sum(),
    };
    proposed_batch_candidate(
        dim,
        ratio,
        kv_heads,
        batch,
        |tile| request_shapes.iter().map(|&(_, q)| q.div_ceil(tile) as u128).sum(),
        |slice| {
            request_shapes
                .iter()
                .map(|&(h, q)| (h + q).div_ceil(slice) as u128)
                .sum()
        },
        policy,
    )
    .unwrap()
}

#[test]
fn test_batch_decode_uses_actual_partitioned_grid() {
    let policy = Policy {
        decode: DecodePolicy::Adaptive {
            min_grid_groups: 160,
            target_groups: 1280,
        },
        ..RESOURCE_POLICY
    };
    let mut requests = [(0, 1); 8];
    requests[0] = (7935, 1);
    // Four groups times (31 long splits + seven short splits) give 152, below 160.
    let single = fixture_batch_candidate(256, 12, 2, &requests, policy);
    assert_eq!(
        (single.layout, single.tile, single.heads, single.slice),
        (Layout::SlicedKv, 1, 8, 256)
    );
    let uniform = Workload {
        history: 7935,
        queries: 1,
        requests: 8,
        ratio: 12,
        dim: 256,
        kv_heads: 2,
        page: 16,
    };
    assert_eq!(
        proposed_candidate(uniform, policy).unwrap().layout,
        Layout::QTokensAndHeads
    );
    requests[0] = (8192, 1);
    // Four groups times (33 long splits + seven short splits) reach 160.
    let joint = fixture_batch_candidate(256, 12, 2, &requests, policy);
    assert_eq!(
        (joint.layout, joint.tile, joint.heads, joint.slice),
        (Layout::QTokensAndHeads, 1, 12, 256)
    );
    requests.reverse();
    assert_eq!(fixture_batch_candidate(256, 12, 2, &requests, policy), joint);
}

#[test]
fn test_batch_geometry_uses_actual_query_range_count() {
    let policy = Policy {
        max_head_base_groups: 64,
        ..RESOURCE_POLICY
    };
    let requests = [(1024, 33), (1024, 1)];
    // T=16 has four actual Q ranges. Repeating the longest request would count six.
    let actual = fixture_batch_candidate(128, 6, 1, &requests, policy);
    assert_eq!((actual.layout, actual.tile), (Layout::QHeadsPerToken, 16));
    let uniform = Workload {
        history: 1024,
        queries: 33,
        requests: 2,
        ratio: 6,
        dim: 128,
        kv_heads: 1,
        page: 16,
    };
    let repeated = proposed_candidate(uniform, policy).unwrap();
    assert_eq!((repeated.layout, repeated.tile), (Layout::QTokensAndHeads, 16));
    assert_eq!(
        fixture_batch_candidate(128, 6, 1, &[(1024, 1), (1024, 33)], policy),
        actual
    );
}

#[test]
fn test_policy_capacity_uses_shared_replay_buckets_per_geometry() {
    let workloads = BTreeSet::from([
        Workload {
            history: 16384,
            queries: 1,
            requests: 1,
            ratio: 6,
            dim: 128,
            kv_heads: 2,
            page: 16,
        },
        Workload {
            history: 1024,
            queries: 8,
            requests: 1,
            ratio: 6,
            dim: 128,
            kv_heads: 2,
            page: 16,
        },
    ]);
    let capacity = policy_capacity(&workloads, RESOURCE_POLICY);
    let envelope = &capacity["uniform_replay_envelopes"][0];
    assert_eq!(envelope["max_map_task_templates"], 33);
    assert_eq!(envelope["partial_state_group_capacity"], 160);
    let wide = envelope["geometries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|g| g["geometry"]["tile"] == 8 && g["geometry"]["direct"] == false)
        .unwrap();
    assert_eq!(wide["max_active_map_task_templates"], 17);
    assert_eq!(wide["max_recorded_map_task_templates"], 20);
    assert_eq!(wide["max_replay_reserved_partial_state_groups"], 160);
    assert!(160 < 33 * 8);
    for direct in envelope["geometries"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|g| g["geometry"]["direct"] == true)
    {
        assert_eq!(direct["max_replay_reserved_partial_state_groups"], 0);
    }
}

#[test]
fn test_policy_capacity_accounts_for_direct_q_range_padding() {
    let workloads = BTreeSet::from([
        Workload {
            history: 0,
            queries: 2,
            requests: 5,
            ratio: 6,
            dim: 128,
            kv_heads: 2,
            page: 16,
        },
        Workload {
            history: 0,
            queries: 3,
            requests: 4,
            ratio: 6,
            dim: 128,
            kv_heads: 2,
            page: 16,
        },
    ]);
    let capacity = policy_capacity(&workloads, RESOURCE_POLICY);
    let envelope = &capacity["uniform_replay_envelopes"][0];
    assert_eq!(envelope["analyzed_uniform_input_bounds"]["max_tokens"], 12);
    assert_eq!(envelope["max_map_task_templates"], 6);
    assert_eq!(envelope["partial_state_group_capacity"], 0);
    let two_tokens = envelope["geometries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|g| g["geometry"]["tile"] == 2)
        .unwrap();
    assert_eq!(two_tokens["max_active_map_task_templates"], 5);
    assert_eq!(two_tokens["max_recorded_map_task_templates"], 6);
}

#[test]
fn test_policy_capacity_includes_pre_transition_peaks() {
    let workload = Workload {
        history: 32767,
        queries: 1,
        requests: 8,
        ratio: 6,
        dim: 128,
        kv_heads: 2,
        page: 16,
    };
    let policy = Policy {
        decode: DecodePolicy::Adaptive {
            min_grid_groups: 160,
            target_groups: 1280,
        },
        ..RESOURCE_POLICY
    };
    let capacity = policy_capacity(&BTreeSet::from([workload]), policy);
    let geometry = capacity["profiles"][0]["geometries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|record| record["geometry"]["layout"] == "single")
        .unwrap();
    assert_eq!(geometry["max_splits_per_q_range"], 40);
    assert_eq!(geometry["max_active_map_task_templates"], 320);
    assert_eq!(geometry["map_peak_visible_kv_tokens"], 10239);
    let at_maximum = proposed_candidate(workload, policy).unwrap();
    assert_eq!((workload.history + workload.queries).div_ceil(at_maximum.slice), 32);
}

#[test]
fn test_policy_context_boundaries_cover_continuous_uniform_capacity() {
    let policies = [
        RESOURCE_POLICY,
        Policy {
            max_head_base_groups: 16,
            max_floor_base_groups: 32,
            target_split_groups: 1024,
            upper_slice_shift: 1,
            decode: DecodePolicy::Adaptive {
                min_grid_groups: 160,
                target_groups: 1280,
            },
            ..RESOURCE_POLICY
        },
        Policy {
            short_q_token_heads: 16,
            q_token_head_values: 8192,
            loader_tokens: 8,
            max_head_base_groups: 0,
            max_floor_base_groups: 32,
            target_split_groups: 512,
            upper_slice_shift: 2,
            decode: DecodePolicy::Adaptive {
                min_grid_groups: 80,
                target_groups: 640,
            },
        },
    ];
    for (queries, dim, ratio, requests, kv_heads) in [
        (1, 128, 6, 8, 2),
        (1, 256, 12, 1, 2),
        (2, 128, 6, 1, 8),
        (4, 128, 12, 1, 2),
        (16, 256, 6, 8, 8),
        (64, 128, 1, 2, 4),
        (128, 256, 12, 1, 2),
        (256, 256, 6, 1, 8),
    ] {
        let workload = Workload {
            history: 32768 - queries,
            queries,
            requests,
            ratio,
            dim,
            kv_heads,
            page: 16,
        };
        workload.validate().unwrap();
        for policy in policies {
            let capacity = policy_capacity(&BTreeSet::from([workload]), policy);
            let profile = &capacity["profiles"][0];
            assert!(profile["unavailable_boundary_points"].as_array().unwrap().is_empty());
            let mut expected = BTreeMap::new();
            for end in queries..=32768 {
                let current = Workload {
                    history: end - queries,
                    ..workload
                };
                let candidate = proposed_candidate(current, policy).unwrap();
                let key = (
                    candidate.layout,
                    candidate.tile,
                    candidate.heads,
                    candidate.stage,
                    candidate.direct,
                );
                let value = expected.entry(key).or_insert((0_u64, 0_u32));
                value.0 = value.0.max(candidate.scratch_bytes(current));
                value.1 = value.1.max(end.div_ceil(candidate.slice));
            }
            let records = profile["geometries"].as_array().unwrap();
            assert_eq!(records.len(), expected.len());
            for record in records {
                let geometry = &record["geometry"];
                let key = (
                    serde_json::from_value(geometry["layout"].clone()).unwrap(),
                    geometry["tile"].as_u64().unwrap() as u32,
                    geometry["heads"].as_u64().unwrap() as u32,
                    geometry["stage"].as_u64().unwrap() as u32,
                    geometry["direct"].as_bool().unwrap(),
                );
                let (bytes, splits) = expected[&key];
                let partial_groups = record["max_active_reserved_partial_state_groups"].as_u64().unwrap();
                assert_eq!(
                    partial_groups * ratio as u64 * kv_heads as u64 * (dim as u64 * 2 + 8),
                    bytes
                );
                assert_eq!(record["max_splits_per_q_range"], splits);
                let end = record["map_peak_visible_kv_tokens"].as_u64().unwrap() as u32;
                let witness = Workload {
                    history: end - queries,
                    ..workload
                };
                assert_eq!(
                    proposed_candidate(witness, policy).unwrap().scratch_bytes(witness),
                    bytes
                );
            }
        }
    }
}

#[test]
fn test_policy_fit_requires_tunable_route_differences() {
    let workloads = BTreeSet::from([1024, 8192].map(|history| {
        Workload {
            history,
            queries: 1,
            ..fixture_workload()
        }
    }));
    let summaries = workloads
        .iter()
        .map(|&workload| {
            let record = Summary {
                workload,
                candidate: proposed_candidate(workload, RESOURCE_POLICY).unwrap(),
                median_relative_time: 1.0,
                median_wall_ms: 0.4,
                max_baseline_drift: 0.0,
                relative_sample_range: 0.0,
                stable: true,
            };
            (workload, vec![record])
        })
        .collect();
    assert!(fit_policy(&workloads, &summaries, 40).is_none());
}

#[test]
fn test_policy_fit_retains_constant_geometry_when_parallelism_varies() {
    let workload = Workload {
        history: 8192,
        ..fixture_workload()
    };
    let workloads = BTreeSet::from([workload]);
    let candidates: BTreeSet<_> = parallel_policies(RESOURCE_POLICY)
        .into_iter()
        .map(|policy| proposed_candidate(workload, policy).unwrap())
        .collect();
    let mut records: Vec<_> = candidates
        .into_iter()
        .map(|candidate| {
            let relative = if candidate.layout == Layout::QHeadsPerToken {
                1.0
            } else {
                1.1
            };
            Summary {
                workload,
                candidate,
                median_relative_time: relative,
                median_wall_ms: relative,
                max_baseline_drift: 0.0,
                relative_sample_range: 0.0,
                stable: true,
            }
        })
        .collect();
    records.sort_by(|a, b| a.median_relative_time.total_cmp(&b.median_relative_time));
    let summaries = BTreeMap::from([(workload, records)]);
    let (policy, evidence) = fit_policy(&workloads, &summaries, 40).unwrap();
    assert_eq!(evidence["geometry"]["varied_training_points"], 0);
    assert_eq!(evidence["parallelism"]["varied_training_points"], 1);
    assert_eq!(policy.q_token_head_values, RESOURCE_POLICY.q_token_head_values);
    assert_eq!(policy.loader_tokens, RESOURCE_POLICY.loader_tokens);
    assert_eq!(
        proposed_candidate(workload, policy).unwrap().layout,
        Layout::QHeadsPerToken
    );
}

#[test]
fn test_parallel_fit_does_not_trade_medium_cost_regression_for_slow_cost_gain() {
    let baseline = RESOURCE_POLICY;
    let proposal = Policy {
        short_q_token_heads: 16,
        ..baseline
    };
    let slow = Workload {
        queries: 8,
        dim: 128,
        ..fixture_workload()
    };
    let medium = Workload { ratio: 12, ..slow };
    let workloads = BTreeSet::from([slow, medium]);
    let mut summaries = BTreeMap::new();
    for (w, cost, baseline_time, proposal_time) in [(slow, 2.0, 1.1, 1.0), (medium, 0.4, 1.04, 1.2)] {
        let reference = Candidate {
            layout: Layout::QTokensAndHeads,
            tile: 8,
            heads: w.ratio,
            stage: 16,
            slice: w.history + w.queries,
            direct: true,
        };
        let mut records: Vec<_> = [
            (reference, 1.0),
            (proposed_candidate(w, baseline).unwrap(), baseline_time),
            (proposed_candidate(w, proposal).unwrap(), proposal_time),
        ]
        .into_iter()
        .map(|(candidate, relative)| {
            Summary {
                workload: w,
                candidate,
                median_relative_time: relative,
                median_wall_ms: relative * cost,
                max_baseline_drift: 0.0,
                relative_sample_range: 0.0,
                stable: true,
            }
        })
        .collect();
        records.sort_by(|a, b| a.median_relative_time.total_cmp(&b.median_relative_time));
        summaries.insert(w, records);
    }
    let (selected, evidence, _) =
        fit_candidates(&workloads, &summaries, &[baseline, proposal], Some(baseline)).unwrap();
    assert_eq!(selected.short_q_token_heads, baseline.short_q_token_heads);
    assert_eq!(evidence["training_points"], 2);
    assert_eq!(evidence["frontier"][1]["within_baseline_bounds"], false);
}

#[test]
fn test_single_query_policy_preserves_kv_reuse_and_actual_base_grid() {
    let small = Workload {
        history: 8192,
        queries: 1,
        requests: 1,
        ratio: 12,
        dim: 256,
        kv_heads: 4,
        page: 16,
    };
    let single = proposed_candidate(small, RESOURCE_POLICY).unwrap();
    assert_eq!(single.layout, Layout::SlicedKv);
    assert_eq!(
        (single.tile, single.heads, single.stage, single.slice),
        (1, 8, 256, 512)
    );
    let by_requests = proposed_candidate(Workload { requests: 2, ..small }, RESOURCE_POLICY).unwrap();
    let by_heads = proposed_candidate(Workload { kv_heads: 8, ..small }, RESOURCE_POLICY).unwrap();
    assert_eq!(by_requests, by_heads);
    assert_eq!(by_heads.layout, Layout::QTokensAndHeads);
    assert_eq!(
        (by_heads.tile, by_heads.heads, by_heads.stage, by_heads.slice),
        (1, 12, 16, 512)
    );
    for w in [
        Workload {
            dim: 128,
            requests: 8,
            ..small
        },
        Workload {
            ratio: 8,
            requests: 8,
            ..small
        },
    ] {
        assert_eq!(proposed_candidate(w, RESOURCE_POLICY).unwrap().layout, Layout::SlicedKv);
    }
    let short = proposed_candidate(Workload { history: 511, ..small }, RESOURCE_POLICY).unwrap();
    assert_eq!(short.layout, Layout::SlicedKv);
    assert!(!short.direct);
    assert_eq!(short.slice, 512);
}

#[test]
fn test_single_query_policy_uses_partitioned_grid_and_shared_slice_budget() {
    let policy = Policy {
        decode: DecodePolicy::Adaptive {
            min_grid_groups: 80,
            target_groups: 640,
        },
        ..RESOURCE_POLICY
    };
    let small = Workload {
        history: 1024,
        queries: 1,
        requests: 1,
        ratio: 12,
        dim: 256,
        kv_heads: 2,
        page: 16,
    };
    let single = proposed_candidate(small, policy).unwrap();
    let larger = Workload { history: 8192, ..small };
    let matrix = proposed_candidate(larger, policy).unwrap();
    // The same unsplit base grid needs different geometry after context partitioning.
    assert_eq!((single.layout, single.slice), (Layout::SlicedKv, 256));
    assert_eq!((matrix.layout, matrix.slice), (Layout::QTokensAndHeads, 256));
    assert_eq!(
        proposed_candidate(larger, RESOURCE_POLICY).unwrap().layout,
        Layout::SlicedKv
    );
    let requests = proposed_candidate(Workload { requests: 2, ..larger }, policy).unwrap();
    let heads = proposed_candidate(Workload { kv_heads: 4, ..larger }, policy).unwrap();
    assert_eq!(requests, heads);
    let wide = Workload {
        history: 32768,
        requests: 8,
        kv_heads: 8,
        dim: 128,
        ..small
    };
    let single = proposed_candidate(wide, policy).unwrap();
    assert_eq!((single.layout, single.slice), (Layout::SlicedKv, 1024));
    let short = proposed_candidate(Workload { history: 511, ..small }, policy).unwrap();
    assert_eq!(
        (short.layout, short.slice, short.direct),
        (Layout::SlicedKv, 256, false)
    );
    for w in [
        small,
        larger,
        wide,
        Workload {
            history: 12289,
            ..larger
        },
    ] {
        proposed_candidate(w, policy).unwrap().validate(w).unwrap();
    }
}

#[test]
fn test_single_query_measurements_can_identify_decode_coefficients() {
    let workloads = BTreeSet::from([8192, 16384].map(|history| {
        Workload {
            history,
            queries: 1,
            requests: 1,
            ratio: 12,
            dim: 256,
            kv_heads: 2,
            page: 16,
        }
    }));
    let summaries = workloads
        .iter()
        .map(|&workload| {
            let records = [
                (
                    Candidate {
                        layout: Layout::QTokensAndHeads,
                        tile: 1,
                        heads: 12,
                        stage: 16,
                        slice: 256,
                        direct: false,
                    },
                    1.0,
                ),
                (
                    Candidate {
                        layout: Layout::SlicedKv,
                        tile: 1,
                        heads: 8,
                        stage: 256,
                        slice: 256,
                        direct: false,
                    },
                    1.1,
                ),
                (
                    Candidate {
                        layout: Layout::SlicedKv,
                        tile: 1,
                        heads: 8,
                        stage: 256,
                        slice: 512,
                        direct: false,
                    },
                    1.2,
                ),
            ]
            .map(|(candidate, relative)| {
                Summary {
                    workload,
                    candidate,
                    median_relative_time: relative,
                    median_wall_ms: relative * 0.4,
                    max_baseline_drift: 0.0,
                    relative_sample_range: 0.0,
                    stable: true,
                }
            })
            .to_vec();
            (workload, records)
        })
        .collect();
    let (policy, evidence) = fit_policy(&workloads, &summaries, 40).unwrap();
    assert_eq!(evidence["geometry"]["varied_training_points"], 0);
    assert_eq!(evidence["parallelism"]["varied_training_points"], 0);
    assert_eq!(evidence["decode"]["training_points"], 2);
    assert_eq!(evidence["decode"]["varied_training_points"], 2);
    assert!(matches!(
        policy.decode,
        DecodePolicy::Adaptive {
            min_grid_groups: 80,
            target_groups: 640
        }
    ));
}

#[test]
fn test_unavailable_fit_replaces_previous_export() {
    let out = std::env::temp_dir().join(format!("gqa-policy-export-{}", std::process::id()));
    fs::create_dir_all(&out).unwrap();
    export_policy(&out, "policy", Some(RESOURCE_POLICY), &json!({}), "test/40").unwrap();
    assert!(read_json::<Value>(&out.join("policy.json")).unwrap()["policy"].is_object());
    export_policy(&out, "policy", None, &json!({}), "test/40").unwrap();
    let data = read_json::<Value>(&out.join("policy.json")).unwrap();
    assert!(data["policy"].is_null());
    assert!(data["thresholds"].is_null());
    assert!(
        fs::read_to_string(out.join("policy.rs"))
            .unwrap()
            .contains("const POLICY: Option<Policy> = None;")
    );
    fs::remove_dir_all(out).unwrap();
}

#[test]
fn test_frozen_policy_input_proves_criterion_and_coefficient_domains() {
    let out = std::env::temp_dir().join(format!("gqa-policy-input-{}", std::process::id()));
    fs::create_dir_all(&out).unwrap();
    export_policy(&out, "policy", Some(RESOURCE_POLICY), &json!({}), "test/40").unwrap();
    let path = out.join("policy.json");
    let (policy, source) = read_frozen_policy(&path, "test/40").unwrap();
    assert_eq!(policy.q_token_head_values, RESOURCE_POLICY.q_token_head_values);
    assert_eq!(source["document"]["schema"], 4);
    assert!(read_frozen_policy(&path, "other/40").is_err());
    // A frozen policy must not silently acquire the new short-context routing.
    let mut legacy: Value = read_json(&path).unwrap();
    legacy["schema"] = json!(3);
    legacy["decode_criteria"]["criterion"] = json!("shared-partitioned-grid-v1");
    assert!(decode_frozen_policy(&legacy, "test/40").is_err());
    let adaptive = Policy {
        decode: DecodePolicy::Adaptive {
            min_grid_groups: 80,
            target_groups: 640,
        },
        ..RESOURCE_POLICY
    };
    export_policy(&out, "policy", Some(adaptive), &json!({}), "test/40").unwrap();
    assert!(matches!(
        read_frozen_policy(&path, "test/40").unwrap().0.decode,
        DecodePolicy::Adaptive { .. }
    ));
    let mut invalid: Value = read_json(&path).unwrap();
    invalid["policy"]["decode"]["target_groups"] = json!(0);
    assert!(decode_frozen_policy(&invalid, "test/40").is_err());
    invalid["policy"]["decode"]["target_groups"] = json!(640);
    invalid["schema"] = json!(2);
    invalid["decode_criteria"] = legacy["decode_criteria"].clone();
    assert!(decode_frozen_policy(&invalid, "test/40").is_err());
    let mut invalid: Value = read_json(&path).unwrap();
    invalid["policy"]["upper_slice_shift"] = json!(128);
    write_json(&path, &invalid).unwrap();
    assert!(read_frozen_policy(&path, "test/40").is_err());
    export_policy(&out, "policy", None, &json!({}), "test/40").unwrap();
    assert!(read_frozen_policy(&path, "test/40").is_err());
    fs::remove_dir_all(out).unwrap();
}

fn fixture_workload() -> Workload {
    Workload {
        history: 0,
        queries: 4,
        requests: 1,
        ratio: 6,
        dim: 256,
        kv_heads: 4,
        page: 16,
    }
}
fn fixture_route(tile: u32) -> Route {
    Route {
        layout: Layout::QHeadsPerToken,
        tile,
        heads: 6,
        stage: 16,
        slice: None,
        direct: true,
    }
}
