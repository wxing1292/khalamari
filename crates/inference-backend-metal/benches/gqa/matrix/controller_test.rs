use super::*;
#[test]
fn test_partitions_and_legacy_keys() {
    let w = fixture_workload();
    let candidates = variants(w);
    assert_eq!(
        candidates.iter().copied().collect::<HashSet<_>>().len(),
        candidates.len()
    );
    assert!(candidates.iter().all(|c| c.slice == 4));
    let c = candidates[0];
    assert_eq!(
        encoded(&Command {
            workload: w,
            candidate: c,
            inner: 8,
            repeats: 5
        }),
        "{\"candidate\":{\"direct\":false,\"heads\":6,\"layout\":\"single\",\"slice\":4,\"stage\":256,\"tile\":1},\"\
         inner\":8,\"repeats\":5,\"workload\":{\"dim\":256,\"history\":0,\"kv_heads\":4,\"page\":16,\"queries\":4,\"\
         ratio\":6,\"requests\":1}}"
    );
    assert_eq!(
        Candidate {
            layout: Layout::QHeadsPerToken,
            tile: 4,
            heads: 6,
            ..c
        }
        .threads(),
        128
    );
    assert_eq!(
        Candidate {
            layout: Layout::QTokensAndHeads,
            tile: 4,
            heads: 6,
            ..c
        }
        .threads(),
        96
    );
}
#[test]
fn test_plan_workloads_axis_order() {
    let args = Args::parse_from([
        "bench",
        "--history",
        "32768",
        "64",
        "--queries",
        "4",
        "2",
        "--requests",
        "1",
        "2",
        "--ratios",
        "12",
        "--dims",
        "256",
        "128",
        "--kv-heads",
        "4",
    ]);
    let plan = Plan::new(&args);
    plan.validate().unwrap();
    let workloads = plan.workloads();
    assert_eq!(workloads.len(), 16);
    assert_eq!(workloads.iter().copied().collect::<HashSet<_>>().len(), 16);
    let prefix: Vec<_> = workloads[..5]
        .iter()
        .map(|w| (w.history, w.queries, w.requests, w.dim))
        .collect();
    assert_eq!(
        prefix,
        [
            (32768, 4, 1, 256),
            (32768, 4, 1, 128),
            (32768, 4, 2, 256),
            (32768, 4, 2, 128),
            (64, 4, 1, 256)
        ]
    );
    assert!(workloads[..8].iter().all(|w| w.queries == 4));
    assert!(workloads[8..].iter().all(|w| w.queries == 2));
}

#[test]
fn test_frozen_policy_plan_keeps_comparisons_and_resume_keys() {
    let args = Args::parse_from(["bench", "--mode", "policy"]);
    let mut plan = Plan::new(&args);
    assert!(plan.validate().is_err());
    let policy = Policy {
        short_q_token_heads: 32,
        q_token_head_values: 16384,
        loader_tokens: 16,
        max_head_base_groups: 16,
        max_floor_base_groups: 32,
        target_split_groups: 1024,
        upper_slice_shift: 1,
        decode: DecodePolicy::Fixed,
    };
    for (dim, ratio) in [(128, 6), (128, 12), (256, 6), (256, 12)] {
        for queries in [1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256] {
            let w = Workload {
                history: 8192,
                queries,
                dim,
                ratio,
                ..fixture_workload()
            };
            let candidates = policy_candidates(w, policy);
            let chosen = proposed_candidate(w, policy).unwrap();
            assert!(candidates.contains(&chosen));
            assert!(candidates.len() <= 16);
            for slice in [(chosen.slice / 2).max(1), (chosen.slice * 2).min(w.history + queries)] {
                assert!(candidates.contains(&Candidate {
                    layout: Layout::SlicedKv,
                    tile: 1,
                    heads: ratio.min(8),
                    stage: dim,
                    slice,
                    direct: false,
                }));
                assert!(candidates.contains(&Candidate {
                    layout: Layout::QTokensAndHeads,
                    tile: if chosen.layout == Layout::SlicedKv {
                        queries.min(8).next_power_of_two()
                    } else {
                        chosen.tile
                    },
                    heads: ratio,
                    stage: 16,
                    slice,
                    direct: slice == w.history + queries,
                }));
            }
            for &c in &candidates {
                c.validate(w).unwrap();
            }
            let baseline = candidates
                .iter()
                .find(|c| c.layout == Layout::SlicedKv && c.slice == w.history + queries)
                .unwrap();
            assert!(variants(w).contains(baseline));
            let command = Command {
                workload: w,
                candidate: *baseline,
                inner: 8,
                repeats: 5,
            };
            let round_trip: Command = serde_json::from_str(&encoded(&command)).unwrap();
            assert_eq!(encoded(&round_trip), encoded(&command));
        }
    }
    // The plan owns a frozen source document. A saved coefficient edit must be rejected.
    let document = json!({"schema":4,"device":"test/40",
        "operator":{"io_dtype":"bfloat16","cache_dtype":"fp8_e4m3","visibility":"uniform per-request causal history; no sliding window"},
        "strategies":STRATEGIES,"decode_criteria":{"criterion":"shared-partitioned-grid-v2"},"policy":policy});
    plan.frozen_policy = Some((policy, json!({"path":"/tmp/frozen.json","document":document})));
    plan.validate().unwrap();
    plan.frozen_policy.as_mut().unwrap().0.loader_tokens = 32;
    assert!(plan.validate().is_err());
}

#[test]
fn test_changed_measurement_identity_is_rejected() {
    let out = std::env::temp_dir().join(format!("gqa-identity-{}", std::process::id()));
    fs::create_dir_all(&out).unwrap();
    let identity = json!({"measurement_sha256":"source-a","device":{"chip":"Apple M3 Max","gpu_cores":"40","memory":"48 GB"},"inner":8,"repeats":5});
    check_identity(&out, &identity).unwrap();
    check_identity(&out, &identity).unwrap();
    let mut changed = identity.clone();
    changed["inner"] = json!(16);
    assert!(matches!(check_identity(&out, &changed), Err(Error::InvalidArgument(_))));
    changed = identity;
    changed["measurement_sha256"] = json!("source-b");
    assert!(matches!(check_identity(&out, &changed), Err(Error::InvalidArgument(_))));
    fs::remove_dir_all(out).unwrap();
}

#[test]
fn test_sqlite_commit_survives_reopen_and_preserves_key() {
    let path = std::env::temp_dir().join(format!("gqa-matrix-{}.sqlite3", std::process::id()));
    let command = Command {
        workload: fixture_workload(),
        candidate: variants(fixture_workload())[0],
        inner: 8,
        repeats: 5,
    };
    {
        let mut db = Database::open(&path).unwrap();
        db.insert(&command, &json!({"status":"failed","reason":"quoted ' text"}))
            .unwrap();
    }
    let mut n = 0;
    records(&path, |r| {
        assert_eq!(r["key"], encoded(&command));
        assert_eq!(r["result"]["reason"], "quoted ' text");
        n += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(n, 1);
    fs::remove_file(path).unwrap();
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
