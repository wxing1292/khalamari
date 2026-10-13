use super::*;

fn args(component: &str) -> Args {
    Args::parse_from([
        "component-bench",
        "--component",
        component,
        "--out",
        "/private/tmp/component-bench-test",
    ])
}

#[test]
fn test_matrix_semantics_and_equivalent_work() {
    let dense = args("dense-mlp");
    let cases = dense.cases().unwrap();
    assert!(cases.contains(&Case {
        kind: Kind::Append,
        requests: 8,
        context: 65536,
        queries: 256
    }));
    assert!(cases.contains(&Case {
        kind: Kind::FullPrefill,
        requests: 4,
        context: 0,
        queries: 16384
    }));
    let first = Case {
        kind: Kind::Append,
        requests: 1,
        context: 0,
        queries: 256,
    };
    let second = Case {
        requests: 4,
        context: 65536,
        queries: 64,
        ..first
    };
    assert_eq!(
        first.measurement(Component::DenseMlp, false),
        second.measurement(Component::DenseMlp, false)
    );
    assert_eq!(
        first.measurement(Component::Moe, false),
        second.measurement(Component::Moe, false)
    );
    assert_ne!(
        first.measurement(Component::Gqa, false),
        second.measurement(Component::Gqa, false)
    );
    let gdn = Case { context: 1024, ..first };
    assert_eq!(
        gdn.measurement(Component::Gdn, false),
        Case { context: 65536, ..gdn }.measurement(Component::Gdn, false)
    );
    assert_ne!(
        first.measurement(Component::Gdn, false),
        gdn.measurement(Component::Gdn, false)
    );
    let prefill = Case {
        kind: Kind::FullPrefill,
        ..first
    };
    assert_eq!(
        first.measurement(Component::Gdn, false),
        prefill.measurement(Component::Gdn, false)
    );
    assert_ne!(
        first.measurement(Component::Gdn, true),
        prefill.measurement(Component::Gdn, true)
    );
}

#[test]
fn test_worker_segments_and_total_rows() {
    let case = Case {
        kind: Kind::Append,
        requests: 4,
        context: 8192,
        queries: 128,
    };
    let model = Path::new("/model");
    let gqa = args("gqa").command(case, model, "27b");
    assert!(gqa.windows(2).any(|a| a == ["--gqa-tokens-per-req", "128,128,128,128"]));
    assert!(gqa.windows(2).any(|a| a == ["--max-tokens", "512"]));
    let gdn = args("gdn").command(case, model, "27b");
    assert!(gdn.windows(2).any(|a| a == ["--prefill-requests", "4"]));
    let dense = args("dense-mlp").command(case.measurement(Component::DenseMlp, false), model, "27b");
    assert!(dense.windows(2).any(|a| a == ["--tokens", "512"]));
    assert!(!dense.contains(&"--contexts".to_owned()));
}

#[test]
fn test_input_domain_rejects_overflow_and_zero() {
    let mut a = args("gqa");
    a.requests = vec![8];
    a.queries = vec![u32::MAX];
    assert!(a.cases().is_err());
    a.queries = vec![1];
    a.contexts = vec![u32::MAX];
    assert!(a.cases().is_err());
    a.contexts = vec![0];
    a.queries = vec![0];
    assert!(a.cases().is_err());
}

#[test]
fn test_worker_samples_and_units() {
    let case = Case {
        kind: Kind::Append,
        requests: 1,
        context: 0,
        queries: 8,
    };
    let text = "fixture diagnostic\nperf component=dense-mlp-real backend=metal impl=full_auto tokens=8 iters=10 \
                runs=3 median_us=300.0 samples_us=[200.0,300.0,400.0]\n";
    let values = parse_latencies(text, Component::DenseMlp, case, 3).unwrap();
    assert_eq!(values[0].median_ms, 0.3);
    assert_eq!(values[0].samples_ms, [0.2, 0.3, 0.4]);
    assert!(parse_latencies(&text.replace("tokens=8", "tokens=16"), Component::DenseMlp, case, 3).is_err());
    assert!(parse_latencies(&text.replace("300.0", "NaN"), Component::DenseMlp, case, 3).is_err());
    assert!(parse_latencies(text, Component::DenseMlp, case, 2).is_err());
    assert!(parse_latencies(text, Component::Gqa, case, 3).is_err());
    let gqa = text.replace("component=dense-mlp-real", "component=gqa").replace(
        "impl=full_auto",
        "impl=full-forward-replay num_reqs=1 ctx=0 split_kv_variant=auto",
    );
    assert_eq!(parse_latencies(&gqa, Component::Gqa, case, 3).unwrap()[0].path, "auto");
    assert!(parse_latencies(&gqa.replace("ctx=0", "ctx=1024"), Component::Gqa, case, 3).is_err());
    let moe = text
        .replace("component=dense-mlp-real", "component=moe-real")
        .replace("tokens=8", "num_tokens=8");
    let both = format!(
        "{}{}",
        moe.replace("full_auto", "token_major"),
        moe.replace("full_auto", "expert_major")
    );
    assert_eq!(parse_latencies(&both, Component::Moe, case, 3).unwrap().len(), 2);
    assert!(parse_latencies(&moe.replace("full_auto", "token_major"), Component::Moe, case, 3).is_err());
}

#[test]
fn test_checkpoint_profile_without_weights() {
    let model = std::env::temp_dir().join(format!("component-matrix-profile-{}", std::process::id()));
    fs::create_dir_all(&model).unwrap();
    let config = model.join("config.json");
    fs::write(&config, r#"{"text_config":{"hidden_size":5120}}"#).unwrap();
    assert_eq!(model_profile(&model, Component::Gdn).unwrap(), "27b");
    assert!(model_profile(&model, Component::Moe).is_err());
    fs::write(&config, r#"{"text_config":{"hidden_size":2048}}"#).unwrap();
    assert_eq!(model_profile(&model, Component::Gqa).unwrap(), "35b");
    assert!(model_profile(&model, Component::DenseMlp).is_err());
    fs::remove_dir_all(model).unwrap();
}

#[test]
fn test_dataset_resume_and_interrupted_write() {
    let out = std::env::temp_dir().join(format!("component-matrix-resume-{}", std::process::id()));
    fs::create_dir_all(&out).unwrap();
    let path = out.join("results.jsonl");
    let case = Case {
        kind: Kind::Append,
        requests: 1,
        context: 0,
        queries: 8,
    };
    let record = Record {
        case,
        status: "measured".into(),
        command: vec![],
        latencies: vec![Latency {
            path: "full_auto".into(),
            median_ms: 1.0,
            samples_ms: vec![1.0],
        }],
        log: "run.log".into(),
        swap_before: [0, 0],
        swap_after: [0, 0],
        provenance: json!({}),
    };
    fs::write(
        &path,
        format!("{}\n{{\"case\":", serde_json::to_string(&record).unwrap()),
    )
    .unwrap();
    let records = dataset::read_records(&path).unwrap();
    assert_eq!(records[&case].status, "measured");
    assert!(fs::read(&path).unwrap().ends_with(b"\n"));
    let identity_path = out.join("identity.json");
    dataset::ensure_identity(&identity_path, &json!({"model":"a","iters":10})).unwrap();
    dataset::ensure_identity(&identity_path, &json!({"model":"a","iters":10})).unwrap();
    assert!(dataset::ensure_identity(&identity_path, &json!({"model":"b","iters":10})).is_err());
    let mut a = args("dense-mlp");
    a.out = out.clone();
    a.contexts = vec![0, 65536];
    a.queries = vec![8];
    a.requests = vec![1];
    a.full.clear();
    // Identity is schema-checked before reporting a different component.
    fs::remove_file(identity_path).unwrap();
    dataset::report(&a, &a.cases().unwrap(), &records).unwrap();
    let report: Value = serde_json::from_slice(&fs::read(out.join("report.json")).unwrap()).unwrap();
    assert_eq!(report.as_array().unwrap().len(), 2);
    assert_eq!(report[0]["status"], "measured");
    assert_eq!(report[1]["status"], "measured");
    fs::remove_dir_all(out).unwrap();
}
