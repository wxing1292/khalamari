#!/usr/bin/env python3
"""Plan, validate, run, and report the QSA indexer score experiment."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).parent / "qsa_indexer_score/worker.py"
CONTEXTS = (0, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
SMALL_Q = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16)


def variants():
    result = []

    def add(name, family, qt=8, bn=64, bk=16, **extra):
        result.append(dict(name=name, family=family, qt=qt, bn=bn, bk=bk, **extra))

    for qt, bn, bk in ((8, 32, 16), (8, 64, 16), (16, 64, 16), (16, 64, 32)):
        add(f"steel_flat_q{qt}_g{bn}_d{bk}", "flat", qt, bn, bk)
    for qt, bn in ((16, 32), (32, 32), (32, 64), (64, 64)):
        add(f"steel_head_q{qt}_g{bn}", "head", qt, bn)
    for flat in (True, False):
        for qt, bn in (
            ((8, 32), (8, 64), (16, 64)) if flat else ((16, 32), (32, 64), (64, 64))
        ):
            for qres in (False, True):
                if qres and qt == 64:
                    continue
                name = f"resident_k_{'flat' if flat else 'head'}_q{qt}_g{bn}_{'qreg' if qres else 'qdirect'}"
                add(name, "resident", qt, bn, flat=flat, qres=qres)
    for paged in (True, False):
        for bb in (1, 4):
            add(
                f"tf_simd_{'paged' if paged else 'contiguous'}_q8_b{bb}",
                "tf_simd",
                bb=bb,
                paged=paged,
            )
    for head_major, paged in ((False, False), (True, True), (True, False)):
        add(
            f"layout_head32_{'headmajor' if head_major else 'tokenmajor'}_{'paged' if paged else 'contiguous'}",
            "resident",
            32,
            64,
            flat=False,
            qres=False,
            head_major=head_major,
            paged=paged,
        )
    add("omlx_native_head64", "omlx_native", 64)
    add("omlx_mlx_fp32_flat", "mlx_flat")
    add("tf_mlx_fp32_heads", "mlx_heads")
    return result


def make_case(workload, r, t, h):
    sparse = max(0, t - max(0, 2051 - h))
    return dict(
        id=f"{workload}_r{r}_q{t}_h{h}",
        workload=workload,
        requests=r,
        tokens=t,
        history=h,
        sparse_queries=sparse,
        total_sparse_queries=r * sparse,
        groups=(h + t) // 4,
        score_bytes=r * sparse * ((h + t) // 4) * 4,
    )


def cases(suite, requests, contexts, queries):
    result = []
    if suite in ("prefill", "all"):
        result += [
            make_case("append", r, t, h)
            for r in requests
            for t in queries
            for h in contexts
        ]
        result += [
            make_case("oneshot", r, t, 0) for r in requests for t in (4096, 16384)
        ]
        # Growth references the same append measurement. Do not count repeated shapes twice.
        growth = {
            (r, t, h)
            for r in requests
            for t in (128, 256)
            for end in (4096, 16384, 32768)
            for h in range(0, end, t)
        }
        result += [make_case("growth", r, t, h) for r, t, h in sorted(growth)]
    if suite in ("small-q", "all"):
        result += [
            make_case("small_q", r, t, h)
            for r in (1, 2, 4)
            for t in SMALL_Q
            for h in contexts
        ]
    if suite == "smoke":
        result = [
            make_case("smoke", r, t, h)
            for r, t, h in (
                (1, 1, 4096),
                (1, 33, 4097),
                (2, 17, 8194),
                (4, 128, 16384),
                (1, 128, 0),
                (1, 2112, 0),
            )
        ]
    # Reuse only identical inputs, not just identical flattened token counts.
    unique = {}
    for case in result:
        key = (case["requests"], case["tokens"], case["history"])
        if key not in unique:
            case["uses"] = [case["workload"]]
            unique[key] = case
        elif case["workload"] not in unique[key]["uses"]:
            unique[key]["uses"].append(case["workload"])
    fixed = make_case("preflight_fixed", 2, 17, 4097)
    fixed.update(input_kind="zero_query", uses=["preflight_fixed"])
    return [fixed, *unique.values()]


def numbers(value):
    values = [int(v) for v in value.split(",")]
    if not values or any(v < 0 for v in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("use distinct nonnegative integers")
    return values


def swap():
    text = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
    result = {
        k: int(v) for k, v in re.findall(r"^(Swapins|Swapouts):\s+(\d+)\.", text, re.M)
    }
    if set(result) != {"Swapins", "Swapouts"}:
        raise RuntimeError("vm_stat did not return both swap activity counters")
    return result


def source_identity(config):
    sources = [Path(__file__), WORKER, WORKER.with_name("mma.metal")]
    sources.append(Path(config["python"]).resolve())
    sources += list(
        (ROOT / "crates/inference-backend-metal/src/components/metal").glob(
            "qsa_indexer*.metal"
        )
    )
    sources += [
        Path(config["omlx_dir"])
        / "omlx/custom_kernels/glm_moe_dsa/csrc/kernels/steel_dsa_indexer_score.h",
        Path(config["omlx_dir"])
        / "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/qsa_fast.py",
        Path(config["tensorfold_dir"]) / "attention.py",
        Path(config["tensorfold_dir"]) / "prefill.py",
    ]
    sources += sorted(
        (Path(config["mlx_source_dir"]) / "mlx/backend/metal/kernels").rglob("*.h")
    )
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    env = json.loads(
        subprocess.check_output(
            [
                config["python"],
                "-c",
                "import importlib.metadata as m,json,mlx.core as x,numpy; print(json.dumps(dict(mlx=m.version('mlx'),numpy=numpy.__version__,device=x.device_info())))",
            ],
            text=True,
        )
    )
    return dict(
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        dirty=subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        files=hashes,
        environment=env,
        platform=platform.platform(),
        command=sys.argv,
        omlx_commit=subprocess.check_output(
            ["git", "-C", config["omlx_dir"], "rev-parse", "HEAD"], text=True
        ).strip(),
        scope="Prepared BF16 Q/pooled K; common MLX compiled-graph submit/evaluate envelope. Score and score+fixed256-radix. Source-body adapters, not production ICB or full engine runtimes.",
    )


def supervise(command, log_path, memory_path):
    before = swap()
    started = time.time()
    invalid = False
    with log_path.open("w") as log, memory_path.open("w") as memory:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            while proc.poll() is None:
                current = swap()
                memory.write(json.dumps(dict(time=time.time(), **current)) + "\n")
                memory.flush()
                if current != before:
                    invalid = True
                    proc.terminate()
                    break
                time.sleep(0.5)
            code = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    after = swap()
    reason = "completed"
    if invalid or after != before:
        reason = "swap activity: " + ", ".join(
            f"{name} {after[name] - before[name]:+d}"
            for name in ("Swapins", "Swapouts")
        )
    elif code != 0:
        reason = f"worker exited with code {code}"
    return dict(
        exit_code=code,
        usable=code == 0 and not invalid and after == before,
        reason=reason,
        before=before,
        after=after,
        seconds=time.time() - started,
    )


def feature(case, name):
    if name == "work":
        return case["total_sparse_queries"] * case["groups"]
    return case[name]


FEATURES = ("total_sparse_queries", "sparse_queries", "groups", "work")


def predict(tree, case):
    if "arm" in tree:
        return tree["arm"]
    return predict(
        (
            tree["left"]
            if feature(case, tree["feature"]) <= tree["threshold"]
            else tree["right"]
        ),
        case,
    )


def fit(points, arms, depth=2):
    """Fit at most three shape thresholds. Normalize each shape by its own best arm."""

    def cost(subset, arm):
        return sum(
            p.get("weight", 1) * math.log(p["times"][arm] / min(p["times"].values()))
            for p in subset
        )

    def solve(subset, level):
        winner = min(arms, key=lambda a: cost(subset, a))
        best = {"arm": winner}
        loss = cost(subset, winner)
        if not level or len(subset) < 12:
            return loss, best
        for name in FEATURES:
            values = sorted({feature(p["case"], name) for p in subset})[:-1]
            # Bound report cost. The retained thresholds are measured shape boundaries.
            if len(values) > 20:
                values = [values[i * (len(values) - 1) // 19] for i in range(20)]
            for threshold in values:
                left = [p for p in subset if feature(p["case"], name) <= threshold]
                right = [p for p in subset if feature(p["case"], name) > threshold]
                if min(len(left), len(right)) < 4:
                    continue
                lt = {"arm": min(arms, key=lambda a: cost(left, a))}
                rt = {"arm": min(arms, key=lambda a: cost(right, a))}
                lc, rc = cost(left, lt["arm"]), cost(right, rt["arm"])
                # Require 3% geometric gain before adding another branch.
                if (
                    lc + rc + sum(p.get("weight", 1) for p in subset) * math.log(1.03)
                    < loss
                ):
                    loss = lc + rc
                    best = dict(feature=name, threshold=threshold, left=lt, right=rt)
        if "feature" in best:
            left = [
                p
                for p in subset
                if feature(p["case"], best["feature"]) <= best["threshold"]
            ]
            right = [
                p
                for p in subset
                if feature(p["case"], best["feature"]) > best["threshold"]
            ]
            lc, best["left"] = solve(left, level - 1)
            rc, best["right"] = solve(right, level - 1)
            loss = lc + rc
        return loss, best

    return solve(points, depth)[1]


def report(out):
    config = json.loads((out / "plan.json").read_text())
    rows = []
    completed = set()
    incomplete = []
    for path in sorted((out / "cases").glob("*/status.json")):
        status = json.loads(path.read_text())
        if not status["usable"]:
            incomplete.append(path.parent.name)
            continue
        if status["mode"] == "run":
            completed.add(path.parent.name)
        for line in (path.parent / "samples.jsonl").read_text().splitlines():
            item = json.loads(line)
            if item["kind"] == "sample":
                rows.append(item)
    grouped = defaultdict(list)
    cases_by_id = {c["id"]: c for c in config["cases"]}
    for row in rows:
        grouped[row["case_id"], row["phase"], row["arm"]].append(row["us"])
    medians = {key: statistics.median(values) for key, values in grouped.items()}
    with (out / "summary.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case",
                "requests",
                "q_tokens_per_request",
                "history",
                "sparse_queries",
                "groups",
                "phase",
                "arm",
                "median_us",
                "samples",
            ]
        )
        for (ident, phase, arm), us in sorted(medians.items()):
            c = cases_by_id[ident]
            writer.writerow(
                [
                    ident,
                    c["requests"],
                    c["tokens"],
                    c["history"],
                    c["total_sparse_queries"],
                    c["groups"],
                    phase,
                    arm,
                    us,
                    len(grouped[ident, phase, arm]),
                ]
            )
    pending = [
        c
        for c in config["cases"]
        if c.get("status", "pending") == "pending" and c["id"] not in completed
    ]
    exclusions = sum(
        c.get("status") == "excluded_score_budget" for c in config["cases"]
    )
    dense = sum(c.get("status") == "skipped_all_dense" for c in config["cases"])
    text = [
        "# QSA indexer score study",
        "",
        "Only clean completed cases enter this report.",
        f"Remaining cases: {len(pending)}. Score-budget exclusions: {exclusions}. Dense-only skips: {dense}.",
        (
            "PRELIMINARY: the planned matrix is incomplete."
            if pending
            else "The planned in-budget matrix is complete."
        ),
        "Timing uses one common MLX compiled-graph envelope. These are source-kernel controls, not production ICB or full runtime comparisons.",
        "The common selector is the production exact full radix with 256 threads. Q/K preparation, page construction and Q layout packing are excluded.",
        "FP32 matmul controls include casts, per-head intermediate output and head reduction. The head-major input is prepared outside timing.",
        "",
        f"Completed sample records: {len(rows)}. Unusable attempts: {len(incomplete)}.",
        "",
    ]
    policies = {}
    arms = [v["name"] for v in config["variants"]]
    baseline = "steel_flat_q8_g64_d16" if "steel_flat_q8_g64_d16" in arms else arms[0]
    text += [
        f"Baseline: `{baseline}`. Ratios below divide per-shape medians.",
        "Selection fits at most three thresholds and requires 3% training gain per branch. Training balances workload families.",
        "",
    ]
    for phase in ("score", "score_radix"):
        points = []
        for ident, c in cases_by_id.items():
            if all((ident, phase, a) in medians for a in arms):
                points.append(
                    dict(case=c, times={a: medians[ident, phase, a] for a in arms})
                )
        if not points:
            continue
        sizes = defaultdict(int)
        for point in points:
            sizes[point["case"]["workload"]] += 1
        for point in points:
            point["weight"] = 1 / sizes[point["case"]["workload"]]
        # Contexts are held out as complete groups, including their growth chunks.
        train = [p for p in points if (p["case"]["history"] // 1024) % 3 != 1]
        held = [p for p in points if (p["case"]["history"] // 1024) % 3 == 1]
        text += [
            f"## {phase}",
            "",
            "| Fixed solution | Geomean / baseline | Worst / baseline |",
            "| --- | ---: | ---: |",
        ]
        ratios = {
            a: [p["times"][a] / p["times"][baseline] for p in points] for a in arms
        }
        geo = lambda xs: math.exp(statistics.mean(map(math.log, xs)))
        for arm in sorted(arms, key=lambda a: geo(ratios[a])):
            text.append(f"| {arm} | {geo(ratios[arm]):.3f} | {max(ratios[arm]):.3f} |")
        for workload in sorted(sizes):
            subset = [p for p in points if p["case"]["workload"] == workload]
            text += [
                "",
                f"### {workload}",
                "",
                "| Fixed solution | Geomean / baseline | Worst / baseline |",
                "| --- | ---: | ---: |",
            ]
            for arm in sorted(
                arms,
                key=lambda a: geo(
                    [p["times"][a] / p["times"][baseline] for p in subset]
                ),
            ):
                rel = [p["times"][arm] / p["times"][baseline] for p in subset]
                text.append(f"| {arm} | {geo(rel):.3f} | {max(rel):.3f} |")
        if not train or not held:
            text += [
                "",
                "Insufficient independent context groups for a held-out selection result.",
                "",
            ]
            continue
        policy = fit(train, arms)
        selected = [
            p["times"][predict(policy, p["case"])] / p["times"][baseline] for p in held
        ]
        oracle = [min(p["times"].values()) / p["times"][baseline] for p in held]
        regret = [
            p["times"][predict(policy, p["case"])] / min(p["times"].values())
            for p in held
        ]
        policies[phase] = dict(
            tree=policy,
            training_shapes=len(train),
            heldout_shapes=len(held),
            heldout_vs_baseline=geo(selected),
            oracle_vs_baseline=geo(oracle),
            worst_regret=max(regret),
        )
        text += [
            "",
            f"Held-out shapes: {len(held)}. Selected / baseline: {geo(selected):.3f}. Oracle / baseline: {geo(oracle):.3f}.",
            f"Worst selected / per-shape oracle: {max(regret):.3f}.",
            "",
            "The oracle is an unattainable per-shape lower bound, not a deployable policy.",
            "",
            "```json",
            json.dumps(policy, indent=2),
            "```",
            "",
        ]
        text += [
            "| Held-out selected path vs peer | Geomean ratio | Worst ratio |",
            "| --- | ---: | ---: |",
        ]
        for peer in (
            "omlx_native_head64",
            "omlx_mlx_fp32_flat",
            "tf_mlx_fp32_heads",
            "tf_simd_contiguous_q8_b1",
        ):
            if peer in arms:
                rel = [
                    p["times"][predict(policy, p["case"])] / p["times"][peer]
                    for p in held
                ]
                text.append(f"| {peer} | {geo(rel):.3f} | {max(rel):.3f} |")
        text.append("")
    # Report growth from measured chunks. Do not invent a continuous cache trace.
    text += [
        "## Chunked growth",
        "",
        "Growth uses independent synthetic fixtures per chunk. It is not continuous-request TTFT.",
        "",
        "| Requests | Chunk | End | Phase | Arm | Sum of chunk medians, us |",
        "| ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for r in config["requests"]:
        for t in (128, 256):
            for end in (4096, 16384, 32768):
                ids = [
                    c["id"]
                    for c in config["cases"]
                    if c["requests"] == r
                    and c["tokens"] == t
                    and c["history"] in range(0, end, t)
                    and c["sparse_queries"] > 0
                ]
                expected = sum(
                    make_case("growth", r, t, h)["sparse_queries"] > 0
                    for h in range(0, end, t)
                )
                if len(ids) != expected:
                    continue
                for phase in ("score", "score_radix"):
                    for arm in arms:
                        if all((ident, phase, arm) in medians for ident in ids):
                            total = sum(medians[ident, phase, arm] for ident in ids)
                            text.append(
                                f"| {r} | {t} | {end} | {phase} | {arm} | {total:.2f} |"
                            )
    (out / "report.md").write_text("\n".join(text) + "\n")
    (out / "selection.json").write_text(json.dumps(policies, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", choices=("plan", "validate", "run", "report"), default="plan"
    )
    p.add_argument(
        "--suite", choices=("prefill", "small-q", "all", "smoke"), default="prefill"
    )
    p.add_argument(
        "--out", type=Path, default=ROOT / "target/perf/qsa-indexer-score-study"
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--omlx-dir", type=Path, default=ROOT.parent / "omlx")
    p.add_argument("--tensorfold-dir", type=Path)
    p.add_argument("--mlx-source-dir", type=Path)
    p.add_argument("--requests", type=numbers, default=[1, 2, 4, 8])
    p.add_argument("--contexts", type=numbers, default=list(CONTEXTS))
    p.add_argument("--queries", type=numbers, default=[32, 64, 128, 256, 512, 1024])
    p.add_argument("--arms", help="comma-separated solution names")
    p.add_argument("--seeds", type=numbers, default=[7, 29])
    p.add_argument("--rounds", type=int, default=11)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--invocations", type=int, default=2)
    p.add_argument(
        "--hours",
        type=float,
        default=0,
        help="time budget in hours; 0 runs until completion (default)",
    )
    p.add_argument("--max-score-mib", type=int, default=2048)
    a = p.parse_args()
    executable = shutil.which(a.python)
    if executable is None:
        p.error(f"Python executable not found: {a.python}")
    a.python = executable
    out = a.out.resolve()
    if a.mode == "report":
        report(out)
        print(out / "report.md")
        return
    if any(r not in (1, 2, 4, 8) for r in a.requests) or any(t <= 0 for t in a.queries):
        p.error("requests must be 1/2/4/8 and queries must be positive")
    if (
        a.rounds < 3
        or a.warmups < 1
        or a.invocations < 1
        or a.hours < 0
        or a.max_score_mib <= 0
    ):
        p.error(
            "use rounds>=3, positive warmups/invocations/memory limit, and hours>=0"
        )
    repertoire = variants()
    if a.arms:
        wanted = a.arms.split(",")
        known = {v["name"] for v in repertoire}
        if set(wanted) - known:
            p.error(f"unknown arms: {sorted(set(wanted)-known)}")
        repertoire = [v for v in repertoire if v["name"] in wanted]
    if not a.tensorfold_dir and a.mode != "plan":
        a.tensorfold_dir = Path(
            subprocess.check_output(
                [
                    a.python,
                    "-c",
                    "import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec('tensorfold').origin).parent/'kernels/qwen/flash_next/v1')",
                ],
                text=True,
            ).strip()
        )
    if not a.mlx_source_dir:
        env = os.environ.get("INFERENCE_BACKEND_METAL_MLX_SOURCE_DIR")
        candidates = [Path(env)] if env else []
        if not env and a.mode != "plan":
            candidates.append(
                Path(
                    subprocess.check_output(
                        [
                            a.python,
                            "-c",
                            "import importlib.util,pathlib; print(pathlib.Path(next(iter(importlib.util.find_spec('mlx').submodule_search_locations)))/'include')",
                        ],
                        text=True,
                    ).strip()
                )
            )
        a.mlx_source_dir = next(
            (
                d
                for d in candidates
                if (d / "mlx/backend/metal/kernels/steel/gemm/gemm.h").is_file()
            ),
            None,
        )
    planned = cases(a.suite, a.requests, a.contexts, a.queries)
    for c in planned:
        c["status"] = "skipped_all_dense" if not c["sparse_queries"] else "pending"
        if c["score_bytes"] > a.max_score_mib * 1024**2:
            c["status"] = "excluded_score_budget"
    config = dict(
        schema=1,
        suite=a.suite,
        requests=a.requests,
        cases=planned,
        variants=repertoire,
        seeds=a.seeds,
        rounds=a.rounds,
        warmups=a.warmups,
        invocations=a.invocations,
        mode=a.mode,
        python=str(Path(a.python).absolute()),
        omlx_dir=str(a.omlx_dir.resolve()),
        tensorfold_dir=str(a.tensorfold_dir) if a.tensorfold_dir else None,
        mlx_source_dir=str(a.mlx_source_dir) if a.mlx_source_dir else None,
        max_score_mib=a.max_score_mib,
        scope="score and score+common exact full256 radix; common MLX runtime envelope",
    )
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "plan":
        (out / "plan-preview.json").write_text(json.dumps(config, indent=2) + "\n")
        print(
            f"{len(planned)} logical shapes, {sum(c['status']=='pending' for c in planned)} sparse; {len(repertoire)} solutions. No GPU work.\n{out/'plan-preview.json'}"
        )
        return
    if not a.mlx_source_dir:
        p.error("set --mlx-source-dir to an MLX source tree containing Steel headers")
    identity = source_identity(config)
    # Compare sources, executable environment, and timing plan on resume. Git dirty text is provenance only.
    identity_key = {k: identity[k] for k in ("files", "environment", "platform")}
    identity_key["plan"] = {k: v for k, v in config.items() if k not in ("mode",)}
    key = hashlib.sha256(json.dumps(identity_key, sort_keys=True).encode()).hexdigest()
    manifest = out / "identity.json"
    if manifest.exists() and json.loads(manifest.read_text())["identity_key"] != key:
        p.error(
            "measurement source/environment/plan changed; use a new --out directory"
        )
    identity["identity_key"] = key
    manifest.write_text(json.dumps(identity, indent=2) + "\n")
    (out / "plan.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "cases").mkdir(exist_ok=True)
    # One local controller at a time. The user must also stop unrelated GPU benches.
    import fcntl

    lock = (ROOT / "target/perf/qsa-indexer-score-study.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic() + a.hours * 3600 if a.hours else math.inf
    stopped = False
    try:
        for i, c in enumerate(planned):
            if c["status"] != "pending":
                continue
            dest = out / "cases" / c["id"]
            dest.mkdir(exist_ok=True)
            status_file = dest / "status.json"
            if status_file.exists():
                previous = json.loads(status_file.read_text())
                if previous["usable"] and (
                    a.mode == "validate" or previous["mode"] == "run"
                ):
                    continue
            if time.monotonic() > deadline:
                stopped = True
                break
            # Preserve an unsuccessful attempt before retrying it.
            for old in ("samples.jsonl", "worker.log", "memory.jsonl", "status.json"):
                if (dest / old).exists():
                    (dest / old).rename(dest / f"{old}.{time.time_ns()}.previous")
            cmd = [
                a.python,
                str(WORKER),
                "--config",
                str(out / "plan.json"),
                "--case",
                str(i),
                "--output",
                str(dest / "samples.jsonl"),
            ]
            print(f"[{i+1}/{len(planned)}] {c['id']}", flush=True)
            status = supervise(cmd, dest / "worker.log", dest / "memory.jsonl")
            if any(
                hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest
                for name, digest in identity["files"].items()
            ):
                status.update(
                    usable=False, reason="measurement source changed during run"
                )
            status["mode"] = a.mode
            status_file.write_text(json.dumps(status, indent=2) + "\n")
            if not status["usable"]:
                raise RuntimeError(
                    f"unusable case {c['id']}: {status['reason']}; "
                    f"inspect {dest/'worker.log'} and {dest/'memory.jsonl'}"
                )
    finally:
        report(out)
    print(
        f"{'Paused at time budget; repeat command to resume.' if stopped else 'Complete.'} Results: {out}"
    )


if __name__ == "__main__":
    main()
