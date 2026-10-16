"""GPU worker for the QSA score study. Import only in a GPU worker process."""

import ast
import hashlib
import json
import math
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMPONENT = ROOT / "crates/inference-backend-metal/src/components/metal"
INPUTS = [
    "Q_ALL",
    "PAGES",
    "PAGE_IDS",
    "TOKENS",
    "TASKS",
    "PARAMS",
    "CONTIG",
    "QHEAD",
    "COMPLETE",
]
TYPES = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
using scalar_t = bfloat;
constant uint ratio=4, dim=128, heads=4, top_k=512, selection_threads=256;
struct Token { uint batch_start,token_index,req_slot,src_state_slot,dst_state_slot,group_start; };
struct ScoreTaskTemplate { uint query_start,num_queries; };
inline uint num_visible_groups(Token t) { return (t.token_index+1)/4-t.group_start; }
"""
ALIASES = """
using T = bfloat;
const device T* Q = Q_ALL;
const device T* Pages = PAGES;
const auto* page_ids = PAGE_IDS;
const uint num_active_score_task_templates=PARAMS[0], pages_per_request=PARAMS[1];
const uint groups_per_page=128, num_total_groups=PARAMS[3];
const uint simd_lane_id=thread_index_in_simdgroup, simd_group_id=simdgroup_index_in_threadgroup;
const uint3 grid=threadgroup_position_in_grid;
"""


def body(text, symbol):
    start = text.index("{", text.index(symbol + "("))
    level = 1
    stop = start + 1
    while level:
        level += (text[stop] == "{") - (text[stop] == "}")
        stop += 1
    return text[start + 1 : stop - 1]


def literal(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"{path}: missing literal {name}")


def token_at(index):
    return "Token{" + ",".join(f"TOKENS[6*({index})+{i}]" for i in range(6)) + "}"


def headers(root):
    seen = set()

    def read(rel):
        if rel in seen:
            return ""
        seen.add(rel)
        # MLX already injects these common headers into every custom library.
        if rel in {
            "mlx/backend/metal/kernels/" + name
            for name in ("utils.h", "defines.h", "bf16.h", "bf16_math.h", "complex.h")
        }:
            return ""
        lines = []
        for line in (root / rel).read_text().splitlines():
            match = re.match(r'\s*#include "([^"]+)"', line)
            if match and match[1] == "bf16.h":
                lines.append(read("mlx/backend/metal/kernels/metal_3_1/bf16.h"))
            elif match and match[1].startswith("mlx/backend/metal/kernels/"):
                lines.append(read(match[1]))
            elif "#pragma once" not in line:
                lines.append(line)
        return "\n".join(lines)

    return read("mlx/backend/metal/kernels/steel/gemm/gemm.h")


def make_kernel(name, source, header, inputs=INPUTS):
    digest = hashlib.sha256((source + header).encode()).hexdigest()[:12]
    return mx.fast.metal_kernel(
        name=f"qsa_study_{name}_{digest}",
        input_names=inputs,
        output_names=["O"],
        source=source,
        header=header,
    )


def kernel_source(spec, steel, omlx, tensorfold):
    family, qt, bn, bk = spec["family"], spec["qt"], spec["bn"], spec["bk"]
    constants = f"constexpr int QT={qt},BM={qt * 4 if family == 'flat' else qt},BN={bn},BK={bk},WM=2,WN=2;\n"
    if family in ("flat", "head"):
        text = body(
            (COMPONENT / "qsa_indexer_steel.metal").read_text(),
            "qsa_indexer_steel_score",
        )
        # Forced study tiles own their tail. Production auto dispatch remains unchanged.
        text = re.sub(
            r"  if constexpr \(FLAT\) \{\s*if \(score_task.num_queries > 8u\) return;\s*\} else \{\s*if \(score_task.num_queries <= 8u\) return;\s*\}",
            "",
            text,
        )
        text = text.replace("k < 8", f"k < {128 // bk}")
        text = text.replace(
            "score_tasks[grid.y]",
            "ScoreTaskTemplate{TASKS[2*grid.y],TASKS[2*grid.y+1]}",
        )
        text = text.replace(
            "tokens[score_task.query_start]", token_at("score_task.query_start")
        )
        return (
            constants
            + f"constexpr bool FLAT={'true' if family == 'flat' else 'false'};\n"
            + ALIASES
            + text,
            TYPES + steel + "\nusing namespace mlx::steel;\n",
            128,
        )
    if family.startswith("resident"):
        flat = spec["flat"]
        prefix = f"constexpr int QT={qt},BN={bn}; constexpr bool FLAT={str(flat).lower()},QRES={str(spec['qres']).lower()},QHEADMAJOR={str(spec.get('head_major',False)).lower()},KPAGED={str(spec.get('paged',True)).lower()};\n"
        return prefix + (Path(__file__).parent / "mma.metal").read_text(), TYPES, 128
    if family == "omlx_native":
        text = body(
            (
                omlx
                / "omlx/custom_kernels/glm_moe_dsa/csrc/kernels/steel_dsa_indexer_score.h"
            ).read_text(),
            "qwen4_qsa_indexer_score",
        )
        fields = {
            "M": "int(PARAMS[6])",
            "N": "int(PARAMS[3])",
            "K": "128",
            "lda": "128",
            "ldb": "128",
            "ldd": "int(PARAMS[3])",
            "tiles_m": "int((PARAMS[6]+63)/64)",
            "tiles_n": "int((PARAMS[3]+63)/64)",
            "swizzle_log": "0",
            "gemm_k_iterations_aligned": "8",
        }
        for field, value in fields.items():
            text = text.replace("params->" + field, "(" + value + ")")
        assert "params->" not in text
        prefix = """
using T=bfloat; constexpr int BM=64,BN=64,BK=16,WM=2,WN=2;
const device T* Q=QHEAD; const device T* K=CONTIG;
const uint3 tid=threadgroup_position_in_grid,lid=thread_position_in_threadgroup;
const uint simd_lane_id=thread_index_in_simdgroup,simd_group_id=simdgroup_index_in_threadgroup;
const int mask_ratio=4,mask_q_offset=int(PARAMS[8]); const float score_divisor=sqrt(128.0f);
"""
        return prefix + text, TYPES + steel + "\nusing namespace mlx::steel;\n", 128
    if family == "tf_simd":
        text = literal(tensorfold / "attention.py", "_IDX_SCORES")
        if spec["paged"]:
            text = text.replace(
                "POOLED + size_t(b) * DI + lane * PER",
                "PAGES + (ulong(PAGE_IDS[req*PARAMS[1]+b/128])*128+b%128)*128 + lane*PER",
            )
        prefix = f"constexpr int RB={qt},BB={spec['bb']},DI=128,HI=4,TOP=512;\n" + """
const uint req=threadgroup_position_in_grid.z;
const device bfloat* Q=Q_ALL+ulong(req)*PARAMS[6]*4*128;
const device bfloat* POOLED=CONTIG+ulong(req)*PARAMS[3]*128;
const auto* complete_local=COMPLETE_ALL+req*PARAMS[6];
device float* SC=O+ulong(req)*PARAMS[6]*PARAMS[3];
const int Q_shape[1]={int(PARAMS[6])},POOLED_shape[1]={int(PARAMS[3])};
"""
        return prefix + text.replace("COMPLETE[r]", "complete_local[r]"), TYPES, 256
    raise ValueError(family)


class Fixture:
    def __init__(self, case, seed, zeros=False):
        r, t, h = case["requests"], case["tokens"], case["history"]
        self.start = max(0, 2051 - h)
        self.n = t - self.start
        self.r, self.g = r, (h + t) // 4
        rng = np.random.default_rng(seed)
        self.q = rng.integers(-8, 9, (r * self.n, 4, 128)).astype(np.float32) / 16
        if zeros:
            self.q.fill(0)
        self.k = rng.integers(-8, 9, (r, self.g, 128)).astype(np.float32) / 16
        self.complete = np.tile(
            (h + self.start + np.arange(self.n) + 1) // 4, r
        ).astype(np.uint32)
        ppr = (self.g + 127) // 128
        self.ids = np.arange(r * ppr, dtype=np.uint32)[::-1].reshape(r, ppr).copy()
        pages = np.zeros((r * ppr, 128, 128), np.float32)
        for req in range(r):
            for page in range(ppr):
                src = self.k[req, page * 128 : (page + 1) * 128]
                pages[self.ids[req, page], : len(src)] = src
        tokens = np.zeros((r * self.n, 6), np.uint32)
        for req in range(r):
            sl = slice(req * self.n, (req + 1) * self.n)
            tokens[sl, 0] = req * self.n
            tokens[sl, 1] = h + self.start + np.arange(self.n)
            tokens[sl, 2] = req
        self.tokens = mx.array(tokens)
        self.q_gpu = mx.array(self.q).astype(mx.bfloat16)
        self.k_gpu = mx.array(self.k).astype(mx.bfloat16)
        self.pages = mx.array(pages).astype(mx.bfloat16)
        self.page_ids = mx.array(self.ids)
        self.q_head = mx.contiguous(
            self.q_gpu.reshape(r, self.n, 4, 128).transpose(0, 2, 1, 3)
        )
        self.complete_gpu = mx.array(self.complete)
        self.base = [
            self.q_gpu,
            self.pages,
            self.page_ids,
            self.tokens,
            None,
            None,
            self.k_gpu,
            self.q_head,
            self.complete_gpu,
        ]
        self.offset = h + self.start
        mx.eval(
            self.q_gpu,
            self.k_gpu,
            self.pages,
            self.page_ids,
            self.tokens,
            self.q_head,
            self.complete_gpu,
        )

    def inputs(self, qt):
        tasks = [
            [req * self.n + q, min(qt, self.n - q)]
            for req in range(self.r)
            for q in range(0, self.n, qt)
        ]
        values = list(self.base)
        values[4] = mx.array(tasks, mx.uint32)
        values[5] = mx.array(
            [
                len(tasks),
                self.ids.shape[1],
                128,
                self.g,
                7,
                self.r * self.n,
                self.n,
                self.r,
                self.offset,
            ],
            mx.uint32,
        )
        mx.eval(values[4], values[5])
        return values

    def oracle(self, rows):
        out = []
        for q in rows:
            req = q // self.n
            c = int(self.complete[q])
            dots = self.q[q].astype(np.float64) @ self.k[req, :c].astype(np.float64).T
            out.append(
                (np.maximum(dots, 0).sum(axis=0) / math.sqrt(128)).astype(np.float32)
            )
        return out


def selector():
    text = (COMPONENT / "qsa_indexer.metal").read_text()
    helpers = text[
        text.index("inline void radix_cut") : text.index(
            "// Task: one query's exact top-k"
        )
    ]
    source = """
const uint query=threadgroup_position_in_grid.x,tid=thread_position_in_threadgroup.x;
const uint num_active_tokens=uint(SC_shape[0]),num_total_groups=uint(SC_shape[1]);
const device float* scores=SC; device uint* ids=O;
""" + body(text, "qsa_indexer_select").replace("tokens[query]", token_at("query"))
    return make_kernel("common_radix", source, TYPES + helpers, ["SC", "TOKENS"])


def run_job(config, case, output):
    steel = headers(Path(config["mlx_source_dir"]))
    omlx = Path(config["omlx_dir"])
    tf = Path(config["tensorfold_dir"])
    select = selector()
    case_id = case["id"]
    kernels = {}
    selected = config["variants"]
    sources = output.parent / "expanded"
    sources.mkdir(exist_ok=True)
    for spec in selected:
        if spec["family"] in ("mlx_flat", "mlx_heads"):
            continue
        source, header, threads = kernel_source(spec, steel, omlx, tf)
        # Keep the generated source that the MLX runtime compiler sees.
        (sources / (spec["name"] + ".metal")).write_text(header + "\n" + source)
        kernels[spec["name"]] = (
            make_kernel(
                spec["name"],
                source,
                header,
                (
                    ["COMPLETE_ALL" if n == "COMPLETE" else n for n in INPUTS]
                    if spec["family"] == "tf_simd"
                    else INPUTS
                ),
            ),
            threads,
        )

    with output.open("a", buffering=1) as log:

        def emit(item):
            log.write(json.dumps({"case": case, "case_id": case_id, **item}) + "\n")

        for seed in config["seeds"]:
            fixture = Fixture(case, seed, zeros=case.get("input_kind") == "zero_query")
            # All small fixtures. Stratified rows for large fixtures, all visible keys per checked row.
            rows = (
                list(range(fixture.r * fixture.n))
                if fixture.r * fixture.n <= 32
                else sorted(
                    set(
                        req * fixture.n + q
                        for req in range(fixture.r)
                        for q in (
                            0,
                            1,
                            fixture.n // 3,
                            fixture.n // 2,
                            fixture.n - 2,
                            fixture.n - 1,
                        )
                    )
                )
            )
            refs = fixture.oracle(rows)
            fixture_hash = hashlib.sha256(
                fixture.q.tobytes() + fixture.k.tobytes()
            ).hexdigest()
            fns = {}
            for spec in selected:
                name, family = spec["name"], spec["family"]
                inputs = fixture.inputs(spec["qt"])
                if family in ("mlx_flat", "mlx_heads"):
                    # Same formulas as the peers, without model/cache package imports.
                    # These controls include FP32 casts in the timed graph.
                    def score(
                        q,
                        pages,
                        ids,
                        toks,
                        tasks,
                        params,
                        keys,
                        qhead,
                        complete,
                        family=family,
                    ):
                        if family == "mlx_flat":
                            dots = q.astype(mx.float32).reshape(
                                fixture.r, fixture.n * 4, 128
                            ) @ keys.astype(mx.float32).swapaxes(-1, -2)
                            dots = dots.reshape(fixture.r * fixture.n, 4, fixture.g)
                        else:
                            dots = (
                                qhead.astype(mx.float32)
                                @ keys.astype(mx.float32).swapaxes(-1, -2)[:, None]
                            )
                            dots = dots.transpose(0, 2, 1, 3).reshape(
                                fixture.r * fixture.n, 4, fixture.g
                            )
                        positive = mx.maximum(dots, 0)
                        if family == "mlx_flat":
                            # Keep oMLX's portable reduction, including MLX's choice of kernel.
                            return mx.sum(positive, axis=1) / math.sqrt(128)
                        # TensorFold's prefill helper adds heads in order.
                        return (
                            positive[:, 0]
                            + positive[:, 1]
                            + positive[:, 2]
                            + positive[:, 3]
                        ) / math.sqrt(128)

                else:
                    kernel, threads = kernels[name]
                    if family == "omlx_native":
                        grid = (
                            (fixture.g + 63) // 64 * threads,
                            (fixture.n + 63) // 64,
                            fixture.r,
                        )
                    elif family == "tf_simd":
                        grid = (
                            (fixture.g + 8 * spec["bb"] - 1)
                            // (8 * spec["bb"])
                            * threads,
                            (fixture.n + spec["qt"] - 1) // spec["qt"],
                            fixture.r,
                        )
                    else:
                        grid = (
                            (fixture.g + spec["bn"] - 1) // spec["bn"] * threads,
                            len(inputs[4]),
                            1,
                        )

                    def score(*args, kernel=kernel, threads=threads, grid=grid):
                        return kernel(
                            inputs=list(args),
                            grid=grid,
                            threadgroup=(threads, 1, 1),
                            output_shapes=[(fixture.r * fixture.n, fixture.g)],
                            output_dtypes=[mx.float32],
                        )[0]

                score_fn = mx.compile(score)

                def combined(*args, score_fn=score_fn):
                    sc = score_fn(*args)
                    return select(
                        inputs=[sc, args[3]],
                        grid=(fixture.r * fixture.n * 256, 1, 1),
                        threadgroup=(256, 1, 1),
                        output_shapes=[(fixture.r * fixture.n, 512)],
                        output_dtypes=[mx.uint32],
                    )[0]

                combined_fn = mx.compile(combined)
                result = score_fn(*inputs)
                mx.eval(result)
                actual = np.array(result[mx.array(rows)])
                error = max(
                    float(np.max(np.abs(actual[i, : len(ref)] - ref)))
                    for i, ref in enumerate(refs)
                )
                if not math.isfinite(error) or error > 0.002:
                    raise AssertionError(f"{case_id}/{name}: CPU score error {error}")
                result_ids = np.array(combined_fn(*inputs))
                for i, q in enumerate(rows):
                    want = np.sort(
                        np.lexsort((np.arange(len(refs[i])), -refs[i]))[:512]
                    )
                    np.testing.assert_array_equal(
                        result_ids[q], want, err_msg=f"{case_id}/{name}/q{q}"
                    )
                # Every query must have sorted, distinct, causally visible IDs.
                assert np.all(np.diff(result_ids.astype(np.int64), axis=1) > 0)
                assert np.all(result_ids[:, -1] < fixture.complete)
                emit(
                    {
                        "kind": "correctness",
                        "seed": seed,
                        "arm": name,
                        "checked_rows": rows,
                        "max_error": error,
                        "fixture_sha256": fixture_hash,
                    }
                )
                for phase, fn in (("score", score_fn), ("score_radix", combined_fn)):
                    fns[name, phase] = (fn, inputs)

            if config["mode"] == "validate" or case["workload"] == "preflight_fixed":
                continue
            names = list(fns)
            for key in names:
                fn, inputs = fns[key]
                for _ in range(config["warmups"]):
                    mx.eval(fn(*inputs))
            # Rotate and reverse order so one arm does not own one desktop-load interval.
            for repeat in range(config["rounds"]):
                order = names[repeat % len(names) :] + names[: repeat % len(names)]
                if repeat % 2:
                    order = order[::-1]
                for name, phase in order:
                    fn, inputs = fns[name, phase]
                    mx.synchronize()
                    begin = time.perf_counter_ns()
                    for _ in range(config["invocations"]):
                        mx.eval(fn(*inputs))
                    mx.synchronize()
                    us = (time.perf_counter_ns() - begin) / 1000 / config["invocations"]
                    emit(
                        {
                            "kind": "sample",
                            "seed": seed,
                            "round": repeat,
                            "arm": name,
                            "phase": phase,
                            "us": us,
                            "fixture_sha256": fixture_hash,
                        }
                    )
            mx.clear_cache()


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--case", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    run_job(config, config["cases"][a.case], a.output)


if __name__ == "__main__":
    main()
