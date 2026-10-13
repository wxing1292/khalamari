#!/usr/bin/env python3
"""Run a resumable, fixed-work Qwen3.5-family serving latency matrix."""

import argparse
import gc
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

CONTEXTS = [0, 2048, 4096, 8192, 16384, 32768]
TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
FULL = [2048, 4096, 8192, 16384]
MASK = (1 << 64) - 1


def token(req, pos, vocab):
    value = pos ^ (((req + 1) * 0x9E3779B97F4A7C15) & MASK)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return (value ^ (value >> 31)) % vocab


def streams(req, ctx, q, iteration, vocab, shared_prefix=False):
    # Share the prefix and up to four lookahead tokens across samples. MTP
    # cache lanes need those identities. Vary the remaining suffix before it can
    # publish another complete prefix-cache block.
    return [
        [token(0 if shared_prefix else r, p, vocab) for p in range(ctx)]
        + [
            token(
                (0 if shared_prefix else r)
                if p < ctx + 4
                else r + (iteration + 1) * 65536 + q * 16,
                p,
                vocab,
            )
            for p in range(ctx, ctx + q)
        ]
        for r in range(req)
    ]


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def shapes(args):
    return [
        (r, "append", c, q)
        for r in args.requests
        for c in args.contexts
        for q in args.tokens
    ] + [(r, "full", 0, q) for r in args.requests for q in args.full]


def key(engine, mode, shape):
    return ":".join(map(str, (engine, mode, *shape)))


def read_rows(out):
    path = out / "results.jsonl"
    rows = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    exclusions = out / "timing-exclusions.jsonl"
    if exclusions.exists():
        excluded = {
            (entry["key"], entry.get("result_offset")): entry["status"]
            for entry in map(json.loads, exclusions.read_text().splitlines())
        }
        for row in rows:
            status = excluded.get((row["key"], row.get("result_offset")))
            if status is not None:
                row["status"] = status
    return rows


def write_row(args, row):
    with (args.out / "results.jsonl").open("a") as f:
        # A retry has the same case key. Its durable ledger offset identifies
        # the result to exclude without invalidating later clean measurements.
        row["result_offset"] = f.tell()
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(
        json.dumps(
            {
                k: row.get(k)
                for k in ("key", "status", "ttft_ms", "decode_ms", "total_ms")
            }
        ),
        flush=True,
    )
    report(args)


def report(args):
    rows = {row["key"]: row for row in read_rows(args.out)}
    provenance_rows = [
        json.loads(p.read_text()) for p in sorted(args.out.glob("*-provenance.json"))
    ]
    context = [
        f"- {r['engine']} / {r['mode']}: versions `{r['versions']}`; commit `{r['source_commit']}`; output/request {r['output_tokens']}; timed chunk {r['chunk']}; KV {r['kv_cache']}."
        for r in provenance_rows
    ]
    for metric, path, title in (
        ("total_ms", "report.md", "Total latency"),
        ("ttft_ms", "ttft.md", "Time to first token"),
        ("decode_ms", "decode.md", "Decode latency"),
    ):
        lines = [
            f"# Qwen3.5-family serving matrix: {title}",
            "",
            "Times are batch wall latency in milliseconds. Positive differences mean psi-dec is slower.",
            "TTFT ends when every request has its first token. Decode ends when every request finishes.",
            "N/A is not a timing result. Native TensorFold 1.0.4 is not the Python fixed-chain path measured here.",
            "",
            "[Total](report.md) · [TTFT](ttft.md) · [Decode](decode.md)",
            "",
            f"Planned shapes: {len(shapes(args))}. Recorded engine/mode points: {len(rows)}.",
            "",
            *context,
            "",
        ]
        for mode in args.modes:
            lines += [f"## {mode}", ""]
            for kind in ("append", "full"):
                for req in args.requests:
                    lines += [
                        f"### {kind}, requests = {req}",
                        "",
                        "| ctx | input/request | psi-dec | oMLX | TensorFold Python | psi vs oMLX | psi vs TF |",
                        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                    ]
                    for shape in shapes(args):
                        if shape[0] != req or shape[1] != kind:
                            continue
                        found = [
                            rows.get(key(engine, mode, shape))
                            for engine in ("psi-dec", "omlx", "tensorfold")
                        ]

                        def cell(row):
                            if row is None:
                                return "pending"
                            if row["status"] != "measured":
                                return row["status"]
                            return f"{row[metric]:.1f}"

                        def diff(other):
                            if (
                                not found[0]
                                or not other
                                or found[0]["status"] != "measured"
                                or other["status"] != "measured"
                            ):
                                return "—"
                            assert (
                                found[0]["input_sha256"] == other["input_sha256"]
                            ), "comparison input mismatch"
                            assert [
                                s["output_tokens"] for s in found[0]["samples"]
                            ] == [
                                s["output_tokens"] for s in other["samples"]
                            ], "comparison output count mismatch"
                            return (
                                f"{100 * (found[0][metric] / other[metric] - 1):+.1f}%"
                            )

                        lines.append(
                            f"| {shape[2]} | {shape[3]} | {cell(found[0])} | {cell(found[1])} | {cell(found[2])} | {diff(found[1])} | {diff(found[2])} |"
                        )
                    lines.append("")
        (args.out / path).write_text("\n".join(lines) + "\n")


def model_view(args):
    source = args.model.resolve()
    dest = args.out / (
        "model-native-mtp-no-eos"
        if args.engine == "omlx" and args.mode.startswith("mtp")
        else "model-no-eos"
    )
    dest.mkdir(exist_ok=True)
    for p in source.iterdir():
        q = dest / p.name
        if q.exists():
            continue
        if p.name in ("generation_config.json", "tokenizer_config.json"):
            value = json.loads(p.read_text())
            if p.name == "generation_config.json":
                value["eos_token_id"] = []
            else:
                value["eos_token"] = None
            q.write_text(json.dumps(value, indent=2) + "\n")
        else:
            q.symlink_to(p)
    if args.engine == "omlx" and args.mode.startswith("mtp"):
        mtp_source = args.mtp.resolve()
        mtp_config = json.loads((mtp_source / "config.json").read_text())
        mtp_index = json.loads(
            (mtp_source / "model.safetensors.index.json").read_text()
        )
        main_index = json.loads((source / "model.safetensors.index.json").read_text())
        for filename in set(mtp_index["weight_map"].values()):
            alias = dest / ("mtp-native-" + filename)
            if not alias.exists():
                alias.symlink_to(mtp_source / filename)
        main_index["weight_map"].update(
            {
                "language_model.mtp." + name: "mtp-native-" + filename
                for name, filename in mtp_index["weight_map"].items()
            }
        )
        main_index["metadata"]["total_size"] += mtp_index["metadata"]["total_size"]
        index_path = dest / "model.safetensors.index.json"
        index_path.unlink()
        index_path.write_text(json.dumps(main_index, indent=2) + "\n")
        config = json.loads((source / "config.json").read_text())
        config["text_config"]["mtp_num_hidden_layers"] = mtp_config["text_config"][
            "mtp_num_hidden_layers"
        ]
        quant = mtp_config["quantization"]
        for name in mtp_index["weight_map"]:
            if name.endswith(".scales"):
                path = name.removesuffix(".scales")
                config["quantization"]["language_model.mtp." + path] = {
                    key: quant.get(path, {}).get(key, quant[key])
                    for key in ("bits", "group_size", "mode")
                }
        if "quantization_config" in config:
            config["quantization_config"] = config["quantization"].copy()
        config_path = dest / "config.json"
        config_path.unlink()
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    return dest


def provenance(args):
    gpus = json.loads(
        subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"], text=True
        )
    )["SPDisplaysDataType"]
    names = (
        ["omlx", "mlx", "mlx-lm", "mlx-vlm"]
        if args.engine == "omlx"
        else (
            ["tensorfold", "mlx", "mlx-lm"]
            if args.engine == "tensorfold"
            else ["grpcio"]
        )
    )
    record = dict(
        engine=args.engine,
        mode=args.mode,
        versions={n: importlib.metadata.version(n) for n in names},
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.root, text=True
        ).strip(),
        source_dirty=subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=args.root, text=True
        ),
        model=str(args.model.resolve()),
        mtp=str(args.mtp.resolve()),
        dflash=str(args.dflash.resolve()),
        main_metadata_sha256=digest(
            json.loads((args.model / "config.json").read_text())
        ),
        mtp_metadata_sha256=digest(json.loads((args.mtp / "config.json").read_text())),
        dflash_metadata_sha256=digest(
            json.loads((args.dflash / "config.json").read_text())
        ),
        launcher_sha256=(
            hashlib.sha256(
                (args.root / "target/release/qwen3_5_dense").read_bytes()
            ).hexdigest()
            if args.engine == "psi-dec"
            else None
        ),
        hardware=dict(
            gpu=[
                dict(name=gpu["sppci_model"], cores=int(gpu["sppci_cores"]))
                for gpu in gpus
            ],
            machine=platform.machine(),
            os=platform.mac_ver()[0],
            cpu=subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip(),
            memory_bytes=int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            ),
        ),
        kv_cache="E4M3FN FP8" if args.engine == "psi-dec" else "BF16",
        candidate_policy=(
            "fixed chain 7; full vocabulary" if args.mode == "dflash2" else args.mode
        ),
        mtp_backend=(
            "native embedded head from supplied MTP source; context copy disabled"
            if args.engine == "omlx" and args.mode.startswith("mtp")
            else None
        ),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        chunk=args.chunk,
        prefix_chunk=args.prefix_chunk,
        mlx_cache_gib=args.mlx_cache_gib if args.engine != "psi-dec" else None,
        omlx_hot_cache_gib=args.omlx_hot_cache_gib,
        cache_pages_per_request=args.cache_pages_per_request,
        shared_prefix=args.shared_prefix,
        output_tokens=args.output_tokens,
        warmup=args.warmup,
        repeats=args.repeats,
        full_repeats=args.full_repeats,
        seed=42,
        temperature=0,
        eos="disabled in benchmark metadata only",
        metric="normal wall clock; no force sync profiling",
        command=sys.argv,
    )
    path = args.out / f"{args.engine}-{args.mode}-provenance.json"
    if path.exists():
        old = json.loads(path.read_text())
        # CLI subsets may change on resume. Workload and implementation must not.
        assert {k: v for k, v in old.items() if k != "command"} == {
            k: v for k, v in record.items() if k != "command"
        }, "benchmark identity changed; select a new output directory"
    else:
        path.write_text(json.dumps(record, indent=2) + "\n")


class PSI:
    def __init__(self, args, view, req):
        import grpc
        from grpc_tools import protoc

        proto = args.root / "crates/inference-runtime-proto/proto"
        assert (
            protoc.main(
                [
                    "grpc_tools.protoc",
                    "-I" + str(proto),
                    "--python_out=" + str(args.out),
                    "--grpc_python_out=" + str(args.out),
                    str(proto / "inference_runtime.proto"),
                ]
            )
            == 0
        )
        sys.path.insert(0, str(args.out))
        import inference_runtime_pb2 as pb
        import inference_runtime_pb2_grpc as rpc

        self.pb = pb
        self.grpc = grpc
        self.address = f"127.0.0.1:{args.port}"
        command = [
            str(args.root / "target/release/qwen3_5_dense"),
            "--hf-model-dir",
            str(view),
            "--grpc-listen-addr",
            self.address,
            "--http-listen-addr",
            f"127.0.0.1:{args.port+1}",
            "--max-requests",
            str(req),
            "--max-tokens",
            str(req * args.chunk),
            "--max-tokens-per-request",
            str(args.chunk),
            "--num-cache-pages",
            str(req * args.cache_pages_per_request),
            "--logging",
            "debug",
        ]
        if args.mode != "off":
            mtp = args.mode.startswith("mtp")
            k = int(args.mode[-1]) if mtp else 7
            command += [
                "--hf-spec-model-dir",
                str(args.mtp if mtp else args.dflash),
                "--spec-type",
                "mtp" if mtp else "dflash2",
                "--num-spec-tokens",
                str(k),
            ]
        self.log_path = args.out / f"psi-dec-{args.mode}-server.log"
        self.log = self.log_path.open("ab")
        self.process = subprocess.Popen(
            command, cwd=args.root, stdout=self.log, stderr=subprocess.STDOUT
        )
        try:
            channel = grpc.insecure_channel(self.address)
            grpc.channel_ready_future(channel).result(timeout=240)
            channel.close()
            lanes = int(args.mode[-1]) + 1 if args.mode.startswith("mtp") else 1
            self.worker_sha256 = hashlib.sha256(
                (args.root / f"target/release/qwen3_5_dense_lanes_{lanes}").read_bytes()
            ).hexdigest()
            self.args = args
            self.rpc = rpc
        except BaseException:
            self.close()
            raise

    async def batch(self, prompts, limit, ctx):
        start = time.perf_counter()
        offset = self.log_path.stat().st_size

        async def one(ids):
            async with self.grpc.aio.insecure_channel(self.address) as channel:
                stub = self.rpc.InferenceRuntimeStub(channel)
                tokens = []
                first = None
                rid = None
                req = self.pb.GenerateTokensRequest(
                    tokens=ids,
                    generation=self.pb.GenerationConfig(
                        max_sampled_tokens=limit,
                        temperature=0,
                        top_k=1,
                        top_p=1,
                        seed=42,
                    ),
                )
                async for response in stub.GenerateTokens(req, timeout=1800):
                    rid = response.request_id
                    if response.HasField("chunk"):
                        if first is None:
                            first = time.perf_counter()
                        tokens.extend(response.chunk.tokens)
                    else:
                        assert response.completion.num_output_tokens == limit
                assert len(tokens) == limit
                return dict(
                    tokens=tokens, first=first, end=time.perf_counter(), rid=rid
                )

        try:
            outputs = await asyncio.gather(*(one(ids) for ids in prompts))
        except self.grpc.aio.AioRpcError as error:
            with self.log_path.open("rb") as f:
                f.seek(offset)
                text = f.read().decode(errors="replace")
            if "kIOGPUCommandBufferCallbackErrorOutOfMemory" in text:
                raise RuntimeError(
                    "[METAL] Insufficient Memory: psi-dec replay submission"
                ) from error
            raise
        with self.log_path.open("rb") as f:
            f.seek(offset)
            text = f.read().decode(errors="replace")
        # The first submitted Main row proves the actual prefix-cache boundary.
        for output in outputs:
            matches = re.findall(
                rf'req_id={output["rid"]} req_slot=\d+ kind=[\w-]+ epoch=\d+ token_index=(\d+)',
                text,
            )
            assert matches, "server request trace is missing"
            output["cached"] = int(matches[0])
        perf = "\n".join(
            line for line in text.splitlines() if 'phase="executor.batch.perf"' in line
        )
        proposed = sum(map(int, re.findall(r"num_spec_tokens=(\d+)", perf)))
        accepted = sum(map(int, re.findall(r"num_verified_tokens=(\d+)", perf)))
        result = measurement(start, outputs, ctx, proposed, accepted)
        result["worker_sha256"] = self.worker_sha256
        return result

    async def prime(self, prefixes, lookahead=None):
        if self.args.shared_prefix:
            prefixes = prefixes[:1]
            if lookahead is not None:
                lookahead = lookahead[:1]
        if prefixes and prefixes[0]:
            # The extra unmeasured token publishes the full prefix boundary.
            await self.batch(
                [
                    p + (lookahead[i] if lookahead is not None else [123])
                    for i, p in enumerate(prefixes)
                ],
                1,
                None,
            )

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.log.close()


class OMLX:
    @classmethod
    async def create(cls, args, view, req):
        import mlx.core as mx
        from omlx.engine.vlm import VLMBatchedEngine
        from omlx.scheduler import SchedulerConfig
        from omlx.model_settings import ModelSettings
        from omlx.engine_core import get_mlx_executor

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), mx.set_cache_limit, args.mlx_cache_gib * 1024**3
        )
        self = cls()
        self.args = args
        config = SchedulerConfig(
            max_num_seqs=req,
            completion_batch_size=req,
            embedding_batch_size=req,
            max_num_batched_tokens=req * args.chunk,
            prefill_step_size=args.chunk,
            paged_cache_block_size=2048,
            decode_fairness=False,
            model_name=str(view),
            model_path=str(view),
            paged_ssd_cache_dir=str(args.out / f"omlx-{args.mode}-cache"),
            hot_cache_only=True,
            hot_cache_max_size=args.omlx_hot_cache_gib * 1024**3,
        )
        settings = ModelSettings(
            mtp_enabled=args.mode.startswith("mtp"),
            mtp_fixed_depth=int(args.mode[-1]) if args.mode.startswith("mtp") else None,
            dflash_enabled=args.mode == "dflash2",
            dflash_draft_model=str(args.dflash),
            dflash_block_size=8,
            dflash_verify_mode="dflash",
        )
        self.engine = VLMBatchedEngine(
            str(view), scheduler_config=config, model_settings=settings
        )
        original_load = mx.load

        def load(path, *a, **kw):
            values = original_load(path, *a, **kw)
            if Path(path).name.startswith("mtp-native-"):
                return {
                    "language_model.mtp." + name: value
                    for name, value in values.items()
                }
            return values

        from omlx.patches.mlx_lm_mtp import norm_repair

        original_repair = norm_repair.repair_legacy_head_norms

        def same_norms(weights):
            values, count = original_repair(weights)
            assert count == 0, "native loader repaired shared MTP norms; weights differ"
            return values, count

        mx.load = load
        norm_repair.repair_legacy_head_norms = same_norms
        try:
            await self.engine.start()
        finally:
            mx.load = original_load
            norm_repair.repair_legacy_head_norms = original_repair
        if args.mode == "dflash2":
            from omlx.speculative.dflash_drafter import load_dflash_drafter

            drafter = await loop.run_in_executor(
                get_mlx_executor(),
                load_dflash_drafter,
                str(args.dflash),
                self.engine.vlm_model,
            )
            self.engine.set_dflash_drafter(drafter)
        if args.mode != "off":
            from omlx.patches.mlx_lm_mtp import batch_generator as bg

            if args.mode.startswith("mtp"):
                # Fixed-depth MTP must use the supplied head for every draft.
                # Native prompt lookup can replace it with a wider copy window.
                self.original_copy = bg._context_copy_drafts
                bg._context_copy_drafts = lambda gen_batch, state, committed: []
            self.native_verify = []
            self.original_verify = bg._run_verify_cycle_chain
            depth = int(args.mode[-1]) if args.mode.startswith("mtp") else 7

            def verify(gen_batch, state, **kwargs):
                count = int(state.drafts.shape[0])
                assert count == depth, ("candidate count", count, depth)
                before = state.stats.accepts
                result = self.original_verify(gen_batch, state, **kwargs)
                self.native_verify.append((state.stats.accepts - before, count))
                return result

            bg._run_verify_cycle_chain = verify
        return self

    async def batch(self, prompts, limit, ctx, store=False):
        from omlx.request import SamplingParams

        start = time.perf_counter()
        verify_start = len(self.native_verify) if self.args.mode != "off" else 0
        params = SamplingParams(
            max_tokens=limit, temperature=0, top_k=1, top_p=1, seed=42
        )

        async def one(ids):
            output = await self.engine._engine.generate(
                ids, params, skip_cache_store=not store, benchmark_trace=True
            )

            assert len(output.output_token_ids) == limit, (
                "output length mismatch",
                limit,
                output.completion_tokens,
            )
            return dict(
                tokens=output.output_token_ids,
                first=output.first_token_at,
                end=time.perf_counter(),
                prefill_chunks=output.benchmark_prefill_chunks,
                cached=output.cached_tokens,
            )

        outputs = await asyncio.gather(*(one(ids) for ids in prompts))
        stats = self.native_verify[verify_start:] if self.args.mode != "off" else []
        result = measurement(
            start,
            outputs,
            ctx,
            sum(p for a, p in stats) if stats else None,
            sum(a for a, p in stats) if stats else None,
        )
        if self.args.mode.startswith("mtp") and not stats and "status" not in result:
            return dict(status="N/A: MTP fallback", excluded_timing=result)
        return result

    async def prime(self, prefixes):
        if self.args.shared_prefix:
            prefixes = prefixes[:1]
        if prefixes and prefixes[0]:
            # A different end token keeps the primer out of measured suffixes.
            config = self.engine._engine.engine.scheduler.config
            config.prefill_step_size = self.args.prefix_chunk
            try:
                await self.batch([p + [123] for p in prefixes], 1, None, store=True)
            finally:
                config.prefill_step_size = self.args.chunk

    async def close(self):
        await self.engine.stop()
        if self.args.mode != "off":
            from omlx.patches.mlx_lm_mtp import batch_generator as bg

            bg._run_verify_cycle_chain = self.original_verify
            if self.args.mode.startswith("mtp"):
                bg._context_copy_drafts = self.original_copy


class TensorFold:
    def __init__(self, args, view, req):
        import mlx.core as mx
        from tensorfold.families.qwen3_5 import (
            load_lane_model,
            install_row_decoder,
            load_drafter,
        )
        from tensorfold.families.qwen3_5.family import Qwen35Family
        from tensorfold.engine.lane_engine import LaneEngine
        from tensorfold.engine.prefill_plan import PrefillPlan

        mx.set_cache_limit(args.mlx_cache_gib * 1024**3)
        model, _ = load_lane_model(view)
        assert install_row_decoder(model)
        drafter = None
        if args.mode == "dflash2":
            os.environ["TF_DRAFT_VOCAB"] = "full"
            # TensorFold's vendored loader expects BF16. The benchmark supplies
            # the existing affine checkpoint without requantizing its values.
            import mlx.nn as nn
            from tensorfold.drafters.dflash_drafter import _vendor

            vendor = _vendor()
            original_class = vendor.DFlash2DraftModel
            original_load = mx.load
            quant = json.loads((args.dflash / "config.json").read_text())[
                "quantization"
            ]
            keys = json.loads(
                (args.dflash / "model.safetensors.index.json").read_text()
            )["weight_map"]
            paths = {
                name.removesuffix(".scales")
                for name in keys
                if name.endswith(".scales")
            }

            class AffineDraft(original_class):
                def __init__(self, config):
                    super().__init__(config)

                    def predicate(path, module):
                        if path not in paths:
                            return False
                        return {
                            k: quant.get(path, {}).get(k, quant[k])
                            for k in ("bits", "group_size", "mode")
                        }

                    nn.quantize(
                        self,
                        group_size=quant["group_size"],
                        bits=quant["bits"],
                        mode=quant["mode"],
                        class_predicate=predicate,
                    )

            def load(path, *a, **kw):
                values = original_load(path, *a, **kw)
                if Path(path).parent.resolve() == args.dflash.resolve():
                    for name in ("predecessor_codebook", "successor_codebook"):
                        key = "candidate_selector." + name
                        values[key] = values.pop(key + ".weight")
                return values

            vendor.DFlash2DraftModel = AffineDraft
            mx.load = load
            try:
                drafter = load_drafter(model, str(args.dflash), drafter_bits=0)
            finally:
                vendor.DFlash2DraftModel = original_class
                mx.load = original_load
            # The CPU chain selector reads codebook.weight directly. Expand
            # the same quantized embedding values once, outside timing.
            selector = drafter.model.candidate_selector
            for name in ("predecessor_codebook", "successor_codebook"):
                book = getattr(selector, name)
                plain = nn.Embedding(
                    book.weight.shape[0], book.weight.shape[1] * 32 // book.bits
                )
                plain.weight = book(mx.arange(book.weight.shape[0]))
                mx.eval(plain.weight)
                setattr(selector, name, plain)
        # Keep the published family row limit. A fixed eight-row verification
        # window limits each shared forward to batch_rows // 8 streams.
        self.family = Qwen35Family(
            model, drafter=drafter, nodes=7, rows=True, widest=8, first_copy_rows=1
        )
        if drafter is not None:
            self.family.head_drafts.chains = True
            self.family.max_streams = min(
                self.family.max_streams, self.family.batch_rows // 8
            )
        self.engine = LaneEngine(
            self.family,
            max_rows=128,
            max_draft=7,
            prefill_plan=PrefillPlan(args.chunk),
            prefill_pass=1,
        )
        if drafter is not None:
            # Fixed-policy comparison: retain production drafting, verify,
            # rollback, and commit. Disable historical acceptance allocation.
            self.engine._depth = lambda stream: 7
            self.engine._draft_budgets = lambda streams: [7] * len(streams)
            self.engine._allocate = lambda plans: None
        self.args = args
        self.prefixes = []
        self.prefix_ids = []
        mx.eval(model.parameters())

    async def prime(self, prefixes):
        from tensorfold.engine.prefill_plan import PrefillPlan

        self.engine.prefill_plan = PrefillPlan(self.args.prefix_chunk)
        caches = []
        unique = {}
        try:
            for i, ids in enumerate(prefixes):
                identity = tuple(ids)
                if not ids:
                    cache = None
                elif identity in unique:
                    cache = unique[identity]
                else:
                    previous = self.prefix_ids[i] if self.prefix_ids else []
                    cached_tokens = len(previous)
                    cache = None
                    if previous and ids[:cached_tokens] == previous:
                        cache = self.engine.copy_single_cache(self.prefixes[i])
                    else:
                        cached_tokens = 0
                    cache = self.engine.prefill_prefix(
                        ids, cache=cache, cached_tokens=cached_tokens
                    )
                    unique[identity] = cache
                caches.append(cache)
        finally:
            self.engine.prefill_plan = PrefillPlan(self.args.chunk)
        self.prefixes = caches
        self.prefix_ids = prefixes

    async def batch(self, prompts, limit, ctx):
        import mlx.core as mx
        from tensorfold.engine.lane_engine import LaneStream

        start = time.perf_counter()
        live = []
        outputs = []
        for i, ids in enumerate(prompts):
            stream = LaneStream(
                str(i), ids, limit, drafts=self.args.mode != "off", retain=False
            )
            cache = (
                self.engine.copy_single_cache(self.prefixes[i]) if ctx else None
            )
            self.engine.add_stream(stream, cache=cache, cached_tokens=ctx)
            assert stream.cached_tokens == ctx
            live.append(stream)
            outputs.append(
                dict(
                    tokens=stream.emitted,
                    first=None,
                    end=None,
                    cached=stream.cached_tokens,
                )
            )
        while self.engine.active_count:
            self.engine.step()
            for i, stream in enumerate(live):
                if outputs[i]["first"] is None and stream.emitted:
                    outputs[i]["first"] = time.perf_counter()
                if stream.finished and outputs[i]["end"] is None:
                    outputs[i]["end"] = time.perf_counter()
        mx.synchronize()
        for i, stream in enumerate(live):
            assert len(stream.emitted) == limit
            if outputs[i]["end"] is None:
                outputs[i]["end"] = time.perf_counter()
        result = measurement(
            start,
            outputs,
            ctx,
            sum(s.drafted for s in live),
            sum(s.accepted for s in live),
        )
        self.engine.reset()
        self.engine.release_rounds()
        return result

    async def close(self):
        self.engine.reset()
        self.prefixes.clear()
        self.prefix_ids.clear()
        self.engine.release_rounds()


def measurement(start, outputs, ctx, proposed, accepted):
    first = max(o["first"] for o in outputs)
    end = max(o["end"] for o in outputs)
    assert start <= first <= end
    cached = [o["cached"] for o in outputs]
    if ctx is not None:
        assert all(c <= ctx for c in cached), "prefix cache exceeds requested boundary"
        if cached != [ctx] * len(outputs):
            return dict(
                status="N/A: prefix not retained",
                available_prefix_tokens=cached,
                requested_context=ctx,
                excluded_ttft_ms=(first - start) * 1000,
                excluded_total_ms=(end - start) * 1000,
            )
    return dict(
        ttft_ms=(first - start) * 1000,
        decode_ms=(end - first) * 1000,
        total_ms=(end - start) * 1000,
        cached_tokens=cached,
        output_tokens=[len(o["tokens"]) for o in outputs],
        output_sha256=digest([o["tokens"] for o in outputs]),
        proposed=proposed,
        accepted=accepted,
        prefill_chunks=[o.get("prefill_chunks") for o in outputs],
    )


async def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    provenance(args)
    view = model_view(args)
    done = {
        r["key"]
        for r in {r["key"]: r for r in read_rows(args.out)}.values()
        if r["status"] not in ("N/A: Metal OOM", "unusable: swap activity")
    }
    planned = [s for s in shapes(args) if key(args.engine, args.mode, s) not in done]
    if not planned:
        return
    if args.engine == "tensorfold" and args.mode.startswith("mtp"):
        for s in planned:
            write_row(
                args,
                dict(
                    key=key(args.engine, args.mode, s),
                    engine=args.engine,
                    mode=args.mode,
                    shape=s,
                    status="N/A: MTP unsupported",
                ),
            )
        return
    if args.engine == "psi-dec" and args.mode.startswith("mtp"):
        depth = int(args.mode[-1])
        unavailable = [s for s in planned if s[3] < depth]
        for s in unavailable:
            write_row(
                args,
                dict(
                    key=key(args.engine, args.mode, s),
                    engine=args.engine,
                    mode=args.mode,
                    shape=s,
                    status="N/A: MTP lookahead",
                    required_lookahead_tokens=depth,
                ),
            )
        planned = [s for s in planned if s not in unavailable]
    cfg = json.loads((args.model / "config.json").read_text())
    vocab = cfg.get("text_config", cfg)["vocab_size"]
    for req in args.requests:
        if not any(s[0] == req for s in planned):
            continue
        engine = (
            PSI(args, view, req)
            if args.engine == "psi-dec"
            else (
                await OMLX.create(args, view, req)
                if args.engine == "omlx"
                else TensorFold(args, view, req)
            )
        )
        try:
            for kind, ctx in dict.fromkeys(
                (s[1], s[2]) for s in planned if s[0] == req
            ):
                group = [s for s in planned if s[:3] == (req, kind, ctx)]
                failed = group
                prefixes = [
                    [token(0 if args.shared_prefix else r, p, vocab) for p in range(ctx)]
                    for r in range(req)
                ]
                if args.engine == "psi-dec" and args.mode.startswith("mtp"):
                    await engine.prime(
                        prefixes,
                        [
                            [
                                token(0 if args.shared_prefix else r, ctx + j, vocab)
                                for j in range(int(args.mode[-1]))
                            ]
                            for r in range(req)
                        ],
                    )
                else:
                    await engine.prime(prefixes)
                for shape in group:
                    failed = [shape]
                    samples = []
                    input_hashes = []
                    warmup = 0 if kind == "full" else args.warmup
                    repeats = args.full_repeats if kind == "full" else args.repeats
                    for iteration in range(warmup + repeats):
                        prompts = streams(
                            req, ctx, shape[3], iteration, vocab, args.shared_prefix
                        )
                        result = await engine.batch(prompts, args.output_tokens, ctx)
                        result["input_sha256"] = digest(prompts)
                        if iteration >= warmup:
                            if "status" in result:
                                break
                            samples.append(result)
                            input_hashes.append(result["input_sha256"])
                    if "status" in result:
                        write_row(
                            args,
                            dict(
                                key=key(args.engine, args.mode, shape),
                                engine=args.engine,
                                mode=args.mode,
                                shape=shape,
                                **result,
                            ),
                        )
                        continue
                    row = dict(
                        key=key(args.engine, args.mode, shape),
                        engine=args.engine,
                        mode=args.mode,
                        shape=shape,
                        status="measured",
                        samples=samples,
                        input_sha256=input_hashes,
                        **{
                            m: statistics.median(s[m] for s in samples)
                            for m in ("ttft_ms", "decode_ms", "total_ms")
                        },
                    )
                    write_row(args, row)
        except RuntimeError as error:
            if "[METAL]" not in str(error) or "Insufficient Memory" not in str(error):
                raise
            for shape in failed:
                write_row(
                    args,
                    dict(
                        key=key(args.engine, args.mode, shape),
                        engine=args.engine,
                        mode=args.mode,
                        shape=shape,
                        status="N/A: Metal OOM",
                        error=str(error),
                    ),
                )
            # Exit this GPU resource owner and stop the controller.
            # Resume only after the memory budget is corrected.
            return 86
        finally:
            if args.engine == "psi-dec":
                engine.close()
            else:
                await engine.close()
            del engine
            gc.collect()


def swap_counters():
    text = subprocess.check_output(["vm_stat"], text=True)
    return {
        name: int(value)
        for name, value in re.findall(r"^(Swapins|Swapouts):\s+(\d+)\.", text, re.M)
    }


def guarded_worker(args, command):
    # The supervisor remains outside the GPU owner. Native calls can block its
    # Python event loop during loading, prefix setup, and model execution.
    ledger = args.out / "results.jsonl"
    clean_offset = ledger.stat().st_size if ledger.exists() else 0
    previous = swap_counters()
    swapped = False
    with (args.out / f"{args.engine}-{args.mode}-memory.jsonl").open("a") as history:
        history.write(
            json.dumps(dict(time=time.time(), command=command, **previous)) + "\n"
        )
        history.flush()
        process = subprocess.Popen(command, cwd=args.root, start_new_session=True)
        try:
            while True:
                offset = ledger.stat().st_size if ledger.exists() else 0
                current = swap_counters()
                history.write(
                    json.dumps(dict(time=time.time(), pid=process.pid, **current)) + "\n"
                )
                history.flush()
                if any(
                    current[name] > previous[name] for name in ("Swapins", "Swapouts")
                ):
                    swapped = True
                    break
                clean_offset = offset
                previous = current
                if process.poll() is not None:
                    break
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                except ProcessLookupError:
                    # The process group can exit between poll and signal.
                    process.wait()
        if swapped:
            # Keep raw evidence. Exclude completed rows since the last clean
            # observation, including any row written during worker teardown.
            if ledger.exists():
                with (
                    ledger.open("rb") as raw,
                    (args.out / "timing-exclusions.jsonl").open("a") as exclusions,
                ):
                    raw.seek(clean_offset)
                    for line in raw:
                        row = json.loads(line)
                        if row["status"] == "measured":
                            exclusions.write(
                                json.dumps(dict(
                                    key=row["key"], result_offset=row.get("result_offset"),
                                    status="unusable: swap activity",
                                    before=previous, after=current,
                                )) + "\n"
                            )
            report(args)
            raise RuntimeError(
                "Swap activity detected. Timings in that observation interval "
                "are unusable. Worker stopped."
            )
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--engine",
        choices=["psi-dec", "omlx", "tensorfold", "all", "report", "plan"],
        required=True,
    )
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--omlx-python", type=Path)
    p.add_argument("--tensorfold-python", type=Path)
    p.add_argument(
        "--mode",
        choices=["off", "mtp1", "mtp2", "mtp3", "mtp4", "dflash2"],
        default="off",
    )
    p.add_argument(
        "--modes", nargs="+", default=["off", "mtp1", "mtp2", "mtp3", "mtp4", "dflash2"]
    )
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument(
        "--model", type=Path, default=Path.home() / "Workspace/models/Qwen3.8-27B-4bit"
    )
    p.add_argument(
        "--mtp",
        type=Path,
        default=Path.home() / "Workspace/models/Qwen3.8-27B-MTP-4bit",
    )
    p.add_argument(
        "--dflash",
        type=Path,
        default=Path.home() / "Workspace/models/Qwen3.8-27B-DFlash2-affine",
    )
    p.add_argument("--requests", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--contexts", type=int, nargs="*", default=CONTEXTS)
    p.add_argument("--tokens", type=int, nargs="+", default=TOKENS)
    p.add_argument("--full", type=int, nargs="*", default=FULL)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--prefix-chunk", type=int, default=2048)
    p.add_argument(
        "--shared-prefix",
        action="store_true",
        help="prepare one identical prefix for all requests",
    )
    p.add_argument("--mlx-cache-gib", type=int, default=1)
    p.add_argument("--omlx-hot-cache-gib", type=int, default=20)
    p.add_argument("--cache-pages-per-request", type=int, default=98304)
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--full-repeats", type=int, default=1)
    p.add_argument("--port", type=int, default=50071)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out = args.out.resolve()
    assert (
        args.chunk > 0
        and args.prefix_chunk > 0
        and args.mlx_cache_gib >= 0
        and args.omlx_hot_cache_gib > 0
        and args.cache_pages_per_request > 0
        and args.output_tokens > 0
        and args.repeats > 0
        and args.full_repeats > 0
        and args.warmup >= 0
    )
    assert all(0 < r <= 8 for r in args.requests) and all(
        c >= 0 and c % 2048 == 0 for c in args.contexts
    )
    assert all(q > 0 for q in args.tokens + args.full)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.engine == "report":
        report(args)
    elif args.engine == "all":
        assert (
            args.omlx_python and args.tensorfold_python
        ), "--engine all requires both engine Python paths"
        for mode in args.modes:
            for engine in ("psi-dec", "omlx", "tensorfold"):
                python = (
                    args.tensorfold_python
                    if engine == "tensorfold"
                    else args.omlx_python
                )
                command = [
                    str(python),
                    str(Path(__file__).resolve()),
                    "--engine",
                    engine,
                    "--mode",
                    mode,
                    "--root",
                    str(args.root),
                    "--model",
                    str(args.model),
                    "--mtp",
                    str(args.mtp),
                    "--dflash",
                    str(args.dflash),
                    "--chunk",
                    str(args.chunk),
                    "--prefix-chunk",
                    str(args.prefix_chunk),
                    "--mlx-cache-gib",
                    str(args.mlx_cache_gib),
                    "--omlx-hot-cache-gib",
                    str(args.omlx_hot_cache_gib),
                    "--cache-pages-per-request",
                    str(args.cache_pages_per_request),
                    "--output-tokens",
                    str(args.output_tokens),
                    "--warmup",
                    str(args.warmup),
                    "--repeats",
                    str(args.repeats),
                    "--full-repeats",
                    str(args.full_repeats),
                    "--port",
                    str(args.port),
                    "--out",
                    str(args.out),
                    "--requests",
                    *map(str, args.requests),
                    "--contexts",
                    *map(str, args.contexts),
                    "--tokens",
                    *map(str, args.tokens),
                    "--full",
                    *map(str, args.full),
                    "--modes",
                    *args.modes,
                ]
                if args.shared_prefix:
                    command.append("--shared-prefix")
                print(json.dumps({"command": command}), flush=True)
                command.append("--worker")
                # Use the child's identity for the memory observation log.
                worker_args = argparse.Namespace(**vars(args))
                worker_args.engine, worker_args.mode = engine, mode
                guarded_worker(worker_args, command)
        report(args)
    elif args.engine == "plan":
        print(
            json.dumps(
                {
                    "shapes": len(shapes(args)),
                    "points": len(shapes(args)) * len(args.modes) * 3,
                    "workload": shapes(args),
                }
            )
        )
    elif args.worker:
        logging.basicConfig(level=logging.INFO)
        sys.exit(asyncio.run(run(args)))
    else:
        guarded_worker(args, [
            sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"
        ])


if __name__ == "__main__":
    main()
