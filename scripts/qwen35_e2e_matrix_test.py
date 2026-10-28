"""CPU-only workload identity contracts for the serving matrix."""

import asyncio
import importlib.util
import copy
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "matrix", Path(__file__).with_name("qwen35_e2e_matrix.py")
)
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class MatrixTest(unittest.TestCase):
    def test_shared_prefix_keeps_request_suffixes_independent(self):
        prompts = matrix.streams(4, 2048, 8, 0, 248320, shared_prefix=True)
        for ids in prompts:
            self.assertEqual(ids[:2052], prompts[0][:2052])
        self.assertEqual(len({tuple(ids[2052:]) for ids in prompts}), 4)

    def test_tensorfold_prefix_reuse_preserves_state_and_identity(self):
        calls = []
        assert_equal = self.assertEqual

        class Engine:
            prefill_plan = None

            @staticmethod
            def copy_single_cache(cache):
                return copy.deepcopy(cache)

            def prefill_prefix(self, ids, *, cache, cached_tokens):
                calls.append((tuple(ids), cached_tokens))
                if cached_tokens:
                    assert_equal(cache["tokens"], ids[:cached_tokens])
                return {"tokens": ids.copy(), "gdn_position": len(ids)}

        adapter = matrix.TensorFold.__new__(matrix.TensorFold)
        adapter.engine = Engine()
        adapter.args = SimpleNamespace(prefix_chunk=2, chunk=1)
        adapter.prefixes = []
        adapter.prefix_ids = []
        module = SimpleNamespace(PrefillPlan=lambda size: size)
        with patch.dict(sys.modules, {"tensorfold.engine.prefill_plan": module}):
            asyncio.run(adapter.prime([[1, 2], [1, 2]]))
            original = adapter.prefixes[0]
            self.assertIs(adapter.prefixes[0], adapter.prefixes[1])
            asyncio.run(adapter.prime([[1, 2, 3, 4], [1, 2, 3, 4]]))
            self.assertEqual(original, {"tokens": [1, 2], "gdn_position": 2})
            asyncio.run(adapter.prime([[1], [1]]))
            asyncio.run(adapter.prime([[7, 8], [7, 8]]))
        self.assertEqual(calls, [((1, 2), 0), ((1, 2, 3, 4), 2), ((1,), 0), ((7, 8), 0)])
        self.assertEqual(adapter.engine.prefill_plan, 1)

    def test_swap_after_worker_completion_excludes_last_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ledger = path / "results.jsonl"
            ledger.write_text(json.dumps(dict(key="earlier", status="measured")) + "\n")

            class Process:
                pid = 123
                returncode = None

                def poll(self):
                    return self.returncode

                def wait(self, timeout):
                    with ledger.open("a") as f:
                        f.write(json.dumps(dict(key="latest", status="measured")) + "\n")
                    self.returncode = 0

            args = SimpleNamespace(out=path, root=path, engine="khalamari", mode="off")
            with (
                patch.object(matrix.subprocess, "Popen", return_value=Process()),
                patch.object(matrix, "swap_counters", side_effect=[
                    dict(Swapins=100, Swapouts=200),
                    dict(Swapins=100, Swapouts=200),
                    dict(Swapins=101, Swapouts=200),
                ]),
                patch.object(matrix, "report"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Swap activity detected"):
                    matrix.guarded_worker(args, ["worker"])
            exclusions = [json.loads(l) for l in (path / "timing-exclusions.jsonl").read_text().splitlines()]
            self.assertEqual([r["key"] for r in exclusions], ["latest"])
            self.assertEqual(len(matrix.read_rows(path)), 2)

    def test_existing_swap_counts_do_not_reject_a_clean_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            process = SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)
            args = SimpleNamespace(out=path, root=path, engine="khalamari", mode="off")
            with (
                patch.object(matrix.subprocess, "Popen", return_value=process),
                patch.object(matrix, "swap_counters", return_value=dict(Swapins=100, Swapouts=200)),
            ):
                matrix.guarded_worker(args, ["worker"])
            self.assertFalse((path / "timing-exclusions.jsonl").exists())

    def test_resume_retries_oom_and_swap_without_excluding_clean_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text('{"vocab_size":248320}')
            args = SimpleNamespace(
                out=path, model=path, engine="khalamari", mode="off", modes=["off"],
                requests=[1], contexts=[0], tokens=[8, 16, 24, 32], full=[],
                shared_prefix=False, warmup=0, repeats=1, output_tokens=2,
            )
            rows = [
                dict(key=matrix.key("khalamari", "off", (1, "append", 0, q)), status=status)
                for q, status in [(8, "measured"), (16, "N/A: Metal OOM"), (24, "measured")]
            ]
            ledger = path / "results.jsonl"
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
            last = dict(key=matrix.key("khalamari", "off", (1, "append", 0, 32)), status="measured")
            with patch.object(matrix, "report"):
                matrix.write_row(args, last.copy())
                matrix.write_row(args, last)
            (path / "timing-exclusions.jsonl").write_text(
                json.dumps(dict(key=rows[2]["key"], status="unusable: swap activity")) + "\n"
                + json.dumps(dict(key=last["key"], result_offset=last["result_offset"],
                                  status="unusable: swap activity")) + "\n"
            )
            calls = []

            class Engine:
                async def prime(self, prefixes):
                    pass

                async def batch(self, prompts, limit, ctx):
                    calls.append(len(prompts[0]))
                    return dict(ttft_ms=1, decode_ms=1, total_ms=2, output_tokens=[limit])

                def close(self):
                    pass

            with (
                patch.object(matrix, "provenance"),
                patch.object(matrix, "model_view"),
                patch.object(matrix, "report"),
                patch.object(matrix, "KHALAMARI", return_value=Engine()) as engine,
            ):
                asyncio.run(matrix.run(args))
                self.assertEqual(calls, [16, 24, 32])
                self.assertTrue(all(
                    row["status"] == "measured"
                    for row in {row["key"]: row for row in matrix.read_rows(path)}.values()
                ))
                engine.reset_mock()
                asyncio.run(matrix.run(args))
                engine.assert_not_called()
            self.assertEqual(json.loads(ledger.read_text().splitlines()[1])["status"], "N/A: Metal OOM")

    def test_complete_shape_contract(self):
        args = SimpleNamespace(
            requests=[1, 2, 4, 8],
            contexts=matrix.CONTEXTS,
            tokens=matrix.TOKENS,
            full=matrix.FULL,
        )
        shapes = matrix.shapes(args)
        self.assertEqual(len(shapes), 232)
        self.assertEqual(len(set(shapes)), 232)
        self.assertEqual(sum(s[1] == "append" for s in shapes), 216)
        self.assertEqual(sum(s[1] == "full" for s in shapes), 16)

    def test_only_prefix_is_shared_between_samples_and_shapes(self):
        first = matrix.streams(2, 2048, 8, 0, 248320)
        next_sample = matrix.streams(2, 2048, 8, 1, 248320)
        next_shape = matrix.streams(2, 2048, 16, 0, 248320)
        for r in range(2):
            self.assertEqual(first[r][:2052], next_sample[r][:2052])
            self.assertEqual(first[r][:2048], next_shape[r][:2048])
            self.assertNotEqual(first[r][2048:], next_sample[r][2048:])
            self.assertNotEqual(first[r][2048:], next_shape[r][2048:2056])
        self.assertNotEqual(first[0], first[1])

    def test_mtp_proposal_count_is_distinct_from_full_prefill_shape(self):
        shape = (4, "append", 2048, 8)
        self.assertNotEqual(
            matrix.key("omlx", "mtp1", shape), matrix.key("omlx", "mtp4", shape)
        )
        self.assertNotEqual(
            matrix.key("khalamari", "off", (1, "full", 0, 2048)),
            matrix.key("khalamari", "off", (1, "append", 0, 2048)),
        )

    def test_mtp_short_input_is_excluded_before_engine_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text('{"vocab_size":248320}')
            args = SimpleNamespace(
                shared_prefix=False,
                out=path,
                model=path,
                engine="khalamari",
                mode="mtp2",
                modes=["mtp2"],
                requests=[1, 2, 4, 8],
                contexts=[0, 2048],
                tokens=[1],
                full=[],
            )
            with (
                patch.object(matrix, "provenance"),
                patch.object(matrix, "model_view"),
                patch.object(matrix, "KHALAMARI") as engine,
            ):
                asyncio.run(matrix.run(args))
            engine.assert_not_called()
            rows = matrix.read_rows(path)
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(r["status"] == "N/A: MTP lookahead" for r in rows))
            self.assertTrue(all(r["required_lookahead_tokens"] == 2 for r in rows))

    def test_report_rejects_unmatched_work(self):
        shape = (1, "append", 2048, 8)
        for mismatch in ("input", "output", "swap"):
            with self.subTest(
                mismatch=mismatch
            ), tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(
                    out=Path(directory),
                    requests=[1],
                    contexts=[2048],
                    tokens=[8],
                    full=[],
                    modes=["off"],
                )
                rows = []
                for engine in ("khalamari", "omlx"):
                    rows.append(
                        dict(
                            key=matrix.key(engine, "off", shape),
                            status="measured",
                            input_sha256=[
                                (
                                    "other"
                                    if engine == "omlx" and mismatch == "input"
                                    else "same"
                                )
                            ],
                            samples=[
                                dict(
                                    output_tokens=[
                                        (
                                            16
                                            if engine == "omlx" and mismatch == "output"
                                            else 32
                                        )
                                    ]
                                )
                            ],
                            total_ms=100,
                            ttft_ms=20,
                            decode_ms=80,
                        )
                    )
                (args.out / "results.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                if mismatch == "swap":
                    (args.out / "timing-exclusions.jsonl").write_text(json.dumps(dict(
                        key=matrix.key("omlx", "off", shape),
                        status="unusable: swap activity",
                    )) + "\n")
                    matrix.report(args)
                    for name in ("report.md", "ttft.md", "decode.md"):
                        text = (args.out / name).read_text()
                        self.assertIn("unusable: swap activity | pending | — | — |", text)
                else:
                    with self.assertRaises(AssertionError):
                        matrix.report(args)

    def test_cache_miss_is_excluded_from_hot_prefix_timings(self):
        output = dict(first=10.02, end=10.08, tokens=[1] * 32, cached=2048)
        result = matrix.measurement(10, [output], 4096, 7, 1)
        self.assertEqual(result["status"], "N/A: prefix not retained")
        self.assertEqual(result["available_prefix_tokens"], [2048])
        self.assertNotIn("total_ms", result)
        self.assertAlmostEqual(result["excluded_total_ms"], 80)

    def test_tensorfold_ttft_includes_prefill_before_token_emission(self):
        stream = SimpleNamespace(
            emitted=[], finished=False, cached_tokens=0, drafted=0, accepted=0
        )
        clock = SimpleNamespace(now=10, step=0)
        engine = object.__new__(matrix.TensorFold)
        engine.args = SimpleNamespace(mode="off")

        def step():
            clock.step += 1
            clock.now = {1: 11, 2: 12, 3: 15}[clock.step]
            if clock.step >= 2:
                stream.emitted.append(clock.step)
            if clock.step == 3:
                stream.finished = True
                engine.engine.active_count = 0

        engine.engine = SimpleNamespace(
            active_count=1,
            add_stream=lambda *a, **kw: None,
            step=step,
            reset=lambda: None,
            release_rounds=lambda: None,
        )
        mx = SimpleNamespace(synchronize=lambda: None)
        with (
            patch.dict(
                "sys.modules",
                {
                    "mlx.core": mx,
                    "mlx": SimpleNamespace(core=mx),
                    "tensorfold.engine.lane_engine": SimpleNamespace(
                        LaneStream=lambda *a, **kw: stream
                    ),
                },
            ),
            patch.object(matrix.time, "perf_counter", side_effect=lambda: clock.now),
        ):
            result = asyncio.run(engine.batch([[1]], 2, 0))
        self.assertEqual(result["ttft_ms"], 2000)
        self.assertEqual(result["decode_ms"], 3000)
        self.assertEqual(result["total_ms"], 5000)

    def test_resume_rejects_changed_mlx_allocator_budget(self):
        def probe(command, **kwargs):
            if command[0] == "system_profiler":
                return json.dumps(
                    {
                        "SPDisplaysDataType": [
                            {"sppci_model": "GPU", "sppci_cores": "40"}
                        ]
                    }
                )
            if command[0] == "git":
                return "commit\n" if command[1] == "rev-parse" else ""
            return (
                "CPU\n"
                if command[-1] == "machdep.cpu.brand_string"
                else str(48 * 1024**3)
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            args = SimpleNamespace(
                shared_prefix=False,
                engine="omlx",
                mode="off",
                root=root,
                model=root,
                mtp=root,
                dflash=root,
                out=root,
                chunk=128,
                prefix_chunk=2048,
                mlx_cache_gib=1,
                omlx_hot_cache_gib=20,
                cache_pages_per_request=98304,
                output_tokens=32,
                warmup=1,
                repeats=2,
                full_repeats=1,
            )
            with (
                patch.object(matrix.subprocess, "check_output", side_effect=probe),
                patch.object(matrix.importlib.metadata, "version", return_value="test"),
            ):
                matrix.provenance(args)
                matrix.provenance(args)
                args.mlx_cache_gib = 2
                with self.assertRaisesRegex(
                    AssertionError, "benchmark identity changed"
                ):
                    matrix.provenance(args)

    def test_native_mtp_view_preserves_sources_and_head_quantization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, mtp, out = (root / name for name in ("model", "mtp", "out"))
            for path in (model, mtp, out):
                path.mkdir()
            main_config = {
                "text_config": {"mtp_num_hidden_layers": 0},
                "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
            }
            head_config = {
                "text_config": {"mtp_num_hidden_layers": 1},
                "quantization": {
                    "bits": 8,
                    "group_size": 128,
                    "mode": "affine",
                    "fc": {"bits": 4},
                },
            }
            main_index = {
                "metadata": {"total_size": 10},
                "weight_map": {
                    "language_model.model.embed_tokens.weight": "model.safetensors"
                },
            }
            head_index = {
                "metadata": {"total_size": 5},
                "weight_map": {
                    name: "model.safetensors"
                    for name in ("fc.weight", "fc.scales", "fc.biases", "norm.weight")
                },
            }
            for path, config, index in (
                (model, main_config, main_index),
                (mtp, head_config, head_index),
            ):
                (path / "config.json").write_text(json.dumps(config))
                (path / "model.safetensors.index.json").write_text(json.dumps(index))
                (path / "model.safetensors").touch()
            args = SimpleNamespace(
                engine="omlx", mode="mtp1", model=model, mtp=mtp, out=out
            )
            for _ in range(2):
                view = matrix.model_view(args)
                index = json.loads((view / "model.safetensors.index.json").read_text())
                config = json.loads((view / "config.json").read_text())
                self.assertEqual(index["metadata"]["total_size"], 15)
                self.assertEqual(len(index["weight_map"]), 5)
                self.assertEqual(
                    (
                        view / index["weight_map"]["language_model.mtp.fc.weight"]
                    ).resolve(),
                    (mtp / "model.safetensors").resolve(),
                )
                self.assertEqual(config["text_config"]["mtp_num_hidden_layers"], 1)
                self.assertEqual(
                    config["quantization"]["language_model.mtp.fc"],
                    {"bits": 4, "group_size": 128, "mode": "affine"},
                )
                self.assertEqual(
                    json.loads((model / "config.json").read_text()), main_config
                )
                self.assertEqual(
                    json.loads((model / "model.safetensors.index.json").read_text()),
                    main_index,
                )
                self.assertEqual(
                    json.loads((mtp / "config.json").read_text()), head_config
                )
            args.mode = "off"
            main_view = matrix.model_view(args)
            self.assertNotEqual(main_view, view)
            self.assertEqual(
                json.loads((main_view / "model.safetensors.index.json").read_text()),
                main_index,
            )
            self.assertEqual(
                json.loads((main_view / "config.json").read_text()), main_config
            )

    def test_mtp_fallback_is_not_a_speculative_timing(self):
        engine = object.__new__(matrix.OMLX)
        engine.args = SimpleNamespace(mode="mtp1")
        engine.native_verify = []

        async def generate(*args, **kwargs):
            return SimpleNamespace(
                output_token_ids=[1] * 32,
                completion_tokens=32,
                first_token_at=time.perf_counter(),
                benchmark_prefill_chunks=[],
                cached_tokens=4096,
            )

        engine.engine = SimpleNamespace(_engine=SimpleNamespace(generate=generate))
        with patch.dict(
            "sys.modules",
            {"omlx.request": SimpleNamespace(SamplingParams=lambda **values: values)},
        ):
            result = asyncio.run(engine.batch([[1]], 32, 4096))
        self.assertEqual(result["status"], "N/A: MTP fallback")
        self.assertNotIn("total_ms", result)
        self.assertEqual(result["excluded_timing"]["output_tokens"], [32])

    def test_speculation_counts_use_complete_verify_windows(self):
        for mode in ("mtp2", "dflash2"):
            with self.subTest(mode=mode):
                engine = object.__new__(matrix.OMLX)
                engine.args = SimpleNamespace(mode=mode)
                # Earlier requests must not contribute to this sample.
                engine.native_verify = [(1, 7)]

                async def generate(*args, **kwargs):
                    width = 2 if mode == "mtp2" else 7
                    engine.native_verify.extend([(1, width), (2, width), (0, width)])
                    return SimpleNamespace(
                        output_token_ids=[1] * 32,
                        completion_tokens=32,
                        first_token_at=time.perf_counter(),
                        benchmark_prefill_chunks=[],
                        cached_tokens=4096,
                    )

                engine.engine = SimpleNamespace(
                    _engine=SimpleNamespace(generate=generate)
                )
                with patch.dict(
                    "sys.modules",
                    {"omlx.request": SimpleNamespace(SamplingParams=lambda **kw: kw)},
                ):
                    result = asyncio.run(engine.batch([[1]], 32, 4096))
                self.assertEqual(result["proposed"], 6 if mode == "mtp2" else 21)
                self.assertEqual(result["accepted"], 3)

    def test_khalamari_oom_is_distinct_from_other_rpc_failures(self):
        class RPCError(Exception):
            pass

        class Channel:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        for message, expected in (
            ("kIOGPUCommandBufferCallbackErrorOutOfMemory", RuntimeError),
            ("Socket closed", RPCError),
        ):
            with self.subTest(
                message=message
            ), tempfile.TemporaryDirectory() as directory:
                engine = object.__new__(matrix.KHALAMARI)
                engine.log_path = Path(directory) / "server.log"
                # An earlier failure must not classify a later RPC failure.
                engine.log_path.write_text(
                    "kIOGPUCommandBufferCallbackErrorOutOfMemory\n"
                )

                async def generate(request, timeout):
                    with engine.log_path.open("a") as log:
                        log.write(message + "\n")
                    raise RPCError("Socket closed")
                    yield

                engine.grpc = SimpleNamespace(
                    aio=SimpleNamespace(
                        insecure_channel=lambda address: Channel(),
                        AioRpcError=RPCError,
                    )
                )
                engine.rpc = SimpleNamespace(
                    InferenceRuntimeStub=lambda channel: SimpleNamespace(
                        GenerateTokens=generate
                    )
                )
                engine.pb = SimpleNamespace(
                    GenerateTokensRequest=lambda **values: values,
                    GenerationConfig=lambda **values: values,
                )
                engine.address = "test"
                with self.assertRaises(expected):
                    asyncio.run(engine.batch([[1]], 1, None))


if __name__ == "__main__":
    unittest.main()
