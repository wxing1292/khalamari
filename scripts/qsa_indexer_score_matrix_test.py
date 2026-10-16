"""CPU-only contracts for the QSA experiment plan and selection report."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "matrix", Path(__file__).with_name("qsa_indexer_score_matrix.py")
)
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class MatrixTest(unittest.TestCase):
    def test_supervise_reports_swap_and_worker_exit(self):
        before = dict(Swapins=100, Swapouts=200)
        swapin = dict(Swapins=104, Swapouts=200)
        swapout = dict(Swapins=100, Swapouts=201)
        cases = (
            (
                "swapin",
                None,
                -15,
                [before, swapin, swapin],
                "swap activity: Swapins +4, Swapouts +0",
            ),
            (
                "swapout",
                None,
                -15,
                [before, swapout, swapout],
                "swap activity: Swapins +0, Swapouts +1",
            ),
            (
                "swap_at_exit",
                0,
                0,
                [before, swapin],
                "swap activity: Swapins +4, Swapouts +0",
            ),
            ("worker_error", 1, 1, [before, before], "worker exited with code 1"),
            ("success", 0, 0, [before, before], "completed"),
        )
        for name, poll, code, counters, reason in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                proc = mock.Mock()
                proc.poll.return_value = poll
                proc.wait.return_value = code
                with mock.patch.object(
                    matrix, "swap", side_effect=counters
                ), mock.patch.object(matrix.subprocess, "Popen", return_value=proc):
                    status = matrix.supervise(
                        ["worker"],
                        Path(directory) / "worker.log",
                        Path(directory) / "memory.jsonl",
                    )
                self.assertEqual(status["reason"], reason)
                self.assertEqual(status["usable"], name == "success")
                self.assertEqual(status["exit_code"], code)
                self.assertEqual(proc.terminate.call_count, int(poll is None))

    def test_dense_budget_crossing_counts_only_sparse_queries(self):
        case = matrix.make_case("append", 2, 8, 2048)
        self.assertEqual(case["sparse_queries"], 5)
        self.assertEqual(case["total_sparse_queries"], 10)
        self.assertEqual(case["groups"], 514)
        self.assertEqual(case["score_bytes"], 10 * 514 * 4)
        self.assertEqual(matrix.make_case("append", 1, 128, 0)["sparse_queries"], 0)

    def test_small_query_matrix_preserves_request_segmentation(self):
        plan = matrix.cases("small-q", [8], [4096], [])
        measured = [c for c in plan if c["workload"] == "small_q"]
        self.assertEqual(
            {(c["requests"], c["tokens"]) for c in measured},
            {(r, q) for r in (1, 2, 4) for q in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16)},
        )
        # Equal flattened counts do not identify the same request-local Q/K work.
        self.assertNotEqual(
            matrix.make_case("append", 1, 8, 4096)["id"],
            matrix.make_case("append", 2, 4, 4096)["id"],
        )

    def test_growth_reuses_identical_append_measurements(self):
        plan = matrix.cases("prefill", [1], [4096], [128])
        same = [
            c
            for c in plan
            if (c["requests"], c["tokens"], c["history"]) == (1, 128, 4096)
        ]
        self.assertEqual(len(same), 1)
        self.assertEqual(set(same[0]["uses"]), {"append", "growth"})

    def test_selection_uses_a_shape_threshold_on_unseen_shapes(self):
        points = []
        for q in (16, 24, 32, 48, 128, 160, 192, 256):
            for g in (1024, 4096):
                case = dict(total_sparse_queries=q, sparse_queries=q, groups=g)
                times = {"small": 1 if q <= 48 else 3, "large": 3 if q <= 48 else 1}
                points.append(dict(case=case, times=times))
        tree = matrix.fit(points, ["small", "large"])
        self.assertEqual(
            matrix.predict(
                tree, dict(total_sparse_queries=40, sparse_queries=40, groups=2048)
            ),
            "small",
        )
        self.assertEqual(
            matrix.predict(
                tree, dict(total_sparse_queries=224, sparse_queries=224, groups=2048)
            ),
            "large",
        )

    def test_report_excludes_unusable_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            case = matrix.make_case("append", 1, 128, 4096)
            (out / "plan.json").write_text(
                json.dumps(
                    dict(
                        cases=[case],
                        variants=[dict(name="steel_flat_q8_g64_d16")],
                        requests=[1],
                    )
                )
            )
            dest = out / "cases" / case["id"]
            dest.mkdir(parents=True)
            (dest / "status.json").write_text(json.dumps(dict(usable=False)))
            (dest / "samples.jsonl").write_text(
                json.dumps(
                    dict(
                        kind="sample",
                        case_id=case["id"],
                        phase="score",
                        arm="steel_flat_q8_g64_d16",
                        us=1,
                    )
                )
                + "\n"
            )
            matrix.report(out)
            self.assertIn(
                "Completed sample records: 0. Unusable attempts: 1.",
                (out / "report.md").read_text(),
            )

    def test_report_keeps_fixed_and_heldout_selection_costs_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cases = [
                matrix.make_case("append", 1, q, h)
                for q in (32, 64, 128, 256)
                for h in (4096, 8192, 16384, 32768)
            ]
            arms = ["steel_flat_q8_g64_d16", "omlx_mlx_fp32_flat"]
            (out / "plan.json").write_text(
                json.dumps(
                    dict(
                        cases=cases, variants=[dict(name=a) for a in arms], requests=[1]
                    )
                )
            )
            for case in cases:
                dest = out / "cases" / case["id"]
                dest.mkdir(parents=True)
                (dest / "status.json").write_text(
                    json.dumps(dict(usable=True, mode="run"))
                )
                records = []
                for phase in ("score", "score_radix"):
                    for arm in arms:
                        records.append(
                            dict(
                                kind="sample",
                                case_id=case["id"],
                                phase=phase,
                                arm=arm,
                                us=(
                                    10
                                    if arm == arms[0]
                                    else (5 if case["tokens"] >= 128 else 20)
                                ),
                            )
                        )
                (dest / "samples.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records)
                )
            matrix.report(out)
            policies = json.loads((out / "selection.json").read_text())
            self.assertEqual(policies["score"]["heldout_shapes"], 8)
            self.assertEqual(policies["score"]["training_shapes"], 8)
            report = (out / "report.md").read_text()
            self.assertIn("Fixed solution", report)
            self.assertIn("Held-out selected path vs peer", report)
            self.assertIn("Remaining cases: 0", report)


if __name__ == "__main__":
    unittest.main()
