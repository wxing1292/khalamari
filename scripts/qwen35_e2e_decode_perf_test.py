"""CPU-only contracts for the Qwen performance helper. No server or GPU runs."""

import json
import os
import re
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("qwen35_e2e_decode_perf.sh").read_text()


def section(start, end):
    return SCRIPT.split(start, 1)[1].split(end, 1)[0]


def function(name):
    start = SCRIPT.index(name + "() {")
    return SCRIPT[start : SCRIPT.index("\n}\n", start) + 3]


def bash(code, **env):
    return subprocess.run(
        ["/bin/bash", "-euc", code],
        text=True,
        capture_output=True,
        env={**os.environ, **env},
        check=True,
    ).stdout


class PerfScriptTest(unittest.TestCase):
    def cases(self, cases, override=""):
        selection = "IFS=, read -r -a selected_cases" + section(
            "IFS=, read -r -a selected_cases", "require_dir() {"
        )
        return bash(
            function("require_positive_integer")
            + selection
            + '\nprintf "%s\\n" "${selected_cases[@]}"',
            CASES=cases,
            BLOCK_SPEC_TOKENS=override,
        ).splitlines()

    def test_default_matrix(self):
        default = re.search(r'^CASES="(.*)"$', SCRIPT, re.M)[1]
        expected = ["27b_off", "35b_off"]
        expected += [f"{model}_mtp{k}" for k in range(1, 5) for model in ("27b", "35b")]
        expected += [
            f"{model}_{mode}"
            for model in ("27b", "35b")
            for mode in ("dflash2",)
        ]
        self.assertEqual(self.cases(default), expected)

    def test_explicit_sweep_and_override_deduplication(self):
        for model in ("27b", "35b"):
            self.assertEqual(
                self.cases(f"{model}_mtp"), [f"{model}_mtp{k}" for k in range(1, 5)]
            )
            self.assertEqual(
                self.cases(f"{model}_on"),
                [f"{model}_mtp{k}" for k in range(1, 5)]
                + [f"{model}_dflash2"],
            )
            for mode in ("dflash2",):
                sweep = [f"{model}_{mode}_k{k}" for k in (1, 2, 3, 4, 8)]
                self.assertEqual(self.cases(",".join(sweep)), sweep)
                self.assertEqual(
                    self.cases(",".join([f"{model}_{mode}"] + sweep), "2"), [sweep[1]]
                )

    def test_invalid_counts_and_cases(self):
        for count in ("0", "00", "-1", "abc", ""):
            with (
                self.subTest(count=count),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                bash(
                    function("require_positive_integer")
                    + '\nrequire_positive_integer K "$VALUE"',
                    VALUE=count,
                )
        with self.assertRaises(subprocess.CalledProcessError):
            self.cases("27b_mtp5")

    def launch(self, case, dflash2):
        functions = "\n".join(
            function(name)
            for name in ("run_mtp_case", "run_block_spec_case", "run_named_case")
        )
        stub = "run_server_case() { python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' \"$@\"; }"
        return json.loads(
            bash(
                functions + "\n" + stub + "\nrun_named_case " + shlex.quote(case),
                TOKENIZER="",
                PORT="50061",
                NUM_CACHE_PAGES="393216",
                MAX_REQUESTS="4",
                MAX_TOKENS="128",
                MAX_TOKENS_PER_REQUEST="64",
                MODEL_27B="main",
                MODEL_35B="main",
                MTP_27B="mtp",
                MTP_35B="mtp",
                DFLASH2_27B=dflash2,
                DFLASH2_35B=dflash2,
            )
        )

    def test_checkpoint_defaults_and_explicit_counts_reach_server(self):
        with tempfile.TemporaryDirectory() as directory:
            dflash2 = Path(directory) / "dflash2"
            dflash2.mkdir()
            (dflash2 / "config.json").write_text(
                json.dumps({"dflash_config": {"block_size": 4}})
            )
            for model in ("27b", "35b"):
                for mode in ("dflash2",):
                    for suffix, k in [("", 3)] + [
                        (f"_k{k}", k) for k in (1, 2, 3, 4, 8)
                    ]:
                        args = self.launch(
                            f"{model}_{mode}{suffix}", str(dflash2)
                        )
                        self.assertEqual(args[0], f"{model}_{mode}_k{k}")
                        self.assertEqual(args[1], "256 1024")
                        self.assertEqual(
                            args[args.index("--num-spec-tokens") + 1], str(k)
                        )
                for k in range(1, 5):
                    args = self.launch(f"{model}_mtp{k}", str(dflash2))
                    self.assertEqual(args[1], "256 1024")
                    self.assertEqual(args[args.index("--num-spec-tokens") + 1], str(k))

    def test_reference_contains_only_reported_rows(self):
        ref = function("reference_row")
        rows = re.findall(r'gpu_cores:([^)]*)\) echo "([^"]*)"', ref)
        self.assertEqual(len(rows), 44)
        self.assertFalse(any(":384:" in key for key, _ in rows))
        self.assertFalse(any("_k8:" in key for key, _ in rows))
        self.assertIn(("27b_mtp4:1024:chat", "18.693|2.197|1024|1853|558"), rows)
        self.assertIn(("35b_mtp4:1024:chat", "105.720|2.592|1024|1573|629"), rows)
        self.assertIn(
            ("27b_mtp2:256:gsm8k_typing_average", "41.780|2.747|239|172|152"), rows
        )
        self.assertIn(("35b_mtp3:1024:chat", "117.661|2.421|1024|1266|601"), rows)
        self.assertTrue(all(len(values.split("|")) == 5 for _, values in rows))

    def summary(self, **changes):
        code = "import os\nimport statistics\n" + section(
            "import os\nimport statistics\n", "\nPY"
        )
        env = {
            "VALS": "44.277 44.277",
            "INPUTS": "137 137",
            "CHUNKS": "87 87",
            "SAMPLES": "249 249",
            "TTFTS": "1 1",
            "PROMPT_RATES": "1 1",
            "INTER_CHUNK_P50S": "1 1",
            "INTER_CHUNK_P95S": "1 1",
            "PROPOSED_SPECS": "172 172",
            "VERIFIED_SPECS": "162 162",
            "ACCEPTANCE_RATES": "0.94 0.94",
            "TOKENS_PER_CHUNKS": "2.862069 2.862069",
            "SPEC_BY_INDEX": "- -",
            "VERIFIED_BY_INDEX": "- -",
            "LABEL": "27b_mtp2",
            "TOKENS": "256",
            "PROMPT_ID": "gsm8k_typing_average",
            "OUTPUT_SHA256": "test",
            "REFERENCE_DECODE": "44.277",
            "REFERENCE_TOKENS_PER_CHUNK": "2.862",
            "REFERENCE_SAMPLED": "249",
            "REFERENCE_PROPOSED_SPEC": "172",
            "REFERENCE_VERIFIED_SPEC": "162",
            "REFERENCE_STATUS": "summary-only",
            "REFERENCE_MISMATCH": "",
        }
        env.update(changes)
        return subprocess.run(
            ["python3", "-c", code],
            env={**os.environ, **env},
            text=True,
            capture_output=True,
            check=True,
        ).stdout

    def test_summary_comparison_does_not_invent_missing_metrics(self):
        summary = self.summary()
        self.assertIn("reference_status=summary-only", summary)
        self.assertIn("observed_decode_delta_pct=+0.00", summary)
        self.assertNotIn("reference_ttft", summary)
        self.assertNotIn("ttft_delta", summary)
        for changes in (
            {"SAMPLES": "248 248"},
            {"REFERENCE_STATUS": "config-mismatch"},
            {"REFERENCE_STATUS": "no-reference-row", "REFERENCE_DECODE": ""},
        ):
            self.assertNotIn("observed_decode_delta_pct", self.summary(**changes))


if __name__ == "__main__":
    unittest.main()
