#!/usr/bin/env python3
"""Build and run the standalone GQA matrix binary. All matrix logic lives in Rust."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    build = subprocess.Popen(
        ["cargo", "build", "--release", "-p", "inference-backend-metal", "--example",
         "gqa-sdpa-bench", "--message-format=json"],
        cwd=ROOT, stdout=subprocess.PIPE, text=True,
    )
    executable = None
    for line in build.stdout:
        event = json.loads(line)
        if event.get("reason") == "compiler-artifact" and event.get("target", {}).get("name") == "gqa-sdpa-bench":
            executable = event.get("executable")
    if build.wait() or not executable:
        raise SystemExit("GQA benchmark build failed")
    os.execv(executable, [executable, *sys.argv[1:]])


if __name__ == "__main__":
    main()
