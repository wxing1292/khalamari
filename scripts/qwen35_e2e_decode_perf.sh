#!/usr/bin/env bash
set -euo pipefail

RUNS=7
BLOCK_SPEC_TOKENS=""
PORT=50061
BUILD=1
REFERENCE=1
SHOW_RUNS=0
# One fixed GSM8K prompt and one chat prompt.
PROMPT_SET="representative2"
PROMPT_IDS=(
    "gsm8k_typing_average"
    "chat"
)
PROMPTS=(
    "Jared is trying to increase his typing speed. He starts with 47 words per minute (WPM). After some lessons the next time he tests his typing speed it has increased to 52 WPM. If he continues to increase his typing speed once more by 5 words, what will be the average of the three measurements?
Please reason step by step, and put your final answer within \boxed{}."
    "你好，北京有什么好玩的景点？香山如何？早上去晚上去？单纯爬山么？还有什么可以在香山玩的？"
)
MODEL_ROOT="$HOME/Workspace/models"
TOKENIZER=""
MODEL_27B=""
MTP_27B=""
DFLASH2_27B=""
MODEL_35B=""
MTP_35B=""
DFLASH2_35B=""
MODEL_27B_REPO="mlx-community/Qwen3.8-27B-4bit"
MTP_27B_REPO="mlx-community/Qwen3.8-27B-MTP-4bit"
MODEL_35B_REPO="mlx-community/Qwen3.6-35B-A3B-4bit"
MTP_35B_REPO="mlx-community/Qwen3.6-35B-A3B-MTP-4bit"
# Block-spec cases use checkpoint defaults unless a case or CLI overrides K.
CASES="27b_off,35b_off,27b_mtp1,35b_mtp1,27b_mtp2,35b_mtp2,27b_mtp3,35b_mtp3,27b_mtp4,35b_mtp4,27b_dflash2,35b_dflash2"
CASE_COOLDOWN_SECS=8
LOGGING=info
SEED=42
TEMPERATURE=0.7
TOP_K=20
TOP_P=0.8
NUM_CACHE_PAGES=393216
MAX_REQUESTS=4
MAX_TOKENS=128
MAX_TOKENS_PER_REQUEST=64
CACHE_BLOCK_TOKENS=2048
# User-supplied compact results received on REFERENCE_DATE.
# The user identified main tip. The commit below is HEAD at receipt, before history cleanup.
# This reference precedes the SlicedQ pipeline-capacity fix. Dirty state was not supplied.
# Config checks below use the producing script's defaults, not recovered raw CONFIG rows.
# Run count, variance, TTFT, and exact token hashes were not supplied.
REFERENCE_MACHINE="apple_m3_max_40_gpu_cores"
REFERENCE_DATE="2026-09-19"
REFERENCE_COMMIT="d6dc5da7e6a4cbfea6a2b9654f577a22aa23415e"
REFERENCE_SOURCE_TREE="38f972e038150ee351c16dba6ae33bde7242670f"
REFERENCE_DIRTY="unknown"
REFERENCE_OS_VERSION="27.0"
REFERENCE_ARCH="arm64"
REFERENCE_NUM_CACHE_PAGES=393216
REFERENCE_CACHE_BLOCK_TOKENS=2048
REFERENCE_MAX_REQUESTS=4
REFERENCE_MAX_TOKENS=128
REFERENCE_MAX_TOKENS_PER_REQUEST=64
REFERENCE_CASE_COOLDOWN_SECS=8
REFERENCE_LOGGING="info"
REFERENCE_SEED=42
REFERENCE_TEMPERATURE=0.7
REFERENCE_TOP_K=20
REFERENCE_TOP_P=0.8
REFERENCE_RUNS="unknown"
REFERENCE_CASES="27b_off,27b_mtp1,27b_mtp2,27b_mtp3,27b_mtp4,27b_dflash2_k7,35b_off,35b_mtp1,35b_mtp2,35b_mtp3,35b_mtp4"
REFERENCE_PROMPT_SET="representative2"
REFERENCE_PROMPT_SET_SHA256="1821fb9bef6c6a310635c0eb99e8bd0dc305db33fcaedb8ca707b225b6e9b254"
REFERENCE_MODEL_27B_DIR_NAME="Qwen3.8-27B-4bit"
REFERENCE_MTP_27B_DIR_NAME="Qwen3.8-27B-MTP-4bit"
REFERENCE_DFLASH2_27B_DIR_NAME="Qwen3.8-27B-DFlash2-affine"
REFERENCE_MODEL_35B_DIR_NAME="Qwen3.6-35B-A3B-4bit"
REFERENCE_MTP_35B_DIR_NAME="Qwen3.6-35B-A3B-MTP-4bit"
ACTIVE_SERVER_PID=""
REPORT_FILE=""

cleanup_active_server() {
    if [[ -n "$ACTIVE_SERVER_PID" ]]; then
        kill "$ACTIVE_SERVER_PID" >/dev/null 2>&1 || true
        wait "$ACTIVE_SERVER_PID" >/dev/null 2>&1 || true
        ACTIVE_SERVER_PID=""
    fi
}

cleanup() {
    cleanup_active_server
    if [[ -n "$REPORT_FILE" ]]; then
        rm -f "$REPORT_FILE"
        REPORT_FILE=""
    fi
}

trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

usage() {
    cat <<'EOF'
Usage: scripts/qwen35_e2e_decode_perf.sh [options]

Runs Qwen3.5/3.6/3.8 replay e2e decode perf one server at a time and reports
decode throughput, TTFT, inter-chunk latency, and speculative acceptance.

Options:
  --runs N              Runs per prompt and token-count case. Default: 7
  --block-spec-tokens K  Generate K tokens in each DFlash2 proposal.
                        Default: checkpoint geometry. This also overrides each `_kN` case.
  --cases LIST          Comma-separated cases.
                        Default order:
                          27b_off,35b_off,
                          27b_mtp1,35b_mtp1,
                          27b_mtp2,35b_mtp2,
                          27b_mtp3,35b_mtp3,
                          27b_mtp4,35b_mtp4,
                          27b_dflash2,
                          35b_dflash2
                        Available modes: off, mtp, dflash2.
                        The *_mtp alias runs proposal counts 1, 2, 3, and 4.
                        Base DFlash2 cases use checkpoint geometry.
                        Add _k1/_k2/_k3/_k4/_k8 to request a proposal count.
                        --block-spec-tokens K overrides K for every block-spec case.
                        Group cases: 27b_on and 35b_on select all Spec modes.
                        Missing Spec checkpoints are warned and skipped.
  --port N              Server port. Default: 50061
  --prompt TEXT         Use one custom prompt instead of the default representative2 set.
  --seed N              Fixed request seed. Default: 42
  --temperature N       Sampling temperature. Default: 0.7
  --top-k N             Sampling top-k. Default: 20
  --top-p N             Sampling top-p. Default: 0.8
  --num-cache-pages N   Shared cache pages. Default: 393216
  --max-requests N      Scheduler request capacity. Default: 4
  --max-tokens N        Scheduler flattened-token capacity. Default: 128
  --max-tokens-per-request N
                        Scheduler per-request token capacity. Default: 64
  --model-root DIR      Default checkpoint root. Default: $HOME/Workspace/models
  --tokenizer DIR       Tokenizer/chat-template override. Default: each case's Main model.
  --model-27b DIR       27B Main directory. Default: MODEL_ROOT/Qwen3.8-27B-4bit
  --mtp-27b DIR         27B MTP directory. Default: MODEL_ROOT/Qwen3.8-27B-MTP-4bit
  --dflash2-27b DIR     27B affine DFlash2 directory. Default: MODEL_ROOT/Qwen3.8-27B-DFlash2-affine
  --model-35b DIR       35B Main directory. Default: MODEL_ROOT/Qwen3.6-35B-A3B-4bit
  --mtp-35b DIR         35B MTP directory. Default: MODEL_ROOT/Qwen3.6-35B-A3B-MTP-4bit
  --dflash2-35b DIR     35B affine DFlash2 directory. Default: MODEL_ROOT/Qwen3.6-35B-A3B-DFlash2-affine
  --model-27b-repo REPO Hugging Face repo used if the 27B Main model is missing.
  --mtp-27b-repo REPO   Hugging Face repo used if the 27B MTP model is missing.
  --model-35b-repo REPO Hugging Face repo used if the 35B Main model is missing.
  --mtp-35b-repo REPO   Hugging Face repo used if the 35B MTP model is missing.
  --no-build            Skip release build.
  --no-reference        Do not compare summaries with the checked-in reference run.
  --show-runs           Print each machine-readable RUN and SUMMARY row.
  --case-cooldown-secs N
                        Idle time between model cases. Default: 8
                        Pass 0 for an intentional sustained-load run.
  --logging LEVEL       Server logging: info or debug. Default: info.
                        Debug adds request/response and replay-stage details.
                        The benchmark always enables the executor perf DEBUG target.
  -h, --help            Show this help.

Examples:
  scripts/qwen35_e2e_decode_perf.sh \
    --model-root "$HOME/Workspace/models" \
    --cases 27b_off,35b_off,27b_mtp1,35b_mtp1,27b_dflash2_k8 \
    --runs 3

  scripts/qwen35_e2e_decode_perf.sh \
    --model-35b "$HOME/Workspace/models/Qwen3.6-35B-A3B-4bit" \
    --mtp-35b "$HOME/Workspace/models/Qwen3.6-35B-A3B-MTP-4bit" \
    --dflash2-35b "$HOME/Workspace/models/Qwen3.6-35B-A3B-DFlash2-affine" \
    --cases 35b_mtp1,35b_mtp2,35b_mtp3 --runs 3
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --runs)
        [[ $# -ge 2 ]] || {
            echo "--runs requires a value" >&2
            exit 2
        }
        RUNS="$2"
        shift 2
        ;;
    --cases)
        [[ $# -ge 2 ]] || {
            echo "--cases requires a value" >&2
            exit 2
        }
        CASES="$2"
        shift 2
        ;;
    --block-spec-tokens)
        [[ $# -ge 2 ]] || {
            echo "--block-spec-tokens requires a value" >&2
            exit 2
        }
        BLOCK_SPEC_TOKENS="$2"
        shift 2
        ;;
    --port)
        [[ $# -ge 2 ]] || {
            echo "--port requires a value" >&2
            exit 2
        }
        PORT="$2"
        shift 2
        ;;
    --prompt)
        [[ $# -ge 2 ]] || {
            echo "--prompt requires a value" >&2
            exit 2
        }
        PROMPT_SET="custom"
        PROMPT_IDS=("custom")
        PROMPTS=("$2")
        shift 2
        ;;
    --seed)
        [[ $# -ge 2 ]] || {
            echo "--seed requires a value" >&2
            exit 2
        }
        SEED="$2"
        shift 2
        ;;
    --temperature)
        [[ $# -ge 2 ]] || {
            echo "--temperature requires a value" >&2
            exit 2
        }
        TEMPERATURE="$2"
        shift 2
        ;;
    --top-k)
        [[ $# -ge 2 ]] || {
            echo "--top-k requires a value" >&2
            exit 2
        }
        TOP_K="$2"
        shift 2
        ;;
    --top-p)
        [[ $# -ge 2 ]] || {
            echo "--top-p requires a value" >&2
            exit 2
        }
        TOP_P="$2"
        shift 2
        ;;
    --num-cache-pages)
        [[ $# -ge 2 ]] || {
            echo "--num-cache-pages requires a value" >&2
            exit 2
        }
        NUM_CACHE_PAGES="$2"
        shift 2
        ;;
    --max-requests)
        [[ $# -ge 2 ]] || {
            echo "--max-requests requires a value" >&2
            exit 2
        }
        MAX_REQUESTS="$2"
        shift 2
        ;;
    --max-tokens)
        [[ $# -ge 2 ]] || {
            echo "--max-tokens requires a value" >&2
            exit 2
        }
        MAX_TOKENS="$2"
        shift 2
        ;;
    --max-tokens-per-request)
        [[ $# -ge 2 ]] || {
            echo "--max-tokens-per-request requires a value" >&2
            exit 2
        }
        MAX_TOKENS_PER_REQUEST="$2"
        shift 2
        ;;
    --model-root)
        [[ $# -ge 2 ]] || {
            echo "--model-root requires a value" >&2
            exit 2
        }
        MODEL_ROOT="$2"
        shift 2
        ;;
    --tokenizer)
        [[ $# -ge 2 ]] || {
            echo "--tokenizer requires a value" >&2
            exit 2
        }
        TOKENIZER="$2"
        shift 2
        ;;
    --model-27b)
        [[ $# -ge 2 ]] || {
            echo "--model-27b requires a value" >&2
            exit 2
        }
        MODEL_27B="$2"
        shift 2
        ;;
    --mtp-27b)
        [[ $# -ge 2 ]] || {
            echo "--mtp-27b requires a value" >&2
            exit 2
        }
        MTP_27B="$2"
        shift 2
        ;;
    --dflash2-27b)
        [[ $# -ge 2 ]] || {
            echo "--dflash2-27b requires a value" >&2
            exit 2
        }
        DFLASH2_27B="$2"
        shift 2
        ;;
    --model-35b)
        [[ $# -ge 2 ]] || {
            echo "--model-35b requires a value" >&2
            exit 2
        }
        MODEL_35B="$2"
        shift 2
        ;;
    --mtp-35b)
        [[ $# -ge 2 ]] || {
            echo "--mtp-35b requires a value" >&2
            exit 2
        }
        MTP_35B="$2"
        shift 2
        ;;
    --dflash2-35b)
        [[ $# -ge 2 ]] || {
            echo "--dflash2-35b requires a value" >&2
            exit 2
        }
        DFLASH2_35B="$2"
        shift 2
        ;;
    --model-27b-repo)
        [[ $# -ge 2 ]] || {
            echo "--model-27b-repo requires a value" >&2
            exit 2
        }
        MODEL_27B_REPO="$2"
        shift 2
        ;;
    --mtp-27b-repo)
        [[ $# -ge 2 ]] || {
            echo "--mtp-27b-repo requires a value" >&2
            exit 2
        }
        MTP_27B_REPO="$2"
        shift 2
        ;;
    --model-35b-repo)
        [[ $# -ge 2 ]] || {
            echo "--model-35b-repo requires a value" >&2
            exit 2
        }
        MODEL_35B_REPO="$2"
        shift 2
        ;;
    --mtp-35b-repo)
        [[ $# -ge 2 ]] || {
            echo "--mtp-35b-repo requires a value" >&2
            exit 2
        }
        MTP_35B_REPO="$2"
        shift 2
        ;;
    --no-build)
        BUILD=0
        shift
        ;;
    --no-reference)
        REFERENCE=0
        shift
        ;;
    --show-runs)
        SHOW_RUNS=1
        shift
        ;;
    --case-cooldown-secs)
        [[ $# -ge 2 ]] || {
            echo "--case-cooldown-secs requires a value" >&2
            exit 2
        }
        CASE_COOLDOWN_SECS="$2"
        shift 2
        ;;
    --logging)
        [[ $# -ge 2 ]] || {
            echo "--logging requires a value" >&2
            exit 2
        }
        LOGGING="$2"
        shift 2
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        echo "unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
done

require_positive_integer() {
    local option="$1"
    local value="$2"
    case "$value" in
    "" | *[!0-9]* | 0*)
        echo "$option expects a positive integer" >&2
        exit 2
        ;;
    esac
}

require_nonnegative_integer() {
    local option="$1"
    local value="$2"
    case "$value" in
    "" | *[!0-9]*)
        echo "$option expects a non-negative integer" >&2
        exit 2
        ;;
    esac
}

require_positive_integer "--runs" "$RUNS"
if [[ -n "$BLOCK_SPEC_TOKENS" ]]; then
    require_positive_integer "--block-spec-tokens" "$BLOCK_SPEC_TOKENS"
fi
require_positive_integer "--port" "$PORT"
require_positive_integer "--num-cache-pages" "$NUM_CACHE_PAGES"
require_positive_integer "--max-requests" "$MAX_REQUESTS"
require_positive_integer "--max-tokens" "$MAX_TOKENS"
require_positive_integer "--max-tokens-per-request" "$MAX_TOKENS_PER_REQUEST"
require_nonnegative_integer "--case-cooldown-secs" "$CASE_COOLDOWN_SECS"
require_nonnegative_integer "--seed" "$SEED"
require_positive_integer "--top-k" "$TOP_K"

if ! TEMPERATURE="$TEMPERATURE" TOP_P="$TOP_P" python3 - <<'PY'
import math
import os

for name in ("TEMPERATURE", "TOP_P"):
    try:
        value = float(os.environ[name])
    except ValueError as error:
        raise SystemExit(f"--{name.lower().replace('_', '-')} expects a number") from error
    if not math.isfinite(value):
        raise SystemExit(f"--{name.lower().replace('_', '-')} must be finite")
temperature = float(os.environ["TEMPERATURE"])
top_p = float(os.environ["TOP_P"])
if temperature < 0:
    raise SystemExit("--temperature must be non-negative")
if not 0 <= top_p <= 1:
    raise SystemExit("--top-p must be in [0, 1]")
PY
then
    exit 2
fi

if ((${#PROMPTS[@]} == 0 || ${#PROMPT_IDS[@]} != ${#PROMPTS[@]})); then
    echo "prompt IDs and prompts must define the same nonzero workload set" >&2
    exit 2
fi
for prompt_index in "${!PROMPTS[@]}"; do
    if [[ -z "${PROMPTS[$prompt_index]}" ]]; then
        echo "prompt ${PROMPT_IDS[$prompt_index]} must not be empty" >&2
        exit 2
    fi
done

case "$LOGGING" in
info | debug) ;;
*)
    echo "--logging must be info or debug" >&2
    exit 2
    ;;
esac

if ((PORT > 65535)); then
    echo "--port must be at most 65535" >&2
    exit 2
fi
if ((MAX_TOKENS_PER_REQUEST > MAX_TOKENS)); then
    echo "--max-tokens-per-request must not exceed --max-tokens" >&2
    exit 2
fi

[[ -n "$MODEL_27B" ]] || MODEL_27B="$MODEL_ROOT/Qwen3.8-27B-4bit"
[[ -n "$MTP_27B" ]] || MTP_27B="$MODEL_ROOT/Qwen3.8-27B-MTP-4bit"
[[ -n "$DFLASH2_27B" ]] || DFLASH2_27B="$MODEL_ROOT/Qwen3.8-27B-DFlash2-affine"
[[ -n "$MODEL_35B" ]] || MODEL_35B="$MODEL_ROOT/Qwen3.6-35B-A3B-4bit"
[[ -n "$MTP_35B" ]] || MTP_35B="$MODEL_ROOT/Qwen3.6-35B-A3B-MTP-4bit"
[[ -n "$DFLASH2_35B" ]] || DFLASH2_35B="$MODEL_ROOT/Qwen3.6-35B-A3B-DFlash2-affine"

IFS=, read -r -a selected_cases <<<"$CASES"
requested_cases=("${selected_cases[@]}")
selected_cases=()

append_case() {
    local candidate="$1"
    case "$candidate" in
    *_dflash2 | *_dflash2_k*)
        if [[ -n "$BLOCK_SPEC_TOKENS" ]]; then
            candidate="${candidate%_k*}_k${BLOCK_SPEC_TOKENS}"
        fi
        ;;
    esac
    local existing
    if ((${#selected_cases[@]})); then
        for existing in "${selected_cases[@]}"; do
            if [[ "$existing" == "$candidate" ]]; then
                return
            fi
        done
    fi
    selected_cases+=("$candidate")
}

for case_name in "${requested_cases[@]}"; do
    case "$case_name" in
    27b_mtp)
        append_case 27b_mtp1
        append_case 27b_mtp2
        append_case 27b_mtp3
        append_case 27b_mtp4
        ;;
    27b_dflash2)
        append_case 27b_dflash2
        ;;
    27b_on)
        append_case 27b_mtp1
        append_case 27b_mtp2
        append_case 27b_mtp3
        append_case 27b_mtp4
        append_case 27b_dflash2
        ;;
    35b_mtp)
        append_case 35b_mtp1
        append_case 35b_mtp2
        append_case 35b_mtp3
        append_case 35b_mtp4
        ;;
    35b_dflash2)
        append_case 35b_dflash2
        ;;
    35b_on)
        append_case 35b_mtp1
        append_case 35b_mtp2
        append_case 35b_mtp3
        append_case 35b_mtp4
        append_case 35b_dflash2
        ;;
    *) append_case "$case_name" ;;
    esac
done

for case_name in "${selected_cases[@]}"; do
    case "$case_name" in
    *_dflash2_k*)
        require_positive_integer "--cases $case_name" "${case_name##*_k}"
        ;;
    esac
done

need_27b=0
need_27b_mtp=0
need_27b_dflash2=0
need_35b=0
need_35b_mtp=0
need_35b_dflash2=0
for case_name in "${selected_cases[@]}"; do
    case "$case_name" in
    27b_off) need_27b=1 ;;
    27b_mtp1 | 27b_mtp2 | 27b_mtp3 | 27b_mtp4)
        need_27b=1
        need_27b_mtp=1
        ;;
    27b_dflash2 | 27b_dflash2_k*)
        need_27b=1
        need_27b_dflash2=1
        ;;
    35b_off) need_35b=1 ;;
    35b_mtp1 | 35b_mtp2 | 35b_mtp3 | 35b_mtp4)
        need_35b=1
        need_35b_mtp=1
        ;;
    35b_dflash2 | 35b_dflash2_k*)
        need_35b=1
        need_35b_dflash2=1
        ;;
    *)
        echo "unknown case: $case_name" >&2
        exit 2
        ;;
    esac
done
CASES="$(
    IFS=,
    printf '%s' "${selected_cases[*]}"
)"

if [[ ${#selected_cases[@]} -eq 0 ]]; then
    echo "--cases must include at least one case" >&2
    exit 2
fi

require_dir() {
    local option="$1"
    local dir="$2"
    if [[ -z "$dir" || ! -d "$dir" ]]; then
        echo "$option must name an existing directory" >&2
        exit 2
    fi
}


validate_affine_dflash2() {
    local option="$1"
    local dir="$2"
    DFLASH2_DIR="$dir" DFLASH2_OPTION="$option" python3 - <<'PY'
import json
import os
import struct
from pathlib import Path

option = os.environ["DFLASH2_OPTION"]
model_dir = Path(os.environ["DFLASH2_DIR"])
config_path = model_dir / "config.json"
try:
    config = json.loads(config_path.read_text())
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"unable to read {config_path}: {exc}") from exc

quantization = config.get("quantization")
if not isinstance(quantization, dict) or quantization.get("mode") != "affine":
    raise SystemExit(f"{option} must name an affine-quantized DFlash2 checkpoint")
if config.get("architectures") != ["DFlash2DraftModel"]:
    raise SystemExit(f"{config_path} must describe one DFlash2DraftModel")

for tensor_path in sorted(model_dir.glob("*.safetensors")):
    try:
        with tensor_path.open("rb") as stream:
            header_bytes = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_bytes))
    except (OSError, struct.error, json.JSONDecodeError) as exc:
        raise SystemExit(f"unable to inspect {tensor_path}: {exc}") from exc
    bf16_matrices = [name for name, metadata in header.items()
                     if name != "__metadata__"
                     and metadata.get("dtype") == "BF16"
                     and len(metadata.get("shape", [])) == 2
                     and not name.endswith(".scales")
                     and not name.endswith(".biases")]
    if bf16_matrices:
        raise SystemExit(
            f"{option} contains unquantized BF16 matrices in {tensor_path.name}; "
            "run qwen3x_spec_quantize dflash2 first"
        )
    invalid_affine = [name for name, metadata in header.items()
                      if name != "__metadata__"
                      and (name.endswith(".scales") or name.endswith(".biases"))
                      and metadata.get("dtype") != "BF16"]
    if invalid_affine:
        raise SystemExit(
            f"{option} contains non-BF16 affine parameters in {tensor_path.name}; "
            "regenerate it with qwen3x_spec_quantize dflash2"
        )
PY
}

model_present() {
    local dir="$1"
    [[ -f "$dir/config.json" ]] || return 1
    find "$dir" -type f \
        \( -name '*.safetensors' -o -name '*.npz' \) \
        -print -quit 2>/dev/null | grep -q .
}

ensure_model() {
    local repo="$1"
    local dir="$2"
    local -a download_command

    if model_present "$dir"; then
        echo "==> Found checkpoint: $dir"
        return
    fi

    if [[ -z "$repo" ]]; then
        echo "checkpoint is missing: $dir" >&2
        echo "Pass the matching checkpoint directory or configure its --*-repo option." >&2
        exit 1
    fi

    if command -v hf >/dev/null 2>&1; then
        download_command=(hf download)
    elif command -v huggingface-cli >/dev/null 2>&1; then
        download_command=(huggingface-cli download)
    else
        echo "checkpoint is missing: $dir" >&2
        echo 'Install the Hugging Face CLI with: python3 -m pip install -U "huggingface_hub[hf_xet]"' >&2
        exit 1
    fi

    echo "==> Downloading $repo -> $dir"
    mkdir -p "$dir"
    "${download_command[@]}" "$repo" --local-dir "$dir"

    if ! model_present "$dir"; then
        echo "download completed, but checkpoint is incomplete: $dir" >&2
        exit 1
    fi
}

ensure_optional_model() {
    local case_name="$1"
    local option="$2"
    local repo="$3"
    local dir="$4"

    if model_present "$dir"; then
        echo "==> Found checkpoint: $dir"
        return 0
    fi
    if [[ -z "$repo" ]]; then
        echo "WARNING: skipping $case_name because its checkpoint is missing: $dir" >&2
        echo "Pass $option DIR or ${option}-repo REPO to enable this case." >&2
        return 1
    fi
    ensure_model "$repo" "$dir"
}

ensure_optional_local_model() {
    local case_name="$1"
    local option="$2"
    local dir="$3"

    if ! model_present "$dir"; then
        echo "WARNING: skipping $case_name because its checkpoint is missing: $dir" >&2
        echo "Pass $option DIR to enable this case." >&2
        return 1
    fi
    echo "==> Found checkpoint: $dir"
}

if ((need_27b_mtp)); then
    ensure_optional_model 27b_mtp --mtp-27b "$MTP_27B_REPO" "$MTP_27B" || need_27b_mtp=0
fi
if ((need_27b_dflash2)); then
    ensure_optional_local_model 27b_dflash2 --dflash2-27b "$DFLASH2_27B" || need_27b_dflash2=0
fi
if ((need_35b_mtp)); then
    ensure_optional_model 35b_mtp --mtp-35b "$MTP_35B_REPO" "$MTP_35B" || need_35b_mtp=0
fi
if ((need_35b_dflash2)); then
    ensure_optional_local_model 35b_dflash2 --dflash2-35b "$DFLASH2_35B" || need_35b_dflash2=0
fi

runnable_cases=()
for case_name in "${selected_cases[@]}"; do
    case "$case_name" in
    27b_mtp1 | 27b_mtp2 | 27b_mtp3 | 27b_mtp4) ((need_27b_mtp)) || continue ;;
    27b_dflash2 | 27b_dflash2_k*) ((need_27b_dflash2)) || continue ;;
    35b_mtp1 | 35b_mtp2 | 35b_mtp3 | 35b_mtp4) ((need_35b_mtp)) || continue ;;
    35b_dflash2 | 35b_dflash2_k*) ((need_35b_dflash2)) || continue ;;
    esac
    runnable_cases+=("$case_name")
done
if ((${#runnable_cases[@]})); then
    selected_cases=("${runnable_cases[@]}")
else
    selected_cases=()
fi
if ((${#selected_cases[@]} == 0)); then
    echo "WARNING: no runnable cases remain after checkpoint discovery; exiting." >&2
    exit 0
fi
CASES="$(
    IFS=,
    printf '%s' "${selected_cases[*]}"
)"

need_27b=0
need_35b=0
for case_name in "${selected_cases[@]}"; do
    case "$case_name" in
    27b_*) need_27b=1 ;;
    35b_*) need_35b=1 ;;
    esac
done

if ((need_27b)); then
    ensure_model "$MODEL_27B_REPO" "$MODEL_27B"
fi
if ((need_35b)); then
    ensure_model "$MODEL_35B_REPO" "$MODEL_35B"
fi

if [[ -n "$TOKENIZER" ]]; then
    require_dir "--tokenizer" "$TOKENIZER"
fi
if ((need_27b)); then
    require_dir "--model-27b" "$MODEL_27B"
fi
if ((need_27b_mtp)); then
    require_dir "--mtp-27b" "$MTP_27B"
fi
if ((need_27b_dflash2)); then
    require_dir "--dflash2-27b" "$DFLASH2_27B"
    validate_affine_dflash2 "--dflash2-27b" "$DFLASH2_27B"
fi
if ((need_35b)); then
    require_dir "--model-35b" "$MODEL_35B"
fi
if ((need_35b_mtp)); then
    require_dir "--mtp-35b" "$MTP_35B"
fi
if ((need_35b_dflash2)); then
    require_dir "--dflash2-35b" "$DFLASH2_35B"
    validate_affine_dflash2 "--dflash2-35b" "$DFLASH2_35B"
fi

current_machine_id() {
    local display_info chipset_model gpu_cores normalized_chipset
    display_info="$(system_profiler SPDisplaysDataType 2>/dev/null || true)"
    chipset_model="$(printf '%s\n' "$display_info" | awk -F ': ' '/Chipset Model:/{print $2; exit}')"
    gpu_cores="$(printf '%s\n' "$display_info" | awk -F ': ' '/Total Number of Cores:/{print $2; exit}')"
    if [[ -z "$chipset_model" || -z "$gpu_cores" ]]; then
        echo "unknown"
        return
    fi
    normalized_chipset="$(printf '%s' "$chipset_model" | tr '[:upper:] ' '[:lower:]_' | tr -cd '[:alnum:]_')"
    echo "${normalized_chipset}_${gpu_cores}_gpu_cores"
}

prompt_set_sha256() {
    local prompt_index
    for prompt_index in "${!PROMPTS[@]}"; do
        printf '%s\034%s\035' "${PROMPT_IDS[$prompt_index]}" "${PROMPTS[$prompt_index]}"
    done | shasum -a 256 | awk '{print $1}'
}

prompt_ids_csv() {
    local IFS=,
    printf '%s' "${PROMPT_IDS[*]}"
}

reference_config_mismatches() {
    local current_machine="$1"
    local current_os="$2"
    local current_arch="$3"
    local current_prompt_set_sha256="$4"
    local mismatches=""

    [[ "$current_machine" == "$REFERENCE_MACHINE" ]] || mismatches="machine"
    [[ "$current_os" == "$REFERENCE_OS_VERSION" ]] || mismatches="${mismatches:+$mismatches,}os"
    [[ "$current_arch" == "$REFERENCE_ARCH" ]] || mismatches="${mismatches:+$mismatches,}arch"
    [[ "$NUM_CACHE_PAGES" == "$REFERENCE_NUM_CACHE_PAGES" ]] ||
        mismatches="${mismatches:+$mismatches,}num_cache_pages"
    [[ "$CACHE_BLOCK_TOKENS" == "$REFERENCE_CACHE_BLOCK_TOKENS" ]] ||
        mismatches="${mismatches:+$mismatches,}cache_block_tokens"
    [[ "$MAX_REQUESTS" == "$REFERENCE_MAX_REQUESTS" ]] || mismatches="${mismatches:+$mismatches,}max_requests"
    [[ "$MAX_TOKENS" == "$REFERENCE_MAX_TOKENS" ]] || mismatches="${mismatches:+$mismatches,}max_tokens"
    [[ "$MAX_TOKENS_PER_REQUEST" == "$REFERENCE_MAX_TOKENS_PER_REQUEST" ]] ||
        mismatches="${mismatches:+$mismatches,}max_tokens_per_request"
    [[ "$CASE_COOLDOWN_SECS" == "$REFERENCE_CASE_COOLDOWN_SECS" ]] ||
        mismatches="${mismatches:+$mismatches,}case_cooldown_secs"
    [[ "$LOGGING" == "$REFERENCE_LOGGING" ]] || mismatches="${mismatches:+$mismatches,}logging"
    [[ "$SEED" == "$REFERENCE_SEED" ]] || mismatches="${mismatches:+$mismatches,}seed"
    [[ "$TEMPERATURE" == "$REFERENCE_TEMPERATURE" ]] || mismatches="${mismatches:+$mismatches,}temperature"
    [[ "$TOP_K" == "$REFERENCE_TOP_K" ]] || mismatches="${mismatches:+$mismatches,}top_k"
    [[ "$TOP_P" == "$REFERENCE_TOP_P" ]] || mismatches="${mismatches:+$mismatches,}top_p"
    [[ "$PROMPT_SET" == "$REFERENCE_PROMPT_SET" ]] || mismatches="${mismatches:+$mismatches,}prompt_set"
    [[ "$current_prompt_set_sha256" == "$REFERENCE_PROMPT_SET_SHA256" ]] ||
        mismatches="${mismatches:+$mismatches,}prompt_set_sha256"
    [[ -z "$TOKENIZER" ]] || mismatches="${mismatches:+$mismatches,}tokenizer"
    if ((need_27b)); then
        [[ "$(basename "$MODEL_27B")" == "$REFERENCE_MODEL_27B_DIR_NAME" ]] ||
            mismatches="${mismatches:+$mismatches,}model_27b"
    fi
    if ((need_27b_mtp)); then
        [[ "$(basename "$MTP_27B")" == "$REFERENCE_MTP_27B_DIR_NAME" ]] ||
            mismatches="${mismatches:+$mismatches,}mtp_27b"
    fi
    if ((need_27b_dflash2)); then
        [[ "$(basename "$DFLASH2_27B")" == "$REFERENCE_DFLASH2_27B_DIR_NAME" ]] ||
            mismatches="${mismatches:+$mismatches,}dflash2_27b"
    fi
    if ((need_35b)); then
        [[ "$(basename "$MODEL_35B")" == "$REFERENCE_MODEL_35B_DIR_NAME" ]] ||
            mismatches="${mismatches:+$mismatches,}model_35b"
    fi
    if ((need_35b_mtp)); then
        [[ "$(basename "$MTP_35B")" == "$REFERENCE_MTP_35B_DIR_NAME" ]] ||
            mismatches="${mismatches:+$mismatches,}mtp_35b"
    fi
    echo "$mismatches"
}

reference_row() {
    # Reported fields only: decode tokens/s, tokens/chunk, output, proposed, verified.
    # K7 labels preserve the checkpoint-default geometry of the measured runs.
    case "$1:$2:$3:$4" in
    apple_m3_max_40_gpu_cores:27b_off:256:gsm8k_typing_average) echo "22.770|1.000|256|0|0" ;;
    apple_m3_max_40_gpu_cores:27b_off:1024:gsm8k_typing_average) echo "22.761|1.000|260|0|0" ;;
    apple_m3_max_40_gpu_cores:27b_off:256:chat) echo "22.832|1.000|256|0|0" ;;
    apple_m3_max_40_gpu_cores:27b_off:1024:chat) echo "22.554|1.000|1024|0|0" ;;
    apple_m3_max_40_gpu_cores:27b_mtp1:256:gsm8k_typing_average) echo "39.398|1.869|256|136|119" ;;
    apple_m3_max_40_gpu_cores:27b_mtp1:1024:gsm8k_typing_average) echo "39.110|1.881|299|157|140" ;;
    apple_m3_max_40_gpu_cores:27b_mtp1:256:chat) echo "35.024|1.673|256|151|103" ;;
    apple_m3_max_40_gpu_cores:27b_mtp1:1024:chat) echo "33.988|1.665|1024|614|409" ;;
    apple_m3_max_40_gpu_cores:27b_mtp2:256:gsm8k_typing_average) echo "41.780|2.747|239|172|152" ;;
    apple_m3_max_40_gpu_cores:27b_mtp2:1024:gsm8k_typing_average) echo "39.618|2.747|239|172|152" ;;
    apple_m3_max_40_gpu_cores:27b_mtp2:256:chat) echo "29.436|1.984|256|254|127" ;;
    apple_m3_max_40_gpu_cores:27b_mtp2:1024:chat) echo "30.503|2.147|1024|950|547" ;;
    apple_m3_max_40_gpu_cores:27b_mtp3:256:gsm8k_typing_average) echo "38.978|3.241|256|229|177" ;;
    apple_m3_max_40_gpu_cores:27b_mtp3:1024:gsm8k_typing_average) echo "38.399|3.284|266|239|185" ;;
    apple_m3_max_40_gpu_cores:27b_mtp3:256:chat) echo "24.053|2.032|256|375|130" ;;
    apple_m3_max_40_gpu_cores:27b_mtp3:1024:chat) echo "22.353|2.111|1024|1447|539" ;;
    apple_m3_max_40_gpu_cores:27b_mtp4:256:gsm8k_typing_average) echo "36.724|3.821|256|263|189" ;;
    apple_m3_max_40_gpu_cores:27b_mtp4:1024:gsm8k_typing_average) echo "36.361|3.843|269|273|199" ;;
    apple_m3_max_40_gpu_cores:27b_mtp4:256:chat) echo "19.214|2.000|256|503|128" ;;
    apple_m3_max_40_gpu_cores:27b_mtp4:1024:chat) echo "18.693|2.197|1024|1853|558" ;;
    apple_m3_max_40_gpu_cores:27b_dflash2_k7:256:gsm8k_typing_average) echo "44.196|4.571|256|385|200" ;;
    apple_m3_max_40_gpu_cores:27b_dflash2_k7:1024:gsm8k_typing_average) echo "47.769|4.963|397|548|317" ;;
    apple_m3_max_40_gpu_cores:27b_dflash2_k7:256:chat) echo "18.280|1.896|256|919|121" ;;
    apple_m3_max_40_gpu_cores:27b_dflash2_k7:1024:chat) echo "21.303|2.246|1024|3178|568" ;;
    apple_m3_max_40_gpu_cores:35b_off:256:gsm8k_typing_average) echo "94.962|1.000|256|0|0" ;;
    apple_m3_max_40_gpu_cores:35b_off:1024:gsm8k_typing_average) echo "93.701|1.000|1024|0|0" ;;
    apple_m3_max_40_gpu_cores:35b_off:256:chat) echo "96.103|1.000|256|0|0" ;;
    apple_m3_max_40_gpu_cores:35b_off:1024:chat) echo "93.952|1.000|1024|0|0" ;;
    apple_m3_max_40_gpu_cores:35b_mtp1:256:gsm8k_typing_average) echo "155.132|1.925|256|131|123" ;;
    apple_m3_max_40_gpu_cores:35b_mtp1:1024:gsm8k_typing_average) echo "148.439|1.956|1021|521|499" ;;
    apple_m3_max_40_gpu_cores:35b_mtp1:256:chat) echo "153.694|1.896|256|134|121" ;;
    apple_m3_max_40_gpu_cores:35b_mtp1:1024:chat) echo "135.227|1.756|1024|581|441" ;;
    apple_m3_max_40_gpu_cores:35b_mtp2:256:gsm8k_typing_average) echo "168.074|2.753|256|183|163" ;;
    apple_m3_max_40_gpu_cores:35b_mtp2:1024:gsm8k_typing_average) echo "154.959|2.688|1024|758|643" ;;
    apple_m3_max_40_gpu_cores:35b_mtp2:256:chat) echo "156.309|2.535|256|199|155" ;;
    apple_m3_max_40_gpu_cores:35b_mtp2:1024:chat) echo "128.282|2.193|1024|932|557" ;;
    apple_m3_max_40_gpu_cores:35b_mtp3:256:gsm8k_typing_average) echo "175.555|3.507|256|213|183" ;;
    apple_m3_max_40_gpu_cores:35b_mtp3:1024:gsm8k_typing_average) echo "163.067|3.402|1024|898|723" ;;
    apple_m3_max_40_gpu_cores:35b_mtp3:256:chat) echo "151.519|3.012|256|251|171" ;;
    apple_m3_max_40_gpu_cores:35b_mtp3:1024:chat) echo "117.661|2.421|1024|1266|601" ;;
    apple_m3_max_40_gpu_cores:35b_mtp4:256:gsm8k_typing_average) echo "153.677|3.710|256|269|187" ;;
    apple_m3_max_40_gpu_cores:35b_mtp4:1024:gsm8k_typing_average) echo "148.795|3.724|1024|1094|749" ;;
    apple_m3_max_40_gpu_cores:35b_mtp4:256:chat) echo "139.548|3.325|256|300|179" ;;
    apple_m3_max_40_gpu_cores:35b_mtp4:1024:chat) echo "105.720|2.592|1024|1573|629" ;;
    *) return 1 ;;
    esac
}

print_config_table() {
    CURRENT_COMMIT="$GIT_COMMIT" \
        CURRENT_DIRTY="$GIT_DIRTY" \
        CURRENT_MACHINE="$MACHINE" \
        CURRENT_OS="$OS_VERSION" \
        CURRENT_ARCH="$ARCH" \
        CURRENT_RUNS="$RUNS" \
        CURRENT_CASES="$CASES" \
        CURRENT_BLOCK_SPEC_TOKENS="${BLOCK_SPEC_TOKENS:-checkpoint-or-case}" \
        CURRENT_COOLDOWN="$CASE_COOLDOWN_SECS" \
        CURRENT_CAPACITY="$NUM_CACHE_PAGES pages; $CACHE_BLOCK_TOKENS tokens/block; $MAX_REQUESTS requests; $MAX_TOKENS tokens; $MAX_TOKENS_PER_REQUEST tokens/request" \
        CURRENT_SAMPLING="seed=$SEED; temperature=$TEMPERATURE; top_k=$TOP_K; top_p=$TOP_P; thinking=on" \
        CURRENT_PROMPTS="$PROMPT_SET: $PROMPT_IDS_CSV" \
        CURRENT_PROMPT_SHA="$PROMPT_SET_SHA256" \
        CURRENT_MODEL_27B="$MODEL_27B" \
        CURRENT_MTP_27B="$MTP_27B" \
        CURRENT_DFLASH2_27B="$DFLASH2_27B" \
        CURRENT_MODEL_35B="$MODEL_35B" \
        CURRENT_MTP_35B="$MTP_35B" \
        CURRENT_DFLASH2_35B="$DFLASH2_35B" \
        CURRENT_REFERENCE="enabled=$REFERENCE; $REFERENCE_DATE; $REFERENCE_COMMIT; $REFERENCE_MACHINE; runs=$REFERENCE_RUNS; scope=user-table; config=script-defaults" \
        CURRENT_REFERENCE_MISMATCHES="${REFERENCE_CONFIG_MISMATCHES:-none}" \
        python3 - <<'PY'
import os
import textwrap

rows = [
    ("Source", f"{os.environ['CURRENT_COMMIT']} dirty={os.environ['CURRENT_DIRTY']}"),
    ("Host", f"{os.environ['CURRENT_MACHINE']} os={os.environ['CURRENT_OS']} arch={os.environ['CURRENT_ARCH']}"),
    ("Run", f"runs={os.environ['CURRENT_RUNS']} cooldown={os.environ['CURRENT_COOLDOWN']}s"),
    ("Cases", os.environ["CURRENT_CASES"]),
    ("Block spec K", os.environ["CURRENT_BLOCK_SPEC_TOKENS"]),
    ("Capacity", os.environ["CURRENT_CAPACITY"]),
    ("Sampling", os.environ["CURRENT_SAMPLING"]),
    ("Prompts", os.environ["CURRENT_PROMPTS"]),
    ("Prompt SHA-256", os.environ["CURRENT_PROMPT_SHA"]),
    ("27B Main", os.environ["CURRENT_MODEL_27B"]),
    ("27B MTP", os.environ["CURRENT_MTP_27B"]),
    ("27B DFlash2", os.environ["CURRENT_DFLASH2_27B"]),
    ("35B Main", os.environ["CURRENT_MODEL_35B"]),
    ("35B MTP", os.environ["CURRENT_MTP_35B"]),
    ("35B DFlash2", os.environ["CURRENT_DFLASH2_35B"]),
    ("Reference", os.environ["CURRENT_REFERENCE"]),
    ("Ref mismatch", os.environ["CURRENT_REFERENCE_MISMATCHES"]),
]
key_width = max(len(key) for key, _ in rows)
value_width = 96
separator = "+-" + "-" * key_width + "-+-" + "-" * value_width + "-+"

print(separator)
print(f"| {'Configuration'.ljust(key_width)} | {'Value'.ljust(value_width)} |")
print(separator)
for key, value in rows:
    lines = textwrap.wrap(value, value_width, break_long_words=False, break_on_hyphens=False) or [""]
    for index, line in enumerate(lines):
        label = key if index == 0 else ""
        print(f"| {label.ljust(key_width)} | {line.ljust(value_width)} |")
print(separator)
PY
}

print_summary_table() {
    REPORT_FILE="$REPORT_FILE" python3 - <<'PY'
import os


def stable_value(encoded):
    values = encoded.split(",")
    return values[0] if len(set(values)) == 1 else "mixed"


rows = []
reference_status_counts = {}
with open(os.environ["REPORT_FILE"], encoding="utf-8") as report:
    for line in report:
        fields = dict(part.split("=", 1) for part in line.split()[1:])
        sampled = stable_value(fields["samples"])
        proposed = stable_value(fields["proposed_spec"])
        verified = stable_value(fields["verified_spec"])
        acceptance = "-" if proposed == "0" else f"{verified}/{proposed}"
        prompt = {
            "gsm8k_typing_average": "gsm8k",
        }.get(fields["prompt"], fields["prompt"])
        rows.append(
            [
                fields["label"],
                prompt,
                fields["max_new"],
                sampled,
                fields["median_decode_tok_s"],
                fields["median_tokens_per_chunk"],
                acceptance,
            ]
        )
        status = fields["reference_status"]
        reference_status_counts[status] = reference_status_counts.get(status, 0) + 1

headers = ["Case", "Prompt", "Max", "Out", "Tok/s", "Tok/ch", "Accept"]
numeric_columns = {2, 3, 4, 5, 6}
widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"


def format_row(values):
    cells = []
    for index, value in enumerate(values):
        cells.append(value.rjust(widths[index]) if index in numeric_columns else value.ljust(widths[index]))
    return "| " + " | ".join(cells) + " |"


print()
print(separator)
print(format_row(headers))
print(separator)
previous_case = None
for row in rows:
    if previous_case is not None and row[0] != previous_case:
        print(separator)
    print(format_row(row))
    previous_case = row[0]
print(separator)
print("Tok/s is median decode tokens/s. Accept is verified/proposed speculative tokens.")
statuses = ", ".join(
    f"{status}={count}" for status, count in sorted(reference_status_counts.items())
)
print(f"Reference: {statuses}. Use --show-runs for per-row deltas and details.")
PY
}

if [[ "$BUILD" -eq 1 ]]; then
    cargo build --release --bin qwen3_5_dense --bin qwen3_5_sparse --bin decode
fi

CONFLICT_PATTERN='target/release/qwen3|target/release/decode|inference-runtime-service|cargo bench|cargo run'
if pgrep -fl "$CONFLICT_PATTERN" >/dev/null 2>&1; then
    echo "refusing to run while another qwen/decode/cargo perf process is active:" >&2
    pgrep -fl "$CONFLICT_PATTERN" >&2 || true
    exit 1
fi

wait_for_port() {
    for _ in $(seq 1 240); do
        if nc -z 127.0.0.1 "$PORT" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

run_decode() {
    local label="$1"
    local tokens="$2"
    local prompt_id="$3"
    local prompt="$4"
    local run="$5"
    local server_log="$6"
    local tokenizer="$7"
    local out="/tmp/psi_dec_${label}_${tokens}_${prompt_id}_${run}.out"
    local server_log_offset
    server_log_offset="$(wc -c <"$server_log")"
    if ! target/release/decode \
        --server-url "http://127.0.0.1:${PORT}" \
        --max-sampled-tokens "$tokens" \
        --seed "$SEED" \
        --temperature "$TEMPERATURE" \
        --top-k "$TOP_K" \
        --top-p "$TOP_P" \
        --enable-thinking true \
        --chat-template auto \
        --show-stats \
        --hf-model-dir "$tokenizer" \
        --prompt-str "$prompt" >"$out" 2>&1; then
        echo "DECODE_FAILED label=$label max_new=$tokens prompt=$prompt_id run=$run client_output=$out server_log=$server_log" >&2
        tail -n 80 "$out" >&2 || true
        tail -n 120 "$server_log" >&2 || true
        return 1
    fi
    local json
    json=$(grep "^{" "$out" | tail -n 1 || true)
    if [[ -z "$json" ]]; then
        echo "DECODE_STATS_MISSING label=$label max_new=$tokens prompt=$prompt_id run=$run client_output=$out server_log=$server_log" >&2
        tail -n 80 "$out" >&2 || true
        tail -n 120 "$server_log" >&2 || true
        return 1
    fi
    if ! CLIENT_OUTPUT="$out" JSON_LINE="$json" SERVER_LOG="$server_log" SERVER_LOG_OFFSET="$server_log_offset" \
        python3 - <<'PY'
import hashlib
import json
import os
import re

j = json.loads(os.environ["JSON_LINE"])
with open(os.environ["CLIENT_OUTPUT"], "rb") as f:
    client_output = f.read()
output_separator = b"\n==========\n"
output_end = client_output.rfind(output_separator)
if output_end < 0:
    raise ValueError("decode output separator is missing")
output_sha256 = hashlib.sha256(client_output[:output_end]).hexdigest()
with open(os.environ["SERVER_LOG"], "rb") as f:
    f.seek(int(os.environ["SERVER_LOG_OFFSET"]))
    server_log = f.read().decode("utf-8", errors="replace")
server_log = re.sub(r"\x1b\[[0-9;]*m", "", server_log)
proposed = 0
verified = 0
spec_by_index = []
verified_by_index = []

def add_counts(total, values):
    total.extend([0] * (len(values) - len(total)))
    for index, value in enumerate(values):
        total[index] += value

def index_counts(line, field):
    match = re.search(rf"{field}=\[([^]]*)\]", line)
    if not match or not match.group(1).strip():
        return []
    return [int(value.strip()) for value in match.group(1).split(",")]

for line in server_log.splitlines():
    if 'phase="executor.batch.perf"' not in line:
        continue
    proposed_match = re.search(r"num_spec_tokens=(\d+)", line)
    verified_match = re.search(r"num_verified_tokens=(\d+)", line)
    if proposed_match and verified_match:
        proposed += int(proposed_match.group(1))
        verified += int(verified_match.group(1))
        add_counts(spec_by_index, index_counts(line, "num_spec_token_by_index"))
        add_counts(verified_by_index, index_counts(line, "num_verified_token_by_index"))
acceptance_rate = verified / proposed if proposed else 0.0
acceptance_rate_by_index = [
    verified_count / spec_count if spec_count else 0.0
    for spec_count, verified_count in zip(spec_by_index, verified_by_index)
]
tokens_per_chunk = j["sampled_tokens"] / j["chunk_count"]
encode_counts = lambda values: ":".join(str(value) for value in values) or "-"
encode_rates = lambda values: ":".join(f"{value:.6f}" for value in values) or "-"
print("{:.6f},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{:.6f},{:.6f},{},{},{},{}".format(
    j["decode_tokens_per_s"],
    j["chunk_count"],
    j["sampled_tokens"],
    j["input_tokens"],
    j["ttft_ms"],
    j["prompt_tokens_per_s"],
    j["inter_chunk_p50_ms"],
    j["inter_chunk_p95_ms"],
    proposed,
    verified,
    acceptance_rate,
    tokens_per_chunk,
    encode_counts(spec_by_index),
    encode_counts(verified_by_index),
    encode_rates(acceptance_rate_by_index),
    output_sha256,
))
PY
    then
        echo "DECODE_STATS_INVALID label=$label max_new=$tokens prompt=$prompt_id run=$run client_output=$out server_log=$server_log" >&2
        tail -n 80 "$out" >&2 || true
        tail -n 120 "$server_log" >&2 || true
        return 1
    fi
}

run_server_case() {
    local label="$1"
    local token_list="$2"
    local tokenizer="$3"
    local prompt_index
    shift 3
    local log="/tmp/psi_dec_${label}.log"

    local server_command=("$@")
    local server_rust_log="${RUST_LOG:-info}"
    server_rust_log+=",inference-runtime-service::perf=debug"
    server_command+=(--logging "$LOGGING")
    RUST_LOG="$server_rust_log" "${server_command[@]}" >"$log" 2>&1 &
    ACTIVE_SERVER_PID=$!

    if ! wait_for_port; then
        echo "SERVER_START_FAILED $label" >&2
        tail -n 80 "$log" >&2 || true
        cleanup_active_server
        exit 1
    fi

    for prompt_index in "${!PROMPTS[@]}"; do
        local prompt_id="${PROMPT_IDS[$prompt_index]}"
        local prompt="${PROMPTS[$prompt_index]}"
        for tokens in $token_list; do
            echo "MEASURE case=$label prompt=$prompt_id max_new=$tokens runs=$RUNS"
            local vals=""
            local inputs=""
            local chunks=""
            local samples=""
            local ttfts=""
            local prompt_rates=""
            local inter_chunk_p50s=""
            local inter_chunk_p95s=""
            local proposed_specs=""
            local verified_specs=""
            local acceptance_rates=""
            local tokens_per_chunks=""
            local spec_by_index=""
            local verified_by_index=""
            local expected_output_sha256=""
            local expected_output_file=""
            for run in $(seq 1 "$RUNS"); do
                local parsed tokps chunk sampled input_tokens ttft prompt_rate
                local inter_chunk_p50 inter_chunk_p95 proposed_spec verified_spec
                local acceptance_rate tokens_per_chunk
                local run_spec_by_index run_verified_by_index acceptance_rate_by_index output_sha256
                parsed=$(run_decode "$label" "$tokens" "$prompt_id" "$prompt" "$run" "$log" "$tokenizer")
                IFS=, read -r \
                    tokps chunk sampled input_tokens ttft prompt_rate \
                    inter_chunk_p50 inter_chunk_p95 proposed_spec verified_spec \
                    acceptance_rate tokens_per_chunk run_spec_by_index \
                    run_verified_by_index acceptance_rate_by_index output_sha256 <<<"$parsed"
                local output_file="/tmp/psi_dec_${label}_${tokens}_${prompt_id}_${run}.out"
                if [[ -z "$expected_output_sha256" ]]; then
                    expected_output_sha256="$output_sha256"
                    expected_output_file="$output_file"
                elif [[ "$output_sha256" != "$expected_output_sha256" ]]; then
                    echo "DETERMINISM_FAILED label=$label max_new=$tokens prompt=$prompt_id seed=$SEED" \
                        "expected_sha256=$expected_output_sha256 actual_sha256=$output_sha256" \
                        "expected_output=$expected_output_file actual_output=$output_file" >&2
                    return 1
                fi
                vals="$vals $tokps"
                inputs="$inputs $input_tokens"
                chunks="$chunks $chunk"
                samples="$samples $sampled"
                ttfts="$ttfts $ttft"
                prompt_rates="$prompt_rates $prompt_rate"
                inter_chunk_p50s="$inter_chunk_p50s $inter_chunk_p50"
                inter_chunk_p95s="$inter_chunk_p95s $inter_chunk_p95"
                proposed_specs="$proposed_specs $proposed_spec"
                verified_specs="$verified_specs $verified_spec"
                acceptance_rates="$acceptance_rates $acceptance_rate"
                tokens_per_chunks="$tokens_per_chunks $tokens_per_chunk"
                spec_by_index="$spec_by_index $run_spec_by_index"
                verified_by_index="$verified_by_index $run_verified_by_index"
                if ((SHOW_RUNS)); then
                    echo "RUN label=$label max_new=$tokens prompt=$prompt_id run=$run" \
                        "input_tokens=$input_tokens sampled=$sampled chunks=$chunk" \
                        "proposed_spec=$proposed_spec verified_spec=$verified_spec" \
                        "acceptance_rate=$acceptance_rate acceptance_rate_by_index=$acceptance_rate_by_index" \
                        "output_sha256=$output_sha256" \
                        "tokens_per_chunk=$tokens_per_chunk" \
                        "decode_tok_s=$tokps ttft_ms=$ttft prompt_tok_s=$prompt_rate" \
                        "inter_chunk_p50_ms=$inter_chunk_p50 inter_chunk_p95_ms=$inter_chunk_p95"
                fi
            done

            local reference_record=""
            local reference_decode=""
            local reference_tokens_per_chunk=""
            local reference_sampled=""
            local reference_proposed_spec=""
            local reference_verified_spec=""
            local reference_status="disabled"
            local reference_mismatch=""
            if ((REFERENCE)); then
                reference_record="$(reference_row "$MACHINE" "$label" "$tokens" "$prompt_id" || true)"
                if [[ -z "$reference_record" ]]; then
                    reference_status="no-reference-row"
                    reference_mismatch="row"
                else
                    IFS='|' read -r reference_decode reference_tokens_per_chunk \
                        reference_sampled reference_proposed_spec reference_verified_spec <<<"$reference_record"
                    if [[ -n "$REFERENCE_CONFIG_MISMATCHES" ]]; then
                        reference_status="config-mismatch"
                        reference_mismatch="$REFERENCE_CONFIG_MISMATCHES"
                    else
                        reference_status="summary-only"
                    fi
                fi
            fi

            local summary
            summary="$(
                VALS="$vals" \
                    INPUTS="$inputs" \
                    CHUNKS="$chunks" \
                    SAMPLES="$samples" \
                    TTFTS="$ttfts" \
                    PROMPT_RATES="$prompt_rates" \
                    INTER_CHUNK_P50S="$inter_chunk_p50s" \
                    INTER_CHUNK_P95S="$inter_chunk_p95s" \
                    PROPOSED_SPECS="$proposed_specs" \
                    VERIFIED_SPECS="$verified_specs" \
                    ACCEPTANCE_RATES="$acceptance_rates" \
                    TOKENS_PER_CHUNKS="$tokens_per_chunks" \
                    SPEC_BY_INDEX="$spec_by_index" \
                    VERIFIED_BY_INDEX="$verified_by_index" \
                    LABEL="$label" \
                    TOKENS="$tokens" \
                    PROMPT_ID="$prompt_id" \
                    OUTPUT_SHA256="$expected_output_sha256" \
                    REFERENCE_DECODE="$reference_decode" \
                    REFERENCE_TOKENS_PER_CHUNK="$reference_tokens_per_chunk" \
                    REFERENCE_SAMPLED="$reference_sampled" \
                    REFERENCE_PROPOSED_SPEC="$reference_proposed_spec" \
                    REFERENCE_VERIFIED_SPEC="$reference_verified_spec" \
                    REFERENCE_STATUS="$reference_status" \
                    REFERENCE_MISMATCH="$reference_mismatch" \
                    python3 - <<'PY'
import os
import statistics

vals = [float(x) for x in os.environ["VALS"].split()]
inputs = os.environ["INPUTS"].split()
chunks = os.environ["CHUNKS"].split()
samples = os.environ["SAMPLES"].split()
ttfts = [float(x) for x in os.environ["TTFTS"].split()]
prompt_rates = [float(x) for x in os.environ["PROMPT_RATES"].split()]
inter_chunk_p50s = [float(x) for x in os.environ["INTER_CHUNK_P50S"].split()]
inter_chunk_p95s = [float(x) for x in os.environ["INTER_CHUNK_P95S"].split()]
proposed_specs = os.environ["PROPOSED_SPECS"].split()
verified_specs = os.environ["VERIFIED_SPECS"].split()
acceptance_rates = [float(x) for x in os.environ["ACCEPTANCE_RATES"].split()]
tokens_per_chunks = [float(x) for x in os.environ["TOKENS_PER_CHUNKS"].split()]

def sum_index_counts(name):
    totals = []
    for encoded in os.environ[name].split():
        values = [] if encoded == "-" else [int(value) for value in encoded.split(":")]
        totals.extend([0] * (len(values) - len(totals)))
        for index, value in enumerate(values):
            totals[index] += value
    return totals

spec_by_index = sum_index_counts("SPEC_BY_INDEX")
verified_by_index = sum_index_counts("VERIFIED_BY_INDEX")
acceptance_rate_by_index = [
    verified / spec if spec else 0.0
    for spec, verified in zip(spec_by_index, verified_by_index)
]
acceptance_rate_by_index_text = "[{}]".format(
    ",".join(f"{value:.6f}" for value in acceptance_rate_by_index)
)
median_decode = statistics.median(vals)
median_ttft = statistics.median(ttfts)
median_prompt_rate = statistics.median(prompt_rates)
median_inter_chunk_p50 = statistics.median(inter_chunk_p50s)
median_inter_chunk_p95 = statistics.median(inter_chunk_p95s)
median_acceptance_rate = statistics.median(acceptance_rates)
median_tokens_per_chunk = statistics.median(tokens_per_chunks)
acceptance_rate_text = (
    "{:.6f}".format(median_acceptance_rate)
    if any(int(proposed) > 0 for proposed in proposed_specs)
    else "na"
)
prefix = (
    "SUMMARY label={} max_new={} prompt={} median_decode_tok_s={:.3f} median_ttft_ms={:.3f} "
    "median_prompt_tok_s={:.3f} median_inter_chunk_p50_ms={:.3f} "
    "median_inter_chunk_p95_ms={:.3f} median_tokens_per_chunk={:.3f} "
    "median_acceptance_rate={} acceptance_rate_by_index={} output_sha256={}"
).format(
    os.environ["LABEL"],
    os.environ["TOKENS"],
    os.environ["PROMPT_ID"],
    median_decode,
    median_ttft,
    median_prompt_rate,
    median_inter_chunk_p50,
    median_inter_chunk_p95,
    median_tokens_per_chunk,
    acceptance_rate_text,
    acceptance_rate_by_index_text,
    os.environ["OUTPUT_SHA256"],
)
reference_decode = os.environ.get("REFERENCE_DECODE", "")
reference_status = os.environ["REFERENCE_STATUS"]
reference_mismatch = os.environ.get("REFERENCE_MISMATCH", "")
if reference_status == "summary-only" and (
    any(value != os.environ["REFERENCE_SAMPLED"] for value in samples)
    or any(value != os.environ["REFERENCE_PROPOSED_SPEC"] for value in proposed_specs)
    or any(value != os.environ["REFERENCE_VERIFIED_SPEC"] for value in verified_specs)
    or f"{median_tokens_per_chunk:.3f}" != os.environ["REFERENCE_TOKENS_PER_CHUNK"]
):
    reference_status = "work-count-mismatch"
    reference_mismatch = "work-counts"
if reference_decode:
    reference_decode_value = float(reference_decode)
    prefix += (
        f" reference_decode_tok_s={reference_decode_value:.3f}"
        f" reference_tokens_per_chunk={os.environ['REFERENCE_TOKENS_PER_CHUNK']}"
        f" reference_sampled={os.environ['REFERENCE_SAMPLED']}"
        f" reference_proposed_spec={os.environ['REFERENCE_PROPOSED_SPEC']}"
        f" reference_verified_spec={os.environ['REFERENCE_VERIFIED_SPEC']}"
    )
    if reference_status == "summary-only":
        # This is a descriptive ratio, not a controlled kernel-performance verdict.
        prefix += f" observed_decode_delta_pct={100.0 * (median_decode / reference_decode_value - 1.0):+.2f}"
prefix += f" reference_status={reference_status}"
if reference_mismatch:
    prefix += f" reference_mismatch={reference_mismatch}"
print(
    "{} min_decode_tok_s={:.3f} max_decode_tok_s={:.3f} runs={} input_tokens={} samples={} chunks={} proposed_spec={} verified_spec={}".format(
        prefix,
        min(vals),
        max(vals),
        ",".join("{:.3f}".format(v) for v in vals),
        ",".join(inputs),
        ",".join(samples),
        ",".join(chunks),
        ",".join(proposed_specs),
        ",".join(verified_specs),
    )
)
PY
            )"
            printf '%s\n' "$summary" >>"$REPORT_FILE"
            if ((SHOW_RUNS)); then
                printf '%s\n' "$summary"
            fi
        done
    done

    cleanup_active_server
}

run_mtp_case() {
    local model_label="$1"
    local num_spec_tokens="$2"
    local token_list="$3"
    local server_binary="$4"
    local model_dir="$5"
    local mtp_model_dir="$6"
    local tokenizer="${TOKENIZER:-$model_dir}"
    local label="${model_label}_mtp${num_spec_tokens}"
    run_server_case "$label" "$token_list" "$tokenizer" "$server_binary" \
        --grpc-listen-addr "127.0.0.1:${PORT}" \
        --hf-model-dir "$model_dir" \
        --hf-spec-model-dir "$mtp_model_dir" \
        --spec-type mtp \
        --num-spec-tokens "$num_spec_tokens" \
        --num-cache-pages "$NUM_CACHE_PAGES" \
        --max-requests "$MAX_REQUESTS" \
        --max-tokens "$MAX_TOKENS" \
        --max-tokens-per-request "$MAX_TOKENS_PER_REQUEST"
}

run_block_spec_case() {
    local model_label="$1"
    local spec_mode="$2"
    local num_spec_tokens="$3"
    local token_list="$4"
    local server_binary="$5"
    local model_dir="$6"
    local spec_model_dir="$7"
    local tokenizer="${TOKENIZER:-$model_dir}"

    if [[ -z "$num_spec_tokens" ]]; then
        num_spec_tokens="$(
            python3 - "$spec_mode" "$spec_model_dir/config.json" <<'PY'
import json
import sys

with open(sys.argv[2], encoding="utf-8") as config_file:
    config = json.load(config_file)
print(config["dflash_config"]["block_size"] - 1)
PY
        )"
    fi
    local label="${model_label}_${spec_mode}_k${num_spec_tokens}"
    run_server_case "$label" "$token_list" "$tokenizer" "$server_binary" \
        --grpc-listen-addr "127.0.0.1:${PORT}" \
        --hf-model-dir "$model_dir" \
        --hf-spec-model-dir "$spec_model_dir" \
        --spec-type "$spec_mode" \
        --num-spec-tokens "$num_spec_tokens" \
        --num-cache-pages "$NUM_CACHE_PAGES" \
        --max-requests "$MAX_REQUESTS" \
        --max-tokens "$MAX_TOKENS" \
        --max-tokens-per-request "$MAX_TOKENS_PER_REQUEST"
}

run_named_case() {
    local num_spec_tokens=""
    case "$1" in
    *_dflash2_k*) num_spec_tokens="${1##*_k}" ;;
    esac
    case "$1" in
    27b_off)
        run_server_case 27b_off "256 1024" "${TOKENIZER:-$MODEL_27B}" target/release/qwen3_5_dense \
            --grpc-listen-addr "127.0.0.1:${PORT}" \
            --hf-model-dir "$MODEL_27B" \
            --num-cache-pages "$NUM_CACHE_PAGES" \
            --max-requests "$MAX_REQUESTS" \
            --max-tokens "$MAX_TOKENS" \
            --max-tokens-per-request "$MAX_TOKENS_PER_REQUEST"
        ;;
    27b_mtp1 | 27b_mtp2 | 27b_mtp3 | 27b_mtp4)
        run_mtp_case 27b "${1#27b_mtp}" "256 1024" target/release/qwen3_5_dense "$MODEL_27B" "$MTP_27B"
        ;;
    27b_dflash2 | 27b_dflash2_k*)
        run_block_spec_case 27b dflash2 "$num_spec_tokens" "256 1024" target/release/qwen3_5_dense "$MODEL_27B" "$DFLASH2_27B"
        ;;
    35b_off)
        run_server_case 35b_off "256 1024" "${TOKENIZER:-$MODEL_35B}" target/release/qwen3_5_sparse \
            --grpc-listen-addr "127.0.0.1:${PORT}" \
            --hf-model-dir "$MODEL_35B" \
            --num-cache-pages "$NUM_CACHE_PAGES" \
            --max-requests "$MAX_REQUESTS" \
            --max-tokens "$MAX_TOKENS" \
            --max-tokens-per-request "$MAX_TOKENS_PER_REQUEST"
        ;;
    35b_mtp1 | 35b_mtp2 | 35b_mtp3 | 35b_mtp4)
        run_mtp_case 35b "${1#35b_mtp}" "256 1024" target/release/qwen3_5_sparse "$MODEL_35B" "$MTP_35B"
        ;;
    35b_dflash2 | 35b_dflash2_k*)
        run_block_spec_case 35b dflash2 "$num_spec_tokens" "256 1024" target/release/qwen3_5_sparse "$MODEL_35B" "$DFLASH2_35B"
        ;;
    *)
        echo "unknown case: $1" >&2
        exit 2
        ;;
    esac
}

GIT_COMMIT="$(git rev-parse --verify HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY=unknown
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if [[ -n "$(git status --porcelain)" ]]; then
        GIT_DIRTY=1
    else
        GIT_DIRTY=0
    fi
fi
OS_VERSION="$(sw_vers -productVersion 2>/dev/null || uname -s)"
ARCH="$(uname -m)"
MACHINE="$(current_machine_id)"
PROMPT_SET_SHA256="$(prompt_set_sha256)"
PROMPT_IDS_CSV="$(prompt_ids_csv)"
REFERENCE_CONFIG_MISMATCHES="$(
    reference_config_mismatches "$MACHINE" "$OS_VERSION" "$ARCH" "$PROMPT_SET_SHA256"
)"
REPORT_FILE="$(mktemp "${TMPDIR:-/tmp}/psi_dec_qwen35_perf.XXXXXX")"
if ((SHOW_RUNS)); then
    echo "CONFIG commit=$GIT_COMMIT dirty=$GIT_DIRTY machine=$MACHINE os=$OS_VERSION arch=$ARCH runs=$RUNS build=$BUILD grpc_port=$PORT num_cache_pages=$NUM_CACHE_PAGES cache_block_tokens=$CACHE_BLOCK_TOKENS max_requests=$MAX_REQUESTS max_tokens=$MAX_TOKENS max_tokens_per_request=$MAX_TOKENS_PER_REQUEST mtp_num_spec_tokens=case-specific block_spec_tokens=${BLOCK_SPEC_TOKENS:-checkpoint-or-case} cases=$CASES case_cooldown_secs=$CASE_COOLDOWN_SECS logging=$LOGGING seed=$SEED temperature=$TEMPERATURE top_k=$TOP_K top_p=$TOP_P enable_thinking=1 prompt_set=$PROMPT_SET prompt_count=${#PROMPTS[@]} prompt_ids=$PROMPT_IDS_CSV prompt_set_sha256=$PROMPT_SET_SHA256 tokenizer=${TOKENIZER:-auto-per-model} model_27b=$MODEL_27B mtp_27b=$MTP_27B dflash2_27b=$DFLASH2_27B model_35b=$MODEL_35B mtp_35b=$MTP_35B dflash2_35b=$DFLASH2_35B reference_enabled=$REFERENCE reference_machine=$REFERENCE_MACHINE reference_date=$REFERENCE_DATE reference_commit=$REFERENCE_COMMIT reference_source_tree=$REFERENCE_SOURCE_TREE reference_dirty=$REFERENCE_DIRTY reference_os=$REFERENCE_OS_VERSION reference_arch=$REFERENCE_ARCH reference_runs=$REFERENCE_RUNS reference_scope=user-table reference_config_source=script-defaults reference_cases=$REFERENCE_CASES reference_prompt_set=$REFERENCE_PROMPT_SET reference_prompt_set_sha256=$REFERENCE_PROMPT_SET_SHA256 reference_config_mismatches=${REFERENCE_CONFIG_MISMATCHES:-none}"
fi
print_config_table
for case_index in "${!selected_cases[@]}"; do
    case_name="${selected_cases[$case_index]}"
    if [[ "$case_index" -gt 0 && "$CASE_COOLDOWN_SECS" -gt 0 ]]; then
        echo "COOLDOWN before=$case_name seconds=$CASE_COOLDOWN_SECS"
        sleep "$CASE_COOLDOWN_SECS"
    fi
    run_named_case "$case_name"
done
print_summary_table
