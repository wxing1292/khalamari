#!/usr/bin/env bash
set -euo pipefail

RUNS=7
CASES="14b_off"
TOKENS="256,1024"
GRPC_PORT=50061
HTTP_PORT=8000
BUILD=1
CASE_COOLDOWN_SECS=8
LOGGING=info

PROMPT="你好，北京有什么好玩的景点？香山如何？早上去晚上去？单纯爬山么？还有什么可以在香山玩的？"
SEED=42
TEMPERATURE=0.7
TOP_K=20
TOP_P=0.8

NUM_CACHE_PAGES=294912
MAX_REQUESTS=2
MAX_TOKENS=128
MAX_TOKENS_PER_REQUEST=64
CACHE_BLOCK_TOKENS=16

MODEL_ROOT="$HOME/Workspace/models"
MODEL_DIR=""
TOKENIZER_DIR=""
MODEL_REPO="mlx-community/Qwen3-14B-4bit"

ACTIVE_SERVER_PID=""

cleanup_server() {
    if [[ -n "$ACTIVE_SERVER_PID" ]]; then
        kill "$ACTIVE_SERVER_PID" >/dev/null 2>&1 || true
        wait "$ACTIVE_SERVER_PID" >/dev/null 2>&1 || true
        ACTIVE_SERVER_PID=""
    fi
}

trap cleanup_server EXIT
trap 'cleanup_server; exit 130' INT
trap 'cleanup_server; exit 143' TERM

usage() {
    cat <<'USAGE'
Usage: scripts/qwen3_e2e_decode_perf.sh [options]

Runs Qwen3-14B Main-only, one server at a time.

Cases:
  14b_off              Qwen3-14B Main only

Model options:
  --model-root DIR     Default: $HOME/Workspace/models
  --model DIR          Default: MODEL_ROOT/Qwen3-14B-4bit
  --tokenizer DIR      Default: Main model directory
  --model-repo REPO    Download source if Main is missing.
                       Default: mlx-community/Qwen3-14B-4bit

Benchmark options:
  --cases LIST         Default: 14b_off
  --runs N             Default: 7
  --tokens LIST        Default: 256,1024
  --grpc-port N        Default: 50061
  --http-port N        Default: 8000
  --prompt TEXT
  --seed N             Default: 42
  --temperature N      Default: 0.7
  --top-k N            Default: 20
  --top-p N            Default: 0.8
  --num-cache-pages N  Default: 294912
  --max-requests N     Default: 2
  --max-tokens N       Default: 128
  --max-tokens-per-request N
                       Default: 64
  --case-cooldown-secs N
                       Default: 8
  --logging LEVEL      info or debug. Default: info
                       The benchmark always enables the executor perf DEBUG target.
  --no-build           Skip cargo build --release
  -h, --help

Examples:
  scripts/qwen3_e2e_decode_perf.sh \
    --model-root "$HOME/Workspace/models" \
    --cases 14b_off \
    --runs 3

  scripts/qwen3_e2e_decode_perf.sh \
    --model "$HOME/Workspace/models/Qwen3-14B-4bit" \
    --runs 3
USAGE
}

need_value() {
    local option="$1"
    local count="$2"
    if ((count < 2)); then
        echo "$option requires a value" >&2
        exit 2
    fi
}

while (($# > 0)); do
    case "$1" in
    --runs)
        need_value "$1" "$#"
        RUNS="$2"
        shift 2
        ;;
    --cases)
        need_value "$1" "$#"
        CASES="$2"
        shift 2
        ;;
    --tokens)
        need_value "$1" "$#"
        TOKENS="$2"
        shift 2
        ;;
    --grpc-port)
        need_value "$1" "$#"
        GRPC_PORT="$2"
        shift 2
        ;;
    --http-port)
        need_value "$1" "$#"
        HTTP_PORT="$2"
        shift 2
        ;;
    --prompt)
        need_value "$1" "$#"
        PROMPT="$2"
        shift 2
        ;;
    --seed)
        need_value "$1" "$#"
        SEED="$2"
        shift 2
        ;;
    --temperature)
        need_value "$1" "$#"
        TEMPERATURE="$2"
        shift 2
        ;;
    --top-k)
        need_value "$1" "$#"
        TOP_K="$2"
        shift 2
        ;;
    --top-p)
        need_value "$1" "$#"
        TOP_P="$2"
        shift 2
        ;;
    --num-cache-pages)
        need_value "$1" "$#"
        NUM_CACHE_PAGES="$2"
        shift 2
        ;;
    --max-requests)
        need_value "$1" "$#"
        MAX_REQUESTS="$2"
        shift 2
        ;;
    --max-tokens)
        need_value "$1" "$#"
        MAX_TOKENS="$2"
        shift 2
        ;;
    --max-tokens-per-request)
        need_value "$1" "$#"
        MAX_TOKENS_PER_REQUEST="$2"
        shift 2
        ;;
    --model-root)
        need_value "$1" "$#"
        MODEL_ROOT="$2"
        shift 2
        ;;
    --model)
        need_value "$1" "$#"
        MODEL_DIR="$2"
        shift 2
        ;;
    --tokenizer)
        need_value "$1" "$#"
        TOKENIZER_DIR="$2"
        shift 2
        ;;
    --model-repo)
        need_value "$1" "$#"
        MODEL_REPO="$2"
        shift 2
        ;;
    --case-cooldown-secs)
        need_value "$1" "$#"
        CASE_COOLDOWN_SECS="$2"
        shift 2
        ;;
    --logging)
        need_value "$1" "$#"
        LOGGING="$2"
        shift 2
        ;;
    --no-build)
        BUILD=0
        shift
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

positive_integer() {
    local option="$1"
    local value="$2"
    case "$value" in
    '' | *[!0-9]* | 0)
        echo "$option expects a positive integer" >&2
        exit 2
        ;;
    esac
}

nonnegative_integer() {
    local option="$1"
    local value="$2"
    case "$value" in
    '' | *[!0-9]*)
        echo "$option expects a non-negative integer" >&2
        exit 2
        ;;
    esac
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "required command is unavailable: $1" >&2
        exit 2
    fi
}

positive_integer --runs "$RUNS"
positive_integer --grpc-port "$GRPC_PORT"
positive_integer --http-port "$HTTP_PORT"
positive_integer --top-k "$TOP_K"
positive_integer --num-cache-pages "$NUM_CACHE_PAGES"
positive_integer --max-requests "$MAX_REQUESTS"
positive_integer --max-tokens "$MAX_TOKENS"
positive_integer --max-tokens-per-request "$MAX_TOKENS_PER_REQUEST"
nonnegative_integer --seed "$SEED"
nonnegative_integer --case-cooldown-secs "$CASE_COOLDOWN_SECS"

if ((GRPC_PORT > 65535 || HTTP_PORT > 65535)); then
    echo "ports must be at most 65535" >&2
    exit 2
fi
if ((MAX_TOKENS_PER_REQUEST > MAX_TOKENS)); then
    echo "--max-tokens-per-request must not exceed --max-tokens" >&2
    exit 2
fi

case "$LOGGING" in
info | debug) ;;
*)
    echo "--logging must be info or debug" >&2
    exit 2
    ;;
esac

MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/Qwen3-14B-4bit}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$MODEL_DIR}"

IFS=, read -r -a SELECTED_CASES <<<"$CASES"
if ((${#SELECTED_CASES[@]} == 0)); then
    echo "--cases must include at least one case" >&2
    exit 2
fi

for case_name in "${SELECTED_CASES[@]}"; do
    case "$case_name" in
    14b_off) ;;
    *)
        echo "unknown case: $case_name" >&2
        exit 2
        ;;
    esac
done

IFS=, read -r -a TOKEN_COUNTS <<<"$TOKENS"
if ((${#TOKEN_COUNTS[@]} == 0)); then
    echo "--tokens must include at least one token count" >&2
    exit 2
fi
for token_count in "${TOKEN_COUNTS[@]}"; do
    positive_integer --tokens "$token_count"
done

require_command python3
require_command nc
require_command pgrep
if ((BUILD)); then
    require_command cargo
fi

if ! TEMPERATURE="$TEMPERATURE" TOP_P="$TOP_P" python3 - <<'PY'
import math
import os

for name in ("TEMPERATURE", "TOP_P"):
    try:
        value = float(os.environ[name])
    except ValueError as exc:
        raise SystemExit(f"--{name.lower().replace('_', '-')} must be a number") from exc
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

checkpoint_present() {
    local dir="$1"
    [[ -f "$dir/config.json" ]] || return 1
    find "$dir" -type f \( -name '*.safetensors' -o -name '*.npz' \) \
        -print -quit 2>/dev/null | grep -q .
}

download_checkpoint() {
    local repo="$1"
    local dir="$2"
    local -a downloader

    if command -v hf >/dev/null 2>&1; then
        downloader=(hf download)
    elif command -v huggingface-cli >/dev/null 2>&1; then
        downloader=(huggingface-cli download)
    else
        echo "checkpoint is missing: $dir" >&2
        echo 'Install Hugging Face CLI: python3 -m pip install -U "huggingface_hub[hf_xet]"' >&2
        exit 1
    fi

    echo "==> Downloading $repo -> $dir"
    mkdir -p "$dir"
    "${downloader[@]}" "$repo" --local-dir "$dir"
}

ensure_checkpoint() {
    local label="$1"
    local repo="$2"
    local dir="$3"

    if checkpoint_present "$dir"; then
        echo "==> Found $label checkpoint: $dir"
        return
    fi

    if [[ -z "$repo" ]]; then
        echo "$label checkpoint is missing: $dir" >&2
        exit 1
    fi

    download_checkpoint "$repo" "$dir"
    if ! checkpoint_present "$dir"; then
        echo "$label checkpoint is incomplete after download: $dir" >&2
        exit 1
    fi
}



ensure_checkpoint "Main" "$MODEL_REPO" "$MODEL_DIR"

if [[ ! -d "$TOKENIZER_DIR" ]]; then
    echo "tokenizer directory does not exist: $TOKENIZER_DIR" >&2
    exit 1
fi

CONFLICT_PATTERN='target/release/qwen3|target/release/decode|inference-runtime-service|cargo bench|cargo run'
if pgrep -fl "$CONFLICT_PATTERN" >/dev/null 2>&1; then
    echo "refusing to run while another Qwen/decode/Cargo process is active:" >&2
    pgrep -fl "$CONFLICT_PATTERN" >&2 || true
    exit 1
fi

if nc -z 127.0.0.1 "$GRPC_PORT" >/dev/null 2>&1; then
    echo "gRPC port is already in use: $GRPC_PORT" >&2
    exit 1
fi
if nc -z 127.0.0.1 "$HTTP_PORT" >/dev/null 2>&1; then
    echo "HTTP port is already in use: $HTTP_PORT" >&2
    exit 1
fi

if ((BUILD)); then
    cargo build --release --bin qwen3 --bin decode
fi

if [[ ! -x target/release/qwen3 || ! -x target/release/decode ]]; then
    echo "release binaries are missing; remove --no-build or build qwen3 and decode" >&2
    exit 1
fi

machine_id() {
    local display_info chipset gpu_cores normalized
    display_info="$(system_profiler SPDisplaysDataType 2>/dev/null || true)"
    chipset="$(printf '%s\n' "$display_info" | awk -F ': ' '/Chipset Model:/{print $2; exit}')"
    gpu_cores="$(printf '%s\n' "$display_info" | awk -F ': ' '/Total Number of Cores:/{print $2; exit}')"
    if [[ -z "$chipset" || -z "$gpu_cores" ]]; then
        echo unknown
        return
    fi
    normalized="$(printf '%s' "$chipset" | tr '[:upper:] ' '[:lower:]_' | tr -cd '[:alnum:]_')"
    echo "${normalized}_${gpu_cores}_gpu_cores"
}

wait_for_server() {
    local attempts=240
    for _ in $(seq 1 "$attempts"); do
        if ! kill -0 "$ACTIVE_SERVER_PID" >/dev/null 2>&1; then
            return 1
        fi
        if nc -z 127.0.0.1 "$GRPC_PORT" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

run_decode() {
    local label="$1"
    local token_count="$2"
    local run="$3"
    local server_log="$4"
    local output="/tmp/khalamari_qwen3_${label}_${token_count}_${run}.out"
    local log_offset
    log_offset="$(wc -c <"$server_log")"

    if ! target/release/decode \
        --server-url "http://127.0.0.1:${GRPC_PORT}" \
        --max-sampled-tokens "$token_count" \
        --temperature "$TEMPERATURE" \
        --top-k "$TOP_K" \
        --top-p "$TOP_P" \
        --seed "$SEED" \
        --chat-template auto \
        --show-stats \
        --hf-model-dir "$TOKENIZER_DIR" \
        --prompt-str "$PROMPT" >"$output" 2>&1; then
        echo "DECODE_FAILED label=$label max_new=$token_count run=$run output=$output server_log=$server_log" >&2
        tail -n 80 "$output" >&2 || true
        tail -n 120 "$server_log" >&2 || true
        return 1
    fi

    local json_line
    json_line="$(grep '^{' "$output" | tail -n 1 || true)"
    if [[ -z "$json_line" ]]; then
        echo "DECODE_STATS_MISSING label=$label max_new=$token_count run=$run output=$output" >&2
        tail -n 80 "$output" >&2 || true
        return 1
    fi

    JSON_LINE="$json_line" \
        SERVER_LOG="$server_log" \
        SERVER_LOG_OFFSET="$log_offset" \
        python3 - <<'PY'
import json
import os
import re

stats = json.loads(os.environ["JSON_LINE"])
with open(os.environ["SERVER_LOG"], "rb") as handle:
    handle.seek(int(os.environ["SERVER_LOG_OFFSET"]))
    server_log = handle.read().decode("utf-8", errors="replace")
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

acceptance = verified / proposed if proposed else 0.0
acceptance_rate_by_index = [
    verified_count / spec_count if spec_count else 0.0
    for spec_count, verified_count in zip(spec_by_index, verified_by_index)
]
sampled = stats["sampled_tokens"]
chunks = stats["chunk_count"]
encode_counts = lambda values: ":".join(str(value) for value in values) or "-"
encode_rates = lambda values: ":".join(f"{value:.6f}" for value in values) or "-"
print(
    "{:.6f},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{:.6f},{:.6f},{},{},{}".format(
        stats["decode_tokens_per_s"],
        chunks,
        sampled,
        stats["input_tokens"],
        stats["ttft_ms"],
        stats["prompt_tokens_per_s"],
        stats["inter_chunk_p50_ms"],
        stats["inter_chunk_p95_ms"],
        proposed,
        verified,
        acceptance,
        sampled / chunks,
        encode_counts(spec_by_index),
        encode_counts(verified_by_index),
        encode_rates(acceptance_rate_by_index),
    )
)
PY
}

summarize_runs() {
    VALS="$1" \
        INPUTS="$2" \
        CHUNKS="$3" \
        SAMPLES="$4" \
        TTFTS="$5" \
        PROMPT_RATES="$6" \
        P50S="$7" \
        P95S="$8" \
        PROPOSED="$9" \
        VERIFIED="${10}" \
        ACCEPTANCE="${11}" \
        TOKENS_PER_CHUNK="${12}" \
        SPEC_BY_INDEX="${13}" \
        VERIFIED_BY_INDEX="${14}" \
        LABEL="${15}" \
        MAX_NEW="${16}" \
        python3 - <<'PY'
import os
import statistics

floats = lambda name: [float(x) for x in os.environ[name].split()]
strings = lambda name: os.environ[name].split()

rates = floats("VALS")
ttfts = floats("TTFTS")
prompt_rates = floats("PROMPT_RATES")
p50s = floats("P50S")
p95s = floats("P95S")
acceptance = floats("ACCEPTANCE")
tokens_per_chunk = floats("TOKENS_PER_CHUNK")
proposed = strings("PROPOSED")
has_speculation = any(int(value) > 0 for value in proposed)

def sum_index_counts(name):
    totals = []
    for encoded in strings(name):
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
acceptance_rate_by_index_text = (
    "[{}]".format(",".join(f"{value:.6f}" for value in acceptance_rate_by_index))
    if acceptance_rate_by_index
    else "[]"
)

print(
    "SUMMARY label={} max_new={} median_decode_tok_s={:.3f} median_ttft_ms={:.3f} "
    "median_prompt_tok_s={:.3f} median_inter_chunk_p50_ms={:.3f} "
    "median_inter_chunk_p95_ms={:.3f} median_tokens_per_chunk={:.3f} "
    "median_acceptance_rate={} min_decode_tok_s={:.3f} max_decode_tok_s={:.3f} "
    "acceptance_rate_by_index={} runs={} input_tokens={} samples={} chunks={} "
    "proposed_spec={} verified_spec={}".format(
        os.environ["LABEL"],
        os.environ["MAX_NEW"],
        statistics.median(rates),
        statistics.median(ttfts),
        statistics.median(prompt_rates),
        statistics.median(p50s),
        statistics.median(p95s),
        statistics.median(tokens_per_chunk),
        "{:.6f}".format(statistics.median(acceptance)) if has_speculation else "na",
        min(rates),
        max(rates),
        acceptance_rate_by_index_text,
        ",".join(f"{value:.3f}" for value in rates),
        ",".join(strings("INPUTS")),
        ",".join(strings("SAMPLES")),
        ",".join(strings("CHUNKS")),
        ",".join(proposed),
        ",".join(strings("VERIFIED")),
    )
)
PY
}

run_server_case() {
    local label="$1"
    shift
    local server_log="/tmp/khalamari_qwen3_${label}.log"
    local -a command=("$@")
    local server_rust_log="${RUST_LOG:-info}"
    server_rust_log+=",inference-runtime-service::perf=debug"
    command+=(--logging "$LOGGING")

    printf '==> Starting %s:' "$label"
    printf ' %q' "${command[@]}"
    printf '\n'

    : >"$server_log"
    RUST_LOG="$server_rust_log" "${command[@]}" >"$server_log" 2>&1 &
    ACTIVE_SERVER_PID=$!

    if ! wait_for_server; then
        echo "SERVER_START_FAILED label=$label server_log=$server_log" >&2
        tail -n 120 "$server_log" >&2 || true
        cleanup_server
        exit 1
    fi

    for token_count in "${TOKEN_COUNTS[@]}"; do
        local vals="" inputs="" chunks="" samples="" ttfts="" prompt_rates=""
        local p50s="" p95s="" proposed="" verified="" acceptance="" tokens_per_chunk=""
        local spec_by_index="" verified_by_index=""

        for run in $(seq 1 "$RUNS"); do
            local parsed rate chunk_count sampled input_count ttft prompt_rate p50 p95
            local proposed_count verified_count acceptance_rate tpc
            local run_spec_by_index run_verified_by_index acceptance_rate_by_index
            parsed="$(run_decode "$label" "$token_count" "$run" "$server_log")"
            IFS=, read -r rate chunk_count sampled input_count ttft prompt_rate p50 p95 \
                proposed_count verified_count acceptance_rate tpc \
                run_spec_by_index run_verified_by_index acceptance_rate_by_index <<<"$parsed"

            vals+=" $rate"
            inputs+=" $input_count"
            chunks+=" $chunk_count"
            samples+=" $sampled"
            ttfts+=" $ttft"
            prompt_rates+=" $prompt_rate"
            p50s+=" $p50"
            p95s+=" $p95"
            proposed+=" $proposed_count"
            verified+=" $verified_count"
            acceptance+=" $acceptance_rate"
            tokens_per_chunk+=" $tpc"
            spec_by_index+=" $run_spec_by_index"
            verified_by_index+=" $run_verified_by_index"

            echo "RUN label=$label max_new=$token_count run=$run input_tokens=$input_count sampled=$sampled chunks=$chunk_count proposed_spec=$proposed_count verified_spec=$verified_count acceptance_rate=$acceptance_rate acceptance_rate_by_index=$acceptance_rate_by_index tokens_per_chunk=$tpc decode_tok_s=$rate ttft_ms=$ttft prompt_tok_s=$prompt_rate inter_chunk_p50_ms=$p50 inter_chunk_p95_ms=$p95"
        done

        summarize_runs "$vals" "$inputs" "$chunks" "$samples" "$ttfts" \
            "$prompt_rates" "$p50s" "$p95s" "$proposed" "$verified" \
            "$acceptance" "$tokens_per_chunk" "$spec_by_index" "$verified_by_index" \
            "$label" "$token_count"
    done

    cleanup_server
}


run_case() {
    case "$1" in
    14b_off)
        run_server_case 14b_off \
            target/release/qwen3 \
            --grpc-listen-addr "127.0.0.1:${GRPC_PORT}" \
            --http-listen-addr "127.0.0.1:${HTTP_PORT}" \
            --hf-model-dir "$MODEL_DIR" \
            --num-cache-pages "$NUM_CACHE_PAGES" \
            --max-requests "$MAX_REQUESTS" \
            --max-tokens "$MAX_TOKENS" \
            --max-tokens-per-request "$MAX_TOKENS_PER_REQUEST"
        ;;
    esac
}

GIT_COMMIT="$(git rev-parse --verify HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY=unknown
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    [[ -z "$(git status --porcelain)" ]] && GIT_DIRTY=0 || GIT_DIRTY=1
fi
OS_VERSION="$(sw_vers -productVersion 2>/dev/null || uname -s)"
ARCH="$(uname -m)"
MACHINE="$(machine_id)"
PROMPT_SHA256="$(printf '%s' "$PROMPT" | shasum -a 256 | awk '{print $1}')"

echo "CONFIG commit=$GIT_COMMIT dirty=$GIT_DIRTY machine=$MACHINE os=$OS_VERSION arch=$ARCH model=$MODEL_DIR tokenizer=$TOKENIZER_DIR cases=$CASES max_new=$TOKENS runs=$RUNS build=$BUILD grpc_port=$GRPC_PORT http_port=$HTTP_PORT num_cache_pages=$NUM_CACHE_PAGES cache_block_tokens=$CACHE_BLOCK_TOKENS max_requests=$MAX_REQUESTS max_tokens=$MAX_TOKENS max_tokens_per_request=$MAX_TOKENS_PER_REQUEST case_cooldown_secs=$CASE_COOLDOWN_SECS logging=$LOGGING seed=$SEED temperature=$TEMPERATURE top_k=$TOP_K top_p=$TOP_P enable_thinking=1 prompt_sha256=$PROMPT_SHA256 prompt_chars=${#PROMPT}"

for index in "${!SELECTED_CASES[@]}"; do
    case_name="${SELECTED_CASES[$index]}"
    if ((index > 0 && CASE_COOLDOWN_SECS > 0)); then
        echo "COOLDOWN before=$case_name seconds=$CASE_COOLDOWN_SECS"
        sleep "$CASE_COOLDOWN_SECS"
    fi
    run_case "$case_name"
done
