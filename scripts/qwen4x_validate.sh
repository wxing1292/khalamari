#!/bin/bash
# GPU correctness only. Run this script when the GPU is available.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ "${1:-}" == "--help" ]]; then
    echo 'Usage: bash scripts/qwen4x_validate.sh [MODEL_DIR] [LAYERS|all] [TOKEN_IDS] [MTP_DIR] [DEPTH]'
    echo 'Runs serial fixed/random CPU-reference parity, mixed batches, native MTP fixtures, prefix restore, and hibernation tests.'
    echo 'An optional checkpoint capture defaults to eight layers and tokens 1,2,3,4,5,6,7,8.'
    exit 0
fi
out_dir="target/validation/qwen4x/$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$out_dir"
git rev-parse HEAD > "$out_dir/commit.txt"
git status --short > "$out_dir/dirty.txt"
system_profiler SPHardwareDataType > "$out_dir/hardware.txt"
# No tests or benchmarks execute concurrently.
cargo test --release -p inference-executor-core --lib model::qwen::v4_x -- --test-threads=1 2>&1 | tee "$out_dir/config.log"
cargo test --release -p inference-executor-core --lib gated_residual:: -- --test-threads=1 2>&1 | tee "$out_dir/gr-reference.log"
cargo test --release -p inference-executor-core --lib attn::qsa_indexer:: -- --test-threads=1 2>&1 | tee "$out_dir/indexer-reference.log"
cargo test --release -p inference-backend-metal --lib operators::softmax:: -- --test-threads=1 2>&1 | tee "$out_dir/router-softmax.log"
cargo test --release -p inference-backend-metal --lib components::moe:: -- --test-threads=1 2>&1 | tee "$out_dir/moe-kernels.log"
cargo test --release -p inference-backend-metal --lib components::gated_residual:: -- --test-threads=1 2>&1 | tee "$out_dir/gr-kernels.log"
cargo test --release -p inference-backend-metal --lib components::qsa_indexer:: -- --test-threads=1 2>&1 | tee "$out_dir/indexer-kernels.log"
cargo test --release -p inference-backend-metal --lib components::qsa_sparse_gqa:: -- --test-threads=1 2>&1 | tee "$out_dir/sparse-gqa-kernels.log"
cargo test --release -p inference-executor-metal --lib gated_residual:: -- --test-threads=1 2>&1 | tee "$out_dir/gr.log"
cargo test --release -p inference-executor-metal --lib attn::qsa_indexer:: -- --test-threads=1 2>&1 | tee "$out_dir/indexer.log"
cargo test --release -p inference-executor-metal --lib attn::gdn::backend:: -- --test-threads=1 2>&1 | tee "$out_dir/gdn.log"
cargo test --release -p inference-executor-metal --lib mlp::moe:: -- --test-threads=1 2>&1 | tee "$out_dir/moe.log"
cargo test --release -p inference-executor-metal --lib model::qwen::v4_x -- --test-threads=1 --nocapture 2>&1 | tee "$out_dir/first8.log"
cargo test --release -p inference-executor-metal --lib attn::qsa:: -- --test-threads=1 2>&1 | tee "$out_dir/qsa.log"
cargo test --release -p inference-backend-metal --lib components::ple:: -- --test-threads=1 2>&1 | tee "$out_dir/ple-kernels.log"
cargo test --release -p inference-executor-metal --lib ple::backend::tests:: -- --test-threads=1 2>&1 | tee "$out_dir/ple.log"
if [[ $# -gt 0 ]]; then
    capture_args=("$1" "${2:-8}" "${3:-1,2,3,4,5,6,7,8}")
    if [[ $# -ge 4 ]]; then capture_args+=("$4" "${5:-1}"); fi
    cargo run --release -p inference-executor-metal --example qwen4x_validate -- "${capture_args[@]}" > "$out_dir/main.json"
fi
printf 'Validation results: %s\n' "$repo_dir/$out_dir"
