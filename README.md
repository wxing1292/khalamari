# psi-dec

`psi-dec` is a Qwen inference engine for Apple Silicon.
It combines a model-agnostic Rust runtime, a Qwen executor, and a Metal replay backend.
It supports text generation and Qwen3-ASR audio transcription.

## Architecture

```text
HTTP Chat Completions ──► Qwen codec ─────────► token IDs ───┐
HTTP Transcriptions ────► Qwen3-ASR processor ─► resources ──┼─► Inference::decode
gRPC Decode ───────────────────────────────────► token IDs ──┘          │
                                                                        ▼
                          Runtime core ──► Qwen executor ──► Metal backend
                               ▲                                    │
                               └──────── tokens + lifecycle ────────┘
```

| Layer | Owner |
| --- | --- |
| Runtime core | Scheduling, request/cache lifecycle |
| Qwen executor | Model layout, components, sampling, Spec roles, replay order |
| Metal backend | Devices, buffers, kernels, recording, indirect command buffer (ICB) submission |

## Quick start

Requirements: Apple Silicon Mac, Rust toolchain, Xcode command-line tools, and Hugging Face CLI with checkpoint access.

Run the commands from the repository root. This example uses Qwen3.8 dense 27B with multi-token prediction (MTP).
See [other service modes](#other-service-modes) for sparse, and DFlash2 examples.

### Download checkpoints

Download the matching Main and MTP checkpoints:

```sh
hf auth login

hf download mlx-community/Qwen3.8-27B-4bit \
  --local-dir models/Qwen3.8-27B-4bit

hf download mlx-community/Qwen3.8-27B-MTP-4bit \
  --local-dir models/Qwen3.8-27B-MTP-4bit
```

### Start the service

Start the dense 27B service with MTP:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.8-27B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.8-27B-MTP-4bit" \
  --spec-type mtp \
  --num-spec-tokens 1
```

The service unloads model state and weights after 300 seconds without executable model work.
The next request reloads them automatically.

### Send a request

Stream an HTTP Chat Completions response:

```sh
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
    "stream": true,
    "stream_options": {"include_usage": true},
    "max_completion_tokens": 16,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "seed": 1,
    "enable_thinking": false
  }'
```

The [service guide](docs/service.md) also covers Main-only startup, Qwen3-ASR, gRPC, tool calls, and the HTTP APIs.

## Other service modes

The `qwen3_5_*` binaries support compatible Qwen3.5, Qwen3.6, and Qwen3.8 checkpoints.

Each startup example uses the same listen addresses. Run one service at a time.

### Sparse 35B-A3B with MTP

Download the matching Main and MTP checkpoints:

```sh
hf download mlx-community/Qwen3.6-35B-A3B-4bit \
  --local-dir models/Qwen3.6-35B-A3B-4bit

hf download mlx-community/Qwen3.6-35B-A3B-MTP-4bit \
  --local-dir models/Qwen3.6-35B-A3B-MTP-4bit
```

Start the sparse 35B-A3B service with MTP:

```sh
cargo run --release --bin qwen3_5_sparse -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.6-35B-A3B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.6-35B-A3B-MTP-4bit" \
  --spec-type mtp \
  --num-spec-tokens 1
```

### DFlash2

DFlash2 support is experimental.
See the [service guide](docs/service.md#binaries-and-checkpoints) for checkpoint and compatibility contracts.

Download the BF16 speculative checkpoint from
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2):

```sh
hf download z-lab/Qwen3.8-27B-DFlash2 \
  --local-dir models/Qwen3.8-27B-DFlash2
```

Convert the source checkpoint to the psi-dec affine format:

```sh
cargo run --release -p inference-executor-core --bin qwen3x_spec_quantize -- dflash2 \
  --input-dir "$PWD/models/Qwen3.8-27B-DFlash2" \
  --output-dir "$PWD/models/Qwen3.8-27B-DFlash2-affine"
```

Each output directory must not exist before conversion.
The command uses group size 64 and 4-bit matrix payloads by default.
DFlash2 uses 6 bits for the selected layer 2 and layer 4 projections.
The format stores affine scales and biases as BF16.
See the [service guide](docs/service.md) for the full conversion and startup contracts.

Start the dense 27B service with DFlash2:

```sh
cargo run --release --bin qwen3_5_dense -- \
  --grpc-listen-addr 127.0.0.1:50061 \
  --http-listen-addr 127.0.0.1:8000 \
  --hf-model-dir "$PWD/models/Qwen3.8-27B-4bit" \
  --hf-spec-model-dir "$PWD/models/Qwen3.8-27B-DFlash2-affine" \
  --spec-type dflash2
```

## Reference performance

The latest user-supplied Main-only (Off) results were received on 2026-10-14, after the short-context GQA selection fix.
`27B` is Qwen3.8-27B. `35B` is Qwen3.6-35B-A3B.
The session machine is Apple M3 Max with 40 GPU cores and 48 GiB unified memory.
Values are decode tokens/s. The token limit is an output limit, not a fixed output length.

| Model | Prompt | Limit 256 | Limit 1024 | Output at limit 1024 |
| --- | --- | ---: | ---: | ---: |
| 35B | GSM8K | 96.146 | 94.874 | 1024 |
| 35B | Chat | 97.135 | 94.985 | 1024 |
| 27B | GSM8K | 22.730 | 22.760 | 279 |
| 27B | Chat | 22.834 | 22.236 | 1024 |

All limit-256 cases emitted 256 tokens. Every chunk contained one token.
The [performance record](docs/executor_benchmarks.md#main-only-serving-results-2026-10-14) retains all eight cases and their provenance limits.
The supplied summary does not include run count, variance, or memory telemetry.
These observations are not performance thresholds.

Historical Spec results use the helper's 2026-09-19 baseline and precede current GQA/matmul changes.
See the [perf script](scripts/qwen35_e2e_decode_perf.sh) for its reference cases and provenance limits.

See the [performance helper guide](docs/service.md#end-to-end-performance-helper) for commands, configuration, and comparison limits.

## Workspace map

```text
inference-error             shared typed errors and error logging
inference-runtime-launcher  Qwen CLI and const-specialized worker builds
inference-runtime-core      scheduling, lifecycle, and cache ownership
inference-runtime-service   inference API, RPC, codecs, and model services
inference-executor-core     backend-neutral model/component contracts
inference-executor-metal    Qwen execution, replay, sampling, MTP and DFlash2
inference-backend-metal     Metal resources, kernels, and ICB runtime
```

All paths above live under `crates/`.

## Documentation

- [Service](docs/service.md): Setup, APIs, operations, and end-to-end checks.
- [Pi provider](agent-plugins/pi/README.md): Install and configure the resident-session Pi extension.
- [Runtime core](docs/core.md): Scheduling, request lifecycle, and cache ownership.
- [Executor](docs/executor.md): Qwen execution and component composition.
- [Qwen executor](docs/executor_qwen.md): Vanilla, MTP and DFlash2 ownership and replay architecture.
- [DFlash2](docs/dflash2_design.md): Persistent history, sliding attention, convolution, selection, and lifecycle.
- [Metal backend](crates/inference-backend-metal/README.md): Resources, kernels, and replay.
- [Verification](docs/executor_benchmarks.md): Correctness, benchmarks, profiling, and performance evidence.
- [Documentation index](docs/README.md): Component guides, engineering rules, and current work.

## Acknowledgements

`psi-dec` is an independent Rust and Metal implementation inspired by these projects:

- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)

The Metal backend adapts existing implementations and uses established GPU work partitions.
The reusable backend components and operators own these adaptations. Model code supplies shapes and model semantics.

| Area | Reference or implementation source |
| --- | --- |
| Steel matrix fragments, tiled matmul, and quantized operand loaders | [MLX](https://github.com/ml-explore/mlx) Metal headers |
| Register-based small-row affine QMM | [oMLX](https://github.com/jundot/omlx), `omlx/patches/qwen35_verify_qmm.py` |
| GDN eight-lane state recurrence | [oMLX](https://github.com/jundot/omlx), `omlx/custom_kernels/qwen35_prefill/gdn.py` |
| GEMM tiling and sliced-K/split-K terminology | [CUTLASS efficient GEMM documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#parallelized-reductions) |

Sliced-K partitions K across SIMDgroups inside one threadgroup. Split-K partitions K across threadgroups.
These work partitions are independent of quantization bit width and implementation language.
The CUTLASS reference describes these work partitions. The Metal kernels use Apple SIMDgroup instructions.
See the [dense MLP design](docs/executor_dense_mlp.md#steel-and-sliced-k) for the tensor and threadgroup diagrams.

[NOTICE](NOTICE) records source versions, copyright notices, licenses, and adaptation details.
The [license directory](licenses) contains the additional license text.

The [Qwen team](https://qwen.ai/) develops the supported models.
Model weights are separate artifacts with their own terms.

## License

`psi-dec` is distributed under the [MIT License](LICENSE).
See [NOTICE](NOTICE) for third-party attribution and licenses.
