# Component latency on M3 Max

This report records existing measurements from 2026-10-14. It does not record a new run after the source audit.
The [JSON dataset](component_performance.json) retains all 560 logical rows, exact values, measurement identity, and separate policy probes.

This snapshot precedes the single-query short-context GQA selection correction.
The [latest Main-only serving summary](executor_benchmarks.md#main-only-serving-results-2026-10-14) records the later user-supplied Off results.
Those serving results do not replace or remeasure this component dataset.

## Source and environment

- Integrated MoE source: `0fd531cd6b0f49bf23c61dabb5e4020e42e72521`.
- Measured source: `705aaf3c30ae20b622ec88d2041b326233009a5f` plus the uncommitted MoE candidate.
- Hardware: Apple M3 Max, 40 GPU cores, 48 GiB unified memory, AC power. The desktop was active.
- GQA/GDN/dense checkpoint: `Qwen3.8-27B-4bit`. Sparse MLP/MoE checkpoint: `Qwen3.6-35B-A3B-4bit`.
- The TensorFold values describe its Python lane. They do not describe its native engine.
- Each group ran PSI/oMLX/TF/TF/oMLX/PSI serially. Accepted groups have zero new swap and normal memory pressure.
- Small-input runs used 3 warmup iterations, 5 measured iterations, and 3 runs. Full Prefill used 1 warmup iteration, 2 measured iterations, and 3 runs.
- Latency is ordinary submission-and-wait wall time. Loading, compilation, and preparation are outside timing.
- Each value is the mean of two per-pass medians. Drift is the largest same-engine change between passes.

These framework versions apply to every shape table below.
The JSON dataset retains the worker hashes and dependency versions.

| Table label | Framework and execution path | Measured version or source | Dependencies |
| --- | --- | --- | --- |
| PSI | psi-dec Rust/Metal | `705aaf3c30ae20b622ec88d2041b326233009a5f` plus the uncommitted MoE candidate | See worker hashes in the dataset |
| oMLX | oMLX Python/MLX | `0.7.1.dev1` | MLX `0.32.3`; MLX-LM `0.31.4.dev132+g94cdcae13`; MLX-VLM `0.7.7` |
| TF Python | TensorFold Python/MLX | `0.6.6` | MLX `0.32.3`; MLX-LM `0.32.0` |

## Measurement scope

```text
Hidden[B,Q,H] --> one complete component --> Output[B,Q,H]

MoE:
  hidden -- router/top-k -- layout/pack -- sparse gate/up/SiLU/down --+
         -- shared expert + shared gate --------------------------+--> combine/scatter

Sparse MLP:
  fixed packed routes -- gate/up/SiLU -- down --> packed output
  routing, packing, shared expert, and combine are outside timing

Ctx axis:
  GQA       --> paged visibility and zero KV-cache capacity
  GDN       --> zero or fixed nonzero initial state
  MLP / MoE --> no ctx input; reuse the same hidden-input measurement
```

These values are component forwards. They do not include complete model or service E2E latency.
The hidden input is `BF16(((flat_index % 23)-11)*0.03125)`.
GQA uses PSI FP8 KV and peer BF16 KV. This is a native-path comparison with different cache precision.
GDN uses the same BF16-valued state fixture. PSI writes final BF16 snapshots. Peers update F32 live recurrence.
Sparse MLP uses identical expert IDs and packed order. Complete MoE uses native routing, which can select different expert IDs.
Peer request/sequence shapes remain intact. Only ctx-equivalent rows are collapsed in the tables.
Full Prefill means one component forward from zero history. Different real prefixes were not constructed.

## Read the tables

All latencies use milliseconds. Negative differences mean PSI is faster.
Each difference is `100 * (PSI latency / peer latency - 1)`.
Requests are batch members. Q is the number of input tokens per request.
Ctx `any` covers `0,1024,2048,4096,8192,16384,32768,65536` for the same hidden input.
GDN ctx `nonzero` covers the positive values of that axis with one fixed state fixture.
The JSON dataset retains the expanded ctx rows and their measurement keys.
Recommendation: Interpret a small difference cautiously when it is below the displayed drift.
One GQA case has no PSI latency because its partial-state capacity exceeds the shader index domain. Its peer values remain visible.
The eight-request GQA full-Prefill retries use shape-specific capacities after the shared maximum capacity rejected the initial group.

## GQA

| Operation | Req | Ctx | Q/req | PSI ms | oMLX ms | TF Python ms | PSI vs oMLX | PSI vs TF | Drift |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| append | 1 | 0 | 64 | 1.626 | 1.728 | 2.073 | -5.9% | -21.5% | 5.8% |
| append | 1 | 0 | 128 | 2.844 | 2.928 | 3.617 | -2.9% | -21.4% | 3.8% |
| append | 1 | 0 | 256 | 5.287 | 5.381 | 5.434 | -1.7% | -2.7% | 0.3% |
| append | 1 | 1024 | 64 | 1.863 | 1.919 | 2.825 | -2.9% | -34.1% | 1.5% |
| append | 1 | 1024 | 128 | 3.362 | 3.235 | 5.088 | +3.9% | -33.9% | 0.4% |
| append | 1 | 1024 | 256 | 6.313 | 5.989 | 5.983 | +5.4% | +5.5% | 0.3% |
| append | 1 | 2048 | 64 | 2.055 | 2.388 | 3.253 | -13.9% | -36.8% | 3.2% |
| append | 1 | 2048 | 128 | 3.749 | 3.842 | 5.974 | -2.4% | -37.2% | 0.6% |
| append | 1 | 2048 | 256 | 7.140 | 6.925 | 6.569 | +3.1% | +8.7% | 0.4% |
| append | 1 | 4096 | 64 | 2.545 | 3.118 | 4.268 | -18.4% | -40.4% | 2.6% |
| append | 1 | 4096 | 128 | 4.751 | 4.803 | 7.934 | -1.1% | -40.1% | 0.8% |
| append | 1 | 4096 | 256 | 8.466 | 8.507 | 8.057 | -0.5% | +5.1% | 0.2% |
| append | 1 | 8192 | 64 | 3.284 | 4.576 | 7.138 | -28.2% | -54.0% | 1.2% |
| append | 1 | 8192 | 128 | 6.104 | 6.737 | 15.345 | -9.4% | -60.2% | 8.7% |
| append | 1 | 8192 | 256 | 11.426 | 11.689 | 10.318 | -2.2% | +10.7% | 0.3% |
| append | 1 | 16384 | 64 | 5.020 | 7.494 | 11.169 | -33.0% | -55.1% | 0.9% |
| append | 1 | 16384 | 128 | 9.208 | 10.699 | 22.340 | -13.9% | -58.8% | 0.5% |
| append | 1 | 16384 | 256 | 17.388 | 18.217 | 15.418 | -4.6% | +12.8% | 0.2% |
| append | 1 | 32768 | 64 | 8.048 | 13.540 | 21.053 | -40.6% | -61.8% | 0.6% |
| append | 1 | 32768 | 128 | 15.150 | 18.616 | 41.050 | -18.6% | -63.1% | 0.2% |
| append | 1 | 32768 | 256 | 29.269 | 31.075 | 27.563 | -5.8% | +6.2% | 0.5% |
| append | 1 | 65536 | 64 | 14.189 | 25.807 | 47.958 | -45.0% | -70.4% | 2.2% |
| append | 1 | 65536 | 128 | 27.295 | 34.418 | 83.694 | -20.7% | -67.4% | 1.6% |
| append | 1 | 65536 | 256 | 53.602 | 56.824 | 53.846 | -5.7% | -0.5% | 0.7% |
| append | 2 | 0 | 64 | 2.805 | 2.937 | 3.499 | -4.5% | -19.8% | 1.0% |
| append | 2 | 0 | 128 | 5.201 | 5.381 | 6.467 | -3.3% | -19.6% | 0.5% |
| append | 2 | 0 | 256 | 10.122 | 10.282 | 10.405 | -1.6% | -2.7% | 0.1% |
| append | 2 | 1024 | 64 | 3.304 | 3.248 | 5.193 | +1.7% | -36.4% | 0.4% |
| append | 2 | 1024 | 128 | 6.088 | 5.922 | 11.138 | +2.8% | -45.3% | 0.3% |
| append | 2 | 1024 | 256 | 11.854 | 11.437 | 11.390 | +3.6% | +4.1% | 0.1% |
| append | 2 | 2048 | 64 | 3.745 | 3.823 | 6.178 | -2.0% | -39.4% | 1.1% |
| append | 2 | 2048 | 128 | 7.017 | 6.887 | 13.044 | +1.9% | -46.2% | 0.4% |
| append | 2 | 2048 | 256 | 13.825 | 13.156 | 12.563 | +5.1% | +10.1% | 0.3% |
| append | 2 | 4096 | 64 | 4.489 | 4.786 | 8.315 | -6.2% | -46.0% | 1.2% |
| append | 2 | 4096 | 128 | 8.424 | 8.501 | 17.593 | -0.9% | -52.1% | 0.1% |
| append | 2 | 4096 | 256 | 16.396 | 16.217 | 15.102 | +1.1% | +8.6% | 0.4% |
| append | 2 | 8192 | 64 | 6.066 | 6.757 | 14.168 | -10.2% | -57.2% | 0.6% |
| append | 2 | 8192 | 128 | 11.676 | 11.717 | 29.264 | -0.4% | -60.1% | 1.5% |
| append | 2 | 8192 | 256 | 22.216 | 22.517 | 20.102 | -1.3% | +10.5% | 0.2% |
| append | 2 | 16384 | 64 | 9.057 | 10.729 | 22.601 | -15.6% | -59.9% | 0.1% |
| append | 2 | 16384 | 128 | 17.756 | 18.372 | 44.675 | -3.4% | -60.3% | 0.9% |
| append | 2 | 16384 | 256 | 33.914 | 34.884 | 31.661 | -2.8% | +7.1% | 1.1% |
| append | 2 | 32768 | 64 | 15.459 | 18.706 | 45.382 | -17.4% | -65.9% | 0.5% |
| append | 2 | 32768 | 128 | 29.608 | 31.511 | 86.393 | -6.0% | -65.7% | 0.4% |
| append | 2 | 32768 | 256 | 57.319 | 59.676 | 54.993 | -4.0% | +4.2% | 0.0% |
| append | 2 | 65536 | 64 | 27.600 | 34.592 | 94.715 | -20.2% | -70.9% | 4.5% |
| append | 2 | 65536 | 128 | 53.869 | 57.592 | 172.770 | -6.5% | -68.8% | 0.4% |
| append | 2 | 65536 | 256 | 105.123 | 109.220 | 108.961 | -3.8% | -3.5% | 0.1% |
| append | 4 | 0 | 64 | 5.149 | 5.330 | 6.260 | -3.4% | -17.7% | 1.0% |
| append | 4 | 0 | 128 | 9.966 | 10.181 | 11.854 | -2.1% | -15.9% | 0.3% |
| append | 4 | 0 | 256 | 19.720 | 20.013 | 19.940 | -1.5% | -1.1% | 0.1% |
| append | 4 | 1024 | 64 | 6.076 | 5.906 | 10.253 | +2.9% | -40.7% | 1.1% |
| append | 4 | 1024 | 128 | 11.749 | 11.302 | 19.425 | +4.0% | -39.5% | 0.4% |
| append | 4 | 1024 | 256 | 23.301 | 22.266 | 22.253 | +4.6% | +4.7% | 0.0% |
| append | 4 | 2048 | 64 | 7.009 | 6.851 | 12.615 | +2.3% | -44.4% | 0.3% |
| append | 4 | 2048 | 128 | 13.576 | 13.060 | 24.990 | +4.0% | -45.7% | 3.6% |
| append | 4 | 2048 | 256 | 27.254 | 25.592 | 24.962 | +6.5% | +9.2% | 0.2% |
| append | 4 | 4096 | 64 | 8.407 | 8.479 | 17.446 | -0.8% | -51.8% | 1.4% |
| append | 4 | 4096 | 128 | 16.334 | 16.115 | 33.398 | +1.4% | -51.1% | 1.4% |
| append | 4 | 4096 | 256 | 32.324 | 31.754 | 30.304 | +1.8% | +6.7% | 0.1% |
| append | 4 | 8192 | 64 | 11.651 | 11.739 | 33.078 | -0.8% | -64.8% | 4.6% |
| append | 4 | 8192 | 128 | 22.821 | 22.465 | 57.003 | +1.6% | -60.0% | 1.6% |
| append | 4 | 8192 | 256 | 43.822 | 43.746 | 40.918 | +0.2% | +7.1% | 0.2% |
| append | 4 | 16384 | 64 | 17.544 | 18.438 | 50.913 | -4.8% | -65.5% | 0.2% |
| append | 4 | 16384 | 128 | 34.860 | 34.966 | 94.591 | -0.3% | -63.1% | 1.1% |
| append | 4 | 16384 | 256 | 67.085 | 67.739 | 61.754 | -1.0% | +8.6% | 0.3% |
| append | 4 | 32768 | 64 | 30.387 | 31.636 | 97.080 | -3.9% | -68.7% | 0.7% |
| append | 4 | 32768 | 128 | 58.383 | 59.972 | 179.099 | -2.6% | -67.4% | 0.3% |
| append | 4 | 32768 | 256 | 114.030 | 115.884 | 108.313 | -1.6% | +5.3% | 0.8% |
| append | 4 | 65536 | 64 | 55.209 | 57.948 | 185.395 | -4.7% | -70.2% | 0.2% |
| append | 4 | 65536 | 128 | 106.520 | 109.986 | 345.679 | -3.2% | -69.2% | 0.3% |
| append | 4 | 65536 | 256 | 208.410 | 213.018 | 211.270 | -2.2% | -1.4% | 0.2% |
| append | 8 | 0 | 64 | 9.882 | 10.123 | 11.719 | -2.4% | -15.7% | 0.1% |
| append | 8 | 0 | 128 | 19.437 | 19.750 | 22.840 | -1.6% | -14.9% | 0.1% |
| append | 8 | 0 | 256 | 37.270 | 39.584 | 39.366 | -5.8% | -5.3% | 0.1% |
| append | 8 | 1024 | 64 | 11.720 | 11.226 | 22.648 | +4.4% | -48.3% | 9.4% |
| append | 8 | 1024 | 128 | 23.087 | 21.953 | 41.410 | +5.2% | -44.2% | 5.0% |
| append | 8 | 1024 | 256 | 44.527 | 44.608 | 44.588 | -0.2% | -0.1% | 0.1% |
| append | 8 | 2048 | 64 | 13.133 | 13.031 | 26.747 | +0.8% | -50.9% | 2.2% |
| append | 8 | 2048 | 128 | 25.817 | 25.427 | 51.292 | +1.5% | -49.7% | 6.7% |
| append | 8 | 2048 | 256 | 49.812 | 50.923 | 50.900 | -2.2% | -2.1% | 0.1% |
| append | 8 | 4096 | 64 | 16.325 | 16.143 | 35.622 | +1.1% | -54.2% | 8.1% |
| append | 8 | 4096 | 128 | 32.179 | 31.632 | 67.953 | +1.7% | -52.6% | 3.6% |
| append | 8 | 4096 | 256 | 62.496 | 62.465 | 61.247 | +0.0% | +2.0% | 0.0% |
| append | 8 | 8192 | 64 | 22.101 | 22.553 | 66.697 | -2.0% | -66.9% | 3.1% |
| append | 8 | 8192 | 128 | 43.613 | 43.664 | 122.911 | -0.1% | -64.5% | 0.4% |
| append | 8 | 8192 | 256 | 85.419 | 85.697 | 82.424 | -0.3% | +3.6% | 0.1% |
| append | 8 | 16384 | 64 | 34.474 | 35.110 | 105.848 | -1.8% | -67.4% | 0.3% |
| append | 8 | 16384 | 128 | 69.182 | 67.832 | 194.478 | +2.0% | -64.4% | 0.1% |
| append | 8 | 16384 | 256 | 131.740 | 133.966 | 123.346 | -1.7% | +6.8% | 0.1% |
| append | 8 | 32768 | 64 | 60.085 | 60.331 | 194.778 | -0.4% | -69.2% | 1.6% |
| append | 8 | 32768 | 128 | 116.216 | 116.268 | 361.586 | -0.0% | -67.9% | 0.1% |
| append | 8 | 32768 | 256 | 225.211 | 228.260 | 222.351 | -1.3% | +1.3% | 0.4% |
| append | 8 | 65536 | 64 | 109.547 | 110.728 | 374.198 | -1.1% | -70.7% | 1.1% |
| append | 8 | 65536 | 128 | 212.375 | 213.910 | 719.860 | -0.7% | -70.5% | 0.6% |
| append | 8 | 65536 | 256 | 413.076 | 415.893 | 438.285 | -0.7% | -5.8% | 0.0% |
| full | 1 | 0 | 2048 | 43.573 | 42.923 | 43.703 | +1.5% | -0.3% | 0.1% |
| full | 1 | 0 | 4096 | 101.352 | 100.387 | 98.652 | +1.0% | +2.7% | 0.2% |
| full | 1 | 0 | 8192 | 247.635 | 257.495 | 245.428 | -3.8% | +0.9% | 1.2% |
| full | 1 | 0 | 16384 | 664.760 | 673.230 | 658.050 | -1.3% | +1.0% | 0.9% |
| full | 2 | 0 | 2048 | 86.706 | 88.084 | 87.341 | -1.6% | -0.7% | 0.1% |
| full | 2 | 0 | 4096 | 194.555 | 204.019 | 196.302 | -4.6% | -0.9% | 0.5% |
| full | 2 | 0 | 8192 | 494.784 | 496.998 | 488.712 | -0.4% | +1.2% | 0.5% |
| full | 2 | 0 | 16384 | 1329.377 | 1325.230 | 1288.327 | +0.3% | +3.2% | 0.7% |
| full | 4 | 0 | 2048 | 173.057 | 177.417 | 173.665 | -2.5% | -0.4% | 0.1% |
| full | 4 | 0 | 4096 | 388.529 | 404.821 | 401.725 | -4.0% | -3.3% | 0.3% |
| full | 4 | 0 | 8192 | 988.875 | 970.827 | 964.198 | +1.9% | +2.6% | 0.5% |
| full | 4 | 0 | 16384 | 2658.322 | 2638.959 | 2662.242 | +0.7% | -0.1% | 0.3% |
| full | 8 | 0 | 2048 | 337.873 | 364.763 | 355.209 | -7.4% | -4.9% | 2.7% |
| full | 8 | 0 | 4096 | 780.371 | 795.859 | 814.828 | -1.9% | -4.2% | 2.7% |
| full | 8 | 0 | 8192 | 1914.441 | 1920.188 | 1935.800 | -0.3% | -1.1% | 0.8% |
| full | 8 | 0 | 16384 | N/A: index domain | 5227.271 | 5403.858 | — | — | 0.3% |

## GDN

| Operation | Req | Ctx | Q/req | PSI ms | oMLX ms | TF Python ms | PSI vs oMLX | PSI vs TF | Drift |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| append | 1 | 0 | 64 | 1.808 | 2.009 | 1.930 | -10.0% | -6.3% | 0.8% |
| append | 1 | 0 | 128 | 3.287 | 3.341 | 3.274 | -1.6% | +0.4% | 0.3% |
| append | 1 | 0 | 256 | 6.031 | 6.023 | 6.010 | +0.1% | +0.4% | 0.1% |
| append | 1 | nonzero | 64 | 1.804 | 1.991 | 1.928 | -9.4% | -6.5% | 0.6% |
| append | 1 | nonzero | 128 | 3.289 | 3.346 | 3.280 | -1.7% | +0.3% | 0.3% |
| append | 1 | nonzero | 256 | 6.038 | 6.034 | 6.004 | +0.1% | +0.6% | 0.4% |
| append | 2 | 0 | 64 | 3.287 | 3.373 | 3.355 | -2.5% | -2.0% | 0.8% |
| append | 2 | 0 | 128 | 6.030 | 6.063 | 6.092 | -0.6% | -1.0% | 0.1% |
| append | 2 | 0 | 256 | 11.606 | 11.533 | 11.629 | +0.6% | -0.2% | 0.1% |
| append | 2 | nonzero | 64 | 3.281 | 3.367 | 3.345 | -2.6% | -1.9% | 0.5% |
| append | 2 | nonzero | 128 | 6.029 | 6.060 | 6.099 | -0.5% | -1.1% | 0.2% |
| append | 2 | nonzero | 256 | 11.602 | 11.532 | 11.654 | +0.6% | -0.4% | 0.3% |
| append | 4 | 0 | 64 | 6.042 | 6.082 | 6.123 | -0.7% | -1.3% | 0.3% |
| append | 4 | 0 | 128 | 11.595 | 11.553 | 11.687 | +0.4% | -0.8% | 0.2% |
| append | 4 | 0 | 256 | 22.653 | 22.385 | 22.814 | +1.2% | -0.7% | 0.1% |
| append | 4 | nonzero | 64 | 6.034 | 6.090 | 6.122 | -0.9% | -1.4% | 0.5% |
| append | 4 | nonzero | 128 | 11.579 | 11.566 | 11.658 | +0.1% | -0.7% | 0.2% |
| append | 4 | nonzero | 256 | 22.654 | 22.424 | 22.762 | +1.0% | -0.5% | 0.2% |
| append | 8 | 0 | 64 | 11.603 | 11.551 | 11.661 | +0.5% | -0.5% | 0.2% |
| append | 8 | 0 | 128 | 22.661 | 22.428 | 22.675 | +1.0% | -0.1% | 0.2% |
| append | 8 | 0 | 256 | 42.943 | 44.312 | 45.040 | -3.1% | -4.7% | 0.1% |
| append | 8 | nonzero | 64 | 11.596 | 11.571 | 11.648 | +0.2% | -0.4% | 0.1% |
| append | 8 | nonzero | 128 | 22.662 | 22.429 | 22.710 | +1.0% | -0.2% | 0.1% |
| append | 8 | nonzero | 256 | 42.936 | 44.357 | 44.956 | -3.2% | -4.5% | 0.2% |
| full | 1 | 0 | 2048 | 43.133 | 42.698 | 44.719 | +1.0% | -3.5% | 0.3% |
| full | 1 | 0 | 4096 | 85.803 | 85.218 | 89.364 | +0.7% | -4.0% | 0.2% |
| full | 1 | 0 | 8192 | 171.388 | 174.654 | 183.102 | -1.9% | -6.4% | 0.1% |
| full | 1 | 0 | 16384 | 342.334 | 348.189 | 364.823 | -1.7% | -6.2% | 0.0% |
| full | 2 | 0 | 2048 | 85.549 | 85.140 | 89.999 | +0.5% | -4.9% | 0.0% |
| full | 2 | 0 | 4096 | 170.835 | 174.277 | 184.286 | -2.0% | -7.3% | 0.3% |
| full | 2 | 0 | 8192 | 341.583 | 347.861 | 366.578 | -1.8% | -6.8% | 0.3% |
| full | 2 | 0 | 16384 | 682.392 | 695.852 | 732.459 | -1.9% | -6.8% | 0.1% |
| full | 4 | 0 | 2048 | 170.763 | 174.111 | 184.729 | -1.9% | -7.6% | 0.4% |
| full | 4 | 0 | 4096 | 341.222 | 347.618 | 366.467 | -1.8% | -6.9% | 0.1% |
| full | 4 | 0 | 8192 | 681.707 | 697.366 | 732.184 | -2.2% | -6.9% | 0.3% |
| full | 4 | 0 | 16384 | 1361.874 | 1389.651 | 1463.271 | -2.0% | -6.9% | 0.4% |
| full | 8 | 0 | 2048 | 340.823 | 346.600 | 368.261 | -1.7% | -7.5% | 1.2% |
| full | 8 | 0 | 4096 | 680.908 | 695.158 | 734.892 | -2.0% | -7.3% | 0.9% |
| full | 8 | 0 | 8192 | 1360.849 | 1385.252 | 1464.337 | -1.8% | -7.1% | 0.9% |
| full | 8 | 0 | 16384 | 2720.221 | 2782.992 | 2958.037 | -2.3% | -8.0% | 0.2% |

## Dense MLP

| Operation | Req | Ctx | Q/req | PSI ms | oMLX ms | TF Python ms | PSI vs oMLX | PSI vs TF | Drift |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| append | 1 | any | 64 | 3.486 | 3.380 | 3.388 | +3.1% | +2.9% | 0.2% |
| append | 1 | any | 128 | 6.512 | 6.270 | 6.270 | +3.9% | +3.9% | 0.2% |
| append | 1 | any | 256 | 12.430 | 12.115 | 12.094 | +2.6% | +2.8% | 0.2% |
| append | 2 | any | 64 | 6.523 | 6.287 | 6.267 | +3.8% | +4.1% | 0.5% |
| append | 2 | any | 128 | 12.418 | 12.114 | 12.100 | +2.5% | +2.6% | 0.1% |
| append | 2 | any | 256 | 24.313 | 23.796 | 23.781 | +2.2% | +2.2% | 0.2% |
| append | 4 | any | 64 | 12.427 | 12.112 | 12.125 | +2.6% | +2.5% | 0.1% |
| append | 4 | any | 128 | 24.316 | 23.852 | 23.798 | +1.9% | +2.2% | 0.2% |
| append | 4 | any | 256 | 48.152 | 47.212 | 47.183 | +2.0% | +2.1% | 0.5% |
| append | 8 | any | 64 | 24.364 | 23.825 | 23.856 | +2.3% | +2.1% | 0.3% |
| append | 8 | any | 128 | 48.296 | 47.332 | 47.307 | +2.0% | +2.1% | 0.2% |
| append | 8 | any | 256 | 90.278 | 94.313 | 94.403 | -4.3% | -4.4% | 0.1% |
| full | 1 | 0 | 2048 | 90.261 | 89.911 | 94.822 | +0.4% | -4.8% | 1.1% |
| full | 1 | 0 | 4096 | 180.199 | 179.326 | 189.301 | +0.5% | -4.8% | 1.2% |
| full | 1 | 0 | 8192 | 359.904 | 357.920 | 376.806 | +0.6% | -4.5% | 0.4% |
| full | 1 | 0 | 16384 | 719.755 | 715.014 | 760.169 | +0.7% | -5.3% | 2.6% |
| full | 2 | 0 | 2048 | 180.161 | 179.356 | 189.096 | +0.4% | -4.7% | 1.1% |
| full | 2 | 0 | 4096 | 359.712 | 357.980 | 376.056 | +0.5% | -4.3% | 0.4% |
| full | 2 | 0 | 8192 | 718.638 | 715.071 | 751.332 | +0.5% | -4.4% | 0.0% |
| full | 2 | 0 | 16384 | 1436.986 | 1431.284 | 1506.570 | +0.4% | -4.6% | 0.5% |
| full | 4 | 0 | 2048 | 359.562 | 357.961 | 381.321 | +0.4% | -5.7% | 3.2% |
| full | 4 | 0 | 4096 | 718.465 | 715.305 | 761.884 | +0.4% | -5.7% | 2.9% |
| full | 4 | 0 | 8192 | 1436.911 | 1431.632 | 1521.412 | +0.4% | -5.6% | 2.0% |
| full | 4 | 0 | 16384 | 2873.666 | 2866.855 | 3027.159 | +0.2% | -5.1% | 0.0% |
| full | 8 | 0 | 2048 | 718.226 | 715.032 | 766.438 | +0.4% | -6.3% | 4.1% |
| full | 8 | 0 | 4096 | 1437.052 | 1431.544 | 1533.601 | +0.4% | -6.3% | 4.0% |
| full | 8 | 0 | 8192 | 2873.744 | 2865.810 | 3092.090 | +0.3% | -7.1% | 2.3% |
| full | 8 | 0 | 16384 | 5746.474 | 5819.166 | 6285.391 | -1.2% | -8.6% | 0.2% |

## Sparse MLP

| Operation | Req | Ctx | Q/req | PSI ms | oMLX ms | TF Python ms | PSI vs oMLX | PSI vs TF | Drift |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| append | 1 | any | 64 | 1.337 | 1.511 | 1.520 | -11.5% | -12.0% | 6.5% |
| append | 1 | any | 128 | 1.477 | 1.415 | 1.471 | +4.4% | +0.4% | 1.2% |
| append | 1 | any | 256 | 2.118 | 2.085 | 2.119 | +1.6% | -0.0% | 0.7% |
| append | 2 | any | 64 | 1.541 | 1.436 | 1.476 | +7.3% | +4.4% | 3.5% |
| append | 2 | any | 128 | 2.095 | 2.076 | 2.116 | +0.9% | -1.0% | 1.5% |
| append | 2 | any | 256 | 3.337 | 3.271 | 3.294 | +2.0% | +1.3% | 0.4% |
| append | 4 | any | 64 | 2.104 | 2.090 | 2.148 | +0.7% | -2.0% | 1.7% |
| append | 4 | any | 128 | 3.352 | 3.262 | 3.287 | +2.8% | +2.0% | 0.7% |
| append | 4 | any | 256 | 5.686 | 5.538 | 5.535 | +2.7% | +2.7% | 0.6% |
| append | 8 | any | 64 | 3.334 | 3.262 | 3.288 | +2.2% | +1.4% | 0.2% |
| append | 8 | any | 128 | 5.701 | 5.525 | 5.534 | +3.2% | +3.0% | 0.2% |
| append | 8 | any | 256 | 10.319 | 10.370 | 10.380 | -0.5% | -0.6% | 0.5% |
| full | 1 | 0 | 2048 | 10.323 | 10.389 | 10.356 | -0.6% | -0.3% | 0.4% |
| full | 1 | 0 | 4096 | 19.587 | 19.950 | 19.947 | -1.8% | -1.8% | 0.1% |
| full | 1 | 0 | 8192 | 37.985 | 42.314 | 40.339 | -10.2% | -5.8% | 0.5% |
| full | 1 | 0 | 16384 | 74.035 | 76.629 | 77.771 | -3.4% | -4.8% | 0.5% |
| full | 2 | 0 | 2048 | 19.551 | 19.905 | 19.989 | -1.8% | -2.2% | 0.4% |
| full | 2 | 0 | 4096 | 37.965 | 42.227 | 40.356 | -10.1% | -5.9% | 0.3% |
| full | 2 | 0 | 8192 | 74.050 | 76.631 | 77.896 | -3.4% | -4.9% | 1.4% |
| full | 2 | 0 | 16384 | 146.878 | 169.414 | 155.438 | -13.3% | -5.5% | 0.6% |
| full | 4 | 0 | 2048 | 38.128 | 42.542 | 40.902 | -10.4% | -6.8% | 2.2% |
| full | 4 | 0 | 4096 | 74.322 | 79.097 | 80.015 | -6.0% | -7.1% | 5.1% |
| full | 4 | 0 | 8192 | 147.132 | 169.609 | 158.297 | -13.3% | -7.1% | 3.7% |
| full | 4 | 0 | 16384 | 293.411 | 339.730 | 326.295 | -13.6% | -10.1% | 1.7% |
| full | 8 | 0 | 2048 | 74.432 | 77.324 | 79.079 | -3.7% | -5.9% | 1.6% |
| full | 8 | 0 | 4096 | 147.022 | 167.445 | 156.940 | -12.2% | -6.3% | 2.1% |
| full | 8 | 0 | 8192 | 293.304 | 336.367 | 327.649 | -12.8% | -10.5% | 2.0% |
| full | 8 | 0 | 16384 | 584.463 | 685.535 | 652.101 | -14.7% | -10.4% | 4.5% |

## Complete MoE

| Operation | Req | Ctx | Q/req | PSI ms | oMLX ms | TF Python ms | PSI vs oMLX | PSI vs TF | Drift |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| append | 1 | any | 64 | 1.486 | 1.809 | 1.785 | -17.9% | -16.8% | 10.8% |
| append | 1 | any | 128 | 1.764 | 1.795 | 1.822 | -1.7% | -3.2% | 1.7% |
| append | 1 | any | 256 | 2.525 | 2.491 | 2.505 | +1.4% | +0.8% | 0.3% |
| append | 2 | any | 64 | 1.765 | 1.813 | 2.005 | -2.7% | -12.0% | 13.5% |
| append | 2 | any | 128 | 2.516 | 2.492 | 2.509 | +1.0% | +0.3% | 0.7% |
| append | 2 | any | 256 | 4.046 | 4.030 | 4.098 | +0.4% | -1.3% | 0.9% |
| append | 4 | any | 64 | 2.527 | 2.485 | 2.501 | +1.7% | +1.0% | 0.8% |
| append | 4 | any | 128 | 4.047 | 4.034 | 4.128 | +0.3% | -2.0% | 0.3% |
| append | 4 | any | 256 | 6.869 | 6.967 | 7.026 | -1.4% | -2.2% | 0.7% |
| append | 8 | any | 64 | 4.052 | 4.037 | 4.098 | +0.4% | -1.1% | 1.0% |
| append | 8 | any | 128 | 6.838 | 6.965 | 6.999 | -1.8% | -2.3% | 0.7% |
| append | 8 | any | 256 | 12.551 | 13.404 | 13.203 | -6.4% | -4.9% | 0.4% |
| full | 1 | 0 | 2048 | 12.546 | 12.492 | 13.176 | +0.4% | -4.8% | 0.6% |
| full | 1 | 0 | 4096 | 23.898 | 24.362 | 26.012 | -1.9% | -8.1% | 0.3% |
| full | 1 | 0 | 8192 | 46.589 | 49.903 | 51.687 | -6.6% | -9.9% | 2.4% |
| full | 1 | 0 | 16384 | 91.426 | 110.032 | 115.556 | -16.9% | -20.9% | 2.6% |
| full | 2 | 0 | 2048 | 23.903 | 24.436 | 26.024 | -2.2% | -8.1% | 0.2% |
| full | 2 | 0 | 4096 | 46.596 | 49.199 | 51.713 | -5.3% | -9.9% | 0.2% |
| full | 2 | 0 | 8192 | 91.430 | 109.899 | 116.659 | -16.8% | -21.6% | 1.0% |
| full | 2 | 0 | 16384 | 181.367 | 219.285 | 232.439 | -17.3% | -22.0% | 2.5% |
| full | 4 | 0 | 2048 | 46.667 | 49.342 | 51.642 | -5.4% | -9.6% | 0.4% |
| full | 4 | 0 | 4096 | 91.578 | 109.782 | 115.907 | -16.6% | -21.0% | 0.8% |
| full | 4 | 0 | 8192 | 180.774 | 220.093 | 230.977 | -17.9% | -21.7% | 3.5% |
| full | 4 | 0 | 16384 | 361.819 | 436.983 | 454.919 | -17.2% | -20.5% | 0.5% |
| full | 8 | 0 | 2048 | 91.297 | 109.163 | 115.958 | -16.4% | -21.3% | 1.3% |
| full | 8 | 0 | 4096 | 180.935 | 220.452 | 230.289 | -17.9% | -21.4% | 1.8% |
| full | 8 | 0 | 8192 | 361.820 | 436.260 | 452.169 | -17.1% | -20.0% | 1.1% |
| full | 8 | 0 | 16384 | 724.339 | 892.881 | 927.153 | -18.9% | -21.9% | 0.7% |

## Separate sparse policy probes

The default retains the [per-expert QMV/QMM strategy](executor_moe.md).
The whole-batch alternative selects QMM for recorded routes `>=16` and recorded routes per expert `>=4`. It selects QMV otherwise.
Both variants keep fused gate/up/SiLU and the same QMM tile selection. These values are sparse-expert-only latency.
Every probe uses 10 same-process ABBA epochs. Each block uses 20 warmup and 100 measured submissions.
Pure short and pure long inputs improve. All five measured mixed distributions regress.
These controlled expert distributions do not establish how often each case occurs in serving.

| Input | Tokens | Routes in 1–3-row segments | Existing ms | Whole batch ms | Whole batch vs existing |
| --- | ---: | ---: | ---: | ---: | ---: |
| mixed32 | 32 | 37.5% | 0.862 | 0.928 | +7.8% |
| short64 | 64 | 100.0% | 1.932 | 1.642 | -15.1% |
| mixed127 | 127 | 37.8% | 2.379 | 2.863 | +20.3% |
| mixed128 | 128 | 37.5% | 2.205 | 2.806 | +27.2% |
| real128 | 128 | 0.0% | 1.480 | 1.408 | -4.8% |
| short-heavy128 | 128 | 68.0% | 2.591 | 2.948 | +13.8% |
| mixed256 | 256 | 18.8% | 2.831 | 3.429 | +21.1% |

Both policies pass the existing sparse CPU-reference/replay test.
Pure short and pure long outputs are bitwise equal. Mixed outputs have a relative L2 difference of 0.29%–0.56%.
Expert segments that retain their algorithm are bitwise equal. These descriptive differences are not new correctness tolerances.
The first `mixed256` attempt had 4 new swap-in pages. Its timings are excluded. A clean authorized retry supplies the displayed values.
In a separate original `2 requests x 64 tokens` experiment, removing two empty QMV commands saved 65.940 us (4.40%).
The paired full-replay delta measures net cost. Isolated QMV submission latency includes host overhead and must not be added to it.

## Commands and dataset provenance

The local snapshot was generated by this controller:

```sh
python3 /private/tmp/psi-dec-moe-20261014/component_matrix.py --mode run
```

The snapshot retains worker hashes, checkpoint metadata hashes, commands, versions, source identity, and counter baselines.
Local raw logs remain under `target/perf/moe-705aaf3c-20261014/`. The repository dataset does not require those logs to read its values.
Use the [component benchmark controller](executor_benchmarks.md#component-latency-matrix) for future PSI runs and resume.
A new comparison must record its own native peer commands and measurement identity. Do not append changed workers to this snapshot.
