# N-gram embedding and PLE

This document describes resident per-layer embedding (PLE).
The baseline follows the arithmetic and component boundaries in
[oMLX `64bf5d1`](https://github.com/jundot/omlx/blob/64bf5d1ba6336b45955c00f7fe0e46acd5c85095/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py#L3966-L4360).
The [Qwen4x model owner](executor_qwen4x.md) loads weights and composes these components.
Runtime core does not interpret PLE tensors or state.

The resident n-gram source and selection contract is in [N-gram embedding](executor_ngram_embedding.md).

## Owners

```text
inference-executor-core/src/
  ngram_embedding/core.rs       hash geometry and checkpoint parameter construction
  ngram_embedding/reference.rs  independent eager row-ID reference
  ple/core.rs                   residual-channel and convolution geometry
  ple/reference.rs              eager BF16 gate, normalization, and convolution reference

inference-backend-metal/src/components/
  ngram_embedding.rs            hash and H1/H4 or general sharded lookup recording
  metal/ngram_embedding.metal   GPU hashing and BF16/affine row decoding
  ple.rs                       grouped norm, gate, causal convolution, state commit
  metal/ple.metal               Metal implementation of these operations

inference-executor-metal/src/ple/
  backend.rs                   prepare/project/apply/commit recording
  batch_metadata.rs            token segments, source/destination slots, acceptance counts
  scratch.rs                   reusable preparation outputs and apply intermediates
```

The caller owns table storage, projection weights, source state versions, and destination state slots.
The model owner loads checkpoint tensors and composes PLE with GR and the transformer layers.
The `PLE` component owns kernels and immutable hash parameters. It creates no stream or file reader.

## One concrete forward

Use one request with two input tokens, `C=4`, `H=2560`, and embedding width `E=2560`.
`T` counts tokens. Activations use contiguous `[token,channel,hidden]` layout, and values use `[token,hidden]` layout.
The n-gram orders are two and three. Each order has eight hash heads.
There are 16 row IDs per token and 160 values per table row.
These hash heads are independent of QSA heads and residual channels.

```text
previous token IDs [2] + current token IDs [2]
       |
       +-- 2-gram, 8 heads --+
       +-- 3-gram, 8 heads --+--> IDs [2,16] U32
                                    |
                       resident shard lookup / dequant
                                    |
                         embeddings [2,16,160] BF16
                                    |
                          view E [2,2560] BF16
                                    |
                   +----------------+----------------+
                   |                                 |
             key affine                        value affine
             [10240,2560]                      [2560,2560]
                   |                                 |
           key [2,4,2560]                      V [2,2560]
                   |
             grouped RMSNorm
                   |
            K [2,4,2560] ----------------------+
                                              +--> per-channel dot / sqrt(H)
X [2,4,2560] -> grouped RMSNorm -> Q [2,4,2560] -+          |
                                                signed sqrt / sigmoid
                                                         |
                                                  gate [2,4,1] --+
                                                                +--> multiply -> Z [2,4,2560]
                              V [2,2560] -> view [2,1,2560] ------+

Z [2,4,2560] --+--> grouped RMSNorm -> normalized Z [2,4,2560] --+
              |                                               |
              |           source state [9,4,2560] -------------+
              |                                               |
              |                          causal depthwise conv, then SiLU
              |                                               |
              +-----------------------> add <-----------------+
                                         |
X [2,4,2560] ---------------------------> add -> X' [2,4,2560] BF16
```

PLE updates the complete residual before the next GR read, without calling GR write.

## Preparation and scheduling

| API | Reads -> writes |
| --- | --- |
| `PLE::prepare` | Token IDs/history, hash parameters, weights -> embeddings, values, normalized keys |
| `PLE::project` | Decoded embeddings -> same projected values/normalized keys |
| `PLE::apply` | Prepared outputs + ready residual + convolution state -> updated residual |
| `PLE::commit` | Accepted normalized-Z prefix + source history -> convolution state |

Preparation does not read residual or convolution state.

```text
batch token metadata
    |
    +--> prepare: hash -> lookup -> key/value projections -> key norm --+
    |                                                                 |
    `--> token embedding -> earlier transformer layers -> residual ----+
                                                                      |
                                                                    apply
                                                                      |
                                                               accepted counts
                                                                      |
                                                                    commit
```

The graph permits prepare to overlap earlier layers.
The model owner must select the execution schedule and establish the producer-to-consumer dependency.
A separate stream is one option. A barrier in one stream does not synchronize another stream.
The component does not claim measured overlap or create a second queue.
Source metadata and scratch must remain valid until their consumers complete.
Each concurrently prepared batch needs its own live preparation outputs.

## Metadata and state

`PLEMetadataBuffers` uploads flattened token IDs, request-local history, token segments, and state slots.
Each source state remains immutable through forward and commit.
Destination slots must be distinct and disjoint from all source slots.
`PLE::commit` combines the source history with only the accepted normalized-Z prefix.
Zero acceptance copies the source state unchanged.

`PLEScratch` retains preparation outputs until apply completes.
It retains normalized Z until commit completes.
Recorded total-token capacities can be smaller than the allocated metadata and scratch capacities.
Submission arguments carry active counts.
Projection topology and n-gram lookup topology belong in the owning replay key.
The backend owns shape selection.

The Qwen4x loader streams checkpoint shards into final resident groups.
The [n-gram guide](executor_ngram_embedding.md) defines global row IDs, hash construction, and H1/H4 selection.
SSD offload is not part of the wired model path.

## Verification

Backend tests compare fixed and random gate, convolution, and state-commit outputs with the eager CPU reference.
Executor tests add hash lookup, affine projections, request-local history, and accepted-prefix replay.
The Qwen4x first8 tests add model composition and lifecycle checks.

Run GPU tests serially:

```sh
cargo test --release -p inference-backend-metal --lib components::ple:: -- --test-threads=1
cargo test --release -p inference-executor-metal --lib ple:: -- --test-threads=1
```

The [Qwen4x validation script](executor_qwen4x.md#verification) runs the composed graph and executor PLE tests.
Performance optimization of PLE normalization, gating, and convolution remains [follow-up work](future_work.md#qwen4x-integration).
