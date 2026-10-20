# Model Primitive Executor Components

This document describes the current embedding, unembedding, row-gather, normalization, residual, and fused RoPE
components. It defines their component-specific GPU tasks and selection owners.

## Source layout

| Root under `crates/` | Owners |
| --- | --- |
| `inference-executor-metal/src/model/` | `embedding.rs`, `unembedding.rs`, `gather.rs`, `rms_norm.rs`, `residual_add.rs` |
| `inference-backend-metal/src/components/` | `embedding.rs`, `rms_norm.rs`, `residual_add.rs`, `residual_add_rms_norm.rs`, `rms_norm_rope.rs` |
| `inference-backend-metal/src/operators/` | `affine_quantized.rs`, `bf16_concat_rows.rs`, `row_gather.rs`, `softmax.rs` |

Use [GPU execution vocabulary](gpu_execution.md).

## Embedding

`Embed` owns the model-level quantized embedding weights and one `embedding::Compute`. It records an exact token count
or a fixed replay capacity with a submission-time active-token count.

The kernel computes this mapping:

```text
token_ids[token]
    -> quantized vocabulary row
    -> per-group scale and bias
    -> output_hidden[token, hidden]
```

The current private compile-time constants are:

```text
embedding::KernelConstants
├── scale_bias_dtype
├── output_dtype
└── thread_block
    └── required_threads = 256
```

One non-persistent thread block processes a bounded flat range of `(token, hidden)` output coordinates. One thread
processes one coordinate at a time. The kernel derives the token and hidden indices from the flat coordinate.

Embedding has one current kernel implementation for each supported scale/bias dtype. Dtype selection occurs at
initialization. The runtime token count changes only the grid. Embedding does not need a registry or selector.

## Unembedding

`Unembed` owns one adaptive `affine_quantized::Matmul`. It maps hidden rows to vocabulary-logit rows:

```text
hidden[num_rows, hidden_dim]
    -> affine quantized matrix multiplication
    -> logits[num_rows, vocab_size]
```

`affine_quantized::Matmul` owns a private `Registry` and `Selector`. The registry stores the legal QMV and QMM execution
variants. `Selector::select(...)` returns `(affine_quantized::KernelKind, &affine_quantized::Kernel)` from the
validated matrix configuration and row count. The same selection owns kernel tile geometry and topology boundaries.
`Unembed` supplies model geometry, weights, buffers, and row counts. It must not select a second kernel.

Same-dtype QMV BN8/BK32 with the identity epilogue calls the MLX `qmv_fast_impl` or `qmv_impl` helper.
Kernel construction requires the packed row byte numerator and packed weight byte indices to fit `i32`.
Invocation construction requires input row bases, including SIMD lane offsets, and output element indices to fit `u32`.
These checks run when the kernel or a replay program is created. Cached replay submissions do not repeat them.
The backend panics before GPU submission when a shape exceeds these domains.
These limits apply to relative offsets inside the kernel. Buffer range checks separately validate each bound byte range.
Affine invocation constructors validate these ranges, including byte offsets, before command recording.
Each offset must align to its operand dtype. Packed weights use `u32` alignment.
MLX QMV kernel constructors also reject output widths from five through seven.
The MLX small-output branch exceeds its four-entry result array for these widths.
This limit applies to same-dtype identity BN8/BK32 QMV and gather QMV. Other kernel families keep their own domains.

The [shared affine policy](executor_dense_mlp.md#backend-selection) owns small-row sliced-K, QMV, Steel QMM, and topology boundaries.
`Unembed` must preserve those boundaries in replay buckets instead of duplicating thresholds.
Embedding remains a row lookup with independent initialization-time dtype specialization.

## Row gather

`Gather` owns one `row_gather::Kernel`. It copies indexed input rows to a dense output:

```text
input[row_indices[output_row], column]
    -> output[output_row, column]
```

The private compile-time constants contain the dtype and `thread_block.required_threads = 256`.
One non-persistent thread block processes a bounded flat range of `(output row, column)` coordinates.
Fixed and parameterized active-row values use the same API and kernel.
The total row count changes the grid. The active-row value controls the guard.

Row gather has one current kernel for each supported dtype. It does not need a registry or selector.

Row gather, scatter, and route require separate output buffers.
Invocation construction rejects output aliases with source and metadata buffers.
`repeat_rows` requires a separate output buffer when `num_repeats > 1`.
With `num_repeats == 1`, input and output may share one buffer.

## RMSNorm

The standalone RMSNorm kernel maps one thread block to one token row:

```text
one ThreadBlock
    -> one token row
    -> reduce sum of squares
    -> apply reciprocal RMS and weight
    -> one output row
```

The dtype selects the F32 or BF16-vectorized kernel at initialization. The current kernel requires 1024 threads. The
runtime token count determines the grid dimensions. RMSNorm does not use a registry or selector.
F32 kernel initialization bounds the padded hidden dimension by the shader `u32` loop domain.

## Residual add

The plain residual-add kernel processes a bounded flat range of output values. Its row-prefix replay form preserves
complete row boundaries for active-count guards. The capture form also copies selected output columns to a separate
buffer.

The private compile-time constants use `KernelConstants.thread_block.required_threads = 256`. Exact, row-prefix, and
capture recording use the same requirement.

Dtypes select one static kernel at initialization. The runtime shape changes only the grid and active-prefix guard.
Residual add does not use a registry or selector.

## Residual-add RMSNorm

The fused component maps one thread block to one token row:

```text
lhs row + rhs row
    -> residual_output row
    -> RMSNorm
    -> norm_output row
```

The optional capture form also writes selected residual columns. The backend selects the scalar or BF16-vectorized
kernel from the validated dtype and hidden geometry at initialization. This is a static kernel-kind choice. It is not
a dynamic execution plan.

Replay fusion requires the RMSNorm weight dtype to match its IO dtype.
BF16 IO with F32 weights uses separate residual-add and RMSNorm commands.
Scalar kernel initialization bounds the padded hidden dimension by the shader `u32` loop domain.

## RMSNorm/RoPE

The fused RMSNorm/RoPE component maps one thread block to one `(flat Q token, Q head)` row. It normalizes one head row,
reads the request position from `flat_token_indices`, applies RoPE to `rope_dim`, and writes the complete head row.

The private compile-time constants contain the attention-head geometry, RoPE geometry, epsilon, scaling constants, dtype,
and fixed thread-block requirement. The runtime token count changes only the grid and the active-token guard. The
component has one current kernel for one model configuration and does not need a registry or selector.
Kernel initialization bounds the padded head dimension by the shader `u32` loop domain.

## Replay and ownership

Model owners retain immutable weights. Replay programs retain backend pipelines and buffers according to the common
replay contract. Exact invocations record exact token or row counts. Bucketed invocations record capacity and bind one
active-count parameter.

`EmbedConfig` and `UnembedConfig` delegate quantization geometry validation to their backend configurations at initialization.
Checkpoint loaders validate tensor byte lengths before they construct Metal buffers.
`Unembed::load_tied_weights` validates shared embedding buffers against the unembedding configuration when it attaches them.

The runtime core does not select primitive kernels. The Metal backend owns dtype specialization, affine selection, and
execution configuration. Model code owns semantic composition and buffer roles.
