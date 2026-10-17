# Sampling Executor

This document owns the current top-k/top-p sampling and sparse rejection-sampling contracts.
[`executor_qwen.md`](executor_qwen.md) owns Qwen stage order and Spec proposal ownership.
[`executor.md`](executor.md) owns generic executor composition.

## Source layout

```text
crates/inference-executor-core/src/sampling/
  config.rs          sampler validation and optional request seed
  domain.rs          independent Main, Spec, accept, and resample RNG domains
  reference.rs       CPU top-k/top-p and rejection correctness oracle
  dspark.rs          CPU reference for sequential Markov correction, confidence, and sampling
  rejection_sampling.rs
                     backend-neutral sparse rejection bounds, shape, request, and microbatch-view contracts
  request_state.rs   executor-owned request-slot seed lifecycle
  top_k_sampling.rs  backend-neutral sampling shape and request parameters

crates/inference-backend-metal/src/components/
  sampling/top_k.rs           generic Metal top-k sampling components
  sampling/top_k_test.rs      focused Top-K Metal parity and replay contracts
  sampling/failure.rs         sampling-owned GPU failure record and CPU diagnostics
  sampling/dflash2_selector.rs
                              DFlash2 edge scoring and sequential path walk
  sampling/rejection.rs       sparse rejection component
  sampling/spec_decode_input.rs
                              rejection-output to fixed-block Decode-input transform
  sampling/dspark_markov.rs   fused DSpark Markov, confidence, and Top-K Map component
  metal/sampling.metal
  metal/spec_decode_input.metal
  metal/dflash2_selector.metal
  metal/dspark_markov_sampling.metal

crates/inference-executor-metal/src/sampling/
  sampling_params.rs      SamplingParamsStore request-slot parameter owner
  top_k_sampling.rs       TopKSampling row inputs, scratch, and TopKSamplingOutputBuffers
  top_k_replay.rs         Sampling and DraftSampling replay components
  rejection_replay.rs     sparse rejection replay owner, batch preparation, and bindings
  dspark_markov.rs        sequential DSpark Markov, confidence, and sampling composition
  spec_probs.rs           SpecProbsStore sparse draft/target probability workspace

crates/inference-executor-metal/src/model/qwen/v3_x/dspark/
  sampling.rs             Qwen3x Markov checkpoint weights and generic backend adapter

crates/inference-executor-metal/src/model/qwen/v3_x/dflash2/
  output.rs               raw Top-K, selector checkpoint weights, path walk, and sparse draft storage

crates/inference-executor-metal/src/model/qwen/v3_x/
  spec_decode_input.rs    static and GPU-written fixed-block Decode-input owner
```

Runtime core transports sampler configuration and sampled decisions.
It does not own RNG state, sparse distributions, or replay geometry.
The executor resolves one root seed for each live request slot.
It keeps that seed stable until the slot resets.

## Normal sampling

Each request slot has one stable sampling-parameter row:

```text
sampling_params[req_slot] = {temperature, top_p, seed, top_k}
```

`SamplingParamsStore` owns this buffer. One `Rc<SamplingParamsStore>` is shared by Vanilla, MTP, DSpark, and DFlash2
inside one executor. The executor updates active request-slot rows when it prepares a batch.
The request-slot buffer capacity and each invocation's sampling-row capacity are independent. Qwen Main uses the
compact output-row capacity defined in [`executor_qwen.md`](executor_qwen.md#replay-ownership).
DSpark uses the request-row capacity. Each consumer validates its static capacity when it is initialized.
Backend sampling components validate static shapes, buffer ranges, and dispatch products when they create invocations.
Their `record` methods use these validated values.
`add_replay_arguments` must receive the same shape used to record the replay.
It checks changing active counts and thread-count conversion without repeating static shape validation.
The replay parameter table enforces the recorded domain at submission.

Each sampling invocation separately provides the compact sampling-row to `req_slot` mapping, the logical output-token
position for each row, and one `SamplingDomain`. Main can contain multiple sampling rows for one request. Those rows
reuse the same request-slot parameters and keep distinct positions. DSpark and DFlash2 reuse the same request-level
parameters but retain their different proposal algorithms.
The current default decode policy uses temperature 0.7, top-p 0.8, and top-k 20.
Greedy decoding uses the same contract with top-k 1 and temperature 0.

```text
logits [num_rows, vocab_size]
  -> TopKMap
       one sampling row and one 256-token vocabulary partition per thread block
       write partial candidates
  -> TopKReduce
       merge partial candidates for one sampling row
       apply temperature and top-p
       load params[req_slots[row]]
       draw from (seed, sample_positions[row], sampling_domain)
  -> sampled token IDs + probabilities
```

`top_k::MapCompute` owns a private `Registry` and `Selector`. The registry stores four complete execution variants:
F32 reduction, F32 bitonic, BF16 reduction, and BF16 bitonic. `Selector::select(...)` returns
`(VariantKey, &Variant)` from the logits dtype, top-k, and required output. The small-k path uses repeated maximum
reduction for sample-only top-k <= 32. Larger top-k and write-distribution generation use the bitonic Map path. The
selector performs the complete choice. The component does not have a separate planner or plan object.
Unused static threadblock storage can reduce occupancy.

Each active sampling row must contain at least one finite logit candidate.
A row with no finite candidates must fail. An invalid probability normalization or rejection mass must also fail.
The backend must not substitute token `0` or a greedy distribution for these failures.
A valid distribution can still select token `0`.

The sampling component owns a retained failure record and implements `SubmissionCheck`.
The stream resets registered checks before submission and calls them after successful GPU completion.
The stream does not allocate sampling records or interpret sampling error codes.
The shader captures the first failure for each command, including the row, available request slot, sample position,
RNG domain, distribution index, sampling parameters, and failed numerical value with its F32 bits.
Sparse rejection also captures the target and draft distribution ranges.
Unavailable request slots, sample positions, and distribution indices are marked as unavailable.
Other fields are reported only when the operation owns them.
The row is local to that command, not a request ID or an absolute token position.
The CPU reports the operation and recorded shape with this snapshot. It does not read reused input or scratch buffers
to reconstruct the failed invocation.
`ReplaySubmission::wait()` panics before the executor reads samples or returns a batch response.
This check is active in release builds. It does not require an environment variable.
Commands already encoded in the same submission can finish before the CPU detects the failure.
The executor must not consume any outputs from that failed submission.

The default sampling contract still filters individual non-finite logits when finite candidates remain.
To reject any NaN or infinity at Qwen3.5 computation boundaries, enable the
[numerical checkpoints](executor_qwen.md#numerical-checkpoints).
These checkpoints identify an upstream component before the sampler can discard its non-finite logits.

Sampling returns only token IDs and probabilities.
Partial candidates and reduced rows are private scratch, not model-level API state.
`top_k::ReduceCompute::invoke_merge(...)` exposes raw merged token IDs and logits to model-specific consumers such as
the DFlash2 candidate selector.
It does not apply temperature, top-p, or sampling.
The current input contract does not include repetition, frequency, or presence penalties.
`TopKSamplingOutputBuffers` owns the concrete sampled token and probability buffers.
`OutputBuffers` denotes GPU buffers, not a model lifecycle state machine.

## Sparse distributions and rejection

Qwen MTP and DSpark proposals use the same post-temperature/top-k/top-p distribution family as Main verification.
The production path stores sparse token and probability rows.
It does not scatter them into dense full-vocabulary buffers.

```text
proposal logits
  -> sample + sparse draft distribution

target logits
  -> sparse target distributions

target distributions + draft distributions + flat draft tokens
  -> sparse rejection
       process drafts sequentially within each request
       accept with SamplingDomain::Accept
       on rejection sample max(target - draft, 0) with SamplingDomain::Resample
       if all drafts pass, sample the final target continuation
  -> accepted draft prefix + one fallback/continuation token
```

The target table contains one distribution for each draft token.
It also contains one final continuation distribution for each request.
The draft table contains one distribution for each draft token.
`cu_target_distributions` and `cu_draft_distributions` select the ragged per-request ranges.
Requests are independent.
Drafts within one request are ordered because the first rejection ends that request's acceptance walk.

The sparse rejection component has one fixed `rejection::Compute`. It has no registry or selector. One thread block
owns one request. The threads cooperate
on the sparse-distribution operations for that request. The block processes the draft sequence in order. The current
compile-time constants require 256 threads. The recorded grid contains one thread block for each request capacity
slot. Replay arguments limit execution to the active request count.

`SparseRejectionSamplingOutput` is the borrowed GPU-output boundary for accepted counts and sampled anchor token IDs.
Qwen3 and Qwen3.5 DSpark, and Qwen3.5 DFlash2, pass this view directly to `SpecDecodeInput`.
They do not read rejection output before they record Spec Decode.

Spec Prefill borrows `num_tokens`, request slots, and flat token indices directly from the current Main GQA metadata.
That metadata already contains the complete active Main-row prefix and remains valid through the combined recording.
`SpecDecodeInput::prepare` fills reusable GPU resources for the corresponding Spec Decode recording.

`SpecDecodeInput` owns anchor indices and GPU-written first Spec sample positions for one Spec Decode lifecycle.
It borrows the existing BiDiBlockGQA metadata buffers.
It does not allocate a second GQA metadata owner or repack DSpark and DFlash2 sampling layouts.
`record_decode_prepare` and `record_spec_decode` bind the same physical buffers.
Active row order is semantic. Request row `r` must stay aligned across rejection outputs, anchor indices, token buffers,
BiDiBlockGQA metadata, and sampling destinations. `num_total_requests` is the cached replay bucket and grid capacity.
Active rows are its prefix, and inactive rows return before resource access. The owner fills inactive metadata rows with
unused valid request slots only to materialize a structurally valid fixed-capacity BiDiBlockGQA layout.

The `rejection_to_spec_decode_input` kernel uses one 32-thread threadgroup for each request-capacity slot.
This is one Apple SIMDgroup on current targets, but correctness does not depend on SIMDgroup execution.
Lane 0 writes request scalars.
All threads stride the fixed Spec block and each fixed history-task quota.
Replay arguments limit work to `num_active_requests`.
Acceptance changes positions and range endpoints, but it does not change the grid or task count.
The transform rewrites every active history task and preserves the final block-partial slot for each Q-token range.
A history TaskTemplate is `{q_token_range_index, kv_token_begin, kv_token_end}`. It assigns one GQA Map threadblock the
half-open persistent-history range `[kv_token_begin, kv_token_end)` for one Q-token range. One Map threadblock can loop
over multiple K/V iterations in that range.
The complete replay sequence is CPU-recorded before submission. Spec Decode prepare overwrites active query, range, and
TaskTemplate coordinates before Spec Decode reads them. It cannot change the later replay key or grid, the Q-range and
cumulative-partial structure, or the number of Map TaskTemplates.
CPU preparation uses maximum acceptance to retain the metadata allocator's worst-case recorded split count and
parallelism. Lower acceptance maps excess splits to canonical empty ranges. A smaller seed, including all-reject, can
remain correct if each Q range retains at least one history task and its block partial. However, that seed can
under-partition a later longer history range and change performance.
At construction, the owner validates the model maximum anchor, the K/V iteration width, the Spec-block width, and
the fixed TaskTemplate capacity with widened host arithmetic. This validation proves that all transform position,
iteration-partition, and Task-quota intermediates fit `uint`. The Metal kernel then uses direct `uint` arithmetic.

The transform records one Spec Decode prepare command. It writes the first Spec sample position as
`accepted_anchor_index + 1`. DSpark step `i` adds `i` when it samples sequentially. DFlash2 adds its step index inside
the path walk. No separate sampling-position update command is necessary.

`SpecProbsStore` owns `draft_token_ids`, `draft_probs`, `target_token_ids`, and `target_probs`.
`max_k` is the maximum sparse Top-K row width, not the vocabulary size.
Its draft capacity is `max_requests * max_spec_tokens`.
Its target capacity is the compact Main sampling capacity defined in
[`executor_qwen.md`](executor_qwen.md#replay-ownership). The executor `max_tokens` value remains its upper limit.
Debug builds also retain `expected_draft_token_ids` for lifecycle validation.
Release builds do not allocate, reset, or compare this CPU-only metadata.

Draft distributions cross a batch boundary.
Their row identity is `req_slot * max_spec_tokens + proposal_position`.
Main verification distributions exist only in the current submission.
They use compact active-row indices from zero.
`cu_target_distributions` uses this compact row domain.
The executor `max_tokens` value is the target-distribution capacity for Qwen3 and Qwen3.5.
The request-slot by speculative-position product is only the draft-distribution capacity.

## Replay ownership

Sampling and rejection use `u32` capacity replay keys.
`Sampling`, `DraftSampling`, `RejectionSampler`, and `DSparkMarkovSampling` each own their applicable
`ReplayBucketPolicy` values. Their prepare APIs convert active counts into recorded capacities. Qwen model callers do
not select replay buckets.
Submission-scoped `ReplayArguments` contain exact active thread counts.
Every padded kernel returns inactive lanes before it reads input, changes RNG state, or writes output.
An absent zero-capacity work domain can use an immediate zero.
Every nonempty active work domain remains a replay parameter, including a total capacity of `1`.

Active top-k remains in the replay shape because it changes candidate and scratch geometry.
Temperature, top-p, seed, and active top-k are stable request-slot data for a live batch. Logical positions, row-to-slot
mappings, and the RNG domain are invocation data. They never enter replay keys.

Writing sparse-rejection runtime parameters arms exactly one replay-argument preparation.
That preparation consumes the matching active request count and clears the armed state.
Sampling parameters use stable request-slot identity instead. Each sampling invocation supplies its current compact row
mapping and positions.

The Qwen executor owns these distinct graph and cache stages:

- `Replay<Sampling>` handles ordinary Main output.
- `Replay<DraftSampling>` handles MTP draft sampling and sparse draft-distribution storage.
- `Replay<Qwen3xDSparkSampling>` handles DSpark Markov correction, confidence, sampling, and sparse draft storage.
- `Replay<Qwen3xDFlash2Output>` handles DFlash2 raw Top-K, candidate selection, sampling, and sparse draft storage.
- `Replay<RejectionSampling>` handles target write-distribution generation and sparse rejection.

Main and MTP share one `Rc<TopKSampling>` implementation.
`Sampling::prepare_shape(...)` selects the Main sampling capacity.
`DraftSampling::prepare_shape(...)` selects the MTP request capacity.
`RejectionSampler::prepare_replay_shape(...)` selects request, draft-distribution, and target-distribution capacities.
It validates public prepared counts before it maps them to the `u32` replay domain.
`RejectionSampling` validates the target-sampling and rejection shapes before the replay-cache lookup.
`DSparkMarkovSampling` owns the compact request-slot mapping, partial candidates, and per-step outputs.
It borrows the shared `SamplingParamsStore` through `Rc`.
It accepts borrowed weights at record time.
`Qwen3xDSparkMarkov` owns Qwen checkpoint buffers and delegates execution to this backend owner.
It loads one bounded `TensorMap` for W1, W2, and the required Markov-conditioned confidence head.
It removes all tensors and requires the map to be empty before initialization completes.
It reuses the generic sample-and-write-distribution reducer.
The stages retain separate replay keys and programs.

Qwen3.5 Main and MTP body replays use separate total-token domains.
Qwen3, DSpark, and DFlash2 currently set `num_total_tokens == num_active_tokens` in their fixed-capacity stages.
All body components use the same active/total API.
Sampling and rejection keep their own replay domains, keys, active counts, and bucket policies.
MTP draft sampling is a distinct replay after MTP GatherUnembed.
For K MTP steps, each pass writes one request-local draft-distribution row at `step_index`.
The step index is runtime metadata and does not enter the sampling replay key.
DSpark Markov sampling is a distinct replay after `Qwen3xDSparkGatherUnembed`.
Sparse Main distributions and rejection form one Main-stage replay.

One DSpark Markov step records two commands:

```text
dspark_markov::MapCompute
  input_token_ids[i]
    i = 0: sampled_token / anchor_token
    i > 0: spec_tokens[i - 1]
  -> affine W1 row
  -> affine W2 projection
  -> add one base-logit row
  -> 64-token vocabulary-partition Top-K
  -> confidence projection and sigmoid
  -> spec_confidence[i]

top_k::ReduceCompute::invoke_sample_and_write_distribution_with_layout
  -> global Top-K merge
  -> top-p sampling
  -> spec_token[i], spec_prob[i], and sparse draft distribution
```

The sampled token from one reducer is the Markov input for the next map.
The replay places a barrier at this dependency.
A seven-token block records 14 commands in one submission.
It does not materialize full-vocabulary Markov bias or corrected-logit buffers.

`SampledTokens::Decode` carries `spec_tokens`, `spec_probs`, and `spec_confidences`.
These vectors always have the same length.
MTP uses `1.0` for each speculative confidence because MTP has no confidence head.
Qwen3x DSpark requires and evaluates its confidence head.
The Qwen3 and Qwen3.5 response adapters preserve DSpark confidence values across the executor/runtime boundary.
Runtime core uses cumulative confidence to allocate verification prefixes when the batch budget cannot contain every proposal row.
The [token budget allocator](token_budget_allocator.md) owns this policy.
Proposal generation remains fixed-width. The runtime does not apply an absolute confidence threshold.

The current fused map preserves the earlier BF16 storage boundaries.
It dequantizes W1 to F32 and stores the latent row as BF16.
It accumulates W2 in F32.
It rounds the correction and corrected logit to BF16 before the vocabulary-partition Top-K operation.
Sampling probabilities use F32.

### Top-K execution hierarchy

Top-K sampling has a true Map/Reduce dependency:

```text
top_k::MapCompute
    -> one MapThreadBlockTask
       = one sampling row x one vocabulary partition
    -> top_k::PartialCandidateLayout
       = vocabulary partition size
    -> partial token IDs + partial logits

top_k::ReduceCompute
    -> one ReduceThreadBlockTask
       = one sampling row x all partial-candidate partitions
    -> sampled token/probability and optional sparse distribution
```

The standard Map compile-time constants have `thread_block.max_vocab_tokens = 256` and
`thread_block.required_threads = 256`. The DSpark fused Map constants have `thread_block.max_vocab_tokens = 64` and
`thread_block.required_threads = 128`. Both producers return an explicit `top_k::PartialCandidateLayout`. The Reduce
invocation consumes that layout. The executor does not pass an untyped `vocab_tile_size` between the components.

`top_k::Shape::top_k` defines the number of candidates in each partial row. The layout does not duplicate this
dynamic shape value.

The Reduce constants require 256 threads. The Map and Reduce replay arguments retain their current capacity and
active-thread semantics. The backend exposes component-scoped APIs through `sampling::top_k`,
`sampling::dflash2_selector`, `sampling::dspark_markov`, and `sampling::rejection`. It does not expose compatibility aliases from
`components` or `sampling`.

### DFlash2 candidate selection

DFlash2 uses the raw Top-K merge to select `selector_top_k` unary candidates for each proposal position.
The selector then uses three fixed kernels:

```text
anchor token IDs + unary candidate IDs
  -> predecessor-token IDs
  -> quantized predecessor/successor embedding owners

projected hidden + predecessor embeddings + successor embeddings + unary logits
  -> edge scores [request, step, predecessor, successor]

edge scores + request-slot sampling parameters + first sample positions
  -> sequential request-local path walk
  -> proposal token/probability rows
  -> sparse draft distributions
```

The first step uses the anchor token as every predecessor.
Each later step uses the prior step's unary candidates as predecessors.
The walk reads only the score row that the previous step selects.
The score producer must write finite F32 values.
The walk does not replace invalid scores with a fallback distribution.
It applies temperature to the fixed candidate set.
It does not apply top-p because the DFlash2 reference selector does not apply top-p after candidate construction.
Each sparse distribution contains the exact probabilities that the walk uses.

### DSpark Markov numerical contract

Decision: Retain the current BF16 materialization boundaries.

The Qwen3 implementations in
[vLLM at `bb3b61f2fd2333ab165ebaba13f133db4210b9f2`](https://github.com/vllm-project/vllm/blob/bb3b61f2fd2333ab165ebaba13f133db4210b9f2/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py)
and
[SGLang at `85618cc798ce9b5fdbfdd5c535576515d498acc2`](https://github.com/sgl-project/sglang/blob/85618cc798ce9b5fdbfdd5c535576515d498acc2/python/sglang/srt/models/dspark.py)
add the Markov correction in the model-logit dtype.
They do not explicitly promote the Qwen3 corrected-logit tensor to F32.
[SGLang sampling](https://github.com/sgl-project/sglang/blob/85618cc798ce9b5fdbfdd5c535576515d498acc2/python/sglang/kernels/ops/speculative/dspark/dspark_draft_model.py)
promotes the corrected logits when it computes sampling probabilities.

The Metal map uses F32 where reduction requires it:

```text
BF16 W1 latent in shared memory
  -> F32 W2 partials and SIMDgroup reduction in thread-local values
  -> BF16 correction
  -> BF16 corrected logit
  -> F32 tile Top-K and sampling probabilities
```

For the official rank-256 checkpoint, the map uses 1024 bytes of static shared memory.
The BF16 latent uses 512 bytes.
The 64 F32 tile logits and 64 I32 token IDs use the other 512 bytes.
An F32 corrected-logit candidate does not require more shared memory.
It keeps the correction and corrected logit in thread-local scalar values.
The Metal compiler can keep these values in registers or spill them.
Changing the W1 latent to F32 would increase the static shared-memory requirement to 1536 bytes, but that is a
different numerical contract.

The F32 corrected-logit candidate used this sequence:

```text
BF16 W1 latent
  -> F32 W2 accumulation
  -> F32 base-logit add
  -> F32 tile Top-K and sampling probabilities
```

The A/B investigation used base commit `2804bdece3c03235e9b2b18f9655a933ed7a3220` on macOS 27.0 and an Apple M3 Max
with a 40-core GPU and 48 GB of memory.
The candidate changed only the corrected-logit rounding contract and its CPU reference.
The worktree also contained only the benchmark output controls from this change.
Focused CPU/GPU parity passed for both contracts.

The real-weight Markov benchmark used
`dspark_qwen3_14b_block7-affine`, one request, `temperature=0.7`, `top_k=20`, `top_p=0.8`, and `seed=42`.
It used the production Markov weights and deterministic zero base logits to isolate Markov correction and sampling.

```text
cargo bench --bench qwen3_dspark_sampling -- \
  --dspark-model-dir /Users/wenquanxing/Workspace/models/dspark_qwen3_14b_block7-affine \
  --num-requests 1 --temperature 0.7 --top-k 20 --top-p 0.8 --seed 42 \
  --warmup-iters 10 --iters 50 --runs 5
```

The current BF16 path measured 4.158 ms for the complete seven-step replay.
The F32 candidate measured 4.162 ms.
The candidate did not produce a measurable performance gain.
Both contracts sampled `[5310, 5390, 979, 14550, 448, 5091, 369]` in this isolated case.
The BF16 write-distribution fingerprint was `31642210e096a9d7`.
The F32 fingerprint was `9dacaff10a8cc01d`.
The proposal probability bits also differed.

The fused confidence check used base commit `a36024dc` with only the confidence change present in the worktree.
It used the same machine, Markov weights, sampler settings, and benchmark command.
The converted checkpoint also contained the official BF16 confidence projection and bias.
The complete seven-step replay measured 4.061 ms.
The sampled tokens and BF16 write-distribution fingerprint matched the 4.158 ms path.
The confidence output contained seven finite, nonconstant values.

The deterministic service comparison used Qwen3-14B-4bit, the same DSpark checkpoint, `temperature=0`, `top_k=1`,
`top_p=1`, and `seed=42`.

```text
target/release/qwen3 \
  --grpc-listen-addr 127.0.0.1:50151 \
  --http-listen-addr 127.0.0.1:8011 \
  --hf-model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --hf-spec-model-dir /Users/wenquanxing/Workspace/models/dspark_qwen3_14b_block7-affine \
  --spec-type dspark \
  --num-cache-pages 4096 --max-requests 4 --max-tokens 128 \
  --max-tokens-per-request 64 --logging info

target/release/decode \
  --server-url http://127.0.0.1:50151 \
  --hf-model-dir /Users/wenquanxing/Workspace/models/Qwen3-14B-4bit \
  --prompt-str 'Explain in concise technical terms why the sky appears blue during the day.' \
  --disable-thinking --max-sampled-tokens 256 \
  --temperature 0 --top-k 1 --top-p 1 --seed 42 --show-stats
```

Three runs for each contract produced the same final 98-token text.
The current BF16 contract used 34 verification batches, generated 238 proposals, and accepted 65 proposals.
Its acceptance rate was 27.31%.
Its three decode rates were 42.095, 42.253, and 42.852 tok/s.
The F32 candidate used 33 verification batches, generated 231 proposals, and accepted 66 proposals.
Its acceptance rate was 28.57%.
Its three decode rates were 43.021, 43.320, and 43.169 tok/s.
The throughput values are not a corrected-logit kernel comparison because the contracts used different verification
trajectories.

The changed distributions and replay trajectory make F32 a model numerical-contract change.
One deterministic prompt does not justify that change.
The production path must keep the BF16 contract unless broader checkpoint evidence establishes a different contract.

## Correctness and benchmarks

CPU references define sampling, rejection, and sequential DSpark Markov math.
Each Top-K API records one replay with a total row capacity of `4`.
The Merge, Sample, WriteDistribution, and SampleAndWriteDistribution tests replay
`1, 4, 3, 2` active rows and compare each active logical output with a CPU reference.
The Merge and Sample tests cover F32 and BF16 logits with reduction and bitonic Map topologies.
The distribution tests cover F32 and BF16 bitonic Map topologies, mapped distribution slots, mixed per-row runtime
parameters, deterministic sample positions, and Target and Draft domains.
All Top-K tests use a nonzero logits row offset.
One sparse rejection replay uses the same active-request sequence. It compares active accepted counts, sampled tokens,
sampled probabilities, and accepted-token prefixes with the CPU rejection reference. The fixture covers rejection,
all-accept, zero-draft, and non-contiguous draft-distribution mapping.
The DSpark Markov parity test uses the production `Replay<T>` cache with exact request-stride keys and padded sampling
capacities. It runs the same non-monotonic active sequence with non-contiguous request slots. It compares proposal
tokens, probabilities, confidences, and sparse distributions with the sequential CPU reference.
These tests do not use inactive output tails as an oracle.
GPU tests run serially under the repository Metal reservation/lock rules.
The `qwen3_dspark_sampling` benchmark prints proposal token IDs, exact proposal probability bits, and a stable
fingerprint of the complete sparse draft distribution.
Use `--temperature`, `--top-k`, `--top-p`, and `--seed` to reproduce a sampling contract.

Synthetic backend modes:

```text
cargo bench --bench rejection_sampling -- \
  --mode top-k-sample --rows 1 --num-reqs 1 --spec-tokens 1 \
  --iters 1 --warmup-iters 0 --runs 1

cargo bench --bench rejection_sampling -- \
  --mode dspark-markov-top-k-map --rows 1 --top-k 20 --vocab 151936 \
  --markov-rank 256 --markov-w1-group-size 64 --markov-w1-bits 4 \
  --markov-w2-group-size 64 --markov-w2-bits 8 \
  --iters 1 --warmup-iters 0 --runs 1
```

Supported modes are `top-k-sample`, `top-k-write-distribution`, `top-k-sample-and-write-distribution`,
`rejection-sparse`, and `dspark-markov-top-k-map`.
The model-executor targets are `qwen35_main_sampling` and `qwen3_dspark_sampling`.
Qwen3.5 DSpark reuses the Qwen3x Markov component benchmark and adds end-to-end service validation.
[`executor_benchmarks.md`](executor_benchmarks.md) defines shared measurement and provenance rules.
