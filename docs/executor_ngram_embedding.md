# N-gram embedding

This component maps request-local token n-grams to learned embedding rows.
It owns token-only hashing and resident BF16 or affine-quantized lookup.
It does not own scheduling, checkpoint loading, SSD reads, PLE projection, or layer wiring.

## Owners and contracts

```text
inference-executor-core/src/ngram_embedding/
    core.rs       geometry and immutable checkpoint hash parameters
    host.rs       CPU row IDs for callers that need host-side addresses
    reference.rs  independent eager hash reference

inference-backend-metal/src/components/
    ngram_embedding.rs                  validation, selection, replay recording
    metal/ngram_embedding.metal         GPU hash and general sharded lookup
    metal/ngram_embedding_resident.metal  H1/H4 resident lookup
```

The caller owns the table buffers, metadata, output, stream, and replay cache.
`HashParameters::new` validates geometry and constructs per-layer multipliers, prime vocabulary sizes, and row offsets.
`Compute::new` validates kernel geometry and initializes its immutable legal kernels.
The hash and lookup entry points validate record-time capacities and buffer ranges.
Lookup also validates the contiguous shard partition and each storage layout.
Submission does not repeat those checks.

The caller must supply valid token IDs, source history slots, request boundaries, and hash parameters.
Lookup IDs must address the supplied table partition. The kernel does not repair invalid IDs or metadata.
The table loader must keep global row IDs stable when it combines checkpoint shards into resident groups.
It can fill each final group directly from checkpoint chunks. Lookup does not copy or repack the table.

## Hash and lookup

Use two requests with input lengths three and two, n-gram orders two and three, and eight hash heads per order.
The row dimension is 160. Output uses contiguous `[token,hash head,dimension]` layout.

```text
Request 0: history [h00,h01], input [a,b,c]
Request 1: history [h10,h11], input [d,e]

GMEM token IDs [5] = [a,b,c,d,e]
GMEM history [2,2]
GMEM Token [5] = [(0,0),(0,0),(0,0),(3,1),(3,1)]
                 batch_start,history_slot
       |
hash grid [ceil(5*16/256),1,1], 256 threads/threadblock
    thread gid owns token=gid/16, head=gid%16
    head [0,8)  reads a 2-gram from that request
    head [8,16) reads a 3-gram from that request
    token history -> register XOR -> signed modulo -> head row offset
       |
GMEM row IDs [5,16] U32
       |
resident table rows -> register decode -> GMEM E [5,16,160] BF16
                                            view [5,2560]
```

For hash head `h`, its n-gram order is `floor(h / heads_per_ngram) + 2`.
The hash XORs token IDs multiplied by the corresponding `u64` multipliers, with wrapping multiplication.
An earlier EOS replaces all still-older IDs with EOS. The current EOS retains its preceding context.
The signed modulo result is nonnegative before adding the head's row offset.
The lookup addresses a learned row. Hashing does not compute or train that row's values.

BF16 tables apply the table scale after loading each value.
Affine tables use BF16 scales and biases, with 2/3/4/6/8-bit codes and group sizes 32/64/128.
Decode rounds the product to BF16, adds bias and rounds to BF16, then applies table scale and rounds again.
This order preserves the checkpoint arithmetic. It does not fuse product and bias into one FMA.

## Runtime selection and replay

`Selector::select` returns one `KernelKind` before recording:

| Validated table and geometry | Recorded token capacity | Kind |
| --- | --- | --- |
| 16 heads, dimension 160, Q4/group32, at most eight resident groups | `num_total_tokens <= 256` | `ResidentH1` |
| Same resident layout | `num_total_tokens > 256` | `ResidentH4` |
| Other supported geometry, format, or more groups | Any positive capacity | `Sharded` |

The resident kernel binds all groups in one dispatch. Each ID selects its own group inside the kernel.
Unused group bindings alias the final group. Repeated terminal row starts make those bindings unreachable.
The general path retains one dispatch per shard, with disjoint output writers.

Use the same input at capacity 257 to compare the resident variants:

```text
Same IDs [257,16] U32, same E [257,16,160] BF16
Eight groups: W_g [rows_g,20] U32, S_g/B_g [rows_g,5] BF16

H1: grid [257,16,1], 4112 threadblocks, 160 threads each
    threadblock (17,9): token 17, head 9
    thread d: decode row IDs[17,9] -> register -> E[17,9,d]

H4: grid [257,4,1], 1028 threadblocks, 160 threads each
    threadblock (17,2): token 17, heads [8,12)
    thread d:
        for h in [8,12):
            register id <- GMEM IDs[17,h]
            group g <- constant row starts
            register code/scale/bias <- GMEM W_g/S_g/B_g[row,d]
            GMEM E[17,h,d] <- decoded register value

Both: no shared memory or reduction. No table data is reused across heads.
The selector records H4 for this capacity. Submission can reduce the active count without changing that choice.
```

`lookup_topology` exposes the selected kind. `lookup_topology_boundaries` returns `[257]` for eligible resident tables and `[]` otherwise.
The replay key must contain total capacity, selected topology, table layout, and other record-time binding facts.
`num_active_tokens` is a submission parameter. Inactive token threadblocks return before reading IDs or writing output.
A capacity of 512 with one active token keeps H4. It does not change the recorded program to H1.

## Verification and benchmark

The core fixture checks hash parameters and row IDs against independent integer results.
Fixed and random Metal tests check request-local history, EOS behavior, supported storage formats, and uneven group boundaries against CPU references.
The cached replay test checks the 256/257 boundary, cache hits, and different active counts.

```sh
cargo test -p inference-executor-core ngram_embedding
cargo test -p inference-backend-metal ngram_embedding -- --test-threads=1
cargo bench -p inference-backend-metal --bench ngram_embedding
```

The benchmark uses production hashing, lookup, and selection with eight resident groups.
It uses a reduced-capacity synthetic Q4 table with vocabulary base 65536 per head, approximately 100 MiB in total.
It records 32 invocations per replay for `lookup` and `hash_lookup`.
This replay-amortized microbenchmark does not equal single-submission latency or end-to-end model throughput.

## Selection evidence

The [result snapshot](ngram_embedding_performance.json) records the earlier prototype measurements, provenance, paired ratios, and held-out results.
The measurement source commit was `1f32600f1002ce35ca48eb5d0c49732912c151ed`, with dirty sources recorded by the benchmark.
The device was Apple M3 Max with 48 GiB on macOS 27.0.1.
Peers were oMLX `64bf5d1ba6336b45955c00f7fe0e46acd5c85095` and TensorFold `cb2ebf0540f42604e2759b2ddef497861e928248` through MLX 0.32.4.
The TensorFold adapter measured its native Python hash/lookup kernels, not its complete serving engine.
The reduced-capacity resident table contained 16001906 rows and 1600190600 bytes, approximately 1.49 GiB.

The crossover command was `scripts/ple_compare.py --suite crossover --mode run`, with the peer and table paths in the snapshot.
It measured 34 token counts from 1 through 2048, with 10 warm-ups and 100 iterations per worker.
The first eight rounds fitted one cutoff. The last eight rounds scored that frozen policy.
All 144 arm validations matched exactly. All 2176 timing workers had zero swap delta.
One correctness-only preflight worker observed four swap-ins. It contributed no timing sample.

Validation-round median `hash_lookup` wall latency in microseconds:

| Total tokens | H1 | H4 | TensorFold |
| ---: | ---: | ---: | ---: |
| 1 | 215.52 | 224.29 | 223.17 |
| 256 | 244.37 | 244.66 | 239.26 |
| 512 | 257.64 | 254.62 | 258.27 |
| 2048 | 338.35 | 324.39 | 322.34 |

The fitted cutoff was 288. Cutoff 256 increased the training objective by only 0.0252%.
A post-fit audit found approximately 0.003% paired policy difference on the held-out rounds.
The integration uses 256 as an engineering cutoff. These measurements do not identify an exact hardware transition.
The cutoff-256 policy took approximately 1.0107 times TensorFold latency across the measured small-token matrix.
The earlier long-input comparison found H4 approximately 3.75%/4.08% slower than TensorFold at 32K/64K.
H8, H16, and the Z token axis added no material combined-path gain there.

Production retains the measured X grid and decode arithmetic. It adds the active-count guard and record-time row-start constants.
Correctness validates those integration changes. A new production performance run has not been collected.
These warm, reduced-capacity measurements do not establish checkpoint-sized cache behavior, loading time, or overlap with earlier layers.
