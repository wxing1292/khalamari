# Token Budget Allocator

The allocator divides a batch's token budget among the sticky requests assigned to one compute slot.
It first preserves request progress, then consumes validated input, and then allocates speculative prefixes.
This document describes the current confidence-aware policy.
The public allocator API is in `runtime::scheduler`.
[`core.md`](core.md) owns scheduling and request lifecycle.
[`future_work.md`](future_work.md#confidence-aware-scheduling) owns the remaining calibration, cost, and integration work.

## Scope

| Constraint | Contract |
| --- | --- |
| Ownership | Runtime orders requests and allocates budgets, executor returns fixed-width proposals/probabilities/confidences |
| Hard limits | Separate `max_requests`, `max_tokens`, `max_tokens_per_request` |
| Page feasibility | Separate from token feasibility |
| `prepare()` | Mutates state, not a planning API |
| Output | `AHashMap<RawRequestID, usize>`, one absolute `token_budget` per request |
| Decode trim | Only a request-local speculative suffix, all other input/anchor tokens mandatory |
| Liveness | Every runnable sticky request receives its full minimum progress budget |
| Ordering | Slot-local sticky order, duplicate IDs across slots permitted, never duplicate IDs within one device batch |

Confidence ranks proposal prefixes only when the batch cannot fit all proposal rows.
The policy has no absolute confidence threshold.

## Request token inventory

```text
slot sticky IDs -> schedule queue -> UserRequest::token_estimate
  -> immutable ReqTokenInventory per present request
  -> allocate_sticky_token_budgets(inventories, BatchBudget)
  -> absolute request-ID-to-token-budget map
```

| Inventory contract | Meaning |
| --- | --- |
| Input | Request ID, ready/queued/speculative counts, borrowed confidences |
| Lane count `L` | Request-local implementation parameter, absent from scheduler API |
| `num_spec_tokens` | Must equal `spec_confidences.len()` |
| `prefill_decode_threshold` | Derived from validated tokens and `L` |
| Min/max validated consumption | Request-local bounds, independent of the per-request hard limit |
| `token_consumption()` with zero budget/validated count | Zero |
| Budget covers validated tokens | Can include available speculative verification |
| Smaller budget | Capped at `prefill_decode_threshold` |
| Remaining global budget | Subtract `target_token_budget - current_token_budget` |

Queries are stateless and use all available proposals.
The output map preserves lookup, not allocation order or budget deltas.
The [allocator](../crates/inference-runtime-core/src/runtime/scheduler/token_budget_allocator.rs) owns exact formulas and boundary tests.
The [scheduler](../crates/inference-runtime-core/src/runtime/scheduler/simple_scheduler.rs) owns inventory collection.

## Minimum validated consumption

The request inventory calculates the minimum budget that produces validated progress:

```rust
request.min_validated_token_consumption()
```

Failure is an invariant violation.
A runnable request must have a valid minimum consumption.
The allocator debug-checks that the result does not exceed `max_tokens_per_request`.

A sticky request must receive its full minimum progress budget.
The allocator panics if all sticky minimums do not fit `token_budget`.
The scheduler guarantees that the sticky request count does not exceed `req_budget`.

## Allocation phases

The sticky allocator executes three phases directly.
It does not admit new requests.

### Phase 1: Minimum validated token budgets

The allocator reserves the full minimum validated token budget for every sticky request.
This phase provides a liveness guarantee for the current working set.

### Phase 2: Maximum validated token budgets

The allocator visits sticky requests in their slot-local order.
It increases each request up to `max_tokens_per_request` or the remaining batch budget.
It also caps this phase at `request.max_validated_token_consumption()`.
This cap is the largest absolute budget that cannot include speculative verification.
It then supplies the capped absolute budget to `token_consumption()`.

The query preserves multi-lane discontinuities.
For example, eight validated tokens and four cache lanes give a `prefill_decode_threshold` of five:

```text
budget 5 -> consumption 5
budget 6 -> consumption 5
budget 7 -> consumption 5
budget 8 -> consumption 8
```

If the complete jump fits, Phase 2 allocates it.
If the jump does not fit, the request remains at the last valid partial consumption.

### Phase 3: Speculative token budgets

The allocator adds only the next proposal position of each eligible request to a heap.
It ranks each candidate by cumulative confidence after an identity transform:

```text
score[r,j] = product(confidence[r,t], t=0..j)
```

After the allocator selects position `j`, it can add position `j+1` from that request.
The allocator never uses a future confidence value to select an earlier position.
When cumulative confidence values tie, the heap selects the earlier proposal position first.
When positions also tie, it selects the earlier request in the slot's sticky order.

The identity transform ranks raw executor outputs directly.
It does not prove that the values are calibrated or comparable across requests.
A telemetry gate must validate these properties before a later policy uses an absolute threshold or a measured cost
decision.

### Worked allocation

This example uses sticky order `A, B`, a batch token budget of `5`, and a per-request limit of `3`.
Each request has one validated input token and two available proposals.
Both inventories use one cache lane.
Phase 1 assigns one token to each request.
Phase 2 consumes no additional budget because both validated inputs fit.
Phase 3 can allocate the remaining three tokens:

```text
+---------+----------------------+----------------------------+
| Request | Proposal confidence  | Cumulative candidate score |
+---------+----------------------+----------------------------+
| A       | 0.90, 0.10           | A0 = 0.90, A1 = 0.09       |
| B       | 0.80, 0.80           | B0 = 0.80, B1 = 0.64       |
+---------+----------------------+----------------------------+

+------+-----------------+----------+-----------------+
| Pick | Available       | Selected | Budgets (A, B)  |
+------+-----------------+----------+-----------------+
| 1    | A0, B0          | A0       | (2, 1)          |
| 2    | A1, B0          | B0       | (2, 2)          |
| 3    | A1, B1          | B1       | (2, 3)          |
+------+-----------------+----------+-----------------+
```

`B1` becomes eligible only after `B0` is selected.
The result contains one validated token and one proposal for `A`, and one validated token and two proposals for `B`.

With the same budgets and every confidence set to `1.0`, the selection is:

```text
candidate order: A0 -> B0 -> A1 -> B1
selected:        A0 -> B0 -> A1        (3 proposal tokens)
final budgets:  A = 3, B = 2
```

This tie case visits the earlier proposal position before it advances either request to the next position.

## Proposal modes

MTP confidence value `1.0` is a placeholder.
It is not a calibrated acceptance probability.
An MTP batch can use the same causal-candidate heap because all scores tie at `1.0`.
The tie-break allocates one proposal position across eligible requests before it advances to the next position.
Request order breaks ties within each position.
This behavior is valid only while the policy has no absolute threshold and one runtime batch does not mix proposal
modes.

Runtime integration must keep proposal tokens, probabilities, and confidence values in one request-local state.
The three vectors must have the same length.
Every prefix trim must change all three vectors.

## Current policy status

```text
sticky minimum progress -> sticky validated input -> causal proposal prefixes
  -> FIFOBatcher admits FIFO work with remaining request/token budgets
```

This lexicographic policy is not a throughput optimum.
`max_tokens_per_request` bounds each request. A batch need not leave room for speculative verification.

## Open decisions

[Confidence-aware scheduling](future_work.md#confidence-aware-scheduling) tracks Phase-2 fairness, measured allocation cost, calibration, and page feasibility.
The current policy has no cost estimator.
