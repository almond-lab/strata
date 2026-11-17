# Tensor seeds, bounded improvement and exact exclusions

Tensor rank counts products of a left-linear form and a right-linear form over
GF(2), with free XOR fanout. The input partition and fixed dimensions are part
of the contract. Product counts are construction upper bounds until matched by
a proved lower bound. They do not price arbitrary Boolean circuits, reversible
cleanup, measurements, physical width or a host objective.

## Recover a construction from native gates

`extract_decomposition(circuit, tensor, clean_inputs=...)` reads the circuit's
actual gate sequence. It tracks affine values at CCX controls and propagates
each product through subsequent CX and SWAP gates to recover its final output
mask. Consequently a product with fanout is counted once, even when its target
is temporarily mixed with dirty output accumulators. Products that never reach
an output are recorded with zero output mask and omitted from the seed.

The extractor checks exact affine and bilinear coefficients on every touched
physical wire, restores all declared clean helpers, and checks the resulting
decomposition against the supplied tensor. Untouched physical wires retain
their identity. Declared clean accumulator outputs are recorded separately
from helpers that must return to zero. Register names never imply clean inputs.

```python
from strata.tensor import extract_decomposition, search_decomposition

# `tensor` specifies the same input partition and physical output wires as
# the exact accumulator contract recovered from this complete window.
extracted = extract_decomposition(circuit, tensor, clean_inputs=clean_helpers)
if extracted['status'] == 'verified':
    result = search_decomposition(
        tensor, seed=extracted['decomposition'], timeout=10,
        max_expansions=200_000, max_local_states=64,
    )
```

The supported extraction model is phase-free X/CX/SWAP/CCX with CCX controls
affine in opposite sides of the declared partition. Constants are checked in
the affine remainder. Nonlinear controls, mixed input partitions, conditional
operations, resets, HMR and phase operations return an explicit obstruction.
In particular, CCZ contributes a phase polynomial rather than an accumulator
output; this extractor does not relabel CCZ or measurement uncomputation as
CCX tensor products. An unsupported result is not a claim that the window has
no useful decomposition.

## Improve a seed without a full factor catalog

`search_decomposition(..., seed=...)` verifies the seed before using it and
retains the best verified construction throughout the search. A supplied
42-product witness cannot be replaced by a 144-product coordinate baseline.

Local search uses two exact transformations:

- A linear dependency between two-factor matrices allows one product to be
  removed by redistributing its remaining factor among the other products.
  Each of the three choices of paired factors is tried.
- Shared-factor flips change a pair of products while preserving their tensor
  sum. A bounded frontier explores these alternatives and tries further
  dependency reductions.

Every accepted reduction is checked by exact coefficient equality. State,
expansion and elapsed-time budgets bound the search. No improvement or a
budget-limited neighborhood supplies a rank lower bound. This is a bounded
local method; it does not promise to improve an arbitrary decomposition or to
connect every rank-equivalent factorization.

`search.coordinate_rank`, `search.seed_rank`, `search.local_search` and the
final `verification` distinguish the starting constructions and the performed
work. `search.catalog_size` is the full count
`(2**a - 1) * (2**b - 1) * (2**c - 1)`, also included in the obstruction text
when it exceeds `max_catalog`. Local search does not materialize that catalog.
Regression coverage includes 12×12×12 and 24×24×48 gate-derived seeds, plus a
case where a flip exposes a reduction unavailable in the original factors.

The `bilinear_tensor_decomposition` restructure constructor extracts this seed
automatically. It keeps the existing native implementation if no product-count
improvement is found. A smaller tensor construction still needs reversible
lowering with the available clean helpers and the complete boundary check.
If lowering fails, the report retains the original native circuit and records
the lowering obstruction alongside the algebraic improvement. The construction
loop measures emitted resources; a smaller product count alone does not prove
a lower host cost.

The direct, tensor, guided and learned-template constructors preserve existing
public register definitions and reuse an already recorded boundary wire. A
name collision uses a fresh suffix, allowing retained alternatives to enter
subsequent rounds.

## Preserve completed exhaustive exclusions

The complete search enumerates all nonzero rank-one factors only when the
catalog fits. If every rank through `searched_up_to` has been excluded,
including smaller ranks already ruled out by tensor bounds, then
`proven_lower_bound >= searched_up_to + 1`. This evidence remains valid if a
later rank hits a timeout or expansion budget. An interrupted rank is never
added to `searched_up_to`.

Such a result carries `evidence_kind='exact_small'` and a separate
`lower_bound_evidence` record even when the best upper bound is larger and
`status='construction'`. `optimal_in_model` becomes true only when the proved
lower and verified upper bounds meet. The reason names completed exclusions
instead of reporting only the original coordinate construction.

Complete search is single-threaded Python and combinatorial. The reported
3×3×3 GF(8) rank-five exclusion took approximately 150 seconds in
[issue #46](https://github.com/almond-lab/strata/issues/46); that report is not
a performance guarantee. Even a catalog that fits can have an impractical
combination search. For 4×4×4 and larger tensors, prefer existing circuit seeds,
bounded local improvement or another specialized method over assuming that
the default complete search will scale.
