# Independent file-first acceptance review

The direct file workflow passes the bounded acceptance cases below. It reads
untagged `.kmx` and `.kmb` circuits, recovers complete multi-operation Boolean
functions and sign phase, and constructs cheaper equivalent implementations
without a task adapter or a supplied answer. **This does not complete the
specification's decoder understanding or L3/L4 search requirement.**

The review uses original circuits constructed separately from the implementation.
Physical wire assignments and legal operation schedules are permuted. Only the
circuit enters the optimizer: the expected functions remain in the independent
test oracle. This is a regression suite for the implemented algebraic family,
not a held-out benchmark of unfamiliar algorithms.

## Reproducible evidence

```bash
.venv/bin/python -m pytest -q tests/test_file_first_acceptance.py
```

The suite contains 24 passing cases, including one case that checks 24 additional
random mixed-gate programs. The file CLI tests deliberately point `--project`
at a missing file. Both input formats still produce `candidate.kmx`, recovered
semantics, a contract tree, and an equivalence report.

The main fixture has nine wires and 17 operations. Two interleaved producers
form distinct sums of quadratic and affine terms on intermediate wires. Two
outputs consume these values, sign phase depends on a produced value and a
spectator, and the producers are reversed. All nine initial wire values are
arbitrary, including both intermediates and both output accumulators.

Across the tested permutations, the default static objective decreases from
72 to 54: eight paid operations become six at the same nine-wire width. The
candidate has 15 operations. This includes the cost of the resulting cubic
sign-phase terms. No promised clean initialization or width saving contributes
to the result.

Every candidate is checked independently with native kickmix on all 512 initial
assignments. The checks compare every final wire and exact phase in half-turn
units. The inferred output polynomials are also evaluated against native
execution. The additional mixed programs exercise X, CX, CCX, SWAP, Z, CZ, CCZ,
and NEG, including cases the synthesis construction conservatively leaves alone.

The failure cases check:

- Changed outputs, changed returned intermediates, and changed sign phase each
  break equivalence. Phase-only changes leave every output bit correct and still
  fail.
- Rewriting a materialized product as though its intermediate started at zero
  fails on dirty assignments.
- Forged success flags, corrected artifact hashes, and correctly remeasured
  resources cannot make an incorrect replacement pass a fresh semantic replay.
- HMR, classical conditions, fractional Z rotations, and unsupported CSWAP
  operations remain opaque barriers. Scoped condition bodies remain opaque too.
- An operation-budget limit returns incomplete evidence rather than a proof.
- Ordinary gate cancellation does not qualify as a structural materialization
  gain.

## Finding corrected during review

A compute/consume/consume/uncompute sequence originally qualified as structural
even when the two consumers canceled. A spectator operation preserved the file
width, allowing the optimizer to accept the identity episode's removal and mark
`structural_change=true`.

The implementation now requires a materialization to have a net functional
contribution before qualifying its removal. The regression includes the
canceled-consumer episode and verifies that it produces no accepted structural
claim. This distinction matters because meaningful stage work cannot be counted
by changing labels on algebraic cancellation.

## What the evidence establishes

The recovered contracts describe multi-operation input/output maps and their
phase. The construction derives direct output updates from these maps. It can
remove intermediate writes across a complete producer/consumer/return episode,
including the effect of the intermediate's original dirty value. This is useful
bounded evidence of a materialization-boundary change, consistent with a narrow
L3 mechanism.

The semantic proof applies to arbitrary boundary states for the supported
permutation/sign operations, rather than only the test assignments. The
independent exhaustive tests check that the implemented symbolic semantics agree
with native execution on the original small cases. Unsupported spans are retained
and exact substitutions are checked separately. The certificate preserves the
supplied circuit's behavior; it does not establish that the supplied circuit
solves a contest task.

The construction currently accepts acyclic output XOR updates of degree at most
two and sign-phase polynomials of degree at most three. It preserves physical
width and uses static emitted SDK resources. It does not infer clean ancillas,
measurement-debt transfers, numerical approximations, task distributions, or an
official verifier's acceptance rule. Resource improvement is not an official
contest score.

## What remains before full acceptance

An exact bounded polynomial is useful semantics, but it does not identify an
extractor, field inverse, divider, table interpolation scheme, or decoder
algorithm. The reported compound-algebra coverage must remain separate from the
specification's 90% typed-stage target. High-level stage coverage and the 95%
boundary-agreement target remain unestablished.

The next acceptance work requires:

1. Recover arithmetic and field blocks, register meanings, object lifetimes,
   dependencies, and measurement-debt ownership from whole untagged files, with
   checked block certificates and explicit unknown regions.
2. Derive reusable smaller block instances from those contracts, and compare them
   against the real decoder's recorded failure classes. A bounded input window
   alone does not provide a structural scale parameter.
3. Search changes to representations crossing neighboring contracts and generate
   alternative stage lists. The current output-polynomial construction supplies
   neither general representation design nor L4 algorithm discovery.
4. Measure proposed full-size builds under the host's domain and cost formula,
   run the independent official verifier, and retain replayable lineage with
   verified floors and clear evidence scope.
5. Execute the specification's R1–R4 replays without answer-naming hints, recover
   R5's complete host circuit contract trees, and demonstrate a new verified
   design outside those examples.

These tests establish a working adapter-free subset. They do not establish the
full library's stated acceptance milestones.
