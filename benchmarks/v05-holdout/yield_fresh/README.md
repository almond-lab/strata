# Fresh candidate-yield check

Three new untagged compositions were frozen before the wide portfolio returned
any candidates. The seed is 260493; `freeze.json` binds generator, bytes,
contracts and budgets. The implementation authors received no replacement or
cut. These cases were created after earlier suites were exposed, so they are a
small prospective check, not a blind historical study.

The independently validated results are in `comparison.json` and
`candidate-yield.json`:

| Case | Released best | Pre-wide best | Wide best | Cheaper designs: released / pre-wide / wide |
|---|---:|---:|---:|---:|
| selector_shared_consumer | 156 | 70 | 70 | 0 / 23 / 45 |
| selector_dirty_observable | 156 | 156 | 156 | 0 / 0 / 0 |
| alias_multiplicity_canary | 5 | 5 | 5 | 1 / 12 / 12 |

Counts are distinct canonical executable programs cheaper than each supplied
source, after independent complete-boundary verification and native pricing.
Totals are **1, 35, and 57**. The wide snapshot has **zero** retained discoveries
cheaper than the preceding snapshot's best on these cases. This does not
demonstrate the 1-to-1000 aspiration.

The distinction between executable programs and mechanisms matters. The canary's
12 development programs share one paid-function set and one non-dominated
design. Across the suite, the bounded pure-circuit paid-function-set counts are
1, 17, and 28; measured fingerprints remain explicitly unknown. The extractor
uses native operand truth tables and a Boolean transform, independent of Strata's
symbolic checker, to identify nonlinear paid functions modulo affine terms.
This diagnostic is not a proof of general algorithm equivalence.

There is a useful resource tradeoff on the shared selector. The preceding front
includes `(Q, Toffoli, instructions) = (10, 9, 21)`; the wide front adds
`(9, 8, 20)`, which dominates that point. The best `Q*Toffoli` score remains 70
at `(10, 7, 25)`. All dimensions, including T, rotations and reaction depth, are
recorded in the resource fronts.

All nine selected boundaries were replayed exhaustively after the runs. Every
retained native alternative was independently checked under the same operation
and native-case caps before being counted. Body fragments that fail the complete
parent contract are retained as mismatches in the yield receipt. There were no
native-case or variant-cap omissions here. Hash-only/unmaterialized hypotheses
do not count as discoveries. Older telemetry is not an attempt census.

The released engine uses its broad `restructure_file` catalog because it lacks
the new search CLI. Both development snapshots use their actual `search` CLI
defaults, with common frozen source and semantic budgets. The wide comparison
uses the immutable first installed portfolio wheel; later source changes are
not retroactively attributed to this result. Package hashes are in each summary.

From the repository root:

```sh
.venv/bin/python benchmarks/v05-holdout/yield_fresh/run.py \
  --package-root /path/to/package-root --out /path/to/new-output --label build
.venv/bin/python benchmarks/v05-holdout/candidate_yield.py \
  --fixtures benchmarks/v05-holdout/yield_fresh \
  --run released=/path/to/released-run --run previous=/path/to/previous-run \
  --run wide=/path/to/wide-run --previous previous --out /path/to/new-yield.json
```

The extractor removes display metadata, fixes common input IDs, normalizes
declared clean helpers and independently verified fresh outcome bits, and
collapses symmetric native gate operand order. It separately reports adjacent
involution reduction, pure paid-function sets and resource fronts. No single
one of these counts should be relabeled as a count of new algorithms.

## Final v0.5.0 wheel

The [release replay](v050-release-recheck.json) used wheel SHA-256
`2cdc93d7ad590130c722f7ab3cd1a1b3fee6fe0e51082901bc15799151f6c74a`.
Independent native replay counted 57 distinct cheaper programs across the three
sources. The best costs were 70, 156 and 5 against 156, 156 and 5 on v0.4.2.
The first case accounts for all 45 programs below the previous best.
