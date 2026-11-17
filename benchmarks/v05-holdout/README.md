# Independent generated holdouts

Nine untagged sources were generated and hashed before their first search. The
compiler received only bytes, explicit zero-input contracts, a cost expression,
and common search ceilings. No cut, replacement, or fixture-specific constructor
was supplied. The review agent had read the existing implementation and tests,
so this is a prospective independent-agent check, not a blind historical study.

The original immutable evaluation is in `released-summary.json`,
`full-summary.json`, `ablation-summary.json`, and `comparison.json`. All selected
bytes were replayed again after the three searches. Later runs must use different
receipt names; these original results must not be overwritten.

| Frozen case | Released v0.4.2 cost | Full current cost | Current minus three new families |
|---|---:|---:|---:|
| field_difference_0 | 16 | 16 | 16 |
| field_difference_1 | 22 | 20 | 20 |
| modular_difference_3 | 0 | 0 | 0 |
| modular_difference_4 | 9 | 8 | 8 |
| table_difference_0 | 0 | 0 | 0 |
| table_difference_1 | 8 | 8 | 16 |
| phase_obligation | 48 | 1440 | 1440 |
| dirty_workspace_obligation | 7 | 7 | 14 |
| public_classical_obligation | 704 | 704 | 704 |

The objective is complete emitted `Q*toffoli`. Full search has two modest wins,
six ties and one loss against the release. It has two wins and seven ties against
the controlled ablation. Both ablation wins select `output_basis_search`, and
both tie the released optimizer. These results do not establish that the three
new families broadly outperform the released catalog. The mixed value/phase
case exposes a real limitation of the current deep-only path.

The ablation disables exactly `consumer_functional_decomposition`,
`output_basis_search`, and `factored_observable_search`, retaining current
deep-only eligibility, ordering, and budgets. Each receipt also reports the
earlier narrow criterion: baseline or a verified descendant of a
`removed_materialization_bank` witness. Under that criterion the full engine's
two output-basis selections instead cost 16 and 14. The released engine has no
such annotations; its raw narrow fields cannot be interpreted as comparable
historical acceptance. Its actual broad-catalog best is the release column.

All 27 selected boundaries passed exhaustive native checks; 54 value/phase
mutations were rejected. Every arbitrary physical quantum input, including dirty
outputs, sparse holes and the undeclared scratch wire, was enumerated. All
initial classical bits and every formal HMR/R outcome are also enumerated by the
checker. Ordinary gates run in kickmix; a small independent primitive adapter
forces an HMR/R outcome and adjusts its local sign. The adapter self-check
accepts repaired measured cleanup and rejects a missing repair and a corrupted
public classical value. All selected holdout winners happened to contain no
measurements, so these results are not evidence of measured holdout discovery.

Summed discovery wall times were 2.78 seconds for the release, 29.87 seconds for
full search, and 23.47 seconds for the ablation. The runs executed concurrently;
equal CPU consumption is not claimed. Peak worker RSS and individual times are
recorded. Every source had a 45-second process cap, one round, 12 candidate
attempts, 512 source operations, and the same semantic budgets. There were no
errors, timeouts, or omitted cases in the original evaluation.

`generate.py` uses no Strata code. `freeze.json` binds its seed, hash, fixture
bytes, contracts and budgets. It refuses to regenerate an existing corpus.
`run.py` accepts a directory containing an installed `strata` package; output
directories must be empty so archives cannot leak between sources or runs.
Run from the repository root:

```sh
.venv/bin/python benchmarks/v05-holdout/run.py \
  --package-root /path/to/released/package-root \
  --out benchmarks/v05-holdout/work/released-replay --label released-replay
.venv/bin/python benchmarks/v05-holdout/run.py \
  --package-root src --out benchmarks/v05-holdout/work/full-replay --label full-replay
.venv/bin/python benchmarks/v05-holdout/run.py \
  --package-root src --mode ablation \
  --out benchmarks/v05-holdout/work/ablation-replay --label ablation-replay
```

Use a frozen package copy for comparisons while development continues. Package
file hashes in the receipts, rather than labels, identify the evaluated code.
The original snapshot remains in ignored `work/package-ready/strata` locally.
`check_results.py` rechecks fixture/code/runner bindings, emitted bytes, physical
layout, quantum values, public classical values and exact sign. It produced the
original compact receipts; do not rerun its publishing step over those filenames.
Generated search dumps and package copies stay in ignored `work/`.

The source families cover GF(4) polynomial differences (including zero inputs),
modular arithmetic tables (including wraparound), and paired Boolean evaluator
tables with shared dirty outputs. The negative obligations retain phase, dirty
workspace and public classical effects. There is no global-optimality, official
score, historical-discovery, or release-readiness claim.

Candidate-yield work is recorded separately. `candidate-yield-original-validated.json`
preserves the earlier retained-alternative audit (33 released, 158 full and 112
ablation canonical cheaper programs). The later ordinary-CLI, exposed-corpus
comparison in `wide-cli-exposed-comparison.json` records 33 released, 114 preceding
CLI and 286 wide CLI programs; no candidate beats the preceding CLI's best score.
Pure paid-function-set counts are 17, 12 and 26, showing why executable program
count must not be presented as mechanism count. Timing is not the objective of
these later checks. The separate [fresh yield suite](yield_fresh/README.md) contains
the prospective three-case comparison and its duplicate-materialization canary.

`candidate_yield.py` is the independent extractor. It checks native emitted
alternatives under the complete source boundary, prices them, retains audit
samples, canonicalizes naming and symmetric operands, and produces objective
histograms and resource fronts. Unmaterialized hypotheses and missing telemetry
are not counted as discoveries. `mixed/README.md` describes the separately frozen
value/phase composition checks. The original nine-case summaries above are
immutable; later mixed and portfolio runs are explicitly labeled post hoc.
