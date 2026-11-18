# Change log

## 0.5.2 — 2026-11-18

- Preserve the finite input binding when replaying a cut candidate in its
  parent. Producer cuts can leave declared-clean inputs holding live outputs.
- Use active wires and independent care-row columns for proposal budgets.
  Search exposes `--max-table-cells` and `--max-address-bits`.
- Use bounded SAT for exact joint searches requesting at least four ANDs.
  Replay each SAT witness and retain the per-depth solver receipts.
- Recover reversible proposal models for cancelling carry offsets, proved
  scoped phase repairs, and preserved externally controlled diagonal phases.
  Price measured cleanup and terminal-product variants against the measured source.
- Try ordered in-place updates under an explicit promise and report the
  limitation when the available formulations cannot represent the map.
- Verify candidates that omit trailing proved-clean scratch under a domain.
  Price the emitted width. Independent-map evaluation defaults to 1,024
  proof operations and reports how to raise the limit when needed.

## 0.5.1 — 2026-11-18

- Bind a complete domain directly to an explicit episode with
  `--domain-at-cut`. Preserve the caller's reachability obligation in the receipt.
- Accept proved zero resets in promised lookup search. Recover measured
  producers even when the phase repair rebuilds the controls after HMR.
  Compare measured and reversible erasure of reconstructed product caches.
- Expose algebraic factor budgets and consumer partition budgets. Report
  the effective ANF caps separately from the decision-graph fallback.
- Add a short `search --summary` receipt for stdout and a byte-valued
  `wrap --max-output` option.
- Check a narrower candidate when its omitted trailing wires are declared and
  proved clean. Surface the exact measurement-alignment obligation on evaluation.

- Extend promised-input search with exact care-row XOR/AND search, shared
  cofactor graphs and reversible output encodings. Receipts explain skipped
  formulations and bind independently checked replacements to the input promise.
- Recover the pure value map of standard private measured AND cleanup before
  proposing replacements. Verify and price each replacement against the
  original measured source.
- Expose joint solver AND, time, state and expansion bounds on `search` and
  `restructure`.
- Preserve shared lookup caches while fusing copy-only terminal products.
  Compare each constructor's checked cost with the supplied source cost.
- Retrieve complete product lifetimes before generic windows, deduplicate
  overlapping selections, and replay safe commutation certificates when
  localizing noncontiguous consumers.

## 0.5.0 — 2026-11-17

- Jointly explore input and output representations and shared nonlinear
  factorizations. Independently price and verify multiple finalists and retain
  distinct resource tradeoffs. A discovery receipt counts accepted native
  circuits and distinguishes changes in their nonlinear products on small
  supported boundaries. The CLI screens ineligible families and supports
  bounded, cancellable constructor trials.
- Infer nonlinear intermediate predicates by grouping producer assignments
  with identical effects on every consumer. Search output encodings and
  factored sign observables, with dedicated structural witnesses at fixed width.

- Add `search`: recover whole Boolean output equations, jointly factor them,
  search invertible affine representations and complete reversible schedules,
  and verify emitted candidates under their full parent objective.
- Add algebraic construction beyond bounded truth tables, terminal-product
  consumption, phase-debt-aware measured cleanup and explicit private-bit scopes.
- Add bounded symbolic all-outcome labs, including nonlinear conditions and
  exact dyadic phase; retain explicit failures and node-budget limits.
- Correct live status and legacy FIFO handling, clean-fact transport, range and
  helper-layout search, expected conditional costs, measured value-map floors,
  final factored projections and source-bound boundary observations.
- Add frozen generated v0.4.2 comparison fixtures, separate symbolic scale
  regressions and a [delivery/acceptance plan](docs/V0.5-DISCOVERY.md).

- Search over a complete [finite input promise](docs/PROMISED-DOMAINS.md).
  A compact address set or interval can be checked across arbitrary dirty words.
- Factor [native fractional rotations](docs/FRACTIONAL-OBSERVABLES.md) and check
  their exact dyadic phases together with the output bits.
- Add measurement-debt queries, compact evaluation contracts, a binary Griesmer
  tensor bound, explicit wrapped source dependencies and thread-aware CPU limits.
  The [issue ledger](docs/POST-0.4.2-FIXES.md) gives the scope for #86–104.

The [transfer sweep](benchmarks/v05-transfer/README.md) covers 20 bounded
episodes from a submitted portfolio. Both the conservative and prefix-promise
runs retained the original bytes on all 20 under the fixed-width static paid-gate
objective. On three generated cases, the [release wheel](benchmarks/v05-holdout/yield_fresh/v050-release-recheck.json)
found 57 cheaper programs versus 1 for v0.4.2 and improved the previous best
on one case (156 to 70). General word-arithmetic recovery,
numeric table fitting and unhinted whole-host discovery remain open.

## 0.4.2 — 2026-11-17

- Fix reported execution, scheduling, comparison and lab problems after v0.4.1;
  see the [issue map](docs/POST-0.4.1-FIXES.md) for #60–85 and exact scope.
- Make tiny exact prototypes, explicit scaling estimates, bounded full builds
  and final-only expensive checks the default connected-agent research ladder.
  A small pass never establishes full-size correctness.
- Preserve scientific rejection evidence while recovering infrastructure
  failures; guard artifact identity, draw completeness and calibration leakage.

Validation: **844 full-suite tests passed**, plus **13 late range/floor
regressions** and **27 actual sandbox checks**. A real tiny-to-full research
workflow and the installed wheel passed independent smoke checks. See the
[validation receipt](docs/V0.4.2-VALIDATION.json) for commands and limits.

## 0.4.1 — 2026-11-17

Fixes eight reports filed against v0.4.0, #52–59. See the
[issue map](docs/V0.4.1-FIXES.md) for scopes and limitations.

- Compare compatible candidates produced by different manifests under a pinned
  verifier/objective contract. Verifier source closure, source layout, constants
  and explicit scientific contracts remain part of comparison identity.
- Precommit finite verifier draws and aggregate complete artifact results with
  mean, worst or vector-only policies. Failed, partial, conflicting or unmatched
  draw sets cannot produce a lucky verified winner.
- Preserve staged filenames, extensions, sibling imports and callback sources.
  Accept valid empty text/JSONL logs with an explicit nonempty override.
- Compact verified duplicate attempt files by default while retaining immutable
  artifacts, logs, checkpoints and failed unvalidated outputs. Use private
  reflinks when available; transient disk copies still require capacity.
- Recognize bilinear accumulators with affine terms, recover supported complete
  boundaries with canonical Boolean decisions, and lift complete measured XOR
  tables with workspace, dirty-target and all-outcome phase checks.
- Bound native replay allocation by physical quantum/classical width and batch
  size, including sparse physical IDs. Symbolic budget failures identify their
  failing instruction and remain explicit unknowns.

A real 128-qubit chemistry loader now yields a complete 2,048×40 table and native
replay under its declared clean-workspace contract. The constructor retained the
original: no cheaper circuit or competition gain is claimed. Historical semantic
typing and research-discovery milestones remain open.

Validation: **736 tests passed**; all six default namespace skips were covered
in explicit real offline sandbox checks, including final rejection regressions. The installed wheel passed a
complete independently verified circuit experiment and agent-packet generation
outside the checkout. See [the validation receipt](docs/V0.4.1-VALIDATION.json).

## 0.4.0 — 2026-11-17

Adds a runnable experiment workflow for existing numeric, circuit-generator and
data-processing scripts. JSON/TOML manifests replace handwritten adapters for
command invocation, declared outputs, structured metrics and independent checks.
See [the first-run guide](docs/TEN-MINUTES.md) and
[workflow documentation](docs/V0.4-WORKFLOW.md).

- Bounded sweeps with explicit seeds/replicates, transactional experiment identity,
  owned attempts, cooperative host reservations and durable completion events.
- Separate execution, artifact validation and independent acceptance; verified
  objective comparisons use verifier-bound resources and frozen constants.
- Pinned staged inputs, runtime/environment and readers; streaming artifact
  retention; checked stage lineage and hash-bound checkpoints on resume.
- Owned cancellation and conservative recovery, bounded logs and honest signal
  diagnostics; unsupported hard aggregate quotas fail before launch.
- Calibrated cheap gates with disjoint held-out evidence and deterministic full
  audits. A discovered false negative suspends pruning until recalibration.
- Experimental live-run continuation policies, validated against whole measured
  trajectories including late wins. Advisory by default; enabled pauses retain
  checkpoints, original budgets and explicit override receipts.
- Reproducible reports and agent packets connecting evidence, hypotheses,
  decisive experiments and results; compatible transfer prompts reuse the
  numeric study. The updated agent skill explains the complete loop.

Reference fixtures cover numeric fitting, native circuit generation and a
non-contest data task. Their verifiers are reference implementations, not official
competition scorers. The backend is single-host Linux with mandatory offline
bubblewrap; no cluster service or automatic external submission is installed.
Historical circuit-discovery milestones remain open, and no contest gain is
claimed by this workflow release.

Validation: **671 tests passed**, with five namespace-dependent default skips
covered in the separate selected run (51 tests passed with real sandbox mode
enabled). The twenty-setting reference benchmark produced identical outcomes,
6.10 s sequential versus 4.10 s with two workers. See
[the validation receipt](docs/V0.4-VALIDATION.json) and
[benchmark scope](docs/V0.4-BENCHMARK.md).

## 0.3.1 — 2026-11-17

Fixes thirteen v0.3 reports #36–47 and #51: clean-reset consistency, bounded
segmentation and paid-skeleton retrieval, file hash/clean-contract handling,
search budgets, constructor metadata, seeded tensor search and retained rank
exclusions, numeric diagnostics, extreme-confidence metadata, and change-local
candidate proofs that avoid lifting the entire unchanged circuit.

Adds a measurement-aware lookup constructor for supported quadratic and cubic table
arithmetic. It preserves ordered HMR events and their phase repairs and checks
the complete boundary; larger unsupported synthesis families remain explicit.

See [the patch issue map](docs/V0.3.1-FIXES.md) and
[measurement-based lookups](docs/MEASURED-LOOKUPS.md). The independent research
run's deeper capability gaps are tracked separately in #48–50.

Validation: **583 tests passed**, plus three real offline sandbox checks.
The real 128-qubit loader has 100% bounded local exact coverage with all 80
resets justified; its new cubic plans are more expensive, so the original is
retained. See [scope and evidence](docs/V0.3.1-VALIDATION.json).

## 0.3.0 — 2026-11-17

This release adds generic candidate construction and a more informative
Strata–agent research loop, while fixing the reports filed by v0.2 users.
See [issue coverage](docs/V0.3-ISSUES.md), the [workflow guide](docs/V0.3-GUIDE.md),
and [remaining research limits](docs/V0.3-RESEARCH-LIMITS.md).

- Branch-parametric HMR and classical-control effects, explicitly clean resets,
  CSWAP, phase-debt summaries, and a factored Boolean expression graph alongside
  the bounded exact ANF oracle.
- Finite-domain checks and interior queries with domain hashes, declared clean
  inputs, operation/term budgets, and explicit whole-boundary output obligations.
- Constructive joint-support encoders/decoders, clean phase-workspace elimination,
  direct producer–consumer synthesis and multi-output bilinear decomposition.
- Certified tensor flattening and bounded slice-substitution lower bounds, exact
  small tensor search, and priced reversible lowering with dirty outputs restored.
- Extensible construction rounds preserve verified nonwinning alternatives and
  learn replay-checked native transformation templates; no remote AI is invoked.
- Large-file discovery uses bounded Python windows and SQLite occurrence indexes,
  groups wire-renamed kernels, reports progress and supports instruction ranges.
- Native binary bundles, virtual-environment execution, configurable sandbox
  resource limits, retained failed receipts, safe mathematical cost functions
  and count-aware probability penalties.
- Generic numeric research: content-addressed named arrays and trainable masks,
  pinned calibrated scoring, cost-first pipelines, error budgets, shape prices,
  measured experiment transfer, convergence models and exact small quadratic floors.
- Restartable checkpoint jobs with frozen budgets/stop policies, safe portfolio
  and bet accounting, and readiness-triggered local packaging/full verification.

Historical automatic stage-recognition and no-hint winning-restructure benchmarks
remain open. No official contest gain is claimed by this library release.

Validation: **465 tests passed**, with separate real offline sandbox checks for
binary bundles, virtual environments, checkpoint/resume and automatic local
certification. A measured native benchmark built and simulated **1,000,000
distinct small circuits in 595.85 seconds**, checking **256 million input
trajectories**. Analysis-guided versus unguided decomposition experiments retain
separate counts for algebraic specifications, native artifacts and proofs.
See [the validation receipt](docs/V0.3-VALIDATION.json) for precise scope.

## 0.2.0 — 2026-11-17

This release expands artifact/task integration, strengthens exact-lab and
negative-evidence checks, and adds saved research reports. The
[issue coverage map](docs/V0.2-ISSUES.md) records implemented scope and outstanding
acceptance milestones; it does not claim completion of every historical benchmark.

Release validation: 321 tests passed; the configuration-only certification and
replay test also passed through the actual offline Sandbox. The built wheel
was installed into an isolated target and exercised outside the checkout,
including automatic optimization and store reporting. Source archives include
the human documentation, agent skill, examples and tests.

### Artifact and task integration

- Task-specific frontends and canonical multi-file bundles; numeric/data files,
  frozen opaque components and component width limits participate in ordinary
  Lines, certification and replay.
- Configuration-only tasks with pinned executable/dependency identities and
  typed full-artifact receipts; no Python adapter or invented scale model needed.
- Typed pinned build/check callbacks for trusted external integrations, including
  ResearchRound execution.
- JSONL reversible-operation input with a lossless supported codec, sharing the
  native kickmix backend.

### Lab and analysis

- Explicit promised-domain inputs and hash binding; all-input contracts cannot
  be narrowed by supplying a subset.
- Checked zero resets, cached native segments and batched input simulation.
- Proved affine HMR dependency reduction where applicable, plus conservative
  full-enumeration fallback and independent native audits.
- Deterministic partial shards, scoped component labs, and distinct fixed-full,
  small-size, partial and external-artifact evidence kinds.
- Scoped functional dependency and information-slack APIs; nonlinear rank/degree
  bounds and bounded exact minima in an explicit free-XOR/AND model.
- Stored fixed-region symbolic receipts for source-less files, including
  output/sign-phase maps, contract/byte/code bindings and explicit same-backend
  replay. Unsupported regions remain unknown; no small model is inferred.
- Precommitted probability-policy comparison through project/Line/replay gates
  with per-draw objectives, failure budgets, penalties and refusal of
  candidate-selected favorable draws.

### Research memory and reports

- JSONL, CSV and JSON-array/wrapped-note import, declared contract bindings and
  explicit unbound rows; scoped apply/reopen reporting for a new base.
- Matching artifact requirements for direct and imported strong negatives.
- Store-only JSON and Markdown summaries of automatic file optimization regions,
  research plans/results, Lines,
  mechanisms, measured deltas, evidence scopes, file candidates, prebuild notes,
  shared failures, reopening conditions and per-design saved gap snapshots.
  Reporting runs no generators or third-party code.

### Compatibility and evidence changes

- Recursive integer dictionary keys normalize to strings **before** sorting and
  hashing. Mixed key types, including collisions such as `1` and `"1"`, are
  rejected. Existing string-key JSON encoding and hashes are preserved. Saved
  records are not rewritten, and previously unreadable integer-key records are
  not silently migrated.
- `Store.negative(..., evidence=...)` accepts a stored blob hash or structured
  `artifact_hash`. `Line.park(..., evidence_artifact=...)` forwards the same
  evidence. Missing/corrupt blobs downgrade `proof`, `build` and `exact_small`
  to `estimate` / `priced`; the requested kind remains recorded. Exact-small
  also requires size. Reports recheck old evidence rather than trusting an old
  unsupported closure label.
- Lab `tested` counts actual executed input/branch checks.
  `represented_branches` and `represented_trajectories` count coverage justified
  by the supported reduction. `exact_small` requires explicit scale; partial
  shards are `sample` and cannot certify designs. External full-artifact checks
  do not masquerade as exhaustive small labs.
- `Q` means emitted width; host formulas should use `Q_total` separately.
  Sequential bundles report their maximum component width and each component's
  width. New certificates also bind task/frontend configuration and scoped
  evidence, so legacy certificates lacking those bindings may need recertification
  before current project validation/replay. Reading old immutable records remains
  available; unchanged record hashes do not promise schema-level validation.

### Acceptance limits

The ≥90% structural typing / ≥95% boundary agreement benchmark, historical
≥3/4 no-hint restructure proposal benchmark and R1–R6 replays remain unestablished.
General source-less scale recovery, broad arithmetic recognition, a production
competition integration and a new competitive design are not claimed.

Actual local import checks covered 16 workflow attempts, 87 CSV rows and one
wrapped research note. A separate 262-row event log verifies multi-hundred-row
input handling; it is **not** the missing historical attempt ledger. Synthetic
322-row tests remain synthetic. The config-only bundle pipeline also passed the actual offline Sandbox test
(1 test, 5.36 seconds). A fresh no-hypothesis original eight-bit trial found two automatic rewrites,
reducing static cost 32→16; independent exact checks covered 256 arbitrary dirty
inputs and replay passed. This fixture does not satisfy the historical
no-hint or competitive-design milestones.
