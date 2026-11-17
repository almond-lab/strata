# Acceptance ledger — v0.2

Implemented APIs and passing fixtures do not establish historical competition
acceptance. This ledger records the distinction; detailed issue-by-issue scope
is in [V0.2-ISSUES.md](V0.2-ISSUES.md). The product specification is unchanged.

| Area | State | Evidence and limits |
|---|---|---|
| Contracts, debts, patches, immutable records | Implemented | Ownership, fingerprints, mutation, hash and replay checks; integer-key normalization rejects mixed keys |
| Native kickmix and operation-stream input | Implemented frontend paths | Raw JSONL codec round-trip and matching lab/analysis; both use the native kickmix backend |
| Bundle project orchestration | Implemented | Bundle/numeric fixture covers ingest, edit, build, verify and replay; real offline Sandbox acceptance passed |
| Config-only tasks | Implemented | Pinned executable/dependency request/receipt protocol; config-only full pipeline passed real offline Bubblewrap acceptance |
| Frozen source-less components | Implemented scoped evidence | Hash-pinned opaque files excluded only from small component labs, retained in full-artifact gates; no fictional generator |
| Numeric/data co-design | Implemented | Data-only and joint artifact edits carry numeric acceptance/resources through all gates; schema mathematics supplied by checker |
| External callbacks | Implemented trusted extension | Pinned sources/dependencies and typed scoped receipts; raw success dictionaries refused |
| Width accounting | Implemented | `Q` is emitted width, `Q_total` host width, bundle component widths/caps checked |
| Promised-input lab | Implemented finite domains | Explicit canonical domain hash bound to promise, artifact and scale; all-input contracts cannot be weakened |
| Reset legality and batched lab | Implemented supported subset | `R` active-target zero checks, cached segments, bit-sliced input batches; dirty/clean/phase checks preserved |
| HMR branch reduction | Implemented affine subset | Executed versus represented branches separated; nonlinear classical dependencies fall back to exhaustive enumeration; independent whole-native audit covers 768 trajectories across 24 conditional-HMR programs |
| Lab sharding | Implemented partial-run facility | Deterministic shards labeled `sample`; partial receipts cannot certify designs |
| Full-size lab performance | Bounded engineering improvement | Segment/batching/reduction regressions pass; no general full-contest runtime guarantee |
| Source-less exact checks | Implemented bounded symbolic receipts | Fixed-region Boolean/sign-phase maps bound to source/contract/code, replayed with the same backend; unsupported conditions/HMR stay unknown; no inferred scale or whole-task proof |
| Untagged semantic lift | Partial, required core | Exact algebraic functions and bounded episodes; full arithmetic-stage architecture recovery incomplete |
| Functional dependency / slack | Implemented bounded APIs | Exact polynomial family, affine rank or bounded distinct-state checks; unsupported/budget cases return unknown; no free-scratch inference |
| Logical holders | Implemented exact-value query | Public Queries return inferred contents/dependencies under declared clean inputs; dynamic allocation ownership remains unknown |
| Automatic floors | Implemented bounded XOR/AND model | Nonlinear rank/degree lower bounds; exhaustive small minima or unknown on budget; no automatic Toffoli/host-cost extrapolation |
| Build prices and stored gaps | Implemented | Actual measured builds; per-design gap snapshots retained; absent snapshots reported unknown |
| Probability policies | Implemented through project gates | Nonlinear per-draw pricing survives Line/replay; favorable-draw and policy/certificate mutations refused; no exact population-expectation claim |
| Structural research rounds | Implemented orchestration | Dependency/debt closure, mechanism diversity, configured scales, structural witness and replay quarantine |
| File discovery and synthesis | Implemented bounded subset | Native-profile ranked algebraic synthesis; connected-agent hypothesis/candidate/note feedback; no standalone AI-provider invocation |
| Fresh no-hypothesis file trial | Passed original fixture | Untagged eight-bit file, no adapter or hypotheses: two automatic rewrites, static cost 32→16, exact independent lab over 256 arbitrary dirty inputs and replay; not historical/competition acceptance |
| Negative-result memory | Implemented conservative import/apply | Direct and imported strong evidence requires valid blobs; contract/scale changes reopen scoped claims |
| Store-only reports | Implemented | Mixed success/failure/nonbuilt rounds and file notes/candidates; source/config removal tests prohibit rebuilding |
| Offline execution | Runner implemented and release smoke passed | Real offline Bubblewrap run with staged/hash-checked dependencies; no fallback; not a security audit |
| R1–R4 historical restructures | NOT RUN | Need historical source/patch adapters and independent recovered contracts |
| R5 complete host contract trees | NOT MET | Bounded algebraic lifting does not establish full trees |
| R6 actual 322 historical attempts | NOT RUN | Target ledger not found; synthetic regression and separate real-log smoke do not replace it |
| ≥90% structural typing | NOT MEASURED / NOT CLAIMED | Requires tagged circuits with tags stripped; primitive or algebraic coverage is insufficient |
| ≥95% boundary/tag agreement | NOT MEASURED | Golden boundaries and a benchmark receipt required |
| <30-minute full contest gap table | NOT MEASURED | No full contest timing receipt |
| 16 recorded decoder lab draws | NOT RUN | Dataset/adapter needed |
| ≥3/4 historical restructures proposed without hints | NOT RUN | Existing proposals and synthesis do not establish the historical top-k benchmark |
| New verified competitive design | NOT ACHIEVED | The 72→27 fixture is a demonstration |
| ecdsa.fail-specific frontend | NOT IMPLEMENTED | A second raw reversible-operation format exists; it is not an ecdsa.fail integration |

## Real history import evidence

Local read-only smoke checks imported these actual formats without changing
source bytes or committing their private contents:

| Input category | Rows | Bound/priced | Unbound | Closed |
|---|---:|---:|---:|---:|
| Workflow attempts JSONL | 16 | 0 | 16 | 0 |
| Negative-results CSV | 87 | 5 | 82 | 0 |
| Wrapped research-notes JSON array | 1 | 0 | 1 | 0 |
| Separate historical event JSONL | 262 | 0 | 262 | 0 |

The 262 events are not a multi-hundred attempt ledger. Known declared contract
bindings are retained; missing bindings stay explicit. The 322-row regression
remains synthetic. A real target ledger plus verified binding map is needed for
R6 acceptance.

## Existing receipts

Final v0.2 validation: `python -m pytest -q` passed all 321 tests in 55.80 seconds.
The actual offline Sandbox regression passed separately in 6.96 seconds.
The installed wheel passed imports, CLI, automatic file optimization and a
store-only report outside the source checkout. Wheel modules were compared
byte-for-byte with source; the source archive includes documentation and the
validated agent skill. These are release engineering checks, not contest gains.

`REAL-INPUT-SMOKE.json` records analysis of a pinned inverse file (44Q, 22,143
operations) and explicitly reports zero structural typing. `DEMO-RECEIPT.json`
records the original fixture restructuring. Neither is a historical R1–R6
result. See the release change log for compatibility and evidence changes.

Release sandbox receipt: `STRATA_REAL_SANDBOX=1 .venv/bin/python -m pytest -q tests/test_task_issues.py::test_config_only_bundle_full_pipeline_freezes_all_bytes` passed (1 test, 5.36 seconds). It exercised config-only bundle certification, a numeric Line, complete-file verification and replay through the actual offline runner. This does not certify an arbitrary verifier or constitute a sandbox security audit.

Fresh discovery receipt: `tests/test_fresh_discovery_acceptance.py` supplies only an untagged original circuit to the automatic optimizer, then independently checks its external function. The saved report includes automatic region mechanisms and static costs. Project probability tests pass both the nonlinear-pricing/replay case and favorable-draw rejection; public query tests cover holders, dependency and slack.
