# Change log

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
