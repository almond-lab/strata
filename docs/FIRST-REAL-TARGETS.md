# First real optimization targets: decoder and chemistry

User direction: after the library is ready, use Strata to optimize the decoder
and chemistry circuits. This is the next application priority. The existing
background experiment queue is separate and may continue; its results become
scoped evidence for Strata, not automatic certified designs.

Start from the circuit files and the connected-agent discovery loop. Do not
require the user to reconstruct the stage tree or write a TaskAdapter. Library
host connectors supply contest-specific acceptance and cost data; generator
source is useful when available but is not a prerequisite to file analysis.

## Base

Use the user's confirmed cand1 archive unless the user supplies a newer base.
Recheck hashes before starting task integration or experiments.

- Archive SHA256: `0212e251262a4d3d40ab5a36af664e8e11f24c14305db9c2105ceaf22253e443`.
- Decoder SHA256: `023535bcd7b93acd73bdcc15708a2e304086b1bb76f36c9148ca29c858f4e2ac`.
- Chemistry NPZ SHA256: `dba70ba75b496ac22d22d5100f9ef810446c0ad52a2f219b168d768a4101ceda`.
- Chemistry also includes all 15 loaders. The NPZ hash alone does not identify
  the task. Bind the complete archive/member manifest before qualification.

Archived preflight costs are decoder `341113889586.797` and chemistry
`293921487.61443824`. These are reference receipts, not fresh Strata measurements
or the current public leaderboard score. Inverse square root remains the fixed
third task when calculating portfolio deltas unless explicitly changed.

Private circuits, factorization arrays, verifier sources and credentials stay
outside this library repository.

## Readiness for a real task

A task is ready for Strata-driven optimization when all of the following have
measured evidence:

1. The library's host integration reproduces the exact base's current official
   cost/acceptance receipt, with verifier and dependency identities pinned.
2. The selected producer–consumer–return region has an independently checked
   external contract, ownership and phase-debt specification. Opaque regions
   elsewhere stay opaque; no fictitious full-circuit semantic coverage.
3. A rewrite relying on scale-down has one parameterized generator for small
   and full instances, covering zero/exception branches, dirty state and
   measurement outcomes. A file-only exact rewrite instead needs full-boundary
   equivalence evidence at its actual size; it cannot invent a scale-down model.
4. Measured prices include consumer adaptation, both traversals, recomputation,
   scratch, numerical error/failure effects and phase repayment.
5. Full oracle and official verifier runners work through Strata, and the base
   and one deliberate negative mutation exercise the certificate/replay gates.

These task gates allow useful integration work before every broader product
research milestone is met. They do not declare the full specification complete.
Current missing implementation work remains in [ACCEPTANCE.md](ACCEPTANCE.md).

## Decoder integration and search

Recover and check stage contracts from the submitted decoder and its measured
structure. If source is available, reproduce it and bind it to the exact emitted
file before relying on its annotations. Use the detailed circuit report and the
latest algorithm write-up as evidence to verify, not as authoritative counts
for a different version.

Prioritize structural producer–consumer changes in extraction/restoration and
conversion/payload episodes. Preserve independent syndrome information and
account for all zero/gap branches and HMR phase debt. A component's local wire
reduction does not reduce the task peak automatically. The hard 889Q cap and
agreed failure policy remain binding for candidate promotion.

The host integration must price the actual `R(fail_rate) * Q^2 * mean_toffoli` objective
using the host's pinned failure rule. Commit comparison seeds before evaluating
candidates. Passing a favorable single draw cannot select a winner.

## Chemistry integration and search

Implement a first-class task bundle: all loaders, the factorization and the host
resource formula. A loader-only circuit certificate cannot certify chemistry.
Reproduce the exact chained valid-address/dirty-word domain and its return path.

Prioritize joint loader-state/representation changes, cross-loader sharing and
physically valid numerical co-design. Charge all state preparation and adapters;
shrinking a loader below a host-dominated workspace peak does not lower total Q.

Each candidate needs both circuit checks and the unchanged physical checks for
one-/two-body errors, energy gap and effective normalization. Price the complete
`lambda_eff * T_total * Q_total` objective. Coarse words or a cheap code are only
useful when a corresponding accepted physical factorization exists.

## Retain and report

Every attempted design uses an immutable attempt record or named Strata line.
A successful design has the appropriate exact-equivalence or scaled-lab evidence,
measured build, full oracle, official receipt and replay evidence.
Report exact same-baseline task-cost and portfolio-score deltas, including the
fixed third task and unchanged draw policy. Keep losing siblings and scoped
reopen conditions. Reconcile background attempts by contract and method before
repeating them.

No automatic Kaggle submission. User approval is still required for submission.
