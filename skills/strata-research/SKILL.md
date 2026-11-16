---
name: strata-research
description: Use or extend Strata to propose, test, price, and retain contract-level quantum circuit restructures. Applies to Strata projects and adapters, especially producer-consumer changes and evidence-backed replay.
---

# Strata research

Use this skill for an existing Strata project or when integrating a task into
Strata. Preserve the user's chosen circuit, objective and experiment budget.
The skill does not authorize submissions, publishing, VM creation or unrelated
research. Existing user authorization still applies.

## Locate the implementation

Find the Strata checkout and read its `docs/ACCEPTANCE.md` before relying on an
advanced capability. This skill ships in `skills/strata-research` inside that
checkout; if copied elsewhere, locate the installed checkout instead of assuming
that relative repository paths still work.

Read [the operational reference](references/workflow.md) when opening a line,
connecting an adapter, or interpreting a failed gate. The reference describes
the current API, not speculative product capabilities.

## Decisions that matter

- Prefer changes to an explicit producer–consumer–return boundary for L3/L4
  research. Include adaptation, routing, recomputation and HMR repayment in
  both correctness and pricing. A local saving is not a full-task saving.
- Pin the actual base and adapter. Historical profiles can guide questions but
  do not price a newly submitted circuit unless rebound to its bytes.
- Freeze the task's external mathematical promise. Builders may change internal
  representation; they may not quietly change the reference, allowed failures,
  phase tolerance or required restoration.
- Use the same parameterized generator at small and full scale. A flat gate file
  without source does not automatically supply a valid scale-down model.
- Run `lab`, `build`, and `verify` in order. Report a lab counterexample or missing
  prerequisite precisely. A timeout stops this run; it is not an impossibility
  proof. A component-only pass is not a certified design.
- Lab phase is an exact fraction of π modulo two. HMR phase errors matter even
  when output bits and scratch are correct. Dirty helpers must be restored for
  all their initial values.
- Price actual emitted builds under the declared distribution. Preserve fixed
  comparison draws. Do not promote a theoretical estimate or a lucky draw to
  a measured cost gain.
- Distinguish lower bounds, library constructions and small-size exact minima.
  Unknown floors and unavailable queries remain unknown. Primitive recognition
  does not count as high-level structural lifting.
- Preserve unsuccessful lines and scoped reopen conditions. Historical failures
  attach to their contracts and methods, not an entire research family.

## Structural search rounds

Use `ResearchRound` for a batch of deep hypotheses. Supply known producer,
consumer and return dependencies so widening includes all affected work. Prefer
distinct mathematical mechanisms before multiple variants of one mechanism.
Unknown savings remain eligible; paper estimates do not close a family.

A move named L3/L4 is not enough: qualification also needs changed emitted
bytes, changed contract structure, a passing artifact-bound task-adapter
structural witness, all declared small-size cases and full verification/replay.
Missing witnesses remain `STRUCTURAL_AUDIT_REQUIRED`. A replay failure
quarantines that design from `best` while retaining the failed attempt.
Read the checkout's `docs/STRUCTURAL-RESEARCH.md` for the current API and limits.

## Output

Provide the line/design handle, base and changed-artifact hashes, exact tested
scope, resource/cost delta, and certificate/replay status. State missing gates
and limitations. Use the repository acceptance ledger to avoid claiming that
an implemented API has achieved an unrun Kaggle research milestone.
