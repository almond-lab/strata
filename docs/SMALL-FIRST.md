# Small-first research

Strata agent packets default to this research ladder:

1. **`tiny_exact`**: check the proposed mechanism on a tiny exact instance. Aim
   below one second when feasible; retain a reason for a longer budget or run.
2. **`scaling`**: estimate full wall time, threads, memory, storage and relevant
   objective cost from the retained small run. State the scaling model and its
   assumptions; this is model evidence.
3. **`full`**: build one bounded full candidate after the estimate fits its
   reservation and decision budget.
4. **`final`**: select independently accepted candidate bytes.
5. **`polish` / `multi_seed`**: reserve slow refinement and repeated independent
   verification for the selected candidate. Precommit verification draws.

Tiny exactness concerns the checked instance. It does not prove that the method
scales, that the full artifact is correct, or that a cost estimate is measured.
Full verification still uses the task's independent native or host checker. Its
implementation may differ from the tiny exact oracle.

Ordinary wrapping, `experiment run` and existing callers continue to work.
Strata does not infer the size or purpose of opaque scripts. New agent decisions
declare `research_stage`; an omitted stage is explicitly marked
`advisory_legacy`. This is a preflight on agent decisions, not an operating-system
resource limit or a replacement for the experiment board.

## Retain a tiny exact check

The current generic bridge supports positive integer scale settings, unchanged
non-scale settings, and the same frozen generator, worker inputs, environment
and boundary contract. Worker commands, source bytes and runtime identities are
checked; a receipt for an unrelated generator cannot unlock a full build. Tiny
and full checker implementations may differ. There is no inferred mapping
between unrelated prototypes or automatic reconstruction of a scaled generator.

Create the tiny and target experiments from the parameterized generator, then
retain a successful tiny finish. Its independently accepted checker must consume
the candidate artifact and emit this structured evidence in its JSON receipt:

```json
{
  "accepted": true,
  "exact_check": {
    "method": "exhaustive",
    "passed": true,
    "complete": true,
    "expected_cases": 16,
    "checked_cases": 16,
    "artifact_sha256": "ACTUAL_SHA256_OF_CHECKED_ARTIFACT",
    "scale": {"width": 4}
  }
}
```

This trusts the frozen independent checker for its declared exhaustive case
coverage, as ordinary acceptance trusts that checker. A bare `passed` flag,
partial count, mismatched artifact or mismatched scale is insufficient.

Alternatively, pass `proof=EXACT_LAB_RECORD` containing a complete native exact
lab receipt validated by `validate_lab_receipt`. Its circuit hash and scale must
match the artifact and tiny settings; declare its exact reference contract hash
as `manifest.contract.tiny_contract_hash`. This pins the tiny reference rather
than treating equivalence to any arbitrary reference as success. Retain the
receipt with `store.put('exact_lab_receipt', asdict(lab_result))`.

```python
from strata.research_ladder import record_tiny_check, record_resource_prediction

tiny = record_tiny_check(
    store, result=tiny_finish, target_experiment=full_experiment,
    scale={"width": 4}, artifact="candidate",
)
prediction = record_resource_prediction(
    store, target_experiment=full_experiment, tiny_receipt=tiny,
    estimates={"wall_seconds": 12, "threads": 1,
               "memory_bytes": 134217728, "storage_bytes": 10485760},
    model="Quadratic operation growth plus measured fixed setup overhead.",
    assumptions=["The same generator applies at width 64.",
                 "The declared memory representation is unchanged."],
)
```

The tiny finish supplies measured wall seconds. If it took at least one second,
provide `runtime_justification` to `record_tiny_check`. When proposing an explicit
`tiny_exact` decision, give `budget.wall_seconds`; a budget of at least one second
requires `research_evidence.runtime_justification` as well.

The resource predictor records an explanation, estimates and evidence handles;
it does not automatically fit a curve or prove an asymptotic bound. Put the cost
formula, units, dominant operations and expected scaling in the model/assumptions
when objective cost differs from compute resources.

## Agent decision preflight

Add these fields to the ordinary `agent-decision` JSON:

```json
{
  "research_stage": "full",
  "research_evidence": {
    "tiny_receipt": "TINY_CHECK_HANDLE",
    "resource_prediction": "PREDICTION_HANDLE"
  },
  "budget": {
    "wall_seconds": 15,
    "threads": 1,
    "memory_bytes": 134217728,
    "storage_bytes": 10485760
  }
}
```

Keep the usual packet, hypothesis, decisive experiment, expected outcome and
reopening condition fields. The prediction and tiny record must name that exact
frozen target experiment. All four resource predictions must fit both its
manifest reservation and the decision budget. Later contradictory scientific
receipts invalidate the preflight; an audited infrastructure correction retains
the original and allows a properly checked replacement.

For `final`, `polish` or `multi_seed`, supply
`research_evidence.final_receipt` with an independently accepted finish and an
optional `artifact` name/hash. A new experiment must contain those exact
candidate bytes in its frozen inputs. The same experiment may legitimately be
revisited; this does not grant another execution budget. A final receipt from a
draw-policy experiment requires the complete consistent draw set. `multi_seed`
also requires the proposed experiment's precommitted `draw_policy`.

Final selection from a draw-policy run currently requires a reportable numeric
objective bound to its independent verifier. Boolean-acceptance-only draw plans
have no supported aggregate for this preflight and remain blocked, even when
their individual checks passed. Single accepted receipts without a draw policy
do not need a numeric objective. This limit does not change experiment execution
or the scope of any retained acceptance receipt.

## When an exact tiny instance is unavailable

Use `research_exception` with `kind: "nonparameterizable"` or `"no_exact_tiny"`,
a concrete `reason`, and a nonempty `alternative_evidence` list of retained
receipts. For example, a fixed numeric dataset may permit only a representative
empirical screen. Keep that limitation explicit.

Record the resource prediction with `basis=alternative_evidence` and no
`tiny_receipt`. Supply that prediction when proposing `full`. The exception
waives the exact tiny prerequisite only; it does not waive the resource check,
authorize scientific acceptance, or imply that an empirical screen was exact.
