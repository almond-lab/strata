# Independent alpha review

This bounded review used original, public-safe fixtures and the checked-in specification. It did not use the historical contest circuits, private verifier bundles, or recorded contest draws. Passing these tests establishes the behavior exercised here; it does not establish completion of specification acceptance tests R1–R6.

## Regression findings

The review added `tests/test_review.py`. Each issue below was reproduced before correction; production changes belong to the implementation author.

| Issue | Minimal counterexample | Required behavior |
| --- | --- | --- |
| Fractional phase silently lost | The identity contract accepted `Z_POW q0 0.25` because the simulator's exact rational phase was cast to an integer. | Compare exact native phases in units of pi, modulo two; retain rational values in counterexamples. |
| Mutable patch detached from evidence | Build a fused implementation, mutate its patch dictionary to unfused, then verify. | Reject a changed patch or evidence from a different implementation. |
| Uncertified cost wins ranking | Store a copied valid design with cost 1 and its original cost-72 certificate. | `best` and `replay` must reject mismatched cost evidence. |
| Failed lab promoted by certificate wrapper | Copy a certificate, set its lab pass and exhaustive fields false, retain the outer pass flag. | Check the component receipts, not only the outer flag. |
| Scope bypass through new uncovered operations | Append two cancelling X gates outside a lane named by the patch, retaining all existing child slices. | Account for the entire emitted instruction stream when checking edit scope. |
| Timeout omitted reference evaluation | An identity reference sleeps past the budget after the last simulated operation. | Exceeding the lab time budget cannot produce an exhaustive pass. |
| Replay omitted ancestry | Copy a certified child with a nonexistent parent hash. | Reject an unavailable or invalid parent and validate the patch chain. |
| Legitimate L3 fusion rejected by scope guard | Fuse several selected children into one newly named child. | Explicit `replacement_regions` supports new child identities while preserving the selected region scope. |
| L2 edit changes the contract | A `re-synthesise` edit changes the targeted lane's reference identifier while preserving the root. | L2 redesigns preserve their targeted contracts. |

The phase regressions also require a correct one-quarter-pi reference to pass and a two-pi phase to compare equal to zero. These are positive checks of the phase convention, in addition to rejecting incorrect circuits. A separate positive HMR test exercises conditional measurements and a reused measurement bit over all four inputs and eight branch assignments.

## Validation

After the implementation fixes and the explicit replacement-region API, `.venv/bin/python -m pytest -q` passed all **37 tests**, including **15 independently added review cases**. This result is the local alpha regression check, not a completion claim for the historical acceptance suite.

## Review boundaries

The original suite already exercises all HMR branch outcomes for a measured AND cleanup, dirty helper restoration, clean leakage, fixed stochastic seeds, refusal of overlapping/dependency-conflicting patches, immutable blob integrity, and separation of library/small-size floors from full-size lower bounds. The review retained those checks.

The alpha has a real end-to-end AND-boundary fusion fixture. That fixture is deliberately separate from the historical discovery claims. Primitive instruction recognition is not structural arithmetic lifting, and finite exhaustive examples are not all-size proofs. The specification's historical replay targets, automatic high-level lifting targets, broad move search, domain-specific failure models, and second front end require separate evidence.

Python adapters are trusted extensions. Hash-pinned files and content-addressed records provide reproducibility and corruption detection, not signatures or protection against an adversarial Python extension that controls the reference, builder, and verifier. The subprocess helper requires bubblewrap and fails closed when it is unavailable; its read-only host-root mount is not a confidentiality boundary, and post-run output-size checks are not a disk quota. No sandbox isolation claim was independently verified in this review.
