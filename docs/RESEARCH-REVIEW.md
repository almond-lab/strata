# Independent structural-round review

This bounded review examines the new research-round selector and executor,
using original tiny fixtures and real kickmix. It does not use contest circuits,
Kaggle files, private verifiers, or historical outcome labels. The purpose is to
check whether a retained result actually clears the declared structural and
verification gates, rather than acquiring a deep label from its proposed move.

`tests/test_research_review.py` contains **15 independent regression cases**.
The final focused run passed all 15 in 0.46 seconds:

```bash
.venv/bin/python -m pytest -q tests/test_research_review.py
```

## Findings and corrections

| Finding | Minimal case | Corrected behavior |
| --- | --- | --- |
| Widening could omit a newly included stage's dependency | Four siblings `a,b,c,d`; `a→c` includes `b`, and `b→d` must also include `d`. | Dependency and intervening-span closure iterate together. |
| Local gate cancellation could acquire a deep label | Remove a cancelling CCX pair while erasing the stage tags. The function and meaningful boundary representation stay the same. | Topology changes are candidates; a passing artifact-bound structural witness is required for a qualified deep gain. |
| Replay failure left a failed candidate eligible for `best` | Full verification succeeds, then replay raises an emitted-byte mismatch. | Preserve the evidence and quarantine the candidate from best-design selection. |
| Caller mutation could invent a measured improvement | After creating a plan, set the caller's in-memory base cost to `10**30`. | Reload and validate the immutable base; report certificate-bound cost 72, measured new cost 27, and delta −45. |
| Allowing local experiments mislabeled their outcomes | Set `deep_only=False` and run the local cancellation above. | Local outcome classification remains local and does not increase `qualified_gains`. |

The focused suite also checks that old cost estimates retain their `priced`
status without suppressing a family; selection gives distinct mechanisms a
first slot before returning to one mechanism; mutable hypothesis dictionaries
and returned plan dictionaries cannot rewrite a stored plan; a plan cannot run
on a different certified base; names, kinds, and unchanged artifact bytes do
not establish depth; phase debts close in both directions; and the widened
episode retains the task's external specification.

The scale test uses two independent scale parameters and observes all four
configured small combinations before the full build. A counterexample at the
second small size prevents certification. A suite exceeding its declared
instance budget refuses partial qualification. The positive restructuring
uses the complete producer, consumer, and return transformation and checks
measured costs, the fixed-other-task score multiplier, full verification, and
replay. Its deliberately unrealistic estimate never enters the measured gain.

## Boundaries of the evidence

The adapter supplies the structural witness. The built-in AND example checks
the exact old materialization/consume/uncompute instruction pattern, the new
direct accumulator pattern, and elimination of the product word. Witnesses
bind both emitted artifact hashes, and full-function checks remain mandatory.
This is a narrow executable witness from trusted adapter code. It is not a
general semantic classifier, novelty proof, all-size proof, or independent
audit of arbitrary internal contract assertions.

Closure covers declared dependencies and debts. The review does not establish
that missing source dependencies are inferred. A scientific or competitive
claim still requires the appropriate task adapter, complete return boundary,
cost distribution, and verifier authority. A failed construction leaves its
mechanism family open. Replay quarantine retains immutable evidence of the
earlier verification rather than deleting history.

The experiment and lab budgets limit selected lines, small-scale combinations,
trajectories, and each exhaustive lab run. They are not a subprocess timeout
for trusted Python generator or verifier callbacks. No automatic synthesis or
contest acceptance milestone is established by these tests.

## Planning scalability observation

During review, a single closure on a flat identity tree with a sequential
dependency chain took 0.0235 seconds at 32 stages, 0.2121 seconds at 64, and
1.703 seconds at 128 on this environment. The then-current implementation
repeated whole-tree lookups inside coverage checks. This demonstrates a
planning bottleneck before line-budget selection; it is not a full contest
benchmark. The measurements were reported to the implementation author.
No timing threshold is included in the regression suite because such a test
would be machine-dependent. Large dependency graphs need separate profiling
after optimization.

### Implementation follow-up

After the review, indexed regions and cached coverage replaced repeated tree
searches. A local smoke on the same sequential-chain fixture measured about
0.0006 s / 0.0013 s / 0.0030 s at 32/64/128 stages and 0.018 s at 512.
These are implementation-author measurements, not an independent performance
acceptance result. A proposal-enumeration budget also limits each round.
