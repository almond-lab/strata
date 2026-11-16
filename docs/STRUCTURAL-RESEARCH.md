# A workflow for deep restructuring

Strata now has a research-round selector and executor. It chooses among
explicit mathematical hypotheses, widens their scope to known consumers and
cleanup, and runs supplied implementations through the existing verification
pipeline. It does not invent an efficient algorithm or certify a physical
factorization without the task's acceptance checks.

## What changed

1. **Start from a challenged architectural assumption.** A hypothesis names
   its mechanism, the assumption it changes, the new boundary/stage behavior,
   and a decisive experiment. An implementation can be missing; that remains
   a derivation task rather than a claimed candidate.
2. **Include the complete episode.** Data dependencies and phase debts widen
   the selected regions in both directions. Intervening stages are included,
   then their dependencies are followed too. Unknown source dependencies still
   need a source audit; this is not automatic semantic recovery.
3. **Explore different mechanisms.** A round gives each mechanism a first slot
   before allocating second variants. Budgets cap proposals, lines and variants
   per mechanism. Deferral is not scientific rejection. Identical generator
   requests are deduplicated even if the prose or move label differs.
4. **Keep uncertain ideas alive.** Prior attempts are attached to the plan.
   Expensive estimates never suppress or close a whole family. Unimplemented
   questions and missing bounds remain explicit; numbers in `estimate` are
   never used as measured gains.
5. **Check depth after building.** Changed names, move labels and primitive
   coverage do not establish restructuring. Qualification needs changed bytes,
   changed internal contracts/topology/debt ownership, and a passing structural
   witness bound to both artifacts. A local rewrite plus regrouped labels is
   insufficient.
6. **Run every configured small instance.** The round exhausts the Cartesian
   product of declared small scales before building at full size. Exceeding
   the suite/trajectory budget produces no partial qualification. Individual
   line APIs remain available for exploratory tests.
7. **Measure and retain.** Full build, oracle, verifier and replay follow. A
   structural design that is correct but expensive is retained as a sibling.
   A replay failure quarantines it from `best`; failed evidence remains.

## Runnable example

Use a freshly initialized example project. After an adapter source change, use
a fresh project/store for this example; previous certificates remain bound to
the previous source. Within one source revision, the same project/store preserves
prior certified variants.

```bash
.venv/bin/strata --project examples/and_fusion/strata.toml init
.venv/bin/strata --project examples/and_fusion/strata.toml research and
.venv/bin/strata --project examples/and_fusion/strata.toml research and --run
```

The built-in adapter proposes eliminating the materialized product word. Its
independent, narrow structural checker recognizes the actual old
compute/consume/uncompute pattern, the new direct dirty-accumulator operation,
and removal of the scratch word. It reports 72→27 measured cost, after all small
sizes, 512 full inputs and replay. This is an original demonstration, not a new
Kaggle optimization or blind discovery of an unknown algorithm.

See the [captured CLI receipt](STRUCTURAL-ROUND-RECEIPT.json) and
[independent review](RESEARCH-REVIEW.md). The receipt also checks that the next
round uses the retained improved design.

For explicit questions and known dependencies:

```python
from strata import ResearchRound, ResearchPolicy, Hypothesis, Dependency

round_ = ResearchRound(
    p, base,
    policy=ResearchPolicy(max_lines=4, max_per_mechanism=1),
    dependencies=[
        Dependency('produce', 'consume', 'value handoff', 'source audit receipt'),
        Dependency('produce', 'return', 'cleanup ownership', 'source audit receipt'),
    ],
)
proposal = Hypothesis(
    move='re-represent', regions=('produce',), mechanism='state-representation',
    assumption='The consumer requires a materialized coefficient vector',
    change='Generate and consume evaluation values in a shared representation',
    decisive_test='Check complete small-field forward, consumer and return',
    implementation=None,
    unknowns=('zero branches', 'conversion overhead', 'peak live state'),
)
plan = round_.plan([proposal])
result = round_.run(plan['handle'])  # DERIVATION_REQUIRED, not a circuit result
```

These region names illustrate the API; use the actual task's frozen regions.
With `implementation` supplied by the generator author, the round executes it.
If replacement regions have different names, declare `replacement_regions`.
The CLI also accepts a JSON list of hypothesis fields via `--hypotheses FILE`.
By default it starts from the task's best certified, non-quarantined design so
successive rounds can compound gains. Use `--parent DESIGN_ID` to explore a
different retained branch explicitly. An adapter source change requires a new
certification before a round can execute using that source.

## Adapter structural witnesses

An optional `restructures(tree)` method yields task-specific hypotheses.
Without it, Strata generates architecture, adjacent fusion and representation
questions with **no invented implementation**.

To qualify a deep build, implement:

```python
def audit_restructure(self, before, old_tree, after, new_tree):
    return {
        'passed': ...,  # actual checks, not a constant assertion
        'mechanism': ...,
        'checks': {...},  # named checks, each exactly True on success
        'before_artifact': ...,  # SHA256 of str(before).encode()
        'after_artifact': ...,
    }
```

This adapter is inside the explicit trust boundary. The core checks its binding
and result, but cannot infer what arbitrary internal values mean. Use an exact
pattern check, independently derived representation relation, lifetime witness,
or another source-bound check appropriate to the mechanism. The full external
function remains protected by the lab/oracle/verifier. A witness is neither a
novelty certificate nor a universal optimality proof.

## Read the outcome honestly

| Status | Meaning |
|---|---|
| `DERIVATION_REQUIRED` | No implementation supplied; question remains open |
| `BUILT_NONSTRUCTURAL` | The build lacks a demonstrated deep structural change |
| `STRUCTURAL_AUDIT_REQUIRED` | Tree/bytes changed, but the adapter has no structural witness |
| `ATTEMPT_FAILED` | This run failed a gate or budget; inspect its line events |
| `VERIFIED_STRUCTURAL_NO_GAIN` | Correct/replayed deep variant, but cost did not improve |
| `VERIFIED_STRUCTURAL_GAIN` | Structural witness and full verification/replay pass, measured cost falls |
| `VERIFIED_LOCAL_GAIN/NO_GAIN` | Optional local experiment; never counted as a deep gain |

Reports bind the immutable base price and actual verified new price. The
portfolio multiplier assumes every other task is fixed and uses the project's
number of tasks; it is not an independently measured leaderboard score.

The executor retains all lab receipts in its research result. Build/verifier
callbacks are trusted Python extensions: round counts and lab timeouts are not
hard subprocess timeouts for arbitrary callbacks. Use the sandbox's time and
resource limits for external jobs. Known dependency closure does not prove that
all source dependencies were supplied.

## Decoder and chemistry priorities

For the decoder, supply complete extraction/restoration or conversion/payload
relationships, zero branches and HMR ownership before narrowing an episode.
For chemistry, include the loader bundle, host preparation and physical
factorization acceptance; a cheap lookup with invalid physics is not a winner.
Both integrations still need the production adapters described in
[FIRST-REAL-TARGETS.md](FIRST-REAL-TARGETS.md). The research runner does not turn
those pending integrations into completed capabilities.
