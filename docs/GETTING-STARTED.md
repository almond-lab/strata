# Getting started with Strata

Start with a stage of your generator. Choose the value its caller requires and the resource saving you need. Use the [generator workflow](GENERATOR-WORKFLOW.md) to compare source implementations. Existing scripts and numeric fits can use [experiment runs and sweeps](TEN-MINUTES.md). See the [acceptance ledger](ACCEPTANCE.md) for the scope of the available checks.

## Install

Python 3.14 is tested; the package requires Python 3.11+ and a compatible kickmix
wheel. From a private checkout:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

The examples assume the repository root and use an explicit environment path.

## Run a tiny generator comparison

```bash
.venv/bin/python -m strata.examples.field_product
```

The two generators implement the same field product using different algebra. See [the example source](../src/strata/examples/field_product.py) to change an implementation and recheck the contract.

## Give the connected agent a compiled circuit

```bash
.venv/bin/strata inspect-file /absolute/path/circuit.kmb --out work/inspection
.venv/bin/strata discover /absolute/path/circuit.kmb --out work/research
```

Inspection and automatic optimization also save `region-evidence.json` with
fixed-region symbolic receipts, opaque-region unknowns and a store handle.
Their result includes `region_evidence_record` and counts in `evidence_summary`.
The [adapter guide](ADAPTER-GUIDE.md#floors-queries-and-research-memory) explains
explicit symbolic replay and its scope. No small generator or clean-input
assumption is inferred from these receipts.

The agent follows [the repository skill](../skills/strata-research/SKILL.md),
reads `work/research/agent-brief.json`, originates hypotheses and builds a
candidate when justified. Submit a candidate and optional hypothesis JSON:

```bash
.venv/bin/strata evaluate-file /absolute/path/circuit.kmb /absolute/path/candidate.kmx \
  --out work/research --hypothesis hypothesis.json
```

When no candidate is justified, retain a research note instead:

```json
{
  "note_class": "prebuild_blocker",
  "hypotheses": ["Change the producer representation and adapt its consumer"],
  "blockers": ["The complete return boundary is not yet established"],
  "next_tests": ["Recover the consumer and cleanup obligations"]
}
```

```bash
.venv/bin/strata note /absolute/path/circuit.kmb --out work/research --note note.json
```

A note is reported reasoning, not proof or an accepted gain. Candidate records
include exact-boundary checks where supported, counterexamples, diagnostic cost
deltas and reopening conditions. A lower measured cost with failed equivalence
is not a saving. `discover` updates the next brief with this history. A standalone
CLI invocation does not run an AI provider. See [the discovery loop](AGENT-DISCOVERY.md).

The second input codec is available through `strata.frontends.OperationStream`:
JSONL records such as `{"op":"CCX","qubits":[0,1,2]}` use the same native
execution backend. Its supported operation set is explicit; it is not a second
simulator or an ecdsa.fail-specific integration.

## Read a report without rerunning work

Reports need only the retained `.strata` store, including for the file workflow:

```bash
.venv/bin/strata report --store work/research/.strata --format markdown > work/research/insights.md
.venv/bin/strata report --store work/research/.strata --format json > work/research/insights.json
```

Add `--handle HANDLE` to select a saved round, design, Line or file record. The
Python API also exposes named design/round selectors:

```python
from pathlib import Path
import json
from strata.store import Store
from strata.reports import report, markdown_report

store = Store('work/research/.strata')
saved = report(store)  # Also accepts handle=..., design=..., or round=...
Path('work/research/insights.json').write_text(json.dumps(saved, indent=2))
Path('work/research/insights.md').write_text(markdown_report(saved))
```

The report includes plans/results, Lines and mechanisms, measured deltas,
evidence scopes, file attempts, notes, shared failures and reopening conditions.
Stored gap snapshots are shown per design; a missing snapshot stays unknown.
Reporting neither loads a project configuration nor calls generators/verifiers.
It can still summarize work after original source files have disappeared.

## Run the original generator example

The fixture computes `y ^= a & b` while preserving `a`, `b` and arbitrary initial
accumulator values. The base materializes products into scratch; the candidate
consumes them directly. At three bits, resources change from 12Q/6 Toffolis to
9Q/3 Toffolis, and cost from 72 to 27.

```bash
.venv/bin/strata --project examples/and_fusion/strata.toml init
.venv/bin/strata --project examples/and_fusion/strata.toml analyze and
.venv/bin/strata --project examples/and_fusion/strata.toml gaps and
```

`init` reproduces the supplied base and runs lab, measured pricing and full
gates. `analyze` writes kickmix views under the project's `.strata/analysis`.
`gaps` saves a snapshot as well as returning current bounds.

```python
import strata

p = strata.open('examples/and_fusion/strata.toml')
p.init()
base = p.task('and').base_design()
line = p.open_line(base, 'fuse', ['lane-0', 'lane-1', 'lane-2'],
                   implementation={'fused': True},
                   note='Consume products directly; remove scratch')
lab = line.lab(bits=2)
if not lab.passed:
    raise RuntimeError(lab.counterexample)
price = line.build()
design = line.verify()
print('Measured cost delta:', price['delta'])
print('Design:', design.id)
print('Replay:', p.replay(design.id)['passed'])
print('Portfolio:', p.best())
```

The two-bit lab covers 64 inputs; the full fixture verifier covers 512 at three
bits. The verifier is explicitly a demo verifier. This example is not a new
competition result.

## Understand evidence labels

| Evidence | What it establishes |
|---|---|
| `exact_small` | Complete declared inputs/branches at an explicit finite scale |
| `exact_full` | Complete declared finite input domain for this fixed artifact |
| `affine_hmr` reduction | A proved supported dependency class allows fewer executed branch witnesses; receipts retain both counts |
| `sample` shard | Partial domain checks; cannot certify a design alone |
| `full_artifact` | A scoped pinned external check; no automatic claim of exhaustive small-model coverage |
| `component_scoped` | Component evidence plus frozen-byte obligations; full cross-component checks still required |
| Measured build price | Actual complete-artifact resources under the declared domain |
| Certificate/replay | Bound evidence and reproducibility; not a general theorem or signed attestation |

For HMR, `tested = cases × executed_branches`; `represented_trajectories` may
be larger when the supported reduction applies. No receipt should call the
represented trajectories executed simulations. Checked resets may clear only
already-zero targets. A promised input set is bound by its canonical domain hash.

## Import existing research memory

```bash
.venv/bin/strata memory-import attempts.csv --store work/research/.strata
.venv/bin/strata memory-status --store work/research/.strata
.venv/bin/strata memory-status --store work/research/.strata --contract FINGERPRINT --scale '{"bits":2}'
```

The CLI importer uses bindings already present in the input. Use Python to add a
binding map or apply evidence to a saved design:

```python
from strata.memory import import_attempts, apply_memory

counts = import_attempts(p.store, 'attempts.csv',
                        contract_map={'attempt-17': base.tree.contract.fingerprint})
application = apply_memory(p.store, base.id)
```

JSONL, CSV and JSON arrays/wrapped notes are accepted. Explicit stored
`contract_fingerprint` fields can supply a binding; the mapping overrides them.
Unbound rows remain explicit. Narrative “proof” or “closed” labels cannot close
a family. Strong evidence needs a valid blob in this store, and exact-small
needs a size. `apply_memory` reports scope matches and reopening after contract
or scale changes without rewriting the original records.

To record strong evidence directly, store its bytes first and pass the returned
hash as `Store.negative(..., evidence=hash)` or
`Line.park(..., evidence='build', evidence_artifact=hash)`. The hash confirms
content identity, not the correctness of its interpretation.

## Ask bounded structural questions

These commands load a circuit without a project or generator:

```bash
.venv/bin/strata function-of producer.kmx --wire 2 --at 1 --support 0,1 --clean-inputs 2
.venv/bin/strata slack producer.kmx --at 1 --wires 0,1,2 --clean-inputs 2
.venv/bin/strata floors producer.kmx
```

Here `--at 1` means after the first normalized instruction. Clean inputs are an
explicit assumption; declare them only when the task contract establishes zero.
Functional queries search a stated polynomial family; a `no` is limited to that
family. Slack measures compressible information, not a scratch lease. The CLI
floor uses all final outputs and arbitrary initial inputs, and reports a counting
bound in a free-XOR/NOT, two-input-AND model. It does not accept clean-input
assumptions or claim a measurement/phase/host-cost lower bound. Unsupported
inference returns unknown. Python `exact_minimum` additionally supports bounded
exact-small synthesis.

## Connect your own task

The [adapter guide](ADAPTER-GUIDE.md) covers:

- Config-only tasks with a pinned verifier command and no Python adapter.
- Circuit/data bundles with per-component width caps and frozen components.
- Trusted typed callbacks and component labs.
- Explicit promised domains, probability policies, floors and functional queries.

A source-less artifact can use full-artifact checks without pretending it has a
small generator. Use `Q` for emitted width and `Q_total` for separate host
accounting. Keep verifier code and retained bases unchanged; large/private
artifacts belong outside this repository. Strata has no automatic competition
submission command.
