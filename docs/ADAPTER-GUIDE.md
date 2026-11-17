# Connecting a circuit or numeric task

Strata 0.2 supports three task paths: a trusted Python adapter, pinned typed
callbacks, or an existing artifact plus a verifier command in TOML. All use the
same Line/build/verify/replay pipeline. File discovery remains available without
any project adapter; see [the file workflow](FILE-FIRST.md).

## Artifact frontends and bundles

A task uses `adapter.frontend` when supplied, otherwise its TOML `frontend`
(default `kickmix`). A frontend implements `load(path)`, `parse(text)`,
`serialize(artifact)` and `measured(artifact)`. Serialization must round-trip
canonically. Source identity and task configuration are included in new
certificates. See [protocols.py](../src/strata/protocols.py).

`BundleFrontend` owns a declared set of components and serializes every byte:

```python
from strata.bundle import BundleFrontend, Component

frontend = BundleFrontend([
    Component('loader', 'loader.kmx', kind='kickmix', max_qubits=128),
    Component('coefficients', 'coefficients.npz', kind='data'),
    Component('published', 'published.bin', kind='opaque', frozen=True,
              sha256=published_sha256),
])
```

A bundle base is a directory containing exactly the declared files. Paths are
relative; traversal and symlinks are rejected. Circuit components currently use
kickmix text. Numeric schemas and acceptance mathematics belong to the task's
independent checker; `kind='data'` does not invent a numeric interpretation.
`DataFrontend` supports a single numeric/opaque file.

Frozen components are byte-pinned opaque boundaries. Every load, build and
replay checks their hashes; a patch cannot change them, including through a
`root` edit. Full-artifact oracle/verifier gates still receive the frozen files.

`Q` is reserved for emitted circuit width. For a sequential bundle it is the
maximum component width; `component_widths` reports each circuit. Use `Q_total`
for a host resource that includes preparation or other overhead, and include
that name in the objective. `Q_total` must cover emitted width. Task
`max_qubits` caps emitted width, and each circuit component can set its own
`max_qubits`. Simultaneously resident components need task-specific accounting;
the bundle frontend does not add their widths automatically.

## Trusted adapter responsibilities

| Hook | Input/result | Meaning |
|---|---|---|
| `build(parameters, implementation)` | `(artifact, Region)` | Reproduce the supplied base and generate the declared candidate |
| `reference(values, scale)` | `(expected_registers, phase)` | Independent exact function; phase uses fractions of π |
| `lab_domain(parameters)` | Optional iterable of register-value dictionaries | Explicit promised inputs for the native exact lab |
| `lab_components(parameters, artifact)` | Optional mapping to `ComponentLab` or `FullArtifactLab` | Evidence for every mutable bundle component |
| `lab(artifact, tree, parameters, domain, **limits)` | Optional typed lab result | Scoped external evidence; a raw success dictionary is rejected |
| `measure(artifact, parameters, domain)` | Resource dictionary | Measured complete-artifact resources |
| `oracle(artifact, parameters, domain)` | Receipt with `passed` and domain coverage | Full task-boundary checks |
| `verify(artifact, parameters, domain)` | `passed`, `resources`, `verifier_hash`, coverage | Unchanged independent task verifier |
| `ledger(artifact, tree)` / `floors(tree)` | Attribution / `Floor` records | Scoped measured resources and supported bounds |
| `constrain(parameters, implementation, constraint)` | Optional `(artifact, tree)` | A build that actually meets the requested constraint |

Register an instance with `strata.open(path, adapters={'my_adapter': adapter})`
and set `adapter = "my_adapter"` in the task. Python adapters execute trusted
code. Pin imported dependencies explicitly; a source hash alone cannot establish
verifier independence or correctness.

For scaled bundles, `lab_components` must cover every nonfrozen component.
`ComponentLab(contract, reference, domain=None)` runs against that component's
actual bytes. Frozen components receive hash evidence. Numeric components can
supply typed full-artifact receipts. The aggregate is labeled
`component_scoped`; component proofs do not establish cross-component behavior.

## Exact and promised-input labs

Every emitted wire needs a register owner. `clean` wires start and end at zero;
`dirty` wires are initialized over the declared domain and restored regardless
of reference output. Return original values explicitly for public registers
that must remain unchanged.

```python
from strata.lab import exhaustive

result = exhaustive(circuit, contract, reference, scale={'bits': 2},
                    domain=[{'address': 0, 'word': 3},
                            {'address': 0, 'word': 5}],
                    batch_size=4096, max_trajectories=1_000_000)
```

Each promised row names all nonclean registers; an optional clean value must be
zero. Inputs are deduplicated and canonically hashed. The promise string,
domain hash, artifact, contract and scale are bound into evidence. An explicit
subset cannot weaken `promise='all'`. The lab still requires exact phase and
no allowed failures; approximate acceptance belongs in external task checks.

`R` is allowed only when its active target is zero on every checked input and
branch. Native deterministic instruction runs are cached and batched. For the
supported affine HMR dependency class, the compiler proves that zero and the
unit outcome assignments suffice. Otherwise the lab enumerates all branches
or refuses the budget. `branch_strategy='full'` forces enumeration for audits.

Read receipts literally:

- `branches` / `represented_branches`: all HMR outcome assignments represented.
- `executed_branches`: assignments actually simulated.
- `tested`: actual input/branch checks, not the represented trajectory count.
- `reduction`: `affine_hmr` or `none`.
- `exact_small`: only the explicit scale; `exact_full`: the declared finite input
  domain of this fixed artifact, not a theorem for arbitrary sizes.

Deterministic shards use `shard_index` / `shard_count`, are labeled `sample`,
and cannot certify a design individually. No automatic shard aggregation or
invented small model is supplied. `line.lab(limits={...}, bits=2)` accepts the
native lab budgets and strategy.

## Configuration-only tasks

An existing circuit, data file or bundle can be checked without a Python
adapter. A minimal task uses an already available verifier executable:

```toml
[project]
store = ".strata"
[score]
formula = "geomean(task.base_cost / task.cost)"
[[task]]
id = "numeric"
base = "base"
frontend = "bundle"
cost = "Q_total * rank"
max_qubits = 128
[task.domain]
kind = "finite numerical contract"
[task.accept]
error = "<= 0.02"
[[task.components]]
id = "loader"
path = "loader.kmx"
kind = "kickmix"
max_qubits = 128
[[task.components]]
id = "numbers"
path = "numbers.json"
kind = "data"
[task.verifier]
command = ["/absolute/path/verifier", "{request}"]
sha256 = "REPLACE_WITH_EXECUTABLE_SHA256"
timeout = 60
```

Hash placeholders must be replaced. For interpreter-based commands, pin the
interpreter in `sha256` and each script/data dependency under
`[task.verifier.dependencies]` as `"/absolute/path" = "SHA256"`. Commands are
argument arrays, never shell strings. The offline Bubblewrap runner is required
and fails closed when unavailable; Python callback execution is a separate,
trusted extension path.

The command receives a JSON request path and emits one `CheckReceipt` JSON
object to stdout. Requests identify the stage (`lab`, `build_price`,
`oracle_full`, `official_verifier`), artifact path/hash, complete file manifest,
parameters, domain and scope. Receipts include `stage`, `passed`,
`artifact_hash`, `parameters_hash`, `domain_hash`, `scope`, `resources` and
`details`; scope and hashes must match the request. See
[the typed definitions](../src/strata/callbacks.py) and
[executable fixture](../tests/test_task_issues.py). Pinned dependencies are staged
inside the sandbox work area and checked for modification afterward.

The config-only path has no scalable generator: do not declare `task.scale`.
Its lab evidence is `full_artifact`, with `exhaustive=False` and small-model
availability explicitly recorded. A full verifier pass does not become a
small-instance exhaustive proof.

For a data-only or joint data/circuit edit, open a `co-design-numerics` Line with
`implementation={'files': {'numbers.json': '/path/new.json'}}` and declare the
affected component IDs. Replacement bytes freeze when the Line opens. Lab,
build-price, full verification and replay cover the resulting whole artifact.

## Pinned callbacks

`CallbackAdapter(frontend=..., build=..., lab=..., measure=..., oracle=...,
verify=...)` accepts `PinnedCallback` wrappers. A build callback receives
`BuildRequest` and returns `BuildReceipt(artifact, tree)`. Check callbacks receive
`CheckRequest` and return `CheckReceipt.for_request(request, passed=...,
resources=..., details=...)`. List dependency paths in
`PinnedCallback(callback, dependencies=[...])`; changes invalidate identity.
These adapters work with ordinary Lines and `ResearchRound`. They do not bypass
scope checks, and returning `{'passed': True}` is insufficient.

## Costs and stochastic draws

Cost expressions use restricted arithmetic. Custom adapters may supply
`cost_functions`; no default contest penalty is invented. For a sampled policy,
commit at least two distinct seeds and the comparison rule before evaluation.
`ProbabilityPolicy` supports weighted expected/fixed-seed averages and worst-case
comparison, a failure budget and an optional `R` penalty expression. It averages
per-draw objective values, which matters for nonlinear costs. Receipts carry one
ordered draw per seed. Missing draws, changed seeds and candidate-selected
favorable draws are rejected. Finite-draw evidence is labeled an estimate of the
population expectation. See [probability.py](../src/strata/probability.py).

Enable the same policy in project pricing and replay with a task section:

```toml
[task.probability]
seeds = [17, 29]
compare = "expected"
failure_resource = "fail_rate"
max_failure = 0.1
penalty = "1/(1-fail_rate)"
```

Use the same seeds in the declared domain/oracle evidence. A penalty is applied
when the task objective calls `R(fail_rate)`. Every build-price and official
receipt must include the ordered `draws`, each with `seed` and `resources`, and
the precommitted `seeds`; typed command/callback receipts may carry these under
`details`. A custom adapter's `measure` hook retains its complete draw evidence
as `measurement_receipt`. The project stores and rechecks the policy and
per-draw pricing on certification/replay; changed policy configuration
invalidates the saved task binding.

## Floors, queries and research memory

For source-less files, `inspect_file` and `optimize_file` retain
`region-evidence.json` and a `file_region_evidence` store record. Results expose
`region_evidence`, `region_evidence_record` and `evidence_summary`. Supported
regions have `evidence_kind='symbolic'` with exact Boolean outputs and sign phase
for all arbitrary boundary bits; unsupported regions remain `unknown`.

```python
from strata.region_evidence import replay_region_evidence

replayed = replay_region_evidence('work/inspection/.strata', evidence_handle)
```

This explicit replay reloads pinned source bytes and recomputes the derivation
using the same algebraic backend. It checks the contract tree, physical wire
ownership, maps, span/source hashes, assumptions, budgets and backend identity.
It is not an independent implementation of the proof. These receipts describe a
fixed supplied region, with no inferred clean inputs, scale model, whole-circuit
certificate or task correctness. Recording reuses an existing lift; ordinary
store-only research reporting does not perform this symbolic replay.

`strata.floors.counting_bound` derives nonlinear-rank/degree bounds;
`exact_minimum` exhausts a bounded free-XOR/NOT, two-input-AND model. An exact
minimum has `exact_small` scope; exhausted budgets return `unknown`. These bounds
do not automatically price Toffolis, HMR or the full host objective. Library
constructions remain upper bounds.

`strata.inference.is_function_of` searches an explicit polynomial family in
current support values and returns a witness, scoped `no`, or budget/unsupported
`unknown`. `information_slack` uses exact affine rank or bounded distinct-state
enumeration. Information compression does not establish an available clean
scratch wire or a cheap encoder. High-level stage recovery remains partial.

For retained designs, `Queries(project).is_function_of(wire, at, support,
design=design)`, `.slack(design, at, wires=...)`, and `.holders(design, peak)`
read the saved circuit and use its declared clean inputs. Holders describe
inferred wire contents and input-register dependencies; dynamic allocator
ownership remains unknown without allocation events. The CLI `function-of`,
`slack` and `floors` commands instead accept a circuit file directly.

Direct negative writes and imports share an artifact check:

```python
artifact = project.store.blob(check_output_bytes)
project.store.negative(contract_fingerprint, 'fuse', 'build', result,
                       'reopen after representation change', evidence=artifact)
line.park(result, 'new premise', evidence='proof', evidence_artifact=artifact)
```

`proof`, `build` and `exact_small` without a valid stored blob become `estimate`
/ `priced`, preserving `requested_evidence_kind`; exact-small also needs a size.
A hash establishes retained evidence identity, not independent mathematical
validity. Build failures concern the recorded construction. Use
`import_attempts(store, path, contract_map=None)` for JSONL, CSV or JSON notes and
`apply_memory(store, design_handle)` to see matching evidence and reopening
conditions on a new base. Unknown bindings stay explicit.
