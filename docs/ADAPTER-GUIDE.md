# Connecting a real circuit task

The adapter is the boundary between generic Strata machinery and your contest.
Its Python implementation is trusted code. A code hash records its identity;
it does not establish that its reference or verifier is correct.

Read the [TaskAdapter protocol](../src/strata/protocols.py) and the complete
[AndFusion example](../src/strata/demo.py) for the current executable interface.

## Required responsibilities

| Method | Input and result | Required meaning |
|---|---|---|
| `build(parameters, implementation)` | Returns `(kickmix.Circuit, Region)` | Same generator at small and full sizes; complete ownership contract |
| `reference(values, scale)` | Returns `(expected_registers, phase)` | Independent mathematical function; phase is an exact `Fraction` in units of π |
| `measure(circuit, parameters, domain)` | Resource dictionary | Executed costs under the declared distribution; include the entire candidate |
| `oracle(circuit, parameters, domain)` | Receipt with `passed`, domain/seed coverage | Full boundary values, phase and scratch, using the actual task domain |
| `verify(circuit, parameters, domain)` | `passed`, `resources`, `verifier_hash`, `failure_model`, `seeds` | Calls the unchanged task verifier and reports its measured result |
| `ledger(circuit, tree)` | Dictionary keyed by region ID | Measured resource attribution; shared peak width must not be added across regions |
| `floors(tree)` | List of `Floor` objects | Explicit kind, contract fingerprint, evidence and scale |
| `constrain(parameters, implementation, constraint)` | Optional constrained `(circuit, tree)` | Actually achieves the requested resource change; unavailable constraints raise an error |

Register a trusted adapter explicitly:

```python
p = strata.open('my-contest/strata.toml', adapters={'my_adapter': MyAdapter()})
```

Set `adapter = "my_adapter"` on the corresponding TOML task. No arbitrary Python
module is automatically imported from project configuration. Third-party
executables should run through the offline runner or an appropriately isolated
container; Python callbacks themselves are not sandboxed.

## Contract design

Declare the external function, input promise, phase requirement, failure rule,
and every register. `clean` registers start and finish at zero. `dirty`
registers are exhaustively initialized and must be restored independently of
what the reference returns. The current lab also enumerates initial values of
output accumulators. Every emitted wire must be owned by the contract.

`input`, `through`, and `output` values are checked against the reference. If an
input must be restored, return its original value explicitly. A reference must
not inspect which implementation is being tested.

The built-in exhaustive runner currently supports all-input exact contracts.
Approximate numeric domains, promised inputs and stochastic failure models need
additional domain adapters; declaring a tolerance in TOML alone does not create
that mathematics. Unsupported lab instructions and oversized domains fail
explicitly, rather than receiving a partial-pass certificate.

For L2 (`re-synthesise`, `trade-wires`) the target region contract is preserved.
L3/L4 edits may change internal contracts while retaining the task's external
function. When a fusion replaces old region names, declare the new scope:

```python
line = p.open_line(
    parent=base, move='fuse', regions=['producer', 'consumer', 'cleanup'],
    replacement_regions=['fused_episode'],
    implementation={'representation': 'direct'},
)
```

Operations outside the declared old and replacement spans must remain equal.
Use stable region IDs and occurrences. Byte offsets are not patch identity.
Do not hide consumer adaptation, recomputation, scratch or measurement debt
outside the region used to price a proposal.

## Costs and draws

The cost formula is restricted arithmetic, not Python `eval`. Use resource names
returned by the verifier. Supply contest-specific functions through an adapter's
`cost_functions` dictionary; Strata does not guess the meaning of `R(fail_rate)`.
The built-in portfolio expression is `geomean(task.base_cost / task.cost)`.

Sampled domains require at least two distinct precommitted seeds and
`compare = "expected"` or `"fixed_seeds"`. Both oracle and verifier receipts
must report those seeds in the declared order. This checks consistency; the
adapter is responsible for actually using them and for any failure-rate model.
Do not choose candidates based on favorable candidate-specific draws.

Record dependency hashes as well as the verifier entry point. Strata pins the
adapter source file, but cannot infer every dependency imported by a verifier.

## Floors and memory

`proof` and `counting` may supply full-size lower bounds when their assumptions
match the contract. `library` is a known construction, hence an upper bound.
`exact_small` requires size metadata and never silently extrapolates.

The legacy attempt importer requires an explicit map from attempt IDs to
contract fingerprints. Unbound entries remain unbound. Narrative verdicts or
missing evidence artifacts are imported as estimates and remain priced.

## Current integration limits

The project pipeline currently orchestrates one kickmix circuit per task.
`artifacts.manifest` pins multi-file bundles, but a chemistry loader/factorization
bundle still needs a dedicated orchestration adapter. Automatic high-level
lifting, automatic SAT floor search and the historical Kaggle replay adapters
are not implemented. See [the acceptance ledger](ACCEPTANCE.md).
