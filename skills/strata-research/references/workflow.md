# Current operational interface

## Working checkout

Run commands from the Strata repository with its configured Python environment.
Start with the current `docs/GETTING-STARTED.md` for a runnable fixture and
`docs/ADAPTER-GUIDE.md` for the adapter trust boundary. Python API adapters are
trusted extensions. The optional offline subprocess runner is a separate API;
calling Python adapter methods does not sandbox them.

## Open an implementation line

```python
import strata

p = strata.open(project_file, adapters=task_adapters)
base = p.task(task_id).base_design()
line = p.open_line(
    parent=base,
    move='re-represent',
    regions=['producer', 'consumer', 'return'],
    implementation=generator_options,
    note=hypothesis,
)
lab = line.lab(**declared_small_scale)
if not lab.passed:
    # Save the exact counterexample; do not proceed as if the gate passed.
    print(lab.counterexample)
else:
    price = line.build()
    design = line.verify()
    print(price, design.id, p.replay(design.id))
```

Names and scales above are placeholders for the real project, not universal
registers. When old region names disappear in a fusion, pass
`replacement_regions=['new_region']`. For L2 moves preserve the target contract.
For a full architecture change, target the stable enclosing region such as
`root`. Declare the actual changed scope rather than bypassing scope checks.

`line.park(reason, reopen, evidence='estimate')` preserves a priced line.
Measured build, exact-small and proof evidence require their correct scope;
exact-small is never a full-size closure. `p.merge(first_line, second_line)`
returns a new unverified combined line after checking overlap/dependencies.
Run all gates on that combination. Compatible patches are not automatically
certified.

## Evidence vocabulary

- **Lab pass:** every enumerated input and HMR branch at the specified size.
- **Measured build:** actual emitted and executed resources, priced by the task
  formula; check full versus region-only scope.
- **Full oracle:** the real external task domain, including phase and restoration.
- **Verifier:** unchanged task acceptance, with pinned verifier/dependency hashes.
- **Certificate:** reproducibility evidence binding these results; not a Lean proof.
- **Replay:** regenerate the selected variant and check recorded evidence.

The reference returns `(expected_registers, phase)`, where phase is `Fraction`
in units of π modulo two. Clean helpers are always zeroed; dirty helpers are
always restored independently of reference output. Missing expected boundary
registers are errors.

## Useful APIs

- `base.gaps()` returns ranked evidence, including unknown bounds.
- `base.price(region, resource, delta)` requires a real constrained generator.
- `strata.Queries(p).tried(move, contract_fingerprint)` finds scoped prior results.
- `strata.Queries(p).applies(patch, design)` checks identity/scope only; it does not
  replace a combined build or verifier.
- `p.best()` selects cheapest certified variants; it does not submit anything.
- `p.replay(design_id)` checks artifact, ancestry, verifier identity and cost.

Current `holders` and `is_function_of` queries can return unknown. Do not invent
ownership or dependency proofs to fill that gap. Automatic arithmetic lifting,
SAT floor search and historical replay adapters remain incomplete unless the
checked-out acceptance ledger has newer measured evidence.

## Report format

Use a compact record such as:

```text
Task/base: <task and artifact hash>
Move/boundary: <L2/L3/L4, regions and hypothesis>
Changed artifact: <hash or not built>
Lab: <size, input count, branch count, pass/counterexample>
Build: <Q, executed T, rotations, measured cost; distribution>
Full oracle/verifier: <receipt handles or not run>
Delta: <same-objective measured delta or unknown>
Certificate/replay: <handles and result>
Missing work: <specific obligation>
```

Do not report a change as successful merely because 2,000 tests passed: the task
also needs its actual acceptance conditions and a measured improvement. For
chemistry this includes the physical tensor errors and gap, not just loader
bitstrings. For a probabilistic decoder this includes the agreed draw and
failure accounting.
