# Separately frozen mixed value/phase cases

Three new compositions were frozen with seed 731809 before any corrected-engine
candidate result. The earlier nine-case phase failure was already known. The
original nine-case receipts remain unchanged; their later exposed-case rerun is
separately labeled in `../exposed-nine-posthoc-comparison.json`.

| Case | Previous snapshot | Mixed extension |
|---|---:|---:|
| updated_dirty_target_phase | 88 | 32 |
| dependent_observable_then_adapter | 72 | 72 |
| fractional_observable_guard | 80 | 80 |

The first gain comes from `mixed_observable_search`. The second case produces
no cheaper retained result. The fractional case remains explicitly unsupported
for construction and retains its original bytes. Both snapshots pass all three
independent native checks, covering 128, 32 and 64 complete inputs respectively.
The checker compares exact native `Fraction` phases modulo two, including the
fractional negative case. Candidate value and phase mutations are also rejected.

`comparison.json` binds both immutable source snapshots and retained-byte
replays. The extension was evaluated with `fast=False`; this is separate from
later ordinary CLI-default portfolio measurements. These small cases establish
one additional supported composition, not general mixed-channel or fractional
phase discovery.

Run `mixed/run.py --package-root PACKAGE_ROOT --out NEW_DIRECTORY --label LABEL`
from the repository root using `.venv/bin/python`. The runner supplies no cut or
replacement and uses the frozen one-round, 12-candidate, 128-operation ceilings.
`check_results.py` replays selected bytes and refuses to overwrite the published
mixed comparison.
