# Bounded kernel recovery and continued coverage

`discover` and range inspection retain two separate disk-backed indexes in
`stream/kernels.sqlite`:

- `kernels` / `occurrences` group complete window syntax after canonical quantum
  and classical wire renaming. Each syntax group retains its bounded semantic
  result, and its canonical boundary-zero set is part of the group identity.
  Each occurrence separately records the source of its boundary-zero facts.
- `skeletons` / `skeleton_occurrences` group the ordered CCX, CCZ, CSWAP and HMR
  topology after wire renaming. Non-skeleton layers have a separate
  `linear_hash`; their position between anchors is included in that hash. The
  full occurrence also has a `syntax_hash`. Scope markers, resets and other
  non-skeleton instructions are retained in the layer hash, not discarded.

The brief exposes `paid_skeletons`, including occurrence counts, aggregate static
Toffoli work, distinct linear-layer counts and indexed sample locations. Equal
paid skeletons can compute different functions. They are retrieval observations,
not an equivalence result, an arithmetic-stage identification, or a legal
replacement boundary. Conditional static counts are not execution-weighted cost.

Balanced condition scopes supply candidate cell boundaries. Within unscoped
windows, bounded canonical anchor n-grams detect adjacent repeated paid topology
inside nonperiodic prefixes and suffixes, even when intervening linear gates vary.
Disconnected paid wire neighborhoods provide additional candidate boundaries.
Adjacent linear instructions remain in the indexed spans; the inferred boundary
is not a proof of a complete producer/consumer episode. The search checks at most
128 repeat candidates per window. It can miss repeats across cuts, nonadjacent
repeats without a shared segmentation, or patterns beyond that search budget.

The scan's `coverage` distinguishes exact local semantics, unknown instructions,
indexed paid operations and paid work in repeated skeleton groups. The last two
are observation coverage, not semantic proof coverage. Every occurrence is kept
in SQLite; only bounded top groups and examples are loaded into the brief. Python
scan memory is bounded by the window and the declared zero-fact limit, while the SDK still owns a whole native
circuit and spools canonical text before scanning.

`--max-window` explicitly supplied to `discover` or `inspect-file` sets the scan
cap unless `--growth-window` is also supplied. With neither supplied, the CLI
retains its 8192-instruction recovery cap. The Python APIs use `max_window` as the
scan cap when `growth_window=None`. Reducing the cap can lose complete kernels;
increasing it can exceed symbolic variable/term budgets. Those limitations remain
visible in each syntax group's semantics.

`lift_regions`, used by file optimization, cuts and continues when operation or
variable budgets fill a window. A bounded semantic result can be subdivided at
closed-scope boundaries. `coverage.budget_cuts` and `budget_cut_reasons` count
these cuts. Oversized open condition scopes stay unknown through their closing
POP; following instructions are analyzed again. An HMR or conditioned gate does
not make the entire suffix opaque: later local contracts use arbitrary classical
boundary inputs and do not inherit prior outcome provenance or phase repair.

Explicit artifact clean inputs initialize zero facts only at the original start.
A range starting at zero can use the promise. Conservative transport preserves
an untouched zero, moves it through an unconditional SWAP, and learns the zero
output of an unconditional HMR. A possible write removes the affected fact;
unsupported instructions invalidate the facts. Conditional HMR does not establish
an unconditional zero, and R never creates a zero fact. Thus reused scratch does
not silently regain its original promise at a cut. Classical snapshots and
measurement phase provenance are not transported by these zero facts.

Region `boundary_zero_evidence` and occurrence `zero_evidence` record the initial
promise or HMR origin and any later SWAP transport. Kernel evidence uses canonical
wire IDs, while origins retain physical wire IDs and artifact instruction offsets.
The fact table is capped at `max_boundary_zero_facts = max(max_window,
max_variables)`; `coverage.zero_fact_evictions` reports conservative removals.
An evicted fact, or a zero requiring algebraic cancellation across cuts, can leave
a later reset unknown. No assertion of zero is inferred merely from seeing R.
For Boolean floors, an R with an exact active-zero proof costs no AND gates;
measurement and classical channels remain outside that floor model. An R on an
unproved or dirty value still blocks exact analysis.

## Real loader regression receipt (2026-11-17)

The bounded scanner was exercised on the actual 381,120-byte `angle_k_0.kmb`
loader with source SHA-256
`06a64018000361fdf1e9cb672131a8766ef8b5f3d366ba0e2ae8ad13a8c41944`.
The circuit has 128 quantum wires, 42,153 native operations and 57 register
metadata instructions. Its public registers identify selection wires 0–10 and
five 8-bit words on wires 11–50. Workspace wires 51–127 were explicitly declared
clean because the actual verifier's `dfthcsos/scoreinfra/circuits.py` requires
every nonpublic wire to start and end at zero. That verifier source has SHA-256
`fc027b52fdcebe14404f403a089122485d097ed65d03e6d989623d620b1ab68c`.
The promise was not inferred from the loader's resets.

With a 64-instruction window, 256-variable limit and 4096-term limit, full-range
streaming discovery took 4.29 seconds in the local test environment:

| Observation | Result |
| --- | --- |
| Exact local instruction coverage | 42,210 / 42,210 |
| Exact local paid-operation coverage | 283 / 283 |
| Reset instructions with zero proofs | 80 / 80 |
| Exact syntax occurrences | 716; no unknown occurrences |
| Paid work indexed by skeleton | 283 / 283 |
| Paid work in repeated skeleton groups | 268 / 283 (94.70%) |
| Budget cuts / zero-fact evictions | 658 / 0 |

`lift_regions` also recovered all 42,153 operations and all 80 reset proofs with
both 64- and 256-instruction caps (3.12 and 5.37 seconds respectively). The latter
used 248 additional semantic term-budget splits. A 256-instruction *kernel scan*
without those recursive splits had only 9.63% exact instruction coverage: its
remaining blocker was the term budget, rather than reset assumptions. Window
semantics and whole-loader semantics remain different scopes.

This receipt establishes local bounded semantic support for this real measured
loader, including untouched first-touch resets and scratch recycled after HMR.
It does not establish whole-loader equivalence, host acceptance, high-level stage
recovery, or an optimization gain. No circuit payload or large submission bundle
is retained in this repository.
