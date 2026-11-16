# Getting started with Strata

Strata helps you change a circuit's stage boundaries and keep the evidence for
whether the change works. The current release provides a tested framework and
an executable example. It does not yet automatically discover arbitrary circuit
algorithms. Check [implementation status](ACCEPTANCE.md) before planning a run.

## Install

Clone the private repository using your GitHub account, then install into an
isolated environment. Python 3.14 is the tested version.

```bash
git clone https://github.com/almond-lab/strata.git
cd strata
python3.14 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

The remaining commands assume you are in the repository root. Use the explicit
`.venv/bin/strata` path so you do not accidentally invoke another installation.

## Understand the example

The example takes three bit strings, `a`, `b` and `y`, and performs
`y ^= a & b`. It preserves `a` and `b`; `y` may contain arbitrary initial data.

The base computes products into scratch, copies them into `y`, then clears the
scratch. The redesigned circuit applies products directly to `y`. That changes
what crosses the intermediate boundary and removes the scratch register.

At three bits the base costs 12 qubits × 6 Toffolis = 72. The redesign costs
9 qubits × 3 Toffolis = 27. This is a teaching fixture, not a competition result.

## Ingest and inspect

```bash
.venv/bin/strata --project examples/and_fusion/strata.toml init
.venv/bin/strata --project examples/and_fusion/strata.toml analyze and
.venv/bin/strata --project examples/and_fusion/strata.toml gaps and
.venv/bin/strata --project examples/and_fusion/strata.toml best
```

`init` checks that the adapter reproduces the supplied base, runs the small lab
and full fixture verifier, then stores the base. `analyze` writes structured
metrics and actual kickmix views under
`examples/and_fusion/.strata/analysis/and/`. Open `index.html` in a browser.

The gap table reports unknown floors explicitly. A library construction is an
upper bound; a small-size exact result is limited to that size. Neither is a
full-size impossibility proof.

## Make and verify a structural change

Run this from the repository root:

```bash
.venv/bin/python - <<'PY'
import strata

p = strata.open('examples/and_fusion/strata.toml')
p.init()
base = p.task('and').base_design()
line = p.open_line(
    parent=base,
    move='fuse',
    regions=['lane-0', 'lane-1', 'lane-2'],
    implementation={'fused': True},
    note='Consume each product directly in the output accumulator',
)
lab = line.lab(bits=2)
if not lab.passed:
    raise RuntimeError(lab.counterexample)
price = line.build()
result = line.verify()
print('Measured cost delta:', price['delta'])
print('Design:', result.id)
print('Replay:', p.replay(result.id)['passed'])
print('Portfolio:', p.best())
PY
```

The small lab exhausts 64 inputs, including every initial accumulator. Full
verification covers all 512 inputs at three bits. Replay regenerates the
artifact and rechecks its verifier and recorded cost. Losing but correct
variants can also be retained; `best` selects the cheapest certified one.

The gate order is deliberate: **small lab → measured full build → full oracle
→ pinned verifier**. A passing component test alone does not produce a certified
full design.

## Read the results

| Result | What it establishes |
|---|---|
| Exhaustive lab pass | The declared small instance matches its reference for all enumerated inputs and HMR branches |
| Build price | The emitted full implementation has measured resources under the adapter's distribution |
| Oracle/verifier pass | The task's full-domain or fixed-draw acceptance checks passed |
| Certificate | Those records are bound to this artifact, implementation, adapter and design |
| Replay pass | The recorded implementation rebuilds and rechecks successfully |

Certificates are reproducibility evidence, not Lean proofs. Finite testing is
not a theorem about all sizes. Native HMR lab phases use exact fractions of π,
modulo two.

## Start your own task

See [writing an adapter](ADAPTER-GUIDE.md). A flat circuit can be analyzed without
source, but Strata cannot invent a scale-down generator or mathematical
reference. The CLI ships with the demo adapter only; custom trusted Python
adapters are registered through `strata.open(..., adapters=...)`.

Never edit the retained base or official verifier to accommodate a candidate.
Keep large/private circuit artifacts out of the library repository. Strata has
no automatic competition submission command.
