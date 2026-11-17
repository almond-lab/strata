# Strata

**Contract-driven circuit restructuring on kickmix.** Strata 0.3 recovers bounded
semantic functions and effects, constructs candidates through generic mechanisms,
and retains checked alternatives and reusable transformations. The connected agent
uses its evidence to propose further structural changes. Automatic understanding of a
supplied circuit is a required core capability; broad arithmetic-stage recovery
and the historical discovery benchmarks remain incomplete.

The [product specification](docs/SPECIFICATION.md) is preserved verbatim.
The [acceptance ledger](docs/ACCEPTANCE.md) separates implemented APIs from
unmet research milestones. See [v0.3.1 fixes](docs/V0.3.1-FIXES.md),
[v0.3 issue coverage](docs/V0.3-ISSUES.md) and
[release changes](CHANGELOG.md).

## Documentation

- [v0.4 plan](docs/V0.4-PLAN.md): proposed execution workflow for existing scripts, bounded sweeps, a shared experiment board and checked artifacts; not shipped v0.4 functionality.
- [v0.3 workflow and examples](docs/V0.3-GUIDE.md): the Strata–agent loop, construction search, finite domains and bounded scans.
- [Measured lookups](docs/MEASURED-LOOKUPS.md), [kernel recovery](docs/KERNEL-RECOVERY.md), and [seeded tensor search](docs/TENSOR-SEARCH.md): clean resets, HMR-preserving table construction, paid skeletons and real-size decomposition seeds.
- [v0.3 research limits](docs/V0.3-RESEARCH-LIMITS.md): verified capabilities versus remaining research.
- [Candidate benchmarks](docs/V0.3-CANDIDATE-BENCHMARK.md): million-scale native execution and analysis-guided versus unguided proposals, with separate counts and scope.
- [Numeric candidates and pipelines](docs/V0.3-NUMERIC-CANDIDATES.md): common array format, calibrated scoring, gradients and reusable fitting/rounding stages.
- [Numeric studies](docs/V0.3-NUMERIC-STUDIES.md), [models](docs/V0.3-NUMERIC-MODELS.md), and [checkpointed jobs](docs/V0.3-NUMERIC-JOBS.md): error budgets, shape prices, uncertainty, bounds, experiment sharing and safe-candidate/bet accounting.
- [Getting started](docs/GETTING-STARTED.md): installation, file discovery, a verified change and saved reports.
- [Circuit-file workflow](docs/FILE-FIRST.md): inferred functions, native profiles and checked replacements.
- [Agent discovery loop](docs/AGENT-DISCOVERY.md): evidence-driven hypotheses, candidate evaluation and prebuild notes.
- [Adapter guide](docs/ADAPTER-GUIDE.md): bundles, frozen components, promised domains, config-only tasks, callbacks and evidence scope.
- [Structural research](docs/STRUCTURAL-RESEARCH.md): dependency-complete episodes and measured rounds.
- [Troubleshooting](docs/TROUBLESHOOTING.md), [first real targets](docs/FIRST-REAL-TARGETS.md), and [connected-agent trial](docs/CONNECTED-AGENT-TRIAL.md).
- [Agent skill](skills/strata-research/SKILL.md): the repository's research workflow. It is not automatically installed globally.

## Install and start with a file

Requires Python 3.11+ with a compatible kickmix wheel. The checked environment
uses Python 3.14 and kickmix distribution 1.0.0.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/strata inspect-file circuit.kmb --out work/inspection
.venv/bin/strata discover circuit.kmb --out work/research
.venv/bin/strata optimize circuit.kmb --out work/automatic --cost 'Q*(toffoli+t/4)'
.venv/bin/strata restructure episode.kmb --out work/construct --rounds 2
```

No adapter, project file or suggested replacement is needed for the file
workflow. Supported regions receive exact Boolean functions and sign-phase
analysis, including affine maps, bilinear accumulators and materialization
removal. Unsupported semantics stay opaque. `discover` supplies evidence to the
connected agent, which follows the skill to originate hypotheses and construct
candidates. A standalone CLI call does not invoke an AI provider.

## What 0.3 adds

- Numeric artifacts as research objects: hashed named arrays, explicit units and
  trainable masks, lineage, pinned/calibrated scorers and priced stage pipelines.
- Error-budget attribution, discrete shape pricing, compatible experiment transfer,
  convergence bands, scoped quadratic rounding floors and sensitivity maps.
- Checkpoint/resume jobs with predeclared observed stop rules, bounded offline
  workers, safe-portfolio/bet comparisons and independent verification on readiness.
- Exact bounded Boolean effects for CSWAP, classical controls, HMR outcome
  variables, conditional phase repairs and resets with proved or declared clean inputs.
- Interior-window and finite-promised-domain queries with explicit scope and
  hashes; constructive reversible compression witnesses and phase-only synthesis.
- Generic constructor registry, complete-boundary candidate checks, retained
  nonwinning intermediate representations, Pareto records, and learned native
  transformations whose preconditions and equivalence are checked on reuse.
- Bilinear tensor extraction, flattening and certified slice-substitution bounds,
  bounded decomposition search and reversible lowering. These rank bounds have
  an explicit bilinear model; they are not unrestricted quantum resource floors.
- Disk-backed repeated-kernel indexing, bounded Python windows, progress output
  and range scans. The SDK still loads one native circuit and spools canonical text.
- Binary bundle circuits, preserved virtual environments, configurable sandbox
  budgets, retained rejected receipts and count-aware probability objectives.

The [500,000-line synthetic benchmark](docs/V0.3-STREAMING-BENCHMARK.json)
took 20.8 seconds with approximately 63 MB process peak RSS in the release
environment. Different semantics, unique kernels, term budgets and native input
sizes change runtime and memory; a full 50-million-instruction host circuit was
not benchmarked for this release.

The separate [native candidate benchmark](docs/V0.3-NATIVE-CANDIDATE-BENCHMARK.json)
actually built and simulated **1,000,000 distinct small circuits in 595.85 seconds**,
checking all 256 input assignments per candidate. The [guided search comparison](docs/V0.3-CANDIDATE-BENCHMARK.md)
shows how recovered tensor constraints avoid most invalid proposals. This is
bounded component search, not million-candidate throughput for full host circuits.

## Included from 0.2

- Task-specific frontends, multi-file bundles, numeric data and frozen opaque
  components through the ordinary Line/build/verify/replay pipeline.
- Configuration-only artifact tasks with pinned external verifier commands, plus
  typed pinned callbacks for trusted Python integrations.
- Explicit promised-input domains, checked zero resets, cached native instruction
  segments and batched finite-domain simulation.
- Proved affine HMR branch reduction where applicable, with honest executed and
  represented counts; unsupported dependencies use full enumeration or refuse
  the budget. Deterministic shards remain partial evidence.
- Scoped Boolean dependency and information-slack queries, counting bounds and
  bounded exact XOR/AND minima. These do not imply free scratch or a full quantum
  resource lower bound.
- Source-less region receipts bind recovered exact Boolean/sign-phase maps to
  source bytes and contracts, with explicit same-backend symbolic replay.
  Unsupported regions remain unknown; no scale model is invented.
- Precommitted probability-policy comparisons with failure budgets and per-draw
  objectives; candidate-selected favorable draws are rejected.
- A JSONL reversible-operation frontend sharing the native kickmix execution
  backend, alongside native circuit input.
- Stable integer-key record serialization, artifact-checked negative evidence,
  JSONL/CSV/JSON history import and scoped memory application on a new base.
- Store-only JSON/Markdown research reports covering Lines, outcomes, mechanisms,
  measured deltas, file attempts, prebuild notes, reopening conditions, shared
  failures and stored gap snapshots. Missing gaps stay unknown; reporting runs
  no generators or verifiers.

The core still provides content-addressed artifacts, semantic contracts,
phase debts, scoped patches, measured prices, independent verification gates,
replay quarantine and compatibility-checked composition. See the acceptance
ledger for the scope of each implementation and remaining research milestones.

## Original structural example

```bash
.venv/bin/strata --project examples/and_fusion/strata.toml init
.venv/bin/strata --project examples/and_fusion/strata.toml analyze and
.venv/bin/strata --project examples/and_fusion/strata.toml gaps and
```

```python
import strata

p = strata.open('examples/and_fusion/strata.toml')
p.init()
base = p.task('and').base_design()
line = p.open_line(base, 'fuse', ['lane-0', 'lane-1', 'lane-2'],
                   implementation={'fused': True},
                   note='Consume products directly and remove intermediate scratch')
assert line.lab(bits=2).passed
price = line.build()
design = line.verify()
assert p.replay(design.id)['passed']
```

The fixture implements `y ^= a & b`, preserving arbitrary initial accumulators.
It changes **12Q / 6 Toffolis to 9Q / 3 Toffolis**, reducing `Q × Toffolis` from
**72 to 27**. Its explicitly labeled demo verifier checks all 512 full-size
inputs. This is a teaching fixture, not a competitive result. File-mode
replacement equivalence likewise does not establish the supplied base's contest
eligibility.

## Evidence and compatibility

Python adapters/callbacks are trusted code. External commands use the offline
Bubblewrap runner with no unsafe fallback. Certificates bind evidence to
artifacts and task configuration; they are reproducibility records, not signed
attestations or general mathematical proofs. Approximate numerical acceptance
requires the task's actual independent checks.

String-key record hashes retain their encoding. Integer dictionary keys now
normalize recursively to strings before sorting/hashing; mixed key types are
rejected to prevent collisions. Existing saved records are not rewritten.
Strong negative labels require valid stored blob evidence, and old unsupported
claims are downgraded when memory/report APIs read them. See
[compatibility notes](CHANGELOG.md#compatibility-and-evidence-changes).

A fresh untagged eight-bit fixture also produces two automatic rewrites without
an adapter or supplied hypotheses: static cost 32→16, exact independent checking
of 256 arbitrary dirty inputs, and replay. This is a bounded original fixture,
not historical or competition acceptance.

The ≥90% structural typing / ≥95% boundary-agreement benchmark, historical
3-of-4 restructure discovery and R1–R6 replay targets have not been established.
The actual 322-attempt historical ledger was not found. Real import checks cover
16 workflow attempts, 87 CSV entries, one wrapped research note and a separate
262-row event log; event rows are not counted as historical attempts.

## Storage and publication

Each project or discovery output owns its `.strata` store. Old designs and losing
siblings remain. Private circuits, source snapshots, competition arrays and
credentials do not belong in this repository. Strata contains no automatic
competition submission command.

This repository is private. No open-source license has been granted; dependencies
retain their own licenses.
