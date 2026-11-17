# Strata

**Contract-driven circuit restructuring on kickmix.** Private research library;
automatic understanding of a supplied circuit is a **required core capability**,
not an optional adapter integration. The implementation is still below the full
product acceptance targets. The focus is changes to stage boundaries and architecture
(L3/L4), with reproducible evidence for every retained implementation.

The [product specification](docs/SPECIFICATION.md) is preserved verbatim.
[Implementation status](docs/ACCEPTANCE.md) distinguishes working features,
partial features and unimplemented research milestones.

## Documentation

- [Circuit-file workflow](docs/FILE-FIRST.md): automatic inferred functions, native profiles, and checked replacements without a handwritten adapter.
- [Agent discovery loop](docs/AGENT-DISCOVERY.md): the connected agent derives hypotheses from Strata's evidence and feeds candidates back into the lab.
- [Connected-agent trial](docs/CONNECTED-AGENT-TRIAL.md): what an actual supplied loader revealed about target granularity and missing semantics.
- [Getting started](docs/GETTING-STARTED.md): installation, first structural change, and interpreting results.
- [Adapter guide](docs/ADAPTER-GUIDE.md): connect a real generator, reference, cost model and verifier.
- [Troubleshooting](docs/TROUBLESHOOTING.md): failed gates, scope errors and missing evidence.
- [Agent skill](skills/strata-research/SKILL.md): focused workflow for agents using or extending Strata.
- [Structural research workflow](docs/STRUCTURAL-RESEARCH.md): dependency-complete episodes, diverse hypotheses, and measured candidate rounds.
- [First real targets](docs/FIRST-REAL-TARGETS.md): decoder and chemistry integration/readiness plan.

The agent skill is versioned with this repository. It is not automatically
installed into your global Codex configuration. It can be invoked by pointing an
agent to its `SKILL.md`, or installed through your usual skill installation flow.

## Start with a circuit file

After installation, use the file directly. No `TaskAdapter`, generator source,
region names, desired replacement, or project file is required for these commands:

```bash
strata inspect-file circuit.kmb --out work/inspection
strata optimize circuit.kmb --out work/automatic --cost 'Q*(toffoli+t/4)'
strata discover circuit.kmb --out work/discovery
```

The file path now infers exact Boolean functions and sign phase on supported
regions, including restored-intermediate episodes, affine maps and bilinear
accumulators. It can resynthesize output functions to remove materialization
and cleanup, with every boundary bit arbitrary. Native profiles rank the search;
unsupported operations remain opaque. `discover` supplies the connected AI
agent with evidence and targets; the [skill](skills/strata-research/SKILL.md)
requires that agent to originate hypotheses and construct candidates without
waiting for the user to suggest each change.

This bounded algebraic understanding is **not yet general recovery of the
decoder, divider, or chemistry stage architecture**. The high-level lifting
acceptance target remains unmet. See [current limits](docs/FILE-FIRST.md).

## Installation and the original adapter example

Requires Python 3.11+ **with a compatible kickmix wheel** (the checked environment
uses Python 3.14 and kickmix distribution 1.0.0). The kickmix module reports its
own internal version separately.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest -q
strata --project examples/and_fusion/strata.toml init
strata --project examples/and_fusion/strata.toml analyze and
strata --project examples/and_fusion/strata.toml gaps and
```

The Python path exercises an entire producer/consumer restructuring:

```python
import strata

p = strata.open('examples/and_fusion/strata.toml')
p.init()
d0 = p.task('and').base_design()
line = p.open_line(parent=d0, move='fuse',
                   regions=['lane-0', 'lane-1', 'lane-2'],
                   implementation={'fused': True},
                   note='Consume products directly; remove intermediate scratch')
assert line.lab(bits=2).passed
measured = line.build()
d1 = line.verify()
assert p.replay(d1.id)['passed']
print(p.best())
```

The original generator materializes three products, consumes them and then
uncomputes them. The fused generator consumes each product directly in its
output accumulator. Both implement `y ^= a & b` and preserve `a` and `b`, for
all initial accumulator values. The full example changes **12Q / 6 Toffolis**
to **9Q / 3 Toffolis**, reducing `Q × Toffolis` from **72 to 27**. This is a
small original demonstration of the library; it is **not a new Kaggle result**.
The full fixture verifier exhausts all 512 inputs. Measurement repair has
separate exhaustive-branch and actual-native-random tests.

The example's verifier is explicitly labeled a **demo verifier**. A competition
score still requires its real verifier and acceptance data. File mode proves
supported replacements equivalent to the supplied circuit; it does not validate
the original's contest eligibility or fabricate a scale model.

## What is implemented

- Immutable contract vocabulary: regions, register roles, phase debts, scale
  metadata, stable semantic fingerprints, typed moves and scoped patches.
- Real kickmix `.kmx`/`.kmb` loading, static gate counts, reaction depth, flame
  SVGs and bounded HTML diagrams. Conservative liveness is labeled as such.
  Optional `execution_profile` supplies empirical executed counts, conditional
  dependencies, observed liveness and value fingerprints for explicit inputs.
- Adapter-free algebraic lifting, native profile-driven candidate selection,
  arbitrary-input exact regional equivalence and materialization-removal synthesis.
- Evidence briefs and candidate evaluation for connected-agent hypothesis
  generation. A CLI invocation alone does not launch an AI provider.
- Exhaustive small-instance lab, including every dirty input and every HMR
  branch for its supported gate subset. It checks restored inputs, zero scratch
  and phase, returns concrete counterexamples, and refuses oversized domains.
- A real build-price gate and constrained price queries. Constraints are checked;
  unavailable generators produce an error rather than an estimated price.
- Ordered lab → measured build → full oracle → pinned verifier gates.
  Certificates bind artifacts, implementation, adapter and verification evidence.
- Content-addressed artifacts, transactional SQLite records, immutable branches,
  parked-line memory, contract-level differences and deterministic replay.
- Dependency-aware patch compatibility. Overlapping regions/ancestors or shared
  debt dependencies cannot be auto-composed. A composition still runs all gates.
- Structural research rounds: dependency/debt closure, mechanism diversity, all configured small scales, artifact-bound depth checks, and replay quarantine.
- Typed floors: best constructions are upper bounds; exact small-size results
  never silently become full-size lower bounds. Unknown floors stay unknown.
- Legacy memory import: unstructured “closed” verdicts remain priced. Unbound
  historical attempts are retained instead of assigned invented fingerprints.
- One Python core, CLI and compact agent query answers with full-evidence handles.
- An explicit offline Bubblewrap runner for external generators, with no shell
  or unsafe fallback. Python API adapters are trusted code, not sandboxed code.

## What is not yet implemented

High-level lifting of arbitrary untagged arithmetic, automatic scale recovery,
SAT floor search, automatic synthesis of all ten move types, a production Kaggle
adapter, and the ecdsa.fail front end are pending. The current lift preserves
opaque regions and distinguishes compound algebra from identified arithmetic
stages; it does not claim the target 90%. Exact symbolic liveness/information slack and
object allocation need additional proofs or generator traces. No claim is made
of matching R1–R5, finding 3/4 historical restructures without hints, or finding
a new competitive circuit. The 322-row memory test is synthetic, not the
user's complete historical ledger. These limitations are tracked in
[the acceptance matrix](docs/ACCEPTANCE.md).

## Extend it

Implement the [TaskAdapter protocol](src/strata/protocols.py) and pass an instance
in `strata.open(path, adapters={'name': adapter})`. A single implementation of
`build(parameters, implementation)` produces both small and full instances.
Keep the mathematical reference and official verifier independent of generator
choices. The source file of the adapter is pinned; callers must also pin its
external dependencies in their verifier receipts.

`measure` must return executed resources for the committed input distribution,
not a paper estimate. `verify` returns `passed`, `resources`, `verifier_hash`,
`failure_model`, and the committed `seeds` for sampled domains. Multi-file
submissions can be pinned with `strata.artifacts.manifest`; the current built-in
project orchestration expects one kickmix circuit, so chemistry-style bundles
need a dedicated orchestration adapter before claiming full support.

Configuration is TOML, never executable Python. Cost arithmetic uses a restricted
AST interpreter. Undefined functions such as contest-specific `R(fail_rate)`
must be supplied explicitly; there is no invented default failure penalty.
The only built-in portfolio formula is the specification's geometric mean.

Python adapters and verifier receipts are inside the trust boundary. Certificates
are reproducibility records, **not cryptographic signatures or Lean proofs**.
The finite-size lab is not a theorem about all generator sizes. Its forced HMR
harness uses the exact basis-state sign/reset rule and is cross-checked against
native random HMR; it does not claim a general arbitrary-state quantum proof.

## Storage and safety

Each project owns its `.strata` store. Old designs and losing siblings remain.
Secrets, third-party circuit/source snapshots, competition arrays and generated
circuit binaries do not belong in this repository. The library contains no
Kaggle submission code. Publishing remains a user action.

The repository is private. No open-source license has been granted; dependencies
retain their own licenses. A license can be selected explicitly before any
public release.
