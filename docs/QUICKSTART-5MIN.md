# Strata 0.7 Quickstart: Your First Verified Optimization in 5 Minutes

Strata is an algorithmic co-design and low-latency circuit restructuring engine built on top of Kickmix. Instead of treating your classical preprocessing, numerical tables, and quantum gate generator as isolated stages, Strata lets you optimize across those boundaries and screen thousands of candidate rewrites per second in a persistent C++20 memory-mapped workspace.

## 1. Install and Verify the Native Latency Core

From a source checkout with Python 3.12+ and a C++20 compiler (`g++` or `clang++`):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/strata benchmark-latency --ops 100000 --qubits 48 --cones 16
```

Confirm `native_available` is `true`. On a modern Linux host, opening a binary `.kmb` circuit via `KmbTape` takes a few milliseconds, and checkpointed 256-lane causal-cone equivalence checks complete in roughly `0.3` to `3.0` milliseconds per candidate.

## 2. Profile Any `.kmb` or `.kmx` Circuit Instantly

Zero-copy columnar indexing reads `.kmb` v2 headers and packet offsets without converting tens of millions of gates into Python strings:

```bash
.venv/bin/strata profile-fast path/to/circuit.kmb
.venv/bin/strata trace-cuts path/to/circuit.kmb --cuts 0,500,1000
```

## 3. Open a Hot `Session` and Screen Rewrites in Milliseconds

Open your parent artifact once with `strata.Session`. Every query, cut trace, wire-loan scan, MBU uncompute conversion, and causal-cone check shares the same in-memory checkpoint cache:

```python
import kickmix
import strata

# Build a small reversible stage that computes and uncomputes an intermediate AND wire
b = kickmix.CircuitBuilder()
q = b.alloc_qubits(5)
b.cx(q[0], q[3])          # op 0: q3 holds a copy of q0 (idle loan candidate!)
b.ccx(q[0], q[1], q[4])   # op 1: compute scratch product q4 = q0 & q1
b.cx(q[4], q[2])          # op 2: consume product into output q2
b.ccx(q[0], q[1], q[4])   # op 3: reversible Bennett uncompute (costs 1 Toffoli)
circ = b.finish_circuit()
circ.to_file("stage.kmb", format="kmb")

with strata.Session.open("stage.kmb") as sess:
    # 1. Replace reversible CCX uncompute [3, 4) with 0-Toffoli HMR + CZ_IF
    mbu = strata.demolish_and_piggyback(sess, 3, 4)
    print("Toffolis removed:", mbu["toffolis_removed"], "Verified:", mbu["verified"])

    # 2. Discover live idle wires that are affine functions of other idle wires in [1, 4)
    loans = strata.find_affine_wire_loans(sess, 1, 4)
    print("Discovered affine wire loans:", loans)

    # 3. Screen multiple candidate rewrites through the 4-tier latency funnel
    report = sess.funnel([
        strata.FunnelCandidate("mbu_cleanup", 3, 4, mbu["replacement_kmx"], parent_toffolis=1),
    ])
    print("Survivors:", len(report["survivors"]))
```

## 4. Where to Go Next: Choose Your Co-Design Lever

Real order-of-magnitude circuit wins rarely come from local gate peepholes alone. Use the playbook matching your bottleneck:

1. **Table / Hamiltonian / State-Prep Co-Fitting** (`BoundarySlider`, `GaugeSweep`, `DeadSlotPlanter`, `QuotientChainDP`): see [PLAYBOOK-COFIT.md](PLAYBOOK-COFIT.md).
2. **Aggressive Precision / Carry Truncation + Minimax Rotation Refitting** (`CompensatedTruncator`): see [PLAYBOOK-TRUNCATION-LP.md](PLAYBOOK-TRUNCATION-LP.md).
3. **Peak-Qubit Compression, DAG Pebbling & Affine Wire Loans** (`ScoreModel`, `solve_mbu_pebble_schedule`, `find_affine_wire_loans`): see [PLAYBOOK-SHAPES-PEBBLING.md](PLAYBOOK-SHAPES-PEBBLING.md).
4. **Measurement-Based Uncomputation & Multi-Cut Phase Repair** (`demolish_and_piggyback`, `fit_multi_cut_phase`, `hoist_hmr_across_swaps`): see [PLAYBOOK-MBU-PHASE.md](PLAYBOOK-MBU-PHASE.md).
5. **0-Toffoli Classical Preprocessing, Projective GF(2^k) & Bilinear Tensors** (`ClassicalProgram`, `ProjectiveGF2k`, `lovasz_matroid_parity_rank`, `flip_graph_reduce`): see [PLAYBOOK-BOUNDARY-TENSORS.md](PLAYBOOK-BOUNDARY-TENSORS.md).
6. **C++20 Columnar Engine & 4-Tier Funnel Internals**: see [LATENCY-ARCHITECTURE.md](LATENCY-ARCHITECTURE.md).
