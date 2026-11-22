# Playbook: Measurement-Based Uncomputation & Multi-Cut Phase Fitting (`strata.mbu`)

## Why MBU Cuts Uncompute Toffolis in Half

In reversible quantum arithmetic and table lookups, computing an intermediate AND wire `w = a & b` costs `1` Toffoli (`CCX a b w`), and uncomputing `w` back to `|0>` via Bennett's trick costs **another Toffoli**.

In Kickmix, `HMR q_w b_m` (Hadamard-Measure-Reset in the X basis) resets `q_w` to `|0>` for **0 Toffolis**, introducing only a classical-outcome-dependent sign $(-1)^{b_m (a \cdot b)}$. Repaying that sign requires just `CZ_IF q_a q_b b_m`, which is a Clifford gate costing **0 Toffolis**!

Even better: when an entire tree of `CCX` intermediates is uncomputed at the end of a stage, walking **backward from the final reset** lets every `CCX` in the uncompute cascade be replaced by `HMR + CZ_IF` before its inputs are themselves demolished!

## 1. Automated Cascade MBU (`demolish_and_piggyback`)

`demolish_and_piggyback` inspects any `[start, stop)` span in a `strata.Session`, verifies via 256-lane cut tracing which `CCX` gates genuinely reset their target wire to `|0>`, replaces the entire eligible uncompute tree with `HMR + CZ_IF`, and verifies exact value **and** phase equivalence across both `HMR` branch policies (`mode=1` and `mode=2`):

```python
import kickmix
import strata

# Build a 3-Toffoli AND-tree compute + 3-Toffoli reversible uncompute (6 Toffolis total)
b = kickmix.CircuitBuilder()
q = b.alloc_qubits(7)
b.ccx(q[0], q[1], q[4])   # op 0: t0 = q0 & q1
b.ccx(q[2], q[3], q[5])   # op 1: t1 = q2 & q3
b.ccx(q[4], q[5], q[6])   # op 2: t2 = t0 & t1
b.z(q[6])                 # op 3: phase oracle consume
b.ccx(q[4], q[5], q[6])   # op 4: uncompute t2
b.ccx(q[2], q[3], q[5])   # op 5: uncompute t1
b.ccx(q[0], q[1], q[4])   # op 6: uncompute t0
b.finish_circuit().to_file("and_tree.kmb", format="kmb")

with strata.Session.open("and_tree.kmb") as sess:
    res = strata.demolish_and_piggyback(sess, 4, 7)
    assert res["verified"]
    print("Toffolis removed:", res["toffolis_removed"])  # Removes all 3 uncompute Toffolis!
    print(res["replacement_kmx"])
```

## 2. Multi-Cut Clifford+`T` Phase Fitting (`fit_multi_cut_phase`)

When controls `a` and `b` are no longer directly materialized on single wires at the moment `w` is demolished, you do **not** need to recompute them just to apply `CZ_IF`! Instead, if downstream stages pass through cuts $c_1, c_2, \dots, c_m$ whose basis states span the quadratic product $a \cdot b \pmod{2, 4, 8}$, you can piggyback the phase repair onto existing downstream wires.

`fit_multi_cut_phase` traces basis states across all supplied cuts simultaneously, isolates untouched `Z_POW` wires cleanly without aborting, and solves the linear congruence system over $\mathbb{Z}_2$, $\mathbb{Z}_4$, or $\mathbb{Z}_8$ via `solve_zmod_fast`:

```bash
.venv/bin/strata phase-fit circuit.kmb --cuts 400,850,1200 --modulus 8
```

Or in Python:

```python
with strata.Session.open("circuit.kmb") as sess:
    fit = strata.fit_multi_cut_phase(sess, cuts=[400, 850, 1200], modulus=8)
    print("Consistent:", fit["consistent"], "Untouched wires isolated:", fit["untouched_wires"])
    for term in fit["terms"]:
        print(term)
```

## 3. Hoisting `HMR` Ahead of `SWAP` Networks (`hoist_hmr_across_swaps`)

Routing layers often emit `SWAP` chains before resetting scratch wires, keeping qubits alive across peak-width regions unnecessarily. `hoist_hmr_across_swaps` commutes `HMR` backward across `SWAP`s and disjoint gates and eliminates redundant `SWAP`s on already-reset `|0>` wires:

```python
hoisted = strata.hoist_hmr_across_swaps("SWAP q3 q4\nSWAP q4 q5\nHMR q5 b0\n")
print(hoisted["kmx"])  # Hoists HMR directly onto q3 before the SWAP chain!
```
