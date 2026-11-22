# Playbook: Shape Exploration, MBU DAG Pebbling, and Affine Wire Loans (`strata.shapes`)

## Why Shape Changes Dominate Local Gate Tweaks

Under exponential-width or product resource objectives (such as `2^(Q_ref - Q) / T` or `Q * T`), shaving **1 peak ancilla qubit** across the global bottleneck island can be worth hundreds or thousands of Toffolis. `strata.shapes` gives you three tools to price and execute width-first restructurings algorithmically:

1. **`ScoreModel` & `ShapeExplorer`**: Computes exact marginal **Qubit-to-Toffoli exchange rates** (`toffolis_per_qubit`) and Pareto frontiers so you know immediately how many extra recomputation Toffolis you can pay to save `1` qubit.
2. **`solve_mbu_pebble_schedule`**: Solves the **Hybrid Reversible + Measurement-Based Uncomputation (MBU) DAG Pebbling Game**, showing how to fit a multi-stage computation inside a tight `max_qubits` cap while paying **0 Toffolis** on `HMR` erasures!
3. **`find_affine_wire_loans` & `apply_affine_wire_loan`**: Implements the generalized **`LEND` affine wire-loan mechanism**: temporarily clearing a live idle wire `q_w` whose value is a `GF(2)` affine combination of other idle wires so inner stages can use `q_w` as clean `|0>` scratch without allocating a new qubit!

## 1. Price Your Qubit-vs-Toffoli Exchange Rate (`ScoreModel`)

```python
import strata

model = strata.ScoreModel("invsqrt")
rate = model.exchange_rate(qubits=38, toffolis=1800)
print(f"Saving 1 qubit (38Q -> 37Q) pays for up to +{rate['toffolis_per_qubit']:.1f} extra Toffolis!")
```

## 2. Hybrid Reversible + MBU DAG Pebbling (`solve_mbu_pebble_schedule`)

Suppose you have a 5-stage arithmetic chain `0 -> 1 -> 2 -> 3 -> 4` where each intermediate register is `8` qubits wide and costs `50` Toffolis to compute, and you only need stage `4` at the end inside a `24`-qubit budget (`3` simultaneous registers max):

```python
import strata

edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
widths = [8, 8, 8, 8, 8]
costs = [50, 50, 50, 50, 50]

rev = strata.solve_mbu_pebble_schedule(
    edges, 5, input_nodes=[0], output_nodes=[4],
    max_qubits=24, node_widths=widths, compute_toffolis=costs, allow_hmr=False,
)
mbu = strata.solve_mbu_pebble_schedule(
    edges, 5, input_nodes=[0], output_nodes=[4],
    max_qubits=24, node_widths=widths, compute_toffolis=costs, allow_hmr=True,
)
print(f"Pure reversible Bennett pebbling: {rev.total_toffolis} Toffolis, peak {rev.peak_qubits}Q")
print(f"Hybrid MBU pebbling:              {mbu.total_toffolis} Toffolis, peak {mbu.peak_qubits}Q ({mbu.hmr_erasures} HMR erasures)")
```

Because `HMR` erases an intermediate pebble while its parent is still live (repaying the `CZ_IF` phase kickback immediately on the live parent for `0` Toffolis!), MBU pebbling achieves the same `24Q` tight width cap with dramatically fewer Toffolis than reversible Bennett recomputation!

## 3. Affine Idle-Wire Loans (`find_affine_wire_loans`)

At your circuit's peak-qubit island `[start, stop)`, all `Q` qubits appear "live," so naive allocators assume you cannot shrink `Q` without changing the outer algorithm.

Often, however, one idle wire `q_w` sitting untouched across `[start, stop)` actually holds an affine `GF(2)` parity `q_w == c_0 ^ q_{s1} ^ q_{s2}` of other idle wires `q_{s1}, q_{s2}` (for example, carry-save parity bits, syndrome copies, or encoder fanout wires).

Run from the CLI:

```bash
.venv/bin/strata wire-loans circuit.kmb --range 1200:1850
```

Or apply and verify in Python in `<2 ms`:

```python
import strata

with strata.Session.open("circuit.kmb") as sess:
    loans = strata.find_affine_wire_loans(sess, 1200, 1850)
    if loans:
        best_loan = loans[0]
        # `best_loan["loaned_wire"]` is now guaranteed |0> inside [1200, 1850) between entry_ops and exit_ops!
        wrapped = strata.apply_affine_wire_loan(sess, 1200, 1850, best_loan)
        assert wrapped["verified"]
```
