# Playbook: Boundary Preprocessing, Projective GF(2^k), and Bilinear Tensors

## 1. Zero-Toffoli Classical Preprocessing (`ClassicalProgram` & OOM Guards)

Kickmix charges **0 qubits and 0 Toffolis** for classical `BIT_*` operations (`BIT_XOR`, `BIT_AND`, `BIT_NOT`, `BIT_SET0`, `BIT_SET1`, `PUSH_CONDITION`). Whenever a syndrome vector, lookup index, or gauge choice depends on classical bits (or on a quantum prefix before the first `H` gate!), computing linear maps, Hamming weights, or sorting networks in quantum gates wastes enormous resources.

`strata.boundary.ClassicalProgram` compiles high-level classical kernels directly into `BIT_*` streams, complete with 4-Russians column-pair CSE and **mandatory OOM safety caps** (`max_ops=80_000_000`, `max_kmb_bytes=650_000_000`) so unrolled classical preprocessors never blow past `30 GB` multi-worker verification container memory limits:

```python
import numpy as np
import strata

prog = strata.ClassicalProgram(max_ops=10_000_000, max_kmb_bytes=100_000_000)
in_bits = prog.alloc_bits(8)

# 1. Compile a GF(2) residual matrix with automatic column-pair CSE
M = np.array([
    [1, 1, 0, 1, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 0, 1, 1, 0, 1, 1],
    [0, 0, 1, 1, 1, 1, 0, 1],
], dtype=np.uint8)
residual = prog.xor_linear_map(M, in_bits)

# 2. Compute exact classical Hamming weight (popcount) in 0 quantum Toffolis
weight_bits = prog.popcount_bits(residual)

# 3. Sort candidate weight bit-vectors with a 0-Toffoli Batcher odd-even merge network
k0, k1 = prog.alloc_bits(3), prog.alloc_bits(3)
prog.odd_even_sort([k0, k1], descending=False)

print(f"Emitted {prog.num_ops} classical ops (~{prog.estimated_kmb_bytes()} bytes in .kmb)")
```

## 2. Projective Homogeneous `(N : D)` Coordinates over `GF(2^k)` (`ProjectiveGF2k`)

In algebraic decoding (Berlekamp-Massey, Chien/Forney evaluation, syndrome interpolation), sequential rational accumulations $\sum_{i=1}^S a_i / b_i$ in $\mathbb{F}_{2^k}$ naively invoke $S$ reversible finite-field inversions. Because Itoh-Tsujii inversion costs $O(k \log k)$ multiplications, per-step division dominates the entire circuit.

`ProjectiveGF2k` shifts the internal representation to homogeneous coordinates $(N : D)$:
- Each step updates $N' = N \cdot b_i \oplus a_i \cdot D$ and $D' = D \cdot b_i$ using cheap Karatsuba multiplications (zero inversions!).
- At the closing boundary, **one single inversion** recovers $x = N \cdot D^{-1}$.

```python
import strata

gf = strata.ProjectiveGF2k(8)
price = gf.price_projective_closing(num_steps=12)
print(f"GF(2^8) 12-step sum: affine={price.affine_total_toffolis} Toffolis vs projective={price.projective_total_toffolis} Toffolis")
print(f"Saved: {price.toffolis_saved} Toffolis!")
```

## 3. Bilinear Tensor Rank Bounds, Relation Hyperplanes & Flip Graphs (`strata.tensors`)

For multi-output Boolean tables and finite-field multipliers:
- **`lovasz_matroid_parity_rank`**: Computes the exact polarized shared-`AND` lower bound over $\mathbb{F}_{2^{64}}$ in microseconds via C++ carryless multiplication.
- **`search_relation_hyperplanes`**: Scans output-bit parity combinations $\bigoplus_{b \in \lambda} T_b(x)$ whose ANF degree collapses to affine (`0` Toffolis!) or low-rank quadratic, letting you synthesize fewer QROM columns and reconstruct dropped columns via free `CX` gates.
- **`plan_multipass_qroam`**: Sweeps `(passes, log_block)` SelectSwap schedules with `0`-Toffoli `HMR` scratch demolition under a `max_qubits` ceiling.
- **`flip_graph_reduce`**: Walks the Kauers-Moosbauer `GF(2)` bilinear flip graph to discover lower-rank multiplication tensors (such as Karatsuba-like rank drops) and reduce post-CSE `CX` fanout!

```python
import numpy as np
import strata

# Schoolbook 2-bit polynomial-style rank-4 bilinear setup with mergeable redundant legs -> reduced automatically
U = np.array([[1, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
V = np.array([[1, 0], [0, 1], [0, 1], [1, 1]], dtype=np.uint8)
W = np.array([[1, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)

reduced = strata.flip_graph_reduce(U, V, W, steps=500)
print(f"Rank: {reduced['initial_rank']} -> {reduced['reduced_rank']}")
```

## 4. Cross-Cell Invariants & Failure-Budget Pricing (`strata.invariants`)

- **`scan_cross_cell_invariants(session, cell_cuts)`**: Traces 256-lane basis transitions across repeated evaluation cells `[c_0, c_1, ..., c_L]` and extracts conserved multi-wire `GF(2)` parities (`parkable_carrier_wires` that can be uncomputed across all `L` cells and reconstructed at cell `L`!).
- **`price_failure_budget` & `audit_shot_paired_dominance`**: Prices exact coherent/RUS repetition counts $R(\epsilon)$ against a stochastic failure budget and audits shot-by-shot paired outcomes (`child_only_failures == 0` + 99% Wilson upper confidence bound).
