# Playbook: Classical-Quantum Numerical Co-Fitting (`strata.cofit`)

## When to Use This Playbook

Most quantum compilation pipelines treat classical numbers (Hamiltonian tensor factors, polynomial approximation coefficients, QROM lookup tables, or Givens state-preparation angles) as **frozen inputs** and ask the quantum compiler to synthesize those exact numbers.

In practice, the external verification contract almost never pins the intermediate numbers themselves. Instead, it checks a downstream physical observable, spectral norm, or target fidelity within an error budget $\epsilon$. That leaves a continuous **feasibility manifold** of equivalent or near-equivalent parameter vectors $x$ whose quantum synthesis costs differ dramatically!

`strata.cofit` provides four fast classical solvers that move $x$ inside that allowed region **before** gate emission:

1. **`GaugeSweep`**: Applies exact orthogonal rotations $A \leftarrow A Q$ (preserving $A A^T$) evaluated analytically at `arctan2` zero-crossings to shrink $\ell_1$ / squared-row-$\ell_1$ norms and plant exact structural zeros.
2. **`BoundarySlider`**: Slides parameters along the tangent nullspace $I - J_c^+ J_c$ of your accuracy constraints $c(x) = 0$, paired with damped Gauss-Newton restoration so every step stays strictly feasible.
3. **`DeadSlotPlanter` & `QuotientChainDP`**: Detects table entries or Givens-tree branches with zero sensitivity on the final state and fills those don't-care bits to collapse `GF(2)` rank and sequential XOR loader transitions.
4. **`CircuitAwareRounder`**: Locks quantized bit groups progressively while re-sliding the remaining continuous parameters to absorb rounding drift.

## Step-by-Step Recipes

### 1. Orthogonal Gauge Sweep (`GaugeSweep`)

In Double-Factorized or THC block encodings, a leaf tensor slice $A \in \mathbb{R}^{M \times R}$ represents $G = A A^T$. Any orthogonal $Q \in O(R)$ produces the **exact same operator** $(A Q)(A Q)^T = A A^T$, yet changes both the block-encoding $\lambda$-norm and the number of nonzero entries that QROM must load:

```python
import numpy as np
import strata

rng = np.random.default_rng(0)
A = rng.standard_normal((32, 6))

res = strata.GaugeSweep(metric="l1").sweep(A)
assert np.allclose(res.matrix @ res.matrix.T, A @ A.T, atol=1e-12)
print(f"L1 norm: {res.initial_norm:.3f} -> {res.final_norm:.3f}, planted zeros: {res.zeros_planted}")
```

### 2. Nullspace Boundary Sliding (`BoundarySlider`) + Progressive Rounding (`CircuitAwareRounder`)

Suppose vector $x \in \mathbb{R}^8$ must satisfy exact moment or spectral constraints $C x = b$, and your quantum circuit cost scales with $\|x\|_1$ plus quantization bit-width:

```python
import numpy as np
import strata

C = np.array([
    [1.0, 1.0, 0.5, -0.5, 0.0, 0.0, 0.25, -0.25],
    [0.0, 0.5, -1.0, 1.0, 1.0, -1.0, 0.5, 0.0],
])
x_init = np.array([1.23, -0.41, 0.88, 0.37, -0.92, 0.55, -0.64, 0.79])
target = C @ x_init

slider = strata.BoundarySlider(
    objective_fn=lambda x: float(np.sum(np.abs(x))),
    constraint_fn=lambda x: (C @ x) - target,
    jac_constraint=lambda _: C,
)

slid = slider.slide_multistart(x_init, n_starts=6, seed=42)
print(f"Nullspace dim: {slid.nullspace_dim}, L1: {slid.initial_objective:.4f} -> {slid.objective:.4f}")
print(f"Constraint violation: {slid.constraint_violation:.2e}")

# Now quantize to 1/64 grid while re-sliding unlocked coordinates after every 2-coordinate group!
rounder = strata.CircuitAwareRounder(slider, step_size=1.0 / 64.0, group_size=2)
q_res = rounder.round(slid.x, quantize_mask=[True]*6 + [False]*2, n_seeds=4)
print(f"Naive rounding error: {q_res['naive_violation']:.3e} vs compensated: {q_res['constraint_violation']:.3e}")
```

### 3. Givens-Tree Dead-Slot Planting & Differential XOR Loader DP

Whenever a state-preparation amplitude node has $\sin(\theta_k) = 0$ or $\cos(\theta_k) = 0$ (or a table index lies outside the active care set), its entire subtree of child angles is **mathematically dead**. Setting dead slots to zero naively forces sequential QROM loaders (`y[i] ^= word[i] ^ word[i-1]`) to flip bits off and back on. `QuotientChainDP` solves the exact bitwise shortest path in $O(N \cdot B)$ time:

```python
import strata

targets = [0b110101, 0b000000, 0b110111, 0b000000, 0b010111]
# Slot 1 and Slot 3 are dead (care_mask = 0); Slot 2 only cares about lower 4 bits (0b001111)
care_masks = [0b111111, 0b000000, 0b001111, 0b000000, 0b111111]

dp = strata.QuotientChainDP.optimize_chain(targets, care_masks, bit_width=6)
print("Naive XOR transitions:", dp["naive_xor_weight"], "-> Optimized:", dp["xor_weight"])
print("Saved controlled-XORs:", dp["toffoli_or_cx_saved"])
```
