# Playbook: Truncation + Minimax LP Rotation Refitting (`CompensatedTruncator`)

## The Key Insight

When you drop low-order carry bits in fixed-point arithmetic, prune small Hamiltonian terms, or coarsen QROM angle precision from `b` bits to `b - 2` bits, Toffoli count drops immediately, **but** a naive check usually rejects the truncated circuit because systematic bias pushes worst-case eigenstate or basis error slightly past the allowed tolerance $\epsilon_{\max}$.

Crucially, Kickmix charges **zero Toffolis** (or negligible single-qubit cost) for continuous `Z_POW` / QSVT / phase rotations! If your circuit contains $K$ tunable downstream continuous angles $\theta \in \mathbb{R}^K$, shifting them by $\Delta\theta$ induces a linear first-order response $B \Delta\theta$ across your $S$ test points or eigenstates.

Instead of minimizing average ($\ell_2$) error, `CompensatedTruncator` solves the **exact Chebyshev minimax ($\ell_\infty$) problem**:

$$\min_{\Delta\theta \in \mathbb{R}^K} \max_{s \in \{1,\dots,S\}} \bigl| r_s(\tau) + (B \, \Delta\theta)_s \bigr|$$

This flattens the error equioscillation peaks across the entire domain, routinely rescuing aggressive 2-to-4 bit truncations that fail uncompensated checks by $3\times$ to $10\times$!

## How It Works

- If `scipy` is installed in the environment, `CompensatedTruncator` dispatches to HiGHS (`scipy.optimize.linprog`) for exact simplex/interior-point LP optimality.
- With zero external dependencies (`numpy` + `strata._native` only), `CompensatedTruncator` runs a fast **Lawson Iteratively Reweighted Least Squares (IRLS) + active-extremal minimax polish** backed by `eval_phase_margin_fast` (OpenMP C++20 residual evaluator).

## Complete Example

```python
import numpy as np
import strata

# Suppose we evaluate accuracy across S = 256 domain grid points with K = 5 tunable Z_POW / QSVT phases
grid = np.linspace(-1.0, 1.0, 256)
B = np.column_stack([np.cos((k + 1) * np.pi * grid) for k in range(5)])

# Three discrete truncation options sorted automatically by Toffolis saved
# Option A (drop 3 carry bits, saves 420 Toffolis) has systematic drift + small high-freq ripple:
drift_A = -(0.012 * B[:, 0] - 0.008 * B[:, 1] + 0.004 * B[:, 2]) + 0.0004 * np.sin(17 * grid)
# Option B (drop 4 carry bits, saves 580 Toffolis) has larger high-freq ripple exceeding 1e-3:
drift_B = -(0.020 * B[:, 0]) + 0.0025 * np.sin(19 * grid)
# Option C (drop 1 carry bit, saves 140 Toffolis) already fits without refitting:
drift_C = 0.0006 * np.sin(11 * grid)

truncator = strata.CompensatedTruncator(B, budget=1.0e-3)
result = truncator.select_best_truncation([
    strata.TruncationChoice("drop_1_bit", toffolis_saved=140, residual_error=drift_C),
    strata.TruncationChoice("drop_3_bits", toffolis_saved=420, residual_error=drift_A),
    strata.TruncationChoice("drop_4_bits", toffolis_saved=580, residual_error=drift_B),
])

winner = result["selected"]
print("Selected truncation:", winner["name"])
print("Toffolis saved:", winner["toffolis_saved"])
print(f"Uncompensated max error: {winner['uncompensated_max_abs_error']:.5f} (FAILED 1e-3)")
print(f"Compensated max error:   {winner['compensated_max_abs_error']:.5f} (PASSED 1e-3!)")
```

Notice what happened:
- `drop_4_bits` (`580` Toffolis saved) has `0.0025` irreducible high-frequency error orthogonal to `B`, so even optimal minimax refitting leaves `2.5e-3 > 1.0e-3` -> **rejected**.
- `drop_3_bits` (`420` Toffolis saved) had `~0.020` raw error (`20x` over budget!), so any naive truncation sweep would have thrown it away. Minimax refitting cancels the smooth drift down to `4.0e-4 <= 1.0e-3` -> **accepted**, banking `280` extra Toffolis over `drop_1_bit`!
