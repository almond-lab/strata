"""Cross-cell trace/parity invariants and stochastic failure-budget pricing + shot-paired dominance audits."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .native import gf2_rank_fast
from .session import Session


def _gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
    """Compute an exact row-basis for the right nullspace ``{v in GF(2)^n : M @ v == 0}``."""
    M = (np.asarray(matrix, dtype=np.uint8) & 1).copy()
    m, n = M.shape
    pivot_col_for_row: list[int] = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        piv = -1
        for k in range(r, m):
            if M[k, c]:
                piv = k
                break
        if piv < 0:
            continue
        if piv != r:
            M[[r, piv]] = M[[piv, r]]
        for k in range(m):
            if k != r and M[k, c]:
                M[k] ^= M[r]
        pivot_col_for_row.append(c)
        r += 1

    piv_set = set(pivot_col_for_row)
    free_cols = [c for c in range(n) if c not in piv_set]
    basis = np.zeros((len(free_cols), n), dtype=np.uint8)
    for b_idx, fc in enumerate(free_cols):
        basis[b_idx, fc] = 1
        for r_idx, pc in enumerate(pivot_col_for_row):
            if M[r_idx, fc]:
                basis[b_idx, pc] = 1
    return basis


def scan_cross_cell_invariants(
    session: Session,
    cell_cuts: Sequence[int],
    *,
    wires: Sequence[int] | None = None,
    seed: int = 1,
    clean_wires: Sequence[int] = (),
) -> dict[str, object]:
    """Discover conserved GF(2) parity/trace invariants across repeated evaluation cells ``c_0, ..., c_L``.

    Any parity vector ``v in GF(2)^W`` satisfying ``<v, q(c_k)> == <v, q(c_0)> (mod 2)`` across all
    cells ``k = 1..L`` proves one redundant carrier degree of freedom across the cell loop: one wire in
    the support of ``v`` can be **eliminated / uncomputed across all ``L`` cells** and reconstructed at
    the exit of cell ``L`` from the conserved boundary parity!
    """
    if len(cell_cuts) < 2:
        raise ValueError("Provide at least two cell boundary cuts [c_0, c_1, ...]")
    traced = session.trace_cuts(cell_cuts, seed=seed, clean_wires=clean_wires)
    shots = int(traced["shots"])
    nq = session.num_qubits
    sel_wires = list(range(nq)) if wires is None else [int(w) for w in wires]

    mats = []
    for entry in traced["cuts"]:
        bits = np.unpackbits(entry["packed_qubits"], axis=1, bitorder="big")[:, :shots].T.astype(np.uint8)
        mats.append(bits[:, sel_wires])

    # Stack transitions (q(c_k) ^ q(c_0)) across all k = 1..L: shape ((L)*256, W)
    base = mats[0]
    deltas = np.vstack([m ^ base for m in mats[1:]])
    transition_rank = gf2_rank_fast(deltas)
    null_basis = _gf2_nullspace(deltas)

    # Separate trivial completely-untouched single-wire invariants from multi-wire cross-cell parities
    single_wire_idle: list[int] = []
    multi_wire_parities: list[list[int]] = []
    for row in null_basis:
        supp = [sel_wires[i] for i in np.flatnonzero(row)]
        if len(supp) == 1:
            single_wire_idle.append(supp[0])
        elif len(supp) >= 2:
            multi_wire_parities.append(supp)

    return {
        "cuts": [int(c) for c in cell_cuts],
        "inspected_wires": sel_wires,
        "transition_rank": transition_rank,
        "conserved_dimension": int(null_basis.shape[0]),
        "idle_wires": single_wire_idle,
        "conserved_parities": multi_wire_parities,
        "parkable_carrier_wires": len(multi_wire_parities),
    }


def price_failure_budget(
    per_attempt_fail_prob: float,
    *,
    max_failure_budget: float,
    attempt_toffolis: int,
    attempt_qubits: int,
    repeat_qubit_overhead: int = 0,
    safety_sigma: float = 0.0,
    num_verifier_shots: int = 1000,
) -> dict[str, object]:
    """Compute exact coherent/RUS repetition count ``R(eps)`` and static/expected Toffoli cost under ``max_failure_budget``.

    Accounts for finite-shot binomial tail margin ``safety_sigma`` when targeting stochastic scorers
    with ``num_verifier_shots`` draws.
    """
    p = float(per_attempt_fail_prob)
    eps = float(max_failure_budget)
    if not (0.0 <= p < 1.0):
        raise ValueError("per_attempt_fail_prob must be in [0, 1)")
    if not (0.0 < eps <= 1.0):
        raise ValueError("max_failure_budget must be in (0, 1]")

    # Adjust target failure rate downward if finite-shot safety_sigma margin is requested
    target_eps = eps
    if safety_sigma > 0.0 and num_verifier_shots > 0:
        # Solve p_eff + sigma * sqrt(p_eff*(1-p_eff)/N) <= eps
        lo, hi = 0.0, eps
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            upper = mid + safety_sigma * math.sqrt(mid * max(0.0, 1.0 - mid) / float(num_verifier_shots))
            if upper <= eps:
                lo = mid
            else:
                hi = mid
        target_eps = max(1e-15, lo)

    if p == 0.0 or p <= target_eps:
        R = 1
    else:
        R = max(1, int(math.ceil(math.log(target_eps) / math.log(p))))

    achieved_fail = p ** R
    static_toffolis = R * int(attempt_toffolis)
    # Expected Toffolis under early-exit conditional RUS (1 + p + p^2 + ... + p^{R-1})
    geom_factor = float(R) if p == 1.0 else (1.0 - (p ** R)) / max(1e-15, 1.0 - p)
    expected_toffolis = geom_factor * float(attempt_toffolis)
    peak_qubits = int(attempt_qubits) + max(0, R - 1) * int(repeat_qubit_overhead)

    return {
        "repetitions": R,
        "per_attempt_fail_prob": p,
        "max_failure_budget": eps,
        "effective_target_eps": target_eps,
        "achieved_fail_prob": achieved_fail,
        "static_toffolis": static_toffolis,
        "expected_toffolis": expected_toffolis,
        "peak_qubits": peak_qubits,
    }


def audit_shot_paired_dominance(
    parent_passed: Sequence[bool] | np.ndarray,
    child_passed: Sequence[bool] | np.ndarray,
    *,
    max_failure_budget: float | None = None,
) -> dict[str, object]:
    """Audit shot-by-shot paired outcomes between ``parent`` and ``child`` on identical verifier seeds.

    Enforces ``child_only_failures == 0`` for strict monotone stochastic safety and computes the exact
    Wilson 99% upper confidence bound on ``child``'s failure rate.
    """
    p_ok = np.asarray(parent_passed, dtype=bool).ravel()
    c_ok = np.asarray(child_passed, dtype=bool).ravel()
    if p_ok.size != c_ok.size or p_ok.size == 0:
        raise ValueError("parent_passed and child_passed must be non-empty arrays of identical length")

    n = int(p_ok.size)
    both_pass = int(np.sum(p_ok & c_ok))
    both_fail = int(np.sum((~p_ok) & (~c_ok)))
    child_only_fail = int(np.sum(p_ok & (~c_ok)))
    parent_only_fail = int(np.sum((~p_ok) & c_ok))
    child_fails = both_fail + child_only_fail
    p_hat = child_fails / float(n)

    # 99% Wilson upper confidence bound (z = 2.575829)
    z = 2.5758293035489004
    z2 = z * z
    wilson_ucb99 = (
        p_hat + z2 / (2.0 * n) + z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)
    ) / (1.0 + z2 / n)

    budget_ok = True if max_failure_budget is None else (p_hat <= float(max_failure_budget))
    return {
        "shots": n,
        "both_pass": both_pass,
        "both_fail": both_fail,
        "child_only_failures": child_only_fail,
        "parent_only_failures": parent_only_fail,
        "child_failure_rate": p_hat,
        "wilson_ucb99": wilson_ucb99,
        "shot_monotone": child_only_fail == 0,
        "accepted": (child_only_fail == 0) and budget_ok,
    }
