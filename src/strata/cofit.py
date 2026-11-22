"""Classical-quantum numerical co-fitting, gauge sweeps, dead-slot planting, and truncation refitting."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np

from .native import eval_phase_margin_fast, gf2_rank_fast, lovasz_rank_fast


@dataclass
class SlideResult:
    x: np.ndarray
    objective: float
    initial_objective: float
    constraint_violation: float
    nullspace_dim: int
    iterations: int
    converged: bool


class BoundarySlider:
    """Tangent-nullspace Generalized Reduced Gradient (GRG) + damped Gauss-Newton restoration.

    Optimizes a circuit-cost proxy ``objective_fn(x)`` while keeping exact or soft
    mathematical boundary constraints ``constraint_fn(x) == 0`` satisfied at every step.
    """

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        constraint_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        *,
        jac_objective: Callable[[np.ndarray], np.ndarray] | None = None,
        jac_constraint: Callable[[np.ndarray], np.ndarray] | None = None,
        fd_eps: float = 1e-7,
        tol: float = 1e-10,
        max_iter: int = 100,
        restore_steps: int = 8,
        damping: float = 1e-10,
    ) -> None:
        self.objective_fn = objective_fn
        self.constraint_fn = constraint_fn
        self.jac_objective = jac_objective
        self.jac_constraint = jac_constraint
        self.fd_eps = float(fd_eps)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.restore_steps = int(restore_steps)
        self.damping = float(damping)

    def _eval_c(self, x: np.ndarray) -> np.ndarray:
        if self.constraint_fn is None:
            return np.zeros(0, dtype=np.float64)
        c = np.asarray(self.constraint_fn(x), dtype=np.float64).ravel()
        return c

    def _grad_f(self, x: np.ndarray) -> np.ndarray:
        if self.jac_objective is not None:
            return np.asarray(self.jac_objective(x), dtype=np.float64).ravel()
        n = x.size
        g = np.empty(n, dtype=np.float64)
        eps = self.fd_eps
        xp = x.copy()
        for i in range(n):
            orig = xp[i]
            xp[i] = orig + eps
            fp = float(self.objective_fn(xp))
            xp[i] = orig - eps
            fm = float(self.objective_fn(xp))
            xp[i] = orig
            g[i] = (fp - fm) / (2.0 * eps)
        return g

    def _jac_c(self, x: np.ndarray) -> np.ndarray:
        if self.constraint_fn is None:
            return np.zeros((0, x.size), dtype=np.float64)
        if self.jac_constraint is not None:
            J = np.asarray(self.jac_constraint(x), dtype=np.float64)
            return J.reshape(-1, x.size)
        c0 = self._eval_c(x)
        m = c0.size
        n = x.size
        if m == 0:
            return np.zeros((0, n), dtype=np.float64)
        J = np.empty((m, n), dtype=np.float64)
        eps = self.fd_eps
        xp = x.copy()
        for i in range(n):
            orig = xp[i]
            xp[i] = orig + eps
            cp = self._eval_c(xp)
            xp[i] = orig - eps
            cm = self._eval_c(xp)
            xp[i] = orig
            J[:, i] = (cp - cm) / (2.0 * eps)
        return J

    def restore(self, x: np.ndarray, *, free_mask: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Project ``x`` back onto the constraint manifold ``c(x) == 0`` via damped Gauss-Newton."""
        cur = np.asarray(x, dtype=np.float64).ravel().copy()
        if self.constraint_fn is None:
            return cur, 0.0
        mask = np.ones(cur.size, dtype=bool) if free_mask is None else np.asarray(free_mask, dtype=bool).ravel()
        for _ in range(self.restore_steps):
            c = self._eval_c(cur)
            viol = float(np.max(np.abs(c))) if c.size else 0.0
            if viol <= self.tol or not np.any(mask):
                return cur, viol
            J = self._jac_c(cur)[:, mask]
            # Minimum-norm Gauss-Newton step on free coordinates
            JJt = J @ J.T + self.damping * np.eye(J.shape[0], dtype=np.float64)
            try:
                delta_free = J.T @ np.linalg.solve(JJt, c)
            except np.linalg.LinAlgError:
                delta_free, *_ = np.linalg.lstsq(J, c, rcond=1e-11)
            # Backtracking on constraint norm if strongly nonlinear
            alpha = 1.0
            accepted = False
            for _ in range(8):
                cand = cur.copy()
                cand[mask] -= alpha * delta_free
                c_new = self._eval_c(cand)
                new_viol = float(np.max(np.abs(c_new))) if c_new.size else 0.0
                if new_viol < viol or new_viol <= self.tol:
                    cur = cand
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                break
        c_end = self._eval_c(cur)
        return cur, (float(np.max(np.abs(c_end))) if c_end.size else 0.0)

    def slide(
        self,
        x0: Sequence[float] | np.ndarray,
        *,
        free_mask: Sequence[bool] | np.ndarray | None = None,
        step_scale: float = 1.0,
    ) -> SlideResult:
        """Slide along the constraint nullspace to minimize ``objective_fn``."""
        x = np.asarray(x0, dtype=np.float64).ravel().copy()
        mask = np.ones(x.size, dtype=bool) if free_mask is None else np.asarray(free_mask, dtype=bool).ravel()
        x, viol = self.restore(x, free_mask=mask)
        f_init = float(self.objective_fn(x))
        f_cur = f_init
        null_dim = int(np.sum(mask))
        converged = False
        it = 0

        for it in range(1, self.max_iter + 1):
            g = self._grad_f(x)
            g_free = g[mask]
            J_full = self._jac_c(x)
            J = J_full[:, mask]
            if J.size > 0:
                _, svals, Vt = np.linalg.svd(J, full_matrices=False)
                rank = int(np.sum(svals > 1e-10))
                null_dim = int(mask.sum()) - rank
                if rank > 0:
                    Vr = Vt[:rank]
                    d_free = -(g_free - Vr.T @ (Vr @ g_free))
                else:
                    d_free = -g_free
            else:
                null_dim = int(mask.sum())
                d_free = -g_free

            grad_norm = float(np.linalg.norm(d_free))
            if grad_norm <= 1e-9 or null_dim <= 0:
                converged = True
                break

            # Normalize large gradients for stable manifold retraction
            scale = float(step_scale) / max(1.0, grad_norm)
            alpha = scale
            improved = False
            slope = float(np.dot(g_free, d_free))
            for _ in range(16):
                trial = x.copy()
                trial[mask] += alpha * d_free
                trial, t_viol = self.restore(trial, free_mask=mask)
                if t_viol <= max(1e-7, 10.0 * self.tol):
                    f_trial = float(self.objective_fn(trial))
                    if f_trial < f_cur + 1e-4 * alpha * slope - 1e-13:
                        if abs(f_cur - f_trial) <= 1e-11 * max(1.0, abs(f_cur)):
                            converged = True
                        x = trial
                        f_cur = f_trial
                        viol = t_viol
                        improved = True
                        break
                alpha *= 0.5
            if not improved or converged:
                converged = converged or (grad_norm <= 1e-6)
                break

        return SlideResult(
            x=x,
            objective=f_cur,
            initial_objective=f_init,
            constraint_violation=viol,
            nullspace_dim=max(0, null_dim),
            iterations=it,
            converged=converged,
        )

    def slide_multistart(
        self,
        x0: Sequence[float] | np.ndarray,
        *,
        n_starts: int = 6,
        perturb_scale: float = 0.1,
        seed: int = 0,
    ) -> SlideResult:
        """Explore multiple tangent-nullspace perturbations and return the best feasible minimum."""
        base = np.asarray(x0, dtype=np.float64).ravel()
        best = self.slide(base)
        rng = np.random.default_rng(seed)
        J = self._jac_c(best.x)
        if J.size > 0:
            _, svals, Vt = np.linalg.svd(J, full_matrices=True)
            rank = int(np.sum(svals > 1e-10))
            null_basis = Vt[rank:]
        else:
            null_basis = np.eye(base.size, dtype=np.float64)
        if null_basis.shape[0] == 0:
            return best
        for _ in range(max(0, int(n_starts) - 1)):
            z = rng.standard_normal(null_basis.shape[0]) * float(perturb_scale)
            cand0 = best.x + null_basis.T @ z
            res = self.slide(cand0)
            if res.constraint_violation <= max(1e-7, 10.0 * self.tol) and res.objective < best.objective:
                best = res
        return best


@dataclass
class GaugeResult:
    matrix: np.ndarray
    transform: np.ndarray
    initial_norm: float
    final_norm: float
    zeros_planted: int
    passes: int


class GaugeSweep:
    """Analytical zero-crossing orthogonal pair-rotation and signed-permutation gauge optimizer.

    Given ``A`` of shape ``(M, R)``, finds an orthogonal matrix ``Q`` of shape ``(R, R)``
    so that ``B = A @ Q`` preserves ``B @ B.T == A @ A.T`` while minimizing entrywise L1
    or squared row-L1 norm (such as DF-THC / block-encoding leaf channel norms) and
    planting exact structural zeros at ``arctan2`` kinks.
    """

    def __init__(self, *, metric: str = "l1", zero_tol: float = 1e-12, max_passes: int = 8) -> None:
        if metric not in {"l1", "row_l1_sq"}:
            raise ValueError("metric must be 'l1' or 'row_l1_sq'")
        self.metric = metric
        self.zero_tol = float(zero_tol)
        self.max_passes = int(max_passes)

    def _score(self, A: np.ndarray) -> float:
        absA = np.abs(A)
        if self.metric == "l1":
            return float(np.sum(absA))
        row_sums = np.sum(absA, axis=1)
        return float(np.sum(row_sums * row_sums))

    def sweep(self, matrix: np.ndarray) -> GaugeResult:
        A = np.asarray(matrix, dtype=np.float64).copy()
        if A.ndim != 2:
            raise ValueError("GaugeSweep requires a 2D array (M, R)")
        _, R = A.shape
        Q = np.eye(R, dtype=np.float64)
        init_norm = self._score(A)
        cur_norm = init_norm
        half_pi = 0.5 * math.pi
        passes_run = 0

        for p in range(1, self.max_passes + 1):
            passes_run = p
            improved_pass = False
            for u in range(R):
                for v in range(u + 1, R):
                    col_u = A[:, u]
                    col_v = A[:, v]
                    active = (np.abs(col_u) + np.abs(col_v)) > self.zero_tol
                    if not np.any(active):
                        continue
                    au = col_u[active]
                    av = col_v[active]
                    # Exact zero-crossing angles modulo pi/2 (since +/- swaps & signs leave L1 invariant)
                    ang1 = np.mod(np.arctan2(-au, av), half_pi)
                    ang2 = np.mod(np.arctan2(av, au), half_pi)
                    kinks = np.unique(np.concatenate(([0.0], ang1, ang2)))
                    if self.metric == "row_l1_sq" and kinks.size > 1:
                        mids = 0.5 * (kinks[:-1] + kinks[1:])
                        thetas = np.concatenate((kinks, mids))
                    else:
                        thetas = kinks

                    cos_t = np.cos(thetas)[None, :]
                    sin_t = np.sin(thetas)[None, :]
                    u_rot = col_u[:, None] * cos_t + col_v[:, None] * sin_t
                    v_rot = -col_u[:, None] * sin_t + col_v[:, None] * cos_t

                    if self.metric == "l1":
                        cand_scores = np.sum(np.abs(u_rot) + np.abs(v_rot), axis=0)
                        base_pair = float(np.sum(np.abs(col_u) + np.abs(col_v)))
                    else:
                        rest_row = np.sum(np.abs(A), axis=1) - np.abs(col_u) - np.abs(col_v)
                        tot_rows = rest_row[:, None] + np.abs(u_rot) + np.abs(v_rot)
                        cand_scores = np.sum(tot_rows * tot_rows, axis=0)
                        base_pair = cur_norm

                    best_idx = int(np.argmin(cand_scores))
                    best_val = float(cand_scores[best_idx])
                    best_theta = float(thetas[best_idx])
                    if best_theta != 0.0 and best_val < base_pair - 1e-12:
                        c = math.cos(best_theta)
                        s = math.sin(best_theta)
                        nu = col_u * c + col_v * s
                        nv = -col_u * s + col_v * c
                        nu[np.abs(nu) <= self.zero_tol] = 0.0
                        nv[np.abs(nv) <= self.zero_tol] = 0.0
                        A[:, u] = nu
                        A[:, v] = nv
                        qu = Q[:, u].copy()
                        qv = Q[:, v].copy()
                        Q[:, u] = qu * c + qv * s
                        Q[:, v] = -qu * s + qv * c
                        cur_norm = self._score(A)
                        improved_pass = True
            if not improved_pass:
                break

        # Canonicalize column ordering by descending column L1 norm
        col_norms = np.sum(np.abs(A), axis=0)
        order = np.argsort(-col_norms, kind="stable")
        A = A[:, order]
        Q = Q[:, order]
        zeros = int(np.sum(np.abs(A) <= self.zero_tol))
        return GaugeResult(
            matrix=A,
            transform=Q,
            initial_norm=init_norm,
            final_norm=self._score(A),
            zeros_planted=zeros,
            passes=passes_run,
        )


class DeadSlotPlanter:
    """Detects zero-sensitivity table/tree slots and plants rank-reducing bit relations."""

    @staticmethod
    def detect_dead_slots(
        eval_fn: Callable[[np.ndarray], float | np.ndarray],
        values: np.ndarray,
        *,
        probe_delta: float = 0.125,
        tol: float = 1e-12,
    ) -> np.ndarray:
        """Return a boolean mask of entries in ``values`` with strictly zero downstream sensitivity."""
        arr = np.asarray(values, dtype=np.float64)
        base = np.asarray(eval_fn(arr), dtype=np.float64)
        dead = np.zeros(arr.shape, dtype=bool)
        work = arr.copy()
        for idx in np.ndindex(arr.shape):
            orig = work[idx]
            work[idx] = orig + probe_delta
            diff1 = np.max(np.abs(np.asarray(eval_fn(work), dtype=np.float64) - base))
            if diff1 <= tol:
                work[idx] = orig - probe_delta
                diff2 = np.max(np.abs(np.asarray(eval_fn(work), dtype=np.float64) - base))
                dead[idx] = bool(diff2 <= tol)
            work[idx] = orig
        return dead

    @staticmethod
    def detect_givens_tree_dead_slots(angles: np.ndarray, *, zero_tol: float = 1e-12) -> np.ndarray:
        """Detect dead rotation slots in a binary Givens state-preparation tree of length ``2^n - 1``.

        In heap indexing ``k in [0, 2^n - 1)``, node ``k`` receives incoming amplitude ``amp[k]``
        and splits it into ``left = amp[k] * cos(angles[k])`` (child ``2k + 1``) and
        ``right = amp[k] * sin(angles[k])`` (child ``2k + 2``). Whenever ``|amp[k]| <= zero_tol``,
        ``angles[k]`` has strictly zero effect on the prepared quantum state and can be set to
        any bit pattern that compresses the QROM Multiplexer table!
        """
        ang = np.asarray(angles, dtype=np.float64).ravel()
        m = ang.size
        amps = np.zeros(m, dtype=np.float64)
        dead = np.zeros(m, dtype=bool)
        if m == 0:
            return dead
        amps[0] = 1.0
        for k in range(m):
            if abs(amps[k]) <= zero_tol:
                dead[k] = True
                continue
            left = 2 * k + 1
            right = 2 * k + 2
            if left < m:
                amps[left] = amps[k] * math.cos(float(ang[k]))
            if right < m:
                amps[right] = amps[k] * math.sin(float(ang[k]))
        return dead

    @staticmethod
    def plant_bit_relations(
        words: Sequence[int] | np.ndarray,
        dead_mask: Sequence[bool] | np.ndarray,
        bit_width: int,
    ) -> dict[str, object]:
        """Fill dead table rows to preserve linear bit-plane low-rank and zero-XOR transitions.

        If ``N = 2^a`` is a power of two, plants parity-folding copies from paired live addresses
        (collapsing Multiplexer / Reed-Muller monomials); otherwise repeats neighboring live words.
        """
        w = np.asarray(words, dtype=np.uint64).ravel().copy()
        dead = np.asarray(dead_mask, dtype=bool).ravel()
        if w.size != dead.size:
            raise ValueError("words and dead_mask must have identical length")
        n = w.size
        b = int(bit_width)

        def _bit_matrix(arr: np.ndarray) -> np.ndarray:
            shifts = np.arange(b, dtype=np.uint64)[None, :]
            return ((arr[:, None] >> shifts) & np.uint64(1)).astype(np.uint8)

        before_mat = _bit_matrix(w)
        before_gf2_rank = gf2_rank_fast(before_mat)
        before_transitions = int(np.sum(np.bitwise_count(w[1:] ^ w[:-1]))) if n > 1 else 0

        # If power-of-two table, copy across highest-stride live partner so upper address bit becomes don't-care
        is_pow2 = n > 0 and (n & (n - 1)) == 0
        if is_pow2:
            stride = n >> 1
            while stride >= 1:
                for i in range(n):
                    if dead[i] and not dead[i ^ stride]:
                        w[i] = w[i ^ stride]
                        dead[i] = False
                stride >>= 1

        # Sweep forward and backward for any remaining dead islands to zero out adjacent XOR transitions
        for i in range(1, n):
            if dead[i] and not dead[i - 1]:
                w[i] = w[i - 1]
                dead[i] = False
        for i in range(n - 2, -1, -1):
            if dead[i] and not dead[i + 1]:
                w[i] = w[i + 1]
                dead[i] = False
        w[dead] = 0

        after_mat = _bit_matrix(w)
        after_gf2_rank = gf2_rank_fast(after_mat)
        after_transitions = int(np.sum(np.bitwise_count(w[1:] ^ w[:-1]))) if n > 1 else 0
        return {
            "words": w,
            "before_gf2_rank": before_gf2_rank,
            "after_gf2_rank": after_gf2_rank,
            "before_xor_weight": before_transitions,
            "after_xor_weight": after_transitions,
        }


class CircuitAwareRounder:
    """Progressive coordinate-group quantizer with live tangent-nullspace drift compensation."""

    def __init__(
        self,
        slider: BoundarySlider,
        *,
        step_size: float,
        group_size: int = 4,
    ) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.slider = slider
        self.step_size = float(step_size)
        self.group_size = max(1, int(group_size))

    def round(
        self,
        x0: Sequence[float] | np.ndarray,
        *,
        quantize_mask: Sequence[bool] | np.ndarray | None = None,
        n_seeds: int = 1,
        seed: int = 0,
    ) -> dict[str, object]:
        base = np.asarray(x0, dtype=np.float64).ravel()
        qmask = np.ones(base.size, dtype=bool) if quantize_mask is None else np.asarray(quantize_mask, dtype=bool).ravel()
        q_indices = np.flatnonzero(qmask)
        rng = np.random.default_rng(seed)

        best_x = None
        best_viol = float("inf")
        best_obj = float("inf")

        for s in range(max(1, int(n_seeds))):
            order = q_indices.copy()
            if s > 0:
                rng.shuffle(order)
            cur = base.copy()
            free = np.ones(base.size, dtype=bool)
            for pos in range(0, order.size, self.group_size):
                grp = order[pos : pos + self.group_size]
                cur[grp] = np.round(cur[grp] / self.step_size) * self.step_size
                free[grp] = False
                if np.any(free):
                    res = self.slider.slide(cur, free_mask=free)
                    cur, _ = self.slider.restore(res.x, free_mask=free)
            c = self.slider._eval_c(cur)
            viol = float(np.max(np.abs(c))) if c.size else 0.0
            obj = float(self.slider.objective_fn(cur))
            if (viol, obj) < (best_viol, best_obj):
                best_x = cur
                best_viol = viol
                best_obj = obj

        naive = base.copy()
        naive[qmask] = np.round(naive[qmask] / self.step_size) * self.step_size
        c_naive = self.slider._eval_c(naive)
        naive_viol = float(np.max(np.abs(c_naive))) if c_naive.size else 0.0
        return {
            "x": best_x,
            "objective": best_obj,
            "constraint_violation": best_viol,
            "naive_violation": naive_viol,
            "improvement_ratio": (naive_viol / max(best_viol, 1e-16)) if naive_viol > 0 else 1.0,
        }


class QuotientChainDP:
    """Exact bitwise DP minimizing differential XOR weight across sequential loader states modulo dead bits.

    Because bitwise Hamming weight decomposes independently per bit plane ``b in [0, B)``,
    finding representatives ``y[i]`` satisfying ``(y[i] ^ target[i]) & care_mask[i] == 0``
    that minimize ``sum_{i=1}^{N-1} popcount(y[i] ^ y[i-1])`` (plus optional initial/reset cost
    from ``0``) is solved **globally and exactly in O(N * B)** time!
    """

    @staticmethod
    def optimize_chain(
        targets: Sequence[int] | np.ndarray,
        care_masks: Sequence[int] | np.ndarray,
        bit_width: int,
        *,
        boundary_zero: bool = True,
    ) -> dict[str, object]:
        t = np.asarray(targets, dtype=np.uint64).ravel()
        m = np.asarray(care_masks, dtype=np.uint64).ravel()
        if t.size != m.size:
            raise ValueError("targets and care_masks must have identical length")
        n = t.size
        b_width = int(bit_width)
        if n == 0:
            return {"representatives": np.zeros(0, dtype=np.uint64), "xor_weight": 0, "naive_xor_weight": 0}

        def _chain_cost(arr: np.ndarray) -> int:
            inner = int(np.sum(np.bitwise_count(arr[1:] ^ arr[:-1]))) if n > 1 else 0
            if boundary_zero:
                inner += int(arr[0].bit_count()) + int(arr[-1].bit_count())
            return inner

        naive_masked = t & m
        naive_cost = _chain_cost(naive_masked)

        out = np.zeros(n, dtype=np.uint64)
        inf = 10**12
        back = np.zeros((n, 2), dtype=np.uint8)
        for bit in range(b_width):
            t_bit = ((t >> np.uint64(bit)) & np.uint64(1)).astype(np.int32)
            c_bit = ((m >> np.uint64(bit)) & np.uint64(1)).astype(np.int32)

            dp0 = inf
            dp1 = inf
            for v in (0, 1):
                if c_bit[0] == 0 or v == t_bit[0]:
                    cost0 = v if boundary_zero else 0
                    if v == 0:
                        dp0 = cost0
                    else:
                        dp1 = cost0

            for i in range(1, n):
                ndp0, ndp1 = inf, inf
                for v in (0, 1):
                    if c_bit[i] == 1 and v != t_bit[i]:
                        continue
                    c_from_0 = dp0 + (0 ^ v)
                    c_from_1 = dp1 + (1 ^ v)
                    if c_from_0 <= c_from_1:
                        best_c, prev = c_from_0, 0
                    else:
                        best_c, prev = c_from_1, 1
                    if v == 0:
                        ndp0 = best_c
                        back[i, 0] = prev
                    else:
                        ndp1 = best_c
                        back[i, 1] = prev
                dp0, dp1 = ndp0, ndp1

            if boundary_zero:
                dp1 += 1
            cur_v = 0 if dp0 <= dp1 else 1
            if cur_v:
                out[n - 1] |= np.uint64(1) << np.uint64(bit)
            for i in range(n - 1, 0, -1):
                cur_v = int(back[i, cur_v])
                if cur_v:
                    out[i - 1] |= np.uint64(1) << np.uint64(bit)

        opt_cost = _chain_cost(out)
        return {
            "representatives": out,
            "xor_weight": opt_cost,
            "naive_xor_weight": naive_cost,
            "toffoli_or_cx_saved": naive_cost - opt_cost,
        }


@dataclass
class TruncationChoice:
    name: str
    toffolis_saved: int
    residual_error: np.ndarray
    metadata: dict[str, object] | None = None


class CompensatedTruncator:
    """Minimax (L-infinity Chebyshev) downstream rotation refitter for aggressive circuit truncations.

    Given a Jacobian/response basis ``B`` of shape ``(S, K)`` mapping continuous single-qubit
    angle shifts ``delta_theta`` to observable/phase shifts across ``S`` eigenstates or basis
    samples, solves::

        min_{delta_theta} max_{s} |residual_s + (B @ delta_theta)_s|

    Uses ``scipy.optimize.linprog`` (HiGHS) when installed, and otherwise uses a fast pure-NumPy
    Lawson IRLS + Primal-Dual Chebyshev solver verified via ``eval_phase_margin_fast``.
    """

    def __init__(self, response_basis: np.ndarray, *, budget: float) -> None:
        B = np.asarray(response_basis, dtype=np.float64)
        if B.ndim != 2:
            raise ValueError("response_basis must have shape (S, K)")
        self.B = np.ascontiguousarray(B)
        self.budget = float(budget)

    @staticmethod
    def _solve_chebyshev_numpy(B: np.ndarray, r: np.ndarray, *, max_iter: int = 250) -> np.ndarray:
        """Pure-NumPy Lawson Iteratively Reweighted Least Squares + subgradient polish for L-infinity."""
        S, K = B.shape
        if K == 0 or S == 0:
            return np.zeros(K, dtype=np.float64)
        w = np.ones(S, dtype=np.float64) / float(S)
        best_theta = np.zeros(K, dtype=np.float64)
        best_inf = float(np.max(np.abs(r)))

        # Phase 1: Lawson IRLS (converges rapidly to Chebyshev equioscillation)
        for _ in range(max_iter):
            sw = np.sqrt(w)[:, None]
            Bw = B * sw
            rw = -r * sw[:, 0]
            theta, *_ = np.linalg.lstsq(Bw, rw, rcond=1e-12)
            err = np.abs(r + B @ theta)
            cur_inf = float(np.max(err))
            if cur_inf < best_inf:
                best_inf = cur_inf
                best_theta = theta.copy()
            if cur_inf <= 1e-14:
                return best_theta
            # Smooth Lawson exponent update with floor for numerical stability
            w_new = w * np.maximum(err / cur_inf, 1e-3) ** 1.5
            s_w = float(np.sum(w_new))
            if s_w <= 0.0:
                break
            w_new /= s_w
            if float(np.max(np.abs(w_new - w))) <= 1e-9:
                break
            w = w_new

        # Phase 2: Active-extremal Polyak/minimax polish around best_theta
        theta = best_theta.copy()
        col_norm = max(1.0, float(np.linalg.norm(B, ord=2)))
        step = 0.25 * best_inf / (col_norm * col_norm)
        for t in range(120):
            signed = r + B @ theta
            abs_e = np.abs(signed)
            cur_inf = float(np.max(abs_e))
            if cur_inf < best_inf:
                best_inf = cur_inf
                best_theta = theta.copy()
            # Softmax subgradient over near-extremal rows
            tau = max(1e-12, 0.02 * cur_inf / (1.0 + 0.1 * t))
            logits = (abs_e - cur_inf) / tau
            probs = np.exp(np.clip(logits, -60.0, 0.0))
            probs /= np.sum(probs)
            grad = B.T @ (probs * np.sign(signed))
            gnorm = float(np.linalg.norm(grad))
            if gnorm <= 1e-13:
                break
            theta -= (step / math.sqrt(1.0 + 0.05 * t)) * (grad / gnorm) * col_norm
        return best_theta

    def refit_minimax(self, residual_error: Sequence[float] | np.ndarray) -> dict[str, object]:
        r = np.ascontiguousarray(np.asarray(residual_error, dtype=np.float64).ravel())
        S, K = self.B.shape
        if r.size != S:
            raise ValueError(f"residual_error length {r.size} does not match basis rows {S}")
        uncomp_max = float(np.max(np.abs(r))) if S > 0 else 0.0

        theta_opt = None
        try:
            from scipy.optimize import linprog  # type: ignore[import-not-found]

            # Variables: [delta_theta_0 .. delta_theta_{K-1}, t]
            c = np.zeros(K + 1, dtype=np.float64)
            c[-1] = 1.0
            A_ub = np.vstack(
                [
                    np.hstack([self.B, -np.ones((S, 1), dtype=np.float64)]),
                    np.hstack([-self.B, -np.ones((S, 1), dtype=np.float64)]),
                ]
            )
            b_ub = np.concatenate([-r, r])
            bounds = [(None, None)] * K + [(0.0, None)]
            lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if lp.success and lp.x is not None:
                theta_opt = np.asarray(lp.x[:K], dtype=np.float64)
        except ImportError:
            theta_opt = None

        if theta_opt is None:
            theta_opt = self._solve_chebyshev_numpy(self.B, r)

        comp_vec = np.abs(r + self.B @ theta_opt)
        comp_max = float(np.max(comp_vec)) if S > 0 else 0.0
        rms = float(np.sqrt(np.mean(comp_vec * comp_vec))) if S > 0 else 0.0
        margin = self.budget - comp_max
        return {
            "delta_theta": theta_opt,
            "uncompensated_max_abs_error": uncomp_max,
            "compensated_max_abs_error": comp_max,
            "rms_error": rms,
            "margin": margin,
            "feasible": comp_max <= self.budget,
        }

    def select_best_truncation(self, choices: Sequence[TruncationChoice]) -> dict[str, object]:
        """Screen truncation candidates from largest Toffoli saving downward using minimax refitting."""
        ordered = sorted(choices, key=lambda c: (-int(c.toffolis_saved), c.name))
        evaluations = []
        winner = None
        for ch in ordered:
            # Fast uncompensated check first; run full minimax refit if needed or to certify margin
            refit = self.refit_minimax(ch.residual_error)
            rec = {
                "name": ch.name,
                "toffolis_saved": int(ch.toffolis_saved),
                **refit,
            }
            evaluations.append(rec)
            if winner is None and rec["feasible"]:
                winner = rec
        return {
            "budget": self.budget,
            "selected": winner,
            "evaluations": evaluations,
        }
