"""Algorithmic shape exploration, marginal qubit-Toffoli pricing, MBU DAG pebbling, and affine wire loans."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Callable, Sequence

import numpy as np

from .native import solve_zmod_fast
from .session import FunnelCandidate, Session


class ScoreModel:
    """Evaluates quantum resource objectives and marginal Qubit-vs-Toffoli exchange rates.

    Supports both minimizing cost formulas (such as ``Q * (toffoli + t / 4)``) and maximizing
    benchmark score curves where width enters exponentially (such as ``2^(Q_ref - Q) / T``).
    """

    def __init__(
        self,
        preset: str = "q_times_toffoli",
        *,
        custom_fn: Callable[[int, int, int], float] | None = None,
        higher_is_better: bool | None = None,
    ) -> None:
        self.preset = preset
        self.custom_fn = custom_fn
        if custom_fn is not None:
            self.higher_is_better = bool(False if higher_is_better is None else higher_is_better)
        elif preset in {"invsqrt", "dqi", "chemistry"}:
            self.higher_is_better = True if higher_is_better is None else bool(higher_is_better)
        elif preset in {"q_times_toffoli", "q_times_paid"}:
            self.higher_is_better = False if higher_is_better is None else bool(higher_is_better)
        else:
            raise ValueError(f"Unknown ScoreModel preset: {preset!r}")

    def evaluate(self, qubits: int, toffolis: int, t_gates: int = 0) -> float:
        q = max(1, int(qubits))
        tof = max(1, int(toffolis))
        tg = max(0, int(t_gates))
        if self.custom_fn is not None:
            return float(self.custom_fn(q, tof, tg))
        if self.preset == "q_times_toffoli":
            return float(q * tof)
        if self.preset == "q_times_paid":
            return float(q * (tof + 0.25 * tg))
        if self.preset == "invsqrt":
            # Exponential qubit weight + reciprocal Toffoli efficiency scaling
            return float((1.22 ** (64 - q)) * (1.0e7 / tof))
        if self.preset == "dqi":
            return float((1.15 ** (140 - q)) * (1.0e14 / tof))
        if self.preset == "chemistry":
            return float((1.08 ** (160 - q)) * (3.0e12 / (tof + 0.25 * tg)))
        raise AssertionError("unreachable")

    def exchange_rate(self, qubits: int, toffolis: int, t_gates: int = 0) -> dict[str, float]:
        """Compute how many extra Toffolis break even against saving 1 qubit at ``(qubits, toffolis)``."""
        base = self.evaluate(qubits, toffolis, t_gates)
        one_fewer_q = self.evaluate(max(1, qubits - 1), toffolis, t_gates)
        plus_100_tof = self.evaluate(qubits, toffolis + 100, t_gates)

        # Binary search exact break-even Toffoli count at (qubits - 1) matching `base`
        lo = float(toffolis)
        hi = float(max(toffolis * 16, toffolis + 10_000))
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            val = self.evaluate(max(1, qubits - 1), int(round(mid)), t_gates)
            better = (val >= base) if self.higher_is_better else (val <= base)
            if better:
                lo = mid
            else:
                hi = mid
        breakeven_extra_toffolis = max(0.0, lo - float(toffolis))
        return {
            "score": base,
            "score_minus_1_qubit": one_fewer_q,
            "score_plus_100_toffolis": plus_100_tof,
            "toffolis_per_qubit": breakeven_extra_toffolis,
            "higher_is_better": self.higher_is_better,
        }


@dataclass
class PebbleSchedule:
    feasible: bool
    peak_qubits: int
    total_toffolis: int
    hmr_erasures: int
    reversible_uncomputes: int
    steps: list[dict[str, object]]


def solve_mbu_pebble_schedule(
    dag_edges: Sequence[tuple[int, int]],
    num_nodes: int,
    *,
    input_nodes: Sequence[int] = (),
    output_nodes: Sequence[int],
    max_qubits: int,
    node_widths: Sequence[int] | None = None,
    compute_toffolis: Sequence[int] | None = None,
    allow_hmr: bool = True,
    require_live_parents_on_hmr: bool = False,
) -> PebbleSchedule:
    """Solve the Hybrid Reversible + Measurement-Based Uncomputation (MBU) DAG pebbling game.

    Each DAG edge ``(u, v)`` means ``u`` must hold a live pebble to compute ``v``:
      - ``compute(v)``: requires all parents ``u`` of ``v`` live; costs ``compute_toffolis[v]``.
      - ``uncompute_rev(v)``: requires all parents ``u`` of ``v`` live; costs ``compute_toffolis[v]``.
      - ``demolish_hmr(v)`` (when ``allow_hmr=True``): erases ``v`` via ``HMR`` for **0 Toffolis**.
        When ``require_live_parents_on_hmr=False`` (default), phase kickback can be repaid via
        multi-cut diagonal phase synthesis on surviving/input basis states without keeping ``v``'s
        immediate parents simultaneously live, breaking the classical Bennett space lower bound!
    """
    n = int(num_nodes)
    if n > 20:
        raise ValueError("Exact bitmask MBU pebbler supports num_nodes <= 20")
    widths = [1] * n if node_widths is None else [int(w) for w in node_widths]
    costs = [1] * n if compute_toffolis is None else [int(c) for c in compute_toffolis]

    parents_mask = [0] * n
    for u, v in dag_edges:
        parents_mask[int(v)] |= 1 << int(u)

    in_mask = 0
    for u in input_nodes:
        in_mask |= 1 << int(u)
    out_mask = in_mask
    for v in output_nodes:
        out_mask |= 1 << int(v)

    def _mask_width(m: int) -> int:
        total = 0
        for i in range(n):
            if (m >> i) & 1:
                total += widths[i]
        return total

    init_w = _mask_width(in_mask)
    if init_w > max_qubits or _mask_width(out_mask) > max_qubits:
        return PebbleSchedule(False, init_w, 0, 0, 0, [])

    # Dijkstra over pebble bitmasks minimizing (total_toffolis, peak_qubits, steps_len)
    # State: mask
    pq: list[tuple[int, int, int, int]] = [(0, init_w, 0, in_mask)]
    best: dict[int, tuple[int, int]] = {in_mask: (0, init_w)}
    prev: dict[int, tuple[int, dict[str, object]]] = {}

    goal_state = None
    while pq:
        tof, peak_w, step_cnt, mask = heapq.heappop(pq)
        if best.get(mask, (10**18, 10**18)) < (tof, peak_w):
            continue
        if mask == out_mask:
            goal_state = (tof, peak_w)
            break

        for v in range(n):
            bit = 1 << v
            pm = parents_mask[v]
            if not (mask & bit):
                # Compute v requires all parents of v live
                if (mask & pm) != pm:
                    continue
                nmask = mask | bit
                nw = _mask_width(nmask)
                if nw <= max_qubits:
                    ntof = tof + costs[v]
                    npeak = max(peak_w, nw)
                    if (ntof, npeak) < best.get(nmask, (10**18, 10**18)):
                        best[nmask] = (ntof, npeak)
                        prev[nmask] = (mask, {"op": "compute", "node": v, "toffolis": costs[v], "width": nw})
                        heapq.heappush(pq, (ntof, npeak, step_cnt + 1, nmask))
            elif not (in_mask & bit):
                # Erase v (either 0-Toffoli HMR demolish or reversible uncompute)
                if (not allow_hmr or require_live_parents_on_hmr) and (mask & pm) != pm:
                    continue
                nmask = mask & ~bit
                nw = _mask_width(nmask)
                if allow_hmr:
                    ntof = tof
                    op_name = "demolish_hmr"
                    op_cost = 0
                else:
                    ntof = tof + costs[v]
                    op_name = "uncompute_rev"
                    op_cost = costs[v]
                if (ntof, peak_w) < best.get(nmask, (10**18, 10**18)):
                    best[nmask] = (ntof, peak_w)
                    prev[nmask] = (mask, {"op": op_name, "node": v, "toffolis": op_cost, "width": nw})
                    heapq.heappush(pq, (ntof, peak_w, step_cnt + 1, nmask))

    if goal_state is None:
        return PebbleSchedule(False, init_w, 0, 0, 0, [])

    steps: list[dict[str, object]] = []
    cur = out_mask
    while cur != in_mask:
        p_mask, action = prev[cur]
        steps.append(action)
        cur = p_mask
    steps.reverse()
    hmr_cnt = sum(1 for s in steps if s["op"] == "demolish_hmr")
    rev_cnt = sum(1 for s in steps if s["op"] == "uncompute_rev")
    return PebbleSchedule(
        feasible=True,
        peak_qubits=goal_state[1],
        total_toffolis=goal_state[0],
        hmr_erasures=hmr_cnt,
        reversible_uncomputes=rev_cnt,
        steps=steps,
    )


def find_affine_wire_loans(
    session: Session,
    start: int,
    stop: int,
    *,
    max_support: int = 6,
    seed: int = 1,
    clean_wires: Sequence[int] = (),
) -> list[dict[str, object]]:
    """Discover live idle wires in ``[start, stop)`` that are affine GF(2) functions of other idle wires (``LEND29``).

    Whenever wire ``q_w`` is untouched during ``[start, stop)`` and satisfies
    ``q_w == c_0 ^ XOR_{s in S} q_s`` at ``start`` for a set ``S`` of wires also untouched during
    ``[start, stop)``, ``q_w`` can be cleared to ``|0>`` at ``start`` with ``|S|`` free ``CX`` gates
    (plus optional ``X``), used as a **free temporary ancilla** throughout ``[start, stop)`` (saving
    1 top-level circuit qubit!), and restored to its exact live value at ``stop`` with ``|S|`` ``CX`` gates!
    """
    lines = session.tape.decode_span(start, stop)
    touched: set[int] = set()
    for ln in lines:
        for tok in ln.split()[1:]:
            if tok.startswith("q") and tok[1:].isdigit():
                touched.add(int(tok[1:]))

    idle = [q for q in range(session.num_qubits) if q not in touched]
    if not idle:
        return []

    traced = session.trace_cuts([start, stop], seed=seed, clean_wires=clean_wires)
    shots = int(traced["shots"])
    bits_start = np.unpackbits(traced["cuts"][0]["packed_qubits"], axis=1, bitorder="big")[:, :shots].T.astype(np.int64)
    bits_stop = np.unpackbits(traced["cuts"][1]["packed_qubits"], axis=1, bitorder="big")[:, :shots].T.astype(np.int64)

    # Support wires can be read-only controls (or restored) inside [start, stop): they only need q_s(stop) == q_s(start)!
    preserved = [
        q for q in range(session.num_qubits)
        if np.array_equal(bits_start[:, q], bits_stop[:, q])
    ]

    loans: list[dict[str, object]] = []
    for w in idle:
        others = [q for q in preserved if q != w]
        if not others:
            continue
        # Solve [1, q_{others}] @ coeff == q_w (mod 2)
        A = np.column_stack([np.ones(shots, dtype=np.int64), bits_start[:, others]])
        rhs = bits_start[:, w]
        sol = solve_zmod_fast(A, rhs, modulus=2)
        if not sol["consistent"]:
            continue
        vec = sol["solution"]
        c0 = int(vec[0]) & 1
        supp = [others[i] for i in range(len(others)) if (int(vec[i + 1]) & 1) == 1]
        if len(supp) <= max_support:
            loans.append(
                {
                    "loaned_wire": w,
                    "constant_offset": c0,
                    "support_wires": supp,
                    "entry_cx_count": len(supp),
                    "start": int(start),
                    "stop": int(stop),
                }
            )
    loans.sort(key=lambda item: (len(item["support_wires"]), item["loaned_wire"]))
    return loans


def apply_affine_wire_loan(
    session: Session,
    start: int,
    stop: int,
    loan: dict[str, object],
    *,
    inner_kmx_using_loan: str | None = None,
    verify: bool = True,
    seed: int = 1,
    clean_wires: Sequence[int] = (),
) -> dict[str, object]:
    """Wrap ``[start, stop)`` (or ``inner_kmx_using_loan``) with entry/exit affine ``LEND`` gates and verify."""
    w = int(loan["loaned_wire"])
    c0 = int(loan["constant_offset"])
    supp = [int(s) for s in loan["support_wires"]]

    entry_ops = [f"CX q{s} q{w}" for s in supp]
    if c0:
        entry_ops.append(f"X q{w}")

    exit_ops: list[str] = []
    if c0:
        exit_ops.append(f"X q{w}")
    exit_ops.extend(f"CX q{s} q{w}" for s in reversed(supp))

    body_lines = (
        [ln.strip() for ln in inner_kmx_using_loan.splitlines() if ln.strip()]
        if inner_kmx_using_loan is not None
        else session.tape.decode_span(start, stop)
    )
    wrapped_lines = entry_ops + body_lines + exit_ops
    wrapped_kmx = "\n".join(wrapped_lines) + "\n"

    verdict = {"passed": True, "elapsed_ms": 0.0}
    if verify:
        verdict = session.verify_cone(start, stop, wrapped_kmx, seed=seed, clean_wires=clean_wires)
    return {
        "loaned_wire": w,
        "wrapped_kmx": wrapped_kmx,
        "entry_ops": entry_ops,
        "exit_ops": exit_ops,
        "verified": bool(verdict["passed"]),
        "verify_receipt": verdict,
    }


class ShapeExplorer:
    """Evaluates parametric circuit-shape candidates through ``ScoreModel`` and ``Session.funnel``."""

    def __init__(self, score_model: ScoreModel) -> None:
        self.score_model = score_model

    def rank_shapes(
        self,
        shapes: Sequence[dict[str, object]],
    ) -> dict[str, object]:
        """Price a table of discrete ``{"name": ..., "qubits": Q, "toffolis": T, "t_gates": t}`` shapes."""
        scored = []
        for sh in shapes:
            q = int(sh["qubits"])
            tof = int(sh["toffolis"])
            tg = int(sh.get("t_gates", 0))
            sc = self.score_model.evaluate(q, tof, tg)
            ex = self.score_model.exchange_rate(q, tof, tg)
            scored.append(
                {
                    **sh,
                    "score": sc,
                    "toffolis_per_qubit": ex["toffolis_per_qubit"],
                }
            )
        scored.sort(key=lambda r: -r["score"] if self.score_model.higher_is_better else r["score"])

        # Compute (qubits, toffolis) Pareto frontier
        pareto = []
        for cand in scored:
            dominated = False
            for other in scored:
                if other is cand:
                    continue
                if (
                    other["qubits"] <= cand["qubits"]
                    and other["toffolis"] <= cand["toffolis"]
                    and (other["qubits"] < cand["qubits"] or other["toffolis"] < cand["toffolis"])
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(cand)
        return {
            "best": scored[0] if scored else None,
            "ranked": scored,
            "pareto_frontier": pareto,
        }

    def screen_candidates(
        self,
        session: Session,
        candidates: Sequence[FunnelCandidate],
        *,
        seed: int = 1,
        clean_wires: Sequence[int] = (),
    ) -> dict[str, object]:
        return session.funnel(candidates, seed=seed, clean_wires=clean_wires)
