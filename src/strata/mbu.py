"""Measurement-Based Uncomputation (MBU), multi-cut phase fitting, and SWAP-aware HMR hoisting."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from .native import KmbTape, solve_zmod_fast
from .session import Session


def fit_multi_cut_phase(
    session_or_matrices: Session | KmbTape | Sequence[np.ndarray],
    cuts: Sequence[int] | None = None,
    target_phases_by_cut: Sequence[Sequence[int] | np.ndarray] | np.ndarray | None = None,
    *,
    modulus: int = 8,
    support_wires: Sequence[int] | None = None,
    include_quadratic: bool = True,
    seed: int = 1,
    clean_wires: Sequence[int] = (),
) -> dict[str, object]:
    """Fit a diagonal phase correction polynomial across one or multiple circuit cuts modulo 2, 4, or 8.

    Unlike single-cut solvers that abort when an untouched wire carries ``Z_POW``, this solver:
      1. Traces 256-lane basis states across all requested ``cuts`` via ``Session.trace_cuts``.
      2. Explicitly isolates constant / untouched wires so they never corrupt active support fitting.
      3. Assembles linear ``Z_i`` and quadratic ``Z_i * Z_j`` (i.e. ``CZ`` / ``CP``) features across
         every cut and solves the exact linear congruence system over ``Z_modulus`` in sub-millisecond
         C++ via ``solve_zmod_fast``.
    """
    if modulus not in (2, 4, 8):
        raise ValueError("modulus must be 2, 4, or 8")

    if isinstance(session_or_matrices, KmbTape):
        sess = Session(session_or_matrices)
        own_sess = True
    elif isinstance(session_or_matrices, Session):
        sess = session_or_matrices
        own_sess = False
    else:
        sess = None
        own_sess = False

    try:
        if sess is not None:
            if not cuts:
                raise ValueError("cuts must be provided when fitting from a Session or KmbTape")
            traced = sess.trace_cuts(cuts, seed=seed, clean_wires=clean_wires)
            shots = int(traced["shots"])
            cut_mats = []
            for entry in traced["cuts"]:
                # entry['packed_qubits'] has shape (nq, row_bytes) big-endian -> unpack to (shots, nq)
                bits = np.unpackbits(entry["packed_qubits"], axis=1, bitorder="big")[:, :shots].T.astype(np.int64)
                cut_mats.append(bits)
            if target_phases_by_cut is None:
                # Default: cancel accumulated Z_8 phase residues scaled to `modulus`
                z8 = traced["cuts"][-1]["z8_phases"].astype(np.int64)
                scale_div = 8 // modulus
                rhs = ((-(z8 // scale_div)) % modulus).astype(np.int64)
            elif isinstance(target_phases_by_cut, np.ndarray) and target_phases_by_cut.ndim == 1:
                rhs = np.asarray(target_phases_by_cut, dtype=np.int64) % modulus
            else:
                rhs = np.sum([np.asarray(v, dtype=np.int64) for v in target_phases_by_cut], axis=0) % modulus
        else:
            cut_mats = [np.asarray(m, dtype=np.int64) & 1 for m in session_or_matrices]
            if not cut_mats:
                raise ValueError("At least one cut basis matrix is required")
            if target_phases_by_cut is None:
                raise ValueError("target_phases_by_cut is required when passing raw basis matrices")
            rhs = np.asarray(target_phases_by_cut, dtype=np.int64).ravel() % modulus

        n_samples, nq_total = cut_mats[0].shape
        wires = list(range(nq_total)) if support_wires is None else [int(w) for w in support_wires]

        # Identify untouched/constant wires across all cuts so they are isolated cleanly
        untouched_wires: list[int] = []
        active_wires_by_cut: list[list[int]] = []
        for c_idx, mat in enumerate(cut_mats):
            active_c = []
            for w in wires:
                col = mat[:, w]
                if np.all(col == col[0]):
                    if c_idx == 0:
                        untouched_wires.append(w)
                else:
                    active_c.append(w)
            active_wires_by_cut.append(active_c)

        # Build feature columns: [global_const, (cut, wire) linear, (cut, w1, w2) quadratic]
        cols = [np.ones(n_samples, dtype=np.int64)]
        meta: list[dict[str, object]] = [{"kind": "constant", "cut_index": None, "wires": ()}]
        seen_signatures: set[bytes] = {cols[0].tobytes()}

        for c_idx, mat in enumerate(cut_mats):
            act = active_wires_by_cut[c_idx]
            for w in act:
                col = mat[:, w]
                sig = col.tobytes()
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    cols.append(col)
                    meta.append({"kind": "Z_POW", "cut_index": c_idx, "wires": (w,)})
            if include_quadratic:
                # Over Z_modulus, a CZ gate adds modulus//2 on x_i * x_j; CP adds general weight
                for idx_a in range(len(act)):
                    wa = act[idx_a]
                    ca = mat[:, wa]
                    for idx_b in range(idx_a + 1, len(act)):
                        wb = act[idx_b]
                        prod = ca * mat[:, wb]
                        if np.any(prod):
                            sig = prod.tobytes()
                            if sig not in seen_signatures:
                                seen_signatures.add(sig)
                                cols.append(prod)
                                meta.append({"kind": "CP", "cut_index": c_idx, "wires": (wa, wb)})

        A = np.column_stack(cols)
        sol = solve_zmod_fast(A, rhs, modulus=modulus)
        terms = []
        if sol["consistent"]:
            vec = sol["solution"]
            for coeff, item in zip(vec.tolist(), meta):
                c_mod = int(coeff) % modulus
                if c_mod != 0:
                    # Convert Z_modulus unit to Kickmix half-turn Z_POW exponent (1 turn = 2 half-turns)
                    half_turns = (2.0 * c_mod) / float(modulus)
                    terms.append({**item, "coeff": c_mod, "half_turns": half_turns})

        return {
            "consistent": bool(sol["consistent"]),
            "modulus": modulus,
            "rank": int(sol["rank"]),
            "num_features": len(meta),
            "untouched_wires": sorted(set(untouched_wires)),
            "terms": terms,
            "residual_zero": bool(
                sol["consistent"] and np.all(((A @ sol["solution"]) % modulus) == (rhs % modulus))
            ),
        }
    finally:
        if own_sess and sess is not None:
            sess.close()


_CCX_RE = re.compile(r"^CCX\s+q(\d+)\s+q(\d+)\s+q(\d+)\s*$")
_SWAP_RE = re.compile(r"^SWAP\s+q(\d+)\s+q(\d+)\s*$")
_HMR_RE = re.compile(r"^HMR\s+q(\d+)\s+b(\d+)\s*$")
_Q_TOKEN_RE = re.compile(r"\bq(\d+)\b")


def demolish_and_piggyback(
    session: Session,
    start: int,
    stop: int,
    *,
    scratch_wires: Sequence[int] | None = None,
    next_bit_id: int | None = None,
    verify: bool = True,
    seed: int = 1,
    clean_wires: Sequence[int] = (),
) -> dict[str, object]:
    """Replace reversible ``CCX`` uncompute gates in ``[start, stop)`` with ``0``-Toffoli ``HMR + CZ_IF``.

    Each reversible Bennett uncompute ``CCX q_c0 q_c1 q_w`` that clears ``q_w -> |0>`` (verified across
    all 256 basis lanes at the moment of erasure) is converted into::

        HMR q_w b_m
        CZ q_c0 q_c1 if b_m

    Furthermore, if ``q_c0`` and ``q_c1`` are themselves uncomputed later in ``[start, stop)``, walking
    backward from the end of the uncompute cascade converts the **entire Toffoli uncompute tree** into
    zero-Toffoli ``HMR`` measurements + classical-controlled ``CZ_IF`` phase repairs!
    """
    lines = session.tape.decode_span(start, stop)
    if not lines:
        return {
            "toffolis_removed": 0,
            "replacement_kmx": "",
            "verified": True,
            "demolished_wires": [],
        }

    # Trace basis states step-by-step after each op in [start, stop) from the checkpointed entry state
    traced_end = session.trace_cuts([start, stop], seed=seed, clean_wires=clean_wires)
    end_q = traced_end["cuts"][1]["packed_qubits"]  # (nq, row_bytes)
    clean_at_stop = {
        q for q in range(session.num_qubits) if not np.any(end_q[q])
    }
    eligible_scratch = clean_at_stop if scratch_wires is None else (clean_at_stop & {int(w) for w in scratch_wires})

    bit_cursor = int(session.num_bits if next_bit_id is None else next_bit_id)
    out_lines = list(lines)
    demolished: list[int] = []
    toffolis_removed = 0

    # Walk backward from `stop - 1` down to `start`: if wire `w` is provably clean |0> after line `i`
    # and never touched again in `[i+1, stop)`, and line `i` is `CCX c0 c1 w`, then right BEFORE line `i`,
    # wire `w` holds `|c0 & c1>`! Replacing `CCX c0 c1 w` with `HMR w b_m; CZ_IF c0 c1 b_m` is 100% exact!
    touched_after: set[int] = set()
    for idx in range(len(lines) - 1, -1, -1):
        ln = lines[idx].strip()
        m = _CCX_RE.match(ln)
        qs_in_line = [int(tok) for tok in _Q_TOKEN_RE.findall(ln)]
        if m is not None:
            c0, c1, target = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if target in eligible_scratch and target not in touched_after:
                # Verify via trace_cuts at `start + idx + 1` that `target` is genuinely |0> right after this CCX
                cut_chk = session.trace_cuts([start + idx + 1], seed=seed, clean_wires=clean_wires)
                if not np.any(cut_chk["cuts"][0]["packed_qubits"][target]):
                    b_id = bit_cursor
                    bit_cursor += 1
                    out_lines[idx] = f"HMR q{target} b{b_id}\nCZ q{c0} q{c1} if b{b_id}"
                    demolished.append(target)
                    toffolis_removed += 1
                    # Note: c0 and c1 ARE touched by CZ_IF at idx, so mark c0, c1, target as touched_after for earlier ops
                    touched_after.update((c0, c1, target))
                    continue
        touched_after.update(qs_in_line)

    replacement_kmx = "\n".join(out_lines) + ("\n" if out_lines else "")
    verdict = {"passed": True, "elapsed_ms": 0.0}
    if verify and toffolis_removed > 0:
        verdict = session.verify_cone(
            start,
            stop,
            replacement_kmx,
            seed=seed,
            clean_wires=clean_wires,
        )
    return {
        "toffolis_removed": toffolis_removed,
        "demolished_wires": list(reversed(demolished)),
        "allocated_bits": list(range(session.num_bits if next_bit_id is None else next_bit_id, bit_cursor)),
        "replacement_kmx": replacement_kmx,
        "verified": bool(verdict["passed"]),
        "verify_receipt": verdict,
    }


def hoist_hmr_across_swaps(kmx_or_lines: str | Sequence[str], *, drop_dead_swaps: bool = True) -> dict[str, object]:
    """Hoist ``HMR`` measurements backward across preceding ``SWAP`` chains to shrink live qubit lifetimes.

    Whenever ``SWAP q_a q_b`` is immediately (or across disjoint gates) followed by ``HMR q_b b_m``,
    commuting ``HMR`` ahead of ``SWAP`` turns it into ``HMR q_a b_m; SWAP q_a q_b`` so ``q_a`` is
    freed to ``|0>`` earlier. If ``drop_dead_swaps`` is enabled and neither ``q_a`` nor ``q_b`` is
    used after the ``SWAP`` (or ``SWAP`` only moved a wire solely to demolish it), redundant trailing
    ``SWAP``s on dead wires are reported and commuted cleanly.
    """
    if isinstance(kmx_or_lines, str):
        raw_lines = [ln.strip() for ln in kmx_or_lines.splitlines() if ln.strip()]
    else:
        raw_lines = [str(ln).strip() for ln in kmx_or_lines if str(ln).strip()]

    work = list(raw_lines)
    hoisted_steps = 0
    changed = True
    while changed:
        changed = False
        for i in range(1, len(work)):
            m_hmr = _HMR_RE.match(work[i])
            if m_hmr is None:
                continue
            hq = int(m_hmr.group(1))
            hb = int(m_hmr.group(2))
            prev = work[i - 1]
            m_sw = _SWAP_RE.match(prev)
            if m_sw is not None:
                sa, sb = int(m_sw.group(1)), int(m_sw.group(2))
                if hq == sa or hq == sb:
                    src_q = sb if hq == sa else sa
                    work[i - 1] = f"HMR q{src_q} b{hb}"
                    work[i] = f"SWAP q{sa} q{sb}"
                    hoisted_steps += 1
                    changed = True
                    continue
            # Also hoist HMR across completely disjoint preceding instructions
            prev_qs = {int(t) for t in _Q_TOKEN_RE.findall(prev)}
            prev_bs = {int(t) for t in re.findall(r"\bb(\d+)\b", prev)}
            if (
                not prev.startswith(("HMR", "PUSH_CONDITION", "POP_CONDITION", "REGISTER", "APPEND_TO_REGISTER"))
                and hq not in prev_qs
                and hb not in prev_bs
            ):
                work[i - 1], work[i] = work[i], work[i - 1]
                hoisted_steps += 1
                changed = True

    swaps_eliminated = 0
    if drop_dead_swaps:
        # Track wires never read after index i: if a SWAP swaps two wires that are BOTH currently clean |0>
        # (immediately after HMR on both) or trailing dead SWAP at end of clean scratch block
        clean_now: set[int] = set()
        filtered: list[str] = []
        for ln in work:
            m_hmr = _HMR_RE.match(ln)
            if m_hmr is not None:
                clean_now.add(int(m_hmr.group(1)))
                filtered.append(ln)
                continue
            m_sw = _SWAP_RE.match(ln)
            if m_sw is not None:
                sa, sb = int(m_sw.group(1)), int(m_sw.group(2))
                if sa in clean_now and sb in clean_now:
                    swaps_eliminated += 1
                    continue
                # Permute clean status across SWAP
                ca, cb = (sa in clean_now), (sb in clean_now)
                clean_now.discard(sa)
                clean_now.discard(sb)
                if ca:
                    clean_now.add(sb)
                if cb:
                    clean_now.add(sa)
                filtered.append(ln)
                continue
            for q in (int(t) for t in _Q_TOKEN_RE.findall(ln)):
                clean_now.discard(q)
            filtered.append(ln)
        work = filtered

    return {
        "kmx": "\n".join(work) + ("\n" if work else ""),
        "lines": work,
        "hoisted_steps": hoisted_steps,
        "swaps_eliminated": swaps_eliminated,
    }
