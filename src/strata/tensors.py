"""Algebraic & tensor synthesis: Lovasz matroid parity, relation hyperplanes, multi-pass QROAM, and flip graphs."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .native import gf2_rank_fast, lovasz_rank_fast


def _symplectic_pairs_u64(skew_mat: np.ndarray) -> list[tuple[int, int]]:
    """Decompose a binary skew-symmetric matrix ``P`` (zero diagonal) into symplectic pairs ``(u_mask, v_mask)``."""
    P = (np.asarray(skew_mat, dtype=np.uint8) & 1).copy()
    n = P.shape[0]
    for i in range(n):
        P[i, i] = 0
    # Track basis forms starting from coordinate functionals e_i = (1 << i)
    forms = [1 << i for i in range(n)]
    pairs: list[tuple[int, int]] = []
    active = [True] * n
    while True:
        found = None
        for i in range(n):
            if not active[i]:
                continue
            for j in range(i + 1, n):
                if active[j] and P[i, j]:
                    found = (i, j)
                    break
            if found is not None:
                break
        if found is None:
            break
        a, b = found
        pairs.append((forms[a], forms[b]))
        # Eliminate a and b from remaining active rows/cols via symplectic transvection
        # New coordinate k': x'_k = x_k; functional contributions absorbed into (forms[a], forms[b])
        row_a = P[a].copy()
        row_b = P[b].copy()
        for k in range(n):
            if k in (a, b) or not active[k]:
                continue
            if row_a[k]:
                # Add P[b] to P[k] and update dual forms[b] ^= forms[k]
                forms[b] ^= forms[k]
            if row_b[k]:
                forms[a] ^= forms[k]
        for k in range(n):
            if k in (a, b) or not active[k]:
                continue
            for l in range(n):
                if l in (a, b) or not active[l]:
                    continue
                P[k, l] ^= (row_a[k] & row_b[l]) ^ (row_b[k] & row_a[l])
        active[a] = False
        active[b] = False
    return pairs


def lovasz_matroid_parity_rank(
    slices: np.ndarray,
    *,
    trials: int = 3,
    seed: int = 1337,
) -> dict[str, object]:
    """Compute the exact polarized quadratic AND lower bound over ``GF(2)`` via randomized ``GF(2^64)`` rank.

    Given ``B`` quadratic form matrices ``slices[b]`` of shape ``(n, n)`` (with ``n <= 64``) representing
    degree-2 Boolean polynomials ``f_b(x) = sum_{i < j} S_{b,i,j} x_i x_j``, decomposes each polarized
    alternating form ``P_b = S_b ^ S_b^T`` into symplectic factor pairs and evaluates Lovasz's linear
    matroid-parity rank over ``GF(2^64)``.
    """
    arr = np.asarray(slices, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    n = int(arr.shape[1])
    if n < 1 or n > 64:
        raise ValueError("lovasz_matroid_parity_rank supports n in 1..64")
    pol = (arr ^ np.swapaxes(arr, 1, 2)) & 1
    lhs_list: list[int] = []
    rhs_list: list[int] = []
    for b in range(pol.shape[0]):
        for u_m, v_m in _symplectic_pairs_u64(pol[b]):
            lhs_list.append(u_m)
            rhs_list.append(v_m)
    if not lhs_list:
        return {"min_pair_ands": 0, "max_paired_lines": 0, "matrix_rank": 0, "num_pairs": 0, "dim": n, "elapsed_ms": 0.0}
    lhs_arr = np.asarray(lhs_list, dtype=np.uint64)
    rhs_arr = np.asarray(rhs_list, dtype=np.uint64)
    return lovasz_rank_fast(lhs_arr, rhs_arr, n, trials=trials, seed=seed)


def _anf_mobius_2d(bit_table: np.ndarray) -> np.ndarray:
    """Fast bitwise Mobius transform converting truth tables ``(2^n, B)`` into ANF monomial tables ``(2^n, B)``."""
    N, B = bit_table.shape
    n = (N - 1).bit_length()
    if (1 << n) != N:
        raise ValueError("Truth-table row count must be a power of two")
    anf = (bit_table.astype(np.uint8) & 1).copy()
    for bit in range(n):
        step = 1 << bit
        for base in range(0, N, step << 1):
            anf[base + step : base + (step << 1)] ^= anf[base : base + step]
    return anf


def search_relation_hyperplanes(
    table_words: Sequence[int] | np.ndarray,
    bit_width: int,
    *,
    max_hamming_mask: int = 3,
) -> dict[str, object]:
    """Search non-zero output-bit masks ``lambda`` where ``XOR_{b in lambda} T_b(x)`` drops in ANF degree or rank.

    Whenever ``XOR_{b in lambda} T_b(x)`` has degree ``<= 1`` (purely affine!) or degree ``2`` with
    low polarized rank, one output bit in ``lambda`` does **not** need QROM Multiplexer synthesis at
    all: it can be reconstructed at the end from its sibling output bits via free ``CX`` gates (plus at
    most ``rank // 2`` Toffolis if rank-2)!
    """
    words = np.asarray(table_words, dtype=np.uint64).ravel()
    N = words.size
    B = int(bit_width)
    n = (N - 1).bit_length()
    if N == 0 or (1 << n) != N:
        raise ValueError("table_words length must be a positive power of two (2^n)")

    shifts = np.arange(B, dtype=np.uint64)[None, :]
    bits = ((words[:, None] >> shifts) & np.uint64(1)).astype(np.uint8)  # (N, B)
    anf = _anf_mobius_2d(bits)  # (N, B)
    monomial_deg = np.array([ idx.bit_count() for idx in range(N) ], dtype=np.int32)

    # Pre-extract quadratic slices (deg == 2) for fast Lovasz rank checks
    quad_rows = np.flatnonzero(monomial_deg == 2)
    higher_rows = np.flatnonzero(monomial_deg >= 3)

    # Check exact GF(2) nullspace of higher-degree (>=2) ANF rows first -> finds ALL purely affine relations!
    non_affine_anf = anf[monomial_deg >= 2]  # (M_high, B)
    affine_rank = gf2_rank_fast(non_affine_anf)

    relations: list[dict[str, object]] = []
    # Enumerate sparse masks up to weight `max_hamming_mask` (plus all masks if B <= 10)
    masks_to_test: list[int] = []
    if B <= 10:
        masks_to_test = list(range(1, 1 << B))
    else:
        for b0 in range(B):
            masks_to_test.append(1 << b0)
            if max_hamming_mask >= 2:
                for b1 in range(b0 + 1, B):
                    masks_to_test.append((1 << b0) | (1 << b1))
                    if max_hamming_mask >= 3:
                        for b2 in range(b1 + 1, B):
                            masks_to_test.append((1 << b0) | (1 << b1) | (1 << b2))

    for mask in masks_to_test:
        cols = [b for b in range(B) if (mask >> b) & 1]
        comb = np.bitwise_xor.reduce(anf[:, cols], axis=1) if len(cols) > 1 else anf[:, cols[0]]
        if not np.any(comb[higher_rows]):
            if not np.any(comb[quad_rows]):
                deg = 1 if np.any(comb[monomial_deg == 1]) else 0
                relations.append(
                    {
                        "mask": mask,
                        "columns": cols,
                        "degree": deg,
                        "pair_ands": 0,
                        "kind": "affine_free",
                    }
                )
            else:
                # Degree 2: build n x n symmetric form and compute exact polarized pair_ands
                S = np.zeros((n, n), dtype=np.uint8)
                for r_idx in quad_rows:
                    if comb[r_idx]:
                        # Extract the two set bits of r_idx
                        low = (int(r_idx) & -int(r_idx)).bit_length() - 1
                        high = (int(r_idx) ^ (1 << low)).bit_length() - 1
                        S[low, high] = 1
                        S[high, low] = 1
                rk = lovasz_matroid_parity_rank(S)
                relations.append(
                    {
                        "mask": mask,
                        "columns": cols,
                        "degree": 2,
                        "pair_ands": int(rk["min_pair_ands"]),
                        "kind": "quadratic_residual",
                    }
                )

    relations.sort(key=lambda r: (r["pair_ands"], r["degree"], len(r["columns"]), r["mask"]))
    return {
        "num_addresses": N,
        "bit_width": B,
        "nonlinear_column_rank": affine_rank,
        "free_affine_relations_exist": affine_rank < B,
        "relations": relations[:64],
    }


def plan_multipass_qroam(
    num_entries: int,
    bit_width: int,
    *,
    max_qubits: int | None = None,
    address_qubits: int | None = None,
) -> dict[str, object]:
    """Sweep SelectSwap / Subspace Multi-Pass ``QROAM`` schedules ``(log_block lb, passes P)`` with MBU cleanup.

    For a table of ``N`` entries of ``B`` bits:
      - Splitting ``B`` across ``P`` sequential passes processes ``b_p = ceil(B / P)`` bits per pass.
      - With block factor ``K = 2^lb``, loading requires ``ceil(N / K) - 1`` unary-iteration Toffolis +
        ``(K - 1) * b_p`` swap-network Toffolis, while **MBU cleanup** demolishes the ``K * b_p`` scratch
        data wires with ``HMR`` (`0` swap Toffolis on uncompute!) and repays phase kickback using at most
        ``ceil(N / 2^lb_uncomp)`` Toffolis!
    """
    N = max(2, int(num_entries))
    B = max(1, int(bit_width))
    n_addr = (N - 1).bit_length() if address_qubits is None else int(address_qubits)

    schedules: list[dict[str, object]] = []
    max_lb = min(10, n_addr)
    for passes in (1, 2, 3, 4):
        if passes > B:
            continue
        bp = math.ceil(B / passes)
        for lb in range(0, max_lb + 1):
            K = 1 << lb
            # Qubits: address + output B + unary ancillas (n_addr - lb) + (K - 1) * bp copy scratch
            q_unary = max(0, n_addr - lb - 1)
            q_scratch = (K - 1) * bp
            total_q = n_addr + B + q_unary + q_scratch
            if max_qubits is not None and total_q > max_qubits:
                continue

            # Forward load per pass: ceil(N / K) - 1 unary Toffolis + (K - 1) * bp Fredkin Toffolis
            blocks = math.ceil(N / K)
            load_tof = max(0, blocks - 1) + (K - 1) * bp
            # MBU phase-kickback cleanup per pass (optimal uncompute block factor K_u ~ sqrt(N))
            lb_u = max(0, min(n_addr, int(round(0.5 * math.log2(max(1, N))))))
            Ku = 1 << lb_u
            cleanup_tof = (max(0, math.ceil(N / Ku) - 1) + Ku) if lb > 0 else 0
            total_tof = passes * (load_tof + cleanup_tof)
            schedules.append(
                {
                    "passes": passes,
                    "log_block": lb,
                    "block_size": K,
                    "bits_per_pass": bp,
                    "qubits": total_q,
                    "toffolis": total_tof,
                    "q_times_toffoli": total_q * total_tof,
                }
            )

    schedules.sort(key=lambda s: (s["toffolis"], s["qubits"]))
    best = schedules[0] if schedules else None
    return {
        "num_entries": N,
        "bit_width": B,
        "best": best,
        "schedules": schedules,
    }


def flip_graph_reduce(
    u_factors: np.ndarray,
    v_factors: np.ndarray,
    w_factors: np.ndarray,
    *,
    steps: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    """Kauers-Moosbauer GF(2) bilinear flip-graph search + 2x2 biclique CSE fanout optimizer.

    Given a rank-``R`` GF(2) bilinear tensor decomposition::

        T(a, b) = XOR_{r=0}^{R-1} <u_r, a> * <v_r, b> * w_r

    walks the Kauers-Moosbauer flip graph:
      - **Reduction**: whenever two terms share two factors (or one factor becomes zero), merges them
        in-place to strictly drop bilinear rank ``R -> R - 1`` (saving 1 Toffoli per rank drop!).
      - **Flip**: whenever two terms share one factor (e.g. ``u_i == u_j``), applies the identity
        ``u*(v_i)*w_i ^ u*(v_j)*w_j == u*(v_i ^ v_j)*w_i ^ u*v_j*(w_i ^ w_j)`` to explore neighboring
        decompositions, tracking both minimum Toffoli rank ``R`` AND minimum post-biclique ``CX`` fanout!
    """
    U = (np.asarray(u_factors, dtype=np.uint8) & 1).copy()
    V = (np.asarray(v_factors, dtype=np.uint8) & 1).copy()
    W = (np.asarray(w_factors, dtype=np.uint8) & 1).copy()
    if not (U.ndim == V.ndim == W.ndim == 2 and U.shape[0] == V.shape[0] == W.shape[0]):
        raise ValueError("u_factors, v_factors, w_factors must be 2D arrays with equal row count R")

    def _canonical_reduce(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Drop zero terms and merge any pairs sharing two identical legs
        keep = [
            i for i in range(u.shape[0])
            if np.any(u[i]) and np.any(v[i]) and np.any(w[i])
        ]
        u, v, w = u[keep], v[keep], w[keep]
        merged = True
        while merged and u.shape[0] >= 2:
            merged = False
            R_cur = u.shape[0]
            for i in range(R_cur):
                for j in range(i + 1, R_cur):
                    eq_u = np.array_equal(u[i], u[j])
                    eq_v = np.array_equal(v[i], v[j])
                    eq_w = np.array_equal(w[i], w[j])
                    if eq_u and eq_v:
                        w[i] ^= w[j]
                        w[j] = 0
                        merged = True
                        break
                    if eq_u and eq_w:
                        v[i] ^= v[j]
                        v[j] = 0
                        merged = True
                        break
                    if eq_v and eq_w:
                        u[i] ^= u[j]
                        u[j] = 0
                        merged = True
                        break
                if merged:
                    keep = [
                        k for k in range(u.shape[0])
                        if np.any(u[k]) and np.any(v[k]) and np.any(w[k])
                    ]
                    u, v, w = u[keep], v[keep], w[keep]
                    break
        return u, v, w

    def _cx_fanout_cost(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> int:
        # Linear XOR cost of forming U@a, V@b, and accumulating W^T@p after 2-column CSE
        def _mat_cx(M: np.ndarray) -> int:
            if M.size == 0:
                return 0
            work = M.copy()
            saved = 0
            # Count immediate 2x2 biclique savings
            if work.shape[0] >= 2 and work.shape[1] >= 2:
                overlap = (work.astype(np.int32).T @ work.astype(np.int32))
                np.fill_diagonal(overlap, 0)
                best = int(np.max(overlap))
                if best >= 2:
                    saved = best - 1
            raw = int(np.sum(np.maximum(0, np.sum(work, axis=1) - 1)))
            return max(0, raw - saved)

        return _mat_cx(u) + _mat_cx(v) + _mat_cx(w.T)

    init_rank = U.shape[0]
    init_cx = _cx_fanout_cost(U, V, W)
    U, V, W = _canonical_reduce(U, V, W)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_key = (best_U.shape[0], _cx_fanout_cost(best_U, best_V, best_W))

    rng = np.random.default_rng(seed)
    cur_U, cur_V, cur_W = best_U.copy(), best_V.copy(), best_W.copy()

    for _ in range(max(0, int(steps))):
        R_cur = cur_U.shape[0]
        if R_cur < 2:
            break
        # Collect valid flip pairs sharing u, v, or w
        flips: list[tuple[int, int, int]] = []
        for i in range(R_cur):
            for j in range(i + 1, R_cur):
                if np.array_equal(cur_U[i], cur_U[j]):
                    flips.append((0, i, j))
                if np.array_equal(cur_V[i], cur_V[j]):
                    flips.append((1, i, j))
                if np.array_equal(cur_W[i], cur_W[j]):
                    flips.append((2, i, j))
        if not flips:
            break
        leg, i, j = flips[int(rng.integers(0, len(flips)))]
        if rng.integers(0, 2) == 1:
            i, j = j, i
        if leg == 0:
            # u_i == u_j: (u, v_i, w_i) + (u, v_j, w_j) -> (u, v_i ^ v_j, w_i) + (u, v_j, w_i ^ w_j)
            cur_V[i] ^= cur_V[j]
            cur_W[j] ^= cur_W[i]
        elif leg == 1:
            cur_U[i] ^= cur_U[j]
            cur_W[j] ^= cur_W[i]
        else:
            cur_U[i] ^= cur_U[j]
            cur_V[j] ^= cur_V[i]

        cur_U, cur_V, cur_W = _canonical_reduce(cur_U, cur_V, cur_W)
        cand_key = (cur_U.shape[0], _cx_fanout_cost(cur_U, cur_V, cur_W))
        if cand_key < best_key:
            best_key = cand_key
            best_U, best_V, best_W = cur_U.copy(), cur_V.copy(), cur_W.copy()

    return {
        "u": best_U,
        "v": best_V,
        "w": best_W,
        "initial_rank": init_rank,
        "reduced_rank": best_U.shape[0],
        "toffolis_saved": init_rank - best_U.shape[0],
        "initial_cx_estimate": init_cx,
        "reduced_cx_estimate": best_key[1],
    }
