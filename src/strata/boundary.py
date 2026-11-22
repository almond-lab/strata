"""Boundary representation shifts: 0-Toffoli classical BIT_* compilers, OOM guards, and projective GF(2^k)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import StrataError


class ClassicalProgram:
    """Compiles classical preprocessing maps, popcounts, and sorting networks into 0-Toffoli Kickmix ``BIT_*`` ops.

    Enforces explicit ``max_ops`` and ``max_kmb_bytes`` safety limits so classical unrolling never
    exceeds evaluation-container memory limits (such as 30 GB multi-worker verification sandboxes).
    """

    def __init__(
        self,
        *,
        initial_bits: int = 0,
        max_ops: int = 80_000_000,
        max_kmb_bytes: int = 650_000_000,
    ) -> None:
        self._next_bit = int(initial_bits)
        self.max_ops = int(max_ops)
        self.max_kmb_bytes = int(max_kmb_bytes)
        self.ops: list[tuple[str, ...]] = []

    @property
    def num_bits(self) -> int:
        return self._next_bit

    @property
    def num_ops(self) -> int:
        return len(self.ops)

    def estimated_kmb_bytes(self) -> int:
        # In Kickmix .kmb v2, each BIT_* op uses 1 byte in types[] + 4 bytes in bt[] + optional 4 bytes in bc[] (~9 bytes/op)
        return 104 + 9 * len(self.ops)

    def _guard(self, extra_ops: int = 1) -> None:
        new_ops = len(self.ops) + extra_ops
        est_bytes = 104 + 9 * new_ops
        if new_ops > self.max_ops:
            raise StrataError(
                f"ClassicalProgram exceeded max_ops budget ({new_ops} > {self.max_ops}); "
                "refusing to emit oversized classical stream that risks container OOM"
            )
        if est_bytes > self.max_kmb_bytes:
            raise StrataError(
                f"ClassicalProgram exceeded max_kmb_bytes budget ({est_bytes} > {self.max_kmb_bytes}); "
                "reduce classical candidate sweeps or lower max_kmb_bytes"
            )

    def alloc_bits(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("count must be nonnegative")
        start = self._next_bit
        self._next_bit += int(count)
        return list(range(start, self._next_bit))

    def bit_set(self, target: int, value: int) -> None:
        self._guard(1)
        op = "BIT_STORE1" if (int(value) & 1) else "BIT_STORE0"
        self.ops.append((op, f"b{int(target)}"))

    def bit_not(self, target: int) -> None:
        self._guard(1)
        self.ops.append(("BIT_INVERT", f"b{int(target)}"))

    def bit_xor(self, src: int, dst: int) -> None:
        if int(src) == int(dst):
            self.bit_set(dst, 0)
            return
        self._guard(1)
        self.ops.append(("BIT_INVERT", f"b{int(dst)}", "if", f"b{int(src)}"))

    def bit_and(self, src: int, dst: int) -> None:
        if int(src) == int(dst):
            return
        # dst &= src <=> if not src: dst = 0 <=> invert src, store0 dst if src, invert src
        # Wait: src might be read-only or shared; use a stack condition without mutating src:
        # tmp = 0; PUSH_CONDITION if b_dst; BIT_STORE1 tmp if b_src; POP_CONDITION; copy tmp -> dst
        # Even simpler in 3 instructions using a scratch or invert-restore on src:
        self._guard(3)
        b_s = f"b{int(src)}"
        b_d = f"b{int(dst)}"
        self.ops.append(("BIT_INVERT", b_s))
        self.ops.append(("BIT_STORE0", b_d, "if", b_s))
        self.ops.append(("BIT_INVERT", b_s))

    def bit_copy(self, src: int, dst: int) -> None:
        if int(src) == int(dst):
            return
        self._guard(2)
        self.ops.append(("BIT_STORE0", f"b{int(dst)}"))
        self.ops.append(("BIT_INVERT", f"b{int(dst)}", "if", f"b{int(src)}"))

    def bit_and3(self, a: int, b: int, dst: int) -> None:
        """Compute ``dst = a & b`` in 0 quantum Toffolis using native Kickmix classical ops."""
        if dst == a:
            self.bit_and(b, dst)
        elif dst == b:
            self.bit_and(a, dst)
        else:
            self._guard(4)
            self.ops.append(("BIT_STORE0", f"b{int(dst)}"))
            self.ops.append(("PUSH_CONDITION", "if", f"b{int(a)}"))
            self.ops.append(("BIT_STORE1", f"b{int(dst)}", "if", f"b{int(b)}"))
            self.ops.append(("POP_CONDITION",))

    def xor_linear_map(
        self,
        matrix_gf2: np.ndarray,
        in_bits: Sequence[int],
        out_bits: Sequence[int] | None = None,
        *,
        use_pair_cse: bool = True,
    ) -> list[int]:
        """Compile ``y = M @ x (mod 2)`` into ``BIT_XOR`` streams with optional 4-Russians pair CSE."""
        M = (np.asarray(matrix_gf2, dtype=np.uint8) & 1).copy()
        if M.ndim != 2 or M.shape[1] != len(in_bits):
            raise ValueError("matrix_gf2 shape must be (m, len(in_bits))")
        m, n = M.shape
        outs = self.alloc_bits(m) if out_bits is None else [int(b) for b in out_bits]
        for b in outs:
            self.bit_set(b, 0)

        if use_pair_cse and m >= 4 and n >= 4:
            # Greedily factor repeated column pairs across rows
            while True:
                best_pair = None
                best_count = 2
                # Check pairs of active columns
                col_weights = np.sum(M, axis=0)
                cand_cols = np.flatnonzero(col_weights >= 3)
                for idx_a in range(cand_cols.size):
                    ca = int(cand_cols[idx_a])
                    va = M[:, ca]
                    for idx_b in range(idx_a + 1, cand_cols.size):
                        cb = int(cand_cols[idx_b])
                        cnt = int(np.sum(va & M[:, cb]))
                        if cnt > best_count:
                            best_count = cnt
                            best_pair = (ca, cb)
                if best_pair is None:
                    break
                ca, cb = best_pair
                t_bit = self.alloc_bits(1)[0]
                self.bit_copy(in_bits[ca], t_bit)
                self.bit_xor(in_bits[cb], t_bit)
                both = (M[:, ca] & M[:, cb]) == 1
                M[both, ca] = 0
                M[both, cb] = 0
                for r in np.flatnonzero(both):
                    self.bit_xor(t_bit, outs[int(r)])

        for r in range(m):
            for c in np.flatnonzero(M[r]):
                self.bit_xor(in_bits[int(c)], outs[r])
        return outs

    def popcount_bits(self, in_bits: Sequence[int]) -> list[int]:
        """Compile an exact classical CSA/adder tree computing the binary weight of ``in_bits``."""
        if not in_bits:
            z = self.alloc_bits(1)
            self.bit_set(z[0], 0)
            return z
        # Bucket bits by weight 2^k
        buckets: dict[int, list[int]] = {0: [int(b) for b in in_bits]}
        max_k = (len(in_bits)).bit_length()
        for k in range(max_k + 1):
            col = buckets.get(k, [])
            while len(col) >= 3:
                a, b, c = col.pop(), col.pop(), col.pop()
                # Full adder: sum = a ^ b ^ c, carry = (a & b) ^ (c & (a ^ b))
                ab = self.alloc_bits(1)[0]
                self.bit_copy(a, ab)
                self.bit_xor(b, ab)
                s_bit = self.alloc_bits(1)[0]
                self.bit_copy(ab, s_bit)
                self.bit_xor(c, s_bit)

                t1 = self.alloc_bits(1)[0]
                self.bit_and3(a, b, t1)
                t2 = self.alloc_bits(1)[0]
                self.bit_and3(c, ab, t2)
                self.bit_xor(t2, t1)
                col.append(s_bit)
                buckets.setdefault(k + 1, []).append(t1)
            if len(col) == 2:
                a, b = col.pop(), col.pop()
                s_bit = self.alloc_bits(1)[0]
                self.bit_copy(a, s_bit)
                self.bit_xor(b, s_bit)
                c_bit = self.alloc_bits(1)[0]
                self.bit_and3(a, b, c_bit)
                col.append(s_bit)
                buckets.setdefault(k + 1, []).append(c_bit)
            buckets[k] = col

        out = []
        for k in range(max_k):
            col = buckets.get(k, [])
            if col:
                out.append(col[0])
            else:
                z = self.alloc_bits(1)[0]
                self.bit_set(z, 0)
                out.append(z)
        return out

    def _compare_swap(
        self,
        key_a: Sequence[int],
        key_b: Sequence[int],
        payload_a: Sequence[int],
        payload_b: Sequence[int],
        *,
        descending: bool = False,
    ) -> None:
        """In-place classical compare-and-swap on little-endian unsigned bit-vectors ``key_a, key_b``."""
        w = len(key_a)
        # Compute swap_flag = (key_a > key_b) if ascending else (key_a < key_b)
        gt = self.alloc_bits(1)[0]
        eq = self.alloc_bits(1)[0]
        self.bit_set(gt, 0)
        self.bit_set(eq, 1)
        diff = self.alloc_bits(1)[0]
        term = self.alloc_bits(1)[0]
        for bit in range(w - 1, -1, -1):
            lhs = key_b[bit] if descending else key_a[bit]
            rhs = key_a[bit] if descending else key_b[bit]
            # diff = lhs ^ rhs
            self.bit_copy(lhs, diff)
            self.bit_xor(rhs, diff)
            # term = eq & lhs & ~rhs == eq & diff & lhs
            self.bit_and3(eq, diff, term)
            self.bit_and(lhs, term)
            self.bit_xor(term, gt)
            # eq = eq & ~diff
            self.bit_not(diff)
            self.bit_and(diff, eq)

        # Conditional swap via Fredkin-style classical XOR mux: t = flag & (a ^ b); a ^= t; b ^= t
        mux = self.alloc_bits(1)[0]
        for ba, bb in zip(list(key_a) + list(payload_a), list(key_b) + list(payload_b)):
            self.bit_copy(ba, mux)
            self.bit_xor(bb, mux)
            self.bit_and(gt, mux)
            self.bit_xor(mux, ba)
            self.bit_xor(mux, bb)

    def odd_even_sort(
        self,
        keys_bits: Sequence[Sequence[int]],
        payloads_bits: Sequence[Sequence[int]] | None = None,
        *,
        descending: bool = False,
    ) -> None:
        """Sort classical key bit-vectors (and optional payloads) in-place using 0-Toffoli ``BIT_*`` ops."""
        n = len(keys_bits)
        payloads = [[] for _ in range(n)] if payloads_bits is None else [list(p) for p in payloads_bits]
        if n <= 1:
            return
        # Batcher odd-even merge sort (works for arbitrary n by guarding index bounds < n)
        p = 1
        while p < n:
            k = p
            while k >= 1:
                j = k % p
                while j <= n - 1 - k:
                    for i in range(0, min(k, n - j - k)):
                        if (i + j) // (p * 2) == (i + j + k) // (p * 2):
                            self._compare_swap(
                                keys_bits[i + j],
                                keys_bits[i + j + k],
                                payloads[i + j],
                                payloads[i + j + k],
                                descending=descending,
                            )
                    j += 2 * k
                k //= 2
            p *= 2

    def simulate(self, initial_values: dict[int, int] | Sequence[int]) -> np.ndarray:
        """Simulate the classical ``BIT_*`` stream on a bit state vector for fast testing & verification."""
        state = np.zeros(max(1, self._next_bit), dtype=np.uint8)
        if isinstance(initial_values, dict):
            for k, v in initial_values.items():
                state[int(k)] = int(v) & 1
        else:
            for k, v in enumerate(initial_values):
                state[k] = int(v) & 1

        cond_stack: list[int] = []
        for op_tuple in self.ops:
            op = op_tuple[0]
            if op == "PUSH_CONDITION":
                cond_stack.append(int(op_tuple[2][1:]))
                continue
            if op == "POP_CONDITION":
                cond_stack.pop()
                continue
            active = all(state[c] == 1 for c in cond_stack)
            if len(op_tuple) == 4 and op_tuple[2] == "if":
                active = active and (state[int(op_tuple[3][1:])] == 1)
            if not active:
                continue
            dst = int(op_tuple[1][1:])
            if op == "BIT_STORE0":
                state[dst] = 0
            elif op == "BIT_STORE1":
                state[dst] = 1
            elif op == "BIT_INVERT":
                state[dst] ^= 1
        return state

    def emit_kmx(self) -> str:
        return "\n".join(" ".join(t) for t in self.ops) + ("\n" if self.ops else "")


def hoist_classical_prefix(kmx_text: str, *, classical_wires: Sequence[int]) -> dict[str, object]:
    """Hoist reversible basis gates (``X, CX, CCX, SWAP``) on classical prefix wires into ``0``-Toffoli ``BIT_*`` ops."""
    c_set = {int(w) for w in classical_wires}
    lines = [ln.strip() for ln in kmx_text.splitlines() if ln.strip()]

    # Allocate classical shadow bit b_w for each classical wire q_w
    wire_to_bit = {w: idx for idx, w in enumerate(sorted(c_set))}
    prog = ClassicalProgram(initial_bits=len(wire_to_bit))
    quantum_suffix: list[str] = []
    quantum_tainted: set[int] = set()
    toffolis_hoisted = 0

    for ln in lines:
        parts = ln.split()
        op = parts[0]
        if op in {"REGISTER", "APPEND_TO_REGISTER"}:
            quantum_suffix.append(ln)
            continue
        qs = [int(tok[1:]) for tok in parts[1:] if tok.startswith("q") and tok[1:].isdigit()]
        if qs and all((q in c_set and q not in quantum_tainted) for q in qs):
            if op == "X":
                prog.bit_not(wire_to_bit[qs[0]])
                continue
            if op == "CX":
                prog.bit_xor(wire_to_bit[qs[0]], wire_to_bit[qs[1]])
                continue
            if op == "CCX":
                tmp = prog.alloc_bits(1)[0]
                prog.bit_and3(wire_to_bit[qs[0]], wire_to_bit[qs[1]], tmp)
                prog.bit_xor(tmp, wire_to_bit[qs[2]])
                toffolis_hoisted += 1
                continue
            if op == "SWAP":
                ba, bb = wire_to_bit[qs[0]], wire_to_bit[qs[1]]
                tmp = prog.alloc_bits(1)[0]
                prog.bit_copy(ba, tmp)
                prog.bit_copy(bb, ba)
                prog.bit_copy(tmp, bb)
                continue
        # First non-hoistable gate touching classical wires materializes their current shadow bits onto qubits
        for q in qs:
            if q in c_set and q not in quantum_tainted:
                quantum_tainted.add(q)
        quantum_suffix.append(ln)

    # Materialize classical shadow state back into quantum wires via X ... if b right before suffix
    bridge = [f"X q{w} if b{b}" for w, b in sorted(wire_to_bit.items())]
    combined = prog.emit_kmx() + "\n".join(bridge + quantum_suffix) + "\n"
    return {
        "toffolis_hoisted": toffolis_hoisted,
        "classical_ops_emitted": prog.num_ops,
        "wire_to_bit": wire_to_bit,
        "kmx": combined,
    }


@dataclass
class ProjectiveCostEstimate:
    k: int
    num_steps: int
    mul_toffolis: int
    inv_toffolis: int
    affine_total_toffolis: int
    projective_total_toffolis: int
    toffolis_saved: int


class ProjectiveGF2k:
    """Homogeneous projective ``(N : D)`` coordinate adapter over ``GF(2^k)``.

    Replaces ``S`` sequential rational updates ``x <- x + a_i / b_i`` (which require ``S`` expensive
    finite-field inversions in affine coordinates) with projective polynomial cross-multiplications::

        N' = N * b_i ^ a_i * D
        D' = D * b_i

    followed by **one single finite-field inversion** ``x_final = N_final * D_final^{-1}`` at the
    closing boundary!
    """

    DEFAULT_POLYS: dict[int, int] = {
        2: 0b111,
        3: 0b1011,
        4: 0b10011,
        5: 0b100101,
        6: 0b1000011,
        7: 0b10000011,
        8: 0b100011011,
    }

    def __init__(self, k: int, *, poly: int | None = None) -> None:
        if k < 1 or k > 32:
            raise ValueError("k must be in [1, 32]")
        self.k = int(k)
        if poly is None:
            if k not in self.DEFAULT_POLYS:
                raise ValueError(f"Provide an explicit irreducible polynomial for GF(2^{k})")
            self.poly = self.DEFAULT_POLYS[k]
        else:
            self.poly = int(poly)
        self.mask = (1 << self.k) - 1

    def mul(self, a: int, b: int) -> int:
        x = int(a) & self.mask
        y = int(b) & self.mask
        res = 0
        top = 1 << (self.k - 1)
        for _ in range(self.k):
            if y & 1:
                res ^= x
            carry = x & top
            x = (x << 1) & self.mask
            if carry:
                x ^= self.poly & self.mask
            y >>= 1
        return res

    def inv(self, a: int) -> int:
        x = int(a) & self.mask
        if x == 0:
            return 0
        # Fermat's little theorem in GF(2^k): x^(2^k - 2)
        exp = (1 << self.k) - 2
        res = 1
        base = x
        while exp > 0:
            if exp & 1:
                res = self.mul(res, base)
            base = self.mul(base, base)
            exp >>= 1
        return res

    def eval_affine_sum(self, numerators: Sequence[int], denominators: Sequence[int]) -> int:
        """Evaluate ``sum_i (numerators[i] / denominators[i])`` using per-step affine inversions."""
        acc = 0
        for num, den in zip(numerators, denominators):
            if int(den) == 0:
                raise ZeroDivisionError("Zero denominator in GF(2^k) rational term")
            acc ^= self.mul(int(num), self.inv(int(den)))
        return acc

    def eval_projective_sum(self, numerators: Sequence[int], denominators: Sequence[int]) -> tuple[int, int, int]:
        """Accumulate in homogeneous ``(N : D)`` coordinates and perform ONE closing inversion."""
        N = 0
        D = 1
        for num, den in zip(numerators, denominators):
            b = int(den) & self.mask
            if b == 0:
                raise ZeroDivisionError("Zero denominator in GF(2^k) rational term")
            a = int(num) & self.mask
            N = self.mul(N, b) ^ self.mul(a, D)
            D = self.mul(D, b)
        closed = self.mul(N, self.inv(D))
        return N, D, closed

    @staticmethod
    def _karatsuba_mul_toffolis(k: int) -> int:
        if k <= 1:
            return 1
        half = k // 2
        rem = k - half
        if half == rem:
            return 3 * ProjectiveGF2k._karatsuba_mul_toffolis(half)
        return (
            ProjectiveGF2k._karatsuba_mul_toffolis(half)
            + ProjectiveGF2k._karatsuba_mul_toffolis(rem)
            + ProjectiveGF2k._karatsuba_mul_toffolis(max(half, rem))
        )

    def price_projective_closing(self, num_steps: int) -> ProjectiveCostEstimate:
        """Compare reversible Toffoli cost of ``num_steps`` affine divisions vs projective closing."""
        s = max(1, int(num_steps))
        mul_t = self._karatsuba_mul_toffolis(self.k)
        # Itoh-Tsujii inversion in GF(2^k) uses floor(log2(k-1)) + hw(k-1) - 1 multiplications (reversible x2)
        if self.k == 1:
            inv_muls = 0
        else:
            km1 = self.k - 1
            inv_muls = (km1.bit_length() - 1) + km1.bit_count() - 1
        inv_t = max(1, 2 * inv_muls * mul_t)

        # Affine step: 1 reversible inversion + 1 mul + uncompute inv = inv_t + mul_t per step
        affine_tot = s * (inv_t + mul_t)
        # Projective step: first step just sets (N, D) = (a_0, b_0) (0 muls!); each subsequent step uses 3 muls
        # plus 1 closing inversion + 1 closing mul!
        proj_tot = max(0, s - 1) * (3 * mul_t) + inv_t + mul_t
        return ProjectiveCostEstimate(
            k=self.k,
            num_steps=s,
            mul_toffolis=mul_t,
            inv_toffolis=inv_t,
            affine_total_toffolis=affine_tot,
            projective_total_toffolis=proj_tot,
            toffolis_saved=affine_tot - proj_tot,
        )
