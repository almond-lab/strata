"""Python interface to Strata's C++20 zero-copy .kmb tape and SIMD kernels."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .model import StrataError

try:
    from . import _native as _c
    NATIVE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _c = None
    NATIVE_AVAILABLE = False


def _require_native():
    if not NATIVE_AVAILABLE:
        raise StrataError("strata._native C++20 extension is not compiled in this environment")


class KmbTape:
    """Zero-copy mmap columnar index over a Kickmix .kmb or .kmx circuit."""

    def __init__(self, path: str | Path | None = None, *, kmx_text: str | None = None):
        _require_native()
        try:
            if path is not None:
                self._index = _c.KmbIndex(path=str(path))
            elif kmx_text is not None:
                self._index = _c.KmbIndex(kmx_text=kmx_text)
            else:
                raise StrataError("KmbTape requires either path or kmx_text")
        except (RuntimeError, ValueError) as exc:
            raise StrataError(str(exc)) from exc

    @classmethod
    def open(cls, path: str | Path) -> "KmbTape":
        return cls(path=path)

    @classmethod
    def from_circuit(cls, circuit) -> "KmbTape":
        if isinstance(circuit, KmbTape):
            return circuit
        if isinstance(circuit, (str, Path)) and "\n" not in str(circuit) and Path(circuit).exists():
            return cls(path=circuit)
        return cls(kmx_text=str(circuit))

    def __len__(self) -> int:
        return len(self._index)

    @property
    def num_ops(self) -> int:
        return len(self._index)

    def info(self) -> dict:
        return self._index.info()

    @property
    def sha256(self) -> str:
        return self._index.info()["sha256"]

    @property
    def num_qubits(self) -> int:
        return self._index.info()["num_qubits"]

    @property
    def num_bits(self) -> int:
        return self._index.info()["num_bits"]

    def decode_span(self, start: int = 0, stop: int | None = None) -> list[str]:
        stop = len(self) if stop is None else stop
        if not (0 <= start <= stop <= len(self)):
            raise StrataError(f"Span [{start}, {stop}) out of bounds for {len(self)} operations")
        return self._index.decode_span(int(start), int(stop))

    def fragment_sha256(self, start: int, stop: int) -> str:
        if not (0 <= start <= stop <= len(self)):
            raise StrataError(f"Span [{start}, {stop}) out of bounds")
        return self._index.fragment_sha256(int(start), int(stop))

    def same_span(self, start: int, stop: int, other: "KmbTape", other_start: int, other_stop: int) -> bool:
        return bool(self._index.same_span(int(start), int(stop), other._index, int(other_start), int(other_stop)))

    def causal_cone(self, start: int, stop: int) -> dict:
        if not (0 <= start <= stop <= len(self)):
            raise StrataError(f"Span [{start}, {stop}) out of bounds")
        return self._index.causal_cone(int(start), int(stop))

    def compose_kmb(self, edits, donors_or_output, output_path: str | Path | None = None) -> dict:
        """Compose binary .kmb spans directly from columnar mmap slices.

        Accepts either:
          - ``compose_kmb([(start, stop, donor_idx, d_start, d_stop), ...], donors, output_path)``
          - ``compose_kmb([(start, stop, donor_tape_or_kmx), ...], output_path)``
        """
        if output_path is None:
            out = donors_or_output
            donors: list[KmbTape] = []
            norm_edits: list[tuple[int, int, int, int, int]] = []
            for item in edits:
                if len(item) != 3:
                    raise StrataError("2-arg compose_kmb expects (start, stop, donor) triples")
                a, b, donor_src = item
                dtape = KmbTape.from_circuit(donor_src)
                d_idx = len(donors)
                donors.append(dtape)
                norm_edits.append((int(a), int(b), d_idx, 0, len(dtape)))
        else:
            out = output_path
            donors = [KmbTape.from_circuit(d) for d in donors_or_output]
            norm_edits = [(int(a), int(b), int(di), int(x), int(y)) for a, b, di, x, y in edits]
        try:
            return self._index.compose_kmb(norm_edits, [d._index for d in donors], str(out))
        except RuntimeError as exc:
            raise StrataError(str(exc)) from exc


class BitSlicedSim:
    """256-lane bit-sliced simulator with checkpointed causal-cone differential replay."""

    def __init__(self, tape: KmbTape, *, checkpoint_stride: int = 8192):
        _require_native()
        if not isinstance(tape, KmbTape):
            tape = KmbTape.from_circuit(tape)
        self.tape = tape
        self._engine = _c.BitSlicedEngine(tape._index, checkpoint_stride=int(checkpoint_stride))

    def trace_cuts(self, cuts: list[int] | tuple[int, ...], *, shots: int = 256, seed: int = 1, clean_wires=()) -> dict:
        raw = self._engine.trace_cuts(list(cuts), shots=int(shots), seed=int(seed), clean_wires=tuple(clean_wires))
        nq = raw["num_qubits"]
        rb = raw["row_bytes"]
        parsed_cuts = []
        for entry in raw["cuts"]:
            q_arr = np.frombuffer(entry["packed_qubits"], dtype=np.uint8).reshape((nq, rb)).copy()
            ph_arr = np.frombuffer(entry["z8_phases"], dtype=np.uint8).copy()
            parsed_cuts.append({
                "cut": entry["cut"],
                "packed_qubits": q_arr,
                "z8_phases": ph_arr,
            })
        return {
            "shots": raw["shots"],
            "num_qubits": nq,
            "row_bytes": rb,
            "cuts": parsed_cuts,
            "mean_executed_toffolis": raw["mean_executed_toffolis"],
            "mean_executed_t": raw["mean_executed_t"],
            "elapsed_ms": raw["elapsed_ms"],
        }

    def verify_cone(self, start: int, stop: int, replacement: KmbTape | str, *, seed: int = 1, clean_wires=(), outcomes_mode: int = -1) -> dict:
        rep_arg = replacement._index if isinstance(replacement, KmbTape) else str(replacement)
        try:
            return self._engine.verify_cone(
                int(start), int(stop), rep_arg,
                seed=int(seed), clean_wires=tuple(clean_wires), outcomes_mode=int(outcomes_mode)
            )
        except RuntimeError as exc:
            raise StrataError(str(exc)) from exc

    def verify_cones_batch(self, start: int, stop: int, replacements: list[KmbTape | str], *, seed: int = 1, clean_wires=()) -> dict:
        reps = [r._index if isinstance(r, KmbTape) else str(r) for r in replacements]
        try:
            return self._engine.verify_cones_batch(
                int(start), int(stop), reps, seed=int(seed), clean_wires=tuple(clean_wires)
            )
        except RuntimeError as exc:
            raise StrataError(str(exc)) from exc

    def profile_liveness(self, *, seed: int = 1, clean_wires=()) -> dict:
        return self._engine.profile_liveness(seed=int(seed), clean_wires=tuple(clean_wires))


def gf2_rank_fast(matrix: np.ndarray, row_count: int | None = None):
    """Compute GF(2) rank via word-packed C++ elimination.

    - If ``row_count`` is provided, ``matrix`` is big-endian packed ``(num_functions, ceil(row_count/8))``
      and returns ``(rank, witness_cols)``.
    - If ``row_count`` is ``None``, ``matrix`` is an unpacked ``0/1`` 2D array ``(M, N)`` and returns ``int(rank)``.
    """
    _require_native()
    if row_count is None:
        arr = np.asarray(matrix, dtype=np.uint8) & 1
        if arr.ndim != 2:
            raise StrataError("unpacked GF(2) matrix must be 2D")
        m, n = arr.shape
        if m == 0 or n == 0:
            return 0
        packed = np.ascontiguousarray(np.packbits(arr.T, axis=1, bitorder="big"), dtype=np.uint8)
        rk, _ = _c.gf2_rank_packed(packed, int(n), int(m))
        return int(rk)
    mat = np.ascontiguousarray(matrix, dtype=np.uint8)
    if mat.ndim != 2 or mat.shape[1] != (row_count + 7) // 8:
        raise StrataError("packed_matrix must have shape (num_functions, ceil(row_count/8))")
    return _c.gf2_rank_packed(mat, int(mat.shape[0]), int(row_count))


def solve_zmod_fast(features: np.ndarray, target: np.ndarray, *, modulus: int = 4) -> dict:
    """Solve F @ c = target (mod modulus) for modulus in {2, 4, 8} via C++ OpenMP elimination."""
    _require_native()
    f_arr = np.asarray(features)
    y_arr = np.asarray(target)
    if f_arr.ndim != 2 or y_arr.ndim != 1 or f_arr.shape[0] != y_arr.shape[0]:
        raise StrataError("features must be (N, M) and target must be (N,)")
    f = np.ascontiguousarray(f_arr % modulus, dtype=np.uint8)
    y = np.ascontiguousarray(y_arr % modulus, dtype=np.uint8)
    res = _c.solve_zmod_linear(f, y, int(f.shape[0]), int(f.shape[1]), int(modulus))
    res["consistent"] = bool(res["fits"])
    if res["fits"]:
        res["solution"] = np.asarray(res["coefficients"], dtype=np.int64)
    else:
        res["solution"] = np.zeros(f.shape[1], dtype=np.int64)
    return res


def lovasz_rank_fast(lhs: np.ndarray, rhs: np.ndarray, dim: int, *, trials: int = 3, seed: int = 1) -> dict:
    """Compute exact randomized GF(2^64) Lovasz linear matroid-parity rank in sub-millisecond time."""
    _require_native()
    u = np.ascontiguousarray(lhs, dtype=np.uint64).ravel()
    v = np.ascontiguousarray(rhs, dtype=np.uint64).ravel()
    if u.shape != v.shape:
        raise StrataError("lhs and rhs must have matching lengths")
    res = _c.lovasz_matroid_parity_rank(u, v, int(dim), trials=int(trials), seed=int(seed))
    res["min_pair_ands"] = int(res["max_paired_lines"])
    return res


def eval_phase_margin_fast(target_turns: np.ndarray, feature_masks: np.ndarray, angles_turns: np.ndarray, tolerances_turns: np.ndarray) -> dict:
    """Evaluate worst wrapped phase error / tolerance across millions of inputs via OpenMP."""
    _require_native()
    t = np.ascontiguousarray(target_turns, dtype=np.float64).ravel()
    m = np.ascontiguousarray(feature_masks, dtype=np.uint64).ravel()
    a = np.ascontiguousarray(angles_turns, dtype=np.float64).ravel()
    tol = np.ascontiguousarray(tolerances_turns, dtype=np.float64).ravel()
    return _c.eval_phase_margin(t, m, a, tol)
