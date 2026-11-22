"""Persistent hot workspace session and 4-tier latency funnel on top of Kickmix."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import tempfile
from typing import Callable, Sequence

import kickmix as km
import numpy as np

from .carrier_screen import screen_carrier_cut
from .model import StrataError
from .native import BitSlicedSim, KmbTape, NATIVE_AVAILABLE


class FunnelCandidate:
    """A candidate replacement or full stage design entering the 4-tier latency funnel."""

    def __init__(
        self,
        name: str,
        replacement_or_start: KmbTape | str | int,
        start_or_stop: int = 0,
        stop_or_replacement: int | KmbTape | str | None = None,
        *,
        estimated_cost_delta: float | None = None,
        parent_toffolis: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.name = str(name)
        if isinstance(replacement_or_start, int) and stop_or_replacement is not None and not isinstance(stop_or_replacement, int):
            # Positional convenience form: FunnelCandidate(name, start, stop, replacement, ...)
            self.start = int(replacement_or_start)
            self.stop: int | None = int(start_or_stop)
            self.replacement: KmbTape | str = stop_or_replacement
        else:
            # Canonical form: FunnelCandidate(name, replacement, start=0, stop=None, ...)
            self.replacement = replacement_or_start  # type: ignore[assignment]
            self.start = int(start_or_stop)
            self.stop = None if stop_or_replacement is None else int(stop_or_replacement)  # type: ignore[arg-type]
        self.estimated_cost_delta = estimated_cost_delta
        self.parent_toffolis = parent_toffolis
        self.metadata = metadata


class Session:
    """Hot in-memory circuit arena providing sub-second profiling, cut traces, and cone verification.

    Keeps the mmap KmbTape and checkpointed 256-lane bit-sliced snapshots warm in RAM so
    successive queries, carrier screens, phase fits, and candidate checks run with 0 ms reload.
    """

    def __init__(self, tape: KmbTape, *, clean_wires=(), checkpoint_stride: int = 8192):
        self.tape = tape
        self.clean_wires = tuple(sorted(set(int(w) for w in clean_wires)))
        self.sim = BitSlicedSim(tape, checkpoint_stride=checkpoint_stride)
        self._cached_profile: dict | None = None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._cached_profile = None

    @property
    def num_qubits(self) -> int:
        return self.tape.num_qubits

    @property
    def num_bits(self) -> int:
        return self.tape.num_bits

    @property
    def num_ops(self) -> int:
        return len(self.tape)

    @classmethod
    def open(cls, path: str | Path, *, clean_wires=(), checkpoint_stride: int = 8192) -> "Session":
        return cls.from_file(path, clean_wires=clean_wires, checkpoint_stride=checkpoint_stride)

    @classmethod
    def from_file(cls, path: str | Path, *, clean_wires=(), checkpoint_stride: int = 8192) -> "Session":
        return cls(KmbTape(path=path), clean_wires=clean_wires, checkpoint_stride=checkpoint_stride)

    @classmethod
    def from_circuit(cls, circuit, *, clean_wires=(), checkpoint_stride: int = 8192) -> "Session":
        if isinstance(circuit, km.Circuit):
            return cls(KmbTape(kmx_text=str(circuit)), clean_wires=clean_wires, checkpoint_stride=checkpoint_stride)
        return cls(KmbTape.from_circuit(circuit), clean_wires=clean_wires, checkpoint_stride=checkpoint_stride)

    @classmethod
    def from_generator(cls, builder_fn: Callable, *args, clean_wires=(), checkpoint_stride: int = 8192, **kwargs) -> "Session":
        emitted = builder_fn(*args, **kwargs)
        return cls.from_circuit(emitted, clean_wires=clean_wires, checkpoint_stride=checkpoint_stride)

    def summary(self) -> dict:
        """Fast zero-copy header & opcode census without running simulation."""
        return self.tape.info()

    def _resolve_clean(self, clean_wires: Sequence[int] | None) -> tuple[int, ...]:
        if clean_wires is None or clean_wires == ():
            return self.clean_wires if clean_wires is None else ()
        return tuple(sorted(set(int(w) for w in clean_wires)))

    def profile(self, *, seed: int = 1, force: bool = False) -> dict:
        """Compute an instant Cost + Untapped Freedom Map for the circuit."""
        if self._cached_profile is not None and not force and seed == 1:
            return self._cached_profile
        t0 = perf_counter()
        meta = self.tape.info()
        live = self.sim.profile_liveness(seed=seed, clean_wires=self.clean_wires)
        q = meta["num_qubits"]
        tof = meta["toffoli_static"]
        t_gates = meta["t_static"]
        eq_tof_per_qubit = round((tof + 0.25 * t_gates) / max(1, q - 1), 3) if q > 1 else 0.0

        report = {
            "kind": "strata_hot_profile",
            "sha256": meta["sha256"],
            "is_kmb": meta["is_kmb"],
            "cost_summary": {
                "qubits": q,
                "classical_bits": meta["num_bits"],
                "operations": meta["num_ops"],
                "toffoli_static": tof,
                "t_static": t_gates,
                "rotations_static": meta["rotations_static"],
                "break_even_toffolis_per_saved_qubit": eq_tof_per_qubit,
                "census": meta["census"],
            },
            "untapped_freedom": {
                "r55_peak_live_qubits": live["r55_peak_live_qubits"],
                "peak_window": [live["first_peak_op"], live["last_peak_op"]],
                "semantic_affine_rank_at_peak": live["semantic_affine_rank_at_peak"],
                "redundant_wires_at_peak": max(0, live["sampled_active_at_peak"] - live["semantic_affine_rank_at_peak"]),
                "affine_wire_loans": live["affine_loan_candidates"],
                "classical_prefix_ops": live["classical_prefix_ops"],
                "hmr_demolish_sites": live["hmr_sites"],
            },
            "latency_ms": {
                "open_ms": round(meta["open_ms"], 3),
                "liveness_ms": round(live["elapsed_ms"], 3),
                "total_ms": round((perf_counter() - t0) * 1000.0, 3),
            },
        }
        if seed == 1:
            self._cached_profile = report
        return report

    def trace_cuts(
        self,
        cuts: Sequence[int],
        *,
        shots: int = 256,
        seed: int = 1,
        clean_wires: Sequence[int] | None = None,
        cut_names: list[str] | None = None,
        widths: list[int] | None = None,
        out_npz: str | Path | None = None,
    ) -> dict:
        """Extract multi-cut packed truth matrices (and optional carrier-screen NPZ) in one pass."""
        cuts_tuple = tuple(int(c) for c in cuts)
        if not cuts_tuple:
            raise StrataError("trace_cuts requires at least one cut")
        cw = self.clean_wires if clean_wires is None else tuple(sorted(set(int(w) for w in clean_wires)))
        tr = self.sim.trace_cuts(cuts_tuple, shots=shots, seed=seed, clean_wires=cw)
        names = cut_names or [f"op_{c}" for c in cuts_tuple]
        if len(names) != len(cuts_tuple):
            raise StrataError("cut_names must match length of cuts")
        w_budgets = widths or [tr["num_qubits"]] * len(cuts_tuple)

        if out_npz is not None:
            payload = {
                "cut_names": np.asarray(names, dtype="U"),
                "row_counts": np.asarray([tr["shots"]] * len(cuts_tuple), dtype=np.int64),
                "widths": np.asarray(w_budgets, dtype=np.int64),
            }
            empty_helpers = np.zeros((0, tr["row_bytes"]), dtype=np.uint8)
            for idx, entry in enumerate(tr["cuts"]):
                payload[f"live_{idx}"] = entry["packed_qubits"]
                payload[f"helpers_{idx}"] = empty_helpers
            np.savez(out_npz, **payload)

        carrier_summaries = []
        empty_h = np.zeros((0, tr["row_bytes"]), dtype=np.uint8)
        for idx, entry in enumerate(tr["cuts"]):
            cs = screen_carrier_cut(entry["packed_qubits"], empty_h, row_count=tr["shots"], width=int(w_budgets[idx]))
            carrier_summaries.append({
                "name": names[idx],
                "cut": entry["cut"],
                **cs,
                "z8_phases": entry["z8_phases"],
                "packed_qubits": entry["packed_qubits"],
            })

        return {
            "kind": "strata_cut_trace",
            "circuit_sha256": self.tape.sha256,
            "shots": tr["shots"],
            "num_qubits": tr["num_qubits"],
            "mean_executed_toffolis": tr["mean_executed_toffolis"],
            "elapsed_ms": tr["elapsed_ms"],
            "out_npz": str(out_npz) if out_npz is not None else None,
            "cuts": carrier_summaries,
        }

    def verify_cone(
        self,
        start: int,
        stop: int,
        replacement: KmbTape | str,
        *,
        seed: int = 1,
        clean_wires: Sequence[int] | None = None,
        outcomes_mode: int = -1,
    ) -> dict:
        """Verify a local replacement on [start, stop) in milliseconds via checkpointed causal-cone replay."""
        cw = self.clean_wires if clean_wires is None else tuple(sorted(set(int(w) for w in clean_wires)))
        return self.sim.verify_cone(
            start, stop, replacement, seed=seed, clean_wires=cw, outcomes_mode=outcomes_mode
        )

    def verify_cones_batch(
        self,
        start: int,
        stop: int,
        replacements: list[KmbTape | str],
        *,
        seed: int = 1,
        clean_wires: Sequence[int] | None = None,
    ) -> dict:
        """Evaluate multiple candidate replacements on [start, stop) in parallel OpenMP threads."""
        cw = self.clean_wires if clean_wires is None else tuple(sorted(set(int(w) for w in clean_wires)))
        return self.sim.verify_cones_batch(start, stop, replacements, seed=seed, clean_wires=cw)

    def funnel(
        self,
        candidates: Sequence[FunnelCandidate],
        *,
        max_toffoli_delta: float = 0.0,
        seed: int = 1,
        clean_wires: Sequence[int] | None = None,
        tier0_filter: Callable[[FunnelCandidate], bool] | None = None,
        tier2_seeds: tuple[int, ...] = (11, 29, 47, 83),
        official_verifier: Callable[[km.Circuit], dict] | None = None,
    ) -> dict:
        """Run candidates through Strata's 4-Tier Latency Funnel (Tier 0 -> Tier 1 -> Tier 2 -> Tier 3)."""
        t0 = perf_counter()
        cw = self.clean_wires if clean_wires is None else tuple(sorted(set(int(w) for w in clean_wires)))
        tier0_survivors = []
        rejected = []

        # Tier 0 (< 1 ms): cheap static cost / algebraic filter
        t_tier0 = perf_counter()
        parent_tof = self.tape.info()["toffoli_static"]
        for cand in candidates:
            stop = len(self.tape) if cand.stop is None else cand.stop
            if tier0_filter is not None and not tier0_filter(cand):
                rejected.append({"name": cand.name, "rejected_at": "tier0_algebraic", "reason": "failed_tier0_filter"})
                continue
            rep_tape = KmbTape.from_circuit(cand.replacement)
            if cand.parent_toffolis is not None:
                span_before = int(cand.parent_toffolis)
            else:
                span_before = sum(1 for ln in self.tape.decode_span(cand.start, stop) if ln.startswith(("CCX ", "CCZ ", "CSWAP ")))
            span_after = rep_tape.info()["toffoli_static"]
            delta = span_after - span_before if cand.estimated_cost_delta is None else cand.estimated_cost_delta
            if delta > max_toffoli_delta:
                rejected.append({"name": cand.name, "rejected_at": "tier0_cost", "toffoli_delta": delta})
                continue
            tier0_survivors.append((cand, stop, rep_tape, delta))
        tier0_ms = (perf_counter() - t_tier0) * 1000.0

        # Tier 1 (< 20 ms): 256-shot checkpointed causal-cone replay
        t_tier1 = perf_counter()
        tier1_survivors = []
        for cand, stop, rep_tape, delta in tier0_survivors:
            v1 = self.verify_cone(cand.start, stop, rep_tape, seed=seed, clean_wires=cw, outcomes_mode=-1)
            if not v1["passed"]:
                rejected.append({
                    "name": cand.name,
                    "rejected_at": "tier1_cone_256",
                    "reason": v1["reason"],
                    "counterexample_lane": v1["counterexample_lane"],
                })
                continue
            tier1_survivors.append((cand, stop, rep_tape, delta, v1))
        tier1_ms = (perf_counter() - t_tier1) * 1000.0

        # Tier 2 (multi-batch 1,024+ shots across independent seeds)
        t_tier2 = perf_counter()
        tier2_survivors = []
        for cand, stop, rep_tape, delta, v1 in tier1_survivors:
            ok = True
            for sd in tier2_seeds:
                chk = self.verify_cone(cand.start, stop, rep_tape, seed=sd, clean_wires=cw, outcomes_mode=0)
                if not chk["passed"]:
                    ok = False
                    rejected.append({"name": cand.name, "rejected_at": "tier2_multiseed", "seed": sd, "reason": chk["reason"]})
                    break
            if ok:
                tier2_survivors.append((cand, stop, rep_tape, delta, v1))
        tier2_ms = (perf_counter() - t_tier2) * 1000.0

        # Tier 3 (optional official verifier on full spliced circuit for winners)
        t_tier3 = perf_counter()
        winners = []
        tier2_survivors.sort(key=lambda item: item[3])
        for cand, stop, rep_tape, delta, v1 in tier2_survivors:
            entry = {
                "name": cand.name,
                "start": cand.start,
                "stop": stop,
                "toffoli_delta": delta,
                "projected_toffoli_static": parent_tof + delta,
                "cone_qubits": v1["cone_qubits"],
                "tier1_cone_ms": round(v1["elapsed_ms"], 3),
                "tier": "tier2_verified",
            }
            if official_verifier is not None:
                with tempfile.TemporaryDirectory(prefix="strata-tier3-") as td:
                    merged_path = Path(td) / "merged.kmb"
                    self.tape.compose_kmb([(cand.start, stop, 0, 0, len(rep_tape))], [rep_tape], merged_path)
                    full_circ = km.Circuit.from_file(str(merged_path), format="kmb")
                verdict = official_verifier(full_circ)
                entry["official"] = verdict
                if not verdict.get("passed", verdict.get("status") == "PASS"):
                    rejected.append({"name": cand.name, "rejected_at": "tier3_official", "official": verdict})
                    continue
                entry["tier"] = "tier3_official_verified"
            winners.append(entry)
        tier3_ms = (perf_counter() - t_tier3) * 1000.0

        return {
            "kind": "strata_latency_funnel",
            "submitted_candidates": len(candidates),
            "tier0_count": len(tier0_survivors),
            "tier1_count": len(tier1_survivors),
            "tier2_count": len(tier2_survivors),
            "tier0_survivors": len(tier0_survivors),
            "tier1_survivors": len(tier1_survivors),
            "tier2_survivors": len(tier2_survivors),
            "survivors": winners,
            "winners": winners,
            "rejected": rejected,
            "timings_ms": {
                "tier0_ms": round(tier0_ms, 3),
                "tier1_ms": round(tier1_ms, 3),
                "tier2_ms": round(tier2_ms, 3),
                "tier3_ms": round(tier3_ms, 3),
                "total_ms": round((perf_counter() - t0) * 1000.0, 3),
            },
        }


def benchmark_latency(*, num_ops: int = 100_000, num_qubits: int = 64, num_cones: int = 16) -> dict:
    """Run an end-to-end latency benchmark of Strata v0.7.0's native kernels."""
    from .native import lovasz_rank_fast, solve_zmod_fast

    nq = max(4, int(num_qubits))
    ncones = max(1, int(num_cones))
    b = km.CircuitBuilder()
    qs = b.alloc_qubits(nq)
    bs = b.alloc_bits(16)
    for i in range(num_ops):
        a = i % (nq - 2)
        mod = i % 5
        if mod == 0:
            b.cx(qs[a], qs[a + 1])
        elif mod == 1:
            b.ccx(qs[a], qs[a + 1], qs[a + 2])
        elif mod == 2:
            b.cz(qs[a], qs[a + 2])
        elif mod == 3:
            b.x(qs[a])
        else:
            b.swap(qs[a], qs[a + 1])
    _ = bs
    circ = b.finish_circuit()

    with tempfile.TemporaryDirectory(prefix="strata-bench-") as td:
        kmb_path = Path(td) / "bench.kmb"
        donor_path = Path(td) / "donor.kmb"
        out_path = Path(td) / "composed.kmb"
        circ.to_file(str(kmb_path), format="kmb")

        db = km.CircuitBuilder()
        dq = db.alloc_qubits(nq)
        db.cx(dq[0], dq[1])
        db.x(dq[0])
        db.finish_circuit().to_file(str(donor_path), format="kmb")

        tape = KmbTape(path=kmb_path)
        donor = KmbTape(path=donor_path)
        info = tape.info()
        sess = Session(tape, clean_wires=range(min(16, nq), nq), checkpoint_stride=max(1024, num_ops // 16))
        prof = sess.profile()

        mid = max(0, num_ops // 2 - 2)
        span_lines = "\n".join(tape.decode_span(mid, mid + 2))
        # Warm checkpoint once, then time hot causal-cone verification
        sess.verify_cone(mid, mid + 2, span_lines)
        cone = sess.verify_cone(mid, mid + 2, span_lines)
        batch = sess.verify_cones_batch(mid, mid + 2, [span_lines] * ncones)
        splice = tape.compose_kmb([(mid, mid + 2, 0, 0, len(donor))], [donor], out_path)

    rng = np.random.default_rng(7)
    lhs = rng.integers(1, 1 << 24, size=64, dtype=np.uint64)
    rhs = rng.integers(1, 1 << 24, size=64, dtype=np.uint64)
    lovasz = lovasz_rank_fast(lhs, rhs, 24)

    feats = rng.integers(0, 4, size=(256, 64), dtype=np.uint8)
    coeffs_true = rng.integers(0, 4, size=(64,), dtype=np.uint8)
    target = (feats.astype(np.uint32) @ coeffs_true.astype(np.uint32) % 4).astype(np.uint8)
    z4 = solve_zmod_fast(feats, target, modulus=4)

    open_ms = round(info["open_ms"], 3)
    cone_ms = round(cone["elapsed_ms"], 3)
    per_cand_ms = round(batch["per_candidate_ms"], 4)
    return {
        "kind": "strata_v07_latency_benchmark",
        "native_available": NATIVE_AVAILABLE,
        "num_ops": num_ops,
        "num_qubits": nq,
        "num_cones": ncones,
        "kmb_open_and_index_ms": open_ms,
        "kmb_open_index_ms": open_ms,
        "profile_liveness_ms": prof["latency_ms"]["liveness_ms"],
        "hot_causal_cone_verify_ms": cone_ms,
        "single_cone_256_ms": cone_ms,
        "batched_candidates_per_cand_ms": per_cand_ms,
        "batch_per_candidate_ms": per_cand_ms,
        "binary_kmb_patch_compose_ms": round(splice["elapsed_ms"], 3),
        "lovasz_matroid_parity_24d_ms": round(lovasz["elapsed_ms"], 4),
        "z4_linear_phase_solve_256x64_ms": round(z4["elapsed_ms"], 3),
    }
