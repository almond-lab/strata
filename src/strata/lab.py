"""Exhaustive small-instance contracts, native gate execution and all HMR branches.

The HMR harness injects branch bits and the exact basis-state Kraus sign/reset;
all other supported gates execute in kickmix. This is finite-size evidence, not
an all-size theorem. Runtime budgets fail closed, never become partial passes.
"""
from dataclasses import dataclass, asdict
import itertools
from fractions import Fraction
import time
import shlex
import numpy as np
import kickmix as km
from .kickmix import instructions, PAID
from .model import StrataError, digest


@dataclass
class LabResult:
    passed: bool
    exhaustive: bool
    cases: int
    branches: int
    tested: int
    counterexample: dict | None
    scope: dict
    circuit_hash: str
    contract_hash: str
    seconds: float


def exhaustive(circuit, contract, reference, scale=None, max_trajectories=1_000_000, timeout=60):
    import hashlib
    start = time.monotonic()
    ops = list(instructions(circuit))
    if any(op in {'R', 'PUSH_CONDITION', 'POP_CONDITION'} for _, _, op, _ in ops):
        raise StrataError('exhaustive lab requires explicit clean initialization and flattened conditions')
    allowed = {'X', 'CX', 'CCX', 'SWAP', 'CSWAP', 'Z', 'CZ', 'CCZ', 'NEG', 'Z_POW', 'HMR',
               'BIT_INVERT', 'BIT_STORE0', 'BIT_STORE1', 'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG'}
    if any(op not in allowed for _, _, op, _ in ops):
        raise StrataError('unsupported lab instruction')
    if contract.promise != 'all' or contract.phase != 'exact' or contract.failure != 'none':
        raise StrataError('this exhaustive runner requires all-input exact contracts; supply a domain-specific adapter')
    public = [q for r in contract.registers if r.role != 'clean' for q in r.wires]
    declared = {q for r in contract.registers for q in r.wires}
    if declared != set(range(circuit.num_qubits)):
        raise StrataError('contract must own every emitted wire, including clean scratch')
    hmr = sum(op == 'HMR' for _, _, op, _ in ops)
    cases, branches = 1 << len(public), 1 << hmr
    if cases * branches > max_trajectories:
        raise StrataError('exhaustive domain exceeds budget; cannot label a sample exhaustive')
    tested = 0
    for branch in range(branches):
        # Vectorize all inputs; dirty registers are part of the enumerated domain.
        sim = km.Simulator(batch_size=cases, counting_operations=True, seed=0)
        sim.use_same_registers_as(circuit)
        numbers = np.arange(cases, dtype=np.uint64)
        initial = np.zeros((circuit.num_qubits, cases), dtype=np.bool_)
        for j, q in enumerate(public):
            initial[q] = ((numbers >> j) & 1).astype(bool)
        sim.write_across_shots([km.q(q) for q in range(circuit.num_qubits)], initial)
        m = 0
        for _, line, op, qs in ops:
            if time.monotonic() - start > timeout:
                raise StrataError('exhaustive lab timed out; no certificate')
            if op == 'HMR':
                words = shlex.split(line)
                bit = int(words[2][1:])
                active = np.ones(cases, dtype=bool)
                if 'if' in words:
                    active = sim.read_across_shots(km.b(int(words[-1][1:])))
                outcome = (branch >> m) & 1
                m += 1
                # Phase kickback via a real native Z, with original condition.
                if outcome:
                    sim.do(km.Circuit('Z q' + str(qs[0]) + (' if ' + words[-1] if 'if' in words else '')))
                values = sim.read_across_shots(km.q(qs[0]))
                values[active] = False
                sim.write_across_shots(km.q(qs[0]), values)
                bvalues = sim.read_across_shots(km.b(bit))
                bvalues[active] = bool(outcome)
                sim.write_across_shots(km.b(bit), bvalues)
            else:
                sim.do(km.Circuit(line))
        actual = sim.read_across_shots([km.q(q) for q in range(circuit.num_qubits)])
        for x in range(cases):
            inp = {r.name: sum(int(initial[q, x]) << j for j, q in enumerate(r.wires)) for r in contract.registers}
            expected, expected_phase = reference(inp, scale or {})
            if time.monotonic()-start>timeout:
                raise StrataError('exhaustive lab timed out; no certificate')
            # References specify all output/through values. Clean & dirty are enforced independently.
            mismatch = {}
            for r in contract.registers:
                value = sum(int(actual[q, x]) << j for j, q in enumerate(r.wires))
                want = 0 if r.role == 'clean' else inp[r.name] if r.role == 'dirty' else expected.get(r.name)
                if want is None:
                    raise StrataError('reference omitted boundary register ' + r.name)
                if value != want:
                    mismatch[r.name] = dict(actual=value, expected=want)
            phase = Fraction(sim.read_shot_phase(x)) % 2
            expected_phase = Fraction(expected_phase) % 2
            tested += 1
            if mismatch or phase != expected_phase:
                return LabResult(False, False, cases, branches, tested,
                                 dict(input=inp, hmr_branch=branch, values=mismatch,
                                      phase=str(phase), expected_phase=str(expected_phase)), scale or {},
                                 hashlib.sha256(str(circuit).encode()).hexdigest(), contract.fingerprint,
                                 time.monotonic() - start)
    return LabResult(True, True, cases, branches, tested, None, scale or {},
                     hashlib.sha256(str(circuit).encode()).hexdigest(), contract.fingerprint,
                     time.monotonic() - start)
