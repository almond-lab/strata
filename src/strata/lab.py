"""Exact finite-domain checks using native, bit-sliced kickmix execution.

HMR outcomes are injected exactly. An affine classical-control proof can reduce
the necessary branches to zero and the unit vectors; it never reports these as
2**n simulations. Promised domains come from the adapter and are hash-bound.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
from numbers import Integral
import time

import kickmix as km
import numpy as np

from .lab_program import compile_program
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
    evidence_kind: str
    promise: str
    domain_hash: str
    domain_source: str
    hmr_count: int
    executed_branches: int
    represented_branches: int
    represented_trajectories: int
    executed_cases: int
    reduction: str
    shard: dict


def validate_lab_receipt(lab):
    """Validate a complete exact receipt, including actual versus proved work."""
    def require(condition):
        if not condition:
            raise StrataError('incomplete or inconsistent lab certificate')

    require(isinstance(lab, dict))
    require(lab.get('passed') is True and lab.get('exhaustive') is True)
    require(lab.get('counterexample') is None)
    require(lab.get('evidence_kind') in {'exact_full', 'exact_small'})
    for key in ('cases', 'branches', 'tested', 'hmr_count', 'executed_branches',
                'represented_branches', 'represented_trajectories', 'executed_cases'):
        require(type(lab.get(key)) is int and lab[key] >= (0 if key == 'hmr_count' else 1))
    require(lab['branches'].bit_length() == lab['hmr_count'] + 1)
    require(lab['branches'] & (lab['branches'] - 1) == 0)
    require(lab['represented_branches'] == lab['branches'])
    require(lab['executed_cases'] == lab['cases'])
    require(lab['represented_trajectories'] == lab['cases'] * lab['branches'])
    require(lab['tested'] == lab['cases'] * lab['executed_branches'])
    require(lab.get('shard') == {'index': 0, 'count': 1})
    require(lab.get('domain_source') in {'all', 'adapter'})
    require(isinstance(lab.get('promise'), str) and bool(lab['promise']))
    require(lab['promise'] == 'all' or lab['domain_source'] == 'adapter')
    require(isinstance(lab.get('scope'), dict))
    require(lab['evidence_kind'] != 'exact_small' or bool(lab['scope']))
    for key in ('domain_hash', 'circuit_hash', 'contract_hash'):
        value = lab.get(key)
        require(isinstance(value, str) and len(value) == 64 and
                all(c in '0123456789abcdef' for c in value))
    if lab.get('reduction') == 'none':
        require(lab['executed_branches'] == lab['branches'])
    elif lab.get('reduction') == 'affine_hmr':
        require(lab['hmr_count'] >= 2 and lab['executed_branches'] == lab['hmr_count'] + 1)
    else:
        require(False)


def _domain(contract, domain, max_cases, check_time):
    registers = contract.registers
    public = [r for r in registers if r.role != 'clean']
    full_cases = 1 << sum(len(r.wires) for r in public)
    if domain is None:
        if contract.promise != 'all':
            raise StrataError('promised contract requires an explicit adapter lab domain')
        if full_cases > max_cases:
            raise StrataError('exhaustive domain exceeds budget; cannot label a sample exhaustive')
        rows = []
        for value in range(full_cases):
            check_time()
            row = []
            for r in registers:
                if r.role == 'clean':
                    row.append(0)
                else:
                    row.append(value & ((1 << len(r.wires)) - 1))
                    value >>= len(r.wires)
            rows.append(tuple(row))
        source = 'all'
    else:
        names = {r.name for r in registers}
        required = {r.name for r in public}
        rows = set()
        for inp in domain:
            check_time()
            if not isinstance(inp, dict) or not required <= inp.keys() or inp.keys() - names:
                raise StrataError('lab domain must name every non-clean register and no unknown registers')
            row = []
            for r in registers:
                value = inp.get(r.name, 0)
                if not isinstance(value, Integral) or not 0 <= value < 1 << len(r.wires):
                    raise StrataError('lab domain register value is not an in-range integer: ' + r.name)
                if r.role == 'clean' and value != 0:
                    raise StrataError('lab domain cannot initialize clean scratch to nonzero')
                row.append(int(value))
            rows.add(tuple(row))
            if len(rows) > max_cases:
                raise StrataError('exhaustive domain exceeds budget; cannot label a sample exhaustive')
        if not rows:
            raise StrataError('lab domain must not be empty')
        if contract.promise == 'all' and len(rows) != full_cases:
            raise StrataError('explicit lab domain cannot weaken an all-input contract')
        rows = list(rows)
        source = 'adapter'
    # Ordering and duplicates cannot change either the receipt or shard assignment.
    rows.sort()
    inputs = [dict(zip((r.name for r in registers), row)) for row in rows]
    return inputs, digest(dict(promise=contract.promise, inputs=inputs)), source


def exhaustive(circuit, contract, reference, scale=None, max_trajectories=1_000_000,
               timeout=60, *, domain=None, batch_size=4096, shard_index=0,
               shard_count=1, branch_strategy='auto', evidence_kind=None):
    """Check an entire declared domain, or a deterministic explicitly partial shard.

    ``max_trajectories`` limits executed trajectories, including affine branch
    witnesses. ``branches`` counts represented HMR assignments; ``tested`` counts
    actual input/branch checks. Shards are samples and cannot certify a design.
    ``branch_strategy='full'`` disables the proved reduction for cross-checking.
    """
    start = time.monotonic()

    def check_time():
        if time.monotonic() - start > timeout:
            raise StrataError('exhaustive lab timed out; no certificate')

    for name, value in [('max_trajectories', max_trajectories), ('batch_size', batch_size),
                        ('shard_count', shard_count)]:
        if type(value) is not int or value < 1:
            raise StrataError(name + ' must be a positive integer')
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise StrataError('invalid deterministic lab shard')
    if not isinstance(timeout, (int, float)) or not np.isfinite(timeout) or timeout <= 0:
        raise StrataError('timeout must be finite and positive')
    if branch_strategy not in {'auto', 'full'}:
        raise StrataError('unknown HMR branch strategy')
    if contract.phase != 'exact' or contract.failure != 'none':
        raise StrataError('exhaustive runner requires exact phase and no failure allowance')
    if not isinstance(contract.promise, str) or not contract.promise:
        raise StrataError('contract promise must be nonempty')
    declared = {q for r in contract.registers for q in r.wires}
    if declared != set(range(circuit.num_qubits)):
        raise StrataError('contract must own every emitted wire, including clean scratch')
    program = compile_program(str(circuit))
    check_time()
    hmr = program.hmr_count
    branches = 1 << hmr
    reduced = branch_strategy == 'auto' and hmr >= 2 and program.affine_hmr
    branch_ids = [0, *(1 << m for m in range(hmr))] if reduced else range(branches)
    executed_count = hmr + 1 if reduced else branches
    inputs, domain_hash, domain_source = _domain(
        contract, domain, max_trajectories * shard_count // executed_count, check_time)
    cases = len(inputs)
    selected = list(range(shard_index, cases, shard_count))
    if not selected:
        raise StrataError('lab shard contains no inputs')
    if len(selected) * executed_count > max_trajectories:
        raise StrataError('exhaustive domain exceeds budget; cannot label a sample exhaustive')
    scope = dict(scale or {})
    kind = evidence_kind or ('exact_small' if scope else 'exact_full')
    if kind not in {'exact_full', 'exact_small'} or kind == 'exact_small' and not scope:
        raise StrataError('exact_small evidence requires an explicit scale; invalid evidence kind')
    if shard_count != 1:
        kind = 'sample'
    tested = 0
    executed = set()

    def result(passed, counterexample=None):
        return LabResult(passed, passed and shard_count == 1, cases, branches, tested,
                         counterexample, scope, hashlib.sha256(str(circuit).encode()).hexdigest(),
                         contract.fingerprint, time.monotonic() - start, kind,
                         contract.promise, domain_hash, domain_source, hmr, len(executed),
                         branches, cases * branches, len(selected),
                         'affine_hmr' if reduced else 'none',
                         dict(index=shard_index, count=shard_count))

    # References run once per distinct input, independent of how many HMR branches
    # witness the result. Dirty and clean obligations never come from the oracle.
    expected_values, expected_phases = {}, {}
    for x in selected:
        expected, phase = reference(dict(inputs[x]), scope)
        check_time()
        wants = []
        for r in contract.registers:
            want = 0 if r.role == 'clean' else inputs[x][r.name] if r.role == 'dirty' else expected.get(r.name)
            if want is None:
                raise StrataError('reference omitted boundary register ' + r.name)
            if not isinstance(want, Integral) or not 0 <= want < 1 << len(r.wires):
                raise StrataError('reference returned an invalid boundary register: ' + r.name)
            wants.append(int(want))
        expected_values[x] = wants
        expected_phases[x] = Fraction(phase) % 2

    trajectories = itertools.product(branch_ids, selected)
    qubits = [km.q(q) for q in range(circuit.num_qubits)]
    while batch := list(itertools.islice(trajectories, batch_size)):
        check_time()
        executed.update(branch for branch, _ in batch)
        sim = km.Simulator(batch_size=len(batch), counting_operations=False, seed=0)
        sim.use_same_registers_as(circuit)
        initial = np.zeros((circuit.num_qubits, len(batch)), dtype=np.bool_)
        for r in contract.registers:
            for j, q in enumerate(r.wires):
                initial[q] = [(inputs[x][r.name] >> j) & 1 for _, x in batch]
        if qubits:
            sim.write_across_shots(qubits, initial)
        reset_failure = program.execute(sim, batch, check_time)
        if reset_failure is not None:
            shot, at, q = reset_failure
            branch, x = batch[shot]
            return result(False, dict(input=inputs[x], hmr_branch=branch,
                                      instruction=at, reset_wire=q, reason='reset_nonzero',
                                      values={f'q{q}': dict(actual=1, expected=0)}))
        actual = sim.read_across_shots(qubits) if qubits else initial
        # Native integer phase reads are vectorized; fractional phase is always
        # read as an exact Fraction, never rounded through a phase-flip flag.
        signs = None if program.fractional_phase else sim.read_phase_flipped_across_shots()
        for shot, (branch, x) in enumerate(batch):
            check_time()
            mismatch = {}
            for r, want in zip(contract.registers, expected_values[x]):
                value = sum(int(actual[q, shot]) << j for j, q in enumerate(r.wires))
                if value != want:
                    mismatch[r.name] = dict(actual=value, expected=want)
            phase = Fraction(sim.read_shot_phase(shot)) % 2 if signs is None else Fraction(int(signs[shot]))
            tested += 1
            if mismatch or phase != expected_phases[x]:
                return result(False, dict(input=inputs[x], hmr_branch=branch, values=mismatch,
                                          phase=str(phase), expected_phase=str(expected_phases[x])))
    check_time()
    return result(True)
