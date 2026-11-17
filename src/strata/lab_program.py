"""Cached native segments and a conservative affine-HMR control proof.

For each input, accepted control flow leaves the qubit basis trajectory
independent of the HMR outcomes m. Each HMR contributes m[j] * q[j] to the
phase; outcome-controlled Z/CZ/CCZ contributes an affine Boolean expression.
Thus phase is p0 + sum(m[j] * f[j]) modulo two, with any fractional native
rotation confined to p0. Zero and unit assignments determine p0 and every
coefficient. Nonlinear control or outcome-dependent basis changes decline
this theorem and use full enumeration, subject to the ordinary budget.
"""
from dataclasses import dataclass
from functools import lru_cache
import shlex

import kickmix as km
import numpy as np

from .model import StrataError


_META = {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG'}
_ALLOWED = {'X', 'CX', 'CCX', 'SWAP', 'CSWAP', 'Z', 'CZ', 'CCZ', 'NEG',
            'Z_POW', 'HMR', 'R', 'BIT_INVERT', 'BIT_STORE0', 'BIT_STORE1',
            'PUSH_CONDITION', 'POP_CONDITION'} | _META
_ZERO = frozenset()
_ONE = frozenset({0})


def _and(a, b):
    """Exact Boolean polynomial product, declining non-affine results."""
    terms = set()
    for x in a:
        for y in b:
            term = x | y
            if term in terms:
                terms.remove(term)
            else:
                terms.add(term)
    return None if any(term.bit_count() > 1 for term in terms) else frozenset(terms)


def _affine(ops):
    bits, stack, m = {}, [], 0
    for _, words in ops:
        op = words[0]
        if op in _META:
            continue
        cond = bits.get(int(words[-1][1:]), _ZERO) if 'if' in words else _ONE
        if op == 'PUSH_CONDITION':
            stack.append(cond)
            continue
        if op == 'POP_CONDITION':
            if stack:
                stack.pop()
            continue
        active = cond
        for condition in stack:
            active = _and(active, condition)
            if active is None:
                return False
        dependent = active not in (_ZERO, _ONE)
        if op == 'HMR':
            if dependent:
                return False
            if active == _ONE:
                bits[int(words[2][1:])] = frozenset({1 << m})
            m += 1
        elif op.startswith('BIT_'):
            bit = int(words[1][1:])
            old = bits.get(bit, _ZERO)
            if op == 'BIT_INVERT':
                bits[bit] = old ^ active
            else:
                change = _and(active, old ^ (_ONE if op == 'BIT_STORE1' else _ZERO))
                if change is None:
                    return False
                bits[bit] = old ^ change
        elif dependent and op not in {'Z', 'CZ', 'CCZ'}:
            # Outcome-controlled basis gates, reset, NEG and fractional rotations
            # are not covered by this restricted theorem.
            return False
    return True


@dataclass(frozen=True)
class Program:
    segments: tuple
    hmr_count: int
    affine_hmr: bool
    fractional_phase: bool
    temporary_bit: int

    def execute(self, sim, batch, check_time):
        stack = []
        for kind, payload in self.segments:
            check_time()
            if kind == 'native':
                sim.do(payload)
                continue
            at, words, native, phase_gate, m = payload
            op = words[0]
            if op == 'PUSH_CONDITION':
                # Push snapshots the bit, even inside an inactive outer condition.
                stack.append(sim.read_across_shots(km.b(int(words[-1][1:]))).copy())
                sim.do(native)
                continue
            if op == 'POP_CONDITION':
                if stack:
                    stack.pop()
                sim.do(native)
                continue
            active = np.ones(len(batch), dtype=np.bool_)
            for condition in stack:
                active &= condition
            if 'if' in words:
                active &= sim.read_across_shots(km.b(int(words[-1][1:])))
            q = int(words[1][1:])
            values = sim.read_across_shots(km.q(q))
            if op == 'R':
                bad = np.flatnonzero(values & active)
                if len(bad):
                    return int(bad[0]), at, q
                sim.do(native)
            else:
                outcomes = np.fromiter(((branch >> m) & 1 for branch, _ in batch),
                                       dtype=np.bool_, count=len(batch))
                sim.write_across_shots(km.b(self.temporary_bit), outcomes & active)
                sim.do(phase_gate)
                values[active] = False
                sim.write_across_shots(km.q(q), values)
                bit = km.b(int(words[2][1:]))
                bvalues = sim.read_across_shots(bit)
                bvalues[active] = outcomes[active]
                sim.write_across_shots(bit, bvalues)
        return None


@lru_cache(maxsize=16)
def compile_program(text):
    circuit = km.Circuit(text)
    lines = text.splitlines()
    ops = [(i, shlex.split(line)) for i, line in enumerate(lines) if line.strip()]
    if any(words[0] not in _ALLOWED for _, words in ops):
        raise StrataError('unsupported lab instruction')
    temporary_bit = circuit.num_bits
    segments, pending, m = [], [], 0

    def flush():
        if pending:
            segments.append(('native', km.Circuit('\n'.join(pending))))
            pending.clear()

    for at, words in ops:
        op = words[0]
        line = lines[at]
        if op in {'HMR', 'R', 'PUSH_CONDITION', 'POP_CONDITION'}:
            flush()
            native = km.Circuit(line)
            phase_gate = km.Circuit(f'Z {words[1]} if b{temporary_bit}') if op == 'HMR' else None
            segments.append(('special', (at, words, native, phase_gate, m)))
            m += op == 'HMR'
        else:
            pending.append(line)
            if len(pending) >= 1024:
                flush()
    flush()
    return Program(tuple(segments), m, _affine(ops),
                   any(words[0] == 'Z_POW' for _, words in ops), temporary_bit)
