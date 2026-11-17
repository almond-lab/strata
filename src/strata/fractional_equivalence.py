"""Exact boundary equality for reversible native basis gates and dyadic diagonals."""
from fractions import Fraction

import kickmix as km
import numpy as np

from .factored import DecisionBudget, DecisionDAG
from .semantic import METADATA, _limits, _parse


VALUE_GATES = frozenset({'X', 'CX', 'CCX', 'SWAP', 'CSWAP'})
SIGN_GATES = frozenset({'NEG', 'Z', 'CZ', 'CCZ'})


def _native(circuit):
    return circuit if isinstance(circuit, km.Circuit) else km.Circuit(circuit)


def has_fractional_phase(circuit):
    native = _native(circuit)
    if not native.operation_statistics()['Z_POW']:
        return False
    return any(Fraction(words[2]).denominator != 1 for line in str(native).splitlines()
               if (words := line.split())[0] == 'Z_POW')


def _operations(native, max_operations):
    """Read canonical SDK text, including precisely quantized native exponents."""
    if len(native) > max_operations:
        return None, 'fractional_operation_budget'
    rows = []
    for line in str(native).splitlines():
        item = _parse(line)
        op = item['op']
        if op in METADATA:
            continue
        if item['condition'] is not None or op not in VALUE_GATES | SIGN_GATES | {'Z_POW'}:
            return None, 'requires unconditional reversible basis gates and native dyadic diagonals'
        if item['status'] != 'supported' and item.get('reason') != 'unsupported_fractional_phase':
            return None, item['reason']
        angle = Fraction(line.split()[2]) % 2 if op == 'Z_POW' else Fraction(int(op in SIGN_GATES))
        if angle.denominator & (angle.denominator - 1):
            return None, 'native_exponent_is_not_dyadic'
        rows.append((line, op, item['wires'], angle))
    return rows, None


def add_phase_bits(dag, phase, active, coefficient):
    """Add active * integer coefficient in place, modulo the bit-plane width."""
    carry = 0
    for bit, old in enumerate(phase):
        other = active if coefficient >> bit & 1 else 0
        if other or carry:
            pair = dag.xor(old, other)
            phase[bit] = dag.xor(pair, carry)
            carry = dag.xor(dag.product(old, other), dag.product(pair, carry))


def _execute(rows, dag, initial, denominator):
    state = initial.copy()
    phase = [0] * denominator.bit_length()
    for _, op, qs, angle in rows:
        if op == 'X':
            state[qs[0]] = dag.xor(1, state[qs[0]])
        elif op in {'CX', 'CCX'}:
            control = state[qs[0]]
            if op == 'CCX':
                control = dag.product(control, state[qs[1]])
            state[qs[-1]] = dag.xor(state[qs[-1]], control)
        elif op in {'SWAP', 'CSWAP'}:
            a, b = qs[-2:]
            delta = dag.xor(state[a], state[b])
            if op == 'CSWAP':
                delta = dag.product(state[qs[0]], delta)
            state[a], state[b] = dag.xor(state[a], delta), dag.xor(state[b], delta)
        else:
            predicate = 1
            for q in qs:
                predicate = dag.product(predicate, state[q])
            add_phase_bits(dag, phase, predicate, int(angle * denominator))
    return state, phase


def _native_replay(original, candidate, wires, arbitrary):
    # Values on untouched quantum wires and all read-only classical registers
    # are identities by inspection of the admitted instruction set.
    exhaustive = len(arbitrary) <= 7
    shots = 2 ** len(arbitrary) if exhaustive else 32
    if shots * (max(original.num_qubits, candidate.num_qubits) +
                max(original.num_bits, candidate.num_bits)) > 2 ** 20:
        return dict(status='skipped', reason='native_cross_check_cell_budget')
    rng = np.random.default_rng(7854)
    values = {q: ((np.arange(shots) >> i) & 1).astype(bool) if exhaustive
              else rng.integers(0, 2, size=shots).astype(bool) for i, q in enumerate(arbitrary)}
    results = []
    for circuit in (original, candidate):
        sim = km.Simulator(batch_size=shots, seed=52)
        for q, inputs in values.items():
            sim.write_across_shots(km.q(q), inputs)
        sim.do(circuit)
        results.append(([sim.read_across_shots(km.q(q)) for q in wires],
                        [Fraction(sim.read_shot_phase(i)) % 2 for i in range(shots)]))
    left, right = results
    passed = all(np.array_equal(a, b) for a, b in zip(left[0], right[0])) and left[1] == right[1]
    return dict(status='passed' if passed else 'different', shots=shots,
                scope='all touched arbitrary inputs' if exhaustive else 'deterministic sample',
                phase_read='exact native Fraction modulo two')


def equivalent_fractional_boundaries(original, candidate, *, clean_inputs=(), max_operations=10000,
                                     max_variables=128, max_terms=65536, max_product_terms=1048576,
                                     required_zero_outputs=()):
    """Compare every output wire and the full phase, including its constant."""
    _limits(max_operations=max_operations, max_variables=max_variables,
            max_terms=max_terms, max_product_terms=max_product_terms)
    before, after = _native(original), _native(candidate)
    clean = set(clean_inputs)
    if any(type(q) is not int or q < 0 for q in clean):
        raise ValueError('clean inputs must be nonnegative physical wire indices')
    zero_outputs = set(required_zero_outputs)
    if any(type(q) is not int or not 0 <= q < before.num_qubits for q in zero_outputs):
        raise ValueError('required zero outputs must be physical source wires')
    base = dict(equivalent=None, clean_input_assumptions=sorted(clean), required_zero_outputs=sorted(zero_outputs),
                evidence='canonical_boolean_decisions_with_native_dyadic_phase_bit_planes',
                semantic_scope='unconditional native reversible basis map and exact phase modulo two')
    if after.num_qubits > before.num_qubits or any(q not in clean for q in range(after.num_qubits, before.num_qubits)):
        return dict(base, status='unknown', reason='undeclared_or_expanded_boundary_layout')
    left, left_error = _operations(before, max_operations)
    right, right_error = _operations(after, max_operations)
    if left_error or right_error:
        return dict(base, status='unknown', reason=left_error or right_error,
                    side='original' if left_error else 'candidate')
    bindings = after.register_data
    if after.num_bits != before.num_bits or any(name not in bindings or list(bindings[name]) != list(members)
            for name, members in before.register_data.items()):
        return dict(base, status='different', equivalent=False, reason='public_register_binding_changed')
    wires = sorted(zero_outputs | {q for program in (left, right) for _, _, qs, _ in program for q in qs})
    arbitrary = [q for q in wires if q not in clean]
    if len(wires) > min(512, max_variables):
        return dict(base, status='unknown', reason='fractional_variable_budget')
    denominator = max((angle.denominator for program in (left, right) for _, _, _, angle in program), default=1)
    dag = DecisionDAG(arbitrary, max_nodes=max_terms, max_steps=max_product_terms)
    try:
        initial = {q: 0 if q in clean else dag.variable(q) for q in wires}
        a, p = _execute(left, dag, initial, denominator)
        b, r = _execute(right, dag, initial, denominator)
        for q in sorted(zero_outputs):
            for side, state in (('original', a), ('candidate', b)):
                if state[q]:
                    return dict(base, status='different', equivalent=False, reason='required_zero_output_is_nonzero',
                                side=side, differing_output=f'q{q}', counterexample=dict(
                                    inputs=dag.witness(state[q]), wire=q, actual=1, expected=0))
        pairs = [(f'q{q}', a[q], b[q]) for q in wires]
        pairs.extend(('phase', x, y) for x, y in zip(p, r))
        for name, x, y in pairs:
            if x != y:
                assignment = dag.witness(dag.xor(x, y))
                def exponent(planes):
                    return str(Fraction(sum(dag.evaluate(root, assignment) << i
                                            for i, root in enumerate(planes)), denominator))
                return dict(base, status='different', equivalent=False, differing_output=name,
                            phase_equal=p == r, counterexample=dict(inputs=assignment,
                            before_value=dag.evaluate(x, assignment), after_value=dag.evaluate(y, assignment),
                            before_phase=exponent(p), after_phase=exponent(r)))
    except (DecisionBudget, RecursionError) as exc:
        return dict(base, status='unknown', reason=str(exc) or 'fractional_decision_depth_budget')
    replay = _native_replay(before, after, wires, arbitrary)
    if replay['status'] == 'different':
        return dict(base, status='different', equivalent=False, reason='native_replay_disagreement', native_cross_check=replay)
    return dict(base, status='equivalent', equivalent=True, phase_equal=True, phase_zero=not any(p),
                fractional_phase_on_some_input=any(p[:-1]), phase_bits=denominator.bit_length(),
                decision_nodes=len(dag.nodes), apply_steps=dag.steps, native_cross_check=replay,
                all_public_inputs=True, all_quantum_outputs=True, exact_phase=True)
