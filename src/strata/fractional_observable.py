"""Synthesize a composed native dyadic phase on the entry basis of a circuit.

Two lowerings compete. A common-angle sum of indicators has binary digits
with Boolean carry, even when predicates overlap. Alternatively, exact Walsh
coefficients express each local indicator as a constant plus weighted parities.
The complete output permutation follows the entry diagonal in either case.
"""
from collections import defaultdict
from fractions import Fraction

import kickmix as km

from .accounting import measured_resources
from .algebraic_synthesis import _Bound, _Budget, _Factors
from .boolean_synthesis import _layout, _product_update
from .cost import Formula
from .fractional_equivalence import VALUE_GATES, _native, _operations, equivalent_fractional_boundaries
from .mixed_observable import _without_inverse_padding
from .model import digest
from .semantic import _limits, analyze_window


def _rotation(q, angle):
    angle %= 2
    if not angle:
        return []
    if angle == 1:
        return [f'Z q{q}']
    # Format the exact terminating decimal. Decimal contexts and binary64
    # both round some kickmix exponents before the native parser sees them.
    bits = angle.denominator.bit_length() - 1
    digits = str(angle.numerator * 5**bits).zfill(bits + 1)
    line = f'Z_POW q{q} {digits[:-bits]}.{digits[-bits:]}'
    if Fraction(str(km.Circuit(line)).split()[2]) != angle:
        raise _Bound('required_rotation_below_native_precision')
    return [line]


def _global_angle(angle, clean):
    if angle % 2 == 0:
        return []
    if angle % 2 == 1:
        return ['NEG']
    q = min(clean, default=0)
    flipped = [f'X q{q}', *_rotation(q, angle), f'X q{q}']
    return flipped if q in clean else _rotation(q, angle) + flipped


def _read_observables(rows, clean, limits):
    # The sign projection is solely an instrumentation pass for extracting the
    # Boolean predicate at each instruction. Every original coefficient stays
    # associated with its instruction and is composed in Z[1/2] modulo two.
    marked = [f'Z q{qs[0]}' if op == 'Z_POW' else line for line, op, qs, _ in rows]
    state = analyze_window(marked, clean_inputs=clean, **limits)
    if state['status'] != 'exact':
        return None, state.get('reason', 'fractional_predicate_analysis')
    weights = defaultdict(Fraction)
    clean_contributions = defaultdict(Fraction)
    for event in state['effects']['phase_events']:
        _, op, qs, angle = rows[event['at']]
        predicate = frozenset(frozenset(term) for term in event['contribution']['monomials'])
        if not predicate or not angle:
            continue
        weights[predicate] += angle
        if op == 'Z_POW' and angle.denominator > 1 and qs[0] in clean:
            clean_contributions[qs[0], predicate] += angle
    weights = {p: a % 2 for p, a in weights.items() if a % 2}
    phase_wires = {q for (q, p), a in clean_contributions.items()
                   if a.denominator > 1 and p in weights and weights[p].denominator > 1}
    value_body = [line for line, op, _, _ in rows if op in VALUE_GATES]
    values = _without_inverse_padding(value_body) if state['modified_wires'] else ()
    return (weights, phase_wires, values), None


def _carry_observables(weights, budget):
    groups = defaultdict(list)
    for predicate, angle in weights.items():
        groups[angle].append(predicate)
    result = defaultdict(Fraction)
    for angle, predicates in groups.items():
        bits = []
        for carry in predicates:
            for j, old in enumerate(bits):
                if not carry:
                    break
                budget.charge(len(old) + len(carry))
                bits[j] = budget.polynomial(old ^ carry)
                carry = budget.product(old, carry)
            if carry:
                bits.append(carry)
        for bit, predicate in enumerate(bits):
            if predicate:
                result[predicate] += angle * (1 << bit)
    return {p: a % 2 for p, a in result.items() if a % 2}


def _factored_rotations(weights, clean, budget, max_operations):
    result = []
    for predicate, angle in weights.items():
        inputs = sorted(set().union(*predicate))
        graph = _Factors(inputs, inputs, budget)
        output = graph.factor(predicate)
        if len(graph.gates) > len(clean):
            raise _Bound('fractional_predicate_clean_workspace_budget')
        available, forward = list(inputs), []
        for gate, helper in zip(graph.gates, sorted(clean)):
            forward.extend(_product_update(gate['left_coefficients'], gate['right_coefficients'], available, helper))
            available.append(helper)
        wires = [q for i, q in enumerate(available, 1) if output >> i & 1]
        if wires:
            q = wires[-1]
            adapter = [f'CX q{other} q{q}' for other in wires[:-1]]
            if output & 1:
                adapter.append(f'X q{q}')
            forward.extend(adapter)
            center = _rotation(q, angle)
        else:
            center = _global_angle(angle, clean) if output & 1 else []
        if len(result) + 2 * len(forward) + len(center) > max_operations:
            raise _Bound('fractional_emission_operation_budget')
        result.extend([*forward, *center, *reversed(forward)])
    return result


def _parity_coefficients(weights, budget):
    coefficients = defaultdict(Fraction)
    for predicate, angle in weights.items():
        wires = sorted(set().union(*predicate))
        constant = frozenset() in predicate
        if all(len(term) <= 1 for term in predicate):
            if constant:
                coefficients[()] += angle
            if wires:
                coefficients[tuple(wires)] += -angle if constant else angle
            continue
        # Expand local nonlinear predicates only. A wide affine parity above
        # has a single coefficient regardless of its supporting wire count.
        if len(wires) > 10:
            raise _Bound('fractional_local_walsh_support_budget')
        size = 1 << len(wires)
        budget.charge(size * (len(predicate) + len(wires)))
        positions = {q: i for i, q in enumerate(wires)}
        masks = [sum(1 << positions[q] for q in t) for t in predicate]
        values = [sum((x & mask) == mask for mask in masks) % 2 for x in range(size)]
        coefficients[()] += angle * values[0]
        stride = 1
        while stride < size:
            for start in range(0, size, stride * 2):
                for i in range(start, start + stride):
                    a, b = values[i], values[i + stride]
                    values[i], values[i + stride] = a + b, a - b
            stride *= 2
        for mask, coefficient in enumerate(values[1:], 1):
            if coefficient:
                parity = tuple(q for j, q in enumerate(wires) if mask >> j & 1)
                coefficients[parity] -= angle * Fraction(2 * coefficient, size)
        if len(coefficients) > budget.max_terms:
            raise _Bound('fractional_walsh_term_budget')
    return {parity: a % 2 for parity, a in coefficients.items() if a % 2}


def _parity_rotations(weights, clean, budget, max_operations):
    coefficients = _parity_coefficients(weights, budget)
    result = _global_angle(coefficients.pop((), 0), clean)
    by_pivot = defaultdict(list)
    for wires, angle in sorted(coefficients.items()):
        by_pivot[wires[-1]].append((set(wires[:-1]), angle))
    for pivot, parities in by_pivot.items():
        previous = set()
        for operands, angle in parities:
            result.extend(f'CX q{q} q{pivot}' for q in sorted(previous ^ operands))
            result.extend(_rotation(pivot, angle))
            previous = operands
            if len(result) > max_operations:
                raise _Bound('fractional_emission_operation_budget')
        result.extend(f'CX q{q} q{pivot}' for q in sorted(previous))
    return result


def fractional_boundary_synthesis(circuit, *, clean_inputs=(), cost='Q*toffoli', resource_measure=None,
                                  max_operations=10000, max_variables=128, max_terms=65536,
                                  max_product_terms=1048576):
    from .restructure import _boundary_metadata

    circuit = _native(circuit)
    limits = dict(max_operations=max_operations, max_variables=max_variables, max_terms=max_terms,
                  max_product_terms=max_product_terms)
    _limits(**limits)
    clean = set(_layout(clean_inputs, circuit.num_qubits, 'clean inputs'))
    rows, error = _operations(circuit, max_operations)
    if error:
        return dict(status='unknown' if 'budget' in error else 'not_applicable', reason=error)
    if not any(angle.denominator > 1 for _, _, _, angle in rows):
        return dict(status='not_applicable', reason='no native fractional rotation')
    source, error = _read_observables(rows, clean, limits)
    if error:
        return dict(status='unknown', reason=error)
    weights, phase_wires, values = source
    measure = resource_measure or measured_resources
    objective = Formula(cost) if isinstance(cost, str) else cost
    before = measure(circuit)
    metadata = _boundary_metadata(circuit)
    public_top = circuit.num_qubits - 1
    while public_top in clean:
        public_top -= 1
    attempts, options, seen = [], [], set()
    fractional_predicates = sum(angle.denominator > 1 for angle in weights.values())
    for method in ('common_angle_boolean_carries', 'exact_walsh_parity_rotations', 'individual_boolean_predicates'):
        budget = _Budget(max_product_terms, max_terms, max_terms, max_product_terms, 3.0)
        try:
            if method == 'exact_walsh_parity_rotations':
                phase = _parity_rotations(weights, clean, budget, max_operations)
            else:
                grouped = _carry_observables(weights, budget) if method == 'common_angle_boolean_carries' else weights
                phase = _factored_rotations(grouped, clean, budget, max_operations)
        except (_Bound, RecursionError) as exc:
            attempts.append(dict(method=method, status='not_lowerable', reason=str(exc) or 'fractional_depth_budget'))
            continue
        body = phase + list(values)
        candidate = km.Circuit('\n'.join([*metadata, *body]))
        if candidate.num_qubits <= public_top:
            pinned = _boundary_metadata(circuit, public_top, 'strata_public_boundary')
            candidate = km.Circuit('\n'.join([*pinned, *body]))
        if len(candidate) > max_operations:
            attempts.append(dict(method=method, status='not_lowerable', reason='fractional_emission_operation_budget'))
            continue
        key = digest(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        check = equivalent_fractional_boundaries(circuit, candidate, clean_inputs=clean, **limits)
        if check.get('equivalent') is not True:
            attempts.append(dict(method=method, status=check['status'], check=check))
            continue
        bill = measure(candidate)
        price = objective(bill)
        used = {int(word[1:]) for line in body for word in line.split() if word.startswith('q')}
        retired = sorted(phase_wires - used)
        rotations = sum(line.startswith('Z_POW ') for line in phase)
        combined = fractional_predicates > 1 and rotations < fractional_predicates
        witness = dict(factored_fractional_observable=bool(retired or combined) and check['fractional_phase_on_some_input'],
                       retired_phase_wires=retired, source_computed_fractional_wires=sorted(phase_wires),
                       distinct_fractional_predicates=fractional_predicates, combined_fractional_predicates=combined,
                       emitted_rotations=rotations)
        attempts.append(dict(method=method, status='verified', candidate_hash=key, cost=price, resources=bill))
        options.append(dict(candidate=str(candidate), selected_method=method, resources=bill, cost=price,
                            structural_witness=witness, check=check, native_instructions=len(candidate)))
    if not options:
        return dict(status='unknown', reason='no bounded fractional lowering passed the complete boundary', attempts=attempts)
    options.sort(key=lambda option: (option['cost'], option['native_instructions']))
    best = options[0]
    return dict(status='constructed', family='exact_fractional_observable', **best, alternatives=options[1:],
                attempts=attempts, proof=best['check'], resource_bill=dict(before=before, after=best['resources'],
                before_cost=objective(before), after_cost=best['cost']), optimality='bounded native alternatives')
