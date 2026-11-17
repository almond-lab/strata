"""Factor a mixed permutation/sign boundary into an entry sign and a value map."""
from collections import defaultdict
from math import isfinite

import kickmix as km

from .accounting import measured_resources
from .algebraic_synthesis import algebraic_boundary_synthesis
from .boolean_synthesis import _layout, joint_boundary_synthesis
from .cost import Formula
from .model import StrataError, digest
from .observable_search import observable_boundary_synthesis
from .semantic import METADATA, _parse, analyze_window, equivalent_windows

_VALUE_GATES = {'X', 'CX', 'CCX', 'SWAP', 'CSWAP'}
_SIGN_GATES = ('NEG', 'Z', 'CZ', 'CCZ')


def _body(circuit):
    return tuple(line for line in str(circuit).splitlines() if line.split()[0] not in METADATA)


def _product_count(lines):
    return sum(line.split()[0] in {'CCX', 'CSWAP'} for line in lines)


def _without_inverse_padding(lines):
    """Discard inverse pairs separated only by disjoint gates when attesting a rewrite."""
    records, per_wire = [], defaultdict(list)
    for line in lines:
        item = _parse(line)
        wires = item['wires']
        if item['op'] == 'CCX':
            wires = (*sorted(wires[:2]), wires[2])
        elif item['op'] in {'SWAP', 'CSWAP'}:
            wires = (*wires[:-2], *sorted(wires[-2:]))
        key = (item['op'], wires)
        prior = per_wire[wires[0]]
        if prior and all(per_wire[q] and per_wire[q][-1] == prior[-1] for q in wires):
            index = prior[-1]
            if records[index][0] == key:
                records[index] = None
                for q in wires:
                    per_wire[q].pop()
                continue
        for q in wires:
            per_wire[q].append(len(records))
        records.append((key, line))
    return tuple(record[1] for record in records if record is not None)


def _value_witness(semantics, original, candidate, clean):
    # A genuine source intermediate has a noninitial value when read and
    # returns to its initial value. Merely targeting an unused wire twice
    # does not meet that condition.
    consumed = {item['wire'] for feature in semantics['features'] if feature['kind'] == 'compute_consume_uncompute'
                for item in feature['restored_wires'] if item['wire'] in clean}
    used = {q for line in candidate for q in _parse(line)['wires']}
    retired = sorted(consumed - used)
    nonlinear = sum(len(term) >= 2 for delta in semantics['changes'].values() for term in delta['monomials'])
    before, after = _product_count(original), _product_count(candidate)
    # Count contracted output monomials as well as actual source products.
    # Removing a pair of redundant CCXs from an already direct monomial
    # circuit cannot by itself attest to a new factored value construction.
    contracted = after < before and after < nonlinear
    return dict(nontrivial_value_rewrite=bool(retired or contracted),
                retired_consumed_clean_wires=retired, source_value_products=before,
                emitted_value_products=after, nonlinear_output_monomials=nonlinear,
                factored_value_products=contracted)


def mixed_boundary_synthesis(circuit, *, clean_inputs=(), cost='Q*toffoli', resource_measure=None,
                            max_orders=3, max_combinations=64, timeout=2.0,
                            max_operations=10000, max_variables=128, max_terms=65536,
                            max_product_terms=1048576):
    """Search checked P;V programs for U|x> = (-1)^p(x)|F(x)>.

    V is the value circuit obtained by deleting diagonal operations. Every
    admitted value gate is an involution, so U followed by the reversed V
    supplies an independent phase-only reference on the original input basis.
    Only that proof circuit receives a doubled instruction allowance.
    """
    from .restructure import _boundary_metadata, _check
    limits = dict(max_operations=max_operations, max_variables=max_variables,
                  max_terms=max_terms, max_product_terms=max_product_terms)
    if any(type(n) is not int or n <= 0 for n in (*limits.values(), max_orders, max_combinations)):
        raise StrataError('mixed synthesis budgets must be positive integers')
    if type(timeout) not in (float, int) or not isfinite(timeout) or timeout <= 0:
        raise StrataError('mixed synthesis timeout must be positive and finite')
    clean = set(_layout(clean_inputs, circuit.num_qubits, 'clean inputs'))
    if len(circuit) > max_operations:
        return dict(status='unknown', reason='mixed_source_operation_budget')
    body = _body(circuit)
    values, signs = [], []
    for line in body:
        item = _parse(line)
        if item['status'] != 'supported' or item['condition'] is not None:
            return dict(status='not_applicable', reason=item.get('reason', 'conditional gate outside reversible sign family'))
        if item['op'] in _VALUE_GATES:
            values.append(line)
        elif item['op'] in _SIGN_GATES:
            signs.append(line)
        elif item['op'] == 'Z_POW':
            if item['sign_exponent']:
                signs.append(f'Z q{item["wires"][0]}')
        else:
            return dict(status='not_applicable', reason='mixed synthesis requires reversible quantum value and sign operations')
    if not signs or not values:
        return dict(status='not_applicable', reason='requires both diagonal and value operations')
    source = analyze_window(body, clean_inputs=clean, **limits)
    if source['status'] != 'exact':
        return dict(status='unknown', reason=source.get('reason'), budget_phase='mixed_source_semantics')
    phase = source['phase']['monomials']
    if not source['modified_wires'] or not phase:
        return dict(status='not_applicable', reason='requires both changed values and nonzero composed sign')
    values = tuple(values)
    unpadded_values = _without_inverse_padding(values)
    reference = body + values[::-1]
    reference_limits = dict(limits, max_operations=2 * max_operations)
    projected = analyze_window(reference, clean_inputs=clean, **reference_limits)
    if projected['status'] != 'exact':
        return dict(status='unknown', reason=projected.get('reason'), budget_phase='entry_phase_reference')
    if projected['modified_wires'] or projected['phase']['monomials'] != phase:
        raise StrataError('mixed source followed by the inverse value circuit failed sign projection')
    value_map = analyze_window(unpadded_values, clean_inputs=clean, **limits)
    if value_map['status'] != 'exact':
        return dict(status='unknown', reason=value_map.get('reason'), budget_phase='value_map')
    if value_map['changes'] != source['changes'] or value_map['phase']['monomials']:
        raise StrataError('stripping diagonal gates changed the value map')
    metadata = _boundary_metadata(circuit)
    max_public = circuit.num_qubits - 1
    while max_public in clean:
        max_public -= 1

    def emit(lines, last=max_public):
        native = km.Circuit('\n'.join([*metadata, *lines]))
        if native.num_qubits <= last:
            pinned = _boundary_metadata(circuit, last, 'strata_public_boundary')
            native = km.Circuit('\n'.join([*pinned, *lines]))
        return native

    value_options, phase_options = {}, {}
    def collector(options, mechanism):
        # Enumerate every distinct algebraic order/lowering that reaches its
        # constructor's resource callback. Component rankings do not decide
        # the winner. The host prices only fully verified mixed circuits.
        def collect(candidate):
            options.setdefault(_body(candidate), mechanism)
            return measured_resources(candidate)
        return collect

    phase_options[tuple(signs)] = 'source_diagonals_at_entry'
    if len(phase) <= max_operations and all(len(term) <= 3 for term in phase):
        direct = tuple(' '.join([_SIGN_GATES[len(term)], *(f'q{q}' for q in term)]) for term in phase)
        phase_options.setdefault(direct, 'direct_entry_monomials')
    # Component inputs must also represent declared clean wires that occur
    # only in the other component. None of these temporary register pins are
    # exposed by the final composition.
    ref_native = emit(reference, circuit.num_qubits - 1)
    val_native = emit(values, circuit.num_qubits - 1)
    phase_result = observable_boundary_synthesis(ref_native, clean_inputs=clean, max_orders=max_orders,
        timeout=timeout / 3, resource_measure=collector(phase_options, 'factored_entry_sign'), **reference_limits)
    # The observable callback measures the proof reference as its baseline.
    # It is not a candidate entry implementation.
    phase_options.pop(reference, None)
    if phase_result['status'] == 'constructed':
        phase_options.setdefault(_body(phase_result['candidate']), 'factored_entry_sign')
    value_result = algebraic_boundary_synthesis(val_native, clean_inputs=clean, max_orders=max_orders,
        timeout=timeout / 3, resource_measure=collector(value_options, 'algebraic_value_synthesis'), **limits)
    value_options[values] = 'source_value_map'
    joint_result = joint_boundary_synthesis(val_native, clean_inputs=clean, timeout=timeout / 3, **limits)
    if joint_result['status'] == 'constructed':
        value_options.setdefault(_body(joint_result['candidate']), 'joint_value_synthesis')
    if value_result['status'] == 'constructed':
        value_options.setdefault(_body(value_result['candidate']), 'algebraic_value_synthesis')

    def verified_components(options, expected, bounds):
        verified, rejected = [], []
        for lines, mechanism in options.items():
            if len(lines) > max_operations:
                rejected.append(dict(mechanism=mechanism, reason='mixed_component_operation_budget'))
                continue
            check = equivalent_windows(expected, lines, clean_inputs=clean, **bounds)
            if check.get('equivalent') is True:
                verified.append((lines, mechanism))
            else:
                rejected.append(dict(mechanism=mechanism, check=check))
        return verified, rejected

    phases, failed_phases = verified_components(phase_options, reference, reference_limits)
    alternatives, failed_values = verified_components(value_options, values, limits)
    if not phases or not alternatives:
        return dict(status='unknown', reason='no verified entry sign and value implementation pair',
                    phase_rejections=failed_phases, value_rejections=failed_values)
    # Try the materialization-removing alternatives before the source value
    # spine if the user imposes a small cross-product cap.
    alternatives.sort(key=lambda option: (option[0] == values, _product_count(option[0]), len(option[0])))
    phases.sort(key=lambda option: (sum(line.startswith(('CCX ', 'CCZ ', 'CSWAP ')) for line in option[0]),
                                    len(option[0])))
    measure = resource_measure or measured_resources
    objective = Formula(cost) if isinstance(cost, str) else cost
    before = measure(circuit)
    best = dict(candidate=str(circuit), resources=before, cost=objective(before),
                phase_method='source_incumbent', value_method='source_incumbent',
                native_instructions=len(circuit), structural_witness=dict(separated_value_and_phase=False),
                proof=dict(status='equivalent', method='source_identity'))
    transformed_reads = [event['at'] for event in source['effects']['phase_events'] if event['contribution']['monomials']
                         and any(poly['monomials'] != ([] if int(q) in clean else [[int(q)]])
                                 for q, poly in event['values'].items())]
    attempts = []
    for value_lines, value_method in alternatives:
        value_witness = _value_witness(value_map, unpadded_values, value_lines, clean)
        value_witness['source_value_products_before_inverse_cancellation'] = _product_count(values)
        for phase_lines, phase_method in phases:
            if len(attempts) >= max_combinations:
                break
            row = dict(phase_method=phase_method, value_method=value_method)
            attempts.append(row)
            if len(phase_lines) + len(value_lines) > max_operations:
                row.update(status='unknown', reason='mixed_native_operation_budget')
                continue
            candidate = emit(phase_lines + value_lines)
            if len(candidate) > max_operations:
                row.update(status='unknown', reason='mixed_native_operation_budget')
                continue
            check = _check(circuit, candidate, clean, limits)
            if check.get('equivalent') is not True:
                row.update(status=check.get('status', 'unknown'), check=check)
                continue
            resources = measure(candidate)
            price = objective(resources)
            row.update(status='verified', cost=price, candidate_hash=digest(str(candidate)), resources=resources)
            if (price, len(candidate)) < (best['cost'], best['native_instructions']):
                best = dict(candidate=str(candidate), resources=resources, cost=price, native_instructions=len(candidate),
                    phase_method=phase_method, value_method=value_method, check=check,
                    phase_body='\n'.join(phase_lines), value_body='\n'.join(value_lines),
                    proof=dict(status='equivalent', method='entry_sign_projection_and_complete_mixed_native_boundary',
                               phase_applied_at='entry', clean_inputs=sorted(clean), all_public_inputs=True, exact_phase=True),
                    structural_witness=dict(value_witness,
                        separated_value_and_phase=value_witness['nontrivial_value_rewrite'] and any(phase),
                        source_transformed_phase_read_indices=transformed_reads,
                        phase_read_coordinates='zero-based source body with metadata removed',
                        entry_sign_depends_on_input=any(phase),
                        source_phase_terms=len(phase), phase_projection_verified=True,
                        retired_scratch_wires=sorted(set(value_witness['retired_consumed_clean_wires']) -
                            {q for line in phase_lines for q in _parse(line)['wires']})))
    return dict(status='constructed', family='entry_sign_and_value_synthesis', **best,
                entry_projection=dict(verified=True, phase_terms=len(phase), source_operations=len(body),
                    reference_operations=len(reference), reference_operation_limit=reference_limits['max_operations'],
                    method='source followed by inverse of its diagonal-stripped value spine'),
                resource_bill=dict(before=before, after=best['resources'], before_cost=objective(before),
                                   after_cost=best['cost'], cost_expression=cost if isinstance(cost, str) else getattr(cost, 'expression', None)),
                search=dict(attempts=attempts, phase_rejections=failed_phases, value_rejections=failed_values,
                            phase_status=phase_result['status'], algebraic_status=value_result['status'],
                            joint_status=joint_result['status'], verified_components=dict(phases=len(phases), values=len(alternatives)),
                            combinations_available=len(phases) * len(alternatives), max_combinations=max_combinations,
                            limits=limits, component_timeout=timeout / 3, max_orders=max_orders,
                            optimality='not claimed; bounded component constructions and compositions'),
                witness_hash=digest(dict(source=str(circuit), candidate=best['candidate'], clean_inputs=sorted(clean))))
