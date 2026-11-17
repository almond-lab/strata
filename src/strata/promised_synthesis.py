"""Boolean synthesis from complete finite care sets with independently checked native output."""
from itertools import combinations
import math

import kickmix as km
import numpy as np

from .accounting import measured_resources
from .cost import Formula
from .factored_semantic import equivalent_factored
from .model import digest
from .domains import window_lines, WindowLimit
from .promised_domain import load_search_domain, promised_addresses, check_promised_candidate, _wires
from .semantic import _parse, _limits


def _packed(values):
    return int.from_bytes(np.packbits(values, bitorder='little').tobytes(), 'little')


def _native_table(circuit, bound):
    """Gather proposal data. A compact zero-dirty replay is never its own proof."""
    compact = bound['type'] == 'address_promise'
    count = bound['address_count'] if compact else bound['assignment_count']
    sim = km.Simulator(batch_size=count, seed=431)
    entry = {}
    if compact:
        addresses = list(promised_addresses(bound))
        for bit, q in enumerate(bound['address_wires']):
            col = np.fromiter(((a >> bit) & 1 for a in addresses), bool, count)
            sim.write_across_shots(km.q(q), col)
            entry[q] = _packed(col)
    else:
        for name in bound['assignments'][0]:
            is_qubit = name.startswith('q')
            number = int(name[1:])
            values = np.fromiter((row[name] for row in bound['assignments']), bool, count)
            sim.write_across_shots(km.q(number) if is_qubit else km.b(number), values)
            entry[number if is_qubit else name] = _packed(values)
    sim.do(circuit)
    delta = {}
    for q in range(circuit.num_qubits):
        diff = entry.get(q, 0) ^ _packed(sim.read_across_shots(km.q(q)))
        if diff:
            delta[q] = diff
    sign = _packed(sim.read_phase_flipped_across_shots())
    if sign:
        delta['phase'] = sign
    return entry, delta, count


def _interpolate(columns, outputs, count, max_degree, max_features):
    """Solve one GF(2) interpolation system for every output; low degrees enter first."""
    rows = {}
    terms = []
    remaining = dict(outputs)
    solved = {}
    for degree in range(max_degree + 1):
        if math.comb(len(columns), degree) + len(terms) > max_features:
            break
        for term in combinations(columns, degree):
            col = (1 << count) - 1
            for key in term:
                col &= columns[key]
            if not col:
                continue
            combo = 1 << len(terms)
            while col:
                pivot = col.bit_length() - 1
                row = rows.get(pivot)
                if row is None:
                    terms.append(term)
                    rows[pivot] = col, combo
                    break
                col ^= row[0]
                combo ^= row[1]
        for key, original in list(remaining.items()):
            col, combo = original, 0
            while col and (row := rows.get(col.bit_length() - 1)) is not None:
                col ^= row[0]
                combo ^= row[1]
            if col == 0:
                solved[key] = [term for i,term in enumerate(terms) if combo >> i & 1]
                del remaining[key]
        if not remaining:
            return solved
    return None


def _cofactor_completion(columns, outputs, count):
    """Discard an address selector when its two partial cofactors can agree."""
    keys = list(columns)
    output_keys = list(outputs)
    care = {}
    # Unpack columns once to avoid repeatedly shifting row-length big integers.
    def unpack(col):
        return np.unpackbits(np.frombuffer(col.to_bytes((count+7)//8, 'little'), dtype=np.uint8),
                             bitorder='little', count=count)
    inputs = [unpack(col) for col in columns.values()]
    results = [unpack(col) for col in outputs.values()]
    for row in range(count):
        i = sum(int(col[row]) << j for j,col in enumerate(inputs))
        word = sum(int(col[row]) << j for j,col in enumerate(results))
        if i in care and care[i] != word:
            return None
        care[i] = word

    def complete(partial, size):
        if size == 1:
            return [partial.get(0, 0)]
        half = size // 2
        low = {i: v for i,v in partial.items() if i < half}
        high = {i-half: v for i,v in partial.items() if i >= half}
        if all(low[i] == high[i] for i in low.keys() & high.keys()):
            return complete(low | high, half) * 2
        return complete(low, half) + complete(high, half)

    coefficients = complete(care, 1 << len(keys))
    for bit in range(len(keys)):
        flag = 1 << bit
        for i in range(len(coefficients)):
            if i & flag:
                coefficients[i] ^= coefficients[i ^ flag]
    polys = {key: [] for key in output_keys}
    for i, word in enumerate(coefficients):
        if word:
            term = tuple(q for bit, q in enumerate(keys) if i >> bit & 1)
            for j, key in enumerate(output_keys):
                if word >> j & 1:
                    polys[key].append(term)
    return polys


def _lower(polynomials, clean, max_operations):
    by_term = {}
    for output, terms in polynomials.items():
        for term in terms:
            by_term.setdefault(term, []).append(output)
    native = []
    for term, consumers in by_term.items():
        conditions = [key for key in term if isinstance(key, str)]
        controls = [key for key in term if isinstance(key, int)]
        targets = [q for q in consumers if q != 'phase']
        phase = 'phase' in consumers
        # Two final controls act directly on any XOR target. A pure sign can
        # instead consume three controls in a CCZ. Materialize a shared product
        # for many targets when its compute/cleanup is cheaper.
        final_arity = 2 if targets else 3
        if targets and len(targets) > 2 and len(controls) >= 2 and len(clean) >= len(controls)-1:
            final_arity = 1
        helpers = max(0, len(controls) - final_arity)
        if helpers > len(clean):
            return None
        forward = []
        for q in clean[:helpers]:
            a,b = controls[:2]
            forward.append(f'CCX q{a} q{b} q{q}')
            controls = [q, *controls[2:]]
        ops = []
        if targets:
            prefix = ('X', 'CX', 'CCX')[len(controls)]
            ops.extend(' '.join([prefix, *(f'q{x}' for x in controls), f'q{q}']) for q in targets)
        if phase:
            prefix = ('NEG','Z','CZ','CCZ')[len(controls)]
            ops.append(' '.join([prefix, *(f'q{x}' for x in controls)]))
        native.extend(f'PUSH_CONDITION if {bit}' for bit in conditions)
        native.extend(forward + ops + forward[::-1])
        native.extend('POP_CONDITION' for _ in conditions)
        if len(native) > max_operations:
            return None
    return native


def _emit(circuit, native, classical):
    from .restructure import _boundary_metadata
    metadata = _boundary_metadata(circuit)
    body = native + classical
    result = km.Circuit('\n'.join(metadata + body))
    if result.num_qubits < circuit.num_qubits:
        metadata = _boundary_metadata(circuit, circuit.num_qubits - 1)
        result = km.Circuit('\n'.join(metadata + body))
    if result.num_bits < circuit.num_bits:
        index = result.num_registers
        name = 'strata_classical_boundary'
        while name in result.register_data:
            name += '_'
        metadata += [f'APPEND_TO_REGISTER b{circuit.num_bits-1} r{index}', f'REGISTER r{index} "{name}"']
        result = km.Circuit('\n'.join(metadata + body))
    return result


def synthesize(circuit, domain, *, clean_inputs=(), cost='Q*toffoli', resource_measure=None,
               max_operations=10000, max_assignments=65536, max_variables=128, max_terms=65536,
               max_product_terms=65536, max_table_cells=2097152, max_address_bits=12, max_features=1024):
    """Compare low-degree GF(2) interpolation and cofactor-based Boolean completions."""
    limits = dict(max_operations=max_operations, max_variables=max_variables, max_terms=max_terms,
                  max_product_terms=max_product_terms, max_assignments=max_assignments)
    _limits(**limits, max_table_cells=max_table_cells, max_address_bits=max_address_bits, max_features=max_features)
    clean = sorted(_wires(list(clean_inputs), circuit.num_qubits))
    bound = load_search_domain(domain, circuit, clean_inputs=clean, max_operations=max_operations,
                               max_assignments=max_assignments, max_table_cells=max_table_cells)
    count = bound.get('address_count', bound['assignment_count'])
    base = dict(family='promised_domain_boolean_completion', domain_type=bound['type'],
                domain_hash=bound['domain_hash'], assignment_count=bound['assignment_count'],
                all_input_equivalence=False, host_promise_proven=False)
    if (count > max_assignments or count * max(1, circuit.num_qubits + circuit.num_bits) > max_table_cells
            or circuit.num_qubits + circuit.num_bits > max_variables):
        return dict(base, status='unknown', reason='promised_source_table_budget')
    try:
        lines, _ = window_lines(circuit, max_lines=max_operations)
    except WindowLimit as exc:
        return dict(base, status='unknown', reason=str(exc))
    classical = []
    for line in lines:
        parsed = _parse(line)
        if parsed['status'] == 'opaque' or parsed['op'] in ('HMR', 'R', 'Z_POW'):
            return dict(base, status='not_applicable', reason='completion requires deterministic Boolean and sign gates')
        if parsed['op'].startswith('BIT_') or parsed['op'] in ('PUSH_CONDITION', 'POP_CONDITION'):
            classical.append(line)
    entry, delta, count = _native_table(circuit, bound)
    if set(clean) & delta.keys():
        return dict(base, status='not_applicable', reason='clean_output_not_restored')
    all_ones = (1 << count) - 1
    columns, seen = {}, {0, all_ones}
    for key, column in entry.items():
        if key not in delta and column not in seen:
            columns[key] = column
            seen.add(column)
    measure = resource_measure or measured_resources
    objective = Formula(cost) if isinstance(cost, str) else cost
    before = measure(circuit)
    before_cost = objective(before)
    trials, candidates = [], {}
    for reverse in (False, True):
        ordered = dict(reversed(columns.items())) if reverse else columns
        for family in ('degree_ordered_interpolation', 'compatible_cofactor_completion'):
            if family == 'degree_ordered_interpolation':
                polys = _interpolate(ordered, delta, count, 3,
                                     min(max_features, max_table_cells // max(1, count)))
            else:
                if (len(ordered) > max_address_bits or
                        (1 << len(ordered)) * max(1,len(delta)) > max_table_cells):
                    continue
                polys = _cofactor_completion(ordered, delta, count)
            trial = dict(completion=family, reversed_order=reverse)
            trials.append(trial)
            if polys is None:
                trial['status'] = 'no_fit'
                continue
            native = _lower(polys, clean, max_operations)
            if native is None:
                trial['status'] = 'lowering_budget'
                continue
            candidate = _emit(circuit, native, classical)
            if len(str(candidate).splitlines()) > max_operations:
                trial['status'] = 'operation_budget'
                continue
            resources = measure(candidate)
            value = objective(resources)
            trial.update(status='priced', cost=value, printed_gate_count=len(native))
            if value < before_cost:
                candidates.setdefault(str(candidate), (value, resources, candidate, trial))
    ordered_candidates = sorted(candidates.values(), key=lambda row: (row[0], len(row[2]), str(row[2])))
    base.update(source_rows_replayed=count, trials=trials, distinct_cheaper_candidates=len(candidates))
    for value, resources, candidate, trial in ordered_candidates:
        check = check_promised_candidate(circuit, candidate, bound, clean_inputs=clean,
                                         max_table_cells=max_table_cells, **limits)
        trial['status'] = check['status']
        if check.get('equivalent') is not True:
            trial['reason'] = check.get('reason', 'boundary_mismatch')
            continue
        outside = check.get('off_domain_witness')
        if bound['type'] == 'finite':
            full = equivalent_factored(lines, str(candidate).splitlines(), clean_inputs=clean,
                                       max_operations=max_operations, max_variables=max_variables,
                                       max_nodes=max_terms, max_apply_steps=32*max_product_terms)
            if full.get('equivalent') is False:
                outside = full.get('counterexample')
        witness = dict(promise_specialization=outside is not None, off_domain_witness=outside,
                       selected_completion=trial['completion'], reversed_order=trial['reversed_order'])
        return dict(base, status='constructed', candidate=str(candidate), cost=value, resources=resources,
                    check=check, structural_witness=witness, witness_hash=digest([bound['domain_hash'],
                    check['original_hash'], check['candidate_hash'], clean, witness]),
                    resource_bill=dict(before=before, after=resources, before_cost=before_cost, after_cost=value,
                                       cost_expression=cost if isinstance(cost, str) else getattr(cost, 'expression', None)))
    return dict(base, status='not_applicable', reason='no cheaper completion passed the complete promised boundary')
