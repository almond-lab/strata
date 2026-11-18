"""Boolean synthesis from complete finite care sets with independently checked native output."""
from itertools import combinations, islice
import math

import kickmix as km
import numpy as np

from .accounting import measured_resources
from .cost import Formula
from .factored_semantic import equivalent_factored
from .model import StrataError, digest
from .domains import window_lines, WindowLimit, touched_wires
from .promised_domain import load_search_domain, promised_addresses, check_promised_candidate, _wires
from .semantic import _parse, _limits


def _packed(values):
    return int.from_bytes(np.packbits(values, bitorder='little').tobytes(), 'little')


def _native_table(circuit, bound, quantum_wires):
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
            if is_qubit and number not in quantum_wires:
                continue
            values = np.fromiter((row[name] for row in bound['assignments']), bool, count)
            sim.write_across_shots(km.q(number) if is_qubit else km.b(number), values)
            entry[number if is_qubit else name] = _packed(values)
    sim.do(circuit)
    delta = {}
    for q in sorted(quantum_wires):
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
    remaining_features = max_features
    for degree in range(max_degree + 1):
        examined = min(math.comb(len(columns), degree), remaining_features)
        for term in islice(combinations(columns, degree), examined):
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
        remaining_features -= examined
        if not remaining_features:
            break
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


def _joint_on_care_rows(columns, delta, count, clean, width, solver_limits):
    from .boolean_synthesis import lower_xor_and_witness
    from .floors import exact_minimum_on_rows

    inputs, targets = list(columns), list(delta)
    if any(type(x) is not int for x in inputs + targets) or count > 4096:
        return None, dict(status='screened', reason='exact care-row search needs bounded quantum-only XOR deltas')
    rows = [[(column >> i) & 1 for column in columns.values()] for i in range(count)]
    outputs = [[(column >> i) & 1 for column in delta.values()] for i in range(count)]
    projected = {}
    for row, output in zip(rows, outputs):
        if projected.setdefault(tuple(row), output) != output:
            return None, dict(status='screened', reason='unchanged input projection cannot distinguish output deltas')
    try:
        search = exact_minimum_on_rows(rows, outputs, **solver_limits)
    except StrataError as exc:
        return None, dict(status='unknown', reason=str(exc))
    trial = dict(status=search['kind'], reason=search['reason'], exact_search=search)
    if search['kind'] != 'exact_small':
        return None, trial
    result = lower_xor_and_witness(search, rows, outputs, input_wires=inputs, target_wires=targets,
                                  clean_wires=clean, num_qubits=width)
    if result['status'] != 'constructed':
        return None, dict(trial, status='not_lowerable', reason=result['reason'])
    return result['candidate'].splitlines(), trial


def _shared_factorings(circuit, polynomials, clean, limits):
    from .algebraic_synthesis import _Bound, _Budget, _Factors, _lower as lower_graph, _orders

    if any(type(q) is not int for q in polynomials) or any(
            type(q) is not int for terms in polynomials.values() for term in terms for q in term):
        return
    targets = sorted(polynomials)
    expected = [frozenset(map(frozenset, polynomials[q])) for q in targets]
    inputs = sorted({q for terms in expected for term in terms for q in term})
    for order in _orders(inputs, expected, 4):
        trial = dict(completion='shared_cofactor_graph', order=list(order))
        budget = _Budget(100000, 4096, limits['max_terms'], limits['max_product_terms'], 0.25)
        try:
            graph = _Factors(inputs, order, budget)
            masks = [graph.factor(poly) for poly in expected]
            candidate, _ = lower_graph(circuit, clean, inputs, targets, graph.gates, masks,
                                       budget, limits['max_operations'])
            native = [line for line in str(candidate).splitlines() if _parse(line)['status'] != 'metadata']
            yield native, trial
        except _Bound as exc:
            yield None, dict(trial, status='not_lowerable', reason=str(exc))


def _output_basis_maps(polynomials, clean):
    targets = [q for q in polynomials if type(q) is int and q not in clean]
    pairs = sorted(combinations(targets, 2), key=lambda pair: -len(
        set(polynomials[pair[0]]) & set(polynomials[pair[1]])))
    for control, target in pairs[:12]:
        common = set(polynomials[control]) & set(polynomials[target])
        if not common:
            break
        changed = dict(polynomials)
        changed[target] = sorted(set(polynomials[control]) ^ set(polynomials[target]), key=repr)
        yield changed, control, target


def _ordered_in_place(entry, delta, count, scratch, max_operations, max_features):
    """Try reversible single-target updates in forward and reverse wire order."""
    if any(type(q) is not int for q in delta):
        return
    targets = sorted(delta)
    orders = [targets, targets[::-1]] if len(targets) > 1 else [targets]
    for order in orders:
        values = dict(entry)
        attempt = dict(completion='ordered_in_place_updates', target_order=order)
        native = []
        for target in order:
            columns = {q: col for q, col in values.items() if q != target and q not in scratch}
            fit = _interpolate(columns, {target:delta[target]}, count, 3, max_features)
            update = _lower(fit, scratch, max_operations) if fit is not None else None
            if update is None or len(native) + len(update) > max_operations:
                yield None, dict(attempt, status='no_fit' if fit is None else 'lowering_budget',
                                 blocked_target=target)
                break
            native.extend(update)
            values[target] = values.get(target, 0) ^ delta[target]
        else:
            yield native, attempt


def synthesize(circuit, domain, *, clean_inputs=(), cost='Q*toffoli', resource_measure=None,
               max_operations=10000, max_assignments=65536, max_variables=128, max_terms=65536,
               max_product_terms=65536, max_table_cells=2097152, max_address_bits=12, max_features=1024,
               try_joint=False, max_and=3, timeout=1.0, max_states=10000, max_expansions=100000,
               internal_bits=(), classical_bit_offset=None, required_zero_outputs=None):
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
    try:
        lines, _ = window_lines(circuit, max_lines=max_operations)
    except WindowLimit as exc:
        return dict(base, status='unknown', reason=str(exc))
    used = set(touched_wires(lines))
    classical_wires = set()
    for line in lines:
        parsed = _parse(line)
        classical_wires.update(parsed.get('bits', ()))
        if parsed.get('condition'):
            classical_wires.add(parsed['condition'])
    active_width = len(used) + len(classical_wires)
    base.update(table_width=active_width, table_cells=count * active_width,
                table_limits=dict(max_table_cells=max_table_cells, max_address_bits=max_address_bits))
    if (count > max_assignments or count * max(1, active_width) > max_table_cells
            or active_width > max_variables):
        return dict(base, status='unknown', reason='promised_source_table_budget')
    required_zero = set(clean if required_zero_outputs is None else required_zero_outputs)
    original = circuit
    private = tuple(internal_bits)
    fresh = max(original.num_bits, classical_bit_offset or 0)
    source_has_measurement = any(line.startswith('HMR ') for line in lines)

    def verify(left, right, bits=private):
        return check_promised_candidate(left, right, bound, clean_inputs=clean, internal_bits=bits,
            measurement_mode='independent-map' if bits else 'coupled', max_table_cells=max_table_cells,
            required_zero_outputs=required_zero, **limits)

    if source_has_measurement:
        from .lookup_structure import measured_value_surrogate
        circuit, preparation = measured_value_surrogate(original, clean, private, limits, verify=verify)
        base['measurement_preparation'] = preparation
        if circuit is None:
            return dict(base, status='not_applicable',
                        reason=preparation.get('reason') or preparation.get('proof', {}).get('reason'))
        lines = str(circuit).splitlines()

    if any(line.startswith('R ') for line in lines):
        from .restructure import _boundary_metadata
        retained = [line for line in lines if not line.startswith('R ')]
        normalized = km.Circuit('\n'.join(retained))
        if normalized.num_qubits < circuit.num_qubits:
            metadata = _boundary_metadata(circuit)
            extra = _boundary_metadata(circuit, circuit.num_qubits-1)[len(metadata):]
            normalized = km.Circuit('\n'.join(retained + extra))
        proof = verify(circuit, normalized, tuple(b for b in private if b < circuit.num_bits))
        base['reset_elision_proof'] = proof
        if proof.get('equivalent') is not True:
            return dict(base, status='unknown', reason=proof.get('reason', 'reset_not_proven_zero'))
        circuit, lines = normalized, retained

    from .lookup_structure import split_external_diagonals
    table_lines, external_diagonal = split_external_diagonals(lines)
    if external_diagonal:
        circuit = km.Circuit('\n'.join(table_lines))
    classical = list(external_diagonal)
    for line in table_lines:
        parsed = _parse(line)
        if parsed['status'] == 'opaque' or parsed['op'] in ('HMR', 'Z_POW'):
            return dict(base, status='not_applicable', reason='completion requires deterministic Boolean and sign gates',
                        instruction=line)
        if parsed['op'].startswith('BIT_') or parsed['op'] in ('PUSH_CONDITION', 'POP_CONDITION'):
            classical.append(line)
    entry, delta, count = _native_table(circuit, bound, used)
    if required_zero & delta.keys():
        return dict(base, status='not_applicable', reason='clean_output_not_restored')
    scratch = [q for q in clean if q not in delta]
    all_ones = (1 << count) - 1
    columns, span = {}, {count-1: all_ones}
    for key, column in entry.items():
        if key in delta:
            continue
        remainder = column
        while remainder and (pivot := remainder.bit_length() - 1) in span:
            remainder ^= span[pivot]
        if remainder:
            columns[key] = column
            span[remainder.bit_length()-1] = remainder
    base['projected_input_rank'] = len(columns)
    measure = resource_measure or measured_resources
    objective = Formula(cost) if isinstance(cost, str) else cost
    before = measure(original)
    before_cost = objective(before)
    trials, candidates = [], {}

    def price(native, trial):
        trials.append(trial)
        if native is None:
            trial.setdefault('status', 'lowering_budget')
            return
        pure = _emit(original, native, classical)
        variants = [(pure, (), trial)]
        if source_has_measurement:
            from .lookup_structure import measured_cache_cleanup
            measured = measured_cache_cleanup(pure, clean, fresh)
            if measured is not None:
                measured_trial = dict(trial, cleanup='measured_product_erase')
                trials.append(measured_trial)
                variants.append((measured, (fresh,), measured_trial))
        for candidate, introduced, receipt in variants:
            if len(str(candidate).splitlines()) > max_operations:
                receipt['status'] = 'operation_budget'
                continue
            resources = measure(candidate)
            value = objective(resources)
            receipt.update(status='priced', cost=value, printed_gate_count=len(native))
            if value < before_cost:
                candidates.setdefault(str(candidate), (value, resources, candidate, receipt, introduced))

    factored = set()
    for reverse in (False, True):
        ordered = dict(reversed(columns.items())) if reverse else columns
        for family in ('degree_ordered_interpolation', 'compatible_cofactor_completion'):
            trial = dict(completion=family, reversed_order=reverse)
            if family == 'degree_ordered_interpolation':
                polys = _interpolate(ordered, delta, count, 3,
                                     min(max_features, max_table_cells // max(1, count)))
            else:
                if (len(ordered) > max_address_bits or
                        (1 << len(ordered)) * max(1,len(delta)) > max_table_cells):
                    trials.append(dict(trial, status='screened', reason='cofactor table budget'))
                    continue
                polys = _cofactor_completion(ordered, delta, count)
            if polys is None:
                trials.append(dict(trial, status='no_fit'))
                continue
            price(_lower(polys, scratch, max_operations), trial)
            key = tuple((q, frozenset(terms)) for q, terms in polys.items())
            if key in factored:
                continue
            factored.add(key)
            for native, variant in _shared_factorings(circuit, polys, scratch, limits):
                price(native, variant)
            for changed, control, target in _output_basis_maps(polys, clean):
                adapter = f'CX q{control} q{target}'
                native = _lower(changed, scratch, max_operations)
                price([adapter, *native, adapter] if native is not None else None,
                      dict(completion='encoded_output_pair', adapter=[control, target]))
                for native, trial in _shared_factorings(circuit, changed, scratch, limits):
                    price([adapter, *native, adapter] if native is not None else None,
                          dict(trial, completion='encoded_shared_cofactor_graph', adapter=[control, target]))
    if try_joint:
        native, trial = _joint_on_care_rows(columns, delta, count, scratch, circuit.num_qubits,
            dict(max_and=max_and, timeout=timeout, max_states=max_states, max_expansions=max_expansions))
        price(native, dict(trial, completion='exact_joint_on_care_rows'))
    if (set(delta) - required_zero) & set(entry):
        for native, trial in _ordered_in_place(entry, delta, count, scratch, max_operations,
                                               min(max_features, max_table_cells // max(1, count))):
            price(native, trial)
    ordered_candidates = sorted(candidates.values(), key=lambda row: (row[0], len(row[2])))
    base.update(source_rows_replayed=count, trials=trials, distinct_cheaper_candidates=len(candidates))
    for value, resources, candidate, trial, introduced in ordered_candidates:
        check = verify(original, candidate, private + introduced)
        trial['status'] = check['status']
        if check.get('equivalent') is not True:
            trial['reason'] = check.get('reason', 'boundary_mismatch')
            continue
        outside = check.get('off_domain_witness')
        if bound['type'] == 'finite' and not source_has_measurement:
            full = equivalent_factored(lines, str(candidate).splitlines(), clean_inputs=clean,
                                       max_operations=max_operations, max_variables=max_variables,
                                       max_nodes=max_terms, max_apply_steps=32*max_product_terms)
            if full.get('equivalent') is False:
                outside = full.get('counterexample')
        witness = dict(promise_specialization=outside is not None, off_domain_witness=outside,
                       selected_completion=trial['completion'], reversed_order=trial.get('reversed_order'))
        return dict(base, status='constructed', candidate=str(candidate), cost=value, resources=resources,
                    check=check, introduced_private_bits=list(introduced), structural_witness=witness, witness_hash=digest([bound['domain_hash'],
                    check['original_hash'], check['candidate_hash'], clean, witness]),
                    resource_bill=dict(before=before, after=resources, before_cost=before_cost, after_cost=value,
                                       cost_expression=cost if isinstance(cost, str) else getattr(cost, 'expression', None)))
    reason = 'no cheaper completion passed the complete promised boundary'
    in_place = [trial for trial in trials if trial['completion'] == 'ordered_in_place_updates']
    if (not candidates and (set(delta) - required_zero) & set(entry)
            and not any(trial['status']=='priced' for trial in in_place)):
        seen = {}
        def rows(table):
            vectors = [np.frombuffer(col.to_bytes((count+7)//8, 'little'), dtype=np.uint8)
                       for col in table.values()]
            if not vectors:
                return [b''] * count
            unpacked = np.unpackbits(np.stack(vectors), axis=1, count=count, bitorder='little')
            return map(bytes, np.packbits(unpacked.T, axis=1))
        for key, value in zip(rows(columns), rows(delta)):
            if seen.setdefault(key, value) != value:
                reason = 'in_place_map_out_of_scope'
                break
    return dict(base, status='not_applicable', reason=reason)
