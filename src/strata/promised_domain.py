"""Bind finite input promises and compare complete native basis/sign boundaries."""
from bisect import bisect_left
import hashlib
import json
from pathlib import Path

from .domains import equivalent_on_domain, load_domain, window_lines, WindowLimit
from .factored import DecisionBudget, DecisionDAG
from .factored_semantic import _parsed, _execute
from .model import StrataError, digest
from .semantic import _limits
from .streaming import native_lines


def _read(spec):
    if isinstance(spec, (Path, str)):
        try:
            spec = json.loads(Path(spec).read_text())
        except (OSError, ValueError) as exc:
            raise StrataError(f'cannot read promised domain: {exc}') from exc
    if not isinstance(spec, dict):
        raise StrataError('a promised domain must be a JSON object')
    if spec.get('type') == 'boundary_observations':
        raise StrataError('boundary observations cannot declare a complete input promise')
    return spec


def _wires(values, width):
    if (not isinstance(values, (list, tuple)) or
            any(type(q) is not int or not 0 <= q < width for q in values) or
            len(set(values)) != len(values)):
        raise StrataError('promise wire lists require distinct in-range physical qubit IDs')
    return list(values)


def parse_address_promise(spec, circuit, *, clean_inputs=(), max_addresses=65536):
    """Normalize a complete compact Cartesian promise without expanding its interval.

    A legal address is combined with every arbitrary-wire word, every classical
    input, and zeros on all declared clean wires. Address bit order is LSB first.
    The list budget applies to explicit address lists, never interval length.
    """
    _limits(max_addresses=max_addresses)
    spec = _read(spec)
    fields = {'version', 'type', 'address_wires', 'arbitrary_wires', 'clean_wires', 'interval',
              'addresses', 'num_qubits', 'domain_hash', 'address_count', 'assignment_count'}
    if (set(spec) - fields or spec.get('type') != 'address_promise' or
            type(spec.get('version')) is not int or spec['version'] != 1):
        raise StrataError('expected address_promise version 1 and supported fields')
    width = circuit.num_qubits
    address, arbitrary, clean = [_wires(spec.get(name), width)
                                 for name in ('address_wires', 'arbitrary_wires', 'clean_wires')]
    if not address or len(address + arbitrary + clean) != width or len(set(address + arbitrary + clean)) != width:
        raise StrataError('address, arbitrary, and clean wires must partition the entire physical boundary')
    if sorted(clean) != sorted(_wires(list(clean_inputs), width)):
        raise StrataError('promise clean_wires must agree with declared clean inputs')
    if ('addresses' in spec) == ('interval' in spec):
        raise StrataError('supply exactly one of addresses or an inclusive interval')
    limit = 1 << len(address)
    key = 'interval' if 'interval' in spec else 'addresses'
    values = spec[key]
    if not isinstance(values, list) or not values:
        raise StrataError('promise addresses or interval must be a nonempty integer list')
    if key == 'addresses' and len(values) > max_addresses:
        raise StrataError('explicit address list exceeds assignment budget')
    if any(type(x) is not int or not 0 <= x < limit for x in values):
        raise StrataError('address outside unsigned little-endian register')
    if key == 'interval':
        if len(values) != 2 or values[1] < values[0]:
            raise StrataError('interval must be [inclusive_min, inclusive_max]')
        count = values[1] - values[0] + 1
    else:
        values = sorted(set(values))
        count = len(values)
    bound = dict(version=1, type='address_promise', num_qubits=width, address_wires=address,
                 arbitrary_wires=sorted(arbitrary), clean_wires=sorted(clean), **{key: list(values)})
    bound.update(domain_hash=digest(bound), address_count=count, assignment_count=count << len(arbitrary))
    if any(k in spec and (type(spec[k]) is not type(bound[k]) or spec[k] != bound[k])
           for k in ('domain_hash', 'num_qubits', 'address_count', 'assignment_count')):
        raise StrataError('compact domain binding or size disagrees with its contents')
    return bound


def promised_addresses(bound):
    if 'interval' in bound:
        lo, hi = bound['interval']
        return range(lo, hi + 1)
    return iter(bound['addresses'])


def load_search_domain(domain, circuit, *, clean_inputs=(), max_assignments=65536, max_operations=10000,
                       max_table_cells=2097152):
    """Bind a host declaration to the full supplied circuit boundary."""
    _limits(max_assignments=max_assignments, max_operations=max_operations, max_table_cells=max_table_cells)
    spec = _read(domain)
    if spec.get('type') == 'address_promise':
        return parse_address_promise(spec, circuit, clean_inputs=clean_inputs, max_addresses=max_assignments)
    if (spec.get('type', 'finite') != 'finite' or spec.get('omitted_classical_inputs') or
            spec.get('value_dependency_proof') or
            spec.get('scope', 'finite_promised_boundary_assignments') != 'finite_promised_boundary_assignments'):
        raise StrataError('discovery requires a complete finite input promise')
    width = circuit.num_qubits + circuit.num_bits
    row_budget = min(max_assignments, max_table_cells // max(1, width))
    if not row_budget:
        raise StrataError('finite domain physical size exceeds table-cell budget')
    bound = load_domain(spec, circuit, clean_inputs=clean_inputs, max_operations=max_operations,
                        max_assignments=row_budget)
    if 'assignment_count' in spec and spec['assignment_count'] != bound['assignment_count']:
        raise StrataError('finite domain assignment count disagrees with its contents')
    return bound


def _address_root(dag, bound):
    wires = bound['address_wires']
    if 'interval' in bound:
        def at_most(maximum):
            if maximum < 0:
                return 0
            root = 1
            for bit, q in enumerate(wires):
                x = dag.variable(q)
                if maximum >> bit & 1:
                    root = dag.xor(1, dag.product(x, dag.xor(1, root)))
                else:
                    root = dag.product(dag.xor(1, x), root)
            return root
        lo, hi = bound['interval']
        return dag.product(at_most(hi), dag.xor(1, at_most(lo - 1)))
    # Address variables occur most-significant first in this decision manager.
    values = bound['addresses']
    def trie(bit, begin, end, base):
        if begin == end:
            return 0
        if end - begin == 1 << (bit + 1):
            return 1
        high = base + (1 << bit)
        cut = bisect_left(values, high, begin, end)
        return dag.node(dag.order[wires[bit]], trie(bit-1, begin, cut, base),
                        trie(bit-1, cut, end, high))
    return trie(len(wires)-1, 0, len(values), 0)


def _compact_check(original, candidate, bound, clean, mode, private, limits, evidence):
    try:
        left, right = [_parsed(window_lines(c, max_lines=limits['max_operations'])[0], limits)
                       for c in (original, candidate)]
    except WindowLimit as exc:
        return dict(evidence, status='unknown', equivalent=None, reason=str(exc))
    for side, program in [('baseline', left), ('candidate', right)]:
        if isinstance(program, dict):
            return dict(evidence, status='unknown', equivalent=None, side=side, reason=program['reason'])
    address = bound['address_wires']
    wires, bits = left[1] | right[1], left[2] | right[2]
    outcomes = [f'm{i}' for i in range(max(left[3], right[3]))]
    variables = sorted(bits) + list(reversed(address)) + sorted(wires - set(address)) + outcomes
    if len(variables) > min(limits['max_variables'], 512):
        return dict(evidence, status='unknown', equivalent=None, reason='decision_variable_budget')
    dag = DecisionDAG(variables, max_nodes=limits['max_terms'], max_steps=32 * limits['max_product_terms'])
    try:
        care = _address_root(dag, bound)
        dag.pins.add(care)
        maps = []
        for side, program in [('baseline', left), ('candidate', right)]:
            result = _execute(program, dag, clean)
            if 'status' in result:
                return dict(evidence, status='unknown', equivalent=None, side=side, reason=result['reason'])
            values = {f'q{q}': result['state'].get(q, 0 if q in clean else dag.variable(q)) for q in wires}
            values.update({b: result['classical'].get(b, dag.variable(b)) for b in bits if b not in private})
            values['phase'] = result['phase']
            for q in clean & wires:
                leaked = dag.product(care, values[f'q{q}'])
                if leaked:
                    return dict(evidence, status='different', equivalent=False, side=side,
                                reason='clean_output_not_restored', output=f'q{q}',
                                counterexample={str(k): v for k,v in dag.witness(leaked).items()})
            if mode == 'coupled':
                for i, event in enumerate(result['events']):
                    for field in ('active', 'measured_value'):
                        key = f'event[{i}].{event["wire"]}.{event["bit"]}.{field}'
                        values[key] = event[field]
            else:
                for name, value in values.items():
                    if set(outcomes) & dag.support(dag.product(care, value)):
                        return dict(evidence, status='unknown', equivalent=None, side=side, output=name,
                                    reason='public_value_or_phase_depends_on_outcomes')
            dag.pins.update(values.values())
            maps.append(values)
        a, b = maps
        if a.keys() != b.keys():
            return dict(evidence, status='unknown', equivalent=None, reason='measurement_event_alignment_unproved')
        outside = None
        # Provenance must align before ordinal outcomes can be coupled.
        for name in sorted(a, key=lambda name: not name.startswith('event[')):
            difference = dag.xor(a[name], b[name])
            if not difference:
                continue
            mismatch = dag.product(care, difference)
            if mismatch:
                event = name.startswith('event[')
                return dict(evidence, status='unknown' if event else 'different',
                            equivalent=None if event else False, output=name,
                            reason='measurement_event_alignment_unproved' if event else 'public_value_or_phase_mismatch',
                            counterexample={str(k): v for k,v in dag.witness(mismatch).items()})
            if outside is None and not name.startswith('event['):
                outside = dict(output=name, inputs={str(k):v for k,v in dag.witness(difference).items()})
    except (DecisionBudget, RecursionError) as exc:
        return dict(evidence, status='unknown', equivalent=None, reason=str(exc) or 'decision_depth_budget')
    return dict(evidence, status='equivalent', equivalent=True, clean_outputs_restored=True,
                original_outcomes=left[3], candidate_outcomes=right[3],
                measurement_independent=mode == 'independent-map', exact_phase=True,
                off_domain_witness=outside, decision_nodes=len(dag.nodes), apply_steps=dag.steps)


def _canonical_hash(circuit):
    # Match str(circuit)'s canonical bytes, including its lack of a final LF.
    result = hashlib.sha256()
    with native_lines(circuit) as lines:
        for index, line in enumerate(lines):
            if index:
                result.update(b'\n')
            result.update(line.rstrip('\r\n').encode())
    return result.hexdigest()


def check_promised_candidate(original, candidate, domain, *, clean_inputs=(), max_operations=10000,
                             max_assignments=65536, max_variables=128, max_terms=65536,
                             max_product_terms=65536, measurement_mode='coupled', internal_bits=(),
                             max_outcomes=1024, max_trajectories=1000000, max_table_cells=2097152):
    """Independently check the full physical output, ordered registers, public bits and exact sign.

    Compact domains use a symbolic care mask with universal arbitrary inputs.
    Finite rows use the independent exhaustive basis/sign interpreter.
    """
    limits = dict(max_operations=max_operations, max_variables=max_variables, max_terms=max_terms,
                  max_product_terms=max_product_terms)
    _limits(**limits, max_outcomes=max_outcomes, max_trajectories=max_trajectories)
    clean = set(_wires(list(clean_inputs), original.num_qubits))
    bound = load_search_domain(domain, original, clean_inputs=clean, max_assignments=max_assignments,
                               max_operations=max_operations, max_table_cells=max_table_cells)
    private = set(internal_bits)
    if measurement_mode not in ('coupled', 'independent-map') or (private and measurement_mode != 'independent-map'):
        raise StrataError('private bits require independent-map mode')
    if any(type(bit) is not int or not 0 <= bit < max(original.num_bits,candidate.num_bits) for bit in private):
        raise StrataError('private bit outside compared native boundaries')
    private_names = {f'b{bit}' for bit in private}
    evidence = dict(scope='finite_promised_boundary_assignments', domain_type=bound['type'],
                    domain_hash=bound['domain_hash'], assignment_count=bound['assignment_count'],
                    clean_inputs=sorted(clean), internal_bits=sorted(private), measurement_mode=measurement_mode,
                    original_hash=_canonical_hash(original), candidate_hash=_canonical_hash(candidate),
                    all_input_equivalence=False, host_promise_proven=False,
                    outputs='all physical quantum outputs, public classical outputs, exact sign phase')
    if original.num_qubits != candidate.num_qubits or any(bit not in private for bit in
            range(min(original.num_bits, candidate.num_bits), max(original.num_bits,candidate.num_bits))):
        return dict(evidence, status='unknown', equivalent=None, reason='physical_boundary_layout_mismatch')
    if any(candidate.register_data.get(name) != members for name, members in original.register_data.items()):
        return dict(evidence, status='different', equivalent=False, reason='public_named_register_binding_changed')
    if bound['type'] == 'address_promise':
        evidence.update(address_count=bound['address_count'], arbitrary_wires=bound['arbitrary_wires'],
                        classical_input_scope='all boundary classical inputs arbitrary',
                        evidence='complete_symbolic_basis_sign_difference_under_address_predicate')
        return _compact_check(original, candidate, bound, clean, measurement_mode, private_names, limits, evidence)
    if measurement_mode == 'coupled':
        with native_lines(original) as lines:
            event_count = sum(line.startswith('HMR ') for line in lines)
        if (1 << event_count) * bound['assignment_count'] > max_trajectories:
            return dict(evidence, status='unknown', equivalent=None, reason='measurement_trajectory_budget')
    check = equivalent_on_domain(original, candidate, bound, max_assignments=max_assignments, max_operations=max_operations,
                                 measurement_mode=measurement_mode, internal_bits=private, required_zero_outputs=clean,
                                 max_outcomes=max_outcomes, max_trajectories=max_trajectories)
    return dict(evidence, **{k:v for k,v in check.items() if k not in ('original_hash','candidate_hash')},
                **({} if 'clean_outputs_restored' in check or not check.get('equivalent') else {'clean_outputs_restored': True}))


def promised_domain_synthesis(circuit, domain, **options):
    """Search native lookup and Boolean completions of a complete host promise."""
    from .promised_synthesis import synthesize
    return synthesize(circuit, domain, **options)
