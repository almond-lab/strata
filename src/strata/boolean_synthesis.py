"""Whole-boundary XOR/AND synthesis with reversible native witness lowering.

Final AND values are written directly to the consumer outputs. Only products
used by another AND need materialized workspace, and that workspace is restored.
The search minimum is Boolean; the emitted native construction has its own bill.
"""
from itertools import product
import shlex

from .floors import exact_minimum, verify_minimum_witness
from .model import StrataError, digest


def _layout(values, width, name):
    values = list(values)
    if (len(set(values)) != len(values) or
            any(type(q) is not int or not 0 <= q < width for q in values)):
        raise StrataError('invalid synthesis ' + name)
    return values


def _affine_update(mask, available, target):
    lines = [f'X q{target}'] if mask & 1 else []
    for index, wire in enumerate(available, 1):
        if mask >> index & 1:
            if wire is None:
                raise StrataError('synthesis references an unmaterialized product')
            lines.append(f'CX q{wire} q{target}')
    return lines


def _product_update(left, right, available, target):
    """XOR an affine product into target, restoring the operand wire basis."""
    a, b, ca, cb = left >> 1, right >> 1, left & 1, right & 1
    if not a:
        return _affine_update(right, available, target) if ca else []
    if not b:
        return _affine_update(left, available, target) if cb else []
    if a == b:
        return _affine_update(left, available, target) if ca == cb else []
    p = (a & -a).bit_length() - 1
    # First replace x_p by a*x + ca. Re-express b in that basis before
    # preparing its pivot, so intersecting affine forms remain exact.
    first = []
    for index, wire in enumerate(available):
        if index != p and a >> index & 1:
            first.append(f'CX q{wire} q{available[p]}')
    if ca:
        first.append(f'X q{available[p]}')
    if b >> p & 1:
        b ^= a ^ (1 << p)
        cb ^= ca
    outside = b & ~(1 << p)
    if not outside:
        raise StrataError('synthesis affine controls are not independent')
    q = (outside & -outside).bit_length() - 1
    second = []
    for index, wire in enumerate(available):
        if index != q and b >> index & 1:
            second.append(f'CX q{wire} q{available[q]}')
    if cb:
        second.append(f'X q{available[q]}')
    if any(available[index] is None for index in range(len(available)) if (a | b) >> index & 1):
        raise StrataError('synthesis references an unmaterialized product')
    prepare = first + second
    return prepare + [f'CCX q{available[p]} q{available[q]} q{target}'] + list(reversed(prepare))


def lower_xor_and_witness(receipt, input_rows, output_rows, *, input_wires,
                          target_wires, clean_wires, num_qubits):
    """Lower promised-row output deltas into dirty-target XOR updates.

    Inputs, targets, and clean workspace must be pairwise disjoint. A terminal
    product is never stored: its AND updates each wanted target directly. This
    replay proves the construction on supplied rows and arbitrary target values;
    it does not establish that a source circuit has that boundary contract.
    """
    if type(num_qubits) is not int or num_qubits < 0:
        raise StrataError('invalid synthesis boundary width')
    input_rows, output_rows = list(map(list, input_rows)), list(map(list, output_rows))
    verify_minimum_witness(receipt, input_rows, output_rows)
    inputs = _layout(input_wires, num_qubits, 'input wires')
    targets = _layout(target_wires, num_qubits, 'target wires')
    clean = _layout(clean_wires, num_qubits, 'clean wires')
    if set(inputs) & set(targets) or set(inputs + targets) & set(clean):
        raise StrataError('synthesis inputs, targets, and clean workspace must be disjoint')
    if len(inputs) != len(input_rows[0]) or len(targets) != len(output_rows[0]):
        raise StrataError('synthesis wire layout does not match witness row widths')
    gates, offset = receipt['witness'], len(inputs) + 1
    consumed = {index for gate in gates for name in ('left_coefficients', 'right_coefficients')
                for index in range(len(gates)) if gate[name] >> (offset + index) & 1}
    if len(consumed) > len(clean):
        return dict(status='unknown', reason='insufficient_clean_product_workspace',
                    required_clean_wires=len(consumed), available_clean_wires=len(clean))
    placement = dict(zip(sorted(consumed), clean))
    available = inputs + [placement.get(i) for i in range(len(gates))]
    forward, blocks = [], []
    for index, gate in enumerate(gates):
        if index in consumed:
            block = _product_update(gate['left_coefficients'], gate['right_coefficients'],
                                    available[:len(inputs) + index], placement[index])
            forward.extend(block)
            blocks.append(block)
    consumer = []
    for mask, target in zip(receipt['output_coefficients'], targets):
        materialized = mask
        for index, gate in enumerate(gates):
            if index not in consumed and mask >> (offset + index) & 1:
                materialized ^= 1 << (offset + index)
                consumer.extend(_product_update(gate['left_coefficients'], gate['right_coefficients'],
                                                available[:len(inputs) + index], target))
        consumer.extend(_affine_update(materialized, available, target))
    cleanup = [line for block in reversed(blocks) for line in reversed(block)]
    native = forward + consumer + cleanup
    from .domains import execute_assignment
    # Targets are only XOR-written and never read, so one target assignment
    # suffices: translation proves the same deltas for every dirty target value.
    for row, expected in zip(input_rows, output_rows):
        assignment = {f'q{q}': 0 for q in range(num_qubits)}
        assignment.update({f'q{q}': bit for q, bit in zip(inputs, row)})
        replay = execute_assignment(native, assignment, num_qubits, max_operations=max(1, len(native)))
        final = {int(q): bit for q, bit in replay['outputs'].items()}
        expected_full = {q: assignment[f'q{q}'] for q in range(num_qubits)}
        expected_full.update(dict(zip(targets, expected)))
        if final != expected_full or replay['phase']:
            raise StrataError('native XOR/AND witness failed full-boundary replay')
    import kickmix as km
    from .kickmix import Kickmix
    circuit = km.Circuit('\n'.join(native))
    return dict(status='constructed', candidate=str(circuit),
                resource_bill=dict(native=Kickmix().measured(circuit), boolean_and_count=len(gates),
                                   materialized_products=len(consumed),
                                   forward_instructions=len(forward), consumer_instructions=len(consumer),
                                   cleanup_instructions=len(cleanup)),
                layout=dict(input_wires=inputs, target_wires=targets, used_clean_wires=list(placement.values()),
                            unused_clean_wires=[q for q in clean if q not in placement.values()],
                            external_input_indices_unchanged=True),
                proof=dict(status='verified', method='independent_concrete_full_boundary_replay',
                           assignment_count=len(input_rows), dirty_targets='arbitrary; XOR-written and never read',
                           preserved='all input values, zero workspace, exact zero sign',
                           scope='supplied input rows only; source contract must be separately verified'),
                witness_hash=digest(dict(search=receipt['contract_hash'], native=native, inputs=inputs,
                                         targets=targets, clean=clean, width=num_qubits)))


def _source_structure(lines, clean):
    depths, peak, materialized = {}, 0, set()
    for line in lines:
        words = shlex.split(line, comments=True)
        if not words:
            continue
        op = words[0]
        wires = [int(word[1:]) for word in words[1:] if word.startswith('q') and word[1:].isdigit()]
        if op in {'CCX', 'CSWAP'}:
            controls = wires[:-1] if op == 'CCX' else wires
            depth = max((depths.get(q, 0) for q in controls), default=0) + 1
            for q in wires[-(2 if op == 'CSWAP' else 1):]:
                depths[q] = max(depths.get(q, 0), depth)
                if q in clean:
                    materialized.add(q)
            peak = max(peak, depth)
        elif op == 'CX':
            depths[wires[-1]] = max(depths.get(q, 0) for q in wires)
            if wires[-1] in clean:
                materialized.add(wires[-1])
        elif op == 'SWAP':
            depths[wires[0]], depths[wires[1]] = depths.get(wires[1], 0), depths.get(wires[0], 0)
        elif op in {'HMR', 'R'}:
            depths[wires[0]] = 0
    return dict(source_nonlinear_dataflow_depth=peak, materialized_clean_wires=sorted(materialized),
                depth_scope='syntactic dependency depth including restoration; not a circuit-depth minimum')


def joint_boundary_synthesis(circuit, *, clean_inputs=(), internal_bits=(), max_and=3,
                             timeout=1.0, max_states=10000, max_expansions=100000,
                             max_operations=10000, max_variables=128, max_terms=65536,
                             max_product_terms=1048576, max_truth_inputs=8):
    """Resynthesize all final XOR consumers together, retiring restored scratch.

    Only exact phase-free maps with preserved source inputs are in this family.
    Classical measurement outputs can disappear only when explicitly private.
    Every candidate is checked against all physical quantum outputs and phase.
    """
    from .domains import window_lines, WindowLimit
    from .inference import functions_at, _value_semantics
    clean = set(_layout(clean_inputs, circuit.num_qubits, 'clean inputs'))
    internal_bits = tuple(internal_bits)
    if any(type(bit) is not int or bit < 0 for bit in internal_bits):
        raise StrataError('internal_bits must contain nonnegative classical bit IDs')
    if type(max_truth_inputs) is not int or not 0 <= max_truth_inputs <= 8:
        raise StrataError('max_truth_inputs must be between zero and eight')
    limits = dict(max_operations=max_operations, max_variables=max_variables, max_terms=max_terms,
                  max_product_terms=max_product_terms)
    state = functions_at(circuit, clean_inputs=clean, **limits)
    if state['status'] != 'exact':
        return dict(status='unknown', reason=state.get('reason'), budget_phase='boundary_semantics')
    if state['phase']:
        return dict(status='unknown', reason='joint Boolean synthesis requires an exactly phase-free boundary')
    try:
        lines, _ = window_lines(circuit, max_lines=max_operations)
    except WindowLimit as exc:
        return dict(status='unknown', reason=str(exc))
    sem = _value_semantics(lines, clean_inputs=clean, **limits)
    private = {f'b{bit}' for bit in internal_bits}
    changed_classical = {bit for bit, poly in sem.get('classical_outputs', {}).items()
                         if {frozenset(term) for term in poly['monomials']} != {frozenset({bit})}}
    if changed_classical - private:
        return dict(status='unknown', reason='classical outputs require explicit private-bit declarations',
                    required_internal_bits=sorted(int(b[1:]) for b in changed_classical))
    deltas = {}
    for q, poly in state['functions'].items():
        initial = set() if q in clean else {frozenset({q})}
        if poly != initial:
            deltas[q] = poly ^ initial
    targets = sorted(deltas)
    support = {v for poly in deltas.values() for term in poly for v in term}
    if any(type(v) is not int for v in support):
        return dict(status='unknown', reason='visible values depend on formal classical or measurement inputs')
    if support & set(targets):
        return dict(status='unknown', reason='consumer XOR deltas depend on modified target inputs')
    inputs = sorted(support)
    if len(inputs) > max_truth_inputs:
        return dict(status='unknown', reason='joint synthesis truth-table input budget', input_count=len(inputs),
                    max_truth_inputs=max_truth_inputs)
    remap = {q: i for i, q in enumerate(inputs)}
    outputs = [[[remap[q] for q in sorted(term)] for term in sorted(deltas[target], key=lambda t: (len(t), tuple(sorted(t))))]
               for target in targets]
    search = exact_minimum(outputs, len(inputs), max_and=max_and, timeout=timeout,
                           max_states=max_states, max_expansions=max_expansions)
    if search['kind'] != 'exact_small':
        return dict(status='unknown', reason=search['reason'], exact_search=search)
    input_rows = [list(row) for row in product(range(2), repeat=len(inputs))]
    output_rows = [[sum(all(row[q] for q in term) for term in poly) % 2 for poly in outputs] for row in input_rows]
    helpers = sorted(clean - set(targets))
    lowered = lower_xor_and_witness(search, input_rows, output_rows, input_wires=inputs,
                                    target_wires=targets, clean_wires=helpers, num_qubits=circuit.num_qubits)
    if lowered['status'] != 'constructed':
        return dict(lowered, exact_search=search)
    # Preserve every public physical index, while allowing only unused clean
    # trailing workspace to disappear from the physical width.
    import kickmix as km
    from .kickmix import Kickmix
    used_helpers = set(lowered['layout']['used_clean_wires'])
    retained = sorted((set(range(circuit.num_qubits)) - clean) | set(targets) | used_helpers)
    metadata = []
    max_register = -1
    register_names = set()
    for line in lines:
        words = shlex.split(line, comments=True)
        if words and words[0] in {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG'}:
            if words[0] == 'REGISTER' and len(words) > 2:
                register_names.add(words[2])
            max_register = max([max_register] + [int(w[1:]) for w in words if w.startswith('r') and w[1:].isdigit()])
            if words[0] == 'APPEND_TO_REGISTER' and any(w == f'q{q}' for q in clean - set(retained) for w in words):
                continue
            metadata.append(line)
    if retained:
        name, suffix = 'strata_joint_boundary', 0
        while name in register_names:
            suffix += 1
            name = f'strata_joint_boundary_{suffix}'
        metadata.extend([f'APPEND_TO_REGISTER q{max(retained)} r{max_register + 1}',
                         f'REGISTER r{max_register + 1} "{name}"'])
    candidate = km.Circuit('\n'.join(metadata + [lowered['candidate']]))
    replay = functions_at(candidate, clean_inputs=clean & set(range(candidate.num_qubits)), **limits)
    if replay['status'] != 'exact':
        return dict(status='unknown', reason='emitted candidate symbolic replay budget', replay=replay)
    for q, expected in state['functions'].items():
        if replay['functions'].get(q, set()) != expected:
            raise StrataError('joint synthesis full-boundary function replay mismatch')
    if replay['phase'] != state['phase']:
        raise StrataError('joint synthesis exact phase replay mismatch')
    before, after = Kickmix().measured(circuit), Kickmix().measured(candidate)
    structure = _source_structure(lines, clean)
    restored = {q for q in clean if not state['functions'][q]}
    retired = sorted(set(structure['materialized_clean_wires']) & restored - used_helpers)
    structural = dict(structure, retired_scratch_wires=retired,
                      retained_product_wires=sorted(used_helpers),
                      final_products_materialized=False,
                      boundary_outputs_synthesized_jointly=targets,
                      removed_materialization_bank=len(retired) >= 2 and structure['source_nonlinear_dataflow_depth'] >= 2)
    return dict(status='constructed', family='joint_boundary_xor_and', candidate=str(candidate),
                exact_search=search, lowering=lowered,
                synthesis_contract=dict(input_rows=input_rows, output_rows=output_rows,
                                        input_wires=inputs, target_wires=targets,
                                        clean_wires=helpers, num_qubits=circuit.num_qubits),
                proof=dict(status='equivalent', method='exact_full_quantum_boundary_ANF_and_zero_sign_replay',
                           clean_inputs=sorted(clean), internal_classical_outputs=sorted(private),
                           scope='all arbitrary public quantum inputs with declared clean workspace and private measurement bits',
                           all_input_equivalence=not clean and not private,
                           measurement_scope='basis/sign maps; independent-map verifier required for measurement channels'),
                layout=dict(public_inputs=sorted(set(range(circuit.num_qubits)) - clean),
                            omitted_clean_wires=sorted(clean - set(retained)),
                            restored_outputs={str(q): 0 for q in clean - set(retained)},
                            external_input_indices_unchanged=True),
                structural_witness=structural,
                resource_bill=dict(before=before, after=after, native_cost_status='exact_emitted',
                                   width_reduction=before['Q'] - after['Q'],
                                   toffoli_reduction=before['toffoli'] - after['toffoli']),
                witness_hash=digest(dict(source=str(circuit), candidate=str(candidate), clean=sorted(clean),
                                         private=sorted(private), search=search['contract_hash'])))
