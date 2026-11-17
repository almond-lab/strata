"""Finite, explicit boundary promises and an independent concrete checker.

The JSON schema is deliberately small: ``version: 1``, ``type: "finite"``,
``registers: {name: [physical wires]}``, and ``assignments: [row, ...]``.
Rows contain bits keyed by ``qN`` (or decimal N), or unsigned integers keyed
by register name, least significant bit first. Every physical boundary wire
must be supplied, including dirty workspace. ``union`` may replace assignments
with a list of such specifications. A promise is never an all-input proof.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shlex
from functools import lru_cache
from itertools import islice

from .model import StrataError, digest


@lru_cache(maxsize=4096)
def _assignment_words(line):
    """Bounded parse cache shared by exhaustive branch executions."""
    return tuple(shlex.split(line, comments=True))


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise StrataError(f'{name} must be a positive integer')


class WindowLimit(StrataError):
    pass


def query_text_coordinates(circuit, from_at=0, at=None, *, unit='operation'):
    """Translate native-operation cuts into canonical text cuts with O(1) memory.

    INFO_TAG and control instructions count as operations, matching discover;
    register serialization metadata does not. SDK query internals keep their
    historical text-line convention and all returned hashes use those cuts.
    """
    from .streaming import native_lines
    if unit not in {'operation','text-line'}:
        raise StrataError('range_unit must be operation or text-line')
    if (type(from_at)is not int or from_at<0 or
            (at is not None and (type(at)is not int or at<from_at))):
        raise StrataError('query window must satisfy 0 <= from <= at')
    if unit=='text-line':return from_at,at
    start=None;operation=0;text_count=0
    with native_lines(circuit) as source:
        for index,raw in enumerate(source):
            text_count=index+1
            words=raw.split()
            if words and words[0] in {'REGISTER','APPEND_TO_REGISTER'}:continue
            if operation==from_at:
                start=index
                if at is None:return start,None
            if operation==at:return start,index
            operation+=1
    if operation==from_at:start=text_count
    if start is None or at is not None and at>operation:
        raise StrataError('query range exceeds supplied circuit operation count')
    return start,None if at is None else text_count


def window_lines(circuit, from_at=0, at=None, *, max_lines=10000):
    """Materialize only a bounded selected native text range, never a full str."""
    from .streaming import native_lines
    if (type(from_at) is not int or from_at < 0 or
            (at is not None and (type(at) is not int or at < from_at))):
        raise StrataError('query window must satisfy 0 <= from <= at <= instruction count')
    _positive(max_lines, 'max_lines')
    if at is not None and at - from_at > max_lines:
        raise WindowLimit('operation_budget')
    selected, count, depth = [], 0, 0
    with native_lines(circuit) as source:
        for index, raw in enumerate(source):
            if at is not None and index == at:
                return selected, at
            count = index + 1
            line = raw.rstrip('\r\n')
            op = line.split(' ', 1)[0]
            if index < from_at:
                if op == 'PUSH_CONDITION':
                    depth += 1
                elif op == 'POP_CONDITION':
                    depth = max(0, depth - 1)
                continue
            if index == from_at and depth:
                raise WindowLimit('window_starts_inside_condition_scope')
            if len(selected) >= max_lines:
                raise WindowLimit('operation_budget')
            selected.append(line)
    if count < from_at or at is not None and count < at:
        raise StrataError('query window must satisfy 0 <= from <= at <= instruction count')
    return selected, count if at is None else at


def touched_wires(lines):
    result = set()
    for line in lines:
        words = shlex.split(line, comments=True)
        if not words or words[0] in {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG', 'DEBUG_PRINT'}:
            continue
        result.update(int(w[1:]) for w in words if re.fullmatch(r'q[0-9]+', w))
    return sorted(result)


def _classical_boundary(lines):
    # Read-before-definite-write; a conditional write cannot initialize a bit.
    defined, required = set(), set()
    depth = 0
    for line in lines:
        w = shlex.split(line, comments=True)
        if not w:
            continue
        op = w[0]
        conditional = len(w) >= 3 and w[-2] == 'if'
        if conditional:
            b = w[-1]
            if b not in defined:
                required.add(b)
            w = w[:-2]
        if op == 'PUSH_CONDITION':
            depth += 1
            continue
        if op == 'POP_CONDITION':
            depth = max(0, depth - 1)
            continue
        if op == 'BIT_INVERT' and len(w) == 2 and w[1] not in defined:
            required.add(w[1])
        if op in {'BIT_STORE0', 'BIT_STORE1', 'BIT_INVERT', 'HMR'} and len(w) >= 2:
            b = w[-1]
            if conditional or depth:
                if b not in defined:
                    required.add(b)
            else:
                defined.add(b)
    return sorted(required)


def load_domain(spec, circuit, *, from_at=0, at=None, max_assignments=65536, max_operations=10000,
                value_only=False, clean_inputs=(), max_variables=128, max_terms=65536,
                max_product_terms=1048576):
    """Validate and canonically hash a complete finite boundary assignment set.

    Normalized domains are accepted but revalidated, including their binding.
    Completeness covers ``range(circuit.num_qubits)`` and every classical input
    read by the selected window. Value queries may omit classical inputs only
    after exact ANF proves that no quantum output depends on them. Equivalence
    callers retain the strict default, including every phase and classical bit.
    """
    _positive(max_assignments, 'max_assignments')
    lines, at = window_lines(circuit, from_at, at, max_lines=max_operations)
    if isinstance(spec, (str, Path)):
        try:
            spec = json.loads(Path(spec).read_text())
        except (OSError, ValueError) as exc:
            raise StrataError(f'cannot read finite domain: {exc}') from exc
    width = circuit.num_qubits
    needed = _classical_boundary(lines)
    clean = tuple(clean_inputs)
    if any(type(q) is not int or not 0 <= q < width for q in clean):
        raise StrataError('invalid clean inputs')
    optional, omitted = set(), set()
    if value_only and needed:
        from .inference import _value_semantics
        semantics = _value_semantics(lines, clean_inputs=clean, max_operations=max_operations,
                                   max_variables=max_variables, max_terms=max_terms,
                                   max_product_terms=max_product_terms)
        if semantics['status'] == 'exact':
            dependencies = {variable for poly in semantics['outputs'].values()
                            for term in poly['monomials'] for variable in term}
            optional = set(needed) - dependencies
    # Implicit finite-domain register aliases denote quantum wires only.
    # A native classical bit bN must never silently supply the value of qN.
    import kickmix as km
    circuit_registers = {
        name: [q.id for q in wires]
        for name, wires in getattr(circuit, 'register_data', {}).items()
        if len(wires) and all(isinstance(q, km.q) for q in wires)
    }
    registers, assignments = {}, set()
    total = 0

    def visit(node, inherited):
        nonlocal total
        if not isinstance(node, dict):
            raise StrataError('finite domain must be a JSON object')
        allowed = {'version', 'type', 'registers', 'assignments', 'union', 'num_qubits',
                   'classical_inputs', 'from_at', 'at', 'domain_hash', 'scope', 'assignment_count',
                   'omitted_classical_inputs', 'value_dependency_proof'}
        if set(node) - allowed:
            raise StrataError('unknown finite domain fields: ' + ', '.join(sorted(set(node) - allowed)))
        if type(node.get('version', 1)) is not int or node.get('version', 1) != 1 or node.get('type', 'finite') != 'finite':
            raise StrataError('expected finite domain version 1')
        if 'num_qubits' in node and node['num_qubits'] != width:
            raise StrataError('finite domain boundary width does not match circuit')
        if 'from_at' in node and node['from_at'] != from_at:
            raise StrataError('finite domain starting boundary does not match query')
        if 'at' in node and node['at'] != at:
            raise StrataError('finite domain ending boundary does not match query')
        mapping = dict(inherited)
        declared = node.get('registers', {})
        if not isinstance(declared, dict):
            raise StrataError('domain registers must map names to physical wire lists')
        for name, qs in declared.items():
            if (not isinstance(name, str) or re.fullmatch(r'(?:q|b)?[0-9]+', name)
                    or not isinstance(qs, list) or not qs
                    or any(type(q) is not int or not 0 <= q < width for q in qs)
                    or len(set(qs)) != len(qs)):
                raise StrataError('invalid domain register mapping')
            if name in registers and registers[name] != qs:
                raise StrataError('union domain register mappings disagree')
            mapping[name] = qs
            registers[name] = qs
        declared_bits = node.get('classical_inputs', needed)
        if (not isinstance(declared_bits, list) or
                any(not isinstance(b,str) or not re.fullmatch(r'b[0-9]+',b) for b in declared_bits) or
                not set(declared_bits) <= set(needed) or set(needed) - set(declared_bits) - optional):
            raise StrataError('classical_inputs must list every classical boundary input as bN')
        if ('union' in node) == ('assignments' in node):
            raise StrataError('finite domain needs exactly one of assignments or union')
        if 'union' in node:
            if not isinstance(node['union'], list) or not node['union']:
                raise StrataError('domain union must be a nonempty list')
            for child in node['union']:
                visit(child, mapping)
            return
        rows = node['assignments']
        if not isinstance(rows, list) or not rows:
            raise StrataError('finite assignments must be a nonempty list')
        for row in rows:
            total += 1
            if total > max_assignments:
                raise StrataError('finite domain assignment budget exceeded')
            if isinstance(row, list):
                if len(row) != width or set(needed) - optional:
                    raise StrataError('bit-vector rows require complete quantum-only assignments')
                row = {f'q{q}': value for q, value in enumerate(row)}
            if not isinstance(row, dict):
                raise StrataError('domain assignment must be an object or complete bit vector')
            values = {}
            def put(key, value):
                if type(value) is not int or value not in (0, 1):
                    raise StrataError('domain bit values must be integers 0 or 1')
                if key in values and values[key] != value:
                    raise StrataError('conflicting aliases in finite assignment')
                values[key] = value
            for key, value in row.items():
                key = str(key)
                if re.fullmatch(r'q?[0-9]+', key):
                    q = int(key.removeprefix('q'))
                    if not 0 <= q < width:
                        raise StrataError('domain quantum wire outside boundary')
                    put(f'q{q}', value)
                elif key in needed:
                    put(key, value)
                elif key in mapping:
                    qs = mapping[key]
                    if type(value) is not int or not 0 <= value < (1 << len(qs)):
                        raise StrataError('register assignment exceeds declared width')
                    for offset, q in enumerate(qs):
                        put(f'q{q}', (value >> offset) & 1)
                else:
                    raise StrataError('unknown assignment register or boundary variable: ' + key)
            expected = {f'q{q}' for q in range(width)} | set(needed)
            if set(values) - expected or expected - set(values) - optional:
                missing = sorted(expected - set(values))
                raise StrataError('incomplete boundary assignment; missing ' + ', '.join(missing))
            if any(values[f'q{q}'] for q in clean):
                raise StrataError('finite domain contradicts declared clean boundary inputs')
            omitted.update(expected - set(values))
            values.update({bit: 0 for bit in expected - set(values)})
            assignments.add(tuple(values[f'q{q}'] for q in range(width)) + tuple(values[b] for b in needed))

    visit(spec, circuit_registers)
    # These variables are proved irrelevant to every quantum value. Project
    # them out consistently even if some rows happened to supply them.
    retained_bits = [b for b in needed if b not in omitted]
    projected = {row[:width] + tuple(row[width + needed.index(b)] for b in retained_bits)
                 for row in assignments}
    normalized = dict(version=1, type='finite', num_qubits=width,
                      registers={k: registers[k] for k in sorted(registers)},
                      classical_inputs=needed, from_at=from_at, at=at,
                      assignments=[{**{f'q{q}': row[q] for q in range(width)},
                                    **{b: row[width+i] for i, b in enumerate(retained_bits)}}
                                   for row in sorted(projected)])
    if omitted:
        normalized.update(omitted_classical_inputs=sorted(omitted),
                          value_dependency_proof=dict(method='exact_ANF_quantum_output_independence',
                              window_hash=digest(lines), clean_inputs=sorted(clean),
                              variables=sorted(omitted), phase_and_classical_outputs='not projected by equivalence checks'))
    domain_hash = digest(normalized)
    if 'domain_hash' in spec and spec['domain_hash'] != domain_hash:
        raise StrataError('finite domain hash does not match validated contents')
    return dict(normalized, domain_hash=domain_hash, assignment_count=len(projected),
                scope=('finite_promised_quantum_values_with_arbitrary_omitted_classical_inputs'
                       if omitted else 'finite_promised_boundary_assignments'))


class _Unknown(Exception):
    pass


@lru_cache(maxsize=16)
def _compile_assignment(lines, num_qubits):
    """Parse bounded concrete programs once, independently of the lab compiler."""
    result = []
    arity = {'X': 1, 'CX': 2, 'CCX': 3, 'SWAP': 2, 'CSWAP': 3,
             'Z': 1, 'CZ': 2, 'CCZ': 3, 'NEG': 0, 'R': 1, 'HMR': 1}
    for line in lines:
        words = list(_assignment_words(line))
        if not words:
            continue
        op = words.pop(0)
        if op in {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG', 'DEBUG_PRINT'}:
            continue
        condition = None
        if len(words) >= 2 and words[-2] == 'if':
            condition = words[-1]
            words = words[:-2]
        target_bit = None
        qs = ()
        if op == 'PUSH_CONDITION':
            if condition is None or words:
                raise _Unknown('malformed_condition')
        elif op == 'POP_CONDITION':
            if words or condition is not None:
                raise _Unknown('malformed_condition')
        elif op in {'BIT_STORE0', 'BIT_STORE1', 'BIT_INVERT'}:
            if len(words) != 1 or not re.fullmatch(r'b[0-9]+', words[0]):
                raise _Unknown('unsupported_operands')
            target_bit = words[0]
        else:
            if op not in arity:
                raise _Unknown('unsupported_operation: ' + op)
            if op == 'HMR':
                if not words or not re.fullmatch(r'b[0-9]+', words[-1]):
                    raise _Unknown('unsupported_operands')
                target_bit = words.pop()
            if len(words) != arity[op] or any(not re.fullmatch(r'q[0-9]+', q) for q in words):
                raise _Unknown('unsupported_operands')
            qs = tuple(int(q[1:]) for q in words)
            if len(set(qs)) != len(qs) or any(q >= num_qubits for q in qs):
                raise _Unknown('unsupported_operands')
        result.append((op, condition, qs, target_bit))
    return tuple(result)


def execute_assignment(lines, assignment, num_qubits, *, outcomes=(), max_operations=10000):
    """Independent exact basis/sign interpreter with explicit formal HMR outcomes.

    Every physical input is provided; reset of a nonzero active wire is unknown.
    Unsupported operations never become identities, including inactive gates.
    """
    _positive(max_operations, 'max_operations')
    lines = tuple(islice(lines, max_operations + 1))
    if len(lines) > max_operations:
        raise _Unknown('operation_budget')
    program = _compile_assignment(lines, num_qubits)
    state = [assignment[f'q{q}'] for q in range(num_qubits)]
    bits = {k: v for k, v in assignment.items() if k.startswith('b')}
    phase, stack, cursor, executed = 0, [], 0, 0
    measurement_events = []
    for op, condition, qs, target_bit in program:
        cond = None
        if condition is not None:
            if condition not in bits:
                raise _Unknown('unbound_classical_input')
            cond = bits[condition]
        if op == 'PUSH_CONDITION':
            stack.append(cond)
            continue
        if op == 'POP_CONDITION':
            if stack:
                stack.pop()
            continue
        active = all(stack) and cond != 0
        if op in {'BIT_STORE0', 'BIT_STORE1', 'BIT_INVERT'}:
            if active:
                if op == 'BIT_INVERT' and target_bit not in bits:
                    raise _Unknown('unbound_classical_input')
                bits[target_bit] = 0 if op == 'BIT_STORE0' else 1 if op == 'BIT_STORE1' else bits[target_bit] ^ 1
            executed += int(active)
            continue
        outcome = None
        if op == 'HMR':
            if cursor >= len(outcomes):
                raise _Unknown('measurement_outcome_missing')
            outcome = outcomes[cursor]
            cursor += 1
            measurement_events.append(dict(wire=qs[0], bit=target_bit, active=int(active),
                                           measured_value=state[qs[0]]))
        if not active:
            continue
        executed += 1
        if op == 'X':
            state[qs[0]] ^= 1
        elif op in {'CX', 'CCX'}:
            state[qs[-1]] ^= int(all(state[q] for q in qs[:-1]))
        elif op == 'SWAP' or op == 'CSWAP' and state[qs[0]]:
            a, b = qs[-2:]
            state[a], state[b] = state[b], state[a]
        elif op in {'Z', 'CZ', 'CCZ', 'NEG'}:
            phase ^= int(all(state[q] for q in qs))
        elif op == 'R':
            if state[qs[0]]:
                raise _Unknown('reset_requires_zero')
        elif op == 'HMR':
            phase ^= state[qs[0]] & outcome
            state[qs[0]] = 0
            bits[target_bit] = outcome
    if stack:
        raise _Unknown('unclosed_condition_scope')
    return dict(outputs={str(q): value for q, value in enumerate(state)}, phase=phase,
                classical_outputs={b: bits[b] for b in sorted(bits)}, executed_operations=executed,
                measurement_events=measurement_events)


def _independent_maps(before, after, bound, width, evidence, *, internal_bits,
                      max_outcomes, max_operations, max_trajectories,
                      required_zero_outputs):
    """Prove each construction's exact boundary map is outcome independent.

    Zero/unit assignments suffice only with the affine control theorem. Public
    classical values are checked too; only explicitly private bits are hidden.
    This is an exact basis/sign boundary proof, not a general channel checker.
    """
    from .lab_program import _affine
    tapes = [before, after]
    ops = [[(i, shlex.split(line, comments=True)) for i, line in enumerate(tape)
            if shlex.split(line, comments=True)] for tape in tapes]
    counts = [sum(words[0] == 'HMR' for _, words in rows) for rows in ops]
    hidden = {f'b{bit}' for bit in internal_bits}
    evidence = dict(evidence, measurement_mode='independent-map', internal_classical_outputs=sorted(hidden),
                    outputs='all physical quantum outputs, nonprivate classical outputs, exact sign phase',
                    outcome_scope='independent exact map proof for each construction',
                    limits=dict(max_outcomes=max_outcomes, max_trajectories=max_trajectories,
                                max_operations=max_operations),
                    represented_outcomes=dict(baseline=1 << counts[0], candidate=1 << counts[1]))
    checked, reductions = 0, set()

    def observable(result):
        return dict(outputs=result['outputs'], phase=result['phase'],
                    classical_outputs={b: v for b, v in result['classical_outputs'].items() if b not in hidden})

    try:
        for assignment in bound['assignments']:
            maps = []
            for side, (tape, rows, count) in enumerate(zip(tapes, ops, counts)):
                reduced = count >= 2 and _affine(rows, initial_bits=assignment)
                number = count + 1 if reduced else 1 << count
                if number > max_outcomes:
                    return dict(evidence, status='unknown', equivalent=None, reason='measurement_outcome_budget', checked_cases=checked)
                if checked + number > max_trajectories:
                    return dict(evidence, status='unknown', equivalent=None, reason='measurement_trajectory_budget', checked_cases=checked)
                reductions.add('affine_hmr' if reduced else 'full_enumeration')
                branches = [0, *(1 << m for m in range(count))] if reduced else range(number)
                reference = None
                for encoded in branches:
                    outcomes = [(encoded >> i) & 1 for i in range(count)]
                    result = execute_assignment(tape, assignment, width, outcomes=outcomes, max_operations=max_operations)
                    checked += 1
                    value = observable(result)
                    for q in required_zero_outputs:
                        if result['outputs'][str(q)]:
                            return dict(evidence, status='different', equivalent=False, reason='clean_output_not_restored',
                                        checked_cases=checked, clean_outputs_restored=False,
                                        counterexample=dict(kind='clean_restoration', artifact=('baseline', 'candidate')[side],
                                                            inputs=assignment, outcomes=outcomes, wire=q, output=1))
                    if reference is None:
                        reference = value
                    elif value != reference:
                        return dict(evidence, status='unknown', equivalent=None,
                                    reason='measurement_dependent_boundary_map', checked_cases=checked,
                                    independence_witness=dict(artifact=('baseline', 'candidate')[side], inputs=assignment,
                                                              outcomes=outcomes, zero_outcome_map=reference, branch_map=value))
                maps.append(reference)
            if maps[0] != maps[1]:
                return dict(evidence, status='different', equivalent=False, reason='boundary_function_mismatch',
                            checked_cases=checked, counterexample=dict(inputs=assignment, before=maps[0], after=maps[1]))
    except (_Unknown, ValueError) as exc:
        return dict(evidence, status='unknown', equivalent=None, reason=str(exc), checked_cases=checked)
    return dict(evidence, status='equivalent', equivalent=True, checked_cases=checked,
                reductions=sorted(reductions), clean_outputs_restored=True if required_zero_outputs else None,
                measurement_independent=True, distributional_equivalence=False)


def equivalent_on_domain(original, candidate, domain, *, from_at=0, at=None,
                         max_assignments=65536, max_outcomes=1024, max_operations=10000,
                         measurement_mode='coupled', internal_bits=(), max_trajectories=1000000,
                         required_zero_outputs=()):
    """Check every complete promised assignment with explicit HMR semantics.

    Equal physical width, full outputs, exact sign, and all classical outputs
    are required. HMR events must align by ordered concrete provenance on every
    checked row/outcome; ordinal counts alone do not establish this alignment.
    The default coupled mode requires aligned instruments. Independent-map
    mode proves outcome independence on each side before comparing boundary
    maps; only explicitly listed internal classical outputs may be omitted.
    """
    _positive(max_outcomes, 'max_outcomes')
    _positive(max_trajectories, 'max_trajectories')
    if measurement_mode not in {'coupled', 'independent-map'}:
        raise StrataError('measurement_mode must be coupled or independent-map')
    internal_bits = tuple(internal_bits)
    if any(type(bit) is not int or bit < 0 for bit in internal_bits):
        raise StrataError('internal_bits must contain nonnegative classical bit IDs')
    if internal_bits and measurement_mode != 'independent-map':
        raise StrataError('internal_bits requires independent-map measurement mode')
    required_zero_outputs = tuple(required_zero_outputs)
    if any(type(q) is not int or not 0 <= q < original.num_qubits for q in required_zero_outputs):
        raise StrataError('required_zero_outputs must contain physical quantum wire IDs')
    try:
        before, stop = window_lines(original, from_at, at, max_lines=max_operations)
        after, _ = window_lines(candidate, from_at, None if at is None else at, max_lines=max_operations)
    except WindowLimit as exc:
        return dict(status='unknown', equivalent=None, reason=str(exc),
                    scope='finite_promised_boundary_assignments', all_input_equivalence=False)
    bound = load_domain(domain, original, from_at=from_at, at=stop,
                        max_assignments=max_assignments, max_operations=max_operations)
    from .streaming import native_lines
    def artifact_hash(circuit):
        hasher = hashlib.sha256()
        with native_lines(circuit) as source:
            for line in source:
                hasher.update(line.encode())
        return hasher.hexdigest()
    evidence = dict(scope='finite_promised_boundary_assignments', domain_hash=bound['domain_hash'],
                    measurement_mode=measurement_mode,
                    assignment_count=bound['assignment_count'], all_input_equivalence=False,
                    original_hash=artifact_hash(original), candidate_hash=artifact_hash(candidate),
                    from_at=from_at, at=stop, outputs='all physical quantum and classical outputs; exact sign phase',
                    semantic_scope='finite native outcome-parametric basis trajectories; not a general quantum-channel proof',
                    evidence='independent_exhaustive_basis_sign_interpreter')
    if candidate.num_qubits != original.num_qubits:
        return dict(evidence, status='unknown', equivalent=None, reason='physical_boundary_layout_mismatch',
                    layout_scope='finite-domain checker requires equal physical width; no omitted public outputs or clean-layout mapping is inferred')
    if measurement_mode == 'independent-map':
        return _independent_maps(before, after, bound, original.num_qubits, evidence,
                                 internal_bits=internal_bits, max_outcomes=max_outcomes,
                                 max_operations=max_operations, max_trajectories=max_trajectories,
                                 required_zero_outputs=required_zero_outputs)
    count = lambda lines: sum(bool(w) and w[0] == 'HMR' for w in (shlex.split(s, comments=True) for s in lines))
    measurements = count(before)
    if count(after) != measurements:
        return dict(evidence, status='unknown', equivalent=None, reason='measurement_outcome_spaces_differ')
    if (1 << measurements) > max_outcomes:
        return dict(evidence, status='unknown', equivalent=None, reason='measurement_outcome_budget')
    checked = 0
    try:
        for assignment in bound['assignments']:
            for encoded in range(1 << measurements):
                outcomes = [(encoded >> i) & 1 for i in range(measurements)]
                a = execute_assignment(before, assignment, original.num_qubits, outcomes=outcomes, max_operations=max_operations)
                b = execute_assignment(after, assignment, original.num_qubits, outcomes=outcomes, max_operations=max_operations)
                checked += 1
                for name, result in (('baseline', a), ('candidate', b)):
                    for q in required_zero_outputs:
                        if result['outputs'][str(q)]:
                            return dict(evidence, status='different', equivalent=False, reason='clean_output_not_restored',
                                        checked_cases=checked, clean_outputs_restored=False,
                                        counterexample=dict(kind='clean_restoration', artifact=name, inputs=assignment,
                                                            outcomes=outcomes, wire=q, output=1))
                if a['measurement_events'] != b['measurement_events']:
                    return dict(evidence, status='unknown', equivalent=None, checked_cases=checked,
                                reason='measurement_event_alignment_unproved',
                                outcome_scope='current branch-preserving model requires ordered wire/destination/activity/measured-value provenance',
                                limitation='A host verifier may establish a different measurement-instrument equivalence; this checker does not.',
                                alignment_witness=dict(inputs=assignment, outcomes=outcomes,
                                                       before=a['measurement_events'], after=b['measurement_events']))
                if any(a[key] != b[key] for key in ('outputs', 'phase', 'classical_outputs')):
                    return dict(evidence, status='different', equivalent=False, checked_cases=checked,
                                outcome_scope='all coupled formal HMR outcomes',
                                counterexample=dict(inputs=assignment, outcomes=outcomes, before=a, after=b))
    except (_Unknown, ValueError) as exc:
        return dict(evidence, status='unknown', equivalent=None, reason=str(exc), checked_cases=checked)
    return dict(evidence, status='equivalent', equivalent=True, checked_cases=checked,
                outcome_scope='all coupled formal HMR outcomes', outcome_count=1 << measurements,
                distributional_equivalence=False)
