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

from .model import StrataError, digest


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise StrataError(f'{name} must be a positive integer')


class WindowLimit(StrataError):
    pass


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


def load_domain(spec, circuit, *, from_at=0, at=None, max_assignments=65536, max_operations=10000):
    """Validate and canonically hash a complete finite boundary assignment set.

    Normalized domains are accepted but revalidated, including their binding.
    Completeness covers ``range(circuit.num_qubits)`` and every classical input
    read by the selected window. Metadata names do not establish clean inputs.
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
    circuit_registers = {
        name: [q.id for q in wires if hasattr(q, 'id')]
        for name, wires in getattr(circuit, 'register_data', {}).items()
    }
    registers, assignments = {}, set()
    total = 0

    def visit(node, inherited):
        nonlocal total
        if not isinstance(node, dict):
            raise StrataError('finite domain must be a JSON object')
        allowed = {'version', 'type', 'registers', 'assignments', 'union', 'num_qubits',
                   'classical_inputs', 'from_at', 'at', 'domain_hash', 'scope', 'assignment_count'}
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
                set(declared_bits) != set(needed)):
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
                if len(row) != width or needed:
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
            if set(values) != expected:
                missing = sorted(expected - set(values))
                raise StrataError('incomplete boundary assignment; missing ' + ', '.join(missing))
            assignments.add(tuple(values[f'q{q}'] for q in range(width)) + tuple(values[b] for b in needed))

    visit(spec, circuit_registers)
    normalized = dict(version=1, type='finite', num_qubits=width,
                      registers={k: registers[k] for k in sorted(registers)},
                      classical_inputs=needed, from_at=from_at, at=at,
                      assignments=[{**{f'q{q}': row[q] for q in range(width)},
                                    **{b: row[width+i] for i, b in enumerate(needed)}}
                                   for row in sorted(assignments)])
    domain_hash = digest(normalized)
    if 'domain_hash' in spec and spec['domain_hash'] != domain_hash:
        raise StrataError('finite domain hash does not match validated contents')
    return dict(normalized, domain_hash=domain_hash, assignment_count=len(assignments),
                scope='finite_promised_boundary_assignments')


class _Unknown(Exception):
    pass


def execute_assignment(lines, assignment, num_qubits, *, outcomes=(), max_operations=10000):
    """Independent exact basis/sign interpreter; randomness is supplied explicitly.

    HMR outcomes are independent formal branch bits, with phase old_q*outcome.
    A reset of a nonzero active wire is rejected. This is not a general quantum
    simulator and does not turn unsupported operations into identities.
    """
    _positive(max_operations, 'max_operations')
    state = [assignment[f'q{q}'] for q in range(num_qubits)]
    bits = {k: v for k, v in assignment.items() if k.startswith('b')}
    phase, stack, cursor, executed = 0, [], 0, 0
    measurement_events = []
    arity = {'X': 1, 'CX': 2, 'CCX': 3, 'SWAP': 2, 'CSWAP': 3,
             'Z': 1, 'CZ': 2, 'CCZ': 3, 'NEG': 0, 'R': 1, 'HMR': 1}
    for index, line in enumerate(lines):
        if index >= max_operations:
            raise _Unknown('operation_budget')
        w = shlex.split(line, comments=True)
        if not w:
            continue
        op = w.pop(0)
        if op in {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG', 'DEBUG_PRINT'}:
            continue
        cond = None
        if len(w) >= 2 and w[-2] == 'if':
            name = w[-1]
            if name not in bits:
                raise _Unknown('unbound_classical_input')
            cond = bits[name]
            w = w[:-2]
        if op == 'PUSH_CONDITION':
            if cond is None or w:
                raise _Unknown('malformed_condition')
            stack.append(cond)
            continue
        if op == 'POP_CONDITION':
            if w or cond is not None:
                raise _Unknown('malformed_condition')
            if stack:
                stack.pop()
            continue
        active = all(stack) and cond != 0
        if op in {'BIT_STORE0', 'BIT_STORE1', 'BIT_INVERT'}:
            if len(w) != 1 or not re.fullmatch(r'b[0-9]+', w[0]):
                raise _Unknown('unsupported_operands')
            if active:
                if op == 'BIT_INVERT' and w[0] not in bits:
                    raise _Unknown('unbound_classical_input')
                bits[w[0]] = 0 if op == 'BIT_STORE0' else 1 if op == 'BIT_STORE1' else bits[w[0]] ^ 1
            executed += int(active)
            continue
        if op not in arity:
            raise _Unknown('unsupported_operation: ' + op)
        target_bit = None
        if op == 'HMR':
            if not w or not re.fullmatch(r'b[0-9]+', w[-1]):
                raise _Unknown('unsupported_operands')
            target_bit = w.pop()
        if len(w) != arity[op] or any(not re.fullmatch(r'q[0-9]+', q) for q in w):
            raise _Unknown('unsupported_operands')
        qs = [int(q[1:]) for q in w]
        if len(set(qs)) != len(qs) or any(q >= num_qubits for q in qs):
            raise _Unknown('unsupported_operands')
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


def equivalent_on_domain(original, candidate, domain, *, from_at=0, at=None,
                         max_assignments=65536, max_outcomes=1024, max_operations=10000):
    """Check every complete promised assignment and every coupled HMR outcome.

    Equal physical width, full outputs, exact sign, and all classical outputs
    are required. HMR events must align by ordered concrete provenance on every
    checked row/outcome; ordinal counts alone do not establish this alignment.
    Changed layouts or measurement instruments require a separate proof.
    """
    _positive(max_outcomes, 'max_outcomes')
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
                    assignment_count=bound['assignment_count'], all_input_equivalence=False,
                    original_hash=artifact_hash(original), candidate_hash=artifact_hash(candidate),
                    from_at=from_at, at=stop, outputs='all physical quantum and classical outputs; exact sign phase',
                    semantic_scope='finite native outcome-parametric basis trajectories; not a general quantum-channel proof',
                    evidence='independent_exhaustive_basis_sign_interpreter')
    if candidate.num_qubits != original.num_qubits:
        return dict(evidence, status='unknown', equivalent=None, reason='physical_boundary_layout_mismatch',
                    layout_scope='finite-domain checker requires equal physical width; no omitted public outputs or clean-layout mapping is inferred')
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
