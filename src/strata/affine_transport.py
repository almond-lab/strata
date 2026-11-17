"""Factor a recovered global affine transport out of a nonlinear boundary.

For F with invertible affine part T, try U=T^-1 o F and V=F o T^-1.
If either residual has preserved-input XOR form, synthesize it and emit T o U
or V o T, respectively. The transport acts on arbitrary public wires only;
declared clean inputs retain their zero promises throughout adaptation.
"""
from dataclasses import dataclass
from math import isfinite
import shlex
from time import monotonic

import kickmix as km

from .boolean_synthesis import _layout
from .cost import Formula
from .model import StrataError, digest
from .output_basis_search import _split


def _linear_map(wires, operations):
    positions = {q: i for i, q in enumerate(wires)}
    rows, offsets = [1 << i for i in range(len(wires))], [0] * len(wires)
    for line in operations:
        words = shlex.split(line)
        if not words or words[0] not in {'X', 'CX', 'SWAP'} or len(words) != (2 if words[0] == 'X' else 3):
            raise StrataError('affine transport accepts only unconditional X, CX and SWAP')
        if any(not word.startswith('q') or not word[1:].isdigit() for word in words[1:]):
            raise StrataError('invalid affine transport operand')
        operands = [int(word[1:]) for word in words[1:]]
        if len(set(operands)) != len(operands) or any(q not in positions for q in operands):
            raise StrataError('affine transport operands must be distinct declared public wires')
        indices = [positions[q] for q in operands]
        target = indices[-1]
        if words[0] == 'X':
            offsets[target] ^= 1
        elif words[0] == 'CX':
            rows[target] ^= rows[indices[0]]
            offsets[target] ^= offsets[indices[0]]
        else:
            a, b = indices
            rows[a], rows[b] = rows[b], rows[a]
            offsets[a], offsets[b] = offsets[b], offsets[a]
    return tuple(rows), tuple(offsets)


@dataclass(frozen=True)
class AffineTransport:
    wires: tuple
    matrix_rows: tuple
    offsets: tuple
    forward: tuple
    inverse: tuple

    def __post_init__(self):
        for field in ('wires', 'matrix_rows', 'offsets', 'forward', 'inverse'):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        n = len(self.wires)
        if (len(set(self.wires)) != n or any(type(q) is not int or q < 0 for q in self.wires)
                or len(self.matrix_rows) != n or len(self.offsets) != n
                or any(type(row) is not int or row < 0 or row >> n for row in self.matrix_rows)
                or any(type(bit) is not int or bit not in (0, 1) for bit in self.offsets)):
            raise StrataError('invalid affine transport map')
        if _linear_map(self.wires, self.forward) != (self.matrix_rows, self.offsets):
            raise StrataError('affine transport emission does not implement the recovered map')
        identity = (tuple(1 << i for i in range(n)), (0,) * n)
        if (_linear_map(self.wires, self.forward + self.inverse) != identity
                or _linear_map(self.wires, self.inverse + self.forward) != identity):
            raise StrataError('affine transport inverse failed exact composition')

    def receipt(self):
        return dict(type='global_public_affine_transport', wires=list(self.wires),
            matrix_rows=list(self.matrix_rows), offsets=list(self.offsets),
            forward=list(self.forward), inverse=list(self.inverse),
            proof='exact recovered affine map and inverse compositions in both orders',
            fingerprint=digest(dict(wires=self.wires, rows=self.matrix_rows, offsets=self.offsets)))


def recover_affine_transport(functions, public_wires, *, max_operations=10000):
    """Extract all degree-zero/one coefficients and invert their square matrix."""
    wires = tuple(public_wires)
    positions = {q: i for i, q in enumerate(wires)}
    rows, offsets = [], []
    for q in wires:
        row, offset = 0, 0
        for term in functions[q]:
            if not term:
                offset ^= 1
            elif len(term) == 1:
                variable = next(iter(term))
                if variable not in positions:
                    raise StrataError('affine transport contains a non-public linear input')
                row ^= 1 << positions[variable]
        rows.append(row)
        offsets.append(offset)
    work, elimination = list(rows), []
    for column in range(len(wires)):
        pivot = next((row for row in range(column, len(wires)) if work[row] >> column & 1), None)
        if pivot is None:
            raise StrataError('affine_transport_singular_linear_part')
        if pivot != column:
            work[pivot], work[column] = work[column], work[pivot]
            elimination.append(f'SWAP q{wires[pivot]} q{wires[column]}')
        for row in range(len(wires)):
            if row != column and work[row] >> column & 1:
                work[row] ^= work[column]
                elimination.append(f'CX q{wires[column]} q{wires[row]}')
        if len(elimination) > max_operations:
            raise StrataError('affine_transport_adapter_operation_budget')
    forward = list(reversed(elimination)) + [f'X q{q}' for q, bit in zip(wires, offsets) if bit]
    if len(forward) > max_operations:
        raise StrataError('affine_transport_adapter_operation_budget')
    return AffineTransport(wires, tuple(rows), tuple(offsets), tuple(forward), tuple(reversed(forward)))


def affine_transport_synthesis(circuit, *, clean_inputs=(), cost='Q*toffoli', resource_measure=None,
                                max_orders=8, max_factor_nodes=4096, max_factor_steps=1000000,
                                timeout=2.0, max_operations=10000, max_variables=128,
                                max_terms=65536, max_product_terms=1048576):
    """Unlock preserved-input residual synthesis behind a global affine map."""
    from .accounting import measured_resources
    from .algebraic_synthesis import algebraic_boundary_synthesis
    from .inference import functions_at
    from .semantic import equivalent_windows

    controls = dict(max_orders=max_orders, max_factor_nodes=max_factor_nodes, max_factor_steps=max_factor_steps,
        max_operations=max_operations, max_variables=max_variables, max_terms=max_terms,
        max_product_terms=max_product_terms)
    if any(type(value) is not int or value <= 0 for value in controls.values()):
        raise StrataError('affine transport budgets must be positive integers')
    if type(timeout) not in (int, float) or not isfinite(timeout) or timeout <= 0:
        raise StrataError('affine transport timeout must be finite and positive')
    started = monotonic()
    deadline = started + timeout
    clean = set(_layout(clean_inputs, circuit.num_qubits, 'clean inputs'))
    if circuit.num_qubits - len(clean) > max_variables:
        return dict(status='unknown', reason='affine_transport_public_variable_budget')
    public = sorted(set(range(circuit.num_qubits)) - clean)
    metadata, body = _split(str(circuit).splitlines())
    limits = dict(max_operations=max_operations, max_variables=max_variables,
                  max_terms=max_terms, max_product_terms=max_product_terms)
    if len(body) > max_operations:
        return dict(status='unknown', reason='affine_transport_source_operation_budget')
    for line in body:
        words = shlex.split(line, comments=True)
        if 'if' in words or words[0] not in {'X', 'CX', 'CCX', 'SWAP', 'Z', 'CZ', 'CCZ', 'INFO_TAG'}:
            return dict(status='not_applicable', reason='affine transport requires a quantum permutation boundary')
    state = functions_at(circuit, clean_inputs=clean, **dict(limits, max_operations=max_operations + len(metadata)))
    if state['status'] != 'exact':
        return dict(status='unknown', reason=state.get('reason'), budget_phase='affine_transport_boundary_semantics')
    if state['phase']:
        return dict(status='not_applicable', reason='affine transport requires exact zero boundary phase')
    deltas = {q: poly ^ (set() if q in clean else {frozenset({q})})
              for q, poly in state['functions'].items()
              if poly != (set() if q in clean else {frozenset({q})})}
    modified = set(deltas)
    support = {variable for poly in deltas.values() for term in poly for variable in term}
    if not support & modified:
        return dict(status='not_applicable', reason='boundary already has preserved-input XOR form')
    try:
        transport = recover_affine_transport(state['functions'], public, max_operations=max_operations)
    except StrataError as exc:
        return dict(status='unknown' if 'budget' in str(exc) else 'not_applicable', reason=str(exc))
    if not transport.forward:
        return dict(status='not_applicable', reason='affine_transport_is_identity')
    measure = resource_measure or measured_resources
    objective = Formula(cost) if isinstance(cost, str) else cost
    expression = cost if isinstance(cost, str) else cost.expression
    before = measure(circuit)
    before_cost = objective(before)
    best = dict(candidate=str(circuit), resources=before, cost=before_cost,
        proof=dict(status='equivalent', method='source_identity'), selected_mechanism='source_incumbent',
        structural_witness=dict(affine_transport_factored=False))
    failures, alternatives = [], []
    allocation = max(.001, (deadline - monotonic()) / 2)
    def complete(orientation, candidate):
        _, native = _split(str(candidate).splitlines())
        lines = (list(transport.forward) + native if orientation == 'input'
                 else native + list(transport.forward))
        result = km.Circuit('\n'.join(metadata + lines))
        if public and result.num_qubits <= max(public):
            result = km.Circuit(str(result) + f'\nAPPEND_TO_REGISTER q{max(public)} r{circuit.num_registers}')
        return result
    for orientation in ('output', 'input'):
        remaining = deadline - monotonic()
        if remaining <= 0:
            failures.append(dict(orientation=orientation, reason='affine_transport_timeout', status='unknown'))
            break
        residual_body = (list(transport.inverse) + body if orientation == 'input'
                         else body + list(transport.inverse))
        residual = km.Circuit('\n'.join(metadata + residual_body))
        built = algebraic_boundary_synthesis(residual, clean_inputs=clean, cost=expression,
            resource_measure=lambda candidate: measure(complete(orientation, candidate)),
            max_orders=max_orders, max_factor_nodes=max_factor_nodes, max_factor_steps=max_factor_steps,
            timeout=min(allocation, remaining),
            **dict(limits, max_operations=max_operations + len(metadata) + len(transport.inverse)))
        if built['status'] != 'constructed':
            failures.append(dict(orientation=orientation, status=built['status'], reason=built.get('reason')))
            continue
        candidate = complete(orientation, km.Circuit(built['candidate']))
        _, candidate_body = _split(str(candidate).splitlines())
        if len(candidate_body) > max_operations or candidate.num_qubits > circuit.num_qubits:
            failures.append(dict(orientation=orientation, reason='affine_transport_complete_layout_or_operation_budget'))
            continue
        check = equivalent_windows(body, candidate_body, clean_inputs=clean, **limits)
        if check.get('equivalent') is not True:
            failures.append(dict(orientation=orientation, reason='affine_transport_complete_boundary_check', check=check))
            continue
        resources = measure(candidate)
        value = objective(resources)
        alternative = dict(candidate=str(candidate), candidate_hash=digest(str(candidate)), cost=value,
            resources=resources, orientation=orientation, selected_mechanism='affine_transport',
            residual_synthesis=built, whole_boundary_check=check,
            structural_witness=dict(affine_transport_factored=value < before_cost and bool(built.get('dag', {}).get('gates')),
                original_target_dependent_boundary=True, residual_preserved_input_XOR_form=True,
                affine_transport_orientation=orientation, complete_transport_charged=True),
            proof=dict(status='equivalent', method='exact_affine_inverse_and_complete_native_boundary_equality',
                all_public_quantum_inputs=True, exact_phase=True, dirty_inputs='arbitrary',
                clean_inputs=sorted(clean), source_named_registers_preserved=True,
                residual_polynomial_replay=built.get('polynomial_replay'), truth_rows_enumerated=0),
            conversion_bill=dict(emitted_transport=measured_resources(km.Circuit('\n'.join(transport.forward))),
                residual_native=measured_resources(km.Circuit(built['candidate'])),
                complete_native=measured_resources(candidate),
                inverse_scope='used only to define/prove the residual; not an omitted runtime operation'))
        alternatives.append(alternative)
        if value < best['cost']:
            best = alternative
    if not alternatives:
        return dict(status='unknown' if any(row.get('status') == 'unknown' for row in failures) else 'not_applicable',
            reason='affine_transport_has_no_verified_compatible_residual', transport=transport.receipt(), failures=failures)
    return dict(status='constructed', family='global_affine_transport_factorization', **best,
        transport=transport.receipt(), improved=best['cost'] < before_cost, alternatives=alternatives, failures=failures,
        resource_bill=dict(before=before, after=best['resources'], before_cost=before_cost, after_cost=best['cost'],
                           cost_expression=expression),
        frozen_boundary=dict(public_inputs=public, clean_inputs=sorted(clean), extra_helper_wires=0),
        search=dict(orientations=['output', 'input'], verified=len(alternatives),
                    limits=controls | {'timeout': timeout}, seconds=monotonic() - started,
                    optimality='not claimed; recovered affine part and two residual orientations'))
