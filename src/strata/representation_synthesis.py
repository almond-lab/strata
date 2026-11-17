"""Executable affine input representations for bounded whole-output synthesis.

Representations are invertible GF(2) bit-basis changes and bit translations.
Every adapted producer/consumer is conjugated back to the original boundary,
including both conversion costs. No integer-offset or projective encoding is
inferred from this Boolean family.
"""
from dataclasses import dataclass
from math import isfinite
import shlex
from time import monotonic

import kickmix as km

from .boolean_synthesis import _layout
from .cost import Formula
from .model import StrataError, digest


def _affine_rows(wires, operations):
    positions = {wire: i for i, wire in enumerate(wires)}
    rows = [1 << i for i in range(len(wires))]
    offsets = [0] * len(wires)
    for line in operations:
        if not isinstance(line, str):
            raise StrataError('affine encoding operations must be native instruction strings')
        words = shlex.split(line)
        if not words or words[0] not in {'X', 'CX'} or len(words) != (2 if words[0] == 'X' else 3):
            raise StrataError('affine encoding permits only unconditional X and CX')
        if any(not word.startswith('q') or not word[1:].isdigit() for word in words[1:]):
            raise StrataError('affine encoding requires physical quantum operands')
        operands = [int(word[1:]) for word in words[1:]]
        if any(q not in positions for q in operands) or len(set(operands)) != len(operands):
            raise StrataError('affine encoding operands must be distinct declared input wires')
        target = positions[operands[-1]]
        if words[0] == 'X':
            offsets[target] ^= 1
        else:
            control = positions[operands[0]]
            rows[target] ^= rows[control]
            offsets[target] ^= offsets[control]
    return tuple(rows), tuple(offsets)


@dataclass(frozen=True)
class AffineEncoding:
    """A typed, executable encoding and independently validated inverse."""
    wires: tuple
    forward: tuple
    inverse: tuple

    def __post_init__(self):
        object.__setattr__(self, 'wires', tuple(self.wires))
        object.__setattr__(self, 'forward', tuple(self.forward))
        object.__setattr__(self, 'inverse', tuple(self.inverse))
        if (len(set(self.wires)) != len(self.wires) or
                any(type(q) is not int or q < 0 for q in self.wires)):
            raise StrataError('invalid affine encoding input wires')
        identity = (tuple(1 << i for i in range(len(self.wires))), (0,) * len(self.wires))
        _affine_rows(self.wires, self.forward)
        _affine_rows(self.wires, self.inverse)
        if (_affine_rows(self.wires, self.forward + self.inverse) != identity or
                _affine_rows(self.wires, self.inverse + self.forward) != identity):
            raise StrataError('affine encoding forward and inverse are not mutual inverses')

    @classmethod
    def from_operations(cls, wires, operations):
        operations = tuple(operations)
        return cls(tuple(wires), operations, tuple(reversed(operations)))

    @property
    def fingerprint(self):
        return digest(dict(wires=self.wires, affine_map=_affine_rows(self.wires, self.forward)))

    def receipt(self):
        rows, offsets = _affine_rows(self.wires, self.forward)
        return dict(type='invertible_GF2_input_encoding', wires=list(self.wires),
                    forward=list(self.forward), inverse=list(self.inverse),
                    matrix_rows=list(rows), bit_offsets=list(offsets), fingerprint=self.fingerprint,
                    inverse_proof='exact affine composition in both orders',
                    scope='Boolean input-bit basis and translations; no integer or projective representation claim')


def _substitute(polynomials, operation):
    words = operation.split()
    target = int(words[-1][1:])
    control = int(words[1][1:]) if words[0] == 'CX' else None
    transformed = []
    for polynomial in polynomials:
        out = set()
        for term in polynomial:
            children = [term]
            if target in term:
                residual = term - {target}
                children.append(residual if control is None else residual | {control})
            for child in children:
                if child in out:
                    out.remove(child)
                else:
                    out.add(child)
        transformed.append(frozenset(out))
    return tuple(transformed)


def _score(polynomials, operations):
    nonlinear = {term for poly in polynomials for term in poly if len(term) > 1}
    return sum(map(len, polynomials)), len(nonlinear), len(operations)


def _encoding_beam(inputs, polynomials, *, max_encodings, beam_width, max_depth, deadline, max_steps):
    """Rank actual cofactor/basis transformations; no fixture-provided encoding."""
    identity = AffineEncoding.from_operations(inputs, ())
    seen = {identity.fingerprint}
    beam = [(identity, tuple(polynomials))]
    selected = [(identity, tuple(polynomials))]
    steps, stopped = 0, None
    for depth in range(max_depth):
        if len(selected) >= max_encodings:
            break
        following = []
        for encoding, current in beam:
            support = sorted({q for poly in current for term in poly for q in term})
            # Both translations and pair mixes are derived only from current
            # output support. Shared cofactors rank mixes before unrelated pairs.
            cofactors = {q: {term - {q} for poly in current for term in poly if q in term} for q in support}
            operations = [f'X q{q}' for q in support]
            pairs = [(a, b) for a in support for b in support if a != b]
            pairs.sort(key=lambda pair: (-len(cofactors[pair[0]] & cofactors[pair[1]]), pair))
            operations += [f'CX q{a} q{b}' for a, b in pairs]
            for operation in operations:
                steps += sum(len(poly) for poly in current) + 1
                if steps > max_steps or monotonic() >= deadline:
                    stopped = 'representation_candidate_step_budget' if steps > max_steps else 'representation_timeout'
                    break
                candidate = AffineEncoding.from_operations(inputs, encoding.forward + (operation,))
                if candidate.fingerprint in seen:
                    continue
                seen.add(candidate.fingerprint)
                # For E=g_k...g_1, f(E^-1*y) substitutes g_1 first, then g_2.
                transformed = _substitute(current, operation)
                following.append((candidate, transformed))
            if stopped:
                break
        following.sort(key=lambda item: (_score(item[1], item[0].forward), item[0].forward))
        beam = following[:beam_width]
        available = max_encodings - len(selected)
        # Reserve candidates for deeper affine compositions instead of filling
        # the complete cap with every single-gate encoding at the first depth.
        allocation = max(1, (available + max_depth - depth - 1) // (max_depth - depth))
        selected.extend(following[:min(available, allocation)])
        if stopped or not beam or len(selected) >= max_encodings:
            break
    return selected, dict(candidate_steps=steps, considered_encodings=len(seen), stopped_reason=stopped)


def representation_boundary_synthesis(circuit, *, clean_inputs=(), cost='Q*toffoli', resource_measure=None,
                                       max_encodings=16, beam_width=4, max_encoding_depth=2,
                                       max_representation_steps=1000000, max_orders=2,
                                       max_factor_nodes=4096, max_factor_steps=1000000, timeout=5.0,
                                       max_operations=10000, max_variables=128, max_terms=65536,
                                       max_product_terms=1048576):
    """Search, emit, restore and price concrete affine representations.

    The original verified source is always the incumbent. Every synthesized
    alternative is checked against the frozen public/dirty/clean boundary, and
    representation conversion is included in the complete native objective.
    """
    from .accounting import measured_resources
    from .algebraic_synthesis import algebraic_boundary_synthesis
    from .inference import functions_at
    from .semantic import equivalent_windows
    budgets = dict(max_encodings=max_encodings, beam_width=beam_width, max_encoding_depth=max_encoding_depth,
                   max_representation_steps=max_representation_steps, max_orders=max_orders,
                   max_factor_nodes=max_factor_nodes, max_factor_steps=max_factor_steps,
                   max_operations=max_operations, max_variables=max_variables,
                   max_terms=max_terms, max_product_terms=max_product_terms)
    if any(type(value) is not int or value <= 0 for value in budgets.values()):
        raise StrataError('representation synthesis budgets must be positive integers')
    if type(timeout) not in (int, float) or not isfinite(timeout) or timeout <= 0:
        raise StrataError('representation synthesis timeout must be finite and positive')
    started, clean = monotonic(), set(_layout(clean_inputs, circuit.num_qubits, 'clean inputs'))
    deadline = started + timeout
    limits = dict(max_operations=max_operations, max_variables=max_variables,
                  max_terms=max_terms, max_product_terms=max_product_terms)
    source_lines = str(circuit).splitlines()
    if len(source_lines) > max_operations:
        return dict(status='unknown', reason='representation_source_operation_budget')
    for line in source_lines:
        words = shlex.split(line, comments=True)
        if words and words[0] not in {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG'} and (
                'if' in words or words[0] in {'HMR', 'PUSH_CONDITION', 'POP_CONDITION'} or words[0].startswith('BIT_')):
            return dict(status='not_applicable', reason='affine representation synthesis requires a quantum-only boundary')
    state = functions_at(circuit, clean_inputs=clean, **limits)
    if state['status'] != 'exact':
        return dict(status='unknown', reason=state.get('reason'), budget_phase='representation_boundary_semantics')
    if state['phase']:
        return dict(status='not_applicable', reason='affine representation synthesis requires an exactly phase-free boundary')
    deltas = {q: frozenset(poly ^ (set() if q in clean else {frozenset({q})}))
              for q, poly in state['functions'].items()
              if poly != (set() if q in clean else {frozenset({q})})}
    targets = sorted(deltas)
    inputs = sorted({q for poly in deltas.values() for term in poly for q in term})
    if any(type(q) is not int for q in inputs) or set(inputs) & set(targets):
        return dict(status='not_applicable', reason='affine representation requires preserved inputs disjoint from XOR consumer targets')
    if any(state['functions'][q] != {frozenset({q})} for q in inputs):
        return dict(status='not_applicable', reason='representation input is not preserved at the complete boundary')
    measure = resource_measure or measured_resources
    objective = Formula(cost) if isinstance(cost, str) else cost
    before = measure(circuit)
    before_cost = objective(before)
    identity = AffineEncoding.from_operations(inputs, ())
    candidates, generation = _encoding_beam(inputs, [deltas[q] for q in targets], max_encodings=max_encodings,
        beam_width=beam_width, max_depth=max_encoding_depth, deadline=deadline, max_steps=max_representation_steps)
    best = dict(candidate=str(circuit), cost=before_cost, resources=before, encoding=identity.receipt(),
                selected_mechanism='source_incumbent', structural_witness={},
                proof=dict(status='equivalent', method='source_identity', clean_inputs=sorted(clean)))
    alternatives, failures = [], []
    stopped = generation['stopped_reason']
    for index, (encoding, predicted) in enumerate(candidates):
        remaining = deadline - monotonic()
        if remaining <= 0:
            stopped = 'representation_timeout'
            break
        adapted = km.Circuit('\n'.join([*encoding.inverse, *source_lines, *encoding.forward]))
        if len(str(adapted).splitlines()) > max_operations:
            failures.append(dict(encoding=encoding.receipt(), reason='representation_adapted_operation_budget'))
            continue
        # Share time among remaining encodings, so identity factoring cannot
        # silently consume every opportunity to try another representation.
        allocation = max(0.001, remaining / max(1, len(candidates) - index))
        def complete_measure(body):
            return measure(km.Circuit('\n'.join([*encoding.forward, str(body), *encoding.inverse])))
        built = algebraic_boundary_synthesis(adapted, clean_inputs=clean,
            cost=cost if isinstance(cost, str) else cost.expression, resource_measure=complete_measure,
            max_orders=max_orders, max_factor_nodes=max_factor_nodes, max_factor_steps=max_factor_steps,
            timeout=allocation, **limits)
        if built['status'] != 'constructed':
            failures.append(dict(encoding=encoding.receipt(), reason=built.get('reason'), status=built['status']))
            continue
        complete = km.Circuit('\n'.join([*encoding.forward, built['candidate'], *encoding.inverse]))
        candidate_lines = str(complete).splitlines()
        if len(candidate_lines) > max_operations or complete.num_qubits > circuit.num_qubits:
            failures.append(dict(encoding=encoding.receipt(), reason='representation_complete_layout_or_operation_budget'))
            continue
        check = equivalent_windows(source_lines, candidate_lines, clean_inputs=clean, **limits)
        if check.get('equivalent') is not True:
            failures.append(dict(encoding=encoding.receipt(), reason='representation_complete_boundary_check', check=check))
            continue
        resources = measure(complete)
        value = objective(resources)
        native_encoding = measured_resources(km.Circuit('\n'.join(encoding.forward)))
        native_inverse = measured_resources(km.Circuit('\n'.join(encoding.inverse)))
        alternative = dict(candidate=str(complete), cost=value, resources=resources, encoding=encoding.receipt(),
            selected_mechanism='affine_encoding' if encoding.forward else 'algebraic_identity',
            adapted_synthesis=built, full_boundary_check=check,
            structural_witness=built.get('structural_witness', {}),
            conversion_bill=dict(encoder=native_encoding, decoder=native_inverse,
                                 adapted_body=measured_resources(km.Circuit(built['candidate'])),
                                 complete_native=measured_resources(complete),
                                 scope='complete outer conversion, adapted consumer, and restoration; widths are not additive'),
            proof=dict(status='equivalent', method='exact_affine_inverse_and_complete_native_boundary_replay',
                       clean_inputs=sorted(clean), public_input_indices=inputs, dirty_target_indices=targets,
                       exact_phase=True, all_public_quantum_inputs=True),
            score_hint=list(_score(predicted, encoding.forward)))
        alternatives.append(alternative)
        if value < best['cost']:
            best = alternative
    return dict(status='constructed', family='affine_representation_co_design', **best,
                improved=best['cost'] < before_cost, alternatives=alternatives, failures=failures,
                resource_bill=dict(before=before, after=best['resources'], before_cost=before_cost,
                                   after_cost=best['cost'], cost_expression=cost if isinstance(cost, str) else cost.expression),
                frozen_boundary=dict(public_inputs=sorted(set(range(circuit.num_qubits)) - clean),
                                     clean_inputs=sorted(clean), dirty_xor_targets=targets,
                                     encoding_wires=inputs, extra_helper_wires=0),
                search=dict(generation, attempted=len(alternatives) + len(failures), verified=len(alternatives),
                            stopped_reason=stopped, limits=budgets | {'timeout': timeout}, seconds=monotonic() - started,
                            optimality='not claimed; bounded affine encodings and algebraic orders'),
                scope='invertible Boolean input-bit representations with exact conversion and restoration costs; no integer or projective encoding claim',
                witness_hash=digest(dict(source=str(circuit), candidate=best['candidate'], clean=sorted(clean),
                                         encoding=best['encoding']['fingerprint'])))
