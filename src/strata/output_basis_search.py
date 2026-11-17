"""Paid output-basis search for complete dirty XOR accumulator boundaries.

For an invertible linear map E on targets, E; update(E F); E^-1 implements
update(F), including arbitrary initial target values. This changes the search
space seen by the bounded polynomial factorer, not multiplicative complexity
under unrestricted exact synthesis. Every conversion is emitted and priced.
"""
from dataclasses import dataclass
from math import isfinite
import shlex
from time import monotonic

import kickmix as km

from .boolean_synthesis import _layout
from .cost import Formula
from .model import StrataError, digest
from .representation_synthesis import AffineEncoding, _affine_rows


@dataclass(frozen=True)
class OutputBasis:
    """Executable linear encoding of dirty targets, with a checked inverse."""
    wires: tuple
    forward: tuple
    inverse: tuple

    def __post_init__(self):
        for field in ('wires', 'forward', 'inverse'):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        for line in self.forward + self.inverse:
            words = shlex.split(line) if isinstance(line, str) else []
            if not words or words[0] != 'CX':
                raise StrataError('output basis permits only unconditional CX row operations')
        AffineEncoding(self.wires, self.forward, self.inverse)

    @classmethod
    def from_operations(cls, wires, operations):
        operations = tuple(operations)
        return cls(tuple(wires), operations, tuple(reversed(operations)))

    @property
    def fingerprint(self):
        return digest(dict(wires=self.wires, matrix_rows=_affine_rows(self.wires, self.forward)[0]))

    def receipt(self):
        return dict(type='invertible_GF2_output_basis', dirty_target_wires=list(self.wires),
                    forward=list(self.forward), inverse=list(self.inverse),
                    matrix_rows=list(_affine_rows(self.wires, self.forward)[0]),
                    inverse_proof='exact linear composition in both orders', fingerprint=self.fingerprint)


def _split(lines):
    metadata, body = [], []
    for line in lines:
        words = shlex.split(line, comments=True)
        if words:
            (metadata if words[0] in {'REGISTER', 'APPEND_TO_REGISTER'} else body).append(line)
    return metadata, body


def _poly_rows(polynomials):
    return {str(q): [sorted(term) for term in sorted(poly, key=lambda term: (len(term), tuple(sorted(term))))]
            for q, poly in sorted(polynomials.items())}


def _candidate_bases(wires, deltas, *, max_encodings, beam_width, max_depth, max_steps,
                     max_terms, deadline):
    identity = OutputBasis.from_operations(wires, ())
    selected, beam = [(identity, deltas)], [(identity, deltas)]
    seen, steps, stopped = {identity.fingerprint}, 0, None
    for depth in range(max_depth):
        if len(selected) >= max_encodings:
            break
        following = []
        for encoding, current in beam:
            steps += sum(len(term) + 1 for q in wires for term in current[q])
            if steps > max_steps or monotonic() >= deadline:
                stopped = 'output_basis_candidate_step_budget' if steps > max_steps else 'output_basis_timeout'
                break
            cofactors = {q: {term - {v} for term in current[q] for v in term if len(term) > 1}
                         for q in wires}
            pairs = []
            for a in wires:
                for b in wires:
                    if a == b:
                        continue
                    steps += 1 + min(len(current[a]), len(current[b])) + min(len(cofactors[a]), len(cofactors[b]))
                    if steps > max_steps or monotonic() >= deadline:
                        stopped = 'output_basis_candidate_step_budget' if steps > max_steps else 'output_basis_timeout'
                        break
                    monomials = len(current[a] & current[b])
                    cofactors_shared = len(cofactors[a] & cofactors[b])
                    if monomials or cofactors_shared:
                        pairs.append((a, b, monomials, cofactors_shared))
                if stopped:
                    break
            if stopped:
                break
            pairs.sort(key=lambda pair: (-pair[2], -pair[3], pair[:2]))
            for a, b, monomials, cofactors_shared in pairs:
                steps += len(current[a]) + len(current[b]) + 1
                if steps > max_steps or monotonic() >= deadline:
                    stopped = 'output_basis_candidate_step_budget' if steps > max_steps else 'output_basis_timeout'
                    break
                basis = OutputBasis.from_operations(wires, encoding.forward + (f'CX q{a} q{b}',))
                if basis.fingerprint in seen:
                    continue
                seen.add(basis.fingerprint)
                transformed = dict(current)
                transformed[b] = current[b] ^ current[a]
                if len(transformed[b]) > max_terms:
                    continue
                overlap = monomials + cofactors_shared
                following.append((basis, transformed, overlap))
            if stopped:
                break
        # Large monomial cancellation is useful but is not a pruning theorem:
        # cofactor overlap can improve factorability while increasing terms.
        following.sort(key=lambda row: (-row[2], sum(map(len, row[1].values())), row[0].forward))
        beam = [(basis, polys) for basis, polys, _ in following[:beam_width]]
        available = max_encodings - len(selected)
        allocation = max(1, (available + max_depth - depth - 1) // (max_depth - depth))
        selected.extend((basis, polys) for basis, polys, _ in following[:min(available, allocation)])
        if stopped or not beam or len(selected) >= max_encodings:
            break
    return selected, dict(candidate_steps=steps, considered_bases=len(seen), stopped_reason=stopped)


def output_basis_synthesis(circuit, *, clean_inputs=(), cost='Q*toffoli', resource_measure=None,
                           max_encodings=16, beam_width=4, max_encoding_depth=2,
                           max_representation_steps=1000000, max_orders=8,
                           max_factor_nodes=4096, max_factor_steps=1000000, timeout=5.0,
                           max_operations=10000, max_variables=128, max_terms=65536,
                           max_product_terms=1048576):
    """Search whole-output row bases, retaining an equally budgeted baseline.

    Only original dirty XOR targets participate in E. Clean outputs retain
    their original zero promise and are synthesized jointly but never encoded.
    Unknown/incomplete unencoded comparisons cannot claim a paid basis gain.
    """
    from .accounting import measured_resources
    from .algebraic_synthesis import algebraic_boundary_synthesis
    from .inference import functions_at
    from .semantic import equivalent_windows

    budgets = dict(max_encodings=max_encodings, beam_width=beam_width,
                   max_encoding_depth=max_encoding_depth, max_representation_steps=max_representation_steps,
                   max_orders=max_orders, max_factor_nodes=max_factor_nodes, max_factor_steps=max_factor_steps,
                   max_operations=max_operations, max_variables=max_variables,
                   max_terms=max_terms, max_product_terms=max_product_terms)
    if any(type(value) is not int or value <= 0 for value in budgets.values()):
        raise StrataError('output basis budgets must be positive integers')
    if type(timeout) not in (int, float) or not isfinite(timeout) or timeout <= 0:
        raise StrataError('output basis timeout must be finite and positive')
    started = monotonic()
    deadline = started + timeout
    clean = set(_layout(clean_inputs, circuit.num_qubits, 'clean inputs'))
    source = str(circuit).splitlines()
    metadata, body = _split(source)
    if len(body) > max_operations:
        return dict(status='unknown', reason='output_basis_source_operation_budget')
    for line in body:
        words = shlex.split(line, comments=True)
        if 'if' in words or words[0] not in {'X', 'CX', 'CCX', 'Z', 'CZ', 'CCZ', 'SWAP', 'INFO_TAG'}:
            return dict(status='not_applicable', reason='output basis requires a quantum permutation boundary')
    limits = dict(max_operations=max_operations, max_variables=max_variables,
                  max_terms=max_terms, max_product_terms=max_product_terms)
    state = functions_at(circuit, clean_inputs=clean, **dict(limits, max_operations=max_operations + len(metadata)))
    if state['status'] != 'exact':
        return dict(status='unknown', reason=state.get('reason'), budget_phase='output_basis_boundary_semantics')
    if state['phase']:
        return dict(status='not_applicable', reason='output basis requires exact zero boundary phase')
    deltas = {q: frozenset(poly ^ (set() if q in clean else {frozenset({q})}))
              for q, poly in state['functions'].items()
              if poly != (set() if q in clean else {frozenset({q})})}
    targets = sorted(deltas)
    support = {q for poly in deltas.values() for term in poly for q in term}
    if any(type(q) is not int for q in support) or support & set(targets):
        return dict(status='not_applicable', reason='output XOR deltas depend on modified targets or nonquantum inputs')
    if any(state['functions'][q] != {frozenset({q})} for q in support):
        return dict(status='not_applicable', reason='output basis inputs are not preserved')
    dirty = sorted(set(targets) - clean)
    measure = resource_measure or measured_resources
    objective = Formula(cost) if isinstance(cost, str) else cost
    expression = cost if isinstance(cost, str) else cost.expression
    before = measure(circuit)
    before_cost = objective(before)
    identity = OutputBasis.from_operations(dirty, ())
    candidates, generation = _candidate_bases(dirty, deltas, max_encodings=max_encodings,
        beam_width=beam_width, max_depth=max_encoding_depth, max_steps=max_representation_steps,
        max_terms=max_terms, deadline=deadline)
    best = dict(candidate=str(circuit), cost=before_cost, resources=before,
                encoding=identity.receipt(), selected_mechanism='source_incumbent',
                structural_witness=dict(paid_output_basis_change=False),
                proof=dict(status='equivalent', method='source_identity', clean_inputs=sorted(clean)))
    public = set(range(circuit.num_qubits)) - clean

    def complete_circuit(encoding, synthesized):
        # Restore the source metadata verbatim: a retired helper may still be
        # part of a named interface. Never price a silently shortened register.
        _, native = _split(str(synthesized).splitlines())
        result = km.Circuit('\n'.join(metadata + list(encoding.forward) + native + list(encoding.inverse)))
        if public and result.num_qubits <= max(public):
            result = km.Circuit(str(result) + f'\nAPPEND_TO_REGISTER q{max(public)} r{circuit.num_registers}')
        return result

    alternatives, failures, baseline = [], [], None
    stopped = generation['stopped_reason']
    # Equal per-basis time ceilings, in addition to equal algebraic order/node/
    # step budgets. A partial baseline is reported and cannot qualify the flag.
    allocation = max(0.001, (deadline - monotonic()) / max(1, len(candidates)))
    analysis_cap = max_operations + len(metadata) + 2 * max_encoding_depth
    for encoding, predicted in candidates:
        if monotonic() >= deadline:
            stopped = 'output_basis_timeout'
            break
        adapted = km.Circuit('\n'.join(metadata + list(encoding.inverse) + body + list(encoding.forward)))
        built = algebraic_boundary_synthesis(adapted, clean_inputs=clean, cost=expression,
            resource_measure=lambda candidate: measure(complete_circuit(encoding, candidate)),
            max_orders=max_orders, max_factor_nodes=max_factor_nodes, max_factor_steps=max_factor_steps,
            timeout=min(allocation, max(0.001, deadline - monotonic())),
            **dict(limits, max_operations=analysis_cap))
        if built['status'] != 'constructed':
            failures.append(dict(encoding=encoding.receipt(), status=built['status'], reason=built.get('reason')))
            continue
        complete = complete_circuit(encoding, km.Circuit(built['candidate']))
        _, candidate_body = _split(str(complete).splitlines())
        if len(candidate_body) > max_operations or complete.num_qubits > circuit.num_qubits:
            failures.append(dict(encoding=encoding.receipt(), reason='output_basis_complete_layout_or_operation_budget'))
            continue
        check = equivalent_windows(body, candidate_body, clean_inputs=clean, **limits)
        if check.get('equivalent') is not True:
            failures.append(dict(encoding=encoding.receipt(), reason='output_basis_complete_boundary_check', check=check))
            continue
        resources = measure(complete)
        value = objective(resources)
        alternative = dict(candidate=str(complete), cost=value, resources=resources,
            encoding=encoding.receipt(), selected_mechanism='output_basis' if encoding.forward else 'algebraic_identity',
            adapted_synthesis=built, full_boundary_check=check,
            structural_witness=dict(paid_output_basis_change=False,
                output_basis_changed=bool(encoding.forward), encoded_output_targets=dirty,
                original_output_deltas=_poly_rows(deltas), encoded_output_deltas=_poly_rows(predicted),
                synthesized_joint_outputs=targets, materialized_products=built.get('lowering', {}).get('materialized_products'),
                basis_conversions_emitted=True, input_basis_changed=False),
            conversion_bill=dict(encoder=measured_resources(km.Circuit('\n'.join(encoding.forward))),
                decoder=measured_resources(km.Circuit('\n'.join(encoding.inverse))),
                adapted_body=measured_resources(km.Circuit(built['candidate'])),
                complete_native=measured_resources(complete),
                scope='full encoder, adapted updates and decoder; qubit widths are not additive'),
            proof=dict(status='equivalent', method='exact_output_basis_inverse_and_complete_native_boundary_replay',
                all_public_quantum_inputs=True, exact_phase=True, dirty_targets='arbitrary',
                clean_inputs=sorted(clean), source_named_registers_preserved=True, truth_rows_enumerated=0))
        alternatives.append(alternative)
        if not encoding.forward:
            baseline = alternative
        if value < best['cost']:
            best = alternative
    baseline_search = baseline['adapted_synthesis'].get('search') if baseline else None
    baseline_complete = (bool(baseline_search and baseline_search.get('attempts'))
        and baseline_search.get('stopped_reason') is None
        and all(row.get('status') in {'verified', 'not_lowerable'} for row in baseline_search['attempts']))
    comparator_cost = min(before_cost, baseline['cost']) if baseline else before_cost
    for candidate in alternatives:
        witness = candidate['structural_witness']
        witness.update(paid_output_basis_change=bool(candidate['encoding']['forward']) and baseline_complete
                       and candidate['cost'] < comparator_cost,
                       unencoded_comparator_cost=comparator_cost, unencoded_comparison_complete=baseline_complete,
                       same_factor_order_limit=max_orders,
                       baseline_materialized_products=(baseline['structural_witness']['materialized_products'] if baseline else None))
    return dict(status='constructed', family='joint_output_basis_co_design', **best,
        improved=best['cost'] < before_cost, alternatives=alternatives, failures=failures,
        unencoded_comparator=dict(status='complete' if baseline_complete else 'unknown', cost=comparator_cost,
            source_cost=before_cost, algebraic_cost=baseline['cost'] if baseline else None,
            max_orders=max_orders, max_factor_steps=max_factor_steps, max_factor_nodes=max_factor_nodes,
            per_basis_timeout=allocation, search=baseline['adapted_synthesis'].get('search') if baseline else None),
        resource_bill=dict(before=before, after=best['resources'], before_cost=before_cost,
                           after_cost=best['cost'], cost_expression=expression),
        frozen_boundary=dict(public_inputs=sorted(public), clean_inputs=sorted(clean),
            dirty_xor_targets=dirty, unchanged_input_basis=sorted(support), extra_helper_wires=0),
        search=dict(generation, attempted=len(alternatives) + len(failures), verified=len(alternatives),
            stopped_reason=stopped, limits=budgets | {'timeout': timeout}, seconds=monotonic() - started,
            analysis_operation_limit=analysis_cap,
            optimality='not claimed; output bases change bounded factoring, not unrestricted multiplicative complexity'),
        witness_hash=digest(dict(source=str(circuit), candidate=best['candidate'], clean=sorted(clean),
                                 output_basis=best['encoding']['fingerprint'])))
