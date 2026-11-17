from itertools import product

import kickmix as km
import numpy as np
import pytest

from strata.accounting import measured_resources
from strata.domains import execute_assignment
from strata.model import StrataError
from strata.output_basis_search import OutputBasis, output_basis_synthesis


POLYNOMIALS = [
    [(1,), (3,), (1, 3), (0, 1, 3), (1, 2, 3)],
    [(0, 1), (1, 2), (0, 1, 2)],
    [(0,), (1,), (2,), (3,), (0, 2), (0, 3), (0, 1, 3)],
]


def _source():
    lines = ['APPEND_TO_REGISTER q9 r0', 'REGISTER r0 "declared_capacity"',
             'APPEND_TO_REGISTER q4 r1', 'APPEND_TO_REGISTER q5 r1',
             'APPEND_TO_REGISTER q6 r1', 'REGISTER r1 "dirty_outputs"']
    for target, polynomial in enumerate(POLYNOMIALS, 4):
        for term in polynomial:
            if len(term) == 1:
                lines.append(f'CX q{term[0]} q{target}')
            elif len(term) == 2:
                lines.append(f'CCX q{term[0]} q{term[1]} q{target}')
            else:
                lines.extend([f'CCX q{term[0]} q{term[1]} q7',
                              f'CCX q7 q{term[2]} q{target}',
                              f'CCX q{term[0]} q{term[1]} q7'])
    return km.Circuit('\n'.join(lines))


def test_output_basis_inverse_is_valid_for_arbitrary_dirty_targets():
    basis = OutputBasis.from_operations([4, 5, 6], ['CX q4 q5', 'CX q5 q6'])
    assert basis.inverse == ('CX q5 q6', 'CX q4 q5')
    assert basis.receipt()['type'] == 'invertible_GF2_output_basis'
    for row in product(range(2), repeat=3):
        assignment = {f'q{q}': (row[q - 4] if q >= 4 else 0) for q in range(7)}
        result = execute_assignment(basis.forward + basis.inverse, assignment, 7)
        assert result['outputs'] == {str(q): value for q, value in enumerate([0, 0, 0, 0, *row])}
    with pytest.raises(StrataError, match='mutual inverses'):
        OutputBasis(basis.wires, basis.forward, basis.inverse[:-1])
    for operation in ('X q4', 'CX q4 q5 if b0', 'CX q4 q4', 'CX q4 q7'):
        with pytest.raises(StrataError):
            OutputBasis.from_operations([4, 5, 6], [operation])


def test_output_basis_gain_survives_full_unencoded_order_family_and_complete_dirty_boundary():
    circuit = _source()
    result = output_basis_synthesis(circuit, clean_inputs=[7, 8, 9], timeout=10)
    assert result['selected_mechanism'] == 'output_basis'
    assert result['structural_witness']['paid_output_basis_change']
    baseline = next(row for row in result['alternatives'] if not row['encoding']['forward'])
    assert result['cost'] < baseline['cost']
    assert result['unencoded_comparator']['status'] == 'complete'
    assert baseline['adapted_synthesis']['search']['stopped_reason'] is None
    assert len(baseline['adapted_synthesis']['search']['attempts']) >= 5
    assert not any(row['status'] == 'not_lowerable' for row in baseline['adapted_synthesis']['search']['attempts'])
    assert baseline['conversion_bill']['complete_native']['toffoli'] == 10
    assert result['conversion_bill']['complete_native']['toffoli'] == 8
    assert not result['structural_witness']['input_basis_changed']
    assert result['encoding']['dirty_target_wires'] == [4, 5, 6]
    assert result['conversion_bill']['encoder']['instructions'] > 0
    assert result['conversion_bill']['decoder']['instructions'] > 0
    assert result['conversion_bill']['complete_native'] == measured_resources(km.Circuit(result['candidate']))
    # Frozen named boundaries include a high, unused clean wire: retiring its
    # materialization does not silently remove its public register membership.
    for line in str(circuit).splitlines():
        if line.startswith(('REGISTER', 'APPEND_TO_REGISTER')):
            assert line in result['candidate'].splitlines()
    assert km.Circuit(result['candidate']).num_qubits == circuit.num_qubits == 10
    for row in product(range(2), repeat=7):
        assignment = {f'q{q}': bit for q, bit in enumerate([*row, 0, 0, 0])}
        original = execute_assignment(str(circuit).splitlines(), assignment, 10)
        candidate = execute_assignment(result['candidate'].splitlines(), assignment, 10)
        assert candidate['outputs'] == original['outputs']
        assert candidate['phase'] == original['phase']
    native_results = []
    for native in (circuit, km.Circuit(result['candidate'])):
        sim = km.Simulator(batch_size=128, seed=395)
        initial = np.zeros((10, 128), dtype=np.bool_)
        for q in range(7):
            initial[q] = [(row >> q) & 1 for row in range(128)]
        wires = [km.q(q) for q in range(10)]
        sim.write_across_shots(wires, initial)
        sim.do(native)
        native_results.append((sim.read_across_shots(wires), sim.read_phase_flipped_across_shots()))
    assert all(np.array_equal(a, b) for a, b in zip(*native_results))


def test_identical_products_share_through_output_basis_but_conversions_can_lose():
    circuit = km.Circuit('CCX q0 q1 q2\nCCX q0 q1 q3')
    result = output_basis_synthesis(circuit)
    assert result['structural_witness']['paid_output_basis_change']
    assert result['conversion_bill']['complete_native']['toffoli'] == 1
    assert result['conversion_bill']['complete_native']['instructions'] == 3
    assert result['unencoded_comparator']['cost'] == 8
    expensive = output_basis_synthesis(circuit, cost='instructions')
    assert expensive['selected_mechanism'] == 'source_incumbent'
    assert expensive['cost'] == 2
    assert not expensive['structural_witness']['paid_output_basis_change']


def test_complete_conversions_are_priced_in_host_and_can_lose():
    seen = []
    def host(candidate):
        resources = measured_resources(candidate)
        seen.append(str(candidate))
        return dict(resources, host_cost=resources['instructions'] + 1000 * str(candidate).count('CX q4 q5'))
    result = output_basis_synthesis(_source(), clean_inputs=[7, 8, 9], cost='host_cost',
                                   resource_measure=host, timeout=10)
    assert seen
    for alternative in result['alternatives']:
        assert alternative['cost'] == host(km.Circuit(alternative['candidate']))['host_cost']
        if alternative['encoding']['forward']:
            assert alternative['conversion_bill']['encoder']['instructions']
    if result['structural_witness']['paid_output_basis_change']:
        assert result['cost'] < result['unencoded_comparator']['cost']


def test_ties_incumbent_and_partial_baseline_never_claim_structural_gain():
    result = output_basis_synthesis(km.Circuit('CCX q0 q1 q2'), max_orders=1)
    assert result['selected_mechanism'] == 'source_incumbent'
    assert not result['structural_witness']['paid_output_basis_change']
    result = output_basis_synthesis(_source(), clean_inputs=[7, 8, 9], max_factor_steps=1)
    assert result['unencoded_comparator']['status'] == 'unknown'
    assert not result['structural_witness']['paid_output_basis_change']


def test_clean_output_is_never_encoded_with_dirty_targets():
    result = output_basis_synthesis(_source(), clean_inputs=[4, 7, 8, 9], timeout=10)
    assert result['frozen_boundary']['dirty_xor_targets'] == [5, 6]
    assert all(row['encoding']['dirty_target_wires'] == [5, 6] for row in result['alternatives'])
    assert all('q4' not in line for row in result['alternatives'] for line in row['encoding']['forward'])


def test_mutated_synthesizer_is_rejected_at_complete_boundary(monkeypatch):
    import strata.algebraic_synthesis as algebraic
    monkeypatch.setattr(algebraic, 'algebraic_boundary_synthesis',
                        lambda *args, **kwargs: dict(status='constructed', candidate='X q0'))
    result = output_basis_synthesis(_source(), clean_inputs=[7, 8, 9])
    assert result['selected_mechanism'] == 'source_incumbent'
    assert not result['alternatives']
    assert all(row['reason'] == 'output_basis_complete_boundary_check' for row in result['failures'])


def test_incomplete_unencoded_search_cannot_certify_paid_basis_gain(monkeypatch):
    import strata.algebraic_synthesis as algebraic
    original, calls = algebraic.algebraic_boundary_synthesis, []
    def partial_baseline(*args, **kwargs):
        result = original(*args, **kwargs)
        if not calls:
            result['search']['stopped_reason'] = 'algebraic_factor_step_budget'
        calls.append(result)
        return result
    monkeypatch.setattr(algebraic, 'algebraic_boundary_synthesis', partial_baseline)
    result = output_basis_synthesis(km.Circuit('CCX q0 q1 q2\nCCX q0 q1 q3'))
    assert result['selected_mechanism'] == 'output_basis'
    assert result['unencoded_comparator']['status'] == 'unknown'
    assert not result['structural_witness']['paid_output_basis_change']


def test_budgets_are_explicit_and_quantum_non_xor_boundaries_are_rejected():
    result = output_basis_synthesis(_source(), clean_inputs=[7, 8, 9], max_representation_steps=1)
    assert result['search']['stopped_reason'] == 'output_basis_candidate_step_budget'
    assert not result['structural_witness']['paid_output_basis_change']
    for source in ('Z q0', 'HMR q0 b0', 'CX q0 q1\nCCX q0 q1 q2', 'R q0'):
        assert output_basis_synthesis(km.Circuit(source))['status'] == 'not_applicable'
    for limits in ({'max_orders': 0}, {'beam_width': True}, {'timeout': float('nan')}):
        with pytest.raises(StrataError, match='budget|timeout'):
            output_basis_synthesis(_source(), clean_inputs=[7, 8, 9], **limits)
