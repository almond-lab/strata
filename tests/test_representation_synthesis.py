from itertools import product

import kickmix as km
import pytest

from strata.accounting import measured_resources
from strata.domains import execute_assignment
from strata.model import StrataError
from strata.representation_synthesis import AffineEncoding, representation_boundary_synthesis


def _polynomial_circuit(polynomials, inputs=4):
    """A generic monomial emitter; no representation or factorization hint."""
    scratch = inputs + len(polynomials)
    lines = []
    for target, polynomial in enumerate(polynomials, inputs):
        for term in polynomial:
            if len(term) == 1:
                lines.append(f'CX q{term[0]} q{target}')
            elif len(term) == 2:
                lines.append(f'CCX q{term[0]} q{term[1]} q{target}')
            elif len(term) == 3:
                lines.extend([f'CCX q{term[0]} q{term[1]} q{scratch}',
                              f'CCX q{scratch} q{term[2]} q{target}',
                              f'CCX q{term[0]} q{term[1]} q{scratch}'])
            else:
                raise AssertionError('fixture emitter supports only degrees one through three')
    return km.Circuit('\n'.join(lines)), [scratch]


POLYNOMIALS = [
    [(0, 1), (0, 2), (1, 3), (3,)],
    [(0, 1, 3), (0, 2), (0, 2, 3), (1,), (1, 2), (1, 3), (2,), (2, 3), (3,)],
]


def test_typed_encoding_proves_both_inverse_orders_and_rejects_mutation():
    encoding = AffineEncoding.from_operations([2, 5, 8], ['CX q2 q5', 'X q5', 'CX q5 q8'])
    receipt = encoding.receipt()
    assert receipt['type'] == 'invertible_GF2_input_encoding'
    assert receipt['inverse'] == ['CX q5 q8', 'X q5', 'CX q2 q5']
    assert receipt['inverse_proof'] == 'exact affine composition in both orders'
    for row in product(range(2), repeat=3):
        initial = {f'q{q}': 0 for q in range(9)}
        initial.update({f'q{q}': bit for q, bit in zip(encoding.wires, row)})
        replay = execute_assignment(encoding.forward + encoding.inverse, initial, 9)
        assert replay['outputs'] == {str(q): initial[f'q{q}'] for q in range(9)}
    with pytest.raises(StrataError, match='mutual inverses'):
        AffineEncoding(encoding.wires, encoding.forward, encoding.inverse[:-1])
    with pytest.raises(StrataError, match='only unconditional'):
        AffineEncoding.from_operations([0, 1], ['CX q0 q1 if b0'])
    with pytest.raises(StrataError, match='declared input wires'):
        AffineEncoding.from_operations([0, 1], ['CX q0 q2'])


def test_generated_affine_basis_improves_bounded_factoring_and_restores_dirty_consumers():
    circuit, clean = _polynomial_circuit(POLYNOMIALS)
    result = representation_boundary_synthesis(circuit, clean_inputs=clean, max_orders=1,
                                                max_encodings=12, timeout=5)
    identity = next(candidate for candidate in result['alternatives'] if not candidate['encoding']['forward'])
    assert result['improved'] and result['selected_mechanism'] == 'affine_encoding'
    assert result['cost'] < identity['cost']
    assert result['frozen_boundary']['extra_helper_wires'] == 0
    assert result['encoding']['forward'] and result['encoding']['inverse']
    assert result['conversion_bill']['encoder']['instructions'] > 0
    assert result['conversion_bill']['decoder']['instructions'] > 0
    assert result['conversion_bill']['complete_native'] == measured_resources(km.Circuit(result['candidate']))
    assert result['proof']['all_public_quantum_inputs'] and result['proof']['exact_phase']
    for bits in product(range(2), repeat=6):
        assignment = {f'q{q}': bit for q, bit in enumerate([*bits, 0])}
        original = execute_assignment(str(circuit).splitlines(), assignment, circuit.num_qubits)
        candidate = execute_assignment(result['candidate'].splitlines(), assignment, circuit.num_qubits)
        assert original['outputs'] == candidate['outputs'] and original['phase'] == candidate['phase']


def test_representation_gain_survives_complete_default_order_family_and_same_declared_capacity():
    from strata.algebraic_synthesis import algebraic_boundary_synthesis
    original, _ = _polynomial_circuit(POLYNOMIALS)
    # The extra zero wire is explicitly part of both inputs' physical contract.
    # This lets every generated unencoded factor order finish, avoiding a
    # comparison against an early workspace-budget stop.
    circuit = km.Circuit(str(original) + '\nAPPEND_TO_REGISTER q7 r0\nREGISTER r0 "declared_capacity"')
    raw = algebraic_boundary_synthesis(circuit, clean_inputs=[6, 7], max_orders=8, timeout=5)
    encoded = representation_boundary_synthesis(circuit, clean_inputs=[6, 7], max_orders=8,
                                                 max_encodings=12, timeout=5)
    assert raw['search']['stopped_reason'] is None
    assert len(raw['search']['attempts']) >= 7
    assert encoded['selected_mechanism'] == 'affine_encoding' and encoded['cost'] < raw['cost']
    assert encoded['frozen_boundary']['clean_inputs'] == [6, 7]
    assert encoded['frozen_boundary']['extra_helper_wires'] == 0


def test_conversion_is_charged_in_the_actual_host_objective_and_source_remains_incumbent():
    circuit = km.Circuit('CCX q0 q1 q2')
    seen = []
    def host_measure(candidate):
        values = measured_resources(candidate)
        seen.append(str(candidate))
        return dict(values, host_cost=values['toffoli'] + 100 * values['instructions'])
    result = representation_boundary_synthesis(circuit, cost='host_cost', resource_measure=host_measure,
                                                max_encodings=8, max_orders=1)
    assert seen and not result['improved']
    assert result['selected_mechanism'] == 'source_incumbent'
    assert result['candidate'] == str(circuit)
    assert result['cost'] == 101
    assert any(len(candidate['encoding']['forward']) == 2 for candidate in result['alternatives'])
    for candidate in result['alternatives']:
        resources = measured_resources(km.Circuit(candidate['candidate']))
        assert candidate['cost'] == resources['toffoli'] + 100 * resources['instructions']


def test_representation_budget_keeps_verified_baseline_and_reports_partial_search():
    circuit, clean = _polynomial_circuit(POLYNOMIALS)
    result = representation_boundary_synthesis(circuit, clean_inputs=clean, max_representation_steps=1,
                                                max_orders=1, timeout=5)
    assert result['status'] == 'constructed'
    assert result['search']['stopped_reason'] == 'representation_candidate_step_budget'
    assert result['cost'] <= result['resource_bill']['before_cost']
    for limits in ({'max_encoding_depth': 0}, {'beam_width': True}, {'timeout': float('nan')}):
        with pytest.raises(StrataError, match='budget|timeout'):
            representation_boundary_synthesis(circuit, clean_inputs=clean, **limits)


def test_phase_measurements_and_modified_source_inputs_are_outside_declared_family():
    for source in ('Z q0', 'HMR q0 b0', 'CX q0 q1\nCCX q0 q1 q2'):
        result = representation_boundary_synthesis(km.Circuit(source))
        assert result['status'] == 'not_applicable'


def test_adapted_constructor_claim_is_rechecked_against_complete_frozen_boundary(monkeypatch):
    import strata.algebraic_synthesis as algebraic
    def incorrect(*args, **kwargs):
        return dict(status='constructed', candidate='X q0', structural_witness={})
    monkeypatch.setattr(algebraic, 'algebraic_boundary_synthesis', incorrect)
    circuit = km.Circuit('CCX q0 q1 q2')
    result = representation_boundary_synthesis(circuit, max_encodings=4)
    assert result['candidate'] == str(circuit) and result['selected_mechanism'] == 'source_incumbent'
    assert not result['alternatives']
    assert all(row['reason'] == 'representation_complete_boundary_check' for row in result['failures'])
