from dataclasses import replace

import kickmix as km
import numpy as np
import pytest

from strata.accounting import measured_resources
from strata.affine_transport import affine_transport_synthesis, recover_affine_transport
from strata.algebraic_synthesis import algebraic_boundary_synthesis
from strata.inference import functions_at
from strata.model import StrataError


UPDATE = 'CCX q0 q2 q3\nCCX q1 q2 q3'


@pytest.mark.parametrize('body,orientation', [
    (UPDATE + '\nCX q3 q0', 'output'),
    ('CX q3 q0\n' + UPDATE, 'input'),
    (UPDATE + '\nSWAP q0 q3', None),
    (UPDATE + '\nX q0', None),
    (UPDATE + '\nX q3\nCX q3 q0', 'output'),
])
def test_global_transport_unlocks_synthesis_and_preserves_all_dirty_inputs(body, orientation):
    # q4 is an untouched arbitrary wire; q5 is explicitly clean and its named
    # physical interface is retained even though this example needs no helper.
    circuit = km.Circuit('APPEND_TO_REGISTER q5 r0\nREGISTER r0 "physical_capacity"\n'
                         'APPEND_TO_REGISTER q4 r1\nREGISTER r1 "untouched_dirty"\n' + body)
    assert algebraic_boundary_synthesis(circuit, clean_inputs=[5])['status'] == 'not_applicable'
    result = affine_transport_synthesis(circuit, clean_inputs=[5])
    assert result['status'] == 'constructed' and result['improved']
    assert result['structural_witness']['affine_transport_factored']
    assert result['resource_bill']['before']['toffoli'] == 2
    assert result['resource_bill']['after']['toffoli'] == 1
    assert result['transport']['wires'] == [0, 1, 2, 3, 4]
    assert result['transport']['matrix_rows'][4] == 16 and result['transport']['offsets'][4] == 0
    if orientation:
        assert result['orientation'] == orientation
        assert any(row['orientation'] != orientation and row['status'] == 'not_applicable' for row in result['failures'])
    assert result['conversion_bill']['complete_native'] == measured_resources(km.Circuit(result['candidate']))
    assert result['proof']['exact_phase'] and result['proof']['all_public_quantum_inputs']
    for line in str(circuit).splitlines():
        if line.startswith(('REGISTER', 'APPEND_TO_REGISTER')):
            assert line in result['candidate'].splitlines()
    outputs = []
    for native in (circuit, km.Circuit(result['candidate'])):
        assert native.num_qubits == 6
        sim = km.Simulator(batch_size=32, seed=701)
        initial = np.zeros((6, 32), dtype=np.bool_)
        for q in range(5):
            initial[q] = [(row >> q) & 1 for row in range(32)]
        wires = [km.q(q) for q in range(6)]
        sim.write_across_shots(wires, initial)
        sim.do(native)
        outputs.append((sim.read_across_shots(wires), sim.read_phase_flipped_across_shots()))
    assert all(np.array_equal(a, b) for a, b in zip(*outputs))


def test_recovered_map_emits_real_swap_and_intercept_and_validates_both_inverses():
    circuit = km.Circuit('SWAP q0 q1\nCX q1 q2\nX q0')
    state = functions_at(circuit)
    transport = recover_affine_transport(state['functions'], [0, 1, 2])
    assert transport.offsets == (1, 0, 0)
    assert any(line.startswith('SWAP ') for line in transport.forward)
    assert any(line.startswith('X ') for line in transport.forward)
    with pytest.raises(StrataError, match='inverse'):
        replace(transport, inverse=transport.inverse[:-1])
    with pytest.raises(StrataError, match='recovered map'):
        replace(transport, forward=transport.forward[:-1])


def test_full_parent_pricing_can_reject_a_paid_native_gain():
    circuit = km.Circuit(UPDATE + '\nCX q3 q0')
    calls = []
    def price(candidate):
        resources = measured_resources(candidate)
        calls.append(str(candidate))
        return dict(resources, host_cost=100 * resources['instructions'] + resources['toffoli'])
    result = affine_transport_synthesis(circuit, cost='host_cost', resource_measure=price)
    assert calls and result['selected_mechanism'] == 'source_incumbent'
    assert not result['structural_witness']['affine_transport_factored']
    assert result['cost'] == 302
    for candidate in result['alternatives']:
        assert candidate['cost'] == price(km.Circuit(candidate['candidate']))['host_cost']
        assert candidate['conversion_bill']['emitted_transport']['instructions'] > 0


def test_singular_linear_part_and_coupled_nonlinear_residuals_fail_closed():
    singular = km.Circuit('CX q0 q1\nCCX q0 q1 q2\nCCX q1 q2 q0\nCCX q0 q2 q1\n'
                          'CX q1 q0\nCX q0 q1\nCX q1 q0')
    result = affine_transport_synthesis(singular)
    assert result['status'] == 'not_applicable' and result['reason'] == 'affine_transport_singular_linear_part'
    coupled = affine_transport_synthesis(km.Circuit('CCX q0 q1 q2\nCCX q1 q2 q0\nCX q0 q1'))
    assert coupled['status'] == 'not_applicable'
    assert len(coupled['failures']) == 2 and 'candidate' not in coupled
    for source in (UPDATE + '\nCX q3 q0\nZ q0', 'HMR q0 b0', 'R q0'):
        assert affine_transport_synthesis(km.Circuit(source))['status'] == 'not_applicable'
    assert affine_transport_synthesis(km.Circuit(UPDATE))['reason'] == 'boundary already has preserved-input XOR form'


def test_malformed_residual_constructor_does_not_bypass_complete_boundary(monkeypatch):
    import strata.algebraic_synthesis as algebraic
    monkeypatch.setattr(algebraic, 'algebraic_boundary_synthesis',
                        lambda *args, **kwargs: dict(status='constructed', candidate='X q0'))
    result = affine_transport_synthesis(km.Circuit(UPDATE + '\nCX q3 q0'))
    assert result['status'] == 'not_applicable' and 'candidate' not in result
    assert all(row['reason'] == 'affine_transport_complete_boundary_check' for row in result['failures'])


def test_semantic_and_constructor_limits_are_explicit():
    circuit = km.Circuit(UPDATE + '\nCX q3 q0')
    result = affine_transport_synthesis(circuit, max_operations=2)
    assert result['status'] == 'unknown' and 'operation_budget' in result['reason']
    result = affine_transport_synthesis(circuit, max_factor_steps=1)
    assert result['status'] == 'unknown' and 'candidate' not in result
    wide = km.Circuit(str(circuit) + '\nAPPEND_TO_REGISTER q1000 r0')
    assert affine_transport_synthesis(wide, max_variables=16)['reason'] == 'affine_transport_public_variable_budget'
    for limits in ({'max_orders': 0}, {'max_terms': True}, {'timeout': float('nan')}):
        with pytest.raises(StrataError):
            affine_transport_synthesis(circuit, **limits)
