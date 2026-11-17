from itertools import product

import kickmix as km
import pytest

from strata.boolean_synthesis import lower_xor_and_witness, joint_boundary_synthesis
from strata.domains import execute_assignment
from strata.floors import exact_minimum
from strata.model import StrataError


def test_affine_operand_lowering_replays_all_dirty_full_adder_outputs():
    polynomials = [[[0], [1], [2]], [[0, 1], [0, 2], [1, 2]]]
    witness = exact_minimum(polynomials, 3)
    rows = [list(row) for row in product(range(2), repeat=3)]
    values = [[a ^ b ^ c, (a & b) ^ (a & c) ^ (b & c)] for a, b, c in rows]
    built = lower_xor_and_witness(witness, rows, values, input_wires=[0, 1, 2],
                                  target_wires=[3, 4], clean_wires=[], num_qubits=5)
    assert built['status'] == 'constructed'
    assert built['resource_bill']['native']['toffoli'] == 1
    assert built['resource_bill']['materialized_products'] == 0
    for row, delta in zip(rows, values):
        for dirty in product(range(2), repeat=2):
            initial = row + list(dirty)
            result = execute_assignment(built['candidate'].splitlines(), {f'q{q}': bit for q, bit in enumerate(initial)}, 5)
            assert [result['outputs'][str(q)] for q in range(5)] == row + [dirty[q] ^ delta[q] for q in range(2)]
            assert result['phase'] == 0


def test_shared_intermediate_product_is_materialized_once_and_restored():
    polynomials = [[[0, 1]], [[0, 1, 2]]]
    witness = exact_minimum(polynomials, 3, timeout=3)
    rows = [list(row) for row in product(range(2), repeat=3)]
    values = [[a & b, a & b & c] for a, b, c in rows]
    built = lower_xor_and_witness(witness, rows, values, input_wires=[0, 1, 2],
                                  target_wires=[3, 4], clean_wires=[5, 6, 7], num_qubits=8)
    assert built['resource_bill']['materialized_products'] == 1
    assert built['resource_bill']['native']['toffoli'] == 3
    assert built['layout']['unused_clean_wires'] == [6, 7]
    assert built['proof']['preserved'] == 'all input values, zero workspace, exact zero sign'
    unknown = lower_xor_and_witness(witness, rows, values, input_wires=[0, 1, 2],
                                    target_wires=[3, 4], clean_wires=[], num_qubits=5)
    assert unknown['status'] == 'unknown' and unknown['reason'] == 'insufficient_clean_product_workspace'


def _pipeline(inputs, target, helpers, invert=False):
    a, b, c = inputs
    x, y, z = helpers
    return '\n'.join([f'CCX q{a} q{b} q{x}', f'CCX q{c} q{x} q{y}',
                      f'CX q{y} q{z}', f'CX q{z} q{target}',
                      *([f'X q{target}'] if invert else []),
                      f'CX q{y} q{z}', f'CCX q{c} q{x} q{y}', f'CCX q{a} q{b} q{x}'])


@pytest.mark.parametrize('inputs,target,helpers,invert', [
    ([0, 1, 2], 3, [4, 5, 6], False),
    ([2, 0, 1], 3, [6, 4, 5], True),
    ([5, 2, 0], 1, [3, 4, 6], False),
])
def test_joint_final_boundary_retires_multilayer_intermediate_bank_blind_to_layout(inputs, target, helpers, invert):
    original = km.Circuit(_pipeline(inputs, target, helpers, invert))
    built = joint_boundary_synthesis(original, clean_inputs=helpers, timeout=3)
    assert built['status'] == 'constructed'
    assert built['structural_witness']['removed_materialization_bank']
    assert len(built['structural_witness']['retired_scratch_wires']) == 2
    assert built['resource_bill']['before']['toffoli'] == 4
    assert built['resource_bill']['after']['toffoli'] == 3
    assert built['proof']['status'] == 'equivalent'
    assert built['layout']['external_input_indices_unchanged']
    candidate = built['candidate'].splitlines()
    for row in product(range(2), repeat=4):
        assignment = {f'q{q}': 0 for q in range(original.num_qubits)}
        assignment.update({f'q{q}': bit for q, bit in zip([*inputs, target], row)})
        before = execute_assignment(str(original).splitlines(), assignment, original.num_qubits)
        after = execute_assignment(candidate, assignment, original.num_qubits)
        assert before['outputs'] == after['outputs'] and before['phase'] == after['phase']


def test_phase_and_unapproved_measurement_outputs_are_concrete_obstructions():
    assert joint_boundary_synthesis(km.Circuit('Z q0'))['status'] == 'unknown'
    measured = km.Circuit('CCX q0 q1 q2\nCX q2 q3\nHMR q2 b0\nCZ q0 q1 if b0')
    unknown = joint_boundary_synthesis(measured, clean_inputs=[2])
    assert unknown['status'] == 'unknown' and unknown['required_internal_bits'] == [0]
    built = joint_boundary_synthesis(measured, clean_inputs=[2], internal_bits=[0])
    assert built['status'] == 'constructed' and built['resource_bill']['after']['toffoli'] == 1
    assert built['proof']['internal_classical_outputs'] == ['b0']
    broken = km.Circuit(str(measured).replace('CZ q0 q1 if b0', 'Z q0 if b0'))
    assert joint_boundary_synthesis(broken, clean_inputs=[2], internal_bits=[0])['status'] == 'unknown'


def test_joint_search_bound_and_aliasing_fail_closed():
    original = km.Circuit(_pipeline([0, 1, 2], 3, [4, 5, 6]))
    result = joint_boundary_synthesis(original, clean_inputs=[4, 5, 6], max_expansions=1)
    assert result['status'] == 'unknown' and result['exact_search']['kind'] == 'unknown'
    witness = exact_minimum([[[0, 1]]], 2)
    rows = [list(row) for row in product(range(2), repeat=2)]
    with pytest.raises(StrataError, match='disjoint'):
        lower_xor_and_witness(witness, rows, [[a & b] for a, b in rows],
                              input_wires=[0, 1], target_wires=[1], clean_wires=[], num_qubits=2)


def test_independent_product_restoration_does_not_inflate_multilayer_evidence():
    circuit = km.Circuit('CCX q0 q1 q4\nCCX q0 q2 q5\nCX q4 q3\nCX q5 q3\n'
                         'CCX q0 q2 q5\nCCX q0 q1 q4')
    built = joint_boundary_synthesis(circuit, clean_inputs=[4, 5], timeout=3)
    assert built['status'] == 'constructed'
    assert built['structural_witness']['retired_scratch_wires'] == [4, 5]
    assert built['structural_witness']['source_nonlinear_dataflow_depth'] == 1
    assert not built['structural_witness']['removed_materialization_bank']
