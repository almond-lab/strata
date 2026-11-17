from itertools import product

import kickmix as km
import pytest

from strata.boolean_synthesis import joint_boundary_synthesis, lower_xor_and_witness
from strata.model import StrataError, digest
from strata.scheduled_boolean_synthesis import scheduled_xor_and_witness, scheduled_boundary_synthesis
from strata.semantic import equivalent_windows


def chain_witness(inputs=5):
    rows = [list(row) for row in product(range(2), repeat=inputs)]
    values = [[int(all(row))] for row in rows]
    tables = [(1 << len(rows)) - 1] + [sum(row[i] << j for j, row in enumerate(rows)) for i in range(inputs)]
    gates = []
    for index in range(inputs - 1):
        left = 1 if index == 0 else inputs + index
        right = index + 2
        a, b = tables[left], tables[right]
        gates.append(dict(left=a, right=b, product=a & b,
                          left_coefficients=1 << left, right_coefficients=1 << right))
        tables.append(a & b)
    receipt = dict(value=len(gates), witness=gates,
                   output_coefficients=[1 << (len(tables) - 1)],
                   contract_hash=digest(dict(inputs=rows, outputs=values)))
    return receipt, rows, values


def test_native_schedule_frontier_trades_physical_width_for_complete_recomputation():
    witness, rows, values = chain_witness()
    layout = dict(input_wires=list(range(5)), target_wires=[5], clean_wires=[6, 7, 8], num_qubits=9)
    baseline = lower_xor_and_witness(witness, rows, values, **layout)
    result = scheduled_xor_and_witness(witness, rows, values, **layout)
    assert result['status'] == 'constructed'
    assert result['planning']['pareto_complete']
    assert [(row['resources']['Q'], row['resources']['toffoli']) for row in result['alternatives']] == [(8, 9), (9, 7)]
    assert result['alternatives'][0]['schedule']['recomputations'] == 1
    for row in result['alternatives']:
        assert row['ledger_matches_native']
        assert row['proof']['method'] == 'independent_kickmix_full_boundary_replay'
        assert row['proof']['workspace_restored'] and row['proof']['phase_zero']
        assert row['resource_ledger']['instructions'] == len(km.Circuit(row['candidate']))
        check = equivalent_windows(baseline['candidate'], row['candidate'], clean_inputs=[6, 7, 8],
                                   max_operations=1000, max_variables=32, max_terms=4096)
        assert check['equivalent'] is True, check


def test_existing_clean_capacity_selects_feasible_recomputation_schedule():
    witness, rows, values = chain_witness()
    result = scheduled_xor_and_witness(witness, rows, values, input_wires=list(range(5)),
        target_wires=[5], clean_wires=[6, 7], num_qubits=8)
    assert result['status'] == 'constructed'
    assert [(row['resources']['Q'], row['resources']['toffoli']) for row in result['alternatives']] == [(8, 9)]
    assert result['planning']['minimum_cost_proved'] == '9'


def test_whole_boundary_scheduling_uses_actual_parent_objective_and_preserves_metadata():
    source = km.Circuit('''REGISTER r0 "public"
APPEND_TO_REGISTER q3 r0
CCX q0 q1 q4
CCX q4 q2 q5
CX q5 q3
CCX q4 q2 q5
CCX q0 q1 q4
''')
    joint = joint_boundary_synthesis(source, clean_inputs=[4, 5], max_and=2, timeout=5)
    assert joint['status'] == 'constructed', joint
    seen = []

    def parent_price(native):
        seen.append(str(native))
        return dict(Q=50, toffoli=native.operation_statistics().num_toffoli_like + 100, t=0)

    result = scheduled_boundary_synthesis(source, joint_result=joint, resource_measure=parent_price,
                                           clean_inputs=[4, 5], max_operations=1000)
    assert result['status'] == 'constructed', result
    assert seen and result['objective_scope'] == 'parent_context'
    selected = min(result['alternatives'], key=lambda row: row['cost'])
    assert result['selected_cost'] == 50 * (selected['resources']['toffoli'] + 100)
    assert 'REGISTER r0 "public"' in result['candidate']
    assert selected['resource_ledger']['instructions'] == len(km.Circuit(result['candidate']))
    assert equivalent_windows(str(source), result['candidate'], clean_inputs=[4, 5],
                               max_operations=1000, max_variables=32, max_terms=4096)['equivalent'] is True


def test_layout_holes_keep_physical_width_separate_from_live_storage_peak():
    witness, rows, values = chain_witness()
    result = scheduled_xor_and_witness(witness, rows, values, input_wires=list(range(5)),
        target_wires=[9], clean_wires=[5, 6, 7], num_qubits=10)
    assert result['status'] == 'constructed'
    assert all(row['resources']['Q'] == 10 for row in result['alternatives'])
    assert result['alternatives'][0]['schedule']['peak_width'] == 9
    assert result['alternatives'][0]['resource_ledger']['Q'] == 10


def test_bounded_native_emission_and_mutated_witness_fail_closed():
    witness, rows, values = chain_witness()
    layout = dict(input_wires=list(range(5)), target_wires=[5], clean_wires=[6, 7, 8], num_qubits=9)
    result = scheduled_xor_and_witness(witness, rows, values, max_operations=1, **layout)
    assert result['status'] == 'unknown' and result['alternatives'] == []
    assert result['failures'][0]['reason'] == 'scheduled_native_operation_budget'
    witness['witness'][0]['product'] ^= 1
    with pytest.raises(StrataError, match='AND replay mismatch'):
        scheduled_xor_and_witness(witness, rows, values, **layout)
