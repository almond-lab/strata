"""Regressions for finite cuts, measured repairs, and exact care-row search."""
from itertools import product

import kickmix as km
import pytest

from strata.floors import exact_minimum_on_rows, verify_minimum_witness
from strata.lookup_structure import measured_value_surrogate
from strata.promised_domain import check_promised_candidate, promised_domain_synthesis
from strata.restructure import restructure_file


def address_promise(address, arbitrary, clean):
    return dict(version=1, type='address_promise', address_wires=address, arbitrary_wires=arbitrary,
                clean_wires=clean, interval=[0, (1 << len(address)) - 1])


def finite_rows(width):
    return dict(version=1, type='finite', assignments=[list(row) for row in product(range(2), repeat=width)])


def test_finite_cut_replays_its_actual_register_binding(tmp_path):
    source = tmp_path / 'parent.kmx'
    source.write_text(
        'APPEND_TO_REGISTER q0 r0\nAPPEND_TO_REGISTER q1 r0\nREGISTER r0 "address"\n'
        'APPEND_TO_REGISTER q2 r1\nREGISTER r1 "word"\nX q0\n'
        'CCX q0 q1 q2\nCCX q0 q1 q2\nX q0')
    result = restructure_file(source, tmp_path / 'out', domain=finite_rows(3), domain_at_cut=True,
                              scan_range=(1,3), cost='toffoli')
    assert result['status'] == 'verified_domain_improvement'
    assert result['cost_delta'] == -2
    assert result['check']['domain_hash'] == result['episode']['promise']['domain_hash']


def test_promised_producer_keeps_clean_input_values_live_for_consumers(tmp_path):
    source = tmp_path / 'parent.kmx'
    source.write_text('CCX q0 q1 q3\nCX q3 q2\nR q4')
    promise = address_promise([0,1], [2], [3,4])
    del promise['interval']
    promise['addresses'] = [0,3]
    report = restructure_file(source, tmp_path / 'out', clean_inputs=(3,4), domain=promise,
                              domain_at_cut=True, scan_range=(0,1), cost='toffoli')
    assert report['status'] == 'verified_domain_improvement'
    assert report['cost_delta'] == -1
    assert report['check']['required_zero_outputs'] == []
    assert str(km.Circuit.from_file(report['best']['path'])).splitlines() == [
        'CX q0 q3', 'CX q3 q2', 'R q4',
    ]


@pytest.mark.parametrize('compact', [True,False])
def test_promised_narrower_candidate_checks_every_omitted_clean_wire(compact):
    source = km.Circuit('CCX q0 q1 q3\nCX q3 q2\nCCX q0 q1 q3')
    candidate = km.Circuit('CCX q0 q1 q2')
    promise = address_promise([0,1], [2], [3]) if compact else dict(
        assignments=[[*row, 0] for row in product(range(2), repeat=3)])
    good = check_promised_candidate(source,candidate,promise,clean_inputs=(3,))
    assert good['equivalent'] is True
    assert good['omitted_trailing_clean_wires'] == [3]
    # A trailing clean input is also an observable output if the source writes it.
    leaking = km.Circuit(str(source) + '\nX q3')
    assert check_promised_candidate(leaking,candidate,promise,clean_inputs=(3,))['equivalent'] is False


def test_sparse_classical_ids_use_active_table_budget():
    circuit = km.Circuit('CCX q0 q1 q3\nCX q3 q2\nHMR q3 b605\nCZ q0 q1 if b605')
    result = promised_domain_synthesis(circuit,address_promise([0,1], [2], [3]), clean_inputs=(3,),
                                       internal_bits=(605,), max_variables=16, max_table_cells=32)
    assert result['table_width'] == 5
    assert result['measurement_preparation']['status'] == 'verified'


def test_sat_proves_and_replays_four_and_minimum():
    rows = [[i >> bit & 1 for bit in range(4)] for i in range(16)]
    words = ['0001111101010011', '0111100010111100', '0110100111001000']
    targets = [[int(word[i]) for word in words] for i in range(16)]
    result = exact_minimum_on_rows(rows, targets, max_and=4, timeout=15,
                                   max_states=500000, max_expansions=500000)
    assert result['backend'] == 'sat'
    assert result['kind'] == 'exact_small' and result['value'] == 4
    assert [(trial['depth'],trial['status']) for trial in result['sat_trials']] == [(3,'unsat'),(4,'sat')]
    assert verify_minimum_witness(result,rows,targets)['status'] == 'verified'


@pytest.mark.parametrize('scoped', [False,True])
def test_measured_repair_of_value_with_cancelled_xor(scoped):
    data = 'CCX q0 q1 q4\nCX q2 q4\nCX q4 q3\nCX q2 q4\nHMR q4 b0\n'
    repair = ('PUSH_CONDITION if b0\nCCX q0 q1 q4\nZ q4\nCCX q0 q1 q4\nPOP_CONDITION'
              if scoped else 'CZ q0 q1 if b0')
    circuit = km.Circuit(data+repair)
    limits = dict(max_operations=256,max_variables=32,max_terms=4096,max_product_terms=4096)
    _, receipt = measured_value_surrogate(circuit,(4,),(0,),limits)
    assert receipt['status'] == 'verified'
    assert receipt['proof']['outcome_independence_proved']


def test_external_diagonal_is_replayed_with_arbitrary_classical_input():
    circuit = km.Circuit('PUSH_CONDITION if b912\nZ q0\nPOP_CONDITION\n'
                         'CCX q0 q1 q3\nCX q3 q2\nCCX q0 q1 q3')
    # Proof covers either value of the external repayment bit.
    result = promised_domain_synthesis(circuit, address_promise([0,1], [2], [3]), clean_inputs=(3,),
                                       cost='toffoli', max_table_cells=32)
    assert result['status'] == 'constructed'
    assert result['cost']==1 and 'b912' in result['candidate']
    assert result['check']['equivalent']


def test_input_consuming_map_has_an_explicit_scope_reason():
    circuit = km.Circuit('SWAP q0 q1')
    result = promised_domain_synthesis(circuit,address_promise([0,1], [], []))
    assert result['reason'] == 'in_place_map_out_of_scope'



def test_ordered_in_place_can_consume_preserved_address_on_its_second_update():
    circuit = km.Circuit('CCX q0 q1 q2\n' * 3 + 'CX q2 q0')
    result = promised_domain_synthesis(circuit,address_promise([0,1,2], [], []), cost='toffoli')
    assert result['status'] == 'constructed'
    assert result['cost'] == 1
    assert result['structural_witness']['selected_completion'] == 'ordered_in_place_updates'
    assert result['check']['equivalent']
