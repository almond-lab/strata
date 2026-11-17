"""Issue regressions: promised resets, observed kernels and continued lifting."""
import json
import sqlite3

import kickmix as km
import pytest

from strata.discovery import discovery_brief
from strata.floors import circuit_floors
from strata.semantic import lift_regions
from strata.streaming import scan_circuit


def test_floors_prices_identity_resets_and_zero_debt_measured_value_maps():
    text = 'R q2\nCCX q0 q1 q2\nCX q2 q3\nR q4\nCCX q0 q2 q4\nCX q4 q3'
    circuit = km.Circuit(text)
    assert circuit_floors(circuit, clean_inputs=[2,4])['status'] == 'exact'
    assert circuit_floors(circuit, clean_inputs=[2,4], at=3)['value'] == 1
    assert circuit_floors(circuit)['reason'] == 'reset_not_proven_zero'
    assert circuit_floors(km.Circuit('R q2\nX q2\nR q2'), clean_inputs=[2])['reason'] == 'reset_not_proven_zero'
    measured = circuit_floors(km.Circuit('R q2\nHMR q2 b0'), clean_inputs=[2])
    assert measured['status'] == 'exact' and measured['value'] == 0
    assert measured['global_quantum_floor'] is False


@pytest.mark.parametrize('metadata', ['', 'REGISTER r0 "public"\nAPPEND_TO_REGISTER q0 r0\n'])
def test_clean_lookup_loader_discovery_retains_reset_proofs_and_phase_repair(tmp_path, metadata):
    path = tmp_path/'loader.kmx'
    path.write_text(metadata+'R q2\nCCX q0 q1 q2\nCX q2 q3\nHMR q2 b0\nCZ q0 q1 if b0')
    brief = discovery_brief(path, tmp_path/'brief', clean_inputs=[2])
    assert brief['coverage']['exact_fraction'] == 1
    assert brief['kernel_coverage']['exact_fraction'] == 1
    sem = brief['repeated_kernels'][0]['semantics']
    assert sem['status'] == 'exact'
    # The original q2 is canonical q0 because the reset is its first touch.
    assert sem['clean_input_assumptions'] == [0]
    assert sem['effects']['reset_proofs'][0]['wire'] == 0
    assert sem['effects']['phase_debts'][0]['status'] == 'discharged'


def test_scan_never_reapplies_prefix_clean_promises(tmp_path):
    circuit = km.Circuit('R q0\nX q0\nR q0\nX q0')
    scan = scan_circuit(circuit, tmp_path, max_window=2, clean_inputs=[0])
    assert scan['coverage']['exact_instructions'] == 2
    assert scan['coverage']['unknown_instructions'] == 2
    by_start = {g['examples'][0]['start']: g for g in scan['groups']}
    assert by_start[0]['semantics']['status'] == 'exact'
    assert by_start[2]['semantics']['reason'] == 'reset_not_proven_zero'
    assert by_start[0]['fingerprint'] != by_start[2]['fingerprint']


def test_metadata_only_prefix_does_not_consume_the_original_clean_promise(tmp_path):
    circuit = km.Circuit('REGISTER r0 "public"\nAPPEND_TO_REGISTER q1 r0\nR q0\nX q0\nR q0')
    lifted = lift_regions(circuit, max_window=1, clean_inputs=[0])
    assert next(r for r in lifted['regions'] if r['start']==2)['semantics']['clean_input_assumptions']==[0]
    assert lifted['unknown'][-1]['reason']=='reset_not_proven_zero'
    scan = scan_circuit(circuit,tmp_path,max_window=1,clean_inputs=[0],range_unit='text-line')
    reset_groups = [g for g in scan['groups'] if g['examples'][0]['start'] in {2,4}]
    assert {g['semantics']['status'] for g in reset_groups} == {'exact','opaque'}


def _cell(base, variant):
    return [f'CCX q{base} q{base+1} q{base+2}',
            f'CX q{base+2} q{base+3}'] + [f'X q{base+3}']*variant + [
            f'CCX q{base+1} q{base+3} q{base+4}',
            f'CX q{base+4} q{base+5}',
            f'CCX q{base} q{base+4} q{base+5}',
            f'HMR q{base+5} b{base}']


def test_partial_paid_repeats_survive_nonperiodic_ends_and_linear_variants(tmp_path):
    lines = ['X q99', 'CCZ q70 q71 q72']
    for b,v in ((0,0),(10,1),(20,2)):
        lines.extend(_cell(b,v))
    lines.extend(['CCZ q73 q74 q75','Z q99'])
    scan = scan_circuit(km.Circuit('\n'.join(lines)), tmp_path)
    repeated = next(g for g in scan['paid_skeletons'] if g['occurrences'] == 3)
    assert repeated['anchors_per_occurrence'] == 4
    assert repeated['aggregate_static_toffoli'] == 9
    assert repeated['linear_layer_variants'] == 3
    assert repeated['semantics']['status'] == 'observation'
    assert repeated['rewrite_boundary_verified'] is False
    assert scan['coverage']['skeleton_indexed_paid_fraction'] == 1
    assert scan['coverage']['repeated_skeleton_paid_operations'] >= 9
    assert not any(g['occurrences'] == 3 for g in scan['groups'])
    with sqlite3.connect(scan['index']) as db:
        assert db.execute('SELECT count(*) FROM skeleton_occurrences WHERE fingerprint=?', (repeated['fingerprint'],)).fetchone()[0] == 3


def test_scoped_cells_group_paid_skeleton_with_distinct_linear_hashes(tmp_path):
    lines = []
    for i in range(5):
        lines.extend([f'PUSH_CONDITION if b{i+100}', *_cell(i*10,i), 'POP_CONDITION'])
    scan = scan_circuit(km.Circuit('\n'.join(lines)),tmp_path)
    repeated = next(g for g in scan['paid_skeletons'] if g['occurrences'] == 5)
    assert repeated['aggregate_static_toffoli'] == 15
    assert repeated['linear_layer_variants'] == 5
    assert scan['coverage']['exact_fraction'] == 1
    cut = scan_circuit(km.Circuit('\n'.join(lines)),tmp_path/'cut',max_window=3)
    assert cut['coverage']['unknown_instructions'] > 0
    assert all(g['semantics']['status'] == 'unknown' for g in cut['groups'])


def test_operation_and_variable_budget_cuts_continue_to_end():
    text = 'HMR q0 b0\n' + '\n'.join(f'CCX q{i} q{i+1} q{i+2}' for i in range(1,80))
    lifted = lift_regions(text, max_window=8, max_variables=12)
    assert lifted['coverage']['exact_fraction'] == 1
    assert lifted['coverage']['budget_cut_reasons']['operation_budget'] > 0
    assert max(r['stop'] for r in lifted['regions']) == 80
    wide = lift_regions(text, max_window=16384, max_variables=12)
    assert wide['coverage']['exact_fraction'] == 1
    assert wide['coverage']['budget_cut_reasons']['variable_budget'] > 0
    assert max(r['stop'] for r in wide['regions']) == 80


def test_scope_budget_overflow_ends_at_pop_and_scan_continues():
    text = 'X q9\nPUSH_CONDITION if b0\n' + 'X q0\n'*20 + 'POP_CONDITION\nCCX q1 q2 q3\nX q3'
    lifted = lift_regions(text, max_window=4)
    assert [(r['start'],r['stop']) for r in lifted['regions']] == [(0,1),(23,25)]
    assert lifted['unknown'] == [dict(start=1,stop=23,status='bounded',reason='operation_budget',operation_count=22)]
    assert lifted['coverage']['exact_operations'] == 3
    assert lifted['coverage']['unknown_operations'] == 22
    assert lifted['coverage']['budget_cuts'] == 1


def test_large_window_still_recovers_exact_subwindows_after_semantic_budget():
    text = 'HMR q0 b0\n' + 'CX q0 q1\nCCX q1 q2 q3\n'*100
    lifted = lift_regions(text, max_window=16384, max_product_terms=4)
    assert lifted['coverage']['exact_fraction'] == 1
    assert lifted['coverage']['budget_cuts'] > 0
    assert lifted['regions'][-1]['stop'] == 201


def test_untouched_clean_and_hmr_zero_facts_cross_budget_cuts_with_provenance(tmp_path):
    text = 'X q9\nX q9\nR q0\nCCX q1 q2 q0\nHMR q0 b0\nR q0\nX q0\nR q0'
    circuit = km.Circuit(text)
    lifted = lift_regions(circuit,max_window=1,clean_inputs=[0])
    by_start = {r['start']:r for r in lifted['regions']}
    assert by_start[2]['boundary_zero_evidence']['0']['kind']=='declared_artifact_input'
    assert by_start[5]['boundary_zero_evidence']['0']==dict(kind='unconditional_hmr',origin_at=4,origin_wire=0)
    assert lifted['unknown']==[dict(start=7,stop=8,status='opaque',reason='reset_not_proven_zero',operation_count=1)]
    scan = scan_circuit(circuit,tmp_path,max_window=1,clean_inputs=[0],range_unit='text-line')
    exact_resets = next(g for g in scan['groups'] if g['semantics']['status']=='exact' and g['semantics']['effects']['reset_proofs'])
    assert exact_resets['occurrences']==2
    assert {e['boundary_zero_evidence']['q0']['kind'] for e in exact_resets['examples']}=={'declared_artifact_input','unconditional_hmr'}
    assert next(g for g in scan['groups'] if g['semantics'].get('reason')=='reset_not_proven_zero')['examples'][0]['start']==7


@pytest.mark.parametrize('middle,clean,good_wire,bad_wire',[
    ('SWAP q0 q1',[0],1,0),
    ('SWAP q0 q1 if b0',[0],None,0),
    ('CSWAP q2 q0 q1',[0,2],0,1),
    ('X q0 if b0',[0],None,0),
    ('HMR q0 b0 if b1',[],None,0),
    ('PUSH_CONDITION if b0\nHMR q0 b1\nPOP_CONDITION',[],None,0),
    ('Z_POW q9 0.5',[0],0,1),
])
def test_zero_transport_handles_swaps_conditions_and_opaque_barriers(middle,clean,good_wire,bad_wire):
    tail = ([f'R q{good_wire}'] if good_wire is not None else []) + [f'R q{bad_wire}']
    lines=middle.splitlines()+tail
    report=lift_regions('\n'.join(lines),max_window=1,clean_inputs=clean)
    assert any(r['start']==len(lines)-1 and r['reason']=='reset_not_proven_zero' for r in report['unknown'])
    if good_wire is not None:
        assert any(r['start']==len(lines)-2 and r['semantics']['effects']['reset_proofs'] for r in report['regions'])


def test_prefix_zero_fact_index_is_bounded_and_evictions_are_explicit(tmp_path):
    circuit=km.Circuit('\n'.join(f'HMR q{i} b0' for i in range(20))+'\nR q0')
    scan=scan_circuit(circuit,tmp_path,max_window=1,max_variables=3)
    assert scan['limits']['max_boundary_zero_facts']==3
    assert scan['coverage']['zero_fact_evictions']==17
    assert any(g['semantics'].get('reason')=='reset_not_proven_zero' for g in scan['groups'])
