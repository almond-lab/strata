"""Adversarial measured-lookup review with independent native branch replay."""
import itertools
import json

import kickmix as km
import numpy as np

from strata.lookup import resynthesize_measured_lookup
from strata.restructure import default_registry, restructure_file


QUADRATIC = '''CCX q0 q1 q4
CCX q0 q2 q4
CX q4 q3
HMR q4 b0
CZ q0 q1 if b0
CZ q0 q2 if b0'''


def _forced_native(text, public, width, bit_count, outcomes):
    """Replay every initial basis/classical state for one formal outcome row.

    Replace HMR by its exact basis/sign action with a fresh clean sink. Native
    kickmix then independently evaluates all ordinary and conditional gates.
    The original wire layout, classical outputs and exact sign are compared.
    """
    rows=[]
    fresh=km.Circuit(text).num_qubits
    measurement=0
    for line in str(km.Circuit(text)).splitlines():
        words=line.split()
        if words[0]!='HMR':
            rows.append(line)
            continue
        outcome=outcomes[measurement]
        suffix=' '+' '.join(words[3:]) if len(words)>3 else ''
        if outcome:
            rows.append(f'Z {words[1]}{suffix}')
        rows.extend([f'SWAP {words[1]} q{fresh}{suffix}',f'BIT_STORE{outcome} {words[2]}{suffix}'])
        fresh+=1
        measurement+=1
    assert measurement==len(outcomes)
    inputs=list(itertools.product((False,True),repeat=len(public)+bit_count))
    sim=km.Simulator(batch_size=len(inputs),seed=492)
    for column,q in enumerate(public):
        sim.write_across_shots(km.q(q),np.array([row[column] for row in inputs]))
    for b in range(bit_count):
        sim.write_across_shots(km.b(b),np.array([row[len(public)+b] for row in inputs]))
    sim.do(km.Circuit('\n'.join(rows)))
    return ([sim.read_across_shots(km.q(q)).tolist() for q in range(width)],
            [sim.read_across_shots(km.b(b)).tolist() for b in range(bit_count)],
            sim.read_phase_flipped_across_shots().tolist())


def _all_branches(before, after, *, public, width, bits, measurements):
    for outcomes in itertools.product((0,1),repeat=measurements):
        assert _forced_native(before,public,width,bits,outcomes)==_forced_native(after,public,width,bits,outcomes)


def test_clean_holder_fanout_respects_dependencies_on_dirty_original_targets():
    source='''CCX q3 q2 q6
CCX q0 q1 q3
CCX q0 q2 q3
CCX q0 q1 q4
CCX q0 q2 q4
CCX q0 q1 q5
CCX q0 q2 q5
HMR q5 b0
CZ q0 q1 if b0
CZ q0 q2 if b0'''
    result=resynthesize_measured_lookup(km.Circuit(source),clean_inputs=[5])
    assert result['status']=='constructed'
    assert sum(line.startswith('CCX ') for line in result['candidate'].splitlines())==2
    _all_branches(source,result['candidate'],public=[0,1,2,3,4,6],width=7,bits=1,measurements=1)


def test_conditioned_write_revokes_clean_lease_and_repays_its_outcome_phase():
    source=QUADRATIC+'''
X q4 if b0
CCX q0 q1 q4
CCX q0 q2 q4
CX q4 q3
HMR q4 b1
CZ q0 q1 if b1
CZ q0 q2 if b1
PUSH_CONDITION if b0
NEG if b1
POP_CONDITION'''
    result=resynthesize_measured_lookup(km.Circuit(source),clean_inputs=[4])
    assert result['status']=='constructed'
    assert 'X q4 if b0' in result['candidate']
    assert 4 not in result['segments'][-1]['clean_at_entry']
    _all_branches(source,result['candidate'],public=[0,1,2,3],width=5,bits=2,measurements=2)


def test_conditional_hmr_does_not_turn_a_dirty_wire_into_an_unconditional_clean_lease():
    source=QUADRATIC+'''
PUSH_CONDITION if b0
HMR q3 b1
POP_CONDITION
CX q3 q2
CX q3 q2'''
    # The measured dirty value has an outstanding phase debt. Even though an
    # earlier quadratic segment is profitable, no complete candidate is emitted.
    result=resynthesize_measured_lookup(km.Circuit(source),clean_inputs=[4])
    assert result['status']=='not_applicable'
    assert 'phase debt' in result['reason']
    assert 'candidate' not in result


def test_repaired_conditional_hmr_keeps_inactive_branch_values_until_explicit_uncompute():
    source=QUADRATIC+'''
CX q3 q5
PUSH_CONDITION if b0
HMR q5 b1
Z q3 if b1
POP_CONDITION
CX q5 q2
CX q5 q2
BIT_INVERT b0
CX q3 q5 if b0
BIT_INVERT b0'''
    result=resynthesize_measured_lookup(km.Circuit(source),clean_inputs=[4,5])
    assert result['status']=='constructed'
    # The first HMR cleared q4 unconditionally. The later HMR clears q5 only
    # on b0=1; q5 remains dirty on the inactive branch until the final CX.
    assert 4 in result['segments'][-1]['clean_at_entry']
    assert 5 not in result['segments'][-1]['clean_at_entry']
    _all_branches(source,result['candidate'],public=[0,1,2,3],width=6,bits=2,measurements=2)


def test_higher_degree_measured_table_is_retained_while_supported_neighbor_improves():
    cubic='''CCX q0 q1 q4
CCX q4 q2 q5
CX q5 q6
HMR q5 b0
CCZ q0 q1 q2 if b0
HMR q4 b1
CZ q0 q1 if b1'''
    unsupported=resynthesize_measured_lookup(km.Circuit(cubic),clean_inputs=[4,5])
    assert unsupported['status']=='not_constructed'
    assert unsupported['segments'][0]['reason']=='no proved lower-cost segment'
    assert unsupported['segments'][0]['construction']['degree']==3
    assert unsupported['segments'][0]['construction']['products']==2
    assert unsupported['segments'][0]['construction']['shared_clean_output_holders']==1
    assert unsupported['segments'][0]['construction']['algebraic_verification']['passed']
    source=cubic+'''
CCX q0 q1 q4
CCX q0 q2 q4
CX q4 q6
HMR q4 b2
CZ q0 q1 if b2
CZ q0 q2 if b2'''
    result=resynthesize_measured_lookup(km.Circuit(source),clean_inputs=[4,5])
    assert result['status']=='constructed'
    assert result['segments'][0]['status']=='retained'
    assert result['segments'][0]['reason']=='no proved lower-cost segment'
    assert result['segments'][0]['construction']['status']=='constructed'
    assert result['candidate'].startswith(cubic)
    assert 'no universal lookup synthesis' in result['scope']
    _all_branches(source,result['candidate'],public=[0,1,2,3,6],width=7,bits=3,measurements=3)


def test_default_registry_retains_unused_public_width_and_rejects_phase_and_output_mutations(tmp_path):
    source=QUADRATIC+'\nX q10\nX q10'
    original=km.Circuit(source)
    proposed=resynthesize_measured_lookup(original,clean_inputs=[4])
    assert proposed['status']=='constructed'
    assert km.Circuit(proposed['candidate']).num_qubits==11
    registry=default_registry()
    corruptions={
        'lost_measurement_phase_repair':proposed['candidate'].replace('CZ q0 q1 if b0\n',''),
        'leaked_clean_workspace':proposed['candidate']+'\nX q4',
        'wrong_dirty_output':proposed['candidate']+'\nX q3',
    }
    for name,candidate in corruptions.items():
        registry.register(name,lambda circuit,ctx,text=candidate:dict(status='constructed',candidate=text),
                          'Intentionally corrupted independent-review candidate')
    path=tmp_path/'lookup.kmx'
    path.write_text(source)
    result=restructure_file(path,tmp_path/'out',registry=registry,clean_inputs=[4],rounds=1)
    attempts={row['mechanism']:row for row in result['attempts']}
    assert attempts['measured_lookup_resynthesis']['status']=='verified'
    assert all(attempts[name]['status']=='rejected' for name in corruptions)
    assert result['best']['resources']['Q']==11
    assert result['best']['resources']['toffoli']==1
    assert result['status']=='verified_improvement'


def test_unavailable_complete_semantics_never_emit_a_candidate():
    result=resynthesize_measured_lookup(km.Circuit(QUADRATIC),clean_inputs=[4],max_terms=1)
    assert result['status']=='unknown'
    assert result['reason']=='measured lookup needs exact complete boundary'
    assert 'candidate' not in result


def test_explicit_product_budget_controls_lookup_analysis_and_restructure_checks(tmp_path):
    original=km.Circuit(QUADRATIC)
    limited=resynthesize_measured_lookup(original,clean_inputs=[4],max_product_terms=1)
    # A decision receipt now proves the complete boundary under the separately
    # bounded fallback. Segment construction still obeys the ANF product limit.
    assert limited['status']=='not_constructed' and 'candidate' not in limited
    assert limited['semantics']['status']=='exact_factored'
    assert limited['semantics']['limits']['max_apply_steps']==32
    obstruction=limited['semantics']['anf_obstruction']
    assert obstruction['reason']=='product_budget'
    assert obstruction['limits']['max_product_terms']==1
    assert obstruction['instruction']=='CCX q0 q2 q4'
    assert limited['segments'][0]['status']=='retained'
    table=limited['full_boundary_table']
    assert table['status']=='exact' and table['table']==[0,0,0,1,0,1,0,0]
    assert table['proof']['dirty_target_independence'] and table['proof']['all_measurement_outcomes']
    assert table['native_replay']['status']=='passed'
    enough=resynthesize_measured_lookup(original,clean_inputs=[4],max_product_terms=32)
    assert enough['status']=='constructed'
    assert enough['check']['equivalent'] is True

    source=tmp_path/'lookup.kmx'
    source.write_text(QUADRATIC)
    blocked=restructure_file(source,tmp_path/'limited',clean_inputs=[4],rounds=1,max_product_terms=1)
    lookup=next(row for row in blocked['attempts'] if row['mechanism']=='measured_lookup_resynthesis')
    assert lookup['construction']['status']=='not_constructed' and 'candidate' not in lookup['construction']
    assert lookup['construction']['semantics']['status']=='exact_factored'
    assert lookup['construction']['semantics']['anf_obstruction']['reason']=='product_budget'
    assert lookup['construction']['semantics']['anf_obstruction']['limits']['max_product_terms']==1
    assert blocked['status']=='no_gain'
    assert blocked['budgets']['max_product_terms']==1
    passed=restructure_file(source,tmp_path/'sufficient',clean_inputs=[4],rounds=1,max_product_terms=32)
    lookup=next(row for row in passed['attempts'] if row['mechanism']=='measured_lookup_resynthesis')
    assert passed['status']=='verified_improvement'
    assert lookup['status']=='verified' and lookup['check']['equivalent'] is True
    assert passed['budgets']['max_product_terms']==32


def test_cli_restructure_exposes_product_budget_for_lookup_and_phase_constructors(tmp_path,capsys):
    from strata.cli import main
    source=tmp_path/'lookup.kmx'
    source.write_text(QUADRATIC)
    assert main(['restructure',str(source),'--out',str(tmp_path/'lookup'),
                 '--rounds','1','--clean-inputs','4','--max-product-terms','32'])==0
    report=json.loads(capsys.readouterr().out)
    assert report['budgets']['max_product_terms']==32
    assert report['status']=='verified_improvement'

    phase=tmp_path/'phase.kmx'
    phase.write_text('CCX q0 q1 q2\nZ q2\nCCX q0 q1 q2')
    assert main(['restructure',str(phase),'--out',str(tmp_path/'phase-limited'),
                 '--rounds','1','--clean-inputs','2','--max-product-terms','1'])==0
    report=json.loads(capsys.readouterr().out)
    proposal=next(row for row in report['attempts'] if row['mechanism']=='observable_phase_elimination')
    assert proposal['construction']['status']=='not_applicable'
    assert 'requires explicit ANF' in proposal['construction']['reason']
    boundary=report['boundary_receipts'][report['baseline']['id']]
    assert boundary['status']=='exact_factored' and boundary['phase_zero'] is False
    assert boundary['limits']['max_apply_steps']==32
    assert boundary['anf_obstruction']['reason']=='product_budget'
    assert boundary['anf_obstruction']['limits']['max_product_terms']==1
    assert report['budgets']['max_product_terms']==1
    assert report['status']=='no_gain'
    assert main(['restructure',str(phase),'--out',str(tmp_path/'phase-enough'),
                 '--rounds','1','--clean-inputs','2','--max-product-terms','32'])==0
    report=json.loads(capsys.readouterr().out)
    proposal=next(row for row in report['attempts'] if row['mechanism']=='observable_phase_elimination')
    assert proposal['status']=='verified'
    assert report['status']=='verified_improvement'


CUBIC_SHARED='''CCX q0 q1 q5
CCX q5 q2 q6
CCX q0 q1 q5
CCX q0 q1 q5
CCX q5 q3 q6
CCX q0 q1 q5
CX q6 q4
CX q6 q7
HMR q6 b0
CCZ q0 q1 q2 if b0
CCZ q0 q1 q3 if b0'''


def test_cubic_shared_pairs_and_linear_forms_save_products_with_existing_clean_workspace(tmp_path):
    original=km.Circuit(CUBIC_SHARED)
    result=resynthesize_measured_lookup(original,clean_inputs=[5,6])
    assert result['status']=='constructed'
    plan=result['segments'][0]['construction']
    assert plan['degree']==3 and plan['products']==3
    assert plan['pair_groups']==1 and plan['shared_linear_forms']==1
    assert plan['existing_clean_helper']==5 and plan['helper_restored']
    assert plan['algebraic_verification']['passed']
    assert plan['extra_qubits']==0
    assert km.Circuit(result['candidate']).num_qubits==original.num_qubits==8
    assert result['check']['equivalent'] is True
    _all_branches(CUBIC_SHARED,result['candidate'],public=[0,1,2,3,4,7],width=8,bits=1,measurements=1)
    source=tmp_path/'cubic.kmx';source.write_text(CUBIC_SHARED)
    report=restructure_file(source,tmp_path/'cubic-out',clean_inputs=[5,6],rounds=1)
    assert report['status']=='verified_improvement'
    # Native charged-product cost includes the two unchanged CCZ repairs.
    assert report['baseline']['resources']['toffoli']==8
    assert report['best']['resources']['toffoli']==5
    assert report['best']['resources']['Q']==8


def test_cubic_dependency_layers_preserve_original_dirty_target_values():
    source='CX q4 q7\n'+CUBIC_SHARED.replace('CX q6 q7\n','')
    source=source.replace('CX q6 q4\n','CCX q0 q1 q5\nCCX q5 q8 q6\nCCX q0 q1 q5\nCX q6 q4\n')
    source+='\nCCZ q0 q1 q8 if b0'
    result=resynthesize_measured_lookup(km.Circuit(source),clean_inputs=[5,6])
    assert result['status']=='constructed'
    plan=result['segments'][0]['construction']
    assert plan['dependency_layers']==[[6,7],[4]]
    assert plan['products']==6
    _all_branches(source,result['candidate'],public=[0,1,2,3,4,7,8],width=9,bits=1,measurements=1)


def test_cubic_no_clean_workspace_retains_original_with_priced_obstruction():
    source='''CCX q0 q1 q5
CCX q5 q2 q6
CCX q0 q1 q5
CCX q5 q2 q6
CX q6 q4
HMR q6 b0
CCZ q0 q1 q2 if b0'''
    result=resynthesize_measured_lookup(km.Circuit(source),clean_inputs=[6])
    assert result['status']=='not_constructed'
    plan=result['segments'][0]['construction']
    assert plan['status']=='not_emittable'
    assert plan['reason']=='cubic plan needs an available declared-zero helper'
    assert plan['products']==3 and plan['algebraic_verification']['passed']
    assert 'candidate' not in result


def test_cubic_operation_budget_reports_verified_upper_bound_without_materializing_large_candidate():
    from strata.lookup import _cubic_updates
    from strata.semantic import analyze_window
    block=CUBIC_SHARED.split('HMR')[0]
    sem=analyze_window(block,clean_inputs=[5,6])
    candidate,plan=_cubic_updates(sem,2,[5,6])
    assert candidate is None and plan['status']=='bounded'
    assert plan['instructions']>2 and plan['products']==3
    assert plan['algebraic_verification']['passed']
