"""Adversarial construction and reusable-template acceptance boundaries."""
import itertools
import json
from pathlib import Path

import kickmix as km
import numpy as np

from strata.restructure import ConstructorRegistry, restructure_file
from strata.streaming import scan_circuit


def save(tmp_path, text, name='source.kmx'):
    path=tmp_path/name
    path.write_text(text)
    return path


def native_outputs(circuit, width, clean=()):
    inputs=[q for q in range(width) if q not in clean]
    rows=list(itertools.product((0,1),repeat=len(inputs)))
    sim=km.Simulator(batch_size=len(rows),seed=921)
    for i,q in enumerate(inputs):
        sim.write_across_shots(km.q(q),np.array([row[i] for row in rows],dtype=bool))
    sim.do(circuit)
    return [sim.read_across_shots(km.q(q)).tolist() for q in range(width)],sim.read_phase_flipped_across_shots().tolist()


def test_default_constructor_phase_elimination_replays_native_all_inputs(tmp_path):
    source=save(tmp_path,'CCX q0 q1 q3\nZ q3\nCCX q0 q1 q3')
    result=restructure_file(source,tmp_path/'out',clean_inputs=[3],rounds=1)
    assert result['status']=='verified_improvement'
    before=km.Circuit.from_file(source)
    after=km.Circuit.from_file(result['best']['path'])
    assert after.num_qubits==3 and result['best']['resources']['toffoli']==0
    assert native_outputs(before,4,clean=[3])==native_outputs(after,4,clean=[3])


def test_wrong_phase_and_public_wire_deletion_are_retained_as_rejected_obligations(tmp_path):
    source=save(tmp_path,'CCX q0 q1 q2\nZ q2\nCCX q0 q1 q2')
    registry=ConstructorRegistry()
    registry.register('wrong_phase',lambda circuit,ctx:dict(status='constructed',candidate='X q2\nX q2'),'Intentionally wrong phase')
    registry.register('drop_public_wire',lambda circuit,ctx:dict(status='constructed',candidate='CZ q0 q1'),'Unauthorized deletion of public q2')
    result=restructure_file(source,tmp_path/'out',registry=registry,rounds=1)
    outcomes={row['mechanism']:row for row in result['attempts']}
    assert outcomes['wrong_phase']['status']=='rejected'
    assert outcomes['wrong_phase']['check']['counterexample']['kind']=='phase'
    assert outcomes['drop_public_wire']['status']=='unknown'
    assert outcomes['drop_public_wire']['check']['reason']=='undeclared_or_expanded_boundary_layout'
    assert result['status']=='no_gain' and len(result['states'])==1


def test_nonwinning_verified_bridge_can_enable_later_improvement(tmp_path):
    source=save(tmp_path,'X q0\nX q0\nX q1\nX q1')
    registry=ConstructorRegistry()
    def bridge(circuit,ctx):
        if len(circuit)!=4:
            return dict(status='not_applicable',reason='bridge already visited')
        return dict(status='constructed',candidate=str(circuit)+'\nX q0\nX q0')
    def finish(circuit,ctx):
        if len(circuit)!=6:
            return dict(status='not_applicable',reason='requires alternate representation')
        return dict(status='constructed',candidate='APPEND_TO_REGISTER q1 r0')
    registry.register('bridge',bridge,'Explore a nonwinning identity representation')
    registry.register('finish',finish,'Remove the now-exposed identity episode')
    result=restructure_file(source,tmp_path/'out',registry=registry,cost='instructions',rounds=2)
    assert result['status']=='verified_improvement'
    assert any(row['status']=='verified' and row['cost_delta']>0 for row in result['attempts'])
    assert len(result['states'])==3
    assert result['best']['cost']==0


def test_measurement_reordering_cannot_be_accepted_by_matching_outcome_count(tmp_path):
    source=save(tmp_path,'HMR q0 b0\nHMR q1 b1')
    registry=ConstructorRegistry()
    registry.register('reorder',lambda circuit,ctx:dict(status='constructed',candidate='HMR q1 b1\nHMR q0 b0'),'Change event provenance')
    result=restructure_file(source,tmp_path/'out',registry=registry,rounds=1)
    attempt=result['attempts'][0]
    assert attempt['status']=='unknown'
    assert attempt['check']['reason']=='measurement_event_alignment_unproved'
    assert len(result['states'])==1


def test_learned_template_replays_sparse_wire_binding_and_checks_public_layout(tmp_path):
    source=save(tmp_path,'CCX q0 q1 q2\nZ q2\nCCX q0 q1 q2')
    archive=tmp_path/'mechanisms.json'
    first=restructure_file(source,tmp_path/'learn',clean_inputs=[2],archive=archive,rounds=1)
    assert first['learned_template_count']>=1
    renamed=save(tmp_path,'CCX q3 q5 q7\nZ q7\nCCX q3 q5 q7',name='renamed.kmx')
    # An empty host registry isolates learned constructors from built-in ones.
    second=restructure_file(renamed,tmp_path/'reuse',clean_inputs=[7],archive=archive,
                            registry=ConstructorRegistry(),rounds=1)
    assert second['status']=='verified_improvement'
    assert second['best']['resources']['Q']==7
    assert any(row['mechanism'].startswith('learned_') and row['status']=='verified' for row in second['attempts'])
    before,after=km.Circuit.from_file(renamed),km.Circuit.from_file(second['best']['path'])
    assert native_outputs(before,8,clean=[7])==native_outputs(after,8,clean=[7])


def test_poisoned_learned_replacement_is_independently_rejected(tmp_path):
    source=save(tmp_path,'CCX q0 q1 q2\nZ q2\nCCX q0 q1 q2')
    archive=tmp_path/'mechanisms.json'
    restructure_file(source,tmp_path/'learn',clean_inputs=[2],archive=archive,rounds=1)
    memory=json.loads(archive.read_text())
    for template in memory['templates']:
        template['replacement']='X q0'
    archive.write_text(json.dumps(memory))
    result=restructure_file(source,tmp_path/'poisoned',clean_inputs=[2],archive=archive,
                            registry=ConstructorRegistry(),rounds=1)
    assert result['status']=='no_gain'
    assert result['attempts'] and all(row['status']=='rejected' for row in result['attempts'])
    assert all('counterexample' in row['check'] for row in result['attempts'])


def test_streamed_scope_cuts_and_classical_boundary_contracts(tmp_path):
    circuit=km.Circuit('PUSH_CONDITION if b0\nBIT_STORE0 b0\nX q0\nPOP_CONDITION')
    result=scan_circuit(circuit,tmp_path/'scoped',max_window=2)
    assert all(group['semantics']['status']!='exact' for group in result['groups'])
    assert all(group['rewrite_boundary_verified'] is False for group in result['groups'])
    sliced=scan_circuit(circuit,tmp_path/'sliced',start=1,stop=3,max_window=4)
    assert all(group['semantics']['status']!='exact' for group in sliced['groups'])
    bits=scan_circuit(km.Circuit('BIT_STORE0 b0\nX q0 if b0'),tmp_path/'bits',max_window=1)
    controlled=next(g for g in bits['groups'] if '0' in g['semantics'].get('outputs',{}))
    assert controlled['semantics']['outputs']['0']['monomials']==[[0],['b0']]
    assert controlled['semantics']['classical_inputs']==['b0']
    assert controlled['rewrite_boundary_verified'] is False
