import copy
import hashlib
import json
from pathlib import Path
import kickmix as km
import pytest
from strata.file_workflow import inspect_file, optimize_file, contract_tree
from strata.region_evidence import record_region_evidence, replay_region_evidence
from strata.semantic import lift_regions
from strata.model import StrataError, digest
from strata.store import Store


def test_source_less_regions_have_outcome_parametric_symbolic_measurement_evidence(tmp_path):
    source=tmp_path/'supplied.kmx'
    source.write_text('CCX q0 q1 q2\nZ q2\nHMR q2 b0\nX q3')
    result=inspect_file(source,tmp_path/'inspect')
    report=json.loads(Path(result['region_evidence']).read_text())
    assert report['summary']=={'symbolic':1,'unknown':0}
    exact=[r for r in report['regions'] if r['evidence_kind']=='symbolic'][0]
    assert exact['status']=='exact' and exact['scope']['clean_input_assumptions']==[]
    assert exact['scope']['whole_circuit_certificate'] is False
    assert exact['symbolic_function']['inputs']==[0,1,2,3]
    assert exact['symbolic_function']['outputs']['2']['monomials']==[]
    assert exact['symbolic_function']['phase']['monomials']==[[2],[0,1],[2,'m0'],[0,1,'m0']]
    assert exact['symbolic_function']['classical_outputs']['b0']['monomials']==[['m0']]
    assert exact['scope']['quantum_channel_equivalence'] is False
    assert exact['binding']['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert exact['binding']['contract_hash']==digest(exact['contract'])
    assert exact['effects']['phase_debts'][0]['status']=='outstanding'
    source.unlink()  # Replay uses retained bytes, not an available generator/source path.
    replay=replay_region_evidence(tmp_path/'inspect'/'.strata',result['region_evidence_record'])
    assert replay['replayed'] and replay['summary']['unknown']==0
    assert not replay['official_verifier_run'] and not replay['whole_circuit_certificate']


def test_region_receipts_reuse_existing_lift_without_rechecking(tmp_path,monkeypatch):
    circuit=km.Circuit('CX q0 q1\nCZ q0 q1')
    source=tmp_path/'source.kmx';source.write_text(str(circuit))
    lifted=lift_regions(circuit)
    tree=contract_tree(circuit,lifted)
    def unexpected(*args,**kwargs):raise AssertionError('cached lift should not be re-executed while recording')
    monkeypatch.setattr('strata.semantic.analyze_window',unexpected)
    report=record_region_evidence(source,circuit,tree,lifted,Store(tmp_path/'store'))
    assert report['summary']['symbolic']==1


def test_symbolic_replay_rejects_forged_phase_boundary_and_scope(tmp_path):
    source=tmp_path/'source.kmx';source.write_text('CX q0 q1\nZ q1')
    result=inspect_file(source,tmp_path/'inspect')
    store=Store(tmp_path/'inspect'/'.strata')
    original=store.get(result['region_evidence_record'],'file_region_evidence')
    for change in ('phase','boundary','scope','source'):
        forged=copy.deepcopy(original)
        region=forged['regions'][0]
        if change=='phase':
            region['symbolic_function']['phase']['monomials']=[]
            region['function_hash']=digest(region['symbolic_function'])
        elif change=='boundary':
            region['contract']['registers'][0]['wires']=[9]
            region['binding']['contract_hash']=digest(region['contract'])
        elif change=='scope':
            region['scope']['clean_input_assumptions']=['q0=0']
        else:
            forged['source']['blob']=store.blob(b'CX q0 q1\nX q1')
            forged['source']['sha256']=forged['source']['blob']
        handle=store.put('file_region_evidence',forged)
        with pytest.raises(StrataError,match='replay|source bytes'):
            replay_region_evidence(store,handle)


def test_optimizer_links_source_region_evidence_without_inventing_scale(tmp_path):
    source=tmp_path/'source.kmx';source.write_text('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2\nX q4')
    result=optimize_file(source,tmp_path/'optimized')
    store=Store(tmp_path/'optimized'/'.strata')
    certificate=store.get(result['equivalence_certificate'],'file_equivalence')
    assert certificate['region_evidence']==result['region_evidence_record']
    report=store.get(result['region_evidence_record'],'file_region_evidence')
    assert report['artifact']['sha256']==result['baseline']['sha256']
    assert all(r['evidence_kind']=='symbolic' for r in report['regions'])
    assert all('scale' not in r for r in report['regions'])
    assert replay_region_evidence(store,result['region_evidence_record'])['replayed']


def test_conditional_context_keeps_classical_state_and_raw_frontend_replays(tmp_path):
    from strata.frontends import OperationStream
    circuit=km.Circuit('BIT_STORE1 b0\nX q0 if b0\nZ q1')
    source=tmp_path/'source.ops';source.write_text(OperationStream().serialize(circuit))
    result=inspect_file(source,tmp_path/'inspect')
    report=json.loads(Path(result['region_evidence']).read_text())
    controlled=report['regions'][0]
    assert controlled['evidence_kind']=='symbolic'
    assert controlled['symbolic_function']['outputs']['0']['monomials']==[[],[0]]
    replay=replay_region_evidence(tmp_path/'inspect'/'.strata',result['region_evidence_record'])
    assert replay['replayed'] and replay['summary']['symbolic']==1


def test_inherited_conditional_scope_cannot_be_recast_as_unconditional_evidence(tmp_path):
    source=tmp_path/'source.kmx'
    source.write_text('PUSH_CONDITION if b0\nX q0\nPOP_CONDITION\nZ q1')
    result=inspect_file(source,tmp_path/'inspect')
    report=json.loads(Path(result['region_evidence']).read_text())
    body=report['regions'][0]
    assert body['evidence_kind']=='symbolic'
    assert body['symbolic_function']['outputs']['0']['monomials']==[[0],['b0']]
    assert replay_region_evidence(tmp_path/'inspect'/'.strata',report)['summary']['unknown']==0


def test_explicit_clean_artifact_scope_is_hash_bound_and_replayed(tmp_path):
    circuit=km.Circuit('R q2\nCCX q0 q1 q2\nHMR q2 b0\nCZ q0 q1 if b0')
    source=tmp_path/'clean.kmx';source.write_text(str(circuit))
    lifted=lift_regions(circuit,clean_inputs=[2])
    store=Store(tmp_path/'store')
    report=record_region_evidence(source,circuit,contract_tree(circuit,lifted),lifted,store)
    assert report['limits']['clean_inputs']==[2]
    receipt=report['regions'][0]
    assert receipt['scope']['clean_input_assumptions']==[2]
    assert receipt['symbolic_function']['clean_input_assumptions']==[2]
    assert receipt['effects']['phase_debts'][0]['status']=='discharged'
    assert replay_region_evidence(store,report)['replayed']
    forged=copy.deepcopy(report)
    forged['limits']['clean_inputs']=[]
    with pytest.raises(StrataError,match='contract tree mismatch'):
        replay_region_evidence(store,forged)
