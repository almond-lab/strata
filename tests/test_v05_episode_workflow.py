import copy
import json

import kickmix as km
import pytest

from strata.cli import main
from strata.file_optimizer import optimize_circuit, verify_optimization
from strata.file_workflow import optimize_file
from strata.restructure import restructure_file
from strata.model import StrataError


def test_transported_zero_facts_used_consistently_and_replayed():
    circuit=km.Circuit('HMR q4 b0\nX q5\nX q5\n'
        'CCX q0 q1 q4\nCX q4 q2\nCCX q0 q1 q4\nX q5')
    candidate, report=optimize_circuit(circuit,max_operations=3)
    assert report['status']=='improved'
    row=next(r for r in report['regions'] if r['status']=='accepted')
    assert row['boundary_zero_evidence']['4']['kind']=='unconditional_hmr'
    assert verify_optimization(circuit,candidate,report,max_operations=3)['equivalent']
    bad=copy.deepcopy(report)
    next(r for r in bad['regions'] if r['status']=='accepted')['boundary_zero_evidence']={}
    assert not verify_optimization(circuit,candidate,bad,max_operations=3)['equivalent']


def test_clean_inputs_remain_windowed_and_promise_bound(tmp_path):
    path=tmp_path/'input.kmx'
    path.write_text('CCX q0 q1 q4\nCX q4 q2\nCCX q0 q1 q4\n'+'X q5\n'*50)
    result=optimize_file(path,tmp_path/'out',clean_inputs=(4,),max_window=3)
    assert result['status']=='improved' and result['replay']
    receipt=json.loads((tmp_path/'out'/'result.json').read_text())
    assert receipt['clean_inputs']==[4]
    assert not verify_optimization(km.Circuit.from_file(path),km.Circuit.from_file(result['candidate']),receipt,max_operations=3)['equivalent']


def test_restructure_range_preserves_fractional_outside_and_prices_parent(tmp_path):
    source=tmp_path/'input.kmx'
    source.write_text('Z_POW q0 0.125\nCCX q0 q1 q4\nCX q4 q2\nCCX q0 q1 q4\nX q5')
    result=restructure_file(source,tmp_path/'out',scan_range=(1,4),clean_inputs=(4,),max_operations=8)
    assert result['status']=='verified_improvement'
    candidate=km.Circuit.from_file(result['best']['path'])
    assert str(candidate).splitlines()[0]==str(km.Circuit.from_file(source)).splitlines()[0]
    assert candidate.num_qubits==6 and result['contextual_proof']['width_preserved']
    assert result['best']['resources']['toffoli']==1
    assert '4' in {str(k):v for k,v in result['episode']['boundary_zero_evidence'].items()}


def test_interior_promise_is_not_reapplied_after_write(tmp_path):
    source=tmp_path/'input.kmx'
    source.write_text('X q4\nCCX q0 q1 q4\nCX q4 q2\nCCX q0 q1 q4\nX q5')
    result=restructure_file(source,tmp_path/'out',scan_range=(1,4),clean_inputs=(4,),max_operations=8)
    assert 4 not in result['episode']['boundary_zero_evidence']
    candidate=km.Circuit.from_file(result['best']['path'])
    # Dirty temporary's contribution survives any accepted replacement.
    if result['status']=='verified_improvement':
        assert 'CX q4 q2' in str(candidate)


def test_range_must_close_classical_scope(tmp_path):
    source=tmp_path/'input.kmx'
    source.write_text('PUSH_CONDITION if b0\nCCX q0 q1 q2\nPOP_CONDITION')
    with pytest.raises(StrataError,match='inside a classical condition'):
        restructure_file(source,tmp_path/'out',scan_range=(1,2))


def test_explicit_parent_width_accepts_untouched_helpers(tmp_path):
    source=tmp_path/'input.kmx';source.write_text('CCX q0 q1 q2')
    result=restructure_file(source,tmp_path/'out',width=8,clean_inputs=(5,6,7))
    assert result['baseline']['resources']['Q']==8
    with pytest.raises(StrataError,match='width'):
        restructure_file(source,tmp_path/'bad',width=2)


def test_large_file_automatically_searches_a_complete_episode(tmp_path):
    source=tmp_path/'input.kmx'
    source.write_text('Z_POW q0 0.125\n'+'X q5\n'*40+
        'CCX q0 q1 q4\nCX q4 q2\nCCX q0 q1 q4\n'+'X q5\n'*40)
    result=restructure_file(source,tmp_path/'out',max_operations=8,max_candidates=5)
    assert result['status']=='verified_improvement'
    assert result['cost_delta']<0
    assert result['attempts']


def test_expected_accounting_cli_and_no_default_external_distribution(tmp_path,capsys):
    path=tmp_path/'input.kmx'
    path.write_text('HMR q3 b0\nCCX q0 q1 q2 if b0')
    assert main(['evaluate-file',str(path),str(path),'--out',str(tmp_path/'out'),
                 '--accounting','expected'])==0
    report=json.loads(capsys.readouterr().out)
    assert report['baseline']['resources']['toffoli']==0.5
    assert report['objective']['accounting']=='expected_executed'
    path.write_text('CCX q0 q1 q2 if b0')
    with pytest.raises(StrataError,match='expected accounting unavailable'):
        restructure_file(path,tmp_path/'unknown',accounting='expected')
