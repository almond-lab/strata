"""Actual CLI processes exercise v0.3 evidence boundaries and retained reports."""
import json
import os
from pathlib import Path
import subprocess
import sys

import kickmix as km
import pytest

from strata.callbacks import ExternalCheckError
from test_task_issues import make_callback_project
from test_runtime_v03 import reject_lab, always_wrong_width_measure


def cli(*arguments, expected=0):
    source=Path(__file__).resolve().parents[1]/'src'
    process=subprocess.run([sys.executable,'-m','strata.cli',*[str(a) for a in arguments]],
        env={**os.environ,'PYTHONPATH':str(source)},capture_output=True,text=True,timeout=30)
    assert process.returncode==expected,(process.stdout,process.stderr)
    return json.loads(process.stdout),process.stderr


def test_cli_finite_evaluation_and_floor_bind_the_same_domain(tmp_path):
    before=tmp_path/'before.kmx';before.write_text('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    after=tmp_path/'after.kmx';after.write_text('CCX q0 q1 q3')
    domain=tmp_path/'domain.json'
    domain.write_text(json.dumps(dict(version=1,type='finite',registers={'a':[0],'b':[1],'scratch':[2],'out':[3]},
        assignments=[dict(a=a,b=b,scratch=0,out=z) for a in (0,1) for b in (0,1) for z in (0,1)])))
    arbitrary,_=cli('evaluate-file',before,after,'--out',tmp_path/'arbitrary')
    assert arbitrary['status']=='rejected'
    finite,_=cli('evaluate-file',before,after,'--out',tmp_path/'finite','--domain',domain)
    assert finite['status']=='verified_domain_improvement'
    assert finite['checks']['equivalent'] and not finite['checks']['all_input_equivalence']
    assert finite['checks']['assignment_count']==8 and finite['accepted_gain']==4
    floor,_=cli('floors',before,'--domain',domain)
    assert floor['domain_hash']==finite['checks']['domain_hash']
    assert floor['value']==1 and 'finite domain only' in floor['scope']
    after.write_text('CCX q0 q1 q3\nZ q0')
    phase,_=cli('evaluate-file',before,after,'--out',tmp_path/'phase','--domain',domain)
    assert phase['status']=='rejected' and phase['accepted_gain'] is None
    witness=phase['counterexample']
    assert witness['before']['outputs']==witness['after']['outputs']
    assert witness['before']['phase']!=witness['after']['phase']


def test_cli_interior_queries_ignore_opaque_prefix_and_untouched_width(tmp_path):
    source=tmp_path/'interior.kmx';source.write_text('Z_POW q99 0.25\nCCX q0 q1 q2')
    common=[source,'--from','1','--at','2','--clean-inputs','2','--max-operations','1']
    slack,_=cli('slack',*common)
    assert slack['wires']==[0,1,2] and slack['slack_bits']==1
    assert slack['from_at']==1 and slack['at']==2
    floor,_=cli('floors',*common)
    assert floor['status']=='exact' and floor['value']==1
    assert floor['from_at']==1 and floor['at']==2
    limited,_=cli('floors',source,'--from','1','--at','2','--max-variables','1')
    assert limited['status']=='unknown' and 'value' not in limited


def test_cli_clean_inspection_and_discovery_retain_reset_assumption(tmp_path):
    source=tmp_path/'reset.kmx'
    source.write_text('R q2\nCCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    unknown,_=cli('floors',source)
    assert unknown['status']=='unknown' and unknown['reason']=='reset_not_proven_zero'
    inspected,_=cli('inspect-file',source,'--out',tmp_path/'inspect','--clean-inputs','2')
    assert inspected['coverage']['exact_fraction']==1
    lift=json.loads(Path(inspected['semantic_lift']).read_text())
    assert lift['regions'][0]['semantics']['clean_input_assumptions']==[2]
    brief,_=cli('discover',source,'--out',tmp_path/'discover','--clean-inputs','2')
    assert brief['targets']
    expanded=json.loads(Path(brief['targets'][0]['expand']['path']).read_text())
    assert expanded['region']['semantics']['clean_input_assumptions']==[2]
    assert brief['official_verifier_run'] is False
    error,_=cli('discover',source,'--out',tmp_path/'bad-range','--clean-inputs','2','--range','0:4',expected=2)
    assert 'range scan treats boundaries as arbitrary' in error['error']


def test_cli_tensor_floor_keeps_general_and_and_bilinear_models_separate(tmp_path):
    source=tmp_path/'multiply.kmb'
    km.Circuit('CCX q0 q2 q4\nCCX q1 q3 q4\nCCX q0 q3 q5\nCCX q1 q2 q5\nCCX q1 q3 q5').to_file(source,format='kmb')
    result,_=cli('floors',source)
    assert result['resource']=='and_count' and result['value']==2
    tensor=result['bilinear_bounds']
    assert tensor['shape']==[2,2,2] and tensor['resource']=='bilinear_products'
    assert tensor['flattening_ranks']==dict(left=2,right=2,output=2)
    assert tensor['substitution_bound']['value']==3 and tensor['value']==3
    assert 'not a general AND' in tensor['scope']


@pytest.mark.parametrize('command',['inspect-file','discover'])
def test_cli_binary_range_groups_relabelled_kernels_and_emits_progress(tmp_path,command):
    source=tmp_path/'repeated.kmb'
    km.Circuit('X q99\nCCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2\nINFO_TAG "cut"\nCCX q4 q5 q6\nCX q6 q7\nCCX q4 q5 q6\nX q99').to_file(source,format='kmb')
    result,progress=cli(command,source,'--out',tmp_path/command,'--range','1:8','--growth-window','32')
    scan=result['scan'] if command=='discover' else result
    assert scan['range']==dict(start=1,stop=8) and scan['scanned_operations']==7
    group=next(group for group in scan['groups'] if group['operations_per_occurrence']==3)
    assert group['occurrences']==2 and group['aggregate_static_toffoli']==4
    assert [(row['start'],row['stop']) for row in group['examples']]==[(1,4),(5,8)]
    assert group['rewrite_boundary_verified'] is False
    assert json.loads(progress.splitlines()[-1])['stage']=='scan_complete'
    assert Path(scan['index']).is_file()


def test_cli_selected_condition_body_never_becomes_an_unconditional_proof(tmp_path):
    source=tmp_path/'condition.kmx'
    source.write_text('PUSH_CONDITION if b0\nCCX q0 q1 q2\nCX q2 q3\nPOP_CONDITION')
    result,_=cli('discover',source,'--out',tmp_path/'out','--range','1:3','--quiet-progress')
    assert result['scan']['scanned_operations']==2
    assert all(group['semantics']['status']=='unknown' for group in result['targets'])


def test_cli_report_preserves_failed_line_receipt_after_sources_are_removed(tmp_path):
    project=make_callback_project(tmp_path,reject_lab)
    base=project.init()['numeric']
    line=project.open_line(base,'co-design-numerics',['numbers'],implementation={'rank':2})
    with pytest.raises(ExternalCheckError):line.lab()
    project.path.unlink()
    (project.root/'base'/'numbers.json').unlink()
    result,_=cli('report','--store',project.store.path,'--handle',line.id)
    record=next(row for row in result['lines'] if row['handle']==line.id)
    assert record['status']!='not_run'
    event=record['events'][-1]
    assert event['evidence_kind']=='measured_rejection'
    assert event['receipt']['details']['condition']=='missing correction'
    assert event['receipt_artifact']
    assert result['negative_memory'][0]['evidence_kind']=='build'


def test_cli_report_retains_ingest_resource_rejection_without_any_design(tmp_path):
    project=make_callback_project(tmp_path,always_wrong_width_measure)
    with pytest.raises(ExternalCheckError):project.init()
    failure=project.store.all('external_check_failure')[-1][1]
    project.path.unlink()
    (project.root/'base'/'numbers.json').unlink()
    result,_=cli('report','--store',project.store.path)
    text=json.dumps(result)
    assert failure['receipt_artifact'] in text
    assert 'postcheck_validation_failed' in text and '889' in text
    assert result['counts']['designs']==0
