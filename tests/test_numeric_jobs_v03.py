import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from strata.callbacks import PinnedCallback
from strata.model import StrataError
from strata.numeric_jobs import NumericJob,score_bets
from strata.store import Store


WORKER='''import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text());s=r['settings']
if s.get('fail_stage')==r['stage']:raise RuntimeError('intentional failed stage')
old=json.loads(Path(r['checkpoint']).read_text())['n'] if r['checkpoint'] else 0
if s.get('mutate'):Path(r['base_artifact']).write_text('changed')
Path('next.json').write_text(json.dumps({'n':old+1}))
print(json.dumps(dict(job='wrong' if s.get('wrong_scope') else r['job'],attempt=r['attempt'],stage=r['stage'],
 metrics={'loss':1/(old+1)},metric_kinds={'loss':s.get('kind','measured')},checkpoint_file='next.json')))
'''


@pytest.fixture
def worker_environment(tmp_path,monkeypatch):
    from strata import sandbox
    original=sandbox.run
    calls=[]
    def transport(argv,workdir,**limits):
        calls.append((argv,limits))
        if os.environ.get('STRATA_REAL_SANDBOX')=='1':return original(argv,workdir,**limits)
        done=subprocess.run(argv,cwd=workdir,capture_output=True,text=True,timeout=limits['timeout'])
        if done.returncode:raise StrataError(done.stderr)
        return dict(output=done.stdout,returncode=0,sandbox='test-only-real-worker-transport',limits=limits)
    monkeypatch.setattr(sandbox,'run',transport)
    worker=tmp_path/'worker.py';worker.write_text(WORKER)
    store=Store(tmp_path/'store')
    return store,worker,calls


def new_job(environment,**options):
    store,worker,_=environment
    return NumericJob.create(store,contract='numeric-fixture-v1',base_artifact=store.blob(b'base'),
        command=[sys.executable,str(worker),'{request}'],dependencies=[worker],metric_units={'loss':'dimensionless'},
        **dict(max_stages=3,max_wall_seconds=30,stage_timeout=5,**options))


def test_checkpoint_resume_stop_rules_and_immutable_policy(worker_environment):
    job=new_job(worker_environment,stop_rules=[dict(metric='loss',op='<=',value=.5,after_stages=2)])
    initial=job.policy;initial['max_stages']=100
    assert job.policy['max_stages']==3
    one=job.run()
    assert one['completed_stages']==1 and one['status']=='resumable'
    assert json.loads(job.store.read_blob(one['checkpoint']))=={'n':1}
    reopened=NumericJob(Store(job.store.path),job.id)
    stopped=reopened.run(stages=10)
    assert stopped['status']=='stopped_by_rule' and stopped['completed_stages']==2
    assert stopped['latest']['parent_checkpoint']==one['checkpoint']
    assert len(worker_environment[2])==2
    assert reopened.run()['completed_stages']==2
    assert new_job(worker_environment).id!=new_job(worker_environment).id


def test_failed_stage_keeps_last_complete_checkpoint_and_invalid_scope_is_not_evidence(worker_environment):
    job=new_job(worker_environment,settings={'fail_stage':2})
    previous=job.run()['checkpoint']
    with pytest.raises(StrataError,match='last complete checkpoint retained'):job.run()
    assert job.status()['checkpoint']==previous and job.status()['completed_stages']==1
    assert len(job._records('numeric_job_finish'))==2
    scoped=new_job(worker_environment,settings={'wrong_scope':True})
    with pytest.raises(StrataError,match='scope mismatch'):scoped.run()
    assert scoped.status()['completed_stages']==0


def test_model_metrics_do_not_trigger_measured_stop_rules(worker_environment):
    job=new_job(worker_environment,settings={'kind':'model'},stop_rules=[dict(metric='loss',op='<=',value=1)])
    result=job.run(stages=3)
    assert result['status']=='complete'
    assert result['latest']['metrics']['loss']['evidence_kind']=='model'
    assert all(not record['triggers'] for _,record in job._records('numeric_job_finish'))


def test_input_and_dependency_mutations_are_rejected(worker_environment):
    job=new_job(worker_environment,settings={'mutate':True})
    with pytest.raises(StrataError,match='immutable stage inputs'):job.run()
    assert not job._records('numeric_checkpoint')
    clean=new_job(worker_environment)
    worker_environment[1].write_text(WORKER+'\n# changed\n')
    with pytest.raises(StrataError,match='dependency changed'):clean.run()


def test_checkpoint_finish_transaction_and_unfinished_budget_charge(worker_environment,monkeypatch):
    import strata.numeric_jobs as jobs
    job=new_job(worker_environment)
    def fail_commit(*args):raise StrataError('simulated transaction failure')
    monkeypatch.setattr(jobs,'_put_atomic',fail_commit)
    with pytest.raises(StrataError,match='transaction failure'):job.run()
    assert not job._records('numeric_checkpoint')
    job.store.put('numeric_job_start',dict(job=job.id,attempt='crashed',stage=1,reserved_seconds=30))
    status=job.status()
    assert status['status']=='budget_exhausted' and status['unfinished_attempts']==1
    assert job.run()['status']=='budget_exhausted'


def test_another_process_cannot_run_the_same_job(worker_environment):
    job=new_job(worker_environment)
    code='from strata.numeric_jobs import NumericJob; from strata.store import Store; import sys; NumericJob(Store(sys.argv[1]),sys.argv[2]).run()'
    with job._lock():
        child=subprocess.run([sys.executable,'-c',code,str(job.store.path),job.id],capture_output=True,text=True)
    assert child.returncode!=0 and 'another process' in child.stderr
    assert not job._records('numeric_job_start')


PROMOTION_PROJECT=None


def pack_checkpoint(request):
    assert json.loads(request.checkpoint_bytes)['n']>=2
    base=PROMOTION_PROJECT.task('numeric').base_design()
    return PROMOTION_PROJECT.open_line(base,'co-design-numerics',['numbers'],implementation={'rank':2})


def test_predeclared_readiness_automatically_runs_independent_line_gates(worker_environment,tmp_path):
    global PROMOTION_PROJECT
    from test_task_issues import make_callback_project
    project_root=tmp_path/'project';project_root.mkdir()
    PROMOTION_PROJECT=make_callback_project(project_root)
    PROMOTION_PROJECT.init()
    packager=PinnedCallback(pack_checkpoint,dependencies=[PROMOTION_PROJECT.path])
    job=new_job(worker_environment,readiness_rules=[dict(metric='loss',op='<=',value=.5)],promotion=packager)
    assert job.run()['status']=='resumable'
    with pytest.raises(StrataError,match='same pinned'):NumericJob(job.store,job.id).run()
    result=NumericJob(job.store,job.id).run(promotion=packager)
    assert result['status']=='promoted' and result['completed_stages']==2
    promotion=result['promotion']
    assert promotion['gate_sequence']==['lab','build','full_verification']
    assert promotion['external_submission'] is False
    assert PROMOTION_PROJECT.replay(promotion['design'])['passed']
    assert job.run()['status']=='promoted'


def test_numeric_readiness_cannot_override_official_verifier_failure(worker_environment,tmp_path):
    global PROMOTION_PROJECT
    from test_task_issues import make_callback_project
    from test_runtime_v03 import reject_verify
    project_root=tmp_path/'project';project_root.mkdir()
    PROMOTION_PROJECT=make_callback_project(project_root,reject_verify)
    base=PROMOTION_PROJECT.init()['numeric']
    packager=PinnedCallback(pack_checkpoint,dependencies=[PROMOTION_PROJECT.path])
    job=new_job(worker_environment,readiness_rules=[dict(metric='loss',op='<=',value=.5)],promotion=packager)
    with pytest.raises(StrataError,match='independent certification failed'):job.run(stages=2)
    assert job.status()['completed_stages']==2
    assert json.loads(job.store.read_blob(job.status()['checkpoint']))=={'n':2}
    assert all(row['status']=='failed' for _,row in job._records('numeric_job_promotion'))
    assert PROMOTION_PROJECT.best()['tasks']['numeric']['design']==base
    assert len(PROMOTION_PROJECT.store.all('design'))==1


def test_portfolio_safe_fallback_labels_and_stable_geometric_ratio(tmp_path):
    result=score_bets({'a':10,'b':20},{'a':20,'b':40},[
        dict(id='good',costs={'a':2.5},probability=dict(value=.25,evidence_kind='judgment'),remaining_compute_seconds=10),
        dict(id='bad',costs={'a':40},probability=dict(value=1,evidence_kind='model'))])
    assert result['safe_score']==pytest.approx(2)
    assert result['bets'][0]['success_score']==pytest.approx(4)
    assert result['bets'][0]['expected_score']==pytest.approx(2.5)
    assert result['bets'][1]['expected_score']==pytest.approx(2)
    stable=score_bets({'a':1e-200,'b':1e200},{'a':1e200,'b':1e-200},[])
    assert stable['safe_score']==pytest.approx(1)
    with pytest.raises(StrataError,match='measured probability'):
        score_bets({'a':10},{'a':20},[dict(id='fake',costs={'a':5},probability=dict(value=.5,evidence_kind='measured'))])
    store=Store(tmp_path/'evidence');artifact=store.blob(b'observed Bernoulli experiment')
    result=score_bets({'a':10},{'a':20},[dict(id='observed',costs={'a':5},
        probability=dict(value=.5,evidence_kind='measured',artifact_hash=artifact))],store=store)
    assert result['bets'][0]['evidence_kind']=='measured'
