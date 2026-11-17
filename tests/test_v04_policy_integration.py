import pytest

from strata.store import Store
from test_v04_policy import calibrated_screen, calibration_trajectories, continuation


@pytest.fixture
def policy_command_transport(monkeypatch):
    """Real short subprocesses; only namespace setup is replaced in this test."""
    import subprocess
    from strata import sandbox
    launch = subprocess.Popen
    monkeypatch.setattr(sandbox.shutil, 'which', lambda _: '/test/bwrap')
    def transport(command, **kwargs):
        assert '--unshare-all' in command
        kwargs['cwd'] = command[command.index('--chdir') + 1]
        return launch(command[command.index('--') + 1:], **kwargs)
    monkeypatch.setattr(sandbox.subprocess, 'Popen', transport)


def pipeline_manifest(tmp_path, values):
    import json
    import sys
    script = tmp_path / 'worker.py'
    script.write_text('import json,sys\nfrom pathlib import Path\n'
        'Path("result.json").write_text(json.dumps({"loss":float(sys.argv[1])}))\n'
        'Path("checkpoint.json").write_text(json.dumps({"stage":int(sys.argv[2])}))\n')
    output = dict(name='result',path='result.json',type='json')
    spec = dict(version=1,name='policy-fixture', contract='generic numeric loss',dependencies={'worker':str(script)},
        outputs=[output], stages=[dict(name='stage'+str(i), argv=[sys.executable,'{dependency:worker}',str(v),str(i)],
            outputs=[output],checkpoint={'path':'checkpoint.json','type':'json'}) for i,v in enumerate(values, 1)],
        metrics={'loss':dict(output='result',selector='loss',unit='error',evidence_kind='measured')},
        limits=dict(timeout=5,memory_bytes=268435456,max_file_bytes=1048576),
        resources=dict(threads=1,memory_bytes=268435456,storage_bytes=1048576), max_wall_seconds=30,max_attempts=3)
    path = tmp_path / 'pipeline.json'; path.write_text(json.dumps(spec))
    return path, spec


def test_real_cheap_stage_prunes_then_explicit_override_runs_full(tmp_path, policy_command_transport):
    import json
    from strata.experiments import evaluator_identity, create_experiment, run_experiment
    from strata.experiment_reports import experiment_report
    path, spec = pipeline_manifest(tmp_path, [2, .5]); store = Store(tmp_path / 'store')
    identity = evaluator_identity(path, metric='loss')
    policy, calibration, _, _ = calibrated_screen(store, identity=identity, audit_fraction=.000001)
    spec['screening'] = dict(policy=policy.id, calibration=calibration['record'],identity=identity,
                             metric='loss',after_stage='stage1')
    path.write_text(json.dumps(spec)); exp = create_experiment(store,path)
    first = run_experiment(store,exp)
    assert first['status'] == 'screened_out', first
    assert len(store.all('experiment_stage')) == 1
    assert run_experiment(store,exp)['status'] == 'screened_out'
    second = run_experiment(store,exp,override_policy=True)
    assert second['status'] == 'artifact_validated', second
    assert len(store.all('experiment_stage')) == 2
    assert experiment_report(store,experiment=exp)['screening']['decisions'][0]['action'] == 'prune'


def test_real_full_evaluator_audit_retains_missed_success(tmp_path, policy_command_transport):
    import json
    from strata.experiments import evaluator_identity, create_experiment, run_experiment
    path, spec = pipeline_manifest(tmp_path, [2,.5]); store = Store(tmp_path / 'store')
    identity = evaluator_identity(path,metric='loss')
    policy, calibration, _, _ = calibrated_screen(store,identity=identity,audit_fraction=1)
    spec['screening'] = dict(policy=policy.id,calibration=calibration['record'],identity=identity,
                             metric='loss',after_stage='stage1')
    path.write_text(json.dumps(spec)); exp = create_experiment(store,path)
    result = run_experiment(store,exp)
    assert result['status'] == 'artifact_validated', result
    audits = store.all('screening_audit')
    assert len(audits) == 1 and audits[0][1]['false_negative']


def test_real_forecast_pause_resume_keeps_checkpoint_and_compute(tmp_path, policy_command_transport):
    import json
    from strata.experiments import evaluator_identity,create_experiment,run_experiment,experiment_status
    path,spec = pipeline_manifest(tmp_path,[10]*5); store = Store(tmp_path / 'store')
    identity = evaluator_identity(path,metric='loss')
    train,test = calibration_trajectories(store,identity=identity)
    policy = continuation(store,identity=identity,automation=True)
    calibration = policy.calibrate(train,test)
    spec['continuation'] = dict(policy=policy.id,calibration=calibration['record'],identity=identity,metric='loss')
    path.write_text(json.dumps(spec)); exp = create_experiment(store,path)
    first = run_experiment(store,exp)
    assert first['status'] == 'paused_by_forecast_policy', first
    assert len(store.all('experiment_stage')) == 3 and first['checkpoint']
    spent = experiment_status(store,exp)['spent_wall_seconds']
    assert spent > 0
    assert run_experiment(store,exp)['status'] == 'paused_by_forecast_policy'
    second = run_experiment(store,exp,override_policy=True)
    assert second['status'] == 'artifact_validated', second
    assert len(store.all('experiment_stage')) == 5
    assert experiment_status(store,exp)['spent_wall_seconds'] > spent
    assert store.all('experiment_attempt')[-1][1]['policy_override']


def test_actual_dependency_identity_drift_abstains_despite_matching_declared_strings(tmp_path, policy_command_transport):
    import json
    from pathlib import Path
    from strata.experiments import evaluator_identity,create_experiment,run_experiment
    path,spec = pipeline_manifest(tmp_path,[2,.5]); store = Store(tmp_path / 'store')
    identity = evaluator_identity(path,metric='loss')
    policy,calibration,_,_ = calibrated_screen(store,identity=identity,audit_fraction=.000001)
    spec['screening'] = dict(policy=policy.id,calibration=calibration['record'],identity=identity,
                             metric='loss',after_stage='stage1')
    worker = Path(spec['dependencies']['worker'])
    worker.write_text(worker.read_text()+'\n# evaluator source changed after calibration\n')
    path.write_text(json.dumps(spec))
    assert evaluator_identity(path,metric='loss') != identity
    exp = create_experiment(store,path)
    result = run_experiment(store,exp)
    assert result['status'] == 'artifact_validated', result
    decision = next(row for _,row in store.all('screening_decision'))
    assert decision['action'] == 'abstain' and 'identity' in decision['reason']
    assert len(store.all('experiment_stage')) == 2


def test_agent_packet_decision_real_run_and_updated_packet(tmp_path, policy_command_transport):
    from strata.experiments import create_experiment,run_experiment
    from strata.experiment_reports import agent_packet,record_agent_decision
    path,_ = pipeline_manifest(tmp_path,[2,.5]); store = Store(tmp_path / 'store')
    exp = create_experiment(store,path)
    first = agent_packet(store)
    decision = record_agent_decision(store,packet=first['record'],hypothesis='full stage reduces loss',
        decisive_experiment=exp,expected_outcome='measured loss below 1',budget={'seconds':30},
        reopening_condition='different evaluator version or optimization settings')
    outcome = run_experiment(store,exp)
    assert outcome['status'] == 'artifact_validated'
    second = agent_packet(store)
    assert not first['tried'] and second['tried'][0]['experiment'] == exp
    assert second['prior_agent_decisions'][0]['record'] == decision['record']
    assert second['completed_comparisons'][0]['metrics']['loss']['value'] == .5
    assert second['best_independently_accepted'] is None


def test_cli_derives_actual_evaluator_identity_and_accepts_retained_handle(tmp_path,capsys):
    import json
    from pathlib import Path
    from strata.cli import main
    from strata.experiments import evaluator_identity
    path,spec = pipeline_manifest(tmp_path,[2,.5])
    request = tmp_path/'identity-request.json'
    request.write_text(json.dumps(dict(operation='evaluator_identity',manifest=str(path),metric='loss')))
    command = ['experiment-policy',str(request),'--store',str(tmp_path/'store')]
    assert main(command)==0
    answer = json.loads(capsys.readouterr().out)
    assert answer['identity'] == evaluator_identity(path,metric='loss')
    request.write_text(json.dumps(dict(operation='create_screening',identity=answer['record'],
        proxy_threshold=1,full_threshold=1)))
    assert main(command)==0
    policy = json.loads(capsys.readouterr().out)
    assert policy['identity'] == answer['identity']
    worker = Path(spec['dependencies']['worker'])
    worker.write_text(worker.read_text()+'\n# changed evaluator source\n')
    request.write_text(json.dumps(dict(operation='evaluator_identity',manifest=str(path),metric='loss')))
    assert main(command)==0
    assert json.loads(capsys.readouterr().out)['identity'] != answer['identity']
