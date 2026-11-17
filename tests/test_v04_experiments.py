"""Executable fixtures with a test-only transport; production always uses bwrap."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from strata import sandbox
from strata.experiments import (create_experiment, create_sweep, experiment_status,
                                load_manifest, run_experiment, run_sweep, sweep_status)
from strata.model import StrataError
from strata.store import Store

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples' / 'workflow'


@pytest.fixture
def local_transport(monkeypatch):
    """Exercise real argv, supervision and scripts without claiming namespace coverage."""
    original_popen = subprocess.Popen
    original_which = shutil.which
    monkeypatch.setattr(sandbox.shutil, 'which', lambda value: '/test/bwrap' if value == 'bwrap' else original_which(value))

    def launch(command, **kwargs):
        assert '--unshare-all' in command and '--die-with-parent' in command
        kwargs['cwd'] = command[command.index('--chdir') + 1]
        return original_popen(command[command.index('--') + 1:], **kwargs)

    monkeypatch.setattr(sandbox.subprocess, 'Popen', launch)


def manifest(tmp_path, source, **updates):
    (tmp_path / 'worker.py').write_text(source)
    body = dict(version=1, argv=[sys.executable, '{dependency:worker}'], dependencies={'worker': 'worker.py'},
                outputs=[dict(name='result', path='result.json', type='json', schema={'type': 'object', 'required': ['value']})],
                metrics={'value': dict(output='result', selector='value', unit='units', evidence_kind='measured')},
                limits={'timeout': 2, 'memory_bytes': 134217728, 'max_file_bytes': 1048576},
                resources={'threads': 1, 'memory_bytes': 134217728, 'storage_bytes': 1048576}, max_wall_seconds=20)
    body.update(updates)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(body))
    return path


@pytest.mark.parametrize('fixture', ['numeric.toml', 'circuit.json', 'words.json'])
def test_three_unrelated_existing_scripts_are_independently_checked(tmp_path, local_transport, fixture):
    store = Store(tmp_path / 'store')
    handle = create_experiment(store, EXAMPLES / fixture, seed=0)
    result = run_experiment(store, handle)
    assert result['status'] == 'accepted_by_verifier', result
    assert result['verified_cost']['evidence_kind'] == 'verified'
    assert all(metric['evidence_kind'] == 'verified' for metric in result['metrics'].values())
    assert result['artifacts'] and result['verifier']['receipt']['accepted'] is True
    assert run_experiment(store, handle)['reused'] is True
    assert experiment_status(store, handle)['attempts'] == 1
    if fixture == 'circuit.json':
        assert store.read_blob(result['artifacts']['circuit']['sha256']).startswith(b'CX q0 q4')


def test_exit_zero_is_not_acceptance_and_missing_output_is_failure(tmp_path, local_transport):
    path = manifest(tmp_path, 'print("ran but did not write result")')
    # A stale output adjacent to the manifest never enters an isolated attempt.
    (tmp_path / 'result.json').write_text('{"value": 1}')
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'artifact_validation_failed'
    assert result['stages'][0]['status'] == 'execution_finished'
    assert 'ran but did not write' in store.read_blob(result['stages'][0]['logs']['stdout']).decode()


def test_logs_without_metrics_and_artifacts_are_honest_execution(tmp_path, local_transport):
    path = manifest(tmp_path, 'print("ordinary output")', outputs=[], metrics={})
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'execution_finished'
    assert result['numeric_analysis'].startswith('unavailable') and result['cost'] is None


@pytest.mark.parametrize('source', [
    'from pathlib import Path; Path("result.json").symlink_to("/etc/passwd")',
    'from pathlib import Path; Path("result.json").write_text("{\\"wrong\\": 1}")',
    'from pathlib import Path; Path("result.json").write_text("{\\"value\\": NaN}")',
])
def test_rejects_escaped_invalid_schema_and_nonfinite_outputs(tmp_path, local_transport, source):
    path = manifest(tmp_path, source)
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'artifact_validation_failed'


def test_argument_substitution_never_runs_shell_and_dependency_drift_blocks_resume(tmp_path, local_transport):
    source = 'import json,sys; from pathlib import Path; Path("result.json").write_text(json.dumps({"value":1,"literal":sys.argv[1]}))'
    value = '$(touch escaped); literal value'
    path = manifest(tmp_path, source, settings={'text': value}, argv=[sys.executable, '{dependency:worker}', '{setting:text}'])
    store = Store(tmp_path / 'store')
    handle = create_experiment(store, path)
    result = run_experiment(store, handle)
    assert result['status'] == 'artifact_validated'
    artifact = json.loads(store.read_blob(result['artifacts']['result']['sha256']))
    assert artifact['literal'] == value and not (tmp_path / 'escaped').exists()
    (tmp_path / 'worker.py').write_text('print("changed")')
    with pytest.raises(StrataError, match='drift'):
        run_experiment(store, handle)


@pytest.mark.parametrize('updates,match', [
    ({'argv': 'python worker.py'}, 'argument array'),
    ({'argv': [sys.executable, '{dependency:worker}', '--value={seed}']}, 'entire argument'),
    ({'outputs': [{'name': 'x', 'path': '../escape', 'type': 'json'}]}, 'escapes'),
    ({'environment': {'API_TOKEN': 'must-not-retain'}}, 'secret-like'),
    ({'enforcement': 'hard'}, 'hard aggregate'),
    ({'settings': {'value': 100}}, 'shadow'),
])
def test_invalid_manifests_fail_before_launch(tmp_path, updates, match):
    path = manifest(tmp_path, '', **updates)
    with pytest.raises(StrataError, match=match):
        load_manifest(path)


def test_staged_input_mutation_invalidates_candidate(tmp_path, local_transport):
    (tmp_path / 'input.txt').write_text('pinned input')
    source = 'import sys; from pathlib import Path; Path(sys.argv[1]).write_text("changed"); Path("result.json").write_text("{\\"value\\": 1}")'
    path = manifest(tmp_path, source, inputs={'data': 'input.txt'}, argv=[sys.executable, '{dependency:worker}', '{input:data}'])
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'artifact_validation_failed'
    assert 'modified a staged input' in result['error']['message']
    assert (tmp_path / 'input.txt').read_text() == 'pinned input'


def test_twenty_setting_sweep_and_replicates_resume_without_duplicate_execution(tmp_path, local_transport):
    store = Store(tmp_path / 'store')
    sweep = create_sweep(store, EXAMPLES / 'circuit.json', parameters=[{'width': n} for n in range(1, 6)], seeds=[0, 1], replicates=2, max_jobs=20, workers=4,
                         capacity={'threads': 2, 'memory_bytes': 536870912, 'storage_bytes': 10485760})
    before = store.get(sweep, 'experiment_sweep')
    assert len(before['experiments']) == len(set(before['experiments'])) == 20
    result = run_sweep(store, sweep)
    assert result['status']['counts'] == {'accepted_by_verifier': 20}, result['results']
    again = run_sweep(store, sweep)
    assert all(row.get('reused') for row in again['results'])
    assert len(store.all('experiment_attempt')) == 20
    with pytest.raises(StrataError, match='max_jobs'):
        create_sweep(store, EXAMPLES / 'numeric.toml', {'steps': {'start': 1, 'stop': 10**9}}, [0], 1, 20)


def test_failed_stage_resume_uses_validated_stage_and_retains_budget(tmp_path, local_transport, monkeypatch):
    source = 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"value":int(sys.argv[2])}))'
    outputs = lambda name: [dict(name=name, path=name + '.json', type='json')]
    path = manifest(tmp_path, source, outputs=outputs('last'), metrics={'value': dict(output='last', selector='value', unit='units')},
                    stages=[dict(name='first', argv=[sys.executable, '{dependency:worker}', 'first.json', '1'], outputs=outputs('first'), checkpoint={'path': 'first.json', 'type': 'json'}),
                            dict(name='last', argv=[sys.executable, '{dependency:worker}', 'last.json', '2'], outputs=outputs('last'))])
    from strata import supervision
    original = supervision.run_owned
    calls = []

    def fail_once(board, token, argv, workdir, **limits):
        calls.append(Path(workdir).name)
        if len(calls) == 2:
            return dict(returncode=1, status='nonzero_exit', seconds=.01, stdout='', stderr='intentional', output='intentional')
        return original(board, token, argv, workdir, **limits)

    monkeypatch.setattr(supervision, 'run_owned', fail_once)
    store = Store(tmp_path / 'store')
    handle = create_experiment(store, path)
    first = run_experiment(store, handle)
    assert first['status'] == 'failed' and first['checkpoint']
    second = run_experiment(store, handle)
    assert second['status'] == 'artifact_validated'
    assert calls == ['stage-0000', 'stage-0001', 'stage-0001']
    state = experiment_status(store, handle)
    assert state['attempts'] == 2 and state['spent_wall_seconds'] >= first['seconds']
    assert state['remaining_wall_seconds'] < 20


def test_unverified_worker_cost_never_becomes_verified_cost(tmp_path, local_transport):
    (tmp_path / 'verify.py').write_text('print("{\\"accepted\\": true}")')
    path = manifest(tmp_path, 'from pathlib import Path; Path("result.json").write_text("{\\"value\\":1}")',
                    verifier=dict(argv=[sys.executable, '{dependency:verify}', '{artifact:result}'], dependencies={'verify': 'verify.py'}, version='test/1'),
                    cost=dict(expression='value', unit='units'))
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'accepted_by_verifier'
    assert result['cost']['evidence_kind'] == 'model' and result['verified_cost'] is None


def test_restart_invalidates_downstream_stage_from_another_upstream(tmp_path, local_transport, monkeypatch):
    source = '''import json, sys, time
from pathlib import Path
value = time.time_ns() if sys.argv[1] == 'first' else json.loads(Path(sys.argv[2]).read_text())['value'] + 1
Path(sys.argv[1] + '.json').write_text(json.dumps({'value': value}))
'''
    output = lambda name: [dict(name=name, path=name + '.json', type='json')]
    path = manifest(tmp_path, source, outputs=output('last'), max_wall_seconds=30,
                    metrics={'value': dict(output='last', selector='value', unit='units')}, stages=[
                        dict(name='first', argv=[sys.executable, '{dependency:worker}', 'first'], outputs=output('first')),
                        dict(name='second', argv=[sys.executable, '{dependency:worker}', 'second', '{artifact:first}'], outputs=output('second')),
                        dict(name='last', argv=[sys.executable, '{dependency:worker}', 'last', '{artifact:second}'], outputs=output('last'))])
    from strata import supervision
    original = supervision.run_owned
    calls = []

    def two_failures(board, token, argv, workdir, **limits):
        calls.append(Path(workdir).name)
        if len(calls) in {3, 5}:
            return dict(returncode=1, status='nonzero_exit', seconds=.01, stdout='', stderr='retry', output='retry')
        return original(board, token, argv, workdir, **limits)

    monkeypatch.setattr(supervision, 'run_owned', two_failures)
    store = Store(tmp_path / 'store')
    handle = create_experiment(store, path)
    assert run_experiment(store, handle)['status'] == 'failed'
    assert run_experiment(store, handle, resume=False)['status'] == 'failed'
    result = run_experiment(store, handle)
    assert result['status'] == 'artifact_validated'
    assert calls == ['stage-0000', 'stage-0001', 'stage-0002', 'stage-0000', 'stage-0001', 'stage-0001', 'stage-0002']
    first = json.loads(store.read_blob(result['artifacts']['first']['sha256']))['value']
    last = json.loads(store.read_blob(result['artifacts']['last']['sha256']))['value']
    assert last == first + 2


@pytest.mark.parametrize('mutate,retained', [(False, True), (True, False)])
def test_failed_checkpoint_commit_marker_is_bound_to_exact_bytes(tmp_path, local_transport, mutate, retained):
    source = '''import hashlib, sys
from pathlib import Path
p = Path('checkpoint.json')
p.write_text('{"value": 1}')
Path('committed.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest())
'''
    if mutate:
        source += '''p.write_text('{"value": 2}')\n'''
    source += 'sys.exit(1)\n'
    checkpoint = dict(path='checkpoint.json', type='json', commit_marker='committed.sha256', schema={'type': 'object', 'required': ['value']})
    path = manifest(tmp_path, source, checkpoint=checkpoint)
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'failed'
    assert bool(result['checkpoint']) is retained


def test_reconciled_lost_attempt_preserves_reserved_compute_and_can_resume(tmp_path, local_transport):
    from strata.experiment_board import ExperimentBoard
    path = manifest(tmp_path, 'from pathlib import Path; Path("result.json").write_text("{\\"value\\": 1}")')
    store = Store(tmp_path / 'store')
    handle = create_experiment(store, path)
    board = ExperimentBoard(store)
    job = board.enqueue(handle, {'threads': 1})
    token = board.claim(job, 'lost-owner')
    store.put('experiment_attempt', dict(experiment=handle, attempt='lost', job=job, reserved_seconds=2))
    store.put('experiment_charge', dict(experiment=handle, attempt='lost', operation='run', reserved_seconds=2))
    board.finish(token, 'lost')
    assert experiment_status(store, handle)['status'] == 'reconciliation_required'
    result = run_experiment(store, handle, board=board)
    assert result['status'] == 'artifact_validated'
    state = experiment_status(store, handle)
    assert state['attempts'] == 2 and state['spent_wall_seconds'] >= 2


def test_concurrent_creators_share_one_configuration_handle(tmp_path):
    path = manifest(tmp_path, 'print("not run")')
    store = Store(tmp_path / 'store')
    barrier = threading.Barrier(4)

    def create(_):
        local = Store(store.path)
        try:
            barrier.wait()
            return create_experiment(local, path, seed=7, replicate=2)
        finally:
            local.db.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        handles = list(pool.map(create, range(4)))
    assert len(set(handles)) == 1
    assert len(store.all('experiment')) == 1


def test_runtime_bindings_preserve_invocation_without_mounting_on_symlinks():
    from strata.experiments import _readonly_runtimes
    spec = load_manifest(EXAMPLES / 'numeric.toml')
    assert Path(spec['stages'][0]['argv'][0]).name == 'python3'
    assert all(not Path(path).is_symlink() for path in _readonly_runtimes(spec))


def test_stage_timeout_overrides_global_limit(tmp_path, local_transport):
    path = manifest(tmp_path, 'import time; time.sleep(5)', outputs=[], metrics={},
                    limits={'timeout': 5, 'memory_bytes': 134217728},
                    stages=[dict(name='cheap', argv=[sys.executable, '{dependency:worker}'], limits={'timeout': .05})])
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'failed'
    assert result['stages'][0]['status'] == 'timed_out'
    assert result['stages'][0]['limits']['timeout'] == .05
    assert store.all('experiment_charge')[0][1]['reserved_seconds'] == .05


@pytest.mark.parametrize('stage_limits,match', [({'unknown': 1}, 'unsupported command limit'), ({'memory_bytes': 268435456}, 'aggregate memory reservation')])
def test_stage_resource_limits_are_validated(tmp_path, stage_limits, match):
    path = manifest(tmp_path, '', stages=[dict(name='one', argv=[sys.executable, '{dependency:worker}'], limits=stage_limits)])
    with pytest.raises(StrataError, match=match):
        load_manifest(path)


@pytest.mark.parametrize('restart', [False, True])
def test_checkpoint_only_stage_provenance_survives_or_invalidates_resume(tmp_path, local_transport, monkeypatch, restart):
    source = '''import json, sys, time
from pathlib import Path
name = sys.argv[1]
value = time.time_ns() if name == 'first' else json.loads(Path(sys.argv[2]).read_text())['value']
Path(('cp' if name == 'first' else name) + '.json').write_text(json.dumps({'value': value}))
'''
    output = lambda name: [dict(name=name, path=name + '.json', type='json')]
    path = manifest(tmp_path, source, outputs=output('last'), max_wall_seconds=30,
                    metrics={'value': dict(output='last', selector='value', unit='units')}, stages=[
                        dict(name='first', argv=[sys.executable, '{dependency:worker}', 'first'], outputs=[], checkpoint=dict(path='cp.json', type='json')),
                        dict(name='second', argv=[sys.executable, '{dependency:worker}', 'second', '{checkpoint}'], outputs=output('second')),
                        dict(name='last', argv=[sys.executable, '{dependency:worker}', 'last', '{checkpoint}'], outputs=output('last'))])
    from strata import supervision
    original = supervision.run_owned
    calls = []

    def fail(board, token, argv, workdir, **limits):
        calls.append(Path(workdir).name)
        if len(calls) == 3 or restart and len(calls) == 5:
            return dict(returncode=1, status='nonzero_exit', seconds=.01, stdout='', stderr='retry', output='retry')
        return original(board, token, argv, workdir, **limits)

    monkeypatch.setattr(supervision, 'run_owned', fail)
    store = Store(tmp_path / 'store')
    handle = create_experiment(store, path)
    assert run_experiment(store, handle)['status'] == 'failed'
    if restart:
        assert run_experiment(store, handle, resume=False)['status'] == 'failed'
    result = run_experiment(store, handle)
    assert result['status'] == 'artifact_validated', result
    checkpoint = json.loads(store.read_blob(result['checkpoint']['artifact']['sha256']))['value']
    assert json.loads(store.read_blob(result['artifacts']['second']['sha256']))['value'] == checkpoint
    assert json.loads(store.read_blob(result['artifacts']['last']['sha256']))['value'] == checkpoint
    assert calls == (['stage-0000', 'stage-0001', 'stage-0002', 'stage-0000', 'stage-0001', 'stage-0001', 'stage-0002'] if restart else ['stage-0000', 'stage-0001', 'stage-0002', 'stage-0002'])


@pytest.mark.skipif(os.environ.get('STRATA_REAL_SANDBOX') != '1', reason='requires usable Linux namespaces')
@pytest.mark.parametrize('fixture', ['numeric.toml', 'circuit.json', 'words.json'])
def test_real_sandbox_reference_workloads(tmp_path, fixture):
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, EXAMPLES / fixture, seed=0))
    assert result['status'] == 'accepted_by_verifier', result
    assert result['verified_cost'] and result['verifier']['sandbox'] == 'bubblewrap-unshare-all'
