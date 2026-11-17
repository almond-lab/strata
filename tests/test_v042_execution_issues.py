"""Reported execution regressions with real argv, staging and process rlimits."""
import json
from pathlib import Path
import shutil
import sys

import pytest

from strata import sandbox, supervision
from strata.experiments import create_experiment, experiment_status, load_manifest, run_experiment
from strata.model import StrataError
from strata.store import Store
from test_v04_experiments import local_transport, manifest


@pytest.mark.parametrize('remove', [False, True])
@pytest.mark.parametrize('direct', [False, True])
def test_running_attempt_uses_frozen_sources_after_live_edit(tmp_path, local_transport, monkeypatch, remove, direct):
    source = '#!/usr/bin/python3\nfrom pathlib import Path; Path("result.json").write_text("{\\\"value\\\":1}")\n'
    updates = {'argv': ['{dependency:worker}']} if direct else {}
    path = manifest(tmp_path, source, **updates)
    worker = tmp_path / 'worker.py'
    worker.chmod(0o755)
    original = supervision.run_owned

    def edit_after_launch(*args, **kwargs):
        receipt = original(*args, **kwargs)
        worker.unlink() if remove else worker.write_text('raise RuntimeError("new source")')
        return receipt

    monkeypatch.setattr(supervision, 'run_owned', edit_after_launch)
    store = Store(tmp_path / 'store')
    handle = create_experiment(store, path)
    result = run_experiment(store, handle)
    assert result['status'] == 'artifact_validated', result
    assert json.loads(store.read_blob(result['artifacts']['result']['sha256'])) == {'value': 1}
    with pytest.raises(StrataError):
        run_experiment(store, handle)


def test_later_stages_keep_frozen_bytes_and_modes_after_source_removal(tmp_path, local_transport, monkeypatch):
    source = '#!/usr/bin/python3\nfrom pathlib import Path; Path("result.json").write_text("{\\\"value\\\":1}")\n'
    path = manifest(tmp_path, source, stages=[
        {'name': 'first', 'argv': ['{dependency:worker}'], 'outputs': []},
        {'name': 'last', 'argv': ['{dependency:worker}']}])
    worker = tmp_path / 'worker.py'
    worker.chmod(0o755)
    original = supervision.run_owned

    def remove_after_first_stage(*args, **kwargs):
        receipt = original(*args, **kwargs)
        worker.unlink(missing_ok=True)
        return receipt

    monkeypatch.setattr(supervision, 'run_owned', remove_after_first_stage)
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'artifact_validated', result
    assert len(result['stages']) == 2


def test_executable_in_dependency_directory_survives_live_tree_removal(tmp_path, local_transport, monkeypatch):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    worker = bundle / 'worker'
    worker.write_text('#!/usr/bin/python3\nfrom pathlib import Path; Path("result.json").write_text("{\\\"value\\\":1}")\n')
    worker.chmod(0o755)
    path = manifest(tmp_path, '', dependencies={'bundle': 'bundle'}, stages=[
        {'name': 'first', 'argv': [str(worker)], 'outputs': []},
        {'name': 'last', 'argv': [str(worker)]}])
    original = supervision.run_owned

    def remove_live_bundle(*args, **kwargs):
        receipt = original(*args, **kwargs)
        if bundle.exists():
            shutil.rmtree(bundle)
        return receipt

    monkeypatch.setattr(supervision, 'run_owned', remove_live_bundle)
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'artifact_validated', result
    assert len(result['stages']) == 2


def test_literal_external_runtime_still_checks_live_bytes_when_also_declared(tmp_path, local_transport, monkeypatch):
    worker = tmp_path / 'worker.py'
    path = manifest(tmp_path, '#!/usr/bin/python3\nfrom pathlib import Path; Path("result.json").write_text("{\\\"value\\\":1}")\n',
                    argv=[str(worker)])
    worker.chmod(0o755)
    original = supervision.run_owned

    def change_external_runtime(*args, **kwargs):
        receipt = original(*args, **kwargs)
        worker.write_text('# changed live executable\n')
        return receipt

    monkeypatch.setattr(supervision, 'run_owned', change_external_runtime)
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'artifact_validation_failed', result
    assert 'runtime drift' in result['error']['message']


def test_mixed_staged_and_live_executable_cannot_launch_changed_live_bytes(tmp_path, local_transport, monkeypatch):
    worker = tmp_path / 'worker.py'
    path = manifest(tmp_path, '#!/usr/bin/python3\nfrom pathlib import Path; Path("result.json").write_text("{\\\"value\\\":1}")\n',
                    stages=[{'name': 'staged', 'argv': ['{dependency:worker}'], 'outputs': []},
                            {'name': 'live', 'argv': [str(worker)]}])
    worker.chmod(0o755)
    original = supervision.run_owned
    calls = []

    def change_before_live_invocation(*args, **kwargs):
        calls.append(True)
        receipt = original(*args, **kwargs)
        worker.write_text('# changed executable\n')
        return receipt

    monkeypatch.setattr(supervision, 'run_owned', change_before_live_invocation)
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'artifact_validation_failed', result
    assert 'runtime drift' in result['error']['message']
    assert len(calls) == 1


@pytest.mark.parametrize('top,stage,expected', [({}, {'timeout': 600}, 600),
    ({'timeout': 2.5}, {'timeout': 10.2}, 11),
    ({'timeout': 2, 'cpu_seconds': 4}, {'timeout': 600}, 4),
    ({'cpu_seconds': 4}, {'timeout': 600, 'cpu_seconds': None}, 600)])
def test_stage_and_verifier_derive_cpu_from_own_timeout(tmp_path, top, stage, expected):
    (tmp_path / 'verify.py').write_text('print("verifier")')
    path = manifest(tmp_path, '', limits=top, resources={}, stages=[
        {'name': 'worker', 'argv': [sys.executable, '{dependency:worker}'], 'limits': stage}],
        verifier={'version': 'cpu/1', 'dependencies': {'verify': 'verify.py'},
                  'argv': [sys.executable, '{dependency:verify}', '{artifact:result}'], 'limits': stage})
    frozen = load_manifest(path)
    assert frozen['stages'][0]['limits']['cpu_seconds'] == expected
    assert frozen['verifier']['limits']['cpu_seconds'] == expected


@pytest.mark.parametrize('explicit_cpu,expected', [(None, 'accepted_by_verifier'), (1, 'verification_failed')])
def test_pool_child_cpu_failure_is_not_scientific_rejection(tmp_path, local_transport, explicit_cpu, expected):
    # Use a real child with inherited rlimits; this host does not permit the
    # POSIX semaphore allocation required by ProcessPoolExecutor itself.
    (tmp_path / 'verify.py').write_text('''import json, subprocess, sys
child = subprocess.run([sys.executable, '-c', 'import time; start=time.process_time()\\nwhile time.process_time()-start < 1.3: pass'])
payload = {'accepted': child.returncode == 0, 'child_returncode': child.returncode}
if child.returncode:
    payload['error'] = 'A process in the process pool was terminated abruptly while the future was running or pending.'
print(json.dumps(payload))
''')
    top = {'timeout': 1, 'memory_bytes': 134217728, 'max_file_bytes': 1048576}
    if explicit_cpu is not None:
        top['cpu_seconds'] = explicit_cpu
    path = manifest(tmp_path, 'from pathlib import Path; Path("result.json").write_text("{\\\"value\\\":1}")',
        limits=top, verifier={'version': 'pool/1', 'dependencies': {'verify': 'verify.py'},
            'argv': [sys.executable, '{dependency:verify}', '{artifact:result}'], 'limits': {'timeout': 6}})
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == expected, result
    if explicit_cpu is None:
        assert result['verifier']['limits']['cpu_seconds'] == 6
    else:
        assert result['verifier']['receipt']['accepted'] is False
        assert result['verifier']['receipt']['child_returncode'] == -24
        assert result['error']['diagnosis']['kind'] == 'descendant_process_failed'
        assert result['artifacts']


@pytest.mark.parametrize('payload', ['not-json', '{"other": false}', '{"accepted": "false"}'])
def test_missing_verifier_verdict_is_verification_failed(tmp_path, local_transport, payload):
    (tmp_path / 'verify.py').write_text('print(' + repr(payload) + ')')
    path = manifest(tmp_path, 'from pathlib import Path; Path("result.json").write_text("{\\\"value\\\":1}")',
        verifier={'version': 'bad-receipt/1', 'dependencies': {'verify': 'verify.py'},
                  'argv': [sys.executable, '{dependency:verify}', '{artifact:result}']})
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'verification_failed', result
    assert result['artifacts'] and result['verifier']['logs']['stdout']


def test_verifier_only_metric_has_independent_provenance(tmp_path, local_transport):
    (tmp_path / 'verify.py').write_text('print(\'{"accepted":true,"value":1,"errors":0}\')')
    path = manifest(tmp_path, 'from pathlib import Path; Path("result.json").write_text("{\\\"value\\\":1}")',
        verifier={'version': 'metrics/1', 'dependencies': {'verify': 'verify.py'},
            'argv': [sys.executable, '{dependency:verify}', '{artifact:result}'],
            'metrics': {'value': 'value', 'errors': {'selector': 'errors', 'unit': 'count'}}})
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'accepted_by_verifier', result
    metric = result['metrics']['errors']
    assert metric['value'] == 0 and metric['unit'] == 'count' and metric['evidence_kind'] == 'verified'
    assert metric['source'] == {'verifier_identity': result['verifier']['identity'], 'verifier_selector': 'errors'}
    assert result['metrics']['value']['source']['output'] == 'result'


@pytest.mark.parametrize('outputs', [False, True])
def test_worker_success_survives_attempt_budget_and_is_reused(tmp_path, local_transport, outputs):
    updates = {} if outputs else {'outputs': [], 'metrics': {}}
    path = manifest(tmp_path, 'from pathlib import Path; Path("result.json").write_text("{\\\"value\\\":1}")', max_attempts=1, **updates)
    store = Store(tmp_path / 'store')
    handle = create_experiment(store, path)
    first = run_experiment(store, handle)
    state = experiment_status(store, handle)
    assert state['status'] == ('artifact_validated' if outputs else 'execution_finished')
    assert state['attempt_budget_exhausted'] is True
    second = run_experiment(store, handle)
    assert second['reused'] and second['attempt'] == first['attempt']
    assert experiment_status(store, handle)['attempts'] == 1


def test_success_after_failure_is_reused_at_final_attempt(tmp_path, local_transport, monkeypatch):
    path = manifest(tmp_path, 'from pathlib import Path; Path("result.json").write_text("{\\\"value\\\":1}")', max_attempts=2)
    original = supervision.run_owned
    calls = []
    def fail_once(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            return {'returncode': 1, 'status': 'nonzero_exit', 'seconds': .01}
        return original(*args, **kwargs)
    monkeypatch.setattr(supervision, 'run_owned', fail_once)
    store = Store(tmp_path / 'store')
    handle = create_experiment(store, path)
    assert run_experiment(store, handle)['status'] == 'failed'
    assert run_experiment(store, handle)['status'] == 'artifact_validated'
    assert run_experiment(store, handle)['reused']
    assert len(calls) == 2 and experiment_status(store, handle)['attempt_budget_exhausted']


@pytest.mark.parametrize('declared', [False, True])
def test_relative_output_argument_is_literal_despite_live_filename_collision(tmp_path, local_transport, declared):
    (tmp_path / 'result.json').write_text('unrelated original')
    path = manifest(tmp_path, 'import sys; from pathlib import Path; Path(sys.argv[1]).write_text("{\\\"value\\\":1}")',
        inputs={'old': 'result.json'} if declared else {}, argv=[sys.executable, '{dependency:worker}', 'result.json'])
    assert load_manifest(path)['stages'][0]['argv'][-1] == 'result.json'
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'artifact_validated', result
    assert (tmp_path / 'result.json').read_text() == 'unrelated original'


def test_manifest_filename_can_be_output_argument(tmp_path, local_transport):
    path = manifest(tmp_path, 'import sys; from pathlib import Path; Path(sys.argv[1]).write_text("{\\\"value\\\":1}")',
        argv=[sys.executable, '{dependency:worker}', 'manifest.json'],
        outputs=[{'name': 'result', 'path': 'manifest.json', 'type': 'json'}])
    store = Store(tmp_path / 'store')
    assert run_experiment(store, create_experiment(store, path))['status'] == 'artifact_validated'


def test_pinned_shell_script_worker_and_verifier_use_literal_arguments(tmp_path, local_transport):
    (tmp_path / 'verify.sh').write_text('test -s "$1" && printf \'{"accepted":true}\'\n')
    literal = '$(touch escaped); literal argument'
    path = manifest(tmp_path, 'printf \'%s\' "$1" > result.txt\n',
        settings={'text': literal}, argv=['bash', '-eu', '{dependency:worker}', '{setting:text}'], metrics={},
        outputs=[{'name': 'result', 'path': 'result.txt', 'type': 'text'}],
        verifier={'version': 'shell/1', 'dependencies': {'verify': 'verify.sh'},
                  'argv': ['bash', '{dependency:verify}', '{artifact:result}']})
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'accepted_by_verifier', result
    assert store.read_blob(result['artifacts']['result']['sha256']).decode() == literal
    assert not list((store.path / 'attempts').rglob('escaped'))


@pytest.mark.parametrize('args', [['-c', 'echo unsafe'], ['-lc', 'echo unsafe'], ['worker.py'], [],
                                  ['-file', '{dependency:worker}'], ['-NonInteractive', '{dependency:worker}']])
def test_shell_text_and_unpinned_scripts_still_fail_before_execution(tmp_path, args):
    path = manifest(tmp_path, '', argv=['bash', *args])
    with pytest.raises(StrataError, match='pinned dependency script'):
        load_manifest(path)


@pytest.mark.parametrize('flags,accepted', [(['-NoProfile', '-NonInteractive', '-File'], True),
    (['-e'], False), (['-EncodedCommand'], False), (['-Command'], False)])
def test_powershell_file_flags_do_not_allow_command_text(tmp_path, flags, accepted):
    # Parser coverage does not require PowerShell installed or run an interpreter.
    executable = tmp_path / 'powershell'
    executable.write_text('parser fixture')
    path = manifest(tmp_path, '', argv=[str(executable), *flags, '{dependency:worker}'])
    if accepted:
        assert load_manifest(path)['stages'][0]['argv'][-1] == '{dependency:worker}'
    else:
        with pytest.raises(StrataError, match='pinned dependency script'):
            load_manifest(path)


@pytest.mark.parametrize('field', ['memory_bytes', 'max_file_bytes', 'max_output', 'cpu_seconds'])
def test_resource_limits_accept_integral_floats_and_name_invalid_field(field):
    assert sandbox.effective_limits(**{field: 3600.0})[field] == 3600
    assert type(sandbox.effective_limits(**{field: 3600.0})[field]) is int
    for value in (0, -1, 1.5, True, float('inf'), float('nan')):
        with pytest.raises(StrataError, match=field):
            sandbox.effective_limits(**{field: value})
