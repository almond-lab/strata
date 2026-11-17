"""Real scripts for filename, empty-output and compact-retention regressions."""
import errno
import json
import os
from pathlib import Path
import sys

import pytest

from strata.experiments import create_experiment, load_manifest, run_experiment
from strata.model import StrataError
from strata.store import Store
from test_v04_experiments import local_transport, manifest


def extension_manifest(tmp_path):
    (tmp_path / 'input.npz').write_bytes(b'extension-dispatched-data')
    (tmp_path / 'chemgrad.py').write_text('def compute(value):\n    return value + 1\n')
    source = '''import json, sys, inspect
from pathlib import Path
import chemgrad
from strata.callbacks import PinnedCallback
assert Path(sys.argv[1]).suffix == '.npz'
assert Path(inspect.getsourcefile(chemgrad.compute)).name == 'chemgrad.py'
assert PinnedCallback(chemgrad.compute)(2) == 3
Path('result.npz').write_bytes(Path(sys.argv[1]).read_bytes())
Path('stderr.txt').touch()
'''
    (tmp_path / 'verify.py').write_text('''import json, sys
from pathlib import Path
assert Path(sys.argv[1]).suffix == '.npz'
print(json.dumps({'accepted': Path(sys.argv[1]).read_bytes() == b'extension-dispatched-data'}))
''')
    return manifest(tmp_path, source, dependencies={'kkt': 'worker.py', 'helper': 'chemgrad.py'},
                    inputs={'npz': 'input.npz'}, argv=[sys.executable, '{dependency:kkt}', '{input:npz}'],
                    outputs=[dict(name='result', path='result.npz', type='binary'),
                             dict(name='stderr', path='stderr.txt', type='text')], metrics={},
                    limits={'timeout': 5, 'memory_bytes': 536870912, 'max_file_bytes': 1048576},
                    resources={'threads': 1, 'memory_bytes': 536870912, 'storage_bytes': 1048576},
                    verifier={'argv': [sys.executable, '{dependency:check}', '{artifact:result}'],
                              'dependencies': {'check': 'verify.py'}, 'version': 'fixture/1'})


def test_suffix_sibling_import_and_pinned_callback(tmp_path, local_transport):
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, extension_manifest(tmp_path)))
    assert result['status'] == 'accepted_by_verifier', result
    assert result['artifacts']['stderr']['bytes'] == 0
    assert store.read_blob(result['artifacts']['result']['sha256']) == b'extension-dispatched-data'
    assert result['retention']['removed_files'] > 3


@pytest.mark.skipif(os.environ.get('STRATA_REAL_SANDBOX') != '1', reason='requires usable Linux namespaces')
def test_real_sandbox_suffix_import_callback_and_empty_log(tmp_path):
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, extension_manifest(tmp_path)))
    assert result['status'] == 'accepted_by_verifier', result
    assert result['verifier']['sandbox'] == 'bubblewrap-unshare-all'


@pytest.mark.parametrize('kind', ['text', 'jsonl'])
def test_well_formed_empty_outputs_are_retained(tmp_path, local_transport, kind):
    source = 'from pathlib import Path; Path("empty").touch()'
    path = manifest(tmp_path, source, outputs=[dict(name='empty', path='empty', type=kind)], metrics={})
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'artifact_validated'
    assert store.read_blob(result['artifacts']['empty']['sha256']) == b''
    raw = json.loads(path.read_text())
    raw['outputs'][0]['nonempty'] = True
    path.write_text(json.dumps(raw))
    rejected = run_experiment(store, create_experiment(store, path))
    assert rejected['status'] == 'artifact_validation_failed'
    assert 'empty' in rejected['error']['message']


@pytest.mark.parametrize('kind,nonempty', [('json', False), ('csv', False), ('binary', True)])
def test_empty_json_csv_and_required_binary_do_not_pass(tmp_path, local_transport, kind, nonempty):
    path = manifest(tmp_path, 'from pathlib import Path; Path("empty").touch()',
                    outputs=[dict(name='empty', path='empty', type=kind, nonempty=nonempty)], metrics={})
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'artifact_validation_failed'


def test_nonempty_must_be_boolean(tmp_path):
    path = manifest(tmp_path, '', outputs=[dict(name='x', path='x', type='text', nonempty='false')])
    with pytest.raises(StrataError, match='nonempty must be boolean'):
        load_manifest(path)


@pytest.mark.parametrize('policy', ['compact', 'keep'])
def test_repeated_binary_attempts_retain_one_object_and_resume(tmp_path, local_transport, policy):
    payload = b'not a sparse zero file\n' * 32768
    (tmp_path / 'input.kmb').write_bytes(payload)
    (tmp_path / 'verify.py').write_text('import json, sys; from pathlib import Path; print(json.dumps({"accepted": Path(sys.argv[1]).stat().st_size > 0}))')
    path = manifest(tmp_path, 'import shutil,sys; shutil.copyfile(sys.argv[1], "candidate.kmb")',
                    inputs={'data': 'input.kmb'}, argv=[sys.executable, '{dependency:worker}', '{input:data}'],
                    outputs=[dict(name='candidate', path='candidate.kmb', type='binary')], metrics={},
                    retention=policy, verifier={'argv': [sys.executable, '{dependency:verify}', '{artifact:candidate}'],
                                               'dependencies': {'verify': 'verify.py'}, 'version': 'fixture/1'})
    store = Store(tmp_path / 'store')
    handles = []
    for seed in range(3):
        handle = create_experiment(store, path, seed=seed)
        handles.append(handle)
        result = run_experiment(store, handle)
        assert result['status'] == 'accepted_by_verifier', result
        assert not result['retention']['errors']
    copies = [p for p in store.path.rglob('*') if p.is_file() and p.stat().st_size == len(payload)]
    # The verifier sees only its explicit inputs, so it doesn't restage the
    # unrelated worker input: worker input/output + verifier artifact per run.
    assert len(copies) == (1 if policy == 'compact' else 10)
    assert run_experiment(store, handles[0])['reused'] is True
    assert store.read_blob(result['artifacts']['candidate']['sha256']) == payload


def test_failed_unvalidated_output_is_kept(tmp_path, local_transport):
    path = manifest(tmp_path, 'from pathlib import Path; Path("result.json").write_text("partial output"); raise SystemExit(1)')
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'failed'
    outputs = list((store.path / 'attempts').rglob('result.json'))
    assert len(outputs) == 1 and outputs[0].read_text() == 'partial output'


def test_reflink_fallback_is_private_and_preserves_object(tmp_path, monkeypatch):
    from strata import store as module
    store = Store(tmp_path / 'store')
    handle = store.blob(b'pinned')
    def unsupported(*_):
        raise OSError(errno.EOPNOTSUPP, 'fixture has no reflink')
    monkeypatch.setattr(module.fcntl, 'ioctl', unsupported)
    target = store.copy_blob(handle, tmp_path / 'copy')
    assert target.stat().st_ino != store._blob_path(handle).stat().st_ino
    target.write_bytes(b'changed')
    assert store.read_blob(handle) == b'pinned'


@pytest.mark.parametrize('resources', [{}, {'value': 9}])
def test_verifier_rejection_never_requires_success_cost_fields(tmp_path, request, resources):
    if os.environ.get('STRATA_REAL_SANDBOX') != '1':
        request.getfixturevalue('local_transport')
    payload = json.dumps(dict(accepted=False, resources=resources, reason='physical accuracy rejected'))
    (tmp_path / 'verify.py').write_text('print(' + repr(payload) + ')')
    path = manifest(tmp_path, 'from pathlib import Path; Path("result.json").write_text("{\\"value\\":1}")',
                    verifier=dict(argv=[sys.executable, '{dependency:verify}', '{artifact:result}'],
                                  dependencies={'verify': 'verify.py'}, version='rejection/1'),
                    cost=dict(expression='value', unit='cost', resources={'value': 'resources.value'}))
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == 'rejected_by_verifier', result
    assert result['error'] is None and result['cost'] is None and result['verified_cost'] is None
    assert result['verifier']['receipt']['reason'] == 'physical accuracy rejected'
    assert store.read_blob(result['artifacts']['result']['sha256']) == b'{"value":1}'
