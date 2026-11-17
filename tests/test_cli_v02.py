"""Public v0.2 commands execute their real store and inference integrations."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

from strata import cli
from strata.store import Store


def forbid_project(*args, **kwargs):
    raise AssertionError('standalone commands must not load a project or adapter')


def read_json(capsys):
    return json.loads(capsys.readouterr().out)


def test_store_commands_import_report_and_reopen_without_project(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, 'open_project', forbid_project)
    path = tmp_path / 'memory'
    store = Store(path)
    artifact = store.blob(b'exactly scoped public fixture evidence')
    note = store.put('agent_research_note', dict(parent_sha256='f' * 64,
                    evidence_kind='estimate', reported_note='a retained prebuild observation'))
    source = tmp_path / 'attempts.json'
    source.write_text(json.dumps([
        dict(id='known', contract_fingerprint='contract-a', move='recompute',
             evidence_kind='exact_small', scale={'bits': 1}, artifact_hash=artifact,
             reopen_if='a different width'),
        dict(id='unbound', note='a task name is not a semantic contract'),
    ]))
    assert cli.main(['memory-import', str(source), '--store', str(path)]) == 0
    assert read_json(capsys) == {'imported': 1, 'unbound': 1, 'priced': 0, 'closed': 1}
    assert cli.main(['memory-status', '--store', str(path), '--scale', '{"bits": 1}']) == 0
    status = read_json(capsys)
    assert status['contracts'] == ['contract-a']
    assert status['counts']['closed_at_size'] == 1 and len(status['unbound']) == 1
    assert cli.main(['memory-status', '--store', str(path), '--contract', 'contract-a',
                     '--scale', '{"bits": 2}']) == 0
    assert read_json(capsys)['items'][0]['reason'] == 'scale_changed_or_unknown'
    before = len(store.all('negative'))
    assert cli.main(['report', '--store', str(path), '--format', 'json']) == 0
    report = read_json(capsys)
    assert report['negative_memory'][0]['evidence_kind'] == 'exact_small'
    assert report['unbound_attempts'][0]['identifier'] == 'unbound'
    assert len(store.all('negative')) == before
    assert cli.main(['report', '--store', str(path), '--handle', note, '--format', 'markdown']) == 0
    markdown = capsys.readouterr().out
    assert markdown.startswith('# Strata research report')
    assert 'a retained prebuild observation' in markdown
    assert cli.main(['memory-status', '--store', str(path), '--scale', '[]']) == 2
    assert 'JSON object' in read_json(capsys)['error']


def test_empty_memory_status_and_unknown_report_are_explicit(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, 'open_project', forbid_project)
    path = tmp_path / 'empty'
    assert cli.main(['memory-status', '--store', str(path)]) == 0
    assert read_json(capsys)['items'] == []
    assert cli.main(['report', '--store', str(path), '--handle', 'unknown']) == 2
    assert 'unknown report handle' in read_json(capsys)['error']


@pytest.mark.parametrize('suffix,text', [
    ('.kmx', 'CCX q0 q1 q2\n'),
    ('.ops', '{"op":"CCX","qubits":[0,1,2]}\n'),
])
def test_file_queries_use_real_inference_and_scoped_boolean_floors(tmp_path, capsys, monkeypatch, suffix, text):
    monkeypatch.setattr(cli, 'open_project', forbid_project)
    source = tmp_path / ('input' + suffix)
    source.write_text(text)
    query = ['function-of', str(source), '--wire', '2', '--at', '1', '--support', '0,1']
    assert cli.main(query) == 0
    assert read_json(capsys)['status'] == 'no'
    assert cli.main(query + ['--clean-inputs', '2']) == 0
    function = read_json(capsys)
    assert function['status'] == 'yes' and function['form']['monomials'] == [[0, 1]]
    assert cli.main(['slack', str(source), '--at', '1', '--wires', '0,1,2',
                     '--clean-inputs', '2']) == 0
    slack = read_json(capsys)
    assert slack['slack_bits'] == 1 and slack['information_bits'] == 2
    assert 'does not establish a free clean scratch wire' in slack['note']
    assert cli.main(['floors', str(source)]) == 0
    floor = read_json(capsys)
    assert floor['status'] == 'exact' and floor['kind'] == 'proof'
    assert floor['resource'] == 'and_count' and floor['value'] == 1
    assert floor['clean_inputs'] == [] and floor['wires'] == [0, 1, 2]
    assert 'free XOR/NOT/constants' in floor['model']
    assert 'no measurement/phase cost claim' in floor['scope']


def test_unsupported_and_invalid_queries_never_invent_a_floor(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, 'open_project', forbid_project)
    source = tmp_path / 'measurement.kmx'
    source.write_text('HMR q0 b0\n')
    assert cli.main(['floors', str(source)]) == 0
    unknown = read_json(capsys)
    assert unknown['status'] == 'unknown' and 'value' not in unknown
    assert cli.main(['slack', str(source), '--wires', '-1']) == 2
    assert 'nonnegative' in read_json(capsys)['error']
    assert cli.main(['function-of', str(source), '--wire', '0', '--at', '1',
                     '--support', '1,,2']) == 2
    assert read_json(capsys)['error']


def test_analyze_and_lift_use_configured_bundle_frontend_without_verification(tmp_path, capsys, monkeypatch):
    from strata import sandbox
    monkeypatch.setattr(sandbox, 'run', forbid_project)
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'circuit.kmx').write_text('X q0\nX q0\n')
    (base / 'numbers.json').write_text('{"rank": 2}')
    verifier = tmp_path / 'verifier.py'
    verifier.write_text('raise AssertionError("inventory must not run a verifier")\n')
    config = tmp_path / 'strata.toml'
    config.write_text(f'''[project]
store = ".strata"
[score]
formula = "geomean(task.base_cost / task.cost)"
[[task]]
id = "bundle"
base = "base"
frontend = "bundle"
cost = "Q"
[[task.components]]
id = "circuit"
path = "circuit.kmx"
kind = "kickmix"
[[task.components]]
id = "numbers"
path = "numbers.json"
kind = "data"
[task.verifier]
command = [{json.dumps(sys.executable)}, {json.dumps(str(verifier))}, "{{request}}"]
sha256 = "{hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()}"
[task.verifier.dependencies]
{json.dumps(str(verifier))} = "{hashlib.sha256(verifier.read_bytes()).hexdigest()}"
''')
    args = ['--project', str(config)]
    assert cli.main(args + ['analyze', 'bundle']) == 0
    analyzed = read_json(capsys)
    assert set(analyzed['components']) == {'circuit', 'numbers'}
    assert analyzed['metrics']['Q'] == 1
    assert (tmp_path / '.strata' / 'analysis' / 'bundle' / 'analysis.json').is_file()
    assert cli.main(args + ['lift', 'bundle']) == 0
    lifted = read_json(capsys)
    assert lifted['coverage']['coverage'] == 'component boundaries only'
    assert lifted['confirmed'] is False


def test_typed_extension_interfaces_are_public():
    import strata
    names = ('Bundle', 'BundleFrontend', 'Component', 'DataFrontend',
             'CallbackAdapter', 'PinnedCallback', 'BuildRequest', 'BuildReceipt',
             'CheckRequest', 'CheckReceipt', 'FullArtifactLab', 'ComponentLab',
             'BundleLab', 'StructuralRequest', 'StructuralReceipt',
             'ProbabilityPolicy', 'OperationStream')
    assert all(isinstance(getattr(strata, name), type) for name in names)
