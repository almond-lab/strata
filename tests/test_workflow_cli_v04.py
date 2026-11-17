"""The first-run commands are exercised as an integrated workflow.

Normal CI substitutes only bubblewrap's transport; STRATA_REAL_SANDBOX=1 uses
the actual offline boundary. No such fallback exists in production code.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from strata import sandbox
from strata.cli import main


@pytest.fixture
def transport(monkeypatch):
    if os.environ.get('STRATA_REAL_SANDBOX') == '1':
        return
    original_popen = subprocess.Popen
    original_which = sandbox.shutil.which
    monkeypatch.setattr(sandbox.shutil, 'which',
                        lambda name: '/test/bwrap' if name == 'bwrap' else original_which(name))

    def launch(command, **kwargs):
        assert '--unshare-all' in command and '--die-with-parent' in command
        kwargs['cwd'] = command[command.index('--chdir') + 1]
        return original_popen(command[command.index('--') + 1:], **kwargs)

    monkeypatch.setattr(sandbox.subprocess, 'Popen', launch)


def invoke(capsys, argv, expected_code=0):
    code = main(argv)
    captured = capsys.readouterr()
    assert code == expected_code, captured.err + captured.out
    return json.loads(captured.out)


def test_wrap_run_report_and_twenty_setting_sweep(tmp_path, capsys, transport):
    script = tmp_path / 'existing.py'
    script.write_text('import json,sys\nfrom pathlib import Path\n'
                      'x=int(sys.argv[1])\n'
                      'Path("result.json").write_text(json.dumps({"value":x,"loss":(x-3)**2}))\n')
    manifest = tmp_path / 'pipeline.json'
    store = str(tmp_path / 'store')
    wrapped = invoke(capsys, ['wrap', '--out', str(manifest),
                             '--output', 'result=result.json:json',
                             '--metric', 'loss=result:loss:squared-error',
                             '--setting', 'x=3', '--', sys.executable, str(script), '{setting:x}'])
    assert wrapped['status'] == 'wrapped'
    created = invoke(capsys, ['experiment', 'create', str(manifest), '--store', store])
    result = invoke(capsys, ['experiment', 'run', created['experiment'], '--store', store])
    assert result['status'] == 'artifact_validated'
    assert result['metrics']['loss']['value'] == 0
    assert result['verifier'] is None
    parameters = tmp_path / 'grid.json'
    parameters.write_text(json.dumps({'x': list(range(20))}))
    plan = invoke(capsys, ['sweep', 'create', str(manifest), '--store', store,
                          '--parameters', str(parameters), '--workers', '2', '--max-jobs', '20'])
    completed = invoke(capsys, ['sweep', 'run', plan['sweep'], '--store', store])
    assert len(completed['results']) == 20
    assert all(row['status'] == 'artifact_validated' for row in completed['results'])
    board = invoke(capsys, ['board', '--store', store])
    assert len(board['jobs']) >= 20
    events = invoke(capsys, ['board', '--store', store, '--events-after', '0'])['events']
    assert events and events[-1]['seq'] > events[0]['seq']
    report = invoke(capsys, ['experiment-report', '--store', store])
    assert report
    packet = invoke(capsys, ['agent-packet', '--store', store])
    assert packet


def test_missing_artifact_cannot_finish_and_reuse_cannot_hide_dependency_drift(tmp_path, capsys, transport):
    script = tmp_path / 'missing.py'
    script.write_text('print("finished")\n')
    manifest = tmp_path / 'pipeline.json'
    store = str(tmp_path / 'store')
    invoke(capsys, ['wrap', '--out', str(manifest), '--output', 'table=table.json:json',
                    '--', sys.executable, str(script)])
    made = invoke(capsys, ['experiment', 'create', str(manifest), '--store', store])
    result = invoke(capsys, ['experiment', 'run', made['experiment'], '--store', store], expected_code=1)
    assert result['status'] == 'artifact_validation_failed'
    script.write_text('print("changed script")\n')
    assert main(['experiment', 'run', made['experiment'], '--store', store]) == 2
    captured = capsys.readouterr()
    assert 'drift' in captured.err + captured.out
