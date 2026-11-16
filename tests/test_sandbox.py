import subprocess
from pathlib import Path
import pytest
from strata import sandbox
from strata.model import StrataError


def test_offline_mount_order_and_sanitized_environment(monkeypatch,tmp_path):
    monkeypatch.setattr(sandbox.shutil,'which',lambda name:'/usr/bin/bwrap')
    monkeypatch.setenv('KAGGLE_API_TOKEN','fake-test-only')
    captured={}
    def fake(command,**kw):
        captured['command']=command;captured['env']=kw['env']
        kw['stdout'].write(b'ok')
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(sandbox.subprocess,'run',fake)
    assert sandbox.run(['/bin/true'],tmp_path)['output']=='ok'
    cmd=captured['command']
    assert cmd.index('--tmpfs')<cmd.index('--bind')
    assert '--unshare-all' in cmd and 'KAGGLE_API_TOKEN' not in captured['env']


def test_sandbox_never_falls_back(monkeypatch,tmp_path):
    monkeypatch.setattr(sandbox.shutil,'which',lambda name:None)
    with pytest.raises(StrataError,match='bubblewrap'):sandbox.run(['/bin/true'],tmp_path)
