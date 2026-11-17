"""The command replacement below is a test double, never a runtime fallback."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest
from strata import sandbox
from strata.experiment_board import ExperimentBoard
from strata.model import StrataError
from strata.store import Store
from strata.supervision import run_owned


@pytest.fixture
def local_process_double(monkeypatch):
    original = subprocess.Popen
    monkeypatch.setattr(sandbox.shutil, 'which', lambda _: '/test/bwrap')

    def launch(command, **kwargs):
        assert '--unshare-all' in command and '--die-with-parent' in command
        return original(command[command.index('--') + 1:], **kwargs)

    monkeypatch.setattr(sandbox.subprocess, 'Popen', launch)
    return original


def _owned(tmp_path):
    board = ExperimentBoard(Store(tmp_path / 'store'), dict(threads=1, memory_bytes=10**9))
    job = board.enqueue('test', dict(threads=1, memory_bytes=10**8))
    return board, job, board.claim(job, 'test-worker')


def test_zero_exit_is_only_execution_finished_and_logs_retained(tmp_path, local_process_double):
    board, job, token = _owned(tmp_path)
    run = run_owned(board, token, [sys.executable, '-c', 'print("ok")'], tmp_path)
    assert run['status'] == 'execution_finished'
    assert run['stdout'] == 'ok\n' and run['stderr'] == ''
    assert board.store.read_blob(run['stdout_handle']) == b'ok\n'
    assert board.snapshot()['jobs'][0]['status'] == 'running'
    assert board.store.get(run['execution_handle'], 'owned_execution')['returncode'] == 0
    board.finish(token, 'failed')  # No artifact was created or independently accepted.


def test_nonzero_stderr_timeout_and_bounded_output(tmp_path, local_process_double):
    board, job, token = _owned(tmp_path)
    failed = run_owned(board, token, [sys.executable, '-c', 'import sys; print("bad library", file=sys.stderr); sys.exit(3)'], tmp_path)
    assert failed['status'] == 'nonzero_exit' and failed['returncode'] == 3
    assert failed['stderr'] == 'bad library\n'
    timeout = run_owned(board, token, [sys.executable, '-u', '-c', 'import time; print("checkpoint saved"); time.sleep(10)'], tmp_path, timeout=.1)
    assert timeout['status'] == 'timed_out' and 'checkpoint saved' in timeout['stdout']
    noisy = run_owned(board, token, [sys.executable, '-c', 'print("x"*10000)'], tmp_path, max_output=50)
    assert noisy['status'] == 'output_limit' and len(noisy['output']) == 50
    board.finish(token, 'failed')
    assert len(board.store.all('owned_execution')) == 3


def test_sigkill_never_means_oom(tmp_path, local_process_double):
    board, _, token = _owned(tmp_path)
    run = run_owned(board, token, [sys.executable, '-c', 'import os,signal; os.kill(os.getpid(), signal.SIGKILL)'], tmp_path)
    assert run['diagnosis']['kind'] == 'unknown_external_or_resource_kill'
    assert run['diagnosis']['oom_confirmed'] is False
    assert run['diagnosis']['signal'] == signal.SIGKILL
    board.finish(token, 'failed')


def test_cancel_owned_grandchildren_preserves_unrelated_sentinel(tmp_path, local_process_double):
    board, job, token = _owned(tmp_path)
    sentinel = local_process_double([sys.executable, '-c', 'import time; time.sleep(15)'], start_new_session=True)
    pid_file = tmp_path / 'grandchild.pid'
    script = ('import subprocess,sys,time,pathlib; '
              'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(15)"]); '
              f'pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); time.sleep(15)')
    errors = []

    def request():
        try:
            deadline = time.monotonic() + 5
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert pid_file.exists()
            ExperimentBoard(Store(tmp_path / 'store')).request_cancel(job)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=request)
    thread.start()
    try:
        run = run_owned(board, token, [sys.executable, '-c', script], tmp_path, timeout=8)
        thread.join(5)
        assert not errors
        assert run['status'] == 'cancelled'
        assert sentinel.poll() is None
        grandchild = int(pid_file.read_text())
        try:
            identity = sandbox.process_identity(grandchild)
        except StrataError:
            pass
        else:
            deadline = time.monotonic() + 2
            while sandbox.process_state(identity) == 'alive' and time.monotonic() < deadline:
                time.sleep(.01)
            assert sandbox.process_state(identity) == 'dead'
        board.finish(token, 'cancelled')
    finally:
        sentinel.kill()
        sentinel.wait()
        thread.join(5)


def test_environment_is_declared_and_fail_closed(tmp_path, monkeypatch, local_process_double):
    board, _, token = _owned(tmp_path)
    monkeypatch.setenv('UNDECLARED_PASSWORD', 'not-passed')
    run = run_owned(board, token, [sys.executable, '-c', 'import os; print(os.getenv("OMP_NUM_THREADS")); print(os.getenv("UNDECLARED_PASSWORD"))'], tmp_path, environment={'OMP_NUM_THREADS': '2'})
    assert run['stdout'] == '2\nNone\n'
    with pytest.raises(StrataError, match='secret'):
        run_owned(board, token, ['/bin/true'], tmp_path, environment={'API_TOKEN': 'x'})
    with pytest.raises(StrataError, match='hard aggregate'):
        run_owned(board, token, ['/bin/true'], tmp_path, hard_aggregate=True)
    monkeypatch.setattr(sandbox.shutil, 'which', lambda _: None)
    with pytest.raises(StrataError, match='bubblewrap'):
        run_owned(board, token, ['/bin/true'], tmp_path)
    board.finish(token, 'failed')


def test_prelaunch_cancellation_retains_receipt_without_launch(tmp_path, monkeypatch, local_process_double):
    board, job, token = _owned(tmp_path)
    board.request_cancel(job)
    monkeypatch.setattr(sandbox.subprocess, 'Popen', lambda *a, **k: pytest.fail('cancelled command launched'))
    run = run_owned(board, token, ['/bin/true'], tmp_path)
    assert run['status'] == 'cancelled' and run['process'] is None
    board.finish(token, 'cancelled')


def test_parent_exit_and_closed_pipes_do_not_bypass_wall_timeout(tmp_path, local_process_double):
    board, _, token = _owned(tmp_path)
    script = 'import os,time; os.close(1); os.close(2); time.sleep(10)'
    run = run_owned(board, token, [sys.executable, '-c', script], tmp_path, timeout=.1)
    assert run['status'] == 'timed_out'
    board.finish(token, 'failed')


_REAL_SANDBOX = pytest.mark.skipif(os.environ.get('STRATA_REAL_SANDBOX') != '1',
                                   reason='requires an explicitly selected namespace-enabled Linux host')


def _wait_for_file(path, timeout=5):
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert path.exists(), f'process did not create {path}'


def _detached_writer(marker):
    child = ('import pathlib,time; p=pathlib.Path(' + repr(str(marker)) + '); '
             '\nwhile True:\n p.write_text(str(time.time_ns())); time.sleep(.02)')
    return ('import subprocess,sys,time; '
            f'subprocess.Popen([sys.executable,"-c",{child!r}],start_new_session=True); time.sleep(30)')


@_REAL_SANDBOX
def test_real_bwrap_cancel_reaps_detached_descendant(tmp_path):
    board, job, token = _owned(tmp_path)
    marker = tmp_path / 'detached-heartbeat'
    sentinel = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], start_new_session=True)
    errors = []

    def cancel():
        try:
            _wait_for_file(marker)
            ExperimentBoard(Store(tmp_path / 'store')).request_cancel(job)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=cancel)
    thread.start()
    try:
        run = run_owned(board, token, [sys.executable, '-c', _detached_writer(marker)], tmp_path, timeout=10)
        thread.join(6)
        assert not errors
        assert run['status'] == 'cancelled', run['stderr']
        before = marker.read_bytes()
        time.sleep(.15)
        assert marker.read_bytes() == before
        assert sentinel.poll() is None
        board.finish(token, 'cancelled')
    finally:
        sentinel.kill()
        sentinel.wait()
        thread.join(6)


@_REAL_SANDBOX
def test_real_bwrap_supervisor_death_stops_namespace_and_reconciles(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    board = ExperimentBoard(Store(tmp_path / 'store'), dict(threads=1))
    marker = work / 'detached-heartbeat'
    script = ('from strata.store import Store; from strata.experiment_board import ExperimentBoard; '
              'from strata.supervision import run_owned; import sys; '
              f'b=ExperimentBoard(Store({str(tmp_path / "store")!r})); '
              'j=b.enqueue("crash"); t=b.claim(j,"crashing-owner"); '
              f'run_owned(b,t,[sys.executable,"-c",{_detached_writer(marker)!r}],{str(work)!r},timeout=20)')
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    supervisor = subprocess.Popen([sys.executable, '-c', script], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        _wait_for_file(marker)
        identity = sandbox.process_identity(supervisor.pid)
        assert sandbox.process_state(identity) == 'alive'
        supervisor.kill()
        supervisor.wait(timeout=5)
        time.sleep(.2)
        before = marker.read_bytes()
        time.sleep(.15)
        assert marker.read_bytes() == before
        job = board.snapshot()['jobs'][0]
        deadline = time.monotonic() + 3
        while sandbox.process_state(job['process_identity']) == 'alive' and time.monotonic() < deadline:
            time.sleep(.01)
        with board.db:
            board.db.execute('UPDATE experiment_board_jobs SET lease_until=0 WHERE job_id=?', (job['job_id'],))
        assert board.reconcile(job['job_id'])['status'] == 'lost'
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
        supervisor.wait(timeout=5)
        supervisor.stdout.close()
        supervisor.stderr.close()


def test_shell_style_exit_code_is_not_asserted_to_be_a_signal(tmp_path, local_process_double):
    board, _, token = _owned(tmp_path)
    run = run_owned(board, token, [sys.executable, '-c', 'raise SystemExit(137)'], tmp_path)
    assert run['status'] == 'nonzero_exit'
    assert run['diagnosis']['possible_signal'] == signal.SIGKILL
    assert run['diagnosis']['oom_confirmed'] is False
    assert 'signal' not in run['diagnosis']
    board.finish(token, 'failed')
