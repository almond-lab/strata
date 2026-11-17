"""Concurrent ownership and conservative recovery of durable attempt records."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time

import pytest
from strata.experiment_board import ExperimentBoard
from strata.model import StrataError
from strata.sandbox import process_identity, process_state
from strata.store import Store


def _claim(path, job_id):
    return ExperimentBoard(Store(path)).claim(job_id, f'worker-{os.getpid()}')


def _abandoned(path, launching=False):
    board = ExperimentBoard(Store(path))
    job = board.enqueue('abandoned')
    token = board.claim(job, 'departed', ttl=.01)
    if launching:
        board.begin_launch(token)
    return job


def test_competing_workers_claim_once_and_respect_capacity(tmp_path):
    board = ExperimentBoard(Store(tmp_path), dict(threads=2, memory_bytes=100))
    same = board.enqueue('same', dict(threads=1, memory_bytes=60))
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('fork')) as pool:
        claims = list(pool.map(_claim, [str(tmp_path)] * 8, [same] * 8))
    assert sum(token is not None for token in claims) == 1
    jobs = [board.enqueue(f'exp-{i}', dict(threads=1, memory_bytes=50)) for i in range(6)]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('fork')) as pool:
        claims = list(pool.map(_claim, [str(tmp_path)] * len(jobs), jobs))
    assert not any(claims)  # Unexpired/departed owner still reserves 60/100 bytes.
    assert board.snapshot()['reserved'] == dict(threads=1, memory_bytes=60, storage_bytes=0)


def test_concurrent_jobs_cannot_overspend_thread_pool(tmp_path):
    board = ExperimentBoard(Store(tmp_path), dict(threads=2, memory_bytes=100))
    jobs = [board.enqueue(str(i), dict(threads=1, memory_bytes=10)) for i in range(8)]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('fork')) as pool:
        claims = list(pool.map(_claim, [str(tmp_path)] * len(jobs), jobs))
    assert sum(token is not None for token in claims) == 2
    assert board.snapshot()['reserved']['threads'] == 2


def test_capacity_is_immutable_and_attempts_are_separate(tmp_path):
    store = Store(tmp_path)
    board = ExperimentBoard(store, dict(threads=1, memory_bytes=100))
    with pytest.raises(StrataError, match='immutable'):
        ExperimentBoard(Store(tmp_path), dict(threads=2, memory_bytes=100))
    with pytest.raises(StrataError, match='exceeds'):
        board.enqueue('big', dict(memory_bytes=101))
    first, second = board.enqueue('exp'), board.enqueue('exp')
    assert first != second
    token = board.claim(first, 'worker')
    assert board.claim(second, 'worker') is None
    board.finish(token, 'failed')
    assert board.claim(second, 'worker')
    assert all(job['experiment_id'] == 'exp' for job in board.snapshot()['jobs'])


def test_expired_live_owner_never_duplicates_or_releases(tmp_path):
    board = ExperimentBoard(Store(tmp_path), dict(threads=1))
    job = board.enqueue('live')
    token = board.claim(job, 'live-owner', ttl=.001)
    time.sleep(.01)
    assert board.reconcile(job)['status'] == 'unreconciled'
    duplicate = board.enqueue('live')
    other = board.enqueue('other')
    assert board.claim(duplicate, 'duplicate') is None
    assert board.claim(other, 'other') is None
    board.heartbeat(token)
    board.finish(token, 'failed')
    assert board.claim(other, 'other')


def test_departed_owner_reconciliation_requires_identified_launch(tmp_path):
    board = ExperimentBoard(Store(tmp_path), dict(threads=1))
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('fork')) as pool:
        job = pool.submit(_abandoned, str(tmp_path)).result()
    time.sleep(.02)
    assert board.reconcile(job)['status'] == 'lost'
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('fork')) as pool:
        ambiguous = pool.submit(_abandoned, str(tmp_path), True).result()
    time.sleep(.02)
    assert board.reconcile(ambiguous)['status'] == 'unreconciled'
    assert board.snapshot()['reserved']['threads'] == 1


def test_stale_pid_identity_is_never_signalled(tmp_path, monkeypatch):
    board = ExperimentBoard(Store(tmp_path), dict(threads=1))
    job = board.enqueue('stale')
    board.claim(job, 'departed', ttl=.001)
    old = process_identity()
    old['start_ticks'] -= 1
    with board.db:
        board.db.execute('UPDATE experiment_board_jobs SET owner_identity=?,process_identity=? WHERE job_id=?',
                         (json.dumps(old), json.dumps(old), job))
    monkeypatch.setattr(os, 'kill', lambda *args: pytest.fail('reconciliation must never signal'))
    monkeypatch.setattr(os, 'killpg', lambda *args: pytest.fail('reconciliation must never signal'))
    time.sleep(.01)
    assert board.reconcile(job)['status'] == 'lost'


def test_cursors_and_retained_events_do_not_duplicate_completion(tmp_path):
    board = ExperimentBoard(Store(tmp_path), dict(threads=1))
    job = board.enqueue('events')
    token = board.claim(job, 'worker')
    cursor = board.events()[-1]['seq']
    handle = board.store.put('result', dict(status='failed'))
    board.finish(token, 'failed', handle)
    events = board.events(cursor)
    assert len(events) == 1 and events[0]['type'] == 'finished'
    assert events[0]['result_handle'] == handle
    assert board.events(events[-1]['seq']) == []
    with pytest.raises(StrataError, match='terminal'):
        board.finish(token, 'failed', handle)
    assert len(board.store.all('experiment_board_event')) == 3


def test_cancel_queued_attempt_never_launches(tmp_path):
    board = ExperimentBoard(Store(tmp_path), dict(threads=1))
    job = board.enqueue('queued')
    assert board.request_cancel(job)
    assert not board.request_cancel(job)
    assert board.claim(job, 'worker') is None
    assert board.snapshot()['jobs'][0]['status'] == 'cancelled'


def _orphaned(path):
    board = ExperimentBoard(Store(path))
    job = board.enqueue('orphaned')
    token = board.claim(job, 'short-lived-supervisor', ttl=.01)
    board.begin_launch(token)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                             start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    identity = process_identity(child.pid)
    board.bind_process(token, identity)
    return job, identity


def test_supervisor_death_cannot_release_live_orphan(tmp_path):
    board = ExperimentBoard(Store(tmp_path), dict(threads=1))
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('fork')) as pool:
        job, child = pool.submit(_orphaned, str(tmp_path)).result()
    try:
        time.sleep(.02)
        assert board.reconcile(job)['status'] == 'unreconciled'
        assert board.request_cancel(job)
        assert process_state(child) == 'alive'  # A missing owner cannot perform cancellation.
        retry = board.enqueue('orphaned')
        assert board.claim(retry, 'replacement') is None
    finally:
        if process_state(child) == 'alive':
            os.killpg(child['pid'], signal.SIGKILL)
    deadline = time.monotonic() + 2
    while process_state(child) == 'alive' and time.monotonic() < deadline:
        time.sleep(.01)
    assert board.reconcile(job)['status'] == 'lost'
    assert board.claim(retry, 'replacement')
