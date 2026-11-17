"""Fair shared-board scheduling, explicit capacity and bounded resource waits."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import threading
import time
from types import SimpleNamespace

import pytest

from strata import experiments
from strata.experiment_board import ExperimentBoard
from strata.experiments import create_experiment, create_sweep, run_experiment, run_sweep
from strata.model import StrataError
from strata.store import Store
from test_v04_experiments import local_transport, manifest


def _enqueue_and_exit(path):
    store = Store(path)
    try:
        return ExperimentBoard(store).enqueue('departed-submitter')
    finally:
        store.db.close()


def test_inspecting_default_board_does_not_freeze_a_sweeps_capacity(tmp_path):
    original = ExperimentBoard(Store(tmp_path))
    assert original.snapshot()['capacity_source'] == 'default'
    declared = ExperimentBoard(Store(tmp_path), {'threads': 2})
    assert declared.snapshot()['capacity_source'] == 'explicit'
    assert original.capacity == {'threads': 2, 'memory_bytes': 0, 'storage_bytes': 0}
    assert len(original.store.all('experiment_board_capacity')) == 2


def test_capacity_changes_only_when_idle_and_stale_clients_refresh(tmp_path):
    original = ExperimentBoard(Store(tmp_path), {'threads': 2})
    other = ExperimentBoard(Store(tmp_path))
    job = original.enqueue('old-request', {'threads': 2})
    with pytest.raises(StrataError, match='immutable'):
        other.configure_capacity({'threads': 1})
    token = original.claim(job, 'owner')
    assert other.configure_capacity({'threads': 2})['threads'] == 2
    with pytest.raises(StrataError, match='immutable'):
        other.configure_capacity({'threads': 1})
    original.finish(token, 'failed')
    other.configure_capacity({'threads': 1})
    with pytest.raises(StrataError, match='exceeds'):
        original.enqueue('oversized', {'threads': 2})
    first = original.enqueue('first')
    second = other.enqueue('second')
    assert original.claim(first, 'owner')
    assert other.claim(second, 'owner') is None
    assert original.snapshot()['reserved']['threads'] == 1


def test_concurrent_capacity_reduction_and_enqueue_are_atomic(tmp_path):
    board = ExperimentBoard(Store(tmp_path), {'threads': 2})
    barrier = threading.Barrier(2)

    def race(configure):
        local = Store(tmp_path)
        try:
            client = ExperimentBoard(local)
            barrier.wait(timeout=5)
            try:
                if configure:
                    client.configure_capacity({'threads': 1})
                else:
                    client.enqueue('two-threads', {'threads': 2})
                return True
            except StrataError:
                return False
        finally:
            local.db.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(race, [True, False]))
    assert sum(outcomes) == 1
    snapshot = board.snapshot()
    assert all(job['resources']['threads'] <= snapshot['capacity']['threads'] for job in snapshot['jobs'])


def test_large_fifo_request_cannot_be_overtaken_by_small_requests(tmp_path):
    board = ExperimentBoard(Store(tmp_path), {'threads': 8})
    blocker = board.claim(board.enqueue('blocker', {'threads': 6}), 'owner')
    large = board.enqueue('large', {'threads': 7})
    small = [board.enqueue(f'small-{i}') for i in range(8)]

    def attempt(job):
        local = Store(tmp_path)
        try:
            return ExperimentBoard(local).claim(job, 'competing-small-worker')
        finally:
            local.db.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert not any(pool.map(attempt, small))
    assert board.claim(large, 'large-worker') is None
    board.finish(blocker, 'failed')
    assert board.claim(small[0], 'small-worker') is None
    large_token = board.claim(large, 'large-worker')
    assert large_token
    assert board.claim(small[0], 'small-worker')
    assert board.snapshot()['reserved']['threads'] == 8


def test_proven_dead_queued_submitter_is_withdrawn_without_blocking_fifo(tmp_path):
    board = ExperimentBoard(Store(tmp_path), {'threads': 1})
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('fork')) as pool:
        abandoned = pool.submit(_enqueue_and_exit, str(tmp_path)).result()
    live = board.enqueue('still-here')
    assert board.snapshot()['jobs'][0]['status'] == 'cancelled'
    assert board.job(abandoned)['cancel_requested']
    assert board.claim(live, 'live-owner')


def test_filtered_event_cursor_advances_across_heartbeat_only_pages(tmp_path):
    board = ExperimentBoard(Store(tmp_path), {'threads': 1})
    job = board.enqueue('watched')
    token = board.claim(job, 'owner')
    first = board.event_page(types=['claimed'])
    assert [event['type'] for event in first['events']] == ['claimed']
    board.heartbeat(token)
    board.heartbeat(token)
    quiet = board.event_page(first['cursor'], exclude_types=['heartbeat'])
    assert quiet['events'] == [] and quiet['cursor'] > first['cursor']
    board.finish(token, 'failed')
    completed = board.event_page(quiet['cursor'], types=['finished'], exclude_types=['heartbeat'])
    assert [event['job_id'] for event in completed['events']] == [job]
    assert board.event_page(completed['cursor'], exclude_types=['heartbeat']) == {
        'events': [], 'cursor': completed['cursor']}
    assert board.event_page(completed['cursor'] + 10)['cursor'] == completed['cursor'] + 10


def test_claim_timeout_withdraws_request_and_does_not_consume_attempt(tmp_path):
    path = manifest(tmp_path, 'print("not launched")')
    store = Store(tmp_path / 'store')
    capacity = {'threads': 1, 'memory_bytes': 134217728, 'storage_bytes': 1048576}
    board = ExperimentBoard(store, capacity)
    blocker = board.claim(board.enqueue('blocking'), 'other')
    handle = create_experiment(store, path)
    result = run_experiment(store, handle, board=board, claim_timeout=.03)
    assert result['status'] == 'waiting_for_resources' and result['withdrawn']
    assert board.job(result['job'])['status'] == 'cancelled'
    assert store.all('experiment_attempt') == []
    board.finish(blocker, 'failed')
    assert not any(row['state'] == 'queued' for row in board.snapshot()['jobs'])


def test_interrupted_resource_wait_withdraws_request(tmp_path, monkeypatch):
    path = manifest(tmp_path, 'print("must not launch")')
    store = Store(tmp_path / 'store')
    board = ExperimentBoard(store, {'threads': 1, 'memory_bytes': 134217728, 'storage_bytes': 1048576})
    blocker = board.claim(board.enqueue('blocking'), 'other')
    handle = create_experiment(store, path)

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(experiments, 'time', SimpleNamespace(
        monotonic=time.monotonic, sleep=interrupt, time=time.time))
    with pytest.raises(KeyboardInterrupt):
        run_experiment(store, handle, board=board, claim_timeout=10)
    assert board.snapshot()['jobs'][-1]['state'] == 'cancelled'
    assert store.all('experiment_attempt') == []
    board.finish(blocker, 'failed')


def test_external_cancellation_releases_fifo_waiter_promptly(tmp_path):
    path = manifest(tmp_path, 'print("must not launch")')
    store = Store(tmp_path / 'store')
    board = ExperimentBoard(store, {'threads': 1, 'memory_bytes': 134217728, 'storage_bytes': 1048576})
    blocker = board.claim(board.enqueue('blocking'), 'other')
    handle = create_experiment(store, path)

    def wait_for_resources():
        local = Store(store.path)
        try:
            return run_experiment(local, handle, claim_timeout=10)
        finally:
            local.db.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(wait_for_resources)
        deadline = time.monotonic() + 3
        queued = []
        while not queued and time.monotonic() < deadline:
            queued = [job for job in board.snapshot()['jobs'] if job['state'] == 'queued']
            time.sleep(.01)
        assert len(queued) == 1
        board.request_cancel(queued[0]['job_id'])
        assert future.result(timeout=2)['status'] == 'cancelled'
    assert store.all('experiment_attempt') == []
    board.finish(blocker, 'failed')


def test_sweep_waits_past_old_thirty_second_window_on_partly_busy_board(tmp_path, local_transport, monkeypatch):
    # Speed only the scheduler clock. Child supervision still uses real time,
    # so this is a real shared-board contention test with real worker processes.
    real_monotonic, real_sleep = time.monotonic, time.sleep
    monkeypatch.setattr(experiments, 'time', SimpleNamespace(
        monotonic=lambda: real_monotonic() * 100,
        sleep=lambda seconds: real_sleep(seconds / 100), time=time.time))
    path = manifest(tmp_path, 'import time; time.sleep(.45)', outputs=[], metrics={},
                    resources={'threads': 4, 'memory_bytes': 134217728, 'storage_bytes': 1048576},
                    max_wall_seconds=1000)
    store = Store(tmp_path / 'store')
    capacity = {'threads': 8, 'memory_bytes': 536870912, 'storage_bytes': 10485760}
    board = ExperimentBoard(store, capacity)
    blocker = board.claim(board.enqueue('other-team', {'threads': 1}), 'other-team')
    sweep = create_sweep(store, path, [{}], [0, 1, 2], 1, 3, workers=2, capacity=capacity)
    try:
        result = run_sweep(store, sweep)
        assert [row['status'] for row in result['results']] == ['execution_finished'] * 3
        assert len(store.all('experiment_attempt')) == 3
        assert not any(job['state'] == 'queued' for job in board.snapshot()['jobs'])
        events = board.events()
        queued_at = {event['job_id']: event['timestamp'] for event in events if event['type'] == 'enqueued'}
        assert any(event['timestamp'] - queued_at[event['job_id']] > .3 for event in events if event['type'] == 'claimed')
    finally:
        board.finish(blocker, 'failed')


def test_sweep_shared_deadline_withdraws_all_waiters(tmp_path):
    path = manifest(tmp_path, 'print("must not launch")', outputs=[], metrics={}, max_wall_seconds=.1)
    store = Store(tmp_path / 'store')
    capacity = {'threads': 1, 'memory_bytes': 134217728, 'storage_bytes': 1048576}
    board = ExperimentBoard(store, capacity)
    blocker = board.claim(board.enqueue('other-team'), 'other-team')
    sweep = create_sweep(store, path, [{}], [0, 1], 1, 2, workers=2, capacity=capacity)
    before = time.monotonic()
    result = run_sweep(store, sweep)
    assert time.monotonic() - before < 2
    assert all(row['status'] == 'budget_exhausted' for row in result['results'])
    assert store.all('experiment_attempt') == []
    assert not any(job['state'] == 'queued' for job in board.snapshot()['jobs'])
    board.finish(blocker, 'failed')
