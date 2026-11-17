"""Live experiment status and migration of unclaimed legacy FIFO requests."""
import json
import os
import time

import pytest

from strata.experiment_board import ExperimentBoard
from strata.experiments import create_experiment, experiment_status
from strata.sandbox import process_identity
from strata.store import Store


def unfinished(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(argv=['/usr/bin/true'], limits={'timeout': 3600},
                                        max_wall_seconds=7200)))
    store = Store(tmp_path / 'store')
    handle = create_experiment(store, manifest)
    board = ExperimentBoard(store, {'threads': 2})
    job = board.enqueue(handle)
    token = board.claim(job, 'live-supervisor')
    now = time.time()
    store.put('experiment_attempt', dict(experiment=handle, attempt='active', job=job,
                                         reserved_seconds=3600, started_at=now - 3))
    charge = store.put('experiment_charge', dict(experiment=handle, attempt='active',
                       operation='stage-0', reserved_seconds=3600, started_at=now - 2))
    return store, handle, board, job, token, charge


def test_live_attempt_reports_running_stage_and_elapsed_separately_from_reservation(tmp_path):
    store, handle, board, job, token, charge = unfinished(tmp_path)
    before = board.events()
    result = experiment_status(store, handle)
    assert result['status'] == 'running' and result['latest'] is None
    assert 3 <= result['spent_wall_seconds'] < 30
    assert result['reserved_wall_seconds'] == result['budget_charged_wall_seconds'] == 3600
    assert result['remaining_wall_seconds'] == 3600
    active = result['active_attempts'][0]
    assert active['supervisor_state'] == 'alive' and active['current_operation'] == 'stage-0'
    assert 2 <= active['current_operation_elapsed_seconds'] < 30
    assert board.events() == before
    # Between stages, a live supervisor still owns and advances the attempt.
    store.put('experiment_charge_finish', dict(experiment=handle, attempt='active', charge=charge, seconds=2))
    between = experiment_status(store, handle)
    assert between['status'] == 'running'
    assert between['active_attempts'][0]['current_operation'] is None
    assert between['reserved_wall_seconds'] == 0
    finish = store.put('experiment_finish', dict(experiment=handle, attempt='active',
                                               status='execution_finished', seconds=4))
    board.finish(token, 'execution_finished', finish)
    complete = experiment_status(store, handle)
    assert complete['status'] == 'execution_finished'
    assert complete['active_attempts'] == [] and complete['spent_wall_seconds'] == 4
    assert complete['budget_charged_wall_seconds'] == 4


@pytest.mark.parametrize('identity', ['dead', 'unknown'])
def test_stale_or_unknown_supervisor_retains_charge_and_requires_reconciliation(tmp_path, monkeypatch, identity):
    store, handle, board, job, _, _ = unfinished(tmp_path)
    old = process_identity()
    old['start_ticks'] -= 1
    value = json.dumps(old if identity == 'dead' else {})
    with store.db:
        store.db.execute('UPDATE experiment_board_jobs SET owner_identity=? WHERE job_id=?', (value, job))
    monkeypatch.setattr(os, 'kill', lambda *args: pytest.fail('status cannot signal a process'))
    result = experiment_status(store, handle)
    assert result['status'] == 'reconciliation_required'
    assert result['spent_wall_seconds'] == result['budget_charged_wall_seconds'] == 3600
    assert result['active_attempts'][0]['supervisor_state'] == identity
    assert board.snapshot()['reserved']['threads'] == 1


def test_expired_lease_does_not_misreport_a_live_supervisor_as_dead(tmp_path):
    store, handle, board, job, _, _ = unfinished(tmp_path)
    with store.db:
        store.db.execute('UPDATE experiment_board_jobs SET lease_until=0 WHERE job_id=?', (job,))
    result = experiment_status(store, handle)
    assert result['status'] == 'running' and result['active_attempts'][0]['lease_expired']
    assert board.snapshot()['reserved']['threads'] == 1


def legacy_request(board, name='legacy'):
    job = board.enqueue(name, {'threads': 7})
    with board.db:
        board.db.execute('UPDATE experiment_board_jobs SET owner_identity=NULL WHERE job_id=?', (job,))
    return job


def test_legacy_unowned_queue_rows_are_audited_and_no_longer_block_fifo(tmp_path):
    board = ExperimentBoard(Store(tmp_path), {'threads': 8})
    old = [legacy_request(board, str(i)) for i in range(3)]
    current = board.enqueue('new', {'threads': 2})
    assert board.claim(current, 'new-supervisor')
    snapshot = board.snapshot()
    assert snapshot['reserved']['threads'] == 2
    assert all(board.job(job)['status'] == 'cancelled' for job in old)
    cancelled = [event for event in board.events() if event['type'] == 'cancelled']
    assert {event['job_id'] for event in cancelled} == set(old)
    assert all('legacy queued request' in event['reason'] for event in cancelled)
    before = len(board.events())
    board.snapshot()
    assert len(board.events()) == before


def test_legacy_queue_cleanup_allows_idle_capacity_reconfiguration(tmp_path):
    board = ExperimentBoard(Store(tmp_path), {'threads': 8})
    legacy = legacy_request(board)
    assert board.configure_capacity({'threads': 2})['threads'] == 2
    assert board.job(legacy)['status'] == 'cancelled'


@pytest.mark.parametrize('field,value', [
    ('owner_identity', '{}'), ('owner', 'unknown-supervisor'), ('token_hash', 'unknown-token'),
    ('process_phase', 'launching'), ('process_identity', '{}'), ('heartbeat', 1),
    ('lease_until', 1), ('result_handle', 'result')])
def test_ambiguous_legacy_queue_evidence_never_auto_releases(tmp_path, field, value):
    board = ExperimentBoard(Store(tmp_path), {'threads': 8})
    legacy = legacy_request(board)
    with board.db:
        board.db.execute(f'UPDATE experiment_board_jobs SET {field}=? WHERE job_id=?', (value, legacy))
    current = board.enqueue('new')
    assert board.claim(current, 'new-supervisor') is None
    assert board.job(legacy)['state'] == 'queued'


def test_retained_claim_evidence_prevents_legacy_cleanup(tmp_path):
    board = ExperimentBoard(Store(tmp_path), {'threads': 8})
    legacy = legacy_request(board)
    with board._transaction():
        board._event(legacy, 'launching')
    assert board.snapshot()['jobs'][0]['state'] == 'queued'


def test_live_queue_and_active_claims_survive_legacy_cleanup(tmp_path):
    board = ExperimentBoard(Store(tmp_path), {'threads': 8})
    active = board.enqueue('active')
    board.claim(active, 'live-supervisor')
    legacy = legacy_request(board)
    live = board.enqueue('live-request', {'threads': 7})
    later = board.enqueue('later')
    assert board.claim(later, 'younger') is None
    assert board.job(active)['state'] == 'claimed'
    assert board.job(legacy)['state'] == 'cancelled'
    assert board.job(live)['state'] == 'queued'
    assert board.claim(live, 'live-request-supervisor')
