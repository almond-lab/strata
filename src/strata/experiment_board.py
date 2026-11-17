"""Transactional cooperative reservations for participating workers on one host.

The mutable tables are indexed projections. Every transition is also an immutable
Store record. Expiry is evidence to investigate, never authority to relaunch or
kill an unidentified process.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import time
import uuid
from types import MappingProxyType

from .model import StrataError, canonical, digest
from .sandbox import process_identity, process_state

_RESOURCE_KEYS = ('threads', 'memory_bytes', 'storage_bytes')
_ACTIVE = ('claimed', 'running', 'unreconciled')
_TERMINAL = {'execution_finished', 'artifact_validated', 'accepted_by_verifier',
             'failed', 'cancelled', 'budget_exhausted', 'lost', 'screened_out',
             'artifact_validation_failed', 'verification_failed', 'rejected_by_verifier',
             'paused_by_forecast_policy'}


def _resources(value, *, capacity=False):
    if not isinstance(value, dict) or set(value) - set(_RESOURCE_KEYS):
        raise StrataError('resources require threads, memory_bytes and storage_bytes; hard aggregate quotas are unsupported')
    result = {name: value.get(name, 1 if name == 'threads' else 0) for name in _RESOURCE_KEYS}
    if any(type(v) is not int or v < 0 for v in result.values()) or result['threads'] < 1:
        raise StrataError('resource reservations must be nonnegative integers with positive threads')
    return result


def _host_identity():
    try:
        return Path('/etc/machine-id').read_text().strip()
    except OSError:
        return socket.gethostname()


class ExperimentBoard:
    def __init__(self, store, capacity=None):
        self.store = store
        self.db = store.db
        self.host = _host_identity()
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS experiment_board_config (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1), host TEXT NOT NULL,
                capacity TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS experiment_board_jobs (
                job_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL,
                state TEXT NOT NULL, resources TEXT NOT NULL,
                owner TEXT, token_hash TEXT UNIQUE, owner_identity TEXT,
                process_identity TEXT, process_phase TEXT NOT NULL DEFAULT 'not_started',
                heartbeat REAL, lease_until REAL, cancel_requested INTEGER NOT NULL DEFAULT 0,
                result_handle TEXT, created REAL NOT NULL, updated REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS experiment_board_experiment
                ON experiment_board_jobs(experiment_id, state);
            CREATE TABLE IF NOT EXISTS experiment_board_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                record_handle TEXT NOT NULL);
        ''')
        with self._transaction():
            row = self.db.execute('SELECT host, capacity FROM experiment_board_config WHERE singleton=1').fetchone()
            if row:
                if row[0] != self.host:
                    raise StrataError('experiment board belongs to another host')
                self.capacity = json.loads(row[1])
                if capacity is not None and _resources(capacity, capacity=True) != self.capacity:
                    raise StrataError('experiment board capacity is immutable')
            else:
                default = dict(threads=os.cpu_count() or 1,
                               memory_bytes=os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'),
                               storage_bytes=shutil.disk_usage(store.path).free)
                self.capacity = _resources(default if capacity is None else capacity, capacity=True)
                self.db.execute('INSERT INTO experiment_board_config VALUES (1,?,?)',
                                (self.host, canonical(self.capacity)))
                self._retain('experiment_board_capacity', dict(host=self.host, capacity=self.capacity))
        self.capacity = MappingProxyType(self.capacity)

    @contextmanager
    def _transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
        except BaseException:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def _retain(self, kind, value):
        """Store.put semantics without committing the containing transaction."""
        handle = digest(dict(kind=kind, body=value))
        self.db.execute('INSERT OR IGNORE INTO records VALUES (?,?,?)', (handle, kind, canonical(value)))
        return handle

    def _event(self, job_id, event, **details):
        body = dict(job_id=job_id, type=event, timestamp=time.time(), **details)
        handle = self._retain('experiment_board_event', body)
        self.db.execute('INSERT INTO experiment_board_events(job_id,record_handle) VALUES (?,?)', (job_id, handle))

    def _row(self, job_id):
        cursor = self.db.execute('SELECT * FROM experiment_board_jobs WHERE job_id=?', (job_id,))
        row = cursor.fetchone()
        if row is None:
            raise StrataError(f'unknown experiment attempt: {job_id}')
        return dict(zip((c[0] for c in cursor.description), row))

    def _owned(self, token):
        if not isinstance(token, str) or not token:
            raise StrataError('invalid experiment ownership token')
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        row = self.db.execute('SELECT job_id FROM experiment_board_jobs WHERE token_hash=?', (token_hash,)).fetchone()
        if row is None:
            raise StrataError('unknown or revoked experiment ownership token')
        job = self._row(row[0])
        if job['state'] not in _ACTIVE:
            raise StrataError('experiment attempt is terminal')
        # Token possession alone cannot authorize a different local process to
        # take over cancellation or launch after a worker crash.
        owner = json.loads(job['owner_identity'])
        current = process_identity()
        if any(owner.get(k) != current.get(k) for k in ('pid', 'start_ticks', 'boot_id')):
            raise StrataError('ownership token belongs to another supervisor process')
        return job

    @staticmethod
    def _ttl(ttl):
        if type(ttl) not in (int, float) or not math.isfinite(ttl) or ttl <= 0:
            raise StrataError('lease ttl must be positive and finite')

    def enqueue(self, experiment_id, resources=None):
        if not isinstance(experiment_id, str) or not experiment_id:
            raise StrataError('experiment identity must be a nonempty string')
        requested = _resources({} if resources is None else resources)
        if any(requested[k] > self.capacity[k] for k in _RESOURCE_KEYS):
            raise StrataError('attempt reservation exceeds immutable board capacity')
        job_id, now = uuid.uuid4().hex, time.time()
        with self._transaction():
            self.db.execute('''INSERT INTO experiment_board_jobs
                (job_id,experiment_id,state,resources,created,updated) VALUES (?,?,'queued',?,?,?)''',
                (job_id, experiment_id, canonical(requested), now, now))
            self._event(job_id, 'enqueued', experiment_id=experiment_id, attempt_id=job_id, resources=requested)
        return job_id

    def claim(self, job_id, owner, ttl=30):
        self._ttl(ttl)
        if not isinstance(owner, str) or not owner:
            raise StrataError('claim owner must be a nonempty name')
        with self._transaction():
            job = self._row(job_id)
            if job['state'] != 'queued':
                return None
            if self.db.execute('''SELECT 1 FROM experiment_board_jobs
                WHERE experiment_id=? AND state IN ('claimed','running','unreconciled') LIMIT 1''',
                (job['experiment_id'],)).fetchone():
                return None
            usage = {name: 0 for name in _RESOURCE_KEYS}
            for (body,) in self.db.execute("SELECT resources FROM experiment_board_jobs WHERE state IN ('claimed','running','unreconciled')"):
                for name, value in json.loads(body).items():
                    usage[name] += value
            requested = json.loads(job['resources'])
            if any(usage[name] + requested[name] > self.capacity[name] for name in _RESOURCE_KEYS):
                return None
            token, now, identity = uuid.uuid4().hex + uuid.uuid4().hex, time.time(), process_identity()
            self.db.execute('''UPDATE experiment_board_jobs SET state='claimed', owner=?,
                token_hash=?, owner_identity=?, heartbeat=?, lease_until=?,updated=? WHERE job_id=?''',
                (owner, hashlib.sha256(token.encode()).hexdigest(), canonical(identity), now, now+ttl, now, job_id))
            self._event(job_id, 'claimed', owner=owner, owner_identity=identity, resources=requested,
                        granted_slots=requested['threads'], lease_until=now+ttl)
            return token

    def heartbeat(self, token, ttl=30):
        self._ttl(ttl)
        with self._transaction():
            job, now = self._owned(token), time.time()
            self.db.execute('UPDATE experiment_board_jobs SET heartbeat=?,lease_until=?,updated=? WHERE job_id=?',
                            (now, now+ttl, now, job['job_id']))
            self._event(job['job_id'], 'heartbeat', owner=job['owner'], lease_until=now+ttl)
        return now + ttl

    def begin_launch(self, token):
        with self._transaction():
            job = self._owned(token)
            if job['process_phase'] == 'launching':
                raise StrataError('previous launch is not yet identified')
            if job['process_identity'] and process_state(json.loads(job['process_identity'])) != 'dead':
                raise StrataError('previous owned process is still alive or unidentified')
            self.db.execute("UPDATE experiment_board_jobs SET process_phase='launching',process_identity=NULL,updated=? WHERE job_id=?",
                            (time.time(), job['job_id']))
            self._event(job['job_id'], 'launching')

    def bind_process(self, token, identity):
        if process_state(identity) == 'unknown':
            raise StrataError('cannot bind an unidentified process')
        if identity.get('process_group') != identity.get('pid'):
            raise StrataError('owned process must lead its isolated process group')
        with self._transaction():
            job = self._owned(token)
            self.db.execute("UPDATE experiment_board_jobs SET state='running',process_identity=?,process_phase='bound',updated=? WHERE job_id=?",
                            (canonical(identity), time.time(), job['job_id']))
            self._event(job['job_id'], 'process_bound', process=identity, owner=job['owner'])

    def record_execution(self, token, result_handle):
        self.store.get(result_handle, 'owned_execution')
        with self._transaction():
            job = self._owned(token)
            self.db.execute("UPDATE experiment_board_jobs SET process_phase='execution_finished',updated=? WHERE job_id=?",
                            (time.time(), job['job_id']))
            self._event(job['job_id'], 'execution_finished', result_handle=result_handle)

    def request_cancel(self, job_id):
        with self._transaction():
            job = self._row(job_id)
            if job['state'] in _TERMINAL or job['cancel_requested']:
                return False
            now = time.time()
            if job['state'] == 'queued':
                self.db.execute("UPDATE experiment_board_jobs SET state='cancelled',cancel_requested=1,updated=? WHERE job_id=?", (now, job_id))
                self._event(job_id, 'cancelled', reason='cancelled before claim')
            else:
                self.db.execute('UPDATE experiment_board_jobs SET cancel_requested=1,updated=? WHERE job_id=?', (now, job_id))
                self._event(job_id, 'cancel_requested')
        return True

    def cancel_requested(self, token):
        return bool(self._owned(token)['cancel_requested'])

    def finish(self, token, status, result_handle=None):
        if status not in _TERMINAL:
            raise StrataError(f'unsupported terminal attempt status: {status}')
        if result_handle is not None:
            self.store.get(result_handle)
        with self._transaction():
            job = self._owned(token)
            if job['process_identity'] and process_state(json.loads(job['process_identity'])) != 'dead':
                raise StrataError('cannot release a reservation while its process may be live')
            if job['process_phase'] == 'launching':
                raise StrataError('unidentified launch must be reconciled before releasing its reservation')
            self.db.execute('UPDATE experiment_board_jobs SET state=?,result_handle=?,updated=? WHERE job_id=?',
                            (status, result_handle, time.time(), job['job_id']))
            self._event(job['job_id'], 'finished', status=status, result_handle=result_handle)
        return job['job_id']

    def reconcile(self, job_id):
        with self._transaction():
            job = self._row(job_id)
            if job['state'] not in _ACTIVE:
                return self._public(job)
            if job['lease_until'] > time.time():
                raise StrataError('cannot reconcile an unexpired lease')
            owner_state = process_state(json.loads(job['owner_identity']))
            child_state = process_state(json.loads(job['process_identity'])) if job['process_identity'] else (
                'unknown' if job['process_phase'] == 'launching' else 'dead')
            status = 'lost' if owner_state == child_state == 'dead' else 'unreconciled'
            self.db.execute('UPDATE experiment_board_jobs SET state=?,updated=? WHERE job_id=?',
                            (status, time.time(), job_id))
            self._event(job_id, 'reconciled', status=status, owner_state=owner_state, process_state=child_state,
                        evidence='Linux boot id and process start ticks; no signals sent')
            return self._public(self._row(job_id))

    def _public(self, row):
        value = {k: v for k, v in row.items() if k != 'token_hash'}
        for key in ('resources', 'owner_identity', 'process_identity'):
            value[key] = json.loads(value[key]) if value[key] is not None else None
        value['attempt_id'] = value['job_id']
        value['status'] = value['state']
        value['lease_expired'] = value['state'] in _ACTIVE and value['lease_until'] <= time.time()
        value['cancel_requested'] = bool(value['cancel_requested'])
        value['granted_slots'] = value['resources']['threads'] if value['state'] in _ACTIVE else 0
        return value

    def snapshot(self):
        cursor = self.db.execute('SELECT * FROM experiment_board_jobs ORDER BY created,job_id')
        columns = [column[0] for column in cursor.description]
        jobs = [self._public(dict(zip(columns, row))) for row in cursor.fetchall()]
        reserved = {name: sum(j['resources'][name] for j in jobs if j['state'] in _ACTIVE) for name in _RESOURCE_KEYS}
        return dict(host=self.host, capacity=dict(self.capacity), reserved=reserved, jobs=jobs,
                    reservation_mode='cooperative', hard_aggregate_enforcement=False)

    def events(self, after=0):
        if type(after) is not int or after < 0:
            raise StrataError('event cursor must be a nonnegative integer')
        rows = self.db.execute('SELECT seq,record_handle FROM experiment_board_events WHERE seq>? ORDER BY seq', (after,)).fetchall()
        return [dict(self.store.get(handle, 'experiment_board_event'), seq=seq, record_handle=handle) for seq, handle in rows]
