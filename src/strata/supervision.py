"""Persisted ownership for offline local attempts, with cooperative cancellation.

A request changes the board. Only the process that owns the lease signals its
own sandbox. A missing supervisor remains visible for explicit reconciliation.
"""
import json
import os
import time
from . import sandbox
from .model import StrataError


def run_owned(board, token, argv, workdir, **sandbox_limits):
    """Run one stage and retain evidence; artifact validation/finish is separate.

    A token may run sequential stages, including a verifier. It must not be used
    concurrently. The caller finishes the attempt only after output checks.
    """
    for name in ('check', 'on_start', 'poll', 'before_start'):
        if name in sandbox_limits:
            raise StrataError(f'owned supervision controls sandbox hook {name}')
    if sandbox_limits.get('hard_aggregate'):
        raise StrataError('hard aggregate quotas require delegated cgroups; unsupported by this backend')
    job = board._owned(token)
    resources = json.loads(job['resources'])
    affinity = None
    heartbeat_at = 0.0

    def before_start():
        board.heartbeat(token)
        if board.cancel_requested(token):
            return True
        board.begin_launch(token)

    def on_start(identity):
        nonlocal affinity
        board.bind_process(token, identity)
        try:
            affinity = sorted(os.sched_getaffinity(identity['pid']))
        except (AttributeError, OSError):
            affinity = None

    def poll():
        nonlocal heartbeat_at
        now = time.monotonic()
        if now >= heartbeat_at:
            board.heartbeat(token)
            heartbeat_at = now + 5.0
        return board.cancel_requested(token)

    result = sandbox.run(argv, workdir, check=False, on_start=on_start, poll=poll,
                         before_start=before_start, **sandbox_limits)
    result.update(job_id=job['job_id'], attempt_id=job['job_id'],
                  experiment_id=job['experiment_id'], owner=job['owner'],
                  supervisor=sandbox.process_identity(), requested_resources=resources,
                  granted_slots=resources['threads'], effective_affinity=affinity,
                  usage=dict(wall_seconds=result['seconds'], cpu_seconds=None,
                             peak_memory_bytes=None, measurement='wall time only'))
    # Blobs retain the complete bounded logs, including failures. Immutable
    # execution records survive a wrapper crash before final artifact validation.
    result['stdout_handle'] = board.store.blob(result['stdout'].encode())
    result['stderr_handle'] = board.store.blob(result['stderr'].encode())
    handle = board.store.put('owned_execution', result)
    board.record_execution(token, handle)
    result['execution_handle'] = handle
    return result
