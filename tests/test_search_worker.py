from fractions import Fraction
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time

import kickmix as km
import pytest

from strata.model import StrataError
from strata.search_worker import run_bounded_constructor


pytestmark = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux fork worker')


def _assert_reaped(path):
    pid = int(path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert pid not in {process.pid for process in mp.active_children()}


def test_closures_native_circuits_proofs_and_supported_cache_snapshots_survive():
    prefix = 'closed-over'
    context = dict(joint_result={'old': True}, clean_inputs=[3], price=lambda circuit: circuit.num_qubits)
    def propose(circuit, child_context):
        child_context['joint_result']['old'] = False
        child_context['algebraic_result'] = {'status': 'constructed', 'nested': [1, (2, 3)]}
        child_context['clean_inputs'].append(99)
        child_context['price'] = None
        child_context['forbidden_new_boundary'] = True
        return dict(status='constructed', candidate=str(circuit), proof={'label': prefix,
            'phase': Fraction(1, 2), 'terms': {frozenset({1, 2})}, 'exact': True})
    result, updates, elapsed = run_bounded_constructor(propose, km.Circuit('CCX q0 q1 q2'), context, timeout=3)
    assert result == dict(status='constructed', candidate='CCX q0 q1 q2',
                         proof={'label': prefix, 'phase': Fraction(1, 2), 'terms': {frozenset({1, 2})}, 'exact': True})
    assert updates == {'joint_result': {'old': False}, 'algebraic_result': {'status': 'constructed', 'nested': [1, (2, 3)]}}
    assert context['joint_result'] == {'old': True} and context['clean_inputs'] == [3]
    assert context['price'](km.Circuit('X q4')) == 5
    assert elapsed >= 0


def test_large_result_does_not_block_on_pipe_capacity():
    payload = b'proof-evidence\0' * 400_000
    result, updates, _ = run_bounded_constructor(lambda circuit, context: {
        'status': 'constructed', 'candidate': 'X q0', 'proof': {'exact': True, 'payload': payload}}, None, {}, timeout=5)
    assert result['status'] == 'constructed' and result['proof']['payload'] == payload
    assert updates == {}


def test_deadline_kills_and_reaps_a_sigterm_ignoring_worker(tmp_path):
    pid_file = tmp_path / 'pid'
    def propose(circuit, context):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pid_file.write_text(str(os.getpid()))
        while True:
            time.sleep(1)
    result, updates, elapsed = run_bounded_constructor(propose, None, {}, timeout=.15)
    assert result['reason'] == 'constructor_timeout' and updates == {}
    assert .15 <= elapsed < 3
    _assert_reaped(pid_file)


def test_marker_created_while_running_cancels_only_owned_worker(tmp_path):
    pid_file, marker = tmp_path / 'pid', tmp_path / 'cancel'
    def propose(circuit, context):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pid_file.write_text(str(os.getpid()))
        while True:
            time.sleep(1)
    def cancel_after_start():
        deadline = time.monotonic() + 3
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(.005)
        marker.touch()
        time.sleep(30)
    unrelated = mp.get_context('fork').Process(target=cancel_after_start)
    unrelated.start()
    try:
        result, updates, elapsed = run_bounded_constructor(propose, None, {}, timeout=5, cancel_file=marker)
        assert result['reason'] == 'constructor_cancelled' and updates == {}
        assert elapsed < 3 and unrelated.is_alive()
        _assert_reaped(pid_file)
    finally:
        unrelated.terminate()
        unrelated.join(3)
        unrelated.close()


def test_preexisting_cancel_marker_never_invokes_constructor(tmp_path):
    marker, touched = tmp_path / 'cancel', tmp_path / 'touched'
    marker.touch()
    def propose(circuit, context):
        touched.touch()
        return {'status': 'constructed'}
    result, updates, elapsed = run_bounded_constructor(propose, None, {}, timeout=1, cancel_file=marker)
    assert result['reason'] == 'constructor_cancelled' and updates == {}
    assert elapsed >= 0 and not touched.exists()


@pytest.mark.parametrize('mode,reason', [
    ('exception', 'constructor_exception'), ('crash', 'constructor_worker_crash'),
    ('empty_exit', 'constructor_worker_missing_result'), ('non_dictionary', 'constructor_invalid_result'),
    ('unsupported_data', 'constructor_serialization_error'), ('cycle', 'constructor_serialization_error')])
def test_failures_return_no_context_mutations(mode, reason):
    def propose(circuit, context):
        context['joint_result'] = {'partial': True}
        if mode == 'exception':
            raise ValueError('precise constructor failure')
        if mode == 'crash':
            os._exit(7)
        if mode == 'empty_exit':
            os._exit(0)
        if mode == 'non_dictionary':
            return None
        if mode == 'unsupported_data':
            return {'status': 'constructed', 'callback': lambda: 1}
        result = {'status': 'constructed'}
        result['cycle'] = result
        return result
    context = {}
    result, updates, _ = run_bounded_constructor(propose, None, context, timeout=2)
    assert result['status'] == 'unknown' and result['reason'] == reason
    assert updates == {} and context == {}
    if mode == 'crash':
        assert result['exitcode'] == 7
    if mode == 'exception':
        assert result['exception_type'] == 'ValueError' and result['message'] == 'precise constructor failure'


def test_serialization_is_inside_deadline_and_oversize_receipts_are_explicit(monkeypatch):
    import strata.search_worker as worker
    original = worker._write_result
    def slow(*args):
        time.sleep(30)
        original(*args)
    monkeypatch.setattr(worker, '_write_result', slow)
    result, updates, elapsed = run_bounded_constructor(lambda *args: {'status': 'constructed'}, None, {}, timeout=.08)
    assert result['reason'] == 'constructor_timeout' and updates == {} and elapsed < 3
    monkeypatch.setattr(worker, '_write_result', original)
    monkeypatch.setattr(worker, 'MAX_RESULT_BYTES', 1024)
    result, updates, _ = run_bounded_constructor(lambda *args: {'status': 'constructed', 'proof': b'x' * 2048},
                                                None, {}, timeout=2)
    assert result['reason'] == 'constructor_serialization_error' and 'byte limit' in result['message']
    assert updates == {}


def test_platform_and_budget_errors_are_explicit(monkeypatch):
    import strata.search_worker as worker
    for timeout in (0, -1, True, float('nan'), float('inf')):
        with pytest.raises(StrataError, match='timeout'):
            run_bounded_constructor(lambda *args: {}, None, {}, timeout=timeout)
    monkeypatch.setattr(worker.sys, 'platform', 'darwin')
    result, updates, elapsed = run_bounded_constructor(lambda *args: {}, None, {}, timeout=1)
    assert result['reason'] == 'constructor_worker_unsupported_platform'
    assert updates == {} and elapsed >= 0
