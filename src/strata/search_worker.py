"""Linux fork boundary for trusted, cancellable constructor proposals.

Callbacks and native circuit objects are inherited, not pickled on entry.
Only plain result data and two explicitly supported cache snapshots cross back.
This is a resource/cancellation boundary, not a security sandbox. Constructors
must not spawn descendants; termination affects only the worker we own.
"""
from fractions import Fraction
import io
from math import isfinite
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import pickle
import sys
import tempfile
from time import monotonic

from .model import StrataError


CONTEXT_CACHE_KEYS = ('joint_result', 'algebraic_result')
MAX_RESULT_BYTES = 32 * 1024 * 1024
MAX_RESULT_NODES = 1_000_000
_POLL_SECONDS = 0.02
_TERMINATE_GRACE_SECONDS = 0.05


class _SerializationError(Exception):
    pass


def _plain_data(value):
    """Reject reducers/cycles; cap traversal before serializing trusted data."""
    pending, active, finished, nodes = [(value, False)], set(), set(), 0
    leaves = {type(None), bool, int, float, str, bytes, Fraction}
    containers = {dict, list, tuple, set, frozenset}
    while pending:
        item, exiting = pending.pop()
        if exiting:
            active.remove(id(item))
            finished.add(id(item))
            continue
        nodes += 1
        if nodes > MAX_RESULT_NODES:
            raise _SerializationError('constructor result exceeds the data-node limit')
        if type(item) in leaves:
            continue
        if type(item) not in containers:
            raise _SerializationError(f'unsupported constructor result type: {type(item).__name__}')
        if id(item) in active:
            raise _SerializationError('cyclic constructor result data is unsupported')
        if id(item) in finished:
            continue
        active.add(id(item))
        pending.append((item, True))
        if type(item) is dict:
            for key, child in item.items():
                pending.extend(((key, False), (child, False)))
        else:
            pending.extend((child, False) for child in item)


class _LimitedWriter:
    def __init__(self, stream):
        self.stream, self.size = stream, 0

    def write(self, block):
        self.size += len(block)
        if self.size > MAX_RESULT_BYTES:
            raise _SerializationError('constructor result exceeds the byte limit')
        return self.stream.write(block)


class _ResultReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('fractions', 'Fraction'):
            return Fraction
        raise _SerializationError('constructor result contains an unsupported object reference')


def _failure(reason, **details):
    return dict(status='unknown', reason=reason, **details)


def _write_result(path, result):
    _plain_data(result)
    pending = path + '.pending'
    with open(pending, 'wb') as stream:
        pickle.dump(result, _LimitedWriter(stream), protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(pending, path)


def _worker(propose, circuit, context, result_path):
    try:
        proposal = propose(circuit, context)
        if type(proposal) is not dict:
            result = (_failure('constructor_invalid_result', expected='proposal dictionary'), {})
        else:
            # Snapshots intentionally include unchanged values. Deletions and
            # changes to boundary/cost/callback context are never propagated.
            result = (proposal, {key: context[key] for key in CONTEXT_CACHE_KEYS if key in context})
    except BaseException as exc:
        result = (_failure('constructor_exception', exception_type=type(exc).__name__, message=str(exc)[:2048]), {})
    try:
        _write_result(result_path, result)
    except BaseException as exc:
        # If even this small error cannot be written, parent reports the missing
        # completed result. Partial files are never interpreted as proposals.
        error = (_failure('constructor_serialization_error', exception_type=type(exc).__name__, message=str(exc)[:2048]), {})
        try:
            _write_result(result_path, error)
        except BaseException:
            pass


def _stop_and_join(process):
    """Reap exactly this owned process, escalating if SIGTERM is ignored."""
    if process.is_alive():
        process.terminate()
        process.join(_TERMINATE_GRACE_SECONDS)
        if process.is_alive():
            process.kill()
    process.join()


def run_bounded_constructor(propose, circuit, context, *, timeout, cancel_file=None):
    """Return ``(proposal, supported_cache_updates, elapsed_seconds)``.

    Successful proposal dictionaries are returned unchanged. Failures have
    ``status='unknown'`` and a specific ``constructor_*`` reason; cache updates
    are empty. The caller may apply successful snapshots using context.update.
    Result data supports exact built-in containers/scalars and Fraction, with
    a 32 MiB serialized limit and one million data references. Deserialization
    is bounded by these limits; the deadline is checked before and after it.

    Wall time includes fork, constructor execution, serialization and transfer.
    A deadline/cancel discards partial work and always reaps the worker. The
    measured elapsed time also includes termination grace and local cleanup.
    """
    started = monotonic()
    if type(timeout) not in (int, float) or not isfinite(timeout) or timeout <= 0:
        raise StrataError('constructor timeout must be finite and positive')
    if not callable(propose) or type(context) is not dict:
        raise StrataError('bounded constructor requires a callable and a context dictionary')
    marker = Path(cancel_file) if cancel_file is not None else None
    deadline = started + timeout
    def cancelled():
        return marker is not None and os.path.lexists(marker)
    if cancelled():
        return _failure('constructor_cancelled'), {}, monotonic() - started
    if not sys.platform.startswith('linux') or 'fork' not in mp.get_all_start_methods():
        return _failure('constructor_worker_unsupported_platform', platform=sys.platform,
                        required='Linux with multiprocessing fork'), {}, monotonic() - started
    proposal, updates = _failure('constructor_worker_missing_result'), {}
    with tempfile.TemporaryDirectory(prefix='strata-constructor-') as directory:
        result_path = str(Path(directory) / 'result.pickle')
        process = mp.get_context('fork').Process(target=_worker, args=(propose, circuit, context, result_path),
                                                 name='strata-constructor')
        started_process = False
        try:
            try:
                process.start()
                started_process = True
            except Exception as exc:
                proposal = _failure('constructor_worker_start_error', exception_type=type(exc).__name__, message=str(exc)[:2048])
            if started_process:
                while True:
                    if cancelled():
                        proposal = _failure('constructor_cancelled')
                        break
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        proposal = _failure('constructor_timeout', timeout=timeout)
                        break
                    if not wait([process.sentinel], timeout=min(_POLL_SECONDS, remaining)):
                        continue
                    process.join()
                    if process.exitcode != 0:
                        proposal = _failure('constructor_worker_crash', exitcode=process.exitcode)
                        break
                    if cancelled():
                        proposal = _failure('constructor_cancelled')
                        break
                    if monotonic() >= deadline:
                        proposal = _failure('constructor_timeout', timeout=timeout)
                        break
                    try:
                        with open(result_path, 'rb') as stream:
                            size = os.fstat(stream.fileno()).st_size
                            if size > MAX_RESULT_BYTES:
                                raise _SerializationError('constructor result exceeds the byte limit')
                            raw = stream.read(MAX_RESULT_BYTES + 1)
                        if len(raw) > MAX_RESULT_BYTES:
                            raise _SerializationError('constructor result exceeds the byte limit')
                        envelope = _ResultReader(io.BytesIO(raw)).load()
                        if (type(envelope) is not tuple or len(envelope) != 2 or
                                type(envelope[0]) is not dict or type(envelope[1]) is not dict or
                                any(key not in CONTEXT_CACHE_KEYS for key in envelope[1])):
                            raise _SerializationError('invalid constructor result envelope')
                        proposal, updates = envelope
                    except FileNotFoundError:
                        proposal = _failure('constructor_worker_missing_result')
                    except Exception as exc:
                        proposal = _failure('constructor_serialization_error', exception_type=type(exc).__name__, message=str(exc)[:2048])
                    if cancelled():
                        proposal, updates = _failure('constructor_cancelled'), {}
                    elif monotonic() >= deadline:
                        proposal, updates = _failure('constructor_timeout', timeout=timeout), {}
                    break
        finally:
            if started_process:
                _stop_and_join(process)
            process.close()
    return proposal, updates, monotonic() - started
