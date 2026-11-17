"""Fail-closed, offline subprocess boundary for third-party generators.

Bubblewrap and its kernel permissions are mandatory. Reservations made by the
experiment board are cooperative; these rlimits apply to individual processes.
"""
from pathlib import Path
import os
import subprocess
import shutil
import resource
import math
import re
import selectors
import signal
import time
from .model import StrataError


class _CancelledBeforeLaunch(Exception):
    pass


def process_identity(pid=None):
    """Read Linux process identity; start ticks prevent acting on a reused PID."""
    pid = os.getpid() if pid is None else pid
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        return dict(pid=pid, start_ticks=int(fields[19]), process_group=int(fields[2]),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (OSError, ValueError, IndexError) as exc:
        raise StrataError(f'cannot identify process {pid}') from exc


def process_state(identity):
    """Return alive, dead or unknown without signalling any process."""
    if not isinstance(identity, dict) or not {'pid', 'start_ticks', 'boot_id'} <= identity.keys():
        return 'unknown'
    pid = identity['pid']
    if type(pid) is not int or pid <= 0:
        return 'unknown'
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if boot != identity['boot_id']:
            return 'dead'
        stat = Path(f'/proc/{pid}/stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        if int(fields[19]) != identity['start_ticks'] or fields[0] in {'Z', 'X'}:
            return 'dead'
        return 'alive'
    except FileNotFoundError:
        return 'dead'
    except (OSError, ValueError, IndexError):
        return 'unknown'


def effective_limits(*, timeout=60, max_output=1_000_000, memory_bytes=2*1024**3,
                     max_file_bytes=1024**3, cpu_seconds=None):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise StrataError('sandbox timeout must be positive and finite')
    cpu_seconds = max(1, math.ceil(timeout)) if cpu_seconds is None else cpu_seconds
    values = dict(memory_bytes=memory_bytes, max_file_bytes=max_file_bytes,
                  cpu_seconds=cpu_seconds, max_output=max_output)
    for name, value in values.items():
        if type(value) is float and math.isfinite(value) and value.is_integer():
            value = values[name] = int(value)
        if type(value) is not int or value <= 0:
            raise StrataError(f'sandbox {name} must be a positive integer')
    return dict(timeout=timeout, **values)


def declared_environment(values=None):
    """Accept explicit reproducibility settings, never ambient credentials."""
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise StrataError('sandbox environment must be an explicit mapping')
    result = {}
    for key, value in values.items():
        if (not isinstance(key, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key)
                or not isinstance(value, str) or '\x00' in value):
            raise StrataError('invalid declared sandbox environment')
        if re.search(r'(SECRET|TOKEN|PASSWORD|CREDENTIAL|PRIVATE_KEY|API_KEY|ACCESS_KEY)', key, re.I):
            raise StrataError('secret-like environment names cannot be retained or passed to experiments')
        result[key] = value
    return result


def run(argv, workdir, timeout=60, max_output=1_000_000, memory_bytes=2*1024**3,
        max_file_bytes=1024**3, cpu_seconds=None, readonly_paths=(), *,
        env_overrides=None, environment=None, check=True, on_start=None, poll=None, before_start=None,
        hard_aggregate=False):
    """Run offline; hooks enable persisted ownership without an unsafe fallback.

    ``poll`` may return a cancellation reason. ``check=False`` retains failure
    evidence in the returned receipt; preflight/configuration errors still raise.
    """
    if hard_aggregate:
        raise StrataError('hard aggregate quotas require delegated cgroups; unsupported by this backend')
    configured = effective_limits(timeout=timeout, max_output=max_output, memory_bytes=memory_bytes,
                                  max_file_bytes=max_file_bytes, cpu_seconds=cpu_seconds)
    memory_bytes, max_file_bytes, max_output = (configured[key] for key in ('memory_bytes', 'max_file_bytes', 'max_output'))
    if environment is not None and env_overrides is not None:
        raise StrataError('declare environment only once')
    explicit_env = declared_environment(environment if environment is not None else env_overrides)
    bwrap = shutil.which('bwrap')
    if not bwrap:
        raise StrataError('no-network sandbox requires bubblewrap')
    workdir = Path(workdir).resolve()
    if not workdir.is_dir() or workdir == Path('/'):
        raise StrataError('explicit isolated writable directory required')
    if (not isinstance(argv, (list, tuple)) or not argv
            or any(not isinstance(x, str) or '\x00' in x for x in argv)):
        raise StrataError('command must be an argument list')
    command = [bwrap, '--die-with-parent', '--unshare-all', '--new-session',
               '--ro-bind', '/', '/', '--tmpfs', '/tmp',
               '--bind', str(workdir), str(workdir)]
    for name in readonly_paths:
        path = Path(name).absolute()
        if not path.exists() or path == Path('/') or path == workdir or workdir.is_relative_to(path):
            raise StrataError('invalid sandbox read-only runtime path')
        command += ['--ro-bind', str(path), str(path)]
    command += ['--proc', '/proc', '--dev', '/dev', '--chdir', str(workdir), '--', *argv]
    env = {k: v for k, v in os.environ.items() if k in {'PATH', 'LANG', 'LC_ALL'}}
    env.update(explicit_env)

    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))
        cpu = configured['cpu_seconds']
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))

    started = time.monotonic()
    chunks, streams = [], {'stdout': bytearray(), 'stderr': bytearray()}
    total, reason, identity, process = 0, None, None, None
    error = None
    try:
        if before_start and before_start():
            raise _CancelledBeforeLaunch
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   preexec_fn=limits, start_new_session=True)
        identity = process_identity(process.pid)
        if on_start:
            on_start(identity)
        deadline = started + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, 'stdout')
            selector.register(process.stderr, selectors.EVENT_READ, 'stderr')
            while selector.get_map() or process.poll() is None:
                if poll:
                    cancellation = poll()
                    if cancellation:
                        reason = 'cancelled'
                        break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = 'timed_out'
                    break
                for key, _ in selector.select(min(remaining, .1)):
                    chunk = os.read(key.fd, min(65536, max_output - total + 1))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    kept = chunk[:max_output - total]
                    streams[key.data].extend(kept)
                    chunks.append(kept)
                    total += len(kept)
                    if len(kept) != len(chunk):
                        reason = 'output_limit'
                        break
                if reason:
                    break
    except _CancelledBeforeLaunch:
        reason = 'cancelled'
    except OSError as exc:
        error, reason = str(exc), 'launch_failed' if process is None else 'supervision_failed'
    except BaseException:
        # A failed ownership callback cannot leave an unregistered child running.
        if process is not None and process_state(identity) == 'alive':
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    finally:
        if process is not None:
            # Popen retains the child until wait/poll; the identity check is an
            # additional guard. bwrap's private PID namespace covers descendants.
            if process_state(identity) == 'alive':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            process.stdout.close()
            process.stderr.close()
    code = None if process is None else process.returncode
    output = b''.join(chunks).decode(errors='replace')
    stdout, stderr = (streams[key].decode(errors='replace') for key in ('stdout', 'stderr'))
    signum = -code if code is not None and code < 0 else None
    possible_signal = code - 128 if code is not None and 128 < code <= 192 else None
    status = reason or ('execution_finished' if code == 0 else 'signaled' if signum else 'nonzero_exit')
    diagnosis = dict(kind=status, evidence=[], oom_confirmed=False)
    if possible_signal:
        diagnosis['possible_signal'] = possible_signal
        diagnosis['evidence'].append('exit status resembles 128 + signal; signal cause is unconfirmed')
    if signum:
        diagnosis.update(signal=signum, signal_name=signal.Signals(signum).name if signum in signal.Signals._value2member_map_ else str(signum))
        if signum == signal.SIGKILL and reason is None:
            diagnosis['kind'] = 'unknown_external_or_resource_kill'
            diagnosis['evidence'].append('SIGKILL alone does not establish OOM')
        elif signum == signal.SIGXFSZ:
            diagnosis['evidence'].append(f'file-size limit {max_file_bytes} bytes reached')
        elif signum == signal.SIGXCPU:
            diagnosis['evidence'].append(f'CPU limit {configured["cpu_seconds"]} seconds per process reached')
    for diagnostic in ('MemoryError', 'Failed to allocate', 'Cannot allocate memory'):
        if diagnostic in output:
            diagnosis['evidence'].append(f'{diagnostic}; possible address-space limit {memory_bytes} bytes reached')
    # A verifier can catch a child failure and exit zero. Keep its execution
    # receipt, but expose that failure so it cannot masquerade as a verdict.
    for diagnostic in ('BrokenProcessPool', 'A process in the process pool was terminated abruptly',
                       'concurrent.futures.process.BrokenProcessPool'):
        if diagnostic in output:
            diagnosis['kind'] = 'descendant_process_failed'
            diagnosis['evidence'].append(f'{diagnostic}; child termination cause is unconfirmed')
    if error:
        diagnosis['evidence'].append(error)
    result = dict(argv=list(argv), sandbox='bubblewrap-unshare-all', returncode=code,
                  output=output, stdout=stdout, stderr=stderr, limits=configured,
                  status=status, diagnosis=diagnosis, process=identity,
                  seconds=time.monotonic() - started, environment=explicit_env,
                  enforcement='per-process rlimits; cooperative aggregate reservations',
                  cpu_scope='per process; wall timeout bounds the whole invocation')
    if check and status != 'execution_finished':
        message = (f'sandbox wall timeout {timeout}s reached; no receipt' if status == 'timed_out' else
                   f'sandbox captured output exceeds limit {max_output} bytes' if status == 'output_limit' else
                   f'sandbox failed ({code})')
        detail = '; '.join(diagnosis['evidence'])
        raise StrataError(f'{message}; {detail}; limits={configured}: {output[:1000]}')
    return result
