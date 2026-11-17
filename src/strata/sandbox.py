"""Fail-closed, offline subprocess boundary for third-party generators.

No shell, explicit work directory, finite wall time and bounded output. bwrap
availability and kernel permissions are required; there is no unsafe fallback.
"""
from pathlib import Path
import os
import subprocess
import shutil
import resource
import math
import selectors
import signal
import time
from .model import StrataError


def effective_limits(*, timeout=60, max_output=1_000_000, memory_bytes=2*1024**3,
                     max_file_bytes=1024**3, cpu_seconds=None):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise StrataError('sandbox timeout must be positive and finite')
    cpu_seconds = max(1, math.ceil(timeout)) if cpu_seconds is None else cpu_seconds
    values = dict(memory_bytes=memory_bytes, max_file_bytes=max_file_bytes,
                  cpu_seconds=cpu_seconds, max_output=max_output)
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise StrataError('sandbox resource limits must be positive integers')
    return dict(timeout=timeout, **values)


def run(argv, workdir, timeout=60, max_output=1_000_000, memory_bytes=2*1024**3,
        max_file_bytes=1024**3, cpu_seconds=None, readonly_paths=()):
    configured = effective_limits(timeout=timeout, max_output=max_output, memory_bytes=memory_bytes,
                                  max_file_bytes=max_file_bytes, cpu_seconds=cpu_seconds)
    bwrap = shutil.which('bwrap')
    if not bwrap:
        raise StrataError('no-network sandbox requires bubblewrap')
    workdir = Path(workdir).resolve()
    if not workdir.is_dir() or workdir == Path('/'):
        raise StrataError('explicit isolated writable directory required')
    if not argv or any(not isinstance(x, str) for x in argv):
        raise StrataError('command must be an argument list')
    # Read-only root permits dependency loading; private caller files should be
    # kept outside untrusted-host environments. Network is unshared explicitly.
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
    def limits():
        resource.setrlimit(resource.RLIMIT_AS,(memory_bytes,memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE,(max_file_bytes,max_file_bytes))
        cpu = configured['cpu_seconds']
        resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu+1))
    # A pipe keeps captured output independent of verifier-created file limits.
    # Drain incrementally and stop at the configured capture budget.
    output = bytearray()
    process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, preexec_fn=limits, start_new_session=True)
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise StrataError(f'sandbox wall timeout {timeout}s reached; no receipt')
                for key, _ in selector.select(min(remaining, .25)):
                    chunk = os.read(key.fd, min(65536,max_output-len(output)+1))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    output.extend(chunk)
                    if len(output)>max_output:
                        raise StrataError(f'sandbox captured output exceeds limit {max_output} bytes')
            process.wait(timeout=max(.001,deadline-time.monotonic()))
    except subprocess.TimeoutExpired as exc:
        raise StrataError(f'sandbox wall timeout {timeout}s reached; no receipt') from exc
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
        process.stdout.close()
    output = output.decode(errors='replace')
    if process.returncode:
        hint = ''
        if 'MemoryError' in output or 'Failed to allocate' in output or 'Cannot allocate memory' in output:
            hint = f'; possible sandbox address-space limit {memory_bytes} bytes reached'
        if process.returncode in (-signal.SIGXFSZ,128+signal.SIGXFSZ):
            hint = f'; sandbox file-size limit {max_file_bytes} bytes reached'
        if process.returncode in (-signal.SIGXCPU,128+signal.SIGXCPU):
            hint = f'; sandbox CPU limit {configured["cpu_seconds"]} seconds per process reached'
        raise StrataError(f'sandbox failed ({process.returncode}){hint}; limits={configured}: {output[:1000]}')
    return dict(argv=argv, sandbox='bubblewrap-unshare-all', returncode=0, output=output,
                limits=configured, cpu_scope='per process; wall timeout bounds the whole invocation')
