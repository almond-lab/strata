"""Fail-closed, offline subprocess boundary for third-party generators.

No shell, explicit work directory, finite wall time and bounded output. bwrap
availability and kernel permissions are required; there is no unsafe fallback.
"""
from pathlib import Path
import os
import subprocess
import shutil
import tempfile
import resource
from .model import StrataError


def run(argv, workdir, timeout=60, max_output=1_000_000, memory_bytes=2*1024**3):
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
               '--bind', str(workdir), str(workdir), '--proc', '/proc', '--dev', '/dev',
               '--chdir', str(workdir), '--', *argv]
    env = {k: v for k, v in os.environ.items() if k in {'PATH', 'LANG', 'LC_ALL'}}
    # File output avoids accumulating unbounded captured output in memory.
    def limits():
        resource.setrlimit(resource.RLIMIT_AS,(memory_bytes,memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE,(max_output,max_output))
        resource.setrlimit(resource.RLIMIT_CPU,(max(1,int(timeout)),max(1,int(timeout))+1))
    with tempfile.TemporaryFile() as f:
        try:
            r = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=f, timeout=timeout, preexec_fn=limits)
        except subprocess.TimeoutExpired as exc:
            raise StrataError('sandbox timeout; no receipt') from exc
        size = f.tell()
        if size > max_output:
            raise StrataError('sandbox output exceeds limit')
        f.seek(0)
        output = f.read().decode(errors='replace')
    if r.returncode:
        raise StrataError(f'sandbox failed ({r.returncode}): {output[:1000]}')
    return dict(argv=argv, sandbox='bubblewrap-unshare-all', returncode=0, output=output)
