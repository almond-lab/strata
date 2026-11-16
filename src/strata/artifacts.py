"""Pin single-file and multi-file tasks without reading private data into reports."""
from pathlib import Path
import hashlib
from .model import digest, StrataError


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def manifest(path):
    root=Path(path).resolve()
    files=[root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file())
    if not files:raise StrataError('empty or missing artifact')
    rows={}
    for p in files:
        if p.is_symlink() or (root.is_dir() and not p.resolve().is_relative_to(root)):
            raise StrataError('artifact symlinks are not accepted')
        name=p.name if root.is_file() else str(p.relative_to(root))
        rows[name]=dict(sha256=file_hash(p),bytes=p.stat().st_size)
    return dict(files=rows,sha256=digest(rows))


def verify_manifest(path,expected):
    if manifest(path)!=expected:raise StrataError('input artifact changed')
    return True
