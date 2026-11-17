"""Transactional immutable records and content-addressed artifacts."""
from pathlib import Path
import hashlib
import json
import sqlite3
import tempfile
import os
import stat
from .model import canonical, digest, StrataError


def _stream_file(path, destination=None):
    """Hash a regular file, optionally copying it, with bounded memory."""
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise StrataError('artifact source must be a regular file')
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, 'rb') as source:
        value = hashlib.sha256()
        size = 0
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
            size += len(block)
            if destination is not None:
                destination.write(block)
        after = os.fstat(source.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after) or size != before.st_size:
            raise StrataError('artifact changed while being read')
        return value.hexdigest()


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Store:
    def __init__(self, path):
        self.path = Path(path).resolve()
        (self.path / 'objects').mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path / 'index.sqlite', timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, line TEXT NOT NULL, body TEXT NOT NULL)')
        self.db.commit()

    def blob(self, data):
        h = hashlib.sha256(data).hexdigest()
        p = self.path / 'objects' / h
        if not p.exists():
            fd, name = tempfile.mkstemp(dir=p.parent)
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, p)
        elif self.read_blob(h) != data:
            raise StrataError('hash collision')
        return h

    def read_blob(self, h):
        if not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
            raise StrataError('invalid object hash')
        data = (self.path / 'objects' / h).read_bytes()
        if hashlib.sha256(data).hexdigest() != h:
            raise StrataError('artifact integrity failure')
        return data

    def _blob_path(self, h):
        if not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
            raise StrataError('invalid object hash')
        return self.path / 'objects' / h

    def blob_file(self, path):
        """Retain a file without loading its contents into memory.

        The source is hashed while copied to a temporary object. Only a complete,
        fsynced copy is published under its content hash.
        """
        directory = self.path / 'objects'
        fd, name = tempfile.mkstemp(dir=directory)
        try:
            with os.fdopen(fd, 'wb') as destination:
                handle = _stream_file(path, destination)
                destination.flush()
                os.fsync(destination.fileno())
            target = self._blob_path(handle)
            if target.exists():
                self.verify_blob(handle)
            else:
                os.replace(name, target)
                _sync_directory(directory)
            return handle
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def verify_blob(self, h):
        """Verify an object's filename and bytes using bounded memory."""
        if _stream_file(self._blob_path(h)) != h:
            raise StrataError('artifact integrity failure')
        return True

    def copy_blob(self, h, destination):
        """Verify before atomic replacement; never publish a partial copy."""
        source = self._blob_path(h)
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=target.parent, prefix=f'.{target.name}.')
        try:
            with os.fdopen(fd, 'wb') as output:
                if _stream_file(source, output) != h:
                    raise StrataError('artifact integrity failure')
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, target)
            _sync_directory(target.parent)
            return target
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def put(self, kind, value):
        body = canonical(value)
        h = digest(dict(kind=kind, body=value))
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO records VALUES (?,?,?)', (h, kind, body))
        return h

    def get(self, h, kind=None):
        row = self.db.execute('SELECT kind,body FROM records WHERE id=?', (h,)).fetchone()
        if row is None or kind and row[0] != kind:
            raise StrataError(f'unknown {kind or "record"}: {h}')
        value = json.loads(row[1])
        if digest(dict(kind=row[0], body=value)) != h:
            raise StrataError('record integrity failure')
        return value

    def all(self, kind):
        return [(h, self.get(h, kind)) for (h,) in self.db.execute('SELECT id FROM records WHERE kind=? ORDER BY rowid', (kind,)).fetchall()]

    def event(self, line, event):
        self.get(line, 'line')
        with self.db:
            self.db.execute('INSERT INTO events(line,body) VALUES (?,?)', (line, canonical(event)))

    def events(self, line):
        return [json.loads(x[0]) for x in self.db.execute('SELECT body FROM events WHERE line=? ORDER BY seq', (line,))]

    def negative(self, contract, move, kind, result, reopen, scale=None, *, evidence=None):
        if not isinstance(kind, str) or kind not in {'proof', 'exact_small', 'build', 'estimate'}:
            raise StrataError('unknown negative evidence kind')
        if not reopen or not result:
            raise StrataError('negative result needs evidence and reopen condition')
        requested_kind = kind
        evidence = dict(evidence) if isinstance(evidence, dict) else {'artifact_hash': evidence} if evidence else {}
        evidence_status = 'not_provided'
        if kind != 'estimate':
            try:
                self.read_blob(evidence.get('artifact_hash'))
            except (OSError, StrataError):
                kind = 'estimate'
                evidence_status = 'missing_or_invalid_artifact'
            else:
                evidence_status = 'artifact_hash_verified'
                if kind == 'exact_small' and not scale:
                    kind = 'estimate'
                    evidence_status = 'missing_exact_size'
        return self.put('negative', dict(contract=contract, move=move, evidence_kind=kind,
                        requested_evidence_kind=requested_kind, evidence=evidence,
                        evidence_status=evidence_status,
                        result=result, reopen=reopen, scale=scale,
                        status='priced' if kind == 'estimate' else 'closed_at_size' if kind == 'exact_small' else 'closed',
                        full_size_open=kind == 'exact_small'))

    def tried(self, contract, move):
        return [(h, v) for h, v in self.all('negative') if v['contract'] == contract and v['move'] == move]
