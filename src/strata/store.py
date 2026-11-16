"""Transactional immutable records and content-addressed artifacts."""
from pathlib import Path
import hashlib
import json
import sqlite3
import tempfile
import os
from .model import canonical, digest, StrataError


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
        if len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
            raise StrataError('invalid object hash')
        data = (self.path / 'objects' / h).read_bytes()
        if hashlib.sha256(data).hexdigest() != h:
            raise StrataError('artifact integrity failure')
        return data

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

    def negative(self, contract, move, kind, result, reopen, scale=None):
        if kind not in {'proof', 'exact_small', 'build', 'estimate'}:
            raise StrataError('unknown negative evidence kind')
        if not reopen or not result:
            raise StrataError('negative result needs evidence and reopen condition')
        if kind == 'exact_small' and not scale:
            raise StrataError('exact search needs size')
        return self.put('negative', dict(contract=contract, move=move, evidence_kind=kind,
                        result=result, reopen=reopen, scale=scale,
                        status='priced' if kind == 'estimate' else 'closed_at_size' if kind == 'exact_small' else 'closed',
                        full_size_open=kind == 'exact_small'))

    def tried(self, contract, move):
        return [(h, v) for h, v in self.all('negative') if v['contract'] == contract and v['move'] == move]
