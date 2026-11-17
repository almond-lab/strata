"""Bounded circuit census and repeated-kernel recovery.

The native parser owns the binary input once. Text is spooled by the SDK, never
expanded into a Python list for the complete circuit. Repeated region evidence
and occurrences live in SQLite; only bounded top groups enter the AI brief.
This is an analysis index, not a proof of a legal replacement boundary.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import resource
import sqlite3
import tempfile
import time

from .model import StrataError


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@contextmanager
def native_lines(circuit):
    """Yield a one-pass canonical text stream, with bounded Python memory."""
    if isinstance(circuit, str):
        import io
        with io.StringIO(circuit) as f:
            yield f
        return
    with tempfile.TemporaryFile(mode='w+b') as f:
        circuit.to_file(f, format='kmx')
        f.seek(0)
        import io
        wrapper = io.TextIOWrapper(f, encoding='utf-8')
        try:
            yield wrapper
        finally:
            wrapper.detach()


def canonical_kernel(lines):
    """Wire relabel invariant syntax, preserving quantum/classical namespaces."""
    mapping = {}
    counts = {'q': 0, 'b': 0}
    def replace(m):
        token = m.group()
        if token not in mapping:
            kind = token[0]
            mapping[token] = kind + str(counts[kind])
            counts[kind] += 1
        return mapping[token]
    text = '\n'.join(re.sub(r'\b[qb]\d+\b', replace, line.strip()) for line in lines)
    return text, mapping


def _periodic_parts(lines):
    """Recover exact repeated syntax periods inside a bounded natural run."""
    if not lines:
        return []
    if any(line.split()[0] not in {'X','CX','CCX','SWAP','Z','CZ','CCZ','NEG'} or ' if ' in line for line in lines):
        return [lines]
    # KMP works on opcode/operand positions before proving each full block's
    # canonical source equal. A coincidental shape never authorizes a rewrite.
    shapes = [re.sub(r'\b[qb]\d+\b', lambda m: m.group()[0], line) for line in lines]
    prefix = [0] * len(lines)
    for i in range(1, len(lines)):
        j = prefix[i - 1]
        while j and shapes[i] != shapes[j]:
            j = prefix[j - 1]
        if shapes[i] == shapes[j]:
            j += 1
        prefix[i] = j
    period = len(lines) - prefix[-1]
    if period >= 3 and len(lines) % period == 0 and period * 2 <= len(lines):
        parts = [lines[i:i + period] for i in range(0, len(lines), period)]
        canonical = canonical_kernel(parts[0])[0]
        if all(canonical_kernel(part)[0] == canonical for part in parts[1:]):
            return parts
    return [lines]


def scan_circuit(circuit, out, *, source=None, start=0, stop=None,
                 max_window=8192, max_variables=128, max_terms=4096,
                 max_groups=16, progress=None, progress_every=100000):
    """Native census + disk-backed bounded semantic index of [start,stop).

    Budget cuts in an open classical scope are opaque, never reinterpreted as
    unconditional gates. Each group is analyzed once; capped top-group summaries
    reference every occurrence in the index. Static counts are not mean costs.
    """
    from .semantic import analyze_window, _parse
    if not isinstance(start, int) or start < 0 or (stop is not None and (not isinstance(stop, int) or stop <= start)):
        raise StrataError('range must be a nonempty nonnegative half-open interval')
    if min(max_window, max_variables, max_terms, max_groups, progress_every) <= 0:
        raise StrataError('scan budgets must be positive')
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dbpath = root / 'kernels.sqlite'
    db = sqlite3.connect(dbpath)
    db.executescript('DROP TABLE IF EXISTS kernels; DROP TABLE IF EXISTS occurrences; '
        'CREATE TABLE kernels (fingerprint TEXT PRIMARY KEY, operations INT, paid REAL, count INT, semantics TEXT, native TEXT, boundary TEXT); '
        'CREATE TABLE occurrences (fingerprint TEXT, start INT, stop INT, mapping TEXT); '
        'CREATE INDEX occurrence_group ON occurrences(fingerprint);')
    begun = time.monotonic()
    census = Counter()
    buffer = []
    buffer_start = start
    depth = 0
    tainted = False
    scanned = 0
    occurrences = 0
    canonical_hash = hashlib.sha256()
    last_index = 0
    def notify(done=False):
        if progress:
            progress(dict(stage='scan_complete' if done else 'scan', scanned_operations=scanned,
                          recovered_occurrences=occurrences,
                          seconds=time.monotonic() - begun,
                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))
    def flush(boundary):
        nonlocal buffer, occurrences
        if not buffer:
            return
        offset = buffer_start
        parts = [buffer] if tainted else _periodic_parts(buffer)
        for part in parts:
            text, mapping = canonical_kernel(part)
            key = hashlib.sha256((('opaque-context\n' if tainted else '') + text).encode()).hexdigest()
            old = db.execute('SELECT count FROM kernels WHERE fingerprint=?', (key,)).fetchone()
            if old:
                db.execute('UPDATE kernels SET count=count+1 WHERE fingerprint=?', (key,))
            else:
                sem = (dict(status='unknown', reason='context_scope_crosses_analysis_boundary') if tainted
                       else analyze_window(text.splitlines(), max_operations=max_window,
                                           max_variables=max_variables, max_terms=max_terms))
                paid = sum(line.split()[0] in {'CCX', 'CCZ', 'CSWAP'} for line in part if line.split())
                db.execute('INSERT INTO kernels VALUES (?,?,?,?,?,?,?)',
                           (key, len(part), paid, 1, json.dumps(sem), text, boundary))
            db.execute('INSERT INTO occurrences VALUES (?,?,?,?)',
                       (key, offset, offset + len(part), json.dumps(mapping)))
            offset += len(part)
            occurrences += 1
        buffer = []
        if occurrences % 256 == 0:
            db.commit()
    try:
        notify()
        with native_lines(circuit) as lines:
            for i, raw in enumerate(lines):
                last_index = i + 1
                canonical_hash.update(raw.encode())
                if stop is not None and i >= stop:
                    break
                line = raw.rstrip('\r\n')
                op = line.split(' ', 1)[0]
                if i < start:
                    if op == 'PUSH_CONDITION': depth += 1
                    if op == 'POP_CONDITION': depth = max(0, depth - 1)
                    continue
                if not buffer:
                    buffer_start = i
                    tainted = depth != 0
                scanned += 1
                census[op] += 1
                parsed = _parse(line)
                if parsed['status'] != 'supported' and depth == 0:
                    flush('natural_opaque_or_metadata_cut')
                    buffer_start = i
                    buffer = [line]
                    flush('opaque_or_metadata')
                    continue
                buffer.append(line)
                if op == 'PUSH_CONDITION': depth += 1
                if op == 'POP_CONDITION': depth = max(0, depth - 1)
                if depth == 0 and (op == 'POP_CONDITION' or len(buffer) >= max_window):
                    flush('complete_scope' if op == 'POP_CONDITION' else 'budget_cut_arbitrary_boundary')
                elif len(buffer) >= max_window:
                    tainted = True
                    flush('budget_cut_opaque_context')
                if scanned % progress_every == 0:
                    db.commit()
                    notify()
        if depth: tainted = True
        flush('end_of_selected_range')
        db.commit()
        rows = db.execute('SELECT fingerprint,operations,paid,count,semantics,boundary FROM kernels ORDER BY paid*count DESC,operations*count DESC LIMIT ?', (max_groups,)).fetchall()
        groups = []
        for key, operations, paid, count, semantics, boundary in rows:
            sample = [dict(start=a, stop=b, wire_mapping=json.loads(m)) for a,b,m in db.execute(
                'SELECT start,stop,mapping FROM occurrences WHERE fingerprint=? LIMIT 8', (key,))]
            groups.append(dict(fingerprint=key, operations_per_occurrence=operations,
                               static_toffoli_per_occurrence=paid, occurrences=count,
                               aggregate_static_toffoli=paid*count,
                               semantics=json.loads(semantics), examples=sample, boundary=boundary,
                               contract_scope='arbitrary quantum and classical window boundary; earlier values and measurement provenance are not inferred',
                               occurrence_index=str(dbpath), rewrite_boundary_verified=False))
        group_count = db.execute('SELECT count(*) FROM kernels').fetchone()[0]
        elapsed = time.monotonic() - begun
        stats = circuit.operation_statistics()
        report = dict(schema_version='3', kind='bounded_streaming_analysis',
                      source=str(source) if source else None,
                      source_sha256=file_hash(source) if source else None,
                      range=dict(start=start, stop=min(stop, last_index) if stop else last_index),
                      scanned_operations=scanned, full_file_operations=len(circuit),
                      full_file_resources=dict(Q=circuit.num_qubits, toffoli=stats.num_toffoli_like,
                          t=stats.num_t_like, rotations=stats['Z_POW'], instructions=len(circuit), accounting='static_emitted'),
                      census=dict(census), groups=groups, group_count=group_count,
                      occurrences=occurrences, index=str(dbpath),
                      seconds=elapsed, seconds_per_million_operations=elapsed*1e6/scanned if scanned else None,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      memory_model='one native circuit plus bounded Python window and disk-backed evidence; native parse and text spool precede scan',
                      limits=dict(max_window=max_window,max_variables=max_variables,max_terms=max_terms,max_summary_groups=max_groups),
                      limitations=['Group equality is wire-relabelled syntax; arithmetic intent and rewrite boundaries require proof.',
                                   'Scope-cut segments remain unknown. Static Toffoli counts do not estimate conditional mean cost.',
                                   'Full native circuit storage and SDK conversion are not constant-memory streaming binary decoding.'])
        (root/'scan.json').write_text(json.dumps(report, indent=2) + '\n')
        notify(done=True)
        return report
    finally:
        db.close()
