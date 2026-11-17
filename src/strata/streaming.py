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


SKELETON_OPS = {'CCX', 'CCZ', 'CSWAP', 'HMR'}


def _skeleton_parts(lines, *, max_candidates=128):
    """Bounded repeated paid-anchor runs, allowing unmatched heads and tails.

    Small relabel-invariant anchor n-grams seed adjacent-repeat checks. The
    complete anchor topology is compared before a split; linear instructions
    are deliberately outside that equality. Disconnected paid wire sets offer
    an additional natural boundary. These are retrieval observations only.
    """
    anchors = [(i, line) for i, line in enumerate(lines)
               if line.split() and line.split()[0] in SKELETON_OPS]
    if len(anchors) < 2:
        return [lines]
    anchor_lines = [line for _, line in anchors]
    candidates, checks = [], 0
    # Bound both seed storage and expensive whole-run equality checks by the
    # current window. A pathological repetitive input cannot grow a global
    # in-memory occurrence index.
    for seed in (3, 4, 8):
        previous = {}
        skip_until = 0
        for b in range(len(anchors) - seed + 1):
            if b < skip_until:
                continue
            key = canonical_kernel(anchor_lines[b:b + seed])[0]
            recent = previous.setdefault(key, [])
            for a in reversed(recent):
                period = b - a
                if period < seed or b + period > len(anchors):
                    continue
                checks += 1
                if checks > max_candidates:
                    break
                pattern = canonical_kernel(anchor_lines[a:b])[0]
                if canonical_kernel(anchor_lines[b:b + period])[0] != pattern:
                    continue
                end = b + period
                while end + period <= len(anchors) and canonical_kernel(anchor_lines[end:end + period])[0] == pattern:
                    end += period
                candidates.append((a, end, period))
                skip_until = end
                break
            recent.append(b)
            # Keep distant representatives as well as recent matches: nearby
            # uniform CCX shapes must not hide a longer cell's recurrence.
            if len(recent) > 8:
                del recent[2]
            if checks > max_candidates:
                break
        if checks > max_candidates:
            break
    covered, selected = set(), []
    for a, end, period in sorted(candidates, key=lambda r: (-(r[1] - r[0]), -r[2], r[0])):
        if any(i in covered for i in range(a, end)):
            continue
        covered.update(range(a, end))
        selected.append((a, end, period))
    cuts = {0, len(lines)}
    for a, end, period in selected:
        # Cut just before each paid anchor; adjacent linear gates remain in
        # the preceding occurrence and get their own layer fingerprint.
        cuts.update(anchors[i][0] for i in range(a, end, period))
        if end < len(anchors): cuts.add(anchors[end][0])
    if not selected:
        neighborhood = set()
        for i, line in anchors:
            wires = set(re.findall(r'\bq\d+\b', line))
            if neighborhood and not neighborhood & wires:
                cuts.add(i)
                neighborhood.clear()
            neighborhood.update(wires)
    positions = sorted(cuts)
    return [lines[a:b] for a, b in zip(positions, positions[1:]) if a < b]


def paid_skeleton(lines):
    """Separate topology and positioned non-skeleton layers, never semantics."""
    anchors = [line for line in lines if line.split() and line.split()[0] in SKELETON_OPS]
    text, mapping = canonical_kernel(anchors)
    counts = {kind: sum(token.startswith(kind) for token in mapping) for kind in ('q', 'b')}
    def replace(match):
        token = match.group()
        if token not in mapping:
            kind = token[0]
            mapping[token] = kind + str(counts[kind])
            counts[kind] += 1
        return mapping[token]
    layers, position = [], 0
    for line in lines:
        if line.split() and line.split()[0] in SKELETON_OPS:
            position += 1
        else:
            layers.append((position, re.sub(r'\b[qb]\d+\b', replace, line.strip())))
    return text, hashlib.sha256(json.dumps(layers).encode()).hexdigest(), mapping


def scan_circuit(circuit, out, *, source=None, start=0, stop=None,
                 max_window=8192, max_variables=128, max_terms=4096,
                 max_groups=16, progress=None, progress_every=100000, clean_inputs=(),
                 range_unit='operation'):
    """Native census + disk-backed bounded semantic index of [start,stop).

    Budget cuts in an open classical scope are opaque, never reinterpreted as
    unconditional gates. Each group is analyzed once; capped top-group summaries
    reference every occurrence in the index. Static counts are not mean costs.
    """
    from .semantic import analyze_window, _parse, _advance_zero_facts, _zeros_after, _trim_zero_facts
    if range_unit not in {'operation', 'text-line'}:
        raise StrataError('range_unit must be operation or text-line')
    if not isinstance(start, int) or start < 0 or (stop is not None and (not isinstance(stop, int) or stop <= start)):
        raise StrataError('range must be a nonempty nonnegative half-open interval')
    if min(max_window, max_variables, max_terms, max_groups, progress_every) <= 0:
        raise StrataError('scan budgets must be positive')
    clean_inputs = tuple(sorted(set(clean_inputs)))
    if any(type(q) is not int or q < 0 for q in clean_inputs):
        raise StrataError('clean_inputs must contain nonnegative physical wire IDs')
    if clean_inputs and start:
        raise StrataError('artifact clean inputs apply only to scans starting at zero')
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dbpath = root / 'kernels.sqlite'
    db = sqlite3.connect(dbpath)
    db.executescript('DROP TABLE IF EXISTS kernels; DROP TABLE IF EXISTS occurrences; '
        'CREATE TABLE kernels (fingerprint TEXT PRIMARY KEY, operations INT, paid REAL, count INT, semantics TEXT, native TEXT, boundary TEXT); '
        'CREATE TABLE occurrences (fingerprint TEXT, start INT, stop INT, mapping TEXT, zero_evidence TEXT); '
        'CREATE INDEX occurrence_group ON occurrences(fingerprint); '
        'DROP TABLE IF EXISTS skeletons; DROP TABLE IF EXISTS skeleton_occurrences; '
        'CREATE TABLE skeletons (fingerprint TEXT PRIMARY KEY, native TEXT, paid INT, anchors INT, count INT); '
        'CREATE TABLE skeleton_occurrences (fingerprint TEXT, start INT, stop INT, linear_hash TEXT, syntax_hash TEXT, mapping TEXT, boundary TEXT); '
        'CREATE INDEX skeleton_group ON skeleton_occurrences(fingerprint);')
    db.execute('CREATE TABLE IF NOT EXISTS scan_metadata (key TEXT PRIMARY KEY, value TEXT)')
    db.execute('INSERT OR REPLACE INTO scan_metadata VALUES (?,?)', ('coordinate_unit', range_unit))
    begun = time.monotonic()
    census = Counter()
    buffer = []
    buffer_start = start
    depth = 0
    tainted = False
    scanned = 0
    occurrences = 0
    coverage = Counter(exact_instructions=0, unknown_instructions=0, exact_paid_operations=0,
                       skeleton_indexed_paid_operations=0, skeleton_indexed_instructions=0,
                       budget_cuts=0)
    known_zero = {q: dict(kind='declared_artifact_input', origin_at=0, origin_wire=q) for q in clean_inputs}
    max_zero_facts = max(max_variables, max_window)
    coverage['zero_fact_evictions'] = _trim_zero_facts(known_zero, max_zero_facts)
    buffer_zero_inputs = dict(known_zero)
    canonical_hash = hashlib.sha256()
    last_index = 0
    operation_index = 0
    selected_text_start = selected_text_stop = None
    selected_operation_start = selected_operation_stop = None
    scanned_operations = 0
    stopped_before_eof = False
    def notify(done=False):
        if progress:
            progress(dict(stage='scan_complete' if done else 'scan', scanned_operations=scanned_operations,
                          recovered_occurrences=occurrences,
                          seconds=time.monotonic() - begun,
                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))
    def flush(boundary):
        nonlocal buffer, occurrences
        if not buffer:
            return
        offset = buffer_start
        zero_inputs = dict(buffer_zero_inputs)
        parts = [buffer] if tainted else _periodic_parts(buffer)
        for part in parts:
            text, mapping = canonical_kernel(part)
            promised = sorted(int(mapping[f'q{q}'][1:]) for q in zero_inputs if f'q{q}' in mapping)
            zero_evidence = {mapping[f'q{q}']: proof for q, proof in zero_inputs.items() if f'q{q}' in mapping}
            key = hashlib.sha256((('opaque-context\n' if tainted else '') +
                                  (f'clean:{promised}\n' if promised else '') + text).encode()).hexdigest()
            old = db.execute('SELECT count FROM kernels WHERE fingerprint=?', (key,)).fetchone()
            if old:
                db.execute('UPDATE kernels SET count=count+1 WHERE fingerprint=?', (key,))
            else:
                sem = (dict(status='unknown', reason='context_scope_crosses_analysis_boundary') if tainted
                       else analyze_window(text.splitlines(), max_operations=max_window,
                                           max_variables=max_variables, max_terms=max_terms,
                                           clean_inputs=promised))
                paid = sum(line.split()[0] in {'CCX', 'CCZ', 'CSWAP'} for line in part if line.split())
                db.execute('INSERT INTO kernels VALUES (?,?,?,?,?,?,?)',
                           (key, len(part), paid, 1, json.dumps(sem), text, boundary))
            db.execute('INSERT INTO occurrences VALUES (?,?,?,?,?)',
                       (key, offset, offset + len(part), json.dumps(mapping), json.dumps(zero_evidence)))
            zero_inputs = _zeros_after([(offset+j, line, _parse(line)) for j,line in enumerate(part)], zero_inputs)
            offset += len(part)
            occurrences += 1
            # Reusing syntax evidence is valid only under the same canonical
            # boundary promise, which is included in the group fingerprint.
            if old:
                sem = json.loads(db.execute('SELECT semantics FROM kernels WHERE fingerprint=?', (key,)).fetchone()[0])
            coverage['exact_instructions' if sem['status'] == 'exact' else 'unknown_instructions'] += len(part)
            if sem['status'] == 'exact':
                coverage['exact_paid_operations'] += sum(line.split()[0] in {'CCX','CCZ','CSWAP'} for line in part if line.split())
        offset = buffer_start
        # A balanced condition scope stays together; cutting it into anchors
        # would lose the cell boundary already supplied by its syntax.
        scope = any(line.startswith('PUSH_CONDITION') for line in buffer)
        skeleton_parts = [buffer] if scope or tainted else _skeleton_parts(buffer)
        for part in skeleton_parts:
            native, linear_hash, mapping = paid_skeleton(part)
            if native:
                key = hashlib.sha256(native.encode()).hexdigest()
                syntax_hash = hashlib.sha256(canonical_kernel(part)[0].encode()).hexdigest()
                paid = sum(line.split()[0] in {'CCX','CCZ','CSWAP'} for line in part if line.split())
                db.execute('INSERT INTO skeletons VALUES (?,?,?,?,1) ON CONFLICT(fingerprint) DO UPDATE SET count=count+1',
                           (key, native, paid, len(native.splitlines())))
                db.execute('INSERT INTO skeleton_occurrences VALUES (?,?,?,?,?,?,?)',
                           (key, offset, offset+len(part), linear_hash, syntax_hash, json.dumps(mapping), boundary))
                coverage['skeleton_indexed_paid_operations'] += paid
                coverage['skeleton_indexed_instructions'] += len(part)
            offset += len(part)
        if boundary.startswith('budget_cut'):
            coverage['budget_cuts'] += 1
        buffer = []
        if occurrences % 256 == 0:
            db.commit()
    try:
        notify()
        with native_lines(circuit) as lines:
            for text_index, raw in enumerate(lines):
                line = raw.rstrip('\r\n')
                op = line.split(' ', 1)[0]
                # Native circuit order includes INFO_TAG and control operations,
                # but register declarations are serialization metadata only.
                is_operation = op not in {'REGISTER', 'APPEND_TO_REGISTER'}
                op_index = operation_index
                operation_index += is_operation
                if range_unit == 'operation' and not is_operation:
                    continue
                i = op_index if range_unit == 'operation' else text_index
                last_index = i + 1
                canonical_hash.update(raw.encode())
                if stop is not None and i >= stop:
                    stopped_before_eof = True
                    break
                if i < start:
                    if op == 'PUSH_CONDITION': depth += 1
                    if op == 'POP_CONDITION': depth = max(0, depth - 1)
                    continue
                if selected_text_start is None:
                    selected_text_start = text_index
                    selected_operation_start = op_index
                selected_text_stop = text_index + 1
                selected_operation_stop = operation_index
                scanned_operations += is_operation
                if not buffer:
                    buffer_start = i
                    tainted = depth != 0
                    buffer_zero_inputs = dict(known_zero)
                scanned += 1
                census[op] += 1
                parsed = _parse(line)
                _advance_zero_facts(parsed, known_zero, depth, i)
                coverage['zero_fact_evictions'] += _trim_zero_facts(known_zero, max_zero_facts)
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
            sample = [dict(start=a, stop=b, wire_mapping=json.loads(m), boundary_zero_evidence=json.loads(z)) for a,b,m,z in db.execute(
                'SELECT start,stop,mapping,zero_evidence FROM occurrences WHERE fingerprint=? LIMIT 8', (key,))]
            groups.append(dict(fingerprint=key, operations_per_occurrence=operations,
                               range_unit=range_unit,
                               static_toffoli_per_occurrence=paid, occurrences=count,
                               aggregate_static_toffoli=paid*count,
                               semantics=json.loads(semantics), examples=sample, boundary=boundary,
                               contract_scope='boundary zeros require per-occurrence transport from declared artifact inputs or unconditional HMR; other boundary values are arbitrary; measurement phase provenance is not inferred',
                               occurrence_index=str(dbpath), rewrite_boundary_verified=False))
        group_count = db.execute('SELECT count(*) FROM kernels').fetchone()[0]
        skeletons = []
        for key, paid, anchors, count in db.execute(
                'SELECT fingerprint,paid,anchors,count FROM skeletons ORDER BY paid*count DESC,anchors*count DESC LIMIT ?', (max_groups,)):
            variants, minimum, maximum = db.execute(
                'SELECT count(DISTINCT linear_hash),min(stop-start),max(stop-start) FROM skeleton_occurrences WHERE fingerprint=?', (key,)).fetchone()
            samples = [dict(start=a, stop=b, linear_layer_sha256=h, syntax_sha256=s,
                            wire_mapping=json.loads(m), boundary=boundary)
                       for a,b,h,s,m,boundary in db.execute(
                           'SELECT start,stop,linear_hash,syntax_hash,mapping,boundary FROM skeleton_occurrences WHERE fingerprint=? LIMIT 8', (key,))]
            skeletons.append(dict(fingerprint=key, grouping='paid_skeleton_modulo_linear_layers',
                                  range_unit=range_unit,
                                  occurrences=count, anchors_per_occurrence=anchors,
                                  static_toffoli_per_occurrence=paid, aggregate_static_toffoli=paid*count,
                                  linear_layer_variants=variants, operations_per_occurrence=dict(min=minimum,max=maximum),
                                  examples=samples, occurrence_index=str(dbpath),
                                  semantics=dict(status='observation',reason='paid_topology_equality_does_not_imply_functional_equivalence'),
                                  rewrite_boundary_verified=False))
        repeated_paid = db.execute('SELECT coalesce(sum(paid*count),0) FROM skeletons WHERE count>1').fetchone()[0]
        paid_total = sum(census[op] for op in ('CCX','CCZ','CSWAP'))
        coverage.update(paid_operations=paid_total, repeated_skeleton_paid_operations=repeated_paid)
        coverage_report = dict(coverage, scope='selected range; local window contracts',
                               exact_fraction=coverage['exact_instructions']/scanned if scanned else 0.0,
                               exact_paid_fraction=coverage['exact_paid_operations']/paid_total if paid_total else 0.0,
                               repeated_skeleton_paid_fraction=repeated_paid/paid_total if paid_total else 0.0,
                               skeleton_indexed_paid_fraction=coverage['skeleton_indexed_paid_operations']/paid_total if paid_total else 0.0,
                               skeleton_coverage_is_semantic_proof=False)
        elapsed = time.monotonic() - begun
        stats = circuit.operation_statistics()
        report = dict(schema_version='3', kind='bounded_streaming_analysis',
                      source=str(source) if source else None,
                      source_sha256=file_hash(source) if source else None,
                      range=dict(start=start, stop=min(stop, last_index) if stop else last_index),
                      range_unit=range_unit, occurrence_index_unit=range_unit,
                      text_line_range=dict(start=selected_text_start, stop=selected_text_stop),
                      operation_range=dict(start=selected_operation_start, stop=selected_operation_stop),
                      range_convention='zero-based half-open; operation excludes REGISTER and APPEND_TO_REGISTER, includes INFO_TAG',
                      scanned_operations=scanned_operations, full_file_operations=len(circuit),
                      scanned_instructions=scanned, full_scan=start==0 and not stopped_before_eof,
                      full_file_resources=dict(Q=circuit.num_qubits, toffoli=stats.num_toffoli_like,
                          t=stats.num_t_like, rotations=stats['Z_POW'], instructions=len(circuit), accounting='static_emitted'),
                      census=dict(census), groups=groups, group_count=group_count,
                      paid_skeletons=skeletons, skeleton_group_count=db.execute('SELECT count(*) FROM skeletons').fetchone()[0],
                      coverage=coverage_report, clean_input_assumptions=list(clean_inputs),
                      occurrences=occurrences, index=str(dbpath),
                      seconds=elapsed, seconds_per_million_operations=elapsed*1e6/scanned if scanned else None,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      memory_model='one native circuit plus bounded Python window and disk-backed evidence; native parse and text spool precede scan',
                      limits=dict(max_window=max_window,max_variables=max_variables,max_terms=max_terms,max_summary_groups=max_groups,
                                  max_anchor_repeat_candidates_per_window=128,max_boundary_zero_facts=max_zero_facts),
                      limitations=['Syntax groups and paid-skeleton groups are distinct observations; neither proves arithmetic intent or a replacement boundary.',
                                   'Paid skeletons omit linear layers, whose positioned syntax hashes are retained separately; equal skeletons can implement different functions.',
                                   'Bounded anchor-repeat search can miss repeats spanning windows or beyond its candidate budget; reported coverage is measured, not a completeness claim.',
                                   'Scope-cut segments remain unknown. Static Toffoli counts do not estimate conditional mean cost.',
                                   'Full native circuit storage and SDK conversion are not constant-memory streaming binary decoding.'])
        (root/'scan.json').write_text(json.dumps(report, indent=2) + '\n')
        notify(done=True)
        return report
    finally:
        db.close()
