#!/usr/bin/env python3
"""Replay fixed operation spans of a provided archive. Publish only derived evidence."""
import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

import kickmix as km
import strata
from strata.semantic import _advance_zero_facts, _parse
from native_spans import NativeSpans
from replay import independent_check

HERE = Path(__file__).resolve().parent
ARCHIVE_SHA = '0212e251262a4d3d40ab5a36af664e8e11f24c14305db9c2105ceaf22253e443'
MANIFEST_SHA = '50106a9befebfd1004e0aa85e4eb6ce66446d4de8f4ac665f8053cd3cb690534'
ORACLE = HERE.parent / 'v05-holdout' / 'run.py'
OPTIONS = ['--cost', 'toffoli', '--rounds', '1', '--max-candidates', '24',
           '--max-operations', '320', '--max-variables', '96', '--max-terms', '8192',
           '--max-product-terms', '8192', '--max-architectures', '128',
           '--seconds', '12', '--trial-seconds', '2']
WORKER_TIMEOUT = 30
NATIVE_CAP = 65536
TOKENS = re.compile(r'\b([qb])(\d+)\b')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def build_id():
    package = Path(strata.__file__).parent
    files = {p.name: file_sha(p) for p in sorted(package.glob('*.py'))}
    return sha(json.dumps(files, sort_keys=True).encode())


def dead_bits(tape, stop, bits):
    """Only hide an output when its first suffix use overwrites it unconditionally."""
    pending = set(bits)
    proved = {}
    depth = tape.depth(stop)
    end = min(tape.ops, stop+1024)
    if end > stop and pending:
        for at, line in enumerate(str(tape.span(stop, end)).splitlines(), stop):
            words = line.split()
            uses = {int(i) for kind, i in TOKENS.findall(line) if kind == 'b'}
            if depth == 0 and 'if' not in words and words[0] in ('BIT_STORE0','BIT_STORE1','HMR'):
                target = int(words[-1][1:])
                if target in pending:
                    proved[target] = dict(kind='first_suffix_use_is_unconditional_overwrite', at=at)
            pending -= uses
            depth += int(words[0]=='PUSH_CONDITION') - int(words[0]=='POP_CONDITION')
            if not pending:
                break
    return proved


def sdk_span_crosschecks(binary, cases, directory):
    """Independent native decode and line-by-line hash check for small parents."""
    binary.seek(0)
    c = km.Circuit.from_file(binary, format='kmb')
    hashers = {name:hashlib.sha256() for name, _, _, _ in cases}
    with tempfile.TemporaryFile(dir=directory) as stream:
        c.to_file(stream, format='kmx')
        stream.seek(0)
        index = 0
        for line in stream:
            if line.startswith((b'REGISTER ', b'APPEND_TO_REGISTER ')):
                continue
            for name, _, start, stop in cases:
                if start <= index < stop:
                    if index > start:
                        hashers[name].update(b'\n')
                    hashers[name].update(line.rstrip(b'\r\n'))
            index += 1
    if index != len(c):
        raise ValueError('Native SDK operation count mismatch')
    return {name:h.hexdigest() for name,h in hashers.items()}


def make_source(tape, member, start, stop, work, extended):
    # Only the preceding 1024 decoder operations are queried for zero facts.
    # Start with no decoder entry promises. The parent depth prevents a nested
    # HMR from accidentally becoming an unconditional source of zero facts.
    decoder = member == 'dqi.kmb'
    first = max(0, start-1024) if decoder else 0
    zeros = {}
    if not decoder:
        names = ['x'] if member == 'invsqrt.kmb' else ['selection', *[f'word_{i}' for i in range(5)]]
        if tape.register_names != names:
            raise ValueError('Parent input interface differs from the audited initialization contract')
        zeros = {q: dict(kind='parent_ancilla_zero_at_entry', origin_wire=q, origin_at=0)
                 for q in range(tape.width) if q not in tape.register_qubits}
    depth = tape.depth(first)
    if first < start:
        for at, line in enumerate(str(tape.span(first, start)).splitlines(), first):
            item = _parse(line)
            _advance_zero_facts(item, zeros, depth, at)
            depth += int(item['op'] == 'PUSH_CONDITION') - int(item['op'] == 'POP_CONDITION')
    if tape.depth(start) != depth:
        raise ValueError('Prefix scope mismatch')
    source = tape.span(start, stop)
    body = str(source)
    balance = 0
    for line in body.splitlines():
        op = line.split()[0]
        balance += int(op == 'PUSH_CONDITION') - int(op == 'POP_CONDITION')
        if balance < 0:
            raise ValueError(f'Span at {start} closes an external scope')
    if balance:
        raise ValueError(f'Span at {start} splits a condition scope')

    ids = {kind: sorted({int(i) for k, i in TOKENS.findall(body) if k == kind}) for kind in 'qb'}
    mapped = {kind: {i: j for j, i in enumerate(values)} for kind, values in ids.items()}
    compact = TOKENS.sub(lambda m: m[1]+str(mapped[m[1]][int(m[2])]), body)
    restored = TOKENS.sub(lambda m: m[1]+str(ids[m[1]][int(m[2])]), compact)
    if restored != body or str(km.Circuit(compact)) != compact:
        raise ValueError('Native relabel did not round-trip')
    (work/'source.kmx').write_text(compact)
    clean = sorted(mapped['q'][q] for q in zeros.keys() & mapped['q'].keys())
    private = dead_bits(tape, stop, ids['b']) if extended else {}
    scope = {}
    if extended and member.startswith('dfthcsos/'):
        from loader_promise import derive
        scope.update(derive(tape, start, ids['q'], clean, work))
    paid = source.operation_statistics().num_toffoli_like
    return dict(**scope, source_native_sha256=sha(body.encode()),
                private_local=sorted(mapped['b'][b] for b in private), private_physical=private, compact_source_sha256=sha(compact.encode()),
                physical_qubits=ids['q'], physical_bits=ids['b'], clean_local=clean,
                clean_physical={str(q): zeros[q] for q in ids['q'] if q in zeros},
                parent_entry_condition_depth=depth, before_paid=paid,
                measurements=sum(line.split()[0]=='HMR' for line in body.splitlines()),
                resets=sum(line.split()[0]=='R' for line in body.splitlines()),
                fraction_rotation_in_span=any(line.startswith('Z_POW') for line in body.splitlines()))


def worker(work, clean, internal):
    from strata.cli import main
    domain = json.loads((work/'domain.json').read_text()) if (work/'domain.json').exists() else None
    options = [*OPTIONS, '--internal-bits', ','.join(map(str, internal))]
    if domain is not None:
        options += ['--domain', str(work/'domain.json')]
    with contextlib.redirect_stdout(io.StringIO()):
        result = main(['search', str(work/'source.kmx'), '--out', str(work/'search'),
                       '--clean-inputs', ','.join(map(str, clean)), *options])
    if result != 0:
        raise RuntimeError('Nonzero search exit')
    r = json.loads((work/'search'/'restructure.json').read_text())
    a = km.Circuit.from_file(work/'source.kmx')
    b = km.Circuit.from_file(r['best']['path'])
    private = r['best']['private_bits']
    before, after = [c.operation_statistics().num_toffoli_like for c in (a,b)]
    if [before, after] != [r['baseline']['cost'], r['best']['cost']]:
        raise ValueError('Native circuit cost mismatch')
    if b.num_qubits > a.num_qubits or any(i < a.num_bits and i not in internal for i in private):
        raise ValueError('Unexpected new quantum or colliding classical allocation')
    check = independent_check(a, b, clean, private, domain, NATIVE_CAP)
    sem = r['boundary_receipts'].get(r['baseline']['id'], {})
    attempts = r['attempts']
    reasons = Counter(a.get('construction', {}).get('reason') for a in attempts)
    reasons.pop(None, None)
    included = r.get('discovery_yield', {})
    out = dict(before_native={k:r['baseline']['resources'][k] for k in ('Q','toffoli','t','rotations','instructions')},
               after_native={k:r['best']['resources'][k] for k in ('Q','toffoli','t','rotations','instructions')},
               search_status=r['status'], after_paid=after, delta=after-before,
               compact_candidate_sha256=sha(str(b).encode()), private_fresh_bits=[i for i in private if i >= a.num_bits],
               selected_mechanism=r['best']['mechanism'], independent_check=check,
               semantic_status=sem.get('status'), semantic_reason=sem.get('reason'),
               attempt_count=len(attempts), attempt_outcomes=dict(Counter(a['status'] for a in attempts)),
               attempt_mechanisms=[a['mechanism'] for a in attempts],
               rejection_reasons=dict(reasons), timed_out_trials=r['search_control']['timed_out'],
               search_stop_reason=r['search_control']['stop_reason'],
               cheaper_designs=included.get('unique_verified_cheaper_native_designs'))
    if after < before:
        (work/'compact-winner.kmx').write_text(str(b))
    (work/'row.json').write_text(json.dumps(out))


def execute(work, clean, internal):
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(work),
                             '--clean', json.dumps(clean), '--internal', json.dumps(internal)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, start_new_session=True, text=True)
    try:
        _, err = proc.communicate(timeout=WORKER_TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return dict(search_status='worker_timeout', timeout_seconds=WORKER_TIMEOUT)
    if proc.returncode:
        # General exception class only. A third-party exception can contain input ops.
        return dict(search_status='worker_error', exit_code=proc.returncode,
                    error_type=err.strip().splitlines()[-1].split(':')[0][:80] if err.strip() else None)
    return json.loads((work/'row.json').read_text())


def verify_splice(row, tape, binary, work, directory):
    """Restore physical names, reserve fresh bits, and natively reprice small parents."""
    result = dict(status='identity_retained')
    if row.get('search_status') in ('worker_error', 'worker_timeout'):
        return dict(status='search_incomplete')
    if row['delta'] >= 0:
        return result
    if row['independent_check'].get('equivalent') is not True:
        return dict(status='changed_candidate_lacks_independent_exact_replay')
    body = (work/'compact-winner.kmx').read_text().splitlines()
    body = [line for line in body if line.split()[0] not in ('REGISTER','APPEND_TO_REGISTER')]
    qs, bs = row['physical_qubits'], row['physical_bits']
    def physical(match):
        kind, i = match[1], int(match[2])
        if kind == 'q':
            return 'q'+str(qs[i])
        if i < len(bs):
            return 'b'+str(bs[i])
        if i not in row['private_fresh_bits']:
            raise ValueError('Unreserved candidate bit')
        return 'b'+str(tape.bits+i-len(bs))
    body = [TOKENS.sub(physical, line) for line in body]
    physical_circuit = km.Circuit('\n'.join(body))
    if physical_circuit.num_qubits > tape.width:
        raise ValueError('Replacement exceeds parent width')
    if row['member'] == 'dqi.kmb':
        return dict(status='local_exact_replay_only', physical_body_sha256=sha(str(physical_circuit).encode()))
    binary.seek(0)
    parent = km.Circuit.from_file(binary, format='kmb')
    with tempfile.TemporaryFile(dir=directory) as stream:
        parent.to_file(stream, format='kmx')
        stream.seek(0)
        lines = [line.decode().rstrip() for line in stream]
    offset = len(lines)-len(parent)
    original = lines[offset+row['start']:offset+row['stop']]
    if sha('\n'.join(original).encode()) != row['source_native_sha256']:
        raise ValueError('Contextual binding mismatch')
    lines[offset+row['start']:offset+row['stop']] = body
    changed = km.Circuit('\n'.join(lines))
    if changed.register_data != parent.register_data or changed.num_qubits > parent.num_qubits:
        raise ValueError('Reassembled parent layout mismatch')
    costs = [c.operation_statistics().num_toffoli_like for c in (parent,changed)]
    if costs[1]-costs[0] != row['delta']:
        raise ValueError('Reassembled parent contradicts charged delta')
    with tempfile.TemporaryFile(dir=directory) as stream:
        changed.to_file(stream, format='kmb')
        stream.seek(0)
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(status='native_parent_splice_repriced', parent_before_paid=costs[0], parent_after_paid=costs[1],
                parent_candidate_sha256=digest, parent_before_width=parent.num_qubits,
                parent_after_width=changed.num_qubits, official_verifier_run=False)


def run(archive, scratch, output, extended, wheel):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError('Use a fresh receipt path')
    manifest = archive.with_name('MANIFEST.txt')
    if file_sha(archive) != ARCHIVE_SHA or file_sha(manifest) != MANIFEST_SHA:
        raise ValueError('Archive or companion manifest hash mismatch')
    expected = dict((r.split()[1], r.split()[0]) for r in manifest.read_text().splitlines())
    cases = json.loads((HERE/'cases.json').read_text())
    if len(cases)!=20 or len({r[0] for r in cases})!=len(cases):
        raise ValueError('Expected 20 unique selections')
    build = build_id()
    wheel_digest = None
    if wheel:
        with zipfile.ZipFile(wheel) as package:
            code = {Path(name).name:sha(package.read(name)) for name in package.namelist()
                    if name.startswith('strata/') and name.endswith('.py')}
        if sha(json.dumps(code, sort_keys=True).encode()) != build:
            raise ValueError('Imported package sources differ from the named wheel')
        wheel_digest = file_sha(wheel)
    parents, rows = {}, []
    with tempfile.TemporaryDirectory(dir=scratch, prefix='strata-private-transfer-') as directory, zipfile.ZipFile(archive) as z:
        directory = Path(directory)
        for member in dict.fromkeys(case[1] for case in cases):
            with tempfile.TemporaryFile(dir=directory) as binary:
                with z.open(member) as src:
                    shutil.copyfileobj(src, binary, 1024**2)
                binary.seek(0)
                digest = hashlib.file_digest(binary, 'sha256').hexdigest()
                if digest != expected[member]:
                    raise ValueError('Member digest differs from manifest')
                tape = NativeSpans(binary)
                try:
                    h = tape.histogram(tape.ops)
                    parents[member] = dict(sha256=digest, width=tape.width, bits=tape.bits,
                                            operations=tape.ops,
                                            paid=int(sum(h[i]+h[i+128] for i in (48,49,50))))
                    checks = ({} if member=='dqi.kmb' else
                              sdk_span_crosschecks(binary, [c for c in cases if c[1]==member], directory))
                    for name, case_member, start, stop in cases:
                        if case_member != member:
                            continue
                        with tempfile.TemporaryDirectory(dir=directory) as work:
                            work = Path(work)
                            case_started = time.monotonic()
                            row = dict(case=name, member=member, start=start, stop=stop,
                                       **make_source(tape, member, start, stop, work, extended))
                            if checks and row['source_native_sha256'] != checks[name]:
                                raise ValueError('Native whole-parent decode disagrees with extracted span')
                            row['independent_full_parent_slice_check'] = bool(checks)
                            row.update(execute(work, row['clean_local'], row['private_local']))
                            row['parent_check'] = verify_splice(row, tape, binary, work, directory)
                            row['case_wall_seconds'] = round(time.monotonic()-case_started, 3)
                            rows.append(row)
                            print(name, row['search_status'], row['before_paid'], row.get('after_paid'),
                                  row.get('semantic_reason'), flush=True)
                finally:
                    tape.close()
    if build_id() != build:
        raise RuntimeError('Search package changed during sweep. Use an immutable installed artifact.')
    result = dict(sweep_wall_seconds=round(time.monotonic()-started, 3), contract_mode='suffix_and_loader_promises' if extended else 'universal', archive_sha256=ARCHIVE_SHA, manifest_sha256=MANIFEST_SHA, parent_members=parents,
                  package_version=importlib.metadata.version('strata-restructure'), package_python_sha256=build, installed_wheel_sha256=wheel_digest,
                  kickmix_distribution=importlib.metadata.version('kickmix'), kickmix_library_sha256=file_sha(km.__file__),
                  recipe_sha256={p.name:file_sha(p) for p in [Path(__file__), HERE/'native_spans.py', HERE/'replay.py', HERE/'cases.json', *([HERE/'loader_promise.py'] if extended else [])]}, native_oracle_sha256=file_sha(ORACLE),
                  cli_budget_arguments=OPTIONS, worker_timeout_seconds=WORKER_TIMEOUT, native_case_cap=NATIVE_CAP,
                  coordinates='zero-based half-open native operation index; register records excluded',
                  objective='static emitted CCX + CCZ + CSWAP in supplied span',
                  price_scope='local episode', official_verifier_run=False, results=rows)
    output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, help='verify the imported search source against the named distribution')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--scratch', type=Path, help='writable private directory outside the checkout')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--universal', action='store_true', help='omit suffix privacy and loader domains for the universal control')
    parser.add_argument('--internal', help=argparse.SUPPRESS)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--clean', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, json.loads(args.clean), json.loads(args.internal))
    elif not (args.archive and args.scratch and args.out):
        parser.error('supply --archive, --scratch and --out')
    else:
        repo = HERE.parent.parent
        if args.scratch.resolve().is_relative_to(repo):
            parser.error('choose a private scratch directory outside this checkout')
        run(args.archive, args.scratch, args.out, not args.universal, args.wheel)
