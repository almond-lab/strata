import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from strata.memory import apply_memory, import_attempts
from strata.model import Contract, Register, Region, StrataError, canonical, digest
from strata.store import Store


def test_integer_keys_roundtrip_normalized_before_sorting_and_hashing(tmp_path):
    store = Store(tmp_path)
    original = {'ledger': {2: {'child': {10: 1, 2: 3}}, 10: [1, 2]}}
    handle = store.put('test', original)
    read = store.get(handle)
    assert read == {'ledger': {'2': {'child': {'10': 1, '2': 3}}, '10': [1, 2]}}
    assert store.put('test', read) == handle
    assert digest(original) == digest(json.loads(canonical(original)))
    assert list(original['ledger']) == [2, 10]
    assert '"10"' in canonical(original)


def test_string_key_record_hashes_remain_compatible_and_dataclasses_recurse(tmp_path):
    store = Store(tmp_path)
    value = {'z': [4, {'a': 'x'}], 'a': {'9': 2, '10': 1}}
    old_body = json.dumps({'kind': 'legacy', 'body': value}, sort_keys=True, separators=(',', ':'), allow_nan=False)
    expected = hashlib.sha256(old_body.encode()).hexdigest()
    assert store.put('legacy', value) == expected
    contract = Contract('x', (Register('x', (0,)),), 'identity', scales={10: 2, 2: 1})
    assert store.get(store.put('contract', contract))['scales'] == {'10': 2, '2': 1}


@pytest.mark.parametrize('bad', [{1: 'a', '1': 'b'}, {2: 'a', 'other': 'b'}, {True: 'a'}, {1.5: 'b'}])
def test_ambiguous_keys_fail_before_store_write(tmp_path, bad):
    store = Store(tmp_path)
    with pytest.raises(StrataError, match='keys'):
        store.put('bad', {'nested': [bad]})
    assert store.all('bad') == []


@pytest.mark.parametrize('kind', ['proof', 'build', 'exact_small'])
@pytest.mark.parametrize('evidence', [None, 'prose proves impossible', {'artifact_hash': '0' * 64}, {'artifact_hash': []}])
def test_direct_negative_cannot_close_narrative_or_missing_blob(tmp_path, kind, evidence):
    store = Store(tmp_path)
    handle = store.negative('fp', 'fuse', kind, 'impossible', 'new evidence', {'bits': 2}, evidence=evidence)
    row = store.get(handle)
    assert row['status'] == 'priced'
    assert row['evidence_kind'] == 'estimate'
    assert row['requested_evidence_kind'] == kind


def test_valid_artifact_strength_scope_and_later_missing_blob(tmp_path):
    store = Store(tmp_path)
    artifact = store.blob(b'independent recorded evidence')
    handles = [store.negative('fp', kind, kind, {'measured': 4}, 'new contract',
                             {'bits': 2} if kind == 'exact_small' else None, evidence=artifact)
               for kind in ('proof', 'build', 'exact_small')]
    assert [store.get(h)['status'] for h in handles] == ['closed', 'closed', 'closed_at_size']
    missing_size = store.negative('fp', 'size missing', 'exact_small', 'result', 'size', evidence=artifact)
    assert store.get(missing_size)['evidence_status'] == 'missing_exact_size'
    small = apply_memory(store, contract='fp', scale={'bits': 2})
    assert small['counts'] == {'priced': 1, 'closed': 2, 'closed_at_size': 1, 'reopen': 0}
    large = apply_memory(store, contract='fp', scale={'bits': 10})
    assert large['counts']['reopen'] == 1
    assert next(v for v in large['items'] if v['move'] == 'exact_small')['full_size_open']
    assert apply_memory(store, contract='changed')['counts']['reopen'] == 4
    (store.path / 'objects' / artifact).unlink()
    assert apply_memory(store, contract='fp', scale={'bits': 2})['counts']['priced'] == 4
    assert store.get(handles[0])['status'] == 'closed'  # immutable historical record


def test_memory_application_uses_saved_design_contract_and_scale(tmp_path):
    store = Store(tmp_path)
    contract = Contract('x', (Register('x', (0,)),), 'identity')
    tree = Region('root', 0, 1, contract)
    base = store.put('design', {'tree': asdict(tree), 'parameters': {'bits': 2}})
    blob = store.blob(b'small-size exhaustive evidence')
    store.negative(contract.fingerprint, 'fuse', 'exact_small', 'result', 'new size', {'bits': 2}, evidence=blob)
    assert apply_memory(store, base)['counts']['closed_at_size'] == 1
    new = store.put('design', {'tree': asdict(tree), 'parameters': {'bits': 3}})
    assert apply_memory(store, new)['counts']['reopen'] == 1


def test_jsonl_and_csv_use_the_same_evidence_gate_and_keep_provenance(tmp_path):
    store = Store(tmp_path / 'store')
    artifact = store.blob(b'bound proof artifact')
    rows = [dict(id='bound', mechanism='fuse', contract_fingerprint='fp', evidence={'kind': 'proof', 'artifact_hash': artifact}, reopen_if='new contract'),
            dict(id='prose', mechanism='direct', evidence={'kind': 'proof'}, reopen_if='new premise'),
            dict(id='unknown', mechanism='other', outcome='closed by estimate')]
    for extension in ('jsonl', 'csv'):
        path = tmp_path / ('attempts.' + extension)
        if extension == 'jsonl':
            path.write_text('\n'.join(json.dumps(row) for row in rows))
        else:
            with path.open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted(set(key for row in rows for key in row)))
                writer.writeheader()
                writer.writerows({key: json.dumps(value) if isinstance(value, dict) else value for key, value in row.items()} for row in rows)
        assert import_attempts(store, path, {'prose': 'fp'}) == {'imported': 2, 'unbound': 1, 'priced': 1, 'closed': 1}
    assert len(store.all('negative')) == 4
    assert all(row['binding_status'] == 'unbound' for _, row in store.all('unbound_attempt'))
    assert all(store.read_blob(row['result']['source_blob']) for _, row in store.all('negative'))


def test_realistic_array_notes_and_wrapped_string_notes(tmp_path):
    store = Store(tmp_path / 'store')
    source = tmp_path / 'notes.json'
    source.write_text(json.dumps([{'name': 'consumer', 'result': 'proof by narrative', 'scope': 'small only'}, {'name': 'future', 'note': 'derive next'}]))
    assert import_attempts(store, source, {'consumer': 'fp'}) == {'imported': 1, 'unbound': 1, 'priced': 1, 'closed': 0}
    source.write_text(json.dumps({'time_utc': 'test', 'closed_experiments': [{'name': 'first', 'result': 'closed'}], 'next_research': ['Try a different encoding']}))
    result = import_attempts(store, source)
    assert result['unbound'] == 2
    assert any(isinstance(row['original'], str) for _, row in store.all('unbound_attempt'))


def test_legacy_strong_record_without_artifact_reopens_as_estimate(tmp_path):
    store = Store(tmp_path)
    store.put('negative', {'contract': 'fp', 'move': 'fuse', 'evidence_kind': 'proof', 'result': 'narrative', 'reopen': 'new contract', 'status': 'closed'})
    assert apply_memory(store, contract='fp')['counts']['priced'] == 1


def test_actual_local_historical_logs_without_copying_private_bodies(tmp_path):
    root = Path('/home/dev/algorithms/quantum-circuit-patterns')
    sources = [root / 'workflow/attempts.jsonl', root / 'research/negative-results/ledger.csv',
               Path('/home/dev/algorithms/kaggle-quantum/results/research-notes.json')]
    if not all(path.is_file() for path in sources):
        pytest.skip('local historical inputs are not available')
    store = Store(tmp_path / 'store')
    for source in sources:
        before = source.read_bytes()
        if source.suffix == '.csv':
            import io
            expected = sum(1 for _ in csv.DictReader(io.StringIO(before.decode('utf-8-sig'))))
        elif source.suffix == '.jsonl':
            expected = sum(bool(line.strip()) for line in before.splitlines())
        else:
            document = json.loads(before)
            expected = len(document) if isinstance(document, list) else sum(
                len(rows) for rows in document.values() if isinstance(rows, list))
        counts = import_attempts(store, source)
        assert counts['imported'] + counts['unbound'] == expected
        assert counts['closed'] == 0
        assert source.read_bytes() == before


def test_hundred_historical_rows_are_accounted_for(tmp_path):
    path = tmp_path / 'history.jsonl'
    path.write_text(''.join(json.dumps({'id': str(i), 'move': 'lookup'}) + '\n'
                            for i in range(160)))
    assert import_attempts(Store(tmp_path / 'store'), path) == {
        'imported': 0, 'unbound': 160, 'priced': 0, 'closed': 0}


def test_actual_multi_hundred_event_log_stays_unbound_without_contracts(tmp_path):
    source = Path('/home/dev/algorithms/quantum-circuit-patterns/experiments/chem-precision-relay-20261116/final/events.jsonl')
    if not source.is_file():
        pytest.skip('local historical event log is not available')
    original = source.read_bytes()
    rows = [json.loads(line) for line in original.splitlines() if line.strip()]
    assert len(rows) >= 200
    store = Store(tmp_path / 'store')
    counts = import_attempts(store, source)
    assert counts == {'imported': 0, 'unbound': len(rows), 'priced': 0, 'closed': 0}
    assert source.read_bytes() == original
