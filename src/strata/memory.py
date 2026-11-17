"""Import historical research without turning narrative verdicts into proof."""
import csv
import io
import json
from pathlib import Path
from .model import StrataError, canonical, region_from_dict


STRONG_KINDS = {'proof', 'exact_small', 'build'}


def _rows(data, suffix):
    text = data.decode('utf-8-sig')
    if suffix == '.csv':
        rows = []
        for number, row in enumerate(csv.DictReader(io.StringIO(text)), 2):
            # Preserve prose, but accept structured evidence/scale serialized by
            # CSV writers. Other columns remain exactly as supplied.
            for key in ('evidence', 'scale'):
                if row.get(key, '').lstrip().startswith(('{', '[')):
                    try:
                        row[key] = json.loads(row[key])
                    except ValueError:
                        pass
            rows.append((number, row))
        return rows
    stripped = text.lstrip()
    if stripped.startswith('['):
        parsed = json.loads(text)
        return list(enumerate(parsed, 1))
    try:
        parsed = json.loads(text)
    except ValueError:
        return [(number, json.loads(line)) for number, line in enumerate(text.splitlines(), 1) if line.strip()]
    if isinstance(parsed, dict):
        # Real research notes also occur as named arrays in one JSON document.
        arrays = [(name, value) for name, value in parsed.items() if isinstance(value, list)]
        if arrays and not any(key in parsed for key in ('id', 'attempt', 'mechanism', 'mechanism_id', 'move')):
            return [(f'{name}:{number}', row) for name, rows in arrays for number, row in enumerate(rows, 1)]
        return [(1, parsed)]
    raise StrataError('attempt log must contain JSON objects or arrays of notes')


def import_attempts(store, path, contract_map=None):
    """Read JSONL, CSV or JSON notes; caller mappings override recorded bindings.

    ``contract_map`` maps row id/attempt/name (or its one-based position) to a
    contract fingerprint. Rows with no known binding remain explicitly unbound.
    A fingerprint binds semantics, never a task label or a claimed proof tier.
    """
    path = Path(path)
    raw = path.read_bytes()
    try:
        rows = _rows(raw, path.suffix.lower())
    except (ValueError, UnicodeError, csv.Error) as exc:
        raise StrataError(f'invalid attempt log: {exc}') from exc
    # Validate the whole input before retaining any of its rows.
    for number, row in rows:
        if not isinstance(row, (dict, str)) or not row:
            raise StrataError(f'attempt {number} must be a nonempty object or note string')
        canonical(row)
    source_hash = store.blob(raw)
    contract_map = {str(key): value for key, value in (contract_map or {}).items()}
    counts = {'imported': 0, 'unbound': 0, 'priced': 0, 'closed': 0}
    for number, original in rows:
        row = original if isinstance(original, dict) else {'note': original}
        identifier = str(row.get('id', row.get('attempt', row.get('name', number))))
        contract = contract_map.get(identifier, row.get('contract_fingerprint'))
        if isinstance(contract, str):
            contract = contract.strip()
        if contract is not None and not isinstance(contract, str):
            raise StrataError('contract bindings must be fingerprint strings')
        provenance = dict(line=number, source=str(path.resolve()), source_blob=source_hash,
                          identifier=identifier, original=original)
        if not contract or contract.lower() in {'unknown', 'unbound', 'none', 'n/a'}:
            store.put('unbound_attempt', dict(**provenance, binding_status='unbound'))
            counts['unbound'] += 1
            continue
        evidence = row.get('evidence', {})
        structured = evidence if isinstance(evidence, dict) else {}
        kind = structured.get('kind', row.get('evidence_kind', row.get('evidence_tier', 'estimate')))
        # Narrative labels stay in original; only vocabulary with checked blobs
        # is eligible for stronger status.
        kind = kind if isinstance(kind, str) and kind in STRONG_KINDS | {'estimate'} else 'estimate'
        scale = structured.get('scale', row.get('scale'))
        artifact = structured.get('artifact_hash', row.get('artifact_hash'))
        handle = store.negative(contract,
            row.get('move') or row.get('mechanism') or row.get('mechanism_id') or 'unknown',
            kind, provenance,
            row.get('reopen_if') or row.get('reopen') or row.get('reopening_condition') or 'new evidence or changed contract',
            scale, evidence={**structured, 'artifact_hash': artifact})
        record = store.get(handle, 'negative')
        counts['imported'] += 1
        counts['priced' if record['evidence_kind'] == 'estimate' else 'closed'] += 1
    return counts


def checked_evidence(store, record):
    """Recheck saved claims without mutating old records or trusting old labels."""
    kind = record.get('evidence_kind', 'estimate')
    if kind not in STRONG_KINDS:
        return 'estimate', record.get('evidence_status', 'estimate_only')
    evidence = record.get('evidence', {})
    artifact = evidence.get('artifact_hash') if isinstance(evidence, dict) else evidence
    try:
        store.read_blob(artifact)
    except (OSError, StrataError):
        return 'estimate', 'missing_or_invalid_artifact'
    if kind == 'exact_small' and not record.get('scale'):
        return 'estimate', 'missing_exact_size'
    return kind, 'artifact_hash_verified'


def apply_memory(store, base=None, *, contract=None, scale=None):
    """Report reusable and reopened evidence for a saved design or fingerprint.

    No record is rewritten. Exact-small evidence applies only at its exact size;
    a changed contract reopens the question even when the old proof is intact.
    Build failures remain evidence about a construction, not a mechanism family.
    """
    contracts = set()
    target = base
    if contract:
        contracts.add(contract)
    if base is not None:
        if isinstance(base, dict):
            record = base
            target = None
        else:
            record = store.get(base, 'design')
        if record.get('tree'):
            tree = region_from_dict(record['tree'])
            contracts.update(region.contract.fingerprint for region in tree.walk())
        if record.get('contract_fingerprint'):
            contracts.add(record['contract_fingerprint'])
        if scale is None:
            scale = record.get('parameters')
    if not contracts:
        raise StrataError('memory application needs a stored design or explicit contract fingerprint')
    items = []
    for handle, record in store.all('negative'):
        kind, evidence_status = checked_evidence(store, record)
        same_contract = record.get('contract') in contracts
        scoped = record.get('scale')
        same_size = not scoped or (scale is not None and canonical(scoped) == canonical(scale))
        if not same_contract:
            disposition, reason = 'reopen', 'contract_changed'
        elif not same_size:
            disposition, reason = 'reopen', 'scale_changed_or_unknown'
        elif kind == 'estimate':
            disposition, reason = 'priced', evidence_status
        else:
            disposition = 'closed_at_size' if kind == 'exact_small' else 'closed'
            reason = 'same_contract_and_evidence_scope'
        items.append(dict(handle=handle, contract=record.get('contract'), move=record.get('move'),
                          requested_evidence_kind=record.get('requested_evidence_kind', record.get('evidence_kind')),
                          evidence_kind=kind, evidence_status=evidence_status,
                          scale=scoped, disposition=disposition, reason=reason,
                          reopen=record.get('reopen'), full_size_open=kind == 'exact_small',
                          closure_scope='this construction only' if kind == 'build' else 'recorded contract and scale'))
    unbound = [dict(handle=h, identifier=v.get('identifier'), binding_status='unbound')
               for h, v in store.all('unbound_attempt')]
    return dict(base=target, contracts=sorted(contracts), scale=scale, items=items, unbound=unbound,
                counts={name: sum(item['disposition'] == name for item in items)
                        for name in ('priced', 'closed', 'closed_at_size', 'reopen')})
