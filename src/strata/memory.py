"""Conservative legacy attempt import: narrative verdicts cannot create proof."""
import json
from .model import StrataError


def import_attempts(store,path,contract_map):
    counts={'imported':0,'unbound':0,'priced':0,'closed':0}
    with open(path) as f:
        for number,line in enumerate(f,1):
            if not line.strip():continue
            row=json.loads(line)
            identifier=str(row.get('id',row.get('attempt',number)))
            contract=contract_map.get(identifier)
            if not contract:
                store.put('unbound_attempt',dict(line=number,original=row))
                counts['unbound']+=1
                continue
            # Structured evidence only. A prose string saying "proof" isn't a proof artifact.
            evidence=row.get('evidence',{})
            kind=evidence.get('kind','estimate') if isinstance(evidence,dict) else 'estimate'
            if kind not in {'proof','exact_small','build'} or not evidence.get('artifact_hash'):
                kind='estimate'
            if kind != 'estimate':
                try:store.read_blob(evidence['artifact_hash'])
                except (OSError,StrataError):kind='estimate'
            scale=evidence.get('scale') if isinstance(evidence,dict) else None
            if kind=='exact_small' and not scale:kind='estimate'
            store.negative(contract,row.get('move',row.get('mechanism','unknown')),kind,
                dict(original=row,import_line=number),row.get('reopen_if',row.get('reopen','new evidence or changed contract')),scale)
            counts['imported']+=1
            counts['priced' if kind=='estimate' else 'closed']+=1
    return counts
