#!/usr/bin/env python3
"""Fresh small composition suite, frozen before wide-portfolio candidates exist."""
import hashlib
import json
from pathlib import Path
import random
import re

HERE = Path(__file__).resolve().parent
SEED = 260493


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rename(lines, clean, targets, seed):
    rng = random.Random(seed)
    width = 1 + max(int(q) for line in lines for q in re.findall(r'\bq(\d+)\b', line))
    public = [q for q in range(width) if q not in clean]
    public_ids, helpers = list(range(len(public))), list(range(len(public), width))
    rng.shuffle(public_ids); rng.shuffle(helpers)
    mapping = dict(zip(public, public_ids)) | dict(zip(clean, helpers))
    return ([re.sub(r'\bq(\d+)\b', lambda m: f'q{mapping[int(m[1])]}', line) for line in lines],
            sorted(mapping[q] for q in clean), [mapping[q] for q in targets])


def fixtures():
    bank = ['CCX q0 q1 q7', 'CCX q3 q4 q8', 'CCX q0 q3 q9',
            'CCX q7 q2 q10', 'CCX q8 q2 q11', 'CCX q9 q4 q12']
    consume = ['CX q7 q5', 'CX q10 q5', 'CX q11 q5', 'CX q10 q6', 'CX q12 q6']
    frame = ['CX q4 q1', 'X q2', 'CX q0 q3']
    selector = frame + bank + consume + bank[::-1] + frame[::-1]
    mixed = frame + bank + consume + ['CZ q5 q4', 'Z q6'] + bank[::-1] + frame[::-1]
    # Duplicate products and their descendants cancel at a dirty consumer.
    # A wide generator may expose many named/ordered versions of the same
    # one-product result; none should become independent discoveries merely
    # through changed helpers, display registers, or CCX control order.
    alias_bank = ['CCX q0 q1 q5', 'CCX q1 q0 q6', 'CCX q5 q2 q7', 'CCX q6 q2 q8']
    alias = alias_bank + ['CX q7 q3', 'CX q8 q3', 'CX q5 q3', 'CX q5 q4', 'CX q6 q4'] + alias_bank[::-1]
    rows = []
    for index, (name, lines, clean, targets, family, role) in enumerate([
        ('selector_shared_consumer', selector, list(range(7, 13)), [5, 6], 'selector_cross_output_composition', 'opportunity'),
        ('selector_dirty_observable', mixed, list(range(7, 13)), [5, 6], 'selector_cross_output_composition', 'mixed_opportunity'),
        ('alias_multiplicity_canary', alias, [5, 6, 7, 8], [3, 4], 'duplicate_materialization_consumer', 'deduplication_canary'),
    ]):
        lines, clean, targets = rename(lines, clean, targets, SEED + index)
        text = '\n'.join(lines) + '\n'
        rows.append(dict(name=name, family=family, role=role, seed=SEED + index,
                         text=text, sha256=sha(text.encode()), clean_inputs=clean,
                         output_wires=targets, internal_bits=[]))
    return rows


if __name__ == '__main__':
    target = HERE / 'fixtures.json'
    if target.exists(): raise RuntimeError('Fresh yield fixtures already frozen; cannot regenerate')
    rows = fixtures()
    target.write_text(json.dumps(rows, indent=2) + '\n')
    contract = dict(generator_seed=SEED, generator_sha256=sha(Path(__file__).read_bytes()),
                    fixtures_sha256=sha(target.read_bytes()), fixture_count=len(rows),
                    frozen_before_first_search=True, no_replacement_or_region_supplied=True,
                    provenance='Fresh review-agent compositions and a duplicate-materialization canary, frozen before seeing any wide-portfolio candidate. Earlier suites were already exposed; these exact bytes were not supplied to implementers.',
                    objective='Q*toffoli', budgets=dict(rounds=1, beam=4, max_candidates=12,
                        max_operations=128, max_variables=32, max_terms=65536, max_product_terms=65536),
                    fixture_timeout_seconds=45, native_case_cap=65536,
                    source_hashes={row['name']: row['sha256'] for row in rows})
    (HERE / 'freeze.json').write_text(json.dumps(contract, indent=2) + '\n')
    print(json.dumps(contract, indent=2))
