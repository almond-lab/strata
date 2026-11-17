#!/usr/bin/env python3
"""Freeze new composed value/phase obligations before corrected-engine results."""
import hashlib
import json
from pathlib import Path
import random
import re

HERE = Path(__file__).resolve().parent
SEED = 731809


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rename(lines, clean, targets, seed, holes=0):
    rng = random.Random(seed)
    width = 1 + max(int(q) for line in lines for q in re.findall(r'\bq(\d+)\b', line))
    public = [q for q in range(width) if q not in clean]
    positions = list(range(len(public) + holes))
    if holes: positions.remove(1)
    rng.shuffle(positions)
    helpers = list(range(len(public) + holes, width + holes))
    rng.shuffle(helpers)
    mapping = dict(zip(public, positions)) | dict(zip(clean, helpers))
    return ([re.sub(r'\bq(\d+)\b', lambda m: f'q{mapping[int(m[1])]}', line) for line in lines],
            sorted(mapping[q] for q in clean), [mapping[q] for q in targets])


def fixtures():
    bank = ['CCX q0 q1 q6', 'CCX q0 q2 q7', 'CCX q6 q3 q8', 'CCX q7 q3 q9']
    output = ['CX q8 q4', 'CX q9 q4', 'CX q6 q5', 'CX q7 q5']
    first = ['X q1'] + bank + output + ['CZ q4 q0'] + bank[::-1] + ['X q1']
    second_bank = ['CCX q0 q1 q5', 'CCX q0 q2 q6', 'CCX q5 q2 q7', 'CCX q6 q1 q8']
    second = (['CX q2 q1'] + second_bank + ['CX q7 q3', 'CX q8 q4', 'CZ q3 q4', 'CX q3 q4']
              + second_bank[::-1] + ['CX q2 q1'])
    fractional = ['X q1'] + bank + output + ['Z_POW q4 0.125'] + bank[::-1] + ['X q1']
    rows = []
    for index, (name, lines, clean, targets, role, holes) in enumerate([
        ('updated_dirty_target_phase', first, [6, 7, 8, 9], [4, 5], 'mixed_opportunity', 1),
        ('dependent_observable_then_adapter', second, [5, 6, 7, 8], [3, 4], 'mixed_opportunity', 0),
        ('fractional_observable_guard', fractional, [6, 7, 8, 9], [4, 5], 'unsupported_fractional_guard', 0),
    ]):
        lines, clean, targets = rename(lines, clean, targets, SEED + index, holes)
        text = '\n'.join(lines) + '\n'
        rows.append(dict(name=name, family='composed_dirty_value_and_observable', role=role,
                         seed=SEED + index, text=text, sha256=sha(text.encode()),
                         clean_inputs=clean, output_wires=targets, internal_bits=[]))
    return rows


if __name__ == '__main__':
    target = HERE / 'fixtures.json'
    if target.exists(): raise RuntimeError('Mixed holdouts already frozen; regeneration is prohibited')
    rows = fixtures()
    target.write_text(json.dumps(rows, indent=2) + '\n')
    contract = dict(generator_seed=SEED, generator_sha256=sha(Path(__file__).read_bytes()),
                    fixtures_sha256=sha(target.read_bytes()), fixture_count=len(rows),
                    frozen_before_first_search=True, no_replacement_or_region_supplied=True,
                    provenance='New independent composed value/phase cases, frozen after the original nine-case phase failure was exposed and before any corrected-engine candidate result. Parent received no fixture details before freeze.',
                    objective='Q*toffoli', budgets=dict(rounds=1, beam=4, max_candidates=12,
                        max_operations=128, max_variables=32, max_terms=65536, max_product_terms=65536),
                    fixture_timeout_seconds=45, native_case_cap=65536,
                    source_hashes={row['name']: row['sha256'] for row in rows})
    (HERE / 'freeze.json').write_text(json.dumps(contract, indent=2) + '\n')
    print(json.dumps(contract, indent=2))
