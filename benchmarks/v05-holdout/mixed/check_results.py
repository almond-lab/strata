#!/usr/bin/env python3
"""Validate mixed snapshots and preserve a separate prospective comparison."""
import argparse
import json
from pathlib import Path
import shutil

import kickmix as km

import run


def validate(root):
    root = Path(root).resolve()
    receipt = json.loads((root / 'summary.json').read_text())
    contract, fixtures = run.base.frozen()
    assert receipt['manifest']['fixture_binding'] == contract['fixtures_sha256']
    assert receipt['manifest']['runner_sha256'] == run.base.sha((run.HERE / 'run.py').read_bytes())
    assert receipt['manifest']['base_runner_sha256'] == run.base.sha(run.BASE_PATH.read_bytes())
    assert receipt['manifest']['code'] == run.base.snapshot(Path(receipt['manifest']['package']))
    by_name = {row['name']: row for row in fixtures}
    for row in receipt['results']:
        if row['status'] in {'error', 'timeout'}: continue
        fixture = by_name[row['name']]
        source = root / row['name'] / 'source.kmx'
        candidate = root / row['name'] / 'search' / 'best.kmx'
        assert source.read_bytes() == fixture['text'].encode()
        before, after = km.Circuit.from_file(source), km.Circuit.from_file(candidate)
        assert run.base.sha(str(after).encode()) == row['candidate_sha256']
        assert after.num_qubits <= before.num_qubits
        assert set(range(after.num_qubits, before.num_qubits)) <= set(fixture['clean_inputs'])
        assert all(bit >= before.num_bits for bit in row['private_bits'])
        check = run.base.native_equal(before, after, fixture['clean_inputs'], row['private_bits'], contract['native_case_cap'])
        assert check == row['independent_native_check'] and check['equivalent'] is True
        assert row['oracle_mutations'] == dict(changed_value=False, changed_phase=False)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True)
    parser.add_argument('--corrected', required=True)
    args = parser.parse_args()
    receipts = {name: validate(getattr(args, name)) for name in ['previous', 'corrected']}
    destinations = [run.HERE / f'{name}-summary.json' for name in receipts]
    if any(path.exists() for path in destinations): raise RuntimeError('Published mixed receipts already exist; use a new version')
    for name in receipts:
        shutil.copyfile(Path(getattr(args, name)) / 'summary.json', run.HERE / f'{name}-summary.json')
    previous = {row['name']: row for row in receipts['previous']['results']}
    rows = []
    for row in receipts['corrected']['results']:
        old = previous[row['name']]
        rows.append(dict(name=row['name'], previous_status=old['status'], corrected_status=row['status'],
                         previous_cost=old.get('after_cost'), corrected_cost=row.get('after_cost'),
                         corrected_mechanism=row.get('selected_mechanism'),
                         corrected_native=row.get('independent_native_check')))
    comparison = dict(fixture_binding=receipts['corrected']['manifest']['fixture_binding'], results=rows,
                      source_code={name: receipt['manifest']['code']['sha256'] for name, receipt in receipts.items()},
                      discovery_wall_seconds={name: receipt['discovery_wall_seconds'] for name, receipt in receipts.items()},
                      process_wall_seconds={name: receipt['process_wall_seconds'] for name, receipt in receipts.items()},
                      all_selected_bytes_replayed=True, physical_layouts_checked=True,
                      provenance='New mixed holdouts frozen before corrected-engine results; prior nine-case phase failure was already exposed.',
                      limits='Fixed short ceilings; no global optimum, official score, or generic fractional-phase construction claim.')
    (run.HERE / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    print(json.dumps(comparison, indent=2))
