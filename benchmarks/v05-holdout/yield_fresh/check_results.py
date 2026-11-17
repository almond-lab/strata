#!/usr/bin/env python3
"""Publish the fresh CLI comparison after replaying retained bytes again."""
import argparse
import json
from pathlib import Path
import shutil

import kickmix as km

import run


def validate(root, fixture_dir=None):
    root = Path(root).resolve()
    receipt = json.loads((root / 'summary.json').read_text())
    run.oracle.HERE = Path(fixture_dir).resolve() if fixture_dir else run.HERE
    contract, fixtures = run.oracle.frozen()
    assert receipt['manifest']['fixture_binding'] == contract['fixtures_sha256']
    assert receipt['manifest']['code'] == run.oracle.snapshot(Path(receipt['manifest']['package']))
    expected_runner = receipt['manifest']['runner_sha256']
    if run.oracle.sha((run.HERE / 'run.py').read_bytes()) != expected_runner:
        archived = run.HERE.parent / 'work' / 'runners' / (expected_runner + '.py')
        assert run.oracle.sha(archived.read_bytes()) == expected_runner
    assert run.oracle.sha(run.ORACLE_PATH.read_bytes()) == receipt['manifest']['oracle_sha256']
    by_name = {row['name']: row for row in fixtures}
    for row in receipt['results']:
        if row['status'] in {'error', 'timeout'}: continue
        fixture = by_name[row['name']]
        source = root / row['name'] / 'source.kmx'
        candidate = root / row['name'] / 'search' / 'best.kmx'
        assert source.read_bytes() == fixture['text'].encode()
        before, after = km.Circuit.from_file(source), km.Circuit.from_file(candidate)
        assert run.oracle.sha(str(after).encode()) == row['candidate_sha256']
        assert after.num_qubits <= before.num_qubits
        assert set(range(after.num_qubits, before.num_qubits)) <= set(fixture['clean_inputs'])
        assert all(bit >= before.num_bits for bit in row['private_bits'])
        check = run.oracle.native_equal(before, after, fixture['clean_inputs'], row['private_bits'], contract['native_case_cap'])
        assert check == row['independent_native_check'] and check['equivalent'] is True
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['released', 'previous', 'wide']: parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name)) for name in ['released', 'previous', 'wide']}
    targets = [run.HERE / f'{name}-summary.json' for name in paths] + [run.HERE / 'comparison.json']
    if any(path.exists() for path in targets): raise RuntimeError('Fresh published comparison already exists')
    summaries = {name: validate(path) for name, path in paths.items()}
    yields = json.loads((run.HERE / 'candidate-yield.json').read_text())
    assert all(yields['results'][name]['source_code'] == summary['manifest']['code']['sha256'] for name, summary in summaries.items())
    by_engine = {name: {row['name']: row for row in summary['results']} for name, summary in summaries.items()}
    yield_rows = {name: {row['name']: row for row in report['results']} for name, report in yields['results'].items()}
    rows = []
    for fixture in by_engine['wide']:
        row = dict(name=fixture, best_cost={name: records[fixture].get('after_cost') for name, records in by_engine.items()},
                   verified_cheaper_designs={name: records[fixture]['distinct_verified_cheaper_designs'] for name, records in yield_rows.items()},
                   paid_function_sets={name: records[fixture]['distinct_paid_function_sets'] for name, records in yield_rows.items()},
                   wide_designs_below_previous_best=yield_rows['wide'][fixture]['distinct_verified_cheaper_than_previous_best'],
                   previous_front=yield_rows['previous'][fixture]['verified_resource_front'],
                   wide_front=yield_rows['wide'][fixture]['verified_resource_front'])
        rows.append(row)
    report = dict(fixture_binding=yields['fixture_binding'], source_code={name: value['source_code'] for name, value in yields['results'].items()},
                  results=rows, all_selected_bytes_replayed=True, physical_layouts_checked=True,
                  independent_native_selected_passes=sum(value['independent_exhaustive_passes'] for value in summaries.values()),
                  discovery_counts={name: dict(verified_cheaper_designs=value['verified_cheaper_contract_design_pairs'],
                                               cheaper_than_previous_best=value['verified_cheaper_than_previous_best_pairs'],
                                               paid_function_set_contract_pairs=value['paid_function_set_contract_pairs'])
                                    for name, value in yields['results'].items()},
                  limits=['No 1-to-1000 result is demonstrated.',
                          'Release uses its broad API; both development snapshots use actual CLI defaults with matched explicit corpus budgets.',
                          'Available retained alternatives are independently rechecked; the archive is not a census of attempted designs.',
                          'Canonical executable programs, paid-function sets and resource-front points are different diversity measures.',
                          'Paid-function fingerprints are exact only for the bounded pure-circuit subset; measured cases stay explicitly unknown.'])
    for name, path in paths.items(): shutil.copyfile(path / 'summary.json', run.HERE / f'{name}-summary.json')
    (run.HERE / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['discovery_counts'], indent=2))
