#!/usr/bin/env python3
"""Publish explicitly post-hoc CLI yield coverage on the original nine sources."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'yield_fresh'))
spec = importlib.util.spec_from_file_location('fresh_revalidator', HERE / 'yield_fresh' / 'check_results.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True)
    parser.add_argument('--wide', required=True)
    args = parser.parse_args()
    destinations = [HERE / f'{name}-cli-exposed-summary.json' for name in ['previous', 'wide']]
    destinations.append(HERE / 'wide-cli-exposed-comparison.json')
    if any(path.exists() for path in destinations): raise RuntimeError('Post-hoc CLI receipts already published')
    summaries = {name: validator.validate(getattr(args, name), fixture_dir=HERE) for name in ['previous', 'wide']}
    yields = json.loads((HERE / 'candidate-yield-wide-exposed.json').read_text())
    records = {name: {row['name']: row for row in summary['results']} for name, summary in summaries.items()}
    coverage = {name: {row['name']: row for row in report['results']} for name, report in yields['results'].items()}
    assert all(summary['manifest']['code']['sha256'] == yields['results'][name]['source_code'] for name, summary in summaries.items())
    rows = []
    for fixture, row in records['wide'].items():
        rows.append(dict(name=fixture, previous_best=records['previous'][fixture].get('after_cost'), wide_best=row.get('after_cost'),
                         canonical_cheaper_designs={name: data[fixture]['distinct_verified_cheaper_designs'] for name, data in coverage.items()},
                         paid_function_sets={name: data[fixture]['distinct_paid_function_sets'] for name, data in coverage.items()},
                         wide_below_previous_best=coverage['wide'][fixture]['distinct_verified_cheaper_than_previous_best']))
    report = dict(provenance='POST-HOC EXPOSED ORIGINAL-NINE CORPUS. These are historical corpus coverage measurements, not fresh holdout evidence.',
                  fixture_binding=yields['fixture_binding'], results=rows,
                  all_selected_bytes_replayed=True, original_receipts_preserved=True,
                  counts={name: {key: data[key] for key in ['verified_cheaper_contract_design_pairs',
                              'verified_cheaper_than_previous_best_pairs', 'paid_function_set_contract_pairs']}
                          for name, data in yields['results'].items()},
                  caveat='Canonical programs are not distinct algorithms. Paid-function fingerprints and resource fronts expose repeated affine/basis and cleanup variants.')
    for name in summaries: shutil.copyfile(Path(getattr(args, name)) / 'summary.json', HERE / f'{name}-cli-exposed-summary.json')
    (HERE / 'wide-cli-exposed-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['counts'], indent=2))
