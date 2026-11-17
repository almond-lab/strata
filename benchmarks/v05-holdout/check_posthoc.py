#!/usr/bin/env python3
"""Preserve an explicitly exposed-case rerun without overwriting original evidence."""
import argparse
import json
from pathlib import Path
import shutil

import check_results
import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True)
    parser.add_argument('--corrected', required=True)
    args = parser.parse_args()
    destination = run.HERE / 'exposed-nine-posthoc-summary.json'
    comparison_path = run.HERE / 'exposed-nine-posthoc-comparison.json'
    if destination.exists() or comparison_path.exists(): raise RuntimeError('Post-hoc receipts already published')
    previous, corrected = [check_results.check(getattr(args, name)) for name in ['previous', 'corrected']]
    before = {row['name']: row for row in previous['results']}
    rows = [dict(name=row['name'], previous_status=before[row['name']]['status'], corrected_status=row['status'],
                 previous_cost=before[row['name']].get('after_cost'), corrected_cost=row.get('after_cost'),
                 corrected_mechanism=row.get('selected_mechanism'), native=row.get('independent_native_check'))
            for row in corrected['results']]
    comparison = dict(provenance='POST-HOC EXPOSED-CASE RERUN. The original nine-case phase failure was already known; these results are not fresh holdout evidence.',
                      fixture_binding=corrected['manifest']['fixture_binding'], results=rows,
                      code=dict(previous=previous['manifest']['code']['sha256'], corrected=corrected['manifest']['code']['sha256']),
                      discovery_wall_seconds=dict(previous=previous['total_discovery_seconds'], corrected=corrected['total_discovery_seconds']),
                      original_receipts_preserved=True, all_selected_bytes_replayed=True, physical_layouts_checked=True)
    shutil.copyfile(Path(args.corrected) / 'summary.json', destination)
    comparison_path.write_text(json.dumps(comparison, indent=2) + '\n')
    print(json.dumps(comparison, indent=2))
