#!/usr/bin/env python3
"""Recheck retained bytes, physical layouts and frozen bindings; write compact receipts."""
import argparse
import json
from pathlib import Path
import shutil

import kickmix as km

import run


def check(root):
    root = Path(root).resolve()
    result = json.loads((root / 'summary.json').read_text())
    contract, fixtures = run.frozen()
    assert result['manifest']['fixture_binding'] == contract['fixtures_sha256']
    assert result['manifest']['runner_sha256'] == run.sha((run.HERE / 'run.py').read_bytes())
    assert result['manifest']['code'] == run.snapshot(Path(result['manifest']['package']))
    by_name = {row['name']: row for row in fixtures}
    for row in result['results']:
        if row['status'] in {'error', 'timeout'}:
            continue
        fixture = by_name[row['name']]
        source = root / row['name'] / 'source.kmx'
        candidate = root / row['name'] / 'search' / 'best.kmx'
        assert source.read_bytes() == fixture['text'].encode()
        before, after = km.Circuit.from_file(source), km.Circuit.from_file(candidate)
        assert run.sha(str(after).encode()) == row['candidate_sha256']
        assert after.num_qubits <= before.num_qubits
        assert set(range(after.num_qubits, before.num_qubits)) <= set(fixture['clean_inputs']), 'Arbitrary physical input retired'
        assert all(bit >= before.num_bits for bit in row['private_bits'])
        again = run.native_equal(before, after, fixture['clean_inputs'], row['private_bits'], contract['native_case_cap'])
        assert again == row['independent_native_check']
        assert again['equivalent'] is True
        assert row['oracle_mutations'] == dict(changed_value=False, changed_phase=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--released', required=True)
    parser.add_argument('--full', required=True)
    parser.add_argument('--ablation', required=True)
    args = parser.parse_args()
    destinations = [run.HERE / f'{label}-summary.json' for label in ['released', 'full', 'ablation']]
    destinations.append(run.HERE / 'comparison.json')
    if any(path.exists() for path in destinations):
        raise RuntimeError('Original published receipts are immutable; use a separately named comparison')
    roots = {label: Path(getattr(args, label)) for label in ['released', 'full', 'ablation']}
    results = {label: check(root) for label, root in roots.items()}
    for label, root in roots.items():
        shutil.copyfile(root / 'summary.json', run.HERE / f'{label}-summary.json')
    assert results['full']['manifest']['code'] == results['ablation']['manifest']['code']
    by_label = {label: {row['name']: row for row in result['results']} for label, result in results.items()}
    rows = []
    for name in by_label['full']:
        released, full, ablation = [by_label[label][name] for label in ['released', 'full', 'ablation']]
        if any(row['status'] in {'error', 'timeout'} for row in [released, full, ablation]):
            rows.append(dict(name=name, status='incomplete', results=[released, full, ablation]))
            continue
        rows.append(dict(name=name, family=full['family'], role=full['role'], source_cost=full['before_cost'],
                         released_cost=released['after_cost'], full_cost=full['after_cost'],
                         ablation_cost=ablation['after_cost'], full_old_narrow_cost=full['old_narrow_criterion']['cost'],
                         ablation_old_narrow_cost=ablation['old_narrow_criterion']['cost'],
                         full_mechanism=full['selected_mechanism'],
                         full_vs_released='win' if full['after_cost'] < released['after_cost'] else
                            'loss' if full['after_cost'] > released['after_cost'] else 'tie',
                         full_vs_ablation='win' if full['after_cost'] < ablation['after_cost'] else
                            'loss' if full['after_cost'] > ablation['after_cost'] else 'tie'))
    comparison = dict(fixture_binding=results['full']['manifest']['fixture_binding'],
                      source_code={label: result['manifest']['code']['sha256'] for label, result in results.items()},
                      results=rows, all_selected_bytes_replayed=True, physical_layouts_checked=True,
                      selected_native_boundaries=sum(result['independent_exhaustive_passes'] for result in results.values()),
                      mutation_rejections=2 * sum(result['independent_exhaustive_passes'] for result in results.values()),
                      discovery_wall_seconds={label: result['total_discovery_seconds'] for label, result in results.items()},
                      max_worker_peak_rss_bytes={label: result['max_worker_peak_rss_bytes'] for label, result in results.items()},
                      comparisons={opponent: {outcome: sum(row.get('full_vs_' + opponent) == outcome for row in rows)
                                              for outcome in ['win', 'tie', 'loss']}
                                   for opponent in ['released', 'ablation']},
                      caveats=['The three runs ran concurrently; actual discovery wall times are recorded, not equal CPU time.',
                               'Released v0.4.2 lacks deep-only mode and retains its broad default catalog.',
                               'Ablation removes only the three new families but retains current eligibility; old narrow bank-only selection is separately reported.',
                               'The released engine has no removed_materialization_bank annotation; its raw old-narrow fields are unavailable as a comparable criterion.',
                               'All selected holdout winners happened to be measurement-free; forced-outcome coverage is exercised by the oracle self-check, not evidence of measured holdout discovery.',
                               'This prospective independent-agent suite is not a blind historical or official score benchmark. No global optimality claim.'])
    (run.HERE / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    print(json.dumps({key: comparison[key] for key in ['comparisons', 'selected_native_boundaries', 'mutation_rejections', 'discovery_wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
