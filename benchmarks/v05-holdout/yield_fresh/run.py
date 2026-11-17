#!/usr/bin/env python3
"""Ordinary CLI-default discovery on a frozen small corpus, with native replay."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ORACLE_PATH = HERE.parent / 'run.py'
spec = importlib.util.spec_from_file_location('yield_native_oracle', ORACLE_PATH)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def worker(args):
    sys.path.insert(0, str(Path(args.package_root).resolve()))
    import kickmix as km
    from strata.cli import main
    oracle.HERE = Path(args.fixtures).resolve()
    contract, fixtures = oracle.frozen()
    fixture = next(row for row in fixtures if row['name'] == args.worker)
    root = Path(args.out).resolve(); root.mkdir(parents=True, exist_ok=False)
    source = root / 'source.kmx'; source.write_text(fixture['text'])
    command = ['search', str(source), '--out', str(root / 'search'), '--clean-inputs',
               ','.join(map(str, fixture['clean_inputs'])), '--cost', contract['objective']]
    for name, value in contract['budgets'].items(): command += ['--' + name.replace('_', '-'), str(value)]
    # No mechanism hints, registry modification, goal override, fast/slow
    # override, or new-only portfolio flags: use each snapshot's real CLI.
    with contextlib.redirect_stdout(io.StringIO()):
        if args.legacy_api:
            from strata.restructure import restructure_file
            restructure_file(source, root / 'search', clean_inputs=fixture['clean_inputs'],
                             cost=contract['objective'], **contract['budgets'])
            result = 0
        else:
            result = main(command)
    assert result == 0
    receipt = json.loads((root / 'search' / 'restructure.json').read_text())
    before, after = km.Circuit(fixture['text']), km.Circuit.from_file(receipt['best']['path'])
    private = receipt['best'].get('private_bits', [])
    assert all(b >= before.num_bits for b in private)
    assert after.num_qubits <= before.num_qubits
    assert set(range(after.num_qubits, before.num_qubits)) <= set(fixture['clean_inputs'])
    check = oracle.native_equal(before, after, fixture['clean_inputs'], private, contract['native_case_cap'])
    costs = [c.num_qubits * c.operation_statistics().num_toffoli_like for c in [before, after]]
    assert costs == [receipt['baseline']['cost'], receipt['best']['cost']]
    row = dict(name=fixture['name'], family=fixture['family'], role=fixture['role'],
               status=receipt['status'], source_sha256=fixture['sha256'],
               candidate_sha256=oracle.sha(str(after).encode()), before_cost=costs[0], after_cost=costs[1],
               selected_mechanism=receipt['best'].get('mechanism'), private_bits=private,
               independent_native_check=check, physical_layout_checked=True,
               retained_states=len(receipt.get('states', [])), reported_discovery_yield=receipt.get('discovery_yield'),
               search_control=receipt.get('search_control'), cli_arguments=None if args.legacy_api else command,
               legacy_broad_api=args.legacy_api)
    (root / 'result.json').write_text(json.dumps(row, indent=2) + '\n')


def run(args):
    sys.path.insert(0, str(Path(args.package_root).resolve()))
    package = Path(importlib.util.find_spec('strata').origin).resolve().parent
    code = oracle.snapshot(package)
    oracle.HERE = Path(args.fixtures).resolve()
    contract, fixtures = oracle.frozen()
    root = Path(args.out).resolve()
    if root.exists() and any(root.iterdir()): raise RuntimeError('Fresh output required')
    root.mkdir(parents=True, exist_ok=True)
    runner_hash = oracle.sha(Path(__file__).read_bytes())
    manifest = dict(label=args.label, package=str(package), code=code,
                    fixture_binding=contract['fixtures_sha256'], contract=contract,
                    runner_sha256=runner_hash, oracle_sha256=oracle.sha(ORACLE_PATH.read_bytes()),
                    mode=('released broad restructure_file catalog, no search CLI available, matched frozen budgets'
                          if args.legacy_api else 'ordinary CLI search defaults with common explicitly frozen source/semantic budgets'),
                    no_supplied_region_or_replacement=True, no_custom_constructor_implementation=True,
                    oracle_selfcheck=oracle.oracle_selfcheck(contract['native_case_cap']))
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows = []
    for fixture in fixtures:
        destination = root / fixture['name']
        command = [sys.executable, str(Path(__file__).resolve()), '--package-root', args.package_root,
                   '--fixtures', args.fixtures, '--worker', fixture['name'], '--out', str(destination), '--label', args.label]
        if args.legacy_api: command.append('--legacy-api')
        try:
            completed = subprocess.run(command, text=True, capture_output=True,
                                       timeout=contract['fixture_timeout_seconds'])
            row = (dict(name=fixture['name'], status='error', stderr=completed.stderr) if completed.returncode else
                   json.loads((destination / 'result.json').read_text()))
        except subprocess.TimeoutExpired:
            row = dict(name=fixture['name'], status='timeout')
        rows.append(row)
        print(json.dumps({key: row.get(key) for key in ['name', 'status', 'before_cost', 'after_cost', 'retained_states']}), flush=True)
    assert oracle.snapshot(package) == code
    assert oracle.sha(Path(__file__).read_bytes()) == runner_hash
    oracle.frozen()
    summary = dict(manifest=manifest, results=rows, all_failures_retained=True,
                   independent_exhaustive_passes=sum(row.get('independent_native_check', {}).get('equivalent') is True for row in rows),
                   errors_or_timeouts=sum(row['status'] in {'error', 'timeout'} for row in rows))
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--fixtures', default=str(HERE))
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--legacy-api', action='store_true', help='Use released broad API when search CLI is absent')
    args = parser.parse_args()
    worker(args) if args.worker else run(args)
