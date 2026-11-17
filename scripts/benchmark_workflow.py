"""Measure small existing-script orchestration through actual offline sandboxes.

This benchmark does not measure human onboarding or full quantum-circuit search.
It retains both the sequential command loop and the declared sweep results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

from strata.experiments import create_experiment, run_experiment, create_sweep, run_sweep
from strata.experiment_reports import experiment_report, agent_packet
from strata.store import Store


def run(output, *, count=20, workers=2):
    if not 1 <= count <= 100 or not 1 <= workers <= 8:
        raise ValueError('small reference benchmark accepts 1..100 settings and 1..8 workers')
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'benchmark.json').exists():
        raise ValueError('choose a new output directory for an independent benchmark run')
    examples = Path(__file__).resolve().parents[1] / 'examples' / 'workflow'
    manifest = examples / 'numeric.toml'
    settings = [{'steps': value} for value in range(10, 10 + count)]
    manual = Store(output / 'sequential')
    sweep_store = Store(output / 'sweep')
    try:
        start = time.monotonic()
        sequential = []
        for parameters in settings:
            experiment = create_experiment(manual, manifest, settings=parameters, seed=0)
            sequential.append(run_experiment(manual, experiment, owner='sequential-reference'))
        sequential_seconds = time.monotonic() - start
        started = time.monotonic()
        sweep = create_sweep(sweep_store, manifest, parameters=settings, seeds=[0],
                             replicates=1, max_jobs=count, workers=workers)
        result = run_sweep(sweep_store, sweep)
        sweep_seconds = time.monotonic() - started
        def normalized(store, rows):
            invalid = [dict(experiment=r.get('experiment'), status=r.get('status'),
                            error=r.get('error')) for r in rows
                       if r.get('status') not in {'accepted_by_verifier', 'rejected_by_verifier'}]
            if invalid:
                raise RuntimeError('reference pipeline did not reach its verifier: ' + json.dumps(invalid[:3]))
            return sorted((store.get(row['experiment'], 'experiment')['settings']['steps'],
                           row['status'], row['metrics']['mse']['value'],
                           row['cost']['value']) for row in rows)
        identical = normalized(manual, sequential) == normalized(sweep_store, result['results'])
        if not identical:
            raise RuntimeError('sequential and sweep outcomes differ')
        replay_start = time.monotonic()
        resumed = run_sweep(sweep_store, sweep)
        replay_seconds = time.monotonic() - replay_start
        report_started = time.monotonic()
        report = experiment_report(sweep_store)
        packet = agent_packet(sweep_store)
        report_seconds = time.monotonic() - report_started
        fixtures = []
        for name in ('circuit.json', 'words.json'):
            store = Store(output / name.removesuffix('.json'))
            try:
                start = time.monotonic()
                experiment = create_experiment(store, examples / name, seed=0)
                receipt = run_experiment(store, experiment, owner='genericity-reference')
                if receipt['status'] != 'accepted_by_verifier':
                    raise RuntimeError(f'{name} did not pass: {receipt}')
                fixtures.append(dict(manifest=name, seconds=time.monotonic() - start,
                                     experiment=experiment, result=receipt['handle'],
                                     status=receipt['status'], verified_cost=receipt.get('verified_cost')))
            finally:
                store.db.close()
        summary = dict(version=1, python=platform.python_version(), platform=platform.platform(),
            settings=count, workers=workers, actual_offline_sandbox=True,
            sequential_seconds=sequential_seconds, declared_sweep_seconds=sweep_seconds,
            identical_outcomes=identical,
            accepted=sum(r['status'] == 'accepted_by_verifier' for r in result['results']),
            rejected=sum(r['status'] == 'rejected_by_verifier' for r in result['results']),
            other_statuses=sorted({r['status'] for r in result['results']}
                                  - {'accepted_by_verifier', 'rejected_by_verifier'}),
            resumed_seconds=replay_seconds,
            accepted_results_reused=sum(r.get('reused') is True for r in resumed['results']),
            report_packet_seconds=report_seconds, report=report['record'], packet=packet['record'],
            genericity_fixtures=fixtures,
            controller_calls=dict(sequential_runs=count, declared_sweep_create_and_run=2,
                                  evidence='counted orchestration calls, not measured human interventions'),
            scope='Small reference pipelines. This measures launch/retention overhead and bounded concurrency, not human time saved, full circuit throughput, or production optimization gain.')
        (output / 'benchmark.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        manual.db.close()
        sweep_store.db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    run(args.out, count=args.count, workers=args.workers)
