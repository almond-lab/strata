#!/usr/bin/env python3
"""Compare immutable engines on separately frozen composed value/phase cases."""
import argparse
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent / 'run.py'
spec = importlib.util.spec_from_file_location('holdout_base', BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.HERE = HERE
sign_trajectory = base.trajectory


def exact_phase_trajectory(circuit, quantum_width, classical_width, public, assignments, outcomes_at):
    import kickmix as km
    import numpy as np
    lines = str(circuit).splitlines()
    if any(line.split()[0] in {'HMR', 'R'} for line in lines):
        values, bits, phase = sign_trajectory(circuit, quantum_width, classical_width, public,
                                              assignments, outcomes_at)
        return values, bits, np.array([Fraction(int(value)) for value in phase], dtype=object)
    sim = km.Simulator(batch_size=len(assignments), seed=68119)
    wires, bits = [km.q(q) for q in range(quantum_width)], [km.b(b) for b in range(classical_width)]
    values = np.zeros((quantum_width, len(assignments)), dtype=np.bool_)
    for level, q in enumerate(public): values[q] = (assignments >> level) & 1
    if wires: sim.write_across_shots(wires, values)
    if bits:
        sim.write_across_shots(bits, np.array([((assignments >> (len(public) + b)) & 1).astype(bool)
                                              for b in range(classical_width)]))
    sim.do(circuit)
    phase = np.array([Fraction(sim.read_shot_phase(shot)) % 2 for shot in range(len(assignments))], dtype=object)
    return sim.read_across_shots(wires), sim.read_across_shots(bits) if bits else None, phase


base.trajectory = exact_phase_trajectory


def run(args):
    if args.package_root: sys.path.insert(0, str(Path(args.package_root).resolve()))
    package = Path(importlib.util.find_spec('strata').origin).resolve().parent
    code = base.snapshot(package)
    contract, fixtures = base.frozen()
    root = Path(args.out).resolve()
    if root.exists() and any(root.iterdir()): raise RuntimeError('Fresh output required; no reused archive')
    root.mkdir(parents=True, exist_ok=True)
    runner_hash, base_hash = base.sha(Path(__file__).read_bytes()), base.sha(BASE_PATH.read_bytes())
    manifest = dict(label=args.label, package=str(package), code=code, contract=contract,
                    fixture_binding=contract['fixtures_sha256'], runner_sha256=runner_hash,
                    base_runner_sha256=base_hash, oracle_selfcheck=base.oracle_selfcheck(contract['native_case_cap']),
                    phase_comparison='exact native Fraction modulo two; no sign-only projection for unmeasured programs',
                    no_supplied_region_or_replacement=True, no_custom_constructor_implementation=True,
                    provenance='New mixed holdouts frozen before any corrected-engine candidate result; earlier nine-case failure was known.')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows, begin = [], time.monotonic()
    for fixture in fixtures:
        destination = root / fixture['name']
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', fixture['name'],
                   '--out', str(destination), '--label', args.label]
        if args.package_root: command += ['--package-root', str(Path(args.package_root).resolve())]
        started = time.monotonic()
        try:
            completed = subprocess.run(command, text=True, capture_output=True,
                                       timeout=contract['fixture_timeout_seconds'])
            if completed.returncode:
                row = dict(name=fixture['name'], status='error', stderr=completed.stderr,
                           seconds=time.monotonic() - started)
            else:
                row = json.loads((destination / 'result.json').read_text())
        except subprocess.TimeoutExpired:
            row = dict(name=fixture['name'], status='timeout', seconds=time.monotonic() - started)
        rows.append(row)
        print(json.dumps({key: row.get(key) for key in ['name', 'status', 'before_cost', 'after_cost']}), flush=True)
    assert base.snapshot(package) == code
    assert base.sha(Path(__file__).read_bytes()) == runner_hash and base.sha(BASE_PATH.read_bytes()) == base_hash
    base.frozen()
    summary = dict(manifest=manifest, results=rows, all_failures_retained=True,
                   independent_exhaustive_passes=sum(r.get('independent_native_check', {}).get('equivalent') is True for r in rows),
                   errors_or_timeouts=sum(r['status'] in {'error', 'timeout'} for r in rows),
                   discovery_wall_seconds=sum(r.get('discovery_seconds', 0) for r in rows),
                   process_wall_seconds=time.monotonic() - begin,
                   max_worker_peak_rss_bytes=max((r.get('process_peak_rss_bytes', 0) for r in rows), default=0))
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in ['independent_exhaustive_passes', 'errors_or_timeouts', 'discovery_wall_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--package-root')
    parser.add_argument('--label', required=True)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.mode = 'structural'
    base.worker(args) if args.worker else run(args)
