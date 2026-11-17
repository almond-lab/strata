#!/usr/bin/env python3
"""Replay immutable structural fixtures against one installed Strata package.

Run current and baseline in separate invocations with the same interpreter.
The output directory must be new or empty, so learned archives cannot leak
between invocations. This runner supplies no replacement or marked region.
"""
import argparse
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import random
import resource
import sys
import time


HERE = Path(__file__).resolve().parent
OBJECTIVE = 'Q*toffoli'
BUDGETS = {
    'frozen': dict(rounds=1, beam=4, max_candidates=12, max_operations=200,
                   max_variables=32, max_terms=65536, max_product_terms=65536),
    'scale': dict(rounds=1, beam=4, max_candidates=12, max_operations=512,
                  max_variables=128, max_terms=65536, max_product_terms=65536),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def package_snapshot(root):
    files = {str(path.relative_to(root)): sha256(path.read_bytes())
             for path in sorted(root.rglob('*.py'))}
    return dict(files=files, sha256=sha256(json.dumps(files, sort_keys=True).encode()))


def load_fixtures(suite):
    path = HERE / ('fixtures.json' if suite == 'frozen' else 'scale-fixtures.json')
    raw = path.read_bytes()
    rows = json.loads(raw)
    for row in rows:
        if sha256(row['text'].encode()) != row['sha256']:
            raise RuntimeError('frozen input hash mismatch: ' + row['name'])
    return rows, dict(path=path.name, sha256=sha256(raw))


def native_boundary(circuit, width, clean, assignments, km, np):
    public = [q for q in range(width) if q not in clean]
    sim = km.Simulator(batch_size=len(assignments), seed=199)
    initial = np.zeros((width, len(assignments)), dtype=np.bool_)
    for bit, q in enumerate(public):
        initial[q] = [(row >> bit) & 1 for row in assignments]
    wires = [km.q(q) for q in range(width)]
    sim.write_across_shots(wires, initial)
    sim.do(circuit)
    return (sim.read_across_shots(wires),
            [Fraction(sim.read_shot_phase(row)) % 2 for row in range(len(assignments))])


def assignments_for(width, clean):
    public_bits = width - len(clean)
    count = 1 << public_bits
    if count <= 65536:
        return list(range(count)), dict(scope='all_public_inputs', cases=count,
                                       phase='exact native Fraction modulo two')
    rng = random.Random(199)
    rows = {0, count - 1, *(1 << bit for bit in range(public_bits))}
    while len(rows) < 256:
        rows.add(rng.randrange(count))
    return sorted(rows), dict(scope='native_smoke_sample_only', cases=len(rows),
                              full_public_cases=count, seed=199,
                              proof='complete input coverage comes only from symbolic boundary equality')


def summary_attempt(attempt):
    construction = attempt.get('construction', {})
    return dict(mechanism=attempt['mechanism'], status=attempt['status'],
                reason=attempt.get('reason'), cost_delta=attempt.get('cost_delta'),
                check_status=attempt.get('check', {}).get('status'),
                equivalent=attempt.get('check', {}).get('equivalent'),
                structural_witness=construction.get('structural_witness'),
                exact_search=construction.get('exact_search'),
                bounded_search=construction.get('search'))


def run(args):
    if args.package_root:
        sys.path.insert(0, str(Path(args.package_root).resolve()))
    spec = importlib.util.find_spec('strata')
    if not spec or not spec.origin:
        raise RuntimeError('Strata is unavailable; select an installed package with --package-root')
    package = Path(spec.origin).resolve().parent
    before_code = package_snapshot(package)
    import kickmix as km
    import numpy as np
    from strata.restructure import restructure_file

    root = Path(args.out).resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('benchmark output must be new or empty; existing archives cannot be reused')
    root.mkdir(parents=True, exist_ok=True)
    fixtures, fixture_binding = load_fixtures(args.suite)
    native_deep_only = 'deep_only' in inspect.signature(restructure_file).parameters
    mode = ('native_deep_only' if native_deep_only else
            'legacy_registry_with_posthoc_deep_acceptance_filter')
    manifest = dict(label=args.label, suite=args.suite, fixture_binding=fixture_binding,
                    objective=OBJECTIVE, budgets=BUDGETS[args.suite], search_mode=mode,
                    no_custom_constructor=True, no_supplied_cut_or_replacement=True,
                    fresh_archive_per_fixture=True, source_package=str(package),
                    source_code=before_code, runner_sha256=sha256(Path(__file__).read_bytes()),
                    runtime=dict(python=sys.version, platform=platform.platform(),
                                 kickmix=importlib.metadata.version('kickmix'), numpy=np.__version__),
                    provenance=('independent-agent generated fixtures frozen before their first search; targeted acceptance, not a blind historical benchmark'
                                if args.suite == 'frozen' else
                                'nonheldout symbolic-scale regression examples used during constructor development'),
                    equal_cpu_time_claim=False)
    (root / 'run-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (root / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    rows, started = [], time.monotonic()
    for fixture in fixtures:
        source = root / (fixture['name'] + '.kmx')
        source.write_text(fixture['text'])
        options = dict(BUDGETS[args.suite])
        if native_deep_only:
            options['deep_only'] = True
        begin = time.monotonic()
        result = restructure_file(source, root / fixture['name'], clean_inputs=fixture['clean'],
                                  cost=OBJECTIVE, **options)
        discovery_wall = time.monotonic() - begin
        if native_deep_only:
            selected = result['best']
        else:
            # v0.4.2 has no deep-only mode. Keep its native constructor catalog
            # unchanged; independently select only verified whole-bank gains.
            eligible = [state for state in result['states']
                        if state['cost'] < result['baseline']['cost']
                        and sum(q >= state['resources']['Q'] for q in fixture['clean']) >= 2]
            selected = min(eligible, key=lambda state: state['cost']) if eligible else result['baseline']
        before = km.Circuit(fixture['text'])
        after = km.Circuit(Path(selected['path']).read_text())
        cases, native_scope = assignments_for(before.num_qubits, fixture['clean'])
        native_scope['measurement_coverage'] = (
            'one native draw per checked input; all-outcome evidence is the complete symbolic boundary proof'
            if any(line.startswith('HMR ') for line in str(after).splitlines()) else 'no measurements')
        check_started = time.monotonic()
        original = native_boundary(before, before.num_qubits, fixture['clean'], cases, km, np)
        emitted = native_boundary(after, before.num_qubits, fixture['clean'], cases, km, np)
        native_equal = np.array_equal(original[0], emitted[0]) and original[1] == emitted[1]
        retired = [q for q in fixture['clean'] if q >= after.num_qubits]
        checked = [attempt for attempt in result['attempts'] if attempt.get('candidate') == selected['id']]
        complete_symbolic = (selected['id'] == result['baseline']['id']
                             or selected.get('check', {}).get('equivalent') is True) or any(
            attempt.get('check', {}).get('equivalent') is True for attempt in checked)
        deep = native_equal and complete_symbolic and len(retired) >= 2 and selected['cost'] < result['baseline']['cost']
        row = dict(name=fixture['name'], family=fixture['family'], source_sha256=fixture['sha256'],
                   discovery_status=result['status'], selected_candidate=selected['id'],
                   selected_mechanism=selected['mechanism'], complete_symbolic_boundary_proof=complete_symbolic,
                   private_measurement_bits=selected.get('private_bits', []),
                   independent_native_equal=native_equal, native_check=native_scope,
                   before=result['baseline']['resources'], after=selected['resources'],
                   before_cost=result['baseline']['cost'], after_cost=selected['cost'],
                   physical_clean_wires_retired=retired, deep_bank_retirement=deep,
                   discovery_wall_seconds=discovery_wall, engine_search_seconds=result['seconds'],
                   native_check_seconds=time.monotonic() - check_started,
                   attempt_count=len(result['attempts']),
                   mechanisms=[summary_attempt(attempt) for attempt in result['attempts']],
                   receipt=str(Path(fixture['name']) / 'restructure.json'))
        rows.append(row)
        print(json.dumps({key: row[key] for key in ('name', 'deep_bank_retirement', 'before_cost',
                                                   'after_cost', 'discovery_wall_seconds', 'independent_native_equal')}), flush=True)
    after_code = package_snapshot(package)
    _, after_fixtures = load_fixtures(args.suite)
    if before_code != after_code or fixture_binding != after_fixtures:
        raise RuntimeError('source or fixtures changed during benchmark; no source-bound summary published')
    summary = dict(manifest=manifest, source_binding_verified_unchanged=True, results=rows,
                   deep_bank_retirements=sum(row['deep_bank_retirement'] for row in rows),
                   bounded_or_no_deep_gain=sum(not row['deep_bank_retirement'] for row in rows),
                   independent_native_checks_passed=sum(row['independent_native_equal'] for row in rows),
                   discovery_wall_seconds=sum(row['discovery_wall_seconds'] for row in rows),
                   engine_search_seconds=sum(row['engine_search_seconds'] for row in rows),
                   native_check_seconds=sum(row['native_check_seconds'] for row in rows),
                   total_wall_seconds=time.monotonic() - started,
                   process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
                       if sys.platform.startswith('linux') else None,
                   memory_scope='Linux process high-water RSS, including imports, discovery and native checks; not a per-constructor peak')
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if not all(row['independent_native_equal'] and row['complete_symbolic_boundary_proof'] for row in rows):
        raise RuntimeError('one or more independently checked boundaries failed')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', help='Directory containing the selected strata package')
    parser.add_argument('--out', required=True, help='New or empty result directory')
    parser.add_argument('--label', required=True, help='Descriptive build label; source hashes are authoritative')
    parser.add_argument('--suite', choices=tuple(BUDGETS), default='frozen')
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({key: summary[key] for key in ('deep_bank_retirements', 'bounded_or_no_deep_gain',
                      'independent_native_checks_passed', 'discovery_wall_seconds', 'native_check_seconds',
                      'total_wall_seconds')}), flush=True)


if __name__ == '__main__':
    main()
