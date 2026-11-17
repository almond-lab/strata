#!/usr/bin/env python3
"""Run frozen no-hint holdouts in fresh processes, then exhaust native trajectories."""
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
LEGACY = {'observable_phase_elimination', 'direct_consumer_synthesis',
          'bilinear_tensor_decomposition', 'guided_bilinear_candidate_search',
          'measured_lookup_resynthesis'}
NEW_FAMILIES = {'consumer_functional_decomposition', 'output_basis_search', 'factored_observable_search'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(package):
    files = {str(p.relative_to(package)): sha(p.read_bytes()) for p in sorted(package.rglob('*.py'))}
    return dict(sha256=sha(json.dumps(files, sort_keys=True).encode()), files=files)


def frozen():
    contract = json.loads((HERE / 'freeze.json').read_text())
    raw = (HERE / 'fixtures.json').read_bytes()
    assert sha(raw) == contract['fixtures_sha256'], 'Frozen fixture bytes changed'
    assert sha((HERE / 'generate.py').read_bytes()) == contract['generator_sha256']
    rows = json.loads(raw)
    assert {row['name']: sha(row['text'].encode()) for row in rows} == contract['source_hashes']
    return contract, rows


def trajectory(circuit, quantum_width, classical_width, public, assignments, outcomes_at):
    """All ordinary gates run in kickmix. HMR/R outcomes are exhaustively forced.

    After native HMR, replace the drawn result with the enumerated result and
    adjust only its known local sign old_q*(drawn XOR forced). Native execution
    still supplies the reset, condition scope, all other gates and final values.
    This local primitive adapter is independent of Strata's symbolic checker.
    """
    import kickmix as km
    import numpy as np
    sim = km.Simulator(batch_size=len(assignments), seed=3981)
    wires = [km.q(q) for q in range(quantum_width)]
    bits = [km.b(b) for b in range(classical_width)]
    qvalues = np.zeros((quantum_width, len(assignments)), dtype=np.bool_)
    for level, q in enumerate(public):
        qvalues[q] = (assignments >> level) & 1
    if wires: sim.write_across_shots(wires, qvalues)
    if bits:
        bvalues = np.array([((assignments >> (len(public) + b)) & 1).astype(bool)
                            for b in range(classical_width)])
        sim.write_across_shots(bits, bvalues)
    lines = str(circuit).splitlines()
    if not any(line.split()[0] in {'HMR', 'R'} for line in lines):
        sim.do(circuit)
    else:
        stack, event = [], 0
        for line in lines:
            words = shlex.split(line)
            op = words[0]
            condition = sim.read_across_shots(km.b(int(words[-1][1:]))).copy() if 'if' in words else np.ones(len(assignments), dtype=bool)
            if op == 'PUSH_CONDITION': stack.append(condition)
            elif op == 'POP_CONDITION' and stack: stack.pop()
            if op not in {'HMR', 'R'}:
                sim.do(km.Circuit(line))
                continue
            active = condition.copy()
            for scope in stack: active &= scope
            old_q = sim.read_across_shots(km.q(int(words[1][1:]))).copy()
            before_phase = sim.read_phase_flipped_across_shots().copy()
            sim.do(km.Circuit(line))
            forced = ((assignments >> (outcomes_at + event)) & 1).astype(bool)
            event += 1
            if op == 'HMR':
                bit = km.b(int(words[2][1:]))
                drawn = sim.read_across_shots(bit).copy()
                phase_flip = active & old_q & (drawn ^ forced)
                drawn[active] = forced[active]
                sim.write_across_shots(bit, drawn)
            else:
                drawn_phase = sim.read_phase_flipped_across_shots() ^ before_phase
                phase_flip = drawn_phase ^ (active & old_q & forced)
            for shot in np.flatnonzero(phase_flip):
                sim.write_shot_phase(int(shot), Fraction(sim.read_shot_phase(int(shot))) + 1)
    phase = sim.read_phase_flipped_across_shots().copy()
    # Sources and accepted constructors in this suite use exact sign phases.
    for line in lines:
        words = line.split()
        if words[0] == 'Z_POW' and Fraction(words[2]).denominator != 1:
            raise RuntimeError('Native oracle requires exact sign phases for this frozen suite')
    return sim.read_across_shots(wires), sim.read_across_shots(bits) if bits else None, phase


def native_equal(before, after, clean, private, cap):
    import numpy as np
    public = [q for q in range(before.num_qubits) if q not in clean]
    classical_width = max(before.num_bits, after.num_bits)
    counts = [sum(line.split()[0] in {'HMR', 'R'} for line in str(c).splitlines()) for c in [before, after]]
    outcomes = max(counts)
    dimensions = len(public) + classical_width + outcomes
    cases = 1 << dimensions
    base = dict(scope='all physical public quantum inputs, all initial classical bits, every HMR/R outcome',
                quantum_input_bits=len(public), classical_input_bits=classical_width,
                formal_outcome_bits=outcomes, total_cases=cases,
                primitive_adapter='native HMR/R plus exact local outcome/sign forcing',
                introduced_bits_initial_values='also exhaustively arbitrary')
    if cases > cap:
        return dict(base, equivalent=None, status='bounded', reason='native_case_cap', cap=cap)
    if after.num_qubits > before.num_qubits:
        return dict(base, equivalent=False, status='different', reason='expanded_quantum_layout')
    visible_bits = [b for b in range(classical_width) if b not in private]
    for begin in range(0, cases, 4096):
        assignments = np.arange(begin, min(begin + 4096, cases), dtype=np.uint64)
        original = trajectory(before, before.num_qubits, classical_width, public, assignments,
                              len(public) + classical_width)
        candidate = trajectory(after, before.num_qubits, classical_width, public, assignments,
                               len(public) + classical_width)
        difference = np.any(original[0] != candidate[0], axis=0) | (original[2] != candidate[2])
        if visible_bits: difference |= np.any(original[1][visible_bits] != candidate[1][visible_bits], axis=0)
        if difference.any():
            at = int(np.flatnonzero(difference)[0])
            return dict(base, equivalent=False, status='different', counterexample_assignment=int(assignments[at]))
    return dict(base, equivalent=True, status='exhaustive_equal')


def oracle_selfcheck(cap):
    import kickmix as km
    original = km.Circuit('CCX q0 q1 q3\nCCX q3 q2 q4\nCCX q0 q1 q3')
    candidate = km.Circuit('CCX q0 q1 q3\nCCX q3 q2 q4\nHMR q3 b0\nCZ q0 q1 if b0')
    checks = dict(
        repaired=native_equal(original, candidate, [3], [0], cap)['equivalent'],
        missing_repair=native_equal(original, km.Circuit(str(candidate).rsplit('\n', 1)[0]), [3], [0], cap)['equivalent'],
        wrong_classical=native_equal(km.Circuit('BIT_STORE1 b0\nX q0'), km.Circuit('X q0'), [], [], cap)['equivalent'])
    assert checks == dict(repaired=True, missing_repair=False, wrong_classical=False), checks
    return checks


def worker(args):
    if args.package_root: sys.path.insert(0, str(Path(args.package_root).resolve()))
    import kickmix as km
    from strata.restructure import default_registry, restructure_file
    contract, fixtures = frozen()
    fixture = next(row for row in fixtures if row['name'] == args.worker)
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root / 'source.kmx'
    source.write_text(fixture['text'])
    options = dict(contract['budgets'])
    supports_deep = 'deep_only' in inspect.signature(restructure_file).parameters
    if args.mode == 'legacy':
        registry = default_registry()
        registry.constructors = {name: item for name, item in registry.constructors.items() if name in LEGACY}
        options['registry'] = registry
    elif args.mode == 'ablation':
        registry = default_registry()
        registry.constructors = {name: item for name, item in registry.constructors.items()
                                 if name not in NEW_FAMILIES}
        options['registry'] = registry
        if supports_deep: options['deep_only'] = True
    elif supports_deep:
        options['deep_only'] = True
    started = time.monotonic()
    result = restructure_file(source, root / 'search', clean_inputs=fixture['clean_inputs'],
                              cost=contract['objective'], **options)
    discovery_seconds = time.monotonic() - started
    selected = result['best']
    before, after = km.Circuit(fixture['text']), km.Circuit.from_file(selected['path'])
    private = selected.get('private_bits', [])
    assert all(b >= before.num_bits for b in private), 'Source classical output was silently made private'
    check = native_equal(before, after, fixture['clean_inputs'], private, contract['native_case_cap'])
    stats_before, stats_after = before.operation_statistics(), after.operation_statistics()
    costs = (before.num_qubits * stats_before.num_toffoli_like,
             after.num_qubits * stats_after.num_toffoli_like)
    assert (result['baseline']['cost'], selected['cost']) == costs, 'Parent cost differs from independent native census'
    retired = [q for q in fixture['clean_inputs'] if q >= after.num_qubits]
    phase_wire = next(q for q in range(before.num_qubits)
                      if q not in fixture['clean_inputs'] and q not in fixture['output_wires'])
    mutations = dict(
        changed_value=native_equal(before, km.Circuit(str(after) + f'\nX q{fixture["output_wires"][0]}'),
                                   fixture['clean_inputs'], private, contract['native_case_cap'])['equivalent'],
        changed_phase=native_equal(before, km.Circuit(str(after) + f'\nZ q{phase_wire}'),
                                   fixture['clean_inputs'], private, contract['native_case_cap'])['equivalent'])
    assert all(value is False for value in mutations.values()), mutations
    # Reproduce the earlier retention criterion separately without changing
    # discovery. This distinguishes new constructor gains from a broadened
    # criterion that now also admits paid representation/observable changes.
    bank_ids = {a['candidate'] for a in result['attempts'] if a.get('candidate')
                and a.get('check', {}).get('equivalent') is True
                and a.get('construction', {}).get('structural_witness', {}).get('removed_materialization_bank')}
    for state in result['states']:
        if state.get('parent') in bank_ids: bank_ids.add(state['id'])
    narrow = min([result['baseline']] + [s for s in result['states'] if s['id'] in bank_ids],
                 key=lambda s: s['cost'])
    narrow_check = (check if narrow['id'] == selected['id'] else
                    native_equal(before, km.Circuit.from_file(narrow['path']), fixture['clean_inputs'],
                                 narrow.get('private_bits', []), contract['native_case_cap']))
    row = dict(name=fixture['name'], family=fixture['family'], role=fixture['role'],
               source_sha256=fixture['sha256'], candidate_sha256=sha(str(after).encode()),
               search_mode=('native_structural_minus_new_three' if args.mode == 'ablation' else
                            'native_structural' if supports_deep and args.mode == 'structural' else 'legacy_catalog'),
               status=result['status'], before_cost=costs[0], after_cost=costs[1],
               before=dict(Q=before.num_qubits, toffoli=stats_before.num_toffoli_like, instructions=len(before)),
               after=dict(Q=after.num_qubits, toffoli=stats_after.num_toffoli_like, instructions=len(after)),
               retired_clean_wires=retired, private_bits=private, independent_native_check=check,
               oracle_mutations=mutations, complete_parent_cost_checked=True,
               bank_retirement_gain=check['equivalent'] is True and costs[1] < costs[0] and len(retired) >= 2,
               engine_structural_qualification=next((a.get('structural_qualification')
                    for a in result['attempts'] if a.get('candidate') == selected['id']), None),
               old_narrow_criterion=dict(cost=narrow['cost'], candidate=narrow['id'],
                    interpretation='baseline or verified descendant of removed_materialization_bank witness only',
                    native_equivalent=narrow_check['equivalent']),
               discovery_seconds=discovery_seconds, total_seconds=time.monotonic() - started,
               process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               selected_mechanism=selected.get('mechanism'),
               attempts=[dict(mechanism=a['mechanism'], status=a['status'], reason=a.get('reason'),
                              cost_delta=a.get('cost_delta'), check=a.get('check', {}).get('equivalent'))
                         for a in result['attempts']])
    (root / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps({k: row[k] for k in ['name', 'before_cost', 'after_cost', 'bank_retirement_gain']}), flush=True)


def run(args):
    if args.package_root: sys.path.insert(0, str(Path(args.package_root).resolve()))
    package = Path(importlib.util.find_spec('strata').origin).resolve().parent
    code = snapshot(package)
    contract, fixtures = frozen()
    root = Path(args.out).resolve()
    if root.exists() and any(root.iterdir()): raise RuntimeError('Output must be new/empty; archive reuse is forbidden')
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(label=args.label, mode=args.mode, package=str(package), code=code,
                    fixture_binding=contract['fixtures_sha256'], runner_sha256=sha(Path(__file__).read_bytes()),
                    contract=contract, oracle_selfcheck=oracle_selfcheck(contract['native_case_cap']),
                    discovery_inputs='source bytes, zero-input contract, frozen budget and cost only',
                    no_supplied_region_or_replacement=True, no_custom_constructor_implementation=True,
                    ablation_semantics=('Disable only consumer_functional_decomposition, output_basis_search, and factored_observable_search; keep current deep-only eligibility and discovery budgets. Also report the earlier narrow bank-only criterion post hoc.'
                                        if args.mode == 'ablation' else None),
                    equal_wall_time_or_cpu_claim=False)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows = []
    for fixture in fixtures:
        destination = root / fixture['name']
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', fixture['name'],
                   '--out', str(destination), '--mode', args.mode, '--label', args.label]
        if args.package_root: command += ['--package-root', str(Path(args.package_root).resolve())]
        started = time.monotonic()
        try:
            completed = subprocess.run(command, text=True, capture_output=True,
                                       timeout=contract['fixture_timeout_seconds'])
            if completed.returncode:
                row = dict(name=fixture['name'], family=fixture['family'], role=fixture['role'],
                           status='error', stderr=completed.stderr, seconds=time.monotonic() - started)
            else:
                row = json.loads((destination / 'result.json').read_text())
        except subprocess.TimeoutExpired:
            row = dict(name=fixture['name'], family=fixture['family'], role=fixture['role'],
                       status='timeout', seconds=time.monotonic() - started)
        rows.append(row)
        print(json.dumps({key: row.get(key) for key in ['name', 'status', 'before_cost', 'after_cost', 'bank_retirement_gain']}), flush=True)
    assert snapshot(package) == code, 'Package changed during run; summary cannot bind to mutable code'
    frozen()
    summary = dict(manifest=manifest, results=rows, all_failures_retained=True,
                   independent_exhaustive_passes=sum(r.get('independent_native_check', {}).get('equivalent') is True for r in rows),
                   bank_retirement_gains=sum(r.get('bank_retirement_gain', False) for r in rows),
                   errors_or_timeouts=sum(r['status'] in {'error', 'timeout'} for r in rows),
                   total_discovery_seconds=sum(r.get('discovery_seconds', 0) for r in rows),
                   max_worker_peak_rss_bytes=max((r.get('process_peak_rss_bytes', 0) for r in rows), default=0))
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in ['independent_exhaustive_passes', 'bank_retirement_gains', 'errors_or_timeouts']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--package-root')
    parser.add_argument('--label', required=True)
    parser.add_argument('--mode', choices=['structural', 'legacy', 'ablation'], default='structural')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    worker(args) if args.worker else run(args)
