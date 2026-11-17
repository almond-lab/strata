#!/usr/bin/env python3
"""Count retained executable breadth and independently verified cheaper designs.

This reads emitted alternatives, not an execution-attempt log. Missing native
bytes remain unknown telemetry and never imply an engine tried no candidates.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import shlex
import time

import kickmix as km

import run as oracle

NATIVE_KEYS = {'candidate', 'native_text', 'candidate_text', 'native_circuit'}
METADATA = {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG'}
INVOLUTIONS = {'X', 'CX', 'CCX', 'SWAP', 'CSWAP', 'Z', 'CZ', 'CCZ', 'NEG'}
FRONT_METRICS = ('Q', 'toffoli', 't', 'rotations', 'instructions', 'reaction_depth')


def native_resources(candidate):
    stats = candidate.operation_statistics()
    return dict(Q=candidate.num_qubits, toffoli=stats.num_toffoli_like, t=stats.num_t_like,
                rotations=stats['Z_POW'], instructions=len(candidate), reaction_depth=candidate.reaction_depth())


def pareto_front(discoveries):
    vectors = defaultdict(set)
    for discovery in discoveries:
        for resources in discovery['resource_variants']:
            vectors[tuple(resources[key] for key in FRONT_METRICS)].add(discovery['design'])
    front = [vector for vector in vectors if not any(
        other != vector and all(a <= b for a, b in zip(other, vector)) for other in vectors)]
    by_id = {row['design']: row for row in discoveries}
    return [dict(resources=dict(zip(FRONT_METRICS, vector)), designs=sorted(vectors[vector]),
                 distinct_design_count=len(vectors[vector]),
                 paid_function_sets=sorted({by_id[design]['paid_functions']['set_hash'] for design in vectors[vector]
                                            if by_id[design]['paid_functions']['status'] == 'exact'}))
            for vector in sorted(front)]


def named_registers(circuit):
    return {name: list(members) for name, members in circuit.register_data.items() if isinstance(name, str)}


def common_inputs(source, clean):
    quantum, classical = [], []
    for line in str(source).splitlines():
        for word in shlex.split(line):
            if re.fullmatch(r'q\d+', word):
                q = int(word[1:])
                if q not in clean and q not in quantum: quantum.append(q)
            elif re.fullmatch(r'b\d+', word):
                b = int(word[1:])
                if b not in classical: classical.append(b)
    quantum += [q for q in range(source.num_qubits) if q not in clean and q not in quantum]
    classical += [b for b in range(source.num_bits) if b not in classical]
    return {q: f'input{at}' for at, q in enumerate(quantum)}, {b: f'bit{at}' for at, b in enumerate(classical)}


def canonical_design(candidate, source, clean, private):
    qmap, bmap = common_inputs(source, clean)
    helpers, fresh = {}, {}
    lines, reduced = [], []
    for line in str(candidate).splitlines():
        words = shlex.split(line)
        if words[0] in METADATA: continue
        normalized = []
        for word in words:
            if re.fullmatch(r'q\d+', word):
                q = int(word[1:])
                if q in clean:
                    normalized.append(helpers.setdefault(q, f'clean{len(helpers)}'))
                else:
                    normalized.append(qmap.get(q, f'undeclared_q{q}'))
            elif re.fullmatch(r'b\d+', word):
                b = int(word[1:])
                if b in private:
                    normalized.append(fresh.setdefault(b, f'fresh{len(fresh)}'))
                else:
                    normalized.append(bmap.get(b, f'extra_public_b{b}'))
            else:
                normalized.append(word)
        if normalized[0] in {'CCX', 'CZ', 'SWAP'}:
            normalized[1:3] = sorted(normalized[1:3])
        elif normalized[0] == 'CCZ':
            normalized[1:4] = sorted(normalized[1:4])
        elif normalized[0] == 'CSWAP':
            normalized[2:4] = sorted(normalized[2:4])
        value = ' '.join(normalized)
        lines.append(value)
        if reduced and normalized[0] in INVOLUTIONS and reduced[-1] == value:
            reduced.pop()
        else:
            reduced.append(value)
    interface = dict(public_quantum_inputs=len(qmap), public_classical_inputs=len(bmap))
    return (oracle.sha(json.dumps(dict(interface, executable=lines), sort_keys=True).encode()),
            oracle.sha(json.dumps(dict(interface, executable=reduced), sort_keys=True).encode()))


def paid_functions(candidate, source, clean):
    """Independent pure-circuit fingerprint of paid nonlinear functions.

    Native execution supplies every gate operand over the complete initial
    cube. A Boolean Möbius transform removes free affine terms. Repeated paid
    functions, cleanup, control order, and schedule order do not multiply sets.
    This is a diversity diagnostic, not a complete algorithm-equivalence proof.
    """
    import numpy as np
    qmap, _ = common_inputs(source, clean)
    n = len(qmap)
    lines = str(candidate).splitlines()
    if n > 12 or candidate.num_bits or any(line.split()[0] in {'HMR', 'R'} for line in lines):
        return dict(status='unknown', reason='paid_function_scope_requires_at_most_12_quantum_inputs_and_no_classical_or_reset_channel')
    count = 1 << n
    sim = km.Simulator(batch_size=count, seed=991)
    assignments = np.arange(count, dtype=np.uint64)
    for level, q in enumerate(qmap): sim.write_across_shots(km.q(q), ((assignments >> level) & 1).astype(bool))
    signatures, paid_count = set(), 0
    for line in lines:
        words = shlex.split(line)
        op = words[0]
        qs = [int(word[1:]) for word in words[1:] if re.fullmatch(r'q\d+', word)]
        values = None
        if op in {'CCX', 'CCZ'}:
            controls = qs[:2] if op == 'CCX' else qs
            values = np.ones(count, dtype=bool)
            for q in controls: values &= sim.read_across_shots(km.q(q))
        elif op == 'CSWAP':
            values = sim.read_across_shots(km.q(qs[0])) & (sim.read_across_shots(km.q(qs[1])) ^ sim.read_across_shots(km.q(qs[2])))
        if values is not None:
            paid_count += 1
            coefficients = values.astype(np.uint8)
            for level in range(n):
                step = 1 << level
                blocks = coefficients.reshape(-1, step * 2)
                blocks[:, step:] ^= blocks[:, :step]
            coefficients[0] = 0
            for level in range(n): coefficients[1 << level] = 0
            if coefficients.any(): signatures.add(oracle.sha(np.packbits(coefficients, bitorder='little').tobytes()))
        sim.do(km.Circuit(line))
    functions = sorted(signatures)
    return dict(status='exact', set_hash=oracle.sha(json.dumps(dict(inputs=n, functions=functions), sort_keys=True).encode()),
                nonlinear_function_count=len(functions), native_paid_gate_count=paid_count,
                exhaustive_inputs=count, functions=functions)


def dedup_selfcheck():
    """Control examples validate the counter only; none enter discovery totals."""
    source = km.Circuit('CCX q0 q1 q4\nCX q4 q2\nCCX q0 q1 q4\nX q5\nX q5')
    clean = {4, 5}
    direct = [km.Circuit('CCX q0 q1 q2\nAPPEND_TO_REGISTER q3 r0'),
              km.Circuit('CCX q1 q0 q2\nAPPEND_TO_REGISTER q3 r2\nREGISTER r2 "display"')]
    helper = [km.Circuit('CCX q0 q1 q4\nCX q4 q2\nCCX q0 q1 q4'),
              km.Circuit('CCX q1 q0 q5\nCX q5 q2\nCCX q1 q0 q5')]
    measured = [(km.Circuit('CCX q0 q1 q4\nCX q4 q2\nHMR q4 b0\nCZ q0 q1 if b0'), {0}),
                (km.Circuit('CCX q1 q0 q5\nCX q5 q2\nHMR q5 b7\nCZ q1 q0 if b7'), {7})]
    for candidate in direct + helper:
        assert oracle.native_equal(source, candidate, clean, set(), 65536)['equivalent'] is True
    for candidate, private in measured:
        assert oracle.native_equal(source, candidate, clean, private, 65536)['equivalent'] is True
    checks = dict(metadata_and_control_aliases=canonical_design(direct[0], source, clean, set())[0] == canonical_design(direct[1], source, clean, set())[0],
                  clean_helper_aliases=canonical_design(helper[0], source, clean, set())[0] == canonical_design(helper[1], source, clean, set())[0],
                  proved_fresh_outcome_aliases=canonical_design(measured[0][0], source, clean, measured[0][1])[0] == canonical_design(measured[1][0], source, clean, measured[1][1])[0])
    checks['compute_cleanup_paid_function_multiplicity'] = paid_functions(direct[0], source, clean)['set_hash'] == paid_functions(helper[0], source, clean)['set_hash']
    assert all(checks.values()), checks
    return checks


def retained_candidates(receipt, source):
    """Native states plus every explicit nested emitted alternative, with provenance."""
    inventory, hash_references = [], 0
    states = {state['id']: state for state in receipt.get('states', [])}
    for state in states.values():
        path = Path(state['path'])
        if path.exists():
            inventory.append(dict(text=path.read_text(), private=state.get('private_bits', []),
                                  provenance='states.' + state['id'], reported_outer_verified=state.get('parent') is not None))

    def walk(value, path, inherited_private, outer=False):
        nonlocal hash_references
        if isinstance(value, dict):
            private = set(inherited_private)
            for key in ['private_bits', 'introduced_private_bits', 'required_internal_bits']:
                bits = value.get(key)
                if isinstance(bits, (list, tuple)) and all(type(bit) is int for bit in bits): private.update(bits)
            for key, item in value.items():
                if key in NATIVE_KEYS and isinstance(item, str):
                    if re.fullmatch(r'[0-9a-f]{64}', item):
                        hash_references += 1
                    elif item == '' or '\n' in item or re.match(r'^[A-Z_]+(?:\s|$)', item):
                        inventory.append(dict(text=item, private=sorted(private), provenance=path + '.' + key,
                                              reported_outer_verified=outer and key == 'candidate'))
                # Sibling alternatives must not inherit another alternative's
                # selected fresh-bit list. Only actual ancestor state bits flow.
                if isinstance(item, (dict, list)):
                    walk(item, path + '.' + key, inherited_private)
        elif isinstance(value, list):
            for at, item in enumerate(value): walk(item, path + '.' + str(at), inherited_private)

    for at, attempt in enumerate(receipt.get('attempts', [])):
        parent = states.get(attempt.get('parent'), {})
        walk(attempt.get('construction', {}), f'attempts.{at}.construction', parent.get('private_bits', []),
             outer=attempt.get('check', {}).get('equivalent') is True)
    return inventory, hash_references


def fixture_yield(root, fixture, cap, max_variants, previous_best, max_operations):
    receipt_path = root / fixture['name'] / 'search' / 'restructure.json'
    if not receipt_path.exists():
        return dict(name=fixture['name'], family=fixture['family'], status='no_retained_receipt')
    receipt = json.loads(receipt_path.read_text())
    assert receipt['source_sha256'] == fixture['sha256']
    source = km.Circuit(fixture['text'])
    clean = set(fixture['clean_inputs'])
    cost_before = source.num_qubits * source.operation_statistics().num_toffoli_like
    occurrences, references = retained_candidates(receipt, source)
    variants, unparsable = {}, 0
    for item in occurrences:
        try:
            candidate = km.Circuit(item['text'])
        except (ValueError, RuntimeError):
            unparsable += 1
            continue
        text = str(candidate)
        key = oracle.sha(json.dumps(dict(text=text, private=sorted(set(item['private']))), sort_keys=True).encode())
        if key not in variants:
            variants[key] = dict(candidate=candidate, private=sorted(set(item['private'])), origins=[],
                                 reported_outer_verified=False)
        variants[key]['origins'].append(item['provenance'])
        variants[key]['reported_outer_verified'] |= item['reported_outer_verified']
    designs, outcomes = {}, Counter()
    before_registers = named_registers(source)
    checked = 0
    for key, row in variants.items():
        if checked >= max_variants:
            outcomes['variant_cap'] += 1
            continue
        checked += 1
        candidate, private = row['candidate'], set(row['private'])
        if len(candidate) > max_operations:
            outcomes['candidate_operation_cap'] += 1
            continue
        depth, balanced = 0, True
        for line in str(candidate).splitlines():
            op = line.split()[0]
            if op == 'PUSH_CONDITION': depth += 1
            elif op == 'POP_CONDITION': depth -= 1
            if depth < 0: balanced = False
        if depth or not balanced:
            outcomes['unbalanced_classical_scope'] += 1
            continue
        # Existing public bits can never be made private by candidate telemetry.
        if any(bit < source.num_bits and bit not in fixture.get('internal_bits', []) for bit in private):
            outcomes['unauthorized_private_contract'] += 1
            continue
        removed = set(range(candidate.num_qubits, source.num_qubits))
        if candidate.num_qubits > source.num_qubits or not removed <= clean:
            outcomes['invalid_physical_layout'] += 1
            continue
        registers = named_registers(candidate)
        if any(registers.get(name) != members for name, members in before_registers.items()):
            outcomes['changed_named_public_register'] += 1
            continue
        cost = candidate.num_qubits * candidate.operation_statistics().num_toffoli_like
        try:
            check = oracle.native_equal(source, candidate, clean, private, cap)
        except (ValueError, RuntimeError):
            outcomes['unsupported_native_oracle'] += 1
            continue
        if check['equivalent'] is not True:
            outcomes['unknown_native_budget' if check['equivalent'] is None else 'boundary_mismatch'] += 1
            continue
        outcomes['native_equivalent'] += 1
        design, reduced = canonical_design(candidate, source, clean, private)
        value = designs.setdefault(design, dict(design=design, adjacent_involution_reduced_design=reduced,
                        cost=cost, costs=[], raw_variants=0, reported_outer_verified=False, origins=[],
                        native_case_count=check['total_cases'], executable_sample=str(candidate),
                        sample_private_bits=sorted(private), resource_variants=[]))
        if cost < value['cost']:
            value.update(executable_sample=str(candidate), sample_private_bits=sorted(private),
                         native_case_count=check['total_cases'])
        value['cost'] = min(value['cost'], cost)
        value['costs'].append(cost)
        value['raw_variants'] += 1
        resources = native_resources(candidate)
        if resources not in value['resource_variants']: value['resource_variants'].append(resources)
        value['reported_outer_verified'] |= row['reported_outer_verified']
        value['origins'] += row['origins']
    discoveries = [row for row in designs.values() if row['cost'] < cost_before]
    for discovery in discoveries:
        sample = km.Circuit(discovery['executable_sample'])
        assert sample.num_qubits * sample.operation_statistics().num_toffoli_like == discovery['cost']
        assert canonical_design(sample, source, clean, set(discovery['sample_private_bits']))[0] == discovery['design']
        discovery['paid_functions'] = paid_functions(sample, source, clean)
    beyond = [row for row in discoveries if previous_best is not None and row['cost'] < previous_best]
    # Keep native samples for every independently verified cheaper design so the
    # count is auditable; a hash without emitted bytes is never a discovery.
    return dict(name=fixture['name'], family=fixture['family'], role=fixture['role'], status='checked',
                source_sha256=fixture['sha256'], source_cost=cost_before, previous_best_cost=previous_best,
                retained_native_occurrences=len(occurrences), retained_native_variants=len(variants),
                unparsable_native_occurrences=unparsable, hash_only_candidate_reference_occurrences=references,
                available_breadth_before_verification=len(variants), validation_outcomes=dict(outcomes),
                distinct_verified_executable_designs=len(designs),
                distinct_verified_cheaper_designs=len(discoveries),
                distinct_verified_cheaper_than_previous_best=len(beyond),
                distinct_verified_at_previous_best=sum(row['cost'] == previous_best for row in discoveries),
                distinct_paid_function_sets=len({row['paid_functions']['set_hash'] for row in discoveries if row['paid_functions']['status'] == 'exact'}),
                designs_with_unknown_paid_function_fingerprint=sum(row['paid_functions']['status'] != 'exact' for row in discoveries),
                adjacent_involution_reduced_cheaper_designs=len({r['adjacent_involution_reduced_design'] for r in discoveries}),
                previously_outer_verified_cheaper_designs=sum(r['reported_outer_verified'] for r in discoveries),
                objective_values=[dict(cost=cost, designs=count) for cost, count in sorted(Counter(r['cost'] for r in discoveries).items())],
                verified_resource_front=pareto_front(discoveries),
                discoveries=sorted(discoveries, key=lambda r: (r['cost'], r['design'])),
                telemetry_scope='retained states and explicit emitted alternatives; not a census of attempted or implicit candidates')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, metavar='LABEL=DIR')
    parser.add_argument('--previous', required=True, help='Label whose retained best scores define the prior-best comparison')
    parser.add_argument('--out', required=True)
    parser.add_argument('--fixtures', help='Frozen corpus directory; defaults to the original nine cases')
    parser.add_argument('--max-variants-per-fixture', type=int, default=2048)
    args = parser.parse_args()
    if args.max_variants_per_fixture < 1: raise ValueError('Variant cap must be positive')
    if args.fixtures: oracle.HERE = Path(args.fixtures).resolve()
    extractor_hash = oracle.sha(Path(__file__).read_bytes())
    oracle_hash = oracle.sha(Path(oracle.__file__).read_bytes())
    destination = Path(args.out)
    if destination.exists(): raise RuntimeError('Yield receipt already exists; use a new path')
    contract, fixtures = oracle.frozen()
    selfcheck = dedup_selfcheck()
    roots = {label: Path(path).resolve() for label, path in (item.split('=', 1) for item in args.run)}
    assert args.previous in roots
    summaries = {label: json.loads((root / 'summary.json').read_text()) for label, root in roots.items()}
    for summary in summaries.values():
        assert summary['manifest']['fixture_binding'] == contract['fixtures_sha256']
    prior = {row['name']: row.get('after_cost') for row in summaries[args.previous]['results']}
    start, results = time.monotonic(), {}
    for label, root in roots.items():
        rows = [fixture_yield(root, fixture, contract['native_case_cap'], args.max_variants_per_fixture,
                              prior[fixture['name']], contract['budgets']['max_operations']) for fixture in fixtures]
        families = {}
        for family in sorted({row['family'] for row in rows}):
            chosen = [row for row in rows if row['family'] == family]
            families[family] = dict(verified_cheaper_contract_design_pairs=sum(row.get('distinct_verified_cheaper_designs', 0) for row in chosen),
                    verified_cheaper_than_previous_best_pairs=sum(row.get('distinct_verified_cheaper_than_previous_best', 0) for row in chosen),
                    distinct_normalized_executable_patterns=len({d['design'] for row in chosen for d in row.get('discoveries', [])}))
        result = dict(source_code=summaries[label]['manifest']['code']['sha256'], results=rows, families=families,
                      retained_native_occurrences=sum(row.get('retained_native_occurrences', 0) for row in rows),
                      available_retained_native_variants=sum(row.get('retained_native_variants', 0) for row in rows),
                      verified_cheaper_contract_design_pairs=sum(row.get('distinct_verified_cheaper_designs', 0) for row in rows),
                      verified_cheaper_than_previous_best_pairs=sum(row.get('distinct_verified_cheaper_than_previous_best', 0) for row in rows),
                      verified_at_previous_best_pairs=sum(row.get('distinct_verified_at_previous_best', 0) for row in rows),
                      paid_function_set_contract_pairs=sum(row.get('distinct_paid_function_sets', 0) for row in rows),
                      distinct_normalized_executable_patterns=len({d['design'] for row in rows for d in row.get('discoveries', [])}))
        results[label] = result
        print(json.dumps(dict(label=label, **{key: result[key] for key in ['available_retained_native_variants', 'verified_cheaper_contract_design_pairs',
                                                                         'verified_cheaper_than_previous_best_pairs', 'distinct_normalized_executable_patterns']})), flush=True)
    assert oracle.sha(Path(__file__).read_bytes()) == extractor_hash
    assert oracle.sha(Path(oracle.__file__).read_bytes()) == oracle_hash
    output = dict(fixture_binding=contract['fixtures_sha256'], objective=contract['objective'], prior_best_label=args.previous,
                  extractor_sha256=extractor_hash, oracle_sha256=oracle_hash,
                  canonicalization_selfcheck=selfcheck,
                  caps=dict(native_cases=contract['native_case_cap'], variants_per_fixture=args.max_variants_per_fixture),
                  results=results, extraction_seconds=time.monotonic() - start,
                  counting_rules=['Count only retained emitted native bytes that pass independent complete-boundary and physical-layout checks and full native pricing.',
                                  'Normalize common public input IDs by source first occurrence; normalize declared clean helpers and independently verified fresh private bits by candidate first occurrence; remove display metadata.',
                                  'Normalize symmetric controls/targets for CCX, CZ, CCZ, SWAP and CSWAP before hashing; operand-order aliases are not discoveries.',
                                  'Use the cheapest verified physical-layout variant for each normalized executable design; also report adjacent involution cancellation reduction.',
                                  'Meaningful fronts use native Q, Toffoli, T, rotations, instructions and reaction depth. Pure-circuit paid-function sets additionally collapse repeated nonlinear functions modulo affine terms, order and cleanup; this diagnostic is not a complete algorithm-equivalence relation.',
                                  'Contract-design pairs and shared normalized executable patterns are separate totals. Different instruction schedules may remain different designs; distinct algorithms are not proved.',
                                  'Available retained breadth is not verified yield and is not an attempt census. Missing older telemetry never means an engine tried nothing.',
                                  'The 1-to-1000 aspiration is a target, not a result or an assumption. No implicit or unverified alternatives count.'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
