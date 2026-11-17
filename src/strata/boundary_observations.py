"""Bounded, replayable observations of values reachable at a native boundary.

An observation file is not a finite-domain promise. Import replays its source
inputs, and exhaustive coverage is derived from the full input cube, never from
a caller-supplied ``exhaustive`` flag. Sampled support remains observational.
"""
import hashlib
from itertools import islice
import json
from pathlib import Path
import shlex

from .model import StrataError, digest
from .domains import load_domain, window_lines, _classical_boundary, execute_assignment, _Unknown, WindowLimit


def _source_hash(circuit):
    from .streaming import native_lines
    hasher = hashlib.sha256(str(circuit.num_qubits).encode())
    with native_lines(circuit) as lines:
        for line in lines:
            hasher.update(line.rstrip('\r\n').encode())
            hasher.update(b'\n')
    return hasher.hexdigest()


def observe_boundary(circuit, at, *, assignments=None, clean_inputs=(), max_inputs=4096,
                     max_rows=65536, max_outcomes=1024, max_operations=10000,
                     max_variables=128, max_terms=65536, max_product_terms=1048576, output=None):
    """Capture a bounded support from a full input cube or a declared iterator.

    The iterator supplies complete physical/classical initial assignments. With
    no iterator, nonclean inputs are enumerated automatically in binary order.
    No unlisted input is silently initialized to zero.
    """
    from .inference import _wires
    for value, name in ((max_inputs, 'max_inputs'), (max_rows, 'max_rows'), (max_outcomes, 'max_outcomes')):
        if type(value) is not int or value < 1:
            raise StrataError(name + ' must be a positive integer')
    clean = set(_wires(circuit, clean_inputs, 'capture clean inputs'))
    try:
        lines, at = window_lines(circuit, 0, at, max_lines=max_operations)
    except WindowLimit as exc:
        return dict(status='unknown', reason=str(exc), capture_at=at)
    classical = _classical_boundary(lines)
    variables = [f'q{q}' for q in range(circuit.num_qubits) if q not in clean] + classical
    full_cube = assignments is None
    if full_cube:
        def cube():
            for encoded in range(min(1 << len(variables), max_inputs + 1)):
                row = {f'q{q}': 0 for q in range(circuit.num_qubits)}
                row.update({name: (encoded >> i) & 1 for i, name in enumerate(variables)})
                yield row
        assignments = cube()
    try:
        initial = list(islice(iter(assignments), max_inputs + 1))
    except TypeError as exc:
        raise StrataError('capture assignments must be an iterator of complete boundary rows') from exc
    truncated_inputs = len(initial) > max_inputs
    initial = initial[:max_inputs]
    if not initial:
        raise StrataError('capture needs at least one initial assignment')
    promised = load_domain(dict(assignments=initial), circuit, at=at, max_assignments=max_inputs,
                           max_operations=max_operations, clean_inputs=clean)
    initial = promised['assignments']
    from .inference import _value_semantics
    semantics = _value_semantics(lines, clean_inputs=clean, max_operations=max_operations,
                               max_variables=max_variables, max_terms=max_terms,
                               max_product_terms=max_product_terms)
    count = sum(bool(words) and words[0] == 'HMR' for words in
                (shlex.split(line, comments=True) for line in lines))
    outcomes = 1 << count
    omitted_classical = []
    reduction = 'full_outcome_enumeration'
    if outcomes > max_outcomes:
        if semantics['status'] != 'exact':
            return dict(status='unknown', reason='capture_measurement_outcome_budget', capture_at=at)
        formal = set(semantics['measurement_variables'])
        if any(formal & set(term) for poly in semantics['outputs'].values() for term in poly['monomials']):
            return dict(status='unknown', reason='capture_measurement_outcome_budget', capture_at=at)
        omitted_classical = sorted(bit for bit, poly in semantics['classical_outputs'].items()
                                   if any(formal & set(term) for term in poly['monomials']))
        outcomes = 1
        reduction = 'exact_ANF_quantum_output_independence; outcome-dependent classical outputs omitted'
    rows, checked, truncated_rows = {}, 0, False
    try:
        for assignment in initial:
            for encoded in range(outcomes):
                replay = execute_assignment(lines, assignment, circuit.num_qubits,
                    outcomes=[(encoded >> i) & 1 for i in range(count)], max_operations=max_operations)
                row = {**{f'q{q}': bit for q, bit in replay['outputs'].items()},
                       **{bit: value for bit, value in replay['classical_outputs'].items() if bit not in omitted_classical}}
                key = tuple(sorted(row.items()))
                checked += 1
                if key not in rows and len(rows) >= max_rows:
                    truncated_rows = True
                    break
                rows[key] = row
            if truncated_rows:
                break
    except (_Unknown, ValueError) as exc:
        return dict(status='unknown', reason=str(exc), capture_at=at, checked_trajectories=checked)
    # Every stored row is validated above and unique. Completeness of the cube
    # is decidable from this count even for an imported finite input list.
    complete_cube = len(initial) == 1 << len(variables) and not truncated_rows
    coverage = 'exhaustive_initial_cube' if complete_cube else 'sampled_reachable_support'
    report = dict(version=1, type='boundary_observations', status='captured',
                  source_hash=_source_hash(circuit), capture_at=at, num_qubits=circuit.num_qubits,
                  clean_inputs=sorted(clean), initial_variables=variables, initial_assignments=initial,
                  input_origin='automatic_cube' if full_cube else 'declared_assignment_iterator',
                  declared_input_coverage=('all_supplied_initial_assignments' if not truncated_inputs and not truncated_rows
                                           else 'bounded_prefix_of_supplied_initial_assignments'),
                  declared_input_scope='explicit supplied initial rows; no assumption that a declared subset covers host inputs',
                  rows=[rows[key] for key in sorted(rows)], omitted_classical_outputs=omitted_classical,
                  coverage=coverage, initial_assignment_count=len(initial), distinct_rows=len(rows),
                  checked_trajectories=checked, represented_outcomes=1 << count,
                  evaluated_outcomes_per_input=outcomes, outcome_reduction=reduction,
                  truncated_inputs=truncated_inputs, truncated_rows=truncated_rows,
                  scope='reachable basis values under explicit initial inputs; phases and outcome distributions are not reconstructed',
                  limits=dict(max_inputs=max_inputs, max_rows=max_rows, max_outcomes=max_outcomes,
                              max_operations=max_operations, max_variables=max_variables,
                              max_terms=max_terms, max_product_terms=max_product_terms))
    report['observation_hash'] = digest(report)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


def load_boundary_observations(spec, circuit, *, from_at=0, at=None, max_assignments=65536,
                               max_operations=10000, max_outcomes=1024, max_variables=128,
                               max_terms=65536, max_product_terms=1048576):
    """Bind and independently replay support; never trust a coverage assertion."""
    if isinstance(spec, (str, Path)):
        try:
            spec = json.loads(Path(spec).read_text())
        except (OSError, ValueError) as exc:
            raise StrataError('cannot read boundary observations: ' + str(exc)) from exc
    if not isinstance(spec, dict) or spec.get('type') != 'boundary_observations' or spec.get('version') != 1:
        raise StrataError('expected boundary observations version 1')
    if spec.get('source_hash') != _source_hash(circuit) or spec.get('num_qubits') != circuit.num_qubits:
        raise StrataError('boundary observations source hash or width mismatch')
    if spec.get('capture_at') != from_at:
        raise StrataError('boundary observations were captured at a different starting boundary')
    if 'observation_hash' in spec and spec['observation_hash'] != digest({k: v for k, v in spec.items() if k != 'observation_hash'}):
        raise StrataError('boundary observation hash mismatch')
    inputs = spec.get('initial_assignments')
    rows = spec.get('rows')
    if not isinstance(inputs, list) or len(inputs) > max_assignments or not isinstance(rows, list) or len(rows) > max_assignments:
        raise StrataError('boundary observation replay assignment budget')
    capture_rows = spec.get('limits', {}).get('max_rows', max_assignments)
    capture_outcomes = spec.get('limits', {}).get('max_outcomes', max_outcomes)
    if type(capture_rows) is not int or capture_rows < 1 or type(capture_outcomes) is not int or capture_outcomes < 1:
        raise StrataError('invalid boundary observation row limit')
    replay = observe_boundary(circuit, from_at, assignments=inputs, clean_inputs=spec.get('clean_inputs', []),
                              max_inputs=max_assignments, max_rows=min(capture_rows, max_assignments),
                              max_outcomes=min(max_outcomes, capture_outcomes), max_operations=max_operations,
                              max_variables=max_variables, max_terms=max_terms, max_product_terms=max_product_terms)
    if replay['status'] != 'captured':
        raise StrataError('boundary observation replay unavailable: ' + replay.get('reason', 'unknown'))
    if replay['rows'] != rows or replay['omitted_classical_outputs'] != spec.get('omitted_classical_outputs', []):
        raise StrataError('boundary observation rows disagree with independent source replay')
    lines, stop = window_lines(circuit, from_at, at, max_lines=max_operations)
    needed = set(_classical_boundary(lines))
    projected = [{key: value for key, value in row.items() if key.startswith('q') or key in needed} for row in rows]
    # This remains a data carrier for a finite value query. The wrapping query
    # below carries the replay provenance and downgrades sampled conclusions.
    return dict(domain=dict(version=1, type='finite', assignments=projected),
                coverage=replay['coverage'], capture_at=from_at, at=stop,
                source_hash=replay['source_hash'], observation_hash=digest(replay),
                original_observation_hash=spec.get('observation_hash'), distinct_rows=len(rows),
                replay=dict(status='verified', checked_trajectories=replay['checked_trajectories'],
                            proof='independent concrete prefix replay; cube coverage recomputed from complete initial assignments',
                            declared_input_scope='every stored initial assignment is replayed, unless the recorded row budget truncates support'),
                clean_initial_inputs=replay['clean_inputs'], omitted_classical_outputs=replay['omitted_classical_outputs'])


def query_observations(kind, circuit, observations, *, at=None, from_at=0, **options):
    """Run an existing value query on replayed support with explicit coverage."""
    if options.get('domain') is not None:
        raise StrataError('choose either a declared domain or boundary observations')
    options.pop('domain', None)
    replay_keys = ('max_assignments', 'max_operations', 'max_outcomes', 'max_variables', 'max_terms', 'max_product_terms')
    bound = load_boundary_observations(observations, circuit, from_at=from_at, at=at,
                                      **{key: options[key] for key in replay_keys if key in options})
    if kind == 'functions':
        from .inference import functions_at as query
    elif kind == 'function-of':
        from .inference import is_function_of as query
    elif kind == 'slack':
        from .inference import information_slack as query
    elif kind == 'floors':
        from .floors import circuit_floors as query
    else:
        raise StrataError('unknown observation query kind')
    report = query(circuit, at=at, from_at=from_at, domain=bound.pop('domain'), **options)
    report['reachability'] = bound
    report['scope'] = ('all reachable basis values under explicit initial clean promises; independently replayed'
                       if bound['coverage'] == 'exhaustive_initial_cube' else
                       'observed reachable support only; unobserved states may invalidate inferred relations or compression')
    report['all_input_equivalence'] = False
    if bound['coverage'] != 'exhaustive_initial_cube' and report.get('status') != 'unknown':
        report['observed_status'] = report['status']
        report['status'] = 'observed_' + report['status'] if kind == 'function-of' else 'observation'
        report['host_guarantee_established'] = False
        if kind == 'floors':
            report['observed_care_set_value'] = report.pop('value', None)
            report.update(value=None, floor_established=False, minimum_established=False)
        elif kind == 'slack':
            report['observed_information_bits'] = report.pop('information_bits', None)
            report['observed_slack_bits'] = report.pop('slack_bits', None)
            report.update(information_bits=None, slack_bits=None)
    return report
