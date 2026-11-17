from copy import deepcopy

import kickmix as km
import pytest

from strata.boundary_observations import observe_boundary, load_boundary_observations
from strata.floors import circuit_floors
from strata.inference import is_function_of, information_slack
from strata.model import StrataError, digest


def test_full_initial_cube_capture_proves_interior_support_without_external_simulator(tmp_path):
    circuit = km.Circuit('CCX q0 q1 q2\nX q0\nX q0')
    path = tmp_path / 'support.json'
    capture = observe_boundary(circuit, 1, clean_inputs=[2], output=path)
    assert capture['coverage'] == 'exhaustive_initial_cube' and capture['distinct_rows'] == 4
    relation = is_function_of(circuit, 2, 3, [0, 1], from_at=1, observations=path)
    assert relation['status'] == 'yes' and relation['form']['monomials'] == [[0, 1]]
    assert relation['reachability']['replay']['status'] == 'verified'
    assert 'reachable' in relation['scope']
    assert is_function_of(circuit, 2, 3, [0, 1], from_at=1)['status'] == 'no'
    slack = information_slack(circuit, 3, from_at=1, wires=[0, 1, 2], observations=path)
    assert slack['information_bits'] == 2 and slack['slack_bits'] == 1
    assert slack['reachability']['clean_initial_inputs'] == [2]


def test_samples_never_become_proved_dependency_slack_or_host_floor():
    circuit = km.Circuit('CCX q0 q1 q2\nX q0\nX q0')
    capture = observe_boundary(circuit, 1, clean_inputs=[2], max_inputs=2)
    assert capture['coverage'] == 'sampled_reachable_support'
    relation = is_function_of(circuit, 2, 3, [0, 1], from_at=1, observations=capture)
    assert relation['status'] == 'observed_yes' and not relation['host_guarantee_established']
    slack = information_slack(circuit, 3, from_at=1, wires=[0, 1, 2], observations=capture)
    assert slack['status'] == 'observation' and slack['slack_bits'] is None
    assert slack['observed_slack_bits'] >= 1
    floor = circuit_floors(circuit, from_at=1, observations=capture, outputs=[2], exact=True)
    assert floor['status'] == 'observation' and floor['value'] is None
    assert not floor['floor_established'] and not floor['minimum_established']
    assert floor['observed_care_set_value'] == 0


def test_import_replays_values_and_recomputes_coverage_instead_of_trusting_flags():
    circuit = km.Circuit('CX q0 q1\nX q0\nX q0')
    capture = observe_boundary(circuit, 1, clean_inputs=[1], max_inputs=1)
    forged = deepcopy(capture)
    forged.pop('observation_hash')
    forged['coverage'] = 'exhaustive_initial_cube'
    forged['exhaustive'] = True
    loaded = load_boundary_observations(forged, circuit, from_at=1)
    assert loaded['coverage'] == 'sampled_reachable_support'
    forged['rows'][0]['q1'] ^= 1
    with pytest.raises(StrataError, match='replay'):
        load_boundary_observations(forged, circuit, from_at=1)
    with pytest.raises(StrataError, match='source hash'):
        load_boundary_observations(capture, km.Circuit('X q0\nX q0\nX q1'), from_at=1)
    with pytest.raises(StrataError, match='starting boundary'):
        load_boundary_observations(capture, circuit, from_at=2)


def test_bounded_capture_streams_iterator_and_deduplicates_without_claiming_exhaustion():
    circuit = km.Circuit('HMR q0 b0\nZ q1 if b0')
    seen = []
    def assignments():
        for index in range(1000):
            seen.append(index)
            yield [index & 1, 0]
    capture = observe_boundary(circuit, 1, assignments=assignments(), max_inputs=3, max_rows=1)
    assert len(seen) == 4 and capture['truncated_inputs'] and capture['truncated_rows']
    assert capture['distinct_rows'] == 1 and capture['coverage'] == 'sampled_reachable_support'
    loaded = load_boundary_observations(capture, circuit, from_at=1)
    assert loaded['distinct_rows'] == 1 and loaded['coverage'] == 'sampled_reachable_support'


def test_many_outcomes_reduce_only_with_proved_value_independence():
    source = '\n'.join(f'HMR q{i} b{i}' for i in range(12))
    source += '\nPUSH_CONDITION if b11\nZ q0\nPOP_CONDITION'
    circuit = km.Circuit(source)
    capture = observe_boundary(circuit, 12, clean_inputs=range(12), max_outcomes=1)
    assert capture['status'] == 'captured' and capture['represented_outcomes'] == 4096
    assert capture['evaluated_outcomes_per_input'] == 1 and len(capture['omitted_classical_outputs']) == 12
    query = information_slack(circuit, from_at=12, observations=capture)
    assert query['status'] == 'exact' and query['information_bits'] == 0
    dependent = km.Circuit('HMR q0 b0\nX q1 if b0')
    assert observe_boundary(dependent, 2, clean_inputs=[0, 1], max_outcomes=1)['status'] == 'unknown'


def test_explicit_initial_rows_are_complete_and_never_silently_clean():
    circuit = km.Circuit('CX q0 q2')
    with pytest.raises(StrataError, match='incomplete'):
        observe_boundary(circuit, 1, assignments=[dict(q0=0, q2=0)])
    with pytest.raises(StrataError, match='contradicts'):
        observe_boundary(circuit, 1, assignments=[[0, 0, 1]], clean_inputs=[2])
    capture = observe_boundary(circuit, 1, assignments=[[0, 0, 0]])
    with pytest.raises(StrataError, match='either'):
        information_slack(circuit, from_at=1, observations=capture, domain=dict(assignments=[[0, 0, 0]]))
