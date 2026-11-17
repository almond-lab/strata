"""Boolean relaxations of whole measured episodes keep their evidence boundary."""
from copy import deepcopy
from itertools import product

import kickmix as km
import pytest

from strata.domains import load_domain, equivalent_on_domain
from strata.floors import circuit_floors, exact_minimum, exact_minimum_on_rows, verify_minimum_witness
from strata.inference import functions_at, is_function_of
from strata.model import StrataError


EPISODE = 'CCX q0 q1 q2\nCX q2 q3\nHMR q2 b0\nCZ q0 q1 if b0'


def test_full_measured_episode_has_visible_boolean_floor_and_retains_phase_proof():
    report = circuit_floors(km.Circuit(EPISODE), clean_inputs=[2], outputs=[3], exact=True)
    assert report['value'] == 1 and report['minimum_established']
    assert report['measurement_analysis']['phase_obligations_discharged']
    assert report['measurement_analysis']['outcome_independent_values']
    assert report['global_quantum_floor'] is False
    assert 'postselection' in report['model_restrictions']['excluded']
    assert 'not established' in report['output_scope']
    assert report['exact_search']['witness']


@pytest.mark.parametrize('source', [
    EPISODE.rsplit('\n', 1)[0],
    EPISODE.replace('CZ q0 q1 if b0', 'Z q0 if b0'),
    EPISODE + '\nX q3 if b0',
])
def test_missing_wrong_phase_repair_and_outcome_dependent_values_fail_closed(source):
    report = circuit_floors(km.Circuit(source), clean_inputs=[2], outputs=[3])
    assert report['status'] == 'unknown' and not report['floor_established']
    assert 'value' not in report


def test_wanted_outputs_are_a_relaxation_and_validate_wire_boundaries():
    circuit = km.Circuit('CCX q0 q1 q2\nCX q0 q3')
    assert circuit_floors(circuit, clean_inputs=[2, 3])['value'] == 1
    wanted = circuit_floors(circuit, clean_inputs=[2, 3], outputs=[3], exact=True)
    assert wanted['value'] == 0 and wanted['omitted_outputs'] == [0, 1, 2]
    for outputs in ([4], [-1], [0, 0], [True]):
        with pytest.raises(StrataError, match='outputs'):
            circuit_floors(circuit, outputs=outputs)


def test_phase_only_boundary_bits_are_formal_for_values_but_strict_for_equivalence():
    circuit = km.Circuit('CCX q0 q1 q2\nPUSH_CONDITION if b8\nZ q0\nPOP_CONDITION')
    domain = dict(version=1, type='finite', assignments=[[a, b, 0] for a, b in product(range(2), repeat=2)])
    report = circuit_floors(circuit, domain=domain, outputs=[2], exact=True)
    assert report['value'] == 1 and report['omitted_classical_inputs'] == ['b8']
    assert report['value_dependency_proof']['method'] == 'exact_ANF_quantum_output_independence'
    relation = is_function_of(circuit, 2, 4, [0, 1], domain=domain)
    assert relation['status'] == 'yes'
    normalized = load_domain(domain, circuit, value_only=True)
    assert load_domain(normalized, circuit, value_only=True) == normalized
    with pytest.raises(StrataError, match='complete'):
        equivalent_on_domain(circuit, circuit, domain)
    changed = km.Circuit(str(circuit).replace('Z q0', 'X q0'))
    with pytest.raises(StrataError, match='complete'):
        functions_at(changed, domain=domain)
    normalized['assignments'][0]['q0'] ^= 1
    with pytest.raises(StrataError, match='hash'):
        load_domain(normalized, circuit, value_only=True)


def test_omission_proof_respects_captured_conditions_and_clean_promises():
    circuit = km.Circuit('PUSH_CONDITION if b2\nBIT_STORE0 b2\nX q0\nPOP_CONDITION')
    with pytest.raises(StrataError, match='complete'):
        functions_at(circuit, domain=dict(assignments=[[0]]))
    clean_sensitive = km.Circuit('CX q0 q1 if b2')
    with pytest.raises(StrataError, match='contradicts'):
        load_domain(dict(assignments=[[1, 0]]), clean_sensitive, value_only=True, clean_inputs=[0])


def test_many_hmr_outcomes_reduce_by_proved_value_independence():
    circuit = km.Circuit('\n'.join(f'HMR q{i} b{i}' for i in range(20)))
    result = functions_at(circuit, domain=dict(assignments=[[0] * 20]), max_outcomes=1)
    assert result['status'] == 'exact' and result['outcome_count'] == 1 << 20
    assert result['evaluated_outcomes'] == 1
    assert result['outcome_reduction'] == 'exact_ANF_quantum_output_independence'
    measured = circuit_floors(circuit, domain=dict(assignments=[[0] * 20]), max_outcomes=1)
    assert measured['value'] == 0


def test_care_set_dont_cares_change_minimum_without_claiming_full_domain():
    assert exact_minimum([[[0, 1]]], 2)['value'] == 1
    restricted = exact_minimum([[[0, 1]]], 2, care_set=[[0, 0], [0, 1], [1, 0]])
    assert restricted['value'] == 0 and 'other inputs are unspecified' in restricted['scope']['domain']
    # Wide physical boundary, small affine rank: no exponential 2**30 input table.
    inputs = [[a, b] + [0] * 28 for a, b in product(range(2), repeat=2)]
    outputs = [[row[0] & row[1]] for row in inputs]
    report = exact_minimum_on_rows(inputs, outputs)
    assert report['value'] == 1 and report['scope']['inputs'] == 30
    assert verify_minimum_witness(report, inputs, outputs)['status'] == 'verified'
    changed = deepcopy(report)
    changed['witness'][0]['product'] ^= 1
    with pytest.raises(StrataError, match='replay'):
        verify_minimum_witness(changed, inputs, outputs)
    changed = deepcopy(report)
    changed['output_coefficients'][0] ^= 1
    with pytest.raises(StrataError, match='output replay'):
        verify_minimum_witness(changed, inputs, outputs)
    with pytest.raises(StrataError, match='hash'):
        verify_minimum_witness(report, inputs, [[0]] * 4)


def test_exact_search_budget_and_conflicting_care_rows_never_become_minima():
    source = km.Circuit('CCX q0 q1 q2')
    report = circuit_floors(source, clean_inputs=[2], exact=True, max_and=0)
    assert report['value'] == 1 and not report['minimum_established']
    assert report['exact_search']['kind'] == 'unknown' and report['exact_search']['value'] is None
    found = circuit_floors(source, clean_inputs=[2], exact=True, max_expansions=8)
    assert found['minimum_established'] and found['exact_search']['expansions'] == 1
    with pytest.raises(StrataError, match='conflicting'):
        exact_minimum_on_rows([[0], [0]], [[0], [1]])
    for limits in ({'timeout': float('nan')}, {'max_and': 1.5}, {'max_states': True}):
        with pytest.raises(StrataError, match='budget'):
            exact_minimum([[[0, 1]]], 2, **limits)


def test_supplied_classical_side_information_is_free_input_in_care_set_floor():
    source = km.Circuit('X q2 if b0')
    domain = dict(assignments=[dict(q0=a, q1=b, q2=0, b0=a & b) for a, b in product(range(2), repeat=2)])
    report = circuit_floors(source, domain=domain, outputs=[2], exact=True)
    assert report['value'] == 0 and report['boolean_input_variables'] == ['q0', 'q1', 'q2', 'b0']


def test_final_factored_projection_recovers_sparse_measured_outputs_after_intermediate_blowup():
    from test_factored_v041 import zero_test
    from strata.semantic import analyze_window
    text, clean = zero_test(13)
    text = text.replace('Z q24\n', '') + '\nCCX q0 q1 q25'
    clean.append(25)
    assert analyze_window(text, clean_inputs=clean, max_operations=500, max_variables=100,
                          max_terms=4096)['status'] == 'bounded'
    circuit = km.Circuit(text)
    state = functions_at(circuit, clean_inputs=clean, max_operations=500, max_variables=100, max_terms=4096)
    assert state['status'] == 'exact' and state['semantic_source_status'] == 'exact_factored'
    assert state['functions'][25] == {frozenset({0, 1})} and not state['phase']
    floor = circuit_floors(circuit, clean_inputs=clean, outputs=[25], max_operations=500,
                           max_variables=100, max_terms=4096)
    assert floor['value'] == 1 and floor['measurement_analysis']['phase_obligations_discharged']
