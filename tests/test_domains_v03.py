import itertools
import json

import kickmix as km
import pytest

from strata.domains import load_domain, equivalent_on_domain, execute_assignment
from strata.inference import (functions_at, information_slack, is_function_of,
                              constructive_compression, execute_compression,
                              discover_dependencies, discover_compressions, eliminate_clean_phase_workspace)
from strata.model import StrataError


def domain(rows, **extra):
    return dict(version=1, type='finite', assignments=rows, **extra)


def test_named_register_union_is_canonical_complete_and_hash_bound(tmp_path):
    circuit = km.Circuit('CX q0 q2')
    promised = dict(version=1, type='finite', registers={'data': [0, 1], 'work': [2]},
                    union=[domain([{'data': 0, 'work': 0}, {'data': 1, 'work': 0}]),
                           domain([{'data': 1, 'work': 0}, {'data': 3, 'work': 1}])])
    path = tmp_path / 'promise.json'
    path.write_text(json.dumps(promised))
    normalized = load_domain(path, circuit)
    assert normalized['assignment_count'] == 3
    assert normalized['registers'] == {'data': [0, 1], 'work': [2]}
    assert load_domain(normalized, circuit) == normalized
    normalized['assignments'][0]['q2'] = 1
    with pytest.raises(StrataError, match='hash'):
        load_domain(normalized, circuit)
    with pytest.raises(StrataError, match='incomplete'):
        load_domain(domain([{'q0': 0, 'q1': 1}]), circuit)
    with pytest.raises(StrataError, match='conflicting'):
        load_domain(domain([{'data': 1, 'q0': 0, 'q2': 0}], registers={'data': [0, 1]}), circuit)


def test_finite_equivalence_preserves_dirty_outputs_and_exact_phase():
    before = km.Circuit('CX q0 q1')
    after = km.Circuit('X q1\nX q1')
    promise = domain([[0, 0], [0, 1]])
    result = equivalent_on_domain(before, after, promise)
    assert result['equivalent'] is True and result['all_input_equivalence'] is False
    assert result['assignment_count'] == 2
    different = equivalent_on_domain(before, after, domain([[1, 1]]))
    assert different['equivalent'] is False
    assert different['counterexample']['before']['outputs']['1'] == 0
    phase = equivalent_on_domain(km.Circuit('Z q0'), km.Circuit('X q0\nX q0'), domain([[1]]))
    assert phase['equivalent'] is False
    assert phase['counterexample']['before']['phase'] == 1


def test_finite_promise_cannot_silently_remove_a_public_physical_input():
    original = km.Circuit('X q1\nX q1')
    candidate = km.Circuit('X q0\nX q0')
    result = equivalent_on_domain(original, candidate, domain([[1, 1]]))
    assert result['status'] == 'unknown' and result['equivalent'] is None
    assert result['reason'] == 'physical_boundary_layout_mismatch'


def test_equal_measurement_count_does_not_prove_outcome_alignment():
    original = km.Circuit('HMR q0 b0\nHMR q1 b1')
    candidate = km.Circuit('HMR q1 b0\nHMR q0 b1')
    result = equivalent_on_domain(original, candidate, domain([[0, 0]]))
    assert result['status'] == 'unknown' and result['equivalent'] is None
    assert result['reason'] == 'measurement_event_alignment_unproved'


def test_finite_outcomes_classical_bits_and_debt_are_compared():
    circuit = km.Circuit('HMR q0 b0\nZ q1 if b0')
    promise = domain([[0, 0], [1, 1]])
    result = equivalent_on_domain(circuit, circuit, promise)
    assert result['equivalent'] is True and result['outcome_count'] == 2
    assert result['distributional_equivalence'] is False
    broken = equivalent_on_domain(circuit, km.Circuit('HMR q0 b0\nX q1\nX q1'), promise)
    assert broken['equivalent'] is False
    assert broken['counterexample']['outcomes'] == [1]
    input_condition = km.Circuit('X q0 if b4')
    with pytest.raises(StrataError, match='complete'):
        load_domain(domain([[0]]), input_condition)
    promised = domain([{'q0': 0, 'b4': 0}, {'q0': 0, 'b4': 1}], classical_inputs=['b4'])
    assert equivalent_on_domain(input_condition, input_condition, promised)['equivalent'] is True
    assert equivalent_on_domain(km.Circuit('R q0'), km.Circuit('R q0'), domain([[1]]))['status'] == 'unknown'


def test_interior_queries_do_not_execute_prefix_or_infer_clean_inputs():
    circuit = km.Circuit('Z_POW q0 0.25\nCCX q0 q1 q2')
    assert functions_at(circuit)['status'] == 'unknown'
    relation = is_function_of(circuit, 2, 2, [0, 1], from_at=1, clean_inputs=[2])
    assert relation['status'] == 'yes'
    assert relation['form']['monomials'] == [[0, 1]]
    assert is_function_of(circuit, 2, 2, [0, 1], from_at=1)['status'] == 'no'
    assert functions_at(circuit, 2, from_at=1, max_operations=1)['status'] == 'exact'
    assert functions_at(circuit, 2, from_at=1, max_variables=1)['status'] == 'unknown'
    with pytest.raises(StrataError, match='window'):
        functions_at(circuit, 1, from_at=2)


def test_slack_default_is_touched_window_and_finite_support_is_joint():
    circuit = km.Circuit('X q8\nCX q0 q2')
    slack = information_slack(circuit, from_at=1, clean_inputs=[2])
    assert slack['wires'] == [0, 2] and slack['slack_bits'] == 1
    assert information_slack(circuit, from_at=1, clean_inputs=[2], wires=range(9))['information_bits'] == 8
    base = km.Circuit('X q2\nX q2')
    promise = domain([[a, b, a ^ b] for a, b in itertools.product(range(2), repeat=2)])
    finite = information_slack(base, wires=[0, 1, 2], domain=promise)
    assert finite['distinct_states'] == 4 and finite['slack_bits'] == 1
    relation = is_function_of(base, 2, 2, [0, 1], domain=promise)
    assert relation['status'] == 'yes' and relation['evidence']['domain_hash'] == finite['domain_hash']
    assert is_function_of(base, 2, 2, [0, 1])['status'] == 'no'


def test_joint_support_does_not_treat_eleven_codes_plus_bit_as_four_bits():
    circuit = km.Circuit('X q4\nX q4')
    promise = domain([[(value >> bit) & 1 for bit in range(4)] + [tag]
                      for value in range(11) for tag in (0, 1)])
    slack = information_slack(circuit, wires=range(5), domain=promise)
    assert slack['distinct_states'] == 22 and slack['information_bits'] == 5
    assert constructive_compression(circuit, wires=range(5), domain=promise)['status'] == 'no_compression'


@pytest.mark.parametrize('source,clean,expected', [
    ('CX q0 q2\nCX q1 q2', [2], 'CX'),
    ('CCX q0 q1 q2', [2], 'CCX'),
])
def test_generic_compression_native_encoder_and_inverse(source, clean, expected):
    circuit = km.Circuit(source)
    built = constructive_compression(circuit, clean_inputs=clean)
    assert built['status'] == 'constructed'
    witness = built['witness']
    assert witness['information_bits'] == 2 and witness['distinct_states'] == 4
    assert expected in witness['native_encoder']
    for row in itertools.product(range(2), repeat=3):
        encoded = execute_compression(witness, row)
        assert execute_compression(witness, encoded, inverse=True) == list(row)
    for a, b in itertools.product(range(2), repeat=2):
        state = execute_assignment(source.splitlines(), {'q0': a, 'q1': b, 'q2': 0}, 3)
        current = [state['outputs'][str(q)] for q in range(3)]
        assert execute_compression(witness, current)[2] == 0
    assert built['resource_bill']['width_reduction'] == 0
    assert built['resource_bill']['decoder_instructions'] == built['resource_bill']['encoder_instructions']
    found = discover_dependencies(circuit, clean_inputs=clean)
    assert any(row['wire'] == 2 for row in found['relations'])


@pytest.mark.parametrize('source,clean,expected', [
    ('CCX q0 q1 q3\nCZ q2 q3\nCCX q0 q1 q3', [3], 'CCZ q0 q1 q2'),
    ('CX q1 q3\nCX q0 q3\nZ q3\nCX q0 q3\nCX q1 q3', [3], 'Z q0\nZ q1'),
])
def test_generic_clean_phase_elimination_emits_and_prices_native_candidate(source, clean, expected):
    circuit = km.Circuit(source)
    built = eliminate_clean_phase_workspace(circuit, clean_inputs=clean, drop_clean=True)
    assert built['status'] == 'constructed'
    assert expected in built['candidate']
    assert built['resource_bill']['width_reduction'] >= 1
    assert built['independent_replay']['status'] == 'verified'
    assert built['layout']['public_inputs'] == [0, 1, 2]
    assert built['layout']['wire_map'] == {'0': 0, '1': 1, '2': 2}
    assert built['layout']['omitted_clean_wires'] == [3]
    assert eliminate_clean_phase_workspace(circuit)['status'] == 'unknown'


def test_interior_clean_wire_elimination_never_renumbers_external_input():
    circuit = km.Circuit('CCX q0 q1 q2\nCZ q2 q3\nCCX q0 q1 q2')
    built = eliminate_clean_phase_workspace(circuit, clean_inputs=[2], drop_clean=True)
    assert built['layout']['wire_map']['3'] == 3
    assert built['resource_bill']['width_reduction'] == 0
    assert built['proof']['all_input_equivalence'] is False
    assert 'CCZ q0 q1 q3' in built['candidate']


def test_nonrestored_and_measurement_circuits_are_outside_phase_constructor():
    assert eliminate_clean_phase_workspace(km.Circuit('CX q0 q1'), clean_inputs=[1])['status'] == 'unknown'
    assert eliminate_clean_phase_workspace(km.Circuit('HMR q0 b0'), clean_inputs=[0])['status'] == 'unknown'


def test_automatic_cuts_find_a_temporary_nonlinear_word_and_budget_is_explicit():
    circuit = km.Circuit('CCX q0 q1 q2\nZ q2\nCCX q0 q1 q2')
    result = discover_compressions(circuit, clean_inputs=[2])
    assert result['status'] == 'complete_in_family'
    nonlinear = next(row for row in result['candidates'] if row['witness']['at'] == 1)
    assert nonlinear['witness']['native_encoder'] == 'CCX q0 q1 q2'
    assert nonlinear['resource_bill']['native_decoder_counts']['CCX'] == 1
    assert discover_compressions(circuit, clean_inputs=[2], max_cuts=1)['status'] == 'partial'


def test_serialized_compression_rejects_permutation_that_is_the_wrong_inverse():
    built = constructive_compression(km.Circuit('CCX q0 q1 q2'), clean_inputs=[2])
    witness = built['witness']
    witness['decoder'] = list(range(8))
    with pytest.raises(StrataError, match='mutual inverses'):
        execute_compression(witness, [0, 0, 0])


def test_window_queries_materialize_only_budget_and_refuse_inherited_scope():
    long = km.Circuit('\n'.join('X q0' for _ in range(100)))
    result = functions_at(long, max_operations=4)
    assert result['status'] == 'unknown' and result['reason'] == 'operation_budget'
    selected = functions_at(long, from_at=98, max_operations=2)
    assert selected['status'] == 'exact' and selected['at'] == 100
    scoped = km.Circuit('BIT_STORE1 b0\nPUSH_CONDITION if b0\nX q0\nPOP_CONDITION')
    assert functions_at(scoped, from_at=2)['reason'] == 'window_starts_inside_condition_scope'
