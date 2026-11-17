from copy import deepcopy
from fractions import Fraction
import hashlib

import kickmix as km
import numpy as np
import pytest

from strata.cost import Formula
from strata.file_optimizer import _resynthesize, optimize_circuit, verify_optimization
from strata.model import StrataError
from strata.semantic import analyze_window


def native_result(circuit, width):
    shots = 1 << width
    numbers = np.arange(shots, dtype=np.uint64)
    initial = np.array([((numbers >> q) & 1).astype(bool) for q in range(width)])
    sim = km.Simulator(batch_size=shots, seed=401)
    wires = [km.q(q) for q in range(width)]
    sim.write_across_shots(wires, initial)
    sim.do(circuit)
    return sim.read_across_shots(wires), tuple(Fraction(sim.read_shot_phase(i)) % 2
                                             for i in range(shots))


def test_dependency_schedule_keeps_original_controls_until_their_last_consumer():
    # q1 changes inside the episode.  Its effect on helper cleanup is compensated
    # explicitly, so q4 is returned for every arbitrary initial helper value.
    original = km.Circuit('CCX q0 q1 q4\nCX q4 q3\nCX q0 q1\n'
                          'CCX q0 q1 q4\nCX q0 q4')
    candidate, report = optimize_circuit(original)
    assert report['status'] == 'improved'
    lines = str(candidate).splitlines()
    assert lines.index('CCX q0 q1 q3') < lines.index('CX q0 q1')
    before, before_phase = native_result(original, 5)
    after, after_phase = native_result(candidate, 5)
    assert np.array_equal(before, after)
    assert before_phase == after_phase


def test_phase_is_synthesized_on_original_inputs_and_preserves_global_sign():
    original = km.Circuit('X q2\nCCX q0 q1 q2\nCX q2 q3\nZ q3\n'
                          'Z q2\nCCX q0 q1 q2\nX q2')
    candidate, report = optimize_circuit(original)
    assert report['status'] == 'improved'
    before, before_phase = native_result(original, 4)
    after, after_phase = native_result(candidate, 4)
    assert np.array_equal(before, after)
    assert before_phase == after_phase
    assert report['checks']['phase'] == 'exact'


def test_cyclic_original_input_dependencies_fail_closed():
    semantics = analyze_window(['CX q0 q1', 'CX q2 q0', 'CX q1 q2'])
    emitted, reason = _resynthesize(semantics, max_operations=128)
    assert emitted is None
    assert reason == 'cyclic_original_input_dependencies'


@pytest.mark.parametrize('consumer,reason', [
    ('CCX q2 q3 q4', 'output_delta_degree_exceeds_two'),
    ('CCZ q2 q3 q4', 'phase_degree_exceeds_three'),
])
def test_higher_degree_functions_are_preserved(consumer, reason):
    original = km.Circuit('CCX q0 q1 q2\n' + consumer + '\nCCX q0 q1 q2')
    candidate, report = optimize_circuit(original)
    assert str(candidate) == str(original)
    assert report['regions'][0]['reason'] == reason
    assert report['checks']['equivalent'] is True


def test_static_objective_is_required_to_decrease():
    original = km.Circuit('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    candidate, report = optimize_circuit(original, Formula('Q'))
    assert str(candidate) == str(original)
    assert report['regions'][0]['reason'] == 'static_objective_did_not_decrease'
    candidate, report = optimize_circuit(original, 'instructions')
    assert report['candidate']['cost'] == 2
    assert report['baseline']['cost'] == 3


def test_disjoint_substitutions_keep_opaque_and_metadata_boundaries():
    first = 'CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2'
    barrier = 'INFO_TAG "barrier"\nHMR q4 b0\nCZ q1 q3 if b0'
    original = km.Circuit(first + '\n' + barrier + '\n' + first)
    candidate, report = optimize_circuit(original)
    accepted = [row for row in report['regions'] if row['status'] == 'accepted']
    assert len(accepted) == 2
    assert barrier in str(candidate)
    assert verify_optimization(original, candidate, report)['equivalent'] is True
    assert report['checks']['checked_regions'] == 2
    corrupted = km.Circuit(str(candidate).replace('CZ q1 q3 if b0', 'CZ q0 q3 if b0'))
    forged = deepcopy(report)
    forged['candidate']['sha256'] = hashlib.sha256(str(corrupted).encode()).hexdigest()
    assert verify_optimization(original, corrupted, forged)['reason'] == 'outside_region_changed'


def test_report_does_not_make_changed_artifacts_equivalent():
    original = km.Circuit('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    candidate, report = optimize_circuit(original)
    mutant = km.Circuit(str(candidate).replace('CX q2 q3', 'CX q1 q3'))
    assert verify_optimization(original, mutant, report)['reason'] == 'artifact_hash_mismatch'
    forged = deepcopy(report)
    forged['candidate']['sha256'] = hashlib.sha256(str(mutant).encode()).hexdigest()
    forged['regions'][0]['replacement_sha256'] = forged['candidate']['sha256']
    forged['regions'][0]['check']['equivalent'] = True
    assert verify_optimization(original, mutant, forged)['reason'] == 'region_equivalence_not_established'


def test_metadata_cannot_be_inserted_under_an_equivalent_function_receipt():
    original = km.Circuit('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    candidate, report = optimize_circuit(original)
    mutant = km.Circuit(str(candidate) + '\nINFO_TAG "inserted"')
    forged = deepcopy(report)
    forged['candidate']['sha256'] = hashlib.sha256(str(mutant).encode()).hexdigest()
    forged['regions'][0]['replacement_sha256'] = forged['candidate']['sha256']
    forged['regions'][0]['candidate_stop'] += 1
    assert verify_optimization(original, mutant, forged)['reason'] == 'replacement_contains_unsupported_or_metadata_operation'


def test_resource_evidence_is_bound_to_the_artifacts():
    original = km.Circuit('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    candidate, report = optimize_circuit(original)
    report['candidate']['resources']['toffoli'] = 0
    assert verify_optimization(original, candidate, report)['reason'] == 'resource_measurement_mismatch'


def test_objective_evidence_is_recomputed_and_can_be_pinned():
    original = km.Circuit('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    candidate, report = optimize_circuit(original)
    assert verify_optimization(original, candidate, report, formula='Q')['reason'] == 'objective_mismatch'
    report['candidate']['cost'] = 0
    assert verify_optimization(original, candidate, report)['reason'] == 'objective_value_mismatch'
    custom = Formula('weight(Q)*toffoli', functions={'weight': lambda q: q * q})
    candidate, report = optimize_circuit(original, custom)
    assert verify_optimization(original, candidate, report, formula=custom)['equivalent'] is True


def test_operation_and_variable_budgets_do_not_allow_partial_equivalence():
    original = km.Circuit('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    for limits in ({'max_operations': 2}, {'max_wires': 2}, {'max_terms': 1}):
        candidate, report = optimize_circuit(original, **limits)
        assert str(candidate) == str(original)
        assert report['checks']['equivalent'] is True  # artifact equality, no proposed rewrite
        assert report['checks']['checked_regions'] == 0
    with pytest.raises(StrataError, match='budgets'):
        optimize_circuit(original, max_operations=0)


def test_cancelled_consumers_with_unrelated_effect_are_not_restructuring():
    original = km.Circuit('CCX q0 q1 q2\nCX q2 q3\nX q4\n'
                          'CX q2 q3\nCCX q0 q1 q2')
    candidate, report = optimize_circuit(original)
    assert str(candidate) == str(original)
    assert not any(row['structural_change'] for row in report['regions'])
    assert report['regions'][0]['reason'] == 'materialization_has_no_proven_net_boundary_effect'


def test_profile_priority_budget_selects_later_larger_candidate():
    from strata.semantic import lift_regions

    original = km.Circuit('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2\n'
                          'INFO_TAG "boundary"\n'
                          'CCX q4 q5 q9\nCCX q6 q7 q9\nCX q9 q8\n'
                          'CCX q6 q7 q9\nCCX q4 q5 q9')
    ids = [region['id'] for region in lift_regions(original)['regions']]
    _, first_report = optimize_circuit(original, max_candidates=1)
    candidate, ranked_report = optimize_circuit(original, region_order=list(reversed(ids)), max_candidates=1)
    assert first_report['attempted_order'] == [ids[0]]
    assert ranked_report['attempted_order'] == [ids[-1]]
    assert ranked_report['selection_priority'][0] == ids[-1]
    assert ranked_report['candidate']['cost'] < first_report['candidate']['cost']
    assert str(candidate).startswith('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    assert ranked_report['search_budget']['deferred'] == 1
    deferred = [r for r in ranked_report['regions'] if r['status'] == 'deferred']
    assert [r['id'] for r in deferred] == [ids[0]]
    assert deferred[0]['disposition'] == 'open'
    assert verify_optimization(original, candidate, ranked_report)['equivalent'] is True


def test_cheap_ineligible_regions_do_not_consume_candidate_budget():
    from strata.semantic import lift_regions

    original = km.Circuit('CCX q0 q1 q2\nCCX q2 q3 q4\nCCX q0 q1 q2\n'
                          'INFO_TAG "boundary"\n'
                          'CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    ids = [region['id'] for region in lift_regions(original)['regions']]
    candidate, report = optimize_circuit(original, region_order=ids, max_candidates=1)
    assert report['attempted_order'] == [ids[-1]]
    assert report['search_budget']['eligible'] == 1
    assert report['status'] == 'improved'
    unchanged, zero = optimize_circuit(original, max_candidates=0)
    assert str(unchanged) == str(original)
    assert zero['search_budget']['attempted'] == 0
    assert zero['search_budget']['deferred'] == 1
