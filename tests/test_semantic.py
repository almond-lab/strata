"""Native-file semantic lifting, with arbitrary dirty inputs and exact sign."""
import json
import random

import kickmix as km
import numpy as np
import pytest

from strata.semantic import analyze_window, equivalent_windows, lift_regions


def evaluate(poly, assignment):
    result = 0
    for term in poly['monomials']:
        result ^= all(assignment[q] for q in term)
    return result


def test_dirty_compute_consume_uncompute_reports_full_function():
    result = analyze_window('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    assert result['status'] == 'exact'
    assert result['kind'] == 'compute_consume_uncompute'
    assert result['outputs']['3']['monomials'] == [[2], [3], [0, 1]]
    assert result['outputs']['2']['monomials'] == [[2]]
    assert result['boundary_width'] == 4
    assert result['preserved_wires'] == [0, 1, 2]
    episode = next(f for f in result['features'] if f['kind'] == 'compute_consume_uncompute')
    assert episode['restored_wires'][0]['wire'] == 2
    assert equivalent_windows('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2',
                              'CCX q0 q1 q3')['status'] == 'different'
    assert equivalent_windows('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2',
                              'CX q2 q3\nCCX q0 q1 q3')['equivalent'] is True


def test_affine_map_is_a_multi_operation_function():
    result = analyze_window('X q0\nCX q0 q1\nSWAP q1 q2')
    assert result['outputs']['0']['monomials'] == [[], [0]]
    assert result['outputs']['1']['monomials'] == [[2]]
    assert result['outputs']['2']['monomials'] == [[], [0], [1]]
    assert result['kind'] == 'affine_transform'
    feature = next(f for f in result['features'] if f['kind'] == 'affine_transform')
    assert feature['matrix']['2'] == [0, 1]
    assert feature['offset']['2'] == 1


def test_bilinear_multi_output_accumulator_and_nonbipartite_quadratic():
    text = 'CCX q0 q2 q4\nCCX q1 q3 q4\nCCX q0 q3 q5\nCCX q1 q2 q5'
    result = analyze_window(text)
    assert result['kind'] == 'bilinear_accumulator'
    feature = next(f for f in result['features'] if f['kind'] == 'bilinear_accumulator')
    assert feature['factor_partition'] == [[0, 1], [2, 3]]
    assert feature['deltas']['4']['monomials'] == [[0, 2], [1, 3]]
    triangle = analyze_window('CCX q0 q1 q3\nCCX q1 q2 q3\nCCX q0 q2 q3')
    assert triangle['kind'] == 'quadratic_accumulator'


def test_boolean_monomials_are_idempotent_and_cancel_over_gf2():
    result = analyze_window('CX q0 q1\nCCX q0 q1 q2')
    assert result['outputs']['2']['monomials'] == [[0], [2], [0, 1]]
    assert equivalent_windows('CX q0 q1\nCX q0 q1', '')['equivalent'] is True
    assert equivalent_windows('X q71\nX q71', '')['equivalent'] is True


def test_phase_polynomial_tracks_gate_time_values_and_global_sign():
    result = analyze_window('X q0\nCZ q0 q1\nX q0')
    assert result['phase']['monomials'] == [[1], [0, 1]]
    assert result['kind'] == 'phase_polynomial'
    assert equivalent_windows('X q0\nZ q0\nX q0', 'Z q0')['status'] == 'different'
    assert equivalent_windows('X q0\nZ q0\nX q0', 'NEG\nZ q0')['equivalent'] is True
    assert equivalent_windows('CCZ q0 q1 q2', '')['phase_equal'] is False


@pytest.mark.parametrize('gate', ['R q0', 'Z_POW q0 0.25', 'PUSH_CONDITION if b0',
                                  'CX q0 q0', 'X q0 q1', 'CX q0 b0'])
def test_unsupported_never_silently_dropped(gate):
    result = analyze_window(f'X q0\n{gate}\nX q0')
    assert result['status'] == 'opaque'
    assert 'outputs' not in result
    assert equivalent_windows(f'X q0\n{gate}\nX q0', '')['equivalent'] is None


def test_conditions_lift_as_whole_scopes_including_nested_conditions():
    circuit = km.Circuit('X q0\nX q0\nPUSH_CONDITION if b0\n'
                         'CCX q0 q1 q2\nPUSH_CONDITION if b1\nX q3\n'
                         'POP_CONDITION\nCX q2 q3\nPOP_CONDITION\nX q4\nX q4')
    report = lift_regions(circuit)
    assert [(r['start'], r['stop']) for r in report['regions']] == [(0, 11)]
    assert report['unknown'] == []
    assert report['coverage']['exact_operations'] == 11
    assert report['coverage']['exact_paid_operations'] == 1
    assert report['regions'][0]['outputs']['2']['monomials'] == [[2], [0, 1, 'b0']]


def test_hmr_lifts_fresh_outcome_without_clean_input_assumptions():
    circuit = km.Circuit('CCX q0 q1 q2\nHMR q2 b0\nCX q2 q3\nCX q2 q4')
    report = lift_regions(circuit)
    assert [(r['start'], r['stop']) for r in report['regions']] == [(0, 4)]
    assert report['regions'][0]['outputs']['3']['monomials'] == [[3]]
    assert report['regions'][0]['phase']['monomials'] == [[2, 'm0'], [0, 1, 'm0']]
    assert report['unknown'] == []


@pytest.mark.parametrize('format', ['kmx', 'kmb'])
def test_native_text_and_binary_files_lift_without_adapter(tmp_path, format):
    circuit = km.Circuit('INFO_TAG "hint, not a semantic assumption"\n'
                         'CCX q7 q9 q11\nCCX q8 q10 q11\nCCZ q7 q9 q11')
    path = tmp_path / ('untagged.' + format)
    circuit.to_file(path, format=format)
    report = lift_regions(path)
    json.dumps(report, allow_nan=False)
    assert report['coverage']['scope'] == 'full'
    assert report['regions'][0]['boundary_width'] == 5
    assert report['regions'][0]['start'] == 1
    assert report['metadata'] == [{'at': 0, 'operation': 'INFO_TAG'}]
    assert report['coverage']['exact_paid_fraction'] == 1
    assert report['coverage']['structural_paid_fraction'] == 1


def test_single_primitive_does_not_claim_compound_lifting_coverage():
    report = lift_regions(km.Circuit('CCX q0 q1 q2'))
    assert report['coverage']['exact_paid_fraction'] == 1
    assert report['coverage']['structural_paid_fraction'] == 0
    assert not report['regions'][0]['structural']


def test_restored_intermediates_recover_episode_boundaries_inside_a_window():
    text = ('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2\n'
            'CCX q4 q5 q6\nCX q6 q7\nCCX q4 q5 q6')
    report = lift_regions(text, max_window=128)
    assert [(r['start'], r['stop']) for r in report['regions']] == [(0, 3), (3, 6)]
    assert all(r['boundary_basis'] == 'restored_intermediate_episode' for r in report['regions'])
    assert report['coverage']['compound_algebra_paid_fraction'] == 1
    assert report['coverage']['high_level_stage_coverage'] is None


def test_window_budgets_fail_closed_and_lifter_splits_exact_subwindows():
    assert analyze_window('X q0\nX q1', max_variables=1)['reason'] == 'variable_budget'
    assert analyze_window('X q0\nX q0', max_operations=1)['reason'] == 'operation_budget'
    assert analyze_window('X q0', max_terms=1)['reason'] == 'term_budget'
    assert analyze_window('X q0\nX q1\nCCX q0 q1 q2', max_product_terms=1)['reason'] == 'product_budget'
    report = lift_regions('X q0\nX q1\nCCX q0 q1 q2', max_terms=4)
    assert report['coverage']['scope'] == 'full'
    assert report['coverage']['exact_operations'] == 3
    assert len(report['regions']) > 1
    bounded = lift_regions('X q0', max_terms=1)
    assert bounded['coverage']['exact_operations'] == 0
    assert bounded['unknown'][0]['status'] == 'bounded'


def test_scan_budget_separates_unscanned_suffix_from_coverage():
    result = lift_regions('X q0\nX q0\nCCX q0 q1 q2', max_ops=2)
    assert result['coverage']['scope'] == 'prefix'
    assert result['coverage']['total_instructions'] == 3
    assert result['unknown'][-1] == dict(start=2, stop=3, status='unscanned', reason='scan_budget')


def test_exact_anf_matches_native_simulator_on_all_inputs_for_random_circuits():
    rng = random.Random(2701)
    width, cases = 5, 32
    inputs = np.array([[(case >> q) & 1 for case in range(cases)] for q in range(width)], dtype=bool)
    arity = {'X': 1, 'CX': 2, 'CCX': 3, 'SWAP': 2, 'Z': 1, 'CZ': 2, 'CCZ': 3, 'NEG': 0}
    for _ in range(12):
        lines = []
        for _ in range(35):
            op = rng.choice(list(arity))
            lines.append(' '.join([op, *(f'q{q}' for q in rng.sample(range(width), arity[op]))]))
        result = analyze_window(lines)
        assert result['status'] == 'exact'
        sim = km.Simulator(batch_size=cases, seed=7)
        for q in range(width):
            sim.write_across_shots(km.q(q), inputs[q])
        sim.do(km.Circuit('\n'.join(lines)))
        for q in range(width):
            expected = [evaluate(result['outputs'][str(q)], inputs[:, case]) for case in range(cases)]
            assert np.array_equal(sim.read_across_shots(km.q(q)), expected)
        expected_phase = [evaluate(result['phase'], inputs[:, case]) for case in range(cases)]
        assert np.array_equal(sim.read_phase_flipped_across_shots(), expected_phase)


@pytest.mark.parametrize(('before', 'after', 'kind'), [
    ('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2', 'CCX q0 q1 q3', 'output'),
    ('CX q0 q1\nCCX q0 q1 q2', 'CCX q0 q1 q2', 'output'),
    ('X q0\nCZ q0 q1\nX q0', 'CZ q0 q1', 'phase'),
    ('CCZ q0 q1 q2', '', 'phase'),
    ('', 'X q71', 'output'),
    ('', 'NEG', 'phase'),
])
def test_counterexample_replays_complete_outputs_and_sign_in_native_sdk(before, after, kind):
    result = equivalent_windows(before, after)
    example = result['counterexample']
    assert example['kind'] == kind
    assert example == equivalent_windows(before, after)['counterexample']
    assert set(example['inputs']) == {str(q) for q in result['inputs']}
    for name, text in [('before', before), ('after', after)]:
        sim = km.Simulator(batch_size=1, seed=6)
        for q, bit in example['inputs'].items():
            sim.write_across_shots(km.q(int(q)), np.array([bit], dtype=bool))
        sim.do(km.Circuit(text))
        assert {q: int(sim.read_across_shots(km.q(int(q)))[0]) for q in example['inputs']} == example[name + '_outputs']
        assert int(sim.read_phase_flipped_across_shots()[0]) == example[name + '_phase']
    assert (example['before_outputs'] != example['after_outputs'] or
            example['before_phase'] != example['after_phase'])


def test_counterexample_retains_nonzero_arbitrary_dirty_helper():
    result = equivalent_windows('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2',
                                'CCX q0 q1 q3')
    assert result['counterexample']['inputs'] == {'0': 0, '1': 0, '2': 1, '3': 0}


def test_counterexample_shannon_search_handles_many_variables_without_enumeration():
    text = '\n'.join(f'CZ q{q} q{q+1}' for q in range(0, 64, 2))
    result = equivalent_windows(text, '', max_variables=64)
    example = result['counterexample']
    assert example['inputs'] == {str(q): int(q >= 62) for q in range(64)}
    assert example['before_phase'] == 1 and example['after_phase'] == 0
    assert example['before_outputs'] == example['after_outputs']


def test_equivalent_or_unknown_windows_do_not_invent_counterexamples():
    assert 'counterexample' not in equivalent_windows('X q0\nX q0', '')
    assert 'counterexample' not in equivalent_windows('HMR q0 b0', '')
