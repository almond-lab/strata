"""Outcome-parametric native semantics checked against independently forced SDK circuits."""
import itertools
import random
import shlex

import kickmix as km
import numpy as np
import pytest

from strata.semantic import analyze_window, equivalent_windows, lift_regions


def evaluate(poly, assignment):
    value = 0
    for term in poly['monomials']:
        value ^= int(all(assignment[q] for q in term))
    return value


def forced_native(text, outcomes):
    """Let native kickmix implement all control/stack/bit logic independently."""
    native = km.Circuit(text)
    fresh = native.num_qubits
    rows, measurement = [], 0
    for line in str(native).splitlines():
        words = shlex.split(line)
        if words[0] != 'HMR':
            rows.append(line)
            continue
        suffix = ' ' + ' '.join(words[words.index('if'):]) if 'if' in words else ''
        outcome = outcomes[measurement]
        if outcome:
            rows.append(f'Z {words[1]}{suffix}')
        rows.append(f'SWAP {words[1]} q{fresh}{suffix}')
        rows.append(f'BIT_STORE{outcome} {words[2]}{suffix}')
        measurement += 1
        fresh += 1
    return km.Circuit('\n'.join(rows))


def check_native(text, clean_inputs=()):
    result = analyze_window(text, clean_inputs=clean_inputs, max_variables=64)
    assert result['status'] == 'exact', result
    variables = [q for q in result['inputs'] if q not in clean_inputs] + result['classical_inputs']
    rows = [dict(zip(variables, values), **{}) for values in itertools.product((0, 1), repeat=len(variables))]
    for assignment in rows:
        assignment.update({q: 0 for q in clean_inputs})
    for outcomes in itertools.product((0, 1), repeat=len(result['measurement_variables'])):
        sim = km.Simulator(batch_size=len(rows), seed=731)
        for q in result['inputs']:
            sim.write_across_shots(km.q(q), np.array([a[q] for a in rows], dtype=bool))
        for bit in result['classical_inputs']:
            sim.write_across_shots(km.b(int(bit[1:])), np.array([a[bit] for a in rows], dtype=bool))
        sim.do(forced_native(text, outcomes))
        assignments = [dict(a, **dict(zip(result['measurement_variables'], outcomes))) for a in rows]
        for q, poly in result['outputs'].items():
            assert list(sim.read_across_shots(km.q(int(q))).astype(int)) == [evaluate(poly, a) for a in assignments]
        for bit, poly in result['classical_outputs'].items():
            assert list(sim.read_across_shots(km.b(int(bit[1:]))).astype(int)) == [evaluate(poly, a) for a in assignments]
        assert list(sim.read_phase_flipped_across_shots().astype(int)) == [evaluate(result['phase'], a) for a in assignments]
        # The independently retained factored roots must mean the same values.
        for assignment in assignments:
            values = []
            for node in result['factored']['nodes']:
                if node['op'] == 'constant':
                    value = node['value']
                elif node['op'] == 'variable':
                    value = assignment[node['variable']]
                elif node['op'] == 'and':
                    value = int(all(values[i] for i in node['arguments']))
                else:
                    value = sum(values[i] for i in node['arguments']) % 2
                values.append(value)
            for q, root in result['factored']['outputs'].items():
                assert values[root] == evaluate(result['outputs'][q], assignment)
            assert values[result['factored']['phase']] == evaluate(result['phase'], assignment)
    return result


@pytest.mark.parametrize('text', [
    'CSWAP q0 q1 q2\nCSWAP q1 q2 q3 if b0\nNEG if b1',
    'BIT_STORE1 b0\nBIT_INVERT b1 if b0\nX q0 if b1\nBIT_STORE0 b0 if b1\nZ q1 if b0',
    'PUSH_CONDITION if b0\nBIT_STORE0 b0\nX q0\nPUSH_CONDITION if b1\nBIT_INVERT b1\nCZ q0 q1\nPOP_CONDITION\nX q1\nPOP_CONDITION',
    'CCX q0 q1 q2\nHMR q2 b0\nCZ q0 q1 if b0',
    'HMR q0 b0 if b0\nX q1 if b0\nHMR q1 b0 if b1\nNEG if b0',
    'HMR q0 b0\nBIT_INVERT b0\nHMR q1 b1 if b0\nBIT_INVERT b1 if b0\nCZ q2 q3 if b1',
    'HMR q0 b0\nPUSH_CONDITION if b0\nBIT_STORE0 b0\nHMR q1 b1\nZ q2 if b1\nPOP_CONDITION',
    'Z_POW q0 3.0\nZ_POW q1 -2.0\nZ_POW q0 -1.0 if b0',
])
def test_native_outcomes_classical_snapshot_and_fresh_measurement_semantics(text):
    check_native(text)


def test_reset_requires_explicit_boundary_promise_or_proved_zero():
    assert analyze_window('R q9')['reason'] == 'reset_not_proven_zero'
    assert analyze_window('X q0\nR q9')['reason'] == 'reset_not_proven_zero'
    assert analyze_window('X q0\nR q0', clean_inputs=[0])['reason'] == 'reset_not_proven_zero'
    result = check_native('R q2\nCCX q0 q1 q2\nCCX q0 q1 q2\nR q2', clean_inputs=[2])
    assert len(result['effects']['reset_proofs']) == 2
    assert result['clean_input_assumptions'] == [2]
    # Legality is proved on the active branch, not by requiring unconditional zero.
    check_native('HMR q0 b0\nX q0 if b0\nBIT_INVERT b0\nR q0 if b0')


def test_measurement_debt_preserves_dirty_boundary_and_checks_current_values():
    text = 'CCX q0 q1 q2\nHMR q2 b0\nCZ q0 q1 if b0'
    dirty = analyze_window(text)
    assert dirty['phase']['monomials'] == [[2, 'm0']]
    assert dirty['effects']['phase_debts'][0]['status'] == 'outstanding'
    clean = check_native(text, clean_inputs=[2])
    assert clean['effects']['phase_debts'][0]['status'] == 'discharged'
    wrong = analyze_window('CCX q0 q1 q2\nHMR q2 b0\nCX q0 q1\nCZ q0 q1 if b0', clean_inputs=[2])
    assert wrong['phase']['monomials'] == [[0, 'm0']]
    repaired = check_native('CCX q0 q1 q2\nHMR q2 b0\nCX q0 q1\nCZ q0 q1 if b0\nZ q0 if b0', clean_inputs=[2])
    assert repaired['effects']['phase_debts'][0]['status'] == 'discharged'
    assert repaired['effects']['phase_events'][0]['values']['1']['monomials'] == [[0], [1]]


def test_measurement_variable_names_and_events_are_never_physical_wires():
    result = analyze_window('HMR q71 b0\nHMR q99 b0\nZ q0 if b0')
    assert result['inputs'] == result['quantum_inputs'] == [0, 71, 99]
    assert result['measurement_variables'] == ['m0', 'm1']
    assert result['classical_outputs']['b0']['monomials'] == [['m1']]
    assert result['is_permutation'] is False
    reordered = equivalent_windows('HMR q0 b0\nHMR q1 b1', 'HMR q1 b1\nHMR q0 b0')
    assert reordered['status'] == 'unknown'
    assert reordered['reason'] == 'measurement_event_alignment_unproved'


def test_outcome_counterexample_replays_and_detects_classical_output_change():
    before, after = 'HMR q0 b0\nZ q1 if b0', 'HMR q0 b0'
    result = equivalent_windows(before, after)
    witness = result['counterexample']
    assert result['status'] == 'different' and witness['measurement_outcomes'] == {'m0': 1}
    for name, text in [('before', before), ('after', after)]:
        sim = km.Simulator(batch_size=1, seed=5)
        for q, value in witness['inputs'].items():
            sim.write_across_shots(km.q(int(q)), np.array([value], dtype=bool))
        sim.do(forced_native(text, tuple(witness['measurement_outcomes'].values())))
        assert int(sim.read_phase_flipped_across_shots()[0]) == witness[name + '_phase']
    assert equivalent_windows('BIT_INVERT b0', '')['counterexample']['kind'] == 'classical_output'
    assert equivalent_windows('HMR q0 b0\nX q1\nX q1', 'HMR q0 b0')['equivalent']


def test_fractional_phase_remains_unknown_including_sign_invisible_difference():
    result = equivalent_windows('Z_POW q0 0.25', '')
    assert result['status'] == 'unknown'
    assert result['before']['reason'] == 'unsupported_fractional_phase'


def test_bounded_or_broken_scope_is_never_reinterpreted_as_unconditional():
    text = 'BIT_STORE1 b0\nPUSH_CONDITION if b0\nBIT_STORE0 b0\nX q0\nPOP_CONDITION'
    complete = lift_regions(text)
    assert complete['coverage']['exact_operations'] == 5
    assert complete['regions'][0]['outputs']['0']['monomials'] == [[], [0]]
    bounded = lift_regions(text, max_window=2)
    assert bounded['regions'] == []
    assert bounded['unknown'][0]['reason'] == 'operation_budget'
    broken = lift_regions('PUSH_CONDITION if b0\nZ_POW q0 0.5\nX q0\nPOP_CONDITION\nX q1')
    assert [(r['start'], r['stop']) for r in broken['regions']] == [(4, 5)]


def test_factored_dag_retains_product_of_sums():
    result = analyze_window('CX q0 q1\nCX q2 q3\nCCX q1 q3 q4')
    roots = {n['id']: n for n in result['factored']['nodes']}
    products = [n for n in roots.values() if n['op'] == 'and']
    assert any(all(roots[child]['op'] == 'xor' for child in node['arguments']) for node in products)
    assert result['outputs']['4']['term_count'] == 5


def test_random_native_classical_and_measurement_programs():
    rng = random.Random(511)
    choices = ['X q0 if b0', 'CX q0 q1 if b1', 'CSWAP q0 q1 q2',
               'BIT_INVERT b0 if b1', 'BIT_STORE1 b1 if b0', 'BIT_STORE0 b0',
               'CZ q0 q1 if b0', 'NEG if b1', 'BIT_INVERT b1']
    for _ in range(8):
        lines = rng.choices(choices, k=7)
        lines.insert(2, 'HMR q0 b0 if b1')
        lines.insert(6, 'HMR q1 b1')
        check_native('\n'.join(lines))


def test_clean_artifact_promises_are_never_reapplied_at_window_cuts():
    text = 'R q0\nX q0\nX q0\nR q0'
    result = lift_regions(text, clean_inputs=[0])
    assert [(r['start'], r['stop']) for r in result['regions']] == [(0, 4)]
    assert result['regions'][0]['semantics']['clean_input_assumptions'] == [0]
    assert result['limits']['clean_inputs'] == [0]
    for limits in ({'max_window': 2}, {'max_ops': 2}):
        bounded = lift_regions(text, clean_inputs=[0], **limits)
        assert bounded['regions'] == []
        assert bounded['unknown'][0]['reason'] == 'clean_inputs_require_complete_single_window'
    # A dirty reset at an interior metadata cut must not inherit the input promise.
    dirty = lift_regions('R q0\nX q0\nINFO_TAG "cut"\nR q0', clean_inputs=[0])
    assert dirty['regions'] == []
    assert dirty['unknown'][0]['reason'] == 'reset_not_proven_zero'


def test_contextual_overflow_is_one_unknown_span_and_never_resumes_unconditionally():
    report = lift_regions('HMR q0 b0\n' + 'X q0 if b0\n' * 10000, max_window=8)
    assert report['regions'] == []
    assert report['unknown'] == [dict(start=0, stop=10001, status='bounded',
                                      reason='operation_budget', operation_count=10001)]
