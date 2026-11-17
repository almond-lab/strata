from itertools import product

import kickmix as km
import numpy as np
import pytest

from strata.inference import eliminate_clean_phase_workspace
from strata.kickmix import Kickmix
from strata.observable_search import _phase_gate, observable_boundary_synthesis
from strata.semantic import equivalent_windows


def _bank(groups):
    pairs = list(product(groups[0], groups[1]))
    start = max(q for group in groups for q in group) + 1
    first = list(range(start, start + len(pairs)))
    second = [q + len(first) for q in first]
    forward = [f'CCX q{a} q{b} q{w}' for (a, b), w in zip(pairs[::-1], first)]
    forward += [f'CCX q{c} q{w} q{t}' for c in groups[2] for w, t in zip(first, second)]
    signs = [f'Z q{q}' for q in second[::-1]]
    return km.Circuit('\n'.join(forward + signs + forward[::-1])), first + second, second


def _execute(circuit, width, inputs):
    count = 1 << len(inputs)
    values = np.zeros((width, count), dtype=np.bool_)
    for i, wire in enumerate(inputs):
        values[wire] = np.arange(count) >> i & 1
    sim = km.Simulator(batch_size=count, seed=183)
    wires = [km.q(q) for q in range(width)]
    sim.write_across_shots(wires, values)
    sim.do(circuit)
    return sim.read_across_shots(wires), sim.read_phase_flipped_across_shots()


def test_complete_distributed_phase_bank_beats_the_existing_monomial_pass():
    groups = [(1, 8, 4), (3, 6, 0), (7, 2, 5)]
    source, clean, word = _bank(groups)
    result = observable_boundary_synthesis(source, clean_inputs=clean)
    assert result['status'] == 'constructed', result
    old = eliminate_clean_phase_workspace(source, clean_inputs=clean, drop_clean=True, max_check_inputs=0)
    bill, witness = result['resource_bill'], result['structural_witness']
    assert old['resource_bill']['after']['toffoli'] == result['direct_phase_comparator']['resources']['toffoli'] == 27
    assert bill['before']['toffoli'] == 72
    assert bill['after']['toffoli'] == 1
    assert bill['after']['Q'] == 9
    assert witness['retired_scratch_wires'] == clean
    assert witness['source_phase_word_wires'] == word
    assert witness['removed_phase_word'] and witness['removed_materialization_bank']
    assert witness['factored_phase_observable']
    assert result['polynomial_replay']['status'] == 'verified'
    assert result['check']['phase_equal'] and result['check']['equivalent']
    candidate = km.Circuit(result['candidate'])
    initial = _execute(source, source.num_qubits, range(9))
    final = _execute(candidate, source.num_qubits, range(9))
    assert all(np.array_equal(a, b) for a, b in zip(initial, final))


def test_no_clean_promise_needed_to_factor_same_width_and_retain_public_metadata():
    groups = [(1, 4), (2, 5), (0, 3)]
    monomials = [f'CCZ q{a} q{b} q{c}' for a, b, c in product(*groups)]
    source = km.Circuit('\n'.join(['REGISTER r0 "public wire"', 'APPEND_TO_REGISTER q9 r0',
                                   'INFO_TAG "observation"', 'NEG', *monomials]))
    def measure(circuit):
        resources = Kickmix().measured(circuit)
        return dict(resources, native_price=12 * resources['toffoli'] + resources['instructions'])
    result = observable_boundary_synthesis(source, cost='native_price', resource_measure=measure)
    candidate = km.Circuit(result['candidate'])
    assert result['status'] == 'constructed'
    assert 'INFO_TAG "observation"' in str(candidate)
    assert 'NEG' in str(candidate)
    assert source.register_data == candidate.register_data
    assert candidate.num_qubits == source.num_qubits == 10
    assert result['cost'] == measure(candidate)['native_price'] < result['direct_phase_comparator']['cost']
    assert result['structural_witness']['factored_phase_observable']
    assert not result['structural_witness']['removed_phase_word']
    assert result['proof']['truth_rows_enumerated'] == 0


def test_high_degree_observable_spends_only_available_workspace_and_retirement_is_literal():
    forward = ['CCX q0 q1 q4', 'CCX q2 q4 q5']
    source = km.Circuit('\n'.join(forward + ['CZ q3 q5'] + forward[::-1]))
    result = observable_boundary_synthesis(source, clean_inputs=(4, 5))
    assert result['resource_bill']['after']['toffoli'] == 3
    assert result['lowering']['used_clean_wires'] == [4]
    assert result['structural_witness']['retired_scratch_wires'] == [5]
    assert result['structural_witness']['removed_phase_word']
    assert result['direct_phase_comparator']['status'] == 'not_applicable'
    still_used = observable_boundary_synthesis(km.Circuit(result['candidate']), clean_inputs=(4,))
    assert still_used['lowering']['used_clean_wires'] == [4]
    assert not still_used['structural_witness']['removed_phase_word']
    assert not still_used['structural_witness']['factored_phase_observable']


@pytest.mark.parametrize('forms', [(6, 12, 10), (6, 12, 11), (7, 12, 9), (6, 7), (1,), ()])
def test_dependent_and_shifted_affine_phase_forms_match_direct_evaluation(forms):
    native = km.Circuit('\n'.join(_phase_gate(forms, [0, 1, 2])))
    values, signs = _execute(native, 3, range(3))
    expected = [all((form & (x << 1 | 1)).bit_count() % 2 for form in forms) for x in range(8)]
    assert np.array_equal(signs, expected)
    assert np.array_equal(values, [np.arange(8) >> j & 1 for j in range(3)])


@pytest.mark.parametrize('suffix', ['X q1', 'HMR q2 b0', 'CZ q0 q1 if b0', 'R q2', 'Z_POW q0 0.25', 'POP_CONDITION'])
def test_changed_boundary_or_unreconstructed_quantum_channel_fails_closed(suffix):
    source = km.Circuit('CX q0 q2\nCZ q1 q2\nCX q0 q2\n' + suffix)
    result = observable_boundary_synthesis(source, clean_inputs=(2,))
    assert result['status'] in {'not_applicable', 'unknown'}
    assert 'candidate' not in result


def test_cancelled_reads_do_not_attest_a_meaningful_retired_observable():
    source = km.Circuit('Z q1\nCCX q0 q1 q2\nZ q2\nZ q2\nCCX q0 q1 q2')
    result = observable_boundary_synthesis(source, clean_inputs=[2])
    assert result['structural_witness']['retired_scratch_wires'] == [2]
    assert not result['structural_witness']['removed_phase_word']
    assert not result['structural_witness']['factored_phase_observable']


def test_incorrect_native_lowering_cannot_escape_the_boundary_check(monkeypatch):
    from strata import observable_search
    monkeypatch.setattr(observable_search, '_phase_gate', lambda forms, available: [])
    result = observable_boundary_synthesis(km.Circuit('CCX q0 q1 q2\nZ q2\nCCX q0 q1 q2'),
                                           clean_inputs=(2,))
    assert result['status'] == 'unknown' and 'candidate' not in result
    assert any(a.get('check', {}).get('equivalent') is False for a in result['search']['attempts'])


def test_output_leak_and_sign_mutation_fail_and_public_width_survives():
    source = km.Circuit('CX q0 q2\nCZ q1 q2\nCX q0 q2\nX q6\nX q6')
    result = observable_boundary_synthesis(source, clean_inputs=(2,))
    candidate = km.Circuit(result['candidate'])
    assert candidate.num_qubits == 7
    for mutation in ('NEG', 'X q2', 'X q6'):
        assert equivalent_windows(str(source), str(candidate) + '\n' + mutation, clean_inputs=(2,))['equivalent'] is False
    for limits in (dict(max_factor_steps=1), dict(max_operations=1), dict(max_variables=1)):
        bounded = observable_boundary_synthesis(source, clean_inputs=(2,), **limits)
        assert bounded['status'] == 'unknown' and 'candidate' not in bounded
