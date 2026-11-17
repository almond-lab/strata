from itertools import product

import kickmix as km
import numpy as np
import pytest

from strata.kickmix import Kickmix
from strata.mixed_observable import mixed_boundary_synthesis
from strata.semantic import equivalent_windows


def _source(signs=None, metadata=()):
    # Independently materialize a four-entry product word. The dirty target
    # receives the parity of this word before it becomes a sign operand.
    pairs = list(product((0, 3), (2, 1)))
    word = list(range(7, 11))
    compute = [f'CCX q{a} q{b} q{w}' for (a, b), w in zip(pairs, word)]
    consume = [f'CX q{w} q6' for w in word[::-1]]
    observe = signs if signs is not None else ['CZ q4 q6', 'CZ q5 q6']
    return km.Circuit('\n'.join([*metadata, *compute, *consume, *observe, *compute[::-1]])), word


def _native(circuit, width, inputs):
    rows = 1 << len(inputs)
    state = np.zeros((width, rows), dtype=np.bool_)
    for i, q in enumerate(inputs):
        state[q] = np.arange(rows) >> i & 1
    sim = km.Simulator(batch_size=rows, seed=875)
    wires = [km.q(q) for q in range(width)]
    sim.write_across_shots(wires, state)
    sim.do(circuit)
    return sim.read_across_shots(wires), sim.read_phase_flipped_across_shots()


def test_pulls_back_nonconstant_sign_through_a_changed_dirty_output_then_rewrites_value_bank():
    source, clean = _source()
    result = mixed_boundary_synthesis(source, clean_inputs=clean)
    assert result['status'] == 'constructed'
    assert result['phase_method'] == 'factored_entry_sign'
    assert result['proof']['phase_applied_at'] == 'entry'
    assert result['check']['equivalent'] and result['check']['phase_equal']
    witness = result['structural_witness']
    assert witness['separated_value_and_phase']
    assert witness['retired_scratch_wires'] == clean
    assert len(witness['source_transformed_phase_read_indices']) == 2
    assert witness['emitted_value_products'] == 1 < witness['source_value_products']
    assert result['resources']['toffoli'] == 2
    assert result['resource_bill']['after_cost'] < result['resource_bill']['before_cost']
    assert any(a['mechanism'] == 'source_diagonals_at_entry' and a['check']['equivalent'] is False
               for a in result['search']['phase_rejections'])
    candidate = km.Circuit(result['candidate'])
    old = _native(source, source.num_qubits, range(7))
    new = _native(candidate, source.num_qubits, range(7))
    assert all(np.array_equal(a, b) for a, b in zip(old, new))
    backwards = result['value_body'] + '\n' + result['phase_body']
    assert equivalent_windows(str(source), backwards, clean_inputs=clean)['equivalent'] is False


def test_host_callback_prices_only_equivalent_complete_programs_and_can_choose_a_different_component():
    source, clean = _source()
    bills = []
    def measure(candidate):
        check = equivalent_windows(str(source), str(candidate), clean_inputs=clean)
        assert check['equivalent'] is True, check
        resources = Kickmix().measured(candidate)
        # An alternative backend charges a high conversion cost for CX basis
        # changes and charges phase gates and the remaining operations less.
        cx = sum(line.startswith('CX ') for line in str(candidate).splitlines())
        resources['host_cost'] = 8 * resources['Q'] + resources['instructions'] + 8 * cx
        bills.append(resources['host_cost'])
        return resources
    result = mixed_boundary_synthesis(source, clean_inputs=clean, resource_measure=measure, cost='host_cost')
    assert result['cost'] == min(bills)
    assert len(bills) == 1 + sum(a['status'] == 'verified' for a in result['search']['attempts'])
    assert len(set(bills)) > 1


def test_unchanged_source_diagonal_may_move_to_entry_only_after_verification_and_preserves_bindings():
    metadata = ['REGISTER r0 "named scratch"', 'APPEND_TO_REGISTER q10 r0', 'INFO_TAG "compute the word"']
    source, clean = _source(['NEG', 'Z_POW q0 1.0'], metadata)
    result = mixed_boundary_synthesis(source, clean_inputs=clean)
    assert result['phase_method'] == 'source_diagonals_at_entry'
    assert result['structural_witness']['source_transformed_phase_read_indices'] == []
    candidate = km.Circuit(result['candidate'])
    assert candidate.register_data == source.register_data
    assert candidate.num_qubits == source.num_qubits
    assert 'INFO_TAG "compute the word"' in str(candidate)
    assert equivalent_windows(str(source), str(candidate), clean_inputs=clean)['phase_equal']


def test_disjoint_inverse_padding_and_phase_reordering_cannot_attest_a_new_value_map():
    forward = ['CX q2 q0', 'CCX q0 q1 q3', 'CX q2 q0']
    padding = ['CCX q0 q2 q4', 'X q5', 'CCX q2 q0 q4', 'X q5']
    source = km.Circuit('\n'.join(padding + forward + ['Z q3']))
    result = mixed_boundary_synthesis(source, clean_inputs=[4, 5], cost='instructions')
    assert result['status'] == 'constructed'
    assert not result['structural_witness']['separated_value_and_phase']


@pytest.mark.parametrize('trailer', ['HMR q1 b0', 'R q0', 'Z q0 if b0', 'POP_CONDITION', 'Z_POW q0 0.25'])
def test_unsupported_channels_return_no_candidate(trailer):
    source, clean = _source()
    result = mixed_boundary_synthesis(km.Circuit(str(source) + '\n' + trailer), clean_inputs=clean)
    assert result['status'] == 'not_applicable' and 'candidate' not in result


def test_only_reference_budget_doubles_and_cross_product_limit_is_respected():
    source, clean = _source()
    result = mixed_boundary_synthesis(source, clean_inputs=clean, max_operations=len(source), max_combinations=1)
    assert result['entry_projection']['reference_operations'] > len(source)
    assert result['entry_projection']['reference_operation_limit'] == 2 * len(source)
    assert len(km.Circuit(result['candidate'])) <= len(source)
    assert len(result['search']['attempts']) == 1
    for limit in ({'max_operations': len(source) - 1}, {'max_terms': 1}):
        bounded = mixed_boundary_synthesis(source, clean_inputs=clean, **limit)
        assert bounded['status'] == 'unknown' and 'candidate' not in bounded
