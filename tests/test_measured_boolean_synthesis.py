from itertools import product

import kickmix as km
import pytest

from strata.accounting import measured_resources
from strata.boolean_synthesis import joint_boundary_synthesis, lower_xor_and_witness
from strata.measured_boolean_synthesis import (measured_xor_and_witness, measured_boundary_synthesis,
                                              prove_product_cleanup, verify_measured_witness_candidate,
                                              measured_symbolic_boundary_synthesis)
from strata.measurement_equivalence import equivalent_independent_outputs
from strata.model import StrataError, digest
from strata.restructure import restructure_file
from strata.semantic import equivalent_windows
from test_scheduled_boolean_synthesis import chain_witness


LAYOUT = dict(input_wires=list(range(5)), target_wires=[5], clean_wires=[6, 7, 8], num_qubits=9)


def measured_chain(**options):
    witness, rows, values = chain_witness()
    result = measured_xor_and_witness(witness, rows, values, **LAYOUT, **options)
    measured = [row for row in result['alternatives'] if row['cleanup_strategy'] == 'guarded_hmr']
    return witness, rows, values, result, measured


def test_hmr_cleanup_frontier_has_real_native_gains_and_full_bills():
    witness, rows, values, result, measured = measured_chain()
    assert result['status'] == 'constructed'
    assert [(a['resources']['Q'], a['resources']['toffoli']) for a in measured] == [(8, 5), (9, 4)]
    reverse = [row for row in result['alternatives'] if row['cleanup_strategy'] == 'reverse']
    assert [(a['resources']['Q'], a['resources']['toffoli']) for a in reverse] == [(8, 9), (9, 7)]
    pure = lower_xor_and_witness(witness, rows, values, **LAYOUT)['candidate']
    for row in measured:
        assert row['ledger_matches_native']
        assert row['measurement_bill']['hmr'] == len(row['introduced_private_bits'])
        assert len(set(row['introduced_private_bits'])) == len(row['introduced_private_bits'])
        assert row['resources_static']['toffoli'] == row['resources_expected']['toffoli']
        assert row['resources_expected']['instructions'] < row['resources_static']['instructions']
        assert all(step['cleanup_proof']['status'] == 'proved' for step in row['kernel_ledger'] if step['hmr'])
        check = equivalent_independent_outputs(pure, row['candidate'], clean_inputs=[6, 7, 8],
                                               internal_bits=row['required_internal_bits'])
        assert check['equivalent'] is True, check
        assert equivalent_windows(pure, row['candidate'], clean_inputs=[6, 7, 8], max_operations=1000,
                                  max_variables=128, max_terms=65536).get('equivalent') is not True


def test_dropped_or_wrong_phase_repair_is_rejected_for_all_outcomes():
    witness, rows, values, _, measured = measured_chain()
    good = measured[-1]
    lines = good['candidate'].splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith('CZ '))
    variants = [lines[:at] + lines[at + 1:], lines[:at] + ['Z q0 if ' + lines[at].split()[-1]] + lines[at + 1:]]
    pure = lower_xor_and_witness(witness, rows, values, **LAYOUT)['candidate']
    for mutation in variants:
        bad = '\n'.join(mutation)
        check = verify_measured_witness_candidate(bad, witness, rows, values, **LAYOUT,
                                                  internal_bits=good['required_internal_bits'])
        assert check['equivalent'] is False and check['reason'] == 'measurement_phase_not_repaired'
        check = equivalent_independent_outputs(pure, bad, clean_inputs=[6, 7, 8],
                                               internal_bits=good['required_internal_bits'])
        assert check.get('equivalent') is not True


def test_cleanup_requires_actual_current_product_and_preserved_parents():
    args = dict(left=2, right=4, available=[0, 1], target=2, bit=0)
    assert prove_product_cleanup('CCX q0 q1 q2', clean_inputs=[2], **args)['status'] == 'proved'
    dirty = prove_product_cleanup('CCX q0 q1 q2', **args)
    assert dirty['reason'] == 'target_not_current_operand_product'
    overwritten = prove_product_cleanup('CCX q0 q1 q2\nX q0', clean_inputs=[2], **args)
    assert overwritten['reason'] == 'target_not_current_operand_product'
    conditional = prove_product_cleanup('CCX q0 q1 q2 if b1', clean_inputs=[2], **{**args, 'bit': 2})
    assert conditional['reason'] == 'target_not_current_operand_product'
    restored = prove_product_cleanup('CCX q0 q1 q2\nX q0\nX q0', clean_inputs=[2], **args)
    assert restored['status'] == 'proved'
    with pytest.raises(StrataError, match='unavailable'):
        prove_product_cleanup('CCX q0 q1 q2', **{**args, 'available': [0, None]})
    with pytest.raises(StrataError, match='aliases'):
        prove_product_cleanup('CCX q0 q1 q2', **{**args, 'target': 0})


def test_nonlinear_parent_is_kept_for_a_two_wire_repair_and_affine_degeneracy_reverses():
    nonlinear = prove_product_cleanup('CCX q0 q1 q3\nCCX q3 q2 q4', left=2, right=4,
                                      available=[3, 2], target=4, bit=0, clean_inputs=[3, 4])
    assert nonlinear['status'] == 'proved'
    assert nonlinear['repair'] == 'CZ q3 q2 if b0'
    overwritten = prove_product_cleanup('CCX q0 q1 q3\nCCX q3 q2 q4\nCCX q0 q1 q3',
                                        left=2, right=4, available=[3, 2], target=4, bit=0, clean_inputs=[3, 4])
    assert overwritten['reason'] == 'target_not_current_operand_product'
    linear = prove_product_cleanup('CX q0 q1', left=2, right=2, available=[0], target=1, bit=0, clean_inputs=[1])
    assert linear['status'] == 'not_applicable' and linear['mode'] == 'reverse'
    assert linear['lines'] == ['CX q0 q1']


def affine_witness():
    rows = [list(row) for row in product(range(2), repeat=4)]
    tables = [(1 << len(rows)) - 1] + [sum(row[i] << j for j, row in enumerate(rows)) for i in range(4)]
    gates = []
    for left, right in [(1 | 2 | 4, 4 | 8), (1 << 5, 1 << 4)]:
        a = b = 0
        for i, table in enumerate(tables):
            if left >> i & 1: a ^= table
            if right >> i & 1: b ^= table
        gates.append(dict(left=a, right=b, product=a & b, left_coefficients=left, right_coefficients=right))
        tables.append(a & b)
    values = [[(tables[-1] >> i) & 1] for i in range(len(rows))]
    return dict(value=2, witness=gates, output_coefficients=[1 << 6],
                contract_hash=digest(dict(inputs=rows, outputs=values))), rows, values


def test_overlapping_complemented_affine_controls_reprepare_exactly():
    witness, rows, values = affine_witness()
    layout = dict(input_wires=[0, 1, 2, 3], target_wires=[4], clean_wires=[5], num_qubits=6)
    result = measured_xor_and_witness(witness, rows, values, **layout)
    candidate = next(row for row in result['alternatives'] if row['cleanup_strategy'] == 'guarded_hmr')
    assert 'HMR q5 b0' in candidate['candidate']
    assert 'CX ' in candidate['candidate'] and 'X ' in candidate['candidate']
    pure = lower_xor_and_witness(witness, rows, values, **layout)['candidate']
    assert equivalent_independent_outputs(pure, candidate['candidate'], clean_inputs=[5], internal_bits=[0])['equivalent']


SMALL = 'CCX q0 q1 q4\nCCX q4 q2 q5\nCX q5 q3\nCCX q4 q2 q5\nCCX q0 q1 q4'


def test_wrapper_preserves_existing_classical_bits_and_allocates_above_parent_width():
    source = km.Circuit('BIT_INVERT b0\nBIT_INVERT b0\n' + SMALL)
    result = measured_boundary_synthesis(source, clean_inputs=[4, 5], classical_bit_offset=21,
                                         max_and=2, timeout=5)
    assert result['status'] == 'constructed', result
    assert result['selected_cleanup_strategy'] == 'guarded_hmr'
    assert result['introduced_private_bits'] == [21]
    assert result['required_internal_bits'] == [21]
    assert result['proof']['equivalent'] is True
    assert equivalent_independent_outputs(str(source), result['candidate'], clean_inputs=[4, 5],
                                          internal_bits=[21])['equivalent'] is True
    with pytest.raises(StrataError, match='classical layout'):
        measured_boundary_synthesis(source, clean_inputs=[4, 5], classical_bit_offset=0)


def test_existing_changed_classical_outputs_need_explicit_permission():
    source = km.Circuit('BIT_STORE1 b0\n' + SMALL)
    denied = measured_boundary_synthesis(source, clean_inputs=[4, 5], max_and=2, timeout=5)
    assert denied['status'] == 'unknown' and denied['required_internal_bits'] == [0]
    accepted = measured_boundary_synthesis(source, clean_inputs=[4, 5], internal_bits=[0], max_and=2, timeout=5)
    assert accepted['status'] == 'constructed'
    assert accepted['required_internal_bits'] == [0, 1]
    assert equivalent_independent_outputs(str(source), accepted['candidate'], clean_inputs=[4, 5],
                                          internal_bits=[1])['equivalent'] is False


def test_reverse_fallback_remains_available_for_instruction_sensitive_objective():
    result = measured_boundary_synthesis(km.Circuit(SMALL), clean_inputs=[4, 5],
                                         cost='instructions', max_and=2, timeout=5)
    assert result['status'] == 'constructed'
    assert result['selected_cleanup_strategy'] == 'reverse'
    assert result['introduced_private_bits'] == []
    assert any(a['cleanup_strategy'] == 'guarded_hmr' for a in result['alternatives'])


def test_phase_repairs_with_ccz_are_charged_in_static_and_expected_costs():
    text = 'HMR q4 b0\nCCZ q0 q1 q2 if b0'
    assert measured_resources(text)['toffoli'] == 1
    assert measured_resources(text, accounting='expected')['toffoli'] == 0.5


def test_episode_new_outcomes_do_not_collide_with_parent_classical_bits(tmp_path):
    source = tmp_path / 'source.kmx'
    source.write_text('BIT_INVERT b20\nBIT_INVERT b20\n' + SMALL)
    result = restructure_file(source, tmp_path / 'out', clean_inputs=[4, 5], scan_range=(2, 7),
                              max_operations=100, rounds=1)
    assert result['status'] == 'verified_improvement', result
    candidate = (tmp_path / 'out' / 'best.kmx').read_text()
    outcomes = [int(line.split()[2][1:]) for line in candidate.splitlines() if line.startswith('HMR ')]
    assert outcomes and min(outcomes) >= 21
    assert equivalent_independent_outputs(source.read_text(), candidate, clean_inputs=[4, 5],
                                          internal_bits=outcomes)['equivalent'] is True


def test_guard_budget_exhaustion_retains_verified_reverse_alternative():
    _, _, _, result, _ = measured_chain(max_apply_steps=1)
    assert result['status'] == 'constructed'
    assert any(row['cleanup_strategy'] == 'reverse' for row in result['alternatives'])
    assert not any(row['cleanup_strategy'] == 'guarded_hmr' for row in result['alternatives'])


def test_automatic_retrieval_propagates_new_private_bits_at_top_level(tmp_path):
    source = tmp_path / 'source.kmx'
    source.write_text('BIT_INVERT b20\nBIT_INVERT b20\n' + 'X q10\n' * 12 +
                      SMALL + '\n' + 'X q10\n' * 12)
    result = restructure_file(source, tmp_path / 'out', clean_inputs=[4, 5],
                              max_operations=8, max_candidates=24, rounds=1)
    assert result['status'] == 'verified_improvement', result
    candidate = (tmp_path / 'out' / 'best.kmx').read_text()
    outcomes = [int(line.split()[2][1:]) for line in candidate.splitlines() if line.startswith('HMR ')]
    assert outcomes and min(outcomes) >= 21
    assert set(result['best']['private_bits']) == set(outcomes)
    assert equivalent_independent_outputs(source.read_text(), candidate, clean_inputs=[4, 5],
                                          internal_bits=result['best']['private_bits'])['equivalent'] is True


def wide_parity_product():
    basis = [f'CX q{i+1} q{i}' for i in range(0, 12, 2)]
    chain = ['CCX q0 q2 q13'] + [f'CCX q{12+i} q{2*i+2} q{13+i}' for i in range(1, 5)]
    return km.Circuit('\n'.join(basis + chain + ['CX q17 q12'] + chain[::-1] + basis[::-1]))


def test_twelve_input_symbolic_dag_reaches_reverse_scheduling_and_measured_cleanup():
    from strata.algebraic_synthesis import algebraic_boundary_synthesis
    from strata.symbolic_boolean_graph import verify_symbolic_graph
    source = wide_parity_product(); clean = list(range(13, 18))
    algebraic = algebraic_boundary_synthesis(source, clean_inputs=clean, max_orders=2, timeout=3)
    assert algebraic['status'] == 'constructed', algebraic
    assert len(algebraic['dag']['input_wires']) == 12
    graph_proof = verify_symbolic_graph(source, algebraic['dag'], clean_inputs=clean)
    assert graph_proof['truth_rows_enumerated'] == 0 and graph_proof['minimum_claimed'] is False
    result = measured_symbolic_boundary_synthesis(source, algebraic_result=algebraic, clean_inputs=clean,
                                                   max_operations=1000)
    assert result['status'] == 'constructed', result
    assert result['witness_basis'] == 'verified_symbolic_polynomial_DAG'
    assert result['truth_rows_enumerated'] == 0
    assert result['resource_bill']['Q'] == 17 and result['resource_bill']['toffoli'] == 5
    reverse = [row for row in result['alternatives'] if row['cleanup_strategy'] == 'reverse']
    measured = [row for row in result['alternatives'] if row['cleanup_strategy'] == 'guarded_hmr']
    assert [(row['resources']['Q'], row['resources']['toffoli']) for row in reverse] == [(16, 11), (17, 9)]
    assert [(row['resources']['Q'], row['resources']['toffoli']) for row in measured] == [(16, 6), (17, 5)]
    assert all(row['proof']['truth_rows_enumerated'] == 0 for row in result['alternatives'])
    assert all(row['whole_boundary_check']['equivalent'] is True for row in result['alternatives'])
    broken = dict(algebraic['dag'], output_coefficients=[0])
    with pytest.raises(StrataError, match='polynomial replay mismatch'):
        verify_symbolic_graph(source, broken, clean_inputs=clean)
    bad_phase = result['candidate'].replace(next(line for line in result['candidate'].splitlines()
                                                if line.startswith('CZ ')) + '\n', '', 1)
    check = equivalent_independent_outputs(str(source), bad_phase, clean_inputs=clean,
                                           internal_bits=result['introduced_private_bits'])
    assert check.get('equivalent') is not True


def test_symbolic_dag_can_select_reverse_cleanup_under_full_parent_pricing():
    from strata.algebraic_synthesis import algebraic_boundary_synthesis
    source = wide_parity_product(); clean = list(range(13, 18))
    algebraic = algebraic_boundary_synthesis(source, clean_inputs=clean, max_orders=1, timeout=3)
    result = measured_symbolic_boundary_synthesis(source, algebraic_result=algebraic, clean_inputs=clean,
        cost='instructions', max_operations=1000,
        resource_measure=lambda circuit: dict(instructions=len(circuit) + 1000 * circuit.operation_statistics()['HMR']))
    assert result['status'] == 'constructed'
    assert result['selected_cleanup_strategy'] == 'reverse' and result['introduced_private_bits'] == []


def test_twelve_input_factor_schedule_and_hmr_are_reached_by_actual_deep_search(tmp_path):
    # Four three-input parities feed a distributed two-layer product bank and
    # two dirty output consumers. The full support exceeds the truth-table cap.
    basis = [f'CX q{start + offset} q{start}' for start in (0, 3, 6, 9) for offset in (1, 2)]
    bank = ['CCX q0 q3 q14', 'CCX q0 q6 q15', 'CCX q14 q9 q16', 'CCX q15 q9 q17']
    consumer = ['CX q16 q12', 'CX q17 q12', 'CX q14 q13', 'CX q15 q13']
    original = '\n'.join(basis + bank + consumer + bank[::-1] + basis[::-1])
    source = tmp_path / 'source.kmx'; source.write_text(original)
    result = restructure_file(source, tmp_path / 'out', clean_inputs=[14, 15, 16, 17],
                              rounds=1, max_candidates=5, max_operations=1000, deep_only=True)
    assert result['status'] == 'verified_improvement', result
    measured = next(row for row in result['attempts'] if row['mechanism'] == 'measurement_debt_lowering')
    assert measured['status'] == 'verified'
    construction = measured['construction']
    assert construction['selected_graph_source'] == 'algebraic_result'
    assert construction['truth_rows_enumerated'] == 0
    assert construction['witness_basis'] == 'verified_symbolic_polynomial_DAG'
    assert construction['structural_witness']['removed_materialization_bank']
    assert construction['introduced_private_bits']
    candidate = (tmp_path / 'out' / 'best.kmx').read_text()
    assert result['best']['resources']['toffoli'] == 2
    assert result['best']['resources']['Q'] == 15
    assert equivalent_independent_outputs(original, candidate, clean_inputs=[14, 15, 16, 17],
                                          internal_bits=result['best']['private_bits'])['equivalent'] is True
