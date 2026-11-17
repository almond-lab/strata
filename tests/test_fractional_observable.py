from fractions import Fraction
from random import Random

import kickmix as km
import numpy as np
import pytest

from strata.accounting import measured_resources
from strata.fractional_equivalence import equivalent_fractional_boundaries, has_fractional_phase
from strata.fractional_observable import fractional_boundary_synthesis


def exclusive(angle='0.1'):
    # q4 holds x0*x1 and q7 holds x0*(1-x1). q6 is a dirty public output.
    forward = ['CCX q0 q1 q4', 'X q1', 'CCX q0 q1 q7', 'X q1']
    metadata = ['APPEND_TO_REGISTER q6 r0', 'APPEND_TO_REGISTER b5 r0', 'REGISTER r0 "physical io"']
    return km.Circuit('\n'.join([*metadata, *forward, f'Z_POW q4 {angle}', f'Z_POW q7 {angle}',
                                  *reversed(forward), 'CX q0 q6']))


def native_map(text, width, inputs):
    native = km.Circuit(str(text))
    shots = 1 << len(inputs)
    sim = km.Simulator(batch_size=shots, seed=75)
    for i, q in enumerate(inputs):
        sim.write_across_shots(km.q(q), ((np.arange(shots) >> i) & 1).astype(bool))
    sim.do(native)
    return [sim.read_across_shots(km.q(q)).tolist() for q in range(width)], [
        Fraction(sim.read_shot_phase(s)) % 2 for s in range(shots)]


def test_exclusive_predicates_merge_and_restore_public_dirty_outputs_at_native_precision():
    src = exclusive()
    result = fractional_boundary_synthesis(src, clean_inputs=[4, 7])
    new = km.Circuit(result['candidate'])
    bill = result['resource_bill']
    assert bill['before']['toffoli'] == 4 and bill['after']['toffoli'] == 0
    assert (bill['before']['rotations'], bill['after']['rotations']) == (2, 1)
    assert (src.num_qubits, new.num_qubits) == (8, 7)
    assert src.register_data == new.register_data
    assert result['check']['phase_bits'] == 128
    assert result['structural_witness']['retired_phase_wires'] == [4, 7]
    assert result['structural_witness']['combined_fractional_predicates']
    assert native_map(src, 8, [0, 1, 2, 3, 5, 6]) == native_map(new, 8, [0, 1, 2, 3, 5, 6])
    # With no promise the phase depends on both original flag values.
    assert equivalent_fractional_boundaries(src, new)['equivalent'] is not True
    pinned = km.Circuit(str(new) + '\nAPPEND_TO_REGISTER q7 r1')
    assert equivalent_fractional_boundaries(src, pinned)['equivalent'] is False


def test_incompatible_objectives_price_full_native_adapter_and_rotation_cost():
    src = km.Circuit('CCX q0 q1 q2\nZ_POW q2 0.25\nCCX q0 q1 q2')
    cheap = fractional_boundary_synthesis(src, clean_inputs=[2], cost='100*toffoli+rotations+instructions')
    protected = fractional_boundary_synthesis(src, clean_inputs=[2], cost='100*rotations+toffoli+instructions')
    for result, count, paid, expected in [(cheap, 3, 0, 8), (protected, 1, 2, 105)]:
        assert result['cost'] == expected
        assert (result['resources']['rotations'], result['resources']['toffoli']) == (count, paid)
        assert result['resources'] == measured_resources(km.Circuit(result['candidate']))
    assert cheap['structural_witness']['factored_fractional_observable']
    assert not protected['structural_witness']['factored_fractional_observable']
    assert native_map(src, 3, [0, 1]) == native_map(cheap['candidate'], 3, [0, 1])


def test_overlapping_flags_have_an_integer_carry_and_wrong_xor_is_detected():
    forward = ['CX q0 q3', 'CX q1 q3', 'CX q1 q4', 'CX q2 q4']
    src = km.Circuit('\n'.join([*forward, 'Z_POW q3 0.25', 'Z_POW q4 0.25', *reversed(forward)]))
    result = fractional_boundary_synthesis(src, clean_inputs=[3, 4], cost='Q+instructions+rotations')
    assert result['status'] == 'constructed'
    expected = native_map(src, 5, [0, 1, 2])
    assert any(p == Fraction(1, 2) for p in expected[1])
    for option in [result] + result['alternatives']:
        assert native_map(option['candidate'], 5, [0, 1, 2]) == expected
    wrong = km.Circuit('CX q0 q2\nZ_POW q2 0.25\nCX q0 q2')
    check = equivalent_fractional_boundaries(src, wrong, clean_inputs=[3, 4])
    assert check['differing_output'] == 'phase'
    assert check['counterexample']['before_phase'] == '1/2' and check['counterexample']['after_phase'] == '0'


def test_distinguishes_angles_below_float_precision_and_declines_rounded_native_emission():
    exact = 'Z_POW q1 0.1'
    almost = 'Z_POW q1 0.09999999999999999999999999999999999999'
    different = equivalent_fractional_boundaries(exact, almost)
    a, b = (Fraction(different['counterexample'][f'{side}_phase']) for side in ('before', 'after'))
    assert different['equivalent'] is False and a != b and float(a) == float(b)
    # The Fourier decomposition of this controlled angle demands half an SDK tick.
    # Its algebraic construction remains executable using one declared helper.
    src = km.Circuit('CCX q0 q1 q2\nZ_POW q2 0.1\nCCX q0 q1 q2')
    r = fractional_boundary_synthesis(src, clean_inputs=[2])
    assert r['check']['equivalent'] and r['resources']['toffoli'] == 2
    assert any(a.get('reason') == 'required_rotation_below_native_precision' for a in r['attempts'])


def test_constant_phase_and_empty_emission_preserve_named_registers_and_width():
    for body, clean, total in [('X q2\nZ_POW q2 0.25\nX q2', [2], Fraction(1, 4)),
                               ('Z_POW q0 0.125\nZ_POW q0 1.875', [], 0),
                               ('Z_POW q0 0.125\nX q0\nZ_POW q0 0.125\nX q0', [], Fraction(1, 8))]:
        src = km.Circuit('APPEND_TO_REGISTER q3 r0\nREGISTER r0 "visible"\n' + body)
        r = fractional_boundary_synthesis(src, clean_inputs=clean)
        out = km.Circuit(r['candidate'])
        assert src.register_data == out.register_data and out.num_qubits == 4
        assert set(native_map(out, 4, [q for q in range(4) if q not in clean])[1]) == {total}
        if not total:
            assert r['resources']['instructions'] == 0 and r['check']['phase_zero']
            assert not r['structural_witness']['factored_fractional_observable']


@pytest.mark.parametrize('mutation', ['HMR q0 b1', 'R q0', 'BIT_STORE0 b1', 'Z_POW q1 0.25 if b3',
                                      'PUSH_CONDITION if b0\nX q1\nPOP_CONDITION', 'DEBUG_PRINT q0'])
def test_refuses_classical_public_flag_changes_measurements_and_unsupported_effects(mutation):
    text = km.Circuit('Z_POW q0 0.125\n' + mutation)
    r = fractional_boundary_synthesis(text)
    assert r['status'] == 'not_applicable' and 'candidate' not in r
    check = equivalent_fractional_boundaries(text, text)
    assert check['status'] == 'unknown' and check['equivalent'] is None


def test_physical_bindings_required_restoration_and_delta_witnesses():
    src = 'APPEND_TO_REGISTER q2 r0\nREGISTER r0 "output"\nZ_POW q0 0.25\nCX q0 q1'
    rebind = src.replace('APPEND_TO_REGISTER q2', 'APPEND_TO_REGISTER q1') + '\nAPPEND_TO_REGISTER q2 r1'
    assert equivalent_fractional_boundaries(src, rebind)['reason'] == 'public_register_binding_changed'
    check = equivalent_fractional_boundaries(src, src, clean_inputs=[1], required_zero_outputs=[1])
    assert check['equivalent'] is False and check['differing_output'] == 'q1' and check['side'] == 'original'
    assert check['counterexample']['inputs'][0]
    check = equivalent_fractional_boundaries(src, src.replace('CX q0 q1', 'X q1'), clean_inputs=[1])
    assert check['equivalent'] is False and check['differing_output'] == 'q1' and check['phase_equal']
    # A zero output required on an untouched arbitrary register is unproved.
    assert equivalent_fractional_boundaries(src, src, required_zero_outputs=[2])['equivalent'] is False


def test_budget_boundaries_and_unpromised_shared_quantum_controls():
    src = exclusive()
    assert has_fractional_phase(src) and not has_fractional_phase('Z_POW q0 1.0')
    assert fractional_boundary_synthesis(src, clean_inputs=[4, 7], max_operations=1)['status'] == 'unknown'
    assert equivalent_fractional_boundaries(src, src, max_variables=1)['equivalent'] is None
    assert equivalent_fractional_boundaries(src, src, max_terms=2)['equivalent'] is None
    assert equivalent_fractional_boundaries(src, src, max_product_terms=1)['equivalent'] is None


def test_random_mixed_permutations_and_diagonals_match_exhaustive_native_simulator():
    rng = Random(443)
    for _ in range(24):
        gates = ['Z_POW q1 0.125']
        for _ in range(9):
            q, r, s = rng.sample(range(4), 3)
            gates.append(rng.choice([f'X q{q}', f'CX q{q} q{r}', f'CCX q{q} q{r} q{s}',
                f'SWAP q{q} q{r}', f'CSWAP q{q} q{r} q{s}', f'Z_POW q{q} 0.25', f'Z_POW q{r} 1.875',
                f'CZ q{r} q{s}', 'NEG', f'CCZ q{q} q{r} q{s}']))
        source = km.Circuit('APPEND_TO_REGISTER q3 r0\n' + '\n'.join(gates))
        result = fractional_boundary_synthesis(source, clean_inputs=[3])
        assert result['status'] == 'constructed'
        for option in [result] + result['alternatives']:
            assert native_map(source, 4, range(3)) == native_map(option['candidate'], 4, range(3))


def test_deep_file_workflow_rechecks_and_prices_physical_candidate(tmp_path):
    from strata.restructure import restructure_file

    src = exclusive()
    source = tmp_path / 'supplied.kmx'
    source.write_text(str(src))
    report = restructure_file(source, tmp_path / 'out', deep_only=True, fast=True, clean_inputs=[4, 7],
                              rounds=1, max_candidates=1, cost='Q*(toffoli+rotations)')
    assert report['status'] == 'verified_improvement'
    assert report['structural_gains'][0]['qualification'] == 'factored_fractional_observable'
    check = report['attempts'][0]['check']
    assert check['all_public_inputs'] and check['exact_phase']
    chosen = km.Circuit.from_file(str(tmp_path / 'out' / 'best.kmx'))
    assert report['best']['resources'] == measured_resources(chosen)
    assert report['best']['cost'] == 7 < report['baseline']['cost'] == 48


def test_repeated_fractional_reads_that_reduce_to_integer_sign_do_not_attest_fractional_structure():
    src = km.Circuit('CCX q0 q1 q2\n' + 'Z_POW q2 0.25\n' * 4 + 'CCX q0 q1 q2')
    result = fractional_boundary_synthesis(src, clean_inputs=[2])
    assert result['check']['equivalent'] and not result['check']['fractional_phase_on_some_input']
    assert not result['structural_witness']['factored_fractional_observable']


def test_evaluate_fractional_intervals_keep_parent_width_and_label_whole_contract(tmp_path):
    from strata.discovery import evaluate_candidate
    from strata.model import digest

    source, candidate = tmp_path / 'old.kmx', tmp_path / 'new.kmx'
    shared = 'X q6\n'
    source.write_text(shared + 'Z_POW q5 0.125\nZ_POW q5 1.875')
    candidate.write_text(shared)
    report = evaluate_candidate(source, candidate, tmp_path / 'result', cost='instructions', max_window=2)
    assert report['status'] == 'verified_improvement'
    assert report['checks']['phase'] == report['contract']['phase'] == 'exact_native_dyadic'
    assert report['contract_hash'] == digest(report['contract'])
    assert report['intervals'][0]['check']['phase_bits'] == 4
    source.write_text(shared + 'Z_POW q5 0.125')
    rejected = evaluate_candidate(source, candidate, tmp_path / 'bad', max_window=2)
    assert rejected['status'] == 'rejected' and rejected['counterexample']['before_phase'] == '1/8'


def test_integer_rotation_in_measured_dirty_boundary_retains_coupled_trajectory_check(tmp_path):
    from strata.discovery import evaluate_candidate

    src, dst = tmp_path / 'src.kmx', tmp_path / 'dst.kmx'
    src.write_text('Z_POW q0 1.0\nHMR q1 b0')
    dst.write_text('HMR q1 b0\nZ q0')
    r = evaluate_candidate(src, dst, tmp_path / 'report')
    assert r['status'] == 'verified_no_gain' and r['contract']['phase'] == 'exact_sign'
    assert r['intervals'][0]['check']['measurement_variables'] == ['m0']
