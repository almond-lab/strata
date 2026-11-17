"""Regression proofs for #61, #73, #76 and #80, including unsafe near misses."""
import itertools
import json
import tracemalloc

import kickmix as km
import pytest

from strata.circuit_tape import canonical_tape
from strata.discovery import _opcodes, evaluate_candidate
from strata.domains import equivalent_on_domain
from strata.lab import exhaustive
from strata.lab_program import compile_program
from strata.model import Contract, Register
from strata.streaming import scan_circuit


def finite(rows):
    return dict(version=1, type='finite', assignments=rows)


def independent(before, after, rows, **kw):
    return equivalent_on_domain(km.Circuit(before), km.Circuit(after), finite(rows),
                                measurement_mode='independent-map', internal_bits=[0, 1], **kw)


def test_188_hmr_instrument_matches_unmeasured_exact_map_with_linear_witnesses():
    block = 'CCX q0 q1 q2\nHMR q2 b0\nCZ q0 q1 if b0'
    before = '\n'.join([block] * 188)
    after = 'X q2\nX q2'
    rows = [[a, b, 0] for a, b in itertools.product((0, 1), repeat=2)]
    result = independent(before, after, rows)
    assert result['equivalent'] is True
    assert result['checked_cases'] == 4 * (189 + 1)
    assert result['represented_outcomes']['baseline'] == 1 << 188
    assert result['reductions'] == ['affine_hmr', 'full_enumeration']
    coupled = equivalent_on_domain(km.Circuit(before), km.Circuit(after), finite(rows))
    assert coupled['reason'] == 'measurement_outcome_spaces_differ'
    public = equivalent_on_domain(km.Circuit(before), km.Circuit(after), finite(rows),
                                  measurement_mode='independent-map')
    assert public['reason'] == 'measurement_dependent_boundary_map'


def test_independent_instruments_preserve_dirty_phase_and_reset_obligations():
    repaired = 'CX q0 q1\nHMR q1 b0\nZ q0 if b0'
    identity = 'X q1\nX q1'
    assert independent(repaired, identity, [[0, 0], [1, 0]])['equivalent'] is True
    dirty = independent(repaired, identity, [[0, 1]])
    assert dirty['equivalent'] is not True
    debt = independent('CX q0 q1\nHMR q1 b0', identity, [[1, 0]])
    assert debt['reason'] == 'measurement_dependent_boundary_map'
    phase = independent(repaired + '\nZ q0', identity, [[1, 0]])
    assert phase['equivalent'] is False and phase['counterexample']['before']['phase'] == 1
    reset = independent('R q1\nHMR q1 b0', identity, [[0, 1]])
    assert reset['reason'] == 'reset_requires_zero'
    cleanup = independent('HMR q1 b0\nX q1', identity, [[0, 0]], required_zero_outputs=[1])
    assert cleanup['reason'] == 'clean_output_not_restored'


def test_nonlinear_neg_requires_combined_branches_and_budget_is_unknown():
    text = 'HMR q0 b0\nHMR q1 b1\nPUSH_CONDITION if b0\nNEG if b1\nPOP_CONDITION'
    assert compile_program(text).affine_hmr is False
    identity = 'X q1\nX q1'
    result = independent(text, identity, [[0, 0]])
    assert result['reason'] == 'measurement_dependent_boundary_map'
    assert result['independence_witness']['outcomes'] == [1, 1]
    assert independent(text, identity, [[0, 0]], max_outcomes=3)['reason'] == 'measurement_outcome_budget'
    linear = 'HMR q0 b0\nHMR q1 b1'
    assert independent(linear, identity, [[0, 0]], max_trajectories=3)['reason'] == 'measurement_trajectory_budget'


def test_affine_theorem_uses_explicit_classical_boundary_assignment():
    text = 'HMR q0 b0\nHMR q1 b1\nNEG if b7'
    rows = [{'q0': 0, 'q1': 0, 'b7': 1}]
    result = independent(text, 'X q1\nX q1', rows)
    assert result['equivalent'] is False
    assert result['counterexample']['before']['phase'] == 1


def test_kickmix_iadd_conditioned_neg_reduction_matches_full_enumeration():
    builder = km.CircuitBuilder()
    reg = builder.create_quantum_register(4, 'r')
    builder.iadd(1, target=reg)
    circuit = km.Circuit(str(builder.finish_circuit()) + '\nHMR q4 b1\nHMR q5 b2')
    assert '\nNEG\n' in str(circuit)
    assert compile_program(str(circuit)).affine_hmr is True
    contract = Contract('increment', (Register('r', tuple(range(4))),
                        Register('scratch', tuple(range(4, circuit.num_qubits)), role='clean')), 'increment')
    oracle = lambda inputs, scale: ({'r': (inputs['r'] + 1) % 16}, 0)
    reduced = exhaustive(circuit, contract, oracle)
    full = exhaustive(circuit, contract, oracle, branch_strategy='full')
    assert reduced.passed and full.passed
    assert reduced.reduction == 'affine_hmr'
    assert reduced.executed_branches == compile_program(str(circuit)).hmr_count + 1
    broken = km.Circuit(str(circuit).replace('\nNEG\n', '\n', 1))
    assert exhaustive(broken, contract, oracle).passed is False


@pytest.mark.parametrize('format', ['kmx', 'kmb'])
def test_range_units_match_native_order_with_register_metadata(tmp_path, format):
    c = km.Circuit('REGISTER r0 "data"\nAPPEND_TO_REGISTER q0 r0\nAPPEND_TO_REGISTER q1 r0\n'
                   'INFO_TAG "start"\nX q0\nCCX q0 q1 q2\nINFO_TAG "stop"')
    path = tmp_path / ('c.' + format)
    c.to_file(path, format=format)
    loaded = km.Circuit.from_file(path)
    operation = scan_circuit(loaded, tmp_path/'ops', start=2, stop=3)
    text = scan_circuit(loaded, tmp_path/'text', start=5, stop=6, range_unit='text-line')
    assert operation['census'] == text['census'] == {'CCX': 1}
    assert operation['range_unit'] == 'operation'
    assert operation['text_line_range'] == text['text_line_range'] == dict(start=5, stop=6)
    assert operation['operation_range'] == text['operation_range'] == dict(start=2, stop=3)
    assert operation['full_file_operations'] == 4
    assert operation['groups'][0]['examples'][0]['start'] == 2
    assert text['groups'][0]['examples'][0]['start'] == 5


def test_repetitive_tape_deletion_is_local_and_python_memory_bounded(tmp_path):
    # The changed pair is in the interior, followed by a long repetitive suffix.
    prefix = 'CX q0 q1\n' * 120000
    suffix = 'CX q1 q0\n' * 120000 + 'R q2'
    base, candidate = tmp_path/'base.kmx', tmp_path/'candidate.kmx'
    base.write_text(prefix + 'X q2\nX q2\n' + suffix)
    candidate.write_text(prefix + suffix)
    del prefix, suffix
    tracemalloc.start()
    result = evaluate_candidate(base, candidate, tmp_path/'report', clean_inputs=[2], max_window=8)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert result['status'] == 'verified_no_gain'
    assert result['alignment']['changed_intervals'] == [dict(start=120000, stop=120002,
                                                            candidate_start=120000, candidate_stop=120000)]
    assert result['recovery']['semantic_comparisons'] == 1
    assert result['recovery']['attempts'][0]['source_operations'] == 2
    assert result['checks']['clean_outputs_restored'] is True
    assert peak < 12 * 1024 * 1024


def test_clean_local_substitution_does_not_fabricate_cleanup(tmp_path):
    # Dirty reset in an unchanged prefix still blocks the whole clean contract.
    base, candidate = tmp_path/'base.kmx', tmp_path/'candidate.kmx'
    base.write_text('R q0\n' + 'CX q0 q1\n'*20 + 'X q2\nX q2\nR q2')
    candidate.write_text('R q0\n' + 'CX q0 q1\n'*20 + 'R q2')
    result = evaluate_candidate(base, candidate, tmp_path/'report', clean_inputs=[2], max_window=4)
    assert result['status'] == 'unknown'
    assert result['checks']['reason'] == 'clean_restoration_semantics_unavailable'
    assert result['recovery']['semantic_comparisons'] == 1


def test_alignment_never_materializes_an_unbounded_unmatched_middle(tmp_path):
    a = km.Circuit('X q0\n' * 10000)
    b = km.Circuit('X q1\n' * 10000)
    with canonical_tape(a) as before, canonical_tape(b) as after:
        assert _opcodes(before, after) == [('replace', 0, 10000, 0, 10000)]
        assert len(before.offsets) == 10


def test_cli_independent_map_binds_private_records_and_cleanup(tmp_path, capsys):
    from strata.cli import main
    base, candidate, promise = tmp_path/'base.kmx', tmp_path/'candidate.kmx', tmp_path/'domain.json'
    base.write_text('CCX q0 q1 q2\nHMR q2 b0\nCZ q0 q1 if b0')
    candidate.write_text('X q2\nX q2')
    promise.write_text(json.dumps(finite([[a, b, 0] for a, b in itertools.product((0, 1), repeat=2)])))
    assert main(['evaluate-file', str(base), str(candidate), '--out', str(tmp_path/'report'),
                 '--domain', str(promise), '--measurement-mode', 'independent-map',
                 '--internal-bits', '0', '--clean-inputs', '2']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'verified_domain_improvement'
    assert result['contract']['internal_classical_outputs'] == [0]
    assert result['checks']['clean_outputs_restored'] is True


@pytest.mark.parametrize('command', ['discover', 'inspect-file'])
def test_cli_native_operation_range_has_explicit_text_line_alternative(tmp_path, capsys, command):
    from strata.cli import main
    source = tmp_path/'source.kmb'
    km.Circuit('REGISTER r0 "data"\nAPPEND_TO_REGISTER q0 r0\nX q0\nCCX q0 q1 q2').to_file(source, format='kmb')
    for unit, selected in [('operation', '1:2'), ('text-line', '3:4')]:
        assert main([command, str(source), '--out', str(tmp_path/unit), '--range', selected,
                     '--range-unit', unit, '--quiet-progress']) == 0
        result = json.loads(capsys.readouterr().out)
        scan = result.get('scan', result)
        assert scan['range_unit'] == unit and scan['census'] == {'CCX': 1}


def test_concrete_interpreter_stops_reading_a_generator_at_its_budget():
    from strata.domains import execute_assignment, _Unknown
    def operations():
        yield from ['X q0'] * 4
        raise AssertionError('consumed beyond operation budget plus overflow witness')
    with pytest.raises(_Unknown, match='operation_budget'):
        execute_assignment(operations(), {'q0': 0}, 1, max_operations=3)


def test_parse_failure_cannot_bind_changed_raw_bytes_to_an_earlier_hash(tmp_path, monkeypatch):
    from strata.model import StrataError
    from strata.store import Store
    base, candidate = tmp_path/'base.kmx', tmp_path/'candidate.kmx'
    base.write_text('X q0')
    candidate.write_text('INVALID_BEFORE q0')
    retain = Store.blob_file
    def mutate_at_retention(store, path):
        if path == candidate:
            candidate.write_text('INVALID_AFTER q0')
        return retain(store, path)
    monkeypatch.setattr(Store, 'blob_file', mutate_at_retention)
    with pytest.raises(StrataError, match='candidate artifact changed before retention'):
        evaluate_candidate(base, candidate, tmp_path/'report')
    assert not (tmp_path/'report'/'evaluation.json').exists()
