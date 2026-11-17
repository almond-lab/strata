"""Changed-neighborhood recovery never lifts an entire large source tape."""
import kickmix as km
import pytest

import strata.discovery as discovery


def write(tmp_path, name, text):
    path = tmp_path / name
    km.Circuit(text).to_file(path, format='kmb')
    return path


@pytest.mark.parametrize('edited,expected', [('X q2\nX q2', 'verified_no_gain'),
                                            ('CX q0 q1\nCX q0 q2', 'rejected')])
def test_large_dense_file_proves_only_the_changed_neighborhood(tmp_path, monkeypatch, edited, expected):
    prefix = 'CCX q0 q1 q2\nCX q2 q3\n' * 50_000
    source = write(tmp_path, 'source.kmb', prefix + edited + '\n' + prefix)
    child = write(tmp_path, 'child.kmb', prefix + prefix)
    monkeypatch.setattr(discovery, 'lift_regions', lambda *a, **k: pytest.fail('whole-file lift is forbidden'))
    actual = discovery.equivalent_windows
    lengths = []
    def bounded(before, after, **limits):
        lengths.append((len(before), len(after)))
        assert max(len(before), len(after)) <= 32
        return actual(before, after, **limits)
    monkeypatch.setattr(discovery, 'equivalent_windows', bounded)
    report = discovery.evaluate_candidate(source, child, tmp_path / 'out', max_window=32)
    assert report['status'] == expected
    assert report['recovery']['strategy'] == 'bounded_changed_neighborhoods'
    assert len(lengths) <= report['recovery']['limits']['max_proof_attempts']
    assert all(100_000-32 <= row['start'] <= row['stop'] <= 100_002+32
               for row in report['recovery']['attempts'])
    if expected == 'verified_no_gain':
        assert lengths == [(2, 0)]
        assert report['checks']['equivalent'] is True
    else:
        assert report['counterexample']['kind'] == 'output'
        assert report['checks']['equivalent'] is False


def test_failed_minimal_span_grows_to_include_compensating_edits(tmp_path):
    source = write(tmp_path, 'source.kmb', 'CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    child = write(tmp_path, 'child.kmb', 'CX q2 q3\nCCX q0 q1 q3')
    report = discovery.evaluate_candidate(source, child, tmp_path / 'out')
    assert report['status'] == 'verified_improvement'
    attempts = report['recovery']['attempts']
    assert attempts[0]['equivalent'] is False and attempts[-1]['equivalent'] is True
    assert (attempts[-1]['start'], attempts[-1]['stop']) == (0, 3)
    assert len(report['intervals']) == 1
    bounded = discovery.evaluate_candidate(source, child, tmp_path / 'bounded', max_window=2)
    assert bounded['status'] == 'unknown'
    assert bounded['recovery']['stop_reason'] == 'local_proof_operation_budget'
    assert bounded['recovery']['stopped_by'] == 'max_operations'
    assert bounded['recovery']['attempts'][-1]['reason'] == 'operation_budget'


def test_attempt_budget_does_not_promote_a_partial_failure_or_success(tmp_path):
    source = write(tmp_path, 'source.kmb', 'CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    child = write(tmp_path, 'child.kmb', 'CX q2 q3\nCCX q0 q1 q3')
    result = discovery.evaluate_candidate(source, child, tmp_path / 'out', max_proof_attempts=1)
    assert result['status'] == 'unknown' and result['accepted_gain'] is None
    assert result['checks']['equivalent'] is None
    assert result['recovery']['stop_reason'] == 'local_proof_attempt_budget'
    assert result['recovery']['stopped_by'] == 'max_proof_attempts'
    assert len(result['recovery']['attempts']) == 1


def test_joint_proof_can_skip_many_edits_then_check_a_distant_neighborhood(tmp_path):
    before = 'CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2\n' * 20
    after = 'CX q2 q3\nCCX q0 q1 q3\n' * 20
    padding = 'CX q4 q5\n' * 1000
    source = write(tmp_path, 'source.kmb', before + padding + 'X q2\nX q2')
    child = write(tmp_path, 'child.kmb', after + padding)
    result = discovery.evaluate_candidate(source, child, tmp_path / 'out', max_window=128, max_proof_attempts=4)
    assert result['status'] == 'verified_improvement'
    assert result['recovery']['changed_spans'] > 4
    assert len(result['recovery']['attempts']) <= 4
    assert result['recovery']['attempts'][-1]['start'] >= 1060


def test_inherited_condition_is_enclosed_before_any_semantic_comparison(tmp_path, monkeypatch):
    source = write(tmp_path, 'source.kmb', 'PUSH_CONDITION if b0\nX q0\nPOP_CONDITION\nX q1\nX q1')
    child = write(tmp_path, 'child.kmb', 'PUSH_CONDITION if b0\nX q1\nPOP_CONDITION\nX q1\nX q1')
    actual = discovery.equivalent_windows
    def enclosed(before, after, **limits):
        assert before[0] == after[0] == 'PUSH_CONDITION if b0'
        assert 'POP_CONDITION' in before and 'POP_CONDITION' in after
        return actual(before, after, **limits)
    monkeypatch.setattr(discovery, 'equivalent_windows', enclosed)
    result = discovery.evaluate_candidate(source, child, tmp_path / 'out', max_window=8)
    assert result['status'] == 'rejected'
    assert result['counterexample']['classical_inputs'] == {'b0': 1}
    assert result['recovery']['attempts'][0]['reason'] == 'inherited_condition_scope'


def test_large_inherited_condition_scope_remains_unknown(tmp_path, monkeypatch):
    padding = 'CX q0 q1\n' * 100
    source = write(tmp_path, 'source.kmb', 'PUSH_CONDITION if b0\n' + padding + 'X q0\nX q0\n' + padding + 'POP_CONDITION')
    child = write(tmp_path, 'child.kmb', 'PUSH_CONDITION if b0\n' + padding + padding + 'POP_CONDITION')
    monkeypatch.setattr(discovery, 'equivalent_windows', lambda *a, **k: pytest.fail('inherited condition was discarded'))
    result = discovery.evaluate_candidate(source, child, tmp_path / 'out', max_window=16)
    assert result['status'] == 'unknown' and result['checks']['equivalent'] is None
    assert all(row['reason'] == 'inherited_condition_scope' for row in result['recovery']['attempts'])


def test_local_check_preserves_hmr_provenance_and_exact_sign(tmp_path):
    source = write(tmp_path, 'source.kmb', 'HMR q0 b0\nHMR q1 b1')
    child = write(tmp_path, 'child.kmb', 'HMR q1 b0\nHMR q0 b1')
    result = discovery.evaluate_candidate(source, child, tmp_path / 'out')
    assert result['status'] == 'unknown'
    assert result['recovery']['attempts'][-1]['reason'] == 'measurement_event_alignment_unproved'
    source = write(tmp_path, 'phase.kmb', 'X q0\nX q0\nZ q0')
    child = write(tmp_path, 'no-phase.kmb', 'X q0\nX q0')
    result = discovery.evaluate_candidate(source, child, tmp_path / 'phase-out')
    assert result['status'] == 'rejected' and result['counterexample']['kind'] == 'phase'
