"""Artifact promises, early hash checks, and independently timed query search."""
import hashlib
import json
from pathlib import Path

import kickmix as km
import pytest

from strata.discovery import evaluate_candidate
from strata.model import StrataError, digest
from strata.store import Store


def circuit_file(tmp_path, name, text):
    path = tmp_path / name
    km.Circuit(text).to_file(path, format='kmb' if path.suffix == '.kmb' else 'kmx')
    return path


@pytest.fixture
def clean_pair(tmp_path):
    original = circuit_file(tmp_path, 'original.kmb', 'CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    candidate = circuit_file(tmp_path, 'candidate.kmx', 'CCX q0 q1 q3')
    return original, candidate


def test_clean_candidate_is_verified_under_a_hashed_restoration_contract(tmp_path, clean_pair):
    original, candidate = clean_pair
    dirty = evaluate_candidate(original, candidate, tmp_path / 'dirty')
    clean = evaluate_candidate(original, candidate, tmp_path / 'clean', clean_inputs=[2])
    assert dirty['status'] == 'rejected'
    assert clean['status'] == 'verified_improvement'
    assert clean['checks']['clean_outputs_restored'] is True
    assert clean['checks']['all_inputs_arbitrary'] is False
    assert clean['contract']['clean_inputs'] == clean['contract']['required_zero_outputs'] == [2]
    assert clean['contract_hash'] == digest(clean['contract']) != dirty['contract_hash']
    store = Store(tmp_path / 'clean' / '.strata')
    stored = store.get(clean['record'], 'file_candidate_attempt')
    assert stored['contract'] == clean['contract']
    assert hashlib.sha256(store.read_blob(stored['baseline']['blob'])).hexdigest() == clean['baseline']['sha256']


def test_clean_promise_still_checks_phase_and_dirty_public_outputs(tmp_path, clean_pair):
    original, candidate = clean_pair
    candidate.write_text(candidate.read_text() + '\nZ q0')
    report = evaluate_candidate(original, candidate, tmp_path / 'out', clean_inputs=[2])
    assert report['status'] == 'rejected' and report['counterexample']['kind'] == 'phase'
    assert report['counterexample']['inputs']['2'] == 0


def test_identical_leaking_clean_workspace_is_rejected(tmp_path):
    source = circuit_file(tmp_path, 'leak.kmx', 'CCX q0 q1 q2')
    report = evaluate_candidate(source, source, tmp_path / 'out', clean_inputs=[2])
    assert report['status'] == 'rejected'
    assert report['checks']['reason'] == 'clean_output_not_restored'
    assert report['checks']['clean_outputs_restored'] is False
    assert report['counterexample']['kind'] == 'clean_restoration'


def test_file_clean_promise_is_not_reasserted_at_an_interior_boundary(tmp_path):
    source = circuit_file(tmp_path, 'before.kmx', 'X q2\nCX q2 q3\nX q2')
    child = circuit_file(tmp_path, 'after.kmx', 'X q2\nX q2\nX q3\nX q3')
    report = evaluate_candidate(source, child, tmp_path / 'out', clean_inputs=[2])
    assert report['status'] == 'rejected'
    assert report['counterexample']['inputs']['2'] == 0
    assert report['intervals'][0]['start'] == 0
    bounded = evaluate_candidate(source, source, tmp_path / 'bounded', clean_inputs=[2], max_window=2)
    assert bounded['status'] == 'verified_no_gain' and bounded['checks']['equivalent'] is True
    restoration = bounded['checks']['clean_restoration']
    assert restoration['evidence'] == 'bounded_constant_transport_and_reset_obligations'
    assert restoration['restored'] is True and restoration['unresolved_zero_outputs'] == []
    assert restoration['unresolved_reset_legality'] is False
    assert bounded['recovery']['semantic_comparisons'] == 0


def test_clean_edit_regions_still_constrain_changes(tmp_path, clean_pair):
    report = evaluate_candidate(*clean_pair, tmp_path / 'out', clean_inputs=[2], regions=[(0, 1)])
    assert report['checks']['reason'] == 'changes_outside_requested_regions'


def test_clean_promise_does_not_authorize_register_boundary_changes(tmp_path):
    source = circuit_file(tmp_path, 'before.kmx', 'APPEND_TO_REGISTER q0 r0\nREGISTER r0 "public"\nX q1\nX q1')
    child = circuit_file(tmp_path, 'after.kmx', 'APPEND_TO_REGISTER q1 r0\nREGISTER r0 "public"\nX q1\nX q1')
    report = evaluate_candidate(source, child, tmp_path / 'out', clean_inputs=[1])
    assert report['status'] == 'unknown' and report['checks']['reason'] == 'metadata_boundary_changed'


def test_clean_and_finite_promises_are_conjoined_and_restored(tmp_path):
    source = circuit_file(tmp_path, 'leak.kmx', 'CCX q0 q1 q2')
    domain = lambda row: dict(version=1, type='finite', assignments=[row])
    good = evaluate_candidate(source, source, tmp_path / 'good', clean_inputs=[2], domain=domain([0, 0, 0]))
    assert good['status'] == 'verified_domain_no_gain'
    assert good['contract']['domain_hash'] == good['domain']['domain_hash']
    assert good['checks']['clean_outputs_restored'] is True
    leak = evaluate_candidate(source, source, tmp_path / 'leak', clean_inputs=[2], domain=domain([1, 1, 0]))
    assert leak['status'] == 'rejected' and leak['checks']['reason'] == 'clean_output_not_restored'
    with pytest.raises(StrataError, match='contradicts'):
        evaluate_candidate(source, source, tmp_path / 'bad', clean_inputs=[2], domain=domain([0, 0, 1]))


def test_clean_finite_restoration_covers_every_measurement_outcome(tmp_path):
    source = circuit_file(tmp_path, 'branch.kmx', 'HMR q0 b0\nX q1 if b0')
    report = evaluate_candidate(source, source, tmp_path / 'out', clean_inputs=[1],
                                domain=dict(version=1, type='finite', assignments=[[0, 0]]))
    assert report['status'] == 'rejected'
    assert report['counterexample']['outcomes'] == [1]
    assert report['counterexample']['kind'] == 'clean_restoration'


@pytest.mark.parametrize('clean', [[2, 2], [-1], [4], [True]])
def test_invalid_clean_boundary_indices_are_rejected(tmp_path, clean_pair, clean):
    with pytest.raises(StrataError, match='clean inputs'):
        evaluate_candidate(*clean_pair, tmp_path / 'out', clean_inputs=clean)


@pytest.mark.parametrize('encoding', ['source_bytes', 'canonical_text', None])
def test_binary_parent_hash_encodings_preserve_canonical_artifact_bindings(tmp_path, clean_pair, encoding):
    original, candidate = clean_pair
    canonical = str(km.Circuit.from_file(original)).encode()
    source_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    canonical_hash = hashlib.sha256(canonical).hexdigest()
    expected = canonical_hash if encoding == 'canonical_text' else source_hash
    hypothesis = dict(parent_sha256=expected)
    if encoding is not None:
        hypothesis['parent_hash_encoding'] = encoding
    report = evaluate_candidate(original, candidate, tmp_path / 'out', clean_inputs=[2], hypothesis=hypothesis)
    assert report['status'] == 'verified_improvement'
    assert report['baseline']['sha256'] == report['parent_sha256'] == canonical_hash
    assert report['baseline']['source_sha256'] == source_hash
    assert report['hypothesis_binding']['matched_encoding'] == (encoding or 'source_bytes')


def test_legacy_canonical_hash_still_matches_before_candidate_loading(tmp_path, clean_pair):
    original, candidate = clean_pair
    expected = hashlib.sha256(str(km.Circuit.from_file(original)).encode()).hexdigest()
    report = evaluate_candidate(original, candidate, tmp_path / 'out', clean_inputs=[2],
                                hypothesis=dict(parent_sha256=expected))
    assert report['hypothesis_binding']['matched_encoding'] == 'canonical_text'


def test_streaming_discovery_parent_can_be_copied_directly_into_hypothesis(tmp_path, clean_pair):
    from strata.discovery import discovery_brief
    original, candidate = clean_pair
    brief = discovery_brief(original, tmp_path / 'brief', scan_range=(0, 3))
    report = evaluate_candidate(original, candidate, tmp_path / 'out', clean_inputs=[2],
                                hypothesis={key: brief[key] for key in ('parent_sha256', 'parent_hash_encoding')})
    assert report['status'] == 'verified_improvement'
    assert report['hypothesis_binding']['matched_encoding'] == 'source_bytes'


def test_labelled_source_mismatch_is_streamed_before_any_native_load(tmp_path, monkeypatch):
    import strata.discovery as discovery
    source = tmp_path / 'huge.kmb'
    with source.open('wb') as output:
        output.truncate(51_000_000)
    monkeypatch.setattr(discovery, 'load_circuit', lambda *a: pytest.fail('preflight must precede native load'))
    read_bytes = Path.read_bytes
    def no_full_source_read(path):
        assert path != source
        return read_bytes(path)
    monkeypatch.setattr(Path, 'read_bytes', no_full_source_read)
    report = evaluate_candidate(source, tmp_path / 'not-even-opened.kmb', tmp_path / 'out',
                                hypothesis=dict(parent_sha256='0' * 64, parent_hash_encoding='source_bytes'))
    assert report['checks']['reason'] == 'hypothesis_parent_hash_mismatch'
    assert report['kind'] == 'file_candidate_preflight'
    assert report['baseline']['sha256'] is None and report['candidate']['source_sha256'] is None
    assert report['cost_delta'] is None and report['accepted_gain'] is None
    store = Store(tmp_path / 'out' / '.strata')
    assert store.get(report['record'], 'file_candidate_preflight')['hypothesis_binding']['status'] == 'mismatch'


def test_source_binding_is_content_not_path(tmp_path, clean_pair, monkeypatch):
    import strata.discovery as discovery
    original, candidate = clean_pair
    expected = hashlib.sha256(original.read_bytes()).hexdigest()
    original.write_bytes(candidate.read_bytes())
    monkeypatch.setattr(discovery, 'load_circuit', lambda *a: pytest.fail('changed source must fail preflight'))
    report = evaluate_candidate(original, candidate, tmp_path / 'out',
                                hypothesis=dict(source_sha256=expected))
    assert report['checks']['reason'] == 'hypothesis_parent_hash_mismatch'


@pytest.mark.parametrize('query', ['function', 'slack'])
def test_query_search_timeout_excludes_slow_window_semantics(monkeypatch, query):
    import strata.inference as inference
    clock = [0.0]
    original = inference.functions_at
    def slow_semantics(*args, **kwargs):
        clock[0] += 36
        return original(*args, **kwargs)
    monkeypatch.setattr(inference, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(inference, 'functions_at', slow_semantics)
    circuit = km.Circuit('Z_POW q0 0.25\nCCX q0 q1 q2')
    if query == 'function':
        result = inference.is_function_of(circuit, 2, 2, [0, 1], from_at=1, clean_inputs=[2], timeout=5)
        assert result['status'] == 'yes' and result['form']['monomials'] == [[0, 1]]
    else:
        result = inference.information_slack(circuit, 2, from_at=1, clean_inputs=[2], timeout=5)
        assert result['status'] == 'exact' and result['slack_bits'] == 1
    assert result['timings'] == dict(window_semantics_seconds=36, search_seconds=0)


def test_phase_constructor_preserves_unique_boundary_metadata_when_reapplied():
    from strata.inference import eliminate_clean_phase_workspace
    source = km.Circuit('APPEND_TO_REGISTER q0 r0\nREGISTER r0 "strata_boundary"\n'
                        'CCX q0 q1 q2\nZ q2\nCCX q0 q1 q2')
    first = eliminate_clean_phase_workspace(source, clean_inputs=[2])
    assert first['status'] == 'constructed'
    child = km.Circuit(first['candidate'])
    assert set(child.register_data) == {'strata_boundary', 'strata_boundary_1'}
    second = eliminate_clean_phase_workspace(child, clean_inputs=[2])
    assert second['status'] == 'constructed' and second['candidate'] == first['candidate']
    assert second['independent_replay']['status'] == 'verified'


def test_search_expiration_and_semantic_limit_have_distinct_phases(monkeypatch):
    import strata.inference as inference
    circuit = km.Circuit('CCX q0 q1 q2')
    ticks = iter(range(1000))
    monkeypatch.setattr(inference, 'monotonic', lambda: next(ticks))
    search = inference.is_function_of(circuit, 2, 1, [0, 1], clean_inputs=[2], timeout=0.1)
    assert search['status'] == 'unknown' and search['budget_phase'] == 'functional_search'
    assert search['reason'] == 'functional search timeout'
    semantics = inference.is_function_of(circuit, 2, 1, [0, 1], max_variables=1)
    assert semantics['status'] == 'unknown' and semantics['budget_phase'] == 'window_semantics'
    assert semantics['timings']['search_seconds'] == 0


def test_cli_exposes_clean_and_query_search_budgets(tmp_path, clean_pair, capsys):
    from strata.cli import main
    original, candidate = clean_pair
    assert main(['evaluate-file', str(original), str(candidate), '--clean-inputs', '2', '--out', str(tmp_path / 'out')]) == 0
    assert json.loads(capsys.readouterr().out)['checks']['clean_outputs_restored'] is True
    assert main(['function-of', str(original), '--wire', '2', '--support', '0,1', '--at', '1',
                 '--clean-inputs', '2', '--timeout', '10', '--degree', '2', '--max-forms', '4']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'yes' and result['search_limits']['timeout_seconds'] == 10
    assert {'load_seconds', 'window_semantics_seconds', 'search_seconds', 'query_seconds'} <= result['timings'].keys()
    assert main(['slack', str(original), '--at', '1', '--timeout', '10', '--max-inputs', '1']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'unknown' and result['reason'] == 'distinct-state enumeration budget'


@pytest.mark.parametrize('command', ['discover', 'inspect-file'])
@pytest.mark.parametrize('arguments,cap', [([], 8192), (['--max-window', '32'], 32),
                                         (['--max-window', '32', '--growth-window', '64'], 64)])
def test_cli_streaming_window_precedence(tmp_path, monkeypatch, capsys, command, arguments, cap):
    import strata.discovery as discovery
    import strata.file_workflow as workflow
    from strata.cli import main
    def capture(*args, **kwargs):
        assert kwargs['growth_window'] == cap
        return dict(cap=cap)
    monkeypatch.setattr(discovery, 'discovery_brief', capture)
    monkeypatch.setattr(workflow, 'inspect_file', capture)
    assert main([command, 'unused.kmb', '--out', str(tmp_path / 'out'), *arguments]) == 0
    assert json.loads(capsys.readouterr().out)['cap'] == cap
