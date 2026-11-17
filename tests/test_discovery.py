"""A connected AI can bring novel files and learn from retained evidence."""
import json
import hashlib
from pathlib import Path

import kickmix as km
import numpy as np
import pytest

from strata.discovery import discovery_brief, evaluate_candidate, record_research_note
from strata.model import StrataError
from strata.file_optimizer import optimize_circuit
from strata.store import Store


def write_circuit(tmp_path, name, text):
    path = tmp_path / name
    km.Circuit(text).to_file(path, format='kmb' if path.suffix == '.kmb' else 'kmx')
    return path


def native(circuit, assignment):
    sim = km.Simulator(batch_size=1, seed=0)
    wires = [km.q(q) for q in range(circuit.num_qubits)]
    sim.write_across_shots(wires, np.array([[bool(assignment.get(str(q), 0))]
                                          for q in range(circuit.num_qubits)]))
    sim.do(circuit)
    return sim.read_across_shots(wires).tolist(), sim.read_shot_phase(0)


@pytest.fixture
def materialization(tmp_path):
    return write_circuit(tmp_path, 'unlabeled.kmb', 'CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')


def test_two_round_feedback_contains_failure_then_verified_new_parent(tmp_path, materialization):
    out = tmp_path / 'discovery'
    bad = write_circuit(tmp_path, 'wrong.kmx', 'CCX q0 q1 q3')
    failed = evaluate_candidate(materialization, bad, out,
                                hypothesis={'mechanism': 'direct consumer', 'rationale': 'avoid storage'})
    assert failed['status'] == 'rejected'
    assert failed['cost_delta'] < 0 and failed['accepted_gain'] is None
    witness = failed['counterexample']
    assert native(km.Circuit.from_file(materialization), witness['inputs']) != native(km.Circuit.from_file(bad), witness['inputs'])
    brief = discovery_brief(materialization, out)
    assert brief['history_total'] == 1
    feedback = brief['history'][0]['feedback']
    assert feedback['counterexample'] == witness
    assert 'construction' in feedback['inference_scope']
    assert feedback['next_test'] and feedback['reopening_condition']
    assert feedback['task_domain_counterexample_established'] is False
    expanded = json.loads(open(brief['targets'][0]['expand']['path']).read())
    assert expanded['history'][0]['feedback']['check_status'] == 'rejected'
    assert expanded['region']['semantics']['outputs']

    good = write_circuit(tmp_path, 'new-candidate.kmx', 'CX q2 q3\nCCX q0 q1 q3')
    passed = evaluate_candidate(materialization, good, out,
                                hypothesis={'parent_sha256': brief['parent_sha256'],
                                            'mechanism': 'preserve borrowed input while forwarding product'})
    assert passed['status'] == 'verified_improvement'
    assert passed['accepted_gain'] == 4
    assert passed['feedback']['resource_delta']['toffoli'] == -1
    assert passed['checks']['equivalent'] is True
    followup = discovery_brief(good, out)
    assert followup['parent_sha256'] == passed['candidate']['sha256']
    assert followup['history_total'] == 1
    assert followup['history'][0]['relationship'] == 'verified_parent_of_current'
    assert followup['history'][0]['feedback']['check_status'] == 'verified_improvement'
    assert 'connected AI' in followup['agent_assignment']['executor']


def test_evaluator_accepts_construction_outside_builtin_optimizer_family(tmp_path):
    original = write_circuit(tmp_path, 'source.kmx', 'CX q0 q1\nCX q1 q0\nCX q0 q1')
    candidate = write_circuit(tmp_path, 'invented.kmb', 'SWAP q0 q1')
    _, built_in = optimize_circuit(km.Circuit.from_file(original), 'instructions')
    assert built_in['status'] == 'unchanged'
    report = evaluate_candidate(original, candidate, tmp_path / 'out', cost='instructions',
                                 hypothesis={'mechanism': 'permutation basis change'})
    assert report['status'] == 'verified_improvement'
    assert report['baseline']['cost'] == 3 and report['candidate']['cost'] == 1
    assert report['structural_depth'] == 'not_established_by_equivalence'
    assert report['intervals'][0]['function_hash']


def test_widening_merges_adjacent_inferred_functions(tmp_path):
    original = write_circuit(tmp_path, 'source.kmx',
        'CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2\nCCX q0 q1 q4\nCX q4 q5\nCCX q0 q1 q4')
    candidate = write_circuit(tmp_path, 'candidate.kmx',
        'CCX q0 q1 q3\nCX q2 q3\nCCX q0 q1 q5\nCX q4 q5')
    report = evaluate_candidate(original, candidate, tmp_path / 'out')
    assert report['status'] == 'verified_improvement'
    assert len(report['intervals']) == 1
    assert (report['intervals'][0]['start'], report['intervals'][0]['stop']) == (0, 6)


@pytest.mark.parametrize('extra', ['', '\nX q0\nX q0'])
def test_identical_or_nonimproving_candidate_is_retained(tmp_path, materialization, extra):
    text = str(km.Circuit.from_file(materialization)) + extra
    candidate = write_circuit(tmp_path, 'candidate.kmx', text)
    report = evaluate_candidate(materialization, candidate, tmp_path / 'out')
    assert report['status'] == 'verified_no_gain'
    stored = Store(tmp_path / 'out' / '.strata').get(report['record'], 'file_candidate_attempt')
    assert stored['checks']['equivalent'] is True
    assert 'no measured improvement' in stored['feedback']['next_test']
    assert stored['feedback']['inference_scope'].endswith('no mechanism family is closed')


def test_phase_only_failure_has_native_counterexample(tmp_path, materialization):
    candidate = write_circuit(tmp_path, 'candidate.kmx',
                              str(km.Circuit.from_file(materialization)) + '\nZ q0')
    report = evaluate_candidate(materialization, candidate, tmp_path / 'out')
    assert report['status'] == 'rejected'
    witness = report['feedback']['counterexample']
    before = native(km.Circuit.from_file(materialization), witness['inputs'])
    after = native(km.Circuit.from_file(candidate), witness['inputs'])
    assert before[0] == after[0] and before[1] != after[1]
    assert witness['kind'] == 'phase'


@pytest.mark.parametrize('barrier', ['HMR q2 b0', 'Z_POW q2 0.25'])
def test_changed_opaque_or_conditional_spans_stay_unknown(tmp_path, barrier):
    original = write_circuit(tmp_path, 'source.kmx', f'CCX q0 q1 q2\n{barrier}\nCX q2 q3')
    candidate = write_circuit(tmp_path, 'candidate.kmx', 'CCX q0 q1 q2\nCX q2 q3')
    report = evaluate_candidate(original, candidate, tmp_path / 'out')
    assert report['status'] == 'unknown'
    assert report['checks']['equivalent'] is None
    assert report['feedback']['unresolved_obligations']
    assert len(Store(tmp_path / 'out' / '.strata').all('file_candidate_attempt')) == 1


def test_small_proof_budget_is_unknown_not_sampled_success(tmp_path, materialization):
    candidate = write_circuit(tmp_path, 'candidate.kmx', 'CX q2 q3\nCCX q0 q1 q3')
    report = evaluate_candidate(materialization, candidate, tmp_path / 'out', max_window=2)
    assert report['status'] == 'unknown'
    assert report['checks']['equivalent'] is None
    assert 'budget' in report['checks']['reason']


def test_parent_binding_and_requested_edit_boundaries(tmp_path, materialization):
    candidate = write_circuit(tmp_path, 'candidate.kmx', 'CX q2 q3\nCCX q0 q1 q3')
    report = evaluate_candidate(materialization, candidate, tmp_path / 'out',
                                 hypothesis={'parent_sha256': '0' * 64})
    assert report['checks']['reason'] == 'hypothesis_parent_hash_mismatch'
    report = evaluate_candidate(materialization, candidate, tmp_path / 'out', regions=[(0, 1)])
    assert report['checks']['reason'] == 'changes_outside_requested_regions'
    assert report['checks']['equivalent'] is None


def test_brief_has_disjoint_expandable_observations_not_preselected_answers(tmp_path):
    original = write_circuit(tmp_path, 'source.kmx',
        'CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2\nCCX q0 q1 q4\nCX q4 q5\nCCX q0 q1 q4')
    brief = discovery_brief(original, tmp_path / 'out', max_targets=2)
    assert len(brief['targets']) == 2
    ranges = sorted((r['start'], r['stop']) for r in brief['targets'])
    assert ranges[0][1] <= ranges[1][0]
    for target in brief['targets']:
        assert target['function_hash'] and target['static_resources']['toffoli'] == 2
        assert target['achievable_saving'] is None
        assert 'implementation' not in target
    assert brief['unknowns']['high_level_stage_coverage'] is None
    assert 'without waiting for user-supplied proposals' in brief['agent_assignment']['task']
    assert brief['manifest'].endswith('/agent-brief.json')
    assert 'data only' in brief['artifact_text_policy']


def test_brief_ingests_automatic_search_deferrals_and_skip_reasons(tmp_path, materialization):
    from strata.file_workflow import optimize_file

    out = tmp_path / 'out'
    optimize_file(materialization, out, max_candidates=0)
    brief = discovery_brief(materialization, out)
    assert brief['history_total'] == 1
    automatic = brief['history'][0]
    assert automatic['kind'] == 'file_optimization'
    assert automatic['feedback']['skip_reason_counts']['candidate_budget'] == 1
    assert automatic['feedback']['search_budget']['deferred'] == 1
    assert 'remain open' in automatic['feedback']['inference_scope']
    assert brief['targets'][0]['history_count'] == 1


def test_cli_discover_evaluate_discover_loop_without_project(tmp_path, materialization, capsys):
    from strata.cli import main

    out = tmp_path / 'out'
    common = ['--project', str(tmp_path / 'missing.toml')]
    assert main(common + ['discover', str(materialization), '--out', str(out)]) == 0
    first = json.loads(capsys.readouterr().out)
    wrong = write_circuit(tmp_path, 'wrong.kmx', 'CCX q0 q1 q3')
    hypothesis = tmp_path / 'hypothesis.json'
    hypothesis.write_text(json.dumps({'parent_sha256': first['parent_sha256'], 'mechanism': 'forward product'}))
    assert main(common + ['evaluate-file', str(materialization), str(wrong), '--out', str(out),
                          '--hypothesis', str(hypothesis)]) == 0
    failed = json.loads(capsys.readouterr().out)
    assert failed['status'] == 'rejected' and failed['accepted_gain'] is None
    assert main(common + ['discover', str(materialization), '--out', str(out)]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second['history'][0]['feedback']['counterexample']
    good = write_circuit(tmp_path, 'good.kmx', 'CX q2 q3\nCCX q0 q1 q3')
    assert main(common + ['evaluate-file', str(materialization), str(good), '--out', str(out),
                          '--hypothesis', str(hypothesis)]) == 0
    improved = json.loads(capsys.readouterr().out)
    assert improved['status'] == 'verified_improvement'
    assert main(common + ['discover', str(good), '--out', str(out)]) == 0
    third = json.loads(capsys.readouterr().out)
    assert third['parent_sha256'] == improved['candidate']['sha256']
    assert third['history'][0]['relationship'] == 'verified_parent_of_current'


def test_unparseable_candidate_remains_actionable_retained_evidence(tmp_path, materialization):
    candidate = tmp_path / 'malformed.kmx'
    candidate.write_text('THIS_IS_NOT_A_NATIVE_GATE q0')
    out = tmp_path / 'out'
    report = evaluate_candidate(materialization, candidate, out,
                                 hypothesis={'mechanism': 'experimental construction'})
    assert report['status'] == 'unknown'
    assert report['checks']['reason'] == 'candidate_parse_failed'
    assert report['accepted_gain'] is None and report['candidate']['resources'] is None
    assert 'syntax' in report['feedback']['next_test']
    store = Store(out / '.strata')
    assert store.read_blob(report['candidate']['blob']) == candidate.read_bytes()
    brief = discovery_brief(materialization, out)
    assert brief['history'][0]['feedback']['check_reason'] == 'candidate_parse_failed'


def test_fragmented_analysis_does_not_invent_recovery_targets_across_opaque_cuts(tmp_path):
    source = write_circuit(tmp_path, 'fragmented.kmx',
                           '\n'.join(['CCX q0 q1 q2', 'Z_POW q2 0.5'] * 24))
    brief = discovery_brief(source, tmp_path / 'out')
    granularity = brief['analysis_granularity']
    assert granularity['total_static_paid_units'] == 24
    assert granularity['largest_inferred_region_paid_units'] == 1
    assert granularity['largest_inferred_region_share'] == pytest.approx(1/24)
    assert granularity['recovery_needed'] is True
    # Every actual run is below the 5% target threshold. Fixed arithmetic
    # chunks would cross unsupported phase boundaries and invent a larger unit.
    assert brief['recovery_targets'] == []


def test_prebuild_notes_feed_next_brief_without_becoming_proofs_or_gains(tmp_path, materialization):
    evidence = tmp_path / 'bounded-analysis.json'
    evidence.write_text(json.dumps({'status': 'bounded', 'reason': 'term_budget'}))
    note = dict(note_class='prebuild_blocker', hypotheses=[{'mechanism': 'broader boundary'}],
                blockers=['Term budget prevents recovering the full episode.'],
                evidence_refs=[dict(path=str(evidence), sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())],
                status='verified_improvement', equivalent=True, accepted_gain=999, qualified_gain=True)
    out = tmp_path / 'out'
    recorded = record_research_note(materialization, out, note)
    assert recorded['authority'] == 'reported_by_connected_agent'
    assert recorded['evidence_kind'] == 'agent_report_not_proof'
    assert recorded['status'] == 'reported'
    assert recorded['equivalent'] is None and recorded['accepted_gain'] is None
    assert recorded['qualified_gain'] is False
    assert recorded['evidence_refs'][0]['content_hash_checked'] is True
    note['blockers'].clear()
    brief = discovery_brief(materialization, out)
    assert brief['history_total'] == brief['research_note_total'] == 1
    assert brief['history'][0]['kind'] == 'agent_research_note'
    assert brief['research_notes'][0]['feedback']['reported_blockers']
    assert brief['research_notes'][0]['qualified_gain'] is False
    assert brief['task_context']['placement_at_host_peak'] == 'not_provided'
    assert brief['task_context']['host_cost_model'] == 'not_provided'


@pytest.mark.parametrize('note', [dict(note_class='proof'), dict(note_class=['observation']),
                                 dict(parent_sha256='0' * 64)])
def test_note_classes_and_parent_binding_cannot_fabricate_authority(tmp_path, materialization, note):
    with pytest.raises(StrataError):
        record_research_note(materialization, tmp_path / 'out', note)


def test_attempt_code_and_research_evidence_are_content_addressed_snapshots(tmp_path, materialization):
    candidate = write_circuit(tmp_path, 'candidate.kmx', 'CX q2 q3\nCCX q0 q1 q3')
    out = tmp_path / 'out'
    report = evaluate_candidate(materialization, candidate, out)
    store = Store(out / '.strata')
    package = Path(__file__).parents[1] / 'src' / 'strata'
    for name, sha in report['code'].items():
        assert store.read_blob(sha) == (package / name).read_bytes()
    evidence = tmp_path / 'evidence.txt'
    evidence.write_text('unverified analysis note')
    with pytest.raises(StrataError, match='hash mismatch'):
        record_research_note(materialization, out,
                             dict(evidence_refs=[dict(path=str(evidence), sha256='0' * 64)]))


def test_cli_note_retains_a_prebuild_blocker_for_next_discovery(tmp_path, materialization, capsys):
    from strata.cli import main

    note = tmp_path / 'note.json'
    note.write_text(json.dumps(dict(note_class='analysis_request',
                                    blockers=['Need broader exact boundary before constructing a candidate.'])))
    out = tmp_path / 'out'
    assert main(['note', str(materialization), '--out', str(out), '--note', str(note)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['authority'] == 'reported_by_connected_agent'
    assert result['qualified_gain'] is False
    assert main(['discover', str(materialization), '--out', str(out)]) == 0
    brief = json.loads(capsys.readouterr().out)
    assert brief['research_note_total'] == 1
    assert brief['research_notes'][0]['record'] == result['record']


def test_changed_scoped_condition_is_rejected_with_classical_counterexample(tmp_path):
    original = write_circuit(tmp_path, 'source.kmx',
                             'CCX q0 q1 q2\nPUSH_CONDITION if b0\nX q2\nPOP_CONDITION\nCX q2 q3')
    candidate = write_circuit(tmp_path, 'candidate.kmx', 'CCX q0 q1 q2\nCX q2 q3')
    report = evaluate_candidate(original, candidate, tmp_path / 'out')
    assert report['status'] == 'rejected'
    assert report['checks']['equivalent'] is False
    assert report['feedback']['counterexample']['classical_inputs'] == {'b0': 1}


def test_repeated_hmr_span_is_lifted_with_fresh_outcomes(tmp_path):
    source = write_circuit(tmp_path, 'measurements.kmx',
                           '\n'.join(['CCX q0 q1 q2', 'HMR q2 b0'] * 24))
    brief = discovery_brief(source, tmp_path / 'out')
    granularity = brief['analysis_granularity']
    assert granularity['largest_inferred_region_paid_units'] == 24
    assert granularity['largest_inferred_region_share'] == 1
