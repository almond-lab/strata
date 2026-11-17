from dataclasses import replace
import json

import pytest

from strata.discovery import discovery_brief, evaluate_candidate, record_research_note
from strata.model import StrataError
from strata.project import Project
from strata.reports import markdown_report, report
from strata.research import ResearchPolicy, ResearchRound
from strata.store import Store
from test_research_review import RecordedFusion, make_project, question


class MixedResults(RecordedFusion):
    def build(self, parameters, implementation):
        if implementation.get('bad'):
            raise StrataError('construction cannot satisfy its return contract')
        return super().build(parameters, implementation)


def test_report_from_saved_round_mixed_success_failure_unbuilt_and_unknown_gaps(tmp_path, monkeypatch):
    project = make_project(tmp_path, MixedResults())
    base = project.task('and').base_design()
    round_ = ResearchRound(project, base, ResearchPolicy(max_lines=6))
    plan = round_.plan([question('successful', implementation={'fused': True}),
                       question('failed-one', implementation={'bad': 1}),
                       question('failed-two', implementation={'bad': 2}),
                       question('derivation'), question('nonstructural', implementation={})])
    result = round_.run(plan['handle'])
    assert result['qualified_gains'] == 1
    saved_gaps = [{'region': 'root', 'floor_kind': 'proof', 'weighted_gap': 4}]
    project.store.put('gap_snapshot', {'design': base.id, 'gaps': saved_gaps})
    record_count = project.store.db.execute('SELECT COUNT(*) FROM records').fetchone()[0]

    def forbidden(*args, **kwargs):
        raise AssertionError('report executed live project code')

    monkeypatch.setattr(MixedResults, 'build', forbidden)
    monkeypatch.setattr(Project, '__init__', forbidden)
    # The report remains usable after the config and input circuit disappear.
    project.path.unlink()
    (project.root / 'base.kmx').unlink()
    store = Store(project.store.path)
    data = report(store, round=plan['handle'])
    assert data['counts']['plans'] == data['counts']['results'] == 1
    assert data['counts']['lines'] == 4
    assert data['counts']['designs'] == 2
    assert data['counts']['outcome_statuses']['DERIVATION_REQUIRED'] == 1
    success = next(v for v in data['lines'] if v['mechanism'] == 'successful')
    assert success['measured_delta'] == -45
    failed = [v for v in data['lines'] if v['mechanism'].startswith('failed')]
    assert all(v['measured_delta'] is None for v in failed)
    assert len(data['shared_failures']) == 1 and data['shared_failures'][0]['count'] == 2
    gap_by_design = {v['design']: v for v in data['gaps']}
    assert gap_by_design[base.id]['gaps'] == saved_gaps
    child = next(v for v in data['designs'] if v['parent'])
    assert gap_by_design[child['handle']]['status'] == 'unknown'
    assert gap_by_design[child['handle']]['gaps'] is None
    markdown = markdown_report(data)
    for expected in ('DERIVATION_REQUIRED', 'ATTEMPT_FAILED', 'VERIFIED_STRUCTURAL_GAIN', '-45', 'unknown', 'construction cannot satisfy', 'nonstructural'):
        assert expected in markdown
    assert json.loads(json.dumps(data)) == data
    assert report(store, handle=result['handle'])['counts'] == data['counts']
    child_report = report(store, design=child['handle'])
    assert any(v['handle'] == success['handle'] for v in child_report['lines'])
    assert child_report['results'][0]['handle'] == result['handle']
    assert store.db.execute('SELECT COUNT(*) FROM records').fetchone()[0] == record_count


def test_file_only_report_includes_failed_successful_and_prebuild_notes_without_source_files(tmp_path, monkeypatch):
    original = tmp_path / 'original.kmx'
    original.write_text('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    bad = tmp_path / 'bad.kmx'
    bad.write_text('CCX q0 q1 q3')
    good = tmp_path / 'good.kmx'
    good.write_text('CX q2 q3\nCCX q0 q1 q3')
    out = tmp_path / 'research'
    failure = evaluate_candidate(original, bad, out, hypothesis={'mechanism': 'direct'})
    success = evaluate_candidate(original, good, out, hypothesis={'mechanism': 'preserved borrowed input'})
    note = record_research_note(original, out, {'note_class': 'prebuild_blocker', 'hypotheses': ['new encoder'],
                                              'blockers': ['representation semantics unknown'], 'next_tests': ['derive return']})
    discovery_brief(original, out)
    for source in (original, good, bad): source.unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError('report attempted circuit execution')
    monkeypatch.setattr('strata.kickmix.Kickmix.load', forbidden)
    data = report(Store(out / '.strata'), handle=note['record'])
    assert data['counts']['file_attempts'] == 2
    assert data['counts']['notes'] == 1
    assert data['counts']['designs'] == 0
    assert len(data['discovery']) >= 1
    by_handle = {v['handle']: v for v in data['file_attempts']}
    assert by_handle[failure['record']]['accepted_gain'] is None
    assert by_handle[failure['record']]['checks']['equivalent'] is False
    assert by_handle[success['record']]['accepted_gain'] == 4
    assert data['notes'][0]['qualified_gain'] is False
    markdown = markdown_report(data)
    for expected in ('prebuild_blocker', 'representation semantics unknown', 'rejected', 'verified_improvement', 'agent_report_not_proof'):
        assert expected in markdown


def test_reports_downgrade_missing_negative_artifacts_without_changing_history(tmp_path):
    store = Store(tmp_path)
    artifact = store.blob(b'proof evidence')
    negative = store.negative('fp', 'fuse', 'proof', 'no result', 'changed premise', evidence=artifact)
    assert report(store)['negative_memory'][0]['status'] == 'closed'
    (store.path / 'objects' / artifact).unlink()
    data = report(store)
    assert data['negative_memory'][0]['status'] == 'priced'
    assert data['negative_memory'][0]['requested_evidence_kind'] == 'proof'
    assert data['negative_memory'][0]['evidence_status'] == 'missing_or_invalid_artifact'
    assert store.get(negative)['status'] == 'closed'


def test_report_selector_rejects_unknown_or_conflicting_handles(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(StrataError): report(store, handle='absent')
    with pytest.raises(StrataError): report(store, design='a', round='b')
    with pytest.raises(StrataError): report(store, round=store.put('design', {'parent': None}))
    assert report(store)['gaps'][0]['status'] == 'unknown'


@pytest.mark.parametrize('parent_field', ['parent_sha256', 'input_sha256', 'original_sha256', 'input', 'original'])
def test_file_parent_aliases_include_automatic_runs_and_render_skips(tmp_path, parent_field):
    store = Store(tmp_path)
    parent = 'a' * 64
    run = store.put('file_optimization', dict(
        baseline={'sha256': parent, 'cost': 32}, candidate={'sha256': 'b' * 64, 'cost': 16},
        status='improved', objective={'accounting': 'static_emitted'},
        score_status='static file objective only', checks={'equivalent': True}, replay={'equivalent': True},
        regions=[dict(id='changed', start=0, stop=3, status='accepted', mechanism='resynthesize_output_functions_and_phase',
                      explanation='Preserved dirty input while removing intermediate writes',
                      before={'cost': 32}, after={'cost': 16}, check={'equivalent': True}),
                 dict(id='skipped', start=3, stop=4, status='skipped', reason='no_internal_materialization_removed')]))
    note = store.put('agent_research_note', {parent_field: parent, 'note_class': 'observation', 'reported_note': {'note': 'same source'}})
    store.put('file_optimization', {'baseline': {'sha256': 'c' * 64}, 'status': 'unchanged'})
    for selector in (run, note, parent):
        saved = report(store, handle=selector)
        assert saved['counts']['file_optimizations'] == 1
        assert saved['notes'][0]['handle'] == note
        text = markdown_report(saved)
        for expected in ('32', '16', 'static file objective only', 'resynthesize_output_functions_and_phase',
                         'Preserved dirty input', 'no_internal_materialization_removed', 'skipped'):
            assert expected in text
