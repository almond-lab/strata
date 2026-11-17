import json
from pathlib import Path

import kickmix as km
import pytest

from strata.file_profile import profile_regions
from strata.kickmix import Kickmix
from strata.semantic import lift_regions


def fixture_circuit():
    # Register-free, separated native episodes.  The larger paid region is last.
    return km.Circuit('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2\n'
                      'INFO_TAG "untrusted separator"\n'
                      'CCX q4 q5 q8\nCCX q6 q7 q8\nCX q8 q9\n'
                      'CCX q6 q7 q8\nCCX q4 q5 q8')


def test_ranking_uses_global_objective_not_region_wire_index(tmp_path):
    circuit = fixture_circuit()
    lift = lift_regions(circuit)
    report = profile_regions(circuit, lift, tmp_path)
    assert report['candidate_order'] == [lift['regions'][1]['id'], lift['regions'][0]['id']]
    big, small = report['ranked_regions']
    assert big['static_resources']['toffoli'] == 4
    assert small['static_resources']['toffoli'] == 2
    assert big['objective_removal_ceiling'] == 40
    assert small['objective_removal_ceiling'] == 20
    assert small['removal_projection']['global_after']['Q'] == 10
    assert small['boundary_width'] == 4
    assert big['materialization_count'] == small['materialization_count'] == 1
    assert big['repeated_write_count'] == 3
    assert big['floor'] is None and big['achievable_saving'] is None
    assert big['search'][0]['mechanism'] == 'remove_restored_materializations'
    assert report['unknowns']['high_level_stage_coverage'] is None


def test_actual_native_views_and_bounded_region_diagrams_are_written(tmp_path):
    circuit = fixture_circuit()
    report = profile_regions(circuit, lift_regions(circuit), tmp_path, max_graphs=1)
    assert set(report['views']) >= {'native_flame', 'native_diagram', 'native_analysis', 'profile', 'index'}
    assert all(Path(path).is_file() for path in report['views'].values())
    diagram = report['ranked_regions'][0]['diagram']
    assert Path(diagram['path']).is_file() and diagram['coverage'] == 'full_region'
    assert 'diagram' not in report['ranked_regions'][1]
    assert json.loads((tmp_path / 'profile.json').read_text()) == report
    assert report['native_analysis']['reaction_depth'] == Kickmix().measured(circuit)['reaction_depth']


def test_custom_objective_is_evaluated_on_full_global_resources(tmp_path):
    circuit = fixture_circuit()
    report = profile_regions(circuit, lift_regions(circuit), tmp_path,
                             formula='Q*toffoli*toffoli', max_graphs=0)
    assert report['baseline']['cost'] == 360
    big, small = report['ranked_regions']
    assert big['objective_removal_ceiling'] == 360 - 10*2*2
    assert small['objective_removal_ceiling'] == 360 - 10*4*4


def test_nonmonotone_or_nonadditive_objective_does_not_get_fake_ceiling(tmp_path):
    circuit = fixture_circuit()
    report = profile_regions(circuit, lift_regions(circuit), tmp_path,
                             formula='100-toffoli', max_graphs=0)
    assert not report['objective']['monotone_additive_ceiling_supported']
    assert all(row['objective_removal_ceiling'] is None for row in report['ranked_regions'])
    assert report['ranked_regions'][0]['removal_projection']['hypothetical_delta'] == -4
    other = profile_regions(circuit, lift_regions(circuit), tmp_path / 'depth',
                            formula='reaction_depth+toffoli', max_graphs=0)
    assert all(row['objective_removal_ceiling'] is None for row in other['ranked_regions'])


def test_opaque_phase_context_and_wire_signals_remain_uncertain(tmp_path):
    circuit = km.Circuit('CCX q0 q1 q2\nCZ q2 q3\nCCX q0 q1 q2\nZ_POW q2 0.25')
    report = profile_regions(circuit, lift_regions(circuit), tmp_path, max_graphs=0)
    row = report['ranked_regions'][0]
    assert row['phase_dependency']['term_count'] > 0
    assert row['phase_dependency']['within_region'] == 'exact_sign_polynomial'
    assert row['phase_dependency']['cross_boundary'] == 'unresolved_opaque_context'
    assert row['phase_dependency']['adjacent_opaque_barriers'][0]['start'] == 3
    assert row['wire_signal']['native_reported_peak'] == 4
    assert 'neither exact liveness nor a clean-lease proof' in row['wire_signal']['meaning']
    assert report['unknowns']['clean_lease_proof'] is None
    assert report['unknowns']['opaque_or_bounded_regions'] == 1


def test_native_analysis_prefix_is_explicit_and_views_exist_only_when_generated(tmp_path, monkeypatch):
    circuit = fixture_circuit()
    original = profile_regions(circuit, lift_regions(circuit), tmp_path, max_graphs=0)
    assert 'native_flame' in original['views']
    monkeypatch.setattr('strata.file_profile.NATIVE_ANALYSIS_MAX_OPS', 3)
    report = profile_regions(circuit, lift_regions(circuit), tmp_path, max_graphs=0)
    assert report['native_analysis']['coverage'] == 'prefix'
    assert report['native_analysis']['scanned'] == 3
    assert report['native_analysis']['flame_status'] == 'skipped_size_budget'
    assert 'native_flame' not in report['views']
    assert 'flame.svg' not in Path(report['views']['native_index']).read_text()
    assert not (tmp_path / 'native' / 'flame.svg').exists()
    assert report['semantic_coverage']['scope'] == 'full'
    assert report['ranked_regions'][0]['wire_signal']['scope'] == 'prefix_or_unscanned'


def test_primitives_do_not_become_ranked_deep_searches(tmp_path):
    circuit = km.Circuit('CCX q0 q1 q2')
    report = profile_regions(circuit, lift_regions(circuit), tmp_path)
    assert report['candidate_order'] == []
    assert report['excluded_primitive_regions'] == 1


def test_graph_budget_rejected_before_any_output(tmp_path):
    circuit = fixture_circuit()
    with pytest.raises(ValueError):
        profile_regions(circuit, lift_regions(circuit), tmp_path, max_graphs=-1)
    assert list(tmp_path.iterdir()) == []
