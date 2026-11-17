import json
from pathlib import Path

import kickmix as km

from strata.restructure import ConstructorRegistry, restructure_file
from strata.semantic import equivalent_windows


def test_episode_at_native_operation_budget_keeps_parent_layout_metadata(tmp_path):
    source = tmp_path / 'source.kmx'
    source.write_text('CCX q0 q1 q4\nCX q4 q2\nCCX q0 q1 q4\nX q10')
    result = restructure_file(source, tmp_path / 'out', scan_range=(0, 3), clean_inputs=[4],
                              max_operations=3, rounds=1)
    assert result['status'] == 'verified_improvement', result
    assert result['best']['resources']['Q'] == 11
    assert result['cost_delta'] == -11


def test_malicious_interior_clean_assumption_is_rejected(tmp_path):
    source = tmp_path / 'source.kmx'
    source.write_text('X q4\nCCX q0 q1 q4\nCX q4 q2\nCCX q0 q1 q4\nX q5')
    registry = ConstructorRegistry()
    registry.register('unsafe_clean_elimination', lambda circuit, context:
                      dict(status='constructed', candidate='CCX q0 q1 q2\nX q5\nX q5'),
                      'Adversarial candidate wrongly assumes q4 is zero after the prefix')
    result = restructure_file(source, tmp_path / 'out', registry=registry, scan_range=(1, 4),
                              clean_inputs=[4], max_operations=8, rounds=1)
    assert result['status'] == 'no_gain'
    assert 4 not in result['episode']['boundary_zero_evidence']
    assert result['construction']['attempts'][0]['check']['equivalent'] is False
    assert (tmp_path / 'out' / 'best.kmx').read_text() == str(km.Circuit.from_file(source))


def test_hmr_zero_lease_does_not_retire_an_arbitrary_parent_input(tmp_path):
    source = tmp_path / 'source.kmx'
    original = 'HMR q4 b0\nCCX q0 q1 q4\nCX q4 q2\nCCX q0 q1 q4'
    source.write_text(original)
    result = restructure_file(source, tmp_path / 'out', scan_range=(1, 4), max_operations=8, rounds=1)
    assert result['status'] == 'verified_improvement'
    assert result['best']['resources']['Q'] == result['baseline']['resources']['Q'] == 5
    assert result['contextual_proof']['retired_clean_parent_wires'] == []
    candidate = (tmp_path / 'out' / 'best.kmx').read_text()
    assert candidate.startswith('HMR q4 b0\n')
    assert equivalent_windows(original, candidate, max_operations=100, max_variables=32,
                               max_terms=4096)['equivalent'] is True


def test_swapped_zero_fact_follows_value_and_preserves_dirty_output(tmp_path):
    source = tmp_path / 'source.kmx'
    original = 'SWAP q3 q4\nCCX q0 q1 q3\nCX q3 q2\nCCX q0 q1 q3'
    source.write_text(original)
    result = restructure_file(source, tmp_path / 'out', scan_range=(1, 4), clean_inputs=[4],
                              max_operations=8, rounds=1)
    facts = result['episode']['boundary_zero_evidence']
    assert 3 in facts and 4 not in facts
    candidate = (tmp_path / 'out' / 'best.kmx').read_text()
    assert equivalent_windows(original, candidate, clean_inputs=[4], max_operations=100,
                               max_variables=32, max_terms=4096)['equivalent'] is True
    assert result['best']['resources']['Q'] == 5


def test_automatic_sibling_regions_share_total_constructor_budget(tmp_path):
    source = tmp_path / 'source.kmx'
    blocks = []
    for index in range(4):
        q = 3 + index
        blocks += [f'CCX q0 q1 q{q}', f'CX q{q} q2', f'CCX q0 q1 q{q}'] + ['X q10'] * 7
    source.write_text('\n'.join(blocks))
    calls = []
    registry = ConstructorRegistry()

    def no_change(circuit, context):
        calls.append(str(circuit))
        return dict(status='constructed', candidate=str(circuit))

    registry.register('bounded_probe', no_change, 'Return the unchanged episode')
    result = restructure_file(source, tmp_path / 'out', registry=registry, max_operations=6,
                              max_candidates=3, rounds=4)
    assert len(calls) <= 3
    attempted = sum(len(json.loads(Path(row['report']).read_text())['construction']['attempts'])
                    for row in result['attempts'] if 'report' in row)
    assert attempted <= 3
    assert result['construction_budget']['remaining'] >= 0
