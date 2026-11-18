import json
from itertools import product

import kickmix as km
import pytest

from strata.cli import main
from strata.episode_search import select_episodes
from strata.restructure import restructure_file


def stored_terminal(cleanup='CCX q0 q1 q4'):
    return f'CCX q0 q1 q4\nCX q4 q2\n{cleanup}\nX q5\nX q5'


@pytest.mark.parametrize('command', ['search', 'restructure'])
def test_cli_threads_joint_limits_to_exact_solver(tmp_path, capsys, command):
    source = tmp_path / 'source.kmx'
    source.write_text(stored_terminal())
    assert main([command, str(source), '--out', str(tmp_path / 'out'), '--clean-inputs', '4,5',
                 '--rounds', '1', '--max-candidates', '1', '--max-and', '5',
                 '--joint-timeout', '0.75', '--trial-seconds', '3',
                 '--joint-max-states', '321', '--joint-max-expansions', '1234']) == 0
    report = json.loads(capsys.readouterr().out)
    trial = report['attempts'][0]
    assert trial['mechanism'] == 'joint_boundary_synthesis'
    assert trial['construction']['exact_search']['limits'] == {
        'max_and': 5, 'timeout': .75, 'max_states': 321, 'max_expansions': 1234,
    }


def test_care_set_trials_have_budget_and_proof_scope(tmp_path):
    source = tmp_path / 'source.kmx'
    source.write_text('CCX q0 q1 q2')
    domain = dict(assignments=[[a, a, y] for a, y in product(range(2), repeat=2)])
    report = restructure_file(source, tmp_path / 'out', domain=domain, max_and=5, joint_timeout=.75)
    assert report['status'] == 'verified_improvement'
    assert report['cost_delta'] < 0
    first = report['attempts'][0]
    trials = {trial['completion']: trial for trial in first['construction']['trials']}
    assert {'degree_ordered_interpolation', 'compatible_cofactor_completion',
            'shared_cofactor_graph', 'exact_joint_on_care_rows'} <= set(trials)
    exact = trials['exact_joint_on_care_rows']['exact_search']
    assert exact['value'] == 0 and exact['limits']['max_and'] == 5
    assert first['check']['domain_hash'] == report['contract']['domain_hash']
    assert not first['check']['all_input_equivalence']


def test_private_measured_erase_unlocks_value_search_at_original_price(tmp_path):
    source = tmp_path / 'source.kmx'
    # First clean product controls a terminal update. The second product is
    # an independently removable materialization to leave a strict saving.
    source.write_text(stored_terminal('HMR q4 b0\nCZ q0 q1 if b0') +
                      '\nCCX q0 q1 q5\nCX q5 q3\nCCX q0 q1 q5')
    report = restructure_file(source, tmp_path / 'private', clean_inputs=(4,5), internal_bits=(0,),
                              cost='toffoli', deep_only=True, fast=True, rounds=1)
    reduction = report['measurement_reductions'][report['baseline']['id']]
    assert reduction['status'] == 'verified'
    assert reduction['proof']['outcome_independence_proved']
    assert report['baseline']['cost'] == 3 and report['best']['cost'] == 1
    assert any(row.get('construction_source') == 'verified_measured_value_surrogate'
               for row in report['attempts'])
    public = restructure_file(source, tmp_path / 'public', clean_inputs=(4,5),
                              cost='toffoli', deep_only=True, fast=True, rounds=1)
    blockers = [row['blocking_instruction'] for row in public['search_control']['screened']
                if row.get('blocking_instruction')]
    assert blockers and blockers[0]['instruction'].startswith('HMR')


def test_wrong_phase_repair_is_never_normalized(tmp_path):
    source = tmp_path / 'source.kmx'
    source.write_text(stored_terminal('HMR q4 b0\nCZ q0 q2 if b0'))
    report = restructure_file(source, tmp_path / 'out', clean_inputs=(4,5), internal_bits=(0,),
                              cost='toffoli', deep_only=True, fast=True, rounds=1)
    failed = report['measurement_reductions'][report['baseline']['id']]
    assert failed['status'] != 'verified'
    assert failed['proof']['reason'] == 'public_value_or_phase_depends_on_outcomes'
    assert report['status'] == 'no_gain'


def test_lookup_retains_shared_cache_and_prices_stronger_baseline(tmp_path):
    source = tmp_path / 'lookup.kmx'
    source.write_text('\n'.join([
        'CCX q0 q1 q5', 'CCX q2 q5 q6', 'CX q6 q3', 'CCX q2 q5 q6',
        'CCX q2 q5 q4', 'CCX q0 q1 q5',
    ]))
    report = restructure_file(source, tmp_path / 'out', clean_inputs=(5,6), cost='toffoli',
                              deep_only=True, fast=True, rounds=1)
    summary = report['discovery_yield']['constructor_comparison']
    assert summary['source_cost'] == 5
    assert summary['constructors']['terminal_lookup_fusion']['best_verified_cost'] == 4
    row = next(row for row in report['attempts'] if row['mechanism'] == 'terminal_lookup_fusion')
    assert 'CCX q0 q1 q5' in row['construction']['candidate']
    assert 'CCX q2 q5 q3' in row['construction']['candidate']
    assert row['check']['equivalent']


def test_automatic_retrieval_localizes_sparse_product_episode(tmp_path):
    source = tmp_path / 'long.kmx'
    source.write_text('CCX q0 q1 q4\n' + 'X q5\n'*24 + 'CX q4 q2\n' +
                      'X q5\n'*24 + 'CCX q0 q1 q4\n')
    report = restructure_file(source, tmp_path / 'out', clean_inputs=(4,), cost='toffoli',
                              max_operations=8, deep_only=True, fast=True, max_candidates=24)
    assert report['cost_delta'] == -1
    target = report['attempts'][0]['target']
    assert target['retrieval'] == 'commuting_product_lifetime'
    assert target['dependency']['wire'] == 4
    assert target['dependency']['consumers'] == [25]
    assert target['localization']['selected'] == [0,25,50]
    # Reading an output requires its original relative order.
    blocked = source.read_text().replace('CX q4 q2\nX q5', 'CX q4 q2\nCX q2 q5')
    assert not any(row.get('localization') for row in select_episodes(
        km.Circuit(blocked), max_operations=8, clean_inputs=(4,)))


def test_measured_cache_reuse_and_delayed_phase_repairs(tmp_path):
    source = tmp_path / 'source.kmx'
    source.write_text('\n'.join([
        'CCX q0 q1 q4', 'CX q4 q2', 'HMR q4 b0', 'X q6', 'CZ q0 q1 if b0',
        'CCX q0 q1 q5', 'CX q5 q3', 'HMR q5 b0', 'CZ q0 q1 if b0',
    ]))
    report = restructure_file(source, tmp_path / 'out', clean_inputs=(4,5), internal_bits=(0,),
                              cost='toffoli', fast=True, deep_only=True, rounds=1)
    reduction = report['measurement_reductions'][report['baseline']['id']]
    assert reduction['status'] == 'verified'
    assert len(reduction['matched']) == 2
    assert report['best']['cost'] == 1


def test_regional_measurement_is_private_until_a_proven_suffix_overwrite(tmp_path):
    source = tmp_path / 'source.kmx'
    episode = ('CCX q0 q1 q4\nCX q4 q2\nHMR q4 b0\nCZ q0 q1 if b0\n'
               'CCX q0 q1 q5\nCX q5 q3\nCCX q0 q1 q5')
    source.write_text(episode + '\nHMR q7 b0')
    report = restructure_file(source, tmp_path / 'out', clean_inputs=(4,5), scan_range=(0,7),
                              cost='toffoli', deep_only=True, fast=True, rounds=1)
    assert report['cost_delta'] == -2
    assert report['episode']['suffix_overwritten_bits'] == [0]
    assert 0 not in report['best']['private_bits']
    assert str(km.Circuit.from_file(report['best']['path'])).endswith('HMR q7 b0')
    source.write_text(episode + '\nX q6 if b0\nHMR q7 b0')
    blocked = restructure_file(source, tmp_path / 'blocked', clean_inputs=(4,5), scan_range=(0,7),
                               cost='toffoli', deep_only=True, fast=True, rounds=1)
    assert blocked['episode']['suffix_overwritten_bits'] == []
    assert blocked['construction']['measurement_reductions'][blocked['construction']['baseline']['id']]['status'] != 'verified'
