"""Independent structural-round review on original fixtures and native kickmix."""
from dataclasses import replace
from pathlib import Path

import kickmix as km
import pytest

from strata import open as open_project
from strata.demo import AndFusion
from strata.model import Contract, Debt, Register, Region, StrataError
from strata.research import Dependency, Hypothesis, ResearchPolicy, ResearchRound, close_episode, structural_change


def make_project(tmp_path, adapter=None, extra_scale=False):
    adapter = adapter or AndFusion()
    config = (Path(__file__).parents[1] / 'examples/and_fusion/strata.toml').read_text()
    if extra_scale:
        config = config.replace('bits = { full = 3, small = [1, 2] }',
                                'bits = { full = 3, small = [1, 2] }\nmode = { full = 0, small = [0, 1] }')
    (tmp_path / 'strata.toml').write_text(config)
    params = {'bits': 3, **({'mode': 0} if extra_scale else {})}
    circuit, _ = adapter.build(params, {})
    (tmp_path / 'base.kmx').write_text(str(circuit))
    project = open_project(tmp_path / 'strata.toml', adapters={'and_fusion': adapter})
    project.init()
    return project


def question(mechanism='materialization', implementation=None, **kwargs):
    return Hypothesis('replace-stages', ('root',), mechanism,
                      'Intermediate storage is necessary',
                      'Consume the product directly',
                      'Check all external inputs and clean return',
                      implementation=implementation, **kwargs)


def four_stage_tree():
    contract = Contract('identity', (Register('x', (0,)),), 'identity')
    children = tuple(Region(name, i, i + 1, contract) for i, name in enumerate('abcd'))
    return Region('root', 0, 4, contract, children)


def test_dependency_closure_revisits_intervening_siblings():
    tree = four_stage_tree()
    links = (Dependency('a', 'c', 'value consumed', 'source audit'),
             Dependency('b', 'd', 'cleanup owner', 'source audit'))
    episode = close_episode(tree, ('a',), links)
    assert episode['regions'] == ['a', 'b', 'c', 'd']
    assert len(episode['dependencies']) == 2


def test_phase_debt_follows_creator_and_repayment_in_both_directions():
    tree = four_stage_tree()
    tree = replace(tree, contract=replace(tree.contract, debts=(Debt('sign', 'a', 'd', 'm0', 'a*b'),)))
    for requested in ('a', 'd'):
        episode = close_episode(tree, (requested,))
        assert episode['regions'] == ['a', 'b', 'c', 'd']
        assert episode['dependencies'][0]['reason'] == 'phase debt sign'


def test_estimate_history_does_not_close_or_demote_a_family(tmp_path):
    project = make_project(tmp_path)
    base = project.task('and').base_design()
    project.store.negative(base.tree.contract.fingerprint, 'replace-stages', 'estimate',
                           'old estimated cost is too high', 'different representation')
    plan = ResearchRound(project, base).plan([question(estimate={'cost': 10**30})])
    assert len(plan['selected']) == 1
    item = plan['selected'][0]
    assert item['history'][0]['status'] == 'priced'
    assert item['readiness'] == 'derivation_required'
    result = ResearchRound(project, base).run(plan['handle'])
    assert result['qualified_gains'] == 0
    assert result['outcomes'][0]['status'] == 'DERIVATION_REQUIRED'
    assert len(project.store.all('design')) == 1


def test_selection_diversifies_mechanisms_before_second_choice(tmp_path):
    project = make_project(tmp_path)
    hypotheses = [replace(question('materialization'), change=f'candidate {i}') for i in range(8)]
    hypotheses += [question('representation'), question('algorithm')]
    plan = ResearchRound(project, project.task('and').base_design(), ResearchPolicy(max_lines=3)).plan(hypotheses)
    assert [x['hypothesis']['mechanism'] for x in plan['selected']] == ['materialization', 'representation', 'algorithm']
    assert len(plan['deferred']) == 7


def test_persisted_plan_ignores_mutation_of_input_and_returned_dicts(tmp_path):
    project = make_project(tmp_path)
    base = project.task('and').base_design()
    impl = {'fused': True}
    plan = ResearchRound(project, base).plan([question(implementation=impl)])
    impl['fused'] = False
    plan['selected'][0]['hypothesis']['implementation']['fused'] = False
    stored = project.store.get(plan['handle'], 'research_plan')
    assert stored['selected'][0]['hypothesis']['implementation'] == {'fused': True}


def test_plan_cannot_execute_on_another_certified_base(tmp_path):
    project = make_project(tmp_path)
    base = project.task('and').base_design()
    plan = ResearchRound(project, base).plan([question()])
    line = project.open_line(base, 'replace-stages', ['root'], implementation={'fused': True})
    assert line.lab(bits=1).passed
    line.build()
    child = line.verify()
    with pytest.raises(StrataError, match='different base'):
        ResearchRound(project, child).run(plan['handle'])


def test_relabeling_and_unchanged_artifacts_do_not_become_deep():
    tree = four_stage_tree()
    renamed = replace(tree, id='renamed', kind='new algorithm',
                      children=tuple(replace(r, id='new-' + r.id, kind='fused') for r in tree.children))
    assert not structural_change(tree, renamed, True)['deep']
    assert not structural_change(tree, replace(tree, children=()), False)['deep']


class RecordedFusion(AndFusion):
    def __init__(self):
        self.candidate_scales = []

    def build(self, parameters, implementation):
        circuit, tree = super().build(parameters, implementation)
        if implementation.get('fused'):
            self.candidate_scales.append(dict(parameters))
            merged = Region('merged', tree.start, tree.stop, tree.contract, kind='direct-product-consumer')
            tree = replace(tree, children=(merged,))
        return circuit, tree


def test_every_configured_small_size_combination_precedes_full_build(tmp_path):
    adapter = RecordedFusion()
    project = make_project(tmp_path, adapter, extra_scale=True)
    round_ = ResearchRound(project, project.task('and').base_design())
    plan = round_.plan([question(implementation={'fused': True})])
    result = round_.run(plan['handle'])
    assert adapter.candidate_scales[:4] == [dict(bits=1, mode=0), dict(bits=1, mode=1),
                                            dict(bits=2, mode=0), dict(bits=2, mode=1)]
    outcome = result['outcomes'][0]
    assert outcome['status'] == 'VERIFIED_STRUCTURAL_GAIN'
    assert len(outcome['labs']) == 4
    assert outcome['cost_before'] == 72 and outcome['cost_after'] == 27
    assert outcome['portfolio_multiplier_fixed_other_tasks'] == pytest.approx(72 / 27)
    assert outcome['replay'] is True


class WrongSecondScale(RecordedFusion):
    def build(self, parameters, implementation):
        circuit, tree = super().build(parameters, implementation)
        if implementation.get('fused') and parameters['bits'] == 2:
            circuit = km.Circuit(str(circuit) + '\nX q0')
            tree = replace(tree, stop=tree.stop + 1)
        return circuit, tree


def test_a_counterexample_at_later_small_size_never_certifies(tmp_path):
    project = make_project(tmp_path, WrongSecondScale())
    round_ = ResearchRound(project, project.task('and').base_design())
    result = round_.run(round_.plan([question(implementation={'fused': True})])['handle'])
    assert result['outcomes'][0]['status'] == 'ATTEMPT_FAILED'
    assert result['qualified_gains'] == 0
    assert len(project.store.all('design')) == len(project.store.all('certificate')) == 1


class WrongExternalContract(RecordedFusion):
    def build(self, parameters, implementation):
        circuit, tree = super().build(parameters, implementation)
        if implementation.get('fused'):
            tree = replace(tree, contract=replace(tree.contract, reference='weakened-reference'))
        return circuit, tree


def test_widened_round_preserves_the_external_specification(tmp_path):
    project = make_project(tmp_path, WrongExternalContract())
    round_ = ResearchRound(project, project.task('and').base_design())
    result = round_.run(round_.plan([question(implementation={'fused': True})])['handle'])
    assert result['outcomes'][0]['status'] == 'ATTEMPT_FAILED'
    assert 'external specification' in result['outcomes'][0]['failure']['error']
    assert len(project.store.all('design')) == 1


class TopologyOnlyWithCancellation(AndFusion):
    def build(self, parameters, implementation):
        circuit, tree = super().build(parameters, {})
        if implementation.get('local_cancel'):
            return circuit, replace(tree, children=())
        n = parameters['bits']
        circuit = km.Circuit(str(circuit) + f'\nCCX q0 q{n} q{2*n}\nCCX q0 q{n} q{2*n}')
        last = replace(tree.children[-1], stop=tree.stop + 2)
        return circuit, replace(tree, stop=tree.stop + 2, children=(*tree.children[:-1], last))


def test_local_cancellation_plus_erased_stage_tags_is_not_a_deep_gain(tmp_path):
    project = make_project(tmp_path, TopologyOnlyWithCancellation())
    round_ = ResearchRound(project, project.task('and').base_design())
    result = round_.run(round_.plan([question(implementation={'local_cancel': True})])['handle'])
    assert result['qualified_gains'] == 0
    assert result['outcomes'][0]['status'] in {'BUILT_NONSTRUCTURAL', 'STRUCTURAL_AUDIT_REQUIRED', 'ATTEMPT_FAILED'}


def test_replay_failure_does_not_leave_a_failed_candidate_eligible_for_best(tmp_path, monkeypatch):
    project = make_project(tmp_path, RecordedFusion())
    base = project.task('and').base_design()
    round_ = ResearchRound(project, base)

    def fail_replay(design):
        raise StrataError('replayed builder changed its emitted bytes')

    monkeypatch.setattr(project, 'replay', fail_replay)
    result = round_.run(round_.plan([question(implementation={'fused': True})])['handle'])
    assert result['outcomes'][0]['status'] == 'ATTEMPT_FAILED'
    assert result['qualified_gains'] == 0
    assert project.best()['tasks']['and']['design'] == base.id


def test_mutating_callers_base_record_cannot_invent_measured_gain(tmp_path):
    project = make_project(tmp_path, RecordedFusion())
    base = project.task('and').base_design()
    round_ = ResearchRound(project, base)
    plan = round_.plan([question(implementation={'fused': True}, estimate={'cost': 0.00001})])
    base.record['cost'] = 10**30
    outcome = round_.run(plan['handle'])['outcomes'][0]
    assert outcome['status'] == 'VERIFIED_STRUCTURAL_GAIN'
    assert outcome['cost_before'] == 72
    assert outcome['cost_after'] == 27
    assert outcome['cost_delta'] == -45


def test_small_suite_budget_refuses_partial_qualification(tmp_path):
    project = make_project(tmp_path, RecordedFusion(), extra_scale=True)
    round_ = ResearchRound(project, project.task('and').base_design(),
                           ResearchPolicy(max_small_instances=3))
    result = round_.run(round_.plan([question(implementation={'fused': True})])['handle'])
    assert result['outcomes'][0]['status'] == 'ATTEMPT_FAILED'
    assert 'budget' in result['outcomes'][0]['failure']['error']
    assert len(project.store.all('design')) == 1


def test_allowing_local_moves_does_not_relabel_them_as_structural_gains(tmp_path):
    project = make_project(tmp_path, TopologyOnlyWithCancellation())
    round_ = ResearchRound(project, project.task('and').base_design(), ResearchPolicy(deep_only=False))
    result = round_.run(round_.plan([question(implementation={'local_cancel': True})])['handle'])
    assert result['qualified_gains'] == 0
    assert not result['outcomes'][0]['status'].startswith('VERIFIED_STRUCTURAL')
