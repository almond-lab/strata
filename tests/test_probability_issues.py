import pytest
from strata.probability import ProbabilityPolicy,compare_candidates
from strata.model import StrataError


def receipt(a,b):
    return {'seeds':[17,29],'draws':[{'seed':17,'resources':{'Q':2,'toffoli':a[0],'fail_rate':a[1]}},
                                    {'seed':29,'resources':{'Q':2,'toffoli':b[0],'fail_rate':b[1]}}]}


def test_expected_ranking_includes_failure_penalty_per_draw():
    p=ProbabilityPolicy((17,29),max_failure=.6,penalty='1/(1-fail_rate)')
    formula='R(fail_rate)*Q^2*toffoli'
    ranked=compare_candidates(p,{'cheap_but_fails':receipt((6,.5),(6,.5)),
                                 'slower_reliable':receipt((10,0),(10,0))},formula)
    assert ranked[0]['candidate']=='slower_reliable'
    assert p.evaluate(receipt((2,0),(10,.5)),formula)['cost']==44


def test_lucky_selection_missing_draws_and_failure_budget_refused():
    p=ProbabilityPolicy((17,29),max_failure=.1)
    r=receipt((2,0),(99,0));r['selected_seed']=17
    with pytest.raises(StrataError,match='favourable'):p.evaluate(r,'toffoli')
    r=receipt((2,0),(99,0));r['draws'].pop()
    with pytest.raises(StrataError,match='every'):p.evaluate(r,'toffoli')
    with pytest.raises(StrataError,match='failure budget'):p.evaluate(receipt((2,.5),(2,0)),'toffoli')


def test_worst_and_policy_binding():
    p=ProbabilityPolicy((17,29),compare='worst')
    assert p.evaluate(receipt((2,0),(20,.2)),'toffoli')['cost']==20
    assert p.fingerprint!=ProbabilityPolicy((17,31),compare='worst').fingerprint
