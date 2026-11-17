import json
import sys
from fractions import Fraction

import numpy as np
import pytest

from strata.callbacks import PinnedCallback
from strata.declarative import CommandAdapter
from strata.model import StrataError
from strata.numeric_candidates import NumericCandidate, NumericScorer
from strata.numeric_models import exact_grid_floor, quadratic_analysis
from strata.probability import ProbabilityPolicy
from strata.store import Store


def gate_reference(request):
    x=request['arrays']['x']
    return dict(metrics={'loss':float(np.sum(x*x))},checks={'gate':bool(np.all(x<=1))})


def gate_gradient(request):
    return dict(kind='continuous',gradients={'x':2*request['arrays']['x']})


def bad_gate_gradient(request):
    return dict(kind='continuous',gradients={'x':np.zeros_like(request['arrays']['x'])})


def test_gradient_agreement_does_not_require_probe_acceptance(tmp_path):
    store=Store(tmp_path/'store')
    try:
        candidate=NumericCandidate.create(store,contract='gate-tight',arrays={'x':np.array([1.,0.])},units={'x':'unit'})
        scorer=NumericScorer(store,contract='gate-tight',reference=PinnedCallback(gate_reference),
            formula='loss',metric_units={'loss':'squared unit'},gradient=PinnedCallback(gate_gradient))
        assert scorer.assess(candidate)['eligible']
        result=scorer.check_gradients(candidate,metric='loss')
        assert result['passed'] and result['status']=='consistent_on_probes'
        row=result['rows'][0]
        assert row['derivative_agrees'] and not row['probe_eligible']
        assert not row['plus_eligible'] and row['minus_eligible']
        assert not store.get(row['plus_score'],'numeric_score')['eligible']
        assert result['rows'][1]['analytic']==0 and result['rows'][1]['passed']
        scorer.gradient=PinnedCallback(bad_gate_gradient)
        wrong=scorer.check_gradients(candidate,metric='loss')
        assert not wrong['passed'] and wrong['status']=='inconsistent_on_probes'
    finally:store.db.close()


@pytest.mark.parametrize('steps', [[.1,None,2],[.1,0,2],[.1,2]])
def test_continuous_coordinates_do_not_need_grid_steps(steps):
    result=quadratic_analysis([[4,2,0],[2,3,0],[0,0,1]],continuous=[1],steps=steps)
    reference=quadratic_analysis([[4,2,0],[2,3,0],[0,0,1]],continuous=[1],steps=[.1,1,2])
    assert result['sensitivity']==reference['sensitivity']
    assert result['model_hash']==reference['model_hash']


@pytest.mark.parametrize('steps', [[0,None],[None,0],[-1,None]])
def test_quantized_steps_still_must_be_positive(steps):
    with pytest.raises(StrataError):quadratic_analysis([[1,0],[0,1]],continuous=[1],steps=steps)


def test_rational_floor_has_safe_float_summaries_and_explicit_budget():
    result=exact_grid_floor([[2.,.3],[.3,1.]],target=[.13,0.],grids={0:[0,1]},continuous=[1])
    assert result['certified'] and result['minimum_float']==pytest.approx(float(Fraction(result['minimum'])))
    assert result['minimizer_float']==pytest.approx([float(Fraction(x)) for x in result['minimizer']])
    json.dumps(result,allow_nan=False)
    bounded=exact_grid_floor([[2.,.3],[.3,1.]],target=[.13,0.],grids={0:[0,1]},continuous=[1],max_fraction_bits=4)
    assert bounded['status']=='bounded' and not bounded['certified']
    big=exact_grid_floor([[1]],target=[0],grids={0:[0]},offset=10**400)
    assert big['minimum_float'] is None
    json.dumps(big,allow_nan=False)


def test_raised_rational_budget_can_emit_strings_beyond_interpreter_digit_limit():
    before=sys.get_int_max_str_digits()
    result=exact_grid_floor([[1]],target=[Fraction(1,10**5000)],grids={0:[0]},max_fraction_bits=65536)
    assert result['certified'] and result['minimum']=='1/1'+'0'*10000
    assert result['minimum_float']==0 and result['minimizer_float']==[0.]
    json.dumps(result,allow_nan=False)
    replay=exact_grid_floor(**result['model'],max_fraction_bits=65536)
    assert replay['minimum']==result['minimum'] and replay['model_hash']==result['model_hash']
    huge=exact_grid_floor([[1]],target=[0],grids={0:[0]},offset=10**5000,max_fraction_bits=65536)
    assert huge['minimum']=='1'+'0'*5000 and huge['minimum_float'] is None
    assert sys.get_int_max_str_digits()==before


def count_receipt(**confidence):
    return {'seeds':[17,29],'draws':[
        dict(seed=seed,resources=dict(Q=2,toffoli=10,fail_rate=0.,fail_count=0,shots=1000,eps_plus=.1),
            confidence_bounds={'eps_plus':dict(kind='upper',method='host binomial',fail_count=0,shots=1000,**confidence)})
        for seed in (17,29)]}


@pytest.mark.parametrize('confidence',[{'log2_delta':-128},{'delta':2.**-128},{'confidence':.95},
                                    {'delta':.25,'log2_delta':-2,'confidence':.75}])
def test_extreme_confidence_preserved_without_subtraction(confidence):
    policy=ProbabilityPolicy((17,29),penalty='1+eps_plus+fail_count/shots')
    result=policy.evaluate(count_receipt(**confidence),'Q*toffoli*R(fail_rate)')
    assert result['cost']==22
    assert all(result['failure_inputs'][0]['confidence_bounds']['eps_plus'][k]==v for k,v in confidence.items())
    assert result['failure_inputs'][0]['empirical_rate']==0


@pytest.mark.parametrize('confidence',[{}, {'confidence':1.}, {'delta':0}, {'log2_delta':0},
    {'log2_delta':float('-inf')}, {'log2_delta':True}, {'log2_delta':-(10**400)}, {'delta':.25,'log2_delta':-3}])
def test_invalid_or_contradictory_confidence_rejected(confidence):
    with pytest.raises(StrataError):ProbabilityPolicy((17,29)).evaluate(count_receipt(**confidence),'toffoli')


def test_count_schema_preflight_does_not_invoke_verifier(tmp_path):
    policy=ProbabilityPolicy((17,29),penalty='1+eps_plus+fail_count/shots')
    schema=policy.receipt_schema('Q*toffoli*R(fail_rate)')
    assert schema['required_draw_resources']==['Q','eps_plus','fail_count','fail_rate','shots','toffoli']
    with pytest.raises(StrataError,match='fail_count'):
        policy.validate_resource_schema(['Q','toffoli','fail_rate','failures','shots','eps_plus'],'Q*toffoli*R(fail_rate)')
    (tmp_path/'base.kmx').write_text('X q0\n')
    config=dict(base='base.kmx',cost='Q*toffoli*R(fail_rate)',
        probability=dict(seeds=[17,29],penalty='1+fail_count/shots'),
        verifier=dict(receipt_schema=dict(draw_resources=['Q','toffoli','fail_rate','failures','shots']),
                      command=['/never/executed','{request}']))
    with pytest.raises(StrataError,match='draw resource schema missing: fail_count'):
        CommandAdapter(tmp_path,config)


@pytest.mark.parametrize('count_resource,missing',[('shots','fail_count'),('fail_count','shots')])
def test_declared_count_resources_require_the_pair_even_when_not_priced(count_resource,missing):
    policy=ProbabilityPolicy((17,29))
    with pytest.raises(StrataError,match=missing):
        policy.validate_resource_schema(['toffoli','fail_rate',count_resource],'toffoli')


def test_empirical_rate_cannot_round_positive_counts_down_to_zero():
    policy=ProbabilityPolicy((17,29))
    draws=[dict(seed=seed,resources=dict(toffoli=1,fail_rate=0.,fail_count=1,shots=10**13))
           for seed in policy.seeds]
    with pytest.raises(StrataError,match='empirical failure rate disagrees'):
        policy.evaluate(dict(seeds=list(policy.seeds),draws=draws),'toffoli')
    for draw in draws:draw['resources']['fail_rate']=1/10**13
    result=policy.evaluate(dict(seeds=list(policy.seeds),draws=draws),'toffoli')
    assert result['mean_failure']==1/10**13


def test_supplied_upper_bound_cannot_be_below_a_small_empirical_rate():
    receipt=count_receipt(log2_delta=-128)
    for draw in receipt['draws']:
        draw['resources'].update(fail_count=1,shots=10**13,fail_rate=1/10**13,eps_plus=0.)
        draw['confidence_bounds']['eps_plus'].update(fail_count=1,shots=10**13)
    with pytest.raises(StrataError,match='invalid verifier-supplied confidence bound'):
        ProbabilityPolicy((17,29)).evaluate(receipt,'toffoli')
