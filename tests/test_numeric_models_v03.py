"""Model uncertainty is distinct from observed stopping and exact finite floors."""
from fractions import Fraction
from itertools import product
import json

import numpy as np
import pytest

from strata.model import StrataError, canonical
from strata.numeric_models import (forecast_convergence,declare_stop_rules,evaluate_stop_rules,
                                  quadratic_analysis,exact_grid_floor)
from strata.store import Store


IDENTITY=dict(checkpoint='checkpoint-sha256:fixed',objective='residual',unit='absolute_error')


def trajectories(n=10):
    rng=np.random.default_rng(61)
    runs=[]
    for i in range(n):
        points=[dict(step=j,value=float(.15+.8*np.exp(-j/3)+rng.normal(0,.015))) for j in range(10)]
        runs.append(dict(run_id=f'repeat-{i}',independent=True,**IDENTITY,points=points))
    return runs


def forecast(runs,**kwargs):
    return forecast_convergence(runs,**IDENTITY,target=.2,horizon=8,bootstrap_samples=64,seed=19,**kwargs)


def test_forecast_seeded_bands_and_probability_remain_model_estimates():
    a=forecast(trajectories());b=forecast(trajectories())
    assert a==b
    assert a['status']=='fitted' and 0<=a['target_probability']<=1
    assert a['evidence_kind']=='fitted_trend' and not a['calibrated_certificate']
    assert 'not a calibrated' in a['probability_kind']
    for name in ('mean_trend_band','new_run_prediction_band'):
        bands=a[name]
        assert all(lo<=mid<=hi for lo,mid,hi in zip(bands['lower'],bands['median'],bands['upper']))
        assert bands['steps'][-1]==17
    assert a['calibration']['status']=='held_out_diagnostic'
    assert set(a['calibration']['holdout_run_ids']).isdisjoint(a['calibration']['training_run_ids'])
    assert not a['calibration']['calibrated_certificate']
    assert all('holdout_mse' in fit for fit in a['selected_fits'])
    json.dumps(a,allow_nan=False)


def test_single_curve_correlated_runs_and_small_samples_cannot_get_confident_probability():
    assert forecast(trajectories(1))['target_probability'] is None
    small=forecast(trajectories(3))
    assert small['status']=='fitted' and small['target_probability'] is None
    copied=trajectories(6)
    for row in copied:row['points']=copied[0]['points']
    assert forecast(copied)['target_probability'] is None
    correlated=trajectories(6);correlated[2]['independent']=False
    assert forecast(correlated)['status']=='insufficient_evidence'


def test_forecast_frozen_identity_units_and_run_ids_are_enforced():
    for field in IDENTITY:
        runs=trajectories();runs[0][field]='different'
        with pytest.raises(StrataError,match='frozen'):forecast(runs)
    runs=trajectories();runs[1]['run_id']=runs[0]['run_id']
    with pytest.raises(StrataError,match='distinct'):forecast(runs)
    runs=trajectories();runs[0]['points'][0]['value']=float('nan')
    with pytest.raises(StrataError,match='finite'):forecast(runs)


def test_maximization_forecast_keeps_direction_and_observed_data_identity():
    runs=trajectories(6)
    for run in runs:
        for point in run['points']:point['value']=-point['value']
    result=forecast_convergence(runs,**IDENTITY,target=-.2,horizon=8,direction='max',bootstrap_samples=32)
    assert result['direction']=='max' and 0<=result['target_probability']<=1
    assert result['mean_trend_band']['median'][-1]>result['mean_trend_band']['median'][0]


def observed(store,step,value):
    payload=dict(**IDENTITY,step=step,value=value)
    return dict(payload,kind='measured',artifact_hash=store.blob(canonical(payload).encode()))


def test_predeclared_stop_uses_real_observations_and_cannot_take_forecast(tmp_path):
    store=Store(tmp_path/'store')
    declaration=declare_stop_rules([dict(kind='threshold',operator='<=',value=.2,patience=2),dict(kind='budget',step=10)],
                                  **IDENTITY,declared_before_step=0)
    rows=[observed(store,1,.3),observed(store,2,.19),observed(store,3,.18)]
    result=evaluate_stop_rules(rows,declaration,store=store)
    assert result['stop'] and result['rules'][0]['triggered'] and not result['rules'][1]['triggered']
    assert result['forecast_used'] is False
    rows[2]['kind']='fitted_trend'
    with pytest.raises(StrataError,match='measurements only'):evaluate_stop_rules(rows,declaration,store=store)
    rows[2]['kind']='measured';rows[2]['value']=.01
    with pytest.raises(StrataError,match='bind'):evaluate_stop_rules(rows,declaration,store=store)
    rows[2]=observed(store,3,.18);declaration['rules'][0]['value']=.01
    with pytest.raises(StrataError,match='hash'):evaluate_stop_rules(rows,declaration,store=store)


def test_stop_rule_predeclaration_and_stagnation_are_observed(tmp_path):
    store=Store(tmp_path/'store');rows=[observed(store,i,1.) for i in (1,2,3)]
    declaration=declare_stop_rules([dict(kind='stagnation',window=3,minimum_improvement=0)],**IDENTITY,declared_before_step=0)
    assert evaluate_stop_rules(rows,declaration,store=store)['stop']
    late=declare_stop_rules([dict(kind='budget',step=2)],**IDENTITY,declared_before_step=2)
    with pytest.raises(StrataError,match='before'):evaluate_stop_rules(rows,late,store=store)


def test_exact_rational_grid_with_continuous_compensation_has_analytic_minimum():
    result=exact_grid_floor([[2,1],[1,2]],target=['1/2',0],grids={0:[0,1]},continuous=[1])
    # Schur complement is 3/2; both nearest grid errors are +/-1/2.
    assert result['status']=='exact' and result['minimum']=='3/8'
    assert result['schur_matrix']==[['3/2']]
    assert result['minimizer']==['0','1/4']
    assert result['enumerated_states']==2 and result['certified']
    assert result['certificate_scope']=='supplied finite rational model only'
    json.dumps(result,allow_nan=False)


def test_singular_continuous_compensation_and_all_continuous_edge_cases():
    result=exact_grid_floor([[1,1,0],[1,1,0],[0,0,0]],target=['1/3',0,0],grids={0:[0,1]},continuous=[1,2])
    assert result['minimum']=='0' and result['rational_psd_rank']==1
    assert result['minimizer']==['0','1/3','0']
    free=exact_grid_floor([[1,0],[0,0]],target=[4,5],grids={},continuous=[0,1],offset='1/7')
    assert free['minimum']=='1/7' and free['enumerated_states']==1


def test_exact_grid_matches_independent_fraction_enumeration_for_coupled_model():
    H=[[4,1,-1],[1,3,1],[-1,1,2]];target=[Fraction(1,3),Fraction(-2,5),Fraction(1,2)]
    grids={0:[-1,0,1],1:[-1,0,1],2:[0,1]}
    expected=min(sum((Fraction(x[i])-target[i])*H[i][j]*(Fraction(x[j])-target[j]) for i in range(3) for j in range(3))
                 for x in product(grids[0],grids[1],grids[2]))
    actual=exact_grid_floor(H,target=target,grids=grids)
    assert Fraction(actual['minimum'])==expected and actual['enumerated_states']==18


def test_budget_fallback_never_becomes_exact_or_nonlinear_pipeline_floor():
    result=exact_grid_floor([[2,1],[1,2]],target=['1/2',0],grids={0:[0,1]},continuous=[1],max_states=1)
    assert result['status']=='bounded' and result['certified'] is False
    assert result['model_kind']=='numerical_quadratic_model'
    assert result['numerical_lower_bound_estimate']==pytest.approx(3/8)
    assert 'not a nonlinear pipeline floor' in result['scope']
    assert result['analysis']['numerical_bound_certified'] is False


def test_curvature_ranking_steps_and_cross_coupling_are_dimension_independent():
    result=quadratic_analysis([[4,2,0],[2,3,0],[0,0,1]],continuous=[1],steps=[.1,1,2],parameter_names=['a','b','c'])
    assert result['quantized_indices']==[0,2]
    assert result['sensitivity'][0]['name']=='c'
    assert result['sensitivity'][0]['compensated_scale_score']==4
    a=next(row for row in result['sensitivity'] if row['name']=='a')
    assert a['compensated_curvature']==pytest.approx(4-4/3)
    assert result['cross_coupling'][0]['value']==2
    assert result['continuous_compensator'][0][0]==pytest.approx(2/3)


@pytest.mark.parametrize('H', [[[1,2],[2,1]],[[0,1],[1,0]],[[-1,0],[0,1]]])
def test_indefinite_quadratic_is_never_certified(H):
    with pytest.raises(StrataError,match='positive semidefinite'):
        exact_grid_floor(H,target=[0,0],grids={0:[0],1:[0]})
    with pytest.raises(StrataError,match='positive semidefinite'):
        quadratic_analysis(H)


def test_grid_completeness_and_exact_symmetry_are_required():
    with pytest.raises(StrataError,match='every noncontinuous'):
        exact_grid_floor([[1,0],[0,1]],target=[0,0],grids={0:[0]})
    with pytest.raises(StrataError,match='symmetric'):
        quadratic_analysis([[1,0],[1,1]])


def test_model_identity_changes_with_forecast_configuration_and_grid_step():
    runs=trajectories(6)
    a=forecast(runs)
    b=forecast_convergence(runs,**IDENTITY,target=.1,horizon=8,bootstrap_samples=64,seed=19)
    assert a['source_hash']==b['source_hash'] and a['model_hash']!=b['model_hash']
    assert quadratic_analysis([[1]],steps=[1])['model_hash']!=quadratic_analysis([[1]],steps=[2])['model_hash']


def test_rational_bit_budget_never_labels_unexecuted_exact_work_as_certified():
    result=exact_grid_floor([[2]],target=[0],grids={0:[1]},max_fraction_bits=1)
    assert result['status']=='bounded' and not result['certified']
    assert 'rational' in result['reason']
    assert result['numerical_lower_bound_estimate']==2
