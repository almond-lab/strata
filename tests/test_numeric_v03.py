from dataclasses import asdict, replace
import json
import math
import pytest

from strata.cost import Formula, ContextFunction
from strata.model import StrataError, canonical, digest
from strata.numeric import (EvidenceValue, ResourceReceipt, NumericStudy,
                            error_budget, design_space, break_even_scale)
from strata.store import Store


@pytest.fixture
def store(tmp_path):
    value=Store(tmp_path/'shared')
    yield value
    value.db.close()


def evidence(store,value,unit='error',kind='model',artifact=None):
    return EvidenceValue(value,unit,kind,artifact_hash=artifact,
                         source_hash=store.blob(b'caller supplied numerical model'))


def receipt(store,resources,artifact,verifier,context,settings,passed=True):
    fields=dict(resources=resources,artifact_hash=artifact,verifier_hash=verifier,
                context_hash=digest(context),settings_hash=digest(settings),passed=passed)
    return ResourceReceipt(**fields,receipt_hash=store.blob(canonical(fields).encode()))


def test_error_ledger_overlapping_hierarchy_signed_cross_terms_and_rounding(store):
    terms=[dict(id='fit',evidence=evidence(store,.3),labels={'stage':'first','source':'do'}),
           dict(id='tail',evidence=evidence(store,.4),labels={'block':'second','source':'oo'}),
           dict(id='coupling',evidence=evidence(store,-.1),role='cross',cross_of=['fit','tail']),
           dict(id='round',evidence=evidence(store,.05),role='rounding')]
    budget=error_budget(terms,unit='error',limit=.8,
        groups={'first':['fit','coupling'],'second':['tail','coupling'],'all':['first','second','round']},
        roots=['all','first'],store=store)
    assert budget['total']==pytest.approx(.65)
    assert budget['remaining']==pytest.approx(.15)
    assert budget['allowed_rounding_budget']==pytest.approx(.2)
    assert 'new rounding may change cross terms' in budget['headroom_scope']
    assert budget['counted_leaf_ids']==['coupling','fit','round','tail']
    assert budget['evidence_totals']['model']==pytest.approx(.65)
    assert next(row for row in budget['leaves'] if row['id']=='tail')['labels']['source']=='oo'
    assert budget['ledger_hash']==digest({k:v for k,v in budget.items() if k!='ledger_hash'})


def test_error_budget_rejects_units_duplicate_subtotals_cycles_and_unbacked_measurement(store):
    term=dict(id='a',evidence=evidence(store,1))
    with pytest.raises(StrataError,match='units'):
        error_budget([term],unit='different',limit=4,store=store)
    with pytest.raises(StrataError,match='unique'):
        error_budget([term,term],unit='error',limit=4,store=store)
    with pytest.raises(StrataError,match='cyclic'):
        error_budget([term],unit='error',limit=4,groups={'x':['y'],'y':['x']},store=store)
    with pytest.raises(StrataError,match='measured evidence'):
        error_budget([dict(id='a',evidence=evidence(store,1,kind='measured'))],unit='error',limit=4,store=store)
    with pytest.raises(StrataError,match='cross terms'):
        error_budget([dict(id='a',evidence=evidence(store,-1))],unit='error',limit=4,store=store)


def test_design_space_prices_actual_discontinuous_width_with_exact_formula(store):
    source=store.blob(b'exact caller resource evaluator source')
    seen=[]
    def resources(shape,context):
        seen.append((shape,context))
        n=shape['width']
        return {'Q':4 if n<=3 else 9,'steps':12-n,'overhead':context['overhead']}
    result=design_space({'width':[2,3,4]},formula=Formula('Q*steps+overhead'),resource_hook=resources,
        source_hash=source,context={'overhead':7},store=store)
    assert [r['resources']['Q'] for r in result['points']]==[4,4,9]
    assert [r['cost'] for r in result['points']]==[47,43,79]
    assert result['best']['settings']=={'width':3}
    assert all(r['evidence_kind']=='model' for r in result['points'])
    assert len(seen)==3
    bounded=design_space({'width':[2,3,4]},formula='Q*steps+overhead',resource_hook=resources,
        source_hash=source,context={'overhead':7},store=store,max_points=1)
    assert bounded['status']=='bounded' and bounded['total_points']==3


def test_design_space_measured_receipts_bind_verifier_artifact_shape_and_context(store):
    artifact=store.blob(b'actual emitted numeric artifact')
    verifier=store.blob(b'pinned verifier source');source=store.blob(b'pinned resource hook')
    context={'contract':'contract-a','dataset':'fixed'}
    def hook(settings,context):return receipt(store,{'Q':7,'work':9},artifact,verifier,context,settings)
    result=design_space({'shape':[3]},formula='Q*work',resource_hook=hook,source_hash=source,
        context=context,store=store,verifier_hash=verifier)
    assert result['points'][0]['cost']==63
    assert result['points'][0]['evidence_kind']=='measured'
    saved=hook({'shape':3},context)
    with pytest.raises(StrataError,match='context/settings'):
        design_space({'shape':[4]},formula='Q*work',resource_hook=lambda *_:saved,source_hash=source,
            context=context,store=store,verifier_hash=verifier)
    with pytest.raises(StrataError,match='payload differs'):
        design_space({'shape':[3]},formula='Q*work',resource_hook=lambda *_:replace(saved,resources={'Q':1,'work':1}),
            source_hash=source,context=context,store=store,verifier_hash=verifier)
    with pytest.raises(StrataError,match='verifier identity'):
        design_space({'shape':[3]},formula='Q*work',resource_hook=lambda *_:saved,source_hash=source,
            context=context,store=store,verifier_hash=source)


def test_design_space_error_budget_excludes_over_budget_model_points(store):
    source=store.blob(b'caller model')
    def budget(settings,context):
        return error_budget([dict(id='round',role='rounding',evidence=evidence(store,.2/settings['bits']))],
                            unit='error',limit=.08,store=store)
    result=design_space({'bits':[1,2,3]},formula='work',resource_hook=lambda s,c:{'work':s['bits']},
        source_hash=source,context={},store=store,budget_hook=budget)
    assert result['best']['settings']=={'bits':3}
    assert result['points'][0]['within_error_budget'] is False
    assert result['best']['evidence_kind']=='model'


def record(study,store,line,settings,cost,*,baseline=None,verified=True,factors=None,base=None):
    base=base or store.blob(b'common frozen base')
    artifact=store.blob(canonical({'line':line,'settings':settings,'cost':cost}).encode())
    verifier=store.blob(b'pinned numerical verifier')
    factors=factors or {'objective':'same','family':'arbitrary'}
    context=dict(contract='frozen-contract',base=base,factors=factors)
    measured=evidence(store,cost,unit='work',kind='measured',artifact=artifact)
    check=receipt(store,{'cost':cost},artifact,verifier,context,settings) if verified else None
    return study.record(contract='frozen-contract',base=base,artifacts={'output':artifact},settings=settings,
        factors=factors,line=line,observations={'cost':measured},baseline=baseline,
        verifier_receipt=check,verifier_hash=verifier if verified else None,siblings=['A','B','C'])


def test_shared_experiments_suggest_only_untested_verified_winning_settings(store):
    study=NumericStudy(store)
    baseline=record(study,store,'A',{'rank':4},100)
    winner=record(study,store,'A',{'rank':2},70,baseline=baseline)
    other_connection=Store(store.path)
    try:
        other=NumericStudy(other_connection)
        factors={'objective':'same','family':'arbitrary'}
        result=other.suggest_transfers(line='B',contract='frozen-contract',factors=factors)
        assert result['suggestions'][0]['settings']=={'rank':2}
        assert result['suggestions'][0]['source_experiment']==winner
        assert result['suggestions'][0]['gain_established_on_target'] is False
        assert other.suggest_transfers(line='B',contract='changed',factors=factors)['suggestions']==[]
        assert other.suggest_transfers(line='B',contract='frozen-contract',factors={'objective':'different'})['suggestions']==[]
        record(study,store,'B',{'rank':2},120)
        assert other.suggest_transfers(line='B',contract='frozen-contract',factors=factors)['suggestions']==[]
        assert study.get(winner)['baseline']==baseline
    finally:other_connection.db.close()


def test_unverified_claimed_win_never_becomes_transfer_and_artifacts_rechecked(store):
    study=NumericStudy(store);base=record(study,store,'A',{'rank':4},100)
    row=record(study,store,'A',{'rank':1},1,baseline=base,verified=False)
    args=dict(line='B',contract='frozen-contract',factors={'objective':'same','family':'arbitrary'})
    assert study.suggest_transfers(**args)['suggestions']==[]
    artifact=study.get(row)['artifacts']['output']
    (store.path/'objects'/artifact).write_bytes(b'tampered')
    with pytest.raises(StrataError,match='missing or invalid'):
        study.get(row)


def test_combination_query_preserves_conflicts_and_budgets(store):
    study=NumericStudy(store);base=record(study,store,'A',{},100)
    record(study,store,'A',{'rank':2},80,baseline=base)
    record(study,store,'A',{'precision':7},70,baseline=base)
    record(study,store,'A',{'rank':3},60,baseline=base)
    args=dict(line='B',contract='frozen-contract',factors={'objective':'same','family':'arbitrary'})
    result=study.combinations(**args,max_candidates=20)
    assert result['conflicts']==1
    assert any(r['settings']=={'rank':2,'precision':7} for r in result['candidates'])
    assert result['gain_established'] is False
    assert study.combinations(**args,max_candidates=1)['status']=='bounded'


def test_already_tested_individual_wins_still_propose_missing_joint_setting(store):
    study=NumericStudy(store);base=record(study,store,'A',{},100)
    record(study,store,'A',{'rank':2},80,baseline=base)
    record(study,store,'A',{'precision':7},75,baseline=base)
    record(study,store,'B',{'rank':2},90)
    record(study,store,'B',{'precision':7},95)
    args=dict(line='B',contract='frozen-contract',factors={'objective':'same','family':'arbitrary'})
    assert study.suggest_transfers(**args)['suggestions']==[]
    result=study.combinations(**args)
    assert len(result['candidates'])==1
    assert result['candidates'][0]['settings']=={'rank':2,'precision':7}


def test_explicit_compatible_block_transfer_keeps_other_factors_frozen(store):
    study=NumericStudy(store)
    source={'family':'shared','block':'left'};target={'family':'shared','block':'right'}
    base=record(study,store,'A',{},100,factors=source)
    record(study,store,'A',{'precision':7},70,factors=source,baseline=base)
    args=dict(line='B',contract='frozen-contract',factors=target)
    assert study.suggest_transfers(**args)['suggestions']==[]
    suggestion=study.suggest_transfers(**args,transfer_keys=['block'])['suggestions'][0]
    assert suggestion['source_factors']==source and suggestion['factors']==target
    assert suggestion['gain_established_on_target'] is False
    assert study.suggest_transfers(line='B',contract='frozen-contract',factors={'family':'different','block':'right'},transfer_keys=['block'])['suggestions']==[]


def test_break_even_linear_scale_is_explicit_and_nonlinearity_rejected():
    result=break_even_scale('costscale*(Q*steps)+fixed',{'Q':3,'steps':7,'fixed':4},
        scale_parameter='costscale',target=46,linear_scale=True)
    assert result['break_even']==2 and result['coefficient']==21 and result['offset']==4
    assert result['evidence_kind']=='model'
    logarithmic=break_even_scale('s*log(x)+2',{'x':math.e**2},scale_parameter='s',target=6,linear_scale=True)
    assert logarithmic['break_even']==pytest.approx(2)
    with pytest.raises(StrataError,match='explicit'):
        break_even_scale('s*x',{'x':2},scale_parameter='s',target=4)
    for expression in ['s*s*x','x/s','sqrt(s)*x','s**2*x']:
        with pytest.raises(StrataError):
            break_even_scale(expression,{'x':2},scale_parameter='s',target=4,linear_scale=True)
    impossible=break_even_scale('s*x+10',{'x':2},scale_parameter='s',target=4,linear_scale=True)
    assert impossible['status']=='unattainable_with_nonnegative_scale'
    hidden=Formula('s*x+hidden()',functions={'hidden':ContextFunction(lambda values:values['s']**2)})
    with pytest.raises(StrataError,match='context-aware'):
        break_even_scale(hidden,{'x':2},scale_parameter='s',target=4,linear_scale=True)


def test_raw_stored_experiment_cannot_forge_a_win_against_its_pinned_receipt(store):
    study=NumericStudy(store);base=record(study,store,'A',{'rank':4},100)
    candidate=record(study,store,'A',{'rank':2},120,baseline=base)
    tampered=study.get(candidate)
    tampered['observations']['cost']['value']=1
    handle=store.put('numeric_experiment',tampered)
    with pytest.raises(StrataError,match='differs from verifier'):
        study.get(handle)
