import numpy as np
import pytest

from strata.callbacks import PinnedCallback
from strata.model import StrataError
from strata.numeric_candidates import NumericCandidate,NumericScorer,NumericStage,NumericPipeline
from strata.store import Store


def reference(request):
    arrays=request['arrays'];theta=arrays['theta'];words=arrays['words']
    continuous=float(np.sum(theta**2));discrete=float(np.sum(words.astype(float)**2))
    return {'metrics':{'energy':continuous+discrete,'work':float(1+np.sum(np.abs(words)))},
            'checks':{'finite':True},'partitions':{'components':{'metric':'energy','unit':'error',
                'terms':{'continuous':continuous,'discrete':discrete}}}}


def fast_same(request):return reference(request)

def fast_drift(request):
    row=reference(request);row['metrics']['energy']+=1e-7
    row['partitions']['components']['terms']['continuous']+=1e-7
    return row


def bad_attribution(request):
    row=reference(request);row['partitions']['components']['terms']['continuous']+=1
    return row


def derivative(request):return {'kind':'continuous','gradients':{'theta':2*request['arrays']['theta']}}

def wrong_derivative(request):return {'kind':'continuous','gradients':{'theta':np.zeros_like(request['arrays']['theta'])}}

def discrete_derivative(request):return {'kind':'continuous','gradients':{'words':2*request['arrays']['words']}}

def surrogate(request):return {'kind':'surrogate','gradients':{'words':2*request['arrays']['words']}}


def child(parent,arrays,*,masks=None,keep_parent=True):
    manifest=parent.manifest
    return NumericCandidate.create(parent.store,contract=manifest['contract'],arrays=arrays,
        units={key:row['unit'] for key,row in manifest['components'].items()},
        trainable=parent.masks() if masks is None else masks,
        parameters=manifest['parameters'],parent=parent.id if keep_parent else None,stage='test')


def shrink(request):
    parent=request['candidate'];arrays={k:v.copy() for k,v in parent.arrays().items()}
    arrays['theta'][parent.masks()['theta']]*=request['settings'].get('factor',.5)
    return child(parent,arrays)


def neutral(request):
    parent=request['candidate'];arrays={k:v.copy() for k,v in parent.arrays().items()}
    arrays['theta']=np.array([-arrays['theta'][1],arrays['theta'][0]])
    return child(parent,arrays)


def thaw(request):
    parent=request['candidate']
    return child(parent,parent.arrays(),masks={key:np.ones_like(value,dtype=bool) for key,value in parent.arrays().items()})


def lose_parent(request):return child(request['candidate'],request['candidate'].arrays(),keep_parent=False)


def mutate_score(request):
    parent=request['candidate'];result=shrink(request)
    request['score']['metrics']['energy']=reference({'arrays':result.arrays()})['metrics']['energy']
    request['settings']['factor']=100
    return result


@pytest.fixture
def store(tmp_path):
    result=Store(tmp_path/'store');yield result;result.db.close()


def candidate(store,*,frozen=False):
    return NumericCandidate.create(store,contract='quadratic-toy',arrays={'theta':np.array([1.,2.]),'words':np.array([1,2],dtype=np.int16)},
        units={'theta':'amplitude','words':'integer'},trainable={'theta':np.array([not frozen,True]),'words':np.array([True,True])})


def scorer(store,*,fast=None,gradient=None,formula='energy*10+work',reference_callback=reference):
    return NumericScorer(store,contract='quadratic-toy',reference=PinnedCallback(reference_callback),
        metric_units={'energy':'error','work':'operations'},formula=formula,
        fast=PinnedCallback(fast) if fast else None,gradient=PinnedCallback(gradient) if gradient else None)


def test_candidate_round_trip_scalar_dtype_masks_handoff_and_corruption(store):
    c=NumericCandidate.create(store,contract='scalar',arrays={'scalar':np.array(3.,dtype='>f8')},units={'scalar':'u'},
        trainable={'scalar':np.array(False)})
    assert c.arrays()['scalar'].shape==() and c.arrays()['scalar'].dtype.byteorder in ('<','=')
    assert c.masks()['scalar'].shape==() and not bool(c.masks()['scalar'])
    c.manifest['contract']='mutated in memory'
    assert store.get(c.push('agent handoff'),'numeric_handoff')['contract']=='scalar'
    saved=NumericCandidate.load(store,c.id)
    handle=saved.manifest['components']['scalar']['data']
    (store.path/'objects'/handle).write_bytes(b'tamper')
    with pytest.raises(StrataError,match='integrity'):saved.save()
    with pytest.raises(StrataError,match='finite numeric'):
        NumericCandidate.create(store,contract='x',arrays={'x':np.array([object()])},units={'x':'u'})


def test_scorer_prices_reference_first_and_reloads_frozen_metadata(store):
    c=candidate(store,frozen=True);s=scorer(store)
    c.manifest['components']['theta']['trainable_count']=100
    result=s.assess(c)
    assert result['metrics']['energy']==10 and result['cost']==104
    assert result['frozen_parameters']['theta']=={'total':2,'trainable':1,'frozen':1}
    assert result['eligible'] and result['evidence_kind']=='measured'
    assert result['artifact_hash']==c.artifact_hash
    assert store.read_blob(result['artifact_hash'])
    assert result['partitions']['components']['terms']=={'continuous':5,'discrete':5}
    with pytest.raises(StrataError,match='sum'):
        scorer(store,reference_callback=bad_attribution).assess(c)


def test_fast_calibration_checks_cost_amplification_and_stays_model(store):
    c=candidate(store);s=scorer(store,fast=fast_drift,formula='energy*1000000000000')
    with pytest.raises(StrataError,match='calibration'):s.assess(c,fast=True)
    assert s.calibrate([c],atol=1e-6)['passed'] is False
    with pytest.raises(StrataError,match='failed or stale'):s.assess(c,fast=True)
    assert s.calibrate([c],atol=1e-6,cost_atol=1e6)['passed'] is True
    assert s.assess(c,fast=True)['evidence_kind']=='model'
    assert s.assess(c)['cost']==1e13


def test_callback_dependency_change_invalidates_calibration(store,tmp_path):
    dep=tmp_path/'model-data';dep.write_text('one')
    s=scorer(store,fast=fast_same);s.fast=PinnedCallback(fast_same,dependencies=[dep])
    c=candidate(store);assert s.calibrate([c])['passed']
    dep.write_text('two')
    with pytest.raises(StrataError,match='dependency changed'):s.assess(c,fast=True)


def test_analytic_gradient_matches_central_differences_and_bad_gradient_fails(store):
    c=candidate(store)
    result=scorer(store,gradient=derivative).check_gradients(c,metric='energy')
    assert result['passed'] and result['evaluations']==4
    assert [row['analytic'] for row in result['rows']]==[2,4]
    assert [row['finite_difference'] for row in result['rows']]==pytest.approx([2,4],rel=1e-8)
    assert all(store.get(row['plus_candidate'],'numeric_candidate')['parent']==c.id for row in result['rows'])
    assert scorer(store,gradient=wrong_derivative).check_gradients(c,metric='energy')['passed'] is False
    bounded=scorer(store,gradient=derivative).check_gradients(c,metric='energy',max_evaluations=2)
    assert bounded['status']=='bounded' and bounded['passed'] is None


def test_discrete_words_have_finite_moves_not_continuous_derivatives(store):
    c=candidate(store)
    with pytest.raises(StrataError,match='no continuous derivative'):
        scorer(store,gradient=discrete_derivative).gradients(c,metric='energy')
    s=scorer(store,gradient=surrogate)
    assert s.gradients(c,metric='energy')['kind']=='surrogate'
    with pytest.raises(StrataError,match='surrogate'):s.check_gradients(c,metric='energy')
    moves=s.discrete_moves(c,[{'component':'words','index':[1],'value':0}])
    assert moves['moves'][0]['cost_delta']==-42
    assert moves['moves'][0]['eligible'] and moves['evaluations']==2
    with pytest.raises(StrataError,match='dtype range'):
        s.discrete_moves(c,[{'component':'words','index':[0],'value':100000}])
    with pytest.raises(StrataError,match='2 through 16'):
        s.discrete_moves(c,[],max_evaluations=17)


def test_pipeline_cost_after_all_stages_checkpoint_resume_and_neutrality(store):
    c=candidate(store);s=scorer(store)
    pipe=NumericPipeline(store,s,[NumericStage('neutral',PinnedCallback(neutral),invariant_metrics={'energy':0}),
                                NumericStage('refit',PinnedCallback(shrink),settings={'factor':.5})])
    result=pipe.run(c)
    assert result['score']['cost']==66.5 and len(result['checkpoints'])==2
    first=store.get(result['checkpoints'][0],'numeric_pipeline_checkpoint')
    assert first['cost_delta']==0
    resumed=pipe.run(c,resume=result['checkpoints'][0])
    assert resumed['candidate']==result['candidate']
    assert result['external_submission'] is False
    forged=dict(first,next_stage=99)
    handle=store.put('numeric_pipeline_checkpoint',forged)
    with pytest.raises(StrataError,match='stage mismatch'):pipe.run(c,resume=handle)


def test_pipeline_masks_lineage_and_immutable_settings(store):
    c=candidate(store,frozen=True);s=scorer(store)
    for callback,match in [(thaw,'trainable'),(lose_parent,'lineage')]:
        pipe=NumericPipeline(store,s,[NumericStage('bad',PinnedCallback(callback))])
        with pytest.raises(StrataError,match=match):pipe.run(c)
        assert store.all('numeric_pipeline_failure')
    settings={'factor':.5};stage=NumericStage('refit',PinnedCallback(shrink),settings=settings)
    pipe=NumericPipeline(store,s,[stage]);settings['factor']=.25
    pipe.policy['stages'][0]['settings']['factor']=.25
    with pytest.raises(StrataError,match='policy changed'):pipe.run(c)
    good=NumericPipeline(store,s,[NumericStage('refit',PinnedCallback(shrink))]).run(c)
    assert NumericCandidate.load(store,good['candidate']).arrays()['theta'][0]==1


def test_plugin_cannot_mutate_comparison_score_to_fake_neutral_stage(store):
    c=candidate(store);s=scorer(store);settings={'factor':.5}
    pipe=NumericPipeline(store,s,[NumericStage('not-neutral',PinnedCallback(mutate_score),settings=settings,invariant_metrics={'energy':0})])
    with pytest.raises(StrataError,match='declared invariant'):pipe.run(c)
    assert settings=={'factor':.5}
    failure=store.all('numeric_pipeline_failure')[-1][1]
    assert failure['candidate']==c.id and failure['last_checkpoint'] is None
