import pytest

from strata.experiment_reports import (experiment_report, render_report_markdown,
                                       agent_packet, record_agent_decision)
from strata.model import StrataError, digest
from strata.numeric import design_space
from strata.store import Store


def experiment(store, settings=None, manifest=None):
    manifest = manifest or store.put('experiment_manifest', dict(name='generic',
        verifier=dict(version='pinned-1', accepted_selector='accepted', metrics={'loss':'loss'})))
    return store.put('experiment', dict(manifest=manifest, settings=settings or {},
                                       identity=digest(settings or {})))


def finish(store, exp, value=None, accepted=False, seconds=None):
    manifest = store.get(exp)['manifest']
    pin = (store.get(manifest) if isinstance(manifest,str) else manifest)['verifier']
    return store.put('experiment_finish', dict(experiment=exp, attempt='attempt-'+exp,
        status='accepted_by_verifier' if accepted else 'artifact_validated', seconds=seconds,
        metrics={} if value is None else {'loss':dict(value=value, unit='error', evidence_kind='verified' if accepted else 'measured',
            source=dict(verifier_identity=digest(pin),selector='loss') if accepted else {})},
        artifacts={'candidate':{'sha256':store.blob(b'candidate')}},
        verifier=dict(identity=digest(pin),receipt={'accepted':True,'loss':value},returncode=0) if accepted else None))


def test_reports_missing_is_not_zero_and_unaccepted_win_cannot_replace_safe(tmp_path):
    store = Store(tmp_path)
    a, b, c = [experiment(store, {'setting':x}) for x in range(3)]
    finish(store, a, 2., accepted=True, seconds=1.)
    finish(store, b, 0., seconds=2.)
    finish(store, c)
    report = experiment_report(store)
    assert report['best_independently_accepted']['experiment'] == a
    assert report['outcomes'][1]['metrics']['loss']['value'] == 0
    assert report['outcomes'][2]['metrics']['loss'] is None
    assert not report['outcomes'][2]['numeric_analysis_available']
    assert report['accounting']['attempts_missing_seconds'] == 1
    markdown = render_report_markdown(report)
    assert 'loss: unavailable' in markdown and '0.0 error (measured)' in markdown
    assert experiment_report(Store(tmp_path)) == report


def test_report_requires_pinned_independent_acceptance(tmp_path):
    store = Store(tmp_path); exp = experiment(store)
    store.put('experiment_finish', dict(experiment=exp, status='accepted_by_verifier',
        metrics={'loss':dict(value=0, unit='error', evidence_kind='verified')},
        verifier={'passed':True}, seconds=0))
    assert experiment_report(store)['best_independently_accepted'] is None


def test_worker_resource_claim_is_not_verified_by_boolean_acceptance(tmp_path):
    store = Store(tmp_path); exp = experiment(store)
    handle = finish(store, exp, 0, accepted=True)
    row = store.get(handle, 'experiment_finish')
    row['metrics']['loss']['evidence_kind'] = 'measured'
    # Use a new store so no previous verified metric can become the baseline.
    other = Store(tmp_path / 'other')
    for kind in ('experiment_manifest', 'experiment'):
        for _, record in store.all(kind):
            other.put(kind, record)
    other.blob(b'candidate')
    other.put('experiment_finish', row)
    report = experiment_report(other)
    assert report['outcomes'][0]['independently_accepted']
    assert report['best_independently_accepted'] is None


def test_verified_cost_recomputes_pinned_shape_formula_and_artifact_bytes(tmp_path):
    store = Store(tmp_path)
    verifier = dict(version='pinned',accepted_selector='accepted')
    declared = dict(expression='ceil(points/4)*steps',unit='batches',resources={'points':'resources.points'})
    manifest = store.put('experiment_manifest',dict(verifier=verifier,cost=declared))
    exp = experiment(store,{'steps':3},manifest)
    artifact = store.blob(b'real candidate')
    cost = dict(value=6,expression=declared['expression'],unit='batches',resource_selectors=declared['resources'],
        inputs={'points':5,'steps':3},verifier_identity=digest(verifier),evidence_kind='verified')
    store.put('experiment_finish',dict(experiment=exp,status='accepted_by_verifier',seconds=1,
        artifacts={'candidate':{'sha256':artifact}},metrics={'points':dict(value=1,unit='items',evidence_kind='measured')},
        cost=cost,verified_cost=cost,verifier=dict(identity=digest(verifier),returncode=0,
            receipt={'accepted':True,'resources':{'points':5}})))
    report = experiment_report(store)
    assert report['best_independently_accepted']['value'] == 6
    assert report['best_independently_accepted']['unit'] == 'batches'
    (store.path/'objects'/artifact).write_bytes(b'corrupted')
    with pytest.raises(StrataError,match='integrity'):
        experiment_report(store)


def test_report_verifier_selectors_match_json_pointer_and_last_reader_semantics(tmp_path):
    store = Store(tmp_path)
    verifier = dict(version='pinned',accepted_selector='/check~1result/accepted',
                    metrics={'loss':['history','last','loss']})
    manifest = store.put('experiment_manifest',dict(verifier=verifier))
    exp = experiment(store,manifest=manifest)
    store.put('experiment_finish',dict(experiment=exp,status='accepted_by_verifier',seconds=1,
        artifacts={'candidate':{'sha256':store.blob(b'checked candidate')}},
        metrics={'loss':dict(value=.25,unit='error',evidence_kind='verified',
            source={'verifier_identity':digest(verifier),'verifier_selector':['history','last','loss']})},
        verifier=dict(identity=digest(verifier),returncode=0,
            receipt={'check/result':{'accepted':True},'history':[{'loss':1},{'loss':.25}]})))
    assert experiment_report(store)['best_independently_accepted']['value']==.25


def test_generic_nonadditive_budget_discontinuous_map_and_declared_bets(tmp_path):
    store = Store(tmp_path)
    evidence = lambda value: dict(value=value, unit='error', evidence_kind='model', source_hash='a'*64)
    budget = dict(unit='error',limit=1.,terms=[
        dict(id='a',evidence=evidence(.4)), dict(id='b',evidence=evidence(.3)),
        dict(id='interaction',role='cross',cross_of=['a','b'],evidence=evidence(-.2))])
    space = design_space({'n':[3,4,5]}, formula='blocks*factor',
        resource_hook=lambda setting, context: {'blocks':(setting['n']+3)//4,'factor':2},
        source_hash='a'*64, context={'contract':'generic'})
    map_id = store.put('test_design_map', space)
    payload = dict(error_budget=budget, design_space=map_id, score_bets=dict(
        current_costs={'x':4}, baseline_costs={'x':8}, bets=[dict(id='idea',costs={'x':2},
        probability=dict(value=.5,evidence_kind='judgment'))]))
    report = experiment_report(store, payload=payload)
    assert report['analyses']['error_budget']['total'] == pytest.approx(.5)
    assert [r['cost'] for r in report['analyses']['design_space']['points']] == [2,2,4]
    assert report['analyses']['score_bets']['bets'][0]['evidence_kind'] == 'judgment'
    assert report['best_independently_accepted'] is None
    assert experiment_report(store, payload=report['payload']) == report


def test_packet_decision_experiment_feedback_and_reopening(tmp_path):
    store = Store(tmp_path); exp = experiment(store)
    first = agent_packet(store, payload={'unresolved_questions':['Does the new representation help?']})
    decision = record_agent_decision(store, packet=first['record'], hypothesis='representation reduces error',
        decisive_experiment=exp, expected_outcome='loss below 1', budget={'seconds':10},
        reopening_condition='different precision or stronger optimizer')
    store.put('experiment_finish', dict(experiment=exp,attempt='attempt',status='failed',seconds=2,
                                       metrics={},error='optimizer did not reach target'))
    second = agent_packet(store)
    assert second['prior_agent_decisions'][0]['record'] == decision['record']
    assert second['failed_hypotheses'][0]['reopening_condition']
    assert second['tried'][0]['experiment'] == exp
    assert second['best_independently_accepted'] is None
    assert store.get(decision['record'])['packet'] == first['record']
    assert agent_packet(store) == second


def test_packet_rejects_incomplete_transfer_identity(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(StrataError, match='shape, units'):
        agent_packet(store, transfer_target=dict(line='target', contract='generic', factors={'shape':[2]}))


def test_packet_suggests_compatible_untested_win_and_rejects_shape(tmp_path):
    from strata.numeric import NumericStudy
    from test_numeric_v03 import record
    store = Store(tmp_path); study = NumericStudy(store)
    factors = dict(shape=[2,3],units={'cost':'work'},parameter_meanings={'rank':'matrix rank'},dependencies={'solver':'v1'})
    base = record(study,store,'A',{'rank':4},100,factors=factors)
    win = record(study,store,'A',{'rank':2},70,baseline=base,factors=factors)
    incompatible = record(study,store,'C',{'rank':1},10,factors=dict(factors,shape=[3,3]))
    target = dict(line='B',contract='frozen-contract',factors=factors)
    packet = agent_packet(store,transfer_target=target)
    transfers = packet['compatible_untried_transfers']
    assert transfers['suggestions'][0]['source_experiment'] == win
    assert transfers['suggestions'][0]['gain_established_on_target'] is False
    assert next(row for row in transfers['rejected'] if row['experiment']==incompatible)['reasons'] == ['shape']
    record(study,store,'B',{'rank':2},120,factors=factors)
    assert not agent_packet(store,transfer_target=target)['compatible_untried_transfers']['suggestions']
    with pytest.raises(StrataError,match='cannot be relaxed'):
        agent_packet(store,transfer_target=dict(target,transfer_keys=['shape']))


def native_sibling_fixture(tmp_path, *, different_input=False, transfer_inputs=False):
    import json
    import sys
    from strata.experiments import create_experiment
    (tmp_path/'worker.py').write_text('print("worker")\n')
    (tmp_path/'verify.py').write_text('print("verifier")\n')
    (tmp_path/'left.txt').write_text('left block\n')
    (tmp_path/'right.txt').write_text('different block\n' if different_input else 'left block\n')
    store = Store(tmp_path/'store')
    compatibility = dict(shape=[2,3],units={'loss':'error'},parameter_meanings={'rank':'matrix rank'},dependencies={'algorithm':'v1'})
    if transfer_inputs:
        compatibility['transfer_inputs'] = ['block']
    spec = dict(name='native-transfer',contract='generic independent loss',settings={'rank':4},compatibility=compatibility,
        argv=[sys.executable,'{dependency:worker}'],dependencies={'worker':'worker.py'},
        inputs={'block':'left.txt'},outputs=[dict(name='candidate',path='result.json',type='json')],
        metrics={'loss':dict(output='candidate',selector='loss',unit='error')},
        verifier=dict(version='v1',argv=[sys.executable,'{dependency:verify}','{artifact:candidate}'],
            dependencies={'verify':'verify.py'},metrics={'loss':'loss'}))
    left = tmp_path/'left.json'; left.write_text(json.dumps(dict(spec,line='left')))
    baseline = create_experiment(store,left)
    winner = create_experiment(store,left,settings={'rank':2})
    finish(store,baseline,10,accepted=True)
    finish(store,winner,5,accepted=True)
    right = tmp_path/'right.json'; right.write_text(json.dumps(dict(spec,line='right',inputs={'block':'right.txt'})))
    target = create_experiment(store,right)
    return store,baseline,winner,target,right,spec


def test_native_manifest_transfer_reaches_packet_without_numeric_study(tmp_path,capsys):
    import json
    from strata.cli import main
    store,baseline,winner,target,right,spec = native_sibling_fixture(tmp_path)
    assert not store.all('numeric_experiment')
    payload = tmp_path/'packet-input.json'; payload.write_text(json.dumps({'transfer_experiment':target,'metric':'loss'}))
    assert main(['agent-packet','--store',str(store.path),'--payload',str(payload)]) == 0
    packet = json.loads(capsys.readouterr().out)
    suggestion = packet['compatible_untried_transfers']['suggestions'][0]
    assert suggestion['source_experiment']==winner and suggestion['target_experiment']==target
    assert suggestion['source_baseline']['experiment']==baseline
    assert suggestion['settings']=={'rank':2} and suggestion['observed_metric_delta']==-5
    assert suggestion['evidence_kind']=='model' and not suggestion['gain_established_on_target']
    assert suggestion['source_identity'] and suggestion['target_identity']
    assert not suggestion['changed_inputs']
    assert agent_packet(store,experiment=target)['compatible_untried_transfers']['suggestions']


def test_native_transfer_rejects_shape_unverified_and_already_tried(tmp_path):
    import json
    from strata.experiments import create_experiment
    store,_,winner,target,right,spec = native_sibling_fixture(tmp_path)
    bad_path = tmp_path/'bad.json'
    bad_path.write_text(json.dumps(dict(spec,line='bad',compatibility=dict(spec['compatibility'],shape=[3,3]))))
    bad = create_experiment(store,bad_path,settings={'rank':1}); finish(store,bad,1,accepted=True)
    unverified_path = tmp_path/'unverified.json'
    unverified_path.write_text(json.dumps({key:value for key,value in dict(spec,line='unverified').items() if key!='verifier'}))
    unverified = create_experiment(store,unverified_path,settings={'rank':1})
    tried = create_experiment(store,right,settings={'rank':2}); finish(store,tried,20)
    transfers = agent_packet(store,payload={'transfer_experiment':target})['compatible_untried_transfers']
    assert transfers['suggestions']==[]
    reasons = {r['experiment']:r['reasons'] for r in transfers['rejected']}
    assert 'shape' in reasons[bad]
    assert 'actual evaluator/dependency/verifier/runtime identity' in reasons[unverified]
    assert 'settings already attempted or selected on target line' in reasons[winner]


@pytest.mark.parametrize('allow',[False,True])
def test_native_transfer_different_blocks_require_explicit_input_opt_in(tmp_path,allow):
    store,_,winner,target,_,_ = native_sibling_fixture(tmp_path,different_input=True,transfer_inputs=allow)
    transfers = agent_packet(store,payload={'transfer_experiment':target})['compatible_untried_transfers']
    if allow:
        suggestion = transfers['suggestions'][0]
        assert suggestion['source_experiment']==winner and suggestion['changed_inputs']==['block']
        assert suggestion['source_input_hashes'] != suggestion['target_input_hashes']
        assert not suggestion['gain_established_on_target']
    else:
        assert transfers['suggestions']==[]


def test_native_transfers_and_agent_history_are_bounded_with_full_receipts(tmp_path):
    from strata.experiments import create_experiment
    store,_,_,target,_,_ = native_sibling_fixture(tmp_path)
    for rank,value in [(1,4),(3,3)]:
        candidate = create_experiment(store,tmp_path/'left.json',settings={'rank':rank})
        finish(store,candidate,value,accepted=True)
    original = agent_packet(store)
    for number in range(3):
        record_agent_decision(store,packet=original['record'],hypothesis=f'hypothesis {number}',
            decisive_experiment=target,expected_outcome='lower verified loss',budget={'seconds':1},
            reopening_condition='new source evidence')
        store.put('experiment_board_event',dict(job_id=f'job-{number}',type='enqueued',
            experiment_id=target,resources={'threads':1},timestamp=number))
    packet = agent_packet(store,payload={'transfer_experiment':target,'transfer_limit':2,'packet_limit':2})
    transfers = packet['compatible_untried_transfers']
    assert len(transfers['suggestions'])==2 and transfers['total_eligible']==3 and transfers['truncated']
    assert len(packet['tried'])==2 and packet['counts']['tried']==4 and packet['truncated']['tried']
    assert len(packet['prior_agent_decisions'])==2 and packet['counts']['prior_agent_decisions']==3
    assert len(packet['active_jobs'])==3
    full = store.get(packet['report'],'experiment_report')
    assert len(full['outcomes'])==4 and len(full['agent_decisions'])==3
    with pytest.raises(StrataError,match='transfer_limit'):
        agent_packet(store,payload={'transfer_experiment':target,'transfer_limit':257})
