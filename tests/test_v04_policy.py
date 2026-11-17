import pytest

from strata.experiment_policy import (ScreeningPolicy, ContinuationPolicy,
    record_screening_candidate, record_screening_pair, record_trajectory, policy_request)
from strata.model import StrataError
from strata.store import Store


IDENTITY = dict(scorer='cheap-v1/full-v1', dependencies={'worker': 'frozen'},
                domain={'family': 'generic'}, settings={'precision': 64}, unit='error')


def receipt(store, key, value, *, kind='measured', identity=IDENTITY, run_id=None):
    return store.put('test_evaluation', dict(identity=identity, candidate_key=key,
        **({'run_id': run_id} if run_id else {}),
        metrics={'loss': dict(value=value, unit=identity['unit'], evidence_kind=kind)}))


def candidate(store, key, value, identity=IDENTITY):
    source = receipt(store, key, value, kind='model', identity=identity)
    return record_screening_candidate(store, identity=identity, proxy=value,
                                      source_record=source, candidate_key=key)


def pair(store, key, proxy, full, identity=IDENTITY):
    cheap = candidate(store, key, proxy, identity=identity)
    source = receipt(store, key, full, identity=identity)
    return record_screening_pair(store, candidate=cheap, full=full, full_record=source)


def calibrated_screen(store, identity=IDENTITY, **kwargs):
    policy = ScreeningPolicy.create(store, identity=identity, proxy_threshold=1,
        full_threshold=1, boundary_margin=.1, min_samples=3, max_false_negative_rate=.9,
        **kwargs)
    train = [pair(store, 'train-' + str(i), x, x, identity=identity) for i, x in enumerate([.5, 1., 2.])]
    heldout = [pair(store, 'test-' + str(i), x, x, identity=identity) for i, x in enumerate([.5, 1., 2.])]
    calibration = policy.calibrate(train, heldout)
    return policy, calibration, train, heldout


def trajectory(store, run_id, values, *, identity=IDENTITY, kind='measured'):
    points = []
    for step, value in enumerate(values, 1):
        source = store.put('test_trajectory_point', dict(identity=identity, run_id=run_id, step=step,
            metrics={'loss': dict(value=value, unit=identity['unit'], evidence_kind=kind)}))
        points.append(dict(step=step, value=value, compute_seconds=step * 3,
                           source_record=source, metric='loss'))
    return record_trajectory(store, identity=identity, run_id=run_id, points=points)


def calibration_trajectories(store, identity=IDENTITY):
    train = [trajectory(store, 'train-fail1', [10]*5, identity=identity), trajectory(store, 'train-fail2', [10]*5, identity=identity),
             trajectory(store, 'train-late', [1, 1, 1, 0, 0], identity=identity)]
    test = [trajectory(store, 'test-fail', [10]*5, identity=identity), trajectory(store, 'test-late1', [1, 1, 1, 0, 0], identity=identity),
            trajectory(store, 'test-late2', [1, 1, 1, 1, 0], identity=identity)]
    return train, test


def continuation(store, identity=IDENTITY, **kwargs):
    return ContinuationPolicy.create(store, identity=identity, target=0, horizon=5,
        min_samples=3, min_neighbors=1, max_distance=.1, max_false_negative_rate=.9,
        **kwargs)


def test_screening_calibrated_risk_and_deterministic_full_audits(tmp_path):
    store = Store(tmp_path)
    policy, calibration, _, _ = calibrated_screen(store, audit_fraction=.5)
    assert calibration['status'] == 'passed'
    assert calibration['false_negative_rate'] == 0
    assert calibration['false_negative_upper_bound'] > 0
    decisions = []
    for i in range(30):
        c = candidate(store, 'future-' + str(i), 2.)
        first = policy.decide(c, calibration=calibration['record'], identity=IDENTITY)
        assert policy.decide(c, calibration=calibration['record'], identity=IDENTITY) == first
        assert first['evidence_kind'] == 'model' and not first['acceptance']
        decisions.append(first)
    assert {r['action'] for r in decisions} == {'prune', 'audit_full'}
    audited = next(r for r in decisions if r['action'] == 'audit_full')
    full = receipt(store, audited['candidate_key'], .5)
    paired = record_screening_pair(store, candidate=audited['candidate'], full=.5, full_record=full)
    result = policy.audit(audited['record'], paired)
    assert result['false_negative'] is True
    assert store.get(result['record'], 'screening_audit')['decision'] == audited['record']
    followup = policy.decide(candidate(store,'after-audit',2),calibration=calibration['record'],identity=IDENTITY)
    assert followup['action'] == 'abstain' and 'recalibration' in followup['reason']


def test_screening_leakage_identity_and_insufficient_evidence_abstain(tmp_path):
    store = Store(tmp_path)
    policy, calibration, train, test = calibrated_screen(store)
    with pytest.raises(StrataError, match='leakage'):
        policy.calibrate(train, train)
    invalid = policy.calibrate(train[:1], test[:1])
    assert invalid['status'] == 'insufficient_calibration'
    c = candidate(store, 'new', 2.)
    assert policy.decide(c, calibration=invalid['record'], identity=IDENTITY)['action'] == 'abstain'
    for field, value in [('dependencies', {'worker': 'changed'}), ('unit', 'other'),
                         ('settings', {'precision': 32}), ('scorer', 'changed'), ('domain', 'other')]:
        assert policy.decide(c, calibration=calibration['record'], identity=dict(IDENTITY, **{field:value}))['action'] == 'abstain'
    reused = store.get(test[0], 'screening_pair')['candidate']
    assert policy.decide(reused, calibration=calibration['record'], identity=IDENTITY)['action'] == 'abstain'
    with pytest.raises(StrataError, match='differs'):
        record_screening_candidate(store, identity=IDENTITY, proxy=99,
            candidate_key='wrong', source_record=receipt(store, 'wrong', 2))


def test_screening_false_negative_bound_blocks_apparently_good_small_sample(tmp_path):
    store = Store(tmp_path)
    policy = ScreeningPolicy.create(store, identity=IDENTITY, proxy_threshold=1,
        full_threshold=1, boundary_margin=.1, min_samples=3, max_false_negative_rate=.01)
    train = [pair(store, 'tr'+str(i), x, x) for i,x in enumerate([.5, 1., 2.])]
    test = [pair(store, 'te'+str(i), x, x) for i,x in enumerate([.5, 1., 2.])]
    result = policy.calibrate(train, test)
    assert result['false_negatives'] == 0
    assert result['status'] == 'insufficient_calibration'


def test_continuation_whole_run_holdout_late_wins_and_advisory_default(tmp_path):
    store = Store(tmp_path); train, test = calibration_trajectories(store)
    policy = continuation(store)
    calibration = policy.calibrate(train, test)
    assert calibration['status'] == 'passed'
    assert calibration['late_wins'] == 2 and calibration['lost_wins'] == 0
    assert calibration['compute_saved_seconds'] == 6
    live = trajectory(store, 'live-fail', [10, 10, 10])
    decision = policy.decide(live, calibration=calibration['record'], identity=IDENTITY)
    assert decision['action'] == 'advisory_pause'
    assert decision['prediction']['probability'] == 0
    assert decision['acceptance'] is False
    assert policy.decide(test[0], calibration=calibration['record'], identity=IDENTITY)['action'] == 'abstain'
    winner = trajectory(store, 'live-winner', [1, 1, 1])
    assert policy.decide(winner, calibration=calibration['record'], identity=IDENTITY)['action'] == 'continue'


def test_continuation_automation_requires_valid_calibration_and_preserves_charges(tmp_path):
    store = Store(tmp_path); train, test = calibration_trajectories(store)
    policy = continuation(store, automation=True)
    calibration = policy.calibrate(train, test)
    live = trajectory(store, 'live', [10, 10, 10])
    decision = policy.decide(live, calibration=calibration['record'], identity=IDENTITY)
    assert decision['action'] == 'paused_by_forecast_policy'
    assert decision['resumable'] and decision['charged_compute_seconds'] == 9
    drift = dict(IDENTITY, dependencies={'worker':'v2'})
    assert policy.decide(live, calibration=calibration['record'], identity=drift)['action'] == 'abstain'
    outside = trajectory(store, 'outside', [100, 100, 100])
    assert policy.decide(outside, calibration=calibration['record'], identity=IDENTITY)['action'] == 'abstain'


def test_continuation_late_improvements_defeat_stagnation_model(tmp_path):
    store = Store(tmp_path); train, _ = calibration_trajectories(store)
    policy = continuation(store, automation=True)
    test = [trajectory(store, 'late-high1', [10,10,10,10,0]),
            trajectory(store, 'late-high2', [10,10,10,10,0]),
            trajectory(store, 'fail', [10]*5)]
    calibration = policy.calibrate(train, test)
    assert calibration['status'] == 'insufficient_calibration'
    assert calibration['lost_wins'] == 2 and calibration['false_prunes'] == 2
    assert calibration['lost_win_upper_bound'] == 1
    live = trajectory(store, 'live', [10,10,10])
    assert policy.decide(live, calibration=calibration['record'], identity=IDENTITY)['action'] == 'advisory_pause'
    with pytest.raises(StrataError, match='leakage'):
        policy.calibrate(train, train)
    with pytest.raises(StrataError, match='measured'):
        trajectory(store, 'fake', [10]*5, kind='model')


def test_json_policy_dispatch_retains_declaration(tmp_path):
    store = Store(tmp_path)
    result = policy_request(store, dict(operation='create_continuation', identity=IDENTITY,
                                       target=0, horizon=5))
    assert result['kind'] == 'continuation'
    assert store.get(result['policy'], 'continuation_policy')['automation'] is False

