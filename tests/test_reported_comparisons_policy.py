"""Reported multi-agent comparison and calibration failures (#64, #65, #72)."""
import json
import os
import subprocess
import sys

import pytest

from strata.experiment_comparisons import IDENTITY_SCHEMA, verifier_identity
from strata.experiment_observations import invalidate_verification
from strata.experiment_policy import (ScreeningPolicy, record_screening_candidate,
                                      record_screening_pair)
from strata.experiment_reports import agent_packet, experiment_report
from strata.experiments import create_experiment
from strata.model import StrataError
from strata.store import Store
from test_v041_comparisons import manifest, retained_finish


def failed_finish(store, exp, *, legacy=False):
    e = store.get(exp)
    return store.put('experiment_finish', dict(experiment=exp, status='rejected_by_verifier' if legacy else 'verification_failed',
        artifacts={'candidate': {'sha256': store.blob(b'candidate')}}, metrics={}, cost=None,
        verifier=dict(identity=verifier_identity(e['manifest'], e), identity_schema=IDENTITY_SCHEMA,
                      returncode=0 if legacy else -24,
                      receipt={'accepted': False, 'process': {'returncode': -24}})))


def test_failed_execution_does_not_poison_rerun_but_missing_draw_stays_pending(tmp_path):
    store = Store(tmp_path / 's')
    path = manifest(tmp_path, draw=True, policy={'aggregate': 'mean', 'draws': [{'draw': 0}, {'draw': 2}]})
    a, b = [create_experiment(store, path, {'draw': d}) for d in (0, 2)]
    failure = failed_finish(store, b)
    retained_finish(store, a, resource=2)
    first = experiment_report(store, payload={'metric': 'cost'})
    assert first['best_independently_accepted'] is None
    assert len(first['artifact_comparisons'][0]['missing_draws']) == 1
    retained_finish(store, b, resource=4)
    final = experiment_report(store, payload={'metric': 'cost'})
    assert final['best_independently_accepted']['value'] == 10
    obs = final['artifact_comparisons'][0]['observations']
    assert next(o for o in obs if o['result'] == failure)['disposition'] == 'no_verdict'


def test_legacy_infrastructure_correction_keeps_original_and_requires_replacement(tmp_path):
    store = Store(tmp_path / 's')
    exp = create_experiment(store, manifest(tmp_path))
    h = failed_finish(store, exp, legacy=True)
    original = store.get(h)
    correction = invalidate_verification(store, h, actor='researcher', reason='child killed by CPU limit')
    assert correction['evidence']['returncode'] == -24
    assert experiment_report(store, payload={'metric': 'cost'})['best_independently_accepted'] is None
    retained_finish(store, exp, resource=3)
    report = experiment_report(store, payload={'metric': 'cost'})
    assert report['best_independently_accepted']['value'] == 9
    assert store.get(h) == original
    assert report['artifact_comparisons'][0]['observations'][0]['invalidation']['actor'] == 'researcher'


def test_genuine_rejection_cannot_be_invalidated_or_hidden_by_success(tmp_path):
    store = Store(tmp_path / 's')
    exp = create_experiment(store, manifest(tmp_path))
    h = retained_finish(store, exp, accepted=False)
    with pytest.raises(StrataError, match='scientific rejection'):
        invalidate_verification(store, h, actor='a', reason='prefer a successful draw')
    accepted = retained_finish(store, exp)
    with pytest.raises(StrataError, match='accepted scientific'):
        invalidate_verification(store, accepted, actor='a', reason='remove')
    assert experiment_report(store)['best_independently_accepted'] is None


def test_experiment_focus_cannot_hide_sibling_rejection(tmp_path):
    store = Store(tmp_path / 's')
    a = create_experiment(store, manifest(tmp_path, 'a'))
    b = create_experiment(store, manifest(tmp_path, 'b'))
    retained_finish(store, a)
    retained_finish(store, b, accepted=False)
    assert experiment_report(store, experiment=a)['best_independently_accepted'] is None


def test_scope_comparison_across_all_draw_experiments_and_agent_packet(tmp_path):
    store = Store(tmp_path / 's')
    policy = {'aggregate': 'mean', 'draws': [{'draw': 0}, {'draw': 2}]}
    path = manifest(tmp_path, draw=True, policy=policy)
    for d in (0, 2):
        retained_finish(store, create_experiment(store, path, {'draw': d}), resource=2+d)
    other = create_experiment(store, manifest(tmp_path, 'other', contract={'task': 'other'}))
    retained_finish(store, other)
    all_groups = experiment_report(store)
    assert all_groups['best_independently_accepted'] is None
    identity = next(a['comparison_identity'] for a in all_groups['artifact_comparisons'] if a['aggregate'] == 'mean')
    report = experiment_report(store, payload={'comparison_identity': identity})
    assert report['best_independently_accepted']['value'] == 10
    assert len(report['outcomes']) == 2
    assert agent_packet(store, payload={'comparison_identity': identity})['best_independently_accepted']['value'] == 10
    assert experiment_report(store, payload={'comparison_identity': 'absent'})['best_independently_accepted'] is None


PROXY = dict(scorer='cheap', dependencies={'code': 'pinned'}, domain='finite-v1', settings={}, unit='proxy-error')


def cheap(store, key, value, artifact=b'candidate'):
    source = store.put('cheap_test_receipt', dict(candidate_key=key,
        metrics={'loss': {'value': value, 'unit': PROXY['unit'], 'evidence_kind': 'measured'}},
        artifacts={'candidate': {'sha256': store.blob(artifact)}}))
    return record_screening_candidate(store, identity=PROXY, proxy=value, source_record=source, candidate_key=key)


def test_proxy_units_differ_from_independently_verified_cost_and_bind_artifact(tmp_path):
    store = Store(tmp_path / 's')
    exp = create_experiment(store, manifest(tmp_path))
    full = retained_finish(store, exp, resource=3)
    c = cheap(store, 'x', .25)
    pair = store.get(record_screening_pair(store, candidate=c, full=9, full_record=full, metric='cost'))
    assert pair['full_unit'] == 'work2'
    assert pair['identity']['unit'] == 'proxy-error'
    with pytest.raises(StrataError, match='same artifact'):
        record_screening_pair(store, candidate=cheap(store, 'bad', .25, b'other'), full=9, full_record=full, metric='cost')
    with pytest.raises(StrataError, match='value/unit'):
        record_screening_pair(store, candidate=c, full=3, full_record=full, metric='cost')
    with pytest.raises(StrataError, match='artifact identity'):
        record_screening_pair(store, candidate=c, full=9, full_record=full, metric='cost', artifact_identity='wrong')


def test_screen_can_pair_complete_aggregate_not_single_lucky_draw(tmp_path):
    store = Store(tmp_path / 's')
    path = manifest(tmp_path, draw=True, policy={'aggregate': 'mean', 'draws': [{'draw': 0}, {'draw': 2}]})
    a, b = [create_experiment(store, path, {'draw': d}) for d in (0, 2)]
    first = retained_finish(store, a, resource=2)
    c = cheap(store, 'x', .25)
    with pytest.raises(StrataError, match='complete draw aggregate'):
        record_screening_pair(store, candidate=c, full=4, full_record=first, metric='cost')
    incomplete = experiment_report(store)
    with pytest.raises(StrataError, match='complete, consistent'):
        record_screening_pair(store, candidate=c, full=4, full_record=incomplete['record'], metric='cost')
    retained_finish(store, b, resource=4)
    report = experiment_report(store)
    paired = store.get(record_screening_pair(store, candidate=c, full=10, full_record=report['record'], metric='cost'))
    assert paired['full'] == 10
    # The retained report cannot hide a subsequent conflicting measurement.
    retained_finish(store, b, resource=5)
    with pytest.raises(StrataError, match='complete, consistent'):
        record_screening_pair(store, candidate=c, full=10, full_record=report['record'], metric='cost')


def test_direct_full_finish_cannot_hide_a_conflicting_scientific_rejection(tmp_path):
    store = Store(tmp_path / 's')
    exp = create_experiment(store, manifest(tmp_path))
    full = retained_finish(store, exp, resource=3)
    retained_finish(store, exp, accepted=False)
    with pytest.raises(StrataError, match='conflicting or incomplete'):
        record_screening_pair(store, candidate=cheap(store, 'x', .25), full=9, full_record=full, metric='cost')


def test_worst_full_aggregate_keeps_maximization_direction(tmp_path):
    store = Store(tmp_path / 's')
    path = manifest(tmp_path, draw=True, policy={'aggregate': 'worst', 'draws': [{'draw': 0}, {'draw': 2}]})
    for d in (0, 2):
        retained_finish(store, create_experiment(store, path, {'draw': d}), resource=d+2)
    report = experiment_report(store, payload={'direction': 'max'})
    c = cheap(store, 'x', .25)
    paired = record_screening_pair(store, candidate=c, full=4, full_record=report['record'], metric='cost')
    assert store.get(paired)['aggregation_direction'] == 'max'
    p = ScreeningPolicy.create(store, identity=PROXY, proxy_threshold=1, full_threshold=5, direction='min')
    with pytest.raises(StrataError, match='direction differs'):
        p.calibrate([paired], [])


def test_insufficient_calibration_stays_advisory_with_new_units(tmp_path):
    store = Store(tmp_path / 's')
    exp = create_experiment(store, manifest(tmp_path))
    full = retained_finish(store, exp, resource=3)
    c = cheap(store, 'x', .25)
    pair = record_screening_pair(store, candidate=c, full=9, full_record=full, metric='cost')
    policy = ScreeningPolicy.create(store, identity=PROXY, proxy_threshold=1, full_threshold=10, full_unit='work2')
    cal = policy.calibrate([pair], [])
    assert cal['status'] == 'insufficient_calibration'
    assert cal['full_unit'] == 'work2'
    assert policy.decide(c, calibration=cal['record'], identity=PROXY)['action'] == 'abstain'
    wrong = ScreeningPolicy.create(store, identity=PROXY, proxy_threshold=1, full_threshold=10, full_unit='wrong')
    with pytest.raises(StrataError, match='unit drift'):
        wrong.calibrate([pair], [])


def test_later_rejection_invalidates_retained_calibration_and_decisions(tmp_path):
    store = Store(tmp_path / 's')
    pairs, experiments = [], []
    for i in range(4):
        artifact = str(i).encode()
        exp = create_experiment(store, manifest(tmp_path, f'full-{i}'))
        resource = 2 if i % 2 == 0 else 4
        full = retained_finish(store, exp, resource=resource, artifact=artifact)
        candidate = cheap(store, f'c{i}', .2 if i % 2 == 0 else 2, artifact)
        pairs.append(record_screening_pair(store, candidate=candidate, full=resource**2,
                                           full_record=full, metric='cost'))
        experiments.append(exp)
    p = ScreeningPolicy.create(store, identity=PROXY, proxy_threshold=1, full_threshold=10,
                               boundary_margin=10, min_samples=2, max_false_negative_rate=.99)
    cal = p.calibrate(pairs[:2], pairs[2:])
    assert cal['status'] == 'passed'
    alias = cheap(store, 'renamed-calibration-artifact', 2, b'1')
    assert p.decide(alias, calibration=cal['record'], identity=PROXY)['action'] == 'abstain'
    future = cheap(store, 'future', 2, b'future')
    assert p.decide(future, calibration=cal['record'], identity=PROXY)['action'] in {'prune', 'audit_full'}
    retained_finish(store, experiments[0], accepted=False, artifact=b'0')
    with pytest.raises(StrataError, match='conflicting or incomplete'):
        p.calibrate(pairs[:2], pairs[2:])
    blocked = p.decide(future, calibration=cal['record'], identity=PROXY)
    assert blocked['action'] == 'abstain'
    assert 'no longer valid' in blocked['reason']


def test_legacy_verified_metric_pairs_are_revalidated_after_rejection(tmp_path):
    store = Store(tmp_path / 's')
    pairs, experiments = [], []
    for i in range(4):
        path = manifest(tmp_path, f'old-{i}')
        raw = json.loads(path.read_text())
        raw['metrics'] = {'loss': dict(output='candidate', selector='', unit=PROXY['unit'])}
        raw['verifier']['metrics'] = {'loss': 'loss'}
        path.write_text(json.dumps(raw))
        exp = create_experiment(store, path)
        e = store.get(exp)
        pin = verifier_identity(e['manifest'], e)
        value = 4 if i % 2 == 0 else 16
        full = store.put('experiment_finish', dict(experiment=exp, status='accepted_by_verifier', seconds=1,
            artifacts={'candidate': {'sha256': store.blob(str(i).encode())}},
            metrics={'loss': dict(value=value, unit=PROXY['unit'], evidence_kind='verified',
                                  source=dict(verifier_identity=pin, verifier_selector='loss'))},
            verifier=dict(identity=pin, identity_schema=IDENTITY_SCHEMA, returncode=0,
                          receipt={'accepted': True, 'loss': value})))
        c = cheap(store, f'old-{i}', .2 if i % 2 == 0 else 2, str(i).encode())
        row = store.get(c)
        pairs.append(store.put('screening_pair', dict(version=1, candidate=c, identity=PROXY,
            candidate_key=row['candidate_key'], proxy=row['proxy'], full=value, full_record=full,
            source_record=row['source_record'], metric='loss', evidence_kind='measured')))
        experiments.append(exp)
    p = ScreeningPolicy.create(store, identity=PROXY, proxy_threshold=1, full_threshold=10,
                               boundary_margin=10, min_samples=2, max_false_negative_rate=.99)
    cal = p.calibrate(pairs[:2], pairs[2:])
    assert cal['status'] == 'passed'
    retained_finish(store, experiments[0], accepted=False, artifact=b'0')
    with pytest.raises(StrataError, match='conflicting or incomplete'):
        p.calibrate(pairs[:2], pairs[2:])
    assert p.decide(cheap(store, 'future', 2, b'future'), calibration=cal['record'], identity=PROXY)['action'] == 'abstain'


def test_python_module_entrypoint_and_comparison_flag():
    env = dict(os.environ, PYTHONPATH='src')
    r = subprocess.run([sys.executable, '-m', 'strata', 'experiment-report', '--help'],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert '--comparison-identity' in r.stdout
