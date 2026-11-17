"""Retained empirical screening and live-run continuation decisions.

Policies are immutable and deliberately conservative. Their risk estimates are
conditional on the declared population and independently held-out records; they
are neither acceptance certificates nor impossibility proofs.
"""
from __future__ import annotations

import math

from .model import StrataError, digest


def _number(value, name, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise StrataError(f'{name} must be finite')
    if minimum is not None and value < minimum:
        raise StrataError(f'{name} must be at least {minimum}')
    return float(value)


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise StrataError(f'{name} must be a positive integer')
    return value


def _identity(identity):
    if not isinstance(identity, dict) or not {'scorer', 'dependencies', 'domain', 'settings', 'unit'} <= set(identity):
        raise StrataError('policy identity needs scorer, dependencies, domain, settings and unit')
    if any(not isinstance(identity[k], str) or not identity[k] for k in ('scorer', 'unit')):
        raise StrataError('scorer and unit identities must be nonempty strings')
    if not isinstance(identity['settings'], dict) or not isinstance(identity['dependencies'], dict) or not identity['domain']:
        raise StrataError('explicit dependencies, domain and settings identities are required')
    digest(identity)
    return identity


def _metric(store, source_record, metric, value, unit, measured=False):
    """Require the value to occur in the immutable receipt, with its real label."""
    source = store.get(source_record)
    metrics = source.get('metrics', source.get('observations', {}))
    if metric is None:
        if len(metrics) == 1:
            metric = next(iter(metrics))
        elif 'value' in source:
            metrics = {'value': source}; metric = 'value'
        else:
            raise StrataError('metric selector required for a multi-metric receipt')
    if metric not in metrics:
        raise StrataError('metric is missing from retained receipt')
    evidence = metrics[metric]
    if not isinstance(evidence, dict):
        evidence = dict(value=evidence, unit=source.get('metric_units', {}).get(metric),
                        evidence_kind=source.get('evidence_kind'))
    if evidence.get('value') != value or evidence.get('unit') != unit:
        raise StrataError('policy value/unit differs from retained receipt')
    if measured and evidence.get('evidence_kind') not in ('measured', 'verified'):
        raise StrataError('full evaluations and trajectories require measured receipts')
    if evidence.get('evidence_kind') not in ('measured', 'verified', 'model', 'fitted_trend'):
        raise StrataError('policy input needs a measured or model evidence label')
    return metric, source


def record_screening_candidate(store, *, identity, proxy, source_record, candidate_key, metric=None):
    identity = _identity(identity); proxy = _number(proxy, 'proxy')
    if not isinstance(candidate_key, str) or not candidate_key:
        raise StrataError('candidate key is required')
    metric, source = _metric(store, source_record, metric, proxy, identity['unit'])
    if 'identity' in source and source['identity'] != identity:
        raise StrataError('candidate receipt identity differs')
    return store.put('screening_candidate', dict(version=1, identity=identity, proxy=proxy,
        source_record=source_record, candidate_key=candidate_key, metric=metric,
        evidence_kind='model', scope='cheap evaluator; full independent acceptance remains required'))


def record_screening_pair(store, *, candidate, full, full_record, metric=None):
    row = store.get(candidate, 'screening_candidate'); full = _number(full, 'full score')
    metric, source = _metric(store, full_record, metric, full, row['identity']['unit'], measured=True)
    if full_record == row['source_record']:
        raise StrataError('cheap and full evaluations need separate receipts')
    if 'identity' in source and source['identity'] != row['identity']:
        raise StrataError('full evaluator identity differs')
    if source.get('candidate_key', row['candidate_key']) != row['candidate_key']:
        raise StrataError('full evaluator scored a different candidate')
    cheap = store.get(row['source_record'])
    for field in ('experiment', 'candidate'):
        if field in source and field in cheap and source[field] != cheap[field]:
            raise StrataError('paired evaluations belong to different candidates/experiments')
    return store.put('screening_pair', dict(version=1, candidate=candidate,
        identity=row['identity'], candidate_key=row['candidate_key'], proxy=row['proxy'], full=full,
        full_record=full_record, source_record=row['source_record'], metric=metric,
        evidence_kind='measured', scope='paired observations on this candidate only'))


def _upper_binomial(misses, count, alpha):
    """One-sided exact binomial bound; independence is a declared assumption."""
    if count == 0 or misses == count:
        return 1.
    if misses == 0:
        return 1. - alpha ** (1. / count)
    def cdf(p):
        # Log-space terms avoid overflowing combinatorial coefficients.
        return math.fsum(math.exp(math.lgamma(count + 1) - math.lgamma(k + 1)
            - math.lgamma(count - k + 1) + k * math.log(p)
            + (count - k) * math.log1p(-p)) for k in range(misses + 1))
    lo, hi = misses / count, 1.
    for _ in range(60):
        mid = (lo + hi) / 2
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def _split(store, training, heldout, kind, identity, key):
    if not isinstance(training, (list, tuple)) or not isinstance(heldout, (list, tuple)):
        raise StrataError('training and heldout must be explicit record lists')
    rows = []
    keys = set(); sources = set()
    for handles in (training, heldout):
        part = []
        for handle in handles:
            row = store.get(handle, kind)
            if row['identity'] != identity:
                raise StrataError('calibration identity drift')
            if row[key] in keys:
                raise StrataError('training/heldout duplicate or leakage')
            keys.add(row[key])
            refs = ({row['source_record'], row['full_record']} if kind == 'screening_pair'
                    else {p['source_record'] for p in row['points']})
            if refs & sources:
                raise StrataError('training/heldout source receipt leakage')
            sources.update(refs)
            part.append(dict(record=handle, **row))
        rows.append(part)
    return rows


def _success(value, threshold, direction):
    return value <= threshold if direction == 'min' else value >= threshold


def _risk_options(direction, risk, alpha, min_samples):
    if direction not in ('min', 'max'):
        raise StrataError('direction must be min or max')
    if not 0 <= _number(risk, 'pruning risk') < 1 or not 0 < _number(alpha, 'alpha') < 1:
        raise StrataError('risk must be in [0,1), alpha in (0,1)')
    _positive_int(min_samples, 'minimum calibration samples')


class ScreeningPolicy:
    def __init__(self, store, handle):
        self.store = store; self.id = handle
        store.get(handle, 'screening_policy')

    @property
    def policy(self):
        return self.store.get(self.id, 'screening_policy')

    @classmethod
    def create(cls, store, *, identity, proxy_threshold, full_threshold,
               direction='min', boundary_margin=0., max_false_negative_rate=.05,
               min_samples=20, alpha=.05, audit_fraction=.1, audit_seed=0):
        _identity(identity); _risk_options(direction, max_false_negative_rate, alpha, min_samples)
        if not 0 < _number(audit_fraction, 'audit fraction') <= 1:
            raise StrataError('deterministic full-score audits require a fraction in (0,1]')
        if type(audit_seed) is not int:
            raise StrataError('audit seed must be an integer')
        body = dict(version=1, identity=identity, proxy_threshold=_number(proxy_threshold, 'proxy threshold'),
            full_threshold=_number(full_threshold, 'full threshold'), direction=direction,
            boundary_margin=_number(boundary_margin, 'boundary margin', 0),
            max_false_negative_rate=max_false_negative_rate, min_samples=min_samples,
            alpha=alpha, audit_fraction=audit_fraction, audit_seed=audit_seed,
            evidence_kind='model', risk_estimand='fraction of full-evaluator successes screened out',
            risk_assumption='independent heldout candidates representative of the declared domain')
        return cls(store, store.put('screening_policy', body))

    def calibrate(self, training, heldout):
        p = self.policy
        train, test = _split(self.store, training, heldout, 'screening_pair', p['identity'], 'candidate_key')
        reasons = []; coverage = {}
        for label, rows in [('training', train), ('heldout', test)]:
            counts = dict(successes=sum(_success(r['full'], p['full_threshold'], p['direction']) for r in rows),
                          failures=sum(not _success(r['full'], p['full_threshold'], p['direction']) for r in rows),
                          boundary=sum(abs(r['full'] - p['full_threshold']) <= p['boundary_margin'] for r in rows))
            coverage[label] = counts
            if len(rows) < p['min_samples'] or not all(counts.values()):
                reasons.append(f'{label} needs minimum samples, successes, failures and boundary cases')
        decisions = [dict(pair=r['record'], would_prune=not _success(r['proxy'], p['proxy_threshold'], p['direction']),
                          full_success=_success(r['full'], p['full_threshold'], p['direction'])) for r in test]
        successes = sum(r['full_success'] for r in decisions)
        misses = sum(r['would_prune'] and r['full_success'] for r in decisions)
        upper = _upper_binomial(misses, successes, p['alpha'])
        if upper > p['max_false_negative_rate']:
            reasons.append('heldout false-negative upper bound exceeds declared risk')
        body = dict(version=1, policy=self.id, identity=p['identity'], training=list(training), heldout=list(heldout),
            status='passed' if not reasons else 'insufficient_calibration', reasons=reasons, coverage=coverage,
            heldout_decisions=decisions, false_negatives=misses, successes=successes,
            false_negative_rate=misses / successes if successes else None, false_negative_upper_bound=upper,
            confidence=1 - p['alpha'], evidence_kind='model',
            scope='heldout empirical risk for frozen thresholds; no acceptance certificate')
        return dict(record=self.store.put('screening_calibration', body), **body)

    def decide(self, candidate, *, calibration=None, identity):
        p = self.policy; row = self.store.get(candidate, 'screening_candidate')
        action = 'abstain'; reason = 'calibration is absent'; cal = None
        if calibration is not None:
            cal = self.store.get(calibration, 'screening_calibration')
        if identity != p['identity'] or row['identity'] != p['identity']:
            reason = 'identity outside calibrated domain/settings/dependencies/units'
        elif cal is None:
            pass
        elif cal['policy'] != self.id or cal['identity'] != p['identity']:
            reason = 'calibration belongs to another policy or identity'
        elif cal['status'] != 'passed':
            reason = 'insufficient calibration'
        elif any(a['policy'] == self.id and a['false_negative'] and a['pair'] not in cal['training'] + cal['heldout']
                 for _, a in self.store.all('screening_audit')):
            reason = 'new audited false negative requires recalibration before further pruning'
        elif any(self.store.get(h, 'screening_pair')['candidate_key'] == row['candidate_key']
                 for h in cal['training'] + cal['heldout']):
            reason = 'candidate belongs to calibration data; independent evaluation required'
        elif _success(row['proxy'], p['proxy_threshold'], p['direction']):
            action = 'evaluate_full'; reason = 'proxy passed the cheap gate'
        else:
            draw = int(digest(dict(policy=self.id, seed=p['audit_seed'], candidate=row['candidate_key']))[:16], 16) / 2**64
            action = 'audit_full' if draw < p['audit_fraction'] else 'prune'
            reason = 'deterministic full-evaluator audit' if action == 'audit_full' else 'calibrated cheap gate'
        body = dict(version=1, policy=self.id, calibration=calibration, candidate=candidate,
            candidate_key=row['candidate_key'], action=action, reason=reason, evidence_kind='model',
            proxy=row['proxy'], identity=identity, acceptance=False,
            scope='screening decision, never proof of failure or candidate promotion')
        return dict(record=self.store.put('screening_decision', body), **body)

    def audit(self, decision, pair):
        d = self.store.get(decision, 'screening_decision'); r = self.store.get(pair, 'screening_pair'); p = self.policy
        if d['policy'] != self.id or d['candidate'] != r['candidate'] or d['action'] not in ('audit_full', 'prune'):
            raise StrataError('audit must bind this policy and its rejected candidate')
        body = dict(policy=self.id, decision=decision, pair=pair, evidence_kind='measured',
                    false_negative=_success(r['full'], p['full_threshold'], p['direction']),
                    scope='observed full result of a candidate selected for rejection')
        return dict(record=self.store.put('screening_audit', body), **body)


def record_trajectory(store, *, identity, run_id, points):
    _identity(identity)
    if not isinstance(run_id, str) or not run_id or not isinstance(points, (list, tuple)) or not points:
        raise StrataError('trajectory needs run identity and measured points')
    normalized = []
    for point in points:
        step = _number(point.get('step'), 'trajectory step', 0)
        value = _number(point.get('value'), 'trajectory value')
        compute = _number(point.get('compute_seconds'), 'cumulative compute seconds', 0)
        metric, source = _metric(store, point.get('source_record'), point.get('metric'), value, identity['unit'], True)
        if 'identity' in source and source['identity'] != identity:
            raise StrataError('trajectory receipt identity differs')
        if source.get('run_id', run_id) != run_id:
            raise StrataError('trajectory point belongs to another run')
        if normalized and (step <= normalized[-1]['step'] or compute < normalized[-1]['compute_seconds']):
            raise StrataError('steps must increase and cumulative compute cannot decrease')
        normalized.append(dict(step=step, value=value, compute_seconds=compute,
                               source_record=point['source_record'], metric=metric))
    return store.put('continuation_trajectory', dict(version=1, identity=identity, run_id=run_id,
        points=normalized, evidence_kind='measured', scope='retained measured trajectory; no inferred endpoints'))


class ContinuationPolicy:
    """Nearest measured prefix model for continuation of this particular run.

    Entire heldout trajectories replay the first proposed stop, including later
    wins. Training endpoints never enter heldout prefix features. Automation is
    available only after the heldout loss bound passes the predeclared risk.
    """
    def __init__(self, store, handle):
        self.store = store; self.id = handle
        store.get(handle, 'continuation_policy')

    @property
    def policy(self):
        return self.store.get(self.id, 'continuation_policy')

    @classmethod
    def create(cls, store, *, identity, target, horizon, cadence=1, min_evidence=3,
               min_samples=20, direction='min', max_false_negative_rate=.05,
               probability_threshold=.05, min_neighbors=3, max_distance=1.,
               feature_scale=1., alpha=.05, automation=False):
        _identity(identity); _risk_options(direction, max_false_negative_rate, alpha, min_samples)
        _positive_int(cadence, 'decision cadence'); _positive_int(min_evidence, 'minimum evidence')
        _positive_int(min_neighbors, 'minimum neighbors'); _positive_int(horizon, 'absolute horizon')
        if min_evidence < 2 or min_evidence >= horizon:
            raise StrataError('minimum evidence must be at least two and before horizon')
        if not 0 <= _number(probability_threshold, 'probability threshold') < 1:
            raise StrataError('probability threshold must be in [0,1)')
        if _number(feature_scale, 'feature scale') <= 0 or _number(max_distance, 'maximum neighbor distance') <= 0:
            raise StrataError('feature scale and maximum distance must be positive')
        if type(automation) is not bool:
            raise StrataError('automation must be explicit boolean')
        body = dict(version=1, identity=identity, target=_number(target, 'target'), horizon=horizon,
            cadence=cadence, min_evidence=min_evidence, min_samples=min_samples, direction=direction,
            max_false_negative_rate=max_false_negative_rate, probability_threshold=probability_threshold,
            min_neighbors=min_neighbors, max_distance=max_distance, feature_scale=feature_scale, alpha=alpha,
            automation=automation, model_identity='nearest-measured-prefix-v1', evidence_kind='model',
            decision_target='this live run reaches target after this prefix by the absolute measured horizon',
            risk_estimand='fraction of successful heldout trajectories lost by first policy pause',
            risk_assumption='independent heldout trajectories representative of the declared population')
        return cls(store, store.put('continuation_policy', body))

    def _features(self, points):
        p = self.policy; sign = 1 if p['direction'] == 'min' else -1
        values = [sign * row['value'] for row in points]
        return [(min(values) - sign * p['target']) / p['feature_scale'],
                (values[-1] - values[-2]) / p['feature_scale']]

    def _predict(self, points, train):
        p = self.policy; step = points[-1]['step']
        if len(points) < p['min_evidence'] or step >= p['horizon'] or step % p['cadence']:
            return dict(status='unsupported', reason='outside decision cadence or evidence/horizon limits')
        if any(_success(r['value'], p['target'], p['direction']) for r in points):
            return dict(status='reached_target', probability=1., would_pause=False)
        feature = self._features(points); neighbors = []
        for row in train:
            prefix = [r for r in row['points'] if r['step'] <= step]
            if len(prefix) < p['min_evidence'] or prefix[-1]['step'] != step:
                continue
            other = self._features(prefix)
            distance = math.sqrt(sum((a - b)**2 for a, b in zip(feature, other)))
            if distance <= p['max_distance']:
                success = any(_success(r['value'], p['target'], p['direction'])
                    for r in row['points'] if step < r['step'] <= p['horizon'])
                neighbors.append(dict(run_id=row['run_id'], distance=distance, success=success))
        neighbors.sort(key=lambda r: (r['distance'], r['run_id']))
        if len(neighbors) < p['min_neighbors']:
            return dict(status='unsupported', reason='too few measured neighboring trajectory prefixes')
        probability = sum(r['success'] for r in neighbors) / len(neighbors)
        return dict(status='supported', probability=probability, neighbors=neighbors,
                    would_pause=probability <= p['probability_threshold'],
                    uncertainty='empirical neighbor frequency; heldout policy-risk bound reported separately')

    def calibrate(self, training, heldout):
        p = self.policy
        train, test = _split(self.store, training, heldout, 'continuation_trajectory', p['identity'], 'run_id')
        reasons = []
        if min(len(train), len(test)) < p['min_samples']:
            reasons.append('insufficient independent training or heldout trajectories')
        if any(r['points'][-1]['step'] != p['horizon'] for r in train + test):
            raise StrataError('calibration trajectories must end at the declared measured horizon')
        outcomes = []; late_wins = 0
        for row in test:
            win = any(_success(r['value'], p['target'], p['direction']) for r in row['points'])
            pause = None; supported = 0; late = False
            for at in range(p['min_evidence'], len(row['points'])):
                prefix = row['points'][:at]
                later = row['points'][at:]
                if (prefix[-1]['value'] == prefix[-2]['value'] and
                    not any(_success(r['value'], p['target'], p['direction']) for r in prefix) and
                    any(_success(r['value'], p['target'], p['direction']) for r in later)):
                    late = True
                prediction = self._predict(prefix, train)
                supported += prediction['status'] == 'supported'
                if pause is None and prediction.get('would_pause'):
                    pause = prefix[-1]
            late_wins += late
            outcomes.append(dict(trajectory=row['record'], run_id=row['run_id'], full_success=win,
                late_win=late, supported_decisions=supported, pause_step=pause['step'] if pause else None,
                lost_win=bool(pause and win), compute_saved_seconds=(row['points'][-1]['compute_seconds']
                    - pause['compute_seconds']) if pause else 0.))
        wins = sum(r['full_success'] for r in outcomes); lost = sum(r['lost_win'] for r in outcomes)
        paused = sum(r['pause_step'] is not None for r in outcomes)
        upper = _upper_binomial(lost, wins, p['alpha'])
        if not wins or wins == len(outcomes) or not late_wins:
            reasons.append('heldout validation needs successful, failing and plateau-then-late-win trajectories')
        if not paused:
            reasons.append('no supported heldout pause decisions to validate')
        if upper > p['max_false_negative_rate']:
            reasons.append('heldout lost-win upper bound exceeds declared risk')
        body = dict(version=1, policy=self.id, identity=p['identity'], training=list(training), heldout=list(heldout),
            model_identity=p['model_identity'], decision_target=p['decision_target'],
            status='passed' if not reasons else 'insufficient_calibration', reasons=reasons, outcomes=outcomes,
            heldout_successes=wins, late_wins=late_wins, false_prunes=lost, lost_wins=lost,
            paused_trajectories=paused, false_prune_rate=lost / paused if paused else None,
            lost_win_rate=lost / wins if wins else None, lost_win_upper_bound=upper,
            compute_saved_seconds=math.fsum(r['compute_saved_seconds'] for r in outcomes),
            compute_scope='retrospective measured trajectory replay, not actual saved compute',
            confidence=1 - p['alpha'], evidence_kind='model', acceptance=False)
        return dict(record=self.store.put('continuation_calibration', body), **body)

    def decide(self, trajectory, *, calibration=None, identity):
        p = self.policy; row = self.store.get(trajectory, 'continuation_trajectory')
        action = 'abstain'; reason = 'calibration is absent'; prediction = None
        cal = self.store.get(calibration, 'continuation_calibration') if calibration else None
        if identity != p['identity'] or row['identity'] != p['identity']:
            reason = 'identity outside calibrated population'
        elif cal is None:
            pass
        elif cal['policy'] != self.id or cal['identity'] != p['identity']:
            reason = 'calibration belongs to another policy or identity'
        else:
            train = [self.store.get(h, 'continuation_trajectory') for h in cal['training']]
            known = train + [self.store.get(h, 'continuation_trajectory') for h in cal['heldout']]
            refs = {point['source_record'] for point in row['points']}
            if any(r['run_id'] == row['run_id'] or refs & {x['source_record'] for x in r['points']} for r in known):
                reason = 'live trajectory overlaps calibration evidence'
            else:
                prediction = self._predict(row['points'], train)
                if prediction['status'] == 'unsupported':
                    reason = prediction['reason']
                elif not prediction['would_pause']:
                    action = 'continue'; reason = 'model does not recommend pausing'
                elif cal['status'] != 'passed':
                    action = 'advisory_pause'; reason = 'insufficient calibration; automation withheld'
                else:
                    action = 'paused_by_forecast_policy' if p['automation'] else 'advisory_pause'
                    reason = 'heldout-calibrated continuation recommendation'
        body = dict(version=1, policy=self.id, calibration=calibration, trajectory=trajectory,
            run_id=row['run_id'], action=action, reason=reason, prediction=prediction,
            evidence_kind='model', acceptance=False, resumable=action == 'paused_by_forecast_policy',
            charged_compute_seconds=row['points'][-1]['compute_seconds'],
            scope='live-run continuation advice; no proof of failure and no promotion')
        return dict(record=self.store.put('continuation_decision', body), **body)


def policy_request(store, spec):
    """JSON-facing dispatcher shared by the CLI and programmatic agents."""
    spec = dict(spec); operation = spec.pop('operation', spec.pop('op', None))
    if operation == 'evaluator_identity':
        from .experiments import evaluator_identity
        manifest = spec.pop('manifest', None)
        if not isinstance(manifest, str) or not manifest:
            raise StrataError('evaluator_identity requires an existing manifest path')
        identity = evaluator_identity(manifest, **spec)
        body = dict(identity=identity, manifest=manifest,
                    scope='derived from actual manifest, declared source bytes, runtime, reader and environment')
        return dict(record=store.put('experiment_evaluator_identity', body), **body)
    if isinstance(spec.get('identity'), str):
        spec['identity'] = store.get(spec['identity'], 'experiment_evaluator_identity')['identity']
    if operation in ('create_screening', 'create_continuation'):
        cls = ScreeningPolicy if operation == 'create_screening' else ContinuationPolicy
        policy = cls.create(store, **spec)
        return dict(policy=policy.id, kind='screening' if cls is ScreeningPolicy else 'continuation', **policy.policy)
    if operation == 'record_candidate':
        return dict(record=record_screening_candidate(store, **spec))
    if operation == 'record_pair':
        return dict(record=record_screening_pair(store, **spec))
    if operation == 'record_trajectory':
        return dict(record=record_trajectory(store, **spec))
    handle = spec.pop('policy', None)
    kind = store.db.execute('SELECT kind FROM records WHERE id=?', (handle,)).fetchone()
    if not kind or kind[0] not in ('screening_policy', 'continuation_policy'):
        raise StrataError('unknown screening or continuation policy')
    policy = (ScreeningPolicy if kind[0] == 'screening_policy' else ContinuationPolicy)(store, handle)
    if operation == 'calibrate':
        return policy.calibrate(**spec)
    if operation == 'decide':
        return policy.decide(**spec)
    if operation == 'audit' and isinstance(policy, ScreeningPolicy):
        return policy.audit(**spec)
    raise StrataError('unknown policy operation')
