"""Connect cheap measurements to artifact-bound independent objectives."""
from __future__ import annotations

from .model import StrataError


def candidate_artifact(store, source, artifact=None):
    artifacts = source.get('artifacts', {})
    if artifact is None:
        if len(artifacts) != 1:
            return None
        item = next(iter(artifacts.values()))
    elif artifact in artifacts:
        item = artifacts[artifact]
    else:
        item = next((a for a in artifacts.values() if a.get('sha256') == artifact), None)
        if item is None:
            raise StrataError('screening artifact is absent from the cheap receipt')
    store.verify_blob(item['sha256'])
    return item['sha256']


def pair_artifact(store, pair):
    return pair.get('artifact_sha256') or candidate_artifact(store, store.get(pair['source_record']))


def _verified_artifacts(store, finish):
    exp = store.get(finish['experiment'], 'experiment')
    manifest = exp['manifest']
    if isinstance(manifest, str):
        manifest = store.get(manifest)
    names = {a[len('{artifact:'):-1] for a in manifest['verifier']['argv']
             if a.startswith('{artifact:') and a.endswith('}')}
    return {finish['artifacts'][n]['sha256'] for n in names if n in finish.get('artifacts', {})}


def independent_objective(store, record, *, metric='cost', comparison_identity=None,
                          artifact_identity=None):
    """Revalidate a finish or complete aggregate; do not trust a caller's number."""
    from .experiment_reports import (_independently_accepted, _verified_objective,
                                     experiment_report, _manifest)
    from .experiment_comparisons import comparison_identity as identity_of, artifact_identity as artifact_of
    kind = store.db.execute('SELECT kind FROM records WHERE id=?', (record,)).fetchone()
    if kind is None or kind[0] not in {'experiment_finish', 'experiment_report'}:
        return None
    source = store.get(record)
    if kind[0] == 'experiment_finish':
        exp = store.get(source['experiment'], 'experiment')
        manifest = _manifest(store, exp)
        if not manifest.get('verifier') and metric != 'cost':
            return None  # Existing measured full-stage metrics retain their scope.
        if not _independently_accepted(store, source):
            raise StrataError('full evaluation requires an independently accepted receipt')
        if manifest.get('draw_policy'):
            raise StrataError('full evaluation requires a complete draw aggregate, not an individual draw')
        evidence = _verified_objective(store, source, metric)
        if not evidence:
            raise StrataError('full objective is not independently verified')
        identity = identity_of(manifest, exp, metric, evidence['unit'])
        if comparison_identity is not None and comparison_identity != identity:
            raise StrataError('full comparison identity differs')
        actual_artifact = artifact_of(manifest, source['artifacts'])
        if artifact_identity is not None and artifact_identity != actual_artifact:
            raise StrataError('full artifact identity differs')
        current = experiment_report(store, payload={'metric': metric})
        group = next((a for a in current['artifact_comparisons'] if
                      a['comparison_identity'] == identity and
                      a['artifact_identity'] == artifact_of(manifest, source['artifacts'])), None)
        if group is None or not group['comparison_eligible'] or group['value'] != evidence['value']:
            raise StrataError('full observation has conflicting or incomplete independent evidence')
        return dict(value=evidence['value'], unit=evidence['unit'],
                    comparison_identity=identity, artifact_identity=group['artifact_identity'],
                    artifacts=_verified_artifacts(store, source),
                    evidence_kind='verified')
    # Recompute from immutable observations and current correction records. An
    # old report must not hide a subsequently retained conflicting draw.
    direction = source.get('direction', 'min')
    current = experiment_report(store, payload={'metric': metric, 'direction': direction})
    rows = [a for a in source['artifact_comparisons'] if a['metric'] == metric and
            (comparison_identity is None or a['comparison_identity'] == comparison_identity) and
            (artifact_identity is None or a['artifact_identity'] == artifact_identity)]
    if len(rows) != 1:
        raise StrataError('select one full comparison_identity and artifact_identity')
    chosen = rows[0]
    actual = next((a for a in current['artifact_comparisons'] if
                   a['comparison_identity'] == chosen['comparison_identity'] and
                   a['artifact_identity'] == chosen['artifact_identity']), None)
    if actual is None or not actual['comparison_eligible']:
        raise StrataError('full evaluation requires a complete, consistent verified aggregate')
    artifacts = None
    for observation in actual['observations']:
        if observation.get('disposition', 'observation') != 'observation':
            continue
        hashes = _verified_artifacts(store, store.get(observation['result'], 'experiment_finish'))
        artifacts = hashes if artifacts is None else artifacts & hashes
    return dict(value=actual['value'], unit=actual['unit'],
                comparison_identity=actual['comparison_identity'], artifact_identity=actual['artifact_identity'],
                artifacts=artifacts or set(),
                aggregation_direction=direction if actual['aggregate'] == 'worst' else None,
                evidence_kind='verified')


def revalidate_pair(store, pair):
    """A retained pair cannot outlive a conflicting verdict on its full artifact."""
    if pair.get('version', 1) < 2:
        kind = store.db.execute('SELECT kind FROM records WHERE id=?', (pair['full_record'],)).fetchone()
        source = store.get(pair['full_record'])
        if not kind or kind[0] != 'experiment_finish' or not source.get('verifier'):
            return  # Historical measured-only evaluators retain their scope.
        current = independent_objective(store, pair['full_record'], metric=pair['metric'])
        cheap = store.get(pair['source_record'])
        artifact = pair.get('artifact_sha256') or candidate_artifact(store, cheap)
        if (current is None or current['value'] != pair['full'] or
            current['unit'] != pair['identity']['unit'] or not artifact or artifact not in current['artifacts']):
            raise StrataError('legacy screening pair no longer has compatible full evidence')
        return
    current = independent_objective(store, pair['full_record'], metric=pair['metric'],
                                   comparison_identity=pair['full_comparison_identity'],
                                   artifact_identity=pair.get('full_artifact_identity'))
    if (current is None or current['value'] != pair['full'] or
        current['unit'] != pair['full_unit'] or
        current['comparison_identity'] != pair['full_comparison_identity'] or
        current.get('aggregation_direction') != pair.get('aggregation_direction') or
        pair.get('artifact_sha256') not in current['artifacts']):
        raise StrataError('retained screening pair no longer has compatible full evidence')
