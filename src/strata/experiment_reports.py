"""Deterministic reports and agent packets over immutable experiment receipts."""
from __future__ import annotations

import math

from .model import StrataError, digest


def _records(store, kind, experiment=None):
    return [dict(record=h, **row) for h, row in store.all(kind)
            if experiment is None or row.get('experiment') == experiment]


def _manifest(store, experiment):
    value = experiment['manifest']
    return store.get(value) if isinstance(value, str) else value


def _select(value, selector):
    # Share JSON Pointer escaping, array "last", empty selectors and path lists
    # with the reader that produced the independent verifier receipt.
    from .experiments import _selector
    return _selector(value, selector)


def _board_projection(store, experiment=None):
    """Rebuild a deterministic compact view without touching live process state."""
    jobs = {}; events = _records(store, 'experiment_board_event')
    for event in events:
        job = jobs.setdefault(event['job_id'], dict(attempt=event['job_id'], status='unknown', receipts=[]))
        job['receipts'].append(event['record'])
        kind = event['type']
        if kind == 'enqueued':
            job.update(experiment=event['experiment_id'], status='queued', resources=event['resources'])
        elif kind == 'claimed':
            job.update(owner=event['owner'], status='claimed', lease_until=event['lease_until'],
                       granted_slots=event['granted_slots'])
        elif kind == 'process_bound':
            job.update(status='running', process=event['process'])
        elif kind in ('finished', 'reconciled'):
            job.update(status=event['status'], result_record=event.get('result_handle'))
        elif kind == 'cancelled':
            job['status'] = 'cancelled'
        elif kind == 'cancel_requested':
            job['cancel_requested'] = True
    rows = [row for row in jobs.values() if experiment is None or row.get('experiment') == experiment]
    capacities = _records(store, 'experiment_board_capacity')
    return dict(jobs=rows, active_jobs=[r for r in rows if r['status'] in ('queued', 'claimed', 'running', 'unreconciled')],
        capacity=capacities[-1]['capacity'] if capacities else None,
        reserved={name:sum(r.get('resources', {}).get(name, 0) for r in rows
                    if r['status'] in ('claimed', 'running', 'unreconciled'))
                  for name in ('threads', 'memory_bytes', 'storage_bytes')},
        event_count=len(events), evidence_kind='measured',
        scope='projection of retained cooperative board events; live lease expiry is not inferred')


def _analyses(store, payload):
    result = {}
    if 'error_budget' in payload:
        from .numeric import error_budget
        result['error_budget'] = error_budget(store=store, **payload['error_budget'])
    if 'score_bets' in payload:
        from .numeric_jobs import score_bets
        result['score_bets'] = score_bets(store=store, **payload['score_bets'])
    if 'design_space' in payload:
        # design_space can execute caller callbacks; reporting consumes its
        # already retained map instead and never reruns an experiment.
        record = payload['design_space']
        if not isinstance(record, str):
            raise StrataError('design_space report input must name a retained result record')
        value = store.get(record)
        unsigned = dict(value); map_hash = unsigned.pop('map_hash', None)
        if map_hash is None or map_hash != digest(unsigned):
            raise StrataError('retained design-space map hash mismatch')
        result['design_space'] = dict(record=record, **value)
    return result


def _independently_accepted(store, finish):
    if finish.get('status') != 'accepted_by_verifier':
        return False
    verifier = finish.get('verifier')
    if isinstance(verifier, str):
        verifier = store.get(verifier)
    if not isinstance(verifier, dict):
        return False
    experiment = store.get(finish['experiment'], 'experiment')
    manifest = _manifest(store, experiment)
    pinned = manifest.get('verifier')
    if not isinstance(pinned, dict):
        return False
    from .experiment_comparisons import IDENTITY_SCHEMA, verifier_identity
    schema = verifier.get('identity_schema')
    expected = verifier_identity(manifest, experiment) if schema == IDENTITY_SCHEMA else digest(pinned)
    if schema not in (None, IDENTITY_SCHEMA) or verifier.get('identity') != expected:
        return False
    accepted = verifier.get('receipt')
    selector = pinned.get('accepted_selector', 'accepted')
    if not isinstance(selector, (str, list)):
        return False
    try:
        accepted = _select(accepted, selector)
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    if accepted is not True or verifier.get('returncode') != 0:
        return False
    artifacts = finish.get('artifacts', {})
    if not artifacts:
        return False
    for artifact in artifacts.values():
        store.verify_blob(artifact['sha256'])
    return True


def _verified_objective(store, finish, metric):
    """Boolean artifact acceptance does not validate worker-reported resources."""
    experiment = store.get(finish['experiment'], 'experiment'); manifest = _manifest(store, experiment)
    verifier = finish.get('verifier')
    if isinstance(verifier, str):
        verifier = store.get(verifier)
    try:
        if metric == 'cost' and finish.get('verified_cost'):
            from .cost import Formula
            cost = finish['verified_cost']; declared = manifest['cost']
            if (cost.get('evidence_kind') != 'verified' or cost.get('verifier_identity') != verifier['identity'] or
                cost.get('expression') != declared['expression'] or cost.get('unit') != declared['unit'] or
                cost.get('resource_selectors') != declared.get('resources', {})):
                return None
            values = dict(experiment.get('settings', {}))
            for name, selector in cost['resource_selectors'].items():
                values[name] = _select(verifier['receipt'], selector)
            actual = Formula(cost['expression'], functions={'ceil':math.ceil, 'floor':math.floor})(values)
            if actual != cost['value']:
                return None
            return cost
        evidence = finish.get('metrics', {}).get(metric)
        if not evidence or evidence.get('evidence_kind') != 'verified':
            return None
        declared_unit = manifest.get('metrics', {}).get(metric, {}).get('unit')
        if declared_unit is not None and evidence.get('unit') != declared_unit:
            return None
        source = evidence.get('source', {})
        selector = manifest['verifier']['metrics'][metric]
        if (not isinstance(source, dict) or source.get('verifier_identity') != verifier['identity'] or
            source.get('verifier_selector', source.get('selector')) != selector or
            _select(verifier['receipt'], selector) != evidence['value']):
            return None
        return evidence
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return None


def experiment_report(store, experiment=None, board=None, payload=None):
    """Generate JSON-compatible results solely from retained evidence.

    ``payload`` optionally supplies an error_budget specification, a retained
    design_space result, score_bets arguments, metric/direction preferences and
    research annotations. These inputs are themselves retained for regeneration.
    """
    if experiment is not None:
        store.get(experiment, 'experiment')
    if isinstance(payload, str):
        payload_id = payload; payload = store.get(payload, 'experiment_report_payload')
    else:
        payload = {} if payload is None else payload
        if not isinstance(payload, dict):
            raise StrataError('report payload must be an object or retained payload handle')
        payload_id = store.put('experiment_report_payload', payload)
    direction = payload.get('direction', 'min')
    if direction not in ('min', 'max'):
        raise StrataError('report comparison direction must be min or max')
    experiments = [dict(record=h, **r) for h, r in store.all('experiment') if experiment is None or h == experiment]
    finishes = _records(store, 'experiment_finish', experiment)
    starts = _records(store, 'experiment_attempt', experiment)
    stages = _records(store, 'experiment_stage', experiment)
    metric_names = sorted({name for r in finishes for name in r.get('metrics', {})})
    comparisons = []
    by_id = {row['record']: row for row in experiments}
    for finish in finishes:
        measurements = {name: finish.get('metrics', {}).get(name) for name in metric_names}
        exp = by_id.get(finish['experiment'], {})
        comparisons.append(dict(experiment=finish['experiment'], attempt=finish.get('attempt'),
            result=finish['record'], status=finish['status'], settings=exp.get('settings', {}),
            manifest=digest(_manifest(store, exp)) if exp else None, metrics=measurements,
            numeric_analysis_available=any(v is not None for v in measurements.values()),
            independently_accepted=_independently_accepted(store, finish),
            seconds=finish.get('seconds'), artifacts=finish.get('artifacts', {}),
            failure=finish.get('error'), verifier=finish.get('verifier'), cost=finish.get('cost'),
            verified_cost=finish.get('verified_cost')))
    metric = payload.get('metric', 'cost' if 'cost' in metric_names or any(r.get('cost') for r in finishes)
                         else metric_names[0] if len(metric_names) == 1 else None)
    # Scientific comparison contracts exclude worker implementation details;
    # finite draw policies require complete artifact-bound matched observations.
    baselines, aggregates = [], []
    if metric is not None:
        from .experiment_comparisons import verified_comparisons
        baselines, aggregates = verified_comparisons(store, by_id, comparisons, finishes,
            metric, direction, _verified_objective)
    projection = _board_projection(store, experiment)
    # The optional argument is intentionally not read: the board is backed by
    # this Store and receipts make rendering independent of wall-clock time.
    numeric_jobs = []
    if experiment is None:
        from .numeric_jobs import NumericJob
        numeric_jobs = [dict(NumericJob(store, h).status()) for h, _ in store.all('numeric_job')]
    screening = _records(store, 'screening_decision')
    continuation = _records(store, 'continuation_decision')
    gates = _records(store, 'experiment_gate', experiment)
    if experiment is not None:
        references = set()
        for decision in [r.get('decision') for r in gates] + [d for f in finishes for d in f.get('policy_decisions', [])]:
            references.add(decision.get('record') if isinstance(decision, dict) else decision)
        screening = [r for r in screening if r['record'] in references]
        continuation = [r for r in continuation if r['record'] in references]
    audits = _records(store, 'screening_audit')
    if experiment is not None:
        references = {r['record'] for r in screening}
        audits = [r for r in audits if r['decision'] in references]
    reasons = [dict(experiment=r['experiment'], result=r['result'], status=r['status'], failure=r['failure'],
                    reopening_condition='changed hypothesis, settings or additional evidence; this execution is not a proof')
               for r in comparisons if r['status'] in ('failed', 'screened_out', 'budget_exhausted', 'paused_by_forecast_policy')]
    result = dict(version=1, experiment=experiment, payload=payload_id, experiments=experiments,
        attempts=starts, outcomes=comparisons, stages=stages, board=projection, numeric_jobs=numeric_jobs,
        metric=metric, direction=direction, verified_baselines=baselines, artifact_comparisons=aggregates,
        best_independently_accepted=baselines[0] if len(baselines) == 1 and
            len({a['comparison_identity'] for a in aggregates}) == 1 else None,
        modeled_bets=payload.get('modeled_bets', []), analyses=_analyses(store, payload),
        failures=reasons, screening=dict(decisions=screening, audits=audits,
            audited_candidates=len(audits), audited_false_negatives=sum(r['false_negative'] for r in audits),
            accepted_fraction_among_audited_rejections=sum(r['false_negative'] for r in audits)/len(audits) if audits else None,
            recommended_skips=sum(r['action']=='prune' for r in screening),
            scope='audit result rate among rejected candidates; recommendations are not measured compute savings'),
        continuation=dict(decisions=continuation, calibrations=_records(store, 'continuation_calibration')),
        hypotheses=payload.get('hypotheses', []), unresolved_questions=payload.get('unresolved_questions', []),
        agent_decisions=[r for r in _records(store,'agent_decision')
                         if experiment is None or r['decisive_experiment']==experiment],
        accounting=dict(completed_seconds=math.fsum(r['seconds'] for r in comparisons if r['seconds'] is not None),
            attempts_missing_seconds=sum(r['seconds'] is None for r in comparisons),
            scope='reported completed wall seconds; missing measurements excluded, never replaced with zero'),
        evidence_scope='measured receipts and labelled models; only independently accepted artifacts enter verified baselines')
    return dict(record=store.put('experiment_report', result), **result)


def render_report_markdown(report):
    lines = ['# Experiment report', '', f"Retained report: `{report.get('record', 'unsaved')}`", '',
             f"Completed attempts: {len(report['outcomes'])}. Active jobs: {len(report['board']['active_jobs'])}.", '']
    baseline = report['best_independently_accepted']
    if baseline:
        lines.append(f"Independently accepted baseline: {baseline['metric']} = {baseline['value']} {baseline['unit']} (verified).")
    elif report['verified_baselines']:
        lines.append('Independently accepted baselines have different verifier/objective contracts or precommitted draw sets; no cross-contract best is implied.')
    else:
        lines.append('Independently accepted baseline: unavailable.')
    for aggregate in report.get('artifact_comparisons', []):
        if aggregate['aggregate'] != 'deterministic' or aggregate['reasons']:
            lines += ['', f"Artifact `{aggregate['artifact_identity'][:12]}`: {aggregate['aggregate']} aggregate; "
                + (f"{aggregate['value']} {aggregate['unit']} (verified finite draws)." if aggregate['comparison_eligible']
                   else '; '.join(aggregate['reasons']) or 'all-draw vector retained; no scalar ranking declared.')]
    lines += ['', '| Experiment | Status | Metrics and evidence | Wall seconds |', '| --- | --- | --- | --- |']
    for row in report['outcomes']:
        metric_text = []
        for name, value in sorted(row['metrics'].items()):
            metric_text.append(f'{name}: unavailable' if value is None else
                f"{name}: {value.get('value', 'unavailable')} {value.get('unit', '')} ({value.get('evidence_kind', 'unknown')})")
        lines.append(f"| {row['experiment'][:12]} | {row['status']} | {'; '.join(metric_text) or 'numeric analysis unavailable'} | {row['seconds'] if row['seconds'] is not None else 'unavailable'} |")
    for row in report['failures']:
        lines += ['', f"Failure evidence `{row['result']}`: {row['failure'] or row['status']}. Reopen: {row['reopening_condition']}."]
    if report['modeled_bets'] or 'score_bets' in report['analyses']:
        lines += ['', 'Modelled bets remain separate from independently accepted baselines.']
    if report['screening']['decisions']:
        audit = report['screening']
        lines += ['', f"Cheap gate: {audit['recommended_skips']} recommended skips; {audit['audited_candidates']} full-evaluator audits with {audit['audited_false_negatives']} missed successes."]
    for calibration in report['continuation']['calibrations']:
        lines += ['', f"Continuation model `{calibration['policy'][:12]}`: {calibration['status']}; {calibration['lost_wins']} lost wins, {calibration['false_prunes']} false prunes, {calibration['compute_saved_seconds']} seconds saved in heldout trajectory replay (model estimate, not actual saved compute)."]
    for name, value in report['analyses'].items():
        if name == 'error_budget':
            lines += ['', f"Error budget: {value['total']} {value['unit']} against {value['limit']}; explicit cross terms are retained."]
        elif name == 'design_space':
            lines += ['', f"Design-space report: retained map `{value['record']}`; rows preserve their measured/model labels."]
        elif name == 'score_bets':
            lines += ['', f"Declared portfolio formula: `{value['formula']}`. Its supplied safe-score calculation is {value['safe_score']}; portfolio verification is separate."]
    lines += ['', 'Missing measurements remain unavailable. Screening and continuation forecasts are model evidence and never promote a candidate.', '']
    return '\n'.join(lines)


def _native_fingerprint(manifest):
    """Compare execution meaning, retaining byte identities rather than paths."""
    required = {'stages', 'metrics', 'runtimes', 'reader', 'environment', 'dependencies', 'inputs'}
    if not required <= set(manifest) or not manifest.get('verifier'):
        return None
    def sources(values):
        if not isinstance(values, dict) or any(not isinstance(v, dict) or not isinstance(v.get('files'), dict)
                or not v['files'] for v in values.values()):
            raise StrataError('transfer requires frozen dependency file identities')
        return {name:dict(files=row['files'], directory=row.get('directory', False)) for name,row in values.items()}
    verifier = dict(manifest['verifier'])
    verifier['dependencies'] = sources(verifier.get('dependencies', {}))
    # A default source descriptor duplicates the dependency paths. Its actual
    # content identities are already retained above; explicit strings stay pinned.
    if isinstance(verifier.get('source'), dict):
        verifier['source'] = sources(verifier['source'])
    inputs = sources(manifest['inputs'])
    transfer_inputs = manifest.get('compatibility',{}).get('transfer_inputs',[])
    if (not isinstance(transfer_inputs,list) or any(not isinstance(name,str) for name in transfer_inputs)
        or len(set(transfer_inputs)) != len(transfer_inputs) or not set(transfer_inputs) <= set(inputs)):
        raise StrataError('compatibility.transfer_inputs must list distinct declared input names')
    for name in transfer_inputs:
        inputs[name] = dict(transfer_content=True,directory=inputs[name]['directory'])
    return digest(dict(stages=manifest['stages'], metrics=manifest['metrics'],
        runtimes=manifest['runtimes'], reader=manifest['reader'], environment=manifest['environment'],
        dependencies=sources(manifest['dependencies']), inputs=inputs,
        verifier=verifier, cost=manifest.get('cost'), limits=manifest.get('limits'),
        resources=manifest.get('resources')))


def _native_transfers(store, target_handle, metric=None, direction='min', limit=32):
    target = store.get(target_handle, 'experiment'); target_manifest = _manifest(store, target)
    required = {'shape', 'units', 'parameter_meanings', 'dependencies'}
    compatibility = target_manifest.get('compatibility', {})
    if (not isinstance(compatibility, dict) or not required <= set(compatibility) or
        not target_manifest.get('line') or not target_manifest.get('contract')):
        raise StrataError('native transfers require manifest line, contract and compatibility shape, units, parameter_meanings, dependencies')
    fingerprint = _native_fingerprint(target_manifest)
    if fingerprint is None:
        raise StrataError('native transfer target requires a frozen pipeline and independent verifier')
    if metric is None:
        metric = 'cost' if target_manifest.get('cost') else next(iter(target_manifest['metrics'])) if len(target_manifest['metrics']) == 1 else None
    if metric is None:
        raise StrataError('native transfers require an explicit comparison metric')
    experiments = {h:r for h,r in store.all('experiment')}
    attempted = {r['experiment'] for _,r in store.all('experiment_attempt')} | {r['experiment'] for _,r in store.all('experiment_finish')}
    tested = {digest(r['settings']) for h,r in experiments.items() if h in attempted
              and _manifest(store,r).get('line') == target_manifest['line']}
    tested.add(digest(target['settings']))
    rejected = []; compatible = {}
    for handle,row in experiments.items():
        manifest = _manifest(store,row)
        if manifest.get('line') == target_manifest['line']:
            continue
        reasons = []
        if not manifest.get('line'):
            reasons.append('source line missing')
        if manifest.get('contract') != target_manifest['contract']:
            reasons.append('contract')
        other = manifest.get('compatibility', {})
        reasons.extend(key for key in required if not isinstance(other,dict) or other.get(key) != compatibility[key])
        if isinstance(other,dict) and sorted(other.get('transfer_inputs',[])) != sorted(compatibility.get('transfer_inputs',[])):
            reasons.append('explicit transferable input names')
        if set(row['settings']) != set(target['settings']):
            reasons.append('setting keys')
        if _native_fingerprint(manifest) != fingerprint:
            reasons.append('actual evaluator/dependency/verifier/runtime identity')
        verifier_args = (manifest.get('verifier') or {}).get('argv', [])
        if manifest.get('draw_policy') or any(arg.startswith('{setting:') or arg in ('{seed}', '{replicate}')
                                             for arg in verifier_args):
            reasons.append('verifier-varying transfers require complete artifact draw aggregates; individual draw gains are not transferable')
        if reasons:
            rejected.append(dict(experiment=handle,reasons=sorted(reasons)))
        else:
            compatible[handle] = (row,manifest)
    # Source wins are measured against that line's first compatible accepted
    # reference. Planned records and worker claims never establish a win.
    by_line = {}
    for handle,finish in store.all('experiment_finish'):
        if finish['experiment'] not in compatible:
            continue
        row,manifest = compatible[finish['experiment']]
        if not _independently_accepted(store,finish):
            rejected.append(dict(experiment=finish['experiment'],result=handle,reasons=['not independently accepted']))
            continue
        value = _verified_objective(store,finish,metric)
        if value is None:
            rejected.append(dict(experiment=finish['experiment'],result=handle,reasons=['objective not independently verified']))
            continue
        input_identity = digest({name:item['files'] for name,item in manifest['inputs'].items()})
        by_line.setdefault((manifest['line'],input_identity),[]).append(dict(experiment=finish['experiment'],result=handle,
            settings=row['settings'],identity=row.get('identity'),value=value['value'],unit=value['unit']))
    suggestions = []; seen = set()
    for (line,_),rows in sorted(by_line.items()):
        baseline = rows[0]
        for row in rows[1:]:
            if row['unit'] != baseline['unit'] or digest(row['settings']) == digest(baseline['settings']):
                continue
            better = row['value'] < baseline['value'] if direction == 'min' else row['value'] > baseline['value']
            if not better:
                continue
            key = digest(row['settings'])
            if key in tested:
                rejected.append(dict(experiment=row['experiment'],reasons=['settings already attempted or selected on target line']))
                continue
            if key in seen:
                continue
            seen.add(key)
            source_manifest = compatible[row['experiment']][1]
            source_inputs = {name:item['files'] for name,item in source_manifest['inputs'].items()}
            target_inputs = {name:item['files'] for name,item in target_manifest['inputs'].items()}
            suggestions.append(dict(settings=row['settings'],source_experiment=row['experiment'],source_result=row['result'],
                source_identity=row['identity'],source_line=line,source_baseline=baseline,
                target_experiment=target_handle,target_identity=target.get('identity'),target_line=target_manifest['line'],
                contract=target_manifest['contract'],compatibility=compatibility,execution_fingerprint=fingerprint,
                source_input_hashes=source_inputs,target_input_hashes=target_inputs,
                changed_inputs=sorted(name for name in source_inputs if source_inputs[name] != target_inputs[name]),
                observed_metric_delta=row['value']-baseline['value'],metric=metric,unit=row['unit'],
                evidence_kind='model',status='transfer_hypothesis',gain_established_on_target=False,
                reason='verified source improvement with matching declared compatibility and actual frozen evaluator; target still needs an experiment'))
    return dict(suggestions=suggestions[:limit],rejected=rejected,total_eligible=len(suggestions),truncated=len(suggestions)>limit,
                target_experiment=target_handle,scope='native retained source wins; settings transfer remains an untested hypothesis')


def agent_packet(store, experiment=None, payload=None, transfer_target=None):
    report = experiment_report(store, experiment=experiment, payload=payload)
    saved_payload = store.get(report['payload'], 'experiment_report_payload')
    target = transfer_target or saved_payload.get('transfer_target')
    transfer_limit = saved_payload.get('transfer_limit',(target or {}).get('limit',32))
    packet_limit = saved_payload.get('packet_limit',50)
    if type(transfer_limit) is not int or not 1<=transfer_limit<=256:
        raise StrataError('transfer_limit must be an integer from 1 through 256')
    if type(packet_limit) is not int or not 1<=packet_limit<=256:
        raise StrataError('packet_limit must be an integer from 1 through 256')
    transfers = dict(suggestions=[], rejected=[], scope='no explicit target compatibility contract supplied')
    if target:
        from .numeric import NumericStudy
        target = dict(target)
        target['limit'] = transfer_limit
        required = {'shape', 'units', 'parameter_meanings', 'dependencies'}
        factors = target.get('factors', {})
        if not required <= set(factors):
            raise StrataError('transfer target factors must declare shape, units, parameter_meanings and dependencies')
        if required & set(target.get('transfer_keys', [])):
            raise StrataError('shape, units, parameter meanings and dependencies cannot be relaxed for transfers')
        transfers = NumericStudy(store).suggest_transfers(**target)
        rejected = []
        for row in NumericStudy(store).experiments():
            mismatches = [key for key in required if row['factors'].get(key) != factors[key]]
            if row['contract'] != target['contract']:
                mismatches.append('contract')
            if mismatches:
                rejected.append(dict(experiment=row['handle'], reasons=sorted(mismatches)))
        transfers['rejected'] = rejected
    native_target = saved_payload.get('transfer_experiment')
    if native_target is None and experiment is not None:
        selected = store.get(experiment,'experiment')
        if _manifest(store,selected).get('compatibility'):
            native_target = experiment
    if native_target is not None:
        native = _native_transfers(store,native_target,metric=saved_payload.get('metric'),direction=report['direction'],limit=transfer_limit)
        if target:
            total = transfers['total_eligible']+native['total_eligible']
            transfers = dict(suggestions=(transfers['suggestions']+native['suggestions'])[:transfer_limit],
                rejected=transfers['rejected']+native['rejected'],numeric_study=transfers,native_experiments=native,
                total_eligible=total,truncated=total>transfer_limit,
                scope='retained source improvements; every proposed transfer needs independent target evaluation')
        else:
            transfers = native
    decisions = report['agent_decisions']
    experiments = {row['record'] for row in report['experiments']}
    prior = [r for r in decisions if experiment is None or r['decisive_experiment'] in experiments]
    body = dict(version=1, report=report['record'], verified_baselines=report['verified_baselines'],
        artifact_comparisons=report['artifact_comparisons'],
        best_independently_accepted=report['best_independently_accepted'], modeled_bets=report['modeled_bets'],
        tried=[dict(experiment=r['experiment'], result=r['result'], settings=r['settings'], status=r['status']) for r in report['outcomes']],
        failed_hypotheses=report['hypotheses'] + report['failures'],
        active_jobs=report['board']['active_jobs'], completed_comparisons=report['outcomes'],
        unresolved_questions=report['unresolved_questions'], compatible_untried_transfers=transfers,
        prior_agent_decisions=prior, scope='compact retained research state; transfers and bets remain hypotheses')
    fields = ('tried','completed_comparisons','prior_agent_decisions','verified_baselines','artifact_comparisons',
              'failed_hypotheses','unresolved_questions','modeled_bets')
    body['counts'] = {key:len(body[key]) for key in fields}
    body['counts']['active_jobs'] = len(body['active_jobs'])
    body['truncated'] = {key:len(body[key])>packet_limit for key in fields}
    body['packet_limit'] = packet_limit
    for key in fields:
        body[key] = body[key][-packet_limit:]
    # Rejection diagnostics are bounded too; every original evidence receipt
    # stays in the Store and the complete ordinary report remains addressable.
    transfers['rejected_total'] = len(transfers.get('rejected',[]))
    transfers['rejected_truncated'] = transfers['rejected_total']>packet_limit
    transfers['rejected'] = transfers.get('rejected',[])[-packet_limit:]
    # Do not duplicate nested unbounded suggestion/rejection collections.
    for key in ('numeric_study','native_experiments'):
        if key in transfers:
            detail = transfers[key]
            transfers[key] = {name:detail[name] for name in ('total_eligible','truncated','scope')}
    return dict(record=store.put('agent_packet', body), **body)


def record_agent_decision(store, *, packet, hypothesis, decisive_experiment,
                          expected_outcome, budget, reopening_condition, actor='agent'):
    previous = store.get(packet, 'agent_packet')
    store.get(decisive_experiment, 'experiment')
    for name, value in [('hypothesis', hypothesis), ('expected outcome', expected_outcome),
                        ('reopening condition', reopening_condition), ('actor', actor)]:
        if not isinstance(value, str) or not value.strip():
            raise StrataError(f'agent decision requires {name}')
    if not isinstance(budget, dict) or not budget or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in budget.values()):
        raise StrataError('agent decision requires positive finite named compute/resource budgets')
    body = dict(version=1, packet=packet, report=previous['report'], hypothesis=hypothesis,
        decisive_experiment=decisive_experiment, expected_outcome=expected_outcome, budget=budget,
        reopening_condition=reopening_condition, actor=actor, evidence_kind='judgment',
        scope='bounded proposed experiment; successful execution still requires independent acceptance')
    return dict(record=store.put('agent_decision', body), **body)
