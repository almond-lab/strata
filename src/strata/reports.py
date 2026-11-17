"""Read-only research reports from immutable store records.

This module deliberately has no Project/adapter/front-end dependency: rendering
saved evidence must never build circuits, import generators, or rerun verifiers.
"""
from collections import Counter, defaultdict
import json
from .memory import checked_evidence
from .model import StrataError


KINDS = ('design', 'line', 'research_plan', 'research_result', 'gap_snapshot',
         'negative', 'unbound_attempt', 'file_candidate_attempt', 'agent_research_note',
         'file_optimization', 'discovery_target', 'semantic_recovery_target', 'discovery_brief',
         'quarantine','external_check_failure','restructure_run',
         'numeric_candidate','numeric_handoff','numeric_score','numeric_scorer_calibration',
         'numeric_experiment','numeric_pipeline','numeric_pipeline_checkpoint','numeric_pipeline_failure',
         'numeric_job','numeric_job_start','numeric_job_finish','numeric_checkpoint',
         'numeric_budget','numeric_forecast','numeric_floor','numeric_bets')
FILE_KINDS = {'file_candidate_attempt', 'agent_research_note', 'file_optimization',
              'discovery_target', 'semantic_recovery_target', 'discovery_brief','restructure_run'}


def _file_parent_hashes(record):
    """Read parent aliases used by saved file formats without guessing identity."""
    values = [record.get(key) for key in ('parent_sha256', 'input_sha256', 'original_sha256')]
    for key in ('baseline', 'original', 'input'):
        value = record.get(key)
        values.append(value.get('sha256') if isinstance(value, dict) else value)
    return {value for value in values if isinstance(value, str) and value}


def _blob_status(store, handle):
    if not handle:
        return 'unknown'
    try:
        store.read_blob(handle)
    except (OSError, StrataError):
        return 'missing_or_invalid'
    return 'hash_verified'


def report(store, handle=None, design=None, round=None):
    """Return JSON-ready saved evidence, optionally scoped to one handle.

    ``round`` accepts a research plan or result handle. ``design`` accepts a
    design handle. ``handle`` additionally accepts a Line, file attempt, note,
    discovery record, or parent circuit hash. Missing gap snapshots stay unknown.
    """
    if sum(value is not None for value in (handle, design, round)) > 1:
        raise StrataError('choose one report handle, design, or round')
    records = {kind: store.all(kind) for kind in KINDS}
    index = {h: (kind, value) for kind, rows in records.items() for h, value in rows}
    selected = handle or design or round
    selected_kind, selected_record = index.get(selected, (None, None))
    if design is not None and selected_kind != 'design':
        raise StrataError('report design must name a stored design')
    if round is not None and selected_kind not in {'research_plan', 'research_result'}:
        raise StrataError('report round must name a stored plan or result')
    file_hashes = set()
    if selected is not None and selected_kind is None:
        if any(selected in _file_parent_hashes(value) for kind in FILE_KINDS for _, value in records[kind]):
            file_hashes.add(selected)
        else:
            raise StrataError('unknown report handle: ' + str(selected))
    plan_ids, result_ids, design_ids, line_ids = set(), set(), set(), set()
    if selected_kind == 'research_plan':
        plan_ids.add(selected)
    elif selected_kind == 'research_result':
        result_ids.add(selected)
        plan_ids.add(selected_record.get('plan'))
    elif selected_kind == 'line':
        line_ids.add(selected)
        design_ids.add(selected_record.get('parent'))
    elif selected_kind == 'design':
        design_ids.add(selected)
        plan_ids.update(h for h, value in records['research_plan'] if value.get('parent') == selected)
        line_ids.update(h for h, value in records['line'] if value.get('parent') == selected)
        for h, value in records['research_result']:
            if value.get('parent') == selected or any(item.get('design') == selected for item in value.get('outcomes', [])):
                result_ids.add(h)
                plan_ids.add(value.get('plan'))
        # A verified design can also have been produced outside ResearchRound.
        for h, _ in records['line']:
            if any(event.get('design') == selected for event in store.events(h)):
                line_ids.add(h)
                design_ids.add(index[h][1].get('parent'))
    elif selected_kind in FILE_KINDS:
        file_hashes = _file_parent_hashes(selected_record)
    result_ids.update(h for h, value in records['research_result'] if value.get('plan') in plan_ids)
    for h, value in records['research_result']:
        if h in result_ids:
            design_ids.add(value.get('parent'))
            for outcome in value.get('outcomes', []):
                line_ids.add(outcome.get('line'))
                design_ids.add(outcome.get('design'))
    for h, value in records['research_plan']:
        if h in plan_ids:
            design_ids.add(value.get('parent'))
    for h, value in records['design']:
        if value.get('parent') in design_ids and selected_kind == 'design':
            # One generation is reported; no recursive walk or regeneration.
            if value.get('parent') == selected:
                design_ids.add(h)
    if selected_kind == 'line':
        design_ids.update(e.get('design') for e in store.events(selected) if e.get('stage') == 'verified')
    def included(kind, h, value):
        if selected is None:
            return True
        if kind == 'design': return h in design_ids
        if kind == 'line': return h in line_ids
        if kind == 'research_plan': return h in plan_ids
        if kind == 'research_result': return h in result_ids
        if kind in FILE_KINDS:
            return h == selected or bool(file_hashes & _file_parent_hashes(value))
        if kind=='external_check_failure':return h==selected or value.get('line') in line_ids
        if kind.startswith('numeric_'):
            return h==selected or value.get('candidate')==selected or value.get('job')==selected or value.get('pipeline')==selected
        return False
    scoped = {kind: [(h, v) for h, v in rows if included(kind, h, v)] for kind, rows in records.items()}
    plans = [dict(handle=h, **v) for h, v in scoped['research_plan']]
    results = [dict(handle=h, **v) for h, v in scoped['research_result']]
    outcomes = {item['line']: item for value in results for item in value.get('outcomes', []) if item.get('line')}
    contracts = set()
    for _, value in scoped['line']:
        contracts.update(value.get('patch', {}).get('contracts', []))
    # Read contract fingerprints from the saved tree only. No adapter is loaded.
    from .model import region_from_dict
    for _, value in scoped['design']:
        if value.get('tree'):
            contracts.update(r.contract.fingerprint for r in region_from_dict(value['tree']).walk())
    lines = []
    for h, value in scoped['line']:
        events = store.events(h)
        patch = value.get('patch', {})
        built = next((event for event in reversed(events) if event.get('stage') == 'build_price' and event.get('passed')), {})
        outcome = outcomes.get(h, {})
        lines.append(dict(handle=h, parent=value.get('parent'), task=value.get('task'),
                          move=patch.get('move'), mechanism=outcome.get('mechanism', patch.get('move')),
                          scope={key: patch.get(key) for key in ('regions', 'contracts', 'occurrences')},
                          status=outcome.get('status', events[-1].get('stage') if events else 'not_run'),
                          measured_delta=outcome.get('cost_delta', outcome.get('measured_delta', built.get('delta'))),
                          measured_delta_scope='saved executed build/verification' if built else 'unknown',
                          artifact=built.get('artifact'), artifact_status=_blob_status(store, built.get('artifact')),
                          events=events, reopen=[event.get('reopen') for event in events if event.get('reopen')]))
    negatives = []
    for h, value in records['negative']:
        if selected is not None and value.get('contract') not in contracts:
            continue
        kind, evidence_status = checked_evidence(store, value)
        negatives.append(dict(handle=h, contract=value.get('contract'), move=value.get('move'),
                              evidence_kind=kind, requested_evidence_kind=value.get('requested_evidence_kind', value.get('evidence_kind')),
                              evidence=value.get('evidence'), evidence_status=evidence_status,
                              status='priced' if kind == 'estimate' else 'closed_at_size' if kind == 'exact_small' else 'closed',
                              scale=value.get('scale'), full_size_open=kind == 'exact_small',
                              result=value.get('result'), reopen=value.get('reopen'),
                              closure_scope='this construction only' if kind == 'build' else 'recorded contract and scale'))
    designs = []
    gaps = []
    for h, value in scoped['design']:
        certificate = None
        certificate_status = 'unknown'
        if value.get('certificate'):
            try:
                saved = store.get(value['certificate'], 'certificate')
            except StrataError:
                certificate_status = 'missing_or_invalid'
            else:
                certificate_status = 'saved_record'
                certificate = {key: saved.get(key) for key in ('passed', 'artifact', 'verifier_hash', 'failure_model')}
        designs.append(dict(handle=h, parent=value.get('parent'), task=value.get('task'),
                            cost=value.get('cost'), resources=value.get('resources'),
                            artifact=value.get('artifact'), artifact_status=_blob_status(store, value.get('artifact')),
                            certificate_handle=value.get('certificate'), certificate=certificate,
                            certificate_status=certificate_status,
                            quarantined=any(v.get('design') == h for _, v in records['quarantine'])))
        snapshots = [dict(handle=k, **v) for k, v in records['gap_snapshot'] if v.get('design') == h]
        gaps.append(dict(design=h, status='stored_snapshot' if snapshots else 'unknown',
                         snapshots=snapshots, gaps=snapshots[-1].get('gaps') if snapshots else None,
                         reason=None if snapshots else 'No gap snapshot was stored for this design.'))
    attempts = []
    for h, value in scoped['file_candidate_attempt']:
        feedback = value.get('feedback', {})
        hypothesis = value.get('hypothesis')
        mechanism = hypothesis.get('mechanism') if isinstance(hypothesis, dict) else None
        attempts.append(dict(handle=h, parent_sha256=value.get('parent_sha256'), status=value.get('status'),
                             mechanism=mechanism or feedback.get('mechanism', 'unspecified'),
                             scope=value.get('intervals', feedback.get('affected_episodes')),
                             checks=value.get('checks'), cost_delta=value.get('cost_delta'),
                             cost_delta_scope=feedback.get('cost_delta_scope', 'saved diagnostic measurement'),
                             accepted_gain=value.get('accepted_gain'), feedback=feedback,
                             candidate=value.get('candidate'), baseline=value.get('baseline'),
                             candidate_artifact_status=_blob_status(store, value.get('candidate', {}).get('blob')),
                             reopen=feedback.get('reopening_condition')))
    notes = [dict(handle=h, **value) for h, value in scoped['agent_research_note']]
    failures = defaultdict(list)
    for line in lines:
        outcome = outcomes.get(line['handle'], {})
        failure = outcome.get('failure')
        if failure:
            failures[(failure.get('stage', 'unknown'), failure.get('error', 'unknown'))].append(line['handle'])
        else:
            event = next((e for e in reversed(line['events']) if e.get('passed') is False and e.get('stage') != 'parked'), None)
            if event:
                detail = event.get('failure', {})
                failures[(detail.get('stage', event.get('stage')), detail.get('error', event.get('reason', 'failed recorded check')))].append(line['handle'])
    for attempt in attempts:
        checks = attempt.get('checks') or {}
        if checks.get('equivalent') is not True:
            failures[('file_boundary_check', checks.get('reason', 'unproved'))].append(attempt['handle'])
    for note in notes:
        feedback = note.get('feedback', {})
        blockers = feedback.get('reported_blockers', [])
        if not isinstance(blockers, list): blockers = [blockers]
        for blocker in blockers:
            label = blocker if isinstance(blocker, str) else json.dumps(blocker, sort_keys=True)
            failures[('reported_prebuild_blocker', label)].append(note['handle'])
    shared = [dict(stage=stage, reason=reason, count=len(set(handles)), handles=list(dict.fromkeys(handles)),
                   scope='observed or reported constructions only; no mechanism family closure')
              for (stage, reason), handles in sorted(failures.items()) if len(set(handles)) > 1]
    return dict(schema_version='1', selection=dict(handle=handle, design=design, round=round),
                source='immutable store records only; generators and verifiers were not run',
                counts=dict(plans=len(plans), results=len(results), lines=len(lines), designs=len(designs),
                            file_attempts=len(attempts), file_optimizations=len(scoped['file_optimization']),
                            notes=len(notes), negatives=len(negatives),
                            outcome_statuses=dict(Counter(item.get('status', 'unknown') for result in results for item in result.get('outcomes', [])))),
                plans=plans, results=results, lines=lines, designs=designs, file_attempts=attempts, notes=notes,
                gaps=gaps, negative_memory=negatives,
                external_check_failures=[dict(handle=h,**v) for h,v in scoped['external_check_failure']],
                restructuring_runs=[dict(handle=h,**v) for h,v in scoped['restructure_run']],
                numeric={kind:[dict(handle=h,**v) for h,v in values] for kind,values in scoped.items() if kind.startswith('numeric_')},
                unbound_attempts=[dict(handle=h, identifier=value.get('identifier'), line=value.get('line'), binding_status='unbound')
                                  for h, value in records['unbound_attempt']] if selected is None else [],
                shared_failures=shared,
                discovery=[dict(value, handle=h, record_kind=kind) for kind in ('discovery_target', 'semantic_recovery_target', 'discovery_brief')
                           for h, value in scoped[kind]],
                file_optimizations=[dict(handle=h, **value) for h, value in scoped['file_optimization']])


def markdown_report(data):
    """Render the same saved report as a portable Markdown document."""
    def cell(value):
        if value is None: return 'unknown'
        if isinstance(value, (dict, list)): value = json.dumps(value, sort_keys=True, ensure_ascii=False)
        return str(value).replace('|', '\\|').replace('\n', '<br>')
    def table(headers, rows):
        if not rows: return ['No stored records.', '']
        return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |',
                *['| ' + ' | '.join(cell(value) for value in row) + ' |' for row in rows], '']
    out = ['# Strata research report', '', data['source'], '', '## Counts', '',
           cell(data['counts']), '', '## Research plans', '']
    for plan in data['plans']:
        out += [f"### Plan {plan['handle']}", '', f"Parent: {plan.get('parent', 'unknown')}", '']
        out += table(['Disposition', 'Mechanism', 'Change', 'Scope / reason', 'Readiness'],
                     [(name, item.get('hypothesis', {}).get('mechanism'), item.get('hypothesis', {}).get('change'),
                       item.get('episode', item.get('reason')), item.get('readiness'))
                      for name in ('selected', 'deferred', 'skipped') for item in plan.get(name, [])])
    out += ['## Research results', '']
    out += table(['Result', 'Mechanism', 'Status', 'Measured delta', 'Evidence / failure / unknowns'],
                 [(result['handle'], item.get('mechanism'), item.get('status'), item.get('cost_delta', item.get('measured_delta')),
                   item.get('failure', item.get('certificate', item.get('unknowns'))))
                  for result in data['results'] for item in result.get('outcomes', [])])
    out += ['## Lines', '']
    out += table(['Line', 'Mechanism', 'Status', 'Measured delta', 'Scope', 'Artifact evidence', 'Reopen'],
                 [(v['handle'], v['mechanism'], v['status'], v['measured_delta'], v['scope'], v['artifact_status'], v['reopen']) for v in data['lines']])
    out += ['## File discovery attempts', '']
    out += table(['Attempt', 'Mechanism', 'Status', 'Measured delta', 'Scope', 'Checks', 'Reopen'],
                 [(v['handle'], v['mechanism'], v['status'], v['cost_delta'], v['scope'], v['checks'], v['reopen']) for v in data['file_attempts']])
    out += ['## Structural construction runs','']
    out += table(['Run','Status','Measured delta','Alternatives','Scope'],
                 [(v['handle'],v.get('status'),v.get('cost_delta'),len(v.get('states',[])),v.get('contract'))
                  for v in data.get('restructuring_runs',[])])
    out += ['## External verifier postcheck failures','']
    out += table(['Record','Stage','Reason','Retained receipt'],
                 [(v['handle'],v.get('stage'),v.get('reason',v.get('error')),v.get('receipt_blob',v.get('receipt')))
                  for v in data.get('external_check_failures',[])])
    out += ['## Numeric candidates and experiments','']
    numeric=data.get('numeric',{})
    out += table(['Candidate','Cost','Unit','Evidence','Eligible','Frozen parameters'],
                 [(v.get('candidate'),v.get('cost'),v.get('cost_unit'),v.get('evidence_kind'),v.get('eligible'),v.get('frozen_parameters'))
                  for v in numeric.get('numeric_score',[])])
    out += table(['Experiment','Line','Settings','Factors','Observations'],
                 [(v['handle'],v.get('line'),v.get('settings'),v.get('factors'),v.get('observations'))
                  for v in numeric.get('numeric_experiment',[])])
    out += table(['Job','Stage','Outcome','Elapsed seconds','Evidence'],
                 [(v.get('job'),v.get('stage'),v.get('status'),v.get('seconds'),v.get('receipt',v.get('error')))
                  for v in numeric.get('numeric_job_finish',[])])
    out += ['## Automatic file optimization runs', '']
    out += table(['Run', 'Status', 'Baseline cost', 'Candidate cost', 'Objective / scope', 'Checks / replay'],
                 [(v['handle'], v.get('status'), v.get('baseline', {}).get('cost'), v.get('candidate', {}).get('cost'),
                   dict(objective=v.get('objective'), scope=v.get('score_status', 'saved file objective; official acceptance unknown')),
                   dict(checks=v.get('checks'), replay=v.get('replay'))) for v in data.get('file_optimizations', [])])
    for run in data.get('file_optimizations', []):
        out += [f"### File regions for {run['handle']}", '']
        out += table(['Region / scope', 'Status', 'Mechanism', 'Explanation / skip reason', 'Before / after cost', 'Checks'],
                     [(dict(id=row.get('id'), start=row.get('start'), stop=row.get('stop')), row.get('status'),
                       row.get('mechanism'), row.get('explanation', row.get('reason', row.get('note'))),
                       [row.get('before', {}).get('cost'), row.get('after', {}).get('cost')], row.get('check'))
                      for row in run.get('regions', [])])
    out += ['## Research notes and prebuild blockers', '']
    out += table(['Note', 'Class', 'Evidence', 'Reported reasoning', 'Reopen'],
                 [(v['handle'], v.get('note_class'), v.get('evidence_kind'), v.get('reported_note'),
                   v.get('feedback', {}).get('reopening_condition')) for v in data['notes']])
    out += ['## Saved design costs and gaps', '']
    out += table(['Design', 'Cost', 'Resources', 'Certificate', 'Artifact', 'Quarantined'],
                 [(v['handle'], v['cost'], v['resources'], v['certificate_status'], v['artifact_status'], v['quarantined']) for v in data['designs']])
    out += table(['Design', 'Gap status', 'Saved gaps'], [(v['design'], v['status'], v['gaps']) for v in data['gaps']])
    out += ['## Negative evidence and reopening conditions', '']
    out += table(['Record', 'Move', 'Contract / size', 'Requested / effective evidence', 'Status', 'Reopen'],
                 [(v['handle'], v['move'], dict(contract=v['contract'], scale=v['scale']),
                   [v['requested_evidence_kind'], v['evidence_kind'], v['evidence_status']], v['status'], v['reopen']) for v in data['negative_memory']])
    out += ['## Shared failures', '']
    out += table(['Stage', 'Reason', 'Count', 'Scope'], [(v['stage'], v['reason'], v['count'], v['scope']) for v in data['shared_failures']])
    out += ['## Unbound imported attempts', '', cell(data['unbound_attempts']), '']
    return '\n'.join(out)
