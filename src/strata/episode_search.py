"""Bounded automatic episode retrieval and contextual replacement.

Retrieval suggests boundaries, never certifies them. Every candidate is checked
on the complete selected boundary with independently transported zero facts;
untouched prefix/suffix and the parent's physical width are preserved.
"""
from pathlib import Path
import hashlib
import json

import kickmix as km

from .model import StrataError
from .cost import Formula
from .kickmix import Kickmix
from .semantic import _parse, _advance_zero_facts, equivalent_windows


def resolve_range(lines, region, unit='operation'):
    if unit not in {'operation', 'text-line'}:
        raise StrataError('range_unit must be operation or text-line')
    if (len(region) != 2 or any(type(x) is not int for x in region)
            or not 0 <= region[0] < region[1]):
        raise StrataError('range must be a nonempty half-open START:STOP interval')
    indices = ([i for i, line in enumerate(lines)
                if line.split()[0] not in {'REGISTER', 'APPEND_TO_REGISTER'}]
               if unit == 'operation' else list(range(len(lines))))
    indices.append(len(lines))
    a, b = region
    if b >= len(indices):
        raise StrataError('range exceeds supplied circuit')
    return indices[a], indices[b]


def boundary_facts(lines, start, stop, clean_inputs):
    zeros = {q: dict(kind='declared_artifact_input', origin_at=0, origin_wire=q)
             for q in clean_inputs}
    depth = 0
    for at, line in enumerate(lines[:stop]):
        if at == start:
            if depth:
                raise StrataError('selected episode starts inside a classical condition scope')
            boundary = dict(zeros)
        item = _parse(line)
        _advance_zero_facts(item, zeros, depth, at)
        if item['op'] == 'PUSH_CONDITION':
            depth += 1
        elif item['op'] == 'POP_CONDITION':
            depth -= 1
            if depth < 0:
                raise StrataError('unbalanced classical condition scope')
    if depth:
        raise StrataError('selected episode ends inside a classical condition scope')
    return boundary


def select_episodes(circuit, *, max_operations, max_regions=8):
    """Find repeated write/restore anchors across arbitrary scan windows.

    Matching involutions only retrieve candidate intervals. Controls may have
    changed and scratch may be dirty; the exact boundary solver decides what
    function the full episode implements. No source tags or proposed move are
    required. Existing bounded semantic regions provide additional candidates.
    """
    from .semantic import lift_regions
    lines = str(circuit).splitlines()
    seen, spans, depth = {}, {}, 0
    paid = [0]
    for at, line in enumerate(lines):
        item = _parse(line)
        paid.append(paid[-1] + (item['op'] in {'CCX', 'CCZ', 'CSWAP'}))
        if item['op'] == 'PUSH_CONDITION':
            depth += 1
        elif item['op'] == 'POP_CONDITION':
            depth -= 1
        if depth or item.get('condition') or item['status'] != 'supported':
            continue
        if item['op'] in {'CCX', 'CX', 'SWAP'}:
            key = line
            previous = seen.get(key)
            if previous is not None and 2 < at + 1 - previous <= max_operations:
                spans[previous, at + 1] = 'matched_involution_dependency_hypothesis'
            seen[key] = at
        # Bound anchor history even for a very long nonrepeating stream.
        if len(seen) > max_operations:
            seen = {k: v for k, v in seen.items() if at - v < max_operations}
    lift = lift_regions(circuit, max_window=max_operations,
                        max_variables=128, max_terms=4096)
    for region in lift['regions']:
        if region['structural']:
            spans.setdefault((region['start'], region['stop']), 'bounded_semantic_region')
    # A bank of related intermediates may be restored in a different order:
    # overlapping hypotheses are widened together before any synthesis.
    merged=[]
    for a,b in sorted(spans):
        if merged and a < merged[-1][1] and max(b,merged[-1][1])-merged[-1][0] <= max_operations:
            merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:
            merged.append((a,b))
    for interval in merged:
        spans.setdefault(interval,'joint_intermediate_bank')
    ranked = [dict(start=a, stop=b, paid=paid[b]-paid[a], retrieval=reason)
              for (a, b), reason in spans.items() if paid[b] > paid[a]]
    ranked.sort(key=lambda x: (-x['paid'], -(x['stop']-x['start']), x['start']))
    return ranked[:max_regions]


def restructure_episode(source, circuit, out, *, scan_range, range_unit,
                        clean_inputs, cost, accounting='static',external_distribution=None,internal_bits=(),
                        _parent_catalog=None, _parent_design_dir=None, _region_index=None, **options):
    from .restructure import restructure_file, _check
    import re
    root = Path(out); root.mkdir(parents=True, exist_ok=True)
    lines = str(circuit).splitlines()
    start, stop = resolve_range(lines, scan_range, range_unit)
    if stop-start > options['max_operations']:
        raise StrataError('selected episode exceeds max_operations; narrow the range or raise the explicit budget')
    body = lines[start:stop]
    outside='\n'.join(lines[:start]+lines[stop:])
    if any(re.search(r'\bb'+str(bit)+r'\b',outside) for bit in internal_bits):
        raise StrataError('a private episode bit is referenced outside the selected boundary; widen the episode')
    if any(line.split()[0] in {'REGISTER', 'APPEND_TO_REGISTER'} for line in body):
        raise StrataError('selected episode includes public register declarations; select its operations only')
    facts = boundary_facts(lines, start, stop, clean_inputs)
    fragment = root/'episode.kmx'
    if fragment.resolve() == Path(source).resolve():
        raise StrataError('episode output would overwrite its source')
    fragment.write_text('\n'.join(body))
    from .accounting import measured_resources
    def strip_metadata(c):
        return [line for line in str(c).splitlines()
                if line.split()[0] not in {'REGISTER','APPEND_TO_REGISTER'}]
    def price_in_parent(c):
        parent=km.Circuit('\n'.join(lines[:start]+strip_metadata(c)+lines[stop:]))
        return measured_resources(parent,accounting=accounting,external_distribution=external_distribution)
    run = restructure_file(fragment, root/'construction', clean_inputs=tuple(facts),
                           cost=cost, width=circuit.num_qubits,accounting=accounting,
                           external_distribution=external_distribution,_resource_measure=price_in_parent,
                           internal_bits=internal_bits,_classical_bit_offset=circuit.num_bits,**options)
    from .discovery_yield import design_key
    from .restructure import _boundary_metadata
    from .search_screen import stop_reason
    limits = {key: options[key] for key in ('max_operations','max_variables','max_terms','max_product_terms')}
    formula = Formula(cost)
    def measure(c):
        return measured_resources(c,accounting=accounting,external_distribution=external_distribution)
    before = measure(circuit)
    baseline_cost = formula(before)
    original_tags = [line for line in body if line.startswith('INFO_TAG ')]
    def parent_layout(c):
        if not circuit.num_qubits:
            return c
        metadata = _boundary_metadata(c)
        extra = _boundary_metadata(c,circuit.num_qubits-1,'strata_episode_layout')[len(metadata):]
        return km.Circuit(str(c)+'\n'+'\n'.join(extra))
    contract = parent_layout(km.Circuit('\n'.join(body)))
    def stopped():
        return stop_reason(options.get('_deadline'),options.get('cancel_file'))

    local = run.get('discovery_yield', {})
    qualifications = {row['id']:row['qualifications'] for row in local.get('designs', [])}
    states = {row['id']:row for row in run['states']}
    selected = run['best']['id']
    candidates = [selected, *(key for key in qualifications if key != selected)]
    cap = options.get('max_finalists', 24)
    archive = {} if _parent_catalog is None else _parent_catalog
    artifact_dir = _parent_design_dir or root/'parents'
    designs, checks, accepted_ids = {}, [], set()
    best, best_cost, best_resources, best_bits = circuit, baseline_cost, before, list(internal_bits)
    best_check = None
    for local_id in candidates[:1 + cap]:
        if stopped():
            break
        row = dict(local_state=local_id, selected_locally=local_id == selected,
                   qualifications=qualifications.get(local_id, []))
        checks.append(row)
        try:
            state = states[local_id]
            proposed = km.Circuit.from_file(state['path'])
            if hashlib.sha256(str(proposed).encode()).hexdigest() != local_id:
                raise StrataError('local finalist bytes differ from the checked state')
            replacement = strip_metadata(proposed)
            if [line for line in replacement if line.startswith('INFO_TAG ')] != original_tags:
                raise StrataError('constructor changed episode metadata')
            if len(replacement) > limits['max_operations']:
                raise StrataError('parent splice exceeds the episode operation budget')
            private = state.get('private_bits', list(internal_bits))
            if (any(type(bit) is not int or bit < 0 for bit in private)
                    or not set(internal_bits) <= set(private)
                    or any(bit < circuit.num_bits and bit not in internal_bits for bit in private)):
                raise StrataError('local finalist would conceal an undeclared parent bit')
            local_bits = {int(b) for line in replacement if not line.startswith('INFO_TAG ')
                          for b in re.findall(r'\bb(\d+)\b', line)}
            if any(b >= circuit.num_bits and b not in private for b in local_bits):
                raise StrataError('local finalist introduces an undeclared classical bit')
            changed = km.Circuit('\n'.join(replacement))
            full = km.Circuit('\n'.join(lines[:start]+replacement+lines[stop:]))
            removed = set(range(full.num_qubits,circuit.num_qubits))
            layout_valid = full.num_qubits <= circuit.num_qubits and removed <= set(clean_inputs)
            contextual = dict(prefix_unchanged=True,suffix_unchanged=True,layout_valid=layout_valid,
                width_preserved=full.num_qubits==circuit.num_qubits,retired_clean_parent_wires=sorted(removed),
                clean_facts='conservative prefix transport; no interior clean promise invented')
            check = _check(contract,parent_layout(changed),tuple(facts),limits,internal_bits=private)
            row.update(check=check,contextual_proof=contextual,private_bits=private)
            if not layout_valid or check.get('equivalent') is not True:
                row.update(status='rejected' if not layout_valid or check.get('equivalent') is False else 'unknown',
                           reason='parent_layout' if not layout_valid else 'episode_boundary_proof')
                continue
            resources = measure(full)
            value = formula(resources)
            row.update(cost=value,resources=resources,status='no_gain')
            if value >= baseline_cost:
                continue
            accepted_ids.add(local_id)
            sha = hashlib.sha256(str(full).encode()).hexdigest()
            row.update(status='verified_improvement',full_parent_sha256=sha)
            if row['qualifications']:
                key = design_key(full,circuit.num_bits,private)
                if key not in archive:
                    artifact_dir.mkdir(parents=True,exist_ok=True)
                    path = artifact_dir/(sha+'.kmx')
                    path.write_text(str(full))
                    archive[key] = dict(full_parent_sha256=sha,design_key=key,cost=value,path=str(path),
                                        resources=resources,private_bits=private,qualifications=[],
                                        episode=_region_index,source_range=[start,stop],local_state=local_id,
                                        report=str(root/'restructure.json'))
                stored = archive[key]
                stored['qualifications'] = sorted(set(stored['qualifications']) | set(row['qualifications']))
                designs[stored['full_parent_sha256']] = dict(stored)
                row['parent_design_key'] = key
            if value < best_cost:
                best, best_cost, best_resources, best_bits, best_check = full, value, resources, private, row
        except (KeyError,OSError,StrataError,ValueError,RuntimeError) as exc:
            row.update(status='rejected',reason=str(exc))

    destination = root/'best.kmx'
    destination.write_text(str(best))
    representative = best_check or (checks[0] if checks else {})
    extra_checked = max(0,len(checks)-1)
    result = dict(status='verified_improvement' if best_check else 'no_gain',
        baseline=dict(resources=before,cost=baseline_cost),
        best=dict(path=str(destination),resources=best_resources,cost=best_cost,private_bits=best_bits),
        candidate_cost=representative.get('cost'), cost_delta=best_cost-baseline_cost,
        source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        episode=dict(start=start,stop=stop,range_unit='text-line',requested_range=list(scan_range),
                     requested_range_unit=range_unit,boundary_zero_evidence=facts),
        check=representative.get('check',dict(equivalent=None,reason=representative.get('reason') or stopped())),
        construction=run,parent_width=circuit.num_qubits,contextual_proof=representative.get('contextual_proof',{}),
        objective=dict(expression=cost,accounting=before['accounting'],external_distribution=external_distribution),
        official_verifier_run=False)
    result['structural_gains'] = [row for row in run.get('structural_gains',[]) if row['candidate'] in accepted_ids]
    result['search_control'] = dict(run.get('search_control',{}))
    if stopped():
        result['search_control']['stop_reason'] = stopped()
    result['discovery_yield'] = dict(unique_verified_cheaper_native_designs=len(designs),designs=designs,
        selected_full_parent_sha256=(best_check['full_parent_sha256'] if best_check and best_check['qualifications'] else None),
        extra_region_native_designs=local.get('unique_verified_cheaper_native_designs',0),
        extra_parent_finalists_checked=extra_checked,extra_finalists_charged=max(extra_checked,local.get('extra_finalists_checked',0)),
        max_extra_finalists=cap,contextual_checks=checks,unchecked_local_finalists=len(candidates)-len(checks),
        scope='individually verified cheaper complete parents, modulo display metadata and fresh private-bit names')
    (root/'restructure.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def restructure_regions(source, circuit, out, *, clean_inputs, cost,accounting='static',external_distribution=None,**options):
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    from .search_screen import stop_reason
    def stopped():
        return stop_reason(options.get('_deadline'),options.get('cancel_file'))
    regions=[] if stopped() else select_episodes(circuit,max_operations=options['max_operations'],
                            max_regions=min(8, options['max_candidates']))
    attempts=[]; best=circuit; best_result=None; verified_parents={}
    finalist_cap=options.get('max_finalists',24)
    finalists_remaining=finalist_cap
    contextual_extra_checks=0
    remaining=options['max_candidates']
    # Siblings share one frozen parent. Composition is deliberately not assumed.
    for i, region in enumerate(regions):
        if remaining<=0 or stopped():break
        try:
            quota=min(finalists_remaining,max(1,finalists_remaining//(len(regions)-i)))
            bounded=dict(options,max_candidates=max(1,remaining//(len(regions)-i)),max_finalists=quota)
            run=restructure_episode(source,circuit,root/f'region-{i}',
                scan_range=(region['start'],region['stop']),range_unit='text-line',
                clean_inputs=clean_inputs,cost=cost,accounting=accounting,external_distribution=external_distribution,
                _parent_catalog=verified_parents,_parent_design_dir=root/'parents',_region_index=i,**bounded)
            remaining-=max(1,len(run['construction'].get('attempts',[])))
            attempts.append(dict(target=region,status=run['status'],cost_delta=run['cost_delta'],
                                 search_control=run.get('search_control',{}),
                                 report=str(root/f'region-{i}'/'restructure.json')))
            if run['status']=='verified_improvement' and (best_result is None or run['best']['cost']<best_result['best']['cost']):
                best_result=run;best=km.Circuit.from_file(run['best']['path'])
            finalists_remaining-=run['discovery_yield']['extra_finalists_charged']
            contextual_extra_checks+=run['discovery_yield']['extra_parent_finalists_checked']
        except (StrataError,ValueError,RuntimeError) as exc:
            # A failed region may have consumed its entire local finalist allowance.
            finalists_remaining-=quota
            remaining-=1
            attempts.append(dict(target=region,status='unknown',reason=str(exc)))
    from .accounting import measured_resources
    resources=measured_resources(circuit,accounting=accounting,external_distribution=external_distribution)
    after=measured_resources(best,accounting=accounting,external_distribution=external_distribution);formula=Formula(cost)
    destination=root/'best.kmx';destination.write_text(str(best))
    result=dict(status='verified_improvement' if best_result else 'no_gain',
        baseline=dict(resources=resources,cost=formula(resources)),
        best=dict(path=str(destination),resources=after,cost=formula(after),
                  private_bits=best_result['best'].get('private_bits',[]) if best_result else list(options.get('internal_bits',()))),
        cost_delta=formula(after)-formula(resources),attempts=attempts,
        search='automatic bounded episode selection and certified construction; no supplied mechanism',
        source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        structural_gains=best_result.get('structural_gains',[]) if best_result else [],
        construction_budget=dict(max_candidates=options['max_candidates'],remaining=max(0,remaining)),
        objective=dict(expression=cost,accounting=resources['accounting'],external_distribution=external_distribution),official_verifier_run=False,
        limitations=['bounded constructor families; no global optimum or arithmetic typing claim',
                     'sibling improvements are not composed without a new search on retained bytes'])
    result['search_control']=dict(fast=options.get('fast',False),stop_reason=stopped(),
        search_seconds=options.get('search_seconds'),trial_seconds=options.get('trial_seconds'),
        screened_count=sum(row.get('search_control',{}).get('screened_count',0) for row in attempts),
        timed_out=sum(row.get('search_control',{}).get('timed_out',0) for row in attempts))
    result['discovery_yield']=dict(unique_verified_cheaper_native_designs=len(verified_parents),
        designs={record['full_parent_sha256']:record for record in verified_parents.values()},
        max_extra_finalists=finalist_cap,extra_finalists_charged=finalist_cap-finalists_remaining,
        extra_parent_finalists_checked=contextual_extra_checks,remaining_extra_finalists=finalists_remaining,
        scope='one primary per region plus a shared budget of extra finalists; distinct verified cheaper complete parents modulo display metadata and fresh private-bit names')
    (root/'restructure.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
