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


def _localize_product(items, selected, start, stop):
    """Stably move a product episode to the right across independent operations."""
    from .lookup_structure import _writes

    chosen = set(selected)
    read, written = set(), set()
    passed_measurement = False
    xor_updates = {'X', 'CX', 'CCX'}
    movable = xor_updates | {'SWAP', 'CSWAP', 'NEG', 'Z', 'CZ', 'CCZ', 'HMR', 'R',
                             'BIT_STORE0', 'BIT_STORE1', 'BIT_INVERT'}
    for at in reversed(range(start, stop)):
        item = items[at]
        op = item['op']
        if item['status'] != 'supported' or op not in movable:
            return None
        writes = set(_writes(item)) | set(item['bits'])
        reads = set(item['wires'][:-1] if op in xor_updates else item['wires'])
        if op == 'BIT_INVERT':
            reads.update(item['bits'])
        if item['condition']:
            reads.add(item['condition'])
            # Conditional writes other than target XORs must preserve a skipped
            # write to that target when their condition is false.
            if op not in xor_updates:
                reads.update(writes)
        if at in chosen:
            if read & writes or written & reads or (op == 'HMR' and passed_measurement):
                return None
            if op not in xor_updates and written & writes:
                return None
        else:
            read.update(reads)
            # Only an XOR update can overlap another XOR write to its target.
            if op not in xor_updates:
                read.update(writes)
            written.update(writes)
            passed_measurement |= op == 'HMR'
    return dict(start=start, stop=stop, selected=selected,
                proof='stable permutation with disjoint read/write dependencies and commuting target XORs; same HMR order')


def select_episodes(circuit, *, max_operations, max_regions=8, clean_inputs=(), diagnostics=None):
    """Retrieve complete stored-product dependencies before generic windows."""
    from .lookup_structure import product_lifetimes
    from .semantic import lift_regions

    lines = str(circuit).splitlines()
    items, lifetimes = product_lifetimes(lines, clean_inputs)
    paid = [0]
    for item in items:
        paid.append(paid[-1] + (item['op'] in {'CCX', 'CCZ', 'CSWAP'}))
    candidates = []
    for record in lifetimes:
        if not record['consumers']:
            continue
        start, stop = record['start'], record['stop']
        selected = [start, *record['consumers'], stop-1]
        if record['cleanup'] == 'HMR':
            if stop == len(items):
                continue
            repair = items[stop]
            if (repair['op'] != 'CZ' or repair['condition'] != record['outcome']
                    or sorted(repair['wires']) != record['controls']):
                continue
            selected.append(stop)
            stop += 1
        candidates.append(dict(start=start, stop=stop, paid=paid[stop]-paid[start],
                               dependency=record, selected=selected, retrieval='complete_product_lifetime'))
    candidates.sort(key=lambda row: (not row['dependency']['clean_entry'],
                                     -len(row['dependency']['consumers']), row['stop']-row['start']))
    ranked = []
    window_limited = commute_blocked = duplicates = 0
    for row in candidates:
        a, b = row['start'], row['stop']
        if b-a > max_operations:
            selected = row['selected']
            if len(selected) > max_operations or b-a > 32 * max_operations:
                window_limited += 1
                continue
            localization = _localize_product(items, selected, a, b)
            if localization is None:
                commute_blocked += 1
                continue
            row.update(start=b-len(selected), paid=sum(
                items[i]['op'] in {'CCX', 'CCZ', 'CSWAP'} for i in selected),
                localization=localization, retrieval='commuting_product_lifetime')
        ranked.append(row)
        if len(ranked) >= 4 * max_regions:
            break
    if len(ranked) < max_regions:
        lift = lift_regions(circuit, max_window=max_operations, max_variables=128, max_terms=4096)
        windows = [dict(start=a, stop=b, paid=paid[b]-paid[a], retrieval='bounded_semantic_region')
                   for region in lift['regions'] if region['structural']
                   for a, b in [(region['start'], region['stop'])]
                   if 0 < b-a <= max_operations and paid[b] > paid[a]]
        ranked.extend(sorted(windows, key=lambda row: (-row['paid'], row['start'])))
    distinct = []
    for row in ranked:
        a, b = row['start'], row['stop']
        positions = set(row.get('selected', ()))
        overlap = False
        for prior in distinct:
            c, d = prior['start'], prior['stop']
            other = set(prior.get('selected', ()))
            if 'localization' not in row and 'localization' not in prior:
                overlap = 5 * (min(b,d)-max(a,c)) >= 4 * min(b-a,d-c)
            elif positions and other:
                overlap = 5 * len(positions & other) >= 4 * max(len(positions), len(other))
            if overlap:
                break
        if overlap:
            duplicates += 1
        else:
            distinct.append(row)
        if len(distinct) == max_regions:
            break
    if diagnostics is not None:
        diagnostics.update(matched_products=len(lifetimes), complete_dependency_anchors=len(candidates),
                           window_limited=window_limited, commuting_proof_blocked=commute_blocked,
                           near_duplicates_skipped=duplicates, selected=len(distinct))
    return distinct


def _episode_private_bits(body, suffix, declared):
    """An overwritten outcome may be omitted locally while final outputs stay public."""
    import re

    local = {int(bit) for line in body if line.startswith('HMR ')
             for bit in re.findall(r'\bb(\d+)\b', line)}
    declared_here = {int(bit) for line in body for bit in re.findall(r'\bb(\d+)\b', line)} & set(declared)
    unresolved, overwritten = local | declared_here, set()
    depth = 0
    for line in suffix:
        if not unresolved:
            break
        op = line.split()[0]
        if op in {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG'}:
            continue
        if op == 'PUSH_CONDITION':
            depth += 1
        elif op == 'POP_CONDITION':
            depth -= 1
        referenced = {int(bit) for bit in re.findall(r'\bb(\d+)\b', line)} & unresolved
        if referenced and not depth and op in {'HMR', 'BIT_STORE0', 'BIT_STORE1'} and ' if ' not in line:
            overwritten.update(referenced)
        unresolved.difference_update(referenced)
    return (set(declared) & unresolved) | overwritten, overwritten


def restructure_episode(source, circuit, out, *, scan_range, range_unit,
                        clean_inputs, cost, accounting='static',external_distribution=None,internal_bits=(),
                        _parent_catalog=None, _parent_design_dir=None, _region_index=None, _localization=None, **options):
    from .restructure import restructure_file, _check
    import re
    root = Path(out); root.mkdir(parents=True, exist_ok=True)
    lines = str(circuit).splitlines()
    if _localization is not None:
        a, b = _localization['start'], _localization['stop']
        parsed = {i: _parse(lines[i]) for i in range(a, b)}
        if _localize_product(parsed, _localization['selected'], a, b) != _localization:
            raise StrataError('episode reordering certificate does not replay against the original parent')
        selected = _localization['selected']
        chosen = set(selected)
        lines[a:b] = [lines[i] for i in range(a, b) if i not in chosen] + [lines[i] for i in selected]
    start, stop = resolve_range(lines, scan_range, range_unit)
    if stop-start > options['max_operations']:
        raise StrataError('selected episode exceeds max_operations; narrow the range or raise the explicit budget')
    body = lines[start:stop]
    local_private, overwritten = _episode_private_bits(body, lines[stop:], internal_bits)
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
                           internal_bits=local_private,_classical_bit_offset=circuit.num_bits,**options)
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
            private = state.get('private_bits', list(local_private))
            if (any(type(bit) is not int or bit < 0 for bit in private)
                    or not local_private <= set(private)
                    or any(bit < circuit.num_bits and bit not in local_private for bit in private)):
                raise StrataError('local finalist would conceal an undeclared parent bit')
            local_bits = {int(b) for line in replacement if not line.startswith('INFO_TAG ')
                          for b in re.findall(r'\bb(\d+)\b', line)}
            if any(b >= circuit.num_bits and b not in private for b in local_bits):
                raise StrataError('local finalist introduces an undeclared classical bit')
            changed = km.Circuit('\n'.join(replacement))
            full = km.Circuit('\n'.join(lines[:start]+replacement+lines[stop:]))
            removed = set(range(full.num_qubits,circuit.num_qubits))
            layout_valid = full.num_qubits <= circuit.num_qubits and removed <= set(clean_inputs)
            contextual = dict(prefix_unchanged=_localization is None,suffix_unchanged=True,layout_valid=layout_valid,
                localization=_localization,
                width_preserved=full.num_qubits==circuit.num_qubits,retired_clean_parent_wires=sorted(removed),
                clean_facts='conservative prefix transport; no interior clean promise invented')
            check = _check(contract,parent_layout(changed),tuple(facts),limits,internal_bits=private)
            row.update(check=check,contextual_proof=contextual,private_bits=private)
            if not layout_valid or check.get('equivalent') is not True:
                row.update(status='rejected' if not layout_valid or check.get('equivalent') is False else 'unknown',
                           reason='parent_layout' if not layout_valid else 'episode_boundary_proof')
                continue
            parent_private = sorted(set(internal_bits) | {bit for bit in private if bit >= circuit.num_bits})
            resources = measure(full)
            value = formula(resources)
            row.update(cost=value,resources=resources,status='no_gain')
            if value >= baseline_cost:
                continue
            accepted_ids.add(local_id)
            sha = hashlib.sha256(str(full).encode()).hexdigest()
            row.update(status='verified_improvement',full_parent_sha256=sha)
            if row['qualifications']:
                key = design_key(full,circuit.num_bits,parent_private)
                if key not in archive:
                    artifact_dir.mkdir(parents=True,exist_ok=True)
                    path = artifact_dir/(sha+'.kmx')
                    path.write_text(str(full))
                    archive[key] = dict(full_parent_sha256=sha,design_key=key,cost=value,path=str(path),
                                        resources=resources,private_bits=parent_private,qualifications=[],
                                        episode=_region_index,source_range=[start,stop],local_state=local_id,
                                        report=str(root/'restructure.json'))
                stored = archive[key]
                stored['qualifications'] = sorted(set(stored['qualifications']) | set(row['qualifications']))
                designs[stored['full_parent_sha256']] = dict(stored)
                row['parent_design_key'] = key
            if value < best_cost:
                best, best_cost, best_resources, best_bits, best_check = full, value, resources, parent_private, row
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
                     requested_range_unit=range_unit,boundary_zero_evidence=facts,
                     internal_bits=sorted(local_private), suffix_overwritten_bits=sorted(overwritten)),
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
        constructor_comparison=local.get('constructor_comparison'),
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
    retrieval = {}
    regions=[] if stopped() else select_episodes(circuit,max_operations=options['max_operations'],
                            max_regions=min(8, options['max_candidates']), clean_inputs=clean_inputs,
                            diagnostics=retrieval)
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
                _parent_catalog=verified_parents,_parent_design_dir=root/'parents',_region_index=i,
                _localization=region.get('localization'), **bounded)
            remaining-=max(1,len(run['construction'].get('attempts',[])))
            attempts.append(dict(target=region,status=run['status'],cost_delta=run['cost_delta'],
                                 search_control=run.get('search_control',{}),
                                 constructor_comparison=run['discovery_yield'].get('constructor_comparison'),
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
        cost_delta=formula(after)-formula(resources),attempts=attempts,retrieval=retrieval,
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
        episode_constructor_comparisons=[dict(target=a['target'], costs=a['constructor_comparison'])
                                         for a in attempts if a.get('constructor_comparison')],
        max_extra_finalists=finalist_cap,extra_finalists_charged=finalist_cap-finalists_remaining,
        extra_parent_finalists_checked=contextual_extra_checks,remaining_extra_finalists=finalists_remaining,
        scope='one primary per region plus a shared budget of extra finalists; distinct verified cheaper complete parents modulo display metadata and fresh private-bit names')
    (root/'restructure.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
