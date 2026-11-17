"""Construct lookup arithmetic while preserving measurement and phase debts.

This constructor factors exact XOR-update functions between HMR/control/phase
boundaries. It keeps every measurement and its correction sequence in order;
the ordinary complete-boundary checker independently checks measured values,
classical outputs, dirty outputs, and sign. It does not equate instruments with
different measurement histories or infer clean inputs from a reset instruction.
"""
from __future__ import annotations

import heapq
import shlex
import kickmix as km
from itertools import combinations

from .semantic import analyze_window, equivalent_windows


def _factor_quadratic(terms):
    """Exact GF(2) completion of products, including x*x=x linear terms."""
    residual={frozenset(t) for t in terms}
    factors=[]
    while any(len(t)==2 for t in residual):
        i,j=min(tuple(sorted(t)) for t in residual if len(t)==2)
        left={i}|{next(iter(t-{j})) for t in residual if len(t)==2 and j in t and i not in t}
        right={j}|{next(iter(t-{i})) for t in residual if len(t)==2 and i in t and j not in t}
        for a in left:
            for b in right:
                term=frozenset((a,b))
                if term in residual:residual.remove(term)
                else:residual.add(term)
        factors.append((i,j,sorted(left-{i}),sorted(right-{j})))
    return factors,sorted(residual,key=lambda term:(len(term),tuple(sorted(term))))


def _quadratic_updates(sem,max_operations):
    changes={int(q):p['monomials'] for q,p in sem['changes'].items()}
    if sem['phase']['monomials'] or any(len(term)>2 for terms in changes.values() for term in terms):
        return None,'requires quadratic XOR updates'
    if any(any(type(x)is not int for x in term) or q in term for q,terms in changes.items() for term in terms):
        return None,'requires original-input XOR accumulators'
    incoming={q:0 for q in changes};edges={q:set() for q in changes}
    for q,terms in changes.items():
        for dependency in {v for term in terms for v in term}&changes.keys():
            edges[q].add(dependency);incoming[dependency]+=1
    ready=[q for q in changes if not incoming[q]];heapq.heapify(ready);order=[]
    while ready:
        q=heapq.heappop(ready);order.append(q)
        for other in sorted(edges[q]):
            incoming[other]-=1
            if not incoming[other]:heapq.heappush(ready,other)
    if len(order)!=len(changes):return None,'cyclic accumulator dependencies'
    emitted=[];products=0
    groups={}
    for q,terms in changes.items():
        key=tuple(sorted(tuple(term) for term in terms))
        groups.setdefault(key,[]).append(q)
    holders={q:next((p for p in targets if p in sem['clean_input_assumptions']),None)
             for targets in groups.values() if len(targets)>1 for q in targets}
    computed=set()

    def emit_update(q):
        nonlocal products
        factors,linear=_factor_quadratic(changes[q])
        for i,j,left,right in factors:
            setup=[f'CX q{k} q{i}' for k in left]+[f'CX q{k} q{j}' for k in right]
            emitted.extend(setup);emitted.append(f'CCX q{i} q{j} q{q}');emitted.extend(reversed(setup));products+=1
        for term in linear:
            emitted.append(f'CX q{next(iter(term))} q{q}' if term else f'X q{q}')
    for q in order:
        holder=holders.get(q)
        if holder is not None:
            # Its original value is zero, so no other update reads an old
            # value on this holder. Fanout to dirty targets still follows the
            # original dependency order; do not clobber them prematurely.
            if holder not in computed:
                emit_update(holder);computed.add(holder)
            if q!=holder:emitted.append(f'CX q{holder} q{q}')
        else:emit_update(q)
        if len(emitted)>max_operations:return None,'candidate operation budget'
    return emitted,dict(products=products,targets=order,method='exact quadratic-form factorization with restored control bases')


def _cubic_updates(sem,max_operations,scratch_wires):
    """Construct degree-three accumulators with bounded existing workspace.

    Dependency layers retain the original values of dirty inputs. Within a
    layer, cubic terms share a quadratic pair, then a GF(2) matrix factorization
    shares third-input linear forms and output fanout. Every scratch product is
    uncomputed. A clean output already equal to a selected pair can hold that
    pair permanently until the original measurement/consumer boundary.
    """
    changes={int(q):{frozenset(t) for t in p['monomials']} for q,p in sem['changes'].items()}
    degree=max((len(t) for terms in changes.values() for t in terms),default=0)
    if degree<=2:return _quadratic_updates(sem,max_operations)
    if degree>3:return None,dict(reason='polynomial degree exceeds cubic accumulator construction',degree=degree)
    if sem['phase']['monomials'] or any(q in term or any(type(x)is not int for x in term)
                                      for q,terms in changes.items() for term in terms):
        return None,dict(reason='requires phase-free original-input XOR accumulators',degree=degree)
    dependencies={q:{v for term in terms for v in term}&changes.keys() for q,terms in changes.items()}
    incoming={q:0 for q in changes}
    for values in dependencies.values():
        for q in values:incoming[q]+=1
    pending=set(changes);layers=[]
    while pending:
        layer=sorted(q for q in pending if incoming[q]==0)
        if not layer:return None,dict(reason='cyclic accumulator dependencies',degree=degree)
        layers.append(layer);pending.difference_update(layer)
        for q in layer:
            for dependency in dependencies[q]:incoming[dependency]-=1
    clean=set(sem['clean_input_assumptions'])
    support={v for terms in changes.values() for term in terms for v in term}
    idle=sorted(set(scratch_wires)&clean-set(changes)-support)
    helper=idle[0] if idle else None
    emitted=[];instructions=0;products=0;cx=0;pair_count=0;shared_forms=0;holder_count=0
    needs_missing_helper=False;coefficient_checks=0

    def emit(op,*wires):
        nonlocal instructions,products,cx
        instructions+=1;products+=op=='CCX';cx+=op=='CX'
        if len(emitted)<=max_operations:
            emitted.append(' '.join([op,*[f'q{q}' for q in wires]]))

    for outputs in layers:
        original={}
        for k,q in enumerate(outputs):
            for term in changes[q]:
                if len(term)==3:
                    key=tuple(sorted(term));original[key]=original.get(key,0)^(1<<k)
        # Each triple is assigned exactly once to the currently most shared
        # pair. Sparse updates bound planning by the supplied ANF size.
        pairs={};weights={}
        for triple,mask in original.items():
            for pair in combinations(triple,2):
                third=next(v for v in triple if v not in pair)
                pairs.setdefault(pair,{})[third]=mask
                weights[pair]=weights.get(pair,0)+mask.bit_count()
        heap=[(-weight,pair) for pair,weight in weights.items()];heapq.heapify(heap)
        reconstructed={};computed_holders=set()
        while heap:
            negative,pair=heapq.heappop(heap)
            if not pairs[pair] or -negative!=weights[pair]:continue
            thirds=dict(pairs[pair]);pair_count+=1
            for third,mask in thirds.items():
                triple=tuple(sorted((*pair,third)))
                for other in combinations(triple,2):
                    remainder=next(v for v in triple if v not in other)
                    del pairs[other][remainder]
                    weights[other]-=mask.bit_count()
                    if weights[other]:heapq.heappush(heap,(-weights[other],other))
            # Factor the third-input/output matrix into independent row forms.
            variables=sorted(thirds);basis={};forms=[];coefficients=[]
            for k in range(len(outputs)):
                value=sum(1<<i for i,v in enumerate(variables) if thirds[v]>>k&1)
                combination=0
                while value:
                    pivot=value.bit_length()-1
                    if pivot not in basis:
                        index=len(forms);forms.append(value);basis[pivot]=(value,index)
                        combination^=1<<index
                        break
                    vector,index=basis[pivot];value^=vector;combination^=1<<index
                coefficients.append(combination)
            holders=[q for q in outputs if q in clean and changes[q]=={frozenset(pair)}]
            holder=holders[0] if holders else None
            product_wire=holder if holder is not None else helper
            if product_wire is None:
                needs_missing_helper=True
                product_wire='required_clean_helper'
            emit('CCX',*pair,product_wire)
            if holder is not None:computed_holders.add(holder);holder_count+=1
            for index,form in enumerate(forms):
                controls=[v for i,v in enumerate(variables) if form>>i&1]
                targets=[q for k,q in enumerate(outputs) if coefficients[k]>>index&1]
                mask=sum(1<<k for k in range(len(outputs)) if coefficients[k]>>index&1)
                for control in controls:
                    triple=tuple(sorted((*pair,control)))
                    reconstructed[triple]=reconstructed.get(triple,0)^mask
                shared_forms+=1
                pivot=controls[0];target=targets[0]
                for q in controls[1:]:emit('CX',q,pivot)
                for q in targets[1:]:emit('CX',target,q)
                emit('CCX',product_wire,pivot,target)
                for q in reversed(targets[1:]):emit('CX',target,q)
                for q in reversed(controls[1:]):emit('CX',q,pivot)
            if holder is None:emit('CCX',*pair,product_wire)
        if {t:m for t,m in reconstructed.items() if m}!=original:
            raise ValueError('internal cubic coefficient identity failure')
        coefficient_checks+=1
        for q in outputs:
            if q in computed_holders:continue
            residual={t for t in changes[q] if len(t)<=2}
            factors,linear=_factor_quadratic(residual)
            rebuilt=set(linear)
            for i,j,left,right in factors:
                for a in [i,*left]:
                    for b in [j,*right]:
                        term=frozenset((a,b))
                        if term in rebuilt:rebuilt.remove(term)
                        else:rebuilt.add(term)
                for k in left:emit('CX',k,i)
                for k in right:emit('CX',k,j)
                emit('CCX',i,j,q)
                for k in reversed(right):emit('CX',k,j)
                for k in reversed(left):emit('CX',k,i)
            if rebuilt!=residual:raise ValueError('internal quadratic residual identity failure')
            for term in linear:
                if term:emit('CX',next(iter(term)),q)
                else:emit('X',q)
    details=dict(method='shared quadratic pairs and third-input/output matrix factorization',degree=degree,
                 products=products,instructions=instructions,cx=cx,pair_groups=pair_count,shared_linear_forms=shared_forms,
                 dependency_layers=layers,existing_clean_helper=helper,shared_clean_output_holders=holder_count,
                 helper_restored=True,extra_qubits=0,
                 algebraic_verification=dict(passed=True,evidence='exact cubic coefficients and quadratic residual equality over GF(2)',
                                             layers_checked=coefficient_checks),
                 cost_scope='complete emitted X/CX/CCX counts, including pair preparation, fanout and cleanup')
    if needs_missing_helper:
        return None,dict(details,status='not_emittable',reason='cubic plan needs an available declared-zero helper',
                         helper_restored=None,extra_qubits=None)
    if instructions>max_operations:
        return None,dict(details,status='bounded',reason='cubic construction exceeds the operation budget',max_operations=max_operations)
    return emitted,dict(details,status='constructed')


def resynthesize_measured_lookup(circuit,*,clean_inputs=(),max_operations=10000,
                                max_variables=128,max_terms=65536,max_product_terms=65536):
    """Return a complete candidate or an explicit bounded construction result.

Only unconditional arithmetic outside condition scopes is rewritten. Every
physical measurement/phase/control boundary is retained. The supported segment
family is a degree-at-most-three XOR accumulator, including dirty output words.
Cubic plans use only existing declared-zero workspace and restore it. No public
index is changed and no extra physical wire is allocated.
    """
    lines=str(circuit).splitlines()
    limits=dict(max_operations=max_operations,max_variables=max_variables,max_terms=max_terms,
                max_product_terms=max_product_terms)
    whole=analyze_window(lines,clean_inputs=clean_inputs,**limits)
    if whole['status']!='exact':
        return dict(status='unknown',reason='measured lookup needs exact complete boundary',semantics=whole)
    if not whole['measurement_events']:
        return dict(status='not_applicable',reason='no measurement-based lookup cleanup')
    if whole['phase']['monomials'] or any(d['status']!='discharged' for d in whole['effects']['phase_debts']):
        return dict(status='not_applicable',reason='lookup has unresolved phase debt or a nonzero boundary phase')
    # Construction may not discard a promised clean output leak even if the
    # supplied circuit contains one.
    if any(whole['outputs'].get(str(q),{}).get('monomials',[]) for q in clean_inputs):
        return dict(status='not_applicable',reason='declared clean lookup workspace is not restored')
    known_zero=set(clean_inputs);output=[];segment=[];start=0;depth=0;attempts=[]

    def flush(stop):
        nonlocal segment,known_zero
        if not segment:return
        sem=analyze_window(segment,clean_inputs=known_zero,**limits)
        row=dict(start=start,stop=stop,clean_at_entry=sorted(known_zero))
        candidate,details=(None,'exact segment unavailable') if sem['status']!='exact' else _cubic_updates(
            sem,max_operations,[q for q in known_zero if q<circuit.num_qubits])
        chosen=segment
        if candidate is not None:
            before_paid=sum(line.split()[0]=='CCX' for line in segment)
            after_paid=sum(line.split()[0]=='CCX' for line in candidate)
            lower_cost=(after_paid,len(candidate))<(before_paid,len(segment))
            check=(equivalent_windows(segment,candidate,clean_inputs=known_zero,**limits) if lower_cost
                   else dict(status='not_run',reason='constructed candidate is not lower cost'))
            if check.get('equivalent')is True and lower_cost:
                chosen=candidate;row.update(status='replaced',before_products=before_paid,after_products=after_paid,
                                           construction=details,check=check)
            else:row.update(status='retained',reason='no proved lower-cost segment',check=check,construction=details,
                            before_products=before_paid,after_products=after_paid)
        elif isinstance(details,dict):row.update(status='retained',reason=details['reason'],construction=details)
        else:row.update(status='retained',reason=details)
        output.extend(chosen);attempts.append(row)
        if sem['status']=='exact':
            known_zero-=set(sem['inputs']);known_zero|=set(sem['effects']['clean_outputs'])
        else:
            # Do not transport any clean promise through an unanalysed write.
            for line in segment:
                words=shlex.split(line)
                if words[0]=='SWAP':known_zero.difference_update(int(w[1:]) for w in words[1:3])
                else:known_zero.discard(int(words[-1][1:]))
        segment=[]

    for index,line in enumerate(lines):
        words=shlex.split(line)
        if not words:continue
        op=words[0]
        if depth==0 and 'if' not in words and op in {'X','CX','CCX','SWAP'}:
            if not segment:start=index
            segment.append(line)
            continue
        flush(index);output.append(line)
        if op=='PUSH_CONDITION':depth+=1
        elif op=='POP_CONDITION':depth=max(0,depth-1)
        elif op=='HMR':
            q=int(words[1][1:])
            if depth==0 and 'if' not in words:known_zero.add(q)
        elif op in {'X','CX','CCX','CSWAP','SWAP','R'}:
            # Conditioned updates cannot establish an unconditional lease.
            touched=[int(w[1:]) for w in words[1:] if w.startswith('q') and w[1:].isdigit()]
            if op in {'SWAP','CSWAP'}:known_zero.difference_update(touched[-2:])
            elif op!='R' and touched:known_zero.discard(touched[-1])
    flush(len(lines))
    if not any(row['status']=='replaced' for row in attempts):
        return dict(status='not_constructed',reason='no profitable verified measured-lookup construction within the supplied budgets',segments=attempts)
    candidate='\n'.join(output)
    if km.Circuit(candidate).num_qubits<circuit.num_qubits:
        # Keep public layout without introducing an extra named register into
        # a host that may restrict register names. Two X gates are an identity.
        output.extend([f'X q{circuit.num_qubits-1}']*2)
        candidate='\n'.join(output)
    check=equivalent_windows(lines,output,clean_inputs=clean_inputs,**limits)
    if check.get('equivalent')is not True:
        return dict(status='unknown',reason='complete measured-lookup equivalence not established',check=check,segments=attempts)
    return dict(status='constructed',candidate=candidate,segments=attempts,check=check,
        reason='factor complete lookup updates while preserving measured values and all HMR phase repairs',
        measurement_scope='same ordered wire/outcome provenance, all inputs and formal outcomes; no changed-instrument theorem',
        scope='quadratic and cubic accumulator segments with existing declared-zero workspace; no universal lookup synthesis or width reduction claim')
