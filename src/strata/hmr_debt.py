"""Recover a measured predicate from surviving wire values at subsequent cuts."""
from time import monotonic

import kickmix as km

from .domains import load_domain, query_text_coordinates, window_lines, _classical_boundary
from .inference import _function_search, _wires, functions_at
from .model import StrataError, digest
from .semantic import _parse


def hmr_debt(circuit, measurement, cuts, *, support=None, clean_inputs=(), domain=None,
             unit='operation', degree=3, max_forms=4096, timeout=5, max_operations=10000,
             max_variables=128,max_terms=65536,max_product_terms=1048576,
             max_assignments=65536,max_outcomes=1024):
    """Use a virtual value-only recorder, never a helper for a proposed circuit."""
    from math import isfinite
    if not cuts or type(degree) is not int or not 0<=degree<=3 or type(max_forms) is not int or max_forms<1:
        raise StrataError('hmr-debt needs at least one later cut, degree 0..3 and a positive form budget')
    if type(timeout) not in (int,float) or not isfinite(timeout) or timeout<=0:
        raise StrataError('timeout must be positive and finite')
    clean=tuple(_wires(circuit,clean_inputs,'clean inputs'))
    if support is not None:
        support=tuple(_wires(circuit,support,'support'))
    _,event=query_text_coordinates(circuit,0,measurement,unit=unit)
    positions=[query_text_coordinates(circuit,0,cut,unit=unit)[1] for cut in cuts]
    if min(positions)<=event:
        raise StrataError('all requested cuts must be after the measured instruction')
    lines,_=window_lines(circuit,0,max(positions),max_lines=max_operations)
    item=_parse(lines[event])
    if item.get('status')!='supported' or item['op']!='HMR':
        raise StrataError('selected instruction is not an HMR')
    virtual=circuit.num_qubits
    measured=item['wires'][0]
    condition=f" if {item['condition']}" if item.get('condition') else ''
    record=f'CX q{measured} q{virtual}{condition}'
    bound=load_domain(domain,circuit,at=max(positions),max_operations=max_operations,
                      max_assignments=max_assignments,clean_inputs=clean) if domain is not None else None
    if bound and any(row[f'q{q}'] for row in bound['assignments'] for q in clean):
        raise StrataError('finite domain contradicts clean inputs')
    result=[]
    for cut,at in zip(cuts,positions):
        prefix=lines[:event]+[record]+lines[event:at]
        instrumented=km.Circuit('\n'.join(prefix))
        promise=None
        if bound:
            required=set(_classical_boundary(prefix))
            promise=dict(assignments=[dict({k:v for k,v in row.items() if not k.startswith('b') or k in required},
                                            **{f'q{virtual}':0}) for row in bound['assignments']])
            # A later classical input can be irrelevant at an earlier cut.
            promise['assignments']=list({tuple(sorted(r.items())):r for r in promise['assignments']}.values())
        state=functions_at(instrumented,clean_inputs=(*clean,virtual),domain=promise,
                           max_operations=max_operations+1,max_variables=max_variables,max_terms=max_terms,
                           max_product_terms=max_product_terms,max_assignments=max_assignments,max_outcomes=max_outcomes)
        row=dict(at=cut,text_cut=at,status=state['status'])
        result.append(row)
        if state['status']!='exact':
            row['reason']=state.get('reason')
            continue
        available=list(range(virtual)) if support is None else list(support)
        # Identical columns and exact constants cannot improve the minimum degree.
        if support is None:
            distinct={}
            for q in available:
                if 'finite_values' in state:
                    value=tuple(v[q] for v in state['finite_values'])
                    if len(set(value))<2:continue
                else:
                    value=frozenset(state['functions'][q])
                    if not value or value=={frozenset()}:continue
                distinct.setdefault(value,q)
            available=list(distinct.values())
        for d in range(degree+1):
            witness=_function_search(state,virtual,available,family=dict(max_degree=d,support=available),
                degree=d,max_forms=max_forms,timeout=timeout,start=monotonic(),from_at=0,domain=promise,
                budgets=dict(max_terms=max_terms))
            row.clear()
            row.update(witness)
            row.update(at=cut, text_cut=at)
            if witness['status']=='yes':
                row.pop('wire',None)
                row.update(minimum_degree_in_support=d,
                           conditioned_ccz_in_this_polynomial=sum(len(term)==3 for term in witness['form']['monomials']))
                break
            if witness['status']=='unknown':break
    return dict(kind='hmr_debt_query',measurement=measurement,coordinate_unit=unit,measured_wire=measured,
        classical_destination=item['bits'][0],prefix_hash=digest(lines),prefix_stop=max(positions),
        clean_inputs=clean,domain_hash=bound['domain_hash'] if bound else None,results=result,
        scope='Value predicate of this event when active, zero otherwise. Cubic term count prices this witness in the surviving wire basis.')
