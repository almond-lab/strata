"""Derived bounds and exhaustive small XOR/AND straight-line synthesis.

The gate model is explicit: constants and XOR/NOT are free, each two-input AND
costs one. These are not automatically bounds on measured quantum trajectories.
"""
from math import ceil,log2
from time import monotonic
from .model import StrataError,digest


def _basis(values):
    basis={}
    for v in values:
        while v and v.bit_length()-1 in basis:v^=basis[v.bit_length()-1]
        if v:basis[v.bit_length()-1]=v
    for pivot in sorted(basis):
        for other in basis:
            if other>pivot and basis[other]>>pivot&1:basis[other]^=basis[pivot]
    return tuple(basis[p] for p in sorted(basis,reverse=True))


def _inside(value,basis):
    for row in basis:
        if value>> (row.bit_length()-1)&1:value^=row
    return value==0


def _normalize(outputs):
    """Materialize each input exactly once before validation or evaluation."""
    normalized=[]
    try:
        for poly in outputs:
            value=set()
            for raw_term in poly:
                term=tuple(raw_term)
                if any(type(q)is not int or q<0 for q in term):raise StrataError('invalid Boolean input index')
                term=frozenset(term)
                if term in value:value.remove(term)
                else:value.add(term)
            normalized.append([list(t) for t in sorted(tuple(sorted(t)) for t in value)])
    except TypeError as exc:
        raise StrataError('outputs must be Boolean ANF iterables of monomials') from exc
    return normalized


def counting_bound(outputs):
    """Outputs are Boolean ANF iterables of monomials over arbitrary input IDs."""
    outputs=_normalize(outputs)
    terms=sorted({tuple(sorted(t)) for p in outputs for t in p if len(t)>1})
    index={t:i for i,t in enumerate(terms)}
    rows=[];degree=0
    for p in outputs:
        row=0
        for t in p:
            degree=max(degree,len(t))
            if len(t)>1:row^=1<<index[tuple(sorted(t))]
        rows.append(row)
    nonlinear_rank=len(_basis(rows));degree_bound=(degree-1).bit_length() if degree else 0
    return dict(kind='proof',resource='and_count',value=max(nonlinear_rank,degree_bound),
                nonlinear_rank=nonlinear_rank,degree=degree,
                model='free XOR/NOT/constants; two-input AND straight-line circuits',
                scope='the supplied fixed-size Boolean output functions; no measurement/phase cost claim',
                contract_hash=digest(outputs),
                reasoning='Each AND adds at most one independent function modulo the affine span; degree after k ANDs is at most 2^k.')


def circuit_floors(circuit, *, from_at=0, at=None, clean_inputs=(), domain=None,
                   max_operations=10000, max_variables=128, max_terms=65536,
                   max_product_terms=1048576, max_assignments=65536):
    """Scoped Boolean floors plus bilinear tensor evidence when recoverable."""
    from .inference import functions_at
    budgets=dict(max_operations=max_operations,max_variables=max_variables,max_terms=max_terms,
                 max_product_terms=max_product_terms,max_assignments=max_assignments)
    inferred=functions_at(circuit,at,from_at=from_at,clean_inputs=clean_inputs,domain=domain,**budgets)
    if inferred['status']!='exact':return inferred
    from .domains import window_lines
    from .semantic import analyze_window
    lines,_=window_lines(circuit,from_at,inferred['at'],max_lines=max_operations)
    sem=analyze_window(lines,clean_inputs=clean_inputs,max_operations=max_operations,
                       max_variables=max_variables,max_terms=max_terms,max_product_terms=max_product_terms)
    # Exact R instructions have a zero-value proof, so they are identities on
    # this promised domain. They need no measurement/channel gate pricing.
    identity_resets = sem['status']=='exact' and bool(sem.get('effects',{}).get('reset_proofs'))
    if (inferred.get('measurement_variables') or inferred.get('classical_inputs') or
            (inferred.get('is_permutation')is False and not identity_resets)):
        return dict(status='unknown',reason='Boolean gate floor does not price measurement or classical channels',from_at=from_at,at=inferred['at'])
    wires=inferred['touched_wires']
    if domain is not None:
        from .domains import load_domain
        bound=load_domain(domain,circuit,from_at=from_at,at=inferred['at'],max_assignments=max_assignments,max_operations=max_operations)
        rows=bound['assignments'];n=len(rows)
        affine=[(1<<n)-1]+[sum(row[f'q{q}']<<i for i,row in enumerate(rows)) for q in range(circuit.num_qubits)]
        outputs=[sum(row[q]<<i for i,row in enumerate(inferred['finite_values'])) for q in wires]
        value=len(_basis(affine+outputs))-len(_basis(affine))
        return dict(status='exact',kind='proof',value=value,resource='and_count',domain_hash=bound['domain_hash'],
                    from_at=from_at,at=inferred['at'],wires=wires,clean_inputs=sorted(clean_inputs),
                    scope='explicit finite domain only; no all-input or phase/measurement floor',
                    model='free XOR/NOT/constants; two-input AND straight-line circuits restricted to promised rows',
                    reasoning='Each AND adds at most one dimension to functions restricted to the declared input rows.',
                    assignment_count=n)
    outputs=[[sorted(term) for term in inferred['functions'][q]] for q in wires]
    report=dict(status='exact',at=inferred['at'],from_at=from_at,wires=wires,
                clean_inputs=sorted(clean_inputs),**counting_bound(outputs))
    from .tensor import tensor_from_anf,tensor_bounds,quadratic_bounds
    if sem['status']=='exact':
        for feature in sem['features']:
            if feature['kind']=='bilinear_accumulator':
                partition=feature['factor_partition']
                tensor=tensor_from_anf(sem['changes'],partition[0],partition[1],output_wires=[int(q) for q in sem['changes']])
                report['tensor']=tensor;report['bilinear_bounds']=tensor_bounds(tensor)
        if all(len(term)<=2 for poly in outputs for term in poly):report['quadratic_bounds']=quadratic_bounds(outputs)
    return report


def exact_minimum(outputs,n_inputs,*,max_and=3,timeout=1.0,max_states=10000,max_expansions=100000):
    """Exhaust all affine choices around each AND, stopping honestly on budget."""
    if type(n_inputs)is not int or not 0<=n_inputs<=8 or max_and<0 or timeout<=0 or max_states<1 or max_expansions<1:
        raise StrataError('invalid exact floor budget')
    outputs=_normalize(outputs)
    if any(type(q)is not int or not 0<=q<n_inputs for p in outputs for term in p for q in term):
        raise StrataError('floor input outside declared domain')
    start=monotonic();bound=counting_bound(outputs);cases=1<<n_inputs;mask=(1<<cases)-1
    initial=_basis([mask]+[sum(((x>>q)&1)<<x for x in range(cases)) for q in range(n_inputs)])
    targets=[]
    for p in outputs:
        targets.append(sum((sum(all(x>>q&1 for q in term) for term in p)%2)<<x for x in range(cases)))
    frontier={initial:()};expansions=states=0
    def receipt(kind,value,reason,depth,witness=None):
        return dict(kind=kind,resource='and_count',value=value,reason=reason,
                    scope={'inputs':n_inputs,'outputs':len(outputs)},model=bound['model'],
                    proven_lower_bound=bound['value'],searched_up_to=depth,
                    states=states,expansions=expansions,seconds=monotonic()-start,
                    witness=witness,contract_hash=bound['contract_hash'])
    for depth in range(max_and+1):
        for basis,witness in frontier.items():
            if all(_inside(t,basis) for t in targets):
                return receipt('exact_small',depth,'minimum in complete XOR/AND search',depth,
                               [dict(left=a,right=b,product=a&b) for a,b in witness])
        if depth==max_and:break
        following={}
        for basis,witness in frontier.items():
            span=[0]
            for row in basis:span += [v^row for v in span]
            for i,a in enumerate(span):
                for b in span[i+1:]:
                    expansions+=1
                    if expansions>max_expansions or states>max_states or monotonic()-start>timeout:
                        return receipt('unknown',None,'search budget exhausted; no minimum claimed',depth)
                    product=a&b
                    if _inside(product,basis):continue
                    child=_basis((*basis,product))
                    if child not in following:
                        following[child]=(*witness,(a,b));states+=1
        frontier=following
    return receipt('unknown',None,'no construction within AND depth bound; minimum not established',max_and)
