"""Derived bounds and exhaustive small XOR/AND straight-line synthesis.

The gate model is explicit: constants and XOR/NOT are free, each two-input AND
costs one. These are not automatically bounds on measured quantum trajectories.
"""
from math import isfinite
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


def circuit_floors(circuit, *, from_at=0, at=None, clean_inputs=(), domain=None, observations=None, outputs=None,
                   exact=False, max_and=3, timeout=1.0, max_states=10000, max_expansions=100000,
                   max_operations=10000, max_variables=128, max_terms=65536,
                   max_product_terms=1048576, max_assignments=65536, max_outcomes=1024):
    """A value-map lower bound, including complete repaired HMR episodes.

    Selecting outputs relaxes the boundary; it never proves that omitted garbage
    can be erased. Exact search also solves only this Boolean relaxation.
    """
    if observations is not None:
        from .boundary_observations import query_observations
        return query_observations('floors', circuit, observations, at=at, from_at=from_at,
            clean_inputs=clean_inputs, domain=domain, outputs=outputs, exact=exact, max_and=max_and,
            timeout=timeout, max_states=max_states, max_expansions=max_expansions,
            max_operations=max_operations, max_variables=max_variables, max_terms=max_terms,
            max_product_terms=max_product_terms, max_assignments=max_assignments, max_outcomes=max_outcomes)
    from .inference import _wires
    if exact:
        _validate_search_limits(max_and,timeout,max_states,max_expansions)
    clean_inputs = _wires(circuit, clean_inputs, 'clean inputs')
    selected = None if outputs is None else _wires(circuit, outputs, 'floor outputs')
    if domain is None:
        from .domains import window_lines, WindowLimit
        from .semantic import analyze_complete
        try:
            complete_lines,complete_at=window_lines(circuit,from_at,at,max_lines=max_operations)
        except WindowLimit:
            complete_lines=None
        if complete_lines is not None:
            complete=analyze_complete(complete_lines,clean_inputs=clean_inputs,max_operations=max_operations,
                max_variables=max_variables,max_terms=max_terms,max_product_terms=max_product_terms)
            if complete['status']=='exact_factored':
                from .factored_semantic import projected_value_semantics
                projected=projected_value_semantics(complete_lines,clean_inputs=clean_inputs,max_operations=max_operations,
                    max_variables=max_variables,max_terms=max_terms,max_product_terms=max_product_terms)
                if projected['status']!='exact':
                    return dict(status='unknown',reason='Boolean gate floor requires bounded final ANFs',
                                from_at=from_at,at=complete_at,semantics=complete,
                                semantic_status='exact_factored',floor_established=False)
    from .inference import functions_at
    budgets=dict(max_operations=max_operations,max_variables=max_variables,max_terms=max_terms,
                 max_product_terms=max_product_terms,max_assignments=max_assignments,max_outcomes=max_outcomes)
    inferred=functions_at(circuit,at,from_at=from_at,clean_inputs=clean_inputs,domain=domain,**budgets)
    if inferred['status']!='exact':return inferred
    from .domains import window_lines
    from .inference import _value_semantics
    lines,_=window_lines(circuit,from_at,inferred['at'],max_lines=max_operations)
    sem=_value_semantics(lines,clean_inputs=clean_inputs,max_operations=max_operations,
                       max_variables=max_variables,max_terms=max_terms,max_product_terms=max_product_terms)
    wires=inferred['touched_wires'] if selected is None else selected
    common=dict(from_at=from_at,at=inferred['at'],wires=wires,clean_inputs=sorted(clean_inputs),
                selected_outputs=wires, omitted_outputs=sorted(set(inferred['touched_wires'])-set(wires)),
                output_scope='wanted Boolean values only; omitted garbage erasure and restoration are not established',
                floor_established=True, global_quantum_floor=False,
                window_hash=inferred['window_hash'])
    measured=bool(inferred.get('measurement_variables'))
    if measured and sem['status']!='exact':
        return dict(common,status='unknown',floor_established=False,
                    reason='measurement phase obligations require exact full-window semantics')
    bound=None
    if domain is not None:
        from .domains import load_domain
        bound=load_domain(domain,circuit,from_at=from_at,at=inferred['at'],max_assignments=max_assignments,
                          max_operations=max_operations,value_only=True,clean_inputs=clean_inputs,
                          max_variables=max_variables,max_terms=max_terms,max_product_terms=max_product_terms)
    # Substituting a promised row must leave no formal outcome in any wanted
    # value or in the complete phase. Other classical phase variables remain
    # symbolic, including those legitimately omitted by a value-only domain.
    if sem['status']=='exact':
        formal=set(sem.get('measurement_variables', []))
        phase={frozenset(t) for t in sem['phase']['monomials']}
        values=[inferred['functions'][q] for q in wires]
        if bound is None:
            value_independent=not any(type(v)is not int for poly in values for term in poly for v in term)
            phase_independent=not any(term & formal for term in phase)
        else:
            value_independent=phase_independent=True
            for row in bound['assignments']:
                assignment={int(k[1:]) if k.startswith('q') else k:v for k,v in row.items()}
                value_independent &= not any(term for poly in values for term in _restrict(poly,assignment))
                phase_independent &= not any(term & formal for term in _restrict(phase,assignment))
        common['measurement_analysis']=dict(
            outcome_independent_values=value_independent, phase_obligations_discharged=phase_independent,
            measurement_variables=sorted(formal), phase=sem['phase'],
            phase_debts=sem.get('effects',{}).get('phase_debts',[]),
            proof='exact full-window ANF, restricted to each promised row when supplied',
            scope='native basis/sign trajectories; not general quantum-channel equivalence')
        if not value_independent or not phase_independent:
            return dict(common,status='unknown',floor_established=False,
                        reason=('wanted_values_depend_on_measurement_or_classical_inputs' if not value_independent
                                else 'measurement_phase_obligations_outstanding'))
    if bound is not None:
        rows=bound['assignments']
        bits=[b for b in bound['classical_inputs'] if b not in bound.get('omitted_classical_inputs',[])]
        input_rows=[[row[f'q{q}'] for q in range(circuit.num_qubits)]+[row[b] for b in bits] for row in rows]
        stride=inferred.get('evaluated_outcomes',inferred.get('outcome_count',1))
        output_rows=[[inferred['finite_values'][i*stride][q] for q in wires] for i in range(len(rows))]
        initial,targets=_row_tables(input_rows,output_rows)
        value=len(_basis((*initial,*targets)))-len(_basis(initial))
        report=dict(common,status='exact',kind='proof',value=value,resource='and_count',domain_hash=bound['domain_hash'],
                    scope='explicit finite domain only; Boolean value relaxation, with no all-input quantum gate floor',
                    model='free XOR/NOT/constants; two-input AND straight-line circuits restricted to promised rows',
                    reasoning='Each AND adds at most one dimension to functions restricted to the declared input rows.',
                    assignment_count=len(rows), omitted_classical_inputs=bound.get('omitted_classical_inputs',[]),
                    value_dependency_proof=bound.get('value_dependency_proof'),
                    boolean_input_variables=[f'q{q}' for q in range(circuit.num_qubits)]+bits)
    else:
        polynomials=[[sorted(term) for term in inferred['functions'][q]] for q in wires]
        report=dict(common,status='exact',**counting_bound(polynomials))
    if measured:
        report['model_restrictions']=dict(
            basis_value_gates='NOT/CNOT/SWAP and counted binary AND updates (CCX/CSWAP)',
            measurement='HMR fresh independent outcomes and zeroed measured wires; exact full-window sign debt discharged',
            classical='Boolean control computation is included in the XOR/AND model; no free nonlinear classical oracle',
            phases='diagonal signs do not change basis values; their synthesis cost is not bounded here',
            excluded='arbitrary quantum gates, interference, postselection, and general quantum/Toffoli lower bounds')
    from .tensor import tensor_from_anf,tensor_bounds,quadratic_bounds
    if sem['status']=='exact' and bound is None and selected is None and not measured:
        for feature in sem.get('features',[]):
            if feature['kind']=='bilinear_accumulator':
                partition=feature['factor_partition']
                tensor=tensor_from_anf(sem['changes'],partition[0],partition[1],output_wires=[int(q) for q in sem['changes']])
                report['tensor']=tensor;report['bilinear_bounds']=tensor_bounds(tensor)
        if all(len(term)<=2 for poly in polynomials for term in poly):report['quadratic_bounds']=quadratic_bounds(polynomials)
    if exact:
        limits=dict(max_and=max_and,timeout=timeout,max_states=max_states,max_expansions=max_expansions)
        if bound is not None:
            search=exact_minimum_on_rows(input_rows,output_rows,**limits)
        else:
            variables=sorted({v for poly in polynomials for term in poly for v in term})
            if len(variables)>8:
                search=dict(kind='unknown',value=None,reason='full truth-table exact search input budget; supply a care set',
                            proven_lower_bound=report['value'],witness=None)
            else:
                remap={v:i for i,v in enumerate(variables)}
                search=exact_minimum([[[remap[v] for v in term] for term in poly] for poly in polynomials],len(variables),**limits)
                search['input_variables']=variables
        report['exact_search']=search
        report['minimum_established']=search['kind']=='exact_small'
        if report['minimum_established']:
            report['counting_lower_bound']=report['value']
            report['value']=search['value']
            report['kind']='exact_small'
    return report


def _restrict(poly,assignment):
    result=set()
    for term in poly:
        if any(v in assignment and not assignment[v] for v in term):
            continue
        remaining=frozenset(v for v in term if v not in assignment)
        if remaining in result:result.remove(remaining)
        else:result.add(remaining)
    return result


def _validate_search_limits(max_and,timeout,max_states,max_expansions):
    if (type(max_and)is not int or max_and<0 or type(timeout)not in (int,float)
            or not isfinite(timeout) or timeout<=0 or type(max_states)is not int or max_states<1
            or type(max_expansions)is not int or max_expansions<1):
        raise StrataError('invalid exact floor budget')


def _canonical_rows(input_rows,output_rows):
    try:
        inputs=[tuple(row) for row in input_rows]
        outputs=[tuple(row) for row in output_rows]
    except TypeError as exc:
        raise StrataError('care-set inputs and outputs must be bit-vector rows') from exc
    if not inputs or len(inputs)!=len(outputs):
        raise StrataError('care set needs matching nonempty input and output rows')
    n_inputs,n_outputs=len(inputs[0]),len(outputs[0])
    rows={}
    for source,target in zip(inputs,outputs):
        if (len(source)!=n_inputs or len(target)!=n_outputs or
                any(type(bit)is not int or bit not in (0,1) for bit in (*source,*target))):
            raise StrataError('care-set rows must have uniform widths and integer bits')
        if source in rows and rows[source]!=target:
            raise StrataError('care set assigns conflicting outputs to one input')
        rows[source]=target
    inputs=sorted(rows)
    return inputs,[rows[row] for row in inputs]


def _row_tables(input_rows,output_rows):
    count=len(input_rows)
    initial=[(1<<count)-1]+[sum(row[q]<<i for i,row in enumerate(input_rows)) for q in range(len(input_rows[0]))]
    targets=[sum(row[q]<<i for i,row in enumerate(output_rows)) for q in range(len(output_rows[0]))]
    return initial,targets


def _coefficients(value,available):
    """An executable affine combination of the constant, inputs and earlier ANDs."""
    basis={}
    for index,row in enumerate(available):
        mask=1<<index
        while row and row.bit_length()-1 in basis:
            previous,coeff=basis[row.bit_length()-1]
            row^=previous;mask^=coeff
        if row:basis[row.bit_length()-1]=(row,mask)
    result=0
    while value:
        if value.bit_length()-1 not in basis:
            raise StrataError('exact minimum witness operand is unavailable')
        row,mask=basis[value.bit_length()-1]
        value^=row;result^=mask
    return result


def exact_minimum(outputs,n_inputs,*,care_set=None,max_and=3,timeout=1.0,max_states=10000,max_expansions=100000):
    """Exhaust XOR/AND circuits on all inputs, or only the supplied care rows.

    Values outside a care set are unspecified. A successful construction is a
    globally defined XOR/AND circuit, but its correctness is promised-row only.
    """
    _validate_search_limits(max_and,timeout,max_states,max_expansions)
    if type(n_inputs)is not int or n_inputs<0 or (care_set is None and n_inputs>8):
        raise StrataError('full-domain exact floors require 0..8 inputs; use a care set for wider boundaries')
    outputs=_normalize(outputs)
    if any(q>=n_inputs for poly in outputs for term in poly for q in term):
        raise StrataError('floor input outside declared domain')
    if care_set is None:
        input_rows=[[(x>>q)&1 for q in range(n_inputs)] for x in range(1<<n_inputs)]
    else:
        try:input_rows=[list(row) for row in care_set]
        except TypeError as exc:raise StrataError('care set must contain bit-vector rows') from exc
        if not input_rows or any(len(row)!=n_inputs or any(type(b)is not int or b not in (0,1) for b in row) for row in input_rows):
            raise StrataError('care-set input width or bit value is invalid')
    output_rows=[[sum(all(row[q] for q in term) for term in poly)%2 for poly in outputs] for row in input_rows]
    lower=counting_bound(outputs)['value'] if care_set is None else 0
    report=exact_minimum_on_rows(input_rows,output_rows,max_and=max_and,timeout=timeout,
                                 max_states=max_states,max_expansions=max_expansions,_lower_bound=lower)
    report['scope']['domain']='full Boolean cube' if care_set is None else 'explicit care set; other inputs are unspecified'
    report['anf_contract_hash']=digest(outputs)
    return report


def exact_minimum_on_rows(input_rows,output_rows,*,max_and=3,timeout=1.0,max_states=10000,
                          max_expansions=100000,_lower_bound=0):
    """Complete span search on promised rows, with bounded state/affine expansion."""
    _validate_search_limits(max_and,timeout,max_states,max_expansions)
    start=monotonic()
    input_rows,output_rows=_canonical_rows(input_rows,output_rows)
    available,targets=_row_tables(input_rows,output_rows)
    initial=_basis(available)
    nonlinear_dimension = len(_basis((*initial, *targets))) - len(initial)
    lower = max(_lower_bound, nonlinear_dimension)
    bounds = {}
    # A bounded finite-row certificate, valid even on an incomplete care set.
    if max_and >= 4 and len(initial) <= 12 and len(input_rows) <= 256:
        from .tight_span import quadratic_quotient_floor
        bounds['quadratic_quotient'] = quadratic_quotient_floor(initial, targets)
        lower = max(lower, bounds['quadratic_quotient'])
    from .quadratic_rank_floor import quadratic_rank_floor
    polar = quadratic_rank_floor(input_rows, output_rows)
    if polar is not None:
        bounds['quadratic_polar_rank'] = polar
        lower = max(lower, polar['value'])
    contract_hash=digest(dict(inputs=input_rows,outputs=output_rows))
    frontier={initial:()};expansions=states=0
    def receipt(kind,value,reason,depth,witness=None):
        report=dict(kind=kind,resource='and_count',value=value,reason=reason,
                    scope=dict(inputs=len(input_rows[0]),outputs=len(output_rows[0]),assignments=len(input_rows),
                               domain='explicit care set; other inputs are unspecified'),
                    model='free XOR/NOT/constants; two-input AND straight-line circuits restricted to promised rows',
                    proven_lower_bound=lower,searched_up_to=depth,
                    states=states,expansions=expansions,seconds=monotonic()-start,
                    limits=dict(max_and=max_and,timeout=timeout,max_states=max_states,max_expansions=max_expansions),
                    witness=None,contract_hash=contract_hash)
        if bounds:
            report['structural_lower_bounds'] = bounds
        if witness is not None:
            current=list(available);gates=[]
            for a,b in witness:
                gates.append(dict(left=a,right=b,product=a&b,left_coefficients=_coefficients(a,current),
                                  right_coefficients=_coefficients(b,current)))
                current.append(a&b)
            report.update(witness=gates,output_coefficients=[_coefficients(t,current) for t in targets],
                          witness_encoding='bit 0 is constant 1, then boundary inputs, then preceding AND results')
            verify_minimum_witness(report,input_rows,output_rows)
        return report
    if 5 <= lower <= max_and and lower == nonlinear_dimension + 1:
        from .tight_span import construct_one_helper
        gates, pairs = construct_one_helper(initial, targets,
            max_pairs=min(max_expansions, max_states, 100000), deadline=start+timeout/4)
        if gates is not None and len(gates) == lower:
            return dict(receipt('exact_small',lower,'replayed witness meets the proved rank floor',
                                lower,gates), backend='one_helper_span', pairs_checked=pairs)
    if max_and >= 4:
        # Compact the available affine basis without changing the XOR/AND
        # model. The emitted witness is decoded on the original input columns.
        from .and_sat import solve_and_count
        from importlib.util import find_spec
        if find_spec('pysat') is not None:
            basis_columns = [available[0]]
            basis = _basis(basis_columns)
            for column in available[1:]:
                if not _inside(column, basis):
                    basis_columns.append(column)
                    basis = _basis((*basis, column))
            sat_trials = []
            for depth in range(lower, max_and+1):
                status, witness, clauses = solve_and_count(basis_columns, targets, len(input_rows), depth,
                    deadline=start+timeout, clause_limit=min(max_expansions, 2_000_000),
                    conflict_limit=min(max_states, max_expansions))
                sat_trials.append(dict(depth=depth, status=status, clauses=clauses))
                if status == 'sat':
                    return dict(receipt('exact_small',depth,'minimum in exact care-row SAT search',depth,witness),
                                backend='sat', sat_trials=sat_trials)
                if status != 'unsat':
                    return dict(receipt('unknown',None,status,depth-1), backend='sat', sat_trials=sat_trials)
                lower = depth+1
            return dict(receipt('unknown',None,'no construction within AND count bound; minimum not established',max_and),
                        backend='sat', sat_trials=sat_trials)
    for depth in range(max_and+1):
        for basis,witness in frontier.items():
            if monotonic()-start>timeout:
                return receipt('unknown',None,'search timeout; no minimum claimed',depth)
            if all(_inside(t,basis) for t in targets):
                return receipt('exact_small',depth,'minimum in complete care-set XOR/AND search',depth,witness)
        lower=max(lower,depth+1)
        if depth==max_and:break
        following={}
        for basis,witness in frontier.items():
            # Never allocate an exponential affine span before checking the
            # explicit expansion cap. Exhausting this cap is not a proof.
            if (1<<len(basis))>max_expansions+1:
                return receipt('unknown',None,'affine span expansion budget; no minimum claimed',depth)
            span=[0]
            for row in basis:
                if monotonic()-start>timeout:
                    return receipt('unknown',None,'search timeout; no minimum claimed',depth)
                span += [v^row for v in span]
            # Equivalent spaces can have many witnesses. Prefer already
            # available wires before affine combinations to reduce native
            # operand preparation in a later reversible lowering.
            native_values=[*available[1:],*[a&b for a,b in witness],available[0],0]
            preferred={value:index for index,value in reversed(list(enumerate(native_values)))}
            span.sort(key=lambda value:(preferred.get(value,len(native_values)),value))
            for i,a in enumerate(span):
                for j in range(i+1,len(span)):
                    if monotonic()-start>timeout:
                        return receipt('unknown',None,'search timeout; no minimum claimed',depth)
                    if expansions>=max_expansions:
                        return receipt('unknown',None,'expansion budget exhausted; no minimum claimed',depth)
                    expansions+=1;b=span[j];product=a&b
                    if _inside(product,basis):continue
                    child=_basis((*basis,product))
                    if child not in following:
                        if states>=max_states:
                            return receipt('unknown',None,'state budget exhausted; no minimum claimed',depth)
                        following[child]=(*witness,(a,b));states+=1
                        if all(_inside(target,child) for target in targets):
                            return receipt('exact_small',depth+1,'minimum in complete care-set XOR/AND search',
                                           depth+1,following[child])
        frontier=following
    return receipt('unknown',None,'no construction within AND count bound; minimum not established',max_and)


def verify_minimum_witness(receipt,input_rows,output_rows):
    """Replay an affine/AND construction; this checks an upper bound, not optimality."""
    input_rows,output_rows=_canonical_rows(input_rows,output_rows)
    if receipt.get('contract_hash')!=digest(dict(inputs=input_rows,outputs=output_rows)):
        raise StrataError('exact minimum witness contract hash mismatch')
    available,targets=_row_tables(input_rows,output_rows)
    def combination(mask):
        if type(mask)is not int or mask<0 or mask>>len(available):
            raise StrataError('exact minimum witness references an unavailable operand')
        result=0
        for i,row in enumerate(available):
            if mask>>i&1:result^=row
        return result
    gates=receipt.get('witness')
    if not isinstance(gates,list) or receipt.get('value')!=len(gates):
        raise StrataError('exact minimum witness gate count mismatch')
    try:
        for gate in gates:
            a,b=combination(gate['left_coefficients']),combination(gate['right_coefficients'])
            if (gate['left'],gate['right'],gate['product'])!=(a,b,a&b):
                raise StrataError('exact minimum witness AND replay mismatch')
            available.append(a&b)
        if [combination(mask) for mask in receipt['output_coefficients']]!=targets:
            raise StrataError('exact minimum witness output replay mismatch')
    except (KeyError,TypeError) as exc:
        raise StrataError('malformed exact minimum witness') from exc
    return dict(status='verified',and_count=len(gates),scope='construction on the specified care set only; see the independent lower-bound evidence for optimality')
