"""GF(2) bilinear tensors, scoped rank bounds and executable constructions.

Tensor rank prices products of a left-linear and a right-linear form with
arbitrary XOR fanout. It is not a general AND-circuit or quantum-resource floor.
Every decomposition is an upper bound until matched by a proved lower bound.
"""
from itertools import product
from math import prod
from time import monotonic

from .model import StrataError, digest


MODEL = 'GF(2) bilinear algorithms: left-linear times right-linear products; free XOR fanout'
SCOPE = 'the supplied fixed-size tensor and declared input bipartition; not a general AND or quantum-cost floor'


def _wires(values, label):
    result = tuple(values)
    if any(type(q) is not int or q < 0 for q in result) or len(set(result)) != len(result):
        raise StrataError(label + ' must contain distinct nonnegative wire IDs')
    return result


def _normalize(poly):
    if isinstance(poly, dict):
        poly = poly.get('monomials')
    result = set()
    try:
        for raw in poly:
            term = frozenset(raw)
            if any(type(q) is not int or q < 0 for q in term):
                raise StrataError('tensor ANF needs nonnegative physical input IDs')
            if term in result: result.remove(term)
            else: result.add(term)
    except TypeError as exc:
        raise StrataError('tensor outputs must be ANF monomial lists') from exc
    return sorted((sorted(term) for term in result), key=lambda term: (len(term), term))


def tensor_from_anf(outputs, left, right, *, output_wires=None):
    """Extract bilinear *deltas*, retaining affine terms for reversible lowering.

Accept a sequence of ANFs or a mapping like a semantic accumulator's ``deltas``.
For full output functions the caller must first subtract each target's identity
term. Inputs are preserved and targets must be disjoint from both input words.
"""
    left, right = _wires(left, 'left inputs'), _wires(right, 'right inputs')
    if set(left) & set(right): raise StrataError('tensor input bipartition overlaps')
    labels = [str(name) for name in outputs] if isinstance(outputs, dict) else None
    polynomials = [_normalize(poly) for poly in (outputs.values() if isinstance(outputs, dict) else outputs)]
    labels = labels if labels is not None else [str(i) for i in range(len(polynomials))]
    if not polynomials: raise StrataError('tensor needs at least one output')
    if len(set(labels))!=len(labels): raise StrataError('tensor output labels must be distinct')
    if len(left)*len(right)*len(polynomials)>4_000_000:
        raise StrataError('tensor exceeds the bounded dense coefficient space')
    targets = None if output_wires is None else _wires(output_wires, 'output wires')
    if targets is not None and (len(targets) != len(polynomials) or set(targets) & set(left + right)):
        raise StrataError('tensor output wires must be disjoint and match output count')
    a, b = {q:i for i,q in enumerate(left)}, {q:i for i,q in enumerate(right)}
    entries, affine = [], []
    for k, poly in enumerate(polynomials):
        affine.append([])
        for term in poly:
            if set(term) - set(left + right):
                raise StrataError('tensor output references a wire outside the declared input words')
            if len(term) <= 1:
                affine[k].append(term)
            elif len(term) == 2 and len(set(term) & set(left)) == 1:
                x = next(q for q in term if q in a)
                y = next(q for q in term if q in b)
                entries.append([a[x], b[y], k])
            else:
                raise StrataError('output is not bilinear in the declared bipartition')
    tensor = dict(format='strata.gf2_bilinear_tensor.v1', left=list(left), right=list(right),
                  outputs=labels, output_wires=list(targets) if targets is not None else None,
                  shape=[len(left),len(right),len(polynomials)], entries=sorted(entries),
                  affine=affine, model=MODEL)
    tensor['tensor_hash'] = digest(tensor)
    return tensor


def _validate(tensor):
    if not isinstance(tensor, dict) or tensor.get('format') != 'strata.gf2_bilinear_tensor.v1':
        raise StrataError('expected a typed GF(2) tensor')
    value = {key:item for key,item in tensor.items() if key != 'tensor_hash'}
    if digest(value) != tensor.get('tensor_hash'):
        raise StrataError('tensor identity mismatch')
    left, right = _wires(tensor['left'],'left inputs'), _wires(tensor['right'],'right inputs')
    shape = tensor['shape']
    if (shape != [len(left),len(right),len(tensor['outputs'])] or not shape[2]
        or any(type(n) is not int or n<0 for n in shape) or prod(shape)>4_000_000 or set(left)&set(right)):
        raise StrataError('invalid tensor shape or partition')
    if len(tensor['affine'])!=shape[2] or any(_normalize(poly)!=poly or any(len(t)>1 or set(t)-set(left+right) for t in poly) for poly in tensor['affine']):
        raise StrataError('invalid tensor affine remainder')
    seen = set()
    for entry in tensor['entries']:
        if len(entry)!=3 or any(type(i) is not int or not 0<=i<n for i,n in zip(entry,shape)) or tuple(entry) in seen:
            raise StrataError('invalid tensor entry')
        seen.add(tuple(entry))
    return shape


def _rank(rows):
    return len(_basis(rows))


def _basis(rows):
    pivots = {}
    for value in rows:
        while value:
            bit = value.bit_length()-1
            if bit not in pivots:
                pivots[bit] = value
                break
            value ^= pivots[bit]
    return list(pivots.values())


def _bits(value):
    while value:
        bit = value & -value
        yield bit.bit_length()-1
        value ^= bit


def _encoded(tensor):
    _, b, c = tensor['shape']
    return sum(1 << ((i*b+j)*c+k) for i,j,k in tensor['entries'])


def _flattening_rows(encoded, shape):
    a,b,c = shape
    rows = [[0]*a,[0]*b,[0]*c]
    for index in _bits(encoded):
        pair,k = divmod(index,c)
        i,j = divmod(pair,b)
        rows[0][i] ^= 1 << (j*c+k)
        rows[1][j] ^= 1 << (i*c+k)
        rows[2][k] ^= 1 << (i*b+j)
    return rows


def _flattenings(encoded, shape):
    return [_rank(mode) for mode in _flattening_rows(encoded,shape)]


def _substitution(encoded, shape, *, max_combinations, timeout):
    """Certify d-1+minrank(slice space), only using proved minrank values.

Proof: reduce the selected mode to its d-dimensional slice span. In a rank-r
decomposition choose d-1 independent first factors. A nonzero linear functional
annihilating them yields a nonzero matrix slice with at most r-d+1 rank-one
summands. Therefore minrank <= r-d+1 over the same field (here GF(2)).
"""
    if type(max_combinations) is not int or max_combinations<0 or type(timeout) not in (int,float) or not 0<timeout<float('inf'):
        raise StrataError('invalid substitution bound budget')
    start=monotonic();used=0;result={}
    a,b,c=shape
    for name,rows,(matrix_rows,matrix_columns) in zip(('left','right','output'),_flattening_rows(encoded,shape),((b,c),(a,c),(a,b))):
        basis=_basis(rows);dimension=len(basis)
        total=(1<<dimension)-1
        minimum=None;checked=0;current=0;previous=0
        reason='complete nonzero slice-space enumeration'
        for index in range(1,total+1):
            if used>=max_combinations or monotonic()-start>timeout:
                reason='budget exhausted; observed minimum is only an upper bound on minrank'
                break
            gray=index^(index>>1)
            current ^= basis[(gray^previous).bit_length()-1]
            previous=gray
            mask=(1<<matrix_columns)-1
            rank=_rank((current>>(row*matrix_columns))&mask for row in range(matrix_rows))
            minimum=rank if minimum is None else min(minimum,rank)
            used+=1;checked+=1
            if minimum==1:
                reason='nonzero slice of rank one matches the universal minrank lower bound'
                break
        established=dimension==0 or checked==total or minimum==1
        result[name]=dict(status='proved' if established else 'unknown',slice_space_dimension=dimension,
            minimum_nonzero_rank=(minimum or 0) if established else None,
            observed_minimum_rank=minimum,combinations_checked=checked,total_nonzero_combinations=total,
            value=(dimension-1+minimum if dimension else 0) if established else None,reason=reason)
    values=[row['value'] for row in result.values() if row['value'] is not None]
    return dict(kind='proof' if values else 'unknown',value=max(values) if values else None,per_mode=result,
        model=MODEL,scope=SCOPE,formula='slice_space_dimension - 1 + minimum_nonzero_slice_rank',
        reasoning='Annihilate d-1 independent factors in a rank-r decomposition; a nonzero residual slice has matrix rank at most r-d+1.',
        limits=dict(max_combinations=max_combinations,timeout=timeout),combinations_checked=used)


def quadratic_bounds(outputs):
    """Scalar alternating-matrix rank/2 bounds, never summed across outputs."""
    polynomials = [_normalize(poly) for poly in (outputs.values() if isinstance(outputs,dict) else outputs)]
    result = []
    for poly in polynomials:
        if any(len(term)>2 for term in poly): raise StrataError('quadratic bound requires degree at most two')
        wires = sorted({q for term in poly for q in term})
        index = {q:i for i,q in enumerate(wires)}
        rows = [0]*len(wires)
        for term in poly:
            if len(term)==2:
                i,j = (index[q] for q in term)
                rows[i] ^= 1 << j
                rows[j] ^= 1 << i
        rank = _rank(rows)
        result.append(dict(alternating_rank=rank, value=rank//2, inputs=wires))
    return dict(kind='proof',resource='quadratic_products',value=max((r['value'] for r in result),default=0),
                per_output=result, aggregation='maximum, not sum; products may be shared',
                model='quadratic forms over GF(2), expressed as sums of products of affine linear forms',
                scope='supplied fixed-size scalar output forms; no reversible-cleanup or measurement cost claim',
                contract_hash=digest(polynomials))


def tensor_bounds(tensor, *, max_substitution_combinations=4095, substitution_timeout=1.0):
    shape = _validate(tensor)
    ranks = _flattenings(_encoded(tensor),shape)
    substitution=_substitution(_encoded(tensor),shape,max_combinations=max_substitution_combinations,timeout=substitution_timeout)
    deltas = [list(terms) for terms in tensor['affine']]
    for i,j,k in tensor['entries']:
        deltas[k].append([tensor['left'][i],tensor['right'][j]])
    return dict(kind='proof',resource='bilinear_products',shape=shape,
                flattening_ranks=dict(left=ranks[0],right=ranks[1],output=ranks[2]),
                value=max(*ranks,substitution['value'] or 0),model=MODEL,scope=SCOPE,tensor_hash=tensor['tensor_hash'],
                reasoning='Each rank-one tensor contributes matrix rank at most one in every flattening.',
                quadratic=quadratic_bounds(deltas),
                substitution_bound=substitution)


def _outer(left,right,out,shape):
    _,b,c = shape
    return sum(1 << ((i*b+j)*c+k) for i,j,k in product(_bits(left),_bits(right),_bits(out)))


def verify_decomposition(tensor,decomposition):
    shape = _validate(tensor)
    encoded = 0
    for term in decomposition:
        if not isinstance(term,dict) or set(term)!={'left_mask','right_mask','output_mask'}:
            raise StrataError('decomposition requires typed left/right/output masks')
        masks = [term[name] for name in ('left_mask','right_mask','output_mask')]
        if any(type(mask) is not int or not 0<mask<(1<<n) for mask,n in zip(masks,shape)):
            raise StrataError('decomposition mask outside tensor dimensions')
        encoded ^= _outer(*masks,shape)
    valid = encoded == _encoded(tensor)
    return dict(passed=valid, tensor_hash=tensor['tensor_hash'], decomposition_hash=digest(decomposition),
                evidence='exact coefficient equality over GF(2)',scope=SCOPE,
                product_count=len(decomposition),model=MODEL)


def search_decomposition(tensor, *, max_rank=4, timeout=1.0, max_catalog=4096, max_expansions=100000):
    """Bounded complete rank-one search; return a verified baseline on timeout.

The bounded search enumerates all nonzero factor vectors only when the catalog
fits. Search failures never turn a found construction into a lower bound.
"""
    shape = _validate(tensor)
    if (type(max_rank) is not int or max_rank<0 or type(max_catalog) is not int or max_catalog<1
        or type(max_expansions) is not int or max_expansions<1 or type(timeout) not in (int,float)
        or not 0<timeout<float('inf')):
        raise StrataError('invalid tensor search budget')
    start = monotonic()
    grouped = {}
    for i,j,k in tensor['entries']: grouped[i,j] = grouped.get((i,j),0) ^ (1<<k)
    baseline = [dict(left_mask=1<<i,right_mask=1<<j,output_mask=mask) for (i,j),mask in sorted(grouped.items())]
    bounds = tensor_bounds(tensor,substitution_timeout=min(timeout,.25))
    lower = bounds['value']
    candidate = baseline
    expansions = 0
    searched = None
    reason = 'direct coordinate-product construction'
    catalog_size = prod((1<<n)-1 for n in shape)
    optimal = len(candidate)==lower

    class BudgetExceeded(Exception): pass
    def check_budget():
        nonlocal expansions
        expansions += 1
        if expansions>max_expansions or monotonic()-start>timeout: raise BudgetExceeded

    if not optimal and catalog_size<=max_catalog and max_rank>=lower:
        catalog = []
        try:
            for x,y,z in product(*(range(1,1<<n) for n in shape)):
                check_budget()
                catalog.append((_outer(x,y,z,shape),dict(left_mask=x,right_mask=y,output_mask=z)))
            lookup = {encoded:i for i,(encoded,_) in enumerate(catalog)}
            def find(residual,count,first):
                check_budget()
                if not count:return () if not residual else None
                if max(_flattenings(residual,shape))>count:return None
                if count==1:
                    i=lookup.get(residual)
                    return (i,) if i is not None and i>=first else None
                for i in range(first,len(catalog)-count+1):
                    found=find(residual^catalog[i][0],count-1,i+1)
                    if found is not None:return (i,*found)
                return None
            for rank in range(lower,min(max_rank,len(baseline)-1)+1):
                found=find(_encoded(tensor),rank,0)
                if found is not None:
                    candidate=[catalog[i][1] for i in found]
                    optimal=True
                    reason='complete search below construction rank within the declared bilinear model'
                    break
                searched=rank
            if searched is not None and searched>=len(baseline)-1:
                optimal=True
                reason='complete search excludes every smaller bilinear decomposition'
        except BudgetExceeded:
            reason='search budget exhausted; retained verified coordinate construction'
    elif catalog_size>max_catalog:
        reason='rank-one factor catalog exceeds budget; retained verified coordinate construction'
    if len(candidate)==lower:
        reason='construction matches proved tensor lower bound'
    proof=verify_decomposition(tensor,candidate)
    if not proof['passed']:raise StrataError('internal tensor synthesis identity failure')
    return dict(status='optimal_in_bilinear_model' if optimal else 'construction',
                evidence_kind='exact_small' if optimal else 'verified_construction',
                decomposition=candidate,rank=len(candidate),upper_bound=len(candidate),
                proven_lower_bound=len(candidate) if optimal else lower,optimal_in_model=optimal,model=MODEL,scope=SCOPE,
                tensor_hash=tensor['tensor_hash'],verification=proof,bounds=bounds,
                search=dict(reason=reason,catalog_size=catalog_size,searched_up_to=searched,excluded_by_bounds_below=lower,
                            max_rank=max_rank,expansions=expansions,seconds=monotonic()-start,
                            limits=dict(timeout=timeout,max_catalog=max_catalog,max_expansions=max_expansions)))


def lower_decomposition(tensor, decomposition, *, output_wires=None, clean_helpers=None, max_operations=100000):
    """Emit X/CX/CCX preserving dirty outputs and restoring clean helpers.

Two clean helpers form linear operands. Products directly accumulate into an
output pivot, with conjugating CNOTs distributing to other dirty outputs. This
uses one CCX per product regardless of fanout, without a product ancilla.
"""
    proof=verify_decomposition(tensor,decomposition)
    if not proof['passed']:raise StrataError('cannot lower an incorrect tensor decomposition')
    outputs=tensor.get('output_wires') if output_wires is None else output_wires
    if outputs is None:raise StrataError('reversible tensor lowering needs physical output wires')
    outputs=_wires(outputs,'output wires')
    inputs=tensor['left']+tensor['right']
    if len(outputs)!=tensor['shape'][2] or set(outputs)&set(inputs):
        raise StrataError('tensor output wires must be disjoint and match output count')
    if type(max_operations) is not int or max_operations<1:raise StrataError('invalid lowering operation budget')
    needs_left=any(term['left_mask'].bit_count()>1 for term in decomposition)
    needs_right=any(term['right_mask'].bit_count()>1 for term in decomposition)
    needed=int(needs_left)+int(needs_right)
    if clean_helpers is None:
        start=max(inputs+list(outputs),default=-1)+1
        helpers=tuple(range(start,start+needed))
    else:
        helpers=_wires(clean_helpers,'clean helpers')
        if len(helpers)<needed:raise StrataError('insufficient clean helpers for linear operands')
        helpers=helpers[:needed]
    if set(helpers)&set(inputs+list(outputs)):raise StrataError('clean helpers overlap inputs or dirty outputs')
    left_helper=helpers[0] if needs_left else None
    right_helper=helpers[int(needs_left)] if needs_right else None
    lines=[]
    def emit(line):
        if len(lines)>=max_operations:raise StrataError('tensor lowering exceeds operation budget')
        lines.append(line)
    for term in decomposition:
        controls=[];prepare=[]
        for name,wires,helper in [('left_mask',tensor['left'],left_helper),('right_mask',tensor['right'],right_helper)]:
            factors=[wires[i] for i in _bits(term[name])]
            if len(factors)==1:controls.append(factors[0])
            else:
                controls.append(helper)
                prepare.extend(f'CX q{q} q{helper}' for q in factors)
        for line in prepare:emit(line)
        targets=[outputs[k] for k in _bits(term['output_mask'])]
        pivot=targets[0]
        for q in targets[1:]:emit(f'CX q{pivot} q{q}')
        emit(f'CCX q{controls[0]} q{controls[1]} q{pivot}')
        for q in reversed(targets[1:]):emit(f'CX q{pivot} q{q}')
        for line in reversed(prepare):emit(line)
    for output,poly in zip(outputs,tensor['affine']):
        for term in poly:
            if not term:emit(f'X q{output}')
            else:emit(f'CX q{term[0]} q{output}')
    return dict(status='constructed',candidate='\n'.join(lines),tensor_hash=tensor['tensor_hash'],
                decomposition_hash=proof['decomposition_hash'],algebraic_verification=proof,
                resources=dict(toffoli=len(decomposition),cx=sum(line.startswith('CX ') for line in lines),
                               instructions=len(lines),Q=max(inputs+list(outputs)+list(helpers),default=-1)+1,
                               clean_helpers=len(helpers),dirty_outputs=len(outputs)),
                boundary=dict(preserved_inputs=inputs,arbitrary_initial_outputs=list(outputs),
                              clean_inputs=list(helpers),restored_clean_outputs=list(helpers)),
                semantics='outputs XOR supplied bilinear and affine deltas; inputs preserved; helpers zero on entry and exit; phase +1',
                cost_scope='exact emitted X/CX/CCX counts; no task-level host objective claimed')
