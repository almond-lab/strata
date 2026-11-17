"""GF(2) bilinear tensors, scoped rank bounds and executable constructions.

Tensor rank prices products of a left-linear and a right-linear form with
arbitrary XOR fanout. It is not a general AND-circuit or quantum-resource floor.
Every decomposition is an upper bound until matched by a proved lower bound.
"""
from collections import deque
from itertools import combinations, product
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
    code_bounds = {}
    for mode,row in substitution['per_mode'].items():
        k,d = row['slice_space_dimension'],row['minimum_nonzero_rank']
        terms = [((d-1) >> i) + 1 for i in range(k)] if d is not None else None
        code_bounds[mode] = dict(status=row['status'],dimension=k,minimum_distance_lower_bound=d,
                                 terms=terms,value=sum(terms) if terms is not None else None)
    griesmer = max((row['value'] for row in code_bounds.values() if row['value'] is not None),default=None)
    deltas = [list(terms) for terms in tensor['affine']]
    for i,j,k in tensor['entries']:
        deltas[k].append([tensor['left'][i],tensor['right'][j]])
    return dict(kind='proof',resource='bilinear_products',shape=shape,
                flattening_ranks=dict(left=ranks[0],right=ranks[1],output=ranks[2]),
                value=max(*ranks,substitution['value'] or 0,griesmer or 0),model=MODEL,scope=SCOPE,tensor_hash=tensor['tensor_hash'],
                reasoning='Each rank-one tensor contributes matrix rank at most one in every flattening.',
                quadratic=quadratic_bounds(deltas),
                substitution_bound=substitution,
                griesmer_bound=dict(value=griesmer,per_mode=code_bounds,field='GF(2)',
                    reasoning='In compressed mode dimension k, each nonzero slice uses at least d rank-one terms. '
                              'The resulting binary linear code has dimension k and distance at least d. '
                              'Its length is at least the sum of ceil(d/2^i) over 0 <= i < k.'))


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


def extract_decomposition(circuit, tensor, *, clean_inputs=(), max_operations=10000,
                          max_variables=4096):
    """Recover CCX products and their final XOR fanout from a native window.

    X/CX/SWAP propagate affine values and product provenance. Every CCX must
    multiply affine forms from opposite sides of the declared partition.
    Constants contribute to the affine remainder, not to tensor rank. All
    physical outputs, preserved inputs and clean restoration are checked.
    Phase gates (including CCZ), measurements and nonlinear controls are outside
    this accumulator extractor's scope, even if their final effects cancel.
    """
    from .semantic import _lines, _parse

    shape = _validate(tensor)
    clean = set(_wires(clean_inputs, 'clean inputs'))
    targets = tensor.get('output_wires')
    scope = ('phase-free X/CX/SWAP/CCX bilinear accumulators with affine controls '
             'in the declared partition; excludes CCZ phase tensors, measurement '
             'instruments, conditional operations and nonlinear controls')
    if targets is None:
        raise StrataError('gate decomposition extraction needs physical output wires')
    targets = _wires(targets, 'output wires')
    if (len(targets) != shape[2] or set(targets) & set(tensor['left'] + tensor['right'])
        or clean & set(tensor['left'] + tensor['right'])):
        raise StrataError('extraction requires disjoint arbitrary input words and output wires')
    if any(type(n) is not int or n < 1 for n in (max_operations, max_variables)):
        raise StrataError('invalid decomposition extraction budget')
    operations = []
    wires = set(tensor['left'] + tensor['right'] + list(targets)) | clean
    source = str(circuit) if hasattr(circuit, 'num_qubits') else circuit
    for index, line in enumerate(_lines(source)):
        if index >= max_operations:
            return dict(status='unknown', reason='extraction operation budget', scope=scope)
        item = _parse(line)
        wires.update(item['wires'])
        if len(wires) > max_variables:
            return dict(status='unknown', reason='extraction variable budget', scope=scope)
        if item['status'] == 'metadata':
            continue
        if (item['status'] != 'supported' or item.get('condition') is not None
            or item['op'] not in {'X', 'CX', 'SWAP', 'CCX'}):
            return dict(status='not_applicable', reason='unsupported accumulator operation',
                        operation=item['op'], at=index, scope=scope)
        operations.append((index, item))
    if len(wires) > max_variables:
        return dict(status='unknown', reason='extraction variable budget', scope=scope)
    indices = {q: i for i, q in enumerate(sorted(wires))}
    left_bits = sum(1 << indices[q] for q in tensor['left'])
    right_bits = sum(1 << indices[q] for q in tensor['right'])
    # Each value is (linear inputs, constant, bilinear coefficients, gate tokens).
    initial = {q: (0 if q in clean else 1 << indices[q], 0, 0, 0) for q in wires}
    state = dict(initial)
    products = []

    def xor(a, b):
        return tuple(x ^ y for x, y in zip(a, b))

    for index, item in operations:
        op, qs = item['op'], item['wires']
        if op == 'X':
            state[qs[0]] = xor(state[qs[0]], (0, 1, 0, 0))
        elif op == 'CX':
            state[qs[1]] = xor(state[qs[1]], state[qs[0]])
        elif op == 'SWAP':
            state[qs[0]], state[qs[1]] = state[qs[1]], state[qs[0]]
        else:
            x, y = state[qs[0]], state[qs[1]]
            if x[2] or y[2]:
                return dict(status='not_applicable', reason='CCX has nonlinear controls', at=index, scope=scope)
            if not (x[0] & ~left_bits or y[0] & ~right_bits):
                pass
            elif not (y[0] & ~left_bits or x[0] & ~right_bits):
                x, y = y, x
            else:
                return dict(status='not_applicable', reason='CCX controls do not respect the declared bipartition',
                            at=index, scope=scope)
            left = sum(1 << i for i, q in enumerate(tensor['left']) if x[0] >> indices[q] & 1)
            right = sum(1 << j for j, q in enumerate(tensor['right']) if y[0] >> indices[q] & 1)
            quadratic = _outer(left, right, 1, (shape[0], shape[1], 1))
            token = 0
            if quadratic:
                token = 1 << len(products)
                products.append(dict(at=index, controls=list(qs[:2]), target=qs[2],
                                     left_mask=left, right_mask=right))
            delta = ((x[0] if y[1] else 0) ^ (y[0] if x[1] else 0),
                     x[1] & y[1], quadratic, token)
            state[qs[2]] = xor(state[qs[2]], delta)

    expected = dict(initial)
    for k, q in enumerate(targets):
        linear = initial[q][0]
        constant = 0
        for term in tensor['affine'][k]:
            if term:
                linear ^= 1 << indices[term[0]]
            else:
                constant ^= 1
        quadratic = sum(1 << (i * shape[1] + j) for i, j, out in tensor['entries'] if out == k)
        expected[q] = (linear, constant, quadratic, 0)
    for q in sorted(wires):
        if state[q][:3] != expected[q][:3]:
            return dict(status='not_applicable', reason='circuit boundary does not match the supplied tensor',
                        wire=q, scope=scope, boundary_verification=dict(passed=False))

    masks = [0] * len(products)
    for k, q in enumerate(targets):
        for index in _bits(state[q][3]):
            masks[index] ^= 1 << k
    decomposition = [dict(left_mask=row['left_mask'], right_mask=row['right_mask'], output_mask=mask)
                     for row, mask in zip(products, masks) if mask]
    proof = verify_decomposition(tensor, decomposition)
    if not proof['passed']:
        raise StrataError('internal gate decomposition identity failure')
    return dict(status='verified', rank=len(decomposition), decomposition=decomposition,
                verification=proof, scope=scope, tensor_hash=tensor['tensor_hash'],
                ccx_count=sum(item['op'] == 'CCX' for _, item in operations),
                products=[dict(row, output_mask=mask) for row, mask in zip(products, masks)],
                boundary_verification=dict(passed=True, evidence='exact affine and bilinear coefficient equality on every physical wire',
                                           clean_inputs=sorted(clean), restored_clean_outputs=sorted(clean-set(targets)),
                                           arbitrary_initial_outputs=[q for q in targets if q not in clean],
                                           clean_output_accumulators=sorted(clean & set(targets)), phase='+1'))


_FACTORS = ('left_mask', 'right_mask', 'output_mask')


def _canonical_terms(terms):
    parity = set()
    for term in terms:
        term = tuple(term)
        if not all(term):
            continue
        if term in parity:
            parity.remove(term)
        else:
            parity.add(term)
    return tuple(sorted(parity))


def _dependent_reduction(terms, shape, check_budget):
    """Remove a product whose two-factor matrix lies in the others' span.

    A GF(2) relation among paired factors lets us XOR the removed product's
    third vector into each other vector in the relation, preserving the tensor.
    All three choices of the remaining vector are tried.
    """
    for remaining in (2, 1, 0):
        a, b = [axis for axis in range(3) if axis != remaining]
        basis = {}
        for index, term in enumerate(terms):
            check_budget()
            value = _outer(term[a], term[b], 1, (shape[a], shape[b], 1))
            relation = 1 << index
            while value:
                check_budget()
                pivot = value.bit_length() - 1
                if pivot not in basis:
                    basis[pivot] = (value, relation)
                    break
                vector, combination = basis[pivot]
                value ^= vector
                relation ^= combination
            if not value:
                changed = [list(t) for t in terms]
                for other in _bits(relation ^ (1 << index)):
                    changed[other][remaining] ^= term[remaining]
                del changed[index]
                return _canonical_terms(changed)
    return None


def _local_decomposition(tensor, seed, check_budget, max_states, accept, report):
    """Bounded exact dependency reductions and shared-factor flip-graph moves."""
    shape = tensor['shape']
    initial = tuple(tuple(t[name] for name in _FACTORS) for t in seed)
    current = _canonical_terms(initial)
    if len(current) < len(initial):
        accept(current)
        report['reductions'] += 1
    queue = deque([current])
    visited = {current}
    while queue and report['states_visited'] < max_states:
        current = queue.popleft()
        report['states_visited'] += 1
        while True:
            reduced = _dependent_reduction(current, shape, check_budget)
            if reduced is None:
                break
            accept(reduced)
            report['reductions'] += 1
            current = reduced
        for i, j in combinations(range(len(current)), 2):
            check_budget()
            for shared in range(3):
                if current[i][shared] != current[j][shared]:
                    continue
                other = [axis for axis in range(3) if axis != shared]
                for a, b in (other, other[::-1]):
                    check_budget()
                    changed = [list(t) for t in current]
                    changed[i][a] ^= current[j][a]
                    changed[j][b] ^= current[i][b]
                    flipped = _canonical_terms(changed)
                    if flipped in visited:
                        continue
                    report['flip_moves'] += 1
                    # Admit at most max_states distinct frontier states, so
                    # large ranks cannot turn a state limit into unbounded RAM.
                    if len(visited) >= max_states:
                        report['status'] = 'state_budget_exhausted'
                        continue
                    visited.add(flipped)
                    queue.append(flipped)
                    if len(flipped) < len(current):
                        accept(flipped)
        if report['status'] != 'state_budget_exhausted':
            report['status'] = 'searched_neighborhood'


def search_decomposition(tensor, *, max_rank=4, timeout=1.0, max_catalog=4096,
                         max_expansions=100000, seed=None, max_local_states=64):
    """Verified seed improvement plus bounded complete rank-one search.

    Seeded dependency reductions and shared-factor flips do not enumerate the
    full catalog. Only fully completed exhaustive rank exclusions strengthen a
    lower bound; neither a local-search failure nor an interrupted rank does.
    """
    shape = _validate(tensor)
    if (type(max_rank) is not int or max_rank<0 or type(max_catalog) is not int or max_catalog<1
        or type(max_expansions) is not int or max_expansions<1 or type(timeout) not in (int,float)
        or not 0<timeout<float('inf') or type(max_local_states) is not int or max_local_states < 0):
        raise StrataError('invalid tensor search budget')
    start = monotonic()
    grouped = {}
    for i,j,k in tensor['entries']: grouped[i,j] = grouped.get((i,j),0) ^ (1<<k)
    baseline = [dict(left_mask=1<<i,right_mask=1<<j,output_mask=mask) for (i,j),mask in sorted(grouped.items())]
    if seed is not None:
        seed = [dict(term) for term in seed]
        if not verify_decomposition(tensor, seed)['passed']:
            raise StrataError('tensor search seed does not match the supplied tensor')
    bounds = tensor_bounds(tensor,substitution_timeout=min(timeout,.25))
    lower = bounds['value']
    candidate = seed if seed is not None and len(seed) <= len(baseline) else baseline
    expansions = 0
    searched = None
    reasons = []
    catalog_size = prod((1<<n)-1 for n in shape)
    local = dict(status='not_run', initial_rank=None if seed is None else len(seed),
                 final_rank=None, reductions=0, flip_moves=0, states_visited=0,
                 exact_verifications=0, max_states=max_local_states,
                 scope='bounded dependency reductions and shared-factor flips; no rank exclusion')

    class BudgetExceeded(Exception): pass
    def check_budget():
        nonlocal expansions
        expansions += 1
        if expansions>max_expansions or monotonic()-start>timeout: raise BudgetExceeded

    def accept(terms):
        nonlocal candidate
        proposal = [dict(zip(_FACTORS, term)) for term in terms]
        if not verify_decomposition(tensor, proposal)['passed']:
            raise StrataError('internal local tensor search identity failure')
        local['exact_verifications'] += 1
        if len(proposal) < len(candidate):
            candidate = proposal

    if seed is not None and max_local_states and len(candidate) > lower:
        try:
            _local_decomposition(tensor, seed, check_budget, max_local_states, accept, local)
        except BudgetExceeded:
            local['status'] = 'budget_exhausted'
        local['final_rank'] = len(candidate)

    if len(candidate)>lower and catalog_size<=max_catalog and max_rank>=lower:
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
            for rank in range(lower,min(max_rank,len(candidate)-1)+1):
                found=find(_encoded(tensor),rank,0)
                if found is not None:
                    candidate=[catalog[i][1] for i in found]
                    lower=rank
                    reasons.append('complete search found a minimum-rank decomposition in the declared bilinear model')
                    break
                searched=rank
        except BudgetExceeded:
            reasons.append('search budget exhausted; retained best verified construction')
    elif catalog_size>max_catalog:
        reasons.append(f'rank-one factor catalog size {catalog_size} exceeds max_catalog {max_catalog}; retained best verified construction')
    if searched is not None:
        lower=max(lower,searched+1)
        reasons.append(f'complete exhaustive search excluded all ranks through {searched}, including ranks excluded by bounds')
    optimal=len(candidate)==lower
    if optimal:
        reasons.append('construction matches proved tensor lower bound')
    if not reasons:
        reasons.append('verified seeded construction' if seed is not None else 'direct coordinate-product construction')
    proof=verify_decomposition(tensor,candidate)
    if not proof['passed']:raise StrataError('internal tensor synthesis identity failure')
    return dict(status='optimal_in_bilinear_model' if optimal else 'construction',
                evidence_kind='exact_small' if optimal or searched is not None else 'verified_construction',
                decomposition=candidate,rank=len(candidate),upper_bound=len(candidate),
                proven_lower_bound=lower,optimal_in_model=optimal,model=MODEL,scope=SCOPE,
                tensor_hash=tensor['tensor_hash'],verification=proof,bounds=bounds,
                lower_bound_evidence=dict(kind='exact_small' if searched is not None else 'proof',
                                          value=lower,searched_up_to=searched,scope=SCOPE),
                search=dict(reason='; '.join(reasons),catalog_size=catalog_size,searched_up_to=searched,excluded_by_bounds_below=bounds['value'],
                            coordinate_rank=len(baseline),seed_rank=None if seed is None else len(seed),local_search=local,
                            max_rank=max_rank,expansions=expansions,seconds=monotonic()-start,
                            limits=dict(timeout=timeout,max_catalog=max_catalog,max_expansions=max_expansions,max_local_states=max_local_states)))


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
