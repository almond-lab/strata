"""Bounded exact semantics for native kickmix permutation/sign circuits.

An input variable denotes the arbitrary bit on a physical wire at the *window*
boundary.  In particular, register names and apparent scratch usage never imply
zero.  ANF monomials use physical wire numbers; [] is the constant one, while an
empty list of monomials is zero.  Addition is XOR and multiplication is AND in
GF(2)[x]/(x*x+x).  Unsupported instructions are barriers, not identities.
"""
from __future__ import annotations

from collections import Counter
from io import StringIO
from os import PathLike
import re
import shlex


ARITY = {'X': 1, 'CX': 2, 'CCX': 3, 'SWAP': 2,
         'Z': 1, 'CZ': 2, 'CCZ': 3, 'NEG': 0}
METADATA = {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG'}
PAID = {'CCX', 'CCZ', 'CSWAP'}
ZERO = frozenset()
ONE = frozenset({frozenset()})


class _BoundExceeded(Exception):
    pass


def _variable(q):
    return frozenset({frozenset({q})})


def _poly(value):
    terms = [sorted(term) for term in sorted(value, key=lambda t: (len(t), sorted(t)))]
    return dict(monomials=terms,
                expression=' ^ '.join('*'.join(f'x{q}' for q in t) or '1' for t in terms) or '0',
                degree=max(map(len, terms), default=0), term_count=len(terms))


def _limits(**limits):
    if any(type(v) is not int or v <= 0 for v in limits.values()):
        raise ValueError('semantic budgets must be positive integers')


def _lines(lines):
    if isinstance(lines, str):
        return StringIO(lines)
    return iter(lines)


def _parse(line):
    try:
        words = shlex.split(line, comments=True)
    except ValueError:
        return dict(op='', wires=(), status='opaque', reason='malformed_instruction')
    if not words:
        return dict(op='', wires=(), status='metadata')
    op = words[0]
    qs = tuple(int(w[1:]) for w in words[1:] if re.fullmatch(r'q[0-9]+', w))
    if 'if' in words or op in {'PUSH_CONDITION', 'POP_CONDITION'}:
        return dict(op=op, wires=qs, status='opaque', reason='classical_condition')
    if op in METADATA:
        return dict(op=op, wires=qs, status='metadata')
    if op not in ARITY:
        return dict(op=op, wires=qs, status='opaque', reason='unsupported_operation')
    if len(words) != ARITY[op] + 1 or len(qs) != ARITY[op] or len(set(qs)) != len(qs):
        return dict(op=op, wires=qs, status='opaque', reason='unsupported_operands')
    return dict(op=op, wires=qs, status='supported')


def _bipartition(polynomials):
    """Find a genuine disjoint factor partition of homogeneous quadratic terms."""
    graph = {}
    for value in polynomials:
        for term in value:
            if len(term) != 2:
                return None
            a, b = sorted(term)
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
    colors = {}
    for root in sorted(graph):
        if root in colors:
            continue
        colors[root] = 0
        stack = [root]
        while stack:
            a = stack.pop()
            for b in sorted(graph[a]):
                if b in colors and colors[b] == colors[a]:
                    return None
                if b not in colors:
                    colors[b] = 1 - colors[a]
                    stack.append(b)
    return [[q for q in sorted(colors) if colors[q] == c] for c in (0, 1)] if graph else None


def analyze_window(lines, *, max_variables=32, max_terms=4096,
                   max_operations=128, max_product_terms=65536):
    """Return complete exact ANF, or an explicit opaque/bounded result.

    ``lines`` is native instruction text or an iterable of instruction strings.
    The term budget bounds all live polynomials together, and the product budget
    bounds multiplication work across the whole window.  No partial result is
    marked exact.  Metadata supplies neither assumptions nor semantic intent.
    """
    _limits(max_variables=max_variables, max_terms=max_terms,
            max_operations=max_operations, max_product_terms=max_product_terms)
    parsed, wires, metadata, count = [], set(), [], 0
    for index, line in enumerate(_lines(lines)):
        count += 1
        if count > max_operations:
            return dict(status='bounded', reason='operation_budget', limit=max_operations)
        item = _parse(line)
        if item['status'] == 'opaque':
            return dict(status='opaque', reason=item['reason'], at=index, instruction=line.rstrip())
        if item['status'] == 'metadata':
            metadata.append(index)
            continue
        parsed.append((index, item['op'], item['wires']))
        wires.update(item['wires'])
        if len(wires) > max_variables:
            return dict(status='bounded', reason='variable_budget', limit=max_variables)
    state = {q: _variable(q) for q in sorted(wires)}
    phase, products = ZERO, 0
    writes = {q: [] for q in wires}
    consumed = {q: [] for q in wires}

    def check():
        if len(phase) + sum(map(len, state.values())) > max_terms:
            raise _BoundExceeded('term_budget')

    def mul(a, b):
        nonlocal products
        products += len(a) * len(b)
        if products > max_product_terms:
            raise _BoundExceeded('product_budget')
        result = set()
        for x in a:
            for y in b:
                term = x | y
                if term in result:
                    result.remove(term)
                else:
                    result.add(term)
                if len(result) > max_terms:
                    raise _BoundExceeded('term_budget')
        return frozenset(result)

    try:
        check()
        for index, op, qs in parsed:
            reads = qs[:-1] if op in {'CX', 'CCX'} else qs if op in {'Z', 'CZ', 'CCZ'} else ()
            for q in reads:
                if state[q] != _variable(q):
                    consumed[q].append(index)
            if op == 'X':
                state[qs[0]] ^= ONE
            elif op == 'CX':
                state[qs[1]] ^= state[qs[0]]
            elif op == 'CCX':
                state[qs[2]] ^= mul(state[qs[0]], state[qs[1]])
            elif op == 'SWAP':
                state[qs[0]], state[qs[1]] = state[qs[1]], state[qs[0]]
            else:
                contribution = ONE
                for q in qs:
                    contribution = mul(contribution, state[q])
                phase ^= contribution
            targets = qs if op == 'SWAP' else qs[-1:] if op in {'X', 'CX', 'CCX'} else ()
            for q in targets:
                writes[q].append(index)
            check()
    except _BoundExceeded as exc:
        return dict(status='bounded', reason=str(exc), operation_count=len(parsed),
                    limits=dict(max_terms=max_terms, max_product_terms=max_product_terms))

    changed = {q: p ^ _variable(q) for q, p in state.items() if p != _variable(q)}
    preserved = sorted(wires - changed.keys())
    features = []
    output_degree = max((len(t) for p in state.values() for t in p), default=0)
    if not changed and not phase:
        features.append(dict(kind='identity_map'))
    elif output_degree <= 1:
        features.append(dict(kind='affine_transform',
                             matrix={str(q): sorted(next(iter(t)) for t in p if t)
                                     for q, p in state.items()},
                             offset={str(q): int(frozenset() in p) for q, p in state.items()}))
    if changed and all(not (term & changed.keys()) for p in changed.values() for term in p):
        degree = max((len(t) for p in changed.values() for t in p), default=0)
        kind = 'affine_accumulator' if degree <= 1 else 'quadratic_accumulator' if degree == 2 else 'polynomial_accumulator'
        feature = dict(kind=kind, targets=sorted(changed),
                       deltas={str(q): _poly(p) for q, p in changed.items()})
        partition = _bipartition(changed.values())
        if partition is not None:
            feature.update(kind='bilinear_accumulator', factor_partition=partition)
        features.append(feature)
    episodes = []
    for q in preserved:
        if consumed[q] and len(writes[q]) >= 2:
            episodes.append(dict(wire=q, compute_at=writes[q][0],
                                 consumed_at=consumed[q], restored_at=writes[q][-1],
                                 initial_value=f'x{q}', final_value=f'x{q}'))
    if episodes:
        features.append(dict(kind='compute_consume_uncompute', restored_wires=episodes,
                             guarantee='restores arbitrary boundary values; does not prove clean ancillas'))
    if phase:
        features.append(dict(kind='phase_polynomial', exponent=_poly(phase), convention='(-1)^exponent'))
    kind = 'compute_consume_uncompute' if episodes else next(
        (f['kind'] for f in features if f['kind'].endswith('_accumulator')),
        features[0]['kind'] if features else 'boolean_polynomial_map')
    if not changed and phase:
        kind = 'phase_polynomial'
    return dict(status='exact', kind=kind, inputs=sorted(wires), boundary_width=len(wires),
                boundary=dict(input_width=len(wires), output_width=len(wires),
                              modified_width=len(changed), preserved_width=len(preserved)),
                outputs={str(q): _poly(p) for q, p in state.items()}, phase=_poly(phase),
                changes={str(q): _poly(p) for q, p in changed.items()},
                modified_wires=sorted(changed), preserved_wires=preserved,
                features=features, operation_count=len(parsed), instruction_count=count,
                metadata_indices=metadata, max_output_degree=output_degree,
                instruction_offsets='relative_to_window',
                assumptions='all boundary wire values arbitrary; no clean ancillas inferred',
                evidence='exact_boolean_anf_with_exact_sign_phase')


def _distinguishing_assignment(a, b, wires):
    """Find a point where two different ANFs disagree, without enumeration.

    A polynomial is split as q + x*r.  A nonzero q has a witness with x=0;
    otherwise nonzero r has a witness with x=1.  Bit masks keep each split
    linear in the current number of terms, including for sparse physical IDs.
    """
    bits = {q: 1 << i for i, q in enumerate(wires)}
    polynomial = {sum(bits[q] for q in term) for term in a} ^ {sum(bits[q] for q in term) for term in b}
    assignment = {str(q): 0 for q in wires}
    for q in wires:
        if polynomial == {0}:
            break
        bit = bits[q]
        independent = {term for term in polynomial if not term & bit}
        if independent:
            polynomial = independent
        else:
            assignment[str(q)] = 1
            polynomial = {term ^ bit for term in polynomial}
    return assignment


def _evaluate_terms(terms, assignment):
    value = 0
    for term in terms:
        value ^= int(all(assignment[str(q)] for q in term))
    return value


def equivalent_windows(original, candidate, **limits):
    """Compare complete canonical maps on the union of physical input wires.

    A wire absent from one side has its identity polynomial on that side.  Global
    sign is compared too.  Mismatches include a deterministic concrete input
    and both complete boundary results; phase values are sign-exponent bits.
    Unsupported/budget-limited windows return unknown.
    """
    before, after = analyze_window(original, **limits), analyze_window(candidate, **limits)
    if before['status'] != 'exact' or after['status'] != 'exact':
        return dict(status='unknown', equivalent=None, before=before, after=after,
                    evidence='exact_semantics_unavailable')
    wires = sorted(set(before['inputs']) | set(after['inputs']))
    differences = []
    for q in wires:
        identity = [[q]]
        a = before['outputs'].get(str(q), {}).get('monomials', identity)
        b = after['outputs'].get(str(q), {}).get('monomials', identity)
        if a != b:
            differences.append(q)
    phase_equal = before['phase']['monomials'] == after['phase']['monomials']
    same = not differences and phase_equal
    result = dict(status='equivalent' if same else 'different', equivalent=same,
                  differing_wires=differences, phase_equal=phase_equal, inputs=wires,
                  evidence='canonical_boolean_anf_equality_with_exact_sign_phase',
                  assumptions='all boundary wire values arbitrary; no clean ancillas inferred')
    if not same:
        if differences:
            q = differences[0]
            a = before['outputs'].get(str(q), {}).get('monomials', [[q]])
            b = after['outputs'].get(str(q), {}).get('monomials', [[q]])
        else:
            a, b = before['phase']['monomials'], after['phase']['monomials']
        assignment = _distinguishing_assignment(a, b, wires)
        result['counterexample'] = dict(
            inputs=assignment, kind='output' if differences else 'phase',
            before_outputs={str(q): _evaluate_terms(before['outputs'].get(str(q), {}).get('monomials', [[q]]), assignment)
                            for q in wires},
            after_outputs={str(q): _evaluate_terms(after['outputs'].get(str(q), {}).get('monomials', [[q]]), assignment)
                           for q in wires},
            before_phase=_evaluate_terms(before['phase']['monomials'], assignment),
            after_phase=_evaluate_terms(after['phase']['monomials'], assignment))
    return result


def lift_regions(circuit, *, max_ops=None, max_window=128, max_variables=32,
                 max_terms=4096, max_product_terms=65536):
    """Discover exact compound regions across native text or a loaded circuit.

    A PathLike is loaded through kickmix's text/binary autodetection.  All other
    inputs are native text or Circuit objects.  The entire artifact is scanned
    by default.  Regions are disjoint bounded windows; unknown spans and scoped
    condition bodies stay explicit.  Coverage of recognized compound structure
    is separate from exact primitive coverage and from unscanned instructions.
    """
    _limits(max_window=max_window, max_variables=max_variables,
            max_terms=max_terms, max_product_terms=max_product_terms)
    if max_ops is not None:
        _limits(max_ops=max_ops)
    if isinstance(circuit, PathLike):
        import kickmix as km
        circuit = km.Circuit.from_file(circuit)
    regions, unknown, metadata, pending = [], [], [], []
    pending_wires, census = set(), Counter()
    scanned = total = depth = exact_ops = structural_ops = exact_paid = structural_paid = 0
    total_ops = paid = 0
    budgets = dict(max_variables=max_variables, max_terms=max_terms,
                   max_operations=max_window, max_product_terms=max_product_terms)

    def flush_chunk(chunk, boundary_basis='bounded_native_operation_window'):
        nonlocal exact_ops, structural_ops, exact_paid, structural_paid
        if not chunk:
            return
        result = analyze_window([line for _, line, _ in chunk], **budgets)
        if result['status'] == 'bounded' and len(chunk) > 1:
            midpoint = len(chunk) // 2
            flush_chunk(chunk[:midpoint], 'semantic_budget_split')
            flush_chunk(chunk[midpoint:], 'semantic_budget_split')
            return
        start, stop = chunk[0][0], chunk[-1][0] + 1
        if result['status'] != 'exact':
            unknown.append(dict(start=start, stop=stop, reason=result.get('reason'),
                                status=result['status'], operation_count=len(chunk)))
            return
        # Recover producer/use/restoration boundaries inside the bounded scan
        # window.  Overlapping episodes belong to one compound region; gaps
        # retain their own exact maps.  This does not assign arithmetic names.
        episode_intervals = sorted((e['compute_at'], e['restored_at'] + 1)
                                   for f in result['features']
                                   if f['kind'] == 'compute_consume_uncompute'
                                   for e in f['restored_wires'])
        merged = []
        for lo, hi in episode_intervals:
            if merged and lo < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        if merged and merged != [(0, len(chunk))]:
            cursor = 0
            for lo, hi in merged:
                flush_chunk(chunk[cursor:lo], 'between_restoration_episodes')
                flush_chunk(chunk[lo:hi], 'restoration_candidate_boundary')
                cursor = hi
            flush_chunk(chunk[cursor:], 'between_restoration_episodes')
            return
        if merged:
            boundary_basis = 'restored_intermediate_episode'
        recognized = any(f['kind'] not in {'polynomial_accumulator'} for f in result['features'])
        structural = len(chunk) > 1 and recognized
        # Boolean polynomial maps remain complete semantics even when no named
        # affine/quadratic/phase/episode structure has been recognized.
        count_paid = sum(item['op'] in PAID for _, _, item in chunk)
        exact_ops += len(chunk)
        exact_paid += count_paid
        structural_ops += len(chunk) if structural else 0
        structural_paid += count_paid if structural else 0
        regions.append(dict(id=f'semantic-{start}-{stop}', start=start, stop=stop,
                            status='exact', kind=result['kind'], structural=structural,
                            boundary_basis=boundary_basis,
                            boundary_width=result['boundary_width'], semantics=result,
                            inputs=result['inputs'], outputs=result['outputs'],
                            phase=result['phase'], evidence=result['evidence'],
                            modified_wires=result['modified_wires'],
                            preserved_wires=result['preserved_wires'],
                            paid_operations=count_paid))

    def flush():
        flush_chunk(pending)
        pending.clear()
        pending_wires.clear()

    for index, line in enumerate(StringIO(str(circuit))):
        total = index + 1
        if max_ops is not None and index >= max_ops:
            continue
        scanned += 1
        item = _parse(line)
        op = item['op']
        if op:
            census[op] += 1
        if item['status'] != 'metadata':
            total_ops += 1
            paid += int(op in PAID)
        if op == 'PUSH_CONDITION':
            depth += 1
        in_condition = depth > 0 or op == 'POP_CONDITION'
        if item['status'] == 'metadata' and not in_condition:
            flush()
            metadata.append(dict(at=index, operation=op))
        elif item['status'] != 'supported' or in_condition:
            flush()
            unknown.append(dict(start=index, stop=index + 1, status='opaque',
                                reason='classical_condition_scope' if in_condition else item['reason'],
                                instruction=line.rstrip(), operation_count=1))
        else:
            qs = set(item['wires'])
            if pending and (len(pending) >= max_window or len(pending_wires | qs) > max_variables):
                flush()
            pending.append((index, line, item))
            pending_wires.update(qs)
        if op == 'POP_CONDITION' and depth:
            depth -= 1
    flush()
    if total > scanned:
        unknown.append(dict(start=scanned, stop=total, status='unscanned', reason='scan_budget'))
    return dict(version='1', regions=regions, unknown=unknown, metadata=metadata,
                census=dict(census),
                coverage=dict(scope='full' if scanned == total else 'prefix',
                              denominator_scope='scanned_operations',
                              total_instructions=total, scanned_instructions=scanned,
                              unscanned_instructions=total-scanned,
                              operations=total_ops, exact_operations=exact_ops,
                              structural_operations=structural_ops,
                              compound_algebra_operations=structural_ops,
                              unknown_operations=total_ops-exact_ops,
                              exact_fraction=exact_ops / total_ops if total_ops else 0.0,
                              structural_fraction=structural_ops / total_ops if total_ops else 0.0,
                              compound_algebra_fraction=structural_ops / total_ops if total_ops else 0.0,
                              paid_operations=paid, exact_paid_operations=exact_paid,
                              structural_paid_operations=structural_paid,
                              compound_algebra_paid_operations=structural_paid,
                              exact_paid_fraction=exact_paid / paid if paid else 0.0,
                              structural_paid_fraction=structural_paid / paid if paid else 0.0,
                              compound_algebra_paid_fraction=structural_paid / paid if paid else 0.0,
                              high_level_stage_coverage=None,
                              meaning='structural/compound_algebra count exact multi-operation algebra; source-language arithmetic stages remain unidentified'),
                assumptions='all window boundary wires arbitrary; no inferred clean ancillas or source-language intent',
                evidence='exact sparse Boolean ANF and sign phase on each reported exact window',
                limits=dict(max_ops=max_ops, max_window=max_window, max_variables=max_variables,
                            max_terms=max_terms, max_product_terms=max_product_terms))
