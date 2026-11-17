"""Bounded exact semantics for native kickmix permutation/sign circuits.

An input variable denotes the arbitrary bit on a physical wire at the *window*
boundary.  In particular, register names and apparent scratch usage never imply
zero.  ANF quantum variables use physical wire numbers; classical inputs and fresh
measurement outcomes use separate string namespaces. [] is the constant one; an
empty list of monomials is zero.  Addition is XOR and multiplication is AND in
GF(2)[x]/(x*x+x).  Unsupported instructions are barriers, not identities.
"""
from __future__ import annotations

from collections import Counter
from io import StringIO
from fractions import Fraction
from .factored import BooleanDAG
from os import PathLike
import re
import shlex


ARITY = {'X': 1, 'CX': 2, 'CCX': 3, 'SWAP': 2,
         'CSWAP': 3, 'Z': 1, 'CZ': 2, 'CCZ': 3, 'NEG': 0, 'R': 1}
METADATA = {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG'}
PAID = {'CCX', 'CCZ', 'CSWAP'}
ZERO = frozenset()
ONE = frozenset({frozenset()})


class _BoundExceeded(Exception):
    pass


def _variable(q):
    return frozenset({frozenset({q})})


def _variable_key(value):
    return (0, value) if isinstance(value, int) else (1, value)


def _poly(value):
    terms = [sorted(term, key=_variable_key) for term in value]
    terms.sort(key=lambda t: (len(t), [_variable_key(q) for q in t]))
    return dict(monomials=terms,
                expression=' ^ '.join('*'.join(f'x{q}' if isinstance(q, int) else q
                                               for q in t) or '1' for t in terms) or '0',
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
    item = dict(op=op, wires=qs, bits=(), condition=None)
    if op in METADATA:
        return dict(item, status='metadata')
    if 'if' in words:
        if len(words) < 3 or words[-2] != 'if' or not re.fullmatch(r'b[0-9]+', words[-1]) or words.count('if') != 1:
            return dict(item, status='opaque', reason='unsupported_condition')
        item['condition'] = words[-1]
        words = words[:-2]
    args = words[1:]
    valid = False
    if op in ARITY:
        valid = len(args) == ARITY[op] == len(qs) and len(set(qs)) == len(qs)
    elif op == 'HMR':
        valid = len(args) == 2 and len(qs) == 1 and bool(re.fullmatch(r'b[0-9]+', args[1]))
        if valid:
            item['bits'] = (args[1],)
    elif op in {'BIT_INVERT', 'BIT_STORE0', 'BIT_STORE1'}:
        valid = len(args) == 1 and bool(re.fullmatch(r'b[0-9]+', args[0]))
        if valid:
            item['bits'] = (args[0],)
    elif op == 'PUSH_CONDITION':
        valid = not args and item['condition'] is not None
    elif op == 'POP_CONDITION':
        valid = not args and item['condition'] is None
    elif op == 'Z_POW':
        if len(args) != 2 or len(qs) != 1 or not re.fullmatch(r'[-+]?[0-9]+\.[0-9]+', args[1]):
            return dict(item, status='opaque', reason='unsupported_operands')
        try:
            exponent = Fraction(args[1])
        except (ValueError, ZeroDivisionError):
            return dict(item, status='opaque', reason='unsupported_operands')
        if exponent.denominator != 1:
            return dict(item, status='opaque', reason='unsupported_fractional_phase')
        item['sign_exponent'] = exponent.numerator % 2
        valid = True
    else:
        return dict(item, status='opaque', reason='unsupported_operation')
    return dict(item, status='supported' if valid else 'opaque',
                **({} if valid else dict(reason='unsupported_operands')))


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
                   max_operations=128, max_product_terms=65536, clean_inputs=()):
    """Exact outcome-parametric native trajectory semantics on a bounded window.

    Physical inputs and classical boundary bits are arbitrary unless physical
    wires are explicitly listed in ``clean_inputs``. HMR introduces a fresh
    independent outcome, clears its target and adds measured_value * outcome
    to the exact sign exponent. This is not an arbitrary-state quantum-channel
    equivalence theorem. A dirty reset and fractional phases fail closed.
    """
    _limits(max_variables=max_variables, max_terms=max_terms,
            max_operations=max_operations, max_product_terms=max_product_terms)
    clean_inputs = frozenset(clean_inputs)
    if any(type(q) is not int or q < 0 for q in clean_inputs):
        raise ValueError('clean_inputs must contain nonnegative physical wire IDs')
    parsed, wires, bits, metadata, count = [], set(), set(), [], 0
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
        parsed.append((index, item))
        wires.update(item['wires'])
        bits.update(item['bits'])
        if item['condition']:
            bits.add(item['condition'])
        if len(wires) + len(bits) > max_variables:
            return dict(status='bounded', reason='variable_budget', limit=max_variables)
    dag = BooleanDAG()
    # Each value has an independently canonical ANF and a factored DAG root.
    zero, one = (ZERO, dag.zero), (ONE, dag.one)
    def variable(name):
        return (_variable(name), dag.variable(name))
    def xor(a, b):
        result = a[0] ^ b[0]
        return result, dag.zero if not result else dag.xor(a[1], b[1])
    initial = {q: zero if q in clean_inputs else variable(q) for q in sorted(wires)}
    state = dict(initial)
    classical = {bit: variable(bit) for bit in sorted(bits)}
    phase, products, stack = zero, 0, []
    writes = {q: [] for q in wires}
    consumed = {q: [] for q in wires}
    measurements, resets, phase_events = [], [], []

    def check():
        if len(phase[0]) + sum(len(p[0]) for p in state.values()) + sum(len(p[0]) for p in classical.values()) > max_terms:
            raise _BoundExceeded('term_budget')

    def mul(a, b):
        nonlocal products
        if not a[0] or not b[0]:
            return zero
        if a[0] == ONE:
            return b
        if b[0] == ONE:
            return a
        products += len(a[0]) * len(b[0])
        if products > max_product_terms:
            raise _BoundExceeded('product_budget')
        result = set()
        for x in a[0]:
            for y in b[0]:
                term = x | y
                if term in result:
                    result.remove(term)
                else:
                    result.add(term)
                if len(result) > max_terms:
                    raise _BoundExceeded('term_budget')
        result = frozenset(result)
        return result, dag.zero if not result else dag.product(a[1], b[1])

    try:
        check()
        for index, item in parsed:
            op, qs = item['op'], item['wires']
            condition = classical[item['condition']] if item['condition'] else one
            if op == 'PUSH_CONDITION':
                stack.append(condition)  # Snapshot, including inside inactive scopes.
                continue
            if op == 'POP_CONDITION':
                if stack:
                    stack.pop()
                continue
            active = condition
            for captured in stack:
                active = mul(active, captured)
            reads = qs[:-1] if op in {'CX', 'CCX'} else qs if op in {'Z', 'CZ', 'CCZ', 'HMR', 'CSWAP'} else ()
            for q in reads:
                if state[q][0] != initial[q][0]:
                    consumed[q].append(index)
            targets = ()
            if op in {'BIT_INVERT', 'BIT_STORE0', 'BIT_STORE1'}:
                bit = item['bits'][0]
                old = classical[bit]
                change = active if op == 'BIT_INVERT' else mul(active, xor(old, one if op == 'BIT_STORE1' else zero))
                classical[bit] = xor(old, change)
            elif op == 'HMR':
                q, bit = qs[0], item['bits'][0]
                name = f'm{len(measurements)}'
                if len(wires) + len(bits) + len(measurements) + 1 > max_variables:
                    raise _BoundExceeded('variable_budget')
                outcome, old = variable(name), state[q]
                contribution = mul(mul(old, outcome), active)
                phase = xor(phase, contribution)
                measurements.append(dict(variable=name, at=index, wire=q, bit=bit,
                                         active=_poly(active[0]), measured_value=_poly(old[0]),
                                         created_phase=_poly(contribution[0])))
                classical[bit] = xor(classical[bit], mul(active, xor(classical[bit], outcome)))
                state[q] = xor(old, mul(active, old))
                targets = qs
            elif op == 'R':
                q = qs[0]
                if mul(active, state[q])[0]:
                    return dict(status='opaque', reason='reset_not_proven_zero', at=index,
                                wire=q, active=_poly(active[0]), value=_poly(state[q][0]),
                                clean_input_assumptions=sorted(clean_inputs))
                resets.append(dict(at=index, wire=q, active=_poly(active[0]),
                                   proof='active_times_value_is_zero', value=_poly(state[q][0])))
            elif op == 'X':
                state[qs[0]] = xor(state[qs[0]], active)
                targets = qs
            elif op in {'CX', 'CCX'}:
                contribution = active
                for q in qs[:-1]:
                    contribution = mul(contribution, state[q])
                state[qs[-1]] = xor(state[qs[-1]], contribution)
                targets = qs[-1:]
            elif op in {'SWAP', 'CSWAP'}:
                a, b = qs[-2:]
                control = mul(active, state[qs[0]]) if op == 'CSWAP' else active
                delta = mul(control, xor(state[a], state[b]))
                state[a], state[b] = xor(state[a], delta), xor(state[b], delta)
                targets = (a, b)
            else:
                contribution = active
                for q in qs:
                    contribution = mul(contribution, state[q])
                if op == 'Z_POW' and not item['sign_exponent']:
                    contribution = zero
                phase = xor(phase, contribution)
                phase_events.append(dict(at=index, operation=op, contribution=_poly(contribution[0]),
                                         values={str(q): _poly(state[q][0]) for q in qs}))
            for q in targets:
                writes[q].append(index)
            check()
    except _BoundExceeded as exc:
        return dict(status='bounded', reason=str(exc), operation_count=len(parsed),
                    limits=dict(max_terms=max_terms, max_product_terms=max_product_terms))
    if stack:
        return dict(status='opaque', reason='unclosed_condition_scope', depth=len(stack))

    factored = dag.summary({str(q): p[1] for q, p in state.items()},
                           {b: p[1] for b, p in classical.items()}, phase[1])
    state = {q: p[0] for q, p in state.items()}
    phase = phase[0]
    changed = {q: p ^ initial[q][0] for q, p in state.items() if p != initial[q][0]}
    preserved = sorted(wires - changed.keys())
    features = []
    output_degree = max((len(t) for p in state.values() for t in p), default=0)
    nonquantum = sorted(bits) + [event['variable'] for event in measurements]
    is_permutation = not measurements and not resets and not any(item['op'].startswith('BIT_') for _, item in parsed)
    pure_quantum = not nonquantum and not resets
    if not changed and not phase and pure_quantum:
        features.append(dict(kind='identity_map'))
    elif output_degree <= 1:
        features.append(dict(kind='affine_transform',
                             matrix={str(q): sorted((next(iter(t)) for t in p if t), key=_variable_key)
                                     for q, p in state.items()},
                             offset={str(q): int(frozenset() in p) for q, p in state.items()}))
    if changed and pure_quantum and all(not (term & changed.keys()) for p in changed.values() for term in p):
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
                                 initial_value=_poly(initial[q][0])['expression'],
                                 final_value=_poly(state[q])['expression']))
    if episodes:
        features.append(dict(kind='compute_consume_uncompute', restored_wires=episodes,
                             guarantee='restores the stated boundary values; no clean ancillas inferred'))
    debts = []
    for event in measurements:
        m = event['variable']
        coefficient = frozenset(term - {m} for term in phase if m in term)
        debts.append(dict(measurement=m, created_at=event['at'], wire=event['wire'],
                          created=event['created_phase'], residual_coefficient=_poly(coefficient),
                          status='discharged' if not coefficient else 'outstanding',
                          proof='final_phase_coefficient_in_fresh_outcome',
                          phase_use_indices=[e['at'] for e in phase_events
                                             if any(m in term for term in e['contribution']['monomials'])]))
    if measurements:
        features.append(dict(kind='measurement_cleanup', measurements=measurements, phase_debts=debts,
                             all_phase_debts_discharged=all(d['status'] == 'discharged' for d in debts)))
    if phase:
        features.append(dict(kind='phase_polynomial', exponent=_poly(phase), convention='(-1)^exponent'))
    kind = 'measurement_cleanup' if measurements else 'compute_consume_uncompute' if episodes else next(
        (f['kind'] for f in features if f['kind'].endswith('_accumulator')),
        features[0]['kind'] if features else 'boolean_polynomial_map')
    if not changed and phase and not measurements:
        kind = 'phase_polynomial'
    assumptions = ('all boundary wire and classical bit values arbitrary; no clean ancillas inferred'
                   if not clean_inputs else f'explicit zero boundary wires: {sorted(clean_inputs)}; all other boundary values arbitrary')
    return dict(status='exact', kind=kind, inputs=sorted(wires), quantum_inputs=sorted(wires),
                classical_inputs=sorted(bits), measurement_variables=[e['variable'] for e in measurements],
                measurement_events=measurements, nonquantum_variables=nonquantum,
                classical_outputs={b: _poly(p[0]) for b, p in classical.items()},
                is_permutation=is_permutation, clean_input_assumptions=sorted(clean_inputs),
                boundary_width=len(wires),
                boundary=dict(input_width=len(wires), output_width=len(wires),
                              modified_width=len(changed), preserved_width=len(preserved)),
                outputs={str(q): _poly(p) for q, p in state.items()}, phase=_poly(phase),
                changes={str(q): _poly(p) for q, p in changed.items()},
                modified_wires=sorted(changed), preserved_wires=preserved,
                features=features, operation_count=len(parsed), instruction_count=count,
                metadata_indices=metadata, max_output_degree=output_degree,
                instruction_offsets='relative_to_window', assumptions=assumptions,
                effects=dict(reset_proofs=resets, phase_debts=debts, phase_events=phase_events,
                             clean_outputs=sorted(q for q, p in state.items() if not p)),
                factored=factored,
                semantic_scope='native_outcome_parametric_basis_trajectories',
                phase_scope='exact_sign_only; fractional rotations are unsupported',
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


def semantic_function(result):
    """Canonical receipt payload, including every nonquantum effect and promise."""
    function = dict(outputs=result['outputs'], phase=result['phase'])
    if result.get('nonquantum_variables') or result.get('clean_input_assumptions'):
        function.update(classical_outputs=result.get('classical_outputs', {}),
                        measurement_events=result.get('measurement_events', []),
                        clean_input_assumptions=result.get('clean_input_assumptions', []),
                        semantic_scope=result['semantic_scope'])
    return function


def equivalent_windows(original, candidate, **limits):
    """Compare outputs, classical state and sign for every input and outcome.

    Measurement events must have the same ordered provenance (wire, classical
    destination, activation and measured value). Merely matching outcome counts
    does not align reordered or unrelated measurements. Such comparisons return
    unknown instead of making a branch-correspondence claim.
    """
    before, after = analyze_window(original, **limits), analyze_window(candidate, **limits)
    if before['status'] != 'exact' or after['status'] != 'exact':
        return dict(status='unknown', equivalent=None, before=before, after=after,
                    evidence='exact_semantics_unavailable')
    def event_signature(result):
        return [dict(wire=e['wire'], bit=e['bit'], active=e['active']['monomials'],
                     measured_value=e['measured_value']['monomials'])
                for e in result['measurement_events']]
    if event_signature(before) != event_signature(after):
        return dict(status='unknown', equivalent=None, reason='measurement_event_alignment_unproved',
                    before=before, after=after, evidence='measurement_provenance_mismatch')
    wires = sorted(set(before['inputs']) | set(after['inputs']))
    clean = set(limits.get('clean_inputs', ()))
    bits = sorted(set(before['classical_inputs']) | set(after['classical_inputs']))
    outcomes = before['measurement_variables']
    variables = wires + bits + outcomes
    def qvalue(result, q):
        return result['outputs'].get(str(q), {}).get('monomials', [] if q in clean else [[q]])
    def bvalue(result, bit):
        return result['classical_outputs'].get(bit, {}).get('monomials', [[bit]])
    differences = [q for q in wires if qvalue(before, q) != qvalue(after, q)]
    classical_differences = [bit for bit in bits if bvalue(before, bit) != bvalue(after, bit)]
    phase_equal = before['phase']['monomials'] == after['phase']['monomials']
    same = not differences and not classical_differences and phase_equal
    result = dict(status='equivalent' if same else 'different', equivalent=same,
                  differing_wires=differences, differing_classical_bits=classical_differences,
                  phase_equal=phase_equal, inputs=wires, classical_inputs=bits,
                  measurement_variables=outcomes, clean_input_assumptions=sorted(clean),
                  evidence='canonical_boolean_anf_equality_with_exact_sign_phase',
                  semantic_scope='native_outcome_parametric_basis_trajectories',
                  assumptions=before['assumptions'])
    if not same:
        if differences:
            q = differences[0]
            a, b, kind = qvalue(before, q), qvalue(after, q), 'output'
        elif classical_differences:
            bit = classical_differences[0]
            a, b, kind = bvalue(before, bit), bvalue(after, bit), 'classical_output'
        else:
            a, b, kind = before['phase']['monomials'], after['phase']['monomials'], 'phase'
        assignment = _distinguishing_assignment(a, b, variables)
        example = dict(
            inputs={str(q): assignment[str(q)] for q in wires}, kind=kind,
            before_outputs={str(q): _evaluate_terms(qvalue(before, q), assignment) for q in wires},
            after_outputs={str(q): _evaluate_terms(qvalue(after, q), assignment) for q in wires},
            before_phase=_evaluate_terms(before['phase']['monomials'], assignment),
            after_phase=_evaluate_terms(after['phase']['monomials'], assignment))
        if bits or outcomes:
            example.update(classical_inputs={b: assignment[b] for b in bits},
                           measurement_outcomes={m: assignment[m] for m in outcomes},
                           before_classical_outputs={b: _evaluate_terms(bvalue(before, b), assignment) for b in bits},
                           after_classical_outputs={b: _evaluate_terms(bvalue(after, b), assignment) for b in bits})
        result['counterexample'] = example
    return result


def lift_regions(circuit, *, max_ops=None, max_window=128, max_variables=32,
                 max_terms=4096, max_product_terms=65536, clean_inputs=()):
    """Discover exact compound regions across native text or a loaded circuit.

    A PathLike is loaded through kickmix's text/binary autodetection.  All other
    inputs are native text or Circuit objects.  The entire artifact is scanned
    by default.  Regions are disjoint bounded windows; unknown spans and scoped
    condition bodies stay explicit.  Coverage of recognized compound structure
    is separate from exact primitive coverage and from unscanned instructions.
    Explicit clean inputs apply only to a complete, single-window artifact.
    A budget cut never reapplies those promises at an interior boundary.
    """
    _limits(max_window=max_window, max_variables=max_variables,
            max_terms=max_terms, max_product_terms=max_product_terms)
    if max_ops is not None:
        _limits(max_ops=max_ops)
    clean_inputs = set(clean_inputs)
    if any(type(q) is not int or q < 0 for q in clean_inputs):
        raise ValueError('clean_inputs must contain nonnegative physical wire IDs')
    clean_inputs = sorted(clean_inputs)
    if isinstance(circuit, PathLike):
        import kickmix as km
        circuit = km.Circuit.from_file(circuit)
    regions, unknown, metadata, pending = [], [], [], []
    pending_wires, census = set(), Counter()
    pending_context = False
    poisoned_depth = 0
    overflow = None
    scanned = total = depth = exact_ops = structural_ops = exact_paid = structural_paid = 0
    total_ops = paid = 0
    budgets = dict(max_variables=max_variables, max_terms=max_terms,
                   max_operations=max_window, max_product_terms=max_product_terms,
                   clean_inputs=clean_inputs)

    def flush_chunk(chunk, boundary_basis='bounded_native_operation_window'):
        nonlocal exact_ops, structural_ops, exact_paid, structural_paid
        if not chunk:
            return
        result = analyze_window([line for _, line, _ in chunk], **budgets)
        contextual = bool(clean_inputs) or any(item.get('condition') or item['op'] in {'HMR', 'PUSH_CONDITION', 'POP_CONDITION'}
                         or item['op'].startswith('BIT_') for _, _, item in chunk)
        if result['status'] == 'bounded' and len(chunk) > 1 and not contextual:
            midpoint = len(chunk) // 2
            flush_chunk(chunk[:midpoint], 'semantic_budget_split')
            flush_chunk(chunk[midpoint:], 'semantic_budget_split')
            return
        start, stop = chunk[0][0], chunk[-1][0] + 1
        if result['status'] != 'exact':
            unknown.append(dict(start=start, stop=stop, reason=result.get('reason'),
                                status=result['status'], operation_count=sum(
                                    item['status'] != 'metadata' for _, _, item in chunk)))
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
        if merged and merged != [(0, len(chunk))] and not contextual:
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
        operation_count = sum(item['status'] != 'metadata' for _, _, item in chunk)
        exact_ops += operation_count
        exact_paid += count_paid
        structural_ops += operation_count if structural else 0
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
        nonlocal pending_context
        flush_chunk(pending)
        pending.clear()
        pending_wires.clear()
        pending_context = False

    from .streaming import native_lines
    with native_lines(circuit) as stream:
        for index, line in enumerate(stream):
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
            if overflow is not None:
                # Do not retain an unbounded contextual run just to label it
                # unknown. A later unsupported boundary may start a fresh local
                # contract; global input promises never restart after a cut.
                if not clean_inputs and item['status'] == 'opaque' and not depth:
                    overflow = None
                else:
                    overflow['stop'] = index + 1
                    overflow['operation_count'] += item['status'] != 'metadata'
                    if item['status'] == 'metadata':
                        metadata.append(dict(at=index, operation=op))
                    if op == 'POP_CONDITION' and depth:
                        depth -= 1
                    continue
            if clean_inputs:
                if item['status'] == 'metadata':
                    metadata.append(dict(at=index, operation=op))
                pending.append((index, line, item))
                if len(pending) > max_window:
                    overflow = dict(start=0, stop=index + 1, status='bounded',
                                    reason='clean_inputs_require_complete_single_window',
                                    limit=max_window,
                                    operation_count=sum(row['status'] != 'metadata' for _, _, row in pending))
                    unknown.append(overflow)
                    pending.clear()
                continue
            if item['status'] == 'metadata':
                metadata.append(dict(at=index, operation=op))
                if pending_context:
                    pending.append((index, line, item))
                else:
                    flush()
            elif item['status'] != 'supported' or poisoned_depth:
                flush()
                unknown.append(dict(start=index, stop=index + 1, status='opaque',
                                    reason='classical_condition_context_unavailable' if poisoned_depth else item['reason'],
                                    instruction=line.rstrip(), operation_count=1))
                poisoned_depth = depth
            else:
                qs = set(item['wires'])
                context = bool(item.get('condition') or op in {'HMR', 'PUSH_CONDITION', 'POP_CONDITION'}
                               or op.startswith('BIT_'))
                if pending and not pending_context and not context and (len(pending) >= max_window or len(pending_wires | qs) > max_variables):
                    flush()
                pending_context |= context
                pending.append((index, line, item))
                pending_wires.update(qs)
            if op == 'POP_CONDITION' and depth:
                depth -= 1
                if poisoned_depth:
                    poisoned_depth = depth
            if pending_context and len(pending) > max_window:
                flush()
                overflow = unknown[-1]
    if clean_inputs and total > scanned and pending:
        unknown.append(dict(start=0, stop=scanned, status='bounded',
                            reason='clean_inputs_require_complete_single_window',
                            operation_count=total_ops))
    else:
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
                assumptions=(f'explicit zero artifact-boundary wires: {clean_inputs}; complete single-window scope only'
                             if clean_inputs else 'all window boundary wires arbitrary; no inferred clean ancillas or source-language intent'),
                evidence='exact sparse Boolean ANF and sign phase on each reported exact window',
                limits=dict(max_ops=max_ops, max_window=max_window, max_variables=max_variables,
                            max_terms=max_terms, max_product_terms=max_product_terms,
                            clean_inputs=clean_inputs))
