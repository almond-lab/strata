"""Discover a Boolean intermediate from the consumer's distinguishable inputs.

For a bound set A, substitute all assignments of A in every output ANF. Two
assignments are equivalent exactly when the residual output polynomials agree
for every assignment of the remaining inputs. If there are two classes, one
bit h(A) suffices. Search and lower F0 + h*(F0+F1) across every consumer.
"""
from itertools import combinations, islice

from .accounting import measured_resources
from math import isfinite

from .algebraic_synthesis import _Budget, _Bound, _Factors, _lower, _replay
from .boolean_synthesis import _layout
from .cost import Formula
from .inference import functions_at
from .model import StrataError


def _screen_table(inputs, outputs, budget):
    """Bit-parallel values of all consumers for up to twelve support bits."""
    if len(inputs)>12 or len(outputs)>32:
        return None
    count=1 << len(inputs)
    ones=(1 << count)-1
    positions={q:i for i,q in enumerate(inputs)}
    columns={q:ones // ((1 << (1 << i))+1) << (1 << i) for q,i in positions.items()}
    packed=[]
    for polynomial in outputs:
        word=0
        for term in polynomial:
            budget.charge(len(term)+1)
            monomial=ones
            for q in term:
                monomial &= columns[q]
            word ^= monomial
        packed.append(word)
    return positions,packed,count-1


def _third_signature(table, group, budget):
    """Three different observed cofactor rows disprove a one-bit interface."""
    positions,outputs,all_bits=table
    bits=[1 << positions[q] for q in group]
    outside=all_bits ^ sum(bits)
    probes=list(dict.fromkeys(p & outside for p in (0,all_bits,0xaaa,0x555,0x333,0xccc)))
    rows=[0]
    for bit in bits:
        rows += [r|bit for r in rows]
    signatures=set()
    for assignment in rows:
        budget.charge(len(outputs)*len(probes))
        signature=tuple((word >> (assignment|probe)) & 1 for probe in probes for word in outputs)
        signatures.add(signature)
        if len(signatures)>2:
            return True
    return False


def _classes(outputs, group, budget):
    bound = frozenset(group)
    cofactors = []
    truth = []
    for assignment in range(1 << len(group)):
        absent = {q for i, q in enumerate(group) if not (assignment >> i) & 1}
        row = []
        for polynomial in outputs:
            budget.charge(len(polynomial) + 1)
            residual = set()
            for term in polynomial:
                if term.isdisjoint(absent):
                    residual.symmetric_difference_update((term - bound,))
            row.append(frozenset(residual))
        row = tuple(row)
        if row not in cofactors:
            if len(cofactors) == 2:
                return None
            cofactors.append(row)
        truth.append(cofactors.index(row))
    if len(cofactors) != 2:
        return None
    # Mobius inversion over the bound set gives the exact ANF of its class bit.
    coeff = truth.copy()
    for bit in range(len(group)):
        for mask in range(len(coeff)):
            if mask & (1 << bit):
                coeff[mask] ^= coeff[mask ^ (1 << bit)]
    selector = frozenset(frozenset(q for j, q in enumerate(group) if mask & (1 << j))
                         for mask, value in enumerate(coeff) if value)
    if not any(len(term) >= 2 for term in selector) or len(selector) < 2:
        return None
    if set().union(*selector) != bound:
        return None
    difference = [a ^ b for a, b in zip(*cofactors)]
    if not any(term for poly in difference for term in poly):
        return None
    return selector, cofactors[0], difference, truth


def _selector_mask(graph, selector, group):
    """Try one product of two parities before recursively factoring the selector."""
    quadratic = {term for term in selector if len(term) == 2}
    if quadratic and all(len(term) <= 2 for term in selector):
        # Boolean squares are linear. Match the off-diagonal quadratic part,
        # then restore the independently computed linear/constant remainder.
        pairs = [(i,j) for i,j in combinations(range(len(group)),2)]
        for left,right in combinations(range(1, 1 << len(group)), 2):
            graph.budget.charge(1 + len(pairs))
            terms = {frozenset((group[i],group[j])) for i,j in pairs
                     if ((left >> i & right >> j) ^ (left >> j & right >> i)) & 1}
            if terms != quadratic:
                continue
            diagonal = {frozenset((q,)) for i,q in enumerate(group) if (left & right) >> i & 1}
            def native_mask(mask):
                return sum(1 << (graph.positions[q]+1) for i,q in enumerate(group) if mask >> i & 1)
            product = graph.product(native_mask(left), native_mask(right))
            return product ^ graph.factor(selector ^ terms ^ diagonal)
    return graph.factor(selector)


def _groups(inputs, outputs, cap):
    # Rank variables by monomial frequency, then interleave
    # group sizes. Bound generation itself even when input support is wide.
    frequency = {q: sum(q in t for p in outputs for t in p) for q in inputs}
    order = sorted(inputs, key=lambda q: (-frequency[q], q))
    pool = order[:10]
    candidates = [iter(combinations(pool, size)) for size in (3, 4, 2) if size < len(inputs)]
    def generate():
        while candidates:
            for iterator in candidates.copy():
                group = next(iterator, None)
                if group is None:
                    candidates.remove(iterator)
                else:
                    yield group
    yield from islice(generate(), cap)


def decomposition_synthesis(circuit, *, clean_inputs=(), cost='Q*toffoli', resource_measure=None,
                            max_partitions=96, max_steps=250000, max_operations=10000,
                            max_variables=128, max_terms=65536, max_product_terms=65536,
                            screen_partitions=True, timeout=2.0, max_factor_nodes=4096):
    """Bounded exact two-class detection and heuristic, verified reversible lowering."""
    from .restructure import _check
    if any(type(n) is not int or n < 1 for n in (max_partitions, max_steps, max_factor_nodes)):
        raise StrataError('decomposition search budgets must be positive integers')
    if type(timeout) not in (int, float) or not isfinite(timeout) or timeout <= 0:
        raise StrataError('decomposition timeout must be positive and finite')
    clean = set(_layout(clean_inputs, circuit.num_qubits, 'clean inputs'))
    limits = dict(max_operations=max_operations, max_variables=max_variables,
                  max_terms=max_terms, max_product_terms=max_product_terms)
    if circuit.num_bits:
        return dict(status='not_applicable', reason='decomposition currently requires a quantum-only boundary')
    state = functions_at(circuit, clean_inputs=clean, **limits)
    if state['status'] != 'exact' or state['phase']:
        return dict(status='not_applicable', reason='requires exact zero-phase Boolean output functions')
    deltas = {}
    for q, polynomial in state['functions'].items():
        initial = set() if q in clean else {frozenset({q})}
        if polynomial != initial:
            deltas[q] = frozenset(polynomial ^ initial)
    targets = sorted(deltas)
    outputs = [deltas[q] for q in targets]
    support = {q for p in outputs for t in p for q in t}
    if len(support) < 3 or support.intersection(targets) or any(type(q) is not int for q in support):
        return dict(status='not_applicable', reason='requires preserved inputs and a separable XOR-output boundary')
    inputs = sorted(support)
    measure, objective = resource_measure or measured_resources, Formula(cost)
    budget = _Budget(max_steps, max_factor_nodes, max_terms, max_product_terms, timeout)
    best = None
    tried = matches = verified = screened = 0
    stopped = None
    try:
        screen=_screen_table(inputs,outputs,budget) if screen_partitions else None
        for group in _groups(inputs, outputs, max_partitions):
            tried += 1
            if screen is not None and _third_signature(screen,group,budget):
                screened += 1
                continue
            witness = _classes(outputs, group, budget)
            if witness is None:
                continue
            matches += 1
            selector, base, difference, classes = witness
            graph = _Factors(inputs, inputs, budget)
            selected = _selector_mask(graph, selector, group)
            coefficients = [graph.factor(p) ^ graph.product(selected, graph.factor(delta))
                            for p, delta in zip(base, difference)]
            replay = _replay(inputs, graph.gates, coefficients, outputs, budget)
            try:
                candidate, lowering = _lower(circuit, clean, inputs, targets, graph.gates,
                                              coefficients, budget, max_operations)
            except _Bound as exc:
                if str(exc) in {'algebraic_clean_product_workspace_budget','algebraic_native_operation_budget'}:
                    continue
                raise
            check = _check(circuit, candidate, clean, limits)
            if check.get('equivalent') is not True:
                continue
            verified += 1
            resources = measure(candidate)
            price = objective(resources)
            if best is None or price < best['cost']:
                best = dict(candidate=str(candidate), cost=price, resources=resources, check=check,
                            lowering=lowering, polynomial_replay=replay,
                            structural_witness=dict(discovered_consumer_class_bit=True, bound_inputs=list(group),
                                classes_by_assignment=classes, residual_class_count=2,
                                selector_anf=sorted(sorted(term) for term in selector),
                                consumer_targets=targets, compared_all_remaining_inputs_symbolically=True))
    except (_Bound, RecursionError) as exc:
        stopped = str(exc) or 'recursion budget'
    search = dict(partitions_tested=tried, rejected_by_three_distinct_cofactors=screened,
                  full_partition_checks=tried-screened, two_class_nonlinear_maps=matches, verified_lowerings=verified,
                  step_count=budget.steps, stop_reason=stopped, max_partitions=max_partitions)
    if best is None:
        return dict(status='unknown' if stopped or matches else 'not_applicable',
                    reason=stopped or 'no verified nonlinear two-class lowering among tested partitions', search=search)
    return dict(status='constructed', family='consumer_equivalence_decomposition', search=search, **best)
