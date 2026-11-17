"""Compose a restored boundary's complete sign and synthesize its observable.

The sign is an XOR of products, so a terminal product can be observed without
storing it. Distributing one further AND into a terminal product exposes native
CCZ terms on three wires. Only surviving internal values need clean workspace.
"""
from math import isfinite
from time import monotonic

import kickmix as km

from .accounting import measured_resources
from .algebraic_synthesis import _Bound, _Budget, _Factors, _orders, _replay
from .boolean_synthesis import _layout, _product_update, _source_structure
from .cost import Formula
from .model import StrataError, digest
from .semantic import analyze_window


def _indices(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _phase_gate(forms, available):
    """Phase the intersection of at most three affine equations L_i = 1."""
    forms = list(forms)
    prepare, pivots = [], []
    for i, form in enumerate(forms):
        # Earlier pivots equal one on the active subspace. Substitute them so
        # the next basis change can never overwrite an earlier control.
        for p in pivots:
            bit = 2 << p
            if form & bit:
                form ^= bit | 1
        if form < 2:
            if not form:
                return []
            continue
        p = next(_indices(form >> 1))
        target = available[p]
        delta = form ^ (2 << p)
        prepare.extend(f'CX q{available[j]} q{target}' for j in _indices(delta >> 1))
        if delta & 1:
            prepare.append(f'X q{target}')
        for j in range(i + 1, len(forms)):
            if forms[j] & (2 << p):
                forms[j] ^= delta
        pivots.append(p)
    if len(pivots) > 3:
        raise StrataError('observable term exceeds native sign arity')
    op = ('NEG', 'Z', 'CZ', 'CCZ')[len(pivots)]
    phase = ' '.join([op, *(f'q{available[p]}' for p in pivots)])
    return prepare + [phase] + prepare[::-1]


def _operands(gate):
    return gate['left_coefficients'], gate['right_coefficients']


def _terminal_terms(output, gates, offset, unfold, budget):
    base_mask = (1 << offset) - 1
    terms = []
    for index in _indices(output >> offset):
        pair = _operands(gates[index])
        expandable = [side for side in (0, 1) if pair[side] >> offset]
        if not unfold or not expandable:
            terms.append(pair)
            continue
        # Distribute A*(B + sum_i C_i D_i) into A*B and A*C_i*D_i.
        # Choose a deterministic side when both operands contain products.
        side = min(expandable, key=lambda j: (pair[j] >> offset).bit_count())
        other, split = pair[1 - side], pair[side]
        if split & base_mask:
            terms.append((other, split & base_mask))
        for child in _indices(split >> offset):
            budget.charge()
            terms.append((other, *_operands(gates[child])))
    return output & base_mask, terms


def _lower(inputs, gates, output, clean, unfold, budget, max_operations):
    offset = len(inputs) + 1
    affine, terms = _terminal_terms(output, gates, offset, unfold, budget)
    needed = {i for term in terms for form in term for i in _indices(form >> offset)}
    pending = list(needed)
    while pending:
        index = pending.pop()
        for form in _operands(gates[index]):
            for child in _indices(form >> offset):
                if child not in needed:
                    needed.add(child)
                    pending.append(child)
        budget.charge()
    if len(needed) > len(clean):
        raise _Bound('observable_clean_product_workspace_budget')
    placement = dict(zip(sorted(needed), sorted(clean)))
    available = inputs + [placement.get(i) for i in range(len(gates))]
    forward = []
    for i, wire in placement.items():
        forward.extend(_product_update(*_operands(gates[i]), available[:len(inputs) + i], wire))
        if 2 * len(forward) > max_operations:
            raise _Bound('observable_native_operation_budget')
    consumer = ['NEG'] if affine & 1 else []
    consumer.extend(f'Z q{inputs[j]}' for j in _indices(affine >> 1))
    for term in terms:
        block = _phase_gate(term, available)
        consumer.extend(block)
        budget.charge(1 + len(block))
        if 2 * len(forward) + len(consumer) > max_operations:
            raise _Bound('observable_native_operation_budget')
    native = forward + consumer + forward[::-1]
    signs = [line.split()[0] for line in consumer if line.split()[0] in {'NEG', 'Z', 'CZ', 'CCZ'}]
    return native, dict(used_clean_wires=sorted(placement.values()),
                        product_placement={str(i): q for i, q in placement.items()},
                        terminal_affine_coefficients=affine, phase_product_coefficients=terms,
                        unfolded_terminal_products=unfold, forward_instructions=len(forward),
                        observable_instructions=len(consumer), cleanup_instructions=len(forward),
                        native_phase_gates=len(signs), native_cubic_phase_gates=signs.count('CCZ'))


def _direct_comparator(phase, emit, verify, measure, objective, max_operations):
    """Price the established degree-three native monomial construction on this layout."""
    if any(len(term) > 3 for term in phase):
        return dict(status='not_applicable', reason='direct native phase degree exceeds three')
    if len(phase) > max_operations:
        return dict(status='unknown', reason='observable_direct_operation_budget')
    terms = sorted((sorted(term) for term in phase), key=lambda term: (len(term), term))
    native = [' '.join([('NEG', 'Z', 'CZ', 'CCZ')[len(term)], *(f'q{q}' for q in term)]) for term in terms]
    candidate = emit(native)
    if len(candidate) > max_operations:
        return dict(status='unknown', reason='observable_direct_operation_budget')
    check = verify(candidate)
    if check.get('equivalent') is not True:
        return dict(status='unknown', reason='direct phase equivalence unavailable', check=check)
    resources = measure(candidate)
    return dict(status='verified', method='direct_native_phase_monomials', candidate_hash=digest(str(candidate)),
                phase_monomials=len(phase), cost=objective(resources), resources=resources, check=check)


def observable_boundary_synthesis(circuit, *, clean_inputs=(), cost='Q*toffoli', resource_measure=None,
                                 max_orders=4, max_factor_steps=1000000, max_factor_nodes=4096,
                                 timeout=3.0, max_operations=10000, max_variables=128,
                                 max_terms=65536, max_product_terms=1048576):
    """Search exact sign circuits under a declared restored quantum boundary."""
    from .restructure import _boundary_metadata, _check
    limits = dict(max_operations=max_operations, max_variables=max_variables, max_terms=max_terms,
                  max_product_terms=max_product_terms)
    if any(type(n) is not int or n <= 0 for n in (*limits.values(), max_orders, max_factor_nodes, max_factor_steps)):
        raise StrataError('observable synthesis budgets must be positive integers')
    if type(timeout) not in (int, float) or not isfinite(timeout) or timeout <= 0:
        raise StrataError('observable synthesis timeout must be positive and finite')
    clean = set(_layout(clean_inputs, circuit.num_qubits, 'clean inputs'))
    lines = str(circuit).splitlines()
    state = analyze_window(lines, clean_inputs=clean, **limits)
    if state['status'] != 'exact':
        return dict(status='unknown', reason=state.get('reason'), budget_phase='boundary_semantics')
    if (not state['is_permutation'] or state['classical_inputs'] or state['modified_wires']
            or any(line.split()[:1] == ['POP_CONDITION'] for line in lines)):
        return dict(status='not_applicable', reason='observable synthesis requires every quantum output restored and a reversible quantum-only boundary')
    phase = frozenset(frozenset(term) for term in state['phase']['monomials'])
    if not phase:
        return dict(status='not_applicable', reason='boundary has zero sign polynomial')
    inputs = sorted({q for term in phase for q in term})
    if len(inputs) + 2 > max_factor_nodes:
        return dict(status='unknown', reason='algebraic_polynomial_node_budget')
    metadata = _boundary_metadata(circuit)
    max_public = circuit.num_qubits - 1
    while max_public in clean:
        max_public -= 1
    def emit(native):
        candidate = km.Circuit('\n'.join(metadata + native))
        if candidate.num_qubits <= max_public:
            full_metadata = _boundary_metadata(circuit, max_public, 'strata_public_boundary')
            candidate = km.Circuit('\n'.join(full_metadata + native))
        return candidate
    def verify(candidate):
        return _check(circuit, candidate, clean, limits)
    measure, objective = resource_measure or measured_resources, Formula(cost) if isinstance(cost, str) else cost
    before = measure(circuit)
    budget = _Budget(max_factor_steps, max_factor_nodes, max_terms, max_product_terms, timeout)
    started = monotonic()
    attempts, seen, best, stopped = [], set(), None, None
    try:
        for order in _orders(inputs, [phase], max_orders):
            graph = _Factors(inputs, order, budget)
            output = graph.factor(phase)
            replay = _replay(inputs, graph.gates, [output], [phase], budget)
            for unfold in (True, False):
                budget.charge()
                try:
                    native, lowering = _lower(inputs, graph.gates, output, clean, unfold, budget, max_operations)
                except _Bound as exc:
                    if str(exc) not in {'observable_clean_product_workspace_budget', 'observable_native_operation_budget'}:
                        raise
                    attempts.append(dict(status='not_lowerable', reason=str(exc), order=list(order), unfold=unfold))
                    continue
                candidate = emit(native)
                key = digest(str(candidate))
                if key in seen:
                    continue
                seen.add(key)
                if len(candidate) > max_operations:
                    attempts.append(dict(status='not_lowerable', reason='observable_native_operation_budget'))
                    continue
                check = verify(candidate)
                if check.get('equivalent') is not True:
                    attempts.append(dict(status=check.get('status', 'unknown'), check=check))
                    continue
                resources = measure(candidate)
                value = objective(resources)
                attempts.append(dict(status='verified', candidate_hash=key, cost=value,
                                     order=list(order), unfold=unfold, resources=resources))
                if best is None or (value, len(candidate)) < (best['cost'], best['native_instructions']):
                    best = dict(candidate=str(candidate), resources=resources, cost=value, check=check,
                                native_instructions=len(candidate), lowering=lowering, polynomial_replay=replay,
                                dag=dict(input_wires=inputs, gates=graph.gates, phase_coefficients=output,
                                         variable_order=list(order)))
    except (_Bound, RecursionError) as exc:
        stopped = str(exc) or 'observable_recursion_budget'
    search = dict(attempts=attempts, factor_steps=budget.steps, stopped_reason=stopped,
                  limits=dict(limits, max_orders=max_orders, max_factor_steps=max_factor_steps,
                              max_factor_nodes=max_factor_nodes, factor_timeout=timeout),
                  seconds=monotonic() - started, optimality='not claimed; bounded factorizations and native lowerings')
    if best is None:
        return dict(status='unknown', reason=stopped or (attempts[-1].get('reason') if attempts else None)
                    or 'no observable construction passed the complete boundary', search=search)
    events = [e for e in state['effects']['phase_events'] if e['contribution']['monomials']]
    # Identical phase reads can cancel completely. Attribute a word only to
    # reads whose exponent still contributes to the composed sign.
    contributions = {}
    for event in events:
        operands = frozenset(int(q) for q in event['values'] if int(q) in clean)
        if operands:
            polynomial = contributions.setdefault(operands, set())
            polynomial.symmetric_difference_update(frozenset(t) for t in event['contribution']['monomials'])
    net = set()
    for polynomial in contributions.values():
        net.symmetric_difference_update(polynomial)
    word = {q for operands, polynomial in contributions.items() if polynomial & net & phase for q in operands}
    structure = _source_structure(lines, clean)
    retained = set(best['lowering']['used_clean_wires'])
    retired = (set(structure['materialized_clean_wires']) | word) - retained
    eliminated = bool(word) and word.isdisjoint(retained)
    comparator = _direct_comparator(phase, emit, verify, measure, objective, max_operations)
    contracted = (best['lowering']['native_phase_gates'] < len(phase) or
                  best['lowering']['native_cubic_phase_gates'] < sum(len(t) == 3 for t in phase))
    factored_win = contracted and comparator['status'] == 'verified' and best['cost'] < comparator['cost']
    return dict(status='constructed', family='factored_boundary_sign_observable', **best, search=search,
                direct_phase_comparator=comparator,
                structural_witness=dict(structure, source_phase_word_wires=sorted(word),
                    source_phase_reads=[dict(at=e['at'], operation=e['operation'], wires=sorted(int(q) for q in e['values']))
                                        for e in events],
                    phase_word_scope='clean operand groups whose combined contribution survives in the boundary sign',
                    retired_scratch_wires=sorted(retired), retained_product_wires=sorted(retained),
                    removed_phase_word=eliminated, removed_materialization_bank=eliminated and len(word) >= 2,
                    factored_phase_observable=factored_win),
                proof=dict(status='equivalent', method='independent_polynomial_DAG_replay_and_complete_native_boundary_equality',
                           clean_inputs=sorted(clean), source_width=circuit.num_qubits,
                           public_input_count=circuit.num_qubits - len(clean), all_public_inputs=True,
                           public_input_layout='original physical indices outside clean_inputs',
                           all_quantum_outputs_restored=True, exact_phase=True, truth_rows_enumerated=0),
                witness_hash=digest(dict(source=str(circuit), candidate=best['candidate'], clean_inputs=sorted(clean))),
                resource_bill=dict(before=before, after=best['resources'], before_cost=objective(before),
                                   after_cost=best['cost'], cost_expression=cost if isinstance(cost, str) else getattr(cost, 'expression', None)))
