"""Bounded whole-output ANF factoring and shared reversible product graphs.

This constructor never enumerates a truth table. It factors F = L*A XOR B,
where L groups variables having the same residual A, and shares canonical
polynomial subexpressions across all outputs. Alternative variable orders are
bounded constructive searches, not Boolean or native optimality proofs.
"""
from collections import Counter
from math import isfinite
import shlex
from time import monotonic

import kickmix as km

from .boolean_synthesis import _affine_update, _layout, _product_update, _source_structure
from .cost import Formula
from .model import StrataError, digest


ZERO = frozenset()
ONE = frozenset({frozenset()})


class _Bound(Exception):
    pass


class _Budget:
    def __init__(self, steps, nodes, terms, product_terms, timeout):
        self.max_steps, self.max_nodes = steps, nodes
        self.max_terms, self.max_product_terms = terms, product_terms
        self.deadline, self.steps = monotonic() + timeout, 0

    def charge(self, steps=1):
        self.steps += steps
        if self.steps > self.max_steps:
            raise _Bound('algebraic_factor_step_budget')
        if monotonic() > self.deadline:
            raise _Bound('algebraic_factor_time_budget')

    def polynomial(self, value):
        value = frozenset(value)
        if len(value) > self.max_terms:
            raise _Bound('algebraic_polynomial_term_budget')
        return value

    def product(self, left, right):
        if len(left) * len(right) > self.max_product_terms:
            raise _Bound('algebraic_polynomial_product_budget')
        self.charge(len(left) * len(right))
        terms = set()
        for a in left:
            for b in right:
                term = a | b
                if term in terms:
                    terms.remove(term)
                else:
                    terms.add(term)
        return self.polynomial(terms)


class _Factors:
    def __init__(self, inputs, order, budget):
        self.inputs, self.order, self.budget = inputs, order, budget
        self.positions = {q: i for i, q in enumerate(inputs)}
        self.values = {ZERO: 0, ONE: 1}
        self.values.update((frozenset({frozenset({q})}), 1 << (i + 1))
                           for i, q in enumerate(inputs))
        self.polynomials = [ONE] + [frozenset({frozenset({q})}) for q in inputs]
        self.gates = []

    def _remember(self, polynomial, expression):
        if polynomial not in self.values and len(self.values) >= self.budget.max_nodes:
            raise _Bound('algebraic_polynomial_node_budget')
        self.values[polynomial] = expression
        return expression

    def expression(self, mask):
        result = ZERO
        while mask:
            bit = (mask & -mask).bit_length() - 1
            result ^= self.polynomials[bit]
            mask &= mask - 1
            self.budget.charge()
        return self.budget.polynomial(result)

    def product(self, left, right):
        if not left or not right:
            return 0
        if left == 1 or left == right:
            return right
        if right == 1:
            return left
        polynomial = self.budget.product(self.expression(left), self.expression(right))
        if polynomial in self.values:
            return self.values[polynomial]
        if all(len(term) <= 1 for term in polynomial):
            return self.factor(polynomial)
        self.gates.append(dict(left_coefficients=left, right_coefficients=right))
        mask = 1 << len(self.polynomials)
        self.polynomials.append(polynomial)
        return self._remember(polynomial, mask)

    def factor(self, polynomial):
        self.budget.charge()
        polynomial = self.budget.polynomial(polynomial)
        if polynomial in self.values:
            return self.values[polynomial]
        if all(len(term) <= 1 for term in polynomial):
            mask = 0
            for term in polynomial:
                mask ^= 1 << (self.positions[next(iter(term))] + 1) if term else 1
            return self._remember(polynomial, mask)
        support = set().union(*polynomial)
        pivot = next(q for q in self.order if q in support)
        residual = frozenset(term - {pivot} for term in polynomial if pivot in term)
        residual_support = set().union(*residual)
        # Any v outside support(A) whose complete v*A block is present can
        # share this factor. Those blocks are pairwise disjoint, so removing
        # them preserves GF(2) coefficients without an expansion heuristic.
        linear, removed = 0, set()
        for q in self.order:
            if q not in support or q in residual_support:
                continue
            self.budget.charge(len(residual))
            block = {term | {q} for term in residual}
            if block <= polynomial:
                linear ^= 1 << (self.positions[q] + 1)
                removed.update(block)
        self.budget.charge(len(residual))
        if residual <= polynomial:
            linear ^= 1
            removed.update(residual)
        tail = polynomial - removed
        a, b = self.factor(residual), self.factor(tail)
        result = self.product(linear, a) ^ b
        return self._remember(polynomial, result)


def _orders(inputs, polynomials, limit):
    frequency = Counter(q for polynomial in polynomials for term in polynomial for q in term)
    degree = Counter()
    for polynomial in polynomials:
        for term in polynomial:
            for q in term:
                degree[q] += len(term)
    frequent = sorted(inputs, key=lambda q: (-frequency[q], q))
    candidates = [frequent, list(reversed(frequent)), list(inputs), list(reversed(inputs)),
                  sorted(inputs, key=lambda q: (-degree[q], -frequency[q], q))]
    candidates += [frequent[i:] + frequent[:i] for i in range(1, min(len(inputs), limit))]
    seen = set()
    for candidate in candidates:
        order = tuple(candidate)
        if order not in seen:
            seen.add(order)
            yield order
            if len(seen) >= limit:
                break


def _replay(inputs, gates, output_masks, expected, budget):
    """Reconstruct polynomials from coefficients, ignoring search caches."""
    values = [ONE] + [frozenset({frozenset({q})}) for q in inputs]
    def affine(mask):
        if type(mask) is not int or mask < 0 or mask >> len(values):
            raise StrataError('algebraic DAG references an unavailable value')
        out = set()
        for i, value in enumerate(values):
            if mask >> i & 1:
                budget.charge(len(value))
                out.symmetric_difference_update(value)
        return budget.polynomial(out)
    for gate in gates:
        left, right = affine(gate['left_coefficients']), affine(gate['right_coefficients'])
        if len(left) * len(right) > budget.max_product_terms:
            raise _Bound('algebraic_replay_product_budget')
        budget.charge(len(left) * len(right))
        counts = Counter(a | b for a in left for b in right)
        values.append(budget.polynomial(term for term, coefficient in counts.items() if coefficient & 1))
    actual = [affine(mask) for mask in output_masks]
    if actual != list(expected):
        raise StrataError('algebraic DAG polynomial replay mismatch')
    return dict(status='verified', method='independent_sparse_polynomial_replay',
                input_count=len(inputs), output_count=len(expected), truth_rows_enumerated=0)


def _lower(circuit, clean, inputs, targets, gates, outputs, budget, max_operations):
    offset = len(inputs) + 1
    needed = {i for mask in outputs for i in range(len(gates)) if mask >> (offset + i) & 1}
    consumed, pending = set(), list(needed)
    while pending:
        i = pending.pop()
        for mask in (gates[i]['left_coefficients'], gates[i]['right_coefficients']):
            for j in range(i):
                if mask >> (offset + j) & 1:
                    consumed.add(j)
                    if j not in needed:
                        needed.add(j)
                        pending.append(j)
        budget.charge()
    helpers = sorted(set(clean) - set(targets))
    if len(consumed) > len(helpers):
        raise _Bound('algebraic_clean_product_workspace_budget')
    placement = dict(zip(sorted(consumed), helpers))
    available = list(inputs) + [placement.get(i) for i in range(len(gates))]
    forward, blocks, consumer = [], [], []
    for i in sorted(consumed):
        gate = gates[i]
        block = _product_update(gate['left_coefficients'], gate['right_coefficients'],
                                available[:len(inputs) + i], placement[i])
        forward.extend(block)
        blocks.append(block)
        budget.charge(len(block))
    for mask, target in zip(outputs, targets):
        materialized = mask
        for i in sorted(needed - consumed):
            if mask >> (offset + i) & 1:
                materialized ^= 1 << (offset + i)
                gate = gates[i]
                consumer.extend(_product_update(gate['left_coefficients'], gate['right_coefficients'],
                                                 available[:len(inputs) + i], target))
        consumer.extend(_affine_update(materialized, available, target))
        budget.charge()
    native = forward + consumer + [line for block in reversed(blocks) for line in reversed(block)]
    if len(native) > max_operations:
        raise _Bound('algebraic_native_operation_budget')
    retained = (set(range(circuit.num_qubits)) - set(clean)) | set(targets) | set(placement.values())
    metadata, names, highest = [], set(), -1
    for line in str(circuit).splitlines():
        words = shlex.split(line, comments=True)
        if not words or words[0] not in {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG'}:
            continue
        if words[0] == 'APPEND_TO_REGISTER' and words[1].startswith('q') and int(words[1][1:]) not in retained:
            continue
        metadata.append(line)
        if words[0] == 'REGISTER' and len(words)>2:
            names.add(words[2])
        highest = max([highest] + [int(w[1:]) for w in words if w.startswith('r') and w[1:].isdigit()])
    boundary_present = retained and any(shlex.split(line)[:2] == ['APPEND_TO_REGISTER', f'q{max(retained)}']
                                        for line in metadata)
    if retained and not boundary_present:
        name, suffix = 'strata_algebraic_boundary', 0
        while name in names:
            suffix += 1
            name = f'strata_algebraic_boundary_{suffix}'
        metadata += [f'APPEND_TO_REGISTER q{max(retained)} r{highest + 1}',
                     f'REGISTER r{highest + 1} "{name}"']
    candidate = km.Circuit('\n'.join(metadata + native))
    return candidate, dict(input_wires=list(inputs), target_wires=list(targets),
                           used_clean_wires=sorted(placement.values()),
                           emitted_and_nodes=len(needed), materialized_products=len(consumed),
                           terminal_products=len(needed - consumed),
                           final_products_materialized=False,
                           forward_instructions=len(forward), consumer_instructions=len(consumer),
                           cleanup_instructions=len(forward))


def algebraic_boundary_synthesis(circuit, *, clean_inputs=(), cost='Q*toffoli', resource_measure=None,
                                 max_orders=8, max_factor_nodes=4096, max_factor_steps=1000000,
                                 timeout=3.0, max_operations=10000, max_variables=128,
                                 max_terms=65536, max_product_terms=1048576):
    """Factor all output deltas jointly, without a truth-table input cap.

    All non-clean wires stay arbitrary. Only phase-free quantum XOR maps whose
    deltas do not depend on their targets are accepted. Classical/measurement
    interfaces remain outside this constructor. Search exhaustion is explicit;
    an earlier independently verified candidate can still be returned.
    """
    from .accounting import measured_resources
    from .inference import functions_at
    from .semantic import equivalent_windows
    limits = dict(max_operations=max_operations, max_variables=max_variables,
                  max_terms=max_terms, max_product_terms=max_product_terms)
    controls = dict(max_orders=max_orders, max_factor_nodes=max_factor_nodes,
                    max_factor_steps=max_factor_steps, timeout=timeout)
    if any(type(value) is not int or value <= 0 for value in
           (max_orders, max_factor_nodes, max_factor_steps, *limits.values())):
        raise StrataError('algebraic synthesis budgets must be positive integers')
    if type(timeout) not in (int, float) or not isfinite(timeout) or timeout <= 0:
        raise StrataError('algebraic synthesis timeout must be positive and finite')
    clean = set(_layout(clean_inputs, circuit.num_qubits, 'clean inputs'))
    rows = str(circuit).splitlines()
    if len(rows) > max_operations:
        return dict(status='unknown', reason='algebraic_source_operation_budget', limits=controls | limits)
    for line in rows:
        words = shlex.split(line, comments=True)
        if words and words[0] not in {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG'} and (
                'if' in words or words[0] in {'HMR', 'PUSH_CONDITION', 'POP_CONDITION'} or words[0].startswith('BIT_')):
            return dict(status='not_applicable', reason='algebraic synthesis requires a quantum boundary without classical or measurement channels')
    started = monotonic()
    budget = _Budget(max_factor_steps, max_factor_nodes, max_terms, max_product_terms, timeout)
    state = functions_at(circuit, clean_inputs=clean, **limits)
    if state['status'] != 'exact':
        return dict(status='unknown', reason=state.get('reason'), budget_phase='boundary_semantics')
    if state['phase']:
        return dict(status='not_applicable', reason='algebraic synthesis requires exact zero boundary phase')
    deltas = {q: frozenset(poly ^ (set() if q in clean else {frozenset({q})}))
              for q, poly in state['functions'].items()
              if poly != (set() if q in clean else {frozenset({q})})}
    targets = sorted(deltas)
    support = set().union(*(set().union(*poly) if poly else set() for poly in deltas.values()))
    if any(type(q) is not int for q in support) or support & set(targets):
        return dict(status='not_applicable', reason='algebraic XOR deltas depend on modified targets or non-quantum inputs')
    inputs, expected = sorted(support), [deltas[q] for q in targets]
    if len(inputs) + 2 > max_factor_nodes:
        return dict(status='unknown', reason='algebraic_polynomial_node_budget')
    measure, objective = resource_measure or measured_resources, Formula(cost)
    before = measure(circuit)
    trials, best, stopped = [], None, None
    original_body = [line for line in rows if line.split()[0] not in {'REGISTER', 'APPEND_TO_REGISTER'}]
    try:
        for index, order in enumerate(_orders(inputs, expected, max_orders)):
            budget.charge()
            graph = _Factors(inputs, order, budget)
            masks = {}
            for target in targets[::(-1 if index & 1 else 1)]:
                masks[target] = graph.factor(deltas[target])
            outputs = [masks[q] for q in targets]
            replay = _replay(inputs, graph.gates, outputs, expected, budget)
            try:
                candidate, lowering = _lower(circuit, clean, inputs, targets, graph.gates, outputs,
                                              budget, max_operations)
            except _Bound as exc:
                # Another algebraic order can fit the same physical workspace
                # or emission limit. These are local construction obstructions;
                # global time/step/node budgets still stop the entire search.
                if str(exc) not in {'algebraic_clean_product_workspace_budget',
                                    'algebraic_native_operation_budget'}:
                    raise
                trials.append(dict(order=list(order), status='not_lowerable', reason=str(exc),
                                   product_nodes=len(graph.gates)))
                continue
            candidate_body = [line for line in str(candidate).splitlines()
                              if line.split()[0] not in {'REGISTER', 'APPEND_TO_REGISTER'}]
            check = equivalent_windows(original_body, candidate_body, clean_inputs=clean, **limits)
            if check.get('equivalent') is not True:
                trials.append(dict(order=list(order), status='rejected' if check.get('equivalent') is False else 'unknown', check=check))
                continue
            budget.charge()
            resources = measure(candidate)
            value = objective(resources)
            witness = dict(input_wires=inputs, target_wires=targets, gates=graph.gates,
                           output_coefficients=outputs, variable_order=list(order),
                           polynomial_nodes=len(graph.values), truth_rows_enumerated=0)
            trial = dict(order=list(order), status='verified', candidate_hash=digest(str(candidate)),
                         cost=value, resources=resources, product_nodes=len(graph.gates),
                         materialized_products=lowering['materialized_products'])
            trials.append(trial)
            if best is None or value < best['cost']:
                best = dict(candidate=str(candidate), cost=value, resources=resources,
                            dag=witness, lowering=lowering, polynomial_replay=replay, check=check)
    except (_Bound, RecursionError) as exc:
        stopped = str(exc) or 'algebraic_recursion_budget'
    if best is None:
        return dict(status='unknown', reason=stopped or 'no algebraic construction passed the complete boundary',
                    attempts=trials, limits=controls | limits, factor_steps=budget.steps,
                    seconds=monotonic() - started, optimality='not claimed')
    structure = _source_structure(rows, clean)
    restored = {q for q in clean if not state['functions'][q]}
    retired = sorted(set(structure['materialized_clean_wires']) & restored - set(best['lowering']['used_clean_wires']))
    return dict(status='constructed', family='symbolic_anf_factor_and_share', **best,
                resource_bill=dict(before=before, after=best['resources'], before_cost=objective(before),
                                   after_cost=best['cost'], cost_expression=cost),
                proof=dict(status='equivalent', method='independent_polynomial_DAG_replay_and_complete_boundary_equality',
                           all_public_inputs=True, dirty_targets='arbitrary; XOR-only updates',
                           clean_inputs=sorted(clean), exact_phase=True, truth_rows_enumerated=0),
                structural_witness=dict(structure, retired_scratch_wires=retired,
                    retained_product_wires=best['lowering']['used_clean_wires'],
                    boundary_outputs_synthesized_jointly=targets, final_products_materialized=False,
                    removed_materialization_bank=len(retired) >= 2 and structure['source_nonlinear_dataflow_depth'] >= 2),
                search=dict(attempts=trials, stopped_reason=stopped, factor_steps=budget.steps,
                            limits=controls | limits, seconds=monotonic() - started,
                            optimality='not claimed; bounded algebraic constructions'))
