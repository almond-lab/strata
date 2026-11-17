"""Lazy joint input/output representations and shared polynomial DAG search.

The wide stage manipulates only Boolean polynomials, linear maps and DAGs.
It does not emit or simulate candidate circuits. Canonical complete DAG/layout
formulations are counted separately from factor orders and representation moves.
A bounded diverse set is then lowered, fully priced and independently checked.
"""
from collections import deque
from math import isfinite
import shlex
from time import monotonic

import kickmix as km

from .algebraic_synthesis import _Bound, _Budget, _Factors, _lower, _orders, _replay
from .boolean_synthesis import _layout
from .cost import Formula
from .model import StrataError, digest
from .output_basis_search import OutputBasis, _split
from .representation_synthesis import AffineEncoding, _substitute


def _canonical_graph(inputs, targets, gates, outputs):
    """Canonicalize AND operand order, XOR order and topological node labels."""
    offset = len(inputs) + 1
    atoms = ['1'] + [f'q{q}' for q in inputs]
    records, old_keys = {}, []
    def expression(mask):
        if type(mask) is not int or mask < 0 or mask >> len(atoms):
            raise StrataError('architecture DAG references an unavailable value')
        return tuple(sorted(atoms[i] for i in range(len(atoms)) if mask >> i & 1))
    for gate in gates:
        factors = tuple(sorted((expression(gate['left_coefficients']), expression(gate['right_coefficients']))))
        key = 'p' + digest(factors)
        records[key] = factors
        old_keys.append(key)
        atoms.append(key)
    roots = [expression(mask) for mask in outputs]
    pending = [atom for root in roots for atom in root if atom.startswith('p')]
    needed, consumed = set(), set()
    while pending:
        key = pending.pop()
        if key in needed:
            continue
        needed.add(key)
        dependencies = {atom for factor in records[key] for atom in factor if atom.startswith('p')}
        consumed.update(dependencies)
        pending.extend(dependencies)
    # Canonical topological order eliminates duplicate hypotheses caused only
    # by independent product enumeration or helper placement renaming.
    ordered, remaining = [], set(needed)
    while remaining:
        ready = sorted(key for key in remaining if not any(
            atom in remaining for factor in records[key] for atom in factor))
        if not ready:
            raise StrataError('architecture DAG contains a cycle')
        ordered.extend(ready)
        remaining.difference_update(ready)
    positions = {atom: i for i, atom in enumerate(['1'] + [f'q{q}' for q in inputs] + ordered)}
    def coefficients(expression):
        mask = 0
        for atom in expression:
            mask ^= 1 << positions[atom]
        return mask
    canonical = [dict(left_coefficients=coefficients(records[key][0]),
                      right_coefficients=coefficients(records[key][1])) for key in ordered]
    output_masks = [coefficients(root) for root in roots]
    terminal_uses = sum(atom in needed - consumed for root in roots for atom in root)
    graph_hash = digest(dict(inputs=inputs, targets=targets,
                             nodes=[(key, records[key]) for key in sorted(needed)], outputs=roots))
    return dict(gates=canonical, output_coefficients=output_masks, graph_hash=graph_hash,
                product_keys=ordered, materialized_keys=[key for key in ordered if key in consumed],
                fixed_kernel_toffoli=2 * len(consumed) + terminal_uses)


def _category(input_basis, output_basis):
    a = input_basis.receipt()['matrix_rows'] != [1 << i for i in range(len(input_basis.wires))]
    b = output_basis.receipt()['matrix_rows'] != [1 << i for i in range(len(output_basis.wires))]
    return 'joint' if a and b else 'input_only' if a else 'output_only' if b else 'identity'


def _representations(inputs, dirty, targets, polynomials, budget, *, mode, max_maps, max_depth, frontier_width):
    """Fair lazy edge traversal reaches mixed maps before exhausting one axis."""
    input_identity = AffineEncoding.from_operations(inputs, ())
    output_identity = OutputBasis.from_operations(dirty, ())
    input_moves = [('input', a, b) for a in inputs for b in inputs if a != b] if mode in {'joint', 'input'} else []
    output_moves = [('output', a, b) for a in dirty for b in dirty if a != b] if mode in {'joint', 'output'} else []
    positions = {q: i for i, q in enumerate(targets)}
    def moves(category):
        # Prefer the other representation axis immediately after a pure move.
        first, second = (output_moves, input_moves) if category == 'input_only' else (input_moves, output_moves)
        for index in range(max(len(first), len(second))):
            if index < len(first):
                yield first[index]
            if index < len(second):
                yield second[index]
    def make(a, b, polys, depth):
        return dict(input_basis=a, output_basis=b, polynomials=tuple(polys), depth=depth,
                    category=_category(a, b), representation_hash=digest((a.fingerprint, b.fingerprint)))
    initial = make(input_identity, output_identity, polynomials, 0)
    yield initial
    queue = deque([(initial, moves('identity'))])
    seen = {initial['representation_hash']}
    while queue and len(seen) < max_maps:
        state, edges = queue.popleft()
        try:
            axis, control, target = next(edges)
        except StopIteration:
            continue
        queue.append((state, edges))
        operation = f'CX q{control} q{target}'
        budget.charge(1 + sum(len(term) + 1 for poly in state['polynomials'] for term in poly))
        a, b = state['input_basis'], state['output_basis']
        if axis == 'input':
            a = AffineEncoding.from_operations(inputs, a.forward + (operation,))
            transformed = _substitute(state['polynomials'], operation)
        else:
            b = OutputBasis.from_operations(dirty, b.forward + (operation,))
            transformed = list(state['polynomials'])
            transformed[positions[target]] ^= transformed[positions[control]]
        transformed = tuple(budget.polynomial(poly) for poly in transformed)
        child = make(a, b, transformed, state['depth'] + 1)
        if child['representation_hash'] in seen:
            continue
        seen.add(child['representation_hash'])
        yield child
        if child['depth'] < max_depth:
            queue.append((child, moves(child['category'])))
        while len(queue) > frontier_width:
            queue.popleft()


def _finalists(formulations, limit):
    """Keep resource leaders, representation diversity and deterministic wildcards."""
    if not formulations:
        return []
    ranked = sorted(formulations, key=lambda row: (row['ranking_proxy'], row['formulation_hash']))
    selected, seen = [], set()
    def add(row):
        if row['formulation_hash'] not in seen and len(selected) < limit:
            selected.append(row)
            seen.add(row['formulation_hash'])
    identity = next((row for row in ranked if row['category'] == 'identity'), None)
    if identity is not None:
        add(identity)
    add(ranked[0])
    for category in ('identity', 'joint', 'input_only', 'output_only'):
        candidate = next((row for row in ranked if row['category'] == category), None)
        if candidate is not None:
            add(candidate)
    # Distinct product graphs survive even when an adapter-heavy hypothesis has
    # a weaker simple bound; host objectives are evaluated only after lowering.
    graphs = {row['graph_hash'] for row in selected}
    for row in ranked:
        if len(selected) >= max(1, 3 * limit // 4):
            break
        if row['graph_hash'] not in graphs:
            add(row)
            graphs.add(row['graph_hash'])
    wildcard = sorted(formulations, key=lambda row: row['formulation_hash'])
    for row in wildcard[::max(1, len(wildcard) // max(1, limit // 4))]:
        if len(selected) >= limit:
            break
        add(row)
    for row in ranked:
        add(row)
    return selected


def architecture_portfolio(circuit, *, clean_inputs=(), cost='Q*toffoli', resource_measure=None,
                            max_hypotheses=1024, max_representations=1024, max_factor_orders=8,
                            max_representation_depth=4, frontier_width=64, max_finalists=12,
                            max_work_steps=10_000_000, max_factor_nodes=4096, timeout=2.5,
                            representation_mode='joint', max_operations=10000, max_variables=128,
                            max_terms=65536, max_product_terms=1048576):
    """Explore joint representations without native candidate evaluation in the wide stage.

    Counts refer to canonical executable-layout DAG formulations, not random
    labels or repeated variable orders. Bounds rank fixed lowering hypotheses;
    arbitrary parent objectives are evaluated on every completed finalist.
    """
    from .accounting import measured_resources
    from .inference import functions_at
    from .semantic import equivalent_windows

    controls = dict(max_hypotheses=max_hypotheses, max_representations=max_representations,
        max_factor_orders=max_factor_orders, max_representation_depth=max_representation_depth,
        frontier_width=frontier_width, max_finalists=max_finalists, max_work_steps=max_work_steps,
        max_factor_nodes=max_factor_nodes, max_operations=max_operations, max_variables=max_variables,
        max_terms=max_terms, max_product_terms=max_product_terms)
    if any(type(value) is not int or value <= 0 for value in controls.values()):
        raise StrataError('architecture portfolio budgets must be positive integers')
    if type(timeout) not in (int, float) or not isfinite(timeout) or timeout <= 0:
        raise StrataError('architecture portfolio timeout must be finite and positive')
    if representation_mode not in {'identity', 'input', 'output', 'joint'}:
        raise StrataError('unknown architecture representation mode')
    started = monotonic()
    deadline = started + timeout
    clean = set(_layout(clean_inputs, circuit.num_qubits, 'clean inputs'))
    metadata, source_body = _split(str(circuit).splitlines())
    limits = dict(max_operations=max_operations, max_variables=max_variables,
                  max_terms=max_terms, max_product_terms=max_product_terms)
    if len(source_body) > max_operations:
        return dict(status='unknown', reason='architecture_source_operation_budget')
    for line in source_body:
        words = shlex.split(line, comments=True)
        if 'if' in words or words[0] not in {'X', 'CX', 'CCX', 'Z', 'CZ', 'CCZ', 'SWAP', 'INFO_TAG'}:
            return dict(status='not_applicable', reason='architecture portfolio requires a quantum permutation boundary')
    state = functions_at(circuit, clean_inputs=clean, **dict(limits, max_operations=max_operations + len(metadata)))
    if state['status'] != 'exact':
        return dict(status='unknown', reason=state.get('reason'), budget_phase='architecture_boundary_semantics')
    if state['phase']:
        return dict(status='not_applicable', reason='architecture portfolio requires exact zero boundary phase')
    deltas = {q: frozenset(poly ^ (set() if q in clean else {frozenset({q})}))
              for q, poly in state['functions'].items()
              if poly != (set() if q in clean else {frozenset({q})})}
    targets = sorted(deltas)
    support = {q for poly in deltas.values() for term in poly for q in term}
    if any(type(q) is not int for q in support) or support & set(targets):
        return dict(status='not_applicable', reason='architecture XOR deltas depend on modified targets or nonquantum values')
    inputs, dirty, helpers = sorted(support), sorted(set(targets) - clean), sorted(clean - set(targets))
    objective = Formula(cost) if isinstance(cost, str) else cost
    measure = resource_measure or measured_resources
    before = measure(circuit)
    before_cost = objective(before)
    best = dict(candidate=str(circuit), resources=before, cost=before_cost,
                proof=dict(status='equivalent', method='source_identity'),
                structural_witness=dict(joint_architecture_change=False), selected_mechanism='source_incumbent')
    # Reserve a separate lowering/checking interval; broad graph enumeration
    # cannot consume the entire constructor wall budget before any verification.
    generation_deadline = started + .65 * timeout
    budget = _Budget(max(1, 4 * max_work_steps // 5), max_factor_nodes, max_terms, max_product_terms,
                     max(.001, generation_deadline - monotonic()))
    formulations, seen, failures = [], set(), []
    attempts, representations, not_lowerable = 0, 0, 0
    stopped = None
    public = set(range(circuit.num_qubits)) - clean
    metadata_wires = {int(words[1][1:]) for line in metadata if (words := shlex.split(line))
                      and words[0] == 'APPEND_TO_REGISTER' and words[1].startswith('q') and words[1][1:].isdigit()}
    try:
        for representation in _representations(inputs, dirty, targets, [deltas[q] for q in targets], budget,
                mode=representation_mode, max_maps=max_representations, max_depth=max_representation_depth,
                frontier_width=frontier_width):
            representations += 1
            expected = representation['polynomials']
            for index, order in enumerate(_orders(inputs, expected, max_factor_orders)):
                attempts += 1
                graph = _Factors(inputs, order, budget)
                masks = {}
                for target_index in list(range(len(targets)))[::(-1 if index & 1 else 1)]:
                    masks[target_index] = graph.factor(expected[target_index])
                canonical = _canonical_graph(inputs, targets, graph.gates, [masks[i] for i in range(len(targets))])
                materialized = canonical['materialized_keys']
                if len(materialized) > len(helpers):
                    not_lowerable += 1
                    continue
                layout = dict(zip(materialized, helpers))
                formulation_hash = digest(dict(representation=representation['representation_hash'],
                                               graph=canonical['graph_hash'], helper_layout=layout))
                if formulation_hash in seen:
                    continue
                try:
                    replay = _replay(inputs, canonical['gates'], canonical['output_coefficients'], expected, budget)
                except StrataError as exc:
                    if len(failures) < 16:
                        failures.append(dict(reason='architecture_polynomial_replay_failed', detail=str(exc)))
                    continue
                seen.add(formulation_hash)
                width = 1 + max(public | set(targets) | set(layout.values()) | metadata_wires, default=-1)
                adapter_count = 2 * (len(representation['input_basis'].forward) + len(representation['output_basis'].forward))
                paid = canonical['fixed_kernel_toffoli']
                formulations.append(dict(formulation_hash=formulation_hash,
                    representation_hash=representation['representation_hash'], category=representation['category'],
                    input_encoding=representation['input_basis'].receipt(),
                    output_encoding=representation['output_basis'].receipt(),
                    graph_hash=canonical['graph_hash'],
                    dag=dict(input_wires=inputs, target_wires=targets, gates=canonical['gates'],
                             output_coefficients=canonical['output_coefficients']),
                    layout=dict(helper_wires=list(layout.values()), product_to_helper=layout,
                                arbitrary_dirty_targets=dirty, clean_inputs=sorted(clean)),
                    polynomial_replay=replay,
                    resource_bounds=dict(Q=width, fixed_lowering_toffoli=paid,
                        minimum_instructions=paid + adapter_count, adapter_instructions=adapter_count,
                        scope='fixed product kernels and complete layout; not a bound on other algorithms or arbitrary host cost'),
                    ranking_proxy=[width * paid, paid, adapter_count, width],
                    variable_order=list(order)))
                if len(formulations) >= max_hypotheses:
                    stopped = 'architecture_hypothesis_budget'
                    break
            if stopped:
                break
    except (_Bound, RecursionError) as exc:
        stopped = str(exc) or 'architecture_recursion_budget'
    except StrataError as exc:
        stopped = 'architecture_symbolic_generation_failed'
        failures.append(dict(reason=stopped, detail=str(exc)))
    generation_seconds = monotonic() - started
    finalists = _finalists(formulations, max_finalists)
    verified, native_seen, lowered, lowering_steps = [], set(), 0, 0
    for formulation in finalists:
        if monotonic() >= deadline:
            stopped = 'architecture_timeout'
            break
        remaining_steps = max_work_steps - budget.steps - lowering_steps
        if remaining_steps <= 0:
            stopped = 'architecture_work_step_budget'
            break
        dag = formulation['dag']
        local = None
        try:
            local = _Budget(remaining_steps, max_factor_nodes, max_terms, max_product_terms,
                            max(.001, deadline - monotonic()))
            body, lowering = _lower(circuit, clean, inputs, targets, dag['gates'], dag['output_coefficients'],
                                     local, max_operations)
            lowered += 1
            _, native = _split(str(body).splitlines())
            a, b = formulation['input_encoding'], formulation['output_encoding']
            complete = km.Circuit('\n'.join(metadata + a['forward'] + b['forward'] + native + b['inverse'] + a['inverse']))
            if public and complete.num_qubits <= max(public):
                complete = km.Circuit(str(complete) + f'\nAPPEND_TO_REGISTER q{max(public)} r{circuit.num_registers}')
            _, candidate_body = _split(str(complete).splitlines())
            if len(candidate_body) > max_operations or complete.num_qubits > circuit.num_qubits:
                failures.append(dict(formulation_hash=formulation['formulation_hash'], reason='architecture_native_layout_or_operation_budget'))
                continue
            candidate_hash = digest(str(complete))
            if candidate_hash in native_seen:
                continue
            check = equivalent_windows(source_body, candidate_body, clean_inputs=clean, **limits)
            if check.get('equivalent') is not True:
                failures.append(dict(formulation_hash=formulation['formulation_hash'], reason='architecture_complete_boundary_check', check=check))
                continue
            native_bill = measured_resources(complete)
            if (native_bill['toffoli'] != formulation['resource_bounds']['fixed_lowering_toffoli']
                    or native_bill['Q'] != formulation['resource_bounds']['Q']):
                failures.append(dict(formulation_hash=formulation['formulation_hash'], reason='architecture_fixed_kernel_ledger_mismatch'))
                continue
            resources = measure(complete)
            value = objective(resources)
            native_seen.add(candidate_hash)
            witness = dict(joint_architecture_change=False,
                input_basis_changed=formulation['category'] in {'input_only', 'joint'},
                output_basis_changed=formulation['category'] in {'output_only', 'joint'},
                complete_joint_output_synthesis=True, both_adapter_pairs_charged=True,
                formulation_hash=formulation['formulation_hash'], graph_hash=formulation['graph_hash'])
            alternative = dict(candidate=str(complete), candidate_hash=candidate_hash,
                formulation_hash=formulation['formulation_hash'], cost=value, resources=resources,
                native_resources=native_bill, structural_witness=witness, lowering=lowering,
                selected_mechanism='joint_architecture' if formulation['category'] != 'identity' else 'algebraic_identity',
                category=formulation['category'], input_encoding=a, output_encoding=b,
                dag=dag, whole_boundary_check=check,
                proof=dict(status='equivalent', method='independent_polynomial_replay_and_complete_native_boundary_equality',
                    polynomial_replay=formulation['polynomial_replay'], all_public_quantum_inputs=True,
                    dirty_targets='arbitrary', exact_phase=True, clean_inputs=sorted(clean),
                    source_named_registers_preserved=True, truth_rows_enumerated=0))
            verified.append(alternative)
            if value < best['cost']:
                best = alternative
        except (_Bound, StrataError, RecursionError) as exc:
            failures.append(dict(formulation_hash=formulation['formulation_hash'], reason='architecture_finalist_failed', detail=str(exc)))
        finally:
            if local is not None:
                lowering_steps += local.steps
    if monotonic() >= deadline:
        stopped = 'architecture_timeout'
    identity_values = [row['cost'] for row in verified if row['category'] == 'identity']
    comparator_cost = min([before_cost] + identity_values)
    axis_values = {category: min((row['cost'] for row in verified if row['category'] == category), default=None)
                   for category in ('input_only', 'output_only', 'joint')}
    for alternative in verified:
        witness = alternative['structural_witness']
        witness.update(joint_architecture_change=bool(identity_values) and alternative['category'] != 'identity'
            and bool(alternative['dag']['gates']) and len(targets) >= 2 and alternative['cost'] < comparator_cost,
            verified_identity_comparator_cost=comparator_cost if identity_values else None,
            identity_comparison_available=bool(identity_values),
            measured_joint_interaction_gain=alternative['category'] == 'joint' and bool(identity_values)
                and all(axis_values[key] is not None and alternative['cost'] < axis_values[key]
                        for key in ('input_only', 'output_only')) and alternative['cost'] < comparator_cost,
            interaction_scope='only the measured axis finalists; not exhaustive separate-axis optima')
    def dominates(a, b):
        keys = ('Q', 'toffoli', 'instructions')
        return (all(a['native_resources'][key] <= b['native_resources'][key] for key in keys)
                and any(a['native_resources'][key] < b['native_resources'][key] for key in keys))
    pareto = [row['candidate_hash'] for row in verified if not any(dominates(other, row) for other in verified)]
    categories = {category: sum(row['category'] == category for row in formulations)
                  for category in ('identity', 'input_only', 'output_only', 'joint')}
    return dict(status='constructed', family='wide_joint_representation_factorization', **best,
        improved=best['cost'] < before_cost, verified_elites=verified, alternatives=verified,
        pareto_native_hashes=pareto, pareto_scope='verified native boundary Q, toffoli and instruction counts',
        formulations=formulations, failures=failures,
        identity_comparator=dict(status='verified' if identity_values else 'unknown',
            cost=comparator_cost, source_cost=before_cost,
            best_verified_identity_cost=min(identity_values) if identity_values else None,
            verified_identity_count=len(identity_values), same_factor_order_limit=max_factor_orders),
        measured_axis_costs=axis_values,
        resource_bill=dict(before=before, after=best['resources'], before_cost=before_cost, after_cost=best['cost'],
            cost_expression=cost if isinstance(cost, str) else cost.expression),
        search=dict(proposed_attempts=attempts, unique_formulations=len(formulations),
            unique_representations=representations, unique_product_graphs=len({row['graph_hash'] for row in formulations}),
            independently_replayed=len(formulations), not_lowerable=not_lowerable,
            selected_finalists=len(finalists), lowered=lowered, verified=len(verified),
            native_improvements=sum(row['cost'] < before_cost for row in verified), categories=categories,
            generation_seconds=generation_seconds, seconds=monotonic() - started, factor_steps=budget.steps,
            lowering_steps=lowering_steps, accounted_work_steps=budget.steps + lowering_steps,
            stopped_reason=stopped, representation_mode=representation_mode,
            limits=controls | {'timeout': timeout}, native_evaluations_in_wide_stage=0,
            optimality='not claimed; bounded joint linear representations, polynomial factorizations and diverse finalists'),
        frozen_boundary=dict(clean_inputs=sorted(clean), public_inputs=sorted(public),
                             input_wires=inputs, dirty_xor_targets=dirty, extra_helper_wires=0))
