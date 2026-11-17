"""Bounded exact storage/recomputation planning for a supplied reversible DAG.

Nodes denote complete kernels with explicit storage and compute/uncompute
prices. The planner changes their lifetime and recomputation schedule. It
does not synthesize a kernel, infer its reversibility, or prove a native
circuit. Its Pareto optimality and lower bounds apply only to this DAG and
the supplied kernel cost model, never to every circuit for the same function.
"""
from collections import Counter
from fractions import Fraction
import heapq

from .model import StrataError


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise StrataError(f'{name} must be an integer at least {minimum}')
    return value


def _cost(value, name):
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        raise StrataError(f'{name} must be a nonnegative exact rational') from None
    if result < 0:
        raise StrataError(f'{name} must be a nonnegative exact rational')
    return result


def plan_reversible_pebbles(nodes, outputs, *, inputs=(), input_width=0, fixed_width=0,
                            max_width=None, max_states=100000, consume_cost=None,
                            consume_scratch_width=0):
    """Return the complete cost/peak-storage Pareto frontier when bounded search finishes.

    Each node supplies ``id``, ``dependencies``, ``width``, ``compute_cost``;
    ``uncompute_cost`` defaults to the same explicit compute price.
    ``scratch_width`` and ``uncompute_scratch_width`` price transient kernel
    workspace at the peak, including the node being computed/uncomputed.

    ``inputs`` names immutable external dependencies. Charge their storage in
    ``input_width``; ``fixed_width`` charges other permanently allocated host
    wires. Output mode retains just the requested nodes and restores all other
    node storage. Supplying ``consume_cost`` instead requires a once-only
    observable consumer with all requested nodes present, followed by complete
    cleanup. The consumer must preserve those nodes for the later uncompute.

    Search labels are nondominated (cost, peak) pairs for a live-node set and
    consumer state. Dependencies are required for both compute and uncompute.
    Thus no inversion or cleanup is assumed free and recomputation is explicit.
    """
    input_width = _integer(input_width, 'input_width')
    fixed_width = _integer(fixed_width, 'fixed_width')
    max_states = _integer(max_states, 'max_states', 1)
    consume_scratch_width = _integer(consume_scratch_width, 'consume_scratch_width')
    if max_width is not None: _integer(max_width, 'max_width')
    consume = None if consume_cost is None else _cost(consume_cost, 'consume_cost')
    supplied, inputs, outputs = list(nodes), list(inputs), list(outputs)

    def valid_id(value):
        return type(value) in {str, int} and (not isinstance(value, str) or bool(value))

    if any(not valid_id(v) for v in inputs + outputs) or len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs):
        raise StrataError('input and output IDs must be distinct strings or integers')
    external, graph = set(inputs), {}
    for raw in supplied:
        if not isinstance(raw, dict) or not valid_id(raw.get('id')):
            raise StrataError('each DAG node requires a string or integer id')
        name = raw['id']
        if name in graph or name in external:
            raise StrataError('DAG node IDs must be unique and separate from external inputs')
        dependencies = raw.get('dependencies', [])
        if not isinstance(dependencies, (list, tuple)) or any(not valid_id(v) for v in dependencies):
            raise StrataError('DAG dependencies must be a sequence of input or node IDs')
        compute = _cost(raw.get('compute_cost'), 'compute_cost')
        scratch = _integer(raw.get('scratch_width', 0), 'scratch_width')
        graph[name] = dict(id=name, dependencies=set(dependencies),
                           width=_integer(raw.get('width'), 'width'),
                           compute=compute, uncompute=_cost(raw.get('uncompute_cost', compute), 'uncompute_cost'),
                           scratch=scratch,
                           uncompute_scratch=_integer(raw.get('uncompute_scratch_width', scratch), 'uncompute_scratch_width'))
    if not set(outputs) <= graph.keys():
        raise StrataError('requested outputs must name computed DAG nodes')
    if any(not row['dependencies'] <= graph.keys() | external for row in graph.values()):
        raise StrataError('DAG dependency is neither a node nor a declared input')
    ready, remaining = set(external), set(graph)
    while remaining:
        layer = {name for name in remaining if graph[name]['dependencies'] <= ready}
        if not layer: raise StrataError('kernel dependencies must form an acyclic DAG')
        ready.update(layer); remaining.difference_update(layer)
    relevant, pending = set(), list(outputs)
    while pending:
        name = pending.pop()
        if name in external or name in relevant: continue
        relevant.add(name); pending.extend(graph[name]['dependencies'])
    order = [name for name in graph if name in relevant]
    kernels = [graph[name] for name in order]
    indices = {name: index for index, name in enumerate(order)}
    dependencies = [sum(1 << indices[name] for name in row['dependencies'] if name in indices) for row in kernels]
    output_mask = sum(1 << indices[name] for name in outputs)
    baseline = input_width + fixed_width
    full_storage = baseline + sum(row['width'] for row in kernels)
    transient = max([consume_scratch_width, *(max(row['scratch'], row['uncompute_scratch']) for row in kernels)])
    cap = full_storage + transient if max_width is None else max_width
    mandatory = sum((row['compute'] + (row['uncompute'] if consume is not None or row['id'] not in outputs else 0)
                     for row in kernels), Fraction(0)) + (consume or 0)
    scope = 'supplied reversible kernel DAG, dependency rules, explicit prices and physical width cap'
    base = dict(kind='reversible_dag_storage_schedule', executable=False,
                status_scope=scope, mode='consume_then_cleanup' if consume is not None else 'retain_outputs_then_cleanup',
                fixed_width=fixed_width, input_width=input_width, max_width=cap,
                outputs=outputs, inputs=inputs, node_count=len(kernels),
                omitted_irrelevant_nodes=[name for name in graph if name not in relevant],
                limits=dict(max_states=max_states),
                lowering_obligation='bind every complete kernel and consumer, allocate clean storage, preserve external values, emit and independently check the full native circuit')
    if baseline > cap:
        return dict(base, status='infeasible', reason='fixed_input_width_exceeds_cap',
                    alternatives=[], pareto_complete=True, explored_states=0,
                    lower_bound=dict(cost=str(mandatory), scope=scope, evidence='mandatory kernel compute and cleanup costs'))

    # An immutable parent chain retains witnesses even if their search label is
    # later dominated. max_states bounds all allocated labels, including these.
    labels = [dict(mask=0, consumed=False, peak=baseline, cost=Fraction(0), parent=None, action=None, width=baseline)]
    best = {(0, False): {baseline: 0}}
    queue = [(Fraction(0), baseline, 0)]
    finals, explored, bounded, minimum_proved = [], 0, False, None

    def goal(mask, consumed):
        return (consumed and not mask) if consume is not None else mask == output_mask

    if goal(0, False): finals.append(0)

    def insert(parent, mask, consumed, width, peak, price, action):
        nonlocal bounded
        if peak > cap: return
        total = labels[parent]['cost'] + price
        state = (mask, consumed)
        live = best.setdefault(state, {})
        if any(old_peak <= peak and labels[index]['cost'] <= total for old_peak, index in live.items()): return
        if any(labels[index]['peak'] <= peak and labels[index]['cost'] <= total for index in finals): return
        if len(labels) >= max_states:
            bounded = True; return
        for old_peak, index in list(live.items()):
            if peak <= old_peak and total <= labels[index]['cost']: del live[old_peak]
        index = len(labels)
        labels.append(dict(mask=mask, consumed=consumed, width=width, peak=peak,
                           cost=total, parent=parent, action=action))
        live[peak] = index
        heapq.heappush(queue, (total, peak, index))
        if goal(mask, consumed): finals.append(index)

    while queue and not bounded:
        total, peak, index = heapq.heappop(queue)
        current = labels[index]
        mask, consumed, width = current['mask'], current['consumed'], current['width']
        if best[(mask, consumed)].get(peak) != index: continue
        explored += 1
        if goal(mask, consumed):
            if minimum_proved is None: minimum_proved = total
            continue
        for bit, row in enumerate(kernels):
            if dependencies[bit] & mask != dependencies[bit]: continue
            present = bool(mask >> bit & 1)
            next_width = width + (-row['width'] if present else row['width'])
            scratch = row['uncompute_scratch'] if present else row['scratch']
            step_peak = max(width, next_width) + scratch
            price = row['uncompute'] if present else row['compute']
            action = dict(action='uncompute' if present else 'compute', node=row['id'],
                          cost=str(price), width_before=width, width_after=next_width,
                          scratch_width=scratch, operation_peak=step_peak)
            insert(index, mask ^ (1 << bit), consumed, next_width, max(peak, step_peak), price, action)
            if bounded: break
        if not bounded and consume is not None and not consumed and mask & output_mask == output_mask:
            step_peak = width + consume_scratch_width
            action = dict(action='consume', nodes=outputs, cost=str(consume),
                          width_before=width, width_after=width,
                          scratch_width=consume_scratch_width, operation_peak=step_peak)
            insert(index, mask, True, width, max(peak, step_peak), consume, action)

    frontier = []
    for index in sorted(finals, key=lambda i: (labels[i]['peak'], labels[i]['cost'])):
        final = labels[index]
        if any(old['peak_width'] <= final['peak'] and Fraction(old['cost']) <= final['cost'] for old in frontier): continue
        schedule, cursor = [], index
        while labels[cursor]['parent'] is not None:
            schedule.append(labels[cursor]['action']); cursor = labels[cursor]['parent']
        schedule.reverse()
        compute_counts = Counter(step['node'] for step in schedule if step['action'] == 'compute')
        uncompute_counts = Counter(step['node'] for step in schedule if step['action'] == 'uncompute')
        frontier.append(dict(cost=str(final['cost']), peak_width=final['peak'],
                             final_width=final['width'], schedule=schedule,
                             compute_counts=dict(compute_counts), uncompute_counts=dict(uncompute_counts),
                             recomputations=sum(max(0, count - 1) for count in compute_counts.values()),
                             cleanup_complete=True, consumer_executed=final['consumed']))
    lower = mandatory if minimum_proved is None else minimum_proved
    return dict(base, status='bounded' if bounded else 'exact' if frontier else 'infeasible',
                reason='state_label_budget' if bounded else None,
                alternatives=frontier, pareto_complete=not bounded,
                explored_states=explored, allocated_labels=len(labels),
                minimum_cost_proved=None if minimum_proved is None else str(minimum_proved),
                lower_bound=dict(cost=str(lower), scope=scope,
                    evidence='Dijkstra minimum complete schedule under width cap' if minimum_proved is not None else 'mandatory kernel compute and cleanup costs'),
                no_global_algorithmic_optimality_claim=True)
