from fractions import Fraction
import heapq
import random

import pytest

from strata.model import StrataError
from strata.pebbling import plan_reversible_pebbles


def chain(length):
    return [dict(id=f'v{i}', dependencies=['x' if not i else f'v{i-1}'], width=1, compute_cost=1)
            for i in range(length)]


def replay(nodes, outputs, alternative, *, inputs=(), baseline=0, consume=None):
    graph = {row['id']: row for row in nodes}
    present, cost, peak, consumed = set(inputs), Fraction(0), baseline, False
    for step in alternative['schedule']:
        width = baseline + sum(graph[name]['width'] for name in present if name in graph)
        assert step['width_before'] == width
        if step['action'] == 'consume':
            assert not consumed and set(outputs) <= present
            assert Fraction(step['cost']) == Fraction(consume)
            consumed = True
        else:
            name = step['node']; row = graph[name]
            assert set(row['dependencies']) <= present
            if step['action'] == 'compute':
                assert name not in present
                present.add(name)
                assert Fraction(step['cost']) == Fraction(row['compute_cost'])
            else:
                assert name in present
                present.remove(name)
                assert Fraction(step['cost']) == Fraction(row.get('uncompute_cost', row['compute_cost']))
        after = baseline + sum(graph[name]['width'] for name in present if name in graph)
        assert after == step['width_after']
        assert step['operation_peak'] == max(width, after) + step['scratch_width']
        peak = max(peak, step['operation_peak'])
        cost += Fraction(step['cost'])
    assert present == set(inputs) | (set(outputs) if consume is None else set())
    assert consumed == (consume is not None)
    assert peak == alternative['peak_width'] and cost == Fraction(alternative['cost'])
    assert alternative['cleanup_complete'] is True


def test_retained_output_frontier_prices_recomputation_and_full_cleanup():
    nodes = chain(4)
    result = plan_reversible_pebbles(nodes, ['v3'], inputs=['x'], input_width=2, fixed_width=3)
    assert result['status'] == 'exact' and result['pareto_complete']
    assert [(r['peak_width'], r['cost']) for r in result['alternatives']] == [(8, '9'), (9, '7')]
    assert result['alternatives'][0]['recomputations'] == 1
    assert result['lower_bound']['cost'] == '7'
    assert result['executable'] is False and result['no_global_algorithmic_optimality_claim']
    for row in result['alternatives']: replay(nodes, ['v3'], row, inputs=['x'], baseline=5)
    constrained = plan_reversible_pebbles(nodes, ['v3'], inputs=['x'], input_width=2, fixed_width=3, max_width=8)
    assert constrained['minimum_cost_proved'] == '9'


def test_observable_consumer_runs_once_then_cleans_entire_dag():
    nodes = chain(3)
    result = plan_reversible_pebbles(nodes, ['v2'], inputs=['x'], input_width=2, consume_cost=3)
    assert [(r['peak_width'], r['cost']) for r in result['alternatives']] == [(4, '11'), (5, '9')]
    for row in result['alternatives']:
        replay(nodes, ['v2'], row, inputs=['x'], baseline=2, consume=3)
        assert row['final_width'] == 2
        assert sum(step['action'] == 'consume' for step in row['schedule']) == 1


def test_kernel_transient_workspace_and_asymmetric_uncompute_prices_are_charged():
    nodes = [dict(id='a', dependencies=['x'], width=2, compute_cost='1/3', uncompute_cost='2/3', scratch_width=4),
             dict(id='b', dependencies=['a'], width=3, compute_cost='5/3', uncompute_cost='7/3',
                  scratch_width=1, uncompute_scratch_width=5)]
    result = plan_reversible_pebbles(nodes, ['b'], inputs=['x'], input_width=2, consume_cost='1/2',
                                    consume_scratch_width=7)
    assert result['status'] == 'exact'
    assert [(r['peak_width'], r['cost']) for r in result['alternatives']] == [(12, '13/2'), (14, '11/2')]
    for row in result['alternatives']:
        replay(nodes, ['b'], row, inputs=['x'], baseline=2, consume='1/2')
    impossible = plan_reversible_pebbles(nodes, ['b'], inputs=['x'], input_width=2,
                                        consume_cost='1/2', consume_scratch_width=7, max_width=11)
    assert impossible['status'] == 'infeasible' and impossible['alternatives'] == []


def brute_cost(nodes, outputs, cap, consume):
    """Independent single-objective exhaustive search for each fixed capacity."""
    ids = [row['id'] for row in nodes]
    target = frozenset(outputs)
    start = (frozenset(), False)
    distance, queue, sequence = {start: Fraction(0)}, [(Fraction(0), 0, start)], 0
    while queue:
        cost, _, state = heapq.heappop(queue)
        if distance[state] != cost: continue
        live, consumed = state
        if (consumed and not live) if consume is not None else live == target: return cost
        width = sum(nodes[ids.index(name)]['width'] for name in live)
        transitions = []
        for row in nodes:
            if not set(row['dependencies']) <= live | {'x'}: continue
            name = row['id']; removing = name in live
            after = live - {name} if removing else live | {name}
            next_width = width + (-row['width'] if removing else row['width'])
            scratch = row.get('uncompute_scratch_width', row.get('scratch_width', 0)) if removing else row.get('scratch_width', 0)
            price = row.get('uncompute_cost', row['compute_cost']) if removing else row['compute_cost']
            if max(width, next_width) + scratch <= cap:
                transitions.append(((frozenset(after), consumed), Fraction(price)))
        if consume is not None and not consumed and target <= live:
            transitions.append(((live, True), Fraction(consume)))
        for next_state, price in transitions:
            total = cost + price
            if next_state not in distance or total < distance[next_state]:
                distance[next_state] = total; sequence += 1
                heapq.heappush(queue, (total, sequence, next_state))
    return None


def test_pareto_frontier_matches_independent_capacity_search_on_generated_dags():
    rng = random.Random(618)
    for _ in range(14):
        nodes = []
        for i in range(5):
            deps = [f'v{j}' for j in range(i) if rng.randrange(2)] or ['x']
            nodes.append(dict(id=f'v{i}', dependencies=deps, width=rng.randrange(1, 3),
                              compute_cost=rng.randrange(1, 4), uncompute_cost=rng.randrange(1, 4),
                              scratch_width=rng.randrange(2)))
        outputs = ['v3', 'v4']; consume = rng.choice([None, 2])
        result = plan_reversible_pebbles(nodes, outputs, inputs=['x'], consume_cost=consume)
        assert result['status'] == 'exact'
        for alternative in result['alternatives']:
            replay(nodes, outputs, alternative, inputs=['x'], consume=consume)
        for cap in range(sum(row['width'] for row in nodes) + 2):
            brute = brute_cost(nodes, outputs, cap, consume)
            projected = [Fraction(row['cost']) for row in result['alternatives'] if row['peak_width'] <= cap]
            assert (min(projected) if projected else None) == brute


def test_budget_exhaustion_never_claims_pareto_optimality_or_native_emission():
    result = plan_reversible_pebbles(chain(8), ['v7'], inputs=['x'], max_states=4)
    assert result['status'] == 'bounded' and result['reason'] == 'state_label_budget'
    assert result['pareto_complete'] is False and result['minimum_cost_proved'] is None
    assert result['lower_bound']['cost'] == '15'
    assert 'supplied' in result['lower_bound']['scope']
    assert result['executable'] is False


def test_invalid_graphs_missing_prices_and_fixed_width_fail_closed():
    with pytest.raises(StrataError, match='acyclic'):
        plan_reversible_pebbles([dict(id='a', dependencies=['a'], width=1, compute_cost=1)], ['a'])
    with pytest.raises(StrataError, match='neither a node'):
        plan_reversible_pebbles(chain(2), ['v1'])
    with pytest.raises(StrataError, match='compute_cost'):
        plan_reversible_pebbles([dict(id='a', width=1)], ['a'])
    result = plan_reversible_pebbles(chain(2), ['v1'], inputs=['x'], input_width=4, max_width=3)
    assert result['status'] == 'infeasible'
