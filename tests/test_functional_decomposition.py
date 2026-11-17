import itertools

import kickmix as km

from strata.algebraic_synthesis import _Budget
from strata.domains import execute_assignment
from strata.functional_decomposition import _classes, decomposition_synthesis


def nonlinear_consumer():
    # Expanded monomials, emitted in target order, with one reused work wire.
    lines = []
    for target, other in ((5, 3), (6, 4)):
        for a, b in ((0, 1), (0, 2), (1, 2)):
            gate = f'CCX q{a} q{b} q7'
            lines.extend([gate, f'CCX q7 q{other} q{target}', gate])
    # Explicitly available zero workspace with a fixed public physical width.
    return km.Circuit('\n'.join(lines + ['X q8', 'X q8', 'X q9', 'X q9']))


def test_detect_and_lower_hidden_consumer_partition_on_arbitrary_dirty_targets():
    source = nonlinear_consumer()
    result = decomposition_synthesis(source, clean_inputs=[7, 8], max_partitions=96)
    assert result['status'] == 'constructed'
    assert result['resources']['Q'] == 10
    from strata.algebraic_synthesis import algebraic_boundary_synthesis
    direct = algebraic_boundary_synthesis(source, clean_inputs=[7,8])
    assert result['resources']['toffoli'] == 4 < direct['resources']['toffoli']
    w = result['structural_witness']
    assert set(w['bound_inputs']) == {0, 1, 2}
    assert w['classes_by_assignment'] == [0, 0, 0, 1, 0, 1, 1, 1]
    assert result['search']['partitions_tested'] > 1
    for bits in itertools.product(range(2), repeat=8):
        row = {f'q{i}': v for i,v in enumerate([*bits[:7], 0, 0, bits[7]])}
        output = execute_assignment(result['candidate'].splitlines(), row, 10)
        h = int(sum(bits[:3]) >= 2)
        expected = row.copy()
        for target, control in ((5,3),(6,4)):
            expected[f'q{target}'] ^= h & bits[control]
        assert output['outputs'] == {str(i): expected[f'q{i}'] for i in range(10)}
        assert not output['phase']


def test_additional_consumer_prevents_an_invalid_merge():
    poly = lambda *terms: frozenset(frozenset(t) for t in terms)
    first = poly((0,1,3), (0,2,3), (1,2,3))
    outputs = [first, poly((0,4))]
    budget = _Budget(10000,100,100,100,1)
    assert _classes(outputs,(0,1,2),budget) is None


def test_decomposition_budget_does_not_claim_a_floor():
    res = decomposition_synthesis(nonlinear_consumer(), clean_inputs=[7,8], max_steps=1)
    assert res['status'] == 'unknown'
    assert res['search']['stop_reason'] == 'algebraic_factor_step_budget'
