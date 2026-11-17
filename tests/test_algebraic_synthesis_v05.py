"""Symbolic discovery beyond truth-table limits on whole materialization banks."""
import kickmix as km
import numpy as np
import pytest

from strata.algebraic_synthesis import algebraic_boundary_synthesis
from strata.kickmix import Kickmix
from strata.model import StrataError
from strata.semantic import equivalent_windows


def layered_bank(group_size, *, expanded=False, shifted=False, reverse=False):
    """Independent emitter: two full banks feed two arbitrary accumulators."""
    size = group_size
    public = 3 * size
    pairs = [(a, b) for a in range(size) for b in range(size, 2 * size)] if expanded else [
        (a, size) for a in range(size)]
    if reverse:
        pairs.reverse()
    first = list(range(public + 2, public + 2 + len(pairs)))
    second = list(range(first[-1] + 1, first[-1] + 1 + len(pairs)))
    b_basis = [] if expanded else [f'CX q{q} q{size}' for q in range(size + 1, 2 * size)]
    c_basis = [f'CX q{q} q{2 * size}' for q in range(2 * size + 1, 3 * size)]
    forward = b_basis + [f'CCX q{a} q{b} q{p}' for (a, b), p in zip(pairs, first)] + b_basis[::-1]
    forward += c_basis + [f'CCX q{p} q{2*size} q{h}' for p, h in zip(first, second)] + c_basis[::-1]
    consumers = [f'CX q{q} q{public}' for q in second] + [f'CX q{q} q{public+1}' for q in first]
    if reverse:
        consumers.reverse()
    basis = ['X q0'] if shifted else []
    return km.Circuit('\n'.join(basis + forward + consumers + forward[::-1] + basis[::-1])), tuple(first + second)


def native(circuit, width, clean, rows):
    public = [q for q in range(width) if q not in clean]
    sim = km.Simulator(batch_size=len(rows), seed=319)
    initial = np.zeros((width, len(rows)), dtype=np.bool_)
    for i, wire in enumerate(public):
        initial[wire] = [(row >> i) & 1 for row in rows]
    wires = [km.q(q) for q in range(width)]
    sim.write_across_shots(wires, initial)
    sim.do(circuit)
    return sim.read_across_shots(wires), sim.read_phase_flipped_across_shots()


@pytest.mark.parametrize('size,expanded,shifted,reverse', [
    (4, True, False, False), (8, False, True, True)])
def test_symbolic_12_and_24_input_families_retire_multilevel_materialization_banks(size, expanded, shifted, reverse):
    circuit, clean = layered_bank(size, expanded=expanded, shifted=shifted, reverse=reverse)
    result = algebraic_boundary_synthesis(circuit, clean_inputs=clean, timeout=5)
    assert result['status'] == 'constructed', result
    assert len(result['dag']['input_wires']) == 3 * size
    assert result['proof']['truth_rows_enumerated'] == 0
    assert result['polynomial_replay']['status'] == 'verified'
    assert result['search']['optimality'].startswith('not claimed')
    assert result['structural_witness']['removed_materialization_bank']
    assert len(result['structural_witness']['retired_scratch_wires']) >= len(clean) - 2
    after = km.Circuit(result['candidate'])
    assert after.num_qubits <= 3 * size + 4
    assert result['resource_bill']['after']['toffoli'] <= 4
    assert result['resource_bill']['after_cost'] < result['resource_bill']['before_cost'] / 5
    assert equivalent_windows(str(circuit), str(after), clean_inputs=clean,
                              max_operations=10000, max_variables=128, max_terms=65536)['equivalent']
    if size == 4:
        # Independent complete native cross-check includes both dirty targets.
        rows = list(range(1 << (3 * size + 2)))
    else:
        # Native smoke only; the 24-input proof is symbolic, not this sample.
        rows = [0, 1, (1 << (3 * size + 2)) - 1, 0x123456, 0x2345678, 0x1555555]
    original, emitted = native(circuit, circuit.num_qubits, clean, rows), native(after, circuit.num_qubits, clean, rows)
    assert all(np.array_equal(left, right) for left, right in zip(original, emitted))


def test_source_bank_schedule_and_wire_permutation_do_not_need_a_supplied_cut():
    from test_v05_structural_discovery_acceptance import fixtures
    for fixture in fixtures():
        if fixture.name in {'high_support_bound'}:
            continue
        result = algebraic_boundary_synthesis(km.Circuit(fixture.text), clean_inputs=fixture.clean)
        assert result['status'] == 'constructed', (fixture.name, result)
        assert result['check']['equivalent'] is True
        assert result['resource_bill']['after_cost'] < result['resource_bill']['before_cost']


def test_every_alternative_is_ranked_using_the_emitted_resource_callback():
    from test_v05_structural_discovery_acceptance import fixtures
    fixture = next(f for f in fixtures() if f.name == 'distributed_twolevel')
    calls = []
    def measure(circuit):
        resources = Kickmix().measured(circuit)
        calls.append(str(circuit))
        return dict(resources, complete_bill=100 * resources['toffoli'] + resources['instructions'])
    result = algebraic_boundary_synthesis(km.Circuit(fixture.text), clean_inputs=fixture.clean,
                                          resource_measure=measure, cost='complete_bill')
    attempts = result['search']['attempts']
    assert len(calls) == 1 + len(attempts)
    assert len({attempt['cost'] for attempt in attempts}) > 1
    assert result['cost'] == min(attempt['cost'] for attempt in attempts)
    assert result['resource_bill']['cost_expression'] == 'complete_bill'


@pytest.mark.parametrize('options,reason', [
    ({'max_factor_steps': 1}, 'step_budget'),
    ({'max_factor_nodes': 2}, 'node_budget'),
    ({'max_variables': 2}, 'variable_budget'),
    ({'max_operations': 1}, 'operation_budget'),
    ({'max_terms': 1}, 'budget')])
def test_algebraic_limits_do_not_become_approximate_proofs(options, reason):
    circuit, clean = layered_bank(4, expanded=True)
    result = algebraic_boundary_synthesis(circuit, clean_inputs=clean, **options)
    assert result['status'] == 'unknown' and reason in result['reason']
    assert 'candidate' not in result


def test_order_limit_and_early_stop_are_explicit_not_optimality_claims():
    circuit, clean = layered_bank(4, expanded=True)
    result = algebraic_boundary_synthesis(circuit, clean_inputs=clean, max_orders=1)
    assert result['status'] == 'constructed'
    assert len(result['search']['attempts']) == 1
    assert 'not claimed' in result['search']['optimality']


def test_affine_terms_need_no_product_storage_and_repeated_metadata_is_stable():
    circuit = km.Circuit('CX q0 q3\nCCX q1 q2 q3\nX q4\nX q4')
    result = algebraic_boundary_synthesis(circuit, clean_inputs=(4,))
    assert result['status'] == 'constructed'
    assert len(result['dag']['gates']) == 1
    assert result['lowering']['materialized_products'] == 0
    repeated = algebraic_boundary_synthesis(km.Circuit(result['candidate']))
    assert repeated['status'] == 'constructed'
    assert repeated['candidate'] == result['candidate']


def test_phase_and_measurement_boundaries_decline_and_dirty_wires_cannot_disappear():
    circuit, clean = layered_bank(4, expanded=True)
    phase = algebraic_boundary_synthesis(km.Circuit(str(circuit) + '\nZ q0'), clean_inputs=clean)
    assert phase['status'] == 'not_applicable' and 'phase' in phase['reason']
    measurement = algebraic_boundary_synthesis(km.Circuit(str(circuit) + f'\nHMR q{clean[-1]} b0'), clean_inputs=clean)
    assert measurement['status'] == 'not_applicable' and 'measurement' in measurement['reason']
    dirty = algebraic_boundary_synthesis(circuit, max_orders=1, timeout=.5)
    if dirty['status'] == 'constructed':
        assert km.Circuit(dirty['candidate']).num_qubits == circuit.num_qubits
        assert dirty['check']['equivalent'] is True
    else:
        assert dirty['status'] == 'unknown'


def test_independent_replay_rejects_an_incorrect_factoring_result(monkeypatch):
    from strata import algebraic_synthesis
    circuit, clean = layered_bank(4, expanded=True)
    monkeypatch.setattr(algebraic_synthesis._Factors, 'product', lambda self, left, right: 1)
    with pytest.raises(StrataError, match='polynomial replay mismatch'):
        algebraic_boundary_synthesis(circuit, clean_inputs=clean)


def test_phase_mutation_and_clean_output_leak_are_rejected_by_full_boundary_verifier():
    circuit, clean = layered_bank(4, expanded=True)
    result = algebraic_boundary_synthesis(circuit, clean_inputs=clean)
    for mutation in ('Z q0', f'X q{clean[-1]}'):
        check = equivalent_windows(str(circuit), result['candidate'] + '\n' + mutation,
                                   clean_inputs=clean, max_operations=10000, max_variables=128,
                                   max_terms=65536)
        assert check['equivalent'] is False
