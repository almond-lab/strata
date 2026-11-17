import kickmix as km
import numpy as np
import pytest

from strata.accounting import measured_resources
from strata.architecture_portfolio import _canonical_graph, architecture_portfolio
from strata.model import StrataError


POLYNOMIALS = [
    [(1,), (3,), (1, 3), (0, 1, 3), (1, 2, 3)],
    [(0, 1), (1, 2), (0, 1, 2)],
    [(0,), (1,), (2,), (3,), (0, 2), (0, 3), (0, 1, 3)],
]


def source(polynomials=POLYNOMIALS):
    helper = 4 + len(polynomials)
    lines = [f'APPEND_TO_REGISTER q{helper + 2} r0', 'REGISTER r0 "declared_capacity"']
    for target, polynomial in enumerate(polynomials, 4):
        lines += [f'APPEND_TO_REGISTER q{target} r1']
        for term in polynomial:
            if len(term) == 1:
                lines.append(f'CX q{term[0]} q{target}')
            elif len(term) == 2:
                lines.append(f'CCX q{term[0]} q{term[1]} q{target}')
            else:
                lines += [f'CCX q{term[0]} q{term[1]} q{helper}',
                          f'CCX q{helper} q{term[2]} q{target}',
                          f'CCX q{term[0]} q{term[1]} q{helper}']
    lines.append('REGISTER r1 "arbitrary_outputs"')
    return km.Circuit('\n'.join(lines)), [helper, helper + 1, helper + 2]


def test_a_thousand_canonical_complete_formulations_precede_bounded_native_finalists():
    circuit, clean = source()
    calls = []
    def price(candidate):
        calls.append(str(candidate))
        return measured_resources(candidate)
    result = architecture_portfolio(circuit, clean_inputs=clean, resource_measure=price,
                                    max_hypotheses=1024, timeout=10)
    search = result['search']
    assert search['unique_formulations'] == search['independently_replayed'] == 1024
    assert search['proposed_attempts'] > search['unique_formulations']
    assert len({row['formulation_hash'] for row in result['formulations']}) == 1024
    assert len({row['graph_hash'] for row in result['formulations']}) == search['unique_product_graphs']
    assert search['categories']['joint'] > 0 and search['categories']['input_only'] > 0
    assert search['categories']['output_only'] > 0 and search['categories']['identity'] > 0
    assert search['native_evaluations_in_wide_stage'] == 0
    assert len(calls) == 1 + search['verified'] <= 13
    assert search['lowered'] <= search['selected_finalists'] <= 12
    assert result['identity_comparator']['status'] == 'verified'
    assert result['cost'] < result['identity_comparator']['cost'] < result['resource_bill']['before_cost']
    assert result['structural_witness']['joint_architecture_change']
    assert all(len(row['layout']['helper_wires']) <= len(clean) for row in result['formulations'])
    assert all(row['polynomial_replay']['status'] == 'verified' for row in result['formulations'])
    assert set(result['pareto_native_hashes']) <= {row['candidate_hash'] for row in result['verified_elites']}
    assert len(result['verified_elites']) == len({row['candidate_hash'] for row in result['verified_elites']})
    native_results = []
    for native in (circuit, km.Circuit(result['candidate'])):
        sim = km.Simulator(batch_size=128, seed=889)
        initial = np.zeros((10, 128), dtype=np.bool_)
        for q in range(7):
            initial[q] = [(row >> q) & 1 for row in range(128)]
        wires = [km.q(q) for q in range(10)]
        sim.write_across_shots(wires, initial)
        sim.do(native)
        native_results.append((sim.read_across_shots(wires), sim.read_phase_flipped_across_shots()))
    assert all(np.array_equal(a, b) for a, b in zip(*native_results))
    for elite in result['verified_elites']:
        assert elite['whole_boundary_check']['equivalent'] and elite['proof']['all_public_quantum_inputs']
        assert elite['native_resources'] == measured_resources(km.Circuit(elite['candidate']))
        assert km.Circuit(elite['candidate']).num_qubits == circuit.num_qubits
        for line in str(circuit).splitlines():
            if line.startswith(('REGISTER', 'APPEND_TO_REGISTER')):
                assert line in elite['candidate'].splitlines()


def test_isomorphic_graphs_operand_commutation_and_unused_nodes_do_not_inflate_count():
    first = [dict(left_coefficients=2, right_coefficients=4),
             dict(left_coefficients=4, right_coefficients=8)]
    second = [dict(left_coefficients=8, right_coefficients=4),
              dict(left_coefficients=4, right_coefficients=2),
              dict(left_coefficients=2, right_coefficients=8)]
    a = _canonical_graph([0, 1, 2], [3, 4], first, [16, 32])
    b = _canonical_graph([0, 1, 2], [3, 4], second, [32, 16])
    assert a == b


def test_joint_and_single_axis_ablations_use_the_same_search_and_lowering_budgets():
    circuit, clean = source()
    results = {mode: architecture_portfolio(circuit, clean_inputs=clean, representation_mode=mode,
        max_hypotheses=256, max_finalists=12, timeout=10) for mode in ('joint', 'input', 'output')}
    assert all(row['search']['unique_formulations'] == 256 for row in results.values())
    assert all(row['search']['limits']['max_factor_orders'] == 8 for row in results.values())
    assert all(row['search']['selected_finalists'] == 12 for row in results.values())
    assert results['joint']['cost'] <= min(results['input']['cost'], results['output']['cost'])
    # The held fixture ties input-only and improves output-only. The receipt
    # reports measured alternatives; it does not fabricate universal synergy.
    assert results['joint']['cost'] == results['input']['cost'] < results['output']['cost']
    assert not results['input']['search']['categories']['joint']
    assert not results['output']['search']['categories']['joint']


def test_full_host_objective_includes_both_input_and_output_adapters():
    circuit, clean = source()
    def price(candidate):
        resources = measured_resources(candidate)
        return dict(resources, host_cost=resources['instructions'] * 100 + resources['toffoli'])
    result = architecture_portfolio(circuit, clean_inputs=clean, resource_measure=price,
        cost='host_cost', max_hypotheses=64, max_finalists=8, timeout=5)
    assert result['cost'] <= price(circuit)['host_cost']
    for alternative in result['verified_elites']:
        assert alternative['cost'] == price(km.Circuit(alternative['candidate']))['host_cost']
        assert alternative['structural_witness']['both_adapter_pairs_charged']
    named = km.Circuit(str(circuit).replace('declared_capacity', 'q999'))
    result = architecture_portfolio(named, clean_inputs=clean, max_hypotheses=16, timeout=5)
    assert all(row['resource_bounds']['Q'] == 10 for row in result['formulations'])
    assert result['verified_elites']


def test_joint_axes_can_improve_both_equally_budgeted_single_axis_searches():
    circuit, clean = source(POLYNOMIALS + [POLYNOMIALS[0]])
    results = {mode: architecture_portfolio(circuit, clean_inputs=clean, representation_mode=mode,
        max_hypotheses=512, max_finalists=16, timeout=10) for mode in ('joint', 'input', 'output')}
    assert all(row['search']['unique_formulations'] == 512 for row in results.values())
    assert results['joint']['native_resources']['toffoli'] == 7
    assert results['input']['native_resources']['toffoli'] == results['output']['native_resources']['toffoli'] == 8
    assert results['joint']['category'] == 'joint'
    assert results['joint']['cost'] < min(results['input']['cost'], results['output']['cost'])
    native_results = []
    for native in (circuit, km.Circuit(results['joint']['candidate'])):
        sim = km.Simulator(batch_size=256, seed=57)
        initial = np.zeros((11, 256), dtype=np.bool_)
        for q in range(8):
            initial[q] = [(row >> q) & 1 for row in range(256)]
        wires = [km.q(q) for q in range(11)]
        sim.write_across_shots(wires, initial)
        sim.do(native)
        native_results.append((sim.read_across_shots(wires), sim.read_phase_flipped_across_shots()))
    assert all(np.array_equal(a, b) for a, b in zip(*native_results))


def test_identity_factoring_gains_do_not_credit_gratuitous_basis_changes():
    circuit = km.Circuit('CCX q0 q2 q3\nCCX q1 q2 q3\nCX q0 q4')
    result = architecture_portfolio(circuit, cost='toffoli', max_hypotheses=64, timeout=5)
    assert result['identity_comparator']['cost'] == result['cost'] == 1
    assert result['resource_bill']['before_cost'] == 2
    assert any(row['category'] != 'identity' and row['cost'] == 1 for row in result['verified_elites'])
    assert all(not row['structural_witness']['joint_architecture_change'] for row in result['verified_elites'])
    one = architecture_portfolio(circuit, cost='toffoli', max_hypotheses=16, max_finalists=1, timeout=5)
    assert one['verified_elites'][0]['category'] == 'identity'


def test_missing_identity_proof_withholds_structural_credit(monkeypatch):
    import strata.architecture_portfolio as portfolio
    lower, calls = portfolio._lower, []
    def corrupt_control(*args, **kwargs):
        circuit, layout = lower(*args, **kwargs)
        if not calls:
            circuit = km.Circuit('X q0')
        calls.append(True)
        return circuit, layout
    monkeypatch.setattr(portfolio, '_lower', corrupt_control)
    circuit, clean = source()
    result = portfolio.architecture_portfolio(circuit, clean_inputs=clean, max_hypotheses=64,
                                              max_finalists=2, timeout=5)
    assert result['identity_comparator']['status'] == 'unknown'
    assert not result['structural_witness']['joint_architecture_change']
    assert any(row['reason'] == 'architecture_complete_boundary_check' for row in result['failures'])


def test_polynomial_or_native_mutation_cannot_create_a_verified_architecture(monkeypatch):
    import strata.architecture_portfolio as portfolio
    circuit, clean = source()
    original = portfolio._Factors.product
    monkeypatch.setattr(portfolio._Factors, 'product', lambda *args: 0)
    result = portfolio.architecture_portfolio(circuit, clean_inputs=clean, max_representations=4, timeout=5)
    assert not result['verified_elites'] and not result['formulations']
    assert result['selected_mechanism'] == 'source_incumbent'
    monkeypatch.setattr(portfolio._Factors, 'product', original)
    monkeypatch.setattr(portfolio, '_lower', lambda *args, **kwargs: (km.Circuit('X q0'), {}))
    result = portfolio.architecture_portfolio(circuit, clean_inputs=clean, max_hypotheses=16, timeout=5)
    assert not result['verified_elites'] and result['selected_mechanism'] == 'source_incumbent'


def test_budgets_and_unsupported_boundaries_remain_explicit():
    circuit, clean = source()
    result = architecture_portfolio(circuit, clean_inputs=clean, max_work_steps=1)
    assert result['search']['stopped_reason'] == 'algebraic_factor_step_budget'
    assert result['selected_mechanism'] == 'source_incumbent'
    for native in ('Z q0', 'HMR q0 b0', 'CX q0 q1\nCCX q0 q1 q2', 'R q0'):
        assert architecture_portfolio(km.Circuit(native))['status'] == 'not_applicable'
    for limits in ({'max_finalists': 0}, {'max_hypotheses': True}, {'timeout': float('nan')},
                   {'representation_mode': 'unproved'}):
        with pytest.raises(StrataError):
            architecture_portfolio(circuit, clean_inputs=clean, **limits)
