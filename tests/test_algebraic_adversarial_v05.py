"""Independent adversarial checks of emitted algebraic candidates and billing."""
import random

import kickmix as km
import numpy as np
import pytest

from strata.algebraic_synthesis import algebraic_boundary_synthesis
from strata.kickmix import Kickmix
from strata.semantic import equivalent_windows


def materialized_polynomials(seed):
    rng = random.Random(seed)
    terms = set()
    while len(terms) < 8:
        terms.add(tuple(sorted(rng.sample(range(6), rng.choice((2, 3, 4))))))
    mapping = rng.sample(range(6), 6)
    forward, consumer, next_wire = [], [], 8
    for term in sorted(terms):
        controls = [mapping[q] for q in term]
        target = next_wire
        next_wire += 1
        forward.append(f'CCX q{controls[0]} q{controls[1]} q{target}')
        for control in controls[2:]:
            old, target = target, next_wire
            next_wire += 1
            forward.append(f'CCX q{old} q{control} q{target}')
        outputs = (6, 7) if rng.randrange(3) == 0 else (rng.choice((6, 7)),)
        consumer.extend(f'CX q{target} q{output}' for output in outputs)
    basis = [f'CX q{a} q{b}' for a, b in (rng.sample(range(6), 2) for _ in range(3))]
    return km.Circuit('\n'.join(basis + forward + consumer + forward[::-1] + basis[::-1])), tuple(range(8, next_wire))


def all_native(circuit, width, public=8):
    cases = 1 << public
    sim = km.Simulator(batch_size=cases, seed=902)
    wires = [km.q(q) for q in range(width)]
    initial = np.zeros((width, cases), dtype=np.bool_)
    for q in range(public):
        initial[q] = [bool(row >> q & 1) for row in range(cases)]
    sim.write_across_shots(wires, initial)
    sim.do(circuit)
    return sim.read_across_shots(wires), sim.read_phase_flipped_across_shots()


@pytest.mark.parametrize('seed', [17, 39, 61, 83, 105, 127])
def test_unrelated_anf_emitters_preserve_arbitrary_dirty_targets_and_all_scratch(seed):
    circuit, clean = materialized_polynomials(seed)
    result = algebraic_boundary_synthesis(circuit, clean_inputs=clean, timeout=2)
    assert result['status'] == 'constructed', result
    original = all_native(circuit, circuit.num_qubits)
    emitted = all_native(km.Circuit(result['candidate']), circuit.num_qubits)
    assert all(np.array_equal(a, b) for a, b in zip(original, emitted))
    assert result['polynomial_replay']['truth_rows_enumerated'] == 0


def test_complete_parent_resource_measure_includes_fixed_width_and_context():
    circuit, clean = materialized_polynomials(17)
    def parent(child):
        return km.Circuit('X q63\nX q63\n' + str(child))
    def measure(child):
        return Kickmix().measured(parent(child))
    result = algebraic_boundary_synthesis(circuit, clean_inputs=clean,
                                          cost='Q*(toffoli+instructions/16)', resource_measure=measure)
    assert result['status'] == 'constructed'
    assert result['resource_bill']['before']['Q'] == result['resource_bill']['after']['Q'] == 64
    assert result['resource_bill']['after'] == measure(km.Circuit(result['candidate']))
    assert result['cost'] == min(row['cost'] for row in result['search']['attempts'] if row['status'] == 'verified')
    assert equivalent_windows(str(parent(circuit)), str(parent(km.Circuit(result['candidate']))),
                              clean_inputs=clean, max_operations=10000, max_variables=128,
                              max_terms=65536)['equivalent'] is True


@pytest.mark.parametrize('suffix', ['HMR q8 b0', 'BIT_STORE0 b0', 'X q0 if b0'])
def test_private_looking_measurement_or_classical_effect_is_not_silently_erased(suffix):
    circuit, clean = materialized_polynomials(17)
    result = algebraic_boundary_synthesis(km.Circuit(str(circuit) + '\n' + suffix), clean_inputs=clean)
    assert result['status'] == 'not_applicable'
    assert 'candidate' not in result


@pytest.mark.parametrize('steps', [1, 16, 128, 1024])
def test_order_budget_returns_only_exact_candidates_or_an_explicit_unknown(steps):
    circuit, clean = materialized_polynomials(39)
    result = algebraic_boundary_synthesis(circuit, clean_inputs=clean, max_factor_steps=steps)
    if result['status'] == 'constructed':
        assert result['check']['equivalent'] is True
        assert 'not claimed' in result['search']['optimality']
        assert all(np.array_equal(a, b) for a, b in zip(all_native(circuit, circuit.num_qubits),
                   all_native(km.Circuit(result['candidate']), circuit.num_qubits)))
    else:
        assert result['status'] == 'unknown' and 'budget' in result['reason']


def test_modified_source_as_target_dependency_is_not_an_xor_translation():
    circuit = km.Circuit('CX q1 q0\nCCX q0 q2 q3')
    result = algebraic_boundary_synthesis(circuit)
    assert result['status'] == 'not_applicable'
    assert 'modified targets' in result['reason']


def test_an_infeasible_order_does_not_prevent_later_feasible_factorizations():
    # Independent representation-search audit found that one helper fits the
    # first and third orders but not the second. No search order is supplied.
    source = '''CCX q0 q1 q4
CCX q0 q2 q4
CCX q1 q3 q4
CX q3 q4
CCX q0 q1 q6
CCX q6 q3 q5
CCX q0 q1 q6
CCX q0 q2 q5
CCX q0 q2 q6
CCX q6 q3 q5
CCX q0 q2 q6
CX q1 q5
CCX q1 q2 q5
CCX q1 q3 q5
CX q2 q5
CCX q2 q3 q5
CX q3 q5'''
    circuit = km.Circuit(source)
    result = algebraic_boundary_synthesis(circuit, clean_inputs=(6,), max_orders=8)
    assert result['status'] == 'constructed' and result['search']['stopped_reason'] is None
    attempts = result['search']['attempts']
    failed = next(i for i, row in enumerate(attempts) if row['status'] == 'not_lowerable')
    later = [row for row in attempts[failed + 1:] if row['status'] == 'verified']
    assert later and min(row['cost'] for row in later) < attempts[0]['cost']
    assert result['cost'] == min(row['cost'] for row in attempts if row['status'] == 'verified')
    assert all(np.array_equal(a, b) for a, b in zip(all_native(circuit, 7, public=6),
               all_native(km.Circuit(result['candidate']), 7, public=6)))
