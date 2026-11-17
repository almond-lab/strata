"""Blind file-first acceptance cases; the fixtures supply no optimizer adapter.

The oracle below uses native kickmix over every initial wire assignment.  In
particular, restored intermediates are arbitrary dirty inputs, not assumed zero.
These are original small examples, not contest circuit extracts.
"""
from fractions import Fraction
import copy
import hashlib
import json
import random

import kickmix as km
import numpy as np
import pytest


def hidden_materializations(seed):
    """Two nonlinear, multi-operation producers with two outputs and sign phase."""
    rng = random.Random(seed)
    wires = list(range(9))
    rng.shuffle(wires)
    a, b, c, d, t, u, y, z, s = wires
    producers = [
        f'CCX q{a} q{b} q{t}', f'CCX q{c} q{d} q{t}', f'CX q{a} q{t}',
        f'CCX q{a} q{c} q{u}', f'CCX q{b} q{d} q{u}', f'X q{u}',
    ]
    rng.shuffle(producers)
    consumers = [f'CX q{t} q{y}', f'CX q{u} q{z}', f'CZ q{t} q{s}', f'Z q{u}', 'NEG']
    rng.shuffle(consumers)
    circuit = km.Circuit('\n'.join([*producers, *consumers, *reversed(producers)]))
    return circuit, wires


def native_state(circuit, width=None):
    """Independent native execution, including every dirty wire and exact phase."""
    width = max(width or 0, circuit.num_qubits)
    cases = 1 << width
    numbers = np.arange(cases, dtype=np.uint64)
    initial = np.array([((numbers >> q) & 1).astype(bool) for q in range(width)])
    sim = km.Simulator(batch_size=cases, seed=846)
    wires = [km.q(q) for q in range(width)]
    sim.write_across_shots(wires, initial)
    sim.do(circuit)
    final = sim.read_across_shots(wires)
    phase = tuple(Fraction(sim.read_shot_phase(i)) % 2 for i in range(cases))
    return initial, final, phase


def assert_hidden_reference(circuit, wires):
    a, b, c, d, t, u, y, z, s = wires
    initial, actual, phase = native_state(circuit, width=9)
    product_t = initial[a] & initial[b] ^ initial[c] & initial[d] ^ initial[a]
    product_u = initial[a] & initial[c] ^ initial[b] & initial[d] ^ True
    expected = initial.copy()
    expected[y] ^= initial[t] ^ product_t
    expected[z] ^= initial[u] ^ product_u
    expected_phase = (initial[t] ^ product_t) & initial[s] ^ initial[u] ^ product_u ^ True
    assert np.array_equal(actual, expected)
    assert phase == tuple(Fraction(int(v)) for v in expected_phase)
    # Clean-only tests would omit the very assignments protecting the contract.
    assert np.any(initial[t] & initial[u])


def eval_polynomial(description, initial):
    value = np.zeros(initial.shape[1], dtype=bool)
    for monomial in description['monomials']:
        term = np.ones(initial.shape[1], dtype=bool)
        for q in monomial:
            term &= initial[q]
        value ^= term
    return value


@pytest.mark.parametrize('seed', [3, 17, 61])
def test_inferred_multioperation_functions_match_independent_native_oracle(seed):
    from strata.semantic import analyze_window

    original, wires = hidden_materializations(seed)
    assert_hidden_reference(original, wires)
    result = analyze_window(str(original).splitlines())
    assert result['status'] == 'exact'
    initial, actual, phase = native_state(original)
    for q in range(original.num_qubits):
        assert np.array_equal(eval_polynomial(result['outputs'][str(q)], initial), actual[q])
    assert tuple(Fraction(int(v)) for v in eval_polynomial(result['phase'], initial)) == phase
    assert set(result['preserved_wires']) == set(wires) - {wires[6], wires[7]}
    assert set(result['modified_wires']) == {wires[6], wires[7]}
    assert result['boundary_width'] == 9


@pytest.mark.parametrize('seed,format', [(3, 'kmx'), (17, 'kmb'), (61, 'kmx')])
def test_unknown_file_infers_and_restructures_without_adapter(tmp_path, seed, format):
    from strata.file_optimizer import optimize_circuit
    from strata.kickmix import Kickmix

    original, wires = hidden_materializations(seed)
    path = tmp_path / f'unknown-{seed}.{format}'
    original.to_file(path, format=format)
    loaded = Kickmix().load(path)
    candidate, report = optimize_circuit(loaded)
    assert str(candidate) != str(loaded)
    assert report['candidate']['cost'] < report['baseline']['cost']
    assert report['checks']['equivalent'] is True
    assert report['checks']['all_inputs_arbitrary'] is True
    assert report['checks']['width_preserved'] is True
    assert any(region.get('removed_materializations') for region in report['regions'])
    assert candidate.num_qubits == loaded.num_qubits
    assert_hidden_reference(candidate, wires)


@pytest.mark.parametrize('format', ['kmx', 'kmb'])
def test_file_cli_needs_no_project_adapter_or_expected_answer(tmp_path, capsys, format):
    from strata.cli import main

    original, wires = hidden_materializations(127)
    source = tmp_path / f'unnamed.{format}'
    original.to_file(source, format=format)
    output = tmp_path / 'review-output'
    assert main(['--project', str(tmp_path / 'deliberately-missing.toml'),
                 'optimize', str(source), '--out', str(output)]) == 0
    json.loads(capsys.readouterr().out)
    candidate = km.Circuit.from_file(output / 'candidate.kmx')
    report = json.loads((output / 'result.json').read_text())
    assert report['checks']['equivalent'] is True
    assert report['candidate']['cost'] < report['baseline']['cost']
    assert (output / 'semantic-lift.json').is_file()
    assert (output / 'contract-tree.json').is_file()
    assert_hidden_reference(candidate, wires)


@pytest.mark.parametrize('mutation', ['output', 'dirty_helper', 'sign_phase'])
def test_mutated_output_helper_or_phase_is_not_equivalent(mutation):
    from strata.file_optimizer import optimize_circuit
    from strata.semantic import equivalent_windows

    original, wires = hidden_materializations(37)
    candidate, _ = optimize_circuit(original)
    gate = {'output': f'X q{wires[6]}', 'dirty_helper': f'X q{wires[4]}',
            'sign_phase': f'Z q{wires[0]}'}[mutation]
    mutant = km.Circuit(str(candidate) + '\n' + gate)
    result = equivalent_windows(str(original).splitlines(), str(mutant).splitlines())
    assert result['equivalent'] is False
    assert result['status'] == 'different'
    _, before, before_phase = native_state(original)
    _, after, after_phase = native_state(mutant)
    assert not np.array_equal(before, after) or before_phase != after_phase
    if mutation == 'sign_phase':
        assert np.array_equal(before, after)


@pytest.mark.parametrize('barrier', ['HMR q2 b0', 'CX q2 q3 if b0',
                                   'Z_POW q2 0.25', 'CSWAP q2 q3 q4'])
def test_native_effects_are_preserved_and_fractional_phase_remains_unknown(barrier):
    from strata.file_optimizer import optimize_circuit
    from strata.semantic import analyze_window, equivalent_windows

    original = km.Circuit(f'CCX q0 q1 q2\n{barrier}\nCCX q0 q1 q2')
    result = analyze_window(str(original).splitlines())
    assert result['status'] == ('opaque' if barrier.startswith('Z_POW') else 'exact')
    candidate, report = optimize_circuit(original)
    # Exact lifting alone does not authorize dropping effects or dependencies.
    assert str(candidate) == str(original)
    stripped = km.Circuit('CCX q0 q1 q2\nCCX q0 q1 q2')
    check = equivalent_windows(str(original).splitlines(), str(stripped).splitlines())
    assert check['equivalent'] is not True
    assert report['candidate']['cost'] == report['baseline']['cost']


def test_semantic_budget_exhaustion_does_not_become_a_proof():
    from strata.semantic import analyze_window, equivalent_windows

    original, _ = hidden_materializations(53)
    lines = str(original).splitlines()
    result = analyze_window(lines, max_operations=2)
    assert result['status'] != 'exact'
    result = equivalent_windows(lines, lines, max_operations=2)
    assert result['equivalent'] is not True


def test_scoped_condition_body_is_not_treated_as_unconditional():
    from strata.file_optimizer import optimize_circuit
    from strata.semantic import lift_regions

    original = km.Circuit('PUSH_CONDITION if b0\nCCX q0 q1 q2\n'
                          'CX q2 q3\nCCX q0 q1 q2\nPOP_CONDITION')
    candidate, _ = optimize_circuit(original)
    assert str(candidate) == str(original)
    result = lift_regions(original)
    assert result['unknown'] == []
    assert len(result['regions']) == 1
    assert result['regions'][0]['outputs']['3']['monomials'] == [[3], [2, 'b0'], [0, 1, 'b0']]


def test_unknown_file_does_not_imply_zero_initialized_intermediate():
    from strata.semantic import equivalent_windows

    original = km.Circuit('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    clean_only_replacement = km.Circuit('CCX q0 q1 q3')
    check = equivalent_windows(str(original).splitlines(), str(clean_only_replacement).splitlines())
    assert check['equivalent'] is False
    _, before, phase_before = native_state(original, 4)
    _, after, phase_after = native_state(clean_only_replacement, 4)
    assert np.any(before != after)
    assert phase_before == phase_after


def test_mixed_native_gate_programs_match_semantics_and_any_emitted_rewrite():
    from strata.file_optimizer import optimize_circuit
    from strata.semantic import analyze_window

    arities = {'X': 1, 'CX': 2, 'CCX': 3, 'SWAP': 2,
               'Z': 1, 'CZ': 2, 'CCZ': 3, 'NEG': 0}
    for seed in range(24):
        rng = random.Random(1000 + seed)
        lines = []
        for _ in range(18):
            gate = rng.choice(list(arities))
            operands = rng.sample(range(6), arities[gate])
            lines.append(' '.join([gate, *(f'q{q}' for q in operands)]))
        original = km.Circuit('\n'.join(lines))
        semantics = analyze_window(lines)
        assert semantics['status'] == 'exact', seed
        initial, actual, phase = native_state(original)
        for q, polynomial in semantics['outputs'].items():
            assert np.array_equal(eval_polynomial(polynomial, initial), actual[int(q)]), seed
        assert tuple(Fraction(int(v)) for v in eval_polynomial(semantics['phase'], initial)) == phase, seed
        candidate, report = optimize_circuit(original)
        _, rewritten, rewritten_phase = native_state(candidate, original.num_qubits)
        assert np.array_equal(actual, rewritten), seed
        assert phase == rewritten_phase, seed
        assert report['candidate']['cost'] <= report['baseline']['cost'], seed


def test_local_cancellation_does_not_invent_a_consumed_materialization():
    from strata.file_optimizer import optimize_circuit

    original = km.Circuit('CCX q0 q1 q2\nCCX q0 q1 q2')
    candidate, report = optimize_circuit(original)
    assert not any(region.get('removed_materializations') for region in report['regions'])
    _, before, before_phase = native_state(original)
    _, after, after_phase = native_state(candidate, original.num_qubits)
    assert np.array_equal(before, after)
    assert before_phase == after_phase


def test_canceled_consumers_do_not_qualify_as_a_structural_gain():
    from strata.file_optimizer import optimize_circuit

    original = km.Circuit('CCX q0 q1 q2\nCX q2 q3\nCX q2 q3\nCCX q0 q1 q2\nX q4')
    _, report = optimize_circuit(original)
    assert not any(region['status'] == 'accepted' and region['structural_change']
                   for region in report['regions'])


@pytest.mark.parametrize('mutation', ['output', 'dirty_helper', 'sign_phase'])
def test_artifact_rehashed_mutations_fail_fresh_equivalence_replay(mutation):
    from strata.cost import Formula
    from strata.file_optimizer import optimize_circuit, verify_optimization
    from strata.kickmix import Kickmix

    original, wires = hidden_materializations(313)
    candidate, report = optimize_circuit(original)
    gate = {'output': f'X q{wires[6]}', 'dirty_helper': f'X q{wires[4]}',
            'sign_phase': f'Z q{wires[0]}'}[mutation]
    mutant = km.Circuit(str(candidate) + '\n' + gate)
    forged = copy.deepcopy(report)
    forged['checks']['equivalent'] = True
    forged['candidate']['sha256'] = hashlib.sha256(str(mutant).encode()).hexdigest()
    forged['candidate']['resources'] = Kickmix().measured(mutant)
    forged['candidate']['cost'] = Formula(forged['objective']['expression'])(forged['candidate']['resources'])
    accepted = [region for region in forged['regions'] if region['status'] == 'accepted']
    assert len(accepted) == 1
    region = accepted[0]
    region['candidate_stop'] = len(str(mutant).splitlines())
    replacement = '\n'.join(str(mutant).splitlines()[region['candidate_start']:])
    region['replacement_sha256'] = hashlib.sha256(replacement.encode()).hexdigest()
    region['after'].update(resources=forged['candidate']['resources'],
                           cost=forged['candidate']['cost'],
                           region_operations=region['candidate_stop'] - region['candidate_start'])
    replay = verify_optimization(original, mutant, forged)
    assert replay['equivalent'] is False
    assert replay['reason'] == 'region_equivalence_not_established'
