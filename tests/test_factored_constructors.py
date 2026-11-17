import kickmix as km
import numpy as np

from strata.factored_semantic import constructor_semantics, projected_value_semantics
from strata.lookup import resynthesize_measured_lookup, resynthesize_supported_segments
from strata.semantic import analyze_complete
from test_factored_v041 import zero_test
from test_semantic_effects import forced_native


def native_compare(original, candidate, clean):
    before, after = km.Circuit(original), km.Circuit(candidate)
    assert after.num_qubits == before.num_qubits
    assert after.operation_statistics().num_toffoli_like < before.operation_statistics().num_toffoli_like
    random = np.random.default_rng(190)
    inputs = random.integers(0, 2, size=(before.num_qubits, 32)).astype(bool)
    count = sum(line.startswith('HMR ') for line in str(before).splitlines())
    for branch in ([0] * count, [1] * count, [i % 2 for i in range(count)]):
        sims = [km.Simulator(batch_size=32, seed=11), km.Simulator(batch_size=32, seed=11)]
        for sim, text in zip(sims, [original, candidate]):
            for q in range(before.num_qubits):
                if q not in clean: sim.write_across_shots(km.q(q), inputs[q])
            sim.do(forced_native(text, branch))
        for q in range(before.num_qubits):
            assert np.array_equal(*(sim.read_across_shots(km.q(q)) for sim in sims))
        for b in range(before.num_bits):
            assert np.array_equal(*(sim.read_across_shots(km.b(b)) for sim in sims))
        assert np.array_equal(*(sim.read_phase_flipped_across_shots() for sim in sims))


def test_factored_nonzero_phase_episode_feeds_lookup_constructor():
    prefix, clean = zero_test(13)
    suffix = '''R q25
CCX q0 q1 q25
CCX q0 q2 q25
CCX q1 q2 q25
CX q25 q26
HMR q25 b20
CZ q0 q1 if b20
CZ q0 q2 if b20
CZ q1 q2 if b20
'''
    source = prefix + '\n' + suffix
    clean += [25]
    result = resynthesize_measured_lookup(source, clean_inputs=clean, max_terms=4096)
    assert result['status'] == 'constructed', result
    assert result['semantic_status'] == 'exact_factored'
    assert result['check']['equivalent'] is True
    assert analyze_complete(source, clean_inputs=clean, max_operations=10000,
                            max_variables=128, max_terms=4096)['phase_zero'] is False
    native_compare(source, result['candidate'], clean)


def test_factored_episode_feeds_phase_and_direct_segment_constructors():
    prefix, clean = zero_test(13)
    for mode, consume in [('phase', 'Z q25'), ('direct', 'CX q25 q26')]:
        source = prefix + '\nR q25\nCCX q0 q1 q25\n' + consume + '\nCCX q0 q1 q25'
        result = resynthesize_supported_segments(source, clean_inputs=clean + [25],
                                                 max_terms=4096, mode=mode)
        assert result['status'] == 'constructed', result
        assert result['semantic_status'] == 'exact_factored'
        assert result['check']['equivalent'] is True
        native_compare(source, result['candidate'], clean + [25])


def test_factored_final_projection_cancels_large_intermediate_before_synthesis():
    width = 8
    gates = [f'X q{q}' for q in range(width)]
    gates += ['CCX q0 q1 q8'] + [f'CCX q{width+i-2} q{i} q{width+i-1}' for i in range(2, width)]
    source = '\n'.join(gates + list(reversed(gates)))
    clean = list(range(width, 2 * width - 1))
    semantics = constructor_semantics(source, clean_inputs=clean, max_product_terms=64, max_terms=4096)
    assert semantics['status'] == 'exact', semantics
    assert semantics['source_status'] == 'exact_factored'
    assert semantics['changes'] == {} and semantics['phase']['monomials'] == []
    result = resynthesize_supported_segments(source, clean_inputs=clean, max_product_terms=64, max_terms=4096)
    assert result['status'] == 'constructed', result
    assert any(s['semantic_source'] == 'exact_factored' for s in result['segments'])
    native_compare(source, result['candidate'], clean)


def test_final_value_projection_preserves_measured_factored_effect_receipts():
    source, clean = zero_test(8)
    source = source.replace('Z q14', '')
    result = projected_value_semantics(source, clean_inputs=clean, max_operations=500,
                                        max_variables=100, max_terms=4096, max_product_terms=64)
    assert result['status'] == 'exact', result
    assert result['phase']['monomials'] == []
    assert result['outputs']['0']['monomials'] == [[0]]
    assert all(result['outputs'][str(q)]['monomials'] == [] for q in clean)
    assert result['classical_outputs']['b1']['monomials'] == [['m6']]
    assert len(result['measurement_variables']) == 7
    assert all(debt['status'] == 'discharged' for debt in result['effects']['phase_debts'])
    assert result['factored_semantics']['status'] == 'exact_factored'
    assert result['is_permutation'] is False
