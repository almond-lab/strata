"""Gate-derived seeds and local search beyond complete factor catalogs."""
import random

import pytest

from strata.model import StrataError
from strata.tensor import (extract_decomposition, lower_decomposition, search_decomposition,
                           tensor_from_anf, verify_decomposition)


def tensor_for_terms(shape, terms):
    """Independent sparse polynomial expansion, with physical dirty targets."""
    a, b, c = shape
    outputs = [set() for _ in range(c)]
    for term in terms:
        for i in range(a):
            for j in range(b):
                if term['left_mask'] >> i & term['right_mask'] >> j & 1:
                    for k in range(c):
                        if term['output_mask'] >> k & 1:
                            pair = (i, a + j)
                            if pair in outputs[k]:
                                outputs[k].remove(pair)
                            else:
                                outputs[k].add(pair)
    return tensor_from_anf([sorted(poly) for poly in outputs], range(a), range(a, a+b),
                           output_wires=range(a+b, a+b+c))


def term(left, right, output):
    return dict(left_mask=left, right_mask=right, output_mask=output)


def large_seed(shape, count, *, sparse):
    rng = random.Random(8103)
    # Three distinct paired products have a linear dependence, so their output
    # vectors can be redistributed to remove one product without a full catalog.
    terms = [term(7, 3, 1), term(7, 12, 2), term(7, 15, 4)]
    while len(terms) < count:
        masks = [sum(1 << i for i in rng.sample(range(n), min(n, 3 if axis < 2 else 5)))
                 if sparse else rng.randrange(1, 1 << n) for axis, n in enumerate(shape)]
        candidate = term(*masks)
        if candidate not in terms:
            terms.append(candidate)
    return terms


@pytest.mark.parametrize('shape,count,sparse', [((12, 12, 12), 42, False), ((24, 24, 48), 173, True)])
def test_real_size_gate_seed_with_linear_forms_and_propagated_output_fanout(shape, count, sparse):
    seed = large_seed(shape, count, sparse=sparse)
    tensor = tensor_for_terms(shape, seed)
    native = lower_decomposition(tensor, seed)
    extracted = extract_decomposition(native['candidate'], tensor,
                                      clean_inputs=native['boundary']['clean_inputs'])
    assert extracted['status'] == 'verified'
    assert extracted['rank'] == extracted['ccx_count'] == count
    assert extracted['decomposition'] == seed
    assert extracted['boundary_verification']['passed']
    assert any(row['output_mask'].bit_count() > 1 for row in extracted['products'])
    result = search_decomposition(tensor, seed=extracted['decomposition'], max_catalog=1,
                                  max_local_states=1, timeout=5)
    assert result['rank'] < count
    assert result['search']['coordinate_rank'] > count
    assert result['search']['seed_rank'] == count
    assert result['search']['local_search']['reductions'] >= 1
    assert result['search']['local_search']['exact_verifications'] >= 1
    catalog_size = 1
    for dimension in shape:
        catalog_size *= (1 << dimension) - 1
    assert result['search']['catalog_size'] == catalog_size
    assert str(catalog_size) in result['search']['reason']
    assert result['search']['searched_up_to'] is None
    assert verify_decomposition(tensor, result['decomposition'])['passed']
    if not sparse:
        assert result['search']['coordinate_rank'] == 144


def test_shared_factor_flip_opens_a_reduction_beyond_the_catalog_limit():
    seed = [term(1, 2, 3), term(4, 4, 1), term(4, 5, 4), term(5, 2, 1)]
    seed += [term(1 << i, 1 << i, 1 << i) for i in range(3, 12)]
    tensor = tensor_for_terms((12, 12, 12), seed)
    result = search_decomposition(tensor, seed=seed, max_catalog=1, max_local_states=32, timeout=5)
    assert result['rank'] == 12 < len(seed)
    assert result['search']['local_search']['flip_moves'] > 0
    assert result['search']['local_search']['reductions'] > 0
    assert result['verification']['passed']


def test_seed_survives_budget_exhaustion_and_invalid_seed_is_rejected():
    seed = large_seed((12, 12, 12), 42, sparse=False)
    tensor = tensor_for_terms((12, 12, 12), seed)
    result = search_decomposition(tensor, seed=seed, max_catalog=1, max_expansions=1)
    assert result['rank'] == 42
    assert result['search']['coordinate_rank'] == 144
    assert result['search']['local_search']['status'] == 'budget_exhausted'
    assert result['verification']['passed']
    with pytest.raises(StrataError, match='seed does not match'):
        search_decomposition(tensor, seed=seed[:-1])


def test_tensor_constructor_retains_42_product_native_seed_instead_of_144_coordinate_products():
    import kickmix as km
    from strata.restructure import _tensor
    from strata.semantic import analyze_window
    seed=large_seed((12,12,12),45,sparse=False)[3:]
    tensor=tensor_for_terms((12,12,12),seed)
    native=lower_decomposition(tensor,seed)
    circuit=km.Circuit(native['candidate'])
    context=dict(clean_inputs=native['boundary']['clean_inputs'],max_operations=10000,
                 max_variables=128,max_terms=65536)
    context['semantics']=analyze_window(str(circuit),**context)
    assert context['semantics']['status']=='exact'
    result=_tensor(circuit,context)
    assert result['status']=='constructed'
    assert result['seed_extraction']['rank']==42
    assert result['tensor_search']['search']['coordinate_rank']==144
    assert result['tensor_search']['rank']==42
    assert result['candidate']==str(circuit)


def test_in_place_linear_operands_do_not_require_new_clean_helpers():
    import kickmix as km
    from strata.restructure import _tensor
    from strata.semantic import analyze_window
    circuit=km.Circuit('CX q0 q1\nCX q2 q3\nCCX q1 q3 q4\nCX q2 q3\nCX q0 q1')
    context=dict(clean_inputs=[],max_operations=1000,max_variables=32,max_terms=4096)
    context['semantics']=analyze_window(str(circuit),**context)
    result=_tensor(circuit,context)
    assert result['status']=='constructed'
    assert result['seed_extraction']['decomposition']==[term(3,3,1)]
    assert result['candidate']==str(circuit)


def test_affine_control_constants_are_checked_separately_from_tensor_products():
    tensor = tensor_from_anf([[[0, 1], [1]]], [0], [1], output_wires=[2])
    extracted = extract_decomposition('X q0\nCCX q1 q0 q2\nX q0', tensor)
    assert extracted['status'] == 'verified'
    assert extracted['decomposition'] == [term(1, 1, 1)]
    assert extracted['boundary_verification']['passed']


def test_clean_output_accumulator_is_not_claimed_as_restored_workspace():
    tensor=tensor_from_anf([[[0,1]]],[0],[1],output_wires=[2])
    extracted=extract_decomposition('CCX q0 q1 q2',tensor,clean_inputs=[2])
    assert extracted['status']=='verified'
    assert extracted['boundary_verification']['clean_output_accumulators']==[2]
    assert extracted['boundary_verification']['restored_clean_outputs']==[]
    assert extracted['boundary_verification']['arbitrary_initial_outputs']==[]


def test_extraction_propagates_products_through_swaps_and_cancelling_helpers():
    tensor = tensor_from_anf([[[0, 1]], [[0, 1]]], [0], [1], output_wires=[2, 3])
    circuit = 'CCX q0 q1 q4\nCX q4 q2\nSWAP q4 q5\nCX q5 q3\nCCX q1 q0 q5'
    extracted = extract_decomposition(circuit, tensor, clean_inputs=[4, 5])
    assert extracted['status'] == 'verified'
    assert extracted['decomposition'] == [term(1, 1, 3)]
    assert [p['output_mask'] for p in extracted['products']] == [3, 0]
    broken = extract_decomposition(circuit + '\nX q4', tensor, clean_inputs=[4, 5])
    assert broken['status'] == 'not_applicable'
    assert not broken['boundary_verification']['passed']


@pytest.mark.parametrize('suffix,operation', [('CCZ q0 q1 q2', 'CCZ'), ('HMR q2 b0', 'HMR'),
                                            ('X q0 if b0', 'X'), ('R q2', 'R')])
def test_phase_measurement_and_conditional_operations_are_outside_accumulator_scope(suffix, operation):
    tensor = tensor_from_anf([[[0, 1]]], [0], [1], output_wires=[2])
    extracted = extract_decomposition('CCX q0 q1 q2\n' + suffix, tensor)
    assert extracted['status'] == 'not_applicable'
    assert extracted['operation'] == operation
    assert 'excludes CCZ' in extracted['scope']
    assert 'decomposition' not in extracted


def test_nonlinear_controls_and_unrestored_public_inputs_fail_closed():
    tensor = tensor_from_anf([[[0, 1]]], [0], [1], output_wires=[2])
    nonlinear = extract_decomposition('CCX q0 q1 q3\nCCX q3 q1 q2', tensor, clean_inputs=[3])
    assert nonlinear['reason'] == 'CCX has nonlinear controls'
    public = extract_decomposition('CCX q0 q1 q2\nX q1000', tensor)
    assert public['wire'] == 1000 and not public['boundary_verification']['passed']


def test_completed_exclusion_strengthens_lower_bound_without_finding_construction():
    tensor = tensor_from_anf([[[0, 2], [1, 2], [1, 3]], [[0, 2], [0, 3]]], [0, 1], [2, 3])
    result = search_decomposition(tensor, max_rank=2, timeout=5)
    assert result['rank'] == 4 and not result['optimal_in_model']
    assert result['bounds']['value'] == 2
    assert result['search']['searched_up_to'] == 2
    assert result['proven_lower_bound'] == 3
    assert result['evidence_kind'] == result['lower_bound_evidence']['kind'] == 'exact_small'
    assert 'excluded all ranks through 2' in result['search']['reason']

    # The next rank is interrupted. The completed exclusion survives, but the
    # interrupted rank supplies no stronger evidence.
    interrupted = search_decomposition(tensor, max_rank=3, timeout=5,
                                        max_expansions=result['search']['expansions'])
    assert interrupted['search']['searched_up_to'] == 2
    assert interrupted['proven_lower_bound'] == 3 and interrupted['rank'] == 4
    partial = search_decomposition(tensor, max_rank=2, timeout=5, max_expansions=40)
    assert partial['search']['searched_up_to'] is None
    assert partial['proven_lower_bound'] == 2
    assert partial['evidence_kind'] == 'verified_construction'


def test_partial_timeout_does_not_claim_a_completed_exclusion(monkeypatch):
    import strata.tensor as module
    tensor = tensor_from_anf([[[0, 2], [1, 2], [1, 3]], [[0, 2], [0, 3]]], [0, 1], [2, 3])
    ticks = 0

    def clock():
        nonlocal ticks
        ticks += 1
        return ticks / 100

    monkeypatch.setattr(module, 'monotonic', clock)
    result = search_decomposition(tensor, max_rank=2, timeout=.3)
    assert result['search']['searched_up_to'] is None
    assert result['proven_lower_bound'] == 2
    assert not result['optimal_in_model']
