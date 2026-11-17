import kickmix as km

from strata.semantic import analyze_window
from strata.tensor import tensor_from_anf, extract_decomposition
from strata.restructure import _tensor


def test_affine_bilinear_accumulator_retains_affine_and_seeds_constructor():
    text = 'CCX q0 q1 q4\nCX q0 q4\nCX q2 q4\nX q4'
    sem = analyze_window(text)
    feature = next(f for f in sem['features'] if f['kind'] == 'bilinear_accumulator')
    assert feature['factor_partition'] == [[0, 2], [1]]
    tensor = tensor_from_anf(feature['deltas'], *feature['factor_partition'], output_wires=[4])
    assert tensor['affine'] == [[[], [0], [2]]]
    seed = extract_decomposition(km.Circuit(text), tensor)
    assert seed['status'] == 'verified'
    proposal = _tensor(km.Circuit(text), dict(semantics=sem, clean_inputs=[], max_operations=1000, max_variables=32))
    assert proposal['status'] == 'constructed'
    assert proposal['seed_extraction']['status'] == 'verified'


def test_non_bipartite_and_higher_degree_remain_non_bilinear():
    for text in ('CCX q0 q1 q3\nCCX q1 q2 q3\nCCX q0 q2 q3\nX q3',
                 'CCX q0 q1 q4\nCCX q4 q2 q3'):
        sem = analyze_window(text, clean_inputs=[4])
        assert not any(f['kind'] == 'bilinear_accumulator' for f in sem['features'])


def test_product_budget_reports_failing_instruction_and_dimensions():
    text = 'CX q0 q1\nCX q2 q3\nCCX q1 q3 q4'
    result = analyze_window(text, max_product_terms=1)
    assert result['reason'] == 'product_budget'
    assert result['at'] == 2
    assert result['instruction'] == 'CCX q1 q3 q4'
    assert result['product_details'] == dict(left_terms=2, right_terms=2, requested_products=4, products_before=0)
