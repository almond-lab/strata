from pathlib import Path

import kickmix as km
from strata.discovery_yield import nonlinear_function_key
from strata.restructure import restructure_file


def test_algebraic_diversity_ignores_free_adapters_helpers_and_repeated_products():
    a = km.Circuit('CCX q0 q1 q2\nCCX q2 q3 q4\nCCX q0 q1 q2')
    b = km.Circuit('X q5\nX q5\nCCX q1 q0 q6\nCCX q6 q3 q4\nCCX q1 q0 q6')
    c = km.Circuit('CCX q0 q3 q2\nCCX q2 q1 q4\nCCX q0 q3 q2')
    assert nonlinear_function_key(a,[2]) == nonlinear_function_key(b,[6])
    assert nonlinear_function_key(a,[2]) != nonlinear_function_key(c,[2])
    assert nonlinear_function_key(km.Circuit('HMR q1 b0'),[]) is None
    assert nonlinear_function_key(km.Circuit('CCZ q0 q1 q3'),[]) != nonlinear_function_key(c,[2])


def test_search_retains_and_proves_distinct_lower_cost_architectures(tmp_path):
    source = tmp_path/'source.kmx'
    # Three arbitrary target bits consume overlapping functions of four inputs.
    polys = [[(1,3),(0,1,3),(1,2,3)],[(0,1),(1,2),(0,1,2)],[(0,2),(0,3),(0,1,3)]]
    lines = ['APPEND_TO_REGISTER q9 r0','REGISTER r0 "capacity"']
    for target,poly in enumerate(polys,4):
        for a,b,*rest in poly:
            if rest:
                lines.extend([f'CCX q{a} q{b} q7',f'CCX q7 q{rest[0]} q{target}',f'CCX q{a} q{b} q7'])
            else:
                lines.append(f'CCX q{a} q{b} q{target}')
    source.write_text('\n'.join(lines))
    result = restructure_file(source, tmp_path/'results', clean_inputs=[7,8,9], deep_only=True,fast=True,
                              max_architectures=128,max_candidates=5,max_finalists=12,rounds=1)
    discovery = result['discovery_yield']
    trial, = discovery['architecture_trials']
    assert trial['unique_formulations'] == 128
    assert trial['unique_product_graphs'] > 1
    assert trial['unique_formulations'] >= trial['verified'] >= 2
    assert discovery['unique_verified_cheaper_native_designs'] >= 2
    assert len(discovery['designs']) == discovery['unique_verified_cheaper_native_designs']
    assert discovery['distinct_nonlinear_function_sets'] > 0
    assert discovery['pareto_resource_points']
    assert all(row['cost'] < result['baseline']['cost'] for row in discovery['designs'])
    assert all(Path(tmp_path/'results'/f"{row['id']}.kmx").is_file() for row in discovery['designs'])
    assert all(state['check']['equivalent'] for state in result['states'] if 'check' in state)


def test_search_routes_in_place_update_to_affine_transport_on_first_attempt(tmp_path):
    source=tmp_path/'transport.kmx'
    source.write_text('CCX q0 q2 q3\nCCX q1 q2 q3\nCX q3 q0')
    result=restructure_file(source,tmp_path/'out',deep_only=True,fast=True,rounds=1,max_candidates=1)
    assert result['best']['mechanism']=='affine_transport_factorization'
    assert result['best']['resources']['toffoli']==1
    assert result['discovery_yield']['designs_by_qualification']['affine_transport_factored']>=1


def test_unqualified_verified_saving_is_visible_without_a_structural_discovery():
    from strata.discovery_yield import summarize

    # These are the independently verified states handed to the yield reporter.
    source = dict(id='source', cost=7, path='original.kmx')
    cheaper = dict(id='cheaper', cost=3, path='smaller.kmx')
    receipt = summarize([source, cheaper], source, {}, 0)
    assert receipt['unique_verified_cheaper_native_designs'] == 0
    assert receipt['verified_cheaper_state_count'] == 1
    assert receipt['cheapest_verified_state'] == cheaper
    unchanged = summarize([source], source, {}, 0)
    assert unchanged['verified_cheaper_state_count'] == 0
    assert unchanged['cheapest_verified_state'] is None
