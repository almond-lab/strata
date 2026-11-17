import kickmix as km
from strata.inference import is_function_of,information_slack


def test_quadratic_current_support_found_with_exact_clean_promise():
    c=km.Circuit('CCX q0 q1 q2')
    r=is_function_of(c,2,1,[0,1],clean_inputs=[2])
    assert r['status']=='yes' and r['form']['monomials']==[[0,1]]
    assert is_function_of(c,2,1,[0,1])['status']=='no'


def test_search_uses_current_values_not_input_indices():
    c=km.Circuit('CX q0 q1\nCCX q0 q1 q2')
    r=is_function_of(c,2,2,[0,1],clean_inputs=[2])
    assert r['status']=='yes' and r['form']['monomials']==[[0,1]]


def test_no_is_scoped_and_budget_is_unknown():
    c=km.Circuit('CCX q0 q1 q2\nX q3')
    r=is_function_of(c,3,2,[0,1,2])
    assert r['status']=='no' and r['searched_family']['max_degree']==2
    assert is_function_of(c,3,2,[0,1,2],max_forms=1)['status']=='unknown'
    reset = is_function_of(km.Circuit('HMR q0 b0'),0,1,[])
    assert reset['status']=='yes' and reset['form']['monomials']==[]


def test_slack_rank_and_nonlinear_distinct_states_are_not_clean_leases():
    linear=km.Circuit('CX q0 q1')
    r=information_slack(linear,clean_inputs=[1])
    assert r['information_bits']==1 and r['slack_bits']==1
    nonlinear=km.Circuit('CCX q0 q1 q2')
    r=information_slack(nonlinear,clean_inputs=[2])
    assert r['information_bits']==2 and r['distinct_states']==4 and r['slack_bits']==1
    assert information_slack(nonlinear)['slack_bits']==0


def test_public_queries_recover_contents_from_stored_design(tmp_path):
    from pathlib import Path
    from strata import open as open_project
    from strata.queries import Queries
    example=Path(__file__).parents[1]/'examples/and_fusion'
    for name in ('strata.toml','base.kmx'):
        (tmp_path/name).write_bytes((example/name).read_bytes())
    project=open_project(tmp_path/'strata.toml')
    design=project.init()['and']
    queries=Queries(project)
    relation=queries.is_function_of(9,1,[0,3],design=design)
    assert relation['status']=='yes' and relation['form']['monomials']==[[0,3]]
    holders=queries.holders(design,1)
    helper=next(row for row in holders['holders'] if row['wire']==9)
    assert helper['polynomial']==[[0,3]] and helper['depends_on']==['a[0]','b[0]']
    assert 'unknown' in holders['allocation_ownership']
    slack=queries.slack(design,1)
    assert slack['information_bits']==9 and slack['slack_bits']==3
