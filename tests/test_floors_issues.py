from strata.floors import counting_bound,exact_minimum


def test_nonlinear_rank_and_degree_are_scoped_bounds():
    r=counting_bound([[[0,1]],[[0,2]]])
    assert r['value']==2 and r['nonlinear_rank']==2 and r['resource']=='and_count'
    assert counting_bound([[[0,1,2,3,4]]])['value']==3
    assert counting_bound([[[0,1],[0,1]]])['value']==0
    assert counting_bound([[[0,0]]])['value']==0


def test_exact_known_small_optima_without_library_answer_hint():
    assert exact_minimum([[[0],[1]]],2)['value']==0
    assert exact_minimum([[[0,1]]],2)['value']==1
    # One-bit full-adder sum and carry; carry=(a^b)*(a^c)^a.
    r=exact_minimum([[[0],[1],[2]],[[0,1],[0,2],[1,2]]],3)
    assert r['kind']=='exact_small' and r['value']==1 and len(r['witness'])==1


def test_search_timeout_does_not_become_false_floor():
    # GF(4) multiplication in the x^2+x+1 polynomial basis.
    outputs=[[[0,2],[1,3]],[[0,3],[1,2],[1,3]]]
    r=exact_minimum(outputs,4,max_expansions=1)
    assert r['kind']=='unknown' and r['value'] is None
    assert r['proven_lower_bound']==3
    assert r['structural_lower_bounds']['quadratic_polar_rank']['value']==3
