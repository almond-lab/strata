from itertools import product
import pytest

from strata.model import StrataError
from strata.tensor import (tensor_from_anf, tensor_bounds, quadratic_bounds,
                           search_decomposition, verify_decomposition, lower_decomposition)


def small_field_tensor():
    # Multiplication modulo u^2+u+1, with distinct arbitrary output accumulators.
    return tensor_from_anf([[[0,2],[1,3]], [[0,3],[1,2],[1,3]]],
                           [0,1],[2,3],output_wires=[4,5])


def test_three_flattening_bounds_and_exact_small_decomposition():
    tensor=small_field_tensor()
    bounds=tensor_bounds(tensor)
    assert bounds['shape']==[2,2,2]
    assert bounds['flattening_ranks']==dict(left=2,right=2,output=2)
    assert bounds['value']==3
    assert bounds['substitution_bound']['value']==3
    assert bounds['substitution_bound']['per_mode']['output']['minimum_nonzero_rank']==2
    result=search_decomposition(tensor,max_rank=3,timeout=5)
    assert result['rank']==3 and result['optimal_in_model']
    assert result['search']['excluded_by_bounds_below']==3
    assert result['verification']['passed']
    assert 'not a general AND' in result['scope']
    corrupt=[dict(term) for term in result['decomposition']]
    corrupt[0]['output_mask']=1 if corrupt[0]['output_mask']!=1 else 2
    assert not verify_decomposition(tensor,corrupt)['passed']


def test_reversible_lowering_preserves_dirty_outputs_and_restores_helpers():
    tensor=small_field_tensor()
    result=search_decomposition(tensor,max_rank=3,timeout=5)
    lowered=lower_decomposition(tensor,result['decomposition'])
    assert lowered['resources']['toffoli']==3
    assert lowered['resources']['clean_helpers']==2
    assert lowered['boundary']['arbitrary_initial_outputs']==[4,5]
    for inputs in product((0,1),repeat=6):
        state=list(inputs)+[0,0]
        for line in lowered['candidate'].splitlines():
            op,*wires=line.split()
            wires=[int(q[1:]) for q in wires]
            if op=='CX':state[wires[1]]^=state[wires[0]]
            elif op=='CCX':state[wires[2]]^=state[wires[0]]&state[wires[1]]
            elif op=='X':state[wires[0]]^=1
            else:raise AssertionError(op)
        a,b,c,d,z,w=inputs
        assert state==[a,b,c,d,z^(a&c)^(b&d),w^(a&d)^(b&c)^(b&d),0,0]
    import kickmix as km
    from strata.semantic import equivalent_windows
    naive='CCX q0 q2 q4\nCCX q1 q3 q4\nCCX q0 q3 q5\nCCX q1 q2 q5\nCCX q1 q3 q5'
    checked=equivalent_windows(str(km.Circuit(naive)),lowered['candidate'],
        clean_inputs=lowered['boundary']['clean_inputs'],max_operations=1000,max_variables=16)
    assert checked['equivalent']


def test_tensor_search_budget_never_promotes_upper_bound_to_floor():
    tensor=small_field_tensor()
    result=search_decomposition(tensor,max_catalog=1)
    assert result['rank']==4 and not result['optimal_in_model']
    assert result['proven_lower_bound']==3
    assert result['evidence_kind']=='verified_construction'
    assert verify_decomposition(tensor,result['decomposition'])['passed']
    result=search_decomposition(tensor,max_expansions=1)
    assert not result['optimal_in_model'] and result['rank']==4


def test_incomplete_substitution_never_uses_an_observed_minimum_as_lower_bound():
    tensor=small_field_tensor()
    bound=tensor_bounds(tensor,max_substitution_combinations=1)
    assert bound['value']==2
    assert bound['substitution_bound']['kind']=='unknown'
    assert bound['substitution_bound']['per_mode']['left']['observed_minimum_rank']==2
    assert bound['substitution_bound']['per_mode']['left']['minimum_nonzero_rank'] is None


def test_quadratic_bound_uses_max_not_sum_for_shared_outputs():
    forms=[[[0,1],[2,3]], [[0,1],[2,3]]]
    result=quadratic_bounds(forms)
    assert result['value']==2
    assert [row['alternating_rank'] for row in result['per_output']]==[4,4]
    assert result['aggregation'].startswith('maximum')


def test_bilinear_validation_affine_lowering_and_wide_bounds():
    with pytest.raises(StrataError,match='not bilinear'):
        tensor_from_anf([[[0,1]]],[0,1],[2])
    with pytest.raises(StrataError,match='bipartition overlaps'):
        tensor_from_anf([[[0,1]]],[0],[0,1])
    with pytest.raises(StrataError,match='degree'):
        quadratic_bounds([[[0,1,2]]])
    tensor=tensor_from_anf([[[0,1],[0],[]]],[0],[1],output_wires=[2])
    result=search_decomposition(tensor)
    lowered=lower_decomposition(tensor,result['decomposition'])
    assert lowered['candidate']=='CCX q0 q1 q2\nX q2\nCX q0 q2'
    assert lowered['resources']['clean_helpers']==0
    wide=tensor_from_anf([[[i,12+i]] for i in range(12)],range(12),range(12,24))
    assert tensor_bounds(wide)['flattening_ranks']==dict(left=12,right=12,output=12)
    assert search_decomposition(wide)['rank']==12


def test_bounds_never_exceed_independent_exact_ranks_for_all_two_by_two_by_two_tensors():
    # Breadth-first addition in the 256-element tensor space is an independent
    # exact oracle, including dependent slices and rank-one/zero edge cases.
    from collections import deque
    atoms=[]
    for x,y,z in product(range(1,4),repeat=3):
        atoms.append(sum(1<<(4*i+2*j+k) for i,j,k in product(range(2),repeat=3)
                         if (x>>i)&(y>>j)&(z>>k)&1))
    distance={0:0};queue=deque([0])
    while queue:
        value=queue.popleft()
        for atom in atoms:
            target=value^atom
            if target not in distance:
                distance[target]=distance[value]+1;queue.append(target)
    assert len(distance)==256
    for encoded,rank in distance.items():
        outputs=[[],[]]
        for i,j,k in product(range(2),repeat=3):
            if encoded>>(4*i+2*j+k)&1:outputs[k].append([i,2+j])
        tensor=tensor_from_anf(outputs,[0,1],[2,3])
        assert tensor_bounds(tensor)['value']<=rank
        construction=search_decomposition(tensor,max_rank=3,timeout=5)
        assert construction['rank']==rank and construction['optimal_in_model']
