"""Unique search-spec accounting and independently checked native finalists."""
from itertools import combinations
from math import comb

import numpy as np
import pytest

from strata.candidate_search import (atom_catalog,combination_rank,unrank_combinations,
    screen_coefficients,coefficient_code,generated_contract,search_candidates)
from strata.model import StrataError
from strata.tensor import tensor_from_anf


def tensor():
    return tensor_from_anf([[[0,2],[1,3]],[[0,3],[1,2],[1,3]]],
                           [0,1],[2,3],output_wires=[4,5])


def test_combinadic_candidate_identity_has_no_permutation_or_cancellation_duplicates():
    expected=list(combinations(range(10),4))
    indices=[combination_rank(spec) for spec in expected]
    assert set(indices)==set(range(comb(10,4)))
    recovered=unrank_combinations(indices,10)
    assert recovered.tolist()==[list(spec) for spec in expected]
    assert np.all(np.diff(recovered,axis=1)>0)


def test_guided_matches_exhaustive_same_family_and_native_counters_are_distinct():
    t=tensor();atoms,codes=atom_catalog(t['shape']);family=comb(len(atoms),4)
    specs=unrank_combinations(np.arange(family),len(atoms))
    oracle=[i for i,spec in enumerate(specs) if
            int(codes[spec[0]])^int(codes[spec[1]])^int(codes[spec[2]])^int(codes[spec[3]])==coefficient_code(t)]
    assert np.flatnonzero(screen_coefficients(specs,codes,coefficient_code(t))).tolist()==oracle
    random=search_candidates(t,strategy='random',candidate_budget=family,native_limit=2,materialization_limit=4)
    guided=search_candidates(t,strategy='guided',candidate_budget=family,native_limit=2,materialization_limit=4)
    assert random['counts']['unique_complete_specifications_screened']==family
    assert random['counts']['exact_coefficient_survivors']==len(oracle)
    assert guided['counts']['unique_complete_specifications_screened']==len(oracle)
    assert guided['counts']['exact_coefficient_survivors']==len(oracle)
    assert guided['counts']['partial_search']['pair_records']==comb(len(atoms),2)
    assert guided['counts']['partial_search']['complete_family_exhausted'] is True
    for result in [random,guided]:
        counts=result['counts']
        assert counts['native_candidates_checked']==counts['final_exact_proofs']==2
        assert counts['native_input_shots']==2*64
        assert counts['materialized_program_duplicates']==0
        assert result['native_validation']['status']=='verified'
        assert result['forced_measurement_events']==0
        assert len({row['artifact_sha256'] for row in result['verified_candidates']})==2
        assert all(row['exact_verification']['equivalent'] for row in result['verified_candidates'])


def test_sampling_budget_is_seeded_uniform_without_replacement():
    t=tensor()
    a=search_candidates(t,strategy='random',candidate_budget=15000,seed=511,native_limit=1)
    b=search_candidates(t,strategy='random',candidate_budget=15000,seed=511,native_limit=1)
    assert a['counts']==b['counts']
    assert a['selection']=='uniform_without_replacement'
    assert [r['candidate_index'] for r in a['verified_candidates']]==[r['candidate_index'] for r in b['verified_candidates']]
    assert a['counts']['unique_complete_specifications_screened']==15000


def test_native_phase_corruption_cannot_pass_algebraic_screen(monkeypatch):
    import strata.candidate_search as search
    original=search.lower_decomposition
    def corrupt(tensor,decomposition,**kwargs):
        result=original(tensor,decomposition,**kwargs)
        if kwargs.get('clean_helpers'):
            result['candidate']+='\nZ q0'
        return result
    monkeypatch.setattr(search,'lower_decomposition',corrupt)
    with pytest.raises(StrataError,match='native or exact'):
        search_candidates(tensor(),strategy='guided',candidate_budget=10,native_limit=1)


def test_generic_heldout_partition_uses_recovered_contract_and_explicit_helpers():
    t,metadata=generated_contract((2,4,2),seed=19331)
    assert t['shape']==[2,4,2]
    assert metadata['generating_witness_exposed_to_search'] is False
    result=search_candidates(t,strategy='guided',candidate_budget=4,native_limit=2)
    assert result['counts']['partial_search']['complete_family_exhausted'] is False
    assert result['native_validation']['clean_helper_inputs']==[8,9]
    assert result['counts']['native_input_shots']==512
    assert result['accepted_gain']>=0


def test_catalog_and_helper_bounds_fail_closed():
    with pytest.raises(StrataError,match='budget'):
        atom_catalog([2**40,2**40,2**40])
    with pytest.raises(StrataError,match='clean helpers'):
        search_candidates(tensor(),clean_helpers=[0,6],candidate_budget=1)


def test_proved_floor_prunes_an_impossible_four_product_family():
    t=tensor_from_anf([[[q,5+q] for q in range(5)]],range(5),range(5,10),output_wires=[10])
    with pytest.raises(StrataError,match='proved tensor lower bound'):
        search_candidates(t,strategy='guided',candidate_budget=10)
