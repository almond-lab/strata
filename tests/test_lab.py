from dataclasses import replace
import pytest
import kickmix as km
from strata.model import Contract, Register, StrataError
from strata.lab import exhaustive
from strata.demo import AndFusion


def test_both_structures_exhaustive_dirty_accumulator():
    a=AndFusion()
    for bits in (1,2,3):
        for fused in (False,True):
            c,t=a.build({'bits':bits},{'fused':fused})
            r=exhaustive(c,t.contract,a.reference,{'bits':bits})
            assert r.passed and r.exhaustive and r.cases==2**(3*bits)


def hmr_fixture():
    c=Contract('and-clean',(Register('a',(0,)),Register('b',(1,)),Register('s',(2,),role='clean')),'and-clean')
    return c, lambda x,scale: ({'a':x['a'],'b':x['b']},0)


def test_every_measurement_branch_and_phase_mutation():
    contract,reference=hmr_fixture()
    good=km.Circuit('CCX q0 q1 q2\nHMR q2 b0\nCZ q0 q1 if b0')
    r=exhaustive(good,contract,reference)
    assert r.passed and r.cases==4 and r.branches==2 and r.tested==8
    bad=km.Circuit('CCX q0 q1 q2\nHMR q2 b0')
    r=exhaustive(bad,contract,reference)
    assert not r.passed and r.counterexample['hmr_branch']==1
    assert r.counterexample['phase']!='0' and not r.counterexample['values']


def test_dirty_restoration_is_not_optional_reference_field():
    contract=Contract('dirty',(Register('dirty',(0,),role='dirty'),),'id')
    r=exhaustive(km.Circuit('X q0'),contract,lambda x,s: ({'dirty':not x['dirty']},0))
    assert not r.passed


def test_clean_leak_and_spec_mutations():
    contract,reference=hmr_fixture()
    r=exhaustive(km.Circuit('CCX q0 q1 q2'),contract,reference)
    assert not r.passed and 's' in r.counterexample['values']
    badref=lambda x,s: ({'a':x['a']^1,'b':x['b']},0)
    assert not exhaustive(km.Circuit('CX q0 q1\nCX q0 q1\nCX q2 q2') if False else km.Circuit('CCX q0 q1 q2\nCCX q0 q1 q2'),contract,badref).passed


def test_sampling_cannot_be_certified_exhaustive():
    c,t=AndFusion().build({'bits':3},{})
    with pytest.raises(StrataError,match='budget'):
        exhaustive(c,t.contract,AndFusion.reference,max_trajectories=5)
    with pytest.raises(StrataError,match='own every'):
        exhaustive(c,replace(t.contract,registers=t.contract.registers[:-1]),AndFusion.reference)


def test_native_random_hmr_matches_exhaustive():
    import numpy as np
    c=km.Circuit('CCX q0 q1 q2\nHMR q2 b0\nCZ q0 q1 if b0')
    for seed in (1,2,3):
        s=km.Simulator(batch_size=256,seed=seed)
        s.write_across_shots(km.q(0),np.ones(256,dtype=bool))
        s.write_across_shots(km.q(1),np.ones(256,dtype=bool));s.do(c)
        assert not np.any(s.read_phase_flipped_across_shots())
        assert not np.any(s.read_across_shots(km.q(2)))
