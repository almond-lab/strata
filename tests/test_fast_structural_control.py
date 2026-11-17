from pathlib import Path
import time

from strata.restructure import ConstructorRegistry, restructure_file


def test_expired_trial_keeps_previously_verified_design(tmp_path):
    source=tmp_path/'source.kmx'
    source.write_text('CCX q0 q1 q2\nCCX q0 q1 q2\nX q2')
    registry=ConstructorRegistry()
    registry.register('identity_pair',lambda c,ctx:dict(status='constructed',candidate='X q2'),'verified candidate')
    def delayed(circuit,context):
        time.sleep(5)
        return dict(status='constructed',candidate='Z q0')
    registry.register('slow',delayed,'bounded trial')
    r=restructure_file(source,tmp_path/'out',registry=registry,rounds=1,trial_seconds=.2,cost='instructions')
    assert r['status']=='verified_improvement' and r['search_control']['timed_out']==1
    assert Path(r['best']['path']).read_text().strip()=='X q2'


def test_precancel_and_automatic_screen_are_reported(tmp_path):
    source=tmp_path/'source.kmx'
    source.write_text('CCZ q0 q1 q2\nCCZ q0 q1 q3')
    marker=tmp_path/'STOP';marker.touch()
    paused=restructure_file(source,tmp_path/'paused',deep_only=True,fast=True,cancel_file=marker)
    assert paused['search_control']['stop_reason']=='cancel_requested' and not paused['attempts']
    r=restructure_file(source,tmp_path/'active',deep_only=True,fast=True,rounds=1,trial_seconds=4)
    assert r['best']['cost']<r['baseline']['cost']
    assert r['discovery_yield']['unique_verified_cheaper_native_designs']>=1
    assert r['search_control']['screened_count']>0


def test_class_bit_screen_is_one_sided_and_preserves_all_witnesses():
    from random import Random
    from itertools import combinations
    from strata.algebraic_synthesis import _Budget
    from strata.functional_decomposition import _classes,_screen_table,_third_signature
    rng=Random(819)
    wires=[0,4,5,9,12,13]
    budget=_Budget(1000000,8192,1024,8192,5)
    for _ in range(5):
        outputs=[frozenset(frozenset(t) for k in (1,2,3) for t in combinations(wires,k) if rng.randrange(3)==0) for j in range(2)]
        table=_screen_table(wires,outputs,budget)
        for group in combinations(wires,3):
            if _third_signature(table,group,budget):
                assert _classes(outputs,group,budget) is None


def test_measured_source_screens_quantum_only_families_and_keeps_joint_if_resolved():
    from strata.search_screen import inapplicable
    context=dict(clean_inputs=(5,6),classical_trajectory=True,semantics={'status':'exact'})
    assert inapplicable('architecture_portfolio',{},context)
    assert inapplicable('mixed_observable_search',{},context)
    assert inapplicable('joint_boundary_synthesis',{},context) is None
    context['semantics']=dict(status='opaque',reason='reset_not_proven_zero')
    assert 'reset' in inapplicable('joint_boundary_synthesis',{},context)
