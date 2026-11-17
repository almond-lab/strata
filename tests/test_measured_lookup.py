import itertools
import random

import kickmix as km
import numpy as np

from strata.lookup import _factor_quadratic, resynthesize_measured_lookup
from strata.restructure import ConstructorRegistry, restructure_file
from strata.semantic import analyze_window, equivalent_windows


LOADER='''REGISTER r0 "address"
APPEND_TO_REGISTER q0 r0
APPEND_TO_REGISTER q1 r0
APPEND_TO_REGISTER q2 r0
R q4
CCX q0 q1 q4
CCX q0 q2 q4
CCX q1 q2 q4
CX q4 q3
HMR q4 b0
CZ q0 q1 if b0
CZ q0 q2 if b0
CZ q1 q2 if b0
R q4
CCX q0 q1 q4
CCX q0 q2 q4
CX q4 q3
HMR q4 b1
CZ q0 q1 if b1
CZ q0 q2 if b1
'''


def _forced(text,outcomes):
    rows=[];index=0;fresh=km.Circuit(text).num_qubits
    for line in str(km.Circuit(text)).splitlines():
        words=line.split()
        if words[0]!='HMR':rows.append(line);continue
        outcome=outcomes[index]
        if outcome:rows.append(f'Z {words[1]}')
        rows.extend([f'SWAP {words[1]} q{fresh}',f'BIT_STORE{outcome} {words[2]}'])
        fresh+=1;index+=1
    return km.Circuit('\n'.join(rows))


def _native(text,outcomes):
    # All address values, both initial dirty-output values and classical bits.
    inputs=list(itertools.product((False,True),repeat=6))
    sim=km.Simulator(batch_size=len(inputs),seed=57)
    for q in range(4):sim.write_across_shots(km.q(q),np.array([x[q] for x in inputs]))
    for b in range(2):sim.write_across_shots(km.b(b),np.array([x[4+b] for x in inputs]))
    sim.do(_forced(text,outcomes))
    return ([sim.read_across_shots(km.q(q)).tolist() for q in range(5)],
            [sim.read_across_shots(km.b(b)).tolist() for b in range(2)],
            sim.read_phase_flipped_across_shots().tolist())


def test_quadratic_factorization_against_independent_truth_tables():
    rng=random.Random(931)
    for _ in range(80):
        terms=[list(t) for size in range(3) for t in itertools.combinations(range(6),size) if rng.randrange(2)]
        factors,linear=_factor_quadratic(terms)
        for bits in itertools.product((0,1),repeat=6):
            expected=sum(all(bits[i] for i in t) for t in terms)%2
            value=sum(all(bits[i] for i in t) for t in linear)%2
            for i,j,left,right in factors:
                value^=((bits[i]+sum(bits[k] for k in left))%2)&((bits[j]+sum(bits[k] for k in right))%2)
            assert value==expected


def test_measured_lookup_restructures_table_and_preserves_all_outcomes(tmp_path):
    original=km.Circuit(LOADER)
    result=resynthesize_measured_lookup(original,clean_inputs=[4])
    assert result['status']=='constructed',result
    before=analyze_window(LOADER,clean_inputs=[4])
    after=analyze_window(result['candidate'],clean_inputs=[4])
    assert all(d['status']=='discharged' for d in after['effects']['phase_debts'])
    assert after['outputs']['4']['monomials']==[]
    assert len(before['measurement_events'])==len(after['measurement_events'])==2
    assert sum(line.startswith('CCX ') for line in result['candidate'].splitlines())==2
    for outcomes in itertools.product((0,1),repeat=2):
        assert _native(LOADER,outcomes)==_native(result['candidate'],outcomes)
    source=tmp_path/'loader.kmx';source.write_text(LOADER)
    registry=ConstructorRegistry()
    registry.register('measured_lookup',lambda c,ctx:resynthesize_measured_lookup(c,**{k:ctx[k] for k in
        ('clean_inputs','max_operations','max_variables','max_terms')}),'Measured lookup regression')
    report=restructure_file(source,tmp_path/'run',registry=registry,clean_inputs=[4],rounds=2)
    assert report['status']=='verified_improvement'
    assert report['baseline']['resources']['toffoli']==5
    assert report['best']['resources']['toffoli']==2
    assert report['best']['resources']['Q']==5
    assert report['cost_delta']==-15
    assert report['official_verifier_run']is False


def test_measured_lookup_preserves_unresolved_phase_and_requires_clean_promise():
    assert resynthesize_measured_lookup(km.Circuit(LOADER))['status']=='unknown'
    broken=LOADER.replace('CZ q0 q1 if b0\n','')
    result=resynthesize_measured_lookup(km.Circuit(broken),clean_inputs=[4])
    assert result['status']=='constructed',result
    assert result['check']['equivalent'] is True
    assert any(d['status']=='outstanding' for d in result['phase_preservation']['original_phase_debts'])
    assert analyze_window(result['candidate'],clean_inputs=[4])['phase']==analyze_window(broken,clean_inputs=[4])['phase']
    for outcomes in itertools.product((0,1),repeat=2):
        assert _native(broken,outcomes)==_native(result['candidate'],outcomes)
    fixed=resynthesize_measured_lookup(km.Circuit(LOADER),clean_inputs=[4])['candidate']
    assert equivalent_windows(LOADER,fixed.replace('CZ q0 q1 if b0\n',''),clean_inputs=[4])['equivalent']is False


def test_measured_lookup_preserves_nonzero_external_phase():
    original=LOADER+'Z q0\n'
    result=resynthesize_measured_lookup(km.Circuit(original),clean_inputs=[4])
    assert result['status']=='constructed',result
    assert analyze_window(result['candidate'],clean_inputs=[4])['phase']['monomials']==[[0]]
    for outcomes in itertools.product((0,1),repeat=2):
        assert _native(original,outcomes)==_native(result['candidate'],outcomes)


def test_measured_lookup_conditional_writes_do_not_create_clean_leases():
    circuit=km.Circuit(LOADER+'PUSH_CONDITION if b0\nX q4\nX q4\nPOP_CONDITION\n')
    result=resynthesize_measured_lookup(circuit,clean_inputs=[4])
    assert result['status']=='constructed'
    assert 'PUSH_CONDITION if b0' in result['candidate']
    assert result['check']['equivalent']is True
