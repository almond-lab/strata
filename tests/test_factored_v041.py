import itertools
import kickmix as km
import numpy as np

from strata.factored_semantic import analyze_factored, equivalent_factored, lift_lookup_table
from strata.semantic import analyze_complete, analyze_window, equivalent_windows


def evaluate(receipt, root, assignment):
    graph=receipt['factored']; values=[0,1]
    for variable,low,high in graph['nodes']:
        values.append(values[high] if assignment.get(variable,0) else values[low])
    return values[graph['roots'][root]]


def zero_test(width):
    lines=['SWAP q0 q1 if b0']+[f'X q{q}' for q in range(width)]
    cells=[(0,1,width)]
    cells += [(width+i-2,i,width+i-1) for i in range(2,width)]
    for a,b,target in cells: lines.append(f'CCX q{a} q{b} q{target}')
    lines.append(f'Z q{2*width-2}')
    for i,(a,b,target) in reversed(list(enumerate(cells))):
        lines += [f'HMR q{target} b{i+1}',f'CZ q{a} q{b} if b{i+1}']
    lines += [f'X q{q}' for q in range(width)]+['SWAP q0 q1 if b0']
    return '\n'.join(lines),list(range(width,2*width-1))


def test_zero_test_factored_phase_clean_return_and_mutation():
    text,clean=zero_test(13)
    sem=analyze_complete(text,clean_inputs=clean,max_operations=500,max_variables=100,max_terms=4096,max_product_terms=65536)
    assert sem['status']=='exact_factored',sem
    assert sem['modified_wires']==[]
    assert sem['effects']['clean_outputs']==clean
    assert all(d['status']=='discharged' for d in sem['effects']['phase_debts'])
    for value in [0,1,2,4095,8191]:
        for branch in [0,1]:
            assignment={q:(value>>q)&1 for q in range(13)}|{'b0':branch}|{f'm{i}':branch for i in range(12)}
            assert evaluate(sem,'phase',assignment)==int(value==0)
    broken=text.replace('CZ q0 q1 if b1','Z q0 if b1')
    assert equivalent_factored(text,broken,clean_inputs=clean)['equivalent'] is False
    assert analyze_factored('R q13\n'+text)['reason']=='reset_not_proven_zero'


def lookup_fixture(address=5,targets=4):
    workspace=address+targets
    lines=[]
    for a,b in itertools.combinations(range(address),2):
        lines += [f'R q{workspace}',f'CCX q{a} q{b} q{workspace}']
        lines += [f'CX q{workspace} q{address+t}' for t in range(targets) if (a+b+t)%2==0]
        lines += [f'HMR q{workspace} b0',f'CZ q{a} q{b} if b0']
    return '\n'.join(lines),[workspace]


def test_full_table_independence_phase_and_workspace_proved():
    text,clean=lookup_fixture()
    result=lift_lookup_table(km.Circuit(text),clean_inputs=clean)
    assert result['status']=='exact',result
    assert result['native_replay']['status']=='passed'
    assert result['proof']==dict(dirty_target_independence=True,workspace_restored=True,phase_zero=True,
                                all_measurement_outcomes=True,all_other_inputs_arbitrary=True)
    assert len(result['table'])==32
    assert result['target_wires']==[5,6,7,8]
    broken=text.replace('CZ q0 q1 if b0','Z q0 if b0',1)
    assert 'phase' in lift_lookup_table(km.Circuit(broken),clean_inputs=clean)['reason']
    assert 'restored' in lift_lookup_table(km.Circuit(text+'\nX q9'),clean_inputs=clean)['reason']
    assert 'dirty targets' in lift_lookup_table(km.Circuit(text+'\nCX q5 q6'),clean_inputs=clean)['reason']
    assert lift_lookup_table(km.Circuit(text))['semantics']['reason']=='reset_not_proven_zero'


def test_factored_decisions_match_anf_for_every_input_and_outcome():
    text='SWAP q0 q1 if b1\nCCX q0 q1 q3\nCX q3 q2\nHMR q3 b0\nCZ q0 q1 if b0\nSWAP q0 q1 if b1'
    exact=analyze_window(text,clean_inputs=[3]); factored=analyze_factored(text,clean_inputs=[3])
    variables=[0,1,2,'b0','b1','m0']
    for bits in itertools.product((0,1),repeat=len(variables)):
        assignment=dict(zip(variables,bits))
        for root,poly in [(f'q{q}',p) for q,p in exact['outputs'].items()]+[('phase',exact['phase'])]:
            expected=sum(all(assignment.get(v,0) for v in term) for term in poly['monomials'])%2
            assert evaluate(factored,root,assignment)==expected
    assert equivalent_factored(text,text+'\nX q2\nX q2',clean_inputs=[3])['equivalent'] is True
    assert equivalent_factored('HMR q0 b0','HMR q1 b0')['reason']=='measurement_event_alignment_unproved'


def test_bounded_factored_receipt_reports_exact_instruction():
    text,clean=zero_test(13)
    result=analyze_factored(text,clean_inputs=clean,max_nodes=64)
    assert result['status']=='bounded'
    assert result['reason']=='decision_node_budget'
    assert isinstance(result['at'],int) and result['instruction']


def test_large_full_table_has_all_addresses_and_arbitrary_wide_targets():
    text,clean=lookup_fixture(11,40)
    # Repeated parity basis transformations are charged and exactly restored.
    prefix='\n'.join(['CX q0 q1','CX q1 q2','CX q1 q2','CX q0 q1']*1000)
    result=lift_lookup_table(km.Circuit(prefix+'\n'+text),clean_inputs=clean,max_operations=10000)
    assert result['status']=='exact',result
    assert len(result['table'])==2048 and len(result['target_wires'])==40
    assert result['native_replay']['status']=='passed'


def test_factored_random_programs_against_independent_native_forced_branches():
    import random
    from test_semantic_effects import forced_native
    rng=random.Random(18441)
    for case in range(10):
        lines=['PUSH_CONDITION if b0','BIT_INVERT b0','PUSH_CONDITION if b1']
        for _ in range(9):
            op=rng.choice(['X','CX','CCX','SWAP','CSWAP','Z','CZ','CCZ','BIT_INVERT'])
            if op=='BIT_INVERT': line=f'BIT_INVERT b{rng.randrange(2)}'
            else:
                arity={'X':1,'CX':2,'CCX':3,'SWAP':2,'CSWAP':3,'Z':1,'CZ':2,'CCZ':3}[op]
                line=op+' '+' '.join(f'q{q}' for q in rng.sample(range(4),arity))
            if rng.randrange(2): line+=f' if b{rng.randrange(2)}'
            lines.append(line)
        lines += ['HMR q3 b1 if b0','POP_CONDITION','BIT_STORE1 b0','POP_CONDITION',
                  'HMR q2 b0','CZ q0 q1 if b0','BIT_INVERT b1 if b0']
        text='\n'.join(lines)
        sem=analyze_factored(text)
        anf=analyze_window(text,max_variables=64,max_terms=100000,max_product_terms=1000000)
        assert sem['status']=='exact_factored' and anf['status']=='exact'
        variables=[0,1,2,3,'b0','b1'];inputs=list(itertools.product((0,1),repeat=6))
        for outcomes in itertools.product((0,1),repeat=2):
            sim=km.Simulator(batch_size=len(inputs),seed=71)
            for index,q in enumerate(variables):
                wire=km.q(q) if isinstance(q,int) else km.b(int(q[1:]))
                sim.write_across_shots(wire,np.array([row[index] for row in inputs],dtype=bool))
            sim.do(forced_native(text,outcomes))
            assignments=[dict(zip(variables,row))|dict(zip(['m0','m1'],outcomes)) for row in inputs]
            for q in range(4):
                assert sim.read_across_shots(km.q(q)).astype(int).tolist()==[evaluate(sem,f'q{q}',a) for a in assignments]
            for bit in range(2):
                assert sim.read_across_shots(km.b(bit)).astype(int).tolist()==[evaluate(sem,f'b{bit}',a) for a in assignments]
            assert sim.read_phase_flipped_across_shots().astype(int).tolist()==[evaluate(sem,'phase',a) for a in assignments]
            for a in assignments:
                expected=sum(all(a[v] for v in term) for term in anf['phase']['monomials'])%2
                assert evaluate(sem,'phase',a)==expected


def test_lookup_every_hmr_branch_with_every_dirty_target_input():
    from test_semantic_effects import forced_native
    text,clean=lookup_fixture(3,2)
    result=lift_lookup_table(km.Circuit(text),clean_inputs=clean)
    assert result['status']=='exact'
    for branch in itertools.product((0,1),repeat=3):
        inputs=list(itertools.product((0,1),repeat=5));sim=km.Simulator(batch_size=32,seed=12)
        for q in range(5):sim.write_across_shots(km.q(q),np.array([row[q] for row in inputs],dtype=bool))
        sim.do(forced_native(text,branch))
        for bit,q in enumerate(result['target_wires']):
            expected=[row[q]^((result['table'][sum(row[i]<<i for i in range(3))]>>bit)&1) for row in inputs]
            assert sim.read_across_shots(km.q(q)).astype(int).tolist()==expected
        assert not sim.read_across_shots(km.q(clean[0])).any()
        assert not sim.read_phase_flipped_across_shots().any()


def test_factored_invalid_scope_and_depth_limits_fail_closed():
    assert analyze_factored('PUSH_CONDITION if b0\nX q0')['reason']=='unclosed_condition_scope'
    assert equivalent_factored('POP_CONDITION\nX q0','X q0')['equivalent'] is True
    assert analyze_factored('\n'.join(f'X q{q}' for q in range(600)),max_variables=1024)['reason']=='decision_depth_budget'


def test_complete_lookup_table_reaches_constructor_and_checks_full_boundary(tmp_path):
    from strata.restructure import ConstructorRegistry,restructure_file
    text,clean=zero_test(8)
    text=text.replace('Z q14','CX q14 q16')
    source=tmp_path/'loader.kmx';source.write_text(text)
    registry=ConstructorRegistry();seen=[]
    def propose(circuit,context):
        table=context['lookup_table'];seen.append(table)
        assert context['semantics']['status']=='exact_factored'
        assert table['status']=='exact' and len(table['table'])==256
        assert table['target_wires']==[16]
        return dict(status='constructed',candidate=str(circuit)+'\nX q16\nX q16')
    registry.register('table_consumer',propose,'complete-table propagation regression')
    report=restructure_file(source,tmp_path/'run',registry=registry,clean_inputs=clean,
        rounds=1,max_operations=300,max_variables=64,max_terms=4096,max_product_terms=64)
    assert seen and report['attempts'][0]['check']['equivalent'] is True
    assert report['lookup_tables'][report['baseline']['id']]['proof']['all_measurement_outcomes'] is True


def test_complete_factored_receipt_survives_inspection_and_floor_queries():
    from strata.semantic import lift_regions
    from strata.floors import circuit_floors
    text,clean=zero_test(13)
    lifted=lift_regions(text,clean_inputs=clean,max_window=500,max_variables=100,max_terms=4096)
    assert any(row['semantics']['status']=='exact_factored' for row in lifted['factored_regions'])
    floor=circuit_floors(km.Circuit(text),clean_inputs=clean,max_operations=500,max_variables=100,max_terms=4096)
    assert floor['semantic_status']=='exact_factored' and floor['floor_established'] is False


def test_gc_pins_first_circuit_roots_during_second_execution():
    # Large, varied address functions allocate over 1024 nodes and trigger GC
    # after 128 instructions; retained first-circuit roots must never be reused.
    text,clean=lookup_fixture(11,40)
    same=equivalent_factored(text,text+'\nX q11\nX q11',clean_inputs=clean,max_operations=5000,
                            max_nodes=20000,max_apply_steps=2000000)
    assert same['equivalent'] is True,same
    for mutation in ['X q11','Z q11']:
        result=equivalent_factored(text,text+'\n'+mutation,clean_inputs=clean,max_operations=5000,
                                  max_nodes=20000,max_apply_steps=2000000)
        assert result['equivalent'] is False,result
        assert result['counterexample']['before']!=result['counterexample']['after']


def test_native_lookup_replay_bounds_sparse_physical_ids_before_allocating(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('native simulator must not allocate outside the replay budget')
    monkeypatch.setattr(km,'Simulator',forbidden)
    for source in ['X q1000000','X q0\nBIT_STORE0 b1000000']:
        result=lift_lookup_table(km.Circuit(source))
        assert result['status']=='bounded' and result['reason']=='native_replay_cell_budget'
        assert result['requested_cells']>result['limit']
        assert result['semantics']['status']=='exact_factored'
