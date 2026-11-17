"""Measured search over distinct bounded bilinear decomposition specifications.

The million-scale counter covers canonical algebraic specifications, not native
programs or proofs. Only materialized finalists enter native exhaustive replay
and exact symbolic equivalence. Guided proposals solve a tensor residual;
proposal precision is consequently not an unbiased success probability.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product
from math import comb, prod
from time import perf_counter
import hashlib

import kickmix as km
import numpy as np

from .kickmix import Kickmix
from .model import StrataError, digest
from .semantic import analyze_window, equivalent_windows
from .tensor import tensor_from_anf, tensor_bounds, lower_decomposition, verify_decomposition


def atom_catalog(shape, *, max_atoms=512):
    """Nonzero rank-one factors, ordered left mask, right mask, output mask."""
    if len(shape)!=3 or any(type(n)is not int or n<1 for n in shape):
        raise StrataError('candidate search requires three positive tensor dimensions')
    if prod(shape)>63 or prod((1<<n)-1 for n in shape)>max_atoms:
        raise StrataError('candidate catalog exceeds exact uint64 or atom budget')
    a,b,c=shape
    atoms=[];codes=[]
    for left,right,out in product(range(1,1<<a),range(1,1<<b),range(1,1<<c)):
        code=0
        for i,j,k in product(range(a),range(b),range(c)):
            if (left>>i)&(right>>j)&(out>>k)&1:
                code |= 1<<((i*b+j)*c+k)
        atoms.append(dict(left_mask=left,right_mask=right,output_mask=out));codes.append(code)
    return atoms,np.array(codes,dtype=np.uint64)


def coefficient_code(tensor):
    _,b,c=tensor['shape']
    return sum(1<<((i*b+j)*c+k) for i,j,k in tensor['entries'])


def combination_rank(spec):
    return sum(comb(int(value),index+1) for index,value in enumerate(spec))


def unrank_combinations(ranks, atom_count, rank=4):
    """Vectorized combinadic inverse; every row is a sorted distinct atom set."""
    ranks=np.asarray(ranks,dtype=np.int64)
    if np.any(ranks<0) or np.any(ranks>=comb(atom_count,rank)):
        raise StrataError('candidate index outside combinatorial family')
    residual=ranks.copy();result=np.empty((len(ranks),rank),dtype=np.int64)
    for size in range(rank,0,-1):
        thresholds=np.array([comb(i,size) for i in range(atom_count)],dtype=np.int64)
        column=np.searchsorted(thresholds,residual,side='right')-1
        result[:,size-1]=column
        residual-=thresholds[column]
    return result


def screen_coefficients(specifications,codes,target):
    return np.bitwise_xor.reduce(codes[specifications],axis=1)==target


def _guided_four(codes,target,budget):
    """Meet-in-the-middle residual matching; complete specs counted separately."""
    pairs=defaultdict(list)
    for i,j in combinations(range(len(codes)),2):
        pairs[int(codes[i]^codes[j])].append((i,j))
    seen=set();matching_pair_visits=0;exhausted=True
    for value,left_pairs in pairs.items():
        complement=value^target
        if value>complement:
            continue
        for left in left_pairs:
            for right in pairs.get(complement,()):
                matching_pair_visits+=1
                if set(left)&set(right):
                    continue
                spec=tuple(sorted(left+right));key=combination_rank(spec)
                if key in seen:
                    continue
                if len(seen)>=budget:
                    exhausted=False
                    return sorted(seen),dict(pair_records=comb(len(codes),2),matching_pair_visits=matching_pair_visits,
                                             complete_family_exhausted=exhausted)
                seen.add(key)
    return sorted(seen),dict(pair_records=comb(len(codes),2),matching_pair_visits=matching_pair_visits,
                            complete_family_exhausted=exhausted)


def _reference(tensor):
    by_pair={}
    for i,j,k in tensor['entries']:
        by_pair[i,j]=by_pair.get((i,j),0)^(1<<k)
    decomposition=[dict(left_mask=1<<i,right_mask=1<<j,output_mask=mask)
                   for (i,j),mask in sorted(by_pair.items())]
    return km.Circuit(lower_decomposition(tensor,decomposition,clean_helpers=())['candidate'])


def _native_batch(reference,finalists,helpers,max_input_bits):
    public=sorted(set(range(max([reference.num_qubits]+[c.num_qubits for _,c,_ in finalists])))-set(helpers))
    if len(public)>max_input_bits:
        return dict(status='unknown',reason='native exhaustive input budget',distinct_candidates=0,shots=0),[]
    cases=1<<len(public);count=len(finalists);shots=count*cases
    if not finalists:
        return dict(status='not_run',distinct_candidates=0,shots=0),[]
    width=max(reference.num_qubits,max(c.num_qubits for _,c,_ in finalists),max(helpers,default=-1)+1)
    indices=np.arange(cases,dtype=np.uint64)
    initial={q:((indices>>i)&1).astype(bool) for i,q in enumerate(public)}
    before=km.Simulator(batch_size=cases,seed=101)
    for q,values in initial.items():before.write_across_shots(km.q(q),values)
    before.do(reference)
    expected=[before.read_across_shots(km.q(q)) for q in range(width)]
    expected_phase=before.read_phase_flipped_across_shots()
    sim=km.Simulator(batch_size=shots,seed=102)
    for q,values in initial.items():sim.write_across_shots(km.q(q),np.tile(values,count))
    segments=[]
    for index,(_,circuit,_) in enumerate(finalists):
        selector=np.zeros(shots,dtype=bool);selector[index*cases:(index+1)*cases]=True
        sim.write_across_shots(km.b(index),selector)
        segments.append(f'PUSH_CONDITION if b{index}\n{circuit}\nPOP_CONDITION')
    sim.do(km.Circuit('\n'.join(segments)))
    actual=[sim.read_across_shots(km.q(q)) for q in range(width)]
    phase=sim.read_phase_flipped_across_shots();checks=[]
    for index,(key,_,_) in enumerate(finalists):
        span=slice(index*cases,(index+1)*cases)
        passed=all(np.array_equal(actual[q][span],expected[q]) for q in range(width)) and np.array_equal(phase[span],expected_phase)
        checks.append(dict(candidate_index=key,passed=passed))
    return dict(status='verified' if all(c['passed'] for c in checks) else 'failed',
                distinct_candidates=count,shots=shots,input_assignments_per_candidate=cases,
                quantum_width=width,public_input_wires=public,clean_helper_inputs=list(helpers),
                method='one native kickmix selector batch; all public and dirty-output assignments; exact sign and helper restoration',
                baseline_native_shots=cases,checks=checks),checks


def search_candidates(tensor, *, strategy='guided',candidate_budget=1_000_000,seed=123,
                      batch_size=65536,materialization_limit=128,native_limit=8,
                      clean_helpers=None,max_native_input_bits=12):
    """Search the same four-product family by random sampling or exact residuals.

    Random selection is uniform without replacement. Guided matching may exhaust
    all algebraically equivalent members before reaching the candidate cap.
    Pair construction is additional guided work and is reported separately.
    """
    if strategy not in {'guided','random'} or any(type(v)is not int or v<1 for v in
        (candidate_budget,batch_size,materialization_limit,native_limit,max_native_input_bits)):
        raise StrataError('invalid candidate search strategy or budgets')
    started=perf_counter();bounds=tensor_bounds(tensor)
    if bounds['value']>4:
        raise StrataError('four-product candidate family is below the proved tensor lower bound')
    atoms,codes=atom_catalog(tensor['shape'])
    target=coefficient_code(tensor);family=comb(len(atoms),4)
    if not family:raise StrataError('four-product family needs at least four rank-one atoms')
    if tensor.get('output_wires') is None:raise StrataError('native candidate search requires physical output wires')
    reference=_reference(tensor)
    helper_start=max(tensor['left']+tensor['right']+tensor['output_wires'])+1
    helpers=tuple(range(helper_start,helper_start+2)) if clean_helpers is None else tuple(clean_helpers)
    if len(helpers)<2 or any(type(q)is not int or q<0 for q in helpers) or len(set(helpers))!=len(helpers) or set(helpers)&set(tensor['left']+tensor['right']+tensor['output_wires']):
        raise StrataError('candidate search needs two distinct explicitly clean helpers outside the public wires')
    analysis_seconds=perf_counter()-started;proposal_started=perf_counter();partial={}
    if strategy=='guided':
        ranks,partial=_guided_four(codes,target,min(candidate_budget,family))
        ranks=np.asarray(ranks,dtype=np.int64)
    else:
        ranks=np.random.default_rng(seed).choice(family,size=min(candidate_budget,family),replace=False,shuffle=False)
    proposal_seconds=perf_counter()-proposal_started
    survivors=[];screen_seconds=0.;unrank_seconds=0.
    for offset in range(0,len(ranks),batch_size):
        chunk=ranks[offset:offset+batch_size]
        then=perf_counter();specs=unrank_combinations(chunk,len(atoms));unrank_seconds+=perf_counter()-then
        then=perf_counter();passed=screen_coefficients(specs,codes,target);screen_seconds+=perf_counter()-then
        survivors.extend(int(value) for value in chunk[passed])
    # Canonical index order makes materialization selection deterministic. All
    # survivor counts are reported before either materialization/native cap.
    survivors.sort();materialized=[];program_hashes=set();duplicates=0
    materialize_started=perf_counter()
    for key in survivors[:materialization_limit]:
        spec=unrank_combinations([key],len(atoms))[0]
        decomposition=[atoms[int(i)] for i in spec]
        if not verify_decomposition(tensor,decomposition)['passed']:
            raise StrataError('independent coefficient replay rejected a screened candidate')
        lowered=lower_decomposition(tensor,decomposition,clean_helpers=helpers)
        circuit=km.Circuit(lowered['candidate']);text=str(circuit);code_hash=hashlib.sha256(text.encode()).hexdigest()
        if code_hash in program_hashes:
            duplicates+=1;continue
        program_hashes.add(code_hash);resources=Kickmix().measured(circuit)
        materialized.append((key,circuit,dict(specification=[int(i) for i in spec],decomposition=decomposition,
            artifact_sha256=code_hash,native_text=text,resources=resources,cost=resources['Q']*resources['toffoli'])))
    materialization_seconds=perf_counter()-materialize_started
    materialized.sort(key=lambda row:(row[2]['cost'],row[2]['resources']['instructions'],row[0]))
    finalists=materialized[:native_limit]
    then=perf_counter();native,checks=_native_batch(reference,finalists,helpers,max_native_input_bits);native_seconds=perf_counter()-then
    verified=[];then=perf_counter()
    for (key,circuit,row),native_check in zip(finalists,checks):
        proof=equivalent_windows(str(reference),str(circuit),clean_inputs=helpers,
            max_variables=64,max_operations=10000,max_terms=65536,max_product_terms=1048576)
        if not native_check['passed'] or proof.get('equivalent')is not True:
            raise StrataError('candidate failed native or exact complete-boundary verification')
        verified.append(dict(candidate_index=key,**row,exact_verification=proof))
    proof_seconds=perf_counter()-then
    baseline=Kickmix().measured(reference);baseline_cost=baseline['Q']*baseline['toffoli']
    best=min(verified,key=lambda row:row['cost']) if verified else None
    return dict(schema_version='1',kind='bounded_bilinear_candidate_search',strategy=strategy,seed=seed,
        tensor_hash=tensor['tensor_hash'],tensor_shape=tensor['shape'],tensor=tensor,bounds=bounds,
        family=dict(products=4,atoms=len(atoms),distinct_specifications=family,
                    identity='unordered set of four distinct rank-one products; no repeated factors or ordering duplicates'),
        selection='uniform_without_replacement' if strategy=='random' else 'exact_residual_meet_in_middle',
        counts=dict(candidate_budget=candidate_budget,unique_complete_specifications_screened=len(ranks),
                    exact_coefficient_survivors=len(survivors),partial_search=partial,
                    specifications_materialized=min(len(survivors),materialization_limit),
                    unique_native_artifacts=len(materialized),materialized_program_duplicates=duplicates,
                    native_candidates_checked=native['distinct_candidates'],native_input_shots=native['shots'],
                    final_exact_proofs=len(verified)),
        timings_seconds=dict(analysis=analysis_seconds,proposal=proposal_seconds,combination_unranking=unrank_seconds,
            coefficient_screen=screen_seconds,materialization=materialization_seconds,native_replay=native_seconds,
            final_exact_proof=proof_seconds,total=perf_counter()-started),
        baseline=dict(resources=baseline,cost=baseline_cost,native=str(reference)),native_validation=native,
        best=best,verified_candidates=verified,accepted_gain=max(0,baseline_cost-best['cost']) if best else None,
        objective=dict(expression='Q*toffoli',accounting='emitted native circuit; two clean helper wires priced when used'),
        limits=dict(batch_size=batch_size,materialization_limit=materialization_limit,native_limit=native_limit),
        interpretation='restricted algebraic solver experiment; guided precision is not an unbiased success probability; equal candidate caps do not imply equal computation',
        forced_measurement_events=0,
        native_scope='every public/dirty output assignment and zero declared helpers; complete outputs and exact sign; not a host-task certificate')


def generated_contract(shape, *, seed, minimum_coordinate_products=6):
    """Seeded benchmark instance; return only its recovered circuit contract.

    Generating factors are discarded. Search sees the emitted coordinate-product
    circuit and its independently recovered tensor, never the seed witness.
    """
    atoms,codes=atom_catalog(shape);rng=np.random.default_rng(seed)
    a,b,c=shape
    for _ in range(10000):
        chosen=rng.choice(len(atoms),size=4,replace=False)
        encoded=int(np.bitwise_xor.reduce(codes[chosen]))
        outputs=[[] for _ in range(c)];pairs=set()
        for i,j,k in product(range(a),range(b),range(c)):
            if encoded>>((i*b+j)*c+k)&1:
                outputs[k].append([i,a+j]);pairs.add((i,j))
        if len(pairs)<minimum_coordinate_products:
            continue
        tensor=tensor_from_anf(outputs,range(a),range(a,a+b),output_wires=range(a+b,a+b+c))
        source=_reference(tensor)
        sem=analyze_window(str(source),max_variables=64,max_operations=10000)
        feature=next((f for f in sem.get('features',[]) if f['kind']=='bilinear_accumulator'),None)
        if feature is None:continue
        left,right=feature['factor_partition']
        # Require a connected partition of the requested dimensions; otherwise
        # independent bipartite components permit alternative recolorings.
        if [len(left),len(right),len(feature['targets'])]!=list(shape):continue
        recovered=tensor_from_anf(feature['deltas'],left,right,output_wires=feature['targets'])
        return recovered,dict(seed=seed,shape=list(shape),source_native=str(source),
            source_sha256=hashlib.sha256(str(source).encode()).hexdigest(),generating_witness_exposed_to_search=False,
            semantic_contract_hash=digest(dict(outputs=sem['outputs'],phase=sem['phase'])))
    raise StrataError('could not generate requested bounded benchmark contract')
