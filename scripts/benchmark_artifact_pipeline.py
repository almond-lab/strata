"""Measure real native emission, parsing and exhaustive SDK candidate replay.

Every counted candidate becomes a distinct canonical kickmix circuit and runs
on all 256 assignments of six controls plus two dirty output accumulators.
Two declared-zero helpers are restored. No candidate is screened out in Python.
This is a small bilinear-episode pipeline benchmark, not a host-task benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from itertools import product
import json
from math import comb
from pathlib import Path
import platform
import random
import resource
import sys
from time import perf_counter

import kickmix as km
import numpy as np

from strata.kickmix import Kickmix
from strata.tensor import tensor_from_anf, lower_decomposition


ATOMS = tuple(product(range(1, 8), range(1, 8), range(1, 4)))
INPUT_SHOTS = 256
LAYOUT = dict(left=[0,1,2], right=[3,4,5], dirty_outputs=[6,7], clean_helpers=[8,9])


def coefficients(spec):
    encoded=0
    for index in spec:
        a,b,c=ATOMS[index]
        for i,j,k in product(range(3),range(3),range(2)):
            if (a>>i)&(b>>j)&(c>>k)&1:
                encoded ^= 1<<((i*3+j)*2+k)
    return encoded


def expected_outputs(encoded, inputs):
    expected=np.zeros((10,INPUT_SHOTS),dtype=bool)
    expected[:8]=inputs
    for i,j,k in product(range(3),range(3),range(2)):
        if encoded>>((i*3+j)*2+k)&1:
            expected[6+k]^=inputs[i]&inputs[3+j]
    return expected


def emit(spec, encoded):
    outputs=[[],[]]
    for i,j,k in product(range(3),range(3),range(2)):
        if encoded>>((i*3+j)*2+k)&1:
            outputs[k].append([i,3+j])
    tensor=tensor_from_anf(outputs,LAYOUT['left'],LAYOUT['right'],output_wires=LAYOUT['dirty_outputs'])
    factors=[dict(left_mask=ATOMS[i][0],right_mask=ATOMS[i][1],output_mask=ATOMS[i][2]) for i in spec]
    lowered=lower_decomposition(tensor,factors,clean_helpers=LAYOUT['clean_helpers'])
    return 'APPEND_TO_REGISTER q9 r0\nREGISTER r0 "benchmark_boundary"\n'+lowered['candidate']


def run(*, candidates=1000, seed=417921, progress=None, progress_every=50000):
    if type(candidates) is not int or not 1<=candidates<=comb(len(ATOMS),4):
        raise ValueError('candidate count must fit the four-distinct-atom family')
    random_source=random.Random(seed)
    inputs=np.array([[(value>>wire)&1 for value in range(INPUT_SHOTS)] for wire in range(8)],dtype=bool)
    # A fixed reference is separate from random candidate generation. All native
    # results are compared both to it and to their own coefficient specification.
    target_spec=(2,19,73,131)
    target_encoded=coefficients(target_spec)
    target=expected_outputs(target_encoded,inputs)
    timings={name:0.0 for name in ('spec_generation','coefficient_and_reversible_emission',
        'native_parse_and_static_measurement','native_simulator_setup','native_simulation',
        'native_readback','oracle_comparison','canonical_hashing')}
    seen_specs=set();seen_programs=set();seen_functions=set();duplicate_draws=0
    sequence_hash=hashlib.sha256();examples=[];matches=0;native_calls=0;instructions=0;phase_failures=0
    min_instructions=None;max_instructions=0
    frontend=Kickmix();wires=[km.q(q) for q in range(10)]
    begun=perf_counter()
    for ordinal in range(candidates):
        mark=perf_counter()
        while True:
            spec=tuple(sorted(random_source.sample(range(len(ATOMS)),4)))
            if spec not in seen_specs:
                seen_specs.add(spec);break
            duplicate_draws+=1
        timings['spec_generation']+=perf_counter()-mark

        mark=perf_counter()
        encoded=coefficients(spec)
        source=emit(spec,encoded)
        timings['coefficient_and_reversible_emission']+=perf_counter()-mark

        mark=perf_counter()
        circuit=km.Circuit(source)
        measured=frontend.measured(circuit)
        if circuit.num_qubits!=10 or measured['toffoli']!=4:
            raise AssertionError('candidate escaped the fixed width/product family')
        instructions+=measured['instructions']
        min_instructions=measured['instructions'] if min_instructions is None else min(min_instructions,measured['instructions'])
        max_instructions=max(max_instructions,measured['instructions'])
        timings['native_parse_and_static_measurement']+=perf_counter()-mark

        mark=perf_counter()
        simulator=km.Simulator(batch_size=INPUT_SHOTS,seed=seed)
        for q in range(8):simulator.write_across_shots(km.q(q),inputs[q])
        timings['native_simulator_setup']+=perf_counter()-mark

        mark=perf_counter()
        simulator.do(circuit)
        native_calls+=1
        timings['native_simulation']+=perf_counter()-mark

        mark=perf_counter()
        outputs=simulator.read_across_shots(wires)
        phases=simulator.read_phase_flipped_across_shots()
        timings['native_readback']+=perf_counter()-mark

        mark=perf_counter()
        expected=expected_outputs(encoded,inputs)
        if not np.array_equal(outputs,expected):
            raise AssertionError(f'native circuit disagrees with its tensor oracle: {spec}')
        if np.any(phases):
            phase_failures+=1
            raise AssertionError(f'native circuit changes relative sign: {spec}')
        matches+=np.array_equal(outputs,target)
        seen_functions.add(encoded)
        timings['oracle_comparison']+=perf_counter()-mark

        mark=perf_counter()
        canonical=str(circuit).encode()
        program_hash=hashlib.sha256(canonical).hexdigest()
        seen_programs.add(program_hash)
        sequence_hash.update(len(canonical).to_bytes(8,'big'));sequence_hash.update(canonical)
        if ordinal<3:
            examples.append(dict(spec=list(spec),tensor_coefficients=encoded,
                canonical_sha256=program_hash,canonical_program=canonical.decode(),resources=measured))
        timings['canonical_hashing']+=perf_counter()-mark
        if progress is not None and (native_calls%progress_every==0 or native_calls==candidates):
            seconds=perf_counter()-begun
            progress(dict(stage='native_candidate_progress',native_candidates=native_calls,
                unique_native_programs=len(seen_programs),input_trajectories=native_calls*INPUT_SHOTS,
                seconds=seconds,native_candidates_per_second=native_calls/seconds,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    elapsed=perf_counter()-begun
    return dict(schema_version=1,benchmark='actual_native_bilinear_candidate_pipeline',
        candidate_family=dict(atom_order='product(range(1,8),range(1,8),range(1,4))',atoms=len(ATOMS),
            terms=4,identity='unordered four distinct atom IDs; canonical increasing order',
            complete_family_size=comb(len(ATOMS),4),sampling='seeded random sample with explicit duplicate rejection',seed=seed),
        counts=dict(requested_candidates=candidates,unique_specs=len(seen_specs),rejected_duplicate_draws=duplicate_draws,
            materialized_native_circuits=native_calls,unique_native_canonical_programs=len(seen_programs),
            native_canonical_duplicates=native_calls-len(seen_programs),unique_boolean_maps=len(seen_functions),
            native_simulator_calls=native_calls,shots_per_candidate=INPUT_SHOTS,
            native_input_trajectories=native_calls*INPUT_SHOTS,emitted_instruction_total=instructions,
            instruction_trajectory_products=instructions*INPUT_SHOTS,measurement_events=0,forced_measurement_outcomes=0,
            independently_spec_checked_candidates=native_calls,native_reference_matches=int(matches),phase_failures=phase_failures),
        boundary=dict(**LAYOUT,physical_width=10,dirty_assignments_exhaustive=True,helpers_zero_on_entry_and_exit=True,
            outputs_checked='all ten physical wires and sign phase'),
        timings=dict(wall_seconds=elapsed,phases_seconds=timings,actual_native_candidates_per_second=native_calls/elapsed,
            native_simulation_only_candidates_per_second=native_calls/timings['native_simulation']),
        program_sizes=dict(min_instructions=min_instructions,max_instructions=max_instructions,
                           mean_instructions=instructions/native_calls,toffoli_per_candidate=4),
        verification=dict(independent_oracle='GF(2) tensor coefficient evaluation on all 256 assignments',
            reference_spec=list(target_spec),reference_tensor_coefficients=target_encoded,
            every_candidate_ran_in_native_sdk=True,python_prefilter_rejections=0,
            compared_to_own_spec_and_fixed_reference=True),
        artifacts=dict(native_program_sequence_sha256=sequence_hash.hexdigest(),examples=examples),
        environment=dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),
            kickmix=importlib.metadata.version('kickmix'),numpy=np.__version__,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        limitations=['Measured small bilinear episodes, not whole host circuits or official score verification.',
            'Instruction/input-trajectory products are reported separately and never counted as candidates.',
            'Many distinct native programs implement the same Boolean map; both counts are reported.',
            'No million-circuit throughput is claimed from a smaller measured run.',
            'This measures emission and native validation, not hypothesis generation quality or persistent per-candidate storage.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=417921)
    parser.add_argument('--progress-every',type=int,default=50000)
    parser.add_argument('--out',required=True)
    arguments=parser.parse_args()
    if arguments.progress_every<1:parser.error('--progress-every must be positive')
    result=run(candidates=arguments.candidates,seed=arguments.seed,
               progress=lambda row:print(json.dumps(row),flush=True),progress_every=arguments.progress_every)
    path=Path(arguments.out);path.parent.mkdir(parents=True,exist_ok=True)
    result['artifacts']['benchmark_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(counts=result['counts'],timings=result['timings'],output=str(path)),indent=2))


if __name__=='__main__':main()
