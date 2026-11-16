"""Bounded empirical execution views. Observations never become proofs."""
from collections import defaultdict
import math
import numpy as np
import kickmix as km
from .kickmix import instructions
from .model import StrataError


def execution_profile(circuit, initial, seed=0, max_ops=5000, fingerprint_limit=128):
    initial=np.asarray(initial,dtype=bool)
    if initial.ndim!=2 or initial.shape[0]!=circuit.num_qubits or not initial.shape[1]:
        raise StrataError('initial states must have shape (num_qubits, shots)')
    if len(circuit)>max_ops:raise StrataError('execution profile operation budget exceeded')
    sim=km.Simulator(batch_size=initial.shape[1],seed=seed,counting_operations=True)
    wires=[km.q(q) for q in range(circuit.num_qubits)]
    sim.write_across_shots(wires,initial)
    trace=[];peaks=[];peak=-1
    producer={};conditionals=[]
    for i,line,op,qs in instructions(circuit):
        before=sim.operation_statistics().num_toffoli_like
        sim.do(km.Circuit(line))
        executed=(sim.operation_statistics().num_toffoli_like-before)/initial.shape[1]
        state=sim.read_across_shots(wires)
        live=np.any(state,axis=1)
        count=int(live.sum())
        trace.append(dict(at=i,mean_toffoli=executed,observed_nonzero_wires=count))
        if count>peak:peak=count;peaks=[]
        if count==peak and len(peaks)<16:peaks.append(dict(at=i,wires=np.flatnonzero(live).tolist()))
        words=line.split()
        if op=='HMR':producer[words[2]]=i
        elif op.startswith('BIT_'):
            if len(words)>1:producer[words[1]]=i
        if 'if' in words:
            bit=words[-1];conditionals.append(dict(at=i,bit=bit,last_assignment=producer.get(bit)))
    groups=defaultdict(list)
    state=sim.read_across_shots(wires)
    for q in range(min(circuit.num_qubits,fingerprint_limit)):
        groups[np.packbits(state[q]).tobytes()].append(q)
    equal=[v for v in groups.values() if len(v)>1]
    # Distinct observed whole states are not an entropy or independence proof.
    distinct=len({np.packbits(state[:,j]).tobytes() for j in range(initial.shape[1])})
    return dict(evidence='empirical_fixed_input_batch',seed=seed,shots=initial.shape[1],
                executed_cost=dict(mean_toffoli=sim.operation_statistics().num_toffoli_like/initial.shape[1],trace=trace),
                liveness=dict(kind='observed_lower_bound_on_possible_nonzero',peak=peak,peaks=peaks),
                fingerprints=dict(equal_sampled_wires=equal,proof=False,at='output'),conditionals=conditionals,
                slack=dict(status='unknown',observed_distinct_states=distinct,
                           note='Finite observations cannot certify information slack or free scratch'),
                allocation=dict(registers={str(k):[str(q) for q in v] for k,v in circuit.register_data.items()},
                                ownership='Declared registers only; dynamic object ownership needs generator trace'))
