"""Independent bounded native replay of a changed episode on its declared inputs."""
from fractions import Fraction
import importlib.util
from pathlib import Path

import kickmix as km
import numpy as np

ORACLE = Path(__file__).resolve().parent.parent / 'v05-holdout' / 'run.py'


def native_run(circuit, quantum, classical):
    sim = km.Simulator(batch_size=quantum.shape[1], seed=45219)
    qs = [km.q(i) for i in range(len(quantum))]
    bs = [km.b(i) for i in range(len(classical))]
    sim.write_across_shots(qs, quantum)
    if bs:
        sim.write_across_shots(bs, classical)
    for line in str(circuit).splitlines():
        words = line.split()
        if words[0] == 'R' and np.any(sim.read_across_shots(km.q(int(words[1][1:])))):
            return None
        sim.do(km.Circuit(line))
    phase = [Fraction(sim.read_shot_phase(i)) % 2 for i in range(quantum.shape[1])]
    return sim.read_across_shots(qs), sim.read_across_shots(bs) if bs else None, phase


def independent_check(before, after, clean, private, domain, cap):
    if before == after:
        return dict(equivalent=True, method='exact_native_circuit_identity')
    width = before.num_qubits
    bits = max(before.num_bits, after.num_bits)
    has_measurement = any(line.split()[0]=='HMR' for c in (before, after) for line in str(c).splitlines())
    if has_measurement:
        if domain is not None or any('Z_POW' in str(c) for c in (before, after)):
            return dict(equivalent=None, reason='mixed_native_replay_scope')
        spec = importlib.util.spec_from_file_location('holdout_native_oracle', ORACLE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.native_equal(before, after, clean, private, cap)
    address_wires = domain['address_wires'] if domain else []
    arbitrary = (domain['arbitrary_wires'] if domain else
                 sorted(set(range(width))-set(clean)))
    if domain and 'interval' in domain:
        a, b = domain['interval']
        addresses = range(a, b+1)
    else:
        addresses = domain['addresses'] if domain else [0]
    count = len(addresses) << (len(arbitrary) + bits)
    if count > cap:
        return dict(equivalent=None, reason='native_case_cap', total_cases=count, cap=cap)
    visible = [i for i in range(bits) if i not in private]
    for start in range(0, count, 1024):
        batch = range(start, min(count, start+1024))
        quantum = np.zeros((width, len(batch)), dtype=np.bool_)
        classical = np.zeros((bits, len(batch)), dtype=np.bool_)
        for j, index in enumerate(batch):
            remaining, addr = divmod(index, len(addresses))
            for i, q in enumerate(address_wires):
                quantum[q,j] = (addresses[addr] >> i) & 1
            for i, q in enumerate(arbitrary):
                quantum[q,j] = (remaining >> i) & 1
            for i in range(bits):
                classical[i,j] = (remaining >> (len(arbitrary)+i)) & 1
        left = native_run(before, quantum, classical)
        right = native_run(after, quantum, classical)
        if left is None or right is None:
            return dict(equivalent=None, reason='dirty_reset_needs_outcome_enumeration', total_cases=count)
        if (not np.array_equal(left[0], right[0]) or left[2] != right[2] or
                (visible and not np.array_equal(left[1][visible], right[1][visible]))):
            return dict(equivalent=False, reason='native_boundary_mismatch', batch_start=start)
    return dict(equivalent=True, method='exhaustive_native_arbitrary_word_and_classical_assignments',
                total_cases=count, phases='exact native Fraction modulo two',
                reset_check='every reset operand zero at invocation on every included input')
