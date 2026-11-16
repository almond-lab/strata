"""Original, public-safe structural fixture: eliminate a materialized AND boundary.

This demonstrates L3 producer/consumer/cleanup fusion, NOT a Kaggle discovery.
The independent reference is y ^= a & b. Dirty y and all input values are tested.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import numpy as np
import kickmix as km
from .model import Contract, Register, Region, Floor, StrataError
from .kickmix import Kickmix


class AndFusion:
    def build(self, parameters, implementation):
        n = parameters['bits']
        fused = implementation.get('fused', False)
        gates = []
        for i in range(n):
            if fused:
                gates.append(f'CCX q{i} q{n+i} q{2*n+i}')
            else:
                gates.extend([f'CCX q{i} q{n+i} q{3*n+i}', f'CX q{3*n+i} q{2*n+i}', f'CCX q{i} q{n+i} q{3*n+i}'])
        circuit = km.Circuit('\n'.join(gates))
        regs = [Register('a', tuple(range(n)), meaning='uint'), Register('b', tuple(range(n, 2*n)), meaning='uint'),
                Register('y', tuple(range(2*n, 3*n)), role='output', meaning='xor accumulator')]
        if not fused:
            regs.append(Register('scratch', tuple(range(3*n, 4*n)), role='clean'))
        root = Contract('and-xor', tuple(regs), reference='strata.demo:and-xor:v1', scales={'bits': n})
        children = []
        for i in range(n):
            start = i if fused else 3*i
            width = 1 if fused else 3
            # Full episode contracts are named by occurrence, stable across offset changes.
            r = Contract('lane', (Register('a', (i,)), Register('b', (n+i,)), Register('y', (2*n+i,), role='output')),
                         reference='and-xor-bit:v1')
            children.append(Region(f'lane-{i}', start, start+width, r, kind='and-xor', evidence='exhaustive fixture', occurrence=i))
        return circuit, Region('root', 0, len(gates), root, tuple(children), kind='and-xor')

    @staticmethod
    def reference(values, scale):
        return {'a': values['a'], 'b': values['b'], 'y': values['y'] ^ (values['a'] & values['b'])}, 0

    def oracle(self, circuit, parameters, domain):
        n = parameters['bits']
        cases = 1 << (3*n)
        if cases > 1_000_000:
            raise StrataError('demo full-domain budget exceeded')
        x = np.arange(cases, dtype=np.uint64)
        sim = km.Simulator(batch_size=cases, seed=7, counting_operations=True)
        wires = [km.q(i) for i in range(circuit.num_qubits)]
        bits = np.zeros((circuit.num_qubits, cases), dtype=bool)
        for i in range(3*n):
            bits[i] = ((x >> i) & 1).astype(bool)
        sim.write_across_shots(wires, bits)
        sim.do(circuit)
        actual = sim.read_across_shots(wires)
        expected = bits.copy()
        expected[2*n:3*n] ^= bits[:n] & bits[n:2*n]
        passed = bool(np.array_equal(actual, expected) and not np.any(sim.read_phase_flipped_across_shots()))
        return dict(passed=passed, cases=cases, domain='all inputs including dirty accumulator',
                    phase='exact 0 (fixture has no rotations)',
                    resources=dict(Q=circuit.num_qubits, toffoli=sim.operation_statistics().num_toffoli_like / cases,
                                   t=0, rotations=0), seeds=domain.get('seeds', []))

    def measure(self, circuit, parameters, domain):
        result = self.oracle(circuit, parameters, domain)
        if not result['passed']:
            raise StrataError('measured build violates function')
        return result['resources']

    def verify(self, circuit, parameters, domain):
        # For this built-in fixture only, the official verifier is this local runner.
        r = self.oracle(circuit, parameters, domain)
        r['verifier_hash'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        r['authority'] = 'Strata demo verifier, not a competition verifier'
        r['failure_model'] = dict(kind='exact', failure_rate=0, checked_cases=r['cases'])
        return r

    def ledger(self, circuit, tree):
        lines = str(circuit).splitlines()
        return {r.id: Kickmix().measured(km.Circuit('\n'.join(lines[r.start:r.stop]))) for r in tree.walk()}

    def floors(self, tree):
        # A zero-AND XOR/NOT circuit is affine; AND-XOR is not. This is only a
        # one-bit lower bound. Do not multiply it into an unproved direct-sum bound.
        return [Floor(r.contract.fingerprint, 'toffoli', 1, 'proof',
                      'one-bit AND-XOR is non-affine; XOR/NOT-only networks are affine')
                for r in tree.children]

    def constrain(self, parameters, implementation, constraint):
        if constraint != dict(region='root', resource='wire', delta=-parameters['bits']):
            raise StrataError('this fixture supports only removing the materialized product register')
        return self.build(parameters, {**implementation, 'fused': True})
