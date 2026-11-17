"""Nonlinear all-outcome labs, exact phase, budgets, and native cross-checks."""
from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction
import random

import kickmix as km
import numpy as np
import pytest

from strata.lab import exhaustive, validate_lab_receipt
from strata.lab_program import compile_program
from strata.model import Contract, Register, StrataError


CONTRACT = Contract('nonlinear', (
    Register('x', (0,)), Register('scratch', (1,), role='clean'),
    Register('dirty', (2,), role='dirty'),
    Register('mask', (500,), namespace='classical'),
    Register('saved', (501,), namespace='classical', role='dirty'),
    Register('temporary', (502,), namespace='classical', role='clean')), 'id')
REFERENCE = lambda inp, _: ({'x': inp['x'], 'mask': inp['mask']}, 0)


def nonlinear(count=102, repair=True):
    # A nonlinear XOR-of-products controls a snapshotted scope; the public
    # classical mask then controls whether the final measurement occurs.
    rows = [f'HMR q1 b{i}' for i in range(count)]
    rows += ['BIT_STORE0 b200', 'PUSH_CONDITION if b0', 'BIT_INVERT b200 if b1',
             'POP_CONDITION', 'BIT_INVERT b200 if b2', 'PUSH_CONDITION if b200',
             'BIT_STORE0 b200', 'CX q0 q1 if b500', f'HMR q1 b{count} if b500']
    if repair:
        rows.append(f'Z q0 if b{count}')
    rows += ['X q2 if b501', 'X q2 if b501', 'BIT_INVERT b501', 'BIT_INVERT b501',
             'Z_POW q0 0.125', 'Z_POW q0 1.875', 'POP_CONDITION', 'BIT_STORE0 b502']
    return '\n'.join(rows)


def test_103_correlated_measurements_prove_all_outcomes_and_reject_missing_repair():
    circuit = km.Circuit(nonlinear())
    assert not compile_program(str(circuit)).affine_hmr
    result = exhaustive(circuit, CONTRACT, REFERENCE, max_trajectories=16)
    assert result.passed and result.exhaustive and result.reduction == 'symbolic_hmr'
    assert result.hmr_count == 103 and result.represented_branches == 1 << 103
    assert result.cases == result.executed_cases == result.tested == 16
    assert result.executed_branches == 1
    assert result.represented_trajectories == 16 * (1 << 103)
    assert result.proof['proved_cases'] == 16
    validate_lab_receipt(asdict(result))
    bad = exhaustive(km.Circuit(nonlinear(repair=False)), CONTRACT, REFERENCE,
                     max_trajectories=16)
    assert not bad.passed and not bad.exhaustive
    assert bad.tested == bad.executed_branches == 0
    assert bad.proof['native_cross_check_branches'] == []
    assert bad.counterexample['input']['x'] == bad.counterexample['input']['mask'] == 1
    assert bad.counterexample['hmr_branch'] >> 102 & 1
    assert bad.counterexample['phase'] == '1'
    # The reported witness is executable by the separate native branch runner.
    inp, branch = bad.counterexample['input'], bad.counterexample['hmr_branch']
    native = km.Circuit(nonlinear(repair=False))
    sim = km.Simulator(batch_size=1)
    sim.use_same_registers_as(native)
    for register in CONTRACT.registers:
        factory = km.q if register.namespace == 'quantum' else km.b
        for j, wire in enumerate(register.wires):
            sim.write_across_shots(factory(wire), np.array([bool(inp[register.name] >> j & 1)]))
    assert compile_program(str(native)).execute(sim, [(branch, 0)], lambda: None) is None
    assert Fraction(sim.read_shot_phase(0)) % 2 == 1


def test_symbolic_promises_and_shards_keep_domain_and_scope_binding():
    contract = replace(CONTRACT, promise='mask-is-zero')
    domain = [dict(x=x, dirty=d, mask=0, saved=s) for x in range(2)
              for d in range(2) for s in range(2)]
    circuit = km.Circuit(nonlinear(repair=False))
    whole = exhaustive(circuit, contract, REFERENCE, domain=domain, max_trajectories=8)
    assert whole.passed and whole.cases == 8
    validate_lab_receipt(asdict(whole))
    shard = exhaustive(circuit, contract, REFERENCE, domain=reversed(domain),
                       max_trajectories=4, shard_index=1, shard_count=2)
    assert shard.passed and not shard.exhaustive and shard.evidence_kind == 'sample'
    assert shard.domain_hash == whole.domain_hash and shard.proof['proved_cases'] == 4
    with pytest.raises(StrataError):
        validate_lab_receipt(asdict(shard))


@pytest.mark.parametrize('suffix,register', [
    ('X q1 if b0', 'scratch'), ('X q2 if b1', 'dirty'),
    ('BIT_INVERT b501 if b0', 'saved'), ('BIT_STORE1 b502 if b0', 'temporary'),
    ('BIT_STORE0 b500', 'mask')])
def test_symbolic_quantum_and_classical_boundary_restoration(suffix, register):
    result = exhaustive(km.Circuit(nonlinear() + '\n' + suffix), CONTRACT, REFERENCE,
                        max_trajectories=16)
    assert not result.passed and register in result.counterexample['values']


def test_symbolic_reset_proves_legality_for_every_outcome():
    result = exhaustive(km.Circuit(nonlinear() + '\nX q1 if b0\nR q1'), CONTRACT,
                        REFERENCE, max_trajectories=16)
    assert not result.passed and result.counterexample['reason'] == 'reset_nonzero'
    assert result.counterexample['hmr_branch'] & 1


def test_symbolic_fractional_phase_uses_native_exact_dyadic_exponents():
    circuit = km.Circuit(nonlinear() + '\nZ_POW q0 0.1')
    native = km.Simulator(batch_size=1)
    native.do(km.Circuit('X q0\nZ_POW q0 0.1'))
    exponent = Fraction(native.read_shot_phase(0)) % 2
    reference = lambda inp, _: ({'x': inp['x'], 'mask': inp['mask']}, exponent * inp['x'])
    result = exhaustive(circuit, CONTRACT, reference, max_trajectories=16)
    assert result.passed and result.proof['phase_bits'] == exponent.denominator.bit_length()
    mutation = km.Circuit(nonlinear() + '\nZ_POW q0 0.125 if b0')
    bad = exhaustive(mutation, CONTRACT, REFERENCE, max_trajectories=16)
    assert not bad.passed and bad.counterexample['phase'] == '1/8'


@pytest.mark.parametrize('limits', [
    {'max_symbolic_nodes': 2}, {'max_symbolic_apply_steps': 1},
    {'max_symbolic_operations': 1}, {'max_symbolic_variables': 102},
    {'max_symbolic_phase_bits': 1}])
def test_symbolic_budget_exhaustion_never_returns_a_sample_as_proof(limits):
    with pytest.raises(StrataError, match='symbolic HMR.*budget.*no certificate'):
        exhaustive(km.Circuit(nonlinear()), CONTRACT, REFERENCE,
                   max_trajectories=16, **limits)


def test_symbolic_operation_limit_is_global_across_inputs():
    circuit = km.Circuit('X q0\nX q0')
    contract = Contract('two-inputs', (Register('x', (0,)),), 'id')
    with pytest.raises(StrataError, match='symbolic_operation_budget'):
        exhaustive(circuit, contract, lambda inp, _: (inp, 0), branch_strategy='symbolic',
                   max_symbolic_operations=3)


def test_symbolic_receipt_cannot_invent_work_or_drop_the_all_outcome_proof():
    receipt = asdict(exhaustive(km.Circuit(nonlinear()), CONTRACT, REFERENCE,
                               max_trajectories=16))
    for field, value in [('proved_cases', 1), ('all_measurement_outcomes', False),
                         ('exact_phase', False), ('measurement_variables', 102),
                         ('mode', 'sampled'), ('native_cross_check_branches', [0, 1]),
                         ('apply_steps', 2000000), ('decision_nodes', 1)]:
        altered = deepcopy(receipt)
        altered['proof'][field] = value
        with pytest.raises(StrataError):
            validate_lab_receipt(altered)
    for field, value in [('proof', None), ('executed_branches', 1 << 103),
                         ('tested', receipt['represented_trajectories'])]:
        with pytest.raises(StrataError):
            validate_lab_receipt({**receipt, field: value})
    # Existing receipts have no mandatory new field.
    legacy = asdict(exhaustive(km.Circuit('X q0\nX q0'),
                    Contract('id', (Register('x', (0,)),), 'id'), lambda inp, _: (inp, 0)))
    del legacy['proof']
    validate_lab_receipt(legacy)


def test_small_symbolic_runs_match_full_native_branch_enumeration():
    rng = random.Random(418)
    contract = Contract('crosscheck', (Register('x', (0,)), Register('scratch', (1,), role='clean'),
                                     Register('mask', (3,), namespace='classical')), 'id')
    reference = lambda inp, _: ({'x': inp['x'], 'mask': inp['mask']}, 0)
    gates = ['X q0', 'CX q0 q1', 'SWAP q0 q1', 'Z q0', 'Z_POW q0 0.125',
             'BIT_STORE1 b0', 'BIT_STORE0 b1', 'BIT_INVERT b1', 'HMR q1 b1']
    for _ in range(24):
        rows = ['HMR q1 b0', 'HMR q1 b1', 'PUSH_CONDITION if b0']
        for _ in range(8):
            gate = rng.choice(gates)
            rows.append(gate + f' if b{rng.choice([0, 1, 3])}')
        rows += ['POP_CONDITION', 'BIT_INVERT b3', 'BIT_INVERT b3']
        circuit = km.Circuit('\n'.join(rows))
        symbolic = exhaustive(circuit, contract, reference, branch_strategy='symbolic')
        native = exhaustive(circuit, contract, reference, branch_strategy='full')
        assert symbolic.passed is native.passed
        if symbolic.passed:
            validate_lab_receipt(asdict(symbolic))


def test_symbolic_component_lab_receipt_validates():
    from strata.bundle import Bundle, Component, BundleFrontend
    from strata.component_lab import ComponentLab, run_component_labs, validate_component_lab
    frontend = BundleFrontend((Component('decoder', 'decoder.kmx', 'kickmix'),))
    artifact = Bundle({'decoder.kmx': nonlinear().encode()})
    class Adapter:
        def lab_components(self, parameters):
            return {'decoder': ComponentLab(CONTRACT, REFERENCE)}
    result = run_component_labs(frontend, artifact, frontend.tree(artifact, 'decoder'), {}, Adapter(),
                                options={'max_trajectories': 16})
    assert result.passed and result.exhaustive
    assert result.components['decoder']['lab']['reduction'] == 'symbolic_hmr'
    validate_component_lab(asdict(result), frontend)
