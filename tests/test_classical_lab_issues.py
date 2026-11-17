"""Classical boundary registers and explicit nonlinear-HMR lab limits (#82/#83)."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import kickmix as km
import pytest

from strata.lab import exhaustive, validate_lab_receipt
from strata.lab_program import compile_program
from strata.model import Contract, Register, Region, StrataError, digest, region_from_dict
from strata.queries import Queries
from strata.project import Line


def classical(name, wires, role='through'):
    return Register(name, wires, role=role, namespace='classical')


def test_register_namespaces_are_distinct_and_quantum_fingerprints_stay_stable():
    q = Register('x', (0,))
    b = classical('mask', (0,))
    contract = Contract('controlled', (q, b), 'controlled')
    assert len(contract.registers) == 2
    with pytest.raises(StrataError, match='overlapping'):
        Contract('invalid', (b, classical('other', (0,))), 'id')
    old_payload = dict(registers=[dict(width=1, role='through', meaning='bits')], reference='id',
                       promise='all', phase='exact', failure='none', scales={}, debts=[], version='1')
    assert Contract('old', (q,), 'id').fingerprint == digest(old_payload)
    assert Contract('classical', (replace(q, namespace='classical'),), 'id').fingerprint != digest(old_payload)
    tree = Region('root', 0, 1, contract)
    assert region_from_dict(asdict(tree)) == tree
    legacy = asdict(Region('root', 0, 1, Contract('old', (q,), 'id')))
    del legacy['contract']['registers'][0]['namespace']
    assert region_from_dict(legacy).contract.fingerprint == digest(old_payload)


def test_lab_enumerates_quantum_and_classical_inputs_and_checks_public_outputs():
    circuit = km.Circuit('X q0 if b0')
    contract = Contract('controlled-x', (Register('x', (0,)), classical('mask', (0,))), 'controlled-x')
    reference = lambda inp, scale: ({'x': inp['x'] ^ inp['mask'], 'mask': inp['mask']}, 0)
    result = exhaustive(circuit, contract, reference)
    assert result.passed and result.cases == result.tested == 4
    validate_lab_receipt(asdict(result))
    bad = exhaustive(circuit, contract, lambda inp, scale: (inp, 0))
    assert not bad.passed and bad.counterexample['input']['mask'] == 1
    assert 'x' in bad.counterexample['values']


def test_classical_only_phase_contract_and_domain_bind_every_input():
    circuit = km.Circuit('NEG if b0')
    contract = Contract('mask-phase', (classical('mask', (0,)),), 'mask-phase')
    result = exhaustive(circuit, contract, lambda inp, scale: (inp, inp['mask']))
    assert result.passed and result.cases == 2
    broken = exhaustive(circuit, contract, lambda inp, scale: (inp, 0))
    assert not broken.passed and broken.counterexample['phase'] == '1'
    with pytest.raises(StrataError, match='weaken'):
        exhaustive(circuit, contract, lambda inp, scale: (inp, inp['mask']), domain=[{'mask': 0}])
    promised = replace(contract, promise='mask-is-zero')
    result = exhaustive(circuit, promised, lambda inp, scale: (inp, 0), domain=[{'mask': 0}])
    assert result.passed and result.domain_source == 'adapter'
    with pytest.raises(StrataError, match='name every'):
        exhaustive(circuit, promised, lambda inp, scale: (inp, 0), domain=[{}])


def test_classical_dirty_and_clean_restoration_do_not_come_from_oracle():
    dirty = Contract('dirty', (classical('saved', (0,), role='dirty'),), 'restore')
    result = exhaustive(km.Circuit('BIT_INVERT b0'), dirty,
                        lambda inp, scale: ({'saved': inp['saved'] ^ 1}, 0))
    assert not result.passed and 'saved' in result.counterexample['values']
    clean = Contract('clean', (classical('scratch', (0,), role='clean'),), 'zero')
    result = exhaustive(km.Circuit('BIT_STORE1 b0'), clean, lambda inp, scale: ({'scratch': 1}, 0))
    assert not result.passed and result.counterexample['values']['scratch']['expected'] == 0


def test_hmr_cannot_overwrite_declared_classical_through_register():
    contract = Contract('hmr', (Register('scratch', (0,), role='clean'), classical('mask', (0,))), 'preserve-mask')
    result = exhaustive(km.Circuit('HMR q0 b0'), contract, lambda inp, scale: ({'mask': inp['mask']}, 0))
    assert not result.passed and 'mask' in result.counterexample['values']
    cleaned = replace(contract, registers=(contract.registers[0], classical('mask', (0,), role='clean')))
    result = exhaustive(km.Circuit('HMR q0 b0\nBIT_STORE0 b0'), cleaned, lambda inp, scale: ({}, 0))
    assert result.passed and result.tested == 2


def test_classical_input_can_activate_nonlinear_hmr_control_hidden_at_zero():
    text = 'HMR q0 b0 if b2\nHMR q1 b1\nPUSH_CONDITION if b0\nNEG if b1\nPOP_CONDITION'
    assert compile_program(text).affine_hmr is True  # Only the default-zero boundary.
    contract = Contract('conditional', (Register('scratch', (0, 1), role='clean'), classical('mask', (2,))), 'id')
    reference = lambda inp, scale: ({'mask': inp['mask']}, 0)
    result = exhaustive(km.Circuit(text), contract, reference)
    assert not result.passed and result.reduction == 'none'
    assert result.counterexample['input']['mask'] == 1 and result.counterexample['hmr_branch'] == 3
    promised = replace(contract, promise='mask-is-zero')
    reduced = exhaustive(km.Circuit(text), promised, reference, domain=[{'mask': 0}], max_trajectories=3)
    assert reduced.passed and reduced.reduction == 'affine_hmr'


def test_classical_storage_outside_circuit_cannot_alias_hmr_temporary_bit():
    contract = Contract('outside', (Register('q', (0,), role='clean'), classical('mask', (1,))), 'id')
    with pytest.raises(StrataError, match='outside the emitted circuit'):
        exhaustive(km.Circuit('HMR q0 b0'), contract, lambda inp, scale: ({'mask': inp['mask']}, 0))


def test_quantum_only_queries_explicitly_decline_classical_contracts():
    contract = Contract('classical', (classical('mask', (0,)),), 'id')
    design = SimpleNamespace(validate=lambda: None, tree=SimpleNamespace(contract=contract))
    with pytest.raises(StrataError, match='quantum-only'):
        Queries(SimpleNamespace())._circuit(design)


def test_project_edit_cannot_swap_a_public_register_namespace():
    old = Region('root', 0, 1, Contract('old', (Register('x', (0,)),), 'id'))
    new = replace(old, contract=replace(old.contract, registers=(classical('x', (0,)),)))
    with pytest.raises(StrataError, match='external register contract changed'):
        Line.validate_edit(SimpleNamespace(), None, old, None, new)


def test_47_repaired_measurements_are_affine_after_neg_fix():
    text = '\n'.join(['X q0\nHMR q0 b0\nNEG if b0'] * 47)
    contract = Contract('repaired', (Register('scratch', (0,), role='clean'),), 'id')
    result = exhaustive(km.Circuit(text), contract, lambda inp, scale: ({}, 0), max_trajectories=48)
    assert result.passed and result.reduction == 'affine_hmr'
    assert result.branches == 1 << 47 and result.tested == result.executed_branches == 48


def nonlinear_identity(count):
    measurements = '\n'.join(f'HMR q0 b{i}' for i in range(count))
    return measurements + '\nPUSH_CONDITION if b0\nNEG if b1\nNEG if b1\nPOP_CONDITION'


def test_true_nonlinear_47_outcome_lab_is_symbolic_when_enumeration_exceeds_budget():
    contract = Contract('nonlinear', (Register('scratch', (0,), role='clean'),), 'id')
    reference = lambda inp, scale: ({}, 0)
    small = exhaustive(km.Circuit(nonlinear_identity(2)), contract, reference)
    assert small.passed and small.reduction == 'none' and small.tested == 4
    text = nonlinear_identity(47)
    assert compile_program(text).affine_hmr is False
    result = exhaustive(km.Circuit(text), contract, reference)
    assert result.passed and result.reduction == 'symbolic_hmr'
    assert result.branches == 1 << 47 and result.tested == 1
    validate_lab_receipt(asdict(result))
    with pytest.raises(StrataError, match='symbolic HMR.*budget.*no certificate'):
        exhaustive(km.Circuit(text), contract, reference, max_symbolic_nodes=2)


def test_multibit_classical_register_obeys_little_endian_order():
    circuit = km.Circuit('BIT_INVERT b0 if b1')
    contract = Contract('classical-update', (classical('word', (0, 1)),), 'controlled-not')
    result = exhaustive(circuit, contract,
                        lambda inp, scale: ({'word': inp['word'] ^ ((inp['word'] >> 1) & 1)}, 0))
    assert result.passed and result.cases == 4


def test_component_lab_inherits_classical_boundary_enumeration():
    from strata.bundle import Bundle, Component, BundleFrontend
    from strata.component_lab import ComponentLab, run_component_labs, validate_component_lab
    contract = Contract('controlled', (Register('x', (0,)), classical('mask', (0,))), 'controlled')
    frontend = BundleFrontend((Component('model', 'model.kmx', 'kickmix'),))
    artifact = Bundle({'model.kmx': b'X q0 if b0'})
    tree = Region('root', 0, 1, contract)
    class Adapter:
        def lab_components(self, parameters, artifact):
            return {'model': ComponentLab(contract,
                lambda inp, scale: ({'x': inp['x'] ^ inp['mask'], 'mask': inp['mask']}, 0))}
    result = run_component_labs(frontend, artifact, tree, {}, Adapter())
    assert result.passed and result.components['model']['lab']['cases'] == 4
    validate_component_lab(asdict(result), frontend)
